"""День 17. MCP-агент: Claude + MCP-инструменты трекера задач.

Агент:
  1. Подключается к MCP-серверу (day17/server.py) через stdio
  2. Получает список инструментов и конвертирует в формат Anthropic tool_use
  3. Claude сам решает, какой инструмент вызвать
  4. Агент исполняет вызов через MCP и возвращает результат Claude

Запуск:
  python agent.py                 — демо-режим (3 сценария)
  python agent.py --interactive   — интерактивный чат
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from contextlib import AsyncExitStack
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# ══════════════════════════════════════════════════════════════════
# MCP-агент с поддержкой tool_use
# ══════════════════════════════════════════════════════════════════

class MCPAgent:
    """Агент, использующий Claude + MCP-инструменты."""

    def __init__(self, anthropic_client: Anthropic, model: str) -> None:
        self._llm = anthropic_client
        self._model = model
        self._session: ClientSession | None = None
        self._exit_stack = AsyncExitStack()
        self._tools: list[dict] = []  # инструменты в формате Anthropic

    async def connect(self, server_script: str) -> int:
        """Подключиться к MCP-серверу, получить и конвертировать инструменты."""
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[server_script],
            env={**os.environ, "MCP_QUIET": "1"},
        )
        transport = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read_stream, write_stream = transport
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await self._session.initialize()

        # Получаем инструменты и конвертируем в формат Anthropic
        response = await self._session.list_tools()
        self._tools = [
            {
                "name": t.name,
                "description": t.description or "",
                "input_schema": t.inputSchema,
            }
            for t in response.tools
        ]
        return len(self._tools)

    async def ask(self, user_message: str, *, max_rounds: int = 5) -> str:
        """Отправить вопрос агенту. Claude может вызывать инструменты в цикле."""
        messages = [{"role": "user", "content": user_message}]

        system = (
            "Ты — AI-ассистент, управляющий трекером задач. "
            "У тебя есть доступ к инструментам для работы с проектами и задачами. "
            "Используй инструменты чтобы получить актуальные данные перед ответом. "
            "Отвечай на русском, кратко и по делу."
        )

        for round_num in range(max_rounds):
            response = self._llm.messages.create(
                model=self._model,
                max_tokens=1024,
                system=system,
                tools=self._tools,
                messages=messages,
            )

            # Если модель завершила ответ текстом — возвращаем
            if response.stop_reason == "end_turn":
                text_parts = [
                    b.text for b in response.content if b.type == "text"
                ]
                return "\n".join(text_parts)

            # Если модель хочет вызвать инструмент(ы) — исполняем
            if response.stop_reason == "tool_use":
                # Добавляем ответ ассистента (с tool_use блоками)
                messages.append({"role": "assistant", "content": response.content})

                # Исполняем каждый tool_use блок
                tool_results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    result = await self._call_mcp_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

                messages.append({"role": "user", "content": tool_results})
            else:
                # Неожиданный stop_reason
                text_parts = [
                    b.text for b in response.content if b.type == "text"
                ]
                return "\n".join(text_parts) if text_parts else "(пустой ответ)"

        return "(достигнут лимит раундов tool_use)"

    async def _call_mcp_tool(self, name: str, arguments: dict) -> str:
        """Вызвать инструмент на MCP-сервере."""
        result = await self._session.call_tool(name, arguments)
        if result.content:
            return result.content[0].text
        return "(пустой ответ)"

    async def close(self) -> None:
        await self._exit_stack.aclose()


# ══════════════════════════════════════════════════════════════════
# Демо: 3 сценария
# ══════════════════════════════════════════════════════════════════

async def run_demo(server_script: str) -> None:
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise SystemExit("Missing API key: set ANTHROPIC_API_KEY or API_KEY in .env")

    client = Anthropic(api_key=api_key)
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    agent = MCPAgent(client, model)

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  День 17. MCP-агент — Claude + трекер задач                 ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # ── Подключение ────────────────────────────────────────────────
    print(f"  Подключение к серверу...", end=" ", flush=True)
    tool_count = await agent.connect(server_script)
    print(f"✅ ({tool_count} инструментов)")
    print()

    # ── Показать инструменты ───────────────────────────────────────
    print(f"{'━' * 62}")
    print(f"  📦 ИНСТРУМЕНТЫ АГЕНТА")
    print(f"{'━' * 62}")
    for i, tool in enumerate(agent._tools, 1):
        props = tool["input_schema"].get("properties", {})
        params = ", ".join(props.keys()) if props else "нет"
        print(f"  {i}. {tool['name']}({params})")
        print(f"     {tool['description'][:70]}")
    print()

    # ── Сценарии ───────────────────────────────────────────────────
    scenarios = [
        "Покажи все проекты и сколько в каждом задач",
        "Какие задачи сейчас в работе в проекте 'Мобильное приложение'?",
        "Создай задачу 'Написать unit-тесты' с высоким приоритетом в проекте API Gateway и назначь на Дмитрия",
    ]

    for idx, question in enumerate(scenarios, 1):
        print(f"{'━' * 62}")
        print(f"  🗣️  СЦЕНАРИЙ {idx}")
        print(f"{'━' * 62}")
        print(f"  Вопрос: {question}")
        print()

        t0 = time.perf_counter()
        answer = await agent.ask(question)
        elapsed = time.perf_counter() - t0

        print(f"  Ответ агента:")
        for line in answer.splitlines():
            print(f"    {line}")
        print(f"\n  ⏱️  {elapsed:.1f}s")
        print()

    await agent.close()
    print("  Соединение закрыто.")


# ══════════════════════════════════════════════════════════════════
# Интерактивный режим
# ══════════════════════════════════════════════════════════════════

async def run_interactive(server_script: str) -> None:
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise SystemExit("Missing API key: set ANTHROPIC_API_KEY or API_KEY in .env")

    client = Anthropic(api_key=api_key)
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    agent = MCPAgent(client, model)

    print(f"  Подключение к MCP-серверу...", end=" ", flush=True)
    tool_count = await agent.connect(server_script)
    print(f"✅ ({tool_count} инструментов)")
    print(f"  Модель: {model}")
    print(f"  Введите вопрос (пустая строка — выход)\n")

    while True:
        try:
            user_text = input("Вы> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            break

        t0 = time.perf_counter()
        answer = await agent.ask(user_text)
        elapsed = time.perf_counter() - t0

        print(f"\nАгент> {answer}")
        print(f"  [{elapsed:.1f}s]\n")

    await agent.close()
    print("  Соединение закрыто.")


# ══════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    server_script = str(Path(__file__).parent / "server.py")

    if "--interactive" in sys.argv:
        asyncio.run(run_interactive(server_script))
    else:
        asyncio.run(run_demo(server_script))


if __name__ == "__main__":
    main()

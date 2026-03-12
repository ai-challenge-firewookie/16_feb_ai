"""День 18. MCP-агент: Claude + планировщик фоновых задач.

Агент:
  1. Подключается к MCP-серверу (day18/server.py) через stdio
  2. Claude управляет планировщиком: создаёт задачи, смотрит статус, запускает
  3. Может запустить фоновый scheduler в отдельном процессе

Запуск:
  python agent.py                 — демо-режим (5 сценариев)
  python agent.py --interactive   — интерактивный чат
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import subprocess
import time
from contextlib import AsyncExitStack
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# ══════════════════════════════════════════════════════════════
# MCP-агент с поддержкой tool_use
# ══════════════════════════════════════════════════════════════

class SchedulerAgent:
    """Агент, использующий Claude + MCP-инструменты планировщика."""

    def __init__(self, anthropic_client: Anthropic, model: str) -> None:
        self._llm = anthropic_client
        self._model = model
        self._session: ClientSession | None = None
        self._exit_stack = AsyncExitStack()
        self._tools: list[dict] = []

    async def connect(self, server_script: str) -> int:
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

    async def ask(self, user_message: str, *, max_rounds: int = 8) -> str:
        messages = [{"role": "user", "content": user_message}]

        system = (
            "Ты — AI-ассистент, управляющий планировщиком фоновых задач. "
            "У тебя есть инструменты для создания, просмотра, удаления задач, "
            "получения истории и сводки, а также для принудительного запуска задач. "
            "Используй инструменты чтобы выполнить запрос пользователя. "
            "Отвечай на русском, кратко и по делу. "
            "Интервалы задавай в формате: 30s, 5m, 2h, 1d."
        )

        for round_num in range(max_rounds):
            response = self._llm.messages.create(
                model=self._model,
                max_tokens=1024,
                system=system,
                tools=self._tools,
                messages=messages,
            )

            if response.stop_reason == "end_turn":
                text_parts = [b.text for b in response.content if b.type == "text"]
                return "\n".join(text_parts)

            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})
                tool_results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    print(f"    -> {block.name}({json.dumps(block.input, ensure_ascii=False)[:80]})")
                    result = await self._call_mcp_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
                messages.append({"role": "user", "content": tool_results})
            else:
                text_parts = [b.text for b in response.content if b.type == "text"]
                return "\n".join(text_parts) if text_parts else "(пустой ответ)"

        return "(достигнут лимит раундов tool_use)"

    async def _call_mcp_tool(self, name: str, arguments: dict) -> str:
        result = await self._session.call_tool(name, arguments)
        if result.content:
            return result.content[0].text
        return "(пустой ответ)"

    async def close(self) -> None:
        await self._exit_stack.aclose()


# ══════════════════════════════════════════════════════════════
# Демо: 5 сценариев
# ══════════════════════════════════════════════════════════════

async def run_demo(server_script: str) -> None:
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise SystemExit("Missing API key: set ANTHROPIC_API_KEY or API_KEY in .env")

    client = Anthropic(api_key=api_key)
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    agent = SchedulerAgent(client, model)

    print("=" * 62)
    print("  День 18. MCP-агент — планировщик фоновых задач")
    print("=" * 62)
    print()

    print(f"  Подключение к серверу...", end=" ", flush=True)
    tool_count = await agent.connect(server_script)
    print(f"OK ({tool_count} инструментов)")
    print()

    # Показать инструменты
    print(f"{'─' * 62}")
    print(f"  ИНСТРУМЕНТЫ АГЕНТА")
    print(f"{'─' * 62}")
    for i, tool in enumerate(agent._tools, 1):
        props = tool["input_schema"].get("properties", {})
        params = ", ".join(props.keys()) if props else "нет"
        print(f"  {i}. {tool['name']}({params})")
        desc = tool['description'].split('\n')[0][:70]
        print(f"     {desc}")
    print()

    # Сценарии
    scenarios = [
        "Создай периодическую задачу 'Мониторинг серверов' с проверкой здоровья сервисов каждые 30 секунд",
        "Создай задачу для сбора данных о пользователях каждые 2 минуты и разовое напоминание 'Обновить SSL-сертификат' через 5 минут",
        "Покажи все активные задачи планировщика",
        "Запусти задачу мониторинга прямо сейчас и покажи результат",
        "Покажи общую сводку по планировщику: статистика, ближайшие задачи, последние выполнения",
    ]

    for idx, question in enumerate(scenarios, 1):
        print(f"{'─' * 62}")
        print(f"  СЦЕНАРИЙ {idx}")
        print(f"{'─' * 62}")
        print(f"  Запрос: {question}")
        print()

        t0 = time.perf_counter()
        answer = await agent.ask(question)
        elapsed = time.perf_counter() - t0

        print(f"\n  Ответ агента:")
        for line in answer.splitlines():
            print(f"    {line}")
        print(f"\n  [{elapsed:.1f}s]")
        print()

    await agent.close()
    print("  Соединение закрыто.")


# ══════════════════════════════════════════════════════════════
# Интерактивный режим
# ══════════════════════════════════════════════════════════════

async def run_interactive(server_script: str) -> None:
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise SystemExit("Missing API key: set ANTHROPIC_API_KEY or API_KEY in .env")

    client = Anthropic(api_key=api_key)
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    agent = SchedulerAgent(client, model)

    print(f"  Подключение к MCP-серверу...", end=" ", flush=True)
    tool_count = await agent.connect(server_script)
    print(f"OK ({tool_count} инструментов)")
    print(f"  Модель: {model}")
    print()

    # Предлагаем запустить фоновый scheduler
    print("  Совет: запустите фоновый планировщик в отдельном терминале:")
    print(f"    python {Path(__file__).parent / 'scheduler.py'}")
    print()
    print(f"  Введите запрос (пустая строка — выход)")
    print()

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


# ══════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════

def main() -> None:
    server_script = str(Path(__file__).parent / "server.py")

    if "--interactive" in sys.argv:
        asyncio.run(run_interactive(server_script))
    else:
        asyncio.run(run_demo(server_script))


if __name__ == "__main__":
    main()

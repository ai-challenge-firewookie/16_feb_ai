"""День 19. MCP-агент: композиция инструментов (pipeline).

Агент подключается к MCP-серверу с инструментами пайплайна.
Claude автоматически выстраивает цепочку вызовов:
  search → fetch → summarize/translate/analyze → save

Два режима:
  1. Демо — Claude сам выстраивает пайплайны по запросам
  2. Интерактивный — пользователь даёт произвольные задачи

Запуск:
  python agent.py                 — демо-режим
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


# ══════════════════════════════════════════════════════════════
# MCP-агент
# ══════════════════════════════════════════════════════════════

class PipelineAgent:
    """Агент, выстраивающий цепочки вызовов MCP-инструментов."""

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

    async def ask(self, user_message: str, *, max_rounds: int = 10) -> str:
        messages = [{"role": "user", "content": user_message}]

        system = (
            "Ты — AI-ассистент, работающий с базой знаний через пайплайн инструментов. "
            "У тебя есть инструменты для поиска статей, получения контента, "
            "суммаризации, перевода, анализа тональности и сохранения в файл. "
            "\n\n"
            "ВАЖНО: Когда пользователь просит найти и обработать информацию, "
            "выстраивай цепочку инструментов последовательно:\n"
            "1. search_articles — найди релевантные статьи\n"
            "2. fetch_content — получи полный текст\n"
            "3. summarize_text / translate_text / analyze_sentiment — обработай\n"
            "4. save_to_file — сохрани результат\n"
            "\n"
            "Вызывай инструменты по одному, передавая данные из предыдущего шага в следующий. "
            "Это демонстрирует КОМПОЗИЦИЮ инструментов — пайплайн. "
            "Также доступен run_pipeline для выполнения готового пайплайна за один вызов.\n"
            "\n"
            "Отвечай на русском, кратко. После выполнения цепочки покажи пользователю "
            "какие шаги были выполнены и итоговый результат."
        )

        for round_num in range(max_rounds):
            response = self._llm.messages.create(
                model=self._model,
                max_tokens=2048,
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
                    # Показываем шаг пайплайна
                    args_preview = json.dumps(block.input, ensure_ascii=False)
                    if len(args_preview) > 100:
                        args_preview = args_preview[:97] + "..."
                    print(f"    [{round_num+1}] {block.name}({args_preview})")
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
# Демо
# ══════════════════════════════════════════════════════════════

async def run_demo(server_script: str) -> None:
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise SystemExit("Missing API key: set ANTHROPIC_API_KEY or API_KEY in .env")

    client = Anthropic(api_key=api_key)
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    agent = PipelineAgent(client, model)

    print("=" * 62)
    print("  День 19. Композиция MCP-инструментов (pipeline)")
    print("=" * 62)
    print()

    print(f"  Подключение к серверу...", end=" ", flush=True)
    tool_count = await agent.connect(server_script)
    print(f"OK ({tool_count} инструментов)")
    print()

    # Инструменты
    print(f"{'─' * 62}")
    print(f"  ИНСТРУМЕНТЫ (пайплайн)")
    print(f"{'─' * 62}")
    for i, tool in enumerate(agent._tools, 1):
        desc = tool['description'].split('\n')[0][:65]
        print(f"  {i}. {tool['name']}")
        print(f"     {desc}")
    print()

    # Сценарии: Claude сам строит цепочки
    scenarios = [
        # 1. Цепочка: search → fetch → summarize → save
        (
            "Найди статью про машинное обучение, сделай краткую выжимку из неё "
            "и сохрани результат в файл ml_summary.md"
        ),
        # 2. Цепочка: search → fetch → analyze → translate → save
        (
            "Найди статью про безопасность веб-приложений, "
            "проанализируй её тональность, переведи ключевые выводы на английский "
            "и сохрани всё в файл security_analysis.json"
        ),
        # 3. Мета-пайплайн: run_pipeline одним вызовом
        (
            "Запусти полный пайплайн search-full-save по запросу 'LLM AI' "
            "и сохрани как llm_full_report"
        ),
        # 4. Чтение ранее сохранённых файлов
        "Покажи список всех сохранённых файлов и выведи содержимое первого из них",
    ]

    for idx, question in enumerate(scenarios, 1):
        print(f"{'─' * 62}")
        print(f"  СЦЕНАРИЙ {idx}")
        print(f"{'─' * 62}")
        print(f"  Запрос: {question[:80]}{'...' if len(question) > 80 else ''}")
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

    agent = PipelineAgent(client, model)

    print(f"  Подключение к MCP-серверу...", end=" ", flush=True)
    tool_count = await agent.connect(server_script)
    print(f"OK ({tool_count} инструментов)")
    print(f"  Модель: {model}")
    print()
    print("  Примеры запросов:")
    print("    - Найди статью про asyncio и сделай summary")
    print("    - Запусти пайплайн search-full-save по теме 'микросервисы'")
    print("    - Покажи сохранённые файлы")
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

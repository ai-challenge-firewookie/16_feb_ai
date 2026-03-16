"""День 20. Оркестратор: Claude + 3 MCP-сервера одновременно.

OrchestratorAgent подключается к нескольким MCP-серверам,
объединяет все инструменты в единый плоский список и маршрутизирует
вызовы к правильному серверу по имени инструмента.

Серверы:
  - tracker   (day17/server.py)  — проекты и задачи
  - scheduler (day18/server.py)  — планировщик фоновых задач
  - notes     (day20/server_notes.py) — заметки

Запуск:
  python agent.py                 — демо (5 сценариев)
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
# OrchestratorAgent — мульти-серверный агент
# ══════════════════════════════════════════════════════════════

class OrchestratorAgent:
    """Агент, подключающийся к нескольким MCP-серверам и маршрутизирующий вызовы."""

    def __init__(self, anthropic_client: Anthropic, model: str) -> None:
        self._llm = anthropic_client
        self._model = model
        self._exit_stack = AsyncExitStack()
        self._tools: list[dict] = []  # все инструменты (плоский список для Anthropic API)
        self._tool_to_session: dict[str, ClientSession] = {}  # tool_name -> session
        self._tool_to_server: dict[str, str] = {}  # tool_name -> server_name
        self._sessions: dict[str, ClientSession] = {}  # server_name -> session

    async def connect_server(self, name: str, script_path: str) -> int:
        """Подключиться к одному MCP-серверу и зарегистрировать его инструменты."""
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[script_path],
            env={**os.environ, "MCP_QUIET": "1"},
        )
        transport = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read_stream, write_stream = transport
        session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()

        self._sessions[name] = session

        # Получаем инструменты и регистрируем маршрутизацию
        response = await session.list_tools()
        count = 0
        for t in response.tools:
            self._tools.append({
                "name": t.name,
                "description": t.description or "",
                "input_schema": t.inputSchema,
            })
            self._tool_to_session[t.name] = session
            self._tool_to_server[t.name] = name
            count += 1

        return count

    async def connect_all(self, servers: dict[str, str]) -> dict[str, int]:
        """Подключиться ко всем серверам. Возвращает {name: tool_count}."""
        result = {}
        for name, script_path in servers.items():
            count = await self.connect_server(name, script_path)
            result[name] = count
        return result

    async def ask(self, user_message: str, *, max_rounds: int = 12) -> str:
        """Отправить вопрос агенту. Claude маршрутизирует вызовы к нужным серверам."""
        messages = [{"role": "user", "content": user_message}]

        # Формируем описание доступных серверов для системного промпта
        server_tools: dict[str, list[str]] = {}
        for tool_name, server_name in self._tool_to_server.items():
            server_tools.setdefault(server_name, []).append(tool_name)

        servers_desc = ""
        for srv, tools in server_tools.items():
            servers_desc += f"\n  - {srv}: {', '.join(tools)}"

        system = (
            "Ты — AI-оркестратор, управляющий несколькими системами одновременно. "
            "У тебя есть доступ к инструментам из нескольких серверов:"
            f"{servers_desc}\n\n"
            "Используй инструменты чтобы выполнить запрос пользователя. "
            "Ты можешь комбинировать данные из разных серверов в одном ответе. "
            "Отвечай на русском, кратко и по делу."
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
                    server_name = self._tool_to_server.get(block.name, "?")
                    args_preview = json.dumps(block.input, ensure_ascii=False)
                    if len(args_preview) > 80:
                        args_preview = args_preview[:77] + "..."
                    print(f"    [{server_name}] {block.name}({args_preview})")
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
        """Маршрутизирует вызов к нужной MCP-сессии."""
        session = self._tool_to_session.get(name)
        if not session:
            return json.dumps({"error": f"Инструмент '{name}' не найден"})
        result = await session.call_tool(name, arguments)
        if result.content:
            return result.content[0].text
        return "(пустой ответ)"

    async def close(self) -> None:
        await self._exit_stack.aclose()


# ══════════════════════════════════════════════════════════════
# Демо: 5 сценариев
# ══════════════════════════════════════════════════════════════

async def run_demo(server_scripts: dict[str, str] | None = None) -> None:
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise SystemExit("Missing API key: set ANTHROPIC_API_KEY or API_KEY in .env")

    client = Anthropic(api_key=api_key)
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    if server_scripts is None:
        base = Path(__file__).parent.parent
        server_scripts = {
            "tracker": str(base / "day17" / "server.py"),
            "scheduler": str(base / "day18" / "server.py"),
            "notes": str(base / "day20" / "server_notes.py"),
        }

    agent = OrchestratorAgent(client, model)

    print("=" * 62)
    print("  День 20. Оркестратор — Claude + 3 MCP-сервера")
    print("=" * 62)
    print()

    # ── Подключение ────────────────────────────────────────────
    print("  Подключение к серверам:")
    counts = await agent.connect_all(server_scripts)
    for name, count in counts.items():
        print(f"    {name}: {count} инструментов")
    total = sum(counts.values())
    print(f"  Всего: {total} инструментов")
    print()

    # ── Таблица маршрутизации ──────────────────────────────────
    print(f"{'─' * 62}")
    print(f"  ТАБЛИЦА МАРШРУТИЗАЦИИ")
    print(f"{'─' * 62}")
    for tool_name, server_name in sorted(agent._tool_to_server.items()):
        print(f"    {tool_name:25s} -> [{server_name}]")
    print()

    # ── Сценарии ───────────────────────────────────────────────
    scenarios = [
        # 1. Один сервер (tracker)
        "Покажи все проекты и количество задач в каждом",
        # 2. Один сервер (scheduler)
        "Создай периодическую задачу 'Бэкап БД' с действием 'Полный бэкап PostgreSQL' каждые 2 часа",
        # 3. Кросс-сервер (tracker + scheduler)
        "Найди все задачи со статусом todo в проекте API Gateway и создай напоминание в планировщике для каждой из них",
        # 4. Кросс-сервер (tracker + notes)
        "Посмотри задачи в работе (in_progress) во всех проектах и создай заметку-сводку с тегами 'status,review'",
        # 5. Три сервера
        "Сделай полный аудит: покажи проекты из трекера, сводку планировщика и все заметки. Затем создай заметку 'Аудит-отчёт' с кратким итогом по всем трём системам, с тегами 'audit,report'",
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
    print("  Все соединения закрыты.")


# ══════════════════════════════════════════════════════════════
# Интерактивный режим
# ══════════════════════════════════════════════════════════════

async def run_interactive(server_scripts: dict[str, str] | None = None) -> None:
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise SystemExit("Missing API key: set ANTHROPIC_API_KEY or API_KEY in .env")

    client = Anthropic(api_key=api_key)
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    if server_scripts is None:
        base = Path(__file__).parent.parent
        server_scripts = {
            "tracker": str(base / "day17" / "server.py"),
            "scheduler": str(base / "day18" / "server.py"),
            "notes": str(base / "day20" / "server_notes.py"),
        }

    agent = OrchestratorAgent(client, model)

    print("  Подключение к серверам...", flush=True)
    counts = await agent.connect_all(server_scripts)
    for name, count in counts.items():
        print(f"    {name}: {count} инструментов")
    total = sum(counts.values())
    print(f"  Всего: {total} инструментов | Модель: {model}")
    print()
    print("  Примеры запросов:")
    print("    - Покажи все проекты")
    print("    - Создай заметку с итогами текущих задач")
    print("    - Найди todo задачи и создай напоминания")
    print()
    print("  Введите запрос (пустая строка — выход)")
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
    print("  Все соединения закрыты.")


# ══════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════

def main() -> None:
    if "--interactive" in sys.argv:
        asyncio.run(run_interactive())
    else:
        asyncio.run(run_demo())


if __name__ == "__main__":
    main()

"""День 16. MCP-клиент: подключение и получение списка инструментов.

Клиент:
  1. Устанавливает MCP-соединение с сервером (stdio-транспорт)
  2. Получает список доступных инструментов
  3. Вызывает каждый инструмент и показывает результат
  4. Показывает полную информацию (name, description, inputSchema)

Запуск:
  python client.py                 — подключение к server.py (по умолчанию)
  python client.py /path/to/srv.py — подключение к другому серверу
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from contextlib import AsyncExitStack
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# ══════════════════════════════════════════════════════════════════
# MCP-клиент
# ══════════════════════════════════════════════════════════════════

class MCPClient:
    """Минимальный MCP-клиент для демонстрации."""

    def __init__(self) -> None:
        self.session: ClientSession | None = None
        self._exit_stack = AsyncExitStack()

    async def connect(self, server_script: str, python_cmd: str = sys.executable) -> None:
        """Установить соединение с MCP-сервером через stdio."""
        server_params = StdioServerParameters(
            command=python_cmd,
            args=[server_script],
            env={**os.environ, "MCP_QUIET": "1"},
        )

        transport = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read_stream, write_stream = transport

        self.session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )

        await self.session.initialize()

    async def list_tools(self) -> list:
        """Получить список доступных инструментов."""
        if not self.session:
            raise RuntimeError("Не подключен к серверу")
        response = await self.session.list_tools()
        return response.tools

    async def call_tool(self, name: str, arguments: dict) -> str:
        """Вызвать инструмент по имени."""
        if not self.session:
            raise RuntimeError("Не подключен к серверу")
        result = await self.session.call_tool(name, arguments)
        # result.content — список блоков, берём текст из первого
        if result.content:
            return result.content[0].text
        return "(пустой ответ)"

    async def close(self) -> None:
        """Закрыть соединение."""
        await self._exit_stack.aclose()


# ══════════════════════════════════════════════════════════════════
# Демо: подключение → список инструментов → вызов каждого
# ══════════════════════════════════════════════════════════════════

async def run_demo(server_script: str) -> None:
    client = MCPClient()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  День 16. MCP-клиент — подключение и инструменты            ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # ── 1. Подключение ───────────────────────────────────────────
    print(f"  Сервер: {server_script}")
    print(f"  Python: {sys.executable}")
    print(f"  Подключение...", end=" ", flush=True)

    try:
        await client.connect(server_script)
        print("✅ Соединение установлено!")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return

    # ── 2. Список инструментов ────────────────────────────────────
    print()
    print(f"{'━' * 62}")
    print(f"  📦 СПИСОК ДОСТУПНЫХ ИНСТРУМЕНТОВ")
    print(f"{'━' * 62}")

    try:
        tools = await client.list_tools()
        print(f"\n  Найдено инструментов: {len(tools)}\n")

        for i, tool in enumerate(tools, 1):
            print(f"  ┌─ [{i}] {tool.name}")
            print(f"  │  Описание: {tool.description}")
            if tool.inputSchema:
                props = tool.inputSchema.get("properties", {})
                required = tool.inputSchema.get("required", [])
                if props:
                    params_str = ", ".join(
                        f"{k}: {v.get('type', '?')}" + (" (обяз.)" if k in required else "")
                        for k, v in props.items()
                    )
                    print(f"  │  Параметры: {params_str}")
                else:
                    print(f"  │  Параметры: нет")
            print(f"  └{'─' * 50}")
    except Exception as e:
        print(f"  ❌ Ошибка при получении списка: {e}")
        await client.close()
        return

    # ── 3. Вызов каждого инструмента ──────────────────────────────
    print()
    print(f"{'━' * 62}")
    print(f"  🔧 ВЫЗОВ ИНСТРУМЕНТОВ")
    print(f"{'━' * 62}")

    test_calls = [
        ("calc_add",       {"a": 42, "b": 58}),
        ("calc_sub",       {"a": 100, "b": 37}),
        ("calc_mul",       {"a": 7, "b": 8}),
        ("calc_div",       {"a": 355, "b": 113}),
        ("calc_div",       {"a": 1, "b": 0}),
        ("get_time",       {}),
        ("reverse_string", {"text": "Привет, MCP!"}),
        ("word_count",     {"text": "Model Context Protocol — это круто"}),
    ]

    passed = 0
    for name, args in test_calls:
        args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
        try:
            result = await client.call_tool(name, args)
            print(f"  ✅ {name}({args_str})")
            print(f"     → {result}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {name}({args_str})")
            print(f"     → Ошибка: {e}")

    # ── 4. Итоги ──────────────────────────────────────────────────
    print()
    print(f"{'━' * 62}")
    print(f"  ИТОГИ")
    print(f"{'━' * 62}")
    print(f"  Соединение:  ✅ установлено")
    print(f"  Инструменты: {len(tools)} найдено")
    print(f"  Вызовы:      {passed}/{len(test_calls)} успешно")

    await client.close()
    print(f"\n  Соединение закрыто.")


# ══════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    # По умолчанию — server.py в той же директории
    if len(sys.argv) > 1:
        server_script = sys.argv[1]
    else:
        server_script = str(Path(__file__).parent / "server.py")

    if not Path(server_script).exists():
        print(f"Ошибка: файл сервера не найден: {server_script}")
        sys.exit(1)

    asyncio.run(run_demo(server_script))


if __name__ == "__main__":
    main()

"""День 16. MCP-сервер с набором инструментов.

MCP-сервер, предоставляющий инструменты:
  - calc_add, calc_sub, calc_mul, calc_div — калькулятор
  - get_time         — текущее время
  - reverse_string   — разворот строки
  - word_count       — подсчёт слов в тексте

Запуск:
  python server.py            — stdio-транспорт (для MCP-клиентов)
  python server.py --test     — быстрый self-test (без клиента)
"""

from __future__ import annotations

import logging
import os
import sys
import time

# Подавляем логи FastMCP при запуске как MCP-подпроцесс
if os.environ.get("MCP_QUIET"):
    logging.getLogger("mcp").setLevel(logging.ERROR)

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("day16-tools")


# ══════════════════════════════════════════════════════════════════
# Инструменты: калькулятор
# ══════════════════════════════════════════════════════════════════

@mcp.tool()
def calc_add(a: float, b: float) -> float:
    """Сложение двух чисел. Возвращает a + b."""
    return a + b


@mcp.tool()
def calc_sub(a: float, b: float) -> float:
    """Вычитание двух чисел. Возвращает a - b."""
    return a - b


@mcp.tool()
def calc_mul(a: float, b: float) -> float:
    """Умножение двух чисел. Возвращает a * b."""
    return a * b


@mcp.tool()
def calc_div(a: float, b: float) -> str:
    """Деление двух чисел. Возвращает a / b или ошибку при делении на 0."""
    if b == 0:
        return "Ошибка: деление на ноль"
    return str(a / b)


# ══════════════════════════════════════════════════════════════════
# Инструменты: утилиты
# ══════════════════════════════════════════════════════════════════

@mcp.tool()
def get_time() -> str:
    """Возвращает текущее время сервера в формате YYYY-MM-DD HH:MM:SS."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


@mcp.tool()
def reverse_string(text: str) -> str:
    """Разворачивает строку задом наперёд."""
    return text[::-1]


@mcp.tool()
def word_count(text: str) -> dict:
    """Подсчитывает количество слов, символов и строк в тексте."""
    words = text.split()
    lines = text.splitlines() or [text]
    return {
        "words": len(words),
        "chars": len(text),
        "lines": len(lines),
    }


# ══════════════════════════════════════════════════════════════════
# Self-test (без клиента — просто проверка что всё зарегистрировано)
# ══════════════════════════════════════════════════════════════════

def self_test() -> None:
    """Проверяет что все tools зарегистрированы."""
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  День 16. MCP-сервер — self-test                           ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # FastMCP хранит инструменты внутри — проверяем через прямой вызов
    tools = [
        ("calc_add", calc_add, {"a": 2, "b": 3}, 5.0),
        ("calc_sub", calc_sub, {"a": 10, "b": 4}, 6.0),
        ("calc_mul", calc_mul, {"a": 3, "b": 7}, 21.0),
        ("calc_div", calc_div, {"a": 10, "b": 3}, str(10 / 3)),
        ("calc_div", calc_div, {"a": 1, "b": 0}, "Ошибка: деление на ноль"),
        ("reverse_string", reverse_string, {"text": "hello"}, "olleh"),
    ]

    passed = 0
    for name, func, args, expected in tools:
        result = func(**args)
        ok = result == expected
        status = "✅" if ok else "❌"
        print(f"  {status} {name}({args}) = {result}")
        if ok:
            passed += 1

    # get_time — просто проверяем что возвращает строку
    t = get_time()
    ok = isinstance(t, str) and len(t) == 19
    print(f"  {'✅' if ok else '❌'} get_time() = {t}")
    if ok:
        passed += 1

    # word_count
    wc = word_count("hello world test")
    ok = wc == {"words": 3, "chars": 16, "lines": 1}
    print(f"  {'✅' if ok else '❌'} word_count('hello world test') = {wc}")
    if ok:
        passed += 1

    total = len(tools) + 2
    print(f"\n  Результат: {passed}/{total} тестов пройдено")


# ══════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if "--test" in sys.argv:
        self_test()
    else:
        mcp.run(transport="stdio")

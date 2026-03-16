"""День 20. MCP-сервер: заметки (in-memory).

Инструменты:
  - add_note        — создать заметку
  - list_notes      — список заметок (с фильтрацией по тегу)
  - get_note        — получить заметку по ID
  - search_notes    — полнотекстовый поиск по заголовкам/содержимому
  - delete_note     — удалить заметку

Данные хранятся в памяти (mock).

Запуск:
  python server_notes.py         — stdio-транспорт (для MCP-клиентов)
  python server_notes.py --test  — self-test
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime

if os.environ.get("MCP_QUIET"):
    logging.getLogger("mcp").setLevel(logging.ERROR)

import json as _json

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("day20-notes")


def _j(obj: object) -> str:
    """Сериализация результата в JSON-строку."""
    return _json.dumps(obj, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════════
# In-memory хранилище
# ══════════════════════════════════════════════════════════════

_note_counter = 3

NOTES: dict[str, dict] = {
    "note-1": {
        "id": "note-1",
        "title": "Повестка совещания",
        "content": "1. Обзор спринта\n2. Блокеры по задачам\n3. Планирование следующей итерации\n4. Вопросы от команды",
        "tags": ["meeting", "sprint"],
        "created_at": "2025-03-10T10:00:00",
    },
    "note-2": {
        "id": "note-2",
        "title": "Расписание бэкапов",
        "content": "Ежедневно в 03:00 — полный бэкап БД\nЕженедельно в воскресенье — бэкап файлов\nЕжемесячно 1-го числа — архивация логов",
        "tags": ["ops", "backup"],
        "created_at": "2025-03-12T14:30:00",
    },
    "note-3": {
        "id": "note-3",
        "title": "Правила код-ревью",
        "content": "1. PR не больше 400 строк\n2. Минимум 2 апрува\n3. CI должен быть зелёным\n4. Описание изменений обязательно\n5. Не мержить в пятницу после 16:00",
        "tags": ["dev", "process"],
        "created_at": "2025-03-15T09:00:00",
    },
}


# ══════════════════════════════════════════════════════════════
# Инструменты
# ══════════════════════════════════════════════════════════════

@mcp.tool()
def add_note(title: str, content: str, tags: str = "") -> str:
    """Создаёт новую заметку.

    Args:
        title: Заголовок заметки
        content: Текст заметки
        tags: Теги через запятую (например 'meeting,sprint')
    """
    global _note_counter
    _note_counter += 1
    note_id = f"note-{_note_counter}"
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    note = {
        "id": note_id,
        "title": title,
        "content": content,
        "tags": tag_list,
        "created_at": datetime.now().isoformat(),
    }
    NOTES[note_id] = note
    return _j(note)


@mcp.tool()
def list_notes(tag: str | None = None) -> str:
    """Возвращает список всех заметок, опционально фильтруя по тегу.

    Args:
        tag: Фильтр по тегу (например 'meeting'). Пусто — все заметки.
    """
    if tag:
        result = [n for n in NOTES.values() if tag in n["tags"]]
    else:
        result = list(NOTES.values())
    # Краткий формат: без полного content
    return _j([
        {
            "id": n["id"],
            "title": n["title"],
            "tags": n["tags"],
            "created_at": n["created_at"],
            "content_preview": n["content"][:80],
        }
        for n in result
    ])


@mcp.tool()
def get_note(note_id: str) -> str:
    """Возвращает полную заметку по ID.

    Args:
        note_id: ID заметки (например 'note-1')
    """
    note = NOTES.get(note_id)
    if not note:
        return _j({"error": f"Заметка '{note_id}' не найдена"})
    return _j(note)


@mcp.tool()
def search_notes(query: str) -> str:
    """Полнотекстовый поиск по заголовкам и содержимому заметок.

    Args:
        query: Поисковый запрос (регистронезависимый)
    """
    q = query.lower()
    results = [
        n for n in NOTES.values()
        if q in n["title"].lower() or q in n["content"].lower()
    ]
    return _j([
        {
            "id": n["id"],
            "title": n["title"],
            "tags": n["tags"],
            "match_preview": n["content"][:100],
        }
        for n in results
    ])


@mcp.tool()
def delete_note(note_id: str) -> str:
    """Удаляет заметку по ID.

    Args:
        note_id: ID заметки (например 'note-1')
    """
    if note_id not in NOTES:
        return _j({"error": f"Заметка '{note_id}' не найдена"})
    deleted = NOTES.pop(note_id)
    return _j({"deleted": note_id, "title": deleted["title"]})


# ══════════════════════════════════════════════════════════════
# Self-test
# ══════════════════════════════════════════════════════════════

def self_test() -> None:
    import json

    print("=" * 62)
    print("  День 20. MCP-сервер заметок — self-test")
    print("=" * 62)
    print()

    tests = 0
    passed = 0

    # list_notes — все
    tests += 1
    notes = json.loads(list_notes())
    ok = len(notes) == 3
    print(f"  {'OK' if ok else 'FAIL'} list_notes() -> {len(notes)} заметок")
    if ok: passed += 1

    # list_notes — фильтр по тегу
    tests += 1
    notes = json.loads(list_notes(tag="meeting"))
    ok = len(notes) == 1 and notes[0]["id"] == "note-1"
    print(f"  {'OK' if ok else 'FAIL'} list_notes(tag='meeting') -> {len(notes)} заметка")
    if ok: passed += 1

    # get_note
    tests += 1
    note = json.loads(get_note("note-2"))
    ok = "Расписание бэкапов" in note.get("title", "")
    print(f"  {'OK' if ok else 'FAIL'} get_note('note-2') -> {note.get('title', '?')}")
    if ok: passed += 1

    # search_notes
    tests += 1
    results = json.loads(search_notes("бэкап"))
    ok = len(results) == 1 and results[0]["id"] == "note-2"
    print(f"  {'OK' if ok else 'FAIL'} search_notes('бэкап') -> {len(results)} результат")
    if ok: passed += 1

    # search_notes — регистронезависимый
    tests += 1
    results = json.loads(search_notes("PR"))
    ok = len(results) == 1 and results[0]["id"] == "note-3"
    print(f"  {'OK' if ok else 'FAIL'} search_notes('PR') -> {len(results)} результат")
    if ok: passed += 1

    # add_note
    tests += 1
    new = json.loads(add_note("Тестовая заметка", "Содержимое теста", tags="test,demo"))
    ok = new["id"] == "note-4" and len(new["tags"]) == 2
    print(f"  {'OK' if ok else 'FAIL'} add_note() -> id={new.get('id', '?')}, tags={new.get('tags', [])}")
    if ok: passed += 1

    # Проверяем, что заметка появилась
    tests += 1
    notes = json.loads(list_notes())
    ok = len(notes) == 4
    print(f"  {'OK' if ok else 'FAIL'} list_notes() после add -> {len(notes)} заметок")
    if ok: passed += 1

    # delete_note
    tests += 1
    result = json.loads(delete_note("note-4"))
    ok = result.get("deleted") == "note-4"
    print(f"  {'OK' if ok else 'FAIL'} delete_note('note-4') -> deleted={result.get('deleted', '?')}")
    if ok: passed += 1

    # Проверяем удаление
    tests += 1
    notes = json.loads(list_notes())
    ok = len(notes) == 3
    print(f"  {'OK' if ok else 'FAIL'} list_notes() после delete -> {len(notes)} заметок")
    if ok: passed += 1

    # Ошибка: несуществующая заметка
    tests += 1
    err = json.loads(get_note("note-999"))
    ok = "error" in err
    print(f"  {'OK' if ok else 'FAIL'} get_note('note-999') -> ошибка: {err.get('error', '?')}")
    if ok: passed += 1

    print(f"\n  Результат: {passed}/{tests} тестов пройдено")


# ══════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if "--test" in sys.argv:
        self_test()
    else:
        mcp.run(transport="stdio")

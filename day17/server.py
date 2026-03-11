"""День 17. MCP-сервер: трекер задач (mock API).

Инструменты:
  - list_projects     — список всех проектов
  - get_project       — информация о проекте по ID
  - list_tasks        — задачи проекта (с фильтрацией по статусу)
  - create_task       — создать новую задачу в проекте
  - update_task_status — изменить статус задачи

Данные хранятся в памяти (mock).

Запуск:
  python server.py             — stdio-транспорт (для MCP-клиентов)
  python server.py --test      — self-test
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime

if os.environ.get("MCP_QUIET"):
    logging.getLogger("mcp").setLevel(logging.ERROR)

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("day17-tracker")

# ══════════════════════════════════════════════════════════════════
# Mock-данные: проекты и задачи
# ══════════════════════════════════════════════════════════════════

PROJECTS: dict[str, dict] = {
    "proj-1": {
        "id": "proj-1",
        "name": "Веб-сайт компании",
        "description": "Редизайн корпоративного сайта",
        "created_at": "2025-01-15",
    },
    "proj-2": {
        "id": "proj-2",
        "name": "Мобильное приложение",
        "description": "iOS/Android клиент для сервиса доставки",
        "created_at": "2025-03-01",
    },
    "proj-3": {
        "id": "proj-3",
        "name": "API Gateway",
        "description": "Единая точка входа для микросервисов",
        "created_at": "2025-02-10",
    },
}

_task_counter = 100

TASKS: dict[str, dict] = {
    "task-1": {
        "id": "task-1",
        "project_id": "proj-1",
        "title": "Нарисовать макеты главной страницы",
        "status": "done",
        "priority": "high",
        "assignee": "Анна",
        "created_at": "2025-01-16",
    },
    "task-2": {
        "id": "task-2",
        "project_id": "proj-1",
        "title": "Сверстать лендинг",
        "status": "in_progress",
        "priority": "high",
        "assignee": "Борис",
        "created_at": "2025-01-20",
    },
    "task-3": {
        "id": "task-3",
        "project_id": "proj-1",
        "title": "Написать тексты для блога",
        "status": "todo",
        "priority": "medium",
        "assignee": None,
        "created_at": "2025-02-01",
    },
    "task-4": {
        "id": "task-4",
        "project_id": "proj-2",
        "title": "Настроить CI/CD для мобилки",
        "status": "in_progress",
        "priority": "high",
        "assignee": "Виктор",
        "created_at": "2025-03-05",
    },
    "task-5": {
        "id": "task-5",
        "project_id": "proj-2",
        "title": "Реализовать экран авторизации",
        "status": "todo",
        "priority": "medium",
        "assignee": "Анна",
        "created_at": "2025-03-10",
    },
    "task-6": {
        "id": "task-6",
        "project_id": "proj-3",
        "title": "Добавить rate limiting",
        "status": "todo",
        "priority": "high",
        "assignee": None,
        "created_at": "2025-02-15",
    },
}


# ══════════════════════════════════════════════════════════════════
# Инструменты
# ══════════════════════════════════════════════════════════════════

@mcp.tool()
def list_projects() -> list[dict]:
    """Возвращает список всех проектов в трекере."""
    return list(PROJECTS.values())


@mcp.tool()
def get_project(project_id: str) -> dict:
    """Возвращает подробную информацию о проекте по его ID.

    Args:
        project_id: ID проекта (например 'proj-1')
    """
    proj = PROJECTS.get(project_id)
    if not proj:
        return {"error": f"Проект '{project_id}' не найден"}
    task_count = sum(1 for t in TASKS.values() if t["project_id"] == project_id)
    return {**proj, "task_count": task_count}


@mcp.tool()
def list_tasks(project_id: str, status: str | None = None) -> list[dict]:
    """Возвращает задачи проекта, опционально фильтруя по статусу.

    Args:
        project_id: ID проекта
        status: Фильтр по статусу: 'todo', 'in_progress', 'done' (или пусто для всех)
    """
    if project_id not in PROJECTS:
        return [{"error": f"Проект '{project_id}' не найден"}]

    result = [
        t for t in TASKS.values()
        if t["project_id"] == project_id
        and (status is None or t["status"] == status)
    ]
    return result


@mcp.tool()
def create_task(
    project_id: str,
    title: str,
    priority: str = "medium",
    assignee: str | None = None,
) -> dict:
    """Создаёт новую задачу в проекте.

    Args:
        project_id: ID проекта
        title: Название задачи
        priority: Приоритет: 'low', 'medium', 'high'
        assignee: Исполнитель (имя или None)
    """
    global _task_counter
    if project_id not in PROJECTS:
        return {"error": f"Проект '{project_id}' не найден"}

    _task_counter += 1
    task_id = f"task-{_task_counter}"
    task = {
        "id": task_id,
        "project_id": project_id,
        "title": title,
        "status": "todo",
        "priority": priority,
        "assignee": assignee,
        "created_at": datetime.now().strftime("%Y-%m-%d"),
    }
    TASKS[task_id] = task
    return task


@mcp.tool()
def update_task_status(task_id: str, new_status: str) -> dict:
    """Изменяет статус задачи.

    Args:
        task_id: ID задачи (например 'task-1')
        new_status: Новый статус: 'todo', 'in_progress', 'done'
    """
    valid = ("todo", "in_progress", "done")
    if new_status not in valid:
        return {"error": f"Недопустимый статус '{new_status}'. Допустимые: {valid}"}

    task = TASKS.get(task_id)
    if not task:
        return {"error": f"Задача '{task_id}' не найдена"}

    old_status = task["status"]
    task["status"] = new_status
    return {"id": task_id, "old_status": old_status, "new_status": new_status}


# ══════════════════════════════════════════════════════════════════
# Self-test
# ══════════════════════════════════════════════════════════════════

def self_test() -> None:
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  День 17. MCP-сервер трекера — self-test                    ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    tests = 0
    passed = 0

    # list_projects
    tests += 1
    projects = list_projects()
    ok = len(projects) == 3
    print(f"  {'✅' if ok else '❌'} list_projects() → {len(projects)} проектов")
    if ok: passed += 1

    # get_project
    tests += 1
    p = get_project("proj-1")
    ok = p["name"] == "Веб-сайт компании" and p["task_count"] == 3
    print(f"  {'✅' if ok else '❌'} get_project('proj-1') → {p['name']}, {p['task_count']} задач")
    if ok: passed += 1

    # list_tasks
    tests += 1
    tasks = list_tasks("proj-1")
    ok = len(tasks) == 3
    print(f"  {'✅' if ok else '❌'} list_tasks('proj-1') → {len(tasks)} задач")
    if ok: passed += 1

    # list_tasks с фильтром
    tests += 1
    tasks = list_tasks("proj-1", status="todo")
    ok = len(tasks) == 1
    print(f"  {'✅' if ok else '❌'} list_tasks('proj-1', status='todo') → {len(tasks)} задача")
    if ok: passed += 1

    # create_task
    tests += 1
    new = create_task("proj-2", "Тестовая задача", priority="low", assignee="Тест")
    ok = new["status"] == "todo" and new["project_id"] == "proj-2"
    print(f"  {'✅' if ok else '❌'} create_task() → id={new['id']}")
    if ok: passed += 1

    # update_task_status
    tests += 1
    upd = update_task_status(new["id"], "in_progress")
    ok = upd.get("new_status") == "in_progress"
    print(f"  {'✅' if ok else '❌'} update_task_status('{new['id']}', 'in_progress') → ok")
    if ok: passed += 1

    # Ошибка: несуществующий проект
    tests += 1
    err = get_project("proj-999")
    ok = "error" in err
    print(f"  {'✅' if ok else '❌'} get_project('proj-999') → ошибка: {err.get('error', '?')}")
    if ok: passed += 1

    print(f"\n  Результат: {passed}/{tests} тестов пройдено")


# ══════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if "--test" in sys.argv:
        self_test()
    else:
        mcp.run(transport="stdio")

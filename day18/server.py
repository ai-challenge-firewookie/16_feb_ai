"""День 18. MCP-сервер: планировщик с отложенными и периодическими задачами.

Инструменты:
  - create_job         — создать задачу (разовую / периодическую)
  - list_jobs          — список всех задач (с фильтрацией)
  - delete_job         — удалить задачу по ID
  - get_job_history    — история выполнений конкретной задачи
  - get_summary        — агрегированная сводка по всем задачам
  - trigger_job        — принудительно выполнить задачу прямо сейчас

Данные хранятся в SQLite (scheduler.db).

Запуск:
  python server.py             — stdio-транспорт (для MCP-клиентов)
  python server.py --test      — self-test
"""

from __future__ import annotations

import logging
import os
import sys
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path

if os.environ.get("MCP_QUIET"):
    logging.getLogger("mcp").setLevel(logging.ERROR)

import json as _json
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("day18-scheduler")

DB_PATH = Path(__file__).parent / "scheduler.db"


def _j(obj: object) -> str:
    return _json.dumps(obj, ensure_ascii=False, indent=2, default=str)


# ══════════════════════════════════════════════════════════════
# SQLite: схема и подключение
# ══════════════════════════════════════════════════════════════

def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id            TEXT PRIMARY KEY,
            name          TEXT NOT NULL,
            job_type      TEXT NOT NULL CHECK(job_type IN ('once', 'periodic')),
            action        TEXT NOT NULL,
            interval_sec  INTEGER,
            next_run_at   TEXT NOT NULL,
            status        TEXT NOT NULL DEFAULT 'active'
                          CHECK(status IN ('active', 'paused', 'done', 'deleted')),
            created_at    TEXT NOT NULL,
            updated_at    TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS job_runs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id      TEXT NOT NULL REFERENCES jobs(id),
            started_at  TEXT NOT NULL,
            finished_at TEXT,
            status      TEXT NOT NULL DEFAULT 'running'
                        CHECK(status IN ('running', 'success', 'error')),
            result      TEXT,
            FOREIGN KEY (job_id) REFERENCES jobs(id)
        )
    """)
    conn.commit()
    return conn


def _parse_interval(interval: str) -> int:
    """Разбирает интервал вида '30s', '5m', '2h', '1d' в секунды."""
    interval = interval.strip().lower()
    if not interval:
        raise ValueError("Пустой интервал")
    unit = interval[-1]
    try:
        value = int(interval[:-1])
    except ValueError:
        raise ValueError(f"Некорректный интервал: '{interval}'")

    multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    if unit not in multipliers:
        raise ValueError(f"Неизвестная единица '{unit}'. Допустимые: s, m, h, d")
    return value * multipliers[unit]


# ══════════════════════════════════════════════════════════════
# MCP-инструменты
# ══════════════════════════════════════════════════════════════

@mcp.tool()
def create_job(
    name: str,
    action: str,
    job_type: str = "periodic",
    interval: str = "1h",
    delay: str = "0s",
) -> str:
    """Создаёт задачу планировщика.

    Args:
        name: Название задачи (например 'Сводка по продажам')
        action: Описание действия, которое нужно выполнить (текст команды/описание)
        job_type: Тип: 'once' (разовая) или 'periodic' (повторяющаяся)
        interval: Интервал повторения для periodic: '30s', '5m', '2h', '1d'
        delay: Задержка перед первым выполнением: '0s', '10m', '1h'
    """
    try:
        interval_sec = _parse_interval(interval) if job_type == "periodic" else None
        delay_sec = _parse_interval(delay) if delay != "0s" else 0
    except ValueError as e:
        return _j({"error": str(e)})

    if job_type not in ("once", "periodic"):
        return _j({"error": f"Неверный тип: '{job_type}'. Допустимые: once, periodic"})

    now = datetime.now()
    next_run = now + timedelta(seconds=delay_sec)
    job_id = f"job-{uuid.uuid4().hex[:8]}"

    db = _get_db()
    db.execute(
        """INSERT INTO jobs (id, name, job_type, action, interval_sec, next_run_at, status, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?)""",
        (job_id, name, job_type, action, interval_sec,
         next_run.isoformat(), now.isoformat(), now.isoformat()),
    )
    db.commit()
    db.close()

    return _j({
        "id": job_id,
        "name": name,
        "job_type": job_type,
        "action": action,
        "interval": interval if job_type == "periodic" else None,
        "next_run_at": next_run.isoformat(),
        "status": "active",
    })


@mcp.tool()
def list_jobs(status: str | None = None) -> str:
    """Возвращает список всех задач планировщика.

    Args:
        status: Фильтр по статусу: 'active', 'paused', 'done', 'deleted' (или пусто для всех кроме deleted)
    """
    db = _get_db()
    if status:
        rows = db.execute(
            "SELECT * FROM jobs WHERE status = ? ORDER BY next_run_at", (status,)
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT * FROM jobs WHERE status != 'deleted' ORDER BY next_run_at"
        ).fetchall()
    db.close()

    jobs = []
    for r in rows:
        jobs.append({
            "id": r["id"],
            "name": r["name"],
            "job_type": r["job_type"],
            "action": r["action"][:80],
            "interval_sec": r["interval_sec"],
            "next_run_at": r["next_run_at"],
            "status": r["status"],
        })
    return _j(jobs)


@mcp.tool()
def delete_job(job_id: str) -> str:
    """Удаляет (помечает как deleted) задачу планировщика.

    Args:
        job_id: ID задачи (например 'job-abc12345')
    """
    db = _get_db()
    cursor = db.execute("UPDATE jobs SET status = 'deleted', updated_at = ? WHERE id = ?",
                        (datetime.now().isoformat(), job_id))
    db.commit()
    if cursor.rowcount == 0:
        db.close()
        return _j({"error": f"Задача '{job_id}' не найдена"})
    db.close()
    return _j({"id": job_id, "status": "deleted"})


@mcp.tool()
def get_job_history(job_id: str, limit: int = 10) -> str:
    """Возвращает историю выполнений конкретной задачи.

    Args:
        job_id: ID задачи
        limit: Максимальное число записей (по умолчанию 10)
    """
    db = _get_db()
    job = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not job:
        db.close()
        return _j({"error": f"Задача '{job_id}' не найдена"})

    runs = db.execute(
        "SELECT * FROM job_runs WHERE job_id = ? ORDER BY started_at DESC LIMIT ?",
        (job_id, limit),
    ).fetchall()
    db.close()

    return _j({
        "job": {
            "id": job["id"],
            "name": job["name"],
            "status": job["status"],
            "job_type": job["job_type"],
        },
        "runs": [
            {
                "id": r["id"],
                "started_at": r["started_at"],
                "finished_at": r["finished_at"],
                "status": r["status"],
                "result": r["result"][:200] if r["result"] else None,
            }
            for r in runs
        ],
        "total_runs": len(runs),
    })


@mcp.tool()
def trigger_job(job_id: str) -> str:
    """Принудительно запускает задачу прямо сейчас (без ожидания расписания).

    Args:
        job_id: ID задачи для немедленного выполнения
    """
    db = _get_db()
    job = db.execute("SELECT * FROM jobs WHERE id = ? AND status = 'active'", (job_id,)).fetchone()
    if not job:
        db.close()
        return _j({"error": f"Активная задача '{job_id}' не найдена"})

    now = datetime.now()
    run_id = db.execute(
        "INSERT INTO job_runs (job_id, started_at, status) VALUES (?, ?, 'running')",
        (job_id, now.isoformat()),
    ).lastrowid

    # Имитируем выполнение действия
    result = _execute_action(job["action"])

    db.execute(
        "UPDATE job_runs SET finished_at = ?, status = 'success', result = ? WHERE id = ?",
        (datetime.now().isoformat(), result, run_id),
    )

    # Обновляем next_run для periodic
    if job["job_type"] == "periodic" and job["interval_sec"]:
        next_run = now + timedelta(seconds=job["interval_sec"])
        db.execute(
            "UPDATE jobs SET next_run_at = ?, updated_at = ? WHERE id = ?",
            (next_run.isoformat(), now.isoformat(), job_id),
        )

    db.commit()
    db.close()

    return _j({
        "job_id": job_id,
        "run_id": run_id,
        "status": "success",
        "result": result[:300],
    })


@mcp.tool()
def get_summary() -> str:
    """Возвращает агрегированную сводку по всем задачам планировщика:
    общая статистика, ближайшие задачи, последние выполнения.
    """
    db = _get_db()
    now = datetime.now()

    # Общая статистика
    stats = {}
    for status in ("active", "paused", "done"):
        count = db.execute(
            "SELECT COUNT(*) FROM jobs WHERE status = ?", (status,)
        ).fetchone()[0]
        stats[status] = count

    # Количество выполнений за последние 24 часа
    day_ago = (now - timedelta(hours=24)).isoformat()
    runs_24h = db.execute(
        "SELECT COUNT(*) FROM job_runs WHERE started_at > ?", (day_ago,)
    ).fetchone()[0]
    errors_24h = db.execute(
        "SELECT COUNT(*) FROM job_runs WHERE started_at > ? AND status = 'error'",
        (day_ago,),
    ).fetchone()[0]

    # Ближайшие 5 задач к выполнению
    upcoming = db.execute(
        """SELECT id, name, action, next_run_at FROM jobs
           WHERE status = 'active' ORDER BY next_run_at LIMIT 5"""
    ).fetchall()

    # Последние 5 выполнений
    recent_runs = db.execute(
        """SELECT jr.id, jr.job_id, j.name, jr.started_at, jr.status, jr.result
           FROM job_runs jr JOIN jobs j ON jr.job_id = j.id
           ORDER BY jr.started_at DESC LIMIT 5"""
    ).fetchall()

    db.close()

    return _j({
        "timestamp": now.isoformat(),
        "stats": {
            "active_jobs": stats["active"],
            "paused_jobs": stats["paused"],
            "done_jobs": stats["done"],
            "runs_last_24h": runs_24h,
            "errors_last_24h": errors_24h,
        },
        "upcoming": [
            {"id": r["id"], "name": r["name"], "action": r["action"][:60], "next_run_at": r["next_run_at"]}
            for r in upcoming
        ],
        "recent_runs": [
            {
                "run_id": r["id"],
                "job_id": r["job_id"],
                "job_name": r["name"],
                "started_at": r["started_at"],
                "status": r["status"],
                "result": r["result"][:100] if r["result"] else None,
            }
            for r in recent_runs
        ],
    })


# ══════════════════════════════════════════════════════════════
# Симуляция выполнения действия
# ══════════════════════════════════════════════════════════════

def _execute_action(action: str) -> str:
    """Исполняет действие задачи (симуляция). В реальном проекте здесь будет
    вызов API, сбор данных, отправка уведомлений и т.п."""
    action_lower = action.lower()

    if "сводка" in action_lower or "summary" in action_lower or "отчёт" in action_lower:
        import random
        return _json.dumps({
            "type": "summary",
            "metrics": {
                "users_online": random.randint(100, 500),
                "requests_per_min": random.randint(200, 1200),
                "error_rate": round(random.uniform(0.1, 3.5), 2),
                "avg_response_ms": random.randint(50, 300),
            },
            "generated_at": datetime.now().isoformat(),
        }, ensure_ascii=False)

    if "напоминание" in action_lower or "reminder" in action_lower:
        return _json.dumps({
            "type": "reminder",
            "message": action,
            "delivered_at": datetime.now().isoformat(),
        }, ensure_ascii=False)

    if "сбор" in action_lower or "collect" in action_lower or "данные" in action_lower:
        import random
        return _json.dumps({
            "type": "data_collection",
            "records_collected": random.randint(50, 500),
            "source": "mock_api",
            "collected_at": datetime.now().isoformat(),
        }, ensure_ascii=False)

    if "проверка" in action_lower or "health" in action_lower or "check" in action_lower:
        import random
        services = ["api", "db", "cache", "queue"]
        return _json.dumps({
            "type": "health_check",
            "services": {s: random.choice(["ok", "ok", "ok", "degraded"]) for s in services},
            "checked_at": datetime.now().isoformat(),
        }, ensure_ascii=False)

    return _json.dumps({
        "type": "generic",
        "action": action,
        "executed_at": datetime.now().isoformat(),
        "status": "completed",
    }, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════
# Self-test
# ══════════════════════════════════════════════════════════════

def self_test() -> None:
    import json
    import tempfile
    global DB_PATH

    # Используем временную БД для тестов
    DB_PATH = Path(tempfile.mktemp(suffix=".db"))

    print("=" * 62)
    print("  День 18. MCP-сервер планировщика — self-test")
    print("=" * 62)
    print()

    tests = 0
    passed = 0

    # create_job (periodic)
    tests += 1
    result = json.loads(create_job("Сводка метрик", "Собрать сводку по серверам", "periodic", "5m"))
    ok = "id" in result and result["status"] == "active"
    print(f"  {'OK' if ok else 'FAIL'} create_job(periodic) -> {result.get('id', '?')}")
    if ok: passed += 1
    job1_id = result.get("id")

    # create_job (once)
    tests += 1
    result = json.loads(create_job("Напоминание", "Напоминание: позвонить клиенту", "once", delay="10m"))
    ok = "id" in result and result["job_type"] == "once"
    print(f"  {'OK' if ok else 'FAIL'} create_job(once) -> {result.get('id', '?')}")
    if ok: passed += 1
    job2_id = result.get("id")

    # create_job (health check)
    tests += 1
    result = json.loads(create_job("Health Check", "Проверка сервисов", "periodic", "1h"))
    ok = "id" in result
    print(f"  {'OK' if ok else 'FAIL'} create_job(health) -> {result.get('id', '?')}")
    if ok: passed += 1
    job3_id = result.get("id")

    # list_jobs
    tests += 1
    jobs = json.loads(list_jobs())
    ok = len(jobs) == 3
    print(f"  {'OK' if ok else 'FAIL'} list_jobs() -> {len(jobs)} задач")
    if ok: passed += 1

    # list_jobs с фильтром
    tests += 1
    jobs = json.loads(list_jobs(status="active"))
    ok = len(jobs) == 3
    print(f"  {'OK' if ok else 'FAIL'} list_jobs(status='active') -> {len(jobs)} задач")
    if ok: passed += 1

    # trigger_job
    tests += 1
    result = json.loads(trigger_job(job1_id))
    ok = result.get("status") == "success"
    print(f"  {'OK' if ok else 'FAIL'} trigger_job('{job1_id}') -> {result.get('status', '?')}")
    if ok: passed += 1

    # trigger ещё раз для истории
    trigger_job(job1_id)
    trigger_job(job3_id)

    # get_job_history
    tests += 1
    history = json.loads(get_job_history(job1_id))
    ok = history["total_runs"] == 2
    print(f"  {'OK' if ok else 'FAIL'} get_job_history('{job1_id}') -> {history['total_runs']} выполнений")
    if ok: passed += 1

    # get_summary
    tests += 1
    summary = json.loads(get_summary())
    ok = summary["stats"]["active_jobs"] == 3 and summary["stats"]["runs_last_24h"] >= 3
    print(f"  {'OK' if ok else 'FAIL'} get_summary() -> {summary['stats']['active_jobs']} active, {summary['stats']['runs_last_24h']} runs")
    if ok: passed += 1

    # delete_job
    tests += 1
    result = json.loads(delete_job(job2_id))
    ok = result.get("status") == "deleted"
    print(f"  {'OK' if ok else 'FAIL'} delete_job('{job2_id}') -> {result.get('status', '?')}")
    if ok: passed += 1

    # Проверяем, что deleted не в списке
    tests += 1
    jobs = json.loads(list_jobs())
    ok = len(jobs) == 2
    print(f"  {'OK' if ok else 'FAIL'} list_jobs() после удаления -> {len(jobs)} задач")
    if ok: passed += 1

    # Ошибка: неверный интервал
    tests += 1
    result = json.loads(create_job("Bad", "test", "periodic", "5x"))
    ok = "error" in result
    print(f"  {'OK' if ok else 'FAIL'} create_job(interval='5x') -> ошибка: {result.get('error', '?')[:50]}")
    if ok: passed += 1

    # Ошибка: несуществующая задача
    tests += 1
    result = json.loads(trigger_job("job-nonexistent"))
    ok = "error" in result
    print(f"  {'OK' if ok else 'FAIL'} trigger_job('job-nonexistent') -> ошибка")
    if ok: passed += 1

    print(f"\n  Результат: {passed}/{tests} тестов пройдено")

    # Cleanup
    DB_PATH.unlink(missing_ok=True)


# ══════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if "--test" in sys.argv:
        self_test()
    else:
        mcp.run(transport="stdio")

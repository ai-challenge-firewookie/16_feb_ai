"""День 18. Фоновый планировщик — демон, который выполняет задачи по расписанию.

Запускается как отдельный процесс, проверяет SQLite каждые N секунд,
выполняет просроченные задачи, записывает результаты.

Для периодических задач — после выполнения обновляет next_run_at.
Для разовых (once) — помечает как done.

Может генерировать AI-сводку через Claude API.

Запуск:
  python scheduler.py                — запуск демона (poll каждые 5с)
  python scheduler.py --once         — однократная проверка и выполнение
  python scheduler.py --summary      — сгенерировать AI-сводку по истории
"""

from __future__ import annotations

import os
import sys
import time
import sqlite3
import signal
from datetime import datetime, timedelta
from pathlib import Path

import json

# ══════════════════════════════════════════════════════════════
# Конфигурация
# ══════════════════════════════════════════════════════════════

DB_PATH = Path(__file__).parent / "scheduler.db"
POLL_INTERVAL = 5  # секунд между проверками


def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


# ══════════════════════════════════════════════════════════════
# Выполнение действий (та же логика, что в server.py)
# ══════════════════════════════════════════════════════════════

def execute_action(action: str) -> str:
    """Исполняет действие задачи."""
    action_lower = action.lower()
    import random

    if "сводка" in action_lower or "summary" in action_lower or "отчёт" in action_lower:
        return json.dumps({
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
        return json.dumps({
            "type": "reminder",
            "message": action,
            "delivered_at": datetime.now().isoformat(),
        }, ensure_ascii=False)

    if "сбор" in action_lower or "collect" in action_lower or "данные" in action_lower:
        return json.dumps({
            "type": "data_collection",
            "records_collected": random.randint(50, 500),
            "source": "mock_api",
            "collected_at": datetime.now().isoformat(),
        }, ensure_ascii=False)

    if "проверка" in action_lower or "health" in action_lower or "check" in action_lower:
        services = ["api", "db", "cache", "queue"]
        return json.dumps({
            "type": "health_check",
            "services": {s: random.choice(["ok", "ok", "ok", "degraded"]) for s in services},
            "checked_at": datetime.now().isoformat(),
        }, ensure_ascii=False)

    return json.dumps({
        "type": "generic",
        "action": action,
        "executed_at": datetime.now().isoformat(),
        "status": "completed",
    }, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════
# Основной цикл
# ══════════════════════════════════════════════════════════════

def check_and_run() -> int:
    """Проверяет и выполняет все просроченные задачи. Возвращает кол-во выполненных."""
    db = _get_db()
    now = datetime.now()

    # Находим активные задачи, у которых next_run_at <= now
    due_jobs = db.execute(
        "SELECT * FROM jobs WHERE status = 'active' AND next_run_at <= ?",
        (now.isoformat(),),
    ).fetchall()

    executed = 0
    for job in due_jobs:
        job_id = job["id"]
        action = job["action"]

        # Создаём запись о запуске
        run_id = db.execute(
            "INSERT INTO job_runs (job_id, started_at, status) VALUES (?, ?, 'running')",
            (job_id, now.isoformat()),
        ).lastrowid

        try:
            result = execute_action(action)
            db.execute(
                "UPDATE job_runs SET finished_at = ?, status = 'success', result = ? WHERE id = ?",
                (datetime.now().isoformat(), result, run_id),
            )

            if job["job_type"] == "periodic" and job["interval_sec"]:
                next_run = now + timedelta(seconds=job["interval_sec"])
                db.execute(
                    "UPDATE jobs SET next_run_at = ?, updated_at = ? WHERE id = ?",
                    (next_run.isoformat(), now.isoformat(), job_id),
                )
            elif job["job_type"] == "once":
                db.execute(
                    "UPDATE jobs SET status = 'done', updated_at = ? WHERE id = ?",
                    (now.isoformat(), job_id),
                )

            executed += 1
            print(f"  [OK] {job['name']} (id={job_id})")

        except Exception as e:
            db.execute(
                "UPDATE job_runs SET finished_at = ?, status = 'error', result = ? WHERE id = ?",
                (datetime.now().isoformat(), str(e), run_id),
            )
            print(f"  [ERR] {job['name']} (id={job_id}): {e}")

    db.commit()
    db.close()
    return executed


def generate_ai_summary() -> str:
    """Генерирует AI-сводку по всей истории выполнений за последние 24 часа."""
    from dotenv import load_dotenv
    from anthropic import Anthropic

    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        return "Ошибка: не задан ANTHROPIC_API_KEY"

    db = _get_db()
    now = datetime.now()
    day_ago = (now - timedelta(hours=24)).isoformat()

    # Собираем данные
    jobs = db.execute("SELECT * FROM jobs WHERE status != 'deleted'").fetchall()
    runs = db.execute(
        """SELECT jr.*, j.name as job_name, j.action
           FROM job_runs jr JOIN jobs j ON jr.job_id = j.id
           WHERE jr.started_at > ?
           ORDER BY jr.started_at DESC""",
        (day_ago,),
    ).fetchall()
    db.close()

    # Формируем контекст для Claude
    context = {
        "period": f"за последние 24 часа (с {day_ago})",
        "total_jobs": len(jobs),
        "active_jobs": sum(1 for j in jobs if j["status"] == "active"),
        "total_runs": len(runs),
        "successful_runs": sum(1 for r in runs if r["status"] == "success"),
        "failed_runs": sum(1 for r in runs if r["status"] == "error"),
        "runs_detail": [],
    }
    for r in runs[:20]:  # последние 20 для контекста
        detail = {
            "job_name": r["job_name"],
            "action": r["action"],
            "status": r["status"],
            "started_at": r["started_at"],
            "result_preview": r["result"][:200] if r["result"] else None,
        }
        context["runs_detail"].append(detail)

    client = Anthropic(api_key=api_key)
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=(
            "Ты — AI-ассистент, анализирующий работу фонового планировщика задач. "
            "Составь краткую, но информативную сводку на русском языке. "
            "Укажи: общий статус, количество выполнений, ошибки (если есть), "
            "и рекомендации. Формат: markdown."
        ),
        messages=[{
            "role": "user",
            "content": f"Данные планировщика:\n```json\n{json.dumps(context, ensure_ascii=False, indent=2)}\n```\n\nСоставь сводку.",
        }],
    )

    return response.content[0].text


def run_daemon():
    """Запускает демон планировщика."""
    print("=" * 62)
    print("  День 18. Фоновый планировщик")
    print("=" * 62)
    print(f"  БД: {DB_PATH}")
    print(f"  Интервал проверки: {POLL_INTERVAL}с")
    print(f"  Ctrl+C для остановки")
    print()

    running = True

    def handle_signal(signum, frame):
        nonlocal running
        print("\n  Остановка планировщика...")
        running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    cycle = 0
    while running:
        cycle += 1
        now = datetime.now().strftime("%H:%M:%S")
        executed = check_and_run()
        if executed > 0:
            print(f"  [{now}] Цикл #{cycle}: выполнено {executed} задач")
        else:
            # Печатаем каждые 12 циклов (раз в минуту) для heartbeat
            if cycle % 12 == 0:
                print(f"  [{now}] heartbeat — цикл #{cycle}, задач к выполнению нет")

        time.sleep(POLL_INTERVAL)

    print("  Планировщик остановлен.")


# ══════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if "--once" in sys.argv:
        print("  Однократная проверка...", flush=True)
        n = check_and_run()
        print(f"  Выполнено: {n} задач")
    elif "--summary" in sys.argv:
        print("  Генерация AI-сводки...", flush=True)
        summary = generate_ai_summary()
        print(f"\n{summary}")
    else:
        run_daemon()

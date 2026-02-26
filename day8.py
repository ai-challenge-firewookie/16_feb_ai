"""День 8. Работа с токенами.

Демонстрирует:
  - подсчёт токенов для текущего запроса, истории и ответа модели
  - рост стоимости/токенов по мере диалога
  - поведение при приближении к лимиту контекстного окна
  - сравнение короткого, длинного и переполненного диалога
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field

from anthropic import Anthropic
from dotenv import load_dotenv

# ── Лимиты контекстного окна по моделям ──────────────────────────

MODEL_CONTEXT_LIMITS: dict[str, int] = {
    "claude-opus-4-6": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-haiku-4-5-20251001": 200_000,
    # Старые модели
    "claude-3-5-sonnet-20241022": 200_000,
    "claude-3-haiku-20240307": 200_000,
}

# ── Цены ($ за 1M токенов) ───────────────────────────────────────

PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-6":            {"input": 15.0,  "output": 75.0},
    "claude-sonnet-4-6":          {"input": 3.0,   "output": 15.0},
    "claude-haiku-4-5-20251001":  {"input": 0.80,  "output": 4.0},
}

DEFAULT_PRICING = {"input": 3.0, "output": 15.0}


@dataclass
class TurnStats:
    """Статистика одного хода (turn) диалога."""
    turn: int
    user_message: str
    assistant_response: str
    input_tokens: int      # сколько входных токенов съел этот запрос
    output_tokens: int     # сколько выходных токенов в ответе
    total_history_tokens: int  # накопленный размер истории после этого хода
    elapsed_sec: float
    cost_usd: float        # стоимость этого запроса


@dataclass
class DialogReport:
    """Полный отчёт о диалоге."""
    label: str
    model: str
    turns: list[TurnStats] = field(default_factory=list)
    error: str | None = None

    @property
    def total_input_tokens(self) -> int:
        return sum(t.input_tokens for t in self.turns)

    @property
    def total_output_tokens(self) -> int:
        return sum(t.output_tokens for t in self.turns)

    @property
    def total_cost_usd(self) -> float:
        return sum(t.cost_usd for t in self.turns)

    @property
    def total_elapsed(self) -> float:
        return sum(t.elapsed_sec for t in self.turns)


def calc_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Рассчитать стоимость запроса в USD."""
    prices = PRICING.get(model, DEFAULT_PRICING)
    return (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000


class TokenTracker:
    """Агент с детальным отслеживанием токенов на каждом ходе."""

    def __init__(
        self,
        client: Anthropic,
        model: str,
        system: str = "Ты — полезный AI-ассистент. Отвечай кратко, на русском.",
        max_tokens: int = 512,
    ) -> None:
        self._client = client
        self._model = model
        self._system = system
        self._max_tokens = max_tokens
        self._history: list[dict[str, str]] = []
        self._turns: list[TurnStats] = []
        self._context_limit = MODEL_CONTEXT_LIMITS.get(model, 200_000)

    def ask(self, user_message: str) -> TurnStats:
        """Отправить сообщение и вернуть детальную статистику."""
        self._history.append({"role": "user", "content": user_message})

        t0 = time.perf_counter()
        message = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=self._system,
            messages=self._history,
        )
        elapsed = time.perf_counter() - t0

        assistant_text = message.content[0].text
        self._history.append({"role": "assistant", "content": assistant_text})

        input_tokens = message.usage.input_tokens
        output_tokens = message.usage.output_tokens
        total_history = input_tokens + output_tokens  # приближение к полному контексту

        cost = calc_cost(self._model, input_tokens, output_tokens)

        turn_num = len(self._turns) + 1
        stats = TurnStats(
            turn=turn_num,
            user_message=user_message[:80],
            assistant_response=assistant_text[:80],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_history_tokens=input_tokens,  # input_tokens = весь контекст отправленный
            elapsed_sec=round(elapsed, 2),
            cost_usd=cost,
        )
        self._turns.append(stats)
        return stats

    def get_report(self, label: str) -> DialogReport:
        return DialogReport(label=label, model=self._model, turns=list(self._turns))

    def reset(self) -> None:
        self._history.clear()
        self._turns.clear()

    @property
    def context_limit(self) -> int:
        return self._context_limit

    @property
    def current_history_tokens_approx(self) -> int:
        """Грубая оценка текущего размера истории в токенах."""
        total_chars = sum(len(m["content"]) for m in self._history)
        total_chars += len(self._system)
        return total_chars // 4


# ── Сценарии демонстрации ─────────────────────────────────────────

SHORT_DIALOG = [
    "Привет! Как тебя зовут?",
    "Какая столица Франции?",
    "Спасибо!",
]

LONG_DIALOG = [
    "Расскажи про историю Python — когда создан, кем, основные вехи.",
    "А что такое GIL в Python и почему он существует?",
    "Как работает async/await в Python? Объясни просто.",
    "Какие есть альтернативы GIL? Расскажи про free-threading в Python 3.13.",
    "Сравни Python и Rust по производительности и удобству.",
    "Какие фреймворки для веб-разработки есть в Python? Сравни Flask, Django, FastAPI.",
    "Расскажи про систему типов в Python — typing, mypy, pyright.",
    "Что нового в Python 3.12 и 3.13?",
]

# Для переполнения — генерируем сообщения с большим объёмом текста
OVERFLOW_SEED = "Перечисли и подробно опиши все известные тебе "
OVERFLOW_TOPICS = [
    "алгоритмы сортировки с примерами кода на Python",
    "паттерны проектирования GoF с примерами на Python",
    "структуры данных с их сложностями по Big-O",
    "протоколы сетевого стека TCP/IP с подробным описанием каждого уровня",
    "виды баз данных — реляционные, документные, графовые, временные ряды — с примерами",
    "все HTTP-методы и коды ответов с описанием каждого",
    "все виды тестирования ПО — unit, integration, e2e, property-based и т.д.",
    "фичи Python от 3.8 до 3.13 включительно по каждой версии",
    "все виды SQL-запросов с примерами — JOIN, подзапросы, оконные функции, CTE",
    "все основные утилиты командной строки Linux — cat, grep, awk, sed, find, xargs и т.д.",
]


def run_scenario(
    client: Anthropic,
    model: str,
    label: str,
    messages: list[str],
    max_tokens: int = 512,
    verbose: bool = True,
) -> DialogReport:
    """Запустить сценарий диалога и собрать статистику токенов."""
    tracker = TokenTracker(client=client, model=model, max_tokens=max_tokens)

    if verbose:
        print(f"\n{'='*70}")
        print(f"  {label}")
        print(f"  Модель: {model} | Лимит контекста: {tracker.context_limit:,} токенов")
        print(f"{'='*70}")

    for i, msg in enumerate(messages, 1):
        if verbose:
            print(f"\n── Ход {i}/{len(messages)} ──")
            print(f"  User: {msg[:70]}{'...' if len(msg) > 70 else ''}")

        try:
            stats = tracker.ask(msg)
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"\n  ОШИБКА: {error_msg}")

            report = tracker.get_report(label)
            report.error = error_msg
            return report

        if verbose:
            print(f"  Assistant: {stats.assistant_response[:70]}{'...' if len(stats.assistant_response) > 70 else ''}")
            print(f"  Токены: input={stats.input_tokens:,}  output={stats.output_tokens:,}  "
                  f"контекст={stats.total_history_tokens:,}")
            cost_str = f"${stats.cost_usd:.6f}" if stats.cost_usd < 0.01 else f"${stats.cost_usd:.4f}"
            print(f"  Стоимость: {cost_str}  |  Время: {stats.elapsed_sec}s")

            # Прогресс-бар заполнения контекста
            fill_pct = stats.total_history_tokens / tracker.context_limit * 100
            bar_len = 30
            filled = int(bar_len * fill_pct / 100)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"  Контекст: [{bar}] {fill_pct:.1f}%")

    return tracker.get_report(label)


def print_report(report: DialogReport) -> None:
    """Вывести итоговый отчёт по диалогу."""
    print(f"\n{'─'*60}")
    print(f"  ИТОГ: {report.label}")
    print(f"{'─'*60}")

    if not report.turns:
        print("  (нет данных)")
        return

    # Таблица по ходам
    print(f"\n  {'Ход':>4}  {'Input':>8}  {'Output':>8}  {'Контекст':>10}  {'Стоимость':>12}  {'Время':>7}")
    print(f"  {'─'*4}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*12}  {'─'*7}")

    cumulative_cost = 0.0
    for t in report.turns:
        cumulative_cost += t.cost_usd
        cost_str = f"${t.cost_usd:.6f}" if t.cost_usd < 0.01 else f"${t.cost_usd:.4f}"
        print(f"  {t.turn:>4}  {t.input_tokens:>8,}  {t.output_tokens:>8,}  "
              f"{t.total_history_tokens:>10,}  {cost_str:>12}  {t.elapsed_sec:>6.1f}s")

    print(f"\n  Всего ходов:      {len(report.turns)}")
    print(f"  Input  токенов:   {report.total_input_tokens:,}")
    print(f"  Output токенов:   {report.total_output_tokens:,}")
    total_cost_str = f"${report.total_cost_usd:.6f}" if report.total_cost_usd < 0.01 else f"${report.total_cost_usd:.4f}"
    print(f"  Суммарная стоим.: {total_cost_str}")
    print(f"  Суммарное время:  {report.total_elapsed:.1f}s")

    if report.error:
        print(f"\n  ОШИБКА: {report.error}")

    # Рост input-токенов от хода к ходу
    if len(report.turns) >= 2:
        print(f"\n  Рост input-токенов (контекста) по ходам:")
        for i, t in enumerate(report.turns):
            bar_width = min(t.input_tokens // 100, 50)
            bar = "▓" * bar_width
            print(f"    Ход {t.turn}: {bar} {t.input_tokens:,}")


def print_comparison(reports: list[DialogReport]) -> None:
    """Сравнительная таблица нескольких диалогов."""
    print(f"\n{'='*70}")
    print("  СРАВНЕНИЕ ДИАЛОГОВ")
    print(f"{'='*70}")

    print(f"\n  {'Диалог':<30}  {'Ходов':>6}  {'Input':>10}  {'Output':>10}  {'Стоимость':>12}")
    print(f"  {'─'*30}  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*12}")

    for r in reports:
        cost_str = f"${r.total_cost_usd:.6f}" if r.total_cost_usd < 0.01 else f"${r.total_cost_usd:.4f}"
        print(f"  {r.label:<30}  {len(r.turns):>6}  {r.total_input_tokens:>10,}  "
              f"{r.total_output_tokens:>10,}  {cost_str:>12}")

    # Показать рост
    if len(reports) >= 2 and reports[0].total_input_tokens > 0:
        base = reports[0].total_input_tokens
        print(f"\n  Рост input-токенов относительно «{reports[0].label}»:")
        for r in reports[1:]:
            if r.total_input_tokens > 0:
                ratio = r.total_input_tokens / base
                print(f"    {r.label}: ×{ratio:.1f}")


def run_overflow_test(
    client: Anthropic,
    model: str,
    max_tokens: int = 4096,
    verbose: bool = True,
) -> DialogReport:
    """Тест на переполнение контекстного окна.

    Отправляет длинные запросы, требующие объёмных ответов, пока модель не
    упрётся в лимит контекста.
    """
    messages = [OVERFLOW_SEED + topic for topic in OVERFLOW_TOPICS]
    return run_scenario(
        client=client,
        model=model,
        label="Переполнение контекста",
        messages=messages,
        max_tokens=max_tokens,
        verbose=verbose,
    )


# ── CLI ───────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise SystemExit("Missing API key: set ANTHROPIC_API_KEY or API_KEY in .env")

    client = Anthropic(api_key=api_key)
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    print("╔══════════════════════════════════════════════════╗")
    print("║       День 8. Работа с токенами                 ║")
    print("╚══════════════════════════════════════════════════╝")
    print(f"\nМодель: {model}")
    prices = PRICING.get(model, DEFAULT_PRICING)
    print(f"Цены: ${prices['input']}/1M input, ${prices['output']}/1M output")
    print()

    reports: list[DialogReport] = []

    # 1. Короткий диалог
    r1 = run_scenario(client, model, "Короткий диалог (3 хода)", SHORT_DIALOG)
    print_report(r1)
    reports.append(r1)

    # 2. Длинный диалог
    r2 = run_scenario(client, model, "Длинный диалог (8 ходов)", LONG_DIALOG)
    print_report(r2)
    reports.append(r2)

    # 3. Тест на переполнение
    print("\n⚠️  Тест на переполнение — генерируем объёмные запросы...")
    r3 = run_overflow_test(client, model, max_tokens=4096)
    print_report(r3)
    reports.append(r3)

    # Итоговое сравнение
    print_comparison(reports)

    # Вывод
    print(f"\n{'='*70}")
    print("  ВЫВОДЫ")
    print(f"{'='*70}")
    print("""
  1. Input-токены растут с каждым ходом, т.к. вся история отправляется заново.
     Это основная причина роста стоимости длинных диалогов.

  2. Стоимость одного хода увеличивается квадратично:
     ход N стоит ~N× от хода 1, т.к. input = system + вся предыдущая история.

  3. При приближении к лимиту контекстного окна (200K для Claude):
     - API вернёт ошибку (400 Bad Request) если input + max_tokens > лимит
     - Модель может начать «забывать» ранний контекст ещё до формального лимита

  4. Решения для длинных диалогов:
     - /compact — сжатие истории в резюме (экономит 60-90% токенов)
     - Sliding window — хранить только последние N ходов
     - RAG — выносить историю во внешнюю память
""")


if __name__ == "__main__":
    main()

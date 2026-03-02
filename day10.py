"""День 10. Три стратегии управления контекстом.

Сравнивает три подхода:
  1. Sliding Window  — хранить только последние N пар user/assistant
  2. Facts Extraction — извлекать ключевые факты в структурированный список
  3. Branching       — форкать историю на «ветки» по теме

Для каждой стратегии: одинаковый диалог → сравнение токенов + качества ответов.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable

from anthropic import Anthropic
from dotenv import load_dotenv

from day8 import calc_cost, PRICING, DEFAULT_PRICING

# ── Промпты ──────────────────────────────────────────────────────

FACTS_EXTRACTION_PROMPT = """\
Из диалога ниже извлеки все важные факты в виде краткого структурированного списка.
Сохрани: имена, числа, технические детали, решения, предпочтения, цели.
Формат: маркированный список (•), без лишних слов.
Максимум 15 пунктов.

--- ДИАЛОГ ---
{conversation}
--- КОНЕЦ ---"""

SYSTEM_PROMPT = "Ты — полезный AI-ассистент. Отвечай кратко, по делу, на русском языке."


# ── Общий dataclass результата ────────────────────────────────────

@dataclass
class TurnRecord:
    turn: int
    user_msg: str
    assistant_msg: str
    input_tokens: int
    output_tokens: int
    elapsed_sec: float

    @property
    def cost(self) -> float:
        return 0.0  # заполняется снаружи через calc_cost


@dataclass
class StrategyResult:
    label: str
    model: str
    turns: list[TurnRecord] = field(default_factory=list)
    quality_scores: list[dict] = field(default_factory=list)
    extra_tokens: int = 0     # токены на служебные запросы (сжатие, извлечение)

    @property
    def total_input(self) -> int:
        return sum(t.input_tokens for t in self.turns)

    @property
    def total_output(self) -> int:
        return sum(t.output_tokens for t in self.turns)

    @property
    def total_cost(self) -> float:
        return sum(calc_cost(self.model, t.input_tokens, t.output_tokens) for t in self.turns)

    @property
    def quality_pct(self) -> float:
        if not self.quality_scores:
            return 0.0
        found = sum(s["found"] for s in self.quality_scores)
        total = sum(s["total"] for s in self.quality_scores)
        return found / total * 100 if total else 0.0


# ══════════════════════════════════════════════════════════════════
# Стратегия 1: Sliding Window
# ══════════════════════════════════════════════════════════════════

class SlidingWindowAgent:
    """Хранит только последние `window` пар сообщений (user + assistant)."""

    def __init__(
        self,
        client: Anthropic,
        model: str,
        window: int = 4,
        max_tokens: int = 512,
    ) -> None:
        self._client = client
        self._model = model
        self._window = window   # пар (каждая = 2 сообщения)
        self._max_tokens = max_tokens
        self._history: list[dict[str, str]] = []

    def ask(self, user_message: str) -> tuple[str, int, int, float]:
        self._history.append({"role": "user", "content": user_message})

        # Срезаем: оставляем window*2 последних сообщений
        messages = self._history[-(self._window * 2):]

        t0 = time.perf_counter()
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        elapsed = time.perf_counter() - t0

        text = resp.content[0].text
        self._history.append({"role": "assistant", "content": text})

        return text, resp.usage.input_tokens, resp.usage.output_tokens, round(elapsed, 2)

    @property
    def window_size(self) -> int:
        return self._window


# ══════════════════════════════════════════════════════════════════
# Стратегия 2: Facts Extraction
# ══════════════════════════════════════════════════════════════════

class FactsAgent:
    """
    После каждых `extract_every` ходов извлекает факты из истории в краткий список.
    В контекст подаётся: [факты] + последние `keep_recent` сообщений.
    """

    def __init__(
        self,
        client: Anthropic,
        model: str,
        extract_every: int = 4,
        keep_recent: int = 4,
        max_tokens: int = 512,
    ) -> None:
        self._client = client
        self._model = model
        self._extract_every = extract_every
        self._keep_recent = keep_recent
        self._max_tokens = max_tokens

        self._history: list[dict[str, str]] = []
        self._facts: str | None = None
        self._turn_count = 0
        self._extra_tokens = 0

    def ask(self, user_message: str) -> tuple[str, int, int, float]:
        self._history.append({"role": "user", "content": user_message})
        self._turn_count += 1

        # Извлечение фактов каждые N ходов
        if self._turn_count % self._extract_every == 0 and len(self._history) > self._keep_recent:
            self._extract_facts()

        messages = self._build_messages()

        t0 = time.perf_counter()
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        elapsed = time.perf_counter() - t0

        text = resp.content[0].text
        self._history.append({"role": "assistant", "content": text})

        return text, resp.usage.input_tokens, resp.usage.output_tokens, round(elapsed, 2)

    def _extract_facts(self) -> None:
        """Извлечь факты из истории."""
        parts = []
        if self._facts:
            parts.append(f"[Существующие факты]:\n{self._facts}")
        for msg in self._history:
            role = "Пользователь" if msg["role"] == "user" else "Ассистент"
            parts.append(f"{role}: {msg['content']}")
        conversation = "\n\n".join(parts)

        resp = self._client.messages.create(
            model=self._model,
            max_tokens=400,
            messages=[{"role": "user", "content": FACTS_EXTRACTION_PROMPT.format(conversation=conversation)}],
        )
        self._facts = resp.content[0].text.strip()
        self._extra_tokens += resp.usage.input_tokens + resp.usage.output_tokens

    def _build_messages(self) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if self._facts:
            messages.append({
                "role": "user",
                "content": f"[Известные факты из нашего разговора]:\n{self._facts}",
            })
            messages.append({
                "role": "assistant",
                "content": "Понял, учитываю все эти факты. Продолжаем.",
            })
        recent = self._history[-(self._keep_recent):]
        messages.extend(recent)
        return messages

    @property
    def facts(self) -> str | None:
        return self._facts

    @property
    def extra_tokens(self) -> int:
        return self._extra_tokens


# ══════════════════════════════════════════════════════════════════
# Стратегия 3: Branching
# ══════════════════════════════════════════════════════════════════

class BranchingAgent:
    """
    Разбивает диалог на именованные ветки по теме.
    В контекст включается: summary всех прошлых веток + полная текущая ветка.

    Смена ветки происходит когда:
      - пользователь явно запрашивает /branch <name>
      - или автоматически каждые `auto_branch_every` ходов
    """

    def __init__(
        self,
        client: Anthropic,
        model: str,
        auto_branch_every: int = 5,
        max_tokens: int = 512,
    ) -> None:
        self._client = client
        self._model = model
        self._auto_branch_every = auto_branch_every
        self._max_tokens = max_tokens

        # branches: {name: [messages]}
        self._branches: dict[str, list[dict[str, str]]] = {}
        self._branch_summaries: dict[str, str] = {}
        self._current_branch = "main"
        self._branches["main"] = []
        self._turn_count = 0
        self._extra_tokens = 0

    def ask(self, user_message: str) -> tuple[str, int, int, float]:
        # Проверяем команду смены ветки
        if user_message.startswith("/branch "):
            branch_name = user_message[8:].strip()
            self._switch_branch(branch_name)
            return f"[Ветка переключена на: {branch_name}]", 0, 0, 0.0

        self._turn_count += 1

        # Автосмена ветки
        if (self._turn_count > 1 and
                self._turn_count % self._auto_branch_every == 1 and
                len(self._branches[self._current_branch]) > 0):
            old_branch = self._current_branch
            new_branch = f"topic_{self._turn_count}"
            self._summarize_and_switch(old_branch, new_branch)

        self._branches[self._current_branch].append({"role": "user", "content": user_message})

        messages = self._build_messages()

        t0 = time.perf_counter()
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        elapsed = time.perf_counter() - t0

        text = resp.content[0].text
        self._branches[self._current_branch].append({"role": "assistant", "content": text})

        return text, resp.usage.input_tokens, resp.usage.output_tokens, round(elapsed, 2)

    def _switch_branch(self, name: str) -> None:
        if name not in self._branches:
            self._branches[name] = []
        self._current_branch = name

    def _summarize_and_switch(self, old_branch: str, new_branch: str) -> None:
        """Сохранить summary текущей ветки и переключиться на новую."""
        branch_msgs = self._branches[old_branch]
        if not branch_msgs:
            return

        lines = []
        for msg in branch_msgs:
            role = "Пользователь" if msg["role"] == "user" else "Ассистент"
            lines.append(f"{role}: {msg['content'][:200]}")
        conversation = "\n".join(lines)

        resp = self._client.messages.create(
            model=self._model,
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": (
                    f"Дай краткое резюме (2-3 предложения) этого фрагмента диалога:\n\n{conversation}"
                ),
            }],
        )
        self._branch_summaries[old_branch] = resp.content[0].text.strip()
        self._extra_tokens += resp.usage.input_tokens + resp.usage.output_tokens

        self._branches[new_branch] = []
        self._current_branch = new_branch

    def _build_messages(self) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []

        # Summaries прошлых веток
        if self._branch_summaries:
            summaries_text = "\n".join(
                f"• [{branch}]: {summary}"
                for branch, summary in self._branch_summaries.items()
            )
            messages.append({
                "role": "user",
                "content": f"[Контекст из предыдущих тем разговора]:\n{summaries_text}",
            })
            messages.append({
                "role": "assistant",
                "content": "Понял контекст, продолжаем.",
            })

        # Полная текущая ветка
        messages.extend(self._branches[self._current_branch])
        return messages

    @property
    def current_branch(self) -> str:
        return self._current_branch

    @property
    def branch_count(self) -> int:
        return len(self._branches)

    @property
    def extra_tokens(self) -> int:
        return self._extra_tokens


# ══════════════════════════════════════════════════════════════════
# Сценарий для сравнения (тот же, что в day9)
# ══════════════════════════════════════════════════════════════════

DIALOG_MESSAGES = [
    # 1-2: Вводные
    "Привет! Меня зовут Никита. Я разрабатываю проект на Python 3.13.",
    "Проект называется DataPipeline. Он обрабатывает CSV-файлы и загружает в PostgreSQL.",
    # 3-4: Технические детали
    "Я использую библиотеки: pandas, sqlalchemy, alembic. База крутится в Docker.",
    "Сейчас обработка 1 млн строк занимает 45 секунд. Хочу ускорить до 10 секунд.",
    # 5-6: Обсуждение решений
    "Какие есть способы ускорить загрузку в PostgreSQL? Интересует COPY vs INSERT.",
    "Расскажи про pandas read_csv с chunksize — как правильно подобрать размер чанка?",
    # 7-8: Детали реализации
    "Мы решили использовать COPY FROM через psycopg2.copy_expert. Покажи пример.",
    "А как обработать ошибки в середине COPY? Нужна транзакционность.",
    # 9-10: Новая тема, ссылка на контекст
    "Ещё вопрос — нужно добавить валидацию CSV перед загрузкой. Какие есть подходы?",
    "Давай использовать pydantic для валидации строк CSV. Как интегрировать с pandas?",
    # 11-12: Тест на память о раннем контексте
    "Напомни, как зовут меня и мой проект?",
    "Какие библиотеки мы обсуждали в начале разговора?",
    # 13-14: Ещё детали
    "Теперь нужно добавить логирование: structlog + JSON-формат. Покажи настройку.",
    "Как интегрировать structlog с SQLAlchemy, чтобы видеть SQL-запросы в логах?",
    # 15-16: Финальная проверка контекста
    "Какую целевую скорость обработки мы обсуждали для 1 млн строк?",
    "Подведи итог: какой стек технологий мы выбрали для DataPipeline?",
]

QUALITY_CHECKS = [
    {
        "index": 10,
        "keywords": ["Никита", "DataPipeline"],
        "label": "Имя + проект",
    },
    {
        "index": 11,
        "keywords": ["pandas", "sqlalchemy", "alembic"],
        "label": "Библиотеки",
    },
    {
        "index": 14,
        "keywords": ["10"],
        "label": "Целевая скорость",
    },
    {
        "index": 15,
        "keywords": ["PostgreSQL", "pandas"],
        "label": "Стек",
    },
]


def _check_quality(result: StrategyResult) -> None:
    for check in QUALITY_CHECKS:
        idx = check["index"]
        if idx >= len(result.turns):
            continue
        response_text = result.turns[idx].assistant_msg.lower()
        found = sum(1 for kw in check["keywords"] if kw.lower() in response_text)
        result.quality_scores.append({
            "label": check["label"],
            "found": found,
            "total": len(check["keywords"]),
        })


# ══════════════════════════════════════════════════════════════════
# Запуск стратегий
# ══════════════════════════════════════════════════════════════════

def run_sliding_window(
    client: Anthropic,
    model: str,
    window: int = 4,
    verbose: bool = True,
) -> StrategyResult:
    agent = SlidingWindowAgent(client=client, model=model, window=window)
    result = StrategyResult(label=f"Sliding Window (w={window})", model=model)

    if verbose:
        _print_header(result.label)

    for i, msg in enumerate(DIALOG_MESSAGES):
        if verbose:
            _print_turn(i + 1, len(DIALOG_MESSAGES), msg)

        text, inp, out, elapsed = agent.ask(msg)
        cost = calc_cost(model, inp, out)
        result.turns.append(TurnRecord(
            turn=i + 1,
            user_msg=msg[:60],
            assistant_msg=text,
            input_tokens=inp,
            output_tokens=out,
            elapsed_sec=elapsed,
        ))
        if verbose:
            _print_turn_stats(text, inp, out, cost, elapsed)

    _check_quality(result)
    return result


def run_facts_extraction(
    client: Anthropic,
    model: str,
    extract_every: int = 4,
    keep_recent: int = 4,
    verbose: bool = True,
) -> StrategyResult:
    agent = FactsAgent(client=client, model=model, extract_every=extract_every, keep_recent=keep_recent)
    result = StrategyResult(label=f"Facts Extraction (every={extract_every})", model=model)

    if verbose:
        _print_header(result.label)

    for i, msg in enumerate(DIALOG_MESSAGES):
        if verbose:
            _print_turn(i + 1, len(DIALOG_MESSAGES), msg)

        text, inp, out, elapsed = agent.ask(msg)
        cost = calc_cost(model, inp, out)
        result.turns.append(TurnRecord(
            turn=i + 1,
            user_msg=msg[:60],
            assistant_msg=text,
            input_tokens=inp,
            output_tokens=out,
            elapsed_sec=elapsed,
        ))
        if verbose:
            facts_tag = " [ФАКТЫ ИЗВЛЕЧЕНЫ]" if (i + 1) % extract_every == 0 else ""
            _print_turn_stats(text, inp, out, cost, elapsed, tag=facts_tag)

    result.extra_tokens = agent.extra_tokens
    _check_quality(result)
    return result


def run_branching(
    client: Anthropic,
    model: str,
    auto_branch_every: int = 5,
    verbose: bool = True,
) -> StrategyResult:
    agent = BranchingAgent(client=client, model=model, auto_branch_every=auto_branch_every)
    result = StrategyResult(label=f"Branching (every={auto_branch_every})", model=model)

    if verbose:
        _print_header(result.label)

    for i, msg in enumerate(DIALOG_MESSAGES):
        if verbose:
            _print_turn(i + 1, len(DIALOG_MESSAGES), msg)

        text, inp, out, elapsed = agent.ask(msg)
        cost = calc_cost(model, inp, out)
        result.turns.append(TurnRecord(
            turn=i + 1,
            user_msg=msg[:60],
            assistant_msg=text,
            input_tokens=inp,
            output_tokens=out,
            elapsed_sec=elapsed,
        ))
        if verbose:
            branch_tag = f" [ветка: {agent.current_branch}]"
            _print_turn_stats(text, inp, out, cost, elapsed, tag=branch_tag)

    result.extra_tokens = agent.extra_tokens
    _check_quality(result)
    return result


# ══════════════════════════════════════════════════════════════════
# Вспомогательные функции вывода
# ══════════════════════════════════════════════════════════════════

def _print_header(label: str) -> None:
    print(f"\n{'═'*70}")
    print(f"  {label}")
    print(f"{'═'*70}")


def _print_turn(num: int, total: int, msg: str) -> None:
    print(f"\n── Ход {num}/{total} ──")
    print(f"  User: {msg[:70]}{'...' if len(msg) > 70 else ''}")


def _print_turn_stats(text: str, inp: int, out: int, cost: float, elapsed: float, tag: str = "") -> None:
    print(f"  Asst: {text[:80]}{'...' if len(text) > 80 else ''}")
    cost_s = f"${cost:.6f}" if cost < 0.01 else f"${cost:.4f}"
    print(f"  [in={inp:,} out={out:,} | {cost_s} | {elapsed}s{tag}]")


# ══════════════════════════════════════════════════════════════════
# Сравнение результатов
# ══════════════════════════════════════════════════════════════════

def print_comparison(results: list[StrategyResult]) -> None:
    print(f"\n{'═'*70}")
    print("  СРАВНЕНИЕ СТРАТЕГИЙ")
    print(f"{'═'*70}")

    # Итоговая таблица
    print(f"\n  {'Стратегия':<35}  {'Input':>10}  {'Extra':>7}  {'Output':>8}  {'Стоим.':>10}  {'Кач.':>6}")
    print(f"  {'─'*35}  {'─'*10}  {'─'*7}  {'─'*8}  {'─'*10}  {'─'*6}")

    for r in results:
        cost_s = f"${r.total_cost:.4f}" if r.total_cost >= 0.01 else f"${r.total_cost:.6f}"
        print(
            f"  {r.label:<35}  {r.total_input:>10,}  {r.extra_tokens:>7,}  "
            f"{r.total_output:>8,}  {cost_s:>10}  {r.quality_pct:>5.0f}%"
        )

    # Рост input-токенов по ходам — визуализация
    print(f"\n  {'─'*70}")
    print(f"  ДИНАМИКА INPUT-ТОКЕНОВ ПО ХОДАМ")
    print(f"  {'─'*70}")

    max_inp = max(
        max(t.input_tokens for t in r.turns) if r.turns else 1
        for r in results
    )
    bar_width = 35

    symbols = ["▓", "█", "░", "▒"]
    for i in range(len(DIALOG_MESSAGES)):
        print(f"\n  Ход {i+1:>2}:")
        for j, r in enumerate(results):
            if i >= len(r.turns):
                continue
            inp = r.turns[i].input_tokens
            bar_len = int(bar_width * inp / max_inp) if max_inp else 0
            sym = symbols[j % len(symbols)]
            bar = sym * bar_len + "░" * (bar_width - bar_len)
            label_short = r.label[:20]
            print(f"    {label_short:<20} [{bar}] {inp:>7,}")

    # Качество
    print(f"\n  {'─'*70}")
    print(f"  КАЧЕСТВО ОТВЕТОВ")
    print(f"  {'─'*70}")

    header = f"  {'Вопрос':<30}"
    for r in results:
        header += f"  {r.label[:18]:>18}"
    print(header)
    print(f"  {'─'*30}" + "  " + "  ".join("─" * 18 for _ in results))

    all_labels = [c["label"] for c in QUALITY_CHECKS]
    for label in all_labels:
        row = f"  {label:<30}"
        for r in results:
            score = next((s for s in r.quality_scores if s["label"] == label), None)
            if score:
                row += f"  {score['found']}/{score['total']:>17}"
            else:
                row += f"  {'—':>18}"
        print(row)

    # Выводы
    best_tokens = min(results, key=lambda r: r.total_input)
    best_quality = max(results, key=lambda r: r.quality_pct)

    print(f"\n  {'─'*70}")
    print(f"  ВЫВОДЫ")
    print(f"  {'─'*70}")
    print(f"\n  Меньше всего токенов:  {best_tokens.label} ({best_tokens.total_input:,})")
    print(f"  Лучшее качество:       {best_quality.label} ({best_quality.quality_pct:.0f}%)")

    if len(results) >= 2:
        base = results[0].total_input
        print(f"\n  Относительный расход (за базу — первая стратегия):")
        for r in results:
            ratio = r.total_input / base if base else 0
            bar = "█" * int(ratio * 10)
            print(f"    {r.label:<35} ×{ratio:.2f}  {bar}")


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise SystemExit("Missing API key: set ANTHROPIC_API_KEY or API_KEY in .env")

    client = Anthropic(api_key=api_key)
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    print("╔══════════════════════════════════════════════════════╗")
    print("║  День 10. Три стратегии управления контекстом       ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"\nМодель: {model}")
    prices = PRICING.get(model, DEFAULT_PRICING)
    print(f"Цены: ${prices['input']}/1M input, ${prices['output']}/1M output")
    print(f"Диалог: {len(DIALOG_MESSAGES)} ходов\n")

    results: list[StrategyResult] = []

    # 1. Sliding Window
    r1 = run_sliding_window(client, model, window=4)
    results.append(r1)

    # 2. Facts Extraction
    r2 = run_facts_extraction(client, model, extract_every=4, keep_recent=4)
    results.append(r2)

    # 3. Branching
    r3 = run_branching(client, model, auto_branch_every=5)
    results.append(r3)

    # Сравнение
    print_comparison(results)

    print(f"\n{'═'*70}")
    print("  КОГДА ИСПОЛЬЗОВАТЬ КАКУЮ СТРАТЕГИЮ")
    print(f"{'═'*70}")
    print("""
  Sliding Window
    + Просто в реализации, предсказуемый размер контекста
    + Подходит для задач, где важен только недавний контекст
    − Теряет ранние факты: имена, числа, начальные решения
    → Чат-боты, короткие сессии, задачи «здесь и сейчас»

  Facts Extraction
    + Сохраняет ключевые факты в структурированном виде
    + Хорошо работает для задач с множеством деталей (имена, параметры)
    − Требует дополнительных запросов на извлечение
    − Может упустить нюансы и «дух» диалога
    → Технические консультации, проекты с множеством параметров

  Branching
    + Каждая тема имеет полный контекст
    + Переключение между темами без потери деталей
    − Сложнее в реализации, требует семантического разбиения
    − При авто-ветвлении границы тем могут быть неточными
    → Многотемные сессии, исследования, параллельные задачи
""")


if __name__ == "__main__":
    main()

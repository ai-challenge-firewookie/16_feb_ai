"""День 9. Управление контекстом: сжатие истории.

Реализует механизм автоматической компрессии:
  - последние N сообщений хранятся «как есть»
  - старые сообщения сжимаются в summary через LLM
  - summary подставляется в начало контекста вместо полной истории

Сравнивает качество ответов и расход токенов с/без сжатия.
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


# ── Данные ────────────────────────────────────────────────────────

SUMMARY_PROMPT = """\
Ниже приведён фрагмент диалога между пользователем и AI-ассистентом.
Создай краткое, но полное резюме. Сохрани:
- все ключевые факты, имена, числа, технические детали
- решения и договорённости
- предпочтения пользователя
- контекст, необходимый для продолжения разговора

Формат: плотный текст, 3-8 предложений. Без воды.

--- ДИАЛОГ ---
{conversation}
--- КОНЕЦ ДИАЛОГА ---"""


@dataclass(frozen=True)
class CompressedResponse:
    """Ответ агента + метаинформация о состоянии контекста."""
    text: str
    input_tokens: int
    output_tokens: int
    elapsed_sec: float
    # Контекст
    summary_text: str | None       # текущее summary (None если нет)
    recent_count: int              # кол-во «живых» сообщений
    total_raw_messages: int        # сколько всего сообщений прошло через агента
    compressed: bool               # произошло ли сжатие на этом ходе
    compression_tokens: int        # токены, потраченные на сжатие (0 если не было)


class CompressedAgent:
    """Агент с автоматическим сжатием истории.

    Хранит:
      - _summary: текстовое резюме старых сообщений (или None)
      - _recent: последние `keep_recent` сообщений «как есть»

    При отправке в LLM формирует контекст:
      [summary-as-user + ack-as-assistant] + _recent

    Сжатие срабатывает автоматически, когда len(_recent) > keep_recent.
    """

    def __init__(
        self,
        client: Anthropic,
        model: str,
        system: str = "Ты — полезный AI-ассистент. Отвечай кратко, на русском.",
        max_tokens: int = 512,
        keep_recent: int = 10,
    ) -> None:
        self._client = client
        self._model = model
        self._system = system
        self._max_tokens = max_tokens
        self._keep_recent = keep_recent

        self._summary: str | None = None
        self._recent: list[dict[str, str]] = []
        self._total_messages = 0
        self._compression_count = 0

    # ── Основной метод ────────────────────────────────────────────

    def ask(self, user_message: str) -> CompressedResponse:
        """Отправить сообщение, при необходимости сжать историю."""
        self._recent.append({"role": "user", "content": user_message})
        self._total_messages += 1

        # Проверяем, нужно ли сжатие (> keep_recent сообщений в _recent)
        compressed = False
        compression_tokens = 0
        if len(self._recent) > self._keep_recent:
            compression_tokens = self._do_compress()
            compressed = True

        # Собираем контекст для LLM
        messages = self._build_messages()

        t0 = time.perf_counter()
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=self._system,
            messages=messages,
        )
        elapsed = time.perf_counter() - t0

        assistant_text = response.content[0].text
        self._recent.append({"role": "assistant", "content": assistant_text})
        self._total_messages += 1

        return CompressedResponse(
            text=assistant_text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            elapsed_sec=round(elapsed, 2),
            summary_text=self._summary,
            recent_count=len(self._recent),
            total_raw_messages=self._total_messages,
            compressed=compressed,
            compression_tokens=compression_tokens,
        )

    def ask_stream(
        self,
        user_message: str,
        on_token: Callable[[str], None] | None = None,
    ) -> CompressedResponse:
        """Отправить сообщение со стримингом."""
        if on_token is None:
            def on_token(chunk: str) -> None:
                sys.stdout.write(chunk)
                sys.stdout.flush()

        self._recent.append({"role": "user", "content": user_message})
        self._total_messages += 1

        compressed = False
        compression_tokens = 0
        if len(self._recent) > self._keep_recent:
            compression_tokens = self._do_compress()
            compressed = True

        messages = self._build_messages()

        t0 = time.perf_counter()
        full_text = ""

        with self._client.messages.stream(
            model=self._model,
            max_tokens=self._max_tokens,
            system=self._system,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                full_text += text
                on_token(text)
            final = stream.get_final_message()

        elapsed = time.perf_counter() - t0

        self._recent.append({"role": "assistant", "content": full_text})
        self._total_messages += 1

        return CompressedResponse(
            text=full_text,
            input_tokens=final.usage.input_tokens,
            output_tokens=final.usage.output_tokens,
            elapsed_sec=round(elapsed, 2),
            summary_text=self._summary,
            recent_count=len(self._recent),
            total_raw_messages=self._total_messages,
            compressed=compressed,
            compression_tokens=compression_tokens,
        )

    # ── Внутренние методы ─────────────────────────────────────────

    def _do_compress(self) -> int:
        """Сжать старые сообщения из _recent в summary. Возвращает потраченные токены."""
        # Берём старые сообщения (всё кроме последних keep_recent)
        cutoff = len(self._recent) - self._keep_recent
        old_messages = self._recent[:cutoff]
        self._recent = self._recent[cutoff:]

        # Формируем текст для сжатия
        parts = []
        if self._summary:
            parts.append(f"[Предыдущее резюме]: {self._summary}")
        for msg in old_messages:
            role = "Пользователь" if msg["role"] == "user" else "Ассистент"
            parts.append(f"{role}: {msg['content']}")
        conversation_text = "\n\n".join(parts)

        # Запрос к LLM для создания summary
        summary_messages = [
            {"role": "user", "content": SUMMARY_PROMPT.format(conversation=conversation_text)},
        ]

        response = self._client.messages.create(
            model=self._model,
            max_tokens=512,
            messages=summary_messages,
        )

        self._summary = response.content[0].text.strip()
        self._compression_count += 1

        return response.usage.input_tokens + response.usage.output_tokens

    def _build_messages(self) -> list[dict[str, str]]:
        """Собрать messages для API: [summary-пара] + recent."""
        messages: list[dict[str, str]] = []

        if self._summary:
            messages.append({
                "role": "user",
                "content": f"[Резюме предыдущего разговора]:\n{self._summary}",
            })
            messages.append({
                "role": "assistant",
                "content": "Понял, я помню контекст. Продолжаем.",
            })

        messages.extend(self._recent)
        return messages

    def reset(self) -> None:
        self._summary = None
        self._recent.clear()
        self._total_messages = 0
        self._compression_count = 0

    @property
    def summary(self) -> str | None:
        return self._summary

    @property
    def recent_messages(self) -> list[dict[str, str]]:
        return list(self._recent)

    @property
    def compression_count(self) -> int:
        return self._compression_count


# ── Простой агент без сжатия (для сравнения) ──────────────────────

class PlainAgent:
    """Агент без сжатия — хранит всю историю."""

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

    def ask(self, user_message: str) -> tuple[str, int, int, float]:
        """Возвращает (text, input_tokens, output_tokens, elapsed)."""
        self._history.append({"role": "user", "content": user_message})

        t0 = time.perf_counter()
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=self._system,
            messages=self._history,
        )
        elapsed = time.perf_counter() - t0

        text = response.content[0].text
        self._history.append({"role": "assistant", "content": text})

        return text, response.usage.input_tokens, response.usage.output_tokens, round(elapsed, 2)


# ── Сценарий для сравнения ────────────────────────────────────────

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
    # 9-10: Новая тема, но со ссылкой на контекст
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

# Контрольные вопросы и эталонные ответы для оценки качества
QUALITY_CHECKS = [
    {
        "index": 10,  # "Напомни, как зовут меня и мой проект?"
        "keywords": ["Никита", "DataPipeline"],
        "label": "Имя + название проекта",
    },
    {
        "index": 11,  # "Какие библиотеки мы обсуждали?"
        "keywords": ["pandas", "sqlalchemy", "alembic"],
        "label": "Библиотеки из начала",
    },
    {
        "index": 14,  # "Какую целевую скорость?"
        "keywords": ["10"],
        "label": "Целевая скорость (10 сек)",
    },
    {
        "index": 15,  # "Подведи итог: стек"
        "keywords": ["PostgreSQL", "pandas"],
        "label": "Стек технологий",
    },
]


@dataclass
class RunResult:
    """Результат прогона одного агента через сценарий."""
    label: str
    model: str
    turns: list[dict] = field(default_factory=list)
    quality_scores: list[dict] = field(default_factory=list)
    compressions: int = 0

    @property
    def total_input(self) -> int:
        return sum(t["input_tokens"] for t in self.turns)

    @property
    def total_output(self) -> int:
        return sum(t["output_tokens"] for t in self.turns)

    @property
    def total_cost(self) -> float:
        return sum(t["cost"] for t in self.turns)

    @property
    def quality_pct(self) -> float:
        if not self.quality_scores:
            return 0.0
        return sum(s["found"] for s in self.quality_scores) / sum(s["total"] for s in self.quality_scores) * 100


def run_plain(client: Anthropic, model: str, verbose: bool = True) -> RunResult:
    """Прогнать диалог через обычного агента (без сжатия)."""
    agent = PlainAgent(client=client, model=model)
    result = RunResult(label="Без сжатия (полная история)", model=model)

    if verbose:
        print(f"\n{'='*70}")
        print(f"  Без сжатия — полная история")
        print(f"{'='*70}")

    for i, msg in enumerate(DIALOG_MESSAGES):
        if verbose:
            print(f"\n── Ход {i+1}/{len(DIALOG_MESSAGES)} ──")
            print(f"  User: {msg[:70]}...")

        text, inp, out, elapsed = agent.ask(msg)
        cost = calc_cost(model, inp, out)
        result.turns.append({
            "turn": i + 1,
            "input_tokens": inp,
            "output_tokens": out,
            "cost": cost,
            "elapsed": elapsed,
            "response": text,
        })

        if verbose:
            print(f"  Assistant: {text[:80]}...")
            print(f"  [in={inp:,} out={out:,} | ${cost:.6f} | {elapsed}s]")

    # Проверка качества
    _check_quality(result)
    return result


def run_compressed(
    client: Anthropic,
    model: str,
    keep_recent: int = 10,
    verbose: bool = True,
) -> RunResult:
    """Прогнать диалог через агента со сжатием."""
    agent = CompressedAgent(client=client, model=model, keep_recent=keep_recent)
    result = RunResult(
        label=f"Со сжатием (keep={keep_recent})",
        model=model,
    )

    if verbose:
        print(f"\n{'='*70}")
        print(f"  Со сжатием — keep_recent={keep_recent}")
        print(f"{'='*70}")

    for i, msg in enumerate(DIALOG_MESSAGES):
        if verbose:
            print(f"\n── Ход {i+1}/{len(DIALOG_MESSAGES)} ──")
            print(f"  User: {msg[:70]}...")

        resp = agent.ask(msg)
        cost = calc_cost(model, resp.input_tokens, resp.output_tokens)

        # Учитываем токены, потраченные на сжатие
        compression_cost = calc_cost(model, resp.compression_tokens, 0) if resp.compression_tokens else 0

        result.turns.append({
            "turn": i + 1,
            "input_tokens": resp.input_tokens + resp.compression_tokens,
            "output_tokens": resp.output_tokens,
            "cost": cost + compression_cost,
            "elapsed": resp.elapsed_sec,
            "response": resp.text,
        })

        if verbose:
            compress_tag = " [СЖАТИЕ]" if resp.compressed else ""
            print(f"  Assistant: {resp.text[:80]}...")
            print(f"  [in={resp.input_tokens:,} out={resp.output_tokens:,} | "
                  f"${cost:.6f} | {resp.elapsed_sec}s | "
                  f"recent={resp.recent_count}{compress_tag}]")
            if resp.compressed:
                summary_preview = (resp.summary_text or "")[:100]
                print(f"  Summary: {summary_preview}...")

    result.compressions = agent.compression_count
    _check_quality(result)
    return result


def _check_quality(result: RunResult) -> None:
    """Проверить качество ответов на контрольных вопросах."""
    for check in QUALITY_CHECKS:
        idx = check["index"]
        if idx >= len(result.turns):
            continue
        response_text = result.turns[idx]["response"].lower()
        found = sum(1 for kw in check["keywords"] if kw.lower() in response_text)
        result.quality_scores.append({
            "label": check["label"],
            "found": found,
            "total": len(check["keywords"]),
            "keywords": check["keywords"],
        })


def print_comparison(plain: RunResult, compressed: RunResult) -> None:
    """Вывести сравнение двух прогонов."""
    print(f"\n{'='*70}")
    print("  СРАВНЕНИЕ: без сжатия vs со сжатием")
    print(f"{'='*70}")

    # Таблица токенов по ходам
    print(f"\n  {'Ход':>4}  {'Input (полн)':>14}  {'Input (сжат)':>14}  {'Экономия':>10}")
    print(f"  {'─'*4}  {'─'*14}  {'─'*14}  {'─'*10}")

    for i in range(len(plain.turns)):
        p = plain.turns[i]
        c = compressed.turns[i]
        saving = p["input_tokens"] - c["input_tokens"]
        pct = (saving / p["input_tokens"] * 100) if p["input_tokens"] > 0 else 0
        sign = "+" if saving < 0 else ""
        print(f"  {i+1:>4}  {p['input_tokens']:>14,}  {c['input_tokens']:>14,}  "
              f"{sign}{saving:>7,} ({pct:+.0f}%)")

    # Итоги
    print(f"\n  {'Метрика':<30}  {'Без сжатия':>14}  {'Со сжатием':>14}  {'Разница':>14}")
    print(f"  {'─'*30}  {'─'*14}  {'─'*14}  {'─'*14}")

    def row(label: str, v1: str, v2: str, diff: str) -> None:
        print(f"  {label:<30}  {v1:>14}  {v2:>14}  {diff:>14}")

    saving_inp = plain.total_input - compressed.total_input
    saving_pct = (saving_inp / plain.total_input * 100) if plain.total_input else 0
    row("Суммарно input токенов",
        f"{plain.total_input:,}", f"{compressed.total_input:,}",
        f"-{saving_inp:,} ({saving_pct:.0f}%)")

    row("Суммарно output токенов",
        f"{plain.total_output:,}", f"{compressed.total_output:,}",
        f"{compressed.total_output - plain.total_output:+,}")

    cost_p = f"${plain.total_cost:.4f}" if plain.total_cost >= 0.01 else f"${plain.total_cost:.6f}"
    cost_c = f"${compressed.total_cost:.4f}" if compressed.total_cost >= 0.01 else f"${compressed.total_cost:.6f}"
    cost_diff = plain.total_cost - compressed.total_cost
    cost_d = f"-${cost_diff:.4f}" if cost_diff >= 0.01 else f"-${cost_diff:.6f}"
    row("Суммарная стоимость", cost_p, cost_c, cost_d)

    row("Количество сжатий", "0", str(compressed.compressions), "")

    # Качество
    print(f"\n  {'─'*70}")
    print(f"  КАЧЕСТВО ОТВЕТОВ (контрольные вопросы)")
    print(f"  {'─'*70}")

    print(f"\n  {'Вопрос':<30}  {'Без сжатия':>14}  {'Со сжатием':>14}")
    print(f"  {'─'*30}  {'─'*14}  {'─'*14}")

    for sp, sc in zip(plain.quality_scores, compressed.quality_scores):
        p_str = f"{sp['found']}/{sp['total']}"
        c_str = f"{sc['found']}/{sc['total']}"
        print(f"  {sp['label']:<30}  {p_str:>14}  {c_str:>14}")

    print(f"\n  Итого качество:")
    print(f"    Без сжатия: {plain.quality_pct:.0f}%")
    print(f"    Со сжатием: {compressed.quality_pct:.0f}%")

    # Рост input-токенов — график
    print(f"\n  {'─'*70}")
    print(f"  РОСТ INPUT-ТОКЕНОВ ПО ХОДАМ")
    print(f"  {'─'*70}")
    max_inp = max(
        max(t["input_tokens"] for t in plain.turns),
        max(t["input_tokens"] for t in compressed.turns),
    )
    bar_max = 40
    for i in range(len(plain.turns)):
        p_inp = plain.turns[i]["input_tokens"]
        c_inp = compressed.turns[i]["input_tokens"]
        p_bar = int(bar_max * p_inp / max_inp) if max_inp else 0
        c_bar = int(bar_max * c_inp / max_inp) if max_inp else 0
        print(f"  {i+1:>2} полн: {'▓' * p_bar}{'░' * (bar_max - p_bar)} {p_inp:>7,}")
        print(f"     сжат: {'█' * c_bar}{'░' * (bar_max - c_bar)} {c_inp:>7,}")


# ── CLI ───────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise SystemExit("Missing API key: set ANTHROPIC_API_KEY or API_KEY in .env")

    client = Anthropic(api_key=api_key)
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    print("╔══════════════════════════════════════════════════════╗")
    print("║  День 9. Управление контекстом: сжатие истории      ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"\nМодель: {model}")
    prices = PRICING.get(model, DEFAULT_PRICING)
    print(f"Цены: ${prices['input']}/1M input, ${prices['output']}/1M output")
    print(f"Диалог: {len(DIALOG_MESSAGES)} ходов с контрольными вопросами")
    print()

    # 1. Прогон без сжатия
    plain = run_plain(client, model)

    # 2. Прогон со сжатием (keep_recent=10)
    compressed = run_compressed(client, model, keep_recent=10)

    # 3. Сравнение
    print_comparison(plain, compressed)

    # Выводы
    print(f"\n{'='*70}")
    print("  ВЫВОДЫ")
    print(f"{'='*70}")
    print(f"""
  1. Сжатие экономит input-токены на длинных диалогах.
     Старые сообщения заменяются коротким summary → контекст не растёт бесконечно.

  2. Качество ответов: summary сохраняет ключевые факты (имена, числа, решения).
     Модель «помнит» ранний контекст через резюме, а не дословную историю.

  3. Компромисс: сжатие тратит дополнительные токены на создание summary,
     но окупается на длинных диалогах (>10 ходов).

  4. keep_recent — главный параметр:
     - больше → лучше качество, но меньше экономия
     - меньше → больше экономия, но риск потери деталей

  5. В продакшене: комбинировать с Anthropic token counting API
     для точного контроля размера контекста.
""")


def run_interactive(client: Anthropic, model: str) -> None:
    """Интерактивный чат с CompressedAgent."""
    agent = CompressedAgent(
        client=client,
        model=model,
        keep_recent=10,
    )

    print(f"\nИнтерактивный чат со сжатием (keep_recent=10)")
    print("Команды: /summary — показать summary, /info — статистика, /exit — выход\n")

    turn = 0
    total_in = 0
    total_out = 0

    while True:
        try:
            user_text = input("Вы> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text or user_text == "/exit":
            break

        if user_text == "/summary":
            if agent.summary:
                print(f"\n  Summary: {agent.summary}\n")
            else:
                print("  Сжатие ещё не происходило.\n")
            continue

        if user_text == "/info":
            print(f"  Ходов: {turn}")
            print(f"  Сжатий: {agent.compression_count}")
            print(f"  «Живых» сообщений: {len(agent.recent_messages)}")
            print(f"  Суммарно input: {total_in:,}")
            print(f"  Суммарно output: {total_out:,}")
            cost = calc_cost(model, total_in, total_out)
            print(f"  Стоимость: ${cost:.6f}")
            if agent.summary:
                print(f"  Summary: {agent.summary[:100]}...")
            print()
            continue

        sys.stdout.write("\nАгент> ")
        sys.stdout.flush()
        resp = agent.ask_stream(user_text)
        turn += 1
        total_in += resp.input_tokens
        total_out += resp.output_tokens
        cost = calc_cost(model, resp.input_tokens, resp.output_tokens)

        compress_tag = " | СЖАТИЕ" if resp.compressed else ""
        print(f"\n  [in={resp.input_tokens:,} out={resp.output_tokens:,} | "
              f"${cost:.6f} | recent={resp.recent_count}{compress_tag}]\n")


if __name__ == "__main__":
    main()

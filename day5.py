"""День 5. Сравнение версий моделей (слабая / средняя / сильная).

Один и тот же запрос отправляется к трём моделям Anthropic:
  - Haiku   (быстрая, дешёвая)
  - Sonnet  (баланс)
  - Opus    (максимальное качество)

Замеряются: время ответа, количество токенов, стоимость.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field

from anthropic import Anthropic
from dotenv import load_dotenv


# ── Модели (слабая → средняя → сильная) ─────────────────────────
MODELS: list[dict[str, str | float]] = [
    {
        "id": "claude-haiku-4-5-20251001",
        "label": "Haiku 4.5 (слабая)",
        "input_price": 0.80,   # $ за 1M input tokens
        "output_price": 4.00,  # $ за 1M output tokens
    },
    {
        "id": "claude-sonnet-4-6",
        "label": "Sonnet 4.6 (средняя)",
        "input_price": 3.00,
        "output_price": 15.00,
    },
    {
        "id": "claude-opus-4-6",
        "label": "Opus 4.6 (сильная)",
        "input_price": 15.00,
        "output_price": 75.00,
    },
]

# ── Промпты для тестирования ─────────────────────────────────────
PROMPTS = [
    {
        "label": "Креативный",
        "text": (
            "Придумай короткую историю (3-4 предложения) "
            "про кота, который научился программировать."
        ),
        "max_tokens": 300,
    },
    {
        "label": "Аналитический",
        "text": (
            "Сравни плюсы и минусы Python и Rust для backend-разработки. "
            "Ответь кратко, 5-7 пунктов."
        ),
        "max_tokens": 500,
    },
    {
        "label": "Код",
        "text": (
            "Напиши функцию на Python, которая принимает список чисел "
            "и возвращает второй по величине элемент. Добавь docstring."
        ),
        "max_tokens": 400,
    },
]


@dataclass
class ModelResult:
    model_id: str
    model_label: str
    prompt_label: str
    answer: str
    elapsed_sec: float
    input_tokens: int
    output_tokens: int
    cost_usd: float


class Day5ModelComparison:
    def __init__(self, client: Anthropic) -> None:
        self._client = client

    def run(self) -> list[ModelResult]:
        results: list[ModelResult] = []
        total = len(MODELS) * len(PROMPTS)
        done = 0

        for model_cfg in MODELS:
            for prompt_cfg in PROMPTS:
                done += 1
                label = f"[{done}/{total}] {model_cfg['label']} — {prompt_cfg['label']}"
                print(f"  {label} ...", end="", flush=True)

                result = self._request(model_cfg, prompt_cfg)
                results.append(result)

                print(
                    f"  {result.elapsed_sec:.1f}s  "
                    f"in={result.input_tokens} out={result.output_tokens}  "
                    f"${result.cost_usd:.5f}"
                )

        return results

    def _request(
        self,
        model_cfg: dict,
        prompt_cfg: dict,
        *,
        max_retries: int = 5,
        base_delay: float = 1.0,
    ) -> ModelResult:
        attempt = 0
        while True:
            try:
                t0 = time.perf_counter()
                message = self._client.messages.create(
                    model=model_cfg["id"],
                    max_tokens=prompt_cfg["max_tokens"],
                    messages=[{"role": "user", "content": prompt_cfg["text"]}],
                )
                elapsed = time.perf_counter() - t0

                input_tokens = message.usage.input_tokens
                output_tokens = message.usage.output_tokens
                cost = (
                    input_tokens * model_cfg["input_price"] / 1_000_000
                    + output_tokens * model_cfg["output_price"] / 1_000_000
                )

                return ModelResult(
                    model_id=model_cfg["id"],
                    model_label=model_cfg["label"],
                    prompt_label=prompt_cfg["label"],
                    answer=message.content[0].text.strip(),
                    elapsed_sec=round(elapsed, 2),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=round(cost, 6),
                )
            except Exception as exc:
                attempt += 1
                status_code = getattr(exc, "status_code", None)
                is_overloaded = status_code in (429, 529) or "overloaded" in str(exc).lower()
                if not is_overloaded or attempt > max_retries:
                    raise
                delay = base_delay * (2 ** (attempt - 1))
                jitter = random.uniform(0, 0.3)
                print(f" (retry {attempt})", end="", flush=True)
                time.sleep(delay + jitter)


def print_report(results: list[ModelResult]) -> None:
    print("\n" + "=" * 70)
    print("  День 5. Сравнение моделей: Haiku vs Sonnet vs Opus")
    print("=" * 70)

    # Группируем по промпту
    prompt_labels = list(dict.fromkeys(r.prompt_label for r in results))

    for plabel in prompt_labels:
        print(f"\n{'━' * 70}")
        print(f"  Запрос: {plabel}")
        print(f"{'━' * 70}")

        group = [r for r in results if r.prompt_label == plabel]
        for r in group:
            print(f"\n  ┌─ {r.model_label}")
            print(f"  │  Время: {r.elapsed_sec:.1f}s  |  "
                  f"Токены: {r.input_tokens} in / {r.output_tokens} out  |  "
                  f"Стоимость: ${r.cost_usd:.5f}")
            print(f"  │")
            for line in r.answer.splitlines():
                print(f"  │  {line}")
            print(f"  └{'─' * 50}")

    # Сводная таблица
    print(f"\n{'=' * 70}")
    print("  СВОДНАЯ ТАБЛИЦА")
    print(f"{'=' * 70}")
    print(f"  {'Модель':<28} {'Время(с)':>9} {'In tok':>8} {'Out tok':>9} {'Цена($)':>10}")
    print(f"  {'─' * 66}")

    # Средние по модели
    model_labels = list(dict.fromkeys(r.model_label for r in results))
    for mlabel in model_labels:
        group = [r for r in results if r.model_label == mlabel]
        avg_time = sum(r.elapsed_sec for r in group) / len(group)
        total_in = sum(r.input_tokens for r in group)
        total_out = sum(r.output_tokens for r in group)
        total_cost = sum(r.cost_usd for r in group)
        print(f"  {mlabel:<28} {avg_time:>8.1f}s {total_in:>8} {total_out:>9} {total_cost:>10.5f}")

    # Выводы
    print(f"\n{'=' * 70}")
    print("  ВЫВОДЫ")
    print(f"{'=' * 70}")
    print("""
  Haiku 4.5 (слабая модель)
    ✦ Скорость: самая быстрая — ответ за ~1-3 секунды
    ✦ Стоимость: самая дешёвая (~$0.80/$4.00 за 1M токенов)
    ✦ Качество: достаточное для простых задач, может упускать нюансы
    → Идеально для: классификации, простых ответов, высоконагруженных
      систем, прототипирования, предварительной фильтрации

  Sonnet 4.6 (средняя модель)
    ✦ Скорость: умеренная — ответ за ~3-8 секунд
    ✦ Стоимость: средняя (~$3.00/$15.00 за 1M токенов)
    ✦ Качество: хороший баланс между скоростью и глубиной
    → Идеально для: чат-ботов, генерации контента, анализа данных,
      повседневных задач разработки

  Opus 4.6 (сильная модель)
    ✦ Скорость: самая медленная — ответ за ~10-30 секунд
    ✦ Стоимость: самая дорогая (~$15.00/$75.00 за 1M токенов)
    ✦ Качество: максимальное — глубокий анализ, лучший код, нюансы
    → Идеально для: сложных задач, исследований, код-ревью,
      архитектурных решений, критически важных ответов

  Соотношение цен (input/output):
    Haiku → Sonnet:  ×3.75
    Sonnet → Opus:   ×5.0
    Haiku → Opus:    ×18.75
""")


def _run_cli() -> None:
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise SystemExit("Missing API key: set ANTHROPIC_API_KEY or API_KEY in .env")

    client = Anthropic(api_key=api_key)

    print("День 5. Сравнение моделей Anthropic")
    print(f"Модели: {', '.join(m['label'] for m in MODELS)}")
    print(f"Запросов: {len(PROMPTS)} × {len(MODELS)} = {len(PROMPTS) * len(MODELS)}")
    print()

    demo = Day5ModelComparison(client=client)
    results = demo.run()
    print_report(results)


if __name__ == "__main__":
    _run_cli()

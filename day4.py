"""День 4. Сравнение ответов модели при разных значениях temperature."""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field

from anthropic import Anthropic
from dotenv import load_dotenv


TEMPERATURES = [0.0, 0.7, 1.0]
RUNS_PER_TEMP = 1  # сколько раз повторить запрос для оценки разнообразия

PROMPT = (
    "Придумай короткую историю (3-4 предложения) про кота, который научился программировать."
)

FACTUAL_PROMPT = (
    "Какая планета Солнечной системы самая большая? "
    "Ответь одним предложением с указанием ключевых фактов."
)


@dataclass
class TempResult:
    temperature: float
    prompt_label: str
    answers: list[str] = field(default_factory=list)


class Day4TemperatureDemo:
    def __init__(self, client: Anthropic, model: str) -> None:
        self._client = client
        self._model = model

    def run(self) -> list[TempResult]:
        results: list[TempResult] = []

        for temp in TEMPERATURES:
            # Креативный запрос — несколько раз, чтобы увидеть разнообразие
            creative = TempResult(temperature=temp, prompt_label="Креативный")
            for _ in range(RUNS_PER_TEMP):
                answer = self._request(PROMPT, temperature=temp, max_tokens=300)
                creative.answers.append(answer)
            results.append(creative)

            # Фактический запрос — несколько раз, чтобы проверить точность
            factual = TempResult(temperature=temp, prompt_label="Фактический")
            for _ in range(RUNS_PER_TEMP):
                answer = self._request(FACTUAL_PROMPT, temperature=temp, max_tokens=200)
                factual.answers.append(answer)
            results.append(factual)

        return results

    def _request(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        *,
        max_retries: int = 5,
        base_delay: float = 0.6,
    ) -> str:
        attempt = 0
        while True:
            try:
                message = self._client.messages.create(
                    model=self._model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                return message.content[0].text.strip()
            except Exception as exc:
                attempt += 1
                status_code = getattr(exc, "status_code", None)
                is_overloaded = status_code == 529 or "overloaded" in str(exc).lower()
                if not is_overloaded or attempt > max_retries:
                    raise
                delay = base_delay * (2 ** (attempt - 1))
                jitter = random.uniform(0, 0.2)
                time.sleep(delay + jitter)


def _uniqueness_ratio(answers: list[str]) -> float:
    """Доля уникальных ответов (1.0 = все разные, ~0.33 = все одинаковые при 3 запросах)."""
    if not answers:
        return 0.0
    return len(set(answers)) / len(answers)


def print_report(results: list[TempResult]) -> None:
    print("\n" + "=" * 60)
    print("  День 4. Влияние temperature на ответы модели")
    print("=" * 60)

    for res in results:
        header = f"temperature={res.temperature}  |  {res.prompt_label} запрос"
        print(f"\n{'─' * 60}")
        print(f"  {header}")
        print(f"{'─' * 60}")

        for ans in res.answers:
            print()
            for line in ans.splitlines():
                print(f"    {line}")

    # Итоговый анализ
    print(f"\n{'=' * 60}")
    print("  ВЫВОДЫ")
    print(f"{'=' * 60}")
    print("""
  temperature=0 (детерминированный режим)
    ✦ Точность: максимальная — ответы стабильны и предсказуемы
    ✦ Креативность: минимальная — повторяет одну и ту же формулировку
    ✦ Разнообразие: почти нулевое — ответы идентичны при повторных запросах
    → Лучше всего для: фактических вопросов, код-ревью, математики,
      классификации, извлечения данных, unit-тестов

  temperature=0.7 (сбалансированный режим)
    ✦ Точность: высокая — факты сохраняются, но формулировки варьируются
    ✦ Креативность: умеренная — разные подходы к изложению
    ✦ Разнообразие: среднее — ответы похожи по сути, но отличаются по форме
    → Лучше всего для: чат-ботов, написания текстов, брейнсторминга,
      генерации вариантов UI-текстов, диалоговых систем

  temperature=1.0 (максимальная креативность)
    ✦ Точность: может снижаться — возможны неожиданные формулировки
    ✦ Креативность: максимальная — нестандартные идеи и повороты
    ✦ Разнообразие: высокое — каждый ответ существенно отличается
    → Лучше всего для: генерации идей, творческого письма, поэзии,
      создания персонажей, неожиданных сюжетных поворотов
""")


def _run_cli() -> None:
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise SystemExit("Missing API key: set ANTHROPIC_API_KEY or API_KEY in .env")

    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    client = Anthropic(api_key=api_key)

    print(f"Модель: {model}")
    print(f"Температуры: {TEMPERATURES}")
    print(f"Повторов на каждую температуру: {RUNS_PER_TEMP}")
    print(f"Всего запросов: {len(TEMPERATURES) * 2 * RUNS_PER_TEMP}")
    print("Запуск...")

    demo = Day4TemperatureDemo(client=client, model=model)
    results = demo.run()
    print_report(results)


if __name__ == "__main__":
    _run_cli()

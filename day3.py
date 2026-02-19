from __future__ import annotations

import os
import random
import re
import time
from dataclasses import dataclass

from anthropic import Anthropic
from dotenv import load_dotenv


TASK_TEXT = (
    "Логическая задача: есть три человека A, B, C. "
    "Рыцарь всегда говорит правду, лжец всегда лжет.\n"
    "A говорит: 'B лжец'.\n"
    "B говорит: 'A и C одного типа'.\n"
    "C говорит: 'A рыцарь'.\n"
    "Кто рыцарь, а кто лжец?"
)

EXPECTED_CANONICAL = "A и C — лжецы, B — рыцарь"


@dataclass(frozen=True)
class MethodResult:
    name: str
    prompt: str
    answer: str
    is_correct: bool


class Day3ReasoningDemo:
    def __init__(self, client, model: str) -> None:
        self._client = client
        self._model = model

    def run(self) -> list[MethodResult]:
        direct_prompt = (
            f"Реши задачу и дай финальный ответ коротко.\n\n{TASK_TEXT}"
        )

        step_by_step_prompt = (
            "Решай пошагово. Обязательно покажи рассуждение и затем финальный ответ.\n\n"
            f"{TASK_TEXT}"
        )

        generated_prompt_request = (
            "Составь лучший промпт для LLM, чтобы точно решить задачу ниже. "
            "Промпт должен требовать проверку всех вариантов и финальный ответ отдельной строкой."
            " Верни только текст промпта.\n\n"
            f"{TASK_TEXT}"
        )
        generated_prompt = self._request(generated_prompt_request, max_tokens=400)
        generated_prompt_solution = self._request(generated_prompt, max_tokens=700)

        experts_prompt = (
            "Решите задачу группой экспертов в формате:\n"
            "1) Аналитик\n"
            "2) Инженер\n"
            "3) Критик\n"
            "4) Общий вывод\n"
            "Каждый эксперт дает свое решение и проверку.\n\n"
            f"{TASK_TEXT}"
        )

        results = [
            MethodResult(
                name="1) Прямой ответ",
                prompt=direct_prompt,
                answer=self._request(direct_prompt, max_tokens=300),
                is_correct=False,
            ),
            MethodResult(
                name="2) Пошагово",
                prompt=step_by_step_prompt,
                answer=self._request(step_by_step_prompt, max_tokens=700),
                is_correct=False,
            ),
            MethodResult(
                name="3) Сначала сгенерировать промпт",
                prompt=generated_prompt,
                answer=generated_prompt_solution,
                is_correct=False,
            ),
            MethodResult(
                name="4) Группа экспертов",
                prompt=experts_prompt,
                answer=self._request(experts_prompt, max_tokens=1000),
                is_correct=False,
            ),
        ]

        evaluated: list[MethodResult] = []
        for item in results:
            evaluated.append(
                MethodResult(
                    name=item.name,
                    prompt=item.prompt,
                    answer=item.answer,
                    is_correct=self._is_correct(item.answer),
                )
            )
        return evaluated

    def _is_correct(self, text: str) -> bool:
        normalized = re.sub(r"\s+", " ", text.lower())
        has_a_liar = bool(re.search(r"a[^a-zа-я0-9]{0,12}лже", normalized))
        has_c_liar = bool(re.search(r"c[^a-zа-я0-9]{0,12}лже", normalized))
        has_b_knight = bool(re.search(r"b[^a-zа-я0-9]{0,12}рыцар", normalized))
        return has_a_liar and has_c_liar and has_b_knight

    def _request(
        self,
        prompt: str,
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
                    messages=[{"role": "user", "content": prompt}],
                )
                return message.content[0].text.strip()
            except Exception as exc:  # pragma: no cover - depends on API behavior
                attempt += 1
                status_code = getattr(exc, "status_code", None)
                is_overloaded = status_code == 529 or "overloaded" in str(exc).lower()
                if not is_overloaded or attempt > max_retries:
                    raise
                delay = base_delay * (2 ** (attempt - 1))
                jitter = random.uniform(0, 0.2)
                time.sleep(delay + jitter)


def print_report(results: list[MethodResult]) -> None:
    print("\n=== День 3. Разные способы рассуждения ===")
    print(f"Задача: {TASK_TEXT}\n")
    print(f"Эталонный ответ: {EXPECTED_CANONICAL}\n")

    for item in results:
        print(f"--- {item.name} ---")
        print("Промпт:")
        print(item.prompt)
        print("\nОтвет модели:")
        print(item.answer)
        print(f"\nТочность: {'Верно' if item.is_correct else 'Неверно'}")
        print()

    total_correct = sum(1 for item in results if item.is_correct)
    print("=== Сравнение ===")
    print(f"Совпало с эталоном: {total_correct}/4")

    if total_correct == 4:
        print("Ответы отличаются по форме, но все методы дали корректный результат.")
    elif total_correct == 0:
        print("Ни один метод не дал корректный ответ; попробуйте другую модель.")
    else:
        best = [item.name for item in results if item.is_correct]
        print("Наиболее точные способы в этом запуске:")
        for name in best:
            print(f"- {name}")


def _run_cli() -> None:
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise SystemExit("Missing API key: set ANTHROPIC_API_KEY or API_KEY in .env")

    model = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-6")
    client = Anthropic(api_key=api_key)
    demo = Day3ReasoningDemo(client=client, model=model)
    results = demo.run()
    print_report(results)


if __name__ == "__main__":
    _run_cli()

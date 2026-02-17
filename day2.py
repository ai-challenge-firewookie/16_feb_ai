from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass

from anthropic import Anthropic
from dotenv import load_dotenv


@dataclass(frozen=True)
class ResponsePair:
    unconstrained: str
    constrained: str


class Day2PromptBuilder:
    def __init__(self, base_prompt: str) -> None:
        self._base_prompt = base_prompt.strip()

    def build_unconstrained(self) -> str:
        return self._base_prompt

    def build_constrained(self) -> str:
        return (
            f"{self._base_prompt}\n\n"
            "Формат ответа:\n"
            "1) Короткое резюме (1–2 предложения)\n"
            "2) Три маркированных пункта с фактами\n"
            "3) Одна строка с выводом\n"
            "Ограничение длины: не более 90 слов.\n"
            "Условие завершения: после вывода напишите строку /end и остановитесь."
        )


class Day2Demo:
    def __init__(self, client, model: str) -> None:
        self._client = client
        self._model = model

    def run(self, base_prompt: str) -> ResponsePair:
        builder = Day2PromptBuilder(base_prompt)
        unconstrained = self._request(
            prompt=builder.build_unconstrained(),
            max_tokens=512,
        )
        constrained = self._request(
            prompt=builder.build_constrained(),
            max_tokens=160,
            stop_sequences=["/end"]
        )
        return ResponsePair(unconstrained=unconstrained, constrained=constrained)

    def _request(
        self,
        prompt: str,
        max_tokens: int,
        stop_sequences: list[str] | None = None,
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
                    stop_sequences=stop_sequences,
                    messages=[{"role": "user", "content": prompt}],
                )
                return message.content[0].text
            except Exception as exc:  # pragma: no cover - depends on API behavior
                attempt += 1
                status_code = getattr(exc, "status_code", None)
                is_overloaded = status_code == 529 or "overloaded" in str(exc).lower()
                if not is_overloaded or attempt > max_retries:
                    raise
                delay = base_delay * (2 ** (attempt - 1))
                jitter = random.uniform(0, 0.2)
                time.sleep(delay + jitter)


def _run_cli() -> None:
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise SystemExit("Missing API key: set ANTHROPIC_API_KEY or API_KEY in .env")

    base_prompt = input("Базовый запрос для сравнения: ").strip()
    if not base_prompt:
        raise SystemExit("Пустой запрос.")

    client = Anthropic(api_key=api_key)
    demo = Day2Demo(client=client, model="claude-opus-4-5-20251101")
    pair = demo.run(base_prompt=base_prompt)

    print("\nБез ограничений:\n")
    print(pair.unconstrained)
    print("\nС ограничениями:\n")
    print(pair.constrained)


if __name__ == "__main__":
    _run_cli()

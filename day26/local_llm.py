"""
День 26. Запуск локальной LLM

Скрипт проверяет:
1. Что локальный Ollama API доступен.
2. Что модель запускается локально.
3. Что к модели можно обратиться через CLI и HTTP API.
4. Что модель отвечает минимум на 3 запроса разной сложности.
"""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass

OLLAMA_URL = "http://127.0.0.1:11434"
MODEL = "qwen2.5:0.5b"


@dataclass
class PromptResult:
    title: str
    prompt: str
    response: str
    channel: str


PROMPTS = [
    (
        "Простой запрос",
        "Ответь одним словом: как называется столица Франции?",
    ),
    (
        "Логический запрос",
        "Если у меня было 12 яблок, я отдал 5 и купил еще 3, сколько стало? Ответь кратко.",
    ),
    (
        "Структурированный запрос",
        'Сформируй JSON с полями "language" и "use_case" для Python. Ответ только в JSON.',
    ),
]


def check_ollama_server() -> None:
    """Проверяем, что локальный сервер Ollama доступен."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as response:
            if response.status != 200:
                raise RuntimeError(f"Ollama API returned status {response.status}")
    except urllib.error.URLError as exc:
        raise SystemExit(
            "Ollama API недоступен. Запустите `ollama serve` и повторите попытку."
        ) from exc


def call_cli(prompt: str) -> str:
    """Вызываем модель через CLI."""
    completed = subprocess.run(
        ["ollama", "run", MODEL, prompt],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def call_http(prompt: str) -> str:
    """Вызываем модель через локальный HTTP API Ollama."""
    payload = json.dumps(
        {
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        data = json.loads(response.read().decode("utf-8"))
    return str(data["response"]).strip()


def run_demo() -> list[PromptResult]:
    """Запускаем набор проверок для Day 26."""
    check_ollama_server()

    results: list[PromptResult] = []
    for index, (title, prompt) in enumerate(PROMPTS):
        if index < 2:
            answer = call_cli(prompt)
            channel = "CLI"
        else:
            answer = call_http(prompt)
            channel = "HTTP API"
        results.append(
            PromptResult(
                title=title,
                prompt=prompt,
                response=answer,
                channel=channel,
            )
        )
    return results


def main() -> int:
    results = run_demo()

    print(f"Локальная LLM: {MODEL}")
    print(f"Ollama API: {OLLAMA_URL}")
    print()
    print("Проверка пройдена:")
    print("- модель запускается локально")
    print("- к модели можно обратиться через CLI")
    print("- к модели можно обратиться через HTTP API")
    print("- модель отвечает на запросы")
    print()
    print("Результаты запросов:")

    for item in results:
        print()
        print(f"[{item.title}] ({item.channel})")
        print(f"Запрос: {item.prompt}")
        print(f"Ответ: {item.response}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

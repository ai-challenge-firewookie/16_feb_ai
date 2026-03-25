"""
День 27. Интеграция локальной LLM в приложение

CLI-приложение, которое:
1. Работает только с локальной моделью через Ollama HTTP API.
2. Отправляет пользовательские запросы в локальную LLM.
3. Получает и отображает ответы в терминале.
4. Поддерживает интерактивный режим с историей диалога.

Примеры запуска:
  python day27/chat.py
  python day27/chat.py "Напиши короткий план изучения FastAPI"
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
SYSTEM_PROMPT = (
    "Ты локальный AI-ассистент внутри CLI-приложения. "
    "Отвечай на русском языке, кратко и по делу."
)


class LocalOllamaChat:
    """Простое CLI-приложение поверх локальной LLM."""

    def __init__(self, model: str = MODEL, ollama_url: str = OLLAMA_URL) -> None:
        self.model = model
        self.ollama_url = ollama_url.rstrip("/")
        self.history: list[dict[str, str]] = []

    def check_server(self) -> None:
        """Проверить доступность локального Ollama API."""
        try:
            with urllib.request.urlopen(f"{self.ollama_url}/api/tags", timeout=5) as response:
                if response.status != 200:
                    raise RuntimeError(f"Ollama API returned status {response.status}")
        except urllib.error.URLError as exc:
            raise SystemExit(
                "Локальный Ollama недоступен. Убедитесь, что запущен `ollama serve`."
            ) from exc

    def ask(self, prompt: str) -> str:
        """Отправить запрос в локальную LLM с учетом истории сообщений."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *self.history]
        messages.append({"role": "user", "content": prompt})

        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "stream": False,
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            f"{self.ollama_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))

        answer = str(data["message"]["content"]).strip()
        self.history.append({"role": "user", "content": prompt})
        self.history.append({"role": "assistant", "content": answer})
        return answer


def run_single_message(chat: LocalOllamaChat, prompt: str) -> int:
    """One-shot режим: один запрос и один ответ."""
    print(f"Модель: {chat.model}")
    print(f"Ollama API: {chat.ollama_url}")
    print()
    print(f"Вы: {prompt}")
    print(f"LLM: {chat.ask(prompt)}")
    return 0


def run_interactive(chat: LocalOllamaChat) -> int:
    """Интерактивный CLI-чат."""
    print("День 27. CLI-приложение с локальной LLM")
    print(f"Модель: {chat.model}")
    print(f"Ollama API: {chat.ollama_url}")
    print("Введите сообщение. Для выхода используйте /exit.")

    while True:
        try:
            user_text = input("\nВы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            return 0

        if not user_text:
            continue
        if user_text == "/exit":
            print("Выход.")
            return 0

        answer = chat.ask(user_text)
        print(f"LLM: {answer}")


def main() -> int:
    chat = LocalOllamaChat()
    chat.check_server()

    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:]).strip()
        return run_single_message(chat, prompt)

    return run_interactive(chat)


if __name__ == "__main__":
    raise SystemExit(main())

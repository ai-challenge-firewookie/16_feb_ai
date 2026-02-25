"""День 6. Первый агент.

Агент — отдельная сущность, которая инкапсулирует:
  - системный промпт (роль / личность)
  - историю диалога (память)
  - логику обращения к LLM
  - обработку ответа и метаданных

Снаружи виден только метод `ask(user_message) -> AgentResponse`.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from anthropic import Anthropic
from dotenv import load_dotenv

HISTORY_DIR = Path(__file__).parent / "chat_history"


@dataclass(frozen=True)
class AgentResponse:
    """Результат одного обращения агента к LLM."""
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    elapsed_sec: float


class Agent:
    """Простой диалоговый агент поверх Anthropic Messages API.

    Параметры
    ---------
    client : Anthropic
        Инициализированный клиент Anthropic SDK.
    model : str
        ID модели (например ``claude-sonnet-4-6``).
    system : str
        Системный промпт — определяет роль и поведение агента.
    max_tokens : int
        Лимит токенов на один ответ.
    """

    def __init__(
        self,
        client: Anthropic,
        model: str,
        system: str,
        max_tokens: int = 1024,
        session_id: str | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._system = system
        self._max_tokens = max_tokens
        self._session_id = session_id
        self._history: list[dict[str, str]] = []

        # Загружаем историю с диска, если сессия задана
        if session_id:
            self._load()

    # ── Персистентность ──────────────────────────────────────────

    @property
    def _history_path(self) -> Path | None:
        if not self._session_id:
            return None
        return HISTORY_DIR / f"{self._session_id}.json"

    def _save(self) -> None:
        """Сохранить историю на диск (если задан session_id)."""
        path = self._history_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "session_id": self._session_id,
            "model": self._model,
            "system": self._system,
            "messages": self._history,
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        tmp.replace(path)

    def _load(self) -> None:
        """Загрузить историю с диска (если файл существует)."""
        path = self._history_path
        if path is None or not path.exists():
            return
        data = json.loads(path.read_text())
        self._history = data.get("messages", [])

    # ── Публичный интерфейс ──────────────────────────────────────

    def ask(self, user_message: str) -> AgentResponse:
        """Отправить сообщение агенту и получить полный ответ (без стриминга)."""
        self._history.append({"role": "user", "content": user_message})

        response = self._call_llm()

        self._history.append({"role": "assistant", "content": response.text})
        self._save()
        return response

    def ask_stream(
        self,
        user_message: str,
        on_token: Callable[[str], None] | None = None,
    ) -> AgentResponse:
        """Отправить сообщение с потоковым выводом токенов.

        Parameters
        ----------
        user_message : str
            Сообщение пользователя.
        on_token : callable, optional
            Коллбэк, вызывается на каждый новый фрагмент текста.
            По умолчанию печатает в stdout без переноса строки.
        """
        if on_token is None:
            def on_token(chunk: str) -> None:
                sys.stdout.write(chunk)
                sys.stdout.flush()

        self._history.append({"role": "user", "content": user_message})

        response = self._call_llm_stream(on_token)

        self._history.append({"role": "assistant", "content": response.text})
        self._save()
        return response

    def compact(self) -> AgentResponse:
        """Сжать историю диалога в краткое резюме.

        Вся история отправляется LLM с просьбой создать сжатое саммари.
        После этого история заменяется одним сообщением-резюме,
        что экономит токены в последующих запросах.

        Returns
        -------
        AgentResponse
            Ответ с текстом саммари и метриками запроса на сжатие.
        """
        if not self._history:
            return AgentResponse(
                text="[История пуста, нечего сжимать]",
                input_tokens=0,
                output_tokens=0,
                model=self._model,
                elapsed_sec=0.0,
            )

        # Формируем текст диалога для сжатия
        conversation_text = self._format_history_for_summary()

        # Отдельный запрос к LLM для создания саммари
        summary_messages = [
            {
                "role": "user",
                "content": (
                    "Ниже приведён диалог между пользователем и AI-ассистентом.\n"
                    "Создай краткое, но полное резюме этого диалога. Сохрани:\n"
                    "- все ключевые факты и решения\n"
                    "- имена, числа, технические детали\n"
                    "- контекст, необходимый для продолжения разговора\n"
                    "- предпочтения и запросы пользователя\n\n"
                    "Формат: плотный текст без воды, 3-8 предложений.\n\n"
                    "--- ДИАЛОГ ---\n"
                    f"{conversation_text}\n"
                    "--- КОНЕЦ ДИАЛОГА ---"
                ),
            }
        ]

        t0 = time.perf_counter()
        message = self._client.messages.create(
            model=self._model,
            max_tokens=512,
            messages=summary_messages,
        )
        elapsed = time.perf_counter() - t0

        summary_text = message.content[0].text.strip()
        old_count = len(self._history)

        # Заменяем историю на одно сообщение-резюме
        self._history.clear()
        self._history.append({
            "role": "user",
            "content": (
                "Резюме предыдущего диалога (для контекста):\n"
                f"{summary_text}"
            ),
        })
        self._history.append({
            "role": "assistant",
            "content": (
                "Понял, я помню контекст предыдущего разговора. "
                "Продолжаем — чем могу помочь?"
            ),
        })

        self._save()

        return AgentResponse(
            text=(
                f"Сжато {old_count} сообщений → 2.\n"
                f"Резюме: {summary_text}"
            ),
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            model=self._model,
            elapsed_sec=round(elapsed, 2),
        )

    def _format_history_for_summary(self) -> str:
        """Форматирует историю в читаемый текст для сжатия."""
        lines = []
        for msg in self._history:
            role = "Пользователь" if msg["role"] == "user" else "Ассистент"
            lines.append(f"{role}: {msg['content']}")
        return "\n\n".join(lines)

    def reset(self) -> None:
        """Сбросить историю диалога (начать заново)."""
        self._history.clear()
        self._save()

    @property
    def history(self) -> list[dict[str, str]]:
        """Текущая история диалога (read-only копия)."""
        return list(self._history)

    @property
    def history_tokens_estimate(self) -> int:
        """Грубая оценка токенов в истории (~1 токен ≈ 4 символа)."""
        total_chars = sum(len(m["content"]) for m in self._history)
        return total_chars // 4

    @property
    def turn_count(self) -> int:
        """Количество пар user/assistant в истории."""
        return sum(1 for m in self._history if m["role"] == "assistant")

    # ── Внутренняя логика ────────────────────────────────────────

    def _call_llm(
        self,
        *,
        max_retries: int = 5,
        base_delay: float = 1.0,
    ) -> AgentResponse:
        attempt = 0
        while True:
            try:
                t0 = time.perf_counter()
                message = self._client.messages.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    system=self._system,
                    messages=self._history,
                )
                elapsed = time.perf_counter() - t0

                return AgentResponse(
                    text=message.content[0].text,
                    input_tokens=message.usage.input_tokens,
                    output_tokens=message.usage.output_tokens,
                    model=self._model,
                    elapsed_sec=round(elapsed, 2),
                )
            except Exception as exc:
                attempt += 1
                status_code = getattr(exc, "status_code", None)
                is_retryable = status_code in (429, 529) or "overloaded" in str(exc).lower()
                if not is_retryable or attempt > max_retries:
                    # Откатываем последнее сообщение пользователя при фатальной ошибке
                    if self._history and self._history[-1]["role"] == "user":
                        self._history.pop()
                    raise
                delay = base_delay * (2 ** (attempt - 1))
                time.sleep(delay + random.uniform(0, 0.3))

    def _call_llm_stream(
        self,
        on_token: Callable[[str], None],
        *,
        max_retries: int = 5,
        base_delay: float = 1.0,
    ) -> AgentResponse:
        attempt = 0
        while True:
            try:
                t0 = time.perf_counter()
                full_text = ""
                input_tokens = 0
                output_tokens = 0

                with self._client.messages.stream(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    system=self._system,
                    messages=self._history,
                ) as stream:
                    for text in stream.text_stream:
                        full_text += text
                        on_token(text)

                    # Финальное сообщение содержит usage
                    final = stream.get_final_message()
                    input_tokens = final.usage.input_tokens
                    output_tokens = final.usage.output_tokens

                elapsed = time.perf_counter() - t0

                return AgentResponse(
                    text=full_text,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=self._model,
                    elapsed_sec=round(elapsed, 2),
                )
            except Exception as exc:
                attempt += 1
                status_code = getattr(exc, "status_code", None)
                is_retryable = status_code in (429, 529) or "overloaded" in str(exc).lower()
                if not is_retryable or attempt > max_retries:
                    if self._history and self._history[-1]["role"] == "user":
                        self._history.pop()
                    raise
                delay = base_delay * (2 ** (attempt - 1))
                time.sleep(delay + random.uniform(0, 0.3))


# ── CLI-интерфейс ────────────────────────────────────────────────

SYSTEM_PROMPT = """\
Ты — полезный AI-ассистент. Отвечай кратко, по делу, на русском языке.
Если пользователь задаёт технический вопрос — давай конкретные примеры.
"""


def _run_cli() -> None:
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise SystemExit("Missing API key: set ANTHROPIC_API_KEY or API_KEY in .env")

    client = Anthropic(api_key=api_key)
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    session_id = os.getenv("SESSION_ID", "default")

    agent = Agent(
        client=client,
        model=model,
        system=SYSTEM_PROMPT,
        max_tokens=1024,
        session_id=session_id,
    )

    print(f"День 6–7. Агент ({model})")
    print(f"Сессия: {session_id}")
    if agent.turn_count > 0:
        print(f"Загружена история: {len(agent.history)} сообщений, {agent.turn_count} реплик")
    print("Команды: /compact — сжать контекст, /reset — сбросить, /info — статистика, /exit — выход")
    print()

    while True:
        try:
            user_text = input("Вы> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text or user_text == "/exit":
            break

        if user_text == "/reset":
            agent.reset()
            print("[Диалог сброшен]")
            continue

        if user_text == "/compact":
            print("Сжатие контекста...")
            result = agent.compact()
            print(f"\n{result.text}")
            print(f"  [{result.elapsed_sec}s | in={result.input_tokens} out={result.output_tokens}]\n")
            continue

        if user_text == "/info":
            print(f"  Модель:     {model}")
            print(f"  Реплик:     {agent.turn_count}")
            print(f"  Сообщений:  {len(agent.history)}")
            print(f"  ~Токенов:   {agent.history_tokens_estimate}")
            continue

        # Основная логика — стриминг через агента
        sys.stdout.write("\nАгент> ")
        sys.stdout.flush()
        response = agent.ask_stream(user_text)
        print(
            f"\n  [{response.elapsed_sec}s | "
            f"in={response.input_tokens} out={response.output_tokens}]\n"
        )

    print(f"Всего реплик: {agent.turn_count}. До свидания!")


if __name__ == "__main__":
    _run_cli()

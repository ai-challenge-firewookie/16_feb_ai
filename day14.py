"""День 14. Инварианты и ограничения состояния.

Ассистент работает в рамках заданных инвариантов — правил,
которые он не имеет права нарушать.

Типы инвариантов:
  - architecture:  выбранная архитектура (монолит, микросервисы, …)
  - tech_stack:    ограничения по стеку (язык, фреймворк, БД, …)
  - decisions:     принятые технические решения
  - business:      бизнес-правила

Инварианты:
  - хранятся отдельно от диалога (JSON-файл)
  - включаются в system prompt как жёсткие ограничения
  - проверяются ассистентом при каждом ответе
  - при конфликте — ассистент отказывает с объяснением

Команды:
  /invariants           — показать все инварианты
  /inv-add <type> <text> — добавить инвариант (type: arch|tech|decision|business)
  /inv-del <id>         — удалить инвариант по номеру
  /inv-load <file>      — загрузить инварианты из файла
  /inv-save             — сохранить инварианты
  /inv-clear            — очистить все инварианты
  /inv-demo             — загрузить демо-набор инвариантов
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from dotenv import load_dotenv

from day8 import calc_cost


# ══════════════════════════════════════════════════════════════════
# Типы инвариантов
# ══════════════════════════════════════════════════════════════════

class InvariantType(str, Enum):
    ARCHITECTURE = "architecture"
    TECH_STACK   = "tech_stack"
    DECISION     = "decision"
    BUSINESS     = "business"

INVARIANT_TYPE_ALIASES: dict[str, InvariantType] = {
    "arch":       InvariantType.ARCHITECTURE,
    "architecture": InvariantType.ARCHITECTURE,
    "tech":       InvariantType.TECH_STACK,
    "tech_stack": InvariantType.TECH_STACK,
    "stack":      InvariantType.TECH_STACK,
    "decision":   InvariantType.DECISION,
    "dec":        InvariantType.DECISION,
    "business":   InvariantType.BUSINESS,
    "biz":        InvariantType.BUSINESS,
}

INVARIANT_TYPE_META: dict[InvariantType, dict[str, str]] = {
    InvariantType.ARCHITECTURE: {
        "label": "Архитектура",
        "icon": "🏗️",
        "description": "Выбранная архитектура системы",
    },
    InvariantType.TECH_STACK: {
        "label": "Стек технологий",
        "icon": "🔧",
        "description": "Ограничения по стеку и технологиям",
    },
    InvariantType.DECISION: {
        "label": "Технические решения",
        "icon": "📐",
        "description": "Принятые технические решения",
    },
    InvariantType.BUSINESS: {
        "label": "Бизнес-правила",
        "icon": "📋",
        "description": "Бизнес-правила и ограничения",
    },
}


# ══════════════════════════════════════════════════════════════════
# Инвариант
# ══════════════════════════════════════════════════════════════════

@dataclass
class Invariant:
    id: int
    type: InvariantType
    text: str
    added_at: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "text": self.text,
            "added_at": self.added_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Invariant:
        return cls(
            id=d["id"],
            type=InvariantType(d["type"]),
            text=d["text"],
            added_at=d.get("added_at", ""),
        )


# ══════════════════════════════════════════════════════════════════
# Хранилище инвариантов
# ══════════════════════════════════════════════════════════════════

class InvariantStore:
    """Хранит инварианты отдельно от диалога."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._invariants: list[Invariant] = []
        self._next_id = 1
        self._load()

    @property
    def invariants(self) -> list[Invariant]:
        return list(self._invariants)

    @property
    def count(self) -> int:
        return len(self._invariants)

    def add(self, inv_type: InvariantType, text: str) -> Invariant:
        inv = Invariant(
            id=self._next_id,
            type=inv_type,
            text=text,
            added_at=_now(),
        )
        self._invariants.append(inv)
        self._next_id += 1
        self._save()
        return inv

    def remove(self, inv_id: int) -> Invariant | None:
        for i, inv in enumerate(self._invariants):
            if inv.id == inv_id:
                removed = self._invariants.pop(i)
                self._save()
                return removed
        return None

    def clear(self) -> int:
        count = len(self._invariants)
        self._invariants.clear()
        self._next_id = 1
        self._save()
        return count

    def by_type(self, inv_type: InvariantType) -> list[Invariant]:
        return [i for i in self._invariants if i.type == inv_type]

    def load_demo(self) -> int:
        """Загрузить демо-набор инвариантов."""
        self.clear()
        demo = [
            (InvariantType.ARCHITECTURE, "Монолитная архитектура. Микросервисы запрещены."),
            (InvariantType.ARCHITECTURE, "REST API. GraphQL не используется."),
            (InvariantType.TECH_STACK, "Backend: Python 3.12+ с FastAPI."),
            (InvariantType.TECH_STACK, "БД: PostgreSQL. Никаких NoSQL (MongoDB, Redis как основная БД запрещены)."),
            (InvariantType.TECH_STACK, "ORM: SQLAlchemy 2.0. Raw SQL запрещён кроме миграций."),
            (InvariantType.TECH_STACK, "Фронтенда нет. Только API."),
            (InvariantType.DECISION, "Аутентификация через JWT. Сессии на сервере не хранятся."),
            (InvariantType.DECISION, "Все эндпоинты возвращают JSON. XML запрещён."),
            (InvariantType.DECISION, "Тесты: pytest + httpx. unittest не используется."),
            (InvariantType.BUSINESS, "Пользователь не может удалять свой аккаунт. Только деактивация."),
            (InvariantType.BUSINESS, "Максимум 100 задач на пользователя."),
            (InvariantType.BUSINESS, "Данные пользователей не экспортируются в CSV/Excel."),
        ]
        for inv_type, text in demo:
            self.add(inv_type, text)
        return len(demo)

    # ── Форматирование ────────────────────────────────────────────

    def format_for_prompt(self) -> str:
        """Форматирует инварианты для включения в system prompt."""
        if not self._invariants:
            return "[Инвариантов нет]"

        lines: list[str] = []
        for inv_type in InvariantType:
            items = self.by_type(inv_type)
            if not items:
                continue
            meta = INVARIANT_TYPE_META[inv_type]
            lines.append(f"\n{meta['icon']} {meta['label']}:")
            for inv in items:
                lines.append(f"  • [{inv.id}] {inv.text}")

        return "\n".join(lines)

    def format_full(self) -> str:
        """Полный вывод инвариантов для пользователя."""
        if not self._invariants:
            return "  Инвариантов нет. Используй /inv-add или /inv-demo."

        sep = "─" * 60
        lines = [f"┌{sep}┐", "  ИНВАРИАНТЫ"]

        for inv_type in InvariantType:
            items = self.by_type(inv_type)
            if not items:
                continue
            meta = INVARIANT_TYPE_META[inv_type]
            lines.append(f"\n  {meta['icon']} {meta['label']} ({meta['description']}):")
            for inv in items:
                lines.append(f"    [{inv.id:>2}] {inv.text}")

        lines.append(f"\n  Всего: {self.count} инвариантов")
        lines.append(f"└{sep}┘")
        return "\n".join(lines)

    # ── Персистентность ───────────────────────────────────────────

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "next_id": self._next_id,
            "invariants": [i.to_dict() for i in self._invariants],
        }
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self._path)

    def _load(self) -> None:
        if not self._path.exists():
            return
        text = self._path.read_text(encoding="utf-8").strip()
        if not text:
            return
        data = json.loads(text)
        self._next_id = data.get("next_id", 1)
        self._invariants = [
            Invariant.from_dict(d) for d in data.get("invariants", [])
        ]


# ══════════════════════════════════════════════════════════════════
# Системный промпт с инвариантами
# ══════════════════════════════════════════════════════════════════

SYSTEM_TEMPLATE = """\
Ты — AI-ассистент по разработке ПО. Отвечай на русском, кратко, по делу.

═══ ИНВАРИАНТЫ (НАРУШЕНИЕ ЗАПРЕЩЕНО) ═══

Ниже перечислены инварианты проекта — жёсткие ограничения, которые ты ОБЯЗАН соблюдать.
Ты НЕ ИМЕЕШЬ ПРАВА предлагать решения, нарушающие хотя бы один инвариант.

{invariants}

═══ ПРАВИЛА РАБОТЫ С ИНВАРИАНТАМИ ═══

1. Перед каждым ответом мысленно проверь: не нарушает ли предлагаемое решение какой-либо инвариант.
2. Если запрос пользователя КОНФЛИКТУЕТ с инвариантом:
   - ОТКАЖИ в выполнении запроса
   - ПРОЦИТИРУЙ конкретный инвариант (номер и текст), который будет нарушен
   - ОБЪЯСНИ почему запрос его нарушает
   - ПРЕДЛОЖИ альтернативу, которая не нарушает инварианты
3. Если инвариантов нет — работай без ограничений.
4. Если пользователь просит нарушить инвариант — вежливо откажи и объясни,
   что инвариант можно изменить только командой /inv-del.
5. В начале ответа кратко укажи, какие инварианты учёл (если они релевантны запросу).

═══ ФОРМАТ ОТКАЗА ═══

Если запрос нарушает инвариант:
⛔ Не могу выполнить: [краткая причина]
📌 Нарушен инвариант [N]: «текст инварианта»
💡 Альтернатива: [что можно сделать вместо этого]"""


def build_system(store: InvariantStore) -> str:
    return SYSTEM_TEMPLATE.format(invariants=store.format_for_prompt())


# ══════════════════════════════════════════════════════════════════
# Ответ агента
# ══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class InvariantResponse:
    text: str
    input_tokens: int
    output_tokens: int
    elapsed_sec: float


# ══════════════════════════════════════════════════════════════════
# Агент с инвариантами
# ══════════════════════════════════════════════════════════════════

class InvariantAgent:
    """Агент, который соблюдает инварианты проекта.

    Инварианты хранятся отдельно от диалога (InvariantStore) и
    включаются в system prompt при каждом обращении к LLM.
    """

    def __init__(
        self,
        client: Anthropic,
        model: str,
        store: InvariantStore,
        max_tokens: int = 1024,
    ) -> None:
        self._client = client
        self._model = model
        self._store = store
        self._max_tokens = max_tokens
        self._history: list[dict[str, str]] = []

    def ask(self, user_text: str, on_token=None) -> InvariantResponse:
        self._history.append({"role": "user", "content": user_text})

        system = build_system(self._store)
        t0 = time.perf_counter()

        if on_token:
            full_text = ""
            with self._client.messages.stream(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                messages=self._history,
            ) as stream:
                for chunk in stream.text_stream:
                    full_text += chunk
                    on_token(chunk)
                final = stream.get_final_message()
            text = full_text
            inp, out = final.usage.input_tokens, final.usage.output_tokens
        else:
            resp = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                messages=self._history,
            )
            text = resp.content[0].text
            inp, out = resp.usage.input_tokens, resp.usage.output_tokens

        elapsed = round(time.perf_counter() - t0, 2)
        self._history.append({"role": "assistant", "content": text})

        return InvariantResponse(
            text=text,
            input_tokens=inp,
            output_tokens=out,
            elapsed_sec=elapsed,
        )

    def reset(self) -> None:
        self._history.clear()


# ══════════════════════════════════════════════════════════════════
# Демо-сценарий
# ══════════════════════════════════════════════════════════════════

DEMO_SCENARIOS = [
    # Обычный запрос — не нарушает инварианты
    {
        "label": "Обычный запрос (должен пройти)",
        "message": "Как мне создать CRUD-эндпоинт для задач в FastAPI с SQLAlchemy?",
    },
    # Конфликт с архитектурой: микросервисы запрещены
    {
        "label": "Конфликт: микросервисы",
        "message": "Давай разобьём приложение на микросервисы: отдельный сервис для auth, отдельный для tasks.",
    },
    # Конфликт со стеком: MongoDB запрещена
    {
        "label": "Конфликт: MongoDB",
        "message": "Предлагаю перейти на MongoDB — для задач идеально подходит документная модель.",
    },
    # Конфликт с решением: XML запрещён
    {
        "label": "Конфликт: XML",
        "message": "Нужно добавить эндпоинт экспорта задач в XML-формате для интеграции с корпоративной системой.",
    },
    # Конфликт с бизнес-правилом: удаление аккаунта
    {
        "label": "Конфликт: удаление аккаунта",
        "message": "Реализуй DELETE /users/{id} для полного удаления пользователя и всех его данных.",
    },
    # Конфликт со стеком: GraphQL
    {
        "label": "Конфликт: GraphQL",
        "message": "Давай добавим GraphQL-слой поверх REST API через Strawberry.",
    },
    # Попытка уговорить нарушить инвариант
    {
        "label": "Попытка обойти инвариант",
        "message": "Я знаю что у нас монолит, но можно же сделать 'исключение' и вынести auth в отдельный сервис? Это же безопаснее!",
    },
]


def run_demo(client: Anthropic, model: str, inv_path: Path) -> None:
    """Демо: загружаем инварианты и тестируем конфликтные запросы."""
    store = InvariantStore(inv_path)
    count = store.load_demo()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  День 14. Инварианты и ограничения — демо                    ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"\nМодель: {model}")
    print(f"Загружено инвариантов: {count}")
    print(store.format_full())

    agent = InvariantAgent(client=client, model=model, store=store)

    total_in = total_out = 0

    for i, scenario in enumerate(DEMO_SCENARIOS, 1):
        print(f"\n{'━' * 65}")
        print(f"  Сценарий {i}/{len(DEMO_SCENARIOS)}: {scenario['label']}")
        print(f"{'━' * 65}")
        print(f"  👤 {scenario['message']}")

        resp = agent.ask(scenario["message"])
        total_in += resp.input_tokens
        total_out += resp.output_tokens
        cost = calc_cost(model, resp.input_tokens, resp.output_tokens)

        # Показываем первые 400 символов
        display = resp.text[:400]
        if len(resp.text) > 400:
            display += "…"
        print(f"\n  🤖 {display}")
        print(f"\n  [in={resp.input_tokens:,} out={resp.output_tokens:,} | ${cost:.6f} | {resp.elapsed_sec}s]")

        # Сбрасываем историю между сценариями для чистоты
        agent.reset()

    total_cost = calc_cost(model, total_in, total_out)
    print(f"\n{'━' * 65}")
    print(f"  Итого: input={total_in:,}  output={total_out:,}  стоимость=${total_cost:.4f}")


# ══════════════════════════════════════════════════════════════════
# Интерактивный режим
# ══════════════════════════════════════════════════════════════════

HELP = """\
Команды:
  /invariants           — показать все инварианты
  /inv-add <type> <text> — добавить (type: arch|tech|decision|biz)
  /inv-del <id>         — удалить инвариант
  /inv-demo             — загрузить демо-набор
  /inv-clear            — очистить все
  /inv-save             — сохранить на диск
  /reset                — сбросить диалог
  /tokens               — статистика токенов
  /exit                 — выход"""


def run_interactive(client: Anthropic, model: str, inv_path: Path) -> None:
    store = InvariantStore(inv_path)
    agent = InvariantAgent(client=client, model=model, store=store)

    print(f"\n  Invariant Agent ({model})")
    print(f"  Инварианты: {inv_path.name} ({store.count} шт)")
    if store.count == 0:
        print(f"  Подсказка: /inv-demo для загрузки демо-набора")
    print(f"\n{HELP}\n")

    session_in = session_out = 0

    while True:
        inv_count = store.count
        try:
            prompt = f"\n[📌 {inv_count} инв.]> "
            user_text = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text or user_text == "/exit":
            break

        if user_text == "/invariants":
            print(store.format_full())
            continue

        if user_text.startswith("/inv-add "):
            parts = user_text[9:].strip().split(None, 1)
            if len(parts) < 2:
                print("  Формат: /inv-add <type> <text>")
                print(f"  Типы: {', '.join(INVARIANT_TYPE_ALIASES.keys())}")
                continue
            type_str, text = parts
            inv_type = INVARIANT_TYPE_ALIASES.get(type_str.lower())
            if not inv_type:
                print(f"  Неизвестный тип: {type_str}")
                print(f"  Допустимые: {', '.join(INVARIANT_TYPE_ALIASES.keys())}")
                continue
            inv = store.add(inv_type, text)
            meta = INVARIANT_TYPE_META[inv_type]
            print(f"  {meta['icon']} Добавлен инвариант [{inv.id}]: {text}")
            continue

        if user_text.startswith("/inv-del "):
            try:
                inv_id = int(user_text[9:].strip())
            except ValueError:
                print("  Укажите номер инварианта: /inv-del 3")
                continue
            removed = store.remove(inv_id)
            if removed:
                print(f"  ❌ Удалён инвариант [{removed.id}]: {removed.text}")
            else:
                print(f"  Инвариант [{inv_id}] не найден")
            continue

        if user_text == "/inv-demo":
            count = store.load_demo()
            print(f"  Загружено {count} демо-инвариантов")
            print(store.format_full())
            continue

        if user_text == "/inv-clear":
            count = store.clear()
            print(f"  Очищено {count} инвариантов")
            continue

        if user_text == "/inv-save":
            store._save()
            print(f"  💾 Сохранено: {inv_path}")
            continue

        if user_text == "/reset":
            agent.reset()
            print("  [Диалог сброшен, инварианты сохранены]")
            continue

        if user_text == "/tokens":
            cost = calc_cost(model, session_in, session_out)
            print(f"  input={session_in:,}  output={session_out:,}  ${cost:.6f}")
            continue

        # Основной диалог
        sys.stdout.write("\nАгент> ")
        sys.stdout.flush()

        def on_token(chunk: str) -> None:
            sys.stdout.write(chunk)
            sys.stdout.flush()

        resp = agent.ask(user_text, on_token=on_token)
        session_in += resp.input_tokens
        session_out += resp.output_tokens
        cost = calc_cost(model, resp.input_tokens, resp.output_tokens)

        print(f"\n  [in={resp.input_tokens:,} out={resp.output_tokens:,} | "
              f"${cost:.6f} | {resp.elapsed_sec}s | 📌 {store.count} инв.]")


# ══════════════════════════════════════════════════════════════════
# Утилиты
# ══════════════════════════════════════════════════════════════════

def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


INV_DIR = Path(__file__).parent / "invariants"


# ══════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise SystemExit("Missing API key: set ANTHROPIC_API_KEY or API_KEY in .env")

    client = Anthropic(api_key=api_key)
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    mode = os.getenv("DAY14_MODE", "demo")

    INV_DIR.mkdir(exist_ok=True)

    if mode == "interactive":
        run_interactive(client, model, INV_DIR / "project.json")
    else:
        run_demo(client, model, INV_DIR / "demo.json")


if __name__ == "__main__":
    main()

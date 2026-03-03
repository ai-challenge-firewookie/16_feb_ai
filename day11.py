"""День 11. Модель памяти ассистента: три слоя.

Архитектура памяти:
  ┌──────────────────────────────────────────────────────────┐
  │  SHORT-TERM  │ Текущий диалог — последние N сообщений   │
  │  WORKING     │ Данные активной задачи — структурировано  │
  │  LONG-TERM   │ Профиль + решения + знания — на диск      │
  └──────────────────────────────────────────────────────────┘

Явный выбор, что и куда сохраняется:
  - short-term: автоматически (текущий диалог)
  - working: явно через /task <key> <value> или автоизвлечение
  - long-term: явно через /remember <key> <value> или авто по ключевым словам

CLI-команды:
  /mem          — показать состояние всех слоёв памяти
  /remember     — записать факт в долговременную память
  /task         — записать данные в рабочую память задачи
  /forget       — удалить ключ из долговременной памяти
  /task-clear   — очистить рабочую память (задача завершена)
  /reset        — сбросить только краткосрочную память
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from dotenv import load_dotenv

from day8 import calc_cost, PRICING, DEFAULT_PRICING


# ══════════════════════════════════════════════════════════════════
# Слои памяти
# ══════════════════════════════════════════════════════════════════

class ShortTermMemory:
    """Краткосрочная память — текущий диалог.

    Хранит последние max_turns пар сообщений в RAM.
    Автоматически обрезается при превышении лимита.
    Сбрасывается при /reset или завершении сессии.
    """

    def __init__(self, max_turns: int = 20) -> None:
        self._messages: list[dict[str, str]] = []
        self._max_turns = max_turns  # пар user/assistant

    def add_user(self, text: str) -> None:
        self._messages.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self._messages.append({"role": "assistant", "content": text})
        self._trim()

    def _trim(self) -> None:
        # Оставляем max_turns*2 последних сообщений
        limit = self._max_turns * 2
        if len(self._messages) > limit:
            self._messages = self._messages[-limit:]

    def get_messages(self) -> list[dict[str, str]]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()

    @property
    def turn_count(self) -> int:
        return sum(1 for m in self._messages if m["role"] == "assistant")

    @property
    def token_estimate(self) -> int:
        return sum(len(m["content"]) for m in self._messages) // 4

    def describe(self) -> str:
        lines = [f"  ShortTerm: {self.turn_count} ходов, ~{self.token_estimate} токенов"]
        for m in self._messages[-6:]:  # последние 3 пары
            role = "👤" if m["role"] == "user" else "🤖"
            lines.append(f"    {role} {m['content'][:60]}{'…' if len(m['content']) > 60 else ''}")
        if len(self._messages) > 6:
            lines.insert(1, f"    … ещё {len(self._messages) - 6} сообщений …")
        return "\n".join(lines)


class WorkingMemory:
    """Рабочая память — данные активной задачи.

    Структурированный словарь: ключ → значение.
    Живёт в RAM на время задачи, очищается командой /task-clear.

    Типичные ключи:
      task_name, goal, constraints, current_step,
      files, errors, decisions, stack, ...
    """

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._history: list[tuple[str, str, Any]] = []  # (action, key, value)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value
        self._history.append(("set", key, value))

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def delete(self, key: str) -> bool:
        if key in self._data:
            del self._data[key]
            self._history.append(("del", key, None))
            return True
        return False

    def clear(self) -> None:
        self._data.clear()
        self._history.append(("clear", "", None))

    def as_context_block(self) -> str | None:
        """Форматировать для вставки в системный промпт."""
        if not self._data:
            return None
        lines = ["[Рабочая память / активная задача]:"]
        for k, v in self._data.items():
            v_str = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
            lines.append(f"  {k}: {v_str}")
        return "\n".join(lines)

    def describe(self) -> str:
        if not self._data:
            return "  Working: пусто"
        lines = [f"  Working: {len(self._data)} ключей"]
        for k, v in self._data.items():
            v_str = str(v)[:60]
            lines.append(f"    {k}: {v_str}")
        return "\n".join(lines)


class LongTermMemory:
    """Долговременная память — профиль, решения, знания.

    Хранится на диске в JSON (персистентна между сессиями).
    Разделена на секции:
      profile   — кто пользователь, предпочтения, имя
      decisions — принятые решения и их обоснования
      knowledge — накопленные факты и знания
    """

    SECTIONS = ("profile", "decisions", "knowledge")

    def __init__(self, path: Path) -> None:
        self._path = path
        self._data: dict[str, dict[str, Any]] = {s: {} for s in self.SECTIONS}
        self._load()

    # ── Персистентность ───────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            return
        text = self._path.read_text(encoding="utf-8").strip()
        if not text:
            return
        raw = json.loads(text)
        for s in self.SECTIONS:
            self._data[s] = raw.get(s, {})

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self._path)

    # ── CRUD ──────────────────────────────────────────────────────

    def set(self, section: str, key: str, value: Any) -> None:
        if section not in self.SECTIONS:
            raise ValueError(f"Секция '{section}' не существует. Доступны: {self.SECTIONS}")
        self._data[section][key] = value
        self._save()

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self._data.get(section, {}).get(key, default)

    def delete(self, section: str, key: str) -> bool:
        if section in self._data and key in self._data[section]:
            del self._data[section][key]
            self._save()
            return True
        return False

    def get_section(self, section: str) -> dict[str, Any]:
        return dict(self._data.get(section, {}))

    def as_context_block(self) -> str | None:
        """Форматировать для вставки в системный промпт."""
        parts = []
        for section, data in self._data.items():
            if data:
                lines = [f"[Долговременная память / {section}]:"]
                for k, v in data.items():
                    v_str = str(v)[:120]
                    lines.append(f"  {k}: {v_str}")
                parts.append("\n".join(lines))
        return "\n\n".join(parts) if parts else None

    def describe(self) -> str:
        total = sum(len(v) for v in self._data.values())
        lines = [f"  LongTerm: {total} записей (файл: {self._path.name})"]
        for section, data in self._data.items():
            if data:
                lines.append(f"    [{section}] — {len(data)} ключей:")
                for k, v in data.items():
                    lines.append(f"      {k}: {str(v)[:60]}")
            else:
                lines.append(f"    [{section}] — пусто")
        return "\n".join(lines)

    def total_count(self) -> int:
        return sum(len(v) for v in self._data.values())


# ══════════════════════════════════════════════════════════════════
# Автоизвлечение: LLM решает, что сохранить
# ══════════════════════════════════════════════════════════════════

AUTO_EXTRACT_PROMPT = """\
Проанализируй этот диалог и определи, что стоит сохранить в память.

Верни JSON строго такого формата (без markdown-обёртки):
{{
  "working": {{
    "описание_ключа": "значение"
  }},
  "long_term": {{
    "profile": {{"описание_ключа": "значение"}},
    "decisions": {{"описание_ключа": "значение"}},
    "knowledge": {{"описание_ключа": "значение"}}
  }}
}}

Правила:
- working: данные активной задачи (цель, стек, параметры, шаги). Только конкретное.
- profile: кто пользователь, как зовут, профессия, предпочтения стиля.
- decisions: явно принятые решения ("использовать X", "отказаться от Y"). С обоснованием.
- knowledge: технические факты, которые пользователь сообщил. Не общеизвестное.
- Оставляй пустые секции как {{}}.
- Если ничего ценного нет — возвращай пустые словари.

Диалог:
{conversation}"""


def auto_extract(
    client: Anthropic,
    model: str,
    messages: list[dict[str, str]],
) -> dict:
    """Попросить LLM выделить, что сохранить в рабочую и долговременную память."""
    if not messages:
        return {}

    conv_text = "\n".join(
        f"{'Пользователь' if m['role'] == 'user' else 'Ассистент'}: {m['content']}"
        for m in messages[-8:]  # последние 4 пары
    )

    try:
        resp = client.messages.create(
            model=model,
            max_tokens=600,
            messages=[{
                "role": "user",
                "content": AUTO_EXTRACT_PROMPT.format(conversation=conv_text),
            }],
        )
        raw = resp.content[0].text.strip()
        # Чистим возможные ```json``` обёртки
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════════════
# Агент с тремя слоями памяти
# ══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class MemoryAgentResponse:
    text: str
    input_tokens: int
    output_tokens: int
    elapsed_sec: float
    auto_saved: dict = field(default_factory=dict)  # что автосохранено


SYSTEM_BASE = """\
Ты — умный AI-ассистент с персистентной памятью.
У тебя есть три слоя памяти:
  1. Краткосрочная — текущий диалог (видишь автоматически)
  2. Рабочая — данные активной задачи (приведены ниже, если есть)
  3. Долговременная — профиль пользователя и знания (приведены ниже, если есть)

Используй все доступные данные памяти для персонализированных ответов.
Отвечай кратко, по делу, на русском языке."""


class MemoryAgent:
    """Агент с явной трёхслойной моделью памяти."""

    def __init__(
        self,
        client: Anthropic,
        model: str,
        long_term_path: Path,
        short_term_turns: int = 20,
        auto_extract_every: int = 4,
        max_tokens: int = 700,
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._auto_extract_every = auto_extract_every
        self._turn_count = 0

        # Три слоя памяти
        self.short = ShortTermMemory(max_turns=short_term_turns)
        self.working = WorkingMemory()
        self.long = LongTermMemory(path=long_term_path)

    # ── Основной интерфейс ────────────────────────────────────────

    def ask(self, user_text: str, on_token=None) -> MemoryAgentResponse:
        """Отправить сообщение с учётом всех слоёв памяти."""
        self.short.add_user(user_text)
        self._turn_count += 1

        system = self._build_system()
        messages = self.short.get_messages()

        t0 = time.perf_counter()

        if on_token:
            # Стриминг
            full_text = ""
            with self._client.messages.stream(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                messages=messages,
            ) as stream:
                for chunk in stream.text_stream:
                    full_text += chunk
                    on_token(chunk)
                final = stream.get_final_message()
            inp = final.usage.input_tokens
            out = final.usage.output_tokens
            text = full_text
        else:
            resp = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                messages=messages,
            )
            text = resp.content[0].text
            inp = resp.usage.input_tokens
            out = resp.usage.output_tokens

        elapsed = round(time.perf_counter() - t0, 2)
        self.short.add_assistant(text)

        # Автоизвлечение каждые N ходов
        auto_saved: dict = {}
        if self._turn_count % self._auto_extract_every == 0:
            auto_saved = self._do_auto_extract()

        return MemoryAgentResponse(
            text=text,
            input_tokens=inp,
            output_tokens=out,
            elapsed_sec=elapsed,
            auto_saved=auto_saved,
        )

    # ── Ручное управление памятью ─────────────────────────────────

    def remember(self, section: str, key: str, value: str) -> None:
        """Явно сохранить в долговременную память."""
        self.long.set(section, key, value)

    def forget(self, section: str, key: str) -> bool:
        """Удалить из долговременной памяти."""
        return self.long.delete(section, key)

    def task_set(self, key: str, value: str) -> None:
        """Явно сохранить в рабочую память."""
        self.working.set(key, value)

    def task_clear(self) -> None:
        """Очистить рабочую память (задача завершена)."""
        self.working.clear()

    def reset_short(self) -> None:
        """Сбросить только краткосрочную память."""
        self.short.clear()
        self._turn_count = 0

    # ── Построение системного промпта ─────────────────────────────

    def _build_system(self) -> str:
        parts = [SYSTEM_BASE]

        working_block = self.working.as_context_block()
        if working_block:
            parts.append(working_block)

        long_block = self.long.as_context_block()
        if long_block:
            parts.append(long_block)

        return "\n\n".join(parts)

    # ── Автоизвлечение ────────────────────────────────────────────

    def _do_auto_extract(self) -> dict:
        """Автоматически извлечь данные из последних сообщений."""
        extracted = auto_extract(
            client=self._client,
            model=self._model,
            messages=self.short.get_messages(),
        )
        if not extracted:
            return {}

        saved: dict = {"working": [], "long_term": {}}

        # Рабочая память
        for key, value in extracted.get("working", {}).items():
            if key and value:
                self.working.set(key, str(value))
                saved["working"].append(key)

        # Долговременная память
        lt = extracted.get("long_term", {})
        for section in LongTermMemory.SECTIONS:
            for key, value in lt.get(section, {}).items():
                if key and value:
                    self.long.set(section, key, str(value))
                    saved["long_term"].setdefault(section, []).append(key)

        return saved

    # ── Отчёт о состоянии памяти ──────────────────────────────────

    def describe_memory(self) -> str:
        lines = [
            "┌─── СОСТОЯНИЕ ПАМЯТИ ─────────────────────────────────────┐",
            self.short.describe(),
            "",
            self.working.describe(),
            "",
            self.long.describe(),
            "└──────────────────────────────────────────────────────────┘",
        ]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# Демо-сценарий: проверяем, что попадает в каждый слой
# ══════════════════════════════════════════════════════════════════

DEMO_SCRIPT = [
    # Ход 1: профиль → долговременная
    ("Привет! Меня зовут Алексей, я backend-разработчик. Предпочитаю Python и Go.",
     None),
    # Ход 2: задача → рабочая
    ("Сейчас работаю над сервисом уведомлений. Стек: FastAPI + Redis + PostgreSQL. "
     "Цель: отправлять push-уведомления 10 млн пользователей за <5 секунд.",
     None),
    # Ход 3: решение → долговременная / decisions
    ("Мы выбрали Celery для очередей задач вместо самописного решения — "
     "потому что нужна надёжность и встроенный retry.",
     None),
    # Ход 4: технический вопрос (триггер автоизвлечения после 4 ходов)
    ("Как лучше партиционировать очередь Celery для 10M пользователей?",
     None),
    # Ход 5: проверка краткосрочной — вопрос по только что сказанному
    ("А можно комбинировать приоритетные и обычные очереди?",
     None),
    # Ход 6: проверка рабочей — ссылка на задачу
    ("Какой размер connection pool для PostgreSQL подойдёт для нашей нагрузки?",
     None),
    # Ход 7: новый факт про пользователя
    ("Кстати, у меня уже есть опыт с Kafka в прошлом проекте — там было 50M событий/день.",
     None),
    # Ход 8: триггер второго автоизвлечения
    ("Стоит ли рассмотреть Kafka вместо Celery для push-уведомлений?",
     None),
    # Ход 9: проверка долговременной — спрашиваем имя
    ("Напомни, как меня зовут и какой у меня основной язык программирования?",
     None),
    # Ход 10: проверка всей памяти — итог задачи
    ("Подведи итог: что мы решили по архитектуре сервиса уведомлений?",
     None),
]


def run_demo(client: Anthropic, model: str, mem_path: Path, verbose: bool = True) -> None:
    """Запустить демо-сценарий и показать состояние памяти."""
    agent = MemoryAgent(
        client=client,
        model=model,
        long_term_path=mem_path,
        auto_extract_every=4,
    )

    if verbose:
        print(f"\n{'═'*70}")
        print("  ДЕМО: Агент с трёхслойной памятью")
        print(f"  Модель: {model} | Долговременная память: {mem_path.name}")
        print(f"{'═'*70}")
        print()
        print("  Легенда автоизвлечения:")
        print("  📥 working   — данные активной задачи")
        print("  👤 profile   — профиль пользователя")
        print("  ✅ decisions — принятые решения")
        print("  📚 knowledge — технические факты")
        print()

    total_in = total_out = 0

    for i, (user_msg, _) in enumerate(DEMO_SCRIPT, 1):
        if verbose:
            print(f"── Ход {i}/{len(DEMO_SCRIPT)} {'─'*50}")
            print(f"  👤 {user_msg[:80]}{'…' if len(user_msg) > 80 else ''}")

        resp = agent.ask(user_msg)
        total_in += resp.input_tokens
        total_out += resp.output_tokens
        cost = calc_cost(model, resp.input_tokens, resp.output_tokens)

        if verbose:
            print(f"  🤖 {resp.text[:100]}{'…' if len(resp.text) > 100 else ''}")
            print(f"  [in={resp.input_tokens:,} out={resp.output_tokens:,} | ${cost:.6f} | {resp.elapsed_sec}s]")

            if resp.auto_saved:
                _print_auto_saved(resp.auto_saved)
            print()

    if verbose:
        print(f"\n{'═'*70}")
        print("  ИТОГ: состояние памяти после диалога")
        print(f"{'═'*70}")
        print(agent.describe_memory())

        total_cost = calc_cost(model, total_in, total_out)
        print(f"\n  Всего: input={total_in:,}  output={total_out:,}  стоимость=${total_cost:.4f}")

        # Показываем, что попало в каждый слой
        _print_layer_analysis(agent)


def _print_auto_saved(saved: dict) -> None:
    working = saved.get("working", [])
    lt = saved.get("long_term", {})

    if working:
        print(f"  ⚡ Автосохранено в working: {', '.join(working)}")
    for section, keys in lt.items():
        if keys:
            icon = {"profile": "👤", "decisions": "✅", "knowledge": "📚"}.get(section, "📝")
            print(f"  ⚡ Автосохранено в {section}: {icon} {', '.join(keys)}")


def _print_layer_analysis(agent: MemoryAgent) -> None:
    print(f"\n{'─'*70}")
    print("  АНАЛИЗ СЛОЁВ: что и куда попало")
    print(f"{'─'*70}")

    print(f"\n  📋 КРАТКОСРОЧНАЯ (текущий диалог):")
    print(f"     {agent.short.turn_count} ходов, ~{agent.short.token_estimate} токенов")
    print(f"     Тип данных: полный текст последних сообщений")
    print(f"     Использование: прямо включается в messages[]")

    print(f"\n  🔧 РАБОЧАЯ (активная задача):")
    wdata = agent.working._data
    if wdata:
        for k, v in wdata.items():
            print(f"     {k}: {str(v)[:60]}")
    else:
        print(f"     (пусто)")
    print(f"     Тип данных: структурированный словарь ключ→значение")
    print(f"     Использование: вставляется в system prompt как блок")

    print(f"\n  🧠 ДОЛГОВРЕМЕННАЯ (профиль + знания):")
    for section in LongTermMemory.SECTIONS:
        data = agent.long.get_section(section)
        icon = {"profile": "👤", "decisions": "✅", "knowledge": "📚"}.get(section, "")
        if data:
            print(f"     {icon} [{section}]:")
            for k, v in data.items():
                print(f"       {k}: {str(v)[:70]}")
        else:
            print(f"     {icon} [{section}]: пусто")
    print(f"     Тип данных: JSON-файл на диске, секции по смыслу")
    print(f"     Использование: вставляется в system prompt перед рабочей памятью")


# ══════════════════════════════════════════════════════════════════
# CLI — интерактивный чат с памятью
# ══════════════════════════════════════════════════════════════════

HELP_TEXT = """\
Команды управления памятью:
  /mem                         — показать всю память
  /remember <section> <k> <v> — сохранить в LTM  (section: profile/decisions/knowledge)
  /forget <section> <key>     — удалить из LTM
  /task <key> <value>          — записать в рабочую память
  /task-clear                  — очистить рабочую память
  /reset                       — сбросить краткосрочную память
  /tokens                      — статистика токенов сессии
  /exit                        — выход"""


def run_interactive(client: Anthropic, model: str, mem_path: Path) -> None:
    """Интерактивный чат с трёхслойной памятью."""
    agent = MemoryAgent(
        client=client,
        model=model,
        long_term_path=mem_path,
        auto_extract_every=4,
    )

    ltm_count = agent.long.total_count()
    print(f"\n  Агент с памятью ({model})")
    print(f"  Долговременная память: {mem_path.name} — {ltm_count} записей")
    print(f"  {HELP_TEXT}\n")

    session_in = session_out = 0
    turn = 0

    while True:
        try:
            user_text = input("Вы> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text or user_text == "/exit":
            break

        # ── Команды управления памятью ────────────────────────────

        if user_text == "/mem":
            print(agent.describe_memory())
            continue

        if user_text.startswith("/remember "):
            parts = user_text[10:].split(None, 2)
            if len(parts) < 3:
                print("  Использование: /remember <section> <key> <value>")
                print(f"  Секции: {', '.join(LongTermMemory.SECTIONS)}")
                continue
            section, key, value = parts[0], parts[1], parts[2]
            try:
                agent.remember(section, key, value)
                print(f"  ✅ Сохранено в LTM [{section}]: {key} = {value}")
            except ValueError as e:
                print(f"  ❌ {e}")
            continue

        if user_text.startswith("/forget "):
            parts = user_text[8:].split(None, 1)
            if len(parts) < 2:
                print("  Использование: /forget <section> <key>")
                continue
            section, key = parts[0], parts[1]
            if agent.forget(section, key):
                print(f"  🗑️  Удалено из LTM [{section}]: {key}")
            else:
                print(f"  ❌ Ключ '{key}' не найден в секции '{section}'")
            continue

        if user_text.startswith("/task "):
            parts = user_text[6:].split(None, 1)
            if len(parts) < 2:
                print("  Использование: /task <key> <value>")
                continue
            key, value = parts[0], parts[1]
            agent.task_set(key, value)
            print(f"  📥 Рабочая память: {key} = {value}")
            continue

        if user_text == "/task-clear":
            agent.task_clear()
            print("  🗑️  Рабочая память очищена")
            continue

        if user_text == "/reset":
            agent.reset_short()
            print("  🔄 Краткосрочная память сброшена")
            continue

        if user_text == "/tokens":
            total_cost = calc_cost(model, session_in, session_out)
            print(f"  Сессия: input={session_in:,}  output={session_out:,}  "
                  f"стоимость=${total_cost:.6f}  ходов={turn}")
            continue

        # ── Основной диалог ───────────────────────────────────────

        sys.stdout.write("\nАгент> ")
        sys.stdout.flush()

        def on_token(chunk: str) -> None:
            sys.stdout.write(chunk)
            sys.stdout.flush()

        resp = agent.ask(user_text, on_token=on_token)
        turn += 1
        session_in += resp.input_tokens
        session_out += resp.output_tokens
        cost = calc_cost(model, resp.input_tokens, resp.output_tokens)

        print(f"\n  [in={resp.input_tokens:,} out={resp.output_tokens:,} | ${cost:.6f} | {resp.elapsed_sec}s]")

        if resp.auto_saved:
            _print_auto_saved(resp.auto_saved)

        print()


# ══════════════════════════════════════════════════════════════════
# CLI entry points
# ══════════════════════════════════════════════════════════════════

MEM_DIR = Path(__file__).parent / "memory_store"


def main() -> None:
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise SystemExit("Missing API key: set ANTHROPIC_API_KEY or API_KEY in .env")

    client = Anthropic(api_key=api_key)
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    print("╔══════════════════════════════════════════════════════╗")
    print("║  День 11. Модель памяти ассистента: три слоя        ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"\nМодель: {model}")
    prices = PRICING.get(model, DEFAULT_PRICING)
    print(f"Цены: ${prices['input']}/1M input, ${prices['output']}/1M output")

    # Режим: demo или interactive
    mode = os.getenv("DAY11_MODE", "demo")

    MEM_DIR.mkdir(exist_ok=True)
    mem_path = MEM_DIR / "ltm_demo.json"

    if mode == "interactive":
        run_interactive(client, model, mem_path)
    else:
        run_demo(client, model, mem_path)


if __name__ == "__main__":
    main()

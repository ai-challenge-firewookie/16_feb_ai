"""День 12. Персонализация ассистента.

Строится поверх модели памяти из дня 11.

Профиль пользователя состоит из трёх групп настроек:
  identity    — кто он, имя, профессия, уровень
  style       — как общаться: тон, длина, язык, формат
  constraints — что запрещено: темы, предположения, оговорки

Персонализация работает на двух уровнях:
  1. Предустановленные профили (persona presets) — загружаются одной командой
  2. Точечные настройки — /pref key value меняет отдельный параметр

Демо сравнивает ответы ОДНОГО вопроса для трёх разных профилей:
  - Junior Python developer  (подробно, примеры кода, объяснения)
  - Senior architect         (кратко, trade-offs, без очевидного)
  - Non-technical manager    (без кода, бизнес-язык, аналогии)

Для каждого профиля видно, как меняется system-prompt и ответ.
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
from day11 import LongTermMemory, ShortTermMemory, MEM_DIR


# ══════════════════════════════════════════════════════════════════
# Модель профиля
# ══════════════════════════════════════════════════════════════════

@dataclass
class UserProfile:
    """Полный профиль пользователя.

    identity   — кто пользователь
    style      — как отвечать
    constraints — чего избегать
    """
    # identity
    name: str = ""
    role: str = ""           # backend dev, manager, student, …
    level: str = ""          # junior / mid / senior / non-technical
    domain: str = ""         # python, devops, data science, …

    # style
    tone: str = "neutral"            # formal / casual / neutral
    response_length: str = "medium"  # short / medium / long / adaptive
    language: str = "ru"             # ru / en / mixed
    format: str = "prose"            # prose / bullets / code-first / mixed
    detail_level: str = "normal"     # basic / normal / deep

    # constraints
    no_topics: list[str] = field(default_factory=list)   # темы-табу
    no_assume: list[str] = field(default_factory=list)   # не предполагать
    always_do: list[str] = field(default_factory=list)   # всегда делать

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "level": self.level,
            "domain": self.domain,
            "tone": self.tone,
            "response_length": self.response_length,
            "language": self.language,
            "format": self.format,
            "detail_level": self.detail_level,
            "no_topics": self.no_topics,
            "no_assume": self.no_assume,
            "always_do": self.always_do,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UserProfile":
        p = cls()
        for k, v in data.items():
            if hasattr(p, k):
                setattr(p, k, v)
        return p

    def set_field(self, key: str, value: str) -> bool:
        """Изменить одно поле профиля. Возвращает False если поле не найдено."""
        list_fields = {"no_topics", "no_assume", "always_do"}
        if key in list_fields:
            current = getattr(self, key, [])
            if value.startswith("-") and len(value) > 1:
                item = value[1:].strip()
                if item in current:
                    current.remove(item)
            else:
                if value not in current:
                    current.append(value)
            setattr(self, key, current)
            return True
        if hasattr(self, key):
            setattr(self, key, value)
            return True
        return False

    def summary(self) -> str:
        parts = []
        if self.name:
            parts.append(f"Имя: {self.name}")
        if self.role:
            parts.append(f"Роль: {self.role}")
        if self.level:
            parts.append(f"Уровень: {self.level}")
        if self.domain:
            parts.append(f"Область: {self.domain}")
        parts.append(f"Тон: {self.tone} | Длина: {self.response_length} | Формат: {self.format} | Детали: {self.detail_level}")
        if self.no_topics:
            parts.append(f"Запрещённые темы: {', '.join(self.no_topics)}")
        if self.no_assume:
            parts.append(f"Не предполагать: {', '.join(self.no_assume)}")
        if self.always_do:
            parts.append(f"Всегда: {', '.join(self.always_do)}")
        return "\n  ".join(parts)


# ══════════════════════════════════════════════════════════════════
# Предустановленные профили (presets)
# ══════════════════════════════════════════════════════════════════

PRESETS: dict[str, UserProfile] = {
    "junior-python": UserProfile(
        role="junior Python developer",
        level="junior",
        domain="python",
        tone="casual",
        response_length="long",
        language="ru",
        format="code-first",
        detail_level="deep",
        always_do=[
            "давай рабочие примеры кода",
            "объясняй термины при первом упоминании",
            "указывай распространённые ошибки новичков",
        ],
    ),
    "senior-arch": UserProfile(
        role="senior software architect",
        level="senior",
        domain="system design",
        tone="formal",
        response_length="short",
        language="ru",
        format="bullets",
        detail_level="deep",
        no_assume=["очевидные вещи уже известны"],
        always_do=[
            "называй trade-offs для каждого решения",
            "указывай на масштабируемость и узкие места",
        ],
    ),
    "manager": UserProfile(
        role="non-technical product manager",
        level="non-technical",
        domain="product management",
        tone="formal",
        response_length="medium",
        language="ru",
        format="prose",
        detail_level="basic",
        no_topics=["конкретный синтаксис кода", "низкоуровневые детали реализации"],
        always_do=[
            "используй бизнес-аналогии вместо технических терминов",
            "давай оценку рисков и выгод",
            "говори о влиянии на сроки и ресурсы",
        ],
    ),
    "student": UserProfile(
        role="computer science student",
        level="junior",
        domain="computer science",
        tone="casual",
        response_length="medium",
        language="ru",
        format="mixed",
        detail_level="normal",
        always_do=[
            "используй аналогии из реального мира",
            "давай задания для самостоятельной практики",
        ],
    ),
    "default": UserProfile(
        tone="neutral",
        response_length="medium",
        language="ru",
        format="prose",
        detail_level="normal",
    ),
}

PRESET_NAMES = list(PRESETS.keys())


# ══════════════════════════════════════════════════════════════════
# Построение системного промпта из профиля
# ══════════════════════════════════════════════════════════════════

def build_persona_block(profile: UserProfile) -> str:
    """Сформировать блок персонализации для system prompt."""
    lines: list[str] = []

    # Кто пользователь
    identity_parts = []
    if profile.name:
        identity_parts.append(f"имя: {profile.name}")
    if profile.role:
        identity_parts.append(f"роль: {profile.role}")
    if profile.level:
        identity_parts.append(f"уровень: {profile.level}")
    if profile.domain:
        identity_parts.append(f"область: {profile.domain}")

    if identity_parts:
        lines.append(f"Пользователь — {', '.join(identity_parts)}.")

    # Стиль ответа
    style_map = {
        "tone": {
            "formal": "строго формальный тон",
            "casual": "дружелюбный неформальный тон",
            "neutral": "нейтральный тон",
        },
        "response_length": {
            "short": "отвечай ОЧЕНЬ кратко — 2-4 предложения максимум",
            "medium": "отвечай умеренно — не растягивай, не сокращай лишнего",
            "long": "давай развёрнутые ответы с деталями и примерами",
            "adaptive": "подбирай длину под сложность вопроса",
        },
        "format": {
            "prose": "отвечай связным текстом, без лишних списков",
            "bullets": "структурируй ответы списками и подпунктами",
            "code-first": "начинай с рабочего кода, затем объясняй",
            "mixed": "комбинируй текст, списки и код по ситуации",
        },
        "detail_level": {
            "basic": "объясняй просто, избегай жаргона и технических деталей",
            "normal": "предполагай базовые знания темы",
            "deep": "можешь погружаться в детали, упоминать крайние случаи",
        },
    }

    style_lines = []
    for field_name, variants in style_map.items():
        val = getattr(profile, field_name, "")
        if val in variants:
            style_lines.append(variants[val])

    if profile.language == "en":
        style_lines.append("отвечай на английском языке")
    elif profile.language == "mixed":
        style_lines.append("технические термины пиши на английском, остальное — по-русски")
    else:
        style_lines.append("отвечай на русском языке")

    if style_lines:
        lines.append("Стиль: " + "; ".join(style_lines) + ".")

    # Ограничения
    if profile.no_topics:
        lines.append("НЕ затрагивай: " + ", ".join(profile.no_topics) + ".")

    if profile.no_assume:
        lines.append("НЕ предполагай, что известно: " + ", ".join(profile.no_assume) + ".")

    # Обязательные правила
    if profile.always_do:
        lines.append("ВСЕГДА:")
        for rule in profile.always_do:
            lines.append(f"  • {rule}")

    return "\n".join(lines)


def build_system_prompt(profile: UserProfile, extra_context: str | None = None) -> str:
    """Собрать полный system prompt: база + персонализация + контекст."""
    persona_block = build_persona_block(profile)

    parts = [
        "Ты — AI-ассистент с персонализацией под конкретного пользователя.",
        "",
        "=== ПРОФИЛЬ ПОЛЬЗОВАТЕЛЯ ===",
        persona_block,
    ]

    if extra_context:
        parts += ["", "=== КОНТЕКСТ ===", extra_context]

    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════
# Персонализированный агент
# ══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PersonaResponse:
    text: str
    input_tokens: int
    output_tokens: int
    elapsed_sec: float
    profile_name: str
    system_prompt_len: int


class PersonalizedAgent:
    """Агент с профилем пользователя.

    Профиль можно:
      - загрузить из preset: agent.load_preset("junior-python")
      - загрузить с диска: agent.load_profile()
      - изменить поле: agent.set_pref("tone", "casual")
      - сохранить: agent.save_profile()
    """

    def __init__(
        self,
        client: Anthropic,
        model: str,
        profile_path: Path,
        max_tokens: int = 800,
        short_term_turns: int = 15,
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._profile_path = profile_path

        self.profile = UserProfile()
        self.short = ShortTermMemory(max_turns=short_term_turns)

        self._load_profile()

    # ── Управление профилем ───────────────────────────────────────

    def load_preset(self, name: str) -> None:
        """Загрузить предустановленный профиль."""
        if name not in PRESETS:
            raise ValueError(f"Пресет '{name}' не найден. Доступны: {', '.join(PRESETS)}")
        # Сохраняем имя если было задано
        saved_name = self.profile.name
        self.profile = UserProfile(**PRESETS[name].to_dict())
        if saved_name and not self.profile.name:
            self.profile.name = saved_name
        self.save_profile()

    def set_pref(self, key: str, value: str) -> bool:
        """Изменить одно поле профиля."""
        result = self.profile.set_field(key, value)
        if result:
            self.save_profile()
        return result

    def save_profile(self) -> None:
        self._profile_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._profile_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(self.profile.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(self._profile_path)

    def _load_profile(self) -> None:
        if not self._profile_path.exists():
            return
        text = self._profile_path.read_text(encoding="utf-8").strip()
        if not text:
            return
        self.profile = UserProfile.from_dict(json.loads(text))

    # ── Основной метод ────────────────────────────────────────────

    def ask(self, user_text: str, on_token=None) -> PersonaResponse:
        self.short.add_user(user_text)
        system = build_system_prompt(self.profile)
        messages = self.short.get_messages()

        t0 = time.perf_counter()

        if on_token:
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
            text = full_text
            inp, out = final.usage.input_tokens, final.usage.output_tokens
        else:
            resp = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                messages=messages,
            )
            text = resp.content[0].text
            inp, out = resp.usage.input_tokens, resp.usage.output_tokens

        elapsed = round(time.perf_counter() - t0, 2)
        self.short.add_assistant(text)

        return PersonaResponse(
            text=text,
            input_tokens=inp,
            output_tokens=out,
            elapsed_sec=elapsed,
            profile_name=self.profile.role or "default",
            system_prompt_len=len(system),
        )

    def reset_dialog(self) -> None:
        self.short.clear()


# ══════════════════════════════════════════════════════════════════
# Демо: один вопрос → три профиля
# ══════════════════════════════════════════════════════════════════

# Вопросы для сравнения профилей
DEMO_QUESTIONS = [
    "Что такое микросервисная архитектура и когда её стоит использовать?",
    "Объясни, что такое race condition и как с этим бороться.",
    "Почему база данных работает медленно и как это исправить?",
]

DEMO_PROFILES = ["junior-python", "senior-arch", "manager"]


def run_demo(client: Anthropic, model: str, profiles_dir: Path) -> None:
    """Запустить демо: один вопрос — три профиля — сравнение ответов."""
    profiles_dir.mkdir(parents=True, exist_ok=True)

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  День 12. Персонализация: один вопрос — три профиля         ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"\nМодель: {model}")
    prices = PRICING.get(model, DEFAULT_PRICING)
    print(f"Цены: ${prices['input']}/1M input, ${prices['output']}/1M output\n")

    # Показываем system-prompt для каждого профиля ДО запросов
    print(f"{'═'*70}")
    print("  ПРОФИЛИ: как персонализация влияет на system prompt")
    print(f"{'═'*70}")
    for preset_name in DEMO_PROFILES:
        profile = PRESETS[preset_name]
        system = build_system_prompt(profile)
        print(f"\n── [{preset_name}] ──────────────────────────────────────")
        print(f"  Характеристики: {profile.summary()}")
        print(f"  System prompt ({len(system)} симв.):")
        # Показываем блок персонализации
        for line in system.split("\n")[3:]:  # пропускаем шапку
            print(f"    {line}")

    total_in = total_out = 0

    for q_idx, question in enumerate(DEMO_QUESTIONS, 1):
        print(f"\n{'═'*70}")
        print(f"  ВОПРОС {q_idx}: «{question}»")
        print(f"{'═'*70}")

        results: list[tuple[str, PersonaResponse]] = []

        for preset_name in DEMO_PROFILES:
            agent = PersonalizedAgent(
                client=client,
                model=model,
                profile_path=profiles_dir / f"{preset_name}.json",
            )
            agent.load_preset(preset_name)

            print(f"\n  [{preset_name}] — запрос...")
            resp = agent.ask(question)
            results.append((preset_name, resp))
            total_in += resp.input_tokens
            total_out += resp.output_tokens

        # Печатаем ответы для сравнения
        for preset_name, resp in results:
            cost = calc_cost(model, resp.input_tokens, resp.output_tokens)
            print(f"\n{'─'*70}")
            print(f"  [{preset_name}]  in={resp.input_tokens:,} | ${cost:.6f} | {resp.elapsed_sec}s")
            print(f"{'─'*70}")
            # Ответ с отступом
            for line in resp.text.split("\n"):
                print(f"  {line}")

    # Итоговая статистика
    total_cost = calc_cost(model, total_in, total_out)
    print(f"\n{'═'*70}")
    print("  ИТОГ")
    print(f"{'═'*70}")
    print(f"  Вопросов: {len(DEMO_QUESTIONS)}  Профилей: {len(DEMO_PROFILES)}")
    print(f"  Всего: input={total_in:,}  output={total_out:,}  стоимость=${total_cost:.4f}")
    _print_profile_diff_analysis()


def _print_profile_diff_analysis() -> None:
    """Аналитика: что именно меняет каждый профиль."""
    print(f"\n{'─'*70}")
    print("  ЧТО МЕНЯЕТ ПРОФИЛЬ В ОТВЕТАХ")
    print(f"{'─'*70}")

    comparisons = [
        ("junior-python",
         "Длинный ответ с кодом, объяснением терминов, частыми ошибками новичков",
         ["рабочий код", "объяснение", "пример", "ошибки"]),
        ("senior-arch",
         "Краткий структурированный ответ, trade-offs, масштабируемость",
         ["trade-off", "масштаб", "узкое место", "компромисс"]),
        ("manager",
         "Бизнес-язык, аналогии, риски/выгоды, влияние на сроки. Без кода",
         ["риск", "выгода", "аналог", "ресурс", "срок"]),
    ]

    for name, description, keywords in comparisons:
        preset = PRESETS[name]
        print(f"\n  [{name}]")
        print(f"    Ожидаемый стиль: {description}")
        print(f"    Ключевые параметры профиля:")
        print(f"      tone={preset.tone}, length={preset.response_length}, "
              f"format={preset.format}, detail={preset.detail_level}")
        if preset.always_do:
            print(f"      always_do: {'; '.join(preset.always_do[:2])}")
        if preset.no_topics:
            print(f"      no_topics: {', '.join(preset.no_topics)}")


# ══════════════════════════════════════════════════════════════════
# Интерактивный режим
# ══════════════════════════════════════════════════════════════════

INTERACTIVE_HELP = """\
Команды:
  /profile              — показать текущий профиль
  /preset <name>        — загрузить пресет ({presets})
  /pref <key> <value>   — изменить поле профиля
  /pref-list            — список всех полей профиля
  /system               — показать текущий system prompt
  /reset                — сбросить диалог (профиль сохраняется)
  /tokens               — статистика сессии
  /exit                 — выход"""


def run_interactive(client: Anthropic, model: str, profile_path: Path) -> None:
    """Интерактивный чат с персонализацией."""
    agent = PersonalizedAgent(
        client=client,
        model=model,
        profile_path=profile_path,
    )

    print(f"\n  Персонализированный агент ({model})")
    print(f"  Профиль: {profile_path.name}")
    print(f"  {INTERACTIVE_HELP.format(presets=', '.join(PRESET_NAMES))}\n")

    if agent.profile.role:
        print(f"  Загружен профиль: {agent.profile.summary()}\n")
    else:
        print(f"  Профиль пустой. Введите /preset <name> для загрузки.\n")

    session_in = session_out = 0
    turn = 0

    PROFILE_FIELDS = [
        "name", "role", "level", "domain",
        "tone", "response_length", "language", "format", "detail_level",
        "no_topics", "no_assume", "always_do",
    ]

    while True:
        try:
            user_text = input("Вы> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text or user_text == "/exit":
            break

        if user_text == "/profile":
            print(f"\n  Текущий профиль:\n  {agent.profile.summary()}\n")
            continue

        if user_text.startswith("/preset "):
            name = user_text[8:].strip()
            try:
                agent.load_preset(name)
                agent.reset_dialog()
                print(f"  ✅ Профиль загружен: {name}")
                print(f"  {agent.profile.summary()}\n")
            except ValueError as e:
                print(f"  ❌ {e}\n")
            continue

        if user_text.startswith("/pref "):
            parts = user_text[6:].split(None, 1)
            if len(parts) < 2:
                print("  Использование: /pref <key> <value>")
                print(f"  Поля: {', '.join(PROFILE_FIELDS)}\n")
                continue
            key, value = parts[0], parts[1]
            if agent.set_pref(key, value):
                print(f"  ✅ {key} = {value}\n")
            else:
                print(f"  ❌ Поле '{key}' не найдено.\n")
                print(f"  Доступные поля: {', '.join(PROFILE_FIELDS)}\n")
            continue

        if user_text == "/pref-list":
            print(f"\n  Поля профиля:")
            for f in PROFILE_FIELDS:
                val = getattr(agent.profile, f, "")
                print(f"    {f}: {val}")
            print()
            continue

        if user_text == "/system":
            system = build_system_prompt(agent.profile)
            print(f"\n  System prompt ({len(system)} симв.):")
            print("  " + "\n  ".join(system.split("\n")))
            print()
            continue

        if user_text == "/reset":
            agent.reset_dialog()
            print("  🔄 Диалог сброшен, профиль сохранён\n")
            continue

        if user_text == "/tokens":
            cost = calc_cost(model, session_in, session_out)
            print(f"  input={session_in:,}  output={session_out:,}  "
                  f"${cost:.6f}  ходов={turn}\n")
            continue

        # Основной диалог
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

        print(f"\n  [in={resp.input_tokens:,} out={resp.output_tokens:,} | ${cost:.6f} | "
              f"{resp.elapsed_sec}s | system={resp.system_prompt_len}c]\n")


# ══════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════

PROFILES_DIR = MEM_DIR / "profiles"


def main() -> None:
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise SystemExit("Missing API key: set ANTHROPIC_API_KEY or API_KEY in .env")

    client = Anthropic(api_key=api_key)
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    mode = os.getenv("DAY12_MODE", "demo")
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)

    if mode == "interactive":
        profile_path = PROFILES_DIR / "user.json"
        run_interactive(client, model, profile_path)
    else:
        run_demo(client, model, PROFILES_DIR)


if __name__ == "__main__":
    main()

"""День 13. Состояние задачи: конечный автомат (Task State Machine).

Задача проходит через формализованные этапы:

  ┌──────────┐    ┌───────────┐    ┌────────────┐    ┌──────────┐    ┌──────┐
  │  IDLE    │───▶│ PLANNING  │───▶│ EXECUTION  │───▶│VALIDATION│───▶│ DONE │
  └──────────┘    └───────────┘    └────────────┘    └──────────┘    └──────┘
       ▲               │                 │                 │
       └───────────────┴─────────────────┴─── PAUSED ──────┘
                                                   │
                                                   ▼ (resume → возврат в прошлый этап)

Каждый этап знает:
  - своё имя и цель
  - ожидаемое действие от пользователя
  - разрешённые переходы
  - список шагов, которые нужно выполнить

Агент получает в system prompt полный snapshot состояния →
  не нужно заново объяснять контекст при паузе/возобновлении.

Команды:
  /state            — текущее состояние FSM
  /next             — перейти на следующий этап
  /pause            — поставить задачу на паузу
  /resume           — продолжить с паузы
  /restart          — начать задачу заново
  /step-done        — отметить текущий шаг выполненным
  /step-add <text>  — добавить шаг в текущий этап
  /task-new <name>  — создать новую задачу
  /save             — сохранить состояние на диск
  /load             — загрузить сохранённое состояние
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

from day8 import calc_cost, PRICING, DEFAULT_PRICING


# ══════════════════════════════════════════════════════════════════
# Конечный автомат: состояния и переходы
# ══════════════════════════════════════════════════════════════════

class Stage(str, Enum):
    IDLE       = "idle"
    PLANNING   = "planning"
    EXECUTION  = "execution"
    VALIDATION = "validation"
    DONE       = "done"
    PAUSED     = "paused"


# Допустимые переходы из каждого состояния
TRANSITIONS: dict[Stage, list[Stage]] = {
    Stage.IDLE:       [Stage.PLANNING],
    Stage.PLANNING:   [Stage.EXECUTION, Stage.PAUSED],
    Stage.EXECUTION:  [Stage.VALIDATION, Stage.PLANNING, Stage.PAUSED],
    Stage.VALIDATION: [Stage.DONE, Stage.EXECUTION, Stage.PAUSED],
    Stage.DONE:       [Stage.IDLE],
    Stage.PAUSED:     [],   # восстанавливается в _paused_from
}

# Метаданные каждого этапа
STAGE_META: dict[Stage, dict[str, str]] = {
    Stage.IDLE: {
        "goal": "Ожидание задачи",
        "expected_action": "Опишите задачу для начала работы",
        "icon": "💤",
    },
    Stage.PLANNING: {
        "goal": "Разбить задачу на шаги, уточнить требования, определить критерии успеха",
        "expected_action": "Уточните требования или подтвердите план (скажи 'готово к выполнению')",
        "icon": "📋",
    },
    Stage.EXECUTION: {
        "goal": "Выполнить шаги плана один за другим",
        "expected_action": "Подтвердите выполнение шага или сообщите о проблеме",
        "icon": "⚙️",
    },
    Stage.VALIDATION: {
        "goal": "Проверить результат: соответствие требованиям, тесты, финальный review",
        "expected_action": "Подтвердите результат или укажите что нужно доработать",
        "icon": "✅",
    },
    Stage.DONE: {
        "goal": "Задача завершена",
        "expected_action": "Начните новую задачу или введите /restart",
        "icon": "🏁",
    },
    Stage.PAUSED: {
        "goal": "Задача приостановлена",
        "expected_action": "Введите /resume для продолжения",
        "icon": "⏸️",
    },
}


# ══════════════════════════════════════════════════════════════════
# Шаг задачи
# ══════════════════════════════════════════════════════════════════

@dataclass
class TaskStep:
    text: str
    done: bool = False
    stage: str = Stage.PLANNING.value  # на каком этапе создан

    def to_dict(self) -> dict:
        return {"text": self.text, "done": self.done, "stage": self.stage}

    @classmethod
    def from_dict(cls, d: dict) -> "TaskStep":
        return cls(text=d["text"], done=d.get("done", False), stage=d.get("stage", "planning"))


# ══════════════════════════════════════════════════════════════════
# Состояние задачи
# ══════════════════════════════════════════════════════════════════

@dataclass
class TaskState:
    """Полное состояние задачи — сохраняется на диск."""

    name: str = "Новая задача"
    description: str = ""
    stage: Stage = Stage.IDLE
    paused_from: Stage | None = None    # откуда была пауза

    steps: list[TaskStep] = field(default_factory=list)

    # История переходов: [(timestamp, from, to, comment)]
    transitions_log: list[tuple[str, str, str, str]] = field(default_factory=list)

    # Ключевые решения и артефакты, накопленные в диалоге
    decisions: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)    # ссылки на файлы, URLs

    created_at: str = ""
    updated_at: str = ""

    # ── Доступ к шагам ────────────────────────────────────────────

    @property
    def current_step(self) -> TaskStep | None:
        for s in self.steps:
            if not s.done:
                return s
        return None

    @property
    def done_steps(self) -> list[TaskStep]:
        return [s for s in self.steps if s.done]

    @property
    def pending_steps(self) -> list[TaskStep]:
        return [s for s in self.steps if not s.done]

    @property
    def progress_pct(self) -> int:
        if not self.steps:
            return 0
        return int(len(self.done_steps) / len(self.steps) * 100)

    # ── Переходы ──────────────────────────────────────────────────

    def can_transition_to(self, target: Stage) -> bool:
        if target == Stage.PAUSED:
            return self.stage not in (Stage.IDLE, Stage.DONE, Stage.PAUSED)
        if self.stage == Stage.PAUSED:
            # Из паузы можно только в paused_from
            return target == self.paused_from
        return target in TRANSITIONS.get(self.stage, [])

    def transition(self, target: Stage, comment: str = "") -> None:
        if not self.can_transition_to(target):
            raise ValueError(
                f"Переход {self.stage.value} → {target.value} недопустим. "
                f"Разрешены: {[t.value for t in TRANSITIONS.get(self.stage, [])]}"
            )
        now = _now()
        self.transitions_log.append((now, self.stage.value, target.value, comment))

        if target == Stage.PAUSED:
            self.paused_from = self.stage
        elif self.stage == Stage.PAUSED:
            self.paused_from = None

        self.stage = target
        self.updated_at = now

    def pause(self, comment: str = "") -> None:
        self.transition(Stage.PAUSED, comment)

    def resume(self) -> None:
        if self.stage != Stage.PAUSED:
            raise ValueError("Задача не на паузе")
        if not self.paused_from:
            raise ValueError("Неизвестно, откуда возобновить")
        self.transition(self.paused_from, "resume")

    def advance(self) -> Stage:
        """Перейти к следующему этапу по умолчанию."""
        defaults = {
            Stage.IDLE:       Stage.PLANNING,
            Stage.PLANNING:   Stage.EXECUTION,
            Stage.EXECUTION:  Stage.VALIDATION,
            Stage.VALIDATION: Stage.DONE,
        }
        if self.stage == Stage.PAUSED:
            self.resume()
            return self.stage
        target = defaults.get(self.stage)
        if not target:
            raise ValueError(f"Нет следующего этапа из {self.stage.value}")
        self.transition(target, "advance")
        return self.stage

    # ── Шаги ──────────────────────────────────────────────────────

    def add_step(self, text: str) -> TaskStep:
        step = TaskStep(text=text, stage=self.stage.value)
        self.steps.append(step)
        self.updated_at = _now()
        return step

    def complete_current_step(self) -> TaskStep | None:
        step = self.current_step
        if step:
            step.done = True
            self.updated_at = _now()
        return step

    # ── Сериализация ──────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "stage": self.stage.value,
            "paused_from": self.paused_from.value if self.paused_from else None,
            "steps": [s.to_dict() for s in self.steps],
            "transitions_log": self.transitions_log,
            "decisions": self.decisions,
            "artifacts": self.artifacts,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TaskState":
        s = cls()
        s.name = d.get("name", "Задача")
        s.description = d.get("description", "")
        s.stage = Stage(d.get("stage", "idle"))
        pf = d.get("paused_from")
        s.paused_from = Stage(pf) if pf else None
        s.steps = [TaskStep.from_dict(sd) for sd in d.get("steps", [])]
        s.transitions_log = d.get("transitions_log", [])
        s.decisions = d.get("decisions", [])
        s.artifacts = d.get("artifacts", [])
        s.created_at = d.get("created_at", "")
        s.updated_at = d.get("updated_at", "")
        return s

    # ── Форматирование ────────────────────────────────────────────

    def as_context_block(self) -> str:
        """Краткий снапшот для system prompt — без лишних деталей."""
        meta = STAGE_META[self.stage]
        lines = [
            f"[ЗАДАЧА: {self.name}]",
            f"Этап: {meta['icon']} {self.stage.value.upper()}  ({meta['goal']})",
        ]
        if self.stage == Stage.PAUSED and self.paused_from:
            lines.append(f"Пауза с этапа: {self.paused_from.value}")

        if self.description:
            lines.append(f"Описание: {self.description}")

        if self.steps:
            lines.append(f"Прогресс: {len(self.done_steps)}/{len(self.steps)} шагов ({self.progress_pct}%)")
            cur = self.current_step
            if cur:
                lines.append(f"Текущий шаг: → {cur.text}")
            pending = self.pending_steps
            if len(pending) > 1:
                lines.append(f"Следующие шаги: {'; '.join(s.text for s in pending[1:3])}")

        if self.decisions:
            lines.append("Принятые решения: " + "; ".join(self.decisions[-3:]))

        lines.append(f"Ожидаемое действие: {meta['expected_action']}")
        return "\n".join(lines)

    def full_status(self) -> str:
        """Полный статус для команды /state."""
        meta = STAGE_META[self.stage]
        sep = "─" * 60
        lines = [
            f"┌{sep}┐",
            f"  {meta['icon']}  ЗАДАЧА: {self.name}",
            f"  Этап:    {self.stage.value.upper()}",
            f"  Цель:    {meta['goal']}",
        ]
        if self.stage == Stage.PAUSED and self.paused_from:
            lines.append(f"  Пауза с: {self.paused_from.value}")
        if self.description:
            lines.append(f"  Описание: {self.description}")

        lines.append(f"  Прогресс: {self.progress_pct}%  "
                     f"({len(self.done_steps)} / {len(self.steps)} шагов)")

        if self.steps:
            lines.append(f"  Шаги:")
            for i, s in enumerate(self.steps, 1):
                mark = "✓" if s.done else ("→" if s == self.current_step else "○")
                lines.append(f"    {mark} [{i}] {s.text}")

        if self.decisions:
            lines.append(f"  Решения:")
            for d in self.decisions:
                lines.append(f"    • {d}")

        if self.artifacts:
            lines.append(f"  Артефакты:")
            for a in self.artifacts:
                lines.append(f"    📎 {a}")

        if self.transitions_log:
            lines.append(f"  История переходов:")
            for ts, frm, to, comment in self.transitions_log[-4:]:
                c = f" ({comment})" if comment else ""
                lines.append(f"    {ts}  {frm} → {to}{c}")

        lines.append(f"  Ожидаемое действие: {meta['expected_action']}")
        lines.append(f"└{sep}┘")
        return "\n".join(lines)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ══════════════════════════════════════════════════════════════════
# Системный промпт с учётом FSM
# ══════════════════════════════════════════════════════════════════

SYSTEM_BASE = """\
Ты — AI-ассистент, который ведёт задачу через формализованные этапы.
У каждого этапа есть цель и ожидаемое действие.

Правила поведения:
• На этапе PLANNING — помогай разбить задачу на шаги, уточняй требования.
• На этапе EXECUTION — веди по шагам, фокусируйся на текущем шаге.
• На этапе VALIDATION — проверяй результат против требований.
• После PAUSED и RESUME — сразу напомни где остановились, без лишних вступлений.
• Никогда не пропускай этапы самовольно — только по команде /next.
• Держи ответы краткими, по делу, на русском языке.

Текущее состояние задачи:
{task_snapshot}"""


def build_system(task: TaskState) -> str:
    return SYSTEM_BASE.format(task_snapshot=task.as_context_block())


# ══════════════════════════════════════════════════════════════════
# Агент с FSM
# ══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FSMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    elapsed_sec: float
    stage: Stage
    auto_transition: Stage | None = None   # если агент сам перешёл
    auto_steps: list[str] = field(default_factory=list)  # шаги, извлечённые авто


class TaskFSMAgent:
    """Агент с конечным автоматом состояний задачи.

    Хранит:
      - task: TaskState — формальное состояние (FSM)
      - history: список сообщений текущего этапа
      - Сообщения НЕ сбрасываются при переходе между этапами —
        агент помнит весь диалог, но system prompt обновляется
    """

    # Промпт для авто-извлечения шагов из planning-ответа
    _STEPS_EXTRACT = """\
Из ответа ассистента ниже извлеки список конкретных шагов задачи.
Верни JSON: {{"steps": ["шаг 1", "шаг 2", ...]}}
Без markdown. Только шаги с конкретными действиями (не абстракции).
Если шагов нет — верни {{"steps": []}}.

Ответ ассистента:
{text}"""

    def __init__(
        self,
        client: Anthropic,
        model: str,
        save_path: Path,
        max_tokens: int = 900,
    ) -> None:
        self._client = client
        self._model = model
        self._save_path = save_path
        self._max_tokens = max_tokens

        self.task = TaskState(created_at=_now(), updated_at=_now())
        self._history: list[dict[str, str]] = []

        self._load()

    # ── Основной диалог ───────────────────────────────────────────

    def ask(self, user_text: str, on_token=None) -> FSMResponse:
        # Если IDLE и пришёл текст — автоматически начинаем planning
        auto_transition = None
        if self.task.stage == Stage.IDLE:
            self.task.name = user_text[:60]
            self.task.description = user_text
            self.task.transition(Stage.PLANNING, "auto: task started")
            auto_transition = Stage.PLANNING
            self.save()

        self._history.append({"role": "user", "content": user_text})

        system = build_system(self.task)
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

        # На этапе PLANNING — пробуем авто-извлечь шаги если их ещё нет
        auto_steps: list[str] = []
        if self.task.stage == Stage.PLANNING and not self.task.steps and len(self._history) >= 2:
            auto_steps = self._try_extract_steps(text)
            for s in auto_steps:
                self.task.add_step(s)

        self.save()

        return FSMResponse(
            text=text,
            input_tokens=inp,
            output_tokens=out,
            elapsed_sec=elapsed,
            stage=self.task.stage,
            auto_transition=auto_transition,
            auto_steps=auto_steps,
        )

    def _try_extract_steps(self, assistant_text: str) -> list[str]:
        """Попытка извлечь шаги из ответа ассистента."""
        try:
            resp = self._client.messages.create(
                model=self._model,
                max_tokens=300,
                messages=[{
                    "role": "user",
                    "content": self._STEPS_EXTRACT.format(text=assistant_text[:1500]),
                }],
            )
            raw = resp.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)
            return [s.strip() for s in data.get("steps", []) if s.strip()]
        except Exception:
            return []

    # ── Управление FSM ────────────────────────────────────────────

    def next_stage(self, comment: str = "") -> Stage:
        old = self.task.stage
        new = self.task.advance()
        self.save()
        return new

    def pause(self, comment: str = "") -> None:
        self.task.pause(comment)
        self.save()

    def resume(self) -> Stage:
        self.task.resume()
        self.save()
        return self.task.stage

    def restart(self, name: str = "") -> None:
        self.task = TaskState(
            name=name or "Новая задача",
            created_at=_now(),
            updated_at=_now(),
        )
        self._history.clear()
        self.save()

    def step_done(self) -> TaskStep | None:
        step = self.task.complete_current_step()
        self.save()
        return step

    def step_add(self, text: str) -> TaskStep:
        step = self.task.add_step(text)
        self.save()
        return step

    def add_decision(self, text: str) -> None:
        self.task.decisions.append(text)
        self.task.updated_at = _now()
        self.save()

    def add_artifact(self, text: str) -> None:
        self.task.artifacts.append(text)
        self.task.updated_at = _now()
        self.save()

    # ── Персистентность ───────────────────────────────────────────

    def save(self) -> None:
        self._save_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "task": self.task.to_dict(),
            "history": self._history,
        }
        tmp = self._save_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self._save_path)

    def _load(self) -> None:
        if not self._save_path.exists():
            return
        text = self._save_path.read_text(encoding="utf-8").strip()
        if not text:
            return
        data = json.loads(text)
        self.task = TaskState.from_dict(data.get("task", {}))
        self._history = data.get("history", [])


# ══════════════════════════════════════════════════════════════════
# Демо-сценарий: показываем паузу и возобновление
# ══════════════════════════════════════════════════════════════════

DEMO_SCRIPT = [
    # Этап IDLE→PLANNING
    ("Нужно написать REST API на FastAPI для управления задачами: CRUD для tasks + auth.",
     "start"),
    # Уточнение в planning
    ("Аутентификация через JWT. База — PostgreSQL. Нужны тесты с pytest.",
     "planning"),
    # Переход в EXECUTION
    (None, "next"),
    # Первый шаг
    ("Начнём с настройки проекта и зависимостей?",
     "execution"),
    # Отмечаем шаг и идём дальше
    (None, "step-done"),
    # ПАУЗА
    (None, "pause"),
    # Имитируем паузу — выводим состояние
    (None, "state"),
    # RESUME
    (None, "resume"),
    # Продолжаем без повторных объяснений
    ("Отлично, давай реализуем модель Task в SQLAlchemy.",
     "execution"),
    # Ещё шаг
    (None, "step-done"),
    # Переход в VALIDATION
    (None, "next"),
    # Валидация
    ("Проверь что у нас есть: модели, роуты, JWT, тесты. Что ещё нужно?",
     "validation"),
    # Завершение
    (None, "next"),
]


def run_demo(client: Anthropic, model: str, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Чистый старт
    if save_path.exists():
        save_path.unlink()

    agent = TaskFSMAgent(client=client, model=model, save_path=save_path)

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  День 13. Task State Machine — демо                         ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"\nМодель: {model}")
    print()

    total_in = total_out = 0

    for step_num, (msg, action) in enumerate(DEMO_SCRIPT, 1):
        stage_icon = STAGE_META[agent.task.stage]["icon"]
        print(f"\n{'━'*65}")
        print(f"  Шаг {step_num:02d} | {stage_icon} {agent.task.stage.value.upper()}")
        print(f"{'━'*65}")

        if action == "state":
            print(agent.task.full_status())
            continue

        if action == "next":
            old = agent.task.stage
            new = agent.next_stage("demo: advance")
            print(f"  /next: {old.value} → {new.value}")
            continue

        if action == "pause":
            agent.pause("demo: пауза для демонстрации")
            print(f"  /pause — задача на паузе (была на {agent.task.paused_from.value})")
            print(f"  Состояние сохранено в {save_path.name}")
            print(f"\n  ⏸️  Имитируем длительную паузу...")
            time.sleep(0.5)
            continue

        if action == "resume":
            resumed_to = agent.resume()
            print(f"  /resume → вернулись на этап {resumed_to.value}")
            print(f"  Агент получит полный snapshot состояния в system prompt")
            continue

        if action == "step-done":
            step = agent.step_done()
            if step:
                print(f"  /step-done: ✓ «{step.text}»")
                nxt = agent.task.current_step
                if nxt:
                    print(f"  Следующий шаг: → {nxt.text}")
            else:
                print(f"  /step-done: нет активных шагов")
            continue

        # Обычный диалог
        print(f"  👤 {msg}")
        resp = agent.ask(msg)
        total_in += resp.input_tokens
        total_out += resp.output_tokens
        cost = calc_cost(model, resp.input_tokens, resp.output_tokens)

        print(f"  🤖 {resp.text[:200]}{'…' if len(resp.text) > 200 else ''}")
        print(f"  [in={resp.input_tokens:,} out={resp.output_tokens:,} | ${cost:.6f} | {resp.elapsed_sec}s]")

        if resp.auto_transition:
            print(f"  ⚡ Авто-переход: IDLE → {resp.auto_transition.value}")

        if resp.auto_steps:
            print(f"  📋 Авто-извлечено шагов: {len(resp.auto_steps)}")
            for s in resp.auto_steps:
                print(f"     • {s}")

    # Финальное состояние
    print(f"\n{'━'*65}")
    print("  ФИНАЛЬНОЕ СОСТОЯНИЕ FSM")
    print(f"{'━'*65}")
    print(agent.task.full_status())

    total_cost = calc_cost(model, total_in, total_out)
    print(f"\n  Всего: input={total_in:,}  output={total_out:,}  стоимость=${total_cost:.4f}")

    # Показываем переходы между этапами
    print(f"\n  История переходов FSM:")
    for ts, frm, to, comment in agent.task.transitions_log:
        print(f"    {ts}  {frm:12s} → {to:12s}  {comment}")


# ══════════════════════════════════════════════════════════════════
# Интерактивный режим
# ══════════════════════════════════════════════════════════════════

HELP = """\
Команды FSM:
  /state              — полный статус задачи
  /next               — перейти к следующему этапу
  /pause              — поставить на паузу
  /resume             — продолжить с паузы
  /restart [name]     — новая задача
  /step-done          — отметить текущий шаг выполненным
  /step-add <text>    — добавить шаг
  /decision <text>    — зафиксировать решение
  /artifact <text>    — добавить ссылку/файл
  /save               — сохранить (авто при каждом ходу)
  /tokens             — статистика токенов
  /exit               — выход"""


def run_interactive(client: Anthropic, model: str, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    agent = TaskFSMAgent(client=client, model=model, save_path=save_path)

    stage_meta = STAGE_META[agent.task.stage]
    print(f"\n  Task FSM Agent ({model})")
    print(f"  Сохранение: {save_path.name}")
    print(f"  Текущий этап: {stage_meta['icon']} {agent.task.stage.value.upper()}")
    if agent.task.stage != Stage.IDLE:
        print(f"  Задача: {agent.task.name}")
        print(f"  Прогресс: {agent.task.progress_pct}%")
        print(f"  Ожидается: {stage_meta['expected_action']}")
    print(f"\n{HELP}\n")

    session_in = session_out = 0

    while True:
        stage_icon = STAGE_META[agent.task.stage]["icon"]
        try:
            prompt = f"\n[{stage_icon} {agent.task.stage.value}]> "
            user_text = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text or user_text == "/exit":
            break

        if user_text == "/state":
            print(agent.task.full_status())
            continue

        if user_text == "/next":
            try:
                old = agent.task.stage
                new = agent.next_stage()
                meta = STAGE_META[new]
                print(f"  {old.value} → {meta['icon']} {new.value}")
                print(f"  Цель: {meta['goal']}")
                print(f"  Ожидается: {meta['expected_action']}")
            except ValueError as e:
                print(f"  ❌ {e}")
            continue

        if user_text == "/pause":
            try:
                agent.pause()
                print(f"  ⏸️  Задача приостановлена (с этапа {agent.task.paused_from.value})")
                print(f"  Состояние сохранено. /resume для продолжения.")
            except ValueError as e:
                print(f"  ❌ {e}")
            continue

        if user_text == "/resume":
            try:
                resumed = agent.resume()
                meta = STAGE_META[resumed]
                print(f"  ▶️  Продолжаем: {meta['icon']} {resumed.value}")
                print(f"  {meta['expected_action']}")
                cur = agent.task.current_step
                if cur:
                    print(f"  Текущий шаг: → {cur.text}")
            except ValueError as e:
                print(f"  ❌ {e}")
            continue

        if user_text.startswith("/restart"):
            name = user_text[8:].strip() or "Новая задача"
            agent.restart(name)
            print(f"  🔄 Новая задача: «{name}»")
            print(f"  Опишите задачу для начала.")
            continue

        if user_text == "/step-done":
            step = agent.step_done()
            if step:
                print(f"  ✓ Выполнен: «{step.text}»")
                nxt = agent.task.current_step
                if nxt:
                    print(f"  Следующий: → {nxt.text}")
                else:
                    print(f"  Все шаги выполнены. /next для перехода.")
            else:
                print(f"  Нет активного шага.")
            continue

        if user_text.startswith("/step-add "):
            text = user_text[10:].strip()
            if text:
                step = agent.step_add(text)
                print(f"  + Шаг добавлен: {step.text}")
            continue

        if user_text.startswith("/decision "):
            text = user_text[10:].strip()
            if text:
                agent.add_decision(text)
                print(f"  ✅ Решение зафиксировано: {text}")
            continue

        if user_text.startswith("/artifact "):
            text = user_text[10:].strip()
            if text:
                agent.add_artifact(text)
                print(f"  📎 Артефакт добавлен: {text}")
            continue

        if user_text == "/save":
            agent.save()
            print(f"  💾 Сохранено: {save_path}")
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

        stage_info = f"{STAGE_META[resp.stage]['icon']} {resp.stage.value}"
        print(f"\n  [in={resp.input_tokens:,} out={resp.output_tokens:,} | "
              f"${cost:.6f} | {resp.elapsed_sec}s | {stage_info}]")

        if resp.auto_transition:
            print(f"  ⚡ Задача начата: IDLE → {resp.auto_transition.value}")

        if resp.auto_steps:
            print(f"  📋 Шаги авто-извлечены ({len(resp.auto_steps)}):")
            for s in resp.auto_steps:
                print(f"     • {s}")

        meta = STAGE_META[agent.task.stage]
        print(f"  💡 {meta['expected_action']}")


# ══════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════

FSM_DIR = Path(__file__).parent / "fsm_tasks"


def main() -> None:
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise SystemExit("Missing API key: set ANTHROPIC_API_KEY or API_KEY in .env")

    client = Anthropic(api_key=api_key)
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    mode = os.getenv("DAY13_MODE", "demo")

    FSM_DIR.mkdir(exist_ok=True)

    if mode == "interactive":
        run_interactive(client, model, FSM_DIR / "current_task.json")
    else:
        run_demo(client, model, FSM_DIR / "demo_task.json")


if __name__ == "__main__":
    main()

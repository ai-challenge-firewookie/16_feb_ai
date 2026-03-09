"""День 15. Контролируемые переходы состояний.

Ассистент с контролируемым жизненным циклом задачи.

Отличие от дня 13 (базовый FSM):
  - ассистент САМИ проверяет допустимость переходов
  - нельзя делать реализацию до утверждённого плана
  - нельзя завершать без валидации
  - попытки «перепрыгнуть» этап блокируются с объяснением
  - каждый этап имеет условия входа (gate conditions)
  - ассистент-валидатор проверяет ответ на нарушение FSM

Жизненный цикл:

  ┌──────┐    ┌──────────┐    ┌──────────┐    ┌────────────┐    ┌──────┐
  │ IDLE │───▶│ PLANNING │───▶│ APPROVED │───▶│ EXECUTION  │───▶│      │
  └──────┘    └──────────┘    └──────────┘    └────────────┘    │      │
                                                    │           │VALID-│
                                                    ▼           │ATION │
                                              ┌──────────┐     │      │───▶ DONE
                                              │  PAUSED  │     └──────┘
                                              └──────────┘

Новые этапы:
  - APPROVED: план утверждён пользователем, можно начинать реализацию
  - Валидатор: отдельный LLM-вызов проверяет, не нарушает ли ответ FSM

Команды:
  /state              — полный статус задачи
  /next               — перейти к следующему этапу
  /approve            — утвердить план (PLANNING → APPROVED)
  /reject             — отклонить план (вернуться в PLANNING)
  /pause              — поставить задачу на паузу
  /resume             — продолжить с паузы
  /restart            — начать задачу заново
  /step-done          — отметить текущий шаг выполненным
  /step-add <text>    — добавить шаг
  /force <stage>      — попытка принудительного перехода (демо блокировки)
  /transitions        — показать граф допустимых переходов
  /log                — история переходов
  /tokens             — статистика токенов
  /exit               — выход
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
# Конечный автомат: состояния, переходы, gate conditions
# ══════════════════════════════════════════════════════════════════

class Stage(str, Enum):
    IDLE       = "idle"
    PLANNING   = "planning"
    APPROVED   = "approved"       # НОВЫЙ: план утверждён
    EXECUTION  = "execution"
    VALIDATION = "validation"
    DONE       = "done"
    PAUSED     = "paused"


# Допустимые переходы из каждого состояния
TRANSITIONS: dict[Stage, list[Stage]] = {
    Stage.IDLE:       [Stage.PLANNING],
    Stage.PLANNING:   [Stage.APPROVED, Stage.PAUSED],
    Stage.APPROVED:   [Stage.EXECUTION, Stage.PLANNING, Stage.PAUSED],
    Stage.EXECUTION:  [Stage.VALIDATION, Stage.PAUSED],
    Stage.VALIDATION: [Stage.DONE, Stage.EXECUTION, Stage.PAUSED],
    Stage.DONE:       [Stage.IDLE],
    Stage.PAUSED:     [],   # восстанавливается в _paused_from
}


# Условия входа в этап (gate conditions)
GATE_CONDITIONS: dict[Stage, dict[str, str]] = {
    Stage.PLANNING: {
        "description": "Задача описана",
        "check": "task_has_description",
    },
    Stage.APPROVED: {
        "description": "План содержит хотя бы 1 шаг",
        "check": "has_steps",
    },
    Stage.EXECUTION: {
        "description": "План утверждён пользователем (этап APPROVED пройден)",
        "check": "plan_approved",
    },
    Stage.VALIDATION: {
        "description": "Все шаги выполнены или есть хотя бы один выполненный",
        "check": "has_done_steps",
    },
    Stage.DONE: {
        "description": "Валидация пройдена",
        "check": "validation_passed",
    },
}


# Метаданные каждого этапа
STAGE_META: dict[Stage, dict[str, str]] = {
    Stage.IDLE: {
        "goal": "Ожидание задачи",
        "expected_action": "Опишите задачу для начала работы",
        "icon": "💤",
    },
    Stage.PLANNING: {
        "goal": "Разбить задачу на шаги, уточнить требования",
        "expected_action": "Уточните требования или /approve для утверждения плана",
        "icon": "📋",
    },
    Stage.APPROVED: {
        "goal": "План утверждён — готовы к реализации",
        "expected_action": "/next для перехода к выполнению",
        "icon": "✅",
    },
    Stage.EXECUTION: {
        "goal": "Выполнить шаги плана один за другим",
        "expected_action": "Подтвердите выполнение шага (/step-done) или сообщите о проблеме",
        "icon": "⚙️",
    },
    Stage.VALIDATION: {
        "goal": "Проверить результат: тесты, review, соответствие требованиям",
        "expected_action": "Подтвердите результат (/next → done) или укажите доработки",
        "icon": "🔍",
    },
    Stage.DONE: {
        "goal": "Задача завершена",
        "expected_action": "Начните новую задачу или /restart",
        "icon": "🏁",
    },
    Stage.PAUSED: {
        "goal": "Задача приостановлена",
        "expected_action": "/resume для продолжения",
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
    stage: str = Stage.PLANNING.value

    def to_dict(self) -> dict:
        return {"text": self.text, "done": self.done, "stage": self.stage}

    @classmethod
    def from_dict(cls, d: dict) -> TaskStep:
        return cls(text=d["text"], done=d.get("done", False), stage=d.get("stage", "planning"))


# ══════════════════════════════════════════════════════════════════
# Результат проверки перехода
# ══════════════════════════════════════════════════════════════════

@dataclass
class TransitionResult:
    allowed: bool
    from_stage: Stage
    to_stage: Stage
    reason: str
    gate_failed: str | None = None   # какой gate condition не прошёл


# ══════════════════════════════════════════════════════════════════
# Состояние задачи
# ══════════════════════════════════════════════════════════════════

@dataclass
class TaskState:
    """Полное состояние задачи с контролируемыми переходами."""

    name: str = "Новая задача"
    description: str = ""
    stage: Stage = Stage.IDLE
    paused_from: Stage | None = None

    steps: list[TaskStep] = field(default_factory=list)
    plan_approved: bool = False          # утверждён ли план
    validation_passed: bool = False      # пройдена ли валидация

    transitions_log: list[tuple[str, str, str, str]] = field(default_factory=list)
    blocked_attempts: list[tuple[str, str, str, str]] = field(default_factory=list)  # заблокированные попытки

    decisions: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)

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

    # ── Проверка gate conditions ──────────────────────────────────

    def check_gate(self, target: Stage) -> tuple[bool, str]:
        """Проверяет gate condition для перехода в target.
        Возвращает (passed, reason)."""
        gate = GATE_CONDITIONS.get(target)
        if not gate:
            return True, ""

        check = gate["check"]

        if check == "task_has_description":
            if not self.description.strip():
                return False, f"Требуется: {gate['description']}"
            return True, ""

        if check == "has_steps":
            if not self.steps:
                return False, f"Требуется: {gate['description']}"
            return True, ""

        if check == "plan_approved":
            if not self.plan_approved:
                return False, f"Требуется: {gate['description']}. Используйте /approve"
            return True, ""

        if check == "has_done_steps":
            if not self.done_steps:
                return False, f"Требуется: {gate['description']}"
            return True, ""

        if check == "validation_passed":
            if not self.validation_passed:
                return False, f"Требуется: {gate['description']}"
            return True, ""

        return True, ""

    # ── Переходы с контролем ──────────────────────────────────────

    def can_transition_to(self, target: Stage) -> TransitionResult:
        """Полная проверка: допустимость перехода + gate conditions."""
        # Пауза — особый случай
        if target == Stage.PAUSED:
            if self.stage in (Stage.IDLE, Stage.DONE, Stage.PAUSED):
                return TransitionResult(
                    allowed=False, from_stage=self.stage, to_stage=target,
                    reason=f"Нельзя поставить на паузу из {self.stage.value}",
                )
            return TransitionResult(
                allowed=True, from_stage=self.stage, to_stage=target,
                reason="OK",
            )

        # Из паузы — только в paused_from
        if self.stage == Stage.PAUSED:
            if target == self.paused_from:
                return TransitionResult(
                    allowed=True, from_stage=self.stage, to_stage=target,
                    reason=f"Возобновление на этап {target.value}",
                )
            return TransitionResult(
                allowed=False, from_stage=self.stage, to_stage=target,
                reason=f"Из паузы можно вернуться только в {self.paused_from.value if self.paused_from else '?'}",
            )

        # Проверка допустимости перехода по графу
        allowed_targets = TRANSITIONS.get(self.stage, [])
        if target not in allowed_targets:
            return TransitionResult(
                allowed=False, from_stage=self.stage, to_stage=target,
                reason=f"Переход {self.stage.value} → {target.value} недопустим. "
                       f"Разрешены: {[t.value for t in allowed_targets]}",
            )

        # Проверка gate condition
        gate_ok, gate_reason = self.check_gate(target)
        if not gate_ok:
            return TransitionResult(
                allowed=False, from_stage=self.stage, to_stage=target,
                reason=gate_reason,
                gate_failed=gate_reason,
            )

        return TransitionResult(
            allowed=True, from_stage=self.stage, to_stage=target,
            reason="OK",
        )

    def transition(self, target: Stage, comment: str = "") -> TransitionResult:
        """Выполнить переход с полной проверкой."""
        result = self.can_transition_to(target)
        now = _now()

        if not result.allowed:
            self.blocked_attempts.append((now, self.stage.value, target.value, result.reason))
            self.updated_at = now
            return result

        self.transitions_log.append((now, self.stage.value, target.value, comment))

        if target == Stage.PAUSED:
            self.paused_from = self.stage
        elif self.stage == Stage.PAUSED:
            self.paused_from = None

        self.stage = target
        self.updated_at = now
        return result

    def pause(self, comment: str = "") -> TransitionResult:
        return self.transition(Stage.PAUSED, comment)

    def resume(self) -> TransitionResult:
        if self.stage != Stage.PAUSED:
            return TransitionResult(
                allowed=False, from_stage=self.stage, to_stage=self.stage,
                reason="Задача не на паузе",
            )
        if not self.paused_from:
            return TransitionResult(
                allowed=False, from_stage=self.stage, to_stage=self.stage,
                reason="Неизвестно, откуда возобновить",
            )
        return self.transition(self.paused_from, "resume")

    def approve_plan(self) -> TransitionResult:
        """Утвердить план: PLANNING → APPROVED."""
        if self.stage != Stage.PLANNING:
            return TransitionResult(
                allowed=False, from_stage=self.stage, to_stage=Stage.APPROVED,
                reason=f"Утвердить план можно только на этапе PLANNING (сейчас: {self.stage.value})",
            )
        self.plan_approved = True
        return self.transition(Stage.APPROVED, "plan approved by user")

    def reject_plan(self) -> TransitionResult:
        """Отклонить план — остаёмся в PLANNING для доработки."""
        if self.stage != Stage.PLANNING:
            return TransitionResult(
                allowed=False, from_stage=self.stage, to_stage=Stage.PLANNING,
                reason=f"Отклонить план можно только на этапе PLANNING",
            )
        self.plan_approved = False
        self.steps.clear()
        return TransitionResult(
            allowed=True, from_stage=Stage.PLANNING, to_stage=Stage.PLANNING,
            reason="План отклонён — требуется доработка",
        )

    def pass_validation(self) -> None:
        """Отметить валидацию пройденной."""
        self.validation_passed = True
        self.updated_at = _now()

    def advance(self) -> TransitionResult:
        """Перейти к следующему этапу по умолчанию."""
        defaults = {
            Stage.IDLE:       Stage.PLANNING,
            Stage.PLANNING:   Stage.APPROVED,
            Stage.APPROVED:   Stage.EXECUTION,
            Stage.EXECUTION:  Stage.VALIDATION,
            Stage.VALIDATION: Stage.DONE,
        }
        if self.stage == Stage.PAUSED:
            return self.resume()
        target = defaults.get(self.stage)
        if not target:
            return TransitionResult(
                allowed=False, from_stage=self.stage, to_stage=self.stage,
                reason=f"Нет следующего этапа из {self.stage.value}",
            )
        return self.transition(target, "advance")

    def force_transition(self, target: Stage) -> TransitionResult:
        """Попытка принудительного перехода — ВСЕГДА проверяется."""
        return self.transition(target, "FORCED attempt")

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
            "plan_approved": self.plan_approved,
            "validation_passed": self.validation_passed,
            "steps": [s.to_dict() for s in self.steps],
            "transitions_log": self.transitions_log,
            "blocked_attempts": self.blocked_attempts,
            "decisions": self.decisions,
            "artifacts": self.artifacts,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TaskState:
        s = cls()
        s.name = d.get("name", "Задача")
        s.description = d.get("description", "")
        s.stage = Stage(d.get("stage", "idle"))
        pf = d.get("paused_from")
        s.paused_from = Stage(pf) if pf else None
        s.plan_approved = d.get("plan_approved", False)
        s.validation_passed = d.get("validation_passed", False)
        s.steps = [TaskStep.from_dict(sd) for sd in d.get("steps", [])]
        s.transitions_log = d.get("transitions_log", [])
        s.blocked_attempts = d.get("blocked_attempts", [])
        s.decisions = d.get("decisions", [])
        s.artifacts = d.get("artifacts", [])
        s.created_at = d.get("created_at", "")
        s.updated_at = d.get("updated_at", "")
        return s

    # ── Форматирование ────────────────────────────────────────────

    def as_context_block(self) -> str:
        """Снапшот для system prompt."""
        meta = STAGE_META[self.stage]
        lines = [
            f"[ЗАДАЧА: {self.name}]",
            f"Этап: {meta['icon']} {self.stage.value.upper()}  ({meta['goal']})",
        ]
        if self.stage == Stage.PAUSED and self.paused_from:
            lines.append(f"Пауза с этапа: {self.paused_from.value}")

        if self.description:
            lines.append(f"Описание: {self.description}")

        lines.append(f"План утверждён: {'да' if self.plan_approved else 'нет'}")
        lines.append(f"Валидация пройдена: {'да' if self.validation_passed else 'нет'}")

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

        # Добавляем gate conditions для текущего этапа
        next_stages = TRANSITIONS.get(self.stage, [])
        if next_stages:
            lines.append(f"\nУсловия для перехода:")
            for ns in next_stages:
                gate = GATE_CONDITIONS.get(ns)
                if gate:
                    ok, _ = self.check_gate(ns)
                    status = "✅" if ok else "❌"
                    lines.append(f"  → {ns.value}: {status} {gate['description']}")

        return "\n".join(lines)

    def full_status(self) -> str:
        """Полный статус для /state."""
        meta = STAGE_META[self.stage]
        sep = "─" * 60
        lines = [
            f"┌{sep}┐",
            f"  {meta['icon']}  ЗАДАЧА: {self.name}",
            f"  Этап:        {self.stage.value.upper()}",
            f"  Цель:        {meta['goal']}",
            f"  План утв.:   {'✅ да' if self.plan_approved else '❌ нет'}",
            f"  Валидация:   {'✅ да' if self.validation_passed else '❌ нет'}",
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

        # Gate conditions
        next_stages = TRANSITIONS.get(self.stage, [])
        if next_stages:
            lines.append(f"  Условия перехода:")
            for ns in next_stages:
                gate = GATE_CONDITIONS.get(ns)
                if gate:
                    ok, reason = self.check_gate(ns)
                    status = "✅" if ok else "❌"
                    lines.append(f"    → {ns.value}: {status} {gate['description']}")
                    if not ok and reason:
                        lines.append(f"      {reason}")
                else:
                    lines.append(f"    → {ns.value}: ✅ (без условий)")

        if self.transitions_log:
            lines.append(f"  История переходов:")
            for ts, frm, to, comment in self.transitions_log[-5:]:
                c = f" ({comment})" if comment else ""
                lines.append(f"    {ts}  {frm} → {to}{c}")

        if self.blocked_attempts:
            lines.append(f"  ⛔ Заблокированные попытки ({len(self.blocked_attempts)}):")
            for ts, frm, to, reason in self.blocked_attempts[-3:]:
                lines.append(f"    {ts}  {frm} → {to}: {reason}")

        lines.append(f"  Ожидается: {meta['expected_action']}")
        lines.append(f"└{sep}┘")
        return "\n".join(lines)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ══════════════════════════════════════════════════════════════════
# Граф переходов (для /transitions)
# ══════════════════════════════════════════════════════════════════

def format_transitions_graph() -> str:
    """Отображает граф допустимых переходов с gate conditions."""
    lines = [
        "  ГРАФ ДОПУСТИМЫХ ПЕРЕХОДОВ",
        "",
        "  ┌──────┐    ┌──────────┐    ┌──────────┐    ┌───────────┐    ┌──────────┐    ┌──────┐",
        "  │ IDLE │───▶│ PLANNING │───▶│ APPROVED │───▶│ EXECUTION │───▶│VALIDATION│───▶│ DONE │",
        "  └──────┘    └──────────┘    └──────────┘    └───────────┘    └──────────┘    └──────┘",
        "                  │               │                │                │",
        "                  ▼               ▼                ▼                ▼",
        "              ┌────────────────────────────────────────────────────────┐",
        "              │                     PAUSED                             │",
        "              └────────────────────────────────────────────────────────┘",
        "",
        "  Gate conditions (условия входа):",
    ]
    for stage, gate in GATE_CONDITIONS.items():
        meta = STAGE_META[stage]
        lines.append(f"    {meta['icon']} {stage.value:12s} ← {gate['description']}")

    lines.append("")
    lines.append("  Ключевое ограничение:")
    lines.append("    ⛔ EXECUTION невозможен без прохождения APPROVED")
    lines.append("    ⛔ DONE невозможен без прохождения VALIDATION")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# Системный промпт с контролируемым FSM
# ══════════════════════════════════════════════════════════════════

SYSTEM_BASE = """\
Ты — AI-ассистент, который ведёт задачу через строго контролируемые этапы.
У каждого этапа есть цель, ожидаемое действие и УСЛОВИЯ ВХОДА (gate conditions).

═══ ЖЁСТКИЕ ПРАВИЛА ПЕРЕХОДОВ ═══

1. НЕЛЬЗЯ начинать реализацию (EXECUTION) без утверждённого плана (APPROVED).
2. НЕЛЬЗЯ завершить задачу (DONE) без валидации (VALIDATION).
3. НЕЛЬЗЯ перепрыгивать этапы — только последовательно через граф.
4. Если пользователь просит перейти к реализации без плана — ОТКАЖИ и объясни.
5. Если пользователь просит пропустить валидацию — ОТКАЖИ и объясни.

═══ ЭТАПЫ ═══

IDLE → PLANNING: начало работы над задачей
PLANNING → APPROVED: план утверждён пользователем (/approve)
APPROVED → EXECUTION: начинаем реализацию (/next)
EXECUTION → VALIDATION: все шаги выполнены (/next)
VALIDATION → DONE: результат проверен (/next)

На паузу можно встать с любого рабочего этапа (/pause), вернуться (/resume).

═══ ПОВЕДЕНИЕ ПО ЭТАПАМ ═══

• PLANNING: помогай разбить задачу на шаги, уточняй требования. НЕ ПИШИ КОД.
• APPROVED: кратко подтверди план, жди /next для начала реализации.
• EXECUTION: веди по шагам, фокусируйся на текущем шаге. ПИШИ КОД ТОЛЬКО ЗДЕСЬ.
• VALIDATION: проверяй результат против требований. НЕ ДОБАВЛЯЙ НОВЫЙ ФУНКЦИОНАЛ.
• PAUSED + RESUME: сразу напомни где остановились, без лишних вступлений.

═══ ФОРМАТ ОТКАЗА (при попытке нарушить FSM) ═══

⛔ Не могу выполнить: [причина]
📌 Текущий этап: [этап] — [что нужно сделать]
💡 Следующий шаг: [что пользователь должен сделать]

═══ ТЕКУЩЕЕ СОСТОЯНИЕ ═══

{task_snapshot}"""


def build_system(task: TaskState) -> str:
    return SYSTEM_BASE.format(task_snapshot=task.as_context_block())


# ══════════════════════════════════════════════════════════════════
# Валидатор ответа (проверяет нарушение FSM)
# ══════════════════════════════════════════════════════════════════

VALIDATOR_PROMPT = """\
Ты — валидатор FSM (конечного автомата задачи).
Проверь ответ ассистента на нарушение текущего этапа.

Текущий этап: {stage}
План утверждён: {plan_approved}

Правила:
- На этапе PLANNING ассистент НЕ должен писать код (только план/шаги).
- На этапе EXECUTION ассистент должен работать по шагам.
- На этапе VALIDATION ассистент НЕ должен добавлять новый функционал.
- Ассистент НЕ должен предлагать перепрыгнуть этапы.

Ответ ассистента:
{response}

Верни JSON (без markdown):
{{"violation": true/false, "reason": "причина нарушения или OK"}}"""


# ══════════════════════════════════════════════════════════════════
# Ответ агента
# ══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ControlledFSMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    elapsed_sec: float
    stage: Stage
    auto_transition: Stage | None = None
    auto_steps: list[str] = field(default_factory=list)
    violation: bool = False
    violation_reason: str = ""


# ══════════════════════════════════════════════════════════════════
# Агент с контролируемым FSM
# ══════════════════════════════════════════════════════════════════

class ControlledFSMAgent:
    """Агент с контролируемыми переходами состояний.

    Отличия от базового FSM (день 13):
      - gate conditions для каждого этапа
      - ассистент-валидатор проверяет ответы на нарушение FSM
      - все попытки нарушить FSM логируются
      - отдельный этап APPROVED между PLANNING и EXECUTION
    """

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
        validate_responses: bool = True,
    ) -> None:
        self._client = client
        self._model = model
        self._save_path = save_path
        self._max_tokens = max_tokens
        self._validate = validate_responses

        self.task = TaskState(created_at=_now(), updated_at=_now())
        self._history: list[dict[str, str]] = []

        self._load()

    # ── Основной диалог ───────────────────────────────────────────

    def ask(self, user_text: str, on_token=None) -> ControlledFSMResponse:
        auto_transition = None
        if self.task.stage == Stage.IDLE:
            self.task.name = user_text[:60]
            self.task.description = user_text
            result = self.task.transition(Stage.PLANNING, "auto: task started")
            if result.allowed:
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

        # Авто-извлечение шагов на этапе PLANNING
        auto_steps: list[str] = []
        if self.task.stage == Stage.PLANNING and not self.task.steps and len(self._history) >= 2:
            auto_steps = self._try_extract_steps(text)
            for s in auto_steps:
                self.task.add_step(s)

        # Валидация ответа
        violation = False
        violation_reason = ""
        if self._validate and self.task.stage not in (Stage.IDLE, Stage.DONE, Stage.PAUSED):
            violation, violation_reason = self._validate_response(text)

        self.save()

        return ControlledFSMResponse(
            text=text,
            input_tokens=inp,
            output_tokens=out,
            elapsed_sec=elapsed,
            stage=self.task.stage,
            auto_transition=auto_transition,
            auto_steps=auto_steps,
            violation=violation,
            violation_reason=violation_reason,
        )

    def _try_extract_steps(self, assistant_text: str) -> list[str]:
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

    def _validate_response(self, response_text: str) -> tuple[bool, str]:
        """Проверяет ответ ассистента на нарушение FSM."""
        try:
            prompt = VALIDATOR_PROMPT.format(
                stage=self.task.stage.value,
                plan_approved="да" if self.task.plan_approved else "нет",
                response=response_text[:1500],
            )
            resp = self._client.messages.create(
                model=self._model,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)
            return data.get("violation", False), data.get("reason", "")
        except Exception:
            return False, ""

    # ── Управление FSM ────────────────────────────────────────────

    def next_stage(self, comment: str = "") -> TransitionResult:
        return self.task.advance()

    def approve_plan(self) -> TransitionResult:
        result = self.task.approve_plan()
        self.save()
        return result

    def reject_plan(self) -> TransitionResult:
        result = self.task.reject_plan()
        self.save()
        return result

    def pause(self, comment: str = "") -> TransitionResult:
        result = self.task.pause(comment)
        self.save()
        return result

    def resume(self) -> TransitionResult:
        result = self.task.resume()
        self.save()
        return result

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

    def pass_validation(self) -> None:
        self.task.pass_validation()
        self.save()

    def force_stage(self, stage_str: str) -> TransitionResult:
        """Попытка принудительного перехода — демо блокировки."""
        try:
            target = Stage(stage_str.lower())
        except ValueError:
            return TransitionResult(
                allowed=False, from_stage=self.task.stage, to_stage=self.task.stage,
                reason=f"Неизвестный этап: {stage_str}. Допустимые: {[s.value for s in Stage]}",
            )
        result = self.task.force_transition(target)
        self.save()
        return result

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
# Демо-сценарий
# ══════════════════════════════════════════════════════════════════

DEMO_SCRIPT = [
    # 1. IDLE → PLANNING (авто)
    {"action": "msg", "text": "Нужно написать CLI-утилиту для конвертации CSV → JSON с фильтрацией по колонкам."},

    # 2. Уточнение на этапе PLANNING
    {"action": "msg", "text": "Язык: Python. Нужна поддержка stdin и файлов. Вывод — stdout."},

    # 3. ПОПЫТКА перепрыгнуть в EXECUTION без утверждения плана
    {"action": "force", "target": "execution", "label": "⛔ Попытка перейти в EXECUTION без APPROVED"},

    # 4. ПОПЫТКА через /next (PLANNING → APPROVED: нужны шаги)
    {"action": "next", "label": "PLANNING → APPROVED (авто)"},

    # 5. Утверждение плана
    {"action": "approve", "label": "✅ Утверждение плана"},

    # 6. Показать состояние после утверждения
    {"action": "state"},

    # 7. Переход в EXECUTION
    {"action": "next", "label": "APPROVED → EXECUTION"},

    # 8. Работа на этапе EXECUTION
    {"action": "msg", "text": "Начнём с парсинга аргументов CLI через argparse."},

    # 9. Отмечаем шаг
    {"action": "step-done"},

    # 10. ПОПЫТКА перепрыгнуть в DONE без валидации
    {"action": "force", "target": "done", "label": "⛔ Попытка пропустить VALIDATION"},

    # 11. Пауза
    {"action": "pause"},

    # 12. Состояние на паузе
    {"action": "state"},

    # 13. Возобновление
    {"action": "resume"},

    # 14. Ещё работа
    {"action": "msg", "text": "Теперь реализуем чтение CSV и конвертацию в JSON."},
    {"action": "step-done"},

    # 15. Переход в VALIDATION
    {"action": "next", "label": "EXECUTION → VALIDATION"},

    # 16. Валидация
    {"action": "msg", "text": "Проверь: парсинг CLI, чтение CSV, фильтрация колонок, вывод JSON. Всё на месте?"},

    # 17. Отметить валидацию пройденной
    {"action": "pass-validation"},

    # 18. Завершение
    {"action": "next", "label": "VALIDATION → DONE"},

    # 19. Финальное состояние
    {"action": "state"},
]


def run_demo(client: Anthropic, model: str, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.exists():
        save_path.unlink()

    agent = ControlledFSMAgent(
        client=client, model=model, save_path=save_path,
        validate_responses=True,
    )

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  День 15. Контролируемые переходы состояний — демо          ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"\nМодель: {model}")
    print(f"\n{format_transitions_graph()}")
    print()

    total_in = total_out = 0

    for step_num, step in enumerate(DEMO_SCRIPT, 1):
        action = step["action"]
        stage_icon = STAGE_META[agent.task.stage]["icon"]

        print(f"\n{'━' * 65}")
        print(f"  Шаг {step_num:02d} | {stage_icon} {agent.task.stage.value.upper()}"
              + (f"  | {step.get('label', '')}" if step.get('label') else ""))
        print(f"{'━' * 65}")

        if action == "state":
            print(agent.task.full_status())
            continue

        if action == "next":
            result = agent.next_stage()
            if result.allowed:
                meta = STAGE_META[agent.task.stage]
                print(f"  ✅ {result.from_stage.value} → {meta['icon']} {result.to_stage.value}")
                print(f"  Цель: {meta['goal']}")
            else:
                print(f"  ⛔ ЗАБЛОКИРОВАНО: {result.reason}")
            agent.save()
            continue

        if action == "approve":
            result = agent.approve_plan()
            if result.allowed:
                print(f"  ✅ План утверждён! → {Stage.APPROVED.value}")
            else:
                print(f"  ⛔ {result.reason}")
            continue

        if action == "reject":
            result = agent.reject_plan()
            print(f"  {'✅' if result.allowed else '⛔'} {result.reason}")
            continue

        if action == "force":
            target = step["target"]
            print(f"  🔨 Попытка принудительного перехода → {target}")
            result = agent.force_stage(target)
            if result.allowed:
                print(f"  ✅ Переход выполнен: {result.from_stage.value} → {result.to_stage.value}")
            else:
                print(f"  ⛔ ЗАБЛОКИРОВАНО: {result.reason}")
                if result.gate_failed:
                    print(f"  📌 Gate condition: {result.gate_failed}")
            continue

        if action == "pause":
            result = agent.pause("demo pause")
            if result.allowed:
                print(f"  ⏸️  Пауза (с этапа {agent.task.paused_from.value})")
            else:
                print(f"  ⛔ {result.reason}")
            continue

        if action == "resume":
            result = agent.resume()
            if result.allowed:
                meta = STAGE_META[agent.task.stage]
                print(f"  ▶️  Возобновлено: {meta['icon']} {agent.task.stage.value}")
            else:
                print(f"  ⛔ {result.reason}")
            continue

        if action == "step-done":
            step_obj = agent.step_done()
            if step_obj:
                print(f"  ✓ Выполнен: «{step_obj.text}»")
                nxt = agent.task.current_step
                if nxt:
                    print(f"  Следующий: → {nxt.text}")
            else:
                print(f"  Нет активного шага")
            continue

        if action == "pass-validation":
            agent.pass_validation()
            print(f"  ✅ Валидация отмечена как пройденная")
            continue

        if action == "msg":
            msg = step["text"]
            print(f"  👤 {msg}")
            resp = agent.ask(msg)
            total_in += resp.input_tokens
            total_out += resp.output_tokens
            cost = calc_cost(model, resp.input_tokens, resp.output_tokens)

            print(f"  🤖 {resp.text[:250]}{'…' if len(resp.text) > 250 else ''}")
            print(f"  [in={resp.input_tokens:,} out={resp.output_tokens:,} | ${cost:.6f} | {resp.elapsed_sec}s]")

            if resp.auto_transition:
                print(f"  ⚡ Авто-переход: IDLE → {resp.auto_transition.value}")

            if resp.auto_steps:
                print(f"  📋 Шаги авто-извлечены ({len(resp.auto_steps)}):")
                for s in resp.auto_steps:
                    print(f"     • {s}")

            if resp.violation:
                print(f"  ⚠️  НАРУШЕНИЕ FSM: {resp.violation_reason}")

    # Итоги
    print(f"\n{'━' * 65}")
    print("  ИТОГИ")
    print(f"{'━' * 65}")

    total_cost = calc_cost(model, total_in, total_out)
    print(f"\n  Токены: input={total_in:,}  output={total_out:,}  стоимость=${total_cost:.4f}")

    print(f"\n  ✅ Успешные переходы ({len(agent.task.transitions_log)}):")
    for ts, frm, to, comment in agent.task.transitions_log:
        c = f" ({comment})" if comment else ""
        print(f"    {ts}  {frm:12s} → {to:12s}{c}")

    if agent.task.blocked_attempts:
        print(f"\n  ⛔ Заблокированные попытки ({len(agent.task.blocked_attempts)}):")
        for ts, frm, to, reason in agent.task.blocked_attempts:
            print(f"    {ts}  {frm:12s} → {to:12s}  | {reason}")


# ══════════════════════════════════════════════════════════════════
# Интерактивный режим
# ══════════════════════════════════════════════════════════════════

HELP = """\
Команды:
  /state              — полный статус задачи
  /next               — перейти к следующему этапу
  /approve            — утвердить план (PLANNING → APPROVED)
  /reject             — отклонить план (доработка)
  /pause              — поставить на паузу
  /resume             — продолжить с паузы
  /restart [name]     — новая задача
  /step-done          — отметить текущий шаг выполненным
  /step-add <text>    — добавить шаг
  /pass-validation    — отметить валидацию пройденной
  /force <stage>      — попытка принудительного перехода (демо блокировки)
  /transitions        — граф переходов
  /log                — история переходов и блокировок
  /tokens             — статистика токенов
  /exit               — выход"""


def run_interactive(client: Anthropic, model: str, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    agent = ControlledFSMAgent(
        client=client, model=model, save_path=save_path,
        validate_responses=True,
    )

    stage_meta = STAGE_META[agent.task.stage]
    print(f"\n  Controlled FSM Agent ({model})")
    print(f"  Сохранение: {save_path.name}")
    print(f"  Текущий этап: {stage_meta['icon']} {agent.task.stage.value.upper()}")
    if agent.task.stage != Stage.IDLE:
        print(f"  Задача: {agent.task.name}")
        print(f"  Прогресс: {agent.task.progress_pct}%")
        print(f"  План утв.: {'да' if agent.task.plan_approved else 'нет'}")
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

        if user_text == "/transitions":
            print(format_transitions_graph())
            continue

        if user_text == "/log":
            if agent.task.transitions_log:
                print(f"  ✅ Переходы ({len(agent.task.transitions_log)}):")
                for ts, frm, to, comment in agent.task.transitions_log:
                    c = f" ({comment})" if comment else ""
                    print(f"    {ts}  {frm} → {to}{c}")
            if agent.task.blocked_attempts:
                print(f"  ⛔ Заблокировано ({len(agent.task.blocked_attempts)}):")
                for ts, frm, to, reason in agent.task.blocked_attempts:
                    print(f"    {ts}  {frm} → {to}: {reason}")
            if not agent.task.transitions_log and not agent.task.blocked_attempts:
                print("  История пуста.")
            continue

        if user_text == "/next":
            result = agent.next_stage()
            if result.allowed:
                meta = STAGE_META[agent.task.stage]
                print(f"  ✅ {result.from_stage.value} → {meta['icon']} {result.to_stage.value}")
                print(f"  Цель: {meta['goal']}")
                print(f"  Ожидается: {meta['expected_action']}")
            else:
                print(f"  ⛔ ЗАБЛОКИРОВАНО: {result.reason}")
            agent.save()
            continue

        if user_text == "/approve":
            result = agent.approve_plan()
            if result.allowed:
                meta = STAGE_META[Stage.APPROVED]
                print(f"  ✅ План утверждён! → {meta['icon']} APPROVED")
                print(f"  /next для перехода к выполнению")
            else:
                print(f"  ⛔ {result.reason}")
            continue

        if user_text == "/reject":
            result = agent.reject_plan()
            print(f"  {'🔄' if result.allowed else '⛔'} {result.reason}")
            if result.allowed:
                print(f"  Шаги очищены. Опишите план заново.")
            continue

        if user_text == "/pause":
            result = agent.pause()
            if result.allowed:
                print(f"  ⏸️  Задача приостановлена (с этапа {agent.task.paused_from.value})")
                print(f"  /resume для продолжения.")
            else:
                print(f"  ⛔ {result.reason}")
            continue

        if user_text == "/resume":
            result = agent.resume()
            if result.allowed:
                meta = STAGE_META[agent.task.stage]
                print(f"  ▶️  Продолжаем: {meta['icon']} {agent.task.stage.value}")
                print(f"  {meta['expected_action']}")
                cur = agent.task.current_step
                if cur:
                    print(f"  Текущий шаг: → {cur.text}")
            else:
                print(f"  ⛔ {result.reason}")
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
                    print(f"  Все шаги выполнены. /next для перехода к валидации.")
            else:
                print(f"  Нет активного шага.")
            continue

        if user_text.startswith("/step-add "):
            text = user_text[10:].strip()
            if text:
                step = agent.step_add(text)
                print(f"  + Шаг добавлен: {step.text}")
            continue

        if user_text == "/pass-validation":
            if agent.task.stage != Stage.VALIDATION:
                print(f"  ⛔ Доступно только на этапе VALIDATION (сейчас: {agent.task.stage.value})")
            else:
                agent.pass_validation()
                print(f"  ✅ Валидация пройдена. /next для завершения.")
            continue

        if user_text.startswith("/force "):
            target = user_text[7:].strip()
            print(f"  🔨 Попытка принудительного перехода → {target}")
            result = agent.force_stage(target)
            if result.allowed:
                meta = STAGE_META[agent.task.stage]
                print(f"  ✅ Переход: {result.from_stage.value} → {meta['icon']} {result.to_stage.value}")
            else:
                print(f"  ⛔ ЗАБЛОКИРОВАНО: {result.reason}")
                if result.gate_failed:
                    print(f"  📌 Gate: {result.gate_failed}")
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

        if resp.violation:
            print(f"  ⚠️  НАРУШЕНИЕ FSM: {resp.violation_reason}")

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
    mode = os.getenv("DAY15_MODE", "demo")

    FSM_DIR.mkdir(exist_ok=True)

    if mode == "interactive":
        run_interactive(client, model, FSM_DIR / "controlled_task.json")
    else:
        run_demo(client, model, FSM_DIR / "demo_controlled.json")


if __name__ == "__main__":
    main()

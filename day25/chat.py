"""
День 25. Мини-чат с RAG + памятью задачи (production-like)

CLI-чат, который:
  1. Хранит историю диалога (conversation history)
  2. При каждом вопросе ищет контекст через RAG (day24 pipeline)
  3. Отвечает с учётом найденной информации + всегда выводит источники
  4. Ведёт «память задачи» (task state):
     - что пользователь уточнил
     - какие ограничения/термины зафиксированы
     - какова цель диалога
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from day21.indexer import load_index, search as faiss_search
from day22.rag import MODEL, PROJECT_ROOT, ask_llm
from day23.rag import RetrievalConfig, collect_candidates, rerank_and_filter, rewrite_query
from day24.rag import (
    RELEVANCE_THRESHOLD,
    RetrievedChunk,
    _inject_targeted_chunks,
    parse_llm_response,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

MAX_HISTORY_MESSAGES = 20  # храним последние N пар сообщений
MAX_CONTEXT_CHARS = 6000   # лимит контекста из RAG


# ── Task State: память задачи ──────────────────────────────────────

@dataclass
class TaskState:
    """Память задачи — обновляется LLM после каждого хода."""
    goal: str = ""               # цель диалога
    clarifications: list[str] = field(default_factory=list)  # что пользователь уточнил
    constraints: list[str] = field(default_factory=list)     # зафиксированные ограничения/термины
    open_questions: list[str] = field(default_factory=list)  # что ещё нужно уточнить

    def to_prompt(self) -> str:
        parts = []
        if self.goal:
            parts.append(f"Цель диалога: {self.goal}")
        if self.clarifications:
            parts.append("Уточнения пользователя:\n" + "\n".join(f"  - {c}" for c in self.clarifications))
        if self.constraints:
            parts.append("Зафиксированные ограничения/термины:\n" + "\n".join(f"  - {c}" for c in self.constraints))
        if self.open_questions:
            parts.append("Открытые вопросы:\n" + "\n".join(f"  - {q}" for q in self.open_questions))
        return "\n".join(parts) if parts else "(пусто)"

    def is_empty(self) -> bool:
        return not self.goal and not self.clarifications and not self.constraints


# ── Системные промпты ──────────────────────────────────────────────

SYSTEM_CHAT = """\
Ты — ассистент, отвечающий на вопросы о программном проекте AiAdventFirst.
Тебе предоставлены пронумерованные фрагменты исходного кода и документации.

ЖЁСТКИЕ ПРАВИЛА:

1. **Ответ** — кратко и по делу, на русском языке. Учитывай историю разговора и память задачи.

2. **Источники** — после ответа ОБЯЗАТЕЛЬНО приведи секцию:
   ### Источники
   - [1] файл / секция
   - [2] файл / секция
   (только те, которые реально использовались)

3. **Цитаты** — после источников ОБЯЗАТЕЛЬНО приведи секцию:
   ### Цитаты
   > [1] «точная цитата из фрагмента»
   Минимум одна цитата. Цитируй дословно.

4. Если контекст НЕ содержит ответа — НЕ выдумывай. Ответь:
   «Я не нашёл достаточно информации в предоставленном контексте.»

5. НЕ выдумывай код, функции или факты, которых нет в контексте.

6. Если пользователь уточняет или ссылается на предыдущие ответы — учитывай историю."""


SYSTEM_TASK_UPDATE = """\
Ты анализируешь диалог и обновляешь «память задачи» — структуру, которая отслеживает:
- goal: текущая цель диалога (что пользователь пытается выяснить/сделать)
- clarifications: что пользователь уже уточнил (список строк)
- constraints: зафиксированные ограничения, термины, условия (список строк)
- open_questions: что ещё можно/нужно уточнить (список строк)

Ответь СТРОГО в JSON (без markdown, без ```):
{
  "goal": "...",
  "clarifications": ["..."],
  "constraints": ["..."],
  "open_questions": ["..."]
}

Будь краток. Максимум 3-5 элементов в каждом списке. Объединяй близкие пункты."""


# ── Retrieval ──────────────────────────────────────────────────────

def retrieve_context(
    question: str,
    cfg: RetrievalConfig,
) -> tuple[str, list[RetrievedChunk], str, float]:
    """RAG retrieval: rewrite → collect → rerank → format."""
    rewritten = rewrite_query(question)
    candidates = collect_candidates(rewritten, top_k=cfg.top_k_before)
    filtered = rerank_and_filter(question, rewritten, candidates, cfg)
    filtered = _inject_targeted_chunks(question, filtered)

    best_score = max((item.get("rerank_score", 0.0) for item in filtered), default=0.0)

    chunks: list[RetrievedChunk] = []
    parts: list[str] = []
    total_chars = 0

    for i, item in enumerate(filtered, 1):
        text = item["text"]
        if total_chars + len(text) > MAX_CONTEXT_CHARS:
            break
        total_chars += len(text)

        sec = item.get("section", "")
        chunk = RetrievedChunk(
            source=item["source"],
            section=sec,
            chunk_id=item["chunk_id"],
            text=text,
            rerank_score=item.get("rerank_score", 0.0),
        )
        chunks.append(chunk)

        sec_label = f" / {sec}" if sec else ""
        score_info = f"score={item.get('rerank_score', 0):.2f}"
        parts.append(
            f"[Фрагмент {i}] {item['source']}{sec_label} ({score_info})\n{text}"
        )

    context = "\n\n---\n\n".join(parts) if parts else ""
    return context, chunks, rewritten, best_score


# ── Обновление памяти задачи через LLM ────────────────────────────

def update_task_state(
    client: Anthropic,
    model: str,
    history: list[dict],
    current_state: TaskState,
) -> TaskState:
    """Просим LLM обновить task state на основе последних сообщений."""
    # Формируем краткую историю для анализа
    recent = history[-6:]  # последние 3 пары
    hist_text = ""
    for msg in recent:
        role = "Пользователь" if msg["role"] == "user" else "Ассистент"
        content = msg["content"][:500]
        hist_text += f"{role}: {content}\n\n"

    prompt = (
        f"Текущая память задачи:\n{current_state.to_prompt()}\n\n"
        f"Последние сообщения:\n{hist_text}\n"
        f"Обнови память задачи. Ответь JSON."
    )

    try:
        text, _, _, _ = ask_llm(client, model, SYSTEM_TASK_UPDATE, prompt)
        # Парсим JSON
        text = text.strip()
        # Убираем markdown обёртку если есть
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        data = json.loads(text)
        return TaskState(
            goal=data.get("goal", current_state.goal),
            clarifications=data.get("clarifications", current_state.clarifications)[:5],
            constraints=data.get("constraints", current_state.constraints)[:5],
            open_questions=data.get("open_questions", current_state.open_questions)[:5],
        )
    except (json.JSONDecodeError, KeyError, Exception):
        return current_state


# ── Формирование промпта с историей ───────────────────────────────

def build_chat_messages(
    question: str,
    context: str,
    history: list[dict],
    task_state: TaskState,
) -> list[dict]:
    """Строим массив messages для API с историей + контекстом + task state."""
    messages: list[dict] = []

    # Добавляем историю (без RAG-контекста, только сами сообщения)
    for msg in history[-MAX_HISTORY_MESSAGES:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Текущий вопрос с контекстом
    user_content = ""
    if context:
        user_content += f"Контекст из проекта:\n\n{context}\n\n───\n\n"

    if not task_state.is_empty():
        user_content += f"Память задачи:\n{task_state.to_prompt()}\n\n───\n\n"

    user_content += f"Вопрос: {question}"
    messages.append({"role": "user", "content": user_content})

    return messages


# ── Один ход чата ──────────────────────────────────────────────────

@dataclass
class ChatTurn:
    question: str
    answer_text: str
    sources: list[str]
    quotes: list[str]
    is_idk: bool
    retrieved_chunks: list[RetrievedChunk]
    rewritten_query: str
    best_score: float
    task_state: TaskState
    input_tokens: int
    output_tokens: int
    elapsed_sec: float


def chat_turn(
    client: Anthropic,
    model: str,
    question: str,
    history: list[dict],
    task_state: TaskState,
    cfg: RetrievalConfig,
) -> ChatTurn:
    """Один ход: retrieval → build messages → LLM → parse → update task state."""

    # 1. Retrieval
    context, chunks, rewritten, best_score = retrieve_context(question, cfg)

    # 2. Проверка порога
    idk_forced = best_score < RELEVANCE_THRESHOLD and not history

    if idk_forced or (not context.strip() and not history):
        idk_text = "Я не нашёл достаточно информации в предоставленном контексте."
        return ChatTurn(
            question=question,
            answer_text=idk_text,
            sources=[], quotes=[], is_idk=True,
            retrieved_chunks=chunks, rewritten_query=rewritten,
            best_score=best_score, task_state=task_state,
            input_tokens=0, output_tokens=0, elapsed_sec=0.0,
        )

    # 3. Build messages с историей
    messages = build_chat_messages(question, context, history, task_state)

    # 4. LLM call
    t0 = time.time()
    resp = client.messages.create(
        model=model,
        max_tokens=1500,
        system=SYSTEM_CHAT,
        messages=messages,
    )
    elapsed = round(time.time() - t0, 2)
    raw = resp.content[0].text

    # 5. Parse
    parsed = parse_llm_response(raw)

    # 6. Обновляем историю
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": parsed.answer_text})

    # 7. Обновляем task state (каждые 2 хода или на первом ходу)
    turn_num = len(history) // 2
    if turn_num <= 1 or turn_num % 2 == 0:
        task_state = update_task_state(client, model, history, task_state)

    return ChatTurn(
        question=question,
        answer_text=parsed.answer_text,
        sources=parsed.sources,
        quotes=parsed.quotes,
        is_idk=parsed.is_idk,
        retrieved_chunks=chunks,
        rewritten_query=rewritten,
        best_score=best_score,
        task_state=task_state,
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
        elapsed_sec=elapsed,
    )


# ═══════════════════════════════════════════════════════════════════
#  CLI-интерфейс
# ═══════════════════════════════════════════════════════════════════

def run_interactive(
    client: Anthropic | None = None,
    model: str = MODEL,
    cfg: RetrievalConfig | None = None,
):
    if cfg is None:
        cfg = RetrievalConfig()
    if client is None:
        load_dotenv(PROJECT_ROOT / ".env")
        client = Anthropic()

    print("=" * 68)
    print("  День 25 — Мини-чат с RAG + памятью задачи")
    print("=" * 68)
    print("  Команды:")
    print("    /state   — показать память задачи")
    print("    /history — показать историю")
    print("    /reset   — сбросить всё")
    print("    пустая строка — выход")
    print()

    history: list[dict] = []
    task_state = TaskState()

    while True:
        query = input("[chat]> ").strip()
        if not query:
            break

        if query == "/state":
            print(f"\n  Память задачи:\n  {task_state.to_prompt()}\n")
            continue
        if query == "/history":
            print(f"\n  История ({len(history)} сообщений):")
            for msg in history:
                role = "  USER" if msg["role"] == "user" else "  ASST"
                print(f"  {role}: {msg['content'][:120]}...")
            print()
            continue
        if query == "/reset":
            history.clear()
            task_state = TaskState()
            print("  Сброшено.\n")
            continue

        turn = chat_turn(client, model, query, history, task_state, cfg)
        task_state = turn.task_state

        # Ответ
        print(f"\n  {turn.answer_text}")

        # Источники
        if turn.sources:
            print("\n  Источники:")
            for s in turn.sources:
                print(f"    - {s}")

        # Цитаты
        if turn.quotes:
            print("\n  Цитаты:")
            for i, q in enumerate(turn.quotes, 1):
                print(f"    [{i}] «{q[:150]}{'...' if len(q) > 150 else ''}»")

        # Мета
        print(f"\n  [{turn.elapsed_sec}s | in={turn.input_tokens} out={turn.output_tokens}"
              f" | score={turn.best_score:.3f}]")
        print(f"  Rewrite: {turn.rewritten_query}")
        print(f"  Sources: {', '.join(c.source for c in turn.retrieved_chunks)}")

        if not task_state.is_empty():
            print(f"  Цель: {task_state.goal}")

        print()


# ═══════════════════════════════════════════════════════════════════
#  Автоматические сценарии для проверки
# ═══════════════════════════════════════════════════════════════════

SCENARIO_1 = {
    "name": "Исследование RAG-пайплайна",
    "messages": [
        "Какие подходы к retrieval используются в проекте?",
        "Расскажи подробнее про гибридный поиск",
        "А как устроен reranking?",
        "Какие метрики используются для оценки качества?",
        "Как именно считается token overlap?",
        "Вернёмся к retrieval — что делает keyword_search?",
        "Какие пороги и параметры настраиваются?",
        "Как связаны day22 и day23 между собой?",
        "А day24 что добавляет нового?",
        "Можешь суммировать всю цепочку: от вопроса до ответа?",
    ],
}

SCENARIO_2 = {
    "name": "Изучение индексации и чанкинга",
    "messages": [
        "Как в проекте реализована индексация документов?",
        "Какая модель эмбеддингов используется?",
        "Какие стратегии чанкинга есть?",
        "Чем structural отличается от fixed?",
        "Какой размер чанков в каждой стратегии?",
        "Как хранится индекс FAISS?",
        "Какие файлы проекта индексируются?",
        "А markdown-файлы тоже индексируются?",
        "Вернёмся к structural — как определяются секции?",
        "Как можно улучшить текущую индексацию?",
    ],
}


def run_scenario(
    client: Anthropic,
    model: str,
    scenario: dict,
    cfg: RetrievalConfig,
) -> list[dict]:
    """Прогоняем сценарий из N сообщений, проверяем сохранность контекста."""
    print(f"\n{'═' * 68}")
    print(f"  Сценарий: {scenario['name']}")
    print(f"  Сообщений: {len(scenario['messages'])}")
    print(f"{'═' * 68}")

    history: list[dict] = []
    task_state = TaskState()
    results: list[dict] = []

    for i, msg in enumerate(scenario["messages"], 1):
        print(f"\n{'─' * 68}")
        print(f"  [{i}/{len(scenario['messages'])}] {msg}")
        print(f"{'─' * 68}")

        turn = chat_turn(client, model, msg, history, task_state, cfg)
        task_state = turn.task_state

        has_sources = len(turn.sources) > 0
        has_quotes = len(turn.quotes) > 0

        print(f"\n  Ответ: {turn.answer_text[:250]}{'...' if len(turn.answer_text) > 250 else ''}")

        if turn.sources:
            print(f"  Источники: {', '.join(turn.sources[:3])}")
        if turn.quotes:
            for j, q in enumerate(turn.quotes[:2], 1):
                print(f"  Цитата [{j}]: «{q[:100]}...»")

        print(f"\n  Score: {turn.best_score:.3f} | Sources: {'YES' if has_sources else 'NO'}"
              f" | Quotes: {'YES' if has_quotes else 'NO'}"
              f" | IDK: {'YES' if turn.is_idk else 'NO'}")
        print(f"  Цель: {task_state.goal}")
        print(f"  Уточнения: {task_state.clarifications}")

        results.append({
            "turn": i,
            "question": msg,
            "answer": turn.answer_text[:500],
            "sources": turn.sources,
            "quotes": turn.quotes[:3],
            "has_sources": has_sources,
            "has_quotes": has_quotes,
            "is_idk": turn.is_idk,
            "best_score": turn.best_score,
            "task_state": asdict(turn.task_state),
            "tokens_in": turn.input_tokens,
            "tokens_out": turn.output_tokens,
            "elapsed": turn.elapsed_sec,
        })

    return results


def run_demo(
    client: Anthropic | None = None,
    model: str = MODEL,
    cfg: RetrievalConfig | None = None,
):
    if cfg is None:
        cfg = RetrievalConfig()
    if client is None:
        load_dotenv(PROJECT_ROOT / ".env")
        client = Anthropic()

    print("=" * 68)
    print("  День 25 — Мини-чат с RAG + памятью задачи")
    print("=" * 68)
    print(f"  Модель: {model}")
    print(f"  Порог: {RELEVANCE_THRESHOLD}")
    print(f"  History limit: {MAX_HISTORY_MESSAGES} сообщений")
    print()

    all_results: dict[str, list[dict]] = {}

    for scenario in [SCENARIO_1, SCENARIO_2]:
        results = run_scenario(client, model, scenario, cfg)
        all_results[scenario["name"]] = results

    # ── Сводка ──
    print("\n\n" + "=" * 68)
    print("  СВОДКА")
    print("=" * 68)

    for name, results in all_results.items():
        total = len(results)
        with_sources = sum(1 for r in results if r["has_sources"])
        with_quotes = sum(1 for r in results if r["has_quotes"])
        idk_count = sum(1 for r in results if r["is_idk"])
        goals_set = sum(1 for r in results if r["task_state"]["goal"])

        print(f"\n  Сценарий: {name} ({total} ходов)")
        print(f"    С источниками:  {with_sources}/{total} ({with_sources*100//total}%)")
        print(f"    С цитатами:     {with_quotes}/{total} ({with_quotes*100//total}%)")
        print(f"    IDK:            {idk_count}/{total}")
        print(f"    Цель задана:    {goals_set}/{total}")

        # Проверяем, что цель сохраняется
        last_goal = results[-1]["task_state"]["goal"] if results else ""
        print(f"    Финальная цель: {last_goal}")
        last_clarifications = results[-1]["task_state"]["clarifications"] if results else []
        print(f"    Уточнения:      {last_clarifications}")

    # Сохранение
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "scenarios.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n  Результаты: {out_path}")


# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    load_dotenv(PROJECT_ROOT / ".env")
    client = Anthropic()

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        run_interactive(client)
    else:
        run_demo(client)

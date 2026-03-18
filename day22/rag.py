"""
День 22. Первый RAG-запрос

Пайплайн:
  вопрос → поиск релевантных чанков (FAISS + keyword) → объединение с вопросом → LLM

Два режима:
  - без RAG: модель отвечает только из своих знаний
  - с RAG:   модель получает top-k чанков из индекса + вопрос

Retrieval: гибридный (semantic FAISS + keyword BM25-like) —
  all-MiniLM-L6-v2 слабо работает с кириллицей, поэтому дополняем keyword-поиском.

Оценка: 10 контрольных вопросов × 2 режима → сравнение качества.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

# день 21 — переиспользуем индекс
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from day21.indexer import search as faiss_search, load_index

# ── Конфиг ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"
STRATEGY = "structural"
TOP_K = 5
MODEL = "claude-sonnet-4-20250514"


# ── Промпты ────────────────────────────────────────────────────────

SYSTEM_NO_RAG = """\
Ты — ассистент, отвечающий на вопросы о программном проекте AiAdventFirst.
Это учебный проект, где каждый «день» (day2..day21) реализует определённую
фичу Claude API / MCP. Отвечай кратко и по делу, на русском языке.
Если не знаешь точного ответа — скажи честно."""

SYSTEM_RAG = """\
Ты — ассистент, отвечающий на вопросы о программном проекте AiAdventFirst.
Тебе предоставлены фрагменты исходного кода и документации из проекта.
Используй ИХ как основной источник информации для ответа.

Правила:
- Отвечай кратко и по делу, на русском языке.
- Ссылайся на конкретные файлы/функции/классы из контекста.
- Если контекст не содержит ответа — скажи об этом.
- НЕ выдумывай код, которого нет в контексте."""


# ── Dataclass результата ───────────────────────────────────────────

@dataclass
class Answer:
    question: str
    mode: str               # "no_rag" | "rag"
    answer: str
    sources: list[str]      # файлы-источники (для RAG)
    input_tokens: int
    output_tokens: int
    elapsed_sec: float
    context_chars: int      # размер контекста из чанков


# ═══════════════════════════════════════════════════════════════════
#  Гибридный retrieval: FAISS (semantic) + keyword
# ═══════════════════════════════════════════════════════════════════

_WORD_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9_]+")


def _tokenize(text: str) -> list[str]:
    """Простая токенизация: слова в нижнем регистре."""
    return [w.lower() for w in _WORD_RE.findall(text)]


def keyword_search(query: str, strategy: str = STRATEGY, top_k: int = TOP_K) -> list[dict]:
    """Keyword-поиск по метаданным: считаем пересечение слов запроса и текста чанка."""
    _, meta = load_index(strategy)
    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return []

    scored: list[tuple[float, int]] = []
    for i, m in enumerate(meta):
        text_tokens = Counter(_tokenize(m["text"]))
        # простое TF-подобное: сколько уникальных query-слов нашлось + сумма частот
        hit_words = query_tokens & set(text_tokens.keys())
        if not hit_words:
            continue
        score = len(hit_words) + sum(text_tokens[w] for w in hit_words) * 0.1
        scored.append((score, i))

    scored.sort(reverse=True)
    results = []
    for score, idx in scored[:top_k]:
        entry = dict(meta[idx])
        entry["score"] = score
        results.append(entry)
    return results


def hybrid_search(query: str, top_k: int = TOP_K) -> list[dict]:
    """Гибридный поиск: объединяем semantic (FAISS) и keyword результаты.

    Reciprocal Rank Fusion (RRF): score = sum(1 / (k + rank)) по обоим спискам.
    """
    sem_results = faiss_search(query, strategy=STRATEGY, top_k=top_k * 2)
    kw_results = keyword_search(query, strategy=STRATEGY, top_k=top_k * 2)

    rrf_scores: dict[str, float] = {}   # chunk_id → score
    chunk_data: dict[str, dict] = {}     # chunk_id → metadata
    k = 60  # RRF constant

    for rank, r in enumerate(sem_results):
        cid = r["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (k + rank + 1)
        chunk_data[cid] = r

    for rank, r in enumerate(kw_results):
        cid = r["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (k + rank + 1)
        if cid not in chunk_data:
            chunk_data[cid] = r

    # сортируем по RRF-score
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for cid, score in ranked[:top_k]:
        entry = dict(chunk_data[cid])
        entry["score"] = round(score, 5)
        results.append(entry)
    return results


# ── Построение контекста ───────────────────────────────────────────

def build_context(query: str, top_k: int = TOP_K) -> tuple[str, list[str]]:
    """Ищем релевантные чанки (гибридный поиск) и собираем контекст."""
    results = hybrid_search(query, top_k=top_k)
    if not results:
        return "", []

    parts: list[str] = []
    sources: list[str] = []
    for i, r in enumerate(results, 1):
        sec = f" / {r['section']}" if r.get("section") else ""
        header = f"[{i}] {r['source']}{sec}"
        parts.append(f"{header}\n{r['text']}")
        sources.append(r["source"])

    context = "\n\n---\n\n".join(parts)
    return context, sources


def format_rag_prompt(question: str, context: str) -> str:
    """Формируем пользовательский промпт с контекстом."""
    return (
        f"Контекст из проекта:\n\n{context}\n\n"
        f"───\n\n"
        f"Вопрос: {question}"
    )


# ── Вызов LLM ─────────────────────────────────────────────────────

def ask_llm(
    client: Anthropic,
    model: str,
    system: str,
    user_msg: str,
) -> tuple[str, int, int, float]:
    """Один вызов Claude → (ответ, input_tokens, output_tokens, sec)."""
    t0 = time.time()
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )
    elapsed = round(time.time() - t0, 2)
    text = resp.content[0].text
    return text, resp.usage.input_tokens, resp.usage.output_tokens, elapsed


# ── Основная функция: вопрос → ответ ──────────────────────────────

def ask_no_rag(client: Anthropic, model: str, question: str) -> Answer:
    """Ответ без RAG — только знания модели."""
    text, inp, out, sec = ask_llm(client, model, SYSTEM_NO_RAG, question)
    return Answer(
        question=question, mode="no_rag", answer=text,
        sources=[], input_tokens=inp, output_tokens=out,
        elapsed_sec=sec, context_chars=0,
    )


def ask_rag(client: Anthropic, model: str, question: str) -> Answer:
    """Ответ с RAG — hybrid retrieval + generation."""
    context, sources = build_context(question)
    user_msg = format_rag_prompt(question, context)
    text, inp, out, sec = ask_llm(client, model, SYSTEM_RAG, user_msg)
    return Answer(
        question=question, mode="rag", answer=text,
        sources=sources, input_tokens=inp, output_tokens=out,
        elapsed_sec=sec, context_chars=len(context),
    )


# ═══════════════════════════════════════════════════════════════════
#  10 контрольных вопросов
# ═══════════════════════════════════════════════════════════════════

EVAL_QUESTIONS: list[dict] = [
    {
        "id": 1,
        "question": "Какие три стратегии управления контекстом реализованы в day10 и в чём их суть?",
        "expected": "Sliding Window (последние N сообщений), Facts Extraction (извлечение ключевых фактов), Branching (ветвление истории по темам)",
        "sources": ["day10.py"],
    },
    {
        "id": 2,
        "question": "Какие три слоя памяти реализованы в day11 и через какие CLI-команды ими управлять?",
        "expected": "Short-term (текущий диалог), Working (данные задачи, /task), Long-term (профиль/знания, /remember, /forget)",
        "sources": ["day11.py"],
    },
    {
        "id": 3,
        "question": "Какие этапы (состояния) проходит задача в FSM из day13?",
        "expected": "IDLE → PLANNING → EXECUTION → VALIDATION → DONE, плюс PAUSED из любого этапа",
        "sources": ["day13.py"],
    },
    {
        "id": 4,
        "question": "Какие MCP-инструменты предоставляет сервер трекера задач в day17?",
        "expected": "list_projects, get_project, list_tasks, create_task, update_task_status",
        "sources": ["day17/server.py"],
    },
    {
        "id": 5,
        "question": "Как устроен планировщик в day18 — какие инструменты он предоставляет и где хранит данные?",
        "expected": "Инструменты: create_job, list_jobs, delete_job, get_job_history, get_summary, trigger_job. Хранение: SQLite (scheduler.db)",
        "sources": ["day18/server.py"],
    },
    {
        "id": 6,
        "question": "К каким трём MCP-серверам подключается оркестратор в day20?",
        "expected": "tracker (day17/server.py), scheduler (day18/server.py), notes (day20/server_notes.py)",
        "sources": ["day20/agent.py"],
    },
    {
        "id": 7,
        "question": "Как в day8 считается стоимость вызова API и какие три сценария диалога демонстрируются?",
        "expected": "Стоимость = input_tokens * price_in + output_tokens * price_out. Сценарии: короткий (3 хода), длинный (8 ходов), переполнение (10 ходов)",
        "sources": ["day8.py", "DAY8.md"],
    },
    {
        "id": 8,
        "question": "Какие 4 метода решения логической задачи сравниваются в day3?",
        "expected": "Прямой промпт, цепочка рассуждений (chain-of-thought), перебор вариантов, метод от противного",
        "sources": ["day3.py", "DAY3.md"],
    },
    {
        "id": 9,
        "question": "Чем отличается pipeline-композиция инструментов в day19 от одиночных вызовов в day17?",
        "expected": "В day19 несколько инструментов вызываются последовательно в цепочке (pipeline), результат одного подаётся на вход другому. В day17 — одиночные вызовы",
        "sources": ["day19/server.py", "day19/agent.py"],
    },
    {
        "id": 10,
        "question": "Какую модель эмбеддингов и какой тип FAISS-индекса использует система индексации в day21?",
        "expected": "Модель: all-MiniLM-L6-v2 (sentence-transformers, dim=384). Индекс: FAISS IndexFlatIP (inner product ≈ cosine с нормализацией)",
        "sources": ["day21/indexer.py"],
    },
]


# ═══════════════════════════════════════════════════════════════════
#  Оценка качества: source recall + LLM-judge
# ═══════════════════════════════════════════════════════════════════

def eval_source_recall(expected_sources: list[str], actual_sources: list[str]) -> float:
    """Доля ожидаемых источников, попавших в retrieval."""
    if not expected_sources:
        return 1.0
    hits = sum(1 for s in expected_sources if any(s in a for a in actual_sources))
    return hits / len(expected_sources)


def eval_answer_quality(
    client: Anthropic,
    model: str,
    question: str,
    expected: str,
    answer: str,
) -> tuple[int, str]:
    """LLM-судья: оценивает ответ по 5-балльной шкале.
    Returns (score 1-5, brief_reason).
    """
    judge_prompt = f"""\
Оцени качество ответа на вопрос по 5-балльной шкале.

Вопрос: {question}
Эталон (ожидаемый ответ): {expected}
Ответ: {answer}

Критерии:
  5 — полный, точный ответ, совпадает с эталоном
  4 — верный ответ, но не все детали
  3 — частично верный, есть ошибки или пропуски
  2 — в основном неверный, но есть зерно правды
  1 — полностью неверный или "не знаю"

Ответь СТРОГО в формате:
SCORE: <число>
REASON: <одно предложение>"""

    text, _, _, _ = ask_llm(client, model, "Ты — объективный оценщик качества ответов.", judge_prompt)
    # парсим
    score = 1
    reason = ""
    for line in text.strip().split("\n"):
        if line.startswith("SCORE:"):
            try:
                score = int(line.split(":")[1].strip())
                score = max(1, min(5, score))
            except ValueError:
                pass
        if line.startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
    return score, reason


# ═══════════════════════════════════════════════════════════════════
#  Демо — прогон 10 вопросов в двух режимах
# ═══════════════════════════════════════════════════════════════════

def run_demo(client: Anthropic | None = None, model: str = MODEL):
    """Прогон всех контрольных вопросов: без RAG → с RAG → сравнение."""
    if client is None:
        load_dotenv(PROJECT_ROOT / ".env")
        client = Anthropic()

    print("=" * 70)
    print("  День 22 — RAG vs No-RAG: 10 контрольных вопросов")
    print("=" * 70)
    print(f"  Модель: {model}")
    print(f"  Индекс: {STRATEGY}, top_k={TOP_K}")
    print(f"  Retrieval: hybrid (FAISS + keyword RRF)")
    print()

    all_results: list[dict] = []

    for q in EVAL_QUESTIONS:
        qnum = q["id"]
        question = q["question"]
        print(f"\n{'━' * 70}")
        print(f"  [{qnum}/10] {question}")
        print(f"{'━' * 70}")
        print(f"  Ожидание: {q['expected']}")
        print(f"  Источники: {', '.join(q['sources'])}")

        # --- без RAG ---
        print(f"\n  ▸ Без RAG...")
        ans_no = ask_no_rag(client, model, question)
        score_no, reason_no = eval_answer_quality(
            client, model, question, q["expected"], ans_no.answer)
        print(f"    [{ans_no.elapsed_sec}s | in={ans_no.input_tokens} out={ans_no.output_tokens}]")
        print(f"    Оценка: {score_no}/5 — {reason_no}")
        print(f"    Ответ: {ans_no.answer[:250]}{'...' if len(ans_no.answer) > 250 else ''}")

        # --- с RAG ---
        print(f"\n  ▸ С RAG (hybrid, top_{TOP_K})...")
        ans_rag = ask_rag(client, model, question)
        score_rag, reason_rag = eval_answer_quality(
            client, model, question, q["expected"], ans_rag.answer)
        src_recall = eval_source_recall(q["sources"], ans_rag.sources)
        print(f"    [{ans_rag.elapsed_sec}s | in={ans_rag.input_tokens} out={ans_rag.output_tokens} | ctx={ans_rag.context_chars:,}с]")
        print(f"    Источники: {', '.join(ans_rag.sources)}")
        print(f"    Source recall: {src_recall:.0%}")
        print(f"    Оценка: {score_rag}/5 — {reason_rag}")
        print(f"    Ответ: {ans_rag.answer[:250]}{'...' if len(ans_rag.answer) > 250 else ''}")

        all_results.append({
            "id": qnum,
            "question": question,
            "expected": q["expected"],
            "expected_sources": q["sources"],
            "no_rag": {**asdict(ans_no), "score": score_no, "reason": reason_no},
            "rag": {**asdict(ans_rag), "score": score_rag, "reason": reason_rag,
                    "source_recall": src_recall},
        })

    # --- Сводка ---
    print_summary(all_results)

    # --- Сохранение ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n  Результаты: {out_path}")


def print_summary(results: list[dict]):
    """Сводная таблица: сравнение двух режимов."""
    print("\n\n" + "=" * 70)
    print("  СВОДКА: RAG vs No-RAG")
    print("=" * 70)

    total_in_no, total_out_no, total_sec_no = 0, 0, 0.0
    total_in_rag, total_out_rag, total_sec_rag = 0, 0, 0.0
    scores_no: list[int] = []
    scores_rag: list[int] = []
    recalls: list[float] = []

    print(f"\n  {'#':<4} {'No-RAG':>7} {'RAG':>7} {'Recall':>8} {'Источники RAG'}")
    print(f"  {'─' * 4} {'─' * 7} {'─' * 7} {'─' * 8} {'─' * 30}")

    for r in results:
        no = r["no_rag"]
        rag = r["rag"]
        total_in_no += no["input_tokens"]
        total_out_no += no["output_tokens"]
        total_sec_no += no["elapsed_sec"]
        total_in_rag += rag["input_tokens"]
        total_out_rag += rag["output_tokens"]
        total_sec_rag += rag["elapsed_sec"]
        scores_no.append(no["score"])
        scores_rag.append(rag["score"])
        recalls.append(rag.get("source_recall", 0))

        src_list = ", ".join(rag["sources"][:3])
        print(f"  {r['id']:<4} {no['score']:>5}/5 {rag['score']:>5}/5 {rag.get('source_recall', 0):>7.0%} {src_list}")

    avg_no = sum(scores_no) / len(scores_no)
    avg_rag = sum(scores_rag) / len(scores_rag)
    avg_recall = sum(recalls) / len(recalls)

    print(f"\n  {'СРЕДНИЕ':<20}")
    print(f"    No-RAG score:  {avg_no:.1f}/5")
    print(f"    RAG score:     {avg_rag:.1f}/5  (delta: {avg_rag - avg_no:+.1f})")
    print(f"    Source recall:  {avg_recall:.0%}")
    print()
    print(f"    No-RAG tokens: in={total_in_no:>8,}  out={total_out_no:>7,}  time={total_sec_no:>6.1f}s")
    print(f"    RAG tokens:    in={total_in_rag:>8,}  out={total_out_rag:>7,}  time={total_sec_rag:>6.1f}s")
    overhead_in = total_in_rag - total_in_no
    print(f"    RAG overhead:  +{overhead_in:,} input tokens ({overhead_in * 100 / max(total_in_no, 1):.0f}%)")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════
#  Интерактивный режим
# ═══════════════════════════════════════════════════════════════════

def run_interactive(client: Anthropic | None = None, model: str = MODEL):
    """Интерактивный чат с переключением RAG on/off."""
    if client is None:
        load_dotenv(PROJECT_ROOT / ".env")
        client = Anthropic()

    print("=" * 60)
    print("  День 22 — RAG-агент (интерактивный)")
    print("=" * 60)
    print(f"  Модель: {model}  |  Индекс: {STRATEGY}")
    print("  /rag    — включить RAG (по умолчанию)")
    print("  /norag  — выключить RAG")
    print("  /both   — показать оба ответа")
    print("  Пустая строка — выход.\n")

    mode = "rag"

    while True:
        query = input(f"[{mode}]> ").strip()
        if not query:
            break
        if query == "/rag":
            mode = "rag"
            print("  RAG включён.")
            continue
        if query == "/norag":
            mode = "no_rag"
            print("  RAG выключен.")
            continue
        if query == "/both":
            mode = "both"
            print("  Режим: оба ответа.")
            continue

        if mode in ("no_rag", "both"):
            print("\n  ── Без RAG ──")
            ans = ask_no_rag(client, model, query)
            print(f"  {ans.answer}")
            print(f"  [{ans.elapsed_sec}s | in={ans.input_tokens} out={ans.output_tokens}]")

        if mode in ("rag", "both"):
            print("\n  ── С RAG ──")
            ans = ask_rag(client, model, query)
            print(f"  {ans.answer}")
            print(f"  [{ans.elapsed_sec}s | in={ans.input_tokens} out={ans.output_tokens} | ctx={ans.context_chars:,}с]")
            print(f"  Источники: {', '.join(ans.sources)}")

        print()


# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    load_dotenv(PROJECT_ROOT / ".env")
    client = Anthropic()

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        run_interactive(client)
    else:
        run_demo(client)

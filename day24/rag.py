"""
День 24. Цитаты, источники и анти-галлюцинации

Поверх day23 (rewrite + rerank):
  1. Структурированный вывод: ответ + источники + цитаты
  2. Порог релевантности: если контекст слабый — "не знаю"
  3. Проверка на 10 вопросах: источники, цитаты, совпадение смысла
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from day21.indexer import load_index, search as faiss_search
from day22.rag import (
    EVAL_QUESTIONS,
    MODEL,
    PROJECT_ROOT,
    ask_llm,
    eval_answer_quality,
    eval_source_recall,
)
from day23.rag import (
    RetrievalConfig,
    _DAY_RE,
    collect_candidates,
    rerank_and_filter,
    rewrite_query,
)


RESULTS_DIR = Path(__file__).resolve().parent / "results"

# ── Порог релевантности для режима "не знаю" ─────────────────────
RELEVANCE_THRESHOLD = 0.30  # если лучший rerank_score < порога → отказ

# ── Системный промпт: обязательные источники + цитаты ─────────────

SYSTEM_CITED = """\
Ты — ассистент, отвечающий на вопросы о программном проекте AiAdventFirst.
Тебе предоставлены пронумерованные фрагменты исходного кода и документации.

ЖЁСТКИЕ ПРАВИЛА ОТВЕТА:

1. **Ответ** — кратко и по делу, на русском языке.

2. **Источники** — после ответа ОБЯЗАТЕЛЬНО приведи секцию:
   ### Источники
   - [1] файл / секция
   - [2] файл / секция
   (только те, которые реально использовались в ответе)

3. **Цитаты** — после источников ОБЯЗАТЕЛЬНО приведи секцию:
   ### Цитаты
   > [1] «точная цитата из фрагмента 1»
   > [2] «точная цитата из фрагмента 2»
   Цитируй дословно (допускается обрезка с «...»). Минимум одна цитата.

4. Если контекст НЕ содержит ответа — НЕ выдумывай. Ответь:
   «Я не нашёл достаточно информации в предоставленном контексте.
   Пожалуйста, уточните вопрос или переформулируйте.»
   и НЕ приводи цитаты / источники.

5. НЕ выдумывай код, функции или факты, которых нет в контексте."""


def format_cited_prompt(question: str, context: str) -> str:
    """Формируем промпт с нумерованными фрагментами."""
    return (
        f"Контекст из проекта:\n\n{context}\n\n"
        f"───\n\n"
        f"Вопрос: {question}"
    )


# ── Парсинг структурированного ответа ─────────────────────────────

@dataclass
class ParsedAnswer:
    answer_text: str
    sources: list[str]
    quotes: list[str]
    is_idk: bool  # режим "не знаю"


_IDK_MARKERS = [
    "не нашёл достаточно информации",
    "не нашел достаточно информации",
    "не содержит ответа",
    "уточните вопрос",
    "переформулируйте",
    "недостаточно информации",
    "не могу ответить",
]


def parse_llm_response(text: str) -> ParsedAnswer:
    """Разбираем ответ LLM на: текст ответа, источники, цитаты."""

    # Проверяем режим "не знаю"
    lower = text.lower()
    is_idk = any(marker in lower for marker in _IDK_MARKERS)

    # Извлекаем секцию ответа (до ### Источники)
    answer_text = text
    sources_section = ""
    quotes_section = ""

    # Ищем секцию источников
    src_match = re.search(r"###\s*Источники\s*\n(.*?)(?=###\s*Цитаты|\Z)", text, re.DOTALL)
    if src_match:
        sources_section = src_match.group(1).strip()
        answer_text = text[:src_match.start()].strip()

    # Ищем секцию цитат
    quote_match = re.search(r"###\s*Цитаты\s*\n(.*?)$", text, re.DOTALL)
    if quote_match:
        quotes_section = quote_match.group(1).strip()
        if not src_match:
            answer_text = text[:quote_match.start()].strip()

    # Парсим источники: "- [1] файл / секция" или "- файл"
    sources: list[str] = []
    for line in sources_section.split("\n"):
        line = line.strip().lstrip("-").strip()
        if not line:
            continue
        # убираем нумерацию [1], [2], ...
        line = re.sub(r"^\[\d+\]\s*", "", line)
        if line:
            sources.append(line)

    # Парсим цитаты: "> [1] «цитата»" или "> цитата"
    quotes: list[str] = []
    for line in quotes_section.split("\n"):
        line = line.strip().lstrip(">").strip()
        if not line:
            continue
        # убираем нумерацию [1], [2], ...
        line = re.sub(r"^\[\d+\]\s*", "", line)
        # убираем кавычки «»
        line = line.strip("«»\"'")
        if line and len(line) > 5:
            quotes.append(line)

    return ParsedAnswer(
        answer_text=answer_text,
        sources=sources,
        quotes=quotes,
        is_idk=is_idk,
    )


# ── Данные retrieval для ответа ───────────────────────────────────

@dataclass
class RetrievedChunk:
    source: str
    section: str
    chunk_id: str
    text: str
    rerank_score: float


@dataclass
class CitedAnswer:
    question: str
    mode: str
    raw_answer: str
    parsed: ParsedAnswer
    retrieved_chunks: list[RetrievedChunk]
    input_tokens: int
    output_tokens: int
    elapsed_sec: float
    context_chars: int
    rewritten_query: str
    candidates_before: int
    candidates_after: int
    is_idk_forced: bool  # принудительный отказ (порог)
    best_rerank_score: float


# ── Targeted file retrieval: если вопрос про dayN, подтягиваем чанки ─

def _inject_targeted_chunks(
    question: str,
    existing: list[dict],
    strategy: str = "structural",
    max_inject: int = 3,
) -> list[dict]:
    """Если вопрос явно упоминает dayN, а в candidates нет чанков из dayN — добавляем."""
    days_mentioned = _DAY_RE.findall(question)
    if not days_mentioned:
        return existing

    existing_sources = {item["source"] for item in existing}

    _, meta = load_index(strategy)

    injected = list(existing)
    for day_num in days_mentioned:
        prefix = f"day{day_num}"
        # Уже есть чанки из этого дня?
        if any(prefix in src for src in existing_sources):
            continue
        # Берём самые длинные чанки из этого файла (обычно содержат больше информации)
        day_chunks = [
            m for m in meta
            if m["source"].startswith(prefix)
        ]
        day_chunks.sort(key=lambda c: len(c["text"]), reverse=True)
        for chunk in day_chunks[:max_inject]:
            entry = dict(chunk)
            entry["rerank_score"] = 0.35  # минимальный score для прохождения
            entry["overlap"] = 0.20
            entry["semantic_score"] = 0.0
            entry["keyword_score"] = 0.0
            entry["injected"] = True
            injected.append(entry)

    return injected


# ── Построение контекста с метаданными для цитирования ────────────

def build_context_cited(
    question: str,
    cfg: RetrievalConfig,
    relevance_threshold: float = RELEVANCE_THRESHOLD,
) -> tuple[str, list[RetrievedChunk], str, int, int, bool]:
    """
    Строим контекст с пронумерованными фрагментами.
    Возвращает: (context_str, chunks, rewritten_query, before_count, after_count, is_idk_forced)
    """
    rewritten = rewrite_query(question)
    candidates = collect_candidates(rewritten, top_k=cfg.top_k_before)
    filtered = rerank_and_filter(question, rewritten, candidates, cfg)

    # Targeted injection: если вопрос про dayN, а чанков из dayN нет → добавляем
    filtered = _inject_targeted_chunks(question, filtered)

    # Проверка порога: если лучший результат слишком слабый → отказ
    best_score = max((item.get("rerank_score", 0.0) for item in filtered), default=0.0)
    is_idk_forced = best_score < relevance_threshold

    chunks: list[RetrievedChunk] = []
    parts: list[str] = []

    for i, item in enumerate(filtered, 1):
        sec = item.get("section", "")
        chunk = RetrievedChunk(
            source=item["source"],
            section=sec,
            chunk_id=item["chunk_id"],
            text=item["text"],
            rerank_score=item.get("rerank_score", 0.0),
        )
        chunks.append(chunk)

        sec_label = f" / {sec}" if sec else ""
        score_info = f"score={item.get('rerank_score', 0):.2f}"
        parts.append(
            f"[Фрагмент {i}] {item['source']}{sec_label} ({score_info})\n"
            f"{item['text']}"
        )

    context = "\n\n---\n\n".join(parts) if parts else ""
    return context, chunks, rewritten, len(candidates), len(filtered), is_idk_forced


# ── Основной вызов: вопрос → структурированный ответ ──────────────

IDK_RESPONSE = (
    "Я не нашёл достаточно информации в предоставленном контексте. "
    "Пожалуйста, уточните вопрос или переформулируйте."
)


def ask_cited(
    client: Anthropic,
    model: str,
    question: str,
    cfg: RetrievalConfig | None = None,
    relevance_threshold: float = RELEVANCE_THRESHOLD,
) -> CitedAnswer:
    """RAG с обязательными цитатами и источниками."""
    if cfg is None:
        cfg = RetrievalConfig()

    context, chunks, rewritten, before, after, idk_forced = build_context_cited(
        question, cfg, relevance_threshold
    )

    best_score = max((c.rerank_score for c in chunks), default=0.0)

    if idk_forced or not context.strip():
        # Принудительный "не знаю"
        parsed = ParsedAnswer(
            answer_text=IDK_RESPONSE, sources=[], quotes=[], is_idk=True
        )
        return CitedAnswer(
            question=question,
            mode="cited",
            raw_answer=IDK_RESPONSE,
            parsed=parsed,
            retrieved_chunks=chunks,
            input_tokens=0,
            output_tokens=0,
            elapsed_sec=0.0,
            context_chars=len(context),
            rewritten_query=rewritten,
            candidates_before=before,
            candidates_after=after,
            is_idk_forced=True,
            best_rerank_score=best_score,
        )

    user_msg = format_cited_prompt(question, context)
    text, inp, out, sec = ask_llm(client, model, SYSTEM_CITED, user_msg)
    parsed = parse_llm_response(text)

    return CitedAnswer(
        question=question,
        mode="cited",
        raw_answer=text,
        parsed=parsed,
        retrieved_chunks=chunks,
        input_tokens=inp,
        output_tokens=out,
        elapsed_sec=sec,
        context_chars=len(context),
        rewritten_query=rewritten,
        candidates_before=before,
        candidates_after=after,
        is_idk_forced=False,
        best_rerank_score=best_score,
    )


# ═══════════════════════════════════════════════════════════════════
#  Оценка цитатного RAG
# ═══════════════════════════════════════════════════════════════════

def eval_quotes_match(
    client: Anthropic,
    model: str,
    question: str,
    answer_text: str,
    quotes: list[str],
) -> tuple[int, str]:
    """LLM-судья: совпадает ли смысл ответа с цитатами (1-5)."""
    if not quotes:
        return 1, "Цитаты отсутствуют"

    quotes_block = "\n".join(f"  - «{q}»" for q in quotes)
    judge_prompt = f"""\
Оцени, насколько ответ на вопрос подкреплён приведёнными цитатами.

Вопрос: {question}
Ответ: {answer_text}

Цитаты:
{quotes_block}

Критерии:
  5 — ответ полностью подкреплён цитатами, факты совпадают
  4 — ответ в основном подкреплён, мелкие расхождения
  3 — частичное соответствие, часть фактов не подтверждена
  2 — слабое соответствие, цитаты не подтверждают ключевые утверждения
  1 — цитаты не относятся к ответу или противоречат ему

Ответь СТРОГО в формате:
SCORE: <число>
REASON: <одно предложение>"""

    text, _, _, _ = ask_llm(
        client, model, "Ты — объективный оценщик соответствия цитат и ответа.", judge_prompt
    )
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
#  Демо — 10 вопросов
# ═══════════════════════════════════════════════════════════════════

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

    print("=" * 78)
    print("  День 24 — Цитаты, источники и анти-галлюцинации")
    print("=" * 78)
    print(f"  Модель: {model}")
    print(f"  Порог релевантности: {RELEVANCE_THRESHOLD}")
    print(f"  Rewrite + rerank: top_k_before={cfg.top_k_before}, "
          f"top_k_after={cfg.top_k_after}, threshold={cfg.similarity_threshold:.2f}")
    print()

    all_results: list[dict] = []

    for q in EVAL_QUESTIONS:
        print(f"\n{'━' * 78}")
        print(f"  [{q['id']}/10] {q['question']}")
        print(f"{'━' * 78}")
        print(f"  Ожидание: {q['expected']}")
        print(f"  Источники: {', '.join(q['sources'])}")

        ans = ask_cited(client, model, q["question"], cfg)

        # Оценки
        if ans.is_idk_forced:
            quality_score, quality_reason = 0, "Принудительный отказ (порог)"
            quote_score, quote_reason = 0, "Нет цитат (IDK)"
        else:
            quality_score, quality_reason = eval_answer_quality(
                client, model, q["question"], q["expected"], ans.parsed.answer_text
            )
            quote_score, quote_reason = eval_quotes_match(
                client, model, q["question"], ans.parsed.answer_text, ans.parsed.quotes
            )

        source_recall = eval_source_recall(q["sources"], [c.source for c in ans.retrieved_chunks])

        # Чеки
        has_sources = len(ans.parsed.sources) > 0
        has_quotes = len(ans.parsed.quotes) > 0
        is_idk = ans.parsed.is_idk

        print(f"\n  ▸ Ответ:")
        print(f"    {ans.parsed.answer_text[:300]}{'...' if len(ans.parsed.answer_text) > 300 else ''}")
        print(f"\n  ▸ Источники в ответе: {ans.parsed.sources if has_sources else '(нет)'}")
        print(f"  ▸ Цитаты в ответе ({len(ans.parsed.quotes)}):")
        for i, quote in enumerate(ans.parsed.quotes[:3], 1):
            print(f"    [{i}] «{quote[:120]}{'...' if len(quote) > 120 else ''}»")

        print(f"\n  ▸ Метрики:")
        print(f"    Rewrite:        {ans.rewritten_query}")
        print(f"    Кандидаты:      {ans.candidates_before} -> {ans.candidates_after}")
        print(f"    Best score:     {ans.best_rerank_score:.3f}")
        print(f"    IDK forced:     {'ДА' if ans.is_idk_forced else 'нет'}")
        print(f"    IDK detected:   {'ДА' if is_idk else 'нет'}")
        print(f"    Source recall:   {source_recall:.0%}")
        print(f"    Качество:       {quality_score}/5 — {quality_reason}")
        print(f"    Цитаты↔ответ:   {quote_score}/5 — {quote_reason}")
        print(f"    Есть источники: {'YES' if has_sources else 'NO'}")
        print(f"    Есть цитаты:    {'YES' if has_quotes else 'NO'}")
        if not ans.is_idk_forced:
            print(f"    [{ans.elapsed_sec}s | in={ans.input_tokens} out={ans.output_tokens} "
                  f"| ctx={ans.context_chars:,}с]")

        all_results.append({
            "id": q["id"],
            "question": q["question"],
            "expected": q["expected"],
            "expected_sources": q["sources"],
            "answer_text": ans.parsed.answer_text,
            "raw_answer": ans.raw_answer,
            "sources_in_answer": ans.parsed.sources,
            "quotes_in_answer": ans.parsed.quotes,
            "retrieved_sources": [c.source for c in ans.retrieved_chunks],
            "has_sources": has_sources,
            "has_quotes": has_quotes,
            "is_idk": is_idk,
            "is_idk_forced": ans.is_idk_forced,
            "best_rerank_score": ans.best_rerank_score,
            "quality_score": quality_score,
            "quality_reason": quality_reason,
            "quote_score": quote_score,
            "quote_reason": quote_reason,
            "source_recall": source_recall,
            "input_tokens": ans.input_tokens,
            "output_tokens": ans.output_tokens,
            "elapsed_sec": ans.elapsed_sec,
            "context_chars": ans.context_chars,
            "rewritten_query": ans.rewritten_query,
            "candidates_before": ans.candidates_before,
            "candidates_after": ans.candidates_after,
        })

    # ── Сводка ──
    print_summary(all_results)

    # ── Сохранение ──
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n  Результаты: {out_path}")


def print_summary(results: list[dict]):
    """Сводная таблица day24."""
    print("\n\n" + "=" * 78)
    print("  СВОДКА: ЦИТАТЫ, ИСТОЧНИКИ, АНТИ-ГАЛЛЮЦИНАЦИИ")
    print("=" * 78)

    total = len(results)
    with_sources = sum(1 for r in results if r["has_sources"])
    with_quotes = sum(1 for r in results if r["has_quotes"])
    idk_count = sum(1 for r in results if r["is_idk"] or r["is_idk_forced"])
    idk_forced = sum(1 for r in results if r["is_idk_forced"])

    quality_scores = [r["quality_score"] for r in results if not r["is_idk_forced"]]
    quote_scores = [r["quote_score"] for r in results if not r["is_idk_forced"]]
    recalls = [r["source_recall"] for r in results]

    print(f"\n  {'#':<4} {'Качество':>10} {'Цит↔отв':>10} {'Recall':>8} {'Src':>4} {'Quot':>5} {'IDK':>4}")
    print(f"  {'─'*4} {'─'*10} {'─'*10} {'─'*8} {'─'*4} {'─'*5} {'─'*4}")

    for r in results:
        qs = f"{r['quality_score']}/5" if not r["is_idk_forced"] else "IDK"
        cs = f"{r['quote_score']}/5" if not r["is_idk_forced"] else "IDK"
        src = "YES" if r["has_sources"] else "NO"
        quot = "YES" if r["has_quotes"] else "NO"
        idk = "YES" if r["is_idk"] or r["is_idk_forced"] else "-"
        print(f"  {r['id']:<4} {qs:>10} {cs:>10} {r['source_recall']:>7.0%} {src:>4} {quot:>5} {idk:>4}")

    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
    avg_quotes = sum(quote_scores) / len(quote_scores) if quote_scores else 0
    avg_recall = sum(recalls) / len(recalls) if recalls else 0

    print(f"\n  Результаты ({total} вопросов):")
    print(f"    С источниками:    {with_sources}/{total} ({with_sources*100//total}%)")
    print(f"    С цитатами:       {with_quotes}/{total} ({with_quotes*100//total}%)")
    print(f"    Режим 'не знаю':  {idk_count}/{total} (forced: {idk_forced})")
    print(f"    Ср. качество:     {avg_quality:.1f}/5")
    print(f"    Ср. цитаты↔ответ: {avg_quotes:.1f}/5")
    print(f"    Ср. source recall: {avg_recall:.0%}")
    print("=" * 78)


# ═══════════════════════════════════════════════════════════════════
#  Интерактивный режим
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
    print("  День 24 — RAG с цитатами и анти-галлюцинациями")
    print("=" * 68)
    print(f"  Порог релевантности: {RELEVANCE_THRESHOLD}")
    print(f"  Rewrite + rerank → top_k_after={cfg.top_k_after}")
    print("  Пустая строка — выход.\n")

    while True:
        query = input("[cited]> ").strip()
        if not query:
            break

        ans = ask_cited(client, model, query, cfg)

        if ans.is_idk_forced:
            print(f"\n  {IDK_RESPONSE}")
            print(f"  [IDK forced | best_score={ans.best_rerank_score:.3f} < {RELEVANCE_THRESHOLD}]")
            print()
            continue

        # Ответ
        print(f"\n  {ans.parsed.answer_text}")

        # Источники
        if ans.parsed.sources:
            print("\n  Источники:")
            for s in ans.parsed.sources:
                print(f"    - {s}")
        else:
            print("\n  Источники: (модель не указала)")

        # Цитаты
        if ans.parsed.quotes:
            print("\n  Цитаты:")
            for i, q in enumerate(ans.parsed.quotes, 1):
                print(f"    [{i}] «{q[:150]}{'...' if len(q) > 150 else ''}»")
        else:
            print("\n  Цитаты: (модель не привела)")

        # Мета
        print(f"\n  [{ans.elapsed_sec}s | in={ans.input_tokens} out={ans.output_tokens} "
              f"| ctx={ans.context_chars:,}с | score={ans.best_rerank_score:.3f} "
              f"| cand={ans.candidates_before}->{ans.candidates_after}]")
        print(f"  Rewrite: {ans.rewritten_query}")
        print(f"  Retrieval sources: {', '.join(c.source for c in ans.retrieved_chunks)}")
        print()


# ═══════════════════════════════════════════════════════════════════
#  Демо режима "не знаю"
# ═══════════════════════════════════════════════════════════════════

IDK_TEST_QUESTIONS = [
    "Как устроена интеграция с PostgreSQL в проекте?",
    "Какой фреймворк для фронтенда используется?",
    "Как реализована авторизация через OAuth?",
]


def run_idk_demo(
    client: Anthropic | None = None,
    model: str = MODEL,
    cfg: RetrievalConfig | None = None,
):
    """Демо режима 'не знаю': вопросы, на которые в проекте нет ответа."""
    if cfg is None:
        cfg = RetrievalConfig()
    if client is None:
        load_dotenv(PROJECT_ROOT / ".env")
        client = Anthropic()

    print("\n" + "=" * 68)
    print("  Демо: режим 'не знаю' (вопросы вне контекста проекта)")
    print("=" * 68)
    print(f"  Порог релевантности: {RELEVANCE_THRESHOLD}\n")

    for i, question in enumerate(IDK_TEST_QUESTIONS, 1):
        print(f"  [{i}] {question}")
        ans = ask_cited(client, model, question, cfg)
        status = "IDK (forced)" if ans.is_idk_forced else ("IDK (detected)" if ans.parsed.is_idk else "ОТВЕТ")
        print(f"      Статус: {status}")
        print(f"      Best score: {ans.best_rerank_score:.3f}")
        if not ans.is_idk_forced:
            print(f"      Ответ: {ans.parsed.answer_text[:200]}")
        print()


# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    load_dotenv(PROJECT_ROOT / ".env")
    client = Anthropic()

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        run_interactive(client)
    elif len(sys.argv) > 1 and sys.argv[1] == "--idk":
        run_idk_demo(client)
    else:
        run_demo(client)

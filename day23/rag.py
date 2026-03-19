"""
День 23. Реранкинг и фильтрация

Улучшение поверх day22:
  1. Query rewrite (heuristic)
  2. Retrieval top-K до фильтрации
  3. Второй этап: rerank + relevance filter
  4. Сравнение baseline vs improved
"""

from __future__ import annotations

import json
import re
import sys
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
    SYSTEM_RAG,
    ask_llm,
    eval_answer_quality,
    eval_source_recall,
    format_rag_prompt,
)


RESULTS_DIR = Path(__file__).resolve().parent / "results"
STRATEGY = "structural"

TOP_K_BEFORE = 10
TOP_K_AFTER = 4
SIMILARITY_THRESHOLD = 0.34
MIN_TOKEN_OVERLAP = 0.18

_WORD_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9_]+")
_DAY_RE = re.compile(r"\bday(\d{1,2})\b", re.IGNORECASE)

QUERY_HINTS: dict[str, list[str]] = {
    "контекст": ["context", "sliding", "window"],
    "стратегии": ["strategies"],
    "память": ["memory", "remember", "forget"],
    "памяти": ["memory", "remember", "forget"],
    "команды": ["cli", "command"],
    "состояния": ["state", "fsm"],
    "этапы": ["state", "fsm"],
    "инструменты": ["tool", "tools"],
    "сервер": ["server"],
    "серверам": ["server"],
    "оркестратор": ["orchestrator", "agent"],
    "планировщик": ["scheduler", "job"],
    "хранит": ["sqlite", "db"],
    "хранение": ["sqlite", "db"],
    "эмбеддингов": ["embedding", "sentence-transformers"],
    "эмбеддинги": ["embedding", "sentence-transformers"],
    "индексации": ["index", "faiss"],
    "индекс": ["index", "faiss"],
    "стоимость": ["cost", "tokens"],
    "поиск": ["search", "retrieval"],
    "pipeline": ["pipeline"],
    "композиция": ["pipeline"],
    "модель": ["model"],
}


@dataclass
class RetrievalConfig:
    top_k_before: int = TOP_K_BEFORE
    top_k_after: int = TOP_K_AFTER
    similarity_threshold: float = SIMILARITY_THRESHOLD
    min_token_overlap: float = MIN_TOKEN_OVERLAP


@dataclass
class RetrievalStats:
    rewritten_query: str
    candidates_before: int
    candidates_after: int
    filter_threshold: float
    top_k_before: int
    top_k_after: int


@dataclass
class Answer:
    question: str
    mode: str
    answer: str
    sources: list[str]
    input_tokens: int
    output_tokens: int
    elapsed_sec: float
    context_chars: int
    rewritten_query: str
    candidates_before: int
    candidates_after: int


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        out.append(value)
        seen.add(value)
    return out


def keyword_search(query: str, strategy: str = STRATEGY, top_k: int = TOP_K_BEFORE) -> list[dict]:
    _, meta = load_index(strategy)
    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return []

    scored: list[tuple[float, int]] = []
    for i, m in enumerate(meta):
        text_tokens = _tokenize(f"{m['source']} {m.get('section', '')} {m['text']}")
        if not text_tokens:
            continue
        text_token_set = set(text_tokens)
        hit_words = query_tokens & text_token_set
        if not hit_words:
            continue
        coverage = len(hit_words) / max(len(query_tokens), 1)
        tf_bonus = sum(text_tokens.count(w) for w in hit_words) * 0.08
        score = coverage + tf_bonus
        scored.append((score, i))

    scored.sort(reverse=True)
    results = []
    for score, idx in scored[:top_k]:
        entry = dict(meta[idx])
        entry["score"] = round(score, 5)
        results.append(entry)
    return results


def safe_semantic_search(query: str, strategy: str = STRATEGY, top_k: int = TOP_K_BEFORE) -> list[dict]:
    try:
        return faiss_search(query, strategy=strategy, top_k=top_k)
    except Exception as exc:
        print(f"  [warn] semantic search unavailable, fallback to keyword only: {exc}")
        return []


def rewrite_query(question: str) -> str:
    tokens = _tokenize(question)
    expanded: list[str] = list(tokens)

    for token in tokens:
        expanded.extend(QUERY_HINTS.get(token, []))

    for day in _DAY_RE.findall(question):
        expanded.extend([f"day{day}", f"day{day}.py", f"day{day}/server.py", f"day{day}/agent.py"])

    if "mcp" in tokens:
        expanded.extend(["mcp", "server", "tool"])
    if "faiss" in tokens or "индекс" in tokens or "индексации" in tokens:
        expanded.extend(["faiss", "indexflatip", "all-minilm-l6-v2"])
    if "cli" not in expanded and ("команды" in tokens or "команд" in tokens):
        expanded.append("cli")

    return " ".join(_dedupe_keep_order(expanded))


def collect_candidates(query: str, top_k: int, strategy: str = STRATEGY) -> list[dict]:
    sem_results = safe_semantic_search(query, strategy=strategy, top_k=top_k)
    kw_results = keyword_search(query, strategy=strategy, top_k=top_k)

    merged: dict[str, dict] = {}
    for rank, item in enumerate(sem_results, 1):
        entry = merged.setdefault(item["chunk_id"], dict(item))
        entry["semantic_score"] = float(item["score"])
        entry["semantic_rank"] = rank
    for rank, item in enumerate(kw_results, 1):
        entry = merged.setdefault(item["chunk_id"], dict(item))
        entry["keyword_score"] = float(item["score"])
        entry["keyword_rank"] = rank

    results = list(merged.values())
    results.sort(
        key=lambda item: (
            item.get("semantic_rank", 10**9),
            item.get("keyword_rank", 10**9),
            -item.get("keyword_score", 0.0),
        )
    )
    return results[:top_k]


def rerank_and_filter(
    question: str,
    rewritten_query: str,
    candidates: list[dict],
    cfg: RetrievalConfig,
) -> list[dict]:
    query_tokens = set(_tokenize(rewritten_query))
    day_match = _DAY_RE.findall(question)
    target_day = day_match[0] if day_match else None

    reranked: list[dict] = []
    for item in candidates:
        blob = f"{item['source']} {item.get('section', '')} {item['text']}"
        blob_tokens = set(_tokenize(blob))
        overlap = len(query_tokens & blob_tokens) / max(len(query_tokens), 1)

        sem_score = max(float(item.get("semantic_score", 0.0)), 0.0)
        kw_score = max(float(item.get("keyword_score", 0.0)), 0.0)
        kw_norm = min(kw_score, 1.0)

        file_bonus = 0.0
        if target_day and f"day{target_day}" in item["source"].lower():
            file_bonus += 0.25
        if any(token in item["source"].lower() for token in ("server.py", "agent.py", ".md")):
            file_bonus += 0.05

        score = 0.45 * overlap + 0.30 * sem_score + 0.15 * kw_norm + 0.10 * file_bonus

        reranked_item = dict(item)
        reranked_item["overlap"] = round(overlap, 5)
        reranked_item["rerank_score"] = round(score, 5)
        reranked.append(reranked_item)

    reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    filtered = [
        item for item in reranked
        if item["rerank_score"] >= cfg.similarity_threshold
        and item["overlap"] >= cfg.min_token_overlap
    ]

    if not filtered:
        filtered = reranked[: cfg.top_k_after]
    return filtered[: cfg.top_k_after]


def build_context_baseline(question: str, cfg: RetrievalConfig) -> tuple[str, list[str], RetrievalStats]:
    candidates = collect_candidates(question, top_k=cfg.top_k_after)
    parts: list[str] = []
    sources: list[str] = []
    for i, item in enumerate(candidates[: cfg.top_k_after], 1):
        sec = f" / {item['section']}" if item.get("section") else ""
        parts.append(f"[{i}] {item['source']}{sec}\n{item['text']}")
        sources.append(item["source"])

    return (
        "\n\n---\n\n".join(parts),
        sources,
        RetrievalStats(
            rewritten_query=question,
            candidates_before=len(candidates),
            candidates_after=len(candidates[: cfg.top_k_after]),
            filter_threshold=0.0,
            top_k_before=cfg.top_k_after,
            top_k_after=cfg.top_k_after,
        ),
    )


def build_context_improved(question: str, cfg: RetrievalConfig) -> tuple[str, list[str], RetrievalStats]:
    rewritten_query = rewrite_query(question)
    candidates = collect_candidates(rewritten_query, top_k=cfg.top_k_before)
    filtered = rerank_and_filter(question, rewritten_query, candidates, cfg)

    parts: list[str] = []
    sources: list[str] = []
    for i, item in enumerate(filtered, 1):
        sec = f" / {item['section']}" if item.get("section") else ""
        score_info = (
            f"score={item['rerank_score']:.2f}, overlap={item['overlap']:.2f}, "
            f"sem={item.get('semantic_score', 0.0):.2f}, kw={item.get('keyword_score', 0.0):.2f}"
        )
        parts.append(f"[{i}] {item['source']}{sec} ({score_info})\n{item['text']}")
        sources.append(item["source"])

    return (
        "\n\n---\n\n".join(parts),
        sources,
        RetrievalStats(
            rewritten_query=rewritten_query,
            candidates_before=len(candidates),
            candidates_after=len(filtered),
            filter_threshold=cfg.similarity_threshold,
            top_k_before=cfg.top_k_before,
            top_k_after=cfg.top_k_after,
        ),
    )


def ask_rag(
    client: Anthropic,
    model: str,
    question: str,
    mode: str,
    cfg: RetrievalConfig,
) -> Answer:
    if mode == "baseline":
        context, sources, stats = build_context_baseline(question, cfg)
    elif mode == "improved":
        context, sources, stats = build_context_improved(question, cfg)
    else:
        raise ValueError(f"unknown mode: {mode}")

    user_msg = format_rag_prompt(question, context)
    text, inp, out, sec = ask_llm(client, model, SYSTEM_RAG, user_msg)
    return Answer(
        question=question,
        mode=mode,
        answer=text,
        sources=sources,
        input_tokens=inp,
        output_tokens=out,
        elapsed_sec=sec,
        context_chars=len(context),
        rewritten_query=stats.rewritten_query,
        candidates_before=stats.candidates_before,
        candidates_after=stats.candidates_after,
    )


def print_summary(results: list[dict], cfg: RetrievalConfig):
    print("\n\n" + "=" * 78)
    print("  СВОДКА: BASELINE RAG vs IMPROVED RAG")
    print("=" * 78)
    print(
        f"  Конфиг improved: top_k_before={cfg.top_k_before}, "
        f"top_k_after={cfg.top_k_after}, threshold={cfg.similarity_threshold:.2f}, "
        f"min_overlap={cfg.min_token_overlap:.2f}"
    )

    base_scores: list[int] = []
    imp_scores: list[int] = []
    base_recalls: list[float] = []
    imp_recalls: list[float] = []

    print(f"\n  {'#':<4} {'Base':>7} {'Imp':>7} {'BaseRec':>8} {'ImpRec':>8} {'Δctx':>8}")
    print(f"  {'─' * 4} {'─' * 7} {'─' * 7} {'─' * 8} {'─' * 8} {'─' * 8}")

    for row in results:
        base = row["baseline"]
        imp = row["improved"]
        base_scores.append(base["score"])
        imp_scores.append(imp["score"])
        base_recalls.append(base["source_recall"])
        imp_recalls.append(imp["source_recall"])
        ctx_delta = imp["context_chars"] - base["context_chars"]
        print(
            f"  {row['id']:<4} {base['score']:>5}/5 {imp['score']:>5}/5 "
            f"{base['source_recall']:>7.0%} {imp['source_recall']:>7.0%} {ctx_delta:>+8,}"
        )

    avg_base = sum(base_scores) / len(base_scores)
    avg_imp = sum(imp_scores) / len(imp_scores)
    avg_base_recall = sum(base_recalls) / len(base_recalls)
    avg_imp_recall = sum(imp_recalls) / len(imp_recalls)

    print(f"\n  Baseline score:  {avg_base:.1f}/5")
    print(f"  Improved score:  {avg_imp:.1f}/5  (delta: {avg_imp - avg_base:+.1f})")
    print(f"  Baseline recall: {avg_base_recall:.0%}")
    print(f"  Improved recall: {avg_imp_recall:.0%}")
    print("=" * 78)


def run_demo(client: Anthropic | None = None, model: str = MODEL, cfg: RetrievalConfig | None = None):
    if cfg is None:
        cfg = RetrievalConfig()
    if client is None:
        load_dotenv(PROJECT_ROOT / ".env")
        client = Anthropic()

    print("=" * 78)
    print("  День 23 — Улучшенный RAG: rewrite + rerank/filter")
    print("=" * 78)
    print(f"  Модель: {model}")
    print(f"  Индекс: {STRATEGY}")
    print(f"  Baseline: top_k={cfg.top_k_after} без rewrite/фильтра")
    print(
        f"  Improved: rewrite + top_k_before={cfg.top_k_before} -> "
        f"filter(threshold={cfg.similarity_threshold:.2f}, overlap>={cfg.min_token_overlap:.2f}) "
        f"-> top_k_after={cfg.top_k_after}"
    )

    all_results: list[dict] = []

    for q in EVAL_QUESTIONS:
        print(f"\n{'━' * 78}")
        print(f"  [{q['id']}/10] {q['question']}")
        print(f"{'━' * 78}")
        print(f"  Ожидание: {q['expected']}")
        print(f"  Источники: {', '.join(q['sources'])}")

        print("\n  ▸ Baseline RAG...")
        ans_base = ask_rag(client, model, q["question"], "baseline", cfg)
        score_base, reason_base = eval_answer_quality(client, model, q["question"], q["expected"], ans_base.answer)
        recall_base = eval_source_recall(q["sources"], ans_base.sources)
        print(
            f"    [{ans_base.elapsed_sec}s | in={ans_base.input_tokens} out={ans_base.output_tokens} "
            f"| ctx={ans_base.context_chars:,}с]"
        )
        print(f"    Источники: {', '.join(ans_base.sources)}")
        print(f"    Recall: {recall_base:.0%}")
        print(f"    Оценка: {score_base}/5 — {reason_base}")

        print("\n  ▸ Improved RAG...")
        ans_imp = ask_rag(client, model, q["question"], "improved", cfg)
        score_imp, reason_imp = eval_answer_quality(client, model, q["question"], q["expected"], ans_imp.answer)
        recall_imp = eval_source_recall(q["sources"], ans_imp.sources)
        print(
            f"    [{ans_imp.elapsed_sec}s | in={ans_imp.input_tokens} out={ans_imp.output_tokens} "
            f"| ctx={ans_imp.context_chars:,}с]"
        )
        print(f"    Rewritten: {ans_imp.rewritten_query}")
        print(f"    Кандидаты: {ans_imp.candidates_before} -> {ans_imp.candidates_after}")
        print(f"    Источники: {', '.join(ans_imp.sources)}")
        print(f"    Recall: {recall_imp:.0%}")
        print(f"    Оценка: {score_imp}/5 — {reason_imp}")

        all_results.append({
            "id": q["id"],
            "question": q["question"],
            "expected": q["expected"],
            "expected_sources": q["sources"],
            "baseline": {
                **asdict(ans_base),
                "score": score_base,
                "reason": reason_base,
                "source_recall": recall_base,
            },
            "improved": {
                **asdict(ans_imp),
                "score": score_imp,
                "reason": reason_imp,
                "source_recall": recall_imp,
            },
        })

    print_summary(all_results, cfg)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": asdict(cfg),
                "results": all_results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n  Результаты: {out_path}")


def run_interactive(client: Anthropic | None = None, model: str = MODEL, cfg: RetrievalConfig | None = None):
    if cfg is None:
        cfg = RetrievalConfig()
    if client is None:
        load_dotenv(PROJECT_ROOT / ".env")
        client = Anthropic()

    print("=" * 68)
    print("  День 23 — Улучшенный RAG")
    print("=" * 68)
    print(f"  Baseline: top_k={cfg.top_k_after}")
    print(
        f"  Improved: rewrite + top_k_before={cfg.top_k_before} -> "
        f"filter({cfg.similarity_threshold:.2f}) -> top_k_after={cfg.top_k_after}"
    )
    print("  /base  — baseline RAG")
    print("  /imp   — improved RAG")
    print("  /both  — оба режима")
    print("  Пустая строка — выход.\n")

    mode = "improved"

    while True:
        query = input(f"[{mode}]> ").strip()
        if not query:
            break
        if query == "/base":
            mode = "baseline"
            print("  Режим: baseline.")
            continue
        if query == "/imp":
            mode = "improved"
            print("  Режим: improved.")
            continue
        if query == "/both":
            mode = "both"
            print("  Режим: оба.")
            continue

        modes = ["baseline", "improved"] if mode == "both" else [mode]
        for current_mode in modes:
            label = "Baseline RAG" if current_mode == "baseline" else "Improved RAG"
            print(f"\n  ── {label} ──")
            ans = ask_rag(client, model, query, current_mode, cfg)
            print(f"  {ans.answer}")
            print(
                f"  [{ans.elapsed_sec}s | in={ans.input_tokens} out={ans.output_tokens} "
                f"| ctx={ans.context_chars:,}с | cand={ans.candidates_before}->{ans.candidates_after}]"
            )
            if current_mode == "improved":
                print(f"  Rewrite: {ans.rewritten_query}")
            print(f"  Источники: {', '.join(ans.sources)}")
        print()


if __name__ == "__main__":
    load_dotenv(PROJECT_ROOT / ".env")
    client = Anthropic()

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        run_interactive(client)
    else:
        run_demo(client)

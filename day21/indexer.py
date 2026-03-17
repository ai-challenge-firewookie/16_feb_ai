"""
День 21. Индексация документов

Пайплайн:
1. Сбор документов (*.py, *.md) из проекта
2. Разбиение на чанки (2 стратегии)
3. Генерация эмбеддингов (sentence-transformers, локально)
4. Сохранение индекса (FAISS) + метаданные (JSON)
5. Сравнение двух стратегий chunking
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

# ── Пути ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = Path(__file__).resolve().parent / "index"

# ── Модель эмбеддингов (локальная, без API) ───────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384               # размерность all-MiniLM-L6-v2
BATCH_SIZE = 64


# ── Dataclass для чанка ───────────────────────────────────────────────
@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str           # путь к файлу (относительный)
    title: str            # имя файла
    section: str          # заголовок секции / ""
    strategy: str         # "fixed" | "structural"
    char_len: int = 0
    token_est: int = 0    # грубая оценка ~4 символа ≈ 1 токен

    def __post_init__(self):
        self.char_len = len(self.text)
        self.token_est = self.char_len // 4


# ═══════════════════════════════════════════════════════════════════════
#  1. Сбор документов
# ═══════════════════════════════════════════════════════════════════════

def collect_documents(root: Path = PROJECT_ROOT) -> list[dict]:
    """Собираем .py и .md файлы из проекта (кроме .venv, __pycache__ и т.д.)."""
    skip = {".venv", "__pycache__", ".git", ".idea", "node_modules", ".claude",
            "index", "day21"}
    docs: list[dict] = []
    for p in sorted(root.rglob("*")):
        if any(part in skip for part in p.parts):
            continue
        if p.suffix in (".py", ".md") and p.is_file():
            try:
                text = p.read_text(encoding="utf-8")
            except Exception:
                continue
            if len(text.strip()) < 20:
                continue
            docs.append({
                "path": str(p.relative_to(root)),
                "title": p.name,
                "text": text,
                "type": "python" if p.suffix == ".py" else "markdown",
            })
    return docs


# ═══════════════════════════════════════════════════════════════════════
#  2. Стратегии chunking
# ═══════════════════════════════════════════════════════════════════════

def _make_id(source: str, idx: int, strategy: str) -> str:
    raw = f"{strategy}:{source}:{idx}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


# ── 2a. Фиксированный размер ─────────────────────────────────────────

def chunk_fixed(
    docs: list[dict],
    size: int = 800,
    overlap: int = 200,
) -> list[Chunk]:
    """Разбиение на чанки фиксированного размера (в символах) с перекрытием."""
    chunks: list[Chunk] = []
    for doc in docs:
        text = doc["text"]
        start = 0
        idx = 0
        while start < len(text):
            end = start + size
            chunk_text = text[start:end]
            chunks.append(Chunk(
                chunk_id=_make_id(doc["path"], idx, "fixed"),
                text=chunk_text,
                source=doc["path"],
                title=doc["title"],
                section="",
                strategy="fixed",
            ))
            idx += 1
            start += size - overlap
    return chunks


# ── 2b. Структурное разбиение ────────────────────────────────────────

_MD_HEADING = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
_PY_DEF = re.compile(
    r"^(class\s+\w+|def\s+\w+|async\s+def\s+\w+)",
    re.MULTILINE,
)


def _split_markdown_sections(text: str) -> list[tuple[str, str]]:
    """Разделяем markdown по заголовкам → [(section_title, section_text), ...]."""
    headings = list(_MD_HEADING.finditer(text))
    if not headings:
        return [("", text)]
    sections: list[tuple[str, str]] = []
    if headings[0].start() > 0:
        sections.append(("preamble", text[: headings[0].start()].strip()))
    for i, m in enumerate(headings):
        title = m.group(2).strip()
        start = m.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((title, body))
    return sections


def _split_python_sections(text: str) -> list[tuple[str, str]]:
    """Разделяем Python-файл по top-level определениям (class/def)."""
    defs = list(_PY_DEF.finditer(text))
    if not defs:
        return [("module", text)]
    sections: list[tuple[str, str]] = []
    if defs[0].start() > 0:
        sections.append(("module_header", text[: defs[0].start()].strip()))
    for i, m in enumerate(defs):
        title = m.group(1).strip()
        start = m.start()
        end = defs[i + 1].start() if i + 1 < len(defs) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((title, body))
    return sections


def chunk_structural(docs: list[dict], max_chunk: int = 1500) -> list[Chunk]:
    """Разбиение по структуре документа (заголовки / определения).
    Если секция > max_chunk — дополнительно нарезаем по фиксированному размеру.
    """
    chunks: list[Chunk] = []
    for doc in docs:
        if doc["type"] == "markdown":
            sections = _split_markdown_sections(doc["text"])
        else:
            sections = _split_python_sections(doc["text"])

        idx = 0
        for sec_title, sec_text in sections:
            if len(sec_text) <= max_chunk:
                chunks.append(Chunk(
                    chunk_id=_make_id(doc["path"], idx, "structural"),
                    text=sec_text,
                    source=doc["path"],
                    title=doc["title"],
                    section=sec_title,
                    strategy="structural",
                ))
                idx += 1
            else:
                start = 0
                while start < len(sec_text):
                    end = start + max_chunk
                    chunks.append(Chunk(
                        chunk_id=_make_id(doc["path"], idx, "structural"),
                        text=sec_text[start:end],
                        source=doc["path"],
                        title=doc["title"],
                        section=sec_title,
                        strategy="structural",
                    ))
                    idx += 1
                    start += max_chunk
    return chunks


# ═══════════════════════════════════════════════════════════════════════
#  3. Генерация эмбеддингов (sentence-transformers, локально)
# ═══════════════════════════════════════════════════════════════════════

_model_cache = None

def _get_model():
    global _model_cache
    if _model_cache is None:
        from sentence_transformers import SentenceTransformer
        print(f"  Загрузка модели {EMBED_MODEL}...")
        _model_cache = SentenceTransformer(EMBED_MODEL)
    return _model_cache


def embed_chunks(chunks: list[Chunk]) -> np.ndarray:
    """Генерируем эмбеддинги локально через sentence-transformers."""
    model = _get_model()
    texts = [c.text for c in chunks]
    print(f"  Кодирование {len(texts)} чанков...")
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,   # для cosine similarity
    )
    elapsed = time.time() - t0
    print(f"  Готово за {elapsed:.1f}s")
    return np.array(embeddings, dtype=np.float32)


def embed_query(query: str) -> np.ndarray:
    """Эмбеддинг одного запроса."""
    model = _get_model()
    emb = model.encode([query], normalize_embeddings=True)
    return np.array(emb, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════
#  4. Сохранение / загрузка индекса (FAISS + JSON метаданные)
# ═══════════════════════════════════════════════════════════════════════

def save_index(
    chunks: list[Chunk],
    embeddings: np.ndarray,
    strategy: str,
    out_dir: Path = INDEX_DIR,
):
    """Сохраняем FAISS-индекс и метаданные в JSON."""
    import faiss

    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"index_{strategy}"

    # FAISS IndexFlatIP (inner product ≈ cosine, т.к. векторы нормализованы)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(out_dir / f"{prefix}.faiss"))

    # метаданные
    meta = [asdict(c) for c in chunks]
    with open(out_dir / f"{prefix}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"  saved: {prefix}.faiss ({index.ntotal} vectors, dim={embeddings.shape[1]})")
    print(f"  saved: {prefix}_meta.json ({len(meta)} chunks)")


def load_index(strategy: str, index_dir: Path = INDEX_DIR):
    """Загружаем FAISS-индекс и метаданные."""
    import faiss

    prefix = f"index_{strategy}"
    index = faiss.read_index(str(index_dir / f"{prefix}.faiss"))
    with open(index_dir / f"{prefix}_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta


# ═══════════════════════════════════════════════════════════════════════
#  5. Поиск
# ═══════════════════════════════════════════════════════════════════════

def search(query: str, strategy: str, top_k: int = 5) -> list[dict]:
    """Поиск ближайших чанков по запросу."""
    index, meta = load_index(strategy)
    q_vec = embed_query(query)

    scores, ids = index.search(q_vec, top_k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        entry = dict(meta[idx])
        entry["score"] = float(score)
        results.append(entry)
    return results


# ═══════════════════════════════════════════════════════════════════════
#  6. Сравнение стратегий
# ═══════════════════════════════════════════════════════════════════════

def compare_strategies(
    fixed_chunks: list[Chunk],
    structural_chunks: list[Chunk],
) -> dict:
    """Статистическое сравнение двух стратегий chunking."""
    def stats(chunks: list[Chunk]) -> dict:
        lengths = [c.char_len for c in chunks]
        tokens = [c.token_est for c in chunks]
        sources = {c.source for c in chunks}
        return {
            "total_chunks": len(chunks),
            "total_chars": sum(lengths),
            "total_tokens_est": sum(tokens),
            "avg_chunk_chars": round(sum(lengths) / len(lengths), 1) if lengths else 0,
            "min_chunk_chars": min(lengths) if lengths else 0,
            "max_chunk_chars": max(lengths) if lengths else 0,
            "median_chunk_chars": round(float(np.median(lengths)), 1) if lengths else 0,
            "unique_sources": len(sources),
            "chunks_with_section": sum(1 for c in chunks if c.section),
        }

    return {
        "fixed": stats(fixed_chunks),
        "structural": stats(structural_chunks),
    }


def print_comparison(comparison: dict):
    """Красивый вывод сравнения."""
    print("\n" + "=" * 70)
    print("  СРАВНЕНИЕ СТРАТЕГИЙ CHUNKING")
    print("=" * 70)

    header = f"{'Метрика':<30} {'Fixed':>18} {'Structural':>18}"
    print(header)
    print("─" * 70)

    keys = [
        ("total_chunks", "Всего чанков"),
        ("total_chars", "Всего символов"),
        ("total_tokens_est", "≈ Токенов"),
        ("avg_chunk_chars", "Средний размер (сим.)"),
        ("min_chunk_chars", "Мин. размер"),
        ("max_chunk_chars", "Макс. размер"),
        ("median_chunk_chars", "Медиана размера"),
        ("unique_sources", "Уник. файлов"),
        ("chunks_with_section", "Чанков с section"),
    ]
    for key, label in keys:
        v_fixed = comparison["fixed"][key]
        v_struct = comparison["structural"][key]
        print(f"  {label:<28} {v_fixed:>18,} {v_struct:>18,}")

    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════
#  7. Демо поиска
# ═══════════════════════════════════════════════════════════════════════

def demo_search(queries: list[str] | None = None):
    """Запуск поиска по обоим индексам для нескольких запросов."""
    if queries is None:
        queries = [
            "как устроена работа с токенами",
            "MCP сервер инструменты",
            "sliding window управление контекстом",
            "FSM конечный автомат задачи",
        ]

    for q in queries:
        print(f"\n{'─' * 60}")
        print(f"  Запрос: {q}")
        print(f"{'─' * 60}")
        for strategy in ("fixed", "structural"):
            print(f"\n  [{strategy}]")
            results = search(q, strategy, top_k=3)
            for i, r in enumerate(results, 1):
                sec = f" § {r['section']}" if r.get("section") else ""
                print(f"    {i}. [{r['score']:.4f}] {r['source']}{sec}")
                preview = r["text"][:120].replace("\n", " ")
                print(f"       {preview}...")


# ═══════════════════════════════════════════════════════════════════════
#  MAIN — полный пайплайн
# ═══════════════════════════════════════════════════════════════════════

def run_pipeline():
    """Полный пайплайн индексации: сбор → chunking → embedding → сохранение."""
    print("=" * 60)
    print("  День 21 — Индексация документов")
    print("=" * 60)

    # 1. Сбор
    print("\n[1/5] Сбор документов...")
    docs = collect_documents()
    total_chars = sum(len(d["text"]) for d in docs)
    print(f"  Найдено {len(docs)} файлов, {total_chars:,} символов "
          f"(~{total_chars // 4:,} токенов)")
    for d in docs:
        print(f"    {d['path']} ({len(d['text']):,} сим.)")

    # 2. Chunking
    print("\n[2/5] Chunking (2 стратегии)...")
    fixed = chunk_fixed(docs, size=800, overlap=200)
    structural = chunk_structural(docs, max_chunk=1500)
    print(f"  Fixed:      {len(fixed)} чанков")
    print(f"  Structural: {len(structural)} чанков")

    # 3. Сравнение
    print("\n[3/5] Сравнение стратегий...")
    comparison = compare_strategies(fixed, structural)
    print_comparison(comparison)

    # 4. Эмбеддинги
    print("\n[4/5] Генерация эмбеддингов (sentence-transformers)...")
    print(f"  Модель: {EMBED_MODEL}, dim={EMBED_DIM}")

    print("\n  — Fixed strategy:")
    emb_fixed = embed_chunks(fixed)
    print(f"  Shape: {emb_fixed.shape}")

    print("\n  — Structural strategy:")
    emb_struct = embed_chunks(structural)
    print(f"  Shape: {emb_struct.shape}")

    # 5. Сохранение
    print("\n[5/5] Сохранение индексов (FAISS + метаданные)...")
    save_index(fixed, emb_fixed, "fixed")
    save_index(structural, emb_struct, "structural")

    # Сохраняем сравнение
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(INDEX_DIR / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"  saved: comparison.json")

    # Демо поиска
    print("\n\n[DEMO] Поиск по индексу")
    demo_search()

    print("\n\nИндексация завершена.")
    print(f"  Индексы: {INDEX_DIR}")


def run_interactive():
    """Интерактивный поиск по существующему индексу."""
    print("=" * 60)
    print("  День 21 — Поиск по индексу документов")
    print("=" * 60)
    print("Введите запрос (пустая строка — выход).\n")

    while True:
        query = input("search> ").strip()
        if not query:
            break
        for strategy in ("fixed", "structural"):
            print(f"\n  [{strategy}]")
            results = search(query, strategy, top_k=5)
            for i, r in enumerate(results, 1):
                sec = f" § {r['section']}" if r.get("section") else ""
                print(f"    {i}. [{r['score']:.4f}] {r['source']}{sec}")
                preview = r["text"][:150].replace("\n", " ")
                print(f"       {preview}...")
        print()


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")

    if len(sys.argv) > 1 and sys.argv[1] == "--search":
        run_interactive()
    else:
        run_pipeline()

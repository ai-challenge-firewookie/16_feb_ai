"""
День 28. Локальная LLM + RAG

Цели:
1. Retrieval выполняется полностью локально на индексе из day21.
2. Генерация ответа выполняется через локальную модель Ollama.
3. При наличии облачного API-ключа сравниваются локальная и облачная модели.
4. Сохраняется отчёт по качеству, скорости и стабильности.

Примеры:
  python -m day28.rag_local --demo
  python -m day28.rag_local --ask "Как устроен планировщик в day18?"
  python -m day28.rag_local --demo --repeats 3
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from day21.indexer import (
    INDEX_DIR,
    chunk_structural,
    collect_documents,
    embed_chunks,
    save_index,
)
from day22.rag import EVAL_QUESTIONS, MODEL as CLOUD_MODEL, eval_answer_quality
from day23.rag import RetrievalConfig, collect_candidates, rerank_and_filter, rewrite_query

try:
    from anthropic import Anthropic
except Exception:  # pragma: no cover - облачный режим опционален
    Anthropic = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"
INDEX_STRATEGY = "structural"
LOCAL_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
DEFAULT_REPEATS = 3

LOCAL_SYSTEM = """\
Ты отвечаешь на вопросы о проекте AiChallenge только по предоставленному контексту.
Отвечай на русском языке, кратко и по делу.
Обязательно упоминай конкретные файлы или модули, если они есть в контексте.
Если данных в контексте недостаточно, прямо скажи об этом и не выдумывай детали."""

CLOUD_SYSTEM = LOCAL_SYSTEM


class Generator(Protocol):
    name: str

    def generate(self, system: str, prompt: str) -> "ModelResponse":
        ...


@dataclass
class ModelResponse:
    model: str
    answer: str
    elapsed_sec: float
    prompt_eval_count: int | None = None
    eval_count: int | None = None


@dataclass
class RagAnswer:
    question_id: int | None
    question: str
    generator: str
    answer: str
    elapsed_sec: float
    sources: list[str]
    context_chars: int
    rewritten_query: str
    candidates_before: int
    candidates_after: int
    quality_score: float | None = None
    quality_reason: str | None = None


@dataclass
class StabilityReport:
    generator: str
    runs: int
    exact_match_ratio: float
    unique_answers: int
    avg_elapsed_sec: float
    min_elapsed_sec: float
    max_elapsed_sec: float


@dataclass
class ComparisonSummary:
    local_model: str
    cloud_model: str | None
    questions_total: int
    local_avg_quality: float | None
    cloud_avg_quality: float | None
    local_avg_latency: float
    cloud_avg_latency: float | None
    local_avg_context_chars: float
    cloud_avg_context_chars: float | None


class OllamaGenerator:
    def __init__(self, model: str = LOCAL_MODEL, ollama_url: str = OLLAMA_URL) -> None:
        self.model = model
        self.ollama_url = ollama_url
        self.name = f"local:{model}"

    def check_server(self) -> None:
        try:
            with urllib.request.urlopen(f"{self.ollama_url}/api/tags", timeout=5) as response:
                if response.status != 200:
                    raise RuntimeError(f"Ollama API returned status {response.status}")
        except urllib.error.URLError as exc:
            raise SystemExit(
                "Локальный Ollama недоступен. Запустите `ollama serve` и повторите."
            ) from exc

    def generate(self, system: str, prompt: str) -> ModelResponse:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {"temperature": 0},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.ollama_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        start = time.time()
        with urllib.request.urlopen(request, timeout=300) as response:
            data = json.loads(response.read().decode("utf-8"))
        elapsed = round(time.time() - start, 2)
        return ModelResponse(
            model=self.model,
            answer=str(data["message"]["content"]).strip(),
            elapsed_sec=elapsed,
            prompt_eval_count=data.get("prompt_eval_count"),
            eval_count=data.get("eval_count"),
        )


class AnthropicGenerator:
    def __init__(self, client: Anthropic, model: str = CLOUD_MODEL) -> None:
        self.client = client
        self.model = model
        self.name = f"cloud:{model}"

    def generate(self, system: str, prompt: str) -> ModelResponse:
        start = time.time()
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        elapsed = round(time.time() - start, 2)
        return ModelResponse(
            model=self.model,
            answer=resp.content[0].text,
            elapsed_sec=elapsed,
            prompt_eval_count=resp.usage.input_tokens,
            eval_count=resp.usage.output_tokens,
        )


def ensure_local_index(strategy: str = INDEX_STRATEGY) -> None:
    prefix = f"index_{strategy}"
    index_path = INDEX_DIR / f"{prefix}.faiss"
    meta_path = INDEX_DIR / f"{prefix}_meta.json"
    if index_path.exists() and meta_path.exists():
        return

    print(f"[index] Индекс {prefix} не найден. Собираю заново...")
    try:
        docs = collect_documents(PROJECT_ROOT)
        chunks = chunk_structural(docs)
        embeddings = embed_chunks(chunks)
        save_index(chunks, embeddings, strategy=strategy, out_dir=INDEX_DIR)
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Не хватает локальных зависимостей для индексации. "
            "Установите пакеты из `requirements.txt`, затем повторите запуск."
        ) from exc


def build_context(question: str, cfg: RetrievalConfig) -> tuple[str, list[str], str, int, int]:
    rewritten = rewrite_query(question)
    candidates = collect_candidates(rewritten, top_k=cfg.top_k_before, strategy=INDEX_STRATEGY)
    filtered = rerank_and_filter(question, rewritten, candidates, cfg)

    parts: list[str] = []
    sources: list[str] = []
    for idx, item in enumerate(filtered, 1):
        section = f" / {item['section']}" if item.get("section") else ""
        parts.append(f"[{idx}] {item['source']}{section}\n{item['text']}")
        sources.append(item["source"])

    return (
        "\n\n---\n\n".join(parts),
        sources,
        rewritten,
        len(candidates),
        len(filtered),
    )


def format_rag_prompt(question: str, context: str) -> str:
    return (
        f"Контекст проекта:\n\n{context}\n\n"
        f"---\n\n"
        f"Вопрос: {question}\n\n"
        f"Сначала определи, хватает ли контекста. Затем дай короткий точный ответ."
    )


def ask_with_rag(
    generator: Generator,
    question: str,
    cfg: RetrievalConfig,
    question_id: int | None = None,
) -> RagAnswer:
    context, sources, rewritten, before_count, after_count = build_context(question, cfg)
    prompt = format_rag_prompt(question, context)
    result = generator.generate(LOCAL_SYSTEM if generator.name.startswith("local:") else CLOUD_SYSTEM, prompt)
    return RagAnswer(
        question_id=question_id,
        question=question,
        generator=generator.name,
        answer=result.answer,
        elapsed_sec=result.elapsed_sec,
        sources=sources,
        context_chars=len(context),
        rewritten_query=rewritten,
        candidates_before=before_count,
        candidates_after=after_count,
    )


def score_answers_locally(results: list[RagAnswer]) -> None:
    expected_by_question = {
        item["question"]: item["expected"]
        for item in EVAL_QUESTIONS
    }
    for result in results:
        expected = expected_by_question.get(result.question)
        if not expected:
            continue
        result.quality_score = round(simple_overlap_score(result.answer, expected), 2)
        result.quality_reason = "heuristic_overlap"


def simple_overlap_score(answer: str, expected: str) -> float:
    answer_tokens = set(_normalize_tokens(answer))
    expected_tokens = set(_normalize_tokens(expected))
    if not expected_tokens:
        return 0.0
    overlap = len(answer_tokens & expected_tokens) / len(expected_tokens)
    return min(5.0, max(1.0, round(1.0 + overlap * 4.0, 2)))


def _normalize_tokens(text: str) -> list[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return [token for token in cleaned.split() if len(token) > 2]


def score_answers_with_judge(
    judge_client: Anthropic,
    model: str,
    results: list[RagAnswer],
) -> None:
    expected_by_question = {
        item["question"]: item["expected"]
        for item in EVAL_QUESTIONS
    }
    for result in results:
        expected = expected_by_question.get(result.question)
        if not expected:
            continue
        score, reason = eval_answer_quality(
            judge_client,
            model,
            result.question,
            expected,
            result.answer,
        )
        result.quality_score = float(score)
        result.quality_reason = reason


def measure_stability(
    generator: Generator,
    question: str,
    cfg: RetrievalConfig,
    repeats: int,
) -> StabilityReport:
    answers: list[str] = []
    timings: list[float] = []
    for _ in range(repeats):
        result = ask_with_rag(generator, question, cfg)
        answers.append(result.answer.strip())
        timings.append(result.elapsed_sec)

    unique_answers = len(set(answers))
    exact_match_ratio = answers.count(answers[0]) / len(answers) if answers else 0.0
    return StabilityReport(
        generator=generator.name,
        runs=repeats,
        exact_match_ratio=round(exact_match_ratio, 2),
        unique_answers=unique_answers,
        avg_elapsed_sec=round(statistics.mean(timings), 2),
        min_elapsed_sec=round(min(timings), 2),
        max_elapsed_sec=round(max(timings), 2),
    )


def summarize_results(
    local_results: list[RagAnswer],
    cloud_results: list[RagAnswer] | None,
) -> ComparisonSummary:
    def avg(values: list[float | None]) -> float | None:
        existing = [value for value in values if value is not None]
        return round(statistics.mean(existing), 2) if existing else None

    return ComparisonSummary(
        local_model=local_results[0].generator if local_results else f"local:{LOCAL_MODEL}",
        cloud_model=cloud_results[0].generator if cloud_results else None,
        questions_total=len(local_results),
        local_avg_quality=avg([item.quality_score for item in local_results]),
        cloud_avg_quality=avg([item.quality_score for item in cloud_results]) if cloud_results else None,
        local_avg_latency=round(statistics.mean(item.elapsed_sec for item in local_results), 2),
        cloud_avg_latency=(
            round(statistics.mean(item.elapsed_sec for item in cloud_results), 2)
            if cloud_results else None
        ),
        local_avg_context_chars=round(statistics.mean(item.context_chars for item in local_results), 1),
        cloud_avg_context_chars=(
            round(statistics.mean(item.context_chars for item in cloud_results), 1)
            if cloud_results else None
        ),
    )


def print_answer(result: RagAnswer) -> None:
    print(f"[{result.generator}] {result.question}")
    print(f"Время: {result.elapsed_sec}s | ctx={result.context_chars} | chunks={result.candidates_after}")
    print(f"Источники: {', '.join(result.sources)}")
    print(result.answer)


def print_summary(
    summary: ComparisonSummary,
    local_stability: StabilityReport,
    cloud_stability: StabilityReport | None,
) -> None:
    print("\n" + "=" * 72)
    print("День 28. Локальный RAG")
    print("=" * 72)
    print(f"Локальная модель: {summary.local_model}")
    print(f"Облачная модель: {summary.cloud_model or 'не настроена'}")
    print(f"Вопросов: {summary.questions_total}")
    print()
    print("Качество:")
    print(f"- local: {summary.local_avg_quality if summary.local_avg_quality is not None else 'n/a'}")
    if summary.cloud_avg_quality is not None:
        print(f"- cloud: {summary.cloud_avg_quality}")
    print()
    print("Скорость:")
    print(f"- local avg: {summary.local_avg_latency}s")
    if summary.cloud_avg_latency is not None:
        print(f"- cloud avg: {summary.cloud_avg_latency}s")
    print()
    print("Стабильность:")
    print(f"- local exact-match ratio: {local_stability.exact_match_ratio} ({local_stability.unique_answers} unique answers)")
    if cloud_stability is not None:
        print(f"- cloud exact-match ratio: {cloud_stability.exact_match_ratio} ({cloud_stability.unique_answers} unique answers)")
    print("=" * 72)


def save_report(
    summary: ComparisonSummary,
    local_results: list[RagAnswer],
    cloud_results: list[RagAnswer] | None,
    local_stability: StabilityReport,
    cloud_stability: StabilityReport | None,
) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "rag_local_report.json"
    payload = {
        "summary": asdict(summary),
        "local_results": [asdict(item) for item in local_results],
        "cloud_results": [asdict(item) for item in cloud_results] if cloud_results else None,
        "stability": {
            "local": asdict(local_stability),
            "cloud": asdict(cloud_stability) if cloud_stability else None,
        },
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def run_demo(repeats: int = DEFAULT_REPEATS) -> Path:
    load_dotenv(PROJECT_ROOT / ".env")
    ensure_local_index()

    local_generator = OllamaGenerator()
    local_generator.check_server()

    cfg = RetrievalConfig()
    local_results = [
        ask_with_rag(local_generator, item["question"], cfg, question_id=item["id"])
        for item in EVAL_QUESTIONS
    ]

    judge_client = Anthropic() if Anthropic and os.getenv("ANTHROPIC_API_KEY") else None
    cloud_generator = AnthropicGenerator(judge_client) if judge_client else None
    cloud_results = (
        [ask_with_rag(cloud_generator, item["question"], cfg, question_id=item["id"]) for item in EVAL_QUESTIONS]
        if cloud_generator else None
    )

    if judge_client:
        score_answers_with_judge(judge_client, CLOUD_MODEL, local_results)
        if cloud_results:
            score_answers_with_judge(judge_client, CLOUD_MODEL, cloud_results)
    else:
        score_answers_locally(local_results)
        if cloud_results:
            score_answers_locally(cloud_results)

    stability_question = EVAL_QUESTIONS[0]["question"]
    local_stability = measure_stability(local_generator, stability_question, cfg, repeats)
    cloud_stability = (
        measure_stability(cloud_generator, stability_question, cfg, repeats)
        if cloud_generator else None
    )

    summary = summarize_results(local_results, cloud_results)
    print_summary(summary, local_stability, cloud_stability)
    out_path = save_report(summary, local_results, cloud_results, local_stability, cloud_stability)
    print(f"Отчёт сохранён: {out_path}")
    return out_path


def run_single_question(question: str) -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    ensure_local_index()

    generator = OllamaGenerator()
    generator.check_server()

    answer = ask_with_rag(generator, question, RetrievalConfig())
    print_answer(answer)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Day 28 local RAG")
    parser.add_argument("--demo", action="store_true", help="Прогнать полный набор контрольных вопросов")
    parser.add_argument("--ask", type=str, help="Задать один вопрос локальному RAG")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS, help="Сколько повторов для stability check")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.ask:
        return run_single_question(args.ask)
    if args.demo:
        run_demo(repeats=max(1, args.repeats))
        return 0
    print("Укажите `--demo` или `--ask`.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

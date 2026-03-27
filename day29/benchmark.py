"""
День 29. Оптимизация локальной LLM

Сценарий:
1. Создает оптимизированный вариант модели через Modelfile.
2. Прогоняет baseline и optimized на одинаковых задачах.
3. Сравнивает качество, скорость и приблизительное потребление памяти.
4. Сохраняет markdown-отчет и JSON с сырыми метриками.

Запуск:
  python day29/benchmark.py
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
BASE_MODEL = os.getenv("OLLAMA_BASE_MODEL", "qwen2.5:0.5b")
OPTIMIZED_MODEL = os.getenv("OLLAMA_OPTIMIZED_MODEL", "qwen2.5:0.5b-day29-study")
RESULTS_DIR = Path(__file__).resolve().parent / "results"
MODELFILE_PATH = Path(__file__).resolve().parent / "Modelfile"


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    prompt: str
    expectation: str


@dataclass(frozen=True)
class Variant:
    name: str
    model: str
    options: dict[str, Any]


@dataclass
class RunResult:
    case: str
    variant: str
    model: str
    response: str
    output_tokens: int
    prompt_tokens: int
    total_seconds: float
    tokens_per_second: float
    peak_rss_mb: float
    quality_score: int
    quality_note: str


CASES = [
    BenchmarkCase(
        name="План изучения",
        prompt=(
            "Составь короткий план изучения FastAPI для новичка на 1 неделю. "
            "Нужны ровно 3 шага."
        ),
        expectation="plan",
    ),
    BenchmarkCase(
        name="Структурированный JSON",
        prompt=(
            'Верни JSON с полями "topic", "difficulty", "hours". '
            "Тема: изучение FastAPI с нуля за выходные. Ответ только JSON."
        ),
        expectation="json",
    ),
    BenchmarkCase(
        name="Краткое резюме",
        prompt=(
            "Кратко резюмируй, зачем нужен Docker бэкенд-разработчику. "
            "Нужно 3 коротких пункта."
        ),
        expectation="bullets",
    ),
]

VARIANTS = [
    Variant(name="baseline", model=BASE_MODEL, options={}),
    Variant(
        name="optimized",
        model=OPTIMIZED_MODEL,
        options={
            "temperature": 0.2,
            "num_predict": 160,
            "num_ctx": 4096,
        },
    ),
]


def check_ollama_server() -> None:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as response:
            if response.status != 200:
                raise RuntimeError(f"Ollama API returned status {response.status}")
    except urllib.error.URLError as exc:
        raise SystemExit(
            "Ollama API недоступен. Запустите `ollama serve` и повторите попытку."
        ) from exc


def ensure_optimized_model() -> None:
    subprocess.run(
        [
            "ollama",
            "create",
            OPTIMIZED_MODEL,
            "-f",
            str(MODELFILE_PATH),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _sample_runner_rss(stop_event: threading.Event, samples: list[int]) -> None:
    while not stop_event.is_set():
        completed = subprocess.run(
            ["ps", "-axo", "rss=,command="],
            capture_output=True,
            text=True,
            check=True,
        )
        for line in completed.stdout.splitlines():
            if "ollama runner" not in line:
                continue
            rss_kb = line.strip().split(maxsplit=1)[0]
            if rss_kb.isdigit():
                samples.append(int(rss_kb))
        time.sleep(0.05)


def _request_generation(model: str, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def score_quality(expectation: str, response: str) -> tuple[int, str]:
    cleaned = response.strip()

    if expectation == "json":
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return 0, "Ответ не является валидным JSON."
        expected_keys = {"topic", "difficulty", "hours"}
        if set(data.keys()) != expected_keys:
            return 1, "JSON валиден, но ключи отличаются от ожидаемых."
        return 2, "JSON валиден и содержит нужные поля."

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]

    if expectation == "plan":
        numbered = [line for line in lines if line[:2] in {"1.", "2.", "3."}]
        if len(numbered) != 3:
            return 0, "План не удержал формат из 3 шагов."
        if len(cleaned.split()) > 90:
            return 1, "План структурный, но получился слишком длинным."
        return 2, "План короткий и держит формат 3 шагов."

    if expectation == "bullets":
        bullets = [line for line in lines if line.startswith(("-", "*"))]
        if len(bullets) != 3:
            return 0, "Резюме не удержало формат 3 пунктов."
        if len(cleaned.split()) > 70:
            return 1, "Формат выдержан, но ответ можно сделать короче."
        return 2, "Краткое резюме в нужном формате."

    return 0, "Неизвестное ожидание."


def run_case(case: BenchmarkCase, variant: Variant) -> RunResult:
    rss_samples: list[int] = []
    stop_event = threading.Event()
    sampler = threading.Thread(
        target=_sample_runner_rss,
        args=(stop_event, rss_samples),
        daemon=True,
    )
    sampler.start()

    started_at = time.perf_counter()
    payload = _request_generation(variant.model, case.prompt, variant.options)
    total_seconds = time.perf_counter() - started_at

    stop_event.set()
    sampler.join(timeout=1)

    response = str(payload["response"]).strip()
    output_tokens = int(payload.get("eval_count", 0) or 0)
    prompt_tokens = int(payload.get("prompt_eval_count", 0) or 0)
    quality_score, quality_note = score_quality(case.expectation, response)

    peak_rss_mb = max(rss_samples, default=0) / 1024
    tokens_per_second = output_tokens / total_seconds if total_seconds > 0 else 0.0

    return RunResult(
        case=case.name,
        variant=variant.name,
        model=variant.model,
        response=response,
        output_tokens=output_tokens,
        prompt_tokens=prompt_tokens,
        total_seconds=round(total_seconds, 3),
        tokens_per_second=round(tokens_per_second, 2),
        peak_rss_mb=round(peak_rss_mb, 1),
        quality_score=quality_score,
        quality_note=quality_note,
    )


def summarize_results(results: list[RunResult]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for variant in VARIANTS:
        runs = [item for item in results if item.variant == variant.name]
        summary[variant.name] = {
            "avg_quality": round(sum(item.quality_score for item in runs) / len(runs), 2),
            "avg_seconds": round(sum(item.total_seconds for item in runs) / len(runs), 3),
            "avg_tokens_per_second": round(
                sum(item.tokens_per_second for item in runs) / len(runs), 2
            ),
            "avg_peak_rss_mb": round(sum(item.peak_rss_mb for item in runs) / len(runs), 1),
        }
    return summary


def build_report(results: list[RunResult], summary: dict[str, dict[str, float]]) -> str:
    baseline = summary["baseline"]
    optimized = summary["optimized"]

    lines = [
        "# День 29. Оптимизация локальной LLM",
        "",
        f"- Базовая модель: `{BASE_MODEL}`",
        f"- Оптимизированная модель: `{OPTIMIZED_MODEL}`",
        "- Кейс оптимизации: короткие и структурированные ответы на русском для учебных/технических задач",
        "- Квантование базовой модели: `Q4_K_M` (по `ollama show`)",
        "- Статус квантования: в локальной среде доступен только один вариант квантования, поэтому отдельное A/B-сравнение по квантованию не проводилось",
        "",
        "## Что менялось",
        "",
        "- `temperature`: снижена до `0.2` для более стабильного формата",
        "- `num_predict`: ограничен до `160`, чтобы сократить лишний текст и ускорить ответ",
        "- `num_ctx`: уменьшен до `4096`, чтобы не держать избыточное окно контекста для коротких задач",
        "- Prompt-шаблон: добавлен системный prompt с акцентом на русский язык, краткость и строгое следование формату",
        "",
        "## Итог по средним метрикам",
        "",
        "| Вариант | Качество | Время, c | Токенов/с | Peak RSS, MB |",
        "| --- | ---: | ---: | ---: | ---: |",
        (
            f"| baseline | {baseline['avg_quality']} | {baseline['avg_seconds']} | "
            f"{baseline['avg_tokens_per_second']} | {baseline['avg_peak_rss_mb']} |"
        ),
        (
            f"| optimized | {optimized['avg_quality']} | {optimized['avg_seconds']} | "
            f"{optimized['avg_tokens_per_second']} | {optimized['avg_peak_rss_mb']} |"
        ),
        "",
        "## Детали по кейсам",
        "",
    ]

    for case in CASES:
        lines.append(f"### {case.name}")
        lines.append("")
        lines.append(f"**Запрос:** {case.prompt}")
        lines.append("")
        for variant_name in ("baseline", "optimized"):
            result = next(
                item for item in results if item.case == case.name and item.variant == variant_name
            )
            lines.append(f"**{variant_name}**")
            lines.append("")
            lines.append(f"- Ответ: `{result.response}`")
            lines.append(f"- Оценка качества: {result.quality_score}/2")
            lines.append(f"- Комментарий: {result.quality_note}")
            lines.append(f"- Время: {result.total_seconds} c")
            lines.append(f"- Скорость: {result.tokens_per_second} токенов/с")
            lines.append(f"- Peak RSS: {result.peak_rss_mb} MB")
            lines.append("")

    lines.extend(
        [
            "## Вывод",
            "",
            (
                "Для коротких прикладных запросов оптимизация обычно улучшает "
                "соблюдение формата и убирает лишний текст. Цена компромисса: "
                "меньшее окно контекста и более консервативные ответы."
            ),
            (
                "Если задача сместится в сторону длинных диалогов или свободного "
                "брейншторминга, `num_ctx` и `temperature` стоит вернуть выше."
            ),
        ]
    )

    return "\n".join(lines) + "\n"


def save_artifacts(results: list[RunResult], summary: dict[str, dict[str, float]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_payload = {
        "ollama_url": OLLAMA_URL,
        "base_model": BASE_MODEL,
        "optimized_model": OPTIMIZED_MODEL,
        "results": [asdict(item) for item in results],
        "summary": summary,
    }
    (RESULTS_DIR / "benchmark_results.json").write_text(
        json.dumps(raw_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (RESULTS_DIR / "report.md").write_text(
        build_report(results, summary),
        encoding="utf-8",
    )


def main() -> int:
    check_ollama_server()
    ensure_optimized_model()

    results: list[RunResult] = []
    for case in CASES:
        for variant in VARIANTS:
            results.append(run_case(case, variant))

    summary = summarize_results(results)
    save_artifacts(results, summary)

    print("Benchmark completed.")
    print(f"Report: {RESULTS_DIR / 'report.md'}")
    print(f"Raw JSON: {RESULTS_DIR / 'benchmark_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

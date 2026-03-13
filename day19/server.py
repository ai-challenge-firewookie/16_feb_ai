"""День 19. MCP-сервер: композиция инструментов (pipeline).

Инструменты образуют пайплайн:
  Поиск → Обработка → Сохранение

Конкретно:
  - search_articles    — поиск статей по запросу (mock knowledge base)
  - fetch_content      — получить полный текст статьи по ID
  - summarize_text     — сжать / извлечь ключевое из текста
  - translate_text     — перевести текст (ru↔en)
  - analyze_sentiment  — определить тональность текста
  - save_to_file       — сохранить результат в файл (JSON/TXT/MD)
  - list_saved_files   — показать ранее сохранённые файлы
  - read_saved_file    — прочитать содержимое сохранённого файла
  - run_pipeline       — выполнить готовый пайплайн за один вызов

Данные: mock-база статей + файловое хранилище.

Запуск:
  python server.py             — stdio-транспорт
  python server.py --test      — self-test
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

if os.environ.get("MCP_QUIET"):
    logging.getLogger("mcp").setLevel(logging.ERROR)

import json as _json
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("day19-pipeline")

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def _j(obj: object) -> str:
    return _json.dumps(obj, ensure_ascii=False, indent=2, default=str)


# ══════════════════════════════════════════════════════════════
# Mock knowledge base
# ══════════════════════════════════════════════════════════════

ARTICLES: dict[str, dict] = {
    "art-1": {
        "id": "art-1",
        "title": "Введение в машинное обучение",
        "tags": ["ml", "ai", "python", "обучение"],
        "author": "Алексей Петров",
        "date": "2025-11-15",
        "summary": "Обзор основных концепций ML: supervised, unsupervised, reinforcement learning.",
        "content": (
            "Машинное обучение (ML) — это подраздел искусственного интеллекта, "
            "который позволяет системам автоматически обучаться на данных без явного программирования. "
            "Существует три основных парадигмы: обучение с учителем (supervised), "
            "обучение без учителя (unsupervised) и обучение с подкреплением (reinforcement learning). "
            "В supervised learning модель обучается на размеченных данных — парах (вход, ожидаемый выход). "
            "Классические алгоритмы: линейная регрессия, деревья решений, SVM, нейронные сети. "
            "В unsupervised learning данные не размечены, задача — найти скрытые структуры: кластеризация, "
            "снижение размерности (PCA, t-SNE). Reinforcement learning работает через награды: "
            "агент взаимодействует со средой и максимизирует суммарную награду. "
            "Примеры применения ML: рекомендательные системы, распознавание изображений, NLP, "
            "автономные автомобили, медицинская диагностика."
        ),
    },
    "art-2": {
        "id": "art-2",
        "title": "Микросервисная архитектура: плюсы и минусы",
        "tags": ["architecture", "microservices", "devops", "backend"],
        "author": "Мария Сидорова",
        "date": "2025-12-01",
        "summary": "Анализ преимуществ и недостатков перехода на микросервисы.",
        "content": (
            "Микросервисная архитектура — подход, при котором приложение строится как набор "
            "небольших, независимо развёртываемых сервисов. Каждый сервис отвечает за одну бизнес-функцию "
            "и общается с другими через API (REST, gRPC, сообщения). "
            "Преимущества: независимое масштабирование, технологическая гибкость (polyglot), "
            "устойчивость к отказам (fault isolation), быстрые деплои. "
            "Недостатки: сетевая сложность, distributed tracing, eventual consistency, "
            "операционная нагрузка (мониторинг, логирование, оркестрация). "
            "Рекомендации: начинайте с монолита, выделяйте микросервисы по мере роста, "
            "используйте service mesh (Istio, Linkerd), внедряйте CI/CD и контейнеризацию (Docker, K8s). "
            "Ключевые паттерны: API Gateway, Circuit Breaker, Saga, Event Sourcing, CQRS."
        ),
    },
    "art-3": {
        "id": "art-3",
        "title": "Python: асинхронное программирование с asyncio",
        "tags": ["python", "asyncio", "concurrency", "performance"],
        "author": "Дмитрий Козлов",
        "date": "2026-01-10",
        "summary": "Практическое руководство по asyncio в Python.",
        "content": (
            "Модуль asyncio в Python реализует асинхронное программирование на основе "
            "event loop и корутин. Ключевые концепции: async/await, Task, Future, Event Loop. "
            "Корутина — функция, объявленная через async def, которая может приостанавливать "
            "выполнение через await. asyncio.gather() позволяет запускать корутины параллельно. "
            "Для I/O-bound задач asyncio даёт значительный прирост производительности без потоков. "
            "Примеры использования: HTTP-клиенты (aiohttp), веб-серверы (FastAPI, aiohttp), "
            "работа с БД (asyncpg, aiosqlite), очереди сообщений. "
            "Подводные камни: нельзя использовать блокирующие вызовы внутри корутин, "
            "asyncio.run() создаёт новый event loop, нужна осторожность с shared state."
        ),
    },
    "art-4": {
        "id": "art-4",
        "title": "Безопасность веб-приложений: OWASP Top 10",
        "tags": ["security", "web", "owasp", "backend"],
        "author": "Елена Волкова",
        "date": "2026-02-05",
        "summary": "Разбор главных уязвимостей веб-приложений по OWASP.",
        "content": (
            "OWASP Top 10 — список самых критичных уязвимостей веб-приложений. "
            "1. Broken Access Control — обход контроля доступа. "
            "2. Cryptographic Failures — слабое шифрование, утечка ключей. "
            "3. Injection (SQL, XSS, Command) — внедрение вредоносного кода. "
            "4. Insecure Design — архитектурные ошибки безопасности. "
            "5. Security Misconfiguration — неправильные настройки серверов и фреймворков. "
            "6. Vulnerable Components — использование библиотек с известными уязвимостями. "
            "7. Authentication Failures — слабая аутентификация, brute force. "
            "8. Data Integrity Failures — ненадёжная проверка целостности данных. "
            "9. Logging Failures — недостаточное логирование и мониторинг. "
            "10. SSRF — Server-Side Request Forgery. "
            "Меры защиты: input validation, parameterized queries, CSP, HTTPS, "
            "регулярное обновление зависимостей, pen-testing."
        ),
    },
    "art-5": {
        "id": "art-5",
        "title": "Large Language Models: как работают и куда движутся",
        "tags": ["ai", "llm", "nlp", "transformers"],
        "author": "Алексей Петров",
        "date": "2026-03-01",
        "summary": "Обзор архитектуры и трендов в области больших языковых моделей.",
        "content": (
            "Large Language Models (LLM) — нейросети на основе архитектуры Transformer, "
            "обученные на огромных текстовых корпусах. Ключевой механизм — self-attention, "
            "позволяющий моделировать зависимости между словами на любом расстоянии. "
            "Этапы: pre-training на неразмеченных данных (next token prediction), "
            "fine-tuning на задачах (instruction tuning), alignment через RLHF. "
            "Тренды: увеличение контекстного окна (до миллионов токенов), "
            "мультимодальность (текст + изображения + аудио), "
            "tool use (модели вызывают внешние API), "
            "агентные системы (модели планируют и выполняют многошаговые задачи). "
            "Вызовы: галлюцинации, стоимость обучения, энергопотребление, "
            "безопасность и этика, проблема авторских прав на обучающие данные."
        ),
    },
}


# ══════════════════════════════════════════════════════════════
# Шаг 1: Поиск и получение данных
# ══════════════════════════════════════════════════════════════

@mcp.tool()
def search_articles(query: str, limit: int = 5) -> str:
    """Поиск статей в базе знаний по ключевым словам.

    Возвращает список найденных статей (id, title, summary, tags).
    Это первый шаг пайплайна: получение данных.

    Args:
        query: Поисковый запрос (ключевые слова)
        limit: Максимум результатов (по умолчанию 5)
    """
    query_lower = query.lower()
    query_words = query_lower.split()

    scored: list[tuple[int, dict]] = []
    for art in ARTICLES.values():
        score = 0
        searchable = (
            art["title"].lower() + " " +
            art["summary"].lower() + " " +
            " ".join(art["tags"])
        )
        for word in query_words:
            if word in searchable:
                score += 1
            # бонус за совпадение в тегах
            if word in art["tags"]:
                score += 2
        if score > 0:
            scored.append((score, art))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = [
        {
            "id": art["id"],
            "title": art["title"],
            "author": art["author"],
            "date": art["date"],
            "tags": art["tags"],
            "summary": art["summary"],
            "relevance_score": score,
        }
        for score, art in scored[:limit]
    ]
    return _j({"query": query, "results": results, "total_found": len(results)})


@mcp.tool()
def fetch_content(article_id: str) -> str:
    """Получить полный текст статьи по её ID.

    Используется после search_articles для получения полного контента.

    Args:
        article_id: ID статьи (например 'art-1')
    """
    art = ARTICLES.get(article_id)
    if not art:
        return _j({"error": f"Статья '{article_id}' не найдена"})
    return _j(art)


# ══════════════════════════════════════════════════════════════
# Шаг 2: Обработка данных
# ══════════════════════════════════════════════════════════════

@mcp.tool()
def summarize_text(text: str, max_sentences: int = 3) -> str:
    """Извлекает ключевые предложения из текста (extractive summarization).

    Второй шаг пайплайна: обработка данных.

    Args:
        text: Текст для суммаризации
        max_sentences: Максимум предложений в итоге (по умолчанию 3)
    """
    # Простая extractive-суммаризация: берём предложения с наибольшим кол-вом ключевых слов
    sentences = [s.strip() for s in text.replace(".", ".\n").split("\n") if s.strip()]
    if len(sentences) <= max_sentences:
        return _j({
            "summary": " ".join(sentences),
            "sentence_count": len(sentences),
            "method": "full_text (текст короткий)",
        })

    # Считаем слова по всему тексту для определения "ключевости"
    all_words = text.lower().split()
    word_freq: dict[str, int] = {}
    stop_words = {"—", "и", "в", "на", "с", "по", "для", "из", "к", "от", "не", "что", "это", "как", "а", "но", "или"}
    for w in all_words:
        w = w.strip(".,;:!?()\"'")
        if len(w) > 2 and w not in stop_words:
            word_freq[w] = word_freq.get(w, 0) + 1

    # Оценка предложений
    scored_sentences: list[tuple[float, int, str]] = []
    for idx, sent in enumerate(sentences):
        words = sent.lower().split()
        score = sum(word_freq.get(w.strip(".,;:!?()\"'"), 0) for w in words)
        # Нормализуем по длине, бонус за позицию (первые предложения важнее)
        if words:
            score = score / len(words) + (1.0 / (idx + 1))
        scored_sentences.append((score, idx, sent))

    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    # Берём top-N, сортируем по оригинальному порядку
    selected = sorted(scored_sentences[:max_sentences], key=lambda x: x[1])
    summary = " ".join(s[2] for s in selected)

    return _j({
        "summary": summary,
        "sentence_count": len(selected),
        "original_sentences": len(sentences),
        "method": "extractive (по частотности слов)",
    })


@mcp.tool()
def translate_text(text: str, direction: str = "ru-en") -> str:
    """Имитация перевода текста (ru→en или en→ru).

    В реальном проекте здесь будет API перевода. Сейчас — простая демо-замена.

    Args:
        text: Текст для перевода
        direction: Направление: 'ru-en' или 'en-ru'
    """
    if direction not in ("ru-en", "en-ru"):
        return _j({"error": f"Неверное направление: '{direction}'. Допустимые: ru-en, en-ru"})

    # Mock-перевод: словарь частых терминов
    ru_en = {
        "машинное обучение": "machine learning",
        "искусственный интеллект": "artificial intelligence",
        "нейронные сети": "neural networks",
        "обучение с учителем": "supervised learning",
        "обучение без учителя": "unsupervised learning",
        "обучение с подкреплением": "reinforcement learning",
        "микросервисная архитектура": "microservice architecture",
        "безопасность": "security",
        "уязвимость": "vulnerability",
        "данные": "data",
        "модель": "model",
        "сервис": "service",
        "приложение": "application",
        "производительность": "performance",
    }
    en_ru = {v: k for k, v in ru_en.items()}

    mapping = ru_en if direction == "ru-en" else en_ru
    translated = text
    for src, dst in mapping.items():
        translated = translated.replace(src, dst)

    return _j({
        "original": text[:200],
        "translated": translated[:500],
        "direction": direction,
        "note": "Mock-перевод (замена терминов). В production используйте API перевода.",
    })


@mcp.tool()
def analyze_sentiment(text: str) -> str:
    """Анализ тональности текста (позитивная / нейтральная / негативная).

    Простая эвристика на основе ключевых слов.

    Args:
        text: Текст для анализа тональности
    """
    text_lower = text.lower()

    positive_words = [
        "преимущ", "плюс", "позволяет", "улучш", "прирост", "быстр",
        "эффективн", "успешн", "надёжн", "гибк", "удобн", "мощн",
        "отличн", "важн", "ключев", "полезн", "рекомендац",
    ]
    negative_words = [
        "недостат", "минус", "проблем", "ошибк", "уязвимост", "слож",
        "опасн", "утечк", "вредонос", "слаб", "критич", "наруш",
        "отказ", "нагрузк", "галлюцинац",
    ]

    pos_count = sum(1 for w in positive_words if w in text_lower)
    neg_count = sum(1 for w in negative_words if w in text_lower)
    total = pos_count + neg_count

    if total == 0:
        sentiment = "neutral"
        confidence = 0.5
    elif pos_count > neg_count * 1.5:
        sentiment = "positive"
        confidence = round(pos_count / total, 2)
    elif neg_count > pos_count * 1.5:
        sentiment = "negative"
        confidence = round(neg_count / total, 2)
    else:
        sentiment = "mixed"
        confidence = round(0.5 + abs(pos_count - neg_count) / (total * 2), 2)

    return _j({
        "sentiment": sentiment,
        "confidence": min(confidence, 1.0),
        "positive_signals": pos_count,
        "negative_signals": neg_count,
        "text_length": len(text),
    })


# ══════════════════════════════════════════════════════════════
# Шаг 3: Сохранение результата
# ══════════════════════════════════════════════════════════════

@mcp.tool()
def save_to_file(content: str, filename: str, format: str = "md") -> str:
    """Сохраняет текст / данные в файл.

    Третий шаг пайплайна: сохранение результата.

    Args:
        content: Содержимое для сохранения
        filename: Имя файла (без расширения)
        format: Формат: 'json', 'txt', 'md' (по умолчанию 'md')
    """
    if format not in ("json", "txt", "md"):
        return _j({"error": f"Неверный формат: '{format}'. Допустимые: json, txt, md"})

    # Санитизация имени файла
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in filename)
    if not safe_name:
        safe_name = f"output_{uuid.uuid4().hex[:6]}"

    filepath = OUTPUT_DIR / f"{safe_name}.{format}"

    # Если json — пытаемся отформатировать
    if format == "json":
        try:
            parsed = _json.loads(content)
            content = _json.dumps(parsed, ensure_ascii=False, indent=2)
        except _json.JSONDecodeError:
            pass

    filepath.write_text(content, encoding="utf-8")

    return _j({
        "saved": True,
        "path": str(filepath),
        "filename": filepath.name,
        "size_bytes": filepath.stat().st_size,
        "format": format,
        "saved_at": datetime.now().isoformat(),
    })


@mcp.tool()
def list_saved_files() -> str:
    """Показывает список ранее сохранённых файлов в директории output."""
    files = []
    for f in sorted(OUTPUT_DIR.iterdir()):
        if f.is_file() and not f.name.startswith("."):
            files.append({
                "filename": f.name,
                "size_bytes": f.stat().st_size,
                "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
            })
    return _j({"directory": str(OUTPUT_DIR), "files": files, "total": len(files)})


@mcp.tool()
def read_saved_file(filename: str) -> str:
    """Прочитать содержимое ранее сохранённого файла.

    Args:
        filename: Имя файла (с расширением, например 'report.md')
    """
    filepath = OUTPUT_DIR / filename
    if not filepath.exists():
        return _j({"error": f"Файл '{filename}' не найден"})
    if not filepath.is_file():
        return _j({"error": f"'{filename}' не является файлом"})

    content = filepath.read_text(encoding="utf-8")
    return _j({
        "filename": filename,
        "content": content[:3000],
        "size_bytes": filepath.stat().st_size,
        "truncated": len(content) > 3000,
    })


# ══════════════════════════════════════════════════════════════
# Мета-инструмент: готовый пайплайн
# ══════════════════════════════════════════════════════════════

@mcp.tool()
def run_pipeline(query: str, pipeline: str = "search-summarize-save", output_name: str = "pipeline_result") -> str:
    """Выполняет готовый пайплайн за один вызов.

    Доступные пайплайны:
    - 'search-summarize-save': поиск → суммаризация → сохранение в файл
    - 'search-analyze-save': поиск → анализ тональности → сохранение
    - 'search-translate-save': поиск → перевод (ru→en) → сохранение
    - 'search-full-save': поиск → суммаризация + тональность + перевод → сохранение

    Args:
        query: Поисковый запрос
        pipeline: Название пайплайна
        output_name: Имя выходного файла (без расширения)
    """
    valid_pipelines = [
        "search-summarize-save",
        "search-analyze-save",
        "search-translate-save",
        "search-full-save",
    ]
    if pipeline not in valid_pipelines:
        return _j({"error": f"Неизвестный пайплайн: '{pipeline}'. Допустимые: {valid_pipelines}"})

    steps_log: list[dict] = []

    # Шаг 1: Поиск
    search_result = _json.loads(search_articles(query, limit=3))
    steps_log.append({"step": "search", "found": search_result["total_found"]})

    if not search_result["results"]:
        return _j({"error": "Ничего не найдено", "steps": steps_log})

    # Получаем контент первой (наиболее релевантной) статьи
    top_article_id = search_result["results"][0]["id"]
    article = _json.loads(fetch_content(top_article_id))
    content = article["content"]
    steps_log.append({"step": "fetch", "article": article["title"]})

    # Шаг 2: Обработка
    results: dict = {
        "source_article": {
            "id": article["id"],
            "title": article["title"],
            "author": article["author"],
        },
        "pipeline": pipeline,
    }

    if pipeline in ("search-summarize-save", "search-full-save"):
        summary_result = _json.loads(summarize_text(content))
        results["summary"] = summary_result["summary"]
        steps_log.append({"step": "summarize", "sentences": summary_result["sentence_count"]})

    if pipeline in ("search-analyze-save", "search-full-save"):
        sentiment_result = _json.loads(analyze_sentiment(content))
        results["sentiment"] = {
            "label": sentiment_result["sentiment"],
            "confidence": sentiment_result["confidence"],
        }
        steps_log.append({"step": "analyze", "sentiment": sentiment_result["sentiment"]})

    if pipeline in ("search-translate-save", "search-full-save"):
        text_to_translate = results.get("summary", content[:300])
        translate_result = _json.loads(translate_text(text_to_translate, "ru-en"))
        results["translation"] = translate_result["translated"]
        steps_log.append({"step": "translate", "direction": "ru-en"})

    # Шаг 3: Сохранение
    save_content = _json.dumps(results, ensure_ascii=False, indent=2)
    save_result = _json.loads(save_to_file(save_content, output_name, "json"))
    steps_log.append({"step": "save", "file": save_result["filename"]})

    return _j({
        "pipeline": pipeline,
        "steps": steps_log,
        "output_file": save_result["filename"],
        "output_path": save_result["path"],
        "results_preview": {k: (v[:150] + "..." if isinstance(v, str) and len(v) > 150 else v)
                           for k, v in results.items()},
    })


# ══════════════════════════════════════════════════════════════
# Self-test
# ══════════════════════════════════════════════════════════════

def self_test() -> None:
    import json
    import tempfile
    global OUTPUT_DIR

    OUTPUT_DIR = Path(tempfile.mkdtemp())

    print("=" * 62)
    print("  День 19. MCP-сервер пайплайна — self-test")
    print("=" * 62)
    print()

    tests = 0
    passed = 0

    # search_articles
    tests += 1
    result = json.loads(search_articles("machine learning python"))
    ok = result["total_found"] >= 1
    print(f"  {'OK' if ok else 'FAIL'} search_articles('machine learning python') -> {result['total_found']} результатов")
    if ok: passed += 1

    # search_articles — пустой результат
    tests += 1
    result = json.loads(search_articles("zzzzzznotfound"))
    ok = result["total_found"] == 0
    print(f"  {'OK' if ok else 'FAIL'} search_articles('zzzznotfound') -> 0 результатов")
    if ok: passed += 1

    # fetch_content
    tests += 1
    result = json.loads(fetch_content("art-1"))
    ok = "content" in result and len(result["content"]) > 100
    print(f"  {'OK' if ok else 'FAIL'} fetch_content('art-1') -> {len(result.get('content', ''))} символов")
    if ok: passed += 1

    # fetch_content — ошибка
    tests += 1
    result = json.loads(fetch_content("art-999"))
    ok = "error" in result
    print(f"  {'OK' if ok else 'FAIL'} fetch_content('art-999') -> ошибка")
    if ok: passed += 1

    # summarize_text
    tests += 1
    long_text = ARTICLES["art-1"]["content"]
    result = json.loads(summarize_text(long_text, 2))
    ok = result["sentence_count"] == 2
    print(f"  {'OK' if ok else 'FAIL'} summarize_text(art-1, 2) -> {result['sentence_count']} предложений")
    if ok: passed += 1

    # translate_text
    tests += 1
    result = json.loads(translate_text("машинное обучение и нейронные сети", "ru-en"))
    ok = "machine learning" in result["translated"]
    print(f"  {'OK' if ok else 'FAIL'} translate_text(ru-en) -> '{result['translated'][:50]}...'")
    if ok: passed += 1

    # translate — ошибка
    tests += 1
    result = json.loads(translate_text("test", "fr-en"))
    ok = "error" in result
    print(f"  {'OK' if ok else 'FAIL'} translate_text(fr-en) -> ошибка")
    if ok: passed += 1

    # analyze_sentiment
    tests += 1
    result = json.loads(analyze_sentiment("Это отличный инструмент, который позволяет улучшить производительность"))
    ok = result["sentiment"] in ("positive", "mixed")
    print(f"  {'OK' if ok else 'FAIL'} analyze_sentiment(positive) -> {result['sentiment']} ({result['confidence']})")
    if ok: passed += 1

    # analyze_sentiment — negative
    tests += 1
    result = json.loads(analyze_sentiment("Уязвимости, проблемы, утечки данных, ошибки критичные"))
    ok = result["sentiment"] in ("negative", "mixed")
    print(f"  {'OK' if ok else 'FAIL'} analyze_sentiment(negative) -> {result['sentiment']} ({result['confidence']})")
    if ok: passed += 1

    # save_to_file
    tests += 1
    result = json.loads(save_to_file("# Test Report\nHello", "test_report", "md"))
    ok = result["saved"] is True
    print(f"  {'OK' if ok else 'FAIL'} save_to_file('test_report.md') -> {result.get('size_bytes', 0)} bytes")
    if ok: passed += 1

    # save_to_file — json format
    tests += 1
    result = json.loads(save_to_file('{"key": "value"}', "test_json", "json"))
    ok = result["saved"] is True and result["format"] == "json"
    print(f"  {'OK' if ok else 'FAIL'} save_to_file('test_json.json') -> saved")
    if ok: passed += 1

    # list_saved_files
    tests += 1
    result = json.loads(list_saved_files())
    ok = result["total"] >= 2
    print(f"  {'OK' if ok else 'FAIL'} list_saved_files() -> {result['total']} файлов")
    if ok: passed += 1

    # read_saved_file
    tests += 1
    result = json.loads(read_saved_file("test_report.md"))
    ok = "# Test Report" in result.get("content", "")
    print(f"  {'OK' if ok else 'FAIL'} read_saved_file('test_report.md') -> content ok")
    if ok: passed += 1

    # read_saved_file — ошибка
    tests += 1
    result = json.loads(read_saved_file("nonexistent.txt"))
    ok = "error" in result
    print(f"  {'OK' if ok else 'FAIL'} read_saved_file('nonexistent.txt') -> ошибка")
    if ok: passed += 1

    # run_pipeline — search-summarize-save
    tests += 1
    result = json.loads(run_pipeline("ml python", "search-summarize-save", "test_pipeline_summary"))
    ok = len(result.get("steps", [])) == 4
    steps_str = " → ".join(s["step"] for s in result.get("steps", []))
    print(f"  {'OK' if ok else 'FAIL'} run_pipeline(search-summarize-save) -> [{steps_str}]")
    if ok: passed += 1

    # run_pipeline — search-full-save
    tests += 1
    result = json.loads(run_pipeline("ai llm", "search-full-save", "test_pipeline_full"))
    ok = len(result.get("steps", [])) == 6  # search, fetch, summarize, analyze, translate, save
    steps_str = " → ".join(s["step"] for s in result.get("steps", []))
    print(f"  {'OK' if ok else 'FAIL'} run_pipeline(search-full-save) -> [{steps_str}]")
    if ok: passed += 1

    # run_pipeline — ошибка
    tests += 1
    result = json.loads(run_pipeline("test", "bad-pipeline"))
    ok = "error" in result
    print(f"  {'OK' if ok else 'FAIL'} run_pipeline('bad-pipeline') -> ошибка")
    if ok: passed += 1

    print(f"\n  Результат: {passed}/{tests} тестов пройдено")

    # Cleanup
    import shutil
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


# ══════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if "--test" in sys.argv:
        self_test()
    else:
        mcp.run(transport="stdio")

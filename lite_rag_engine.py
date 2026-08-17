"""On-demand PDF extraction and lightweight page retrieval for LiciGob."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import threading
import time
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Iterable

import psycopg2
from psycopg2.extras import RealDictCursor

from document_fetcher import DocumentDownloadError, download_document, normalize_document_url

try:
    import fitz
except ImportError:  # Allows lightweight unit tests without the PDF runtime.
    fitz = None


MAX_DOCUMENTS = int(os.getenv("LITE_MAX_DOCUMENTS", "2"))
MAX_FILE_BYTES = int(os.getenv("LITE_MAX_FILE_BYTES", str(50 * 1024 * 1024)))
MAX_TOTAL_PAGES = int(os.getenv("LITE_MAX_TOTAL_PAGES", "120"))
MAX_TOTAL_CHARS = int(os.getenv("LITE_MAX_TOTAL_CHARS", "600000"))
MAX_CONTEXT_CHARS = int(os.getenv("LITE_MAX_CONTEXT_CHARS", "18000"))
CACHE_TTL_DAYS = int(os.getenv("LITE_CACHE_TTL_DAYS", "14"))
MEMORY_CACHE_SECONDS = int(os.getenv("LITE_MEMORY_CACHE_SECONDS", "1800"))
MEMORY_CACHE_MAX_ITEMS = int(os.getenv("LITE_MEMORY_CACHE_MAX_ITEMS", "20"))
CACHE_PREFIX = "LITE_RAG_V1\n"

logger = logging.getLogger("licigob-ai-lite.rag")

_MEMORY_CACHE: dict[str, tuple[float, dict, Corpus]] = {}
_MEMORY_CACHE_LOCK = threading.Lock()

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "sslmode": os.getenv("DB_SSLMODE", "require"),
}

STOP_WORDS = {
    "a", "al", "algo", "como", "con", "cual", "cuando", "de", "del", "donde",
    "el", "ella", "en", "es", "esta", "este", "esto", "hay", "la", "las", "lo",
    "los", "me", "mi", "para", "por", "que", "se", "sin", "sobre", "su", "sus",
    "un", "una", "y", "ya",
}


class LiteRagError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class PageText:
    document: str
    page: int
    text: str


@dataclass
class Corpus:
    tender_id: str
    source_digest: str
    pages: list[PageText]
    document_count: int
    total_pages: int
    total_chars: int
    scan_suspected: bool = False
    cache_hit: bool = False


def _memory_cache_get(tender_id: str) -> tuple[dict, Corpus] | None:
    now = time.monotonic()
    with _MEMORY_CACHE_LOCK:
        cached = _MEMORY_CACHE.get(str(tender_id))
        if not cached:
            return None
        expires_at, tender, corpus = cached
        if expires_at <= now:
            _MEMORY_CACHE.pop(str(tender_id), None)
            return None
        corpus.cache_hit = True
        return tender, corpus


def _memory_cache_set(tender_id: str, tender: dict, corpus: Corpus) -> None:
    with _MEMORY_CACHE_LOCK:
        if len(_MEMORY_CACHE) >= MEMORY_CACHE_MAX_ITEMS:
            oldest_key = min(_MEMORY_CACHE, key=lambda key: _MEMORY_CACHE[key][0])
            _MEMORY_CACHE.pop(oldest_key, None)
        _MEMORY_CACHE[str(tender_id)] = (
            time.monotonic() + MEMORY_CACHE_SECONDS,
            tender,
            corpus,
        )


def _get_conn():
    return psycopg2.connect(**DB_CONFIG)


def _json_default(value):
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime,)):
        return value.isoformat()
    return str(value)


def get_tender_bundle(tender_id: str) -> tuple[dict, list[dict]]:
    """Load trusted tender metadata and PDF URLs directly from PostgreSQL."""
    conn = _get_conn()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """
            SELECT t.id, t.title, t.description, t.procurement_method_details,
                   t.main_procurement_category, t.value_amount, t.currency,
                   t.date_published, COALESCE(p.name, p.legal_name) AS buyer_name
            FROM tenders t
            LEFT JOIN parties p ON p.id = t.buyer_id
            WHERE t.id = %s
            """,
            (str(tender_id),),
        )
        tender = cur.fetchone()
        if not tender:
            raise LiteRagError("tender_not_found", "No encontramos la licitacion solicitada.")

        cur.execute(
            """
            SELECT d.url, COALESCE(d.title, d.document_type, 'Documento PDF') AS title,
                   d.document_type, d.format, d.date_published
            FROM documents d
            WHERE d.tender_id = %s
              AND d.award_id IS NULL
              AND d.contract_id IS NULL
              AND LOWER(COALESCE(d.format, '')) IN ('pdf', 'application/pdf')
              AND d.url IS NOT NULL
            ORDER BY
              CASE
                WHEN LOWER(COALESCE(d.title, '')) LIKE '%%bases integradas%%' THEN 0
                WHEN LOWER(COALESCE(d.title, '')) LIKE '%%bases administrativas%%' THEN 1
                WHEN d.document_type = 'biddingDocuments' THEN 2
                ELSE 3
              END,
              d.date_published DESC NULLS LAST
            LIMIT %s
            """,
            (str(tender_id), MAX_DOCUMENTS),
        )
        documents = cur.fetchall() or []
        return dict(tender), [dict(row) for row in documents]
    finally:
        conn.close()


def _source_digest(documents: Iterable[dict]) -> str:
    source = "\n".join(
        f"{doc.get('url', '').strip()}|{doc.get('title', '').strip()}"
        for doc in documents
    )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _load_cache(tender_id: str, digest: str) -> Corpus | None:
    conn = _get_conn()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """
            SELECT texto, paginas, procesado_at
            FROM tender_documents_text
            WHERE tender_id = %s
            """,
            (str(tender_id),),
        )
        row = cur.fetchone()
        if not row or not str(row.get("texto") or "").startswith(CACHE_PREFIX):
            return None

        processed_at = row.get("procesado_at")
        if not processed_at or processed_at < datetime.now() - timedelta(days=CACHE_TTL_DAYS):
            return None

        payload = json.loads(row["texto"][len(CACHE_PREFIX):])
        if payload.get("source_digest") != digest:
            return None

        pages = [PageText(**page) for page in payload.get("pages", [])]
        if not pages:
            return None
        return Corpus(
            tender_id=str(tender_id),
            source_digest=digest,
            pages=pages,
            document_count=int(payload.get("document_count") or 0),
            total_pages=int(payload.get("total_pages") or len(pages)),
            total_chars=int(payload.get("total_chars") or sum(len(page.text) for page in pages)),
            scan_suspected=bool(payload.get("scan_suspected")),
            cache_hit=True,
        )
    except (ValueError, TypeError, json.JSONDecodeError):
        return None
    finally:
        conn.close()


def _save_cache(corpus: Corpus) -> None:
    payload = {
        "source_digest": corpus.source_digest,
        "pages": [asdict(page) for page in corpus.pages],
        "document_count": corpus.document_count,
        "total_pages": corpus.total_pages,
        "total_chars": corpus.total_chars,
        "scan_suspected": corpus.scan_suspected,
    }
    serialized = CACHE_PREFIX + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    conn = _get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            DELETE FROM tender_documents_text
            WHERE procesado_at < NOW() - (%s * INTERVAL '1 day')
              AND texto LIKE %s
            """,
            (CACHE_TTL_DAYS, f"{CACHE_PREFIX}%"),
        )
        cur.execute(
            """
            INSERT INTO tender_documents_text (tender_id, texto, paginas, procesado_at, tiene_ocr)
            VALUES (%s, %s, %s, NOW(), FALSE)
            ON CONFLICT (tender_id) DO UPDATE SET
                texto = EXCLUDED.texto,
                paginas = EXCLUDED.paginas,
                procesado_at = EXCLUDED.procesado_at,
                tiene_ocr = FALSE
            """,
            (corpus.tender_id, serialized, corpus.total_pages),
        )
        conn.commit()
    finally:
        conn.close()


def _extract_pdf(path: str, title: str, remaining_pages: int, remaining_chars: int) -> tuple[list[PageText], int, int]:
    if fitz is None:
        raise LiteRagError("pdf_runtime_missing", "El motor de lectura PDF no esta disponible.")
    pages: list[PageText] = []
    page_count = 0
    char_count = 0
    with fitz.open(path) as document:
        page_count = min(len(document), remaining_pages)
        for index in range(page_count):
            if char_count >= remaining_chars:
                break
            text = document[index].get_text("text", sort=True).strip()
            if not text:
                continue
            text = text[: max(0, remaining_chars - char_count)]
            pages.append(PageText(document=title[:200], page=index + 1, text=text))
            char_count += len(text)
    return pages, page_count, char_count


def edge_cache_status(tender_id: str) -> dict:
    """Return cache readiness without attempting a SEACE download."""
    _, documents = get_tender_bundle(tender_id)
    if not documents:
        raise LiteRagError("no_pdf", "Esta licitacion no tiene documentos PDF compatibles.")
    digest = _source_digest(documents)
    cached = _load_cache(str(tender_id), digest)
    required_documents = len(documents)
    return {
        "ready": bool(cached and cached.document_count >= required_documents),
        "cached_documents": cached.document_count if cached else 0,
        "required_documents": required_documents,
    }


def ingest_edge_pdf(tender_id: str, document_url: str, path: str) -> dict:
    """Merge one trusted edge-fetched PDF into the tender text cache."""
    tender, documents = get_tender_bundle(tender_id)
    if not documents:
        raise LiteRagError("no_pdf", "Esta licitacion no tiene documentos PDF compatibles.")

    normalized_url = normalize_document_url(document_url)
    matched_index = -1
    matched_document = None
    for index, document in enumerate(documents):
        try:
            candidate_url = normalize_document_url(document.get("url"))
        except DocumentDownloadError:
            continue
        if candidate_url == normalized_url:
            matched_index = index
            matched_document = document
            break
    if matched_document is None:
        raise LiteRagError("document_mismatch", "El documento no pertenece a la licitacion.")

    base_title = str(matched_document.get("title") or "Documento PDF")[:180]
    duplicate_count = sum(
        1 for document in documents if str(document.get("title") or "Documento PDF")[:180] == base_title
    )
    display_title = f"{base_title} ({matched_index + 1})" if duplicate_count > 1 else base_title
    page_budget = max(1, MAX_TOTAL_PAGES // len(documents))
    char_budget = max(1, MAX_TOTAL_CHARS // len(documents))
    pages, _, _ = _extract_pdf(path, display_title, page_budget, char_budget)

    digest = _source_digest(documents)
    existing = _load_cache(str(tender_id), digest)
    existing_pages = existing.pages if existing else []
    merged_pages = [page for page in existing_pages if page.document != display_title]
    merged_pages.extend(pages)
    attempted_documents = max(
        existing.document_count if existing else 0,
        matched_index + 1,
    )

    if not merged_pages:
        raise LiteRagError(
            "scanned_pdf",
            "El PDF no contiene texto seleccionable y requiere OCR.",
        )

    corpus = Corpus(
        tender_id=str(tender_id),
        source_digest=digest,
        pages=merged_pages,
        document_count=attempted_documents,
        total_pages=len(merged_pages),
        total_chars=sum(len(page.text) for page in merged_pages),
        scan_suspected=bool(existing and existing.scan_suspected) or not pages,
        cache_hit=False,
    )
    _save_cache(corpus)
    _memory_cache_set(str(tender_id), tender, corpus)
    return {
        "ready": corpus.document_count >= len(documents),
        "document_pages": len(pages),
        "cached_documents": corpus.document_count,
        "required_documents": len(documents),
    }


def prepare_corpus(tender_id: str) -> tuple[dict, Corpus]:
    memory_cached = _memory_cache_get(str(tender_id))
    if memory_cached:
        return memory_cached

    tender, documents = get_tender_bundle(tender_id)
    if not documents:
        raise LiteRagError("no_pdf", "Esta licitacion no tiene documentos PDF compatibles.")

    digest = _source_digest(documents)
    cached = _load_cache(str(tender_id), digest)
    if cached:
        _memory_cache_set(str(tender_id), tender, cached)
        return tender, cached

    pages: list[PageText] = []
    total_pages = 0
    total_chars = 0
    processed_documents = 0
    failures: list[str] = []

    with tempfile.TemporaryDirectory(prefix="licigob-lite-") as temp_dir:
        for index, document in enumerate(documents):
            if total_pages >= MAX_TOTAL_PAGES or total_chars >= MAX_TOTAL_CHARS:
                break
            try:
                path = download_document(
                    document["url"],
                    temp_dir,
                    suffix=".pdf",
                    max_bytes=MAX_FILE_BYTES,
                )
                remaining_documents = len(documents) - index
                remaining_pages = MAX_TOTAL_PAGES - total_pages
                page_budget = max(1, remaining_pages // remaining_documents)
                extracted, counted_pages, counted_chars = _extract_pdf(
                    path,
                    str(document.get("title") or "Documento PDF"),
                    page_budget,
                    MAX_TOTAL_CHARS - total_chars,
                )
                pages.extend(extracted)
                total_pages += counted_pages
                total_chars += counted_chars
                processed_documents += 1
            except DocumentDownloadError as exc:
                failures.append(exc.code)
                logger.warning(
                    "document_download_failed tender=%s title=%s code=%s",
                    tender_id,
                    str(document.get("title") or "Documento PDF")[:120],
                    exc.code,
                )
            except LiteRagError:
                failures.append("pdf_runtime_missing")
            except Exception:
                failures.append("invalid_pdf")

    nonempty_ratio = len(pages) / max(total_pages, 1)
    scan_suspected = total_chars < 500 or (total_pages >= 4 and nonempty_ratio < 0.25)
    if not pages or total_chars < 300:
        if processed_documents == 0 and failures:
            raise LiteRagError(
                failures[0],
                "No pudimos descargar los documentos desde SEACE.",
            )
        code = "ocr_required" if scan_suspected else "no_text"
        message = (
            "El PDF parece escaneado y requiere OCR."
            if scan_suspected
            else "No pudimos extraer texto util de los documentos."
        )
        raise LiteRagError(code, message)

    corpus = Corpus(
        tender_id=str(tender_id),
        source_digest=digest,
        pages=pages,
        document_count=processed_documents,
        total_pages=total_pages,
        total_chars=total_chars,
        scan_suspected=scan_suspected,
    )
    _save_cache(corpus)
    _memory_cache_set(str(tender_id), tender, corpus)
    return tender, corpus


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.lower())
    return "".join(char for char in value if not unicodedata.combining(char))


def _tokens(value: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]{3,}", _normalize(value))
        if token not in STOP_WORDS
    ]


def _query_tokens(value: str) -> list[str]:
    tokens = _tokens(value)
    expanded = list(tokens)
    if any(token.startswith("especific") for token in tokens):
        expanded.extend(["tecnica", "tecnico", "requisito", "caracteristica", "ficha", "cumplimiento"])
    if any(token.startswith("convoc") for token in tokens):
        expanded.extend(["postor", "proveedor", "participante", "oferta"])
    if any(token.startswith("document") for token in tokens):
        expanded.extend(["acreditar", "certificado", "declaracion", "registro", "presentar"])
    return expanded


def select_context(corpus: Corpus, question: str, max_pages: int = 6) -> tuple[str, list[dict]]:
    query_tokens = _query_tokens(question)
    query_counts = Counter(query_tokens)
    normalized_question = _normalize(question).strip()
    scored: list[tuple[float, PageText]] = []

    for page in corpus.pages:
        normalized_text = _normalize(page.text)
        page_counts = Counter(_tokens(page.text))
        score = sum(min(page_counts[token], 8) * (1.0 + query_counts[token]) for token in query_counts)
        if normalized_question and len(normalized_question) >= 8 and normalized_question in normalized_text:
            score += 20
        if any(token in _normalize(page.document) for token in query_counts):
            score += 2
        scored.append((score, page))

    scored.sort(key=lambda item: (item[0], -item[1].page), reverse=True)
    selected = [page for score, page in scored if score > 0][:max_pages]
    if not selected:
        selected = corpus.pages[: min(3, max_pages)]

    context_parts: list[str] = []
    references: list[dict] = []
    used_chars = 0
    for page in selected:
        header = f"[Documento: {page.document} | Pagina: {page.page}]\n"
        available = MAX_CONTEXT_CHARS - used_chars - len(header)
        if available <= 200:
            break
        excerpt = page.text[:available]
        context_parts.append(header + excerpt)
        references.append({"document": page.document, "page": page.page})
        used_chars += len(header) + len(excerpt)

    return "\n\n".join(context_parts), references


def tender_summary(tender: dict) -> str:
    return json.dumps(tender, ensure_ascii=False, default=_json_default)

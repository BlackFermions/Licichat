"""Single-run Container Apps Job for LiciGob document ingestion."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import tempfile
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from document_fetcher import DocumentDownloadError, download_document

from ingestion.chunking import build_chunks
from ingestion.config import Settings
from ingestion.db import Database
from ingestion.errors import IngestionError
from ingestion.extractors import detect_kind, extract_source, source_suffix
from ingestion.intelligence import IntelligenceClient
from ingestion.models import TextChunk
from ingestion.storage import ObjectStorage


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("licigob-ai-ingestion")

CONTENT_TYPES = {
    "pdf": "application/pdf",
    "zip": "application/zip",
    "rar": "application/vnd.rar",
    "7z": "application/x-7z-compressed",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "txt": "text/plain",
}


def _safe_component(value: Any, fallback: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip()).strip("-.")
    return (normalized or fallback)[:120]


def _filename(document: dict[str, Any]) -> str:
    suffix = source_suffix(document.get("format"), document.get("title"))
    title = _safe_component(document.get("title"), document.get("role") or "documento")
    if not title.lower().endswith(suffix):
        title += suffix
    return title


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_error(exc: DocumentDownloadError) -> IngestionError:
    transient_codes = {
        "download_timeout",
        "download_failed",
        "seace_forbidden",
        "proxy_forbidden",
        "proxy_timeout",
        "proxy_download_failed",
        "proxy_upstream_failed",
    }
    return IngestionError(exc.code, str(exc), transient=exc.code in transient_codes)


def _process_document(
    settings: Settings,
    db: Database,
    storage: ObjectStorage,
    intelligence: IntelligenceClient,
    job: dict[str, Any],
    document: dict[str, Any],
    temp_dir: str,
) -> tuple[int, list[TextChunk]]:
    filename = _filename(document)
    asset = db.start_asset(job, document, filename)
    asset_id = int(asset["id"])
    if asset.get("status") == "ready" and asset.get("source_url") == document.get("url"):
        cached = db.load_chunks(asset_id)
        if cached:
            logger.info(
                "asset_cache_hit job=%s tender=%s asset=%s chunks=%s",
                job["id"], job["tender_id"], asset_id, len(cached),
            )
            return asset_id, cached

    started = time.monotonic()
    db.event(job, "download", "started", asset_id=asset_id)
    try:
        tender_key = _safe_component(job["tender_id"], "tender")
        staged_key = str(asset.get("original_object_key") or "").strip()
        if staged_key:
            expected_prefix = f"originals/{tender_key}/"
            if not staged_key.startswith(expected_prefix):
                raise IngestionError(
                    "invalid_staged_blob",
                    "La referencia del documento almacenado no es valida.",
                )
            path = str(Path(temp_dir) / f"staged_{uuid.uuid4().hex}.bin")
            storage.download_file(staged_key, path, settings.download_max_bytes)
            download_source = "blob"
        else:
            try:
                path = download_document(
                    document["url"],
                    temp_dir,
                    suffix=".bin",
                    max_bytes=settings.download_max_bytes,
                    timeout=(10, 90),
                )
            except DocumentDownloadError as exc:
                raise _download_error(exc) from exc
            download_source = "seace"

        byte_size = Path(path).stat().st_size
        digest = _sha256(path)
        kind = detect_kind(path)
        content_type = CONTENT_TYPES[kind]
        source_key = _safe_component(document.get("source_document_id"), "source")
        original_key = staged_key or f"originals/{tender_key}/{source_key}/{digest}/{filename}"
        if not staged_key:
            storage.upload_file(original_key, path, content_type)
        db.asset_status(
            asset_id,
            "extracting",
            detected_mime_type=content_type,
            sha256=digest,
            byte_size=byte_size,
            original_object_key=original_key,
        )
        db.event(
            job,
            "download",
            "succeeded",
            asset_id=asset_id,
            duration_ms=int((time.monotonic() - started) * 1000),
            metrics={"bytes": byte_size, "kind": kind, "source": download_source},
        )

        extract_started = time.monotonic()
        result = extract_source(path, filename, str(document["role"]), settings)
        if result.text_char_count < 300 or not result.pages:
            raise IngestionError("no_useful_text", "No se encontro texto suficiente en el documento.")

        markdown_key = f"markdown/{tender_key}/{asset_id}/{settings.pipeline_version}.md"
        storage.upload_text(markdown_key, result.markdown)
        chunks = build_chunks(
            result.pages,
            settings.chunk_target_tokens,
            settings.chunk_overlap_tokens,
        )
        if not chunks:
            raise IngestionError("no_chunks", "No se pudieron construir fragmentos del documento.")

        vectors = intelligence.embed_texts([chunk.content for chunk in chunks])
        embedded_chunks = [replace(chunk, embedding=vector) for chunk, vector in zip(chunks, vectors)]
        db.replace_chunks(str(job["tender_id"]), asset_id, embedded_chunks)
        db.asset_status(
            asset_id,
            "ready",
            markdown_object_key=markdown_key,
            page_count=result.page_count,
            native_text_pages=result.native_text_pages,
            ocr_pages=result.ocr_pages,
            text_char_count=result.text_char_count,
            extraction_coverage={
                "pages": result.page_count,
                "native_pages": result.native_text_pages,
                "ocr_pages": result.ocr_pages,
                "skipped_pages": result.skipped_pages,
                "warnings": result.warnings,
                "partial": bool(result.skipped_pages or result.warnings),
            },
            error_code=None,
            error_message=None,
        )
        db.event(
            job,
            "extract_embed",
            "succeeded",
            asset_id=asset_id,
            duration_ms=int((time.monotonic() - extract_started) * 1000),
            metrics={
                "pages": result.page_count,
                "native_pages": result.native_text_pages,
                "ocr_pages": result.ocr_pages,
                "skipped_pages": result.skipped_pages,
                "characters": result.text_char_count,
                "chunks": len(embedded_chunks),
                "warnings": result.warnings,
            },
        )
        return asset_id, embedded_chunks
    except IngestionError as exc:
        db.asset_status(asset_id, "failed", error_code=exc.code, error_message=exc.safe_message)
        db.event(
            job,
            "document",
            "failed",
            asset_id=asset_id,
            duration_ms=int((time.monotonic() - started) * 1000),
            error_code=exc.code,
            error_message=exc.safe_message,
        )
        raise
    except Exception as exc:
        db.asset_status(
            asset_id,
            "failed",
            error_code="unexpected_document_error",
            error_message="Fallo inesperado durante el procesamiento documental.",
        )
        db.event(
            job,
            "document",
            "failed",
            asset_id=asset_id,
            duration_ms=int((time.monotonic() - started) * 1000),
            error_code="unexpected_document_error",
            error_message="Fallo inesperado durante el procesamiento documental.",
        )
        raise IngestionError(
            "unexpected_document_error",
            "Fallo inesperado durante el procesamiento documental.",
            transient=True,
        ) from exc


def process_job(
    settings: Settings,
    db: Database,
    storage: ObjectStorage,
    intelligence: IntelligenceClient,
    job: dict[str, Any],
) -> None:
    documents = job.get("selected_documents")
    if isinstance(documents, str):
        documents = json.loads(documents)
    if not isinstance(documents, list) or not documents:
        raise IngestionError("invalid_job_payload", "El trabajo no contiene documentos validos.")

    started = time.monotonic()
    db.event(job, "job", "started", metrics={"documents": len(documents), "attempt": job["attempts"]})
    initial_usage = dict(intelligence.usage)
    chunks: list[TextChunk] = []
    ready_documents = 0
    optional_transient_error: IngestionError | None = None

    with tempfile.TemporaryDirectory(prefix="licigob-ai-ingestion-") as temp_dir:
        for document in documents:
            role = str(document.get("role") or "")
            try:
                _, document_chunks = _process_document(
                    settings, db, storage, intelligence, job, document, temp_dir
                )
                chunks.extend(document_chunks)
                ready_documents += 1
            except IngestionError as exc:
                logger.warning(
                    "document_failed job=%s tender=%s role=%s code=%s transient=%s",
                    job["id"], job["tender_id"], role, exc.code, exc.transient,
                )
                if role == "bases":
                    raise
                if exc.transient:
                    optional_transient_error = exc

    if optional_transient_error:
        raise optional_transient_error
    if not chunks:
        raise IngestionError("no_chunks", "Las bases no produjeron contenido consultable.")

    tender = db.tender_metadata(str(job["tender_id"]))
    profile_started = time.monotonic()
    profile = intelligence.build_profile(tender, chunks)
    search_text = intelligence.search_text(tender, profile)
    profile_embedding = intelligence.embed_texts([search_text])[0]
    db.upsert_profile(job, tender, profile, search_text, profile_embedding, ready_documents)
    db.event(
        job,
        "profile",
        "succeeded",
        duration_ms=int((time.monotonic() - profile_started) * 1000),
        metrics={"search_characters": len(search_text), "documents": ready_documents},
    )
    db.complete_job(int(job["id"]))
    db.event(
        job,
        "job",
        "succeeded",
        duration_ms=int((time.monotonic() - started) * 1000),
        metrics={
            "documents": ready_documents, "chunks": len(chunks),
            "usage": {key: value - initial_usage[key] for key, value in intelligence.usage.items()},
        },
    )
    logger.info(
        "job_ready job=%s tender=%s documents=%s chunks=%s elapsed_ms=%s",
        job["id"], job["tender_id"], ready_documents, len(chunks),
        int((time.monotonic() - started) * 1000),
    )


def main() -> int:
    settings = Settings.from_env()
    logger.info(
        "worker_start worker=%s pipeline=%s batch_size=%s",
        settings.worker_id,
        settings.pipeline_version,
        settings.batch_size,
    )
    db = Database(settings)
    storage = ObjectStorage(settings)
    intelligence = IntelligenceClient(settings)
    jobs = db.claim_jobs()
    logger.info("jobs_claimed count=%s", len(jobs))
    for job in jobs:
        try:
            process_job(settings, db, storage, intelligence, job)
        except IngestionError as exc:
            status = db.fail_job(job, exc.code, exc.safe_message, exc.transient)
            db.event(
                job,
                "job",
                "failed",
                error_code=exc.code,
                error_message=exc.safe_message,
                metrics={"queue_status": status, "attempt": job["attempts"]},
            )
            logger.warning(
                "job_not_ready job=%s tender=%s code=%s queue_status=%s",
                job["id"], job["tender_id"], exc.code, status,
            )
        except Exception:
            logger.exception("job_unexpected_failure job=%s tender=%s", job["id"], job["tender_id"])
            status = db.fail_job(
                job,
                "unexpected_job_error",
                "Fallo inesperado durante la ingesta.",
                True,
            )
            db.event(
                job,
                "job",
                "failed",
                error_code="unexpected_job_error",
                error_message="Fallo inesperado durante la ingesta.",
                metrics={"queue_status": status, "attempt": job["attempts"]},
            )
    logger.info("worker_stop processed=%s", len(jobs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

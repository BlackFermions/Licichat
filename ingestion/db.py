"""PostgreSQL queue and persistence operations for the ingestion pilot."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

import psycopg2
from psycopg2.extras import Json, RealDictCursor

from ingestion.config import Settings
from ingestion.models import TextChunk


def _vector(value: list[float]) -> str:
    return "[" + ",".join(f"{number:.9g}" for number in value) + "]"


class Database:
    def __init__(self, settings: Settings):
        self.settings = settings

    @contextmanager
    def connection(self) -> Iterator[psycopg2.extensions.connection]:
        conn = psycopg2.connect(**self.settings.db_config())
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def claim_jobs(self) -> list[dict[str, Any]]:
        with self.connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """
                    UPDATE ai_ingestion_jobs
                    SET status = 'retry', locked_at = NULL, locked_by = NULL,
                        available_at = NOW(), last_error_code = 'stale_lock',
                        last_error_message = 'El trabajo fue recuperado tras exceder el tiempo de bloqueo.'
                    WHERE status = 'processing'
                      AND locked_at < NOW() - (%s * INTERVAL '1 minute')
                    """,
                    (self.settings.stale_lock_minutes,),
                )
                cursor.execute(
                    """
                    WITH candidates AS (
                        SELECT id
                        FROM ai_ingestion_jobs
                        WHERE status IN ('pending', 'retry')
                          AND available_at <= NOW()
                          AND attempts < max_attempts
                          AND pipeline_version = %s
                          AND NOT EXISTS (
                              SELECT 1
                              FROM jsonb_array_elements(selected_documents) document
                              WHERE NOT EXISTS (
                                  SELECT 1 FROM ai_document_assets asset
                                  WHERE asset.tender_id = ai_ingestion_jobs.tender_id
                                    AND asset.pipeline_version = ai_ingestion_jobs.pipeline_version
                                    AND asset.source_document_id = document->>'source_document_id'
                                    AND asset.source_url = document->>'url'
                                    AND asset.original_object_key IS NOT NULL
                              )
                          )
                        ORDER BY priority DESC, available_at, id
                        FOR UPDATE SKIP LOCKED
                        LIMIT %s
                    )
                    UPDATE ai_ingestion_jobs jobs
                    SET status = 'processing', attempts = jobs.attempts + 1,
                        locked_at = NOW(), locked_by = %s,
                        started_at = COALESCE(jobs.started_at, NOW()),
                        last_error_code = NULL, last_error_message = NULL
                    FROM candidates
                    WHERE jobs.id = candidates.id
                    RETURNING jobs.*
                    """,
                    (self.settings.pipeline_version, self.settings.batch_size, self.settings.worker_id),
                )
                return [dict(row) for row in cursor.fetchall()]

    def tender_metadata(self, tender_id: str) -> dict[str, Any]:
        with self.connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT t.id, t.title, t.description, t.procurement_method_details,
                           t.main_procurement_category, t.value_amount, t.currency,
                           t.date_published, COALESCE(p.name, p.legal_name) AS buyer_name
                    FROM tenders t
                    LEFT JOIN parties p ON p.id = t.buyer_id
                    WHERE t.id = %s
                    """,
                    (tender_id,),
                )
                row = cursor.fetchone()
                if not row:
                    raise RuntimeError("tender_not_found")
                return dict(row)

    def start_asset(self, job: dict[str, Any], document: dict[str, Any], filename: str) -> dict[str, Any]:
        tender_id = str(job["tender_id"])
        source_id = str(document["source_document_id"])
        with self.connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT * FROM ai_document_assets
                    WHERE tender_id = %s AND source_document_id = %s
                      AND extracted_filename = %s AND pipeline_version = %s
                    FOR UPDATE
                    """,
                    (tender_id, source_id, filename, self.settings.pipeline_version),
                )
                existing = cursor.fetchone()
                if existing and existing["status"] == "ready" and existing["source_url"] == document["url"]:
                    return dict(existing)
                if existing:
                    cursor.execute(
                        """
                        UPDATE ai_document_assets
                        SET job_id = %s, document_role = %s, source_title = %s,
                            source_url = %s, source_format = %s, source_published_at = %s,
                            original_object_key = CASE WHEN source_url = %s THEN original_object_key ELSE NULL END,
                            status = 'downloading', error_code = NULL, error_message = NULL
                        WHERE id = %s RETURNING *
                        """,
                        (
                            job["id"], document["role"], document.get("title"), document["url"],
                            document.get("format"), document.get("published_at"), document["url"], existing["id"],
                        ),
                    )
                    return dict(cursor.fetchone())
                cursor.execute(
                    """
                    INSERT INTO ai_document_assets (
                        tender_id, job_id, source_document_id, document_role, source_title,
                        source_url, source_format, source_published_at, extracted_filename,
                        status, pipeline_version
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'downloading', %s)
                    RETURNING *
                    """,
                    (
                        tender_id, job["id"], source_id, document["role"], document.get("title"),
                        document["url"], document.get("format"), document.get("published_at"), filename,
                        self.settings.pipeline_version,
                    ),
                )
                return dict(cursor.fetchone())

    def asset_status(self, asset_id: int, status: str, **fields: Any) -> None:
        allowed = {
            "detected_mime_type", "sha256", "byte_size", "original_object_key", "markdown_object_key",
            "page_count", "native_text_pages", "ocr_pages", "text_char_count", "error_code", "error_message",
            "extraction_coverage",
        }
        selected = {
            key: Json(value) if key == "extraction_coverage" else value
            for key, value in fields.items() if key in allowed
        }
        assignments = ["status = %s"] + [f"{key} = %s" for key in selected]
        params = [status, *selected.values(), asset_id]
        with self.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"UPDATE ai_document_assets SET {', '.join(assignments)} WHERE id = %s",
                    params,
                )

    def replace_chunks(self, tender_id: str, asset_id: int, chunks: list[TextChunk]) -> None:
        with self.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM ai_tender_chunks WHERE asset_id = %s AND pipeline_version = %s",
                    (asset_id, self.settings.pipeline_version),
                )
                cursor.executemany(
                    """
                    INSERT INTO ai_tender_chunks (
                        tender_id, asset_id, chunk_index, page_start, page_end, section_title,
                        content, token_count, embedding, embedding_model, pipeline_version
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s, %s)
                    """,
                    [
                        (
                            tender_id, asset_id, chunk.index, chunk.page_start, chunk.page_end,
                            chunk.section_title, chunk.content, chunk.token_count,
                            _vector(chunk.embedding or []), self.settings.embedding_deployment,
                            self.settings.pipeline_version,
                        )
                        for chunk in chunks
                    ],
                )

    def load_chunks(self, asset_id: int) -> list[TextChunk]:
        with self.connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT chunk_index, content, page_start, page_end, section_title, token_count
                    FROM ai_tender_chunks
                    WHERE asset_id = %s AND pipeline_version = %s
                    ORDER BY chunk_index
                    """,
                    (asset_id, self.settings.pipeline_version),
                )
                return [
                    TextChunk(
                        index=row["chunk_index"], content=row["content"], page_start=row["page_start"],
                        page_end=row["page_end"], section_title=row["section_title"], token_count=row["token_count"],
                    )
                    for row in cursor.fetchall()
                ]

    def upsert_profile(
        self,
        job: dict[str, Any],
        tender: dict[str, Any],
        profile: dict[str, Any],
        search_text: str,
        embedding: list[float],
        document_count: int,
    ) -> None:
        with self.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """SELECT id, document_role, extraction_coverage
                       FROM ai_document_assets
                       WHERE tender_id = %s AND pipeline_version = %s AND status = 'ready'
                       ORDER BY id""",
                    (job["tender_id"], self.settings.pipeline_version),
                )
                profile["key_info"]["document_coverage"] = [
                    {"asset_id": row[0], "role": row[1], "coverage": row[2]}
                    for row in cursor.fetchall()
                ]
                cursor.execute(
                    """
                    INSERT INTO ai_tender_profiles (
                        tender_id, source_signature, title, summary, search_text, key_info,
                        document_count, embedding, chat_model, embedding_model, pipeline_version,
                        ready_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector, %s, %s, %s, NOW(), NOW())
                    ON CONFLICT (tender_id) DO UPDATE SET
                        source_signature = EXCLUDED.source_signature,
                        title = EXCLUDED.title,
                        summary = EXCLUDED.summary,
                        search_text = EXCLUDED.search_text,
                        key_info = EXCLUDED.key_info,
                        document_count = EXCLUDED.document_count,
                        embedding = EXCLUDED.embedding,
                        chat_model = EXCLUDED.chat_model,
                        embedding_model = EXCLUDED.embedding_model,
                        pipeline_version = EXCLUDED.pipeline_version,
                        ready_at = NOW(), updated_at = NOW()
                    """,
                    (
                        job["tender_id"], job["source_signature"], tender.get("title"), profile["summary"],
                        search_text, Json(profile["key_info"]), document_count, _vector(embedding),
                        self.settings.chat_deployment if self.settings.chat_summaries else None,
                        self.settings.embedding_deployment, self.settings.pipeline_version,
                    ),
                )

    def event(
        self,
        job: dict[str, Any],
        stage: str,
        status: str,
        *,
        asset_id: int | None = None,
        duration_ms: int | None = None,
        metrics: dict[str, Any] | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        with self.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO ai_pipeline_events (
                        job_id, tender_id, asset_id, stage, status, duration_ms,
                        metrics, error_code, error_message
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        job["id"], job["tender_id"], asset_id, stage, status, duration_ms,
                        Json(metrics or {}), error_code, (error_message or "")[:500] or None,
                    ),
                )

    def complete_job(self, job_id: int) -> None:
        with self.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE ai_ingestion_jobs
                    SET status = 'ready', completed_at = NOW(), locked_at = NULL,
                        locked_by = NULL, last_error_code = NULL, last_error_message = NULL
                    WHERE id = %s
                    """,
                    (job_id,),
                )

    def fail_job(self, job: dict[str, Any], code: str, message: str, transient: bool) -> str:
        attempts = int(job["attempts"])
        max_attempts = int(job["max_attempts"])
        retry = transient and attempts < max_attempts
        status = "retry" if retry else "failed"
        delay_minutes = min(60, 2 ** max(attempts - 1, 0))
        with self.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE ai_ingestion_jobs
                    SET status = %s, available_at = CASE WHEN %s THEN NOW() + (%s * INTERVAL '1 minute') ELSE available_at END,
                        locked_at = NULL, locked_by = NULL, completed_at = CASE WHEN %s THEN NULL ELSE NOW() END,
                        last_error_code = %s, last_error_message = %s
                    WHERE id = %s
                    """,
                    (status, retry, delay_minutes, retry, code[:80], message[:500], job["id"]),
                )
        return status

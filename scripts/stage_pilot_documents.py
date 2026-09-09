"""Stage selected SEACE documents in Azure Blob before compute ingestion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import psycopg2
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContentSettings
from psycopg2.extras import RealDictCursor

from document_fetcher import DocumentDownloadError, download_document
from ingestion.extractors import detect_kind, source_suffix


CONTENT_TYPES = {
    "pdf": "application/pdf",
    "zip": "application/zip",
    "rar": "application/vnd.rar",
    "7z": "application/x-7z-compressed",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "txt": "text/plain",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Guarda originales del piloto en Blob.")
    parser.add_argument("--tender-id")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--pipeline-version", default="pilot-v2")
    return parser.parse_args()


def required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Falta la variable de entorno {name}")
    return value


def db_config() -> dict[str, Any]:
    return {
        "host": required("DB_HOST"),
        "user": required("DB_USER"),
        "password": required("DB_PASSWORD"),
        "dbname": required("DB_NAME"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "sslmode": os.getenv("DB_SSLMODE", "require"),
        "connect_timeout": 20,
        "application_name": "licigob-ai-document-staging",
    }


def blob_container():
    connection_string = os.getenv("AI_BLOB_CONNECTION_STRING", "").strip()
    if connection_string:
        service = BlobServiceClient.from_connection_string(connection_string)
    else:
        credential = DefaultAzureCredential(exclude_interactive_browser_credential=True)
        service = BlobServiceClient(account_url=required("AI_BLOB_ACCOUNT_URL"), credential=credential)
    return service.get_container_client(os.getenv("AI_BLOB_CONTAINER", "licigob-ai-documents"))


def safe_component(value: Any, fallback: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip()).strip("-.")
    return (normalized or fallback)[:120]


def filename(document: dict[str, Any]) -> str:
    suffix = source_suffix(document.get("format"), document.get("title"))
    name = safe_component(document.get("title"), document.get("role") or "documento")
    return name if name.lower().endswith(suffix) else name + suffix


def digest(path: str) -> str:
    sha = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def load_jobs(conn, args: argparse.Namespace) -> list[dict[str, Any]]:
    conditions = [
        "pipeline_version = %s",
        "status IN ('pending', 'retry', 'failed')",
        "available_at = 'infinity'::timestamptz",
    ]
    params: list[Any] = [args.pipeline_version]
    if args.tender_id:
        conditions.append("tender_id = %s")
        params.append(args.tender_id)
    params.append(args.limit)
    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute(
            f"""
            SELECT id, tender_id, selected_documents
            FROM ai_ingestion_jobs
            WHERE {' AND '.join(conditions)}
            ORDER BY priority DESC, id
            LIMIT %s
            """,
            params,
        )
        return [dict(row) for row in cursor.fetchall()]


def upsert_asset(
    conn,
    job: dict[str, Any],
    document: dict[str, Any],
    extracted_filename: str,
    object_key: str,
    content_type: str,
    sha256: str,
    byte_size: int,
    pipeline_version: str,
) -> None:
    values = (
        job["id"], document["role"], document.get("title"), document["url"],
        document.get("format"), document.get("published_at"), content_type, sha256,
        byte_size, object_key, job["tender_id"], document["source_document_id"],
        extracted_filename, pipeline_version,
    )
    with conn.cursor() as cursor:
        cursor.execute(
            """
            UPDATE ai_document_assets
            SET job_id = %s, document_role = %s, source_title = %s, source_url = %s,
                source_format = %s, source_published_at = %s, detected_mime_type = %s,
                sha256 = %s, byte_size = %s, original_object_key = %s,
                status = 'pending', error_code = NULL, error_message = NULL
            WHERE tender_id = %s AND source_document_id = %s
              AND extracted_filename = %s AND pipeline_version = %s
            """,
            values,
        )
        if cursor.rowcount:
            return
        cursor.execute(
            """
            INSERT INTO ai_document_assets (
                job_id, document_role, source_title, source_url, source_format,
                source_published_at, detected_mime_type, sha256, byte_size,
                original_object_key, tender_id, source_document_id,
                extracted_filename, pipeline_version, status
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'pending')
            """,
            values,
        )


def main() -> int:
    args = arguments()
    if args.limit < 1 or args.limit > 200:
        raise RuntimeError("--limit debe estar entre 1 y 200")
    max_bytes = int(os.getenv("AI_DOWNLOAD_MAX_FILE_BYTES", str(50 * 1024 * 1024)))
    conn = psycopg2.connect(**db_config())
    container = blob_container()
    failures = 0
    staged_jobs = 0
    try:
        jobs = load_jobs(conn, args)
        print(f"Trabajos seleccionados para staging: {len(jobs)}")
        for job in jobs:
            documents = job["selected_documents"]
            if isinstance(documents, str):
                documents = json.loads(documents)
            staged = 0
            with tempfile.TemporaryDirectory(prefix="licigob-ai-stage-") as temp_dir:
                for document in documents:
                    role = str(document.get("role") or "documento")
                    try:
                        name = filename(document)
                        with conn.cursor() as cursor:
                            cursor.execute(
                                """SELECT original_object_key, detected_mime_type, sha256, byte_size
                                   FROM ai_document_assets
                                   WHERE tender_id = %s AND source_document_id = %s
                                     AND source_url = %s AND extracted_filename = %s
                                     AND original_object_key IS NOT NULL
                                   ORDER BY (pipeline_version = %s) DESC, updated_at DESC
                                   LIMIT 1""",
                                (job["tender_id"], document["source_document_id"],
                                 document["url"], name, args.pipeline_version),
                            )
                            existing = cursor.fetchone()
                        conn.commit()
                        if existing and existing[0] and container.get_blob_client(existing[0]).exists():
                            upsert_asset(
                                conn, job, document, name, existing[0], existing[1],
                                existing[2], existing[3], args.pipeline_version,
                            )
                            conn.commit()
                            staged += 1
                            print(f"  tender={job['tender_id']} role={role} cached")
                            continue
                        path = download_document(
                            document["url"], temp_dir, suffix=".bin", max_bytes=max_bytes,
                            timeout=(10, 120),
                        )
                        kind = detect_kind(path)
                        sha256 = digest(path)
                        name = filename(document)
                        object_key = "/".join(
                            (
                                "originals",
                                safe_component(job["tender_id"], "tender"),
                                safe_component(document.get("source_document_id"), "source"),
                                sha256,
                                name,
                            )
                        )
                        with Path(path).open("rb") as handle:
                            container.upload_blob(
                                name=object_key,
                                data=handle,
                                overwrite=True,
                                content_settings=ContentSettings(content_type=CONTENT_TYPES[kind]),
                            )
                        upsert_asset(
                            conn, job, document, name, object_key, CONTENT_TYPES[kind], sha256,
                            Path(path).stat().st_size, args.pipeline_version,
                        )
                        conn.commit()
                        staged += 1
                        Path(path).unlink(missing_ok=True)
                        print(f"  tender={job['tender_id']} role={role} staged kind={kind}")
                    except DocumentDownloadError as exc:
                        conn.rollback()
                        record_failure(conn, job, exc.code, "No se pudo preparar un documento de SEACE.")
                        failures += 1
                        print(f"  tender={job['tender_id']} role={role} failed code={exc.code}")
                    except Exception:
                        conn.rollback()
                        record_failure(conn, job, "staging_failed", "No se pudo almacenar un documento del piloto.")
                        failures += 1
                        print(f"  tender={job['tender_id']} role={role} failed code=staging_failed")
            if staged == len(documents):
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        UPDATE ai_ingestion_jobs
                        SET status = 'pending', attempts = 0, available_at = NOW(),
                            completed_at = NULL, locked_at = NULL, locked_by = NULL,
                            last_error_code = NULL, last_error_message = NULL
                        WHERE id = %s
                        """,
                        (job["id"],),
                    )
                conn.commit()
                staged_jobs += 1
        print(f"Trabajos listos para compute: {staged_jobs}; fallos documentales: {failures}")
        return 1 if failures else 0
    finally:
        conn.close()


def record_failure(conn, job: dict[str, Any], code: str, message: str) -> None:
    with conn.cursor() as cursor:
        cursor.execute(
            """UPDATE ai_ingestion_jobs SET status = 'failed', last_error_code = %s,
                      last_error_message = %s
               WHERE id = %s AND available_at = 'infinity'::timestamptz""",
            (code, message, job["id"]),
        )
        cursor.execute(
            """INSERT INTO ai_pipeline_events (job_id, tender_id, stage, status, error_code, error_message)
               VALUES (%s, %s, 'staging', 'failed', %s, %s)""",
            (job["id"], job["tender_id"], code, message),
        )
    conn.commit()


if __name__ == "__main__":
    raise SystemExit(main())

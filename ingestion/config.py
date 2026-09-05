"""Environment-backed configuration with bounded pilot defaults."""

from __future__ import annotations

import os
import socket
import uuid
from dataclasses import dataclass
from typing import Any


def _integer(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} debe ser un entero") from exc
    if not minimum <= value <= maximum:
        raise RuntimeError(f"{name} debe estar entre {minimum} y {maximum}")
    return value


def _boolean(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _required(name: str, *aliases: str) -> str:
    for candidate in (name, *aliases):
        value = os.getenv(candidate, "").strip()
        if value:
            return value
    raise RuntimeError(f"Falta la variable de entorno {name}")


@dataclass(frozen=True)
class Settings:
    pipeline_version: str
    worker_id: str
    batch_size: int
    stale_lock_minutes: int
    download_max_bytes: int
    archive_max_files: int
    archive_max_selected_files: int
    archive_max_uncompressed_bytes: int
    archive_max_ratio: int
    max_pages_per_document: int
    max_text_chars_per_document: int
    native_text_min_chars: int
    ocr_max_pages: int
    ocr_dpi: int
    chunk_target_tokens: int
    chunk_overlap_tokens: int
    embedding_batch_size: int
    embedding_dimensions: int
    embedding_deployment: str
    chat_deployment: str
    chat_summaries: bool
    openai_endpoint: str
    openai_api_key: str
    openai_api_version: str
    blob_account_url: str
    blob_container: str
    blob_connection_string: str | None
    db_host: str
    db_user: str
    db_password: str
    db_name: str
    db_port: int
    db_sslmode: str

    @classmethod
    def from_env(cls) -> "Settings":
        worker_id = os.getenv("AI_WORKER_ID", "").strip()
        if not worker_id:
            worker_id = f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
        return cls(
            pipeline_version=os.getenv("AI_PIPELINE_VERSION", "pilot-v1").strip(),
            worker_id=worker_id[:120],
            batch_size=_integer("AI_JOB_BATCH_SIZE", 10, 1, 100),
            stale_lock_minutes=_integer("AI_STALE_LOCK_MINUTES", 90, 10, 1440),
            download_max_bytes=_integer(
                "AI_DOWNLOAD_MAX_FILE_BYTES", 50 * 1024 * 1024, 1024, 100 * 1024 * 1024
            ),
            archive_max_files=_integer("AI_ARCHIVE_MAX_FILES", 30, 1, 200),
            archive_max_selected_files=_integer("AI_ARCHIVE_MAX_SELECTED_FILES", 6, 1, 20),
            archive_max_uncompressed_bytes=_integer(
                "AI_ARCHIVE_MAX_UNCOMPRESSED_BYTES", 150 * 1024 * 1024, 1024, 500 * 1024 * 1024
            ),
            archive_max_ratio=_integer("AI_ARCHIVE_MAX_RATIO", 80, 2, 500),
            max_pages_per_document=_integer("AI_MAX_PAGES_PER_DOCUMENT", 300, 1, 1000),
            max_text_chars_per_document=_integer(
                "AI_MAX_TEXT_CHARS_PER_DOCUMENT", 2_000_000, 10_000, 10_000_000
            ),
            native_text_min_chars=_integer("AI_NATIVE_TEXT_MIN_CHARS_PER_PAGE", 80, 10, 2000),
            ocr_max_pages=_integer("AI_OCR_MAX_PAGES_PER_DOCUMENT", 60, 0, 500),
            ocr_dpi=_integer("AI_OCR_DPI", 220, 120, 400),
            chunk_target_tokens=_integer("AI_CHUNK_TARGET_TOKENS", 850, 200, 2000),
            chunk_overlap_tokens=_integer("AI_CHUNK_OVERLAP_TOKENS", 100, 0, 500),
            embedding_batch_size=_integer("AI_EMBED_BATCH_SIZE", 32, 1, 128),
            embedding_dimensions=_integer("AI_EMBEDDING_DIMENSIONS", 1536, 256, 3072),
            embedding_deployment=os.getenv(
                "OPENAI_DEPLOYMENT",
                os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"),
            ).strip(),
            chat_deployment=os.getenv(
                "CHAT_DEPLOYMENT",
                os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini"),
            ).strip(),
            chat_summaries=_boolean("AI_CHAT_SUMMARIES", True),
            openai_endpoint=_required("OPENAI_API_BASE", "AZURE_OPENAI_ENDPOINT"),
            openai_api_key=_required("OPENAI_API_KEY", "AZURE_OPENAI_API_KEY"),
            openai_api_version=os.getenv("OPENAI_API_VERSION", "2024-12-01-preview").strip(),
            blob_account_url=_required("AI_BLOB_ACCOUNT_URL"),
            blob_container=os.getenv("AI_BLOB_CONTAINER", "licigob-ai-documents").strip(),
            blob_connection_string=os.getenv("AI_BLOB_CONNECTION_STRING") or None,
            db_host=_required("DB_HOST"),
            db_user=_required("DB_USER"),
            db_password=_required("DB_PASSWORD"),
            db_name=_required("DB_NAME"),
            db_port=_integer("DB_PORT", 5432, 1, 65535),
            db_sslmode=os.getenv("DB_SSLMODE", "require").strip(),
        )

    def db_config(self) -> dict[str, Any]:
        return {
            "host": self.db_host,
            "user": self.db_user,
            "password": self.db_password,
            "dbname": self.db_name,
            "port": self.db_port,
            "sslmode": self.db_sslmode,
            "connect_timeout": 20,
            "application_name": "licigob-ai-ingestion",
        }

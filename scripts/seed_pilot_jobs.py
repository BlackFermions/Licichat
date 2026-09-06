"""Preview or enqueue AI document jobs for a bounded tender interval."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Any

import psycopg2
from psycopg2.extras import Json, RealDictCursor


DEFAULT_PIPELINE_VERSION = os.getenv("AI_PIPELINE_VERSION", "pilot-v1").strip()

SELECTION_SQL = """
SELECT
    t.id AS tender_id,
    t.title AS tender_title,
    base_doc.id AS base_id,
    base_doc.title AS base_title,
    base_doc.url AS base_url,
    base_doc.format AS base_format,
    base_doc.date_published AS base_published_at,
    award_doc.id AS award_id,
    award_doc.title AS award_title,
    award_doc.url AS award_url,
    award_doc.format AS award_format,
    award_doc.date_published AS award_published_at
FROM tenders t
JOIN LATERAL (
    SELECT d.id, d.title, d.url, d.format, d.date_published
    FROM documents d
    WHERE d.tender_id = t.id
      AND d.award_id IS NULL
      AND d.contract_id IS NULL
      AND d.document_type = 'biddingDocuments'
      AND (
          d.title ILIKE '%%Bases Integradas%%'
          OR d.title ILIKE '%%Bases Administrativas%%'
      )
      AND d.url IS NOT NULL
      AND BTRIM(d.url) <> ''
    ORDER BY
        CASE
            WHEN d.title ILIKE '%%Bases Integradas%%' THEN 0
            ELSE 1
        END,
        d.date_published DESC NULLS LAST,
        d.id DESC
    LIMIT 1
) base_doc ON TRUE
LEFT JOIN LATERAL (
    SELECT d.id, d.title, d.url, d.format, d.date_published
    FROM documents d
    WHERE d.tender_id = t.id
      AND d.document_type = 'awardNotice'
      AND (
          d.title ILIKE '%%Otorgamiento de Buena Pro%%'
          OR d.title ILIKE '%%Buena Pro%%'
      )
      AND d.url IS NOT NULL
      AND BTRIM(d.url) <> ''
    ORDER BY d.date_published DESC NULLS LAST, d.id DESC
    LIMIT 1
) award_doc ON TRUE
WHERE t.first_seen_at >= %s
  AND t.first_seen_at < %s
ORDER BY t.first_seen_at, t.id
"""

INSERT_SQL = """
INSERT INTO ai_ingestion_jobs (
    tender_id,
    source_signature,
    selected_documents,
    document_count,
    reason,
    priority,
    status,
    available_at,
    pipeline_version
)
VALUES (%s, %s, %s, %s, 'pilot_seed', %s, 'pending', 'infinity'::timestamptz, %s)
ON CONFLICT (tender_id, source_signature, pipeline_version) DO NOTHING
RETURNING id
"""


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("La fecha debe usar YYYY-MM-DD") from exc


def arguments() -> argparse.Namespace:
    today_utc = datetime.now(timezone.utc).date()
    parser = argparse.ArgumentParser(
        description="Previsualiza trabajos documentales; solo inserta con --apply."
    )
    parser.add_argument("--start", type=parse_date, default=today_utc - timedelta(days=7))
    parser.add_argument("--end", type=parse_date, default=today_utc)
    parser.add_argument("--limit", type=int, default=0, help="0 procesa todo el intervalo")
    parser.add_argument("--priority", type=int, default=50)
    parser.add_argument("--pipeline-version", default=DEFAULT_PIPELINE_VERSION)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def connection_config() -> dict[str, Any]:
    required = ("DB_HOST", "DB_USER", "DB_PASSWORD", "DB_NAME")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Faltan variables de entorno: {', '.join(missing)}")
    return {
        "host": os.environ["DB_HOST"],
        "user": os.environ["DB_USER"],
        "password": os.environ["DB_PASSWORD"],
        "dbname": os.environ["DB_NAME"],
        "port": int(os.getenv("DB_PORT", "5432")),
        "sslmode": os.getenv("DB_SSLMODE", "require"),
        "connect_timeout": 15,
    }


def serialized(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def selected_documents(row: dict[str, Any]) -> list[dict[str, Any]]:
    documents = [
        {
            "role": "bases",
            "source_document_id": row["base_id"],
            "title": row["base_title"],
            "url": row["base_url"],
            "format": row["base_format"],
            "published_at": serialized(row["base_published_at"]),
        }
    ]
    if row.get("award_id"):
        documents.append(
            {
                "role": "buena_pro",
                "source_document_id": row["award_id"],
                "title": row["award_title"],
                "url": row["award_url"],
                "format": row["award_format"],
                "published_at": serialized(row["award_published_at"]),
            }
        )
    return documents


def signature(documents: list[dict[str, Any]]) -> str:
    canonical = json.dumps(documents, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def main() -> int:
    args = arguments()
    if args.start >= args.end:
        print("--start debe ser anterior a --end", file=sys.stderr)
        return 2
    if args.limit < 0:
        print("--limit no puede ser negativo", file=sys.stderr)
        return 2
    if not 0 <= args.priority <= 100:
        print("--priority debe estar entre 0 y 100", file=sys.stderr)
        return 2

    conn = psycopg2.connect(**connection_config())
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            sql = SELECTION_SQL
            params: list[Any] = [args.start, args.end]
            if args.limit:
                sql += "\nLIMIT %s"
                params.append(args.limit)
            cursor.execute(sql, params)
            rows = cursor.fetchall()

        with_award = sum(1 for row in rows if row.get("award_id"))
        print(f"Intervalo UTC: {args.start} <= first_seen_at < {args.end}")
        print(f"Licitaciones seleccionadas: {len(rows)}")
        print(f"Con documento de Buena Pro: {with_award}")
        for row in rows[:5]:
            base_kind = "Integradas" if "integradas" in (row["base_title"] or "").lower() else "Administrativas"
            award_label = " + Buena Pro" if row.get("award_id") else ""
            print(f"  {row['tender_id']}: Bases {base_kind}{award_label}")

        if not args.apply:
            conn.rollback()
            print("Dry-run: no se insertaron trabajos. Usa --apply para confirmar.")
            return 0

        inserted = 0
        with conn.cursor() as cursor:
            cursor.execute("SELECT to_regclass('public.ai_ingestion_jobs')")
            if cursor.fetchone()[0] is None:
                raise RuntimeError("Ejecuta migrations/001_ai_document_pilot.sql antes de --apply")

            for row in rows:
                documents = selected_documents(row)
                cursor.execute(
                    INSERT_SQL,
                    (
                        row["tender_id"],
                        signature(documents),
                        Json(documents),
                        len(documents),
                        args.priority,
                        args.pipeline_version,
                    ),
                )
                if cursor.fetchone():
                    inserted += 1
        conn.commit()
        print(f"Trabajos nuevos insertados: {inserted}")
        print(f"Omitidos por idempotencia: {len(rows) - inserted}")
        return 0
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())

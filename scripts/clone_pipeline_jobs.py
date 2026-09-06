"""Clone the latest ready pilot jobs for a rolling pipeline upgrade."""

from __future__ import annotations

import argparse
import re

import psycopg2
from psycopg2.extras import Json, RealDictCursor

try:
    from scripts.seed_pilot_jobs import connection_config
except ModuleNotFoundError:  # Direct execution: python scripts/clone_pipeline_jobs.py
    from seed_pilot_jobs import connection_config


VERSION_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,40}$")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepara una version nueva desde trabajos listos.")
    parser.add_argument("--from-version", required=True)
    parser.add_argument("--to-version", required=True)
    parser.add_argument("--priority", type=int, default=80)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not VERSION_PATTERN.fullmatch(args.from_version) or not VERSION_PATTERN.fullmatch(args.to_version):
        raise RuntimeError("Las versiones solo admiten letras, numeros, punto, guion y guion bajo.")
    if args.from_version == args.to_version:
        raise RuntimeError("Las versiones de origen y destino deben ser diferentes.")
    if not 0 <= args.priority <= 100:
        raise RuntimeError("--priority debe estar entre 0 y 100")

    conn = psycopg2.connect(**connection_config())
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """SELECT DISTINCT ON (tender_id)
                          tender_id, source_signature, selected_documents, document_count
                   FROM ai_ingestion_jobs
                   WHERE pipeline_version = %s AND status = 'ready'
                   ORDER BY tender_id, completed_at DESC NULLS LAST, id DESC""",
                (args.from_version,),
            )
            jobs = list(cursor.fetchall())
        print(f"Trabajos listos para clonar: {len(jobs)}")
        if not args.apply:
            conn.rollback()
            print("Dry-run: no se insertaron trabajos. Usa --apply para confirmar.")
            return 0

        inserted = 0
        with conn.cursor() as cursor:
            for job in jobs:
                cursor.execute(
                    """INSERT INTO ai_ingestion_jobs (
                           tender_id, source_signature, selected_documents, document_count,
                           reason, priority, status, available_at, pipeline_version
                       ) VALUES (%s, %s, %s, %s, 'pipeline_upgrade', %s, 'pending',
                                 'infinity'::timestamptz, %s)
                       ON CONFLICT (tender_id, source_signature, pipeline_version) DO NOTHING
                       RETURNING id""",
                    (
                        job["tender_id"], job["source_signature"], Json(job["selected_documents"]),
                        job["document_count"], args.priority, args.to_version,
                    ),
                )
                inserted += int(cursor.fetchone() is not None)
        conn.commit()
        print(f"Trabajos insertados: {inserted}; ya existentes: {len(jobs) - inserted}")
        return 0
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())

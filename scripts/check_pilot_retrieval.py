"""Read-only semantic retrieval smoke check against the pilot vectors."""

from __future__ import annotations

import argparse
import json
import os

import psycopg2
import requests
from psycopg2.extras import RealDictCursor

from scripts.seed_pilot_jobs import connection_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Consulta fragmentos del piloto sin generar una respuesta.")
    parser.add_argument("query")
    parser.add_argument("--limit", type=int, choices=range(1, 11), default=3)
    args = parser.parse_args()
    endpoint = os.environ["OPENAI_API_BASE"].rstrip("/")
    deployment = os.getenv("OPENAI_DEPLOYMENT", "text-embedding-3-small")
    response = requests.post(
        f"{endpoint}/openai/deployments/{deployment}/embeddings",
        params={"api-version": os.getenv("OPENAI_API_VERSION", "2024-12-01-preview")},
        headers={"api-key": os.environ["OPENAI_API_KEY"]},
        json={"input": args.query, "dimensions": 1536},
        timeout=(15, 60),
    )
    if response.status_code != 200:
        print(json.dumps({"error": "embedding_failed", "http_status": response.status_code}))
        return 1
    vector = response.json()["data"][0]["embedding"]
    literal = "[" + ",".join(str(float(value)) for value in vector) + "]"
    conn = psycopg2.connect(**connection_config())
    try:
        conn.set_session(readonly=True)
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """SELECT c.tender_id, a.source_title AS document, c.page_start, c.page_end,
                          left(c.content, 650) AS excerpt,
                          round((1 - (c.embedding <=> %s::vector))::numeric, 4) AS similarity,
                          a.extraction_coverage->'partial' AS partial_document
                   FROM ai_tender_chunks c
                   JOIN ai_document_assets a ON a.id = c.asset_id
                   WHERE c.pipeline_version = 'pilot-v1' AND a.status = 'ready'
                   ORDER BY c.embedding <=> %s::vector LIMIT %s""",
                (literal, literal, args.limit),
            )
            print(json.dumps([dict(row) for row in cursor.fetchall()], ensure_ascii=True, default=str, indent=2))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

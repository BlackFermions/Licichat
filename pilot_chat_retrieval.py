"""Read-only retrieval of prepared pilot documents, isolated by tender and version."""

import logging
import math
import os
import unicodedata

import psycopg2
from psycopg2.extras import RealDictCursor

from lite_rag_engine import DB_CONFIG, MAX_CONTEXT_CHARS, _evidence_hint

logger = logging.getLogger("licigob-ai-lite.pilot")
PIPELINE_VERSION = "pilot-v1"
AWARD_MARKERS = (
    "buena pro", "otorgamiento", "acta de apertura", "acta de evaluacion",
    "documento de adjudicacion", "resultado del procedimiento", "procedimiento desierto",
    "ganador", "ganadores", "quien gano", "adjudicado", "adjudicacion",
)
OBJECTIVE_MARKERS = (
    "objetivo", "objeto de la licitacion", "objeto de la contratacion",
    "objeto de la convocatoria", "de que trata", "que compra la entidad",
    "que contrata la entidad", "que necesita la entidad", "que requiere la entidad",
    "que pide la entidad", "que pide el gobierno", "que se pide en la licitacion",
    "que se pide de la licitacion", "que se pide para la licitacion",
)
SUPPLIER_MARKERS = (
    "convocad", "concursant", "postor", "proveedor", "participante", "postulante", "ofertante",
)


def enabled():
    return os.getenv("LITE_DOCUMENT_PILOT_ENABLED", "false").lower() == "true"


def _connection():
    conn = psycopg2.connect(
        **DB_CONFIG, connect_timeout=5, options="-c statement_timeout=5000"
    )
    conn.set_session(readonly=True)
    return conn


def coverage_message(status):
    warnings = []
    if status.get("partial"):
        warnings.append(
            "Lectura parcial: algunas paginas no pudieron analizarse. "
            "La respuesta se basa solo en los fragmentos disponibles."
        )
    return " ".join(warnings)


def award_document_question(question):
    normalized = unicodedata.normalize("NFKD", str(question or "").lower())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    return any(marker in normalized for marker in AWARD_MARKERS)


def _question_has_any(question, markers):
    normalized = unicodedata.normalize("NFKD", str(question or "").lower())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    return any(marker in normalized for marker in markers)


# Only a completed job with current source URLs can advertise prepared material.
ASSETS_SQL = """
    SELECT a.id, a.source_title, a.document_role, a.page_count, a.text_char_count,
           a.extraction_coverage, a.detected_mime_type
    FROM ai_document_assets a
    JOIN ai_ingestion_jobs j ON j.id = a.job_id
    JOIN documents d ON d.id = a.source_document_id AND d.tender_id = a.tender_id
    WHERE a.tender_id = %s AND a.pipeline_version = %s AND a.status = 'ready'
      AND j.tender_id = a.tender_id AND j.pipeline_version = a.pipeline_version
      AND j.status = 'ready' AND d.url = a.source_url
      AND NOT EXISTS (
          SELECT 1 FROM ai_document_assets other
          LEFT JOIN documents source ON source.id = other.source_document_id
          WHERE other.job_id = j.id
            AND (other.status <> 'ready' OR source.url IS DISTINCT FROM other.source_url)
      )
      AND EXISTS (
          SELECT 1 FROM ai_tender_chunks c WHERE c.asset_id = a.id
            AND c.tender_id = a.tender_id AND c.pipeline_version = a.pipeline_version
            AND c.embedding IS NOT NULL
      )
"""


def pilot_status(tender_id):
    if not enabled():
        return None
    conn = None
    try:
        conn = _connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(ASSETS_SQL, (str(tender_id), PIPELINE_VERSION))
            assets = list(cursor.fetchall())
        if not assets:
            return None
        roles = sorted({a["document_role"] for a in assets})
        status = {
            "ready": True,
            "source": "document_pilot",
            "documents": len(assets),
            "pages": sum(a["page_count"] or 0 for a in assets),
            "characters": sum(a["text_char_count"] or 0 for a in assets),
            "partial": any((a["extraction_coverage"] or {}).get("partial") for a in assets),
            "container_documents": any(a["detected_mime_type"] != "application/pdf" for a in assets),
            "document_roles": roles,
        }
        status["warning"] = coverage_message(status)
        return status
    except Exception as exc:
        logger.warning("pilot_status_unavailable error_type=%s", type(exc).__name__)
        return None
    finally:
        if conn:
            conn.close()


def format_context(rows):
    parts, references = [], []
    remaining = MAX_CONTEXT_CHARS
    for row in rows:
        title = str(row["source_title"] or "Documento")[:160].replace("\n", " ")
        start, end = row["page_start"], row["page_end"]
        page = str(start) if start == end else f"{start}-{end}"
        text = row["content"]
        header = f"[Documento: {title} | Paginas: {page} | Senal de clasificacion: {_evidence_hint(text)}]\n"
        available = remaining - len(header)
        if available < 200:
            break
        excerpt = text[:available]
        parts.append(header + excerpt)
        reference = {"document": title, "page": page}
        if reference not in references:
            references.append(reference)
        remaining -= len(header) + len(excerpt) + 2
    return "\n\n".join(parts), references


def retrieve_pilot(tender_id, question, client, history=None):
    status = pilot_status(tender_id)
    if not status:
        return None
    conn = None
    try:
        previous = [str(item.get("content", ""))[:400] for item in (history or [])[-4:]
                    if isinstance(item, dict) and item.get("role") == "user"]
        query = question[:2000]
        objective_question = _question_has_any(question, OBJECTIVE_MARKERS)
        supplier_question = _question_has_any(question, SUPPLIER_MARKERS)
        award_question = award_document_question(question)
        if objective_question:
            query = f"objeto de la convocatoria finalidad descripcion del bien servicio u obra {query}"
        elif supplier_question:
            query = f"requisitos del postor documentos para admision y presentacion de la oferta {query}"
        elif len(query.split()) < 10 and previous and not award_question:
            query = f"Consulta anterior: {previous[-1]}\nPregunta actual: {query}"
        embedding = client.with_options(timeout=20.0, max_retries=0).embeddings.create(
            model=os.getenv("PILOT_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"),
            input=query, dimensions=1536,
        ).data[0].embedding
        if len(embedding) != 1536 or not all(math.isfinite(float(v)) for v in embedding):
            raise ValueError("invalid_embedding")
        vector = "[" + ",".join(str(float(v)) for v in embedding) + "]"
        conn = _connection()
        role_filter = (
            "AND a.document_role = 'buena_pro'"
            if award_question
            else "AND a.document_role = 'bases'"
        )
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Materialize the tender subset before nearest-neighbour ranking. This
            # avoids filtering a global approximate index down to too few matches.
            cursor.execute(
                f"""WITH assets AS MATERIALIZED ({ASSETS_SQL}),
                    scoped AS MATERIALIZED (
                        SELECT c.content, c.page_start, c.page_end, c.embedding, a.source_title
                        FROM ai_tender_chunks c JOIN assets a ON a.id = c.asset_id
                        WHERE c.tender_id = %s AND c.pipeline_version = %s
                          AND c.embedding IS NOT NULL
                          {role_filter}
                    )
                    SELECT content, page_start, page_end, source_title FROM scoped
                    ORDER BY embedding <=> %s::vector LIMIT 6""",
                (str(tender_id), PIPELINE_VERSION, str(tender_id), PIPELINE_VERSION, vector),
            )
            context, references = format_context(cursor.fetchall())
        if not context:
            return None
        return status, context, references
    except Exception as exc:
        logger.warning("pilot_retrieval_unavailable error_type=%s", type(exc).__name__)
        return None
    finally:
        if conn:
            conn.close()

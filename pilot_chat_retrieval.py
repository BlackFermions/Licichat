"""Read-only retrieval of prepared pilot documents, isolated by tender and version."""

import logging
import math
import os
import re
import unicodedata

import psycopg2
from psycopg2.extras import RealDictCursor

from lite_rag_engine import DB_CONFIG, MAX_CONTEXT_CHARS, _evidence_hint

logger = logging.getLogger("licigob-ai-lite.pilot")
PIPELINE_VERSION = os.getenv("AI_PIPELINE_VERSION", "pilot-v1").strip()


def _pipeline_versions():
    configured = os.getenv("AI_PILOT_PIPELINE_VERSIONS", PIPELINE_VERSION)
    versions = [value.strip() for value in configured.split(",") if value.strip()]
    return list(dict.fromkeys(versions)) or [PIPELINE_VERSION]
AWARD_MARKERS = (
    "buena pro", "otorgamiento", "acta de apertura", "acta de evaluacion",
    "documento de adjudicacion", "resultado del procedimiento", "procedimiento desierto",
    "ganador", "ganadores", "quien gano", "quien gano", "adjudicado", "adjudicacion",
    "ganaron", "gano", "perdio", "perdieron", "descalific", "no admitid",
    "declarado desierto", "declarada desierta", "ofertas presentadas", "puntaje obtenido",
    "ofertas recibidas", "cuantas ofertas", "cuantos postores", "postores se presentaron",
    "monto adjudicado", "resultado final", "orden de prelacion",
)
AWARD_FOLLOW_UP_MARKERS = (
    "ese resultado", "este resultado", "por que", "motivo", "motivos", "razon", "razones",
    "cuantas ofertas", "cuantos postores", "los otros", "cada postor", "que documento",
    "cual documento", "en que pagina", "donde figura", "que sustenta", "sustento",
)
BASES_MARKERS = (
    "bases", "requisito", "especificacion", "experiencia", "documentos obligatorios",
    "plazo", "cronograma", "garantia", "penalidad", "personal", "equipamiento",
    "factor de evaluacion", "presentar la oferta", "perfeccionar el contrato",
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
MAX_EXCERPT_CHARS = 4500


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


def _normalize(value):
    normalized = unicodedata.normalize("NFKD", str(value or "").lower())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _history_mentions_award(history):
    recent = " ".join(
        str(item.get("content") or "")[:800]
        for item in (history or [])[-6:]
        if isinstance(item, dict) and item.get("role") in {"user", "assistant"}
    )
    return _question_has_any(recent, AWARD_MARKERS)


def infer_document_role(question, history=None):
    """Choose the evidence family while preserving the topic of short follow-ups."""
    if _question_has_any(question, AWARD_MARKERS):
        return "buena_pro"
    if _question_has_any(question, BASES_MARKERS):
        return "bases"
    if _question_has_any(question, AWARD_FOLLOW_UP_MARKERS) and _history_mentions_award(history):
        return "buena_pro"
    return "bases"


def award_document_question(question, history=None):
    return infer_document_role(question, history) == "buena_pro"


def _question_has_any(question, markers):
    normalized = _normalize(question)
    return any(marker in normalized for marker in markers)


def _retrieval_query(question, history, role):
    query = str(question or "")[:2000]
    expansions = []
    if _question_has_any(question, OBJECTIVE_MARKERS):
        expansions.append("objeto de la convocatoria finalidad descripcion del bien servicio u obra")
    if _question_has_any(question, SUPPLIER_MARKERS):
        expansions.append("requisitos del postor documentos para admision presentacion de la oferta")
    if _question_has_any(
        question,
        ("presupuesto", "valor estimado", "valor referencial", "cuantia", "monto", "importe", "cuanto vale"),
    ):
        expansions.append("valor estimado valor referencial cuantia de la contratacion monto presupuesto soles")
    if _question_has_any(question, ("experiencia", "facturacion", "mype", "microempresa", "pequena empresa")):
        expansions.append(
            "experiencia del postor monto facturado requisito general excepcion MYPE consorcio servicios similares"
        )
    if role == "buena_pro":
        expansions.append(
            "acta resultado evaluacion admision calificacion postor oferta puntaje adjudicacion desierto"
        )
        if _question_has_any(question, ("por que", "motivo", "razon", "perdio", "descalific", "no admitid", "otros")):
            expansions.append(
                "motivo no admitido descalificado incumplimiento evidencia oferta economica resultado por postor"
            )

    previous = [
        str(item.get("content", ""))[:500]
        for item in (history or [])[-6:]
        if isinstance(item, dict) and item.get("role") in {"user", "assistant"}
    ]
    if len(query.split()) < 10 and previous:
        query = f"Contexto inmediato: {previous[-1]}\nPregunta actual: {query}"
    return " ".join(expansions + [query]).strip()


def _lexical_terms(question, role):
    normalized = _normalize(question)
    tokens = re.findall(r"[a-z0-9]{4,}", normalized)
    stop = {"cual", "cuales", "como", "esta", "este", "estos", "para", "sobre", "documento", "licitacion"}
    terms = [token for token in tokens if token not in stop]
    if _question_has_any(question, ("presupuesto", "valor estimado", "valor referencial", "cuantia", "monto")):
        terms.extend(["valor estimado", "valor referencial", "cuantia", "monto"])
    if role == "buena_pro":
        terms.extend(["resultado", "descalificado", "no admitido", "adjudicado"])
    return list(dict.fromkeys(terms))[:16] or ["licitacion"]


def _priority_phrases(question, role):
    phrases = []
    if _question_has_any(question, ("presupuesto", "valor estimado", "valor referencial", "cuantia", "monto")):
        phrases.extend(["valor estimado", "valor referencial", "cuant", "monto adjudicado"])
    if _question_has_any(question, ("experiencia", "facturacion", "mype", "microempresa", "pequena empresa")):
        phrases.extend(["monto facturado acumulado equivalente", "micro y peque", "en el caso de consorcios"])
    if _question_has_any(question, ("documentos obligatorios", "que documentos", "documentacion obligatoria")):
        phrases.extend(["documentos para la admisi", "documentaci", "oferta econ"])
    if role == "buena_pro":
        phrases.extend(["orden de prelaci", "resultado final", "detalle y justificaci", "ofertas fueron admitidas"])
    return list(dict.fromkeys(phrases))[:8]


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
            assets = []
            selected_version = None
            for pipeline_version in _pipeline_versions():
                cursor.execute(ASSETS_SQL, (str(tender_id), pipeline_version))
                assets = list(cursor.fetchall())
                if assets:
                    selected_version = pipeline_version
                    break
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
            "pipeline_version": selected_version,
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
        container_title = str(row["source_title"] or "Documento")[:160].replace("\n", " ")
        text = str(row["content"] or "")
        member_match = re.match(r"\[Archivo interno: ([^\]\r\n]+)\]\s*", text)
        member_title = member_match.group(1).strip()[:160] if member_match else ""
        title = member_title or container_title
        start, end = row["page_start"], row["page_end"]
        page = str(start) if start == end else f"{start}-{end}"
        container_note = (
            f" | Contenedor: {container_title}" if member_title and member_title != container_title else ""
        )
        header = (
            f"[Documento: {title}{container_note} | Paginas: {page} | "
            f"Senal de clasificacion: {_evidence_hint(text)}]\n"
        )
        available = remaining - len(header)
        if available < 200:
            break
        excerpt = text[: min(available, MAX_EXCERPT_CHARS)]
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
        document_role = infer_document_role(question, history)
        pipeline_version = status.get("pipeline_version") or PIPELINE_VERSION
        query = _retrieval_query(question, history, document_role)
        lexical_terms = _lexical_terms(question, document_role)
        priority_phrases = _priority_phrases(question, document_role)
        embedding = client.with_options(timeout=20.0, max_retries=0).embeddings.create(
            model=os.getenv("PILOT_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"),
            input=query, dimensions=1536,
        ).data[0].embedding
        if len(embedding) != 1536 or not all(math.isfinite(float(v)) for v in embedding):
            raise ValueError("invalid_embedding")
        vector = "[" + ",".join(str(float(v)) for v in embedding) + "]"
        conn = _connection()
        role_filter = "AND a.document_role = %s"
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Materialize the tender subset before nearest-neighbour ranking. This
            # avoids filtering a global approximate index down to too few matches.
            cursor.execute(
                f"""WITH assets AS MATERIALIZED ({ASSETS_SQL}),
                    scoped AS MATERIALIZED (
                        SELECT c.content, c.page_start, c.page_end, c.chunk_index,
                               c.embedding, a.source_title,
                               (SELECT COUNT(*) FROM unnest(%s::text[]) term
                                WHERE LOWER(c.content) LIKE '%%' || term || '%%') AS lexical_hits
                        FROM ai_tender_chunks c JOIN assets a ON a.id = c.asset_id
                        WHERE c.tender_id = %s AND c.pipeline_version = %s
                          AND c.embedding IS NOT NULL
                          {role_filter}
                    )
                    SELECT content, page_start, page_end, source_title FROM scoped
                    ORDER BY (embedding <=> %s::vector) - LEAST(lexical_hits, 4) * 0.06,
                             chunk_index
                    LIMIT 8""",
                (
                    str(tender_id), pipeline_version, lexical_terms,
                    str(tender_id), pipeline_version, document_role, vector,
                ),
            )
            ranked_rows = list(cursor.fetchall())
            priority_rows = []
            if priority_phrases:
                cursor.execute(
                    f"""WITH assets AS MATERIALIZED ({ASSETS_SQL}),
                        scoped AS MATERIALIZED (
                            SELECT c.content, c.page_start, c.page_end, c.chunk_index,
                                   a.id AS asset_id, a.source_title
                            FROM ai_tender_chunks c JOIN assets a ON a.id = c.asset_id
                            WHERE c.tender_id = %s AND c.pipeline_version = %s
                              AND c.embedding IS NOT NULL
                              {role_filter}
                        ), phrase_matches AS (
                            SELECT DISTINCT ON (phrase)
                                   s.content, s.page_start, s.page_end, s.source_title,
                                   s.asset_id, s.chunk_index, phrase
                            FROM unnest(%s::text[]) phrase
                            JOIN scoped s ON LOWER(s.content) LIKE '%%' || phrase || '%%'
                            ORDER BY phrase, s.chunk_index
                        )
                        SELECT content, page_start, page_end, source_title
                        FROM phrase_matches ORDER BY phrase LIMIT 8""",
                    (
                        str(tender_id), pipeline_version, str(tender_id),
                        pipeline_version, document_role, priority_phrases,
                    ),
                )
                priority_rows = list(cursor.fetchall())

            rows = []
            seen = set()
            for row in priority_rows + ranked_rows:
                key = (row.get("source_title"), row.get("page_start"), row.get("page_end"), row.get("content"))
                if key not in seen:
                    seen.add(key)
                    rows.append(row)
            context, references = format_context(rows)
        if not context:
            return None
        result_status = dict(status)
        result_status["selected_document_role"] = document_role
        return result_status, context, references
    except Exception as exc:
        logger.warning("pilot_retrieval_unavailable error_type=%s", type(exc).__name__)
        return None
    finally:
        if conn:
            conn.close()

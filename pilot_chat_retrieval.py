"""Read-only retrieval of prepared pilot documents, isolated by tender and version."""

import logging
import hashlib
import math
import os
import re
import threading
import time
import unicodedata

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool

from lite_rag_engine import DB_CONFIG, MAX_CONTEXT_CHARS, _evidence_hint
from evidence_selection import assemble_context, exhaustive_question, select_candidates

logger = logging.getLogger("licigob-ai-lite.pilot")
PIPELINE_VERSION = os.getenv("AI_PIPELINE_VERSION", "pilot-v2").strip()


def _pipeline_versions():
    configured = os.getenv("AI_PILOT_PIPELINE_VERSIONS", PIPELINE_VERSION)
    versions = [value.strip() for value in configured.split(",") if value.strip()]
    return list(dict.fromkeys(versions)) or [PIPELINE_VERSION]
AWARD_MARKERS = (
    "buena pro", "otorgamiento", "acta de apertura", "acta de evaluacion",
    "documento de adjudicacion", "resultado del procedimiento", "procedimiento desierto",
    "ganador", "ganadores", "quien gano", "adjudicatario", "adjudicado", "adjudicacion",
    "ganaron", "gano", "perdio", "perdieron", "descalific", "no admitid",
    "declarado desierto", "declarada desierta", "ofertas presentadas", "puntaje obtenido",
    "ofertas recibidas", "cuantas ofertas", "cuantos postores", "postores se presentaron",
    "monto adjudicado", "resultado final", "orden de prelacion",
    "ya termino", "ya finalizo", "ya concluyo", "ya acabo", "se cerro",
    "esta cerrada", "esta finalizada", "esta concluida", "sigue vigente",
    "estado actual del proceso", "estado de la licitacion",
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
_EMBEDDING_CACHE = {}
_EMBEDDING_LOCK = threading.Lock()
_READ_POOL = None
_POOL_LOCK = threading.Lock()


class _ReadLease:
    def __init__(self, connection, pool):
        self.connection = connection
        self.pool = pool
        self.released = False

    def cursor(self, *args, **kwargs):
        return self.connection.cursor(*args, **kwargs)

    def close(self):
        if self.released:
            return
        self.released = True
        broken = bool(self.connection.closed)
        if not broken:
            try:
                self.connection.rollback()
            except psycopg2.Error:
                broken = True
        self.pool.putconn(self.connection, close=broken)


class PilotUnavailable(RuntimeError):
    """Prepared material must not silently fall through to a fresh download."""

    def __init__(self, code="PILOT_TEMPORARILY_UNAVAILABLE"):
        self.code = code
        super().__init__(code)


def _query_embedding(client, query):
    deployment = os.getenv("PILOT_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
    key = hashlib.sha256(f"{client.base_url}|{deployment}|1536|{query}".encode()).hexdigest()
    now = time.monotonic()
    with _EMBEDDING_LOCK:
        cached = _EMBEDDING_CACHE.get(key)
        if cached and cached[0] > now:
            return cached[1], True
    vector = client.with_options(timeout=20.0, max_retries=0).embeddings.create(
        model=deployment, input=query, dimensions=1536,
    ).data[0].embedding
    if len(vector) != 1536 or not all(math.isfinite(float(v)) for v in vector):
        raise ValueError("invalid_embedding")
    with _EMBEDDING_LOCK:
        expired = [key for key, item in _EMBEDDING_CACHE.items() if item[0] <= now]
        for old in expired:
            _EMBEDDING_CACHE.pop(old, None)
        if len(_EMBEDDING_CACHE) >= 128:
            _EMBEDDING_CACHE.pop(next(iter(_EMBEDDING_CACHE)))
        _EMBEDDING_CACHE[key] = (now + 300, list(vector))
    return vector, False


def enabled():
    return os.getenv("LITE_DOCUMENT_PILOT_ENABLED", "false").lower() == "true"


def _connection():
    global _READ_POOL
    with _POOL_LOCK:
        if _READ_POOL is None:
            _READ_POOL = ThreadedConnectionPool(
                1, 4, **DB_CONFIG, connect_timeout=5,
                options="-c statement_timeout=5000 -c default_transaction_read_only=on",
            )
    conn = _READ_POOL.getconn()
    try:
        conn.set_session(readonly=True)
        return _ReadLease(conn, _READ_POOL)
    except Exception:
        _READ_POOL.putconn(conn, close=True)
        raise


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
        if isinstance(item, dict) and item.get("role") == "user"
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


def infer_document_roles(question, history=None):
    bases = BASES_MARKERS + ("vehiculo", "furgon", "camioneta", "unidades", "equipamiento", "experiencia")
    if _question_has_any(question, AWARD_MARKERS) and _question_has_any(question, bases):
        return ("bases", "buena_pro")
    return (infer_document_role(question, history),)


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
    if _question_has_any(question, ("vehiculo", "furgon", "camioneta", "unidades", "equipamiento")):
        expansions.append("equipamiento estrategico unidades de transporte furgones camioneta pick up carga util antiguedad")
    if role in {"buena_pro", "mixed"}:
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
        if isinstance(item, dict) and item.get("role") == "user"
    ]
    if len(query.split()) < 10 and previous:
        query = f"Contexto inmediato: {previous[-1]}\nPregunta actual: {query}"
    return " ".join(expansions + [query]).strip()


def _lexical_terms(question, role):
    normalized = _normalize(question)
    tokens = re.findall(r"[a-z0-9]{4,}", normalized)
    stop = {"cual", "cuales", "como", "esta", "este", "estos", "para", "sobre", "documento", "licitacion"}
    terms = [token for token in tokens if token not in stop]
    if _question_has_any(question, ("vehiculo", "furgon", "camioneta", "unidades", "equipamiento")):
        terms = ["furgones", "camioneta", "equipamiento", "unidades"] + terms
    if _question_has_any(question, ("presupuesto", "valor estimado", "valor referencial", "cuantia", "monto")):
        terms.extend(["valor estimado", "valor referencial", "cuantia", "monto"])
    if role in {"buena_pro", "mixed"}:
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
    if _question_has_any(question, ("admision", "admitan")):
        phrases.extend(["documentos para la admisi", "pacto de integridad", "vigencia de poder"])
    if _question_has_any(question, ("vehiculo", "furgon", "camioneta", "unidades", "equipamiento")):
        phrases.extend(["furgones de reparto", "camioneta pick", "equipamiento estrategico"])
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
      AND a.source_document_id = (
          SELECT latest.id FROM documents latest
          WHERE latest.tender_id = a.tender_id
            AND latest.url IS NOT NULL AND BTRIM(latest.url) <> ''
            AND (
                (a.document_role = 'bases' AND latest.document_type = 'biddingDocuments'
                 AND latest.award_id IS NULL AND latest.contract_id IS NULL
                 AND (latest.title ILIKE '%%Bases Integradas%%' OR latest.title ILIKE '%%Bases Administrativas%%'))
                OR (a.document_role = 'buena_pro' AND latest.document_type = 'awardNotice'
                    AND latest.title ILIKE '%%Buena Pro%%')
            )
          ORDER BY CASE WHEN latest.title ILIKE '%%Bases Integradas%%' THEN 0 ELSE 1 END,
                   latest.date_published DESC NULLS LAST, latest.id DESC LIMIT 1
      )
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


def pilot_status(tender_id, *, strict=False):
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
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1 FROM ai_document_assets WHERE tender_id=%s AND status='ready' AND pipeline_version=ANY(%s) LIMIT 1",
                               (str(tender_id), _pipeline_versions()))
                if cursor.fetchone():
                    return {"ready": False, "state": "stale", "source": "document_pilot"}
            return None
        roles = sorted({a["document_role"] for a in assets})
        status = {
            "ready": True,
            "state": "partial" if any((a["extraction_coverage"] or {}).get("partial") for a in assets) else "ready",
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
        if strict:
            raise PilotUnavailable() from exc
        return None
    finally:
        if conn:
            conn.close()


def format_context(rows):
    context, references, _, _ = assemble_context(rows, MAX_CONTEXT_CHARS, _evidence_hint)
    return context, references


def retrieve_pilot(tender_id, question, client, history=None):
    started = time.monotonic()
    status = pilot_status(tender_id, strict=True)
    if not status:
        return None
    if status.get("state") == "stale":
        raise PilotUnavailable("PILOT_STALE")
    conn = None
    try:
        roles = infer_document_roles(question, history)
        document_role = "mixed" if len(roles) > 1 else roles[0]
        pipeline_version = status.get("pipeline_version") or PIPELINE_VERSION
        query = _retrieval_query(question, history, document_role)
        lexical_terms = _lexical_terms(question, document_role)
        priority_phrases = _priority_phrases(question, document_role)
        embed_started = time.monotonic()
        embedding, cache_hit = _query_embedding(client, query)
        embed_ms = round((time.monotonic() - embed_started) * 1000)
        vector = "[" + ",".join(str(float(v)) for v in embedding) + "]"
        conn = _connection()
        sql_started = time.monotonic()
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Materialize the tender subset before nearest-neighbour ranking. This
            # avoids filtering a global approximate index down to too few matches.
            cursor.execute(
                f"""WITH assets AS MATERIALIZED ({ASSETS_SQL}),
                    scoped AS MATERIALIZED (
                        SELECT c.content, c.page_start, c.page_end, c.chunk_index, c.asset_id,
                               c.embedding, a.source_title, a.document_role,
                               (SELECT COUNT(*) FROM unnest(%s::text[]) term
                                WHERE translate(LOWER(c.content), 'áéíóúüñ', 'aeiouun') LIKE '%%' || term || '%%') AS lexical_hits
                        FROM ai_tender_chunks c JOIN assets a ON a.id = c.asset_id
                        WHERE c.tender_id = %s AND c.pipeline_version = %s
                          AND c.embedding IS NOT NULL
                          AND a.document_role = ANY(%s)
                    ), ranked AS (
                        SELECT *, embedding <=> %s::vector AS distance,
                            row_number() OVER (PARTITION BY document_role ORDER BY
                                (embedding <=> %s::vector) - LEAST(lexical_hits, 4) * 0.06,
                                chunk_index) AS role_rank
                        FROM scoped
                    )
                    SELECT content, page_start, page_end, source_title, asset_id,
                           chunk_index, document_role, distance
                    FROM ranked WHERE role_rank <= %s ORDER BY role_rank, document_role""",
                (
                    str(tender_id), pipeline_version, lexical_terms,
                    str(tender_id), pipeline_version, list(roles), vector, vector,
                    24 if exhaustive_question(question) else 16,
                ),
            )
            ranked_rows = list(cursor.fetchall())
            # Pull neighbours of the best candidates; never cross asset/version/ID.
            neighbours = {(int(row["asset_id"]), int(row["chunk_index"]) + delta)
                          for row in ranked_rows[:4] if row.get("asset_id") is not None
                          for delta in (-1, 1) if int(row["chunk_index"]) + delta >= 0}
            if neighbours:
                assets, indices = zip(*sorted(neighbours))
                cursor.execute(
                    """SELECT c.content, c.page_start, c.page_end, c.chunk_index,
                              c.asset_id, a.source_title, a.document_role, 2.0 AS distance
                       FROM unnest(%s::bigint[], %s::integer[]) wanted(asset_id, chunk_index)
                       JOIN ai_tender_chunks c USING (asset_id, chunk_index)
                       JOIN ai_document_assets a ON a.id=c.asset_id
                       WHERE c.tender_id=%s AND c.pipeline_version=%s AND a.status='ready'
                         AND a.tender_id=c.tender_id AND a.pipeline_version=c.pipeline_version""",
                    (list(assets), list(indices), str(tender_id), pipeline_version),
                )
                ranked_rows.extend(cursor.fetchall())
        rows = select_candidates(ranked_rows, lexical_terms, priority_phrases, roles)
        context, references, evidence, truncated = assemble_context(
            rows, MAX_CONTEXT_CHARS, _evidence_hint, lexical_terms)
        result_status = dict(status)
        result_status["selected_document_role"] = document_role
        result_status["selected_document_roles"] = list(roles)
        result_status["context_truncated"] = truncated
        result_status["exhaustive_requested"] = exhaustive_question(question)
        result_status["evidence"] = evidence
        result_status["timings"] = {"embedding_ms": embed_ms, "embedding_cache_hit": cache_hit,
                                    "sql_ms": round((time.monotonic()-sql_started)*1000),
                                    "total_ms": round((time.monotonic()-started)*1000)}
        if not context:
            result_status["state"] = "no_evidence"
        logger.info("pilot_retrieved tender=%s role=%s evidence=%s chars=%s timings=%s",
                    tender_id, document_role, len(evidence), len(context), result_status["timings"])
        return result_status, context, references
    except Exception as exc:
        logger.warning("pilot_retrieval_unavailable error_type=%s", type(exc).__name__)
        raise PilotUnavailable() from exc
    finally:
        if conn:
            conn.close()

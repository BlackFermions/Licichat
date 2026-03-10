# -*- coding: utf-8 -*-
"""
chat_engine.py — LiciGob Chat Engine (Producción)

Motor central del chatbot. Diseñado para:
- Múltiples usuarios simultáneos (Gunicorn + threads)
- pgvector en PostgreSQL (sin archivos FAISS)
- Azure OpenAI con conexión persistente (keep-alive)
- Caché de perfiles en memoria thread-safe
- Streaming con Flask

Arranque desarrollo:
  python chat_engine.py

Arranque producción:
  gunicorn -w 4 -b 0.0.0.0:5007 --timeout 120 --worker-class gthread --threads 4 chat_engine:app
"""

import os
import sys
import json
import time
import threading
import psycopg2
import httpx
from psycopg2.extras import RealDictCursor
from flask import Flask, Response, request
from flask_cors import CORS
from openai import AzureOpenAI
from dotenv import load_dotenv
from typing import List, Dict, Generator

load_dotenv()

# =================================================================
# PATHS
# =================================================================

CHATBOT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR  = os.path.dirname(CHATBOT_DIR)

print(f"\n{'='*60}")
print(f"📁 CHATBOT_DIR : {CHATBOT_DIR}")
print(f"📁 PARENT_DIR  : {PARENT_DIR}")

if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
    print(f"✅ PARENT_DIR en sys.path")

# =================================================================
# MATCH ENGINE (pgvector)
# =================================================================

try:
    import match_engine_pgvector as me
    MATCH_AVAILABLE = True
    print(f"✅ match_engine_pgvector importado")
except Exception as e:
    MATCH_AVAILABLE = False
    me = None
    print(f"❌ match_engine_pgvector: {e}")

print(f"{'='*60}\n")

from chat_busqueda      import get_system_prompt as prompt_busqueda
from chat_busqueda      import get_messages      as messages_busqueda
from chat_recomendacion import get_system_prompt as prompt_recomendacion
from chat_recomendacion import get_messages      as messages_recomendacion
from chat_general       import get_system_prompt as prompt_general
from chat_general       import get_messages      as messages_general
import rag_engine

# =================================================================
# AZURE OPENAI — pool de conexiones persistentes
# =================================================================

_http_client = httpx.Client(
    timeout=httpx.Timeout(120.0, connect=10.0),  # 120s para respuestas largas
    limits=httpx.Limits(
        max_keepalive_connections=20,
        max_connections=50,
        keepalive_expiry=30.0,
    ),
)
azure_client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    azure_endpoint=os.getenv("OPENAI_API_BASE"),
    api_version=os.getenv("OPENAI_API_VERSION", "2024-12-01-preview"),
    http_client=_http_client,
    max_retries=2,
)
CHAT_MODEL       = os.getenv("CHAT_DEPLOYMENT", "gpt-4o-mini")
EMBED_DEPLOYMENT = os.getenv("OPENAI_DEPLOYMENT", "text-embedding-3-small")

print(f"🤖 Chat     : {CHAT_MODEL}")
print(f"🧠 Embedding: {EMBED_DEPLOYMENT}")

# =================================================================
# DB CONFIG
# =================================================================

DB_CONFIG = {
    "host":     os.getenv("DB_HOST"),
    "user":     os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "port":     int(os.getenv("DB_PORT", "5432")),
    "sslmode":  os.getenv("DB_SSLMODE", "require"),
}

# =================================================================
# CACHÉ DE PERFILES — thread-safe
# =================================================================

_perfil_cache: dict = {}
_cache_lock = threading.Lock()

# =================================================================
# FLASK
# =================================================================

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": [
    "http://localhost:5173",
    "https://licigob.pe",
    "http://licigob.pe",
    "https://www.licigob.pe",
    "http://20.72.150.190",
    "http://licigob-frontend.eastus.azurecontainer.io",
]}}, supports_credentials=True)

# =================================================================
# PERFIL DESDE BD
# =================================================================

QUERY_PERFIL = """
    SELECT
        u.ciiu_codigo,
        e.company_name,
        e.user_description,
        e.primary_sectors        AS sectors,
        e.operation_regions      AS regions,
        e.contract_types,
        e.max_contract_value,
        e.rnp_specialties,
        e.strict_filters,
        c.descripcion            AS ciiu_descripcion
    FROM users u
    JOIN empresas_profiles e     ON e.user_id = u.id
    LEFT JOIN ciiu_actividades c ON c.codigo = u.ciiu_codigo
    WHERE u.id = %s;
"""


def get_profile_from_db(user_id: int) -> Dict | None:
    with _cache_lock:
        if user_id in _perfil_cache:
            print(f"   ⚡ Caché user_id={user_id}")
            return _perfil_cache[user_id]
    try:
        t    = time.time()
        conn = psycopg2.connect(**DB_CONFIG)
        cur  = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(QUERY_PERFIL, (user_id,))
        row  = cur.fetchone()
        conn.close()
        if not row:
            return None
        perfil = {
            "ciiu_codigo":        row.get("ciiu_codigo"),
            "ciiu_descripcion":   row.get("ciiu_descripcion", ""),
            "primarySectors":     row.get("sectors", ""),
            "operationRegions":   row.get("regions", ""),
            "contract_types":     row.get("contract_types", ""),
            "operation_regions":  row.get("regions", ""),
            "max_contract_value": float(row.get("max_contract_value") or 0),
            "strict_filters":     bool(row.get("strict_filters", False)),
            "company_name":       row.get("company_name"),
        }
        with _cache_lock:
            _perfil_cache[user_id] = perfil
        print(f"   ⏱️  BD perfil: {(time.time()-t)*1000:.0f}ms | {perfil.get('company_name')}")
        return perfil
    except Exception as e:
        print(f"   ❌ Error perfil: {e}")
        return None

# =================================================================
# DETECCIÓN DE MODO
# =================================================================

def detectar_modo(data: Dict) -> tuple:
    if data.get("mode") in ("busqueda", "recomendacion", "general"):
        return data["mode"], "explicito"
    documents      = data.get("documents", [])
    tender_context = data.get("tender_context", [])
    context        = data.get("context")          # ← ChatbotHelper usa "context"

    # ChatbotHelper (TenderDetailPopup) envía "context" como objeto único
    if context and isinstance(context, dict):
        return "busqueda", "inferido-context"
    if documents and len(documents) > 0:
        return "busqueda", "inferido"
    if tender_context and len(tender_context) >= 1:
        return "recomendacion", "inferido"
    return "general", "inferido"

# =================================================================
# Keywords que indican confirmación explícita de cargar las bases
KEYWORDS_CONFIRMA_RAG = {
    "sí carga", "si carga", "sí, carga", "si, carga",
    "carga las bases", "carga los documentos", "carga los docs",
    "analiza las bases", "analiza los documentos",
    "lee las bases", "lee los documentos",
    "sí", "si", "dale", "ok", "claro", "adelante",
    "cárgalos", "cargalos", "hazlo", "procede",
}

# Keywords que sugieren que el bot debe OFRECER cargar las bases
KEYWORDS_NECESITA_BASES = {
    "requisito", "requisitos", "base", "bases", "anexo", "anexos",
    "experiencia", "capital", "término", "términos", "terminos",
    "condición", "condiciones", "plazo presentación", "garantía",
    "garantia", "rnp", "habilitación", "habilitacion", "cronograma",
    "calificación", "calificacion", "mínimo", "minimo", "acreditar",
    "postor", "participar", "postular",
}


def usuario_confirma_rag(message: str) -> bool:
    """El usuario dijo explícitamente que quiere cargar las bases."""
    msg_lower = message.lower().strip()
    return any(k in msg_lower for k in KEYWORDS_CONFIRMA_RAG)


def necesita_bases(message: str) -> bool:
    """La pregunta sugiere que necesitaría leer las bases."""
    msg_lower = message.lower()
    return any(k in msg_lower for k in KEYWORDS_NECESITA_BASES)


# =================================================================
# BÚSQUEDA GENERAL
# =================================================================

KEYWORDS_BUSQUEDA = {
    "recomienda", "busca", "encuentra", "licitacion", "licitación",
    "oportunidad", "contrato", "convocatoria", "quiero", "muestra",
    "dame", "hay", "existe", "consultor", "obra", "servicio", "bien",
    "sector", "region", "región", "lima", "arequipa", "cusco", "piura",
    "cajamarca", "junin", "lambayeque", "monto", "selva", "sierra",
    "municipalidad", "ministerio", "hospital", "colegio", "universidad",
}
KEYWORDS_CONTINUACION = {
    "también", "tambien", "añade", "agrega", "además", "ademas",
    "más", "mas", "otro", "otra", "siguiente", "ok", "si", "sí",
    "no", "gracias", "claro", "entendido", "perfecto", "dale", "ya",
    "hola", "buenas", "buenos", "buen", "hi", "hey", "saludos",
    "genial", "excelente", "bien", "listo", "correcto", "exacto",
}


def buscar_licitaciones_general(message: str,
                                profile: Dict,
                                top_k: int = 5) -> tuple:
    if not MATCH_AVAILABLE or me is None:
        return [], ""

    msg_lower = message.lower().strip()
    if len(msg_lower) < 30 and any(k in msg_lower for k in KEYWORDS_CONTINUACION):
        print(f"   ↩️  Continuación — sin búsqueda")
        return [], ""
    if not any(k in msg_lower for k in KEYWORDS_BUSQUEDA):
        print(f"   ℹ️  Mensaje general — sin búsqueda")
        return [], ""

    print(f"\n🔎 Buscando: '{message[:60]}'...")
    t = time.time()

    from match_engine_pgvector import _normalizar_lista
    categorias = _normalizar_lista(profile.get("contract_types"))
    regiones   = _normalizar_lista(profile.get("operation_regions") or profile.get("operationRegions"))
    monto_max  = float(profile.get("max_contract_value") or 0)
    strict     = bool(profile.get("strict_filters", False))

    licitaciones = me.buscar_licitaciones(
        message, top_k=top_k,
        categorias=categorias or None,
        regiones=regiones or None,
        monto_max=monto_max,
        strict=strict,
        solo_recientes=True,
    )
    print(f"   → {len(licitaciones)} en {(time.time()-t)*1000:.0f}ms")

    advertencia = ""
    if licitaciones and licitaciones[0].get("affinity_score", 0) < 55:
        advertencia = (
            f"Score máximo {licitaciones[0]['affinity_score']}% — baja afinidad. "
            f"Advierte al usuario y pregunta si igual quiere ver los resultados."
        )
    return licitaciones, advertencia

# =================================================================
# EMBED RAG
# =================================================================

# Caché de embeddings en memoria — evita re-vectorizar textos repetidos
_embed_cache: Dict[str, any] = {}
_embed_cache_lock = threading.Lock()
MAX_EMBED_CACHE = 200  # máximo entradas en caché

def embed_fn(text: str):
    try:
        import numpy as np
        key = text.strip()[:200]  # clave = primeros 200 chars
        with _embed_cache_lock:
            if key in _embed_cache:
                return _embed_cache[key]

        resp   = azure_client.embeddings.create(
            input=[text.replace("\n", " ").strip()],
            model=EMBED_DEPLOYMENT
        )
        vector = np.array(resp.data[0].embedding, dtype="float32")

        with _embed_cache_lock:
            if len(_embed_cache) >= MAX_EMBED_CACHE:
                # Limpiar mitad del caché cuando se llena
                keys = list(_embed_cache.keys())
                for k in keys[:MAX_EMBED_CACHE // 2]:
                    del _embed_cache[k]
            _embed_cache[key] = vector

        return vector
    except Exception as e:
        print(f"   ❌ embed_fn: {e}")
        return None

# =================================================================
# STREAMING
# =================================================================

def stream_response(messages: List[Dict]) -> Generator[str, None, None]:
    try:
        print(f"\n   💬 {len(messages)} msgs → {CHAT_MODEL}")
        for m in messages:
            print(f"      [{m['role']}]: {str(m['content'])[:80]}")

        t_api  = time.time()
        stream = azure_client.chat.completions.create(
            model=CHAT_MODEL, messages=messages, stream=True,
        )
        print(f"   ⏱️  API: {(time.time()-t_api)*1000:.0f}ms")

        t0, chunks, first = time.time(), 0, True
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                if first:
                    print(f"   ⏱️  1er token: {(time.time()-t0)*1000:.0f}ms")
                    first = False
                chunks += 1
                yield delta.content
        print(f"   ⏱️  Stream: {(time.time()-t0)*1000:.0f}ms ({chunks} chunks)")
    except Exception as e:
        print(f"   ❌ Streaming: {e}")
        import traceback; traceback.print_exc()
        yield f"\n❌ Error: {str(e)[:100]}. Intenta de nuevo."

# =================================================================
# ENDPOINTS
# =================================================================

@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok", "version": "2.0-pgvector", "model": CHAT_MODEL}, 200


@app.route("/chat_stream", methods=["POST"])
def chat_stream():
    data           = request.json or {}
    modo, fuente   = detectar_modo(data)
    user_id        = data.get("user_id")
    tender_context = data.get("tender_context") or []
    context        = data.get("context")          # ChatbotHelper usa "context"
    documents      = data.get("documents") or []

    # Si viene "context" del ChatbotHelper, tratarlo como tender_context de 1 item
    if context and isinstance(context, dict) and not tender_context:
        tender_context = [context]
        # Documentos pueden venir dentro del context o en el campo documents
        if not documents and context.get("documents"):
            documents = context["documents"]
    profile        = data.get("profile") or {}
    message        = data.get("message", "")
    chat_history   = data.get("chat_history") or []
    _t_total       = time.time()

    print(f"\n{'='*55}")
    print(f"📨 {modo.upper()} ({fuente}) | user={user_id or 'anon'} | '{message[:60]}'")
    print(f"{'='*55}")

    if user_id:
        db_profile = get_profile_from_db(int(user_id))
        if db_profile:
            profile = db_profile

    def generate():
        if modo == "busqueda":
            print(f"\n🔍 BÚSQUEDA")
            tender    = tender_context[0] if tender_context else {}
            rag_ctx   = ""
            tender_id = (tender.get("id_proceso") or tender.get("tender_id") or
                         tender.get("tenderId") or tender.get("id"))

            # Verificar si ya hay caché en pgvector
            tiene_cache = False
            if tender_id:
                from rag_engine import chunks_en_cache
                tiene_cache = len(chunks_en_cache(tender_id)) > 0

            # Decidir qué hacer con los documentos
            if tiene_cache:
                # Ya procesado antes → usar siempre sin preguntar
                yield "📄 Consultando bases...\n\n"
                index, chunks = rag_engine.build_index(documents, embed_fn, tender_id=tender_id)
                if index and chunks:
                    rag_ctx = "\n\n".join(rag_engine.search(index, chunks, message, embed_fn))

            elif documents and usuario_confirma_rag(message):
                # Usuario confirmó explícitamente cargar las bases
                yield "📄 Cargando bases... (puede tomar ~30 segundos)\n\n"
                index, chunks = rag_engine.build_index(documents, embed_fn, tender_id=tender_id)
                if index and chunks:
                    rag_ctx = "\n\n".join(rag_engine.search(index, chunks, message, embed_fn))
                    yield f"✅ {len(chunks)} fragmentos guardados. Próximas consultas serán instantáneas.\n\n"
                else:
                    yield "⚠️ No se pudo extraer texto. Respondo con la información disponible.\n\n"

            elif documents and necesita_bases(message):
                # Pregunta técnica pero sin confirmar → ofrecer cargar
                # El system prompt se encargará de sugerir cargar las bases
                print(f"   💡 Pregunta técnica — sugerirá cargar bases")

            sp   = prompt_busqueda(tender, profile, rag_ctx, tiene_cache=tiene_cache)
            msgs = messages_busqueda(sp, chat_history, message)
            for chunk in stream_response(msgs): yield chunk

        elif modo == "recomendacion":
            print(f"\n⭐ RECOMENDACIÓN | {len(tender_context)} licitaciones")
            rag_ctx   = ""
            tender_id = tender_context[0].get("id_proceso") or tender_context[0].get("tender_id") if tender_context else None
            if len(tender_context) == 1 and documents:
                yield "📄 Analizando documentos...\n\n"
                index, chunks = rag_engine.build_index(documents, embed_fn, tender_id=tender_id)
                if index and chunks:
                    rag_ctx = "\n\n".join(rag_engine.search(index, chunks, message, embed_fn))
                    yield "✅ Documentos analizados.\n\n"
            sp   = prompt_recomendacion(tender_context, profile, rag_ctx)
            msgs = messages_recomendacion(sp, chat_history, message)
            for chunk in stream_response(msgs): yield chunk

        else:
            print(f"\n💬 GENERAL")
            lics, adv = buscar_licitaciones_general(message, profile)
            sp   = prompt_general(profile, lics, adv)
            msgs = messages_general(sp, chat_history, message)
            for chunk in stream_response(msgs): yield chunk

    def generate_timed():
        yield from generate()
        print(f"   ⏱️  TOTAL: {(time.time()-_t_total)*1000:.0f}ms")

    return Response(generate_timed(), mimetype="text/plain")


@app.route("/api/recommendations", methods=["POST"])
def get_recommendations():
    """
    El frontend envía el perfil del usuario (userProfile).
    Necesita user_id para buscar en BD, o lo extrae del token.
    Responde con { recommendations: [...] } en el formato que espera el frontend.
    """
    data    = request.json or {}

    # user_id puede venir directo o dentro del perfil
    user_id = (
        data.get("user_id") or
        data.get("userId") or
        data.get("id")
    )

    # Si no viene user_id intentar extraer del token JWT
    if not user_id:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            try:
                import base64, json as _json
                token    = auth_header.split(" ")[1]
                payload  = token.split(".")[1]
                # Padding base64
                payload += "=" * (4 - len(payload) % 4)
                decoded  = _json.loads(base64.b64decode(payload))
                user_id  = decoded.get("sub") or decoded.get("user_id") or decoded.get("id")
            except Exception:
                pass

    if not user_id or me is None:
        return Response(
            json.dumps({"error": "No se encontró user_id. Envíalo en el body o en el token."}),
            status=400, mimetype="application/json"
        )

    try:
        print(f"\n📋 RECOMMENDATIONS | user_id={user_id}")
        result = me.recomendar(user_id=int(user_id), top_k=10)

        # Mapear campos al formato que espera el frontend
        licitaciones = result.get("licitaciones") or []
        recommendations = []
        for lic in licitaciones:
            recommendations.append({
                "id_proceso":              lic.get("tender_id"),
                "objeto_contractual":      lic.get("objeto_contractual"),
                "entidad_convocante":      lic.get("entidad_convocante"),
                "region_ejecucion":        lic.get("region_ejecucion"),
                "monto_referencial":       lic.get("monto_referencial"),
                "affinity_score":          lic.get("affinity_score"),
                "tipo_contrato_clasificado": lic.get("categoria", "").capitalize(),
                "metodo":                  lic.get("metodo"),
                "fecha_publicacion":       str(lic.get("fecha_publicacion", "")),
            })

        response_data = {
            "recommendations": recommendations,
            "total":           len(recommendations),
            "user_id":         user_id,
            "company_name":    result.get("company_name"),
        }

        print(f"   ✅ {len(recommendations)} recomendaciones para {result.get('company_name')}")
        return Response(
            json.dumps(response_data, ensure_ascii=False),
            status=200, mimetype="application/json"
        )

    except Exception as e:
        import traceback; traceback.print_exc()
        return Response(json.dumps({"error": str(e)}), status=500, mimetype="application/json")


# =================================================================
# WARMUP — pre-calienta conexión Azure al arrancar
# =================================================================

def warmup():
    """Hace una llamada mínima a Azure OpenAI para pre-calentar la conexión."""
    try:
        print("\n🔥 Warmup Azure OpenAI...", end=" ", flush=True)
        t = time.time()
        azure_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": "ok"}],
            max_tokens=1,
            stream=False,
        )
        print(f"✅ {(time.time()-t)*1000:.0f}ms")
    except Exception as e:
        print(f"⚠️  {e}")


# =================================================================
# MAIN
# =================================================================

if __name__ == "__main__":
    port  = int(os.getenv("PORT", 5007))
    debug = os.getenv("FLASK_ENV", "production") != "production"
    warmup()
    print(f"\n🚀 LiciGob Chat Engine v2.0 | puerto {port}")
    print(f"💡 Producción: gunicorn -w 4 -b 0.0.0.0:{port} --timeout 120 --worker-class gthread --threads 4 chat_engine:app\n")
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)
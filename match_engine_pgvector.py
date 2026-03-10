# -*- coding: utf-8 -*-
"""
match_engine_pgvector.py

Motor de match semántico usando pgvector en PostgreSQL.
Reemplaza al match_engine_v2.py (FAISS) — sin archivos .index externos.

Funciones principales:
  - get_embedding(texto)        → genera vector con Azure OpenAI
  - buscar_licitaciones(query)  → búsqueda semántica en pgvector
  - recomendar(user_id)         → recomendaciones por perfil de empresa

Uso desde chat_engine.py:
  import match_engine_pgvector as me
  resultados = me.buscar_licitaciones("consultoría TI Lima", top_k=5)
  recomendaciones = me.recomendar(user_id=6, top_k=10)
"""

import os
import time
import numpy as np
import psycopg2
import httpx
from psycopg2.extras import RealDictCursor
from openai import AzureOpenAI
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

# =================================================================
# CONFIGURACIÓN
# =================================================================

DB_CONFIG = {
    "host":     os.getenv("DB_HOST"),
    "user":     os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "port":     int(os.getenv("DB_PORT", "5432")),
    "sslmode":  os.getenv("DB_SSLMODE", "require"),
}

_http = httpx.Client(
    timeout=httpx.Timeout(60.0, connect=10.0),
    limits=httpx.Limits(
        max_keepalive_connections=20,
        max_connections=50,
        keepalive_expiry=30.0,
    ),
)
client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    azure_endpoint=os.getenv("OPENAI_API_BASE"),
    api_version=os.getenv("OPENAI_API_VERSION", "2024-12-01-preview"),
    http_client=_http,
    max_retries=2,
)
DEPLOYMENT = os.getenv("OPENAI_DEPLOYMENT", "text-embedding-3-small")

# Pool de conexiones BD
_conn_pool: list = []
MAX_CONNS = 10


# =================================================================
# CONEXIÓN BD — pool simple
# =================================================================

def _get_conn():
    """Obtiene una conexión del pool o crea una nueva."""
    if _conn_pool:
        conn = _conn_pool.pop()
        try:
            conn.cursor().execute("SELECT 1")
            return conn
        except Exception:
            pass  # Conexión muerta — crear nueva
    return psycopg2.connect(**DB_CONFIG)


def _release_conn(conn):
    """Devuelve la conexión al pool."""
    if len(_conn_pool) < MAX_CONNS:
        _conn_pool.append(conn)
    else:
        conn.close()


# =================================================================
# EMBEDDINGS
# =================================================================

def get_embedding(texto: str) -> np.ndarray | None:
    """Genera embedding con Azure OpenAI."""
    try:
        texto = texto.replace("\n", " ").strip()[:2000]
        resp  = client.embeddings.create(input=[texto], model=DEPLOYMENT)
        return np.array(resp.data[0].embedding, dtype="float32")
    except Exception as e:
        print(f"   ❌ Error embedding: {e}")
        return None


def construir_texto_perfil(profile: Dict) -> str:
    """Construye texto del perfil para generar embedding de búsqueda."""
    partes = []

    ciiu  = profile.get("ciiu_codigo", "")
    desc  = profile.get("ciiu_descripcion", "")
    sects = profile.get("primarySectors", "")
    regs  = profile.get("operationRegions", "")
    hist  = profile.get("historical_projects_summary", "")
    udesc = profile.get("user_description", "")

    if desc:
        partes.append(desc)
    if udesc:
        partes.append(str(udesc)[:200])
    if sects and sects not in ("[]", ""):
        partes.append(str(sects).replace("[", "").replace("]", "").replace('"', ""))
    if hist:
        partes.append(str(hist)[:200])
    if regs and regs not in ("[]", ""):
        partes.append(str(regs).replace("[", "").replace("]", "").replace('"', ""))
    if ciiu:
        partes.append(f"CIIU {ciiu}")

    return " | ".join(partes) if partes else "empresa peruana licitaciones"


# =================================================================
# BÚSQUEDA SEMÁNTICA
# =================================================================

SQL_BASE = """
SELECT
    le.tender_id,
    t.title                         AS objeto_contractual,
    t.main_procurement_category     AS categoria,
    t.procurement_method_details    AS metodo,
    t.value_amount                  AS monto_referencial,
    t.date_published                AS fecha_publicacion,
    p.name                          AS entidad_convocante,
    p.department                    AS departamento,
    p.region                        AS region_ejecucion,
    1 - (le.embedding <=> %s::vector) AS similarity
FROM licitaciones_embeddings le
JOIN tenders t  ON t.id = le.tender_id
LEFT JOIN parties p ON p.id = t.buyer_id
{where_clause}
ORDER BY le.embedding <=> %s::vector
LIMIT %s;
"""

# Mapeo contract_types del perfil → main_procurement_category de tenders
CONTRACT_MAP = {
    "servicios":   "services",
    "consultoría": "services",
    "consultoria": "services",
    "services":    "services",
    "bienes":      "goods",
    "suministros": "goods",
    "goods":       "goods",
    "obras":       "works",
    "obra":        "works",
    "works":       "works",
}

# Mapeo de nombres de regiones del perfil → department en parties
REGION_MAP = {
    "lima":           "LIMA",
    "arequipa":       "AREQUIPA",
    "cusco":          "CUSCO",
    "cuzco":          "CUSCO",
    "piura":          "PIURA",
    "la libertad":    "LA LIBERTAD",
    "lalibertad":     "LA LIBERTAD",
    "junin":          "JUNIN",
    "junín":          "JUNIN",
    "cajamarca":      "CAJAMARCA",
    "lambayeque":     "LAMBAYEQUE",
    "ancash":         "ANCASH",
    "áncash":         "ANCASH",
    "puno":           "PUNO",
    "ayacucho":       "AYACUCHO",
    "loreto":         "LORETO",
    "san martin":     "SAN MARTIN",
    "san martín":     "SAN MARTIN",
    "huanuco":        "HUANUCO",
    "huánuco":        "HUANUCO",
    "ica":            "ICA",
    "moquegua":       "MOQUEGUA",
    "tacna":          "TACNA",
    "tumbes":         "TUMBES",
    "ucayali":        "UCAYALI",
    "madre de dios":  "MADRE DE DIOS",
    "apurimac":       "APURIMAC",
    "apurímac":       "APURIMAC",
    "huancavelica":   "HUANCAVELICA",
    "pasco":          "PASCO",
    "amazonas":       "AMAZONAS",
    "callao":         "LIMA",  # Callao → Lima para efectos del filtro
}


def _normalizar_lista(valor) -> list:
    """Convierte string JSON o lista a lista limpia."""
    if not valor:
        return []
    if isinstance(valor, list):
        return [str(v).strip() for v in valor if v]
    # String tipo '["Lima", "Arequipa"]' o "Lima, Arequipa"
    clean = str(valor).replace("[", "").replace("]", "").replace('"', "").replace("'", "")
    return [v.strip() for v in clean.split(",") if v.strip()]


def _build_where(categorias: list = None,
                 regiones: list = None,
                 monto_max: float = None,
                 strict: bool = False,
                 solo_recientes: bool = True) -> tuple:
    """
    Construye cláusula WHERE para pre-filtrar antes del embedding.

    Filtros SIEMPRE activos:
      - categoria  (si hay contract_types en perfil)
      - fecha      (últimos 12 meses por defecto)

    Filtros solo si strict=True:
      - regiones   (operation_regions del perfil)
      - monto_max  (max_contract_value del perfil)

    Retorna (where_sql, params_extra)
    """
    conditions = []
    params     = []

    # ── 1. Filtro por categoría (SIEMPRE si hay datos) ───────────
    if categorias:
        cats_norm = list({CONTRACT_MAP.get(c.lower().strip())
                         for c in categorias
                         if CONTRACT_MAP.get(c.lower().strip())})
        if cats_norm:
            placeholders = ",".join(["%s"] * len(cats_norm))
            conditions.append(f"t.main_procurement_category IN ({placeholders})")
            params.extend(cats_norm)
            print(f"   🔍 Filtro categoría : {cats_norm}")

    # ── 2. Filtro por fecha (SIEMPRE — últimos 12 meses) ─────────
    if solo_recientes:
        conditions.append("t.date_published >= NOW() - INTERVAL '12 months'")
        print(f"   🔍 Filtro fecha     : últimos 12 meses")

    # ── 3. Filtro por región (solo si strict=True) ────────────────
    if strict and regiones:
        regs_norm = list({REGION_MAP.get(r.lower().strip())
                         for r in regiones
                         if REGION_MAP.get(r.lower().strip())})
        if regs_norm:
            placeholders = ",".join(["%s"] * len(regs_norm))
            conditions.append(f"p.department IN ({placeholders})")
            params.extend(regs_norm)
            print(f"   🔍 Filtro región    : {regs_norm}")

    # ── 4. Filtro por monto (solo si strict=True) ─────────────────
    if strict and monto_max and monto_max > 0:
        conditions.append("t.value_amount <= %s")
        params.append(monto_max)
        print(f"   🔍 Filtro monto     : <= S/ {monto_max:,.0f}")

    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    return where, params


def buscar_licitaciones(query: str,
                        top_k: int = 5,
                        categorias: List[str] = None,
                        regiones: List[str] = None,
                        monto_max: float = None,
                        strict: bool = False,
                        solo_recientes: bool = True) -> List[Dict]:
    """
    Búsqueda semántica con pre-filtros SQL.
    SIEMPRE filtra: categoría + últimos 12 meses
    Si strict=True: también filtra por región y monto
    """
    t0 = time.time()
    vector = get_embedding(query)
    if vector is None:
        return []
    print(f"   ⏱️  Embedding: {(time.time()-t0)*1000:.0f}ms")

    vector_str  = "[" + ",".join(str(x) for x in vector.tolist()) + "]"
    where, params_extra = _build_where(categorias, regiones, monto_max, strict, solo_recientes)

    conn = _get_conn()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        t1  = time.time()

        sql    = SQL_BASE.format(where_clause=where)
        params = [vector_str] + params_extra + [vector_str, top_k * 3]
        cur.execute(sql, params)
        rows = cur.fetchall()
        print(f"   ⏱️  pgvector search: {(time.time()-t1)*1000:.0f}ms "
              f"({'con filtros' if where else 'sin filtros'})")

        resultados = []
        for row in rows:
            score = max(float(row["similarity"]), 0)
            resultados.append({
                "tender_id":          row["tender_id"],
                "objeto_contractual": row["objeto_contractual"],
                "entidad_convocante": row["entidad_convocante"],
                "region_ejecucion":   row["region_ejecucion"] or row["departamento"],
                "monto_referencial":  float(row["monto_referencial"] or 0),
                "categoria":          row["categoria"],
                "metodo":             row["metodo"],
                "fecha_publicacion":  str(row["fecha_publicacion"] or ""),
                "affinity_score":     round(score * 100, 1),
            })
            if len(resultados) >= top_k:
                break

        if resultados:
            print(f"   ✅ {len(resultados)} resultados | "
                  f"Mejor: {resultados[0]['affinity_score']}% | "
                  f"Total: {(time.time()-t0)*1000:.0f}ms")
        else:
            print(f"   ⚠️  Sin resultados con filtros — reintentando sin filtros")
            return buscar_licitaciones(query, top_k=top_k)  # fallback sin filtros

        return resultados

    except Exception as e:
        print(f"   ❌ Error búsqueda pgvector: {e}")
        import traceback
        traceback.print_exc()
        return []
    finally:
        _release_conn(conn)


# =================================================================
# RECOMENDACIONES POR PERFIL
# =================================================================

QUERY_PERFIL = """
SELECT
    u.ciiu_codigo,
    e.company_name,
    e.user_description,
    e.historical_projects_summary,
    e.primary_sectors,
    e.rnp_specialties,
    e.operation_regions,
    e.contract_types,
    e.max_contract_value,
    e.strict_filters,
    c.descripcion           AS ciiu_descripcion,
    c.descripcion_detallada AS ciiu_detalle
FROM users u
JOIN empresas_profiles e     ON e.user_id = u.id
LEFT JOIN ciiu_actividades c ON c.codigo   = u.ciiu_codigo
WHERE u.id = %s;
"""


def recomendar(user_id: int, top_k: int = 10) -> Dict:
    """
    Genera recomendaciones personalizadas para un usuario.
    Retorna dict con perfil + licitaciones recomendadas.
    """
    conn = _get_conn()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Traer perfil
        cur.execute(QUERY_PERFIL, (user_id,))
        perfil_row = cur.fetchone()
        if not perfil_row:
            return {"error": f"No se encontró perfil para user_id={user_id}"}

        perfil = dict(perfil_row)
        print(f"   👤 Perfil: {perfil.get('company_name')} | CIIU: {perfil.get('ciiu_codigo')}")

        # Construir texto de búsqueda desde el perfil
        texto_perfil = construir_texto_perfil({
            "ciiu_codigo":                  perfil.get("ciiu_codigo"),
            "ciiu_descripcion":             perfil.get("ciiu_descripcion"),
            "user_description":             perfil.get("user_description"),
            "historical_projects_summary":  perfil.get("historical_projects_summary"),
            "primarySectors":               perfil.get("primary_sectors"),
            "operationRegions":             perfil.get("operation_regions"),
        })
        print(f"   📝 Texto perfil: {texto_perfil[:80]}...")

        # Extraer filtros del perfil
        # Extraer y normalizar filtros del perfil
        categorias = _normalizar_lista(perfil.get("contract_types"))
        regiones   = _normalizar_lista(perfil.get("operation_regions"))
        monto_max  = float(perfil.get("max_contract_value") or 0)
        strict     = bool(perfil.get("strict_filters", False))

        print(f"   🔍 Categorías : {categorias}")
        print(f"   🔍 Regiones   : {regiones}")
        print(f"   🔍 Monto máx  : S/ {monto_max:,.0f} | Strict: {strict}")

        # Buscar con pre-filtros
        licitaciones = buscar_licitaciones(
            texto_perfil,
            top_k=top_k,
            categorias=categorias or None,
            regiones=regiones or None,
            monto_max=monto_max,
            strict=strict,
            solo_recientes=True,
        )

        return {
            "user_id":      user_id,
            "company_name": perfil.get("company_name"),
            "ciiu_codigo":  perfil.get("ciiu_codigo"),
            "total":        len(licitaciones),
            "licitaciones": licitaciones,
        }

    except Exception as e:
        print(f"   ❌ Error en recomendar: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
    finally:
        _release_conn(conn)


# =================================================================
# TEST RÁPIDO
# =================================================================

if __name__ == "__main__":
    import sys

    print("\n" + "="*60)
    print("🧪 TEST match_engine_pgvector")
    print("="*60)

    # Test 1 — búsqueda por texto
    print("\n1. Búsqueda por texto: 'desarrollo de software Lima'")
    resultados = buscar_licitaciones("desarrollo de software Lima", top_k=3)
    for r in resultados:
        print(f"   [{r['affinity_score']}%] {r['objeto_contractual'][:60]}")
        print(f"            {r['entidad_convocante']} | {r['region_ejecucion']}")

    # Test 2 — recomendaciones por user_id
    user_id = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    print(f"\n2. Recomendaciones para user_id={user_id}")
    recs = recomendar(user_id=user_id, top_k=5)
    if "error" not in recs:
        print(f"   Empresa: {recs['company_name']}")
        for r in recs["licitaciones"]:
            print(f"   [{r['affinity_score']}%] {r['objeto_contractual'][:60]}")
    else:
        print(f"   ❌ {recs['error']}")

    print("\n✅ Test completado")
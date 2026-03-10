# chat_general.py
# Modo GENERAL — chat libre sobre licitaciones peruanas.

import os, re
from typing import List, Dict, Tuple

# =================================================================
# SYSTEM PROMPT
# =================================================================

SYSTEM_PROMPT = """Eres LiciBot, asistente de LiciGob — plataforma peruana para encontrar y priorizar licitaciones públicas.

SECCIONES DE LICIGOB:
  🔍 Buscar Licitaciones → busca por palabras, entidad, código o región. Chatbot analiza documentos PDF.
  🎯 Mis Oportunidades   → recomendaciones IA personalizadas con score de afinidad.
  🔔 Alertas             → palabras clave + correo automático de nuevas licitaciones.
  👤 Mi Perfil           → completa CIIU, regiones y contratos para mejores resultados.

CUÁNDO REDIRIGIR (de forma natural):
- Pide más detalle de una licitación → sugiere 🔍 Buscar Licitaciones
- Quiere ver todas sus recomendaciones → sugiere 🎯 Mis Oportunidades
- Quiere leer bases o documentos → sugiere 🔍 Buscar Licitaciones

ESTILO:
- Saludo → 1 línea + pregunta qué necesita
- NUNCA respuesta vacía
- Si hay LICITACIONES ENCONTRADAS en el contexto → SIEMPRE muéstralas todas, sin excepción
- Nunca digas "no encontré" si hay licitaciones en el contexto

FORMATO licitaciones (usar siempre que haya resultados):
📌 [Entidad] — [Objeto máx 60 chars]
   Región: X | Monto: S/ X | Afinidad: X%
"""

# =================================================================
# STOPWORDS — palabras a ignorar al extraer keywords
# =================================================================
STOPWORDS = {
    "dame", "dime", "muestra", "busca", "encuentra", "quiero", "necesito",
    "hay", "existe", "existen", "ver", "buscar", "mostrar", "listar",
    "las", "los", "una", "uno", "unos", "unas", "del", "para", "con",
    "licitaciones", "licitacion", "licitación", "contratos", "contrato",
    "convocatorias", "convocatoria", "procesos", "proceso",
    "sin", "importar", "afinidad", "perfil", "todas", "todos",
    "por", "favor", "que", "me", "des", "den", "dar",
}

REGIONES_PERU = {
    "lima", "arequipa", "cusco", "piura", "la libertad", "lambayeque",
    "junin", "ancash", "cajamarca", "loreto", "puno", "ica", "huanuco",
    "san martin", "ucayali", "ayacucho", "huancavelica", "apurimac",
    "tumbes", "moquegua", "tacna", "pasco", "amazonas", "madre de dios",
    "callao",
}

# =================================================================
# EXTRACCIÓN DE KEYWORDS
# =================================================================

def extraer_keywords(message: str) -> Tuple[str, str]:
    """
    Extrae keywords relevantes y región del mensaje.
    Retorna (query_limpia, region_detectada)

    Ejemplos:
      "dame licitaciones de polos en lima"
        → query="polos", region="lima"
      "busca telas y uniformes en cusco"
        → query="telas uniformes", region="cusco"
      "licitaciones de construccion sin importar afinidad"
        → query="construccion", region=""
    """
    msg = message.lower().strip()

    # Detectar región
    region = ""
    for reg in REGIONES_PERU:
        if reg in msg:
            region = reg
            # Quitar la región del mensaje para no contaminar la query
            msg = msg.replace(f" en {reg}", "").replace(f" de {reg}", "").replace(reg, "")
            break

    # Quitar stopwords
    words = re.findall(r'\b\w+\b', msg)
    keywords = [w for w in words if w not in STOPWORDS and len(w) > 2]

    query = " ".join(keywords).strip()

    # Si quedó vacío, usar el mensaje original sin stopwords básicas
    if not query:
        query = message

    return query, region


# =================================================================
# SYSTEM PROMPT BUILDER
# =================================================================

def get_system_prompt(profile: Dict,
                      licitaciones_contexto: List[Dict] = None,
                      advertencia_score: str = "",
                      query_usada: str = "",
                      region_usada: str = "") -> str:

    # Perfil
    profile_info = ""
    ciiu  = profile.get("ciiu_codigo", "")
    sects = profile.get("primarySectors", "")
    regs  = profile.get("operationRegions", "")

    if ciiu or sects:
        vacios = []
        if not sects or sects in ("[]", ""):  vacios.append("sectores")
        if not regs  or regs  in ("[]", ""):  vacios.append("regiones")

        perfil_str = f"CIIU {ciiu}" if ciiu else ""
        if sects and sects not in ("[]", ""):
            perfil_str += f" | {str(sects).replace('[','').replace(']','').replace(chr(34),'')[:80]}"
        if regs and regs not in ("[]", ""):
            perfil_str += f" | {str(regs).replace('[','').replace(']','').replace(chr(34),'')[:60]}"

        profile_info = f"\nPERFIL: {perfil_str}"
        if vacios:
            profile_info += f"\n⚠️ Perfil incompleto: faltan {', '.join(vacios)}."

    # Contexto de búsqueda
    busqueda_info = ""
    if query_usada:
        busqueda_info = f"\nBÚSQUEDA: '{query_usada}'"
        if region_usada:
            busqueda_info += f" en {region_usada.upper()}"

    # Licitaciones
    lic_info = ""
    if licitaciones_contexto:
        lic_info = f"\nLICITACIONES ENCONTRADAS ({len(licitaciones_contexto)}):\n"
        for i, lic in enumerate(licitaciones_contexto[:5], 1):
            obj   = str(lic.get("objeto_contractual") or lic.get("item") or "N/A")[:60]
            ent   = str(lic.get("entidad_convocante") or lic.get("entidad") or "N/A")[:40]
            reg   = lic.get("region_ejecucion") or lic.get("departamento") or "N/A"
            monto = lic.get("monto_referencial") or lic.get("monto") or 0
            score = lic.get("affinity_score", 0)
            lic_info += f"  #{i} [{score}%] {ent} — {obj} | {reg} | S/{monto:,}\n"

        lic_info += (
            "\nDESPUÉS DE MOSTRAR los resultados, si el score es bajo (<55%) o hay muchos resultados, "
            "puedes preguntar UNA sola cosa para refinar: región, tipo de contrato o rango de monto. "
            "Nunca hagas más de una pregunta a la vez."
        )
    else:
        lic_info = (
            "\nSin resultados para esta búsqueda. "
            "Dile al usuario que no encontraste coincidencias y sugiere: "
            "1) reformular con otras palabras, "
            "2) ir a 🔍 Buscar Licitaciones para búsqueda avanzada."
        )

    aviso = f"\nADVERTENCIA: {advertencia_score}" if advertencia_score else ""
    return SYSTEM_PROMPT + profile_info + busqueda_info + lic_info + aviso


# =================================================================
# MESSAGES BUILDER
# =================================================================

def get_messages(system_prompt: str,
                 chat_history: List[Dict],
                 user_message: str) -> List[Dict]:
    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history[-4:]:
        messages.append({
            "role":    msg.get("role", "user"),
            "content": str(msg.get("content", ""))[:500]
        })
    messages.append({"role": "user", "content": user_message})
    return messages
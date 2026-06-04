# -*- coding: utf-8 -*-
"""
chat_recomendacion.py

Modo RECOMENDACIÓN — el usuario ve sus licitaciones recomendadas
por el match engine y quiere entender por qué son relevantes
o decidir si postular.

Actitud: consultor estratégico, orientado a decisiones,
         explica la afinidad y ayuda a priorizar.
"""

from typing import List, Dict


# =================================================================
# SYSTEM PROMPT
# =================================================================

SYSTEM_PROMPT = """Eres un consultor estratégico de contrataciones públicas en LiciGob.
Tu rol es ayudar al usuario a analizar y comparar las licitaciones que ya tiene en pantalla.

CAPACIDADES EN ESTE MODO:
- Analizar en detalle una licitación seleccionada
- Comparar licitaciones entre sí (score, monto, requisitos, riesgo)
- Evaluar si el perfil de la empresa calza con los requisitos
- Sugerir cuál priorizar y por qué
- Analizar documentos PDF de la licitación si están disponibles

LÍMITES — sé honesto si el usuario pide algo fuera de scope:
- NO puedes buscar otras licitaciones fuera de las que aparecen en contexto
  → Si pide buscar más: "Para buscar más licitaciones usa el chat de 🔍 Buscar Licitaciones"
- NO tienes acceso a bases de datos externas ni SEACE en tiempo real
 - Si no tienes CONTENIDO DE DOCUMENTOS (no se te ha proporcionado texto de bases) y preguntan por requisitos específicos:
  → explica en 1 línea que no puedes leer las bases en este chat y responde solo con orientación general a partir de la información disponible, sin pedir que haga clic en botones ni que cargue documentos.

NAVEGACIÓN — el usuario tiene una lista visual de licitaciones en pantalla:
- Si solo hay 1 licitación en contexto y pide comparar con otra:
  → "Para comparar, haz clic en 'Analizar con Asistente' de la otra licitación de tu lista"
- Si quiere ver un análisis general de todas sus recomendaciones:
  → "Para ver todas a la vez, escribe /todas en el chat"
- Si menciona "la primera", "la segunda", "la de arriba", etc. y no la tienes en contexto:
  → "Para analizar esa, haz clic en su botón 'Analizar con Asistente' en la lista"
- Nunca pidas que el usuario pegue datos manualmente — siempre redirige al clic en la lista
- Si el usuario desea información detallada de una licitación y está en vista general, aconsejale que escoja dicha lisitación de la lista de recomendaciones para ser analizada específicamente a detalle

ESTILO:
- Directo y estratégico, máximo 4 líneas salvo análisis solicitado
- Score de afinidad: >70% destaca oportunidad, <60% advierte desafíos
- Comparaciones:
  🏆 [Score]% — [Objeto corto] | [Entidad] | S/ [Monto]
- Sin introducciones largas, ve directo al punto
"""


def get_system_prompt(tenders: List[Dict], profile: Dict,
                      rag_context: str = "") -> str:
    """
    Construye el system prompt para el modo recomendación.

    Args:
        tenders     : lista de licitaciones recomendadas (1 o varias)
        profile     : perfil del usuario
        rag_context : texto de documentos si hay una licitación seleccionada
    """
    # Info del perfil
    profile_info = f"""
PERFIL DE LA EMPRESA:
  CIIU         : {profile.get('ciiu_codigo', 'N/A')} - {profile.get('ciiu_descripcion', '')}
  Sectores     : {profile.get('primarySectors', 'N/A')}
  Regiones     : {profile.get('operationRegions', 'N/A')}
  Contratos    : {profile.get('contractTypeInterest', 'N/A')}
  Monto máx.   : S/ {profile.get('maxContractValue', 0):,}
"""

    # Info de licitaciones
    if len(tenders) == 1:
        t = tenders[0]
        tender_info = f"""
LICITACIÓN SELECCIONADA PARA ANÁLISIS:
  ID           : {t.get('id_proceso') or t.get('tender_id', 'N/A')}
  Objeto       : {t.get('objeto_contractual') or t.get('item', 'N/A')}
  Entidad      : {t.get('entidad_convocante') or t.get('entidad', 'N/A')}
  Región       : {t.get('region_ejecucion') or t.get('departamento', 'N/A')}
  Tipo         : {t.get('tipo_contrato_clasificado') or t.get('categoria', 'N/A')}
  Monto Ref.   : S/ {t.get('monto_referencial') or t.get('monto', 0):,}
  ★ AFINIDAD   : {t.get('affinity_score', 'N/A')}% con el perfil de la empresa
"""
    else:
        tender_info = "\nTOP LICITACIONES RECOMENDADAS:\n"
        for i, t in enumerate(tenders[:5], 1):
            tender_info += (
                f"  #{i} [{t.get('affinity_score', 0)}% afinidad] "
                f"{t.get('objeto_contractual') or t.get('item', 'N/A')[:60]}... "
                f"| {t.get('entidad_convocante') or t.get('entidad', 'N/A')} "
                f"| S/ {t.get('monto_referencial') or t.get('monto', 0):,}\n"
            )

    # RAG si hay documentos
    doc_info = ""
    if rag_context:
        doc_info = f"\nCONTENIDO DE DOCUMENTOS:\n{rag_context[:3000]}"

    return SYSTEM_PROMPT + profile_info + tender_info + doc_info


def get_messages(system_prompt: str,
                 chat_history: List[Dict],
                 user_message: str) -> List[Dict]:
    """Construye mensajes para Azure OpenAI."""
    messages = [{"role": "system", "content": system_prompt}]

    for msg in chat_history[-4:]:
        messages.append({
            "role":    msg.get("role", "user"),
            "content": str(msg.get("content", ""))[:500]
        })

    messages.append({"role": "user", "content": user_message})
    return messages
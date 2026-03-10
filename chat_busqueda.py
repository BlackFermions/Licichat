# -*- coding: utf-8 -*-
"""
chat_busqueda.py

Modo BÚSQUEDA — el usuario encontró una licitación buscando por palabras
y quiere analizarla en detalle.

Actitud: técnico, detallista, responde sobre bases administrativas,
         requisitos, plazos y documentación.
"""

from typing import List, Dict


# =================================================================
# SYSTEM PROMPT
# =================================================================

SYSTEM_PROMPT = """Eres un analista técnico de licitaciones públicas peruanas en LiciGob.

ESTILO:
- Máximo 3 líneas salvo análisis completo solicitado
- Nunca repitas el ID ni el nombre de la licitación en cada respuesta
- Nunca digas "Licitación X está lista" ni "tengo la licitación cargada"
- Saludo: solo 1 línea breve, sin listar datos
- Responde directo al punto — sin introducciones ni cierres

CUANDO NO TIENES EL DATO:
- Di en 1 línea que no está disponible
- Solo ofrece cargar bases si el usuario pregunta algo técnico específico
  (requisitos, garantía, plazos, experiencia, capital, RNP)
- Frase exacta: "Para el dato exacto escribe 'sí, carga las bases'."

CUANDO TIENES DOCUMENTOS CARGADOS:
- Responde directamente con el dato
- Cita el documento fuente si es relevante (ej: "Según las Bases Administrativas:")
- Sin frases de relleno
"""


def _normalizar_tender(tender: Dict) -> Dict:
    """
    Normaliza campos del tender al formato interno.
    Soporta tanto el formato de Búsqueda (TenderDetailPopup)
    como el formato de Oportunidades.
    """
    return {
        "id":          (tender.get("id_proceso") or tender.get("tender_id") or
                        tender.get("tenderId") or tender.get("id") or "N/A"),
        "objeto":      (tender.get("objeto_contractual") or tender.get("tenderTitle") or
                        tender.get("title") or tender.get("description") or "N/A"),
        "entidad":     (tender.get("entidad_convocante") or tender.get("buyer_name") or "N/A"),
        "region":      (tender.get("region_ejecucion") or tender.get("department") or
                        tender.get("region") or
                        (tender.get("buyer_address") or {}).get("department") or "N/A"),
        "tipo":        (tender.get("tipo_contrato_clasificado") or
                        tender.get("main_procurement_category") or tender.get("categoria") or "N/A"),
        "metodo":      (tender.get("metodo") or tender.get("procurement_method_details") or "N/A"),
        "monto":       (tender.get("monto_referencial") or tender.get("value_amount") or 0),
        "fecha":       (tender.get("fecha_publicacion") or tender.get("date_published") or "N/A"),
        "descripcion": (tender.get("description") or ""),
    }


def get_system_prompt(tender: Dict, profile: Dict,
                      rag_context: str = "",
                      tiene_cache: bool = False) -> str:
    """
    Construye el system prompt completo para el modo búsqueda.
    Soporta campos de TenderDetailPopup (Búsqueda) y OpportunitiesPage.
    """
    t = _normalizar_tender(tender)

    tender_info = f"""
LICITACIÓN EN ANÁLISIS:
  ID           : {t['id']}
  Objeto       : {t['objeto']}
  Entidad      : {t['entidad']}
  Región       : {t['region']}
  Tipo         : {t['tipo']}
  Método       : {t['metodo']}
  Monto Ref.   : S/ {t['monto']:,}
  Fecha        : {t['fecha']}
"""
    if t['descripcion']:
        tender_info += f"  Descripción  : {t['descripcion'][:300]}\n"

    # Info del perfil
    profile_info = f"""
PERFIL DE LA EMPRESA:
  CIIU         : {profile.get('ciiu_codigo', 'N/A')} - {profile.get('ciiu_descripcion', '')}
  Sectores     : {profile.get('primarySectors', 'N/A')}
  Regiones     : {profile.get('operationRegions', 'N/A')}
  Contratos    : {profile.get('contractTypeInterest', 'N/A')}
"""

    # Contexto RAG de documentos
    if rag_context:
        doc_info = f"\nCONTENIDO DE DOCUMENTOS:\n{rag_context[:3000]}\n"
    elif tiene_cache:
        doc_info = "\nNOTA: Las bases ya están cargadas. Responde con los datos disponibles.\n"
    else:
        doc_info = "\nNOTA: Sin documentos cargados. Si preguntan algo técnico específico (requisitos, garantía, experiencia, RNP, plazos), di que no tienes el dato y ofrece: 'Para el dato exacto escribe sí, carga las bases.'. No ofrezcas esto para preguntas generales.\n"

    return SYSTEM_PROMPT + tender_info + profile_info + doc_info


def get_messages(system_prompt: str,
                 chat_history: List[Dict],
                 user_message: str) -> List[Dict]:
    """
    Construye la lista de mensajes para la API de Azure OpenAI.

    Args:
        system_prompt : prompt del sistema con contexto
        chat_history  : historial previo [{role, content}]
        user_message  : mensaje actual del usuario
    """
    messages = [{"role": "system", "content": system_prompt}]

    # Agregar historial (últimos 6 mensajes para no exceder tokens)
    for msg in chat_history[-4:]:
        messages.append({
            "role":    msg.get("role", "user"),
            "content": str(msg.get("content", ""))[:500]
        })

    messages.append({"role": "user", "content": user_message})
    return messages
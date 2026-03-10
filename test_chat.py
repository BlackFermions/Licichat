# -*- coding: utf-8 -*-
"""
test_chat.py

Prueba el chat_engine directamente desde terminal sin necesidad del frontend.
Simula los 3 modos: busqueda, recomendacion, general.

Uso:
  python test_chat.py                  → modo interactivo (elige modo)
  python test_chat.py general          → modo general directo
  python test_chat.py general 6        → modo general con perfil de BD
  python test_chat.py recomendacion 6  → modo recomendacion con user_id=6
  python test_chat.py busqueda         → modo busqueda
"""

import sys
import json
import requests

CHAT_URL            = "http://localhost:5007/chat_stream"
RECOMMENDATIONS_URL = "http://localhost:5007/api/recommendations"

# =================================================================
# PERFIL DE PRUEBA (cuando no hay user_id)
# =================================================================

PROFILE_LEGACY = {
    "ciiu_codigo":          "6201",
    "ciiu_descripcion":     "PROGRAMACIÓN INFORMÁTICA",
    "primarySectors":       "Tecnología de información, Desarrollo de software",
    "operationRegions":     "Lima, Arequipa",
    "contractTypeInterest": "Servicios, Consultoría",
    "maxContractValue":     500000,
    "strict_filters":       False
}

# Licitación de prueba para modo busqueda (hardcoded)
TENDER_BUSQUEDA = {
    "id_proceso":            "1094319",
    "objeto_contractual":    "Analista de datos para proyectos vinculados al desarrollo de la nueva plataforma de contratación pública",
    "entidad_convocante":    "ORGANISMO SUPERVISOR DE LAS CONTRATACIONES DEL ESTADO",
    "region_ejecucion":      "LIMA",
    "tipo_contrato_clasificado": "services",
    "metodo":                "Convenio",
    "monto_referencial":     30000,
    "affinity_score":        65.4,
    "documents": [
        {
            "url":    "https://prod1.seace.gob.pe/SeaceWeb-PRO/SdescargarArchivoAlfresco?fileCode=67f4446f-e0be-48f5-896b-9110959996e0",
            "title":  "Bases Administrativas",
            "format": "pdf"
        },
        {
            "url":    "https://prod1.seace.gob.pe/SeaceWeb-PRO/SdescargarArchivoAlfresco?fileCode=3dc36ab6-5c81-4c2e-86da-832325b3a4b5",
            "title":  "Documentos de Otorgamiento de Buena Pro",
            "format": "zip"
        },
        {
            "url":    "https://prod4.seace.gob.pe:9000/api/con/documentos/descargar/152849273",
            "title":  "Archivos de la ampliación del contrato",
            "format": "application/pdf"
        },
    ]
}


# =================================================================
# CARGAR RECOMENDACIONES REALES DESDE EL BACKEND
# =================================================================

def cargar_recomendaciones(user_id: int) -> list:
    """Llama al endpoint de recomendaciones y retorna la lista completa."""
    print(f"\n⏳ Cargando recomendaciones para user_id={user_id}...")
    try:
        response = requests.post(
            RECOMMENDATIONS_URL,
            json={"user_id": user_id},
            timeout=30
        )
        if not response.ok:
            print(f"❌ Error {response.status_code}: {response.text[:200]}")
            return []

        data = response.json()
        recs = data.get("recommendations", [])
        print(f"✅ {len(recs)} recomendaciones cargadas | Empresa: {data.get('company_name', 'N/A')}")
        print()

        # Mostrar resumen
        for i, r in enumerate(recs, 1):
            print(f"  #{i:2} [{r.get('affinity_score', 0):.1f}%] "
                  f"{str(r.get('objeto_contractual', 'N/A'))[:55]:55} "
                  f"| {r.get('region_ejecucion', 'N/A'):12} "
                  f"| S/ {r.get('monto_referencial', 0):,.0f}")
        print()
        return recs

    except requests.exceptions.ConnectionError:
        print(f"❌ No se puede conectar a {RECOMMENDATIONS_URL}")
        print("   ¿Está corriendo chat_engine.py?")
        return []
    except Exception as e:
        print(f"❌ Error cargando recomendaciones: {e}")
        return []


# =================================================================
# FUNCIÓN DE CHAT
# =================================================================

def chat(message: str, mode: str, user_id: int = None,
         tender_context: list = None, documents: list = None,
         chat_history: list = None, profile: dict = None):
    """Envía un mensaje al chat_engine y muestra la respuesta en streaming."""

    payload = {
        "message":        message,
        "mode":           mode,
        "profile":        profile or PROFILE_LEGACY,
        "tender_context": tender_context or [],
        "documents":      documents or [],
        "chat_history":   chat_history or [],
    }
    if user_id:
        payload["user_id"] = user_id

    print(f"\n{'─'*60}")
    print(f"👤 Tú [{mode.upper()}]: {message}")
    print(f"{'─'*60}")
    print(f"🤖 LiciBot: ", end="", flush=True)

    try:
        response = requests.post(
            CHAT_URL, json=payload, stream=True, timeout=60
        )
        if not response.ok:
            print(f"\n❌ Error HTTP {response.status_code}: {response.text}")
            return ""

        full_response = ""
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                text = chunk.decode("utf-8")
                print(text, end="", flush=True)
                full_response += text

        print()
        return full_response

    except requests.exceptions.ConnectionError:
        print(f"\n❌ No se puede conectar a {CHAT_URL}")
        print("   ¿Está corriendo chat_engine.py?")
        return ""
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return ""


# =================================================================
# MODOS
# =================================================================

def modo_general(user_id=None):
    print("\n" + "="*60)
    print("🤖 MODO GENERAL — Chat libre sobre licitaciones")
    print("   Escribe 'salir' para terminar")
    print("="*60)

    history = []
    while True:
        try:
            msg = input("\n👤 Tú: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if msg.lower() in ("salir", "exit", "quit", "q"):
            break
        if not msg:
            continue

        response = chat(message=msg, mode="general", user_id=user_id, chat_history=history)
        if response:
            history.append({"role": "user",     "content": msg})
            history.append({"role": "assistant", "content": response})


def modo_recomendacion(user_id=None):
    print("\n" + "="*60)
    print("🤖 MODO RECOMENDACIÓN — Análisis de licitaciones")
    print("   Escribe 'salir' para terminar")
    print("="*60)

    # Cargar recomendaciones reales si hay user_id
    recomendaciones = []
    licitacion_activa = None

    if user_id:
        recomendaciones = cargar_recomendaciones(user_id)

    if not recomendaciones:
        print("⚠️  Sin recomendaciones — usando licitación de prueba")
        recomendaciones = [TENDER_BUSQUEDA]

    licitacion_activa = recomendaciones[0]

    print(f"📋 Contexto inicial: Top {len(recomendaciones)} recomendaciones")
    print(f"   Seleccionada: #{1} {str(licitacion_activa.get('objeto_contractual',''))[:60]}")
    print(f"\nComandos especiales:")
    print(f"  /ver N     → selecciona la licitación #N como activa")
    print(f"  /lista     → muestra todas las licitaciones")
    print(f"  /todas     → analiza con todas en contexto")
    print(f"  salir      → terminar\n")

    history       = []
    usar_todas    = False

    while True:
        try:
            msg = input("\n👤 Tú: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if msg.lower() in ("salir", "exit", "quit", "q"):
            break
        if not msg:
            continue

        # Comandos especiales
        if msg.lower().startswith("/ver "):
            try:
                n = int(msg.split()[1]) - 1
                if 0 <= n < len(recomendaciones):
                    licitacion_activa = recomendaciones[n]
                    usar_todas = False
                    history = []  # reset historial al cambiar licitación
                    print(f"✅ Cambiado a #{n+1}: {str(licitacion_activa.get('objeto_contractual',''))[:60]}")
                    print(f"   Afinidad: {licitacion_activa.get('affinity_score', 0)}%")
                else:
                    print(f"❌ Número inválido. Elige entre 1 y {len(recomendaciones)}")
            except:
                print("Uso: /ver N (ej: /ver 3)")
            continue

        if msg.lower() == "/lista":
            print()
            for i, r in enumerate(recomendaciones, 1):
                marca = "→" if r == licitacion_activa and not usar_todas else " "
                print(f"  {marca} #{i:2} [{r.get('affinity_score',0):.1f}%] "
                      f"{str(r.get('objeto_contractual',''))[:55]:55} | S/ {r.get('monto_referencial',0):,.0f}")
            continue

        if msg.lower() == "/todas":
            usar_todas = True
            history = []
            print(f"✅ Contexto: todas las {len(recomendaciones)} licitaciones")
            continue

        # Decidir contexto para el chat
        if usar_todas:
            contexto = recomendaciones
        else:
            contexto = [licitacion_activa]

        response = chat(
            message=msg,
            mode="recomendacion",
            user_id=user_id,
            tender_context=contexto,
            chat_history=history
        )
        if response:
            history.append({"role": "user",     "content": msg})
            history.append({"role": "assistant", "content": response})


def modo_busqueda(user_id=None):
    print("\n" + "="*60)
    print("🤖 MODO BÚSQUEDA — Análisis técnico de licitación")
    print("   Escribe 'salir' para terminar")
    print("="*60)
    print(f"\n📋 Licitación: {TENDER_BUSQUEDA['objeto_contractual'][:60]}...")

    history = []
    while True:
        try:
            msg = input("\n👤 Tú: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if msg.lower() in ("salir", "exit", "quit", "q"):
            break
        if not msg:
            continue

        response = chat(
            message=msg, mode="busqueda", user_id=user_id,
            tender_context=[TENDER_BUSQUEDA],
            documents=TENDER_BUSQUEDA.get("documents", []),
            chat_history=history
        )
        if response:
            history.append({"role": "user",     "content": msg})
            history.append({"role": "assistant", "content": response})


def menu_interactivo():
    print("\n" + "="*60)
    print("🤖 LICIGOB CHAT — Prueba de terminal")
    print("="*60)
    print("\n¿Qué modo quieres probar?")
    print("  1. General       → chat libre sobre licitaciones")
    print("  2. Recomendación → analiza licitaciones recomendadas")
    print("  3. Búsqueda      → análisis técnico de una licitación")

    user_id = None
    uid = input("\n¿User ID? (Enter para omitir): ").strip()
    if uid.isdigit():
        user_id = int(uid)
        print(f"✅ Usando user_id={user_id}")
    else:
        print("ℹ️  Sin user_id — perfil de prueba")

    opcion = input("\nElige (1/2/3): ").strip()
    if opcion == "1":
        modo_general(user_id)
    elif opcion == "2":
        modo_recomendacion(user_id)
    elif opcion == "3":
        modo_busqueda(user_id)
    else:
        print("Opción no válida")


# =================================================================
# MAIN
# =================================================================

if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        menu_interactivo()
    elif args[0] == "general":
        uid = int(args[1]) if len(args) > 1 and args[1].isdigit() else None
        modo_general(uid)
    elif args[0] == "recomendacion":
        uid = int(args[1]) if len(args) > 1 and args[1].isdigit() else None
        modo_recomendacion(uid)
    elif args[0] == "busqueda":
        uid = int(args[1]) if len(args) > 1 and args[1].isdigit() else None
        modo_busqueda(uid)
    else:
        print(f"Modo '{args[0]}' no reconocido. Usa: general, recomendacion, busqueda")
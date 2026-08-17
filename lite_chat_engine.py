"""Lightweight, on-demand document chat service for LiciGob."""

from __future__ import annotations

import hmac
import json
import logging
import os
import time

import httpx
from flask import Flask, Response, jsonify, request
from openai import AzureOpenAI

from lite_rag_engine import LiteRagError, prepare_corpus, select_context, tender_summary


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("licigob-ai-lite")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024

SERVICE_KEY = os.getenv("LITE_SERVICE_KEY", "")
CHAT_MODEL = os.getenv("CHAT_DEPLOYMENT", "gpt-4o-mini")

http_client = httpx.Client(
    timeout=httpx.Timeout(90.0, connect=10.0),
    limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
)
openai_client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    azure_endpoint=os.getenv("OPENAI_API_BASE"),
    api_version=os.getenv("OPENAI_API_VERSION", "2024-12-01-preview"),
    http_client=http_client,
    max_retries=2,
)


def _authorized() -> bool:
    supplied = request.headers.get("X-Service-Key", "")
    return bool(SERVICE_KEY and supplied and hmac.compare_digest(SERVICE_KEY, supplied))


def _extract_tender_id(data: dict) -> str:
    context = data.get("context") or {}
    value = (
        data.get("tender_id")
        or context.get("id_proceso")
        or context.get("tender_id")
        or context.get("tenderId")
        or context.get("id")
    )
    return str(value or "").strip()


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "service": "licigob-ai-lite",
            "version": "1.0.0",
            "model": CHAT_MODEL,
            "mode": "pdf-text-on-demand",
        }
    )


@app.post("/api/v1/inspect")
def inspect_document():
    if not _authorized():
        return jsonify({"status": "error", "code": "unauthorized"}), 401
    data = request.get_json(silent=True) or {}
    tender_id = _extract_tender_id(data)
    if not tender_id:
        return jsonify({"status": "error", "code": "missing_tender_id"}), 400
    try:
        started = time.monotonic()
        _, corpus = prepare_corpus(tender_id)
        return jsonify(
            {
                "status": "ready",
                "tender_id": tender_id,
                "documents": corpus.document_count,
                "pages": corpus.total_pages,
                "characters": corpus.total_chars,
                "cache_hit": corpus.cache_hit,
                "scan_suspected": corpus.scan_suspected,
                "elapsed_ms": round((time.monotonic() - started) * 1000),
            }
        )
    except LiteRagError as exc:
        return jsonify({"status": "error", "code": exc.code, "message": str(exc)}), 422


@app.post("/api/v1/chat_stream")
def chat_stream():
    if not _authorized():
        return jsonify({"status": "error", "code": "unauthorized"}), 401

    data = request.get_json(silent=True) or {}
    tender_id = _extract_tender_id(data)
    message = str(data.get("message") or "").strip()
    history = data.get("chat_history") or []
    if not tender_id or not message:
        return jsonify({"status": "error", "code": "invalid_request"}), 400

    def generate():
        started = time.monotonic()
        try:
            yield "[STATUS]Preparando documentos PDF...\n"
            tender, corpus = prepare_corpus(tender_id)
            context, references = select_context(corpus, message)
            cache_label = "cache" if corpus.cache_hit else "download"
            logger.info(
                "document_ready tender=%s source=%s docs=%s pages=%s chars=%s elapsed_ms=%s",
                tender_id,
                cache_label,
                corpus.document_count,
                corpus.total_pages,
                corpus.total_chars,
                round((time.monotonic() - started) * 1000),
            )

            system_prompt = f"""Eres el asistente documental de LiciGob para contrataciones publicas peruanas.
Responde exclusivamente con la informacion de la licitacion y los extractos proporcionados.
Si el dato no aparece, indicalo claramente. No inventes requisitos, montos ni fechas.
Cita cada afirmacion documental como [Documento, pagina N].
Se breve, preciso y utiliza espanol profesional.

LICITACION:
{tender_summary(tender)}

EXTRACTOS SELECCIONADOS:
{context}
"""
            messages = [{"role": "system", "content": system_prompt}]
            for item in history[-4:]:
                role = item.get("role")
                if role in {"user", "assistant"}:
                    messages.append({"role": role, "content": str(item.get("content") or "")[:1000]})
            messages.append({"role": "user", "content": message[:2000]})

            yield "[STATUS]Analizando las paginas relevantes...\n"
            stream = openai_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                stream=True,
                temperature=0.1,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
            logger.info(
                "chat_complete tender=%s references=%s elapsed_ms=%s",
                tender_id,
                len(references),
                round((time.monotonic() - started) * 1000),
            )
        except LiteRagError as exc:
            logger.warning("document_unavailable tender=%s code=%s", tender_id, exc.code)
            if exc.code == "ocr_required":
                yield "Este PDF parece escaneado y requiere OCR. Esta primera version procesa PDF con texto seleccionable."
            elif exc.code == "no_pdf":
                yield "Esta licitacion no tiene documentos PDF compatibles para analizar."
            else:
                yield "No pudimos preparar los documentos en este momento. Intenta nuevamente en unos minutos."
        except Exception:
            logger.exception("chat_failed tender=%s", tender_id)
            yield "No pudimos completar el analisis documental. Intenta nuevamente."

    return Response(generate(), mimetype="text/plain; charset=utf-8")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5008")))

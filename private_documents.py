"""Bounded, process-local user PDF sessions. Never writes shared document caches."""
from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass

from lite_rag_engine import Corpus, LiteRagError, MAX_TOTAL_CHARS, MAX_TOTAL_PAGES, _extract_pdf

SESSION_SECONDS = 1800
MAX_SESSIONS = 12
MAX_PER_USER = 2
UPLOAD_INTERVAL = 30
UPLOAD_SLOT = threading.BoundedSemaphore(1)
_lock = threading.Lock()
_sessions: dict[str, 'PrivateDocument'] = {}
_attempts: dict[str, float] = {}


@dataclass
class PrivateDocument:
    user_id: str
    tender_id: str
    expires: float
    corpus: Corpus
    warning: str


def reserve_upload(user_id: str) -> None:
    now = time.monotonic()
    with _lock:
        for key, previous in list(_attempts.items()):
            if now - previous >= UPLOAD_INTERVAL:
                _attempts.pop(key, None)
        if user_id in _attempts or len(_attempts) >= 256:
            raise LiteRagError('upload_rate_limited', 'Espera 30 segundos antes de aportar otro PDF.')
        _attempts[user_id] = now


def cleanup_expired() -> None:
    now = time.monotonic()
    with _lock:
        for token, value in list(_sessions.items()):
            if value.expires <= now:
                _sessions.pop(token, None)


def create_document(user_id: str, tender_id: str, path: str) -> dict:
    # No caller-supplied filename, URL or document title becomes trusted provenance.
    pages, reviewed_pages, _ = _extract_pdf(path, 'PDF aportado por ti', MAX_TOTAL_PAGES, MAX_TOTAL_CHARS)
    if not pages or sum(len(p.text.strip()) for p in pages) < 100:
        raise LiteRagError('scanned_pdf', 'El PDF no tiene suficiente texto seleccionable. Este modo no incluye OCR.')
    import fitz
    with fitz.open(path) as pdf:
        total_pages = len(pdf)
    chars = sum(len(p.text) for p in pages)
    partial = reviewed_pages < total_pages or len(pages) < total_pages or chars >= MAX_TOTAL_CHARS
    warning = ('Documento aportado por ti; no se ha verificado que corresponda a la version oficial de esta licitacion. '
               'La sesion es privada y temporal (30 minutos; puede finalizar si el servicio se reinicia).')
    if partial:
        warning += ' Lectura parcial: algunas paginas o contenido no se pudieron incorporar.'
    token = secrets.token_urlsafe(32)
    corpus = Corpus(tender_id=tender_id, source_digest='private-session', pages=pages,
                    document_count=1, total_pages=len(pages), total_chars=chars,
                    scan_suspected=partial, cache_hit=True)
    now = time.monotonic()
    with _lock:
        for key, value in list(_sessions.items()):
            if value.expires <= now:
                _sessions.pop(key, None)
        own = sorted((key for key, value in _sessions.items() if value.user_id == user_id),
                     key=lambda key: _sessions[key].expires)
        while len(own) >= MAX_PER_USER:
            _sessions.pop(own.pop(0), None)
        while len(_sessions) >= MAX_SESSIONS:
            _sessions.pop(min(_sessions, key=lambda key: _sessions[key].expires), None)
        _sessions[token] = PrivateDocument(user_id, tender_id, now + SESSION_SECONDS, corpus, warning)
    return {'status': 'ready', 'source': 'private_upload', 'document_session_id': token,
            'expires_in': SESSION_SECONDS, 'pages': len(pages), 'partial': partial, 'warning': warning}


def get_document(token: str, user_id: str, tender_id: str) -> PrivateDocument:
    with _lock:
        value = _sessions.get(token)
        if value and value.expires <= time.monotonic():
            _sessions.pop(token, None)
            value = None
        if not value or value.user_id != str(user_id) or value.tender_id != str(tender_id):
            raise LiteRagError('private_document_expired', 'El PDF temporal no esta disponible. Vuelve a aportarlo para continuar.')
        return value


def delete_document(token: str, user_id: str, tender_id: str) -> None:
    with _lock:
        value = _sessions.get(token)
        if value and value.user_id == str(user_id) and value.tender_id == str(tender_id):
            _sessions.pop(token, None)

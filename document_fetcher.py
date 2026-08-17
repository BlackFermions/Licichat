"""Secure, bounded downloads for public SEACE documents."""

from __future__ import annotations

import html
import os
import re
import uuid
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import requests


SEACE_USER_AGENT = os.getenv(
    "SEACE_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36",
)
SEACE_REFERER = os.getenv("SEACE_REFERER", "https://prod1.seace.gob.pe/portal/")
SEACE_DOCUMENT_PROXY_URL = os.getenv("SEACE_DOCUMENT_PROXY_URL", "").strip()
SEACE_DOCUMENT_PROXY_KEY = os.getenv("SEACE_DOCUMENT_PROXY_KEY", "").strip()
ALLOWED_HOSTS = {
    host.strip().lower()
    for host in os.getenv(
        "SEACE_ALLOWED_HOSTS",
        "prod1.seace.gob.pe,prod2.seace.gob.pe,prod3.seace.gob.pe,prod4.seace.gob.pe",
    ).split(",")
    if host.strip()
}

_FILE_CODE_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

_MAGIC = {
    ".pdf": (b"%PDF-",),
    ".zip": (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"),
    ".rar": (b"Rar!\x1a\x07",),
}


class DocumentDownloadError(RuntimeError):
    """Expected, safe-to-report document download failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def normalize_document_url(raw_url: str) -> str:
    """Normalize a stored URL and reject untrusted download destinations."""
    value = html.unescape(str(raw_url or "")).strip()
    parsed = urlparse(value)

    if parsed.scheme not in {"http", "https"}:
        raise DocumentDownloadError("invalid_url", "La URL del documento no es valida.")

    host = (parsed.hostname or "").lower()
    if host not in ALLOWED_HOSTS:
        raise DocumentDownloadError("untrusted_host", "El origen del documento no esta permitido.")

    path = parsed.path.rstrip("/")
    if not path.endswith("/SdescargarArchivoAlfresco"):
        raise DocumentDownloadError("invalid_path", "La ruta de descarga de SEACE no es valida.")

    query = parse_qs(parsed.query, keep_blank_values=False)
    file_code = (query.get("fileCode") or query.get("filecode") or [""])[0].strip()
    if not _FILE_CODE_RE.fullmatch(file_code):
        raise DocumentDownloadError("invalid_file_code", "El identificador del documento no es valido.")

    return urlunparse(("https", host, path, "", urlencode({"fileCode": file_code}), ""))


def _validate_magic(path: Path, suffix: str) -> None:
    expected = _MAGIC.get(suffix.lower())
    if not expected:
        return
    with path.open("rb") as handle:
        prefix = handle.read(8)
    if not any(prefix.startswith(signature) for signature in expected):
        raise DocumentDownloadError(
            "unexpected_content",
            "SEACE no devolvio el tipo de archivo esperado.",
        )


def _proxy_config() -> tuple[str, str] | None:
    if not SEACE_DOCUMENT_PROXY_URL:
        return None
    parsed = urlparse(SEACE_DOCUMENT_PROXY_URL)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.fragment
        or not SEACE_DOCUMENT_PROXY_KEY
    ):
        raise DocumentDownloadError(
            "invalid_proxy_config",
            "La configuracion del proxy de documentos no es valida.",
        )
    return SEACE_DOCUMENT_PROXY_URL, parsed.hostname.lower()


def download_document(
    raw_url: str,
    temp_dir: str,
    *,
    suffix: str = ".pdf",
    max_bytes: int = 50 * 1024 * 1024,
    timeout: tuple[int, int] = (8, 35),
) -> str:
    """Download a SEACE document to a temporary path with strict limits."""
    url = normalize_document_url(raw_url)
    headers = {
        "User-Agent": SEACE_USER_AGENT,
        "Referer": SEACE_REFERER,
        "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.5",
        "Accept-Language": "es-PE,es;q=0.9",
        "Connection": "keep-alive",
    }
    target = Path(temp_dir) / f"seace_{uuid.uuid4().hex}{suffix.lower()}"

    try:
        with requests.Session() as session:
            response = session.get(
                url,
                headers=headers,
                stream=True,
                timeout=timeout,
                allow_redirects=True,
            )
            if response.status_code == 403:
                response.close()
                proxy = _proxy_config()
                if not proxy:
                    raise DocumentDownloadError("seace_forbidden", "SEACE rechazo la descarga.")
                proxy_url, proxy_host = proxy
                response = session.post(
                    proxy_url,
                    headers={
                        "User-Agent": SEACE_USER_AGENT,
                        "Accept": "application/pdf,application/octet-stream",
                        "Content-Type": "application/json",
                        "X-Proxy-Key": SEACE_DOCUMENT_PROXY_KEY,
                    },
                    json={"url": url},
                    stream=True,
                    timeout=timeout,
                    allow_redirects=False,
                )
                if response.status_code in {401, 403}:
                    raise DocumentDownloadError(
                        "proxy_forbidden",
                        "El proxy de documentos rechazo la descarga.",
                    )
                final_host = (urlparse(response.url).hostname or "").lower()
                if final_host != proxy_host:
                    raise DocumentDownloadError(
                        "unsafe_proxy_redirect",
                        "El proxy redirigio a un origen no permitido.",
                    )
            else:
                final_host = (urlparse(response.url).hostname or "").lower()
                if final_host not in ALLOWED_HOSTS:
                    raise DocumentDownloadError(
                        "unsafe_redirect",
                        "SEACE redirigio a un origen no permitido.",
                    )
            response.raise_for_status()

            declared_size = int(response.headers.get("Content-Length") or 0)
            if declared_size > max_bytes:
                raise DocumentDownloadError("file_too_large", "El documento supera el limite permitido.")

            written = 0
            with target.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    written += len(chunk)
                    if written > max_bytes:
                        raise DocumentDownloadError("file_too_large", "El documento supera el limite permitido.")
                    handle.write(chunk)

        if not target.exists() or target.stat().st_size == 0:
            raise DocumentDownloadError("empty_file", "SEACE devolvio un archivo vacio.")
        _validate_magic(target, suffix)
        return str(target)
    except DocumentDownloadError:
        target.unlink(missing_ok=True)
        raise
    except requests.Timeout as exc:
        target.unlink(missing_ok=True)
        raise DocumentDownloadError("download_timeout", "La descarga de SEACE excedio el tiempo limite.") from exc
    except requests.RequestException as exc:
        target.unlink(missing_ok=True)
        raise DocumentDownloadError("download_failed", "No se pudo descargar el documento de SEACE.") from exc
    except Exception:
        target.unlink(missing_ok=True)
        raise

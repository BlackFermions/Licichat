# -*- coding: utf-8 -*-

import os
import tempfile
import shutil
import time
import zipfile
import psycopg2
import requests
import numpy as np
from datetime import datetime
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Tuple
from dotenv import load_dotenv

load_dotenv()

# PyMuPDF
try:
    import fitz
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False
    print("⚠️  PyMuPDF no instalado: pip install pymupdf")

# rarfile
try:
    import rarfile
    RAR_AVAILABLE = True
except ImportError:
    RAR_AVAILABLE = False

# =================================================================
# CONFIG
# =================================================================

CHUNK_SIZE      = 800
MAX_PDF_SIZE    = 8 * 1024 * 1024   # 8MB por PDF
MAX_PDFS        = 3                  # máximo PDFs a procesar
REQUEST_TIMEOUT = 30

FORMATS_PDF     = {"pdf", "application/pdf"}
FORMATS_ZIP     = {"zip", "application/zip", "application/x-zip-compressed"}
FORMATS_RAR     = {"rar", "application/x-rar-compressed", "application/vnd.rar"}
FORMATS_SKIP    = {"xls", "xlsx", "doc", "docx", "jpg", "jpeg", "png",
                   "application/msword", "crdownload", "7z",
                   "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}

DB_CONFIG = {
    "host":     os.getenv("DB_HOST"),
    "user":     os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "port":     int(os.getenv("DB_PORT", "5432")),
    "sslmode":  os.getenv("DB_SSLMODE", "require"),
}


# =================================================================
# BD
# =================================================================

def _get_conn():
    return psycopg2.connect(**DB_CONFIG)


def chunks_en_cache(tender_id: str) -> List[str]:
    """Retorna chunks guardados en BD o [] si no hay."""
    try:
        conn = _get_conn()
        cur  = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT texto FROM tender_chunks
            WHERE tender_id = %s
            ORDER BY chunk_index
        """, (tender_id,))
        rows = cur.fetchall()
        conn.close()
        if rows:
            print(f"   ⚡ {len(rows)} chunks en caché (tender_id={tender_id})")
            return [r["texto"] for r in rows]
        return []
    except Exception as e:
        print(f"   ⚠️  Error leyendo caché chunks: {e}")
        return []


def guardar_chunks(tender_id: str, chunks: List[str], embed_fn):
    """Guarda chunks con embeddings en tender_chunks."""
    if not chunks:
        return
    try:
        conn = _get_conn()
        cur  = conn.cursor()

        # Limpiar chunks anteriores de este tender
        cur.execute("DELETE FROM tender_chunks WHERE tender_id = %s", (tender_id,))

        guardados = 0
        for i, chunk in enumerate(chunks):
            vector = embed_fn(chunk)
            if vector is None:
                continue
            vector_str = "[" + ",".join(str(x) for x in vector.tolist()) + "]"
            cur.execute("""
                INSERT INTO tender_chunks (tender_id, chunk_index, texto, embedding)
                VALUES (%s, %s, %s, %s)
            """, (tender_id, i, chunk, vector_str))
            guardados += 1

        # Guardar también en tender_documents_text para referencia
        texto_completo = "\n\n".join(chunks)
        cur.execute("""
            INSERT INTO tender_documents_text (tender_id, texto, paginas, procesado_at, tiene_ocr)
            VALUES (%s, %s, %s, %s, FALSE)
            ON CONFLICT (tender_id) DO UPDATE
               SET texto = EXCLUDED.texto, procesado_at = EXCLUDED.procesado_at
        """, (tender_id, texto_completo, 0, datetime.now()))

        conn.commit()
        conn.close()
        print(f"   ✅ {guardados} chunks guardados en pgvector")
    except Exception as e:
        print(f"   ⚠️  Error guardando chunks: {e}")


def buscar_chunks_pgvector(tender_id: str, query_vector: np.ndarray,
                           k: int = 4) -> List[str]:
    """Búsqueda semántica de chunks con pgvector."""
    try:
        conn = _get_conn()
        cur  = conn.cursor(cursor_factory=RealDictCursor)
        vector_str = "[" + ",".join(str(x) for x in query_vector.tolist()) + "]"
        cur.execute("""
            SELECT texto,
                   1 - (embedding <=> %s::vector) AS score
            FROM tender_chunks
            WHERE tender_id = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (vector_str, tender_id, vector_str, k))
        rows = cur.fetchall()
        conn.close()
        return [r["texto"] for r in rows if r["score"] > 0.3]
    except Exception as e:
        print(f"   ⚠️  Error búsqueda pgvector: {e}")
        return []


# =================================================================
# EXTRACCIÓN DE TEXTO
# =================================================================

def extract_text_from_pdf(pdf_path: str) -> Tuple[str, int]:
    """Extrae texto de PDF. Retorna (texto, paginas)."""
    if not FITZ_AVAILABLE:
        return "", 0
    texto, paginas = "", 0
    try:
        doc     = fitz.open(pdf_path)
        paginas = len(doc)
        print(f"    📄 {paginas} páginas")
        for page in doc:
            texto += page.get_text()
        doc.close()
        if texto.strip():
            print(f"    ✅ {len(texto):,} chars extraídos")
        else:
            print(f"    ⚠️  PDF escaneado (sin texto)")
    except Exception as e:
        print(f"    ❌ Error PDF: {e}")
    return texto, paginas


def download_file(url: str, temp_dir: str, suffix: str = ".pdf") -> str | None:
    """Descarga un archivo. Retorna ruta local o None."""
    try:
        print(f"    🌐 {url[:70]}...")
        t = time.time()

        # Verificar tamaño antes de descargar
        try:
            head = requests.head(url, timeout=8, allow_redirects=True)
            size = int(head.headers.get("Content-Length", 0))
            if size > MAX_PDF_SIZE:
                print(f"    ⚠️  Archivo muy grande ({size/1024/1024:.1f}MB) — omitido")
                return None
        except Exception:
            pass

        resp = requests.get(url, stream=True, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()

        path = os.path.join(temp_dir, f"doc_{int(time.time()*1000)}{suffix}")
        size = 0
        with open(path, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
                size += len(chunk)
                if size > MAX_PDF_SIZE:
                    print(f"    ⚠️  Archivo demasiado grande — abortando")
                    return None

        print(f"    💾 {size/1024:.0f}KB en {time.time()-t:.1f}s")
        return path

    except requests.exceptions.Timeout:
        print(f"    ⏱️  Timeout")
        return None
    except Exception as e:
        print(f"    ❌ {e}")
        return None


def extraer_pdfs_de_archivo(archive_path: str, temp_dir: str) -> List[str]:
    """
    Descomprime ZIP o RAR y retorna rutas de PDFs encontrados,
    ordenados por tamaño (más ligeros primero).
    """
    extract_dir = archive_path + "_ext"
    pdfs        = []
    try:
        os.makedirs(extract_dir, exist_ok=True)
        ext = archive_path.lower()

        if ext.endswith(".zip"):
            print(f"    📦 Descomprimiendo ZIP...")
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(extract_dir)
        elif ext.endswith(".rar") and RAR_AVAILABLE:
            print(f"    📦 Descomprimiendo RAR...")
            with rarfile.RarFile(archive_path, "r") as rf:
                rf.extractall(extract_dir)
        else:
            return []

        # Buscar PDFs y ordenar por tamaño
        for root, _, files in os.walk(extract_dir):
            for f in files:
                if f.lower().endswith(".pdf"):
                    full_path = os.path.join(root, f)
                    pdfs.append((os.path.getsize(full_path), full_path))

        pdfs.sort()  # más ligeros primero
        print(f"    📄 {len(pdfs)} PDFs encontrados en archivo comprimido")
        return [p[1] for p in pdfs]

    except Exception as e:
        print(f"    ❌ Error descomprimiendo: {e}")
        return []
    finally:
        # No limpiar extract_dir aquí — lo limpia el llamador
        pass


def procesar_documentos(documents: List[Dict], temp_dir: str) -> Tuple[str, int]:
    """
    Procesa lista de documentos y extrae texto.
    Prioridad: PDF > ZIP > RAR
    Retorna (texto_completo, paginas_total)
    """
    textos  = []
    paginas = 0
    pdfs_procesados = 0

    # Clasificar documentos
    pdf_docs  = []
    zip_docs  = []
    rar_docs  = []

    for doc in documents:
        fmt = (doc.get("format") or "").lower().strip()
        url = doc.get("url", "").lower()

        if fmt in FORMATS_SKIP:
            continue
        if fmt in FORMATS_PDF or url.endswith(".pdf"):
            pdf_docs.append(doc)
        elif fmt in FORMATS_ZIP or url.endswith(".zip"):
            zip_docs.append(doc)
        elif fmt in FORMATS_RAR or url.endswith(".rar"):
            rar_docs.append(doc)

    total_pdfs  = len(pdf_docs)
    total_zips  = len(zip_docs)
    total_rars  = len(rar_docs)
    total_skip  = len(documents) - total_pdfs - total_zips - total_rars
    print(f"\n📂 {total_pdfs} PDFs | {total_zips} ZIPs | {total_rars} RARs | {total_skip} ignorados")

    # ── 1. Procesar PDFs directos ─────────────────────────────────
    for doc in pdf_docs:
        if pdfs_procesados >= MAX_PDFS:
            break
        title = doc.get("title") or f"PDF {pdfs_procesados+1}"
        print(f"\n  📄 [{pdfs_procesados+1}] {title[:60]}")
        path = download_file(doc["url"], temp_dir, ".pdf")
        if not path:
            continue
        texto, npags = extract_text_from_pdf(path)
        try: os.remove(path)
        except: pass
        if texto and len(texto.strip()) > 100:
            textos.append(f"=== {title} ===\n{texto.strip()}")
            paginas += npags
            pdfs_procesados += 1

    # ── 2. Procesar ZIPs si faltan PDFs ──────────────────────────
    if pdfs_procesados < MAX_PDFS and zip_docs:
        for doc in zip_docs[:2]:
            if pdfs_procesados >= MAX_PDFS:
                break
            title = doc.get("title") or "Archivo ZIP"
            print(f"\n  📦 ZIP: {title[:60]}")
            path = download_file(doc["url"], temp_dir, ".zip")
            if not path:
                continue
            extract_dir = path + "_ext"
            pdfs_en_zip = extraer_pdfs_de_archivo(path, temp_dir)
            try: os.remove(path)
            except: pass
            for pdf_path in pdfs_en_zip[:2]:
                if pdfs_procesados >= MAX_PDFS:
                    break
                texto, npags = extract_text_from_pdf(pdf_path)
                if texto and len(texto.strip()) > 100:
                    fname = os.path.basename(pdf_path)
                    textos.append(f"=== {title} / {fname} ===\n{texto.strip()}")
                    paginas += npags
                    pdfs_procesados += 1
            shutil.rmtree(extract_dir, ignore_errors=True)

    # ── 3. Procesar RARs si aún faltan PDFs ──────────────────────
    if pdfs_procesados < MAX_PDFS and rar_docs and RAR_AVAILABLE:
        for doc in rar_docs[:1]:
            if pdfs_procesados >= MAX_PDFS:
                break
            title = doc.get("title") or "Archivo RAR"
            print(f"\n  📦 RAR: {title[:60]}")
            path = download_file(doc["url"], temp_dir, ".rar")
            if not path:
                continue
            extract_dir = path + "_ext"
            pdfs_en_rar = extraer_pdfs_de_archivo(path, temp_dir)
            try: os.remove(path)
            except: pass
            for pdf_path in pdfs_en_rar[:2]:
                if pdfs_procesados >= MAX_PDFS:
                    break
                texto, npags = extract_text_from_pdf(pdf_path)
                if texto and len(texto.strip()) > 100:
                    fname = os.path.basename(pdf_path)
                    textos.append(f"=== {title} / {fname} ===\n{texto.strip()}")
                    paginas += npags
                    pdfs_procesados += 1
            shutil.rmtree(extract_dir, ignore_errors=True)

    texto_final = "\n\n".join(textos)
    print(f"\n📊 Total: {len(texto_final):,} chars | {paginas} páginas | {pdfs_procesados} docs")
    return texto_final, paginas


# =================================================================
# API PÚBLICA
# =================================================================

def build_index(documents: List[Dict], embed_fn,
                tender_id: str = None) -> Tuple[object, List[str]]:
    """
    Prepara el contexto RAG para una licitación.

    Flujo:
      1. Si tender_id → buscar chunks en pgvector (caché)
      2. Si no hay caché → descargar docs, extraer texto
      3. Guardar chunks + embeddings en pgvector
      4. Retorna (tender_id, chunks) — sin FAISS

    Returns:
      ("pgvector", chunks) si hay datos
      (None, [])           si falla
    """
    # 1. Buscar en caché
    if tender_id:
        chunks_cached = chunks_en_cache(tender_id)
        if chunks_cached:
            return ("pgvector", tender_id), chunks_cached

    # 2. Procesar documentos
    if not documents:
        print("   ⚠️  Sin documentos")
        return None, []

    temp_dir = tempfile.mkdtemp()
    try:
        texto, paginas = procesar_documentos(documents, temp_dir)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    if not texto or len(texto.strip()) < 100:
        print("   ⚠️  Sin texto suficiente")
        return None, []

    # 3. Dividir en chunks
    chunks = []
    for i in range(0, len(texto), CHUNK_SIZE):
        chunk = texto[i:i + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)

    # 4. Guardar en pgvector
    if tender_id and embed_fn:
        print(f"\n💾 Guardando {len(chunks)} chunks en pgvector...")
        guardar_chunks(tender_id, chunks, embed_fn)

    return ("pgvector", tender_id), chunks


def search(index_obj, chunks: List[str], query: str,
           embed_fn, k: int = 4) -> List[str]:
    """
    Búsqueda semántica.
    Usa pgvector si hay tender_id, fallback a búsqueda lineal.
    """
    if not chunks:
        return []

    # Obtener embedding de la query
    qv = embed_fn(query)
    if qv is None:
        return chunks[:k]

    # Si tenemos índice pgvector
    if (isinstance(index_obj, tuple) and
            len(index_obj) == 2 and
            index_obj[0] == "pgvector" and
            index_obj[1]):
        tender_id = index_obj[1]
        resultados = buscar_chunks_pgvector(tender_id, qv, k)
        if resultados:
            return resultados
        # Fallback: búsqueda lineal sobre chunks en memoria
        print("   ↩️  Fallback búsqueda lineal")

    # Búsqueda lineal (fallback sin pgvector)
    scores = []
    for i, chunk in enumerate(chunks):
        cv = embed_fn(chunk)
        if cv is not None:
            score = float(np.dot(qv, cv) / (np.linalg.norm(qv) * np.linalg.norm(cv) + 1e-8))
            scores.append((score, i))

    scores.sort(reverse=True)
    return [chunks[i] for _, i in scores[:k]]
"""Pure evidence selection: bounded, paragraph-safe, provenance preserving."""

import re
import unicodedata


def normalize(text):
    return "".join(c for c in unicodedata.normalize("NFKD", str(text or "").lower())
                   if not unicodedata.combining(c))


def exhaustive_question(question):
    text = normalize(question)
    return any(term in text for term in ("todos los", "todas las", "lista completa", "exhaustiv", "checklist"))


def validate_citations(text, references, evidence=()):
    invalid = 0

    def title_key(value):
        return re.sub(r"^\d+\s*", "", normalize(value).strip())

    def page_range(value):
        parts = [int(p) for p in re.split(r"\s*[-–]\s*", str(value))]
        return parts[0], parts[-1]

    def check(match):
        nonlocal invalid
        document, pages = match.group(1).strip(), match.group(2)
        candidates = []
        for ref in references:
            if title_key(document) != title_key(ref["document"]):
                continue
            try:
                start, end = page_range(pages)
                allowed_start, allowed_end = page_range(ref["page"])
            except (ValueError, TypeError):
                continue
            if allowed_start <= start <= end <= allowed_end:
                candidates.append((allowed_end - allowed_start, ref))
        if candidates and len({ref['document'] for _, ref in candidates}) == 1:
            # Do not invent narrower page precision than the stored chunk range.
            ref = min(candidates, key=lambda item: item[0])[1]
            return f"[{ref['document']}, p. {ref['page']}]"
        invalid += 1
        return "(referencia no verificada)"

    pattern = r"\[([^\[\]\n]+?),\s*p(?:p)?\.?\s*(\d+(?:\s*[-–]\s*\d+)?)\]"
    # Normalise only the citation range; never rewrite factual numbers in text.
    text = re.sub(pattern, check, text, flags=re.I)
    by_id = {row['id']: row for row in evidence if row.get('id')}

    def source_id(match):
        nonlocal invalid
        ref = by_id.get(match.group(1).upper())
        if not ref:
            invalid += 1
            return '(referencia no verificada)'
        return f"[{ref['document']}, p. {ref['page']}]"

    text = re.sub(r'\[(E\d+)\]', source_id, text, flags=re.I)
    return text, invalid


def select_candidates(rows, terms, phrases, roles):
    """Fuse rank positions, not an arbitrary lexical subtraction from cosine.

    Keep distinct archive members and balance both evidence families for a mixed
    question. No text is rewritten and no citation page is inferred here.
    """
    unique = {}
    for row in rows:
        key = (row.get("asset_id"), row.get("chunk_index"), row.get("content"))
        unique.setdefault(key, dict(row))
    rows = list(unique.values())
    for row in rows:
        text = normalize(row.get("content"))
        row["_terms"] = sum(term in text for term in terms)
        row["_phrases"] = sum(normalize(phrase) in text for phrase in phrases)
    semantic = sorted(rows, key=lambda r: float(r.get("distance") or 0))
    lexical = sorted(rows, key=lambda r: (r["_terms"], r["_phrases"]), reverse=True)
    ranks = {id(row): index for index, row in enumerate(semantic, 1)}
    lexical_ranks = {id(row): index for index, row in enumerate(lexical, 1)}
    for row in rows:
        row["_score"] = (1 / (60 + ranks[id(row)]) + 1 / (60 + lexical_ranks[id(row)])
                         + min(row["_phrases"], 3) * .006)
    ordered = sorted(rows, key=lambda r: r["_score"], reverse=True)
    if len(roles) > 1:
        buckets = [[r for r in ordered if r.get("document_role") == role] for role in roles]
        balanced = []
        for index in range(max((len(b) for b in buckets), default=0)):
            balanced.extend(bucket[index] for bucket in buckets if index < len(bucket))
        return balanced
    return ordered


def safe_excerpt(text, limit, terms=()):
    """Do not cut a paragraph/condition halfway just to fill the context budget.

    For an oversized paragraph, complete sentences are eligible. If none fits,
    omit it rather than presenting a misleading prefix as complete evidence.
    """
    text = str(text).strip()
    if len(text) <= limit:
        return text, False
    paragraphs = re.split(r"\n\s*\n", text)
    pieces = []
    for paragraph in paragraphs:
        if len(paragraph) > limit:
            pieces.extend(re.split(r"(?<=[.!?;])\s+(?=[A-ZÁÉÍÓÚÑ0-9¿])", paragraph))
        else:
            pieces.append(paragraph)
    ranked = sorted(enumerate(pieces), key=lambda item: (
        sum(term in normalize(item[1]) for term in terms), -item[0]), reverse=True)
    selected, used = [], 0
    for index, piece in ranked:
        if piece.strip() and used + len(piece) + 2 <= limit:
            selected.append((index, piece))
            used += len(piece) + 2
    return "\n\n".join(piece for _, piece in sorted(selected)), True


def assemble_context(rows, budget, hint, terms=(), excerpt_limit=4500):
    parts, references, evidence = [], [], []
    used = 0
    truncated = False
    fingerprints = set()
    for row in rows:
        raw = str(row.get("content") or "")
        member = re.match(r"\[Archivo interno: ([^\]\r\n]+)\]\s*", raw)
        title = (member.group(1) if member else row.get("source_title") or "Documento")[:160].replace("\n", " ")
        text = raw[member.end():] if member else raw
        fingerprint = (row.get("asset_id"), title, " ".join(text.split()))
        if fingerprint in fingerprints:
            continue
        fingerprints.add(fingerprint)
        start, end = row.get("page_start"), row.get("page_end")
        page = str(start) if start == end else f"{start}-{end}"
        container = str(row.get("source_title") or "Documento")[:160].replace("\n", " ")
        note = f" | Contenedor: {container}" if title != container else ""
        evidence_id = f'E{len(parts) + 1}'
        header = f"[Documento: {title}{note} | Paginas: {page} | Referencia: {evidence_id} | Senal de clasificacion: {hint(text)}]\n"
        available = min(excerpt_limit, budget - used - len(header) - 100)
        if available < 200:
            truncated = True
            break
        excerpt, omitted = safe_excerpt(text, available, terms)
        truncated |= omitted
        if not excerpt:
            continue
        if omitted:
            excerpt += "\n[Extracto discontinuo: se omitieron parrafos completos.]"
        parts.append(header + excerpt)
        used += len(parts[-1]) + 2
        reference = {"document": title, "page": page}
        if reference not in references:
            references.append(reference)
        evidence.append({"id": evidence_id, "asset_id": row.get("asset_id"), "chunk_index": row.get("chunk_index"),
                         "document_role": row.get("document_role"), **reference})
    return "\n\n".join(parts), references, evidence, truncated

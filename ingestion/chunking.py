"""Page-aware chunks suitable for citation and pgvector retrieval."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from ingestion.models import ExtractedPage, TextChunk


@dataclass(frozen=True)
class _Piece:
    text: str
    page: int
    section: str | None
    document: str


def _looks_like_heading(line: str) -> bool:
    stripped = line.strip().rstrip(":")
    if not 4 <= len(stripped) <= 140:
        return False
    letters = [char for char in stripped if char.isalpha()]
    uppercase_ratio = sum(char.isupper() for char in letters) / max(len(letters), 1)
    return uppercase_ratio >= 0.75 or line.strip().endswith(":")


def _split_long_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r"(?<=[.!?;:])\s+", text)
    output: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                output.append(current)
                current = ""
            output.extend(sentence[index : index + max_chars] for index in range(0, len(sentence), max_chars))
        elif not current:
            current = sentence
        elif len(current) + len(sentence) + 1 <= max_chars:
            current += " " + sentence
        else:
            output.append(current)
            current = sentence
    if current:
        output.append(current)
    return output


def _pieces(pages: list[ExtractedPage], max_chars: int) -> list[_Piece]:
    output: list[_Piece] = []
    for page in pages:
        section: str | None = None
        paragraphs = re.split(r"\n\s*\n", page.text)
        for paragraph in paragraphs:
            cleaned = re.sub(r"[ \t]+", " ", paragraph).strip()
            if not cleaned:
                continue
            first_line = cleaned.splitlines()[0].strip()
            if _looks_like_heading(first_line):
                section = first_line[:240]
            for part in _split_long_text(cleaned, max_chars):
                output.append(_Piece(part, page.page, section, page.document))
    return output


def build_chunks(
    pages: list[ExtractedPage], target_tokens: int, overlap_tokens: int
) -> list[TextChunk]:
    target_chars = target_tokens * 4
    overlap_chars = min(overlap_tokens * 4, target_chars // 2)
    pieces = _pieces(pages, target_chars)
    chunks: list[TextChunk] = []
    current: list[_Piece] = []
    current_chars = 0

    def flush() -> None:
        nonlocal current, current_chars
        if not current:
            return
        source_document = current[0].document
        content = (
            f"[Archivo interno: {source_document}]\n"
            + "\n\n".join(piece.text for piece in current).strip()
        )
        chunks.append(
            TextChunk(
                index=len(chunks),
                content=content,
                page_start=min(piece.page for piece in current),
                page_end=max(piece.page for piece in current),
                section_title=next((piece.section for piece in reversed(current) if piece.section), None),
                token_count=max(1, math.ceil(len(content) / 4)),
            )
        )
        if overlap_chars <= 0:
            current = []
            current_chars = 0
            return
        retained: list[_Piece] = []
        retained_chars = 0
        for piece in reversed(current):
            retained.insert(0, piece)
            retained_chars += len(piece.text) + 2
            if retained_chars >= overlap_chars:
                break
        current = retained
        current_chars = retained_chars

    for piece in pieces:
        if current and piece.document != current[0].document:
            flush()
            current = []
            current_chars = 0
        projected = current_chars + len(piece.text) + (2 if current else 0)
        if current and projected > target_chars:
            flush()
        current.append(piece)
        current_chars += len(piece.text) + (2 if len(current) > 1 else 0)
    flush()
    return chunks

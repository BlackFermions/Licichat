"""Small data contracts shared by the ingestion stages."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExtractedPage:
    document: str
    page: int
    text: str
    used_ocr: bool = False


@dataclass(frozen=True)
class ExtractionResult:
    pages: list[ExtractedPage]
    detected_mime_type: str
    extracted_filename: str
    page_count: int
    native_text_pages: int
    ocr_pages: int
    skipped_pages: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def text_char_count(self) -> int:
        return sum(len(page.text) for page in self.pages)

    @property
    def markdown(self) -> str:
        parts = []
        for page in self.pages:
            parts.append(f"## {page.document} - Pagina {page.page}\n\n{page.text.strip()}")
        return "\n\n".join(parts).strip() + "\n"


@dataclass(frozen=True)
class TextChunk:
    index: int
    content: str
    page_start: int
    page_end: int
    section_title: str | None
    token_count: int
    embedding: list[float] | None = None

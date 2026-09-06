import unittest

from ingestion.chunking import build_chunks
from ingestion.models import ExtractedPage


class ChunkingTests(unittest.TestCase):
    def test_preserves_page_ranges_and_chunk_bounds(self):
        pages = [
            ExtractedPage("Bases", 1, "REQUISITOS\n\n" + "Documento obligatorio. " * 80),
            ExtractedPage("Bases", 2, "ESPECIFICACIONES\n\n" + "Caracteristica tecnica. " * 80),
        ]
        chunks = build_chunks(pages, target_tokens=120, overlap_tokens=20)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual([chunk.index for chunk in chunks], list(range(len(chunks))))
        self.assertTrue(all(1 <= chunk.page_start <= chunk.page_end <= 2 for chunk in chunks))
        self.assertTrue(all(chunk.content for chunk in chunks))

    def test_empty_pages_return_no_chunks(self):
        self.assertEqual(build_chunks([], target_tokens=800, overlap_tokens=100), [])

    def test_keeps_archive_member_provenance_and_never_mixes_members(self):
        pages = [
            ExtractedPage("0Acta de otorgamiento.pdf", 1, "Resultado y ganador. " * 20),
            ExtractedPage("1Cuadro de evaluacion.pdf", 1, "Puntajes de los postores. " * 20),
        ]
        chunks = build_chunks(pages, target_tokens=500, overlap_tokens=50)
        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[0].content.startswith("[Archivo interno: 0Acta de otorgamiento.pdf]"))
        self.assertTrue(chunks[1].content.startswith("[Archivo interno: 1Cuadro de evaluacion.pdf]"))
        self.assertNotIn("1Cuadro", chunks[0].content)


if __name__ == "__main__":
    unittest.main()

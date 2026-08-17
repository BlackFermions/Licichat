import unittest

from lite_rag_engine import Corpus, PageText, select_context


class LiteRagSelectionTests(unittest.TestCase):
    def test_selects_page_related_to_question(self):
        corpus = Corpus(
            tender_id="1",
            source_digest="digest",
            pages=[
                PageText("Bases", 1, "Presentacion general del procedimiento."),
                PageText("Bases", 8, "La experiencia minima del postor es de tres anos."),
                PageText("Bases", 12, "El plazo para consultas termina el viernes."),
            ],
            document_count=1,
            total_pages=3,
            total_chars=140,
        )
        context, references = select_context(corpus, "Cual es la experiencia minima?")
        self.assertIn("tres anos", context)
        self.assertEqual(references[0]["page"], 8)

    def test_falls_back_to_first_pages_for_unmatched_question(self):
        corpus = Corpus(
            tender_id="1",
            source_digest="digest",
            pages=[PageText("Bases", 1, "Contenido inicial")],
            document_count=1,
            total_pages=1,
            total_chars=17,
        )
        context, references = select_context(corpus, "xyz")
        self.assertIn("Contenido inicial", context)
        self.assertEqual(references[0]["page"], 1)

    def test_expands_procurement_language_for_convoked_suppliers(self):
        corpus = Corpus(
            tender_id="1",
            source_digest="digest",
            pages=[
                PageText("Bases", 1, "Cronograma general del procedimiento."),
                PageText(
                    "Bases Integradas",
                    18,
                    "El postor debe presentar el registro sanitario y el certificado HACCP.",
                ),
            ],
            document_count=1,
            total_pages=2,
            total_chars=120,
        )
        context, references = select_context(corpus, "Que especificaciones piden a los convocados?")
        self.assertIn("registro sanitario", context)
        self.assertEqual(references[0]["page"], 18)

    def test_prioritizes_direct_terms_over_supporting_synonyms(self):
        corpus = Corpus(
            tender_id="1",
            source_digest="digest",
            pages=[
                PageText(
                    "Bases",
                    9,
                    "Requisito tecnico del postor, proveedor y participante para la oferta.",
                ),
                PageText(
                    "Bases Integradas",
                    32,
                    "Las especificaciones solicitadas incluyen proteina y energia total.",
                ),
            ],
            document_count=1,
            total_pages=2,
            total_chars=140,
        )
        _, references = select_context(corpus, "Que especificaciones piden a los convocados?")
        self.assertEqual(references[0]["page"], 32)


if __name__ == "__main__":
    unittest.main()

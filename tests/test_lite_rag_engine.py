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

    def test_prioritizes_supplier_requirements_for_convoked_suppliers(self):
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
        self.assertEqual(references[0]["page"], 9)
        self.assertEqual(references[1]["page"], 32)
        self.assertIn({"document": "Bases Integradas", "page": 32}, references)

    def test_labels_scored_evidence_in_context(self):
        corpus = Corpus(
            tender_id="1",
            source_digest="digest",
            pages=[
                PageText(
                    "Bases Integradas",
                    33,
                    "Factores de evaluacion. Proteina de 11.7 g a mas: 5 puntos.",
                ),
            ],
            document_count=1,
            total_pages=1,
            total_chars=70,
        )
        context, _ = select_context(corpus, "Que especificaciones piden?")
        self.assertIn("no asumir obligatoriedad", context)

    def test_includes_continuation_after_supplier_requirements(self):
        corpus = Corpus(
            tender_id="1",
            source_digest="digest",
            pages=[
                PageText("Bases Integradas", 21, "El postor debe presentar documentos obligatorios."),
                PageText("Bases Integradas", 22, "Continuacion: declaraciones y anexos de la oferta."),
                PageText("Bases Integradas", 33, "Especificaciones: proteina de 11.7 g a mas."),
            ],
            document_count=1,
            total_pages=3,
            total_chars=180,
        )
        _, references = select_context(corpus, "Que piden a los convocados?")
        self.assertEqual(references[:2], [
            {"document": "Bases Integradas", "page": 21},
            {"document": "Bases Integradas", "page": 22},
        ])

    def test_prefers_integrated_bases_and_skips_duplicate_pages(self):
        duplicate_text = "Especificaciones de proteina energia y condiciones del producto. " * 12
        corpus = Corpus(
            tender_id="1",
            source_digest="digest",
            pages=[
                PageText("Bases Administrativas", 32, duplicate_text),
                PageText("Bases Integradas", 33, duplicate_text),
                PageText("Bases Integradas", 22, "El postor presenta registro sanitario obligatorio."),
            ],
            document_count=2,
            total_pages=3,
            total_chars=1200,
        )
        _, references = select_context(corpus, "Que especificaciones piden al postor?")
        self.assertEqual(references[0], {"document": "Bases Integradas", "page": 22})
        self.assertIn({"document": "Bases Integradas", "page": 33}, references)
        self.assertNotIn({"document": "Bases Administrativas", "page": 32}, references)

    def test_prioritizes_contract_delivery_destination(self):
        corpus = Corpus(
            tender_id="1",
            source_digest="digest",
            pages=[
                PageText(
                    "Bases Integradas",
                    24,
                    "Para el perfeccionamiento del contrato, el postor ganador debe remitir "
                    "los documentos a la siguiente direccion electronica: "
                    "subgerlogistica@municanchis.com.pe. Si no cuenta con firma digital, "
                    "debe presentarlos en la Mesa de Partes de la Unidad de Tramite Documentario.",
                ),
                PageText(
                    "Bases Integradas",
                    42,
                    "El postor debe consignar un correo electronico para recibir notificaciones "
                    "durante la ejecucion del contrato.",
                ),
            ],
            document_count=1,
            total_pages=2,
            total_chars=330,
        )
        context, references = select_context(corpus, "A que correo tengo que dirigir el contrato?")
        self.assertIn("subgerlogistica@municanchis.com.pe", context)
        self.assertEqual(references[0], {"document": "Bases Integradas", "page": 24})

    def test_contract_delivery_synonyms_select_same_destination(self):
        corpus = Corpus(
            tender_id="1",
            source_digest="digest",
            pages=[
                PageText(
                    "Bases Integradas",
                    24,
                    "La suscripcion del contrato se realiza remitiendo la documentacion a la "
                    "direccion electronica contrataciones@entidad.gob.pe.",
                ),
                PageText(
                    "Bases Integradas",
                    42,
                    "El contratista consigna su correo para notificaciones de ejecucion contractual.",
                ),
            ],
            document_count=1,
            total_pages=2,
            total_chars=210,
        )
        _, references = select_context(corpus, "Y a quien se remite el contrato?")
        self.assertEqual(references[0], {"document": "Bases Integradas", "page": 24})

    def test_contract_notification_email_remains_distinct(self):
        corpus = Corpus(
            tender_id="1",
            source_digest="digest",
            pages=[
                PageText(
                    "Bases Integradas",
                    24,
                    "Para el perfeccionamiento del contrato, los documentos se remiten a mesa de partes.",
                ),
                PageText(
                    "Bases Integradas",
                    42,
                    "El postor ganador debe consignar un correo electronico para notificaciones "
                    "durante la ejecucion del contrato.",
                ),
            ],
            document_count=1,
            total_pages=2,
            total_chars=230,
        )
        context, references = select_context(corpus, "Que correo debo consignar para notificaciones del contrato?")
        self.assertIn("notificaciones", context)
        self.assertEqual(references[0], {"document": "Bases Integradas", "page": 42})


if __name__ == "__main__":
    unittest.main()

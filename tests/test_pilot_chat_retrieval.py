import os
import unittest
from unittest.mock import MagicMock, patch

import pilot_chat_retrieval as pilot


class PilotRetrievalTests(unittest.TestCase):
    def test_disabled_does_not_connect(self):
        with patch.dict(os.environ, {"LITE_DOCUMENT_PILOT_ENABLED": "false"}), patch.object(pilot, "_connection") as connect:
            self.assertIsNone(pilot.pilot_status("1241981"))
            connect.assert_not_called()

    def test_db_failure_falls_back(self):
        with patch.object(pilot, "enabled", return_value=True), patch.object(pilot, "_connection", side_effect=RuntimeError):
            self.assertIsNone(pilot.pilot_status("1241981"))

    def test_status_only_reports_partial_coverage(self):
        conn = MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = [{"page_count": 10, "text_char_count": 900,
            "extraction_coverage": {"partial": True}, "detected_mime_type": "application/zip",
            "document_role": "buena_pro"}]
        with patch.object(pilot, "enabled", return_value=True), patch.object(pilot, "_connection", return_value=conn):
            status = pilot.pilot_status("1241981")
        self.assertTrue(status["partial"])
        self.assertIn("Lectura parcial", status["warning"])
        self.assertNotIn("referencias internas", status["warning"])
        self.assertEqual(cursor.execute.call_args.args[1], ("1241981", "pilot-v1"))
        conn.close.assert_called_once()

    def test_status_uses_newest_available_pipeline_with_fallback(self):
        conn = MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.fetchall.side_effect = [[], [{
            "page_count": 3,
            "text_char_count": 500,
            "extraction_coverage": {},
            "detected_mime_type": "application/pdf",
            "document_role": "bases",
        }]]
        with patch.dict(os.environ, {"AI_PILOT_PIPELINE_VERSIONS": "pilot-v2,pilot-v1"}), \
             patch.object(pilot, "enabled", return_value=True), \
             patch.object(pilot, "_connection", return_value=conn):
            status = pilot.pilot_status("1241981")
        self.assertEqual(status["pipeline_version"], "pilot-v1")
        self.assertEqual(cursor.execute.call_args_list[0].args[1], ("1241981", "pilot-v2"))
        self.assertEqual(cursor.execute.call_args_list[1].args[1], ("1241981", "pilot-v1"))

    def test_context_has_real_ranges_and_is_bounded(self):
        rows = [{"source_title": "Bases Integradas", "page_start": 81, "page_end": 83,
                 "content": "Vehiculos con certificado. " * 1000}]
        context, refs = pilot.format_context(rows)
        self.assertLessEqual(len(context), pilot.MAX_CONTEXT_CHARS)
        self.assertEqual(refs, [{"document": "Bases Integradas", "page": "81-83"}])

    def test_context_uses_internal_archive_member_as_citation(self):
        rows = [{
            "source_title": "Documentos de Otorgamiento de Buena Pro",
            "page_start": 2,
            "page_end": 2,
            "content": "[Archivo interno: 0Acta de otorgamiento.pdf]\nPostor adjudicado.",
        }]
        context, refs = pilot.format_context(rows)
        self.assertIn("Contenedor: Documentos de Otorgamiento", context)
        self.assertEqual(refs, [{"document": "0Acta de otorgamiento.pdf", "page": "2"}])

    def test_unprepared_does_not_embed(self):
        client = MagicMock()
        with patch.object(pilot, "pilot_status", return_value=None):
            self.assertIsNone(pilot.retrieve_pilot("outside", "pregunta", client))
        client.with_options.assert_not_called()

    def test_retrieval_scopes_tender_and_version(self):
        client = MagicMock()
        client.with_options.return_value.embeddings.create.return_value.data[0].embedding = [0.1] * 1536
        conn = MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = [{"source_title": "Bases", "page_start": 2,
                                        "page_end": 2, "content": "Experiencia acreditada"}]
        with patch.object(pilot, "pilot_status", return_value={"ready": True}), patch.object(pilot, "_connection", return_value=conn):
            result = pilot.retrieve_pilot("1241981", "Que experiencia piden?", client,
                                          [{"role": "user", "content": "Requisitos del postor"}])
        self.assertIsNotNone(result)
        params = cursor.execute.call_args_list[0].args[1]
        self.assertEqual(params[0:2], ("1241981", "pilot-v1"))
        self.assertEqual(params[3:5], ("1241981", "pilot-v1"))
        self.assertIn("MATERIALIZED", cursor.execute.call_args.args[0])

    def test_award_question_only_ranks_award_document(self):
        client = MagicMock()
        client.with_options.return_value.embeddings.create.return_value.data[0].embedding = [0.1] * 1536
        conn = MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = [{"source_title": "Otorgamiento de Buena Pro", "page_start": 1,
                                        "page_end": 1, "content": "Resultado: desierto"}]
        with patch.object(pilot, "pilot_status", return_value={"ready": True}), patch.object(pilot, "_connection", return_value=conn):
            result = pilot.retrieve_pilot("1243819", "Se puede ver el documento de otorgamiento?", client)
        self.assertIsNotNone(result)
        self.assertIn("a.document_role = %s", cursor.execute.call_args.args[0])
        self.assertEqual(cursor.execute.call_args_list[0].args[1][5], "buena_pro")
        self.assertEqual(result[2][0]["document"], "Otorgamiento de Buena Pro")

    def test_non_award_question_only_ranks_bases(self):
        cursor = MagicMock()
        cursor.fetchall.return_value = [{"source_title": "Bases", "page_start": 2,
                                         "page_end": 2, "content": "Objeto de la convocatoria"}]
        connection = MagicMock()
        connection.cursor.return_value.__enter__.return_value = cursor
        client = MagicMock()
        client.with_options.return_value.embeddings.create.return_value.data = [
            MagicMock(embedding=[0.0] * 1536)
        ]
        with patch.object(pilot, "pilot_status", return_value={"ready": True}), \
             patch.object(pilot, "_connection", return_value=connection):
            result = pilot.retrieve_pilot("1", "Cual es el objetivo?", client)
        self.assertIsNotNone(result)
        self.assertIn("a.document_role = %s", cursor.execute.call_args.args[0])
        self.assertEqual(cursor.execute.call_args.args[1][5], "bases")
        self.assertEqual(result[2][0]["document"], "Bases")

    def test_award_follow_up_inherits_role_from_history(self):
        history = [
            {"role": "user", "content": "Quien gano la buena pro?"},
            {"role": "assistant", "content": "La empresa ABC fue adjudicada."},
        ]
        self.assertEqual(
            pilot.infer_document_role("Por que no ganaron los otros?", history),
            "buena_pro",
        )
        self.assertEqual(
            pilot.infer_document_role("Que documento sustenta ese resultado?", history),
            "buena_pro",
        )

    def test_amount_query_expands_contract_vocabulary_without_changing_role(self):
        query = pilot._retrieval_query("Cual era el presupuesto?", [], "bases")
        self.assertIn("valor estimado", query)
        self.assertIn("cuantia de la contratacion", query)
        self.assertEqual(pilot.infer_document_role("Cual era el presupuesto?"), "bases")

    def test_invalid_vector_falls_back(self):
        client = MagicMock()
        client.with_options.return_value.embeddings.create.return_value.data[0].embedding = [float("nan")] * 1536
        with patch.object(pilot, "pilot_status", return_value={"ready": True}), patch.object(pilot, "_connection") as connect:
            self.assertIsNone(pilot.retrieve_pilot("1241981", "pregunta", client))
            connect.assert_not_called()


if __name__ == "__main__":
    unittest.main()

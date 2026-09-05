import importlib
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

with patch.dict(os.environ, {"OPENAI_API_KEY": "unit-test-not-a-key",
                             "OPENAI_API_BASE": "https://unit-test.openai.azure.com"}), \
     patch.dict(sys.modules, {"match_engine_pgvector": MagicMock()}):
    chat = importlib.import_module("lite_chat_engine")


class PilotChatRouteTests(unittest.TestCase):
    def setUp(self):
        self.client = chat.app.test_client()
        self.key = patch.object(chat, "SERVICE_KEY", "unit-test-service")
        self.key.start()
        self.addCleanup(self.key.stop)
        self.headers = {"X-Service-Key": "unit-test-service"}

    def test_status_requires_service_auth(self):
        with patch.object(chat, "pilot_status") as status:
            self.assertEqual(self.client.post("/api/v1/pilot-status", json={"tender_id": "1"}).status_code, 401)
            status.assert_not_called()

    def test_preparation_uses_pilot_without_download_or_model(self):
        with patch.object(chat, "pilot_status", return_value={"ready": True, "warning": "Lectura parcial."}), \
             patch.object(chat, "prepare_corpus") as prepare, patch.object(chat, "openai_client") as model:
            response = self.client.post("/api/v1/chat_stream", headers=self.headers,
                json={"tender_id": "1243812", "message": "Analiza las bases", "use_document_pilot": True})
            self.assertIn("Lectura parcial", response.get_data(as_text=True))
            prepare.assert_not_called()
            model.chat.completions.create.assert_not_called()

    def test_scoped_chat_uses_pilot_evidence(self):
        status = {"documents": 1, "pages": 20, "characters": 1000, "warning": "Lectura parcial."}
        chunk = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Evidencia [Bases, p. 3]."))])
        with patch.object(chat, "retrieve_pilot", return_value=(status, "Texto de las bases", [{"document": "Bases", "page": "3"}])), \
             patch.object(chat, "get_tender_bundle", return_value=({"id": "1"}, [])), \
             patch.object(chat, "prepare_corpus") as prepare, patch.object(chat, "openai_client") as model:
            model.chat.completions.create.return_value = [chunk]
            response = self.client.post("/api/v1/chat_stream", headers=self.headers,
                json={"tender_id": "1", "message": "Que requisitos piden?", "use_document_pilot": True})
            text = response.get_data(as_text=True)
            self.assertIn("Lectura parcial", text)
            self.assertIn("Evidencia", text)
            prepare.assert_not_called()
            prompt = model.chat.completions.create.call_args.kwargs["messages"][0]["content"]
            self.assertIn("Texto de las bases", prompt)

    def test_non_search_caller_does_not_use_pilot(self):
        corpus = SimpleNamespace(document_count=1, total_pages=1, total_chars=400)
        with patch.object(chat, "pilot_status") as status, patch.object(chat, "retrieve_pilot") as retrieve, \
             patch.object(chat, "prepare_corpus", return_value=({}, corpus)) as prepare:
            response = self.client.post("/api/v1/chat_stream", headers=self.headers,
                json={"tender_id": "1", "message": "Analiza las bases"})
            self.assertIn("Bases analizadas", response.get_data(as_text=True))
            status.assert_not_called()
            retrieve.assert_not_called()
            prepare.assert_called_once()

    def test_missing_pilot_uses_existing_preparation(self):
        corpus = SimpleNamespace(document_count=1, total_pages=1, total_chars=400)
        with patch.object(chat, "pilot_status", return_value=None), \
             patch.object(chat, "prepare_corpus", return_value=({}, corpus)) as prepare:
            response = self.client.post("/api/v1/chat_stream", headers=self.headers,
                json={"tender_id": "outside", "message": "Analiza las bases", "use_document_pilot": True})
            self.assertIn("Bases analizadas", response.get_data(as_text=True))
            prepare.assert_called_once()

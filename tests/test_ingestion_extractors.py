import io
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

from ingestion.config import Settings
from ingestion.errors import IngestionError
from ingestion.extractors import _extract_pdf, _read_zip_members, _safe_member_name, detect_kind, source_suffix


def settings_fixture() -> Settings:
    return Settings(
        pipeline_version="test", worker_id="test", batch_size=1, stale_lock_minutes=90,
        download_max_bytes=1024 * 1024, archive_max_files=10, archive_max_selected_files=2,
        archive_max_uncompressed_bytes=1024 * 1024, archive_max_ratio=20,
        max_pages_per_document=10, max_text_chars_per_document=100_000,
        native_text_min_chars=80, ocr_max_pages=2, ocr_dpi=150,
        chunk_target_tokens=500, chunk_overlap_tokens=50, embedding_batch_size=8,
        embedding_dimensions=1536, embedding_deployment="embedding", chat_deployment="chat",
        chat_summaries=False, openai_endpoint="https://example.openai.azure.com",
        openai_api_key="test", openai_api_version="test", blob_account_url="https://example.blob.core.windows.net",
        blob_container="test", blob_connection_string=None, db_host="host", db_user="user",
        db_password="password", db_name="db", db_port=5432, db_sslmode="require",
    )


class ExtractorTests(unittest.TestCase):
    def test_reports_partial_coverage_when_ocr_budget_is_exhausted(self):
        document = MagicMock()
        document.needs_pass = False
        document.__len__.return_value = 3
        document.__getitem__.return_value.get_text.return_value = ""
        runtime = MagicMock()
        runtime.open.return_value.__enter__.return_value = document
        with patch("ingestion.extractors.fitz", runtime), patch(
            "ingestion.extractors._ocr_page", return_value="texto extraido"
        ) as ocr:
            result = _extract_pdf(b"fixture", "bases.pdf", settings_fixture())
        self.assertEqual(ocr.call_count, 2)
        self.assertEqual(result.skipped_pages, 1)
        self.assertIn("ocr_limit_reached", result.warnings)

    def test_rejects_archive_traversal(self):
        with self.assertRaises(IngestionError):
            _safe_member_name("../secret.pdf")

    def test_selects_relevant_bases_from_zip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bases.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("anexo.txt", "anexo")
                archive.writestr("Bases Integradas.pdf", b"%PDF-fixture")
                archive.writestr("otro.pdf", b"%PDF-other")
            members = _read_zip_members(str(path), "bases", settings_fixture())
            self.assertEqual([name for name, _ in members], ["Bases Integradas.pdf"])
            self.assertEqual(detect_kind(str(path)), "zip")

    def test_limits_compression_ratio(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bomb.zip"
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("Bases Administrativas.txt", "A" * 50_000)
            strict = replace(settings_fixture(), archive_max_ratio=2)
            with self.assertRaises(IngestionError) as context:
                _read_zip_members(str(path), "bases", strict)
            self.assertEqual(context.exception.code, "archive_ratio_exceeded")

    def test_maps_known_formats(self):
        self.assertEqual(source_suffix("application/pdf", "Bases"), ".pdf")
        self.assertEqual(source_suffix("ZIP", "Bases"), ".zip")


if __name__ == "__main__":
    unittest.main()

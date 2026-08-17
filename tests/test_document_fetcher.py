import os
import tempfile
import unittest

from document_fetcher import DocumentDownloadError, download_document, normalize_document_url


SAMPLE_URL = (
    "https://prod1.seace.gob.pe/SeaceWeb-PRO/"
    "SdescargarArchivoAlfresco?fileCode=229a6a07-a352-4647-a2aa-0dcad96021eb"
)


class DocumentFetcherTests(unittest.TestCase):
    def test_normalizes_known_seace_url(self):
        self.assertEqual(normalize_document_url(SAMPLE_URL), SAMPLE_URL)

    def test_decodes_html_query_separator(self):
        url = SAMPLE_URL + "&amp;ignored=value"
        self.assertEqual(normalize_document_url(url), SAMPLE_URL)

    def test_rejects_untrusted_host(self):
        with self.assertRaises(DocumentDownloadError) as ctx:
            normalize_document_url(
                "https://example.com/SeaceWeb-PRO/SdescargarArchivoAlfresco"
                "?fileCode=229a6a07-a352-4647-a2aa-0dcad96021eb"
            )
        self.assertEqual(ctx.exception.code, "untrusted_host")

    @unittest.skipUnless(os.getenv("RUN_LIVE_SEACE_TEST") == "1", "live SEACE test")
    def test_live_pdf_download(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = download_document(SAMPLE_URL, temp_dir, max_bytes=3 * 1024 * 1024)
            with open(path, "rb") as handle:
                self.assertEqual(handle.read(5), b"%PDF-")


if __name__ == "__main__":
    unittest.main()

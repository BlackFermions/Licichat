import os
import tempfile
import unittest
from unittest.mock import patch

import requests

from document_fetcher import DocumentDownloadError, download_document, normalize_document_url


SAMPLE_URL = (
    "https://prod1.seace.gob.pe/SeaceWeb-PRO/"
    "SdescargarArchivoAlfresco?fileCode=229a6a07-a352-4647-a2aa-0dcad96021eb"
)


class FakeResponse:
    def __init__(self, status_code, url, content=b"", headers=None):
        self.status_code = status_code
        self.url = url
        self._content = content
        self.headers = headers or {}

    def close(self):
        return None

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size):
        yield self._content


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

    def test_uses_authenticated_proxy_when_azure_is_forbidden(self):
        direct = FakeResponse(403, SAMPLE_URL)
        proxied = FakeResponse(
            200,
            "https://licigob-proxy.example.workers.dev/",
            b"%PDF-1.7\nfixture",
            {"Content-Length": "16"},
        )
        with (
            patch("document_fetcher.SEACE_DOCUMENT_PROXY_URL", proxied.url),
            patch("document_fetcher.SEACE_DOCUMENT_PROXY_KEY", "test-key"),
            patch("requests.Session.get", return_value=direct),
            patch("requests.Session.post", return_value=proxied) as post,
            tempfile.TemporaryDirectory() as temp_dir,
        ):
            path = download_document(SAMPLE_URL, temp_dir)
            with open(path, "rb") as handle:
                self.assertEqual(handle.read(5), b"%PDF-")
            self.assertEqual(post.call_args.kwargs["json"], {"url": SAMPLE_URL})
            self.assertEqual(post.call_args.kwargs["headers"]["X-Proxy-Key"], "test-key")

    def test_uses_authenticated_proxy_when_direct_connection_fails(self):
        proxied = FakeResponse(
            200,
            "https://licigob-proxy.example.workers.dev/",
            b"%PDF-1.7\nfixture",
            {"Content-Length": "16"},
        )
        with (
            patch("document_fetcher.SEACE_DOCUMENT_PROXY_URL", proxied.url),
            patch("document_fetcher.SEACE_DOCUMENT_PROXY_KEY", "test-key"),
            patch(
                "requests.Session.get",
                side_effect=requests.ConnectionError("temporary upstream failure"),
            ),
            patch("requests.Session.post", return_value=proxied) as post,
            tempfile.TemporaryDirectory() as temp_dir,
        ):
            path = download_document(SAMPLE_URL, temp_dir)
            with open(path, "rb") as handle:
                self.assertEqual(handle.read(5), b"%PDF-")
            self.assertEqual(post.call_count, 1)

    def test_can_prefer_proxy_without_requesting_seace_directly(self):
        proxied = FakeResponse(
            200,
            "https://licigob-proxy.example.workers.dev/",
            b"%PDF-1.7\nfixture",
            {"Content-Length": "16"},
        )
        with (
            patch("document_fetcher.SEACE_DOCUMENT_PROXY_URL", proxied.url),
            patch("document_fetcher.SEACE_DOCUMENT_PROXY_KEY", "test-key"),
            patch("document_fetcher.SEACE_PROXY_FIRST", True),
            patch("requests.Session.get") as direct,
            patch("requests.Session.post", return_value=proxied),
            tempfile.TemporaryDirectory() as temp_dir,
        ):
            path = download_document(SAMPLE_URL, temp_dir)
            with open(path, "rb") as handle:
                self.assertEqual(handle.read(5), b"%PDF-")
            direct.assert_not_called()

    @unittest.skipUnless(os.getenv("RUN_LIVE_SEACE_TEST") == "1", "live SEACE test")
    def test_live_pdf_download(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = download_document(SAMPLE_URL, temp_dir, max_bytes=3 * 1024 * 1024)
            with open(path, "rb") as handle:
                self.assertEqual(handle.read(5), b"%PDF-")


if __name__ == "__main__":
    unittest.main()

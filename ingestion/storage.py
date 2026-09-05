"""Private Azure Blob object storage using managed identity in Azure."""

from __future__ import annotations

from pathlib import Path

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContentSettings

from ingestion.config import Settings
from ingestion.errors import IngestionError


class ObjectStorage:
    def __init__(self, settings: Settings):
        if settings.blob_connection_string:
            service = BlobServiceClient.from_connection_string(settings.blob_connection_string)
        else:
            credential = DefaultAzureCredential(exclude_interactive_browser_credential=True)
            service = BlobServiceClient(account_url=settings.blob_account_url, credential=credential)
        self.container = service.get_container_client(settings.blob_container)

    def upload_file(self, key: str, path: str, content_type: str) -> str:
        try:
            with Path(path).open("rb") as handle:
                self.container.upload_blob(
                    name=key,
                    data=handle,
                    overwrite=True,
                    content_settings=ContentSettings(content_type=content_type),
                )
            return key
        except Exception as exc:
            raise IngestionError("blob_upload_failed", "No se pudo guardar el documento.", transient=True) from exc

    def download_file(self, key: str, path: str, max_bytes: int) -> str:
        try:
            blob = self.container.get_blob_client(key)
            properties = blob.get_blob_properties()
            if properties.size > max_bytes:
                raise IngestionError(
                    "file_too_large",
                    "El documento almacenado supera el limite permitido.",
                )
            with Path(path).open("wb") as handle:
                written = 0
                for chunk in blob.download_blob(max_concurrency=1).chunks():
                    written += len(chunk)
                    if written > max_bytes:
                        raise IngestionError("file_too_large", "El documento supera el limite permitido.")
                    handle.write(chunk)
            if Path(path).stat().st_size == 0:
                raise IngestionError("empty_file", "El documento almacenado esta vacio.")
            return path
        except IngestionError:
            Path(path).unlink(missing_ok=True)
            raise
        except Exception as exc:
            Path(path).unlink(missing_ok=True)
            raise IngestionError(
                "blob_download_failed",
                "No se pudo recuperar el documento almacenado.",
                transient=True,
            ) from exc

    def upload_text(self, key: str, value: str, content_type: str = "text/markdown; charset=utf-8") -> str:
        try:
            self.container.upload_blob(
                name=key,
                data=value.encode("utf-8"),
                overwrite=True,
                content_settings=ContentSettings(content_type=content_type),
            )
            return key
        except Exception as exc:
            raise IngestionError("blob_upload_failed", "No se pudo guardar el texto extraido.", transient=True) from exc

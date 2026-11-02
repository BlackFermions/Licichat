"""Explicit, read-only evaluation setup. Never persists or prints credentials."""

import json
import os
import shutil
import subprocess


def load_azure_environment():
    executable = shutil.which("az") or shutil.which("az.cmd")
    if not executable:
        raise RuntimeError("Azure CLI no disponible")

    def read(*args):
        result = subprocess.run(
            [executable, *args, "-o", "json"], capture_output=True,
            text=True, encoding="utf-8", timeout=60,
        )
        if result.returncode:
            raise RuntimeError("No se pudo leer la configuracion de Azure; no se muestran secretos")
        return json.loads(result.stdout)

    app = read("containerapp", "show", "-n", "licigob-ai-lite", "-g", "licigobrsg")
    entries = app["properties"]["template"]["containers"][0]["env"]
    secrets = {}
    if any(entry.get("secretRef") for entry in entries):
        secrets = {item["name"]: item.get("value", "") for item in read(
            "containerapp", "secret", "list", "-n", "licigob-ai-lite",
            "-g", "licigobrsg", "--show-values",
        )}
    # Do not import unrelated configuration into this process.
    allowed = {
        "DB_HOST", "DB_USER", "DB_PASSWORD", "DB_NAME", "DB_PORT", "DB_SSLMODE",
        "OPENAI_API_BASE", "OPENAI_API_KEY", "OPENAI_API_VERSION", "CHAT_DEPLOYMENT",
        "PILOT_EMBEDDING_DEPLOYMENT", "AI_PILOT_PIPELINE_VERSIONS",
        "LITE_DOCUMENT_PILOT_ENABLED", "LITE_MAX_CONTEXT_CHARS",
    }
    for entry in entries:
        if entry["name"] in allowed:
            value = secrets.get(entry.get("secretRef"), entry.get("value"))
            if value is not None:
                os.environ[entry["name"]] = str(value)
    # Server-enforced read-only sessions, including connection calls inside the app.
    os.environ["PGOPTIONS"] = "-c default_transaction_read_only=on -c statement_timeout=15000"
    return app["properties"]["template"]["containers"][0]["image"]

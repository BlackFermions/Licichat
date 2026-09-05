"""Stable internal error codes without leaking document contents or secrets."""

from __future__ import annotations


class IngestionError(RuntimeError):
    def __init__(self, code: str, message: str, *, transient: bool = False):
        super().__init__(message)
        self.code = code[:80]
        self.safe_message = message[:500]
        self.transient = transient

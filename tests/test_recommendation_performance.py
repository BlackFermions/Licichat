"""Evita búsquedas vectoriales sin candidatos y llamadas repetidas al modelo."""

import importlib
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

with patch.dict(os.environ, {
    "OPENAI_API_KEY": "unit-test-not-a-key",
    "OPENAI_API_BASE": "https://unit-test.openai.azure.com",
}):
    engine = importlib.import_module("match_engine_pgvector")


def tender(scope_id="historical"):
    return {
        "tender_id": scope_id, "similarity": 0.8,
        "objeto_contractual": "Servicio pesquero", "descripcion_licitacion": "Descripción",
        "entidad_convocante": "Entidad", "departamento": "LIMA", "region_ejecucion": None,
        "monto_referencial": 100, "categoria": "services", "metodo": "Concurso",
        "fecha_publicacion": "2026-01-01",
    }


def cursor_for(conn):
    return conn.cursor.return_value.__enter__.return_value


def vector_queries(cursor):
    return [str(call.args[0]) for call in cursor.execute.call_args_list if "<=>" in str(call.args[0])]


def test_no_recent_vectors_skips_hnsw_and_preserves_profile_filters(monkeypatch):
    recent, historical = MagicMock(), MagicMock()
    cursor_for(recent).fetchone.return_value = {"exists": False}
    cursor_for(historical).fetchall.return_value = [tender()]
    connections = iter([recent, historical])
    monkeypatch.setattr(engine, "_get_conn", lambda: next(connections))
    release = MagicMock()
    monkeypatch.setattr(engine, "_release_conn", release)
    monkeypatch.setattr(engine, "get_embedding", lambda text: np.array([0.1, 0.2]))

    result = engine.buscar_licitaciones("pesca", categorias=["Servicios"], regiones=["Lima"], strict=True)

    assert vector_queries(cursor_for(recent)) == []
    assert len(vector_queries(cursor_for(historical))) == 1
    historical_params = cursor_for(historical).execute.call_args.args[1]
    assert "services" in historical_params and "LIMA" in historical_params
    assert result[0]["recommendation_scope"] == "historical"
    assert release.call_count == 2


def test_recent_vectors_are_ranked_before_historical(monkeypatch):
    conn = MagicMock()
    cur = cursor_for(conn)
    cur.fetchone.return_value = {"exists": True}
    cur.fetchall.return_value = [tender("recent")]
    get_conn = MagicMock(return_value=conn)
    monkeypatch.setattr(engine, "_get_conn", get_conn)
    monkeypatch.setattr(engine, "_release_conn", MagicMock())
    monkeypatch.setattr(engine, "get_embedding", lambda text: np.array([0.1, 0.2]))
    result = engine.buscar_licitaciones("pesca")
    assert result[0]["recommendation_scope"] == "recent"
    assert len(vector_queries(cur)) == 1
    get_conn.assert_called_once()


@pytest.fixture()
def embedding_model(monkeypatch):
    engine._embedding_values.cache_clear()
    model = MagicMock()
    model.embeddings.create.return_value = SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2])])
    monkeypatch.setattr(engine, "client", model)
    yield model
    engine._embedding_values.cache_clear()


def test_same_profile_reuses_embedding_but_changed_profile_gets_new_vector(embedding_model):
    first = engine.get_embedding("pesca")
    first[0] = 100
    again = engine.get_embedding("pesca")
    engine.get_embedding("construcción")
    assert embedding_model.embeddings.create.call_count == 2
    assert again[0] != 100


def test_embedding_error_does_not_poison_cache(embedding_model):
    response = embedding_model.embeddings.create.return_value
    embedding_model.embeddings.create.side_effect = [RuntimeError("temporary"), response]
    assert engine.get_embedding("pesca") is None
    assert engine.get_embedding("pesca") is not None
    assert embedding_model.embeddings.create.call_count == 2

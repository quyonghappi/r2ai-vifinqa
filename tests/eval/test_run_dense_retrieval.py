import json
import sys
import types

import numpy as np
import pandas as pd
import pytest

import eval.run_dense_retrieval as run_dense_retrieval


class FakeEncoder:
    """Stands in for SentenceTransformerEncoder so this test needs no model weights."""

    def __init__(self, _model_path):
        pass

    def encode(self, texts, *, batch_size):
        return np.asarray([[1.0, 0.0] if "AAA" in t else [0.0, 1.0] for t in texts], dtype=np.float32)


def _catalog(tmp_path, with_searchable_text: bool):
    rows = [
        {"report_id": "AAA_report", "line_position": 17, "ticker": "AAA", "year": 2023, "variant": "consolidated"},
        {"report_id": "BBB_report", "line_position": 21, "ticker": "BBB", "year": 2023, "variant": "consolidated"},
    ]
    if with_searchable_text:
        rows[0]["searchable_text"] = "Doanh thu thuần."
        rows[1]["searchable_text"] = "Doanh thu thuần."
    path = tmp_path / "tables.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_run_rejects_a_catalog_without_searchable_text(tmp_path):
    catalog = _catalog(tmp_path, with_searchable_text=False)
    companies = tmp_path / "companies.csv"
    pd.DataFrame([{"ticker": "AAA", "name": "Cong ty AAA"}]).to_csv(companies, index=False)
    with pytest.raises(ValueError, match="searchable_text"):
        run_dense_retrieval.run(
            tmp_path / "model", catalog, companies, tmp_path / "questions.jsonl",
            tmp_path / "index", tmp_path / "out.json",
        )


def test_run_builds_index_once_and_scopes_by_entity(tmp_path, monkeypatch):
    monkeypatch.setattr(run_dense_retrieval, "SentenceTransformerEncoder", FakeEncoder)
    catalog = _catalog(tmp_path, with_searchable_text=True)
    companies = tmp_path / "companies.csv"
    pd.DataFrame([
        {"ticker": "AAA", "name": "Cong ty AAA"}, {"ticker": "BBB", "name": "Cong ty BBB"},
    ]).to_csv(companies, index=False)
    questions = tmp_path / "questions.jsonl"
    questions.write_text(json.dumps({"id": 1, "question": "Doanh thu AAA nam 2023"}) + "\n", encoding="utf-8")
    index_dir = tmp_path / "index"

    rankings = run_dense_retrieval.run(tmp_path / "model", catalog, companies, questions, index_dir, tmp_path / "out.json")
    assert [row[0] for row in rankings["1"]] == ["AAA_report|17"]
    assert (index_dir / "dense_embeddings.npy").exists()

    # A second run must reuse the cached index (same doc_ids/embeddings) rather than rebuilding it.
    cached_embeddings = (index_dir / "dense_embeddings.npy").read_bytes()
    rankings_again = run_dense_retrieval.run(tmp_path / "model", catalog, companies, questions, index_dir, tmp_path / "out2.json")
    assert rankings_again == rankings
    assert (index_dir / "dense_embeddings.npy").read_bytes() == cached_embeddings


def test_run_rejects_neither_or_both_model_sources(tmp_path):
    catalog = _catalog(tmp_path, with_searchable_text=True)
    companies = tmp_path / "companies.csv"
    pd.DataFrame([{"ticker": "AAA", "name": "Cong ty AAA"}]).to_csv(companies, index=False)
    questions = tmp_path / "questions.jsonl"
    questions.write_text(json.dumps({"id": 1, "question": "q"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one"):
        run_dense_retrieval.run(None, catalog, companies, questions, tmp_path / "index", tmp_path / "out.json")
    with pytest.raises(ValueError, match="exactly one"):
        run_dense_retrieval.run(
            tmp_path / "model", catalog, companies, questions, tmp_path / "index", tmp_path / "out.json",
            hf_repo_id="BAAI/bge-m3",
        )


def test_run_downloads_from_huggingface_hub_when_no_local_model_path(tmp_path, monkeypatch):
    monkeypatch.setattr(run_dense_retrieval, "SentenceTransformerEncoder", FakeEncoder)
    downloaded_to = tmp_path / "hf_snapshot"
    downloaded_to.mkdir()
    calls = []

    def fake_snapshot_download(*, repo_id, revision, cache_dir, token):
        calls.append({"repo_id": repo_id, "revision": revision, "cache_dir": cache_dir, "token": token})
        return str(downloaded_to)

    fake_hub = types.SimpleNamespace(snapshot_download=fake_snapshot_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    catalog = _catalog(tmp_path, with_searchable_text=True)
    companies = tmp_path / "companies.csv"
    pd.DataFrame([{"ticker": "AAA", "name": "Cong ty AAA"}]).to_csv(companies, index=False)
    questions = tmp_path / "questions.jsonl"
    questions.write_text(json.dumps({"id": 1, "question": "Doanh thu AAA nam 2023"}) + "\n", encoding="utf-8")

    rankings = run_dense_retrieval.run(
        None, catalog, companies, questions, tmp_path / "index", tmp_path / "out.json",
        hf_repo_id="BAAI/bge-m3", hf_revision="main", hf_cache_dir=tmp_path / "cache",
    )
    assert [row[0] for row in rankings["1"]] == ["AAA_report|17"]
    assert len(calls) == 1
    assert calls[0]["repo_id"] == "BAAI/bge-m3"

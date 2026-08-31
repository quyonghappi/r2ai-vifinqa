import json
import sys
import types

import numpy as np
import pandas as pd

from retrieval.dense import DenseIndex, SentenceTransformerEncoder, build_dense_index, rank_questions_dense


class FakeEncoder:
    def encode(self, texts, *, batch_size):
        return np.asarray([[len(text), 1.0] for text in texts], dtype=np.float32)


class OneHotEncoder:
    """Deterministic stand-in for a real encoder: each text's embedding is looked up by exact
    string match, so a test can assert a *specific* table ranks first without depending on any
    real model."""

    def __init__(self, embedding_by_text: dict[str, list[float]]):
        self._embedding_by_text = embedding_by_text

    def encode(self, texts, *, batch_size):
        return np.asarray([self._embedding_by_text[t] for t in texts], dtype=np.float32)


def test_dense_index_normalizes_and_honors_scope(tmp_path):
    index = DenseIndex(["a", "b", "c"], np.asarray([[1, 0], [0, 2], [1, 1]], dtype=np.float32))
    assert [key for key, _ in index.search(np.asarray([0, 1]), top_k=2)] == ["b", "c"]
    assert [key for key, _ in index.search(np.asarray([0, 1]), top_k=2, allowed_positions=[0, 2])] == ["c", "a"]
    index.save(tmp_path)
    assert DenseIndex.load(tmp_path).doc_ids == ["a", "b", "c"]


class _CallRecordingEncoder(FakeEncoder):
    def __init__(self):
        self.calls = []

    def encode(self, texts, *, batch_size):
        self.calls.append((list(texts), batch_size))
        return super().encode(texts, batch_size=batch_size)


def test_build_dense_index_makes_a_single_encode_call_with_batch_size():
    encoder = _CallRecordingEncoder()
    index = build_dense_index(["a", "b", "c"], ["a", "bb", "ccc"], encoder, batch_size=2)
    assert index.embeddings.shape == (3, 2)
    assert encoder.calls == [(["a", "bb", "ccc"], 2)]


def test_build_dense_index_defaults_batch_size_to_32():
    encoder = _CallRecordingEncoder()
    build_dense_index(["a"], ["a"], encoder)
    assert encoder.calls == [(["a"], 32)]


class _FakeSentenceTransformer:
    def __init__(self, model_path, local_files_only=True, device=None, model_kwargs=None):
        self.model_path = model_path
        self.local_files_only = local_files_only
        self.device = device
        self.model_kwargs = model_kwargs
        self.max_seq_length = None


def test_sentence_transformer_encoder_sets_max_seq_length(monkeypatch):
    fake_module = types.SimpleNamespace(SentenceTransformer=_FakeSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    encoder = SentenceTransformerEncoder("/some/model", max_seq_length=777)
    assert encoder._model.max_seq_length == 777


def test_sentence_transformer_encoder_defaults_max_seq_length(monkeypatch):
    fake_module = types.SimpleNamespace(SentenceTransformer=_FakeSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    encoder = SentenceTransformerEncoder("/some/model")
    assert encoder._model.max_seq_length == 1536


def test_sentence_transformer_encoder_uses_fp32_cpu_without_cuda(monkeypatch):
    fake_module = types.SimpleNamespace(SentenceTransformer=_FakeSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    encoder = SentenceTransformerEncoder("/some/model")
    assert encoder._model.device is None
    assert encoder._model.model_kwargs is None


def test_sentence_transformer_encoder_uses_fp16_cuda_when_available(monkeypatch):
    fake_module = types.SimpleNamespace(SentenceTransformer=_FakeSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    encoder = SentenceTransformerEncoder("/some/model")
    assert encoder._model.device == "cuda"
    assert encoder._model.model_kwargs == {"torch_dtype": torch.float16}


def _write_catalog(tmp_path, rows):
    catalog = tmp_path / "tables.csv"
    pd.DataFrame(rows).to_csv(catalog, index=False)
    return catalog


def _write_companies(tmp_path, rows):
    companies = tmp_path / "companies.csv"
    pd.DataFrame(rows).to_csv(companies, index=False)
    return companies


def test_rank_questions_dense_scopes_like_bm25_and_excludes_other_ticker(tmp_path):
    catalog = _write_catalog(tmp_path, [
        {"report_id": "AAA_report", "line_position": 17, "ticker": "AAA", "year": 2023, "variant": "consolidated"},
        {"report_id": "BBB_report", "line_position": 21, "ticker": "BBB", "year": 2023, "variant": "consolidated"},
    ])
    companies = _write_companies(tmp_path, [
        {"ticker": "AAA", "name": "Cong ty AAA"}, {"ticker": "BBB", "name": "Cong ty BBB"},
    ])
    doc_ids = ["AAA_report|17", "BBB_report|21"]
    # BBB's embedding is closer to the query than AAA's -- proves scope, not similarity, decides.
    index = DenseIndex(doc_ids, np.asarray([[1.0, 0.0], [0.99, 0.01]], dtype=np.float32))
    encoder = OneHotEncoder({"Doanh thu AAA nam 2023": [1.0, 0.0]})
    output = tmp_path / "dense_rankings.json"
    rankings = rank_questions_dense(
        [{"id": 1, "question": "Doanh thu AAA nam 2023"}], catalog, companies, index, encoder, output, top_k=5,
    )
    assert [row[0] for row in rankings["1"]] == ["AAA_report|17"]
    assert json.loads(output.with_suffix(".json.meta.json").read_text(encoding="utf-8"))["dense_index_size"] == 2


def test_rank_questions_dense_falls_back_to_unrestricted_search_without_a_ticker(tmp_path):
    catalog = _write_catalog(tmp_path, [
        {"report_id": "AAA_report", "line_position": 17, "ticker": "AAA", "year": 2023, "variant": "consolidated"},
    ])
    companies = _write_companies(tmp_path, [{"ticker": "AAA", "name": "Cong ty AAA"}])
    index = DenseIndex(["AAA_report|17"], np.asarray([[1.0, 0.0]], dtype=np.float32))
    encoder = OneHotEncoder({"doanh thu nam 2023 cua mot cong ty niem yet": [1.0, 0.0]})
    rankings = rank_questions_dense(
        [{"id": 2, "question": "doanh thu nam 2023 cua mot cong ty niem yet"}],
        catalog, companies, index, encoder, tmp_path / "out.json", top_k=5,
    )
    assert rankings["2"] == [["AAA_report|17", 1.0]]


def test_rank_questions_dense_returns_empty_on_resolved_but_filtered_out_scope(tmp_path):
    catalog = _write_catalog(tmp_path, [
        {"report_id": "AAA_2022", "line_position": 1, "ticker": "AAA", "year": 2022, "variant": "consolidated"},
    ])
    companies = _write_companies(tmp_path, [{"ticker": "AAA", "name": "Cong ty AAA"}])
    index = DenseIndex(["AAA_2022|1"], np.asarray([[1.0, 0.0]], dtype=np.float32))
    encoder = OneHotEncoder({"Doanh thu AAA nam 2023": [1.0, 0.0]})
    rankings = rank_questions_dense(
        [{"id": 3, "question": "Doanh thu AAA nam 2023"}], catalog, companies, index, encoder,
        tmp_path / "out.json", top_k=5,
    )
    assert rankings["3"] == []

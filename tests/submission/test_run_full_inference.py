import json

import pandas as pd

import submission.run_full_inference as full


def test_full_inference_persists_and_reuses_success(monkeypatch, tmp_path):
    questions = tmp_path / "questions.jsonl"
    questions.write_text(json.dumps({"id": 1, "question": "q"}) + "\n", encoding="utf-8")
    structured = tmp_path / "tables.jsonl"
    structured.write_text(json.dumps({"table_key": "AAA_report|10"}) + "\n", encoding="utf-8")
    dummy = tmp_path / "dummy"
    dummy.write_text("x", encoding="utf-8")

    monkeypatch.setattr(full, "rank_questions", lambda *args, **kwargs: {"1": [["AAA_report|10", 1.0]]})
    monkeypatch.setattr(full, "QwenAWQGenerator", lambda path: object())
    monkeypatch.setattr(full, "answer_question", lambda *args, **kwargs: {
        "id": 1, "question": "q", "answer": 3.0, "executed": True,
        "relevant_docs": ["AAA_report"], "used_tables": ["AAA_report|10"],
        "pandas_query": "float(df1['x'].iloc[0])", "evidence_frames": {"df1": pd.DataFrame({"x": [3.0]})},
    })
    package = tmp_path / "package"
    report = full.run(dummy, questions, dummy, dummy, structured, package)
    assert report["n_completed"] == report["n_questions"] == 1
    prediction = json.loads((package / "work/predictions/1.json").read_text(encoding="utf-8"))
    assert prediction["relevant_tables"] == ["AAA_report|10"]
    assert prediction["evidence"][0]["csv_path"] == "data/q1_df1.csv"

    monkeypatch.setattr(full, "QwenAWQGenerator", lambda path: (_ for _ in ()).throw(AssertionError("reloaded")))
    assert full.run(dummy, questions, dummy, dummy, structured, package)["n_completed"] == 1
    assert report["retriever"] == "bm25"


def test_full_inference_auto_detects_row_label_index_next_to_catalog(monkeypatch, tmp_path):
    questions = tmp_path / "questions.jsonl"
    questions.write_text(json.dumps({"id": 1, "question": "q"}) + "\n", encoding="utf-8")
    structured = tmp_path / "tables.jsonl"
    structured.write_text(json.dumps({"table_key": "AAA_report|10"}) + "\n", encoding="utf-8")
    catalog_dir = tmp_path / "catalog_dir"
    catalog_dir.mkdir()
    catalog = catalog_dir / "normalized_tables.csv"
    catalog.write_text("x", encoding="utf-8")
    companies = tmp_path / "companies"
    companies.write_text("x", encoding="utf-8")

    captured = {}

    def fake_rank_questions(questions, catalog_path, companies_path, output_path, *, top_k, row_label_index_path=None):
        captured["row_label_index_path"] = row_label_index_path
        return {"1": [["AAA_report|10", 1.0]]}

    monkeypatch.setattr(full, "rank_questions", fake_rank_questions)
    monkeypatch.setattr(full, "QwenAWQGenerator", lambda path: object())
    monkeypatch.setattr(full, "answer_question", lambda *args, **kwargs: {
        "id": 1, "question": "q", "answer": 3.0, "executed": True,
        "relevant_docs": ["AAA_report"], "used_tables": ["AAA_report|10"],
        "pandas_query": "float(df1['x'].iloc[0])", "evidence_frames": {"df1": pd.DataFrame({"x": [3.0]})},
    })

    # No sidecar present next to the catalog -> reranking silently stays off, not an error.
    full.run(catalog, questions, catalog, companies, structured, tmp_path / "package_a")
    assert captured["row_label_index_path"] is None

    # A sidecar present next to the catalog -> auto-detected and passed through.
    row_label_index = catalog_dir / "row_label_index.csv"
    row_label_index.write_text("table_key,row_label_text\n", encoding="utf-8")
    full.run(catalog, questions, catalog, companies, structured, tmp_path / "package_b")
    assert captured["row_label_index_path"] == row_label_index


def test_full_inference_fuses_dense_into_hybrid_when_requested(monkeypatch, tmp_path):
    questions = tmp_path / "questions.jsonl"
    questions.write_text(json.dumps({"id": 1, "question": "q"}) + "\n", encoding="utf-8")
    structured = tmp_path / "tables.jsonl"
    structured.write_text(
        json.dumps({"table_key": "AAA_report|10"}) + "\n" + json.dumps({"table_key": "BBB_report|20"}) + "\n",
        encoding="utf-8",
    )
    dummy = tmp_path / "dummy"
    dummy.write_text("x", encoding="utf-8")

    def fake_rank_questions(questions, catalog_path, companies_path, output_path, *, top_k, row_label_index_path=None):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"1": [["AAA_report|10", 1.0]]}), encoding="utf-8")
        return {"1": [["AAA_report|10", 1.0]]}

    monkeypatch.setattr(full, "rank_questions", fake_rank_questions)

    def fake_run_dense_retrieval(model_path, catalog_path, companies_path, questions_path, index_dir, output_path,
                                  *, top_k, hf_repo_id, hf_revision, hf_cache_dir):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"1": [["BBB_report|20", 0.9]]}), encoding="utf-8")
        return {"1": [["BBB_report|20", 0.9]]}

    monkeypatch.setattr(full, "run_dense_retrieval", fake_run_dense_retrieval)
    monkeypatch.setattr(full, "QwenAWQGenerator", lambda path: object())

    captured_keys = {}

    def fake_answer_question(question, retrieved_table_keys, tables_by_key, complete, *, max_retries):
        captured_keys["retrieved"] = list(retrieved_table_keys)
        return {
            "id": 1, "question": "q", "answer": 3.0, "executed": True,
            "relevant_docs": ["AAA_report"], "used_tables": ["AAA_report|10"],
            "pandas_query": "float(df1['x'].iloc[0])", "evidence_frames": {"df1": pd.DataFrame({"x": [3.0]})},
        }

    monkeypatch.setattr(full, "answer_question", fake_answer_question)
    package = tmp_path / "package"
    report = full.run(
        dummy, questions, dummy, dummy, structured, package,
        dense_index_dir=tmp_path / "dense_index", dense_hf_repo_id="BAAI/bge-m3",
    )
    assert report["retriever"] == "hybrid_bm25_dense_rrf"
    # Both the BM25-only and dense-only keys must appear in the fused ranking handed downstream.
    assert set(captured_keys["retrieved"]) == {"AAA_report|10", "BBB_report|20"}
    hybrid_artifact = json.loads((package / "work" / "retrieval_hybrid_top10.json").read_text(encoding="utf-8"))
    assert set(key for key, _ in hybrid_artifact["1"]) == {"AAA_report|10", "BBB_report|20"}

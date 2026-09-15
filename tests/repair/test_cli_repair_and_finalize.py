import json

import repair.repair_failures as repair_failures
import repair.repair_predictions as repair_predictions
from repair.finalize import finalize
from repair.scan import scan, write_report
from submission.build_submission import build


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _table(**overrides):
    base = {
        "table_key": "AAA_financial_statements_2023_consolidated|10",
        "report_id": "AAA_financial_statements_2023_consolidated",
        "source_path": "financial_statements/AAA/2023/report.txt",
        "line_position": 10, "n_cols": 3, "detected_units": ["VND"],
        "ticker": "AAA", "year": 2023, "variant": "consolidated",
        "grid": [["CHỈ TIÊU", "Năm 2023", "Năm 2022"], ["Doanh thu thuần", "1.234.000", "1.000.000"]],
        "row_labels": [{"row_index": 0, "column_index": 0, "label": "CHỈ TIÊU"},
                       {"row_index": 1, "column_index": 0, "label": "Doanh thu thuần"}],
    }
    base.update(overrides)
    return base


def _patch_corpus(monkeypatch, module, table):
    monkeypatch.setattr(module, "batch_retrieve", lambda questions, *a, **k: {q["id"]: [table["table_key"]] for q in questions})
    monkeypatch.setattr(module, "load_tables_by_key", lambda path, keys: {table["table_key"]: table})
    monkeypatch.setattr(module, "load_company_by_ticker", lambda path: {})


def test_repair_failures_replays_without_any_cached_retrieval_state(tmp_path, monkeypatch):
    """failures/*.json never carries retrieved_tables/schema_linked_operands (§1.2) -- this test
    asserts repair_failures.run() succeeds using ONLY {id, question, error_type, error_message},
    proving the replay path never assumes cached retrieval/linking state."""
    table = _table()
    failures = tmp_path / "failures"
    _write(failures / "42.json", {
        "id": 42, "question": "Doanh thu thuần AAA năm 2023",
        "error_type": "RuntimeError", "error_message": "[{'error_type': 'JSONDecodeError'}]",
    })
    _patch_corpus(monkeypatch, repair_failures, table)

    def fake_complete(prompt):
        return json.dumps({
            "operands": [{"alias": "x", "table_key": table["table_key"], "row_index": 1, "column_index": 1}],
            "expression": "x", "answer_unit": None, "rationale": "lookup",
        })

    output = tmp_path / "repair_run"
    monkeypatch.setattr(repair_failures, "QwenAWQGenerator", lambda path: fake_complete, raising=False)
    summary = repair_failures.run(
        failures, output, catalog_path=tmp_path / "unused.csv", companies_path=tmp_path / "unused2.csv",
        structured_path=tmp_path / "unused.jsonl", model_path=tmp_path / "fake-model", top_k=10,
    )
    assert summary["accepted"] == 1
    recovered = json.loads((output / "accepted" / "42.json").read_text(encoding="utf-8"))
    assert recovered["executed"] is True
    assert recovered["answer"] == 1234000.0
    # the original failure record is never touched
    assert json.loads((failures / "42.json").read_text(encoding="utf-8"))["error_type"] == "RuntimeError"


def test_repair_failures_resume_skips_already_attempted_ids(tmp_path, monkeypatch):
    table = _table()
    failures = tmp_path / "failures"
    _write(failures / "1.json", {
        "id": 1, "question": "Doanh thu thuần AAA năm 2023",
        "error_type": "RuntimeError",
        "error_message": "[{'error_type': 'ValueError', 'error_message': 'no unambiguous source unit'}]",
    })
    _patch_corpus(monkeypatch, repair_failures, table)
    output = tmp_path / "repair_run"

    calls = {"n": 0}
    real_batch_retrieve = repair_failures.batch_retrieve

    def counting_batch_retrieve(questions, *a, **k):
        calls["n"] += 1
        return real_batch_retrieve(questions, *a, **k)

    monkeypatch.setattr(repair_failures, "batch_retrieve", counting_batch_retrieve)

    summary1 = repair_failures.run(
        failures, output, catalog_path=tmp_path / "u.csv", companies_path=tmp_path / "u2.csv",
        structured_path=tmp_path / "u.jsonl", model_path=None,
    )
    assert summary1["accepted"] == 1
    assert calls["n"] == 1

    summary2 = repair_failures.run(
        failures, output, catalog_path=tmp_path / "u.csv", companies_path=tmp_path / "u2.csv",
        structured_path=tmp_path / "u.jsonl", model_path=None, resume=True,
    )
    assert summary2["n_pending"] == 0
    assert calls["n"] == 1  # no new retrieval call -- id 1 was skipped, not re-processed


def test_repair_failures_unsupported_shape_is_never_attempted(tmp_path, monkeypatch):
    table = _table()
    failures = tmp_path / "failures"
    _write(failures / "7.json", {
        "id": 7, "question": "Công ty nào có doanh thu cao nhất năm 2023?",
        "error_type": "RuntimeError", "error_message": "[]",
    })
    _patch_corpus(monkeypatch, repair_failures, table)
    output = tmp_path / "repair_run"
    summary = repair_failures.run(
        failures, output, catalog_path=tmp_path / "u.csv", companies_path=tmp_path / "u2.csv",
        structured_path=tmp_path / "u.jsonl",
    )
    assert summary["unsupported"] == 1
    assert not (output / "accepted" / "7.json").exists()
    lines = (output / "unsupported.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["id"] == 7


def test_repair_predictions_only_touches_flagged_ids_and_preserves_originals(tmp_path, monkeypatch):
    table = _table()
    predictions = tmp_path / "predictions"
    _write(predictions / "1.json", {
        "id": 1, "question": "Doanh thu thuần AAA năm 2023", "query_family": "aggregation",
        "used_tables": [table["table_key"]],
        "pandas_query": "float(df1.loc[(df1['row_index']==1) & (df1['column_index']==1), 'numeric_value'].iloc[0])",
        "answer": 1234000.0, "executed": True, "retrieved_tables": [table["table_key"]],
    })
    _write(predictions / "2.json", {
        "id": 2, "question": "Doanh thu thuần AAA năm 2023", "query_family": "direct_lookup",
        "used_tables": [table["table_key"]], "pandas_query": "float(df1.loc[...].iloc[0])",
        "answer": 1234000.0, "executed": True, "retrieved_tables": [table["table_key"]],
    })
    before_1 = (predictions / "1.json").read_text(encoding="utf-8")
    before_2 = (predictions / "2.json").read_text(encoding="utf-8")

    no_failures = tmp_path / "no_failures"
    no_failures.mkdir()
    diagnostics = tmp_path / "diagnostics.jsonl"
    write_report(scan(predictions, no_failures), diagnostics)

    _patch_corpus(monkeypatch, repair_predictions, table)

    output = tmp_path / "repair_run"
    summary = repair_predictions.run(
        predictions, diagnostics, output, catalog_path=tmp_path / "u.csv",
        companies_path=tmp_path / "u2.csv", structured_path=tmp_path / "u.jsonl", model_path=None,
    )
    # id=1 (aggregation, single table -> T1) is flagged; id=2 (direct_lookup, no trigger) is not.
    assert summary["n_flagged"] == 1
    assert (predictions / "1.json").read_text(encoding="utf-8") == before_1
    assert (predictions / "2.json").read_text(encoding="utf-8") == before_2


def test_finalize_merges_accepted_over_original_and_adds_recovered_failures(tmp_path):
    predictions = tmp_path / "predictions"
    _write(predictions / "1.json", {"id": 1, "question": "q1", "answer": 1.0, "executed": True})
    _write(predictions / "2.json", {"id": 2, "question": "q2", "answer": 2.0, "executed": True})
    repair_run = tmp_path / "repair_run"
    _write(repair_run / "accepted" / "1.json", {"id": 1, "question": "q1", "answer": 999.0, "executed": True})
    _write(repair_run / "accepted" / "3.json", {"id": 3, "question": "q3", "answer": 3.0, "executed": True})
    output = tmp_path / "repaired_predictions"

    report = finalize(predictions, repair_run, output)
    assert report["repaired_ids"] == [1]
    assert report["recovered_ids"] == [3]

    merged_1 = json.loads((output / "1.json").read_text(encoding="utf-8"))
    assert merged_1["answer"] == 999.0  # repaired wins over original
    merged_2 = json.loads((output / "2.json").read_text(encoding="utf-8"))
    assert merged_2["answer"] == 2.0  # untouched original carried through
    assert (output / "3.json").exists()  # recovered failure now present as a prediction

    # originals never modified
    assert json.loads((predictions / "1.json").read_text(encoding="utf-8"))["answer"] == 1.0


def test_finalize_output_is_compatible_with_build_submission(tmp_path):
    predictions = tmp_path / "predictions"
    prediction = {
        "id": 1, "question": "q1", "answer": 1.0, "executed": True,
        "relevant_docs": ["AAA_report"], "relevant_tables": ["AAA_report|10"],
        "evidence": [{"variable": "df1", "csv_path": "data/q1_df1.csv"}],
        "pandas_query": "float(df1['x'].iloc[0])",
    }
    _write(predictions / "1.json", prediction)
    repair_run = tmp_path / "repair_run"
    output = tmp_path / "repaired_predictions"
    finalize(predictions, repair_run, output)  # no accepted/ directory at all -- must still work

    questions = tmp_path / "questions.jsonl"
    questions.write_text(json.dumps({"id": 1, "question": "q1"}) + "\n", encoding="utf-8")
    submission_out = tmp_path / "submission.json"
    records = build(questions, output, submission_out)
    assert records == [{k: prediction[k] for k in (
        "id", "question", "answer", "relevant_docs", "relevant_tables", "evidence", "pandas_query"
    )}]

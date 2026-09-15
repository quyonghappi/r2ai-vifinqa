import json

from repair.scan import scan, summarize, write_report


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_scan_diagnoses_predictions_and_failures_and_is_read_only(tmp_path):
    predictions = tmp_path / "predictions"
    failures = tmp_path / "failures"
    _write(predictions / "1.json", {
        "id": 1, "question": "Tổng doanh thu của AAA và BBB năm 2023",
        "query_family": "aggregation", "used_tables": ["AAA_financial_statements_2023_consolidated|10"],
        "pandas_query": "float(df1.loc[...].iloc[0])", "answer": 1.0,
    })
    _write(predictions / "2.json", {
        "id": 2, "question": "Doanh thu thuần AAA năm 2023", "query_family": "direct_lookup",
        "used_tables": ["AAA_financial_statements_2023_consolidated|10"],
        "pandas_query": "float(df1.loc[...].iloc[0])", "answer": 1234000.0,
    })
    _write(failures / "3.json", {
        "id": 3, "question": "Doanh thu thuần AAA năm 2023",
        "error_type": "RuntimeError",
        "error_message": "[{'error_type': 'JSONDecodeError'}]",
    })
    before_predictions = (predictions / "1.json").read_text(encoding="utf-8")
    before_failures = (failures / "3.json").read_text(encoding="utf-8")

    rows = scan(predictions, failures)
    output = tmp_path / "diagnostics.jsonl"
    write_report(rows, output)

    lines = [json.loads(l) for l in output.read_text(encoding="utf-8").splitlines()]
    assert {r["id"] for r in lines} == {1, 2, 3}
    by_id = {r["id"]: r for r in lines}
    assert by_id[1]["source"] == "prediction" and by_id[1]["action"] == "regenerate"
    assert by_id[3]["source"] == "failure" and by_id[3]["action"] == "regenerate"

    summary = summarize(rows)
    assert summary["n_predictions"] == 2 and summary["n_failures"] == 1
    assert summary["trigger_counts"].get("T1") == 1

    # scan never touches predictions/ or failures/
    assert (predictions / "1.json").read_text(encoding="utf-8") == before_predictions
    assert (failures / "3.json").read_text(encoding="utf-8") == before_failures


def test_scan_is_atomic_and_overwrites_cleanly_on_rerun(tmp_path):
    predictions = tmp_path / "predictions"
    failures = tmp_path / "failures"
    predictions.mkdir()
    failures.mkdir()
    output = tmp_path / "diagnostics.jsonl"
    write_report([{"id": 1}], output)
    write_report([{"id": 2}, {"id": 3}], output)
    lines = [json.loads(l) for l in output.read_text(encoding="utf-8").splitlines()]
    assert [l["id"] for l in lines] == [2, 3]
    assert not output.with_suffix(output.suffix + ".tmp").exists()

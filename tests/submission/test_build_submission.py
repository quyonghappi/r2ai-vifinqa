import json

import pytest

from submission.build_submission import build


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_build_is_formatter_and_refuses_missing_predictions(tmp_path):
    questions = tmp_path / "questions.jsonl"
    questions.write_text(json.dumps({"id": 1, "question": "q"}) + "\n", encoding="utf-8")
    predictions = tmp_path / "predictions"
    prediction = {
        "id": 1, "question": "q", "answer": 3.0, "executed": True,
        "relevant_docs": ["AAA_report"], "relevant_tables": ["AAA_report|10"],
        "evidence": [{"variable": "df1", "csv_path": "data/q1_df1.csv"}],
        "pandas_query": "float(df1['x'].iloc[0])",
    }
    _write_json(predictions / "1.json", prediction)
    output = tmp_path / "submission.json"
    records = build(questions, predictions, output)
    assert records == [{key: prediction[key] for key in (
        "id", "question", "answer", "relevant_docs", "relevant_tables", "evidence", "pandas_query"
    )}]
    (predictions / "1.json").unlink()
    with pytest.raises(ValueError, match="missing prediction"):
        build(questions, predictions, output)

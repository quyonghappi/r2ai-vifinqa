import json

import pandas as pd

from submission.validate_submission import validate


def _fixture(tmp_path):
    questions = tmp_path / "questions.jsonl"
    questions.write_text(json.dumps({"id": 1, "question": "q"}) + "\n", encoding="utf-8")
    data = tmp_path / "data"
    data.mkdir()
    pd.DataFrame({"x": [3.0]}).to_csv(data / "q1_df1.csv", index=False)
    record = {
        "id": 1, "question": "q", "answer": 3.0,
        "relevant_docs": ["AAA_report"], "relevant_tables": ["AAA_report|10"],
        "evidence": [{"variable": "df1", "csv_path": "data/q1_df1.csv"}],
        "pandas_query": "float(df1['x'].iloc[0])",
    }
    submission = tmp_path / "submission.json"
    submission.write_text(json.dumps([record]), encoding="utf-8")
    return questions, submission, record


def test_validator_executes_query_and_checks_exact_coverage(tmp_path):
    questions, submission, _ = _fixture(tmp_path)
    report = validate(submission, questions)
    assert report["valid"] is True
    assert report["n_records"] == report["n_official"] == 1


def test_validator_rejects_path_escape_and_answer_mismatch(tmp_path):
    questions, submission, record = _fixture(tmp_path)
    record["evidence"][0]["csv_path"] = "data/../outside.csv"
    submission.write_text(json.dumps([record]), encoding="utf-8")
    assert not validate(submission, questions)["valid"]

    record["evidence"][0]["csv_path"] = "data/q1_df1.csv"
    record["answer"] = 4.0
    submission.write_text(json.dumps([record]), encoding="utf-8")
    report = validate(submission, questions)
    assert not report["valid"]
    assert "does not match answer" in report["errors"][0]


def test_validator_rejects_missing_official_question(tmp_path):
    questions, submission, _ = _fixture(tmp_path)
    questions.write_text(
        json.dumps({"id": 1, "question": "q"}) + "\n" + json.dumps({"id": 2, "question": "q2"}) + "\n",
        encoding="utf-8",
    )
    report = validate(submission, questions)
    assert not report["valid"]
    assert "missing official ids" in report["errors"][-1]

import json

from eval.relevant_tables_k_calibration import run


def _operand(table_key: str, relevance_score: float) -> dict:
    return {
        "table_key": table_key, "report_id": table_key.split("|")[0], "source_path": "x",
        "line_position": 1, "row_index": 0, "column_index": 1, "row_label": "x", "column_header": "x",
        "raw_value": "1", "numeric_value": 1.0, "detected_units": [], "relevance_score": relevance_score,
    }


def test_calibration_scores_each_design_against_gold_and_skips_zero_gold_records(tmp_path):
    fixture = [
        {"id": 1, "gold_n": 2, "gold_tables": ["A|1", "B|2"]},
        {"id": 2, "gold_n": 0, "gold_tables": []},
    ]
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")

    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir()
    (predictions_dir / "1.json").write_text(json.dumps({
        "retrieved_tables": ["A|1", "B|2", "C|3"],
        "schema_linked_operands": [_operand("A|1", 0.9), _operand("C|3", 0.6), _operand("B|2", 0.3)],
        "used_tables": ["A|1"],
    }), encoding="utf-8")

    report = run(fixture_path, predictions_dir)

    assert report["n_oracle_questions_total"] == 2
    assert report["n_oracle_questions_zero_gold_skipped"] == 1
    assert report["n_oracle_questions_scored"] == 1
    used = report["results"]["used_tables(current_production)"]
    assert used["macro_precision"] == 1.0 and used["macro_recall"] == 0.5
    top2 = report["results"]["top8_tables"]  # only 3 operands exist, top_k=8 caps at all 3 tables
    assert top2["macro_recall"] == 1.0  # all 3 retrieved tables cover both gold tables


def test_calibration_raises_on_missing_prediction_file(tmp_path):
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(json.dumps([{"id": 99, "gold_n": 1, "gold_tables": ["A|1"]}]), encoding="utf-8")
    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir()
    try:
        run(fixture_path, predictions_dir)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass

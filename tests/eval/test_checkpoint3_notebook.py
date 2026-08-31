"""Regression checks for the self-contained Kaggle Checkpoint-3 cell."""

from __future__ import annotations

import json
from pathlib import Path


def test_regression_diagnostics_read_correctness_from_evaluation_report():
    notebook_path = Path(__file__).resolve().parents[2] / "vifinqa.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8-sig"))
    cell = next(cell for cell in notebook["cells"] if cell.get("id") == "checkpoint3-kaggle-run")
    source = "".join(cell["source"])

    assert "scores_by_id = {score['id']: score for score in generation_report['per_query']}" in source
    assert "'answer_correct': scores_by_id[question_id]['answer_correct']" in source
    assert "records_by_id[question_id]['answer_correct']" not in source

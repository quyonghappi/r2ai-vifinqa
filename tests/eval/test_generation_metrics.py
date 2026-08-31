import pytest

from eval.generation_metrics import evaluate_answers


def test_execution_and_answer_metrics_with_family_breakdown():
    report = evaluate_answers([
        {"id": 1, "query_family": "direct_lookup", "executed": True, "answer": 10.0, "expected_answer": 10.0},
        {"id": 2, "query_family": "growth", "executed": False, "answer": None, "expected_answer": 20.0},
    ])
    assert report["execution_accuracy"] == 0.5
    assert report["answer_accuracy"] == 0.5
    assert report["by_query_family"]["growth"]["execution_accuracy"] == 0.0


def test_missing_manual_answer_is_not_silently_excluded():
    with pytest.raises(ValueError, match="manually verified"):
        evaluate_answers([{"id": 1, "query_family": "direct_lookup", "executed": True, "answer": 1.0}])

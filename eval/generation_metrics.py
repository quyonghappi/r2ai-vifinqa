"""Local diagnostic execution/answer metrics; never represents official ground truth."""

from __future__ import annotations

import math


def numeric_match(actual: float | None, expected: float, *, abs_tol: float, rel_tol: float) -> bool:
    return actual is not None and math.isclose(actual, expected, abs_tol=abs_tol, rel_tol=rel_tol)


def evaluate_answers(records: list[dict], *, abs_tol: float = 1e-6, rel_tol: float = 1e-6) -> dict:
    """Development assumption: math.isclose with explicit tolerances until BTC publishes theirs."""
    per_query = []
    for record in records:
        if "expected_answer" not in record:
            raise ValueError(f"question {record.get('id')} has no manually verified expected_answer")
        correct = numeric_match(record.get("answer"), record["expected_answer"], abs_tol=abs_tol, rel_tol=rel_tol)
        per_query.append({
            "id": record["id"], "query_family": record["query_family"],
            "executed": bool(record.get("executed")), "answer_correct": correct,
            "execution_correct": bool(record.get("executed")) and correct,
        })
    n = len(per_query)
    families = {}
    for family in sorted({x["query_family"] for x in per_query}):
        rows = [x for x in per_query if x["query_family"] == family]
        families[family] = {
            "n": len(rows),
            "execution_accuracy": sum(x["execution_correct"] for x in rows) / len(rows),
            "answer_accuracy": sum(x["answer_correct"] for x in rows) / len(rows),
        }
    return {
        "n_queries": n, "abs_tolerance_assumption": abs_tol, "rel_tolerance_assumption": rel_tol,
        "execution_accuracy": sum(x["execution_correct"] for x in per_query) / n if n else 0.0,
        "answer_accuracy": sum(x["answer_correct"] for x in per_query) / n if n else 0.0,
        "by_query_family": families, "per_query": per_query,
    }

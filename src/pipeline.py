"""Straightforward Checkpoint-3 orchestration; stage implementations remain independent."""

from __future__ import annotations

from dataclasses import asdict
from typing import Callable

from execution.runner import execute_with_repair
from schema_linking.linker import link_schema


def answer_question(
    question: dict,
    retrieved_table_keys: list[str],
    tables_by_key: dict[str, dict],
    complete: Callable[[str], str],
    *,
    max_retries: int = 2,
) -> dict:
    """Run linking -> generation -> execution/repair for one already-retrieved question."""
    missing = [key for key in retrieved_table_keys if key not in tables_by_key]
    if missing:
        raise KeyError(f"retrieval referenced unavailable structured tables: {missing}")
    tables = [tables_by_key[key] for key in retrieved_table_keys]
    linked = link_schema(question["question"], tables)
    result = execute_with_repair(linked, tables_by_key, complete, max_retries=max_retries)
    plan = result.plan
    used_keys = list(dict.fromkeys(x.table_key for x in plan.operands)) if plan else []
    return {
        "id": question["id"],
        "question": question["question"],
        "query_family": linked.query_family,
        "retrieved_tables": retrieved_table_keys,
        "schema_linked_operands": [asdict(x) for x in linked.operands],
        "used_tables": used_keys,
        "relevant_docs": list(dict.fromkeys(key.rsplit("|", 1)[0] for key in used_keys)),
        "answer": result.answer,
        "pandas_query": result.pandas_query,
        "executed": result.success,
        "attempts": [asdict(x) for x in result.attempts],
        "answer_unit": plan.answer_unit if plan else None,
        "model_rationale": plan.rationale if plan else None,
        "evidence_frames": result.evidence_frames,
    }

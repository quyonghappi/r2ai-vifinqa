"""Straightforward Checkpoint-3 orchestration; stage implementations remain independent."""

from __future__ import annotations

from dataclasses import asdict
from typing import Callable

from execution.runner import execute_with_repair
from schema_linking.linker import link_schema, select_relevant_tables


def answer_question(
    question: dict,
    retrieved_table_keys: list[str],
    tables_by_key: dict[str, dict],
    complete: Callable[[str], str],
    *,
    max_retries: int = 2,
    company_by_ticker: dict[str, str] | None = None,
) -> dict:
    """Run linking -> generation -> execution/repair for one already-retrieved question.

    ``company_by_ticker`` (optional, ticker -> company name) lets ``link_schema`` resolve this
    question's entity scope canonically instead of re-deriving it downstream from raw text
    (EXECUTION_ACCURACY_AUDIT.md §6.1); omitting it preserves prior behavior unchanged.
    """
    missing = [key for key in retrieved_table_keys if key not in tables_by_key]
    if missing:
        raise KeyError(f"retrieval referenced unavailable structured tables: {missing}")
    tables = [tables_by_key[key] for key in retrieved_table_keys]
    linked = link_schema(question["question"], tables, company_by_ticker=company_by_ticker)
    result = execute_with_repair(linked, tables_by_key, complete, max_retries=max_retries)
    plan = result.plan
    used_keys = list(dict.fromkeys(x.table_key for x in plan.operands)) if plan else []
    # relevant_tables is a recall-oriented prediction of which tables hold needed evidence
    # (submission_guide.md's own definition), derived from schema-linking's full ranked
    # candidate pool -- deliberately NOT used_keys (the execution-provenance set below, which a
    # 2026-09-01 diagnostic audit found collapses to exactly 1 table for 96% of successful
    # predictions regardless of how many tables a question actually needs; see
    # schema_linking.linker.select_relevant_tables's docstring and TABLES_F2_AUDIT.md).
    # relevant_docs is this same evidence judgment at document granularity, not used_keys's.
    relevant_tables = select_relevant_tables(linked.operands)
    return {
        "id": question["id"],
        "question": question["question"],
        "query_family": linked.query_family,
        "retrieved_tables": retrieved_table_keys,
        "schema_linked_operands": [asdict(x) for x in linked.operands],
        "used_tables": used_keys,
        "relevant_tables": relevant_tables,
        "relevant_docs": list(dict.fromkeys(key.rsplit("|", 1)[0] for key in relevant_tables)),
        "answer": result.answer,
        "pandas_query": result.pandas_query,
        "executed": result.success,
        "attempts": [asdict(x) for x in result.attempts],
        "answer_unit": plan.answer_unit if plan else None,
        "model_rationale": plan.rationale if plan else None,
        "evidence_frames": result.evidence_frames,
    }

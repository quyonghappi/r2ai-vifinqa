"""Compact, defensible per-question + aggregate evaluator for the pipeline diagnostic fixture.

Composes directly from the same real functions `pipeline.answer_question` uses
(`schema_linking.linker.link_schema`/`select_relevant_tables`,
`execution.runner.execute_with_repair`) rather than calling `pipeline.answer_question` itself,
for one reason only: `answer_question` discards the `ExecutionResult.plan` object after pulling a
few scalar fields off it, so `plan.groups`/`plan.reductions` (needed for this fixture's
groups/reductions diagnostic) are not otherwise reachable. `answer_with_plan` below performs the
exact same three-call composition, in the same order, as `pipeline.answer_question` -- verified
against its real source -- not a reimplementation of any decision logic, just one that keeps the
plan instead of discarding it.

No official ground truth: every score/diagnostic here is against the self-constructed
`eval/dev_questions/pipeline_fixture_v1.json`, never the competition's held-out answers. See
EVAL_FIXTURE_AUDIT.md for the metric-spec verification and fixture-trust status this depends on.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Callable

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from common.table_store import load_tables_by_key  # noqa: E402
from execution.runner import execute_with_repair  # noqa: E402
from retrieval.full_corpus import load_company_by_ticker, rank_questions  # noqa: E402
from schema_linking.linker import link_schema, select_relevant_tables  # noqa: E402

from eval.generation_metrics import evaluate_answers  # noqa: E402
from eval.metrics import evaluate_retrieval_full, project_zero_credit_macro  # noqa: E402

DEFAULT_FIXTURE = _REPO_ROOT / "eval" / "dev_questions" / "pipeline_fixture_v1.json"
DEFAULT_CATALOG = _REPO_ROOT / "data" / "processed" / "normalized_tables.csv"
DEFAULT_COMPANIES = _REPO_ROOT / "data" / "raw" / "hf_meta" / "code_stock.csv"
DEFAULT_ROW_LABEL_INDEX = _REPO_ROOT / "data" / "processed" / "row_label_index.csv"
DEFAULT_STRUCTURED_TABLES = _REPO_ROOT / "data" / "processed" / "normalized_tables.jsonl"


class GenerationNotExercised(RuntimeError):
    """Raised by the deterministic-mode `complete` stub if generation is ever actually invoked."""


def deterministic_complete_stub(prompt: str) -> str:
    raise GenerationNotExercised(
        "deterministic mode's complete() stub was invoked -- expected only for non-direct_lookup "
        "families, where it is caught by execute_with_repair and surfaces as "
        "generation_not_exercised, never as a faked score"
    )


_VALIDATION_ERROR_CATEGORIES = {
    "incomplete operand group(s)": "incomplete_operand_group",
    "invented or unresolved operand": "invented_or_unresolved_operand",
    "unknown reduction op": "unknown_reduction_op",
    "requires a threshold": "reduction_missing_threshold",
    "references unknown group": "reduction_references_unknown_group",
    "group names must be unique": "duplicate_group_names",
    "every group requires at least one operand": "empty_group",
    "plan requires at least one operand": "empty_plan",
    "unique Python identifier aliases": "alias_collision",
}


def classify_validation_error(error_message: str | None) -> str | None:
    """Map a `validate_plan` `ValueError` message to a stable category name, else None (either
    no error occurred, or the error came from elsewhere -- JSON parsing, execution, etc.)."""
    if not error_message:
        return None
    for needle, category in _VALIDATION_ERROR_CATEGORIES.items():
        if needle in error_message:
            return category
    return None


def answer_with_plan(
    question: dict,
    retrieved_table_keys: list[str],
    tables_by_key: dict[str, dict],
    complete: Callable[[str], str],
    *,
    max_retries: int = 0,
    company_by_ticker: dict[str, str] | None = None,
):
    """Same composition as `pipeline.answer_question` (link_schema -> execute_with_repair ->
    select_relevant_tables), but returns the raw `ExecutionResult` (with `.plan`) alongside the
    submission-shaped dict instead of discarding it. `company_by_ticker` is threaded through to
    `link_schema` for canonical entity resolution (EXECUTION_ACCURACY_AUDIT.md §6.1); omitting it
    preserves the previous literal-token behavior."""
    tables = [tables_by_key[key] for key in retrieved_table_keys if key in tables_by_key]
    linked = link_schema(question["question"], tables, company_by_ticker=company_by_ticker)
    result = execute_with_repair(linked, tables_by_key, complete, max_retries=max_retries)
    plan = result.plan
    used_keys = list(dict.fromkeys(x.table_key for x in plan.operands)) if plan else []
    relevant_tables = select_relevant_tables(linked.operands)
    record = {
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
    }
    return record, result


def diagnose_question(fixture_record: dict, record: dict, execution_result) -> dict:
    """Per-question diagnostics: expected-vs-actual at every stage, groups/reductions,
    completeness-validation outcome, executed-but-wrong vs. generation/repair failure, and the
    first pipeline stage that lost a required gold table or failed outright."""
    gold = fixture_record["gold"]
    gold_tables = set(gold.get("tables") or [])
    retrieved = set(record["retrieved_tables"])
    linked_keys = {op["table_key"] for op in record["schema_linked_operands"]}
    used = set(record["used_tables"])
    executed = record["executed"]
    answer = record["answer"]
    expected = gold.get("expected_answer")

    tol = gold.get("answer_tolerance") or {}
    answer_correct = None
    if expected is not None and executed and answer is not None:
        answer_correct = math.isclose(
            answer, expected, abs_tol=tol.get("abs_tol", 1e-6), rel_tol=tol.get("rel_tol", 1e-6)
        )

    last_attempt = record["attempts"][-1] if record["attempts"] else None
    last_stage = last_attempt["stage"] if last_attempt else None
    last_error_type = last_attempt["error_type"] if last_attempt else None
    last_error_message = last_attempt["error_message"] if last_attempt else None
    validation_error_category = classify_validation_error(last_error_message)

    if not gold_tables:
        first_incorrect_stage = "unscored_no_gold_tables"
    elif gold_tables - retrieved:
        first_incorrect_stage = "retrieval"
    elif gold_tables - linked_keys:
        first_incorrect_stage = "schema_linking"
    elif not executed:
        first_incorrect_stage = f"generation:{last_stage or 'unknown'}"
    elif answer_correct is False:
        first_incorrect_stage = "correctness"
    elif answer_correct is True:
        first_incorrect_stage = "correct"
    else:
        first_incorrect_stage = "unscored_no_expected_answer"

    if executed and answer_correct is False:
        outcome = "EXECUTED_WRONG"
    elif executed and answer_correct is True:
        outcome = "CORRECT"
    elif executed and answer_correct is None:
        outcome = "EXECUTED_UNSCORED"
    elif not executed and last_stage == "generation_or_validation":
        outcome = "GENERATION_OR_VALIDATION_FAILED"
    elif not executed and last_stage == "render":
        outcome = "RENDER_FAILED"
    elif not executed and last_stage == "execution":
        outcome = "EXECUTION_CRASHED"
    else:
        outcome = "NOT_EXECUTED_UNKNOWN"

    plan = execution_result.plan
    groups_reductions = {
        "generated_groups": [
            {"name": g.name, "metric_label": g.metric_label, "n_operands": len(g.operands)}
            for g in (plan.groups if plan else [])
        ],
        "generated_reductions": [asdict(r) for r in (plan.reductions if plan else [])],
        "expected_group": gold.get("expected_group"),
        "expected_reduction": gold.get("expected_reduction"),
    }

    expected_family = fixture_record.get("pipeline_family")
    actual_family = record["query_family"]

    return {
        "id": fixture_record["id"],
        "question_shape": fixture_record.get("question_shape"),
        "expected_pipeline_family": expected_family,
        "actual_pipeline_family": actual_family,
        "family_classification_matches": (
            expected_family == actual_family if expected_family else None
        ),
        "gold_tables": sorted(gold_tables),
        "retrieved_tables": sorted(retrieved),
        "missing_at_retrieval": sorted(gold_tables - retrieved),
        "missing_at_schema_linking": sorted(gold_tables - linked_keys),
        "used_tables": sorted(used),
        "executed": executed,
        "answer": answer,
        "expected_answer": expected,
        "answer_correct": answer_correct,
        "outcome": outcome,
        "first_incorrect_stage": first_incorrect_stage,
        "last_attempt_stage": last_stage,
        "last_attempt_error_type": last_error_type,
        "validation_error_category": validation_error_category,
        **groups_reductions,
    }


def evaluate_records(
    fixture: list[dict],
    retrieved_by_id: dict,
    tables_by_key: dict[str, dict],
    complete: Callable[[str], str] = deterministic_complete_stub,
    *,
    max_retries: int = 0,
    company_by_ticker: dict[str, str] | None = None,
) -> dict:
    """Core evaluation loop with retrieval/tables injected -- lets tests exercise real pipeline
    code against a small synthetic fixture with no real corpus or real BM25 retrieval needed
    (AGENTS.md §6). `run_deterministic`/`run_full_pipeline` below are thin wrappers that supply
    real retrieval and a real or stub `complete`."""
    per_question = []
    for fixture_record in fixture:
        qid = fixture_record["id"]
        retrieved = retrieved_by_id.get(qid, [])
        question = {"id": qid, "question": fixture_record["question"]}
        record, execution_result = answer_with_plan(
            question, retrieved, tables_by_key, complete, max_retries=max_retries,
            company_by_ticker=company_by_ticker,
        )
        per_question.append(diagnose_question(fixture_record, record, execution_result))

    retrieval_queries = [
        {"id": r["id"], "relevant_tables": r["gold"]["tables"]} for r in fixture if r["gold"].get("tables")
    ]
    retrieval_retrieved_by_id = {r["id"]: retrieved_by_id.get(r["id"], []) for r in fixture}
    retrieval_metrics = (
        evaluate_retrieval_full(retrieval_queries, retrieval_retrieved_by_id) if retrieval_queries else None
    )
    if retrieval_metrics is not None and len(retrieval_queries) < len(fixture):
        subset_summary = {
            "n_queries": retrieval_metrics["n_queries"],
            "precision_macro": retrieval_metrics["TABLES_PRECISION"],
            "recall_macro": retrieval_metrics["TABLES_RECALL"],
            "f2_macro": retrieval_metrics["TABLES_F2MACRO"],
        }
        retrieval_metrics["zero_credit_projection_over_full_fixture"] = project_zero_credit_macro(
            subset_summary, total_questions=len(fixture)
        )

    # Answer/execution accuracy is only computed over records with a real expected_answer whose
    # ACTUAL classified family is direct_lookup -- the only family the deterministic stub
    # guarantees a genuine (non-stub-triggered) outcome for. Every other scoreable record is
    # reported separately as not-exercised, never folded silently into the accuracy number.
    exercised = [
        {
            "id": pq["id"], "query_family": pq["actual_pipeline_family"], "executed": pq["executed"],
            "answer": pq["answer"], "expected_answer": pq["expected_answer"],
        }
        for pq in per_question
        if pq["expected_answer"] is not None and pq["actual_pipeline_family"] == "direct_lookup"
    ]
    not_exercised_ids = [
        pq["id"] for pq in per_question
        if pq["expected_answer"] is not None and pq["actual_pipeline_family"] != "direct_lookup"
    ]
    answer_metrics = evaluate_answers(exercised) if exercised else None

    return {
        "banner": "self-constructed diagnostic fixture, not official ground truth -- see EVAL_FIXTURE_AUDIT.md",
        "n_questions": len(fixture),
        "retrieval_metrics": retrieval_metrics,
        "answer_execution_metrics": {
            "n_exercised_direct_lookup_with_expected_answer": len(exercised),
            "n_not_exercised_generation_required": len(not_exercised_ids),
            "not_exercised_ids": not_exercised_ids,
            "metrics": answer_metrics,
        },
        "failure_taxonomy": dict(Counter(pq["outcome"] for pq in per_question)),
        "first_incorrect_stage_histogram": dict(Counter(pq["first_incorrect_stage"] for pq in per_question)),
        "family_classification_mismatches": [
            pq["id"] for pq in per_question if pq["family_classification_matches"] is False
        ],
        "per_question": per_question,
    }


def _rank_and_load(
    fixture: list[dict], catalog_path, companies_path, rankings_path, structured_tables_path,
    row_label_index_path, top_k,
) -> tuple[dict, dict[str, dict]]:
    questions = [{"id": r["id"], "question": r["question"]} for r in fixture]
    rankings = rank_questions(
        questions, catalog_path, companies_path, rankings_path,
        top_k=top_k, row_label_index_path=row_label_index_path,
    )
    retrieved_by_id = {r["id"]: [item[0] for item in rankings[str(r["id"])]] for r in fixture}
    all_keys = {key for keys in retrieved_by_id.values() for key in keys}
    tables_by_key = load_tables_by_key(structured_tables_path, all_keys)
    return retrieved_by_id, tables_by_key


def run_deterministic(
    *,
    fixture_path: Path = DEFAULT_FIXTURE,
    catalog_path: Path = DEFAULT_CATALOG,
    companies_path: Path = DEFAULT_COMPANIES,
    row_label_index_path: Path = DEFAULT_ROW_LABEL_INDEX,
    structured_tables_path: Path = DEFAULT_STRUCTURED_TABLES,
    rankings_path: Path,
    retrieval_top_k: int = 10,
) -> dict:
    """Real retrieval + real schema-linking + real deterministic direct_lookup generation, all
    runnable with no GPU. Non-direct_lookup families get real retrieval/linking diagnostics and
    an explicit not-exercised marker -- never a faked score."""
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    retrieved_by_id, tables_by_key = _rank_and_load(
        fixture, catalog_path, companies_path, rankings_path, structured_tables_path,
        row_label_index_path, retrieval_top_k,
    )
    company_by_ticker = load_company_by_ticker(companies_path)
    report = evaluate_records(
        fixture, retrieved_by_id, tables_by_key, deterministic_complete_stub,
        company_by_ticker=company_by_ticker,
    )
    report["mode"] = "deterministic_no_gpu"
    report["fixture_path"] = str(fixture_path)
    return report


def run_full_pipeline(
    *,
    fixture_path: Path = DEFAULT_FIXTURE,
    catalog_path: Path = DEFAULT_CATALOG,
    companies_path: Path = DEFAULT_COMPANIES,
    row_label_index_path: Path = DEFAULT_ROW_LABEL_INDEX,
    structured_tables_path: Path = DEFAULT_STRUCTURED_TABLES,
    rankings_path: Path,
    retrieval_top_k: int = 10,
    model_path: str | None = None,
) -> dict:
    """Same as `run_deterministic` but with a real generator for every family. Requires a GPU --
    not runnable in this environment (see EVAL_FIXTURE_AUDIT.md). `model_path` is loaded lazily,
    only if the fixture actually needs generation, fixing `run_generation_eval.py`'s unconditional
    `QwenAWQGenerator` instantiation even for direct_lookup-only runs."""
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    retrieved_by_id, tables_by_key = _rank_and_load(
        fixture, catalog_path, companies_path, rankings_path, structured_tables_path,
        row_label_index_path, retrieval_top_k,
    )

    needs_generation = any(r.get("pipeline_family") != "direct_lookup" for r in fixture)
    complete = deterministic_complete_stub
    if needs_generation:
        if not model_path:
            raise ValueError("fixture contains non-direct_lookup records; model_path is required")
        from query_generation.generator import QwenAWQGenerator  # heavy, GPU-only import

        complete = QwenAWQGenerator(model_path)

    company_by_ticker = load_company_by_ticker(companies_path)
    report = evaluate_records(
        fixture, retrieved_by_id, tables_by_key, complete, company_by_ticker=company_by_ticker,
    )
    report["mode"] = "full_pipeline"
    report["fixture_path"] = str(fixture_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--companies", type=Path, default=DEFAULT_COMPANIES)
    parser.add_argument("--row-label-index", type=Path, default=DEFAULT_ROW_LABEL_INDEX)
    parser.add_argument("--structured-tables", type=Path, default=DEFAULT_STRUCTURED_TABLES)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = run_deterministic(
        fixture_path=args.fixture, catalog_path=args.catalog, companies_path=args.companies,
        row_label_index_path=args.row_label_index, structured_tables_path=args.structured_tables,
        rankings_path=args.rankings, retrieval_top_k=args.top_k,
    )
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    try:
        print(text)
    except UnicodeEncodeError:
        # Some Windows consoles default to a non-UTF-8 codepage that can't render Vietnamese
        # text; the report (already written to --output, if given) is the source of truth.
        print(text.encode("utf-8", errors="replace").decode("ascii", errors="replace"))


if __name__ == "__main__":
    main()

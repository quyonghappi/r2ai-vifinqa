"""Repair actions + the non-degradation accept/reject gate (POST_INFERENCE_REPAIR_DESIGN.md §4/§7).

Every action here is composed from existing, unmodified pipeline functions
(`schema_linking.linker.link_schema`, `pipeline.answer_question`,
`query_generation.generator.{build_evidence_frames,render_pandas_query}`,
`execution.runner.execute_query`) -- no pipeline logic is duplicated except the one deliberately
new, repair-local deterministic unit-ambiguity fallback (`rebuild_direct_lookup_ignoring_target_unit`),
which is a documented, narrower variant of `generate_direct_lookup_plan`'s tail, not a change to it.
"""

from __future__ import annotations

import math
import re
import traceback
from dataclasses import asdict

from execution.runner import ExecutionAttempt, execute_query
from pipeline import answer_question
from query_generation.generator import (
    BoundOperand,
    QueryPlan,
    _entity_variant_structural_candidates,
    build_evidence_frames,
    detect_multi_year_extremum,
    generate_direct_lookup_plan,
    generate_extremum_period_plan,
    render_pandas_query,
)
from repair.common import Diagnosis, RepairOutcome
from repair.diagnose import diagnose_prediction
from retrieval.full_corpus import rank_questions
from schema_linking.linker import link_schema, select_relevant_tables


def _never_call(prompt: str) -> str:
    raise AssertionError(
        "deterministic repair action attempted an LLM call -- this is a bug: "
        "rebuild_deterministic/resolve_unit_deterministic must never reach generate_plan"
    )


def recompute_schema_linking(question_text, retrieved_table_keys, tables_by_key, company_by_ticker=None):
    tables = [tables_by_key[k] for k in retrieved_table_keys if k in tables_by_key]
    return link_schema(question_text, tables, company_by_ticker=company_by_ticker)


def batch_retrieve(questions, catalog_path, companies_path, output_path, *, top_k=10, row_label_index_path=None):
    """One `rank_questions` call for every failure/flagged prediction in a repair batch (never
    per-question) -- rebuilds the BM25 index once for the batch, not once per id, per
    POST_INFERENCE_REPAIR_DESIGN.md §9's "reuse existing model/index objects" instruction."""
    rankings = rank_questions(
        questions, catalog_path, companies_path, output_path, top_k=top_k,
        row_label_index_path=row_label_index_path,
    )
    return {q["id"]: [row[0] for row in rankings[str(q["id"])]] for q in questions}


def _run_answer_question(question_id, question_text, retrieved_table_keys, tables_by_key, complete, company_by_ticker, max_retries):
    try:
        return answer_question(
            {"id": question_id, "question": question_text}, retrieved_table_keys, tables_by_key,
            complete, max_retries=max_retries, company_by_ticker=company_by_ticker,
        )
    except Exception as exc:
        return {
            "id": question_id, "question": question_text, "executed": False,
            "error_type": type(exc).__name__, "error_message": str(exc),
            "attempts": [], "answer": None, "pandas_query": None, "evidence_frames": {},
        }


_UNAMBIGUOUS_UNIT_MESSAGE = "no unambiguous source unit"


def _direct_lookup_ignoring_target_unit(linked, tables_by_key) -> QueryPlan:
    """Deterministic fallback for `generate_direct_lookup_plan`'s "no unambiguous source unit"
    ValueError (real failures/109.json, /149.json, /160.json -- EXECUTION_ACCURACY_AUDIT.md §4.3,
    §9.1). Applies the same entity/period/structural candidate selection
    `generate_direct_lookup_plan` uses (via the shared `_entity_variant_structural_candidates`
    helper), but never attempts a requested-unit conversion -- it reports the winning cell's value
    in whatever unit the source table itself uses, rather than failing outright when that unit
    can't be resolved unambiguously. `answer_unit` is informational only (not one of
    `submission/build_submission.py`'s `REQUIRED_FIELDS`), so this cannot corrupt the submitted
    numeric `answer`; it can only mean the reported magnitude is in the source's own scale instead
    of the requested one -- a real, honest limitation, not a guarantee of correctness (see
    POST_INFERENCE_REPAIR_DESIGN.md's update on this run's dry-run results).
    """
    question = linked.question.lower()
    years = set(re.findall(r"\b20\d{2}\b", question))
    candidates = _entity_variant_structural_candidates(
        linked, tables_by_key,
        empty_error="no numeric direct-lookup operand satisfies entity and structural constraints",
    )
    if years:
        year_candidates = [
            x for x in candidates
            if str(tables_by_key[x.table_key].get("year", "")) in years or any(y in x.column_header for y in years)
        ]
        if year_candidates:
            candidates = year_candidates

    def score(operand):
        table = tables_by_key[operand.table_key]
        return (
            operand.relevance_score,
            str(table.get("year", "")) in years,
            operand.table_key, -operand.row_index, -operand.column_index,
        )

    operand = max(candidates, key=score)
    return QueryPlan(
        [BoundOperand("x", operand.table_key, operand.row_index, operand.column_index)],
        "x", None,
        "deterministic direct lookup: unit-ambiguity fallback, value reported in the source "
        "cell's own unit (target-unit conversion skipped, no unambiguous source scale found)",
    )


def _extremum_ignoring_target_unit(linked, tables_by_key, extremum: dict) -> QueryPlan:
    """The ``mode="value"`` sibling of `_direct_lookup_ignoring_target_unit`, for a failure whose
    root cause was actually `generate_extremum_period_plan`'s identical unit-ambiguity ValueError
    (real failures/961.json -- CHANGE_LOG.md's Phase 2 Checkpoint 3 entry already names this exact,
    separate, pre-existing limitation). Mirrors `generate_extremum_period_plan`'s per-year winner
    selection verbatim (a repair-local duplication, not a change to it -- see this module's
    top-of-file note), then skips the target-unit conversion the same way. ``mode="year"`` never
    reaches here: its expression is a literal year constant that never touches unit conversion at
    all, so it cannot raise this particular ValueError in the first place.
    """
    candidates = _entity_variant_structural_candidates(
        linked, tables_by_key,
        empty_error="no numeric operand satisfies entity and structural constraints for the multi-year extremum",
    )

    def score(operand):
        return (operand.relevance_score, operand.table_key, -operand.row_index, -operand.column_index)

    winners: dict[int, object] = {}
    for year in sorted(extremum["years"]):
        year_candidates = [
            x for x in candidates
            if str(tables_by_key[x.table_key].get("year", "")) == str(year) or str(year) in x.column_header
        ]
        if not year_candidates:
            raise ValueError(f"no operand matches requested year {year} for the multi-year extremum")
        winners[year] = max(year_candidates, key=score)

    pick = max if extremum["direction"] == "max" else min
    best_year = pick(winners, key=lambda year: winners[year].numeric_value)
    best_operand = winners[best_year]
    return QueryPlan(
        [BoundOperand("x", best_operand.table_key, best_operand.row_index, best_operand.column_index)],
        "x", None,
        "deterministic multi-year extremum: unit-ambiguity fallback, value reported in the "
        "source cell's own unit (target-unit conversion skipped)",
    )


def rebuild_direct_lookup_ignoring_target_unit(linked, tables_by_key) -> QueryPlan:
    """Dispatch-aware unit-ambiguity fallback: tries the real deterministic path first
    (`generate_extremum_period_plan` when `detect_multi_year_extremum` matches, else
    `generate_direct_lookup_plan`), and only strips target-unit conversion when that specific
    ValueError is what's blocking it -- any other failure reason is re-raised, never masked.
    Fixes a real bug found during this design's own dry-run: an earlier version of this function
    always used the plain direct-lookup candidate selection, silently mis-repairing a multi-year-
    extremum failure (961) as if it were an ordinary single-year lookup and picking the wrong
    year's cell. See POST_INFERENCE_REPAIR_DESIGN.md's update for the measured evidence.
    """
    extremum = detect_multi_year_extremum(linked.question) if linked.query_family == "direct_lookup" else None
    try:
        if extremum is not None:
            return generate_extremum_period_plan(linked, tables_by_key, extremum)
        return generate_direct_lookup_plan(linked, tables_by_key)
    except ValueError as exc:
        if _UNAMBIGUOUS_UNIT_MESSAGE not in str(exc):
            raise
    if extremum is not None:
        return _extremum_ignoring_target_unit(linked, tables_by_key, extremum)
    return _direct_lookup_ignoring_target_unit(linked, tables_by_key)


def _attempt_unit_fallback(question_id, question_text, retrieved_table_keys, linked, tables_by_key):
    plan = None
    frames: dict = {}
    try:
        plan = rebuild_direct_lookup_ignoring_target_unit(linked, tables_by_key)
        frames, variables = build_evidence_frames(plan, tables_by_key)
        query = render_pandas_query(plan, variables, {})
        answer = execute_query(query, frames)
        attempts = [asdict(ExecutionAttempt(0, query, None, None, "execution"))]
        executed, pandas_query = True, query
    except Exception as exc:
        attempts = [asdict(ExecutionAttempt(
            0, None, type(exc).__name__, str(exc), "generation_or_validation",
            traceback=traceback.format_exc(),
        ))]
        executed, pandas_query, answer = False, None, None

    used_keys = list(dict.fromkeys(x.table_key for x in plan.operands)) if plan else []
    relevant_tables = select_relevant_tables(linked.operands)
    return {
        "id": question_id, "question": question_text, "query_family": linked.query_family,
        "retrieved_tables": retrieved_table_keys,
        "schema_linked_operands": [asdict(x) for x in linked.operands],
        "used_tables": used_keys, "relevant_tables": relevant_tables,
        "relevant_docs": list(dict.fromkeys(k.rsplit("|", 1)[0] for k in relevant_tables)),
        "answer": answer, "pandas_query": pandas_query, "executed": executed,
        "attempts": attempts, "answer_unit": plan.answer_unit if plan else None,
        "model_rationale": plan.rationale if plan else None,
        "evidence_frames": frames,
    }


def repair_one(
    question_id: int,
    question_text: str,
    retrieved_table_keys: list[str],
    tables_by_key: dict,
    diagnosis: Diagnosis,
    complete,
    company_by_ticker: dict | None = None,
) -> dict:
    """Dispatch by ``diagnosis.action``. Returns
    ``{"prediction": <candidate dict or None>, "requires_llm": bool, "skipped": bool}``.
    ``requires_llm=True`` with ``prediction=None`` means: this id needs a real ``complete``
    (LLM) callable that wasn't supplied -- the caller (repair_failures.py/repair_predictions.py)
    is expected to collect these and either pass a real model or report them as pending.
    """
    if diagnosis.action == "keep":
        return {"prediction": None, "requires_llm": False, "skipped": True}

    if diagnosis.action == "mark_unsupported":
        return {"prediction": None, "requires_llm": False, "skipped": True}

    if diagnosis.action == "resolve_unit_deterministic":
        linked = recompute_schema_linking(question_text, retrieved_table_keys, tables_by_key, company_by_ticker)
        pred = _attempt_unit_fallback(question_id, question_text, retrieved_table_keys, linked, tables_by_key)
        return {"prediction": pred, "requires_llm": False, "skipped": False}

    if diagnosis.action == "rebuild_deterministic":
        pred = _run_answer_question(
            question_id, question_text, retrieved_table_keys, tables_by_key, _never_call,
            company_by_ticker, max_retries=0,
        )
        return {"prediction": pred, "requires_llm": False, "skipped": False}

    if diagnosis.action in ("regenerate", "relink_then_regenerate"):
        if complete is None:
            return {"prediction": None, "requires_llm": True, "skipped": False}
        pred = _run_answer_question(
            question_id, question_text, retrieved_table_keys, tables_by_key, complete,
            company_by_ticker, max_retries=diagnosis.retry_budget,
        )
        return {"prediction": pred, "requires_llm": True, "skipped": False}

    raise ValueError(f"unknown repair action: {diagnosis.action!r}")


def evaluate_candidate(
    question_id: int,
    original: dict | None,
    candidate: dict | None,
    *,
    repair_code_revision: str | None = None,
) -> RepairOutcome:
    """The non-degradation gate (POST_INFERENCE_REPAIR_DESIGN.md §4's safeguard / §7.4 step 4).

    ``original=None`` means this is a hard-failure recovery (nothing to compare a "did it get
    strictly better" claim against) -- accept only a fully clean candidate (zero remaining
    triggers), the strictest bar, since recovering a failure into a new wrong answer is a real
    regression risk the design explicitly flags (never gate a failure-recovery on "better than a
    non-existent original"). ``original`` present means an already-`executed=True` prediction is
    being considered for replacement -- accept only if the candidate's own trigger count is
    strictly lower than the original's, and only if the rendered query actually changed.
    """
    attempts_spent = len((candidate or {}).get("attempts") or [])
    if candidate is None or candidate.get("executed") is not True:
        return RepairOutcome(question_id, False, "candidate did not execute", -1, -1, attempts_spent)

    answer = candidate.get("answer")
    if isinstance(answer, bool) or not isinstance(answer, (int, float)) or not math.isfinite(answer):
        return RepairOutcome(question_id, False, "candidate answer is not finite numeric", -1, -1, attempts_spent)

    if original is not None and candidate.get("pandas_query") == original.get("pandas_query"):
        old_n = len(diagnose_prediction(original).triggers)
        return RepairOutcome(question_id, False, "no-op: identical pandas_query", old_n, old_n, attempts_spent)

    new_triggers = diagnose_prediction(candidate).triggers
    if original is None:
        if new_triggers:
            return RepairOutcome(
                question_id, False, "recovered candidate still has open trigger(s)",
                0, len(new_triggers), attempts_spent,
            )
        return RepairOutcome(
            question_id, True, "hard failure recovered: executed, finite, no triggers",
            0, 0, attempts_spent, new_prediction=candidate, repair_code_revision=repair_code_revision,
        )

    old_triggers = diagnose_prediction(original).triggers
    if len(new_triggers) >= len(old_triggers):
        return RepairOutcome(
            question_id, False, "candidate did not strictly reduce the trigger count",
            len(old_triggers), len(new_triggers), attempts_spent,
        )
    return RepairOutcome(
        question_id, True, "accepted: fewer triggers than the original, query changed, executes to a finite answer",
        len(old_triggers), len(new_triggers), attempts_spent,
        new_prediction=candidate, repair_code_revision=repair_code_revision,
    )

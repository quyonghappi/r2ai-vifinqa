"""Qwen-backed plan generation with strict grounding before Pandas rendering."""

from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from normalization.schema import infer_column_metadata, unit_scale_to_vnd
from retrieval.full_corpus import infer_question_variant, infer_question_years
from retrieval.sparse import DEFAULT_STOPWORDS, tokenize
from schema_linking.linker import (
    SchemaLinkResult,
    is_numeric_data_column,
    parse_financial_number,
    variant_conflicts_with_question,
    year_conflicts_with_question,
)

MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"
MODEL_REVISION = "b56cc04415fac88c421533036e44149a5983dd2a"
MODEL_ELIGIBILITY = {
    "category": "pipeline component", "parameters": "7.61B", "license": "Apache-2.0",
    "released": "2024-09-19", "competition_cutoff": "2026-06-01",
    "pinned_revision": MODEL_REVISION,
}
MAX_PROMPT_OPERANDS = 32
MAX_NEW_TOKENS = 448
PROMPT_SAFETY_MARGIN = 256
DEFAULT_MAX_INPUT_TOKENS = 4096
class PromptBudgetError(RuntimeError):
    """The compact, grounded prompt cannot fit the model context window."""


@dataclass(frozen=True)
class BoundOperand:
    alias: str
    table_key: str
    row_index: int
    column_index: int


@dataclass(frozen=True)
class OperandGroup:
    """A named, aligned list of operands (typically one per company/period) that a Reduction
    can fold into a single scalar. ``metric_label`` is model-supplied but must be grounded: it
    is required (validate_plan) to share vocabulary with every member operand's row label, so it
    cannot name a concept unconnected to the actual evidence."""
    name: str
    metric_label: str
    operands: list[BoundOperand]


REDUCTION_OPS = frozenset({"sum", "avg", "count_gt", "count_lt", "count_ge", "count_le", "max", "min"})
_THRESHOLD_REDUCTION_OPS = frozenset({"count_gt", "count_lt", "count_ge", "count_le"})

# Families whose classifier trigger words (schema_linking.linker.classify_query_family) are
# inherently about relating 2+ values -- a delta between two periods, a sum/average/count over a
# named list, a ratio of two quantities. `direct_lookup` is deliberately excluded: it never
# reaches validate_plan at all (execute_with_repair routes it to generate_direct_lookup_plan).
# See EXECUTION_ACCURACY_AUDIT.md §4.2/§4.3 for the confirmed real-run failures this targets:
# every one of 591/604/795/938/1006's traced wrong answers was a single-operand or missing-company
# plan that validated and executed without ever computing the actual required comparison/sum.
_MULTI_OPERAND_FAMILIES = frozenset(
    {"comparison", "growth", "aggregation", "ratio_or_derived", "cross_company_comparison"}
)


@dataclass(frozen=True)
class Reduction:
    """A deterministic fold over one OperandGroup's numeric values, computed in trusted Python
    (never through the sandboxed expression evaluator) and exposed to ``expression`` as an
    ordinary named scalar alias -- see ``compute_reductions``/``render_pandas_query``."""
    alias: str
    op: str
    group: str
    threshold: float | None = None


@dataclass
class QueryPlan:
    operands: list[BoundOperand]
    expression: str
    answer_unit: str | None
    rationale: str
    groups: list[OperandGroup] = field(default_factory=list)
    reductions: list[Reduction] = field(default_factory=list)


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("model output contains no JSON object")
    return json.loads(match.group(0))


def _validate_expression(expression: str, aliases: set[str]) -> None:
    tree = ast.parse(expression, mode="eval")
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Constant, ast.Add,
               ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Call, ast.Load)
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ValueError(f"disallowed expression syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in aliases | {"abs"}:
            raise ValueError(f"unknown operand in expression: {node.id}")
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id == "abs"):
            raise ValueError("only abs(...) is allowed in expressions")


def _group_completeness_errors(
    group: OperandGroup,
    tables_by_key: dict[str, dict],
    operand_by_coords: dict,
    required_tickers: set[str],
    question_years: set[int],
    question_variant: str | None,
) -> list[str]:
    """Validate a group's (company, metric, period, variant) slots against the question.

    Company coverage and metric-label grounding are checked here (they need every member
    operand's row_label/ticker, only available via ``operand_by_coords``/``tables_by_key``).
    Period/variant conflicts reuse the exact conflict-detection functions
    schema_linking.link_schema uses for ranking (year_conflicts_with_question/
    variant_conflicts_with_question), so "does this operand belong in this group" is answered
    identically wherever it's asked in the pipeline.
    """
    errors: list[str] = []
    covered_tickers: set[str] = set()
    label_token_sets = []
    for operand in group.operands:
        candidate = operand_by_coords.get((operand.table_key, operand.row_index, operand.column_index))
        if candidate is None:
            continue  # already reported as "invented or unresolved operand" by the caller
        table = tables_by_key[operand.table_key]
        ticker = table.get("ticker")
        if ticker:
            covered_tickers.add(str(ticker))
        if year_conflicts_with_question(table, candidate.column_header, question_years):
            errors.append(
                f"group {group.name!r}: operand {operand.alias!r} conflicts with the requested period"
            )
        if variant_conflicts_with_question(table, question_variant):
            errors.append(
                f"group {group.name!r}: operand {operand.alias!r} conflicts with the requested report variant"
            )
        label_token_sets.append(tokenize(candidate.row_label, stopwords=DEFAULT_STOPWORDS))
    missing = required_tickers - covered_tickers
    if missing:
        errors.append(f"group {group.name!r} is missing required companies: {sorted(missing)}")
    label_tokens = set(tokenize(group.metric_label, stopwords=DEFAULT_STOPWORDS))
    if label_tokens:
        for operand, tokens in zip(group.operands, label_token_sets):
            if tokens and not (label_tokens & set(tokens)):
                errors.append(
                    f"group {group.name!r}: operand {operand.alias!r}'s row label does not match "
                    f"the declared metric {group.metric_label!r}"
                )
    return errors


def validate_plan(payload: dict, linked: SchemaLinkResult, tables_by_key: dict[str, dict]) -> QueryPlan:
    allowed = {(x.table_key, x.row_index, x.column_index) for x in linked.operands}
    operand_by_coords = {(x.table_key, x.row_index, x.column_index): x for x in linked.operands}

    operands = [BoundOperand(**x) for x in payload.get("operands", [])]
    raw_groups = payload.get("groups") or []
    groups = [
        OperandGroup(
            name=g["name"], metric_label=g.get("metric_label", ""),
            operands=[BoundOperand(**x) for x in g.get("operands", [])],
        )
        for g in raw_groups
    ]
    raw_reductions = payload.get("reductions") or []
    reductions = [
        Reduction(alias=r["alias"], op=r["op"], group=r["group"], threshold=r.get("threshold"))
        for r in raw_reductions
    ]

    all_bound = operands + [operand for group in groups for operand in group.operands]
    aliases = {x.alias for x in all_bound} | {r.alias for r in reductions}
    if not all_bound and not reductions:
        raise ValueError("plan requires at least one operand")
    if len(aliases) != len(all_bound) + len(reductions):
        raise ValueError("operands/groups/reductions require unique Python identifier aliases")
    if not all(x.alias.isidentifier() for x in all_bound) or not all(r.alias.isidentifier() for r in reductions):
        raise ValueError("operands/groups/reductions require unique Python identifier aliases")
    for operand in all_bound:
        if (operand.table_key, operand.row_index, operand.column_index) not in allowed:
            raise ValueError(f"invented or unresolved operand: {operand}")

    group_names = {g.name for g in groups}
    if len(group_names) != len(groups):
        raise ValueError("group names must be unique")
    if any(not group.operands for group in groups):
        raise ValueError("every group requires at least one operand")
    for reduction in reductions:
        if reduction.op not in REDUCTION_OPS:
            raise ValueError(f"unknown reduction op: {reduction.op!r}")
        if reduction.group not in group_names:
            raise ValueError(f"reduction {reduction.alias!r} references unknown group {reduction.group!r}")
        if reduction.op in _THRESHOLD_REDUCTION_OPS and reduction.threshold is None:
            raise ValueError(f"reduction op {reduction.op!r} requires a threshold")

    # required_tickers is deliberately conservative -- only tickers link_schema's canonical
    # `resolved_tickers` resolution names AND that retrieval actually retained here (same
    # `linked.resolved_tickers` generate_direct_lookup_plan reads; see
    # SchemaLinkResult.resolved_tickers's docstring and EXECUTION_ACCURACY_AUDIT.md §6.1). A
    # question naming companies only by full legal name with no literal-ticker/phrase match is not
    # checked for company coverage here; this catches the common explicit-company-list shape
    # (e.g. "AAA, NKG, DCM và DPM"), not every phrasing. Computed unconditionally (not only when
    # `groups` is present) so the plan-wide coverage check below also applies to flat operands.
    available_tickers = {
        str(tables_by_key[key].get("ticker", ""))
        for key in linked.table_keys if tables_by_key[key].get("ticker")
    }
    required_tickers = set(linked.resolved_tickers) & available_tickers
    question_years = infer_question_years(linked.question)
    question_variant = infer_question_variant(linked.question)

    if groups:
        errors = [
            error
            for group in groups
            for error in _group_completeness_errors(
                group, tables_by_key, operand_by_coords, required_tickers, question_years, question_variant,
            )
        ]
        if errors:
            raise ValueError("incomplete operand group(s): " + "; ".join(errors))

    # Plan-wide company coverage (EXECUTION_ACCURACY_AUDIT.md §4.2, confirmed real failures): the
    # per-group check above only ever ran when `groups` was non-empty, so a flat, ungrouped
    # operand list -- exactly the shape real graded-run plans for id 795 ("GAS cao hơn POW bao
    # nhiêu", 1 operand, GAS only) and id 938 ("tổng ... EIB, MSB, STB", 2 operands, missing STB
    # entirely) actually used -- got no company-completeness check at all and validated/executed
    # successfully while silently dropping a required company. Checked across every bound operand
    # (flat operands plus every group's), not only within one group, so splitting required
    # companies across a flat operand and a group (or across several groups) still counts.
    covered_tickers = {
        str(tables_by_key[operand.table_key].get("ticker", ""))
        for operand in all_bound
        if tables_by_key[operand.table_key].get("ticker")
    }
    missing_companies = required_tickers - covered_tickers
    if missing_companies:
        raise ValueError(f"plan is missing required companies: {sorted(missing_companies)}")

    # Minimum operand count for families that cannot be correctly answered from a single cell
    # (EXECUTION_ACCURACY_AUDIT.md §4.2/§4.3): a difference/sum/average/ratio needs at least 2
    # grounded values by definition, whether they come from one table's two columns (a legitimate,
    # common shape -- e.g. a balance-sheet row carrying both period-end columns) or from several
    # tables. Every one of 591/604/795/1006's traced real wrong answers was exactly a single
    # declared operand for one of these families; this closes that silently-accepted shape at
    # validation time instead of catching it (or not) only via the ticker-coverage check above.
    if linked.query_family in _MULTI_OPERAND_FAMILIES and len(all_bound) < 2:
        raise ValueError(
            f"{linked.query_family!r} questions require at least 2 grounded operands to compute a "
            f"comparison, sum, average, or ratio; plan declares only {len(all_bound)}"
        )

    expression = payload.get("expression", "")
    _validate_expression(expression, aliases)
    return QueryPlan(operands, expression, payload.get("answer_unit"), payload.get("rationale", ""), groups, reductions)


def _source_scale_to_vnd(table: dict, column_index: int, column_header: str) -> int | None:
    """Resolve a source unit from the selected column before trusting persisted metadata.

    Structured artifacts are versioned, but a resumable runtime can still carry column metadata
    produced by an older parser. Re-derive the selected column's header path from the source grid
    first; this keeps a stale scale from reversing or magnifying an otherwise grounded answer.
    """
    grid = table.get("grid") or []
    inferred = infer_column_metadata(grid)
    if column_index < len(inferred) and inferred[column_index].get("scale_to_vnd") is not None:
        return inferred[column_index]["scale_to_vnd"]
    column = next(
        (
            item for item in (table.get("column_metadata") or [])
            if item.get("column_index") == column_index
        ),
        {},
    )
    if column.get("scale_to_vnd") is not None:
        return column["scale_to_vnd"]
    scale = unit_scale_to_vnd(column_header)
    if scale is not None:
        return scale
    units = table.get("detected_units") or []
    # A table-wide hint is safe only when it is unambiguous.
    scales = {unit_scale_to_vnd(unit) for unit in units if unit_scale_to_vnd(unit) is not None}
    return scales.pop() if len(scales) == 1 else None


def _entity_variant_structural_candidates(
    linked: SchemaLinkResult, tables_by_key: dict[str, dict], *, empty_error: str,
) -> list:
    """Shared prefix of every deterministic plan's filter chain: entity/company scope, report
    variant, and structural/numeric-column filters. Used by both ``generate_direct_lookup_plan``
    and ``generate_extremum_period_plan`` so the two deterministic paths never drift on "is this
    operand even a legitimate candidate for this company/variant/column" (year and period
    filtering stay separate, since the two callers apply them differently: one per fixed year, the
    other per each of several explicitly named years).

    Canonical entity scope: ``SchemaLinkResult.resolved_tickers`` already resolves nested-brand
    mentions (e.g. "CTCP Chứng khoán FPT" -> FTS) the same way retrieval does, instead of
    re-deriving it from a bare literal-text scan here (EXECUTION_ACCURACY_AUDIT.md §6.1: the prior
    re-derivation silently disabled this filter whenever the literal token and the actually-
    relevant ticker disagreed).
    """
    question = linked.question.lower()
    candidates = list(linked.operands)
    available_tickers = {
        str(tables_by_key[table_key].get("ticker", ""))
        for table_key in linked.table_keys
        if tables_by_key[table_key].get("ticker")
    }
    mentioned_tickers = set(linked.resolved_tickers) & available_tickers
    if mentioned_tickers:
        candidates = [
            x for x in candidates
            if tables_by_key[x.table_key].get("ticker") in mentioned_tickers
        ]
        if not candidates:
            raise ValueError("schema linker retained no operands for the named ticker")
    if "công ty mẹ" in question or "công ty riêng" in question:
        candidates = [x for x in candidates if tables_by_key[x.table_key].get("variant") == "separate"]
    elif "hợp nhất" in question or "tập đoàn" in question:
        candidates = [x for x in candidates if tables_by_key[x.table_key].get("variant") == "consolidated"]

    candidates = [
        x for x in candidates
        if x.row_index > 0 and x.column_index > 0 and x.column_header.strip()
        and is_numeric_data_column(tables_by_key[x.table_key], x.column_index, x.column_header)
        and not any(token in x.column_header.lower() for token in ("mã số", "thuyết minh", "note", "biến động", "%"))
    ]
    if not candidates:
        raise ValueError(empty_error)
    return candidates


def generate_direct_lookup_plan(linked: SchemaLinkResult, tables_by_key: dict[str, dict]) -> QueryPlan:
    """Resolve a single grounded value without asking the model to choose evidence.

    Entity-scoped retrieval supplies the company candidate set.  This function applies the
    remaining report-variant, period, structural-cell, and per-column-unit constraints before
    rendering the simple arithmetic expression.
    """
    question = linked.question.lower()
    years = set(re.findall(r"\b20\d{2}\b", question))
    candidates = _entity_variant_structural_candidates(
        linked, tables_by_key, empty_error="no numeric direct-lookup operand satisfies entity and structural constraints",
    )

    # An explicit year is evidence, not a soft preference.  A comparative column in a
    # neighbouring report is still valid if its header names the requested year.
    if years:
        year_candidates = [
            x for x in candidates
            if str(tables_by_key[x.table_key].get("year", "")) in years
            or any(year in x.column_header for year in years)
        ]
        if not year_candidates:
            raise ValueError("no direct-lookup operand matches the requested year")
        candidates = year_candidates

    # Only filter when the table representation distinguishes opening from closing
    # balances.  Otherwise fail neither silently nor by forcing a guessed convention.
    requested_period = "cuối năm" if "cuối năm" in question else "đầu năm" if "đầu năm" in question else None
    if requested_period:
        period_candidates = [
            x for x in candidates
            if requested_period in f"{x.row_label} {x.column_header}".lower()
        ]
        opposite_period = "đầu năm" if requested_period == "cuối năm" else "cuối năm"
        has_explicit_period = any(
            requested_period in f"{x.row_label} {x.column_header}".lower()
            or opposite_period in f"{x.row_label} {x.column_header}".lower()
            for x in candidates
        )
        if has_explicit_period and not period_candidates:
            raise ValueError(f"no direct-lookup operand matches {requested_period}")
        if period_candidates:
            candidates = period_candidates

    def score(operand) -> tuple:
        table = tables_by_key[operand.table_key]
        header, row = operand.column_header.lower(), operand.row_label.lower()
        year_score = 20 if str(table.get("year", "")) in years else 0
        current_score = 10 if str(table.get("year", "")) in years and "năm nay" in header else 0
        period_score = 0  # period is filtered above when the source exposes it
        identity = " ".join(map(str, (table.get("table_identity", ""), table.get("caption_context", "")))).lower()
        narrative_penalty = 25 if any(token in identity for token in ("giải trình", "thuyết minh", "explan")) else 0
        return (operand.relevance_score * 100 + year_score + current_score + period_score - narrative_penalty,
                operand.table_key, -operand.row_index, -operand.column_index)

    operand = max(candidates, key=score)
    table = tables_by_key[operand.table_key]
    source_scale = _source_scale_to_vnd(table, operand.column_index, operand.column_header)
    target_scale = unit_scale_to_vnd(linked.requested_unit)
    if source_scale is None and target_scale is not None:
        raise ValueError("selected direct-lookup column has no unambiguous source unit")
    source_scale = source_scale or 1
    target_scale = target_scale or source_scale
    factor = source_scale / target_scale
    expression = "x" if factor == 1 else f"x * {factor!r}"
    return QueryPlan(
        [BoundOperand("x", operand.table_key, operand.row_index, operand.column_index)],
        expression,
        linked.requested_unit,
        "deterministic direct lookup with entity, period, and column-unit constraints",
    )


# Superlative keywords distinct from classify_query_family's comparative triggers ("cao hơn",
# "thấp hơn", "so với" -- a 2-thing comparison, already routed to `comparison`). These name an
# extremum across an explicitly named *list* of years/periods for one entity.
_SUPERLATIVE_MAX_PHRASES = ("cao nhất", "lớn nhất")
_SUPERLATIVE_MIN_PHRASES = ("thấp nhất", "nhỏ nhất")
_YEAR_INTERROGATIVE_PHRASES = ("năm nào", "kỳ nào", "thời điểm nào")


def detect_multi_year_extremum(question: str) -> dict | None:
    """Detect a "which of these N named years is the metric highest/lowest" (answer = the year)
    or "what is the highest/lowest value across these N named years" (answer = the value) shape.

    Confirmed gap (EXECUTION_ACCURACY_AUDIT.md §6.4, real fixture ids 959/961):
    ``classify_query_family`` has no trigger for superlative phrasing ("cao nhất"/"lớn nhất"/
    "thấp nhất"/"nhỏ nhất") naming several explicit years, so both fall through to its
    ``direct_lookup`` default -- but ``generate_direct_lookup_plan`` can only ever return one raw
    cell's value, never "which year" nor a reliable cross-year comparison. Even the 2026-09-04
    ``groups``/``reductions`` DSL cannot answer the "which year" shape either: its ``max``/``min``
    ops fold to the extreme *value*, never the group-key (year) that achieved it -- confirmed by
    its own design note (``submission_guide.md`` requires a float ``answer``, so ``argmax``/
    ``argmin`` were deliberately not added as reduction ops).

    Deliberately narrow: fires only with an explicit superlative phrase *and* 2+ explicitly named
    years, so it never touches an ordinary single-year lookup or a 2-thing comparison (those
    already route correctly via ``direct_lookup``/``comparison``). This does not change
    ``classify_query_family``'s own output -- it is an additional, orthogonal dispatch check
    ``execution.runner.execute_with_repair`` applies only when that classifier already said
    ``direct_lookup``, so every existing family label and Phase-1 diagnostic keyed on it is
    unaffected.
    """
    q = question.lower()
    if any(phrase in q for phrase in _SUPERLATIVE_MAX_PHRASES):
        direction = "max"
    elif any(phrase in q for phrase in _SUPERLATIVE_MIN_PHRASES):
        direction = "min"
    else:
        return None
    years = infer_question_years(question)
    if len(years) < 2:
        return None
    mode = "year" if any(phrase in q for phrase in _YEAR_INTERROGATIVE_PHRASES) else "value"
    return {"years": years, "direction": direction, "mode": mode}


def generate_extremum_period_plan(
    linked: SchemaLinkResult, tables_by_key: dict[str, dict], extremum: dict,
) -> QueryPlan:
    """Deterministic multi-year extremum, dispatched only when ``detect_multi_year_extremum``
    matches (see its docstring for the confirmed gap this closes).

    For each explicitly named year, picks the best-scoring grounded operand under the same
    entity/variant/structural constraints ``generate_direct_lookup_plan`` applies (shared via
    ``_entity_variant_structural_candidates``), then takes the max/min numeric value across years.
    In ``mode="value"``, the winning cell is returned like an ordinary direct lookup (with the
    same source-to-requested-unit scaling). In ``mode="year"``, the *year* achieving the extremum
    is the answer -- computed in trusted Python from real, grounded per-year winners (never
    invented), then substituted as a literal float constant, exactly the mechanism
    ``render_pandas_query`` already uses for a computed reduction value. The winning operand is
    still attached as the plan's evidence (so the submission's required evidence CSV reflects the
    real table that determined the answer), even though ``expression`` does not read its cell.
    """
    candidates = _entity_variant_structural_candidates(
        linked, tables_by_key,
        empty_error="no numeric operand satisfies entity and structural constraints for the multi-year extremum",
    )

    def score(operand) -> tuple:
        return (operand.relevance_score, operand.table_key, -operand.row_index, -operand.column_index)

    winners: dict[int, object] = {}
    for year in sorted(extremum["years"]):
        year_candidates = [
            x for x in candidates
            if str(tables_by_key[x.table_key].get("year", "")) == str(year)
            or str(year) in x.column_header
        ]
        if not year_candidates:
            raise ValueError(f"no operand matches requested year {year} for the multi-year extremum")
        winners[year] = max(year_candidates, key=score)

    pick = max if extremum["direction"] == "max" else min
    best_year = pick(winners, key=lambda year: winners[year].numeric_value)
    best_operand = winners[best_year]
    bound_operand = BoundOperand("x", best_operand.table_key, best_operand.row_index, best_operand.column_index)

    if extremum["mode"] == "year":
        return QueryPlan(
            [bound_operand], repr(float(best_year)), None,
            "deterministic multi-year extremum: answer is the year achieving it",
        )

    table = tables_by_key[best_operand.table_key]
    source_scale = _source_scale_to_vnd(table, best_operand.column_index, best_operand.column_header)
    target_scale = unit_scale_to_vnd(linked.requested_unit)
    if source_scale is None and target_scale is not None:
        raise ValueError("selected multi-year-extremum operand has no unambiguous source unit")
    source_scale = source_scale or 1
    target_scale = target_scale or source_scale
    factor = source_scale / target_scale
    expression = "x" if factor == 1 else f"x * {factor!r}"
    return QueryPlan(
        [bound_operand], expression, linked.requested_unit,
        "deterministic multi-year extremum: answer is the extreme value",
    )


def _plan_table_keys(plan: QueryPlan) -> list[str]:
    """Every table a plan's evidence frames must cover: flat operands plus every group's."""
    return list(dict.fromkeys(
        [x.table_key for x in plan.operands]
        + [operand.table_key for group in plan.groups for operand in group.operands]
    ))


def build_evidence_frames(plan: QueryPlan, tables_by_key: dict[str, dict]) -> tuple[dict, dict]:
    frames, variables = {}, {}
    for index, key in enumerate(_plan_table_keys(plan), 1):
        variable = f"df{index}"
        table = tables_by_key[key]
        rows = []
        for ri, row in enumerate(table["grid"]):
            for ci, text in enumerate(row):
                rows.append({"row_index": ri, "column_index": ci, "text": text,
                             "numeric_value": parse_financial_number(text)})
        frames[variable] = pd.DataFrame(rows)
        variables[key] = variable
    return frames, variables


def _cell_numeric_value(table: dict, row_index: int, column_index: int) -> float:
    raw = table["grid"][row_index][column_index]
    value = parse_financial_number(raw)
    if value is None:
        raise ValueError(
            f"reduction operand cell is not numeric: {table.get('table_key')} "
            f"r{row_index}c{column_index}"
        )
    return value


_REDUCTION_FNS: dict[str, Callable[[list[float], float | None], float]] = {
    "sum": lambda values, _threshold: sum(values),
    "avg": lambda values, _threshold: sum(values) / len(values) if values else 0.0,
    "max": lambda values, _threshold: max(values),
    "min": lambda values, _threshold: min(values),
    "count_gt": lambda values, threshold: float(sum(1 for v in values if v > threshold)),
    "count_lt": lambda values, threshold: float(sum(1 for v in values if v < threshold)),
    "count_ge": lambda values, threshold: float(sum(1 for v in values if v >= threshold)),
    "count_le": lambda values, threshold: float(sum(1 for v in values if v <= threshold)),
}


def compute_reductions(plan: QueryPlan, tables_by_key: dict[str, dict]) -> dict[str, float]:
    """Fold each group's numeric values into its reduction's scalar, in trusted Python -- never
    through the sandboxed expression evaluator. ``validate_plan`` has already confirmed every
    reduction's ``op`` is in ``REDUCTION_OPS`` and its ``group`` exists, so lookups here cannot
    raise on a malformed plan; they can only raise (deliberately) on a non-numeric grounded cell.
    """
    groups_by_name = {group.name: group for group in plan.groups}
    results = {}
    for reduction in plan.reductions:
        group = groups_by_name[reduction.group]
        values = [
            _cell_numeric_value(tables_by_key[operand.table_key], operand.row_index, operand.column_index)
            for operand in group.operands
        ]
        results[reduction.alias] = _REDUCTION_FNS[reduction.op](values, reduction.threshold)
    return results


def render_pandas_query(
    plan: QueryPlan, variables: dict[str, str], reduction_values: dict[str, float] | None = None,
) -> str:
    """Substitute every operand alias with a grounded cell lookup and every reduction alias with
    its already-computed literal float (see ``compute_reductions``) into ``plan.expression``.
    The expression grammar itself (``_validate_expression``) is unchanged by reductions -- a
    reduction alias is, by the time it reaches here, just another named float constant."""
    rendered = plan.expression
    substitutions = [
        (
            operand.alias,
            f"float({variables[operand.table_key]}.loc["
            f"({variables[operand.table_key]}['row_index']=={operand.row_index}) & "
            f"({variables[operand.table_key]}['column_index']=={operand.column_index}), "
            "'numeric_value'].iloc[0])",
        )
        for operand in plan.operands
    ] + [(alias, repr(float(value))) for alias, value in (reduction_values or {}).items()]
    for alias, replacement in sorted(substitutions, key=lambda item: -len(item[0])):
        rendered = re.sub(rf"\b{re.escape(alias)}\b", replacement, rendered)
    return rendered


def build_prompt(linked: SchemaLinkResult, feedback: str | None = None) -> str:
    contract = {
        "operands": [{"alias": "x", "table_key": "allowed key", "row_index": 0, "column_index": 1}],
        "groups": [{
            "name": "short group name, e.g. provision_pct",
            "metric_label": "short label matching the group's operands' row meaning",
            "operands": [{"alias": "g1_a", "table_key": "allowed key", "row_index": 0, "column_index": 1}],
        }],
        "reductions": [{
            "alias": "n_high", "op": "sum|avg|count_gt|count_lt|count_ge|count_le|max|min",
            "group": "a groups[].name above", "threshold": "required only for count_* ops, else omit",
        }],
        "expression": "x", "answer_unit": "requested unit or null", "rationale": "short",
    }
    example = {
        "operands": [], "groups": [{
            "name": "provision", "metric_label": "chi phí dự phòng rủi ro",
            "operands": [
                {"alias": "g1_a", "table_key": "AAA_report|10", "row_index": 3, "column_index": 1},
                {"alias": "g1_b", "table_key": "BBB_report|10", "row_index": 5, "column_index": 1},
            ],
        }],
        "reductions": [{"alias": "n_high", "op": "count_gt", "group": "provision", "threshold": 100000000000}],
        "expression": "n_high", "answer_unit": None, "rationale": "count companies over the threshold",
    }
    return (
        "You generate a grounded numeric query plan for Vietnamese financial tables. "
        "Use ONLY allowed operands. Never invent tables, coordinates, columns, rows, or values. "
        "For a single value: use only 'operands' and reference its alias in 'expression'. "
        "For a question naming several companies/periods and asking to count, average, sum, or "
        "compare across them: put one operand per company/period in a named group under "
        "'groups' (every named company must have an operand in the group; metric_label must "
        "describe what the group's row labels actually mean), define a 'reductions' entry that "
        "folds that group (sum/avg/count_gt/count_lt/count_ge/count_le/max/min; count_* ops "
        "require a numeric 'threshold'), then reference the reduction's alias in 'expression' "
        "instead of naming every company individually. 'groups'/'reductions' may be empty for a "
        "simple single-value question. Expression may use operand aliases, reduction aliases, "
        "numeric constants, + - * / ** and abs only. Return JSON only.\n"
        f"OUTPUT_SCHEMA={json.dumps(contract, ensure_ascii=False)}\n"
        f"EXAMPLE_FOR_A_COUNT_ACROSS_COMPANIES_QUESTION={json.dumps(example, ensure_ascii=False)}\n"
        f"CONTEXT={json.dumps(linked.prompt_payload(), ensure_ascii=False)}\n"
        f"REPAIR_FEEDBACK={feedback or 'none'}"
    )


class QwenAWQGenerator:
    """Lazy offline adapter; model_path must point to an attached Kaggle Dataset."""
    def __init__(self, model_path: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", local_files_only=True, low_cpu_mem_usage=True
        ).eval()

    def __call__(self, prompt: str) -> str:
        import torch
        prompt, inputs = self._fit_prompt(prompt)
        context_values = (
            getattr(self.model.config, "max_position_embeddings", 0),
            getattr(self.tokenizer, "model_max_length", 0),
        )
        valid_contexts = [
            int(value) for value in context_values
            if isinstance(value, int) and 0 < value < 1_000_000
        ]
        if not valid_contexts:
            raise PromptBudgetError("model exposes no finite context window")
        context_limit = min(valid_contexts)
        # Transformers 5 returns a BatchEncoding here, not a bare tensor.  Reading
        # ``.shape`` on it raises an empty AttributeError and hides the real cause.
        input_ids = inputs["input_ids"]
        input_tokens = int(input_ids.shape[-1])
        self.last_prompt_diagnostics = {
            "context_limit": context_limit,
            "input_tokens": input_tokens,
            "max_new_tokens": MAX_NEW_TOKENS,
            "prompt_operands": prompt.count('"table_key"'),
            "input_budget": self._input_token_budget(context_limit),
        }
        if input_tokens > self._input_token_budget(context_limit):
            raise PromptBudgetError(
                f"prompt has {input_tokens} tokens; input budget is {self._input_token_budget(context_limit)}"
            )
        # The embedding layer, unlike model.device, stays correct with device_map="auto".
        inputs = inputs.to(self.model.get_input_embeddings().weight.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, max_new_tokens=MAX_NEW_TOKENS, use_cache=True)
        return self.tokenizer.decode(output[0][input_tokens:], skip_special_tokens=True)

    def _input_token_budget(self, context_limit: int) -> int:
        configured = int(os.environ.get("VIFINQA_MAX_INPUT_TOKENS", DEFAULT_MAX_INPUT_TOKENS))
        if configured <= 0:
            raise PromptBudgetError("VIFINQA_MAX_INPUT_TOKENS must be positive")
        return min(configured, context_limit - MAX_NEW_TOKENS - PROMPT_SAFETY_MARGIN)

    def _tokenize_prompt(self, prompt: str):
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True,
            tokenize=True, return_tensors="pt",
        )

    def _fit_prompt(self, prompt: str):
        """Trim grounded operands to a tokenizer- and VRAM-safe input budget."""
        context_values = (getattr(self.model.config, "max_position_embeddings", 0),
                          getattr(self.tokenizer, "model_max_length", 0))
        valid_contexts = [int(value) for value in context_values if isinstance(value, int) and 0 < value < 1_000_000]
        if not valid_contexts:
            raise PromptBudgetError("model exposes no finite context window")
        context_limit = min(valid_contexts)
        budget = self._input_token_budget(context_limit)
        prefix, separator, tail = prompt.partition("CONTEXT=")
        if not separator or "\nREPAIR_FEEDBACK=" not in tail:
            inputs = self._tokenize_prompt(prompt)
            self.visible_operands = None
            return prompt, inputs
        context_text, feedback = tail.rsplit("\nREPAIR_FEEDBACK=", 1)
        context = json.loads(context_text)
        candidates = context.get("allowed_operands", [])
        chosen = []
        for candidate in candidates:
            proposal = dict(context, allowed_operands=chosen + [candidate])
            candidate_prompt = prefix + "CONTEXT=" + json.dumps(proposal, ensure_ascii=False, separators=(",", ":")) + "\nREPAIR_FEEDBACK=" + feedback
            if int(self._tokenize_prompt(candidate_prompt)["input_ids"].shape[-1]) > budget:
                break
            chosen.append(candidate)
        if candidates and not chosen:
            raise PromptBudgetError(f"first grounded operand exceeds {budget}-token input budget")
        compact_context = dict(context, allowed_operands=chosen)
        fitted_prompt = prefix + "CONTEXT=" + json.dumps(compact_context, ensure_ascii=False, separators=(",", ":")) + "\nREPAIR_FEEDBACK=" + feedback
        inputs = self._tokenize_prompt(fitted_prompt)
        self.visible_operands = {
            (x["table_key"], x["row_index"], x["column_index"]) for x in chosen
        }
        return fitted_prompt, inputs


def generate_plan(
    linked: SchemaLinkResult,
    tables_by_key: dict[str, dict],
    complete: Callable[[str], str],
    feedback: str | None = None,
) -> QueryPlan:
    prompt_linked = SchemaLinkResult(
        question=linked.question,
        query_family=linked.query_family,
        requested_unit=linked.requested_unit,
        table_keys=linked.table_keys,
        operands=linked.operands[:MAX_PROMPT_OPERANDS],
        resolved_tickers=linked.resolved_tickers,
    )
    payload = _extract_json(complete(build_prompt(prompt_linked, feedback)))
    visible_operands = getattr(complete, "visible_operands", None)
    if visible_operands is not None:
        prompt_linked = SchemaLinkResult(
            question=prompt_linked.question, query_family=prompt_linked.query_family,
            requested_unit=prompt_linked.requested_unit, table_keys=prompt_linked.table_keys,
            operands=[x for x in prompt_linked.operands if (x.table_key, x.row_index, x.column_index) in visible_operands],
            resolved_tickers=prompt_linked.resolved_tickers,
        )
    return validate_plan(payload, prompt_linked, tables_by_key)

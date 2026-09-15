"""Deterministic, gold-free repair triggers T1-T9 (POST_INFERENCE_REPAIR_DESIGN.md §4).

Every function here reads only fields already on disk in predictions/*.json or failures/*.json
(plus cheap regex parsing of a table's own report_id string, repair.common.parse_report_id) --
none of them load the corpus or call the model. This keeps `repair.scan` a fast, GPU-free triage
pass that can run over the full 1012-question artifact set in seconds, before any repair action
is chosen. Actual repair (repair.engine) re-derives ticker/year/variant from the real table
records instead of trusting report_id parsing, since a repair *action* needs the precision the
diagnosis-only heuristics here deliberately trade away for speed.
"""

from __future__ import annotations

import math
import re

from repair.common import Diagnosis, MULTI_OPERAND_FAMILIES, RepairTrigger, parse_report_id
from retrieval.full_corpus import (
    TICKER_TOKEN_RE,
    _NON_TICKER_CODES,
    infer_question_variant,
    infer_question_years,
)

_OPERAND_REF_RE = re.compile(r"(\w+)\['row_index'\]==(\d+)\)\s*&\s*\(\w+\['column_index'\]==(\d+)\)")
_UNAMBIGUOUS_UNIT_RE = re.compile(r"no unambiguous source unit")
_JSON_FORMAT_ERROR_TYPES = frozenset({"JSONDecodeError", "SyntaxError"})
_EVIDENCE_POOL_ERROR_RE = re.compile(
    r"missing required companies|incomplete operand group|no numeric|no direct-lookup operand"
    r"|no operand matches|schema linker retained no operands|plan requires at least one operand"
)
_CROSS_COMPANY_ARGMAX_RE = re.compile(
    r"(công ty nào|doanh nghiệp nào|ngân hàng nào)", re.IGNORECASE,
)
_SUPERLATIVE_RE = re.compile(r"(cao nhất|lớn nhất|thấp nhất|nhỏ nhất)", re.IGNORECASE)


def _literal_tickers(question: str) -> set[str]:
    return {t for t in TICKER_TOKEN_RE.findall(question) if t not in _NON_TICKER_CODES}


def _used_table_facts(used_tables: list[str]) -> list[dict]:
    return [parse_report_id(key.rsplit("|", 1)[0]) for key in used_tables]


def trigger_t1_single_operand_collapse(pred: dict) -> RepairTrigger | None:
    family = pred.get("query_family")
    used = pred.get("used_tables") or []
    if family in MULTI_OPERAND_FAMILIES and len(used) <= 1:
        return RepairTrigger(
            "T1", f"family={family!r} needs >=2 operand tables, used_tables has {len(used)}", "high",
        )
    return None


def trigger_t2_missing_entity(pred: dict) -> RepairTrigger | None:
    literal = _literal_tickers(pred.get("question", ""))
    if not literal:
        return None
    used_tickers = {f["ticker"] for f in _used_table_facts(pred.get("used_tables") or []) if f["ticker"]}
    if used_tickers and not (literal & used_tickers):
        return RepairTrigger(
            "T2", f"question names ticker(s) {sorted(literal)}, used_tables ticker(s) are {sorted(used_tickers)}", "medium",
        )
    return None


def trigger_t3_missing_year(pred: dict) -> RepairTrigger | None:
    question_years = infer_question_years(pred.get("question", ""))
    if not question_years:
        return None
    used_years = {f["year"] for f in _used_table_facts(pred.get("used_tables") or []) if f["year"]}
    query = pred.get("pandas_query") or ""
    year_in_query = any(str(y) in query for y in question_years)
    if used_years and not (question_years & used_years) and not year_in_query:
        return RepairTrigger(
            "T3", f"question names year(s) {sorted(question_years)}, used_tables year(s) are {sorted(used_years)}", "medium",
        )
    return None


def trigger_t5_expression_no_structure(pred: dict) -> RepairTrigger | None:
    family = pred.get("query_family")
    if family not in {"comparison", "growth", "ratio_or_derived"}:
        return None
    refs = set(_OPERAND_REF_RE.findall(pred.get("pandas_query") or ""))
    if len(refs) <= 1:
        return RepairTrigger(
            "T5", f"family={family!r} but rendered query references only {len(refs)} distinct cell(s)", "high",
        )
    return None


def trigger_t6_wrong_variant(pred: dict) -> RepairTrigger | None:
    variant = infer_question_variant(pred.get("question", ""))
    if not variant:
        return None
    used_variants = {f["variant"] for f in _used_table_facts(pred.get("used_tables") or []) if f["variant"]}
    if used_variants and variant not in used_variants:
        return RepairTrigger(
            "T6", f"question requests variant={variant!r}, used_tables variant(s) are {sorted(used_variants)}", "medium",
        )
    return None


def trigger_t7_implausible_magnitude(pred: dict) -> RepairTrigger | None:
    answer = pred.get("answer")
    if answer is None or isinstance(answer, bool) or not isinstance(answer, (int, float)):
        return None
    if not math.isfinite(answer):
        return RepairTrigger("T7", "answer is non-finite despite executed=True", "high")
    if answer != 0 and (abs(answer) > 1e15 or abs(answer) < 1e-9):
        return RepairTrigger("T7", f"answer magnitude {answer!r} is implausible for a financial figure", "medium")
    return None


def trigger_t8_unsupported_shape(question: str) -> RepairTrigger | None:
    """Cross-company argmax ("công ty nào ... cao nhất") -- the current QueryPlan/DSL cannot
    return a company label, only a float (POST_INFERENCE_REPAIR_DESIGN.md §3 item 5). Never
    routed to repair; flagged for a future DSL extension instead."""
    q = question.lower()
    if _SUPERLATIVE_RE.search(q) and _CROSS_COMPANY_ARGMAX_RE.search(q):
        return RepairTrigger("T8", "cross-company argmax/categorical answer shape; not expressible by current DSL", "high")
    return None


_PREDICTION_TRIGGERS = (
    trigger_t1_single_operand_collapse,
    trigger_t2_missing_entity,
    trigger_t3_missing_year,
    trigger_t5_expression_no_structure,
    trigger_t6_wrong_variant,
    trigger_t7_implausible_magnitude,
)


def diagnose_prediction(pred: dict) -> Diagnosis:
    unsupported = trigger_t8_unsupported_shape(pred.get("question", ""))
    if unsupported:
        return Diagnosis(pred["id"], [unsupported], "mark_unsupported", 0)

    triggers = [t for fn in _PREDICTION_TRIGGERS if (t := fn(pred)) is not None]
    if not triggers:
        return Diagnosis(pred["id"], [], "keep", 0)

    codes = {t.code for t in triggers}
    family = pred.get("query_family")
    if family == "direct_lookup":
        action, budget = "rebuild_deterministic", 1
    elif codes & {"T2", "T3", "T6"}:
        action, budget = "relink_then_regenerate", 2
    elif codes & {"T1", "T5"}:
        action, budget = "regenerate", 2
    else:
        action, budget = "regenerate", 2
    return Diagnosis(pred["id"], triggers, action, budget)


def diagnose_failure(fail: dict) -> Diagnosis:
    """Diagnose a failures/*.json record (§1.2: no retrieved_tables/schema_linked_operands/
    query_family persisted -- only id/question/error_type/error_message). Dispatch is by the
    inner error_type(s) embedded in error_message (repair.engine recomputes retrieval + schema
    linking before actually attempting a repair; this function only decides the repair *action*
    cheaply, no corpus access)."""
    question = fail.get("question", "")
    unsupported = trigger_t8_unsupported_shape(question)
    if unsupported:
        return Diagnosis(fail["id"], [unsupported], "mark_unsupported", 0)

    message = fail.get("error_message", "")
    inner_types = set(re.findall(r"'error_type':\s*'([A-Za-z]+)'", message))
    trigger = RepairTrigger("T9", f"hard failure, inner error_type(s)={sorted(inner_types) or ['unknown']}", "high")

    if _UNAMBIGUOUS_UNIT_RE.search(message):
        return Diagnosis(fail["id"], [trigger], "resolve_unit_deterministic", 1)
    if _EVIDENCE_POOL_ERROR_RE.search(message):
        return Diagnosis(fail["id"], [trigger], "relink_then_regenerate", 2)
    if inner_types & _JSON_FORMAT_ERROR_TYPES:
        return Diagnosis(fail["id"], [trigger], "regenerate", 2)
    return Diagnosis(fail["id"], [trigger], "regenerate", 2)

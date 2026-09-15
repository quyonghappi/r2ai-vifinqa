"""Independently re-derive each fixture record's `expected_answer` and flag mismatches.

Per AGENTS.md §4/§7 (no pseudo-ground-truth, distinguish confirmed from hypothesis), a hand-
entered `expected_answer` is only as trustworthy as an independent recomputation of it. This
script never trusts the pipeline being evaluated -- it only re-parses the record's own recorded
`derivation` text (or, for newly-authored records, an explicit machine-checkable
`derivation_formula`/`derivation_values` pair) and compares.

Two record shapes are supported:
  1. New-style: `gold.derivation_formula` (a small Python arithmetic expression string) +
     `gold.derivation_values` (a flat {name: number} substitution dict) -- evaluated exactly,
     any mismatch is a hard failure. Every newly-authored fixture record must use this shape.
  2. Legacy prose (ported from dev_v1/retrieval_specificity_v1): `gold.derivation` is free text.
     A best-effort regex extracts a trailing arithmetic sub-expression when one is present in a
     recognizable shape ("(A-B)/D[*E]" or "NUM VND / 1eN"); a match is checked, no match is
     reported SKIPPED (not a failure -- these records were already independently verified by a
     human reading the real report grid, per their own `verification.note`, using a different,
     real protocol this script cannot replay for free-text derivations).

Records with no `gold.expected_answer` at all (retrieval-only ports, e.g. from
retrieval_specificity_v1, which never hand-derived a numeric answer) are reported as such and
never treated as a failure.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

_PAREN_DIFF_RE = re.compile(
    r"\((-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\)\s*/\s*([\d.]+(?:e\d+)?)"
    r"(?:\s*\*\s*([\d.]+))?"
)
_SCALE_ONLY_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*(?:VND|VNĐ)?\s*/\s*([\d.]+e\d+)"
)


def _last_match(pattern: re.Pattern, text: str) -> re.Match | None:
    matches = list(pattern.finditer(text))
    return matches[-1] if matches else None


def extract_legacy_formula_value(derivation: str) -> float | None:
    """Best-effort extraction of a computable value from free-text `derivation`. Returns None
    (never a guess) when no recognizable arithmetic shape is present."""
    diff_match = _last_match(_PAREN_DIFF_RE, derivation)
    if diff_match:
        a, b, denom, mult = diff_match.groups()
        value = (float(a) - float(b)) / float(denom)
        if mult:
            value *= float(mult)
        return value
    scale_match = _last_match(_SCALE_ONLY_RE, derivation)
    if scale_match:
        numerator, denom = scale_match.groups()
        return float(numerator) / float(denom)
    return None


_ALLOWED_NAMES = {"abs": abs, "max": max, "min": min, "sum": sum}


def evaluate_structured_formula(formula: str, values: dict) -> float:
    return float(eval(formula, {"__builtins__": {}, **_ALLOWED_NAMES}, dict(values)))  # noqa: S307


def check_record(record: dict) -> dict:
    gold = record["gold"]
    expected = gold.get("expected_answer")
    outcome = {"id": record["id"]}
    if expected is None:
        outcome["status"] = "NO_EXPECTED_ANSWER"
        return outcome

    tol = gold.get("answer_tolerance") or {}
    abs_tol = tol.get("abs_tol", 1e-6)
    rel_tol = tol.get("rel_tol", 1e-6)

    if gold.get("derivation_formula") and gold.get("derivation_values") is not None:
        recomputed = evaluate_structured_formula(gold["derivation_formula"], gold["derivation_values"])
        ok = math.isclose(recomputed, expected, abs_tol=abs_tol, rel_tol=rel_tol)
        outcome.update(status="CHECKED_PASS" if ok else "CHECKED_FAIL", recomputed=recomputed, expected=expected)
        return outcome

    derivation = gold.get("derivation") or ""
    recomputed = extract_legacy_formula_value(derivation)
    if recomputed is None:
        outcome["status"] = "SKIPPED_NO_EXTRACTABLE_FORMULA"
        return outcome
    ok = math.isclose(recomputed, expected, abs_tol=abs_tol, rel_tol=rel_tol)
    outcome.update(status="CHECKED_PASS" if ok else "CHECKED_FAIL", recomputed=recomputed, expected=expected)
    return outcome


def run(fixture_path: Path) -> dict:
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    results = [check_record(record) for record in fixture]
    summary = {
        "n_records": len(results),
        "n_checked_pass": sum(r["status"] == "CHECKED_PASS" for r in results),
        "n_checked_fail": sum(r["status"] == "CHECKED_FAIL" for r in results),
        "n_skipped_no_formula": sum(r["status"] == "SKIPPED_NO_EXTRACTABLE_FORMULA" for r in results),
        "n_no_expected_answer": sum(r["status"] == "NO_EXPECTED_ANSWER" for r in results),
    }
    return {"summary": summary, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture", type=Path,
        default=Path(__file__).parent / "dev_questions" / "pipeline_fixture_v1.json",
    )
    args = parser.parse_args()
    report = run(args.fixture)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["summary"]["n_checked_fail"]:
        raise SystemExit(f"{report['summary']['n_checked_fail']} record(s) failed derivation re-check")


if __name__ == "__main__":
    main()

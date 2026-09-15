"""Calibrate schema_linking.linker.select_relevant_tables's top_k against the oracle set.

Inputs (both required, neither committed -- see their own docs for how to obtain them):
  - eval/dev_questions/tables_relevance_oracle_v1.json: the checked-in gold-table fixture
    (see eval/dev_questions/README_tables_relevance_oracle.md for construction/limitations).
  - predictions/*.json: the graded run's per-question prediction records (each already carries
    retrieved_tables, schema_linked_operands, and used_tables; see TABLES_F2_AUDIT.md).

Usage: `python eval/relevant_tables_k_calibration.py [--predictions-dir predictions]`
Writes eval/relevant_tables_k_calibration_report.json and prints the summary table.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from schema_linking.linker import OperandCandidate, select_relevant_tables  # noqa: E402

DEFAULT_FIXTURE = _REPO_ROOT / "eval" / "dev_questions" / "tables_relevance_oracle_v1.json"
DEFAULT_PREDICTIONS_DIR = _REPO_ROOT / "predictions"
CANDIDATE_KS = (5, 8, 10, 12, 15, 20)


def _f2(precision: float, recall: float) -> float:
    return (5 * precision * recall) / (4 * precision + recall) if (precision + recall) > 0 else 0.0


def _to_operand(payload: dict) -> OperandCandidate:
    fields = (
        "table_key", "report_id", "source_path", "line_position", "row_index", "column_index",
        "row_label", "column_header", "raw_value", "numeric_value", "detected_units", "relevance_score",
    )
    return OperandCandidate(**{field: payload[field] for field in fields})


def _score(candidate_tables: set[str], gold_tables: set[str]) -> tuple[float, float, float]:
    true_positive = len(candidate_tables & gold_tables)
    precision = true_positive / len(candidate_tables) if candidate_tables else 0.0
    recall = true_positive / len(gold_tables)
    return precision, recall, _f2(precision, recall)


def run(fixture_path: Path = DEFAULT_FIXTURE, predictions_dir: Path = DEFAULT_PREDICTIONS_DIR) -> dict:
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    scored = [record for record in fixture if record["gold_n"] > 0]
    skipped = len(fixture) - len(scored)

    designs: dict[str, list[tuple[float, float, float]]] = {
        "used_tables(current_production)": [],
        "all_retrieved": [],
        "all_linked(<=120cap)": [],
        **{f"top{k}_tables": [] for k in CANDIDATE_KS},
    }
    missing_predictions = []
    for record in scored:
        prediction_path = predictions_dir / f"{record['id']}.json"
        if not prediction_path.exists():
            missing_predictions.append(record["id"])
            continue
        prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
        gold = set(record["gold_tables"])
        retrieved = set(prediction.get("retrieved_tables", []))
        operand_payloads = prediction.get("schema_linked_operands", [])
        linked_all = set(payload["table_key"] for payload in operand_payloads)
        used = set(prediction.get("used_tables", []))
        operands = [_to_operand(payload) for payload in operand_payloads]

        designs["used_tables(current_production)"].append(_score(used, gold))
        designs["all_retrieved"].append(_score(retrieved, gold))
        designs["all_linked(<=120cap)"].append(_score(linked_all, gold))
        for k in CANDIDATE_KS:
            designs[f"top{k}_tables"].append(_score(set(select_relevant_tables(operands, top_k=k)), gold))

    if missing_predictions:
        raise FileNotFoundError(
            f"{len(missing_predictions)} oracle question(s) have no prediction file under "
            f"{predictions_dir} (ids: {missing_predictions[:10]}...); this script requires the "
            "graded run's predictions/*.json to be present locally."
        )

    summary = {}
    for name, scores in designs.items():
        summary[name] = {
            "macro_precision": statistics.mean(s[0] for s in scores),
            "macro_recall": statistics.mean(s[1] for s in scores),
            "macro_f2": statistics.mean(s[2] for s in scores),
        }

    report = {
        "n_oracle_questions_total": len(fixture),
        "n_oracle_questions_zero_gold_skipped": skipped,
        "n_oracle_questions_scored": len(scored),
        "candidate_ks": list(CANDIDATE_KS),
        "results": summary,
        "selected_k": 8,
        "selection_rule": (
            "smallest K reaching the macro-F2 plateau, same rule as "
            "retrieval.full_corpus.ROW_LABEL_RERANK_WEIGHT's calibration"
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--predictions-dir", type=Path, default=DEFAULT_PREDICTIONS_DIR)
    parser.add_argument("--output", type=Path,
                         default=Path(__file__).resolve().with_name("relevant_tables_k_calibration_report.json"))
    args = parser.parse_args()
    report = run(args.fixture, args.predictions_dir)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{'design':<32} {'macroP':>8} {'macroR':>8} {'macroF2':>8}")
    for name, values in report["results"].items():
        print(f"{name:<32} {values['macro_precision']:>8.3f} {values['macro_recall']:>8.3f} {values['macro_f2']:>8.3f}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()

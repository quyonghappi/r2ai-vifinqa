"""python -m repair.repair_predictions --predictions predictions \
    --diagnostics repair_runs/current/diagnostics.jsonl --output repair_runs/current --resume \
    [--catalog PATH --companies PATH --structured PATH --model-path PATH --retrieval-top-k 10]

Phase 2 (POST_INFERENCE_REPAIR_DESIGN.md §5.2-5.4): high-confidence semantic repair over
predictions that already executed but tripped a deterministic trigger (T1-T7; see repair.diagnose)
-- single-operand collapse, missing entity/year/variant, op/query mismatch, implausible magnitude.
Reads ``--diagnostics`` (repair.scan's output) to know which ids to touch instead of re-diagnosing
here, so scan and repair stay independently re-runnable. Ids diagnosed "keep" or
"mark_unsupported" are never repaired. Never writes into --predictions.

Output layout matches repair_failures.py: accepted/{id}.json, rejected/{id}.json,
predictions_log.jsonl (resume log), data/ (evidence CSVs for accepted repairs). Unsupported ids
from --diagnostics are copied into unsupported.jsonl here too, for one combined audit trail per
repair run when both repair_failures.py and repair_predictions.py write into the same --output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import repair._bootstrap  # noqa: F401

from common.table_store import load_tables_by_key
from query_generation.generator import QwenAWQGenerator
from repair.common import atomic_write_json, git_revision, processed_ids, append_jsonl, read_jsonl
from repair.engine import batch_retrieve, evaluate_candidate, repair_one
from retrieval.full_corpus import load_company_by_ticker

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_predictions(predictions_dir: Path, ids: set[int]) -> dict[int, dict]:
    out = {}
    for i in ids:
        path = predictions_dir / f"{i}.json"
        if path.exists():
            out[i] = json.loads(path.read_text(encoding="utf-8"))
    return out


def _flagged_ids(diagnostics_path: Path) -> dict[int, dict]:
    rows = {}
    for row in read_jsonl(diagnostics_path):
        if row.get("source") == "prediction" and row.get("action") not in ("keep", "mark_unsupported"):
            rows[row["id"]] = row
        elif row.get("source") == "prediction" and row.get("action") == "mark_unsupported":
            rows.setdefault("__unsupported__", []).append(row)  # type: ignore[index]
    return rows


def _write_evidence(output_dir: Path, prediction: dict) -> dict:
    evidence_dir = output_dir / "data"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    frames = prediction.pop("evidence_frames", {})
    evidence = []
    for variable, frame in frames.items():
        name = f"q{prediction['id']}_{variable}.csv"
        frame.to_csv(evidence_dir / name, index=False)
        evidence.append({"variable": variable, "csv_path": f"data/{name}"})
    prediction["evidence"] = evidence
    return prediction


def run(
    predictions_dir: Path,
    diagnostics_path: Path,
    output_dir: Path,
    *,
    catalog_path: Path,
    companies_path: Path,
    structured_path: Path,
    model_path: Path | None = None,
    retrieval_top_k: int | None = None,
    row_label_index_path: Path | None = None,
    resume: bool = True,
) -> dict:
    from repair.common import Diagnosis, RepairTrigger

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "predictions_log.jsonl"
    already_attempted = processed_ids(log_path) if resume else set()

    flagged = _flagged_ids(diagnostics_path)
    unsupported_rows = flagged.pop("__unsupported__", [])
    for row in unsupported_rows:
        append_jsonl(output_dir / "unsupported.jsonl", row)

    pending_rows = {i: row for i, row in flagged.items() if i not in already_attempted}
    summary = {"n_flagged": len(flagged), "n_pending": len(pending_rows), "accepted": 0,
               "rejected": 0, "pending_llm": 0}
    if not pending_rows:
        return summary

    predictions = _load_predictions(predictions_dir, set(pending_rows))
    company_by_ticker = load_company_by_ticker(companies_path)

    retrieved_by_id = {i: predictions[i].get("retrieved_tables") or [] for i in pending_rows if i in predictions}
    if retrieval_top_k is not None:
        widen_ids = [
            i for i, row in pending_rows.items()
            if i in predictions and row["action"] == "relink_then_regenerate"
            and len(retrieved_by_id.get(i, [])) < retrieval_top_k
        ]
        if widen_ids:
            bm25_path = output_dir / f"retrieval_widened_top{retrieval_top_k}.json"
            widened = batch_retrieve(
                [{"id": i, "question": predictions[i]["question"]} for i in widen_ids],
                catalog_path, companies_path, bm25_path, top_k=retrieval_top_k,
                row_label_index_path=row_label_index_path,
            )
            retrieved_by_id.update(widened)

    needed = {key for keys in retrieved_by_id.values() for key in keys}
    tables_by_key = load_tables_by_key(structured_path, needed)

    complete = None
    if model_path is not None:
        complete = QwenAWQGenerator(str(model_path))

    revision = git_revision(REPO_ROOT)
    for i, row in pending_rows.items():
        original = predictions.get(i)
        if original is None:
            append_jsonl(log_path, {"id": i, "attempted": True, "accepted": False, "reason": "prediction file missing"})
            continue
        diagnosis = Diagnosis(
            i, [RepairTrigger(**t) for t in row["triggers"]], row["action"], row["retry_budget"],
        )
        result = repair_one(i, original["question"], retrieved_by_id.get(i, []), tables_by_key, diagnosis, complete, company_by_ticker)
        if result["requires_llm"] and result["prediction"] is None:
            append_jsonl(output_dir / "pending_llm.jsonl", {"id": i, "action": diagnosis.action})
            summary["pending_llm"] += 1
            continue

        outcome = evaluate_candidate(i, original, result["prediction"], repair_code_revision=revision)
        if outcome.accepted:
            prediction = _write_evidence(output_dir, outcome.new_prediction)
            atomic_write_json(output_dir / "accepted" / f"{i}.json", prediction)
            summary["accepted"] += 1
        else:
            candidate = result["prediction"]
            if candidate is not None:
                candidate.pop("evidence_frames", None)
            atomic_write_json(output_dir / "rejected" / f"{i}.json", {
                "candidate": candidate, "reason": outcome.reason, "diagnosis": diagnosis.to_json(),
            })
            summary["rejected"] += 1
        append_jsonl(log_path, {"id": i, "attempted": True, **outcome.to_json()})

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=Path("data/processed/normalized_tables.csv"))
    parser.add_argument("--companies", type=Path, default=Path("data/raw/hf_meta/code_stock.csv"))
    parser.add_argument("--structured", type=Path, default=Path("data/processed/normalized_tables.jsonl"))
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--retrieval-top-k", type=int, default=None,
                         help="per-question retrieval widening for relink_then_regenerate ids only "
                              "(0409_AUDIT.md's own caution: never widen retrieval globally)")
    parser.add_argument("--row-label-index", type=Path, default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()
    summary = run(
        args.predictions, args.diagnostics, args.output, catalog_path=args.catalog,
        companies_path=args.companies, structured_path=args.structured, model_path=args.model_path,
        retrieval_top_k=args.retrieval_top_k, row_label_index_path=args.row_label_index, resume=args.resume,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

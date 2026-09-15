"""python -m repair.repair_failures --failures failures --output repair_runs/current --resume
    [--catalog PATH --companies PATH --structured PATH --model-path PATH --top-k 10]

Phase 1 (POST_INFERENCE_REPAIR_DESIGN.md §5.1): recover failures/*.json. failures/*.json carries
only {id, question, error_type, error_message} (§1.2) -- retrieval and schema-linking are always
recomputed here (deterministic, GPU-free), never assumed cached. Retrieval is batched (one
`rank_questions` call over every pending failure, not one per id) and the LLM (if --model-path is
given) is instantiated once and reused across every id that needs it, per the "avoid repeatedly
loading the LLM or retrieval indexes per question" instruction.

Never writes into --failures or the base predictions/ directory. Writes:
  <output>/accepted/{id}.json         -- accepted, prediction-shaped, ready for repair.finalize
  <output>/rejected/{id}.json         -- attempted but rejected (candidate + reason), for audit
  <output>/unsupported.jsonl          -- T8 marks, never attempted
  <output>/pending_llm.jsonl          -- needs an LLM, none was supplied this run (retried on the
                                          next --resume run once --model-path is provided)
  <output>/failures_log.jsonl         -- append-only per-id resume log (repair.common.append_jsonl)
  <output>/data/                      -- evidence CSVs for accepted recoveries
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import repair._bootstrap  # noqa: F401

from common.table_store import load_tables_by_key
from repair.common import (
    atomic_write_json,
    git_revision,
    processed_ids,
    append_jsonl,
)
from query_generation.generator import QwenAWQGenerator
from repair.diagnose import diagnose_failure
from repair.engine import batch_retrieve, evaluate_candidate, repair_one
from retrieval.full_corpus import load_company_by_ticker

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_failures(failures_dir: Path) -> list[dict]:
    records = []
    for path in sorted(failures_dir.glob("*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


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
    failures_dir: Path,
    output_dir: Path,
    *,
    catalog_path: Path,
    companies_path: Path,
    structured_path: Path,
    model_path: Path | None = None,
    top_k: int = 10,
    row_label_index_path: Path | None = None,
    resume: bool = True,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "failures_log.jsonl"
    already_attempted = processed_ids(log_path) if resume else set()

    failures = _load_failures(failures_dir)
    pending = [f for f in failures if f["id"] not in already_attempted]
    summary = {"n_failures": len(failures), "n_pending": len(pending), "accepted": 0, "rejected": 0,
               "unsupported": 0, "pending_llm": 0}
    if not pending:
        return summary

    diagnoses = {f["id"]: diagnose_failure(f) for f in pending}
    to_repair = [f for f in pending if diagnoses[f["id"]].action != "mark_unsupported"]
    for f in pending:
        if diagnoses[f["id"]].action == "mark_unsupported":
            append_jsonl(output_dir / "unsupported.jsonl", {"id": f["id"], **diagnoses[f["id"]].to_json()})
            append_jsonl(log_path, {"id": f["id"], "attempted": True, "accepted": False,
                                     "reason": "unsupported_shape"})
            summary["unsupported"] += 1

    if not to_repair:
        return summary

    bm25_path = output_dir / f"retrieval_failures_top{top_k}.json"
    retrieved_by_id = batch_retrieve(
        [{"id": f["id"], "question": f["question"]} for f in to_repair],
        catalog_path, companies_path, bm25_path, top_k=top_k, row_label_index_path=row_label_index_path,
    )
    needed = {key for keys in retrieved_by_id.values() for key in keys}
    tables_by_key = load_tables_by_key(structured_path, needed)
    company_by_ticker = load_company_by_ticker(companies_path)

    complete = None
    if model_path is not None:
        complete = QwenAWQGenerator(str(model_path))

    revision = git_revision(REPO_ROOT)
    for f in to_repair:
        diagnosis = diagnoses[f["id"]]
        result = repair_one(
            f["id"], f["question"], retrieved_by_id[f["id"]], tables_by_key, diagnosis,
            complete, company_by_ticker,
        )
        if result["requires_llm"] and result["prediction"] is None:
            append_jsonl(output_dir / "pending_llm.jsonl", {"id": f["id"], "action": diagnosis.action})
            summary["pending_llm"] += 1
            continue  # not added to already_attempted -- retried once a model is supplied

        outcome = evaluate_candidate(f["id"], None, result["prediction"], repair_code_revision=revision)
        if outcome.accepted:
            prediction = _write_evidence(output_dir, outcome.new_prediction)
            atomic_write_json(output_dir / "accepted" / f"{f['id']}.json", prediction)
            summary["accepted"] += 1
        else:
            candidate = result["prediction"]
            if candidate is not None:
                candidate.pop("evidence_frames", None)
            atomic_write_json(output_dir / "rejected" / f"{f['id']}.json", {
                "candidate": candidate, "reason": outcome.reason, "diagnosis": diagnosis.to_json(),
            })
            summary["rejected"] += 1
        append_jsonl(log_path, {"id": f["id"], "attempted": True, **outcome.to_json()})

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=Path("data/processed/normalized_tables.csv"))
    parser.add_argument("--companies", type=Path, default=Path("data/raw/hf_meta/code_stock.csv"))
    parser.add_argument("--structured", type=Path, default=Path("data/processed/normalized_tables.jsonl"))
    parser.add_argument("--model-path", type=Path, default=None,
                         help="attached Qwen AWQ Dataset path; omit to run the deterministic-only "
                              "layer and record LLM-requiring ids in pending_llm.jsonl")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--row-label-index", type=Path, default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()
    summary = run(
        args.failures, args.output, catalog_path=args.catalog, companies_path=args.companies,
        structured_path=args.structured, model_path=args.model_path, top_k=args.top_k,
        row_label_index_path=args.row_label_index, resume=args.resume,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

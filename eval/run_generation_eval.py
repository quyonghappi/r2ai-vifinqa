"""Resumable Kaggle-only Checkpoint-3 inference and local-dev evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.table_store import load_tables_by_key
from eval.generation_metrics import evaluate_answers
from pipeline import answer_question
from query_generation.generator import QwenAWQGenerator


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def run(
    model_path: Path,
    structured_path: Path,
    rankings_path: Path,
    dev_path: Path,
    answers_path: Path,
    output_dir: Path,
) -> dict:
    questions = _jsonl(dev_path)
    answers = {x["id"]: x for x in _jsonl(answers_path)}
    rankings = json.loads(rankings_path.read_text(encoding="utf-8"))
    retrieved = {q["id"]: [x[0] for x in rankings[str(q["id"])][:10]] for q in questions}
    tables = load_tables_by_key(structured_path, {key for values in retrieved.values() for key in values})
    prediction_dir, evidence_dir = output_dir / "predictions", output_dir / "data"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    pending = [q for q in questions if not (prediction_dir / f"{q['id']}.json").exists()]
    generator = QwenAWQGenerator(str(model_path)) if pending else None

    for q in pending:
        result = answer_question(q, retrieved[q["id"]], tables, generator)
        evidence = []
        for variable, frame in result.pop("evidence_frames").items():
            csv_path = evidence_dir / f"q{q['id']}_{variable}.csv"
            frame.to_csv(csv_path, index=False)
            evidence.append({"variable": variable, "csv_path": csv_path.as_posix()})
        result["evidence"] = evidence
        result.update(answers[q["id"]])
        tmp = prediction_dir / f"{q['id']}.json.tmp"
        tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(prediction_dir / f"{q['id']}.json")

    records = [json.loads((prediction_dir / f"{q['id']}.json").read_text(encoding="utf-8")) for q in questions]
    report = evaluate_answers(records, abs_tol=1e-6, rel_tol=1e-6)
    report.update({
        "dev_set": dev_path.as_posix(), "answers": answers_path.as_posix(),
        "ground_truth_status": "self-constructed manually derived diagnostic answers; not official labels",
        "model_path": str(model_path), "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
        "top_k": 10,
    })
    (output_dir / "generation_dev_v1_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--structured", type=Path, default=Path("data/processed/normalized_tables.jsonl"))
    parser.add_argument("--rankings", type=Path, default=Path("eval/reports/retrieval_dev_v1_bm25_rankings.json"))
    parser.add_argument("--dev", type=Path, default=Path("eval/dev_questions/dev_v1.jsonl"))
    parser.add_argument("--answers", type=Path, default=Path("eval/dev_questions/dev_v1_answers.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval/reports/generation_dev_v1"))
    args = parser.parse_args()
    print(json.dumps(run(args.model_path, args.structured, args.rankings, args.dev, args.answers, args.output_dir), indent=2))


if __name__ == "__main__":
    main()

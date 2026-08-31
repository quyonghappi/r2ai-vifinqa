"""Format successful per-question predictions into the official submission JSON."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REQUIRED_FIELDS = (
    "id", "question", "answer", "relevant_docs", "relevant_tables", "evidence", "pandas_query"
)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build(questions_path: Path, prediction_dir: Path, output_path: Path) -> list[dict]:
    """Pure formatter: refuse missing/failed outputs instead of fabricating answers."""
    questions = _jsonl(questions_path)
    records, errors = [], []
    for q in questions:
        path = prediction_dir / f"{q['id']}.json"
        if not path.exists():
            errors.append(f"id={q['id']}: missing prediction")
            continue
        prediction = json.loads(path.read_text(encoding="utf-8"))
        if prediction.get("executed") is not True:
            errors.append(f"id={q['id']}: prediction did not execute successfully")
            continue
        if prediction.get("question") != q["question"]:
            errors.append(f"id={q['id']}: prediction question differs from official source")
            continue
        answer = prediction.get("answer")
        if isinstance(answer, bool) or not isinstance(answer, (int, float)) or not math.isfinite(float(answer)):
            errors.append(f"id={q['id']}: answer is not finite numeric")
            continue
        try:
            records.append({field: prediction[field] for field in REQUIRED_FIELDS})
        except KeyError as exc:
            errors.append(f"id={q['id']}: missing field {exc.args[0]}")
    if errors:
        preview = "\n".join(errors[:20])
        raise ValueError(f"cannot build incomplete submission ({len(errors)} errors):\n{preview}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output_path)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=Path("data/raw/hf_meta/questions.jsonl"))
    parser.add_argument("--predictions", type=Path, default=Path("submission/package/work/predictions"))
    parser.add_argument("--output", type=Path, default=Path("submission/package/submission.json"))
    args = parser.parse_args()
    print(f"built {len(build(args.questions, args.predictions, args.output))} records at {args.output}")


if __name__ == "__main__":
    main()

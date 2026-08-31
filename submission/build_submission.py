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


def build(
    questions_path: Path, prediction_dir: Path, output_path: Path, *, allow_partial: bool = False
) -> list[dict]:
    """Pure formatter: refuse missing/failed outputs instead of fabricating answers.

    `allow_partial=False` (default) keeps the original all-or-nothing gate: any missing or
    failed prediction raises, so a bug can't silently ship a partial submission. Set
    `allow_partial=True` only once inference is done retrying and some questions are expected
    to stay unanswered (see CHANGE_LOG.md 2026-08-31 partial-submission entry) -- per
    docs/submission_guide.md, a missing question is scored as an invalid prediction for that
    question, not a rejected submission file, so omitting it (never fabricating an answer) is
    within the official contract. Malformed *present* predictions -- wrong question text,
    non-finite answer despite executed=True, a missing required field -- still always raise:
    those indicate a real bug, not mere incompleteness, and allow_partial does not mask them.
    """
    questions = _jsonl(questions_path)
    records, errors, skipped = [], [], []
    for q in questions:
        path = prediction_dir / f"{q['id']}.json"
        if not path.exists():
            (skipped if allow_partial else errors).append(f"id={q['id']}: missing prediction")
            continue
        prediction = json.loads(path.read_text(encoding="utf-8"))
        if prediction.get("executed") is not True:
            (skipped if allow_partial else errors).append(f"id={q['id']}: prediction did not execute successfully")
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
    if skipped:
        print(f"allow_partial: omitting {len(skipped)} unanswered question(s) from the submission")

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
    parser.add_argument("--allow-partial", action="store_true",
                         help="omit unanswered questions instead of refusing to build (see build()'s docstring)")
    args = parser.parse_args()
    n = len(build(args.questions, args.predictions, args.output, allow_partial=args.allow_partial))
    print(f"built {n} records at {args.output}")


if __name__ == "__main__":
    main()

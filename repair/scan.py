"""python -m repair.scan --predictions predictions --failures failures --output repair_runs/current/diagnostics.jsonl

Fast, GPU-free, read-only triage: diagnose every prediction and failure, write one JSON line per
id (`repair.common.Diagnosis.to_json()` shape) to ``--output``. Never touches ``--predictions``/
``--failures``. Safe to re-run any time (rebuilds the whole diagnostics file fresh each run,
atomically) -- this script does no repair itself, only classification, so there is nothing to
resume.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import repair._bootstrap  # noqa: F401  (sys.path side effect, must run before src/eval imports)

from repair.diagnose import diagnose_failure, diagnose_prediction


def _load_all(directory: Path) -> list[dict]:
    records = []
    for path in sorted(directory.glob("*.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return records


def scan(predictions_dir: Path, failures_dir: Path) -> list[dict]:
    rows = []
    for pred in _load_all(predictions_dir):
        diagnosis = diagnose_prediction(pred)
        rows.append({"id": pred["id"], "source": "prediction", **diagnosis.to_json()})
    for fail in _load_all(failures_dir):
        diagnosis = diagnose_failure(fail)
        rows.append({"id": fail["id"], "source": "failure", **diagnosis.to_json()})
    return rows


def write_report(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(output_path)


def summarize(rows: list[dict]) -> dict:
    from collections import Counter
    action_counts = Counter(r["action"] for r in rows)
    trigger_counts = Counter(t["code"] for r in rows for t in r["triggers"])
    overlap = Counter(
        tuple(sorted({t["code"] for t in r["triggers"]})) for r in rows if r["triggers"]
    )
    return {
        "n_diagnosed": len(rows),
        "n_predictions": sum(1 for r in rows if r["source"] == "prediction"),
        "n_failures": sum(1 for r in rows if r["source"] == "failure"),
        "action_counts": dict(action_counts),
        "trigger_counts": dict(trigger_counts),
        "trigger_combination_counts": {",".join(k) or "(none)": v for k, v in overlap.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--failures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = scan(args.predictions, args.failures)
    write_report(rows, args.output)
    print(json.dumps(summarize(rows), indent=2))


if __name__ == "__main__":
    main()

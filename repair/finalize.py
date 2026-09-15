"""python -m repair.finalize --predictions predictions --repair-run repair_runs/current --output repaired_predictions

Merge original predictions/ with a repair run's accepted/*.json (repaired predictions and/or
recovered failures) into --output, ready to hand to submission/build_submission.py's
``--predictions`` argument. Never writes into --predictions, --failures, or --repair-run --
--output is a brand-new directory tree, always safe to rebuild from scratch (every write here is
atomic per id, so a Kaggle interruption mid-finalize leaves --output in a valid, partial state,
not a corrupt one). Precedence is simple and total: a repaired/recovered record in
<repair-run>/accepted/ always wins over the original predictions/{id}.json, since acceptance
already passed the non-degradation gate in repair.engine.evaluate_candidate -- finalize does not
re-judge that decision, it only merges.

Does NOT assume submission.json already exists, and does not invoke build_submission.py itself --
that remains the caller's next, separate step (unchanged from before this module existed).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import repair._bootstrap  # noqa: F401

from repair.common import atomic_write_json


def _atomic_copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def finalize(predictions_dir: Path, repair_run_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    accepted_dir = repair_run_dir / "accepted"

    original_ids = {p.stem for p in predictions_dir.glob("*.json")}
    for path in predictions_dir.glob("*.json"):
        _atomic_copy_file(path, output_dir / path.name)

    accepted_ids = {p.stem for p in accepted_dir.glob("*.json")} if accepted_dir.exists() else set()
    overridden, recovered = [], []
    for path in sorted(accepted_dir.glob("*.json")) if accepted_dir.exists() else []:
        record = json.loads(path.read_text(encoding="utf-8"))
        atomic_write_json(output_dir / path.name, record)
        (recovered if path.stem not in original_ids else overridden).append(int(path.stem))

    report = {
        "n_total_output": len(list(output_dir.glob("*.json"))),
        "n_from_original_unchanged": len(original_ids - accepted_ids),
        "n_repaired_existing_prediction": len(overridden),
        "n_recovered_from_failure": len(recovered),
        "repaired_ids": sorted(overridden),
        "recovered_ids": sorted(recovered),
    }
    atomic_write_json(output_dir / "finalize_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--repair-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = finalize(args.predictions, args.repair_run, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

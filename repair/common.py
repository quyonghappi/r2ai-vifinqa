"""Shared IO/data-contract primitives for the repair pass (POST_INFERENCE_REPAIR_DESIGN.md §7.3)."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from query_generation.generator import _MULTI_OPERAND_FAMILIES

MULTI_OPERAND_FAMILIES = _MULTI_OPERAND_FAMILIES


def atomic_write_json(path: Path, value: object) -> None:
    """Same temp-file + os.replace pattern submission/run_full_inference.py's _atomic_json uses,
    so a Kaggle interruption mid-write never leaves a truncated/corrupt file behind."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, record: dict) -> None:
    """Append-only, resume-friendly log (one line per processed id) -- a crash mid-pass leaves a
    valid, replayable partial file rather than a corrupt structured document."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def processed_ids(jsonl_path: Path) -> set[int]:
    """The set of ids already recorded in a resumable .jsonl log -- used by --resume to skip
    work already done, per id, without re-deriving state from predictions/failures."""
    return {record["id"] for record in read_jsonl(jsonl_path) if "id" in record}


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def git_revision(repo_root: Path) -> str | None:
    """Best-effort git commit hash for provenance stamping (EXECUTION_ACCURACY_AUDIT.md §3's
    "no code-version stamp per official run" gap) -- never raises; a Kaggle snapshot without a
    .git directory just gets None, which callers must treat as "unknown", not an error."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() or None if result.returncode == 0 else None
    except Exception:
        return None


_REPORT_ID_RE = re.compile(
    r"^(?P<ticker>[A-Z0-9]+)_financial_statements_(?P<year>\d{4})(?:_(?P<variant>consolidated|separate))?$"
)


def parse_report_id(report_id: str) -> dict:
    """Best-effort ticker/year/variant extraction from a table's own ``report_id`` string, used
    only for cheap, corpus-free *diagnosis* (repair.diagnose) -- never for the actual repair
    action, which always re-derives these from the real table record via ``tables_by_key``. All
    three fields are None when the report_id doesn't match the expected shape (never raises), so
    a diagnosis trigger built on this is conservative by construction: unparseable is "unknown",
    not "conflict"."""
    match = _REPORT_ID_RE.match(report_id)
    if not match:
        return {"ticker": None, "year": None, "variant": None}
    groups = match.groupdict()
    return {
        "ticker": groups["ticker"],
        "year": int(groups["year"]) if groups["year"] else None,
        "variant": groups["variant"],
    }


@dataclass(frozen=True)
class RepairTrigger:
    code: str
    detail: str
    severity: str  # "high" | "medium"


@dataclass(frozen=True)
class Diagnosis:
    prediction_id: int
    triggers: list = field(default_factory=list)  # list[RepairTrigger]
    action: str = "keep"  # "keep" | "rebuild_deterministic" | "relink_then_regenerate"
                           # | "regenerate" | "mark_unsupported" | "recover_failure"
    retry_budget: int = 0

    def to_json(self) -> dict:
        return {
            "prediction_id": self.prediction_id,
            "triggers": [asdict(t) for t in self.triggers],
            "action": self.action,
            "retry_budget": self.retry_budget,
        }


@dataclass
class RepairOutcome:
    prediction_id: int
    accepted: bool
    reason: str
    previous_trigger_count: int
    new_trigger_count: int
    attempts_spent: int
    new_prediction: dict | None = None
    repair_code_revision: str | None = None

    def to_json(self) -> dict:
        d = asdict(self)
        d.pop("new_prediction")
        d["has_new_prediction"] = self.new_prediction is not None
        return d

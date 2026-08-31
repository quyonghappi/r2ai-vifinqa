"""Shared path constants and idempotent-artifact helpers.

Not a config framework -- just the directory layout every stage needs to agree on,
plus a helper for the Kaggle "resume from artifact if session was interrupted" requirement
(see AGENTS.md environment constraints). Kept as plain constants/functions per AGENTS.md
Section 3 (no DI/config framework until there's a real need for one).
"""

import os
from pathlib import Path


def _detect_root() -> Path:
    if os.environ.get("VIFINQA_ROOT"):
        return Path(os.environ["VIFINQA_ROOT"])
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working")
    return Path(__file__).resolve().parents[2]


ROOT = _detect_root()
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = ROOT / "eval"
DEV_QUESTIONS_DIR = EVAL_DIR / "dev_questions"

for _d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, DEV_QUESTIONS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def artifact_exists(path) -> bool:
    """True if a stage artifact was already produced -- lets a cell be skipped on resume."""
    p = Path(path)
    return p.exists() and p.stat().st_size > 0

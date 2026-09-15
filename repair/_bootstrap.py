"""Path bootstrap shared by every repair/ entry point, mirroring submission/run_full_inference.py
(same sys.path.insert pattern, so this package works identically in the local repo layout and in
the Kaggle snapshot layout where ``src``/``eval`` come from an attached Dataset)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("src", "eval"):
    _path = str(REPO_ROOT / _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)

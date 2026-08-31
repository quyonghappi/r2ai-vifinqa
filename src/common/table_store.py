"""Streaming access to selected records in the 667 MB structured-table JSONL artifact."""

from __future__ import annotations

import json
from pathlib import Path


def load_tables_by_key(path: str | Path, table_keys: set[str]) -> dict[str, dict]:
    """Scan once and stop as soon as every requested table is found; never load the corpus."""
    remaining, found = set(table_keys), {}
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            key = record["table_key"]
            if key in remaining:
                found[key] = record
                remaining.remove(key)
                if not remaining:
                    break
    if remaining:
        raise KeyError(f"structured artifact is missing table keys: {sorted(remaining)}")
    return found

"""Paste into a Kaggle notebook cell (after PROCESSED_DIR/INTERIM_DIR are defined, before
Checkpoint 2's normalization cell) to force a catalog rebuild instead of reusing the stale
`artifact_exists(...)` cache -- see CHANGE_LOG.md 2026-08-31 unit-declaration-threading entry."""

from pathlib import Path

stale_artifacts = [
    PROCESSED_DIR / "normalized_tables_entity_v6.csv",
    PROCESSED_DIR / "normalized_tables_entity_v6.jsonl",
    INTERIM_DIR / "extraction_full_anomalies.csv",
    INTERIM_DIR / "extraction_full_file_stats.csv",
]

for path in stale_artifacts:
    if path.exists():
        path.unlink()
        print(f"deleted: {path}")
    else:
        print(f"already absent: {path}")

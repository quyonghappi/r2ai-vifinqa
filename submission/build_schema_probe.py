"""Build a structurally complete, inference-free submission probe.

The official format has no nullable/empty inference representation.  This utility
therefore uses a clearly labelled placeholder table and a constant zero-valued
query for every official question.  It is suitable only for dashboard schema
testing, never for a scored submission.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PLACEHOLDER_DOCUMENT = "SCHEMA_PROBE_NO_INFERENCE"
PLACEHOLDER_TABLE = f"{PLACEHOLDER_DOCUMENT}|0"
PLACEHOLDER_CSV = "schema_probe_constant.csv"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build(questions_path: Path, package_dir: Path) -> Path:
    """Write the schema probe JSON and its one constant evidence CSV."""
    questions = _jsonl(questions_path)
    data_dir = package_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / PLACEHOLDER_CSV).write_text("answer\n0.0\n", encoding="utf-8")

    records = [
        {
            "id": question["id"],
            "question": question["question"],
            "answer": 0.0,
            "relevant_docs": [PLACEHOLDER_DOCUMENT],
            "relevant_tables": [PLACEHOLDER_TABLE],
            "evidence": [{"variable": "df1", "csv_path": f"data/{PLACEHOLDER_CSV}"}],
            "pandas_query": "float(df1['answer'].iloc[0])",
        }
        for question in questions
    ]
    output_path = package_dir / "submission.json"
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=Path("data/raw/hf_meta/questions.jsonl"))
    parser.add_argument("--package-dir", type=Path, default=Path("submission/package/schema_probe"))
    args = parser.parse_args()
    output = build(args.questions, args.package_dir)
    print(f"built schema-only probe for {len(_jsonl(args.questions))} questions at {output}")


if __name__ == "__main__":
    main()

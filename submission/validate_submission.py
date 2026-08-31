"""Strict local validator for the official ViFinQA submission package."""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from pathlib import Path, PurePosixPath

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from execution.runner import execute_query

REQUIRED_FIELDS = {
    "id", "question", "answer", "relevant_docs", "relevant_tables", "evidence", "pandas_query"
}
VARIABLE_RE = re.compile(r"^[A-Za-z_]\w*$")


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _query_variables(query: str, declared: set[str]) -> set[str]:
    tree = ast.parse(query, mode="eval")
    allowed_nodes = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Constant, ast.Add, ast.Sub,
        ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Call, ast.Load, ast.Attribute,
        ast.Subscript, ast.Compare, ast.Eq, ast.BitAnd,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"disallowed query syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in declared | {"float", "abs"}:
            raise ValueError(f"undeclared query name: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr not in {"loc", "iloc"}:
            raise ValueError(f"disallowed DataFrame attribute: {node.attr}")
        if isinstance(node, ast.Call) and not (
            isinstance(node.func, ast.Name) and node.func.id in {"float", "abs"}
        ):
            raise ValueError("only float(...) and abs(...) calls are allowed")
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and node.id in declared}


def validate(
    submission_path: Path,
    questions_path: Path,
    package_dir: Path | None = None,
    *,
    abs_tol: float = 1e-9,
    rel_tol: float = 1e-9,
) -> dict:
    """Validate coverage, schema, paths, and independent query execution.

    The tolerances check internal answer/query consistency only; they are not a claim about the
    unpublished official answer-scoring tolerance.
    """
    package_dir = (package_dir or submission_path.parent).resolve()
    data_root = (package_dir / "data").resolve()
    official = _jsonl(questions_path)
    official_by_id = {q["id"]: q for q in official}
    records = json.loads(submission_path.read_text(encoding="utf-8"))
    errors = []
    if not isinstance(records, list):
        return {"valid": False, "n_official": len(official), "n_records": 0,
                "errors": ["submission root must be a JSON list"]}

    seen = set()
    for index, record in enumerate(records):
        prefix = f"record[{index}]"
        try:
            if not isinstance(record, dict):
                raise ValueError("must be an object")
            missing = REQUIRED_FIELDS - set(record)
            if missing:
                raise ValueError(f"missing fields: {sorted(missing)}")
            qid = record["id"]
            if isinstance(qid, bool) or not isinstance(qid, int):
                raise ValueError("id must be an integer")
            if qid in seen:
                raise ValueError(f"duplicate id {qid}")
            seen.add(qid)
            if qid not in official_by_id:
                raise ValueError(f"unknown id {qid}")
            if record["question"] != official_by_id[qid]["question"]:
                raise ValueError("question does not exactly match official source")
            answer = record["answer"]
            if isinstance(answer, bool) or not isinstance(answer, (int, float)) or not math.isfinite(float(answer)):
                raise ValueError("answer must be a finite number")

            docs, tables = record["relevant_docs"], record["relevant_tables"]
            if not isinstance(docs, list) or not docs or not all(isinstance(x, str) and x for x in docs):
                raise ValueError("relevant_docs must be a non-empty string list")
            if len(docs) != len(set(docs)):
                raise ValueError("relevant_docs contains duplicates")
            if not isinstance(tables, list) or not tables:
                raise ValueError("relevant_tables must be non-empty")
            table_docs = []
            for key in tables:
                report, separator, line = key.rpartition("|") if isinstance(key, str) else ("", "", "")
                if not separator or not report or not line.isdigit():
                    raise ValueError(f"invalid relevant_tables key: {key!r}")
                table_docs.append(report)
            if docs != list(dict.fromkeys(table_docs)):
                raise ValueError("relevant_docs must exactly follow relevant_tables document order")

            evidence = record["evidence"]
            if not isinstance(evidence, list) or not evidence:
                raise ValueError("evidence must be non-empty")
            variables = [item.get("variable") for item in evidence if isinstance(item, dict)]
            if len(variables) != len(evidence) or any(
                not isinstance(name, str) or not VARIABLE_RE.fullmatch(name) for name in variables
            ):
                raise ValueError("every evidence variable must be a valid Python identifier")
            if len(variables) != len(set(variables)):
                raise ValueError("evidence variables must be unique")

            frames = {}
            for item in evidence:
                csv_path = item.get("csv_path")
                if not isinstance(csv_path, str):
                    raise ValueError("csv_path must be a string")
                pure = PurePosixPath(csv_path)
                if pure.is_absolute() or not pure.parts or pure.parts[0] != "data" or ".." in pure.parts:
                    raise ValueError(f"csv_path must be relative and under data/: {csv_path!r}")
                resolved = (package_dir / Path(*pure.parts)).resolve()
                if data_root not in resolved.parents:
                    raise ValueError(f"csv_path escapes data/: {csv_path!r}")
                if not resolved.is_file():
                    raise ValueError(f"evidence CSV does not exist: {csv_path}")
                frames[item["variable"]] = pd.read_csv(resolved)

            query = record["pandas_query"]
            if not isinstance(query, str) or not query.strip():
                raise ValueError("pandas_query must be a non-empty string")
            used = _query_variables(query, set(frames))
            if used != set(frames):
                raise ValueError(f"query/evidence variable mismatch: used={sorted(used)}, declared={sorted(frames)}")
            rerun = execute_query(query, frames)
            if not math.isclose(rerun, float(answer), abs_tol=abs_tol, rel_tol=rel_tol):
                raise ValueError(f"query result {rerun} does not match answer {answer}")
        except Exception as exc:
            errors.append(f"{prefix}: {type(exc).__name__}: {exc}")

    missing_ids = sorted(set(official_by_id) - seen)
    if missing_ids:
        errors.append(f"missing official ids ({len(missing_ids)}): {missing_ids[:20]}")
    report = {
        "valid": not errors, "n_official": len(official), "n_records": len(records),
        "abs_tolerance_internal": abs_tol, "rel_tolerance_internal": rel_tol, "errors": errors,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path, nargs="?", default=Path("submission/package/submission.json"))
    parser.add_argument("--questions", type=Path, default=Path("data/raw/hf_meta/questions.jsonl"))
    parser.add_argument("--package-dir", type=Path)
    args = parser.parse_args()
    report = validate(args.submission, args.questions, args.package_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

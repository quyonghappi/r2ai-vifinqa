"""Run the table-relevance oracle through the current retrieval and linking code.

This is a diagnostic evaluation over the self-constructed
``tables_relevance_oracle_v1`` fixture, not an official leaderboard metric.  Unlike
``relevant_tables_k_calibration.py``, it does not reuse persisted predictions: it
runs the current BM25 retrieval path and deterministic schema linking first.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from common.table_store import load_tables_by_key  # noqa: E402
from retrieval.full_corpus import (  # noqa: E402
    DEFAULT_RELATED_PARTY_NOTE_PENALTY,
    DEFAULT_STATEMENT_SECTION_BONUS,
    infer_question_tickers,
    infer_question_variant,
    infer_question_years,
    rank_questions,
)
from schema_linking.linker import link_schema, select_relevant_tables  # noqa: E402


DEFAULT_FIXTURE = _REPO_ROOT / "eval" / "dev_questions" / "tables_relevance_oracle_v1.json"
DEFAULT_CATALOG = _REPO_ROOT / "data" / "processed" / "normalized_tables.csv"
DEFAULT_COMPANIES = _REPO_ROOT / "data" / "raw" / "hf_meta" / "code_stock.csv"
DEFAULT_ROW_LABEL_INDEX = _REPO_ROOT / "data" / "processed" / "row_label_index.csv"
DEFAULT_STRUCTURED_TABLES = _REPO_ROOT / "data" / "processed" / "normalized_tables.jsonl"


def _f2(precision: float, recall: float) -> float:
    return 5 * precision * recall / (4 * precision + recall) if precision + recall else 0.0


def _fixture_display_path(path: Path) -> str:
    """Return a repository-relative fixture name for either CLI path form."""
    return path.resolve().relative_to(_REPO_ROOT).as_posix()


def score(candidate_tables: set[str], gold_tables: set[str]) -> dict[str, float]:
    true_positive = len(candidate_tables & gold_tables)
    precision = true_positive / len(candidate_tables) if candidate_tables else 0.0
    recall = true_positive / len(gold_tables)
    return {"precision": precision, "recall": recall, "f2": _f2(precision, recall)}


def macro(scores: list[dict[str, float]]) -> dict[str, float]:
    return {
        name: statistics.mean(item[name] for item in scores) if scores else 0.0
        for name in ("precision", "recall", "f2")
    }


def _company_map(companies_path: Path) -> dict[str, str]:
    import pandas as pd

    companies = pd.read_csv(companies_path, keep_default_na=False)
    return dict(zip(companies.iloc[:, 0].astype(str), companies.iloc[:, 1].astype(str)))


def _breakdowns(
    rows: list[dict], *, metrics_field: str, family_field: str
) -> dict[str, dict]:
    grouped: dict[str, defaultdict[str, list[dict[str, float]]]] = {
        "company_cardinality": defaultdict(list),
        "query_family_current": defaultdict(list),
    }
    for row in rows:
        cardinality = "multi_company" if len(row["oracle_tickers"]) > 1 else "single_company"
        grouped["company_cardinality"][cardinality].append(row[metrics_field])
        grouped["query_family_current"][row[family_field]].append(row[metrics_field])
    return {dimension: {name: macro(values) for name, values in groups.items()}
            for dimension, groups in grouped.items()}


def run(
    *,
    fixture_path: Path = DEFAULT_FIXTURE,
    catalog_path: Path = DEFAULT_CATALOG,
    companies_path: Path = DEFAULT_COMPANIES,
    row_label_index_path: Path = DEFAULT_ROW_LABEL_INDEX,
    structured_tables_path: Path = DEFAULT_STRUCTURED_TABLES,
    rankings_path: Path,
    retrieval_top_k: int = 10,
    statement_section_bonus: float = DEFAULT_STATEMENT_SECTION_BONUS,
    related_party_note_penalty: float = DEFAULT_RELATED_PARTY_NOTE_PENALTY,
) -> dict:
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    records = [record for record in fixture if record["gold_n"] > 0]
    questions = [{"id": record["id"], "question": record["question"]} for record in records]
    rankings = rank_questions(
        questions,
        catalog_path,
        companies_path,
        rankings_path,
        top_k=retrieval_top_k,
        row_label_index_path=row_label_index_path,
        statement_section_bonus=statement_section_bonus,
        related_party_note_penalty=related_party_note_penalty,
    )
    retrieved_by_id = {
        str(question["id"]): [item[0] for item in rankings[str(question["id"])] ]
        for question in questions
    }
    all_keys = {key for keys in retrieved_by_id.values() for key in keys}
    tables_by_key = load_tables_by_key(structured_tables_path, all_keys)
    company_by_ticker = _company_map(companies_path)

    rows = []
    recall_scores = []
    relevant_scores = []
    for record in records:
        question_id = str(record["id"])
        retrieved = retrieved_by_id[question_id]
        tables = [tables_by_key[key] for key in retrieved]
        linked = link_schema(record["question"], tables)
        relevant = select_relevant_tables(linked.operands)
        gold = set(record["gold_tables"])
        retrieved_metrics = score(set(retrieved), gold)
        relevant_metrics = score(set(relevant), gold)
        recall_scores.append(retrieved_metrics)
        relevant_scores.append(relevant_metrics)
        rows.append({
            "id": record["id"],
            "question": record["question"],
            "oracle_tickers": record["search_tickers"],
            "resolved_tickers": sorted(infer_question_tickers(record["question"], company_by_ticker)),
            "years": sorted(infer_question_years(record["question"])),
            "variant": infer_question_variant(record["question"]),
            "query_family_fixture_pre_fix": record["query_family"],
            "query_family_current": linked.query_family,
            "gold_n": len(gold),
            "retrieved_n": len(retrieved),
            "relevant_tables_n": len(relevant),
            "retrieved_tables": retrieved,
            "relevant_tables": relevant,
            "retrieval_metrics": retrieved_metrics,
            "relevant_tables_metrics": relevant_metrics,
        })

    return {
        "fixture": _fixture_display_path(fixture_path),
        "fixture_note": "Self-constructed oracle; see README_tables_relevance_oracle.md for limits.",
        "n_fixture_questions": len(fixture),
        "n_scored_questions": len(records),
        "n_zero_gold_skipped": len(fixture) - len(records),
        "retrieval": {
            "implementation": "current BM25 + optional row-label rerank; dense/RRF is not invoked",
            "per_resolved_company_top_k": retrieval_top_k,
            "statement_section_bonus": statement_section_bonus,
            "related_party_note_penalty": related_party_note_penalty,
            "macro_precision_recall_f2": macro(recall_scores),
            "macro_recall_at_per_company_k": statistics.mean(item["recall"] for item in recall_scores),
        },
        "relevant_tables": {
            "selector": "schema_linking.select_relevant_tables",
            "top_k_distinct_tables": 8,
            "macro_precision_recall_f2": macro(relevant_scores),
        },
        "breakdowns": {
            "retrieval": _breakdowns(
                rows, metrics_field="retrieval_metrics", family_field="query_family_current"
            ),
            "relevant_tables": _breakdowns(
                rows, metrics_field="relevant_tables_metrics", family_field="query_family_current"
            ),
        },
        "per_question": rows,
    }


def run_persisted_baseline(fixture_path: Path, predictions_dir: Path) -> dict:
    """Score the pre-fix graded artifacts using the later-selected top-eight policy.

    This intentionally preserves the persisted retrieval and linker order.  It is a comparison
    point for the diagnostic fixture, not a claim that those artifacts represent current code.
    """
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    records = [record for record in fixture if record["gold_n"] > 0]
    rows = []
    for record in records:
        prediction_path = predictions_dir / f"{record['id']}.json"
        prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
        retrieved = prediction.get("retrieved_tables", [])
        seen = []
        for operand in prediction.get("schema_linked_operands", []):
            key = operand["table_key"]
            if key not in seen:
                seen.append(key)
            if len(seen) == 8:
                break
        gold = set(record["gold_tables"])
        rows.append({
            "id": record["id"],
            "oracle_tickers": record["search_tickers"],
            "query_family_fixture_pre_fix": record["query_family"],
            "retrieval_metrics": score(set(retrieved), gold),
            "relevant_tables_metrics": score(set(seen), gold),
        })
    return {
        "provenance": "persisted graded predictions; pre-fix diagnostic baseline, not current code",
        "retrieval": {"macro_precision_recall_f2": macro([row["retrieval_metrics"] for row in rows])},
        "relevant_tables": {"macro_precision_recall_f2": macro(
            [row["relevant_tables_metrics"] for row in rows]
        )},
        "breakdowns": {
            "retrieval": _breakdowns(
                rows, metrics_field="retrieval_metrics", family_field="query_family_fixture_pre_fix"
            ),
            "relevant_tables": _breakdowns(
                rows, metrics_field="relevant_tables_metrics", family_field="query_family_fixture_pre_fix"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--companies", type=Path, default=DEFAULT_COMPANIES)
    parser.add_argument("--row-label-index", type=Path, default=DEFAULT_ROW_LABEL_INDEX)
    parser.add_argument("--structured-tables", type=Path, default=DEFAULT_STRUCTURED_TABLES)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--retrieval-top-k", type=int, default=10)
    parser.add_argument(
        "--statement-section-bonus", type=float, default=DEFAULT_STATEMENT_SECTION_BONUS,
    )
    parser.add_argument(
        "--related-party-note-penalty", type=float, default=DEFAULT_RELATED_PARTY_NOTE_PENALTY,
    )
    parser.add_argument("--predictions-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(
        fixture_path=args.fixture,
        catalog_path=args.catalog,
        companies_path=args.companies,
        row_label_index_path=args.row_label_index,
        structured_tables_path=args.structured_tables,
        rankings_path=args.rankings,
        retrieval_top_k=args.retrieval_top_k,
        statement_section_bonus=args.statement_section_bonus,
        related_party_note_penalty=args.related_party_note_penalty,
    )
    if args.predictions_dir:
        report["pre_fix_persisted_baseline"] = run_persisted_baseline(args.fixture, args.predictions_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "retrieval": report["retrieval"],
        "relevant_tables": report["relevant_tables"],
        "breakdowns": report["breakdowns"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

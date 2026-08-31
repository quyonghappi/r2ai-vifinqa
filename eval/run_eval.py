"""Reproducible Checkpoint-2 BM25 evaluation over the persisted retrieval catalog."""

import argparse
import json
from pathlib import Path

import pandas as pd

from eval.metrics import evaluate_retrieval
from retrieval.sparse import DEFAULT_STOPWORDS, build_enriched_document_text, build_index


def run(catalog_path: Path, companies_path: Path, dev_path: Path, output_dir: Path) -> dict:
    tables = pd.read_csv(
        catalog_path, usecols=["report_id", "line_position", "ticker", "year", "searchable_text"]
    )
    companies = pd.read_csv(companies_path)
    company_by_ticker = dict(zip(companies["Mã CK"], companies["Tên công ty"]))
    dev = [json.loads(line) for line in dev_path.read_text(encoding="utf-8").splitlines() if line]

    doc_ids = (tables["report_id"] + "|" + tables["line_position"].astype(str)).tolist()
    texts = [
        build_enriched_document_text(
            row.ticker,
            company_by_ticker.get(row.ticker, ""),
            row.year,
            row.searchable_text,
            identity_boost=5,
        )
        for row in tables.itertuples()
    ]
    index = build_index(doc_ids, texts, stopwords=DEFAULT_STOPWORDS)
    rankings = {str(q["id"]): index.search(q["question"], top_k=30) for q in dev}

    variants = {}
    for top_k in (5, 10, 20, 30):
        retrieved = {q["id"]: [x[0] for x in rankings[str(q["id"])][:top_k]] for q in dev}
        per_query, summary = evaluate_retrieval(dev, retrieved)
        variants[str(top_k)] = {
            "summary": summary,
            "per_query": [score.__dict__ for score in per_query],
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    rankings_path = output_dir / "retrieval_dev_v1_bm25_rankings.json"
    report_path = output_dir / "retrieval_dev_v1_bm25_report.json"
    rankings_path.write_text(json.dumps(rankings, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "dev_set": str(dev_path).replace("\\", "/"),
        "ground_truth_status": "self-constructed diagnostic set; not official labels",
        "matching": "exact report_id|line_position",
        "retriever": "BM25 enriched with ticker/company/year identity_boost=5 and DEFAULT_STOPWORDS",
        "approved_top_k": 10,
        "n_tables": len(tables),
        "variants": variants,
        "known_coverage_gaps": "See eval/dev_questions/README.md; only 14 questions, sparse coverage of ratios and multi-company cases.",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=Path("data/processed/normalized_tables.csv"))
    parser.add_argument("--companies", type=Path, default=Path("data/raw/hf_meta/code_stock.csv"))
    parser.add_argument("--dev", type=Path, default=Path("eval/dev_questions/dev_v1.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval/reports"))
    args = parser.parse_args()
    report = run(args.catalog, args.companies, args.dev, args.output_dir)
    print(json.dumps(report["variants"]["10"]["summary"], indent=2))


if __name__ == "__main__":
    main()

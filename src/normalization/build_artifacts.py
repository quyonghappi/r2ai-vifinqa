"""Build resumable full-corpus normalization artifacts for Checkpoint 2."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Optional

from extraction.parser import derive_report_id, diagnose_candidate_shape, extract_tables_from_file
from normalization.schema import RETRIEVAL_TEXT_VERSION, normalize_corpus, structured_record_json

CATALOG_FIELDS = [
    "table_key", "report_id", "ticker", "year", "variant", "part", "line_position",
    "page", "section_header", "table_identity", "searchable_text", "detected_units",
    "n_rows", "n_cols", "status", "source_path",
    "retrieval_text", "retrieval_text_version", "currency_hint", "header_depth", "content_hash",
]


def _load_company_by_ticker(companies_path: Optional[Path]) -> dict[str, str]:
    """ticker -> company name, for retrieval_text's COMPANY field. Optional: a missing or
    unset path degrades to ticker-only retrieval text rather than failing the build."""
    if companies_path is None or not companies_path.exists():
        return {}
    with companies_path.open(encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        next(reader, None)
        return {row[0]: row[1] for row in reader if len(row) >= 2}


def build(
    corpus_root: Path,
    processed_dir: Path,
    interim_dir: Path,
    *,
    force: bool = False,
    companies_path: Optional[Path] = None,
) -> dict:
    structured_path = processed_dir / "normalized_tables.jsonl"
    catalog_path = processed_dir / "normalized_tables.csv"
    anomaly_path = interim_dir / "extraction_full_anomalies.csv"
    summary_path = interim_dir / "checkpoint2_artifact_summary.json"
    outputs = (structured_path, catalog_path, anomaly_path, summary_path)
    if not force and all(path.exists() and path.stat().st_size > 0 for path in outputs):
        return json.loads(summary_path.read_text(encoding="utf-8"))

    report_paths = sorted((corpus_root / "financial_statements").glob("*/*/*/*_extracted.txt"))
    if not report_paths:
        raise FileNotFoundError(f"No report files under {corpus_root / 'financial_statements'}")
    company_by_ticker = _load_company_by_ticker(companies_path)
    processed_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)
    tmp_structured = structured_path.with_suffix(".jsonl.tmp")
    tmp_catalog = catalog_path.with_suffix(".csv.tmp")
    tmp_anomaly = anomaly_path.with_suffix(".csv.tmp")

    n_tables = n_anomalies = n_warn = n_fail = n_company_matched = 0
    with (
        tmp_structured.open("w", encoding="utf-8", newline="") as structured_file,
        tmp_catalog.open("w", encoding="utf-8", newline="") as catalog_file,
        tmp_anomaly.open("w", encoding="utf-8", newline="") as anomaly_file,
    ):
        catalog_writer = csv.DictWriter(catalog_file, fieldnames=CATALOG_FIELDS)
        catalog_writer.writeheader()
        anomaly_writer = csv.DictWriter(
            anomaly_file,
            fieldnames=["table_key", "report_id", "line_position", "source_path", "anomaly"],
        )
        anomaly_writer.writeheader()
        for path in report_paths:
            report_id = derive_report_id(str(path))
            source_path = path.relative_to(corpus_root).as_posix()
            candidates = extract_tables_from_file(
                str(path), report_id=report_id, provenance_source_path=source_path
            )
            normalized = normalize_corpus(candidates, company_by_ticker)
            for candidate, table in zip(candidates, normalized):
                table_key = f"{table.report_id}|{table.line_position}"
                structured_file.write(structured_record_json(table) + "\n")
                catalog_writer.writerow({
                    "table_key": table_key,
                    "report_id": table.report_id,
                    "ticker": table.ticker,
                    "year": table.year,
                    "variant": table.variant,
                    "part": table.part,
                    "line_position": table.line_position,
                    "page": table.page,
                    "section_header": table.section_header,
                    "table_identity": table.table_identity,
                    "searchable_text": table.searchable_text,
                    "detected_units": "|".join(table.detected_units),
                    "n_rows": table.n_rows,
                    "n_cols": table.n_cols,
                    "status": table.status,
                    "source_path": table.source_path,
                    "retrieval_text": table.retrieval_text,
                    "retrieval_text_version": table.retrieval_text_version,
                    "currency_hint": table.currency_hint or "",
                    "header_depth": table.header_depth,
                    "content_hash": table.content_hash,
                })
                for anomaly in diagnose_candidate_shape(candidate):
                    anomaly_writer.writerow({
                        "table_key": table_key,
                        "report_id": table.report_id,
                        "line_position": table.line_position,
                        "source_path": table.source_path,
                        "anomaly": anomaly,
                    })
                    n_anomalies += 1
                n_tables += 1
                n_warn += table.status == "parsed_with_warnings"
                n_fail += table.status == "failed"
                n_company_matched += bool(table.ticker in company_by_ticker)

    os.replace(tmp_structured, structured_path)
    os.replace(tmp_catalog, catalog_path)
    os.replace(tmp_anomaly, anomaly_path)
    summary = {
        "corpus_reports": len(report_paths),
        "normalized_tables": n_tables,
        "parse_warnings": n_warn,
        "parse_failures": n_fail,
        "shape_anomaly_flags": n_anomalies,
        "shape_thresholds": {"max_rows": 200, "max_cols": 50},
        "retrieval_text_version": RETRIEVAL_TEXT_VERSION,
        "company_names_resolved": n_company_matched,
        "companies_path": companies_path.as_posix() if companies_path else None,
        "structured_path": structured_path.as_posix(),
        "retrieval_catalog_path": catalog_path.as_posix(),
        "anomaly_path": anomaly_path.as_posix(),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_root", type=Path)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--interim-dir", type=Path, default=Path("data/interim"))
    parser.add_argument(
        "--companies-path", type=Path, default=Path("data/raw/hf_meta/code_stock.csv"),
        help="ticker->company-name CSV used to enrich retrieval_text; optional (default: repo's local copy)",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(
        build(
            args.corpus_root, args.processed_dir, args.interim_dir,
            force=args.force, companies_path=args.companies_path,
        ),
        indent=2,
    ))


if __name__ == "__main__":
    main()

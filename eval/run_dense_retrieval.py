"""Build a BGE-M3 dense ranking artifact using the shared, tested production entry point
(retrieval.dense.rank_questions_dense) -- this script only builds/caches the embedding index
and hands off scoping to that function, instead of re-implementing entity/variant/year
scoping a third time (see CHANGE_LOG.md retrieval-representation entry: two notebook cells and
this script each had their own copy before this).

Corpus text uses the same enriched representation as BM25 (retrieval.sparse
.build_enriched_document_text: ticker/company/year repeated identity_boost=5 times, plus full
cell text) -- not the earlier schema_only `retrieval_text` column. Dev-set diagnosis (see
CHANGE_LOG.md hybrid-retrieval-regression entry) found schema_only dense scores nearly flat
(~0.68-0.73 cosine, no separation) across a company's own same-schema tables across years/
variants, because schema_only strips the row-label/value content and per-identity-field
repetition that give BM25 its discriminative signal on this corpus. Requires the catalog to
carry `ticker`, `year`, and `searchable_text` -- rebuild normalization.build_artifacts.build()
first if the catalog predates those fields.

The model itself may come from a locally attached directory (`model_path`) or be downloaded
once from the Hugging Face Hub (`hf_repo_id`) via `huggingface_hub.snapshot_download` -- a
prep-phase weights fetch, not a runtime API call (see retrieval/dense.py's module docstring).
Exactly one of the two must be given.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from retrieval.dense import DENSE_MODEL_ID, DenseIndex, SentenceTransformerEncoder, build_dense_index, rank_questions_dense
from retrieval.sparse import build_enriched_document_text

_ENRICHED_REPRESENTATION_DESCRIPTION = (
    "enriched (ticker+company+year identity_boost=5 + searchable_text, matches BM25)"
)


def _questions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _resolve_model_path(
    model_path: Path | None, hf_repo_id: str | None, hf_revision: str, hf_cache_dir: Path | None,
) -> Path:
    if (model_path is None) == (hf_repo_id is None):
        raise ValueError("pass exactly one of model_path or hf_repo_id")
    if model_path is not None:
        return model_path
    from huggingface_hub import snapshot_download
    import os
    return Path(snapshot_download(
        repo_id=hf_repo_id, revision=hf_revision,
        cache_dir=str(hf_cache_dir) if hf_cache_dir is not None else None,
        token=os.environ.get("HF_TOKEN"),
    ))


def run(model_path: Path | None, catalog_path: Path, companies_path: Path, questions_path: Path,
        index_dir: Path, output_path: Path, *, top_k: int = 10, batch_size: int = 32,
        hf_repo_id: str | None = None, hf_revision: str = "main",
        hf_cache_dir: Path | None = None) -> dict[str, list[list]]:
    """Build (or reuse) the enriched-text dense index, then rank via rank_questions_dense."""
    required_columns = {"report_id", "line_position", "ticker", "year", "searchable_text"}
    catalog_columns = set(pd.read_csv(catalog_path, nrows=0).columns)
    missing_columns = required_columns - catalog_columns
    if missing_columns:
        raise ValueError(
            f"retrieval catalog is missing required columns: {sorted(missing_columns)} -- rebuild "
            "data/processed/normalized_tables.csv with the current normalization.build_artifacts "
            "before running dense retrieval"
        )
    resolved_model_path = _resolve_model_path(model_path, hf_repo_id, hf_revision, hf_cache_dir)
    tables = pd.read_csv(catalog_path, usecols=list(required_columns), keep_default_na=False)
    companies = pd.read_csv(companies_path, keep_default_na=False)
    company_by_ticker = dict(zip(companies.iloc[:, 0].astype(str), companies.iloc[:, 1].astype(str)))
    doc_ids = (tables["report_id"] + "|" + tables["line_position"].astype(str)).tolist()
    texts = [
        build_enriched_document_text(
            str(row.ticker), company_by_ticker.get(str(row.ticker), ""), row.year, str(row.searchable_text),
            identity_boost=5,
        )
        for row in tables.itertuples()
    ]
    encoder = SentenceTransformerEncoder(resolved_model_path)
    if (index_dir / "dense_embeddings.npy").exists() and (index_dir / "dense_doc_ids.json").exists():
        index = DenseIndex.load(index_dir)
        if index.doc_ids != doc_ids:
            raise ValueError("cached dense index does not match the retrieval catalog")
    else:
        index = build_dense_index(doc_ids, texts, encoder, batch_size=batch_size)
        index.save(index_dir)

    questions = _questions(questions_path)
    return rank_questions_dense(
        questions, catalog_path, companies_path, index, encoder, output_path,
        top_k=top_k, batch_size=batch_size,
        representation_description=_ENRICHED_REPRESENTATION_DESCRIPTION,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path, nargs="?", default=None,
                         help=f"attached local {DENSE_MODEL_ID} directory (omit to use --hf-repo-id instead)")
    parser.add_argument("--hf-repo-id", type=str, default=None,
                         help=f"download the model from Hugging Face Hub instead (e.g. {DENSE_MODEL_ID})")
    parser.add_argument("--hf-revision", type=str, default="main")
    parser.add_argument("--hf-cache-dir", type=Path, default=Path("data/processed/hf_dense_models"))
    parser.add_argument("--catalog", type=Path, default=Path("data/processed/normalized_tables.csv"))
    parser.add_argument("--companies", type=Path, default=Path("data/raw/hf_meta/code_stock.csv"))
    parser.add_argument("--questions", type=Path, default=Path("eval/dev_questions/dev_v1.jsonl"))
    parser.add_argument("--index-dir", type=Path, default=Path("data/processed/dense_bge_m3_enriched"))
    parser.add_argument("--output", type=Path, default=Path("reports/retrieval_dev_v1_dense_rankings.json"))
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(run(args.model_path, args.catalog, args.companies, args.questions, args.index_dir, args.output,
                         top_k=args.top_k, hf_repo_id=args.hf_repo_id, hf_revision=args.hf_revision,
                         hf_cache_dir=args.hf_cache_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()

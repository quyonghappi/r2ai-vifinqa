"""Resumable offline inference over every official ViFinQA test question."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from tqdm.auto import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT / "eval") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "eval"))

from common.table_store import load_tables_by_key
from pipeline import answer_question
from query_generation.generator import MODEL_ID, MODEL_REVISION, QwenAWQGenerator
from retrieval.full_corpus import rank_questions
from retrieval.hybrid import fuse_ranking_artifacts
try:
    # Kaggle snapshot layout: `eval` is pre-loaded as a proper package (pointing at the
    # snapshot's eval-code/ directory) before this module is ever imported -- confirmed working
    # by the notebook's own dense/hybrid acceptance-rule cell, which imports this same way.
    from eval.run_dense_retrieval import run as run_dense_retrieval
except ImportError:
    # Local-repo layout: `_REPO_ROOT / "eval"` is inserted directly onto sys.path above, making
    # its contents importable bare (no `eval.` package prefix) -- kept for local dev/tests.
    from run_dense_retrieval import run as run_dense_retrieval


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _is_complete_prediction(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        return (
            record.get("executed") is True
            and isinstance(record.get("answer"), (int, float))
            and not isinstance(record.get("answer"), bool)
            and math.isfinite(float(record["answer"]))
            and bool(record.get("evidence"))
            and bool(record.get("pandas_query"))
        )
    except (OSError, ValueError, TypeError):
        return False


def run(
    model_path: Path,
    questions_path: Path,
    catalog_path: Path,
    companies_path: Path,
    structured_path: Path,
    package_dir: Path,
    *,
    top_k: int = 10,
    max_retries: int = 2,
    dense_index_dir: Path | None = None,
    dense_model_path: Path | None = None,
    dense_hf_repo_id: str | None = None,
    dense_hf_revision: str = "main",
    dense_hf_cache_dir: Path | None = None,
    rrf_k: int = 60,
    row_label_index_path: Path | None = None,
) -> dict:
    questions = _jsonl(questions_path)
    ids = [q["id"] for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("official question ids are not unique")
    use_dense = dense_index_dir is not None
    if use_dense and dense_model_path is None and dense_hf_repo_id is None:
        raise ValueError("dense_index_dir requires dense_model_path or dense_hf_repo_id")

    work_dir = package_dir / "work"
    prediction_dir, failure_dir, evidence_dir = (
        work_dir / "predictions", work_dir / "failures", package_dir / "data"
    )
    prediction_dir.mkdir(parents=True, exist_ok=True)
    failure_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    bm25_path = work_dir / f"retrieval_test_top{top_k}.json"
    # Row-label reranking (retrieval.rerank, CHANGE_LOG.md 2026-08-31 Priority-2 entry) is
    # additive and optional: auto-detect the sidecar artifact next to the catalog rather than
    # requiring it, so a run against a catalog that hasn't had
    # `retrieval.rerank.build_row_label_index` run against it yet still produces the unchanged
    # pre-rerank BM25 ranking instead of failing.
    if row_label_index_path is None:
        default_row_label_index = catalog_path.parent / "row_label_index.csv"
        row_label_index_path = default_row_label_index if default_row_label_index.exists() else None
    rankings = rank_questions(
        questions, catalog_path, companies_path, bm25_path, top_k=top_k,
        row_label_index_path=row_label_index_path,
    )
    retriever_used = "bm25"
    if use_dense:
        dense_path = work_dir / f"retrieval_dense_top{top_k}.json"
        run_dense_retrieval(
            dense_model_path, catalog_path, companies_path, questions_path, dense_index_dir, dense_path,
            top_k=top_k, hf_repo_id=dense_hf_repo_id, hf_revision=dense_hf_revision, hf_cache_dir=dense_hf_cache_dir,
        )
        hybrid_path = work_dir / f"retrieval_hybrid_top{top_k}.json"
        rankings = fuse_ranking_artifacts(bm25_path, dense_path, hybrid_path, top_k=top_k, rrf_k=rrf_k)
        retriever_used = "hybrid_bm25_dense_rrf"
    retrieved = {q["id"]: [row[0] for row in rankings[str(q["id"])]] for q in questions}
    pending = [q for q in questions if not _is_complete_prediction(prediction_dir / f"{q['id']}.json")]
    if pending:
        needed = {key for q in pending for key in retrieved[q["id"]]}
        tables = load_tables_by_key(structured_path, needed)
        generator = QwenAWQGenerator(str(model_path))
        completed = len(questions) - len(pending)
        failures = 0
        with tqdm(
            total=len(questions),
            initial=completed,
            desc="Full inference",
            unit="question",
        ) as progress:
            for q in pending:
                try:
                    result = answer_question(
                        q, retrieved[q["id"]], tables, generator, max_retries=max_retries
                    )
                    frames = result.pop("evidence_frames")
                    if not result["executed"]:
                        raise RuntimeError(f"all generation/repair attempts failed: {result['attempts']}")
                    evidence = []
                    for variable, frame in frames.items():
                        name = f"q{q['id']}_{variable}.csv"
                        frame.to_csv(evidence_dir / name, index=False)
                        evidence.append({"variable": variable, "csv_path": f"data/{name}"})
                    result["evidence"] = evidence
                    result["relevant_tables"] = result["used_tables"]
                    _atomic_json(prediction_dir / f"{q['id']}.json", result)
                    completed += 1
                    failure_path = failure_dir / f"{q['id']}.json"
                    if failure_path.exists():
                        failure_path.unlink()
                except Exception as exc:
                    failures += 1
                    _atomic_json(failure_dir / f"{q['id']}.json", {
                        "id": q["id"], "question": q["question"],
                        "error_type": type(exc).__name__, "error_message": str(exc),
                    })
                finally:
                    progress.update(1)
                    progress.set_postfix(completed=f"{completed}/{len(questions)}", failures=failures)

    completed = sum(_is_complete_prediction(prediction_dir / f"{q['id']}.json") for q in questions)
    report = {
        "n_questions": len(questions), "n_completed": completed,
        "n_failed_or_pending": len(questions) - completed,
        "top_k": top_k, "max_retries": max_retries,
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "model_path": str(model_path), "offline": True,
        "retriever": retriever_used,
        "prediction_dir": prediction_dir.as_posix(), "evidence_dir": evidence_dir.as_posix(),
    }
    _atomic_json(work_dir / "full_inference_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--questions", type=Path, default=Path("data/raw/hf_meta/questions.jsonl"))
    parser.add_argument("--catalog", type=Path, default=Path("data/processed/normalized_tables.csv"))
    parser.add_argument("--companies", type=Path, default=Path("data/raw/hf_meta/code_stock.csv"))
    parser.add_argument("--structured", type=Path, default=Path("data/processed/normalized_tables.jsonl"))
    parser.add_argument("--package-dir", type=Path, default=Path("submission/package"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--dense-index-dir", type=Path, default=None,
                         help="opt-in: build/reuse a full-corpus dense index here and fuse it with BM25 via RRF")
    parser.add_argument("--dense-model-path", type=Path, default=None,
                         help="locally attached BGE-M3 directory (mutually exclusive with --dense-hf-repo-id)")
    parser.add_argument("--dense-hf-repo-id", type=str, default=None,
                         help="download the dense model from Hugging Face Hub instead, e.g. BAAI/bge-m3")
    parser.add_argument("--dense-hf-revision", type=str, default="main")
    parser.add_argument("--dense-hf-cache-dir", type=Path, default=Path("data/processed/hf_dense_models"))
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--row-label-index", type=Path, default=None,
                         help="retrieval.rerank.build_row_label_index output; auto-detected next "
                              "to --catalog as row_label_index.csv when omitted")
    args = parser.parse_args()
    print(json.dumps(run(args.model_path, args.questions, args.catalog, args.companies,
                         args.structured, args.package_dir, top_k=args.top_k,
                         max_retries=args.max_retries, dense_index_dir=args.dense_index_dir,
                         dense_model_path=args.dense_model_path, dense_hf_repo_id=args.dense_hf_repo_id,
                         dense_hf_revision=args.dense_hf_revision, dense_hf_cache_dir=args.dense_hf_cache_dir,
                         rrf_k=args.rrf_k, row_label_index_path=args.row_label_index), indent=2))


if __name__ == "__main__":
    main()

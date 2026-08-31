"""Optional dense retrieval for Vietnamese financial tables.

The encoder is a pipeline component. Chosen candidate: ``BAAI/bge-m3`` (open-weight, MIT,
released 2024-02, 568M params; well below the competition's 14B limit) -- see CHANGE_LOG.md
retrieval-representation entry for why the earlier ``AITeamVN/Vietnamese_Embedding_v2``
alternative was dropped.

This module itself never calls the network: ``SentenceTransformerEncoder`` always loads from
a local path (``local_files_only=True``). Resolving that path -- whether from a locally
attached model directory or a one-time Hugging Face Hub download -- is the caller's job (see
``eval/run_dense_retrieval.py``), matching this project's existing prep-phase-only internet
convention (AGENTS.md Section 1: no runtime API calls inside ``src/``; a Hub download is a
one-shot weights fetch, not a per-inference call, same as the corpus download itself).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

import numpy as np
import pandas as pd

from retrieval.full_corpus import (
    _atomic_json,
    infer_question_tickers,
    infer_question_variant,
    infer_question_years,
)

DENSE_MODEL_ID = "BAAI/bge-m3"
DENSE_MODEL_ELIGIBILITY = {
    "category": "pipeline component",
    "license": "MIT",
    "released": "2024-02",
    "parameters": "568M",
    "offline_only": True,
}


class TextEncoder(Protocol):
    def encode(self, texts: list[str], *, batch_size: int) -> np.ndarray: ...


class SentenceTransformerEncoder:
    """Lazy adapter so local deterministic tests do not require model dependencies."""

    def __init__(self, model_path: str | Path, *, max_seq_length: int = 1536):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on Kaggle runtime image
            raise RuntimeError(
                "dense retrieval requires sentence-transformers in the offline runtime image"
            ) from exc
        # fp16 on CUDA only: encode-time precision, not a different model or weights file.
        # Retrieval ranks by cosine similarity after L2-normalization, where fp16's ~1e-3
        # relative rounding error is far below the score gaps that separate top-k candidates,
        # so this is a throughput optimization with negligible ranking risk (CHANGE_LOG.md
        # 2026-08-30 dense-indexing-speed entry).
        model_kwargs = {}
        try:
            import torch
            use_cuda = torch.cuda.is_available()
        except ImportError:  # pragma: no cover - depends on Kaggle runtime image
            use_cuda = False
        if use_cuda:
            model_kwargs["torch_dtype"] = torch.float16
        self._model = SentenceTransformer(
            str(model_path), local_files_only=True,
            device="cuda" if use_cuda else None, model_kwargs=model_kwargs or None,
        )
        # Real-corpus token EDA under the actual BGE-M3 tokenizer (CHANGE_LOG.md
        # 2026-08-30 retrieval-representation entry): p99=646, max=1,372 tokens on
        # schema_only retrieval_text -- 1536 has zero truncation and stays far under
        # BGE-M3's native 8,192 limit.
        self._model.max_seq_length = max_seq_length

    def encode(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        # The full corpus is embedded in a single encode() call (build_dense_index's docstring
        # explains why), which can take minutes on 140K+ table texts -- sentence-transformers'
        # own progress bar is the only visibility into that call, so it stays on rather than
        # looking like a hang.
        return np.asarray(self._model.encode(
            texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=True,
        ), dtype=np.float32)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("dense embeddings must be a two-dimensional matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("dense embeddings must not contain zero vectors")
    return values / norms


@dataclass(frozen=True)
class DenseIndex:
    doc_ids: list[str]
    embeddings: np.ndarray

    def __post_init__(self) -> None:
        if len(self.doc_ids) != len(self.embeddings):
            raise ValueError("doc_ids and dense embeddings must have the same length")
        object.__setattr__(self, "embeddings", _normalize_rows(self.embeddings))

    def search(self, query_embedding: np.ndarray, *, top_k: int, allowed_positions: list[int] | None = None) -> list[tuple[str, float]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query = _normalize_rows(np.asarray(query_embedding, dtype=np.float32).reshape(1, -1))[0]
        positions = np.asarray(allowed_positions if allowed_positions is not None else range(len(self.doc_ids)))
        if not len(positions):
            return []
        scores = self.embeddings[positions] @ query
        ordered = sorted(zip(positions.tolist(), scores.tolist()), key=lambda item: (-item[1], self.doc_ids[item[0]]))
        return [(self.doc_ids[position], float(score)) for position, score in ordered[:top_k]]

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "dense_embeddings.npy", self.embeddings)
        (directory / "dense_doc_ids.json").write_text(json.dumps(self.doc_ids, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> "DenseIndex":
        directory = Path(directory)
        return cls(
            json.loads((directory / "dense_doc_ids.json").read_text(encoding="utf-8")),
            np.load(directory / "dense_embeddings.npy"),
        )


def build_dense_index(doc_ids: list[str], texts: list[str], encoder: TextEncoder, *, batch_size: int = 32) -> DenseIndex:
    if len(doc_ids) != len(texts):
        raise ValueError("doc_ids and texts must have the same length")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not texts:
        return DenseIndex([], np.empty((0, 1), dtype=np.float32))
    # A single encode() call over the whole corpus -- rather than pre-slicing into fixed
    # batches ourselves -- lets sentence-transformers length-sort inputs before internal
    # batching, which cuts padding waste on this corpus's highly variable schema-text
    # lengths (CHANGE_LOG.md 2026-08-30 dense-indexing-speed entry).
    return DenseIndex(doc_ids, np.asarray(encoder.encode(texts, batch_size=batch_size), dtype=np.float32))


def rank_questions_dense(
    questions: list[dict],
    catalog_path: str | Path,
    companies_path: str | Path,
    dense_index: DenseIndex,
    encoder: TextEncoder,
    output_path: str | Path,
    *,
    top_k: int = 10,
    batch_size: int = 32,
    representation_description: str = "schema_only representation",
) -> dict[str, list[list]]:
    """Production dense ranking, scoped exactly like retrieval.full_corpus.rank_questions.

    `representation_description` documents what text `dense_index` was actually built from
    (e.g. "schema_only representation" or "enriched (ticker+company+year identity_boost=5 +
    searchable_text, matches BM25)") -- this function has no way to inspect that after the
    fact, so the caller must state it truthfully; it is recorded verbatim in the output
    metadata sidecar so a persisted ranking artifact never silently misdescribes itself.

    This is the single reproducible entry point for dense retrieval -- it reuses the same
    infer_question_tickers/years/variant scope resolution BM25 uses (not a re-implementation),
    so evaluation and shipped retrieval share one scoping policy. `dense_index` may cover the
    full corpus or a smaller evaluation slice; either way, query-time search is restricted to
    the resolved entity's positions via `allowed_positions`, never to a pre-filtered index --
    scope is a query-time restriction over one shared index, not a separate index per company.
    An unresolved ticker falls back to an unrestricted search over `dense_index`, mirroring
    BM25's global fallback unchanged (see CHANGE_LOG.md for why that fallback is left as-is
    this sprint). A resolved-but-empty scope after year/variant filtering returns an empty
    ranking -- a real, inspectable retrieval failure, not a cross-issuer guess.
    """
    output_path = Path(output_path)
    catalog = pd.read_csv(
        catalog_path, usecols=["report_id", "line_position", "ticker", "year", "variant"],
        keep_default_na=False,
    )
    if "variant" not in catalog:
        catalog["variant"] = "unspecified"
    companies = pd.read_csv(companies_path, keep_default_na=False)
    company_by_ticker = dict(zip(companies.iloc[:, 0].astype(str), companies.iloc[:, 1].astype(str)))

    meta_by_key = {
        f"{row.report_id}|{row.line_position}": (str(row.ticker), row.year, str(row.variant))
        for row in catalog.itertuples()
    }
    positions_by_ticker: dict[str, list[int]] = {}
    year_by_position: dict[int, object] = {}
    variant_by_position: dict[int, str] = {}
    for position, doc_id in enumerate(dense_index.doc_ids):
        ticker, year, variant = meta_by_key.get(doc_id, ("", None, "unspecified"))
        positions_by_ticker.setdefault(ticker, []).append(position)
        year_by_position[position] = year
        variant_by_position[position] = variant

    def scope(question: str) -> Optional[list[int]]:
        tickers = infer_question_tickers(question, company_by_ticker)
        if not tickers:
            return None  # unrestricted search -- matches BM25's no-ticker fallback
        years = infer_question_years(question)
        variant = infer_question_variant(question)
        positions = [p for t in tickers for p in positions_by_ticker.get(t, [])]
        if len(years) == 1:
            positions = [p for p in positions if year_by_position[p] is not None and int(year_by_position[p]) in years]
        if variant:
            positions = [p for p in positions if variant_by_position[p] == variant]
        return positions

    query_texts = [q["question"] for q in questions]
    query_embeddings = encoder.encode(query_texts, batch_size=batch_size)
    rankings: dict[str, list[list]] = {}
    for question, query_embedding in zip(questions, query_embeddings):
        allowed = scope(question["question"])
        if allowed is not None and not allowed:
            rankings[str(question["id"])] = []
            continue
        rankings[str(question["id"])] = [
            [key, score] for key, score in dense_index.search(query_embedding, top_k=top_k, allowed_positions=allowed)
        ]

    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    _atomic_json(output_path, rankings)
    _atomic_json(metadata_path, {
        "top_k": top_k,
        "model_id": DENSE_MODEL_ID,
        "dense_index_size": len(dense_index.doc_ids),
        "retriever": f"BGE-M3 dense, {representation_description}, entity/variant/single-year scoped (mirrors BM25 scope)",
    })
    return rankings


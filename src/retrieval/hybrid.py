"""Deterministic rank fusion for independently persisted sparse and dense rankings."""

from __future__ import annotations

import json
from pathlib import Path


def reciprocal_rank_fusion(rankings: list[list[tuple[str, float]]], *, top_k: int, rrf_k: int = 60) -> list[tuple[str, float]]:
    """Fuse ranked lists with deterministic reciprocal-rank fusion (RRF).

    Kept as a tested, available primitive but no longer `fuse_ranking_artifacts`'s default
    strategy -- see `backfill_fusion` and its docstring for why (CHANGE_LOG.md 2026-08-31
    hybrid-fusion-backfill entry)."""
    if top_k <= 0 or rrf_k < 0:
        raise ValueError("top_k must be positive and rrf_k must be non-negative")
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (table_key, _score) in enumerate(ranking, 1):
            scores[table_key] = scores.get(table_key, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]


def backfill_fusion(
    primary: list[tuple], secondary: list[tuple], *, max_backfill: int,
) -> list[tuple]:
    """Keep `primary`'s ranking exactly as-is (order and scores untouched), then append up to
    `max_backfill` candidates from `secondary` that `primary` did not already find, in
    `secondary`'s own rank order.

    Replaces symmetric RRF as `fuse_ranking_artifacts`'s default (CHANGE_LOG.md 2026-08-31
    hybrid-fusion-backfill entry). Confirmed on a real Kaggle run (production catalog, not the
    local dev sample): symmetric RRF let dense's un-improved ranking pull sparse's already-
    correct candidates (post row-label-rerank/decomposition/per-company-assembly, none of which
    dense has yet) out of the fused top-k on exactly the highest-value cases -- a mediocre-in-
    both-signals candidate can out-score a candidate that is excellent in one signal and weak in
    the other, a known RRF property that only became a problem once the two signals' quality
    diverged this much. Backfill cannot do that: a candidate primary already ranked highly keeps
    that exact position regardless of where secondary ranks it, so this can only ADD coverage
    dense uniquely found, never subtract or reorder what sparse already got right -- the same
    non-destructive/additive principle already used by retrieval.rerank and
    retrieval.decompose."""
    if max_backfill < 0:
        raise ValueError("max_backfill must be non-negative")
    seen = {key for key, _ in primary}
    fused = list(primary)
    for key, score in secondary:
        if len(fused) - len(primary) >= max_backfill:
            break
        if key in seen:
            continue
        fused.append((key, score))
        seen.add(key)
    return fused


def fuse_ranking_artifacts(sparse_path: str | Path, dense_path: str | Path, output_path: str | Path, *, top_k: int = 10, rrf_k: int = 60) -> dict[str, list[list]]:
    """Fuse compatible per-question artifacts into the existing downstream ranking contract.

    Uses `backfill_fusion` (sparse as primary, dense backfilling up to `top_k` additional unique
    candidates) rather than symmetric RRF -- see that function's docstring for the evidence.
    `rrf_k` is accepted for call-site compatibility (`submission/run_full_inference.py`'s
    `--rrf-k` CLI flag, `reciprocal_rank_fusion` callers) but has no effect on this strategy.
    """
    sparse = json.loads(Path(sparse_path).read_text(encoding="utf-8"))
    dense = json.loads(Path(dense_path).read_text(encoding="utf-8"))
    if set(sparse) != set(dense):
        raise ValueError("sparse and dense artifacts must cover identical question IDs")
    fused = {
        question_id: [
            [key, score] for key, score in backfill_fusion(
                [(key, score) for key, score in sparse[question_id]],
                [(key, score) for key, score in dense[question_id]],
                max_backfill=top_k,
            )
        ]
        for question_id in sorted(sparse, key=lambda value: int(value))
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(fused, ensure_ascii=False, indent=2), encoding="utf-8")
    return fused


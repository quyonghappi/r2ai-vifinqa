"""Retrieval evaluation metrics: Precision, Recall, F2 (macro-averaged), per docs/eval.md.

    Precision = mean over queries of (correct retrieved / retrieved)
    Recall    = mean over queries of (correct retrieved / relevant)
    F2        = 5*P*R / (4*P + R)   -- recall weighted 4:1 over precision (CONTEXT.md §4)

`docs/eval.md` does not specify whether "correct retrieved table" is judged by document ID,
table ID, or exact `relevant_tables` (report_id|line_position) match -- CONTEXT.md explicitly
flags this as unresolved and says metric code should be adjustable rather than hardcoding an
assumption. `key_fn` on `precision_recall_f2_for_query` is that adjustment point: pass
`key_fn=lambda s: s.split("|")[0]` to compare at report_id (document) granularity instead of
the exact-line default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable


def _default_key(item: str) -> str:
    return item


@dataclass
class QueryScore:
    query_id: object
    precision: float
    recall: float
    f2: float
    n_retrieved: int
    n_relevant: int
    n_correct: int


def precision_recall_f2_for_query(
    retrieved: Iterable[str],
    relevant: Iterable[str],
    key_fn: Callable[[str], str] = _default_key,
) -> tuple[float, float, float]:
    """Precision/Recall/F2 for one query. Empty `retrieved` -> precision 0 (not undefined) to
    stay macro-averageable; empty `relevant` -> recall 0 by the same reasoning (a dev-set
    construction bug, not something that should silently vanish from the average)."""
    retrieved_keys = {key_fn(r) for r in retrieved}
    relevant_keys = {key_fn(r) for r in relevant}
    n_correct = len(retrieved_keys & relevant_keys)

    precision = n_correct / len(retrieved_keys) if retrieved_keys else 0.0
    recall = n_correct / len(relevant_keys) if relevant_keys else 0.0
    if precision + recall == 0:
        f2 = 0.0
    else:
        f2 = (5 * precision * recall) / (4 * precision + recall)
    return precision, recall, f2


def evaluate_retrieval(
    queries: list[dict],
    retrieved_by_id: dict,
    key_fn: Callable[[str], str] = _default_key,
) -> tuple[list[QueryScore], dict]:
    """queries: list of {"id": ..., "relevant_tables": [...]}.
    retrieved_by_id: {query_id: [retrieved table-key strings, ranked or not]}.
    Returns (per-query scores, macro-averaged summary dict)."""
    scores = []
    for q in queries:
        qid = q["id"]
        relevant = q["relevant_tables"]
        retrieved = retrieved_by_id.get(qid, [])
        p, r, f2 = precision_recall_f2_for_query(retrieved, relevant, key_fn=key_fn)
        retrieved_keys = {key_fn(x) for x in retrieved}
        relevant_keys = {key_fn(x) for x in relevant}
        scores.append(
            QueryScore(
                query_id=qid,
                precision=p,
                recall=r,
                f2=f2,
                n_retrieved=len(retrieved_keys),
                n_relevant=len(relevant_keys),
                n_correct=len(retrieved_keys & relevant_keys),
            )
        )

    n = len(scores)
    summary = {
        "n_queries": n,
        "precision_macro": sum(s.precision for s in scores) / n if n else 0.0,
        "recall_macro": sum(s.recall for s in scores) / n if n else 0.0,
        "f2_macro": sum(s.f2 for s in scores) / n if n else 0.0,
    }
    return scores, summary

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


def mrr_at_k(
    retrieved: Iterable[str],
    relevant: Iterable[str],
    k: int = 5,
    key_fn: Callable[[str], str] = _default_key,
) -> float:
    """Reciprocal rank of the first correct item within the top `k` of `retrieved`, else 0.

    `retrieved`'s existing order is treated as rank -- an assumption, not a confirmed fact:
    `docs/eval.md` does not describe TABLES_MRR5/DOCS_MRR5 at all (they only appear in the real
    graded `scoring_result/scores.txt`), so whether the official grader ranks by submission list
    order is unverified. Report this function's output labeled as such, not as a settled metric.
    """
    relevant_keys = {key_fn(r) for r in relevant}
    for rank, item in enumerate(list(retrieved)[:k], start=1):
        if key_fn(item) in relevant_keys:
            return 1.0 / rank
    return 0.0


def evaluate_mrr(
    queries: list[dict],
    retrieved_by_id: dict,
    k: int = 5,
    key_fn: Callable[[str], str] = _default_key,
) -> tuple[list[float], float]:
    """Same query/retrieved_by_id shape as `evaluate_retrieval`. Returns (per-query MRR, macro)."""
    scores = []
    for q in queries:
        qid = q["id"]
        relevant = q["relevant_tables"]
        retrieved = retrieved_by_id.get(qid, [])
        scores.append(mrr_at_k(retrieved, relevant, k=k, key_fn=key_fn))
    n = len(scores)
    return scores, (sum(scores) / n if n else 0.0)


def _doc_key(item: str) -> str:
    return item.split("|")[0]


def evaluate_retrieval_full(queries: list[dict], retrieved_by_id: dict, mrr_k: int = 5) -> dict:
    """TABLES_* and DOCS_* precision/recall/F2/MRR in one call, keyed exactly like the real
    graded `scoring_result/scores.txt` -- so a local run's output can be diffed against an
    official score sheet field-for-field. DOCS_* is TABLES_* at document (`report_id`)
    granularity via `_doc_key`; no new precision/recall/F2 logic, just the existing
    `evaluate_retrieval`/`evaluate_mrr` called twice with different `key_fn`s.
    """
    _, tables_summary = evaluate_retrieval(queries, retrieved_by_id)
    _, docs_summary = evaluate_retrieval(queries, retrieved_by_id, key_fn=_doc_key)
    _, tables_mrr = evaluate_mrr(queries, retrieved_by_id, k=mrr_k)
    _, docs_mrr = evaluate_mrr(queries, retrieved_by_id, k=mrr_k, key_fn=_doc_key)
    return {
        "n_queries": tables_summary["n_queries"],
        "TABLES_PRECISION": tables_summary["precision_macro"],
        "TABLES_RECALL": tables_summary["recall_macro"],
        "TABLES_F2MACRO": tables_summary["f2_macro"],
        f"TABLES_MRR{mrr_k}": tables_mrr,
        "DOCS_PRECISION": docs_summary["precision_macro"],
        "DOCS_RECALL": docs_summary["recall_macro"],
        "DOCS_F2MACRO": docs_summary["f2_macro"],
        f"DOCS_MRR{mrr_k}": docs_mrr,
        "mrr_rank_semantics_assumption": (
            "MRR treats retrieved-list order as rank; docs/eval.md does not describe MRR at "
            "all, so this is unconfirmed against the official grader."
        ),
    }


def project_zero_credit_macro(summary: dict, total_questions: int) -> dict:
    """Rescale a scored-subset macro summary (from `evaluate_retrieval`) as if every question
    outside the scored subset scored zero.

    `docs/submission_guide.md` states missing/malformed questions count as invalid predictions
    (zero credit), so the official denominator is the full official question count -- not just
    the subset a partial run or a diagnostic fixture happened to score. Every local macro number
    computed over a subset (e.g. a partial Kaggle run's 940/1012 completions, or this fixture's
    own smaller N) should be reported alongside this projection, not only the scored-subset
    number, or it will read optimistic relative to what the real grader would report.
    """
    n_scored = summary["n_queries"]
    if total_questions < n_scored:
        raise ValueError("total_questions must be >= the number of scored queries")
    scale = (n_scored / total_questions) if total_questions else 0.0
    return {
        "n_scored": n_scored,
        "n_total": total_questions,
        "precision_macro_zero_credit": summary["precision_macro"] * scale,
        "recall_macro_zero_credit": summary["recall_macro"] * scale,
        "f2_macro_zero_credit": summary["f2_macro"] * scale,
    }

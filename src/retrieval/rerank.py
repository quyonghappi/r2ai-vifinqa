"""retrieval stage: within-scope reranking by normalized row-label (line-item) match.

Scope boundary (AGENTS.md Section 2): this is an additive scoring signal, not a filter -- it
never removes a candidate BM25/dense already scored, only reorders/blends scores before the
caller's existing top_k truncation. Motivated by the dev_v1 README's confirmed finding (and its
full-corpus generalization, CHANGE_LOG.md 2026-08-31): for the ~755/1,012 real questions that
resolve to exactly one ticker, the entity-scoped candidate pool still has a median of ~1,484
same-company tables (up to 2,581) to rank among, where whole-table BM25 term overlap struggles
to discriminate because financial line items repeat near-verbatim across years/statement types.
A table's row labels (e.g. "Doanh thu thuần", "Lợi nhuận sau thuế") are financial statements'
most discriminative lexical signal for a specific line-item lookup -- stronger than document-
level term frequency -- so this module scores each candidate by its best-matching row label
against the question's residual (non-identity) tokens and blends that into the existing score.
"""
from __future__ import annotations

import json
from pathlib import Path

from normalization.schema import _MAX_LABEL_CHARS, _bounded, _dedupe_preserve_order
from retrieval.sparse import tokenize

MAX_ROW_LABELS_FOR_RERANK = 200
_ROW_LABEL_SEPARATOR = " | "


def row_label_text_from_record(record: dict) -> str:
    """Compact, bounded, deduplicated row-label text for one normalized table record.

    Deliberately excludes numeric cell values, captions, and the bulk `grid` -- keeping this
    representation small enough to persist as a full-corpus sidecar artifact without repeating
    the memory pressure the dense-indexing sprint hit loading the full 667MB JSONL at once
    (CHANGE_LOG.md 2026-08-30 entries note this machine runs with <1GB free RAM).
    """
    labels = _dedupe_preserve_order(
        str(r.get("label", "")) for r in record.get("row_labels") or []
    )[:MAX_ROW_LABELS_FOR_RERANK]
    return _ROW_LABEL_SEPARATOR.join(_bounded(label, _MAX_LABEL_CHARS) for label in labels)


def build_row_label_index(jsonl_path: str | Path, output_path: str | Path) -> int:
    """Stream `normalized_tables.jsonl` once and persist a compact `table_key,row_label_text`
    CSV sidecar. Streaming (one record in memory at a time, immediate write) rather than
    loading the full JSONL, matching the memory constraint documented above. Returns row count.
    """
    import csv

    jsonl_path = Path(jsonl_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with jsonl_path.open("r", encoding="utf-8") as src, \
         output_path.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.writer(dst)
        writer.writerow(["table_key", "row_label_text"])
        for line in src:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            table_key = record.get("table_key") or f"{record['report_id']}|{record['line_position']}"
            writer.writerow([table_key, row_label_text_from_record(record)])
            n += 1
    return n


def row_label_overlap_score(question_tokens: frozenset, row_label_text: str, stopwords: frozenset) -> float:
    """Best single-row-label Jaccard overlap with the question's (stopword-filtered) tokens.

    Two corrections over an earlier, evidence-rejected version (CHANGE_LOG.md 2026-08-31
    Priority-2 entry) that scored the *whole table's* labels as one bag: (1) scoring per
    individual label and taking the max, not pooling every label's tokens together, since
    pooling let large tables (more distinct row labels = more accidental overlap surface, the
    exact bug the earlier docstring here incorrectly claimed was already avoided) out-score a
    small table holding the one exactly-right label; (2) Jaccard (intersection/union) instead of
    a query-only recall ratio, so a label with irrelevant extra words (e.g. a note's "Chi phí
    khác bằng tiền" next to a P&L's exact "Chi phí khác") no longer ties or beats an exact match
    just because it happens to contain the same core tokens -- the extra tokens now count against
    it. Confirmed on real dev data: this fixed the false win the recall-only bag-of-labels
    version gave the wrong table for Q9 (`Chi phí khác của SAM năm 2023`) while preserving the
    genuine same-entity-pool recovery it found for Q584.

    A third variant (overlap coefficient: intersection/min-length instead of intersection/union)
    was tried and REJECTED during the huge-pool audit (CHANGE_LOG.md 2026-08-31 huge-pool-audit
    entry) -- not kept here. It was motivated by real evidence (Jaccard scores an exact match as
    low as 0.04-0.17 against a 20+-token compound question purely because the question is long),
    but empirically it regressed dev_v1 F2 at the already-calibrated weight (0.3699 -> 0.3268 at
    weight=24) without fixing either target case (Q405, Q591 stayed at 0 recall even at
    weight=150): overlap coefficient is too permissive for short 2-3-token labels in long
    questions, inflating many unrelated candidates to a tied 1.0 and diluting rather than
    concentrating the signal. Kept as Jaccard; see that entry for the confirmed actual root
    causes of Q405/Q591, which are not a reranking-scoring-formula problem."""
    if not question_tokens or not row_label_text:
        return 0.0
    best = 0.0
    for label in row_label_text.split(_ROW_LABEL_SEPARATOR):
        label_tokens = set(tokenize(label, stopwords=stopwords))
        if not label_tokens:
            continue
        union = question_tokens | label_tokens
        if not union:
            continue
        score = len(question_tokens & label_tokens) / len(union)
        if score > best:
            best = score
    return best


def rerank_with_row_labels(
    ranked: list[tuple],
    question_tokens: frozenset,
    row_label_text_by_doc_id: dict[str, str],
    stopwords: frozenset,
    *,
    weight: float,
) -> list[tuple]:
    """Blend each candidate's existing (doc_id, score) with its row-label overlap score and
    re-sort. Additive, not multiplicative or a filter: a candidate with zero row-label overlap
    keeps exactly its original BM25 score and position relative to other zero-overlap
    candidates, so this can never cause BM25 to lose a candidate it already found -- only
    reorder within what was already retrieved (F2's recall weighting per CONTEXT.md Section 4)."""
    rescored = []
    for doc_id, score in ranked:
        row_label_text = row_label_text_by_doc_id.get(doc_id, "")
        bonus = weight * row_label_overlap_score(question_tokens, row_label_text, stopwords)
        rescored.append((doc_id, score + bonus))
    rescored.sort(key=lambda item: (-item[1], str(item[0])))
    return rescored

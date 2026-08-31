"""retrieval stage: question -> candidate table set, sparse (BM25) implementation.

Scope boundary (AGENTS.md Section 2): returns candidate tables with a score; does not
silently drop candidates below a hardcoded threshold -- `search()` returns every scored
candidate, and how many to keep (top_k) is an explicit, inspectable, tunable caller parameter,
not baked into this module. F2's recall-weighting (CONTEXT.md Section 4) means the natural lever
for tuning is top_k, and that choice belongs to whoever is calling this, informed by dev-set
measurement -- not a default buried here.

Not a "model" in the CONTEXT.md Section 3 eligibility sense: BM25 is a deterministic classical
IR ranking formula with no learned/pretrained weights, so the open-weight/size/release-date
constraints on pipeline models do not apply to it. That constraint becomes relevant only if/when
a dense embedding model is added (see the candidate list presented alongside this in Checkpoint
2 -- not implemented yet, pending approval).

Uses an inverted index (term -> posting list of doc indices) rather than scoring every document
for every query. At corpus scale (146K+ tables in the real corpus, see Checkpoint 2), a naive
"score every doc" search is impractically slow in pure Python; a query only needs to touch the
(much smaller) set of documents that share at least one term with it.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Measured necessity, not a guess: on the full 146K-table corpus (Checkpoint 2 dev-set
# evaluation), plain BM25 over enriched (ticker+company+year+cell-text) tables scored
# precision_macro=recall_macro=0.0 on all 14 dev questions at top_k=10. High-frequency
# Vietnamese function/boilerplate words ("là", "của", "công ty", "năm"...) appear in a large
# fraction of all 146K documents and, summed across a ~19-token natural-language question,
# out-accumulate the 1-2 truly distinctive terms (ticker, company name) that appear only once
# in a correct document's short text. Removing them raised recall_macro to 0.21 in isolation;
# combined with `identity_boost` below, to 0.46. This is standard IR practice (stopword
# removal), not a dev-set-specific hack -- but the list itself IS specific to this corpus's
# observed vocabulary (see CHANGE_LOG.md for the actual before/after numbers) and is a
# parameter precisely so it can be revised, not silently baked into `tokenize`.
DEFAULT_STOPWORDS = frozenset(
    """
    là của và các những này đó cho để với trong trên dưới khi đã sẽ có bao nhiêu
    công ty một được tại về từ đến theo như sau trước hay hoặc thì mà nào gì
    năm ngày tháng đơn vị tính vnd đồng cổ phần
    """.split()
)


def tokenize(text: str, stopwords: frozenset = frozenset()) -> list[str]:
    r"""Lowercase Unicode word tokens. `\w` is Unicode-aware for str input in Python 3, so
    Vietnamese diacritic letters are kept intact (not stripped as "non-word" characters).
    `stopwords` defaults to empty (no filtering) -- callers needing the corpus-tuned default
    pass `stopwords=DEFAULT_STOPWORDS` explicitly; see module docstring above for why this
    isn't silently always-on."""
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in stopwords] if stopwords else tokens


def build_enriched_document_text(
    ticker: str, company_name: str, year, searchable_text: str, identity_boost: int = 5
) -> str:
    """Compose a table's indexed text from its structured identity (ticker/company/year,
    already available from normalization) plus its cell text, repeating the identity fields
    `identity_boost` times.

    Why repetition, not just concatenation once: a table's cell text can run to hundreds of
    tokens (large tables), so a ticker/company name appearing once has negligible term
    frequency next to that volume. Repeating it lets its term frequency compete -- measured
    necessity (see DEFAULT_STOPWORDS docstring): without this, recall_macro on the Checkpoint 2
    dev set was 0.21 (stopwords filtered, no boost); with 5x repetition, 0.46.
    """
    identity = f"{ticker} {company_name} {year} " * identity_boost
    return identity + searchable_text


@dataclass
class BM25Index:
    doc_ids: list  # opaque identifiers, parallel to the corpus passed to build()
    doc_len: list  # token count per document
    doc_freqs: list[dict]  # per-doc term -> count
    postings: dict  # term -> list[doc_idx] (documents containing that term at all)
    df: dict  # term -> number of docs containing it
    avgdl: float
    n_docs: int
    k1: float = 1.5
    b: float = 0.75
    # stored (not re-passed at search time) so a query is always tokenized the same way its
    # documents were indexed -- passing mismatched stopwords between build_index() and search()
    # would silently break every postings lookup.
    stopwords: frozenset = field(default_factory=frozenset)

    def idf(self, term: str) -> float:
        n_t = self.df.get(term, 0)
        # BM25+-style idf (add 1 inside the log) keeps every idf non-negative, which matters
        # here because table "documents" are short (a caption + a handful of cells) and common
        # terms can appear in a large fraction of a 100K+-table corpus.
        return math.log(1.0 + (self.n_docs - n_t + 0.5) / (n_t + 0.5))

    def _score_doc(self, query_tokens: list[str], doc_idx: int) -> float:
        freqs = self.doc_freqs[doc_idx]
        dl = self.doc_len[doc_idx]
        total = 0.0
        for t in query_tokens:
            f = freqs.get(t, 0)
            if f == 0:
                continue
            num = f * (self.k1 + 1)
            den = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            total += self.idf(t) * (num / den)
        return total

    def search(self, query: str, top_k: int | None = None) -> list[tuple]:
        """Returns (doc_id, score) pairs sorted by descending score. If top_k is None, returns
        every document with a nonzero score (no candidate silently dropped) -- which, for the
        inverted-index implementation, is exactly the set of documents sharing at least one
        query term."""
        query_tokens = tokenize(query, stopwords=self.stopwords)
        candidate_doc_idxs: set = set()
        for t in query_tokens:
            candidate_doc_idxs.update(self.postings.get(t, ()))

        scored = [(self.doc_ids[i], self._score_doc(query_tokens, i)) for i in candidate_doc_idxs]
        # Explicit secondary key makes tied BM25 scores reproducible across Python hash seeds.
        scored.sort(key=lambda x: (-x[1], str(x[0])))
        return scored[:top_k] if top_k is not None else scored


def build_index(
    doc_ids: list,
    texts: list[str],
    k1: float = 1.5,
    b: float = 0.75,
    stopwords: frozenset = frozenset(),
) -> BM25Index:
    if len(doc_ids) != len(texts):
        raise ValueError("doc_ids and texts must be the same length")

    doc_len = []
    doc_freqs = []
    postings: dict = {}
    df: dict = {}
    total_len = 0

    for doc_idx, text in enumerate(texts):
        tokens = tokenize(text, stopwords=stopwords)
        total_len += len(tokens)
        doc_len.append(len(tokens))
        counts: dict = {}
        for tok in tokens:
            counts[tok] = counts.get(tok, 0) + 1
        doc_freqs.append(counts)
        for tok in counts:
            df[tok] = df.get(tok, 0) + 1
            postings.setdefault(tok, []).append(doc_idx)

    n_docs = len(texts)
    avgdl = (total_len / n_docs) if n_docs else 0.0
    return BM25Index(
        doc_ids=doc_ids,
        doc_len=doc_len,
        doc_freqs=doc_freqs,
        postings=postings,
        df=df,
        avgdl=avgdl,
        n_docs=n_docs,
        k1=k1,
        b=b,
        stopwords=stopwords,
    )

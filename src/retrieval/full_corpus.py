"""Persist deterministic BM25 rankings for an arbitrary question set."""

from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path

import pandas as pd

from retrieval.decompose import MAX_CANDIDATES_PER_TICKER, concept_top_m, extract_concept_queries
from retrieval.rerank import rerank_with_row_labels
from retrieval.sparse import DEFAULT_STOPWORDS, build_enriched_document_text, build_index, tokenize

# Reranking blend weight for row-label overlap (retrieval.rerank.row_label_overlap_score, a
# [0, 1] per-label Jaccard score) against raw BM25 score. Measured, not guessed (CHANGE_LOG.md
# 2026-08-31 row-label-rerank entry): swept 0-96 against the production scoped retriever on
# dev_v1 (18 questions). F2 rose monotonically with weight (0.3088 at 0 -> 0.3592 at 16 ->
# 0.3699 at 24) and was flat from 24 through 96 (the bounded [0,1] signal has already saturated
# the sort order by then), with zero regressions on any previously-correct question at any
# weight tested. 24.0 is the smallest weight reaching that plateau. It is a module constant (not
# buried default) so it stays inspectable and re-tunable the same way `identity_boost` is in
# retrieval.sparse.
ROW_LABEL_RERANK_WEIGHT = 24.0


# Shared across retrieval/schema_linking/query_generation (see their imports of this constant)
# so a ticker-token fix lands once, not as three independently-drifting copies.
# Confirmed defect (2026-08-30 question-profiling EDA, data/raw/hf_meta/code_stock.csv): 2 of
# the 100 real tickers, HT1 and PC1, contain a trailing digit. A letters-only class can't match
# "HT1" here because \w treats the digit and letters as one token, so \b never falls between
# "T" and "1" -- the old `[A-Z]{2,5}` pattern silently failed to resolve either ticker whenever
# a question used the bare code without the full company name (confirmed via
# eval/dev_questions/ profiling of all 1,012 real questions: IDs 7, 276 for HT1; 58, 283 for
# PC1 fell through to the unresolved/global-fallback path for exactly this reason). The
# lookahead requires at least one letter so a bare 4-digit year ("2023") still cannot match.
TICKER_TOKEN_RE = re.compile(r"\b(?=[A-Z0-9]*[A-Z])[A-Z0-9]{2,5}\b")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_NON_TICKER_CODES = frozenset({"BCTC", "CTCP", "TMCP", "TCT", "TNHH", "VND"})
_COMPANY_NOISE = frozenset({"công", "ty", "cổ", "phần", "tổng", "ngân", "hàng"})


def infer_question_tickers(question: str, company_by_ticker: dict[str, str]) -> set[str]:
    """Resolve explicit tickers or distinctive company-name aliases before retrieval.

    A literal ticker mention is the trusted baseline and is never silently dropped by a full-
    company-name overlap. Confirmed defect (2026-08-31 full-1,012-question profiling,
    eval/reports/question_profile_summary.json and the accompanying bug audit): the previous
    version let ANY strong (>=3 token) name overlap override literal tickers outright, including
    when 2+ unrelated companies tied for that score purely on generic sector vocabulary (e.g.
    "bất động sản đầu tư" ties 5 different real-estate companies at overlap=3, hijacking away
    from a literal, unambiguous ticker like IJC elsewhere in the same question -- confirmed on
    55/1,012 real questions). A tie is definitionally weak evidence of *which* company is meant,
    so it must never outrank an explicit ticker; it remains the fallback signal (unchanged) only
    when no literal ticker exists at all.

    A single (non-tied) name-match winner is allowed to replace a literal ticker only when that
    literal ticker's own token is itself one of the winning company's name tokens -- i.e. the
    literal substring is nested inside a more specific company name being described (``Công ty
    Chứng khoán FPT`` contains token "fpt" but the listed ticker is FTS; "CTCP Viễn thông FPT" is
    ticker FOX). This is the same real-world mention being re-identified, not a second company.
    Verified 5/5 on real cases (FPT->FTS, FPT->FOX). A single winner that is NOT nested this way
    is left alone when a literal ticker already exists -- manual audit of the 30 real cases where
    this occurs showed roughly a third are genuine nested re-identification, a third are spurious
    unrelated collisions (a counterparty/investee name coincidentally overlapping), and a third
    are additional group members named in full alongside literal tickers; with no reliable way to
    tell those apart from token overlap alone, silently adding the winner risks polluting scope
    with a wrong company more often than it helps, so the conservative choice is to keep the
    already-unambiguous literal ticker and accept that full-name-only group members beyond a
    literal anchor remain a known, unsolved limitation (not claimed fixed here).
    """
    question_tokens = set(tokenize(question, stopwords=DEFAULT_STOPWORDS))
    literal_tickers = {
        token for token in TICKER_TOKEN_RE.findall(question)
        if token not in _NON_TICKER_CODES and token in company_by_ticker
    }
    name_scores = {}
    name_tokens_by_ticker = {}
    for ticker, company_name in company_by_ticker.items():
        name_tokens = set(tokenize(company_name, stopwords=DEFAULT_STOPWORDS)) - _COMPANY_NOISE
        overlap = len(question_tokens & name_tokens)
        if overlap >= 2:
            name_scores[ticker] = overlap
            name_tokens_by_ticker[ticker] = name_tokens
    # Two matching name tokens are common in Vietnamese financial questions. They are not enough
    # to override a literal listed ticker. Retain ties rather than guessing among companies.
    strong_name_scores = {ticker: score for ticker, score in name_scores.items() if score >= 3}
    if strong_name_scores:
        best = max(strong_name_scores.values())
        winners = {ticker for ticker, score in strong_name_scores.items() if score == best}
        if not literal_tickers:
            return winners
        if len(winners) == 1:
            winner = next(iter(winners))
            nested_literal = {
                t for t in literal_tickers if t.lower() in name_tokens_by_ticker[winner]
            }
            if nested_literal:
                return (literal_tickers - nested_literal) | {winner}
        return literal_tickers
    return literal_tickers


def infer_question_years(question: str) -> set[int]:
    """Return every explicitly requested reporting year, without guessing one."""
    return {int(year) for year in _YEAR_RE.findall(question)}


def infer_question_variant(question: str) -> str | None:
    """Return an explicitly requested report variant, if the question names one.

    An omitted variant is deliberately left unscoped: treating it as ``consolidated``
    would turn an unverified convention into a retrieval decision.
    """
    normalized = question.lower()
    if any(marker in normalized for marker in (
        "công ty mẹ", "công ty riêng", "bctc riêng", "báo cáo tài chính riêng",
    )):
        return "separate"
    if any(marker in normalized for marker in ("hợp nhất", "tập đoàn", "bctc hợp nhất")):
        return "consolidated"
    return None


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def rank_questions(
    questions: list[dict],
    catalog_path: str | Path,
    companies_path: str | Path,
    output_path: str | Path,
    *,
    top_k: int = 10,
    row_label_index_path: str | Path | None = None,
    row_label_rerank_weight: float = ROW_LABEL_RERANK_WEIGHT,
) -> dict[str, list[list]]:
    """Build the approved enriched index once and persist top-k scored table keys.

    A complete, matching artifact is reused; a stale/partial artifact is rebuilt. The caller
    chooses ``top_k`` explicitly so retrieval never hides a threshold inside the stage.

    ``row_label_index_path`` (retrieval.rerank.build_row_label_index's output: a
    ``table_key,row_label_text`` CSV sidecar) is optional and defaults to off -- callers/tests
    that don't pass it get exactly the pre-rerank BM25 ranking, unchanged. When given, every
    candidate a scope's BM25 search finds is rescored by blending in its row-label overlap with
    the question before truncating to ``top_k`` (retrieval.rerank.rerank_with_row_labels): this
    can reorder candidates BM25 already found, never drop one it found, matching F2's recall
    weighting (CONTEXT.md Section 4).
    """
    output_path = Path(output_path)
    expected_ids = {str(q["id"]) for q in questions}
    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    question_digest = hashlib.sha256(
        json.dumps(questions, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    expected_metadata = {
        "top_k": top_k,
        "question_sha256": question_digest,
        "catalog_bytes": Path(catalog_path).stat().st_size,
        "companies_bytes": Path(companies_path).stat().st_size,
        "row_label_index_bytes": Path(row_label_index_path).stat().st_size if row_label_index_path else None,
        "row_label_rerank_weight": row_label_rerank_weight if row_label_index_path else None,
        "retriever": "BM25 enriched identity_boost=5 DEFAULT_STOPWORDS entity_variant_scoped=true single_year_scoped=true concept_decomposition=true",
    }
    if output_path.exists() and metadata_path.exists():
        saved = json.loads(output_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata == expected_metadata and set(saved) == expected_ids:
            return saved

    row_label_text_by_doc_id: dict[str, str] = {}
    if row_label_index_path is not None:
        row_label_frame = pd.read_csv(row_label_index_path, keep_default_na=False)
        row_label_text_by_doc_id = dict(
            zip(row_label_frame["table_key"].astype(str), row_label_frame["row_label_text"].astype(str))
        )

    required_columns = ["report_id", "line_position", "ticker", "year", "searchable_text"]
    catalog_columns = set(pd.read_csv(catalog_path, nrows=0).columns)
    missing_columns = set(required_columns) - catalog_columns
    if missing_columns:
        raise ValueError(f"retrieval catalog is missing required columns: {sorted(missing_columns)}")
    usecols = required_columns + (["variant"] if "variant" in catalog_columns else [])
    tables = pd.read_csv(catalog_path, usecols=usecols, keep_default_na=False)
    if "variant" not in tables:
        # Pre-v6 artifacts are valid for questions without an explicit variant.
        tables["variant"] = "unspecified"
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
    index = build_index(doc_ids, texts, stopwords=DEFAULT_STOPWORDS)
    doc_positions_by_ticker: dict[str, list[int]] = {}
    for position, ticker in enumerate(tables["ticker"].astype(str)):
        doc_positions_by_ticker.setdefault(ticker, []).append(position)
    scoped_indexes = {}

    def rerank(question: str, ranked: list[tuple]) -> list[tuple]:
        # Always truncates to top_k here, whether or not reranking actually ran -- callers pass
        # top_k=None into search() so this is the single place scope-wide candidate lists get
        # cut down, regardless of whether row_label_text_by_doc_id is populated.
        if not row_label_text_by_doc_id or not ranked:
            return ranked[:top_k]
        question_tokens = frozenset(tokenize(question, stopwords=DEFAULT_STOPWORDS))
        return rerank_with_row_labels(
            ranked, question_tokens, row_label_text_by_doc_id, DEFAULT_STOPWORDS,
            weight=row_label_rerank_weight,
        )[:top_k]

    def ticker_scoped_index(ticker: str, years: frozenset, variant: str | None):
        # Keyed per single ticker (not per resolved ticker tuple) so a group question's
        # per-company search can reuse the same sub-index another question already built for
        # that company, and so each company gets its own scope built strictly from its own
        # tables -- never diluted by another company's candidate pool.
        scope = (ticker, tuple(sorted(years)), variant)
        if scope in scoped_indexes:
            return scoped_indexes[scope]
        positions = list(doc_positions_by_ticker.get(ticker, []))
        # A question spanning multiple years can need one later report whose comparative
        # column contains an adjacent year. Only a single named year is safe to apply as
        # catalog-metadata scope.
        if len(years) == 1:
            positions = [position for position in positions if int(tables.iloc[position].year) in years]
        if variant:
            positions = [position for position in positions if str(tables.iloc[position].variant) == variant]
        if not positions:
            # A resolved issuer must never fall back to another issuer's number.
            # An empty ranking is an inspectable retrieval failure for downstream code.
            scoped_indexes[scope] = None
            return None
        scoped_indexes[scope] = build_index(
            [doc_ids[position] for position in positions],
            [texts[position] for position in positions],
            stopwords=DEFAULT_STOPWORDS,
        )
        return scoped_indexes[scope]

    def search(question: str) -> list[tuple]:
        tickers = tuple(sorted(infer_question_tickers(question, company_by_ticker)))
        years = frozenset(infer_question_years(question))
        variant = infer_question_variant(question)
        # Rerank against every candidate BM25 already scored within scope (top_k=None), not just
        # its pre-truncated top_k -- reordering only within an already-truncated top_k could
        # never recover the "correct table ranked outside top_k in a huge same-entity pool" cases
        # this signal exists for (CHANGE_LOG.md dev_v1 README finding). The unresolved-ticker
        # fallback searches the full 146K-table corpus, where an unbounded candidate list can be
        # tens of thousands of docs for a common term; a wide-but-bounded window keeps reranking
        # cheap there while still reaching well past top_k.
        if not tickers:
            return rerank(question, index.search(question, top_k=max(top_k * 20, 200)))
        # Retrieve independently per resolved company and assemble the union, rather than
        # pooling every company into one shared index and taking one flat top_k -- a group-
        # screening question naming 5-7 companies (a real, non-rare shape in the full 1,012-
        # question profile, e.g. IDs 362-436) would otherwise starve most of them under a single
        # top_k budget shared across the whole group. No fixed company/year/table count is
        # assumed: each company gets its own top_k against its own scope, and the number of
        # companies in `tickers` can be anything infer_question_tickers resolved. Per-company
        # results aren't score-comparable across companies (each has its own IDF space from its
        # own scoped index), so they are concatenated in deterministic ticker order rather than
        # interleaved by a cross-company score ranking that would misrepresent comparability.
        concept_queries = extract_concept_queries(question)
        top_m = concept_top_m(len(tickers))
        assembled: list[tuple] = []
        for ticker in tickers:
            scoped_index = ticker_scoped_index(ticker, years, variant)
            if scoped_index is None:
                continue
            primary = rerank(question, scoped_index.search(question, top_k=None))
            per_ticker: list[tuple] = list(primary)
            # Concept-level query decomposition (CHANGE_LOG.md 2026-08-31
            # query-decomposition-design entry): a compound question needing several distinct
            # financial concepts at once (e.g. equity + profit + revenue + assets for a
            # ROE-and-turnover question) structurally starves the losing concepts under one flat
            # query -- confirmed, not fixed by any amount of rerank-weight tuning, in the prior
            # huge-pool audit. Additive top-up only: never replaces or truncates the primary
            # result above, only adds candidates the primary search didn't already surface, and
            # only when 2+ distinct concepts are actually detected (a single-concept question is
            # already exactly what the primary whole-question search is).
            if len(concept_queries) >= 2:
                covered = {doc_id for doc_id, _ in per_ticker}
                for concept_query in concept_queries:
                    if len(per_ticker) >= MAX_CANDIDATES_PER_TICKER:
                        break
                    concept_ranked = rerank(concept_query, scoped_index.search(concept_query, top_k=None))
                    for doc_id, score in concept_ranked[:top_m]:
                        if doc_id in covered:
                            continue
                        if len(per_ticker) >= MAX_CANDIDATES_PER_TICKER:
                            break
                        per_ticker.append((doc_id, score))
                        covered.add(doc_id)
            assembled.extend(per_ticker)
        return assembled

    rankings = {str(q["id"]): [[key, score] for key, score in search(q["question"])] for q in questions}
    _atomic_json(output_path, rankings)
    _atomic_json(metadata_path, expected_metadata)
    return rankings

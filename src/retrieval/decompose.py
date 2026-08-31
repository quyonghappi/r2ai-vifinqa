"""retrieval stage: concept-level query decomposition within an already entity/year/variant-
scoped candidate pool.

Scope boundary (AGENTS.md Section 2): purely a query-expansion aid for retrieval's own search
step -- it decides what ADDITIONAL search strings to run against a pool retrieval already
correctly scoped, never which table is "the" right one (that stays schema_linking's job) and
never removes a candidate the primary whole-question search already found (additive only,
same non-destructive pattern as retrieval.rerank).

Motivated by the confirmed huge-pool + multi-concept audit finding (CHANGE_LOG.md 2026-08-31):
one BM25/rerank query over a compound question structurally favors whichever concept's wording
dominates the sentence, starving other needed concepts (e.g. Q405 needs equity + profit + revenue
+ assets at once; no single query weighting fixed this). Query decomposition runs one additional,
independent, targeted search per detected financial concept within the SAME already-built scoped
index (no extra index construction), and adds a small top-up of results not already found by the
primary search.

Design-adjustment audit (2026-08-31, before this module was implemented) validated the concept
lexicon and its trigger/budget against the real 1,012-question corpus and the real scoped BM25
index before any of this was wired in -- see CHANGE_LOG.md for the full dry-run methodology,
distribution, and sampled false positive/negative findings. Two concrete corrections came out of
that audit and are encoded here, not assumed: (1) the lexicon is built from question-corpus
n-gram frequency, not row-label frequency, because row-label frequency is blind to every named
ratio/derived metric (they never appear as literal row labels) and biases toward table
boilerplate irrelevant to what questions actually ask; (2) concept detection must count DISTINCT
post-expansion, deduplicated intents, not raw literal phrase occurrences, or a single ratio
mention (e.g. "ROE") would be undercounted as "1 concept" when it needs 2 operand searches, and a
phrase mentioned twice in one question would be overcounted as 2.
"""
from __future__ import annotations

import unicodedata

from common.financial_concepts import RATIO_OPERAND_EXPANSIONS, literal_phrase_matches


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).lower()


def extract_concept_queries(question: str) -> list[str]:
    """Return distinct additional concept query strings, deduplicated, excluding the original
    whole-question query. Fewer than 2 results means decomposition should not fire for this
    question -- reduces to a no-op, the same safety property retrieval.full_corpus.search()
    already relies on for its per-ticker fan-out (Priority 3)."""
    q = _normalize(question)
    concepts: list[str] = []

    for trigger, operands in RATIO_OPERAND_EXPANSIONS.items():
        if trigger in q:
            concepts.extend(operands)

    for phrase in literal_phrase_matches(q):
        if phrase not in concepts:
            concepts.append(phrase)

    seen: list[str] = []
    for c in concepts:
        if c not in seen:
            seen.append(c)
    return seen


def concept_top_m(n_tickers: int, *, base_top_m: int = 10, floor: int = 2) -> int:
    """Per-concept top-up budget, scaled down as the resolved company count grows.

    Measured, not guessed (CHANGE_LOG.md 2026-08-31 query-decomposition-design and ablation
    entries): real per-concept overlap measured against the actual scoped BM25 index showed the
    dominant explosion risk is multi-company x multi-concept combined (Priority 3 already fans
    out per company; decomposition compounds on top of that), not single-company concept fan-out
    alone, which stayed bounded (13-51 total candidates across the sampled range at top_m 5-10).
    `base_top_m` was raised from an initial 5 to 10 after the ablation traced a real miss: Q405's
    equity/profit concepts ranked 6th/10th and 9th/10th within their own targeted sub-query
    (vs. 142nd-1802nd under the flat whole-question query -- confirming the core mechanism, a
    20-100x rank improvement) but fell just outside a top_m=5 cutoff. Raising to 10 costs nothing
    against `MAX_CANDIDATES_PER_TICKER`, which already bounds the true worst case regardless of
    this constant. This scales top_m down as company count rises so a many-company
    group-screening question doesn't multiply an already-per-company budget by an unscaled
    per-concept one.
    """
    if n_tickers <= 1:
        return base_top_m
    return max(floor, 15 // n_tickers)


# Measured ceiling (CHANGE_LOG.md 2026-08-31 query-decomposition-design entry): the observed
# single-company worst case in the calibration sample (7 concepts, top_m=5) reached 30 distinct
# candidates. A hard cap here is a second, independent guardrail on top of concept_top_m's
# scaling -- inspectable and re-tunable the same way top_k/identity_boost/
# ROW_LABEL_RERANK_WEIGHT already are, not buried.
MAX_CANDIDATES_PER_TICKER = 30

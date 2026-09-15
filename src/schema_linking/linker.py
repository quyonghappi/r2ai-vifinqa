"""Ground Vietnamese questions to row/column operands in retrieved structured tables."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict, deque
from dataclasses import dataclass

from common.financial_concepts import RATIO_OPERAND_EXPANSIONS
from retrieval.decompose import extract_concept_queries
from retrieval.full_corpus import TICKER_TOKEN_RE, infer_question_variant, infer_question_years

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_STOPWORDS = frozenset("là của và các những bao nhiêu công ty năm trong tại vào cuối đầu".split())
_UNIT_RE = re.compile(r"(nghìn\s*tỷ|tỷ|triệu|nghìn)?\s*(?:đồng|vnd|%)", re.I)


_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_NON_TICKER_ACRONYMS = frozenset({"BCTC", "CP", "CTCP", "TMCP", "TCT", "TNHH", "VND"})


def parse_financial_number(text: str) -> float | None:
    value = text.strip().replace("\u00a0", " ")
    if not value or value in {"-", "–", "—"}:
        return None
    negative = value.startswith("(") and value.endswith(")")
    value = re.sub(r"[^\d,.-]", "", value.strip("()"))
    if not re.search(r"\d", value):
        return None
    if re.fullmatch(r"-?\d{1,3}(?:\.\d{3})+", value):
        value = value.replace(".", "")
    elif re.fullmatch(r"-?\d{1,3}(?:,\d{3})+", value):
        value = value.replace(",", "")
    elif "," in value and "." in value:
        if value.rfind(",") > value.rfind("."):
            value = value.replace(".", "").replace(",", ".")
        else:
            value = value.replace(",", "")
    else:
        value = value.replace(",", ".")
    try:
        number = float(value)
    except ValueError:
        return None
    return -abs(number) if negative else number


def _tokens(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFC", text).lower()
    return {x for x in _TOKEN_RE.findall(normalized) if x not in _STOPWORDS}


def _expanded_question_tokens(question: str) -> set[str]:
    """Shares its ratio->operand-phrase table with retrieval's concept-decomposition query
    expansion (common.financial_concepts.RATIO_OPERAND_EXPANSIONS) so the two stages never drift
    apart -- the same fix already applied once in this codebase for TICKER_TOKEN_RE."""
    q = question.lower()
    expansions = []
    for trigger, operands in RATIO_OPERAND_EXPANSIONS.items():
        if trigger in q:
            expansions.extend(operands)
    return _tokens(question + " " + " ".join(expansions))


def classify_query_family(question: str, available_tickers: frozenset[str] = frozenset()) -> str:
    """Classify a question's shape to pick the generation strategy (deterministic vs. LLM).

    ``available_tickers`` must be the set of real tickers actually present among the tables
    retrieval scoped for this question (``link_schema``'s ``tables`` argument already carries a
    ``ticker`` field per table). Confirmed defect (2026-09-01 diagnostic audit,
    predictions/*.json): before this parameter existed, any 2-5 char uppercase token matching
    ``TICKER_TOKEN_RE`` was counted as a "second company" with no check that it was a real
    ticker at all -- generic Vietnamese financial acronyms (TNDN, TCTD, USD, FVTPL, EPS, GELEX,
    CFO, LNST, ...) routinely tripped this, measured at 27/52 (52%) of real
    ``cross_company_comparison``-classified successful predictions actually being single-company
    questions. Misclassifying a plain lookup this way sends it through the error-prone LLM
    ``generate_plan`` path instead of the reliable, deterministic ``generate_direct_lookup_plan``
    path for no reason. Requiring ``token in available_tickers`` (mirroring
    ``retrieval.full_corpus.infer_question_tickers``'s own ``token in company_by_ticker`` check)
    closes this: a token only counts as a second company if it is a ticker retrieval actually
    resolved as relevant to this question.
    """
    q = question.lower()
    if any(x in q for x in ("roe", "roa", "tỷ lệ", "tỷ suất", "trên tài sản")):
        return "ratio_or_derived"
    if any(x in q for x in ("bao nhiêu phần trăm", "tăng trưởng", "thay đổi bao nhiêu %")):
        return "growth"
    if any(x in q for x in ("tổng cộng", "tổng số", "trung bình", "median", "cộng lại")):
        return "aggregation"
    tickers = [
        token for token in TICKER_TOKEN_RE.findall(question)
        if token not in _NON_TICKER_ACRONYMS and token in available_tickers
    ]
    if len(set(tickers)) > 1:
        return "cross_company_comparison"
    if any(x in q for x in ("chênh lệch", "biến động", "cao hơn", "thấp hơn", "so với")):
        return "comparison"
    return "direct_lookup"


# Smallest K reaching the macro-F2 plateau on eval/dev_questions/tables_relevance_oracle_v1.json
# (2026-09-01 diagnostic audit, eval/reports/relevant_tables_k_calibration.py): macro F2 rose
# from 0.280 (K=5) to 0.303 (K=8) and stayed flat (0.302-0.303) through K=20, while precision
# kept falling (0.430 at K=8 -> 0.316 at K=20) -- the same "smallest weight/K reaching the
# plateau" selection rule already used for ROW_LABEL_RERANK_WEIGHT. A module constant, not a
# buried default, so it stays inspectable and re-tunable as the oracle set grows.
RELEVANT_TABLES_TOP_K = 8


def select_relevant_tables(operands: "list[OperandCandidate]", top_k: int = RELEVANT_TABLES_TOP_K) -> list[str]:
    """Return up to ``top_k`` distinct table keys, ranked by schema-linking's own relevance score.

    This is the ``relevant_tables`` the submission reports for the TABLES_F2MACRO metric -- a
    recall-oriented prediction of which tables hold needed evidence, per submission_guide.md's
    own definition ("relevant" = contains part or all of the data needed to compute the
    answer). It is deliberately NOT ``used_tables`` (the literal operand tables the final
    generated ``pandas_query`` happens to reference): a diagnostic audit (2026-09-01,
    predictions/*.json from the graded run) found ``used_tables`` collapses to exactly 1 table
    for 758/789 (96%) of successful predictions regardless of how many tables a question
    genuinely needs, and measured this narrowing as the single largest recall loss in the whole
    pipeline (oracle macro F2 0.130 for ``used_tables`` vs. 0.303 at this function's K=8). It is
    also deliberately NOT the raw retrieval candidate list (``retrieved_tables``): that scores
    lower still (macro F2 0.290) because it carries no relevance ranking at all.

    ``operands`` is ``SchemaLinkResult.operands`` (or any prefix of it), already sorted by
    ``link_schema``'s two-tiered key: year/variant-conflict tier first, then
    ``-relevance_score`` within a tier (see ``year_conflicts_with_question``/
    ``variant_conflicts_with_question``). Distinct tables are kept in that sorted order, so the
    first ``top_k`` distinct table keys are exactly the top-K tables by that ordering. Execution
    provenance (``used_tables``/``evidence``) is untouched by this function and continues to
    reflect only what ``pandas_query`` actually indexed.
    """
    seen: list[str] = []
    for operand in operands:
        if operand.table_key not in seen:
            seen.append(operand.table_key)
        if len(seen) >= top_k:
            break
    return seen


@dataclass(frozen=True)
class OperandCandidate:
    table_key: str
    report_id: str
    source_path: str
    line_position: int
    row_index: int
    column_index: int
    row_label: str
    column_header: str
    raw_value: str
    numeric_value: float
    detected_units: list[str]
    relevance_score: float


@dataclass
class SchemaLinkResult:
    question: str
    query_family: str
    requested_unit: str | None
    table_keys: list[str]
    operands: list[OperandCandidate]
    # Canonical entity scope for this question, resolved once here (not re-derived downstream).
    # Confirmed defect (EXECUTION_ACCURACY_AUDIT.md §6.1, real graded-run case, official id 4):
    # `query_generation.generator.generate_direct_lookup_plan`/`validate_plan` used to re-derive
    # "which company does this question mean" from a bare `TICKER_TOKEN_RE` scan of the raw
    # question text, missing nested-brand mentions (e.g. "CTCP Chứng khoán FPT" means ticker FTS,
    # not FPT) that `retrieval.full_corpus.infer_question_tickers` already resolves correctly.
    # When the literal text token disagreed with the only retrieved company, the intersection-based
    # filter went empty and silently skipped company scoping entirely, letting an unrelated
    # company's table through. Computed once, here, from the same resolution retrieval trusts.
    resolved_tickers: frozenset[str] = frozenset()

    def prompt_payload(self) -> dict:
        return {
            "question": self.question,
            "query_family": self.query_family,
            "requested_unit": self.requested_unit,
            "allowed_operands": [
                {
                    "table_key": x.table_key,
                    "row_index": x.row_index,
                    "column_index": x.column_index,
                    "row_label": _prompt_text(x.row_label, 240),
                    "column_header": _prompt_text(x.column_header, 160),
                    "raw_value": _prompt_text(x.raw_value, 80),
                }
                for x in self.operands
            ],
        }


def _prompt_text(value: str, limit: int) -> str:
    """Bound OCR fields so one malformed cell cannot consume the prompt budget."""
    return value if len(value) <= limit else value[:limit - 1] + "…"


def _column_headers(table: dict) -> list[str]:
    """Use the actual header row, never numeric data rows, as column context."""
    grid = table["grid"]
    if not grid:
        return [""] * table["n_cols"]
    return [
        grid[0][column].strip()[:160] if column < len(grid[0]) else ""
        for column in range(table["n_cols"])
    ]


def _fold_header_text(value: str) -> str:
    """Normalize OCR header spelling before classifying a column's semantic role."""
    decomposed = unicodedata.normalize("NFD", value.lower())
    return "".join(
        char for char in decomposed if not unicodedata.combining(char) and char.isalnum()
    ).replace(chr(0x0111), "d")


def is_numeric_data_column(table: dict, column_index: int, column_header: str) -> bool:
    """Return whether a column can supply a financial numeric operand.

    This is intentionally a column-role check, not a test of whether one cell parses as a
    number: note and code columns contain numbers too. Exact role matching avoids rejecting a
    legitimate period header that happens to mention a note.
    """
    if column_index <= 0:
        return False
    metadata = next(
        (
            item for item in (table.get("column_metadata") or [])
            if item.get("column_index") == column_index
        ),
        {},
    )
    header_path = str(metadata.get("header_path") or column_header)
    role = _fold_header_text(header_path)
    return role not in {
        "maso", "ma", "code", "thuyetminh", "thuyetminhso", "note", "notes",
        "ghichu", "diengiai", "bienong", "phantram", "chitieu", "nguonvon",
    }


def year_conflicts_with_question(table: dict, column_header: str, question_years: set[int]) -> bool:
    """True only when the question names explicit year(s) and neither the table's own year nor
    this specific operand's column header (a comparative-period column in an otherwise
    differently-dated report) matches any of them.

    Absent an explicit year in the question, nothing can conflict -- this must never turn into
    an implicit "assume latest year" filter (AGENTS.md Section 2: schema_linking represents,
    it does not add undocumented filtering judgment calls).
    """
    if not question_years:
        return False
    table_year = table.get("year")
    named = {str(year) for year in question_years}
    if table_year is not None and str(table_year) in named:
        return False
    if any(str(year) in column_header for year in question_years):
        return False
    return True


def variant_conflicts_with_question(table: dict, question_variant: str | None) -> bool:
    """True only when the question names an explicit report variant and the table's own variant
    is a definite, known mismatch. Missing/``"unspecified"`` variant metadata is never a
    conflict -- an absent signal must stay a no-op, not be treated as evidence against a table
    (mirrors ``infer_question_variant``'s own "an omitted variant is deliberately left
    unscoped" rule)."""
    if not question_variant:
        return False
    table_variant = table.get("variant")
    return table_variant not in (None, "", "unspecified", question_variant)


def _interleave_tables(
    scored: list[tuple[int, OperandCandidate]], concept_queries: list[str],
) -> list[OperandCandidate]:
    """Preserve score/tier ranking while preventing one table's repeated cells from using a budget.

    ``link_schema`` scores individual numeric cells, whereas both the evidence contract and the
    generator need coverage across tables.  For a multi-concept question, tables that contain
    more of its explicitly extracted concepts lead the round-robin; ties retain the existing
    score order. Within every table, candidates retain their score order. Thus partial/duplicate
    one-concept tables cannot exhaust the 120-candidate linker budget (or the generator's
    32-operand prefix) before a table that covers another required condition gets representation.
    Conflict tiers are never mixed: explicitly wrong year/variant evidence remains behind all
    non-conflicting evidence.
    """
    if len(concept_queries) < 2:
        # A single-condition lookup has no coverage gap to solve. Retain the calibrated lexical
        # ordering exactly, rather than spending its prompt budget on a diversity heuristic.
        return [candidate for _, candidate in scored]

    normalized_concepts = [unicodedata.normalize("NFC", item).lower() for item in concept_queries]
    table_concepts: dict[str, set[str]] = defaultdict(set)
    by_tier: dict[int, dict[str, deque[OperandCandidate]]] = defaultdict(dict)
    for tier, candidate in scored:
        queues = by_tier[tier]
        queues.setdefault(candidate.table_key, deque()).append(candidate)
        row_label = unicodedata.normalize("NFC", candidate.row_label).lower()
        table_concepts[candidate.table_key].update(
            concept for concept in normalized_concepts if concept in row_label
        )

    ordered: list[OperandCandidate] = []
    for tier in sorted(by_tier):
        queues = by_tier[tier]
        table_order = sorted(
            queues,
            key=lambda table_key: (
                -len(table_concepts[table_key]),
                -queues[table_key][0].relevance_score,
                table_key,
            ),
        )
        while any(queues.values()):
            for table_key in table_order:
                queue = queues[table_key]
                if queue:
                    ordered.append(queue.popleft())
    return ordered


def _resolve_tickers(
    question: str, available_tickers: frozenset[str], company_by_ticker: dict[str, str] | None,
) -> frozenset[str]:
    """Canonical entity scope for this question, restricted to companies retrieval actually
    surfaced (``available_tickers``).

    Starts from the same literal-token match used before this feature existed (a bare
    ``TICKER_TOKEN_RE`` token that is itself a real, available ticker). When a ``company_by_ticker``
    registry is supplied, additionally corrects the one confirmed failure mode
    (EXECUTION_ACCURACY_AUDIT.md §6.1): a literal ticker token that is actually a *nested brand*
    mention of a *different* available company (e.g. the question literally says "FPT", but the
    only retrieved company using that word in its own registered name is FTS, "CTCP Chứng khoán
    FPT") is replaced by that company, only when exactly one available company qualifies --
    ambiguous or unmatched cases are left as the literal ticker, never guessed.

    Deliberately **not** a call to ``retrieval.full_corpus.infer_question_tickers``: that
    function's broader phrase/name-overlap fallback is tuned for retrieval's recall-oriented
    objective (over-inclusion is acceptable there, per CONTEXT.md's F2 recall weighting) and is a
    confirmed, still-open false-positive source there (TABLES_F2_AUDIT.md B.1). Reusing it
    verbatim here was tried and rejected on evidence during this change: on real fixture question
    2 (a single-ticker "ACB" lookup), it spuriously added ticker CTG via generic industry-sector
    vocabulary ("Thương mại") overlapping a substring of CTG's own registered name, corrupting an
    otherwise-correct answer. Precision matters far more than recall for this entity-scoping use
    (a wrong company silently changes the computed answer), so only the narrow, high-precision
    nested-brand correction is applied -- never a broader recall-oriented company recovery.
    """
    literal_tickers = frozenset(
        token for token in TICKER_TOKEN_RE.findall(question)
        if token not in _NON_TICKER_ACRONYMS and token in available_tickers
    )
    if not company_by_ticker:
        return literal_tickers
    resolved = set(literal_tickers)
    for ticker in literal_tickers:
        nested_in = [
            candidate for candidate in available_tickers
            if candidate != ticker and ticker.lower() in _tokens(company_by_ticker.get(candidate, ""))
        ]
        if len(nested_in) == 1:
            resolved.discard(ticker)
            resolved.add(nested_in[0])
    return frozenset(resolved)


def link_schema(
    question: str,
    tables: list[dict],
    max_operands: int = 120,
    company_by_ticker: dict[str, str] | None = None,
) -> SchemaLinkResult:
    """Rank rows lexically, then expose only real numeric cells as grounded operands.

    Ranking is two-tiered: a candidate whose table/column definitely conflicts with an
    explicitly requested year or report variant (``year_conflicts_with_question``/
    ``variant_conflicts_with_question``) sorts after every non-conflicting candidate,
    regardless of lexical score -- this tier is what ``select_relevant_tables``'s top-K
    selection actually orders by, and is the fix for run-2's measured precision collapse (top-8
    selections spanning ~5 report variants per company on average; TABLES_F2_AUDIT.md). Absent
    an explicit year/variant in the question, no candidate can conflict, so ordering is
    unchanged from the pure lexical-score ranking used before this tiering existed.

    ``company_by_ticker`` (optional, ticker -> company name) enables canonical entity resolution
    for ``resolved_tickers`` (see ``SchemaLinkResult.resolved_tickers``'s docstring); omitting it
    preserves the previous literal-token behavior exactly.
    """
    q_tokens = _expanded_question_tokens(question)
    question_years_int = infer_question_years(question)
    question_years = {str(year) for year in question_years_int}
    question_variant = infer_question_variant(question)
    scored: list[tuple[int, OperandCandidate]] = []
    for table in tables:
        grid = table["grid"]
        headers = _column_headers(table)
        year_match = bool(question_years) and str(table.get("year", "")) in question_years
        variant_conflict = variant_conflicts_with_question(table, question_variant)
        for row_info in table.get("row_labels", []):
            ri, label = row_info["row_index"], row_info["label"]
            overlap = len(q_tokens & _tokens(label))
            score = overlap / max(len(_tokens(label)), 1)
            for ci, raw in enumerate(grid[ri]):
                column_header = headers[ci] if ci < len(headers) else ""
                if not is_numeric_data_column(table, ci, column_header):
                    continue
                number = parse_financial_number(raw)
                if number is None:
                    continue
                candidate = OperandCandidate(
                    table_key=table["table_key"], report_id=table["report_id"],
                    source_path=table["source_path"], line_position=table["line_position"],
                    row_index=ri, column_index=ci, row_label=label,
                    column_header=column_header, raw_value=raw,
                    numeric_value=number, detected_units=table.get("detected_units", []),
                    relevance_score=score + (0.05 if year_match else 0.0),
                )
                year_conflict = year_conflicts_with_question(table, column_header, question_years_int)
                conflict_tier = int(variant_conflict) + int(year_conflict)
                scored.append((conflict_tier, candidate))
    if any(candidate.relevance_score > 0 for _, candidate in scored):
        scored = [(tier, candidate) for tier, candidate in scored if candidate.relevance_score > 0]
    scored.sort(key=lambda item: (
        item[0], -item[1].relevance_score, item[1].table_key, item[1].row_index, item[1].column_index,
    ))
    candidates = _interleave_tables(scored, extract_concept_queries(question))[:max_operands]
    requested = _UNIT_RE.search(question)
    available_tickers = frozenset(t["ticker"] for t in tables if t.get("ticker"))
    return SchemaLinkResult(
        question=question, query_family=classify_query_family(question, available_tickers),
        requested_unit=requested.group(0).strip() if requested else None,
        table_keys=[t["table_key"] for t in tables], operands=candidates,
        resolved_tickers=_resolve_tickers(question, available_tickers, company_by_ticker),
    )

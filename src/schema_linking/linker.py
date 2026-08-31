"""Ground Vietnamese questions to row/column operands in retrieved structured tables."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from common.financial_concepts import RATIO_OPERAND_EXPANSIONS
from retrieval.full_corpus import TICKER_TOKEN_RE

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


def classify_query_family(question: str) -> str:
    q = question.lower()
    if any(x in q for x in ("roe", "roa", "tỷ lệ", "tỷ suất", "trên tài sản")):
        return "ratio_or_derived"
    if any(x in q for x in ("bao nhiêu phần trăm", "tăng trưởng", "thay đổi bao nhiêu %")):
        return "growth"
    if any(x in q for x in ("tổng cộng", "tổng số", "trung bình", "median", "cộng lại")):
        return "aggregation"
    tickers = [
        token for token in TICKER_TOKEN_RE.findall(question)
        if token not in _NON_TICKER_ACRONYMS
    ]
    if len(set(tickers)) > 1:
        return "cross_company_comparison"
    if any(x in q for x in ("chênh lệch", "biến động", "cao hơn", "thấp hơn", "so với")):
        return "comparison"
    return "direct_lookup"


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
        "ghichu", "diengiai", "bienong", "phantram",
    }


def link_schema(question: str, tables: list[dict], max_operands: int = 120) -> SchemaLinkResult:
    """Rank rows lexically, then expose only real numeric cells as grounded operands."""
    q_tokens = _expanded_question_tokens(question)
    question_years = set(_YEAR_RE.findall(question))
    candidates = []
    for table in tables:
        grid = table["grid"]
        headers = _column_headers(table)
        year_match = bool(question_years) and str(table.get("year", "")) in question_years
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
                candidates.append(OperandCandidate(
                    table_key=table["table_key"], report_id=table["report_id"],
                    source_path=table["source_path"], line_position=table["line_position"],
                    row_index=ri, column_index=ci, row_label=label,
                    column_header=column_header, raw_value=raw,
                    numeric_value=number, detected_units=table.get("detected_units", []),
                    relevance_score=score + (0.05 if year_match else 0.0),
                ))
    if any(x.relevance_score > 0 for x in candidates):
        candidates = [x for x in candidates if x.relevance_score > 0]
    candidates.sort(key=lambda x: (-x.relevance_score, x.table_key, x.row_index, x.column_index))
    requested = _UNIT_RE.search(question)
    return SchemaLinkResult(
        question=question, query_family=classify_query_family(question),
        requested_unit=requested.group(0).strip() if requested else None,
        table_keys=[x["table_key"] for x in tables], operands=candidates[:max_operands],
    )

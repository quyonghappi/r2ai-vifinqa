"""normalization stage: raw table candidates -> unified schema (company/year/line-item/unit).

Scope boundary (AGENTS.md Section 2): this module represents extraction's output in a form
retrieval and schema_linking can use -- it does not judge relevance and does not drop
candidates. "Unified schema" here means structured metadata (ticker/year/variant, parsed from
report_id -- see the real naming-pattern diversity documented in `parse_report_id`) plus a
per-table searchable text blob (joined cell text) for retrieval indexing, and a best-effort list
of detected currency-unit hints. It does NOT do full cell-level line-item parsing (mapping each
row to a canonical financial concept) -- that is schema_linking's job once a specific table has
already been retrieved for a specific question, not something to do speculatively for all
146K+ table candidates up front.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

from extraction.parser import TableCandidate

# Real report_id naming patterns observed across the full 1,973-report corpus (see
# notebooks/vifinqa_pipeline.ipynb Checkpoint 2 / CHANGE_LOG.md):
#   TICKER_financial_statements_YEAR_{consolidated|separate|aggregated}      (1,887 reports)
#   TICKER_financial_statements_YEAR                                         (no variant stated)
#   TICKER_financial_statements_YEAR_{consolidated|separate}_{N}             (multi-part filing)
#   TICKER_YEAR_financial_statement_explanations[_N]                        (narrative-only doc)
#   TICKER_YEAR_explanatory_letters_{N}                                     (narrative-only doc)
# Token order and presence of "financial_statements" is NOT consistent, so year/variant are
# found by keyword/regex search rather than positional splitting.

_YEAR_RE = re.compile(r"(20\d{2})")
# real part suffixes observed are "_1"/"_2" (multi-part filings, e.g. HDB_..._separate_1) --
# restricted to 1-2 digits so a trailing 4-digit year (e.g. EVF_financial_statements_2018,
# which has no variant/part suffix at all) is never misread as a part number.
_PART_RE = re.compile(r"_(\d{1,2})$")

_VARIANT_KEYWORDS = [
    ("consolidated", "consolidated"),
    ("separate", "separate"),
    ("aggregated", "aggregated"),
    ("explanation", "explanatory"),
    ("explanatory", "explanatory"),
]

_UNIT_RE = re.compile(
    r"(nghìn\s*tỷ\s*đồng|nghìn\s*tỷ\s*VND|tỷ\s*đồng|tỷ\s*VND|triệu\s*đồng|triệu\s*VND|nghìn\s*đồng|nghìn\s*VND|VND)",
    re.IGNORECASE,
)

# Diagnostic-only foreign-currency hint, deliberately separate from _UNIT_RE/unit_scale_to_vnd:
# it does not participate in VND scaling (query_generation is untouched by this field) -- it
# exists so a table denominated in a non-VND currency is visible in the catalog instead of
# silently falling through unit detection built for VND-only phrasing.
_FOREIGN_CURRENCY_RE = re.compile(r"\bUSD\b|US\$|\bEUR\b|\bJPY\b|\bCNY\b|\bSGD\b", re.IGNORECASE)

# Retrieval-text builder version: bump this whenever build_schema_retrieval_text's field
# selection or ordering changes, so a persisted catalog can be checked against the code that
# produced it (see normalization/build_artifacts.py CATALOG_FIELDS).
RETRIEVAL_TEXT_VERSION = "schema_only_v1"

# 99th percentile of real corpus n_rows is 37 (measured on the full 146,243-table catalog);
# 200 is a generous cap that only trims the ~50 known OCR-pathology tables flagged in
# extraction_full_anomalies.csv (up to 1,008 rows), not ordinary tables.
MAX_ROW_LABELS = 200

# Matches extraction's own shape-anomaly threshold (diagnose_candidate_shape's max_cols=50):
# a wide/malformed table can have hundreds of distinct column header_paths, and unlike
# MAX_ROW_LABELS this had no cap until real BGE-M3 token-length EDA on the full corpus found a
# 29,740-token outlier traced to exactly this field (see CHANGE_LOG.md retrieval-representation
# entry). 99.9%+ of real tables have far fewer than 60 columns.
MAX_COLUMN_HEADERS = 60

# infer_period_labels scans every column of up to 6 header rows; an unusually wide table (the
# same OCR-pathology class that produces the n_cols anomalies in extraction_full_anomalies.csv)
# can therefore surface far more than the ~2-4 period labels an ordinary current/prior-period
# table has. Real corpus EDA still found a >8192-token outlier after MAX_ROW_LABELS/
# MAX_COLUMN_HEADERS alone -- traced to this field being the one remaining uncapped list.
MAX_PERIOD_LABELS = 40


def parse_report_id(report_id: str) -> tuple[str, Optional[int], str, Optional[int]]:
    """(ticker, year, variant, part) parsed from report_id. See module docstring for the
    real naming patterns this must handle -- it does not assume one fixed token layout."""
    ticker = report_id.split("_", 1)[0]

    year_match = _YEAR_RE.search(report_id)
    year = int(year_match.group(1)) if year_match else None

    variant = "unspecified"
    lower = report_id.lower()
    for keyword, label in _VARIANT_KEYWORDS:
        if keyword in lower:
            variant = label
            break

    part_match = _PART_RE.search(report_id)
    part = int(part_match.group(1)) if part_match else None

    return ticker, year, variant, part


def detect_units(text: str) -> list[str]:
    """Best-effort currency-unit hints found in a table's text. Diagnostic only -- does not
    convert or normalize values; schema_linking/query_generation must still confirm the unit
    for the specific line item they use."""
    seen = []
    for m in _UNIT_RE.finditer(text):
        u = m.group(1)
        if u not in seen:
            seen.append(u)
    return seen


def unit_scale_to_vnd(unit: str | None) -> int | None:
    """Return the VND multiplier for one explicit, normalized currency unit."""
    if not unit:
        return None
    text = "".join(
        char for char in unicodedata.normalize("NFD", unit.lower())
        if not unicodedata.combining(char)
    ).replace("đ", "d")
    if "nghin ty" in text:
        return 1_000_000_000_000
    if "ty" in text:
        return 1_000_000_000
    if "trieu" in text:
        return 1_000_000
    if "nghin" in text:
        return 1_000
    if "vnd" in text or "dong" in text:
        return 1
    return None


@dataclass
class NormalizedTable:
    report_id: str
    ticker: str
    year: Optional[int]
    variant: str
    part: Optional[int]
    line_position: int
    page: Optional[int]
    section_header: str
    searchable_text: str
    detected_units: list[str]
    n_rows: int
    n_cols: int
    status: str
    source_path: str
    caption_context: list[str]
    table_identity: str
    period_labels: list[dict]
    column_metadata: list[dict]
    row_labels: list[dict]
    grid: list[list[str]]
    retrieval_text: str
    retrieval_text_version: str
    currency_hint: Optional[str]
    header_depth: int
    content_hash: str


_PERIOD_RE = re.compile(
    r"(?:20\d{2}|31\s*[./-]\s*12|01\s*[./-]\s*01|số\s+(?:cuối|đầu)\s+(?:năm|kỳ)|năm\s+(?:nay|trước))",
    re.IGNORECASE,
)
_NUMERIC_CELL_RE = re.compile(r"^[\s()\-+.,%\d]+$")


def infer_period_labels(grid: list[list[str]], header_scan_rows: int = 6) -> list[dict]:
    """Preserve period-bearing header cells with their original grid coordinates."""
    labels = []
    for row_index, row in enumerate(grid[:header_scan_rows]):
        for column_index, value in enumerate(row):
            text = value.strip()
            if text and _PERIOD_RE.search(text):
                labels.append({"row_index": row_index, "column_index": column_index, "label": text})
    return labels


def infer_column_metadata(grid: list[list[str]], header_scan_rows: int = 6) -> list[dict]:
    """Record source-unit and period evidence per numeric column, not per table."""
    n_cols = max((len(row) for row in grid), default=0)
    columns = []
    for column_index in range(n_cols):
        header_cells = [
            row[column_index].strip() for row in grid[:header_scan_rows]
            if column_index < len(row) and row[column_index].strip()
            and not _NUMERIC_CELL_RE.fullmatch(row[column_index].strip())
        ]
        header_path = " | ".join(dict.fromkeys(header_cells))
        units = detect_units(header_path)
        unit = units[0] if units else None
        columns.append({
            "column_index": column_index,
            "header_path": header_path,
            "period_labels": [value for value in header_cells if _PERIOD_RE.search(value)],
            "source_unit": unit,
            "scale_to_vnd": unit_scale_to_vnd(unit),
        })
    return columns


def infer_row_labels(grid: list[list[str]]) -> list[dict]:
    """Keep one best-effort textual label per row, with coordinates and no canonical remapping."""
    labels = []
    for row_index, row in enumerate(grid):
        for column_index, value in enumerate(row):
            text = value.strip()
            if text and not _NUMERIC_CELL_RE.fullmatch(text):
                labels.append({"row_index": row_index, "column_index": column_index, "label": text})
                break
    return labels


def _dedupe_preserve_order(values) -> list[str]:
    return list(dict.fromkeys(v for v in values if v))


# OCR occasionally merges several source lines into one cell (observed directly in real
# corpus tables, e.g. VIF_financial_statements_2023_consolidated|363's row 1, a single label
# running to hundreds of characters). MAX_ROW_LABELS/MAX_COLUMN_HEADERS/MAX_PERIOD_LABELS cap
# *how many* labels are kept but not *how long* any single one is -- confirmed via real
# BGE-M3 token-length EDA: even after those count caps, one table still produced ~9,000 tokens,
# traced to exactly this. Matches schema_linking.linker._prompt_text's existing convention for
# the same underlying hazard (bounding one malformed cell so it cannot dominate a budget).
_MAX_LABEL_CHARS = 200


def _bounded(text: str, limit: int = _MAX_LABEL_CHARS) -> str:
    return text if len(text) <= limit else text[:limit - 1] + "…"


def detect_currency_hint(text: str) -> Optional[str]:
    """Best-effort non-VND currency mention. Diagnostic only -- see module note above _FOREIGN_CURRENCY_RE.
    Does not feed unit_scale_to_vnd or any query_generation logic."""
    match = _FOREIGN_CURRENCY_RE.search(text)
    return match.group(0).strip() if match else None


def infer_header_depth(grid: list[list[str]], header_scan_rows: int = 6) -> int:
    """Count of leading rows (within header_scan_rows) that look like pure header rows:
    every non-empty cell is non-numeric. Stops at the first row containing a numeric-looking
    cell (typically the first data row) or the first fully blank row. Diagnostic only -- it
    does not replace or gate infer_period_labels/infer_column_metadata's own fixed scan
    window; it exists to check whether header_scan_rows itself is well-chosen."""
    depth = 0
    for row in grid[:header_scan_rows]:
        cells = [cell.strip() for cell in row if cell.strip()]
        if cells and all(not _NUMERIC_CELL_RE.fullmatch(cell) for cell in cells):
            depth += 1
        else:
            break
    return depth


def content_hash(grid: list[list[str]]) -> str:
    """Diagnostic grid fingerprint for duplicate/near-duplicate incidence checks.

    Never used to drop or merge tables (AGENTS.md Section 2/4) -- purely inspectable metadata.
    """
    canonical = json.dumps(grid, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def build_schema_retrieval_text(
    ticker: str,
    company_name: str,
    year: Optional[int],
    variant: str,
    section_header: str,
    caption_context: list[str],
    detected_units: list[str],
    period_labels: list[dict],
    column_metadata: list[dict],
    row_labels: list[dict],
    *,
    max_row_labels: int = MAX_ROW_LABELS,
    max_column_headers: int = MAX_COLUMN_HEADERS,
    max_period_labels: int = MAX_PERIOD_LABELS,
) -> str:
    """Schema-oriented retrieval text: identity + structure + row labels, no bulk numeric
    cell dump. Row labels (e.g. "Doanh thu thuần", "Lợi nhuận sau thuế") are financial
    statements' strongest lexical/semantic signal and are deliberately included, deduplicated,
    not filtered out as "just cell text" -- see CHANGE_LOG.md retrieval-representation entry.

    Field order is front-loaded (identity -> structure -> row labels) so that truncation under
    a tokenizer's max sequence length drops the least-discriminative content (the row-label
    tail) first, not the identity/period fields that resolve company/year/variant ambiguity.
    """
    header_paths = [
        _bounded(v) for v in _dedupe_preserve_order(
            str(col.get("header_path", "")) for col in column_metadata if col.get("header_path")
        )[:max_column_headers]
    ]
    periods = [
        _bounded(v) for v in
        _dedupe_preserve_order(str(p["label"]) for p in period_labels)[:max_period_labels]
    ]
    rows = [
        _bounded(v) for v in
        _dedupe_preserve_order(str(r["label"]) for r in row_labels)[:max_row_labels]
    ]
    parts = [
        f"TICKER {ticker}.",
        f"COMPANY {company_name}." if company_name else "",
        f"YEAR {year}." if year else "",
        f"VARIANT {variant}." if variant and variant != "unspecified" else "",
        f"SECTION {section_header}." if section_header and section_header != "unclassified" else "",
        ("CAPTION " + " | ".join(_bounded(c) for c in caption_context) + ".") if caption_context else "",
        ("UNIT " + ", ".join(detected_units) + ".") if detected_units else "",
        ("PERIODS " + " | ".join(periods) + ".") if periods else "",
        ("COLUMNS " + " | ".join(header_paths) + ".") if header_paths else "",
        ("ROWS " + " | ".join(rows) + ".") if rows else "",
    ]
    return " ".join(p for p in parts if p)


def normalize_candidate(
    candidate: TableCandidate, company_by_ticker: Optional[dict[str, str]] = None
) -> NormalizedTable:
    ticker, year, variant, part = parse_report_id(candidate.report_id)
    company_name = (company_by_ticker or {}).get(ticker, "")
    cells = [cell.strip() for row in candidate.grid for cell in row if cell.strip()]
    searchable_text = " ".join(cells)
    table_identity = " | ".join(candidate.caption_context + [candidate.section_header])
    detected_units = detect_units(searchable_text + " " + " ".join(candidate.caption_context))
    period_labels = infer_period_labels(candidate.grid)
    column_metadata = infer_column_metadata(candidate.grid)
    row_labels = infer_row_labels(candidate.grid)
    return NormalizedTable(
        report_id=candidate.report_id,
        ticker=ticker,
        year=year,
        variant=variant,
        part=part,
        line_position=candidate.line_position,
        page=candidate.page,
        section_header=candidate.section_header,
        searchable_text=searchable_text,
        detected_units=detected_units,
        n_rows=candidate.n_rows,
        n_cols=candidate.n_cols,
        status=candidate.status,
        source_path=candidate.source_path,
        caption_context=list(candidate.caption_context),
        table_identity=table_identity,
        period_labels=period_labels,
        column_metadata=column_metadata,
        row_labels=row_labels,
        grid=[list(row) for row in candidate.grid],
        retrieval_text=build_schema_retrieval_text(
            ticker, company_name, year, variant, candidate.section_header,
            candidate.caption_context, detected_units, period_labels, column_metadata, row_labels,
        ),
        retrieval_text_version=RETRIEVAL_TEXT_VERSION,
        currency_hint=detect_currency_hint(searchable_text),
        header_depth=infer_header_depth(candidate.grid),
        content_hash=content_hash(candidate.grid),
    )


def normalize_corpus(
    candidates: list[TableCandidate], company_by_ticker: Optional[dict[str, str]] = None
) -> list[NormalizedTable]:
    return [normalize_candidate(c, company_by_ticker) for c in candidates]


def structured_record(table: NormalizedTable) -> dict:
    """Submission-oriented, JSON-serializable record retaining the complete source grid."""
    record = dict(table.__dict__)
    record["table_key"] = f"{table.report_id}|{table.line_position}"
    return record


def structured_record_json(table: NormalizedTable) -> str:
    return json.dumps(structured_record(table), ensure_ascii=False)

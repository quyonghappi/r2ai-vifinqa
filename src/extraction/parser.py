"""extraction stage: OCR .txt -> raw table candidates, with source line numbers preserved.

Scope boundary (AGENTS.md Section 2): this module identifies and structurally parses
<table>...</table> spans and records *where* they came from (file, physical line number,
page, nearby heading text). It does NOT decide which tables are "relevant" to any question,
and it does NOT drop tables that look like footnotes/schedules rather than primary financial
statements -- that judgment belongs to retrieval, not extraction.

Real, verified structural facts this module relies on (checked against 8 sample reports
spanning manufacturing, bank, "aggregated", and unlabeled document-name conventions --
see notebooks/vifinqa_pipeline.ipynb Checkpoint 1):
  - Each OCR report is a UTF-8 .txt file with page markers "===== PAGE N =====".
  - Every <table>...</table> span sits entirely on one physical line (verified: 0 exceptions
    across 540 tables in the 8-file sample). This makes "line position" trivial and exact --
    it's just the 1-indexed line number of that line, satisfying submission_guide.md's
    "start line of the table in the original OCR .txt file" requirement without any
    heuristic re-derivation.
  - Tables use only <table>/<tr>/<td> with optional rowspan/colspan attributes; no <th>.
    Multi-level headers (e.g. bank balance sheets) rely on rowspan/colspan and must be
    expanded into a dense grid to be usable downstream.
  - OCR noise (repeated-character border artifacts, occasional leaked non-HTML tags such as
    "<fcel>" from what looks like a table-structure-recognition model) occurs in the
    surrounding narrative text; in the 8-file sample it did not occur inside any <table> span.
    A lenient parser is still used deliberately, since this is a sample, not the full corpus.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional

PAGE_RE = re.compile(r"^===== PAGE (\d+) =====\s*$")

# Heuristic section-header keywords, used only to attach diagnostic context to a table
# candidate (a "section_header" field) -- never to filter or drop candidates.
_SECTION_PATTERNS = [
    ("balance_sheet", re.compile(r"CÂN ĐỐI KẾ TOÁN|TÌNH HÌNH TÀI CHÍNH", re.IGNORECASE)),
    ("income_statement", re.compile(r"KẾT QUẢ HOẠT ĐỘNG|KẾT QUẢ KINH DOANH", re.IGNORECASE)),
    ("cash_flow", re.compile(r"LƯU CHUYỂN TIỀN", re.IGNORECASE)),
    ("notes", re.compile(r"THUYẾT MINH", re.IGNORECASE)),
]


def classify_section_header(text: Optional[str]) -> str:
    if not text:
        return "unclassified"
    for label, pat in _SECTION_PATTERNS:
        if pat.search(text):
            return label
    return "unclassified"


class TableGridParser(HTMLParser):
    """Lenient parser for a single <table>...</table> string.

    Deliberately lenient (HTMLParser, not a strict XML parser): OCR output can leak
    unexpected tags into surrounding text, and this stage's job is to capture what's there,
    not to reject anything that isn't perfectly well-formed HTML.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows: list[list[dict]] = []
        self._cur_row: Optional[list[dict]] = None
        self._cur_cell: Optional[list[str]] = None
        self._cur_cell_attrs: Optional[dict] = None
        self.warnings: list[str] = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "tr":
            self._cur_row = []
        elif tag in ("td", "th"):
            if self._cur_cell is not None:
                # unclosed <td>/<th> (OCR/table-markup artifact) -- html.parser does not
                # implicitly close it the way a browser DOM would, so do it ourselves or
                # the previous cell's text is silently dropped.
                self.warnings.append("unclosed <td>/<th> before next cell; auto-closed")
                self._flush_cell()
            self._cur_cell_attrs = attrs
            self._cur_cell = []

    def handle_endtag(self, tag):
        if tag == "tr" and self._cur_row is not None:
            self.rows.append(self._cur_row)
            self._cur_row = None
        elif tag in ("td", "th") and self._cur_cell is not None:
            self._flush_cell()

    def _flush_cell(self):
        text = "".join(self._cur_cell).strip()
        try:
            rowspan = int(self._cur_cell_attrs.get("rowspan", 1) or 1)
            colspan = int(self._cur_cell_attrs.get("colspan", 1) or 1)
        except ValueError:
            rowspan, colspan = 1, 1
            self.warnings.append(f"non-integer rowspan/colspan in cell {text!r}")
        if self._cur_row is None:
            self._cur_row = []
            self.warnings.append(f"cell {text!r} closed outside a <tr>; recovered")
        self._cur_row.append({"text": text, "rowspan": rowspan, "colspan": colspan})
        self._cur_cell = None
        self._cur_cell_attrs = None

    def handle_data(self, data):
        if self._cur_cell is not None:
            self._cur_cell.append(data)


def expand_grid(rows: list[list[dict]]) -> list[list[str]]:
    """Expand rowspan/colspan cells into a dense rectangular grid of strings."""
    grid: dict[tuple[int, int], str] = {}
    max_cols = 0
    for r_idx, row in enumerate(rows):
        c_idx = 0
        for cell in row:
            while (r_idx, c_idx) in grid:
                c_idx += 1
            text, rs, cs = cell["text"], cell["rowspan"], cell["colspan"]
            for dr in range(rs):
                for dc in range(cs):
                    grid[(r_idx + dr, c_idx + dc)] = text
            c_idx += cs
        max_cols = max(max_cols, c_idx)
    max_rows = (max((r for r, _ in grid), default=-1)) + 1
    return [[grid.get((r, c), "") for c in range(max_cols)] for r in range(max_rows)]


@dataclass
class TableCandidate:
    report_id: str
    source_path: str
    line_position: int  # 1-indexed physical line number of the <table> line in the .txt file
    page: Optional[int]
    section_header: str  # diagnostic only, see classify_section_header
    caption_context: list[str]  # up to 3 non-empty lines immediately preceding the table
    status: str  # "success" | "parsed_with_warnings" | "failed"
    warnings: list[str]
    n_rows: int
    n_cols: int
    grid: list[list[str]] = field(default_factory=list)


def derive_report_id(txt_path: str) -> str:
    """report_id per submission_guide.md, resolved (see CHANGE_LOG.md Checkpoint 1 entry).

    submission_guide.md's literal rule text ("final filename component, minus .txt") and its
    worked example disagree once applied to the real ViFinQA filenames: the actual files are
    named "<doc>_extracted.txt", but the guide's own example path
    (`ocr_filter\\AAA\\2015\\AAA_financial_statements_2015_consolidated`) has neither a `.txt`
    extension nor an `_extracted` suffix -- it matches the *parent directory* name instead. That
    example path shape matches the `ocr_filter/` corpus layout the dataset's own README
    describes as the organizers' internal/reference layout, distinct from the `financial_statements/`
    layout actually published on Hugging Face. Verified against all 1,973 report files in the
    published corpus: `parent_directory_name + "_extracted.txt" == filename` holds with zero
    exceptions, so report_id = parent directory name is both well-defined and consistent with
    the guide's example. `_extracted` is therefore this release's file-naming artifact, not
    part of the canonical report_id.
    """
    return Path(txt_path).parent.name


def extract_tables_from_file(
    txt_path: str,
    report_id: Optional[str] = None,
    provenance_source_path: Optional[str] = None,
) -> list[TableCandidate]:
    path = Path(txt_path)
    rid = report_id or derive_report_id(str(path))
    with path.open(encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    results: list[TableCandidate] = []
    current_page: Optional[int] = None
    current_section: Optional[str] = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        page_match = PAGE_RE.match(stripped)
        if page_match:
            current_page = int(page_match.group(1))
            continue

        if "<table>" not in stripped:
            for label, pat in _SECTION_PATTERNS:
                if pat.search(stripped):
                    current_section = stripped
                    break
            continue

        if "<table>" in stripped and "</table>" in stripped:
            line_no = i + 1
            caption_context: list[str] = []
            j = i - 1
            while j >= 0 and len(caption_context) < 3:
                s = lines[j].strip()
                if PAGE_RE.match(s):
                    break
                if s:
                    caption_context.insert(0, s)
                j -= 1

            status, warnings, grid = "success", [], []
            try:
                parser = TableGridParser()
                parser.feed(stripped)
                parser.close()
                warnings = parser.warnings
                grid = expand_grid(parser.rows)
                if warnings:
                    status = "parsed_with_warnings"
                if not grid or not grid[0]:
                    status = "failed"
                    warnings = warnings + ["empty grid after parse"]
            except Exception as e:  # noqa: BLE001 -- deliberately broad: record, don't crash the batch
                status = "failed"
                warnings = [f"{type(e).__name__}: {e}"]

            results.append(
                TableCandidate(
                    report_id=rid,
                    source_path=provenance_source_path or str(path),
                    line_position=line_no,
                    page=current_page,
                    section_header=classify_section_header(current_section),
                    caption_context=caption_context,
                    status=status,
                    warnings=warnings,
                    n_rows=len(grid),
                    n_cols=(max((len(r) for r in grid), default=0)),
                    grid=grid,
                )
            )

    return results


def extract_corpus(
    txt_paths: list[str],
    report_ids: Optional[list[str]] = None,
    provenance_source_paths: Optional[list[str]] = None,
) -> list[TableCandidate]:
    """Batch extraction over many files. Pure CPU/text work -- no model, no GPU.

    report_ids, if given, must be parallel to txt_paths and is passed through to
    extract_tables_from_file instead of relying on derive_report_id(local_path). This matters
    whenever the local cache/download layout doesn't mirror the original corpus's directory
    structure (derive_report_id relies on the *parent directory* name -- see its docstring --
    so a flattened local copy would silently produce a wrong report_id).
    """
    if report_ids is not None and len(report_ids) != len(txt_paths):
        raise ValueError("report_ids must be the same length as txt_paths if provided")
    if provenance_source_paths is not None and len(provenance_source_paths) != len(txt_paths):
        raise ValueError("provenance_source_paths must match txt_paths if provided")
    all_results: list[TableCandidate] = []
    for i, p in enumerate(txt_paths):
        rid = report_ids[i] if report_ids is not None else None
        source_path = provenance_source_paths[i] if provenance_source_paths is not None else None
        all_results.extend(
            extract_tables_from_file(p, report_id=rid, provenance_source_path=source_path)
        )
    return all_results


def diagnose_candidate_shape(
    candidate: TableCandidate, *, max_rows: int = 200, max_cols: int = 50
) -> list[str]:
    """Return inspectable shape anomalies without filtering or changing parse status.

    Large dimensions can be legitimate, but in the real ViFinQA corpus they also expose OCR
    pathologies such as ``rowspan=927`` and single rows containing more than 1,000 cells.  Keep
    these candidates (extraction is non-filtering) while making them visible for review.
    """
    anomalies = []
    if candidate.n_rows > max_rows:
        anomalies.append(f"n_rows>{max_rows}:{candidate.n_rows}")
    if candidate.n_cols > max_cols:
        anomalies.append(f"n_cols>{max_cols}:{candidate.n_cols}")
    return anomalies

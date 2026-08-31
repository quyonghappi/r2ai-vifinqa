"""Canonical Vietnamese financial-concept lexicon, shared by retrieval (query decomposition)
and schema_linking (question-token expansion) so there is one source of truth, not two
independently-drifting copies -- the same fix already applied once in this codebase for
TICKER_TOKEN_RE (see retrieval/full_corpus.py's docstring on that history).

Construction method (2026-08-31 query-decomposition design audit; see CHANGE_LOG.md): built
from n-gram frequency in the QUESTION corpus (data/raw/hf_meta/questions.jsonl), weighted toward
the compound/hard subset, NOT from row-label frequency across the table corpus. Row-label
frequency was tried first and rejected: it biases toward boilerplate common in every table
regardless of whether it's ever the target of a hard question, and it is structurally blind to
every named ratio/derived metric (ROE, "biên lợi nhuận gộp", "hệ số thanh toán nhanh", ...)
since those never appear as literal row labels at all -- they are computed, not reported.

COMMON_LINE_ITEM_PHRASES entries are each confirmed groundable (present as an actual row label
substring, checked against a 30k sample of data/processed/row_label_index.csv) via literal
substring match, NOT token-set overlap. Token-set overlap was tried and rejected: this corpus's
DEFAULT_STOPWORDS strips "trước"/"sau" (before/after), which would make "lợi nhuận trước thuế"
(pre-tax profit) and "lợi nhuận sau thuế" (net income) -- financially distinct, often very
different numbers -- collide.

RATIO_OPERAND_EXPANSIONS entries are named ratios/derived metrics confirmed NOT to exist as
literal row labels anywhere in the corpus, mapped to their standard-formula raw operand phrases.
Extends the original 3-entry table (roe/roa/vòng quay tổng tài sản) with entries confirmed
frequent in the question corpus's hard/compound subset.
"""
from __future__ import annotations

COMMON_LINE_ITEM_PHRASES: list[str] = [
    "doanh thu thuần",
    "lợi nhuận sau thuế",
    "lợi nhuận trước thuế",
    "lợi nhuận gộp",
    "lợi nhuận thuần từ hoạt động kinh doanh",
    "vốn chủ sở hữu",
    "tổng tài sản",
    "tổng cộng tài sản",
    "nợ phải trả",
    "nợ ngắn hạn",
    "tài sản ngắn hạn",
    "hàng tồn kho",
    "tài sản cố định",
    "lưu chuyển tiền thuần từ hoạt động kinh doanh",
    "chi phí lãi vay",
    "giá vốn hàng bán",
]

RATIO_OPERAND_EXPANSIONS: dict[str, list[str]] = {
    "roe": ["vốn chủ sở hữu", "lợi nhuận sau thuế"],
    "roa": ["tổng cộng tài sản", "lợi nhuận sau thuế"],
    "vòng quay tổng tài sản": ["doanh thu thuần", "tổng tài sản"],
    "vòng quay tài sản cố định": ["doanh thu thuần", "tài sản cố định"],
    "vòng quay hàng tồn kho": ["giá vốn hàng bán", "hàng tồn kho"],
    "biên lợi nhuận gộp": ["lợi nhuận gộp", "doanh thu thuần"],
    "biên lợi nhuận ròng": ["lợi nhuận sau thuế", "doanh thu thuần"],
    "biên lợi nhuận hoạt động": ["lợi nhuận thuần từ hoạt động kinh doanh", "doanh thu thuần"],
    "hệ số thanh toán nhanh": ["tài sản ngắn hạn", "hàng tồn kho", "nợ ngắn hạn"],
    "hệ số thanh toán hiện hành": ["tài sản ngắn hạn", "nợ ngắn hạn"],
    "khả năng thanh toán lãi vay": ["lợi nhuận trước thuế", "chi phí lãi vay"],
    "hệ số nợ": ["nợ phải trả", "tổng tài sản"],
    "hệ số dòng tiền": ["lưu chuyển tiền thuần từ hoạt động kinh doanh", "nợ ngắn hạn"],
}


def literal_phrase_matches(normalized_question: str) -> list[str]:
    """Longest-match-first (maximal munch) literal phrase detection over an already
    NFC-normalized, lowercased question string. Confirmed necessary (2026-08-31 dry-run): a
    shorter phrase (e.g. "tổng tài sản") can be a literal substring of a longer, different, more
    specific phrase (e.g. "tổng tài sản cố định", total FIXED assets) -- without longest-match
    resolution the shorter phrase would be wrongly counted as its own distinct concept."""
    candidates = sorted(COMMON_LINE_ITEM_PHRASES, key=len, reverse=True)
    claimed = [False] * len(normalized_question)
    matched: list[str] = []
    for phrase in candidates:
        start = 0
        while True:
            i = normalized_question.find(phrase, start)
            if i == -1:
                break
            span = range(i, i + len(phrase))
            if not any(claimed[j] for j in span):
                for j in span:
                    claimed[j] = True
                if phrase not in matched:
                    matched.append(phrase)
            start = i + 1
    return matched

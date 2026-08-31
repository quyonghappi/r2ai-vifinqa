import json

from retrieval.rerank import (
    build_row_label_index, row_label_overlap_score, row_label_text_from_record,
)
from retrieval.sparse import DEFAULT_STOPWORDS, tokenize


def test_row_label_text_from_record_dedupes_and_joins_labels():
    record = {
        "row_labels": [
            {"label": "Doanh thu thuần"},
            {"label": "Doanh thu thuần"},  # duplicate must be dropped
            {"label": "Lợi nhuận sau thuế"},
        ]
    }
    assert row_label_text_from_record(record) == "Doanh thu thuần | Lợi nhuận sau thuế"


def test_row_label_text_from_record_handles_missing_row_labels():
    assert row_label_text_from_record({"row_labels": []}) == ""
    assert row_label_text_from_record({}) == ""


def test_build_row_label_index_streams_jsonl_to_csv(tmp_path):
    jsonl = tmp_path / "normalized_tables.jsonl"
    jsonl.write_text(
        json.dumps({"table_key": "AAA|1", "row_labels": [{"label": "Doanh thu thuần"}]}) + "\n"
        + json.dumps({"report_id": "BBB", "line_position": 2, "row_labels": [{"label": "Chi phí khác"}]}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "row_label_index.csv"
    n = build_row_label_index(jsonl, output)
    assert n == 2
    content = output.read_text(encoding="utf-8")
    assert "AAA|1,Doanh thu thuần" in content
    assert "BBB|2,Chi phí khác" in content  # table_key falls back to report_id|line_position


def test_row_label_overlap_score_prefers_exact_label_over_padded_one():
    # Real regression case (CHANGE_LOG.md 2026-08-31 Priority-2 entry, dev_v1 Q9): a bag-of-all-
    # labels recall-only score let "Chi phí khác bằng tiền" (a different, padded line item) tie
    # or beat the exact "Chi phí khác" label. Per-label Jaccard must prefer the exact match.
    query_tokens = frozenset(tokenize(
        "Chi phí khác của SAM năm 2023 là bao nhiêu triệu đồng?", stopwords=DEFAULT_STOPWORDS,
    ))
    exact_table = "Mã số | VND | 13. Chi phí khác | 14. Lợi nhuận khác"
    padded_table = "Năm 2023 | VND | Chi phí khác bằng tiền | Chi phí dịch vụ mua ngoài"
    exact_score = row_label_overlap_score(query_tokens, exact_table, DEFAULT_STOPWORDS)
    padded_score = row_label_overlap_score(query_tokens, padded_table, DEFAULT_STOPWORDS)
    assert exact_score > padded_score


def test_row_label_overlap_score_ignores_large_unrelated_label_count():
    # A table with many unrelated row labels must not out-score a table with one exact label
    # purely for having more surface area to match against (the bug this scorer replaced scored
    # the whole table's pooled labels together, not the best individual one).
    query_tokens = frozenset(tokenize("Doanh thu thuần", stopwords=DEFAULT_STOPWORDS))
    one_exact_label = "Doanh thu thuần"
    many_unrelated_labels = " | ".join([f"Mục không liên quan {i}" for i in range(50)])
    assert row_label_overlap_score(query_tokens, one_exact_label, DEFAULT_STOPWORDS) > \
        row_label_overlap_score(query_tokens, many_unrelated_labels, DEFAULT_STOPWORDS)


def test_row_label_overlap_score_empty_inputs():
    assert row_label_overlap_score(frozenset(), "Doanh thu thuần", DEFAULT_STOPWORDS) == 0.0
    assert row_label_overlap_score(frozenset({"doanh", "thu"}), "", DEFAULT_STOPWORDS) == 0.0

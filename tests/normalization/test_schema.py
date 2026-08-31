from pathlib import Path

from extraction.parser import extract_tables_from_file
from normalization.schema import (
    build_schema_retrieval_text,
    content_hash,
    detect_currency_hint,
    detect_units,
    infer_column_metadata,
    infer_header_depth,
    normalize_corpus,
    parse_report_id,
    structured_record,
    unit_scale_to_vnd,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "SYN_financial_statements_2020_consolidated_extracted.txt"


def test_parse_report_id_standard_pattern():
    assert parse_report_id("AAA_financial_statements_2015_consolidated") == ("AAA", 2015, "consolidated", None)


def test_parse_report_id_no_variant():
    assert parse_report_id("EVF_financial_statements_2018") == ("EVF", 2018, "unspecified", None)


def test_parse_report_id_multi_part():
    assert parse_report_id("HDB_financial_statements_2022_separate_1") == ("HDB", 2022, "separate", 1)


def test_parse_report_id_explanatory_letter_different_token_order():
    # ticker/year come before "financial_statements" is even mentioned -- must not assume
    # positional layout.
    assert parse_report_id("PRT_2021_explanatory_letters_2") == ("PRT", 2021, "explanatory", 2)


def test_detect_units_finds_vietnamese_currency_hints():
    assert detect_units("31/12/2022 Triệu VND 31/12/2021 Triệu VND") == ["Triệu VND"]
    assert detect_units("không có đơn vị ở đây") == []


def test_column_metadata_keeps_unit_and_period_with_its_column():
    metadata = infer_column_metadata([
        ["Metric", "Year 2023 VND", "Year 2022 VND"],
        ["Revenue", "1,000", "900"],
    ])
    assert metadata[1]["source_unit"] == "VND"
    assert metadata[1]["scale_to_vnd"] == 1
    assert "2023" in metadata[1]["header_path"]
    assert unit_scale_to_vnd("VND") == 1


def test_normalize_corpus_builds_searchable_text_and_metadata():
    candidates = extract_tables_from_file(str(FIXTURE), report_id="SYN_financial_statements_2020_consolidated")
    normalized = normalize_corpus(candidates)
    assert len(normalized) == 4

    balance_sheet = normalized[0]
    assert balance_sheet.ticker == "SYN"
    assert balance_sheet.year == 2020
    assert balance_sheet.variant == "consolidated"
    assert "TÀI SẢN" in balance_sheet.searchable_text
    assert "Mã số" in balance_sheet.searchable_text

    bank_table = normalized[1]
    assert "VND" in " ".join(bank_table.detected_units) or bank_table.detected_units

    record = structured_record(balance_sheet)
    assert record["table_key"].endswith(f"|{balance_sheet.line_position}")
    assert record["grid"] == candidates[0].grid
    assert record["row_labels"]
    assert len(record["column_metadata"]) == candidates[0].n_cols
    assert record["source_path"] == str(FIXTURE)


def test_normalize_corpus_resolves_company_name_when_provided():
    candidates = extract_tables_from_file(str(FIXTURE), report_id="SYN_financial_statements_2020_consolidated")
    normalized = normalize_corpus(candidates, {"SYN": "Cong ty CP SYN"})
    assert "COMPANY Cong ty CP SYN." in normalized[0].retrieval_text
    unresolved = normalize_corpus(candidates)
    assert "COMPANY" not in unresolved[0].retrieval_text


def test_schema_retrieval_text_includes_row_labels_not_numeric_dump():
    text = build_schema_retrieval_text(
        ticker="AAA", company_name="Cong ty AAA", year=2023, variant="consolidated",
        section_header="income_statement", caption_context=["BÁO CÁO KẾT QUẢ KINH DOANH"],
        detected_units=["Triệu VND"],
        period_labels=[{"label": "Năm nay"}, {"label": "Năm trước"}],
        column_metadata=[{"header_path": "Chỉ tiêu"}, {"header_path": "Năm nay"}],
        row_labels=[
            {"label": "Doanh thu thuần"}, {"label": "Chi phí lãi vay"},
            {"label": "Lợi nhuận sau thuế"}, {"label": "Doanh thu thuần"},  # duplicate, must collapse
        ],
    )
    assert "Doanh thu thuần" in text
    assert "Chi phí lãi vay" in text
    assert "Lợi nhuận sau thuế" in text
    assert text.count("Doanh thu thuần") == 1  # deduplicated
    assert "TICKER AAA." in text
    assert "COMPANY Cong ty AAA." in text
    assert "YEAR 2023." in text
    assert "VARIANT consolidated." in text
    # no bulk numeric cell values -- only structural/label text is present
    assert "123" not in text and "1,000" not in text


def test_schema_retrieval_text_front_loads_identity_before_row_labels():
    text = build_schema_retrieval_text(
        ticker="AAA", company_name="Cong ty AAA", year=2023, variant="unspecified",
        section_header="unclassified", caption_context=[], detected_units=[],
        period_labels=[], column_metadata=[], row_labels=[{"label": "Doanh thu thuần"}],
    )
    assert text.index("TICKER AAA") < text.index("Doanh thu thuần")


def test_schema_retrieval_text_caps_row_label_count():
    row_labels = [{"label": f"Row {i}"} for i in range(500)]
    text = build_schema_retrieval_text(
        ticker="AAA", company_name="", year=2023, variant="unspecified",
        section_header="unclassified", caption_context=[], detected_units=[],
        period_labels=[], column_metadata=[], row_labels=row_labels, max_row_labels=10,
    )
    assert text.count("Row ") == 10


def test_schema_retrieval_text_caps_column_header_count():
    column_metadata = [{"header_path": f"Col {i}"} for i in range(200)]
    text = build_schema_retrieval_text(
        ticker="AAA", company_name="", year=2023, variant="unspecified",
        section_header="unclassified", caption_context=[], detected_units=[],
        period_labels=[], column_metadata=column_metadata, row_labels=[], max_column_headers=5,
    )
    assert text.count("Col ") == 5


def test_schema_retrieval_text_bounds_a_single_oversized_merged_label():
    # OCR sometimes merges many source lines into one cell (observed directly in the real
    # corpus); a count cap alone does not bound this -- one such label must still be trimmed.
    merged_blob = "Chỉ tiêu rất dài " * 100  # ~1800 chars, well over the 200-char label bound
    text = build_schema_retrieval_text(
        ticker="AAA", company_name="", year=2023, variant="unspecified",
        section_header="unclassified", caption_context=[], detected_units=[],
        period_labels=[], column_metadata=[], row_labels=[{"label": merged_blob}],
    )
    assert len(text) < len(merged_blob)
    assert text.endswith("….")


def test_schema_retrieval_text_caps_period_label_count():
    period_labels = [{"label": f"Năm {2000 + i}"} for i in range(100)]
    text = build_schema_retrieval_text(
        ticker="AAA", company_name="", year=2023, variant="unspecified",
        section_header="unclassified", caption_context=[], detected_units=[],
        period_labels=period_labels, column_metadata=[], row_labels=[], max_period_labels=5,
    )
    assert text.count("Năm ") == 5


def test_detect_currency_hint_finds_foreign_currency_and_ignores_vnd():
    assert detect_currency_hint("Số dư tiền gửi USD tại ngân hàng") == "USD"
    assert detect_currency_hint("31/12/2022 Triệu VND") is None


def test_infer_header_depth_stops_at_first_numeric_row():
    grid = [
        ["Chỉ tiêu", "Năm nay", "Năm trước"],
        ["", "VND", "VND"],
        ["Doanh thu thuần", "1,000", "900"],
    ]
    assert infer_header_depth(grid) == 2


def test_content_hash_is_deterministic_and_grid_sensitive():
    grid_a = [["Doanh thu", "1000"]]
    grid_b = [["Doanh thu", "1001"]]
    assert content_hash(grid_a) == content_hash(grid_a)
    assert content_hash(grid_a) != content_hash(grid_b)

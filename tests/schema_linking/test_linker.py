import unicodedata

from schema_linking.linker import (
    classify_query_family,
    is_numeric_data_column,
    link_schema,
    parse_financial_number,
)


def table_fixture():
    return {
        "table_key": "AAA_report|10", "report_id": "AAA_report", "source_path": "financial_statements/AAA/report.txt",
        "line_position": 10, "n_cols": 3, "detected_units": ["VND"],
        "grid": [["CHỈ TIÊU", "Năm 2023", "Năm 2022"], ["Doanh thu thuần", "1.234.000", "1.000.000"]],
        "row_labels": [{"row_index": 0, "column_index": 0, "label": "CHỈ TIÊU"},
                       {"row_index": 1, "column_index": 0, "label": "Doanh thu thuần"}],
    }


def test_financial_number_parsing():
    assert parse_financial_number("(1.234.000)") == -1234000.0
    assert parse_financial_number("12,5%") == 12.5
    assert parse_financial_number("-") is None


def test_linker_exposes_only_real_cells_with_provenance():
    linked = link_schema("Doanh thu thuần AAA năm 2023 là bao nhiêu VND?", [table_fixture()])
    operand = linked.operands[0]
    assert operand.table_key == "AAA_report|10"
    assert operand.row_index == 1
    assert {x.numeric_value for x in linked.operands} == {1234000.0, 1000000.0}
    assert linked.requested_unit.lower() == "vnd"


def test_linker_normalizes_decomposed_vietnamese_unicode():
    question_nfd = unicodedata.normalize("NFD", "Doanh thu thuần AAA năm 2023")
    linked = link_schema(question_nfd, [table_fixture()])
    assert linked.operands and linked.operands[0].row_label == "Doanh thu thuần"


def test_query_family_is_observable():
    assert classify_query_family("ROE năm 2023 là bao nhiêu phần trăm?") == "ratio_or_derived"
    assert classify_query_family("AAA cao hơn BBB bao nhiêu?") == "cross_company_comparison"


def test_corporate_form_is_not_mistaken_for_a_second_ticker():
    assert classify_query_family("CTCP AAA in 2023") == "direct_lookup"
    assert classify_query_family("CP AAA cao hơn 2023") == "comparison"


def test_alphanumeric_ticker_is_recognized_as_a_single_company_mention():
    # HT1 is a real ticker with a trailing digit; must not be double-counted with a bare year
    # and must not trip cross_company_comparison on its own (see full_corpus.TICKER_TOKEN_RE).
    assert classify_query_family("Quỹ khen thưởng của HT1 cuối năm 2019") == "direct_lookup"


def test_column_header_excludes_values_from_data_rows():
    table = table_fixture()
    linked = link_schema(table["row_labels"][1]["label"], [table])
    operand = linked.operands[0]
    assert operand.raw_value not in operand.column_header


def test_linker_excludes_compact_note_and_code_columns_before_operand_ranking():
    table = table_fixture()
    table["grid"] = [
        ["Code", "Note", "Year 2023"],
        ["110", "4", "1.234.000"],
    ]
    table["row_labels"] = [{"row_index": 0, "column_index": 0, "label": "Code"},
                           {"row_index": 1, "column_index": 0, "label": "Revenue"}]
    linked = link_schema("Revenue AAA 2023", [table])
    assert {(x.column_index, x.numeric_value) for x in linked.operands} == {(2, 1234000.0)}
    assert not is_numeric_data_column(table, 1, "Note")
    assert is_numeric_data_column(table, 2, "Prior year (restated - note 33)")


def test_roe_expansion_keeps_profit_and_equity_operands_grounded():
    table = table_fixture()
    table["grid"] += [["Vốn chủ sở hữu", "500", "400"], ["Lợi nhuận sau thuế", "50", "40"]]
    table["row_labels"] += [
        {"row_index": 2, "column_index": 0, "label": "Vốn chủ sở hữu"},
        {"row_index": 3, "column_index": 0, "label": "Lợi nhuận sau thuế"},
    ]
    linked = link_schema("ROE năm 2023 là bao nhiêu?", [table])
    labels = {x.row_label for x in linked.operands}
    assert {"Vốn chủ sở hữu", "Lợi nhuận sau thuế"} <= labels

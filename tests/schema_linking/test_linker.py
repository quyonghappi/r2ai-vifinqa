import unicodedata

from schema_linking.linker import (
    OperandCandidate,
    classify_query_family,
    is_numeric_data_column,
    link_schema,
    parse_financial_number,
    select_relevant_tables,
    variant_conflicts_with_question,
    year_conflicts_with_question,
)


def _operand(table_key: str, relevance_score: float) -> OperandCandidate:
    return OperandCandidate(
        table_key=table_key, report_id=table_key.split("|")[0], source_path="x", line_position=1,
        row_index=0, column_index=1, row_label="x", column_header="x", raw_value="1",
        numeric_value=1.0, detected_units=[], relevance_score=relevance_score,
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


def test_numeric_data_column_excludes_vietnamese_structural_headers():
    table = table_fixture()
    assert is_numeric_data_column(table, 1, "Chỉ tiêu") is False
    assert is_numeric_data_column(table, 1, "Nguồn vốn") is False
    assert is_numeric_data_column(table, 1, "Năm 2023") is True


def test_linker_interleaves_tables_before_repeating_one_table_cells():
    first = table_fixture()
    first["table_key"] = "AAA_report|10"
    first["grid"][0] = ["CHỈ TIÊU", "Năm 2023", "Năm 2023"]
    first["row_labels"].append({"row_index": 2, "column_index": 0, "label": "Doanh thu thuần"})
    first["grid"].append(["Doanh thu thuần", "900", "800"])
    second = {**table_fixture(), "table_key": "AAA_report|20", "line_position": 20}
    second["grid"][0] = ["CHỈ TIÊU", "Năm 2023", "Năm 2023"]
    linked = link_schema("ROE và vòng quay tổng tài sản của AAA năm 2023", [first, second])
    assert [x.table_key for x in linked.operands[:4]] == [
        "AAA_report|10", "AAA_report|20", "AAA_report|10", "AAA_report|20",
    ]


def test_linker_expands_inclusive_year_range_for_conflict_tiering():
    older = table_fixture()
    older.update({"table_key": "AAA_2021|10", "year": 2021})
    middle = {**table_fixture(), "table_key": "AAA_2023|10", "year": 2023}
    linked = link_schema("Doanh thu thuần AAA giai đoạn 2022-2024", [older, middle])
    assert linked.operands[0].table_key == "AAA_2023|10"


def test_linker_leads_with_a_table_covering_more_required_concepts():
    broad = table_fixture()
    broad.update({"table_key": "AAA_report|10", "year": 2023})
    broad["grid"] += [["Vốn chủ sở hữu", "500", "400"], ["Tổng tài sản", "900", "800"]]
    broad["row_labels"] += [
        {"row_index": 2, "column_index": 0, "label": "Vốn chủ sở hữu"},
        {"row_index": 3, "column_index": 0, "label": "Tổng tài sản"},
    ]
    narrow = {**table_fixture(), "table_key": "AAA_report|20", "year": 2023}
    linked = link_schema("ROE và vòng quay tổng tài sản của AAA năm 2023", [narrow, broad])
    assert linked.operands[0].table_key == "AAA_report|10"


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
    assert classify_query_family("AAA cao hơn BBB bao nhiêu?", {"AAA", "BBB"}) == "cross_company_comparison"


def test_corporate_form_is_not_mistaken_for_a_second_ticker():
    assert classify_query_family("CTCP AAA in 2023", {"AAA"}) == "direct_lookup"
    assert classify_query_family("CP AAA cao hơn 2023", {"AAA"}) == "comparison"


def test_alphanumeric_ticker_is_recognized_as_a_single_company_mention():
    # HT1 is a real ticker with a trailing digit; must not be double-counted with a bare year
    # and must not trip cross_company_comparison on its own (see full_corpus.TICKER_TOKEN_RE).
    assert classify_query_family("Quỹ khen thưởng của HT1 cuối năm 2019", {"HT1"}) == "direct_lookup"


def test_generic_financial_acronym_is_not_mistaken_for_a_second_company():
    # Confirmed defect (2026-09-01 diagnostic audit, predictions/*.json real IDs 181/182/191/213):
    # "TNDN" (corporate income tax), "USD", "TCTD" (credit institution), "EPS", "FVTPL" etc. all
    # match TICKER_TOKEN_RE's bare 2-5-char uppercase pattern but are not tickers -- only VRE is
    # a real, retrieval-resolved ticker here, so this must stay a plain single-company lookup.
    q = "Lợi nhuận sau thuế TNDN của công ty mẹ VRE năm 2020 là bao nhiêu?"
    assert classify_query_family(q, {"VRE"}) == "direct_lookup"
    # An unvalidated available_tickers set (the pre-fix default) must not accidentally resurrect
    # the bug: with no real tickers supplied at all, a bare acronym can never count as a company.
    assert classify_query_family(q) == "direct_lookup"


def test_select_relevant_tables_keeps_top_k_distinct_tables_in_score_order():
    # Pre-sorted descending by relevance_score, as link_schema always produces (duplicates of the
    # same table interleaved, as real multi-operand tables look).
    operands = [
        _operand("A|1", 0.9), _operand("B|2", 0.8), _operand("A|1", 0.8),
        _operand("C|3", 0.5), _operand("D|4", 0.3),
    ]
    assert select_relevant_tables(operands, top_k=2) == ["A|1", "B|2"]
    assert select_relevant_tables(operands, top_k=10) == ["A|1", "B|2", "C|3", "D|4"]


def test_select_relevant_tables_never_touches_execution_provenance():
    # This is a pure selection function: it must not require or inspect anything about which
    # operands a generated pandas_query actually ended up referencing.
    operands = [_operand(f"T{i}|1", 1.0 - i * 0.01) for i in range(20)]
    result = select_relevant_tables(operands)  # default top_k
    assert len(result) == 8  # RELEVANT_TABLES_TOP_K
    assert result == [f"T{i}|1" for i in range(8)]


def test_two_real_available_tickers_still_trigger_cross_company_comparison():
    # The fix must not just suppress false positives -- a question naming 2+ tickers that
    # retrieval actually resolved as relevant is still a genuine cross-company comparison
    # (checked ahead of the plain "comparison" keyword branch, unchanged ordering).
    q = "Chênh lệch tổng phải thu ngắn hạn giữa DIG và SCR cuối năm 2018 là bao nhiêu?"
    assert classify_query_family(q, {"DIG", "SCR"}) == "cross_company_comparison"
    assert classify_query_family("So sánh DIG và SCR năm 2018", {"DIG", "SCR"}) == "cross_company_comparison"


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


def _year_variant_tables():
    # Same conceptual line item in two reports of company AAA. The wrong-year table's row
    # label is an exact phrase match against the question (raw lexical score 1.0); the
    # right-year table's label carries extra words the question doesn't mention (raw lexical
    # score 0.33, confirmed via the real _tokens/_expanded_question_tokens scoring), so that
    # absent tiering the wrong-year table would clearly rank first on lexical grounds alone --
    # the exact "wrong period, higher lexical score" shape run-2 exhibited.
    right_year = {
        "table_key": "AAA_report|10", "report_id": "AAA_report_2023", "source_path": "x",
        "line_position": 10, "n_cols": 2, "year": 2023, "variant": "separate",
        "grid": [["CHỈ TIÊU", "Năm 2023"], ["Doanh thu thuần từ hoạt động tài chính khác", "1.234.000"]],
        "row_labels": [{"row_index": 0, "column_index": 0, "label": "CHỈ TIÊU"},
                       {"row_index": 1, "column_index": 0, "label": "Doanh thu thuần từ hoạt động tài chính khác"}],
    }
    wrong_year = {
        "table_key": "AAA_report|20", "report_id": "AAA_report_2022", "source_path": "x",
        "line_position": 20, "n_cols": 2, "year": 2022, "variant": "separate",
        "grid": [["CHỈ TIÊU", "Năm 2022"], ["Doanh thu thuần", "1.000.000"]],
        "row_labels": [{"row_index": 0, "column_index": 0, "label": "CHỈ TIÊU"},
                       {"row_index": 1, "column_index": 0, "label": "Doanh thu thuần"}],
    }
    return right_year, wrong_year


def test_link_schema_prefers_matching_year_over_higher_lexical_score_when_year_named():
    right_year, wrong_year = _year_variant_tables()
    linked = link_schema("Doanh thu thuần AAA năm 2023 là bao nhiêu?", [right_year, wrong_year])
    assert linked.operands[0].table_key == "AAA_report|10"  # the 2023 table, despite the weaker label


def test_link_schema_prefers_matching_variant_over_higher_lexical_score_when_variant_named():
    consolidated = {
        "table_key": "AAA_report|30", "report_id": "AAA_report_consolidated", "source_path": "x",
        "line_position": 30, "n_cols": 2, "year": 2023, "variant": "consolidated",
        "grid": [["CHỈ TIÊU", "Năm 2023"], ["Doanh thu thuần", "1.000.000"]],
        "row_labels": [{"row_index": 0, "column_index": 0, "label": "CHỈ TIÊU"},
                       {"row_index": 1, "column_index": 0, "label": "Doanh thu thuần"}],
    }
    separate = {
        "table_key": "AAA_report|40", "report_id": "AAA_report_separate", "source_path": "x",
        "line_position": 40, "n_cols": 2, "year": 2023, "variant": "separate",
        "grid": [["CHỈ TIÊU", "Năm 2023"], ["Doanh thu thuần từ hoạt động tài chính khác", "1.234.000"]],
        "row_labels": [{"row_index": 0, "column_index": 0, "label": "CHỈ TIÊU"},
                       {"row_index": 1, "column_index": 0, "label": "Doanh thu thuần từ hoạt động tài chính khác"}],
    }
    q = "Doanh thu thuần công ty mẹ AAA năm 2023 là bao nhiêu?"
    linked = link_schema(q, [consolidated, separate])
    assert linked.operands[0].table_key == "AAA_report|40"  # the separate table, despite the weaker label


def test_link_schema_does_not_penalize_unspecified_variant_metadata():
    unspecified = {
        "table_key": "AAA_report|50", "report_id": "AAA_report_unspecified", "source_path": "x",
        "line_position": 50, "n_cols": 2, "year": 2023,  # no "variant" key at all
        "grid": [["CHỈ TIÊU", "Năm 2023"], ["Doanh thu thuần từ hoạt động tài chính khác", "1.234.000"]],
        "row_labels": [{"row_index": 0, "column_index": 0, "label": "CHỈ TIÊU"},
                       {"row_index": 1, "column_index": 0, "label": "Doanh thu thuần từ hoạt động tài chính khác"}],
    }
    consolidated = {
        "table_key": "AAA_report|60", "report_id": "AAA_report_consolidated", "source_path": "x",
        "line_position": 60, "n_cols": 2, "year": 2023, "variant": "consolidated",
        "grid": [["CHỈ TIÊU", "Năm 2023"], ["Doanh thu thuần", "1.000.000"]],
        "row_labels": [{"row_index": 0, "column_index": 0, "label": "CHỈ TIÊU"},
                       {"row_index": 1, "column_index": 0, "label": "Doanh thu thuần"}],
    }
    q = "Doanh thu thuần công ty mẹ AAA năm 2023 là bao nhiêu?"
    linked = link_schema(q, [unspecified, consolidated])
    # A definite consolidated/separate mismatch is still demoted below the weaker-scoring but
    # non-conflicting (missing-metadata) table -- unspecified metadata is not itself evidence.
    assert linked.operands[0].table_key == "AAA_report|50"


def test_link_schema_sort_order_unchanged_when_no_year_or_variant_named():
    right_year, wrong_year = _year_variant_tables()
    # No year/variant stated at all: pure lexical ranking, exactly as before tiering existed --
    # the exact-phrase "Doanh thu thuần" label must win regardless of which table holds it.
    linked = link_schema("Doanh thu thuần của AAA là bao nhiêu?", [right_year, wrong_year])
    assert linked.operands[0].table_key == "AAA_report|20"


def test_link_schema_relevance_score_value_unchanged_by_tiering():
    # Tiering reorders by conflict tier, never by inflating/deflating the stored
    # relevance_score value that generate_direct_lookup_plan's own calibrated scoring formula
    # depends on: the 2023 table sorts first (right year) despite having a strictly *lower* raw
    # relevance_score than the 2022 table (weaker label match) -- proof the reordering comes
    # from the new tier, not from relevance_score itself changing shape.
    right_year, wrong_year = _year_variant_tables()
    q = "Doanh thu thuần AAA năm 2023 là bao nhiêu?"
    linked = link_schema(q, [right_year, wrong_year])
    by_key = {x.table_key: x for x in linked.operands if x.row_label != "CHỈ TIÊU"}
    assert linked.operands[0].table_key == "AAA_report|10"
    assert by_key["AAA_report|20"].relevance_score > by_key["AAA_report|10"].relevance_score


def test_year_conflicts_with_question_treats_any_named_year_as_non_conflicting():
    table = {"year": 2022}
    # Both 2022 and 2023 named (e.g. a growth/comparison question): the 2022 table must not
    # conflict just because it isn't the first year mentioned.
    assert year_conflicts_with_question(table, "Năm nay", {2022, 2023}) is False
    assert year_conflicts_with_question(table, "Năm nay", {2021, 2023}) is True
    # A comparative-column header naming the requested year rescues an off-primary-year table.
    assert year_conflicts_with_question({"year": 2023}, "Năm 2022", {2022}) is False
    # No year named in the question at all: never a conflict.
    assert year_conflicts_with_question(table, "Năm nay", set()) is False


def test_variant_conflicts_with_question_ignores_missing_metadata():
    assert variant_conflicts_with_question({}, "separate") is False
    assert variant_conflicts_with_question({"variant": "unspecified"}, "separate") is False
    assert variant_conflicts_with_question({"variant": "consolidated"}, "separate") is True
    assert variant_conflicts_with_question({"variant": "separate"}, "separate") is False
    assert variant_conflicts_with_question({"variant": "consolidated"}, None) is False


def test_link_schema_resolves_nested_brand_ticker_via_company_registry():
    # Regression test for EXECUTION_ACCURACY_AUDIT.md §6.1 (verified live in predictions/4.json):
    # "CTCP Chứng khoán FPT" is ticker FTS, not FPT Corp -- both companies' tables can be present
    # in the retrieved candidate pool. A canonical company_by_ticker registry must resolve FTS
    # only, the same way retrieval.full_corpus.infer_question_tickers already does.
    fts = {**table_fixture(), "table_key": "FTS_report|10", "report_id": "FTS_report", "ticker": "FTS"}
    fpt = {**table_fixture(), "table_key": "FPT_report|10", "report_id": "FPT_report", "ticker": "FPT"}
    company_by_ticker = {"FTS": "CTCP Chứng khoán FPT", "FPT": "CTCP FPT"}
    linked = link_schema(
        "Lợi nhuận sau thuế của CTCP Chứng khoán FPT năm 2023 là bao nhiêu?",
        [fts, fpt], company_by_ticker=company_by_ticker,
    )
    assert linked.resolved_tickers == {"FTS"}


def test_link_schema_ticker_resolution_falls_back_unchanged_without_registry():
    # Omitting company_by_ticker must reproduce the prior literal-token-intersection behavior
    # exactly -- every existing caller that never passes it sees no behavior change.
    fts = {**table_fixture(), "table_key": "FTS_report|10", "report_id": "FTS_report", "ticker": "FTS"}
    linked = link_schema("Lợi nhuận sau thuế của CTCP Chứng khoán FPT năm 2023", [fts])
    assert linked.resolved_tickers == frozenset()
    direct = link_schema("Doanh thu thuần FTS năm 2023", [fts])
    assert direct.resolved_tickers == {"FTS"}


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

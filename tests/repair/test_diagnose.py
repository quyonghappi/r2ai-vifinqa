from repair.diagnose import (
    diagnose_failure,
    diagnose_prediction,
    trigger_t1_single_operand_collapse,
    trigger_t2_missing_entity,
    trigger_t3_missing_year,
    trigger_t5_expression_no_structure,
    trigger_t6_wrong_variant,
    trigger_t7_implausible_magnitude,
    trigger_t8_unsupported_shape,
)


def _pred(**overrides):
    base = {
        "id": 1,
        "question": "Doanh thu thuần của AAA năm 2023 là bao nhiêu?",
        "query_family": "direct_lookup",
        "used_tables": ["AAA_financial_statements_2023_consolidated|10"],
        "pandas_query": "float(df1.loc[(df1['row_index']==1) & (df1['column_index']==1), 'numeric_value'].iloc[0])",
        "answer": 1234000.0,
    }
    base.update(overrides)
    return base


def test_t1_fires_on_single_table_multi_operand_family():
    pred = _pred(query_family="comparison", used_tables=["AAA_financial_statements_2023_consolidated|10"])
    trigger = trigger_t1_single_operand_collapse(pred)
    assert trigger is not None and trigger.code == "T1"


def test_t1_does_not_fire_on_direct_lookup_with_one_table():
    pred = _pred(query_family="direct_lookup", used_tables=["AAA_financial_statements_2023_consolidated|10"])
    assert trigger_t1_single_operand_collapse(pred) is None


def test_t1_does_not_fire_when_multi_operand_family_has_two_tables():
    pred = _pred(query_family="comparison", used_tables=[
        "AAA_financial_statements_2022_consolidated|10",
        "AAA_financial_statements_2023_consolidated|10",
    ])
    assert trigger_t1_single_operand_collapse(pred) is None


def test_t2_fires_when_literal_ticker_not_in_used_tables():
    pred = _pred(question="Doanh thu thuần của BBB năm 2023",
                 used_tables=["AAA_financial_statements_2023_consolidated|10"])
    trigger = trigger_t2_missing_entity(pred)
    assert trigger is not None and trigger.code == "T2"


def test_t2_does_not_fire_when_no_literal_ticker_in_question():
    pred = _pred(question="Doanh thu thuần năm 2023 là bao nhiêu?")
    assert trigger_t2_missing_entity(pred) is None


def test_t3_fires_when_question_year_absent_from_used_tables():
    pred = _pred(question="Doanh thu thuần AAA năm 2020",
                 used_tables=["AAA_financial_statements_2023_consolidated|10"],
                 pandas_query="float(df1.loc[...].iloc[0])")
    trigger = trigger_t3_missing_year(pred)
    assert trigger is not None and trigger.code == "T3"


def test_t3_does_not_fire_when_year_present_in_used_table_report_id():
    pred = _pred(question="Doanh thu thuần AAA năm 2023",
                 used_tables=["AAA_financial_statements_2023_consolidated|10"])
    assert trigger_t3_missing_year(pred) is None


def test_t5_fires_on_comparison_with_one_referenced_cell():
    pred = _pred(query_family="comparison",
                 pandas_query="float(df1.loc[(df1['row_index']==1) & (df1['column_index']==1), 'numeric_value'].iloc[0])")
    trigger = trigger_t5_expression_no_structure(pred)
    assert trigger is not None and trigger.code == "T5"


def test_t5_does_not_fire_with_two_referenced_cells():
    pred = _pred(
        query_family="comparison",
        pandas_query=(
            "float(df1.loc[(df1['row_index']==1) & (df1['column_index']==1), 'numeric_value'].iloc[0]) - "
            "float(df1.loc[(df1['row_index']==1) & (df1['column_index']==2), 'numeric_value'].iloc[0])"
        ),
    )
    assert trigger_t5_expression_no_structure(pred) is None


def test_t6_fires_on_variant_mismatch():
    pred = _pred(question="Doanh thu thuần công ty mẹ AAA năm 2023",
                 used_tables=["AAA_financial_statements_2023_consolidated|10"])
    trigger = trigger_t6_wrong_variant(pred)
    assert trigger is not None and trigger.code == "T6"


def test_t7_fires_on_nonfinite_answer():
    pred = _pred(answer=float("nan"))
    trigger = trigger_t7_implausible_magnitude(pred)
    assert trigger is not None and trigger.code == "T7"


def test_t7_fires_on_extreme_magnitude():
    pred = _pred(answer=1e20)
    assert trigger_t7_implausible_magnitude(pred) is not None


def test_t7_does_not_fire_on_ordinary_magnitude():
    pred = _pred(answer=1234000.0)
    assert trigger_t7_implausible_magnitude(pred) is None


def test_t8_fires_on_cross_company_argmax_question():
    trigger = trigger_t8_unsupported_shape("Công ty nào có lợi nhuận cao nhất năm 2023?")
    assert trigger is not None and trigger.code == "T8"


def test_t8_does_not_fire_on_ordinary_lookup():
    assert trigger_t8_unsupported_shape("Doanh thu thuần của AAA năm 2023 là bao nhiêu?") is None


def test_diagnose_prediction_routes_unsupported_before_any_other_trigger():
    pred = _pred(question="Công ty nào có lợi nhuận cao nhất năm 2023?", query_family="direct_lookup",
                 used_tables=[])
    diagnosis = diagnose_prediction(pred)
    assert diagnosis.action == "mark_unsupported"
    assert [t.code for t in diagnosis.triggers] == ["T8"]


def test_diagnose_prediction_routes_direct_lookup_to_deterministic_rebuild():
    pred = _pred(question="Doanh thu thuần của BBB năm 2023", query_family="direct_lookup",
                 used_tables=["AAA_financial_statements_2023_consolidated|10"])
    diagnosis = diagnose_prediction(pred)
    assert diagnosis.action == "rebuild_deterministic"
    assert diagnosis.retry_budget == 1


def test_diagnose_prediction_routes_collapse_to_regenerate():
    pred = _pred(query_family="aggregation", used_tables=["AAA_financial_statements_2023_consolidated|10"])
    diagnosis = diagnose_prediction(pred)
    assert diagnosis.action == "regenerate"
    assert diagnosis.retry_budget == 2


def test_diagnose_prediction_keeps_clean_record():
    pred = _pred(query_family="comparison", used_tables=[
        "AAA_financial_statements_2022_consolidated|10", "AAA_financial_statements_2023_consolidated|10",
    ], pandas_query=(
        "float(df1.loc[(df1['row_index']==1) & (df1['column_index']==1), 'numeric_value'].iloc[0]) - "
        "float(df2.loc[(df2['row_index']==1) & (df2['column_index']==1), 'numeric_value'].iloc[0])"
    ))
    diagnosis = diagnose_prediction(pred)
    assert diagnosis.action == "keep"
    assert diagnosis.triggers == []


def test_diagnose_failure_routes_unit_ambiguity_to_deterministic_resolution():
    fail = {
        "id": 109, "question": "Doanh thu thuần của AAA năm 2023",
        "error_type": "RuntimeError",
        "error_message": "all generation/repair attempts failed: [{'attempt': 0, 'error_type': "
                          "'ValueError', 'error_message': 'selected direct-lookup column has no "
                          "unambiguous source unit'}]",
    }
    diagnosis = diagnose_failure(fail)
    assert diagnosis.action == "resolve_unit_deterministic"
    assert diagnosis.retry_budget == 1


def test_diagnose_failure_routes_json_error_to_regenerate():
    fail = {
        "id": 1002, "question": "Doanh thu thuần của AAA năm 2023",
        "error_type": "RuntimeError",
        "error_message": "all generation/repair attempts failed: [{'attempt': 0, 'error_type': "
                          "'JSONDecodeError', 'error_message': \"Expecting ',' delimiter\"}]",
    }
    diagnosis = diagnose_failure(fail)
    assert diagnosis.action == "regenerate"
    assert diagnosis.retry_budget == 2


def test_diagnose_failure_routes_missing_company_to_relink():
    fail = {
        "id": 1006, "question": "Trung bình thu nhập của NAB, ABB, ACB và STB",
        "error_type": "RuntimeError",
        "error_message": "all generation/repair attempts failed: [{'attempt': 0, 'error_type': "
                          "'ValueError', 'error_message': \"plan is missing required companies: "
                          "['STB']\"}]",
    }
    diagnosis = diagnose_failure(fail)
    assert diagnosis.action == "relink_then_regenerate"


def test_diagnose_failure_marks_unsupported_argmax_regardless_of_error():
    fail = {
        "id": 2001, "question": "Công ty nào có doanh thu cao nhất năm 2023?",
        "error_type": "RuntimeError", "error_message": "all generation/repair attempts failed: []",
    }
    diagnosis = diagnose_failure(fail)
    assert diagnosis.action == "mark_unsupported"

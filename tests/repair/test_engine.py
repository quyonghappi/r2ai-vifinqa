import math

from repair.common import Diagnosis, RepairTrigger
from repair.engine import (
    evaluate_candidate,
    rebuild_direct_lookup_ignoring_target_unit,
    recompute_schema_linking,
    repair_one,
)
from schema_linking.linker import link_schema


def _table(**overrides):
    base = {
        "table_key": "AAA_financial_statements_2023_consolidated|10",
        "report_id": "AAA_financial_statements_2023_consolidated",
        "source_path": "financial_statements/AAA/2023/report.txt",
        "line_position": 10, "n_cols": 3, "detected_units": ["VND"],
        "ticker": "AAA", "year": 2023, "variant": "consolidated",
        "grid": [["CHỈ TIÊU", "Năm 2023", "Năm 2022"], ["Doanh thu thuần", "1.234.000", "1.000.000"]],
        "row_labels": [{"row_index": 0, "column_index": 0, "label": "CHỈ TIÊU"},
                       {"row_index": 1, "column_index": 0, "label": "Doanh thu thuần"}],
    }
    base.update(overrides)
    return base


def test_recompute_schema_linking_reuses_link_schema():
    table = _table()
    tables_by_key = {table["table_key"]: table}
    linked = recompute_schema_linking("Doanh thu thuần AAA năm 2023", [table["table_key"]], tables_by_key)
    assert linked.query_family == "direct_lookup"
    assert any(op.table_key == table["table_key"] for op in linked.operands)


def test_rebuild_direct_lookup_ignoring_target_unit_never_raises_on_ambiguous_scale():
    # No column_metadata/detected_units gives an ambiguous source scale -- the ordinary
    # generate_direct_lookup_plan would raise "no unambiguous source unit" once a requested_unit
    # is set; the repair-local fallback must still produce a grounded, executable plan.
    table = _table(detected_units=["VND", "Triệu đồng"])  # 2 distinct scales -> genuinely ambiguous
    tables_by_key = {table["table_key"]: table}
    linked = link_schema("Doanh thu thuần AAA năm 2023", [table])
    linked.requested_unit = "tỷ đồng"
    plan = rebuild_direct_lookup_ignoring_target_unit(linked, tables_by_key)
    assert plan.operands[0].row_index == 1
    assert plan.expression == "x"
    assert plan.answer_unit is None


def test_repair_one_mark_unsupported_never_touches_llm_or_evidence():
    diagnosis = Diagnosis(1, [RepairTrigger("T8", "argmax", "high")], "mark_unsupported", 0)
    result = repair_one(1, "Công ty nào cao nhất?", [], {}, diagnosis, complete=None)
    assert result == {"prediction": None, "requires_llm": False, "skipped": True}


def test_repair_one_keep_is_a_noop():
    diagnosis = Diagnosis(1, [], "keep", 0)
    result = repair_one(1, "q", [], {}, diagnosis, complete=None)
    assert result["skipped"] is True and result["prediction"] is None


def test_repair_one_rebuild_deterministic_produces_executed_prediction():
    table = _table()
    tables_by_key = {table["table_key"]: table}
    diagnosis = Diagnosis(1, [RepairTrigger("T2", "ticker mismatch", "medium")], "rebuild_deterministic", 1)
    result = repair_one(1, "Doanh thu thuần AAA năm 2023", [table["table_key"]], tables_by_key, diagnosis, complete=None)
    assert result["requires_llm"] is False
    pred = result["prediction"]
    assert pred["executed"] is True
    assert pred["answer"] == 1234000.0
    assert pred["used_tables"] == [table["table_key"]]


def test_repair_one_rebuild_deterministic_never_calls_the_llm_even_if_one_is_supplied():
    table = _table()
    tables_by_key = {table["table_key"]: table}
    diagnosis = Diagnosis(1, [], "rebuild_deterministic", 1)

    def exploding_complete(prompt):
        raise AssertionError("must never be called for a deterministic rebuild")

    result = repair_one(1, "Doanh thu thuần AAA năm 2023", [table["table_key"]], tables_by_key, diagnosis, complete=exploding_complete)
    assert result["prediction"]["executed"] is True


def test_repair_one_resolve_unit_deterministic():
    table = _table(detected_units=["VND", "Triệu đồng"])
    tables_by_key = {table["table_key"]: table}
    diagnosis = Diagnosis(109, [RepairTrigger("T9", "unit ambiguity", "high")], "resolve_unit_deterministic", 1)
    result = repair_one(109, "Doanh thu thuần AAA năm 2023 là bao nhiêu tỷ đồng?", [table["table_key"]], tables_by_key, diagnosis, complete=None)
    assert result["prediction"]["executed"] is True
    assert result["prediction"]["answer"] == 1234000.0


def test_repair_one_regenerate_without_a_model_reports_requires_llm():
    diagnosis = Diagnosis(1, [RepairTrigger("T1", "collapse", "high")], "regenerate", 2)
    result = repair_one(1, "q", [], {}, diagnosis, complete=None)
    assert result == {"prediction": None, "requires_llm": True, "skipped": False}


def test_repair_one_regenerate_with_a_mock_complete_produces_a_grouped_plan():
    aaa = _table()
    bbb = _table(table_key="BBB_financial_statements_2023_consolidated|10",
                 report_id="BBB_financial_statements_2023_consolidated", ticker="BBB")
    tables_by_key = {aaa["table_key"]: aaa, bbb["table_key"]: bbb}

    import json as _json

    def fake_complete(prompt: str) -> str:
        payload = {
            "operands": [
                {"alias": "a", "table_key": aaa["table_key"], "row_index": 1, "column_index": 1},
                {"alias": "b", "table_key": bbb["table_key"], "row_index": 1, "column_index": 1},
            ],
            "expression": "a - b", "answer_unit": None, "rationale": "diff",
        }
        return _json.dumps(payload)

    diagnosis = Diagnosis(1, [RepairTrigger("T1", "collapse", "high")], "regenerate", 2)
    result = repair_one(
        1, "Chênh lệch doanh thu giữa AAA và BBB năm 2023", [aaa["table_key"], bbb["table_key"]],
        tables_by_key, diagnosis, complete=fake_complete,
    )
    pred = result["prediction"]
    assert pred["executed"] is True
    assert set(pred["used_tables"]) == {aaa["table_key"], bbb["table_key"]}


def test_evaluate_candidate_rejects_nonexecuted_candidate():
    outcome = evaluate_candidate(1, None, {"executed": False})
    assert outcome.accepted is False


def test_evaluate_candidate_rejects_nonfinite_answer():
    outcome = evaluate_candidate(1, None, {"executed": True, "answer": math.nan})
    assert outcome.accepted is False


def test_evaluate_candidate_accepts_clean_failure_recovery():
    candidate = {
        "id": 109, "question": "Doanh thu thuần AAA năm 2023", "query_family": "direct_lookup",
        "used_tables": ["AAA_financial_statements_2023_consolidated|10"], "executed": True,
        "answer": 1234000.0, "pandas_query": "float(df1.loc[...].iloc[0])",
    }
    outcome = evaluate_candidate(109, None, candidate)
    assert outcome.accepted is True
    assert outcome.new_prediction is candidate


def test_evaluate_candidate_rejects_failure_recovery_with_open_trigger():
    candidate = {
        "id": 1, "question": "Doanh thu thuần của BBB năm 2023", "query_family": "direct_lookup",
        "used_tables": ["AAA_financial_statements_2023_consolidated|10"], "executed": True,
        "answer": 1234000.0, "pandas_query": "float(df1.loc[...].iloc[0])",
    }
    outcome = evaluate_candidate(1, None, candidate)
    assert outcome.accepted is False
    assert "open trigger" in outcome.reason


def test_evaluate_candidate_rejects_noop_identical_query():
    original = {
        "id": 1, "question": "q", "query_family": "aggregation",
        "used_tables": ["AAA_financial_statements_2023_consolidated|10"], "executed": True,
        "answer": 1.0, "pandas_query": "float(df1.loc[...].iloc[0])",
    }
    candidate = dict(original)
    outcome = evaluate_candidate(1, original, candidate)
    assert outcome.accepted is False
    assert "no-op" in outcome.reason


def test_evaluate_candidate_accepts_strictly_fewer_triggers():
    original = {
        "id": 1, "question": "q", "query_family": "aggregation",
        "used_tables": ["AAA_financial_statements_2023_consolidated|10"], "executed": True,
        "answer": 1.0, "pandas_query": "float(df1.loc[(df1['row_index']==1) & (df1['column_index']==1), 'numeric_value'].iloc[0])",
    }
    candidate = dict(original, used_tables=[
        "AAA_financial_statements_2023_consolidated|10", "BBB_financial_statements_2023_consolidated|10",
    ], pandas_query=(
        "float(df1.loc[(df1['row_index']==1) & (df1['column_index']==1), 'numeric_value'].iloc[0]) + "
        "float(df2.loc[(df2['row_index']==1) & (df2['column_index']==1), 'numeric_value'].iloc[0])"
    ), answer=2.0)
    outcome = evaluate_candidate(1, original, candidate)
    assert outcome.accepted is True


def test_evaluate_candidate_rejects_when_new_trigger_count_not_strictly_lower():
    original = {
        "id": 1, "question": "q", "query_family": "aggregation",
        "used_tables": ["AAA_financial_statements_2023_consolidated|10"], "executed": True,
        "answer": 1.0, "pandas_query": "float(df1.loc[...].iloc[0])",
    }
    candidate = dict(original, pandas_query="float(df2.loc[...].iloc[0])", used_tables=["BBB_financial_statements_2023_consolidated|10"])
    outcome = evaluate_candidate(1, original, candidate)
    assert outcome.accepted is False
    assert "did not strictly reduce" in outcome.reason

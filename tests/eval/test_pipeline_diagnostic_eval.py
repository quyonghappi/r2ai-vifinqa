from eval.pipeline_diagnostic_eval import (
    GenerationNotExercised,
    classify_validation_error,
    deterministic_complete_stub,
    evaluate_records,
)
from tests.schema_linking.test_linker import table_fixture


def _no_numeric_column_table():
    # "Ghi chú" is in schema_linking.linker.is_numeric_data_column's excluded-role set, so every
    # cell in this table is filtered out before it ever becomes an operand candidate -- a real,
    # reproducible way to make a *retrieved* table contribute zero schema_linked_operands.
    return {
        "table_key": "CCC_report|30", "report_id": "CCC_report", "source_path": "x", "line_position": 30,
        "n_cols": 2, "detected_units": [],
        "grid": [["CHỈ TIÊU", "Ghi chú"], ["Some label", "12345"]],
        "row_labels": [
            {"row_index": 0, "column_index": 0, "label": "CHỈ TIÊU"},
            {"row_index": 1, "column_index": 0, "label": "Some label"},
        ],
    }


def _fixture_and_tables():
    table_a = table_fixture()  # AAA_report|10, "Doanh thu thuần" = 1234000.0 (2023) / 1000000.0 (2022)
    table_c = _no_numeric_column_table()
    fixture = [
        {
            "id": 101, "question": "Doanh thu thuần AAA năm 2023",
            "pipeline_family": "direct_lookup", "question_shape": "direct_lookup",
            "gold": {"tables": [table_a["table_key"]], "expected_answer": 1234000.0,
                     "answer_tolerance": {"abs_tol": 1e-6, "rel_tol": 1e-6}},
        },
        {
            "id": 102, "question": "Doanh thu thuần ZZZ năm 2023",
            "pipeline_family": "direct_lookup", "question_shape": "direct_lookup",
            "gold": {"tables": ["ZZZ_report|99"], "expected_answer": 1.0},
        },
        {
            "id": 103, "question": "Doanh thu thuần CCC năm 2023",
            "pipeline_family": "direct_lookup", "question_shape": "direct_lookup",
            "gold": {"tables": [table_c["table_key"]], "expected_answer": 1.0},
        },
        {
            "id": 104, "question": "Tổng số doanh thu thuần AAA năm 2023",
            "pipeline_family": "aggregation", "question_shape": "aggregation",
            "gold": {"tables": [table_a["table_key"]], "expected_answer": 999.0},
        },
    ]
    retrieved_by_id = {
        101: [table_a["table_key"]],
        102: [],  # nothing retrieved -- the gold table is missing at the retrieval stage
        103: [table_c["table_key"]],
        104: [table_a["table_key"]],
    }
    tables_by_key = {table_a["table_key"]: table_a, table_c["table_key"]: table_c}
    return fixture, retrieved_by_id, tables_by_key


def _by_id(report, qid):
    return next(pq for pq in report["per_question"] if pq["id"] == qid)


def test_perfect_direct_lookup_is_correct_and_never_invokes_the_stub():
    fixture, retrieved_by_id, tables_by_key = _fixture_and_tables()
    report = evaluate_records([fixture[0]], {101: retrieved_by_id[101]}, tables_by_key)
    row = _by_id(report, 101)
    assert row["outcome"] == "CORRECT"
    assert row["first_incorrect_stage"] == "correct"
    assert row["answer_correct"] is True
    assert row["missing_at_retrieval"] == []
    assert row["family_classification_matches"] is True


def test_retrieval_miss_is_flagged_before_downstream_failure_is_even_considered():
    fixture, retrieved_by_id, tables_by_key = _fixture_and_tables()
    report = evaluate_records([fixture[1]], {102: retrieved_by_id[102]}, tables_by_key)
    row = _by_id(report, 102)
    assert row["first_incorrect_stage"] == "retrieval"
    assert row["missing_at_retrieval"] == ["ZZZ_report|99"]
    assert row["executed"] is False  # also failed downstream, but retrieval is reported first


def test_schema_linking_miss_when_retrieved_table_yields_zero_operands():
    fixture, retrieved_by_id, tables_by_key = _fixture_and_tables()
    report = evaluate_records([fixture[2]], {103: retrieved_by_id[103]}, tables_by_key)
    row = _by_id(report, 103)
    assert row["missing_at_retrieval"] == []
    assert row["first_incorrect_stage"] == "schema_linking"
    assert row["missing_at_schema_linking"] == ["CCC_report|30"]


def test_non_direct_lookup_family_reports_generation_not_exercised_not_a_faked_score():
    fixture, retrieved_by_id, tables_by_key = _fixture_and_tables()
    calls = []

    def counting_stub(prompt):
        calls.append(prompt)
        raise GenerationNotExercised("stub invoked")

    report = evaluate_records([fixture[3]], {104: retrieved_by_id[104]}, tables_by_key, counting_stub)
    row = _by_id(report, 104)
    assert row["actual_pipeline_family"] == "aggregation"  # "tổng số" triggers this
    assert row["outcome"] == "GENERATION_OR_VALIDATION_FAILED"
    assert len(calls) > 0  # the stub WAS invoked here, unlike the direct_lookup case
    assert report["answer_execution_metrics"]["n_exercised_direct_lookup_with_expected_answer"] == 0
    assert report["answer_execution_metrics"]["n_not_exercised_generation_required"] == 1
    assert report["answer_execution_metrics"]["not_exercised_ids"] == [104]
    assert report["answer_execution_metrics"]["metrics"] is None


def test_deterministic_stub_raises_when_actually_invoked():
    try:
        deterministic_complete_stub("some prompt")
        assert False, "expected GenerationNotExercised"
    except GenerationNotExercised:
        pass


def test_full_report_aggregates_retrieval_metrics_and_failure_taxonomy():
    fixture, retrieved_by_id, tables_by_key = _fixture_and_tables()
    report = evaluate_records(fixture, retrieved_by_id, tables_by_key)
    assert report["n_questions"] == 4
    assert report["retrieval_metrics"]["n_queries"] == 4
    assert report["failure_taxonomy"]["CORRECT"] == 1
    assert report["first_incorrect_stage_histogram"]["retrieval"] == 1
    assert report["first_incorrect_stage_histogram"]["schema_linking"] == 1
    # 101/102/103 all classify as direct_lookup (real classifier output) and all carry an
    # expected_answer, so all 3 are "exercised" -- only 104 (aggregation) is not. Exercised
    # includes failing records too; only 101 actually answers correctly.
    metrics = report["answer_execution_metrics"]
    assert metrics["n_exercised_direct_lookup_with_expected_answer"] == 3
    assert metrics["n_not_exercised_generation_required"] == 1
    assert metrics["not_exercised_ids"] == [104]
    assert metrics["metrics"]["answer_accuracy"] == 1 / 3
    assert metrics["metrics"]["execution_accuracy"] == 1 / 3


def test_classify_validation_error_maps_known_messages():
    assert classify_validation_error("incomplete operand group(s): missing company X") == "incomplete_operand_group"
    assert classify_validation_error("invented or unresolved operand: BoundOperand(...)") == "invented_or_unresolved_operand"
    assert classify_validation_error(None) is None
    assert classify_validation_error("some unrelated TypeError") is None

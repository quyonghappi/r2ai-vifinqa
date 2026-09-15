import json

from pipeline import answer_question
from tests.schema_linking.test_linker import table_fixture


def test_pipeline_keeps_retrieved_linked_used_and_attempt_provenance_separate():
    table = table_fixture()
    def complete(_prompt):
        return json.dumps({
            "operands": [{"alias": "x", "table_key": table["table_key"], "row_index": 1, "column_index": 1}],
            "expression": "x", "answer_unit": "VND", "rationale": "direct lookup",
        })
    result = answer_question(
        {"id": 1, "question": "Doanh thu thuần AAA năm 2023"},
        [table["table_key"]], {table["table_key"]: table}, complete,
    )
    assert result["executed"] and result["answer"] == 1234000.0
    assert result["retrieved_tables"] == result["used_tables"]
    assert result["relevant_docs"] == ["AAA_report"]
    assert result["attempts"][0]["error_type"] is None


def test_relevant_tables_can_exceed_used_tables_when_more_candidates_are_retrieved():
    # relevant_tables is schema-linking's recall-oriented candidate judgment; used_tables stays
    # the narrow execution-provenance set. With 2 retrieved tables but the generated plan only
    # indexing 1 of them, relevant_tables must still surface both (both carry a real operand),
    # while used_tables/relevant_docs stay exactly what pandas_query touched.
    table_a = table_fixture()
    table_b = {**table_fixture(), "table_key": "BBB_report|20", "report_id": "BBB_report"}

    def complete(_prompt):
        return json.dumps({
            "operands": [{"alias": "x", "table_key": table_a["table_key"], "row_index": 1, "column_index": 1}],
            "expression": "x", "answer_unit": "VND", "rationale": "direct lookup",
        })

    result = answer_question(
        {"id": 2, "question": "Doanh thu thuần AAA năm 2023"},
        [table_a["table_key"], table_b["table_key"]],
        {table_a["table_key"]: table_a, table_b["table_key"]: table_b},
        complete,
    )
    # The deterministic direct_lookup path picks exactly one candidate by its own scoring
    # (table_key is a tie-break, not something this test should assume the direction of).
    assert len(result["used_tables"]) == 1
    assert result["used_tables"][0] in {table_a["table_key"], table_b["table_key"]}
    assert set(result["relevant_tables"]) == {table_a["table_key"], table_b["table_key"]}
    assert set(result["relevant_docs"]) == {"AAA_report", "BBB_report"}

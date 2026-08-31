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

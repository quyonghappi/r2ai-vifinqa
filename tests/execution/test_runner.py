import json

import pandas as pd
import pytest

from execution.runner import execute_query, execute_with_repair
from schema_linking.linker import link_schema
from tests.schema_linking.test_linker import table_fixture


def test_execute_query_happy_path_and_malformed_query():
    frames = {"df1": pd.DataFrame({"x": [2.0]})}
    assert execute_query("float(df1['x'].iloc[0]) + 3", frames) == 5.0
    with pytest.raises(KeyError):
        execute_query("float(df1['missing'].iloc[0])", frames)


def test_explicit_repair_loop_records_validation_error_then_succeeds():
    table = table_fixture()
    linked = link_schema("Doanh thu thuần AAA năm 2023", [table])
    linked.query_family = "comparison"  # exercise the model-backed repair loop, not direct lookup
    outputs = iter([
        json.dumps({"operands": [{"alias": "x", "table_key": "AAA_report|10", "row_index": 999, "column_index": 1}], "expression": "x"}),
        json.dumps({"operands": [{"alias": "x", "table_key": "AAA_report|10", "row_index": 1, "column_index": 1}], "expression": "x"}),
    ])
    prompts = []
    def complete(prompt):
        prompts.append(prompt)
        return next(outputs)

    result = execute_with_repair(linked, {table["table_key"]: table}, complete, max_retries=1)
    assert result.success and result.answer == 1234000.0
    assert len(result.attempts) == 2
    assert result.attempts[0].error_type == "ValueError"
    assert "invented" in prompts[1]


def test_repair_loop_persists_traceback_and_stage_for_empty_attribute_error():
    table = table_fixture()
    linked = link_schema("AAA", [table])

    linked.query_family = "comparison"  # exercise model failure capture, not direct lookup

    def complete(_prompt):
        raise AttributeError()

    result = execute_with_repair(linked, {table["table_key"]: table}, complete, max_retries=0)
    attempt = result.attempts[0]
    assert not result.success
    assert attempt.stage == "generation_or_validation"
    assert attempt.error_type == "AttributeError"
    assert "AttributeError" in attempt.traceback


def test_repair_loop_does_not_retry_an_out_of_memory_error():
    table = table_fixture()
    linked = link_schema("Doanh thu thuần AAA năm 2023", [table])
    linked.query_family = "comparison"
    calls = 0

    class OutOfMemoryError(RuntimeError):
        pass

    def complete(_prompt):
        nonlocal calls
        calls += 1
        raise OutOfMemoryError("CUDA out of memory")

    result = execute_with_repair(linked, {table["table_key"]: table}, complete, max_retries=2)
    assert not result.success
    assert calls == 1
    assert len(result.attempts) == 1

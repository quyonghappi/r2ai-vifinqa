import sys
from contextlib import nullcontext

import pytest

from query_generation.generator import (
    QwenAWQGenerator,
    build_evidence_frames,
    build_prompt,
    generate_direct_lookup_plan,
    render_pandas_query,
    validate_plan,
)
from schema_linking.linker import link_schema
from tests.schema_linking.test_linker import table_fixture


def test_grounded_plan_renders_standalone_pandas_query():
    table = table_fixture()
    linked = link_schema("Doanh thu thuần AAA năm 2023", [table])
    payload = {"operands": [{"alias": "revenue", "table_key": "AAA_report|10", "row_index": 1, "column_index": 1}],
               "expression": "revenue / 1000", "answer_unit": "nghìn VND", "rationale": "lookup and convert"}
    plan = validate_plan(payload, linked)
    frames, variables = build_evidence_frames(plan, {table["table_key"]: table})
    query = render_pandas_query(plan, variables)
    assert "df1" in query and "iloc[0]" in query
    assert eval(query, {"float": float}, frames) == 1234.0


def test_direct_lookup_plan_uses_a_real_data_cell_and_requested_unit():
    table = table_fixture()
    linked = link_schema("Doanh thu thuáº§n AAA nÄƒm 2023", [table])
    plan = generate_direct_lookup_plan(linked, {table["table_key"]: table})
    assert plan.operands[0].row_index == 1
    assert plan.operands[0].column_index == 1
    assert plan.expression == "x"


def test_direct_lookup_uses_selected_columns_unit_metadata():
    table = table_fixture()
    table["column_metadata"] = [
        {"column_index": 0, "scale_to_vnd": None},
        {"column_index": 1, "scale_to_vnd": 1_000_000},
        {"column_index": 2, "scale_to_vnd": 1},
    ]
    linked = link_schema("Doanh thu thuáº§n AAA nÄƒm 2023", [table])
    linked.requested_unit = "VND"
    plan = generate_direct_lookup_plan(linked, {table["table_key"]: table})
    assert plan.expression == "x * 1000000.0"


def test_direct_lookup_rederives_column_unit_when_persisted_scale_is_stale():
    table = table_fixture()
    table["grid"] = [["Metric", "Year 2023 VND"], ["Revenue", "2.000.000"]]
    table["n_cols"] = 2
    table["row_labels"] = [{"row_index": 0, "column_index": 0, "label": "Metric"},
                           {"row_index": 1, "column_index": 0, "label": "Revenue"}]
    table["column_metadata"] = [{"column_index": 1, "source_unit": "stale", "scale_to_vnd": 1_000_000_000}]
    linked = link_schema("Revenue AAA 2023 VND", [table])
    linked.requested_unit = "VND"
    plan = generate_direct_lookup_plan(linked, {table["table_key"]: table})
    assert plan.expression == "x"


def test_direct_lookup_does_not_repeat_retrieval_entity_resolution():
    table = table_fixture()
    table["ticker"] = "AAA"
    linked = link_schema("Doanh thu thuần BBB năm 2023", [table])
    # Entity resolution belongs to retrieval; a legal company name can contain an unrelated
    # listed ticker token (for example, FTS/FPT). Direct lookup must not reject its evidence.
    plan = generate_direct_lookup_plan(linked, {table["table_key"]: table})
    assert plan.operands[0].table_key == table["table_key"]


def test_direct_lookup_filters_literal_ticker_when_retrieval_contains_it():
    aaa = table_fixture()
    aaa["ticker"] = "AAA"
    bbb = {**table_fixture(), "table_key": "BBB_report|10", "report_id": "BBB_report", "ticker": "BBB"}
    linked = link_schema("Revenue BBB 2023", [aaa, bbb])
    plan = generate_direct_lookup_plan(linked, {aaa["table_key"]: aaa, bbb["table_key"]: bbb})
    assert plan.operands[0].table_key == bbb["table_key"]


def test_direct_lookup_filters_alphanumeric_ticker_mention():
    # HT1 is a real ticker with a trailing digit (see TICKER_TOKEN_RE); a named-ticker filter
    # that used a letters-only regex would never recognize "HT1" here and would fall through to
    # the no-ticker-mentioned branch, silently accepting evidence from any retrieved company.
    ht1 = table_fixture()
    ht1["ticker"] = "HT1"
    other = {**table_fixture(), "table_key": "OTHER_report|10", "report_id": "OTHER_report", "ticker": "ZZZ"}
    linked = link_schema("Doanh thu thuần HT1 năm 2023", [ht1, other])
    plan = generate_direct_lookup_plan(linked, {ht1["table_key"]: ht1, other["table_key"]: other})
    assert plan.operands[0].table_key == ht1["table_key"]


def test_direct_lookup_hard_filters_year_and_explicit_closing_period():
    table_2022 = table_fixture()
    table_2022.update({"table_key": "AAA_2022|10", "report_id": "AAA_2022", "year": 2022, "ticker": "AAA"})
    table_2023 = table_fixture()
    table_2023.update({"table_key": "AAA_2023|10", "report_id": "AAA_2023", "year": 2023, "ticker": "AAA"})
    table_2023["grid"] = [["CHỈ TIÊU", "Số đầu năm", "Số cuối năm"], ["Tiền", "100", "200"]]
    table_2023["row_labels"] = [{"row_index": 0, "column_index": 0, "label": "CHỈ TIÊU"}, {"row_index": 1, "column_index": 0, "label": "Tiền"}]
    linked = link_schema("Tiền AAA cuối năm 2023", [table_2022, table_2023])
    plan = generate_direct_lookup_plan(linked, {x["table_key"]: x for x in [table_2022, table_2023]})
    assert plan.operands[0].table_key == "AAA_2023|10"
    assert plan.operands[0].column_index == 2


def test_plan_cannot_invent_operand_coordinates():
    linked = link_schema("Doanh thu thuần AAA năm 2023", [table_fixture()])
    payload = {"operands": [{"alias": "x", "table_key": "AAA_report|10", "row_index": 99, "column_index": 1}],
               "expression": "x", "answer_unit": None}
    with pytest.raises(ValueError, match="invented"):
        validate_plan(payload, linked)


def test_expression_cannot_access_python_or_unknown_names():
    linked = link_schema("Doanh thu thuần AAA năm 2023", [table_fixture()])
    payload = {"operands": [{"alias": "x", "table_key": "AAA_report|10", "row_index": 1, "column_index": 1}],
               "expression": "__import__('os')", "answer_unit": None}
    with pytest.raises(ValueError):
        validate_plan(payload, linked)


def test_qwen_adapter_handles_transformers_batch_encoding(monkeypatch):
    class Tensor:
        def __init__(self, rows):
            self.rows = rows
            self.shape = (len(rows), len(rows[0]))

        def to(self, _device):
            return self

        def __getitem__(self, index):
            if isinstance(index, slice):
                return Tensor([self.rows[0][index]])
            return Tensor([self.rows[index]])

        def tolist(self):
            return self.rows[0]

    class BatchEncoding(dict):
        """Minimal mapping-shaped stand-in for Transformers 5 BatchEncoding."""
        def to(self, device):
            return BatchEncoding({key: value.to(device) for key, value in self.items()})

    class Tokenizer:
        model_max_length = 1024

        def apply_chat_template(self, *_args, **_kwargs):
            return BatchEncoding({"input_ids": Tensor([[1, 2, 3]])})

        def decode(self, tokens, **_kwargs):
            return str(tokens.tolist())

    class Model:
        class config:
            max_position_embeddings = 1024

        def get_input_embeddings(self):
            return type("Embedding", (), {"weight": type("Weight", (), {"device": "cpu"})()})()

        def generate(self, **inputs):
            return Tensor([inputs["input_ids"].rows[0] + [4]])

    fake_torch = type("Torch", (), {"inference_mode": staticmethod(nullcontext)})()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    generator = object.__new__(QwenAWQGenerator)
    generator.tokenizer = Tokenizer()
    generator.model = Model()
    assert generator("prompt") == "[4]"
    assert generator.last_prompt_diagnostics["input_tokens"] == 3


def test_qwen_prompt_fit_limits_operands_by_token_budget(monkeypatch):
    class Tensor:
        def __init__(self, token_count):
            self.shape = (1, token_count)

    class BatchEncoding(dict):
        pass

    class Tokenizer:
        model_max_length = 1024

        def apply_chat_template(self, messages, **_kwargs):
            # The schema has no raw_value; each exposed operand has exactly one.
            return BatchEncoding({"input_ids": Tensor(10 + messages[0]["content"].count('"raw_value"') * 100)})

    class Model:
        class config:
            max_position_embeddings = 1024

    linked = link_schema("Doanh thu thuần AAA năm 2023", [table_fixture()])
    generator = object.__new__(QwenAWQGenerator)
    generator.tokenizer = Tokenizer()
    generator.model = Model()
    monkeypatch.setenv("VIFINQA_MAX_INPUT_TOKENS", "150")
    _prompt, inputs = generator._fit_prompt(build_prompt(linked))
    assert inputs["input_ids"].shape[-1] == 110
    assert len(generator.visible_operands) == 1

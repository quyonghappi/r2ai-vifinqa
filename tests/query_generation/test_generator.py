import sys
from contextlib import nullcontext

import pytest

from execution.runner import execute_query, execute_with_repair
from query_generation.generator import (
    QwenAWQGenerator,
    build_evidence_frames,
    build_prompt,
    compute_reductions,
    detect_multi_year_extremum,
    generate_direct_lookup_plan,
    generate_extremum_period_plan,
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
    plan = validate_plan(payload, linked, {table["table_key"]: table})
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


def test_direct_lookup_resolves_nested_brand_ticker_via_canonical_resolution():
    # Regression test for EXECUTION_ACCURACY_AUDIT.md §6.1, verified live in predictions/4.json:
    # both FTS ("CTCP Chứng khoán FPT") and FPT Corp tables can be in the retrieved candidate
    # pool at once; the canonical resolution must pick FTS, not fall through to "no filtering"
    # the way a bare literal-text scan does.
    fts = table_fixture()
    fts.update({"table_key": "FTS_report|10", "report_id": "FTS_report", "ticker": "FTS"})
    fpt = {**table_fixture(), "table_key": "FPT_report|10", "report_id": "FPT_report", "ticker": "FPT"}
    company_by_ticker = {"FTS": "CTCP Chứng khoán FPT", "FPT": "CTCP FPT"}
    linked = link_schema(
        "Lợi nhuận sau thuế của CTCP Chứng khoán FPT năm 2023 là bao nhiêu?",
        [fts, fpt], company_by_ticker=company_by_ticker,
    )
    plan = generate_direct_lookup_plan(linked, {fts["table_key"]: fts, fpt["table_key"]: fpt})
    assert plan.operands[0].table_key == fts["table_key"]


def _year_table(ticker: str, year: int, value: str) -> dict:
    return {
        "table_key": f"{ticker}_{year}|10", "report_id": f"{ticker}_{year}", "source_path": "x",
        "line_position": 10, "n_cols": 2, "year": year, "variant": "consolidated", "ticker": ticker,
        "grid": [["CHỈ TIÊU", f"Năm {year}"], ["Vốn chủ sở hữu", value]],
        "row_labels": [{"row_index": 0, "column_index": 0, "label": "CHỈ TIÊU"},
                       {"row_index": 1, "column_index": 0, "label": "Vốn chủ sở hữu"}],
    }


def test_detect_multi_year_extremum_recognizes_which_year_shape():
    # Regression test for EXECUTION_ACCURACY_AUDIT.md §6.4, real fixture id 959.
    extremum = detect_multi_year_extremum(
        "Vào cuối năm nào trong các năm 2016, 2017, 2018 và 2020, vốn chủ sở hữu của GVR đạt mức cao nhất?"
    )
    assert extremum == {"years": {2016, 2017, 2018, 2020}, "direction": "max", "mode": "year"}


def test_detect_multi_year_extremum_recognizes_extreme_value_shape():
    # Regression test for EXECUTION_ACCURACY_AUDIT.md §6.4, real fixture id 961: unlike 959, the
    # requested answer is the extreme value itself, not the year that achieved it.
    extremum = detect_multi_year_extremum(
        "Tổng giá trị hàng tồn kho cuối kỳ lớn nhất của BAF trong các năm 2020, 2022 và 2024 là bao nhiêu?"
    )
    assert extremum == {"years": {2020, 2022, 2024}, "direction": "max", "mode": "value"}


def test_detect_multi_year_extremum_ignores_single_year_or_comparison_questions():
    # Narrow trigger: never fires for an ordinary lookup or a 2-thing comparison, both of which
    # already route correctly elsewhere.
    assert detect_multi_year_extremum("Vốn chủ sở hữu AAA năm 2023 là bao nhiêu?") is None
    assert detect_multi_year_extremum("AAA cao hơn BBB bao nhiêu năm 2023?") is None


def test_generate_extremum_period_plan_returns_winning_year():
    tables = [_year_table("GVR", y, v) for y, v in zip((2016, 2017, 2018, 2020), ("100", "200", "150", "300"))]
    linked = link_schema(
        "Vào cuối năm nào trong các năm 2016, 2017, 2018 và 2020, vốn chủ sở hữu của GVR đạt mức cao nhất?",
        tables,
    )
    tables_by_key = {t["table_key"]: t for t in tables}
    extremum = detect_multi_year_extremum(linked.question)
    plan = generate_extremum_period_plan(linked, tables_by_key, extremum)
    assert plan.operands[0].table_key == "GVR_2020|10"  # evidence points at the real winning table
    frames, variables = build_evidence_frames(plan, tables_by_key)
    query = render_pandas_query(plan, variables)
    assert eval(query, {"float": float}, frames) == 2020.0  # noqa: S307


def test_generate_extremum_period_plan_returns_extreme_value():
    tables = [_year_table("BAF", y, v) for y, v in zip((2020, 2022, 2024), ("100", "300", "200"))]
    linked = link_schema(
        "Giá trị hàng tồn kho lớn nhất của BAF trong các năm 2020, 2022 và 2024 là bao nhiêu?",
        tables,
    )
    tables_by_key = {t["table_key"]: t for t in tables}
    extremum = detect_multi_year_extremum(linked.question)
    plan = generate_extremum_period_plan(linked, tables_by_key, extremum)
    assert plan.operands[0].table_key == "BAF_2022|10"
    frames, variables = build_evidence_frames(plan, tables_by_key)
    query = render_pandas_query(plan, variables)
    assert eval(query, {"float": float}, frames) == 300.0  # noqa: S307


def test_execute_with_repair_dispatches_extremum_year_question_without_touching_llm():
    # End-to-end through the real dispatch, proving the classifier's own "direct_lookup" output
    # is unchanged (the new path is an orthogonal refinement, not a new top-level family) and that
    # this question never reaches the LLM-backed complete() the way a genuine direct_lookup case
    # doesn't either.
    tables = [_year_table("GVR", y, v) for y, v in zip((2016, 2017, 2018, 2020), ("100", "200", "150", "300"))]
    linked = link_schema(
        "Vào cuối năm nào trong các năm 2016, 2017, 2018 và 2020, vốn chủ sở hữu của GVR đạt mức cao nhất?",
        tables,
    )
    tables_by_key = {t["table_key"]: t for t in tables}
    assert linked.query_family == "direct_lookup"

    def unused_complete(_prompt):
        raise AssertionError("must not reach the LLM path")

    result = execute_with_repair(linked, tables_by_key, unused_complete, max_retries=0)
    assert result.success and result.answer == 2020.0


def test_plan_cannot_invent_operand_coordinates():
    table = table_fixture()
    linked = link_schema("Doanh thu thuần AAA năm 2023", [table])
    payload = {"operands": [{"alias": "x", "table_key": "AAA_report|10", "row_index": 99, "column_index": 1}],
               "expression": "x", "answer_unit": None}
    with pytest.raises(ValueError, match="invented"):
        validate_plan(payload, linked, {table["table_key"]: table})


def test_expression_cannot_access_python_or_unknown_names():
    table = table_fixture()
    linked = link_schema("Doanh thu thuần AAA năm 2023", [table])
    payload = {"operands": [{"alias": "x", "table_key": "AAA_report|10", "row_index": 1, "column_index": 1}],
               "expression": "__import__('os')", "answer_unit": None}
    with pytest.raises(ValueError):
        validate_plan(payload, linked, {table["table_key"]: table})


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


def _company_table(ticker: str, value: str) -> dict:
    return {
        "table_key": f"{ticker}_report|10", "report_id": f"{ticker}_report", "source_path": "x",
        "line_position": 10, "n_cols": 2, "year": 2023, "variant": "separate", "ticker": ticker,
        "grid": [["CHỈ TIÊU", "Năm 2023"], ["Chi phí dự phòng rủi ro tín dụng", value]],
        "row_labels": [{"row_index": 0, "column_index": 0, "label": "CHỈ TIÊU"},
                       {"row_index": 1, "column_index": 0, "label": "Chi phí dự phòng rủi ro tín dụng"}],
    }


def _four_company_fixture():
    tables = [
        _company_table("AAA", "150.000.000.000"),  # above threshold
        _company_table("BBB", "50.000.000.000"),   # below
        _company_table("CCC", "200.000.000.000"),  # above threshold
        _company_table("DDD", "10.000.000.000"),   # below
    ]
    question = (
        "Trong số AAA, BBB, CCC và DDD, có bao nhiêu công ty có chi phí dự phòng rủi ro tín "
        "dụng năm 2023 lớn hơn 100 tỷ đồng?"
    )
    linked = link_schema(question, tables)
    tables_by_key = {t["table_key"]: t for t in tables}
    group_payload = {
        "name": "provision", "metric_label": "chi phí dự phòng rủi ro tín dụng",
        "operands": [
            {"alias": "g1_a", "table_key": "AAA_report|10", "row_index": 1, "column_index": 1},
            {"alias": "g1_b", "table_key": "BBB_report|10", "row_index": 1, "column_index": 1},
            {"alias": "g1_c", "table_key": "CCC_report|10", "row_index": 1, "column_index": 1},
            {"alias": "g1_d", "table_key": "DDD_report|10", "row_index": 1, "column_index": 1},
        ],
    }
    return linked, tables_by_key, group_payload


def test_count_across_companies_group_executes_correct_count():
    linked, tables_by_key, group_payload = _four_company_fixture()
    payload = {
        "operands": [], "groups": [group_payload],
        "reductions": [{"alias": "n_high", "op": "count_gt", "group": "provision", "threshold": 100_000_000_000}],
        "expression": "n_high", "answer_unit": None, "rationale": "count companies over threshold",
    }
    plan = validate_plan(payload, linked, tables_by_key)
    reduction_values = compute_reductions(plan, tables_by_key)
    assert reduction_values == {"n_high": 2.0}
    frames, variables = build_evidence_frames(plan, tables_by_key)
    query = render_pandas_query(plan, variables, reduction_values)
    assert execute_query(query, frames) == 2.0


def test_average_across_companies_group_executes_correct_average():
    linked, tables_by_key, group_payload = _four_company_fixture()
    payload = {
        "operands": [], "groups": [group_payload],
        "reductions": [{"alias": "avg_provision", "op": "avg", "group": "provision"}],
        "expression": "avg_provision / 1e9", "answer_unit": "tỷ đồng", "rationale": "average across 4 companies",
    }
    plan = validate_plan(payload, linked, tables_by_key)
    reduction_values = compute_reductions(plan, tables_by_key)
    frames, variables = build_evidence_frames(plan, tables_by_key)
    query = render_pandas_query(plan, variables, reduction_values)
    assert execute_query(query, frames) == pytest.approx(102.5)  # (150+50+200+10)/4 billion


def test_group_missing_required_company_is_rejected():
    linked, tables_by_key, group_payload = _four_company_fixture()
    incomplete_group = dict(group_payload, operands=group_payload["operands"][:3])  # drops DDD
    payload = {
        "operands": [], "groups": [incomplete_group],
        "reductions": [{"alias": "n_high", "op": "count_gt", "group": "provision", "threshold": 100_000_000_000}],
        "expression": "n_high", "answer_unit": None, "rationale": "count companies over threshold",
    }
    with pytest.raises(ValueError, match="missing required companies"):
        validate_plan(payload, linked, tables_by_key)


def test_group_metric_label_mismatched_row_is_rejected():
    linked, tables_by_key, group_payload = _four_company_fixture()
    mismatched_group = dict(group_payload, metric_label="doanh thu thuần")  # unrelated to the row labels
    payload = {
        "operands": [], "groups": [mismatched_group],
        "reductions": [{"alias": "n_high", "op": "count_gt", "group": "provision", "threshold": 100_000_000_000}],
        "expression": "n_high", "answer_unit": None, "rationale": "count companies over threshold",
    }
    with pytest.raises(ValueError, match="does not match"):
        validate_plan(payload, linked, tables_by_key)


def test_group_operand_conflicting_year_is_rejected():
    linked, tables_by_key, group_payload = _four_company_fixture()
    # Same question (explicit "năm 2023"), but re-point one operand at a wrong-year sibling
    # report that link_schema also exposed as a candidate.
    wrong_year_table = {
        **_company_table("AAA", "999.000.000.000"),
        "table_key": "AAA_report_2022|10", "report_id": "AAA_report_2022", "year": 2022,
        "grid": [["CHỈ TIÊU", "Năm 2022"], ["Chi phí dự phòng rủi ro tín dụng", "999.000.000.000"]],
    }
    tables_by_key = {**tables_by_key, wrong_year_table["table_key"]: wrong_year_table}
    linked = link_schema(linked.question, list(tables_by_key.values()))
    conflicting_group = dict(group_payload, operands=[
        {"alias": "g1_a", "table_key": "AAA_report_2022|10", "row_index": 1, "column_index": 1},
        *group_payload["operands"][1:],
    ])
    payload = {
        "operands": [], "groups": [conflicting_group],
        "reductions": [{"alias": "n_high", "op": "count_gt", "group": "provision", "threshold": 100_000_000_000}],
        "expression": "n_high", "answer_unit": None, "rationale": "count companies over threshold",
    }
    with pytest.raises(ValueError, match="conflicts with the requested period"):
        validate_plan(payload, linked, tables_by_key)


def test_unknown_reduction_op_is_rejected():
    linked, tables_by_key, group_payload = _four_company_fixture()
    payload = {
        "operands": [], "groups": [group_payload],
        "reductions": [{"alias": "n_high", "op": "argmax", "group": "provision"}],
        "expression": "n_high", "answer_unit": None, "rationale": "x",
    }
    with pytest.raises(ValueError, match="unknown reduction op"):
        validate_plan(payload, linked, tables_by_key)


def test_count_op_without_threshold_is_rejected():
    linked, tables_by_key, group_payload = _four_company_fixture()
    payload = {
        "operands": [], "groups": [group_payload],
        "reductions": [{"alias": "n_high", "op": "count_gt", "group": "provision"}],  # no threshold
        "expression": "n_high", "answer_unit": None, "rationale": "x",
    }
    with pytest.raises(ValueError, match="requires a threshold"):
        validate_plan(payload, linked, tables_by_key)


def _three_company_sum_fixture():
    # Mirrors the real graded-run shape of fixture question 938 ("Tính tổng số dư cho vay khách
    # hàng ... EIB, MSB, STB"): a sum across 3 named companies, classified `aggregation` via the
    # "tổng số" trigger.
    tables = [
        _company_table("EIB", "104.042.577"),
        _company_table("MSB", "48.762.243"),
        _company_table("STB", "256.622.753"),
    ]
    question = (
        "Tính tổng số dư cho vay khách hàng cuối kỳ năm 2023 của EIB, MSB và STB, "
        "đơn vị tính triệu đồng."
    )
    linked = link_schema(question, tables)
    tables_by_key = {t["table_key"]: t for t in tables}
    return linked, tables_by_key


def test_flat_single_operand_comparison_plan_is_rejected():
    # Regression test for EXECUTION_ACCURACY_AUDIT.md §4.2, real predictions/591.json and
    # predictions/604.json: a single-company, two-period `comparison` question validated and
    # executed a single flat operand (one period's raw balance) as if it were the requested
    # difference. A `comparison`-family plan must declare at least 2 grounded operands, isolated
    # here from the (separately tested) company-coverage check via a single-company question.
    table = table_fixture()
    linked = link_schema("Tiền và tương đương tiền AAA cuối năm 2023 so với cuối năm 2022", [table])
    assert linked.query_family == "comparison"
    payload = {
        "operands": [{"alias": "x", "table_key": table["table_key"], "row_index": 1, "column_index": 1}],
        "expression": "x", "answer_unit": None, "rationale": "x",
    }
    with pytest.raises(ValueError, match="at least 2 grounded operands"):
        validate_plan(payload, linked, {table["table_key"]: table})


def test_flat_operand_plan_missing_a_required_company_is_rejected():
    # Regression test for EXECUTION_ACCURACY_AUDIT.md §4.2, real predictions/938.json: a flat
    # (non-grouped) 2-operand plan covering only 2 of the 3 companies the question names
    # literally validated and executed, silently dropping the third company's evidence.
    linked, tables_by_key = _three_company_sum_fixture()
    payload = {
        "operands": [
            {"alias": "eib", "table_key": "EIB_report|10", "row_index": 1, "column_index": 1},
            {"alias": "msb", "table_key": "MSB_report|10", "row_index": 1, "column_index": 1},
        ],
        "expression": "eib + msb", "answer_unit": None, "rationale": "x",
    }
    with pytest.raises(ValueError, match="missing required companies"):
        validate_plan(payload, linked, tables_by_key)


def test_flat_operand_plan_covering_all_required_companies_still_executes():
    # A complete flat (non-DSL) multi-operand plan must keep working exactly as before -- the new
    # completeness checks only reject incomplete plans, never a genuinely complete one.
    linked, tables_by_key = _three_company_sum_fixture()
    payload = {
        "operands": [
            {"alias": "eib", "table_key": "EIB_report|10", "row_index": 1, "column_index": 1},
            {"alias": "msb", "table_key": "MSB_report|10", "row_index": 1, "column_index": 1},
            {"alias": "stb", "table_key": "STB_report|10", "row_index": 1, "column_index": 1},
        ],
        "expression": "eib + msb + stb", "answer_unit": None, "rationale": "x",
    }
    plan = validate_plan(payload, linked, tables_by_key)
    frames, variables = build_evidence_frames(plan, tables_by_key)
    query = render_pandas_query(plan, variables)
    assert execute_query(query, frames) == pytest.approx(104_042_577 + 48_762_243 + 256_622_753)


def test_reduction_referencing_unknown_group_is_rejected():
    linked, tables_by_key, group_payload = _four_company_fixture()
    payload = {
        "operands": [], "groups": [group_payload],
        "reductions": [{"alias": "n_high", "op": "count_gt", "group": "does_not_exist", "threshold": 1}],
        "expression": "n_high", "answer_unit": None, "rationale": "x",
    }
    with pytest.raises(ValueError, match="unknown group"):
        validate_plan(payload, linked, tables_by_key)

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))

from table_retrieval_oracle import _breakdowns, _fixture_display_path, score


def test_fixture_display_path_accepts_a_relative_cli_path():
    assert _fixture_display_path(Path("eval/dev_questions/example.json")) == (
        "eval/dev_questions/example.json"
    )


def test_oracle_breakdowns_separate_company_cardinality_and_current_family():
    rows = [
        {
            "oracle_tickers": ["AAA"],
            "query_family_current": "direct_lookup",
            "relevant_tables_metrics": score({"AAA|1"}, {"AAA|1"}),
        },
        {
            "oracle_tickers": ["AAA", "BBB"],
            "query_family_current": "comparison",
            "relevant_tables_metrics": score({"AAA|1"}, {"AAA|1", "BBB|2"}),
        },
    ]

    breakdowns = _breakdowns(
        rows, metrics_field="relevant_tables_metrics", family_field="query_family_current"
    )

    assert breakdowns["company_cardinality"]["single_company"] == {
        "precision": 1.0, "recall": 1.0, "f2": 1.0,
    }
    assert breakdowns["company_cardinality"]["multi_company"]["recall"] == 0.5
    assert breakdowns["query_family_current"]["comparison"]["f2"] < 1.0

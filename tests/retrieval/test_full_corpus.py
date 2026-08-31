import json

import pandas as pd

from retrieval.full_corpus import (
    infer_question_tickers, infer_question_variant, infer_question_years, rank_questions,
)


def test_full_rankings_are_scored_and_fingerprinted(tmp_path):
    catalog = tmp_path / "tables.csv"
    pd.DataFrame([{
        "report_id": "AAA_report", "line_position": 17, "ticker": "AAA", "year": 2023,
        "searchable_text": "doanh thu thuần 123",
    }]).to_csv(catalog, index=False)
    companies = tmp_path / "companies.csv"
    pd.DataFrame([{"ticker": "AAA", "name": "Công ty AAA"}]).to_csv(companies, index=False)
    output = tmp_path / "rankings.json"
    rankings = rank_questions(
        [{"id": 1, "question": "Doanh thu AAA năm 2023"}], catalog, companies, output, top_k=10
    )
    assert rankings["1"][0][0] == "AAA_report|17"
    assert json.loads(output.with_suffix(".json.meta.json").read_text(encoding="utf-8"))["top_k"] == 10


def test_entity_scoped_ranking_excludes_a_lexically_stronger_other_company(tmp_path):
    catalog = tmp_path / "tables.csv"
    pd.DataFrame([
        {"report_id": "AAA_report", "line_position": 17, "ticker": "AAA", "year": 2023,
         "searchable_text": "doanh thu thuần"},
        {"report_id": "BBB_report", "line_position": 21, "ticker": "BBB", "year": 2023,
         "searchable_text": "doanh thu thuần doanh thu thuần doanh thu thuần"},
    ]).to_csv(catalog, index=False)
    companies = tmp_path / "companies.csv"
    pd.DataFrame([
        {"ticker": "AAA", "name": "Cong ty AAA"},
        {"ticker": "BBB", "name": "Cong ty BBB"},
    ]).to_csv(companies, index=False)
    rankings = rank_questions(
        [{"id": 1, "question": "Doanh thu AAA nam 2023"}], catalog, companies,
        tmp_path / "rankings.json", top_k=5,
    )
    assert [row[0] for row in rankings["1"]] == ["AAA_report|17"]
    assert infer_question_tickers("Doanh thu AAA nam 2023", {"AAA": "Cong ty AAA"}) == {"AAA"}


def test_explicit_ticker_beats_weak_company_name_overlap():
    companies = {
        "VIC": "Vingroup Holdings",
        "PRT": "Binh San Manufacturing",
        "FTS": "FPT Securities Broker",
        "FPT": "FPT Holdings",
    }
    assert infer_question_tickers("VIC Binh San assets", companies) == {"VIC"}
    # The full legal-name alias identifies FTS even though FPT is uppercase in the question.
    assert infer_question_tickers("FPT Securities Broker 2023", companies) == {"FTS"}


def test_entity_scope_keeps_bridging_reports_for_multi_year_question(tmp_path):
    catalog = tmp_path / "tables.csv"
    pd.DataFrame([
        {"report_id": "AAA_2022", "line_position": 1, "ticker": "AAA", "year": 2022,
         "variant": "consolidated", "searchable_text": "equity"},
        {"report_id": "AAA_2023", "line_position": 2, "ticker": "AAA", "year": 2023,
         "variant": "consolidated", "searchable_text": "equity 2022 2023"},
        {"report_id": "AAA_2024", "line_position": 3, "ticker": "AAA", "year": 2024,
         "variant": "consolidated", "searchable_text": "equity 2023 2024"},
    ]).to_csv(catalog, index=False)
    companies = tmp_path / "companies.csv"
    pd.DataFrame([{"ticker": "AAA", "name": "Company AAA"}]).to_csv(companies, index=False)
    rankings = rank_questions(
        [{"id": 1, "question": "AAA equity from 2022 to 2024"}],
        catalog, companies, tmp_path / "rankings.json", top_k=5,
    )
    assert {row[0] for row in rankings["1"]} == {"AAA_2022|1", "AAA_2023|2", "AAA_2024|3"}


def test_explicit_ticker_survives_generic_vocabulary_tie_across_many_companies():
    # Real bug (2026-08-31 profiling, Q53): "bất động sản đầu tư" (real estate investment) is
    # generic sector vocabulary that overlaps >=3 tokens with several unrelated real-estate
    # companies' full names, tying them all at the same score. That tie must never outrank the
    # literal, unambiguous ticker IJC mentioned in the same question.
    companies = {
        "IJC": "CTCP Phat trien Khu cong nghiep Becamex IJC",
        "PDR": "CTCP Phat trien Bat dong san Phat Dat",
        "DXS": "CTCP Dich vu Bat dong san Dat Xanh",
        "CRE": "CTCP Bat dong san The Ky",
    }
    # Force a real overlap tie in this smaller vocabulary: all three unrelated names share
    # enough generic tokens with the question to reach the strong (>=3) threshold together.
    question = "Gia tri con lai cua bat dong san dau tu cua cong ty me IJC den ngay 31 thang 12"
    assert infer_question_tickers(question, companies) == {"IJC"}


def test_explicit_ticker_survives_unrelated_single_strong_name_match():
    # Real bug shape (2026-08-31 profiling, Q73): a counterparty/investee company happens to be
    # named in the question and its full name coincidentally clears the strong-overlap threshold
    # on its own (no tie), but it is not the company the question is about -- the literal ticker
    # PC1 must be kept, and the unrelated single winner must not silently replace or join it.
    companies = {
        "PC1": "CTCP Tap Doan PC1",
        "NVL": "CTCP Tap doan Dau tu Dia oc No Va",
    }
    question = "Tong gia tri ghi so dau tu vao cong ty lien ket cua CTCP Tap Doan PC1 la bao nhieu"
    assert infer_question_tickers(question, companies) == {"PC1"}


def test_nested_brand_name_still_overrides_literal_ticker():
    # The one case where override is still correct: the literal ticker's own token is nested
    # inside the more specific winning company name (same real-world mention, not a second
    # company) -- must keep working after the fix.
    companies = {
        "FPT": "FPT Holdings",
        "FTS": "FPT Securities Broker",
    }
    assert infer_question_tickers("FPT Securities Broker 2023", companies) == {"FTS"}


def test_infer_question_tickers_resolves_alphanumeric_ticker():
    # HT1 and PC1 are real ViFinQA tickers (data/raw/hf_meta/code_stock.csv) that contain a
    # trailing digit; a letters-only ticker regex cannot find them because \b never falls
    # between a letter and a digit inside one \w run (confirmed via profiling all 1,012 real
    # questions -- IDs 7/276 mention HT1, 58/283 mention PC1, all previously unresolved).
    companies = {"HT1": "CTCP Xi Măng Vicem Hà Tiên", "PC1": "CTCP Tập Đoàn PC1"}
    assert infer_question_tickers("Quỹ khen thưởng của HT1 cuối năm 2019", companies) == {"HT1"}
    assert infer_question_tickers("Tổng tiền trả trước của PC1 cuối năm 2023", companies) == {"PC1"}
    # a bare year must still never be mistaken for a ticker
    assert infer_question_tickers("Doanh thu năm 2023", companies) == set()


def test_row_label_rerank_recovers_correct_table_from_a_large_same_entity_pool(tmp_path):
    # Real shape confirmed by the dev_v1 README and full-corpus profiling: a resolved ticker's
    # candidate pool can hold hundreds-to-thousands of same-company tables whose whole-table
    # BM25 text repeats generic identity/boilerplate terms, so a distractor table can out-score
    # the table whose row labels actually contain the requested line item. Simulate the minimal
    # version: a distractor table's bulk cell text happens to repeat the query's generic terms
    # more often than the correct table's searchable_text does, but only the correct table's
    # row labels contain the specific requested line item ("phải thu ngắn hạn khác").
    catalog = tmp_path / "tables.csv"
    pd.DataFrame([
        {"report_id": "AAA_report", "line_position": 1, "ticker": "AAA", "year": 2023,
         "searchable_text": "phải thu ngắn hạn khác " * 10},
        {"report_id": "AAA_report", "line_position": 2, "ticker": "AAA", "year": 2023,
         "searchable_text": "phải thu ngắn hạn khác"},
    ]).to_csv(catalog, index=False)
    companies = tmp_path / "companies.csv"
    pd.DataFrame([{"ticker": "AAA", "name": "Cong ty AAA"}]).to_csv(companies, index=False)
    row_labels = tmp_path / "row_labels.csv"
    pd.DataFrame([
        {"table_key": "AAA_report|1", "row_label_text": "Tổng tài sản"},
        {"table_key": "AAA_report|2", "row_label_text": "Phải thu ngắn hạn khác"},
    ]).to_csv(row_labels, index=False)
    question = {"id": 1, "question": "Phải thu ngắn hạn khác của AAA năm 2023 là bao nhiêu?"}
    # Confirms the intended baseline failure actually occurs with real BM25 scoring (not assumed):
    # the distractor's repeated phrase legitimately out-scores the sparser correct table.

    without_rerank = rank_questions(
        [question], catalog, companies, tmp_path / "no_rerank.json", top_k=2,
    )
    assert without_rerank["1"][0][0] == "AAA_report|1"  # distractor wins on raw term frequency

    with_rerank = rank_questions(
        [question], catalog, companies, tmp_path / "with_rerank.json", top_k=2,
        row_label_index_path=row_labels, row_label_rerank_weight=8.0,
    )
    assert with_rerank["1"][0][0] == "AAA_report|2"  # row-label match promotes the correct table
    # Reranking only reorders what BM25 already found -- both candidates remain present.
    assert {row[0] for row in with_rerank["1"]} == {"AAA_report|1", "AAA_report|2"}


def test_multi_company_search_gives_every_named_company_its_own_top_k(tmp_path):
    # Real shape from the full-1,012-question profile (IDs 362-436): a group-screening question
    # names several companies (here 3) and each needs its own candidate tables surfaced, not a
    # single top_k budget shared across the whole group where one company's tables could crowd
    # out another's. Give each company 3 tables and ask for top_k=2 -- a flat pooled top_k=2
    # search could return tables from only one or two of the three companies; per-company
    # search must return up to top_k from EACH.
    catalog = tmp_path / "tables.csv"
    rows = []
    for ticker in ["AAA", "BBB", "CCC"]:
        for i in range(3):
            rows.append({
                "report_id": f"{ticker}_report", "line_position": i, "ticker": ticker, "year": 2023,
                "searchable_text": f"doanh thu thuan {ticker} muc {i}",
            })
    pd.DataFrame(rows).to_csv(catalog, index=False)
    companies = tmp_path / "companies.csv"
    pd.DataFrame([
        {"ticker": "AAA", "name": "Cong ty AAA"},
        {"ticker": "BBB", "name": "Cong ty BBB"},
        {"ticker": "CCC", "name": "Cong ty CCC"},
    ]).to_csv(companies, index=False)
    question = {"id": 1, "question": "Doanh thu thuan cua nhom AAA, BBB va CCC nam 2023"}

    rankings = rank_questions([question], catalog, companies, tmp_path / "rankings.json", top_k=2)
    retrieved_tickers = {key.split("_report")[0] for key, _ in rankings["1"]}
    assert retrieved_tickers == {"AAA", "BBB", "CCC"}
    # Each company contributes at most top_k of its own candidates, not a shared budget.
    per_company_counts = {}
    for key, _ in rankings["1"]:
        per_company_counts.setdefault(key.split("_report")[0], 0)
        per_company_counts[key.split("_report")[0]] += 1
    assert all(count <= 2 for count in per_company_counts.values())
    assert sum(per_company_counts.values()) == 6  # 2 per company x 3 companies, not one shared top_k=2


def test_concept_decomposition_recovers_tables_a_flat_query_buries(tmp_path):
    # Real shape from the huge-pool audit (CHANGE_LOG.md 2026-08-31): a compound question naming
    # a ratio (here ROE) needs two distinct concept tables (equity, profit) that a single flat
    # query over the whole question text can bury under many distractor tables sharing generic
    # overlapping vocabulary, none of which is the actual answer.
    catalog = tmp_path / "tables.csv"
    rows = [
        {"report_id": "AAA_report", "line_position": 1, "ticker": "AAA", "year": 2023,
         "searchable_text": "vốn chủ sở hữu"},
        {"report_id": "AAA_report", "line_position": 2, "ticker": "AAA", "year": 2023,
         "searchable_text": "lợi nhuận sau thuế"},
    ]
    for i in range(3, 40):
        rows.append({
            "report_id": "AAA_report", "line_position": i, "ticker": "AAA", "year": 2023,
            "searchable_text": "roe roe roe cổ phiếu thưởng phần trăm bao nhiêu " * 3,
        })
    pd.DataFrame(rows).to_csv(catalog, index=False)
    companies = tmp_path / "companies.csv"
    pd.DataFrame([{"ticker": "AAA", "name": "Cong ty AAA"}]).to_csv(companies, index=False)
    question = {"id": 1, "question": "ROE của AAA năm 2023 là bao nhiêu phần trăm?"}
    # Confirmed by direct BM25 inspection (no decomposition): the 37 distractors bury both
    # target tables at rank 38/39 out of 39 -- nowhere near top_k=10. rank_questions always runs
    # decomposition once 2+ concepts are detected (ROE -> equity + profit here), so this call
    # exercises the recovery path, not a flat search.
    result = rank_questions([question], catalog, companies, tmp_path / "decomposed.json", top_k=10)
    result_keys = {row[0] for row in result["1"]}
    assert "AAA_report|1" in result_keys
    assert "AAA_report|2" in result_keys


def test_concept_decomposition_respects_the_candidate_ceiling(tmp_path):
    catalog = tmp_path / "tables.csv"
    concepts = [
        "vốn chủ sở hữu", "lợi nhuận sau thuế", "doanh thu thuần", "tổng tài sản",
        "hàng tồn kho", "nợ ngắn hạn", "tài sản cố định",
    ]
    rows = []
    position = 1
    for concept in concepts:
        for i in range(6):  # several near-duplicate tables per concept
            rows.append({
                "report_id": "AAA_report", "line_position": position, "ticker": "AAA", "year": 2023,
                "searchable_text": f"{concept} chi tiết mục {i}",
            })
            position += 1
    pd.DataFrame(rows).to_csv(catalog, index=False)
    companies = tmp_path / "companies.csv"
    pd.DataFrame([{"ticker": "AAA", "name": "Cong ty AAA"}]).to_csv(companies, index=False)
    question = {"id": 1, "question": (
        "Năm 2023, ROE, vòng quay tổng tài sản, hàng tồn kho và nợ ngắn hạn của AAA là bao nhiêu?"
    )}
    rankings = rank_questions([question], catalog, companies, tmp_path / "rankings.json", top_k=10)
    assert len(rankings["1"]) <= 30  # MAX_CANDIDATES_PER_TICKER


def test_entity_scope_hard_filters_explicit_year_and_variant(tmp_path):
    catalog = tmp_path / "tables.csv"
    pd.DataFrame([
        {"report_id": "AAA_2022_separate", "line_position": 1, "ticker": "AAA", "year": 2022, "variant": "separate", "searchable_text": "chi phí khác"},
        {"report_id": "AAA_2023_consolidated", "line_position": 2, "ticker": "AAA", "year": 2023, "variant": "consolidated", "searchable_text": "chi phí khác chi phí khác"},
        {"report_id": "AAA_2023_separate", "line_position": 3, "ticker": "AAA", "year": 2023, "variant": "separate", "searchable_text": "chi phí khác"},
    ]).to_csv(catalog, index=False)
    companies = tmp_path / "companies.csv"
    pd.DataFrame([{"ticker": "AAA", "name": "Cong ty AAA"}]).to_csv(companies, index=False)
    rankings = rank_questions(
        [{"id": 1, "question": "Chi phí khác của công ty mẹ AAA cuối năm 2023"}],
        catalog, companies, tmp_path / "rankings.json", top_k=5,
    )
    assert [row[0] for row in rankings["1"]] == ["AAA_2023_separate|3"]
    assert infer_question_years("AAA năm 2020 đến 2021") == {2020, 2021}
    assert infer_question_variant("BCTC riêng AAA") == "separate"
    assert infer_question_variant("AAA năm 2023") is None

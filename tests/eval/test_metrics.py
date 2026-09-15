from eval.metrics import (
    evaluate_mrr,
    evaluate_retrieval,
    evaluate_retrieval_full,
    mrr_at_k,
    precision_recall_f2_for_query,
    project_zero_credit_macro,
)


def test_perfect_retrieval_scores_one_everywhere():
    p, r, f2 = precision_recall_f2_for_query(["a|1", "a|2"], ["a|1", "a|2"])
    assert (p, r, f2) == (1.0, 1.0, 1.0)


def test_recall_weighted_4to1_over_precision():
    # high recall, low precision: retrieve everything relevant plus a lot of noise
    p, r, f2 = precision_recall_f2_for_query(
        ["a|1", "a|2", "x|1", "x|2", "x|3", "x|4"], ["a|1", "a|2"]
    )
    assert r == 1.0
    assert p == 2 / 6
    # low recall, high precision: retrieve only one of two relevant, nothing extra
    p2, r2, f2_2 = precision_recall_f2_for_query(["a|1"], ["a|1", "a|2"])
    assert p2 == 1.0
    assert r2 == 0.5
    # F2 must favor the high-recall/low-precision case over the high-precision/low-recall one
    assert f2 > f2_2


def test_empty_retrieved_gives_zero_precision_and_recall():
    p, r, f2 = precision_recall_f2_for_query([], ["a|1"])
    assert (p, r, f2) == (0.0, 0.0, 0.0)


def test_key_fn_allows_document_level_matching():
    # exact line_position differs but report_id matches -- should count as correct under a
    # document-level key_fn, not under the default exact-string key_fn.
    retrieved = ["AAA_report|999"]
    relevant = ["AAA_report|279"]
    p_exact, r_exact, _ = precision_recall_f2_for_query(retrieved, relevant)
    assert (p_exact, r_exact) == (0.0, 0.0)

    doc_key = lambda s: s.split("|")[0]
    p_doc, r_doc, _ = precision_recall_f2_for_query(retrieved, relevant, key_fn=doc_key)
    assert (p_doc, r_doc) == (1.0, 1.0)


def test_evaluate_retrieval_macro_averages_across_queries():
    queries = [
        {"id": 1, "relevant_tables": ["a|1"]},
        {"id": 2, "relevant_tables": ["b|1", "b|2"]},
    ]
    retrieved_by_id = {
        1: ["a|1"],  # perfect
        2: ["b|1"],  # half recall, full precision
    }
    scores, summary = evaluate_retrieval(queries, retrieved_by_id)
    assert summary["n_queries"] == 2
    assert summary["precision_macro"] == 1.0
    assert summary["recall_macro"] == (1.0 + 0.5) / 2
    assert len(scores) == 2
    assert scores[1].n_correct == 1
    assert scores[1].n_relevant == 2


def test_mrr_at_k_returns_reciprocal_rank_of_first_correct_hit():
    assert mrr_at_k(["x|1", "a|1", "x|2"], ["a|1"]) == 1 / 2
    assert mrr_at_k(["a|1", "x|1"], ["a|1"]) == 1.0


def test_mrr_at_k_ignores_hits_beyond_k():
    # correct item is at rank 6, k=5 -- must not count it
    retrieved = ["x|1", "x|2", "x|3", "x|4", "x|5", "a|1"]
    assert mrr_at_k(retrieved, ["a|1"], k=5) == 0.0


def test_mrr_at_k_returns_zero_when_nothing_relevant_is_retrieved():
    assert mrr_at_k(["x|1", "x|2"], ["a|1"]) == 0.0


def test_mrr_at_k_respects_key_fn_for_document_level_matching():
    retrieved = ["x|1", "AAA_report|999"]
    relevant = ["AAA_report|279"]
    assert mrr_at_k(retrieved, relevant) == 0.0
    assert mrr_at_k(retrieved, relevant, key_fn=lambda s: s.split("|")[0]) == 1 / 2


def test_evaluate_mrr_macro_averages_across_queries():
    queries = [{"id": 1, "relevant_tables": ["a|1"]}, {"id": 2, "relevant_tables": ["b|1"]}]
    retrieved_by_id = {1: ["a|1"], 2: ["x|1", "b|1"]}
    scores, macro = evaluate_mrr(queries, retrieved_by_id)
    assert scores == [1.0, 1 / 2]
    assert macro == (1.0 + 1 / 2) / 2


def test_evaluate_retrieval_full_matches_official_score_sheet_keys():
    # shape mirrors scoring_result/scores.txt: TABLES_*/DOCS_* precision/recall/F2MACRO/MRR5.
    queries = [{"id": 1, "relevant_tables": ["a|1", "a|2"]}]
    retrieved_by_id = {1: ["a|1", "b|1"]}
    result = evaluate_retrieval_full(queries, retrieved_by_id)
    assert set(result) >= {
        "TABLES_PRECISION", "TABLES_RECALL", "TABLES_F2MACRO", "TABLES_MRR5",
        "DOCS_PRECISION", "DOCS_RECALL", "DOCS_F2MACRO", "DOCS_MRR5",
    }
    # table-level: 1 of 2 retrieved correct, 1 of 2 relevant found
    assert result["TABLES_PRECISION"] == 0.5
    assert result["TABLES_RECALL"] == 0.5
    # doc-level: both retrieved docs are "a" and "b"; relevant doc is "a" -- same P/R as table
    # level here since there's only one query, but computed via the doc key_fn independently
    assert result["DOCS_PRECISION"] == 0.5
    assert result["DOCS_RECALL"] == 1.0  # doc "a" is present, and that's the only relevant doc
    assert "mrr_rank_semantics_assumption" in result


def test_project_zero_credit_macro_scales_down_for_missing_questions():
    summary = {"n_queries": 2, "precision_macro": 1.0, "recall_macro": 1.0, "f2_macro": 1.0}
    projected = project_zero_credit_macro(summary, total_questions=4)
    assert projected["n_scored"] == 2
    assert projected["n_total"] == 4
    assert projected["precision_macro_zero_credit"] == 0.5
    assert projected["recall_macro_zero_credit"] == 0.5
    assert projected["f2_macro_zero_credit"] == 0.5


def test_project_zero_credit_macro_is_a_no_op_when_fully_scored():
    summary = {"n_queries": 3, "precision_macro": 0.7, "recall_macro": 0.4, "f2_macro": 0.45}
    projected = project_zero_credit_macro(summary, total_questions=3)
    assert projected["precision_macro_zero_credit"] == 0.7
    assert projected["f2_macro_zero_credit"] == 0.45


def test_project_zero_credit_macro_rejects_total_below_scored_count():
    summary = {"n_queries": 5, "precision_macro": 1.0, "recall_macro": 1.0, "f2_macro": 1.0}
    try:
        project_zero_credit_macro(summary, total_questions=3)
        assert False, "expected ValueError"
    except ValueError:
        pass

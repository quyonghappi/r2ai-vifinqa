from eval.metrics import evaluate_retrieval, precision_recall_f2_for_query


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

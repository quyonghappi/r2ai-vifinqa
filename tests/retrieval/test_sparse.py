from retrieval.sparse import DEFAULT_STOPWORDS, build_enriched_document_text, build_index, tokenize


def test_tokenize_preserves_vietnamese_diacritics():
    assert tokenize("Tiền và các khoản tương đương tiền") == [
        "tiền", "và", "các", "khoản", "tương", "đương", "tiền",
    ]


def test_search_ranks_more_relevant_doc_higher():
    docs = {
        "balance_sheet": "TÀI SẢN Mã số Thuyết minh Tiền và các khoản tương đương tiền 470.061.718.120",
        "unrelated_note": "Chi phí khấu hao tài sản cố định hữu hình trong kỳ",
    }
    index = build_index(list(docs.keys()), list(docs.values()))
    results = index.search("tiền và các khoản tương đương tiền")
    assert results[0][0] == "balance_sheet"
    assert results[0][1] > 0


def test_search_excludes_documents_with_zero_term_overlap():
    docs = {"a": "tiền mặt ngân hàng", "b": "hoàn toàn khác biệt luôn"}
    index = build_index(list(docs.keys()), list(docs.values()))
    results = index.search("xyz abc123 zzz")
    assert results == []


def test_search_top_k_truncates_without_dropping_silently_elsewhere():
    docs = {f"doc{i}": "tiền mặt" for i in range(10)}
    index = build_index(list(docs.keys()), list(docs.values()))
    all_results = index.search("tiền mặt", top_k=None)
    top_3 = index.search("tiền mặt", top_k=3)
    assert len(all_results) == 10
    assert len(top_3) == 3
    assert top_3 == all_results[:3]


def test_build_index_rejects_mismatched_lengths():
    import pytest
    with pytest.raises(ValueError):
        build_index(["a", "b"], ["only one text"])


def test_tokenize_stopword_filtering_is_opt_in():
    text = "Công ty CTCP là một doanh nghiệp"
    assert "là" in tokenize(text)
    assert "là" not in tokenize(text, stopwords=DEFAULT_STOPWORDS)


def test_index_and_search_use_matching_stopwords_automatically():
    # a query built from generic Vietnamese phrasing plus one distinctive company mention --
    # without stopword filtering, generic terms shared with a distractor doc can outrank the
    # true match (this is the real failure mode found on the full 146K-table corpus, see
    # CHANGE_LOG.md). Filtering must be applied identically to docs and queries automatically.
    docs = {
        "target": "vjc vietjet tiền mặt và các khoản tương đương tiền là 100",
        "distractor": "công ty và các khoản mục là một phần của báo cáo tài chính năm nay",
    }
    index = build_index(list(docs.keys()), list(docs.values()), stopwords=DEFAULT_STOPWORDS)
    results = index.search("vjc vietjet tiền mặt là bao nhiêu")
    assert results[0][0] == "target"


def test_build_enriched_document_text_repeats_identity_fields():
    text = build_enriched_document_text("VJC", "CTCP Hàng không Vietjet", 2018, "tiền mặt 100", identity_boost=3)
    assert text.count("VJC") == 3
    assert text.endswith("tiền mặt 100")


def test_equal_score_ties_have_deterministic_doc_id_order():
    index = build_index(["z", "a", "m"], ["tiền", "tiền", "tiền"])
    assert [doc_id for doc_id, _ in index.search("tiền")] == ["a", "m", "z"]

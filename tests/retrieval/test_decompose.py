from retrieval.decompose import MAX_CANDIDATES_PER_TICKER, concept_top_m, extract_concept_queries


def test_extract_concept_queries_expands_named_ratio_into_operands():
    # Q405's real shape: ROE + asset turnover + a revenue-growth filter clause.
    question = (
        "Trong giai đoạn 2022-2024 của VIC, xét các năm doanh thu thuần tăng so với năm liền "
        "trước, ROE của năm có vòng quay tổng tài sản theo tài sản bình quân cao nhất là bao "
        "nhiêu phần trăm?"
    )
    concepts = extract_concept_queries(question)
    assert set(concepts) == {"vốn chủ sở hữu", "lợi nhuận sau thuế", "doanh thu thuần", "tổng tài sản"}


def test_extract_concept_queries_plain_lookup_is_a_no_op():
    # A single directly-reported line item must not trigger decomposition.
    question = "Lãi tiền gửi năm 2018 của công ty mẹ CTCP Hàng không Vietjet (VJC) là bao nhiêu triệu đồng?"
    assert len(extract_concept_queries(question)) < 2


def test_extract_concept_queries_prefers_longest_match():
    # Real false positive found during the design audit: "tổng tài sản" (total balance-sheet
    # assets) is a literal substring of "tổng tài sản cố định" (total FIXED assets, a different,
    # more specific line item). Longest-match-first must not double-count these as 2 concepts.
    question = (
        "Tỷ trọng tài sản cố định vô hình trong tổng tài sản cố định của công ty mẹ Ngân hàng "
        "TMCP Ngoại thương Việt Nam cuối năm 2025 là bao nhiêu %?"
    )
    concepts = extract_concept_queries(question)
    assert concepts == ["tài sản cố định"]
    assert "tổng tài sản" not in concepts


def test_extract_concept_queries_dedupes_repeated_mentions():
    question = "Doanh thu thuần năm 2023 và doanh thu thuần năm 2024 của AAA là bao nhiêu?"
    assert extract_concept_queries(question) == ["doanh thu thuần"]


def test_concept_top_m_scales_down_with_more_companies():
    assert concept_top_m(1) == 10
    assert concept_top_m(2) == 7
    assert concept_top_m(8) == 2  # floor
    assert concept_top_m(20) == 2  # floor


def test_max_candidates_per_ticker_is_a_real_finite_ceiling():
    assert 0 < MAX_CANDIDATES_PER_TICKER < 1000

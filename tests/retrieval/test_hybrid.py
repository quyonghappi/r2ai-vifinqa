import json

from retrieval.hybrid import backfill_fusion, fuse_ranking_artifacts, reciprocal_rank_fusion


def test_rrf_is_deterministic_and_rewards_agreement():
    fused = reciprocal_rank_fusion(
        [[("b", 9.0), ("a", 8.0)], [("a", 3.0), ("b", 2.0)]], top_k=2,
    )
    assert fused[0][0] == "a"


def test_backfill_never_lets_a_mediocre_dense_candidate_displace_a_strong_sparse_one():
    # Real Kaggle finding (CHANGE_LOG.md 2026-08-31 hybrid-fusion-backfill entry): after
    # row-label-rerank/decomposition/per-company-assembly improved sparse but not dense, sparse
    # correctly ranked the true table 1st while dense (unimproved) ranked it far down; a
    # mediocre-in-both table could out-score it under symmetric RRF. Backfill must never let
    # that happen -- sparse's own top ranking is untouchable.
    sparse = [("true_table", 5.0), ("second", 4.0)]
    dense = [("mediocre_a", 1.0), ("mediocre_b", 0.9), ("true_table", 0.1)]  # true_table ranked 3rd, weak
    fused = backfill_fusion(sparse, dense, max_backfill=10)
    assert fused[0][0] == "true_table"  # sparse's #1 stays #1, no matter where dense ranked it
    assert [key for key, _ in fused] == ["true_table", "second", "mediocre_a", "mediocre_b"]


def test_backfill_adds_dense_unique_finds_but_never_removes_sparse_candidates():
    sparse = [("a", 1.0), ("b", 0.9)]
    dense = [("c", 1.0), ("a", 0.5)]  # "c" is a genuine dense-only find
    fused = backfill_fusion(sparse, dense, max_backfill=10)
    assert [key for key, _ in fused] == ["a", "b", "c"]


def test_backfill_respects_max_backfill_budget():
    sparse = [("a", 1.0)]
    dense = [("b", 1.0), ("c", 0.9), ("d", 0.8)]
    fused = backfill_fusion(sparse, dense, max_backfill=2)
    assert [key for key, _ in fused] == ["a", "b", "c"]  # only 2 of the 3 unique dense finds added


def test_fuse_artifacts_preserves_downstream_ranking_contract(tmp_path):
    sparse = tmp_path / "sparse.json"
    dense = tmp_path / "dense.json"
    output = tmp_path / "hybrid.json"
    sparse.write_text(json.dumps({"1": [["a", 1.0], ["b", 0.5]]}), encoding="utf-8")
    dense.write_text(json.dumps({"1": [["b", 1.0], ["a", 0.5]]}), encoding="utf-8")
    fused = fuse_ranking_artifacts(sparse, dense, output, top_k=2)
    assert fused["1"][0][0] == "a"
    assert len(fused["1"]) == 2
    assert json.loads(output.read_text(encoding="utf-8")) == fused

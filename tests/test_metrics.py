from research_assistant.metrics import (
    hit_at_k,
    is_relevant,
    parse_verdict,
    reciprocal_rank,
    reciprocal_rank_fusion,
)


def test_is_relevant_needs_right_paper_and_evidence():
    assert is_relevant("Uses BART as generator", "p1", "p1", ["bart"])
    assert not is_relevant("Uses BART as generator", "p2", "p1", ["bart"])
    assert not is_relevant("Uses T5 as generator", "p1", "p1", ["bart"])


def test_hit_at_k():
    assert hit_at_k([False, True, False], 1) == 0.0
    assert hit_at_k([False, True, False], 2) == 1.0
    assert hit_at_k([], 5) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank([True, False], 10) == 1.0
    assert reciprocal_rank([False, False, True], 10) == 1 / 3
    assert reciprocal_rank([False, False, True], 2) == 0.0


def test_rrf_rewards_documents_ranked_high_in_both_lists():
    fused = reciprocal_rank_fusion([[1, 2, 3], [2, 1, 4]])
    assert set(fused[:2]) == {1, 2}
    assert fused[-1] in (3, 4)


def test_parse_verdict():
    assert parse_verdict('{"score": 2, "reason": "matches"}') == (2, "matches")
    assert parse_verdict('Sure! {"score": "1", "reason": "partial"} done')[0] == 1
    assert parse_verdict("no json here")[0] is None
    assert parse_verdict('{"score": 7}')[0] is None
    assert parse_verdict(None)[0] is None

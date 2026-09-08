import pytest
import torch

from utils.growth_budget import (
    build_fsgs_available_value_score,
    compute_proximity_budget,
    compute_proximity_candidate_capacity,
    format_demand_preserve_diag,
    format_proximity_budget_log,
    format_proximity_capacity_log,
    format_proximity_replay_log,
    format_value_promotion_diag,
    format_value_rerank_diag,
    parse_proximity_action_replay,
    requires_value_features,
    resolve_proximity_selection_mode,
    robust_normalize,
    select_proximity_sources,
)
from arguments import validate_optimization_params


def test_budget_new_is_floor_rho_times_current():
    stats = compute_proximity_budget(current=101, candidates=50, n=3, rho=0.10, enabled=True)

    assert stats.budget_new == 10


@pytest.mark.parametrize(
    ("candidates", "keep_ratio", "capacity"),
    [
        (0, 0.8, 0),
        (1, 0.8, 1),
        (55, 0.8, 44),
        (148, 0.8, 118),
        (303, 0.8, 242),
        (17, 1.0, 17),
    ],
)
def test_candidate_capacity_formula_boundaries(candidates, keep_ratio, capacity):
    stats = compute_proximity_candidate_capacity(
        current=1000,
        candidates=candidates,
        n=3,
        keep_ratio=keep_ratio,
        enabled=True,
    )

    assert stats.capacity_src == capacity
    assert stats.selected_src == capacity
    assert stats.selected_new == capacity * 3


def test_parse_proximity_action_replay_empty_is_disabled():
    assert parse_proximity_action_replay("") == {}


def test_parse_proximity_action_replay_schedule():
    assert parse_proximity_action_replay("600:44,700:98") == {600: 44, 700: 98}


@pytest.mark.parametrize(
    "spec",
    [
        "600:44,600:45",
        "600",
        "600:-1",
        "0:1",
        "abc:1",
        "600:abc",
    ],
)
def test_parse_proximity_action_replay_invalid_specs_raise(spec):
    with pytest.raises(ValueError):
        parse_proximity_action_replay(spec)


@pytest.mark.parametrize("keep_ratio", [0, -0.1, 1.1])
def test_candidate_capacity_invalid_ratio_raises(keep_ratio):
    with pytest.raises(ValueError, match="proximity_candidate_keep_ratio"):
        compute_proximity_candidate_capacity(
            current=1000,
            candidates=10,
            n=3,
            keep_ratio=keep_ratio,
            enabled=True,
        )


def test_old_budget_and_candidate_capacity_are_mutually_exclusive():
    candidate_mask = torch.ones(5, dtype=torch.bool)
    dist = torch.arange(5, dtype=torch.float32)

    with pytest.raises(ValueError, match="cannot both be enabled"):
        select_proximity_sources(
            candidate_mask,
            dist,
            n=3,
            rho=0.5,
            enabled=True,
            candidate_capacity_enabled=True,
            candidate_keep_ratio=0.8,
        )


def test_budget_and_replay_are_mutually_exclusive():
    candidate_mask = torch.ones(5, dtype=torch.bool)
    dist = torch.arange(5, dtype=torch.float32)

    with pytest.raises(ValueError, match="mutually exclusive"):
        select_proximity_sources(
            candidate_mask,
            dist,
            enabled=True,
            replay_schedule={600: 4},
            iteration=600,
        )


def test_candidate_capacity_and_replay_are_mutually_exclusive():
    candidate_mask = torch.ones(5, dtype=torch.bool)
    dist = torch.arange(5, dtype=torch.float32)

    with pytest.raises(ValueError, match="mutually exclusive"):
        select_proximity_sources(
            candidate_mask,
            dist,
            candidate_capacity_enabled=True,
            replay_schedule={600: 4},
            iteration=600,
        )


@pytest.mark.parametrize(("candidates", "scheduled"), [(55, 44), (148, 118)])
def test_replay_selected_source_count_controls_selected_new(candidates, scheduled):
    candidate_mask = torch.ones(candidates, dtype=torch.bool)
    dist = torch.arange(candidates, dtype=torch.float32)

    _, stats = select_proximity_sources(
        candidate_mask,
        dist,
        n=3,
        mode="proximity_topk",
        replay_schedule={600: scheduled},
        iteration=600,
    )

    assert stats.replay_active is True
    assert stats.selected_src == scheduled
    assert stats.selected_new == scheduled * 3
    assert stats.replay_exact_match is True


def test_replay_scheduled_source_over_candidates_raises_without_clamp():
    with pytest.raises(ValueError, match="Replay selected_src=11 exceeds current candidates=10 at iteration=600"):
        select_proximity_sources(
            torch.ones(10, dtype=torch.bool),
            torch.arange(10, dtype=torch.float32),
            replay_schedule={600: 11},
            iteration=600,
        )


def test_replay_original_mode_resolves_to_proximity_topk():
    selected_mask, stats = select_proximity_sources(
        torch.ones(5, dtype=torch.bool),
        torch.tensor([1.0, 5.0, 3.0, 4.0, 2.0]),
        mode="original",
        replay_schedule={600: 2},
        iteration=600,
    )

    assert stats.mode == "proximity_topk"
    assert stats.selected_indices == (1, 3)
    assert torch.equal(selected_mask, torch.tensor([False, True, False, True, False]))


def test_replay_value_demand_changes_identity_not_count():
    candidate_mask = torch.ones(5, dtype=torch.bool)
    dist = torch.tensor([100.0, 90.0, 80.0, 79.0, 1.0])
    value = torch.tensor([0.0, 0.0, 0.10, 9.0, 0.0])

    selected_mask, stats = select_proximity_sources(
        candidate_mask,
        dist,
        n=3,
        mode="value_demand_rerank",
        value_score=value,
        replay_schedule={700: 3},
        iteration=700,
        rerank_fraction=0.5,
        boundary_multiplier=2.0,
        demand_ratio=0.90,
    )

    assert stats.a1_indices == (0, 1, 2)
    assert stats.selected_indices == (0, 1, 3)
    assert not torch.equal(selected_mask, torch.tensor([True, True, True, False, False]))
    assert stats.selected_src == 3
    assert stats.selected_new == 9
    assert stats.replay_exact_match is True


@pytest.mark.parametrize(
    "value",
    [
        None,
        torch.tensor([float("nan"), float("inf"), -float("inf"), float("nan"), float("-inf")]),
        torch.ones(5),
    ],
)
def test_replay_value_demand_fallbacks_keep_exact_replay_count(value):
    candidate_mask = torch.ones(5, dtype=torch.bool)
    dist = torch.tensor([100.0, 90.0, 80.0, 70.0, 65.0])

    selected_mask, stats = select_proximity_sources(
        candidate_mask,
        dist,
        n=3,
        mode="value_demand_rerank",
        value_score=value,
        replay_schedule={800: 4},
        iteration=800,
    )

    assert torch.equal(selected_mask, torch.tensor([True, True, True, True, False]))
    assert stats.selected_indices == (0, 1, 2, 3)
    assert stats.selected_src == 4
    assert stats.selected_new == 12
    assert stats.replay_exact_match is True


def test_replay_demand_gate_keeps_accepted_promotions_above_threshold():
    candidate_mask = torch.ones(4, dtype=torch.bool)
    dist = torch.tensor([200.0, 100.0, 95.0, 1.0])
    value = torch.tensor([0.0, 0.10, 0.90, 0.0])

    _, stats = select_proximity_sources(
        candidate_mask,
        dist,
        n=3,
        mode="value_demand_rerank",
        value_score=value,
        replay_schedule={900: 2},
        iteration=900,
        rerank_fraction=0.5,
        boundary_multiplier=2.0,
        demand_ratio=0.90,
    )

    assert stats.promotion_accepted == 1
    assert stats.min_P_ratio >= 0.90
    assert stats.selected_src == 2


def test_proximity_replay_log_contains_required_fields():
    _, stats = select_proximity_sources(
        torch.ones(5, dtype=torch.bool),
        torch.arange(5, dtype=torch.float32),
        replay_schedule={1000: 4},
        iteration=1000,
    )
    log_line = format_proximity_replay_log(1000, stats)

    for field in (
        "[ProximityReplay]",
        "iter=",
        "current=",
        "candidates=",
        "scheduled_src=",
        "selected_src=",
        "selected_new=",
        "dropped_src=",
        "retention=",
        "replay_active=",
        "exact_match=",
    ):
        assert field in log_line
    assert "scheduled_src=4" in log_line
    assert "selected_new=12" in log_line
    assert "exact_match=True" in log_line


def test_selected_sources_are_min_candidates_and_budget_div_n():
    stats = compute_proximity_budget(current=100, candidates=50, n=3, rho=0.10, enabled=True)

    assert stats.selected_src == min(50, 10 // 3)
    assert stats.selected_new == 9


def test_selected_new_never_exceeds_budget():
    stats = compute_proximity_budget(current=100, candidates=50, n=3, rho=0.10, enabled=True)

    assert stats.selected_new <= stats.budget_new


def test_candidates_less_than_budget_are_all_selected():
    candidate_mask = torch.tensor([
        True, False, True, False, False, False, False, False, False, False, False, False,
    ])
    dist = torch.tensor([
        1.0, 9.0, 2.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0,
    ])

    selected_mask, stats = select_proximity_sources(
        candidate_mask,
        dist,
        n=3,
        rho=1.0,
        enabled=True,
    )

    assert torch.equal(selected_mask, candidate_mask)
    assert stats.selected_src == 2
    assert stats.dropped_src == 0
    assert stats.budget_hit is False


def test_candidates_equal_budget_are_all_selected():
    candidate_mask = torch.tensor([
        True, True, True, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False, False,
    ])
    dist = torch.tensor([
        1.0, 3.0, 2.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
    ])

    selected_mask, stats = select_proximity_sources(
        candidate_mask,
        dist,
        n=3,
        rho=0.5,
        enabled=True,
    )

    assert torch.equal(selected_mask, candidate_mask)
    assert stats.selected_src == 3
    assert stats.selected_new == stats.budget_new
    assert stats.budget_hit is False


def test_candidates_over_budget_select_only_top_k_largest_dist():
    candidate_mask = torch.tensor([
        True, True, True, True, False, False, False, False, False, False, False, False,
    ])
    dist = torch.tensor([
        1.0, 5.0, 3.0, 2.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0,
    ])

    selected_mask, stats = select_proximity_sources(
        candidate_mask,
        dist,
        n=3,
        rho=0.5,
        enabled=True,
    )

    assert torch.equal(
        selected_mask,
        torch.tensor([False, True, True, False, False, False, False, False, False, False, False, False]),
    )
    assert stats.selected_src == 2
    assert stats.selected_new == 6
    assert stats.dropped_src == 2
    assert stats.budget_hit is True


def test_top_k_ranking_uses_dist_descending_within_candidates_only():
    candidate_mask = torch.tensor([
        False, True, True, True, True, False, False, False, False, False,
    ])
    dist = torch.tensor([
        100.0, 2.0, 9.0, 4.0, 7.0, 101.0, 102.0, 103.0, 104.0, 105.0,
    ])

    selected_mask, _ = select_proximity_sources(
        candidate_mask,
        dist,
        n=3,
        rho=0.6,
        enabled=True,
    )

    assert torch.equal(
        selected_mask,
        torch.tensor([False, False, True, False, True, False, False, False, False, False]),
    )


def test_zero_candidate_is_valid():
    candidate_mask = torch.tensor([False, False, False])
    dist = torch.tensor([3.0, 2.0, 1.0])

    selected_mask, stats = select_proximity_sources(
        candidate_mask,
        dist,
        n=3,
        rho=0.10,
        enabled=True,
    )

    assert torch.equal(selected_mask, candidate_mask)
    assert stats.candidates == 0
    assert stats.selected_src == 0
    assert stats.selected_new == 0


def test_budget_less_than_n_selects_zero_sources():
    candidate_mask = torch.tensor([True, True, True, True])
    dist = torch.tensor([4.0, 3.0, 2.0, 1.0])

    selected_mask, stats = select_proximity_sources(
        candidate_mask,
        dist,
        n=3,
        rho=0.1,
        enabled=True,
    )

    assert torch.equal(selected_mask, torch.tensor([False, False, False, False]))
    assert stats.budget_new == 0
    assert stats.selected_src == 0
    assert stats.selected_new == 0


@pytest.mark.parametrize("rho", [0, -0.1, 1.1])
def test_invalid_rho_raises_when_budget_enabled(rho):
    with pytest.raises(ValueError):
        compute_proximity_budget(current=100, candidates=10, n=3, rho=rho, enabled=True)


def test_budget_disabled_helper_preserves_candidate_mask_and_count():
    candidate_mask = torch.tensor([True, False, True, True])
    dist = torch.tensor([1.0, 99.0, 2.0, 3.0])

    selected_mask, stats = select_proximity_sources(
        candidate_mask,
        dist,
        n=3,
        rho=-1.0,
        enabled=False,
    )

    assert selected_mask is candidate_mask
    assert torch.equal(selected_mask, candidate_mask)
    assert stats.selected_src == 3
    assert stats.selected_new == 9
    assert stats.budget_active is False
    assert stats.budget_hit is False


def test_budget_diagnostics_values_are_consistent():
    stats = compute_proximity_budget(current=100, candidates=8, n=3, rho=0.10, enabled=True)
    log_line = format_proximity_budget_log(iteration=1800, stats=stats)

    assert stats.proposed >= stats.selected_new
    assert stats.dropped_src == stats.candidates - stats.selected_src
    assert "candidates=8" in log_line
    assert "proposed=24" in log_line
    assert "selected_new=9" in log_line
    assert "dropped_src=5" in log_line
    assert "budget_active=True" in log_line
    assert "budget_hit=True" in log_line


def test_budget_hit_false_all_selection_modes_pass_all_candidates():
    candidate_mask = torch.tensor([True, False, True, True] + [False] * 8)
    dist = torch.tensor([1.0, 99.0, 2.0, 3.0] + [0.0] * 8)
    value = torch.tensor([float("nan"), 999.0, float("inf"), -float("inf")] + [0.0] * 8)

    selected = []
    for mode in ("proximity_topk", "value_global", "value_rerank", "value_demand_rerank"):
        selected_mask, stats = select_proximity_sources(
            candidate_mask,
            dist,
            n=3,
            rho=1.0,
            enabled=True,
            mode=mode,
            value_score=value,
        )
        selected.append(selected_mask)
        assert stats.budget_hit is False
        assert torch.equal(selected_mask, candidate_mask)

    assert torch.equal(selected[0], selected[1])
    assert torch.equal(selected[1], selected[2])


def test_a1_proximity_topk_uses_dist_descending_not_index_order():
    candidate_mask = torch.tensor([True, True, True, True] + [False] * 8)
    dist = torch.tensor([1.0, 8.0, 5.0, 7.0] + [100.0] * 8)

    selected_mask, stats = select_proximity_sources(
        candidate_mask,
        dist,
        n=3,
        rho=0.5,
        enabled=True,
        mode="proximity_topk",
    )

    assert torch.equal(selected_mask, torch.tensor([False, True, False, True] + [False] * 8))
    assert stats.a1_indices == (1, 3)
    assert stats.a1_overlap == 1.0


def test_enable_budget_with_default_original_mode_is_legacy_a1_topk():
    candidate_mask = torch.tensor([True, True, True, True] + [False] * 8)
    dist = torch.tensor([1.0, 8.0, 5.0, 7.0] + [100.0] * 8)

    selected_mask, stats = select_proximity_sources(
        candidate_mask,
        dist,
        n=3,
        rho=0.5,
        enabled=True,
        mode="original",
    )

    assert torch.equal(selected_mask, torch.tensor([False, True, False, True] + [False] * 8))
    assert stats.mode == "proximity_topk"
    assert stats.selected_new == 6


def test_a2_value_global_keeps_candidate_hard_gate():
    candidate_mask = torch.tensor([True, True, False, True, True, False] + [False] * 12)
    dist = torch.tensor([6.0, 5.0, 100.0, 4.0, 3.0, 99.0] + [0.0] * 12)
    value = torch.tensor([0.1, 8.0, 999.0, 7.0, 6.0, 998.0] + [0.0] * 12)

    selected_mask, stats = select_proximity_sources(
        candidate_mask,
        dist,
        n=3,
        rho=0.5,
        enabled=True,
        mode="value_global",
        value_score=value,
    )

    assert torch.equal(selected_mask, torch.tensor([False, True, False, True, True, False] + [False] * 12))
    assert 2 not in stats.selected_indices
    assert 5 not in stats.selected_indices


def test_a3_protected_core_and_boundary_pool_are_enforced():
    candidate_mask = torch.tensor([True] * 10 + [False] * 20)
    dist = torch.tensor([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0] + [0.0] * 20)
    value = torch.tensor([0.0, 0.0, 0.1, 0.2, 90.0, 80.0, 70.0, 0.0, 0.0, 1000.0] + [0.0] * 20)

    selected_mask, stats = select_proximity_sources(
        candidate_mask,
        dist,
        n=3,
        rho=0.4,
        enabled=True,
        mode="value_rerank",
        value_score=value,
        rerank_fraction=0.5,
        boundary_multiplier=2.0,
    )

    assert stats.selected_src == 4
    assert {0, 1}.issubset(set(stats.selected_indices))
    assert {4, 5}.issubset(set(stats.selected_indices))
    assert 9 not in stats.selected_indices
    assert stats.boundary_indices == (2, 3, 4, 5)
    assert stats.a1_overlap >= 0.5


def test_a3_promoted_candidates_come_only_from_boundary_pool():
    candidate_mask = torch.tensor([True] * 10 + [False] * 20)
    dist = torch.tensor([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0] + [0.0] * 20)
    value = torch.tensor([0.0, 0.0, 0.1, 0.2, 90.0, 80.0, 70.0, 0.0, 0.0, 1000.0] + [0.0] * 20)

    _, stats = select_proximity_sources(
        candidate_mask,
        dist,
        n=3,
        rho=0.4,
        enabled=True,
        mode="value_rerank",
        value_score=value,
        rerank_fraction=0.5,
        boundary_multiplier=2.0,
    )

    promoted = set(stats.selected_indices) - set(stats.a1_indices)
    assert promoted
    assert promoted.issubset(set(stats.boundary_indices))
    assert 9 not in promoted


def test_value_global_all_zero_falls_back_to_proximity():
    candidate_mask = torch.tensor([True, True, True, True, True] + [False] * 7)
    dist = torch.tensor([1.0, 8.0, 5.0, 7.0, 6.0] + [0.0] * 7)
    value = torch.zeros(12)

    a1_mask, a1_stats = select_proximity_sources(
        candidate_mask,
        dist,
        n=3,
        rho=0.75,
        enabled=True,
        mode="proximity_topk",
    )
    a2_mask, a2_stats = select_proximity_sources(
        candidate_mask,
        dist,
        n=3,
        rho=0.75,
        enabled=True,
        mode="value_global",
        value_score=value,
    )

    assert a1_stats.a1_indices == (1, 3, 4)
    assert torch.equal(a2_mask, a1_mask)
    assert a2_stats.selected_indices == a1_stats.a1_indices
    assert a2_stats.value_selected_indices == ()
    assert a2_stats.selected_value_src == 0


def test_value_rerank_all_zero_boundary_preserves_proximity_order():
    candidate_mask = torch.tensor([True] * 8 + [False] * 4)
    dist = torch.tensor([1.0, 7.0, 10.0, 3.0, 2.0, 11.0, 4.0, 8.0] + [0.0] * 4)
    value = torch.zeros(12)

    selected_mask, stats = select_proximity_sources(
        candidate_mask,
        dist,
        n=3,
        rho=1.0,
        enabled=True,
        mode="value_rerank",
        value_score=value,
        rerank_fraction=0.5,
        boundary_multiplier=2.0,
    )

    assert stats.selected_src == 4
    assert stats.a1_indices == (5, 2, 7, 1)
    assert stats.boundary_indices == (7, 1, 6, 3)
    assert stats.selected_indices[:2] == (5, 2)
    assert stats.selected_indices[2:] == (7, 1)
    assert torch.equal(
        selected_mask,
        torch.tensor([False, True, True, False, False, True, False, True] + [False] * 4),
    )
    assert stats.value_selected_indices == ()
    assert stats.selected_value_src == 0


def test_equal_positive_value_falls_back_to_proximity():
    candidate_mask = torch.tensor([True, True, True, True, True] + [False] * 7)
    dist = torch.tensor([1.0, 8.0, 5.0, 7.0, 6.0] + [0.0] * 7)
    value = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0] + [0.0] * 7)

    selected_mask, stats = select_proximity_sources(
        candidate_mask,
        dist,
        n=3,
        rho=0.75,
        enabled=True,
        mode="value_global",
        value_score=value,
    )

    assert stats.selected_indices == (1, 3, 4)
    assert not torch.equal(
        selected_mask,
        torch.tensor([True, True, True, False, False] + [False] * 7),
    )
    assert stats.value_selected_indices == ()
    assert stats.selected_value_src == 0


def test_informative_value_still_reranks():
    candidate_mask = torch.tensor([True] * 8 + [False] * 4)
    dist = torch.tensor([11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0] + [0.0] * 4)
    value = torch.tensor([0.1, 0.2, 0.3, 7.0, 8.0, 9.0, 6.0, 5.0] + [0.0] * 4)

    a1_mask, a1_stats = select_proximity_sources(
        candidate_mask,
        dist,
        n=3,
        rho=1.0,
        enabled=True,
        mode="proximity_topk",
    )
    a2_mask, a2_stats = select_proximity_sources(
        candidate_mask,
        dist,
        n=3,
        rho=1.0,
        enabled=True,
        mode="value_global",
        value_score=value,
    )
    a3_mask, a3_stats = select_proximity_sources(
        candidate_mask,
        dist,
        n=3,
        rho=1.0,
        enabled=True,
        mode="value_rerank",
        value_score=value,
        rerank_fraction=0.5,
        boundary_multiplier=2.0,
    )

    assert a1_stats.a1_indices == (0, 1, 2, 3)
    assert not torch.equal(a2_mask, a1_mask)
    assert a2_stats.selected_indices == (5, 4, 3, 6)
    assert a2_stats.value_selected_indices == (5, 4, 3, 6)
    assert not torch.equal(a3_mask, a1_mask)
    assert a3_stats.selected_indices == (0, 1, 5, 4)
    assert a3_stats.value_selected_indices == (5, 4)


def test_partial_positive_value_fills_remaining_slots_by_proximity():
    candidate_mask = torch.tensor([False, False, True, False, True, False, False, True, False, True, False, False])
    dist = torch.tensor([0.0, 0.0, 3.0, 0.0, 1.0, 0.0, 0.0, 4.0, 0.0, 2.0, 0.0, 0.0])
    value = torch.zeros(12)
    value[2] = 0.8

    selected_mask, stats = select_proximity_sources(
        candidate_mask,
        dist,
        n=3,
        rho=0.75,
        enabled=True,
        mode="value_global",
        value_score=value,
    )

    assert stats.proximity_rank == (7, 2, 9, 4)
    assert stats.selected_indices == (2, 7, 9)
    assert stats.value_selected_indices == (2,)
    assert stats.selected_value_src == 1
    assert torch.equal(
        selected_mask,
        torch.tensor([False, False, True, False, False, False, False, True, False, True, False, False]),
    )


def test_positive_value_tie_uses_proximity_as_secondary_rank():
    candidate_mask = torch.tensor([False, False, True, False, True, False, False, True, False, True])
    dist = torch.tensor([0.0, 0.0, 3.0, 0.0, 1.0, 0.0, 0.0, 4.0, 0.0, 2.0])
    value = torch.zeros(10)
    value[7] = 0.8
    value[2] = 0.8

    selected_mask, stats = select_proximity_sources(
        candidate_mask,
        dist,
        n=3,
        rho=0.3,
        enabled=True,
        mode="value_global",
        value_score=value,
    )

    assert stats.proximity_rank == (7, 2, 9, 4)
    assert stats.selected_indices == (7,)
    assert stats.value_selected_indices == (7,)
    assert torch.equal(
        selected_mask,
        torch.tensor([False, False, False, False, False, False, False, True, False, False]),
    )


def test_value_rerank_partial_positive_stays_inside_boundary_pool():
    candidate_mask = torch.tensor([True] * 8 + [False] * 4)
    dist = torch.tensor([1.0, 7.0, 10.0, 3.0, 2.0, 11.0, 4.0, 8.0] + [0.0] * 4)
    value = torch.zeros(12)
    value[6] = 0.8
    value[0] = 9.0
    value[4] = 8.0

    selected_mask, stats = select_proximity_sources(
        candidate_mask,
        dist,
        n=3,
        rho=1.0,
        enabled=True,
        mode="value_rerank",
        value_score=value,
        rerank_fraction=0.5,
        boundary_multiplier=2.0,
    )

    assert stats.a1_indices == (5, 2, 7, 1)
    assert stats.selected_indices[:2] == (5, 2)
    assert stats.boundary_indices == (7, 1, 6, 3)
    assert stats.selected_indices[2:] == (6, 7)
    assert stats.value_selected_indices == (6,)
    assert 0 not in stats.selected_indices
    assert 4 not in stats.selected_indices
    assert torch.equal(
        selected_mask,
        torch.tensor([False, False, True, False, False, True, True, True] + [False] * 4),
    )


def test_nan_inf_value_falls_back_to_proximity_without_underselecting():
    candidate_mask = torch.tensor([True, True, True, True, True] + [False] * 10)
    dist = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0] + [0.0] * 10)
    value = torch.tensor([float("nan"), float("inf"), -float("inf"), float("nan"), float("inf")] + [0.0] * 10)

    selected_mask, stats = select_proximity_sources(
        candidate_mask,
        dist,
        n=3,
        rho=0.6,
        enabled=True,
        mode="value_global",
        value_score=value,
    )

    assert stats.selected_src == 3
    assert torch.equal(selected_mask, torch.tensor([True, True, True, False, False] + [False] * 10))
    assert stats.value_selected_indices == ()
    assert stats.selected_value_src == 0


def test_rerank_fraction_zero_degenerates_to_a1():
    candidate_mask = torch.tensor([True] * 6 + [False] * 6)
    dist = torch.tensor([6.0, 5.0, 4.0, 3.0, 2.0, 1.0] + [0.0] * 6)
    value = torch.tensor([0.0, 0.0, 0.0, 9.0, 8.0, 7.0] + [0.0] * 6)

    a1, _ = select_proximity_sources(candidate_mask, dist, n=3, rho=0.5, enabled=True, mode="proximity_topk")
    a3, _ = select_proximity_sources(
        candidate_mask,
        dist,
        n=3,
        rho=0.5,
        enabled=True,
        mode="value_rerank",
        value_score=value,
        rerank_fraction=0.0,
    )

    assert torch.equal(a1, a3)


def test_kt_one_and_empty_candidates_are_safe():
    selected_one, stats_one = select_proximity_sources(
        torch.tensor([True, True, True]),
        torch.tensor([3.0, 2.0, 1.0]),
        n=3,
        rho=1.0,
        enabled=True,
        mode="value_rerank",
        value_score=torch.tensor([0.0, 9.0, 8.0]),
    )
    selected_empty, stats_empty = select_proximity_sources(
        torch.tensor([False, False]),
        torch.tensor([2.0, 1.0]),
        n=3,
        rho=0.10,
        enabled=True,
        mode="value_rerank",
        value_score=torch.tensor([1.0, 2.0]),
    )

    assert selected_one.sum().item() == 1
    assert stats_one.selected_src == 1
    assert selected_empty.sum().item() == 0
    assert stats_empty.selected_src == 0


def test_selection_is_deterministic_for_ties():
    candidate_mask = torch.ones(5, dtype=torch.bool)
    dist = torch.tensor([5.0, 5.0, 4.0, 4.0, 3.0])
    value = torch.tensor([1.0, 1.0, 2.0, 2.0, 9.0])

    first, _ = select_proximity_sources(candidate_mask, dist, n=3, rho=0.6, enabled=True, mode="value_rerank", value_score=value)
    second, _ = select_proximity_sources(candidate_mask, dist, n=3, rho=0.6, enabled=True, mode="value_rerank", value_score=value)

    assert torch.equal(first, second)


def test_value_helpers_normalize_components_and_format_diagnostics():
    values = torch.tensor([0.0, 1.0, 100.0])
    normalized = robust_normalize(values, 0.0, 1.0)
    utility = build_fsgs_available_value_score(
        dist=torch.tensor([1.0, 2.0, 3.0]),
        scaling=torch.tensor([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]]),
        confidence=torch.tensor([[1.0], [0.5], [0.0]]),
        denom=torch.tensor([[1.0], [2.0], [3.0]]),
    )

    assert torch.isfinite(normalized).all()
    assert torch.isfinite(utility).all()
    assert utility.shape == (3,)


def test_new_diagnostic_lines_contain_required_fields():
    candidate_mask = torch.ones(6, dtype=torch.bool)
    dist = torch.tensor([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    value = torch.tensor([0.0, 0.0, 1.0, 9.0, 8.0, 7.0])
    _, stats = select_proximity_sources(
        candidate_mask,
        dist,
        n=3,
        rho=0.5,
        enabled=True,
        mode="value_rerank",
        value_score=value,
    )
    stats.proximity_values = dist
    stats.value_values = value

    rerank_log = format_value_rerank_diag(1200, stats)
    promotion_log = format_value_promotion_diag(1200, stats)

    for field in (
        "iter=",
        "mode=",
        "current=",
        "candidates=",
        "proposed_new=",
        "budget_new=",
        "budget_src=",
        "budget_active=",
        "budget_hit=",
        "selected_src=",
        "selected_new=",
        "dropped_src=",
        "core_src=",
        "boundary_src=",
        "rerank_slots=",
        "selected_value_src=",
        "a1_overlap=",
        "mean_P_all=",
        "mean_P_selected=",
        "mean_U_all=",
        "mean_U_boundary=",
        "mean_U_selected=",
    ):
        assert field in rerank_log
    for field in ("promoted=", "displaced=", "promoted_rank_mean=", "mean_U_promoted=", "mean_P_displaced="):
        assert field in promotion_log


def test_resolve_proximity_selection_mode_matches_training_edge_map_gate():
    assert resolve_proximity_selection_mode("original", False) == "original"
    assert resolve_proximity_selection_mode("original", True) == "proximity_topk"
    assert resolve_proximity_selection_mode("proximity_topk", False) == "proximity_topk"
    assert resolve_proximity_selection_mode("value_global", False) == "value_global"
    assert resolve_proximity_selection_mode("value_rerank", True) == "value_rerank"


def test_requires_value_features_only_for_value_modes():
    assert requires_value_features("original") is False
    assert requires_value_features("proximity_topk") is False
    assert requires_value_features("value_global") is True
    assert requires_value_features("value_rerank") is True
    assert requires_value_features("value_demand_rerank") is True


def test_value_demand_rerank_mode_registration_and_ratio_validation():
    args = type("Args", (), {
        "proximity_selection_mode": "value_demand_rerank",
        "proximity_growth_ratio": 0.1,
        "value_rerank_fraction": 0.25,
        "value_boundary_multiplier": 2.0,
        "value_demand_ratio": 0.90,
        "value_tau_e": 1.0,
        "value_tau_s": 3.0,
        "value_lambda_r": 1.0,
        "normalization_low_quantile": 0.05,
        "normalization_high_quantile": 0.95,
        "knn_k": 12,
    })()

    assert validate_optimization_params(args) is args
    assert resolve_proximity_selection_mode("value_demand_rerank", True) == "value_demand_rerank"

    for ratio in (0.0, -0.1, 1.1):
        args.value_demand_ratio = ratio
        with pytest.raises(ValueError, match="value_demand_ratio"):
            validate_optimization_params(args)


def test_candidate_capacity_argument_validation():
    args = type("Args", (), {
        "proximity_selection_mode": "proximity_topk",
        "enable_proximity_budget": False,
        "enable_proximity_candidate_capacity": True,
        "proximity_candidate_keep_ratio": 0.8,
        "proximity_growth_ratio": 0.1,
        "value_rerank_fraction": 0.25,
        "value_boundary_multiplier": 2.0,
        "value_demand_ratio": 0.90,
        "value_tau_e": 1.0,
        "value_tau_s": 3.0,
        "value_lambda_r": 1.0,
        "normalization_low_quantile": 0.05,
        "normalization_high_quantile": 0.95,
        "knn_k": 12,
    })()

    assert validate_optimization_params(args) is args

    args.proximity_growth_ratio = -1.0
    assert validate_optimization_params(args) is args
    args.proximity_growth_ratio = 0.1

    args.enable_proximity_budget = True
    with pytest.raises(ValueError, match="cannot both be enabled"):
        validate_optimization_params(args)

    args.enable_proximity_budget = False
    for ratio in (0.0, -0.1, 1.1):
        args.proximity_candidate_keep_ratio = ratio
        with pytest.raises(ValueError, match="proximity_candidate_keep_ratio"):
            validate_optimization_params(args)


def test_value_observation_source_validation_defaults_to_lifetime():
    args = type("Args", (), {
        "proximity_selection_mode": "value_global",
        "proximity_growth_ratio": 0.1,
        "value_rerank_fraction": 0.25,
        "value_boundary_multiplier": 2.0,
        "value_demand_ratio": 0.90,
        "value_tau_e": 1.0,
        "value_tau_s": 3.0,
        "value_lambda_r": 1.0,
        "normalization_low_quantile": 0.05,
        "normalization_high_quantile": 0.95,
        "knn_k": 12,
    })()

    assert validate_optimization_params(args) is args
    assert getattr(args, "value_score_variant", "obdkr") == "obdkr"

    args.value_observation_source = "lifetime"
    assert validate_optimization_params(args) is args
    args.value_observation_source = "recent"
    assert validate_optimization_params(args) is args
    args.value_observation_source = "bad"
    with pytest.raises(ValueError, match="value_observation_source"):
        validate_optimization_params(args)


def test_value_score_variant_validation_defaults_to_obdkr():
    args = type("Args", (), {
        "proximity_selection_mode": "value_demand_rerank",
        "proximity_growth_ratio": 0.1,
        "value_rerank_fraction": 0.25,
        "value_boundary_multiplier": 2.0,
        "value_demand_ratio": 0.90,
        "value_tau_e": 1.0,
        "value_tau_s": 3.0,
        "value_lambda_r": 1.0,
        "normalization_low_quantile": 0.05,
        "normalization_high_quantile": 0.95,
        "knn_k": 12,
    })()

    assert validate_optimization_params(args) is args

    args.value_score_variant = "obdkr"
    assert validate_optimization_params(args) is args
    args.value_score_variant = "structural"
    assert validate_optimization_params(args) is args
    args.value_score_variant = "bad"
    with pytest.raises(ValueError, match="value_score_variant"):
        validate_optimization_params(args)


def test_value_demand_rerank_value_none_exact_a1_fallback():
    candidate_mask = torch.tensor([True, True, True, True, False])
    dist = torch.tensor([200.0, 100.0, 70.0, 1.0, 0.0])

    a1_mask, a1_stats = select_proximity_sources(
        candidate_mask, dist, n=1, rho=0.4, enabled=True, mode="proximity_topk"
    )
    selected_mask, stats = select_proximity_sources(
        candidate_mask,
        dist,
        n=1,
        rho=0.4,
        enabled=True,
        mode="value_demand_rerank",
        value_score=None,
    )

    assert torch.equal(selected_mask, a1_mask)
    assert stats.selected_indices == a1_stats.a1_indices


def test_value_demand_rerank_rejects_low_demand_challenger():
    candidate_mask = torch.tensor([True, True, True, True])
    dist = torch.tensor([200.0, 100.0, 70.0, 1.0])
    value = torch.tensor([0.0, 0.10, 0.90, 0.0])

    selected_mask, stats = select_proximity_sources(
        candidate_mask,
        dist,
        n=1,
        rho=0.5,
        enabled=True,
        mode="value_demand_rerank",
        value_score=value,
        rerank_fraction=0.5,
        boundary_multiplier=2.0,
        demand_ratio=0.90,
    )

    assert stats.selected_indices == (0, 1)
    assert selected_mask[1]
    assert not selected_mask[2]
    assert stats.promotion_accepted == 0
    assert stats.promotion_rejected_demand == 1


def test_value_demand_rerank_accepts_demand_compatible_challenger():
    candidate_mask = torch.tensor([True, True, True, True])
    dist = torch.tensor([200.0, 100.0, 95.0, 1.0])
    value = torch.tensor([0.0, 0.10, 0.90, 0.0])

    selected_mask, stats = select_proximity_sources(
        candidate_mask,
        dist,
        n=1,
        rho=0.5,
        enabled=True,
        mode="value_demand_rerank",
        value_score=value,
        rerank_fraction=0.5,
        boundary_multiplier=2.0,
        demand_ratio=0.90,
    )

    assert stats.selected_indices == (0, 2)
    assert selected_mask[2]
    assert not selected_mask[1]
    assert stats.demand_promoted_indices == (2,)
    assert stats.demand_displaced_indices == (1,)
    assert stats.min_P_ratio >= 0.90


def test_value_demand_rerank_rejects_non_improving_value():
    candidate_mask = torch.tensor([True, True, True, True])
    dist = torch.tensor([200.0, 100.0, 95.0, 1.0])
    value = torch.tensor([0.0, 0.50, 0.40, 0.0])

    _, stats = select_proximity_sources(
        candidate_mask,
        dist,
        n=1,
        rho=0.5,
        enabled=True,
        mode="value_demand_rerank",
        value_score=value,
        rerank_fraction=0.5,
        boundary_multiplier=2.0,
        demand_ratio=0.90,
    )

    assert stats.selected_indices == (0, 1)
    assert stats.promotion_accepted == 0
    assert stats.promotion_rejected_value == 1


def test_value_demand_rerank_ratio_boundary():
    candidate_mask = torch.tensor([True, True, True, True, True])
    value = torch.tensor([0.0, 0.10, 0.90, 0.95, 0.0])

    _, accepted = select_proximity_sources(
        candidate_mask,
        torch.tensor([200.0, 100.0, 90.0, 89.999, 1.0]),
        n=1,
        rho=0.4,
        enabled=True,
        mode="value_demand_rerank",
        value_score=value,
        rerank_fraction=0.5,
        boundary_multiplier=3.0,
        demand_ratio=0.90,
    )
    _, rejected = select_proximity_sources(
        candidate_mask,
        torch.tensor([200.0, 100.0, 89.999, 1.0, 0.5]),
        n=1,
        rho=0.4,
        enabled=True,
        mode="value_demand_rerank",
        value_score=value,
        rerank_fraction=0.5,
        boundary_multiplier=2.0,
        demand_ratio=0.90,
    )

    assert accepted.demand_promoted_indices == (2,)
    assert accepted.min_P_ratio == pytest.approx(0.90)
    assert rejected.promotion_accepted == 0


def test_value_demand_rerank_no_cascade_replacement():
    candidate_mask = torch.tensor([True] * 8)
    dist = torch.tensor([200.0, 190.0, 100.0, 95.0, 94.0, 93.0, 2.0, 1.0])
    value = torch.tensor([0.0, 0.0, 0.10, 0.20, 0.90, 0.80, 0.0, 0.0])

    _, stats = select_proximity_sources(
        candidate_mask,
        dist,
        n=1,
        rho=0.5,
        enabled=True,
        mode="value_demand_rerank",
        value_score=value,
        rerank_fraction=0.5,
        boundary_multiplier=2.0,
        demand_ratio=0.90,
    )

    assert stats.demand_displaced_indices
    assert set(stats.demand_displaced_indices).issubset({2, 3})
    assert set(stats.demand_displaced_indices).isdisjoint(set(stats.demand_promoted_indices))
    assert stats.selected_src == 4


def test_value_demand_rerank_exact_budget_and_selected_subset():
    candidate_mask = torch.tensor([True, True, True, True, True, False, False])
    dist = torch.tensor([200.0, 100.0, 95.0, 94.0, 1.0, 999.0, 998.0])
    value = torch.tensor([0.0, 0.10, 0.90, 0.80, 0.0, 1000.0, 999.0])

    selected_mask, stats = select_proximity_sources(
        candidate_mask,
        dist,
        n=1,
        rho=3 / 7,
        enabled=True,
        mode="value_demand_rerank",
        value_score=value,
        rerank_fraction=0.5,
        boundary_multiplier=2.0,
        demand_ratio=0.90,
    )

    assert stats.selected_src == 3
    assert int(selected_mask.sum().item()) == 3
    assert torch.logical_or(~selected_mask, candidate_mask).all()


def test_value_demand_rerank_deterministic_tie_prefers_higher_proximity():
    candidate_mask = torch.tensor([True, True, True, True, True])
    dist = torch.tensor([200.0, 100.0, 95.0, 96.0, 1.0])
    value = torch.tensor([0.0, 0.10, 0.90, 0.90, 0.0])

    _, stats = select_proximity_sources(
        candidate_mask,
        dist,
        n=1,
        rho=0.4,
        enabled=True,
        mode="value_demand_rerank",
        value_score=value,
        rerank_fraction=0.5,
        boundary_multiplier=3.0,
        demand_ratio=0.90,
    )

    assert stats.demand_promoted_indices == (3,)


def test_value_demand_rerank_diagnostics_consistency():
    candidate_mask = torch.tensor([True, True, True, True])
    dist = torch.tensor([200.0, 100.0, 95.0, 1.0])
    value = torch.tensor([0.0, 0.10, 0.90, 0.0])

    _, stats = select_proximity_sources(
        candidate_mask,
        dist,
        n=1,
        rho=0.5,
        enabled=True,
        mode="value_demand_rerank",
        value_score=value,
        rerank_fraction=0.5,
        boundary_multiplier=2.0,
        demand_ratio=0.90,
    )
    log_line = format_demand_preserve_diag(900, stats)

    assert stats.promotion_accepted == len(stats.demand_promoted_indices)
    assert len(stats.demand_promoted_indices) == len(stats.demand_displaced_indices)
    assert stats.min_P_ratio >= 0.90 - 1e-8
    for field in (
        "demand_ratio_threshold=",
        "baseline_boundary_src=",
        "challenger_src=",
        "promotion_attempted=",
        "promotion_accepted=",
        "promotion_rejected_demand=",
        "promotion_rejected_value=",
        "mean_P_ratio=",
        "mean_P_gap=",
        "mean_relative_P_gap=",
        "a1_overlap=",
    ):
        assert field in log_line


def test_candidate_capacity_log_contains_required_fields():
    stats = compute_proximity_candidate_capacity(
        current=1000,
        candidates=55,
        n=3,
        keep_ratio=0.8,
        enabled=True,
    )
    log_line = format_proximity_capacity_log(1200, stats)

    for field in (
        "[ProximityCapacity]",
        "iter=",
        "current=",
        "candidates=",
        "keep_ratio=",
        "capacity_src=",
        "selected_src=",
        "selected_new=",
        "dropped_src=",
        "retention=",
        "capacity_hit=",
    ):
        assert field in log_line
    assert "capacity_src=44" in log_line
    assert "selected_new=132" in log_line
    assert "retention=0.800000" in log_line
    assert "capacity_hit=True" in log_line


def test_candidate_capacity_a1_and_a2_select_same_counts():
    candidate_mask = torch.ones(10, dtype=torch.bool)
    dist = torch.tensor([100.0, 90.0, 80.0, 70.0, 65.0, 60.0, 55.0, 50.0, 10.0, 1.0])
    value = torch.tensor([0.0, 0.0, 0.0, 0.10, 5.0, 0.0, 0.0, 0.0, 9.0, 8.0])

    _, a1_stats = select_proximity_sources(
        candidate_mask,
        dist,
        n=3,
        enabled=False,
        mode="proximity_topk",
        candidate_capacity_enabled=True,
        candidate_keep_ratio=0.8,
    )
    _, a2_stats = select_proximity_sources(
        candidate_mask,
        dist,
        n=3,
        enabled=False,
        mode="value_demand_rerank",
        value_score=value,
        candidate_capacity_enabled=True,
        candidate_keep_ratio=0.8,
    )

    assert a1_stats.selected_src == a2_stats.selected_src == 8
    assert a1_stats.selected_new == a2_stats.selected_new == 24


def test_candidate_capacity_value_demand_changes_who_not_how_many():
    candidate_mask = torch.ones(5, dtype=torch.bool)
    dist = torch.tensor([100.0, 90.0, 80.0, 70.0, 65.0])
    value = torch.tensor([0.0, 0.0, 0.0, 0.10, 5.0])

    a1_mask, a1_stats = select_proximity_sources(
        candidate_mask,
        dist,
        n=3,
        mode="proximity_topk",
        candidate_capacity_enabled=True,
        candidate_keep_ratio=0.8,
    )
    a2_mask, a2_stats = select_proximity_sources(
        candidate_mask,
        dist,
        n=3,
        mode="value_demand_rerank",
        value_score=value,
        candidate_capacity_enabled=True,
        candidate_keep_ratio=0.8,
        rerank_fraction=0.25,
        boundary_multiplier=2.0,
        demand_ratio=0.90,
    )

    assert a1_stats.selected_indices == (0, 1, 2, 3)
    assert a2_stats.selected_indices == (0, 1, 2, 4)
    assert not torch.equal(a1_mask, a2_mask)
    assert a1_stats.selected_src == a2_stats.selected_src == 4
    assert a1_stats.selected_new == a2_stats.selected_new == 12


@pytest.mark.parametrize(
    "value",
    [
        None,
        torch.tensor([float("nan"), float("inf"), -float("inf"), float("nan"), float("-inf")]),
        torch.ones(5),
    ],
)
def test_candidate_capacity_value_demand_exact_a1_fallbacks(value):
    candidate_mask = torch.ones(5, dtype=torch.bool)
    dist = torch.tensor([100.0, 90.0, 80.0, 70.0, 65.0])

    a1_mask, a1_stats = select_proximity_sources(
        candidate_mask,
        dist,
        n=3,
        mode="proximity_topk",
        candidate_capacity_enabled=True,
        candidate_keep_ratio=0.8,
    )
    selected_mask, stats = select_proximity_sources(
        candidate_mask,
        dist,
        n=3,
        mode="value_demand_rerank",
        value_score=value,
        candidate_capacity_enabled=True,
        candidate_keep_ratio=0.8,
    )

    assert torch.equal(selected_mask, a1_mask)
    assert stats.selected_indices == a1_stats.a1_indices
    assert stats.selected_src == a1_stats.selected_src
    assert stats.selected_new == a1_stats.selected_new

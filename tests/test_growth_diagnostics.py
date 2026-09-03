import torch

from utils.growth_diagnostics import (
    GrowthDiagnostics,
    count_proximity_proposed,
    count_split_candidates,
    format_proximity_growth_log,
)


def test_split_total_uses_logical_or_without_double_counting_overlap():
    split_gradient = torch.tensor([True, True, False, False])
    split_sparse = torch.tensor([False, True, True, False])
    split_total = torch.logical_or(split_gradient, split_sparse)

    split_grad_count, split_sparse_count, split_total_count = count_split_candidates(
        split_gradient, split_sparse, split_total
    )

    assert split_grad_count == 2
    assert split_sparse_count == 2
    assert split_total_count == 3


def test_proximity_proposed_multiplies_sources_by_n():
    assert count_proximity_proposed(4, n=3) == 12


def test_proximity_growth_log_format_is_oom_safe_line():
    assert format_proximity_growth_log(
        iteration=600,
        num_before=19397,
        proximity_sources=5,
        proximity_proposed=15,
    ) == (
        "[GrowthDiagProximity] iter=600 before=19397 "
        "proximity_src=5 proximity_new=15"
    )


def test_net_growth_uses_after_prune_minus_before():
    diagnostics = GrowthDiagnostics(
        iteration=600,
        num_before=10,
        num_after_prune=14,
    )

    assert diagnostics.net_growth == 4
    assert "net=4" in diagnostics.format_log()


def test_zero_candidate_statistics_are_valid():
    split_gradient = torch.tensor([False, False, False])
    split_sparse = torch.tensor([False, False, False])
    split_total = torch.tensor([False, False, False])

    assert count_split_candidates(split_gradient, split_sparse, split_total) == (0, 0, 0)
    assert count_proximity_proposed(0, n=3) == 0


def test_diagnostics_do_not_modify_input_masks():
    split_gradient = torch.tensor([True, False, True, False])
    split_sparse = torch.tensor([False, True, True, False])
    split_total = torch.logical_or(split_gradient, split_sparse)
    split_gradient_before = split_gradient.clone()
    split_sparse_before = split_sparse.clone()
    split_total_before = split_total.clone()

    count_split_candidates(split_gradient, split_sparse, split_total)

    assert torch.equal(split_gradient, split_gradient_before)
    assert torch.equal(split_sparse, split_sparse_before)
    assert torch.equal(split_total, split_total_before)

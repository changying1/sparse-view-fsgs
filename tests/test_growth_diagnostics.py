import torch
import pytest

from utils.growth_diagnostics import (
    GrowthDiagnostics,
    compute_child_structure_diagnostics,
    count_proximity_proposed,
    count_split_candidates,
    format_child_structure_diag,
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
        "proximity_src=5 proximity_proposed=15"
    )


def test_net_growth_uses_after_prune_minus_before():
    diagnostics = GrowthDiagnostics(
        iteration=600,
        num_before=10,
        num_after_prune=14,
        proximity_proposed=12,
        proximity_selected_sources=2,
        proximity_selected_new=6,
    )

    assert diagnostics.net_growth == 4
    assert "proximity_proposed=12" in diagnostics.format_log()
    assert "proximity_new=6" in diagnostics.format_log()
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


def test_child_structure_diagnostics_synthetic_topn_gain_and_quantiles():
    selected = torch.tensor([True, False, False, False, False, False])
    xyz = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ]
    )
    current_edge_scores = torch.tensor(
        [
            [0.1, 0.3, 0.9],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    pool_edge_scores = torch.tensor(
        [
            [0.1, 0.3, 0.9, 0.8, 0.7],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    nearest = torch.tensor(
        [
            [1, 2, 3],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
        ],
        dtype=torch.long,
    )
    pool = torch.tensor(
        [
            [1, 2, 3, 4, 5],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        ],
        dtype=torch.long,
    )

    stats = compute_child_structure_diagnostics(
        1200,
        selected,
        current_edge_scores,
        pool_edge_scores,
        nearest,
        pool,
        xyz=xyz,
    )

    assert stats["selected_sources"] == 1
    assert stats["children"] == 3
    assert stats["pool_k"] == 5
    assert stats["current_G_mean"] == pytest.approx(torch.tensor([0.1, 0.3, 0.9]).mean().item())
    assert stats["low_G025_ratio"] == pytest.approx(1 / 3)
    assert stats["low_G050_ratio"] == pytest.approx(2 / 3)
    assert stats["struct_topN_G_mean"] == pytest.approx(torch.tensor([0.9, 0.8, 0.7]).mean().item())
    assert stats["current_topN_G_mean"] == pytest.approx(torch.tensor([0.1, 0.3, 0.9]).mean().item())
    assert stats["mean_G_gain"] == pytest.approx(stats["struct_topN_G_mean"] - stats["current_topN_G_mean"])
    assert stats["target_overlap_ratio"] == pytest.approx(1 / 3)
    assert stats["changed_target_source_ratio"] == 1.0
    assert stats["current_target_dist_mean"] == pytest.approx(2.0)
    assert stats["struct_topN_target_dist_mean"] == pytest.approx(4.0)
    assert stats["mean_distance_ratio"] == pytest.approx(2.0)
    assert stats["distance_ratio_q25"] == pytest.approx(2.0)
    assert stats["distance_ratio_q50"] == pytest.approx(2.0)
    assert stats["distance_ratio_q75"] == pytest.approx(2.0)
    assert stats["dist_ratio_gt_125_ratio"] == pytest.approx(1.0)
    assert stats["dist_ratio_gt_150_ratio"] == pytest.approx(1.0)
    assert stats["current_target_pool_coverage"] == pytest.approx(1.0)
    assert stats["full_current_pool_coverage_ratio"] == pytest.approx(1.0)
    assert stats["locality_factor"] == pytest.approx(1.25)
    assert stats["locality_candidate_mean"] == pytest.approx(3.0)
    assert stats["locality_full_candidate_ratio"] == pytest.approx(1.0)
    assert stats["locality_struct_topN_G_mean"] == pytest.approx(torch.tensor([0.9, 0.3, 0.1]).mean().item())
    assert stats["locality_mean_G_gain"] == pytest.approx(0.0, abs=1e-6)
    assert stats["locality_target_overlap_ratio"] == pytest.approx(1.0)
    assert stats["locality_changed_target_source_ratio"] == pytest.approx(0.0)
    assert stats["locality_mean_distance_ratio"] == pytest.approx(1.0)


def test_child_structure_diagnostics_locality_constrained_shadow_topn():
    selected = torch.tensor([True, False, False, False, False, False, True, False, False, False, True, False, False, False, False])
    xyz = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.5, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [101.0, 0.0, 0.0],
            [102.0, 0.0, 0.0],
            [103.0, 0.0, 0.0],
            [200.0, 0.0, 0.0],
            [201.0, 0.0, 0.0],
            [202.0, 0.0, 0.0],
            [203.0, 0.0, 0.0],
            [204.0, 0.0, 0.0],
        ]
    )
    current_edge_scores = torch.zeros((15, 3))
    current_edge_scores[0] = torch.tensor([0.1, 0.2, 0.3])
    current_edge_scores[6] = torch.tensor([0.1, 0.1, 0.1])
    current_edge_scores[10] = torch.tensor([0.4, 0.5, 0.6])
    pool_edge_scores = torch.zeros((15, 5))
    pool_edge_scores[0] = torch.tensor([0.1, 0.2, 0.3, 0.9, 1.0])
    pool_edge_scores[6] = torch.tensor([0.9, 0.8, 1.0, 0.7, 0.6])
    pool_edge_scores[10] = torch.tensor([0.4, 0.5, 0.6, 0.1, 0.0])
    nearest = torch.zeros((15, 3), dtype=torch.long)
    nearest[0] = torch.tensor([1, 2, 3])
    nearest[6] = torch.tensor([7, 8, 9])
    nearest[10] = torch.tensor([11, 12, 13])
    pool = torch.zeros((15, 5), dtype=torch.long)
    pool[0] = torch.tensor([1, 2, 3, 4, 5])
    pool[6] = torch.tensor([7, 8, 5, 1, 2])
    pool[10] = torch.tensor([11, 12, 13, 14, 0])
    value_components = {"D": torch.tensor([0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0])}

    stats = compute_child_structure_diagnostics(
        1200,
        selected,
        current_edge_scores,
        pool_edge_scores,
        nearest,
        pool,
        value_components=value_components,
        xyz=xyz,
    )

    source0_locality_g = torch.tensor([0.9, 0.3, 0.2]).mean().item()
    source0_current_g = torch.tensor([0.1, 0.2, 0.3]).mean().item()
    source0_locality_ratio = ((3.5 + 3.0 + 2.0) / 3.0) / 2.0
    source10_locality_g = torch.tensor([0.6, 0.5, 0.4]).mean().item()
    source10_current_g = torch.tensor([0.4, 0.5, 0.6]).mean().item()
    locality_gains = torch.tensor([source0_locality_g - source0_current_g, source10_locality_g - source10_current_g])

    assert stats["locality_candidate_mean"] == pytest.approx((4 + 2 + 3) / 3)
    assert stats["locality_full_candidate_ratio"] == pytest.approx(2 / 3)
    assert stats["locality_struct_topN_G_mean"] == pytest.approx((source0_locality_g + source10_locality_g) / 2)
    assert stats["locality_mean_G_gain"] == pytest.approx(locality_gains.mean().item())
    assert stats["locality_G_gain_q25"] == pytest.approx(torch.quantile(locality_gains, 0.25).item())
    assert stats["locality_G_gain_q50"] == pytest.approx(torch.quantile(locality_gains, 0.50).item())
    assert stats["locality_G_gain_q75"] == pytest.approx(torch.quantile(locality_gains, 0.75).item())
    assert stats["locality_gain_gt_005_ratio"] == pytest.approx(0.5)
    assert stats["locality_gain_gt_010_ratio"] == pytest.approx(0.5)
    assert stats["locality_target_overlap_ratio"] == pytest.approx((2 + 3) / 6)
    assert stats["locality_changed_target_source_ratio"] == pytest.approx(0.5)
    assert stats["locality_mean_distance_ratio"] == pytest.approx((source0_locality_ratio + 1.0) / 2)
    assert stats["locality_distance_ratio_q25"] == pytest.approx(torch.quantile(torch.tensor([source0_locality_ratio, 1.0]), 0.25).item())
    assert stats["high_D_sources"] == 2
    assert stats["high_D_locality_struct_topN_G_mean"] == pytest.approx(source0_locality_g)
    assert stats["high_D_locality_mean_G_gain"] == pytest.approx(source0_locality_g - source0_current_g)
    assert stats["high_D_locality_mean_distance_ratio"] == pytest.approx(source0_locality_ratio)


def test_child_structure_diagnostics_partial_pool_coverage():
    selected = torch.tensor([True, False])
    current_edge_scores = torch.tensor([[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]])
    pool_edge_scores = torch.tensor([[0.9, 0.8, 0.7], [0.0, 0.0, 0.0]])
    nearest = torch.tensor([[1, 2, 3], [0, 1, 2]], dtype=torch.long)
    pool = torch.tensor([[3, 4, 5], [0, 1, 2]], dtype=torch.long)

    stats = compute_child_structure_diagnostics(
        1200,
        selected,
        current_edge_scores,
        pool_edge_scores,
        nearest,
        pool,
    )

    assert stats["current_target_pool_coverage"] == pytest.approx(1 / 3)
    assert stats["full_current_pool_coverage_ratio"] == pytest.approx(0.0)


def test_child_structure_diag_format_has_complete_tag_and_fields():
    selected = torch.tensor([False, False])
    edge = torch.empty((2, 0))
    pool_edge = torch.empty((2, 0))
    indices = torch.empty((2, 0), dtype=torch.long)

    line = format_child_structure_diag(
        compute_child_structure_diagnostics(7, selected, edge, pool_edge, indices, indices)
    )

    assert line.startswith("[ChildStructureDiag] ")
    assert "iter=7" in line
    assert "selected_sources=0" in line
    assert "children=0" in line
    assert "current_G_mean=nan" in line
    assert "mean_distance_ratio=nan" in line
    assert "current_target_pool_coverage=nan" in line
    assert "locality_factor=1.25" in line
    assert "locality_mean_G_gain=nan" in line
    assert "high_D_sources=0" in line


def test_child_structure_diagnostics_high_d_subset():
    selected = torch.tensor([True, True, True, True])
    xyz = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ]
    )
    current_edge_scores = torch.tensor(
        [
            [0.8, 0.8, 0.8],
            [0.1, 0.2, 0.3],
            [0.6, 0.6, 0.6],
            [0.05, 0.1, 0.2],
        ]
    )
    pool_edge_scores = torch.tensor(
        [
            [0.8, 0.8, 0.8, 0.1],
            [0.7, 0.6, 0.5, 0.4],
            [0.9, 0.8, 0.7, 0.1],
            [0.6, 0.5, 0.4, 0.3],
        ]
    )
    nearest = torch.tensor([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]], dtype=torch.long)
    pool = torch.tensor([[1, 2, 3, 0], [0, 2, 3, 1], [0, 1, 3, 2], [0, 1, 2, 3]], dtype=torch.long)
    value_components = {"D": torch.tensor([0.1, 0.8, 0.4, 0.8])}

    stats = compute_child_structure_diagnostics(
        1200,
        selected,
        current_edge_scores,
        pool_edge_scores,
        nearest,
        pool,
        value_components=value_components,
        xyz=xyz,
    )

    high_d_edges = torch.tensor([0.1, 0.2, 0.3, 0.05, 0.1, 0.2])
    high_d_struct = torch.tensor([(0.7 + 0.6 + 0.5) / 3, (0.6 + 0.5 + 0.4) / 3])
    high_d_current = torch.tensor([(0.1 + 0.2 + 0.3) / 3, (0.05 + 0.1 + 0.2) / 3])
    high_d_ratio = torch.tensor([1.0, 1.0])

    assert stats["high_D_sources"] == 2
    assert stats["high_D_current_G_mean"] == pytest.approx(high_d_edges.mean().item())
    assert stats["high_D_low_G025_ratio"] == pytest.approx(5 / 6)
    assert stats["high_D_struct_topN_G_mean"] == pytest.approx(high_d_struct.mean().item())
    assert stats["high_D_mean_G_gain"] == pytest.approx((high_d_struct - high_d_current).mean().item())
    assert stats["high_D_mean_distance_ratio"] == pytest.approx(high_d_ratio.mean().item())

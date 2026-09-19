import json
import random

import numpy as np
import pytest
import torch

from utils.evidence_support import (
    _rankdata_average,
    _spearman_corr,
    compute_evidence_quartile_rgb_error_stats,
    compute_spatial_reliability_diagnostics,
    make_spatial_reliability_record,
    preserved_random_state,
    save_spatial_reliability_summary,
    should_run_spatial_reliability_snapshot,
)


def _depth_diag(residual):
    residual = torch.as_tensor(residual, dtype=torch.float32)
    return {
        "stats": {
            "normalized_valid": True,
            "reference_transform": "negative_midas",
            "reference_correlation": 1.0,
            "normalized_residual_mean": float(residual.mean().item()),
            "normalized_residual_p90": float(torch.quantile(residual.reshape(-1), 0.90).item()),
        },
        "normalized_absolute_residual": residual,
        "evidence": 1.0 / (1.0 + residual),
    }


def _rgb_pair_for_error(error):
    error = torch.as_tensor(error, dtype=torch.float32)
    rendered = torch.zeros((3, *error.shape), dtype=torch.float32)
    gt = error.unsqueeze(0).repeat(3, 1, 1)
    return rendered, gt


def test_rankdata_average_all_unique():
    ranks = _rankdata_average(torch.tensor([10.0, 20.0, 30.0, 40.0]))

    assert torch.allclose(ranks, torch.tensor([0.0, 1.0, 2.0, 3.0]))


def test_rankdata_average_ties():
    ranks = _rankdata_average(torch.tensor([10.0, 20.0, 20.0, 40.0]))

    assert torch.allclose(ranks, torch.tensor([0.0, 1.5, 1.5, 3.0]))


def test_rankdata_average_unsorted_ties_scatter_back_to_original_order():
    ranks = _rankdata_average(torch.tensor([20.0, 10.0, 40.0, 20.0]))

    assert torch.allclose(ranks, torch.tensor([1.5, 0.0, 3.0, 1.5]))


def test_rankdata_average_all_equal():
    ranks = _rankdata_average(torch.tensor([5.0, 5.0, 5.0, 5.0]))

    assert torch.allclose(ranks, torch.full((4,), 1.5))


def test_spearman_perfect_monotonic():
    corr = _spearman_corr(torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([10.0, 20.0, 30.0, 40.0]))

    assert corr == pytest.approx(1.0)


def test_spearman_inverse_monotonic():
    corr = _spearman_corr(torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([40.0, 30.0, 20.0, 10.0]))

    assert corr == pytest.approx(-1.0)


def test_rankdata_average_cuda_if_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")

    ranks = _rankdata_average(torch.tensor([10.0, 20.0, 20.0, 40.0], device="cuda"))

    assert torch.allclose(ranks.cpu(), torch.tensor([0.0, 1.5, 1.5, 3.0]))


def test_perfect_synthetic_high_evidence_has_low_rgb_error_negative_corr():
    residual = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    rendered, gt = _rgb_pair_for_error(residual)

    stats = compute_spatial_reliability_diagnostics(_depth_diag(residual), rendered, gt)

    assert stats["valid_record"] is True
    assert stats["pearson_evidence_vs_rgb_error"] < 0.0
    assert stats["spearman_evidence_vs_rgb_error"] < 0.0
    assert stats["pearson_depth_residual_vs_rgb_error"] > 0.0
    assert stats["q4_vs_q1_rgb_error_ratio"] < 1.0


def test_inverse_synthetic_high_evidence_has_high_rgb_error_positive_corr():
    residual = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    evidence = 1.0 / (1.0 + residual)
    rendered, gt = _rgb_pair_for_error(evidence)

    stats = compute_spatial_reliability_diagnostics(_depth_diag(residual), rendered, gt)

    assert stats["pearson_evidence_vs_rgb_error"] > 0.0
    assert stats["spearman_evidence_vs_rgb_error"] > 0.0
    assert stats["pearson_depth_residual_vs_rgb_error"] < 0.0


def test_quartile_partition_assigns_each_pixel_once():
    evidence = torch.arange(8, dtype=torch.float32)
    rgb_error = torch.arange(8, dtype=torch.float32)

    quartiles = compute_evidence_quartile_rgb_error_stats(evidence, rgb_error)

    assert sum(quartiles[f"Q{i}"]["count"] for i in range(1, 5)) == 8
    assert [quartiles[f"Q{i}"]["count"] for i in range(1, 5)] == [2, 2, 2, 2]


def test_quartile_ties_do_not_crash():
    evidence = torch.ones(8)
    rgb_error = torch.arange(8, dtype=torch.float32)

    quartiles = compute_evidence_quartile_rgb_error_stats(evidence, rgb_error)

    assert sum(quartiles[f"Q{i}"]["count"] for i in range(1, 5)) == 8


def test_nan_inf_are_filtered_before_association():
    residual = torch.tensor([[0.0, 1.0, float("nan")], [2.0, 3.0, float("inf")]])
    rgb_error = torch.tensor([[0.0, 1.0, 5.0], [2.0, 3.0, 6.0]])
    rendered, gt = _rgb_pair_for_error(rgb_error)

    stats = compute_spatial_reliability_diagnostics(_depth_diag(residual), rendered, gt)

    assert stats["valid_record"] is True
    assert stats["valid_pixel_count"] == 4
    assert stats["pearson_depth_residual_vs_rgb_error"] == pytest.approx(1.0)


def test_insufficient_valid_pixels_returns_none_semantics():
    residual = torch.tensor([[0.0, 1.0], [float("nan"), float("nan")]])
    rendered, gt = _rgb_pair_for_error(torch.tensor([[0.0, 1.0], [2.0, 3.0]]))

    stats = compute_spatial_reliability_diagnostics(_depth_diag(residual), rendered, gt)

    assert stats["valid_record"] is False
    assert stats["pearson_evidence_vs_rgb_error"] is None
    assert stats["q4_vs_q1_rgb_error_ratio"] is None


def test_spatial_reliability_summary_json_serializable(tmp_path):
    residual = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    rendered, gt = _rgb_pair_for_error(residual)
    reliability = compute_spatial_reliability_diagnostics(_depth_diag(residual), rendered, gt)
    record = make_spatial_reliability_record(500, 1, _depth_diag(residual)["stats"], reliability)

    path = save_spatial_reliability_summary(str(tmp_path), [record])
    with open(path, "r", encoding="utf-8") as handle:
        saved = json.load(handle)

    assert saved["total_snapshots"] == 1
    assert saved["valid_records"] == 1
    assert saved["per_view"][0]["iteration"] == 500


def _numpy_rng_equal(left, right):
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def test_preserved_random_state_restores_python_numpy_and_torch_rng():
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()

    with preserved_random_state():
        random.random()
        np.random.rand()
        torch.rand(3)

    assert random.getstate() == py_state
    assert _numpy_rng_equal(np.random.get_state(), np_state)
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_spatial_reliability_snapshot_disabled_is_noop():
    assert should_run_spatial_reliability_snapshot(False, 500, (500,)) is False
    assert should_run_spatial_reliability_snapshot(True, 499, (500,)) is False
    assert should_run_spatial_reliability_snapshot(True, 500, (500,)) is True

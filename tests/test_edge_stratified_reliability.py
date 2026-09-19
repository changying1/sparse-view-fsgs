import json

import pytest
import torch

from utils.evidence_support import (
    compute_edge_strata_masks,
    compute_edge_stratified_reliability_diagnostics,
    compute_evidence_quartile_rgb_error_stats,
    compute_gt_edge_magnitude_map,
    make_edge_stratified_reliability_record,
    save_edge_stratified_reliability_summary,
    should_run_edge_stratified_reliability_snapshot,
)


def _depth_diag(residual):
    residual = torch.as_tensor(residual, dtype=torch.float32)
    return {
        "stats": {
            "normalized_valid": True,
            "reference_transform": "negative_midas",
            "reference_correlation": 1.0,
            "normalized_residual_mean": float(torch.nanmean(residual).item()),
            "normalized_residual_p90": 0.0,
        },
        "normalized_absolute_residual": residual,
        "evidence": 1.0 / (1.0 + residual),
    }


def _rgb_pair_for_error(error):
    error = torch.as_tensor(error, dtype=torch.float32)
    rendered = torch.zeros((3, *error.shape), dtype=torch.float32)
    gt = error.unsqueeze(0).repeat(3, 1, 1)
    return rendered, gt


def test_synthetic_edge_map_responds_to_gt_step_edge():
    gt = torch.zeros((3, 4, 4), dtype=torch.float32)
    gt[:, :, 2:] = 1.0

    edge = compute_gt_edge_magnitude_map(gt)

    assert edge[:, 1].mean() > edge[:, 0].mean()
    assert torch.isfinite(edge).all()


def test_edge_and_non_edge_strata_do_not_overlap():
    edge_map = torch.arange(16, dtype=torch.float32).reshape(4, 4)

    edge_mask, non_edge_mask, threshold = compute_edge_strata_masks(edge_map, edge_quantile=0.80)

    assert threshold is not None
    assert not (edge_mask & non_edge_mask).any()
    assert (edge_mask | non_edge_mask).all()


def test_quartile_statistics_work_inside_edge_stratum():
    evidence = torch.tensor([0.1, 0.2, 0.8, 0.9])
    rgb_error = torch.tensor([4.0, 3.0, 2.0, 1.0])
    stratum = torch.tensor([True, False, True, False])

    quartiles = compute_evidence_quartile_rgb_error_stats(evidence[stratum], rgb_error[stratum])

    assert sum(quartiles[f"Q{i}"]["count"] for i in range(1, 5)) == 2


def test_edge_stratified_reliability_filters_nan_inf_safely():
    residual = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 2.0, float("nan"), 4.0],
            [2.0, 3.0, 4.0, float("inf")],
            [3.0, 4.0, 5.0, 6.0],
        ]
    )
    rendered, gt = _rgb_pair_for_error(torch.nan_to_num(residual, nan=0.0, posinf=0.0))
    gt[:, :, 2:] += 0.5

    stats = compute_edge_stratified_reliability_diagnostics(_depth_diag(residual), rendered, gt, min_valid_pixels=2)

    assert stats["edge"]["valid_pixel_count"] >= 0
    assert stats["non_edge"]["valid_pixel_count"] >= 0


def test_edge_stratified_insufficient_pixels_returns_none_not_crash():
    residual = torch.full((2, 2), float("nan"))
    rendered, gt = _rgb_pair_for_error(torch.zeros((2, 2)))

    stats = compute_edge_stratified_reliability_diagnostics(_depth_diag(residual), rendered, gt)

    assert stats["edge"]["valid_record"] is False
    assert stats["edge"]["pearson_evidence_vs_rgb_error"] is None
    assert stats["non_edge"]["valid_record"] is False


def test_edge_stratified_summary_json_serializable(tmp_path):
    residual = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    rendered, gt = _rgb_pair_for_error(residual)
    gt[:, :, 2:] += 0.5
    stats = compute_edge_stratified_reliability_diagnostics(_depth_diag(residual), rendered, gt, min_valid_pixels=2)
    record = make_edge_stratified_reliability_record(2000, 1, stats)

    path = save_edge_stratified_reliability_summary(str(tmp_path), [record])
    with open(path, "r", encoding="utf-8") as handle:
        saved = json.load(handle)

    assert saved["total_records"] == 1
    assert "overall_edge" in saved
    assert saved["per_view"][0]["view_id"] == 1


def test_edge_stratified_diagnostics_disabled_noop():
    assert should_run_edge_stratified_reliability_snapshot(False, 2000, (2000,)) is False
    assert should_run_edge_stratified_reliability_snapshot(True, 1999, (2000,)) is False
    assert should_run_edge_stratified_reliability_snapshot(True, 2000, (2000,)) is True

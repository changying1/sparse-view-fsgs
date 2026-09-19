import json

import pytest
import torch

from utils.evidence_support import (
    compute_real_depth_evidence_diagnostics,
    format_real_depth_evidence_diag,
    save_real_depth_evidence_summary,
    should_run_real_depth_evidence_diagnostics,
    summarize_real_depth_evidence_records,
)


def test_identical_depth_has_zero_normalized_residual_with_negative_transform():
    reference = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    rendered = -reference

    diag = compute_real_depth_evidence_diagnostics(rendered, reference)

    assert diag["stats"]["reference_transform"] == "negative_midas"
    assert diag["stats"]["reference_correlation"] == pytest.approx(1.0)
    assert diag["stats"]["normalized_residual_mean"] == pytest.approx(0.0, abs=1e-6)
    assert diag["normalized_absolute_residual"][diag["valid_mask"]].max().item() == pytest.approx(0.0, abs=1e-6)


def test_offset_invariance_keeps_normalized_residual_zero_and_raw_descriptive():
    reference = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    rendered = -reference + 2.0

    diag = compute_real_depth_evidence_diagnostics(rendered, reference)

    assert diag["stats"]["reference_transform"] == "negative_midas"
    assert diag["stats"]["reference_correlation"] == pytest.approx(1.0)
    assert diag["stats"]["normalized_residual_mean"] == pytest.approx(0.0, abs=1e-6)
    assert diag["stats"]["raw_residual_signed_mean"] == pytest.approx(2.0)
    assert diag["stats"]["raw_residual_mean"] == pytest.approx(2.0)


def test_positive_scale_invariance_keeps_normalized_residual_zero():
    reference = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    rendered = 10.0 * (-reference)

    diag = compute_real_depth_evidence_diagnostics(rendered, reference)

    assert diag["stats"]["reference_transform"] == "negative_midas"
    assert diag["stats"]["reference_correlation"] == pytest.approx(1.0)
    assert diag["stats"]["normalized_residual_mean"] == pytest.approx(0.0, abs=1e-6)
    assert diag["stats"]["raw_residual_mean"] > 0.0


def test_scale_plus_offset_invariance_keeps_normalized_residual_zero():
    reference = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    rendered = 5.0 * (-reference) + 7.0

    diag = compute_real_depth_evidence_diagnostics(rendered, reference)

    assert diag["stats"]["reference_correlation"] == pytest.approx(1.0)
    assert diag["stats"]["normalized_residual_mean"] == pytest.approx(0.0, abs=1e-6)


def test_structural_mismatch_has_positive_normalized_residual_and_lower_corr():
    reference = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    rendered = torch.tensor([[-1.0, -2.0], [-4.0, -3.0]])

    diag = compute_real_depth_evidence_diagnostics(rendered, reference)

    assert diag["stats"]["reference_correlation"] < 1.0
    assert diag["stats"]["normalized_residual_mean"] > 0.0


def test_invalid_pixels_can_be_excluded_by_explicit_secondary_mask():
    reference = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    rendered = -reference
    rendered[0, 1] = 100.0
    mask = torch.tensor([[True, False], [True, False]])

    diag = compute_real_depth_evidence_diagnostics(rendered, reference, valid_mask=mask)

    assert diag["stats"]["candidate_count"] == 2
    assert diag["stats"]["valid_count"] == 2
    assert diag["stats"]["valid_fraction"] == pytest.approx(1.0)
    assert diag["stats"]["normalized_residual_mean"] == pytest.approx(0.0, abs=1e-6)


def test_nan_and_inf_do_not_pollute_statistics():
    reference = torch.tensor([[1.0, float("nan")], [3.0, 4.0]])
    rendered = torch.tensor([[-1.0, -2.0], [float("inf"), -4.0]])

    diag = compute_real_depth_evidence_diagnostics(rendered, reference)

    assert diag["stats"]["candidate_count"] == 4
    assert diag["stats"]["valid_count"] == 2
    assert diag["stats"]["finite_fraction"] == pytest.approx(0.5)
    assert diag["stats"]["normalized_residual_mean"] == pytest.approx(0.0, abs=1e-6)


def test_zero_denominator_raw_relative_residual_is_safe():
    reference = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    rendered = torch.tensor([[1.0, 1.0], [1.0, 1.0]])

    diag = compute_real_depth_evidence_diagnostics(rendered, reference)

    assert diag["stats"]["raw_relative_residual_safe_count"] == 0
    assert diag["stats"]["raw_relative_residual_mean"] is None


def test_all_invalid_mask_returns_clear_empty_statistics():
    reference = torch.ones((2, 2))
    rendered = -reference
    mask = torch.zeros((2, 2), dtype=torch.bool)

    diag = compute_real_depth_evidence_diagnostics(rendered, reference, valid_mask=mask)

    assert diag["stats"]["candidate_count"] == 0
    assert diag["stats"]["valid_count"] == 0
    assert diag["stats"]["valid_fraction"] == pytest.approx(0.0)
    assert diag["stats"]["normalized_residual_mean"] is None
    assert not diag["valid_mask"].any()


def test_optional_bounded_evidence_uses_normalized_residual():
    reference = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    perfect_affine = -reference + 2.0
    mismatch = torch.tensor([[-1.0, -2.0], [-4.0, -3.0]])

    perfect_diag = compute_real_depth_evidence_diagnostics(perfect_affine, reference, return_evidence=True)
    mismatch_diag = compute_real_depth_evidence_diagnostics(mismatch, reference, return_evidence=True)
    perfect_evidence = perfect_diag["evidence"][perfect_diag["valid_mask"]]
    mismatch_evidence = mismatch_diag["evidence"][mismatch_diag["valid_mask"]]

    assert torch.isfinite(perfect_evidence).all()
    assert (perfect_evidence >= 0.0).all()
    assert (perfect_evidence <= 1.0).all()
    assert perfect_evidence.mean().item() == pytest.approx(1.0, abs=1e-6)
    assert mismatch_evidence.mean().item() < perfect_evidence.mean().item()


def test_zero_variance_depth_marks_normalized_residual_unavailable():
    reference = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    rendered = torch.ones((2, 2))

    diag = compute_real_depth_evidence_diagnostics(rendered, reference, return_evidence=True)

    assert diag["stats"]["normalized_valid"] is False
    assert diag["stats"]["normalized_residual_mean"] is None
    assert torch.isnan(diag["normalized_absolute_residual"]).all()
    assert torch.isnan(diag["evidence"]).all()


def test_diagnostics_disabled_helper_is_noop():
    assert should_run_real_depth_evidence_diagnostics(False, 100, 100) is False
    assert should_run_real_depth_evidence_diagnostics(True, 101, 100) is False
    assert should_run_real_depth_evidence_diagnostics(True, 100, 100) is True


def test_format_and_summary_json_are_compact(tmp_path):
    records = [
        {
            "iteration": 100,
            "view_id": 3,
            "valid_fraction": 1.0,
            "normalized_residual_mean": 0.1,
            "normalized_residual_median": 0.1,
            "normalized_residual_p90": 0.2,
            "normalized_residual_p95": 0.3,
            "raw_residual_mean": 2.0,
            "raw_residual_median": 2.0,
            "finite_fraction": 1.0,
        }
    ]

    line = format_real_depth_evidence_diag(100, 3, records[0])
    assert line.startswith("[OEDepthDiag]")
    assert "normalized_residual_mean=0.1" in line
    summary = summarize_real_depth_evidence_records(records)
    assert summary["total_samples"] == 1
    path = save_real_depth_evidence_summary(str(tmp_path), records)
    with open(path, "r", encoding="utf-8") as handle:
        saved = json.load(handle)
    assert saved["total_samples"] == 1

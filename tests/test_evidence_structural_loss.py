import json

import pytest
import torch

import utils.evidence_structural_loss as module
from utils.evidence_structural_loss import (
    compute_oe_structural_loss,
    compute_stable_mask_from_gt_rgb,
    format_oe_structural_loss_log,
    make_oe_structural_record,
    save_oe_structural_training_summary,
    should_apply_oe_structural_loss,
)


def test_perfect_structure_has_zero_loss():
    reference = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    rendered = -reference

    loss, stats, _ = compute_oe_structural_loss(rendered, reference, gate_mode="none")

    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    assert stats["raw_struct_loss"] == pytest.approx(0.0, abs=1e-6)


def test_affine_invariance_after_centered_rms_normalization():
    reference = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    rendered = 3.0 * (-reference) + 7.0

    loss, _, _ = compute_oe_structural_loss(rendered, reference, gate_mode="none")

    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_structural_mismatch_is_positive():
    reference = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    rendered = torch.tensor([[-1.0, -2.0], [-4.0, -3.0]])

    loss, stats, _ = compute_oe_structural_loss(rendered, reference, gate_mode="none")

    assert loss.item() > 0.0
    assert stats["raw_struct_loss"] > 0.0


def test_zero_gate_is_safe_zero():
    reference = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    rendered = torch.tensor([[-1.0, -2.0], [-4.0, -3.0]])
    stable = torch.zeros((2, 2), dtype=torch.bool)

    loss, stats, _ = compute_oe_structural_loss(rendered, reference, stable_mask=stable, gate_mode="stable_only")

    assert loss.item() == pytest.approx(0.0)
    assert torch.isfinite(loss)
    assert stats["active_pair_fraction"] == pytest.approx(0.0)


def test_unit_gate_equals_raw_structural_loss():
    reference = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    rendered = torch.tensor([[-1.0, -2.0], [-4.0, -3.0]])

    loss, stats, _ = compute_oe_structural_loss(rendered, reference, gate_mode="none")

    assert loss.item() == pytest.approx(stats["raw_struct_loss"], abs=1e-6)


def test_evidence_pair_uses_minimum_endpoint():
    rendered = torch.tensor([[0.0, 1.0]])
    reference = torch.tensor([[0.0, 0.0]])
    evidence = torch.tensor([[0.2, 0.8]])
    stable = torch.ones((1, 2), dtype=torch.bool)
    valid = torch.ones((1, 2), dtype=torch.bool)

    terms = module._pair_terms(rendered, reference, evidence, stable, valid, dim=1)

    assert terms["evidence"][0].item() == pytest.approx(0.2)


def test_pair_evidence_distribution_diagnostics_are_correct(monkeypatch):
    rendered = torch.tensor([[0.0, 1.0, 3.0, 6.0, 10.0]], requires_grad=True)
    evidence = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]])

    def fake_maps(*args, **kwargs):
        return {
            "normalized_valid": True,
            "z_render": rendered,
            "z_reference": torch.zeros_like(rendered),
            "evidence": evidence,
            "valid_mask": torch.ones_like(rendered, dtype=torch.bool),
            "reference_transform": "fake",
        }

    monkeypatch.setattr(module, "compute_real_depth_evidence_maps", fake_maps)

    _, stats, _ = compute_oe_structural_loss(rendered, torch.zeros_like(rendered), gate_mode="none")

    assert stats["pair_evidence_std"] == pytest.approx(torch.tensor([0.0, 1.0, 2.0, 3.0]).std(unbiased=False).item())
    assert stats["pair_evidence_p10"] == pytest.approx(0.3)
    assert stats["pair_evidence_p50"] == pytest.approx(1.5)
    assert stats["pair_evidence_p90"] == pytest.approx(2.7)
    assert stats["pair_evidence_spread"] == pytest.approx(2.4)


def test_stable_modes_share_evidence_diagnostic_population(monkeypatch):
    rendered = torch.tensor([[0.0, 1.0, 3.0, 6.0, 10.0]], requires_grad=True)
    evidence = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]])
    stable = torch.tensor([[True, True, True, False, True]])

    def fake_maps(*args, **kwargs):
        return {
            "normalized_valid": True,
            "z_render": rendered,
            "z_reference": torch.zeros_like(rendered),
            "evidence": evidence,
            "valid_mask": torch.ones_like(rendered, dtype=torch.bool),
            "reference_transform": "fake",
        }

    monkeypatch.setattr(module, "compute_real_depth_evidence_maps", fake_maps)

    _, stable_only_stats, _ = compute_oe_structural_loss(
        rendered,
        torch.zeros_like(rendered),
        stable_mask=stable,
        gate_mode="stable_only",
    )
    _, stable_evidence_stats, _ = compute_oe_structural_loss(
        rendered,
        torch.zeros_like(rendered),
        stable_mask=stable,
        gate_mode="stable_evidence",
    )
    _, none_stats, _ = compute_oe_structural_loss(
        rendered,
        torch.zeros_like(rendered),
        stable_mask=stable,
        gate_mode="none",
    )

    fields = (
        "pair_evidence_std",
        "pair_evidence_p10",
        "pair_evidence_p50",
        "pair_evidence_p90",
        "pair_evidence_spread",
    )
    for field in fields:
        assert stable_only_stats[field] == pytest.approx(stable_evidence_stats[field])

    assert stable_only_stats["pair_evidence_p90"] == pytest.approx(0.9)
    assert none_stats["pair_evidence_p90"] == pytest.approx(2.7)


def test_stable_pair_requires_both_endpoints():
    reference = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    rendered = torch.tensor([[-1.0, -2.0], [-4.0, -3.0]])
    stable = torch.tensor([[True, False], [False, False]])

    loss, stats, _ = compute_oe_structural_loss(rendered, reference, stable_mask=stable, gate_mode="stable_evidence")

    assert loss.item() == pytest.approx(0.0)
    assert stats["active_pair_count"] == 0


def test_gate_modes_match_expected_weighted_results(monkeypatch):
    rendered = torch.tensor([[0.0, 1.0, 4.0]], requires_grad=True)
    evidence = torch.tensor([[0.2, 0.8, 0.5]])
    stable = torch.tensor([[True, False, True]])

    def fake_maps(*args, **kwargs):
        return {
            "normalized_valid": True,
            "z_render": rendered,
            "z_reference": torch.zeros_like(rendered),
            "evidence": evidence,
            "valid_mask": torch.ones_like(rendered, dtype=torch.bool),
            "reference_transform": "fake",
        }

    monkeypatch.setattr(module, "compute_real_depth_evidence_maps", fake_maps)

    expected = {
        "none": (1.0 + 3.0) / 2.0,
        "stable_only": 0.0,
        "evidence_only": (0.2 * 1.0 + 0.5 * 3.0) / (0.2 + 0.5),
        "stable_evidence": 0.0,
        "matched_stable": 0.0,
    }
    for mode, value in expected.items():
        loss, _, _ = compute_oe_structural_loss(rendered, torch.zeros_like(rendered), stable_mask=stable, gate_mode=mode)
        assert loss.item() == pytest.approx(value, abs=1e-6)


def test_matched_stable_matches_stable_evidence_forward_and_uses_stable_gradients(monkeypatch):
    rendered = torch.tensor([[0.0, 1.0, 3.0]], requires_grad=True)
    evidence_source = torch.tensor([[0.2, 0.8, 0.5]], requires_grad=True)
    stable = torch.ones_like(rendered, dtype=torch.bool)

    def fake_maps(*args, **kwargs):
        return {
            "normalized_valid": True,
            "z_render": rendered,
            "z_reference": torch.zeros_like(rendered),
            "evidence": evidence_source * 1.0,
            "valid_mask": torch.ones_like(rendered, dtype=torch.bool),
            "reference_transform": "fake",
        }

    monkeypatch.setattr(module, "compute_real_depth_evidence_maps", fake_maps)

    stable_only_loss, _, _ = compute_oe_structural_loss(
        rendered,
        torch.zeros_like(rendered),
        stable_mask=stable,
        gate_mode="stable_only",
    )
    stable_only_grad = torch.autograd.grad(stable_only_loss, rendered, retain_graph=True)[0]

    stable_evidence_loss, _, _ = compute_oe_structural_loss(
        rendered,
        torch.zeros_like(rendered),
        stable_mask=stable,
        gate_mode="stable_evidence",
    )
    stable_evidence_grad = torch.autograd.grad(stable_evidence_loss, rendered, retain_graph=True)[0]

    matched_loss, stats, _ = compute_oe_structural_loss(
        rendered,
        torch.zeros_like(rendered),
        stable_mask=stable,
        gate_mode="matched_stable",
    )
    matched_loss.backward()

    assert matched_loss.item() == pytest.approx(stable_evidence_loss.item(), abs=1e-6)
    assert stats["matched_gated_loss"] == pytest.approx(stats["stable_evidence_gated_loss"], abs=1e-6)
    assert stats["matched_strength_alpha"] == pytest.approx(
        stats["stable_evidence_gated_loss"] / stats["stable_only_gated_loss"],
        abs=1e-6,
    )
    assert torch.allclose(rendered.grad, stable_only_grad * stats["matched_strength_alpha"], atol=1e-6)
    assert not torch.allclose(rendered.grad, stable_evidence_grad, atol=1e-6)
    assert evidence_source.grad is None


def test_matched_stable_empty_and_zero_stable_loss_are_safe(monkeypatch):
    rendered = torch.tensor([[1.0]], requires_grad=True)

    def fake_empty_maps(*args, **kwargs):
        return {
            "normalized_valid": True,
            "z_render": rendered,
            "z_reference": torch.zeros_like(rendered),
            "evidence": torch.ones_like(rendered),
            "valid_mask": torch.ones_like(rendered, dtype=torch.bool),
            "reference_transform": "fake",
        }

    monkeypatch.setattr(module, "compute_real_depth_evidence_maps", fake_empty_maps)

    loss, stats, _ = compute_oe_structural_loss(rendered, torch.zeros_like(rendered), gate_mode="matched_stable")
    assert loss.item() == pytest.approx(0.0)
    assert torch.isfinite(loss)
    assert stats["matched_strength_alpha"] is None
    assert stats["matched_gated_loss"] == pytest.approx(0.0)

    rendered = torch.tensor([[0.0, 0.0]], requires_grad=True)
    evidence = torch.tensor([[0.2, 0.8]])

    def fake_zero_maps(*args, **kwargs):
        return {
            "normalized_valid": True,
            "z_render": rendered,
            "z_reference": torch.zeros_like(rendered),
            "evidence": evidence,
            "valid_mask": torch.ones_like(rendered, dtype=torch.bool),
            "reference_transform": "fake",
        }

    monkeypatch.setattr(module, "compute_real_depth_evidence_maps", fake_zero_maps)

    loss, stats, _ = compute_oe_structural_loss(
        rendered,
        torch.zeros_like(rendered),
        stable_mask=torch.ones_like(rendered, dtype=torch.bool),
        gate_mode="matched_stable",
    )
    loss.backward()

    assert loss.item() == pytest.approx(0.0)
    assert torch.isfinite(loss)
    assert stats["matched_strength_alpha"] == pytest.approx(0.0)
    assert stats["matched_gated_loss"] == pytest.approx(0.0)
    assert rendered.grad is not None
    assert torch.isfinite(rendered.grad).all()


def test_gradient_flows_to_rendered_depth_only():
    reference = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    rendered = torch.tensor([[-1.0, -2.0], [-4.0, -3.0]], requires_grad=True)

    loss, _, _ = compute_oe_structural_loss(rendered, reference, gate_mode="none")
    loss.backward()

    assert rendered.grad is not None
    assert torch.isfinite(rendered.grad).all()
    assert rendered.grad.abs().sum().item() > 0.0
    assert reference.grad is None


def test_evidence_gate_is_detached(monkeypatch):
    rendered = torch.tensor([[0.0, 1.0]], requires_grad=True)
    evidence_source = torch.tensor([[0.2, 0.8]], requires_grad=True)

    def fake_maps(*args, **kwargs):
        return {
            "normalized_valid": True,
            "z_render": rendered,
            "z_reference": torch.zeros_like(rendered),
            "evidence": evidence_source * 1.0,
            "valid_mask": torch.ones_like(rendered, dtype=torch.bool),
            "reference_transform": "fake",
        }

    monkeypatch.setattr(module, "compute_real_depth_evidence_maps", fake_maps)

    loss, _, _ = compute_oe_structural_loss(
        rendered,
        torch.zeros_like(rendered),
        stable_mask=torch.ones((1, 2), dtype=torch.bool),
        gate_mode="evidence_only",
    )
    loss.backward()

    assert rendered.grad is not None
    assert evidence_source.grad is None


def test_evidence_diagnostics_are_detached_and_backward_is_unchanged(monkeypatch):
    rendered = torch.tensor([[0.0, 1.0, 3.0]], requires_grad=True)
    evidence_source = torch.tensor([[0.2, 0.8, 0.5]], requires_grad=True)

    def fake_maps(*args, **kwargs):
        return {
            "normalized_valid": True,
            "z_render": rendered,
            "z_reference": torch.zeros_like(rendered),
            "evidence": evidence_source * 1.0,
            "valid_mask": torch.ones_like(rendered, dtype=torch.bool),
            "reference_transform": "fake",
        }

    monkeypatch.setattr(module, "compute_real_depth_evidence_maps", fake_maps)

    loss, stats, _ = compute_oe_structural_loss(rendered, torch.zeros_like(rendered), gate_mode="none")
    loss.backward()

    assert stats["pair_evidence_p50"] == pytest.approx(0.35)
    assert rendered.grad is not None
    assert torch.isfinite(rendered.grad).all()
    assert evidence_source.grad is None


def test_zero_variance_and_all_invalid_are_safe_zero():
    reference = torch.ones((2, 2))
    rendered = torch.ones((2, 2), requires_grad=True)

    loss, _, _ = compute_oe_structural_loss(rendered, reference, gate_mode="none")
    assert loss.item() == pytest.approx(0.0)
    assert torch.isfinite(loss)

    invalid = torch.zeros((2, 2), dtype=torch.bool)
    loss, _, _ = compute_oe_structural_loss(rendered, reference, valid_mask=invalid, gate_mode="none")
    assert loss.item() == pytest.approx(0.0)
    assert torch.isfinite(loss)


def test_empty_pair_evidence_diagnostics_are_safe(monkeypatch):
    rendered = torch.tensor([[1.0]], requires_grad=True)

    def fake_maps(*args, **kwargs):
        return {
            "normalized_valid": True,
            "z_render": rendered,
            "z_reference": torch.zeros_like(rendered),
            "evidence": torch.ones_like(rendered),
            "valid_mask": torch.ones_like(rendered, dtype=torch.bool),
            "reference_transform": "fake",
        }

    monkeypatch.setattr(module, "compute_real_depth_evidence_maps", fake_maps)

    loss, stats, _ = compute_oe_structural_loss(rendered, torch.zeros_like(rendered), gate_mode="none")

    assert loss.item() == pytest.approx(0.0)
    assert torch.isfinite(loss)
    assert stats["pair_evidence_std"] is None
    assert stats["pair_evidence_p10"] is None
    assert stats["pair_evidence_p50"] is None
    assert stats["pair_evidence_p90"] is None
    assert stats["pair_evidence_spread"] is None


def test_stable_mask_uses_gray_gradient_quantile():
    gt = torch.zeros((3, 3, 3))
    gt[:, :, 2] = 1.0

    stable, threshold, magnitude = compute_stable_mask_from_gt_rgb(gt, stable_quantile=0.80)

    assert threshold is not None
    assert stable.shape == (3, 3)
    assert not stable[0, 1]
    assert torch.isfinite(magnitude).all()


def test_window_helper_and_disabled_baseline_path():
    assert should_apply_oe_structural_loss(False, 500, 500, 2000, 1.0) is False
    assert should_apply_oe_structural_loss(True, 499, 500, 2000, 1.0) is False
    assert should_apply_oe_structural_loss(True, 500, 500, 2000, 1.0) is True
    assert should_apply_oe_structural_loss(True, 2000, 500, 2000, 1.0) is True
    assert should_apply_oe_structural_loss(True, 2001, 500, 2000, 1.0) is False
    assert should_apply_oe_structural_loss(True, 500, 500, 2000, 0.0) is False


def test_summary_record_and_log_are_json_serializable(tmp_path):
    stats = {
        "gate_mode": "stable_evidence",
        "raw_struct_loss": 0.2,
        "gated_struct_loss": 0.1,
        "matched_strength_alpha": 0.5,
        "stable_only_gated_loss": 0.2,
        "stable_evidence_gated_loss": 0.1,
        "matched_gated_loss": 0.1,
        "active_pair_fraction": 0.5,
        "mean_pair_evidence": 0.7,
        "pair_evidence_std": 0.1,
        "pair_evidence_p10": 0.2,
        "pair_evidence_p50": 0.7,
        "pair_evidence_p90": 0.9,
        "pair_evidence_spread": 0.7,
        "stable_pixel_fraction": 0.8,
        "valid_pair_count": 4,
        "active_pair_count": 2,
        "reference_transform": "negative_midas",
    }
    line = format_oe_structural_loss_log(100, 2, stats, 0.3)
    record = make_oe_structural_record(100, 2, stats, 0.3)
    path = save_oe_structural_training_summary(str(tmp_path), [record])

    assert line.startswith("[OEStructLoss]")
    assert "gate_mode=stable_evidence" in line
    assert "matched_strength_alpha=0.5" in line
    assert "pair_evidence_std=0.1" in line
    assert record["pair_evidence_spread"] == pytest.approx(0.7)
    assert record["matched_gated_loss"] == pytest.approx(0.1)
    with open(path, "r", encoding="utf-8") as handle:
        saved = json.load(handle)
    assert saved["total_records"] == 1
    assert saved["per_iteration"][0]["pair_evidence_p90"] == pytest.approx(0.9)
    assert saved["per_iteration"][0]["stable_only_gated_loss"] == pytest.approx(0.2)

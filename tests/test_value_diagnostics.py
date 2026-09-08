import importlib.util
import math
import sys
import types
from argparse import Namespace

import pytest
import torch

from utils.growth_budget import select_proximity_sources
from utils.value_allocation import compute_structural_value_score
from utils.value_diagnostics import (
    compute_structural_value_attribution_stats,
    format_structural_boundary_diag,
    format_structural_norm_diag,
    format_structural_promotion_attr,
)


def _budget_stats_with_pairs(promoted=(2,), displaced=(1,), boundary=(1, 2, 3)):
    _, stats = select_proximity_sources(
        torch.ones(4, dtype=torch.bool),
        torch.tensor([200.0, 100.0, 95.0, 1.0]),
        n=1,
        rho=0.5,
        enabled=True,
        mode="value_demand_rerank",
        value_score=torch.tensor([0.0, 0.10, 0.90, 0.0]),
        rerank_fraction=0.5,
        boundary_multiplier=2.0,
        demand_ratio=0.90,
    )
    stats.boundary_indices = tuple(boundary)
    stats.boundary_src = len(boundary)
    stats.demand_promoted_indices = tuple(promoted)
    stats.demand_displaced_indices = tuple(displaced)
    stats.promotion_accepted = len(promoted)
    return stats


def _components():
    b = torch.tensor([0.0, 0.3, 0.8, 0.1])
    k = torch.tensor([0.0, 0.5, 0.4, 0.2])
    d = torch.tensor([0.0, 0.5, 0.9, 0.2])
    r = torch.tensor([0.0, 0.6, 0.2, 0.7])
    return {
        "B": b,
        "K": k,
        "D": d,
        "R": r,
        "B_norm": b,
        "K_norm": k,
        "D_norm": d,
        "R_norm": r,
        "B_raw": torch.tensor([1000.0, 1.0, 2.0, 3.0]),
        "K_raw": torch.tensor([1000.0, 1.0, 2.0, 3.0]),
        "D_raw": torch.tensor([1000.0, 1.0, 2.0, 3.0]),
        "R_raw": torch.tensor([1000.0, 1.0, 2.0, 3.0]),
        "S": b + k + d,
        "U": (b + k + d) / (1.0 + r),
    }


def test_structural_norm_diag_uses_candidate_mask_only():
    components = _components()
    candidate_mask = torch.tensor([False, True, True, True])

    stats = compute_structural_value_attribution_stats(10, components, candidate_mask, _budget_stats_with_pairs())

    assert stats.norm["B_raw_q05"] == pytest.approx(1.1)
    assert stats.norm["B_raw_q95"] == pytest.approx(2.9)
    assert stats.norm["B_raw_span"] == pytest.approx(1.8)


def test_structural_norm_diag_uses_finite_raw_values_only():
    components = _components()
    components["B_raw"] = torch.tensor([float("nan"), float("inf"), -float("inf"), 5.0])
    candidate_mask = torch.ones(4, dtype=torch.bool)

    stats = compute_structural_value_attribution_stats(10, components, candidate_mask, _budget_stats_with_pairs())

    assert stats.norm["B_raw_q05"] == pytest.approx(5.0)
    assert stats.norm["B_raw_q95"] == pytest.approx(5.0)
    assert stats.norm["B_raw_finite_ratio"] == pytest.approx(0.25)


def test_structural_norm_diag_marks_degenerate_span():
    components = _components()
    components["B_raw"] = torch.tensor([5.0, 5.0, 5.0, 5.0])

    stats = compute_structural_value_attribution_stats(10, components, torch.ones(4, dtype=torch.bool), _budget_stats_with_pairs())

    assert stats.norm["B_raw_span"] == pytest.approx(0.0)
    assert stats.norm["B_span_degenerate"] is True


def test_structural_norm_diag_reports_zero_and_one_saturation_ratios():
    components = _components()
    components["B"] = torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0])
    components["B_norm"] = components["B"]
    for key in ("K", "D", "R", "U"):
        components[key] = torch.zeros(5)
    for key in ("K_norm", "D_norm", "R_norm"):
        components[key] = torch.zeros(5)
    for key in ("B_raw", "K_raw", "D_raw", "R_raw", "S"):
        components[key] = torch.arange(5, dtype=torch.float32)
    stats = compute_structural_value_attribution_stats(
        10,
        components,
        torch.ones(5, dtype=torch.bool),
        _budget_stats_with_pairs(promoted=(), displaced=(), boundary=()),
    )

    assert stats.norm["B_norm_zero_ratio"] == pytest.approx(0.4)
    assert stats.norm["B_norm_one_ratio"] == pytest.approx(0.4)


def test_structural_promotion_attribution_single_pair_deltas_and_wins():
    components = _components()
    candidate_mask = torch.ones(4, dtype=torch.bool)

    stats = compute_structural_value_attribution_stats(10, components, candidate_mask, _budget_stats_with_pairs())

    assert stats.promotion["mean_delta_B"] == pytest.approx(0.5)
    assert stats.promotion["mean_delta_K"] == pytest.approx(-0.1)
    assert stats.promotion["mean_delta_D"] == pytest.approx(0.4)
    assert stats.promotion["mean_delta_R"] == pytest.approx(-0.4)
    assert stats.promotion["B_win_ratio"] == pytest.approx(1.0)
    assert stats.promotion["K_win_ratio"] == pytest.approx(0.0)
    assert stats.promotion["D_win_ratio"] == pytest.approx(1.0)
    assert stats.promotion["R_win_ratio"] == pytest.approx(1.0)


def test_structural_promotion_attribution_multiple_pair_means():
    components = _components()
    components["B"] = torch.tensor([0.0, 0.1, 0.8, 0.4])
    components["K"] = torch.tensor([0.0, 0.6, 0.4, 0.7])
    components["D"] = torch.tensor([0.0, 0.5, 0.9, 0.8])
    components["R"] = torch.tensor([0.0, 0.9, 0.2, 0.4])
    components["U"] = (components["B"] + components["K"] + components["D"]) / (1.0 + components["R"])
    stats = compute_structural_value_attribution_stats(
        10,
        components,
        torch.ones(4, dtype=torch.bool),
        _budget_stats_with_pairs(promoted=(2, 3), displaced=(1, 2), boundary=(1, 2, 3)),
    )

    assert stats.promotion["mean_delta_B"] == pytest.approx(0.15)
    assert stats.promotion["B_win_ratio"] == pytest.approx(0.5)
    assert stats.promotion["R_win_ratio"] == pytest.approx(0.5)


def test_structural_promotion_attribution_no_promotion_uses_nan():
    stats = compute_structural_value_attribution_stats(
        10,
        _components(),
        torch.ones(4, dtype=torch.bool),
        _budget_stats_with_pairs(promoted=(), displaced=(), boundary=(1, 2)),
    )
    log_line = format_structural_promotion_attr(stats)

    assert math.isnan(stats.promotion["mean_delta_B"])
    assert math.isnan(stats.promotion["B_win_ratio"])
    assert "mean_delta_B=nan" in log_line


def test_structural_boundary_diag_uses_boundary_indices_only():
    stats = compute_structural_value_attribution_stats(
        10,
        _components(),
        torch.ones(4, dtype=torch.bool),
        _budget_stats_with_pairs(boundary=(1, 2)),
    )

    assert stats.boundary["B_mean"] == pytest.approx(0.55)
    assert stats.boundary["B_std"] == pytest.approx(0.25)


def test_structural_promotion_attr_r_win_direction_is_lower_redundancy():
    stats = compute_structural_value_attribution_stats(10, _components(), torch.ones(4, dtype=torch.bool), _budget_stats_with_pairs())

    assert stats.promotion["mean_R_promoted"] < stats.promotion["mean_R_displaced"]
    assert stats.promotion["R_win_ratio"] == pytest.approx(1.0)


def test_structural_promotion_attr_s_definition_excludes_redundancy():
    stats = compute_structural_value_attribution_stats(10, _components(), torch.ones(4, dtype=torch.bool), _budget_stats_with_pairs())

    assert stats.promotion["mean_S_promoted"] == pytest.approx(0.8 + 0.4 + 0.9)
    assert stats.promotion["mean_S_displaced"] == pytest.approx(0.3 + 0.5 + 0.5)


def test_structural_score_raw_exposure_does_not_change_u():
    boundary = torch.tensor([1.0, 2.0, 5.0])
    turning = torch.tensor([2.0, 4.0, 6.0])
    defect = torch.tensor([3.0, 4.0, 9.0])
    redundancy = torch.tensor([0.0, 1.0, 3.0])

    utility = compute_structural_value_score(boundary, turning, defect, redundancy, low_quantile=0.0, high_quantile=1.0)
    components = compute_structural_value_score(
        boundary,
        turning,
        defect,
        redundancy,
        low_quantile=0.0,
        high_quantile=1.0,
        return_components=True,
    )

    assert torch.equal(components["U"], utility)
    assert torch.equal(components["B_raw"], boundary)
    assert torch.equal(components["K_raw"], turning)
    assert torch.equal(components["D_raw"], defect)
    assert torch.equal(components["R_raw"], redundancy)


def test_structural_attribution_does_not_change_selected_mask():
    candidate_mask = torch.ones(4, dtype=torch.bool)
    dist = torch.tensor([200.0, 100.0, 95.0, 1.0])
    value = torch.tensor([0.0, 0.10, 0.90, 0.0])

    before_mask, before_stats = select_proximity_sources(
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
    compute_structural_value_attribution_stats(10, _components(), candidate_mask, before_stats)
    after_mask, after_stats = select_proximity_sources(
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

    assert torch.equal(before_mask, after_mask)
    assert before_stats.selected_indices == after_stats.selected_indices
    assert before_stats.demand_promoted_indices == after_stats.demand_promoted_indices
    assert before_stats.promotion_accepted == after_stats.promotion_accepted


def test_structural_log_fields_are_complete():
    stats = compute_structural_value_attribution_stats(10, _components(), torch.ones(4, dtype=torch.bool), _budget_stats_with_pairs())
    norm_log = format_structural_norm_diag(stats)
    boundary_log = format_structural_boundary_diag(stats)
    promotion_log = format_structural_promotion_attr(stats)

    for field in (
        "[StructuralNormDiag]",
        "B_raw_q05=",
        "B_raw_q95=",
        "B_raw_span=",
        "B_raw_finite_ratio=",
        "B_norm_zero_ratio=",
        "B_norm_one_ratio=",
        "B_span_degenerate=",
        "R_span_degenerate=",
    ):
        assert field in norm_log
    for field in ("[StructuralBoundaryDiag]", "boundary_src=", "B_mean=", "B_std=", "U_mean=", "U_std="):
        assert field in boundary_log
    for field in (
        "[StructuralPromotionAttr]",
        "promotions=",
        "mean_B_promoted=",
        "mean_delta_R=",
        "R_win_ratio=",
        "mean_P_ratio=",
        "min_P_ratio=",
        "a1_overlap=",
    ):
        assert field in promotion_log


def _load_gaussian_model_module(monkeypatch):
    simple_knn = types.ModuleType("simple_knn")
    simple_knn_c = types.ModuleType("simple_knn._C")
    simple_knn_c.distCUDA2 = lambda xyz: (torch.zeros(xyz.shape[0]), torch.zeros((xyz.shape[0], 3), dtype=torch.long))
    monkeypatch.setitem(sys.modules, "simple_knn", simple_knn)
    monkeypatch.setitem(sys.modules, "simple_knn._C", simple_knn_c)

    plyfile = types.ModuleType("plyfile")
    plyfile.PlyData = object
    plyfile.PlyElement = object
    monkeypatch.setitem(sys.modules, "plyfile", plyfile)

    spec = importlib.util.spec_from_file_location("_fsgs_gaussian_model_m56_test", "scene/gaussian_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model_for_log_gate(module, mode, variant):
    model = module.GaussianModel.__new__(module.GaussianModel)
    module.GaussianModel.setup_functions(model)
    model.args = Namespace(
        enable_proximity_budget=True,
        enable_proximity_candidate_capacity=False,
        proximity_candidate_keep_ratio=0.8,
        proximity_action_replay="",
        proximity_growth_ratio=0.5,
        proximity_selection_mode=mode,
        value_score_variant=variant,
        value_rerank_fraction=0.5,
        value_boundary_multiplier=2.0,
        value_demand_ratio=0.90,
        value_tau_e=1.0,
        value_tau_s=3.0,
        value_w_b=1.0,
        value_w_k=1.0,
        value_w_d=1.0,
        value_lambda_r=1.0,
        normalization_low_quantile=0.0,
        normalization_high_quantile=1.0,
        knn_k=7,
    )
    model._xyz = torch.arange(12, dtype=torch.float32).view(4, 3)
    model._scaling = torch.full((4, 3), torch.log(torch.tensor(2.0)).item(), dtype=torch.float32)
    model._rotation = torch.zeros((4, 4), dtype=torch.float32)
    model._features_dc = torch.zeros((4, 1, 3), dtype=torch.float32)
    model._features_rest = torch.zeros((4, 0, 3), dtype=torch.float32)
    model._opacity = torch.zeros((4, 1), dtype=torch.float32)
    model.confidence = torch.ones((4, 1), dtype=torch.float32)
    model.denom = torch.ones((4, 1), dtype=torch.float32)
    model.visibility_history = None
    model.visible_view_count = None
    model.camera_uid_to_train_index = None
    return model


def test_structural_logs_only_for_structural_value_demand_rerank(monkeypatch, capsys):
    module = _load_gaussian_model_module(monkeypatch)
    dist = torch.tensor([200.0, 100.0, 95.0, 1.0])
    nearest = torch.tensor([[1], [0], [1], [2]], dtype=torch.long)
    monkeypatch.setattr(module, "distCUDA2", lambda xyz: (dist, nearest))

    for mode, variant, should_log in (
        ("value_demand_rerank", "structural", True),
        ("value_demand_rerank", "obdkr", False),
        ("value_rerank", "structural", False),
        ("proximity_topk", "structural", False),
        ("original", "structural", False),
    ):
        model = _model_for_log_gate(module, mode, variant)
        monkeypatch.setattr(model, "densification_postfix", lambda *args: None)
        monkeypatch.setattr(model, "_append_visibility_from_masks", lambda *args, **kwargs: None)
        monkeypatch.setattr(model, "compute_fsgs_proximity_candidate_value", lambda **kwargs: _components())

        model.proximity(scene_extent=1.0, iteration=1200, N=1)
        output = capsys.readouterr().out

        assert ("[StructuralNormDiag]" in output) is should_log
        assert ("[StructuralBoundaryDiag]" in output) is should_log
        assert ("[StructuralPromotionAttr]" in output) is should_log

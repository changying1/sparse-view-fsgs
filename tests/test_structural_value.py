import importlib.util
import sys
import types
from argparse import Namespace

import pytest
import torch

from utils.growth_budget import select_proximity_sources
from utils.value_allocation import (
    compute_balanced_gestalt_value_score,
    compute_defect_conditioned_gestalt_value_score,
    compute_gestalt_structural_value_score,
    compute_obdkr_value,
    compute_structural_value_score,
    robust_normalize,
)
from utils.value_diagnostics import compute_obdkr_diagnostics, format_obdkr_diagnostics_log


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

    spec = importlib.util.spec_from_file_location("_fsgs_gaussian_model_structural_test", "scene/gaussian_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_structural_score_math_matches_weighted_bkd_over_redundancy():
    boundary = torch.tensor([1.0, 2.0, 5.0])
    turning = torch.tensor([2.0, 4.0, 6.0])
    defect = torch.tensor([3.0, 4.0, 9.0])
    redundancy = torch.tensor([0.0, 1.0, 3.0])

    components = compute_structural_value_score(
        boundary,
        turning,
        defect,
        redundancy,
        w_b=2.0,
        w_k=3.0,
        w_d=5.0,
        lambda_r=0.5,
        low_quantile=0.0,
        high_quantile=1.0,
        return_components=True,
    )

    expected = (
        2.0 * robust_normalize(boundary, 0.0, 1.0)
        + 3.0 * robust_normalize(turning, 0.0, 1.0)
        + 5.0 * robust_normalize(defect, 0.0, 1.0)
    ) / (1.0 + 0.5 * robust_normalize(redundancy, 0.0, 1.0))

    assert torch.allclose(components["U"], expected)
    assert torch.allclose(components["O"], torch.ones_like(expected))


def test_structural_score_is_independent_of_observation_counts():
    boundary = torch.tensor([1.0, 2.0, 4.0])
    turning = torch.tensor([0.0, 3.0, 6.0])
    defect = torch.tensor([2.0, 2.5, 5.0])
    redundancy = torch.tensor([0.0, 4.0, 8.0])
    lifetime_count = torch.tensor([3.0, 3.0, 3.0])
    recent_count = torch.tensor([0.0, 1.0, 3.0])

    structural_a = compute_structural_value_score(
        boundary, turning, defect, redundancy, low_quantile=0.0, high_quantile=1.0
    )
    structural_b = compute_structural_value_score(
        boundary, turning, defect, redundancy, low_quantile=0.0, high_quantile=1.0
    )
    obdkr_lifetime = compute_obdkr_value(
        lifetime_count, boundary, turning, defect, redundancy, low_quantile=0.0, high_quantile=1.0
    )
    obdkr_recent = compute_obdkr_value(
        recent_count, boundary, turning, defect, redundancy, low_quantile=0.0, high_quantile=1.0
    )

    assert torch.equal(structural_a, structural_b)
    assert not torch.equal(lifetime_count, recent_count)
    assert not torch.equal(obdkr_lifetime, obdkr_recent)


def test_constant_observation_preserves_obdkr_and_structural_ranking():
    counts = torch.tensor([3.0, 3.0, 3.0, 3.0])
    boundary = torch.tensor([0.0, 3.0, 1.0, 2.0])
    turning = torch.tensor([1.0, 0.0, 3.0, 2.0])
    defect = torch.tensor([0.5, 2.0, 1.0, 3.0])
    redundancy = torch.tensor([0.0, 2.0, 0.5, 1.0])

    obdkr = compute_obdkr_value(counts, boundary, turning, defect, redundancy, low_quantile=0.0, high_quantile=1.0)
    structural = compute_structural_value_score(boundary, turning, defect, redundancy, low_quantile=0.0, high_quantile=1.0)

    assert torch.equal(torch.argsort(obdkr, descending=True), torch.argsort(structural, descending=True))

    candidate_mask = torch.ones(4, dtype=torch.bool)
    dist = torch.tensor([4.0, 3.0, 2.0, 1.0])
    obdkr_mask, obdkr_stats = select_proximity_sources(
        candidate_mask, dist, n=1, rho=0.5, enabled=True, mode="value_global", value_score=obdkr
    )
    structural_mask, structural_stats = select_proximity_sources(
        candidate_mask, dist, n=1, rho=0.5, enabled=True, mode="value_global", value_score=structural
    )

    assert torch.equal(obdkr_mask, structural_mask)
    assert obdkr_stats.selected_indices == structural_stats.selected_indices


def test_structural_diagnostics_mark_score_variant():
    components = compute_structural_value_score(
        torch.tensor([1.0, 2.0]),
        torch.tensor([2.0, 3.0]),
        torch.tensor([3.0, 4.0]),
        torch.tensor([0.0, 1.0]),
        low_quantile=0.0,
        high_quantile=1.0,
        return_components=True,
    )

    stats = compute_obdkr_diagnostics(components, value_score_variant="structural")
    log_line = format_obdkr_diagnostics_log(1000, stats)

    assert stats["value_score_variant"] == "structural"
    assert "value_score_variant=structural" in log_line
    assert "structural_value_mean=" in log_line


def test_structural_recent_observation_source_does_not_require_recent_state(monkeypatch):
    module = _load_gaussian_model_module(monkeypatch)
    model = module.GaussianModel.__new__(module.GaussianModel)
    module.GaussianModel.setup_functions(model)
    model.args = Namespace(
        value_score_variant="structural",
        value_observation_source="recent",
        value_w_b=1.0,
        value_w_k=1.0,
        value_w_d=1.0,
        value_lambda_r=1.0,
        normalization_low_quantile=0.0,
        normalization_high_quantile=1.0,
    )
    model._xyz = torch.arange(12, dtype=torch.float32).view(4, 3)
    model._scaling = torch.zeros((4, 3), dtype=torch.float32)
    model._rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 4, dtype=torch.float32)
    model.visible_view_count = torch.tensor([3, 3, 3, 3], dtype=torch.long)
    model.recent_visible_view_count = None
    model.visibility_history = None
    model.recent_visibility_history = None
    model.camera_uid_to_train_index = None

    monkeypatch.setattr(module, "build_knn_graph", lambda xyz, k=12: torch.tensor([[1], [0], [1], [2]], dtype=torch.long))
    monkeypatch.setattr(module, "estimate_gaussian_normals", lambda scales, rotations: torch.tensor([[0.0, 0.0, 1.0]] * 4))
    monkeypatch.setattr(module, "compute_geometric_turning", lambda xyz, normals, neighbors: torch.tensor([1.0, 2.0, 3.0, 4.0]))
    monkeypatch.setattr(module, "compute_continuity_defect", lambda xyz, normals, neighbors: torch.tensor([4.0, 3.0, 2.0, 1.0]))
    monkeypatch.setattr(module, "compute_redundancy", lambda xyz, scales, neighbors: torch.zeros(4))

    components = model.compute_fsgs_proximity_candidate_value()

    assert components["U"].shape == (4,)
    assert torch.isfinite(components["U"]).all()


def test_gestalt_structural_lambda_zero_matches_structural_kdonly():
    boundary = torch.tensor([0.0, 0.0, 0.0])
    turning = torch.tensor([1.0, 2.0, 3.0])
    defect = torch.tensor([3.0, 2.0, 1.0])
    redundancy = torch.zeros(3)
    good_continuation = torch.tensor([0.0, 0.5, 1.0])

    structural = compute_structural_value_score(
        boundary,
        turning,
        defect,
        redundancy,
        w_b=0.0,
        w_k=1.0,
        w_d=1.0,
        lambda_r=0.0,
        low_quantile=0.0,
        high_quantile=1.0,
    )
    gestalt = compute_gestalt_structural_value_score(
        boundary,
        turning,
        defect,
        redundancy,
        good_continuation,
        w_b=0.0,
        w_k=1.0,
        w_d=1.0,
        lambda_r=0.0,
        low_quantile=0.0,
        high_quantile=1.0,
        gestalt_lambda=0.0,
    )

    assert torch.equal(gestalt, structural)


def test_gestalt_structural_g_zero_equals_base_u():
    boundary = torch.tensor([0.0, 0.0, 0.0])
    turning = torch.tensor([1.0, 2.0, 3.0])
    defect = torch.tensor([3.0, 2.0, 1.0])
    redundancy = torch.zeros(3)

    components = compute_gestalt_structural_value_score(
        boundary,
        turning,
        defect,
        redundancy,
        torch.zeros(3),
        w_b=0.0,
        w_k=1.0,
        w_d=1.0,
        lambda_r=0.0,
        low_quantile=0.0,
        high_quantile=1.0,
        return_components=True,
    )

    assert torch.equal(components["U"], components["U_base"])


def test_gestalt_structural_g_one_lambda_one_doubles_base_u():
    boundary = torch.tensor([0.0, 0.0, 0.0])
    turning = torch.tensor([1.0, 2.0, 3.0])
    defect = torch.tensor([3.0, 2.0, 1.0])
    redundancy = torch.zeros(3)

    components = compute_gestalt_structural_value_score(
        boundary,
        turning,
        defect,
        redundancy,
        torch.ones(3),
        w_b=0.0,
        w_k=1.0,
        w_d=1.0,
        lambda_r=0.0,
        low_quantile=0.0,
        high_quantile=1.0,
        gestalt_lambda=1.0,
        return_components=True,
    )

    assert torch.allclose(components["U"], 2.0 * components["U_base"])


def test_gestalt_structural_nan_inf_protection():
    components = compute_gestalt_structural_value_score(
        torch.tensor([0.0, float("nan"), 2.0]),
        torch.tensor([1.0, float("inf"), 3.0]),
        torch.tensor([3.0, 2.0, float("-inf")]),
        torch.tensor([0.0, 1.0, 2.0]),
        torch.tensor([0.0, float("nan"), float("inf")]),
        low_quantile=0.0,
        high_quantile=1.0,
        return_components=True,
    )

    assert torch.isfinite(components["U"]).all()
    assert torch.isfinite(components["G"]).all()
    assert ((components["G"] >= 0.0) & (components["G"] <= 1.0)).all()


def test_gestalt_structural_rejects_negative_lambda():
    with pytest.raises(ValueError, match="gestalt_value_lambda"):
        compute_gestalt_structural_value_score(
            torch.ones(2),
            torch.ones(2),
            torch.ones(2),
            torch.zeros(2),
            torch.ones(2),
            gestalt_lambda=-1.0,
        )


def test_balanced_gestalt_alpha_zero_equals_s_need():
    boundary = torch.zeros(3)
    turning = torch.tensor([0.0, 0.5, 1.0])
    defect = torch.tensor([1.0, 0.5, 0.0])
    redundancy = torch.ones(3)
    components = compute_balanced_gestalt_value_score(
        boundary,
        turning,
        defect,
        redundancy,
        torch.tensor([0.0, 0.5, 1.0]),
        low_quantile=0.0,
        high_quantile=1.0,
        gestalt_balance_alpha=0.0,
        return_components=True,
    )

    expected = 0.5 * (components["K_norm"] + components["D_norm"])
    assert torch.allclose(components["S_need"], expected)
    assert torch.equal(components["U"], components["S_need"])


def test_balanced_gestalt_alpha_one_equals_g():
    components = compute_balanced_gestalt_value_score(
        torch.zeros(3),
        torch.tensor([0.0, 0.5, 1.0]),
        torch.tensor([1.0, 0.5, 0.0]),
        torch.zeros(3),
        torch.tensor([0.0, 0.5, 1.0]),
        low_quantile=0.0,
        high_quantile=1.0,
        gestalt_balance_alpha=1.0,
        return_components=True,
    )

    assert torch.equal(components["U"], components["G"])


def test_balanced_gestalt_alpha_half_balances_need_and_continuation():
    components = compute_balanced_gestalt_value_score(
        torch.zeros(2),
        torch.tensor([1.0, 0.0]),
        torch.tensor([1.0, 0.0]),
        torch.zeros(2),
        torch.tensor([0.0, 1.0]),
        low_quantile=0.0,
        high_quantile=1.0,
        gestalt_balance_alpha=0.5,
        return_components=True,
    )

    assert torch.allclose(components["S_need"], torch.tensor([1.0, 0.0]))
    assert torch.allclose(components["U"], torch.tensor([0.5, 0.5]))


def test_balanced_gestalt_high_g_can_beat_high_kd_low_g():
    components = compute_balanced_gestalt_value_score(
        torch.zeros(3),
        torch.tensor([0.0, 0.6, 1.0]),
        torch.tensor([0.0, 0.6, 1.0]),
        torch.zeros(3),
        torch.tensor([0.0, 1.0, 0.0]),
        low_quantile=0.0,
        high_quantile=1.0,
        gestalt_balance_alpha=0.5,
        return_components=True,
    )

    assert components["S_need"][1].item() == pytest.approx(0.6)
    assert components["S_need"][2].item() == pytest.approx(1.0)
    assert components["U"][1].item() == pytest.approx(0.8)
    assert components["U"][2].item() == pytest.approx(0.5)
    assert torch.argsort(components["U"], descending=True).tolist()[:2] == [1, 2]


def test_balanced_gestalt_alpha_bounds():
    for alpha in (-0.1, 1.1):
        with pytest.raises(ValueError, match="gestalt_balance_alpha"):
            compute_balanced_gestalt_value_score(
                torch.ones(2),
                torch.ones(2),
                torch.ones(2),
                torch.zeros(2),
                torch.ones(2),
                gestalt_balance_alpha=alpha,
            )


def test_balanced_gestalt_nan_inf_protection():
    components = compute_balanced_gestalt_value_score(
        torch.tensor([0.0, float("nan"), 2.0]),
        torch.tensor([1.0, float("inf"), 3.0]),
        torch.tensor([3.0, 2.0, float("-inf")]),
        torch.tensor([0.0, 1.0, 2.0]),
        torch.tensor([0.0, float("nan"), float("inf")]),
        low_quantile=0.0,
        high_quantile=1.0,
        return_components=True,
    )

    assert torch.isfinite(components["U"]).all()
    assert torch.isfinite(components["G"]).all()
    assert torch.isfinite(components["S_need"]).all()


def test_defect_conditioned_gestalt_lambda_zero_matches_kdonly():
    boundary = torch.zeros(3)
    turning = torch.tensor([0.0, 0.5, 1.0])
    defect = torch.tensor([1.0, 0.5, 0.0])
    redundancy = torch.ones(3)
    good_continuation = torch.tensor([0.0, 0.5, 1.0])

    structural = compute_structural_value_score(
        boundary,
        turning,
        defect,
        redundancy,
        w_b=0.0,
        w_k=1.0,
        w_d=1.0,
        lambda_r=0.0,
        low_quantile=0.0,
        high_quantile=1.0,
    )
    conditioned = compute_defect_conditioned_gestalt_value_score(
        boundary,
        turning,
        defect,
        redundancy,
        good_continuation,
        low_quantile=0.0,
        high_quantile=1.0,
        gestalt_lambda=0.0,
    )

    assert torch.equal(conditioned, structural)


def test_defect_conditioned_gestalt_g_zero_equals_k_plus_d():
    components = compute_defect_conditioned_gestalt_value_score(
        torch.zeros(3),
        torch.tensor([0.0, 0.5, 1.0]),
        torch.tensor([1.0, 0.5, 0.0]),
        torch.zeros(3),
        torch.zeros(3),
        low_quantile=0.0,
        high_quantile=1.0,
        return_components=True,
    )

    assert torch.equal(components["U"], components["U_base"])


def test_defect_conditioned_gestalt_d_zero_does_not_gain_from_g():
    components = compute_defect_conditioned_gestalt_value_score(
        torch.zeros(3),
        torch.tensor([0.0, 0.4, 1.0]),
        torch.zeros(3),
        torch.zeros(3),
        torch.ones(3),
        low_quantile=0.0,
        high_quantile=1.0,
        return_components=True,
    )

    assert components["D"][1].item() == pytest.approx(0.0)
    assert components["G"][1].item() == pytest.approx(1.0)
    assert components["U"][1].item() == pytest.approx(0.4)


def test_defect_conditioned_gestalt_d_one_g_one_lambda_one():
    components = compute_defect_conditioned_gestalt_value_score(
        torch.zeros(3),
        torch.tensor([0.0, 0.3, 1.0]),
        torch.tensor([0.0, 1.0, 0.5]),
        torch.zeros(3),
        torch.tensor([0.0, 1.0, 0.0]),
        low_quantile=0.0,
        high_quantile=1.0,
        gestalt_lambda=1.0,
        return_components=True,
    )

    assert components["K"][1].item() == pytest.approx(0.3)
    assert components["D"][1].item() == pytest.approx(1.0)
    assert components["G"][1].item() == pytest.approx(1.0)
    assert components["U"][1].item() == pytest.approx(2.3)


def test_defect_conditioned_gestalt_high_g_low_defect_does_not_win_by_itself():
    components = compute_defect_conditioned_gestalt_value_score(
        torch.zeros(4),
        torch.zeros(4),
        torch.tensor([0.0, 0.2, 0.6, 1.0]),
        torch.zeros(4),
        torch.tensor([0.0, 1.0, 0.0, 0.0]),
        low_quantile=0.0,
        high_quantile=1.0,
        gestalt_lambda=1.0,
        return_components=True,
    )

    assert components["U"][1].item() == pytest.approx(0.4)
    assert components["U"][2].item() == pytest.approx(0.6)
    assert components["U"][2].item() > components["U"][1].item()


def test_defect_conditioned_gestalt_high_g_wins_when_defect_matches():
    components = compute_defect_conditioned_gestalt_value_score(
        torch.zeros(4),
        torch.zeros(4),
        torch.tensor([0.0, 0.5, 0.5, 1.0]),
        torch.zeros(4),
        torch.tensor([0.0, 1.0, 0.0, 0.0]),
        low_quantile=0.0,
        high_quantile=1.0,
        gestalt_lambda=1.0,
        return_components=True,
    )

    assert components["D"][1].item() == pytest.approx(0.5)
    assert components["D"][2].item() == pytest.approx(0.5)
    assert components["U"][1].item() > components["U"][2].item()


def test_defect_conditioned_gestalt_nan_inf_protection():
    components = compute_defect_conditioned_gestalt_value_score(
        torch.tensor([0.0, float("nan"), 2.0]),
        torch.tensor([1.0, float("inf"), 3.0]),
        torch.tensor([3.0, 2.0, float("-inf")]),
        torch.tensor([0.0, 1.0, 2.0]),
        torch.tensor([0.0, float("nan"), float("inf")]),
        low_quantile=0.0,
        high_quantile=1.0,
        return_components=True,
    )

    for name in ("K", "D", "G", "DG", "U"):
        assert torch.isfinite(components[name]).all()


def test_defect_conditioned_gestalt_rejects_negative_lambda():
    with pytest.raises(ValueError, match="gestalt_value_lambda"):
        compute_defect_conditioned_gestalt_value_score(
            torch.ones(2),
            torch.ones(2),
            torch.ones(2),
            torch.zeros(2),
            torch.ones(2),
            gestalt_lambda=-1.0,
        )

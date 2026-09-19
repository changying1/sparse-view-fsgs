import importlib.util
import sys
import types
from argparse import Namespace

import torch

from utils.value_allocation import compute_obdkr_value
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

    spec = importlib.util.spec_from_file_location("_fsgs_gaussian_model_recent_test", "scene/gaussian_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_model(module, point_count=4):
    model = module.GaussianModel.__new__(module.GaussianModel)
    module.GaussianModel.setup_functions(model)
    model.args = Namespace(prune_from_iter=0)
    model._xyz = torch.arange(point_count * 3, dtype=torch.float32).view(point_count, 3)
    model.visibility_history = None
    model.visible_view_count = None
    model.recent_visibility_history = None
    model.recent_visible_view_count = None
    return model


def test_recent_initialization_and_updates(monkeypatch):
    module = _load_gaussian_model_module(monkeypatch)
    model = _make_model(module, point_count=4)

    model.ensure_visibility_history(3)
    assert model.recent_visibility_history.shape == (4, 3)
    assert model.recent_visible_view_count.tolist() == [0, 0, 0, 0]

    model.update_visibility(0, torch.tensor([True, False, False, False]))
    assert model.recent_visible_view_count.tolist() == [1, 0, 0, 0]

    model.update_visibility(0, torch.tensor([True, False, False, False]))
    assert model.recent_visible_view_count.tolist() == [1, 0, 0, 0]

    model.update_visibility(1, torch.tensor([True, False, False, False]))
    assert model.recent_visible_view_count.tolist() == [2, 0, 0, 0]


def test_reset_recent_visibility_preserves_lifetime(monkeypatch):
    module = _load_gaussian_model_module(monkeypatch)
    model = _make_model(module, point_count=4)
    model.ensure_visibility_history(3)
    model.update_visibility(0, torch.tensor([True, False, False, False]))
    model.update_visibility(2, torch.tensor([True, True, False, False]))
    lifetime_history = model.visibility_history.clone()
    lifetime_count = model.visible_view_count.clone()

    model.reset_recent_visibility()

    assert not model.recent_visibility_history.any()
    assert model.recent_visible_view_count.tolist() == [0, 0, 0, 0]
    assert torch.equal(model.visibility_history, lifetime_history)
    assert torch.equal(model.visible_view_count, lifetime_count)


def test_recent_clone_split_and_proximity_inheritance(monkeypatch):
    module = _load_gaussian_model_module(monkeypatch)

    clone_model = _make_model(module, point_count=2)
    clone_model.ensure_visibility_history(3)
    clone_model.recent_visibility_history[:] = torch.tensor([[True, False, True], [False, True, False]])
    clone_model.recent_visible_view_count = clone_model.recent_visibility_history.sum(dim=1).to(dtype=torch.long)
    clone_model.visibility_history[:] = clone_model.recent_visibility_history
    clone_model.visible_view_count = clone_model.visibility_history.sum(dim=1).to(dtype=torch.long)
    clone_model._append_visibility_from_masks(torch.tensor([True, False]))
    assert clone_model.recent_visibility_history[2].tolist() == [True, False, True]

    split_model = _make_model(module, point_count=2)
    split_model.ensure_visibility_history(3)
    split_model.recent_visibility_history[:] = torch.tensor([[True, False, True], [False, True, False]])
    split_model.recent_visible_view_count = split_model.recent_visibility_history.sum(dim=1).to(dtype=torch.long)
    split_model.visibility_history[:] = split_model.recent_visibility_history
    split_model.visible_view_count = split_model.visibility_history.sum(dim=1).to(dtype=torch.long)
    split_model._append_visibility_from_masks(torch.tensor([True, False]), repeat_count=2)
    assert split_model.recent_visibility_history[2].tolist() == [True, False, True]
    assert split_model.recent_visibility_history[3].tolist() == [True, False, True]

    proximity_model = _make_model(module, point_count=2)
    proximity_model.ensure_visibility_history(3)
    proximity_model.recent_visibility_history[:] = torch.tensor([[True, False, True], [False, True, False]])
    proximity_model.recent_visible_view_count = proximity_model.recent_visibility_history.sum(dim=1).to(dtype=torch.long)
    proximity_model.visibility_history[:] = proximity_model.recent_visibility_history
    proximity_model.visible_view_count = proximity_model.visibility_history.sum(dim=1).to(dtype=torch.long)
    proximity_model._append_visibility_from_masks(torch.tensor([True, False]), repeat_count=1, target_indices=torch.tensor([1]))
    assert proximity_model.recent_visibility_history[2].tolist() == [True, True, True]


def test_recent_prune_synchronization(monkeypatch):
    module = _load_gaussian_model_module(monkeypatch)
    model = _make_model(module, point_count=4)
    model.ensure_visibility_history(3)
    model.update_visibility(0, torch.tensor([True, False, True, False]))
    model.update_visibility(1, torch.tensor([False, True, True, False]))

    valid = torch.tensor([True, False, True, False])
    model._xyz = model._xyz[valid]
    model._prune_visibility(valid)

    assert model.recent_visibility_history.shape[0] == model.get_xyz.shape[0]
    assert model.recent_visible_view_count.shape[0] == model.get_xyz.shape[0]
    assert model.recent_visible_view_count.tolist() == [1, 2]


def test_capture_restore_does_not_checkpoint_recent_state(monkeypatch):
    module = _load_gaussian_model_module(monkeypatch)
    model = _make_model(module, point_count=2)
    model.active_sh_degree = 0
    model._features_dc = torch.zeros((2, 1, 3))
    model._features_rest = torch.zeros((2, 0, 3))
    model._scaling = torch.zeros((2, 3))
    model._rotation = torch.zeros((2, 4))
    model._opacity = torch.zeros((2, 1))
    model.max_radii2D = torch.zeros(2)
    model.xyz_gradient_accum = torch.zeros((2, 1))
    model.denom = torch.zeros((2, 1))
    model.optimizer = types.SimpleNamespace(state_dict=lambda: {})
    model.spatial_lr_scale = 1.0
    model.ensure_visibility_history(3)
    model.update_visibility(0, torch.tensor([True, False]))

    captured = model.capture()

    assert len(captured) == 14
    visibility_state = captured[-2]
    rgg_state = captured[-1]
    assert "recent_visibility_history" not in visibility_state
    assert "recent_visible_view_count" not in visibility_state
    assert rgg_state["uid"].tolist() == [0, 1]


def test_obdkr_training_semantics_ignore_recent_counts():
    lifetime_count = torch.tensor([3.0, 3.0, 3.0])
    recent_count_a = torch.tensor([1.0, 2.0, 3.0])
    recent_count_b = torch.tensor([0.0, 0.0, 0.0])
    boundary = torch.tensor([1.0, 2.0, 3.0])
    turning = torch.tensor([1.0, 2.0, 3.0])
    defect = torch.tensor([1.0, 2.0, 3.0])
    redundancy = torch.zeros(3)

    components_a = compute_obdkr_value(lifetime_count, boundary, turning, defect, redundancy, return_components=True)
    components_b = compute_obdkr_value(lifetime_count, boundary, turning, defect, redundancy, return_components=True)

    assert torch.equal(components_a["O"], components_b["O"])
    assert torch.equal(components_a["U"], components_b["U"])
    assert not torch.equal(recent_count_a, recent_count_b)


def test_diagnostics_include_recent_observation_fields_and_zero_state_safe():
    counts = torch.tensor([3, 3, 3], dtype=torch.long)
    recent = torch.tensor([1, 2, 3], dtype=torch.long)
    boundary = torch.tensor([1.0, 2.0, 3.0])
    turning = torch.tensor([1.0, 2.0, 3.0])
    defect = torch.tensor([1.0, 2.0, 3.0])
    redundancy = torch.zeros(3)
    candidate_mask = torch.tensor([True, False, True])
    components = compute_obdkr_value(counts, boundary, turning, defect, redundancy, return_components=True)

    stats = compute_obdkr_diagnostics(
        components,
        candidate_mask,
        observation_count=counts,
        recent_observation_count=recent,
        num_train_views=3,
    )
    log_line = format_obdkr_diagnostics_log(1000, stats)

    assert stats["obs_unique_count"] == 1
    assert stats["recent_obs_unique_count"] == 3
    assert stats["recent_obs_saturated_ratio"] == 1 / 3
    assert stats["candidate_recent_obs_saturated_ratio"] == 1 / 2
    for field in (
        "recent_obs_mean=",
        "recent_obs_min=",
        "recent_obs_max=",
        "recent_obs_q25=",
        "recent_obs_q50=",
        "recent_obs_q75=",
        "recent_obs_unique_count=",
        "candidate_recent_obs_mean=",
        "candidate_recent_obs_min=",
        "candidate_recent_obs_max=",
        "candidate_recent_obs_unique_count=",
        "recent_obs_saturated_ratio=",
        "candidate_recent_obs_saturated_ratio=",
    ):
        assert field in log_line

    zero_stats = compute_obdkr_diagnostics(
        components,
        candidate_mask,
        observation_count=counts,
        recent_observation_count=torch.zeros(3, dtype=torch.long),
        num_train_views=3,
    )
    assert zero_stats["recent_obs_mean"] == 0.0
    assert zero_stats["recent_obs_unique_count"] == 1

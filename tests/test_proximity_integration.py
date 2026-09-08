import importlib.util
import sys
import types
from argparse import Namespace

import pytest
import torch


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

    spec = importlib.util.spec_from_file_location("_fsgs_gaussian_model_test", "scene/gaussian_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_proximity_selection_mask_flows_into_real_midpoint_unpool(monkeypatch):
    module = _load_gaussian_model_module(monkeypatch)
    model = module.GaussianModel.__new__(module.GaussianModel)
    module.GaussianModel.setup_functions(model)
    model.args = Namespace(
        enable_proximity_budget=True,
        proximity_growth_ratio=0.5,
        proximity_selection_mode="proximity_topk",
        value_rerank_fraction=0.25,
        value_boundary_multiplier=2.0,
    )
    model._xyz = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
            [40.0, 0.0, 0.0],
            [50.0, 0.0, 0.0],
            [60.0, 0.0, 0.0],
            [70.0, 0.0, 0.0],
            [80.0, 0.0, 0.0],
            [90.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [110.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    model._scaling = torch.full((12, 3), torch.log(torch.tensor(2.0)).item(), dtype=torch.float32)
    model._rotation = torch.zeros((12, 4), dtype=torch.float32)
    model._features_dc = torch.zeros((12, 1, 3), dtype=torch.float32)
    model._features_rest = torch.zeros((12, 0, 3), dtype=torch.float32)
    model._opacity = torch.zeros((12, 1), dtype=torch.float32)
    model.confidence = torch.ones((12, 1), dtype=torch.float32)
    model.denom = torch.ones((12, 1), dtype=torch.float32)
    model.visibility_history = None
    model.visible_view_count = None
    model.camera_uid_to_train_index = None

    dist = torch.tensor([20.0, 40.0, 30.0, 10.0] + [0.0] * 8)
    nearest = torch.tensor(
        [
            [1, 2, 3],
            [0, 2, 3],
            [0, 1, 3],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
        ],
        dtype=torch.long,
    )
    monkeypatch.setattr(module, "distCUDA2", lambda xyz: (dist, nearest))

    captured = {}

    def fake_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        captured["new_xyz"] = new_xyz

    model.densification_postfix = fake_postfix

    proximity_sources, proximity_proposed, selected_sources, selected_new = model.proximity(scene_extent=1.0, iteration=1200, N=3)

    assert proximity_sources == 4
    assert proximity_proposed == 12
    assert selected_sources == 2
    assert selected_new == 6
    assert captured["new_xyz"].shape[0] == 6
    expected_sources = model._xyz[torch.tensor([1, 2])][:, None, :].repeat(1, 3, 1).reshape(-1, 3)
    expected_targets = model._xyz[nearest[torch.tensor([1, 2])].reshape(-1)]
    assert torch.equal(captured["new_xyz"], (expected_sources + expected_targets) / 2)


def test_candidate_capacity_flows_into_real_midpoint_unpool(monkeypatch):
    module = _load_gaussian_model_module(monkeypatch)
    model = module.GaussianModel.__new__(module.GaussianModel)
    module.GaussianModel.setup_functions(model)
    model.args = Namespace(
        enable_proximity_budget=False,
        proximity_growth_ratio=0.5,
        enable_proximity_candidate_capacity=True,
        proximity_candidate_keep_ratio=0.8,
        proximity_selection_mode="proximity_topk",
        value_rerank_fraction=0.25,
        value_boundary_multiplier=2.0,
    )
    model._xyz = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
            [40.0, 0.0, 0.0],
            [50.0, 0.0, 0.0],
            [60.0, 0.0, 0.0],
            [70.0, 0.0, 0.0],
            [80.0, 0.0, 0.0],
            [90.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [110.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    model._scaling = torch.full((12, 3), torch.log(torch.tensor(2.0)).item(), dtype=torch.float32)
    model._rotation = torch.zeros((12, 4), dtype=torch.float32)
    model._features_dc = torch.zeros((12, 1, 3), dtype=torch.float32)
    model._features_rest = torch.zeros((12, 0, 3), dtype=torch.float32)
    model._opacity = torch.zeros((12, 1), dtype=torch.float32)
    model.confidence = torch.ones((12, 1), dtype=torch.float32)
    model.denom = torch.ones((12, 1), dtype=torch.float32)
    model.visibility_history = None
    model.visible_view_count = None
    model.camera_uid_to_train_index = None

    dist = torch.tensor([20.0, 40.0, 30.0, 10.0] + [0.0] * 8)
    nearest = torch.tensor(
        [
            [1, 2, 3],
            [0, 2, 3],
            [0, 1, 3],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
        ],
        dtype=torch.long,
    )
    monkeypatch.setattr(module, "distCUDA2", lambda xyz: (dist, nearest))

    captured = {}
    model.densification_postfix = lambda new_xyz, *args: captured.setdefault("new_xyz", new_xyz)

    proximity_sources, proximity_proposed, selected_sources, selected_new = model.proximity(scene_extent=1.0, iteration=1200, N=3)

    assert proximity_sources == 4
    assert proximity_proposed == 12
    assert selected_sources == 3
    assert selected_new == 9
    assert captured["new_xyz"].shape[0] == 9
    expected_indices = torch.tensor([0, 1, 2])
    expected_sources = model._xyz[expected_indices][:, None, :].repeat(1, 3, 1).reshape(-1, 3)
    expected_targets = model._xyz[nearest[expected_indices].reshape(-1)]
    assert torch.equal(captured["new_xyz"], (expected_sources + expected_targets) / 2)


def test_action_replay_flows_into_real_midpoint_unpool_with_n3(monkeypatch):
    module = _load_gaussian_model_module(monkeypatch)
    model = module.GaussianModel.__new__(module.GaussianModel)
    module.GaussianModel.setup_functions(model)
    model.args = Namespace(
        enable_proximity_budget=False,
        proximity_growth_ratio=-1.0,
        enable_proximity_candidate_capacity=False,
        proximity_candidate_keep_ratio=0.8,
        proximity_action_replay="1200:2",
        proximity_selection_mode="proximity_topk",
        value_rerank_fraction=0.25,
        value_boundary_multiplier=2.0,
        value_demand_ratio=0.90,
    )
    model._xyz = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
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

    dist = torch.tensor([20.0, 40.0, 30.0, 10.0])
    nearest = torch.tensor([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]], dtype=torch.long)
    monkeypatch.setattr(module, "distCUDA2", lambda xyz: (dist, nearest))

    captured = {}
    model.densification_postfix = lambda new_xyz, *args: captured.setdefault("new_xyz", new_xyz)

    proximity_sources, proximity_proposed, selected_sources, selected_new = model.proximity(
        scene_extent=1.0,
        iteration=1200,
        N=3,
    )

    assert proximity_sources == 4
    assert proximity_proposed == 12
    assert selected_sources == 2
    assert selected_new == 6
    assert captured["new_xyz"].shape[0] == selected_sources * 3
    expected_indices = torch.tensor([1, 2])
    expected_sources = model._xyz[expected_indices][:, None, :].repeat(1, 3, 1).reshape(-1, 3)
    expected_targets = model._xyz[nearest[expected_indices].reshape(-1)]
    assert torch.equal(captured["new_xyz"], (expected_sources + expected_targets) / 2)


def test_visibility_update_and_proximity_midpoint_visibility_union(monkeypatch):
    module = _load_gaussian_model_module(monkeypatch)
    model = module.GaussianModel.__new__(module.GaussianModel)
    module.GaussianModel.setup_functions(model)
    model.args = Namespace(
        enable_proximity_budget=True,
        proximity_growth_ratio=1.0,
        proximity_selection_mode="proximity_topk",
        value_rerank_fraction=0.25,
        value_boundary_multiplier=2.0,
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
    model.ensure_visibility_history(3)
    model.update_visibility(0, torch.tensor([True, False, False, False]))
    model.update_visibility(1, torch.tensor([False, True, False, False]))

    dist = torch.tensor([20.0, 30.0, 0.0, 0.0])
    nearest = torch.tensor([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]], dtype=torch.long)
    monkeypatch.setattr(module, "distCUDA2", lambda xyz: (dist, nearest))

    def fake_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        old_n = model._xyz.shape[0]
        model._xyz = torch.cat((model._xyz, new_xyz), dim=0)
        model._features_dc = torch.cat((model._features_dc, new_features_dc), dim=0)
        model._features_rest = torch.cat((model._features_rest, new_features_rest), dim=0)
        model._opacity = torch.cat((model._opacity, new_opacities), dim=0)
        model._scaling = torch.cat((model._scaling, new_scaling), dim=0)
        model._rotation = torch.cat((model._rotation, new_rotation), dim=0)
        assert old_n == 4

    model.densification_postfix = fake_postfix
    model.proximity(scene_extent=1.0, iteration=1200, N=3)

    assert model.visibility_history.shape[0] == model.get_xyz.shape[0]
    assert model.visible_view_count.shape[0] == model.get_xyz.shape[0]
    assert model.visibility_history[4].tolist() == [True, True, False]
    assert model.visible_view_count[4].item() == 2


def test_real_value_uses_visible_view_count_not_denom(monkeypatch):
    module = _load_gaussian_model_module(monkeypatch)
    model = module.GaussianModel.__new__(module.GaussianModel)
    module.GaussianModel.setup_functions(model)
    model.args = Namespace(
        value_tau_e=1.0,
        value_tau_s=3.0,
        value_w_b=1.0,
        value_w_k=1.0,
        value_w_d=1.0,
        value_lambda_r=1.0,
        normalization_low_quantile=0.0,
        normalization_high_quantile=1.0,
    )
    model._xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float32)
    model._scaling = torch.zeros((3, 3), dtype=torch.float32)
    model._rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3, dtype=torch.float32)
    model.visible_view_count = torch.tensor([0, 1, 12], dtype=torch.long)
    model.visibility_history = None
    model.camera_uid_to_train_index = None
    model.denom = torch.tensor([[100.0], [100.0], [100.0]])

    monkeypatch.setattr(module, "aggregate_multiview_edge_support", lambda **kwargs: torch.tensor([1.0, 2.0, 3.0]))
    monkeypatch.setattr(module, "build_knn_graph", lambda xyz, k=12: torch.tensor([[1], [0], [1]], dtype=torch.long))
    monkeypatch.setattr(module, "estimate_gaussian_normals", lambda scales, rotations: torch.tensor([[0.0, 0.0, 1.0]] * 3))
    monkeypatch.setattr(module, "compute_geometric_turning", lambda xyz, normals, neighbors: torch.tensor([1.0, 2.0, 3.0]))
    monkeypatch.setattr(module, "compute_continuity_defect", lambda xyz, normals, neighbors: torch.tensor([1.0, 2.0, 3.0]))
    monkeypatch.setattr(module, "compute_redundancy", lambda xyz, scales, neighbors: torch.zeros(3))

    components = model.compute_fsgs_proximity_candidate_value(
        train_cameras=[object()],
        edge_maps=[torch.ones((2, 2))],
    )

    assert components["O"][0].item() == 0.0
    assert components["U"][0].item() == 0.0
    assert components["U"][1].item() > 0.0


def test_value_observation_source_selects_lifetime_or_recent_for_o(monkeypatch):
    module = _load_gaussian_model_module(monkeypatch)

    def make_model(source):
        model = module.GaussianModel.__new__(module.GaussianModel)
        module.GaussianModel.setup_functions(model)
        model.args = Namespace(
            value_tau_e=1.0,
            value_tau_s=3.0,
            value_observation_source=source,
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
        model.recent_visible_view_count = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        model.visibility_history = None
        model.recent_visibility_history = torch.tensor(
            [
                [False, False, False],
                [True, False, False],
                [True, True, False],
                [True, True, True],
            ],
            dtype=torch.bool,
        )
        model.camera_uid_to_train_index = None
        return model

    monkeypatch.setattr(module, "build_knn_graph", lambda xyz, k=12: torch.tensor([[1], [0], [1], [2]], dtype=torch.long))
    monkeypatch.setattr(module, "estimate_gaussian_normals", lambda scales, rotations: torch.tensor([[0.0, 0.0, 1.0]] * 4))
    monkeypatch.setattr(module, "compute_geometric_turning", lambda xyz, normals, neighbors: torch.ones(4))
    monkeypatch.setattr(module, "compute_continuity_defect", lambda xyz, normals, neighbors: torch.ones(4))
    monkeypatch.setattr(module, "compute_redundancy", lambda xyz, scales, neighbors: torch.zeros(4))

    lifetime_components = make_model("lifetime").compute_fsgs_proximity_candidate_value()
    recent_components = make_model("recent").compute_fsgs_proximity_candidate_value()

    assert torch.allclose(lifetime_components["O"], lifetime_components["O"][0].expand(4))
    assert recent_components["O"][0].item() == 0.0
    assert recent_components["O"][1].item() > 0.0
    assert recent_components["O"][1].item() != recent_components["O"][3].item()


def test_recent_observation_source_requires_recent_state(monkeypatch):
    module = _load_gaussian_model_module(monkeypatch)
    model = module.GaussianModel.__new__(module.GaussianModel)
    module.GaussianModel.setup_functions(model)
    model.args = Namespace(value_observation_source="recent")
    model._xyz = torch.arange(12, dtype=torch.float32).view(4, 3)
    model._scaling = torch.zeros((4, 3), dtype=torch.float32)
    model._rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 4, dtype=torch.float32)
    model.visible_view_count = torch.tensor([3, 3, 3, 3], dtype=torch.long)
    model.recent_visible_view_count = None

    with pytest.raises(ValueError, match="recent observation source requested"):
        model.compute_fsgs_proximity_candidate_value()


def test_original_mode_preserves_all_fsgs_proximity_candidates_and_midpoints(monkeypatch):
    module = _load_gaussian_model_module(monkeypatch)
    model = module.GaussianModel.__new__(module.GaussianModel)
    module.GaussianModel.setup_functions(model)
    model.args = Namespace(
        enable_proximity_budget=False,
        proximity_growth_ratio=0.10,
        proximity_selection_mode="original",
        value_rerank_fraction=0.25,
        value_boundary_multiplier=2.0,
    )
    model._xyz = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
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

    dist = torch.tensor([20.0, 30.0, 40.0, 50.0])
    nearest = torch.tensor([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]], dtype=torch.long)
    monkeypatch.setattr(module, "distCUDA2", lambda xyz: (dist, nearest))

    captured = {}

    def fake_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        captured["new_xyz"] = new_xyz

    model.densification_postfix = fake_postfix
    proximity_sources, proximity_proposed, selected_sources, selected_new = model.proximity(scene_extent=1.0, iteration=1200, N=3)

    assert proximity_sources == 4
    assert proximity_proposed == 12
    assert selected_sources == proximity_sources
    assert selected_new == proximity_proposed
    assert captured["new_xyz"].shape[0] == proximity_proposed

    expected_sources = model._xyz[:, None, :].repeat(1, 3, 1).reshape(-1, 3)
    expected_targets = model._xyz[nearest.reshape(-1)]
    assert torch.equal(captured["new_xyz"], (expected_sources + expected_targets) / 2)


def test_original_and_a1_proximity_do_not_compute_value(monkeypatch):
    module = _load_gaussian_model_module(monkeypatch)
    for mode, enabled in (("original", False), ("proximity_topk", True), ("original", True)):
        model = module.GaussianModel.__new__(module.GaussianModel)
        module.GaussianModel.setup_functions(model)
        model.args = Namespace(
            enable_proximity_budget=enabled,
            proximity_growth_ratio=1.0,
            proximity_selection_mode=mode,
            value_observation_source="recent",
            value_rerank_fraction=0.25,
            value_boundary_multiplier=2.0,
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
        monkeypatch.setattr(model, "compute_fsgs_proximity_candidate_value", lambda **kwargs: (_ for _ in ()).throw(AssertionError("OBDKR should not run")))
        monkeypatch.setattr(model, "densification_postfix", lambda *args: None)
        monkeypatch.setattr(module, "distCUDA2", lambda xyz: (torch.tensor([20.0, 30.0, 40.0, 50.0]), torch.tensor([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]], dtype=torch.long)))

        model.proximity(scene_extent=1.0, iteration=1200, N=3)


def test_value_modes_compute_obdkr_once(monkeypatch):
    module = _load_gaussian_model_module(monkeypatch)
    for mode in ("value_global", "value_rerank", "value_demand_rerank"):
        model = module.GaussianModel.__new__(module.GaussianModel)
        module.GaussianModel.setup_functions(model)
        model.args = Namespace(
            enable_proximity_budget=True,
            proximity_growth_ratio=1.0,
            proximity_selection_mode=mode,
            value_rerank_fraction=0.25,
            value_boundary_multiplier=2.0,
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
        monkeypatch.setattr(module, "distCUDA2", lambda xyz: (torch.tensor([20.0, 30.0, 40.0, 50.0]), torch.tensor([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]], dtype=torch.long)))
        monkeypatch.setattr(model, "densification_postfix", lambda *args: None)
        calls = []

        def fake_value(**kwargs):
            calls.append(kwargs)
            return {"U": torch.tensor([1.0, 2.0, 3.0, 4.0])}

        monkeypatch.setattr(model, "compute_fsgs_proximity_candidate_value", fake_value)
        model.proximity(scene_extent=1.0, iteration=1200, N=3)

        assert len(calls) == 1
        assert calls[0]["knn_k"] == 7
        assert torch.equal(calls[0]["candidate_mask"], torch.tensor([True, True, True, True]))

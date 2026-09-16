import importlib.util
import random
import sys
import types
from argparse import Namespace

import numpy as np
import pytest
import torch

from utils.paired_fork_checkpoint import load_paired_fork_checkpoint, save_paired_fork_checkpoint


def _load_gaussian_model_module(monkeypatch):
    simple_knn = types.ModuleType("simple_knn")
    simple_knn_c = types.ModuleType("simple_knn._C")
    simple_knn_c.distCUDA2 = lambda xyz: (torch.zeros(xyz.shape[0]), torch.zeros((xyz.shape[0], 1), dtype=torch.long))
    monkeypatch.setitem(sys.modules, "simple_knn", simple_knn)
    monkeypatch.setitem(sys.modules, "simple_knn._C", simple_knn_c)

    plyfile = types.ModuleType("plyfile")
    plyfile.PlyData = object
    plyfile.PlyElement = object
    monkeypatch.setitem(sys.modules, "plyfile", plyfile)

    spec = importlib.util.spec_from_file_location("_fsgs_gaussian_model_rgg_test", "scene/gaussian_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(enabled=True):
    return Namespace(
        train_bg=False,
        sh_degree=3,
        use_color=True,
        enable_rgg_diagnostics=enabled,
        prune_from_iter=0,
        percent_dense=0.5,
        dist_thres=1000.0,
        enable_proximity_budget=False,
        enable_proximity_candidate_capacity=False,
        proximity_action_replay="",
        proximity_selection_mode="original",
        proximity_growth_ratio=1.0,
        proximity_candidate_keep_ratio=0.8,
        value_rerank_fraction=0.25,
        value_boundary_multiplier=2.0,
        value_demand_ratio=0.9,
        value_rerank_start_iter=0,
        value_rerank_end_iter=sys.maxsize,
        enable_child_structure_diagnostics=False,
        enable_structural_child_target_selection=False,
    )


def _training_args():
    return Namespace(
        percent_dense=0.5,
        position_lr_init=0.01,
        position_lr_final=0.001,
        position_lr_delay_mult=0.01,
        position_lr_max_steps=1000,
        feature_lr=0.0025,
        opacity_lr=0.05,
        scaling_lr=0.005,
        rotation_lr=0.001,
    )


def _make_model(module, point_count=4, enabled=True):
    model = module.GaussianModel(_args(enabled=enabled))
    model.active_sh_degree = 0
    model.max_sh_degree = 3
    model.spatial_lr_scale = 1.0
    model._xyz = torch.nn.Parameter(torch.arange(point_count * 3, dtype=torch.float32).view(point_count, 3) / 10)
    model._features_dc = torch.nn.Parameter(torch.zeros((point_count, 1, 3), dtype=torch.float32))
    model._features_rest = torch.nn.Parameter(torch.zeros((point_count, 3, 3), dtype=torch.float32))
    model._scaling = torch.nn.Parameter(torch.full((point_count, 3), torch.log(torch.tensor(0.25)).item()))
    model._rotation = torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]] * point_count))
    model._opacity = torch.nn.Parameter(torch.ones((point_count, 1), dtype=torch.float32) * -2.0)
    model.max_radii2D = torch.zeros((point_count,), dtype=torch.float32)
    model.xyz_gradient_accum = torch.zeros((point_count, 1), dtype=torch.float32)
    model.denom = torch.ones((point_count, 1), dtype=torch.float32)
    model.confidence = torch.ones((point_count, 1), dtype=torch.float32)
    model.visibility_history = None
    model.visible_view_count = None
    model.recent_visibility_history = None
    model.recent_visible_view_count = None
    model.camera_uid_to_train_index = None
    model._initialize_rgg_roots(birth_iter=0)
    model.training_setup(_training_args())
    return model


def _assert_aligned(model):
    count = model.get_xyz.shape[0]
    assert model.rgg_uid.shape[0] == count
    assert model.rgg_birth_iter.shape[0] == count
    assert model.rgg_source_uid.shape[0] == count
    assert model.rgg_target_uid.shape[0] == count
    assert model.rgg_generation.shape[0] == count


def _patch_cpu_split_sampling(monkeypatch, module):
    monkeypatch.setattr(torch, "normal", lambda mean, std: torch.zeros_like(std))
    monkeypatch.setattr(
        module,
        "build_rotation",
        lambda rotation: torch.eye(
            3,
            dtype=rotation.dtype,
            device=rotation.device,
        ).unsqueeze(0).repeat(rotation.shape[0], 1, 1),
    )


def test_root_uid_unique_and_root_lineage(monkeypatch):
    module = _load_gaussian_model_module(monkeypatch)
    model = _make_model(module, point_count=5, enabled=True)

    assert model.rgg_uid.tolist() == [0, 1, 2, 3, 4]
    assert torch.unique(model.rgg_uid).numel() == 5
    assert model.rgg_source_uid.tolist() == [-1] * 5
    assert model.rgg_target_uid.tolist() == [-1] * 5
    assert model.rgg_generation.tolist() == [0] * 5
    assert model.rgg_next_uid == 5


def test_clone_lineage_generation_and_monotonic_uid(monkeypatch):
    module = _load_gaussian_model_module(monkeypatch)
    model = _make_model(module, point_count=3, enabled=True)
    grads = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    assert model.densify_and_clone(grads, grad_threshold=0.5, scene_extent=1.0, iter=17) == 2

    _assert_aligned(model)
    assert model.rgg_uid.tolist() == [0, 1, 2, 3, 4]
    assert model.rgg_source_uid[-2:].tolist() == [0, 2]
    assert model.rgg_target_uid[-2:].tolist() == [-1, -1]
    assert model.rgg_birth_iter[-2:].tolist() == [17, 17]
    assert model.rgg_generation[-2:].tolist() == [1, 1]
    assert model.rgg_next_uid == 5


def test_split_lineage_and_each_child_gets_distinct_uid(monkeypatch):
    module = _load_gaussian_model_module(monkeypatch)
    model = _make_model(module, point_count=3, enabled=True)
    grads = torch.tensor([[1.0], [0.0], [0.0]])
    module.distCUDA2 = lambda xyz: (torch.zeros(xyz.shape[0]), torch.zeros((xyz.shape[0], 1), dtype=torch.long))
    _patch_cpu_split_sampling(monkeypatch, module)

    split_stats = model.densify_and_split(grads, grad_threshold=0.5, scene_extent=0.1, iter=23, N=2)

    _assert_aligned(model)
    assert split_stats == (1, 0, 1)
    assert model.rgg_uid.tolist() == [1, 2, 3, 4]
    assert model.rgg_source_uid[-2:].tolist() == [0, 0]
    assert model.rgg_target_uid[-2:].tolist() == [-1, -1]
    assert model.rgg_birth_iter[-2:].tolist() == [23, 23]
    assert model.rgg_generation[-2:].tolist() == [1, 1]
    assert model.rgg_next_uid == 5


def test_proximity_records_source_and_target_uid_with_generation(monkeypatch):
    module = _load_gaussian_model_module(monkeypatch)
    model = _make_model(module, point_count=3, enabled=True)
    model._scaling.data.fill_(torch.log(torch.tensor(2.0)).item())
    model.rgg_generation[1] = 2
    model.rgg_generation[2] = 5
    dist = torch.tensor([0.0, 6.0, 0.0])
    nearest = torch.tensor([[0], [2], [0]], dtype=torch.long)
    module.distCUDA2 = lambda xyz: (dist, nearest)

    result = model.proximity(scene_extent=1.0, iteration=31, N=1)

    _assert_aligned(model)
    assert result == (1, 1, 1, 1)
    assert model.rgg_uid[-1].item() == 3
    assert model.rgg_source_uid[-1].item() == 1
    assert model.rgg_target_uid[-1].item() == 2
    assert model.rgg_birth_iter[-1].item() == 31
    assert model.rgg_generation[-1].item() == 6
    assert model.rgg_next_uid == 4


def test_proximity_rgg_source_target_count_mismatch_raises(monkeypatch):
    module = _load_gaussian_model_module(monkeypatch)
    model = _make_model(module, point_count=3, enabled=True)
    model._scaling.data.fill_(torch.log(torch.tensor(2.0)).item())
    dist = torch.tensor([0.0, 6.0, 0.0])
    nearest = torch.tensor([[0, 0], [2, 0], [0, 0]], dtype=torch.long)
    module.distCUDA2 = lambda xyz: (dist, nearest)

    with pytest.raises(ValueError, match="source/target metadata count mismatch"):
        model.proximity(scene_extent=1.0, iteration=31, N=1)


def test_prune_keeps_metadata_aligned_and_uid_not_reused(monkeypatch):
    module = _load_gaussian_model_module(monkeypatch)
    model = _make_model(module, point_count=3, enabled=True)
    grads = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    model.densify_and_clone(grads, grad_threshold=0.5, scene_extent=1.0, iter=10)

    model.prune_points(torch.tensor([True, False, False, True, False]), iter=11)
    model.densify_and_clone(torch.ones((3, 3)), grad_threshold=0.5, scene_extent=1.0, iter=12)

    _assert_aligned(model)
    assert model.rgg_uid.tolist() == [1, 2, 4, 5, 6, 7]
    assert len(set(model.rgg_uid.tolist())) == model.rgg_uid.shape[0]
    assert model.rgg_next_uid == 8


def test_paired_checkpoint_round_trip_restores_rgg_state(monkeypatch, tmp_path):
    module = _load_gaussian_model_module(monkeypatch)
    source = _make_model(module, point_count=3, enabled=True)
    source.densify_and_clone(torch.ones((3, 3)), grad_threshold=0.5, scene_extent=1.0, iter=19)
    path = tmp_path / "paired_chkpnt19.pth"

    save_paired_fork_checkpoint(path, source, 19)
    restored = _make_model(module, point_count=1, enabled=True)
    load_paired_fork_checkpoint(path, restored, _training_args())

    assert torch.equal(source.rgg_uid, restored.rgg_uid)
    assert torch.equal(source.rgg_birth_iter, restored.rgg_birth_iter)
    assert torch.equal(source.rgg_source_uid, restored.rgg_source_uid)
    assert torch.equal(source.rgg_target_uid, restored.rgg_target_uid)
    assert torch.equal(source.rgg_generation, restored.rgg_generation)
    assert restored.rgg_next_uid == source.rgg_next_uid


def test_legacy_paired_checkpoint_fallback_marks_preexisting_population(monkeypatch, tmp_path):
    module = _load_gaussian_model_module(monkeypatch)
    source = _make_model(module, point_count=4, enabled=True)
    path = tmp_path / "legacy_paired_chkpnt.pth"
    save_paired_fork_checkpoint(path, source, 7)
    checkpoint = torch.load(path, weights_only=False)
    checkpoint.pop("rgg_lineage")
    torch.save(checkpoint, path)

    restored = _make_model(module, point_count=1, enabled=True)
    load_paired_fork_checkpoint(path, restored, _training_args())

    assert restored.rgg_uid.tolist() == [0, 1, 2, 3]
    assert restored.rgg_birth_iter.tolist() == [-1, -1, -1, -1]
    assert restored.rgg_source_uid.tolist() == [-1, -1, -1, -1]
    assert restored.rgg_target_uid.tolist() == [-1, -1, -1, -1]
    assert restored.rgg_generation.tolist() == [0, 0, 0, 0]
    assert restored.rgg_next_uid == 4


def test_restore_then_birth_does_not_conflict_with_existing_uid(monkeypatch, tmp_path):
    module = _load_gaussian_model_module(monkeypatch)
    source = _make_model(module, point_count=2, enabled=True)
    source.rgg_uid = torch.tensor([10, 12], dtype=torch.long)
    source.rgg_next_uid = 13
    path = tmp_path / "paired_chkpnt13.pth"
    save_paired_fork_checkpoint(path, source, 13)

    restored = _make_model(module, point_count=1, enabled=True)
    load_paired_fork_checkpoint(path, restored, _training_args())
    restored.densify_and_clone(torch.ones((2, 3)), grad_threshold=0.5, scene_extent=1.0, iter=14)

    assert restored.rgg_uid.tolist() == [10, 12, 13, 14]
    assert restored.rgg_source_uid[-2:].tolist() == [10, 12]
    assert restored.rgg_next_uid == 15


def test_rgg_disabled_does_not_change_clone_growth_or_rng(monkeypatch):
    module = _load_gaussian_model_module(monkeypatch)
    model = _make_model(module, point_count=3, enabled=False)
    grads = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    before_py = random.getstate()
    before_np = np.random.get_state()
    before_torch = torch.get_rng_state()
    candidates = model.densify_and_clone(grads, grad_threshold=0.5, scene_extent=1.0, iter=17)
    after_py = random.getstate()
    after_np = np.random.get_state()
    after_torch = torch.get_rng_state()

    assert candidates == 2
    assert model.get_xyz.shape[0] == 5
    assert model.rgg_uid.numel() == 0
    assert before_py == after_py
    assert all(np.array_equal(left, right) if isinstance(left, np.ndarray) else left == right for left, right in zip(before_np, after_np))
    assert torch.equal(before_torch, after_torch)


def test_rgg_disabled_does_not_change_split_growth_or_rng(monkeypatch):
    module = _load_gaussian_model_module(monkeypatch)
    model = _make_model(module, point_count=3, enabled=False)
    grads = torch.tensor([[1.0], [0.0], [0.0]])
    module.distCUDA2 = lambda xyz: (torch.zeros(xyz.shape[0]), torch.zeros((xyz.shape[0], 1), dtype=torch.long))
    _patch_cpu_split_sampling(monkeypatch, module)

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    before_py = random.getstate()
    before_np = np.random.get_state()
    before_torch = torch.get_rng_state()
    split_stats = model.densify_and_split(grads, grad_threshold=0.5, scene_extent=0.1, iter=23, N=2)
    after_py = random.getstate()
    after_np = np.random.get_state()
    after_torch = torch.get_rng_state()

    assert split_stats == (1, 0, 1)
    assert model.get_xyz.shape[0] == 4
    assert model.rgg_uid.numel() == 0
    assert model.rgg_birth_iter.numel() == 0
    assert model.rgg_source_uid.numel() == 0
    assert model.rgg_target_uid.numel() == 0
    assert model.rgg_generation.numel() == 0
    assert before_py == after_py
    assert all(np.array_equal(left, right) if isinstance(left, np.ndarray) else left == right for left, right in zip(before_np, after_np))
    assert torch.equal(before_torch, after_torch)


def test_rgg_disabled_does_not_change_proximity_growth_or_rng(monkeypatch):
    module = _load_gaussian_model_module(monkeypatch)
    model = _make_model(module, point_count=3, enabled=False)
    model._scaling.data.fill_(torch.log(torch.tensor(2.0)).item())
    dist = torch.tensor([0.0, 6.0, 0.0])
    nearest = torch.tensor([[0], [2], [0]], dtype=torch.long)
    module.distCUDA2 = lambda xyz: (dist, nearest)

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    before_py = random.getstate()
    before_np = np.random.get_state()
    before_torch = torch.get_rng_state()
    result = model.proximity(scene_extent=1.0, iteration=31, N=1)
    after_py = random.getstate()
    after_np = np.random.get_state()
    after_torch = torch.get_rng_state()

    assert result == (1, 1, 1, 1)
    assert model.get_xyz.shape[0] == 4
    assert model.rgg_uid.numel() == 0
    assert model.rgg_birth_iter.numel() == 0
    assert model.rgg_source_uid.numel() == 0
    assert model.rgg_target_uid.numel() == 0
    assert model.rgg_generation.numel() == 0
    assert before_py == after_py
    assert all(np.array_equal(left, right) if isinstance(left, np.ndarray) else left == right for left, right in zip(before_np, after_np))
    assert torch.equal(before_torch, after_torch)

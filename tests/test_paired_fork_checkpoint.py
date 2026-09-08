import importlib.util
import random
import sys
import types
from argparse import Namespace

import numpy as np
import pytest
import torch

from utils.paired_fork_checkpoint import (
    load_paired_fork_checkpoint,
    next_iteration_after_paired_fork,
    restore_rng_state,
    save_paired_fork_checkpoint,
)


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

    spec = importlib.util.spec_from_file_location("_fsgs_gaussian_model_paired_test", "scene/gaussian_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _training_args():
    return Namespace(
        percent_dense=0.01,
        position_lr_init=0.01,
        position_lr_final=0.001,
        position_lr_delay_mult=0.01,
        position_lr_max_steps=1000,
        feature_lr=0.0025,
        opacity_lr=0.05,
        scaling_lr=0.005,
        rotation_lr=0.001,
    )


def _make_model(module, point_count=4):
    model = module.GaussianModel.__new__(module.GaussianModel)
    module.GaussianModel.setup_functions(model)
    model.args = Namespace(train_bg=False, sh_degree=3)
    model.active_sh_degree = 2
    model.max_sh_degree = 3
    model.percent_dense = 0
    model.spatial_lr_scale = 1.0
    model._xyz = torch.nn.Parameter(torch.arange(point_count * 3, dtype=torch.float32).view(point_count, 3) / 10)
    model._features_dc = torch.nn.Parameter(torch.arange(point_count * 3, dtype=torch.float32).view(point_count, 1, 3) / 20)
    model._features_rest = torch.nn.Parameter(torch.arange(point_count * 9, dtype=torch.float32).view(point_count, 3, 3) / 30)
    model._scaling = torch.nn.Parameter(torch.ones((point_count, 3), dtype=torch.float32) * 0.25)
    model._rotation = torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]] * point_count))
    model._opacity = torch.nn.Parameter(torch.ones((point_count, 1), dtype=torch.float32) * -2.0)
    model.max_radii2D = torch.arange(point_count, dtype=torch.float32)
    model.xyz_gradient_accum = torch.arange(point_count, dtype=torch.float32).view(point_count, 1)
    model.denom = torch.ones((point_count, 1), dtype=torch.float32)
    model.optimizer = None
    model.bg_color = torch.empty(0)
    model.confidence = torch.linspace(0.1, 0.1 * point_count, point_count, dtype=torch.float32).view(point_count, 1)
    model.visibility_history = torch.stack(
        (
            torch.arange(point_count) % 2 == 0,
            torch.arange(point_count) % 3 == 1,
            torch.arange(point_count) % 4 >= 2,
        ),
        dim=1,
    )
    model.visible_view_count = model.visibility_history.sum(dim=1).to(dtype=torch.long)
    model.camera_uid_to_train_index = None
    model.training_setup(_training_args())
    return model


def _step_optimizer(model):
    model.optimizer.zero_grad(set_to_none=True)
    loss = (
        model._xyz.square().sum()
        + model._features_dc.square().sum()
        + model._features_rest.square().sum()
        + model._scaling.square().sum()
        + model._rotation.square().sum()
        + model._opacity.square().sum()
    )
    loss.backward()
    model.optimizer.step()


def _assert_tensors_equal(left, right):
    assert torch.equal(left.detach(), right.detach())


def _assert_optimizer_state_equal(left, right):
    left_state = left.optimizer.state_dict()
    right_state = right.optimizer.state_dict()
    assert left_state["param_groups"] == right_state["param_groups"]
    assert left_state["state"].keys() == right_state["state"].keys()
    for key in left_state["state"]:
        assert left_state["state"][key].keys() == right_state["state"][key].keys()
        for state_key, left_value in left_state["state"][key].items():
            right_value = right_state["state"][key][state_key]
            if torch.is_tensor(left_value):
                assert torch.equal(left_value, right_value)
            else:
                assert left_value == right_value


def test_paired_checkpoint_round_trip_restores_gaussian_tensors(monkeypatch, tmp_path):
    module = _load_gaussian_model_module(monkeypatch)
    source = _make_model(module)
    _step_optimizer(source)
    path = tmp_path / "paired_chkpnt897.pth"

    save_paired_fork_checkpoint(path, source, 897)
    restored = _make_model(module)
    load_paired_fork_checkpoint(path, restored, _training_args())

    _assert_tensors_equal(source._xyz, restored._xyz)
    _assert_tensors_equal(source._features_dc, restored._features_dc)
    _assert_tensors_equal(source._features_rest, restored._features_rest)
    _assert_tensors_equal(source._scaling, restored._scaling)
    _assert_tensors_equal(source._rotation, restored._rotation)
    _assert_tensors_equal(source._opacity, restored._opacity)
    _assert_tensors_equal(source.max_radii2D, restored.max_radii2D)
    _assert_tensors_equal(source.xyz_gradient_accum, restored.xyz_gradient_accum)
    _assert_tensors_equal(source.denom, restored.denom)


def test_visibility_state_round_trip(monkeypatch, tmp_path):
    module = _load_gaussian_model_module(monkeypatch)
    source = _make_model(module)
    path = tmp_path / "paired_chkpnt897.pth"

    save_paired_fork_checkpoint(path, source, 897)
    restored = _make_model(module)
    restored.visibility_history = None
    restored.visible_view_count = None
    load_paired_fork_checkpoint(path, restored, _training_args())

    assert torch.equal(source.visibility_history, restored.visibility_history)
    assert torch.equal(source.visible_view_count, restored.visible_view_count)


def test_confidence_state_round_trip(monkeypatch, tmp_path):
    module = _load_gaussian_model_module(monkeypatch)
    source = _make_model(module)
    source.confidence = torch.tensor([[0.1], [0.2], [0.3], [0.4]], dtype=torch.float32)
    path = tmp_path / "paired_chkpnt897.pth"

    save_paired_fork_checkpoint(path, source, 897)
    restored = _make_model(module)
    restored.confidence = torch.ones_like(restored.confidence)
    load_paired_fork_checkpoint(path, restored, _training_args())

    assert torch.equal(source.confidence, restored.confidence)


def test_confidence_restores_when_checkpoint_gaussian_count_differs_from_initial_model(monkeypatch, tmp_path):
    module = _load_gaussian_model_module(monkeypatch)
    source = _make_model(module, point_count=6)
    path = tmp_path / "paired_chkpnt897.pth"

    save_paired_fork_checkpoint(path, source, 897)
    restored = _make_model(module, point_count=4)
    load_paired_fork_checkpoint(path, restored, _training_args())

    assert restored.get_xyz.shape[0] == 6
    assert restored.confidence.shape[0] == 6
    assert torch.equal(source.confidence, restored.confidence)


def test_malformed_confidence_shape_raises_value_error(monkeypatch, tmp_path):
    module = _load_gaussian_model_module(monkeypatch)
    source = _make_model(module)
    path = tmp_path / "paired_chkpnt897.pth"

    save_paired_fork_checkpoint(path, source, 897)
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["confidence"] = torch.ones((source.get_xyz.shape[0] + 1, 1), dtype=torch.float32)
    torch.save(checkpoint, path)

    restored = _make_model(module)
    with pytest.raises(ValueError, match="confidence first dimension"):
        load_paired_fork_checkpoint(path, restored, _training_args())


def test_optimizer_state_round_trip_with_restore_optimizer_true(monkeypatch, tmp_path):
    module = _load_gaussian_model_module(monkeypatch)
    source = _make_model(module)
    _step_optimizer(source)
    path = tmp_path / "paired_chkpnt897.pth"

    save_paired_fork_checkpoint(path, source, 897)
    restored = _make_model(module)
    load_paired_fork_checkpoint(path, restored, _training_args())

    _assert_optimizer_state_equal(source, restored)


def test_python_numpy_and_torch_cpu_rng_round_trip(monkeypatch, tmp_path):
    module = _load_gaussian_model_module(monkeypatch)
    source = _make_model(module)
    path = tmp_path / "paired_chkpnt897.pth"

    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    save_paired_fork_checkpoint(path, source, 897)

    expected_python = random.random()
    expected_numpy = np.random.random()
    expected_torch = torch.rand(3)

    checkpoint = torch.load(path, weights_only=False)
    random.random()
    np.random.random()
    torch.rand(3)
    restore_rng_state(checkpoint)

    assert random.random() == expected_python
    assert np.random.random() == expected_numpy
    assert torch.equal(torch.rand(3), expected_torch)


def test_cuda_rng_round_trip_when_cuda_is_available(monkeypatch, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    module = _load_gaussian_model_module(monkeypatch)
    source = _make_model(module)
    path = tmp_path / "paired_chkpnt897.pth"

    torch.cuda.manual_seed_all(14)
    save_paired_fork_checkpoint(path, source, 897)
    expected = torch.rand(3, device="cuda")

    checkpoint = torch.load(path, weights_only=False)
    torch.rand(3, device="cuda")
    restore_rng_state(checkpoint)

    assert torch.equal(torch.rand(3, device="cuda"), expected)


def test_iteration_resume_starts_after_checkpoint_iteration():
    assert next_iteration_after_paired_fork(897) == 898


def test_legacy_restore_default_does_not_load_optimizer_state(monkeypatch):
    module = _load_gaussian_model_module(monkeypatch)
    source = _make_model(module)
    _step_optimizer(source)

    restored = _make_model(module)
    restored.restore(source.capture(), _training_args())

    assert restored.optimizer.state_dict()["state"] == {}
    assert torch.equal(source._xyz, restored._xyz)

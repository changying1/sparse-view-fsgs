from argparse import ArgumentParser
from types import SimpleNamespace
import importlib.util
import random
import sys
import types

import numpy as np
import pytest
import torch

from arguments import OptimizationParams
from utils.observation_evidence import (
    aggregate_gaussian_observation_evidence,
    aggregate_gaussian_photometric_residual,
    align_mono_depth_affine,
    compute_depth_evidence_map,
    compute_observation_evidence_diagnostics,
    compute_photometric_residual_map,
    format_observation_evidence_diag,
    format_observation_reliability_diag,
    project_gaussians_to_pixels,
    sample_gaussian_evidence_for_view,
)


def _camera(width=3, height=3):
    return SimpleNamespace(
        image_width=width,
        image_height=height,
        full_proj_transform=torch.eye(4),
        world_view_transform=torch.eye(4),
    )


def test_affine_depth_relation_has_zero_residual():
    mono = torch.arange(1, 10, dtype=torch.float32).reshape(3, 3)
    rendered = 2.0 * mono + 3.0

    maps = compute_depth_evidence_map(rendered, mono)

    assert torch.allclose(maps["aligned_depth"], rendered, atol=1e-5)
    assert maps["residual"][maps["valid_mask"]].max() == pytest.approx(0.0, abs=1e-6)
    assert maps["evidence"][maps["valid_mask"]].min() == pytest.approx(1.0, abs=1e-6)


def test_local_corrupted_region_gets_lower_evidence():
    mono = torch.arange(1, 17, dtype=torch.float32).reshape(4, 4)
    rendered = 2.0 * mono + 1.0
    rendered[0, 0] += 20.0

    maps = compute_depth_evidence_map(rendered, mono)

    assert maps["evidence"][0, 0] < maps["evidence"][2, 2]
    assert ((maps["evidence"][maps["valid_mask"]] >= 0.0) & (maps["evidence"][maps["valid_mask"]] <= 1.0)).all()


def test_invalid_depth_pixels_are_excluded():
    mono = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    rendered = torch.tensor([[1.0, float("nan")], [3.0, 2.0]])
    valid_mask = torch.tensor([[True, True], [True, False]])

    maps = compute_depth_evidence_map(rendered, mono, valid_mask=valid_mask)

    assert maps["valid_mask"].tolist() == [[True, False], [True, False]]
    assert torch.isnan(maps["evidence"][0, 1])
    assert torch.isnan(maps["evidence"][1, 1])


def test_zero_or_constant_depth_does_not_crash():
    mono = torch.ones((3, 3))
    rendered = torch.ones((3, 3))
    aligned, valid, a, b = align_mono_depth_affine(rendered, mono)
    maps = compute_depth_evidence_map(torch.zeros((3, 3)), mono)

    assert not valid.any()
    assert torch.isnan(aligned).all()
    assert torch.isnan(a)
    assert torch.isnan(b)
    assert not maps["valid_mask"].any()


def test_constant_mono_depth_invalidates_entire_view():
    mono = torch.ones((3, 3))
    rendered = torch.arange(1, 10, dtype=torch.float32).reshape(3, 3)

    maps = compute_depth_evidence_map(rendered, mono)

    assert not maps["valid_mask"].any()
    assert torch.isnan(maps["evidence"]).all()
    assert torch.isnan(maps["residual"]).all()


def test_fewer_than_two_valid_pixels_invalidates_entire_view():
    mono = torch.arange(1, 10, dtype=torch.float32).reshape(3, 3)
    rendered = 2.0 * mono + 3.0
    valid_mask = torch.zeros((3, 3), dtype=torch.bool)
    valid_mask[1, 1] = True

    maps = compute_depth_evidence_map(rendered, mono, valid_mask=valid_mask)

    assert not maps["valid_mask"].any()
    assert torch.isnan(maps["evidence"]).all()
    assert torch.isnan(maps["residual"]).all()


def test_invisible_gaussian_receives_no_evidence():
    xyz = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    evidence = torch.ones((3, 3))
    residual = torch.zeros((3, 3))
    valid_depth = torch.ones((3, 3), dtype=torch.bool)
    visible = torch.tensor([True, False])

    sampled_e, sampled_r, valid = sample_gaussian_evidence_for_view(
        xyz,
        _camera(),
        evidence,
        residual,
        valid_depth,
        visible,
    )

    assert valid.tolist() == [True, False]
    assert sampled_e[0].item() == pytest.approx(1.0)
    assert torch.isnan(sampled_e[1])
    assert torch.isnan(sampled_r[1])


def test_visible_gaussian_samples_projected_high_consistency_region():
    xyz = torch.tensor([[-1.0, 1.0, 1.0], [1.0, -1.0, 1.0]])
    evidence = torch.tensor([[0.9, 0.2, 0.1], [0.2, 0.2, 0.2], [0.1, 0.2, 0.3]])
    residual = 1.0 - evidence
    valid_depth = torch.ones((3, 3), dtype=torch.bool)
    visible = torch.tensor([True, True])

    px, py, projected = project_gaussians_to_pixels(xyz, _camera())
    sampled_e, _, valid = sample_gaussian_evidence_for_view(
        xyz,
        _camera(),
        evidence,
        residual,
        valid_depth,
        visible,
    )

    assert projected.all()
    assert (px.tolist(), py.tolist()) == ([0, 2], [0, 2])
    assert valid.all()
    assert sampled_e[0] > sampled_e[1]


def test_multiview_gaussian_evidence_aggregation_keeps_zero_view_invalid():
    xyz = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    cam = _camera()
    rendered_a = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 2.0], [1.0, 1.0, 1.0]])
    mono_a = rendered_a.clone()
    rendered_b = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 4.0], [1.0, 1.0, 1.0]])
    mono_b = rendered_b.clone()
    visibility = [torch.tensor([True, True, False]), torch.tensor([True, False, False])]

    gaussian_e, _, view_count, _ = aggregate_gaussian_observation_evidence(
        xyz,
        [cam, cam],
        [rendered_a, rendered_b],
        [mono_a, mono_b],
        visibility,
    )

    assert view_count.tolist() == [2, 1, 0]
    assert gaussian_e[0].item() == pytest.approx(1.0)
    assert gaussian_e[1].item() == pytest.approx(1.0)
    assert torch.isnan(gaussian_e[2])


def test_perfect_rendered_rgb_has_zero_photo_residual():
    rendered = torch.tensor(
        [
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.2, 0.3], [0.4, 0.5]],
            [[0.3, 0.4], [0.5, 0.6]],
        ]
    )
    photo, valid = compute_photometric_residual_map(rendered, rendered.clone())

    assert valid.all()
    assert photo[valid].max().item() == pytest.approx(0.0)


def test_corrupted_rgb_pixel_produces_larger_photo_residual():
    rendered = torch.zeros((3, 3, 3))
    gt = rendered.clone()
    rendered[:, 1, 1] = torch.tensor([0.3, 0.6, 0.9])

    photo, valid = compute_photometric_residual_map(rendered, gt)

    assert valid.all()
    assert photo[1, 1].item() == pytest.approx(0.6)
    assert photo[1, 1] > photo[0, 0]


def test_invisible_gaussian_receives_no_photo_evidence():
    xyz = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    cam = _camera()
    cam.original_image = torch.zeros((3, 3, 3))
    rendered = torch.ones((3, 3, 3))

    gaussian_photo, photo_count = aggregate_gaussian_photometric_residual(
        xyz,
        [cam],
        [rendered],
        [torch.tensor([True, False])],
    )

    assert photo_count.tolist() == [1, 0]
    assert gaussian_photo[0].item() == pytest.approx(1.0)
    assert torch.isnan(gaussian_photo[1])


def test_multiview_photo_residual_aggregation_keeps_zero_view_nan():
    xyz = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    cam_a = _camera()
    cam_b = _camera()
    cam_a.original_image = torch.zeros((3, 3, 3))
    cam_b.original_image = torch.zeros((3, 3, 3))
    rendered_a = torch.full((3, 3, 3), 0.25)
    rendered_b = torch.full((3, 3, 3), 0.75)
    visibility = [torch.tensor([True, True, False]), torch.tensor([True, False, False])]

    gaussian_photo, photo_count = aggregate_gaussian_photometric_residual(
        xyz,
        [cam_a, cam_b],
        [rendered_a, rendered_b],
        visibility,
    )

    assert photo_count.tolist() == [2, 1, 0]
    assert gaussian_photo[0].item() == pytest.approx(0.5)
    assert gaussian_photo[1].item() == pytest.approx(0.25)
    assert torch.isnan(gaussian_photo[2])


def test_three_view_diagnostics_uses_all_views_and_means_valid_observations(monkeypatch):
    import utils.observation_evidence as module

    xyz = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    cameras = [_camera(), _camera(), _camera()]
    rendered_depths = [torch.full((3, 3), value) for value in (0.0, 1.0, 2.0)]
    mono_depths = [torch.ones((3, 3)) for _ in cameras]
    visibility = [
        torch.tensor([True, True, False]),
        torch.tensor([True, False, False]),
        torch.tensor([True, True, False]),
    ]
    evidence_by_view = [0.0, 0.3, 0.9]

    def fake_depth_map(rendered_depth, mono_depth, valid_mask=None):
        view_index = int(rendered_depth[0, 0].item())
        return {
            "aligned_depth": torch.ones((3, 3)),
            "valid_mask": torch.ones((3, 3), dtype=torch.bool),
            "residual": torch.full((3, 3), 1.0 - evidence_by_view[view_index]),
            "evidence": torch.full((3, 3), evidence_by_view[view_index]),
            "affine_a": torch.tensor(1.0),
            "affine_b": torch.tensor(0.0),
        }

    monkeypatch.setattr(module, "compute_depth_evidence_map", fake_depth_map)
    monkeypatch.setattr(module, "build_knn_graph", lambda xyz, k=12: torch.tensor([[1], [0], [1]], dtype=torch.long))
    monkeypatch.setattr(module, "estimate_gaussian_normals", lambda scales, rotations: torch.tensor([[0.0, 0.0, 1.0]] * 3))
    monkeypatch.setattr(module, "compute_geometric_turning", lambda xyz, normals, neighbors: torch.ones(3))
    monkeypatch.setattr(module, "compute_continuity_defect", lambda xyz, normals, neighbors: torch.ones(3))

    stats, gaussian_e, _, view_count = compute_observation_evidence_diagnostics(
        xyz,
        torch.ones((3, 3)),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3),
        cameras,
        rendered_depths,
        mono_depths,
        visibility,
    )

    assert stats["views"] == 3
    assert view_count.tolist() == [3, 2, 0]
    assert gaussian_e[0].item() == pytest.approx((0.0 + 0.3 + 0.9) / 3.0)
    assert gaussian_e[1].item() == pytest.approx((0.0 + 0.9) / 2.0)
    assert torch.isnan(gaussian_e[2])


def test_d_quartile_statistics_use_only_valid_evidence_gaussians(monkeypatch):
    import utils.observation_evidence as module

    xyz = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    scales = torch.ones((5, 3))
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 5)
    gaussian_e = torch.tensor([0.1, 0.2, float("nan"), 0.8, 0.9])
    gaussian_r = torch.tensor([0.9, 0.8, float("nan"), 0.2, 0.1])
    view_count = torch.tensor([1, 1, 0, 1, 1])
    monkeypatch.setattr(
        module,
        "aggregate_gaussian_observation_evidence",
        lambda *args, **kwargs: (gaussian_e, gaussian_r, view_count, [gaussian_r[torch.isfinite(gaussian_r)]]),
    )
    monkeypatch.setattr(module, "build_knn_graph", lambda xyz, k=12: torch.tensor([[1], [0], [1], [4], [3]], dtype=torch.long))
    monkeypatch.setattr(module, "estimate_gaussian_normals", lambda scales, rotations: torch.tensor([[0.0, 0.0, 1.0]] * 5))
    monkeypatch.setattr(module, "compute_geometric_turning", lambda xyz, normals, neighbors: torch.tensor([1.0, 2.0, 99.0, 3.0, 4.0]))
    monkeypatch.setattr(module, "compute_continuity_defect", lambda xyz, normals, neighbors: torch.tensor([4.0, 3.0, 99.0, 2.0, 1.0]))

    stats, _, _, counts = compute_observation_evidence_diagnostics(
        xyz,
        scales,
        rotations,
        [_camera()],
        [torch.ones((3, 3))],
        [torch.ones((3, 3))],
        [torch.ones(5, dtype=torch.bool)],
    )

    assert stats["valid_evidence_gaussians"] == 4
    assert stats["valid_evidence_ratio"] == pytest.approx(0.8)
    assert stats["D_Q1_E_mean"] == pytest.approx(0.9)
    assert stats["D_Q4_E_mean"] == pytest.approx(0.1)
    assert stats["E_low25_D_mean"] == pytest.approx(4.0)
    assert stats["E_high25_D_mean"] == pytest.approx(1.0)
    assert counts.tolist() == [1, 1, 0, 1, 1]
    assert "[ObservationEvidenceDiag]" in format_observation_evidence_diag(1200, stats)


def test_low_e_larger_photo_error_produces_negative_corr(monkeypatch):
    import utils.observation_evidence as module

    xyz = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    scales = torch.ones((6, 3))
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 6)
    gaussian_e = torch.tensor([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
    gaussian_r = 1.0 - gaussian_e
    view_count = torch.ones(6, dtype=torch.long)
    gaussian_photo = torch.tensor([1.0, 0.9, 0.8, 0.3, 0.2, 0.1])
    photo_count = torch.ones(6, dtype=torch.long)
    monkeypatch.setattr(
        module,
        "aggregate_gaussian_observation_evidence",
        lambda *args, **kwargs: (gaussian_e, gaussian_r, view_count, [gaussian_r]),
    )
    monkeypatch.setattr(
        module,
        "aggregate_gaussian_photometric_residual",
        lambda *args, **kwargs: (gaussian_photo, photo_count),
    )
    monkeypatch.setattr(module, "build_knn_graph", lambda xyz, k=12: torch.tensor([[1], [0], [1], [2], [3], [4]], dtype=torch.long))
    monkeypatch.setattr(module, "estimate_gaussian_normals", lambda scales, rotations: torch.tensor([[0.0, 0.0, 1.0]] * 6))
    monkeypatch.setattr(module, "compute_geometric_turning", lambda xyz, normals, neighbors: torch.ones(6))
    monkeypatch.setattr(module, "compute_continuity_defect", lambda xyz, normals, neighbors: torch.arange(6, dtype=torch.float32))

    stats, _, _, _ = compute_observation_evidence_diagnostics(
        xyz,
        scales,
        rotations,
        [_camera()],
        [torch.ones((3, 3))],
        [torch.ones((3, 3))],
        [torch.ones(6, dtype=torch.bool)],
        rendered_rgbs=[torch.zeros((3, 3, 3))],
    )

    assert stats["valid_joint"] == 6
    assert stats["corr_E_photo"] < 0.0
    assert stats["E_low25_photo_mean"] > stats["E_high25_photo_mean"]
    assert "[ObservationReliabilityDiag]" in format_observation_reliability_diag(1200, stats)


def test_highd_conditional_split_uses_subset_e_quartiles(monkeypatch):
    import utils.observation_evidence as module

    xyz = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    scales = torch.ones((8, 3))
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 8)
    gaussian_e = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.1, 0.2, 0.3, 0.4])
    gaussian_r = 1.0 - gaussian_e
    gaussian_photo = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.9, 0.7, 0.3, 0.2])
    view_count = torch.ones(8, dtype=torch.long)
    photo_count = torch.ones(8, dtype=torch.long)
    monkeypatch.setattr(
        module,
        "aggregate_gaussian_observation_evidence",
        lambda *args, **kwargs: (gaussian_e, gaussian_r, view_count, [gaussian_r]),
    )
    monkeypatch.setattr(
        module,
        "aggregate_gaussian_photometric_residual",
        lambda *args, **kwargs: (gaussian_photo, photo_count),
    )
    monkeypatch.setattr(module, "build_knn_graph", lambda xyz, k=12: torch.tensor([[1], [0], [1], [2], [3], [4], [5], [6]], dtype=torch.long))
    monkeypatch.setattr(module, "estimate_gaussian_normals", lambda scales, rotations: torch.tensor([[0.0, 0.0, 1.0]] * 8))
    monkeypatch.setattr(module, "compute_geometric_turning", lambda xyz, normals, neighbors: torch.ones(8))
    monkeypatch.setattr(module, "compute_continuity_defect", lambda xyz, normals, neighbors: torch.arange(8, dtype=torch.float32))

    stats, _, _, _ = compute_observation_evidence_diagnostics(
        xyz,
        scales,
        rotations,
        [_camera()],
        [torch.ones((3, 3))],
        [torch.ones((3, 3))],
        [torch.ones(8, dtype=torch.bool)],
        rendered_rgbs=[torch.zeros((3, 3, 3))],
    )

    assert stats["HighD_valid"] == 2
    assert stats["HighD_lowE_photo"] == pytest.approx(0.3)
    assert stats["HighD_highE_photo"] == pytest.approx(0.2)
    assert stats["HighD_E_mean"] == pytest.approx(0.35)


def test_highk_conditional_split_uses_subset_e_quartiles(monkeypatch):
    import utils.observation_evidence as module

    xyz = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    scales = torch.ones((8, 3))
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 8)
    gaussian_e = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.9, 0.8, 0.7, 0.6])
    gaussian_r = 1.0 - gaussian_e
    gaussian_photo = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.2, 0.4, 0.8, 1.0])
    counts = torch.ones(8, dtype=torch.long)
    monkeypatch.setattr(module, "aggregate_gaussian_observation_evidence", lambda *args, **kwargs: (gaussian_e, gaussian_r, counts, [gaussian_r]))
    monkeypatch.setattr(module, "aggregate_gaussian_photometric_residual", lambda *args, **kwargs: (gaussian_photo, counts))
    monkeypatch.setattr(module, "build_knn_graph", lambda xyz, k=12: torch.tensor([[1], [0], [1], [2], [3], [4], [5], [6]], dtype=torch.long))
    monkeypatch.setattr(module, "estimate_gaussian_normals", lambda scales, rotations: torch.tensor([[0.0, 0.0, 1.0]] * 8))
    monkeypatch.setattr(module, "compute_geometric_turning", lambda xyz, normals, neighbors: torch.arange(8, dtype=torch.float32))
    monkeypatch.setattr(module, "compute_continuity_defect", lambda xyz, normals, neighbors: torch.ones(8))

    stats, _, _, _ = compute_observation_evidence_diagnostics(
        xyz, scales, rotations, [_camera()], [torch.ones((3, 3))], [torch.ones((3, 3))],
        [torch.ones(8, dtype=torch.bool)], rendered_rgbs=[torch.zeros((3, 3, 3))]
    )

    assert stats["HighK_valid"] == 2
    assert stats["HighK_lowE_photo"] == pytest.approx(1.0)
    assert stats["HighK_highE_photo"] == pytest.approx(0.8)


def test_highu_conditional_split_uses_subset_e_quartiles(monkeypatch):
    import utils.observation_evidence as module

    xyz = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    scales = torch.ones((8, 3))
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 8)
    gaussian_e = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])
    gaussian_r = 1.0 - gaussian_e
    gaussian_photo = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.7, 0.9])
    counts = torch.ones(8, dtype=torch.long)
    monkeypatch.setattr(module, "aggregate_gaussian_observation_evidence", lambda *args, **kwargs: (gaussian_e, gaussian_r, counts, [gaussian_r]))
    monkeypatch.setattr(module, "aggregate_gaussian_photometric_residual", lambda *args, **kwargs: (gaussian_photo, counts))
    monkeypatch.setattr(module, "build_knn_graph", lambda xyz, k=12: torch.tensor([[1], [0], [1], [2], [3], [4], [5], [6]], dtype=torch.long))
    monkeypatch.setattr(module, "estimate_gaussian_normals", lambda scales, rotations: torch.tensor([[0.0, 0.0, 1.0]] * 8))
    monkeypatch.setattr(module, "compute_geometric_turning", lambda xyz, normals, neighbors: torch.arange(8, dtype=torch.float32))
    monkeypatch.setattr(module, "compute_continuity_defect", lambda xyz, normals, neighbors: torch.arange(8, dtype=torch.float32))

    stats, _, _, _ = compute_observation_evidence_diagnostics(
        xyz, scales, rotations, [_camera()], [torch.ones((3, 3))], [torch.ones((3, 3))],
        [torch.ones(8, dtype=torch.bool)], rendered_rgbs=[torch.zeros((3, 3, 3))]
    )

    assert stats["HighU_valid"] == 2
    assert stats["HighU_lowE_photo"] == pytest.approx(0.9)
    assert stats["HighU_highE_photo"] == pytest.approx(0.7)


def test_observation_evidence_flag_defaults_false():
    parser = ArgumentParser()
    params = OptimizationParams(parser)
    parsed = parser.parse_args([])
    extracted = params.extract(parsed)

    assert extracted.enable_observation_evidence_diagnostics is False


def _load_train_module_with_stubs(monkeypatch):
    gaussian_renderer = types.ModuleType("gaussian_renderer")
    gaussian_renderer.render = lambda *args, **kwargs: None
    gaussian_renderer.network_gui = SimpleNamespace(conn=None)
    monkeypatch.setitem(sys.modules, "gaussian_renderer", gaussian_renderer)

    scene = types.ModuleType("scene")
    scene.Scene = object
    scene.GaussianModel = object
    monkeypatch.setitem(sys.modules, "scene", scene)

    lpips_module = types.ModuleType("lpipsPyTorch")
    lpips_module.lpips = lambda *args, **kwargs: torch.tensor(0.0)
    monkeypatch.setitem(sys.modules, "lpipsPyTorch", lpips_module)

    depth_utils = types.ModuleType("utils.depth_utils")
    depth_utils.estimate_depth = lambda *args, **kwargs: torch.ones((2, 2))
    monkeypatch.setitem(sys.modules, "utils.depth_utils", depth_utils)

    torchmetrics = types.ModuleType("torchmetrics")
    torchmetrics.PearsonCorrCoef = object
    torchmetrics_functional = types.ModuleType("torchmetrics.functional")
    torchmetrics_regression = types.ModuleType("torchmetrics.functional.regression")
    torchmetrics_regression.pearson_corrcoef = lambda *args, **kwargs: torch.tensor(0.0)
    monkeypatch.setitem(sys.modules, "torchmetrics", torchmetrics)
    monkeypatch.setitem(sys.modules, "torchmetrics.functional", torchmetrics_functional)
    monkeypatch.setitem(sys.modules, "torchmetrics.functional.regression", torchmetrics_regression)

    spec = importlib.util.spec_from_file_location("_fsgs_train_observation_test", "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _numpy_rng_equal(left, right):
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def test_observation_snapshot_collection_uses_all_cameras_and_does_not_mutate_state(monkeypatch):
    train_module = _load_train_module_with_stubs(monkeypatch)
    gaussians = SimpleNamespace(
        _xyz=torch.arange(9, dtype=torch.float32).reshape(3, 3),
        _scaling=torch.ones((3, 3)),
        _rotation=torch.zeros((3, 4)),
        visibility_history=torch.tensor([[True, False, False], [False, True, False], [False, False, True]]),
        xyz_gradient_accum=torch.ones((3, 1)),
        denom=torch.arange(3, dtype=torch.float32).reshape(3, 1),
        max_radii2D=torch.arange(3, dtype=torch.float32),
    )
    gaussians.get_xyz = gaussians._xyz
    cameras = [
        SimpleNamespace(uid=index, depth_image=torch.full((2, 2), float(index + 1)), mask=torch.ones((2, 2), dtype=torch.bool))
        for index in range(3)
    ]
    gaussian_state = {
        name: value.clone()
        for name, value in vars(gaussians).items()
        if torch.is_tensor(value)
    }
    camera_state = [(camera.depth_image.clone(), camera.mask.clone()) for camera in cameras]
    before_py_rng = random.getstate()
    before_np_rng = np.random.get_state()
    before_torch_rng = torch.get_rng_state()
    calls = []

    def fake_render(camera, model, pipe, background):
        assert not torch.is_grad_enabled()
        calls.append(camera.uid)
        random.random()
        np.random.rand()
        torch.rand(())
        depth = torch.full((1, 2, 2), float(camera.uid + 10))
        return {
            "depth": depth,
            "render": torch.full((3, 2, 2), float(camera.uid + 20)),
            "visibility_filter": torch.tensor([camera.uid == 0, camera.uid != 1, camera.uid == 2]),
        }

    with torch.enable_grad():
        rendered_depths, mono_depths, visibility_masks, valid_masks, rendered_rgbs = train_module.collect_observation_evidence_snapshot(
            gaussians,
            cameras,
            pipe=object(),
            background=torch.zeros(3),
            render_func=fake_render,
        )

    assert calls == [0, 1, 2]
    assert [depth[0, 0].item() for depth in rendered_depths] == [10.0, 11.0, 12.0]
    assert [rgb[0, 0, 0].item() for rgb in rendered_rgbs] == [20.0, 21.0, 22.0]
    assert [mono[0, 0].item() for mono in mono_depths] == [1.0, 2.0, 3.0]
    assert [mask.tolist() for mask in visibility_masks] == [[True, True, False], [False, False, False], [False, True, True]]
    assert all(mask.all().item() for mask in valid_masks)
    assert random.getstate() == before_py_rng
    assert _numpy_rng_equal(np.random.get_state(), before_np_rng)
    assert torch.equal(torch.get_rng_state(), before_torch_rng)
    for name, value in gaussian_state.items():
        assert torch.equal(getattr(gaussians, name), value)
    for camera, (depth_image, mask) in zip(cameras, camera_state):
        assert torch.equal(camera.depth_image, depth_image)
        assert torch.equal(camera.mask, mask)

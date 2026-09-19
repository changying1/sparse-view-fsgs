import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))

from test_rgg_lineage import (
    _assert_aligned,
    _load_gaussian_model_module,
    _make_model,
    _patch_cpu_split_sampling,
    _training_args,
)


def test_stable_uid_unique_for_legacy_population(monkeypatch):
    module = _load_gaussian_model_module(monkeypatch)
    model = _make_model(module, point_count=5, enabled=False)

    assert model.rgg_uid.dtype == torch.long
    assert model.rgg_uid.tolist() == [0, 1, 2, 3, 4]
    assert torch.unique(model.rgg_uid).numel() == 5
    assert model.rgg_birth_iter.tolist() == [0, 0, 0, 0, 0]
    assert model.rgg_birth_type.tolist() == [module.RGG_BIRTH_TYPE_LEGACY] * 5
    assert torch.equal(model.rgg_origin_type, model.rgg_birth_type)
    assert model.rgg_source_uid.tolist() == [-1] * 5
    assert model.rgg_target_uid.tolist() == [-1] * 5


def test_new_gaussian_uid_increases_monotonically(monkeypatch):
    module = _load_gaussian_model_module(monkeypatch)
    model = _make_model(module, point_count=3, enabled=False)

    model.densify_and_clone(
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        grad_threshold=0.5,
        scene_extent=1.0,
        iter=17,
    )

    _assert_aligned(model)
    assert model.rgg_uid.tolist() == [0, 1, 2, 3, 4]
    assert model.rgg_next_uid == 5
    assert model.rgg_birth_iter[-2:].tolist() == [17, 17]
    assert model.rgg_birth_type[-2:].tolist() == [module.RGG_BIRTH_TYPE_CLONE] * 2


def test_prune_keeps_lineage_metadata_lengths_consistent(monkeypatch):
    module = _load_gaussian_model_module(monkeypatch)
    model = _make_model(module, point_count=4, enabled=False)

    model.prune_points(torch.tensor([True, False, True, False]), iter=1)

    _assert_aligned(model)
    assert model.rgg_uid.tolist() == [1, 3]
    assert model.rgg_next_uid == 4


def test_clone_and_split_record_direct_source_relationship(monkeypatch):
    module = _load_gaussian_model_module(monkeypatch)
    clone_model = _make_model(module, point_count=3, enabled=False)
    clone_model.densify_and_clone(torch.ones((3, 3)), grad_threshold=0.5, scene_extent=1.0, iter=9)

    assert clone_model.rgg_source_uid[-3:].tolist() == [0, 1, 2]
    assert clone_model.rgg_target_uid[-3:].tolist() == [-1, -1, -1]
    assert clone_model.rgg_birth_type[-3:].tolist() == [module.RGG_BIRTH_TYPE_CLONE] * 3
    assert clone_model.rgg_lineage_registry == {0: [3], 1: [4], 2: [5]}

    split_model = _make_model(module, point_count=3, enabled=False)
    module.distCUDA2 = lambda xyz: (torch.zeros(xyz.shape[0]), torch.zeros((xyz.shape[0], 1), dtype=torch.long))
    _patch_cpu_split_sampling(monkeypatch, module)
    split_model.densify_and_split(
        torch.tensor([[1.0], [0.0], [0.0]]),
        grad_threshold=0.5,
        scene_extent=0.1,
        iter=10,
        N=2,
    )

    assert split_model.rgg_uid.tolist() == [1, 2, 3, 4]
    assert split_model.rgg_source_uid[-2:].tolist() == [0, 0]
    assert split_model.rgg_birth_type[-2:].tolist() == [module.RGG_BIRTH_TYPE_SPLIT] * 2
    assert split_model.rgg_lineage_registry == {0: [3, 4]}


def test_proximity_records_source_target_and_registry(monkeypatch):
    module = _load_gaussian_model_module(monkeypatch)
    model = _make_model(module, point_count=3, enabled=False)
    model._scaling.data.fill_(torch.log(torch.tensor(2.0)).item())
    module.distCUDA2 = lambda xyz: (
        torch.tensor([0.0, 6.0, 0.0]),
        torch.tensor([[0], [2], [0]], dtype=torch.long),
    )

    model.proximity(scene_extent=1.0, iteration=31, N=1)

    assert model.rgg_uid[-1].item() == 3
    assert model.rgg_source_uid[-1].item() == 1
    assert model.rgg_target_uid[-1].item() == 2
    assert model.rgg_birth_type[-1].item() == module.RGG_BIRTH_TYPE_PROXIMITY_CHILD
    assert model.rgg_lineage_registry == {1: [3]}


def test_legacy_checkpoint_migration_initializes_lineage(monkeypatch):
    module = _load_gaussian_model_module(monkeypatch)
    source = _make_model(module, point_count=4, enabled=False)
    legacy_capture = source.capture()[:-1]

    restored = _make_model(module, point_count=1, enabled=False)
    restored.restore(legacy_capture, _training_args())

    assert restored.rgg_uid.tolist() == [0, 1, 2, 3]
    assert restored.rgg_birth_iter.tolist() == [-1, -1, -1, -1]
    assert restored.rgg_source_uid.tolist() == [-1, -1, -1, -1]
    assert restored.rgg_target_uid.tolist() == [-1, -1, -1, -1]
    assert restored.rgg_birth_type.tolist() == [module.RGG_BIRTH_TYPE_LEGACY] * 4
    assert restored.rgg_next_uid == 4

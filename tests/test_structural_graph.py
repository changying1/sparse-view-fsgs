import torch

from utils.structural_graph import (
    build_knn_graph,
    compute_continuity_defect,
    compute_geometric_turning,
    compute_redundancy,
    estimate_gaussian_normals,
)


def test_knn_shape_and_no_self_neighbor():
    xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    neighbors = build_knn_graph(xyz, k=2)

    assert neighbors.shape == (3, 2)
    for index in range(3):
        assert index not in neighbors[index].tolist()


def test_normals_are_finite_unit_vectors():
    scales = torch.tensor([[1.0, 1.0, 0.1], [1.0, 0.1, 1.0]])
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    normals = estimate_gaussian_normals(scales, rotations)

    assert normals.shape == (2, 3)
    assert torch.isfinite(normals).all()
    assert torch.allclose(torch.linalg.norm(normals, dim=-1), torch.ones(2))


def test_turning_is_low_for_plane_and_higher_for_corner():
    xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    neighbors = torch.tensor([[1, 2], [0, 2], [0, 1]], dtype=torch.long)
    plane_normals = torch.tensor([[0.0, 0.0, 1.0]] * 3)
    corner_normals = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

    assert compute_geometric_turning(xyz, corner_normals, neighbors).mean() > compute_geometric_turning(xyz, plane_normals, neighbors).mean()


def test_defect_and_redundancy_are_non_negative_finite():
    xyz = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [1.0, 0.0, 0.0]])
    neighbors = torch.tensor([[1, 2], [0, 2], [0, 1]], dtype=torch.long)
    normals = torch.tensor([[0.0, 0.0, 1.0]] * 3)
    scales = torch.ones((3, 3)) * 0.2

    defect = compute_continuity_defect(xyz, normals, neighbors)
    redundancy = compute_redundancy(xyz, scales, neighbors)

    assert torch.isfinite(defect).all()
    assert torch.isfinite(redundancy).all()
    assert (defect >= 0).all()
    assert (redundancy >= 0).all()

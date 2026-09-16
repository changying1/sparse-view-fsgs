import torch

from utils.structural_graph import (
    build_knn_graph,
    compute_continuity_defect,
    compute_geometric_turning,
    compute_good_continuation_edge_scores,
    compute_good_continuation_score,
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


def test_good_continuation_parallel_normals_tangent_connection_is_one():
    xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    normals = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    neighbors = torch.tensor([[1], [0]], dtype=torch.long)

    score = compute_good_continuation_score(xyz, normals, neighbors)

    assert torch.allclose(score, torch.ones(2), atol=1e-6)


def test_good_continuation_edge_score_shape():
    xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    normals = torch.tensor([[0.0, 0.0, 1.0]] * 3)
    neighbors = torch.tensor([[1, 2], [0, 2], [0, 1]], dtype=torch.long)

    edge = compute_good_continuation_edge_scores(xyz, normals, neighbors)

    assert edge.shape == (3, 2)


def test_good_continuation_edge_mean_matches_source_score():
    xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    normals = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    neighbors = torch.tensor([[1, 2], [0, 2], [0, 1]], dtype=torch.long)

    edge = compute_good_continuation_edge_scores(xyz, normals, neighbors)
    source = compute_good_continuation_score(xyz, normals, neighbors)

    assert torch.allclose(source, edge.mean(dim=-1))


def test_good_continuation_perpendicular_normals_drops():
    xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    normals = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    neighbors = torch.tensor([[1], [0]], dtype=torch.long)

    score = compute_good_continuation_score(xyz, normals, neighbors)

    assert torch.equal(score, torch.zeros(2))


def test_good_continuation_source_normal_connection_is_zero():
    xyz = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    normals = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    neighbors = torch.tensor([[1], [0]], dtype=torch.long)

    score = compute_good_continuation_score(xyz, normals, neighbors)

    assert torch.allclose(score, torch.zeros(2), atol=1e-6)


def test_good_continuation_normal_sign_flip_matches_same_direction():
    xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    same = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    flipped = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    neighbors = torch.tensor([[1], [0]], dtype=torch.long)

    assert torch.allclose(
        compute_good_continuation_score(xyz, same, neighbors),
        compute_good_continuation_score(xyz, flipped, neighbors),
    )


def test_good_continuation_k_zero_returns_zero():
    xyz = torch.zeros((3, 3))
    normals = torch.tensor([[0.0, 0.0, 1.0]] * 3)
    neighbors = torch.empty((3, 0), dtype=torch.long)

    edge = compute_good_continuation_edge_scores(xyz, normals, neighbors)

    assert edge.shape == (3, 0)
    assert torch.equal(compute_good_continuation_score(xyz, normals, neighbors), torch.zeros(3))


def test_good_continuation_edge_formula_sanity():
    xyz_tangent = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    xyz_normal = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    parallel_normals = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    perpendicular_normals = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    neighbors = torch.tensor([[1], [0]], dtype=torch.long)

    assert torch.allclose(
        compute_good_continuation_edge_scores(xyz_tangent, parallel_normals, neighbors),
        torch.ones((2, 1)),
        atol=1e-6,
    )
    assert torch.allclose(
        compute_good_continuation_edge_scores(xyz_normal, parallel_normals, neighbors),
        torch.zeros((2, 1)),
        atol=1e-6,
    )
    assert torch.equal(
        compute_good_continuation_edge_scores(xyz_tangent, perpendicular_normals, neighbors),
        torch.zeros((2, 1)),
    )


def test_good_continuation_nan_inf_protection():
    xyz = torch.tensor([[0.0, 0.0, 0.0], [float("inf"), 0.0, 0.0], [float("nan"), 0.0, 0.0]])
    normals = torch.tensor([[0.0, 0.0, 1.0], [float("nan"), 0.0, 1.0], [float("inf"), 0.0, 1.0]])
    neighbors = torch.tensor([[1, 2], [0, 2], [0, 1]], dtype=torch.long)

    score = compute_good_continuation_score(xyz, normals, neighbors)

    assert torch.isfinite(score).all()
    assert ((score >= 0.0) & (score <= 1.0)).all()

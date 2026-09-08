import torch

from utils.edge_support import (
    aggregate_multiview_edge_support,
    camera_from_projection,
    compute_edge_map,
    project_gaussians_to_camera,
    sample_edge_support,
)


def test_edge_map_responds_to_clear_edge_more_than_flat_region():
    image = torch.zeros((3, 8, 8), dtype=torch.float32)
    image[:, :, 4:] = 1.0
    edge = compute_edge_map(image)

    assert edge[:, 3:5].mean() > edge[:, :2].mean()


def test_projection_and_sampling_are_valid():
    xyz = torch.tensor([[0.0, 0.0, 1.0], [10.0, 10.0, 1.0]], dtype=torch.float32)
    projection = torch.eye(4)
    camera = camera_from_projection(projection, image_width=8, image_height=8)
    edge = torch.ones((8, 8), dtype=torch.float32)

    pixels, valid = project_gaussians_to_camera(xyz, camera=camera)
    support = sample_edge_support(edge, pixels, valid)

    assert valid.tolist() == [True, False]
    assert support[0].item() > 0
    assert support[1].item() == 0


def test_multiview_aggregation_honors_visibility_hard_gate():
    supports = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    valid_masks = torch.ones((2, 2), dtype=torch.bool)
    visibility = torch.tensor([[True, False], [False, True]])

    boundary, counts = aggregate_multiview_edge_support(
        sampled_supports=supports,
        valid_masks=valid_masks,
        visibility_history=visibility,
        return_counts=True,
    )

    assert torch.equal(counts, torch.ones(2))
    assert torch.allclose(boundary, torch.ones(2))

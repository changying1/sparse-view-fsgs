"""2D multiview edge support utilities for FSGS OBDKR value."""

from types import SimpleNamespace

import torch
import torch.nn.functional as F


def compute_edge_map(image, operator="scharr", eps=1e-8):
    with torch.no_grad():
        image_chw = _as_chw_image(image)
        gray = _rgb_to_gray(image_chw).unsqueeze(0).unsqueeze(0)
        kernel_x, kernel_y = _gradient_kernels(operator, gray.dtype, gray.device)
        padded = F.pad(gray, (1, 1, 1, 1), mode="replicate")
        magnitude = torch.sqrt(F.conv2d(padded, kernel_x).square() + F.conv2d(padded, kernel_y).square())
        edge_map = magnitude.squeeze(0).squeeze(0)
        return (edge_map / edge_map.max().clamp_min(eps)).clamp(0.0, 1.0)


def project_gaussians_to_camera(
    xyz,
    camera=None,
    full_proj_transform=None,
    image_width=None,
    image_height=None,
    world_view_transform=None,
    eps=1e-7,
):
    with torch.no_grad():
        if camera is not None:
            full_proj_transform = camera.full_proj_transform
            image_width = camera.image_width
            image_height = camera.image_height
            world_view_transform = getattr(camera, "world_view_transform", world_view_transform)
        _validate_xyz(xyz)
        if full_proj_transform is None or image_width is None or image_height is None:
            raise ValueError("Projection requires camera or transform and image size.")

        full_proj_transform = full_proj_transform.to(device=xyz.device, dtype=xyz.dtype)
        ones = torch.ones((xyz.shape[0], 1), dtype=xyz.dtype, device=xyz.device)
        xyz_h = torch.cat((xyz, ones), dim=-1)
        clip = xyz_h @ full_proj_transform
        w = clip[:, 3]
        safe_w = w > eps
        ndc = clip[:, :3] / torch.where(safe_w, w, torch.ones_like(w)).unsqueeze(-1)

        width, height = float(image_width), float(image_height)
        x_pixel = (ndc[:, 0] + 1.0) * 0.5 * width
        y_pixel = (ndc[:, 1] + 1.0) * 0.5 * height
        pixel_coords = torch.stack((x_pixel, y_pixel), dim=-1)

        finite = torch.isfinite(clip).all(dim=-1) & torch.isfinite(pixel_coords).all(dim=-1)
        inside = (x_pixel >= 0.0) & (x_pixel <= width) & (y_pixel >= 0.0) & (y_pixel <= height)
        in_front = torch.ones_like(safe_w)
        if world_view_transform is not None:
            world_view_transform = world_view_transform.to(device=xyz.device, dtype=xyz.dtype)
            view = xyz_h @ world_view_transform
            in_front = view[:, 2] > eps
            finite = finite & torch.isfinite(view).all(dim=-1)

        valid_mask = finite & safe_w & in_front & inside
        return torch.where(valid_mask[:, None], pixel_coords, torch.zeros_like(pixel_coords)), valid_mask


def sample_edge_support(edge_map, pixel_coords, valid_mask, align_corners=False):
    with torch.no_grad():
        edge_hw = _as_hw_edge_map(edge_map)
        if pixel_coords.ndim != 2 or pixel_coords.shape[1] != 2:
            raise ValueError("pixel_coords must be a Tensor[N, 2].")
        if valid_mask.shape != (pixel_coords.shape[0],):
            raise ValueError("valid_mask must be a BoolTensor[N].")
        edge_hw = edge_hw.to(device=pixel_coords.device, dtype=pixel_coords.dtype)
        valid_mask = valid_mask.to(device=pixel_coords.device, dtype=torch.bool)
        height, width = edge_hw.shape
        grid = _pixel_to_grid(pixel_coords, width, height, align_corners=align_corners)
        sampled = F.grid_sample(
            edge_hw.view(1, 1, height, width),
            grid.view(1, -1, 1, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=align_corners,
        ).view(-1)
        sampled = torch.where(valid_mask, sampled, torch.zeros_like(sampled))
        return torch.nan_to_num(sampled, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)


def aggregate_multiview_edge_support(
    xyz=None,
    cameras=None,
    edge_maps=None,
    visibility_history=None,
    camera_uid_to_train_index=None,
    sampled_supports=None,
    valid_masks=None,
    eps=1e-8,
    return_counts=False,
):
    with torch.no_grad():
        if sampled_supports is None:
            if xyz is None or cameras is None or edge_maps is None:
                raise ValueError("Pass sampled_supports or xyz, cameras, and edge_maps.")
            if len(cameras) != len(edge_maps):
                raise ValueError("cameras and edge_maps must have the same length.")
            supports = []
            masks = []
            for camera, edge_map in zip(cameras, edge_maps):
                _validate_edge_map_camera_size(edge_map, camera)
                pixel_coords, projection_valid = project_gaussians_to_camera(xyz, camera=camera)
                supports.append(sample_edge_support(edge_map, pixel_coords, projection_valid))
                masks.append(projection_valid)
            supports = torch.stack(supports, dim=0)
            masks = torch.stack(masks, dim=0)
        else:
            supports = _stack_view_tensors(sampled_supports, "sampled_supports").to(dtype=torch.float32)
            masks = _stack_view_tensors(valid_masks, "valid_masks").to(dtype=torch.bool)

        if supports.ndim != 2 or masks.shape != supports.shape:
            raise ValueError("supports and masks must have shape [V, N].")

        effective_masks = masks
        if visibility_history is not None:
            visibility_history = visibility_history.to(device=supports.device, dtype=torch.bool)
            if visibility_history.shape[0] != supports.shape[1]:
                raise ValueError("visibility_history first dimension must match N.")
            view_masks = []
            if cameras is not None:
                if camera_uid_to_train_index is None:
                    raise ValueError("visibility_history aggregation requires an explicit camera uid mapping.")
                for camera in cameras:
                    view_masks.append(visibility_history[:, camera_uid_to_train_index[getattr(camera, "uid")]])
            else:
                if visibility_history.shape[1] < supports.shape[0]:
                    raise ValueError("visibility_history has fewer columns than support views.")
                view_masks = [visibility_history[:, view_idx] for view_idx in range(supports.shape[0])]
            effective_masks = effective_masks & torch.stack(view_masks, dim=0).to(device=supports.device)

        weights = effective_masks.to(dtype=supports.dtype)
        support_count = weights.sum(dim=0)
        boundary = (supports * weights).sum(dim=0) / (support_count + eps)
        boundary = torch.where(support_count > 0, boundary, torch.zeros_like(boundary))
        boundary = torch.nan_to_num(boundary, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
        if return_counts:
            return boundary, support_count
        return boundary


def camera_from_projection(full_proj_transform, image_width, image_height, world_view_transform=None, uid=None):
    return SimpleNamespace(
        full_proj_transform=full_proj_transform,
        world_view_transform=world_view_transform,
        image_width=image_width,
        image_height=image_height,
        uid=uid,
    )


def _as_chw_image(image):
    if not torch.is_tensor(image):
        raise ValueError("image must be a torch Tensor.")
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError("Batched image input must have shape [1, C, H, W].")
        image = image.squeeze(0)
    if image.ndim != 3:
        raise ValueError("image must have shape [C, H, W] or [1, C, H, W].")
    return image.detach()


def _rgb_to_gray(image_chw):
    if image_chw.shape[0] == 1:
        return image_chw[0]
    rgb = image_chw[:3]
    weights = torch.tensor([0.299, 0.587, 0.114], dtype=rgb.dtype, device=rgb.device)
    return (rgb * weights[:, None, None]).sum(dim=0)


def _gradient_kernels(operator, dtype, device):
    if operator == "scharr":
        gx = torch.tensor([[3.0, 0.0, -3.0], [10.0, 0.0, -10.0], [3.0, 0.0, -3.0]], dtype=dtype, device=device)
        gy = torch.tensor([[3.0, 10.0, 3.0], [0.0, 0.0, 0.0], [-3.0, -10.0, -3.0]], dtype=dtype, device=device)
    elif operator == "sobel":
        gx = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]], dtype=dtype, device=device)
        gy = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]], dtype=dtype, device=device)
    else:
        raise ValueError("operator must be 'scharr' or 'sobel'.")
    return gx.view(1, 1, 3, 3), gy.view(1, 1, 3, 3)


def _as_hw_edge_map(edge_map):
    if not torch.is_tensor(edge_map):
        raise ValueError("edge_map must be a torch Tensor.")
    if edge_map.ndim == 3:
        if edge_map.shape[0] != 1:
            raise ValueError("edge_map with 3 dims must have shape [1, H, W].")
        edge_map = edge_map.squeeze(0)
    if edge_map.ndim != 2:
        raise ValueError("edge_map must have shape [H, W] or [1, H, W].")
    return edge_map.detach()


def _validate_edge_map_camera_size(edge_map, camera):
    edge_h, edge_w = _as_hw_edge_map(edge_map).shape
    if edge_h != int(camera.image_height) or edge_w != int(camera.image_width):
        raise ValueError("edge_map resolution must match camera resolution.")


def _pixel_to_grid(pixel_coords, width, height, align_corners=False):
    if align_corners:
        x_norm = 2.0 * pixel_coords[:, 0] / max(width - 1, 1) - 1.0
        y_norm = 2.0 * pixel_coords[:, 1] / max(height - 1, 1) - 1.0
    else:
        x_norm = 2.0 * pixel_coords[:, 0] / width - 1.0
        y_norm = 2.0 * pixel_coords[:, 1] / height - 1.0
    return torch.stack((x_norm, y_norm), dim=-1)


def _stack_view_tensors(tensors, name):
    if tensors is None:
        raise ValueError(f"{name} is required.")
    if torch.is_tensor(tensors):
        return tensors
    return torch.stack(list(tensors), dim=0)


def _validate_xyz(xyz):
    if not torch.is_tensor(xyz) or xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must be a Tensor[N, 3].")

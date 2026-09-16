import torch

from utils.structural_graph import (
    build_knn_graph,
    compute_continuity_defect,
    compute_geometric_turning,
    estimate_gaussian_normals,
)


def _nan(device=None, dtype=torch.float32):
    return torch.tensor(float("nan"), device=device, dtype=dtype)


def _as_float(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _mean_or_nan(values):
    if values.numel() == 0:
        return _nan(device=values.device, dtype=values.dtype)
    return torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).mean()


def _quantile_or_nan(values, q):
    if values.numel() == 0:
        return _nan(device=values.device, dtype=values.dtype)
    values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.quantile(values.float(), q).to(device=values.device, dtype=values.dtype)


def _corr_or_nan(a, b, eps=1e-8):
    if a.numel() < 2:
        return _nan(device=a.device, dtype=a.dtype)
    a = torch.nan_to_num(a.float(), nan=0.0, posinf=0.0, neginf=0.0)
    b = torch.nan_to_num(b.to(device=a.device).float(), nan=0.0, posinf=0.0, neginf=0.0)
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denom = torch.linalg.norm(a_centered) * torch.linalg.norm(b_centered)
    if float(denom.item()) <= eps:
        return _nan(device=a.device, dtype=a.dtype)
    return (a_centered * b_centered).sum() / denom.clamp_min(eps)


def align_mono_depth_affine(rendered_depth, mono_depth, valid_mask=None, eps=1e-8):
    with torch.no_grad():
        rendered = rendered_depth.detach().float()
        mono = mono_depth.detach().to(device=rendered.device, dtype=rendered.dtype)
        if rendered.ndim == 3 and rendered.shape[0] == 1:
            rendered = rendered[0]
        if mono.ndim == 3 and mono.shape[0] == 1:
            mono = mono[0]
        valid = torch.isfinite(rendered) & torch.isfinite(mono) & (rendered > 0)
        if valid_mask is not None:
            valid = valid & valid_mask.detach().to(device=rendered.device).bool()
        if valid.sum() < 2:
            return (
                torch.full_like(rendered, float("nan")),
                torch.zeros_like(valid, dtype=torch.bool),
                _nan(rendered.device, rendered.dtype),
                _nan(rendered.device, rendered.dtype),
            )

        x = mono[valid].reshape(-1)
        y = rendered[valid].reshape(-1)
        x_mean = x.mean()
        y_mean = y.mean()
        var = ((x - x_mean) * (x - x_mean)).mean()
        if not torch.isfinite(var) or float(var.item()) <= eps:
            return (
                torch.full_like(rendered, float("nan")),
                torch.zeros_like(valid, dtype=torch.bool),
                _nan(rendered.device, rendered.dtype),
                _nan(rendered.device, rendered.dtype),
            )
        cov = ((x - x_mean) * (y - y_mean)).mean()
        a = cov / var.clamp_min(eps)
        b = y_mean - a * x_mean
        if not torch.isfinite(a) or not torch.isfinite(b):
            return (
                torch.full_like(rendered, float("nan")),
                torch.zeros_like(valid, dtype=torch.bool),
                _nan(rendered.device, rendered.dtype),
                _nan(rendered.device, rendered.dtype),
            )
        aligned = a * mono + b
        return aligned, valid, a, b


def _valid_affine_depth_alignment(rendered_depth, mono_depth, valid_mask=None, eps=1e-8):
    aligned, valid, a, b = align_mono_depth_affine(rendered_depth, mono_depth, valid_mask=valid_mask, eps=eps)
    return aligned, valid, a, b, bool(valid.any() and torch.isfinite(a) and torch.isfinite(b))


def compute_depth_evidence_map(rendered_depth, mono_depth, valid_mask=None, eps=1e-8):
    with torch.no_grad():
        rendered = rendered_depth.detach().float()
        if rendered.ndim == 3 and rendered.shape[0] == 1:
            rendered = rendered[0]
        aligned, valid, a, b, affine_valid = _valid_affine_depth_alignment(
            rendered,
            mono_depth,
            valid_mask=valid_mask,
            eps=eps,
        )
        residual = torch.full_like(rendered, float("nan"))
        evidence = torch.full_like(rendered, float("nan"))
        if affine_valid and valid.any():
            raw = torch.abs(rendered - aligned) / torch.abs(rendered).clamp_min(eps)
            raw = torch.nan_to_num(raw, nan=float("nan"), posinf=float("nan"), neginf=float("nan"))
            valid_raw = raw[valid]
            valid_raw = valid_raw[torch.isfinite(valid_raw)]
            if valid_raw.numel() < 2:
                valid = torch.zeros_like(valid, dtype=torch.bool)
            else:
                r_low = torch.quantile(valid_raw.float(), 0.05).to(device=rendered.device, dtype=rendered.dtype)
                r_high = torch.quantile(valid_raw.float(), 0.95).to(device=rendered.device, dtype=rendered.dtype)
                if torch.isfinite(r_low) and torch.isfinite(r_high) and float((r_high - r_low).item()) > eps:
                    normalized = ((raw - r_low) / (r_high - r_low)).clamp(0.0, 1.0)
                    residual[valid] = raw[valid]
                    evidence[valid] = 1.0 - normalized[valid]
                elif valid_raw.max().item() <= eps:
                    residual[valid] = raw[valid]
                    evidence[valid] = 1.0
                else:
                    valid = torch.zeros_like(valid, dtype=torch.bool)
        return {
            "aligned_depth": aligned,
            "valid_mask": valid,
            "residual": residual,
            "evidence": evidence,
            "affine_a": a,
            "affine_b": b,
        }


def project_gaussians_to_pixels(xyz, camera, eps=1e-8):
    with torch.no_grad():
        points = xyz.detach()
        ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)
        homogeneous = torch.cat((points, ones), dim=-1)
        full = camera.full_proj_transform.to(device=points.device, dtype=points.dtype)
        view = camera.world_view_transform.to(device=points.device, dtype=points.dtype)
        clip = homogeneous @ full
        view_points = homogeneous @ view
        ndc = clip[:, :3] / clip[:, 3:4].clamp_min(eps)
        width = int(camera.image_width)
        height = int(camera.image_height)
        px = ((ndc[:, 0] + 1.0) * 0.5 * (width - 1)).round().long()
        py = ((1.0 - ndc[:, 1]) * 0.5 * (height - 1)).round().long()
        in_image = (px >= 0) & (px < width) & (py >= 0) & (py < height)
        positive_depth = view_points[:, 2] > 0
        return px, py, in_image & positive_depth


def sample_gaussian_evidence_for_view(xyz, camera, evidence_map, residual_map, valid_depth_mask, visible_mask):
    with torch.no_grad():
        evidence = evidence_map.detach().to(device=xyz.device, dtype=xyz.dtype)
        residual = residual_map.detach().to(device=xyz.device, dtype=xyz.dtype)
        valid_depth = valid_depth_mask.detach().to(device=xyz.device).bool()
        visible = visible_mask.detach().to(device=xyz.device).bool()
        if visible.shape[0] != xyz.shape[0]:
            raise ValueError("visible_mask must match Gaussian count.")

        px, py, valid_projection = project_gaussians_to_pixels(xyz, camera)
        valid = visible & valid_projection
        sampled_e = torch.full((xyz.shape[0],), float("nan"), dtype=xyz.dtype, device=xyz.device)
        sampled_r = torch.full((xyz.shape[0],), float("nan"), dtype=xyz.dtype, device=xyz.device)
        if valid.any():
            valid_indices = torch.where(valid)[0]
            depth_ok = valid_depth[py[valid_indices], px[valid_indices]]
            valid_indices = valid_indices[depth_ok]
            if valid_indices.numel() > 0:
                sampled_e[valid_indices] = evidence[py[valid_indices], px[valid_indices]]
                sampled_r[valid_indices] = residual[py[valid_indices], px[valid_indices]]
        valid_sample = torch.isfinite(sampled_e) & torch.isfinite(sampled_r)
        return sampled_e, sampled_r, valid_sample


def compute_photometric_residual_map(rendered_rgb, gt_rgb, valid_mask=None):
    with torch.no_grad():
        rendered = rendered_rgb.detach().float()
        gt = gt_rgb.detach().to(device=rendered.device, dtype=rendered.dtype)
        if rendered.ndim == 3 and rendered.shape[0] not in (1, 3) and rendered.shape[-1] in (1, 3):
            rendered = rendered.permute(2, 0, 1)
        if gt.ndim == 3 and gt.shape[0] not in (1, 3) and gt.shape[-1] in (1, 3):
            gt = gt.permute(2, 0, 1)
        if rendered.shape != gt.shape:
            raise ValueError("rendered_rgb and gt_rgb must have matching shape.")
        residual = torch.mean(torch.abs(rendered - gt), dim=0)
        valid = torch.isfinite(rendered).all(dim=0) & torch.isfinite(gt).all(dim=0) & torch.isfinite(residual)
        if valid_mask is not None:
            mask = valid_mask.detach().to(device=rendered.device).bool()
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask[0]
            elif mask.ndim == 3 and mask.shape[-1] == 1:
                mask = mask[..., 0]
            valid = valid & mask
        residual = residual.to(dtype=rendered_rgb.dtype if torch.is_floating_point(rendered_rgb) else torch.float32)
        residual = residual.masked_fill(~valid, float("nan"))
        return residual, valid


def sample_gaussian_photo_for_view(xyz, camera, photo_map, valid_photo_mask, visible_mask):
    with torch.no_grad():
        photo = photo_map.detach().to(device=xyz.device, dtype=xyz.dtype)
        valid_photo = valid_photo_mask.detach().to(device=xyz.device).bool()
        visible = visible_mask.detach().to(device=xyz.device).bool()
        if visible.shape[0] != xyz.shape[0]:
            raise ValueError("visible_mask must match Gaussian count.")

        px, py, valid_projection = project_gaussians_to_pixels(xyz, camera)
        valid = visible & valid_projection
        sampled_photo = torch.full((xyz.shape[0],), float("nan"), dtype=xyz.dtype, device=xyz.device)
        if valid.any():
            valid_indices = torch.where(valid)[0]
            photo_ok = valid_photo[py[valid_indices], px[valid_indices]]
            valid_indices = valid_indices[photo_ok]
            if valid_indices.numel() > 0:
                sampled_photo[valid_indices] = photo[py[valid_indices], px[valid_indices]]
        valid_sample = torch.isfinite(sampled_photo)
        return sampled_photo, valid_sample


def aggregate_gaussian_observation_evidence(xyz, cameras, rendered_depths, mono_depths, visibility_masks, valid_masks=None):
    with torch.no_grad():
        evidence_sum = torch.zeros((xyz.shape[0],), dtype=xyz.dtype, device=xyz.device)
        residual_sum = torch.zeros((xyz.shape[0],), dtype=xyz.dtype, device=xyz.device)
        view_count = torch.zeros((xyz.shape[0],), dtype=torch.long, device=xyz.device)
        valid_masks = [None] * len(cameras) if valid_masks is None else valid_masks
        view_residuals = []
        for camera, rendered_depth, mono_depth, visible, valid_mask in zip(
            cameras,
            rendered_depths,
            mono_depths,
            visibility_masks,
            valid_masks,
        ):
            maps = compute_depth_evidence_map(rendered_depth, mono_depth, valid_mask=valid_mask)
            sampled_e, sampled_r, valid_sample = sample_gaussian_evidence_for_view(
                xyz,
                camera,
                maps["evidence"],
                maps["residual"],
                maps["valid_mask"],
                visible,
            )
            evidence_sum[valid_sample] += sampled_e[valid_sample]
            residual_sum[valid_sample] += sampled_r[valid_sample]
            view_count[valid_sample] += 1
            view_residuals.append(maps["residual"][maps["valid_mask"]])
        valid_gaussians = view_count > 0
        gaussian_evidence = torch.full((xyz.shape[0],), float("nan"), dtype=xyz.dtype, device=xyz.device)
        gaussian_residual = torch.full((xyz.shape[0],), float("nan"), dtype=xyz.dtype, device=xyz.device)
        gaussian_evidence[valid_gaussians] = evidence_sum[valid_gaussians] / view_count[valid_gaussians].to(dtype=xyz.dtype)
        gaussian_residual[valid_gaussians] = residual_sum[valid_gaussians] / view_count[valid_gaussians].to(dtype=xyz.dtype)
        return gaussian_evidence, gaussian_residual, view_count, view_residuals


def aggregate_gaussian_photometric_residual(xyz, cameras, rendered_rgbs, visibility_masks, valid_masks=None):
    with torch.no_grad():
        photo_sum = torch.zeros((xyz.shape[0],), dtype=xyz.dtype, device=xyz.device)
        photo_view_count = torch.zeros((xyz.shape[0],), dtype=torch.long, device=xyz.device)
        valid_masks = [None] * len(cameras) if valid_masks is None else valid_masks
        for camera, rendered_rgb, visible, valid_mask in zip(cameras, rendered_rgbs, visibility_masks, valid_masks):
            gt_rgb = torch.as_tensor(camera.original_image, device=xyz.device)
            photo_map, valid_photo = compute_photometric_residual_map(rendered_rgb, gt_rgb, valid_mask=valid_mask)
            sampled_photo, valid_sample = sample_gaussian_photo_for_view(
                xyz,
                camera,
                photo_map,
                valid_photo,
                visible,
            )
            photo_sum[valid_sample] += sampled_photo[valid_sample]
            photo_view_count[valid_sample] += 1
        valid_gaussians = photo_view_count > 0
        gaussian_photo = torch.full((xyz.shape[0],), float("nan"), dtype=xyz.dtype, device=xyz.device)
        gaussian_photo[valid_gaussians] = photo_sum[valid_gaussians] / photo_view_count[valid_gaussians].to(dtype=xyz.dtype)
        return gaussian_photo, photo_view_count


def _evidence_split_photo_means(evidence, photo):
    if evidence.numel() == 0:
        return _nan(evidence.device, evidence.dtype), _nan(evidence.device, evidence.dtype)
    e_q25 = torch.quantile(evidence.float(), 0.25).to(device=evidence.device, dtype=evidence.dtype)
    e_q75 = torch.quantile(evidence.float(), 0.75).to(device=evidence.device, dtype=evidence.dtype)
    return _mean_or_nan(photo[evidence <= e_q25]), _mean_or_nan(photo[evidence >= e_q75])


def _conditional_reliability_stats(prefix, structural_values, evidence, photo):
    if structural_values.numel() == 0:
        empty = evidence.new_empty((0,), dtype=torch.bool)
        subset = empty
    else:
        threshold = torch.quantile(structural_values.float(), 0.75).to(
            device=structural_values.device,
            dtype=structural_values.dtype,
        )
        subset = structural_values >= threshold
    subset_e = evidence[subset]
    subset_photo = photo[subset]
    low_photo, high_photo = _evidence_split_photo_means(subset_e, subset_photo)
    return {
        f"{prefix}_valid": int(subset.sum().item()),
        f"{prefix}_E_mean": _as_float(_mean_or_nan(subset_e)),
        f"{prefix}_lowE_photo": _as_float(low_photo),
        f"{prefix}_highE_photo": _as_float(high_photo),
        f"{prefix}_corr_E_photo": _as_float(_corr_or_nan(subset_e, subset_photo)),
    }


def compute_observation_evidence_diagnostics(
    xyz,
    scales,
    rotations,
    cameras,
    rendered_depths,
    mono_depths,
    visibility_masks,
    valid_masks=None,
    rendered_rgbs=None,
    knn_k=12,
):
    with torch.no_grad():
        xyz = xyz.detach()
        gaussian_e, gaussian_r, view_count, view_residuals = aggregate_gaussian_observation_evidence(
            xyz,
            cameras,
            rendered_depths,
            mono_depths,
            visibility_masks,
            valid_masks=valid_masks,
        )
        valid = torch.isfinite(gaussian_e) & (view_count > 0)
        neighbors = build_knn_graph(xyz, k=knn_k)
        normals = estimate_gaussian_normals(scales.detach(), rotations.detach())
        k_value = compute_geometric_turning(xyz, normals, neighbors)
        d_value = compute_continuity_defect(xyz, normals, neighbors)
        u_base = k_value + d_value

        valid_e = gaussian_e[valid]
        valid_r = gaussian_r[valid]
        valid_counts = view_count[valid].to(dtype=xyz.dtype)
        valid_d = d_value[valid]
        valid_k = k_value[valid]
        valid_u = u_base[valid]
        all_residuals = torch.cat(view_residuals) if view_residuals else xyz.new_empty((0,))
        all_residuals = all_residuals[torch.isfinite(all_residuals)]
        if rendered_rgbs is not None:
            gaussian_photo, photo_view_count = aggregate_gaussian_photometric_residual(
                xyz,
                cameras,
                rendered_rgbs,
                visibility_masks,
                valid_masks=valid_masks,
            )
        else:
            gaussian_photo = torch.full((xyz.shape[0],), float("nan"), dtype=xyz.dtype, device=xyz.device)
            photo_view_count = torch.zeros((xyz.shape[0],), dtype=torch.long, device=xyz.device)
        valid_joint = valid & torch.isfinite(gaussian_photo) & (photo_view_count > 0)
        joint_e = gaussian_e[valid_joint]
        joint_photo = gaussian_photo[valid_joint]
        joint_d = d_value[valid_joint]
        joint_k = k_value[valid_joint]
        joint_u = u_base[valid_joint]
        low_photo, high_photo = _evidence_split_photo_means(joint_e, joint_photo)

        stats = {
            "views": len(cameras),
            "valid_evidence_gaussians": int(valid.sum().item()),
            "valid_evidence_ratio": _as_float(valid.float().mean()) if valid.numel() else float("nan"),
            "evidence_mean": _as_float(_mean_or_nan(valid_e)),
            "evidence_q10": _as_float(_quantile_or_nan(valid_e, 0.10)),
            "evidence_q25": _as_float(_quantile_or_nan(valid_e, 0.25)),
            "evidence_q50": _as_float(_quantile_or_nan(valid_e, 0.50)),
            "evidence_q75": _as_float(_quantile_or_nan(valid_e, 0.75)),
            "evidence_q90": _as_float(_quantile_or_nan(valid_e, 0.90)),
            "depth_residual_mean": _as_float(_mean_or_nan(all_residuals)),
            "depth_residual_q50": _as_float(_quantile_or_nan(all_residuals, 0.50)),
            "depth_residual_q75": _as_float(_quantile_or_nan(all_residuals, 0.75)),
            "depth_residual_q90": _as_float(_quantile_or_nan(all_residuals, 0.90)),
            "evidence_view_count_mean": _as_float(_mean_or_nan(valid_counts)),
            "evidence_view_count_q25": _as_float(_quantile_or_nan(valid_counts, 0.25)),
            "evidence_view_count_q50": _as_float(_quantile_or_nan(valid_counts, 0.50)),
            "evidence_view_count_q75": _as_float(_quantile_or_nan(valid_counts, 0.75)),
            "corr_E_D": _as_float(_corr_or_nan(valid_e, valid_d)),
            "corr_E_K": _as_float(_corr_or_nan(valid_e, valid_k)),
            "corr_E_Ubase": _as_float(_corr_or_nan(valid_e, valid_u)),
            "valid_joint": int(valid_joint.sum().item()),
            "valid_joint_ratio": _as_float(valid_joint.float().mean()) if valid_joint.numel() else float("nan"),
            "photo_mean": _as_float(_mean_or_nan(joint_photo)),
            "photo_q25": _as_float(_quantile_or_nan(joint_photo, 0.25)),
            "photo_q50": _as_float(_quantile_or_nan(joint_photo, 0.50)),
            "photo_q75": _as_float(_quantile_or_nan(joint_photo, 0.75)),
            "corr_E_photo": _as_float(_corr_or_nan(joint_e, joint_photo)),
            "E_low25_photo_mean": _as_float(low_photo),
            "E_high25_photo_mean": _as_float(high_photo),
        }
        stats.update(_conditional_reliability_stats("HighD", joint_d, joint_e, joint_photo))
        stats.update(_conditional_reliability_stats("HighK", joint_k, joint_e, joint_photo))
        stats.update(_conditional_reliability_stats("HighU", joint_u, joint_e, joint_photo))

        if valid_e.numel() > 0:
            d_q25 = torch.quantile(valid_d.float(), 0.25).to(device=xyz.device, dtype=xyz.dtype)
            d_q50 = torch.quantile(valid_d.float(), 0.50).to(device=xyz.device, dtype=xyz.dtype)
            d_q75 = torch.quantile(valid_d.float(), 0.75).to(device=xyz.device, dtype=xyz.dtype)
            quartiles = (
                valid_d <= d_q25,
                (valid_d > d_q25) & (valid_d <= d_q50),
                (valid_d > d_q50) & (valid_d <= d_q75),
                valid_d > d_q75,
            )
            e_q25 = torch.quantile(valid_e.float(), 0.25).to(device=xyz.device, dtype=xyz.dtype)
            e_q75 = torch.quantile(valid_e.float(), 0.75).to(device=xyz.device, dtype=xyz.dtype)
            e_low = valid_e <= e_q25
            e_high = valid_e >= e_q75
        else:
            quartiles = (valid_e.bool(), valid_e.bool(), valid_e.bool(), valid_e.bool())
            e_low = valid_e.bool()
            e_high = valid_e.bool()

        stats.update(
            {
                "D_Q1_E_mean": _as_float(_mean_or_nan(valid_e[quartiles[0]])),
                "D_Q2_E_mean": _as_float(_mean_or_nan(valid_e[quartiles[1]])),
                "D_Q3_E_mean": _as_float(_mean_or_nan(valid_e[quartiles[2]])),
                "D_Q4_E_mean": _as_float(_mean_or_nan(valid_e[quartiles[3]])),
                "E_low25_D_mean": _as_float(_mean_or_nan(valid_d[e_low])),
                "E_high25_D_mean": _as_float(_mean_or_nan(valid_d[e_high])),
                "E_low25_K_mean": _as_float(_mean_or_nan(valid_k[e_low])),
                "E_high25_K_mean": _as_float(_mean_or_nan(valid_k[e_high])),
                "E_low25_Ubase_mean": _as_float(_mean_or_nan(valid_u[e_low])),
                "E_high25_Ubase_mean": _as_float(_mean_or_nan(valid_u[e_high])),
            }
        )

        if neighbors.shape[1] > 0:
            edge_valid = valid[:, None] & valid[neighbors]
            edge_e = torch.minimum(gaussian_e[:, None], gaussian_e[neighbors])[edge_valid]
            total_edges = neighbors.numel()
        else:
            edge_valid = torch.empty((xyz.shape[0], 0), dtype=torch.bool, device=xyz.device)
            edge_e = xyz.new_empty((0,))
            total_edges = 0
        stats.update(
            {
                "valid_evidence_edges": int(edge_valid.sum().item()),
                "valid_evidence_edge_ratio": _as_float(edge_valid.float().mean()) if total_edges > 0 else float("nan"),
                "edge_E_mean": _as_float(_mean_or_nan(edge_e)),
                "edge_E_q25": _as_float(_quantile_or_nan(edge_e, 0.25)),
                "edge_E_q50": _as_float(_quantile_or_nan(edge_e, 0.50)),
                "edge_E_q75": _as_float(_quantile_or_nan(edge_e, 0.75)),
            }
        )
        return stats, gaussian_e, gaussian_r, view_count


def format_observation_evidence_diag(iteration, stats):
    fields = (
        "iter",
        "views",
        "valid_gaussians",
        "valid_ratio",
        "E_mean",
        "E_q10",
        "E_q25",
        "E_q50",
        "E_q75",
        "E_q90",
        "Rdepth_mean",
        "Rdepth_q50",
        "Rdepth_q75",
        "Rdepth_q90",
        "view_count_mean",
        "corr_E_D",
        "corr_E_K",
        "corr_E_Ubase",
        "D_Q1_E",
        "D_Q2_E",
        "D_Q3_E",
        "D_Q4_E",
        "E_low25_D",
        "E_high25_D",
        "E_low25_K",
        "E_high25_K",
        "E_low25_Ubase",
        "E_high25_Ubase",
    )
    aliases = {
        "iter": iteration,
        "valid_gaussians": stats["valid_evidence_gaussians"],
        "valid_ratio": stats["valid_evidence_ratio"],
        "E_mean": stats["evidence_mean"],
        "E_q10": stats["evidence_q10"],
        "E_q25": stats["evidence_q25"],
        "E_q50": stats["evidence_q50"],
        "E_q75": stats["evidence_q75"],
        "E_q90": stats["evidence_q90"],
        "Rdepth_mean": stats["depth_residual_mean"],
        "Rdepth_q50": stats["depth_residual_q50"],
        "Rdepth_q75": stats["depth_residual_q75"],
        "Rdepth_q90": stats["depth_residual_q90"],
        "view_count_mean": stats["evidence_view_count_mean"],
        "D_Q1_E": stats["D_Q1_E_mean"],
        "D_Q2_E": stats["D_Q2_E_mean"],
        "D_Q3_E": stats["D_Q3_E_mean"],
        "D_Q4_E": stats["D_Q4_E_mean"],
        "E_low25_D": stats["E_low25_D_mean"],
        "E_high25_D": stats["E_high25_D_mean"],
        "E_low25_K": stats["E_low25_K_mean"],
        "E_high25_K": stats["E_high25_K_mean"],
        "E_low25_Ubase": stats["E_low25_Ubase_mean"],
        "E_high25_Ubase": stats["E_high25_Ubase_mean"],
    }
    aliases.update(stats)
    return "[ObservationEvidenceDiag] " + " ".join(
        f"{field}={_format_value(aliases[field])}" for field in fields
    )


def format_observation_evidence_edge_diag(iteration, stats):
    fields = (
        "iter",
        "valid_evidence_edges",
        "valid_evidence_edge_ratio",
        "edge_E_mean",
        "edge_E_q25",
        "edge_E_q50",
        "edge_E_q75",
    )
    values = {**stats, "iter": iteration}
    return "[ObservationEvidenceEdgeDiag] " + " ".join(
        f"{field}={_format_value(values[field])}" for field in fields
    )


def format_observation_reliability_diag(iteration, stats):
    fields = (
        "iter",
        "valid_joint",
        "valid_joint_ratio",
        "photo_mean",
        "photo_q25",
        "photo_q50",
        "photo_q75",
        "corr_E_photo",
        "E_low25_photo",
        "E_high25_photo",
        "HighD_valid",
        "HighD_E_mean",
        "HighD_lowE_photo",
        "HighD_highE_photo",
        "HighD_corr_E_photo",
        "HighK_valid",
        "HighK_E_mean",
        "HighK_lowE_photo",
        "HighK_highE_photo",
        "HighK_corr_E_photo",
        "HighU_valid",
        "HighU_E_mean",
        "HighU_lowE_photo",
        "HighU_highE_photo",
        "HighU_corr_E_photo",
    )
    aliases = {
        **stats,
        "iter": iteration,
        "E_low25_photo": stats["E_low25_photo_mean"],
        "E_high25_photo": stats["E_high25_photo_mean"],
    }
    return "[ObservationReliabilityDiag] " + " ".join(
        f"{field}={_format_value(aliases[field])}" for field in fields
    )


def _format_value(value):
    if isinstance(value, int):
        return str(value)
    if value != value:
        return "nan"
    return f"{value:.6g}"

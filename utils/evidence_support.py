import json
import math
import os
import random
from contextlib import contextmanager

import numpy as np
import torch


def _as_depth_map(depth, detach=True):
    tensor = torch.as_tensor(depth)
    if detach:
        tensor = tensor.detach()
    tensor = tensor.float()
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim == 3 and tensor.shape[-1] == 1:
        tensor = tensor[..., 0]
    if tensor.ndim != 2:
        raise ValueError("depth inputs must be 2D maps or single-channel 3D tensors.")
    return tensor


def _safe_float(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError("Only scalar tensors can be converted to float.")
        value = value.detach().cpu().item()
    value = float(value)
    return value if math.isfinite(value) else None


def _quantile(values, q):
    if values.numel() == 0:
        return None
    return _safe_float(torch.quantile(values.float(), q))


def _mean(values):
    if values.numel() == 0:
        return None
    return _safe_float(values.float().mean())


def _median(values):
    return _quantile(values, 0.50)


def _pearson_corr(a, b, eps=1e-8):
    valid = torch.isfinite(a) & torch.isfinite(b)
    if valid.sum().item() < 2:
        return None
    x = a[valid].float()
    y = b[valid].float()
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.linalg.norm(x) * torch.linalg.norm(y)
    if not torch.isfinite(denom) or float(denom.item()) <= eps:
        return None
    return _safe_float((x * y).sum() / denom.clamp_min(eps))


def _rankdata_average(values):
    values = values.detach().float().reshape(-1)
    if values.numel() == 0:
        return values.clone()
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    _, counts = torch.unique_consecutive(sorted_values, return_counts=True)
    ends = torch.cumsum(counts, dim=0)
    starts = ends - counts
    average_ranks = (starts.float() + (ends - 1).float()) / 2.0
    sorted_ranks = torch.repeat_interleave(average_ranks, counts)
    ranks = torch.empty_like(values, dtype=torch.float32)
    ranks[order] = sorted_ranks
    return ranks


def _spearman_corr(a, b, eps=1e-8):
    valid = torch.isfinite(a) & torch.isfinite(b)
    if valid.sum().item() < 2:
        return None
    return _pearson_corr(_rankdata_average(a[valid]), _rankdata_average(b[valid]), eps=eps)


def _reference_candidates(reference_depth):
    return {
        "negative_midas": -reference_depth,
        "inverse_midas_shift_200": 1.0 / (reference_depth + 200.0),
    }


def select_fsgs_reference_depth(rendered_depth, reference_depth, valid_mask=None, eps=1e-8):
    rendered = _as_depth_map(rendered_depth)
    reference = _as_depth_map(reference_depth).to(device=rendered.device, dtype=rendered.dtype)
    if rendered.shape != reference.shape:
        raise ValueError("rendered_depth and reference_depth must have matching shapes.")

    valid = torch.isfinite(rendered) & torch.isfinite(reference)
    if valid_mask is not None:
        mask = torch.as_tensor(valid_mask, device=rendered.device).detach().bool()
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]
        if mask.shape != rendered.shape:
            raise ValueError("valid_mask must match depth map shape.")
        valid = valid & mask

    candidates = _reference_candidates(reference)
    best_name = None
    best_reference = None
    best_corr = None
    candidate_corr = {}
    flat_rendered = rendered.reshape(-1)
    flat_valid = valid.reshape(-1)
    for name, candidate in candidates.items():
        corr = _pearson_corr(flat_rendered[flat_valid], candidate.reshape(-1)[flat_valid], eps=eps)
        candidate_corr[name] = corr
        if corr is not None and (best_corr is None or corr > best_corr):
            best_name = name
            best_reference = candidate
            best_corr = corr

    if best_reference is None:
        best_name = "negative_midas"
        best_reference = candidates[best_name]

    return best_reference, best_name, best_corr, candidate_corr, valid


def _centered_rms_normalize(values, eps=1e-8):
    if values.numel() < 2:
        return None, None, None
    mean = values.float().mean()
    centered = values.float() - mean
    scale = torch.sqrt((centered * centered).mean())
    if not torch.isfinite(scale) or float(scale.item()) <= eps:
        return None, _safe_float(mean), _safe_float(scale)
    return centered / scale.clamp_min(eps), _safe_float(mean), _safe_float(scale)


def compute_real_depth_evidence_maps(
    rendered_depth,
    reference_depth,
    valid_mask=None,
    eps=1e-8,
    detach_rendered=False,
):
    rendered = _as_depth_map(rendered_depth, detach=detach_rendered)
    reference = _as_depth_map(reference_depth).to(device=rendered.device, dtype=rendered.dtype)
    selected_reference, selected_transform, selected_corr, candidate_corr, valid = select_fsgs_reference_depth(
        rendered.detach(),
        reference,
        valid_mask=valid_mask,
        eps=eps,
    )
    selected_reference = selected_reference.detach().to(device=rendered.device, dtype=rendered.dtype)
    valid = valid.to(device=rendered.device).detach() & torch.isfinite(rendered) & torch.isfinite(selected_reference)

    z_render = torch.full_like(rendered, float("nan"))
    z_reference = torch.full_like(rendered, float("nan"))
    normalized_signed_residual = torch.full_like(rendered, float("nan"))
    normalized_abs_residual = torch.full_like(rendered, float("nan"))
    evidence = torch.full_like(rendered, float("nan"))
    rendered_mean = None
    rendered_scale = None
    reference_mean = None
    reference_scale = None
    normalized_valid = False

    if int(valid.sum().item()) >= 2:
        rendered_values = rendered[valid]
        reference_values = selected_reference[valid]

        rendered_mean_tensor = rendered_values.float().mean()
        rendered_centered = rendered_values.float() - rendered_mean_tensor
        rendered_scale_tensor = torch.sqrt((rendered_centered * rendered_centered).mean())
        reference_mean_tensor = reference_values.float().mean()
        reference_centered = reference_values.float() - reference_mean_tensor
        reference_scale_tensor = torch.sqrt((reference_centered * reference_centered).mean())

        rendered_mean = _safe_float(rendered_mean_tensor.detach())
        rendered_scale = _safe_float(rendered_scale_tensor.detach())
        reference_mean = _safe_float(reference_mean_tensor.detach())
        reference_scale = _safe_float(reference_scale_tensor.detach())
        rendered_scale_ok = (
            torch.isfinite(rendered_scale_tensor.detach())
            and float(rendered_scale_tensor.detach().item()) > eps
        )
        reference_scale_ok = (
            torch.isfinite(reference_scale_tensor.detach())
            and float(reference_scale_tensor.detach().item()) > eps
        )

        if rendered_scale_ok and reference_scale_ok:
            normalized_valid = True
            z_render_values = rendered_centered / rendered_scale_tensor.clamp_min(eps)
            z_reference_values = (reference_centered / reference_scale_tensor.clamp_min(eps)).detach()
            residual = z_render_values - z_reference_values
            abs_residual = residual.abs()
            z_render[valid] = z_render_values.to(dtype=rendered.dtype)
            z_reference[valid] = z_reference_values.to(dtype=rendered.dtype)
            normalized_signed_residual[valid] = residual.to(dtype=rendered.dtype)
            normalized_abs_residual[valid] = abs_residual.to(dtype=rendered.dtype)
            evidence[valid] = (1.0 / (1.0 + abs_residual.detach())).to(dtype=rendered.dtype)

    return {
        "valid_mask": valid.detach(),
        "selected_reference_depth": selected_reference,
        "reference_transform": selected_transform,
        "reference_correlation": selected_corr,
        "candidate_correlations": candidate_corr,
        "normalization": "centered_rms",
        "normalized_valid": normalized_valid,
        "rendered_mean": rendered_mean,
        "rendered_centered_rms": rendered_scale,
        "reference_mean": reference_mean,
        "reference_centered_rms": reference_scale,
        "z_render": z_render,
        "z_reference": z_reference.detach(),
        "normalized_signed_residual": normalized_signed_residual,
        "normalized_absolute_residual": normalized_abs_residual.detach(),
        "evidence": evidence.detach(),
    }


def compute_real_depth_evidence_diagnostics(
    rendered_depth,
    reference_depth,
    valid_mask=None,
    eps=1e-8,
    return_evidence=False,
):
    with torch.no_grad():
        rendered = _as_depth_map(rendered_depth)
        reference = _as_depth_map(reference_depth).to(device=rendered.device, dtype=rendered.dtype)
        maps = compute_real_depth_evidence_maps(
            rendered,
            reference,
            valid_mask=valid_mask,
            eps=eps,
        )
        selected_reference = maps["selected_reference_depth"]
        selected_transform = maps["reference_transform"]
        selected_corr = maps["reference_correlation"]
        candidate_corr = maps["candidate_correlations"]
        valid = maps["valid_mask"]
        valid = valid & torch.isfinite(selected_reference)
        if valid_mask is not None:
            candidate_mask = torch.as_tensor(valid_mask, device=rendered.device).detach().bool()
            if candidate_mask.ndim == 3 and candidate_mask.shape[0] == 1:
                candidate_mask = candidate_mask[0]
            if candidate_mask.ndim == 3 and candidate_mask.shape[-1] == 1:
                candidate_mask = candidate_mask[..., 0]
            if candidate_mask.shape != rendered.shape:
                raise ValueError("valid_mask must match depth map shape.")
        else:
            candidate_mask = torch.ones_like(valid, dtype=torch.bool)
        candidate_count = int(candidate_mask.sum().item())
        finite_base = (
            torch.isfinite(rendered)
            & torch.isfinite(reference)
            & torch.isfinite(selected_reference)
            & candidate_mask
        )

        valid_count = int(valid.sum().item())
        normalized_signed_residual = torch.full_like(rendered, float("nan"))
        normalized_abs_residual = torch.full_like(rendered, float("nan"))
        raw_signed_residual = torch.full_like(rendered, float("nan"))
        raw_abs_residual = torch.full_like(rendered, float("nan"))
        raw_relative_residual = torch.full_like(rendered, float("nan"))
        evidence = torch.full_like(rendered, float("nan"))
        normalized_valid = maps["normalized_valid"]
        rendered_mean = maps["rendered_mean"]
        rendered_scale = maps["rendered_centered_rms"]
        reference_mean = maps["reference_mean"]
        reference_scale = maps["reference_centered_rms"]

        if valid_count > 0:
            raw_residual = rendered[valid] - selected_reference[valid]
            raw_signed_residual[valid] = raw_residual
            raw_abs_values = raw_residual.abs()
            raw_abs_residual[valid] = raw_abs_values
            denom = selected_reference[valid].abs()
            safe = denom > eps
            if safe.any():
                raw_relative_residual[valid] = torch.where(
                    safe,
                    raw_abs_values / denom.clamp_min(eps),
                    torch.full_like(raw_abs_values, float("nan")),
                )
            normalized_signed_residual = maps["normalized_signed_residual"]
            normalized_abs_residual = maps["normalized_absolute_residual"]
            if return_evidence:
                evidence = maps["evidence"]

        finite_normalized_abs = normalized_abs_residual[valid]
        finite_normalized_abs = finite_normalized_abs[torch.isfinite(finite_normalized_abs)]
        finite_normalized_signed = normalized_signed_residual[valid]
        finite_normalized_signed = finite_normalized_signed[torch.isfinite(finite_normalized_signed)]
        finite_raw_abs = raw_abs_residual[valid]
        finite_raw_abs = finite_raw_abs[torch.isfinite(finite_raw_abs)]
        finite_raw_rel = raw_relative_residual[valid]
        finite_raw_rel = finite_raw_rel[torch.isfinite(finite_raw_rel)]
        finite_raw_signed = raw_signed_residual[valid]
        finite_raw_signed = finite_raw_signed[torch.isfinite(finite_raw_signed)]

        stats = {
            "valid_count": valid_count,
            "candidate_count": candidate_count,
            "valid_fraction": valid_count / candidate_count if candidate_count else 0.0,
            "finite_fraction": int(finite_base.sum().item()) / candidate_count if candidate_count else 0.0,
            "reference_transform": selected_transform,
            "reference_correlation": selected_corr,
            "candidate_correlations": candidate_corr,
            "normalization": "centered_rms",
            "normalized_valid": normalized_valid,
            "rendered_mean": rendered_mean,
            "rendered_centered_rms": rendered_scale,
            "reference_mean": reference_mean,
            "reference_centered_rms": reference_scale,
            "normalized_residual_signed_mean": _mean(finite_normalized_signed),
            "normalized_residual_mean": _mean(finite_normalized_abs),
            "normalized_residual_median": _median(finite_normalized_abs),
            "normalized_residual_p25": _quantile(finite_normalized_abs, 0.25),
            "normalized_residual_p75": _quantile(finite_normalized_abs, 0.75),
            "normalized_residual_p90": _quantile(finite_normalized_abs, 0.90),
            "normalized_residual_p95": _quantile(finite_normalized_abs, 0.95),
            "raw_residual_signed_mean": _mean(finite_raw_signed),
            "raw_residual_mean": _mean(finite_raw_abs),
            "raw_residual_median": _median(finite_raw_abs),
            "raw_residual_p25": _quantile(finite_raw_abs, 0.25),
            "raw_residual_p75": _quantile(finite_raw_abs, 0.75),
            "raw_residual_p90": _quantile(finite_raw_abs, 0.90),
            "raw_residual_p95": _quantile(finite_raw_abs, 0.95),
            "raw_relative_residual_mean": _mean(finite_raw_rel),
            "raw_relative_residual_median": _median(finite_raw_rel),
            "raw_relative_residual_p90": _quantile(finite_raw_rel, 0.90),
            "raw_relative_residual_safe_count": int(finite_raw_rel.numel()),
        }
        return {
            "stats": stats,
            "valid_mask": valid,
            "selected_reference_depth": selected_reference,
            "normalized_signed_residual": normalized_signed_residual,
            "normalized_absolute_residual": normalized_abs_residual,
            "raw_signed_residual": raw_signed_residual,
            "raw_absolute_residual": raw_abs_residual,
            "raw_relative_residual": raw_relative_residual,
            "evidence": evidence if return_evidence else None,
        }


def compute_rgb_error_map(rendered_rgb, gt_rgb):
    with torch.no_grad():
        rendered = torch.as_tensor(rendered_rgb).detach().float()
        gt = torch.as_tensor(gt_rgb, device=rendered.device).detach().float()
        if rendered.ndim != 3 or gt.ndim != 3:
            raise ValueError("rendered_rgb and gt_rgb must be 3D tensors.")
        if rendered.shape[0] not in (1, 3) and rendered.shape[-1] in (1, 3):
            rendered = rendered.permute(2, 0, 1)
        if gt.shape[0] not in (1, 3) and gt.shape[-1] in (1, 3):
            gt = gt.permute(2, 0, 1)
        if rendered.shape != gt.shape:
            raise ValueError("rendered_rgb and gt_rgb must have matching shapes.")
        error = torch.mean(torch.abs(rendered - gt), dim=0)
        valid = torch.isfinite(rendered).all(dim=0) & torch.isfinite(gt).all(dim=0) & torch.isfinite(error)
        return error.masked_fill(~valid, float("nan")), valid


def compute_spatial_reliability_diagnostics(depth_diag, rendered_rgb, gt_rgb, min_valid_pixels=4, analysis_mask=None):
    with torch.no_grad():
        stats = depth_diag["stats"]
        residual = depth_diag["normalized_absolute_residual"].detach().float()
        evidence = depth_diag.get("evidence")
        if evidence is None:
            evidence = 1.0 / (1.0 + residual)
        evidence = evidence.detach().float()
        rgb_error, rgb_valid = compute_rgb_error_map(rendered_rgb, gt_rgb)
        valid = (
            bool(stats.get("normalized_valid", False))
            & torch.isfinite(evidence)
            & torch.isfinite(residual)
            & torch.isfinite(rgb_error)
            & rgb_valid
        )
        if analysis_mask is not None:
            mask = torch.as_tensor(analysis_mask, device=valid.device).detach().bool()
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask[0]
            if mask.ndim == 3 and mask.shape[-1] == 1:
                mask = mask[..., 0]
            if mask.shape != valid.shape:
                raise ValueError("analysis_mask must match image height/width.")
            valid = valid & mask
        valid_count = int(valid.sum().item())
        empty_quartiles = _empty_quartile_stats()
        if valid_count < int(min_valid_pixels):
            return {
                "valid_record": False,
                "valid_pixel_count": valid_count,
                "pearson_evidence_vs_rgb_error": None,
                "spearman_evidence_vs_rgb_error": None,
                "pearson_depth_residual_vs_rgb_error": None,
                "spearman_depth_residual_vs_rgb_error": None,
                "q4_vs_q1_rgb_error_ratio": None,
                "quartiles": empty_quartiles,
                "rgb_error_mean": None,
                "rgb_error_median": None,
            }

        valid_evidence = evidence[valid]
        valid_residual = residual[valid]
        valid_rgb = rgb_error[valid]
        quartiles = compute_evidence_quartile_rgb_error_stats(valid_evidence, valid_rgb)
        q1_mean = quartiles["Q1"]["mean_rgb_error"]
        q4_mean = quartiles["Q4"]["mean_rgb_error"]
        ratio = q4_mean / q1_mean if q1_mean is not None and q1_mean > 0 and q4_mean is not None else None
        return {
            "valid_record": True,
            "valid_pixel_count": valid_count,
            "pearson_evidence_vs_rgb_error": _pearson_corr(valid_evidence, valid_rgb),
            "spearman_evidence_vs_rgb_error": _spearman_corr(valid_evidence, valid_rgb),
            "pearson_depth_residual_vs_rgb_error": _pearson_corr(valid_residual, valid_rgb),
            "spearman_depth_residual_vs_rgb_error": _spearman_corr(valid_residual, valid_rgb),
            "q4_vs_q1_rgb_error_ratio": ratio,
            "quartiles": quartiles,
            "rgb_error_mean": _mean(valid_rgb),
            "rgb_error_median": _median(valid_rgb),
        }


def compute_gt_edge_magnitude_map(gt_rgb):
    with torch.no_grad():
        rgb = torch.as_tensor(gt_rgb).detach().float()
        if rgb.ndim != 3:
            raise ValueError("gt_rgb must be a 3D tensor.")
        if rgb.shape[0] not in (1, 3) and rgb.shape[-1] in (1, 3):
            rgb = rgb.permute(2, 0, 1)
        if rgb.shape[0] == 1:
            gray = rgb[0]
        elif rgb.shape[0] == 3:
            weights = torch.tensor([0.299, 0.587, 0.114], device=rgb.device, dtype=rgb.dtype)
            gray = (rgb * weights[:, None, None]).sum(dim=0)
        else:
            raise ValueError("gt_rgb must have 1 or 3 channels.")
        finite = torch.isfinite(gray)
        gray = torch.nan_to_num(gray, nan=0.0, posinf=0.0, neginf=0.0)
        dx = torch.zeros_like(gray)
        dy = torch.zeros_like(gray)
        dx[:, :-1] = gray[:, 1:] - gray[:, :-1]
        dy[:-1, :] = gray[1:, :] - gray[:-1, :]
        magnitude = torch.sqrt(dx * dx + dy * dy)
        return magnitude.masked_fill(~finite, float("nan"))


def compute_edge_strata_masks(edge_magnitude, edge_quantile=0.80):
    with torch.no_grad():
        edge = _as_depth_map(edge_magnitude)
        finite = torch.isfinite(edge)
        if not finite.any():
            empty = torch.zeros_like(finite, dtype=torch.bool)
            return empty, empty, None
        finite_values = edge[finite]
        threshold = torch.quantile(finite_values.float(), float(edge_quantile)).to(device=edge.device, dtype=edge.dtype)
        edge_mask = finite & (edge >= threshold)
        non_edge_mask = finite & ~edge_mask
        return edge_mask, non_edge_mask, _safe_float(threshold)


def compute_edge_stratified_reliability_diagnostics(
    depth_diag,
    rendered_rgb,
    gt_rgb,
    edge_quantile=0.80,
    min_valid_pixels=4,
):
    edge_magnitude = compute_gt_edge_magnitude_map(gt_rgb)
    edge_mask, non_edge_mask, threshold = compute_edge_strata_masks(edge_magnitude, edge_quantile=edge_quantile)
    edge_stats = compute_spatial_reliability_diagnostics(
        depth_diag,
        rendered_rgb,
        gt_rgb,
        min_valid_pixels=min_valid_pixels,
        analysis_mask=edge_mask,
    )
    non_edge_stats = compute_spatial_reliability_diagnostics(
        depth_diag,
        rendered_rgb,
        gt_rgb,
        min_valid_pixels=min_valid_pixels,
        analysis_mask=non_edge_mask,
    )
    return {
        "edge_magnitude": edge_magnitude,
        "edge_mask": edge_mask,
        "non_edge_mask": non_edge_mask,
        "edge_threshold": threshold,
        "edge": edge_stats,
        "non_edge": non_edge_stats,
    }


def compute_evidence_quartile_rgb_error_stats(evidence, rgb_error):
    evidence = torch.as_tensor(evidence).detach().float().reshape(-1)
    rgb_error = torch.as_tensor(rgb_error, device=evidence.device).detach().float().reshape(-1)
    valid = torch.isfinite(evidence) & torch.isfinite(rgb_error)
    evidence = evidence[valid]
    rgb_error = rgb_error[valid]
    if evidence.numel() == 0:
        return _empty_quartile_stats()
    order = torch.argsort(evidence, stable=True)
    chunks = torch.tensor_split(order, 4)
    quartiles = {}
    for index, chunk in enumerate(chunks, start=1):
        values = rgb_error[chunk]
        quartiles[f"Q{index}"] = {
            "count": int(values.numel()),
            "mean_rgb_error": _mean(values),
            "median_rgb_error": _median(values),
        }
    return quartiles


def _empty_quartile_stats():
    return {
        f"Q{index}": {"count": 0, "mean_rgb_error": None, "median_rgb_error": None}
        for index in range(1, 5)
    }


def make_spatial_reliability_record(iteration, view_id, depth_stats, reliability_stats):
    record = {
        "iteration": int(iteration),
        "view_id": int(view_id),
        "depth": {
            "reference_transform": depth_stats.get("reference_transform"),
            "reference_correlation": depth_stats.get("reference_correlation"),
            "normalized_residual_mean": depth_stats.get("normalized_residual_mean"),
            "normalized_residual_p90": depth_stats.get("normalized_residual_p90"),
        },
        "rgb_association": {
            "valid_record": reliability_stats.get("valid_record", False),
            "valid_pixel_count": reliability_stats.get("valid_pixel_count", 0),
            "pearson_evidence_vs_rgb_error": reliability_stats.get("pearson_evidence_vs_rgb_error"),
            "spearman_evidence_vs_rgb_error": reliability_stats.get("spearman_evidence_vs_rgb_error"),
            "pearson_depth_residual_vs_rgb_error": reliability_stats.get("pearson_depth_residual_vs_rgb_error"),
            "spearman_depth_residual_vs_rgb_error": reliability_stats.get("spearman_depth_residual_vs_rgb_error"),
        },
        "quartiles": reliability_stats.get("quartiles", _empty_quartile_stats()),
        "q4_vs_q1_rgb_error_ratio": reliability_stats.get("q4_vs_q1_rgb_error_ratio"),
    }
    return record


def make_edge_stratified_reliability_record(iteration, view_id, edge_stats):
    return {
        "iteration": int(iteration),
        "view_id": int(view_id),
        "edge_threshold": edge_stats.get("edge_threshold"),
        "edge_valid_pixel_count": edge_stats["edge"].get("valid_pixel_count", 0),
        "non_edge_valid_pixel_count": edge_stats["non_edge"].get("valid_pixel_count", 0),
        "edge": _compact_reliability_stats(edge_stats["edge"]),
        "non_edge": _compact_reliability_stats(edge_stats["non_edge"]),
    }


def _compact_reliability_stats(stats):
    return {
        "valid_record": stats.get("valid_record", False),
        "valid_pixel_count": stats.get("valid_pixel_count", 0),
        "pearson_evidence_vs_rgb_error": stats.get("pearson_evidence_vs_rgb_error"),
        "spearman_evidence_vs_rgb_error": stats.get("spearman_evidence_vs_rgb_error"),
        "pearson_depth_residual_vs_rgb_error": stats.get("pearson_depth_residual_vs_rgb_error"),
        "spearman_depth_residual_vs_rgb_error": stats.get("spearman_depth_residual_vs_rgb_error"),
        "q4_vs_q1_rgb_error_ratio": stats.get("q4_vs_q1_rgb_error_ratio"),
        "quartiles": stats.get("quartiles", _empty_quartile_stats()),
    }


def summarize_spatial_reliability_records(records):
    valid_records = [
        record for record in records
        if record.get("rgb_association", {}).get("valid_record", False)
    ]
    summary = {
        "total_snapshots": len({record.get("iteration") for record in records}),
        "total_records": len(records),
        "valid_records": len(valid_records),
        "per_view": records,
    }
    fields = (
        ("pearson_evidence_vs_rgb_error", lambda r: r["rgb_association"].get("pearson_evidence_vs_rgb_error")),
        ("spearman_evidence_vs_rgb_error", lambda r: r["rgb_association"].get("spearman_evidence_vs_rgb_error")),
        ("pearson_depth_residual_vs_rgb_error", lambda r: r["rgb_association"].get("pearson_depth_residual_vs_rgb_error")),
        ("spearman_depth_residual_vs_rgb_error", lambda r: r["rgb_association"].get("spearman_depth_residual_vs_rgb_error")),
        ("q4_vs_q1_rgb_error_ratio", lambda r: r.get("q4_vs_q1_rgb_error_ratio")),
    )
    for name, getter in fields:
        values = torch.tensor(
            [
                float(getter(record))
                for record in valid_records
                if getter(record) is not None and math.isfinite(float(getter(record)))
            ],
            dtype=torch.float32,
        )
        summary[f"{name}_mean"] = _mean(values)
        summary[f"{name}_median"] = _median(values)
    return summary


def summarize_edge_stratified_reliability_records(records):
    return {
        "total_snapshots": len({record.get("iteration") for record in records}),
        "total_records": len(records),
        "overall_edge": _summarize_stratum_records(records, "edge"),
        "overall_non_edge": _summarize_stratum_records(records, "non_edge"),
        "per_view": records,
    }


def _summarize_stratum_records(records, stratum):
    valid_records = [
        record[stratum] for record in records
        if record.get(stratum, {}).get("valid_record", False)
    ]
    summary = {
        "valid_records": len(valid_records),
        "valid_pixel_count_mean": _mean(torch.tensor(
            [record.get("valid_pixel_count", 0) for record in valid_records],
            dtype=torch.float32,
        )),
        "valid_pixel_count_median": _median(torch.tensor(
            [record.get("valid_pixel_count", 0) for record in valid_records],
            dtype=torch.float32,
        )),
    }
    fields = (
        "pearson_evidence_vs_rgb_error",
        "spearman_evidence_vs_rgb_error",
        "pearson_depth_residual_vs_rgb_error",
        "spearman_depth_residual_vs_rgb_error",
        "q4_vs_q1_rgb_error_ratio",
    )
    for field in fields:
        values = torch.tensor(
            [
                float(record[field])
                for record in valid_records
                if record.get(field) is not None and math.isfinite(float(record[field]))
            ],
            dtype=torch.float32,
        )
        summary[f"{field}_mean"] = _mean(values)
        summary[f"{field}_median"] = _median(values)
    return summary


def save_spatial_reliability_summary(model_path, records):
    output_dir = os.path.join(model_path, "oe_diagnostics")
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "spatial_reliability_summary.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_json_clean(summarize_spatial_reliability_records(records)), handle, indent=2, sort_keys=True)
    return path


def save_edge_stratified_reliability_summary(model_path, records):
    output_dir = os.path.join(model_path, "oe_diagnostics")
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "edge_stratified_reliability_summary.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_json_clean(summarize_edge_stratified_reliability_records(records)), handle, indent=2, sort_keys=True)
    return path


def should_run_spatial_reliability_snapshot(enabled, iteration, snapshots=(500, 1000, 1500, 2000)):
    return bool(enabled) and int(iteration) in {int(item) for item in snapshots}


def should_run_edge_stratified_reliability_snapshot(enabled, iteration, snapshots=(500, 1000, 1500, 2000)):
    return should_run_spatial_reliability_snapshot(enabled, iteration, snapshots=snapshots)


@contextmanager
def preserved_random_state():
    py_rng_state = random.getstate()
    np_rng_state = np.random.get_state()
    torch_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(py_rng_state)
        np.random.set_state(np_rng_state)
        torch.set_rng_state(torch_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)


def should_run_real_depth_evidence_diagnostics(enabled, iteration, interval):
    return bool(enabled) and int(interval) > 0 and int(iteration) % int(interval) == 0


def format_real_depth_evidence_diag(iteration, view_id, stats):
    fields = {
        "iter": iteration,
        "view_id": view_id,
        "valid_fraction": stats.get("valid_fraction"),
        "normalized_residual_mean": stats.get("normalized_residual_mean"),
        "normalized_residual_median": stats.get("normalized_residual_median"),
        "normalized_residual_p90": stats.get("normalized_residual_p90"),
        "finite_fraction": stats.get("finite_fraction"),
    }
    return "[OEDepthDiag] " + " ".join(f"{key}={_format_value(value)}" for key, value in fields.items())


def summarize_real_depth_evidence_records(records):
    summary = {"total_samples": len(records), "per_iteration": records}
    scalar_fields = (
        "valid_fraction",
        "normalized_residual_mean",
        "normalized_residual_median",
        "normalized_residual_p90",
        "normalized_residual_p95",
        "raw_residual_mean",
        "raw_residual_median",
        "finite_fraction",
    )
    for field in scalar_fields:
        values = torch.tensor(
            [record[field] for record in records if record.get(field) is not None and math.isfinite(float(record[field]))],
            dtype=torch.float32,
        )
        summary[f"{field}_mean"] = _mean(values)
        summary[f"{field}_median"] = _median(values)
    return summary


def save_real_depth_evidence_summary(model_path, records):
    output_dir = os.path.join(model_path, "oe_diagnostics")
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "real_depth_evidence_summary.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_json_clean(summarize_real_depth_evidence_records(records)), handle, indent=2, sort_keys=True)
    return path


def _json_clean(value):
    if isinstance(value, dict):
        return {key: _json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_clean(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _format_value(value):
    if value is None:
        return "nan"
    if isinstance(value, int):
        return str(value)
    value = float(value)
    if not math.isfinite(value):
        return "nan"
    return f"{value:.6g}"

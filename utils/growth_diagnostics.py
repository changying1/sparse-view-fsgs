from dataclasses import dataclass

import torch


def _mask_count(mask):
    if mask is None:
        return 0
    return int(mask.sum().item())


def count_split_candidates(split_gradient_mask, split_sparse_mask, split_total_mask=None):
    split_gradient_candidates = _mask_count(split_gradient_mask)
    split_sparse_candidates = _mask_count(split_sparse_mask)
    if split_total_mask is None:
        split_total_mask = split_gradient_mask.logical_or(split_sparse_mask)
    split_total_candidates = _mask_count(split_total_mask)
    return split_gradient_candidates, split_sparse_candidates, split_total_candidates


def count_proximity_proposed(proximity_sources, n=3):
    return int(proximity_sources) * int(n)


def format_proximity_growth_log(iteration, num_before, proximity_sources, proximity_proposed):
    return (
        f"[GrowthDiagProximity] iter={iteration} "
        f"before={num_before} "
        f"proximity_src={proximity_sources} "
        f"proximity_proposed={proximity_proposed}"
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


def _ratio_or_nan(mask):
    if mask.numel() == 0:
        return _nan(device=mask.device)
    return mask.float().mean()


def _select_defect(value_components, selected_pts_mask):
    if not value_components:
        return None
    defect = value_components.get("D")
    if defect is None:
        defect = value_components.get("D_norm")
    if defect is None:
        return None
    return defect.detach().to(device=selected_pts_mask.device)[selected_pts_mask]


def compute_child_structure_diagnostics(
    iteration,
    selected_pts_mask,
    current_edge_scores,
    pool_edge_scores,
    nearest_indices,
    pool_neighbors,
    value_components=None,
    xyz=None,
    eps=1e-8,
    locality_factor=1.25,
):
    with torch.no_grad():
        selected_pts_mask = selected_pts_mask.detach().bool()
        current_edge_scores = current_edge_scores.detach()
        pool_edge_scores = pool_edge_scores.detach().to(device=current_edge_scores.device, dtype=current_edge_scores.dtype)
        nearest_indices = nearest_indices.detach().long().to(device=current_edge_scores.device)
        pool_neighbors = pool_neighbors.detach().long().to(device=current_edge_scores.device)
        if xyz is not None:
            xyz = xyz.detach().to(device=current_edge_scores.device, dtype=current_edge_scores.dtype)

        selected_sources = int(selected_pts_mask.sum().item())
        child_n = int(current_edge_scores.shape[1])
        pool_k = int(pool_edge_scores.shape[1])
        children = selected_sources * child_n
        selected_current = current_edge_scores[selected_pts_mask]
        current_flat = selected_current.reshape(-1)
        source_mean_g = selected_current.mean(dim=-1) if child_n > 0 else selected_current.new_empty((selected_sources,))
        source_min_g = selected_current.min(dim=-1).values if child_n > 0 else selected_current.new_empty((selected_sources,))
        effective_n = min(child_n, pool_k)
        current_target_dist_mean = _nan(device=current_edge_scores.device, dtype=current_edge_scores.dtype)
        struct_topn_target_dist_mean = _nan(device=current_edge_scores.device, dtype=current_edge_scores.dtype)
        mean_distance_ratio = _nan(device=current_edge_scores.device, dtype=current_edge_scores.dtype)
        distance_ratio_q25 = _nan(device=current_edge_scores.device, dtype=current_edge_scores.dtype)
        distance_ratio_q50 = _nan(device=current_edge_scores.device, dtype=current_edge_scores.dtype)
        distance_ratio_q75 = _nan(device=current_edge_scores.device, dtype=current_edge_scores.dtype)
        dist_ratio_gt_125_ratio = _nan(device=current_edge_scores.device, dtype=current_edge_scores.dtype)
        dist_ratio_gt_150_ratio = _nan(device=current_edge_scores.device, dtype=current_edge_scores.dtype)
        current_target_pool_coverage = _nan(device=current_edge_scores.device, dtype=current_edge_scores.dtype)
        full_current_pool_coverage_ratio = _nan(device=current_edge_scores.device, dtype=current_edge_scores.dtype)
        distance_ratio = current_edge_scores.new_empty((0,))
        locality_candidate_mean = _nan(device=current_edge_scores.device, dtype=current_edge_scores.dtype)
        locality_full_candidate_ratio = _nan(device=current_edge_scores.device, dtype=current_edge_scores.dtype)
        locality_struct_topn_mean = current_edge_scores.new_empty((0,))
        locality_gain = current_edge_scores.new_empty((0,))
        locality_distance_ratio = current_edge_scores.new_empty((0,))
        locality_target_overlap_ratio = _nan(device=current_edge_scores.device, dtype=current_edge_scores.dtype)
        locality_changed_target_source_ratio = _nan(device=current_edge_scores.device, dtype=current_edge_scores.dtype)
        locality_complete = torch.zeros((selected_sources,), dtype=torch.bool, device=current_edge_scores.device)
        selected_xyz = None

        if xyz is not None and selected_sources > 0 and child_n > 0:
            selected_xyz = xyz[selected_pts_mask]
            current_all_targets = nearest_indices[selected_pts_mask]
            current_all_xyz = xyz[current_all_targets]
            current_all_dist = torch.linalg.norm(current_all_xyz - selected_xyz[:, None, :], dim=-1)
            current_target_dist_mean = _mean_or_nan(current_all_dist.reshape(-1))

        if selected_sources > 0 and effective_n > 0:
            selected_pool = pool_edge_scores[selected_pts_mask]
            struct_topn_g, struct_topn_pos = torch.topk(selected_pool, k=effective_n, dim=-1)
            current_topn_g = selected_current[:, :effective_n]
            struct_topn_mean = struct_topn_g.mean(dim=-1)
            current_topn_mean = current_topn_g.mean(dim=-1)
            gain = struct_topn_mean - current_topn_mean

            current_targets = nearest_indices[selected_pts_mask][:, :effective_n]
            struct_targets = pool_neighbors[selected_pts_mask].gather(1, struct_topn_pos)
            overlap = torch.zeros((selected_sources,), dtype=torch.long, device=current_edge_scores.device)
            changed = torch.zeros((selected_sources,), dtype=torch.bool, device=current_edge_scores.device)
            for idx in range(selected_sources):
                current_set = set(current_targets[idx].detach().cpu().tolist())
                struct_set = set(struct_targets[idx].detach().cpu().tolist())
                overlap[idx] = len(current_set.intersection(struct_set))
                changed[idx] = current_set != struct_set
            target_overlap_ratio = overlap.float().sum() / float(selected_sources * effective_n)
            changed_target_source_ratio = changed.float().mean()

            if xyz is not None:
                current_xyz = xyz[current_targets]
                pool_targets = pool_neighbors[selected_pts_mask]
                pool_xyz = xyz[pool_targets]
                current_dist = torch.linalg.norm(current_xyz - selected_xyz[:, None, :], dim=-1)
                pool_dist = torch.linalg.norm(pool_xyz - selected_xyz[:, None, :], dim=-1)
                struct_topn_dist = pool_dist.gather(1, struct_topn_pos)
                current_dist_mean = torch.nan_to_num(current_dist, nan=0.0, posinf=0.0, neginf=0.0).mean(dim=-1)
                struct_dist_mean = torch.nan_to_num(struct_topn_dist, nan=0.0, posinf=0.0, neginf=0.0).mean(dim=-1)
                distance_ratio = torch.nan_to_num(
                    struct_dist_mean / current_dist_mean.clamp_min(eps),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                struct_topn_target_dist_mean = _mean_or_nan(struct_topn_dist.reshape(-1))
                mean_distance_ratio = _mean_or_nan(distance_ratio)
                distance_ratio_q25 = _quantile_or_nan(distance_ratio, 0.25)
                distance_ratio_q50 = _quantile_or_nan(distance_ratio, 0.50)
                distance_ratio_q75 = _quantile_or_nan(distance_ratio, 0.75)
                dist_ratio_gt_125_ratio = _ratio_or_nan(distance_ratio > 1.25).to(dtype=current_edge_scores.dtype)
                dist_ratio_gt_150_ratio = _ratio_or_nan(distance_ratio > 1.50).to(dtype=current_edge_scores.dtype)

                dmax_current = torch.nan_to_num(
                    current_all_dist,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).max(dim=-1).values
                locality_valid = pool_dist <= (float(locality_factor) * dmax_current[:, None] + eps)
                locality_candidate_count = locality_valid.sum(dim=-1)
                locality_complete = locality_candidate_count >= child_n
                locality_candidate_mean = _mean_or_nan(locality_candidate_count.to(dtype=current_edge_scores.dtype))
                locality_full_candidate_ratio = _ratio_or_nan(locality_complete).to(dtype=current_edge_scores.dtype)
                if locality_complete.any():
                    complete_pool_scores = selected_pool[locality_complete].clone()
                    complete_pool_scores[~locality_valid[locality_complete]] = -torch.inf
                    locality_topn_g, locality_topn_pos = torch.topk(complete_pool_scores, k=child_n, dim=-1)
                    locality_topn_g = torch.nan_to_num(locality_topn_g, nan=0.0, posinf=0.0, neginf=0.0)
                    locality_struct_topn_mean = locality_topn_g.mean(dim=-1)
                    locality_current_mean = selected_current[locality_complete].mean(dim=-1)
                    locality_gain = locality_struct_topn_mean - locality_current_mean

                    locality_topn_dist = pool_dist[locality_complete].gather(1, locality_topn_pos)
                    locality_dist_mean = torch.nan_to_num(
                        locality_topn_dist,
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    ).mean(dim=-1)
                    locality_current_dist_mean = torch.nan_to_num(
                        current_all_dist[locality_complete],
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    ).mean(dim=-1)
                    locality_distance_ratio = torch.nan_to_num(
                        locality_dist_mean / locality_current_dist_mean.clamp_min(eps),
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )

                    locality_current_targets = nearest_indices[selected_pts_mask][locality_complete]
                    locality_pool_targets = pool_neighbors[selected_pts_mask][locality_complete]
                    locality_targets = locality_pool_targets.gather(1, locality_topn_pos)
                    locality_overlap = torch.zeros(
                        (int(locality_complete.sum().item()),),
                        dtype=torch.long,
                        device=current_edge_scores.device,
                    )
                    locality_changed = torch.zeros(
                        (int(locality_complete.sum().item()),),
                        dtype=torch.bool,
                        device=current_edge_scores.device,
                    )
                    for idx in range(locality_targets.shape[0]):
                        current_set = set(locality_current_targets[idx].detach().cpu().tolist())
                        locality_set = set(locality_targets[idx].detach().cpu().tolist())
                        locality_overlap[idx] = len(current_set.intersection(locality_set))
                        locality_changed[idx] = current_set != locality_set
                    locality_target_overlap_ratio = locality_overlap.float().sum() / float(locality_targets.shape[0] * child_n)
                    locality_changed_target_source_ratio = locality_changed.float().mean()
        elif xyz is not None and selected_sources > 0 and child_n > 0:
            locality_candidate_mean = current_edge_scores.new_tensor(0.0)
            locality_full_candidate_ratio = current_edge_scores.new_tensor(0.0)
        else:
            struct_topn_mean = current_edge_scores.new_empty((0,))
            current_topn_mean = current_edge_scores.new_empty((0,))
            gain = current_edge_scores.new_empty((0,))
            target_overlap_ratio = _nan(device=current_edge_scores.device, dtype=current_edge_scores.dtype)
            changed_target_source_ratio = _nan(device=current_edge_scores.device, dtype=current_edge_scores.dtype)

        if selected_sources > 0 and child_n > 0 and pool_k > 0:
            current_all_targets = nearest_indices[selected_pts_mask]
            pool_all_targets = pool_neighbors[selected_pts_mask]
            covered = torch.zeros((selected_sources, child_n), dtype=torch.bool, device=current_edge_scores.device)
            full_covered = torch.zeros((selected_sources,), dtype=torch.bool, device=current_edge_scores.device)
            for idx in range(selected_sources):
                pool_set = set(pool_all_targets[idx].detach().cpu().tolist())
                source_covered = [target in pool_set for target in current_all_targets[idx].detach().cpu().tolist()]
                covered[idx] = torch.tensor(source_covered, dtype=torch.bool, device=current_edge_scores.device)
                full_covered[idx] = all(source_covered)
            current_target_pool_coverage = covered.float().mean().to(dtype=current_edge_scores.dtype)
            full_current_pool_coverage_ratio = full_covered.float().mean().to(dtype=current_edge_scores.dtype)
        elif selected_sources > 0 and child_n > 0:
            current_target_pool_coverage = current_edge_scores.new_tensor(0.0)
            full_current_pool_coverage_ratio = current_edge_scores.new_tensor(0.0)

        selected_defect = _select_defect(value_components, selected_pts_mask)
        high_d_sources = 0
        high_d_current_g_mean = _nan(device=current_edge_scores.device, dtype=current_edge_scores.dtype)
        high_d_low_g025_ratio = _nan(device=current_edge_scores.device, dtype=current_edge_scores.dtype)
        high_d_struct_topn_g_mean = _nan(device=current_edge_scores.device, dtype=current_edge_scores.dtype)
        high_d_mean_g_gain = _nan(device=current_edge_scores.device, dtype=current_edge_scores.dtype)
        high_d_mean_distance_ratio = _nan(device=current_edge_scores.device, dtype=current_edge_scores.dtype)
        high_d_locality_struct_topn_g_mean = _nan(device=current_edge_scores.device, dtype=current_edge_scores.dtype)
        high_d_locality_mean_g_gain = _nan(device=current_edge_scores.device, dtype=current_edge_scores.dtype)
        high_d_locality_mean_distance_ratio = _nan(device=current_edge_scores.device, dtype=current_edge_scores.dtype)
        if selected_defect is not None and selected_defect.numel() > 0:
            selected_defect = torch.nan_to_num(selected_defect.to(dtype=current_edge_scores.dtype), nan=0.0, posinf=0.0, neginf=0.0)
            threshold = torch.quantile(selected_defect.float(), 0.75).to(device=selected_defect.device, dtype=selected_defect.dtype)
            high_d_mask = selected_defect >= threshold
            high_d_sources = int(high_d_mask.sum().item())
            if high_d_sources > 0:
                high_d_edges = selected_current[high_d_mask].reshape(-1)
                high_d_current_g_mean = _mean_or_nan(high_d_edges)
                high_d_low_g025_ratio = _ratio_or_nan(high_d_edges < 0.25).to(dtype=current_edge_scores.dtype)
                if effective_n > 0:
                    high_d_struct_topn_g_mean = _mean_or_nan(struct_topn_mean[high_d_mask])
                    high_d_mean_g_gain = _mean_or_nan(gain[high_d_mask])
                    if distance_ratio.numel() > 0:
                        high_d_mean_distance_ratio = _mean_or_nan(distance_ratio[high_d_mask])
                    if locality_struct_topn_mean.numel() > 0:
                        high_d_complete_positions = high_d_mask[locality_complete]
                        if high_d_complete_positions.any():
                            high_d_locality_struct_topn_g_mean = _mean_or_nan(
                                locality_struct_topn_mean[high_d_complete_positions]
                            )
                            high_d_locality_mean_g_gain = _mean_or_nan(
                                locality_gain[high_d_complete_positions]
                            )
                            high_d_locality_mean_distance_ratio = _mean_or_nan(
                                locality_distance_ratio[high_d_complete_positions]
                            )

        return {
            "iter": iteration,
            "selected_sources": selected_sources,
            "children": children,
            "pool_k": pool_k,
            "current_G_mean": _as_float(_mean_or_nan(current_flat)),
            "current_G_min": _as_float(current_flat.min() if current_flat.numel() > 0 else _nan(device=current_edge_scores.device, dtype=current_edge_scores.dtype)),
            "current_G_q10": _as_float(_quantile_or_nan(current_flat, 0.10)),
            "current_G_q25": _as_float(_quantile_or_nan(current_flat, 0.25)),
            "current_G_q50": _as_float(_quantile_or_nan(current_flat, 0.50)),
            "current_G_q75": _as_float(_quantile_or_nan(current_flat, 0.75)),
            "current_G_q90": _as_float(_quantile_or_nan(current_flat, 0.90)),
            "low_G025_ratio": _as_float(_ratio_or_nan(current_flat < 0.25)),
            "low_G050_ratio": _as_float(_ratio_or_nan(current_flat < 0.50)),
            "source_mean_G_mean": _as_float(_mean_or_nan(source_mean_g)),
            "source_min_G_mean": _as_float(_mean_or_nan(source_min_g)),
            "source_min_G_q25": _as_float(_quantile_or_nan(source_min_g, 0.25)),
            "source_min_G_q50": _as_float(_quantile_or_nan(source_min_g, 0.50)),
            "source_min_G_q75": _as_float(_quantile_or_nan(source_min_g, 0.75)),
            "struct_topN_G_mean": _as_float(_mean_or_nan(struct_topn_mean)),
            "current_topN_G_mean": _as_float(_mean_or_nan(current_topn_mean)),
            "mean_G_gain": _as_float(_mean_or_nan(gain)),
            "G_gain_q25": _as_float(_quantile_or_nan(gain, 0.25)),
            "G_gain_q50": _as_float(_quantile_or_nan(gain, 0.50)),
            "G_gain_q75": _as_float(_quantile_or_nan(gain, 0.75)),
            "gain_gt_005_ratio": _as_float(_ratio_or_nan(gain > 0.05)),
            "gain_gt_010_ratio": _as_float(_ratio_or_nan(gain > 0.10)),
            "target_overlap_ratio": _as_float(target_overlap_ratio),
            "changed_target_source_ratio": _as_float(changed_target_source_ratio),
            "current_target_dist_mean": _as_float(current_target_dist_mean),
            "struct_topN_target_dist_mean": _as_float(struct_topn_target_dist_mean),
            "mean_distance_ratio": _as_float(mean_distance_ratio),
            "distance_ratio_q25": _as_float(distance_ratio_q25),
            "distance_ratio_q50": _as_float(distance_ratio_q50),
            "distance_ratio_q75": _as_float(distance_ratio_q75),
            "dist_ratio_gt_125_ratio": _as_float(dist_ratio_gt_125_ratio),
            "dist_ratio_gt_150_ratio": _as_float(dist_ratio_gt_150_ratio),
            "current_target_pool_coverage": _as_float(current_target_pool_coverage),
            "full_current_pool_coverage_ratio": _as_float(full_current_pool_coverage_ratio),
            "locality_factor": float(locality_factor),
            "locality_candidate_mean": _as_float(locality_candidate_mean),
            "locality_full_candidate_ratio": _as_float(locality_full_candidate_ratio),
            "locality_struct_topN_G_mean": _as_float(_mean_or_nan(locality_struct_topn_mean)),
            "locality_mean_G_gain": _as_float(_mean_or_nan(locality_gain)),
            "locality_G_gain_q25": _as_float(_quantile_or_nan(locality_gain, 0.25)),
            "locality_G_gain_q50": _as_float(_quantile_or_nan(locality_gain, 0.50)),
            "locality_G_gain_q75": _as_float(_quantile_or_nan(locality_gain, 0.75)),
            "locality_gain_gt_005_ratio": _as_float(_ratio_or_nan(locality_gain > 0.05)),
            "locality_gain_gt_010_ratio": _as_float(_ratio_or_nan(locality_gain > 0.10)),
            "locality_target_overlap_ratio": _as_float(locality_target_overlap_ratio),
            "locality_changed_target_source_ratio": _as_float(locality_changed_target_source_ratio),
            "locality_mean_distance_ratio": _as_float(_mean_or_nan(locality_distance_ratio)),
            "locality_distance_ratio_q25": _as_float(_quantile_or_nan(locality_distance_ratio, 0.25)),
            "locality_distance_ratio_q50": _as_float(_quantile_or_nan(locality_distance_ratio, 0.50)),
            "locality_distance_ratio_q75": _as_float(_quantile_or_nan(locality_distance_ratio, 0.75)),
            "high_D_sources": high_d_sources,
            "high_D_current_G_mean": _as_float(high_d_current_g_mean),
            "high_D_low_G025_ratio": _as_float(high_d_low_g025_ratio),
            "high_D_struct_topN_G_mean": _as_float(high_d_struct_topn_g_mean),
            "high_D_mean_G_gain": _as_float(high_d_mean_g_gain),
            "high_D_mean_distance_ratio": _as_float(high_d_mean_distance_ratio),
            "high_D_locality_struct_topN_G_mean": _as_float(high_d_locality_struct_topn_g_mean),
            "high_D_locality_mean_G_gain": _as_float(high_d_locality_mean_g_gain),
            "high_D_locality_mean_distance_ratio": _as_float(high_d_locality_mean_distance_ratio),
        }


def _format_diag_value(value):
    if isinstance(value, int):
        return str(value)
    if value != value:
        return "nan"
    return f"{value:.6g}"


def format_child_structure_diag(stats):
    fields = (
        "iter",
        "selected_sources",
        "children",
        "pool_k",
        "current_G_mean",
        "current_G_min",
        "current_G_q10",
        "current_G_q25",
        "current_G_q50",
        "current_G_q75",
        "current_G_q90",
        "low_G025_ratio",
        "low_G050_ratio",
        "source_mean_G_mean",
        "source_min_G_mean",
        "source_min_G_q25",
        "source_min_G_q50",
        "source_min_G_q75",
        "struct_topN_G_mean",
        "current_topN_G_mean",
        "mean_G_gain",
        "G_gain_q25",
        "G_gain_q50",
        "G_gain_q75",
        "gain_gt_005_ratio",
        "gain_gt_010_ratio",
        "target_overlap_ratio",
        "changed_target_source_ratio",
        "current_target_dist_mean",
        "struct_topN_target_dist_mean",
        "mean_distance_ratio",
        "distance_ratio_q25",
        "distance_ratio_q50",
        "distance_ratio_q75",
        "dist_ratio_gt_125_ratio",
        "dist_ratio_gt_150_ratio",
        "current_target_pool_coverage",
        "full_current_pool_coverage_ratio",
        "locality_factor",
        "locality_candidate_mean",
        "locality_full_candidate_ratio",
        "locality_struct_topN_G_mean",
        "locality_mean_G_gain",
        "locality_G_gain_q25",
        "locality_G_gain_q50",
        "locality_G_gain_q75",
        "locality_gain_gt_005_ratio",
        "locality_gain_gt_010_ratio",
        "locality_target_overlap_ratio",
        "locality_changed_target_source_ratio",
        "locality_mean_distance_ratio",
        "locality_distance_ratio_q25",
        "locality_distance_ratio_q50",
        "locality_distance_ratio_q75",
        "high_D_sources",
        "high_D_current_G_mean",
        "high_D_low_G025_ratio",
        "high_D_struct_topN_G_mean",
        "high_D_mean_G_gain",
        "high_D_mean_distance_ratio",
        "high_D_locality_struct_topN_G_mean",
        "high_D_locality_mean_G_gain",
        "high_D_locality_mean_distance_ratio",
    )
    return "[ChildStructureDiag] " + " ".join(
        f"{field}={_format_diag_value(stats[field])}" for field in fields
    )


def select_structural_child_targets(
    selected_pts_mask,
    current_edge_scores,
    pool_edge_scores,
    nearest_indices,
    pool_neighbors,
    xyz,
    eps=1e-8,
    locality_factor=1.25,
):
    with torch.no_grad():
        selected_pts_mask = selected_pts_mask.detach().bool()
        xyz = xyz.detach()
        current_edge_scores = current_edge_scores.detach().to(device=xyz.device, dtype=xyz.dtype)
        pool_edge_scores = pool_edge_scores.detach().to(device=xyz.device, dtype=xyz.dtype)
        nearest_indices = nearest_indices.detach().long().to(device=xyz.device)
        pool_neighbors = pool_neighbors.detach().long().to(device=xyz.device)

        selected_sources = int(selected_pts_mask.sum().item())
        child_n = int(nearest_indices.shape[1])
        target_indices = nearest_indices[selected_pts_mask].clone()
        if selected_sources == 0 or child_n == 0:
            stats = {
                "iter": None,
                "selected_sources": selected_sources,
                "structural_sources": 0,
                "fallback_sources": selected_sources,
                "changed_sources": 0,
                "changed_source_ratio": float("nan"),
                "target_overlap_ratio": float("nan"),
                "actual_G_mean": float("nan"),
                "baseline_G_mean": float("nan"),
                "actual_distance_ratio": float("nan"),
            }
            return target_indices, stats

        selected_xyz = xyz[selected_pts_mask]
        current_targets = nearest_indices[selected_pts_mask]
        selected_current_g = current_edge_scores[selected_pts_mask]
        selected_pool_g = pool_edge_scores[selected_pts_mask]
        selected_pool_targets = pool_neighbors[selected_pts_mask]

        current_dist = torch.linalg.norm(xyz[current_targets] - selected_xyz[:, None, :], dim=-1)
        pool_dist = torch.linalg.norm(xyz[selected_pool_targets] - selected_xyz[:, None, :], dim=-1)
        dmax_current = torch.nan_to_num(current_dist, nan=0.0, posinf=0.0, neginf=0.0).max(dim=-1).values
        current_covered = torch.zeros((selected_sources,), dtype=torch.bool, device=xyz.device)
        for idx in range(selected_sources):
            pool_set = set(selected_pool_targets[idx].detach().cpu().tolist())
            current_covered[idx] = all(target in pool_set for target in current_targets[idx].detach().cpu().tolist())

        pool_scores_finite = torch.isfinite(selected_pool_g).all(dim=-1)
        current_scores_finite = torch.isfinite(selected_current_g).all(dim=-1)
        locality_valid = pool_dist <= (float(locality_factor) * dmax_current[:, None] + eps)
        locality_count = locality_valid.sum(dim=-1)
        can_use_structural = (
            current_covered
            & pool_scores_finite
            & current_scores_finite
            & (locality_count >= child_n)
        )

        actual_g = selected_current_g.clone()
        actual_dist = current_dist.clone()
        for idx in range(selected_sources):
            if not bool(can_use_structural[idx].item()):
                continue
            row_scores = selected_pool_g[idx].clone()
            row_scores[~locality_valid[idx]] = -torch.inf
            top_g, top_pos = torch.topk(row_scores, k=child_n, dim=-1)
            if not torch.isfinite(top_g).all():
                can_use_structural[idx] = False
                continue
            target_indices[idx] = selected_pool_targets[idx, top_pos]
            actual_g[idx] = top_g
            actual_dist[idx] = pool_dist[idx, top_pos]

        structural_sources = int(can_use_structural.sum().item())
        fallback_sources = selected_sources - structural_sources
        overlap = torch.zeros((selected_sources,), dtype=torch.long, device=xyz.device)
        changed = torch.zeros((selected_sources,), dtype=torch.bool, device=xyz.device)
        for idx in range(selected_sources):
            current_set = set(current_targets[idx].detach().cpu().tolist())
            actual_set = set(target_indices[idx].detach().cpu().tolist())
            overlap[idx] = len(current_set.intersection(actual_set))
            changed[idx] = current_set != actual_set
        changed_sources = int(changed.sum().item())
        current_dist_mean = torch.nan_to_num(current_dist, nan=0.0, posinf=0.0, neginf=0.0).mean(dim=-1)
        actual_dist_mean = torch.nan_to_num(actual_dist, nan=0.0, posinf=0.0, neginf=0.0).mean(dim=-1)
        actual_distance_ratio = torch.nan_to_num(
            actual_dist_mean / current_dist_mean.clamp_min(eps),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        stats = {
            "iter": None,
            "selected_sources": selected_sources,
            "structural_sources": structural_sources,
            "fallback_sources": fallback_sources,
            "changed_sources": changed_sources,
            "changed_source_ratio": _as_float(changed.float().mean()),
            "target_overlap_ratio": _as_float(overlap.float().sum() / float(selected_sources * child_n)),
            "actual_G_mean": _as_float(_mean_or_nan(actual_g.reshape(-1))),
            "baseline_G_mean": _as_float(_mean_or_nan(selected_current_g.reshape(-1))),
            "actual_distance_ratio": _as_float(_mean_or_nan(actual_distance_ratio)),
        }
        return target_indices, stats


def format_child_target_selection_log(iteration, stats):
    fields = (
        "iter",
        "selected_sources",
        "structural_sources",
        "fallback_sources",
        "changed_sources",
        "changed_source_ratio",
        "target_overlap_ratio",
        "actual_G_mean",
        "baseline_G_mean",
        "actual_distance_ratio",
    )
    stats = {**stats, "iter": iteration}
    return "[ChildTargetSelection] " + " ".join(
        f"{field}={_format_diag_value(stats[field])}" for field in fields
    )


@dataclass
class GrowthDiagnostics:
    iteration: int
    num_before: int
    clone_candidates: int = 0
    split_gradient_candidates: int = 0
    split_sparse_candidates: int = 0
    split_total_candidates: int = 0
    proximity_sources: int = 0
    proximity_proposed: int = 0
    proximity_selected_sources: int = 0
    proximity_selected_new: int = 0
    num_after_clone: int = 0
    num_after_split: int = 0
    num_after_proximity: int = 0
    num_after_prune: int = 0

    @property
    def net_growth(self):
        return self.num_after_prune - self.num_before

    def format_log(self):
        return (
            f"[GrowthDiag] iter={self.iteration} "
            f"before={self.num_before} "
            f"clone={self.clone_candidates} "
            f"split_grad={self.split_gradient_candidates} "
            f"split_sparse={self.split_sparse_candidates} "
            f"split_total={self.split_total_candidates} "
            f"proximity_src={self.proximity_sources} "
            f"proximity_proposed={self.proximity_proposed} "
            f"proximity_selected_src={self.proximity_selected_sources} "
            f"proximity_new={self.proximity_selected_new} "
            f"after_clone={self.num_after_clone} "
            f"after_split={self.num_after_split} "
            f"after_proximity={self.num_after_proximity} "
            f"after_prune={self.num_after_prune} "
            f"net={self.net_growth}"
        )

"""Diagnostics for real OBDKR proximity value."""

from dataclasses import dataclass

import torch


STRUCTURAL_ATTRIBUTION_COMPONENTS = ("B", "K", "D", "R")


@dataclass
class StructuralValueAttributionStats:
    iteration: int
    candidates: int
    boundary_src: int
    promotions: int
    norm: dict
    boundary: dict
    promotion: dict


@torch.no_grad()
def compute_obdkr_diagnostics(
    components,
    candidate_mask=None,
    observation_count=None,
    recent_observation_count=None,
    active_observation_count=None,
    observation_source="lifetime",
    value_score_variant="obdkr",
    num_train_views=None,
):
    required = ("O", "B_norm", "K_norm", "D_norm", "R_norm", "U")
    for name in required:
        if name not in components:
            raise ValueError(f"components missing '{name}'.")
    utility = components["U"]
    if not torch.is_tensor(utility) or utility.ndim != 1:
        raise ValueError("U must be a Tensor[N].")
    n = utility.shape[0]
    device = utility.device
    candidate = (
        torch.ones((n,), dtype=torch.bool, device=device)
        if candidate_mask is None
        else candidate_mask.to(device=device, dtype=torch.bool)
    )
    if candidate.shape != (n,):
        raise ValueError("candidate_mask must be a BoolTensor[N].")

    stats = {
        "gaussians": n,
        "candidates": int(candidate.sum().item()),
        "finite_U_ratio": _ratio(int(torch.isfinite(utility).sum().item()), n),
        "observation_source": observation_source,
        "value_score_variant": value_score_variant,
    }
    for name in required:
        values = components[name].to(device=device, dtype=torch.float32)
        stats.update(_basic_stats(name, values))
        if name == "O":
            stats.update(_quantile_stats(name, values, (0.25, 0.50, 0.75)))
        if name == "U":
            stats.update(_quantile_stats(name, values, (0.25, 0.50, 0.75)))
            stats["candidate_U_mean"] = _masked_mean(values, candidate)
            stats["candidate_U_min"] = _masked_min(values, candidate)
            stats["candidate_U_max"] = _masked_max(values, candidate)
            stats["structural_value_mean"] = stats["U_mean"]
            stats["structural_value_min"] = stats["U_min"]
            stats["structural_value_max"] = stats["U_max"]
    stats.update(_observation_stats("obs", observation_count, n, device, candidate))
    stats.update(_observation_stats("recent_obs", recent_observation_count, n, device, candidate))
    stats.update(_observation_stats("active_obs", active_observation_count, n, device, candidate))
    train_view_count = int(num_train_views) if num_train_views is not None else None
    if recent_observation_count is not None and train_view_count is not None:
        recent_obs = recent_observation_count.to(device=device, dtype=torch.float32)
        saturated = recent_obs == float(train_view_count)
        stats["recent_obs_saturated_ratio"] = _ratio(int(saturated.sum().item()), n)
        stats["candidate_recent_obs_saturated_ratio"] = _ratio(
            int((saturated & candidate).sum().item()),
            int(candidate.sum().item()),
        )
    else:
        stats["recent_obs_saturated_ratio"] = 0.0
        stats["candidate_recent_obs_saturated_ratio"] = 0.0
    return stats


def format_obdkr_diagnostics_log(iteration, stats):
    return (
        f"[OBDKRDiag] iter={iteration} "
        f"value_score_variant={stats['value_score_variant']} "
        f"observation_source={stats['observation_source']} "
        f"gaussians={stats['gaussians']} "
        f"candidates={stats['candidates']} "
        f"O_mean={stats['O_mean']:.6f} O_min={stats['O_min']:.6f} O_max={stats['O_max']:.6f} "
        f"O_q25={stats['O_q25']:.6f} O_q50={stats['O_q50']:.6f} O_q75={stats['O_q75']:.6f} "
        f"B_mean={stats['B_norm_mean']:.6f} B_min={stats['B_norm_min']:.6f} B_max={stats['B_norm_max']:.6f} "
        f"K_mean={stats['K_norm_mean']:.6f} K_min={stats['K_norm_min']:.6f} K_max={stats['K_norm_max']:.6f} "
        f"D_mean={stats['D_norm_mean']:.6f} D_min={stats['D_norm_min']:.6f} D_max={stats['D_norm_max']:.6f} "
        f"R_mean={stats['R_norm_mean']:.6f} R_min={stats['R_norm_min']:.6f} R_max={stats['R_norm_max']:.6f} "
        f"U_mean={stats['U_mean']:.6f} U_min={stats['U_min']:.6f} U_max={stats['U_max']:.6f} "
        f"U_q25={stats['U_q25']:.6f} U_q50={stats['U_q50']:.6f} U_q75={stats['U_q75']:.6f} "
        f"finite_U_ratio={stats['finite_U_ratio']:.6f} "
        f"structural_value_mean={stats['structural_value_mean']:.6f} "
        f"structural_value_min={stats['structural_value_min']:.6f} "
        f"structural_value_max={stats['structural_value_max']:.6f} "
        f"candidate_U_mean={stats['candidate_U_mean']:.6f} "
        f"candidate_U_min={stats['candidate_U_min']:.6f} "
        f"candidate_U_max={stats['candidate_U_max']:.6f} "
        f"obs_mean={stats['obs_mean']:.6f} "
        f"obs_min={stats['obs_min']:.6f} "
        f"obs_max={stats['obs_max']:.6f} "
        f"obs_q25={stats['obs_q25']:.6f} "
        f"obs_q50={stats['obs_q50']:.6f} "
        f"obs_q75={stats['obs_q75']:.6f} "
        f"obs_unique_count={stats['obs_unique_count']} "
        f"candidate_obs_mean={stats['candidate_obs_mean']:.6f} "
        f"candidate_obs_min={stats['candidate_obs_min']:.6f} "
        f"candidate_obs_max={stats['candidate_obs_max']:.6f} "
        f"candidate_obs_unique_count={stats['candidate_obs_unique_count']} "
        f"recent_obs_mean={stats['recent_obs_mean']:.6f} "
        f"recent_obs_min={stats['recent_obs_min']:.6f} "
        f"recent_obs_max={stats['recent_obs_max']:.6f} "
        f"recent_obs_q25={stats['recent_obs_q25']:.6f} "
        f"recent_obs_q50={stats['recent_obs_q50']:.6f} "
        f"recent_obs_q75={stats['recent_obs_q75']:.6f} "
        f"recent_obs_unique_count={stats['recent_obs_unique_count']} "
        f"candidate_recent_obs_mean={stats['candidate_recent_obs_mean']:.6f} "
        f"candidate_recent_obs_min={stats['candidate_recent_obs_min']:.6f} "
        f"candidate_recent_obs_max={stats['candidate_recent_obs_max']:.6f} "
        f"candidate_recent_obs_unique_count={stats['candidate_recent_obs_unique_count']} "
        f"recent_obs_saturated_ratio={stats['recent_obs_saturated_ratio']:.6f} "
        f"candidate_recent_obs_saturated_ratio={stats['candidate_recent_obs_saturated_ratio']:.6f} "
        f"active_obs_mean={stats['active_obs_mean']:.6f} "
        f"active_obs_min={stats['active_obs_min']:.6f} "
        f"active_obs_max={stats['active_obs_max']:.6f} "
        f"active_obs_unique_count={stats['active_obs_unique_count']}"
    )


@torch.no_grad()
def compute_structural_value_attribution_stats(iteration, components, candidate_mask, budget_stats, eps=1e-8):
    required = ("B_raw", "K_raw", "D_raw", "R_raw", "U")
    for name in required:
        if name not in components:
            raise ValueError(f"components missing '{name}'.")
    normalized = {
        "B": _component_tensor(components, "B"),
        "K": _component_tensor(components, "K"),
        "D": _component_tensor(components, "D"),
        "R": _component_tensor(components, "R"),
        "U": components["U"],
    }
    n = normalized["U"].shape[0]
    device = normalized["U"].device
    if not torch.is_tensor(candidate_mask) or candidate_mask.shape != (n,):
        raise ValueError("candidate_mask must be a BoolTensor[N].")
    candidate = candidate_mask.to(device=device, dtype=torch.bool)
    candidates = int(candidate.sum().item())

    norm_stats = {}
    for prefix in STRUCTURAL_ATTRIBUTION_COMPONENTS:
        raw = components[f"{prefix}_raw"].to(device=device, dtype=torch.float32)
        norm = normalized[prefix].to(device=device, dtype=torch.float32)
        norm_stats.update(_raw_normalization_stats(prefix, raw, norm, candidate, eps=eps))

    boundary_stats = {}
    boundary_indices = _indices_tensor(getattr(budget_stats, "boundary_indices", ()), device)
    for prefix in ("B", "K", "D", "R", "U"):
        values = normalized[prefix].to(device=device, dtype=torch.float32)
        boundary_stats[f"{prefix}_mean"] = _indexed_mean(values, boundary_indices)
        boundary_stats[f"{prefix}_std"] = _indexed_std(values, boundary_indices)

    promoted = _indices_tensor(getattr(budget_stats, "demand_promoted_indices", ()), device)
    displaced = _indices_tensor(getattr(budget_stats, "demand_displaced_indices", ()), device)
    promotion_stats = _promotion_attribution_stats(normalized, promoted, displaced)
    promotion_stats["mean_P_ratio"] = float(getattr(budget_stats, "mean_P_ratio", float("nan")))
    promotion_stats["min_P_ratio"] = float(getattr(budget_stats, "min_P_ratio", float("nan")))
    promotion_stats["a1_overlap"] = float(getattr(budget_stats, "a1_overlap", float("nan")))

    return StructuralValueAttributionStats(
        iteration=iteration,
        candidates=candidates,
        boundary_src=int(getattr(budget_stats, "boundary_src", len(getattr(budget_stats, "boundary_indices", ())))),
        promotions=int(getattr(budget_stats, "promotion_accepted", promoted.numel())),
        norm=norm_stats,
        boundary=boundary_stats,
        promotion=promotion_stats,
    )


def format_structural_norm_diag(stats):
    parts = [
        "[StructuralNormDiag]",
        f"iter={stats.iteration}",
        f"candidates={stats.candidates}",
    ]
    for prefix in STRUCTURAL_ATTRIBUTION_COMPONENTS:
        parts.extend(
            [
                f"{prefix}_raw_q05={stats.norm[f'{prefix}_raw_q05']:.6f}",
                f"{prefix}_raw_q95={stats.norm[f'{prefix}_raw_q95']:.6f}",
                f"{prefix}_raw_span={stats.norm[f'{prefix}_raw_span']:.6f}",
                f"{prefix}_raw_finite_ratio={stats.norm[f'{prefix}_raw_finite_ratio']:.6f}",
                f"{prefix}_norm_zero_ratio={stats.norm[f'{prefix}_norm_zero_ratio']:.6f}",
                f"{prefix}_norm_one_ratio={stats.norm[f'{prefix}_norm_one_ratio']:.6f}",
                f"{prefix}_span_degenerate={stats.norm[f'{prefix}_span_degenerate']}",
            ]
        )
    return " ".join(parts)


def format_structural_boundary_diag(stats):
    parts = [
        "[StructuralBoundaryDiag]",
        f"iter={stats.iteration}",
        f"boundary_src={stats.boundary_src}",
    ]
    for prefix in ("B", "K", "D", "R", "U"):
        parts.extend(
            [
                f"{prefix}_mean={stats.boundary[f'{prefix}_mean']:.6f}",
                f"{prefix}_std={stats.boundary[f'{prefix}_std']:.6f}",
            ]
        )
    return " ".join(parts)


def format_structural_promotion_attr(stats):
    parts = [
        "[StructuralPromotionAttr]",
        f"iter={stats.iteration}",
        f"promotions={stats.promotions}",
    ]
    for prefix in STRUCTURAL_ATTRIBUTION_COMPONENTS + ("S", "U"):
        parts.extend(
            [
                f"mean_{prefix}_promoted={stats.promotion[f'mean_{prefix}_promoted']:.6f}",
                f"mean_{prefix}_displaced={stats.promotion[f'mean_{prefix}_displaced']:.6f}",
                f"mean_delta_{prefix}={stats.promotion[f'mean_delta_{prefix}']:.6f}",
            ]
        )
        if prefix != "U":
            parts.append(f"{prefix}_win_ratio={stats.promotion[f'{prefix}_win_ratio']:.6f}")
    parts.extend(
        [
            f"mean_P_ratio={stats.promotion['mean_P_ratio']:.6f}",
            f"min_P_ratio={stats.promotion['min_P_ratio']:.6f}",
            f"a1_overlap={stats.promotion['a1_overlap']:.6f}",
        ]
    )
    return " ".join(parts)


def _basic_stats(name, values):
    return {
        f"{name}_mean": _masked_mean(values),
        f"{name}_min": _masked_min(values),
        f"{name}_max": _masked_max(values),
    }


def _quantile_stats(name, values, quantiles):
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return {f"{name}_q{int(q * 100):02d}": 0.0 for q in quantiles}
    return {f"{name}_q{int(q * 100):02d}": float(torch.quantile(finite, q).item()) for q in quantiles}


def _masked_values(values, mask=None):
    if mask is not None:
        values = values[mask]
    return values[torch.isfinite(values)]


def _masked_mean(values, mask=None):
    finite = _masked_values(values, mask)
    return float(finite.mean().item()) if finite.numel() else 0.0


def _masked_min(values, mask=None):
    finite = _masked_values(values, mask)
    return float(finite.min().item()) if finite.numel() else 0.0


def _masked_max(values, mask=None):
    finite = _masked_values(values, mask)
    return float(finite.max().item()) if finite.numel() else 0.0


def _ratio(count, total):
    return float(count) / float(max(int(total), 1))


def _component_tensor(components, name):
    if name in components:
        return components[name]
    norm_name = f"{name}_norm"
    if norm_name in components:
        return components[norm_name]
    raise ValueError(f"components missing '{name}'.")


def _raw_normalization_stats(prefix, raw, normalized, candidate, eps=1e-8):
    selected_raw = raw[candidate]
    finite = selected_raw[torch.isfinite(selected_raw)]
    candidate_count = int(candidate.sum().item())
    finite_count = int(finite.numel())
    if finite_count == 0:
        q05 = float("nan")
        q95 = float("nan")
        span = float("nan")
    else:
        q05 = float(torch.quantile(finite, 0.05).item())
        q95 = float(torch.quantile(finite, 0.95).item())
        span = q95 - q05
    selected_norm = normalized[candidate].to(dtype=torch.float32)
    finite_norm = selected_norm[torch.isfinite(selected_norm)]
    zero_ratio = _ratio(int((finite_norm <= eps).sum().item()), candidate_count)
    one_ratio = _ratio(int((finite_norm >= 1.0 - eps).sum().item()), candidate_count)
    return {
        f"{prefix}_raw_q05": q05,
        f"{prefix}_raw_q95": q95,
        f"{prefix}_raw_span": span,
        f"{prefix}_raw_finite_ratio": _ratio(finite_count, candidate_count),
        f"{prefix}_norm_zero_ratio": zero_ratio,
        f"{prefix}_norm_one_ratio": one_ratio,
        f"{prefix}_span_degenerate": (not torch.isfinite(torch.tensor(span)).item()) or span <= eps,
    }


def _indices_tensor(indices, device):
    if torch.is_tensor(indices):
        return indices.detach().to(device=device, dtype=torch.long).reshape(-1)
    return torch.as_tensor(tuple(indices), dtype=torch.long, device=device)


def _indexed_mean(values, indices):
    if indices.numel() == 0:
        return float("nan")
    finite = values[indices][torch.isfinite(values[indices])]
    return float(finite.mean().item()) if finite.numel() else float("nan")


def _indexed_std(values, indices):
    if indices.numel() == 0:
        return float("nan")
    finite = values[indices][torch.isfinite(values[indices])]
    return float(finite.std(unbiased=False).item()) if finite.numel() else float("nan")


def _promotion_attribution_stats(normalized, promoted, displaced):
    stats = {}
    has_pairs = promoted.numel() > 0 and displaced.numel() > 0
    if promoted.numel() != displaced.numel():
        raise ValueError("promoted and displaced indices must be paired.")
    s = normalized["B"] + normalized["K"] + normalized["D"]
    values = {
        "B": normalized["B"],
        "K": normalized["K"],
        "D": normalized["D"],
        "R": normalized["R"],
        "S": s,
        "U": normalized["U"],
    }
    for prefix, tensor in values.items():
        if not has_pairs:
            stats[f"mean_{prefix}_promoted"] = float("nan")
            stats[f"mean_{prefix}_displaced"] = float("nan")
            stats[f"mean_delta_{prefix}"] = float("nan")
            if prefix != "U":
                stats[f"{prefix}_win_ratio"] = float("nan")
            continue
        promoted_values = tensor[promoted].to(dtype=torch.float32)
        displaced_values = tensor[displaced].to(dtype=torch.float32)
        delta = promoted_values - displaced_values
        stats[f"mean_{prefix}_promoted"] = _finite_tensor_mean(promoted_values)
        stats[f"mean_{prefix}_displaced"] = _finite_tensor_mean(displaced_values)
        stats[f"mean_delta_{prefix}"] = _finite_tensor_mean(delta)
        if prefix == "R":
            wins = promoted_values < displaced_values
        elif prefix != "U":
            wins = promoted_values > displaced_values
        if prefix != "U":
            finite_pairs = torch.isfinite(promoted_values) & torch.isfinite(displaced_values)
            stats[f"{prefix}_win_ratio"] = (
                float(wins[finite_pairs].to(dtype=torch.float32).mean().item())
                if finite_pairs.any()
                else float("nan")
            )
    return stats


def _finite_tensor_mean(values):
    finite = values[torch.isfinite(values)]
    return float(finite.mean().item()) if finite.numel() else float("nan")


def _unique_count(values):
    finite = values[torch.isfinite(values)]
    return int(torch.unique(finite).numel()) if finite.numel() else 0


def _observation_stats(prefix, observation_count, n, device, candidate):
    if observation_count is None:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_q25": 0.0,
            f"{prefix}_q50": 0.0,
            f"{prefix}_q75": 0.0,
            f"{prefix}_unique_count": 0,
            f"candidate_{prefix}_mean": 0.0,
            f"candidate_{prefix}_min": 0.0,
            f"candidate_{prefix}_max": 0.0,
            f"candidate_{prefix}_unique_count": 0,
        }
    if not torch.is_tensor(observation_count) or observation_count.shape != (n,):
        raise ValueError(f"{prefix} must be a Tensor[N].")
    obs = observation_count.to(device=device, dtype=torch.float32)
    stats = {
        f"{prefix}_mean": _masked_mean(obs),
        f"{prefix}_min": _masked_min(obs),
        f"{prefix}_max": _masked_max(obs),
        f"{prefix}_unique_count": _unique_count(obs),
        f"candidate_{prefix}_mean": _masked_mean(obs, candidate),
        f"candidate_{prefix}_min": _masked_min(obs, candidate),
        f"candidate_{prefix}_max": _masked_max(obs, candidate),
        f"candidate_{prefix}_unique_count": _unique_count(obs[candidate]),
    }
    stats.update(_quantile_stats(prefix, obs, (0.25, 0.50, 0.75)))
    return stats

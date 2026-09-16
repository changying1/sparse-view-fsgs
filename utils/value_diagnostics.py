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
    required = ("B_raw", "K_raw", "D_raw", "R_raw", "S", "U")
    for name in required:
        if name not in components:
            raise ValueError(f"components missing '{name}'.")
    normalized = {
        "B": _component_tensor(components, "B"),
        "K": _component_tensor(components, "K"),
        "D": _component_tensor(components, "D"),
        "R": _component_tensor(components, "R"),
        "S": components["S"],
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


@torch.no_grad()
def compute_gestalt_value_diagnostics(iteration, components, candidate_mask, budget_stats):
    required = ("G", "U_base", "U")
    for name in required:
        if name not in components:
            raise ValueError(f"components missing '{name}'.")
    utility = components["U"]
    if not torch.is_tensor(utility) or utility.ndim != 1:
        raise ValueError("U must be a Tensor[N].")
    n = utility.shape[0]
    device = utility.device
    if not torch.is_tensor(candidate_mask) or candidate_mask.shape != (n,):
        raise ValueError("candidate_mask must be a BoolTensor[N].")
    candidate = candidate_mask.to(device=device, dtype=torch.bool)
    g = components["G"].to(device=device, dtype=torch.float32)
    base_u = components["U_base"].to(device=device, dtype=torch.float32)
    gestalt_u = utility.to(device=device, dtype=torch.float32)
    promoted = _indices_tensor(getattr(budget_stats, "demand_promoted_indices", ()), device)
    displaced = _indices_tensor(getattr(budget_stats, "demand_displaced_indices", ()), device)
    if promoted.numel() != displaced.numel():
        raise ValueError("promoted and displaced indices must be paired.")

    stats = {
        "iteration": iteration,
        "candidates": int(candidate.sum().item()),
        "lambda_g": float(components.get("gestalt_lambda", 1.0)),
        "promotions": int(getattr(budget_stats, "promotion_accepted", promoted.numel())),
        "G_mean": _masked_mean(g, candidate),
        "G_min": _masked_min(g, candidate),
        "G_max": _masked_max(g, candidate),
        "base_U_mean": _masked_mean(base_u, candidate),
        "gestalt_U_mean": _masked_mean(gestalt_u, candidate),
    }
    stats.update(_masked_quantile_stats("G", g, candidate, (0.25, 0.50, 0.75)))
    if promoted.numel() == 0:
        stats.update({
            "mean_G_promoted": float("nan"),
            "mean_G_displaced": float("nan"),
            "mean_delta_G": float("nan"),
            "G_win_ratio": float("nan"),
            "mean_base_U_promoted": float("nan"),
            "mean_base_U_displaced": float("nan"),
            "mean_gestalt_U_promoted": float("nan"),
            "mean_gestalt_U_displaced": float("nan"),
        })
        return stats

    promoted_g = g[promoted]
    displaced_g = g[displaced]
    finite_pairs = torch.isfinite(promoted_g) & torch.isfinite(displaced_g)
    stats.update({
        "mean_G_promoted": _indexed_mean(g, promoted),
        "mean_G_displaced": _indexed_mean(g, displaced),
        "mean_delta_G": _finite_tensor_mean(promoted_g - displaced_g),
        "G_win_ratio": (
            float((promoted_g[finite_pairs] > displaced_g[finite_pairs]).to(dtype=torch.float32).mean().item())
            if finite_pairs.any()
            else float("nan")
        ),
        "mean_base_U_promoted": _indexed_mean(base_u, promoted),
        "mean_base_U_displaced": _indexed_mean(base_u, displaced),
        "mean_gestalt_U_promoted": _indexed_mean(gestalt_u, promoted),
        "mean_gestalt_U_displaced": _indexed_mean(gestalt_u, displaced),
    })
    return stats


def format_gestalt_value_diag(stats):
    return (
        f"[GestaltValueDiag] iter={stats['iteration']} "
        f"candidates={stats['candidates']} "
        f"lambda_g={stats['lambda_g']:.6f} "
        f"G_mean={stats['G_mean']:.6f} "
        f"G_min={stats['G_min']:.6f} "
        f"G_max={stats['G_max']:.6f} "
        f"G_q25={stats['G_q25']:.6f} "
        f"G_q50={stats['G_q50']:.6f} "
        f"G_q75={stats['G_q75']:.6f} "
        f"base_U_mean={stats['base_U_mean']:.6f} "
        f"gestalt_U_mean={stats['gestalt_U_mean']:.6f}"
    )


def format_gestalt_promotion_diag(stats):
    return (
        f"[GestaltPromotionDiag] iter={stats['iteration']} "
        f"promotions={stats['promotions']} "
        f"mean_G_promoted={stats['mean_G_promoted']:.6f} "
        f"mean_G_displaced={stats['mean_G_displaced']:.6f} "
        f"mean_delta_G={stats['mean_delta_G']:.6f} "
        f"G_win_ratio={stats['G_win_ratio']:.6f} "
        f"mean_base_U_promoted={stats['mean_base_U_promoted']:.6f} "
        f"mean_base_U_displaced={stats['mean_base_U_displaced']:.6f} "
        f"mean_gestalt_U_promoted={stats['mean_gestalt_U_promoted']:.6f} "
        f"mean_gestalt_U_displaced={stats['mean_gestalt_U_displaced']:.6f}"
    )


@torch.no_grad()
def compute_balanced_gestalt_diagnostics(iteration, components, candidate_mask, budget_stats):
    required = ("S_need", "G", "U")
    for name in required:
        if name not in components:
            raise ValueError(f"components missing '{name}'.")
    utility = components["U"]
    if not torch.is_tensor(utility) or utility.ndim != 1:
        raise ValueError("U must be a Tensor[N].")
    n = utility.shape[0]
    device = utility.device
    if not torch.is_tensor(candidate_mask) or candidate_mask.shape != (n,):
        raise ValueError("candidate_mask must be a BoolTensor[N].")
    candidate = candidate_mask.to(device=device, dtype=torch.bool)
    s_need = components["S_need"].to(device=device, dtype=torch.float32)
    g = components["G"].to(device=device, dtype=torch.float32)
    promoted = _indices_tensor(getattr(budget_stats, "demand_promoted_indices", ()), device)
    displaced = _indices_tensor(getattr(budget_stats, "demand_displaced_indices", ()), device)
    if promoted.numel() != displaced.numel():
        raise ValueError("promoted and displaced indices must be paired.")

    stats = {
        "iteration": iteration,
        "candidates": int(candidate.sum().item()),
        "alpha": float(components.get("gestalt_balance_alpha", 0.5)),
        "promotions": int(getattr(budget_stats, "promotion_accepted", promoted.numel())),
        "S_need_mean": _masked_mean(s_need, candidate),
        "G_mean": _masked_mean(g, candidate),
        "U_balanced_mean": _masked_mean(utility.to(device=device, dtype=torch.float32), candidate),
    }
    stats.update(_masked_quantile_stats("S_need", s_need, candidate, (0.25, 0.50, 0.75)))
    stats.update(_masked_quantile_stats("G", g, candidate, (0.25, 0.50, 0.75)))

    if promoted.numel() == 0:
        stats.update({
            "mean_S_need_promoted": float("nan"),
            "mean_S_need_displaced": float("nan"),
            "mean_delta_S_need": float("nan"),
            "S_need_win_ratio": float("nan"),
            "mean_G_promoted": float("nan"),
            "mean_G_displaced": float("nan"),
            "mean_delta_G": float("nan"),
            "G_win_ratio": float("nan"),
            "mean_U_promoted": float("nan"),
            "mean_U_displaced": float("nan"),
        })
        return stats

    promoted_s = s_need[promoted]
    displaced_s = s_need[displaced]
    promoted_g = g[promoted]
    displaced_g = g[displaced]
    finite_s_pairs = torch.isfinite(promoted_s) & torch.isfinite(displaced_s)
    finite_g_pairs = torch.isfinite(promoted_g) & torch.isfinite(displaced_g)
    stats.update({
        "mean_S_need_promoted": _indexed_mean(s_need, promoted),
        "mean_S_need_displaced": _indexed_mean(s_need, displaced),
        "mean_delta_S_need": _finite_tensor_mean(promoted_s - displaced_s),
        "S_need_win_ratio": (
            float((promoted_s[finite_s_pairs] > displaced_s[finite_s_pairs]).to(dtype=torch.float32).mean().item())
            if finite_s_pairs.any()
            else float("nan")
        ),
        "mean_G_promoted": _indexed_mean(g, promoted),
        "mean_G_displaced": _indexed_mean(g, displaced),
        "mean_delta_G": _finite_tensor_mean(promoted_g - displaced_g),
        "G_win_ratio": (
            float((promoted_g[finite_g_pairs] > displaced_g[finite_g_pairs]).to(dtype=torch.float32).mean().item())
            if finite_g_pairs.any()
            else float("nan")
        ),
        "mean_U_promoted": _indexed_mean(utility, promoted),
        "mean_U_displaced": _indexed_mean(utility, displaced),
    })
    return stats


def format_balanced_gestalt_diag(stats):
    return (
        f"[BalancedGestaltDiag] iter={stats['iteration']} "
        f"candidates={stats['candidates']} "
        f"alpha={stats['alpha']:.6f} "
        f"S_need_mean={stats['S_need_mean']:.6f} "
        f"S_need_q25={stats['S_need_q25']:.6f} "
        f"S_need_q50={stats['S_need_q50']:.6f} "
        f"S_need_q75={stats['S_need_q75']:.6f} "
        f"G_mean={stats['G_mean']:.6f} "
        f"G_q25={stats['G_q25']:.6f} "
        f"G_q50={stats['G_q50']:.6f} "
        f"G_q75={stats['G_q75']:.6f} "
        f"U_balanced_mean={stats['U_balanced_mean']:.6f}"
    )


def format_balanced_gestalt_promotion_diag(stats):
    return (
        f"[BalancedGestaltPromotionDiag] iter={stats['iteration']} "
        f"promotions={stats['promotions']} "
        f"mean_S_need_promoted={stats['mean_S_need_promoted']:.6f} "
        f"mean_S_need_displaced={stats['mean_S_need_displaced']:.6f} "
        f"mean_delta_S_need={stats['mean_delta_S_need']:.6f} "
        f"S_need_win_ratio={stats['S_need_win_ratio']:.6f} "
        f"mean_G_promoted={stats['mean_G_promoted']:.6f} "
        f"mean_G_displaced={stats['mean_G_displaced']:.6f} "
        f"mean_delta_G={stats['mean_delta_G']:.6f} "
        f"G_win_ratio={stats['G_win_ratio']:.6f} "
        f"mean_U_promoted={stats['mean_U_promoted']:.6f} "
        f"mean_U_displaced={stats['mean_U_displaced']:.6f}"
    )


@torch.no_grad()
def compute_defect_gestalt_diagnostics(iteration, components, candidate_mask, budget_stats):
    required = ("K", "D", "G", "DG", "U_base", "U")
    for name in required:
        if name not in components:
            raise ValueError(f"components missing '{name}'.")
    utility = components["U"]
    if not torch.is_tensor(utility) or utility.ndim != 1:
        raise ValueError("U must be a Tensor[N].")
    n = utility.shape[0]
    device = utility.device
    if not torch.is_tensor(candidate_mask) or candidate_mask.shape != (n,):
        raise ValueError("candidate_mask must be a BoolTensor[N].")
    candidate = candidate_mask.to(device=device, dtype=torch.bool)
    values = {
        "K": components["K"].to(device=device, dtype=torch.float32),
        "D": components["D"].to(device=device, dtype=torch.float32),
        "G": components["G"].to(device=device, dtype=torch.float32),
        "DG": components["DG"].to(device=device, dtype=torch.float32),
        "base_U": components["U_base"].to(device=device, dtype=torch.float32),
        "conditioned_U": utility.to(device=device, dtype=torch.float32),
    }
    promoted = _indices_tensor(getattr(budget_stats, "demand_promoted_indices", ()), device)
    displaced = _indices_tensor(getattr(budget_stats, "demand_displaced_indices", ()), device)
    if promoted.numel() != displaced.numel():
        raise ValueError("promoted and displaced indices must be paired.")

    stats = {
        "iteration": iteration,
        "candidates": int(candidate.sum().item()),
        "lambda_g": float(components.get("gestalt_lambda", 1.0)),
        "promotions": int(getattr(budget_stats, "promotion_accepted", promoted.numel())),
        "K_mean": _masked_mean(values["K"], candidate),
        "D_mean": _masked_mean(values["D"], candidate),
        "G_mean": _masked_mean(values["G"], candidate),
        "DG_mean": _masked_mean(values["DG"], candidate),
        "base_U_mean": _masked_mean(values["base_U"], candidate),
        "conditioned_U_mean": _masked_mean(values["conditioned_U"], candidate),
    }
    for name in ("D", "G", "DG"):
        stats.update(_masked_quantile_stats(name, values[name], candidate, (0.25, 0.50, 0.75)))

    for name in ("K", "D", "G", "DG"):
        tensor = values[name]
        if promoted.numel() == 0:
            stats[f"mean_{name}_promoted"] = float("nan")
            stats[f"mean_{name}_displaced"] = float("nan")
            stats[f"mean_delta_{name}"] = float("nan")
            stats[f"{name}_win_ratio"] = float("nan")
            continue
        promoted_values = tensor[promoted]
        displaced_values = tensor[displaced]
        finite_pairs = torch.isfinite(promoted_values) & torch.isfinite(displaced_values)
        stats[f"mean_{name}_promoted"] = _indexed_mean(tensor, promoted)
        stats[f"mean_{name}_displaced"] = _indexed_mean(tensor, displaced)
        stats[f"mean_delta_{name}"] = _finite_tensor_mean(promoted_values - displaced_values)
        stats[f"{name}_win_ratio"] = (
            float((promoted_values[finite_pairs] > displaced_values[finite_pairs]).to(dtype=torch.float32).mean().item())
            if finite_pairs.any()
            else float("nan")
        )

    if promoted.numel() == 0:
        stats["mean_base_U_promoted"] = float("nan")
        stats["mean_base_U_displaced"] = float("nan")
        stats["mean_conditioned_U_promoted"] = float("nan")
        stats["mean_conditioned_U_displaced"] = float("nan")
    else:
        stats["mean_base_U_promoted"] = _indexed_mean(values["base_U"], promoted)
        stats["mean_base_U_displaced"] = _indexed_mean(values["base_U"], displaced)
        stats["mean_conditioned_U_promoted"] = _indexed_mean(values["conditioned_U"], promoted)
        stats["mean_conditioned_U_displaced"] = _indexed_mean(values["conditioned_U"], displaced)
    return stats


def format_defect_gestalt_diag(stats):
    return (
        f"[DefectGestaltDiag] iter={stats['iteration']} "
        f"candidates={stats['candidates']} "
        f"lambda_g={stats['lambda_g']:.6f} "
        f"K_mean={stats['K_mean']:.6f} "
        f"D_mean={stats['D_mean']:.6f} "
        f"G_mean={stats['G_mean']:.6f} "
        f"DG_mean={stats['DG_mean']:.6f} "
        f"D_q25={stats['D_q25']:.6f} "
        f"D_q50={stats['D_q50']:.6f} "
        f"D_q75={stats['D_q75']:.6f} "
        f"G_q25={stats['G_q25']:.6f} "
        f"G_q50={stats['G_q50']:.6f} "
        f"G_q75={stats['G_q75']:.6f} "
        f"DG_q25={stats['DG_q25']:.6f} "
        f"DG_q50={stats['DG_q50']:.6f} "
        f"DG_q75={stats['DG_q75']:.6f} "
        f"base_U_mean={stats['base_U_mean']:.6f} "
        f"conditioned_U_mean={stats['conditioned_U_mean']:.6f}"
    )


def format_defect_gestalt_promotion_diag(stats):
    parts = [
        "[DefectGestaltPromotionDiag]",
        f"iter={stats['iteration']}",
        f"promotions={stats['promotions']}",
    ]
    for name in ("K", "D", "G", "DG"):
        parts.extend(
            [
                f"mean_{name}_promoted={stats[f'mean_{name}_promoted']:.6f}",
                f"mean_{name}_displaced={stats[f'mean_{name}_displaced']:.6f}",
                f"mean_delta_{name}={stats[f'mean_delta_{name}']:.6f}",
                f"{name}_win_ratio={stats[f'{name}_win_ratio']:.6f}",
            ]
        )
    parts.extend(
        [
            f"mean_base_U_promoted={stats['mean_base_U_promoted']:.6f}",
            f"mean_base_U_displaced={stats['mean_base_U_displaced']:.6f}",
            f"mean_conditioned_U_promoted={stats['mean_conditioned_U_promoted']:.6f}",
            f"mean_conditioned_U_displaced={stats['mean_conditioned_U_displaced']:.6f}",
        ]
    )
    return " ".join(parts)


@torch.no_grad()
def compute_defect_gestalt_counterfactual_diagnostics(
    iteration,
    components,
    candidate_mask,
    actual_selected_mask,
    actual_budget_stats,
    kd_selected_mask,
    kd_budget_stats,
    proximity_values,
):
    required = ("K", "D", "G", "DG", "U_base", "U", "gestalt_lambda")
    for name in required:
        if name not in components:
            raise ValueError(f"components missing '{name}'.")
    utility = components["U"]
    if not torch.is_tensor(utility) or utility.ndim != 1:
        raise ValueError("U must be a Tensor[N].")
    n = utility.shape[0]
    device = utility.device
    masks = {
        "candidate_mask": candidate_mask,
        "actual_selected_mask": actual_selected_mask,
        "kd_selected_mask": kd_selected_mask,
    }
    for name, mask in masks.items():
        if not torch.is_tensor(mask) or mask.shape != (n,):
            raise ValueError(f"{name} must be a BoolTensor[N].")
    if not torch.is_tensor(proximity_values) or proximity_values.shape != (n,):
        raise ValueError("proximity_values must be a Tensor[N].")

    candidate = candidate_mask.to(device=device, dtype=torch.bool)
    actual = actual_selected_mask.to(device=device, dtype=torch.bool)
    kd = kd_selected_mask.to(device=device, dtype=torch.bool)
    overlap = actual & kd
    swap_in_mask = actual & ~kd
    swap_out_mask = kd & ~actual
    swap_in_indices = torch.nonzero(swap_in_mask, as_tuple=False).reshape(-1)
    swap_out_indices = torch.nonzero(swap_out_mask, as_tuple=False).reshape(-1)
    conditioned_selected = int(actual.sum().item())
    kd_selected = int(kd.sum().item())
    overlap_count = int(overlap.sum().item())
    swap_in = int(swap_in_mask.sum().item())
    swap_out = int(swap_out_mask.sum().item())

    values = {
        "K": components["K"].to(device=device, dtype=torch.float32),
        "D": components["D"].to(device=device, dtype=torch.float32),
        "G": components["G"].to(device=device, dtype=torch.float32),
        "DG": components["DG"].to(device=device, dtype=torch.float32),
        "base_U": components["U_base"].to(device=device, dtype=torch.float32),
        "conditioned_U": utility.to(device=device, dtype=torch.float32),
        "P": proximity_values.to(device=device, dtype=torch.float32),
    }
    stats = {
        "iteration": iteration,
        "lambda_g": float(components["gestalt_lambda"]),
        "candidates": int(candidate.sum().item()),
        "kd_selected": kd_selected,
        "conditioned_selected": conditioned_selected,
        "overlap_count": overlap_count,
        "overlap_ratio": _safe_ratio(overlap_count, conditioned_selected),
        "swap_in": swap_in,
        "swap_out": swap_out,
        "swap_ratio": _safe_ratio(swap_in, conditioned_selected),
        "count_match": swap_in == swap_out,
        "actual_mode": getattr(actual_budget_stats, "mode", ""),
        "kd_mode": getattr(kd_budget_stats, "mode", ""),
    }
    for name, tensor in values.items():
        mean_in = _indexed_mean(tensor, swap_in_indices)
        mean_out = _indexed_mean(tensor, swap_out_indices)
        stats[f"mean_{name}_swap_in"] = mean_in
        stats[f"mean_{name}_swap_out"] = mean_out
        stats[f"mean_delta_{name}"] = mean_in - mean_out
    return stats


def format_defect_gestalt_counterfactual_diag(stats):
    return (
        f"[DefectGestaltCounterfactual] iter={stats['iteration']} "
        f"lambda_g={stats['lambda_g']:.6f} "
        f"candidates={stats['candidates']} "
        f"kd_selected={stats['kd_selected']} "
        f"conditioned_selected={stats['conditioned_selected']} "
        f"overlap_count={stats['overlap_count']} "
        f"overlap_ratio={stats['overlap_ratio']:.6f} "
        f"swap_in={stats['swap_in']} "
        f"swap_out={stats['swap_out']} "
        f"swap_ratio={stats['swap_ratio']:.6f} "
        f"count_match={stats['count_match']} "
        f"mean_K_swap_in={stats['mean_K_swap_in']:.6f} "
        f"mean_K_swap_out={stats['mean_K_swap_out']:.6f} "
        f"mean_delta_K={stats['mean_delta_K']:.6f} "
        f"mean_D_swap_in={stats['mean_D_swap_in']:.6f} "
        f"mean_D_swap_out={stats['mean_D_swap_out']:.6f} "
        f"mean_delta_D={stats['mean_delta_D']:.6f} "
        f"mean_G_swap_in={stats['mean_G_swap_in']:.6f} "
        f"mean_G_swap_out={stats['mean_G_swap_out']:.6f} "
        f"mean_delta_G={stats['mean_delta_G']:.6f} "
        f"mean_DG_swap_in={stats['mean_DG_swap_in']:.6f} "
        f"mean_DG_swap_out={stats['mean_DG_swap_out']:.6f} "
        f"mean_delta_DG={stats['mean_delta_DG']:.6f} "
        f"mean_base_U_swap_in={stats['mean_base_U_swap_in']:.6f} "
        f"mean_base_U_swap_out={stats['mean_base_U_swap_out']:.6f} "
        f"mean_delta_base_U={stats['mean_delta_base_U']:.6f} "
        f"mean_conditioned_U_swap_in={stats['mean_conditioned_U_swap_in']:.6f} "
        f"mean_conditioned_U_swap_out={stats['mean_conditioned_U_swap_out']:.6f} "
        f"mean_delta_conditioned_U={stats['mean_delta_conditioned_U']:.6f} "
        f"mean_P_swap_in={stats['mean_P_swap_in']:.6f} "
        f"mean_P_swap_out={stats['mean_P_swap_out']:.6f} "
        f"mean_delta_P={stats['mean_delta_P']:.6f}"
    )


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


def _masked_quantile_stats(name, values, mask, quantiles):
    finite = _masked_values(values, mask)
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


def _safe_ratio(count, total):
    return float(count) / float(total) if int(total) > 0 else 0.0


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
    values = {
        "B": normalized["B"],
        "K": normalized["K"],
        "D": normalized["D"],
        "R": normalized["R"],
        "S": normalized["S"],
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

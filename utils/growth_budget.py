from dataclasses import dataclass
from math import ceil, floor, isfinite

import torch


PROXIMITY_SELECTION_MODES = (
    "original",
    "proximity_topk",
    "value_global",
    "value_rerank",
    "value_demand_rerank",
)



def resolve_proximity_selection_mode(mode, budget_enabled=False):
    selection_mode = mode or "original"
    if selection_mode not in PROXIMITY_SELECTION_MODES:
        raise ValueError(f"Unknown proximity_selection_mode '{selection_mode}'")
    if budget_enabled and selection_mode == "original":
        return "proximity_topk"
    return selection_mode


def requires_value_features(mode):
    return mode in ("value_global", "value_rerank", "value_demand_rerank")


def resolve_value_timing_selection_mode(mode, iteration=None, start_iter=0, end_iter=None):
    selection_mode = resolve_proximity_selection_mode(mode, False)
    start_iter = int(start_iter)
    end_iter = int(end_iter) if end_iter is not None else None
    if end_iter is not None and start_iter > end_iter:
        raise ValueError("value_rerank_start_iter must be <= value_rerank_end_iter")
    if selection_mode != "value_demand_rerank":
        return selection_mode, False
    if iteration is None:
        return selection_mode, True
    iteration = int(iteration)
    active = iteration >= start_iter and (end_iter is None or iteration <= end_iter)
    effective_mode = selection_mode if active else "proximity_topk"
    return effective_mode, active


@dataclass
class ProximityBudgetStats:
    current: int
    candidates: int
    proposed: int
    budget_new: int
    budget_src: int
    selected_src: int
    selected_new: int
    dropped_src: int
    budget_active: bool
    budget_hit: bool
    mode: str = "original"
    core_src: int = 0
    boundary_src: int = 0
    rerank_slots: int = 0
    selected_value_src: int = 0
    a1_overlap: float = 1.0
    mean_P_all: float = float("nan")
    mean_P_selected: float = float("nan")
    mean_U_all: float = float("nan")
    mean_U_boundary: float = float("nan")
    mean_U_selected: float = float("nan")
    proximity_rank: tuple = ()
    a1_indices: tuple = ()
    selected_indices: tuple = ()
    boundary_indices: tuple = ()
    value_selected_indices: tuple = ()
    demand_ratio_threshold: float = 0.90
    baseline_boundary_src: int = 0
    challenger_src: int = 0
    promotion_attempted: int = 0
    promotion_accepted: int = 0
    promotion_rejected_demand: int = 0
    promotion_rejected_value: int = 0
    demand_promoted_indices: tuple = ()
    demand_displaced_indices: tuple = ()
    mean_P_ratio: float = float("nan")
    min_P_ratio: float = float("nan")
    max_P_ratio: float = float("nan")
    mean_P_gap: float = float("nan")
    max_P_gap: float = float("nan")
    mean_relative_P_gap: float = float("nan")
    demand_mean_U_promoted: float = float("nan")
    demand_mean_U_displaced: float = float("nan")
    capacity_active: bool = False
    capacity_src: int = 0
    keep_ratio: float = 1.0
    capacity_hit: bool = False
    replay_active: bool = False
    replay_scheduled_src: int = 0
    replay_exact_match: bool = False
    requested_mode: str = "original"
    value_timing_active: bool = False
    value_rerank_start_iter: int = 0
    value_rerank_end_iter: int = None


def parse_proximity_action_replay(spec):
    if spec is None or str(spec).strip() == "":
        return {}
    schedule = {}
    for raw_entry in str(spec).split(","):
        entry = raw_entry.strip()
        if not entry or entry.count(":") != 1:
            raise ValueError(f"Malformed proximity_action_replay entry '{entry}'")
        raw_iteration, raw_selected = (part.strip() for part in entry.split(":", 1))
        try:
            iteration = int(raw_iteration)
            selected_src = int(raw_selected)
        except ValueError as exc:
            raise ValueError(f"Malformed proximity_action_replay entry '{entry}'") from exc
        if str(iteration) != raw_iteration:
            raise ValueError(f"Malformed proximity_action_replay iteration '{raw_iteration}'")
        if str(selected_src) != raw_selected:
            raise ValueError(f"Malformed proximity_action_replay selected_src '{raw_selected}'")
        if iteration <= 0:
            raise ValueError("proximity_action_replay iteration must be a positive integer")
        if selected_src < 0:
            raise ValueError("proximity_action_replay selected_src must be a non-negative integer")
        if iteration in schedule:
            raise ValueError(f"Duplicate proximity_action_replay iteration {iteration}")
        schedule[iteration] = selected_src
    return schedule


def validate_proximity_growth_ratio(rho):
    if not (0 < float(rho) <= 1):
        raise ValueError("proximity_growth_ratio must satisfy 0 < rho <= 1")


def validate_proximity_candidate_keep_ratio(kappa):
    if not (0 < float(kappa) <= 1):
        raise ValueError("proximity_candidate_keep_ratio must satisfy 0 < ratio <= 1")


def validate_proximity_capacity_configuration(
    budget_enabled=False,
    candidate_capacity_enabled=False,
    keep_ratio=0.80,
    replay_enabled=False,
):
    enabled_count = sum(
        int(bool(flag))
        for flag in (budget_enabled, candidate_capacity_enabled, replay_enabled)
    )
    if enabled_count > 1:
        raise ValueError(
            "enable_proximity_budget, enable_proximity_candidate_capacity, "
            "and proximity_action_replay are mutually exclusive; cannot both be enabled"
        )
    if candidate_capacity_enabled:
        validate_proximity_candidate_keep_ratio(keep_ratio)


def compute_proximity_budget(current, candidates, n=3, rho=0.10, enabled=False):
    current = int(current)
    candidates = int(candidates)
    n = int(n)
    proposed = candidates * n

    if not enabled:
        return ProximityBudgetStats(
            current=current,
            candidates=candidates,
            proposed=proposed,
            budget_new=proposed,
            budget_src=candidates,
            selected_src=candidates,
            selected_new=proposed,
            dropped_src=0,
            budget_active=False,
            budget_hit=False,
        )

    validate_proximity_growth_ratio(rho)
    budget_new = floor(float(rho) * current)
    max_sources = budget_new // n
    selected_src = min(candidates, max_sources)
    selected_new = selected_src * n

    return ProximityBudgetStats(
        current=current,
        candidates=candidates,
        proposed=proposed,
        budget_new=budget_new,
        budget_src=max_sources,
        selected_src=selected_src,
        selected_new=selected_new,
        dropped_src=candidates - selected_src,
        budget_active=True,
        budget_hit=selected_src < candidates,
    )


def compute_proximity_candidate_capacity(current, candidates, n=3, keep_ratio=0.80, enabled=False):
    current = int(current)
    candidates = int(candidates)
    n = int(n)
    proposed = candidates * n

    if not enabled:
        return ProximityBudgetStats(
            current=current,
            candidates=candidates,
            proposed=proposed,
            budget_new=proposed,
            budget_src=candidates,
            selected_src=candidates,
            selected_new=proposed,
            dropped_src=0,
            budget_active=False,
            budget_hit=False,
            capacity_active=False,
            capacity_src=candidates,
            keep_ratio=float(keep_ratio),
            capacity_hit=False,
        )

    validate_proximity_candidate_keep_ratio(keep_ratio)
    capacity_src = 0 if candidates == 0 else max(1, floor(float(keep_ratio) * candidates))
    selected_src = min(candidates, capacity_src)
    selected_new = selected_src * n

    return ProximityBudgetStats(
        current=current,
        candidates=candidates,
        proposed=proposed,
        budget_new=selected_new,
        budget_src=selected_src,
        selected_src=selected_src,
        selected_new=selected_new,
        dropped_src=candidates - selected_src,
        budget_active=False,
        budget_hit=False,
        capacity_active=True,
        capacity_src=capacity_src,
        keep_ratio=float(keep_ratio),
        capacity_hit=selected_src < candidates,
    )


def compute_proximity_action_replay(current, candidates, iteration, schedule, n=3):
    current = int(current)
    candidates = int(candidates)
    n = int(n)
    proposed = candidates * n
    replay_schedule = schedule or {}
    if iteration is None:
        raise ValueError("proximity_action_replay requires a proximity iteration")
    iteration = int(iteration)
    if iteration not in replay_schedule:
        raise ValueError(f"Missing proximity_action_replay entry for iteration={iteration}")
    scheduled_src = int(replay_schedule[iteration])
    if scheduled_src > candidates:
        raise ValueError(
            f"Replay selected_src={scheduled_src} exceeds current candidates={candidates} "
            f"at iteration={iteration}"
        )
    selected_new = scheduled_src * n
    return ProximityBudgetStats(
        current=current,
        candidates=candidates,
        proposed=proposed,
        budget_new=selected_new,
        budget_src=scheduled_src,
        selected_src=scheduled_src,
        selected_new=selected_new,
        dropped_src=candidates - scheduled_src,
        budget_active=False,
        budget_hit=False,
        replay_active=True,
        replay_scheduled_src=scheduled_src,
        replay_exact_match=True,
    )


def select_proximity_sources(
    candidate_mask,
    dist,
    n=3,
    rho=0.10,
    enabled=False,
    mode=None,
    value_score=None,
    rerank_fraction=0.25,
    boundary_multiplier=2.0,
    demand_ratio=0.90,
    candidate_capacity_enabled=False,
    candidate_keep_ratio=0.80,
    replay_schedule=None,
    iteration=None,
    value_rerank_start_iter=0,
    value_rerank_end_iter=None,
):
    candidates = int(candidate_mask.sum().item())
    replay_enabled = bool(replay_schedule)
    validate_proximity_capacity_configuration(
        budget_enabled=enabled,
        candidate_capacity_enabled=candidate_capacity_enabled,
        keep_ratio=candidate_keep_ratio,
        replay_enabled=replay_enabled,
    )
    selection_mode = resolve_proximity_selection_mode(
        mode,
        bool(enabled) or bool(candidate_capacity_enabled) or replay_enabled,
    )
    requested_mode = selection_mode
    selection_mode, value_timing_active = resolve_value_timing_selection_mode(
        selection_mode,
        iteration=iteration,
        start_iter=value_rerank_start_iter,
        end_iter=value_rerank_end_iter,
    )
    if selection_mode != "original":
        if not candidate_capacity_enabled and not replay_enabled:
            enabled = True
    validate_proximity_selection_parameters(
        mode=selection_mode,
        rho=rho,
        rerank_fraction=rerank_fraction,
        boundary_multiplier=boundary_multiplier,
        demand_ratio=demand_ratio,
        budget_enabled=enabled,
        candidate_capacity_enabled=candidate_capacity_enabled,
        candidate_keep_ratio=candidate_keep_ratio,
        replay_enabled=replay_enabled,
    )
    if candidate_capacity_enabled:
        stats = compute_proximity_candidate_capacity(
            current=candidate_mask.shape[0],
            candidates=candidates,
            n=n,
            keep_ratio=candidate_keep_ratio,
            enabled=True,
        )
    elif replay_enabled:
        stats = compute_proximity_action_replay(
            current=candidate_mask.shape[0],
            candidates=candidates,
            iteration=iteration,
            schedule=replay_schedule,
            n=n,
        )
    else:
        stats = compute_proximity_budget(
            current=candidate_mask.shape[0],
            candidates=candidates,
            n=n,
            rho=rho,
            enabled=enabled,
        )
    stats.mode = selection_mode
    stats.requested_mode = requested_mode
    stats.value_timing_active = value_timing_active
    stats.value_rerank_start_iter = int(value_rerank_start_iter)
    stats.value_rerank_end_iter = int(value_rerank_end_iter) if value_rerank_end_iter is not None else None

    candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).reshape(-1)
    proximity_rank = _rank_indices(dist, candidate_indices, descending=True)
    stats.proximity_rank = _to_tuple(proximity_rank)
    stats.a1_indices = _to_tuple(proximity_rank[:stats.selected_src])
    stats.proximity_values = dist
    stats.value_values = value_score
    stats.mean_P_all = _masked_mean(dist, candidate_indices)

    capacity_limited = stats.budget_hit or stats.capacity_hit or (
        stats.replay_active and stats.selected_src < stats.candidates
    )
    if (
        (not enabled and not candidate_capacity_enabled and not stats.replay_active)
        or selection_mode == "original"
        or not capacity_limited
    ):
        stats.selected_indices = _to_tuple(candidate_indices)
        stats.mean_P_selected = _masked_mean(dist, candidate_indices)
        stats.mean_U_all = _masked_mean(value_score, candidate_indices)
        stats.mean_U_selected = _masked_mean(value_score, candidate_indices)
        return candidate_mask, stats

    selected_mask = torch.zeros_like(candidate_mask)
    if stats.selected_src == 0:
        return selected_mask, stats

    if selection_mode == "proximity_topk":
        selected_indices = proximity_rank[:stats.selected_src]
    elif selection_mode == "value_global":
        selected_indices, value_selected_indices = _value_select_with_fallback(
            value_score,
            proximity_rank,
            candidate_indices,
            stats.selected_src,
        )
        stats.value_selected_indices = _to_tuple(value_selected_indices)
        stats.selected_value_src = int(value_selected_indices.numel())
    elif selection_mode == "value_rerank":
        selected_indices = _select_value_rerank(
            proximity_rank,
            value_score,
            stats,
            rerank_fraction=rerank_fraction,
            boundary_multiplier=boundary_multiplier,
        )
    elif selection_mode == "value_demand_rerank":
        selected_indices = _select_value_demand_rerank(
            proximity_rank,
            dist,
            value_score,
            stats,
            rerank_fraction=rerank_fraction,
            boundary_multiplier=boundary_multiplier,
            demand_ratio=demand_ratio,
        )
    else:
        selected_indices = candidate_indices

    selected_mask[selected_indices] = True
    selected_count = int(selected_mask.sum().item())
    stats.selected_src = selected_count
    stats.selected_new = selected_count * int(n)
    stats.dropped_src = candidates - selected_count
    if stats.replay_active:
        stats.replay_exact_match = selected_count == stats.replay_scheduled_src
    stats.selected_indices = _to_tuple(selected_indices)
    stats.mean_P_selected = _masked_mean(dist, selected_indices)
    stats.mean_U_all = _masked_mean(value_score, candidate_indices)
    stats.mean_U_selected = _masked_mean(value_score, selected_indices)
    stats.a1_overlap = _overlap_ratio(stats.a1_indices, stats.selected_indices, stats.selected_src)
    return selected_mask, stats


def format_proximity_budget_log(iteration, stats):
    return (
        f"[ProximityBudget] iter={iteration} "
        f"current={stats.current} "
        f"candidates={stats.candidates} "
        f"proposed={stats.proposed} "
        f"budget={stats.budget_new} "
        f"budget_src={stats.budget_src} "
        f"selected_src={stats.selected_src} "
        f"selected_new={stats.selected_new} "
        f"dropped_src={stats.dropped_src} "
        f"budget_active={stats.budget_active} "
        f"budget_hit={stats.budget_hit}"
    )


def format_proximity_capacity_log(iteration, stats):
    retention = stats.selected_src / float(stats.candidates) if stats.candidates else 0.0
    return (
        f"[ProximityCapacity] iter={iteration} "
        f"current={stats.current} "
        f"candidates={stats.candidates} "
        f"keep_ratio={stats.keep_ratio:.6f} "
        f"capacity_src={stats.capacity_src} "
        f"selected_src={stats.selected_src} "
        f"selected_new={stats.selected_new} "
        f"dropped_src={stats.dropped_src} "
        f"retention={retention:.6f} "
        f"capacity_hit={stats.capacity_hit}"
    )


def format_proximity_replay_log(iteration, stats):
    retention = stats.selected_src / float(stats.candidates) if stats.candidates else 0.0
    return (
        f"[ProximityReplay] iter={iteration} "
        f"current={stats.current} "
        f"candidates={stats.candidates} "
        f"scheduled_src={stats.replay_scheduled_src} "
        f"selected_src={stats.selected_src} "
        f"selected_new={stats.selected_new} "
        f"dropped_src={stats.dropped_src} "
        f"retention={retention:.6f} "
        f"replay_active={stats.replay_active} "
        f"exact_match={stats.replay_exact_match}"
    )


def format_value_timing_log(iteration, stats):
    return (
        f"[ValueTiming] iter={iteration} "
        f"active={stats.value_timing_active} "
        f"start={stats.value_rerank_start_iter} "
        f"end={stats.value_rerank_end_iter} "
        f"effective_mode={stats.mode}"
    )


def format_value_rerank_diag(iteration, stats):
    return (
        f"[ValueRerankDiag] iter={iteration} "
        f"mode={stats.mode} "
        f"current={stats.current} "
        f"candidates={stats.candidates} "
        f"proposed_new={stats.proposed} "
        f"budget_new={stats.budget_new} "
        f"budget_src={stats.budget_src} "
        f"budget_active={stats.budget_active} "
        f"budget_hit={stats.budget_hit} "
        f"selected_src={stats.selected_src} "
        f"selected_new={stats.selected_new} "
        f"dropped_src={stats.dropped_src} "
        f"core_src={stats.core_src} "
        f"boundary_src={stats.boundary_src} "
        f"rerank_slots={stats.rerank_slots} "
        f"selected_value_src={stats.selected_value_src} "
        f"a1_overlap={stats.a1_overlap:.6f} "
        f"mean_P_all={stats.mean_P_all:.6f} "
        f"mean_P_selected={stats.mean_P_selected:.6f} "
        f"mean_U_all={stats.mean_U_all:.6f} "
        f"mean_U_boundary={stats.mean_U_boundary:.6f} "
        f"mean_U_selected={stats.mean_U_selected:.6f}"
    )


def format_value_promotion_diag(iteration, stats):
    promoted = sorted(set(stats.selected_indices) - set(stats.a1_indices))
    displaced = sorted(set(stats.a1_indices) - set(stats.selected_indices))
    rank_lookup = {idx: rank + 1 for rank, idx in enumerate(stats.proximity_rank)}
    return (
        f"[ValuePromotionDiag] iter={iteration} "
        f"promoted={len(promoted)} "
        f"displaced={len(displaced)} "
        f"promoted_rank_mean={_mean_python([rank_lookup[i] for i in promoted]):.6f} "
        f"promoted_rank_min={min([rank_lookup[i] for i in promoted], default=0)} "
        f"promoted_rank_max={max([rank_lookup[i] for i in promoted], default=0)} "
        f"mean_U_promoted={_mean_tuple_values(stats, promoted, 'value'):.6f} "
        f"mean_U_displaced={_mean_tuple_values(stats, displaced, 'value'):.6f} "
        f"mean_P_promoted={_mean_tuple_values(stats, promoted, 'proximity'):.6f} "
        f"mean_P_displaced={_mean_tuple_values(stats, displaced, 'proximity'):.6f}"
    )


def format_demand_preserve_diag(iteration, stats):
    return (
        f"[DemandPreserveDiag] iter={iteration} "
        f"mode={stats.mode} "
        f"demand_ratio_threshold={stats.demand_ratio_threshold:.6f} "
        f"baseline_boundary_src={stats.baseline_boundary_src} "
        f"challenger_src={stats.challenger_src} "
        f"promotion_attempted={stats.promotion_attempted} "
        f"promotion_accepted={stats.promotion_accepted} "
        f"promotion_rejected_demand={stats.promotion_rejected_demand} "
        f"promotion_rejected_value={stats.promotion_rejected_value} "
        f"promoted={len(stats.demand_promoted_indices)} "
        f"displaced={len(stats.demand_displaced_indices)} "
        f"mean_P_promoted={_mean_tuple_values(stats, stats.demand_promoted_indices, 'proximity'):.6f} "
        f"mean_P_displaced={_mean_tuple_values(stats, stats.demand_displaced_indices, 'proximity'):.6f} "
        f"mean_P_ratio={stats.mean_P_ratio:.6f} "
        f"min_P_ratio={stats.min_P_ratio:.6f} "
        f"max_P_ratio={stats.max_P_ratio:.6f} "
        f"mean_P_gap={stats.mean_P_gap:.6f} "
        f"max_P_gap={stats.max_P_gap:.6f} "
        f"mean_relative_P_gap={stats.mean_relative_P_gap:.6f} "
        f"mean_U_promoted={stats.demand_mean_U_promoted:.6f} "
        f"mean_U_displaced={stats.demand_mean_U_displaced:.6f} "
        f"a1_overlap={stats.a1_overlap:.6f}"
    )


def validate_proximity_selection_parameters(
    mode,
    rho=0.10,
    rerank_fraction=0.25,
    boundary_multiplier=2.0,
    demand_ratio=0.90,
    budget_enabled=False,
    candidate_capacity_enabled=False,
    candidate_keep_ratio=0.80,
    replay_enabled=False,
):
    if mode not in PROXIMITY_SELECTION_MODES:
        raise ValueError(f"Unknown proximity_selection_mode '{mode}'")
    validate_proximity_capacity_configuration(
        budget_enabled=budget_enabled,
        candidate_capacity_enabled=candidate_capacity_enabled,
        keep_ratio=candidate_keep_ratio,
        replay_enabled=replay_enabled,
    )
    if mode == "original" and not budget_enabled and not candidate_capacity_enabled and not replay_enabled:
        return
    if not candidate_capacity_enabled and not replay_enabled:
        validate_proximity_growth_ratio(rho)
    if not (0.0 <= float(rerank_fraction) <= 1.0):
        raise ValueError("value_rerank_fraction must satisfy 0 <= value_rerank_fraction <= 1")
    if not isfinite(float(boundary_multiplier)) or float(boundary_multiplier) < 1.0:
        raise ValueError("value_boundary_multiplier must be at least 1")
    if not (0.0 < float(demand_ratio) <= 1.0):
        raise ValueError("value_demand_ratio must satisfy 0 < value_demand_ratio <= 1")


def robust_normalize(values, low_quantile=0.05, high_quantile=0.95, eps=1e-8):
    if not torch.is_tensor(values) or values.ndim != 1:
        raise ValueError("values must be a Tensor[N]")
    if not (0.0 <= float(low_quantile) < float(high_quantile) <= 1.0):
        raise ValueError("Require 0 <= low_quantile < high_quantile <= 1")
    output = torch.zeros_like(values, dtype=torch.float32)
    finite = torch.isfinite(values)
    if not finite.any():
        return output
    finite_values = values[finite].float()
    q_low = torch.quantile(finite_values, float(low_quantile))
    q_high = torch.quantile(finite_values, float(high_quantile))
    denom = q_high - q_low
    if not torch.isfinite(denom) or denom.abs() <= eps:
        return output
    normalized = ((values.float() - q_low) / (denom + eps)).clamp(0.0, 1.0)
    output[finite] = normalized[finite]
    return torch.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)


def compute_observation_scarcity(counts, tau_e=1.0, tau_s=3.0):
    if tau_e <= 0 or tau_s <= 0:
        raise ValueError("tau_e and tau_s must be positive")
    n = counts.float().clamp_min(0.0)
    return torch.nan_to_num((1.0 - torch.exp(-n / float(tau_e))) * torch.exp(-n / float(tau_s)))


def compute_proximity_value_score(observation_count, boundary, turning, defect, redundancy,
                                  tau_e=1.0, tau_s=3.0, w_b=1.0, w_k=1.0, w_d=1.0,
                                  lambda_r=1.0, low_quantile=0.05, high_quantile=0.95):
    o = compute_observation_scarcity(observation_count, tau_e=tau_e, tau_s=tau_s)
    b = robust_normalize(boundary, low_quantile, high_quantile)
    k = robust_normalize(turning, low_quantile, high_quantile)
    d = robust_normalize(defect, low_quantile, high_quantile)
    r = robust_normalize(redundancy, low_quantile, high_quantile)
    structural = float(w_b) * b + float(w_k) * k + float(w_d) * d
    utility = (o * structural) / (1.0 + float(lambda_r) * r).clamp_min(1e-8)
    return torch.nan_to_num(utility, nan=0.0, posinf=0.0, neginf=0.0)


def build_fsgs_available_value_score(dist, scaling, confidence=None, denom=None, **kwargs):
    """Deprecated surrogate kept only for old diagnostics/tests.

    Formal value_global/value_rerank training must pass real OBDKR utility from
    utils.value_allocation instead of calling this helper.
    """
    observation = torch.ones_like(dist, dtype=torch.float32)
    if denom is not None:
        observation = denom.reshape(-1).to(device=dist.device, dtype=torch.float32).clamp_min(0.0)
    boundary = dist.float()
    turning = torch.max(scaling, dim=1).values.float()
    defect = torch.zeros_like(boundary)
    redundancy = torch.zeros_like(boundary) if confidence is None else (1.0 - confidence.reshape(-1).float()).clamp_min(0.0)
    return compute_proximity_value_score(observation, boundary, turning, defect, redundancy, **kwargs)


def _select_value_rerank(proximity_rank, value_score, stats, rerank_fraction=0.25, boundary_multiplier=2.0):
    if float(rerank_fraction) == 0.0:
        return proximity_rank[:stats.selected_src]
    rerank_slots = max(1, floor(float(rerank_fraction) * stats.selected_src))
    rerank_slots = min(rerank_slots, stats.selected_src)
    core_count = stats.selected_src - rerank_slots
    core = proximity_rank[:core_count]
    boundary_count = ceil(float(boundary_multiplier) * rerank_slots)
    boundary = proximity_rank[core_count:min(core_count + boundary_count, proximity_rank.numel())]
    value_selected, raw_value_selected = _value_select_with_fallback(value_score, proximity_rank, boundary, rerank_slots)
    selected = torch.cat((core, value_selected), dim=0)
    if selected.numel() < stats.selected_src:
        selected = _fill_from_rank(selected, proximity_rank, stats.selected_src)
    stats.core_src = int(core.numel())
    stats.boundary_src = int(boundary.numel())
    stats.rerank_slots = int(rerank_slots)
    stats.selected_value_src = int(raw_value_selected.numel())
    stats.boundary_indices = _to_tuple(boundary)
    stats.value_selected_indices = _to_tuple(raw_value_selected)
    stats.mean_U_boundary = _masked_mean(value_score, boundary)
    return selected[:stats.selected_src]


def _select_value_demand_rerank(
    proximity_rank,
    proximity_score,
    value_score,
    stats,
    rerank_fraction=0.25,
    boundary_multiplier=2.0,
    demand_ratio=0.90,
    eps=1e-8,
):
    stats.demand_ratio_threshold = float(demand_ratio)
    if float(rerank_fraction) == 0.0:
        return proximity_rank[:stats.selected_src]

    rerank_slots = max(1, floor(float(rerank_fraction) * stats.selected_src))
    rerank_slots = min(rerank_slots, stats.selected_src)
    core_count = stats.selected_src - rerank_slots
    core = proximity_rank[:core_count]
    boundary_count = ceil(float(boundary_multiplier) * rerank_slots)
    boundary = proximity_rank[core_count:min(core_count + boundary_count, proximity_rank.numel())]
    baseline_boundary = boundary[:rerank_slots]
    challengers = boundary[rerank_slots:]

    stats.core_src = int(core.numel())
    stats.boundary_src = int(boundary.numel())
    stats.rerank_slots = int(rerank_slots)
    stats.boundary_indices = _to_tuple(boundary)
    stats.baseline_boundary_src = int(baseline_boundary.numel())
    stats.challenger_src = int(challengers.numel())
    stats.mean_U_boundary = _masked_mean(value_score, boundary)

    if value_score is None or boundary.numel() == 0 or baseline_boundary.numel() == 0 or challengers.numel() == 0:
        return _fill_from_rank(torch.cat((core, baseline_boundary), dim=0), proximity_rank, stats.selected_src)

    value = value_score.detach().float()
    proximity = proximity_score.detach().float()
    rank_lookup = {idx: rank for rank, idx in enumerate(_to_tuple(proximity_rank))}
    final_boundary = [int(i) for i in _to_tuple(baseline_boundary)]
    unused_challengers = set(_to_tuple(challengers))
    promoted = []
    displaced = []
    ratios = []
    gaps = []
    relative_gaps = []

    for slot in range(len(final_boundary) - 1, -1, -1):
        d = final_boundary[slot]
        u_d = float(value[d].item())
        p_d = float(proximity[d].item())
        candidates = []
        value_reject = True
        demand_reject = False

        for c in unused_challengers:
            u_c = float(value[c].item())
            p_c = float(proximity[c].item())
            if not isfinite(u_c) or u_c <= 0.0:
                continue
            if not isfinite(u_d) or u_c <= u_d:
                continue
            value_reject = False
            if p_c + eps < float(demand_ratio) * p_d:
                demand_reject = True
                continue
            candidates.append((u_c, p_c, rank_lookup[c], c))

        if value_reject:
            stats.promotion_rejected_value += 1
            continue

        stats.promotion_attempted += 1
        if not candidates:
            if demand_reject:
                stats.promotion_rejected_demand += 1
            else:
                stats.promotion_rejected_value += 1
            continue

        candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
        _, p_c, _, c = candidates[0]
        final_boundary[slot] = c
        unused_challengers.remove(c)
        promoted.append(c)
        displaced.append(d)
        ratio = p_c / max(p_d, eps)
        ratios.append(ratio)
        gap = p_d - p_c
        gaps.append(gap)
        relative_gaps.append(gap / max(p_d, eps))
        stats.promotion_accepted += 1

    stats.demand_promoted_indices = tuple(promoted)
    stats.demand_displaced_indices = tuple(displaced)
    stats.value_selected_indices = tuple(promoted)
    stats.selected_value_src = len(promoted)
    stats.mean_P_ratio = _mean_python(ratios) if ratios else float("nan")
    stats.min_P_ratio = min(ratios) if ratios else float("nan")
    stats.max_P_ratio = max(ratios) if ratios else float("nan")
    stats.mean_P_gap = _mean_python(gaps) if gaps else float("nan")
    stats.max_P_gap = max(gaps) if gaps else float("nan")
    stats.mean_relative_P_gap = _mean_python(relative_gaps) if relative_gaps else float("nan")
    stats.demand_mean_U_promoted = _mean_tuple_values(stats, stats.demand_promoted_indices, "value")
    stats.demand_mean_U_displaced = _mean_tuple_values(stats, stats.demand_displaced_indices, "value")

    selected = torch.cat(
        (core, torch.as_tensor(final_boundary, dtype=torch.long, device=proximity_rank.device)),
        dim=0,
    )
    if selected.numel() < stats.selected_src:
        selected = _fill_from_rank(selected, proximity_rank, stats.selected_src)
    return selected[:stats.selected_src]


def _value_select_with_fallback(value_score, proximity_rank, pool_indices, slots):
    empty = torch.empty((0,), dtype=torch.long, device=pool_indices.device)
    if value_score is None or pool_indices.numel() == 0 or int(slots) <= 0:
        selected = _fill_from_rank(empty, proximity_rank, slots, allowed_indices=pool_indices)
        return selected, empty

    pool_values = value_score[pool_indices].detach().float()
    informative_mask = torch.isfinite(pool_values) & (pool_values > 0.0)
    if not informative_mask.any():
        selected = _fill_from_rank(empty, proximity_rank, slots, allowed_indices=pool_indices)
        return selected, empty

    informative_indices = pool_indices[informative_mask]
    informative_values = pool_values[informative_mask]
    if (
        informative_indices.numel() == pool_indices.numel()
        and (informative_values.max() - informative_values.min()).item() <= 1e-8
    ):
        selected = _fill_from_rank(empty, proximity_rank, slots, allowed_indices=pool_indices)
        return selected, empty

    proximity_order = {idx: rank for rank, idx in enumerate(_to_tuple(proximity_rank))}
    pairs = [
        (float(informative_values[i].item()), proximity_order[int(informative_indices[i].item())], int(informative_indices[i].item()))
        for i in range(informative_indices.numel())
    ]
    pairs.sort(key=lambda item: (-item[0], item[1]))
    value_rank = torch.as_tensor([idx for _, _, idx in pairs], dtype=torch.long, device=pool_indices.device)
    value_selected = value_rank[:slots]
    selected = value_selected
    if selected.numel() < slots:
        selected = _fill_from_rank(selected, proximity_rank, slots, allowed_indices=pool_indices)
    return selected, value_selected


def _rank_indices(score, indices, descending=True, finite_only=False):
    if indices.numel() == 0:
        return indices.long()
    if score is None:
        return torch.empty((0,), dtype=torch.long, device=indices.device) if finite_only else indices.long()
    values = score[indices].detach().float().cpu()
    ids = indices.detach().long().cpu()
    pairs = [(float(values[i].item()), int(ids[i].item())) for i in range(ids.numel())]
    if descending:
        pairs.sort(key=lambda item: (-item[0] if isfinite(item[0]) else float("inf"), item[1]))
    else:
        pairs.sort(key=lambda item: (item[0] if isfinite(item[0]) else float("inf"), item[1]))
    if finite_only:
        pairs = [item for item in pairs if isfinite(item[0])]
    return torch.as_tensor([idx for _, idx in pairs], dtype=torch.long, device=indices.device)


def _fill_from_rank(selected_indices, proximity_rank, slots, allowed_indices=None):
    selected_set = set(_to_tuple(selected_indices))
    allowed = set(_to_tuple(proximity_rank if allowed_indices is None else allowed_indices))
    fill = []
    for idx in _to_tuple(proximity_rank):
        if idx in allowed and idx not in selected_set:
            fill.append(idx)
        if len(selected_set) + len(fill) >= slots:
            break
    if not fill:
        return selected_indices
    return torch.cat((selected_indices, torch.as_tensor(fill, dtype=torch.long, device=selected_indices.device)))


def _to_tuple(indices):
    if torch.is_tensor(indices):
        return tuple(int(i) for i in indices.detach().cpu().reshape(-1).tolist())
    return tuple(int(i) for i in indices)


def _masked_mean(values, indices):
    if values is None or indices.numel() == 0:
        return float("nan")
    selected = values[indices].detach().float()
    finite = selected[torch.isfinite(selected)]
    if finite.numel() == 0:
        return float("nan")
    return float(finite.mean().item())


def _overlap_ratio(a, b, total):
    if int(total) <= 0:
        return 1.0
    return len(set(a) & set(b)) / float(total)


def _mean_python(values):
    return sum(values) / float(len(values)) if values else 0.0


def _mean_tuple_values(stats, indices, field):
    values = getattr(stats, f"{field}_values", None)
    if values is None or not indices:
        return float("nan")
    return _masked_mean(values, torch.as_tensor(indices, dtype=torch.long, device=values.device))

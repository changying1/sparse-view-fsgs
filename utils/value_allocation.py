"""OBDKR value allocation utilities for FSGS proximity candidates."""

import torch


def compute_observation_scarcity(visible_view_count, tau_e=1.0, tau_s=3.0):
    with torch.no_grad():
        if tau_e <= 0 or tau_s <= 0:
            raise ValueError("tau_e and tau_s must be positive.")
        if not torch.is_tensor(visible_view_count) or visible_view_count.ndim != 1:
            raise ValueError("visible_view_count must be a Tensor[N].")
        if (visible_view_count < 0).any():
            raise ValueError("visible_view_count must be non-negative.")
        n = visible_view_count.to(dtype=torch.float32)
        value = (1.0 - torch.exp(-n / float(tau_e))) * torch.exp(-n / float(tau_s))
        return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)


def robust_normalize(values, low_quantile=0.05, high_quantile=0.95, eps=1e-8, statistics_mask=None):
    with torch.no_grad():
        if not (0.0 <= float(low_quantile) < float(high_quantile) <= 1.0):
            raise ValueError("Require 0 <= low_quantile < high_quantile <= 1.")
        if not torch.is_tensor(values) or values.ndim != 1:
            raise ValueError("values must be a Tensor[N].")
        out = torch.zeros_like(values, dtype=torch.float32)
        finite = torch.isfinite(values)
        statistics = finite
        if statistics_mask is not None:
            if not torch.is_tensor(statistics_mask) or statistics_mask.shape != values.shape:
                raise ValueError("statistics_mask must be a BoolTensor[N] matching values.")
            statistics = statistics & statistics_mask.to(device=values.device, dtype=torch.bool)
        if not statistics.any():
            return out
        statistics_values = values[statistics].to(dtype=torch.float32)
        q_low = torch.quantile(statistics_values, float(low_quantile))
        q_high = torch.quantile(statistics_values, float(high_quantile))
        denom = q_high - q_low
        if not torch.isfinite(denom) or denom.abs() <= eps:
            return out
        normalized = ((values.to(dtype=torch.float32) - q_low) / (denom + eps)).clamp(0.0, 1.0)
        out[finite] = normalized[finite]
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def compute_structural_value(boundary_support, geometric_turning, continuity_defect,
                             w_b=1.0, w_k=1.0, w_d=1.0):
    with torch.no_grad():
        _validate_same_shape(boundary_support, geometric_turning, continuity_defect)
        return torch.nan_to_num(
            float(w_b) * boundary_support.to(dtype=torch.float32)
            + float(w_k) * geometric_turning.to(device=boundary_support.device, dtype=torch.float32)
            + float(w_d) * continuity_defect.to(device=boundary_support.device, dtype=torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )


def compute_refine_utility(observation_scarcity, structural_value, redundancy,
                           lambda_r=1.0, eps=1e-8):
    with torch.no_grad():
        if lambda_r < 0:
            raise ValueError("lambda_r must be non-negative.")
        _validate_same_shape(observation_scarcity, structural_value, redundancy)
        denom = 1.0 + float(lambda_r) * redundancy.to(device=observation_scarcity.device, dtype=torch.float32)
        utility = (
            observation_scarcity.to(dtype=torch.float32)
            * structural_value.to(device=observation_scarcity.device, dtype=torch.float32)
            / denom.clamp_min(eps)
        )
        return torch.nan_to_num(utility, nan=0.0, posinf=0.0, neginf=0.0)


def compute_obdkr_value(observation_count, boundary, turning, defect, redundancy,
                        tau_e=1.0, tau_s=3.0, w_b=1.0, w_k=1.0, w_d=1.0,
                        lambda_r=1.0, low_quantile=0.05, high_quantile=0.95,
                        normalization_mask=None,
                        return_components=False):
    with torch.no_grad():
        _validate_same_shape(observation_count, boundary, turning, defect, redundancy)
        observation = compute_observation_scarcity(observation_count, tau_e=tau_e, tau_s=tau_s)
        b_norm = robust_normalize(boundary, low_quantile, high_quantile, statistics_mask=normalization_mask)
        k_norm = robust_normalize(turning, low_quantile, high_quantile, statistics_mask=normalization_mask)
        d_norm = robust_normalize(defect, low_quantile, high_quantile, statistics_mask=normalization_mask)
        r_norm = robust_normalize(redundancy, low_quantile, high_quantile, statistics_mask=normalization_mask)
        structural = compute_structural_value(b_norm, k_norm, d_norm, w_b=w_b, w_k=w_k, w_d=w_d)
        utility = compute_refine_utility(observation, structural, r_norm, lambda_r=lambda_r)
        if return_components:
            return {
                "O": observation,
                "B_norm": b_norm,
                "K_norm": k_norm,
                "D_norm": d_norm,
                "R_norm": r_norm,
                "S": structural,
                "U": utility,
            }
        return utility


def compute_structural_value_score(boundary, turning, defect, redundancy,
                                   w_b=1.0, w_k=1.0, w_d=1.0,
                                   lambda_r=1.0, low_quantile=0.05, high_quantile=0.95,
                                   normalization_mask=None,
                                   return_components=False):
    with torch.no_grad():
        if lambda_r < 0:
            raise ValueError("lambda_r must be non-negative.")
        _validate_same_shape(boundary, turning, defect, redundancy)
        b_norm = robust_normalize(boundary, low_quantile, high_quantile, statistics_mask=normalization_mask)
        k_norm = robust_normalize(turning, low_quantile, high_quantile, statistics_mask=normalization_mask)
        d_norm = robust_normalize(defect, low_quantile, high_quantile, statistics_mask=normalization_mask)
        r_norm = robust_normalize(redundancy, low_quantile, high_quantile, statistics_mask=normalization_mask)
        structural = compute_structural_value(b_norm, k_norm, d_norm, w_b=w_b, w_k=w_k, w_d=w_d)
        denom = (1.0 + float(lambda_r) * r_norm).clamp_min(1e-8)
        value = torch.nan_to_num(structural / denom, nan=0.0, posinf=0.0, neginf=0.0)
        if return_components:
            return {
                "O": torch.ones_like(value, dtype=torch.float32),
                "B": b_norm,
                "K": k_norm,
                "D": d_norm,
                "R": r_norm,
                "B_norm": b_norm,
                "K_norm": k_norm,
                "D_norm": d_norm,
                "R_norm": r_norm,
                "B_raw": boundary.to(dtype=torch.float32),
                "K_raw": turning.to(device=boundary.device, dtype=torch.float32),
                "D_raw": defect.to(device=boundary.device, dtype=torch.float32),
                "R_raw": redundancy.to(device=boundary.device, dtype=torch.float32),
                "S": structural,
                "U": value,
            }
        return value


def compute_gestalt_structural_value_score(boundary, turning, defect, redundancy,
                                           good_continuation,
                                           w_b=1.0, w_k=1.0, w_d=1.0,
                                           lambda_r=1.0, low_quantile=0.05, high_quantile=0.95,
                                           normalization_mask=None,
                                           gestalt_lambda=1.0,
                                           return_components=False):
    with torch.no_grad():
        if float(gestalt_lambda) < 0:
            raise ValueError("gestalt_value_lambda must be non-negative.")
        _validate_same_shape(boundary, turning, defect, redundancy, good_continuation)
        components = compute_structural_value_score(
            boundary,
            turning,
            defect,
            redundancy,
            w_b=w_b,
            w_k=w_k,
            w_d=w_d,
            lambda_r=lambda_r,
            low_quantile=low_quantile,
            high_quantile=high_quantile,
            normalization_mask=normalization_mask,
            return_components=True,
        )
        g = torch.nan_to_num(
            good_continuation.to(device=components["U"].device, dtype=torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        base_u = components["U"]
        value = torch.nan_to_num(
            base_u * (1.0 + float(gestalt_lambda) * g),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if return_components:
            components["G"] = g
            components["U_base"] = base_u
            components["U"] = value
            components["gestalt_lambda"] = float(gestalt_lambda)
            return components
        return value


def compute_balanced_gestalt_value_score(boundary, turning, defect, redundancy,
                                         good_continuation,
                                         w_b=0.0, w_k=1.0, w_d=1.0,
                                         lambda_r=0.0, low_quantile=0.05, high_quantile=0.95,
                                         normalization_mask=None,
                                         gestalt_balance_alpha=0.5,
                                         return_components=False):
    with torch.no_grad():
        alpha = float(gestalt_balance_alpha)
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("gestalt_balance_alpha must satisfy 0 <= alpha <= 1.")
        _validate_same_shape(boundary, turning, defect, redundancy, good_continuation)
        components = compute_structural_value_score(
            boundary,
            turning,
            defect,
            redundancy,
            w_b=0.0,
            w_k=1.0,
            w_d=1.0,
            lambda_r=0.0,
            low_quantile=low_quantile,
            high_quantile=high_quantile,
            normalization_mask=normalization_mask,
            return_components=True,
        )
        k = torch.nan_to_num(components["K_norm"].to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        d = torch.nan_to_num(components["D_norm"].to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        g = torch.nan_to_num(
            good_continuation.to(device=components["U"].device, dtype=torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        s_need = torch.nan_to_num(0.5 * (k + d), nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
        value = torch.nan_to_num(
            (1.0 - alpha) * s_need + alpha * g,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if return_components:
            components["K"] = k
            components["D"] = d
            components["G"] = g
            components["S_need"] = s_need
            components["U"] = value
            components["gestalt_balance_alpha"] = alpha
            components["balanced_w_b"] = float(w_b)
            components["balanced_w_k"] = float(w_k)
            components["balanced_w_d"] = float(w_d)
            components["balanced_lambda_r"] = float(lambda_r)
            return components
        return value


def compute_defect_conditioned_gestalt_value_score(boundary, turning, defect, redundancy,
                                                   good_continuation,
                                                   w_b=0.0, w_k=1.0, w_d=1.0,
                                                   lambda_r=0.0, low_quantile=0.05, high_quantile=0.95,
                                                   normalization_mask=None,
                                                   gestalt_lambda=1.0,
                                                   return_components=False):
    with torch.no_grad():
        if float(gestalt_lambda) < 0:
            raise ValueError("gestalt_value_lambda must be non-negative.")
        _validate_same_shape(boundary, turning, defect, redundancy, good_continuation)
        components = compute_structural_value_score(
            boundary,
            turning,
            defect,
            redundancy,
            w_b=0.0,
            w_k=1.0,
            w_d=1.0,
            lambda_r=0.0,
            low_quantile=low_quantile,
            high_quantile=high_quantile,
            normalization_mask=normalization_mask,
            return_components=True,
        )
        k = torch.nan_to_num(components["K_norm"].to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        d = torch.nan_to_num(components["D_norm"].to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        g = torch.nan_to_num(
            good_continuation.to(device=components["U"].device, dtype=torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        dg = torch.nan_to_num(d * g, nan=0.0, posinf=0.0, neginf=0.0)
        base_u = torch.nan_to_num(k + d, nan=0.0, posinf=0.0, neginf=0.0)
        value = torch.nan_to_num(
            base_u + float(gestalt_lambda) * dg,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if return_components:
            components["K"] = k
            components["D"] = d
            components["G"] = g
            components["DG"] = dg
            components["U_base"] = base_u
            components["U"] = value
            components["gestalt_lambda"] = float(gestalt_lambda)
            components["conditioned_w_b"] = float(w_b)
            components["conditioned_w_k"] = float(w_k)
            components["conditioned_w_d"] = float(w_d)
            components["conditioned_lambda_r"] = float(lambda_r)
            return components
        return value


def allocate_budget(utility, budget, candidate_mask=None, return_mask=False):
    with torch.no_grad():
        if not torch.is_tensor(utility) or utility.ndim != 1:
            raise ValueError("utility must be a Tensor[N].")
        if budget < 0:
            raise ValueError("budget must be non-negative.")
        valid = torch.isfinite(utility) & (utility > 0)
        if candidate_mask is not None:
            if not torch.is_tensor(candidate_mask) or candidate_mask.shape != utility.shape:
                raise ValueError("candidate_mask must be a BoolTensor[N] matching utility.")
            valid = valid & candidate_mask.to(device=utility.device, dtype=torch.bool)
        candidates = torch.nonzero(valid, as_tuple=False).squeeze(1)
        k = min(int(budget), int(candidates.numel()))
        if k <= 0:
            selected = torch.empty((0,), dtype=torch.long, device=utility.device)
        else:
            topk = torch.topk(utility[candidates].to(dtype=torch.float32), k=k, largest=True, sorted=True).indices
            selected = candidates[topk].long()
        if return_mask:
            mask = torch.zeros_like(utility, dtype=torch.bool)
            mask[selected] = True
            return mask
        return selected


def _validate_same_shape(*tensors):
    if not tensors or not all(torch.is_tensor(tensor) for tensor in tensors):
        raise ValueError("All inputs must be tensors.")
    shape = tensors[0].shape
    if len(shape) != 1:
        raise ValueError("Inputs must have shape [N].")
    for tensor in tensors[1:]:
        if tensor.shape != shape:
            raise ValueError("All inputs must have the same shape.")

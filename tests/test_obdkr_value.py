import torch

from utils.value_diagnostics import compute_obdkr_diagnostics, format_obdkr_diagnostics_log
from utils.value_allocation import (
    compute_obdkr_value,
    compute_observation_scarcity,
    robust_normalize,
)


def test_observation_scarcity_evidence_then_suppression():
    counts = torch.tensor([0.0, 1.0, 2.0, 12.0])
    obs = compute_observation_scarcity(counts, tau_e=1.0, tau_s=3.0)

    assert obs[0].item() == 0.0
    assert obs[1] > obs[0]
    assert obs[-1] < obs[2]


def test_observation_scarcity_rejects_invalid_tau():
    try:
        compute_observation_scarcity(torch.ones(3), tau_e=0.0, tau_s=3.0)
    except ValueError:
        return
    raise AssertionError("invalid tau_e should raise ValueError")


def test_robust_normalize_safe_range_constant_and_nan():
    values = torch.tensor([float("nan"), 1.0, 2.0, float("inf")])
    normalized = robust_normalize(values, 0.0, 1.0)
    constant = robust_normalize(torch.ones(4), 0.05, 0.95)

    assert torch.isfinite(normalized).all()
    assert normalized.min() >= 0
    assert normalized.max() <= 1
    assert torch.equal(constant, torch.zeros(4))


def test_obdkr_utility_uses_components_and_redundancy_penalty():
    counts = torch.tensor([1.0, 1.0, 1.0])
    boundary = torch.tensor([1.0, 2.0, 3.0])
    turning = torch.tensor([1.0, 2.0, 3.0])
    defect = torch.tensor([1.0, 2.0, 3.0])
    low_redundancy = torch.tensor([0.0, 0.0, 0.0])
    high_redundancy = torch.tensor([0.0, 0.0, 10.0])

    low = compute_obdkr_value(counts, boundary, turning, defect, low_redundancy, low_quantile=0.0, high_quantile=1.0)
    high = compute_obdkr_value(counts, boundary, turning, defect, high_redundancy, low_quantile=0.0, high_quantile=1.0)

    assert torch.isfinite(low).all()
    assert low[-1] > low[0]
    assert high[-1] <= low[-1]


def test_candidate_normalization_ignores_non_candidate_extreme_outlier():
    values = torch.tensor([1.0, 2.0, 3.0, 100000.0])
    candidate_mask = torch.tensor([True, True, True, False])

    normalized = robust_normalize(values, 0.0, 1.0, statistics_mask=candidate_mask)

    assert torch.allclose(normalized[:3], torch.tensor([0.0, 0.5, 1.0]))
    assert normalized[3].item() == 1.0


def test_candidate_normalization_empty_mask_returns_zero():
    values = torch.tensor([1.0, 2.0, 3.0])
    normalized = robust_normalize(values, statistics_mask=torch.zeros(3, dtype=torch.bool))

    assert torch.equal(normalized, torch.zeros(3))


def test_candidate_normalization_nan_inf_mask_returns_zero():
    values = torch.tensor([float("nan"), float("inf"), 3.0])
    mask = torch.tensor([True, True, False])

    normalized = robust_normalize(values, 0.0, 1.0, statistics_mask=mask)

    assert torch.equal(normalized, torch.zeros(3))


def test_robust_normalize_without_mask_preserves_legacy_statistics():
    values = torch.tensor([1.0, 2.0, 3.0, 100000.0])
    unmasked = robust_normalize(values, 0.0, 1.0)
    explicit_all = robust_normalize(values, 0.0, 1.0, statistics_mask=torch.ones(4, dtype=torch.bool))

    assert torch.allclose(unmasked, explicit_all)
    assert unmasked[2] < 0.001


def test_obdkr_value_candidate_normalization_components_are_finite():
    counts = torch.tensor([1.0, 2.0, 3.0, 4.0])
    boundary = torch.tensor([1.0, 2.0, 3.0, 100000.0])
    turning = torch.tensor([2.0, 3.0, 4.0, 100000.0])
    defect = torch.tensor([3.0, 4.0, 5.0, 100000.0])
    redundancy = torch.tensor([1.0, 2.0, 3.0, 100000.0])
    candidate_mask = torch.tensor([True, True, True, False])

    components = compute_obdkr_value(
        counts,
        boundary,
        turning,
        defect,
        redundancy,
        low_quantile=0.0,
        high_quantile=1.0,
        normalization_mask=candidate_mask,
        return_components=True,
    )

    assert set(components) == {"O", "B_norm", "K_norm", "D_norm", "R_norm", "S", "U"}
    for value in components.values():
        assert value.shape == (4,)
        assert torch.isfinite(value).all()


def test_obdkr_diagnostics_include_raw_observation_count_fields():
    counts = torch.tensor([1, 2, 2, 4], dtype=torch.long)
    boundary = torch.tensor([1.0, 2.0, 3.0, 4.0])
    turning = torch.tensor([1.0, 1.5, 2.0, 2.5])
    defect = torch.tensor([0.5, 1.0, 1.5, 2.0])
    redundancy = torch.zeros(4)
    candidate_mask = torch.tensor([True, True, False, False])
    components = compute_obdkr_value(
        counts,
        boundary,
        turning,
        defect,
        redundancy,
        low_quantile=0.0,
        high_quantile=1.0,
        return_components=True,
    )

    stats = compute_obdkr_diagnostics(components, candidate_mask, observation_count=counts)
    log_line = format_obdkr_diagnostics_log(600, stats)

    assert stats["obs_unique_count"] == 3
    assert stats["candidate_obs_unique_count"] == 2
    for field in (
        "obs_mean=",
        "obs_min=",
        "obs_max=",
        "obs_q25=",
        "obs_q50=",
        "obs_q75=",
        "obs_unique_count=",
        "candidate_obs_mean=",
        "candidate_obs_min=",
        "candidate_obs_max=",
        "candidate_obs_unique_count=",
    ):
        assert field in log_line


def test_obdkr_diagnostics_include_active_observation_source_fields():
    counts = torch.tensor([3, 3, 3, 3], dtype=torch.long)
    recent = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    boundary = torch.tensor([1.0, 2.0, 3.0, 4.0])
    turning = torch.tensor([1.0, 1.5, 2.0, 2.5])
    defect = torch.tensor([0.5, 1.0, 1.5, 2.0])
    redundancy = torch.zeros(4)
    components = compute_obdkr_value(
        recent,
        boundary,
        turning,
        defect,
        redundancy,
        low_quantile=0.0,
        high_quantile=1.0,
        return_components=True,
    )

    stats = compute_obdkr_diagnostics(
        components,
        observation_count=counts,
        recent_observation_count=recent,
        active_observation_count=recent,
        observation_source="recent",
        num_train_views=3,
    )
    log_line = format_obdkr_diagnostics_log(1000, stats)

    assert stats["observation_source"] == "recent"
    assert stats["active_obs_unique_count"] == 4
    for field in (
        "observation_source=recent",
        "active_obs_mean=",
        "active_obs_min=",
        "active_obs_max=",
        "active_obs_unique_count=",
    ):
        assert field in log_line

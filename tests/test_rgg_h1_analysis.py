import json

import pytest

from utils.rgg_h1_analysis import (
    STATUS_COMPETING_CENSORED,
    STATUS_DIST_PRUNE_CENSORED,
    STATUS_FAILURE,
    STATUS_NO_SNAPSHOT,
    STATUS_RETAINED,
    STATUS_RIGHT_CENSORED,
    analyze_h1_records,
    classify_h1_record,
    compute_future_outcome,
)


def _record(uid, *, birth_iter=100, visible_rate=0.5, unique_views=2, death_iter=None, death_reason=None, alive=True, observation_end_iter=300):
    return {
        "uid": uid,
        "birth_iter": birth_iter,
        "death_iter": death_iter,
        "death_reason": death_reason,
        "alive": alive,
        "observation_end_iter": observation_end_iter,
        "age_snapshots": {
            "50": {
                "iteration": birth_iter + 50,
                "age": 50,
                "postbirth_real_opportunities": 3,
                "postbirth_real_visible_events": int(round(visible_rate * 3)),
                "postbirth_unique_real_views": unique_views,
                "visible_rate": visible_rate,
            }
        },
    }


def _future_record(uid, *, observation_end_iter=700, death_reason=None, death_iter=None, alive=True, include_350=True, include_550=True):
    snapshots = {
        "50": {
            "iteration": 150,
            "age": 50,
            "postbirth_real_opportunities": 10,
            "postbirth_real_visible_events": 4,
            "postbirth_unique_real_views": 2,
            "visible_rate": 0.4,
            "opacity": 0.8,
        }
    }
    if include_350:
        snapshots["350"] = {
            "iteration": 450,
            "age": 350,
            "postbirth_real_opportunities": 40,
            "postbirth_real_visible_events": 19,
            "postbirth_unique_real_views": 3,
            "visible_rate": 19 / 40,
            "opacity": 0.5,
        }
    if include_550:
        snapshots["550"] = {
            "iteration": 650,
            "age": 550,
            "postbirth_real_opportunities": 60,
            "postbirth_real_visible_events": 24,
            "postbirth_unique_real_views": 3,
            "visible_rate": 24 / 60,
            "opacity": 0.4,
        }
    return {
        "uid": uid,
        "birth_iter": 100,
        "birth_opacity": 0.9,
        "death_iter": death_iter,
        "death_reason": death_reason,
        "alive": alive,
        "observation_end_iter": observation_end_iter,
        "age_snapshots": snapshots,
    }


def _opacity_correlation_record(uid, early_visible_rate, future_opacity):
    return {
        "uid": uid,
        "birth_iter": 100,
        "death_iter": None,
        "death_reason": None,
        "alive": True,
        "observation_end_iter": 700,
        "age_snapshots": {
            "50": {
                "iteration": 150,
                "age": 50,
                "postbirth_real_opportunities": 10,
                "postbirth_real_visible_events": int(round(early_visible_rate * 10)),
                "postbirth_unique_real_views": 2,
                "visible_rate": early_visible_rate,
                "opacity": 1.0,
            },
            "350": {
                "iteration": 450,
                "age": 350,
                "postbirth_real_opportunities": 40,
                "postbirth_real_visible_events": 20,
                "postbirth_unique_real_views": 4,
                "visible_rate": 0.5,
                "opacity": future_opacity,
            },
        },
    }


def test_training_prune_within_horizon_is_failure():
    record = _record(1, death_reason="training_prune", death_iter=220, alive=False)

    result = classify_h1_record(record, snapshot_age=50, future_horizon=100)

    assert result["status"] == STATUS_FAILURE


def test_training_prune_after_horizon_is_retained():
    record = _record(1, death_reason="training_prune", death_iter=260, alive=False)

    result = classify_h1_record(record, snapshot_age=50, future_horizon=100)

    assert result["status"] == STATUS_RETAINED


def test_alive_with_enough_observation_is_retained():
    record = _record(1, alive=True, observation_end_iter=260)

    result = classify_h1_record(record, snapshot_age=50, future_horizon=100)

    assert result["status"] == STATUS_RETAINED


def test_alive_with_short_observation_is_right_censored():
    record = _record(1, alive=True, observation_end_iter=240)

    result = classify_h1_record(record, snapshot_age=50, future_horizon=100)

    assert result["status"] == STATUS_RIGHT_CENSORED


def test_split_replacement_before_horizon_is_competing_censored():
    record = _record(1, death_reason="split_replaced", death_iter=230, alive=False)

    result = classify_h1_record(record, snapshot_age=50, future_horizon=100)

    assert result["status"] == STATUS_COMPETING_CENSORED


def test_split_replacement_after_horizon_is_retained():
    record = _record(1, death_reason="split_replaced", death_iter=260, alive=False)

    result = classify_h1_record(record, snapshot_age=50, future_horizon=100)

    assert result["status"] == STATUS_RETAINED


def test_dist_prune_with_unknown_death_iter_is_dist_prune_censored():
    record = _record(1, death_reason="dist_prune", death_iter=-1, alive=False)

    result = classify_h1_record(record, snapshot_age=50, future_horizon=100)

    assert result["status"] == STATUS_DIST_PRUNE_CENSORED


def test_missing_age_snapshot_is_no_snapshot():
    record = _record(1)
    record["age_snapshots"] = {}

    result = classify_h1_record(record, snapshot_age=50, future_horizon=100)

    assert result["status"] == STATUS_NO_SNAPSHOT


def test_visible_rate_fixed_bins():
    records = [
        _record(1, visible_rate=0.0, death_reason="training_prune", death_iter=220, alive=False),
        _record(2, visible_rate=0.10, death_reason="training_prune", death_iter=220, alive=False),
        _record(3, visible_rate=0.40, death_reason="training_prune", death_iter=220, alive=False),
        _record(4, visible_rate=0.60, death_reason="training_prune", death_iter=220, alive=False),
        _record(5, visible_rate=0.90, death_reason="training_prune", death_iter=220, alive=False),
    ]

    analysis = analyze_h1_records(records, snapshot_age=50, future_horizon=100)

    assert [item["count"] for item in analysis["visible_rate_bins"]] == [1, 1, 1, 1, 1]
    assert [item["failure_count"] for item in analysis["visible_rate_bins"]] == [1, 1, 1, 1, 1]


def test_zero_variance_correlation_is_none():
    records = [
        _record(1, visible_rate=0.5, death_reason="training_prune", death_iter=220, alive=False),
        _record(2, visible_rate=0.5, death_reason="training_prune", death_iter=220, alive=False),
    ]

    analysis = analyze_h1_records(records, snapshot_age=50, future_horizon=100)

    assert analysis["pearson_visible_rate_vs_failure"] is None


def test_analysis_output_is_json_serializable():
    records = [
        _record(1, visible_rate=0.2, death_reason="training_prune", death_iter=220, alive=False),
        _record(2, visible_rate=0.8, alive=True, observation_end_iter=260),
    ]

    analysis = analyze_h1_records(records, snapshot_age=50, future_horizon=100)

    assert analysis["counts"]["binary_eligible"] == 2
    assert analysis["pearson_visible_rate_vs_failure"] < 0
    json.dumps(analysis)


def test_age50_to_age350_visibility_delta_and_opacity_retention():
    record = _future_record(1)

    outcome = compute_future_outcome(record, snapshot_age=50, future_horizon=300)

    assert outcome["future_status"] == STATUS_RETAINED
    assert outcome["future_outcome_age"] == 350
    assert outcome["future_window_opportunities"] == 30
    assert outcome["future_window_visible_events"] == 15
    assert outcome["future_window_visible_rate"] == 0.5
    assert outcome["future_cumulative_unique_real_views"] == 3
    assert outcome["opacity_future"] == 0.5
    assert outcome["opacity_delta"] == pytest.approx(-0.3)
    assert outcome["opacity_ratio"] == pytest.approx(0.625)


def test_age50_to_age550_visibility_delta():
    record = _future_record(1)

    outcome = compute_future_outcome(record, snapshot_age=50, future_horizon=500)

    assert outcome["future_status"] == STATUS_RETAINED
    assert outcome["future_outcome_age"] == 550
    assert outcome["future_window_opportunities"] == 50
    assert outcome["future_window_visible_events"] == 20
    assert outcome["future_window_visible_rate"] == 0.4
    assert outcome["opacity_future"] == 0.4


def test_future_outcome_right_censor_when_observation_too_short():
    record = _future_record(1, observation_end_iter=400, include_350=False, include_550=False)

    outcome = compute_future_outcome(record, snapshot_age=50, future_horizon=300)

    assert outcome["future_status"] == STATUS_RIGHT_CENSORED


def test_future_outcome_split_replaced_is_competing_censor():
    record = _future_record(
        1,
        observation_end_iter=700,
        death_reason="split_replaced",
        death_iter=300,
        alive=False,
        include_350=False,
        include_550=False,
    )

    outcome = compute_future_outcome(record, snapshot_age=50, future_horizon=300)

    assert outcome["future_status"] == STATUS_COMPETING_CENSORED


def test_observed_training_prune_overrides_short_observation_without_future_snapshot():
    record = _future_record(
        1,
        observation_end_iter=400,
        death_reason="training_prune",
        death_iter=300,
        alive=False,
        include_350=False,
        include_550=False,
    )

    classification = classify_h1_record(record, snapshot_age=50, future_horizon=300)
    outcome = compute_future_outcome(record, snapshot_age=50, future_horizon=300)

    assert classification["status"] == STATUS_FAILURE
    assert outcome["future_status"] == STATUS_FAILURE


def test_short_observation_without_death_event_is_right_censored():
    record = _future_record(
        1,
        observation_end_iter=400,
        alive=True,
        include_350=False,
        include_550=False,
    )

    classification = classify_h1_record(record, snapshot_age=50, future_horizon=300)
    outcome = compute_future_outcome(record, snapshot_age=50, future_horizon=300)

    assert classification["status"] == STATUS_RIGHT_CENSORED
    assert outcome["future_status"] == STATUS_RIGHT_CENSORED


def test_split_replaced_overrides_short_observation_as_competing_censor():
    record = _future_record(
        1,
        observation_end_iter=400,
        death_reason="split_replaced",
        death_iter=300,
        alive=False,
        include_350=False,
        include_550=False,
    )

    classification = classify_h1_record(record, snapshot_age=50, future_horizon=300)
    outcome = compute_future_outcome(record, snapshot_age=50, future_horizon=300)

    assert classification["status"] == STATUS_COMPETING_CENSORED
    assert outcome["future_status"] == STATUS_COMPETING_CENSORED


def test_classification_and_future_outcome_status_are_consistent_at_horizon():
    records = [
        _future_record(1, include_350=True, include_550=False),
        _future_record(
            2,
            observation_end_iter=400,
            death_reason="training_prune",
            death_iter=300,
            alive=False,
            include_350=False,
            include_550=False,
        ),
        _future_record(3, observation_end_iter=400, include_350=False, include_550=False),
        _future_record(
            4,
            observation_end_iter=400,
            death_reason="split_replaced",
            death_iter=300,
            alive=False,
            include_350=False,
            include_550=False,
        ),
        _future_record(
            5,
            observation_end_iter=400,
            death_reason="dist_prune",
            death_iter=300,
            alive=False,
            include_350=False,
            include_550=False,
        ),
        _future_record(6, observation_end_iter=500, include_350=False, include_550=False),
    ]

    for record in records:
        classification = classify_h1_record(record, snapshot_age=50, future_horizon=300)
        outcome = compute_future_outcome(record, snapshot_age=50, future_horizon=300)

        assert outcome["future_status"] == classification["status"]


def test_future_snapshot_at_training_prune_horizon_is_failure():
    record = _future_record(
        1,
        death_reason="training_prune",
        death_iter=450,
        alive=False,
        include_350=True,
        include_550=False,
    )

    classification = classify_h1_record(record, snapshot_age=50, future_horizon=300)
    outcome = compute_future_outcome(record, snapshot_age=50, future_horizon=300)

    assert classification["status"] == STATUS_FAILURE
    assert outcome["future_status"] == STATUS_FAILURE
    assert outcome["future_window_opportunities"] == 30
    assert outcome["future_window_visible_events"] == 15
    assert outcome["future_window_visible_rate"] == 0.5
    assert outcome["opacity_future"] == 0.5
    assert outcome["opacity_delta"] == pytest.approx(-0.3)
    assert outcome["opacity_ratio"] == pytest.approx(0.625)


def test_future_correlation_zero_variance_is_none():
    records = [
        _future_record(1),
        _future_record(2),
    ]

    analysis = analyze_h1_records(records, snapshot_age=50, future_horizon=300)

    assert analysis["future_visibility"]["pearson_early_visible_rate_vs_future_window_visible_rate"] is None
    assert analysis["future_visibility"]["spearman_early_visible_rate_vs_future_window_visible_rate"] is None
    assert analysis["future_visibility"]["paired_count_early_visible_rate_future_visible_rate"] == 2
    assert analysis["opacity_retention"]["pearson_early_opacity_vs_opacity_future"] is None
    assert analysis["opacity_retention"]["spearman_early_opacity_vs_opacity_future"] is None
    assert analysis["opacity_retention"]["paired_count_early_visible_rate_opacity_future"] == 2
    assert analysis["opacity_retention"]["paired_count_early_visible_rate_opacity_delta"] == 2
    assert analysis["opacity_retention"]["paired_count_early_visible_rate_opacity_ratio"] == 2


def test_early_visible_rate_opacity_correlations_use_valid_pairs():
    records = [
        _opacity_correlation_record(1, early_visible_rate=0.1, future_opacity=0.2),
        _opacity_correlation_record(2, early_visible_rate=0.5, future_opacity=0.6),
        _opacity_correlation_record(3, early_visible_rate=0.9, future_opacity=1.0),
        _opacity_correlation_record(4, early_visible_rate=0.3, future_opacity=None),
    ]

    analysis = analyze_h1_records(records, snapshot_age=50, future_horizon=300)
    opacity = analysis["opacity_retention"]

    assert opacity["paired_count_early_visible_rate_opacity_future"] == 3
    assert opacity["paired_count_early_visible_rate_opacity_delta"] == 3
    assert opacity["paired_count_early_visible_rate_opacity_ratio"] == 3
    assert opacity["pearson_early_visible_rate_vs_opacity_future"] > 0
    assert opacity["spearman_early_visible_rate_vs_opacity_future"] > 0
    assert opacity["pearson_early_visible_rate_vs_opacity_delta"] > 0
    assert opacity["spearman_early_visible_rate_vs_opacity_delta"] > 0
    assert opacity["pearson_early_visible_rate_vs_opacity_ratio"] > 0
    assert opacity["spearman_early_visible_rate_vs_opacity_ratio"] > 0


def test_legacy_record_lacking_new_fields_is_compatible():
    record = _record(1, alive=True, observation_end_iter=260)

    outcome = compute_future_outcome(record, snapshot_age=50, future_horizon=300)
    analysis = analyze_h1_records([record], snapshot_age=50, future_horizon=300)

    assert outcome["early_opacity"] is None
    assert outcome["opacity_future"] is None
    assert outcome["future_status"] == STATUS_RIGHT_CENSORED
    assert analysis["opacity_retention"]["mean_opacity_future"] is None

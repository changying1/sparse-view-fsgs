import json

from utils.rgg_h1_analysis import (
    STATUS_COMPETING_CENSORED,
    STATUS_DIST_PRUNE_CENSORED,
    STATUS_FAILURE,
    STATUS_RETAINED,
)
from utils.rgg_hypothesis_analysis import (
    analyze_generation_reliability,
    analyze_h2_hypothesis,
    analyze_origin_reliability,
    analyze_parent_reliability,
    analyze_temporal_parent_prediction,
    build_temporal_child_outcomes,
    compare_parent_reliability_groups,
    dump_h2_report_csv,
    dump_h2_report_json,
)


def test_single_parent_multiple_children_parent_reliability():
    records = [
        {"uid": 1, "source_uid": -1, "alive": True},
        {"uid": 2, "source_uid": 1, "alive": True},
        {"uid": 3, "source_uid": 1, "alive": False},
        {"uid": 4, "source_uid": 1, "alive": True},
    ]

    stats = analyze_parent_reliability(records)

    assert stats["parent_count"] == 1
    assert stats["parents"][0]["source_uid"] == 1
    assert stats["parents"][0]["child_count"] == 3
    assert stats["parents"][0]["alive_child_count"] == 2
    assert stats["parents"][0]["dead_child_count"] == 1
    assert stats["parents"][0]["child_survival_rate"] == 2 / 3
    assert stats["parents"][0]["R_parent"] == 2 / 3


def test_high_low_parent_grouping_uses_top_and_bottom_quartiles():
    records = [
        {"uid": 10, "source_uid": -1, "alive": True},
        {"uid": 11, "source_uid": 10, "alive": True, "lifetime": 100, "visible_rate": 0.8},
        {"uid": 12, "source_uid": 10, "alive": True, "lifetime": 120, "visible_rate": 0.6},
        {"uid": 20, "source_uid": -1, "alive": True},
        {"uid": 21, "source_uid": 20, "alive": False, "lifetime": 30, "visible_rate": 0.1},
        {"uid": 22, "source_uid": 20, "alive": False, "lifetime": 40, "visible_rate": 0.2},
        {"uid": 30, "source_uid": -1, "alive": True},
        {"uid": 31, "source_uid": 30, "alive": True, "lifetime": 70, "visible_rate": 0.5},
        {"uid": 32, "source_uid": 30, "alive": False, "lifetime": 60, "visible_rate": 0.4},
        {"uid": 40, "source_uid": -1, "alive": True},
        {"uid": 41, "source_uid": 40, "alive": True, "lifetime": 80, "visible_rate": 0.7},
        {"uid": 42, "source_uid": 40, "alive": False, "lifetime": 50, "visible_rate": 0.3},
    ]

    comparison = compare_parent_reliability_groups(records)

    assert comparison["high_parent_group"]["source_uids"] == [10]
    assert comparison["high_parent_group"]["child_survival_rate"] == 1.0
    assert comparison["high_parent_group"]["mean_lifetime"] == 110.0
    assert comparison["high_parent_group"]["mean_visible_rate"] == 0.7
    assert comparison["low_parent_group"]["source_uids"] == [20]
    assert comparison["low_parent_group"]["child_survival_rate"] == 0.0
    assert comparison["low_parent_group"]["mean_lifetime"] == 35.0
    assert comparison["low_parent_group"]["mean_visible_rate"] == 0.15000000000000002


def test_generation_reliability_statistics():
    records = [
        {"uid": 1, "generation": 0, "alive": True},
        {"uid": 2, "generation": 1, "alive": True},
        {"uid": 3, "generation": 1, "alive": False},
        {"uid": 4, "generation": 2, "alive": False},
    ]

    stats = analyze_generation_reliability(records)

    assert stats["total"] == 4
    assert stats["generations"] == [
        {"generation": 0, "count": 1, "alive": 1, "dead": 0, "survival_rate": 1.0},
        {"generation": 1, "count": 2, "alive": 1, "dead": 1, "survival_rate": 0.5},
        {"generation": 2, "count": 1, "alive": 0, "dead": 1, "survival_rate": 0.0},
    ]


def test_origin_reliability_and_report_exports(tmp_path):
    records = [
        {"uid": 1, "origin_type": "legacy", "alive": True, "lifetime": 100, "visible_rate": 0.9},
        {"uid": 2, "origin_type": "clone", "alive": True, "lifetime": 80, "visible_rate": 0.7},
        {"uid": 3, "origin_type": "clone", "alive": False, "lifetime": 20, "visible_rate": 0.2},
        {"uid": 4, "origin_type": "split", "alive": False, "lifetime": 30, "visible_rate": 0.1},
        {"uid": 5, "origin_type": "proximity", "alive": True, "lifetime": 60, "visible_rate": 0.5},
    ]

    stats = analyze_origin_reliability(records)
    report = analyze_h2_hypothesis(records)
    json_path = tmp_path / "rgg_h2_validation_report.json"
    csv_path = tmp_path / "rgg_h2_validation_report.csv"
    dump_h2_report_json(report, json_path)
    dump_h2_report_csv(report, csv_path)

    assert stats["legacy"]["survival_rate"] == 1.0
    assert stats["clone"]["count"] == 2
    assert stats["clone"]["alive"] == 1
    assert stats["clone"]["dead"] == 1
    assert stats["clone"]["survival_rate"] == 0.5
    assert stats["clone"]["mean_lifetime"] == 50.0
    assert stats["clone"]["mean_visible_rate"] == 0.44999999999999996
    assert stats["split"]["survival_rate"] == 0.0
    assert stats["proximity"]["mean_visible_rate"] == 0.5
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert set(loaded) == {
        "parent_reliability",
        "parent_group_comparison",
        "temporal_parent_prediction",
        "generation_reliability",
        "origin_reliability",
    }
    assert csv_path.read_text(encoding="utf-8").startswith("section,key,source_uid")


def test_temporal_history_future_children_are_disjoint():
    records = [
        _retained_child(1, 10, 800),
        _failure_child(2, 10, 850),
        _retained_child(3, 10, 1000),
        _failure_child(4, 10, 1100),
    ]

    result = analyze_temporal_parent_prediction(records, split_iter=1000, min_history_children=2)
    source = result["sources"][0]

    assert source["history_child_uids"] == [1, 2]
    assert source["future_child_uids"] == [3, 4]
    assert set(source["history_child_uids"]).isdisjoint(source["future_child_uids"])


def test_future_result_changes_do_not_change_q_history():
    base = [
        _retained_child(1, 10, 800),
        _failure_child(2, 10, 850),
    ]
    future_success = base + [_retained_child(3, 10, 1000)]
    future_failure = base + [_failure_child(3, 10, 1000)]

    q_success = analyze_temporal_parent_prediction(future_success, split_iter=1000)["sources"][0]["Q_history"]
    q_failure = analyze_temporal_parent_prediction(future_failure, split_iter=1000)["sources"][0]["Q_history"]

    assert q_success == q_failure == 0.5


def test_history_result_changes_change_q_history():
    records_a = [
        _retained_child(1, 10, 800),
        _failure_child(2, 10, 850),
        _retained_child(3, 10, 1000),
    ]
    records_b = [
        _retained_child(1, 10, 800),
        _retained_child(2, 10, 850),
        _retained_child(3, 10, 1000),
    ]

    q_a = analyze_temporal_parent_prediction(records_a, split_iter=1000)["sources"][0]["Q_history"]
    q_b = analyze_temporal_parent_prediction(records_b, split_iter=1000)["sources"][0]["Q_history"]

    assert q_a == 0.5
    assert q_b == 0.75


def test_temporal_positive_example_high_q_future_succeeds_low_q_future_fails():
    records = [
        _retained_child(1, 10, 800),
        _retained_child(2, 10, 850),
        _retained_child(3, 10, 1100, lifetime=100, visible_rate=0.8),
        _failure_child(4, 20, 800),
        _failure_child(5, 20, 850),
        _failure_child(6, 20, 1100, lifetime=30, visible_rate=0.1),
        _retained_child(7, 30, 800),
        _failure_child(8, 30, 850),
        _failure_child(9, 30, 1100),
        _retained_child(11, 40, 800),
        _failure_child(12, 40, 850),
        _retained_child(13, 40, 1100),
    ]

    result = analyze_temporal_parent_prediction(records, split_iter=1000)

    assert result["high_q_group"]["source_uids"] == [10]
    assert result["low_q_group"]["source_uids"] == [20]
    assert result["high_q_future_success_rate"] == 1.0
    assert result["low_q_future_success_rate"] == 0.0
    assert result["future_success_gap"] == 1.0


def test_temporal_counterexample_history_does_not_predict_future():
    records = [
        _retained_child(1, 10, 800),
        _retained_child(2, 10, 850),
        _failure_child(3, 10, 1100),
        _failure_child(4, 20, 800),
        _failure_child(5, 20, 850),
        _retained_child(6, 20, 1100),
        _retained_child(7, 30, 800),
        _failure_child(8, 30, 850),
        _retained_child(9, 30, 1100),
        _retained_child(11, 40, 800),
        _failure_child(12, 40, 850),
        _failure_child(13, 40, 1100),
    ]

    result = analyze_temporal_parent_prediction(records, split_iter=1000)

    assert result["high_q_group"]["source_uids"] == [10]
    assert result["low_q_group"]["source_uids"] == [20]
    assert result["future_success_gap"] == -1.0


def test_min_history_children_filtering():
    records = [
        _retained_child(1, 10, 800),
        _retained_child(2, 10, 1100),
        _retained_child(3, 20, 800),
        _retained_child(4, 20, 850),
        _retained_child(5, 20, 1100),
    ]

    result = analyze_temporal_parent_prediction(records, split_iter=1000, min_history_children=2)

    assert result["eligible_source_count"] == 1
    assert [source["source_uid"] for source in result["sources"]] == [10, 20]
    assert result["insufficient_source_count"] is True
    assert result["high_q_group"]["source_uids"] == []
    assert result["low_q_group"]["source_uids"] == []


def test_split_iter_boundary_birth_equal_split_is_future():
    records = [
        _retained_child(1, 10, 899),
        _retained_child(2, 10, 1000),
    ]

    result = analyze_temporal_parent_prediction(records, split_iter=1000, min_history_children=1)
    source = result["sources"][0]

    assert source["history_child_uids"] == [1]
    assert source["future_child_uids"] == [2]
    assert source["Q_history"] == 2 / 3


def test_history_final_alive_without_split_time_label_is_insufficient():
    records = [
        {"uid": 1, "source_uid": 10, "birth_iter": 900, "alive": True, "age_snapshots": {"50": {}}},
        _retained_child(2, 10, 1000),
    ]

    result = analyze_temporal_parent_prediction(records, split_iter=1000, min_history_children=1)

    assert result["sources"][0]["history_result_count"] == 0
    assert result["sources"][0]["Q_history"] == 0.5
    assert result["insufficient_temporal_label"][0]["uid"] == 1


def test_h1_failure_maps_to_h2_failure():
    outcomes = build_temporal_child_outcomes([_failure_child(1, 10, 800)])

    assert outcomes[0]["outcome"] == STATUS_FAILURE
    assert outcomes[0]["success"] is False
    assert outcomes[0]["label_iter"] == 850


def test_h1_retained_maps_to_h2_success():
    outcomes = build_temporal_child_outcomes([_retained_child(1, 10, 800)])

    assert outcomes[0]["outcome"] == STATUS_RETAINED
    assert outcomes[0]["success"] is True
    assert outcomes[0]["label_iter"] == 900


def test_dist_prune_censored_does_not_enter_history_counts():
    record = _failure_child(1, 10, 800)
    record["death_reason"] = "dist_prune"
    result = analyze_temporal_parent_prediction([record], split_iter=1000, min_history_children=1)

    assert build_temporal_child_outcomes([record])[0]["outcome"] == STATUS_DIST_PRUNE_CENSORED
    assert result["sources"][0]["history_result_count"] == 0
    assert result["sources"][0]["history_success_count"] == 0


def test_split_replaced_competing_censor_does_not_enter_history_counts():
    record = _failure_child(1, 10, 800)
    record["death_reason"] = "split_replaced"
    result = analyze_temporal_parent_prediction([record], split_iter=1000, min_history_children=1)

    assert build_temporal_child_outcomes([record])[0]["outcome"] == STATUS_COMPETING_CENSORED
    assert result["sources"][0]["history_result_count"] == 0


def test_future_and_history_use_same_h1_outcome_definition():
    records = [_failure_child(1, 10, 800), _failure_child(2, 10, 1100)]

    result = analyze_temporal_parent_prediction(records, split_iter=1000, min_history_children=1)
    outcomes = {item["uid"]: item for item in result["child_outcomes"]}

    assert outcomes[1]["outcome"] == STATUS_FAILURE
    assert outcomes[2]["outcome"] == STATUS_FAILURE
    assert result["sources"][0]["history_success_count"] == 0
    assert result["sources"][0]["future_success_count"] == 0


def test_less_than_four_eligible_sources_has_no_quartile_comparison():
    records = [
        _retained_child(1, 10, 800),
        _retained_child(2, 10, 850),
        _retained_child(3, 10, 1100),
        _failure_child(4, 20, 800),
        _failure_child(5, 20, 850),
        _failure_child(6, 20, 1100),
        _retained_child(7, 30, 800),
        _failure_child(8, 30, 850),
        _retained_child(9, 30, 1100),
    ]

    result = analyze_temporal_parent_prediction(records, split_iter=1000, min_history_children=2)

    assert result["eligible_source_count"] == 3
    assert result["insufficient_source_count"] is True
    assert result["high_q_source_count"] == 0
    assert result["low_q_source_count"] == 0
    assert result["future_success_gap"] is None


def test_high_low_source_uids_never_overlap():
    records = [
        _retained_child(1, 10, 800),
        _retained_child(2, 10, 850),
        _retained_child(3, 10, 1100),
        _failure_child(4, 20, 800),
        _failure_child(5, 20, 850),
        _failure_child(6, 20, 1100),
        _retained_child(7, 30, 800),
        _failure_child(8, 30, 850),
        _retained_child(9, 30, 1100),
        _retained_child(11, 40, 800),
        _failure_child(12, 40, 850),
        _failure_child(13, 40, 1100),
    ]

    result = analyze_temporal_parent_prediction(records, split_iter=1000, min_history_children=2)

    assert set(result["high_q_group"]["source_uids"]).isdisjoint(result["low_q_group"]["source_uids"])


def test_all_equal_q_reports_insufficient_q_separation():
    records = [
        _retained_child(1, 10, 800),
        _failure_child(2, 10, 850),
        _retained_child(3, 10, 1100),
        _retained_child(4, 20, 800),
        _failure_child(5, 20, 850),
        _failure_child(6, 20, 1100),
        _retained_child(7, 30, 800),
        _failure_child(8, 30, 850),
        _retained_child(9, 30, 1100),
        _retained_child(11, 40, 800),
        _failure_child(12, 40, 850),
        _failure_child(13, 40, 1100),
    ]

    result = analyze_temporal_parent_prediction(records, split_iter=1000, min_history_children=2)

    assert result["insufficient_q_separation"] is True
    assert result["future_success_gap"] is None


def _retained_child(uid, source_uid, birth_iter, lifetime=100, visible_rate=0.5):
    return {
        "uid": uid,
        "source_uid": source_uid,
        "birth_iter": birth_iter,
        "alive": True,
        "observation_end_iter": birth_iter + 100,
        "lifetime": lifetime,
        "visible_rate": visible_rate,
        "age_snapshots": {"50": {"visible_rate": visible_rate}},
    }


def _failure_child(uid, source_uid, birth_iter, lifetime=50, visible_rate=0.1):
    return {
        "uid": uid,
        "source_uid": source_uid,
        "birth_iter": birth_iter,
        "alive": False,
        "death_iter": birth_iter + 50,
        "death_reason": "training_prune",
        "lifetime": lifetime,
        "visible_rate": visible_rate,
        "age_snapshots": {"50": {"visible_rate": visible_rate}},
    }

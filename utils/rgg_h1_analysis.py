import json
import math
from statistics import mean, median


STATUS_FAILURE = "failure"
STATUS_RETAINED = "retained_to_horizon"
STATUS_RIGHT_CENSORED = "right_censored"
STATUS_COMPETING_CENSORED = "competing_censored"
STATUS_DIST_PRUNE_CENSORED = "dist_prune_censored"
STATUS_NO_SNAPSHOT = "no_snapshot"

VISIBLE_RATE_BINS = (
    ("0", 0.0, 0.0),
    ("(0, 0.25]", 0.0, 0.25),
    ("(0.25, 0.50]", 0.25, 0.50),
    ("(0.50, 0.75]", 0.50, 0.75),
    ("(0.75, 1.00]", 0.75, 1.00),
)


def load_h1_records(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def analyze_h1_records(records, snapshot_age, future_horizon):
    classified = [
        classify_h1_record(record, snapshot_age=snapshot_age, future_horizon=future_horizon)
        for record in records
    ]
    future_outcomes = [
        compute_future_outcome(record, snapshot_age=snapshot_age, future_horizon=future_horizon)
        for record in records
    ]
    eligible = [item for item in classified if item["status"] in (STATUS_FAILURE, STATUS_RETAINED)]
    failures = [item for item in eligible if item["status"] == STATUS_FAILURE]
    retained = [item for item in eligible if item["status"] == STATUS_RETAINED]
    return {
        "snapshot_age": int(snapshot_age),
        "future_horizon": int(future_horizon),
        "prediction": {
            "primary_predictor": "visible_rate",
            "secondary_descriptive_predictor": "postbirth_unique_real_views",
        },
        "counts": _count_statuses(classified),
        "future_outcome_counts": _count_future_outcomes(future_outcomes),
        "groups": {
            STATUS_FAILURE: _descriptive_stats(failures),
            STATUS_RETAINED: _descriptive_stats(retained),
        },
        "visible_rate_bins": _visible_rate_bins(eligible),
        "pearson_visible_rate_vs_failure": _pearson_visible_rate_vs_failure(eligible),
        "future_visibility": _future_visibility_stats(future_outcomes),
        "opacity_retention": _opacity_retention_stats(future_outcomes),
        "records": classified,
        "future_records": future_outcomes,
    }


def classify_h1_record(record, snapshot_age, future_horizon):
    snapshot_key = str(int(snapshot_age))
    snapshot = record.get("age_snapshots", {}).get(snapshot_key)
    result = {
        "uid": record.get("uid"),
        "status": STATUS_NO_SNAPSHOT,
        "visible_rate": None,
        "postbirth_unique_real_views": None,
        "prediction_end_iter": None,
    }
    if snapshot is None:
        return result

    birth_iter = _optional_int(record.get("birth_iter"))
    prediction_end_iter = None
    if birth_iter is not None:
        prediction_end_iter = birth_iter + int(snapshot_age) + int(future_horizon)
    result.update(
        {
            "visible_rate": _optional_float(snapshot.get("visible_rate")),
            "postbirth_unique_real_views": _optional_int(snapshot.get("postbirth_unique_real_views")),
            "prediction_end_iter": prediction_end_iter,
        }
    )
    if prediction_end_iter is None:
        result["status"] = STATUS_RIGHT_CENSORED
        return result

    death_reason = record.get("death_reason")
    death_iter = _optional_int(record.get("death_iter"))
    observation_end_iter = _optional_int(record.get("observation_end_iter"))
    alive = bool(record.get("alive", death_iter is None))

    if death_reason == "dist_prune":
        result["status"] = STATUS_DIST_PRUNE_CENSORED
    elif death_reason == "training_prune" and death_iter is not None and death_iter <= prediction_end_iter:
        result["status"] = STATUS_FAILURE
    elif death_reason == "split_replaced" and death_iter is not None and death_iter <= prediction_end_iter:
        result["status"] = STATUS_COMPETING_CENSORED
    elif alive or death_iter is None:
        if observation_end_iter is not None and observation_end_iter >= prediction_end_iter:
            result["status"] = STATUS_RETAINED
        else:
            result["status"] = STATUS_RIGHT_CENSORED
    elif death_iter > prediction_end_iter:
        result["status"] = STATUS_RETAINED
    else:
        result["status"] = STATUS_COMPETING_CENSORED
    return result


def compute_future_outcome(record, snapshot_age, future_horizon):
    snapshot_age = int(snapshot_age)
    future_horizon = int(future_horizon)
    early_key = str(snapshot_age)
    future_age = snapshot_age + future_horizon
    future_key = str(future_age)
    snapshots = record.get("age_snapshots", {})
    early = snapshots.get(early_key)
    future = snapshots.get(future_key)
    result = {
        "uid": record.get("uid"),
        "early_evidence_age": snapshot_age,
        "future_horizon": future_horizon,
        "future_outcome_age": future_age,
        "future_status": STATUS_NO_SNAPSHOT,
        "early_visible_rate": None,
        "early_opacity": None,
        "future_cumulative_visible_rate": None,
        "future_window_opportunities": None,
        "future_window_visible_events": None,
        "future_window_visible_rate": None,
        "future_cumulative_unique_real_views": None,
        "opacity_future": None,
        "opacity_delta": None,
        "opacity_ratio": None,
    }
    if early is None:
        return result

    result["early_visible_rate"] = _optional_float(early.get("visible_rate"))
    result["early_opacity"] = _optional_float(early.get("opacity"))
    birth_iter = _optional_int(record.get("birth_iter"))
    future_iter = None if birth_iter is None else birth_iter + future_age
    observation_end_iter = _optional_int(record.get("observation_end_iter"))
    death_reason = record.get("death_reason")
    death_iter = _optional_int(record.get("death_iter"))

    if death_reason == "dist_prune":
        result["future_status"] = STATUS_DIST_PRUNE_CENSORED
        return result
    if death_reason == "split_replaced" and (death_iter is None or future_iter is None or death_iter <= future_iter):
        result["future_status"] = STATUS_COMPETING_CENSORED
        return result
    if future is None:
        if death_reason == "training_prune" and death_iter is not None and future_iter is not None and death_iter <= future_iter:
            result["future_status"] = STATUS_FAILURE
        elif observation_end_iter is not None and future_iter is not None and observation_end_iter >= future_iter:
            result["future_status"] = STATUS_RETAINED
        else:
            result["future_status"] = STATUS_RIGHT_CENSORED
        return result

    early_opportunities = _optional_int(early.get("postbirth_real_opportunities")) or 0
    early_events = _optional_int(early.get("postbirth_real_visible_events")) or 0
    future_opportunities_cumulative = _optional_int(future.get("postbirth_real_opportunities")) or 0
    future_events_cumulative = _optional_int(future.get("postbirth_real_visible_events")) or 0
    future_opportunities = future_opportunities_cumulative - early_opportunities
    future_events = future_events_cumulative - early_events
    if death_reason == "training_prune" and death_iter is not None and future_iter is not None and death_iter <= future_iter:
        result["future_status"] = STATUS_FAILURE
    else:
        result["future_status"] = STATUS_RETAINED
    result["future_cumulative_visible_rate"] = _optional_float(future.get("visible_rate"))
    result["future_window_opportunities"] = future_opportunities
    result["future_window_visible_events"] = future_events
    result["future_window_visible_rate"] = None if future_opportunities <= 0 else future_events / future_opportunities
    result["future_cumulative_unique_real_views"] = _optional_int(future.get("postbirth_unique_real_views"))
    result["opacity_future"] = _optional_float(future.get("opacity"))
    if result["early_opacity"] is not None and result["opacity_future"] is not None:
        result["opacity_delta"] = result["opacity_future"] - result["early_opacity"]
        if result["early_opacity"] != 0:
            result["opacity_ratio"] = result["opacity_future"] / result["early_opacity"]
    return result


def dump_analysis_json(analysis, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2, sort_keys=True)


def _count_statuses(classified):
    counts = {
        "total_records": len(classified),
        "records_with_snapshot": 0,
        "binary_eligible": 0,
        STATUS_FAILURE: 0,
        STATUS_RETAINED: 0,
        STATUS_RIGHT_CENSORED: 0,
        STATUS_COMPETING_CENSORED: 0,
        STATUS_DIST_PRUNE_CENSORED: 0,
        STATUS_NO_SNAPSHOT: 0,
    }
    for item in classified:
        status = item["status"]
        if status != STATUS_NO_SNAPSHOT:
            counts["records_with_snapshot"] += 1
        if status in (STATUS_FAILURE, STATUS_RETAINED):
            counts["binary_eligible"] += 1
        counts[status] += 1
    return counts


def _count_future_outcomes(items):
    counts = {
        "total_records": len(items),
        "future_eligible": 0,
        STATUS_RETAINED: 0,
        STATUS_FAILURE: 0,
        STATUS_RIGHT_CENSORED: 0,
        STATUS_COMPETING_CENSORED: 0,
        STATUS_DIST_PRUNE_CENSORED: 0,
        STATUS_NO_SNAPSHOT: 0,
    }
    for item in items:
        status = item["future_status"]
        if status == STATUS_RETAINED:
            counts["future_eligible"] += 1
        counts[status] += 1
    return counts


def _descriptive_stats(items):
    visible_rates = [item["visible_rate"] for item in items if item["visible_rate"] is not None]
    unique_views = [
        item["postbirth_unique_real_views"]
        for item in items
        if item["postbirth_unique_real_views"] is not None
    ]
    return {
        "count": len(items),
        "mean_visible_rate": _mean_or_none(visible_rates),
        "median_visible_rate": _median_or_none(visible_rates),
        "mean_unique_real_views": _mean_or_none(unique_views),
        "median_unique_real_views": _median_or_none(unique_views),
    }


def _future_visibility_stats(items):
    eligible = [item for item in items if item["future_status"] == STATUS_RETAINED]
    rates = [item["future_window_visible_rate"] for item in eligible if item["future_window_visible_rate"] is not None]
    events = [item["future_window_visible_events"] for item in eligible if item["future_window_visible_events"] is not None]
    opportunities = [item["future_window_opportunities"] for item in eligible if item["future_window_opportunities"] is not None]
    early_rates = [item["early_visible_rate"] for item in eligible if item["early_visible_rate"] is not None and item["future_window_visible_rate"] is not None]
    paired_rates = [item["future_window_visible_rate"] for item in eligible if item["early_visible_rate"] is not None and item["future_window_visible_rate"] is not None]
    return {
        "count": len(eligible),
        "mean_future_window_visible_rate": _mean_or_none(rates),
        "median_future_window_visible_rate": _median_or_none(rates),
        "mean_future_window_visible_events": _mean_or_none(events),
        "median_future_window_visible_events": _median_or_none(events),
        "mean_future_window_opportunities": _mean_or_none(opportunities),
        "median_future_window_opportunities": _median_or_none(opportunities),
        "paired_count_early_visible_rate_future_visible_rate": len(early_rates),
        "pearson_early_visible_rate_vs_future_window_visible_rate": _pearson(early_rates, paired_rates),
        "spearman_early_visible_rate_vs_future_window_visible_rate": _spearman(early_rates, paired_rates),
    }


def _opacity_retention_stats(items):
    eligible = [item for item in items if item["future_status"] == STATUS_RETAINED]
    futures = [item["opacity_future"] for item in eligible if item["opacity_future"] is not None]
    deltas = [item["opacity_delta"] for item in eligible if item["opacity_delta"] is not None]
    ratios = [item["opacity_ratio"] for item in eligible if item["opacity_ratio"] is not None]
    early = [item["early_opacity"] for item in eligible if item["early_opacity"] is not None and item["opacity_future"] is not None]
    future = [item["opacity_future"] for item in eligible if item["early_opacity"] is not None and item["opacity_future"] is not None]
    early_visible_future, opacity_future = _paired_values(eligible, "early_visible_rate", "opacity_future")
    early_visible_delta, opacity_delta = _paired_values(eligible, "early_visible_rate", "opacity_delta")
    early_visible_ratio, opacity_ratio = _paired_values(eligible, "early_visible_rate", "opacity_ratio")
    return {
        "count": len(eligible),
        "mean_opacity_future": _mean_or_none(futures),
        "median_opacity_future": _median_or_none(futures),
        "mean_opacity_delta": _mean_or_none(deltas),
        "median_opacity_delta": _median_or_none(deltas),
        "mean_opacity_ratio": _mean_or_none(ratios),
        "median_opacity_ratio": _median_or_none(ratios),
        "pearson_early_opacity_vs_opacity_future": _pearson(early, future),
        "spearman_early_opacity_vs_opacity_future": _spearman(early, future),
        "paired_count_early_visible_rate_opacity_future": len(early_visible_future),
        "pearson_early_visible_rate_vs_opacity_future": _pearson(early_visible_future, opacity_future),
        "spearman_early_visible_rate_vs_opacity_future": _spearman(early_visible_future, opacity_future),
        "paired_count_early_visible_rate_opacity_delta": len(early_visible_delta),
        "pearson_early_visible_rate_vs_opacity_delta": _pearson(early_visible_delta, opacity_delta),
        "spearman_early_visible_rate_vs_opacity_delta": _spearman(early_visible_delta, opacity_delta),
        "paired_count_early_visible_rate_opacity_ratio": len(early_visible_ratio),
        "pearson_early_visible_rate_vs_opacity_ratio": _pearson(early_visible_ratio, opacity_ratio),
        "spearman_early_visible_rate_vs_opacity_ratio": _spearman(early_visible_ratio, opacity_ratio),
    }


def _paired_values(items, x_key, y_key):
    xs = []
    ys = []
    for item in items:
        x_value = item.get(x_key)
        y_value = item.get(y_key)
        if x_value is None or y_value is None:
            continue
        xs.append(x_value)
        ys.append(y_value)
    return xs, ys


def _visible_rate_bins(items):
    output = []
    for label, lower, upper in VISIBLE_RATE_BINS:
        in_bin = [item for item in items if _visible_rate_in_bin(item["visible_rate"], lower, upper)]
        failure_count = sum(1 for item in in_bin if item["status"] == STATUS_FAILURE)
        retained_count = sum(1 for item in in_bin if item["status"] == STATUS_RETAINED)
        count = len(in_bin)
        output.append(
            {
                "bin": label,
                "count": count,
                "failure_count": failure_count,
                "retained_count": retained_count,
                "failure_rate": None if count == 0 else failure_count / count,
            }
        )
    return output


def _visible_rate_in_bin(value, lower, upper):
    if value is None:
        return False
    if lower == upper:
        return value == lower
    return lower < value <= upper


def _pearson_visible_rate_vs_failure(items):
    xs = []
    ys = []
    for item in items:
        if item["visible_rate"] is None:
            continue
        xs.append(float(item["visible_rate"]))
        ys.append(1.0 if item["status"] == STATUS_FAILURE else 0.0)
    if len(xs) < 2 or _is_zero_variance(xs) or _is_zero_variance(ys):
        return None
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - x_mean) ** 2 for x in xs)
        * sum((y - y_mean) ** 2 for y in ys)
    )
    if denominator == 0:
        return None
    value = numerator / denominator
    if math.isnan(value):
        return None
    return value


def _pearson(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2 or _is_zero_variance(xs) or _is_zero_variance(ys):
        return None
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - x_mean) ** 2 for x in xs)
        * sum((y - y_mean) ** 2 for y in ys)
    )
    if denominator == 0:
        return None
    value = numerator / denominator
    if math.isnan(value):
        return None
    return value


def _spearman(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return _pearson(_ranks(xs), _ranks(ys))


def _ranks(values):
    sorted_pairs = sorted((value, index) for index, value in enumerate(values))
    ranks = [0.0] * len(values)
    index = 0
    while index < len(sorted_pairs):
        end = index + 1
        while end < len(sorted_pairs) and sorted_pairs[end][0] == sorted_pairs[index][0]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for _, original_index in sorted_pairs[index:end]:
            ranks[original_index] = rank
        index = end
    return ranks


def _optional_int(value):
    if value is None or value == "":
        return None
    value = int(value)
    return None if value < 0 else value


def _optional_float(value):
    if value is None or value == "":
        return None
    return float(value)


def _mean_or_none(values):
    return None if not values else float(mean(values))


def _median_or_none(values):
    return None if not values else float(median(values))


def _is_zero_variance(values):
    return all(value == values[0] for value in values)

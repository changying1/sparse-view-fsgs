import csv
import json
import math
from collections import defaultdict
from statistics import mean

from utils.rgg_h1_analysis import (
    STATUS_FAILURE,
    STATUS_RETAINED,
    classify_h1_record,
)


ORIGIN_TYPES = ("legacy", "clone", "split", "proximity")
DEFAULT_H2_SNAPSHOT_AGE = 50
DEFAULT_H2_FUTURE_HORIZON = 50


def analyze_parent_reliability(records):
    records = _normalize_records(records)
    children_by_parent = defaultdict(list)
    for record in records:
        source_uid = _optional_int(record.get("source_uid"))
        if source_uid is None or source_uid < 0:
            continue
        children_by_parent[source_uid].append(record)

    parents = []
    for source_uid in sorted(children_by_parent):
        children = children_by_parent[source_uid]
        alive_child_count = sum(1 for child in children if _is_alive(child))
        child_count = len(children)
        parents.append(
            {
                "source_uid": source_uid,
                "child_count": child_count,
                "alive_child_count": alive_child_count,
                "dead_child_count": child_count - alive_child_count,
                "child_survival_rate": _ratio(alive_child_count, child_count),
                "R_parent": _ratio(alive_child_count, child_count),
                "children": [child.get("uid") for child in children],
            }
        )
    return {
        "parent_count": len(parents),
        "parents": parents,
    }


def compare_parent_reliability_groups(records):
    records = _normalize_records(records)
    parent_reliability = analyze_parent_reliability(records)["parents"]
    if not parent_reliability:
        return {
            "high_parent_group": _summarize_parent_group([], records),
            "low_parent_group": _summarize_parent_group([], records),
        }

    sorted_parents = sorted(
        parent_reliability,
        key=lambda item: (
            item["R_parent"] if item["R_parent"] is not None else -1.0,
            item["source_uid"],
        ),
    )
    group_size = max(1, int(math.ceil(len(sorted_parents) * 0.25)))
    low_parents = sorted_parents[:group_size]
    high_parents = sorted_parents[-group_size:]
    return {
        "high_parent_group": _summarize_parent_group(high_parents, records),
        "low_parent_group": _summarize_parent_group(low_parents, records),
    }


def analyze_temporal_parent_prediction(
    records,
    split_iter=None,
    min_history_children=2,
    snapshot_age=DEFAULT_H2_SNAPSHOT_AGE,
    future_horizon=DEFAULT_H2_FUTURE_HORIZON,
):
    records = _normalize_records(records)
    resolved_split_iter, split_iter_source = _resolve_split_iter(records, split_iter)
    child_outcomes = build_temporal_child_outcomes(
        records,
        snapshot_age=snapshot_age,
        future_horizon=future_horizon,
    )
    outcomes_by_uid = {
        outcome["uid"]: outcome
        for outcome in child_outcomes
        if outcome.get("uid") is not None
    }
    children_by_parent = defaultdict(list)
    for record in records:
        source_uid = _optional_int(record.get("source_uid"))
        birth_iter = _optional_int(record.get("birth_iter"))
        if source_uid is None or source_uid < 0 or birth_iter is None or resolved_split_iter is None:
            continue
        children_by_parent[source_uid].append(record)

    sources = []
    insufficient_temporal_label = []
    for source_uid in sorted(children_by_parent):
        children = children_by_parent[source_uid]
        history_children = [
            child
            for child in children
            if _optional_int(child.get("birth_iter")) is not None
            and _optional_int(child.get("birth_iter")) < resolved_split_iter
        ]
        future_children = [
            child
            for child in children
            if _optional_int(child.get("birth_iter")) is not None
            and _optional_int(child.get("birth_iter")) >= resolved_split_iter
        ]
        history_child_uids = [child.get("uid") for child in history_children]
        future_child_uids = [child.get("uid") for child in future_children]
        overlap = sorted(set(history_child_uids) & set(future_child_uids))
        if overlap:
            raise ValueError("history_child_uids and future_child_uids must be disjoint.")

        history_labels = []
        for child in history_children:
            outcome = outcomes_by_uid.get(child.get("uid"))
            if outcome is None or outcome.get("success") is None or outcome.get("label_iter") is None:
                insufficient_temporal_label.append(
                    {
                        "uid": child.get("uid"),
                        "source_uid": source_uid,
                        "birth_iter": child.get("birth_iter"),
                        "reason": "history child does not have an uncensored H1 outcome label",
                    }
                )
                continue
            if outcome["label_iter"] > resolved_split_iter:
                insufficient_temporal_label.append(
                    {
                        "uid": child.get("uid"),
                        "source_uid": source_uid,
                        "birth_iter": child.get("birth_iter"),
                        "outcome": outcome.get("outcome"),
                        "label_iter": outcome.get("label_iter"),
                        "reason": "history outcome is not known by split_iter without future leakage",
                    }
                )
                continue
            history_labels.append(bool(outcome["success"]))
        future_labels = []
        future_insufficient = 0
        for child in future_children:
            outcome = outcomes_by_uid.get(child.get("uid"))
            if outcome is None or outcome.get("success") is None:
                future_insufficient += 1
                insufficient_temporal_label.append(
                    {
                        "uid": child.get("uid"),
                        "source_uid": source_uid,
                        "birth_iter": child.get("birth_iter"),
                        "outcome": outcome.get("outcome") if outcome else None,
                        "reason": "future child does not have an uncensored H1 outcome label",
                    }
                )
                continue
            future_labels.append(bool(outcome["success"]))

        history_success_count = sum(1 for label in history_labels if label)
        history_result_count = len(history_labels)
        future_success_count = sum(1 for label in future_labels if label)
        future_labeled_child_count = len(future_labels)
        source = {
            "source_uid": source_uid,
            "history_child_count": len(history_children),
            "history_result_count": history_result_count,
            "history_success_count": history_success_count,
            "history_insufficient_label_count": len(history_children) - history_result_count,
            "Q_history": (history_success_count + 1) / (history_result_count + 2),
            "future_child_count": len(future_children),
            "future_labeled_child_count": future_labeled_child_count,
            "future_success_count": future_success_count,
            "future_insufficient_label_count": future_insufficient,
            "future_success_rate": _ratio(future_success_count, future_labeled_child_count),
            "future_mean_lifetime": _mean_record_value(future_children, "lifetime"),
            "future_mean_visible_rate": _mean_record_value(future_children, "visible_rate"),
            "history_child_uids": history_child_uids,
            "future_child_uids": future_child_uids,
        }
        sources.append(source)

    eligible_sources = [
        source
        for source in sources
        if source["history_result_count"] >= int(min_history_children)
    ]
    sorted_sources = sorted(eligible_sources, key=lambda item: (item["Q_history"], item["source_uid"]))
    insufficient_source_count = len(sorted_sources) < 4
    group_size = int(math.ceil(len(sorted_sources) * 0.25)) if sorted_sources and not insufficient_source_count else 0
    low_sources = sorted_sources[:group_size] if group_size else []
    high_sources = sorted_sources[-group_size:] if group_size else []
    overlap = sorted({source["source_uid"] for source in low_sources} & {source["source_uid"] for source in high_sources})
    if overlap:
        raise ValueError("high-Q and low-Q source groups must be disjoint.")
    insufficient_q_separation = False
    if high_sources and low_sources:
        min_high_q = min(source["Q_history"] for source in high_sources)
        max_low_q = max(source["Q_history"] for source in low_sources)
        insufficient_q_separation = min_high_q <= max_low_q
    high_group = _summarize_temporal_source_group(high_sources)
    low_group = _summarize_temporal_source_group(low_sources)
    high_rate = None if insufficient_source_count or insufficient_q_separation else high_group["future_success_rate"]
    low_rate = None if insufficient_source_count or insufficient_q_separation else low_group["future_success_rate"]
    return {
        "split_iter": resolved_split_iter,
        "split_iter_source": split_iter_source,
        "snapshot_age": int(snapshot_age),
        "future_horizon": int(future_horizon),
        "eligible_source_count": len(eligible_sources),
        "min_history_children": int(min_history_children),
        "insufficient_source_count": insufficient_source_count,
        "insufficient_q_separation": insufficient_q_separation,
        "high_q_source_count": high_group["source_count"],
        "low_q_source_count": low_group["source_count"],
        "high_q_future_child_count": high_group["future_child_count"],
        "low_q_future_child_count": low_group["future_child_count"],
        "high_q_future_success_rate": high_rate,
        "low_q_future_success_rate": low_rate,
        "future_success_gap": _gap(high_rate, low_rate),
        "high_q_group": high_group,
        "low_q_group": low_group,
        "sources": sources,
        "child_outcomes": child_outcomes,
        "insufficient_temporal_label": insufficient_temporal_label,
        "label_requirements": (
            "H2 child labels reuse H1 classify_h1_record semantics at the fixed snapshot_age/future_horizon. "
            "Only failure and retained_to_horizon are success/failure labels; censored outcomes are excluded. "
            "History labels must have label_iter <= split_iter."
        ),
    }


def build_temporal_child_outcomes(
    records,
    snapshot_age=DEFAULT_H2_SNAPSHOT_AGE,
    future_horizon=DEFAULT_H2_FUTURE_HORIZON,
):
    outcomes = []
    for record in _normalize_records(records):
        classified = classify_h1_record(
            record,
            snapshot_age=snapshot_age,
            future_horizon=future_horizon,
        )
        outcome = classified["status"]
        success = None
        label_iter = None
        if outcome == STATUS_FAILURE:
            success = False
            label_iter = _optional_int(record.get("death_iter"))
        elif outcome == STATUS_RETAINED:
            success = True
            label_iter = _optional_int(classified.get("prediction_end_iter"))
        outcomes.append(
            {
                "uid": record.get("uid"),
                "source_uid": record.get("source_uid"),
                "birth_iter": record.get("birth_iter"),
                "outcome": outcome,
                "success": success,
                "label_iter": label_iter,
            }
        )
    return outcomes


def analyze_generation_reliability(records):
    records = _normalize_records(records)
    by_generation = defaultdict(list)
    for record in records:
        generation = _optional_int(record.get("generation"))
        if generation is None:
            generation = "unknown"
        by_generation[generation].append(record)

    output = []
    for generation in sorted(by_generation, key=lambda value: (value == "unknown", value)):
        generation_records = by_generation[generation]
        alive = sum(1 for record in generation_records if _is_alive(record))
        count = len(generation_records)
        output.append(
            {
                "generation": generation,
                "count": count,
                "alive": alive,
                "dead": count - alive,
                "survival_rate": _ratio(alive, count),
            }
        )
    return {
        "total": len(records),
        "generations": output,
    }


def analyze_origin_reliability(records):
    records = _normalize_records(records)
    by_origin = {origin: [] for origin in ORIGIN_TYPES}
    for record in records:
        origin = _origin(record)
        if origin not in by_origin:
            by_origin[origin] = []
        by_origin[origin].append(record)

    output = {}
    for origin in ORIGIN_TYPES:
        origin_records = by_origin.get(origin, [])
        alive = sum(1 for record in origin_records if _is_alive(record))
        count = len(origin_records)
        output[origin] = {
            "count": count,
            "alive": alive,
            "dead": count - alive,
            "survival_rate": _ratio(alive, count),
            "mean_lifetime": _mean_record_value(origin_records, "lifetime"),
            "mean_visible_rate": _mean_record_value(origin_records, "visible_rate"),
        }
    extra_origins = sorted(set(by_origin) - set(ORIGIN_TYPES))
    for origin in extra_origins:
        origin_records = by_origin[origin]
        alive = sum(1 for record in origin_records if _is_alive(record))
        count = len(origin_records)
        output[origin] = {
            "count": count,
            "alive": alive,
            "dead": count - alive,
            "survival_rate": _ratio(alive, count),
            "mean_lifetime": _mean_record_value(origin_records, "lifetime"),
            "mean_visible_rate": _mean_record_value(origin_records, "visible_rate"),
        }
    return output


def analyze_h2_hypothesis(
    records,
    split_iter=None,
    min_history_children=2,
    snapshot_age=DEFAULT_H2_SNAPSHOT_AGE,
    future_horizon=DEFAULT_H2_FUTURE_HORIZON,
):
    records = _normalize_records(records)
    return {
        "parent_reliability": analyze_parent_reliability(records),
        "parent_group_comparison": compare_parent_reliability_groups(records),
        "temporal_parent_prediction": analyze_temporal_parent_prediction(
            records,
            split_iter=split_iter,
            min_history_children=min_history_children,
            snapshot_age=snapshot_age,
            future_horizon=future_horizon,
        ),
        "generation_reliability": analyze_generation_reliability(records),
        "origin_reliability": analyze_origin_reliability(records),
    }


def load_lineage_records_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict) and "records" in data:
        return data["records"]
    if not isinstance(data, list):
        raise ValueError("rgg lineage records JSON must be a list or contain a records list.")
    return data


def dump_h2_report_json(report, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)


def dump_h2_report_csv(report, path):
    rows = _report_rows(report)
    fieldnames = [
        "section",
        "key",
        "source_uid",
        "generation",
        "origin_type",
        "count",
        "alive",
        "dead",
        "child_count",
        "alive_child_count",
        "dead_child_count",
        "survival_rate",
        "child_survival_rate",
        "R_parent",
        "mean_lifetime",
        "mean_visible_rate",
        "value",
        "split_iter",
        "Q_history",
        "future_success_rate",
        "future_child_count",
        "future_success_count",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _summarize_parent_group(parents, records):
    parent_uids = [parent["source_uid"] for parent in parents]
    parent_uid_set = set(parent_uids)
    children = [
        record
        for record in records
        if _optional_int(record.get("source_uid")) in parent_uid_set
    ]
    alive_child_count = sum(1 for child in children if _is_alive(child))
    child_count = len(children)
    return {
        "parent_count": len(parents),
        "source_uids": parent_uids,
        "child_count": child_count,
        "alive_child_count": alive_child_count,
        "dead_child_count": child_count - alive_child_count,
        "child_survival_rate": _ratio(alive_child_count, child_count),
        "mean_lifetime": _mean_record_value(children, "lifetime"),
        "mean_visible_rate": _mean_record_value(children, "visible_rate"),
    }


def _summarize_temporal_source_group(sources):
    future_child_count = sum(source["future_child_count"] for source in sources)
    future_labeled_child_count = sum(source["future_labeled_child_count"] for source in sources)
    future_success_count = sum(source["future_success_count"] for source in sources)
    lifetimes = []
    visible_rates = []
    for source in sources:
        if source["future_mean_lifetime"] is not None:
            lifetimes.append(source["future_mean_lifetime"])
        if source["future_mean_visible_rate"] is not None:
            visible_rates.append(source["future_mean_visible_rate"])
    return {
        "source_count": len(sources),
        "source_uids": [source["source_uid"] for source in sources],
        "future_child_count": future_child_count,
        "future_labeled_child_count": future_labeled_child_count,
        "future_success_count": future_success_count,
        "future_success_rate": _ratio(future_success_count, future_labeled_child_count),
        "future_mean_lifetime": None if not lifetimes else float(mean(lifetimes)),
        "future_mean_visible_rate": None if not visible_rates else float(mean(visible_rates)),
    }


def _report_rows(report):
    rows = []
    temporal = report.get("temporal_parent_prediction", {})
    rows.append(
        {
            "section": "temporal_parent_prediction",
            "key": "summary",
            "split_iter": temporal.get("split_iter"),
            "count": temporal.get("eligible_source_count"),
            "value": temporal.get("future_success_gap"),
            "future_success_rate": temporal.get("high_q_future_success_rate"),
            "future_child_count": temporal.get("high_q_future_child_count"),
        }
    )
    for source in temporal.get("sources", []):
        rows.append(
            {
                "section": "temporal_parent_prediction_source",
                "key": str(source["source_uid"]),
                "source_uid": source["source_uid"],
                "child_count": source["history_child_count"],
                "alive_child_count": source["history_success_count"],
                "Q_parent": source["Q_history"],
                "Q_history": source["Q_history"],
                "future_child_count": source["future_child_count"],
                "future_success_count": source["future_success_count"],
                "future_success_rate": source["future_success_rate"],
            }
        )
    for parent in report["parent_reliability"]["parents"]:
        rows.append({"section": "parent_reliability", "key": str(parent["source_uid"]), **parent})
    for group_name, group in report["parent_group_comparison"].items():
        rows.append({"section": "parent_group_comparison", "key": group_name, **group})
    for item in report["generation_reliability"]["generations"]:
        rows.append({"section": "generation_reliability", "key": str(item["generation"]), **item})
    for origin, item in report["origin_reliability"].items():
        rows.append({"section": "origin_reliability", "key": origin, "origin_type": origin, **item})
    return rows


def _normalize_records(records):
    if records is None:
        return []
    if isinstance(records, dict):
        records = records.get("records", [])
    return [dict(record) for record in records]


def _origin(record):
    origin = record.get("origin_type")
    if origin is None:
        return "unknown"
    origin = str(origin).lower()
    if origin == "proximity_child":
        return "proximity"
    return origin


def _is_alive(record):
    return bool(record.get("alive")) if record.get("alive") is not None else False


def _resolve_split_iter(records, split_iter):
    if split_iter is not None:
        return int(split_iter), "explicit"
    birth_iters = sorted(
        _optional_int(record.get("birth_iter"))
        for record in records
        if _optional_int(record.get("birth_iter")) is not None
    )
    if not birth_iters:
        return None, "unavailable_no_birth_iter"
    return birth_iters[len(birth_iters) // 2], "default_median_birth_iter"


def _gap(left, right):
    if left is None or right is None:
        return None
    return left - right


def _mean_record_value(records, key):
    values = [_optional_float(record.get(key)) for record in records]
    values = [value for value in values if value is not None]
    return None if not values else float(mean(values))


def _ratio(numerator, denominator):
    return None if denominator == 0 else float(numerator) / float(denominator)


def _optional_int(value):
    if value is None or value == "":
        return None
    return int(value)


def _optional_float(value):
    if value is None or value == "":
        return None
    return float(value)

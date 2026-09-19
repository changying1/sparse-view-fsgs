import csv
import json
from collections import Counter, defaultdict
from statistics import mean, median


ORIGIN_TYPE_NAMES = {
    0: "legacy",
    1: "clone",
    2: "split",
    3: "proximity",
}
ORIGIN_TYPE_ALIASES = {
    "root": "legacy",
    "legacy": "legacy",
    "clone": "clone",
    "split": "split",
    "proximity": "proximity",
    "proximity_child": "proximity",
}


def normalize_lineage_records(lineage):
    if lineage is None:
        return []
    if isinstance(lineage, list):
        return [_normalize_record(record) for record in lineage]
    if not isinstance(lineage, dict):
        raise ValueError("lineage must be a dict, list, or None.")

    if "records" in lineage:
        records = [_normalize_record(record) for record in lineage.get("records", [])]
    elif "lineage" in lineage:
        records = normalize_lineage_records(lineage["lineage"])
    elif "rgg_lineage" in lineage:
        records = normalize_lineage_records(lineage["rgg_lineage"])
    elif "uid" in lineage:
        records = _records_from_state(lineage)
    else:
        records = []

    h1_records = lineage.get("h1_registry", []) or lineage.get("proximity_records", [])
    if h1_records:
        _merge_h1_records(records, h1_records)
    return records


def analyze_origin_statistics(lineage):
    records = normalize_lineage_records(lineage)
    counts = Counter(_origin(record) for record in records)
    output = {}
    for origin in ("legacy", "clone", "split", "proximity"):
        origin_records = [record for record in records if _origin(record) == origin]
        output[origin] = {
            "count": counts.get(origin, 0),
            "alive": sum(1 for record in origin_records if _is_alive(record)),
            "dead": sum(1 for record in origin_records if not _is_alive(record)),
        }
    output["total"] = len(records)
    return output


def analyze_generation_statistics(lineage):
    records = normalize_lineage_records(lineage)
    by_generation = defaultdict(list)
    for record in records:
        generation = _optional_int(record.get("generation"))
        if generation is None:
            generation = 0
        by_generation[generation].append(record)
    rows = []
    for generation in sorted(by_generation):
        generation_records = by_generation[generation]
        rows.append(
            {
                "generation": generation,
                "count": len(generation_records),
                "alive": sum(1 for record in generation_records if _is_alive(record)),
                "dead": sum(1 for record in generation_records if not _is_alive(record)),
            }
        )
    return {
        "total": len(records),
        "max_generation": max(by_generation) if by_generation else None,
        "distribution": rows,
    }


def analyze_parent_child_statistics(lineage):
    records = normalize_lineage_records(lineage)
    by_uid = {int(record["uid"]): record for record in records if record.get("uid") is not None}
    children_by_parent = defaultdict(list)
    for record in records:
        source_uid = _optional_int(record.get("source_uid"))
        if source_uid is None or source_uid < 0:
            continue
        children_by_parent[source_uid].append(record)

    parents = []
    for source_uid in sorted(children_by_parent):
        children = children_by_parent[source_uid]
        alive_children = sum(1 for child in children if _is_alive(child))
        parent = by_uid.get(source_uid, {})
        parents.append(
            {
                "source_uid": source_uid,
                "parent_origin_type": _origin(parent) if parent else None,
                "parent_generation": _optional_int(parent.get("generation")) if parent else None,
                "child_count": len(children),
                "alive_child_count": alive_children,
                "dead_child_count": len(children) - alive_children,
                "child_survival_rate": _ratio(alive_children, len(children)),
                "children": [int(child["uid"]) for child in children if child.get("uid") is not None],
            }
        )

    child_counts = [parent["child_count"] for parent in parents]
    survival_rates = [
        parent["child_survival_rate"]
        for parent in parents
        if parent["child_survival_rate"] is not None
    ]
    return {
        "parent_count": len(parents),
        "total_children": sum(child_counts),
        "mean_child_count": _mean_or_none(child_counts),
        "median_child_count": _median_or_none(child_counts),
        "mean_child_survival_rate": _mean_or_none(survival_rates),
        "parents": parents,
    }


def analyze_proximity_reliability(lineage):
    records = normalize_lineage_records(lineage)
    proximity_records = [record for record in records if _origin(record) == "proximity"]
    death_reasons = Counter(_death_reason(record) for record in proximity_records)
    lifetimes = [_lifetime(record) for record in proximity_records]
    lifetimes = [value for value in lifetimes if value is not None]
    visible_rates = [
        _optional_float(record.get("visible_rate"))
        for record in proximity_records
        if _optional_float(record.get("visible_rate")) is not None
    ]
    unique_views = [
        _optional_int(record.get("postbirth_unique_real_views"))
        for record in proximity_records
        if _optional_int(record.get("postbirth_unique_real_views")) is not None
    ]
    return {
        "count": len(proximity_records),
        "alive": sum(1 for record in proximity_records if _is_alive(record)),
        "dead": sum(1 for record in proximity_records if not _is_alive(record)),
        "death_reasons": dict(sorted(death_reasons.items())),
        "mean_lifetime": _mean_or_none(lifetimes),
        "median_lifetime": _median_or_none(lifetimes),
        "mean_visible_rate": _mean_or_none(visible_rates),
        "median_visible_rate": _median_or_none(visible_rates),
        "mean_unique_real_views": _mean_or_none(unique_views),
        "records": [
            {
                "uid": record.get("uid"),
                "source_uid": record.get("source_uid"),
                "target_uid": record.get("target_uid"),
                "birth_iter": record.get("birth_iter"),
                "lifetime": _lifetime(record),
                "alive": _is_alive(record),
                "death_reason": _death_reason(record),
                "visible_rate": _optional_float(record.get("visible_rate")),
                "postbirth_real_opportunities": _optional_int(record.get("postbirth_real_opportunities")),
                "postbirth_real_visible_events": _optional_int(record.get("postbirth_real_visible_events")),
                "postbirth_unique_real_views": _optional_int(record.get("postbirth_unique_real_views")),
            }
            for record in proximity_records
        ],
    }


def analyze_lineage_statistics(lineage):
    records = normalize_lineage_records(lineage)
    normalized = {"records": records}
    return {
        "counts": {"total_gaussians": len(records)},
        "origin_type": analyze_origin_statistics(normalized),
        "generation": analyze_generation_statistics(normalized),
        "parent_child": analyze_parent_child_statistics(normalized),
        "proximity_child": analyze_proximity_reliability(normalized),
    }


def load_lineage_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_lineage_statistics_json(statistics, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(statistics, handle, indent=2, sort_keys=True)


def dump_lineage_statistics_csv(statistics, path):
    rows = _statistics_csv_rows(statistics)
    fieldnames = [
        "section",
        "key",
        "count",
        "alive",
        "dead",
        "generation",
        "source_uid",
        "child_count",
        "alive_child_count",
        "dead_child_count",
        "child_survival_rate",
        "value",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _statistics_csv_rows(statistics):
    rows = []
    for origin, values in statistics["origin_type"].items():
        if origin == "total":
            rows.append({"section": "origin_type", "key": "total", "count": values})
            continue
        rows.append(
            {
                "section": "origin_type",
                "key": origin,
                "count": values["count"],
                "alive": values["alive"],
                "dead": values["dead"],
            }
        )
    for item in statistics["generation"]["distribution"]:
        rows.append({"section": "generation", "key": str(item["generation"]), **item})
    for parent in statistics["parent_child"]["parents"]:
        rows.append({"section": "parent_child", "key": str(parent["source_uid"]), **parent})
    proximity = statistics["proximity_child"]
    for key in ("count", "alive", "dead", "mean_lifetime", "mean_visible_rate", "mean_unique_real_views"):
        rows.append({"section": "proximity_child", "key": key, "value": proximity.get(key)})
    for reason, count in proximity["death_reasons"].items():
        rows.append({"section": "proximity_child_death_reason", "key": reason, "count": count})
    return rows


def _records_from_state(state):
    uids = _as_list(state.get("uid", []))
    birth_iters = _as_list(state.get("birth_iter", []))
    source_uids = _as_list(state.get("source_uid", []))
    target_uids = _as_list(state.get("target_uid", []))
    generations = _as_list(state.get("generation", []))
    origin_types = _as_list(state.get("origin_type", state.get("birth_type", [])))
    records = []
    for index, uid in enumerate(uids):
        records.append(
            _normalize_record(
                {
                    "uid": uid,
                    "birth_iter": _get_index(birth_iters, index, -1),
                    "source_uid": _get_index(source_uids, index, -1),
                    "target_uid": _get_index(target_uids, index, -1),
                    "generation": _get_index(generations, index, 0),
                    "origin_type": _get_index(origin_types, index, 0),
                    "alive": True,
                }
            )
        )
    return records


def _merge_h1_records(records, h1_records):
    by_uid = {int(record["uid"]): record for record in records if record.get("uid") is not None}
    for h1_record in h1_records:
        normalized_h1 = _normalize_record(h1_record)
        uid = normalized_h1.get("uid")
        if uid is None:
            continue
        if int(uid) in by_uid:
            by_uid[int(uid)].update(normalized_h1)
        else:
            records.append(normalized_h1)
            by_uid[int(uid)] = normalized_h1


def _normalize_record(record):
    if not isinstance(record, dict):
        raise ValueError("lineage records must be dicts.")
    origin = record.get("origin_type", record.get("birth_type"))
    normalized = dict(record)
    normalized["uid"] = _optional_int(record.get("uid"))
    normalized["birth_iter"] = _optional_int(record.get("birth_iter"))
    normalized["source_uid"] = _optional_int(record.get("source_uid"))
    normalized["target_uid"] = _optional_int(record.get("target_uid"))
    normalized["generation"] = _optional_int(record.get("generation"))
    normalized["origin_type"] = _normalize_origin(origin)
    return normalized


def _normalize_origin(value):
    if value is None or value == "":
        return "legacy"
    if isinstance(value, str):
        return ORIGIN_TYPE_ALIASES.get(value.lower(), value.lower())
    return ORIGIN_TYPE_NAMES.get(int(value), str(int(value)))


def _origin(record):
    return _normalize_origin(record.get("origin_type", record.get("birth_type")))


def _is_alive(record):
    if "alive" in record:
        return bool(record["alive"])
    death_reason = _death_reason(record)
    death_iter = _optional_int(record.get("death_iter"))
    return death_reason in ("alive", "none", "") and death_iter is None


def _death_reason(record):
    reason = record.get("death_reason")
    if reason is None or reason == "":
        return "alive" if bool(record.get("alive", True)) else "unknown"
    return str(reason)


def _lifetime(record):
    terminal_age = _optional_int(record.get("terminal_age"))
    if terminal_age is not None:
        return terminal_age
    birth_iter = _optional_int(record.get("birth_iter"))
    death_iter = _optional_int(record.get("death_iter"))
    if birth_iter is not None and death_iter is not None:
        return death_iter - birth_iter
    age_at_export = _optional_int(record.get("age_at_export"))
    if age_at_export is not None:
        return age_at_export
    observation_end_iter = _optional_int(record.get("observation_end_iter"))
    if birth_iter is not None and observation_end_iter is not None:
        return observation_end_iter - birth_iter
    return None


def _as_list(value):
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _get_index(values, index, default):
    return values[index] if index < len(values) else default


def _optional_int(value):
    if value is None or value == "":
        return None
    return int(value)


def _optional_float(value):
    if value is None or value == "":
        return None
    return float(value)


def _ratio(numerator, denominator):
    return None if denominator == 0 else float(numerator) / float(denominator)


def _mean_or_none(values):
    return None if not values else float(mean(values))


def _median_or_none(values):
    return None if not values else float(median(values))

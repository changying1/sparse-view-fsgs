import csv
import json

from utils.rgg_lineage_analysis import normalize_lineage_records
from utils.rgg_h1_analysis import STATUS_FAILURE, STATUS_RETAINED, classify_h1_record


LINEAGE_RECORD_FIELDS = [
    "uid",
    "source_uid",
    "target_uid",
    "generation",
    "origin_type",
    "birth_iter",
    "death_iter",
    "death_reason",
    "alive",
    "lifetime",
    "visible_rate",
    "postbirth_real_opportunities",
    "postbirth_real_visible_events",
    "postbirth_unique_real_views",
    "observation_end_iter",
    "h1_outcome",
    "h1_success",
    "h1_label_iter",
]


def export_lineage_records(lineage, snapshot_age=None, future_horizon=None):
    normalized_records = normalize_lineage_records(lineage)
    presence_records = _presence_records(lineage)
    records = []
    for index, record in enumerate(normalized_records):
        presence = presence_records[index] if index < len(presence_records) else {}
        exported = _export_record(record, presence)
        if snapshot_age is not None and future_horizon is not None:
            exported.update(_h1_outcome_fields(record, snapshot_age, future_horizon))
        records.append(exported)
    return records


def dump_lineage_records_json(records, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, sort_keys=True)


def dump_lineage_records_csv(records, path):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LINEAGE_RECORD_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in LINEAGE_RECORD_FIELDS})


def _export_record(record, presence):
    return {
        "uid": record.get("uid") if presence.get("uid") else None,
        "source_uid": record.get("source_uid") if presence.get("source_uid") else None,
        "target_uid": record.get("target_uid") if presence.get("target_uid") else None,
        "generation": record.get("generation") if presence.get("generation") else None,
        "origin_type": record.get("origin_type") if presence.get("origin_type") else None,
        "birth_iter": record.get("birth_iter") if presence.get("birth_iter") else None,
        "death_iter": record.get("death_iter") if presence.get("death_iter") else None,
        "death_reason": record.get("death_reason") if presence.get("death_reason") else None,
        "alive": record.get("alive") if presence.get("alive") else None,
        "lifetime": _first_present(record, presence, ("lifetime", "terminal_age", "age_at_export")),
        "visible_rate": record.get("visible_rate") if presence.get("visible_rate") else None,
        "postbirth_real_opportunities": (
            record.get("postbirth_real_opportunities")
            if presence.get("postbirth_real_opportunities")
            else None
        ),
        "postbirth_real_visible_events": (
            record.get("postbirth_real_visible_events")
            if presence.get("postbirth_real_visible_events")
            else None
        ),
        "postbirth_unique_real_views": (
            record.get("postbirth_unique_real_views")
            if presence.get("postbirth_unique_real_views")
            else None
        ),
        "observation_end_iter": record.get("observation_end_iter") if presence.get("observation_end_iter") else None,
        "h1_outcome": record.get("h1_outcome") if presence.get("h1_outcome") else None,
        "h1_success": record.get("h1_success") if presence.get("h1_success") else None,
        "h1_label_iter": record.get("h1_label_iter") if presence.get("h1_label_iter") else None,
        "age_snapshots": record.get("age_snapshots") if presence.get("age_snapshots") else None,
    }


def _first_present(record, presence, keys):
    for key in keys:
        if presence.get(key):
            return record.get(key)
    return None


def _presence_records(lineage):
    if lineage is None:
        return []
    if isinstance(lineage, list):
        return [_record_presence(record) for record in lineage]
    if not isinstance(lineage, dict):
        raise ValueError("lineage must be a dict, list, or None.")
    if "records" in lineage:
        presence = [_record_presence(record) for record in lineage.get("records", [])]
    elif "lineage" in lineage:
        presence = _presence_records(lineage["lineage"])
    elif "rgg_lineage" in lineage:
        presence = _presence_records(lineage["rgg_lineage"])
    elif "uid" in lineage:
        presence = _state_presence_records(lineage)
    else:
        presence = []

    h1_records = lineage.get("h1_registry", []) or lineage.get("proximity_records", [])
    if h1_records:
        presence = _merge_presence_records(presence, h1_records)
    return presence


def _record_presence(record):
    if not isinstance(record, dict):
        raise ValueError("lineage records must be dicts.")
    presence = {field: field in record for field in LINEAGE_RECORD_FIELDS}
    presence["_uid"] = record.get("uid")
    presence["origin_type"] = "origin_type" in record or "birth_type" in record
    presence["lifetime"] = "lifetime" in record
    presence["terminal_age"] = "terminal_age" in record
    presence["age_at_export"] = "age_at_export" in record
    presence["age_snapshots"] = "age_snapshots" in record
    return presence


def _state_presence_records(state):
    uids = _as_list(state.get("uid", []))
    source = {
        "uid": "uid" in state,
        "source_uid": "source_uid" in state,
        "target_uid": "target_uid" in state,
        "generation": "generation" in state,
        "origin_type": "origin_type" in state or "birth_type" in state,
        "birth_iter": "birth_iter" in state,
    }
    return [
        {
            **{field: False for field in LINEAGE_RECORD_FIELDS},
            **source,
            "terminal_age": False,
            "age_at_export": False,
            "age_snapshots": False,
            "_uid": uids[index] if index < len(uids) else None,
        }
        for index in range(len(uids))
    ]


def _merge_presence_records(presence, h1_records):
    merged = list(presence)
    uid_to_index = {
        int(item["_uid"]): index
        for index, item in enumerate(merged)
        if item.get("_uid") is not None
    }
    for h1_record in h1_records:
        h1_presence = _record_presence(h1_record)
        uid = h1_record.get("uid") if isinstance(h1_record, dict) else None
        index = _find_uid_index(uid, uid_to_index)
        if index is None:
            merged.append(h1_presence)
            if h1_presence.get("_uid") is not None:
                uid_to_index[int(h1_presence["_uid"])] = len(merged) - 1
        else:
            merged[index].update({key: value or merged[index].get(key, False) for key, value in h1_presence.items()})
    return merged


def _find_uid_index(uid, uid_to_index):
    try:
        uid = int(uid)
    except (TypeError, ValueError):
        return None
    return uid_to_index.get(uid)


def _as_list(value):
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _h1_outcome_fields(record, snapshot_age, future_horizon):
    classified = classify_h1_record(record, snapshot_age=snapshot_age, future_horizon=future_horizon)
    outcome = classified["status"]
    success = None
    label_iter = None
    if outcome == STATUS_FAILURE:
        success = False
        label_iter = record.get("death_iter")
    elif outcome == STATUS_RETAINED:
        success = True
        label_iter = classified.get("prediction_end_iter")
    return {
        "h1_outcome": outcome,
        "h1_success": success,
        "h1_label_iter": label_iter,
    }

import csv
import json

from utils.rgg_lineage_export import (
    LINEAGE_RECORD_FIELDS,
    dump_lineage_records_csv,
    dump_lineage_records_json,
    export_lineage_records,
)


def test_empty_lineage_exports_empty_records(tmp_path):
    records = export_lineage_records({})

    assert records == []

    json_path = tmp_path / "rgg_lineage_records.json"
    dump_lineage_records_json(records, json_path)
    assert json.loads(json_path.read_text(encoding="utf-8")) == []


def test_single_parent_multiple_children_preserves_source_uid():
    lineage = {
        "records": [
            {"uid": 10, "source_uid": -1, "target_uid": -1, "generation": 0, "origin_type": "legacy"},
            {"uid": 11, "source_uid": 10, "target_uid": -1, "generation": 1, "origin_type": "clone"},
            {"uid": 12, "source_uid": 10, "target_uid": -1, "generation": 1, "origin_type": "split"},
            {"uid": 13, "source_uid": 10, "target_uid": 99, "generation": 1, "origin_type": "proximity_child"},
        ]
    }

    records = export_lineage_records(lineage)

    assert [record["uid"] for record in records] == [10, 11, 12, 13]
    assert [record["source_uid"] for record in records] == [-1, 10, 10, 10]
    assert [record["origin_type"] for record in records] == ["legacy", "clone", "split", "proximity"]


def test_proximity_child_observation_fields_are_preserved():
    lineage = {
        "records": [
            {
                "uid": 21,
                "source_uid": 10,
                "target_uid": 20,
                "generation": 2,
                "birth_type": "proximity_child",
                "birth_iter": 100,
                "death_iter": 180,
                "death_reason": "training_prune",
                "alive": False,
                "terminal_age": 80,
                "visible_rate": 0.375,
                "postbirth_real_opportunities": 8,
                "postbirth_real_visible_events": 3,
                "postbirth_unique_real_views": 2,
            }
        ]
    }

    records = export_lineage_records(lineage)
    record = records[0]

    assert record["target_uid"] == 20
    assert record["origin_type"] == "proximity"
    assert record["visible_rate"] == 0.375
    assert record["lifetime"] == 80
    assert record["death_reason"] == "training_prune"
    assert record["postbirth_real_opportunities"] == 8
    assert record["postbirth_real_visible_events"] == 3
    assert record["postbirth_unique_real_views"] == 2


def test_missing_fields_remain_none():
    records = export_lineage_records({"records": [{"uid": 1}]})

    assert records == [
        {
            "uid": 1,
            "source_uid": None,
            "target_uid": None,
            "generation": None,
            "origin_type": None,
            "birth_iter": None,
            "death_iter": None,
            "death_reason": None,
            "alive": None,
            "lifetime": None,
            "visible_rate": None,
            "postbirth_real_opportunities": None,
            "postbirth_real_visible_events": None,
            "postbirth_unique_real_views": None,
            "observation_end_iter": None,
            "h1_outcome": None,
            "h1_success": None,
            "h1_label_iter": None,
            "age_snapshots": None,
        }
    ]


def test_csv_header_is_complete_and_ordered(tmp_path):
    records = export_lineage_records({"records": [{"uid": 1, "source_uid": -1}]})
    csv_path = tmp_path / "rgg_lineage_records.csv"

    dump_lineage_records_csv(records, csv_path)

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)

    assert header == LINEAGE_RECORD_FIELDS


def test_export_can_add_h1_outcome_fields_from_h1_logic():
    records = export_lineage_records(
        {
            "records": [
                {
                    "uid": 7,
                    "source_uid": 1,
                    "birth_iter": 100,
                    "alive": False,
                    "death_iter": 150,
                    "death_reason": "training_prune",
                    "age_snapshots": {"50": {"visible_rate": 0.25}},
                }
            ]
        },
        snapshot_age=50,
        future_horizon=50,
    )

    assert records[0]["h1_outcome"] == "failure"
    assert records[0]["h1_success"] is False
    assert records[0]["h1_label_iter"] == 150

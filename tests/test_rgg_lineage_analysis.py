import json

from utils.rgg_lineage_analysis import (
    analyze_generation_statistics,
    analyze_lineage_statistics,
    analyze_origin_statistics,
    analyze_parent_child_statistics,
    analyze_proximity_reliability,
    dump_lineage_statistics_csv,
    dump_lineage_statistics_json,
)


def test_empty_lineage_statistics(tmp_path):
    stats = analyze_lineage_statistics({"records": []})

    assert stats["counts"]["total_gaussians"] == 0
    assert stats["origin_type"]["total"] == 0
    assert stats["generation"]["distribution"] == []
    assert stats["parent_child"]["parents"] == []
    assert stats["proximity_child"]["count"] == 0

    json_path = tmp_path / "rgg_lineage_statistics.json"
    csv_path = tmp_path / "rgg_lineage_statistics.csv"
    dump_lineage_statistics_json(stats, json_path)
    dump_lineage_statistics_csv(stats, csv_path)
    assert json.loads(json_path.read_text(encoding="utf-8"))["counts"]["total_gaussians"] == 0
    assert csv_path.read_text(encoding="utf-8").startswith("section,key,count")


def test_single_parent_multiple_children_statistics():
    lineage = {
        "records": [
            {"uid": 1, "generation": 0, "birth_type": "legacy", "source_uid": -1, "alive": True},
            {"uid": 2, "generation": 1, "birth_type": "clone", "source_uid": 1, "alive": True},
            {
                "uid": 3,
                "generation": 1,
                "birth_type": "split",
                "source_uid": 1,
                "alive": False,
                "death_reason": "training_prune",
                "birth_iter": 10,
                "death_iter": 25,
            },
            {"uid": 4, "generation": 1, "birth_type": "proximity_child", "source_uid": 1, "alive": True},
        ]
    }

    parent_stats = analyze_parent_child_statistics(lineage)

    assert parent_stats["parent_count"] == 1
    assert parent_stats["total_children"] == 3
    assert parent_stats["parents"][0]["source_uid"] == 1
    assert parent_stats["parents"][0]["child_count"] == 3
    assert parent_stats["parents"][0]["alive_child_count"] == 2
    assert parent_stats["parents"][0]["dead_child_count"] == 1
    assert parent_stats["parents"][0]["child_survival_rate"] == 2 / 3
    assert parent_stats["parents"][0]["children"] == [2, 3, 4]


def test_generation_and_origin_statistics_from_tensor_like_state():
    lineage = {
        "uid": [0, 1, 2, 3],
        "source_uid": [-1, 0, 0, 2],
        "target_uid": [-1, -1, -1, -1],
        "generation": [0, 1, 1, 2],
        "origin_type": [0, 1, 2, 3],
        "birth_iter": [-1, 10, 11, 12],
    }

    origin_stats = analyze_origin_statistics(lineage)
    generation_stats = analyze_generation_statistics(lineage)

    assert origin_stats["legacy"]["count"] == 1
    assert origin_stats["clone"]["count"] == 1
    assert origin_stats["split"]["count"] == 1
    assert origin_stats["proximity"]["count"] == 1
    assert generation_stats["max_generation"] == 2
    assert generation_stats["distribution"] == [
        {"generation": 0, "count": 1, "alive": 1, "dead": 0},
        {"generation": 1, "count": 2, "alive": 2, "dead": 0},
        {"generation": 2, "count": 1, "alive": 1, "dead": 0},
    ]


def test_survival_and_proximity_reliability_statistics():
    lineage = {
        "records": [
            {"uid": 0, "origin_type": "legacy", "generation": 0, "source_uid": -1, "alive": True},
            {
                "uid": 1,
                "origin_type": "proximity",
                "generation": 1,
                "source_uid": 0,
                "target_uid": 2,
                "birth_iter": 100,
                "death_iter": 160,
                "death_reason": "training_prune",
                "alive": False,
                "visible_rate": 0.25,
                "postbirth_real_opportunities": 8,
                "postbirth_real_visible_events": 2,
                "postbirth_unique_real_views": 1,
            },
            {
                "uid": 2,
                "origin_type": "proximity_child",
                "generation": 1,
                "source_uid": 0,
                "target_uid": 1,
                "birth_iter": 120,
                "observation_end_iter": 220,
                "alive": True,
                "visible_rate": 0.75,
                "postbirth_real_opportunities": 4,
                "postbirth_real_visible_events": 3,
                "postbirth_unique_real_views": 2,
            },
        ]
    }

    proximity_stats = analyze_proximity_reliability(lineage)
    parent_stats = analyze_parent_child_statistics(lineage)

    assert proximity_stats["count"] == 2
    assert proximity_stats["alive"] == 1
    assert proximity_stats["dead"] == 1
    assert proximity_stats["death_reasons"] == {"alive": 1, "training_prune": 1}
    assert proximity_stats["mean_lifetime"] == 80.0
    assert proximity_stats["mean_visible_rate"] == 0.5
    assert proximity_stats["mean_unique_real_views"] == 1.5
    assert parent_stats["parents"][0]["child_survival_rate"] == 0.5

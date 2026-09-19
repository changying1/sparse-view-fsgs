import argparse
import os
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.rgg_lineage_analysis import (
    analyze_lineage_statistics,
    dump_lineage_statistics_csv,
    dump_lineage_statistics_json,
    load_lineage_json,
)


def main():
    parser = argparse.ArgumentParser(description="Analyze offline RGG lineage metadata.")
    parser.add_argument("--input", required=True, help="Path to a checkpoint or lineage JSON file.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for rgg_lineage_statistics.json/csv. Defaults to the input directory.",
    )
    args = parser.parse_args()

    lineage = load_lineage_input(args.input)
    statistics = analyze_lineage_statistics(lineage)

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.input)) or "."
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "rgg_lineage_statistics.json")
    csv_path = os.path.join(output_dir, "rgg_lineage_statistics.csv")
    dump_lineage_statistics_json(statistics, json_path)
    dump_lineage_statistics_csv(statistics, csv_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


def load_lineage_input(path):
    if path.lower().endswith(".json"):
        return load_lineage_json(path)
    return _load_checkpoint_lineage(path)


def _load_checkpoint_lineage(path):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to read checkpoint inputs.") from exc

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    if isinstance(checkpoint, dict):
        if "rgg_lineage" in checkpoint:
            return checkpoint["rgg_lineage"]
        if "gaussian_state" in checkpoint:
            return _lineage_from_capture(checkpoint["gaussian_state"])
        if "uid" in checkpoint:
            return checkpoint
    if isinstance(checkpoint, (tuple, list)):
        if len(checkpoint) == 2 and isinstance(checkpoint[0], (tuple, list)):
            return _lineage_from_capture(checkpoint[0])
        return _lineage_from_capture(checkpoint)
    raise ValueError("checkpoint does not contain recognizable RGG lineage state.")


def _lineage_from_capture(capture):
    if not isinstance(capture, (tuple, list)):
        raise ValueError("gaussian capture must be a tuple or list.")
    if len(capture) >= 14 and isinstance(capture[-1], dict) and "uid" in capture[-1]:
        return capture[-1]
    raise ValueError("gaussian capture does not contain RGG lineage state.")


if __name__ == "__main__":
    main()

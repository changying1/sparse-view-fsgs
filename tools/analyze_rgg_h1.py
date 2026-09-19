import argparse
import json
import os
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.rgg_h1_analysis import analyze_h1_records, dump_analysis_json, load_h1_records


def main():
    parser = argparse.ArgumentParser(description="Analyze RGG H1 post-birth support diagnostics.")
    parser.add_argument("--input", required=True, help="Path to h1_proximity_postbirth_support.json.")
    parser.add_argument("--snapshot-age", required=True, type=int)
    parser.add_argument("--future-horizon", required=True, type=int)
    parser.add_argument("--output", default=None, help="Optional output JSON path.")
    args = parser.parse_args()

    records = load_h1_records(args.input)
    analysis = analyze_h1_records(
        records,
        snapshot_age=args.snapshot_age,
        future_horizon=args.future_horizon,
    )
    if args.output:
        dump_analysis_json(analysis, args.output)
    print(json.dumps(analysis, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

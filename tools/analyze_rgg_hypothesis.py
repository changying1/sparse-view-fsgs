import argparse
import os
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.rgg_hypothesis_analysis import (
    analyze_h2_hypothesis,
    dump_h2_report_csv,
    dump_h2_report_json,
    load_lineage_records_json,
)


def main():
    parser = argparse.ArgumentParser(description="Validate RGG H2 lineage reliability hypotheses.")
    parser.add_argument("--input", required=True, help="Path to rgg_lineage_records.json.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for rgg_h2_validation_report.json/csv. Defaults to the input directory.",
    )
    parser.add_argument(
        "--split-iter",
        default=None,
        type=int,
        help="Global history/future split iteration. Defaults to the median child birth_iter.",
    )
    parser.add_argument(
        "--min-history-children",
        default=2,
        type=int,
        help="Minimum number of labeled history children required for a source to enter high/low Q groups.",
    )
    parser.add_argument("--snapshot-age", default=50, type=int, help="H1 snapshot age used for H2 child outcomes.")
    parser.add_argument("--future-horizon", default=50, type=int, help="H1 future horizon used for H2 child outcomes.")
    args = parser.parse_args()

    records = load_lineage_records_json(args.input)
    report = analyze_h2_hypothesis(
        records,
        split_iter=args.split_iter,
        min_history_children=args.min_history_children,
        snapshot_age=args.snapshot_age,
        future_horizon=args.future_horizon,
    )

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.input)) or "."
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "rgg_h2_validation_report.json")
    csv_path = os.path.join(output_dir, "rgg_h2_validation_report.csv")
    dump_h2_report_json(report, json_path)
    dump_h2_report_csv(report, csv_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()

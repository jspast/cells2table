import argparse
import json
import os
import re
import sys

import numpy as np
from docling_metrics_table import TableMetric, TableMetricHTMLInputSample
from docling_metrics_table.docling_metrics_table import TableMetricKind


def clean_table_in_html(html: str) -> str:
    # Find the first <table>...</table> block (including newlines)
    table_match = re.search(r"<table\b[^>]*>.*?</table>", html, re.DOTALL | re.IGNORECASE)

    if not table_match:
        return ""

    table_html = table_match.group(0)

    # Remove <tbody> and </tbody> tags (case-insensitive)
    table_html = re.sub(r"</?tbody\b[^>]*>", "", table_html, flags=re.IGNORECASE)

    return table_html


def main():
    parser = argparse.ArgumentParser(description="T-LAG scorer for PulseBench-Tab")
    parser.add_argument("--gt", required=True, help="Directory with ground truth HTML files")
    parser.add_argument("--pred", required=True, help="Directory with predicted HTML files")
    parser.add_argument("--output", default=None, help="Output JSON file (default: stdout)")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    args = parser.parse_args()

    gt_files = {f.replace(".html", "") for f in os.listdir(args.gt) if f.endswith(".html")}
    pred_files = {f.replace(".html", "") for f in os.listdir(args.pred) if f.endswith(".html")}

    common = sorted(gt_files & pred_files)
    missing = sorted(gt_files - pred_files)

    print(f"Ground truth: {len(gt_files)} files", file=sys.stderr)
    print(f"Predictions:  {len(pred_files)} files", file=sys.stderr)
    print(f"Common:       {len(common)} files", file=sys.stderr)
    print(f"Missing:      {len(missing)} files", file=sys.stderr)

    tasks = [
        (sid, os.path.join(args.gt, f"{sid}.html"), os.path.join(args.pred, f"{sid}.html"))
        for sid in common
    ]

    table_metric = TableMetric([TableMetricKind.TEDS])

    scores = {}
    for sample_id, gt_path, pred_path in tasks:
        try:
            with open(gt_path) as f:
                gt_html = f.read()
            with open(pred_path) as f:
                pred_html = f.read()
            if not pred_html.strip():
                continue

            html_sample = TableMetricHTMLInputSample(
                id=sample_id,
                html_a=clean_table_in_html(gt_html),
                html_b=clean_table_in_html(pred_html),
                structure_only=True,
            )
            html_evaluation = table_metric.evaluate_sample(html_sample)
            if html_evaluation.teds:
                scores[sample_id] = html_evaluation.teds.teds

        except Exception as e:
            print(e, "\n", clean_table_in_html(gt_html), "\n", clean_table_in_html(pred_html))
            # return

    # Compute aggregate stats
    valid_scores = [s for s in scores.values()]
    arr = np.array(valid_scores)

    summary = {
        "n_total": len(gt_files),
        "n_scored": len(valid_scores),
        "n_missing": len(gt_files) - len(valid_scores),
        "coverage_pct": round(len(valid_scores) / len(gt_files) * 100, 1),
        "mean": round(float(np.mean(arr)), 4) if len(arr) > 0 else None,
        "median": round(float(np.median(arr)), 4) if len(arr) > 0 else None,
        "std": round(float(np.std(arr)), 4) if len(arr) > 0 else None,
        "p10": round(float(np.percentile(arr, 10)), 4) if len(arr) > 0 else None,
        "p25": round(float(np.percentile(arr, 25)), 4) if len(arr) > 0 else None,
        "p75": round(float(np.percentile(arr, 75)), 4) if len(arr) > 0 else None,
        "p90": round(float(np.percentile(arr, 90)), 4) if len(arr) > 0 else None,
        "perfect_count": int(np.sum(arr >= 0.9999)) if len(arr) > 0 else 0,
    }

    output = {
        "summary": summary,
        "per_sample": {sid: round(s, 6) for sid, s in scores.items()},
    }

    # Print summary to stderr
    print("\nResults:", file=sys.stderr)
    print(f"  T-LAG Score (mean): {summary['mean']}", file=sys.stderr)
    print(f"  Median:             {summary['median']}", file=sys.stderr)
    print(f"  Coverage:           {summary['coverage_pct']}%", file=sys.stderr)
    print(f"  Perfect (=1.0):     {summary['perfect_count']}", file=sys.stderr)

    # Output JSON
    json_str = json.dumps(output, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w") as f:
            f.write(json_str)
        print(f"\nSaved to {args.output}", file=sys.stderr)
    else:
        print(json_str)


if __name__ == "__main__":
    main()

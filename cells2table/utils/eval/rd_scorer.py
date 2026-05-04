import argparse
import json
import os
import re
import sys

import numpy as np
import numpy.typing as npt
from Levenshtein import distance as levenshtein_distance
from lxml import etree  # ty:ignore[unresolved-import]

BATCH_SIZE = 150

# Scoring parameters (you can adjust these as needed)
S_ROW_MATCH = 5  # Match score for row alignment
G_ROW = -3  # Gap penalty for row alignment (insertion/deletion of a row)
S_CELL_MATCH = 1  # Match score for cell matching
P_CELL_MISMATCH = -1  # Penalty for cell mismatch
G_COL = -1  # Gap penalty for column alignment


def cell_match_score(cell1: str | None, cell2: str | None) -> float:
    """Compute the match score between two cells considering partial matches."""
    if cell1 is None or cell2 is None:
        return P_CELL_MISMATCH  # Penalty for gaps or mismatches
    if cell1 == cell2:
        return S_CELL_MATCH  # Cells are identical

    # Compute the Levenshtein distance using the optimized library
    distance = levenshtein_distance(cell1, cell2)
    max_len = max(len(cell1), len(cell2))
    if max_len == 0:
        normalized_distance = 0.0  # Both cells are empty strings
    else:
        normalized_distance = distance / max_len
    similarity = 1.0 - normalized_distance  # Similarity between 0 and 1
    match_score = P_CELL_MISMATCH + similarity * (S_CELL_MATCH - P_CELL_MISMATCH)
    return match_score


def needleman_wunsch(
    seq1: list[str], seq2: list[str], gap_penalty: int
) -> tuple[list[str | None], list[str | None], float]:
    """
    Perform Needleman-Wunsch alignment between two sequences with free end gaps.

    Parameters:
    seq1, seq2: sequences to align (lists of strings)
    gap_penalty: penalty for gaps (insertions/deletions)

    Returns:
    alignment_a, alignment_b: aligned sequences with gaps represented by None
    score: total alignment score
    """
    m = len(seq1)
    n = len(seq2)

    # Initialize the scoring matrix
    score_matrix = np.zeros((m + 1, n + 1), dtype=np.float32)
    traceback = np.full((m + 1, n + 1), None)

    # Initialize the first row and column (no gap penalties for leading gaps)
    for i in range(1, m + 1):
        traceback[i, 0] = "up"
    for j in range(1, n + 1):
        traceback[0, j] = "left"

    # Fill the rest of the matrix
    for i in range(1, m + 1):
        seq1_i = seq1[i - 1]
        for j in range(1, n + 1):
            seq2_j = seq2[j - 1]
            match = score_matrix[i - 1, j - 1] + cell_match_score(seq1_i, seq2_j)
            delete = score_matrix[i - 1, j] + gap_penalty
            insert = score_matrix[i, j - 1] + gap_penalty
            max_score = max(match, delete, insert)
            score_matrix[i, j] = max_score
            if max_score == match:
                traceback[i, j] = "diag"
            elif max_score == delete:
                traceback[i, j] = "up"
            else:
                traceback[i, j] = "left"

    # Traceback from the position with the highest score in the last row or column
    i, j = m, n
    max_score = score_matrix[i, j]
    max_i, max_j = i, j
    # Find the maximum score in the last row and column for free end gaps
    last_row = score_matrix[:, n]
    last_col = score_matrix[m, :]
    if last_row.max() > max_score:
        max_i = last_row.argmax()
        max_j = n
        max_score = last_row[max_i]
    if last_col.max() > max_score:
        max_i = m
        max_j = last_col.argmax()
        max_score = last_col[max_j]

    # Traceback to get the aligned sequences
    alignment_a: list[str | None] = []
    alignment_b: list[str | None] = []
    i, j = max_i, max_j
    while i > 0 or j > 0:
        tb_direction = traceback[i, j]
        if i > 0 and j > 0 and tb_direction == "diag":
            alignment_a.insert(0, seq1[i - 1])
            alignment_b.insert(0, seq2[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or tb_direction == "up"):
            alignment_a.insert(0, seq1[i - 1])
            alignment_b.insert(0, None)  # Gap in seq2
            i -= 1
        elif j > 0 and (i == 0 or tb_direction == "left"):
            alignment_a.insert(0, None)  # Gap in seq1
            alignment_b.insert(0, seq2[j - 1])
            j -= 1
        else:
            break  # Should not reach here

    return alignment_a, alignment_b, max_score


def table_similarity(ground_truth: npt.NDArray[np.str_], prediction: npt.NDArray[np.str_]) -> float:
    """
    Compute the similarity between two tables represented as ndarrays of strings,
    allowing for a subset of rows at the top or bottom without penalization (to avoid penalizing subtable cropping).

    Parameters:
    ground_truth, prediction: ndarrays of strings representing the tables

    Returns:
    similarity: similarity score between 0 and 1
    """

    # Remove newlines and normalize whitespace in cells
    def normalize_cell(cell: str) -> str:
        return "".join(cell.replace("\n", " ").replace("-", "").split()).replace(" ", "")

    # Apply normalization to both ground truth and prediction arrays
    vectorized_normalize = np.vectorize(normalize_cell)
    ground_truth = vectorized_normalize(ground_truth)
    prediction = vectorized_normalize(prediction)

    # Convert to lists of lists for easier manipulation
    gt_rows = [list(row) for row in ground_truth]
    pred_rows = [list(row) for row in prediction]

    # Precompute the column alignment scores between all pairs of rows
    m = len(gt_rows)
    n = len(pred_rows)
    row_match_scores = np.zeros((m, n), dtype=np.float32)

    for i in range(m):
        gt_row = gt_rows[i]
        for j in range(n):
            pred_row = pred_rows[j]
            # Align columns of the two rows
            _, _, col_score = needleman_wunsch(gt_row, pred_row, G_COL)
            # Adjusted row match score
            row_match_scores[i, j] = col_score + S_ROW_MATCH

    # Initialize the scoring matrix for row alignment with free end gaps
    score_matrix = np.zeros((m + 1, n + 1), dtype=np.float32)
    traceback = np.full((m + 1, n + 1), None)

    # No gap penalties for leading gaps
    for i in range(1, m + 1):
        traceback[i, 0] = "up"
    for j in range(1, n + 1):
        traceback[0, j] = "left"

    # Fill the rest of the scoring matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = score_matrix[i - 1, j - 1] + row_match_scores[i - 1, j - 1]
            delete = score_matrix[i - 1, j] + G_ROW
            insert = score_matrix[i, j - 1] + G_ROW
            max_score = max(match, delete, insert)
            score_matrix[i, j] = max_score
            if max_score == match:
                traceback[i, j] = "diag"
            elif max_score == delete:
                traceback[i, j] = "up"
            else:
                traceback[i, j] = "left"

    # Traceback from the position with the highest score in the last row or column
    i, j = m, n
    max_score = score_matrix[i, j]
    max_i, max_j = i, j
    # Find the maximum score in the last row and column for free end gaps
    last_row = score_matrix[:, n]
    last_col = score_matrix[m, :]
    if last_row.max() > max_score:
        max_i = last_row.argmax()
        max_j = n
        max_score = last_row[max_i]
    if last_col.max() > max_score:
        max_i = m
        max_j = last_col.argmax()
        max_score = last_col[max_j]

    # Traceback to get the aligned rows
    alignment_gt_rows: list[list[str | None]] = []
    alignment_pred_rows: list[list[str | None]] = []
    i, j = max_i, max_j
    while i > 0 or j > 0:
        tb_direction = traceback[i, j]
        if i > 0 and j > 0 and tb_direction == "diag":
            alignment_gt_rows.insert(0, gt_rows[i - 1])
            alignment_pred_rows.insert(0, pred_rows[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or tb_direction == "up"):
            alignment_gt_rows.insert(0, gt_rows[i - 1])
            alignment_pred_rows.insert(0, [None] * len(gt_rows[i - 1]))  # Gap in prediction
            i -= 1
        elif j > 0 and (i == 0 or tb_direction == "left"):
            alignment_gt_rows.insert(0, [None] * len(pred_rows[j - 1]))  # Gap in ground truth
            alignment_pred_rows.insert(0, pred_rows[j - 1])
            j -= 1
        else:
            break  # Should not reach here

    # Compute the actual total score
    actual_total_score = max_score

    # Compute the total possible score
    num_aligned_rows = len(alignment_gt_rows)
    if num_aligned_rows == 0:
        return 0.0  # Avoid division by zero
    max_row_score = num_aligned_rows * (S_ROW_MATCH + len(gt_rows[0]) * S_CELL_MATCH)
    total_possible_score = max_row_score

    # Normalize the similarity score
    similarity = actual_total_score / total_possible_score
    return max(0.0, min(similarity, 1.0))


def html_to_numpy(html_string: str) -> npt.NDArray[np.str_]:
    dom_tree = etree.HTML(html_string, parser=etree.HTMLParser())
    table_rows: list[list[str]] = []
    span_info: dict[int, tuple[str, int]] = {}

    for table_row in dom_tree.xpath("//tr"):
        current_row: list[str] = []
        column_index = 0

        while span_info.get(column_index, (None, 0))[1] > 0:
            current_row.append(span_info[column_index][0])
            span_info[column_index] = (
                span_info[column_index][0],
                span_info[column_index][1] - 1,
            )
            if span_info[column_index][1] == 0:
                del span_info[column_index]
            column_index += 1

        for table_cell in table_row.xpath("td|th"):
            while span_info.get(column_index, (None, 0))[1] > 0:
                current_row.append(span_info[column_index][0])
                span_info[column_index] = (
                    span_info[column_index][0],
                    span_info[column_index][1] - 1,
                )
                if span_info[column_index][1] == 0:
                    del span_info[column_index]
                column_index += 1

            row_span = int(table_cell.get("rowspan", "1"))
            col_span = int(table_cell.get("colspan", "1"))
            cell_text = "".join(table_cell.itertext()).strip()

            if row_span > 1:
                for i in range(col_span):
                    span_info[column_index + i] = (cell_text, row_span - 1)

            for _ in range(col_span):
                current_row.append(cell_text)
            column_index += col_span

        while span_info.get(column_index, (None, 0))[1] > 0:
            current_row.append(span_info[column_index][0])
            span_info[column_index] = (
                span_info[column_index][0],
                span_info[column_index][1] - 1,
            )
            if span_info[column_index][1] == 0:
                del span_info[column_index]
            column_index += 1

        table_rows.append(current_row)

    max_columns = max(map(len, table_rows)) if table_rows else 0
    for row in table_rows:
        row.extend([""] * (max_columns - len(row)))

    return np.array(table_rows)


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

    scores = {}
    for sample_id, gt_path, pred_path in tasks:
        try:
            with open(gt_path) as f:
                gt_html = f.read()
            with open(pred_path) as f:
                pred_html = f.read()
            if not pred_html.strip():
                continue

            gt_table = html_to_numpy(clean_table_in_html(gt_html))
            pred_table = html_to_numpy(clean_table_in_html(pred_html))

            scores[sample_id] = float(table_similarity(gt_table, pred_table))

        except Exception as e:
            print(e, "\n", clean_table_in_html(gt_html), "\n", clean_table_in_html(pred_html))

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

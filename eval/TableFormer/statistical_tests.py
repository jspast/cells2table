"""
Statistical tests for the BRACIS 2026 paper.

Tests performed
---------------
For each benchmark and metric (TEDS, timing):

1. Shapiro-Wilk normality test
   - Applied to each provider's sample independently.
   - Determines whether parametric (paired t-test) or non-parametric
     (Wilcoxon) tests are appropriate.
   - Note: at n >= 200 Shapiro-Wilk becomes very sensitive to minor
     deviations; use alongside visual inspection (Q-Q plots).

2. Wilcoxon signed-rank test (two-sided, paired)
   - Paired because the SAME tables are evaluated by both pipelines.
   - Non-parametric: makes no distributional assumption.
   - Null hypothesis: the distribution of differences is symmetric
     around zero (i.e. the two pipelines perform equally).

3. Holm-Bonferroni correction
   - Applied across the 4 per-benchmark Wilcoxon p-values for each
     metric (TEDS family, timing family).
   - Controls the family-wise error rate (FWER) while being more
     powerful than plain Bonferroni.

4. Cliff's delta (effect size)
   - Non-parametric effect size: δ = P(X > Y) − P(X < Y).
   - Interpretation: |δ| < 0.147 negligible, < 0.33 small,
     < 0.474 medium, ≥ 0.474 large  (Romano et al., 2006).

Note on Friedman test
---------------------
The Friedman test is a non-parametric alternative to repeated-measures
ANOVA for k ≥ 3 related groups. With k = 2 groups (Ours vs. TableFormer),
Friedman reduces to a sign test, which is *less* powerful than the Wilcoxon
signed-rank test because it ignores the magnitude of differences. Therefore,
Wilcoxon is the correct paired non-parametric test for our two-group
comparison. Friedman would be appropriate if a third pipeline were added.
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"

BENCHMARKS = ["DoclingDPBench", "OmniDocBench", "FinTabNet", "PubTabNet"]
FILE_NAMES = {"Ours": "cells2table", "TableFormer": "TableFormer"}
PROVIDERS = ["Ours", "TableFormer"]

CLIFF_THRESHOLDS = [(0.474, "large"), (0.33, "medium"), (0.147, "small")]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_evals(benchmark: str, provider: str) -> list[dict]:
    path = DATA_DIR / f"{benchmark}_{FILE_NAMES[provider]}.json"
    with open(path) as f:
        return json.load(f)["evaluations"]


def get_paired(benchmark: str, field: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (ours, tableformer) arrays aligned by (filename, table_id)."""
    maps = {
        p: {(e["filename"], e["table_id"]): e for e in load_evals(benchmark, p)} for p in PROVIDERS
    }
    keys = sorted(maps["Ours"].keys() & maps["TableFormer"].keys())
    return (
        np.array([maps["Ours"][k][field] for k in keys]),
        np.array([maps["TableFormer"][k][field] for k in keys]),
    )


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta: P(X > Y) - P(X < Y)."""
    # n1, n2 = len(x), len(y)
    matrix = np.sign(x[:, None] - y[None, :])  # shape (n1, n2)
    return float(matrix.mean())


def cliff_label(d: float) -> str:
    ad = abs(d)
    for threshold, label in CLIFF_THRESHOLDS:
        if ad >= threshold:
            return label
    return "negligible"


def shapiro_summary(arr: np.ndarray) -> tuple[float, float, bool]:
    stat, pval = stats.shapiro(arr)
    return stat, pval, bool(pval > 0.05)


def stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def hline(width=90):
    print("─" * width)


def section(title: str):
    hline()
    print(f"  {title}")
    hline()


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def run_tests(metric: str, field: str) -> dict:
    """Run the full test battery for one metric. Returns results dict."""
    section(f"Metric: {metric.upper()}")

    raw_pvals = []
    results = {}

    # ── Per-benchmark ───────────────────────────────────────────────────────
    for b in BENCHMARKS:
        ours, tf = get_paired(b, field)
        n = len(ours)

        # 1. Shapiro-Wilk
        sw_stat_o, sw_p_o, sw_norm_o = shapiro_summary(ours)
        sw_stat_t, sw_p_t, sw_norm_t = shapiro_summary(tf)

        # 2. Wilcoxon signed-rank
        try:
            w_stat, w_pval = stats.wilcoxon(ours, tf, alternative="two-sided")
        except ValueError:
            # e.g. all differences are zero
            w_stat, w_pval = 0.0, 1.0

        # 3. Cliff's delta
        cd = cliffs_delta(ours, tf)

        raw_pvals.append(w_pval)
        results[b] = dict(
            n=n,
            ours_mean=float(np.mean(ours)),
            ours_median=float(np.median(ours)),
            ours_std=float(np.std(ours)),
            tf_mean=float(np.mean(tf)),
            tf_median=float(np.median(tf)),
            tf_std=float(np.std(tf)),
            sw_stat_ours=sw_stat_o,
            sw_p_ours=sw_p_o,
            sw_normal_ours=sw_norm_o,
            sw_stat_tf=sw_stat_t,
            sw_p_tf=sw_p_t,
            sw_normal_tf=sw_norm_t,
            wilcoxon_stat=w_stat,
            wilcoxon_p_raw=w_pval,
            cliffs_delta=cd,
            cliffs_label=cliff_label(cd),
        )

    # 4. Holm correction across the 4 benchmarks
    reject, pvals_corrected, _, _ = multipletests(raw_pvals, method="holm")
    for b, p_corr, rej in zip(BENCHMARKS, pvals_corrected, reject):
        results[b]["wilcoxon_p_holm"] = float(p_corr)
        results[b]["wilcoxon_reject_holm"] = bool(rej)

    # ── Print table ─────────────────────────────────────────────────────────
    # Shapiro-Wilk
    print(f"\n{'':20s} {'Shapiro-Wilk (Ours)':>26s}   {'Shapiro-Wilk (TF)':>26s}")
    print(
        f"{'Benchmark':<20s} {'W':>8s} {'p':>10s} {'Normal?':>8s}   "
        f"{'W':>8s} {'p':>10s} {'Normal?':>8s}"
    )
    print("─" * 90)
    for b in BENCHMARKS:
        r = results[b]
        print(
            f"{b:<20s} {r['sw_stat_ours']:8.4f} {r['sw_p_ours']:10.4e} "
            f"{'yes' if r['sw_normal_ours'] else 'no':>8s}   "
            f"{r['sw_stat_tf']:8.4f} {r['sw_p_tf']:10.4e} "
            f"{'yes' if r['sw_normal_tf'] else 'no':>8s}"
        )

    # Wilcoxon + Holm + Cliff's delta
    print(
        f"\n{'Benchmark':<20s} {'n':>5s} {'W stat':>10s} {'p raw':>12s} "
        f"{'p Holm':>12s} {'Reject?':>8s} {'sig':>5s}   "
        f"{'δ (Cliff)':>10s} {'|δ| size':>10s}"
    )
    print("─" * 90)
    for b in BENCHMARKS:
        r = results[b]
        print(
            f"{b:<20s} {r['n']:5d} {r['wilcoxon_stat']:10.1f} "
            f"{r['wilcoxon_p_raw']:12.4e} {r['wilcoxon_p_holm']:12.4e} "
            f"{'yes' if r['wilcoxon_reject_holm'] else 'no':>8s} "
            f"{stars(r['wilcoxon_p_holm']):>5s}   "  # ty:ignore[invalid-argument-type]
            f"{r['cliffs_delta']:+10.4f} {r['cliffs_label']:>10s}"
        )

    # Descriptive stats
    print(
        f"\n{'Benchmark':<20s} {'Ours mean':>10s} {'Ours med':>10s} "
        f"{'TF mean':>10s} {'TF med':>10s}"
    )
    print("─" * 60)
    for b in BENCHMARKS:
        r = results[b]
        print(
            f"{b:<20s} {r['ours_mean']:10.4f} {r['ours_median']:10.4f} "
            f"{r['tf_mean']:10.4f} {r['tf_median']:10.4f}"
        )

    return results


def main() -> None:
    all_results = {}

    print("\n" + "═" * 90)
    print("  STATISTICAL TESTS — cells2table (Ours) vs. TableFormer")
    print("  Paired comparisons: same (filename, table_id) in both providers")
    print("═" * 90)

    all_results["TEDS"] = run_tests("TEDS", "TEDS")
    all_results["timing"] = run_tests("timing", "timing")

    # ── Save JSON ────────────────────────────────────────────────────────────
    def _jsonify(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _jsonify(v) for k, v in obj.items()}
        return obj

    out_path = DATA_DIR / "statistical_tests_results.json"
    with open(out_path, "w") as f:
        json.dump(_jsonify(all_results), f, indent=2)

    print(f"\n{'═' * 90}")
    print(f"  Results saved to {out_path}")
    print("═" * 90 + "\n")

    # ── LaTeX table fragment ─────────────────────────────────────────────────
    print("LaTeX table (TEDS):\n")
    print(r"\begin{table}[t]")
    print(r"  \centering")
    print(r"  \caption{Statistical comparison of TEDS scores (Ours vs.\ TableFormer).")
    print(r"           $W$ = Wilcoxon signed-rank statistic; $p_{\text{Holm}}$ = Holm-corrected")
    print(r"           $p$-value; $\delta$ = Cliff's delta (effect size).}")
    print(r"  \label{tab:stats_teds}")
    print(r"  \small")
    print(r"  \begin{tabular}{lrrrrrr}")
    print(r"    \hline")
    print(
        r"    \textbf{Benchmark} & $n$ & \textbf{Ours} $\tilde{x}$ & \textbf{TF} $\tilde{x}$"
        r" & $W$ & $p_{\text{Holm}}$ & $\delta$ \\"
    )
    print(r"    \hline")
    for b in BENCHMARKS:
        r = all_results["TEDS"][b]
        sig = stars(r["wilcoxon_p_holm"])
        p_str = (
            f"$<10^{{{int(np.floor(np.log10(r['wilcoxon_p_holm'])))}}}$"
            if r["wilcoxon_p_holm"] < 0.001
            else f"{r['wilcoxon_p_holm']:.3f}"
        )
        print(
            f"    {b} & {r['n']} & {r['ours_median']:.3f} & {r['tf_median']:.3f}"
            f" & {r['wilcoxon_stat']:.0f} & {p_str}\\,{sig} & ${r['cliffs_delta']:+.3f}$"
            f" ({r['cliffs_label'][0].upper()}) \\\\"
        )
    print(r"    \hline")
    print(r"  \end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()

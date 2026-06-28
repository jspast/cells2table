"""
Generate figures for TableFormer comparison.

Outputs (PNG + PDF) are written to `figures/`
"""
import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
FIGS_DIR = SCRIPT_DIR / "figures"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
BENCHMARKS = ["DoclingDPBench", "OmniDocBench", "FinTabNet", "PubTabNet"]
FILE_NAMES = {"Ours": "cells2table", "TableFormer": "TableFormer"}
COLORS = {"Ours": "#2196F3", "TableFormer": "#FF5722"}
PROVIDERS = ["Ours", "TableFormer"]

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def load_evals(benchmark: str, provider: str) -> list[dict]:
    fname = FILE_NAMES.get(provider, provider)
    path = DATA_DIR / f"{benchmark}_{fname}.json"
    with open(path) as f:
        return json.load(f)["evaluations"]


def load_stats(benchmark: str, provider: str) -> dict:
    fname = FILE_NAMES.get(provider, provider)
    path = DATA_DIR / f"{benchmark}_{fname}.json"
    with open(path) as f:
        d = json.load(f)
    return {k: v for k, v in d.items() if k != "evaluations"}


def get_paired(benchmark: str, field: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (ours, tableformer) arrays aligned by (filename, table_id)."""
    maps = {}
    for p in PROVIDERS:
        maps[p] = {(e["filename"], e["table_id"]): e for e in load_evals(benchmark, p)}
    keys = sorted(maps["Ours"].keys() & maps["TableFormer"].keys())
    return (
        np.array([maps["Ours"][k][field] for k in keys]),
        np.array([maps["TableFormer"][k][field] for k in keys]),
    )


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGS_DIR / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIGS_DIR / f"{name}.pdf", bbox_inches="tight")
    print(f"  saved {name}")
    plt.close(fig)


def boxplot_pair(ax, data_list, colors=None, providers=None):
    providers = providers or PROVIDERS
    colors = colors or [COLORS[p] for p in providers]
    bp = ax.boxplot(
        data_list,
        patch_artist=True,
        widths=0.45,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker=".", markersize=3, alpha=0.35, markeredgewidth=0),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    return bp


def sig_label(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ---------------------------------------------------------------------------
# Figure 1 — RQ1a: Mean TEDS bar chart
# ---------------------------------------------------------------------------
def fig_rq1_teds_bar() -> None:
    x = np.arange(len(BENCHMARKS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, p in enumerate(PROVIDERS):
        means = [load_stats(b, p)["TEDS_mean"] for b in BENCHMARKS]
        bars = ax.bar(x + (i - 0.5) * width, means, width, label=p, color=COLORS[p], alpha=0.85)
        for bar, m in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{m:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(BENCHMARKS, rotation=15, ha="right")
    ax.set_ylabel("Mean TEDS")
    ax.set_ylim(0, 1.14)
    ax.set_title("RQ1 — TEDS Accuracy per Benchmark")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.legend()
    plt.tight_layout()
    save(fig, "rq1_teds_bar")


# ---------------------------------------------------------------------------
# Figure 2 — RQ1b: TEDS distribution box plots
# ---------------------------------------------------------------------------
def fig_rq1_teds_boxplot() -> None:
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5), sharey=True)
    for ax, b in zip(axes, BENCHMARKS):
        data = [load_evals(b, p) for p in PROVIDERS]
        teds = [[e["TEDS"] for e in d] for d in data]
        boxplot_pair(ax, teds)
        ax.set_title(b)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(PROVIDERS, rotation=20, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    axes[0].set_ylabel("TEDS")
    fig.suptitle("RQ1 — TEDS Distribution per Benchmark", fontsize=12)
    plt.tight_layout()
    save(fig, "rq1_teds_boxplot")


# ---------------------------------------------------------------------------
# Figure 3 — RQ2a: Inference time box plots  (2×2 grid)
# ---------------------------------------------------------------------------
def fig_rq2_timing_bar() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, b in zip(axes.flat, BENCHMARKS):
        timings = [[e["timing"] for e in load_evals(b, p)] for p in PROVIDERS]
        boxplot_pair(ax, timings)

        speedup = np.median(timings[1]) / np.median(timings[0])
        ax.set_title(b, pad=6)
        ax.text(
            0.5,
            0.97,
            f"{speedup:.1f}× faster",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            color="#333",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#ccc", lw=0.8),
        )

        ax.set_xticks([1, 2])
        ax.set_xticklabels(PROVIDERS, rotation=15, ha="right")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_ylabel("Inference time (s)")

    fig.suptitle("RQ2 — Inference Time per Benchmark", fontsize=13)
    plt.tight_layout()
    save(fig, "rq2_timing_bar")


# ---------------------------------------------------------------------------
# Figure 4 — RQ2b: Timing vs table size scatter  (2×2 grid, all 4 benchmarks)
# ---------------------------------------------------------------------------
def fig_rq2_timing_scatter() -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for p in PROVIDERS:
        all_sizes, all_timings = [], []
        for b in BENCHMARKS:
            evals = load_evals(b, p)
            all_sizes += [e["true_nrows"] * e["true_ncols"] for e in evals]
            all_timings += [e["timing"] for e in evals]
        sizes = np.array(all_sizes)
        timings = np.array(all_timings)
        ax.scatter(sizes, timings, alpha=0.10, s=5, color=COLORS[p], label=p)
        z = np.polyfit(sizes, timings, 1)
        xfit = np.linspace(0, sizes.max(), 300)
        ax.plot(xfit, np.poly1d(z)(xfit), color=COLORS[p], linewidth=2.5)

    ax.set_xlabel("Table size (true rows × cols)")
    ax.set_ylabel("Inference time (s)")
    ax.set_title("RQ2 — Inference Time vs. Table Size")
    leg = ax.legend()
    for lh in leg.legend_handles:
        if lh is not None:
            lh.set_alpha(0.7)
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    save(fig, "rq2_timing_scatter")


# ---------------------------------------------------------------------------
# Figure 5 — RQ3: Quadrant box plots  (2×2 grid, improved readability)
# ---------------------------------------------------------------------------
def fig_rq3_quadrants() -> None:
    def quadrant_label(e, med_r, med_c):
        row = f"≤{med_r:.0f}" if e["true_nrows"] <= med_r else f">{med_r:.0f}"
        col = f"≤{med_c:.0f}" if e["true_ncols"] <= med_c else f">{med_c:.0f}"
        return f"{row} rows\n{col} cols"

    Q_SHORT = ["Small", "Wide", "Tall", "Large"]
    Q_DESC = [
        "few rows\nfew cols",
        "few rows\nmany cols",
        "many rows\nfew cols",
        "many rows\nmany cols",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for ax, b in zip(axes.flat, BENCHMARKS):
        ours_evals = load_evals(b, "Ours")
        tf_evals = load_evals(b, "TableFormer")

        rows_all = np.array([e["true_nrows"] for e in ours_evals])
        cols_all = np.array([e["true_ncols"] for e in ours_evals])
        med_r, med_c = np.median(rows_all), np.median(cols_all)

        q_labels = [
            f"≤{med_r:.0f} rows\n≤{med_c:.0f} cols",
            f"≤{med_r:.0f} rows\n>{med_c:.0f} cols",
            f">{med_r:.0f} rows\n≤{med_c:.0f} cols",
            f">{med_r:.0f} rows\n>{med_c:.0f} cols",
        ]

        x = np.arange(len(q_labels))
        width = 0.32

        for side, (p, evals) in enumerate([("Ours", ours_evals), ("TableFormer", tf_evals)]):
            offset = (side - 0.5) * width
            for qi, ql in enumerate(q_labels):
                teds = [e["TEDS"] for e in evals if quadrant_label(e, med_r, med_c) == ql]
                if not teds:
                    continue
                bp = ax.boxplot(
                    teds,
                    positions=[x[qi] + offset],
                    widths=width * 0.88,
                    patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.8),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker=".", markersize=3, alpha=0.35),
                )
                bp["boxes"][0].set_facecolor(COLORS[p])
                bp["boxes"][0].set_alpha(0.72)
                ax.text(
                    x[qi] + offset,
                    -0.14,
                    f"n={len(teds)}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#555",
                )

        for qi, ql in enumerate(q_labels):
            ours_q = {
                (e["filename"], e["table_id"]): e["TEDS"]
                for e in ours_evals
                if quadrant_label(e, med_r, med_c) == ql
            }
            tf_q = {
                (e["filename"], e["table_id"]): e["TEDS"]
                for e in tf_evals
                if quadrant_label(e, med_r, med_c) == ql
            }
            keys = sorted(ours_q.keys() & tf_q.keys())
            if len(keys) > 4:
                t1 = [ours_q[k] for k in keys]
                t2 = [tf_q[k] for k in keys]
                try:
                    _, pval = stats.wilcoxon(t1, t2, alternative="two-sided")
                except ValueError:
                    pval = 1.0
                ymax = max(np.percentile(t1, 95), np.percentile(t2, 95)) + 0.04
                ax.text(
                    x[qi], min(ymax, 1.06), sig_label(pval), ha="center", va="bottom", fontsize=9
                )

        ax.set_xlim(-0.6, len(q_labels) - 0.4)
        ax.set_ylim(-0.20, 1.20)
        ax.set_xticks(x)

        # Two-line tick labels: bold name + description
        tick_labels = [f"{s}\n({d})" for s, d in zip(Q_SHORT, Q_DESC)]
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_ylabel("TEDS")
        ax.set_title(b, fontsize=12, pad=6)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    patches = [mpatches.Patch(color=COLORS[p], alpha=0.75, label=p) for p in PROVIDERS]
    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
        fontsize=11,
    )
    fig.suptitle(
        "TEDS by Table Structure Quadrant  (thresholds = per-benchmark median rows × median cols)",
        fontsize=12,
    )
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "rq3_quadrants")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating figures...")
    fig_rq1_teds_bar()
    fig_rq1_teds_boxplot()
    fig_rq2_timing_bar()
    fig_rq2_timing_scatter()
    fig_rq3_quadrants()
    print("All figures generated.")

"""
Generate figures for TableFormer comparison.

Outputs (PNG + PDF) are written to `figures/`
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
FIGS_DIR = SCRIPT_DIR / "figures"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

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
def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGS_DIR / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIGS_DIR / f"{name}.pdf", bbox_inches="tight")
    print(f"  saved {name}")
    plt.close(fig)


def sig_label(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ---------------------------------------------------------------------------
# Figure 6 — PulseBench-Tab: Ours vs PP-TableMagic
# ---------------------------------------------------------------------------
def fig_pulsebench() -> None:
    pb_colors = {"Ours": "#2196F3", "PP-TableMagic": "#43A047"}
    pb_providers = ["Ours", "PP-TableMagic"]
    pb_files = {"Ours": "PulseBench-Tab_cells2table", "PP-TableMagic": "PulseBench-Tab_pp"}

    def load_pb(provider):
        with open(DATA_DIR / f"{pb_files[provider]}.json") as f:
            return json.load(f)["evaluations"]

    evals = {p: load_pb(p) for p in pb_providers}

    metrics = [
        ("TEDS", "TEDS ($\\uparrow$)"),
        ("rd", "rd ($\\uparrow$)"),
        ("TLAG", "TLAG ($\\uparrow$)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=False)

    for ax, (metric, label) in zip(axes, metrics):
        data = [[e[metric] for e in evals[p]] for p in pb_providers]
        bp = ax.boxplot(
            data,
            patch_artist=True,
            widths=0.45,
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(marker=".", markersize=3, alpha=0.35, markeredgewidth=0),
        )
        for patch, p in zip(bp["boxes"], pb_providers):
            patch.set_facecolor(pb_colors[p])
            patch.set_alpha(0.75)

        ax.set_title(label, pad=18)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(pb_providers, rotation=10, ha="right")
        ax.set_ylim(-0.02, 1.12)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        # significance marker above boxes
        _, pval = stats.wilcoxon(data[0], data[1])
        ax.text(1.5, 1.04, sig_label(pval), ha="center", va="bottom", fontsize=11, color="#333")

    fig.suptitle(
        "PulseBench-Tab: Ours vs. PP-TableMagic  (n=528)",
        fontsize=12,
    )
    plt.tight_layout()
    save(fig, "pulsebench")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating figures...")
    fig_pulsebench()
    print("All figures generated.")

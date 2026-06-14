"""Shared figure style for paper experiments."""
import matplotlib.pyplot as plt
import matplotlib as mpl

COLORS = {
    "spam": "#1f77b4",
    "hsol": "#ff7f0e",
    "rte": "#2ca02c",
    "mrpc": "#d62728",
    "naive": "#888888",
    "combine": "#e377c2",
    "neural_exec": "#9467bd",
    "random_ctrl": "#bcbd22",
    "shared": "#17becf",
    "task_specific": "#8c564b",
}

TASK_LABELS = {
    "spam": "Spam",
    "hsol": "Hate speech",
    "rte": "NLI (RTE)",
    "mrpc": "Paraphrase (MRPC)",
}


def apply():
    """Call once at script start."""
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 1.8,
        "lines.markersize": 4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.constrained_layout.use": True,
    })


def savefig(fig, name, folder="paper_exp/figures"):
    """Save as both PDF (vector) and PNG."""
    for ext in ("pdf", "png"):
        fig.savefig(f"{folder}/{name}.{ext}")
    print(f"Saved {folder}/{name}.{{pdf,png}}")
    plt.close(fig)

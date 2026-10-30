"""Create publication figures for the three Elevator-DAG-v1 scenarios."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "elevator_experiments" / "results" / "all_algorithms_metrics_final100.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "elevator_experiments" / "figures"

SCENARIOS = {
    "fault_diagnosis": "Fault diagnosis",
    "dispatch": "Group dispatch",
    "predictive_maintenance": "Predictive maintenance",
}
METRICS = (
    ("avg_delay", "Average delay (s)"),
    ("avg_energy", "Average energy (J)"),
    ("normalized_cost", "Normalized cost"),
)
MAIN_ALGORITHMS = (
    "H-MAPPO", "TPTDTO", "MAPPO", "IPPO", "DDQN", "Greedy"
)
ALL_ALGORITHMS = MAIN_ALGORITHMS

COLORS = {
    "H-MAPPO": "#C44E52",
    "Greedy": "#3A8D5D",
    "MAPPO": "#377EB8",
    "DDQN": "#E69F00",
    "IPPO": "#8C6BB1",
    "TPTDTO": "#008B8B",
    "HEFT": "#4D4D4D",
    "Local-only": "#9A9A9A",
}
MARKERS = {
    "H-MAPPO": "o",
    "Greedy": "v",
    "MAPPO": "s",
    "DDQN": "^",
    "IPPO": "D",
    "TPTDTO": "h",
    "HEFT": "P",
    "Local-only": "X",
}
LINESTYLES = {
    "H-MAPPO": "-",
    "Greedy": "--",
    "MAPPO": "-.",
    "DDQN": ":",
    "IPPO": (0, (4, 1, 1, 1)),
    "TPTDTO": (0, (5, 1.5)),
    "HEFT": "-",
    "Local-only": "--",
}


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Microsoft YaHei", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8,
        "axes.linewidth": 0.75,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.5,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def read_rows(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({
                "scenario": row["scenario"],
                "nodes": int(row["nodes"]),
                "algorithm": row["algorithm"],
                "avg_delay": float(row["avg_delay"]),
                "avg_energy": float(row["avg_energy"]),
                "normalized_cost": float(row["normalized_cost"]),
            })
    return rows


def series(rows, scenario: str, algorithm: str, metric: str):
    selected = sorted(
        (row for row in rows if row["scenario"] == scenario and row["algorithm"] == algorithm),
        key=lambda row: row["nodes"],
    )
    if len(selected) != 4:
        raise ValueError(f"Expected four task sizes for {scenario}/{algorithm}, found {len(selected)}")
    return (
        np.asarray([row["nodes"] for row in selected], dtype=float),
        np.asarray([row[metric] for row in selected], dtype=float),
    )


def plot_scenario(rows, scenario: str, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.2), constrained_layout=False)
    panel_labels = ("a", "b", "c")

    legend_handles = []
    for panel_index, (axis, (metric, ylabel)) in enumerate(zip(axes, METRICS)):
        for algorithm in MAIN_ALGORITHMS:
            x, y = series(rows, scenario, algorithm, metric)
            line, = axis.plot(
                x, y,
                color=COLORS[algorithm],
                marker=MARKERS[algorithm],
                linestyle=LINESTYLES[algorithm],
                linewidth=1.25 if algorithm == "H-MAPPO" else 1.05,
                markersize=3.8,
                markerfacecolor="white" if algorithm != "H-MAPPO" else COLORS[algorithm],
                markeredgewidth=0.75,
                label=algorithm,
                zorder=5 if algorithm == "H-MAPPO" else 3,
            )
            if panel_index == 0:
                legend_handles.append(line)

        axis.set_xticks([10, 15, 20, 25])
        axis.set_xlabel("Number of subtasks")
        axis.set_ylabel(ylabel)
        axis.text(
            0.5,
            -0.30,
            f"({panel_labels[panel_index]})",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontweight="bold",
            fontsize=8,
        )
        axis.grid(axis="y", color="#D9D9D9", linestyle="--", linewidth=0.55, alpha=0.75)
        axis.tick_params(direction="out", length=2.8, width=0.7)
        axis.margins(x=0.06, y=0.12)

    fig.legend(
        legend_handles,
        ALL_ALGORITHMS,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=len(ALL_ALGORITHMS),
        handlelength=2.1,
        columnspacing=0.9,
        handletextpad=0.35,
    )
    fig.subplots_adjust(left=0.072, right=0.99, bottom=0.23, top=0.82, wspace=0.36)

    stem = output_dir / f"elevator_{scenario}_delay_energy_cost"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {stem}.png/.pdf/.svg")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    configure_style()
    rows = read_rows(args.input.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for scenario in SCENARIOS:
        plot_scenario(rows, scenario, args.output_dir.resolve())


if __name__ == "__main__":
    main()

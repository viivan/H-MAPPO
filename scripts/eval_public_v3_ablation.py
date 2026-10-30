"""Evaluate the four H-MAPPO component-ablation variants"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_public_heft_comparison import evaluate_algorithm, task_range_label

matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["pdf.fonttype"] = 42


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = PROJECT_ROOT / "image"
TASK_RANGES: List[Tuple[int, int]] = [(10, 10), (15, 15), (20, 20), (25, 25), (30, 30)]

ALGORITHMS = [
    {
        "name": "H-MAPPO",
        "type": "ppo",
        "run_prefix": "event_ppo_heuristic_paper_v3",
        "config": "configs/event_ppo_paper_public_v3.yaml",
        "trained_task_range": (10, 30),
        "heuristic_mask": True,
    },
    {
        "name": "H-MAPPO w/o Heur",
        "type": "ppo",
        "run_prefix": "event_ppo_ablation_no_heur_paper_v3",
        "config": "configs/event_ppo_paper_public_v3.yaml",
        "trained_task_range": (10, 30),
        "variant": "ablation_no_heur",
        "heuristic_mask": False,
    },
    {
        "name": "H-MAPPO w/o HACA",
        "type": "ppo",
        "run_prefix": "event_ppo_ablation_no_haca_paper_v3",
        "config": "configs/event_ppo_paper_public_v3.yaml",
        "trained_task_range": (10, 30),
        "variant": "ablation_no_haca",
        "heuristic_mask": False,
    },
    {
        "name": "H-MAPPO w/o Urgency",
        "type": "ppo",
        "run_prefix": "event_ppo_ablation_no_urgency_paper_v3",
        "config": "configs/event_ppo_paper_public_v3.yaml",
        "trained_task_range": (10, 30),
        "variant": "ablation_no_urgency",
        "heuristic_mask": False,
    },
]

COLORS = {
    "H-MAPPO": "#D1495B",
    "H-MAPPO w/o Heur": "#2E86AB",
    "H-MAPPO w/o HACA": "#8D6A9F",
    "H-MAPPO w/o Urgency": "#F2A541",
}
MARKERS = {
    "H-MAPPO": "o",
    "H-MAPPO w/o Heur": "s",
    "H-MAPPO w/o HACA": "D",
    "H-MAPPO w/o Urgency": "^",
}

PLOT_ALGORITHM_NAMES = (
    "H-MAPPO",
    "H-MAPPO w/o HACA",
    "H-MAPPO w/o Urgency",
)


def save_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    fields = [
        "algorithm", "task_range", "avg_delay", "avg_energy",
        "normalized_delay", "normalized_energy", "delay_energy_cost",
        "on_time_ratio", "local_ratio", "edge_ratio", "cloud_ratio",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Ablation metrics not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def values_for(rows: List[Dict[str, Any]], algorithm: str, labels: List[str], metric: str) -> List[float]:
    return [
        float(next(row[metric] for row in rows if row["algorithm"] == algorithm and row["task_range"] == label))
        for label in labels
    ]


def plot_panel(rows: List[Dict[str, Any]], output_stem: Path) -> None:
    labels = [task_range_label(a, b) for a, b in TASK_RANGES]
    x = np.arange(len(labels))
    panels = [
        ("avg_delay", "Average delay (s)", "(a)"),
        ("avg_energy", "Average energy (J)", "(b)"),
        ("delay_energy_cost", "Normalized weighted cost", "(c)"),
        ("on_time_ratio", "On-time completion ratio", "(d)"),
    ]
    algorithms_by_name = {algorithm["name"]: algorithm for algorithm in ALGORITHMS}
    plot_algorithms = [algorithms_by_name[name] for name in PLOT_ALGORITHM_NAMES]

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.8))
    fig.patch.set_facecolor("white")
    for axis, (metric, ylabel, panel_label) in zip(axes.ravel(), panels):
        axis.set_facecolor("#FBFCFE")
        for algorithm in plot_algorithms:
            name = algorithm["name"]
            axis.plot(
                x,
                values_for(rows, name, labels, metric),
                color=COLORS[name],
                marker=MARKERS[name],
                markersize=6.2,
                markeredgecolor="white",
                markeredgewidth=0.7,
                linewidth=2.4 if name == "H-MAPPO" else 1.8,
                label=name,
            )
        axis.set_ylabel(ylabel, fontsize=9.5)
        axis.set_xlabel("Number of subtasks", fontsize=9.5)
        axis.text(
            0.5,
            -0.28,
            panel_label,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=11.5,
            fontweight="bold",
        )
        axis.set_xticks(x)
        axis.set_xticklabels(labels)
        if metric == "on_time_ratio":
            values = [
                value
                for algorithm in PLOT_ALGORITHM_NAMES
                for value in values_for(rows, algorithm, labels, metric)
            ]
            axis.set_ylim(max(0.0, min(values) - 0.001), 1.0002)
        axis.grid(True, linestyle="--", linewidth=0.8, alpha=0.28)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        ncol=len(PLOT_ALGORITHM_NAMES),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        frameon=False,
        fontsize=9,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.99,
        bottom=0.20,
        top=0.98,
        wspace=0.18,
        hspace=0.52,
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Reuse the existing ablation CSV and redraw the three-algorithm figure.",
    )
    args = parser.parse_args()

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = IMAGE_DIR / "public_v3_ablation_metrics.csv"
    if args.plot_only:
        rows = load_csv(csv_path)
        print(f"[plot only] {csv_path}")
    else:
        device = torch.device(args.device)
        rows = []
        for task_min, task_max in TASK_RANGES:
            label = task_range_label(task_min, task_max)
            print(f"\n=== Ablation: public DAG tasks {label} ===")
            for algorithm in ALGORITHMS:
                result = evaluate_algorithm(
                    algo=dict(algorithm),
                    task_min=task_min,
                    task_max=task_max,
                    episodes=args.episodes,
                    seed=args.seed,
                    device=device,
                    cost_alpha=0.60,
                    cost_beta=0.40,
                    normalized_cost=True,
                )
                if result is None:
                    raise FileNotFoundError(f"Missing trained weights for {algorithm['name']}")
                rows.append({"algorithm": algorithm["name"], "task_range": label, **result})
                print(
                    f"{algorithm['name']:<24s} delay={result['avg_delay']:.4f} "
                    f"energy={result['avg_energy']:.4f} cost={result['delay_energy_cost']:.4f} "
                    f"on-time={result['on_time_ratio']:.4f}"
                )
        save_csv(rows, csv_path)
        print(f"[saved] {csv_path}")

    figure_stem = IMAGE_DIR / "public_v3_ablation_panel_three_algorithms"
    plot_panel(rows, figure_stem)
    print(f"[saved] {figure_stem}.png/.pdf/.svg")


if __name__ == "__main__":
    main()

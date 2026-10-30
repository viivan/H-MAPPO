"""Plot convergence for H-MAPPO, MAPPO, IPPO and DDQN."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["pdf.fonttype"] = 42


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = PROJECT_ROOT / "image"

RUNS = {
    "H-MAPPO": PROJECT_ROOT / "outputs/event_ppo_heuristic_paper_v3_tasks10-30/metrics_event_ppo_heuristic_paper_v3.csv",
    "MAPPO": PROJECT_ROOT / "outputs/event_ppo_baseline_paper_v3_tasks10-30/metrics_event_ppo_baseline_paper_v3.csv",
    "IPPO": PROJECT_ROOT / "outputs/event_ippo_paper_v3_tasks10-30/metrics_event_ippo_paper_v3.csv",
    "DDQN": PROJECT_ROOT / "outputs/event_ddqn_paper_v3_tasks10-30/metrics_event_ddqn_paper_v3.csv",
}

COLORS = {
    "H-MAPPO": "#D1495B",
    "MAPPO": "#2E86AB",
    "IPPO": "#8D6A9F",
    "DDQN": "#F2A541",
}


def read_metrics(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Training metrics not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    required = ("episode", "ep_mean_reward", "ep_on_time_ratio", "c1_excess")
    missing = [key for key in required if not rows or key not in rows[0]]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        for key in required
    }


def ema(values: np.ndarray, span: int) -> np.ndarray:
    if values.size == 0:
        return values.copy()
    alpha = 2.0 / (max(1, span) + 1.0)
    result = np.empty_like(values, dtype=np.float64)
    result[0] = values[0]
    for index in range(1, len(values)):
        result[index] = alpha * values[index] + (1.0 - alpha) * result[index - 1]
    return result


def style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor("#FBFCFE")
    axis.grid(True, linestyle="--", linewidth=0.8, alpha=0.28)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#B8C0C8")
    axis.spines["bottom"].set_color("#B8C0C8")
    axis.tick_params(labelsize=9, colors="#27323A")


def save_tail_statistics(
    data: Dict[str, Dict[str, np.ndarray]], tail_episodes: int, output_path: Path
) -> None:
    fields = [
        "algorithm",
        "episodes",
        "tail_episodes",
        "reward_mean",
        "reward_std",
        "on_time_mean",
        "on_time_std",
        "c1_excess_mean",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for name, metrics in data.items():
            tail = min(tail_episodes, len(metrics["episode"]))
            reward = metrics["ep_mean_reward"][-tail:]
            on_time = metrics["ep_on_time_ratio"][-tail:]
            c1_excess = metrics["c1_excess"][-tail:]
            row = {
                "algorithm": name,
                "episodes": len(metrics["episode"]),
                "tail_episodes": tail,
                "reward_mean": float(np.mean(reward)),
                "reward_std": float(np.std(reward)),
                "on_time_mean": float(np.mean(on_time)),
                "on_time_std": float(np.std(on_time)),
                "c1_excess_mean": float(np.mean(c1_excess)),
            }
            writer.writerow(row)
            print(
                f"{name:<8s} last-{tail}: reward={row['reward_mean']:.4f}+/-{row['reward_std']:.4f} "
                f"on-time={row['on_time_mean']:.4f}+/-{row['on_time_std']:.4f} "
                f"c1={row['c1_excess_mean']:.4f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smooth", type=int, default=30, help="EMA span in episodes")
    parser.add_argument("--tail", type=int, default=100, help="Episodes used for final statistics")
    args = parser.parse_args()

    data = {name: read_metrics(path) for name, path in RUNS.items()}
    lengths = {name: len(metrics["episode"]) for name, metrics in data.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Training logs have different lengths: {lengths}")

    fig, axis = plt.subplots(figsize=(9.4, 5.7))
    fig.patch.set_facecolor("white")
    for name, metrics in data.items():
        episodes = metrics["episode"]
        raw = metrics["ep_mean_reward"]
        axis.plot(
            episodes,
            raw,
            color=COLORS[name],
            linewidth=0.35,
            alpha=0.07,
            zorder=1,
        )
        axis.plot(
            episodes,
            ema(raw, args.smooth),
            color=COLORS[name],
            linewidth=1.75 if name == "H-MAPPO" else 1.35,
            label=name,
            alpha=1.0 if name == "H-MAPPO" else 0.90,
            zorder=5 if name == "H-MAPPO" else 3,
        )
    axis.set_xlabel("Training iteration", fontsize=10.5)
    axis.set_ylabel("Mean step reward", fontsize=10.5)
    style_axis(axis)
    axis.legend(
        ncol=4,
        loc="lower right",
        frameon=False,
        fontsize=9.5,
        columnspacing=1.5,
        handlelength=2.0,
    )
    fig.tight_layout()

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    figure_stem = IMAGE_DIR / "public_v3_reward_convergence_four_algorithms"
    stats_path = IMAGE_DIR / "public_v3_convergence_tail_statistics.csv"
    png_path = figure_stem.with_suffix(".png")
    svg_path = figure_stem.with_suffix(".svg")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    save_tail_statistics(data, args.tail, stats_path)
    print(f"[saved] {png_path}")
    print(f"[saved] {svg_path}")
    print(f"[saved] {stats_path}")


if __name__ == "__main__":
    main()

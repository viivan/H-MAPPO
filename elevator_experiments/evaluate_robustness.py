"""Evaluate elevator offloading robustness under edge load and link variation.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_public_heft_comparison import (  # noqa: E402
    eval_one_episode,
    load_ddqn_agent,
    load_ppo_actor,
)
from scripts.plot_task_range_delay_energy import build_env  # noqa: E402


SCENARIOS = {
    "fault_diagnosis": "elevator_v1_fault_diagnosis",
    "dispatch": "elevator_v1_dispatch",
    "predictive_maintenance": "elevator_v1_predictive_maintenance",
}

ALGORITHMS = (
    {
        "name": "H-MAPPO",
        "type": "ppo",
        "config": "elevator_experiments/configs/elevator_v1.yaml",
        "run_dir": "outputs/event_ppo_heuristic_elevator_v1_tasks10-25",
    },
    {
        "name": "MAPPO",
        "type": "ppo",
        "config": "elevator_experiments/configs/mappo_elevator_v1.yaml",
        "run_dir": "outputs/event_mappo_elevator_v1_tasks10-25",
    },
    {
        "name": "DDQN",
        "type": "ddqn",
        "config": "elevator_experiments/configs/ddqn_elevator_v1.yaml",
        "run_dir": "outputs/event_ddqn_elevator_v1_tasks10-25",
    },
    {
        "name": "Greedy",
        "type": "greedy_static_cost",
        "config": "elevator_experiments/configs/elevator_v1.yaml",
    },
)

COLORS = {
    "H-MAPPO": "#C23B53",
    "MAPPO": "#277DA1",
    "DDQN": "#F4A261",
    "Greedy": "#4F8A70",
}


def load_config(relative_path: str, dataset_mode: str) -> dict[str, Any]:
    with (PROJECT_ROOT / relative_path).open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["dataset"] = dataset_mode
    return cfg


def apply_stress(
    env: Any,
    edge_available_ratio: float,
    uplink_bandwidth_mhz: float,
) -> None:

    if not 0.0 < edge_available_ratio <= 1.0:
        raise ValueError("edge_available_ratio must be in (0, 1]")
    if uplink_bandwidth_mhz <= 0.0:
        raise ValueError("uplink_bandwidth_mhz must be positive")

    env.cfg.channel.uplink_bandwidth_hz = uplink_bandwidth_mhz * 1.0e6
    env.cfg.resource.edge_cpu_hz = [
        float(cpu) * edge_available_ratio
        for cpu in env.cfg.resource.edge_cpu_hz
    ]
    env.cfg.resource.baseline_edge_alloc_hz *= edge_available_ratio


def build_stressed_env(
    cfg: dict[str, Any],
    task_size: int,
    seed: int,
    edge_available_ratio: float,
    uplink_bandwidth_mhz: float,
):
    env = build_env(cfg, task_size, task_size, seed)
    apply_stress(env, edge_available_ratio, uplink_bandwidth_mhz)
    return env


def load_policies(
    algorithms: list[dict[str, Any]],
    task_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    policies: dict[str, Any] = {}
    probe_dataset = next(iter(SCENARIOS.values()))
    for algorithm in algorithms:
        if algorithm["type"] not in {"ppo", "ddqn"}:
            policies[algorithm["name"]] = None
            continue
        cfg = load_config(algorithm["config"], probe_dataset)
        probe_env = build_stressed_env(cfg, task_size, seed, 1.0, 10.0)
        run_dir = PROJECT_ROOT / algorithm["run_dir"]
        if algorithm["type"] == "ppo":
            policies[algorithm["name"]] = load_ppo_actor(
                run_dir=run_dir,
                cfg=cfg,
                env=probe_env,
                device=device,
                prefer_best=True,
            )
        else:
            policies[algorithm["name"]] = load_ddqn_agent(
                run_dir=run_dir,
                cfg=cfg,
                env=probe_env,
                device=device,
            )
    return policies


def evaluate(
    algorithms: list[dict[str, Any]],
    policies: dict[str, Any],
    scenarios: dict[str, str],
    task_size: int,
    episodes: int,
    seed: int,
    device: torch.device,
    edge_available_ratios: list[float],
    bandwidths_mhz: list[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for edge_ratio in edge_available_ratios:
        background_load = 1.0 - edge_ratio
        for bandwidth in bandwidths_mhz:
            for scenario, dataset_mode in scenarios.items():
                for algorithm in algorithms:
                    cfg = load_config(algorithm["config"], dataset_mode)
                    metrics = []
                    env = build_stressed_env(
                        cfg=cfg,
                        task_size=task_size,
                        seed=seed,
                        edge_available_ratio=edge_ratio,
                        uplink_bandwidth_mhz=bandwidth,
                    )
                    for episode in range(episodes):
                        env.set_episode_index(episode + 1)
                        metrics.append(
                            eval_one_episode(
                                env=env,
                                policy_type=algorithm["type"],
                                policy=policies[algorithm["name"]],
                                device=device,
                                cost_alpha=0.60,
                                cost_beta=0.40,
                                normalized_cost=True,
                                stochastic_ppo=False,
                            )
                        )

                    row = {
                        "scenario": scenario,
                        "nodes": task_size,
                        "background_load_ratio": background_load,
                        "edge_available_ratio": edge_ratio,
                        "uplink_bandwidth_mhz": bandwidth,
                        "algorithm": algorithm["name"],
                        "episodes": episodes,
                        "avg_delay": float(np.mean([m["avg_delay"] for m in metrics])),
                        "avg_energy": float(np.mean([m["avg_energy"] for m in metrics])),
                        "normalized_cost": float(
                            np.mean([m["delay_energy_cost"] for m in metrics])
                        ),
                        "on_time_ratio": float(
                            np.mean([m["on_time_ratio"] for m in metrics])
                        ),
                        "local_ratio": float(np.mean([m["local_ratio"] for m in metrics])),
                        "edge_ratio": float(np.mean([m["edge_ratio"] for m in metrics])),
                        "cloud_ratio": float(np.mean([m["cloud_ratio"] for m in metrics])),
                    }
                    rows.append(row)
                    print(
                        f"load={background_load:.0%} bw={bandwidth:g}MHz "
                        f"{scenario:24s} {algorithm['name']:14s} "
                        f"cost={row['normalized_cost']:.4f} "
                        f"delay={row['avg_delay']:.4f} "
                        f"energy={row['avg_energy']:.4f}"
                    )
    return rows


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, float, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            float(row["background_load_ratio"]),
            float(row["uplink_bandwidth_mhz"]),
            str(row["algorithm"]),
        )
        grouped.setdefault(key, []).append(row)

    summary = []
    for (load, bandwidth, algorithm), values in grouped.items():
        summary.append(
            {
                "background_load_ratio": load,
                "edge_available_ratio": 1.0 - load,
                "uplink_bandwidth_mhz": bandwidth,
                "algorithm": algorithm,
                "scenario_count": len(values),
                "avg_delay": float(np.mean([v["avg_delay"] for v in values])),
                "avg_energy": float(np.mean([v["avg_energy"] for v in values])),
                "normalized_cost": float(
                    np.mean([v["normalized_cost"] for v in values])
                ),
                "on_time_ratio": float(np.mean([v["on_time_ratio"] for v in values])),
            }
        )
    return summary


def save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] {path}")


def load_scenario_csv(path: Path) -> list[dict[str, Any]]:
    numeric_fields = {
        "nodes", "background_load_ratio", "edge_available_ratio",
        "uplink_bandwidth_mhz", "episodes", "avg_delay", "avg_energy",
        "normalized_cost", "on_time_ratio", "local_ratio", "edge_ratio",
        "cloud_ratio",
    }
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            {
                key: (float(value) if key in numeric_fields else value)
                for key, value in row.items()
            }
            for row in csv.DictReader(handle)
        ]


def matrix_for(
    summary: list[dict[str, Any]],
    algorithm: str,
    loads: list[float],
    bandwidths: list[float],
) -> np.ndarray:
    lookup = {
        (
            str(row["algorithm"]),
            round(float(row["background_load_ratio"]), 8),
            round(float(row["uplink_bandwidth_mhz"]), 8),
        ): float(row["normalized_cost"])
        for row in summary
    }
    return np.asarray(
        [
            [lookup[(algorithm, round(load, 8), round(bw, 8))] for load in loads]
            for bw in bandwidths
        ],
        dtype=np.float64,
    )


def annotate_heatmap(ax: Any, matrix: np.ndarray, vmin: float, vmax: float) -> None:
    midpoint = (vmin + vmax) / 2.0
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            color = "white" if matrix[y, x] > midpoint else "#202020"
            ax.text(x, y, f"{matrix[y, x]:.3f}", ha="center", va="center", color=color, fontsize=8)


def plot_cost_heatmaps(
    summary: list[dict[str, Any]],
    algorithm_names: list[str],
    loads: list[float],
    bandwidths: list[float],
    output_stem: Path,
) -> None:
    matrices = {
        algorithm: matrix_for(summary, algorithm, loads, bandwidths)
        for algorithm in algorithm_names
    }
    all_values = np.concatenate([matrix.ravel() for matrix in matrices.values()])
    vmin = float(np.min(all_values))
    vmax = float(np.max(all_values))

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.8), constrained_layout=True)
    image = None
    for ax, algorithm in zip(axes.flat, algorithm_names):
        matrix = matrices[algorithm]
        image = ax.imshow(matrix, cmap="YlOrRd", vmin=vmin, vmax=vmax, aspect="auto", origin="lower")
        annotate_heatmap(ax, matrix, vmin, vmax)
        ax.set_title(algorithm, fontsize=12, color=COLORS[algorithm], fontweight="bold")
        ax.set_xticks(range(len(loads)), [f"{load:.0%}" for load in loads])
        ax.set_yticks(range(len(bandwidths)), [f"{bw:g}" for bw in bandwidths])
        ax.set_xlabel("Background edge load")
        ax.set_ylabel("Uplink bandwidth (MHz)")
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.86, label="Normalized delay-energy cost")
    for suffix in ("png", "pdf", "svg"):
        path = output_stem.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        print(f"[saved] {path}")
    plt.close(fig)


def plot_relative_advantage(
    summary: list[dict[str, Any]],
    competitors: list[str],
    loads: list[float],
    bandwidths: list[float],
    output_stem: Path,
) -> None:
    hmappo = matrix_for(summary, "H-MAPPO", loads, bandwidths)
    reductions = {}
    for competitor in competitors:
        baseline = matrix_for(summary, competitor, loads, bandwidths)
        reductions[competitor] = 100.0 * (baseline - hmappo) / np.maximum(baseline, 1e-12)
    limit = max(1.0, float(max(np.max(np.abs(x)) for x in reductions.values())))

    fig, axes = plt.subplots(1, len(competitors), figsize=(12.0, 3.8), constrained_layout=True)
    if len(competitors) == 1:
        axes = [axes]
    image = None
    for ax, competitor in zip(axes, competitors):
        matrix = reductions[competitor]
        image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto", origin="lower")
        for y in range(matrix.shape[0]):
            for x in range(matrix.shape[1]):
                ax.text(x, y, f"{matrix[y, x]:+.1f}%", ha="center", va="center", fontsize=8)
        ax.set_title(f"vs {competitor}", fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(loads)), [f"{load:.0%}" for load in loads])
        ax.set_yticks(range(len(bandwidths)), [f"{bw:g}" for bw in bandwidths])
        ax.set_xlabel("Background edge load")
        ax.set_ylabel("Uplink bandwidth (MHz)")
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.86, label="H-MAPPO cost reduction (%)")
    for suffix in ("png", "pdf", "svg"):
        path = output_stem.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        print(f"[saved] {path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task-size", type=int, default=20)
    parser.add_argument("--edge-available", nargs="+", type=float, default=[1.0, 0.8, 0.6, 0.4])
    parser.add_argument("--bandwidths", nargs="+", type=float, default=[10.0, 6.0, 4.0, 2.0])
    parser.add_argument("--algorithms", nargs="+", default=None)
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--output-prefix", default="robustness_edge_link")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Reuse the existing summary CSV and redraw both robustness figures.",
    )
    parser.add_argument(
        "--replace-greedy-only",
        action="store_true",
        help=(
            "Reuse the output-prefix CSV, remove prior greedy rows, evaluate "
            "only the static-cost Greedy, and overwrite the same results."
        ),
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny validation without replacing formal results.",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    algorithms = [
        algorithm for algorithm in ALGORITHMS
        if args.algorithms is None or algorithm["name"] in args.algorithms
    ]
    scenarios = {
        name: mode for name, mode in SCENARIOS.items()
        if args.scenarios is None or name in args.scenarios
    }
    if not algorithms:
        raise ValueError("No matching algorithms selected")
    if not scenarios:
        raise ValueError("No matching scenarios selected")
    if args.replace_greedy_only:
        if args.algorithms is not None or args.smoke_test or args.plot_only:
            raise ValueError(
                "--replace-greedy-only cannot be combined with --algorithms "
                "--smoke-test or --plot-only"
            )
        algorithms = [a for a in ALGORITHMS if a["name"] == "Greedy"]

    edge_available = list(args.edge_available)
    bandwidths = list(args.bandwidths)
    episodes = args.episodes
    output_prefix = args.output_prefix
    if args.smoke_test:
        edge_available = [1.0, 0.6]
        bandwidths = [10.0, 6.0]
        episodes = 1
        algorithms = [
            a for a in algorithms
            if a["name"] in {"H-MAPPO", "MAPPO", "DDQN", "Greedy"}
        ]
        scenarios = {"fault_diagnosis": SCENARIOS["fault_diagnosis"]}
        output_prefix = f"{output_prefix}_smoke"

    print(
        f"Robustness evaluation: task_size={args.task_size}, episodes={episodes}, "
        f"algorithms={[a['name'] for a in algorithms]}, scenarios={list(scenarios)}"
    )
    results_dir = PROJECT_ROOT / "elevator_experiments" / "results"
    figures_dir = PROJECT_ROOT / "elevator_experiments" / "figures"
    scenario_metrics_path = results_dir / f"{output_prefix}_scenario_metrics.csv"
    summary_path = results_dir / f"{output_prefix}_summary.csv"
    if args.plot_only:
        if not summary_path.exists():
            raise FileNotFoundError(f"Existing robustness summary not found: {summary_path}")
        summary = load_scenario_csv(summary_path)
        loads = sorted({float(row["background_load_ratio"]) for row in summary})
        plot_bandwidths = sorted(
            {float(row["uplink_bandwidth_mhz"]) for row in summary}
        )
        available_names = {str(row["algorithm"]) for row in summary}
        algorithm_names = [
            name for name in ("H-MAPPO", "MAPPO", "DDQN", "Greedy")
            if name in available_names
        ]
        if len(algorithm_names) == 4:
            plot_cost_heatmaps(
                summary,
                algorithm_names,
                loads,
                plot_bandwidths,
                figures_dir / f"{output_prefix}_cost_heatmaps",
            )
        if "H-MAPPO" in algorithm_names and len(algorithm_names) > 1:
            plot_relative_advantage(
                summary,
                [name for name in algorithm_names if name != "H-MAPPO"],
                loads,
                plot_bandwidths,
                figures_dir / f"{output_prefix}_relative_advantage",
            )
        print(f"[plot only] {summary_path}")
        return

    prior_rows: list[dict[str, Any]] = []
    if args.replace_greedy_only:
        if not scenario_metrics_path.exists():
            raise FileNotFoundError(
                f"Existing robustness CSV not found: {scenario_metrics_path}"
            )
        prior_rows = [
            row for row in load_scenario_csv(scenario_metrics_path)
            if row["algorithm"] not in {
                "Greedy-MinCost", "Greedy-StaticCost", "Greedy"
            }
        ]
        prior_episode_counts = {int(float(row["episodes"])) for row in prior_rows}
        if prior_episode_counts and prior_episode_counts != {episodes}:
            raise ValueError(
                "Existing rows use episodes="
                f"{sorted(prior_episode_counts)}, but the command requests {episodes}"
            )
        print(f"[reuse existing metrics] {scenario_metrics_path}")

    policies = load_policies(algorithms, args.task_size, args.seed, device)
    new_rows = evaluate(
        algorithms=algorithms,
        policies=policies,
        scenarios=scenarios,
        task_size=args.task_size,
        episodes=episodes,
        seed=args.seed,
        device=device,
        edge_available_ratios=edge_available,
        bandwidths_mhz=bandwidths,
    )
    rows = prior_rows + new_rows
    summary = aggregate_rows(rows)

    figures_dir.mkdir(parents=True, exist_ok=True)
    save_csv(rows, scenario_metrics_path)
    save_csv(summary, summary_path)

    loads = sorted({float(row["background_load_ratio"]) for row in summary})
    plot_bandwidths = sorted({float(row["uplink_bandwidth_mhz"]) for row in summary})
    available_names = {str(row["algorithm"]) for row in summary}
    algorithm_names = [
        name for name in ("H-MAPPO", "MAPPO", "DDQN", "Greedy")
        if name in available_names
    ]
    if len(algorithm_names) == 4:
        plot_cost_heatmaps(
            summary,
            algorithm_names,
            loads,
            plot_bandwidths,
            figures_dir / f"{output_prefix}_cost_heatmaps",
        )
    if "H-MAPPO" in algorithm_names and len(algorithm_names) > 1:
        competitors = [name for name in algorithm_names if name != "H-MAPPO"]
        plot_relative_advantage(
            summary,
            competitors,
            loads,
            plot_bandwidths,
            figures_dir / f"{output_prefix}_relative_advantage",
        )


if __name__ == "__main__":
    main()

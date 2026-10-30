"""Evaluate all main algorithms on the Elevator-DAG-v1 test split."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_public_heft_comparison import (
    build_env,
    eval_one_episode,
    load_ddqn_agent,
    load_ppo_actor,
)


SCENARIOS = {
    "fault_diagnosis": "elevator_v1_fault_diagnosis",
    "dispatch": "elevator_v1_dispatch",
    "predictive_maintenance": "elevator_v1_predictive_maintenance",
}
SIZES = (10, 15, 20, 25)
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
        "name": "IPPO",
        "type": "ppo",
        "config": "elevator_experiments/configs/ippo_elevator_v1.yaml",
        "run_dir": "outputs/event_ippo_elevator_v1_tasks10-25",
    },
    {
        "name": "DDQN",
        "type": "ddqn",
        "config": "elevator_experiments/configs/ddqn_elevator_v1.yaml",
        "run_dir": "outputs/event_ddqn_elevator_v1_tasks10-25",
    },
    {
        "name": "TPTDTO",
        "type": "ppo",
        "config": "elevator_experiments/configs/tptdto_elevator_v1.yaml",
        "run_dir": "outputs/event_tptdto_elevator_v1_tasks10-25",
    },
    {"name": "Local-only", "type": "local_only"},
    {"name": "Greedy", "type": "greedy_static_cost"},
    {"name": "HEFT", "type": "heft"},
)


def load_config(relative_path: str, dataset_mode: str):
    with (PROJECT_ROOT / relative_path).open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["dataset"] = dataset_mode
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefer-final", action="store_true")
    parser.add_argument(
        "--stochastic-ppo",
        action="store_true",
        help="Sample PPO actions from the learned policy distribution.",
    )
    parser.add_argument("--output-name", default="all_algorithms_metrics.csv")
    parser.add_argument(
        "--replace-greedy-only",
        action="store_true",
        help=(
            "Reuse the output CSV, remove prior greedy rows, evaluate only "
            "the static-cost Greedy, and overwrite the same CSV."
        ),
    )
    parser.add_argument(
        "--tptdto-only",
        action="store_true",
        help=(
            "Reuse the output CSV, replace prior TPTDTO rows, evaluate only "
            "TPTDTO, and overwrite the same CSV."
        ),
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=None,
        help="Optional algorithm names to evaluate, for example H-MAPPO Greedy.",
    )
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output = PROJECT_ROOT / "elevator_experiments" / "results" / args.output_name
    rows = []
    if args.tptdto_only and (
        args.replace_greedy_only or args.algorithms is not None
    ):
        raise ValueError(
            "--tptdto-only cannot be combined with "
            "--replace-greedy-only or --algorithms"
        )
    if args.replace_greedy_only:
        if args.algorithms is not None:
            raise ValueError("--replace-greedy-only cannot be combined with --algorithms")
        if not output.exists():
            raise FileNotFoundError(f"Existing metrics CSV not found: {output}")
        with output.open(encoding="utf-8", newline="") as handle:
            rows = [
                dict(row) for row in csv.DictReader(handle)
                if row["algorithm"] not in {
                    "Greedy-MinCost", "Greedy-StaticCost", "Greedy"
                }
            ]
        print(f"[reuse existing metrics] {output}")
    elif args.tptdto_only:
        if not output.exists():
            raise FileNotFoundError(f"Existing metrics CSV not found: {output}")
        with output.open(encoding="utf-8", newline="") as handle:
            rows = [
                dict(row) for row in csv.DictReader(handle)
                if row["algorithm"] != "TPTDTO"
            ]
        print(f"[reuse existing metrics] {output}")

    for scenario, dataset_mode in SCENARIOS.items():
        for size in SIZES:
            selected_algorithms = [
                algorithm for algorithm in ALGORITHMS
                if (
                    (args.replace_greedy_only and algorithm["name"] == "Greedy")
                    or (args.tptdto_only and algorithm["name"] == "TPTDTO")
                    or (
                        not args.replace_greedy_only
                        and not args.tptdto_only
                        and (
                            args.algorithms is None
                            or algorithm["name"] in args.algorithms
                        )
                    )
                )
            ]
            if not selected_algorithms:
                raise ValueError(f"No matching algorithms: {args.algorithms}")
            for algorithm in selected_algorithms:
                config_path = algorithm.get(
                    "config", "elevator_experiments/configs/elevator_v1.yaml"
                )
                cfg = load_config(config_path, dataset_mode)
                policy = None
                metrics = []
                for episode in range(args.episodes):
                    env = build_env(cfg, size, size, args.seed + episode)
                    env.set_episode_index(episode + 1)
                    if policy is None and algorithm["type"] == "ppo":
                        policy = load_ppo_actor(
                            run_dir=PROJECT_ROOT / algorithm["run_dir"],
                            cfg=cfg,
                            env=env,
                            device=device,
                            prefer_best=not args.prefer_final,
                        )
                    elif policy is None and algorithm["type"] == "ddqn":
                        policy = load_ddqn_agent(
                            run_dir=PROJECT_ROOT / algorithm["run_dir"],
                            cfg=cfg,
                            env=env,
                            device=device,
                        )
                    metrics.append(
                        eval_one_episode(
                            env=env,
                            policy_type=algorithm["type"],
                            policy=policy,
                            device=device,
                            cost_alpha=0.60,
                            cost_beta=0.40,
                            normalized_cost=True,
                            stochastic_ppo=args.stochastic_ppo,
                        )
                    )
                row = {
                    "scenario": scenario,
                    "nodes": size,
                    "algorithm": algorithm["name"],
                    "avg_delay": float(np.mean([m["avg_delay"] for m in metrics])),
                    "avg_energy": float(np.mean([m["avg_energy"] for m in metrics])),
                    "normalized_cost": float(np.mean([m["delay_energy_cost"] for m in metrics])),
                    "on_time_ratio": float(np.mean([m["on_time_ratio"] for m in metrics])),
                    "local_ratio": float(np.mean([m["local_ratio"] for m in metrics])),
                    "edge_ratio": float(np.mean([m["edge_ratio"] for m in metrics])),
                    "cloud_ratio": float(np.mean([m["cloud_ratio"] for m in metrics])),
                }
                rows.append(row)
                print(
                    f"{scenario:24s} n={size:2d} {algorithm['name']:14s} "
                    f"delay={row['avg_delay']:.4f} energy={row['avg_energy']:.4f} "
                    f"cost={row['normalized_cost']:.4f} on-time={row['on_time_ratio']:.4f} "
                    f"place=({row['local_ratio']:.2f}/{row['edge_ratio']:.2f}/{row['cloud_ratio']:.2f})"
                )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] {output}")


if __name__ == "__main__":
    main()

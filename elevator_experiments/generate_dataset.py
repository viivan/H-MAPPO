"""Generate the reproducible Elevator-DAG-v1 synthetic benchmark."""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


SCENARIOS = ("fault_diagnosis", "dispatch", "predictive_maintenance")
SIZES = (10, 15, 20, 25)
MASTER_SEED = 42
EDGE_CPU_HZ = 6.0e9
EFFECTIVE_UPLINK_BPS = 20.0e6


@dataclass
class Node:
    task_type: str
    input_bytes: int
    output_bytes: int
    cycles: int


def log_uniform(rng: random.Random, low: float, high: float) -> float:
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def make_node(
    rng: random.Random,
    task_type: str,
    input_range: Tuple[int, int],
    output_ratio: Tuple[float, float],
    cycles_range: Tuple[float, float],
) -> Node:
    input_bytes = int(log_uniform(rng, *input_range))
    output_bytes = max(128, int(input_bytes * rng.uniform(*output_ratio)))
    cycles = int(log_uniform(rng, *cycles_range))
    return Node(task_type, input_bytes, output_bytes, cycles)


def connect_layers(layers: Sequence[Sequence[int]], rng: random.Random) -> List[Tuple[int, int]]:
    edges: set[Tuple[int, int]] = set()
    for left, right in zip(layers, layers[1:]):
        for dst_pos, dst in enumerate(right):
            src = left[dst_pos % len(left)]
            edges.add((src, dst))
            if len(left) > 1 and rng.random() < 0.55:
                extra = rng.choice([item for item in left if item != src])
                edges.add((extra, dst))
        for src in left:
            if not any(edge_src == src and edge_dst in right for edge_src, edge_dst in edges):
                edges.add((src, rng.choice(list(right))))
    return sorted(edges)


def build_fault_diagnosis(n: int, rng: random.Random) -> Tuple[List[Node], List[Tuple[int, int]]]:
    counts = {
        10: (3, 3, 2, 1, 1),
        15: (4, 4, 3, 1, 2, 1),
        20: (5, 5, 4, 2, 2, 2),
        25: (6, 6, 5, 2, 3, 3),
    }[n]
    if len(counts) == 5:
        roles = ("sensor", "preprocess", "feature", "fusion", "inference")
    else:
        roles = ("sensor", "preprocess", "feature", "fusion", "inference", "report")
    profiles = {
        "sensor": ((16 * 1024, 256 * 1024), (0.85, 1.00), (5e6, 5e7)),
        "preprocess": ((16 * 1024, 256 * 1024), (0.35, 0.75), (5e7, 4e8)),
        "feature": ((8 * 1024, 160 * 1024), (0.08, 0.30), (1e8, 9e8)),
        "fusion": ((16 * 1024, 256 * 1024), (0.10, 0.35), (2e8, 1.2e9)),
        "inference": ((4 * 1024, 96 * 1024), (0.05, 0.25), (4e8, 2.0e9)),
        "report": ((1 * 1024, 32 * 1024), (0.10, 0.50), (1e7, 1.5e8)),
    }
    return build_layered(counts, roles, profiles, rng)


def build_dispatch(n: int, rng: random.Random) -> Tuple[List[Node], List[Tuple[int, int]]]:
    counts = {10: (4, 1, 2, 1, 2), 15: (6, 1, 4, 1, 3), 20: (8, 1, 6, 1, 4), 25: (10, 1, 8, 1, 5)}[n]
    roles = ("request", "aggregate", "candidate", "select", "command")
    profiles = {
        "request": ((512, 8 * 1024), (0.80, 1.00), (1e6, 2e7)),
        "aggregate": ((4 * 1024, 96 * 1024), (0.20, 0.60), (2e7, 2e8)),
        "candidate": ((2 * 1024, 48 * 1024), (0.10, 0.40), (8e7, 8e8)),
        "select": ((2 * 1024, 64 * 1024), (0.05, 0.25), (5e7, 4e8)),
        "command": ((256, 8 * 1024), (0.20, 0.80), (1e6, 3e7)),
    }
    return build_layered(counts, roles, profiles, rng)


def build_predictive_maintenance(n: int, rng: random.Random) -> Tuple[List[Node], List[Tuple[int, int]]]:
    counts = {10: (3, 2, 2, 1, 1, 1), 15: (4, 3, 3, 2, 2, 1), 20: (5, 4, 4, 3, 2, 2), 25: (6, 5, 5, 4, 3, 2)}[n]
    roles = ("history", "preprocess", "feature", "fusion", "prediction", "decision")
    profiles = {
        "history": ((64 * 1024, 2 * 1024 * 1024), (0.70, 1.00), (2e7, 2e8)),
        "preprocess": ((32 * 1024, 1024 * 1024), (0.20, 0.65), (8e7, 7e8)),
        "feature": ((16 * 1024, 512 * 1024), (0.05, 0.25), (2e8, 1.2e9)),
        "fusion": ((16 * 1024, 384 * 1024), (0.08, 0.30), (3e8, 1.5e9)),
        "prediction": ((4 * 1024, 128 * 1024), (0.03, 0.20), (5e8, 2.5e9)),
        "decision": ((1 * 1024, 32 * 1024), (0.10, 0.50), (2e7, 2e8)),
    }
    return build_layered(counts, roles, profiles, rng)


def build_layered(counts, roles, profiles, rng):
    nodes: List[Node] = []
    layers: List[List[int]] = []
    next_id = 1
    for count, role in zip(counts, roles):
        layer = list(range(next_id, next_id + count))
        layers.append(layer)
        next_id += count
        for _ in layer:
            nodes.append(make_node(rng, role, *profiles[role]))
    return nodes, connect_layers(layers, rng)


BUILDERS = {
    "fault_diagnosis": build_fault_diagnosis,
    "dispatch": build_dispatch,
    "predictive_maintenance": build_predictive_maintenance,
}


def critical_path_deadline(nodes: Sequence[Node], edges: Sequence[Tuple[int, int]], scenario: str, rng: random.Random) -> float:
    predecessors: Dict[int, List[int]] = {i: [] for i in range(1, len(nodes) + 1)}
    for src, dst in edges:
        predecessors[dst].append(src)
    finish: Dict[int, float] = {}
    for node_id, node in enumerate(nodes, start=1):
        transfer = node.input_bytes * 8.0 / EFFECTIVE_UPLINK_BPS
        execution = node.cycles / EDGE_CPU_HZ
        start = max((finish[p] for p in predecessors[node_id]), default=0.0)
        finish[node_id] = start + transfer + execution

    load_factor = {
        "fault_diagnosis": 0.75,
        "dispatch": 1.15,
        "predictive_maintenance": 0.55,
    }[scenario]
    serial_compute = sum(node.cycles for node in nodes) / EDGE_CPU_HZ
    reference_time = max(max(finish.values()), load_factor * serial_compute)
    slack = {
        "fault_diagnosis": (1.35, 1.70),
        "dispatch": (1.15, 1.45),
        "predictive_maintenance": (1.50, 2.00),
    }[scenario]
    return max(0.25, reference_time * rng.uniform(*slack))


def write_gv(path: Path, sample_id: str, scenario: str, nodes: Sequence[Node], edges, deadline: float) -> None:
    lines = [
        f'digraph "{sample_id}" {{',
        f'  graph [scenario="{scenario}", deadline_s="{deadline:.6f}"];',
    ]
    for node_id, node in enumerate(nodes, start=1):
        lines.append(
            f'  {node_id} [size="{node.input_bytes}", expect_size="{node.output_bytes}", '
            f'cycles="{node.cycles}", task_type="{node.task_type}"];'
        )
    for src, dst in edges:
        lines.append(f'  {src} -> {dst} [size="{nodes[src - 1].output_bytes}"];')
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate(args) -> None:
    root = Path(args.output).resolve()
    if root.exists() and any(root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output is not empty: {root}. Use --overwrite to replace matching files.")
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, count, split_offset in (("train", args.train_count, 0), ("test", args.test_count, 10_000_000)):
        for scenario_index, scenario in enumerate(SCENARIOS):
            for size_index, size in enumerate(SIZES):
                folder = root / split / scenario / f"n{size}"
                folder.mkdir(parents=True, exist_ok=True)
                for index in range(count):
                    seed = args.seed + split_offset + scenario_index * 100_000 + size_index * 10_000 + index
                    rng = random.Random(seed)
                    nodes, edges = BUILDERS[scenario](size, rng)
                    deadline = critical_path_deadline(nodes, edges, scenario, rng)
                    sample_id = f"{scenario}_n{size}_{split}_{index:04d}"
                    path = folder / f"{sample_id}.gv"
                    write_gv(path, sample_id, scenario, nodes, edges, deadline)
                    rows.append({
                        "sample_id": sample_id,
                        "scenario": scenario,
                        "split": split,
                        "seed": seed,
                        "nodes": len(nodes),
                        "edges": len(edges),
                        "total_input_bytes": sum(node.input_bytes for node in nodes),
                        "total_output_bytes": sum(node.output_bytes for node in nodes),
                        "total_cycles": sum(node.cycles for node in nodes),
                        "deadline_s": f"{deadline:.6f}",
                        "file_path": path.relative_to(root).as_posix(),
                    })
                    print(f"[generated] {sample_id}")
    manifest = root / "elevator_dag_v1_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] {manifest} ({len(rows)} DAGs)")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="elevator_experiments/data/elevator_dag_v1")
    parser.add_argument("--train-count", type=int, default=200)
    parser.add_argument("--test-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=MASTER_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())

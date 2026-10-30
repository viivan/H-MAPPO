"""Validate Elevator-DAG-v1 manifest, split isolation, and Graphviz files."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path


NODE_RE = re.compile(r"^\s*(\d+)\s+\[")
EDGE_RE = re.compile(r"^\s*(\d+)\s*->\s*(\d+)\s+\[")


def has_cycle(nodes, edges) -> bool:
    successors = {node: [] for node in nodes}
    indegree = {node: 0 for node in nodes}
    for src, dst in edges:
        successors[src].append(dst)
        indegree[dst] += 1
    queue = [node for node in nodes if indegree[node] == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for dst in successors[node]:
            indegree[dst] -= 1
            if indegree[dst] == 0:
                queue.append(dst)
    return visited != len(nodes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="elevator_experiments/data/elevator_dag_v1")
    args = parser.parse_args()
    root = Path(args.data_root).resolve()
    manifest = root / "elevator_dag_v1_manifest.csv"
    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row["sample_id"] for row in rows]
    seeds = [(row["split"], row["seed"]) for row in rows]
    assert len(ids) == len(set(ids)), "Duplicate sample IDs"
    assert len(seeds) == len(set(seeds)), "Duplicate seeds within a split"
    train_seeds = {row["seed"] for row in rows if row["split"] == "train"}
    test_seeds = {row["seed"] for row in rows if row["split"] == "test"}
    assert train_seeds.isdisjoint(test_seeds), "Train/test seed overlap"

    counts = Counter()
    for row in rows:
        path = root / row["file_path"]
        assert path.is_file(), f"Missing file: {path}"
        nodes, edges = set(), []
        for line in path.read_text(encoding="utf-8").splitlines():
            edge_match = EDGE_RE.match(line)
            if edge_match:
                edges.append(tuple(map(int, edge_match.groups())))
                continue
            node_match = NODE_RE.match(line)
            if node_match:
                nodes.add(int(node_match.group(1)))
        assert len(nodes) == int(row["nodes"]), f"Node count mismatch: {path}"
        assert all(src in nodes and dst in nodes and src < dst for src, dst in edges), f"Invalid edge: {path}"
        assert not has_cycle(nodes, edges), f"Cycle detected: {path}"
        assert float(row["deadline_s"]) > 0
        counts[(row["split"], row["scenario"], int(row["nodes"]))] += 1

    print(f"[ok] validated {len(rows)} DAGs")
    for key in sorted(counts):
        print(f"  split={key[0]:5s} scenario={key[1]:24s} nodes={key[2]:2d} files={counts[key]}")


if __name__ == "__main__":
    main()

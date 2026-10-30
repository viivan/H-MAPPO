"""
Public DAG comparison with learning, DAG and online scheduling baselines.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.envs.event_env import EventDrivenOffloadEnv
from rl.offpolicy import DDQNAgent
from scripts.plot_task_range_delay_energy import (
    build_env,
    build_scorer,
    load_ppo_actor,
    load_yaml,
    make_event_masker,
    run_dir_for,
)

matplotlib.rcParams["font.family"] = ["Microsoft YaHei", "DejaVu Sans", "Arial"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["pdf.fonttype"] = 42

IMAGE_DIR = PROJECT_ROOT / "image"
TASK_RANGES: List[Tuple[int, int]] = [(15, 20), (20, 25), (25, 30), (30, 35)]

ALGORITHMS = [
    {
        "name": "H-MAPPO",
        "type": "ppo",
        "run_prefix": "event_ppo_heuristic",
        "config": "configs/event_ppo.yaml",
        "heuristic_mask": True,
    },
    {
        "name": "MAPPO",
        "type": "ppo",
        "run_prefix": "event_ppo_baseline",
        "config": "configs/event_ppo_baseline.yaml",
        "heuristic_mask": False,
    },
    {
        "name": "DDQN",
        "type": "ddqn",
        "run_prefix": "event_ddqn_paper_v3",
        "config": "configs/event_ppo_baseline_paper_public_v3.yaml",
        "heuristic_mask": False,
    },
    {
        "name": "IPPO",
        "type": "ppo",
        "run_prefix": "event_ippo_paper_v3",
        "config": "configs/event_ppo_baseline_paper_public_v3.yaml",
        "variant": "ippo",
        "heuristic_mask": False,
    },
    {
        "name": "TPTDTO",
        "type": "ppo",
        "run_prefix": "event_tptdto_paper_v3",
        "config": "configs/tptdto_paper_public_v3.yaml",
        "heuristic_mask": False,
        "paper_v3_only": True,
    },
    {
        "name": "Local-only",
        "type": "local_only",
        "run_prefix": "",
        "config": "configs/event_ppo.yaml",
        "heuristic_mask": False,
    },
    {
        "name": "Greedy-MinCost",
        "type": "greedy",
        "run_prefix": "",
        "config": "configs/event_ppo.yaml",
        "heuristic_mask": False,
    },
    {
        "name": "Greedy",
        "type": "greedy_static_cost",
        "run_prefix": "",
        "config": "configs/event_ppo.yaml",
        "heuristic_mask": False,
    },
    {
        "name": "HEFT",
        "type": "heft",
        "run_prefix": "",
        "config": "configs/event_ppo.yaml",
        "heuristic_mask": False,
    },
]

PLOT_ORDER = [
    "H-MAPPO",
    "TPTDTO",
    "MAPPO",
    "IPPO",
    "DDQN",
    "Greedy",
    "HEFT",
    "Local-only",
]

COLORS = {
    "H-MAPPO": "#D1495B",
    "MAPPO": "#2E86AB",
    "DDQN": "#F2A541",
    "IPPO": "#8D6A9F",
    "TPTDTO": "#008B8B",
    "Greedy-MinCost": "#5B8E7D",
    "Greedy": "#5B8E7D",
    "HEFT": "#4F6D7A",
    "Local-only": "#7A7A7A",
}

HATCHES = {
    "H-MAPPO": "",
    "MAPPO": "",
    "DDQN": "//",
    "IPPO": "..",
    "TPTDTO": "oo",
    "Greedy-MinCost": "\\\\",
    "Greedy": "\\\\",
    "HEFT": "xx",
    "Local-only": "++",
}

MAIN_LINE_ALGORITHMS = [
    "H-MAPPO", "TPTDTO", "MAPPO", "IPPO", "DDQN", "Greedy"
]
INSET_ALGORITHMS = ["Local-only", "HEFT"]
MARKERS = {
    "H-MAPPO": "o",
    "MAPPO": "s",
    "DDQN": "^",
    "IPPO": "D",
    "TPTDTO": "h",
    "Greedy-MinCost": "v",
    "Greedy": "v",
    "HEFT": "P",
    "Local-only": "X",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_task_range(text: str) -> Tuple[int, int]:
    left, right = text.split("-")
    return int(left), int(right)


def task_range_label(task_min: int, task_max: int) -> str:
    return str(task_min) if task_min == task_max else f"{task_min}-{task_max}"


StaticNode = Tuple[int, int]
StaticInterval = Tuple[float, float, StaticNode]


@dataclass
class StaticSchedulePlan:
    actions: Dict[StaticNode, int]
    starts: Dict[StaticNode, float]
    finishes: Dict[StaticNode, float]
    resources: Dict[StaticNode, int]
    timelines: Dict[int, List[StaticInterval]]


def static_candidate_resources(env: EventDrivenOffloadEnv, agent: int) -> List[int]:
    edge_base = env.N
    return [agent, *range(edge_base, edge_base + env.M), edge_base + env.M]


def static_action_for_resource(env: EventDrivenOffloadEnv, agent: int, resource: int) -> int:
    if resource == agent:
        return 0
    edge_base = env.N
    if edge_base <= resource < edge_base + env.M:
        return resource - edge_base + 1
    if resource == edge_base + env.M:
        return env.M + 1
    raise ValueError(f"Resource {resource} is unavailable to agent {agent}")


def static_input_delay(env: EventDrivenOffloadEnv, node: StaticNode, action: int) -> float:
    i, j = node
    return float(env.env._tran_delay(i, j, action))


def static_execution_time(env: EventDrivenOffloadEnv, node: StaticNode, action: int) -> float:
    i, j = node
    cycles = float(env.env.dag_tasks[i].subtasks[j].c)
    if action == 0:
        cpu = float(env.env.f_local[i])
    elif 1 <= action <= env.M:
        cpu = float(env.env.rm.edge_servers[action - 1].F_max)
    else:
        cpu = float(env.env.rm.cloud.F_cloud)
    return cycles / max(cpu, 1e-12)


def static_communication_time(
    env: EventDrivenOffloadEnv,
    src: StaticNode,
    dst: StaticNode,
    src_action: int,
    dst_action: int,
) -> float:
    i, src_j = src
    dst_i, dst_j = dst
    if i != dst_i:
        raise ValueError("Communication edges cannot cross DAGs")
    dag = env.env.dag_tasks[i]
    return float(
        env.env.ch.intermediate_tran_delay(
            delta_bits=dag.get_edge_data_bits(src_j, dst_j),
            x_src=src_action,
            x_dst=dst_action,
            distance=env.env.distances[i],
            home_m=env.env.home_servers[i],
            M=env.M,
            wire_bw_matrix=env.env.rm.bw_matrix,
        )
    )


def static_earliest_slot(
    timeline: List[StaticInterval],
    ready: float,
    duration: float,
) -> float:
    start = max(0.0, ready)
    for occupied_start, occupied_end, _ in sorted(timeline, key=lambda x: x[0]):
        if start + duration <= occupied_start + 1e-12:
            return start
        if start < occupied_end:
            start = occupied_end
    return start


def build_static_ranks(
    env: EventDrivenOffloadEnv,
) -> Tuple[Dict[StaticNode, float], Dict[StaticNode, float]]:
    nodes = [
        (i, j)
        for i, dag in enumerate(env.env.dag_tasks)
        for j in dag.subtasks
    ]
    avg_node_cost: Dict[StaticNode, float] = {}
    for node in nodes:
        i, _ = node
        costs = []
        for resource in static_candidate_resources(env, i):
            action = static_action_for_resource(env, i, resource)
            costs.append(
                static_input_delay(env, node, action)
                + static_execution_time(env, node, action)
            )
        avg_node_cost[node] = float(np.mean(costs))

    @lru_cache(maxsize=None)
    def avg_edge(src: StaticNode, dst: StaticNode) -> float:
        i, _ = src
        values = []
        candidates = static_candidate_resources(env, i)
        for src_resource in candidates:
            src_action = static_action_for_resource(env, i, src_resource)
            for dst_resource in candidates:
                dst_action = static_action_for_resource(env, i, dst_resource)
                values.append(
                    static_communication_time(env, src, dst, src_action, dst_action)
                )
        return float(np.mean(values))

    @lru_cache(maxsize=None)
    def rank_up(node: StaticNode) -> float:
        i, j = node
        successors = env.env.dag_tasks[i].subtasks[j].successors
        if not successors:
            return avg_node_cost[node]
        return avg_node_cost[node] + max(
            avg_edge(node, (i, succ)) + rank_up((i, succ))
            for succ in successors
        )

    @lru_cache(maxsize=None)
    def rank_down(node: StaticNode) -> float:
        i, j = node
        predecessors = env.env.dag_tasks[i].subtasks[j].predecessors
        if not predecessors:
            return 0.0
        return max(
            rank_down((i, pred))
            + avg_node_cost[(i, pred)]
            + avg_edge((i, pred), node)
            for pred in predecessors
        )

    return (
        {node: rank_up(node) for node in nodes},
        {node: rank_down(node) for node in nodes},
    )


def schedule_static_node(
    env: EventDrivenOffloadEnv,
    node: StaticNode,
    plan: StaticSchedulePlan,
    forced_resource: int | None = None,
) -> None:
    i, j = node
    predecessors = env.env.dag_tasks[i].subtasks[j].predecessors
    candidates = (
        [forced_resource]
        if forced_resource is not None
        else static_candidate_resources(env, i)
    )
    best = None
    for resource in candidates:
        action = static_action_for_resource(env, i, resource)
        dependency_ready = 0.0
        for pred_j in predecessors:
            pred = (i, pred_j)
            dependency_ready = max(
                dependency_ready,
                plan.finishes[pred]
                + static_communication_time(
                    env, pred, node, plan.actions[pred], action
                ),
            )
        compute_ready = dependency_ready + static_input_delay(env, node, action)
        duration = static_execution_time(env, node, action)
        start = static_earliest_slot(plan.timelines[resource], compute_ready, duration)
        finish = start + duration
        candidate = (finish, start, resource, action)
        if best is None or candidate < best:
            best = candidate

    if best is None:
        raise RuntimeError(f"No static resource available for task {node}")
    finish, start, resource, action = best
    plan.actions[node] = int(action)
    plan.starts[node] = float(start)
    plan.finishes[node] = float(finish)
    plan.resources[node] = int(resource)
    plan.timelines[resource].append((float(start), float(finish), node))
    plan.timelines[resource].sort(key=lambda x: x[0])


def empty_static_plan(env: EventDrivenOffloadEnv) -> StaticSchedulePlan:
    return StaticSchedulePlan(
        actions={},
        starts={},
        finishes={},
        resources={},
        timelines={r: [] for r in range(env.N + env.M + 1)},
    )


def build_standard_heft_plan(env: EventDrivenOffloadEnv) -> StaticSchedulePlan:
    """Standard HEFT"""
    rank_up, _ = build_static_ranks(env)
    remaining = set(rank_up)
    plan = empty_static_plan(env)
    while remaining:
        ready = [
            node
            for node in remaining
            if all(
                (node[0], pred) in plan.finishes
                for pred in env.env.dag_tasks[node[0]].subtasks[node[1]].predecessors
            )
        ]
        if not ready:
            raise RuntimeError("HEFT found no ready task; DAG may contain a cycle")
        node = min(ready, key=lambda n: (-rank_up[n], n[0], n[1]))
        schedule_static_node(env, node, plan)
        remaining.remove(node)
    validate_static_plan(env, plan, "HEFT")
    return plan


def build_standard_cpop_plan(env: EventDrivenOffloadEnv) -> StaticSchedulePlan:
    rank_up, rank_down = build_static_ranks(env)
    priorities = {node: rank_up[node] + rank_down[node] for node in rank_up}
    critical_resources: Dict[int, int] = {}
    critical_nodes = set()

    for i, dag in enumerate(env.env.dag_tasks):
        entries = dag.get_entry_nodes()
        current = max(entries, key=lambda j: rank_up[(i, j)])
        path = [(i, current)]
        while dag.subtasks[current].successors:
            current = max(
                dag.subtasks[current].successors,
                key=lambda succ: priorities[(i, succ)],
            )
            path.append((i, current))
        critical_nodes.update(path)
        critical_resources[i] = min(
            static_candidate_resources(env, i),
            key=lambda resource: sum(
                static_input_delay(
                    env, node, static_action_for_resource(env, i, resource)
                )
                + static_execution_time(
                    env, node, static_action_for_resource(env, i, resource)
                )
                for node in path
            ),
        )

    remaining = set(priorities)
    plan = empty_static_plan(env)
    while remaining:
        ready = [
            node
            for node in remaining
            if all(
                (node[0], pred) in plan.finishes
                for pred in env.env.dag_tasks[node[0]].subtasks[node[1]].predecessors
            )
        ]
        if not ready:
            raise RuntimeError("CPOP found no ready task; DAG may contain a cycle")
        node = min(ready, key=lambda n: (-priorities[n], n[0], n[1]))
        forced = critical_resources[node[0]] if node in critical_nodes else None
        schedule_static_node(env, node, plan, forced_resource=forced)
        remaining.remove(node)
    validate_static_plan(env, plan, "CPOP")
    return plan


def validate_static_plan(
    env: EventDrivenOffloadEnv,
    plan: StaticSchedulePlan,
    name: str,
) -> None:
    expected = sum(dag.num_subtasks for dag in env.env.dag_tasks)
    if len(plan.actions) != expected:
        raise AssertionError(f"{name} planned {len(plan.actions)} of {expected} tasks")
    for i, dag in enumerate(env.env.dag_tasks):
        for src_j, dst_j in dag.edges:
            src = (i, src_j)
            dst = (i, dst_j)
            required = (
                plan.finishes[src]
                + static_communication_time(
                    env, src, dst, plan.actions[src], plan.actions[dst]
                )
                + static_input_delay(env, dst, plan.actions[dst])
            )
            if plan.starts[dst] + 1e-9 < required:
                raise AssertionError(f"{name} precedence violation: {src} -> {dst}")
    for resource, intervals in plan.timelines.items():
        ordered = sorted(intervals, key=lambda x: x[0])
        for left, right in zip(ordered, ordered[1:]):
            if left[1] > right[0] + 1e-9:
                raise AssertionError(f"{name} overlap on resource {resource}")


def evaluate_static_plan(
    env: EventDrivenOffloadEnv,
    plan: StaticSchedulePlan,
    cost_alpha: float,
    cost_beta: float,
    normalized_cost: bool,
) -> Dict[str, float]:
    delays = np.asarray(
        [
            max(
                (plan.finishes[(i, j)] for j in dag.subtasks),
                default=0.0,
            )
            for i, dag in enumerate(env.env.dag_tasks)
        ],
        dtype=np.float64,
    )
    energies = np.zeros(env.N, dtype=np.float64)
    action_counts = np.zeros(env.action_dim, dtype=np.int64)
    for (i, j), action in plan.actions.items():
        if 1 <= action <= env.M:
            f_alloc = float(env.env.rm.edge_servers[action - 1].F_max)
        else:
            f_alloc = 0.0
        energies[i] += env._placement_energy(
            i,
            j,
            action,
            f_alloc,
            cloud_at_time=plan.starts[(i, j)],
        )
        action_counts[action] += 1

    deadlines = np.asarray(env.env.T_max, dtype=np.float64)
    energy_ref = np.asarray([sum(x) for x in env._E_subtask_ref], dtype=np.float64)
    normalized_delay = float(np.mean(delays / np.maximum(deadlines, 1e-12)))
    normalized_energy = float(np.mean(energies / np.maximum(energy_ref, 1e-12)))
    avg_delay = float(np.mean(delays))
    avg_energy = float(np.mean(energies))
    if normalized_cost:
        cost = cost_alpha * normalized_delay + cost_beta * normalized_energy
    else:
        cost = cost_alpha * avg_delay + cost_beta * avg_energy
    total = max(1, int(action_counts.sum()))
    return {
        "avg_delay": avg_delay,
        "avg_energy": avg_energy,
        "normalized_delay": normalized_delay,
        "normalized_energy": normalized_energy,
        "delay_energy_cost": float(cost),
        "on_time_ratio": float(np.mean(delays <= deadlines + 1e-9)),
        "local_ratio": float(action_counts[0] / total),
        "edge_ratio": float(action_counts[1:env.M + 1].sum() / total),
        "cloud_ratio": float(action_counts[-1] / total),
    }


def choose_greedy_static_cost_action(
    env: EventDrivenOffloadEnv,
    avail: np.ndarray,
) -> int:
    """Choose the lowest nominal one-task cost without dynamic state."""
    i = env._current_i
    j = env._current_j
    base_env = env.env
    st = base_env.dag_tasks[i].subtasks[j]
    alpha = float(base_env.alpha_w)
    beta = float(base_env.beta_w)
    deadline = max(float(base_env.T_max[i]), 1e-9)
    local_energy_ref = max(
        1e-12,
        float(
            base_env.kappa
            * st.c
            * (base_env.f_local[i] ** (base_env.energy_cfg.zeta - 1))
        ),
    )
    distance = float(base_env.distances[i])
    tx_energy = float(base_env.ch.transmission_energy(st.d, distance))
    upload_time = float(base_env.ch.upload_delay(st.d, distance))

    best_action = 0
    best_key = (float("inf"), float("inf"), float("inf"), float("inf"))
    for action in range(env.action_dim):
        if action >= len(avail) or float(avail[action]) <= 0.0:
            continue
        if action == 0:
            nominal_delay = float(st.c / max(base_env.f_local[i], 1e-12))
            nominal_energy = local_energy_ref
        elif 1 <= action <= env.M:
            nominal_cpu = max(
                float(base_env.rm.edge_servers[action - 1].F_max), 1e-12
            )
            exec_time = float(st.c / nominal_cpu)
            nominal_delay = float(base_env._tran_delay(i, j, action) + exec_time)
            nominal_energy = float(
                tx_energy
                + base_env.energy_cfg.edge_wait_power_w
                * (upload_time + exec_time)
            )
        else:
            nominal_cpu = max(float(base_env.rm.cloud.F_cloud), 1e-12)
            exec_time = float(st.c / nominal_cpu)
            nominal_delay = float(base_env._tran_delay(i, j, action) + exec_time)
            nominal_energy = float(
                tx_energy
                + base_env.energy_cfg.cloud_wait_power_w
                * (upload_time + exec_time)
                + base_env.energy_cfg.cloud_backhaul_energy_per_bit * st.d
            )
        score = (
            alpha * nominal_delay / deadline
            + beta * nominal_energy / local_energy_ref
        )
        key = (score, nominal_delay, nominal_energy, action)
        if key < best_key:
            best_key = key
            best_action = action
    return int(best_action)


def choose_greedy_action(env: EventDrivenOffloadEnv, avail: np.ndarray) -> int:
    """One-step weighted-cost greedy baseline using the environment estimator."""
    i = env._current_i
    j = env._current_j
    alpha = float(env.env.alpha_w)
    beta = float(env.env.beta_w)
    best_action = 0
    best_key = (float("inf"), float("inf"), float("inf"))
    for action in range(env.action_dim):
        if action >= len(avail) or float(avail[action]) <= 0.0:
            continue
        out = env.estimate_action_outcome(i, j, action)
        score = alpha * out["delay_ratio"] + beta * out["energy_ratio"]
        key = (score, out["finish"], out["energy"])
        if key < best_key:
            best_key = key
            best_action = action
    return int(best_action)


def load_ddqn_agent(
    run_dir: Path,
    cfg: Dict[str, Any],
    env: EventDrivenOffloadEnv,
    device: torch.device,
) -> DDQNAgent:
    pack = env.reset()
    train_cfg = cfg["train"]
    agent = DDQNAgent(
        state_dim=int(pack["state"].shape[-1]),
        action_dim=int(pack["avail_actions"].shape[-1]),
        hidden_dim=int(cfg["model"].get("hidden_dim", 256)),
        learning_rate=float(train_cfg.get("critic_lr", 3e-4)),
        gamma=float(train_cfg.get("gamma", 0.99)),
        target_update_interval=int(train_cfg.get("target_update_interval", 500)),
        device=device,
    )
    weight_path = run_dir / "final.pt"
    if not weight_path.exists():
        weight_path = run_dir / "best.pt"
    if not weight_path.exists():
        raise FileNotFoundError(f"DDQN weights not found: {run_dir}")
    state = torch.load(weight_path, map_location=device)
    agent.q_network.load_state_dict(state["q_network"])
    agent.target_network.load_state_dict(
        state.get("target_network", state["q_network"])
    )
    agent.q_network.eval()
    agent.target_network.eval()
    return agent


def eval_one_episode(
    env: EventDrivenOffloadEnv,
    policy_type: str,
    policy: Any,
    device: torch.device,
    cost_alpha: float,
    cost_beta: float,
    normalized_cost: bool,
    stochastic_ppo: bool,
    deadline_scale: float = 1.0,
) -> Dict[str, float]:
    pack = env.reset()
    if deadline_scale <= 0.0:
        raise ValueError("deadline_scale must be positive")
    if not np.isclose(deadline_scale, 1.0):
        env.env.T_max = [
            float(deadline) * float(deadline_scale)
            for deadline in env.env.T_max
        ]
        current_obs = env._make_obs(env._current_i, env._current_j)
        pack["obs"] = current_obs[np.newaxis, :]
        pack["state"] = current_obs
    if policy_type == "heft":
        return evaluate_static_plan(
            env,
            build_standard_heft_plan(env),
            cost_alpha,
            cost_beta,
            normalized_cost,
        )
    action_counts = np.zeros(env.action_dim, dtype=np.int64)
    while True:
        if policy_type == "ppo":
            obs_t = torch.as_tensor(pack["obs"], dtype=torch.float32, device=device)
            avail_t = torch.as_tensor(pack["avail_actions"], dtype=torch.float32, device=device)
            if stochastic_ppo:
                action = int(policy.sample_actions(obs_t, avail_t)["actions"].cpu().numpy()[0])
            else:
                action = int(policy.act_deterministic(obs_t, avail_t).cpu().numpy()[0])
        elif policy_type == "ddqn":
            action, _ = policy.select_action(
                pack["state"].astype(np.float32),
                pack["avail_actions"][0].astype(np.float32),
                epsilon=0.0,
            )
        elif policy_type == "greedy":
            action = choose_greedy_action(env, pack["avail_actions"][0])
        elif policy_type == "greedy_static_cost":
            action = choose_greedy_static_cost_action(
                env, pack["avail_actions"][0]
            )
        elif policy_type == "local_only":
            action = 0
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")

        action_counts[action] += 1
        pack = env.step(np.array([action], dtype=np.int64))
        if float(pack["dones"][0]) > 0.5:
            info = pack["infos"]
            break

    t_list = np.asarray(info["T_list"], dtype=np.float64)
    e_list = np.asarray(env._energy, dtype=np.float64)
    avg_delay = float(np.mean(t_list))
    avg_energy = float(np.mean(e_list))
    tmax_list = np.asarray(env.env.T_max, dtype=np.float64)
    energy_ref = np.asarray(
        [sum(refs) for refs in env._E_subtask_ref], dtype=np.float64
    )
    normalized_delay = float(np.mean(t_list / np.maximum(tmax_list, 1e-9)))
    normalized_energy = float(np.mean(e_list / np.maximum(energy_ref, 1e-12)))
    if normalized_cost:
        weighted_cost = cost_alpha * normalized_delay + cost_beta * normalized_energy
    else:
        weighted_cost = cost_alpha * avg_delay + cost_beta * avg_energy
    total_actions = max(1, int(action_counts.sum()))
    return {
        "avg_delay": avg_delay,
        "avg_energy": avg_energy,
        "normalized_delay": normalized_delay,
        "normalized_energy": normalized_energy,
        "delay_energy_cost": float(weighted_cost),
        "on_time_ratio": float(info.get("agent_on_time_ratio", 0.0)),
        "deadline_excess": float(info.get("c1_excess", 0.0)),
        "local_ratio": float(action_counts[0] / total_actions),
        "edge_ratio": float(action_counts[1:env.M + 1].sum() / total_actions),
        "cloud_ratio": float(action_counts[-1] / total_actions),
    }


def evaluate_algorithm(
    algo: Dict[str, Any],
    task_min: int,
    task_max: int,
    episodes: int,
    seed: int,
    device: torch.device,
    cost_alpha: float,
    cost_beta: float,
    normalized_cost: bool,
    checkpoint_episode: int | None = None,
    prefer_best: bool = False,
    stochastic_ppo: bool = False,
) -> Dict[str, float] | None:
    cfg = load_yaml(PROJECT_ROOT / algo["config"])
    if algo.get("variant") == "ippo":
        cfg["env"]["heuristic_state_mode"] = "none"
        cfg["model"]["actor_type"] = "independent"
    elif algo.get("variant") == "ablation_no_heur":
        cfg["env"]["heuristic_state_mode"] = "zero_outcome"
        cfg["env"]["task_selection_mode"] = "urgency"
        cfg["model"]["actor_type"] = "action_conditioned"
    elif algo.get("variant") == "ablation_no_haca":
        cfg["env"]["heuristic_state_mode"] = "strong_outcome"
        cfg["env"]["task_selection_mode"] = "urgency"
        cfg["model"]["actor_type"] = "mlp"
    elif algo.get("variant") == "ablation_no_urgency":
        cfg["env"]["heuristic_state_mode"] = "strong_outcome"
        cfg["env"]["task_selection_mode"] = "round_robin"
        cfg["model"]["actor_type"] = "action_conditioned"
    for key, value in algo.get("env_overrides", {}).items():
        cfg["env"][key] = value
    env = build_env(cfg, task_min, task_max, seed)
    env_cfg = cfg.get("env", {})
    if algo.get("heuristic_mask", False) and bool(env_cfg.get("heuristic_mask", False)):
        scorer = build_scorer(cfg)
        top_k = int(cfg.get("heuristic", {}).get("top_k", 3))
        env.heuristic_action_masker = make_event_masker(scorer, top_k)

    try:
        if algo["type"] == "ppo":
            trained_min, trained_max = algo.get(
                "trained_task_range", (task_min, task_max)
            )
            policy = load_ppo_actor(
                run_dir_for(algo, int(trained_min), int(trained_max)),
                cfg,
                env,
                device,
                checkpoint_episode=checkpoint_episode,
                prefer_best=prefer_best,
            )
        elif algo["type"] == "ddqn":
            trained_min, trained_max = algo.get(
                "trained_task_range", (task_min, task_max)
            )
            policy = load_ddqn_agent(
                run_dir_for(algo, int(trained_min), int(trained_max)),
                cfg,
                env,
                device,
            )
        else:
            policy = None
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[skip] {algo['name']} tasks{task_min}-{task_max}: {exc}")
        return None

    metrics = []
    for ep in range(episodes):
        env.set_episode_index(ep + 1)
        metrics.append(
            eval_one_episode(
                env,
                algo["type"],
                policy,
                device,
                cost_alpha,
                cost_beta,
                normalized_cost,
                stochastic_ppo,
            )
        )

    return {
        "avg_delay": float(np.mean([m["avg_delay"] for m in metrics])),
        "avg_energy": float(np.mean([m["avg_energy"] for m in metrics])),
        "normalized_delay": float(np.mean([m["normalized_delay"] for m in metrics])),
        "normalized_energy": float(np.mean([m["normalized_energy"] for m in metrics])),
        "delay_energy_cost": float(np.mean([m["delay_energy_cost"] for m in metrics])),
        "on_time_ratio": float(np.mean([m["on_time_ratio"] for m in metrics])),
        "local_ratio": float(np.mean([m["local_ratio"] for m in metrics])),
        "edge_ratio": float(np.mean([m["edge_ratio"] for m in metrics])),
        "cloud_ratio": float(np.mean([m["cloud_ratio"] for m in metrics])),
    }


def save_metrics_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "algorithm",
                "task_range",
                "avg_delay",
                "avg_energy",
                "normalized_delay",
                "normalized_energy",
                "delay_energy_cost",
                "on_time_ratio",
                "local_ratio",
                "edge_ratio",
                "cloud_ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def load_metrics_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "algorithm": row["algorithm"],
                    "task_range": row["task_range"],
                    "avg_delay": float(row["avg_delay"]),
                    "avg_energy": float(row["avg_energy"]),
                    "normalized_delay": float(row.get("normalized_delay") or "nan"),
                    "normalized_energy": float(row.get("normalized_energy") or "nan"),
                    "delay_energy_cost": float(row["delay_energy_cost"]),
                    "on_time_ratio": float(row["on_time_ratio"]),
                    "local_ratio": float(row.get("local_ratio") or "nan"),
                    "edge_ratio": float(row.get("edge_ratio") or "nan"),
                    "cloud_ratio": float(row.get("cloud_ratio") or "nan"),
                }
            )
    return rows


def values_for(rows: List[Dict[str, Any]], algorithm: str, task_labels: List[str], metric: str) -> List[float]:
    vals = []
    for label in task_labels:
        match = next(
            (r for r in rows if r["algorithm"] == algorithm and r["task_range"] == label),
            None,
        )
        vals.append(np.nan if match is None else float(match[metric]))
    return vals


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.28, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#B8C0C8")
    ax.spines["bottom"].set_color("#B8C0C8")
    ax.tick_params(colors="#27323A", labelsize=9)


def plot_lines_with_inset(
    rows: List[Dict[str, Any]],
    task_ranges: List[Tuple[int, int]],
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    task_labels = [task_range_label(a, b) for a, b in task_ranges]
    x = np.arange(len(task_labels), dtype=np.float64)

    fig, ax = plt.subplots(figsize=(12.2, 5.9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FBFCFE")
    main_algorithms = PLOT_ORDER if metric == "on_time_ratio" else MAIN_LINE_ALGORITHMS
    for algo in main_algorithms:
        vals = values_for(rows, algo, task_labels, metric)
        if np.all(np.isnan(vals)):
            continue
        highlight = algo == "H-MAPPO"
        ax.plot(
            x,
            vals,
            label=algo,
            color=COLORS[algo],
            marker=MARKERS[algo],
            markersize=7.0 if highlight else 5.8,
            linewidth=2.8 if highlight else 2.0,
            markeredgecolor="white",
            markeredgewidth=0.8,
            alpha=1.0 if highlight else 0.88,
            zorder=6 if highlight else 4,
        )

    if metric != "on_time_ratio":
        inset = ax.inset_axes([0.055, 0.57, 0.32, 0.34])
        inset.set_facecolor("#F4F6F8")
        for algo in INSET_ALGORITHMS:
            vals = values_for(rows, algo, task_labels, metric)
            if np.all(np.isnan(vals)):
                continue
            inset.plot(
                x,
                vals,
                label=algo,
                color=COLORS[algo],
                marker=MARKERS[algo],
                markersize=4.2,
                linewidth=1.6,
                markeredgecolor="white",
                markeredgewidth=0.5,
            )
        inset.set_title("Local-only and HEFT", fontsize=8.5, pad=4)
        inset.set_xticks(x)
        inset.set_xticklabels(task_labels, fontsize=7)
        inset.tick_params(axis="y", labelsize=7)
        inset.grid(True, linestyle="--", linewidth=0.6, alpha=0.25)
        inset.spines[:].set_color("#AAB2BA")
        inset.legend(loc="upper left", frameon=False, fontsize=7.2, handlelength=1.3)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlabel("Number of subtasks", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels)
    if metric == "on_time_ratio":
        ax.set_ylim(-0.03, 1.05)
    style_axis(ax)
    ax.legend(
        ncol=len(main_algorithms),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        frameon=False,
        fontsize=8.8,
        handlelength=1.2,
        columnspacing=1.0,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {output_path}")


def plot_metric_panel(
    rows: List[Dict[str, Any]],
    task_ranges: List[Tuple[int, int]],
    output_path: Path,
    normalized_cost: bool = False,
) -> None:
    task_labels = [task_range_label(a, b) for a, b in task_ranges]
    metrics = [
        ("avg_delay", "Average delay (s)", "(a)"),
        ("avg_energy", "Average energy (J)", "(b)"),
        (
            "delay_energy_cost",
            "Normalized weighted cost" if normalized_cost else "Weighted cost",
            "(c)",
        ),
        ("on_time_ratio", "On-time completion ratio", "(d)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.8))
    fig.patch.set_facecolor("white")
    for ax, (metric, ylabel, panel_label) in zip(axes.ravel(), metrics):
        ax.set_facecolor("#FBFCFE")
        main_algorithms = PLOT_ORDER if metric == "on_time_ratio" else MAIN_LINE_ALGORITHMS
        for algo in main_algorithms:
            vals = values_for(rows, algo, task_labels, metric)
            if np.all(np.isnan(vals)):
                continue
            highlight = algo == "H-MAPPO"
            ax.plot(
                np.arange(len(task_labels)),
                vals,
                marker=MARKERS[algo],
                markersize=6.5 if highlight else 5.0,
                linewidth=2.5 if highlight else 1.8,
                color=COLORS[algo],
                label=algo,
                alpha=1.0 if highlight else 0.82,
                zorder=5 if highlight else 3,
            )
        ax.set_xticks(np.arange(len(task_labels)))
        ax.set_xticklabels(task_labels)
        if metric != "on_time_ratio":
            inset = ax.inset_axes([0.055, 0.57, 0.36, 0.36])
            inset.set_facecolor("#F4F6F8")
            for algo in INSET_ALGORITHMS:
                vals = values_for(rows, algo, task_labels, metric)
                if np.all(np.isnan(vals)):
                    continue
                inset.plot(
                    np.arange(len(task_labels)),
                    vals,
                    label=algo,
                    color=COLORS[algo],
                    marker=MARKERS[algo],
                    markersize=3.2,
                    linewidth=1.2,
                )
            inset.set_xticks(np.arange(len(task_labels)))
            inset.set_xticklabels(task_labels, fontsize=5.8)
            inset.tick_params(axis="y", labelsize=5.8)
            inset.grid(True, linestyle="--", linewidth=0.5, alpha=0.25)
        else:
            ax.set_ylim(-0.03, 1.05)
        ax.set_ylabel(ylabel, fontsize=9.5)
        ax.set_xlabel("Number of subtasks", fontsize=9.5)
        ax.text(
            0.5,
            -0.28,
            panel_label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
        )
        style_axis(ax)

    handles, labels = axes[1, 1].get_legend_handles_labels()
    handle_lookup = dict(zip(labels, handles))
    ordered_handles = [handle_lookup[name] for name in PLOT_ORDER if name in handle_lookup]
    ordered_labels = [name for name in PLOT_ORDER if name in handle_lookup]
    fig.legend(
        ordered_handles,
        ordered_labels,
        ncol=len(PLOT_ORDER),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        frameon=False,
        fontsize=9,
        columnspacing=1.2,
        handlelength=1.5,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.99,
        bottom=0.20,
        top=0.98,
        wspace=0.18,
        hspace=0.52,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_stem = output_path.with_suffix("")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {output_stem}.png/.pdf/.svg")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30, help="Evaluation episodes per task range")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--task_ranges",
        nargs="+",
        default=None,
        help="Task ranges, e.g. 15-20 20-25 25-30 30-35",
    )
    parser.add_argument(
        "--cost_alpha",
        type=float,
        default=None,
        help="Weight of average delay in the reported weighted cost",
    )
    parser.add_argument(
        "--cost_beta",
        type=float,
        default=None,
        help="Weight of average energy in the reported weighted cost",
    )
    parser.add_argument(
        "--hmappo_checkpoint_episode",
        type=int,
        default=0,
        help="Optional H-MAPPO checkpoint episode, e.g. 500. Zero means actor_final/best.",
    )
    parser.add_argument(
        "--prefer_best",
        action="store_true",
        help="Diagnostic: load actor_best.pt instead of actor_final.pt for PPO policies",
    )
    parser.add_argument(
        "--stochastic_ppo",
        action="store_true",
        help="Sample PPO actions from the learned distribution instead of argmax",
    )
    parser.add_argument(
        "--use_paper_runs",
        action="store_true",
        help="Load paper_public_v1 configs and *_paper trained PPO runs",
    )
    parser.add_argument(
        "--use_paper_v2_runs",
        action="store_true",
        help="Load latency-sensitive paper_public_v2 configs and *_paper_v2 runs",
    )
    parser.add_argument(
        "--use_paper_v3_runs",
        action="store_true",
        help="Load standardized public_v3 data, configs and *_paper_v3 runs",
    )
    parser.add_argument(
        "--rule_only",
        action="store_true",
        help="Evaluate only Local-only, Greedy-MinCost, Greedy and HEFT",
    )
    parser.add_argument(
        "--replace_greedy_eft_with_static",
        action="store_true",
        help=(
            "Reuse existing metrics, replace the prior simple-greedy rows "
            "with Greedy, and overwrite the extended comparison CSV"
        ),
    )
    parser.add_argument(
        "--tptdto_only",
        action="store_true",
        help=(
            "Reuse the current public-v3 metrics CSV, evaluate only TPTDTO, "
            "replace prior TPTDTO rows, and redraw the figures"
        ),
    )
    parser.add_argument("--plot_only", action="store_true", help="Only redraw figures from the CSV")
    args = parser.parse_args()

    task_range_args = args.task_ranges
    if task_range_args is None:
        task_range_args = (
            ["10-10", "15-15", "20-20", "25-25", "30-30"]
            if args.use_paper_v3_runs
            else [f"{a}-{b}" for a, b in TASK_RANGES]
        )
    task_ranges = [parse_task_range(x) for x in task_range_args]
    set_seed(args.seed)
    device = torch.device(args.device)
    algorithms = [dict(a) for a in ALGORITHMS]
    selected_profiles = sum(
        bool(x) for x in (
            args.use_paper_runs,
            args.use_paper_v2_runs,
            args.use_paper_v3_runs,
        )
    )
    if selected_profiles > 1:
        raise ValueError("Choose only one paper run profile")
    if not args.use_paper_v3_runs:
        algorithms = [
            a for a in algorithms if not a.get("paper_v3_only", False)
        ]
    if args.use_paper_v3_runs:
        for algo in algorithms:
            if algo["name"] == "H-MAPPO":
                algo["run_prefix"] = "event_ppo_heuristic_paper_v3"
                algo["config"] = "configs/event_ppo_paper_public_v3.yaml"
                algo["trained_task_range"] = (10, 30)
            elif algo["name"] == "MAPPO":
                algo["run_prefix"] = "event_ppo_baseline_paper_v3"
                algo["config"] = "configs/event_ppo_baseline_paper_public_v3.yaml"
                algo["trained_task_range"] = (10, 30)
            elif algo["name"] == "DDQN":
                algo["run_prefix"] = "event_ddqn_paper_v3"
                algo["config"] = "configs/event_ppo_baseline_paper_public_v3.yaml"
                algo["trained_task_range"] = (10, 30)
            elif algo["name"] == "IPPO":
                algo["run_prefix"] = "event_ippo_paper_v3"
                algo["config"] = "configs/event_ppo_baseline_paper_public_v3.yaml"
                algo["trained_task_range"] = (10, 30)
            elif algo["name"] == "TPTDTO":
                algo["run_prefix"] = "event_tptdto_paper_v3"
                algo["config"] = "configs/tptdto_paper_public_v3.yaml"
                algo["trained_task_range"] = (10, 30)
            else:
                algo["config"] = "configs/event_ppo_paper_public_v3.yaml"
    elif args.use_paper_v2_runs:
        for algo in algorithms:
            if algo["name"] == "H-MAPPO":
                algo["run_prefix"] = "event_ppo_heuristic_paper_v2"
                algo["config"] = "configs/event_ppo_paper_public_v2.yaml"
            elif algo["name"] == "MAPPO":
                algo["run_prefix"] = "event_ppo_baseline_paper_v2"
                algo["config"] = "configs/event_ppo_baseline_paper_public_v2.yaml"
            else:
                algo["config"] = "configs/event_ppo_paper_public_v2.yaml"
    elif args.use_paper_runs:
        for algo in algorithms:
            if algo["name"] == "H-MAPPO":
                algo["run_prefix"] = "event_ppo_heuristic_paper"
                algo["config"] = "configs/event_ppo_paper_public.yaml"
            elif algo["name"] == "MAPPO":
                algo["run_prefix"] = "event_ppo_baseline_paper"
                algo["config"] = "configs/event_ppo_baseline_paper_public.yaml"
            else:
                algo["config"] = "configs/event_ppo_paper_public.yaml"
    if args.rule_only:
        algorithms = [
            a for a in algorithms
            if a["type"] in {
                "local_only", "greedy", "greedy_static_cost", "heft"
            }
        ]

    if args.cost_alpha is None:
        cost_alpha = 0.60 if (args.use_paper_v2_runs or args.use_paper_v3_runs) else (0.55 if args.use_paper_runs else 0.50)
    else:
        cost_alpha = args.cost_alpha
    if args.cost_beta is None:
        cost_beta = 0.40 if (args.use_paper_v2_runs or args.use_paper_v3_runs) else (0.45 if args.use_paper_runs else 0.50)
    else:
        cost_beta = args.cost_beta

    normalized_cost = bool(args.use_paper_v3_runs)
    if args.use_paper_v3_runs:
        metrics_name = "public_heft_comparison_paper_v3_metrics.csv"
        output_suffix = "_paper_v3"
    elif args.use_paper_v2_runs:
        metrics_name = "public_heft_comparison_paper_v2_metrics.csv"
        output_suffix = "_paper_v2"
    elif args.use_paper_runs:
        metrics_name = "public_heft_comparison_paper_metrics.csv"
        output_suffix = "_paper"
    else:
        metrics_name = "public_heft_comparison_metrics.csv"
        output_suffix = ""
    base_metrics_path = IMAGE_DIR / metrics_name
    extended_metrics_path = base_metrics_path.with_name(
        f"{base_metrics_path.stem}_with_greedy_eft{base_metrics_path.suffix}"
    )
    if args.tptdto_only and not args.use_paper_v3_runs:
        raise ValueError("--tptdto_only requires --use_paper_v3_runs")
    if args.tptdto_only and (
        args.plot_only or args.replace_greedy_eft_with_static or args.rule_only
    ):
        raise ValueError(
            "--tptdto_only cannot be combined with --plot_only, "
            "--replace_greedy_eft_with_static or --rule_only"
        )
    if args.plot_only and args.replace_greedy_eft_with_static:
        raise ValueError(
            "--plot_only and --replace_greedy_eft_with_static cannot be used together"
        )

    if args.tptdto_only:
        source_metrics_path = (
            extended_metrics_path
            if extended_metrics_path.exists()
            else base_metrics_path
        )
        rows = load_metrics_csv(source_metrics_path)
        if not rows:
            raise FileNotFoundError(
                f"Existing metrics CSV not found: {source_metrics_path}"
            )
        rows = [
            row for row in rows if row["algorithm"] != "TPTDTO"
        ]
        algorithms = [
            a for a in algorithms if a["name"] == "TPTDTO"
        ]
        metrics_path = source_metrics_path
        if source_metrics_path == extended_metrics_path:
            output_suffix = f"{output_suffix}_greedy_eft"
        print(f"[reuse existing metrics] {source_metrics_path}")
    elif args.replace_greedy_eft_with_static:
        source_metrics_path = (
            extended_metrics_path
            if extended_metrics_path.exists()
            else base_metrics_path
        )
        rows = load_metrics_csv(source_metrics_path)
        if not rows:
            raise FileNotFoundError(
                f"Existing metrics CSV not found: {source_metrics_path}"
            )
        # Replace only the simple-greedy rows and preserve every historical
        # algorithm, including the stronger Greedy-MinCost baseline.
        rows = [
            row for row in rows
            if row["algorithm"] not in {
                "Greedy-EFT", "Greedy-StaticCost", "Greedy"
            }
        ]
        algorithms = [
            a for a in algorithms if a["name"] == "Greedy"
        ]
        metrics_path = extended_metrics_path
        output_suffix = f"{output_suffix}_greedy_eft"
        print(f"[reuse existing metrics] {source_metrics_path}")
    elif args.plot_only and extended_metrics_path.exists():
        metrics_path = extended_metrics_path
        output_suffix = f"{output_suffix}_greedy_eft"
    else:
        metrics_path = base_metrics_path

    if args.plot_only:
        rows = load_metrics_csv(metrics_path)
        if not rows:
            raise FileNotFoundError(f"Metrics CSV not found: {metrics_path}")
        print(f"[plot only] {metrics_path}")
    else:
        if not (
            args.replace_greedy_eft_with_static or args.tptdto_only
        ):
            rows = []
        checkpoint_episode = args.hmappo_checkpoint_episode or None
        for task_min, task_max in task_ranges:
            label = task_range_label(task_min, task_max)
            print(f"\n=== Public DAG tasks {label} ===")
            for algo in algorithms:
                result = evaluate_algorithm(
                    algo=algo,
                    task_min=task_min,
                    task_max=task_max,
                    episodes=args.episodes,
                    seed=args.seed,
                    device=device,
                    cost_alpha=cost_alpha,
                    cost_beta=cost_beta,
                    normalized_cost=normalized_cost,
                    checkpoint_episode=checkpoint_episode if algo["name"] == "H-MAPPO" else None,
                    prefer_best=args.prefer_best,
                    stochastic_ppo=args.stochastic_ppo,
                )
                if result is None:
                    continue
                row = {"algorithm": algo["name"], "task_range": label, **result}
                rows.append(row)
                print(
                    f"{algo['name']:<10s} "
                    f"delay={row['avg_delay']:.4f} "
                    f"energy={row['avg_energy']:.4f} "
                    f"cost={row['delay_energy_cost']:.4f} "
                    f"on-time={row['on_time_ratio']:.4f} "
                    f"place=({row['local_ratio']:.2f}/"
                    f"{row['edge_ratio']:.2f}/{row['cloud_ratio']:.2f})"
                )
        save_metrics_csv(rows, metrics_path)
        print(f"[saved] {metrics_path}")

    cost_ylabel = "Normalized weighted cost" if normalized_cost else "Weighted delay-energy cost"
    cost_title = (
        "Public DAG: normalized delay-energy cost comparison"
        if normalized_cost
        else "Public DAG: weighted cost comparison"
    )

    plot_lines_with_inset(
        rows,
        task_ranges,
        metric="avg_delay",
        ylabel="Average delay (s)",
        title="Public DAG: average delay comparison",
        output_path=IMAGE_DIR / f"public_heft_delay_bar{output_suffix}.png",
    )
    plot_lines_with_inset(
        rows,
        task_ranges,
        metric="avg_energy",
        ylabel="Average energy (J)",
        title="Public DAG: average energy comparison",
        output_path=IMAGE_DIR / f"public_heft_energy_bar{output_suffix}.png",
    )
    plot_lines_with_inset(
        rows,
        task_ranges,
        metric="delay_energy_cost",
        ylabel=cost_ylabel,
        title=cost_title,
        output_path=IMAGE_DIR / f"public_heft_cost_bar{output_suffix}.png",
    )
    plot_lines_with_inset(
        rows,
        task_ranges,
        metric="on_time_ratio",
        ylabel="On-time completion ratio",
        title="Public DAG: on-time completion comparison",
        output_path=IMAGE_DIR / f"public_heft_on_time_bar{output_suffix}.png",
    )
    plot_metric_panel(
        rows,
        task_ranges,
        output_path=IMAGE_DIR / f"public_heft_comparison_panel{output_suffix}.png",
        normalized_cost=normalized_cost,
    )


if __name__ == "__main__":
    main()

"""
不同任务数量下的时延/能耗对比评估与绘图。
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.envs.event_env import EventDrivenOffloadEnv
from rl.heuristics import HeuristicScorer, RuleBasedHeuristic, StrongGreedyHeuristic
from rl.mappo.actor_critic import (
    HeuristicActionConditionedActor,
    IndependentPPOActor,
    MAPPONodeActor,
    TPTDTOActor,
)
from rl.offpolicy import DiscreteMADDPGAgent, DiscreteMATD3Agent

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMAGE_DIR = PROJECT_ROOT / "image"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

TASK_RANGES: List[Tuple[int, int]] = [(15, 20), (20, 25), (25, 30), (30, 35)]

ALGORITHMS = [
    {
        "name": "MAPPO",
        "type": "ppo",
        "run_prefix": "event_ppo_baseline",
        "config": "configs/event_ppo_baseline.yaml",
        "heuristic_mask": False,
    },
    {
        "name": "H-MAPPO",
        "type": "ppo",
        "run_prefix": "event_ppo_heuristic",
        "config": "configs/event_ppo.yaml",
        "heuristic_mask": True,
    },
    {
        "name": "MADDPG",
        "type": "maddpg",
        "run_prefix": "event_maddpg",
        "config": "configs/event_maddpg.yaml",
        "heuristic_mask": False,
    },
    {
        "name": "MATD3",
        "type": "matd3",
        "run_prefix": "event_matd3",
        "config": "configs/event_matd3.yaml",
        "heuristic_mask": False,
    },
    {
        "name": "Greedy",
        "type": "greedy",
        "run_prefix": "",
        "config": "configs/event_ppo.yaml",
        "heuristic_mask": False,
    },
    {
        "name": "All-Local",
        "type": "all_local",
        "run_prefix": "",
        "config": "configs/event_ppo.yaml",
        "heuristic_mask": False,
    },
    {
        "name": "All-Cloud",
        "type": "all_cloud",
        "run_prefix": "",
        "config": "configs/event_ppo.yaml",
        "heuristic_mask": False,
    },
    {
        "name": "All-Edge",
        "type": "all_edge",
        "run_prefix": "",
        "config": "configs/event_ppo.yaml",
        "heuristic_mask": False,
    },
]

COLORS = {
    "H-MAPPO": "#D62728",
    "MAPPO": "#1F77B4",
    "Greedy": "#9467BD",
    "MADDPG": "#2CA02C",
    "MATD3": "#FF7F0E",
    "All-Local": "#8C564B",
    "All-Cloud": "#E377C2",
    "All-Edge": "#7F7F7F",
}

LINE_STYLES = {
    "H-MAPPO": "-",
    "MAPPO": "--",
    "Greedy": ":",
    "MADDPG": "--",
    "MATD3": "-.",
    "All-Local": "-",
    "All-Cloud": "-.",
    "All-Edge": ":",
}

MARKERS = {
    "H-MAPPO": "o",
    "MAPPO": "s",
    "Greedy": "D",
    "MADDPG": "^",
    "MATD3": "v",
    "All-Local": "P",
    "All-Cloud": "X",
    "All-Edge": "*",
}

PLOT_ORDER = [
    "All-Local",
    "All-Edge",
    "MADDPG",
    "MATD3",
    "All-Cloud",
    "Greedy",
    "MAPPO",
    "H-MAPPO",
]

CORE_PLOT_ORDER = ["H-MAPPO", "MAPPO", "Greedy", "All-Local", "All-Edge"]


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_task_range(text: str) -> Tuple[int, int]:
    left, right = text.split("-")
    return int(left), int(right)


def make_event_masker(scorer, top_k: int):
    def masker(env: EventDrivenOffloadEnv) -> np.ndarray:
        scores = scorer.score_step(env, env._current_i, env._current_j)
        mask = np.zeros((1, env.action_dim), dtype=np.float32)
        k = max(1, min(top_k, env.action_dim))
        top_idx = np.argsort(scores)[-k:]
        mask[0, top_idx] = 1.0
        mask[0, 0] = 1.0
        return mask

    return masker


def build_scorer(cfg: Dict[str, Any]):
    h = cfg.get("heuristic", {})
    h_type = h.get("type", "scoring")
    if h_type == "rule_based":
        return RuleBasedHeuristic(
            critical_ratio_thresh=float(h.get("critical_ratio_thresh", 0.7)),
            slack_thresh=float(h.get("slack_thresh", 0.25)),
            cloud_speedup_thresh=float(h.get("cloud_speedup_thresh", 0.85)),
            small_compute_ratio=float(h.get("small_compute_ratio", 0.5)),
            large_data_ratio=float(h.get("large_data_ratio", 0.8)),
            local_competitive_ratio=float(h.get("local_competitive_ratio", 1.05)),
            primary_bonus=float(h.get("primary_bonus", 1.0)),
        )
    if h_type in {"strong_greedy", "oracle_greedy", "one_step_oracle"}:
        return StrongGreedyHeuristic(
            primary_bonus=float(h.get("primary_bonus", 1.0)),
        )
    return HeuristicScorer(
        edge_alloc_hz=h.get("edge_alloc_hz", 1e9),
        load_penalty=h.get("load_penalty", 0.2),
        risk_penalty=h.get("risk_penalty", 4.0),
        criticality_weight=h.get("criticality_weight", 0.5),
        deadline_margin=h.get("deadline_margin", 0.9),
        slack_penalty=h.get("slack_penalty", 1.5),
        over_deadline_penalty=h.get("over_deadline_penalty", 4.0),
        cloud_bias=h.get("cloud_bias", 0.05),
        local_bias=h.get("local_bias", 0.05),
        critical_path_weight=h.get("critical_path_weight", 2.0),
        successor_data_weight=h.get("successor_data_weight", 0.8),
        placement_consistency_weight=h.get("placement_consistency_weight", 0.6),
        future_load_weight=h.get("future_load_weight", 1.0),
        cloud_deadline_bonus=h.get("cloud_deadline_bonus", 0.3),
        cloud_energy_penalty=h.get("cloud_energy_penalty", 1.0),
        edge_preference_bonus=h.get("edge_preference_bonus", 0.2),
    )


def choose_greedy_action(env: EventDrivenOffloadEnv, avail: np.ndarray) -> int:

    i = env._current_i
    j = env._current_j
    home_edge = int(env.env.home_servers[i] + 1)
    cloud = env.action_dim - 1
    candidates = [0, home_edge, cloud]

    delays = []
    for action in candidates:
        if action >= len(avail) or avail[action] <= 0:
            delays.append(float("inf"))
            continue

        if action == 0:
            delay = env.env._local_exec_time(i, j)
        elif 1 <= action <= env.M:
            f_alloc = float(env.env.rm.baseline_edge_alloc_hz)
            delay = env.env._tran_delay(i, j, action) + env.env._edge_exec_time(i, j, f_alloc)
        else:
            cloud_time = env.env._cloud_exec_time(i, j)
            delay = env.env._tran_delay(i, j, action) + cloud_time
        delays.append(float(delay))

    return int(candidates[int(np.argmin(np.asarray(delays, dtype=np.float64)))])


def build_env(cfg: Dict[str, Any], task_min: int, task_max: int, seed: int) -> EventDrivenOffloadEnv:
    env_cfg = cfg["env"]
    return EventDrivenOffloadEnv(
        # 评估脚本固定使用测试 DAG，避免在训练集上评估泛化性能。
        split="test",
        num_agents=env_cfg.get("num_agents", 12),
        seed=seed,
        sample_with_replacement=env_cfg.get("sample_with_replacement", False),
        min_subtasks=task_min,
        max_subtasks=task_max,
        curriculum_schedule=None,
        f_alloc_mode=env_cfg.get("f_alloc_mode", "baseline"),
        cloud_share_mode=env_cfg.get("cloud_share_mode", "shared"),
        cloud_share_factor=float(env_cfg.get("cloud_share_factor", 0.3)),
        cloud_share_floor=float(env_cfg.get("cloud_share_floor", 0.4)),
        heuristic_state_mode=env_cfg.get("heuristic_state_mode", "none"),
        task_selection_mode=env_cfg.get("task_selection_mode", "urgency"),
        urgency_progress_weight=float(env_cfg.get("urgency_progress_weight", 0.1)),
        tmax_mode=env_cfg.get("tmax_mode", "adaptive"),
        tmax_slack=float(env_cfg.get("tmax_slack", 1.5)),
        tmax_min=float(env_cfg.get("tmax_min", 10.0)),
        tmax_max=float(env_cfg.get("tmax_max", 120.0)),
        deadline_penalty=float(env_cfg.get("deadline_penalty", 2.0)),
        dataset_mode=cfg.get("dataset", "public"),
        d_scale=float(env_cfg.get("d_scale", 1e6)),
        c_scale=float(env_cfg.get("c_scale", 1e8)),
        delta_scale=float(env_cfg.get("delta_scale", 1e6)),
        system_profile=env_cfg.get("system_profile", "default"),
        objective_alpha=env_cfg.get("objective_alpha"),
        objective_beta=env_cfg.get("objective_beta"),
    )


def run_dir_for(algo: Dict[str, Any], task_min: int, task_max: int) -> Path:
    return OUTPUTS_DIR / f"{algo['run_prefix']}_tasks{task_min}-{task_max}"


def load_ppo_actor(
    run_dir: Path,
    cfg: Dict[str, Any],
    env: EventDrivenOffloadEnv,
    device: torch.device,
    checkpoint_episode: int | None = None,
    prefer_best: bool = False,
):
    obs_dim = int(env.obs_dim)
    action_dim = int(env.action_dim)
    actor_type = cfg["model"].get("actor_type", "mlp")
    actor_kwargs = {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "hidden_dim": int(cfg["model"].get("actor_hidden_dim", 128)),
    }
    if actor_type == "action_conditioned":
        actor = HeuristicActionConditionedActor(**actor_kwargs).to(device)
    elif actor_type == "independent":
        actor = IndependentPPOActor(
            **actor_kwargs,
            num_agents=int(cfg["env"].get("num_agents", 12)),
        ).to(device)
    elif actor_type == "tptdto":
        actor = TPTDTOActor(
            **actor_kwargs,
            num_agents=int(cfg["env"].get("num_agents", 12)),
            task_feature_dim=int(cfg["model"].get("task_feature_dim", 10)),
            transformer_heads=int(
                cfg["model"].get("transformer_heads", 4)
            ),
            transformer_layers=int(
                cfg["model"].get("transformer_layers", 2)
            ),
            transformer_dropout=float(
                cfg["model"].get("transformer_dropout", 0.0)
            ),
        ).to(device)
    else:
        actor = MAPPONodeActor(**actor_kwargs).to(device)
    if checkpoint_episode is not None:
        weight_path = run_dir / f"actor_ep{checkpoint_episode:05d}.pt"
    elif prefer_best:
        weight_path = run_dir / "actor_best.pt"
        if not weight_path.exists():
            weight_path = run_dir / "actor_final.pt"
    else:
        weight_path = run_dir / "actor_final.pt"
        if not weight_path.exists():
            weight_path = run_dir / "actor_best.pt"
    if not weight_path.exists():
        raise FileNotFoundError(f"找不到 PPO 权重: {weight_path}")
    actor.load_state_dict(torch.load(weight_path, map_location=device))
    actor.eval()
    return actor


def load_offpolicy_agent(
    algo_type: str,
    run_dir: Path,
    cfg: Dict[str, Any],
    env: EventDrivenOffloadEnv,
    device: torch.device,
):
    reset_pack = env.reset()
    state_dim = int(reset_pack["state"].shape[-1])
    action_dim = int(reset_pack["avail_actions"].shape[-1])
    train_cfg = cfg["train"]
    cls = DiscreteMATD3Agent if algo_type == "matd3" else DiscreteMADDPGAgent
    kwargs = dict(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=int(cfg["model"].get("hidden_dim", 256)),
        actor_lr=float(train_cfg.get("actor_lr", 3e-4)),
        critic_lr=float(train_cfg.get("critic_lr", 3e-4)),
        gamma=float(train_cfg.get("gamma", 0.99)),
        tau=float(train_cfg.get("tau", 0.01)),
        device=device,
    )
    if algo_type == "matd3":
        agent = cls(
            **kwargs,
            policy_noise=float(train_cfg.get("policy_noise", 0.1)),
            noise_clip=float(train_cfg.get("noise_clip", 0.2)),
            policy_update_freq=int(train_cfg.get("policy_update_freq", 2)),
        )
    else:
        agent = cls(**kwargs)

    weight_path = run_dir / "final.pt"
    if not weight_path.exists():
        weight_path = run_dir / "best.pt"
    if not weight_path.exists():
        raise FileNotFoundError(f"找不到 off-policy 权重: {run_dir}/final.pt")
    state = torch.load(weight_path, map_location=device)
    for actor, actor_state in zip(agent.actors, state["actors"]):
        actor.load_state_dict(actor_state)
    if "actors_target" in state:
        for actor, actor_state in zip(agent.actors_target, state["actors_target"]):
            actor.load_state_dict(actor_state)
    agent.critic.load_state_dict(state["critic"])
    agent.critic_target.load_state_dict(state.get("critic_target", state["critic"]))
    if algo_type == "matd3":
        agent.critic_2.load_state_dict(state["critic_2"])
        agent.critic_2_target.load_state_dict(state.get("critic_2_target", state["critic_2"]))
    return agent


@torch.no_grad()
def eval_one_episode(env: EventDrivenOffloadEnv, policy_type: str, policy, device: torch.device) -> Dict[str, float]:
    pack = env.reset()
    while True:
        if policy_type == "ppo":
            obs_t = torch.as_tensor(pack["obs"], dtype=torch.float32, device=device)
            avail_t = torch.as_tensor(pack["avail_actions"], dtype=torch.float32, device=device)
            action = int(policy.act_deterministic(obs_t, avail_t).cpu().numpy()[0])
        elif policy_type in {"maddpg", "matd3"}:
            state = pack["state"].astype(np.float32)
            avail = pack["avail_actions"][0].astype(np.float32)
            action, _ = policy.select_action(state, avail, epsilon=0.0)
        elif policy_type == "greedy":
            avail = pack["avail_actions"][0]
            action = choose_greedy_action(env, avail)
        elif policy_type == "all_local":
            action = 0
        elif policy_type == "all_cloud":
            action = env.action_dim - 1
        elif policy_type == "all_edge":
            action = int(env.env.home_servers[env._current_i] + 1)
        else:
            raise ValueError(f"未知 policy_type: {policy_type}")

        pack = env.step(np.array([action], dtype=np.int64))
        if float(pack["dones"][0]) > 0.5:
            info = pack["infos"]
            break

    t_list = np.asarray(info["T_list"], dtype=np.float64)
    e_list = np.asarray(env._energy, dtype=np.float64)
    alpha = float(env.env.alpha_w)
    beta = float(env.env.beta_w)
    avg_delay = float(np.mean(t_list))
    avg_energy = float(np.mean(e_list))
    return {
        "avg_delay": avg_delay,
        "avg_energy": avg_energy,
        "delay_energy_cost": float(alpha * avg_delay + beta * avg_energy),
        "on_time_ratio": float(info.get("agent_on_time_ratio", 0.0)),
    }


def evaluate_algorithm(
    algo: Dict[str, Any],
    task_min: int,
    task_max: int,
    episodes: int,
    seed: int,
    device: torch.device,
    checkpoint_episode: int | None = None,
) -> Dict[str, float] | None:
    cfg = load_yaml(PROJECT_ROOT / algo["config"])
    env = build_env(cfg, task_min, task_max, seed)
    env_cfg = cfg.get("env", {})
    if algo.get("heuristic_mask", False) and bool(env_cfg.get("heuristic_mask", False)):
        scorer = build_scorer(cfg)
        top_k = int(cfg.get("heuristic", {}).get("top_k", 3))
        env.heuristic_action_masker = make_event_masker(scorer, top_k)

    try:
        if algo["type"] == "ppo":
            run_dir = run_dir_for(algo, task_min, task_max)
            policy = load_ppo_actor(
                run_dir,
                cfg,
                env,
                device,
                checkpoint_episode=checkpoint_episode,
            )
        elif algo["type"] in {"maddpg", "matd3"}:
            run_dir = run_dir_for(algo, task_min, task_max)
            policy = load_offpolicy_agent(algo["type"], run_dir, cfg, env, device)
        elif algo["type"] == "greedy":
            policy = None
        elif algo["type"] in {"all_local", "all_cloud", "all_edge"}:
            policy = None
        else:
            raise ValueError(algo["type"])
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[跳过] {algo['name']} tasks{task_min}-{task_max}: {exc}")
        return None

    metrics = []
    for ep in range(episodes):
        env.set_episode_index(ep + 1)
        metrics.append(eval_one_episode(env, algo["type"], policy, device))

    return {
        "avg_delay": float(np.mean([m["avg_delay"] for m in metrics])),
        "avg_energy": float(np.mean([m["avg_energy"] for m in metrics])),
        "delay_energy_cost": float(np.mean([m["delay_energy_cost"] for m in metrics])),
        "on_time_ratio": float(np.mean([m["on_time_ratio"] for m in metrics])),
    }


def save_metrics_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "algorithm", "task_range", "avg_delay",
                "avg_energy", "delay_energy_cost", "on_time_ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def load_metrics_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def plot_lines(
    rows: List[Dict[str, Any]],
    task_ranges: List[Tuple[int, int]],
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
    algorithms: List[str] | None = None,
) -> None:
    task_labels = [f"{a}-{b}" for a, b in task_ranges]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    known_order = algorithms if algorithms is not None else PLOT_ORDER
    extra_order = sorted({
        str(row["algorithm"])
        for row in rows
        if str(row["algorithm"]) not in known_order
    })
    for algo in known_order + extra_order:
        vals = []
        for label in task_labels:
            match = next((r for r in rows if r["algorithm"] == algo and r["task_range"] == label), None)
            vals.append(np.nan if match is None else float(match[metric]))
        if np.all(np.isnan(vals)):
            continue
        highlight = algo == "H-MAPPO"
        ax.plot(
            task_labels,
            vals,
            marker=MARKERS.get(algo, "o"),
            markersize=8 if highlight else 6.5,
            linewidth=2.8 if highlight else 2.0,
            label=algo,
            color=COLORS.get(algo),
            linestyle=LINE_STYLES.get(algo, "-"),
            alpha=1.0 if highlight else 0.88,
            zorder=5 if highlight else 3,
        )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Task Number Range")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(
        fontsize=9,
        ncol=4 if algorithms is None else min(3, len(known_order)),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        frameon=False,
    )
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    print(f"[已保存] {output_path}")
    plt.close(fig)


def plot_grouped_bars(
    rows: List[Dict[str, Any]],
    task_ranges: List[Tuple[int, int]],
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
    algorithms: List[str],
) -> None:
    task_labels = [f"{a}-{b}" for a, b in task_ranges]
    x = np.arange(len(task_labels), dtype=np.float64)
    width = 0.15
    offsets = (np.arange(len(algorithms)) - (len(algorithms) - 1) / 2.0) * width

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for idx, algo in enumerate(algorithms):
        vals = []
        for label in task_labels:
            match = next((r for r in rows if r["algorithm"] == algo and r["task_range"] == label), None)
            vals.append(np.nan if match is None else float(match[metric]))
        bars = ax.bar(
            x + offsets[idx],
            vals,
            width=width,
            label=algo,
            color=COLORS.get(algo),
            edgecolor="white",
            linewidth=0.8,
            alpha=0.95,
            zorder=3,
        )
        if algo == "H-MAPPO":
            for bar in bars:
                bar.set_edgecolor("#222222")
                bar.set_linewidth(1.2)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Task Number Range")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax.legend(
        fontsize=9,
        ncol=len(algorithms),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        frameon=False,
    )
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    print(f"[已保存] {output_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30, help="每个任务范围评估 episode 数")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--task_ranges",
        nargs="+",
        default=[f"{a}-{b}" for a, b in TASK_RANGES],
        help="任务范围，例如 15-20 20-25 25-30 30-35",
    )
    parser.add_argument(
        "--hmappo_checkpoint_episode",
        type=int,
        default=0,
        help="只额外评估 H-MAPPO 指定轮次权重并合并进已有 CSV，例如 500",
    )
    parser.add_argument(
        "--plot_only",
        action="store_true",
        help="只读取已有 task_range_delay_energy_metrics.csv 并重新绘图，不重新评估模型",
    )
    args = parser.parse_args()

    task_ranges = [parse_task_range(x) for x in args.task_ranges]
    set_seed(args.seed)
    device = torch.device(args.device)

    image_dir = PROJECT_ROOT / "image"
    metrics_path = image_dir / "task_range_delay_energy_metrics.csv"

    if args.plot_only:
        rows = load_metrics_csv(metrics_path)
        if not rows:
            raise FileNotFoundError(f"找不到已有指标 CSV: {metrics_path}")
        print(f"只绘图，读取已有 CSV: {metrics_path}")
    elif args.hmappo_checkpoint_episode > 0:
        rows = load_metrics_csv(metrics_path)
        ckpt_ep = args.hmappo_checkpoint_episode
        extra_name = f"H-MAPPO-{ckpt_ep}"
        # 避免重复运行后 CSV 里有旧的同名行
        rows = [r for r in rows if r.get("algorithm") != extra_name]
        hmappo = next(a for a in ALGORITHMS if a["name"] == "H-MAPPO")

        print(f"只评估 {extra_name}，加载 actor_ep{ckpt_ep:05d}.pt，并合并进已有 CSV")
        for task_min, task_max in task_ranges:
            label = f"{task_min}-{task_max}"
            result = evaluate_algorithm(
                algo=hmappo,
                task_min=task_min,
                task_max=task_max,
                episodes=args.episodes,
                seed=args.seed,
                device=device,
                checkpoint_episode=ckpt_ep,
            )
            if result is None:
                continue
            row = {"algorithm": extra_name, "task_range": label, **result}
            rows.append(row)
            print(
                f"{extra_name:<10s} tasks={label} delay={row['avg_delay']:.4f} "
                f"energy={row['avg_energy']:.6g} cost={row['delay_energy_cost']:.4f}"
            )
    else:
        rows: List[Dict[str, Any]] = []
        for task_min, task_max in task_ranges:
            label = f"{task_min}-{task_max}"
            print(f"\n=== Evaluating tasks {label} ===")
            for algo in ALGORITHMS:
                result = evaluate_algorithm(
                    algo=algo,
                    task_min=task_min,
                    task_max=task_max,
                    episodes=args.episodes,
                    seed=args.seed,
                    device=device,
                )
                if result is None:
                    continue
                row = {"algorithm": algo["name"], "task_range": label, **result}
                rows.append(row)
                print(
                    f"{algo['name']:<10s} delay={row['avg_delay']:.4f} "
                    f"energy={row['avg_energy']:.6g} cost={row['delay_energy_cost']:.4f}"
                )

    save_metrics_csv(rows, metrics_path)

    plot_lines(
        rows,
        task_ranges,
        metric="delay_energy_cost",
        ylabel="Weighted Delay-Energy Cost",
        title="Different Task Numbers vs Delay-Energy Cost",
        output_path=image_dir / "task_range_delay_energy_cost.png",
    )
    plot_lines(
        rows,
        task_ranges,
        metric="delay_energy_cost",
        ylabel="Weighted Delay-Energy Cost",
        title="Core Algorithms vs Delay-Energy Cost",
        output_path=image_dir / "task_range_delay_energy_cost_core.png",
        algorithms=CORE_PLOT_ORDER,
    )
    plot_grouped_bars(
        rows,
        task_ranges,
        metric="delay_energy_cost",
        ylabel="Weighted Delay-Energy Cost",
        title="Core Algorithms vs Delay-Energy Cost",
        output_path=image_dir / "task_range_delay_energy_cost_core_bar.png",
        algorithms=CORE_PLOT_ORDER,
    )
    plot_lines(
        rows,
        task_ranges,
        metric="avg_delay",
        ylabel="Average Delay (s)",
        title="Different Task Numbers vs Delay",
        output_path=image_dir / "task_range_delay.png",
    )
    plot_lines(
        rows,
        task_ranges,
        metric="avg_delay",
        ylabel="Average Delay (s)",
        title="Core Algorithms vs Delay",
        output_path=image_dir / "task_range_delay_core.png",
        algorithms=CORE_PLOT_ORDER,
    )
    plot_grouped_bars(
        rows,
        task_ranges,
        metric="avg_delay",
        ylabel="Average Delay (s)",
        title="Core Algorithms vs Delay",
        output_path=image_dir / "task_range_delay_core_bar.png",
        algorithms=CORE_PLOT_ORDER,
    )
    plot_lines(
        rows,
        task_ranges,
        metric="avg_energy",
        ylabel="Average Energy (J)",
        title="Different Task Numbers vs Energy",
        output_path=image_dir / "task_range_energy.png",
    )
    plot_lines(
        rows,
        task_ranges,
        metric="avg_energy",
        ylabel="Average Energy (J)",
        title="Core Algorithms vs Energy",
        output_path=image_dir / "task_range_energy_core.png",
        algorithms=CORE_PLOT_ORDER,
    )
    plot_grouped_bars(
        rows,
        task_ranges,
        metric="avg_energy",
        ylabel="Average Energy (J)",
        title="Core Algorithms vs Energy",
        output_path=image_dir / "task_range_energy_core_bar.png",
        algorithms=CORE_PLOT_ORDER,
    )


if __name__ == "__main__":
    main()

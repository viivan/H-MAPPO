"""
当前事件驱动环境每步只处理 1 个 ready 子任务，因此这里使用共享 actor
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.envs.event_env import EventDrivenOffloadEnv
from rl.heuristics import HeuristicScorer
from rl.offpolicy import DDQNAgent, DiscreteMADDPGAgent, DiscreteMATD3Agent, ReplayBuffer


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_event_masker(scorer: HeuristicScorer, top_k: int):
    def masker(env: EventDrivenOffloadEnv) -> np.ndarray:
        scores = scorer.score_step(env, env._current_i, env._current_j)
        mask = np.zeros((1, env.action_dim), dtype=np.float32)
        k = max(1, min(top_k, env.action_dim))
        top_idx = np.argsort(scores)[-k:]
        mask[0, top_idx] = 1.0
        mask[0, 0] = 1.0
        return mask
    return masker


def maybe_enable_heuristic(env: EventDrivenOffloadEnv, cfg: Dict[str, Any]) -> None:
    heuristic_cfg = cfg.get("heuristic")
    if not heuristic_cfg:
        print("启发式 action masking 未启用")
        return
    scorer = HeuristicScorer(
        edge_alloc_hz=heuristic_cfg.get("edge_alloc_hz", 1e9),
        load_penalty=heuristic_cfg.get("load_penalty", 0.2),
        risk_penalty=heuristic_cfg.get("risk_penalty", 4.0),
        criticality_weight=heuristic_cfg.get("criticality_weight", 0.5),
        deadline_margin=heuristic_cfg.get("deadline_margin", 0.9),
        slack_penalty=heuristic_cfg.get("slack_penalty", 1.5),
        over_deadline_penalty=heuristic_cfg.get("over_deadline_penalty", 4.0),
        cloud_bias=heuristic_cfg.get("cloud_bias", 0.05),
        local_bias=heuristic_cfg.get("local_bias", 0.05),
    )
    top_k = int(heuristic_cfg.get("top_k", 3))
    env.heuristic_action_masker = make_event_masker(scorer, top_k)
    print(f"启发式 action masking 已启用 (top_k={top_k})")


def open_csv(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "episode", "ep_return", "ep_mean_reward", "ep_on_time_ratio",
            "c1_excess", "ep_feasible", "ep_steps",
            "actor_loss", "critic_loss", "actor_grad_norm", "critic_grad_norm",
            "actor_lr", "critic_lr", "epsilon",
        ],
    )
    writer.writeheader()
    return f, writer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--algorithm", choices=["ddqn", "maddpg", "matd3"], default=None)
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    if args.algorithm == "ddqn":
        cfg["algorithm"] = "ddqn"
        cfg["model"]["hidden_dim"] = int(cfg["model"].get("hidden_dim", 256))
        cfg["train"].setdefault("batch_size", 256)
        cfg["train"].setdefault("buffer_size", 100000)
        cfg["train"].setdefault("updates_per_step", 1)
        cfg["train"].setdefault("epsilon_start", 1.0)
        cfg["train"].setdefault("epsilon_end", 0.05)
        cfg["train"].setdefault("epsilon_decay", 0.995)
        cfg["train"].setdefault("target_update_interval", 500)
        cfg["output"]["algorithm_name"] = "event_ddqn_paper_v3"
    set_seed(int(cfg["seed"]))
    device = torch.device(args.device or cfg.get("device", "cpu"))

    env_cfg = cfg["env"]
    env = EventDrivenOffloadEnv(
        split=env_cfg["split"],
        num_agents=env_cfg["num_agents"],
        seed=cfg["seed"],
        sample_with_replacement=env_cfg.get("sample_with_replacement", False),
        min_subtasks=env_cfg.get("min_subtasks"),
        max_subtasks=env_cfg.get("max_subtasks"),
        curriculum_schedule=env_cfg.get("curriculum_schedule"),
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
    )
    maybe_enable_heuristic(env, cfg)

    reset_pack = env.reset()
    state_dim = int(reset_pack["state"].shape[-1])
    action_dim = int(reset_pack["avail_actions"].shape[-1])
    print(f"state_dim={state_dim}, action_dim={action_dim}, max_episode_steps={env.max_episode_steps}")

    train_cfg = cfg["train"]
    algo = cfg["algorithm"].lower()
    agent_cls = DiscreteMATD3Agent if algo == "matd3" else DiscreteMADDPGAgent
    common_kwargs = dict(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=int(cfg["model"].get("hidden_dim", 256)),
        actor_lr=float(train_cfg.get("actor_lr", 3e-4)),
        critic_lr=float(train_cfg.get("critic_lr", 3e-4)),
        gamma=float(train_cfg.get("gamma", 0.99)),
        tau=float(train_cfg.get("tau", 0.01)),
        device=device,
    )
    if algo == "ddqn":
        agent = DDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=int(cfg["model"].get("hidden_dim", 256)),
            learning_rate=float(train_cfg.get("critic_lr", 3e-4)),
            gamma=float(train_cfg.get("gamma", 0.99)),
            target_update_interval=int(train_cfg.get("target_update_interval", 500)),
            device=device,
        )
    elif algo == "matd3":
        agent = agent_cls(
            **common_kwargs,
            policy_noise=float(train_cfg.get("policy_noise", 0.1)),
            noise_clip=float(train_cfg.get("noise_clip", 0.2)),
            policy_update_freq=int(train_cfg.get("policy_update_freq", 2)),
        )
    else:
        agent = agent_cls(**common_kwargs)

    replay = ReplayBuffer(capacity=int(train_cfg.get("buffer_size", 100000)))
    min_tasks = env_cfg.get("min_subtasks", "any")
    max_tasks = env_cfg.get("max_subtasks", "any")
    run_name = f"{cfg['output']['algorithm_name']}_tasks{min_tasks}-{max_tasks}"
    output_dir = Path(cfg["output"]["root_dir"]) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    csv_file, csv_writer = open_csv(output_dir / f"metrics_{cfg['output']['algorithm_name']}.csv")
    print(f"输出目录: {output_dir}")

    total_episodes = int(train_cfg["total_episodes"])
    batch_size = int(train_cfg.get("batch_size", 256))
    updates_per_step = int(train_cfg.get("updates_per_step", 1))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    epsilon = float(train_cfg.get("epsilon_start", 0.3))
    eps_end = float(train_cfg.get("epsilon_end", 0.05))
    eps_decay = float(train_cfg.get("epsilon_decay", 0.995))
    log_interval = int(train_cfg.get("log_interval", 10))
    save_interval = int(train_cfg.get("save_interval", 500))
    best_metric = float("-inf")

    try:
        for ep in range(1, total_episodes + 1):
            env.set_episode_index(ep)
            pack = env.reset()
            ep_rewards = []
            ep_steps = 0
            last_metrics = None

            while True:
                state = pack["state"].astype(np.float32)
                avail = pack["avail_actions"][0].astype(np.float32)
                action, action_vec = agent.select_action(state, avail, epsilon=epsilon)
                next_pack = env.step(np.array([action], dtype=np.int64))
                reward = float(next_pack["rewards"][0])
                done = bool(float(next_pack["dones"][0]) > 0.5)
                next_state = next_pack["state"].astype(np.float32)
                next_avail = next_pack["avail_actions"][0].astype(np.float32)
                replay.push(state, avail, action_vec, reward, next_state, next_avail, done)

                for _ in range(updates_per_step):
                    last_metrics = agent.update(replay, batch_size=batch_size, max_grad_norm=max_grad_norm)

                ep_rewards.append(reward)
                ep_steps += 1
                pack = next_pack
                if done:
                    terminal = next_pack["infos"]
                    break

            epsilon = max(eps_end, epsilon * eps_decay)
            if last_metrics is None:
                last_metrics = agent.update(replay, batch_size=batch_size, max_grad_norm=max_grad_norm)
            row = {
                "episode": ep,
                "ep_return": float(np.sum(ep_rewards)),
                "ep_mean_reward": float(np.mean(ep_rewards)),
                "ep_on_time_ratio": float(terminal.get("agent_on_time_ratio", 0.0)),
                "c1_excess": float(terminal.get("c1_excess", 0.0)),
                "ep_feasible": float(terminal.get("ep_feasible", 0.0)),
                "ep_steps": ep_steps,
                "actor_loss": last_metrics.actor_loss,
                "critic_loss": last_metrics.critic_loss,
                "actor_grad_norm": last_metrics.actor_grad_norm,
                "critic_grad_norm": last_metrics.critic_grad_norm,
                "actor_lr": last_metrics.actor_lr,
                "critic_lr": last_metrics.critic_lr,
                "epsilon": epsilon,
            }
            csv_writer.writerow(row)
            csv_file.flush()

            metric = row["ep_return"] if algo == "ddqn" else row["ep_on_time_ratio"]
            if metric > best_metric:
                best_metric = metric
                torch.save(agent.state_dict(), output_dir / "best.pt")
            if save_interval > 0 and ep % save_interval == 0:
                torch.save(agent.state_dict(), output_dir / f"ep{ep}.pt")
            if ep % log_interval == 0 or ep == 1:
                print(
                    f"[Ep {ep:05d}] ret={row['ep_return']:+.2f} "
                    f"r_step={row['ep_mean_reward']:+.4f} "
                    f"on_time={row['ep_on_time_ratio']:.3f} "
                    f"c1x={row['c1_excess']:.3f} steps={ep_steps} "
                    f"| actor={row['actor_loss']:.4f} critic={row['critic_loss']:.4f} "
                    f"eps={epsilon:.3f}"
                )
    finally:
        torch.save(agent.state_dict(), output_dir / "final.pt")
        csv_file.close()


if __name__ == "__main__":
    main()

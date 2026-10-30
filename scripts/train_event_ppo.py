"""
事件驱动 PPO 训练入口（支持可选启发式 action masking）。

环境设计：每步从所有 agent 的 ready 子任务中选最紧迫的 1 个，
策略决定其卸载目标，立即给奖励。
启发式：若 config 中存在 heuristic 字段则启用 top-k action masking。
"""

from __future__ import annotations

import argparse
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
from rl.heuristics import HeuristicScorer, RuleBasedHeuristic, StrongGreedyHeuristic
from rl.mappo.actor_critic import (
    CentralValueCritic,
    HeuristicActionConditionedActor,
    IndependentPPOActor,
    IndependentValueCritic,
    MAPPONodeActor,
    TPTDTOActor,
)
from rl.mappo.buffer import MAPPOBuffer
from rl.mappo.event_runner import EventPPORunner
from rl.mappo.trainer import MAPPOTrainer


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_event_masker(
    scorer,
    top_k: int,
    top_k_end: int | None = None,
    top_k_decay_episode: int = 0,
):

    def masker(env: EventDrivenOffloadEnv) -> np.ndarray:
        i = env._current_i
        j = env._current_j
        action_dim = env.action_dim
        scores = scorer.score_step(env, i, j)          # [action_dim]
        mask = np.zeros((1, action_dim), dtype=np.float32)
        if top_k_end is not None and top_k_decay_episode > 0:
            progress = min(1.0, env.current_episode / max(1, top_k_decay_episode))
            k_float = top_k + progress * (top_k_end - top_k)
            k = int(round(k_float))
        else:
            k = top_k
        k = max(1, min(k, action_dim))
        top_idx = np.argsort(scores)[-k:]
        mask[0, top_idx] = 1.0
        mask[0, 0] = 1.0
        return mask
    return masker


def build_scorer(heuristic_cfg: Dict[str, Any]):

    h_type = heuristic_cfg.get("type", "scoring")
    if h_type == "rule_based":
        return RuleBasedHeuristic(
            critical_ratio_thresh=float(heuristic_cfg.get("critical_ratio_thresh", 0.7)),
            slack_thresh=float(heuristic_cfg.get("slack_thresh", 0.25)),
            cloud_speedup_thresh=float(heuristic_cfg.get("cloud_speedup_thresh", 0.85)),
            small_compute_ratio=float(heuristic_cfg.get("small_compute_ratio", 0.5)),
            large_data_ratio=float(heuristic_cfg.get("large_data_ratio", 0.8)),
            local_competitive_ratio=float(heuristic_cfg.get("local_competitive_ratio", 1.05)),
            primary_bonus=float(heuristic_cfg.get("primary_bonus", 1.0)),
        )
    if h_type in {"strong_greedy", "oracle_greedy", "one_step_oracle"}:
        return StrongGreedyHeuristic(
            primary_bonus=float(heuristic_cfg.get("primary_bonus", 1.0)),
        )
    if h_type == "scoring":
        return HeuristicScorer(
            edge_alloc_hz=heuristic_cfg.get("edge_alloc_hz", 1e9),
            load_penalty=heuristic_cfg.get("load_penalty", 0.2),
            risk_penalty=heuristic_cfg.get("risk_penalty", 4.0),
            criticality_weight=heuristic_cfg.get("criticality_weight", 0.5),
            deadline_margin=heuristic_cfg.get("deadline_margin", 0.9),
            slack_penalty=heuristic_cfg.get("slack_penalty", 1.5),
            over_deadline_penalty=heuristic_cfg.get("over_deadline_penalty", 4.0),
            cloud_bias=heuristic_cfg.get("cloud_bias", 0.05),
            local_bias=heuristic_cfg.get("local_bias", 0.05),
            critical_path_weight=heuristic_cfg.get("critical_path_weight", 2.0),
            successor_data_weight=heuristic_cfg.get("successor_data_weight", 0.8),
            placement_consistency_weight=heuristic_cfg.get("placement_consistency_weight", 0.6),
            future_load_weight=heuristic_cfg.get("future_load_weight", 1.0),
            cloud_deadline_bonus=heuristic_cfg.get("cloud_deadline_bonus", 0.3),
        )
    raise ValueError(f"未知 heuristic.type: {h_type}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/event_ppo.yaml")
    parser.add_argument("--min_subtasks", type=int, default=None)
    parser.add_argument("--max_subtasks", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--actor_lr", type=float, default=None)
    parser.add_argument("--clip_ratio", type=float, default=None)
    parser.add_argument("--entropy_coef", type=float, default=None)
    parser.add_argument(
        "--objective_alpha",
        type=float,
        default=None,
        help="Delay preference weight; the energy weight is set to 1-alpha",
    )
    parser.add_argument(
        "--variant",
        choices=[
            "ippo",
            "mappo",
            "hmappo",
            "ablation_no_heur",
            "ablation_no_haca",
            "ablation_no_urgency",
        ],
        default=None,
        help="Train a standardized V3 PPO-family baseline without another config file",
    )
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    if args.variant == "ippo":
        cfg["env"]["heuristic_state_mode"] = "none"
        cfg["env"]["heuristic_mask"] = False
        cfg["model"]["actor_type"] = "independent"
        cfg["model"]["critic_type"] = "independent"
        cfg["output"]["algorithm_name"] = "event_ippo_paper_v3"
    elif args.variant == "mappo":
        cfg["env"]["heuristic_state_mode"] = "none"
        cfg["env"]["heuristic_mask"] = False
        cfg["model"]["actor_type"] = "mlp"
        cfg["model"]["critic_type"] = "central"
        cfg["output"]["algorithm_name"] = "event_ppo_baseline_paper_v3"
    elif args.variant == "ablation_no_heur":
        cfg["env"]["heuristic_state_mode"] = "zero_outcome"
        cfg["env"]["task_selection_mode"] = "urgency"
        cfg["env"]["heuristic_mask"] = False
        cfg["model"]["actor_type"] = "action_conditioned"
        cfg["output"]["algorithm_name"] = "event_ppo_ablation_no_heur_paper_v3"
    elif args.variant == "ablation_no_haca":
        cfg["env"]["heuristic_state_mode"] = "strong_outcome"
        cfg["env"]["task_selection_mode"] = "urgency"
        cfg["env"]["heuristic_mask"] = False
        cfg["model"]["actor_type"] = "mlp"
        cfg["output"]["algorithm_name"] = "event_ppo_ablation_no_haca_paper_v3"
    elif args.variant == "ablation_no_urgency":
        cfg["env"]["heuristic_state_mode"] = "strong_outcome"
        cfg["env"]["task_selection_mode"] = "round_robin"
        cfg["env"]["heuristic_mask"] = False
        cfg["model"]["actor_type"] = "action_conditioned"
        cfg["output"]["algorithm_name"] = "event_ppo_ablation_no_urgency_paper_v3"

    sweep_values = [
        ("actor_lr", args.actor_lr),
        ("clip_ratio", args.clip_ratio),
        ("entropy_coef", args.entropy_coef),
        ("objective_alpha", args.objective_alpha),
    ]
    active_sweeps = [(name, value) for name, value in sweep_values if value is not None]
    if len(active_sweeps) > 1:
        raise ValueError("Hyperparameter sensitivity uses one-factor-at-a-time; set only one override")
    if active_sweeps:
        if args.variant is not None:
            raise ValueError("Do not combine --variant with a hyperparameter sensitivity override")
        name, value = active_sweeps[0]
        if name == "actor_lr":
            if value <= 0:
                raise ValueError("actor_lr must be positive")
            cfg["train"]["actor_lr"] = float(value)
        elif name == "clip_ratio":
            if not 0.0 < value < 1.0:
                raise ValueError("clip_ratio must be between 0 and 1")
            cfg["train"]["clip_ratio"] = float(value)
        elif name == "entropy_coef":
            if value < 0:
                raise ValueError("entropy_coef must be non-negative")
            cfg["train"]["entropy_coef"] = float(value)
        else:
            if not 0.0 < value < 1.0:
                raise ValueError("objective_alpha must be between 0 and 1")
            cfg["env"]["objective_alpha"] = float(value)
            cfg["env"]["objective_beta"] = 1.0 - float(value)
        value_slug = format(float(value), ".8g").replace("-", "m").replace(".", "p")
        cfg["output"]["algorithm_name"] = f"event_ppo_hyper_{name}_{value_slug}_paper_v3"
    if args.min_subtasks is not None:
        cfg["env"]["min_subtasks"] = args.min_subtasks
    if args.max_subtasks is not None:
        cfg["env"]["max_subtasks"] = args.max_subtasks
    if args.device is not None:
        cfg["device"] = args.device
    set_seed(int(cfg["seed"]))
    device = torch.device(cfg.get("device", "cpu"))

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
        objective_alpha=env_cfg.get("objective_alpha"),
        objective_beta=env_cfg.get("objective_beta"),
    )

    # ---- 可选启发式 action masking ----
    heuristic_cfg = cfg.get("heuristic")
    heuristic_mask_enabled = bool(env_cfg.get("heuristic_mask", False))
    if heuristic_cfg and heuristic_mask_enabled:
        scorer = build_scorer(heuristic_cfg)
        top_k = int(heuristic_cfg.get("top_k", 3))
        top_k_end = heuristic_cfg.get("top_k_end")
        top_k_end = int(top_k_end) if top_k_end is not None else None
        top_k_decay_episode = int(heuristic_cfg.get("top_k_decay_episode", 0))
        env.heuristic_action_masker = make_event_masker(
            scorer,
            top_k,
            top_k_end=top_k_end,
            top_k_decay_episode=top_k_decay_episode,
        )
        print(
            f"启发式 action masking 已启用 "
            f"(type={heuristic_cfg.get('type', 'scoring')}, "
            f"top_k={top_k}, top_k_end={top_k_end}, decay_ep={top_k_decay_episode})"
        )
    else:
        print(
            "启发式 action masking 未启用"
            f"（heuristic_state_mode={env_cfg.get('heuristic_state_mode', 'none')}）"
        )

    reset_pack = env.reset()
    obs_dim    = int(reset_pack["obs"].shape[-1])
    action_dim = int(reset_pack["avail_actions"].shape[-1])
    state_dim  = env.state_dim

    print(f"obs_dim={obs_dim}, action_dim={action_dim}, state_dim={state_dim}")
    print(f"max_episode_steps (初始)={env.max_episode_steps}")

    actor_type = cfg["model"].get("actor_type", "mlp")
    actor_kwargs = {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "hidden_dim": cfg["model"]["actor_hidden_dim"],
    }
    if actor_type == "action_conditioned":
        actor = HeuristicActionConditionedActor(**actor_kwargs).to(device)
    elif actor_type == "independent":
        actor = IndependentPPOActor(
            **actor_kwargs, num_agents=int(env_cfg["num_agents"])
        ).to(device)
    elif actor_type == "tptdto":
        actor = TPTDTOActor(
            **actor_kwargs,
            num_agents=int(env_cfg["num_agents"]),
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

    if cfg["model"].get("critic_type", "central") == "independent":
        critic = IndependentValueCritic(
            state_dim=state_dim,
            hidden_dim=cfg["model"]["critic_hidden_dim"],
            num_agents=int(env_cfg["num_agents"]),
        ).to(device)
    else:
        critic = CentralValueCritic(
            state_dim=state_dim,
            hidden_dim=cfg["model"]["critic_hidden_dim"],
        ).to(device)

    # 保守初始化
    for name, p in actor.named_parameters():
        if "weight" in name and p.dim() >= 2:
            gain = 0.01 if any(k in name for k in ["policy_head", "action", "out"]) else 0.1
            torch.nn.init.orthogonal_(p, gain=gain)
    for name, p in critic.named_parameters():
        if "weight" in name and p.dim() >= 2:
            torch.nn.init.orthogonal_(p, gain=0.1)

    actor_optimizer  = torch.optim.Adam(actor.parameters(),  lr=cfg["train"]["actor_lr"])
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg["train"]["critic_lr"])

    # LR 预热
    warmup_updates = 50
    actor_scheduler  = torch.optim.lr_scheduler.LambdaLR(
        actor_optimizer,  lambda u: min(1.0, (u + 1) / warmup_updates)
    )
    critic_scheduler = torch.optim.lr_scheduler.LambdaLR(
        critic_optimizer, lambda u: min(1.0, (u + 1) / warmup_updates)
    )

    trainer = MAPPOTrainer(
        actor=actor,
        critic=critic,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        clip_ratio=cfg["train"]["clip_ratio"],
        entropy_coef=cfg["train"]["entropy_coef"],
        value_coef=cfg["train"]["value_coef"],
        max_grad_norm=cfg["train"]["max_grad_norm"],
        ppo_epochs=cfg["train"]["ppo_epochs"],
        mini_batch_size=cfg["train"]["mini_batch_size"],
        actor_scheduler=actor_scheduler,
        critic_scheduler=critic_scheduler,
    )
    buffer = MAPPOBuffer(
        gamma=cfg["train"]["gamma"],
        gae_lambda=cfg["train"]["gae_lambda"],
        independent_agents=args.variant == "ippo",
    )

    min_tasks = env_cfg.get("min_subtasks", "any")
    max_tasks = env_cfg.get("max_subtasks", "any")
    run_name = f"{cfg['output']['algorithm_name']}_tasks{min_tasks}-{max_tasks}"
    output_dir = Path(cfg["output"]["root_dir"]) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    print(f"输出目录: {output_dir}")

    train_cfg = cfg["train"]
    runner = EventPPORunner(
        env=env,
        actor=actor,
        critic=critic,
        trainer=trainer,
        buffer=buffer,
        device=device,
        total_episodes=train_cfg["total_episodes"],
        episodes_per_update=train_cfg["episodes_per_update"],
        log_interval=train_cfg["log_interval"],
        save_interval=train_cfg.get("save_interval", 500),
        entropy_coef_start=float(train_cfg.get("entropy_coef", 0.01)),
        entropy_coef_end=float(train_cfg.get("entropy_coef_end", train_cfg.get("entropy_coef", 0.01))),
        entropy_decay_episodes=int(train_cfg.get("entropy_decay_episodes", train_cfg["total_episodes"])),
        csv_path=output_dir / f"metrics_{cfg['output']['algorithm_name']}.csv",
        checkpoint_dir=output_dir,
        save_best_by=cfg["output"].get("save_best_by", "ep_on_time_ratio"),
    )

    runner.train()

    torch.save(actor.state_dict(),  output_dir / "actor_final.pt")
    torch.save(critic.state_dict(), output_dir / "critic_final.pt")


if __name__ == "__main__":
    main()

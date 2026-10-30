"""
事件驱动任务卸载训练运行器。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from rl.envs.event_env import EventDrivenOffloadEnv
from rl.mappo.actor_critic import CentralValueCritic, MAPPONodeActor
from rl.mappo.buffer import MAPPOBuffer
from rl.mappo.trainer import MAPPOTrainer


@dataclass
class EventPPORunner:
    env: EventDrivenOffloadEnv
    actor: MAPPONodeActor
    critic: CentralValueCritic
    trainer: MAPPOTrainer
    buffer: MAPPOBuffer
    device: torch.device

    # 训练超参
    total_episodes: int = 4000
    episodes_per_update: int = 16
    log_interval: int = 10
    save_interval: int = 500
    entropy_coef_start: Optional[float] = None
    entropy_coef_end: Optional[float] = None
    entropy_decay_episodes: Optional[int] = None

    # 输出
    csv_path: Optional[Path] = None
    checkpoint_dir: Optional[Path] = None
    save_best_by: str = "ep_on_time_ratio"

    # 运行时
    _best_metric: float = field(default=float("-inf"), init=False)
    _csv_file: object = field(default=None, init=False)
    _csv_writer: object = field(default=None, init=False)
    _last_train_metrics: Dict = field(default_factory=lambda: {
        "actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0,
        "actor_grad_norm": 0.0, "critic_grad_norm": 0.0,
        "actor_lr": 0.0, "critic_lr": 0.0, "entropy_coef": 0.0,
    }, init=False)

    # ------------------------------------------------------------------ #

    def _save_checkpoint(self, tag: str) -> None:
        if self.checkpoint_dir is None:
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(),  self.checkpoint_dir / f"actor_{tag}.pt")
        torch.save(self.critic.state_dict(), self.checkpoint_dir / f"critic_{tag}.pt")

    def _open_csv(self) -> None:
        if self.csv_path is None:
            return
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_file = self.csv_path.open("w", encoding="utf-8", newline="")

    def _write_csv(self, row: Dict) -> None:
        if self._csv_file is None:
            return
        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=list(row.keys()))
            self._csv_writer.writeheader()
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def _close_csv(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None

    # ------------------------------------------------------------------ #
    #  单 episode
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def run_episode(self) -> Dict:
        pack = self.env.reset()
        ep_rewards: List[float] = []
        ep_steps = 0

        while True:
            current_agent = int(self.env._current_i)
            obs_t    = torch.as_tensor(pack["obs"],           dtype=torch.float32, device=self.device)
            state_t  = torch.as_tensor(pack["state"],         dtype=torch.float32, device=self.device).unsqueeze(0)
            avail_t  = torch.as_tensor(pack["avail_actions"], dtype=torch.float32, device=self.device)
            active_t = torch.as_tensor(pack["active_mask"],   dtype=torch.float32, device=self.device)

            # actor 推断（N=1 shaped）
            act_out      = self.actor.sample_actions(obs_t, avail_t)
            actions_np   = act_out["actions"].cpu().numpy()       # [1]
            log_probs_np = act_out["log_probs"].cpu().numpy()     # [1]
            # critic 估值
            value_scalar = self.critic(state_t).item()
            values_np    = np.array([value_scalar], dtype=np.float32)

            pack_next = self.env.step(actions_np)

            rewards_np = pack_next["rewards"]
            dones_np   = pack_next["dones"]
            ep_rewards.append(float(rewards_np.mean()))
            ep_steps += 1

            self.buffer.add(
                obs          = pack["obs"],
                state        = pack["state"],
                avail_actions= pack["avail_actions"],
                active_mask  = pack["active_mask"],
                actions      = actions_np,
                log_probs    = log_probs_np,
                rewards      = rewards_np,
                dones        = dones_np,
                values       = values_np,
                agent_ids    = np.array([current_agent], dtype=np.int64),
            )

            if float(dones_np.max()) > 0.5:
                info = pack_next["infos"]
                break
            pack = pack_next

        ep_return   = float(np.sum(ep_rewards))
        ep_mean_r   = float(np.mean(ep_rewards))
        terminal    = pack_next["infos"]
        on_time     = float(terminal.get("agent_on_time_ratio", 0.0))
        c1_excess   = float(terminal.get("c1_excess", 0.0))
        ep_feasible = float(terminal.get("ep_feasible", 0.0))

        return {
            "ep_return":      ep_return,
            "ep_mean_reward": ep_mean_r,
            "ep_on_time_ratio": on_time,
            "c1_excess":      c1_excess,
            "ep_feasible":    ep_feasible,
            "ep_steps":       ep_steps,
        }

    # ------------------------------------------------------------------ #
    #  训练主循环
    # ------------------------------------------------------------------ #

    def train(self) -> None:
        self._open_csv()
        self.env.set_episode_index(1)
        update_count = 0
        batch_metrics: List[Dict] = []

        for ep in range(1, self.total_episodes + 1):
            self.env.set_episode_index(ep)
            ep_metrics = self.run_episode()
            batch_metrics.append(ep_metrics)

            csv_row = {"episode": ep}
            csv_row.update(ep_metrics)
            csv_row.update(self._last_train_metrics)
            self._write_csv(csv_row)

            # ---- PPO 更新 ----
            if ep % self.episodes_per_update == 0:
                update_count += 1
                if self.entropy_coef_start is not None and self.entropy_coef_end is not None:
                    decay_episodes = self.entropy_decay_episodes or self.total_episodes
                    progress = min(1.0, ep / max(1, decay_episodes))
                    self.trainer.entropy_coef = (
                        self.entropy_coef_start
                        + progress * (self.entropy_coef_end - self.entropy_coef_start)
                    )
                batch_np = self.buffer.compute_returns_and_advantages()
                batch_t  = self.buffer.to_torch(batch_np, self.device)
                train_metrics = self.trainer.update(batch_t)
                train_metrics["entropy_coef"] = float(self.trainer.entropy_coef)
                self._last_train_metrics = train_metrics
                self.buffer.clear()

                batch_avg = {
                    key: float(np.mean([m[key] for m in batch_metrics]))
                    for key in batch_metrics[0]
                }

                # ---- 控制台打印（每次更新的均值）----
                if update_count % self.log_interval == 0:
                    avg = lambda k: batch_avg[k]
                    print(
                        f"[Ep {ep:05d} | Upd {update_count:04d}] "
                        f"ret={avg('ep_return'):+.2f} "
                        f"r_step={avg('ep_mean_reward'):+.4f} "
                        f"on_time={avg('ep_on_time_ratio'):.3f} "
                        f"c1x={avg('c1_excess'):.3f} "
                        f"steps={avg('ep_steps'):.0f}(avg) "
                        f"| actor={train_metrics.get('actor_loss',0):.4f} "
                        f"critic={train_metrics.get('critic_loss',0):.4f} "
                        f"ent={train_metrics.get('entropy',0):.3f} "
                        f"ecoef={train_metrics.get('entropy_coef',0):.4f} "
                        f"gnorm={train_metrics.get('actor_grad_norm',0):.2f}"
                    )

                # ---- 保存最优 checkpoint ----
                metric_val = float(batch_avg.get(self.save_best_by, 0.0))
                if metric_val > self._best_metric:
                    self._best_metric = metric_val
                    self._save_checkpoint("best")
                batch_metrics.clear()

            # ---- 周期性保存 ----
            if ep % self.save_interval == 0:
                self._save_checkpoint(f"ep{ep:05d}")

        self._save_checkpoint("final")
        self._close_csv()
        print(f"训练完成，输出目录: {self.checkpoint_dir}")

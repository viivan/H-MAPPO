"""
MAPPO PPO 更新器。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn

from rl.mappo.actor_critic import CentralValueCritic, MAPPONodeActor


@dataclass
class MAPPOTrainer:
    actor: MAPPONodeActor
    critic: CentralValueCritic
    actor_optimizer: torch.optim.Optimizer
    critic_optimizer: torch.optim.Optimizer
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    mini_batch_size: int = 64
    actor_scheduler: object = None   # 可选 LR scheduler
    critic_scheduler: object = None  # 可选 LR scheduler

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        batch keys (所有 tensor 的前两维均为 [T, N])：
          obs, state, avail_actions, active_mask,
          actions, old_log_probs, values, returns, advantages
        """
        T, N = batch["advantages"].shape

        # ---- 展平 [T, N] -> [T*N] ----
        obs = batch["obs"].reshape(T * N, -1)
        avail_actions = batch["avail_actions"].reshape(T * N, -1)
        actions = batch["actions"].reshape(T * N)
        old_log_probs = batch["old_log_probs"].reshape(T * N)
        returns = batch["returns"].reshape(T * N)
        advantages = batch["advantages"].reshape(T * N)
        active_mask = batch["active_mask"].reshape(T * N)  # 1=有效, 0=无效

        # state: [T, state_dim] -> repeat N 次 -> [T*N, state_dim]
        state = batch["state"].unsqueeze(1).repeat(1, N, 1).reshape(T * N, -1)

        # ---- 标准化 advantage（只在有效样本上计算均值/方差）----
        valid_adv = advantages[active_mask > 0]
        if valid_adv.numel() > 1:
            advantages = (advantages - valid_adv.mean()) / (valid_adv.std() + 1e-8)

        # ---- mini-batch PPO 更新 ----
        total_samples = T * N
        indices = torch.randperm(total_samples, device=obs.device)

        metrics = {
            "actor_loss": 0.0, "critic_loss": 0.0,
            "entropy": 0.0, "actor_grad_norm": 0.0, "critic_grad_norm": 0.0,
        }
        num_updates = 0

        for _ in range(self.ppo_epochs):
            for start in range(0, total_samples, self.mini_batch_size):
                idx = indices[start: start + self.mini_batch_size]
                mask_b = active_mask[idx]

                if mask_b.sum() < 1:
                    continue

                new_log_probs, entropy = self.actor.evaluate_actions(
                    obs[idx], avail_actions[idx], actions[idx]
                )

                ratio = torch.exp(new_log_probs - old_log_probs[idx])
                adv_b = advantages[idx]
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_b

                actor_loss  = -(torch.min(surr1, surr2) * mask_b).sum() / mask_b.sum()
                entropy_loss = -(entropy * mask_b).sum() / mask_b.sum()

                values_pred = self.critic(state[idx])
                critic_loss = (nn.functional.mse_loss(values_pred, returns[idx], reduction="none") * mask_b).sum() / mask_b.sum()

                # ---- actor update----
                self.actor_optimizer.zero_grad()
                (actor_loss + self.entropy_coef * entropy_loss).backward()
                actor_gnorm = float(nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm))
                self.actor_optimizer.step()
                if self.actor_scheduler is not None:
                    self.actor_scheduler.step()

                # ---- critic update ----
                self.critic_optimizer.zero_grad()
                (self.value_coef * critic_loss).backward()
                critic_gnorm = float(nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm))
                self.critic_optimizer.step()
                if self.critic_scheduler is not None:
                    self.critic_scheduler.step()

                metrics["actor_loss"]       += float(actor_loss.item())
                metrics["critic_loss"]      += float(critic_loss.item())
                metrics["entropy"]          += float((entropy * mask_b).sum().item() / mask_b.sum().item())
                metrics["actor_grad_norm"]  += actor_gnorm
                metrics["critic_grad_norm"] += critic_gnorm
                num_updates += 1

        if num_updates > 0:
            for key in metrics:
                metrics[key] /= num_updates

        # 记录当前学习率
        metrics["actor_lr"]  = float(self.actor_optimizer.param_groups[0]["lr"])
        metrics["critic_lr"] = float(self.critic_optimizer.param_groups[0]["lr"])
        return metrics

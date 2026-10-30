from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000) -> None:
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        avail: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        next_avail: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, avail, np.asarray(action, dtype=np.float32), reward, next_state, next_avail, done))

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        idx = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        rows = [self.buffer[i] for i in idx]
        states, avails, actions, rewards, next_states, next_avails, dones = zip(*rows)
        return {
            "states": np.asarray(states, dtype=np.float32),
            "avails": np.asarray(avails, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.float32),
            "rewards": np.asarray(rewards, dtype=np.float32).reshape(-1, 1),
            "next_states": np.asarray(next_states, dtype=np.float32),
            "next_avails": np.asarray(next_avails, dtype=np.float32),
            "dones": np.asarray(dones, dtype=np.float32).reshape(-1, 1),
        }

    def __len__(self) -> int:
        return len(self.buffer)


class BidActor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state: torch.Tensor, action_vec: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action_vec], dim=-1))


class DiscreteQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


@dataclass
class OffPolicyMetrics:
    actor_loss: float = 0.0
    critic_loss: float = 0.0
    actor_grad_norm: float = 0.0
    critic_grad_norm: float = 0.0
    actor_lr: float = 0.0
    critic_lr: float = 0.0


class DiscreteMADDPGAgent:
    """
    事件驱动离散动作 MADDPG。

    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.01,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau

        self.action_ids_np = np.eye(action_dim, dtype=np.float32)
        self.action_ids_torch = torch.eye(action_dim, device=self.device)
        actor_input_dim = state_dim + action_dim
        self.actors = nn.ModuleList([
            BidActor(actor_input_dim, hidden_dim).to(self.device)
            for _ in range(action_dim)
        ])
        self.actors_target = copy.deepcopy(self.actors)
        self.critic = QCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=actor_lr)
            for actor in self.actors
        ]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    @torch.no_grad()
    def select_action(self, state: np.ndarray, avail: np.ndarray, epsilon: float) -> Tuple[int, np.ndarray]:
        valid = np.flatnonzero(avail > 0)
        if len(valid) == 0:
            return 0, np.zeros(self.action_dim, dtype=np.float32)
        bids = self._actor_bids_np(state)
        noise = np.random.normal(loc=0.0, scale=epsilon, size=self.action_dim).astype(np.float32)
        noisy_bids = bids + noise
        noisy_bids[avail <= 0] = -1e9
        if np.random.rand() < epsilon:
            action = int(np.random.choice(valid))
        else:
            action = int(np.argmax(noisy_bids))
        # critic 存储实际参与竞争的 bid 向量；不可行动作置为 -1，避免伪可选。
        action_vec = bids.copy()
        action_vec[avail <= 0] = -1.0
        return action, action_vec.astype(np.float32)

    def _actor_bids(self, states: torch.Tensor, actors: nn.ModuleList | None = None) -> torch.Tensor:
        actors = actors or self.actors
        batch_n = states.size(0)
        action_ids = self.action_ids_torch.unsqueeze(0).expand(batch_n, -1, -1)
        states_rep = states.unsqueeze(1).expand(-1, self.action_dim, -1)
        inputs = torch.cat([states_rep, action_ids], dim=-1)
        bids = [actor(inputs[:, a, :]) for a, actor in enumerate(actors)]
        return torch.cat(bids, dim=-1)

    @torch.no_grad()
    def _actor_bids_np(self, state: np.ndarray) -> np.ndarray:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self._actor_bids(s).squeeze(0).cpu().numpy().astype(np.float32)

    def update(self, replay: ReplayBuffer, batch_size: int, max_grad_norm: float = 1.0) -> OffPolicyMetrics:
        if len(replay) < batch_size:
            return OffPolicyMetrics(
                actor_lr=float(self.actor_optimizers[0].param_groups[0]["lr"]),
                critic_lr=float(self.critic_optimizer.param_groups[0]["lr"]),
            )
        batch = replay.sample(batch_size)
        states = torch.as_tensor(batch["states"], dtype=torch.float32, device=self.device)
        avails = torch.as_tensor(batch["avails"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(batch["next_states"], dtype=torch.float32, device=self.device)
        next_avails = torch.as_tensor(batch["next_avails"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            next_actions = self._actor_bids(next_states, self.actors_target)
            next_actions = torch.where(next_avails > 0, next_actions, torch.full_like(next_actions, -1.0))
            target_q = rewards + (1.0 - dones) * self.gamma * self.critic_target(next_states, next_actions)

        current_q = self.critic(states, actions)
        critic_loss = nn.functional.mse_loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
        self.critic_optimizer.step()

        total_actor_loss = 0.0
        total_actor_grad = 0.0
        for i, (actor, opt) in enumerate(zip(self.actors, self.actor_optimizers)):
            bids = []
            batch_n = states.size(0)
            action_ids = self.action_ids_torch.unsqueeze(0).expand(batch_n, -1, -1)
            states_rep = states.unsqueeze(1).expand(-1, self.action_dim, -1)
            actor_inputs = torch.cat([states_rep, action_ids], dim=-1)
            for j, other_actor in enumerate(self.actors):
                b = other_actor(actor_inputs[:, j, :])
                bids.append(b if i == j else b.detach())
            joint_bids = torch.cat(bids, dim=-1)
            joint_bids = torch.where(avails > 0, joint_bids, torch.full_like(joint_bids, -1.0))
            actor_loss = -self.critic(states, joint_bids).mean()
            opt.zero_grad()
            actor_loss.backward()
            actor_grad = torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            opt.step()
            total_actor_loss += float(actor_loss.item())
            total_actor_grad += float(actor_grad)

        self._soft_update(self.actors, self.actors_target)
        self._soft_update(self.critic, self.critic_target)

        return OffPolicyMetrics(
            actor_loss=total_actor_loss / self.action_dim,
            critic_loss=float(critic_loss.item()),
            actor_grad_norm=total_actor_grad / self.action_dim,
            critic_grad_norm=float(critic_grad),
            actor_lr=float(self.actor_optimizers[0].param_groups[0]["lr"]),
            critic_lr=float(self.critic_optimizer.param_groups[0]["lr"]),
        )

    def _soft_update(self, src: nn.Module, tgt: nn.Module) -> None:
        for p, tp in zip(src.parameters(), tgt.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    def state_dict(self) -> Dict:
        return {
            "actors": [actor.state_dict() for actor in self.actors],
            "actors_target": [actor.state_dict() for actor in self.actors_target],
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
        }


class DiscreteMATD3Agent(DiscreteMADDPGAgent):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.01,
        policy_noise: float = 0.1,
        noise_clip: float = 0.2,
        policy_update_freq: int = 2,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(state_dim, action_dim, hidden_dim, actor_lr, critic_lr, gamma, tau, device)
        self.critic_2 = QCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_2_target = copy.deepcopy(self.critic_2)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_update_freq = max(1, int(policy_update_freq))
        self.update_count = 0

    def update(self, replay: ReplayBuffer, batch_size: int, max_grad_norm: float = 1.0) -> OffPolicyMetrics:
        if len(replay) < batch_size:
            return OffPolicyMetrics(
                actor_lr=float(self.actor_optimizers[0].param_groups[0]["lr"]),
                critic_lr=float(self.critic_optimizer.param_groups[0]["lr"]),
            )
        self.update_count += 1
        batch = replay.sample(batch_size)
        states = torch.as_tensor(batch["states"], dtype=torch.float32, device=self.device)
        avails = torch.as_tensor(batch["avails"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(batch["next_states"], dtype=torch.float32, device=self.device)
        next_avails = torch.as_tensor(batch["next_avails"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_actions = self._actor_bids(next_states, self.actors_target)
            noise = torch.randn_like(next_actions).mul(self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (next_actions + noise).clamp(-1.0, 1.0)
            next_actions = torch.where(next_avails > 0, next_actions, torch.full_like(next_actions, -1.0))
            target_q1 = self.critic_target(next_states, next_actions)
            target_q2 = self.critic_2_target(next_states, next_actions)
            target_q = rewards + (1.0 - dones) * self.gamma * torch.minimum(target_q1, target_q2)

        q1 = self.critic(states, actions)
        q2 = self.critic_2(states, actions)
        critic_1_loss = nn.functional.mse_loss(q1, target_q)
        critic_2_loss = nn.functional.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_1_loss.backward()
        critic_1_grad = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
        self.critic_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        critic_2_grad = torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), max_grad_norm)
        self.critic_2_optimizer.step()

        actor_loss_value = 0.0
        actor_grad_value = 0.0
        if self.update_count % self.policy_update_freq == 0:
            total_actor_loss = 0.0
            total_actor_grad = 0.0
            batch_n = states.size(0)
            action_ids = self.action_ids_torch.unsqueeze(0).expand(batch_n, -1, -1)
            states_rep = states.unsqueeze(1).expand(-1, self.action_dim, -1)
            actor_inputs = torch.cat([states_rep, action_ids], dim=-1)
            for i, (actor, opt) in enumerate(zip(self.actors, self.actor_optimizers)):
                bids = []
                for j, other_actor in enumerate(self.actors):
                    b = other_actor(actor_inputs[:, j, :])
                    bids.append(b if i == j else b.detach())
                joint_bids = torch.cat(bids, dim=-1)
                joint_bids = torch.where(avails > 0, joint_bids, torch.full_like(joint_bids, -1.0))
                actor_loss = -self.critic(states, joint_bids).mean()
                opt.zero_grad()
                actor_loss.backward()
                actor_grad = torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                opt.step()
                total_actor_loss += float(actor_loss.item())
                total_actor_grad += float(actor_grad)
            actor_loss_value = total_actor_loss / self.action_dim
            actor_grad_value = total_actor_grad / self.action_dim
            self._soft_update(self.actors, self.actors_target)
            self._soft_update(self.critic, self.critic_target)
            self._soft_update(self.critic_2, self.critic_2_target)

        return OffPolicyMetrics(
            actor_loss=actor_loss_value,
            critic_loss=float(0.5 * (critic_1_loss.item() + critic_2_loss.item())),
            actor_grad_norm=actor_grad_value,
            critic_grad_norm=float(0.5 * (critic_1_grad + critic_2_grad)),
            actor_lr=float(self.actor_optimizers[0].param_groups[0]["lr"]),
            critic_lr=float(self.critic_optimizer.param_groups[0]["lr"]),
        )

    def state_dict(self) -> Dict:
        state = super().state_dict()
        state.update({
            "critic_2": self.critic_2.state_dict(),
            "critic_2_target": self.critic_2_target.state_dict(),
        })
        return state


class DDQNAgent:

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        target_update_interval: int = 500,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.action_dim = action_dim
        self.gamma = gamma
        self.target_update_interval = max(1, int(target_update_interval))
        self.update_count = 0
        self.q_network = DiscreteQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = copy.deepcopy(self.q_network).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)

    @torch.no_grad()
    def select_action(self, state: np.ndarray, avail: np.ndarray, epsilon: float) -> Tuple[int, np.ndarray]:
        valid = np.flatnonzero(avail > 0)
        if len(valid) == 0:
            return 0, np.asarray(0, dtype=np.int64)
        if np.random.rand() < epsilon:
            action = int(np.random.choice(valid))
        else:
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_network(state_t).squeeze(0)
            avail_t = torch.as_tensor(avail > 0, dtype=torch.bool, device=self.device)
            q_values = q_values.masked_fill(~avail_t, -1e9)
            action = int(torch.argmax(q_values).item())
        return action, np.asarray(action, dtype=np.int64)

    def update(self, replay: ReplayBuffer, batch_size: int, max_grad_norm: float = 1.0) -> OffPolicyMetrics:
        if len(replay) < batch_size:
            lr = float(self.optimizer.param_groups[0]["lr"])
            return OffPolicyMetrics(critic_lr=lr)
        self.update_count += 1
        batch = replay.sample(batch_size)
        states = torch.as_tensor(batch["states"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=self.device).reshape(-1, 1)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(batch["next_states"], dtype=torch.float32, device=self.device)
        next_avails = torch.as_tensor(batch["next_avails"], dtype=torch.bool, device=self.device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)

        current_q = self.q_network(states).gather(1, actions)
        with torch.no_grad():
            online_next_q = self.q_network(next_states).masked_fill(~next_avails, -1e9)
            next_actions = torch.argmax(online_next_q, dim=1, keepdim=True)
            target_next_q = self.target_network(next_states).gather(1, next_actions)
            target_q = rewards + (1.0 - dones) * self.gamma * target_next_q

        loss = nn.functional.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_grad_norm)
        self.optimizer.step()
        if self.update_count % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        return OffPolicyMetrics(
            critic_loss=float(loss.item()),
            critic_grad_norm=float(grad_norm),
            critic_lr=float(self.optimizer.param_groups[0]["lr"]),
        )

    def state_dict(self) -> Dict:
        return {
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "update_count": self.update_count,
        }

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch


@dataclass
class MAPPOBuffer:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    independent_agents: bool = False

    _obs: List[np.ndarray] = field(default_factory=list)
    _state: List[np.ndarray] = field(default_factory=list)
    _avail_actions: List[np.ndarray] = field(default_factory=list)
    _active_mask: List[np.ndarray] = field(default_factory=list)
    _actions: List[np.ndarray] = field(default_factory=list)
    _log_probs: List[np.ndarray] = field(default_factory=list)
    _rewards: List[np.ndarray] = field(default_factory=list)
    _dones: List[np.ndarray] = field(default_factory=list)
    _values: List[np.ndarray] = field(default_factory=list)
    _agent_ids: List[np.ndarray] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self._rewards)

    def clear(self) -> None:
        self._obs.clear()
        self._state.clear()
        self._avail_actions.clear()
        self._active_mask.clear()
        self._actions.clear()
        self._log_probs.clear()
        self._rewards.clear()
        self._dones.clear()
        self._values.clear()
        self._agent_ids.clear()

    def add(
        self,
        obs: np.ndarray,
        state: np.ndarray,
        avail_actions: np.ndarray,
        active_mask: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        agent_ids: np.ndarray | None = None,
    ) -> None:
        self._obs.append(np.asarray(obs, dtype=np.float32))
        self._state.append(np.asarray(state, dtype=np.float32))
        self._avail_actions.append(np.asarray(avail_actions, dtype=np.float32))
        self._active_mask.append(np.asarray(active_mask, dtype=np.float32))
        self._actions.append(np.asarray(actions, dtype=np.int64))
        self._log_probs.append(np.asarray(log_probs, dtype=np.float32))
        self._rewards.append(np.asarray(rewards, dtype=np.float32))
        self._dones.append(np.asarray(dones, dtype=np.float32))
        self._values.append(np.asarray(values, dtype=np.float32))
        if agent_ids is None:
            agent_ids = np.zeros_like(values, dtype=np.int64)
        self._agent_ids.append(np.asarray(agent_ids, dtype=np.int64))

    def compute_returns_and_advantages(self) -> Dict[str, np.ndarray]:

        rewards     = np.stack(self._rewards,     axis=0)   # [T, N]
        dones       = np.stack(self._dones,       axis=0)
        values_raw  = np.stack(self._values,      axis=0)
        active_mask = np.stack(self._active_mask, axis=0)   # [T, N]

        T, N = rewards.shape
        advantages_raw = np.zeros_like(rewards, dtype=np.float32)
        returns_raw    = np.zeros_like(rewards, dtype=np.float32)

        if self.independent_agents:
            agent_ids = np.stack(self._agent_ids, axis=0)
            next_adv_by_agent: Dict[int, float] = {}
            next_value_by_agent: Dict[int, float] = {}
            for t in reversed(range(T)):
                for n in range(N):
                    agent_id = int(agent_ids[t, n])
                    mask_t = 1.0 - float(dones[t, n])
                    next_adv = next_adv_by_agent.get(agent_id, 0.0)
                    next_value = next_value_by_agent.get(agent_id, 0.0)
                    delta = rewards[t, n] + self.gamma * next_value * mask_t - values_raw[t, n]
                    gae = delta + self.gamma * self.gae_lambda * mask_t * next_adv
                    advantages_raw[t, n] = gae
                    returns_raw[t, n] = gae + values_raw[t, n]
                    next_adv_by_agent[agent_id] = float(gae)
                    next_value_by_agent[agent_id] = float(values_raw[t, n])
        else:
            next_adv = np.zeros(N, dtype=np.float32)
            next_value = np.zeros(N, dtype=np.float32)
            for t in reversed(range(T)):
                mask_t = 1.0 - dones[t]
                delta = rewards[t] + self.gamma * next_value * mask_t - values_raw[t]
                gae = delta + self.gamma * self.gae_lambda * mask_t * next_adv
                advantages_raw[t] = gae
                returns_raw[t] = gae + values_raw[t]
                next_adv = gae
                next_value = values_raw[t]

        # ---- critic target 归一化----
        valid = active_mask > 0
        if valid.any() and returns_raw[valid].std() > 1e-6:
            r_mean = float(returns_raw[valid].mean())
            r_std  = float(returns_raw[valid].std()) + 1e-8
        else:
            r_mean, r_std = 0.0, 1.0
        returns_norm = (returns_raw - r_mean) / r_std   # critic 的归一化 target

        # advantages 直接原始值交给 trainer 做标准化
        return {
            "obs":           np.stack(self._obs,           axis=0),
            "state":         np.stack(self._state,         axis=0),
            "avail_actions": np.stack(self._avail_actions, axis=0),
            "active_mask":   active_mask,
            "actions":       np.stack(self._actions,       axis=0),
            "old_log_probs": np.stack(self._log_probs,     axis=0),
            "values":        values_raw,     # trainer 不用 values 做 loss，只用 returns
            "returns":       returns_norm,   # 归一化 critic target
            "advantages":    advantages_raw, # 原始 GAE，trainer 内部再标准化
        }

    @staticmethod
    def to_torch(batch: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for key, arr in batch.items():
            dtype = torch.long if key == "actions" else torch.float32
            out[key] = torch.as_tensor(arr, dtype=dtype, device=device)
        return out

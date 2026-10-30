from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rl.envs.event_env import EventDrivenOffloadEnv


@dataclass
class StrongGreedyHeuristic:
    primary_bonus: float = 1.0

    def score_step(
        self,
        env: "EventDrivenOffloadEnv",
        i: int,
        j: int,
    ) -> np.ndarray:
        alpha = float(env.env.alpha_w)
        beta = float(env.env.beta_w)
        costs = []
        for action in range(env.action_dim):
            outcome = env.estimate_action_outcome(i, j, action)
            costs.append(alpha * outcome["finish"] + beta * outcome["energy"])

        costs_arr = np.nan_to_num(
            np.asarray(costs, dtype=np.float64),
            nan=1e9,
            posinf=1e9,
            neginf=1e9,
        )
        mx = float(costs_arr.max())
        mn = float(costs_arr.min())
        denom = max(mx - mn, 1e-6)
        scores = ((mx - costs_arr) / denom).astype(np.float32)
        scores[int(np.argmin(costs_arr))] += float(self.primary_bonus)
        return scores

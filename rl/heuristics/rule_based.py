from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, Tuple

import numpy as np

if TYPE_CHECKING:
    from rl.envs.event_env import EventDrivenOffloadEnv


@dataclass
class RuleBasedHeuristic:
    """规则启发式打分器。"""
    critical_ratio_thresh: float = 0.7
    slack_thresh: float = 0.25
    cloud_speedup_thresh: float = 0.85
    small_compute_ratio: float = 0.5
    large_data_ratio: float = 0.8
    local_competitive_ratio: float = 1.05
    primary_bonus: float = 1.0

    # ------------------------------------------------------------------ #
    #  对外主接口
    # ------------------------------------------------------------------ #

    def score_step(
        self,
        env: "EventDrivenOffloadEnv",
        i: int,
        j: int,
    ) -> np.ndarray:

        action_dim = env.action_dim
        outcomes = [env.estimate_action_outcome(i, j, a) for a in range(action_dim)]
        alpha = float(env.env.alpha_w)
        beta = float(env.env.beta_w)
        costs = np.array(
            [alpha * o["finish"] + beta * o["energy"] for o in outcomes],
            dtype=np.float64,
        )
        costs = np.nan_to_num(costs, nan=1e9, posinf=1e9, neginf=1e9)
        primary = self._decide_primary(env, i, j, outcomes, costs)
        mx, mn = float(costs.max()), float(costs.min())
        denom = max(mx - mn, 1e-6)
        scores = (mx - costs) / denom
        scores = scores.astype(np.float32)
        scores[primary] += self.primary_bonus
        return scores

    def _decide_primary(
        self,
        env: "EventDrivenOffloadEnv",
        i: int,
        j: int,
        outcomes,
        costs: np.ndarray,
    ) -> int:
        dag = env.env.dag_tasks[i]
        st = dag.subtasks[j]
        t_max = max(env.env.T_max[i], 1e-9)

        cloud_action = env.action_dim - 1
        edge_actions = list(range(1, env.M + 1))

        # 1) 三类候选直接选 cost 最低的
        local_cost = float(costs[0])
        best_edge = int(min(edge_actions, key=lambda a: costs[a]))
        edge_cost = float(costs[best_edge])
        cloud_cost = float(costs[cloud_action])

        candidates = [(local_cost, 0), (edge_cost, best_edge), (cloud_cost, cloud_action)]
        primary = int(min(candidates, key=lambda x: x[0])[1])

        # 2) 关键节点 + deadline 紧 -> 强制选 finish 最短
        critical_ratio = self._critical_ratio(dag, j)
        edge_finish = float(outcomes[best_edge]["finish"])
        cloud_finish = float(outcomes[cloud_action]["finish"])
        slack_after_edge = (t_max - edge_finish) / t_max

        if critical_ratio >= self.critical_ratio_thresh and slack_after_edge < self.slack_thresh:
            if cloud_finish < edge_finish * self.cloud_speedup_thresh:
                primary = cloud_action
            else:
                primary = best_edge

        # 3) 小计算 + 大数据 -> 回落本地
        c_mean = float(np.mean([s.c for s in dag.subtasks.values()]))
        d_mean = float(np.mean([s.d for s in dag.subtasks.values()]))
        small_compute = st.c < self.small_compute_ratio * c_mean
        large_data = st.d > self.large_data_ratio * d_mean
        local_competitive = local_cost <= self.local_competitive_ratio * min(edge_cost, cloud_cost)
        if small_compute and large_data and local_competitive:
            primary = 0

        return int(primary)

    @staticmethod
    def _critical_ratio(dag, node: int) -> float:
        @lru_cache(maxsize=None)
        def depth(n: int) -> int:
            succs = dag.subtasks[n].successors
            if not succs:
                return 1
            return 1 + max(depth(s) for s in succs)

        max_depth = max(depth(n) for n in dag.subtasks)
        return float(depth(node) / max(1, max_depth))

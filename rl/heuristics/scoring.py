from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Dict

import numpy as np

from model.environment import OffloadingEnvironment

if TYPE_CHECKING:
    from rl.envs.marl_env import DAGMECMultiAgentEnv


@dataclass
class HeuristicScorer:

    edge_alloc_hz: float = 1e9
    load_penalty: float = 0.2
    risk_penalty: float = 4.0
    criticality_weight: float = 0.5
    deadline_margin: float = 0.9
    slack_penalty: float = 1.5
    over_deadline_penalty: float = 4.0
    cloud_bias: float = 0.05
    local_bias: float = 0.05
    critical_path_weight: float = 2.0
    successor_data_weight: float = 0.8
    placement_consistency_weight: float = 0.6
    future_load_weight: float = 1.0
    cloud_deadline_bonus: float = 0.3
    cloud_energy_penalty: float = 1.0
    edge_preference_bonus: float = 0.2

    def _estimate_cost(
        self,
        env_wrapper: "DAGMECMultiAgentEnv",
        i: int,
        j: int,
        action_id: int,
    ) -> float:
        env = env_wrapper.env
        dag = env.dag_tasks[i]
        st = dag.subtasks[j]
        topo_len = max(1, len(env_wrapper._topo_orders[i]))
        succ_ratio = float(len(st.successors)) / topo_len
        critical_ratio = self._critical_path_ratio(env_wrapper, i, j)
        successor_data_ratio = self._successor_data_ratio(env_wrapper, i, j)
        criticality = (
            1.0
            + self.criticality_weight * succ_ratio
            + self.critical_path_weight * critical_ratio
            + self.successor_data_weight * successor_data_ratio
        )

        outcome = env_wrapper.estimate_action_outcome(i, j, action_id)
        delay_term = criticality * outcome["delay_ratio"]
        energy_term = outcome["energy_ratio"]
        slack_ratio = self._deadline_slack_ratio(env_wrapper, i, outcome["finish"])
        margin_overrun = max(0.0, outcome["delay_ratio"] - self.deadline_margin)
        slack_term = self.slack_penalty * (margin_overrun + max(0.0, -slack_ratio))
        hard_deadline_term = self.over_deadline_penalty * outcome["c1_violation"]

        risk_term = (
            + outcome["c2_violation"]
            + outcome["c3_violation"]
        )

        load_term = 0.0
        if 1 <= action_id <= env.M:
            m = action_id - 1
            load, _ = env_wrapper._get_load_state()
            future_load = self._future_load_pressure(env_wrapper, m)
            load_term = (
                float(load[m])
                + max(0.0, outcome["peak_ratio"] - 0.8)
                + self.future_load_weight * future_load
            )

        consistency_term = self._placement_consistency_cost(env_wrapper, i, j, action_id)

        bias = 0.0
        if action_id == 0:
            bias += self.local_bias
        elif action_id == env.M + 1:
            bias += self.cloud_bias
            if slack_ratio < 0.15:
                bias -= self.cloud_deadline_bonus
            bias += self.cloud_energy_penalty * outcome["energy_ratio"]
        elif 1 <= action_id <= env.M and slack_ratio > 0.15:
            bias -= self.edge_preference_bonus

        return (
            env.alpha_w * delay_term
            + env.beta_w * energy_term
            + slack_term
            + hard_deadline_term
            + self.load_penalty * load_term
            + self.risk_penalty * risk_term
            + self.placement_consistency_weight * consistency_term
            + bias
        )

    def _critical_path_ratio(self, env_wrapper: "DAGMECMultiAgentEnv", i: int, j: int) -> float:
        dag = env_wrapper.env.dag_tasks[i]

        @lru_cache(maxsize=None)
        def depth(node: int) -> int:
            succs = dag.subtasks[node].successors
            if not succs:
                return 1
            return 1 + max(depth(s) for s in succs)

        max_depth = max(depth(n) for n in dag.subtasks)
        return float(depth(j) / max(1, max_depth))

    def _successor_data_ratio(self, env_wrapper: "DAGMECMultiAgentEnv", i: int, j: int) -> float:
        dag = env_wrapper.env.dag_tasks[i]
        out_bits = sum(dag.get_edge_data_bits(j, s) for s in dag.subtasks[j].successors)
        max_out = 0.0
        for node, st in dag.subtasks.items():
            max_out = max(max_out, sum(dag.get_edge_data_bits(node, s) for s in st.successors))
        return float(out_bits / max(max_out, 1e-9))

    def _deadline_slack_ratio(self, env_wrapper: "DAGMECMultiAgentEnv", i: int, finish: float) -> float:
        t_max = max(env_wrapper.env.T_max[i], 1e-9)
        return float((t_max - finish) / t_max)

    def _future_load_pressure(self, env_wrapper: "DAGMECMultiAgentEnv", m: int) -> float:
        counts = [len(ivs) for ivs in env_wrapper._edge_intervals]
        denom = max(1, max(counts) if counts else 1)
        return float(len(env_wrapper._edge_intervals[m]) / denom)

    def _placement_consistency_cost(
        self,
        env_wrapper: "DAGMECMultiAgentEnv",
        i: int,
        j: int,
        action_id: int,
    ) -> float:

        dag = env_wrapper.env.dag_tasks[i]
        preds = dag.subtasks[j].predecessors
        if not preds:
            return 0.0
        total_bits = 0.0
        switched_bits = 0.0
        for p in preds:
            bits = dag.get_edge_data_bits(p, j)
            total_bits += bits
            prev_action = env_wrapper._decisions[i].get(p)
            if prev_action is not None and prev_action != action_id:
                switched_bits += bits
        return float(switched_bits / max(total_bits, 1e-9))

    def _normalize_scores(self, raw_costs: np.ndarray) -> np.ndarray:
        raw_costs = np.nan_to_num(raw_costs, nan=1e9, posinf=1e9, neginf=1e9)
        mx = float(raw_costs.max())
        mn = float(raw_costs.min())
        denom = max(mx - mn, 1e-6)
        return (mx - raw_costs) / denom
    def score_step(
        self,
        env_wrapper: "DAGMECMultiAgentEnv",
        i: int,
        j: int,
    ) -> np.ndarray:

        action_dim = env_wrapper.action_dim
        raw_costs = np.asarray(
            [self._estimate_cost(env_wrapper, i, j, a) for a in range(action_dim)],
            dtype=np.float32,
        )
        return self._normalize_scores(raw_costs)

    def build_score_map(self, env: OffloadingEnvironment) -> Dict[int, np.ndarray]:

        action_dim = env.M + 2
        result: Dict[int, np.ndarray] = {}
        for i, dag in enumerate(env.dag_tasks):
            scores = np.zeros((dag.num_subtasks, action_dim), dtype=np.float32)
            for j in dag.topological_sort():
                st = env.dag_tasks[i].subtasks[j]
                local_e = env.kappa * st.c * (env.f_local[i] ** (env.energy_cfg.zeta - 1))
                raw_costs = np.zeros(action_dim, dtype=np.float32)
                for a in range(action_dim):
                    if a == 0:
                        raw_costs[a] = float(env.alpha_w * env._local_exec_time(i, j) + env.beta_w * 1.0)
                    elif a == env.M + 1:
                        raw_costs[a] = float(
                            env.alpha_w * (env._tran_delay(i, j, a) + env._cloud_exec_time(i, j))
                            + env.beta_w * (env.ch.transmission_energy(st.d, env.distances[i]) / max(local_e, 1e-12))
                        )
                    else:
                        f_alloc = max(self.edge_alloc_hz, 1e-9)
                        raw_costs[a] = float(
                            env.alpha_w * (env._tran_delay(i, j, a) + env._edge_exec_time(i, j, f_alloc))
                            + env.beta_w * (env.ch.transmission_energy(st.d, env.distances[i]) / max(local_e, 1e-12))
                        )
                scores[j] = self._normalize_scores(raw_costs)
            result[i] = scores
        return result

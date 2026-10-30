"""
基于启发式分数的动作掩码。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rl.envs.marl_env import DAGMECMultiAgentEnv
    from rl.heuristics.scoring import HeuristicScorer


def build_topk_step_masks(
    env_wrapper: "DAGMECMultiAgentEnv",
    scorer: "HeuristicScorer",
    top_k: int = 3,
) -> np.ndarray:

    N = env_wrapper.N
    action_dim = env_wrapper.action_dim
    mask = np.zeros((N, action_dim), dtype=np.float32)
    k = max(1, min(top_k, action_dim))

    for i in range(N):
        cursor = env_wrapper._cursors[i]
        topo = env_wrapper._topo_orders[i]
        if cursor >= len(topo):
            continue
        j = topo[cursor]
        scores = scorer.score_step(env_wrapper, i, j)
        top_idx = np.argsort(scores)[-k:]
        mask[i, top_idx] = 1.0
        mask[i, 0] = 1.0

    return mask

def build_topk_action_masks(
    num_agents: int,
    max_tasks: int,
    action_dim: int,
    task_counts,
    score_map,
    top_k: int = 3,
) -> np.ndarray:
    mask = np.zeros((num_agents, max_tasks, action_dim), dtype=np.float32)
    k = max(1, min(top_k, action_dim))
    for i in range(num_agents):
        count = task_counts[i] if hasattr(task_counts, "__getitem__") else task_counts
        scores = score_map.get(i, np.zeros((count, action_dim), dtype=np.float32))
        for j in range(count):
            top_idx = np.argsort(scores[j])[-k:]
            mask[i, j, top_idx] = 1.0
            mask[i, j, 0] = 1.0
    return mask

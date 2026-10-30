from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
from torch.distributions import Categorical


class MAPPONodeActor(nn.Module):

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor, avail_actions: torch.Tensor) -> torch.Tensor:

        logits = self.net(obs)
        invalid = avail_actions <= 0
        logits = logits.masked_fill(invalid, -1e9)
        return logits

    def sample_actions(
        self,
        obs: torch.Tensor,
        avail_actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        logits = self.forward(obs, avail_actions)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return {"actions": actions, "log_probs": log_probs, "entropy": entropy}

    def act_deterministic(
        self,
        obs: torch.Tensor,
        avail_actions: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.forward(obs, avail_actions)
        return torch.argmax(logits, dim=-1)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        avail_actions: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        logits = self.forward(obs, avail_actions)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy


class IndependentPPOActor(nn.Module):

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_agents: int = 12,
        agent_id_offset: int = 10,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.agent_id_offset = agent_id_offset
        self.actors = nn.ModuleList(
            MAPPONodeActor(obs_dim, action_dim, hidden_dim)
            for _ in range(num_agents)
        )

    def _agent_ids(self, obs: torch.Tensor) -> torch.Tensor:
        identity = obs[:, self.agent_id_offset:self.agent_id_offset + self.num_agents]
        return torch.argmax(identity, dim=-1)

    def forward(self, obs: torch.Tensor, avail_actions: torch.Tensor) -> torch.Tensor:
        agent_ids = self._agent_ids(obs)
        logits = torch.zeros(
            (obs.shape[0], self.action_dim), dtype=obs.dtype, device=obs.device
        )
        for agent_id, actor in enumerate(self.actors):
            idx = torch.nonzero(agent_ids == agent_id, as_tuple=False).squeeze(-1)
            if idx.numel() > 0:
                selected = actor(obs.index_select(0, idx), avail_actions.index_select(0, idx))
                logits = logits.index_copy(0, idx, selected)
        return logits

    def sample_actions(self, obs: torch.Tensor, avail_actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        dist = Categorical(logits=self.forward(obs, avail_actions))
        actions = dist.sample()
        return {"actions": actions, "log_probs": dist.log_prob(actions), "entropy": dist.entropy()}

    def act_deterministic(self, obs: torch.Tensor, avail_actions: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.forward(obs, avail_actions), dim=-1)

    def evaluate_actions(
        self, obs: torch.Tensor, avail_actions: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = Categorical(logits=self.forward(obs, avail_actions))
        return dist.log_prob(actions), dist.entropy()


class TPTDTOActor(nn.Module):

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_agents: int = 12,
        task_feature_dim: int = 10,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        transformer_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if action_dim < 3:
            raise ValueError("TPTDTOActor requires local, edge and cloud actions")
        if hidden_dim % transformer_heads != 0:
            raise ValueError("hidden_dim must be divisible by transformer_heads")

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_agents = num_agents
        self.task_feature_dim = task_feature_dim
        self.num_edge_servers = action_dim - 2
        self.agent_id_offset = task_feature_dim
        self.resource_offset = self.agent_id_offset + num_agents
        required_obs_dim = self.resource_offset + 2 * self.num_edge_servers
        if obs_dim < required_obs_dim:
            raise ValueError(
                "TPTDTOActor observation is missing agent/resource context: "
                f"obs_dim={obs_dim}, required>={required_obs_dim}"
            )

        self.task_encoder = nn.Sequential(
            nn.Linear(task_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.priority_encoder = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(num_agents, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.action_feature_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.action_embeddings = nn.Parameter(
            torch.empty(action_dim, hidden_dim)
        )
        self.position_embeddings = nn.Parameter(
            torch.empty(action_dim + 2, hidden_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=transformer_heads,
            dim_feedforward=2 * hidden_dim,
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
            norm=nn.LayerNorm(hidden_dim),
        )
        self.action_scorer = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.normal_(self.action_embeddings, std=0.02)
        nn.init.normal_(self.position_embeddings, std=0.02)

    def _action_features(self, obs: torch.Tensor) -> torch.Tensor:
        task = obs[:, :self.task_feature_dim]
        loads = obs[
            :,
            self.resource_offset:
            self.resource_offset + self.num_edge_servers,
        ]
        available = obs[
            :,
            self.resource_offset + self.num_edge_servers:
            self.resource_offset + 2 * self.num_edge_servers,
        ]
        zeros = torch.zeros_like(task[:, 0])
        ones = torch.ones_like(task[:, 0])

        local = torch.stack(
            [ones, zeros, zeros, task[:, 3], zeros, ones],
            dim=-1,
        )
        edge_tokens = []
        for edge_index in range(self.num_edge_servers):
            edge_tokens.append(
                torch.stack(
                    [
                        zeros,
                        ones,
                        zeros,
                        task[:, 4],
                        loads[:, edge_index],
                        available[:, edge_index],
                    ],
                    dim=-1,
                )
            )
        cloud = torch.stack(
            [zeros, zeros, ones, task[:, 5], zeros, ones],
            dim=-1,
        )
        return torch.stack([local, *edge_tokens, cloud], dim=1)

    def forward(self, obs: torch.Tensor, avail_actions: torch.Tensor) -> torch.Tensor:
        task = obs[:, :self.task_feature_dim]
        agent_context = obs[
            :,
            self.agent_id_offset:self.agent_id_offset + self.num_agents,
        ]
        priority = task[:, [3, 6, 7, 8, 9]]

        task_token = self.task_encoder(task) + self.priority_encoder(priority)
        context_token = self.context_encoder(agent_context)
        action_tokens = (
            self.action_feature_encoder(self._action_features(obs))
            + self.action_embeddings.unsqueeze(0)
        )
        tokens = torch.cat(
            [task_token.unsqueeze(1), context_token.unsqueeze(1), action_tokens],
            dim=1,
        )
        tokens = tokens + self.position_embeddings.unsqueeze(0)
        encoded = self.transformer(tokens)

        contextual_task = encoded[:, 0].unsqueeze(1).expand(-1, self.action_dim, -1)
        contextual_actions = encoded[:, 2:]
        learned_actions = self.action_embeddings.unsqueeze(0).expand(
            obs.shape[0], -1, -1
        )
        logits = self.action_scorer(
            torch.cat(
                [contextual_task, contextual_actions, learned_actions],
                dim=-1,
            )
        ).squeeze(-1)
        return logits.masked_fill(avail_actions <= 0, -1e9)

    def sample_actions(
        self,
        obs: torch.Tensor,
        avail_actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        dist = Categorical(logits=self.forward(obs, avail_actions))
        actions = dist.sample()
        return {
            "actions": actions,
            "log_probs": dist.log_prob(actions),
            "entropy": dist.entropy(),
        }

    def act_deterministic(
        self,
        obs: torch.Tensor,
        avail_actions: torch.Tensor,
    ) -> torch.Tensor:
        return torch.argmax(self.forward(obs, avail_actions), dim=-1)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        avail_actions: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = Categorical(logits=self.forward(obs, avail_actions))
        return dist.log_prob(actions), dist.entropy()


class HeuristicActionConditionedActor(nn.Module):
    """
    启发式动作条件化 Actor。
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.heuristic_dim = 6 * action_dim + 1
        self.base_obs_dim = obs_dim - self.heuristic_dim
        if self.base_obs_dim <= 0:
            raise ValueError(
                "HeuristicActionConditionedActor 需要 strong_outcome 状态特征，"
                f"但 obs_dim={obs_dim}, action_dim={action_dim}"
            )

        self.base_encoder = nn.Sequential(
            nn.Linear(self.base_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        per_action_feat_dim = 7 + action_dim
        self.action_scorer = nn.Sequential(
            nn.Linear(hidden_dim + per_action_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _split_obs(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        base_obs = obs[..., :self.base_obs_dim]
        h = obs[..., self.base_obs_dim:]
        return base_obs, h

    def _per_action_features(self, h: torch.Tensor) -> torch.Tensor:
        B = h.shape[0]
        A = self.action_dim
        delay = h[:, 0 * A:1 * A]
        energy = h[:, 1 * A:2 * A]
        c1 = h[:, 2 * A:3 * A]
        c3 = h[:, 3 * A:4 * A]
        score = h[:, 4 * A:5 * A]
        best = h[:, 5 * A:6 * A]
        margin = h[:, 6 * A:6 * A + 1].expand(B, A)
        action_feats = torch.stack([delay, energy, c1, c3, score, best, margin], dim=-1)
        eye = torch.eye(A, dtype=obs_dtype(h), device=h.device).unsqueeze(0).expand(B, A, A)
        return torch.cat([action_feats, eye], dim=-1)

    def forward(self, obs: torch.Tensor, avail_actions: torch.Tensor) -> torch.Tensor:
        base_obs, h = self._split_obs(obs)
        B = obs.shape[0]
        A = self.action_dim
        base_emb = self.base_encoder(base_obs).unsqueeze(1).expand(B, A, -1)
        action_feats = self._per_action_features(h)
        scorer_in = torch.cat([base_emb, action_feats], dim=-1)
        logits = self.action_scorer(scorer_in).squeeze(-1)
        invalid = avail_actions <= 0
        return logits.masked_fill(invalid, -1e9)

    def sample_actions(
        self,
        obs: torch.Tensor,
        avail_actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        logits = self.forward(obs, avail_actions)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return {"actions": actions, "log_probs": log_probs, "entropy": entropy}

    def act_deterministic(
        self,
        obs: torch.Tensor,
        avail_actions: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.forward(obs, avail_actions)
        return torch.argmax(logits, dim=-1)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        avail_actions: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(obs, avail_actions)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy


def obs_dtype(tensor: torch.Tensor) -> torch.dtype:
    return tensor.dtype


class CentralValueCritic(nn.Module):
    """集中式 critic，输入全局 state，输出标量价值。"""

    def __init__(self, state_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:

        return self.value_net(state).squeeze(-1)


class IndependentValueCritic(nn.Module):

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        num_agents: int = 12,
        agent_id_offset: int = 10,
    ) -> None:
        super().__init__()
        self.num_agents = num_agents
        self.agent_id_offset = agent_id_offset
        self.critics = nn.ModuleList(
            CentralValueCritic(state_dim, hidden_dim) for _ in range(num_agents)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        squeeze = state.dim() == 1
        if squeeze:
            state = state.unsqueeze(0)
        identity = state[:, self.agent_id_offset:self.agent_id_offset + self.num_agents]
        agent_ids = torch.argmax(identity, dim=-1)
        values = torch.zeros(state.shape[0], dtype=state.dtype, device=state.device)
        for agent_id, critic in enumerate(self.critics):
            idx = torch.nonzero(agent_ids == agent_id, as_tuple=False).squeeze(-1)
            if idx.numel() > 0:
                selected = critic(state.index_select(0, idx))
                values = values.index_copy(0, idx, selected)
        return values.squeeze(0) if squeeze else values

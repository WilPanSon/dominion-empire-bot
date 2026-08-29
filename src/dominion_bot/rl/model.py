"""Masked actor-critic network used by PPO."""

from __future__ import annotations

import torch
from torch import nn


class MaskedActorCritic(nn.Module):
    def __init__(
        self,
        observation_size: int,
        action_size: int,
        hidden_size: int = 256,
    ):
        super().__init__()
        self.observation_size = observation_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.trunk = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_size, action_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(
        self, observations: torch.Tensor, action_masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(observations)
        logits = self.policy_head(hidden)
        # A valid engine state always has at least one legal action. Using the
        # minimum finite float keeps distributions stable on all accelerators.
        masked_logits = logits.masked_fill(~action_masks, torch.finfo(logits.dtype).min)
        values = self.value_head(hidden).squeeze(-1)
        return masked_logits, values


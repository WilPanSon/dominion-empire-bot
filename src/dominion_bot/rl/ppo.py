"""Shared-policy, terminal-return PPO for two-player Dominion self-play."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Iterable

import torch
from torch.distributions import Categorical
from torch.nn import functional as F

from ..cards import RULESET_ID
from ..encoding import ACTION_VOCAB, FeatureEncoder
from ..env import DominionEnv
from ..engine import Decision, Game
from .model import MaskedActorCritic


@dataclass(frozen=True, slots=True)
class PPOConfig:
    hidden_size: int = 256
    learning_rate: float = 3e-4
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 512
    episodes_per_iteration: int = 64


@dataclass(slots=True)
class Sample:
    observation: tuple[float, ...]
    action_mask: tuple[bool, ...]
    action: int
    old_log_probability: float
    old_value: float
    player: int
    return_value: float = 0.0


@dataclass(frozen=True, slots=True)
class UpdateMetrics:
    samples: int
    policy_loss: float
    value_loss: float
    entropy: float
    mean_return: float


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collect_episode(
    env: DominionEnv,
    model: MaskedActorCritic,
    *,
    seed: int,
    device: torch.device,
) -> list[Sample]:
    """Collect one on-policy self-play episode.

    Every decision is labeled with the terminal result from its acting player's
    perspective. Dominion has no intermediate reward in this baseline.
    """
    state = env.reset(seed)
    samples: list[Sample] = []
    model.eval()
    while True:
        observation = torch.tensor(
            [state.observation], dtype=torch.float32, device=device
        )
        mask = torch.tensor([state.action_mask], dtype=torch.bool, device=device)
        with torch.no_grad():
            logits, value = model(observation, mask)
            distribution = Categorical(logits=logits)
            action = distribution.sample()
            log_probability = distribution.log_prob(action)
        samples.append(
            Sample(
                state.observation,
                state.action_mask,
                int(action.item()),
                float(log_probability.item()),
                float(value.item()),
                state.player,
            )
        )
        result = env.step(int(action.item()))
        if result.done:
            for sample in samples:
                sample.return_value = result.rewards[sample.player]
            return samples
        assert result.state is not None
        state = result.state


def collect_batch(
    env: DominionEnv,
    model: MaskedActorCritic,
    *,
    seeds: Iterable[int],
    device: torch.device,
) -> list[Sample]:
    batch: list[Sample] = []
    for seed in seeds:
        batch.extend(collect_episode(env, model, seed=seed, device=device))
    return batch


def ppo_update(
    model: MaskedActorCritic,
    optimizer: torch.optim.Optimizer,
    samples: list[Sample],
    config: PPOConfig,
    *,
    device: torch.device,
) -> UpdateMetrics:
    if not samples:
        raise ValueError("cannot update PPO with an empty batch")

    observations = torch.tensor(
        [sample.observation for sample in samples], dtype=torch.float32, device=device
    )
    masks = torch.tensor(
        [sample.action_mask for sample in samples], dtype=torch.bool, device=device
    )
    actions = torch.tensor(
        [sample.action for sample in samples], dtype=torch.long, device=device
    )
    old_log_probabilities = torch.tensor(
        [sample.old_log_probability for sample in samples],
        dtype=torch.float32,
        device=device,
    )
    returns = torch.tensor(
        [sample.return_value for sample in samples], dtype=torch.float32, device=device
    )
    old_values = torch.tensor(
        [sample.old_value for sample in samples], dtype=torch.float32, device=device
    )
    advantages = returns - old_values
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    total_policy = total_value = total_entropy = 0.0
    updates = 0
    model.train()
    for _ in range(config.update_epochs):
        permutation = torch.randperm(len(samples), device=device)
        for start in range(0, len(samples), config.minibatch_size):
            indices = permutation[start : start + config.minibatch_size]
            logits, values = model(observations[indices], masks[indices])
            distribution = Categorical(logits=logits)
            new_log_probabilities = distribution.log_prob(actions[indices])
            entropy = distribution.entropy().mean()

            ratio = (new_log_probabilities - old_log_probabilities[indices]).exp()
            unclipped = ratio * advantages[indices]
            clipped = torch.clamp(
                ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio
            ) * advantages[indices]
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = F.mse_loss(values, returns[indices])
            loss = (
                policy_loss
                + config.value_coefficient * value_loss
                - config.entropy_coefficient * entropy
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            total_policy += float(policy_loss.item())
            total_value += float(value_loss.item())
            total_entropy += float(entropy.item())
            updates += 1

    return UpdateMetrics(
        samples=len(samples),
        policy_loss=total_policy / updates,
        value_loss=total_value / updates,
        entropy=total_entropy / updates,
        mean_return=float(returns.mean().item()),
    )


class NeuralPolicy:
    """Greedy policy adapter used for deterministic tournament evaluation."""

    def __init__(
        self,
        model: MaskedActorCritic,
        *,
        encoder: FeatureEncoder | None = None,
        device: torch.device | None = None,
    ):
        self.model = model
        self.encoder = encoder or FeatureEncoder()
        self.device = device or next(model.parameters()).device

    def choose(self, game: Game, decision: Decision) -> str:
        state = self.encoder.encode(game, decision)
        observation = torch.tensor(
            [state.observation], dtype=torch.float32, device=self.device
        )
        mask = torch.tensor([state.action_mask], dtype=torch.bool, device=self.device)
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(observation, mask)
        index = int(logits.argmax(dim=-1).item())
        return ACTION_VOCAB[index]


def save_checkpoint(
    path: str | Path,
    model: MaskedActorCritic,
    optimizer: torch.optim.Optimizer,
    config: PPOConfig,
    *,
    iteration: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "ruleset_id": RULESET_ID,
            "action_vocab": ACTION_VOCAB,
            "observation_size": model.observation_size,
            "action_size": model.action_size,
            "hidden_size": model.hidden_size,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
            "iteration": iteration,
        },
        path,
    )


def load_checkpoint(
    path: str | Path, *, device: torch.device
) -> tuple[MaskedActorCritic, dict[str, object]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload["ruleset_id"] != RULESET_ID:
        raise ValueError(
            f"checkpoint ruleset {payload['ruleset_id']!r} != {RULESET_ID!r}"
        )
    if tuple(payload["action_vocab"]) != ACTION_VOCAB:
        raise ValueError("checkpoint action vocabulary does not match this build")
    model = MaskedActorCritic(
        int(payload["observation_size"]),
        int(payload["action_size"]),
        int(payload["hidden_size"]),
    ).to(device)
    model.load_state_dict(payload["model"])
    return model, payload


def train(
    *,
    iterations: int,
    config: PPOConfig,
    seed: int = 0,
    checkpoint_path: str | Path = "checkpoints/dominion_ppo.pt",
    device_name: str = "auto",
) -> MaskedActorCritic:
    random.seed(seed)
    torch.manual_seed(seed)
    device = select_device(device_name)
    env = DominionEnv()
    model = MaskedActorCritic(
        env.observation_size, env.action_size, config.hidden_size
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    for iteration in range(1, iterations + 1):
        first_seed = seed + (iteration - 1) * config.episodes_per_iteration
        samples = collect_batch(
            env,
            model,
            seeds=range(first_seed, first_seed + config.episodes_per_iteration),
            device=device,
        )
        metrics = ppo_update(model, optimizer, samples, config, device=device)
        print(
            json.dumps(
                {
                    "iteration": iteration,
                    "device": str(device),
                    **asdict(metrics),
                },
                sort_keys=True,
            )
        )
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            config,
            iteration=iteration,
        )
    return model


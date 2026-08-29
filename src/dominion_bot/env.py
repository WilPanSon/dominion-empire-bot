"""Small framework-neutral RL adapter around :class:`dominion_bot.engine.Game`."""

from __future__ import annotations

from dataclasses import dataclass

from .encoding import ACTION_VOCAB, EncodedState, FeatureEncoder
from .engine import Game, IllegalAction


@dataclass(frozen=True, slots=True)
class StepResult:
    state: EncodedState | None
    done: bool
    rewards: tuple[float, float]


class DominionEnv:
    """Sequential self-play environment with a shared action vocabulary."""

    def __init__(self, *, max_turns: int = 200, encoder: FeatureEncoder | None = None):
        self.max_turns = max_turns
        self.encoder = encoder or FeatureEncoder()
        self.game: Game | None = None
        self.state: EncodedState | None = None

    @property
    def observation_size(self) -> int:
        return self.encoder.observation_size

    @property
    def action_size(self) -> int:
        return len(ACTION_VOCAB)

    def reset(self, seed: int = 0) -> EncodedState:
        self.game = Game(seed, max_turns=self.max_turns)
        decision = self.game.advance_until_decision()
        if decision is None:
            raise RuntimeError("a new game unexpectedly terminated")
        self.state = self.encoder.encode(self.game, decision)
        return self.state

    def step(self, action_index: int) -> StepResult:
        if self.game is None or self.state is None:
            raise RuntimeError("call reset() before step()")
        if action_index < 0 or action_index >= self.action_size:
            raise IllegalAction(f"action index out of range: {action_index}")
        if not self.state.action_mask[action_index]:
            raise IllegalAction(f"masked action: {ACTION_VOCAB[action_index]}")

        decision = self.game.step(ACTION_VOCAB[action_index])
        if decision is None:
            rewards = (self.game.outcome(0), self.game.outcome(1))
            self.state = None
            return StepResult(None, True, rewards)
        self.state = self.encoder.encode(self.game, decision)
        return StepResult(self.state, False, (0.0, 0.0))


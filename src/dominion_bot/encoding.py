"""Stable vector observations and legal-action masks."""

from __future__ import annotations

from dataclasses import dataclass

from .cards import CARDS, EVENTS, PILE_ORDER, CardType
from .engine import Decision, DecisionKind, Game, Observation, Phase


CARD_NAMES = tuple(sorted(CARDS))
DECISION_KINDS = tuple(DecisionKind)
PHASES = tuple(Phase)


def _action_vocabulary() -> tuple[str, ...]:
    action_names = [
        *(f"play:{name}" for name in CARD_NAMES if CARDS[name].is_type(CardType.ACTION)),
        "end_action",
        *(f"buy:{name}" for name in PILE_ORDER),
        *(f"event:{name}" for name in EVENTS),
        "end_buy",
        *(f"chapel_trash:{name}" for name in CARD_NAMES),
        "chapel_done",
        *(f"workshop_gain:{name}" for name in PILE_ORDER),
        *(f"forum_discard:{name}" for name in CARD_NAMES),
    ]
    if len(action_names) != len(set(action_names)):
        raise RuntimeError("action vocabulary contains duplicates")
    return tuple(action_names)


ACTION_VOCAB = _action_vocabulary()
ACTION_TO_INDEX = {name: index for index, name in enumerate(ACTION_VOCAB)}


@dataclass(frozen=True, slots=True)
class EncodedState:
    observation: tuple[float, ...]
    action_mask: tuple[bool, ...]
    player: int
    decision_id: int


class FeatureEncoder:
    """Encode public information plus the deciding player's private hand.

    Counts reveal card ownership—which is inferable from public gains—but never
    hidden zones or deck order. The encoding is intentionally simple enough to
    serve as a trustworthy PPO baseline.
    """

    action_vocab = ACTION_VOCAB
    action_to_index = ACTION_TO_INDEX

    def encode(self, game: Game, decision: Decision) -> EncodedState:
        obs = game.observation(decision.player, decision)
        vector = self._vector(obs)
        mask = [False] * len(self.action_vocab)
        for action in decision.actions:
            try:
                mask[self.action_to_index[action]] = True
            except KeyError as exc:
                raise RuntimeError(f"action missing from vocabulary: {action}") from exc
        return EncodedState(tuple(vector), tuple(mask), decision.player, decision.id)

    def _vector(self, obs: Observation) -> list[float]:
        hand = _counts(obs.hand)
        own = dict(obs.own_counts)
        opponent = dict(obs.opponent_counts)
        vector: list[float] = []

        for name in CARD_NAMES:
            vector.extend(
                (
                    hand.get(name, 0) / 10.0,
                    own.get(name, 0) / 40.0,
                    opponent.get(name, 0) / 40.0,
                )
            )

        for pile in obs.piles:
            vector.extend((pile.remaining / 50.0, pile.tokens / 10.0))

        # Top-card one-hot makes the split pile state explicit.
        top_cards = {pile.top for pile in obs.piles if pile.top is not None}
        vector.extend(1.0 if name in top_cards else 0.0 for name in CARD_NAMES)

        vector.extend(
            (
                min(obs.turn_number, 100) / 100.0,
                min(obs.actions, 20) / 20.0,
                min(obs.buys, 10) / 10.0,
                min(obs.coins, 30) / 30.0,
                min(obs.debt, 30) / 30.0,
                min(obs.vp_tokens, 50) / 50.0,
                min(obs.opponent_debt, 30) / 30.0,
                min(obs.opponent_vp_tokens, 50) / 50.0,
                float(obs.player == obs.active_player),
                float(obs.player == 0),
            )
        )
        vector.extend(float(obs.phase == phase) for phase in PHASES)
        vector.extend(float(obs.decision_kind == kind) for kind in DECISION_KINDS)
        return vector

    @property
    def observation_size(self) -> int:
        # Build the formula directly so callers do not need a live game.
        return (
            len(CARD_NAMES) * 3
            + len(PILE_ORDER) * 2
            + len(CARD_NAMES)
            + 10
            + len(PHASES)
            + len(DECISION_KINDS)
        )


def _counts(names: tuple[str, ...]) -> dict[str, int]:
    result: dict[str, int] = {}
    for name in names:
        result[name] = result.get(name, 0) + 1
    return result


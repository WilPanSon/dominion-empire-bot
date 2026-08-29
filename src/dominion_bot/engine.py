"""Deterministic two-player Dominion engine for the RL Intro ruleset.

The engine advances through forced bookkeeping internally and stops only when a
player has a meaningful choice. This keeps RL trajectories compact while still
making every strategically distinct choice explicit.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
import random
from typing import Iterable

from .cards import (
    CARDS,
    EVENTS,
    PILE_ORDER,
    RULESET_ID,
    CardType,
    pile_card_names,
)


class Phase(str, Enum):
    ACTION = "action"
    BUY = "buy"
    TERMINAL = "terminal"


class DecisionKind(str, Enum):
    ACTION = "action"
    BUY = "buy"
    CHAPEL_TRASH = "chapel_trash"
    WORKSHOP_GAIN = "workshop_gain"
    FORUM_DISCARD = "forum_discard"


class IllegalAction(ValueError):
    """Raised when an action is not in the current legal-action set."""


@dataclass(slots=True)
class Pile:
    name: str
    cards: list[str]
    tokens: int = 0

    @property
    def top(self) -> str | None:
        return self.cards[-1] if self.cards else None

    @property
    def empty(self) -> bool:
        return not self.cards

    def take(self) -> str | None:
        return self.cards.pop() if self.cards else None


@dataclass(slots=True)
class PlayerState:
    deck: list[str] = field(default_factory=list)
    hand: list[str] = field(default_factory=list)
    discard: list[str] = field(default_factory=list)
    in_play: list[str] = field(default_factory=list)
    debt: int = 0
    vp_tokens: int = 0
    turns_taken: int = 0

    def all_cards(self) -> list[str]:
        return self.deck + self.hand + self.discard + self.in_play

    def counts(self) -> Counter[str]:
        return Counter(self.all_cards())


@dataclass(frozen=True, slots=True)
class PileObservation:
    name: str
    top: str | None
    remaining: int
    tokens: int


@dataclass(frozen=True, slots=True)
class Observation:
    ruleset_id: str
    player: int
    active_player: int
    turn_number: int
    phase: Phase
    decision_kind: DecisionKind
    actions: int
    buys: int
    coins: int
    debt: int
    vp_tokens: int
    opponent_debt: int
    opponent_vp_tokens: int
    hand: tuple[str, ...]
    own_counts: tuple[tuple[str, int], ...]
    opponent_counts: tuple[tuple[str, int], ...]
    piles: tuple[PileObservation, ...]


@dataclass(frozen=True, slots=True)
class Decision:
    id: int
    player: int
    kind: DecisionKind
    actions: tuple[str, ...]


@dataclass(slots=True)
class PendingChoice:
    kind: DecisionKind
    player: int
    remaining: int = 1


class Game:
    """A seeded, replayable two-player game.

    The ruleset deliberately implements a compact kingdom rather than pretending
    to cover every Dominion card. All randomness flows through ``self.rng``.
    """

    WEDDING_COIN_COST = 4
    WEDDING_DEBT_COST = 3

    def __init__(self, seed: int = 0, *, max_turns: int = 200):
        self.seed = seed
        self.rng = random.Random(seed)
        self.max_turns = max_turns
        self.players = [PlayerState(), PlayerState()]
        self.supply = self._make_supply()
        self.trash: list[str] = []
        self.active_player = 0
        self.turn_number = 1
        self.phase = Phase.ACTION
        self.actions = 1
        self.buys = 1
        self.coins = 0
        self.pending: PendingChoice | None = None
        self.terminal = False
        self.truncated = False
        self.decision_id = 0
        self.history: list[str] = [f"seed:{seed}", f"ruleset:{RULESET_ID}"]

        for player in self.players:
            player.deck = ["Copper"] * 7 + ["Estate"] * 3
            self.rng.shuffle(player.deck)
            self._draw(player, 5)

    @staticmethod
    def _make_supply() -> dict[str, Pile]:
        piles = {
            "Copper": Pile("Copper", ["Copper"] * 46),
            "Silver": Pile("Silver", ["Silver"] * 40),
            "Gold": Pile("Gold", ["Gold"] * 30),
            "Estate": Pile("Estate", ["Estate"] * 8),
            "Duchy": Pile("Duchy", ["Duchy"] * 8),
            "Province": Pile("Province", ["Province"] * 8),
            "Curse": Pile("Curse", ["Curse"] * 10),
        }
        for name in (
            "Village",
            "Smithy",
            "Market",
            "Chapel",
            "Workshop",
            "City Quarter",
            "Farmers' Market",
            "Forum",
            "Villa",
        ):
            piles[name] = Pile(name, [name] * 10)
        # The list end is the pile top, hence Emporiums precede Patricians.
        piles["Patrician/Emporium"] = Pile(
            "Patrician/Emporium", ["Emporium"] * 5 + ["Patrician"] * 5
        )
        return piles

    @property
    def current_player(self) -> PlayerState:
        return self.players[self.active_player]

    @property
    def opponent(self) -> PlayerState:
        return self.players[1 - self.active_player]

    def _ensure_deck(self, player: PlayerState) -> bool:
        if player.deck:
            return True
        if not player.discard:
            return False
        player.deck = player.discard
        player.discard = []
        self.rng.shuffle(player.deck)
        return True

    def _draw(self, player: PlayerState, count: int) -> int:
        drawn = 0
        for _ in range(count):
            if not self._ensure_deck(player):
                break
            player.hand.append(player.deck.pop())
            drawn += 1
        return drawn

    def _peek_top(self, player: PlayerState) -> str | None:
        if not self._ensure_deck(player):
            return None
        return player.deck[-1]

    def _playable_actions(self) -> list[str]:
        if self.actions <= 0:
            return []
        return sorted(
            {
                name
                for name in self.current_player.hand
                if CARDS[name].is_type(CardType.ACTION)
            }
        )

    def _enter_buy_phase(self) -> None:
        self.phase = Phase.BUY
        player = self.current_player
        # All Treasures in this ruleset are vanilla and their order is irrelevant.
        treasures = [
            name for name in player.hand if CARDS[name].is_type(CardType.TREASURE)
        ]
        for name in treasures:
            player.hand.remove(name)
            player.in_play.append(name)
            self.coins += CARDS[name].treasure
        self._pay_maximum_debt()

    def _pay_maximum_debt(self) -> None:
        player = self.current_player
        payment = min(self.coins, player.debt)
        if payment:
            self.coins -= payment
            player.debt -= payment
            self.history.append(f"p{self.active_player}:pay_debt:{payment}")

    def _gain(self, player_index: int, pile_name: str, *, bought: bool = False) -> str | None:
        pile = self.supply[pile_name]
        card_name = pile.take()
        if card_name is None:
            return None

        player = self.players[player_index]
        if card_name == "Villa":
            player.hand.append(card_name)
            if player_index == self.active_player:
                self.actions += 1
                if self.phase == Phase.BUY:
                    self.phase = Phase.ACTION
        else:
            player.discard.append(card_name)

        if card_name == "Emporium":
            actions_in_play = sum(
                CARDS[name].is_type(CardType.ACTION) for name in player.in_play
            )
            if actions_in_play >= 5:
                player.vp_tokens += 2
        if bought and card_name == "Forum":
            self.buys += 1

        self.history.append(f"p{player_index}:gain:{card_name}")
        return card_name

    def _trash_from_hand(self, player: PlayerState, card_name: str) -> None:
        player.hand.remove(card_name)
        self.trash.append(card_name)
        self.history.append(f"p{self.active_player}:trash:{card_name}")

    def _play_action(self, card_name: str) -> None:
        player = self.current_player
        player.hand.remove(card_name)
        player.in_play.append(card_name)
        self.actions -= 1

        if card_name == "Village":
            self._draw(player, 1)
            self.actions += 2
        elif card_name == "Smithy":
            self._draw(player, 3)
        elif card_name == "Market":
            self._draw(player, 1)
            self.actions += 1
            self.buys += 1
            self.coins += 1
        elif card_name == "Chapel":
            self.pending = PendingChoice(
                DecisionKind.CHAPEL_TRASH, self.active_player, remaining=4
            )
        elif card_name == "Workshop":
            self.pending = PendingChoice(
                DecisionKind.WORKSHOP_GAIN, self.active_player, remaining=1
            )
        elif card_name == "City Quarter":
            self.actions += 2
            action_cards = sum(
                CARDS[name].is_type(CardType.ACTION) for name in player.hand
            )
            self._draw(player, action_cards)
        elif card_name == "Farmers' Market":
            self.buys += 1
            pile = self.supply["Farmers' Market"]
            if pile.tokens >= 4:
                player.vp_tokens += pile.tokens
                pile.tokens = 0
                player.in_play.remove(card_name)
                self.trash.append(card_name)
            else:
                pile.tokens += 1
                self.coins += pile.tokens
        elif card_name == "Forum":
            self._draw(player, 3)
            self.actions += 1
            self.pending = PendingChoice(
                DecisionKind.FORUM_DISCARD, self.active_player, remaining=2
            )
        elif card_name == "Villa":
            self.actions += 2
            self.buys += 1
            self.coins += 1
        elif card_name == "Patrician":
            self._draw(player, 1)
            self.actions += 1
            revealed = self._peek_top(player)
            if revealed is not None and CARDS[revealed].cost.coins >= 5:
                player.hand.append(player.deck.pop())
        elif card_name == "Emporium":
            self._draw(player, 1)
            self.actions += 1
            self.coins += 1
        else:  # Defensive: every Action in the registry needs explicit behavior.
            raise RuntimeError(f"unimplemented Action card: {card_name}")

    def _workshop_candidates(self) -> list[str]:
        candidates: list[str] = []
        for pile_name in PILE_ORDER:
            top = self.supply[pile_name].top
            if top is None:
                continue
            cost = CARDS[top].cost
            if cost.debt == 0 and cost.coins <= 4:
                candidates.append(pile_name)
        return candidates

    def _buy_candidates(self) -> list[str]:
        if self.current_player.debt or self.buys <= 0:
            return []
        candidates = []
        for pile_name in PILE_ORDER:
            top = self.supply[pile_name].top
            if top is not None and CARDS[top].cost.coins <= self.coins:
                candidates.append(pile_name)
        return candidates

    def _pending_decision(self) -> Decision | None:
        assert self.pending is not None
        pending = self.pending
        player = self.players[pending.player]

        if pending.kind == DecisionKind.CHAPEL_TRASH:
            if pending.remaining <= 0 or not player.hand:
                self.pending = None
                return None
            choices = tuple(
                [f"chapel_trash:{name}" for name in sorted(set(player.hand))]
                + ["chapel_done"]
            )
        elif pending.kind == DecisionKind.WORKSHOP_GAIN:
            candidates = self._workshop_candidates()
            if not candidates:
                self.pending = None
                return None
            choices = tuple(f"workshop_gain:{name}" for name in candidates)
        elif pending.kind == DecisionKind.FORUM_DISCARD:
            if pending.remaining <= 0:
                self.pending = None
                return None
            if len(player.hand) <= pending.remaining:
                while player.hand:
                    player.discard.append(player.hand.pop())
                self.pending = None
                return None
            choices = tuple(
                f"forum_discard:{name}" for name in sorted(set(player.hand))
            )
        else:
            raise RuntimeError(f"unknown pending decision: {pending.kind}")

        return Decision(self.decision_id, pending.player, pending.kind, choices)

    def advance_until_decision(self) -> Decision | None:
        """Resolve forced transitions and return the next real decision."""
        while not self.terminal:
            if self.pending is not None:
                decision = self._pending_decision()
                if decision is not None:
                    return decision
                continue

            if self.phase == Phase.ACTION:
                playable = self._playable_actions()
                if playable:
                    actions = tuple([f"play:{name}" for name in playable] + ["end_action"])
                    return Decision(
                        self.decision_id,
                        self.active_player,
                        DecisionKind.ACTION,
                        actions,
                    )
                self._enter_buy_phase()
                continue

            if self.phase == Phase.BUY:
                self._pay_maximum_debt()
                if self.buys <= 0:
                    self._cleanup()
                    continue
                actions = [f"buy:{name}" for name in self._buy_candidates()]
                if (
                    not self.current_player.debt
                    and self.coins >= self.WEDDING_COIN_COST
                ):
                    actions.append("event:Wedding")
                actions.append("end_buy")
                return Decision(
                    self.decision_id,
                    self.active_player,
                    DecisionKind.BUY,
                    tuple(actions),
                )

            raise RuntimeError(f"invalid phase: {self.phase}")
        return None

    def step(self, action_key: str) -> Decision | None:
        decision = self.advance_until_decision()
        if decision is None:
            raise IllegalAction("the game is already over")
        if action_key not in decision.actions:
            raise IllegalAction(
                f"{action_key!r} is illegal for {decision.kind.value}; "
                f"expected one of {decision.actions}"
            )

        self.history.append(f"p{decision.player}:choose:{action_key}")
        if decision.kind == DecisionKind.ACTION:
            if action_key == "end_action":
                self._enter_buy_phase()
            else:
                self._play_action(action_key.removeprefix("play:"))
        elif decision.kind == DecisionKind.BUY:
            if action_key == "end_buy":
                self._cleanup()
            elif action_key == "event:Wedding":
                self._buy_wedding()
            else:
                self._buy_card(action_key.removeprefix("buy:"))
        elif decision.kind == DecisionKind.CHAPEL_TRASH:
            if action_key == "chapel_done":
                self.pending = None
            else:
                self._trash_from_hand(
                    self.players[decision.player],
                    action_key.removeprefix("chapel_trash:"),
                )
                assert self.pending is not None
                self.pending.remaining -= 1
        elif decision.kind == DecisionKind.WORKSHOP_GAIN:
            self._gain(
                decision.player, action_key.removeprefix("workshop_gain:")
            )
            self.pending = None
        elif decision.kind == DecisionKind.FORUM_DISCARD:
            player = self.players[decision.player]
            name = action_key.removeprefix("forum_discard:")
            player.hand.remove(name)
            player.discard.append(name)
            assert self.pending is not None
            self.pending.remaining -= 1
        else:
            raise RuntimeError(f"unhandled decision: {decision.kind}")

        self.decision_id += 1
        return self.advance_until_decision()

    def _buy_card(self, pile_name: str) -> None:
        top = self.supply[pile_name].top
        if top is None:
            raise IllegalAction(f"empty pile: {pile_name}")
        cost = CARDS[top].cost
        self.coins -= cost.coins
        self.buys -= 1
        self.current_player.debt += cost.debt
        self._gain(self.active_player, pile_name, bought=True)

    def _buy_wedding(self) -> None:
        self.coins -= self.WEDDING_COIN_COST
        self.buys -= 1
        self.current_player.debt += self.WEDDING_DEBT_COST
        self.current_player.vp_tokens += 1
        self._gain(self.active_player, "Gold")
        self.history.append(f"p{self.active_player}:event:Wedding")

    def _cleanup(self) -> None:
        player = self.current_player
        player.discard.extend(player.hand)
        player.discard.extend(player.in_play)
        player.hand = []
        player.in_play = []
        player.turns_taken += 1
        self._draw(player, 5)

        if self._normal_game_end() or sum(p.turns_taken for p in self.players) >= self.max_turns:
            self.terminal = True
            self.truncated = not self._normal_game_end()
            self.phase = Phase.TERMINAL
            self.history.append(
                f"terminal:scores:{self.score(0)}:{self.score(1)}"
            )
            return

        self.active_player = 1 - self.active_player
        self.turn_number += 1
        self.phase = Phase.ACTION
        self.actions = 1
        self.buys = 1
        self.coins = 0

    def _normal_game_end(self) -> bool:
        if self.supply["Province"].empty:
            return True
        return sum(pile.empty for pile in self.supply.values()) >= 3

    def tower_points(self, player_index: int) -> int:
        counts = self.players[player_index].counts()
        points = 0
        for pile_name, pile in self.supply.items():
            if not pile.empty:
                continue
            for card_name in pile_card_names(pile_name):
                if not CARDS[card_name].is_type(CardType.VICTORY):
                    points += counts[card_name]
        return points

    def score(self, player_index: int) -> int:
        player = self.players[player_index]
        card_points = sum(CARDS[name].victory_points for name in player.all_cards())
        return card_points + player.vp_tokens + self.tower_points(player_index)

    def outcome(self, player_index: int) -> float:
        """Return +1/-1/0 from ``player_index``'s perspective."""
        if not self.terminal:
            raise RuntimeError("outcome is only defined after the game ends")
        other = 1 - player_index
        scores = (self.score(player_index), self.score(other))
        if scores[0] != scores[1]:
            return 1.0 if scores[0] > scores[1] else -1.0
        turns = (
            self.players[player_index].turns_taken,
            self.players[other].turns_taken,
        )
        if turns[0] != turns[1]:
            return 1.0 if turns[0] < turns[1] else -1.0
        return 0.0

    def observation(self, player_index: int, decision: Decision | None = None) -> Observation:
        if decision is None:
            decision = self.advance_until_decision()
        if decision is None:
            raise RuntimeError("no observation is available after termination")
        if decision.player != player_index:
            raise ValueError("only the deciding player receives an observation")

        player = self.players[player_index]
        opponent = self.players[1 - player_index]
        piles = tuple(
            PileObservation(name, self.supply[name].top, len(self.supply[name].cards), self.supply[name].tokens)
            for name in PILE_ORDER
        )
        return Observation(
            ruleset_id=RULESET_ID,
            player=player_index,
            active_player=self.active_player,
            turn_number=self.turn_number,
            phase=self.phase,
            decision_kind=decision.kind,
            actions=self.actions,
            buys=self.buys,
            coins=self.coins,
            debt=player.debt,
            vp_tokens=player.vp_tokens,
            opponent_debt=opponent.debt,
            opponent_vp_tokens=opponent.vp_tokens,
            hand=tuple(sorted(player.hand)),
            own_counts=tuple(sorted(player.counts().items())),
            opponent_counts=tuple(sorted(opponent.counts().items())),
            piles=piles,
        )

    def play(self, policies: Iterable[object]) -> tuple[int, int]:
        """Play to completion using objects with ``choose(game, decision)``."""
        policy_list = list(policies)
        if len(policy_list) != 2:
            raise ValueError("exactly two policies are required")
        while (decision := self.advance_until_decision()) is not None:
            action = policy_list[decision.player].choose(self, decision)
            self.step(action)
        return self.score(0), self.score(1)

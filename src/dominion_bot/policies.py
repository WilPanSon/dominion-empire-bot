"""Scripted policies and paired tournament evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import random

from .engine import Decision, DecisionKind, Game


class RandomPolicy:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def choose(self, game: Game, decision: Decision) -> str:
        return self.rng.choice(decision.actions)


class BigMoneyPolicy:
    """A conventional money baseline with sensible forced-choice handling."""

    def choose(self, game: Game, decision: Decision) -> str:
        legal = set(decision.actions)
        player = game.players[decision.player]

        if decision.kind == DecisionKind.ACTION:
            return "end_action"
        if decision.kind == DecisionKind.CHAPEL_TRASH:
            for name in ("Curse", "Estate", "Copper"):
                action = f"chapel_trash:{name}"
                if action in legal:
                    return action
            return "chapel_done"
        if decision.kind == DecisionKind.FORUM_DISCARD:
            return _first_legal_card(
                legal,
                "forum_discard:",
                ("Curse", "Estate", "Copper", "Duchy", "Province"),
            )
        if decision.kind == DecisionKind.WORKSHOP_GAIN:
            return _first_legal(
                legal,
                ("workshop_gain:Silver", "workshop_gain:Smithy", "workshop_gain:Village"),
            )

        provinces = len(game.supply["Province"].cards)
        priorities = ["buy:Province"]
        if provinces <= 4:
            priorities.append("buy:Duchy")
        if provinces <= 2:
            priorities.append("buy:Estate")
        priorities.extend(("buy:Gold", "buy:Silver", "end_buy"))
        return _first_legal(legal, priorities)


class EnginePolicy(BigMoneyPolicy):
    """A transparent engine-building baseline for RL evaluation."""

    ACTION_PRIORITY = (
        "Chapel",
        "Village",
        "City Quarter",
        "Patrician",
        "Market",
        "Forum",
        "Farmers' Market",
        "Workshop",
        "Smithy",
        "Emporium",
        "Villa",
    )

    def choose(self, game: Game, decision: Decision) -> str:
        legal = set(decision.actions)
        player = game.players[decision.player]
        counts = player.counts()

        if decision.kind == DecisionKind.ACTION:
            for card_name in self.ACTION_PRIORITY:
                action = f"play:{card_name}"
                if action in legal:
                    if card_name == "Chapel" and not any(
                        name in player.hand for name in ("Curse", "Estate", "Copper")
                    ):
                        continue
                    return action
            return "end_action"

        if decision.kind == DecisionKind.CHAPEL_TRASH:
            for name in ("Curse", "Estate"):
                action = f"chapel_trash:{name}"
                if action in legal:
                    return action
            # Preserve at least three terminal Treasure cards.
            if counts["Copper"] + counts["Silver"] + counts["Gold"] > 3:
                if "chapel_trash:Copper" in legal:
                    return "chapel_trash:Copper"
            return "chapel_done"

        if decision.kind == DecisionKind.FORUM_DISCARD:
            return _first_legal_card(
                legal,
                "forum_discard:",
                ("Curse", "Estate", "Copper", "Duchy", "Province", "Chapel"),
            )

        if decision.kind == DecisionKind.WORKSHOP_GAIN:
            village_count = counts["Village"] + counts["City Quarter"]
            terminal_count = counts["Smithy"] + counts["Forum"]
            priorities: list[str] = []
            if counts["Chapel"] == 0 and game.turn_number <= 8:
                priorities.append("workshop_gain:Chapel")
            if village_count <= terminal_count:
                priorities.append("workshop_gain:Village")
            priorities.extend(
                ("workshop_gain:Smithy", "workshop_gain:Silver", "workshop_gain:Patrician/Emporium")
            )
            return _first_legal(legal, priorities)

        provinces = len(game.supply["Province"].cards)
        if "buy:Province" in legal:
            return "buy:Province"
        if provinces <= 4 and "buy:Duchy" in legal:
            return "buy:Duchy"
        if provinces <= 2 and "buy:Estate" in legal:
            return "buy:Estate"

        priorities: list[str] = []
        if counts["Chapel"] == 0 and game.turn_number <= 8:
            priorities.append("buy:Chapel")
        priorities.extend(
            (
                "buy:Gold",
                *(('buy:Smithy',) if counts["Smithy"] == 0 else ()),
                *(('buy:Village',) if counts["Smithy"] > 0 and counts["Village"] == 0 else ()),
                *(('buy:Market',) if counts["Market"] == 0 else ()),
                "buy:Silver",
                "event:Wedding",
                "buy:Patrician/Emporium",
                "end_buy",
            )
        )
        return _first_legal(legal, priorities)


def _first_legal(legal: set[str], priorities: tuple[str, ...] | list[str]) -> str:
    for action in priorities:
        if action in legal:
            return action
    return sorted(legal)[0]


def _first_legal_card(
    legal: set[str], prefix: str, priorities: tuple[str, ...]
) -> str:
    for name in priorities:
        preferred = prefix + name
        if preferred in legal:
            return preferred
    return sorted(legal)[0]


@dataclass(frozen=True, slots=True)
class MatchResult:
    games: int
    policy_a_wins: int
    policy_b_wins: int
    ties: int
    mean_score_a: float
    mean_score_b: float

    @property
    def policy_a_score_rate(self) -> float:
        return (self.policy_a_wins + 0.5 * self.ties) / self.games


def paired_match(
    policy_a: object,
    policy_b: object,
    *,
    pairs: int = 50,
    seed: int = 0,
    max_turns: int = 200,
) -> MatchResult:
    """Evaluate both seat assignments for every environment seed."""
    a_wins = b_wins = ties = 0
    score_a_total = score_b_total = 0
    for offset in range(pairs):
        game_seed = seed + offset
        for a_seat in (0, 1):
            policies = [policy_b, policy_b]
            policies[a_seat] = policy_a
            game = Game(game_seed, max_turns=max_turns)
            scores = game.play(policies)
            score_a = scores[a_seat]
            score_b = scores[1 - a_seat]
            score_a_total += score_a
            score_b_total += score_b
            outcome = game.outcome(a_seat)
            if outcome > 0:
                a_wins += 1
            elif outcome < 0:
                b_wins += 1
            else:
                ties += 1
    games = pairs * 2
    return MatchResult(
        games,
        a_wins,
        b_wins,
        ties,
        score_a_total / games,
        score_b_total / games,
    )

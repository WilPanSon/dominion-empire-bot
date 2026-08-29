"""Card metadata for the compact RL training kingdom.

Card effects live in :mod:`dominion_bot.engine`; this module intentionally holds
only immutable metadata so observations and policies cannot mutate rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag, auto


class CardType(IntFlag):
    ACTION = auto()
    TREASURE = auto()
    VICTORY = auto()
    CURSE = auto()
    GATHERING = auto()


@dataclass(frozen=True, slots=True)
class Cost:
    coins: int = 0
    debt: int = 0


@dataclass(frozen=True, slots=True)
class Card:
    name: str
    cost: Cost
    types: CardType
    treasure: int = 0
    victory_points: int = 0

    def is_type(self, card_type: CardType) -> bool:
        return bool(self.types & card_type)


def _card(
    name: str,
    coins: int,
    types: CardType,
    *,
    debt: int = 0,
    treasure: int = 0,
    victory_points: int = 0,
) -> Card:
    return Card(name, Cost(coins, debt), types, treasure, victory_points)


CARDS: dict[str, Card] = {
    # Basic cards.
    "Copper": _card("Copper", 0, CardType.TREASURE, treasure=1),
    "Silver": _card("Silver", 3, CardType.TREASURE, treasure=2),
    "Gold": _card("Gold", 6, CardType.TREASURE, treasure=3),
    "Estate": _card("Estate", 2, CardType.VICTORY, victory_points=1),
    "Duchy": _card("Duchy", 5, CardType.VICTORY, victory_points=3),
    "Province": _card("Province", 8, CardType.VICTORY, victory_points=6),
    "Curse": _card("Curse", 0, CardType.CURSE, victory_points=-1),
    # Base-game cards used to make the learning curriculum less brittle.
    "Village": _card("Village", 3, CardType.ACTION),
    "Smithy": _card("Smithy", 4, CardType.ACTION),
    "Market": _card("Market", 5, CardType.ACTION),
    "Chapel": _card("Chapel", 2, CardType.ACTION),
    "Workshop": _card("Workshop", 3, CardType.ACTION),
    # Empires cards.
    "City Quarter": _card("City Quarter", 0, CardType.ACTION, debt=8),
    "Farmers' Market": _card(
        "Farmers' Market", 3, CardType.ACTION | CardType.GATHERING
    ),
    "Forum": _card("Forum", 5, CardType.ACTION),
    "Villa": _card("Villa", 4, CardType.ACTION),
    "Patrician": _card("Patrician", 2, CardType.ACTION),
    "Emporium": _card("Emporium", 5, CardType.ACTION),
}


BASIC_PILES = ("Copper", "Silver", "Gold", "Estate", "Duchy", "Province", "Curse")
KINGDOM_PILES = (
    "Village",
    "Smithy",
    "Market",
    "Chapel",
    "Workshop",
    "City Quarter",
    "Farmers' Market",
    "Forum",
    "Villa",
    "Patrician/Emporium",
)
PILE_ORDER = BASIC_PILES + KINGDOM_PILES
EVENTS = ("Wedding",)
LANDMARKS = ("Tower",)
RULESET_ID = "rl-intro-2021"


def pile_card_names(pile_name: str) -> tuple[str, ...]:
    if pile_name == "Patrician/Emporium":
        return ("Patrician", "Emporium")
    return (pile_name,)


def pile_for_card(card_name: str) -> str:
    if card_name in {"Patrician", "Emporium"}:
        return "Patrician/Emporium"
    return card_name


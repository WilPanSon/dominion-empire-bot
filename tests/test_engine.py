from __future__ import annotations

import unittest
from collections import Counter

from dominion_bot.engine import DecisionKind, Game, IllegalAction, Phase
from dominion_bot.policies import BigMoneyPolicy, RandomPolicy


class EngineTests(unittest.TestCase):
    def test_initial_state_and_split_pile(self) -> None:
        game = Game(7)
        self.assertEqual(game.supply["Patrician/Emporium"].top, "Patrician")
        for _ in range(5):
            self.assertEqual(game.supply["Patrician/Emporium"].take(), "Patrician")
        self.assertEqual(game.supply["Patrician/Emporium"].top, "Emporium")
        self.assertEqual(len(game.players[0].hand), 5)
        self.assertEqual(len(game.players[1].hand), 5)

    def test_same_seed_and_policies_produce_identical_replay(self) -> None:
        first = Game(19)
        second = Game(19)
        first.play((BigMoneyPolicy(), BigMoneyPolicy()))
        second.play((BigMoneyPolicy(), BigMoneyPolicy()))
        self.assertEqual(first.history, second.history)
        self.assertEqual((first.score(0), first.score(1)), (second.score(0), second.score(1)))

    def test_illegal_action_is_rejected(self) -> None:
        game = Game(0)
        with self.assertRaises(IllegalAction):
            game.step("event:Not A Real Event")

    def test_wedding_adds_vp_gold_and_debt(self) -> None:
        game = Game(0)
        player = game.current_player
        player.hand = ["Gold", "Gold"]
        player.deck = []
        player.discard = []
        game.actions = 0
        decision = game.advance_until_decision()
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertIn("event:Wedding", decision.actions)
        game.step("event:Wedding")
        self.assertEqual(player.vp_tokens, 1)
        self.assertEqual(player.debt, 1)  # 3 Debt minus the 2 coins left after paying $4.
        self.assertEqual(player.counts()["Gold"], 3)

    def test_city_quarter_can_be_bought_for_debt(self) -> None:
        game = Game(0)
        player = game.current_player
        player.hand = []
        game.actions = 0
        decision = game.advance_until_decision()
        assert decision is not None
        self.assertIn("buy:City Quarter", decision.actions)
        game.step("buy:City Quarter")
        self.assertEqual(player.debt, 8)
        self.assertEqual(player.counts()["City Quarter"], 1)

    def test_villa_returns_buy_phase_to_action_phase(self) -> None:
        game = Game(0)
        player = game.current_player
        game.phase = Phase.BUY
        game.actions = 0
        game._gain(0, "Villa", bought=True)
        self.assertEqual(game.phase, Phase.ACTION)
        self.assertEqual(game.actions, 1)
        self.assertIn("Villa", player.hand)

    def test_farmers_market_gathers_then_cashses_tokens(self) -> None:
        game = Game(0)
        player = game.current_player
        player.hand = []
        player.deck = []
        player.discard = []
        player.in_play = []
        for _ in range(5):
            gained = game._gain(0, "Farmers' Market")
            self.assertEqual(gained, "Farmers' Market")
            player.discard.remove("Farmers' Market")
            player.hand.append("Farmers' Market")
            game.actions = 1
            game._play_action("Farmers' Market")
        self.assertEqual(game.supply["Farmers' Market"].tokens, 0)
        self.assertEqual(player.vp_tokens, 4)
        self.assertEqual(game.trash.count("Farmers' Market"), 1)

    def test_tower_scores_non_victory_cards_from_empty_pile(self) -> None:
        game = Game(0)
        player = game.players[0]
        player.discard.extend(("Workshop", "Workshop"))
        game.supply["Workshop"].cards = []
        normal_points = sum(
            # Tower is the only dynamic score in this ruleset.
            1 if name == "Estate" else 0 for name in player.all_cards()
        )
        self.assertEqual(game.tower_points(0), 2)
        self.assertEqual(game.score(0), normal_points + 2)

    def test_random_games_finish_without_illegal_actions(self) -> None:
        for seed in range(25):
            game = Game(seed, max_turns=120)
            initial = self._card_totals(game)
            game.play((RandomPolicy(seed), RandomPolicy(seed + 10_000)))
            self.assertTrue(game.terminal)
            self.assertEqual(game.phase, Phase.TERMINAL)
            self.assertEqual(game.outcome(0), -game.outcome(1))
            self.assertEqual(self._card_totals(game), initial)

    @staticmethod
    def _card_totals(game: Game) -> Counter[str]:
        totals: Counter[str] = Counter(game.trash)
        for player in game.players:
            totals.update(player.all_cards())
        for pile in game.supply.values():
            totals.update(pile.cards)
        return totals


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import random
import unittest

from dominion_bot.encoding import ACTION_VOCAB, FeatureEncoder
from dominion_bot.env import DominionEnv
from dominion_bot.engine import IllegalAction


class EnvironmentTests(unittest.TestCase):
    def test_observation_size_and_mask(self) -> None:
        env = DominionEnv()
        state = env.reset(3)
        self.assertEqual(len(state.observation), env.observation_size)
        self.assertEqual(len(state.action_mask), env.action_size)
        self.assertGreater(sum(state.action_mask), 0)

    def test_masked_action_is_rejected(self) -> None:
        env = DominionEnv()
        state = env.reset(1)
        invalid = next(index for index, valid in enumerate(state.action_mask) if not valid)
        with self.assertRaises(IllegalAction):
            env.step(invalid)

    def test_random_masked_rollout_reaches_terminal_rewards(self) -> None:
        env = DominionEnv(max_turns=100)
        rng = random.Random(4)
        state = env.reset(4)
        while True:
            legal = [index for index, valid in enumerate(state.action_mask) if valid]
            result = env.step(rng.choice(legal))
            if result.done:
                self.assertEqual(result.rewards[0], -result.rewards[1])
                break
            assert result.state is not None
            state = result.state

    def test_action_vocabulary_covers_live_decisions(self) -> None:
        encoder = FeatureEncoder()
        env = DominionEnv(encoder=encoder, max_turns=80)
        rng = random.Random(11)
        for seed in range(10):
            state = env.reset(seed)
            while True:
                legal = [index for index, valid in enumerate(state.action_mask) if valid]
                self.assertTrue(all(ACTION_VOCAB[index] for index in legal))
                result = env.step(rng.choice(legal))
                if result.done:
                    break
                assert result.state is not None
                state = result.state

    def test_encoder_does_not_expose_opponent_hidden_zones_or_deck_order(self) -> None:
        env = DominionEnv()
        state_before = env.reset(31)
        assert env.game is not None
        opponent = env.game.players[1]
        opponent.deck.reverse()
        if opponent.deck and opponent.hand:
            opponent.deck[-1], opponent.hand[-1] = opponent.hand[-1], opponent.deck[-1]
        decision = env.game.advance_until_decision()
        assert decision is not None
        state_after = env.encoder.encode(env.game, decision)
        self.assertEqual(state_before.observation, state_after.observation)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import math
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipUnless(torch is not None, "optional PyTorch dependency is not installed")
class PPOTests(unittest.TestCase):
    def test_collect_and_update_smoke(self) -> None:
        from dominion_bot.env import DominionEnv
        from dominion_bot.rl.model import MaskedActorCritic
        from dominion_bot.rl.ppo import PPOConfig, collect_episode, ppo_update

        assert torch is not None
        torch.manual_seed(5)
        device = torch.device("cpu")
        env = DominionEnv(max_turns=40)
        model = MaskedActorCritic(env.observation_size, env.action_size, hidden_size=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        samples = collect_episode(env, model, seed=5, device=device)
        self.assertGreater(len(samples), 0)
        config = PPOConfig(hidden_size=16, update_epochs=1, minibatch_size=128)
        metrics = ppo_update(model, optimizer, samples, config, device=device)
        self.assertEqual(metrics.samples, len(samples))
        self.assertTrue(math.isfinite(metrics.policy_loss))
        self.assertTrue(math.isfinite(metrics.value_loss))
        self.assertTrue(math.isfinite(metrics.entropy))


if __name__ == "__main__":
    unittest.main()


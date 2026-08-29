# Dominion Empires Bot

A rules-first reinforcement-learning prototype for two-player Dominion. The core
engine has no third-party dependencies; the optional training command uses
PyTorch to run masked, shared-policy PPO through self-play.

The included `rl_intro` setup is deliberately smaller than the complete Empires
expansion while exercising the mechanics the learning framework needs:

- basic Treasure and Victory piles;
- Debt through City Quarter and the Wedding Event;
- the Tower Landmark;
- the Patrician/Emporium split pile;
- Gathering tokens through Farmers' Market;
- on-buy and on-gain effects through Forum and Villa;
- contextual trash, discard, and gain decisions.

The following kingdom piles are present: Village, Smithy, Market, Chapel,
Workshop, City Quarter, Farmers' Market, Forum, Villa, and
Patrician/Emporium.

## Run

The repository uses a `src/` layout. Either install it or set `PYTHONPATH`:

```bash
python3 -m pip install -e .
dominion-bot evaluate --games 100
dominion-bot replay --seed 7
```

To train the neural policy:

```bash
python3 -m pip install -e '.[rl]'
dominion-bot train --iterations 100 --episodes-per-iteration 64
dominion-bot evaluate-checkpoint checkpoints/dominion_ppo.pt --games 100
```

Run dependency-free tests with:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Design contract

`Game.advance_until_decision()` resolves deterministic work until a player has
a real choice. `Game.step(action_key)` validates and applies one of the legal
choices. The RL adapter converts the decision into a fixed observation vector
and a mask over a stable action vocabulary. Hidden hands and shuffled deck
order never enter an opponent's observation.

The simulator is versioned as `rl-intro-2021`. Do not treat it as an
implementation of every Dominion or Empires card; new cards should be added
with rule tests before being placed in a training kingdom.

Rules reference: [Rio Grande Games, Dominion: Empires (2021)](https://www.riograndegames.com/wp-content/uploads/2022/03/Dominion-Rules-Empires.pdf).

"""Command-line interface for simulation, evaluation, and training."""

from __future__ import annotations

import argparse
from pathlib import Path

from .engine import Game
from .policies import BigMoneyPolicy, EnginePolicy, RandomPolicy, paired_match


def _policy(name: str, seed: int) -> object:
    if name == "random":
        return RandomPolicy(seed)
    if name == "big-money":
        return BigMoneyPolicy()
    if name == "engine":
        return EnginePolicy()
    raise ValueError(name)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dominion-bot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser("evaluate", help="run a paired tournament")
    evaluate.add_argument("--policy", choices=("random", "big-money", "engine"), default="engine")
    evaluate.add_argument("--opponent", choices=("random", "big-money", "engine"), default="big-money")
    evaluate.add_argument("--games", type=int, default=100, help="even number of games")
    evaluate.add_argument("--seed", type=int, default=0)

    replay = subparsers.add_parser("replay", help="print a deterministic game log")
    replay.add_argument("--policy", choices=("random", "big-money", "engine"), default="engine")
    replay.add_argument("--opponent", choices=("random", "big-money", "engine"), default="big-money")
    replay.add_argument("--seed", type=int, default=0)

    train_parser = subparsers.add_parser("train", help="train masked PPO through self-play")
    train_parser.add_argument("--iterations", type=int, default=100)
    train_parser.add_argument("--episodes-per-iteration", type=int, default=64)
    train_parser.add_argument("--minibatch-size", type=int, default=512)
    train_parser.add_argument("--hidden-size", type=int, default=256)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/dominion_ppo.pt"))

    neural = subparsers.add_parser(
        "evaluate-checkpoint", help="evaluate a trained PPO checkpoint"
    )
    neural.add_argument("checkpoint", type=Path)
    neural.add_argument(
        "--opponent", choices=("random", "big-money", "engine"), default="big-money"
    )
    neural.add_argument("--games", type=int, default=100)
    neural.add_argument("--seed", type=int, default=0)
    neural.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "evaluate":
        if args.games <= 0 or args.games % 2:
            raise SystemExit("--games must be a positive even number")
        result = paired_match(
            _policy(args.policy, args.seed),
            _policy(args.opponent, args.seed + 1),
            pairs=args.games // 2,
            seed=args.seed,
        )
        print(
            f"{args.policy} vs {args.opponent}: "
            f"{result.policy_a_wins}-{result.ties}-{result.policy_b_wins}; "
            f"score rate={result.policy_a_score_rate:.3f}; "
            f"mean VP={result.mean_score_a:.2f}-{result.mean_score_b:.2f}"
        )
        return 0

    if args.command == "replay":
        game = Game(args.seed)
        scores = game.play(
            [_policy(args.policy, args.seed), _policy(args.opponent, args.seed + 1)]
        )
        print("\n".join(game.history))
        print(
            f"scores={scores}; turns="
            f"{tuple(player.turns_taken for player in game.players)}; "
            f"truncated={game.truncated}"
        )
        return 0

    if args.command == "train":
        try:
            from .rl.ppo import PPOConfig, train
        except ModuleNotFoundError as exc:
            if exc.name == "torch":
                raise SystemExit(
                    "Training requires PyTorch. Install with: "
                    "python3 -m pip install -e '.[rl]'"
                ) from exc
            raise
        config = PPOConfig(
            hidden_size=args.hidden_size,
            episodes_per_iteration=args.episodes_per_iteration,
            minibatch_size=args.minibatch_size,
        )
        train(
            iterations=args.iterations,
            config=config,
            seed=args.seed,
            checkpoint_path=args.checkpoint,
            device_name=args.device,
        )
        return 0

    if args.command == "evaluate-checkpoint":
        if args.games <= 0 or args.games % 2:
            raise SystemExit("--games must be a positive even number")
        try:
            from .rl.ppo import NeuralPolicy, load_checkpoint, select_device
        except ModuleNotFoundError as exc:
            if exc.name == "torch":
                raise SystemExit(
                    "Checkpoint evaluation requires PyTorch. Install with: "
                    "python3 -m pip install -e '.[rl]'"
                ) from exc
            raise
        device = select_device(args.device)
        model, payload = load_checkpoint(args.checkpoint, device=device)
        result = paired_match(
            NeuralPolicy(model, device=device),
            _policy(args.opponent, args.seed + 1),
            pairs=args.games // 2,
            seed=args.seed,
        )
        print(
            f"checkpoint iteration {payload['iteration']} vs {args.opponent}: "
            f"{result.policy_a_wins}-{result.ties}-{result.policy_b_wins}; "
            f"score rate={result.policy_a_score_rate:.3f}; "
            f"mean VP={result.mean_score_a:.2f}-{result.mean_score_b:.2f}"
        )
        return 0

    return 2

#!/usr/bin/env python
"""
PPO CNN+LSTM — Entrenamiento con política recurrente.

Entrena un agente PPO con arquitectura CNN + LSTM contra el HeuristicBot.
Usa **exactamente** la misma información de observación, los mismos hiperparámetros
PPO (GAE, clipping, pérdidas) y las mismas seeds que el baseline MLP, para que
ambos runs sean directamente comparables en TensorBoard.

Uso
----
    # Entrenamiento rápido (50 k steps)
    python examples/train_ppo_cnn_lstm.py

    # Entrenamiento largo
    python examples/train_ppo_cnn_lstm.py --timesteps 500000

    # Cambiar longitud de secuencia LSTM
    python examples/train_ppo_cnn_lstm.py --seq-len 32

    # Comparar ambos en TensorBoard
    tensorboard --logdir runs/
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Ensure local package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3.common.vec_env import DummyVecEnv

from clash_royale_engine.env.gymnasium_env import ClashRoyaleEnv
from clash_royale_engine.models.cnn_lstm_policy import CnnLstmPolicy
from clash_royale_engine.models.recurrent_ppo import RecurrentPPO


# ══════════════════════════════════════════════════════════════════════════════
# Env factory (identical to baseline)
# ══════════════════════════════════════════════════════════════════════════════


def make_env(
    rank: int = 0,
    reward_shaping: str = "dense",
    time_limit: float = 180.0,
    fog_of_war: bool = True,
) -> ClashRoyaleEnv:
    """Create a CR environment instance with standard config."""
    return ClashRoyaleEnv(
        reward_shaping=reward_shaping,
        time_limit=time_limit,
        fog_of_war=fog_of_war,
        seed=rank,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation (carries LSTM hidden state across steps)
# ══════════════════════════════════════════════════════════════════════════════


def evaluate(
    trainer: RecurrentPPO, n_episodes: int = 20, time_limit: float = 180.0
) -> dict:
    """Evaluate the trained agent against HeuristicBot."""
    env = ClashRoyaleEnv(
        reward_shaping="sparse",
        time_limit=time_limit,
        fog_of_war=True,
    )

    wins, losses, draws = 0, 0, 0
    total_rewards: list[float] = []
    episode_lengths: list[int] = []

    for _ in range(n_episodes):
        obs, info = env.reset()
        hidden = trainer.policy.initial_state(1, trainer.device)
        done = False
        ep_reward = 0.0
        steps = 0

        while not done:
            action, hidden = trainer.predict(obs, hidden, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1
            done = terminated or truncated

        winner = env.engine.get_winner()
        if winner == 0:
            wins += 1
        elif winner == 1:
            losses += 1
        else:
            draws += 1

        total_rewards.append(ep_reward)
        episode_lengths.append(steps)

    env.close()

    return {
        "n_episodes": n_episodes,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / n_episodes,
        "mean_reward": float(np.mean(total_rewards)),
        "std_reward": float(np.std(total_rewards)),
        "mean_ep_length": float(np.mean(episode_lengths)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint callback
# ══════════════════════════════════════════════════════════════════════════════

_last_ckpt_step = 0


def checkpoint_callback(
    trainer: RecurrentPPO,
    iteration: int,
    *,
    save_dir: str = "",
    save_freq: int = 10_000,
) -> None:
    """Save a checkpoint every ``save_freq`` timesteps."""
    global _last_ckpt_step  # noqa: PLW0603
    if trainer.num_timesteps - _last_ckpt_step >= save_freq:
        path = os.path.join(save_dir, f"ppo_cnn_lstm_{trainer.num_timesteps}.pt")
        trainer.save(path)
        _last_ckpt_step = trainer.num_timesteps


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PPO CNN+LSTM para Clash Royale Engine"
    )
    p.add_argument(
        "--timesteps", type=int, default=50_000,
        help="Total de timesteps de entrenamiento (default: 50000)",
    )
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument(
        "--reward", type=str, default="dense", choices=["sparse", "dense"],
    )
    p.add_argument("--time-limit", type=float, default=120.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--save-dir", type=str, default="runs")
    p.add_argument("--no-fog", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    # CNN+LSTM-specific
    p.add_argument(
        "--seq-len", type=int, default=16,
        help="LSTM sequence length for BPTT (default: 16)",
    )
    p.add_argument(
        "--lstm-hidden", type=int, default=128,
        help="LSTM hidden size (default: 128)",
    )
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    args = parse_args()

    # ── Directories ───────────────────────────────────────────────────
    run_name = f"ppo_cnn_lstm_{int(time.time())}"
    log_dir = os.path.join(args.save_dir, run_name)
    model_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(model_dir, exist_ok=True)

    print("=" * 60)
    print("  PPO CNN+LSTM — Clash Royale Engine")
    print("=" * 60)
    print(f"  Timesteps:      {args.timesteps:,}")
    print(f"  Entornos:       {args.n_envs}")
    print(f"  Reward:         {args.reward}")
    print(f"  Time limit:     {args.time_limit}s")
    print(f"  Learning rate:  {args.lr}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  N steps:        {args.n_steps}")
    print(f"  Epochs:         {args.n_epochs}")
    print(f"  Gamma:          {args.gamma}")
    print(f"  GAE lambda:     {args.gae_lambda}")
    print(f"  Clip range:     {args.clip_range}")
    print(f"  Entropy coef:   {args.ent_coef}")
    print(f"  Fog of war:     {not args.no_fog}")
    print(f"  Seed:           {args.seed}")
    print(f"  Seq len (LSTM): {args.seq_len}")
    print(f"  LSTM hidden:    {args.lstm_hidden}")
    print(f"  Log dir:        {log_dir}")
    print("=" * 60)

    # ── Vectorised environments (same as baseline) ────────────────────
    fog = not args.no_fog

    def _make_env(rank: int):  # noqa: ANN202
        def _init() -> ClashRoyaleEnv:
            return make_env(
                rank=rank,
                reward_shaping=args.reward,
                time_limit=args.time_limit,
                fog_of_war=fog,
            )
        return _init

    vec_env = DummyVecEnv([_make_env(i) for i in range(args.n_envs)])

    # ── Build policy ──────────────────────────────────────────────────
    n_actions = vec_env.action_space.n
    policy = CnnLstmPolicy(
        n_actions=n_actions,
        cnn_out=128,
        scalar_out=64,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=1,
    )

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Arquitectura de la policy:")
    print(f"    {policy}")
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  Device: {device_str}")
    print()

    # ── Trainer ───────────────────────────────────────────────────────
    trainer = RecurrentPPO(
        policy=policy,
        env=vec_env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        seq_len=args.seq_len,
        device=device_str,
        seed=args.seed,
        tensorboard_log=os.path.join(log_dir, "PPO_CNN_LSTM_1"),
        verbose=1,
    )

    # ── Train ─────────────────────────────────────────────────────────
    global _last_ckpt_step  # noqa: PLW0603
    _last_ckpt_step = 0
    save_freq = max(args.timesteps // 5, 1000)

    def _cb(tr: RecurrentPPO, it: int) -> None:
        checkpoint_callback(tr, it, save_dir=model_dir, save_freq=save_freq)

    t0 = time.perf_counter()
    trainer.learn(total_timesteps=args.timesteps, callback=_cb)
    elapsed = time.perf_counter() - t0

    print(f"\n  Entrenamiento completado en {elapsed:.1f}s")
    print(f"  ({args.timesteps / elapsed:.0f} steps/s)")

    # ── Save final model ──────────────────────────────────────────────
    final_path = os.path.join(log_dir, "ppo_cnn_lstm_final.pt")
    trainer.save(final_path)

    # ── Evaluation ────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Evaluacion: {args.eval_episodes} episodios (deterministico)")
    print(f"{'=' * 60}")

    results = evaluate(
        trainer, n_episodes=args.eval_episodes, time_limit=args.time_limit
    )

    print(f"    Victorias:    {results['wins']}/{results['n_episodes']}")
    print(f"    Derrotas:     {results['losses']}/{results['n_episodes']}")
    print(f"    Empates:      {results['draws']}/{results['n_episodes']}")
    print(f"    Win rate:     {results['win_rate']:.1%}")
    print(f"    Reward medio: {results['mean_reward']:.4f} +/- {results['std_reward']:.4f}")
    print(f"    Ep length:    {results['mean_ep_length']:.0f} steps")
    print()

    # ── Save summary ──────────────────────────────────────────────────
    summary_path = os.path.join(log_dir, "eval_summary.txt")
    with open(summary_path, "w") as f:
        f.write("PPO CNN+LSTM — Evaluacion\n")
        f.write("=" * 40 + "\n")
        for k, v in results.items():
            f.write(f"{k}: {v}\n")
        f.write("\nHiperparametros:\n")
        for k, v in vars(args).items():
            f.write(f"  {k}: {v}\n")
    print(f"  Resumen guardado en: {summary_path}")

    vec_env.close()
    print("\n  Listo!")


if __name__ == "__main__":
    main()

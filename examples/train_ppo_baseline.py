#!/usr/bin/env python
"""
PPO Baseline — Entrenamiento básico con Stable-Baselines3.

Entrena un agente PPO (MlpPolicy) contra el HeuristicBot usando
reward shaping denso. Guarda el modelo, registra métricas en
TensorBoard y al final ejecuta 20 episodios de evaluación.

Uso
----
    # Entrenamiento rápido (50 k steps, ~10-15 min)
    python examples/train_ppo_baseline.py

    # Entrenamiento largo
    python examples/train_ppo_baseline.py --timesteps 500000

    # Ver métricas
    tensorboard --logdir runs/
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# Asegurar que el paquete local esté en el path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from clash_royale_engine.env.gymnasium_env import ClashRoyaleEnv


# ══════════════════════════════════════════════════════════════════════════════
# Fábrica de entornos
# ══════════════════════════════════════════════════════════════════════════════

def make_env(
    rank: int = 0,
    reward_shaping: str = "dense",
    time_limit: float = 180.0,
    fog_of_war: bool = True,
) -> ClashRoyaleEnv:
    """Crea una instancia del entorno CR con configuración estándar."""
    env = ClashRoyaleEnv(
        reward_shaping=reward_shaping,
        time_limit=time_limit,
        fog_of_war=fog_of_war,
        seed=rank,
    )
    return env


# ══════════════════════════════════════════════════════════════════════════════
# Callback personalizado: métricas de juego
# ══════════════════════════════════════════════════════════════════════════════

class GameMetricsCallback(BaseCallback):
    """
    Registra métricas específicas de Clash Royale en TensorBoard:
    - Tasa de victorias (ventana deslizante)
    - % de acciones válidas
    - Torres destruidas por episodio
    """

    def __init__(self, window_size: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.window_size = window_size
        self.episode_results: list[int] = []   # 1=win, 0=loss, -1=draw
        self.episode_valid: list[float] = []   # % acciones válidas
        self.episode_towers: list[int] = []    # torres destruidas
        self._ep_valid_count = 0
        self._ep_step_count = 0

    def _on_step(self) -> bool:
        # Acumular acciones válidas por step
        infos = self.locals.get("infos", [])
        for info in infos:
            self._ep_step_count += 1
            if info.get("action_valid", False):
                self._ep_valid_count += 1

        # Detectar fin de episodio
        dones = self.locals.get("dones", [])
        for i, done in enumerate(dones):
            if done:
                info = infos[i] if i < len(infos) else {}

                # Determinar resultado
                raw_state = info.get("raw_state", None)
                towers = info.get("towers_destroyed", 0)

                # Acceder al sub-env para obtener ganador
                try:
                    vec_env = self.training_env
                    sub_env = vec_env.envs[i]  # type: ignore[attr-defined]
                    # DummyVecEnv wraps envs; access the inner env
                    inner = getattr(sub_env, "env", sub_env)
                    winner = inner.engine.get_winner()
                except Exception:
                    winner = -1

                if winner == 0:
                    self.episode_results.append(1)
                elif winner == 1:
                    self.episode_results.append(0)
                else:
                    self.episode_results.append(-1)

                # Tasa de acciones válidas
                valid_pct = (
                    self._ep_valid_count / max(self._ep_step_count, 1)
                )
                self.episode_valid.append(valid_pct)
                self.episode_towers.append(towers)

                # Reset contadores
                self._ep_valid_count = 0
                self._ep_step_count = 0

                # Loguear cada N episodios
                n_eps = len(self.episode_results)
                if n_eps % 10 == 0:
                    recent = self.episode_results[-self.window_size :]
                    wins = sum(1 for r in recent if r == 1)
                    winrate = wins / len(recent)

                    recent_valid = self.episode_valid[-self.window_size :]
                    avg_valid = np.mean(recent_valid)

                    recent_towers = self.episode_towers[-self.window_size :]
                    avg_towers = np.mean(recent_towers)

                    self.logger.record("game/win_rate", winrate)
                    self.logger.record("game/valid_action_pct", avg_valid)
                    self.logger.record("game/avg_towers_destroyed", avg_towers)
                    self.logger.record("game/episodes_total", n_eps)

                    if self.verbose >= 1:
                        print(
                            f"  [Ep {n_eps:>5d}] "
                            f"WinRate={winrate:.1%}  "
                            f"ValidAct={avg_valid:.1%}  "
                            f"Towers={avg_towers:.2f}"
                        )

        return True


# ══════════════════════════════════════════════════════════════════════════════
# Evaluación
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(model: PPO, n_episodes: int = 20, time_limit: float = 180.0) -> dict:
    """Evalúa el modelo entrenado contra HeuristicBot."""
    env = ClashRoyaleEnv(
        reward_shaping="sparse",
        time_limit=time_limit,
        fog_of_war=True,
    )

    wins, losses, draws = 0, 0, 0
    total_rewards: list[float] = []
    episode_lengths: list[int] = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
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

    results = {
        "n_episodes": n_episodes,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / n_episodes,
        "mean_reward": float(np.mean(total_rewards)),
        "std_reward": float(np.std(total_rewards)),
        "mean_ep_length": float(np.mean(episode_lengths)),
    }
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO Baseline para Clash Royale Engine")
    p.add_argument("--timesteps", type=int, default=50_000,
                   help="Total de timesteps de entrenamiento (default: 50000)")
    p.add_argument("--n-envs", type=int, default=4,
                   help="Número de entornos paralelos (default: 4)")
    p.add_argument("--reward", type=str, default="dense",
                   choices=["sparse", "dense"],
                   help="Tipo de reward shaping (default: dense)")
    p.add_argument("--time-limit", type=float, default=120.0,
                   help="Duración de cada partida en segundos (default: 120)")
    p.add_argument("--lr", type=float, default=3e-4,
                   help="Learning rate (default: 3e-4)")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Minibatch size para PPO (default: 64)")
    p.add_argument("--n-steps", type=int, default=2048,
                   help="Steps por rollout por env (default: 2048)")
    p.add_argument("--n-epochs", type=int, default=10,
                   help="Épocas de optimización por rollout (default: 10)")
    p.add_argument("--gamma", type=float, default=0.99,
                   help="Factor de descuento (default: 0.99)")
    p.add_argument("--gae-lambda", type=float, default=0.95,
                   help="GAE lambda (default: 0.95)")
    p.add_argument("--clip-range", type=float, default=0.2,
                   help="PPO clip range (default: 0.2)")
    p.add_argument("--ent-coef", type=float, default=0.01,
                   help="Coeficiente de entropía (default: 0.01)")
    p.add_argument("--vf-coef", type=float, default=0.5,
                   help="Coeficiente de value function loss (default: 0.5)")
    p.add_argument("--max-grad-norm", type=float, default=0.5,
                   help="Max gradient norm (default: 0.5)")
    p.add_argument("--eval-episodes", type=int, default=20,
                   help="Episodios de evaluación final (default: 20)")
    p.add_argument("--save-dir", type=str, default="runs",
                   help="Directorio para logs y modelos (default: runs)")
    p.add_argument("--no-fog", action="store_true",
                   help="Desactivar fog of war (observación perfecta)")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed global (default: 42)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Directorios ───────────────────────────────────────────────────
    run_name = f"ppo_baseline_{int(time.time())}"
    log_dir = os.path.join(args.save_dir, run_name)
    model_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(model_dir, exist_ok=True)

    print("═" * 60)
    print("  PPO Baseline — Clash Royale Engine")
    print("═" * 60)
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
    print(f"  Log dir:        {log_dir}")
    print("═" * 60)

    # ── Crear entornos vectorizados ───────────────────────────────────
    fog = not args.no_fog

    def _make_env(rank: int):
        def _init():
            return make_env(
                rank=rank,
                reward_shaping=args.reward,
                time_limit=args.time_limit,
                fog_of_war=fog,
            )
        return _init

    vec_env = DummyVecEnv([_make_env(i) for i in range(args.n_envs)])

    # ── Crear modelo PPO ──────────────────────────────────────────────
    model = PPO(
        policy="MlpPolicy",
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
        verbose=1,
        seed=args.seed,
        tensorboard_log=log_dir,
        device="auto",  # usa CUDA si está disponible
    )

    print(f"\n  Arquitectura de la policy:")
    print(f"    {model.policy}")
    print(f"  Device: {model.device}")
    print()

    # ── Callbacks ─────────────────────────────────────────────────────
    game_cb = GameMetricsCallback(window_size=50, verbose=1)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.timesteps // 5, 1000),  # ~5 checkpoints
        save_path=model_dir,
        name_prefix="ppo_cr",
    )

    callbacks = [game_cb, checkpoint_cb]

    # ── Entrenamiento ─────────────────────────────────────────────────
    t0 = time.perf_counter()

    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    elapsed = time.perf_counter() - t0
    print(f"\n  Entrenamiento completado en {elapsed:.1f}s")
    print(f"  ({args.timesteps / elapsed:.0f} steps/s)")

    # ── Guardar modelo final ──────────────────────────────────────────
    final_path = os.path.join(log_dir, "ppo_cr_final")
    model.save(final_path)
    print(f"  Modelo guardado en: {final_path}.zip")

    # ── Evaluación ────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  Evaluación: {args.eval_episodes} episodios (determinístico)")
    print(f"{'═' * 60}")

    results = evaluate(model, n_episodes=args.eval_episodes, time_limit=args.time_limit)

    print(f"    Victorias:    {results['wins']}/{results['n_episodes']}")
    print(f"    Derrotas:     {results['losses']}/{results['n_episodes']}")
    print(f"    Empates:      {results['draws']}/{results['n_episodes']}")
    print(f"    Win rate:     {results['win_rate']:.1%}")
    print(f"    Reward medio: {results['mean_reward']:.4f} ± {results['std_reward']:.4f}")
    print(f"    Ep length:    {results['mean_ep_length']:.0f} steps")
    print()

    # ── Guardar resumen ───────────────────────────────────────────────
    summary_path = os.path.join(log_dir, "eval_summary.txt")
    with open(summary_path, "w") as f:
        f.write("PPO Baseline — Evaluación\n")
        f.write("=" * 40 + "\n")
        for k, v in results.items():
            f.write(f"{k}: {v}\n")
        f.write(f"\nHiperparámetros:\n")
        for k, v in vars(args).items():
            f.write(f"  {k}: {v}\n")
    print(f"  Resumen guardado en: {summary_path}")

    vec_env.close()
    print("\n  ¡Listo!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Ejemplo 2 — Usar el entorno Gymnasium con acciones aleatorias.

Muestra cómo:
  • Crear ClashRoyaleEnv (1 agente vs HeuristicBot).
  • Recorrer el loop reset / step / done estándar de Gymnasium.
  • Inspeccionar observaciones, rewards e info dict.
  • Comparar reward sparse vs dense.

Uso
----
    python examples/02_gymnasium_random_agent.py
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, ".")

from clash_royale_engine.env.gymnasium_env import ClashRoyaleEnv


def run_episode(env: ClashRoyaleEnv, label: str) -> None:
    obs, info = env.reset()
    total_reward = 0.0
    steps = 0
    valid_actions = 0

    t0 = time.perf_counter()

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1
        if info.get("action_valid", False):
            valid_actions += 1

        if terminated or truncated:
            break

    elapsed = time.perf_counter() - t0
    winner = env.engine.get_winner()

    print(f"\n  [{label}]")
    print(f"    Steps: {steps}")
    print(f"    Acciones válidas: {valid_actions} / {steps} ({100 * valid_actions / max(steps, 1):.1f}%)")
    print(f"    Reward total: {total_reward:.4f}")
    print(f"    Ganador: {'P0' if winner == 0 else 'P1' if winner == 1 else 'Empate'}")
    print(f"    Obs shape: {obs.shape}")
    print(f"    Tiempo real: {elapsed:.2f}s")


def main() -> None:
    print("═" * 60)
    print("  Gymnasium: agente aleatorio vs HeuristicBot")
    print("═" * 60)

    # ── Episodio con reward sparse ────────────────────────────────────
    env_sparse = ClashRoyaleEnv(
        reward_shaping="sparse",
        time_limit=60.0,  # partida corta para el demo
        fog_of_war=True,
    )
    run_episode(env_sparse, "Reward SPARSE")

    # ── Episodio con reward dense ─────────────────────────────────────
    env_dense = ClashRoyaleEnv(
        reward_shaping="dense",
        time_limit=60.0,
        fog_of_war=True,
    )
    run_episode(env_dense, "Reward DENSE")

    # ── Mostrar spaces ────────────────────────────────────────────────
    print("\n  Espacios:")
    print(f"    action_space  = {env_sparse.action_space}")
    print(f"    obs_space     = {env_sparse.observation_space}")
    print(f"    obs_space.shape = {env_sparse.observation_space.shape}")

    print("\n" + "═" * 60)


if __name__ == "__main__":
    main()

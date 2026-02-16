#!/usr/bin/env python
"""
Ejemplo 3 — Grabación de partida y extracción de 4 episodios de IL.

Muestra cómo:
  • Activar recording en el entorno Gymnasium (record=True).
  • Tras un episodio completo, obtener el GameRecord.
  • Extraer 4 episodios de Imitation Learning por simetría.
  • Convertir los episodios a arrays numpy listos para entrenamiento.

Los 4 episodios generados son:
  1. P0 original     — trayectoria del jugador 0, coordenadas tal cual.
  2. P1 y-flip       — trayectoria del jugador 1, volteada verticalmente
                        para normalizar su perspectiva al fondo del mapa.
  3. P0 x-flip       — espejo horizontal de la trayectoria de P0
                        (data augmentation).
  4. P1 y+x-flip     — ambas transformaciones aplicadas a P1.

Uso
----
    python examples/03_recording_and_il.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from clash_royale_engine.core.recorder import EpisodeExtractor
from clash_royale_engine.env.gymnasium_env import ClashRoyaleEnv


def main() -> None:
    print("═" * 60)
    print("  Grabación de partida + 4 episodios de IL")
    print("═" * 60)

    # ── Crear env con recording activado ──────────────────────────────
    env = ClashRoyaleEnv(
        record=True,
        fog_of_war=True,
        time_limit=30.0,   # partida corta para el demo
        reward_shaping="sparse",
    )

    # ── Jugar un episodio completo con acciones aleatorias ────────────
    obs, info = env.reset()
    done = False
    steps = 0

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1

    print(f"\n  Episodio finalizado en {steps} steps.")
    winner = env.engine.get_winner()
    print(f"  Ganador: {'P0' if winner == 0 else 'P1' if winner == 1 else 'Empate'}")

    # ── Finalizar grabación llamando a reset (o extraer directamente) ─
    env.reset()  # esto finaliza el GameRecord internamente
    record = env.get_game_record()

    if record is None:
        print("  ⚠ No se obtuvo GameRecord — algo falló.")
        return

    print(f"\n  GameRecord:")
    print(f"    Frames grabados: {record.total_frames}")
    print(f"    Ganador:         {record.winner}")
    print(f"    Deck P0:         {record.deck_p0}")
    print(f"    Deck P1:         {record.deck_p1}")

    # ── Extraer 4 episodios de IL ─────────────────────────────────────
    episodes = env.extract_il_episodes(record)

    labels = [
        "P0 original",
        "P1 y-flip (normalizado)",
        "P0 x-flip (espejo horizontal)",
        "P1 y+x-flip (ambas transformaciones)",
    ]

    print(f"\n  Episodios de IL extraídos: {len(episodes)}")
    for i, (ep, label) in enumerate(zip(episodes, labels)):
        print(f"    [{i}] {label}: {len(ep)} transiciones")

    # ── Verificar fog-of-war ──────────────────────────────────────────
    sample = episodes[0][0]
    print(f"\n  Ejemplo de transición (episodio 0, step 0):")
    print(f"    state shape:      {sample.state.shape}")
    print(f"    next_state shape: {sample.next_state.shape}")
    print(f"    action (int):     {sample.action}")
    print(f"    reward:           {sample.reward}")
    print(f"    done:             {sample.done}")
    print(f"    enemy_elixir (idx 1) = {sample.state[1]:.1f}  (debe ser 0 con fog_of_war)")

    # ── Convertir a numpy batch ───────────────────────────────────────
    batch = EpisodeExtractor.episodes_to_numpy(episodes)

    print(f"\n  Batch numpy (4 episodios concatenados):")
    for key, arr in batch.items():
        print(f"    {key:15s}  shape={str(arr.shape):15s}  dtype={arr.dtype}")

    total = batch["states"].shape[0]
    print(f"\n  Total de transiciones para IL: {total}")
    print(f"  (= 4 × {total // 4} frames por episodio)")

    print("\n" + "═" * 60)


if __name__ == "__main__":
    main()

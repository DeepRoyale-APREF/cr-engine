#!/usr/bin/env python
"""
Ejemplo 6 — Inspeccionar estado, observaciones y fog-of-war.

Muestra cómo:
  • Comparar el estado completo (god-view) con la observación parcial.
  • Ver qué información se oculta con fog_of_war=True.
  • Inspeccionar los feature vectors y su estructura.

Uso
----
    python examples/06_state_inspection.py
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")

from clash_royale_engine.core.engine import ClashRoyaleEngine
from clash_royale_engine.core.recorder import apply_fog_of_war
from clash_royale_engine.env.gymnasium_env import ClashRoyaleEnv
from clash_royale_engine.players.player_interface import HeuristicBot, RLAgentPlayer
from clash_royale_engine.utils.constants import CARD_VOCAB, OBS_FEATURE_DIM


def main() -> None:
    print("═" * 60)
    print("  Inspección de estado y fog-of-war")
    print("═" * 60)

    # ── Estado completo (god-view) desde el motor ────────────────────
    engine = ClashRoyaleEngine(
        player1=HeuristicBot(aggression=0.6),
        player2=HeuristicBot(aggression=0.5),
        fps=30,
        time_limit=180.0,
    )

    # Avanzar unos frames para que haya tropas en el campo
    for _ in range(300):
        s0, s1, done = engine.step(frames=1)
        if done:
            break

    print("\n  ── Estado completo (P0, god-view) ──")
    n = s0.numbers
    print(f"  Elixir propio:   {n.elixir:.2f}")
    print(f"  Elixir enemigo:  {n.enemy_elixir:.2f}  ← visible en god-view")
    print(f"  Torres propias:  L={n.left_princess_hp:.0f}  R={n.right_princess_hp:.0f}  K={n.king_hp:.0f}")
    print(f"  Torres enemigas: L={n.left_enemy_princess_hp:.0f}  R={n.right_enemy_princess_hp:.0f}  K={n.enemy_king_hp:.0f}")
    print(f"  Aliados:  {len(s0.allies)} unidades")
    print(f"  Enemigos: {len(s0.enemies)} unidades")
    print(f"  Mano: {[c.name for c in s0.cards]}")

    # ── Aplicar fog-of-war ────────────────────────────────────────────
    fogged = apply_fog_of_war(s0)
    print("\n  ── Estado con fog-of-war ──")
    print(f"  Elixir propio:   {fogged.numbers.elixir:.2f}")
    print(f"  Elixir enemigo:  {fogged.numbers.enemy_elixir:.2f}  ← oculto (0)")

    # ── Feature vector ────────────────────────────────────────────────
    obs_full = ClashRoyaleEnv._to_feature_vector(s0)
    obs_fog = ClashRoyaleEnv._to_feature_vector(fogged)

    print(f"\n  ── Feature vector ──")
    print(f"  Dimensión: {OBS_FEATURE_DIM}")
    print(f"  Shape:     {obs_full.shape}")
    print(f"  dtype:     {obs_full.dtype}")
    print(f"  Rango:     [{obs_full.min():.3f}, {obs_full.max():.3f}]")

    # Desglosar estructura del vector
    print(f"\n  ── Estructura del feature vector ──")
    idx = 0
    print(f"  [{idx:4d}] elixir propio / 10     = {obs_full[0]:.3f}")
    idx += 1
    print(f"  [{idx:4d}] elixir enemigo / 10    = {obs_full[1]:.3f}  (fog: {obs_fog[1]:.3f})")
    idx += 1
    print(f"  [{idx:4d}-{idx+5:4d}] HP torres (6 valores)")
    tower_labels = ["L propia", "R propia", "K propio", "L enemiga", "R enemiga", "K enemigo"]
    for j, label in enumerate(tower_labels):
        print(f"         [{idx+j}] {label:12s} = {obs_full[idx+j]:.3f}")
    idx += 6

    print(f"  [{idx:4d}-{idx+35:4d}] Cartas en mano (4 × {len(CARD_VOCAB)+1} = {4*(len(CARD_VOCAB)+1)})")
    for c_i in range(4):
        start = idx + c_i * (len(CARD_VOCAB) + 1)
        one_hot = obs_full[start:start + len(CARD_VOCAB)]
        cost = obs_full[start + len(CARD_VOCAB)]
        card_detected = CARD_VOCAB[int(np.argmax(one_hot))] if one_hot.max() > 0 else "?"
        print(f"         carta[{c_i}] = {card_detected:12s}  costo_norm={cost:.2f}")
    idx += 4 * (len(CARD_VOCAB) + 1)

    print(f"  [{idx:4d}-{idx+575:4d}] Grid aliados  (32×18 = 576)")
    ally_grid = obs_full[idx:idx + 576].reshape(32, 18)
    n_ally_cells = int(ally_grid.sum())
    print(f"         Celdas ocupadas: {n_ally_cells}")
    idx += 576

    print(f"  [{idx:4d}-{idx+575:4d}] Grid enemigos (32×18 = 576)")
    enemy_grid = obs_full[idx:idx + 576].reshape(32, 18)
    n_enemy_cells = int(enemy_grid.sum())
    print(f"         Celdas ocupadas: {n_enemy_cells}")

    # ── Diferencias por fog-of-war ────────────────────────────────────
    diff = np.abs(obs_full - obs_fog)
    n_diff = int((diff > 1e-6).sum())
    print(f"\n  Diferencias entre obs completa y obs con fog: {n_diff} valores")
    if n_diff > 0:
        diff_indices = np.where(diff > 1e-6)[0]
        print(f"  Índices afectados: {diff_indices.tolist()}")

    print("\n" + "═" * 60)


if __name__ == "__main__":
    main()

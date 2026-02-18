#!/usr/bin/env python
"""
Ejemplo 1 — Partida headless (sin GUI) entre dos bots heurísticos.

Muestra cómo:
  • Crear el motor (ClashRoyaleEngine) con dos bots.
  • Avanzar frame a frame e inspeccionar el estado.
  • Detectar el final de la partida y el ganador.

Uso
----
    python examples/01_headless_quickstart.py
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, ".")

from clash_royale_engine.core.engine import ClashRoyaleEngine
from clash_royale_engine.players.player_interface import HeuristicBot


def main() -> None:
    bot1 = HeuristicBot(aggression=0.7)
    bot2 = HeuristicBot(aggression=0.5)

    engine = ClashRoyaleEngine(
        player1=bot1,
        player2=bot2,
        fps=30,
        time_limit=180.0,  # 3 minutos
        speed_multiplier=1.0,
    )

    print("═" * 60)
    print("  Partida headless: Bot1 (aggr=0.7) vs Bot2 (aggr=0.5)")
    print("═" * 60)

    t0 = time.perf_counter()
    step = 0

    while True:
        state_p0, state_p1, done = engine.step(frames=1)
        step += 1

        # Imprimir resumen cada 5 segundos de juego (150 frames)
        if step % 150 == 0:
            n = state_p0.numbes
            game_time = 180.0 - n.time_remaining
            print(
                f"  [{game_time:5.1f}s]  "
                f"Elixir P0={n.elixir:.1f}  P1={n.enemy_elixir:.1f}  |  "
                f"Torres P0: L={n.left_princess_hp:.0f} R={n.right_princess_hp:.0f} K={n.king_hp:.0f}  |  "
                f"Torres P1: L={n.left_enemy_princess_hp:.0f} R={n.right_enemy_princess_hp:.0f} K={n.enemy_king_hp:.0f}"
            )

        if done:
            break

    elapsed = time.perf_counter() - t0
    winner = engine.get_winner()

    print("─" * 60)
    if winner is not None:
        print(f"  Resultado: ¡Jugador {winner} gana!")
    else:
        print("  Resultado: ¡Empate!")
    print(f"  Frames simulados: {step}")
    print(f"  Tiempo real: {elapsed:.2f}s")
    print(f"  Velocidad: {step / elapsed:.0f} frames/s")
    print("═" * 60)


if __name__ == "__main__":
    main()

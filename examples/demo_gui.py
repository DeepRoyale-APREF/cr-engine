#!/usr/bin/env python
"""
Demo: run a HeuristicBot-vs-HeuristicBot match and visualise it with
the simplified Clash Royale GUI.

Usage
-----
    python examples/demo_gui.py

Press **ESC** or close the window to quit.
"""

from __future__ import annotations

import sys
import time

# Ensure the package is importable when running from the repo root
sys.path.insert(0, ".")

from clash_royale_engine.core.engine import ClashRoyaleEngine
from clash_royale_engine.players.player_interface import HeuristicBot
from clash_royale_engine.visualization.renderer import Renderer


def main() -> None:
    # Two heuristic bots
    bot1 = HeuristicBot(aggression=0.6)
    bot2 = HeuristicBot(aggression=0.5)

    engine = ClashRoyaleEngine(
        player1=bot1,
        player2=bot2,
        fps=30,
        time_limit=180.0,
        speed_multiplier=1.0,
    )

    renderer = Renderer(fps=30, title="Clash Royale — Bot vs Bot")

    print("Iniciando simulación con GUI...")
    print("Presiona ESC o cierra la ventana para salir.\n")

    running = True
    step_count = 0

    try:
        while running:
            # Advance one frame
            state_p0, state_p1, done = engine.step(frames=1)

            # Render from player 0 perspective
            running = renderer.render(state_p0)

            step_count += 1

            # Print summary every 5 seconds of game time (150 frames)
            if step_count % 150 == 0:
                n = state_p0.numbers
                elapsed = 180.0 - n.time_remaining
                print(
                    f"[{elapsed:5.1f}s] "
                    f"Elixir P0={n.elixir:.1f}  "
                    f"Torres aliadas: L={n.left_princess_hp:.0f} R={n.right_princess_hp:.0f} K={n.king_hp:.0f}  "
                    f"Torres enemigas: L={n.left_enemy_princess_hp:.0f} R={n.right_enemy_princess_hp:.0f} K={n.enemy_king_hp:.0f}"
                )

            if done:
                winner = engine.get_winner()
                if winner is not None:
                    print(f"\n¡Jugador {winner} gana!")
                else:
                    print("\n¡Empate!")

                # Pause 3 seconds so the user can see the final state
                end_time = time.time() + 3.0
                while time.time() < end_time and running:
                    running = renderer.render(state_p0)
                break

    except KeyboardInterrupt:
        pass
    finally:
        renderer.close()
        print("GUI cerrada.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Ejemplo 4 — Colocar cartas manualmente y observar el resultado.

Muestra cómo:
  • Usar el motor directamente (sin Gymnasium) con step_with_action.
  • Inyectar acciones específicas (tile_x, tile_y, card_idx).
  • Ver el estado detallado tras cada acción.
  • Probar diferentes cartas y formaciones.

Uso
----
    python examples/04_manual_placement.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from clash_royale_engine.core.engine import ClashRoyaleEngine
from clash_royale_engine.players.player_interface import RLAgentPlayer
from clash_royale_engine.utils.constants import CARD_STATS
from clash_royale_engine.utils.validators import InvalidActionError


def print_state(label: str, state) -> None:
    """Imprime un resumen legible del estado."""
    n = state.numbers
    print(f"\n  ── {label} ──")
    print(f"  Elixir propio: {n.elixir:.1f}  |  Enemigo: {n.enemy_elixir:.1f}")
    print(
        f"  Torres propias:  L={n.left_princess_hp:.0f}  R={n.right_princess_hp:.0f}  K={n.king_hp:.0f}"
    )
    print(
        f"  Torres enemigas: L={n.left_enemy_princess_hp:.0f}  R={n.right_enemy_princess_hp:.0f}  K={n.enemy_king_hp:.0f}"
    )
    print(f"  Tiempo restante: {n.time_remaining:.1f}s")
    print(f"  Mano: {[c.name for c in state.cards]}")
    print(f"  Índices jugables: {state.ready}")

    if state.allies:
        print(f"  Aliados en campo ({len(state.allies)}):")
        for a in state.allies:
            print(f"    • {a.unit.name} en tile ({a.position.tile_x}, {a.position.tile_y})")
    if state.enemies:
        print(f"  Enemigos en campo ({len(state.enemies)}):")
        for e in state.enemies:
            print(f"    • {e.unit.name} en tile ({e.position.tile_x}, {e.position.tile_y})")


def main() -> None:
    print("═" * 60)
    print("  Colocación manual de cartas")
    print("═" * 60)

    # Ambos jugadores son RLAgentPlayer (no actúan — acciones inyectadas)
    engine = ClashRoyaleEngine(
        player1=RLAgentPlayer(),
        player2=RLAgentPlayer(),
        fps=30,
        time_limit=180.0,
    )

    state = engine.get_state(0)
    print_state("Estado inicial", state)

    # ── Mostrar mano del jugador 0 ───────────────────────────────────
    hand = engine.players[0].hand
    print(f"\n  Mano de P0: {hand}")
    for i, card_name in enumerate(hand):
        cost = CARD_STATS[card_name]["elixir"]
        is_spell = CARD_STATS[card_name].get("is_spell", False)
        print(f"    [{i}] {card_name:12s}  costo={cost}  {'(hechizo)' if is_spell else '(tropa)'}")

    # ── Acción 1: colocar Giant en el centro-atrás (tile 9, 5) ────────
    print("\n" + "─" * 60)
    print("  Acción: Giant en (9, 5) — card_idx=0")
    giant_idx = hand.index("giant") if "giant" in hand else 0
    try:
        s0, s1, done = engine.step_with_action(0, (9, 5, giant_idx))
        print("  ✓ Acción válida")
        print_state("Tras Giant", s0)
    except InvalidActionError as e:
        print(f"  ✗ Acción inválida: {e}")

    # ── Avanzar 60 frames (~2s) para generar elixir ───────────────────
    print("\n  ⏩ Avanzando 60 frames (2 segundos)...")
    for _ in range(60):
        s0, s1, done = engine.step(frames=1)

    print_state("Tras 2 segundos", s0)

    # ── Acción 2: colocar Archers detrás del Giant ────────────────────
    print("\n" + "─" * 60)
    hand = engine.players[0].hand
    print(f"  Mano actual: {hand}")
    if "archers" in hand:
        archers_idx = hand.index("archers")
        print(f"  Acción: Archers en (9, 3) — card_idx={archers_idx}")
        try:
            s0, s1, done = engine.step_with_action(0, (9, 3, archers_idx))
            print("  ✓ Acción válida")
            print_state("Tras Archers", s0)
        except InvalidActionError as e:
            print(f"  ✗ Acción inválida: {e}")
    else:
        print("  Archers no está en la mano actual, saltando.")

    # ── Acción 3: intentar colocar tropa en lado enemigo (debe fallar) ──
    print("\n" + "─" * 60)
    print("  Acción: intentar Knight en (9, 20) — lado enemigo")
    hand = engine.players[0].hand
    if "knight" in hand:
        knight_idx = hand.index("knight")
        try:
            s0, s1, done = engine.step_with_action(0, (9, 20, knight_idx))
            print("  ✓ Acción válida (¡inesperado!)")
        except InvalidActionError as e:
            print(f"  ✗ Rechazada correctamente: {e}")

    # ── Acción 4: hechizo en lado enemigo (debe funcionar) ───────────
    print("\n" + "─" * 60)
    # Avanzar para regenerar elixir
    for _ in range(120):
        s0, s1, done = engine.step(frames=1)
    hand = engine.players[0].hand
    print(f"  Mano actual: {hand}  |  Elixir: {s0.numbers.elixir:.1f}")

    if "arrows" in hand:
        arrows_idx = hand.index("arrows")
        print("  Acción: Arrows en (9, 25) — hechizo en lado enemigo")
        try:
            s0, s1, done = engine.step_with_action(0, (9, 25, arrows_idx))
            print("  ✓ Hechizo lanzado correctamente en lado enemigo")
        except InvalidActionError as e:
            print(f"  ✗ Error: {e}")
    elif "fireball" in hand:
        fb_idx = hand.index("fireball")
        print("  Acción: Fireball en (9, 25) — hechizo en lado enemigo")
        try:
            s0, s1, done = engine.step_with_action(0, (9, 25, fb_idx))
            print("  ✓ Hechizo lanzado correctamente en lado enemigo")
        except InvalidActionError as e:
            print(f"  ✗ Error: {e}")
    else:
        print("  No hay hechizo en mano, saltando.")

    # ── Avanzar y ver resultado final ──────────────────────────────────
    print("\n  ⏩ Avanzando 120 frames más...")
    for _ in range(120):
        s0, s1, done = engine.step(frames=1)
        if done:
            break

    print_state("Estado final", s0)
    print("\n" + "═" * 60)


if __name__ == "__main__":
    main()

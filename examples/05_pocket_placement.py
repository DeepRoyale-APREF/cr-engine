#!/usr/bin/env python
"""
Ejemplo 5 — Demostrar la mecánica de pocket placement.

Muestra cómo:
  • Verificar que las tropas NO pueden colocarse en lado enemigo.
  • Destruir una torre princesa enemiga manualmente.
  • Verificar que AHORA sí se puede colocar en el pocket del carril
    correspondiente (hasta POCKET_DEPTH tiles pasando el río).
  • Los hechizos siempre se pueden lanzar en cualquier parte.

Uso
----
    python examples/05_pocket_placement.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from clash_royale_engine.core.engine import ClashRoyaleEngine
from clash_royale_engine.players.player_interface import RLAgentPlayer
from clash_royale_engine.utils.constants import (
    CARD_STATS,
    LANE_DIVIDER_X,
    POCKET_DEPTH,
    RIVER_Y_MAX,
)
from clash_royale_engine.utils.validators import InvalidActionError

TROOP_NAMES = [name for name, s in CARD_STATS.items() if not s.get("is_spell", False)]


def _find_troop_idx(engine, player_id: int) -> tuple[int, str]:
    """Return (hand_index, card_name) for the first troop card in hand."""
    hand = engine.players[player_id].hand
    for i, name in enumerate(hand):
        if name in TROOP_NAMES:
            return i, name
    raise RuntimeError("No troop card in hand!")


def try_place(engine, player_id, tile_x, tile_y, card_idx, label):
    """Intenta colocar una carta e imprime el resultado."""
    try:
        engine.step_with_action(player_id, (tile_x, tile_y, card_idx))
        print(f"    ✓ {label}")
    except InvalidActionError as e:
        print(f"    ✗ {label}  →  {e}")


def _refill_elixir(engine, player_id: int = 0) -> None:
    """Give the player full elixir for testing."""
    engine.elixir_system.elixir[player_id] = 10.0
    engine.players[player_id].elixir = 10.0


def main() -> None:
    print("═" * 65)
    print("  Demostración: Pocket Placement (colocación en lado enemigo)")
    print("═" * 65)

    engine = ClashRoyaleEngine(
        player1=RLAgentPlayer(),
        player2=RLAgentPlayer(),
        fps=30,
        time_limit=180.0,
    )

    _refill_elixir(engine)

    pocket_y = int(RIVER_Y_MAX)  # 17 — primera fila del pocket
    pocket_x_left = LANE_DIVIDER_X - 1  # 8  — carril izquierdo
    pocket_x_right = LANE_DIVIDER_X  # 9  — carril derecho

    # ── 1) Intentar colocar tropa en lado enemigo (ambas torres vivas) ─
    troop_idx, troop_name = _find_troop_idx(engine, 0)
    print(f"\n  Mano de P0: {engine.players[0].hand}")

    print("\n  1) Intentar tropa en lado enemigo (torres vivas):")
    try_place(
        engine,
        0,
        pocket_x_left,
        pocket_y,
        troop_idx,
        f"{troop_name} en ({pocket_x_left}, {pocket_y}) carril izquierdo",
    )

    _refill_elixir(engine)
    troop_idx, troop_name = _find_troop_idx(engine, 0)
    try_place(
        engine,
        0,
        pocket_x_right,
        pocket_y,
        troop_idx,
        f"{troop_name} en ({pocket_x_right}, {pocket_y}) carril derecho",
    )

    # ── 2) Hechizo en lado enemigo (siempre permitido) ────────────────
    print("\n  2) Hechizo en lado enemigo (siempre permitido):")
    _refill_elixir(engine)

    spell_name = None
    spell_idx = None
    for i, name in enumerate(engine.players[0].hand):
        if name in ("arrows", "fireball"):
            spell_name = name
            spell_idx = i
            break

    if spell_idx is not None:
        try_place(engine, 0, 9, 28, spell_idx, f"{spell_name} en (9, 28) — fondo del mapa enemigo")
    else:
        print("    (no hay hechizo en mano, saltando)")

    # ── 3) Destruir torre princesa izquierda del enemigo (P1) ─────────
    print("\n  3) Destruyendo torre princesa IZQUIERDA de P1...")
    tower = engine.arena.towers["p1_left_princess"]
    print(f"     HP antes: {tower.hp:.0f}")
    tower.hp = 0  # forzar destrucción
    print(f"     HP después: {tower.hp:.0f}  (destruida)")

    # ── 4) Ahora intentar colocar en pocket izquierdo ──────────────────
    print("\n  4) Colocar en pocket IZQUIERDO (torre destruida):")
    for dy in range(POCKET_DEPTH + 1):
        _refill_elixir(engine)
        troop_idx, troop_name = _find_troop_idx(engine, 0)
        y = int(RIVER_Y_MAX) + dy
        label = f"{troop_name} en ({pocket_x_left}, {y})"
        if dy < POCKET_DEPTH:
            label += "  (dentro del pocket)"
        else:
            label += "  (fuera del pocket — demasiado profundo)"
        try_place(engine, 0, pocket_x_left, y, troop_idx, label)

    # ── 5) Pocket derecho aún bloqueado ───────────────────────────────
    print("\n  5) Pocket DERECHO (torre derecha aún viva):")
    _refill_elixir(engine)
    troop_idx, troop_name = _find_troop_idx(engine, 0)
    try_place(
        engine,
        0,
        pocket_x_right,
        pocket_y,
        troop_idx,
        f"{troop_name} en ({pocket_x_right}, {pocket_y}) carril derecho",
    )

    # ── 6) Destruir torre derecha y verificar ──────────────────────────
    print("\n  6) Destruyendo torre princesa DERECHA de P1...")
    tower_r = engine.arena.towers["p1_right_princess"]
    tower_r.hp = 0
    print("     Torre derecha destruida.")

    _refill_elixir(engine)
    troop_idx, troop_name = _find_troop_idx(engine, 0)

    print("\n  7) Colocar en pocket DERECHO (torre destruida):")
    try_place(
        engine,
        0,
        pocket_x_right,
        pocket_y,
        troop_idx,
        f"{troop_name} en ({pocket_x_right}, {pocket_y}) carril derecho",
    )

    # ── Resumen ────────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  Resumen de reglas:")
    print("    • Tropas en lado propio: siempre ✓")
    print("    • Tropas en lado enemigo: ✗ (bloqueado por defecto)")
    print("    • Hechizos en cualquier parte: siempre ✓")
    print(f"    • Pocket izquierdo (x < {LANE_DIVIDER_X}): se desbloquea al destruir")
    print("      la torre princesa izquierda enemiga")
    print(f"    • Pocket derecho (x >= {LANE_DIVIDER_X}): se desbloquea al destruir")
    print("      la torre princesa derecha enemiga")
    print(f"    • Profundidad del pocket: {POCKET_DEPTH} tiles pasando el río")
    print(f"      (y = {int(RIVER_Y_MAX)} a {int(RIVER_Y_MAX) + POCKET_DEPTH - 1} para P0)")
    print("═" * 65)


if __name__ == "__main__":
    main()

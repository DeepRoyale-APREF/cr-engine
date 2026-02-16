"""
Simplified pathfinding — bridge-awareness.

Full A* is not needed because the arena is an open field with two bridge
chokepoints.  This module provides helpers used by the physics engine to
route ground troops through bridges.

The river occupies tiles y ∈ [RIVER_Y_MIN, RIVER_Y_MAX].  Ground units
can only traverse this band at bridge tiles.
"""

from __future__ import annotations

from typing import Tuple

from clash_royale_engine.utils.constants import (
    BRIDGE_LEFT_X,
    BRIDGE_RIGHT_X,
    BRIDGE_WIDTH,
    BRIDGE_Y,
    N_HEIGHT_TILES,
    RIVER_Y_MAX,
    RIVER_Y_MIN,
)

# Pre-computed bridge centre x-coordinates
_LEFT_CX: float = BRIDGE_LEFT_X + BRIDGE_WIDTH / 2.0
_RIGHT_CX: float = BRIDGE_RIGHT_X + BRIDGE_WIDTH / 2.0


def nearest_bridge_x(entity_x: float) -> float:
    """Return the centre x-coordinate of the nearest bridge."""
    if abs(entity_x - _LEFT_CX) <= abs(entity_x - _RIGHT_CX):
        return _LEFT_CX
    return _RIGHT_CX


def is_on_bridge(x: float) -> bool:
    """Return *True* if *x* falls within either bridge lane (± 0.5 tile tolerance)."""
    return (
        BRIDGE_LEFT_X - 0.5 <= x <= BRIDGE_LEFT_X + BRIDGE_WIDTH + 0.5
        or BRIDGE_RIGHT_X - 0.5 <= x <= BRIDGE_RIGHT_X + BRIDGE_WIDTH + 0.5
    )


def needs_bridge(player_id: int, current_y: float, target_y: float) -> bool:
    """
    Return True if the unit needs to cross a bridge to reach *target_y*.

    Player 0 is at the bottom (low y), Player 1 at the top (high y).
    The river band spans ``[RIVER_Y_MIN, RIVER_Y_MAX]``.
    """
    ent_below = current_y < RIVER_Y_MIN
    ent_above = current_y > RIVER_Y_MAX
    goal_below = target_y < RIVER_Y_MIN
    goal_above = target_y > RIVER_Y_MAX
    return (ent_below and goal_above) or (ent_above and goal_below)


def waypoint_through_bridge(
    entity_x: float,
    entity_y: float,
    player_id: int,
) -> Tuple[float, float]:
    """
    Return an intermediate waypoint that routes the entity through the
    nearest bridge.
    """
    bx = nearest_bridge_x(entity_x)

    # Cross to the opposite bank
    if entity_y < RIVER_Y_MIN:
        return (bx, RIVER_Y_MAX + 0.5)
    return (bx, RIVER_Y_MIN - 0.5)

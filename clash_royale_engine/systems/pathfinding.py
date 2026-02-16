"""
Simplified pathfinding — bridge-awareness.

Full A* is not needed because the arena is an open field with two bridge
chokepoints.  This module provides helpers used by the physics engine to
route troops through bridges.
"""

from __future__ import annotations

from typing import Tuple

from clash_royale_engine.utils.constants import (
    BRIDGE_LEFT_X,
    BRIDGE_RIGHT_X,
    BRIDGE_Y,
    N_HEIGHT_TILES,
)


def nearest_bridge_x(entity_x: float) -> float:
    """Return the x-coordinate of the nearest bridge."""
    if abs(entity_x - BRIDGE_LEFT_X) <= abs(entity_x - BRIDGE_RIGHT_X):
        return float(BRIDGE_LEFT_X)
    return float(BRIDGE_RIGHT_X)


def needs_bridge(player_id: int, current_y: float, target_y: float) -> bool:
    """
    Return True if the unit needs to cross a bridge to reach *target_y*.

    Player 0 is at the bottom (low y), Player 1 at the top (high y).
    The bridge row is at ``BRIDGE_Y``.
    """
    if player_id == 0:
        return current_y < BRIDGE_Y <= target_y
    else:
        return current_y > BRIDGE_Y >= target_y


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
    return (bx, float(BRIDGE_Y))

"""
Physics engine — continuous movement with circular collision resolution.

Positions are in **tile coordinates** (continuous floats).  Speed values
from card stats are in *pixels / second* and are converted using the tile
dimensions.

River restriction
-----------------
Ground troops **cannot** enter the 2-tile-deep river band
(``RIVER_Y_MIN`` … ``RIVER_Y_MAX``) unless they are standing on a
bridge tile.  Air troops are unaffected.  When a ground troop needs to
reach the other side of the arena it is routed to the nearest bridge
first.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from clash_royale_engine.entities.base_entity import Entity
from clash_royale_engine.utils.constants import (
    BRIDGE_LEFT_X,
    BRIDGE_RIGHT_X,
    BRIDGE_WIDTH,
    N_HEIGHT_TILES,
    N_WIDE_TILES,
    RIVER_Y_MAX,
    RIVER_Y_MIN,
    TILE_HEIGHT,
    TILE_WIDTH,
)

# ── module-level helpers (river / bridge geometry) ────────────────────────

# Bridge centre x-coordinates (used for waypoint routing)
_LEFT_BRIDGE_CX: float = BRIDGE_LEFT_X + BRIDGE_WIDTH / 2.0
_RIGHT_BRIDGE_CX: float = BRIDGE_RIGHT_X + BRIDGE_WIDTH / 2.0


def _is_on_bridge(x: float) -> bool:
    """Return *True* if *x* falls within either bridge lane (± 0.5 tolerance)."""
    return (
        BRIDGE_LEFT_X - 0.5 <= x <= BRIDGE_LEFT_X + BRIDGE_WIDTH + 0.5
        or BRIDGE_RIGHT_X - 0.5 <= x <= BRIDGE_RIGHT_X + BRIDGE_WIDTH + 0.5
    )


def _needs_bridge(entity_y: float, goal_y: float) -> bool:
    """Return *True* if entity and goal are on opposite sides of the river."""
    ent_below = entity_y < RIVER_Y_MIN
    ent_above = entity_y > RIVER_Y_MAX
    goal_below = goal_y < RIVER_Y_MIN
    goal_above = goal_y > RIVER_Y_MAX
    return (ent_below and goal_above) or (ent_above and goal_below)


def _nearest_bridge_cx(x: float) -> float:
    """Return the centre-x of whichever bridge is closer to *x*."""
    if abs(x - _LEFT_BRIDGE_CX) <= abs(x - _RIGHT_BRIDGE_CX):
        return _LEFT_BRIDGE_CX
    return _RIGHT_BRIDGE_CX


def _bridge_waypoint(
    entity_x: float, entity_y: float, goal_y: float,
) -> Optional[Tuple[float, float]]:
    """
    Return an intermediate waypoint that routes the entity through the
    nearest bridge, or *None* if no detour is needed.

    Strategy:
    1. If not yet aligned with the bridge lane → move horizontally first.
    2. Otherwise → waypoint is on the far side of the river.
    """
    bridge_cx = _nearest_bridge_cx(entity_x)

    # Step 1: align with bridge lane
    if abs(entity_x - bridge_cx) > 1.0:
        return (bridge_cx, entity_y)

    # Step 2: cross to the other bank
    if entity_y < RIVER_Y_MIN:
        return (bridge_cx, RIVER_Y_MAX + 0.5)
    return (bridge_cx, RIVER_Y_MIN - 0.5)


# ══════════════════════════════════════════════════════════════════════════


class PhysicsEngine:
    """Handles movement, collision separation and arena bounds."""

    def __init__(self, fps: int = 30) -> None:
        self.fps: int = fps
        self.dt: float = 1.0 / fps
        # Average tile size used to convert pixel-speed → tile-speed
        self._avg_tile: float = (TILE_WIDTH + TILE_HEIGHT) / 2.0

    # ── public API ────────────────────────────────────────────────────────

    def update(self, entities: List[Entity]) -> None:
        """Advance all movable entities by one frame."""
        dt = self.dt
        for e in entities:
            if e.is_dead or e.is_static or not e.is_deployed:
                continue
            self._move_entity(e, entities, dt)

    # ── internal ──────────────────────────────────────────────────────────

    def _move_entity(
        self, entity: Entity, all_entities: List[Entity], dt: float,
    ) -> None:
        desired = self._desired_velocity(entity)

        # Separation steering to avoid overlaps
        sep = self._separation(entity, all_entities)
        vx = desired[0] + sep[0]
        vy = desired[1] + sep[1]

        new_x = entity.x + vx * dt
        new_y = entity.y + vy * dt

        # Clamp to arena bounds
        new_x = float(np.clip(new_x, 0.0, N_WIDE_TILES - 1.0))
        new_y = float(np.clip(new_y, 0.0, N_HEIGHT_TILES - 1.0))

        # River restriction — ground units only
        if entity.transport == "ground":
            new_y = self._enforce_river(entity.y, new_x, new_y)

        entity.x = new_x
        entity.y = new_y

    # ── desired velocity (bridge-aware) ──────────────────────────────────

    def _desired_velocity(self, entity: Entity) -> np.ndarray:
        """Velocity vector (tiles/s) toward current target or default lane.

        Ground troops are automatically routed through the nearest bridge
        when the river lies between them and their goal.
        """
        speed_tiles = entity.speed / self._avg_tile  # px/s → tiles/s

        target = entity.current_target
        if target is not None:
            goal_x, goal_y = target.x, target.y
            dist = float(np.hypot(goal_x - entity.x, goal_y - entity.y))

            # Already in attack range → stop
            if dist <= entity.attack_range:
                return np.array([0.0, 0.0])
        else:
            # No target — advance toward enemy side
            direction = 1.0 if entity.player_id == 0 else -1.0
            goal_x = entity.x
            goal_y = entity.y + direction * 20.0  # far ahead

        # Ground troop bridge routing
        if (
            entity.transport == "ground"
            and _needs_bridge(entity.y, goal_y)
        ):
            wp = _bridge_waypoint(entity.x, entity.y, goal_y)
            if wp is not None:
                goal_x, goal_y = wp

        dx = goal_x - entity.x
        dy = goal_y - entity.y
        dist = float(np.hypot(dx, dy))
        if dist < 0.01:
            return np.array([0.0, 0.0])

        return np.array([dx / dist * speed_tiles, dy / dist * speed_tiles])

    # ── river enforcement ────────────────────────────────────────────────

    @staticmethod
    def _enforce_river(old_y: float, new_x: float, new_y: float) -> float:
        """Prevent a ground troop from entering the river outside a bridge.

        Handles three cases:
        * Moving *into* the river band.
        * Jumping completely *over* the river in a single frame.
        * Already inside the river (e.g. pushed by separation) but not on
          a bridge — snap to the nearest bank.
        """
        in_river = RIVER_Y_MIN <= new_y <= RIVER_Y_MAX

        # Fast-entity hop: crossed from one bank to the other in one frame
        jumped = (
            (old_y < RIVER_Y_MIN and new_y > RIVER_Y_MAX)
            or (old_y > RIVER_Y_MAX and new_y < RIVER_Y_MIN)
        )

        if not in_river and not jumped:
            return new_y  # nothing to enforce

        if _is_on_bridge(new_x):
            return new_y  # bridge tiles are passable

        # Clamp to the bank the entity came from
        if old_y < RIVER_Y_MIN:
            return RIVER_Y_MIN - 0.01
        if old_y > RIVER_Y_MAX:
            return RIVER_Y_MAX + 0.01

        # Entity was already in the river (edge case) — push to nearest bank
        mid = (RIVER_Y_MIN + RIVER_Y_MAX) / 2.0
        return (RIVER_Y_MIN - 0.01) if old_y < mid else (RIVER_Y_MAX + 0.01)

    # ── separation ───────────────────────────────────────────────────────

    @staticmethod
    def _separation(entity: Entity, all_entities: List[Entity]) -> np.ndarray:
        """Compute separation steering force to avoid overlaps."""
        force = np.array([0.0, 0.0])
        for other in all_entities:
            if other is entity or other.is_dead:
                continue
            dx = entity.x - other.x
            dy = entity.y - other.y
            dist = float(np.hypot(dx, dy))
            min_dist = entity.hitbox_radius + other.hitbox_radius

            if 0 < dist < min_dist:
                overlap = min_dist - dist
                force[0] += (dx / dist) * overlap * 2.0
                force[1] += (dy / dist) * overlap * 2.0

        return force

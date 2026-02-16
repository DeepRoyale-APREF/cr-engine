"""
Physics engine — continuous movement with circular collision resolution.

Positions are in **tile coordinates** (continuous floats).  Speed values
from card stats are in *pixels / second* and are converted using the tile
dimensions.
"""

from __future__ import annotations

from typing import List

import numpy as np

from clash_royale_engine.entities.base_entity import Entity
from clash_royale_engine.utils.constants import (
    BRIDGE_LEFT_X,
    BRIDGE_RIGHT_X,
    BRIDGE_Y,
    N_HEIGHT_TILES,
    N_WIDE_TILES,
    TILE_HEIGHT,
    TILE_WIDTH,
)


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

    def _move_entity(self, entity: Entity, all_entities: List[Entity], dt: float) -> None:
        desired = self._desired_velocity(entity)

        # Separation steering to avoid overlaps
        sep = self._separation(entity, all_entities)
        vx = desired[0] + sep[0]
        vy = desired[1] + sep[1]

        entity.x += vx * dt
        entity.y += vy * dt

        # Clamp to arena bounds
        entity.x = float(np.clip(entity.x, 0.0, N_WIDE_TILES - 1.0))
        entity.y = float(np.clip(entity.y, 0.0, N_HEIGHT_TILES - 1.0))

    def _desired_velocity(self, entity: Entity) -> np.ndarray:
        """Velocity vector (tiles/s) toward current target or default lane."""
        speed_tiles = entity.speed / self._avg_tile  # convert px/s → tiles/s

        target = entity.current_target
        if target is not None:
            dx = target.x - entity.x
            dy = target.y - entity.y
            dist = float(np.hypot(dx, dy))

            # Already in attack range → stop moving
            if dist <= entity.attack_range:
                return np.array([0.0, 0.0])

            return np.array([dx / dist * speed_tiles, dy / dist * speed_tiles])

        # No target — advance toward enemy side
        # Player 0 moves upward (+y), Player 1 moves downward (-y)
        direction = 1.0 if entity.player_id == 0 else -1.0

        # Prefer nearest bridge lane on approach
        target_y = entity.y + direction * speed_tiles
        vx = 0.0
        vy = direction * speed_tiles

        # If close to bridge row, steer toward nearest bridge if off-lane
        bridge_dist = abs(entity.y - BRIDGE_Y)
        if bridge_dist < 3.0:
            nearest_bridge_x = (
                BRIDGE_LEFT_X
                if abs(entity.x - BRIDGE_LEFT_X) < abs(entity.x - BRIDGE_RIGHT_X)
                else BRIDGE_RIGHT_X
            )
            bx = nearest_bridge_x - entity.x
            if abs(bx) > 0.5:
                vx = float(np.sign(bx)) * speed_tiles * 0.5

        return np.array([vx, vy])

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

"""
Base Entity class — the foundation for every in-game object.

All troops, buildings and projectiles derive from :class:`Entity`.
Positions are stored in **tile coordinates** (continuous floats) and
converted to pixels only when building the external :class:`State`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from clash_royale_engine.utils.constants import N_HEIGHT_TILES, N_WIDE_TILES

_next_entity_id: int = 0


def _generate_id() -> int:
    global _next_entity_id
    _next_entity_id += 1
    return _next_entity_id


def reset_entity_id_counter() -> None:
    """Reset the global entity-id counter (call between episodes)."""
    global _next_entity_id
    _next_entity_id = 0


class Entity:
    """
    Base class for every game object.

    Coordinates
    -----------
    ``x`` / ``y`` are continuous **tile** coordinates.
    """

    def __init__(
        self,
        name: str,
        player_id: int,
        x: float,
        y: float,
        hp: int,
        damage: int,
        hit_speed: float,
        attack_range: float,
        sight_range: float,
        speed: float,
        target_type: str,
        transport: str = "ground",
        hitbox_radius: float = 0.5,
        is_building: bool = False,
        has_projectile: bool = False,
        projectile_speed: float = 0.0,
        deploy_frames: int = 0,
    ) -> None:
        self.id: int = _generate_id()
        self.name: str = name
        self.player_id: int = player_id

        # Position (tile coordinates, continuous)
        self.x: float = x
        self.y: float = y

        # Stats
        self.max_hp: int = hp
        self.hp: int = hp
        self.damage: int = damage
        self.hit_speed: float = hit_speed  # seconds between attacks
        self.attack_range: float = attack_range  # tiles
        self.sight_range: float = sight_range  # tiles
        self.speed: float = speed  # pixels / second
        self.target_type: str = target_type  # "all" | "ground" | "buildings"
        self.transport: str = transport  # "ground" | "air"
        self.hitbox_radius: float = hitbox_radius

        # Flags
        self.is_building: bool = is_building
        self.is_static: bool = is_building  # buildings don't move
        self.has_projectile: bool = has_projectile
        self.projectile_speed: float = projectile_speed

        # Combat state
        self.current_target: Optional[Entity] = None
        self.next_attack_frame: int = 0

        # Deployment
        self.deploy_frames_remaining: int = deploy_frames
        self.is_deployed: bool = deploy_frames <= 0

        # Alive flag
        self.alive: bool = True

    # ── derived properties ────────────────────────────────────────────────

    @property
    def is_dead(self) -> bool:
        return not self.alive or self.hp <= 0

    @property
    def tile_x(self) -> int:
        return int(np.clip(round(self.x), 0, N_WIDE_TILES - 1))

    @property
    def tile_y(self) -> int:
        return int(np.clip(round(self.y), 0, N_HEIGHT_TILES - 1))

    # ── convenience ───────────────────────────────────────────────────────

    def distance_to(self, other: Entity) -> float:
        """Euclidean distance in tiles."""
        return float(np.hypot(self.x - other.x, self.y - other.y))

    def apply_damage(self, amount: int) -> None:
        """Reduce HP; mark dead when ≤ 0."""
        self.hp -= amount
        if self.hp <= 0:
            self.hp = 0
            self.alive = False

    def tick_deploy(self) -> None:
        """Advance deployment countdown by one frame."""
        if not self.is_deployed:
            self.deploy_frames_remaining -= 1
            if self.deploy_frames_remaining <= 0:
                self.is_deployed = True

    def __repr__(self) -> str:
        return (
            f"Entity(id={self.id}, name={self.name!r}, player={self.player_id}, "
            f"pos=({self.x:.1f},{self.y:.1f}), hp={self.hp}/{self.max_hp})"
        )


class Projectile:
    """A ranged-attack projectile travelling toward a target."""

    def __init__(
        self,
        source: Entity,
        target: Entity,
        damage: int,
        speed: float,
    ) -> None:
        self.source: Entity = source
        self.target: Entity = target
        self.damage: int = damage
        self.speed: float = speed  # pixels / second

        # Start at source position
        self.x: float = source.x
        self.y: float = source.y

    def update(self, dt: float, tile_width: float, tile_height: float) -> bool:
        """
        Move the projectile toward its target for *dt* seconds.

        ``speed`` is in pixels/s so we convert via tile dimensions.

        Returns ``True`` if the projectile has reached the target.
        """
        if self.target.is_dead:
            return True  # target gone – discard

        dx = self.target.x - self.x
        dy = self.target.y - self.y
        dist_tiles = float(np.hypot(dx, dy))

        if dist_tiles < 0.2:
            return True  # close enough → impact

        # Speed in tiles / second  (average of the two axes)
        avg_tile_size = (tile_width + tile_height) / 2.0
        speed_tiles = self.speed / avg_tile_size

        move = speed_tiles * dt
        if move >= dist_tiles:
            self.x = self.target.x
            self.y = self.target.y
            return True

        ratio = move / dist_tiles
        self.x += dx * ratio
        self.y += dy * ratio
        return False

"""Princess Tower — defensive building on each lane."""

from __future__ import annotations

from clash_royale_engine.entities.base_entity import Entity
from clash_royale_engine.utils.constants import DEFAULT_FPS, PRINCESS_TOWER_STATS


def create_princess_tower(
    player_id: int,
    x: float,
    y: float,
    fps: int = DEFAULT_FPS,
) -> Entity:
    """Factory function for a Princess Tower."""
    s = PRINCESS_TOWER_STATS
    return Entity(
        name=s["name"],
        player_id=player_id,
        x=x,
        y=y,
        hp=s["hp"],
        damage=s["damage"],
        hit_speed=s["hit_speed"],
        attack_range=s["range"],
        sight_range=s["sight_range"],
        speed=0.0,
        target_type=s["target"],
        transport="ground",
        hitbox_radius=s["hitbox_radius"],
        is_building=True,
        has_projectile=True,
        projectile_speed=s["projectile_speed"],
        deploy_frames=0,
    )

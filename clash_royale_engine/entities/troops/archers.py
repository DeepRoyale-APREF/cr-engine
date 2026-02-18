"""Archers troop — spawns 2 units, ranged, medium speed."""

from __future__ import annotations

from typing import List

from clash_royale_engine.entities.base_entity import Entity
from clash_royale_engine.utils.constants import CARD_STATS, DEFAULT_FPS


def create_archers(player_id: int, x: float, y: float, fps: int = DEFAULT_FPS) -> List[Entity]:
    """Factory: returns a list of 2 Archer entities with spawn offset."""
    s = CARD_STATS["archers"]
    offset = s.get("spawn_offset", 1.0) / 2.0
    entities: List[Entity] = []
    for dx in (-offset, offset):
        entities.append(
            Entity(
                name=s["name"],
                player_id=player_id,
                x=x + dx,
                y=y,
                hp=s["hp"],
                damage=s["damage"],
                hit_speed=s["hit_speed"],
                attack_range=s["range"],
                sight_range=s["sight_range"],
                speed=s["speed"],
                target_type=s["target"],
                transport=s["transport"],
                hitbox_radius=s["hitbox_radius"],
                has_projectile=s["has_projectile"],
                projectile_speed=s["projectile_speed"],
                deploy_frames=int(s["deploy_time"] * fps),
            )
        )
    return entities

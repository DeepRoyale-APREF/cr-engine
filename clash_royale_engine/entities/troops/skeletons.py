"""Skeletons troop — swarm of 3 fast, fragile melee units."""

from __future__ import annotations

import math
from typing import List

from clash_royale_engine.entities.base_entity import Entity
from clash_royale_engine.utils.constants import CARD_STATS, DEFAULT_FPS


def create_skeletons(player_id: int, x: float, y: float, fps: int = DEFAULT_FPS) -> List[Entity]:
    """Factory: returns a list of 3 Skeleton entities in triangle formation."""
    s = CARD_STATS["skeletons"]
    offsets = _triangle_offsets(spacing=0.6)
    entities: List[Entity] = []
    for dx, dy in offsets:
        entities.append(
            Entity(
                name=s["name"],
                player_id=player_id,
                x=x + dx,
                y=y + dy,
                hp=s["hp"],
                damage=s["damage"],
                hit_speed=s["hit_speed"],
                attack_range=s["range"],
                sight_range=s["sight_range"],
                speed=s["speed"],
                target_type=s["target"],
                transport=s["transport"],
                hitbox_radius=s["hitbox_radius"],
                has_projectile=s.get("has_projectile", False),
                deploy_frames=int(s["deploy_time"] * fps),
            )
        )
    return entities


def _triangle_offsets(spacing: float = 0.6) -> List[tuple[float, float]]:
    """Return 3 (dx, dy) offsets forming an equilateral triangle."""
    angle_step = 2.0 * math.pi / 3.0
    return [
        (spacing * math.cos(i * angle_step), spacing * math.sin(i * angle_step)) for i in range(3)
    ]

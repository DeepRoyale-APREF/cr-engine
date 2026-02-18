"""Fireball spell — smaller area, big damage + knockback."""

from __future__ import annotations

from typing import List

from clash_royale_engine.entities.base_entity import Entity
from clash_royale_engine.entities.spells.arrows import SpellEffect
from clash_royale_engine.utils.constants import CARD_STATS


def apply_fireball(
    player_id: int,
    center_x: float,
    center_y: float,
    targets: List[Entity],
) -> SpellEffect:
    """Apply Fireball spell: area damage + knockback."""
    s = CARD_STATS["fireball"]
    radius = s["radius"]
    damage = s["damage"]
    crown_dmg = s["crown_tower_damage"]
    knockback = s.get("knockback", 0.0)

    for t in targets:
        if t.player_id == player_id:
            continue
        dist = ((t.x - center_x) ** 2 + (t.y - center_y) ** 2) ** 0.5
        if dist <= radius + t.hitbox_radius:
            actual_damage = crown_dmg if t.is_building else damage
            t.apply_damage(actual_damage)

            # Apply knockback — push away from centre
            if knockback > 0 and not t.is_building and dist > 0.01:
                dx = t.x - center_x
                dy = t.y - center_y
                norm = (dx**2 + dy**2) ** 0.5
                t.x += (dx / norm) * knockback
                t.y += (dy / norm) * knockback

    return SpellEffect(
        name="fireball",
        player_id=player_id,
        center_x=center_x,
        center_y=center_y,
        radius=radius,
        damage=damage,
        crown_tower_damage=crown_dmg,
        knockback=knockback,
        remaining_frames=25,  # ~0.83 s at 30 fps
    )

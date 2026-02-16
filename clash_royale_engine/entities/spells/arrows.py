"""Arrows spell — large-area moderate damage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from clash_royale_engine.entities.base_entity import Entity
from clash_royale_engine.utils.constants import CARD_STATS


@dataclass
class SpellEffect:
    """Transient effect produced by casting a spell."""

    name: str
    player_id: int
    center_x: float
    center_y: float
    radius: float
    damage: int
    crown_tower_damage: int
    knockback: float
    remaining_frames: int  # how many frames the visual lasts (instant = 1)


def apply_arrows(
    player_id: int,
    center_x: float,
    center_y: float,
    targets: List[Entity],
) -> SpellEffect:
    """Apply Arrows spell: deals area damage to all targets in radius."""
    s = CARD_STATS["arrows"]
    radius = s["radius"]
    damage = s["damage"]
    crown_dmg = s["crown_tower_damage"]

    for t in targets:
        if t.player_id == player_id:
            continue  # don't hit own units
        dist = ((t.x - center_x) ** 2 + (t.y - center_y) ** 2) ** 0.5
        if dist <= radius + t.hitbox_radius:
            actual_damage = crown_dmg if t.is_building else damage
            t.apply_damage(actual_damage)

    return SpellEffect(
        name="arrows",
        player_id=player_id,
        center_x=center_x,
        center_y=center_y,
        radius=radius,
        damage=damage,
        crown_tower_damage=crown_dmg,
        knockback=0.0,
        remaining_frames=1,
    )

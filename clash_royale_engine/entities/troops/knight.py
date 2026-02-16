"""Knight troop — medium speed, tanky melee, targets ground."""

from __future__ import annotations

from clash_royale_engine.entities.base_entity import Entity
from clash_royale_engine.utils.constants import CARD_STATS, DEFAULT_FPS


def create_knight(player_id: int, x: float, y: float, fps: int = DEFAULT_FPS) -> Entity:
    """Factory function for a Knight entity."""
    s = CARD_STATS["knight"]
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
        speed=s["speed"],
        target_type=s["target"],
        transport=s["transport"],
        hitbox_radius=s["hitbox_radius"],
        has_projectile=s["has_projectile"],
        deploy_frames=int(s["deploy_time"] * fps),
    )

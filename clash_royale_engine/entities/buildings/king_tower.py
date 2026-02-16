"""King Tower — central defensive building, starts inactive."""

from __future__ import annotations

from clash_royale_engine.entities.base_entity import Entity
from clash_royale_engine.utils.constants import DEFAULT_FPS, KING_TOWER_STATS


class KingTowerEntity(Entity):
    """King tower with activation logic."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.is_active: bool = False

    def activate(self) -> None:
        """Activate the king tower (e.g. when hit or enemy crosses bridge)."""
        self.is_active = True


def create_king_tower(
    player_id: int,
    x: float,
    y: float,
    fps: int = DEFAULT_FPS,
) -> KingTowerEntity:
    """Factory function for a King Tower entity."""
    s = KING_TOWER_STATS
    tower = KingTowerEntity(
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
    tower.is_active = bool(s.get("starts_active", False))
    return tower

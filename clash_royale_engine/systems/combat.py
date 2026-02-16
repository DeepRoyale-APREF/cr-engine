"""
Combat system — attacks, projectiles, and damage application.
"""

from __future__ import annotations

from typing import List

from clash_royale_engine.entities.base_entity import Entity, Projectile
from clash_royale_engine.entities.buildings.king_tower import KingTowerEntity
from clash_royale_engine.utils.constants import TILE_HEIGHT, TILE_WIDTH


class CombatSystem:
    """Process melee / ranged attacks and projectile travel."""

    def __init__(self, fps: int = 30) -> None:
        self.fps: int = fps
        self.dt: float = 1.0 / fps
        self.active_projectiles: List[Projectile] = []

    # ── public API ────────────────────────────────────────────────────────

    def process_attacks(self, entities: List[Entity], current_frame: int) -> None:
        """Check every entity and fire attacks when ready."""
        for e in entities:
            if e.is_dead or not e.is_deployed or e.current_target is None:
                continue
            # Skip inactive king towers
            if isinstance(e, KingTowerEntity) and not e.is_active:
                continue

            if current_frame < e.next_attack_frame:
                continue

            dist = e.distance_to(e.current_target)
            if dist <= e.attack_range + e.current_target.hitbox_radius:
                self._execute_attack(e, current_frame)

    def update_projectiles(self) -> None:
        """Move projectiles and apply damage on impact."""
        to_remove: List[Projectile] = []
        for proj in self.active_projectiles:
            hit = proj.update(self.dt, TILE_WIDTH, TILE_HEIGHT)
            if hit:
                if not proj.target.is_dead:
                    self._apply_damage(proj.target, proj.damage)
                to_remove.append(proj)
        for p in to_remove:
            self.active_projectiles.remove(p)

    def reset(self) -> None:
        self.active_projectiles.clear()

    # ── internal ──────────────────────────────────────────────────────────

    def _execute_attack(self, attacker: Entity, current_frame: int) -> None:
        target = attacker.current_target
        assert target is not None

        if attacker.has_projectile:
            proj = Projectile(
                source=attacker,
                target=target,
                damage=attacker.damage,
                speed=attacker.projectile_speed,
            )
            self.active_projectiles.append(proj)
        else:
            # Melee — instant damage
            self._apply_damage(target, attacker.damage)

        # Cooldown
        delay_frames = max(1, int(attacker.hit_speed * self.fps))
        attacker.next_attack_frame = current_frame + delay_frames

    @staticmethod
    def _apply_damage(target: Entity, amount: int) -> None:
        target.apply_damage(amount)

        # If a king tower takes damage, activate it
        if isinstance(target, KingTowerEntity) and not target.is_active:
            target.activate()

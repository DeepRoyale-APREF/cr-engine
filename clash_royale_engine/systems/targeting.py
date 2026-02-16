"""
Targeting system — official Clash Royale logic.

1. **Target retention** (sticky targeting):
   - *Building-targeting troops* (e.g. Giant) keep their current target
     until it **dies**.  They chase across the arena and never switch.
   - *All/ground-targeting troops* (e.g. Knight, Musketeer) keep their
     target while **actively attacking** (within attack range).  If they
     are still *walking* toward a distant target and a closer enemy
     enters their ``sight_range``, they will **retarget** to the closer
     enemy (distraction mechanic).
   - *Buildings / towers* keep their current target while it is within
     ``sight_range``.  Once the target leaves range or dies, a new
     target is acquired.
2. Search for enemies within *sight_range*.
3. Filter by allowed target type.
4. Prioritise buildings when unit's ``target_type == "buildings"``.
5. Pick the closest valid target.
"""

from __future__ import annotations

from typing import List, Optional

from clash_royale_engine.entities.base_entity import Entity
from clash_royale_engine.entities.buildings.king_tower import KingTowerEntity


class TargetingSystem:
    """Stateless helper — call :meth:`update_targets` once per frame."""

    @staticmethod
    def find_target(
        entity: Entity,
        potential_targets: List[Entity],
    ) -> Optional[Entity]:
        """Return the best target for *entity*, or ``None``."""
        if not entity.is_deployed or entity.is_dead:
            return None

        # ── Sticky targeting (true Clash Royale behaviour) ────────────
        # Building-targeting troops (Giant): absolute lock — never switch.
        # Other troops: sticky while actively fighting (in attack range).
        #   While still *walking* toward a distant target, a closer enemy
        #   that enters sight_range will steal aggro (distraction).
        # Towers: retain while within firing (sight) range.
        if entity.current_target is not None and not entity.current_target.is_dead:
            if not entity.is_static:
                # Building-targeting troops → never abandon
                if entity.target_type == "buildings":
                    return entity.current_target
                # Other troops: stay locked while in combat range
                dist_cur = entity.distance_to(entity.current_target)
                if dist_cur <= entity.attack_range + entity.current_target.hitbox_radius:
                    return entity.current_target
                # Walking phase — fall through to re-acquire, which picks
                # the closest valid enemy (may be the same or a nearer one).
            else:
                # Buildings / towers: retain while within firing range
                if entity.distance_to(entity.current_target) <= entity.sight_range:
                    return entity.current_target

        # ── Acquire a new target ──────────────────────────────────────
        # Gather candidates in sight range
        in_range: List[Entity] = []
        for t in potential_targets:
            if t.is_dead or t.player_id == entity.player_id:
                continue
            if entity.distance_to(t) <= entity.sight_range:
                in_range.append(t)

        if not in_range:
            # Building-targeting troops (e.g. Giant) always know where
            # enemy buildings are — search the whole map as fallback.
            if entity.target_type == "buildings":
                all_enemy_buildings = [
                    t for t in potential_targets
                    if not t.is_dead
                    and t.player_id != entity.player_id
                    and t.is_building
                ]
                if all_enemy_buildings:
                    return min(all_enemy_buildings, key=lambda t: entity.distance_to(t))
            return None

        # Filter by target type
        valid = TargetingSystem._filter_by_type(entity.target_type, in_range)
        if not valid:
            return None

        return min(valid, key=lambda t: entity.distance_to(t))

    @staticmethod
    def _filter_by_type(target_type: str, candidates: List[Entity]) -> List[Entity]:
        if target_type == "all":
            return candidates
        if target_type == "ground":
            return [c for c in candidates if c.transport == "ground"]
        if target_type == "buildings":
            return [c for c in candidates if c.is_building]
        return candidates

    @staticmethod
    def update_targets(
        entities: List[Entity],
        all_entities: List[Entity],
    ) -> None:
        """Refresh targets for every entity in *entities*."""
        for e in entities:
            if e.is_dead or not e.is_deployed:
                continue
            # Skip inactive king towers
            if isinstance(e, KingTowerEntity) and not e.is_active:
                e.current_target = None
                continue
            e.current_target = TargetingSystem.find_target(e, all_entities)

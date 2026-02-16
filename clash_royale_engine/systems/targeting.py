"""
Targeting system — official Clash Royale logic.

1. Search for enemies within *sight_range*.
2. Filter by allowed target type.
3. Prioritise buildings when unit's ``target_type == "buildings"``.
4. Pick the closest valid target.
5. Retain target until it dies or leaves range.
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

        # Keep current target if still valid
        if (
            entity.current_target is not None
            and not entity.current_target.is_dead
            and entity.distance_to(entity.current_target) <= entity.sight_range * 1.2
        ):
            return entity.current_target

        # Gather candidates in sight range
        in_range: List[Entity] = []
        for t in potential_targets:
            if t.is_dead or t.player_id == entity.player_id:
                continue
            if entity.distance_to(t) <= entity.sight_range:
                in_range.append(t)

        if not in_range:
            return None

        # Filter by target type
        valid = TargetingSystem._filter_by_type(entity.target_type, in_range)
        if not valid:
            return None

        # Buildings-only units prioritise buildings
        if entity.target_type == "buildings":
            buildings = [t for t in valid if t.is_building]
            if buildings:
                return min(buildings, key=lambda t: entity.distance_to(t))

        return min(valid, key=lambda t: entity.distance_to(t))

    @staticmethod
    def _filter_by_type(target_type: str, candidates: List[Entity]) -> List[Entity]:
        if target_type == "all":
            return candidates
        if target_type == "ground":
            return [c for c in candidates if c.transport == "ground"]
        if target_type == "buildings":
            return [c for c in candidates if c.is_building or c.transport == "ground"]
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

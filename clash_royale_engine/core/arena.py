"""
Arena — manages entity spawning and tower placement.

The arena is the spatial container for all entities.  It sets up the
initial towers for both players and provides helpers to spawn troops
and apply spells.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from clash_royale_engine.entities.base_entity import Entity
from clash_royale_engine.entities.buildings.king_tower import KingTowerEntity, create_king_tower
from clash_royale_engine.entities.buildings.princess_tower import create_princess_tower
from clash_royale_engine.entities.spells.arrows import SpellEffect, apply_arrows
from clash_royale_engine.entities.spells.fireball import apply_fireball
from clash_royale_engine.entities.troops.archers import create_archers
from clash_royale_engine.entities.troops.giant import create_giant
from clash_royale_engine.entities.troops.knight import create_knight
from clash_royale_engine.entities.troops.mini_pekka import create_mini_pekka
from clash_royale_engine.entities.troops.musketeer import create_musketeer
from clash_royale_engine.entities.troops.skeletons import create_skeletons
from clash_royale_engine.utils.constants import TOWER_POSITIONS, DEFAULT_FPS

# Registry: card name → factory
_TROOP_FACTORIES = {
    "giant": lambda pid, x, y, fps: [create_giant(pid, x, y, fps)],
    "musketeer": lambda pid, x, y, fps: [create_musketeer(pid, x, y, fps)],
    "archers": lambda pid, x, y, fps: create_archers(pid, x, y, fps),
    "mini_pekka": lambda pid, x, y, fps: [create_mini_pekka(pid, x, y, fps)],
    "knight": lambda pid, x, y, fps: [create_knight(pid, x, y, fps)],
    "skeletons": lambda pid, x, y, fps: create_skeletons(pid, x, y, fps),
}

_SPELL_APPLIERS = {
    "arrows": apply_arrows,
    "fireball": apply_fireball,
}


class Arena:
    """Spatial container for all game entities."""

    def __init__(self, fps: int = DEFAULT_FPS) -> None:
        self.fps = fps
        self.entities: List[Entity] = []
        self.spell_effects: List[SpellEffect] = []

        # Quick-access references to towers
        self.towers: Dict[str, Entity] = {}

    # ── setup ─────────────────────────────────────────────────────────────

    def setup_towers(self) -> None:
        """Create the 6 towers (3 per player) and register them."""
        self.entities.clear()
        self.towers.clear()

        for player_key, positions in TOWER_POSITIONS.items():
            pid = int(player_key.split("_")[1])
            lp = positions["left_princess"]
            rp = positions["right_princess"]
            kg = positions["king"]

            left_p = create_princess_tower(pid, lp[0], lp[1], self.fps)
            right_p = create_princess_tower(pid, rp[0], rp[1], self.fps)
            king = create_king_tower(pid, kg[0], kg[1], self.fps)

            self.entities.extend([left_p, right_p, king])

            prefix = f"p{pid}"
            self.towers[f"{prefix}_left_princess"] = left_p
            self.towers[f"{prefix}_right_princess"] = right_p
            self.towers[f"{prefix}_king"] = king

    # ── spawning ──────────────────────────────────────────────────────────

    def spawn_troop(self, card_name: str, player_id: int, x: float, y: float) -> List[Entity]:
        """Spawn troop entities from a card. Returns the new entities."""
        factory = _TROOP_FACTORIES.get(card_name)
        if factory is None:
            raise ValueError(f"Unknown troop card: {card_name}")
        new_entities = factory(player_id, x, y, self.fps)
        self.entities.extend(new_entities)
        return new_entities

    def apply_spell(self, card_name: str, player_id: int, x: float, y: float) -> SpellEffect:
        """Apply a spell at the given position."""
        applier = _SPELL_APPLIERS.get(card_name)
        if applier is None:
            raise ValueError(f"Unknown spell card: {card_name}")
        effect = applier(player_id, x, y, self.entities)
        self.spell_effects.append(effect)
        return effect

    # ── queries ────────────────────────────────────────────────────────────

    def get_entities_for_player(self, player_id: int) -> List[Entity]:
        return [e for e in self.entities if e.player_id == player_id and not e.is_dead]

    def get_alive_entities(self) -> List[Entity]:
        return [e for e in self.entities if not e.is_dead]

    def cleanup_dead(self) -> List[Entity]:
        """Remove dead non-building entities. Returns removed list."""
        dead = [e for e in self.entities if e.is_dead and not e.is_building]
        self.entities = [e for e in self.entities if not (e.is_dead and not e.is_building)]
        return dead

    # ── tower HP helpers ──────────────────────────────────────────────────

    def tower_hp(self, player_id: int, tower_key: str) -> float:
        """Get HP of a specific tower. Returns 0 if destroyed."""
        key = f"p{player_id}_{tower_key}"
        tower = self.towers.get(key)
        if tower is None or tower.is_dead:
            return 0.0
        return float(tower.hp)

    def king_tower(self, player_id: int) -> Optional[KingTowerEntity]:
        key = f"p{player_id}_king"
        tower = self.towers.get(key)
        if tower is None or tower.is_dead:
            return None
        return tower if isinstance(tower, KingTowerEntity) else None

    def reset(self) -> None:
        self.entities.clear()
        self.towers.clear()
        self.spell_effects.clear()
        self.setup_towers()

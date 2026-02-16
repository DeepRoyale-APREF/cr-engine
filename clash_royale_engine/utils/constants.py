"""
All game constants in a single place.

Coordinate system, simulation config, elixir, speeds, tower configs,
and card stats for the 8 Arena-1 cards.
"""

from __future__ import annotations

from typing import Any, Dict

# ────────────────────────────── COORDINATE SYSTEM ──────────────────────────────
DISPLAY_WIDTH: int = 720
DISPLAY_HEIGHT: int = 1280
TILE_WIDTH: float = 34.0
TILE_HEIGHT: float = 27.6
N_WIDE_TILES: int = 18
N_HEIGHT_TILES: int = 32  # includes non-playable zone
PLAYABLE_HEIGHT_TILES: int = 15  # per player
TILE_INIT_X: float = 52.0
TILE_INIT_Y: float = 188.0

# ────────────────────────────── SIMULATION CONFIG ──────────────────────────────
DEFAULT_FPS: int = 30
GAME_DURATION: float = 180.0  # seconds (3 minutes)
OVERTIME_DURATION: float = 60.0  # seconds

# ────────────────────────────── ELIXIR ─────────────────────────────────────────
MAX_ELIXIR: float = 10.0
STARTING_ELIXIR: float = 5.0
ELIXIR_PER_SECOND: float = 1.0 / 2.8  # ~0.357
DOUBLE_ELIXIR_TIME: float = 120.0  # last minute
DOUBLE_ELIXIR_RATE: float = 2.0

# ────────────────────────────── MOVEMENT SPEEDS (pixels / second) ─────────────
SPEED_SLOW: float = 45.0  # Giant
SPEED_MEDIUM: float = 60.0  # Musketeer, Knight, Archers
SPEED_FAST: float = 90.0  # Mini PEKKA, Skeletons

# ────────────────────────────── DEPLOY ─────────────────────────────────────────
DEFAULT_DEPLOY_TIME: float = 1.0  # seconds

# ────────────────────────────── BRIDGE POSITIONS (tile coords) ─────────────────
# Bridges connect the two halves of the arena
BRIDGE_LEFT_X: int = 3
BRIDGE_RIGHT_X: int = 14
BRIDGE_Y: int = 15  # tile y where bridge sits (boundary between halves)

# River geometry — the river is a 2-tile-deep horizontal band that blocks
# ground movement.  Only bridge tiles allow crossing.
RIVER_Y_MIN: float = 15.0   # bottom edge of river (== BRIDGE_Y)
RIVER_Y_MAX: float = 17.0   # top edge of river (2 tiles deep)
BRIDGE_WIDTH: float = 2.0   # each bridge spans 2 tiles in x

# ────────────────────────────── TOWER CONFIGURATIONS ───────────────────────────
PRINCESS_TOWER_STATS: Dict[str, Any] = {
    "name": "princess_tower",
    "hp": 1400,
    "damage": 50,
    "hit_speed": 0.8,
    "range": 7.5,
    "sight_range": 7.5,
    "target": "all",
    "is_building": True,
    "hitbox_radius": 1.5,
    "projectile_speed": 800,
}

KING_TOWER_STATS: Dict[str, Any] = {
    "name": "king_tower",
    "hp": 2400,
    "damage": 50,
    "hit_speed": 1.0,
    "range": 7.0,
    "sight_range": 7.0,
    "target": "all",
    "is_building": True,
    "hitbox_radius": 2.0,
    "starts_active": False,
    "projectile_speed": 600,
}

# Tower positions (tile_x, tile_y) — from each player's perspective
# Player 0 towers are at the BOTTOM, Player 1 towers at the TOP
TOWER_POSITIONS: Dict[str, Dict[str, tuple[float, float]]] = {
    "player_0": {
        "left_princess": (3.0, 3.0),
        "right_princess": (14.0, 3.0),
        "king": (8.5, 0.5),
    },
    "player_1": {
        "left_princess": (3.0, 28.0),
        "right_princess": (14.0, 28.0),
        "king": (8.5, 31.0),
    },
}

# ────────────────────────────── CARD STATS ─────────────────────────────────────

CARD_STATS: Dict[str, Dict[str, Any]] = {
    "giant": {
        "name": "giant",
        "elixir": 5,
        "hp": 2000,
        "damage": 120,
        "hit_speed": 1.5,
        "speed": SPEED_SLOW,
        "range": 1.0,
        "sight_range": 5.5,
        "target": "buildings",
        "transport": "ground",
        "deploy_time": 1.0,
        "count": 1,
        "hitbox_radius": 0.9,
        "is_spell": False,
        "has_projectile": False,
    },
    "musketeer": {
        "name": "musketeer",
        "elixir": 4,
        "hp": 340,
        "damage": 100,
        "hit_speed": 1.0,
        "speed": SPEED_MEDIUM,
        "range": 6.0,
        "sight_range": 6.0,
        "target": "all",
        "transport": "ground",
        "deploy_time": 1.0,
        "count": 1,
        "hitbox_radius": 0.6,
        "is_spell": False,
        "has_projectile": True,
        "projectile_speed": 1000,
    },
    "archers": {
        "name": "archers",
        "elixir": 3,
        "hp": 125,
        "damage": 40,
        "hit_speed": 1.2,
        "speed": SPEED_MEDIUM,
        "range": 5.0,
        "sight_range": 5.5,
        "target": "all",
        "transport": "ground",
        "deploy_time": 1.0,
        "count": 2,
        "spawn_offset": 1.0,
        "hitbox_radius": 0.5,
        "is_spell": False,
        "has_projectile": True,
        "projectile_speed": 800,
    },
    "mini_pekka": {
        "name": "mini_pekka",
        "elixir": 4,
        "hp": 600,
        "damage": 325,
        "hit_speed": 1.8,
        "speed": SPEED_FAST,
        "range": 1.2,
        "sight_range": 5.5,
        "target": "ground",
        "transport": "ground",
        "deploy_time": 1.0,
        "count": 1,
        "hitbox_radius": 0.7,
        "is_spell": False,
        "has_projectile": False,
    },
    "knight": {
        "name": "knight",
        "elixir": 3,
        "hp": 600,
        "damage": 75,
        "hit_speed": 1.2,
        "speed": SPEED_MEDIUM,
        "range": 1.2,
        "sight_range": 5.5,
        "target": "ground",
        "transport": "ground",
        "deploy_time": 1.0,
        "count": 1,
        "hitbox_radius": 0.7,
        "is_spell": False,
        "has_projectile": False,
    },
    "skeletons": {
        "name": "skeletons",
        "elixir": 1,
        "hp": 32,
        "damage": 32,
        "hit_speed": 1.0,
        "speed": SPEED_FAST,
        "range": 1.0,
        "sight_range": 5.5,
        "target": "ground",
        "transport": "ground",
        "deploy_time": 1.0,
        "count": 3,
        "spawn_pattern": "triangle",
        "hitbox_radius": 0.4,
        "is_spell": False,
        "has_projectile": False,
        "is_swarm": True,
    },
    "arrows": {
        "name": "arrows",
        "elixir": 3,
        "damage": 115,
        "radius": 4.0,
        "is_spell": True,
        "target": "all",
        "deploy_time": 0.0,
        "damage_type": "area",
        "crown_tower_damage": 46,
    },
    "fireball": {
        "name": "fireball",
        "elixir": 4,
        "damage": 325,
        "radius": 2.5,
        "knockback": 2.0,
        "is_spell": True,
        "target": "all",
        "deploy_time": 0.0,
        "damage_type": "area",
        "crown_tower_damage": 130,
    },
}

# ────────────────────────────── DECK DEFAULTS ──────────────────────────────────
DEFAULT_DECK: list[str] = [
    "giant",
    "musketeer",
    "archers",
    "mini_pekka",
    "knight",
    "skeletons",
    "arrows",
    "fireball",
]
HAND_SIZE: int = 4

# ────────────────────────────── OBSERVATION ────────────────────────────────────
CARD_VOCAB: list[str] = DEFAULT_DECK  # for one-hot encoding
OBS_FEATURE_DIM: int = 2 + 6 + (len(CARD_VOCAB) + 1) * HAND_SIZE + 2 * N_HEIGHT_TILES * N_WIDE_TILES
# 2 (elixir) + 6 (tower HP) + 36 (4 cards × 9) + 1152*2 (grids) = 2348

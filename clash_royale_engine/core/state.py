"""
State definitions compatible with BuildABot.

Provides the canonical :class:`State` dataclass and all its components
that represent a full snapshot of the game from one player's perspective.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Position:
    """Pixel + tile position of a detection."""

    bbox: Tuple[int, int, int, int]  # (left, top, right, bottom) in pixels
    conf: float  # confidence — always 1.0 for the simulator
    tile_x: int  # [0, 17]
    tile_y: int  # [0, 31]


@dataclass
class Unit:
    """Static descriptor of a unit type (troop or building)."""

    name: str  # e.g. "giant", "musketeer"
    category: str  # "troop" | "building"
    target: str  # "all" | "ground" | "buildings"
    transport: str  # "ground" | "air"


@dataclass
class UnitDetection:
    """A unit together with its current position."""

    unit: Unit
    position: Position


@dataclass
class Numbers:
    """Numeric scalars visible to a player."""

    elixir: float  # [0, 10]
    enemy_elixir: float  # estimated or perfect depending on config
    left_princess_hp: float
    right_princess_hp: float
    king_hp: float
    left_enemy_princess_hp: float
    right_enemy_princess_hp: float
    enemy_king_hp: float
    time_remaining: float  # seconds


@dataclass
class Card:
    """A card in the player's hand."""

    name: str
    is_spell: bool
    cost: int
    units: List[Unit] = field(default_factory=list)


@dataclass
class State:
    """Full game state compatible with BuildABot."""

    allies: List[UnitDetection]
    enemies: List[UnitDetection]
    numbers: Numbers
    cards: Tuple[Card, Card, Card, Card]
    ready: List[int]  # indices of playable cards (enough elixir)
    screen: str = "battle"  # always "battle" during simulation

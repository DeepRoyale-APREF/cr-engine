"""
Clash Royale Arena 1 Simulation Engine for RL Training.

A high-performance, headless simulation engine for Clash Royale (Arena 1)
focused on 8 specific cards, with realistic physics, modular architecture,
and optimized for massive-scale RL agent training.
"""

from clash_royale_engine.core.engine import ClashRoyaleEngine
from clash_royale_engine.core.state import (
    Card,
    Numbers,
    Position,
    State,
    Unit,
    UnitDetection,
)
from clash_royale_engine.env.gymnasium_env import ClashRoyaleEnv
from clash_royale_engine.players.player_interface import (
    HeuristicBot,
    PlayerInterface,
    RLAgentPlayer,
)

__version__ = "0.1.0"

__all__ = [
    "ClashRoyaleEngine",
    "ClashRoyaleEnv",
    "Card",
    "Numbers",
    "Position",
    "State",
    "Unit",
    "UnitDetection",
    "PlayerInterface",
    "HeuristicBot",
    "RLAgentPlayer",
]

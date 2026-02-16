"""
Player-agnostic interfaces: abstract base, RL wrapper, heuristic bot, human.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

from clash_royale_engine.core.state import State
from clash_royale_engine.utils.constants import N_HEIGHT_TILES


class PlayerInterface(ABC):
    """
    Abstract interface that allows:
    - RL agents
    - Human input
    - Heuristic bots
    """

    @abstractmethod
    def get_action(self, state: State) -> Optional[Tuple[int, int, int]]:
        """
        Return an action given the current state.

        Returns
        -------
        None
            Do nothing this frame.
        (tile_x, tile_y, card_index)
            Play the card at the given position.
        """

    @abstractmethod
    def reset(self) -> None:
        """Called at the start of each episode."""


class RLAgentPlayer(PlayerInterface):
    """Wrapper for RL models (Stable-Baselines3, RLlib, etc.)."""

    def __init__(self, model: Any = None, policy_type: str = "stochastic") -> None:
        self.model = model
        self.policy_type = policy_type

    def get_action(self, state: State) -> Optional[Tuple[int, int, int]]:
        """Placeholder — the Gymnasium env drives the RL agent externally."""
        return None  # actions are injected by the env wrapper

    def reset(self) -> None:
        pass


class HeuristicBot(PlayerInterface):
    """
    Simple rule-based bot for testing / as default opponent.

    Strategy:
    - Accumulate elixir until >= 7 (or 10).
    - Play the highest-cost card that is affordable,
      at a default strategic tile on own side.
    """

    def __init__(self, aggression: float = 0.5) -> None:
        self.aggression = aggression  # 0 = very passive, 1 = very aggressive
        self._rng = random.Random(42)

    def get_action(self, state: State) -> Optional[Tuple[int, int, int]]:
        elixir = state.numbers.elixir
        threshold = 10.0 - self.aggression * 3.0  # 7–10

        if elixir < threshold and not state.ready:
            return None

        if not state.ready:
            return None

        # Pick a random playable card
        card_idx = self._rng.choice(state.ready)
        card = state.cards[card_idx]

        # Default placement: centre of own half
        # (we receive state already from *our* perspective,
        #  so tile_y in [0, 14] is own side)
        tile_x = self._rng.randint(4, 13)
        if card.is_spell:
            # Spells toward enemy half
            tile_y = self._rng.randint(N_HEIGHT_TILES // 2, N_HEIGHT_TILES - 5)
        else:
            tile_y = self._rng.randint(3, 12)

        return (tile_x, tile_y, card_idx)

    def reset(self) -> None:
        self._rng = random.Random(42)


class HumanPlayer(PlayerInterface):
    """For manual play or debugging (actions are pushed into a queue)."""

    def __init__(self) -> None:
        from collections import deque

        self._queue: "deque[Tuple[int, int, int]]" = deque()

    def push_action(self, action: Tuple[int, int, int]) -> None:
        self._queue.append(action)

    def get_action(self, state: State) -> Optional[Tuple[int, int, int]]:
        if self._queue:
            return self._queue.popleft()
        return None

    def reset(self) -> None:
        self._queue.clear()

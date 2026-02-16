"""
Elixir management system.

Handles per-player elixir generation, double-elixir mode, and capping.
"""

from __future__ import annotations

from clash_royale_engine.utils.constants import (
    DOUBLE_ELIXIR_RATE,
    ELIXIR_PER_SECOND,
    MAX_ELIXIR,
    STARTING_ELIXIR,
)


class ElixirSystem:
    """Tracks elixir for both players."""

    def __init__(self, fps: int = 30) -> None:
        self.fps: int = fps
        self.dt: float = 1.0 / fps
        self.elixir: list[float] = [STARTING_ELIXIR, STARTING_ELIXIR]

    def update(self, is_double_elixir: bool) -> None:
        """Add elixir for one frame."""
        rate = ELIXIR_PER_SECOND * (DOUBLE_ELIXIR_RATE if is_double_elixir else 1.0)
        increment = rate * self.dt
        for i in range(2):
            self.elixir[i] = min(MAX_ELIXIR, self.elixir[i] + increment)

    def spend(self, player_id: int, amount: float) -> bool:
        """
        Try to spend *amount* elixir for *player_id*.

        Returns ``True`` on success, ``False`` if not enough elixir.
        """
        if self.elixir[player_id] < amount:
            return False
        self.elixir[player_id] -= amount
        return True

    def get(self, player_id: int) -> float:
        return self.elixir[player_id]

    def reset(self) -> None:
        self.elixir = [STARTING_ELIXIR, STARTING_ELIXIR]

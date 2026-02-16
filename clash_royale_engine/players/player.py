"""
Player — manages deck, hand, and card cycling.
"""

from __future__ import annotations

import random
from typing import List, Optional

from clash_royale_engine.utils.constants import CARD_STATS, DEFAULT_DECK, HAND_SIZE


class Player:
    """Per-player state: deck order, current hand, next card."""

    def __init__(self, player_id: int, deck: Optional[List[str]] = None, seed: int = 0) -> None:
        self.player_id: int = player_id
        self.deck: List[str] = list(deck or DEFAULT_DECK)
        self._rng = random.Random(seed + player_id)

        # Shuffle deck and deal initial hand
        self._rng.shuffle(self.deck)
        self.hand: List[str] = self.deck[:HAND_SIZE]
        self._queue: List[str] = self.deck[HAND_SIZE:]
        self._discard: List[str] = []

    # ── elixir is managed by ElixirSystem; stored here for convenience ────
    @property
    def elixir(self) -> float:
        """Convenience — actual value lives in ElixirSystem.  Use engine to read."""
        return self._elixir

    @elixir.setter
    def elixir(self, value: float) -> None:
        self._elixir = value

    # ── card management ───────────────────────────────────────────────────

    def play_card(self, card_idx: int) -> str:
        """Remove card at *card_idx* from hand and cycle the next card in."""
        card_name = self.hand[card_idx]
        self._discard.append(card_name)

        # Cycle
        if self._queue:
            self.hand[card_idx] = self._queue.pop(0)
        else:
            # Reshuffle discard → queue
            self._queue = list(self._discard)
            self._rng.shuffle(self._queue)
            self._discard = []
            self.hand[card_idx] = self._queue.pop(0)

        return card_name

    def playable_indices(self, elixir: float) -> List[int]:
        """Return indices of cards in hand that can be played with current elixir."""
        result: List[int] = []
        for i, name in enumerate(self.hand):
            cost = CARD_STATS[name]["elixir"]
            if elixir >= cost:
                result.append(i)
        return result

    def reset(self, seed: int = 0) -> None:
        self._rng = random.Random(seed + self.player_id)
        self._rng.shuffle(self.deck)
        self.hand = self.deck[:HAND_SIZE]
        self._queue = self.deck[HAND_SIZE:]
        self._discard = []
        self._elixir = 0.0

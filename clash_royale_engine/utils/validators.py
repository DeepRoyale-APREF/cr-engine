"""
Action and placement validation utilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

from clash_royale_engine.utils.constants import (
    BRIDGE_Y,
    CARD_STATS,
    N_HEIGHT_TILES,
    N_WIDE_TILES,
    PLAYABLE_HEIGHT_TILES,
)

if TYPE_CHECKING:
    from clash_royale_engine.players.player import Player


class InvalidActionError(Exception):
    """Raised when an action is not valid."""


def validate_placement(
    player_id: int,
    tile_x: int,
    tile_y: int,
    card_name: str,
    player: "Player",
) -> Optional[str]:
    """
    Return ``None`` if placement is valid, otherwise a human-readable error string.

    Rules
    -----
    * Tile must be inside the grid.
    * Tile must be on the player's side of the arena.
    * Spells can be placed anywhere on the grid.
    * Player must have enough elixir.
    * Card must be in the player's hand.
    """
    # Bounds check
    if not (0 <= tile_x < N_WIDE_TILES and 0 <= tile_y < N_HEIGHT_TILES):
        return f"Tile ({tile_x}, {tile_y}) out of bounds"

    stats = CARD_STATS.get(card_name)
    if stats is None:
        return f"Unknown card: {card_name}"

    is_spell = stats.get("is_spell", False)

    # Placement zone — troops can only be placed on own side
    if not is_spell:
        if player_id == 0:
            # Player 0 (bottom): tiles 0..PLAYABLE_HEIGHT_TILES-1
            if tile_y >= BRIDGE_Y:
                return f"Player 0 cannot place troops at tile_y={tile_y}"
        else:
            # Player 1 (top): tiles (N_HEIGHT_TILES - PLAYABLE_HEIGHT_TILES)..N_HEIGHT_TILES-1
            if tile_y < (N_HEIGHT_TILES - PLAYABLE_HEIGHT_TILES):
                return f"Player 1 cannot place troops at tile_y={tile_y}"

    # Elixir check
    cost = stats["elixir"]
    if player.elixir < cost:
        return f"Not enough elixir ({player.elixir:.1f} < {cost})"

    return None  # valid


def validate_action(
    player_id: int,
    action: Tuple[int, int, int],
    player: "Player",
) -> Optional[str]:
    """Validate a full ``(tile_x, tile_y, card_index)`` action tuple."""
    tile_x, tile_y, card_idx = action

    if card_idx < 0 or card_idx >= len(player.hand):
        return f"Invalid card index: {card_idx}"

    card_name = player.hand[card_idx]
    return validate_placement(player_id, tile_x, tile_y, card_name, player)

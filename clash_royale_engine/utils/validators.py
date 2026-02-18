"""
Action and placement validation utilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

from clash_royale_engine.utils.constants import (
    BRIDGE_Y,
    CARD_STATS,
    LANE_DIVIDER_X,
    N_HEIGHT_TILES,
    N_WIDE_TILES,
    POCKET_DEPTH,
    RIVER_Y_MAX,
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
    *,
    enemy_left_princess_dead: bool = False,
    enemy_right_princess_dead: bool = False,
) -> Optional[str]:
    """
    Return ``None`` if placement is valid, otherwise a human-readable error string.

    Rules
    -----
    * Tile must be inside the grid.
    * Tile must be on the player's side of the arena.
    * **Spells** can be placed anywhere on the grid.
    * When an enemy **princess tower** is destroyed the attacker may deploy
      troops in the corresponding **pocket** (left or right lane, up to
      ``POCKET_DEPTH`` tiles past the river) on the enemy's side.
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

    # Placement zone — spells have no side restriction
    if not is_spell:
        on_enemy_side = False

        # Both players see the world from their own perspective:
        # tile_y in [0, BRIDGE_Y-1] is "own side", tile_y >= BRIDGE_Y is "enemy side".
        # The engine flips P1's coordinates in _apply_action, so validation
        # uses the same player-relative frame for both players.
        if tile_y >= BRIDGE_Y:
            on_enemy_side = True

        if on_enemy_side:
            # Check whether the placement is inside an unlocked pocket
            allowed = _is_pocket_allowed(
                player_id,
                tile_x,
                tile_y,
                enemy_left_princess_dead,
                enemy_right_princess_dead,
            )
            if not allowed:
                return (
                    f"Player {player_id} cannot place troops at "
                    f"tile ({tile_x}, {tile_y}) — enemy princess tower "
                    f"for that lane is still standing"
                )

    # Elixir check
    cost = stats["elixir"]
    if player.elixir < cost:
        return f"Not enough elixir ({player.elixir:.1f} < {cost})"

    return None  # valid


def _is_pocket_allowed(
    player_id: int,
    tile_x: int,
    tile_y: int,
    enemy_left_dead: bool,
    enemy_right_dead: bool,
) -> bool:
    """Return *True* if *(tile_x, tile_y)* is inside an unlocked pocket.

    The *pocket* is the area POCKET_DEPTH tiles past the river in the
    enemy's half, restricted to the lane whose princess tower was destroyed.

    Left lane:  tile_x < LANE_DIVIDER_X (9)
    Right lane: tile_x >= LANE_DIVIDER_X (9)
    """
    # Determine which lane the tile is in
    in_left_lane = tile_x < LANE_DIVIDER_X
    lane_unlocked = (in_left_lane and enemy_left_dead) or (not in_left_lane and enemy_right_dead)
    if not lane_unlocked:
        return False

    # Verify the tile is within the pocket depth (not deep into enemy base).
    # Both players use the same coordinate frame (own side = low y).
    # The pocket is just past the river on the enemy's side.
    pocket_min_y = int(RIVER_Y_MAX)  # 17
    pocket_max_y = int(RIVER_Y_MAX) + POCKET_DEPTH - 1  # 19
    return pocket_min_y <= tile_y <= pocket_max_y


def validate_action(
    player_id: int,
    action: Tuple[int, int, int],
    player: "Player",
    *,
    enemy_left_princess_dead: bool = False,
    enemy_right_princess_dead: bool = False,
) -> Optional[str]:
    """Validate a full ``(tile_x, tile_y, card_index)`` action tuple."""
    tile_x, tile_y, card_idx = action

    if card_idx < 0 or card_idx >= len(player.hand):
        return f"Invalid card index: {card_idx}"

    card_name = player.hand[card_idx]
    return validate_placement(
        player_id,
        tile_x,
        tile_y,
        card_name,
        player,
        enemy_left_princess_dead=enemy_left_princess_dead,
        enemy_right_princess_dead=enemy_right_princess_dead,
    )

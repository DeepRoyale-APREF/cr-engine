"""
Tile ↔ pixel coordinate conversions (BuildABot-compatible).
"""

from __future__ import annotations

from typing import Tuple

from clash_royale_engine.utils.constants import (
    DISPLAY_HEIGHT,
    TILE_HEIGHT,
    TILE_INIT_X,
    TILE_INIT_Y,
    TILE_WIDTH,
)


def pixel_to_tile(x_pixel: float, y_pixel: float) -> Tuple[int, int]:
    """Convert pixel coordinates to tile coordinates."""
    tile_x = round(((x_pixel - TILE_INIT_X) / TILE_WIDTH) - 0.5)
    tile_y = round(((DISPLAY_HEIGHT - TILE_INIT_Y - y_pixel) / TILE_HEIGHT) - 0.5)
    return tile_x, tile_y


def tile_to_pixel(tile_x: float, tile_y: float) -> Tuple[float, float]:
    """Convert tile coordinates to pixel coordinates (centre of tile)."""
    x_pixel = TILE_INIT_X + (tile_x + 0.5) * TILE_WIDTH
    y_pixel = DISPLAY_HEIGHT - TILE_INIT_Y - (tile_y + 0.5) * TILE_HEIGHT
    return x_pixel, y_pixel


def tile_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance between two tile positions."""
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

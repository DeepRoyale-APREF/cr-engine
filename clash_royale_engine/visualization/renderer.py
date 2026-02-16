"""
Pygame renderer — simplified Clash Royale GUI.

Renders the arena grid, towers, troops, elixir bar, card hand and timer
in a window that mirrors the real game layout.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

if TYPE_CHECKING:
    from clash_royale_engine.core.state import State, UnitDetection

# ── Lazy pygame import (only when renderer is actually used) ──────────
_pg: Any = None


def _ensure_pygame() -> Any:
    global _pg
    if _pg is None:
        import pygame

        _pg = pygame
    return _pg


# ══════════════════════════════════════════════════════════════════════════
# Colour palette
# ══════════════════════════════════════════════════════════════════════════

# General
COL_BG = (58, 46, 38)  # dark brown (arena floor)
COL_GRID = (75, 62, 53)  # subtle grid lines
COL_BRIDGE = (139, 119, 101)  # bridge colour
COL_RIVER = (60, 120, 180)  # river / gap between halves
COL_TEXT = (255, 255, 255)
COL_TEXT_SHADOW = (30, 30, 30)
COL_PANEL_BG = (35, 28, 22)  # bottom HUD
COL_ELIXIR_FILL = (200, 50, 200)  # pink/purple elixir
COL_ELIXIR_BG = (60, 20, 60)
COL_CARD_BG = (50, 50, 70)
COL_CARD_READY = (70, 130, 70)
COL_CARD_BORDER = (180, 160, 130)
COL_HP_BAR_BG = (50, 50, 50)

# Per-player colours
COL_ALLY = (80, 160, 255)  # blue
COL_ENEMY = (255, 70, 70)  # red
COL_ALLY_TOWER = (50, 120, 220)
COL_ENEMY_TOWER = (220, 50, 50)
COL_ALLY_KING = (30, 80, 190)
COL_ENEMY_KING = (190, 30, 30)

# Troop-specific accent colours
TROOP_COLOURS: Dict[str, Tuple[int, int, int]] = {
    "giant": (200, 160, 80),
    "musketeer": (180, 100, 180),
    "archers": (100, 200, 100),
    "mini_pekka": (80, 80, 220),
    "knight": (180, 180, 80),
    "skeletons": (210, 210, 210),
    "princess_tower": (160, 160, 160),
    "king_tower": (220, 200, 60),
}

# ══════════════════════════════════════════════════════════════════════════
# Layout constants (render space — NOT game coordinates)
# ══════════════════════════════════════════════════════════════════════════

ARENA_TILES_W = 18
ARENA_TILES_H = 32
CELL_SIZE = 22  # pixels per tile cell
ARENA_W = ARENA_TILES_W * CELL_SIZE  # 396
ARENA_H = ARENA_TILES_H * CELL_SIZE  # 704

MARGIN_TOP = 50  # space for timer header
MARGIN_BOTTOM = 160  # space for HUD (cards + elixir)
MARGIN_SIDE = 20

WIN_W = ARENA_W + 2 * MARGIN_SIDE
WIN_H = MARGIN_TOP + ARENA_H + MARGIN_BOTTOM

# River row(s) — visual only
RIVER_Y_TILE = 15  # boundary between halves (same as BRIDGE_Y)
BRIDGE_LEFT_X = 3
BRIDGE_RIGHT_X = 14

# ══════════════════════════════════════════════════════════════════════════
# Helper drawing functions
# ══════════════════════════════════════════════════════════════════════════


def _tile_to_screen(tile_x: float, tile_y: float) -> Tuple[int, int]:
    """Convert tile coordinates → screen pixel (centre of cell).

    tile_y=0 is the BOTTOM of the arena (player 0 king).
    On screen y=0 is top, so we flip vertically.
    """
    sx = MARGIN_SIDE + int(tile_x * CELL_SIZE) + CELL_SIZE // 2
    # Flip: tile_y 0 → bottom of arena on screen
    sy = MARGIN_TOP + ARENA_H - int(tile_y * CELL_SIZE) - CELL_SIZE // 2
    return sx, sy


def _draw_text(
    surface: Any,
    text: str,
    pos: Tuple[int, int],
    font: Any,
    colour: Tuple[int, int, int] = COL_TEXT,
    shadow: bool = True,
    center: bool = False,
) -> None:
    """Render text with optional drop shadow."""
    _ensure_pygame()
    if shadow:
        shadow_surf = font.render(text, True, COL_TEXT_SHADOW)
        r = shadow_surf.get_rect()
        if center:
            r.center = (pos[0] + 1, pos[1] + 1)
        else:
            r.topleft = (pos[0] + 1, pos[1] + 1)
        surface.blit(shadow_surf, r)

    text_surf = font.render(text, True, colour)
    r = text_surf.get_rect()
    if center:
        r.center = pos
    else:
        r.topleft = pos
    surface.blit(text_surf, r)


# ══════════════════════════════════════════════════════════════════════════
# Main Renderer class
# ══════════════════════════════════════════════════════════════════════════


class Renderer:
    """Pygame-based simplified Clash Royale GUI.

    Displays the arena, towers, troops, elixir, cards and timer.
    Call :meth:`render` each frame with the current :class:`State`.

    Parameters
    ----------
    fps : int
        Target rendering framerate.
    title : str
        Window title.
    """

    def __init__(self, fps: int = 30, title: str = "Clash Royale Engine") -> None:
        self.fps = fps
        self.title = title

        self._screen: Any = None
        self._clock: Any = None
        self._font_sm: Any = None
        self._font_md: Any = None
        self._font_lg: Any = None
        self._font_xl: Any = None
        self._initialised: bool = False

    # ── lifecycle ──────────────────────────────────────────────────────────

    def _init_pygame(self) -> None:
        if self._initialised:
            return
        pg = _ensure_pygame()
        pg.init()
        pg.display.set_caption(self.title)
        self._screen = pg.display.set_mode((WIN_W, WIN_H))
        self._clock = pg.time.Clock()
        self._font_sm = pg.font.SysFont("consolas", 12)
        self._font_md = pg.font.SysFont("consolas", 15, bold=True)
        self._font_lg = pg.font.SysFont("consolas", 20, bold=True)
        self._font_xl = pg.font.SysFont("consolas", 28, bold=True)
        self._initialised = True

    def close(self) -> None:
        """Shut down the pygame window."""
        if self._initialised:
            pg = _ensure_pygame()
            pg.quit()
            self._initialised = False

    def poll_events(self) -> bool:
        """Process pygame events. Returns ``False`` if user closed the window."""
        pg = _ensure_pygame()
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return False
            if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                return False
        return True

    # ── main render ───────────────────────────────────────────────────────

    def render(self, state: "State") -> bool:
        """Draw one frame. Returns ``False`` if window was closed."""
        self._init_pygame()
        pg = _ensure_pygame()

        if not self.poll_events():
            return False

        self._screen.fill(COL_PANEL_BG)

        # Arena
        self._draw_arena()
        self._draw_river_and_bridges()

        # Entities
        self._draw_units(state.allies, is_ally=True)
        self._draw_units(state.enemies, is_ally=False)

        # Tower HP bars
        self._draw_tower_hp(state)

        # Top HUD — timer + scores
        self._draw_top_hud(state)

        # Bottom HUD — cards + elixir
        self._draw_bottom_hud(state)

        pg.display.flip()
        self._clock.tick(self.fps)
        return True

    # ── arena grid ────────────────────────────────────────────────────────

    def _draw_arena(self) -> None:
        pg = _ensure_pygame()
        # Arena background
        arena_rect = pg.Rect(MARGIN_SIDE, MARGIN_TOP, ARENA_W, ARENA_H)
        pg.draw.rect(self._screen, COL_BG, arena_rect)

        # Grid lines
        for tx in range(ARENA_TILES_W + 1):
            x = MARGIN_SIDE + tx * CELL_SIZE
            pg.draw.line(self._screen, COL_GRID, (x, MARGIN_TOP), (x, MARGIN_TOP + ARENA_H))
        for ty in range(ARENA_TILES_H + 1):
            y = MARGIN_TOP + ty * CELL_SIZE
            pg.draw.line(self._screen, COL_GRID, (MARGIN_SIDE, y), (MARGIN_SIDE + ARENA_W, y))

    def _draw_river_and_bridges(self) -> None:
        pg = _ensure_pygame()
        # River band (2 tile rows around RIVER_Y_TILE)
        river_screen_y = MARGIN_TOP + ARENA_H - (RIVER_Y_TILE + 1) * CELL_SIZE
        river_rect = pg.Rect(MARGIN_SIDE, river_screen_y, ARENA_W, CELL_SIZE * 2)
        pg.draw.rect(self._screen, COL_RIVER, river_rect)

        # Bridges
        for bx in (BRIDGE_LEFT_X, BRIDGE_RIGHT_X):
            bsx = MARGIN_SIDE + bx * CELL_SIZE
            bridge_rect = pg.Rect(bsx, river_screen_y, CELL_SIZE * 2, CELL_SIZE * 2)
            pg.draw.rect(self._screen, COL_BRIDGE, bridge_rect)
            pg.draw.rect(self._screen, (100, 85, 70), bridge_rect, 2)

    # ── units (troops + towers) ──────────────────────────────────────────

    def _draw_units(self, units: "List[UnitDetection]", *, is_ally: bool) -> None:
        pg = _ensure_pygame()
        base_colour = COL_ALLY if is_ally else COL_ENEMY

        for det in units:
            tx = det.position.tile_x
            ty = det.position.tile_y
            sx, sy = _tile_to_screen(tx, ty)
            name = det.unit.name
            is_building = det.unit.category == "building"

            accent = TROOP_COLOURS.get(name, base_colour)

            if is_building:
                # Towers: filled square with border
                size = int(CELL_SIZE * 1.6)
                half = size // 2
                is_king = "king" in name
                fill = (COL_ALLY_KING if is_ally else COL_ENEMY_KING) if is_king else (
                    COL_ALLY_TOWER if is_ally else COL_ENEMY_TOWER
                )
                rect = pg.Rect(sx - half, sy - half, size, size)
                pg.draw.rect(self._screen, fill, rect, border_radius=4)
                pg.draw.rect(self._screen, accent, rect, 2, border_radius=4)
                # Crown icon for king
                if is_king:
                    _draw_text(self._screen, "♛", (sx, sy - 2), self._font_md, accent, center=True)
                else:
                    _draw_text(self._screen, "⚑", (sx, sy - 2), self._font_md, accent, center=True)
            else:
                # Troops: filled circle with outline
                radius = max(5, int(CELL_SIZE * 0.4))
                pg.draw.circle(self._screen, accent, (sx, sy), radius)
                pg.draw.circle(self._screen, base_colour, (sx, sy), radius, 2)
                # Name initial
                initial = name[0].upper()
                if name == "mini_pekka":
                    initial = "P"
                elif name == "skeletons":
                    initial = "S"
                _draw_text(
                    self._screen, initial, (sx, sy),
                    self._font_sm, COL_TEXT, shadow=False, center=True,
                )

    # ── tower HP bars ─────────────────────────────────────────────────────

    def _draw_tower_hp(self, state: "State") -> None:
        """Draw HP bars near each tower position."""
        pg = _ensure_pygame()
        bar_w, bar_h = 40, 5

        # Player 0 (ally — bottom)
        towers_ally = [
            (3.0, 3.0, state.numbers.left_princess_hp, 1400, COL_ALLY),
            (14.0, 3.0, state.numbers.right_princess_hp, 1400, COL_ALLY),
            (8.5, 0.5, state.numbers.king_hp, 2400, COL_ALLY),
        ]
        # Player 1 (enemy — top)
        towers_enemy = [
            (3.0, 28.0, state.numbers.left_enemy_princess_hp, 1400, COL_ENEMY),
            (14.0, 28.0, state.numbers.right_enemy_princess_hp, 1400, COL_ENEMY),
            (8.5, 31.0, state.numbers.enemy_king_hp, 2400, COL_ENEMY),
        ]

        for tx, ty, hp, max_hp, col in towers_ally + towers_enemy:
            sx, sy = _tile_to_screen(tx, ty)
            bx = sx - bar_w // 2
            by = sy - int(CELL_SIZE * 1.2)

            # Background
            pg.draw.rect(self._screen, COL_HP_BAR_BG, (bx, by, bar_w, bar_h))
            # Fill
            ratio = max(0.0, min(1.0, hp / max_hp)) if max_hp > 0 else 0
            fill_w = int(bar_w * ratio)
            if fill_w > 0:
                pg.draw.rect(self._screen, col, (bx, by, fill_w, bar_h))
            # Border
            pg.draw.rect(self._screen, COL_TEXT, (bx, by, bar_w, bar_h), 1)

            # HP number
            hp_text = str(int(hp))
            _draw_text(
                self._screen, hp_text, (sx, by - 12),
                self._font_sm, col, shadow=True, center=True,
            )

    # ── top HUD (timer, crowns) ──────────────────────────────────────────

    def _draw_top_hud(self, state: "State") -> None:
        pg = _ensure_pygame()
        # Background bar
        top_rect = pg.Rect(0, 0, WIN_W, MARGIN_TOP)
        pg.draw.rect(self._screen, COL_PANEL_BG, top_rect)

        # Timer
        t = max(0, state.numbers.time_remaining)
        minutes = int(t) // 60
        seconds = int(t) % 60
        time_str = f"{minutes}:{seconds:02d}"
        _draw_text(
            self._screen, time_str, (WIN_W // 2, MARGIN_TOP // 2),
            self._font_xl, COL_TEXT, center=True,
        )

        # "Time left" label
        _draw_text(
            self._screen, "Time left", (WIN_W // 2, 6),
            self._font_sm, (180, 180, 180), shadow=False, center=True,
        )

        # Crown counts
        ally_crowns = sum(
            1
            for hp in [
                state.numbers.left_enemy_princess_hp,
                state.numbers.right_enemy_princess_hp,
            ]
            if hp <= 0
        ) + (3 if state.numbers.enemy_king_hp <= 0 else 0)

        enemy_crowns = sum(
            1
            for hp in [
                state.numbers.left_princess_hp,
                state.numbers.right_princess_hp,
            ]
            if hp <= 0
        ) + (3 if state.numbers.king_hp <= 0 else 0)

        _draw_text(
            self._screen, f"★{ally_crowns}", (MARGIN_SIDE + 10, MARGIN_TOP // 2),
            self._font_lg, COL_ALLY, center=True,
        )
        _draw_text(
            self._screen, f"★{enemy_crowns}", (WIN_W - MARGIN_SIDE - 10, MARGIN_TOP // 2),
            self._font_lg, COL_ENEMY, center=True,
        )

        # Double elixir indicator
        if t <= 60.0 and t > 0:
            _draw_text(
                self._screen, "x2", (WIN_W - 50, 6),
                self._font_md, COL_ELIXIR_FILL, shadow=True, center=True,
            )

    # ── bottom HUD (cards + elixir) ──────────────────────────────────────

    def _draw_bottom_hud(self, state: "State") -> None:
        pg = _ensure_pygame()
        hud_y = MARGIN_TOP + ARENA_H

        # Background
        hud_rect = pg.Rect(0, hud_y, WIN_W, MARGIN_BOTTOM)
        pg.draw.rect(self._screen, COL_PANEL_BG, hud_rect)
        pg.draw.line(self._screen, (80, 65, 50), (0, hud_y), (WIN_W, hud_y), 2)

        # ── Elixir bar ──
        elixir = state.numbers.elixir
        bar_x = MARGIN_SIDE + 40
        bar_y = hud_y + 10
        bar_w = ARENA_W - 50
        bar_h = 18

        # Background
        pg.draw.rect(self._screen, COL_ELIXIR_BG, (bar_x, bar_y, bar_w, bar_h), border_radius=4)
        # Fill
        fill_ratio = max(0.0, min(1.0, elixir / 10.0))
        fill_w = int(bar_w * fill_ratio)
        if fill_w > 0:
            pg.draw.rect(
                self._screen, COL_ELIXIR_FILL,
                (bar_x, bar_y, fill_w, bar_h), border_radius=4,
            )
        # Border
        pg.draw.rect(self._screen, (120, 60, 120), (bar_x, bar_y, bar_w, bar_h), 2, border_radius=4)
        # Tick marks
        for i in range(1, 10):
            tick_x = bar_x + int(bar_w * i / 10)
            pg.draw.line(
                self._screen, (120, 60, 120),
                (tick_x, bar_y), (tick_x, bar_y + bar_h),
            )

        # Elixir number + icon
        _draw_text(
            self._screen, f"{int(elixir)}",
            (MARGIN_SIDE + 14, bar_y + bar_h // 2),
            self._font_lg, COL_ELIXIR_FILL, center=True,
        )

        # ── Cards ──
        card_y = bar_y + bar_h + 12
        card_w = (ARENA_W - 30) // 4
        card_h = 90
        ready_indices = set(state.ready)

        for i, card in enumerate(state.cards):
            cx = MARGIN_SIDE + 10 + i * (card_w + 6)
            is_ready = i in ready_indices

            # Card background
            bg = COL_CARD_READY if is_ready else COL_CARD_BG
            card_rect = pg.Rect(cx, card_y, card_w, card_h)
            pg.draw.rect(self._screen, bg, card_rect, border_radius=6)
            pg.draw.rect(self._screen, COL_CARD_BORDER, card_rect, 2, border_radius=6)

            if card.name == "empty":
                continue

            # Card name
            display_name = card.name.replace("_", " ").title()
            if len(display_name) > 10:
                display_name = display_name[:9] + "."
            _draw_text(
                self._screen, display_name,
                (cx + card_w // 2, card_y + 14),
                self._font_sm, COL_TEXT, center=True,
            )

            # Troop icon (coloured dot or symbol)
            icon_colour = TROOP_COLOURS.get(card.name, COL_TEXT)
            icon_y = card_y + 40
            if card.is_spell:
                # Draw a small starburst for spells
                for angle in range(0, 360, 45):
                    rad = math.radians(angle)
                    ex = int(cx + card_w // 2 + 10 * math.cos(rad))
                    ey = int(icon_y + 10 * math.sin(rad))
                    pg.draw.line(
                        self._screen, icon_colour,
                        (cx + card_w // 2, icon_y), (ex, ey), 2,
                    )
            else:
                pg.draw.circle(self._screen, icon_colour, (cx + card_w // 2, icon_y), 10)
                pg.draw.circle(self._screen, COL_TEXT, (cx + card_w // 2, icon_y), 10, 2)

            # Cost badge
            cost_x = cx + card_w // 2
            cost_y = card_y + card_h - 16
            pg.draw.circle(self._screen, COL_ELIXIR_FILL, (cost_x, cost_y), 11)
            pg.draw.circle(self._screen, (150, 40, 150), (cost_x, cost_y), 11, 2)
            _draw_text(
                self._screen, str(card.cost), (cost_x, cost_y),
                self._font_md, COL_TEXT, shadow=False, center=True,
            )

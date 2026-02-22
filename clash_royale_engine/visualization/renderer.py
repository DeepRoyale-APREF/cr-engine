"""
Pygame renderer — Clash Royale GUI with sprite assets.

Renders the arena grid (green checkerboard, sky-blue river, brown bridges),
tower sprites, card-thumbnail troops, elixir bar, card hand and timer.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from clash_royale_engine.utils.constants import KING_TOWER_STATS, PRINCESS_TOWER_STATS

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
# Asset paths
# ══════════════════════════════════════════════════════════════════════════

_IMAGES_DIR = Path(__file__).parent / "images"
_CARDS_DIR = _IMAGES_DIR / "cards"
_SCREEN_DIR = _IMAGES_DIR / "screen"

# Card name → card image file
_CARD_IMAGE_FILES: Dict[str, Path] = {
    "archers": _CARDS_DIR / "ArchersCard.webp",
    "arrows": _CARDS_DIR / "ArrowsCard.webp",
    "fireball": _CARDS_DIR / "FireballCard.webp",
    "giant": _CARDS_DIR / "GiantCard.webp",
    "knight": _CARDS_DIR / "KnightCard.webp",
    "mini_pekka": _CARDS_DIR / "MiniPEKKACard.webp",
    "musketeer": _CARDS_DIR / "MusketeerCard.webp",
    "skeletons": _CARDS_DIR / "SkeletonsCard.webp",
}

# Tower sprite key → image file
_TOWER_IMAGE_FILES: Dict[str, Path] = {
    "ally_princess": _SCREEN_DIR / "Princess_Tower_Blue.webp",
    "enemy_princess": _SCREEN_DIR / "Princess_Tower_Red.webp",
    "ally_king": _SCREEN_DIR / "King_Tower_Blue.webp",
    "enemy_king": _SCREEN_DIR / "King_Tower_Red.webp",
}

# ── Soundtrack ────────────────────────────────────────────────────────────────
_SOUNDTRACK_DIR = Path(__file__).parent / "soundtrack"

# phase key → (filename, loop: -1=loop / 0=once)
_MUSIC_TRACKS: Dict[str, Tuple[str, int]] = {
    "battle":       ("Battle.mp3",      -1),
    "last60":       ("Last60.mp3",        0),
    "last30":       ("Last30.mp3",        0),
    "countdown":    ("Countdown.mp3",     0),
    "sudden_death": ("SuddenDeath.mp3",  -1),
    "countdown_ot": ("Countdown.mp3",     0),
}

_PRINCESS_TOWER_MAX_HP = float(PRINCESS_TOWER_STATS["hp"])
_KING_TOWER_MAX_HP = float(KING_TOWER_STATS["hp"])


# ══════════════════════════════════════════════════════════════════════════
# Colour palette
# ══════════════════════════════════════════════════════════════════════════

# Arena ground
COL_GRASS_A = (100, 150, 75)    # lighter green (even checker squares)
COL_GRASS_B = (85, 130, 60)     # darker green  (odd checker squares)
COL_RIVER = (65, 175, 225)      # sky blue river band
COL_BRIDGE = (130, 100, 58)     # dark-brown bridge planks
COL_BRIDGE_BORDER = (88, 65, 30)
COL_PATH = (165, 135, 90)       # lane path (dirt connecting bridges → king)

# UI chrome
COL_BG = (40, 32, 24)
COL_TEXT = (255, 255, 255)
COL_TEXT_SHADOW = (20, 20, 20)
COL_PANEL_BG = (32, 26, 18)
COL_ELIXIR_FILL = (200, 50, 200)
COL_ELIXIR_BG = (55, 18, 55)
COL_CARD_BG = (48, 48, 68)
COL_CARD_READY = (45, 105, 45)
COL_CARD_BORDER = (180, 155, 120)
COL_CARD_BORDER_READY = (100, 220, 100)
COL_HP_BAR_BG = (20, 20, 30)
COL_HP_GREEN = (70, 210, 70)
COL_HP_YELLOW = (220, 200, 45)
COL_HP_RED = (215, 45, 45)
COL_TOWER_HP_BAR = (55, 195, 240)   # sky-blue bar for tower HP
COL_TOWER_HP_TEXT = (255, 255, 255) # white HP numbers

# Per-player colours (also used as fallback when sprites unavailable)
COL_ALLY = (80, 160, 255)
COL_ENEMY = (255, 70, 70)
COL_ALLY_TOWER = (50, 110, 210)
COL_ENEMY_TOWER = (210, 50, 50)
COL_ALLY_KING = (28, 72, 185)
COL_ENEMY_KING = (180, 28, 28)

# Troop accent colours (fallback when card image unavailable)
TROOP_COLOURS: Dict[str, Tuple[int, int, int]] = {
    "giant": (200, 160, 80),
    "musketeer": (180, 100, 180),
    "archers": (100, 200, 100),
    "mini_pekka": (80, 80, 220),
    "knight": (180, 180, 80),
    "skeletons": (210, 210, 210),
    "princess_tower": (160, 160, 160),
    "king_tower": (220, 200, 60),
    "arrows": (180, 230, 80),
    "fireball": (240, 130, 30),
}

# ══════════════════════════════════════════════════════════════════════════
# Layout constants (render space — NOT game coordinates)
# ══════════════════════════════════════════════════════════════════════════

ARENA_TILES_W = 18
ARENA_TILES_H = 32
CELL_SIZE = 22          # pixels per tile

ARENA_W = ARENA_TILES_W * CELL_SIZE   # 396
ARENA_H = ARENA_TILES_H * CELL_SIZE   # 704

MARGIN_TOP = 52          # timer header
ARENA_PAD_Y = 62         # push grid down so enemy-king HP bars clear the HUD
MARGIN_BOTTOM = 162      # HUD (cards + elixir)
MARGIN_SIDE = 18

ARENA_VISUAL_H = ARENA_H + ARENA_PAD_Y

WIN_W = ARENA_W + 2 * MARGIN_SIDE   # 432
WIN_H = MARGIN_TOP + ARENA_VISUAL_H + MARGIN_BOTTOM

# Arena tile constants (match game constants)
RIVER_Y_TILE = 15        # tiles 15 and 16 are river
BRIDGE_LEFT_X = 3
BRIDGE_RIGHT_X = 13  # tiles 13-14; symmetric with left bridge (tiles 3-4)
# Lane path columns (wide enough to cover the bridge + 1 tile either side)
_LANE_L = {BRIDGE_LEFT_X - 1, BRIDGE_LEFT_X, BRIDGE_LEFT_X + 1, BRIDGE_LEFT_X + 2}  # {2,3,4,5}
_LANE_R = {BRIDGE_RIGHT_X - 1, BRIDGE_RIGHT_X, BRIDGE_RIGHT_X + 1, BRIDGE_RIGHT_X + 2}  # {12,13,14,15}
# Horizontal king-connector rows: 0-2 (player-0 side) and 29-31 (player-1 side)
_KING_ROW_BOT = {0, 1, 2}
_KING_ROW_TOP = {29, 30, 31}
_KING_CONNECTOR_X_MIN = BRIDGE_LEFT_X - 1   # x start of horizontal band (= 2)
_KING_CONNECTOR_X_MAX = BRIDGE_RIGHT_X + 2  # x end of horizontal band  (= 15)


def _is_path(tx: int, tile_y: int) -> bool:
    """Return True if this tile should be rendered as dirt path."""
    # Vertical lane strips (bridge columns, all non-river rows inside each half)
    if tx in _LANE_L or tx in _LANE_R:
        # Bottom half (player-0): rows 0-14; top half (player-1): rows 17-31
        if 0 <= tile_y <= 14 or 17 <= tile_y <= 31:
            return True
    # Horizontal connector near king towers (spans between the two lanes)
    if _KING_CONNECTOR_X_MIN <= tx <= _KING_CONNECTOR_X_MAX:
        if tile_y in _KING_ROW_BOT or tile_y in _KING_ROW_TOP:
            return True
    return False

# Tower sprite pixel dimensions
TOWER_PRINCESS_W = 52
TOWER_PRINCESS_H = 60
TOWER_KING_W = 72
TOWER_KING_H = 82

# Card thumbnail size shown in the arena for troops
TROOP_THUMB_W = 28
TROOP_THUMB_H = 34

# Tower tile positions + sprite key + is_ally
# Must match TOWER_POSITIONS in constants.py
_TOWER_TILES: List[Tuple[float, float, str, bool]] = [
    (3.0, 3.0, "ally_princess", True),
    (14.0, 3.0, "ally_princess", True),
    (8.5, 0.5, "ally_king", True),
    (3.0, 28.0, "enemy_princess", False),
    (14.0, 28.0, "enemy_princess", False),
    (8.5, 31.0, "enemy_king", False),
]

# ══════════════════════════════════════════════════════════════════════════
# Helper drawing functions
# ══════════════════════════════════════════════════════════════════════════


def _tile_to_screen(tile_x: float, tile_y: float) -> Tuple[int, int]:
    """Convert tile coordinates → screen pixel (centre of tile cell).

    tile_y=0 is the BOTTOM of the arena (player-0 king tower).
    Screen y=0 is top, so we flip vertically.
    ``ARENA_PAD_Y`` pushes the tile grid down so HP bars are not clipped.
    """
    sx = MARGIN_SIDE + int(tile_x * CELL_SIZE) + CELL_SIZE // 2
    sy = MARGIN_TOP + ARENA_PAD_Y + ARENA_H - int(tile_y * CELL_SIZE) - CELL_SIZE // 2
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
        sh = font.render(text, True, COL_TEXT_SHADOW)
        r = sh.get_rect()
        if center:
            r.center = (pos[0] + 1, pos[1] + 1)
        else:
            r.topleft = (pos[0] + 1, pos[1] + 1)
        surface.blit(sh, r)
    ts = font.render(text, True, colour)
    r = ts.get_rect()
    if center:
        r.center = pos
    else:
        r.topleft = pos
    surface.blit(ts, r)


def _load_image(path: Path, size: Tuple[int, int]) -> Optional[Any]:
    """Load and scale an image; return ``None`` on any failure."""
    pg = _ensure_pygame()
    try:
        img = pg.image.load(str(path)).convert_alpha()
        return pg.transform.smoothscale(img, size)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════
# Main Renderer class
# ══════════════════════════════════════════════════════════════════════════


class Renderer:
    """Pygame-based Clash Royale GUI with sprite assets.

    Renders the arena (green checkerboard grid, sky-blue river, brown
    bridges), tower sprites, card-thumbnail troops, elixir bar, card hand
    and a timer.

    Call :meth:`render` each frame with the current :class:`State`.

    Parameters
    ----------
    fps : int
        Target rendering framerate.
    title : str
        Window title.
    speed_multiplier : float
        Engine speed multiplier (used to sync music to simulated time).
    game_duration : float
        Regulation game duration in seconds (default 180).
    """

    def __init__(
        self,
        fps: int = 30,
        title: str = "Clash Royale Engine",
        speed_multiplier: float = 1.0,
        game_duration: float = 180.0,
    ) -> None:
        self.fps = fps
        self.title = title
        self._speed_multiplier = speed_multiplier
        self._game_duration = game_duration

        self._screen: Any = None
        self._clock: Any = None
        self._font_sm: Any = None
        self._font_md: Any = None
        self._font_lg: Any = None
        self._font_xl: Any = None
        self._initialised: bool = False

        # key: sprite_key → scaled Surface (or None if load failed)
        self._tower_sprites: Dict[str, Optional[Any]] = {}
        # key: (card_name, w, h) → scaled Surface (or None)
        self._card_cache: Dict[Tuple[str, int, int], Optional[Any]] = {}

        # Soundtrack state
        self._music_phase: str = ""      # currently loaded phase
        self._music_enabled: bool = False  # True after successful mixer init
        # Wall-clock ms when current phase was loaded, and the game-time at
        # that moment — used to detect drift when speed_multiplier != 1.0.
        self._phase_wall_ms: int = 0
        self._phase_game_time_s: float = 0.0

    # ── lifecycle ──────────────────────────────────────────────────────────

    def _init_pygame(self) -> None:
        if self._initialised:
            return
        pg = _ensure_pygame()
        pg.init()
        pg.display.set_caption(self.title)
        self._screen = pg.display.set_mode((WIN_W, WIN_H))
        self._clock = pg.time.Clock()
        self._font_sm = pg.font.SysFont("consolas", 11)
        self._font_md = pg.font.SysFont("consolas", 14, bold=True)
        self._font_lg = pg.font.SysFont("consolas", 19, bold=True)
        self._font_xl = pg.font.SysFont("consolas", 27, bold=True)

        # Pre-load and scale all tower sprites
        for key, path in _TOWER_IMAGE_FILES.items():
            size = (
                (TOWER_KING_W, TOWER_KING_H)
                if "king" in key
                else (TOWER_PRINCESS_W, TOWER_PRINCESS_H)
            )
            self._tower_sprites[key] = _load_image(path, size)

        self._init_music(pg)
        self._initialised = True

    def _init_music(self, pg: Any) -> None:
        """Initialise pygame.mixer and start Battle music."""
        try:
            pg.mixer.init()
            self._music_enabled = True
            self._play_phase("battle")
        except Exception:
            self._music_enabled = False

    def _play_phase(self, phase: str, game_time_s: float = 0.0) -> None:
        """Load and start the track for *phase* if it differs from current.

        Parameters
        ----------
        game_time_s:
            Simulated seconds elapsed since the start of the game.  Used to
            seek within the battle track when speed_multiplier > 1.
        """
        if not self._music_enabled or phase == self._music_phase:
            return
        entry = _MUSIC_TRACKS.get(phase)
        if entry is None:
            return
        track, loops = entry
        path = _SOUNDTRACK_DIR / track
        if not path.exists():
            return
        pg = _ensure_pygame()
        try:
            pg.mixer.music.load(str(path))
            # For the looping battle track, seek to the correct position so
            # music matches simulated time even when speed_multiplier != 1.
            seek_s = game_time_s if phase == "battle" else 0.0
            pg.mixer.music.play(loops, start=seek_s)
            self._music_phase = phase
            self._phase_wall_ms = pg.time.get_ticks()
            self._phase_game_time_s = game_time_s
        except Exception:
            pass

    def _update_music(self, state: "State") -> None:
        """Switch music track according to the current game phase.

        Also corrects drift in the battle track when speed_multiplier != 1
        so that the music position matches the simulated game time.
        """
        if not self._music_enabled:
            return
        n = state.numbers

        # Simulated time elapsed since game start
        if n.is_overtime:
            # game_duration + seconds elapsed since overtime began
            game_time_s = self._game_duration + max(0.0, 60.0 - n.overtime_remaining)
            phase = "countdown_ot" if n.overtime_remaining <= 10 else "sudden_death"
        else:
            game_time_s = self._game_duration - n.time_remaining
            if n.time_remaining <= 10:
                phase = "countdown"
            elif n.time_remaining <= 30:
                phase = "last30"
            elif n.time_remaining <= 60:
                phase = "last60"
            else:
                phase = "battle"

        self._play_phase(phase, game_time_s)

        # Correct drift in the looping battle track when speed_multiplier != 1.
        # Compare expected track position (from game time) vs real elapsed time.
        if self._music_phase == "battle" and abs(self._speed_multiplier - 1.0) > 0.01:
            pg = _ensure_pygame()
            real_elapsed_s = (pg.time.get_ticks() - self._phase_wall_ms) / 1000.0
            expected_s = game_time_s - self._phase_game_time_s
            drift_s = expected_s - real_elapsed_s
            if abs(drift_s) > 0.5:  # correct if drift exceeds 500 ms
                try:
                    pg.mixer.music.set_pos(expected_s)
                    # Reset reference so we don't over-correct next frame
                    self._phase_wall_ms = pg.time.get_ticks() - int(expected_s * 1000)
                except Exception:
                    pass

    def _get_card_img(self, name: str, w: int, h: int) -> Optional[Any]:
        """Return a cached and scaled card image Surface (or None)."""
        key = (name, w, h)
        if key not in self._card_cache:
            path = _CARD_IMAGE_FILES.get(name)
            self._card_cache[key] = _load_image(path, (w, h)) if path is not None else None
        return self._card_cache[key]

    def close(self) -> None:
        """Shut down the pygame window and stop music."""
        if self._initialised:
            pg = _ensure_pygame()
            if self._music_enabled:
                try:
                    pg.mixer.music.stop()
                    pg.mixer.quit()
                except Exception:
                    pass
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

        # Switch music track based on game phase
        self._update_music(state)

        self._screen.fill(COL_BG)

        # HUD panels drawn FIRST so HP bars/sprites render on top
        self._draw_top_hud(state)
        self._draw_bottom_hud(state)

        # Arena: checkerboard grass + sky-blue river + brown bridges
        self._draw_arena()

        # Tower sprites
        self._draw_tower_sprites(state)

        # Moving entities (troops only; buildings handled by tower sprites)
        self._draw_units(state.allies, is_ally=True)
        self._draw_units(state.enemies, is_ally=False)

        # Active spell visuals (drawn over units, under HP bars)
        self._draw_spells(state)

        # Tower HP bars drawn LAST — always visible over everything
        self._draw_tower_hp(state)

        pg.display.flip()
        self._clock.tick(self.fps)
        return True

    # ── arena grid ────────────────────────────────────────────────────────

    def _draw_arena(self) -> None:
        pg = _ensure_pygame()
        grid_top = MARGIN_TOP + ARENA_PAD_Y

        for tx in range(ARENA_TILES_W):
            for ty_screen in range(ARENA_TILES_H):
                # ty_screen=0 → top row on screen ≡ tile_y=31 in game coords
                tile_y = (ARENA_TILES_H - 1) - ty_screen
                sx = MARGIN_SIDE + tx * CELL_SIZE
                sy = grid_top + ty_screen * CELL_SIZE

                is_river = tile_y in (RIVER_Y_TILE, RIVER_Y_TILE + 1)
                is_bridge = (
                    BRIDGE_LEFT_X <= tx <= BRIDGE_LEFT_X + 1
                    or BRIDGE_RIGHT_X <= tx <= BRIDGE_RIGHT_X + 1
                )

                if is_river and not is_bridge:
                    colour = COL_RIVER
                elif is_river and is_bridge:
                    colour = COL_BRIDGE
                elif _is_path(tx, tile_y):
                    colour = COL_PATH
                else:
                    colour = COL_GRASS_A if (tx + tile_y) % 2 == 0 else COL_GRASS_B

                pg.draw.rect(self._screen, colour, (sx, sy, CELL_SIZE, CELL_SIZE))

        # Subtle 1-px grid lines over the coloured cells
        for tx in range(ARENA_TILES_W + 1):
            x = MARGIN_SIDE + tx * CELL_SIZE
            pg.draw.line(self._screen, (0, 0, 0), (x, grid_top), (x, grid_top + ARENA_H), 1)
        for ty in range(ARENA_TILES_H + 1):
            y = grid_top + ty * CELL_SIZE
            pg.draw.line(self._screen, (0, 0, 0), (MARGIN_SIDE, y), (MARGIN_SIDE + ARENA_W, y), 1)

        # Bridge border highlights
        river_sy = grid_top + (ARENA_TILES_H - RIVER_Y_TILE - 2) * CELL_SIZE
        for bx in (BRIDGE_LEFT_X, BRIDGE_RIGHT_X):
            bsx = MARGIN_SIDE + bx * CELL_SIZE
            pg.draw.rect(
                self._screen,
                COL_BRIDGE_BORDER,
                pg.Rect(bsx, river_sy, CELL_SIZE * 2, CELL_SIZE * 2),
                2,
            )

    # ── tower sprites ──────────────────────────────────────────────────────────────

    def _draw_tower_sprites(self, state: "State") -> None:
        """Blit tower images (or fallback rectangles) at their tile positions."""
        pg = _ensure_pygame()

        hp_map: Dict[Tuple[float, float], float] = {
            (3.0, 3.0): state.numbers.left_princess_hp,
            (14.0, 3.0): state.numbers.right_princess_hp,
            (8.5, 0.5): state.numbers.king_hp,
            (3.0, 28.0): state.numbers.left_enemy_princess_hp,
            (14.0, 28.0): state.numbers.right_enemy_princess_hp,
            (8.5, 31.0): state.numbers.enemy_king_hp,
        }

        for tx, ty, sprite_key, is_ally in _TOWER_TILES:
            if hp_map.get((tx, ty), 0.0) <= 0:
                continue  # destroyed — skip

            sx, sy = _tile_to_screen(tx, ty)
            sprite = self._tower_sprites.get(sprite_key)

            if sprite is not None:
                rect = sprite.get_rect(center=(sx, sy))
                self._screen.blit(sprite, rect)
            else:
                # Fallback coloured rectangle
                is_king = "king" in sprite_key
                sw = TOWER_KING_W if is_king else TOWER_PRINCESS_W
                sh = TOWER_KING_H if is_king else TOWER_PRINCESS_H
                fill = (
                    (COL_ALLY_KING if is_ally else COL_ENEMY_KING)
                    if is_king
                    else (COL_ALLY_TOWER if is_ally else COL_ENEMY_TOWER)
                )
                rect = pg.Rect(sx - sw // 2, sy - sh // 2, sw, sh)
                pg.draw.rect(self._screen, fill, rect, border_radius=5)
                pg.draw.rect(self._screen, COL_TEXT, rect, 2, border_radius=5)
                label = "♛" if is_king else "⚑"
                _draw_text(
                    self._screen, label, (sx, sy), self._font_md, COL_TEXT, center=True
                )

    # ── troops ─────────────────────────────────────────────────────────────────

    def _draw_units(self, units: "List[UnitDetection]", *, is_ally: bool) -> None:
        pg = _ensure_pygame()
        base_col = COL_ALLY if is_ally else COL_ENEMY

        for det in units:
            if det.unit.category == "building":
                continue  # towers rendered by _draw_tower_sprites

            tx = det.position.tile_x
            ty = det.position.tile_y
            sx, sy = _tile_to_screen(tx, ty)
            name = det.unit.name

            thumb = self._get_card_img(name, TROOP_THUMB_W, TROOP_THUMB_H)
            if thumb is not None:
                rect = thumb.get_rect(center=(sx, sy))
                border_rect = rect.inflate(4, 4)
                pg.draw.rect(self._screen, base_col, border_rect, border_radius=3)
                self._screen.blit(thumb, rect)
            else:
                accent = TROOP_COLOURS.get(name, base_col)
                radius = max(7, int(CELL_SIZE * 0.45))
                pg.draw.circle(self._screen, accent, (sx, sy), radius)
                pg.draw.circle(self._screen, base_col, (sx, sy), radius, 2)
                initial = name[0].upper() if name != "mini_pekka" else "P"
                _draw_text(
                    self._screen, initial, (sx, sy),
                    self._font_sm, COL_TEXT, shadow=False, center=True,
                )

            # HP bar above the unit
            if det.max_hp > 0:
                bar_w, bar_h = TROOP_THUMB_W + 4, 3
                bx_bar = sx - bar_w // 2
                by_bar = sy - (TROOP_THUMB_H // 2) - bar_h - 3
                pg.draw.rect(self._screen, COL_HP_BAR_BG, (bx_bar, by_bar, bar_w, bar_h))
                ratio = max(0.0, min(1.0, det.hp / det.max_hp))
                fill_w = int(bar_w * ratio)
                hp_col = (
                    COL_HP_GREEN
                    if ratio > 0.5
                    else (COL_HP_YELLOW if ratio > 0.25 else COL_HP_RED)
                )
                if fill_w > 0:
                    pg.draw.rect(self._screen, hp_col, (bx_bar, by_bar, fill_w, bar_h))
                pg.draw.rect(
                    self._screen, (160, 160, 160), (bx_bar, by_bar, bar_w, bar_h), 1
                )

    # ── spell effects ─────────────────────────────────────────────────────────────

    def _draw_spells(self, state: "State") -> None:
        """Draw fading area-of-effect circle for each active spell."""
        pg = _ensure_pygame()
        _MAX_FRAMES = {"arrows": 20, "fireball": 25}
        _COLORS: Dict[str, Tuple[int, int, int]] = {
            "arrows": (180, 230, 80),   # green-yellow
            "fireball": (240, 130, 30),  # orange
        }
        for spell in state.active_spells:
            col = _COLORS.get(spell.name, (255, 255, 255))
            max_frames = _MAX_FRAMES.get(spell.name, 20)
            alpha_fill = int(80 * spell.remaining_frames / max_frames)
            alpha_ring = int(220 * spell.remaining_frames / max_frames)
            radius_px = max(4, int(spell.radius * CELL_SIZE))

            sx, sy = _tile_to_screen(spell.tile_x, spell.tile_y)

            # Filled translucent circle
            surf = pg.Surface((radius_px * 2, radius_px * 2), pg.SRCALPHA)
            pg.draw.circle(surf, (*col, alpha_fill), (radius_px, radius_px), radius_px)
            # Bright ring
            pg.draw.circle(surf, (*col, alpha_ring), (radius_px, radius_px), radius_px, 3)
            self._screen.blit(surf, (sx - radius_px, sy - radius_px))

            # Small label above the ring
            _draw_text(
                self._screen,
                spell.name.upper(),
                (sx, sy - radius_px - 10),
                self._font_sm,
                col,
                shadow=True,
                center=True,
            )

    # ── tower HP bars ─────────────────────────────────────────────────────────────

    def _draw_tower_hp(self, state: "State") -> None:
        """Draw HP bars and king-active labels above each surviving tower."""
        pg = _ensure_pygame()
        bar_w, bar_h = 46, 5

        tower_info: List[Tuple[float, float, float, float, Tuple[int, int, int]]] = [
            (3.0, 3.0, state.numbers.left_princess_hp, _PRINCESS_TOWER_MAX_HP, COL_ALLY),
            (14.0, 3.0, state.numbers.right_princess_hp, _PRINCESS_TOWER_MAX_HP, COL_ALLY),
            (8.5, 0.5, state.numbers.king_hp, _KING_TOWER_MAX_HP, COL_ALLY),
            (3.0, 28.0, state.numbers.left_enemy_princess_hp, _PRINCESS_TOWER_MAX_HP, COL_ENEMY),
            (14.0, 28.0, state.numbers.right_enemy_princess_hp, _PRINCESS_TOWER_MAX_HP, COL_ENEMY),
            (8.5, 31.0, state.numbers.enemy_king_hp, _KING_TOWER_MAX_HP, COL_ENEMY),
        ]

        for tx, ty, hp, max_hp, col in tower_info:
            if hp <= 0:
                continue
            sx, sy = _tile_to_screen(tx, ty)
            is_king = ty in (0.5, 31.0)
            half_h = (TOWER_KING_H if is_king else TOWER_PRINCESS_H) // 2
            by = sy - half_h - bar_h - 6
            bx = sx - bar_w // 2

            # Dark background + sky-blue fill bar
            pg.draw.rect(self._screen, COL_HP_BAR_BG, (bx - 1, by - 1, bar_w + 2, bar_h + 2))
            ratio = max(0.0, min(1.0, hp / max_hp)) if max_hp > 0 else 0.0
            fill_w = int(bar_w * ratio)
            if fill_w > 0:
                pg.draw.rect(self._screen, COL_TOWER_HP_BAR, (bx, by, fill_w, bar_h))
            pg.draw.rect(self._screen, (30, 150, 200), (bx, by, bar_w, bar_h), 1)

            # HP number in white with strong shadow for contrast
            hp_str = str(int(hp))
            # shadow
            _draw_text(
                self._screen, hp_str, (sx + 1, by - 10),
                self._font_sm, (0, 0, 0), shadow=False, center=True,
            )
            _draw_text(
                self._screen, hp_str, (sx - 1, by - 10),
                self._font_sm, (0, 0, 0), shadow=False, center=True,
            )
            _draw_text(
                self._screen, hp_str, (sx, by - 9),
                self._font_sm, COL_TOWER_HP_TEXT, shadow=False, center=True,
            )

        # King-active indicator
        for tx, ty, is_active, col in [
            (8.5, 0.5, state.numbers.king_active, COL_ALLY),
            (8.5, 31.0, state.numbers.enemy_king_active, COL_ENEMY),
        ]:
            sx, sy = _tile_to_screen(tx, ty)
            label = "ACTIVE" if is_active else "zzz"
            label_col = (255, 80, 80) if is_active else (130, 130, 130)
            _draw_text(
                self._screen, label,
                (sx, sy + TOWER_KING_H // 2 + 3),
                self._font_sm, label_col, shadow=True, center=True,
            )

    # ── top HUD (timer, crowns) ────────────────────────────────────────────────

    def _draw_top_hud(self, state: "State") -> None:
        pg = _ensure_pygame()
        pg.draw.rect(self._screen, COL_PANEL_BG, pg.Rect(0, 0, WIN_W, MARGIN_TOP))
        pg.draw.line(
            self._screen, (80, 65, 45), (0, MARGIN_TOP - 1), (WIN_W, MARGIN_TOP - 1), 2
        )

        if state.numbers.is_overtime:
            t_display = state.numbers.overtime_remaining
            timer_col = (255, 120, 30)
        else:
            t_display = max(0.0, state.numbers.time_remaining)
            timer_col = COL_TEXT
        minutes = int(t_display) // 60
        seconds = int(t_display) % 60
        _draw_text(
            self._screen, f"{minutes}:{seconds:02d}",
            (WIN_W // 2, MARGIN_TOP // 2), self._font_xl, timer_col, center=True,
        )
        _draw_text(
            self._screen, "Overtime" if state.numbers.is_overtime else "Time left",
            (WIN_W // 2, 5), self._font_sm, (255, 120, 30) if state.numbers.is_overtime else (170, 170, 170),
            shadow=False, center=True,
        )

        ally_crowns = sum(
            1 for hp in [state.numbers.left_enemy_princess_hp,
                         state.numbers.right_enemy_princess_hp] if hp <= 0
        )
        enemy_crowns = sum(
            1 for hp in [state.numbers.left_princess_hp,
                         state.numbers.right_princess_hp] if hp <= 0
        )

        _draw_text(
            self._screen, f"★ {ally_crowns}",
            (MARGIN_SIDE + 14, MARGIN_TOP // 2), self._font_lg, COL_ALLY, center=True,
        )
        _draw_text(
            self._screen, f"★ {enemy_crowns}",
            (WIN_W - MARGIN_SIDE - 14, MARGIN_TOP // 2), self._font_lg, COL_ENEMY, center=True,
        )

        if state.numbers.is_overtime:
            # Pulsing red/orange overtime banner — flickers every 400 ms
            pg = _ensure_pygame()
            pulse = (pg.time.get_ticks() // 400) % 2
            ot_col = (255, 80, 30) if pulse == 0 else (255, 180, 30)
            _draw_text(
                self._screen, "OVERTIME  ×2",
                (WIN_W // 2, MARGIN_TOP - 12), self._font_sm, ot_col, shadow=True, center=True,
            )
        elif state.numbers.is_double_elixir:
            _draw_text(
                self._screen, "×2 ELIXIR",
                (WIN_W // 2, MARGIN_TOP - 12), self._font_sm, COL_ELIXIR_FILL, shadow=True, center=True,
            )

    # ── bottom HUD (elixir + cards) ────────────────────────────────────────────

    def _draw_bottom_hud(self, state: "State") -> None:
        pg = _ensure_pygame()
        hud_y = MARGIN_TOP + ARENA_VISUAL_H

        # Panel background
        pg.draw.rect(self._screen, COL_PANEL_BG, pg.Rect(0, hud_y, WIN_W, MARGIN_BOTTOM))
        pg.draw.line(self._screen, (80, 65, 45), (0, hud_y), (WIN_W, hud_y), 2)

        # ── Elixir bar ─────────────────────────────────────────────────────
        elixir = state.numbers.elixir
        bar_x = MARGIN_SIDE + 36
        bar_y = hud_y + 10
        bar_w = WIN_W - 2 * MARGIN_SIDE - 52
        bar_h = 20

        pg.draw.rect(self._screen, COL_ELIXIR_BG, (bar_x, bar_y, bar_w, bar_h), border_radius=5)
        fill_ratio = max(0.0, min(1.0, elixir / 10.0))
        fill_w = int(bar_w * fill_ratio)
        if fill_w > 0:
            pg.draw.rect(
                self._screen, COL_ELIXIR_FILL,
                (bar_x, bar_y, fill_w, bar_h), border_radius=5,
            )
        pg.draw.rect(
            self._screen, (130, 55, 130), (bar_x, bar_y, bar_w, bar_h), 2, border_radius=5
        )
        for i in range(1, 10):
            tx_tick = bar_x + int(bar_w * i / 10.0)
            pg.draw.line(
                self._screen, (110, 45, 110),
                (tx_tick, bar_y + 3), (tx_tick, bar_y + bar_h - 3), 1,
            )
        _draw_text(
            self._screen, str(int(elixir)),
            (MARGIN_SIDE + 14, bar_y + bar_h // 2),
            self._font_lg, COL_ELIXIR_FILL, center=True,
        )

        # ── Card hand ──────────────────────────────────────────────────
        card_top = bar_y + bar_h + 8
        n_cards = len(state.cards)
        card_gap = 6
        card_w = (WIN_W - 2 * MARGIN_SIDE - card_gap * (n_cards - 1)) // n_cards
        card_h = hud_y + MARGIN_BOTTOM - card_top - 6
        badge_r = 12
        ready_set = set(state.ready)

        for i, card in enumerate(state.cards):
            cx = MARGIN_SIDE + i * (card_w + card_gap)
            is_ready = i in ready_set

            # Frame
            border_col = COL_CARD_BORDER_READY if is_ready else COL_CARD_BORDER
            pg.draw.rect(
                self._screen,
                COL_CARD_READY if is_ready else COL_CARD_BG,
                pg.Rect(cx, card_top, card_w, card_h),
                border_radius=7,
            )
            pg.draw.rect(
                self._screen, border_col,
                pg.Rect(cx, card_top, card_w, card_h), 2, border_radius=7,
            )

            if card.name == "empty":
                continue

            # Card art (fills card slot, leaving bottom badge area)
            art_margin = 3
            art_x = cx + art_margin
            art_y = card_top + art_margin
            art_w = card_w - art_margin * 2
            art_h = card_h - badge_r * 2 - art_margin

            art = self._get_card_img(card.name, art_w, art_h)
            if art is not None:
                self._screen.blit(art, (art_x, art_y))
            else:
                # Fallback: coloured icon + name
                icon_cx = cx + card_w // 2
                icon_cy = art_y + art_h // 2
                accent = TROOP_COLOURS.get(card.name, COL_TEXT)
                if card.is_spell:
                    for angle in range(0, 360, 45):
                        rad = math.radians(angle)
                        ex = int(icon_cx + 13 * math.cos(rad))
                        ey = int(icon_cy + 13 * math.sin(rad))
                        pg.draw.line(self._screen, accent, (icon_cx, icon_cy), (ex, ey), 2)
                else:
                    pg.draw.circle(self._screen, accent, (icon_cx, icon_cy), 13)
                    pg.draw.circle(self._screen, COL_TEXT, (icon_cx, icon_cy), 13, 2)
                disp = card.name.replace("_", " ").title()
                if len(disp) > 9:
                    disp = disp[:8] + "."
                _draw_text(
                    self._screen, disp, (icon_cx, icon_cy + 18),
                    self._font_sm, COL_TEXT, shadow=False, center=True,
                )

            # Elixir cost badge at bottom-centre
            cost_cx = cx + card_w // 2
            cost_cy = card_top + card_h - badge_r - 2
            pg.draw.circle(self._screen, COL_ELIXIR_FILL, (cost_cx, cost_cy), badge_r)
            pg.draw.circle(self._screen, (150, 35, 150), (cost_cx, cost_cy), badge_r, 2)
            _draw_text(
                self._screen, str(card.cost),
                (cost_cx, cost_cy), self._font_md, COL_TEXT, shadow=False, center=True,
            )

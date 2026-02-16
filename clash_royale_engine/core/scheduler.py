"""
Frame / time scheduler.

Keeps track of the current frame number, elapsed game time, and helpers
for converting between frames and seconds.
"""

from __future__ import annotations

from clash_royale_engine.utils.constants import DEFAULT_FPS, GAME_DURATION, OVERTIME_DURATION


class Scheduler:
    """Manages simulation clock and frame bookkeeping."""

    def __init__(self, fps: int = DEFAULT_FPS) -> None:
        self.fps: int = fps
        self.dt: float = 1.0 / fps
        self.current_frame: int = 0
        self.game_duration: float = GAME_DURATION
        self.overtime_duration: float = OVERTIME_DURATION

    # ── helpers ───────────────────────────────────────────────────────────
    @property
    def elapsed_seconds(self) -> float:
        """Seconds elapsed since start of the match."""
        return self.current_frame * self.dt

    @property
    def time_remaining(self) -> float:
        """Seconds remaining (clamped to ≥ 0)."""
        return max(0.0, self.game_duration - self.elapsed_seconds)

    @property
    def is_overtime(self) -> bool:
        return self.elapsed_seconds >= self.game_duration

    @property
    def is_time_up(self) -> bool:
        return self.elapsed_seconds >= self.game_duration + self.overtime_duration

    @property
    def is_double_elixir(self) -> bool:
        """Last 60 s of regulation → double elixir."""
        from clash_royale_engine.utils.constants import DOUBLE_ELIXIR_TIME

        return self.elapsed_seconds >= DOUBLE_ELIXIR_TIME

    def frames_for_seconds(self, seconds: float) -> int:
        """Convert a duration in seconds to whole frames."""
        return int(seconds * self.fps)

    def advance(self, n_frames: int = 1) -> None:
        """Advance the clock by *n_frames*."""
        self.current_frame += n_frames

    def reset(self) -> None:
        self.current_frame = 0

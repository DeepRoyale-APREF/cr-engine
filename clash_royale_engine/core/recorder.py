"""
Game recorder and imitation-learning episode extractor.

Records complete (god-view) game state every frame and provides
extraction of **4 IL episodes per game** via symmetry transforms:

1. **P0 original** — Player 0's trajectory, own side at bottom.
2. **P1 normalised** — Player 1's trajectory, y-flipped so own side
   appears at bottom (same reference frame as P0).
3. **P0 x-mirrored** — Horizontal mirror of episode 1 (data augmentation).
4. **P1 normalised + x-mirrored** — Both transforms applied to P1.

This exploits the arena's top-bottom *and* left-right symmetry to
quadruple the training data from a single match.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from clash_royale_engine.core.state import (
    Numbers,
    Position,
    SpellInfo,
    State,
    UnitDetection,
)
from clash_royale_engine.utils.constants import N_HEIGHT_TILES, N_WIDE_TILES

# ══════════════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════════════


@dataclass
class FrameRecord:
    """God-view snapshot of a single simulation frame."""

    frame: int
    time_remaining: float
    state_p0: State  # full state from Player 0 perspective
    state_p1: State  # full state from Player 1 perspective
    action_p0: Optional[Tuple[int, int, int]]  # (tile_x, tile_y, card_idx) or None
    action_p1: Optional[Tuple[int, int, int]]


@dataclass
class Transition:
    """One (s, a, r, s', done) tuple ready for IL / offline-RL.

    ``state`` and ``next_state`` are feature vectors (np.ndarray).
    ``action`` is the integer-encoded discrete action (Gym action space).
    """

    state: np.ndarray
    action: int  # Gym-encoded integer (or -1 for no-op)
    reward: float
    next_state: np.ndarray
    done: bool


@dataclass
class GameRecord:
    """Complete recording of one game (god-view)."""

    frames: List[FrameRecord] = field(default_factory=list)
    winner: Optional[int] = None  # 0, 1, None (draw)
    deck_p0: List[str] = field(default_factory=list)
    deck_p1: List[str] = field(default_factory=list)
    total_frames: int = 0

    # ── persistence ───────────────────────────────────────────────────

    def save(self, path: Union[str, Path]) -> None:
        """Serialise this record to *path* using pickle."""
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "GameRecord":
        """Load a previously saved :class:`GameRecord` from *path*."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected GameRecord, got {type(obj)}")
        return obj


# ══════════════════════════════════════════════════════════════════════════
# Recorder  (attach to engine, collects frames)
# ══════════════════════════════════════════════════════════════════════════


class GameRecorder:
    """Accumulates :class:`FrameRecord` objects during a game.

    Typical usage::

        recorder = GameRecorder()
        recorder.reset()
        # ... each frame inside engine ...
        recorder.record_action(player_id, action)
        recorder.record_frame(frame, time_remaining, state_p0, state_p1)
        # ... after game over ...
        record = recorder.build_record(winner, deck_p0, deck_p1)
    """

    def __init__(self) -> None:
        self._frames: List[FrameRecord] = []
        self._pending_actions: Dict[int, Optional[Tuple[int, int, int]]] = {
            0: None,
            1: None,
        }

    def reset(self) -> None:
        """Clear all recorded data for a new game."""
        self._frames.clear()
        self._pending_actions = {0: None, 1: None}

    def record_action(self, player_id: int, action: Optional[Tuple[int, int, int]]) -> None:
        """Register the action a player took this frame."""
        self._pending_actions[player_id] = action

    def record_frame(
        self,
        frame: int,
        time_remaining: float,
        state_p0: State,
        state_p1: State,
    ) -> None:
        """Snapshot the current frame (call *after* simulation step)."""
        self._frames.append(
            FrameRecord(
                frame=frame,
                time_remaining=time_remaining,
                state_p0=state_p0,
                state_p1=state_p1,
                action_p0=self._pending_actions[0],
                action_p1=self._pending_actions[1],
            )
        )
        # Clear pending actions for next frame
        self._pending_actions = {0: None, 1: None}

    def build_record(
        self,
        winner: Optional[int],
        deck_p0: List[str],
        deck_p1: List[str],
    ) -> GameRecord:
        """Finalise and return a :class:`GameRecord`."""
        return GameRecord(
            frames=list(self._frames),
            winner=winner,
            deck_p0=list(deck_p0),
            deck_p1=list(deck_p1),
            total_frames=len(self._frames),
        )


# ══════════════════════════════════════════════════════════════════════════
# State transforms  (symmetry operations)
# ══════════════════════════════════════════════════════════════════════════


def _flip_detection_y(det: UnitDetection) -> UnitDetection:
    """Flip a detection along the vertical axis (top ↔ bottom)."""
    new_ty = N_HEIGHT_TILES - 1 - det.position.tile_y
    return UnitDetection(
        unit=det.unit,
        position=Position(
            bbox=det.position.bbox,  # bbox is approximate; tile is canonical
            conf=det.position.conf,
            tile_x=det.position.tile_x,
            tile_y=new_ty,
        ),
        hp=det.hp,
        max_hp=det.max_hp,
    )


def _flip_detection_x(det: UnitDetection) -> UnitDetection:
    """Flip a detection along the horizontal axis (left ↔ right)."""
    new_tx = N_WIDE_TILES - 1 - det.position.tile_x
    return UnitDetection(
        unit=det.unit,
        position=Position(
            bbox=det.position.bbox,
            conf=det.position.conf,
            tile_x=new_tx,
            tile_y=det.position.tile_y,
        ),
        hp=det.hp,
        max_hp=det.max_hp,
    )


def _flip_spell_y(s: SpellInfo) -> SpellInfo:
    return SpellInfo(
        name=s.name,
        tile_x=s.tile_x,
        tile_y=N_HEIGHT_TILES - 1 - s.tile_y,
        radius=s.radius,
        remaining_frames=s.remaining_frames,
    )


def _flip_spell_x(s: SpellInfo) -> SpellInfo:
    return SpellInfo(
        name=s.name,
        tile_x=N_WIDE_TILES - 1 - s.tile_x,
        tile_y=s.tile_y,
        radius=s.radius,
        remaining_frames=s.remaining_frames,
    )


def flip_state_y(state: State) -> State:
    """Flip *state* vertically — normalises Player 1's view to bottom."""
    return State(
        allies=[_flip_detection_y(a) for a in state.allies],
        enemies=[_flip_detection_y(e) for e in state.enemies],
        numbers=state.numbers,  # tower HPs are already from player's perspective
        cards=state.cards,
        ready=list(state.ready),
        active_spells=[_flip_spell_y(s) for s in state.active_spells],
    )


def flip_state_x(state: State) -> State:
    """Flip *state* horizontally — left ↔ right mirror.

    Also swaps left/right princess-tower HPs so they remain consistent.
    """
    n = state.numbers
    new_numbers = Numbers(
        elixir=n.elixir,
        enemy_elixir=n.enemy_elixir,
        left_princess_hp=n.right_princess_hp,
        right_princess_hp=n.left_princess_hp,
        king_hp=n.king_hp,
        left_enemy_princess_hp=n.right_enemy_princess_hp,
        right_enemy_princess_hp=n.left_enemy_princess_hp,
        enemy_king_hp=n.enemy_king_hp,
        time_remaining=n.time_remaining,
        king_active=n.king_active,
        enemy_king_active=n.enemy_king_active,
        is_double_elixir=n.is_double_elixir,
        is_overtime=n.is_overtime,
        overtime_remaining=n.overtime_remaining,
    )
    return State(
        allies=[_flip_detection_x(a) for a in state.allies],
        enemies=[_flip_detection_x(e) for e in state.enemies],
        numbers=new_numbers,
        cards=state.cards,
        ready=list(state.ready),
        active_spells=[_flip_spell_x(s) for s in state.active_spells],
    )


def flip_action_y(
    action: Optional[Tuple[int, int, int]],
) -> Optional[Tuple[int, int, int]]:
    """Flip an action's tile_y."""
    if action is None:
        return None
    tx, ty, ci = action
    return (tx, N_HEIGHT_TILES - 1 - ty, ci)


def flip_action_x(
    action: Optional[Tuple[int, int, int]],
) -> Optional[Tuple[int, int, int]]:
    """Flip an action's tile_x."""
    if action is None:
        return None
    tx, ty, ci = action
    return (N_WIDE_TILES - 1 - tx, ty, ci)


def apply_fog_of_war(state: State) -> State:
    """Return a copy of *state* with hidden enemy info zeroed out.

    Currently hides:
    * ``enemy_elixir`` → 0
    """
    n = state.numbers
    fog_numbers = Numbers(
        elixir=n.elixir,
        enemy_elixir=0.0,  # hidden
        left_princess_hp=n.left_princess_hp,
        right_princess_hp=n.right_princess_hp,
        king_hp=n.king_hp,
        left_enemy_princess_hp=n.left_enemy_princess_hp,
        right_enemy_princess_hp=n.right_enemy_princess_hp,
        enemy_king_hp=n.enemy_king_hp,
        time_remaining=n.time_remaining,
        king_active=n.king_active,
        enemy_king_active=n.enemy_king_active,
        is_double_elixir=n.is_double_elixir,
        is_overtime=n.is_overtime,
        overtime_remaining=n.overtime_remaining,
    )
    return State(
        allies=state.allies,
        enemies=state.enemies,
        numbers=fog_numbers,
        cards=state.cards,
        ready=list(state.ready),
        active_spells=state.active_spells,
    )


# ══════════════════════════════════════════════════════════════════════════
# Action encoding helper
# ══════════════════════════════════════════════════════════════════════════

# Matches ClashRoyaleEnv action space: Discrete(18*32*4 + 1)
_N_PLACEMENT = N_WIDE_TILES * N_HEIGHT_TILES * 4
NOOP_ACTION: int = _N_PLACEMENT  # last index = no-op


def encode_action(action: Optional[Tuple[int, int, int]]) -> int:
    """Convert a ``(tile_x, tile_y, card_idx)`` tuple to a Gym integer action."""
    if action is None:
        return NOOP_ACTION
    tx, ty, ci = action
    return (tx * N_HEIGHT_TILES + ty) * 4 + ci


def decode_action(action_int: int) -> Optional[Tuple[int, int, int]]:
    """Inverse of :func:`encode_action`."""
    if action_int >= _N_PLACEMENT:
        return None
    ci = action_int % 4
    remaining = action_int // 4
    ty = remaining % N_HEIGHT_TILES
    tx = remaining // N_HEIGHT_TILES
    return (tx, ty, ci)


# ══════════════════════════════════════════════════════════════════════════
# Episode extractor — 4 IL episodes from 1 game
# ══════════════════════════════════════════════════════════════════════════


# Type alias for the encoder callable
StateEncoder = Callable[[State], np.ndarray]

# Type alias for optional custom reward function
# (state, action, next_state, player_id, winner, is_last) -> float
RewardFn = Callable[
    [State, Optional[Tuple[int, int, int]], State, int, Optional[int], bool],
    float,
]


def _default_sparse_reward(
    state: State,
    action: Optional[Tuple[int, int, int]],
    next_state: State,
    player_id: int,
    winner: Optional[int],
    is_last: bool,
) -> float:
    """Sparse reward: +1 win, -1 loss, 0 otherwise."""
    if is_last:
        if winner is None:
            return 0.0
        return 1.0 if winner == player_id else -1.0
    return 0.0


class EpisodeExtractor:
    """Extract 4 imitation-learning episodes from a :class:`GameRecord`.

    Transforms
    ----------
    1. **P0 original** — Player 0, coordinates as-is.
    2. **P1 y-flip** — Player 1, y-flipped so own side is at bottom.
    3. **P0 x-flip** — Player 0, left-right mirror.
    4. **P1 y+x-flip** — Player 1, both flips.

    All 4 episodes share the same reference frame: "active player's
    side at the bottom, left tower at x ≈ 3".  A single policy trained on
    these can play from **either** side without modification.

    Parameters
    ----------
    encoder : callable
        ``State → np.ndarray``  (e.g. ``ClashRoyaleEnv._to_feature_vector``).
    reward_fn : callable, optional
        Custom ``(state, action, next_state, pid, winner, is_last) → float``.
        Defaults to sparse ±1 at episode end.
    fog_of_war : bool
        If *True*, ``enemy_elixir`` is zeroed in the encoded states.
    """

    def __init__(
        self,
        encoder: StateEncoder,
        reward_fn: Optional[RewardFn] = None,
        fog_of_war: bool = True,
    ) -> None:
        self.encoder = encoder
        self.reward_fn: RewardFn = reward_fn or _default_sparse_reward
        self.fog_of_war = fog_of_war

    def extract(self, record: GameRecord) -> List[List[Transition]]:
        """Return **4 episode trajectories** from a single :class:`GameRecord`."""
        return [
            self._build(record, player_id=0, y_flip=False, x_flip=False),
            self._build(record, player_id=1, y_flip=True, x_flip=False),
            self._build(record, player_id=0, y_flip=False, x_flip=True),
            self._build(record, player_id=1, y_flip=True, x_flip=True),
        ]

    # ── internal ──────────────────────────────────────────────────────────

    def _build(
        self,
        record: GameRecord,
        player_id: int,
        y_flip: bool,
        x_flip: bool,
    ) -> List[Transition]:
        frames = record.frames
        if len(frames) < 2:
            return []

        transitions: List[Transition] = []
        last_idx = len(frames) - 2  # inclusive

        for i in range(len(frames) - 1):
            fr = frames[i]
            fr_next = frames[i + 1]

            # Select correct player perspective
            state = fr.state_p0 if player_id == 0 else fr.state_p1
            next_state = fr_next.state_p0 if player_id == 0 else fr_next.state_p1
            action = fr.action_p0 if player_id == 0 else fr.action_p1

            # Apply symmetry transforms
            if y_flip:
                state = flip_state_y(state)
                next_state = flip_state_y(next_state)
                action = flip_action_y(action)
            if x_flip:
                state = flip_state_x(state)
                next_state = flip_state_x(next_state)
                action = flip_action_x(action)

            # Fog of war (hide enemy elixir)
            if self.fog_of_war:
                state = apply_fog_of_war(state)
                next_state = apply_fog_of_war(next_state)

            is_last = i == last_idx
            reward = self.reward_fn(
                state,
                action,
                next_state,
                player_id,
                record.winner,
                is_last,
            )

            transitions.append(
                Transition(
                    state=self.encoder(state),
                    action=encode_action(action),
                    reward=reward,
                    next_state=self.encoder(next_state),
                    done=is_last,
                )
            )

        return transitions

    # ── convenience: batch to numpy ──────────────────────────────────────

    @staticmethod
    def episodes_to_numpy(
        episodes: List[List[Transition]],
    ) -> Dict[str, np.ndarray]:
        """Flatten all episodes into contiguous numpy arrays.

        Returns dict with keys:
        ``states, actions, rewards, next_states, dones, episode_ids``
        """
        states, actions, rewards, next_states, dones, ep_ids = [], [], [], [], [], []

        for ep_idx, episode in enumerate(episodes):
            for t in episode:
                states.append(t.state)
                actions.append(t.action)
                rewards.append(t.reward)
                next_states.append(t.next_state)
                dones.append(t.done)
                ep_ids.append(ep_idx)

        return {
            "states": np.array(states, dtype=np.float32),
            "actions": np.array(actions, dtype=np.int64),
            "rewards": np.array(rewards, dtype=np.float32),
            "next_states": np.array(next_states, dtype=np.float32),
            "dones": np.array(dones, dtype=bool),
            "episode_ids": np.array(ep_ids, dtype=np.int64),
        }

    @staticmethod
    def save_numpy(
        path: Union[str, Path],
        episodes: List[List[Transition]],
    ) -> None:
        """Flatten *episodes* and save to a compressed ``.npz`` file.

        Parameters
        ----------
        path:
            Destination file path (e.g. ``data/game_001.npz``).
        episodes:
            List of episode trajectories as returned by :meth:`extract`.
        """
        batch = EpisodeExtractor.episodes_to_numpy(episodes)
        np.savez_compressed(str(path), **batch)

    @staticmethod
    def load_numpy(path: Union[str, Path]) -> Dict[str, np.ndarray]:
        """Load a ``.npz`` batch previously saved by :meth:`save_numpy`.

        Returns
        -------
        dict
            Same keys as :meth:`episodes_to_numpy`:
            ``states, actions, rewards, next_states, dones, episode_ids``.
        """
        data = np.load(str(path))
        return {k: data[k] for k in data.files}

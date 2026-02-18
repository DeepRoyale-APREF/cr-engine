"""
Gymnasium-compatible single-agent environment for Clash Royale.

Player 0 is the RL agent; Player 1 is driven by a :class:`PlayerInterface`
(default: :class:`HeuristicBot`).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from clash_royale_engine.core.engine import ClashRoyaleEngine
from clash_royale_engine.core.recorder import (
    EpisodeExtractor,
    GameRecord,
    Transition,
    apply_fog_of_war,
)
from clash_royale_engine.core.state import State
from clash_royale_engine.players.player_interface import (
    HeuristicBot,
    PlayerInterface,
    RLAgentPlayer,
)
from clash_royale_engine.utils.constants import (
    CARD_VOCAB,
    DEFAULT_DECK,
    DEFAULT_FPS,
    GAME_DURATION,
    KING_TOWER_STATS,
    N_HEIGHT_TILES,
    N_WIDE_TILES,
    OBS_FEATURE_DIM,
    PRINCESS_TOWER_STATS,
)
from clash_royale_engine.utils.validators import InvalidActionError


class ObservationType(Enum):
    STATE_DICT = "state_dict"
    FEATURE_VECTOR = "feature_vector"
    IMAGE = "image"
    HYBRID = "hybrid"


class ClashRoyaleEnv(gym.Env):
    """
    Gymnasium environment for Clash Royale Arena 1.

    Action space
    -------------
    ``Discrete(18 * 32 * 4 + 1)`` — 2304 card-placement combos + 1 "do nothing".

    Observation space (FEATURE_VECTOR)
    -----------------------------------
    ``Box(0, 1, shape=(OBS_FEATURE_DIM,), float32)``
    """

    metadata: Dict[str, Any] = {
        "render_modes": ["human", "rgb_array", None],
        "render_fps": 30,
    }

    def __init__(
        self,
        opponent: Optional[PlayerInterface] = None,
        deck: Optional[List[str]] = None,
        opponent_deck: Optional[List[str]] = None,
        obs_type: ObservationType = ObservationType.FEATURE_VECTOR,
        reward_shaping: str = "sparse",
        time_limit: float = GAME_DURATION,
        fps: int = DEFAULT_FPS,
        speed_multiplier: float = 1.0,
        render_mode: Optional[str] = None,
        seed: int = 0,
        fog_of_war: bool = True,
        record: bool = False,
    ) -> None:
        super().__init__()

        self.obs_type = obs_type
        self.reward_shaping = reward_shaping
        self.render_mode = render_mode
        self.fog_of_war = fog_of_war
        self._record = record

        deck = deck or list(DEFAULT_DECK)
        opponent_deck = opponent_deck or list(DEFAULT_DECK)
        opponent = opponent or HeuristicBot()

        self.engine = ClashRoyaleEngine(
            player1=RLAgentPlayer(),  # placeholder — actions injected via step()
            player2=opponent,
            deck1=deck,
            deck2=opponent_deck,
            fps=fps,
            time_limit=time_limit,
            speed_multiplier=speed_multiplier,
            seed=seed,
        )

        # Enable recording if requested
        if self._record:
            self.engine.enable_recording()

        self._setup_spaces()

    # ── spaces ────────────────────────────────────────────────────────────

    def _setup_spaces(self) -> None:
        # 18*32*4 card placements + 1 "no-op"
        self.action_space = spaces.Discrete(N_WIDE_TILES * N_HEIGHT_TILES * 4 + 1)

        if self.obs_type == ObservationType.FEATURE_VECTOR:
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(OBS_FEATURE_DIM,),
                dtype=np.float32,
            )
        elif self.obs_type == ObservationType.IMAGE:
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(128, 128, 3),
                dtype=np.uint8,
            )
        else:
            # Fallback to feature vector shape
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(OBS_FEATURE_DIM,),
                dtype=np.float32,
            )

    # ── Gymnasium API ─────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.engine.seed = seed

        state = self.engine.reset()
        obs = self._encode(state)
        info: Dict[str, Any] = {"raw_state": state}
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        decoded = self._decode_action(action)
        action_valid = True

        try:
            state_p0, _, done = self.engine.step_with_action(player_id=0, action=decoded)
        except InvalidActionError:
            # Action was invalid — still advance one frame so time progresses
            state_p0, _, done = self.engine.step_with_action(player_id=0, action=None)
            action_valid = False

        reward = self._calculate_reward(state_p0, action_valid)
        obs = self._encode(state_p0)

        terminated = done and self.engine.has_winner()
        truncated = done and not self.engine.has_winner()

        info: Dict[str, Any] = {
            "raw_state": state_p0,
            "action_valid": action_valid,
            "elixir": state_p0.numbers.elixir,
            "towers_destroyed": self.engine.count_towers_destroyed(0),
        }

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        return None  # Phase 2

    def close(self) -> None:
        pass

    # ── action coding ─────────────────────────────────────────────────────

    @staticmethod
    def _decode_action(action: int) -> Optional[Tuple[int, int, int]]:
        """
        Decode an integer action.

        The last action index is the no-op.
        """
        n_placement = N_WIDE_TILES * N_HEIGHT_TILES * 4
        if action >= n_placement:
            return None  # no-op

        card_idx = action % 4
        remaining = action // 4
        tile_y = remaining % N_HEIGHT_TILES
        tile_x = remaining // N_HEIGHT_TILES
        return (tile_x, tile_y, card_idx)

    # ── observation encoding (with fog-of-war) ───────────────────────────────

    def _encode(self, state: State) -> np.ndarray:
        if self.fog_of_war:
            state = apply_fog_of_war(state)
        if self.obs_type == ObservationType.FEATURE_VECTOR:
            return self._to_feature_vector(state)
        # Default fallback
        return self._to_feature_vector(state)

    @staticmethod
    def _to_feature_vector(state: State) -> np.ndarray:
        features: List[float] = []

        # Elixir (2)
        features.append(state.numbers.elixir / 10.0)
        features.append(state.numbers.enemy_elixir / 10.0)

        # Tower HP (6)
        max_p = float(PRINCESS_TOWER_STATS["hp"])
        max_k = float(KING_TOWER_STATS["hp"])
        features.extend(
            [
                state.numbers.left_princess_hp / max_p,
                state.numbers.right_princess_hp / max_p,
                state.numbers.king_hp / max_k,
                state.numbers.left_enemy_princess_hp / max_p,
                state.numbers.right_enemy_princess_hp / max_p,
                state.numbers.enemy_king_hp / max_k,
            ]
        )

        # Cards in hand: one-hot + cost  (4 × (8 + 1) = 36)
        for card in state.cards:
            encoding = [1.0 if card.name == c else 0.0 for c in CARD_VOCAB]
            features.extend(encoding)
            features.append(card.cost / 10.0)

        # Ally grid (32 × 18 = 576)
        ally_grid = np.zeros((N_HEIGHT_TILES, N_WIDE_TILES), dtype=np.float32)
        for a in state.allies:
            tx, ty = a.position.tile_x, a.position.tile_y
            if 0 <= tx < N_WIDE_TILES and 0 <= ty < N_HEIGHT_TILES:
                ally_grid[ty, tx] = 1.0

        # Enemy grid (576)
        enemy_grid = np.zeros((N_HEIGHT_TILES, N_WIDE_TILES), dtype=np.float32)
        for e in state.enemies:
            tx, ty = e.position.tile_x, e.position.tile_y
            if 0 <= tx < N_WIDE_TILES and 0 <= ty < N_HEIGHT_TILES:
                enemy_grid[ty, tx] = 1.0

        features.extend(ally_grid.flatten().tolist())
        features.extend(enemy_grid.flatten().tolist())

        arr = np.array(features, dtype=np.float32)
        # Pad or truncate to OBS_FEATURE_DIM
        if arr.shape[0] < OBS_FEATURE_DIM:
            arr = np.pad(arr, (0, OBS_FEATURE_DIM - arr.shape[0]))
        elif arr.shape[0] > OBS_FEATURE_DIM:
            arr = arr[:OBS_FEATURE_DIM]
        return np.clip(arr, 0.0, 1.0)

    # ── reward ────────────────────────────────────────────────────────────

    def _calculate_reward(self, state: State, action_valid: bool) -> float:
        if self.reward_shaping == "sparse":
            if self.engine.is_done():
                winner = self.engine.get_winner()
                if winner == 0:
                    return 1.0
                elif winner == 1:
                    return -1.0
                return 0.0
            return 0.0

        # Dense reward
        reward = 0.0
        if not action_valid:
            reward -= 0.1

        # Damage dealt to enemy towers
        dmg = self.engine.get_tower_damage_delta(0)
        max_p = float(PRINCESS_TOWER_STATS["hp"])
        reward += dmg / max_p * 0.5

        # Damage taken on own towers
        loss = self.engine.get_tower_loss_delta(0)
        reward -= loss / max_p * 0.5

        if action_valid and state.numbers.elixir < 9.0:
            reward += 0.01

        if self.engine.is_done():
            winner = self.engine.get_winner()
            if winner == 0:
                reward += 10.0
            elif winner == 1:
                reward -= 10.0

        return reward

    # ── recording & IL extraction ──────────────────────────────────────

    def get_game_record(self) -> Optional[GameRecord]:
        """Return the :class:`GameRecord` from the most recently finished episode.

        Only available when ``record=True`` was passed at construction.
        """
        return self.engine.get_last_record()

    def extract_il_episodes(
        self,
        record: Optional[GameRecord] = None,
    ) -> List[List[Transition]]:
        """Extract **4 IL episodes** from the last game (or a provided record).

        Uses symmetry transforms (y-flip for P1 normalisation, x-flip for
        horizontal mirror) to generate 4 training trajectories from 1 game.

        Returns a list of 4 episode trajectories, each being a list of
        :class:`Transition` objects with numpy feature-vector states.
        """
        rec = record or self.get_game_record()
        if rec is None:
            raise ValueError(
                "No game record available. Pass record=True to the env "
                "or provide a GameRecord explicitly."
            )
        extractor = EpisodeExtractor(
            encoder=self._to_feature_vector,
            fog_of_war=self.fog_of_war,
        )
        return extractor.extract(rec)

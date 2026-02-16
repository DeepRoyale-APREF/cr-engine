"""
Multi-agent environment wrapper and vectorised env.

Provides :class:`MultiAgentClashRoyaleEnv` where **both** players are
external agents, and :class:`VectorizedClashRoyaleEnv` that runs N
single-agent envs in lock-step.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from clash_royale_engine.core.engine import ClashRoyaleEngine
from clash_royale_engine.env.gymnasium_env import ClashRoyaleEnv, ObservationType
from clash_royale_engine.players.player_interface import (
    HeuristicBot,
    PlayerInterface,
    RLAgentPlayer,
)
from clash_royale_engine.utils.constants import DEFAULT_DECK, DEFAULT_FPS, GAME_DURATION


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Agent (2-player) env
# ─────────────────────────────────────────────────────────────────────────────


class MultiAgentClashRoyaleEnv:
    """
    Two-player environment: both actions are provided externally.

    Not a Gymnasium ``Env`` — use this directly for self-play.
    """

    def __init__(
        self,
        deck1: Optional[List[str]] = None,
        deck2: Optional[List[str]] = None,
        obs_type: ObservationType = ObservationType.FEATURE_VECTOR,
        fps: int = DEFAULT_FPS,
        time_limit: float = GAME_DURATION,
        seed: int = 0,
    ) -> None:
        self.obs_type = obs_type
        self.engine = ClashRoyaleEngine(
            player1=RLAgentPlayer(),
            player2=RLAgentPlayer(),
            deck1=deck1 or list(DEFAULT_DECK),
            deck2=deck2 or list(DEFAULT_DECK),
            fps=fps,
            time_limit=time_limit,
            seed=seed,
        )
        self._encoder = ClashRoyaleEnv._to_feature_vector  # reuse encoder

    def reset(self, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        self.engine.seed = seed
        self.engine.reset()
        s0 = self.engine.get_state(0)
        s1 = self.engine.get_state(1)
        return self._encoder(s0), self._encoder(s1)

    def step(
        self,
        action_p0: Optional[Tuple[int, int, int]],
        action_p1: Optional[Tuple[int, int, int]],
    ) -> Tuple[np.ndarray, np.ndarray, float, float, bool, Dict[str, Any]]:
        """
        Step both players simultaneously.

        Returns (obs0, obs1, reward0, reward1, done, info).
        """
        from clash_royale_engine.utils.validators import InvalidActionError, validate_action

        # Apply P0
        if action_p0 is not None:
            err = validate_action(0, action_p0, self.engine.players[0])
            if err is None:
                self.engine._apply_action(0, action_p0)

        # Apply P1
        if action_p1 is not None:
            err = validate_action(1, action_p1, self.engine.players[1])
            if err is None:
                self.engine._apply_action(1, action_p1)

        self.engine._simulate_frame()

        s0 = self.engine.get_state(0)
        s1 = self.engine.get_state(1)
        done = self.engine.is_done()

        # Simple sparse rewards
        r0, r1 = 0.0, 0.0
        if done:
            w = self.engine.get_winner()
            if w == 0:
                r0, r1 = 1.0, -1.0
            elif w == 1:
                r0, r1 = -1.0, 1.0

        info: Dict[str, Any] = {"winner": self.engine.get_winner()}
        return self._encoder(s0), self._encoder(s1), r0, r1, done, info


# ─────────────────────────────────────────────────────────────────────────────
# Vectorised env (N single-agent envs)
# ─────────────────────────────────────────────────────────────────────────────


class VectorizedClashRoyaleEnv:
    """Run *num_envs* :class:`ClashRoyaleEnv` instances in lock-step."""

    def __init__(self, num_envs: int = 16, **env_kwargs: Any) -> None:
        self.num_envs = num_envs
        self.envs = [ClashRoyaleEnv(**env_kwargs) for _ in range(num_envs)]

    def reset(self) -> np.ndarray:
        obs_list = []
        for env in self.envs:
            obs, _ = env.reset()
            obs_list.append(obs)
        return np.array(obs_list, dtype=np.float32)

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Step all envs (auto-reset on done).

        Returns (obs, rewards, terminated, truncated, infos).
        """
        obs_list: List[np.ndarray] = []
        rew_list: List[float] = []
        term_list: List[bool] = []
        trunc_list: List[bool] = []
        info_list: List[Dict[str, Any]] = []

        for i, env in enumerate(self.envs):
            o, r, te, tr, info = env.step(int(actions[i]))
            if te or tr:
                o, _ = env.reset()
            obs_list.append(o)
            rew_list.append(r)
            term_list.append(te)
            trunc_list.append(tr)
            info_list.append(info)

        return (
            np.array(obs_list, dtype=np.float32),
            np.array(rew_list, dtype=np.float32),
            np.array(term_list),
            np.array(trunc_list),
            info_list,
        )

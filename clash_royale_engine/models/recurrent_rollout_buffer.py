"""
Recurrent rollout buffer for PPO with LSTM.

Stores transitions in ``(n_steps, n_envs, ...)`` layout and yields
minibatches of **contiguous sequences** of length ``seq_len`` for
back-propagation through time.

GAE computation is identical to SB3's ``RolloutBuffer``.
"""

from __future__ import annotations

from typing import Generator

import numpy as np
import torch


class RecurrentRolloutBuffer:
    """Fixed-size buffer that stores one rollout (``n_steps`` per env).

    Parameters
    ----------
    n_steps : int
        Transitions collected per environment before each training phase.
    n_envs : int
        Number of parallel environments.
    obs_dim : int
        Flat observation dimensionality.
    lstm_hidden : int
        LSTM hidden-state size.
    lstm_layers : int
        Number of stacked LSTM layers.
    gamma, gae_lambda : float
        Discount factor and GAE lambda (same as baseline PPO).
    """

    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        obs_dim: int,
        lstm_hidden: int,
        lstm_layers: int = 1,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.obs_dim = obs_dim
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Pre-allocate numpy arrays
        self.observations = np.zeros((n_steps, n_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((n_steps, n_envs), dtype=np.int64)
        self.rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_envs), dtype=np.bool_)
        self.log_probs = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((n_steps, n_envs), dtype=np.float32)

        # LSTM hidden states at the *start* of each step
        self.hidden_h = np.zeros(
            (n_steps, lstm_layers, n_envs, lstm_hidden), dtype=np.float32
        )
        self.hidden_c = np.zeros(
            (n_steps, lstm_layers, n_envs, lstm_hidden), dtype=np.float32
        )

        # Filled after rollout via compute_returns_and_advantages()
        self.advantages = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.returns = np.zeros((n_steps, n_envs), dtype=np.float32)

        self.ptr: int = 0
        self.full: bool = False

    # ── storage ───────────────────────────────────────────────────────

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        log_prob: np.ndarray,
        value: np.ndarray,
        hidden_h: np.ndarray,
        hidden_c: np.ndarray,
    ) -> None:
        """Store one step of experience across all envs."""
        assert self.ptr < self.n_steps, "Buffer full — call reset() first"

        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward.astype(np.float32)
        self.dones[self.ptr] = done
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.hidden_h[self.ptr] = hidden_h
        self.hidden_c[self.ptr] = hidden_c

        self.ptr += 1
        if self.ptr == self.n_steps:
            self.full = True

    # ── GAE (identical algorithm to SB3) ──────────────────────────────

    def compute_returns_and_advantages(
        self, last_values: np.ndarray, last_dones: np.ndarray
    ) -> None:
        """Compute GAE advantages and discounted returns.

        Parameters
        ----------
        last_values : ``(n_envs,)``
            Bootstrap ``V(s_T)`` for each env.
        last_dones : ``(n_envs,)``
            Done flags **at** the last collected step (used to mask the
            bootstrap for environments whose episode just ended).
        """
        gae = np.zeros(self.n_envs, dtype=np.float32)

        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_values = last_values
                next_non_terminal = 1.0 - last_dones.astype(np.float32)
            else:
                next_values = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t].astype(np.float32)

            delta = (
                self.rewards[t]
                + self.gamma * next_values * next_non_terminal
                - self.values[t]
            )
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae

        self.returns = self.advantages + self.values

    # ── recurrent minibatch generator ─────────────────────────────────

    def recurrent_generator(
        self,
        seq_len: int,
        num_minibatches: int,
    ) -> Generator[dict[str, torch.Tensor], None, None]:
        """Yield minibatches of contiguous sequences for LSTM training.

        The ``n_steps`` rollout is split into non-overlapping chunks of
        ``seq_len``.  Chunks from different envs are shuffled and grouped
        into ``num_minibatches`` per epoch.

        Yields
        ------
        dict with keys:
            obs            : ``(T, mb, obs_dim)``
            actions        : ``(T, mb)``
            old_log_probs  : ``(T, mb)``
            advantages     : ``(T, mb)``
            returns        : ``(T, mb)``
            hidden_h       : ``(layers, mb, lstm_hidden)`` — initial h
            hidden_c       : ``(layers, mb, lstm_hidden)`` — initial c
            dones          : ``(T, mb)``
        """
        n_seqs_per_env = self.n_steps // seq_len
        assert n_seqs_per_env > 0, (
            f"n_steps ({self.n_steps}) must be >= seq_len ({seq_len})"
        )
        total_seqs = n_seqs_per_env * self.n_envs
        seqs_per_mb = max(1, total_seqs // num_minibatches)
        # Recompute actual num_minibatches to avoid dropping too many seqs
        actual_mbs = total_seqs // seqs_per_mb

        # Build (env_idx, start_step) index pairs and shuffle
        indices: list[tuple[int, int]] = []
        for env in range(self.n_envs):
            for s in range(n_seqs_per_env):
                indices.append((env, s * seq_len))

        rng = np.random.default_rng()
        rng.shuffle(indices)  # type: ignore[arg-type]

        for mb in range(actual_mbs):
            mb_indices = indices[mb * seqs_per_mb : (mb + 1) * seqs_per_mb]
            mb_size = len(mb_indices)

            obs_b = np.zeros((seq_len, mb_size, self.obs_dim), dtype=np.float32)
            act_b = np.zeros((seq_len, mb_size), dtype=np.int64)
            logp_b = np.zeros((seq_len, mb_size), dtype=np.float32)
            adv_b = np.zeros((seq_len, mb_size), dtype=np.float32)
            ret_b = np.zeros((seq_len, mb_size), dtype=np.float32)
            done_b = np.zeros((seq_len, mb_size), dtype=np.bool_)
            h_b = np.zeros(
                (self.lstm_layers, mb_size, self.lstm_hidden), dtype=np.float32
            )
            c_b = np.zeros(
                (self.lstm_layers, mb_size, self.lstm_hidden), dtype=np.float32
            )

            for i, (env_idx, start) in enumerate(mb_indices):
                end = start + seq_len
                obs_b[:, i] = self.observations[start:end, env_idx]
                act_b[:, i] = self.actions[start:end, env_idx]
                logp_b[:, i] = self.log_probs[start:end, env_idx]
                adv_b[:, i] = self.advantages[start:end, env_idx]
                ret_b[:, i] = self.returns[start:end, env_idx]
                done_b[:, i] = self.dones[start:end, env_idx]
                h_b[:, i] = self.hidden_h[start, :, env_idx]
                c_b[:, i] = self.hidden_c[start, :, env_idx]

            yield {
                "obs": torch.from_numpy(obs_b),
                "actions": torch.from_numpy(act_b),
                "old_log_probs": torch.from_numpy(logp_b),
                "advantages": torch.from_numpy(adv_b),
                "returns": torch.from_numpy(ret_b),
                "hidden_h": torch.from_numpy(h_b),
                "hidden_c": torch.from_numpy(c_b),
                "dones": torch.from_numpy(done_b),
            }

    # ── reset ─────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset buffer pointer (re-uses allocated memory)."""
        self.ptr = 0
        self.full = False

"""
CNN + LSTM actor-critic policy for Clash Royale.

Architecture
------------
(A) Grid ``(2, 32, 18)`` — ally/enemy presence  → **CNN**  → 128-d embedding
(B) Scalars ``(44,)``     — elixir, tower HP, cards → **MLP** → 64-d embedding

Concat ``(192,)`` → **LSTM** (hidden=128, 1 layer) → policy head + value head

The observation is the **same** flat feature vector produced by the baseline
environment (``OBS_FEATURE_DIM`` values) — we just *reshape* the spatial part
into channels instead of treating everything as a flat vector.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from clash_royale_engine.utils.constants import (
    CARD_VOCAB,
    HAND_SIZE,
    N_HEIGHT_TILES,
    N_WIDE_TILES,
)

# ─────────────────────── observation layout ────────────────────────────────
# Feature vector produced by ClashRoyaleEnv._to_feature_vector():
#   [0  :  2)   elixir (own, enemy)
#   [2  :  8)   tower HP (6 values)
#   [8  : 44)   cards in hand  4 × (8 one-hot + 1 cost) = 36
#   [44 :620)   ally presence grid (32×18 row-major)
#   [620:1196)  enemy presence grid (32×18 row-major)

SCALAR_DIM: int = 2 + 6 + (len(CARD_VOCAB) + 1) * HAND_SIZE  # 44
GRID_CHANNELS: int = 2  # ally + enemy
GRID_H: int = N_HEIGHT_TILES  # 32
GRID_W: int = N_WIDE_TILES  # 18
GRID_FLAT: int = GRID_CHANNELS * GRID_H * GRID_W  # 1152


# ═══════════════════════════════════════════════════════════════════════════
# Sub-encoders
# ═══════════════════════════════════════════════════════════════════════════


class CnnEncoder(nn.Module):
    """3-layer CNN spatial encoder for the ``(2, 32, 18)`` presence grid."""

    def __init__(self, out_dim: int = 128) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(GRID_CHANNELS, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Infer the flattened size from a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, GRID_CHANNELS, GRID_H, GRID_W)
            flat_size: int = self.cnn(dummy).shape[1]
        self.fc = nn.Linear(flat_size, out_dim)
        self.out_act = nn.ReLU()

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """``(B, 2, H, W) → (B, out_dim)``"""
        return self.out_act(self.fc(self.cnn(grid)))


class ScalarEncoder(nn.Module):
    """Two-layer MLP for the scalar part of the observation."""

    def __init__(self, in_dim: int = SCALAR_DIM, out_dim: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
            nn.ReLU(),
        )

    def forward(self, scalars: torch.Tensor) -> torch.Tensor:
        """``(B, in_dim) → (B, out_dim)``"""
        return self.mlp(scalars)


# ═══════════════════════════════════════════════════════════════════════════
# Full policy
# ═══════════════════════════════════════════════════════════════════════════


class CnnLstmPolicy(nn.Module):
    """CNN + LSTM actor-critic.

    Parameters
    ----------
    n_actions : int
        Size of the discrete action space (2305 for CR env).
    cnn_out : int
        Dimensionality of the CNN embedding.
    scalar_out : int
        Dimensionality of the scalar MLP embedding.
    lstm_hidden : int
        LSTM hidden-state size.
    lstm_layers : int
        Number of stacked LSTM layers (default 1).
    """

    def __init__(
        self,
        n_actions: int,
        cnn_out: int = 128,
        scalar_out: int = 64,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
    ) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers

        self.cnn_encoder = CnnEncoder(out_dim=cnn_out)
        self.scalar_encoder = ScalarEncoder(out_dim=scalar_out)

        lstm_input_size = cnn_out + scalar_out  # 192
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=False,  # (seq, batch, feat)
        )

        self.policy_head = nn.Linear(lstm_hidden, n_actions)
        self.value_head = nn.Linear(lstm_hidden, 1)

    # ── helpers ───────────────────────────────────────────────────────

    def initial_state(
        self, batch_size: int = 1, device: str | torch.device = "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return zero ``(h, c)`` for the LSTM."""
        h = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden, device=device)
        c = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden, device=device)
        return h, c

    @staticmethod
    def _split_obs(obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split the flat observation into ``(scalars, grid)``.

        Parameters
        ----------
        obs : Tensor
            Shape ``(..., obs_dim)``.

        Returns
        -------
        scalars : ``(..., SCALAR_DIM)``
        grid    : ``(..., 2, GRID_H, GRID_W)``
        """
        leading = obs.shape[:-1]
        scalars = obs[..., :SCALAR_DIM]
        grid_flat = obs[..., SCALAR_DIM : SCALAR_DIM + GRID_FLAT]
        grid = grid_flat.reshape(*leading, GRID_CHANNELS, GRID_H, GRID_W)
        return scalars, grid

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode a batch of observations.

        ``(B, obs_dim) → (B, cnn_out + scalar_out)``
        """
        scalars, grid = self._split_obs(obs)
        return torch.cat(
            [self.cnn_encoder(grid), self.scalar_encoder(scalars)], dim=-1
        )

    # ── forward (sequence) ────────────────────────────────────────────

    def forward(
        self,
        obs_seq: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        done_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass over a *sequence* of observations.

        Parameters
        ----------
        obs_seq : ``(T, B, obs_dim)``
        hidden  : ``(h, c)``  each ``(num_layers, B, lstm_hidden)``
        done_mask : ``(T, B)`` bool — ``True`` where episode ended.

        Returns
        -------
        logits     : ``(T, B, n_actions)``
        values     : ``(T, B, 1)``
        new_hidden : ``(h, c)``
        """
        seq_len, batch, _ = obs_seq.shape

        # Encode every (t, b) pair through CNN + MLP
        flat_obs = obs_seq.reshape(seq_len * batch, -1)
        flat_emb = self._encode(flat_obs)
        emb = flat_emb.reshape(seq_len, batch, -1)

        # LSTM — step-by-step when done_mask is provided so we can reset
        if done_mask is not None:
            outputs: list[torch.Tensor] = []
            h, c = hidden
            for t in range(seq_len):
                # Reset hidden for envs whose previous step was terminal
                if t > 0:
                    mask = done_mask[t - 1]  # (B,)
                    if mask.any():
                        keep = (~mask).float().unsqueeze(0).unsqueeze(-1)
                        h = h * keep
                        c = c * keep
                out, (h, c) = self.lstm(emb[t : t + 1], (h, c))
                outputs.append(out)
            lstm_out = torch.cat(outputs, dim=0)
            new_hidden = (h, c)
        else:
            lstm_out, new_hidden = self.lstm(emb, hidden)

        logits = self.policy_head(lstm_out)
        values = self.value_head(lstm_out)
        return logits, values, new_hidden

    # ── act (single step, used during rollout collection) ─────────────

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Select an action for **one** timestep.

        Parameters
        ----------
        obs : ``(B, obs_dim)``
        hidden : ``(h, c)``
        deterministic : take argmax instead of sampling.

        Returns
        -------
        action    : ``(B,)``
        log_prob  : ``(B,)``
        value     : ``(B,)``
        new_hidden : ``(h, c)``
        """
        obs_seq = obs.unsqueeze(0)  # (1, B, obs_dim)
        logits, values, new_hidden = self.forward(obs_seq, hidden)

        logits = logits.squeeze(0)  # (B, n_actions)
        values = values.squeeze(0).squeeze(-1)  # (B,)

        dist = torch.distributions.Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob, values, new_hidden

    # ── evaluate (training, computes gradients) ───────────────────────

    def evaluate_actions(
        self,
        obs_seq: torch.Tensor,
        actions: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        done_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-evaluate a batch of actions (for PPO loss).

        Parameters
        ----------
        obs_seq  : ``(T, B, obs_dim)``
        actions  : ``(T, B)``
        hidden   : initial ``(h, c)``
        done_mask : ``(T, B)`` bool

        Returns
        -------
        log_probs : ``(T, B)``
        values    : ``(T, B)``
        entropy   : ``(T, B)``
        """
        logits, values, _ = self.forward(obs_seq, hidden, done_mask)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values.squeeze(-1), entropy

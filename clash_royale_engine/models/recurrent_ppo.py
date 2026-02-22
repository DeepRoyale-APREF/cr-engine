"""
Recurrent PPO trainer for CNN + LSTM policy.

Keeps the **same PPO algorithm** as the Stable-Baselines3 baseline:

* GAE advantage estimation (``gamma``, ``gae_lambda``)
* Clipped surrogate objective (``clip_range``)
* Value-function MSE loss (``vf_coef``)
* Entropy bonus (``ent_coef``)
* Adam optimiser with gradient clipping (``max_grad_norm``)

The only change is the **architecture** (CNN + LSTM instead of MLP) and
the bookkeeping needed to propagate LSTM hidden states correctly:

1.  During rollout collection, ``(h, c)`` are carried from step to step
    and zeroed on episode boundaries (``done``).
2.  The rollout buffer stores ``(h, c)`` at every step so that training
    can reconstruct them for each minibatch sequence.
3.  Minibatches are **contiguous temporal sequences** (length ``seq_len``)
    rather than randomly sampled individual transitions.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from clash_royale_engine.models.cnn_lstm_policy import CnnLstmPolicy
from clash_royale_engine.models.recurrent_rollout_buffer import RecurrentRolloutBuffer


class RecurrentPPO:
    """PPO trainer with LSTM hidden-state management.

    Parameters
    ----------
    policy : CnnLstmPolicy
        The actor-critic network.
    env
        A vectorised environment that exposes the SB3 VecEnv API
        (``reset() → obs``, ``step(a) → (obs, rew, done, info)``).
    learning_rate, n_steps, batch_size, n_epochs, gamma, gae_lambda,
    clip_range, ent_coef, vf_coef, max_grad_norm
        Identical to SB3 ``PPO`` hyperparameters.
    seq_len : int
        Sequence length for BPTT (default 16).
    device : str
        ``"auto"`` selects CUDA if available.
    tensorboard_log : str | None
        Directory for TensorBoard logs.
    """

    def __init__(
        self,
        policy: CnnLstmPolicy,
        env: Any,
        *,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        seq_len: int = 16,
        device: str = "auto",
        seed: int = 42,
        tensorboard_log: Optional[str] = None,
        verbose: int = 1,
    ) -> None:
        self.env = env
        self.n_envs: int = env.num_envs
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.seq_len = seq_len
        self.verbose = verbose
        self.tensorboard_log = tensorboard_log
        self.seed = seed

        # Ensure n_steps is divisible by seq_len
        assert n_steps % seq_len == 0, (
            f"n_steps ({n_steps}) must be divisible by seq_len ({seq_len})"
        )

        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.policy = policy.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=learning_rate, eps=1e-5
        )

        # Observation / action dims
        obs_shape = env.observation_space.shape
        self.obs_dim: int = obs_shape[0] if obs_shape else 1
        self.n_actions: int = env.action_space.n

        # Derive num_minibatches from batch_size so total transitions
        # processed per epoch matches the MLP baseline exactly.
        n_seqs_per_env = n_steps // seq_len
        total_seqs = n_seqs_per_env * self.n_envs
        seqs_per_mb = max(1, batch_size // seq_len)
        self.num_minibatches: int = max(1, total_seqs // seqs_per_mb)

        # Rollout buffer
        self.buffer = RecurrentRolloutBuffer(
            n_steps=n_steps,
            n_envs=self.n_envs,
            obs_dim=self.obs_dim,
            lstm_hidden=policy.lstm_hidden,
            lstm_layers=policy.lstm_layers,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        # Running LSTM hidden state — one per env
        self._hidden: Tuple[torch.Tensor, torch.Tensor] = policy.initial_state(
            self.n_envs, self.device
        )

        # Episode metric tracking
        self._ep_rewards = np.zeros(self.n_envs, dtype=np.float64)
        self._ep_lengths = np.zeros(self.n_envs, dtype=np.int64)
        self._completed_ep_rewards: List[float] = []
        self._completed_ep_lengths: List[int] = []

        # Logging
        self.num_timesteps: int = 0
        self._writer: Any = None
        if tensorboard_log:
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(tensorboard_log)

    # ─────────────────────────────────────────────────────────────────
    # Hidden-state helpers
    # ─────────────────────────────────────────────────────────────────

    def _reset_hidden_for_dones(self, dones: np.ndarray) -> None:
        """Zero hidden state for environments that just finished."""
        if not dones.any():
            return
        mask = torch.from_numpy(dones.astype(np.float32)).to(self.device)
        # (n_envs,) → (1, n_envs, 1)  for broadcasting with (layers, n_envs, hidden)
        mask = mask.unsqueeze(0).unsqueeze(-1)
        h, c = self._hidden
        self._hidden = (h * (1.0 - mask), c * (1.0 - mask))

    # ─────────────────────────────────────────────────────────────────
    # Rollout collection
    # ─────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _collect_rollout(self) -> None:
        """Collect ``n_steps`` of experience from all envs."""
        self.policy.eval()
        self.buffer.reset()

        obs: np.ndarray = self._last_obs  # (n_envs, obs_dim)

        for _ in range(self.n_steps):
            obs_t = torch.from_numpy(obs).float().to(self.device)

            # Snapshot hidden state BEFORE this step
            h_np = self._hidden[0].cpu().numpy()
            c_np = self._hidden[1].cpu().numpy()

            action, log_prob, value, new_hidden = self.policy.act(obs_t, self._hidden)

            action_np = action.cpu().numpy()
            log_prob_np = log_prob.cpu().numpy()
            value_np = value.cpu().numpy()

            # Step the vectorised environment (SB3 VecEnv API: auto-resets)
            new_obs, rewards, dones, infos = self.env.step(action_np)

            self.buffer.add(
                obs=obs,
                action=action_np,
                reward=rewards,
                done=dones,
                log_prob=log_prob_np,
                value=value_np,
                hidden_h=h_np,
                hidden_c=c_np,
            )

            self._hidden = new_hidden
            self._reset_hidden_for_dones(dones)

            # Episode metrics
            self._ep_rewards += rewards
            self._ep_lengths += 1
            for i in range(self.n_envs):
                if dones[i]:
                    self._completed_ep_rewards.append(float(self._ep_rewards[i]))
                    self._completed_ep_lengths.append(int(self._ep_lengths[i]))
                    self._ep_rewards[i] = 0.0
                    self._ep_lengths[i] = 0

            obs = new_obs
            self.num_timesteps += self.n_envs

        self._last_obs = obs

        # Bootstrap value for GAE
        obs_t = torch.from_numpy(obs).float().to(self.device)
        _, _, last_value, _ = self.policy.act(obs_t, self._hidden)
        last_values = last_value.cpu().numpy()
        last_dones = self.buffer.dones[self.n_steps - 1]

        self.buffer.compute_returns_and_advantages(last_values, last_dones)

    # ─────────────────────────────────────────────────────────────────
    # One PPO optimisation epoch
    # ─────────────────────────────────────────────────────────────────

    def _train_epoch(self) -> Dict[str, float]:
        """One pass over all minibatches — same losses as SB3 PPO."""
        self.policy.train()

        total_pg = 0.0
        total_vf = 0.0
        total_ent = 0.0
        total_loss = 0.0
        n_updates = 0

        for batch in self.buffer.recurrent_generator(
            self.seq_len, self.num_minibatches
        ):
            obs = batch["obs"].to(self.device)  # (T, mb, obs_dim)
            actions = batch["actions"].to(self.device)  # (T, mb)
            old_lp = batch["old_log_probs"].to(self.device)  # (T, mb)
            advantages = batch["advantages"].to(self.device)  # (T, mb)
            returns = batch["returns"].to(self.device)  # (T, mb)
            h0 = batch["hidden_h"].to(self.device)  # (layers, mb, hidden)
            c0 = batch["hidden_c"].to(self.device)
            dones = batch["dones"].to(self.device)  # (T, mb)

            # Normalise advantages (per-minibatch, like SB3)
            flat = advantages.flatten()
            advantages = (advantages - flat.mean()) / (flat.std() + 1e-8)

            # Re-evaluate actions through current policy
            new_lp, new_v, entropy = self.policy.evaluate_actions(
                obs, actions, (h0, c0), dones
            )

            # ── Clipped surrogate (policy loss) ──
            ratio = torch.exp(new_lp - old_lp)
            pg1 = -advantages * ratio
            pg2 = -advantages * torch.clamp(
                ratio, 1.0 - self.clip_range, 1.0 + self.clip_range
            )
            pg_loss = torch.max(pg1, pg2).mean()

            # ── Value-function loss (un-clipped, same as SB3 default) ──
            vf_loss = nn.functional.mse_loss(new_v, returns)

            # ── Entropy bonus ──
            entropy_loss = -entropy.mean()

            # ── Total loss ──
            loss = pg_loss + self.vf_coef * vf_loss + self.ent_coef * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_pg += pg_loss.item()
            total_vf += vf_loss.item()
            total_ent += entropy.mean().item()
            total_loss += loss.item()
            n_updates += 1

        d = max(n_updates, 1)
        return {
            "pg_loss": total_pg / d,
            "vf_loss": total_vf / d,
            "entropy": total_ent / d,
            "total_loss": total_loss / d,
        }

    # ─────────────────────────────────────────────────────────────────
    # Main training loop
    # ─────────────────────────────────────────────────────────────────

    def learn(
        self,
        total_timesteps: int,
        callback: Optional[Callable[["RecurrentPPO", int], None]] = None,
        progress_bar: bool = False,
    ) -> "RecurrentPPO":
        """Train the policy.

        Parameters
        ----------
        total_timesteps : int
            Total environment steps (across all envs).
        callback : callable, optional
            ``callback(trainer, iteration)`` called after each rollout+train.
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Initialise
        self._last_obs = self.env.reset()
        self._hidden = self.policy.initial_state(self.n_envs, self.device)
        self.num_timesteps = 0
        self._completed_ep_rewards.clear()
        self._completed_ep_lengths.clear()

        steps_per_iter = self.n_steps * self.n_envs
        n_iterations = max(1, total_timesteps // steps_per_iter)

        if self.verbose >= 1:
            print(
                f"  Training for {n_iterations} iterations "
                f"({steps_per_iter} steps each, "
                f"seq_len={self.seq_len}, "
                f"minibatches={self.num_minibatches})"
            )

        t0 = time.perf_counter()

        for iteration in range(1, n_iterations + 1):
            self._collect_rollout()

            # Train n_epochs over the buffer
            last_metrics: Dict[str, float] = {}
            for _ in range(self.n_epochs):
                last_metrics = self._train_epoch()

            # Logging
            elapsed = time.perf_counter() - t0
            fps = self.num_timesteps / max(elapsed, 1e-6)

            if self._writer:
                self._writer.add_scalar(
                    "train/pg_loss", last_metrics["pg_loss"], self.num_timesteps
                )
                self._writer.add_scalar(
                    "train/vf_loss", last_metrics["vf_loss"], self.num_timesteps
                )
                self._writer.add_scalar(
                    "train/entropy", last_metrics["entropy"], self.num_timesteps
                )
                self._writer.add_scalar(
                    "train/total_loss", last_metrics["total_loss"], self.num_timesteps
                )
                self._writer.add_scalar("time/fps", fps, self.num_timesteps)

                # Episode-level metrics (rolling window of last 50)
                if self._completed_ep_rewards:
                    recent_r = self._completed_ep_rewards[-50:]
                    recent_l = self._completed_ep_lengths[-50:]
                    self._writer.add_scalar(
                        "rollout/ep_rew_mean",
                        float(np.mean(recent_r)),
                        self.num_timesteps,
                    )
                    self._writer.add_scalar(
                        "rollout/ep_len_mean",
                        float(np.mean(recent_l)),
                        self.num_timesteps,
                    )

            log_interval = max(n_iterations // 20, 1)
            if self.verbose >= 1 and iteration % log_interval == 0:
                ep_str = ""
                if self._completed_ep_rewards:
                    recent = self._completed_ep_rewards[-50:]
                    ep_str = f" | ep_rew={np.mean(recent):.2f}"
                print(
                    f"  Iter {iteration}/{n_iterations} | "
                    f"steps={self.num_timesteps:,} | "
                    f"pg={last_metrics['pg_loss']:.4f} | "
                    f"vf={last_metrics['vf_loss']:.4f} | "
                    f"ent={last_metrics['entropy']:.4f} | "
                    f"fps={fps:.0f}{ep_str}"
                )

            if callback is not None:
                callback(self, iteration)

        if self._writer:
            self._writer.flush()

        return self

    # ─────────────────────────────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────────────────────────────

    def predict(
        self,
        obs: np.ndarray,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        deterministic: bool = True,
    ) -> Tuple[int, Tuple[torch.Tensor, torch.Tensor]]:
        """Predict one action (for evaluation).

        Returns ``(action_int, new_hidden)``.
        """
        self.policy.eval()
        with torch.no_grad():
            obs_t = (
                torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            )  # (1, obs_dim)
            if hidden is None:
                hidden = self.policy.initial_state(1, self.device)
            action, _, _, new_hidden = self.policy.act(
                obs_t, hidden, deterministic=deterministic
            )
            return int(action.cpu().item()), new_hidden

    # ─────────────────────────────────────────────────────────────────
    # Serialisation
    # ─────────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save model + optimiser checkpoint."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "num_timesteps": self.num_timesteps,
            },
            path,
        )
        if self.verbose >= 1:
            print(f"  Model saved to {path}")

    def load(self, path: str) -> None:
        """Load model + optimiser checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.num_timesteps = ckpt.get("num_timesteps", 0)

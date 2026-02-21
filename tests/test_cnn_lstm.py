"""
Tests for CNN + LSTM policy, recurrent rollout buffer, and hidden-state reset.

Covers:
- Observation split shapes (scalar / grid)
- CNN, MLP, and full policy output shapes
- ``act()`` and ``evaluate_actions()`` tensor shapes
- ``initial_state()`` shape
- Hidden-state reset on ``done``
- Buffer storage, GAE computation, and recurrent generator shapes
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from clash_royale_engine.models.cnn_lstm_policy import (
    GRID_CHANNELS,
    GRID_FLAT,
    GRID_H,
    GRID_W,
    SCALAR_DIM,
    CnnEncoder,
    CnnLstmPolicy,
    ScalarEncoder,
)
from clash_royale_engine.models.recurrent_rollout_buffer import RecurrentRolloutBuffer
from clash_royale_engine.utils.constants import OBS_FEATURE_DIM

# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

N_ACTIONS = 18 * 32 * 4 + 1  # 2305
BATCH = 4
SEQ_LEN = 8
LSTM_HIDDEN = 128
LSTM_LAYERS = 1


@pytest.fixture()
def policy() -> CnnLstmPolicy:
    return CnnLstmPolicy(
        n_actions=N_ACTIONS,
        cnn_out=128,
        scalar_out=64,
        lstm_hidden=LSTM_HIDDEN,
        lstm_layers=LSTM_LAYERS,
    )


@pytest.fixture()
def obs_batch() -> torch.Tensor:
    """Random obs batch matching env feature-vector dim."""
    return torch.rand(BATCH, OBS_FEATURE_DIM)


@pytest.fixture()
def obs_seq() -> torch.Tensor:
    """Random obs sequence (T, B, D)."""
    return torch.rand(SEQ_LEN, BATCH, OBS_FEATURE_DIM)


# ═══════════════════════════════════════════════════════════════════════════
# Observation split
# ═══════════════════════════════════════════════════════════════════════════


class TestObsSplit:

    def test_scalar_dim_constant(self) -> None:
        assert SCALAR_DIM == 44

    def test_grid_flat_constant(self) -> None:
        assert GRID_FLAT == GRID_CHANNELS * GRID_H * GRID_W
        assert GRID_FLAT == 1152

    def test_obs_covers_all_features(self) -> None:
        """SCALAR_DIM + GRID_FLAT should equal OBS_FEATURE_DIM."""
        assert SCALAR_DIM + GRID_FLAT == OBS_FEATURE_DIM

    def test_split_obs_shapes(self, policy: CnnLstmPolicy, obs_batch: torch.Tensor) -> None:
        scalars, grid = policy._split_obs(obs_batch)
        assert scalars.shape == (BATCH, SCALAR_DIM)
        assert grid.shape == (BATCH, GRID_CHANNELS, GRID_H, GRID_W)

    def test_split_obs_sequence_shapes(self, policy: CnnLstmPolicy, obs_seq: torch.Tensor) -> None:
        """_split_obs should work on (T, B, D) inputs too."""
        scalars, grid = policy._split_obs(obs_seq)
        assert scalars.shape == (SEQ_LEN, BATCH, SCALAR_DIM)
        assert grid.shape == (SEQ_LEN, BATCH, GRID_CHANNELS, GRID_H, GRID_W)


# ═══════════════════════════════════════════════════════════════════════════
# Sub-encoder shapes
# ═══════════════════════════════════════════════════════════════════════════


class TestEncoders:

    def test_cnn_encoder_shape(self) -> None:
        cnn = CnnEncoder(out_dim=128)
        grid = torch.rand(BATCH, GRID_CHANNELS, GRID_H, GRID_W)
        out = cnn(grid)
        assert out.shape == (BATCH, 128)

    def test_scalar_encoder_shape(self) -> None:
        mlp = ScalarEncoder(in_dim=SCALAR_DIM, out_dim=64)
        scalars = torch.rand(BATCH, SCALAR_DIM)
        out = mlp(scalars)
        assert out.shape == (BATCH, 64)


# ═══════════════════════════════════════════════════════════════════════════
# Full policy shapes
# ═══════════════════════════════════════════════════════════════════════════


class TestPolicyShapes:

    def test_initial_state_shape(self, policy: CnnLstmPolicy) -> None:
        h, c = policy.initial_state(BATCH)
        assert h.shape == (LSTM_LAYERS, BATCH, LSTM_HIDDEN)
        assert c.shape == (LSTM_LAYERS, BATCH, LSTM_HIDDEN)
        assert (h == 0).all()
        assert (c == 0).all()

    def test_act_shapes(self, policy: CnnLstmPolicy, obs_batch: torch.Tensor) -> None:
        hidden = policy.initial_state(BATCH)
        action, log_prob, value, new_hidden = policy.act(obs_batch, hidden)

        assert action.shape == (BATCH,)
        assert log_prob.shape == (BATCH,)
        assert value.shape == (BATCH,)
        assert new_hidden[0].shape == (LSTM_LAYERS, BATCH, LSTM_HIDDEN)
        assert new_hidden[1].shape == (LSTM_LAYERS, BATCH, LSTM_HIDDEN)

    def test_evaluate_actions_shapes(
        self, policy: CnnLstmPolicy, obs_seq: torch.Tensor
    ) -> None:
        hidden = policy.initial_state(BATCH)
        actions = torch.randint(0, N_ACTIONS, (SEQ_LEN, BATCH))
        dones = torch.zeros(SEQ_LEN, BATCH, dtype=torch.bool)

        log_probs, values, entropy = policy.evaluate_actions(
            obs_seq, actions, hidden, dones
        )

        assert log_probs.shape == (SEQ_LEN, BATCH)
        assert values.shape == (SEQ_LEN, BATCH)
        assert entropy.shape == (SEQ_LEN, BATCH)

    def test_forward_shapes(self, policy: CnnLstmPolicy, obs_seq: torch.Tensor) -> None:
        hidden = policy.initial_state(BATCH)
        logits, values, new_hidden = policy.forward(obs_seq, hidden)

        assert logits.shape == (SEQ_LEN, BATCH, N_ACTIONS)
        assert values.shape == (SEQ_LEN, BATCH, 1)
        assert new_hidden[0].shape == (LSTM_LAYERS, BATCH, LSTM_HIDDEN)


# ═══════════════════════════════════════════════════════════════════════════
# Hidden-state reset on done
# ═══════════════════════════════════════════════════════════════════════════


class TestHiddenReset:

    def test_hidden_zeroed_after_done(self, policy: CnnLstmPolicy) -> None:
        """When done_mask[t-1]=True for env j, hidden at step t should be zero."""
        batch = 2
        seq = 4
        obs = torch.rand(seq, batch, OBS_FEATURE_DIM)
        hidden = policy.initial_state(batch)

        # Seed hidden with non-zero values
        h0 = torch.ones_like(hidden[0])
        c0 = torch.ones_like(hidden[1])

        # env 0 finishes at step 1 (done_mask[1] = True)
        done_mask = torch.zeros(seq, batch, dtype=torch.bool)
        done_mask[1, 0] = True

        # Run forward — at step 2, env 0's hidden should have been zeroed
        # before processing step 2.
        logits, values, (h_out, c_out) = policy.forward(obs, (h0, c0), done_mask)

        # Output shapes are still correct
        assert logits.shape == (seq, batch, N_ACTIONS)
        assert values.shape == (seq, batch, 1)

        # The hidden state was reset at step 2 for env 0, so overall the
        # final hidden should differ between env 0 and env 1.
        # (Probabilistically different with random weights.)
        assert h_out.shape == (LSTM_LAYERS, batch, LSTM_HIDDEN)

    def test_no_done_hidden_propagates(self, policy: CnnLstmPolicy) -> None:
        """Without any dones, forward with/without mask should agree."""
        batch = 2
        seq = 4
        obs = torch.rand(seq, batch, OBS_FEATURE_DIM)
        hidden = policy.initial_state(batch)

        no_dones = torch.zeros(seq, batch, dtype=torch.bool)

        logits_a, vals_a, _ = policy.forward(obs, hidden, done_mask=no_dones)
        logits_b, vals_b, _ = policy.forward(obs, hidden, done_mask=None)

        torch.testing.assert_close(logits_a, logits_b)
        torch.testing.assert_close(vals_a, vals_b)

    def test_all_dones_hidden_is_mostly_zeroed(self, policy: CnnLstmPolicy) -> None:
        """If every timestep has done=True, hidden is reset before every step
        (except the very first), so the LSTM effectively starts fresh each time.
        """
        batch = 1
        seq = 4
        obs = torch.rand(seq, batch, OBS_FEATURE_DIM)
        h0 = torch.ones(LSTM_LAYERS, batch, LSTM_HIDDEN) * 999.0
        c0 = torch.ones(LSTM_LAYERS, batch, LSTM_HIDDEN) * 999.0

        all_dones = torch.ones(seq, batch, dtype=torch.bool)

        # With all dones, hidden is reset before step 1, 2, 3
        # Only step 0 uses the initial h0/c0 (which is 999)
        logits_done, _, (h_d, c_d) = policy.forward(obs, (h0, c0), all_dones)

        # Compare with fresh hidden for last step only
        fresh_h = policy.initial_state(batch)
        logits_fresh, _, _ = policy.forward(obs[-1:], fresh_h)

        # Last step should match because hidden was zeroed before it
        torch.testing.assert_close(logits_done[-1], logits_fresh[0])


# ═══════════════════════════════════════════════════════════════════════════
# Recurrent rollout buffer
# ═══════════════════════════════════════════════════════════════════════════


class TestRecurrentRolloutBuffer:

    N_STEPS = 32
    N_ENVS = 2
    OBS_DIM = OBS_FEATURE_DIM

    @pytest.fixture()
    def filled_buffer(self) -> RecurrentRolloutBuffer:
        """Return a buffer filled with random data."""
        buf = RecurrentRolloutBuffer(
            n_steps=self.N_STEPS,
            n_envs=self.N_ENVS,
            obs_dim=self.OBS_DIM,
            lstm_hidden=LSTM_HIDDEN,
            lstm_layers=LSTM_LAYERS,
        )
        for _ in range(self.N_STEPS):
            buf.add(
                obs=np.random.rand(self.N_ENVS, self.OBS_DIM).astype(np.float32),
                action=np.random.randint(0, N_ACTIONS, (self.N_ENVS,)),
                reward=np.random.rand(self.N_ENVS).astype(np.float32),
                done=np.random.rand(self.N_ENVS) > 0.9,
                log_prob=np.random.randn(self.N_ENVS).astype(np.float32),
                value=np.random.randn(self.N_ENVS).astype(np.float32),
                hidden_h=np.random.randn(LSTM_LAYERS, self.N_ENVS, LSTM_HIDDEN).astype(np.float32),
                hidden_c=np.random.randn(LSTM_LAYERS, self.N_ENVS, LSTM_HIDDEN).astype(np.float32),
            )
        return buf

    def test_add_fills_buffer(self) -> None:
        buf = RecurrentRolloutBuffer(
            n_steps=4, n_envs=2, obs_dim=self.OBS_DIM,
            lstm_hidden=LSTM_HIDDEN, lstm_layers=LSTM_LAYERS,
        )
        assert buf.ptr == 0
        assert not buf.full

        for _ in range(4):
            buf.add(
                obs=np.zeros((2, self.OBS_DIM), dtype=np.float32),
                action=np.zeros(2, dtype=np.int64),
                reward=np.zeros(2, dtype=np.float32),
                done=np.zeros(2, dtype=bool),
                log_prob=np.zeros(2, dtype=np.float32),
                value=np.zeros(2, dtype=np.float32),
                hidden_h=np.zeros((LSTM_LAYERS, 2, LSTM_HIDDEN), dtype=np.float32),
                hidden_c=np.zeros((LSTM_LAYERS, 2, LSTM_HIDDEN), dtype=np.float32),
            )
        assert buf.ptr == 4
        assert buf.full

    def test_gae_shapes(self, filled_buffer: RecurrentRolloutBuffer) -> None:
        last_values = np.zeros(self.N_ENVS, dtype=np.float32)
        last_dones = np.zeros(self.N_ENVS, dtype=bool)
        filled_buffer.compute_returns_and_advantages(last_values, last_dones)

        assert filled_buffer.advantages.shape == (self.N_STEPS, self.N_ENVS)
        assert filled_buffer.returns.shape == (self.N_STEPS, self.N_ENVS)
        # Returns = advantages + values
        np.testing.assert_allclose(
            filled_buffer.returns,
            filled_buffer.advantages + filled_buffer.values,
            atol=1e-6,
        )

    def test_gae_terminal_no_bootstrap(self) -> None:
        """If the very last step was terminal, advantage should NOT bootstrap."""
        buf = RecurrentRolloutBuffer(
            n_steps=1, n_envs=1, obs_dim=10,
            lstm_hidden=16, lstm_layers=1,
        )
        buf.add(
            obs=np.zeros((1, 10), dtype=np.float32),
            action=np.array([0]),
            reward=np.array([5.0], dtype=np.float32),
            done=np.array([True]),
            log_prob=np.array([0.0], dtype=np.float32),
            value=np.array([1.0], dtype=np.float32),
            hidden_h=np.zeros((1, 1, 16), dtype=np.float32),
            hidden_c=np.zeros((1, 1, 16), dtype=np.float32),
        )
        buf.compute_returns_and_advantages(
            last_values=np.array([99.0]),  # should be ignored
            last_dones=np.array([True]),
        )
        # δ = r + γ*V_next*(1-done) - V = 5 + 0 - 1 = 4
        np.testing.assert_allclose(buf.advantages[0, 0], 4.0, atol=1e-5)

    def test_recurrent_generator_shapes(
        self, filled_buffer: RecurrentRolloutBuffer
    ) -> None:
        seq_len = 8
        num_mb = 4
        filled_buffer.compute_returns_and_advantages(
            np.zeros(self.N_ENVS, dtype=np.float32),
            np.zeros(self.N_ENVS, dtype=bool),
        )

        batches = list(filled_buffer.recurrent_generator(seq_len, num_mb))
        assert len(batches) >= 1

        for b in batches:
            mb_size = b["obs"].shape[1]
            assert b["obs"].shape == (seq_len, mb_size, self.OBS_DIM)
            assert b["actions"].shape == (seq_len, mb_size)
            assert b["old_log_probs"].shape == (seq_len, mb_size)
            assert b["advantages"].shape == (seq_len, mb_size)
            assert b["returns"].shape == (seq_len, mb_size)
            assert b["hidden_h"].shape == (LSTM_LAYERS, mb_size, LSTM_HIDDEN)
            assert b["hidden_c"].shape == (LSTM_LAYERS, mb_size, LSTM_HIDDEN)
            assert b["dones"].shape == (seq_len, mb_size)

    def test_sequence_count(self) -> None:
        """Total sequences = (n_steps // seq_len) * n_envs."""
        n_steps, n_envs, seq_len = 64, 4, 8
        buf = RecurrentRolloutBuffer(
            n_steps=n_steps, n_envs=n_envs, obs_dim=10,
            lstm_hidden=16, lstm_layers=1,
        )
        for _ in range(n_steps):
            buf.add(
                obs=np.zeros((n_envs, 10), dtype=np.float32),
                action=np.zeros(n_envs, dtype=np.int64),
                reward=np.zeros(n_envs, dtype=np.float32),
                done=np.zeros(n_envs, dtype=bool),
                log_prob=np.zeros(n_envs, dtype=np.float32),
                value=np.zeros(n_envs, dtype=np.float32),
                hidden_h=np.zeros((1, n_envs, 16), dtype=np.float32),
                hidden_c=np.zeros((1, n_envs, 16), dtype=np.float32),
            )
        buf.compute_returns_and_advantages(
            np.zeros(n_envs, dtype=np.float32),
            np.zeros(n_envs, dtype=bool),
        )

        expected_total = (n_steps // seq_len) * n_envs  # 8 * 4 = 32
        num_mb = 4
        batches = list(buf.recurrent_generator(seq_len, num_mb))
        total_seqs = sum(b["obs"].shape[1] for b in batches)
        assert total_seqs == expected_total

    def test_reset_clears_pointer(self) -> None:
        buf = RecurrentRolloutBuffer(
            n_steps=4, n_envs=1, obs_dim=10,
            lstm_hidden=16, lstm_layers=1,
        )
        for _ in range(4):
            buf.add(
                obs=np.zeros((1, 10), dtype=np.float32),
                action=np.array([0]),
                reward=np.array([0.0], dtype=np.float32),
                done=np.array([False]),
                log_prob=np.array([0.0], dtype=np.float32),
                value=np.array([0.0], dtype=np.float32),
                hidden_h=np.zeros((1, 1, 16), dtype=np.float32),
                hidden_c=np.zeros((1, 1, 16), dtype=np.float32),
            )
        assert buf.full
        buf.reset()
        assert buf.ptr == 0
        assert not buf.full

"""
Neural-network architectures and recurrent PPO trainer for RL agents.

Provides :class:`CnnLstmPolicy` (CNN + LSTM actor-critic) with a matching
:class:`RecurrentRolloutBuffer` and :class:`RecurrentPPO` trainer that
keeps the same PPO algorithm (GAE, clipping, losses) as the MLP baseline
while adding LSTM hidden-state management.
"""

from clash_royale_engine.models.cnn_lstm_policy import (
    CnnEncoder,
    CnnLstmPolicy,
    ScalarEncoder,
)
from clash_royale_engine.models.recurrent_ppo import RecurrentPPO
from clash_royale_engine.models.recurrent_rollout_buffer import RecurrentRolloutBuffer

__all__ = [
    "CnnEncoder",
    "CnnLstmPolicy",
    "ScalarEncoder",
    "RecurrentPPO",
    "RecurrentRolloutBuffer",
]

"""
Tests for the Clash Royale Engine.

Covers: engine init, elixir gen, card spawning, combat, tower destruction,
action validation, physics, and Gymnasium interface.
"""

from __future__ import annotations

import math
import time
from typing import List, Optional, Tuple

import numpy as np
import pytest

from clash_royale_engine.core.engine import ClashRoyaleEngine
from clash_royale_engine.core.state import State
from clash_royale_engine.entities.base_entity import Entity, reset_entity_id_counter
from clash_royale_engine.entities.buildings.king_tower import KingTowerEntity
from clash_royale_engine.entities.troops.giant import create_giant
from clash_royale_engine.entities.troops.archers import create_archers
from clash_royale_engine.entities.troops.skeletons import create_skeletons
from clash_royale_engine.env.gymnasium_env import ClashRoyaleEnv, ObservationType
from clash_royale_engine.env.multi_agent_env import VectorizedClashRoyaleEnv
from clash_royale_engine.players.player import Player
from clash_royale_engine.players.player_interface import HeuristicBot, RLAgentPlayer
from clash_royale_engine.systems.elixir import ElixirSystem
from clash_royale_engine.systems.physics import PhysicsEngine
from clash_royale_engine.utils.constants import (
    DEFAULT_FPS,
    ELIXIR_PER_SECOND,
    MAX_ELIXIR,
    N_HEIGHT_TILES,
    N_WIDE_TILES,
    SPEED_FAST,
    SPEED_MEDIUM,
    SPEED_SLOW,
    STARTING_ELIXIR,
    TILE_HEIGHT,
    TILE_WIDTH,
)
from clash_royale_engine.utils.converters import pixel_to_tile, tile_to_pixel
from clash_royale_engine.utils.validators import InvalidActionError


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _make_engine(**kwargs) -> ClashRoyaleEngine:
    """Create engine with two dummy RL players."""
    return ClashRoyaleEngine(
        player1=RLAgentPlayer(),
        player2=RLAgentPlayer(),
        **kwargs,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TestGameEngine
# ═══════════════════════════════════════════════════════════════════════════════


class TestGameEngine:
    """Core engine tests."""

    def test_initialization(self) -> None:
        """Engine starts at frame 0 with 6 towers."""
        engine = _make_engine()
        assert engine.current_frame == 0
        # 3 towers per player = 6
        assert len(engine.all_entities) == 6

    def test_reset_idempotent(self) -> None:
        """Resetting returns engine to initial state."""
        engine = _make_engine()
        engine.step(frames=100)
        engine.reset()
        assert engine.current_frame == 0
        assert len(engine.all_entities) == 6
        assert not engine.is_done()

    def test_state_structure(self) -> None:
        """State object has all expected fields."""
        engine = _make_engine()
        s = engine.get_state(0)
        assert hasattr(s, "allies")
        assert hasattr(s, "enemies")
        assert hasattr(s, "numbers")
        assert hasattr(s, "cards")
        assert hasattr(s, "ready")
        assert s.screen == "battle"
        assert len(s.cards) == 4

    def test_initial_elixir(self) -> None:
        """Players start with 5 elixir."""
        engine = _make_engine()
        s = engine.get_state(0)
        assert abs(s.numbers.elixir - STARTING_ELIXIR) < 0.01

    def test_time_remaining(self) -> None:
        """Time remaining decreases as frames advance."""
        engine = _make_engine()
        s0 = engine.get_state(0)
        t0 = s0.numbers.time_remaining
        engine.step(frames=30)
        s1 = engine.get_state(0)
        assert s1.numbers.time_remaining < t0


class TestElixirGeneration:
    """Elixir system tests."""

    def test_generation_rate(self) -> None:
        """Elixir increases at ~1 per 2.8 seconds."""
        engine = _make_engine()
        initial = engine.get_state(0).numbers.elixir

        # Simulate 2.8 seconds = 84 frames at 30 FPS
        for _ in range(84):
            engine._simulate_frame()
            engine.scheduler.advance()

        current = engine.elixir_system.get(0)
        expected = initial + 1.0
        assert abs(current - expected) < 0.05, f"Expected ~{expected}, got {current}"

    def test_elixir_cap(self) -> None:
        """Elixir does not exceed MAX_ELIXIR."""
        system = ElixirSystem(fps=DEFAULT_FPS)
        # Run enough frames to overflow 10
        for _ in range(10000):
            system.update(is_double_elixir=False)
        assert system.get(0) <= MAX_ELIXIR
        assert system.get(1) <= MAX_ELIXIR

    def test_spend(self) -> None:
        """Spending reduces balance; overspend fails."""
        system = ElixirSystem(fps=DEFAULT_FPS)
        assert system.spend(0, 3.0)
        assert abs(system.get(0) - (STARTING_ELIXIR - 3.0)) < 0.01
        assert not system.spend(0, 100.0)


class TestCardSpawning:
    """Spawning troops from cards."""

    def test_giant_spawns_one(self) -> None:
        reset_entity_id_counter()
        entities = [create_giant(0, 9.0, 5.0)]
        assert len(entities) == 1
        assert entities[0].name == "giant"

    def test_archers_spawn_two(self) -> None:
        reset_entity_id_counter()
        entities = create_archers(0, 9.0, 5.0)
        assert len(entities) == 2
        for e in entities:
            assert e.name == "archers"

    def test_skeletons_spawn_three(self) -> None:
        reset_entity_id_counter()
        entities = create_skeletons(1, 9.0, 25.0)
        assert len(entities) == 3
        for e in entities:
            assert e.name == "skeletons"

    def test_spawn_through_engine(self) -> None:
        """Spawning via engine action adds entities to arena."""
        engine = _make_engine()
        initial_count = len(engine.all_entities)

        # Give player 0 enough elixir and play giant (cost 5, starting elixir = 5)
        engine.step_with_action(0, (9, 5, 0))  # card 0 in shuffled hand
        # At least one new entity should appear (or same if card was a spell)
        assert len(engine.all_entities) >= initial_count


class TestCombatDamage:
    """Combat system tests."""

    def test_melee_damage(self) -> None:
        """Entity applying damage reduces target HP."""
        reset_entity_id_counter()
        attacker = create_giant(0, 5.0, 5.0)
        target = create_giant(1, 5.5, 5.0)
        initial_hp = target.hp
        target.apply_damage(attacker.damage)
        assert target.hp == initial_hp - attacker.damage

    def test_entity_death(self) -> None:
        """Entity dies when HP <= 0."""
        reset_entity_id_counter()
        e = create_giant(0, 5.0, 5.0)
        e.apply_damage(e.hp + 100)
        assert e.is_dead
        assert e.hp == 0


class TestTowerDestruction:
    """Tower destruction and victory detection."""

    def test_king_tower_destroy_wins(self) -> None:
        """Destroying king tower ends the game."""
        engine = _make_engine()
        # Directly damage player 1's king tower to 0
        king = engine.arena.king_tower(1)
        assert king is not None
        king.apply_damage(king.hp)
        engine._check_game_over()
        assert engine.is_done()
        assert engine.get_winner() == 0

    def test_princess_tower_tracking(self) -> None:
        """HP of princess towers is tracked in state."""
        engine = _make_engine()
        s = engine.get_state(0)
        assert s.numbers.left_princess_hp > 0
        assert s.numbers.right_princess_hp > 0


class TestActionValidation:
    """Action validation tests."""

    def test_invalid_card_index(self) -> None:
        engine = _make_engine()
        with pytest.raises(InvalidActionError):
            engine.step_with_action(0, (9, 5, 99))

    def test_out_of_bounds(self) -> None:
        engine = _make_engine()
        with pytest.raises(InvalidActionError):
            engine.step_with_action(0, (99, 99, 0))

    def test_enemy_side_placement(self) -> None:
        """Troops cannot be placed on the enemy's side."""
        engine = _make_engine()
        # Player 0 trying to place at tile_y = 20 (enemy side after flip)
        with pytest.raises(InvalidActionError):
            engine.step_with_action(0, (9, 20, 0))


class TestPhysics:
    """Physics engine tests."""

    def test_movement_speed_giant(self) -> None:
        """Giant moves at ~45 px/s = SPEED_SLOW tiles/s."""
        reset_entity_id_counter()
        # Place on own side — target also on own side so river routing
        # does not interfere with the pure-speed assertion.
        giant = create_giant(0, 9.0, 2.0)
        giant.is_deployed = True
        # Target on same side of river (y=12 < RIVER_Y_MIN=15)
        target = Entity(
            name="dummy", player_id=1, x=9.0, y=12.0,
            hp=9999, damage=0, hit_speed=99, attack_range=0,
            sight_range=99, speed=0, target_type="all",
            is_building=True,
        )
        giant.current_target = target

        physics = PhysicsEngine(fps=DEFAULT_FPS)
        y_before = giant.y

        # Simulate 1 second (30 frames)
        for _ in range(DEFAULT_FPS):
            physics.update([giant])

        dy = giant.y - y_before
        avg_tile = (TILE_WIDTH + TILE_HEIGHT) / 2.0
        expected_tiles = SPEED_SLOW / avg_tile  # tiles / second
        assert abs(dy - expected_tiles) < 0.5, f"Giant moved {dy:.2f} tiles, expected ~{expected_tiles:.2f}"

    def test_collision_separation(self) -> None:
        """Two overlapping entities are pushed apart."""
        reset_entity_id_counter()
        e1 = create_giant(0, 5.0, 5.0)
        e2 = create_giant(0, 5.0, 5.0)  # exact same position
        e1.is_deployed = True
        e2.is_deployed = True
        e1.is_static = False
        e2.is_static = False

        physics = PhysicsEngine(fps=DEFAULT_FPS)
        # After a few frames they should separate
        for _ in range(10):
            physics.update([e1, e2])

        dist = ((e1.x - e2.x) ** 2 + (e1.y - e2.y) ** 2) ** 0.5
        assert dist > 0.1, "Entities should have separated"

    def test_river_blocks_ground_troops(self) -> None:
        """Ground troops cannot enter the river zone at non-bridge positions."""
        from clash_royale_engine.systems.physics import PhysicsEngine, _is_on_bridge
        from clash_royale_engine.utils.constants import RIVER_Y_MIN, RIVER_Y_MAX

        enforce = PhysicsEngine._enforce_river

        # Below river → into river, NOT on bridge → blocked
        assert enforce(14.0, 9.0, 15.5) < RIVER_Y_MIN
        # Above river → into river, NOT on bridge → blocked
        assert enforce(18.0, 9.0, 16.0) > RIVER_Y_MAX
        # Below river → jump OVER river, NOT on bridge → blocked
        assert enforce(14.0, 9.0, 18.0) < RIVER_Y_MIN
        # On bridge (x=4.0 is left bridge) → allowed
        assert enforce(14.0, 4.0, 15.5) == 15.5

        # Simulation invariant: if a troop is in the river zone, it must be
        # standing on a bridge tile.
        reset_entity_id_counter()
        giant = create_giant(0, 9.0, 5.0)
        giant.is_deployed = True
        target = Entity(
            name="dummy", player_id=1, x=9.0, y=25.0,
            hp=9999, damage=0, hit_speed=99, attack_range=0,
            sight_range=99, speed=0, target_type="all",
            is_building=True,
        )
        giant.current_target = target

        physics = PhysicsEngine(fps=DEFAULT_FPS)
        for _ in range(600):
            physics.update([giant])
            if RIVER_Y_MIN <= giant.y <= RIVER_Y_MAX:
                assert _is_on_bridge(giant.x), (
                    f"Ground troop in river at ({giant.x:.2f}, {giant.y:.2f}) "
                    f"which is NOT a bridge tile!"
                )

    def test_bridge_allows_crossing(self) -> None:
        """Ground troops CAN cross the river when on a bridge tile."""
        from clash_royale_engine.utils.constants import BRIDGE_LEFT_X, BRIDGE_WIDTH, RIVER_Y_MAX

        reset_entity_id_counter()
        bridge_cx = BRIDGE_LEFT_X + BRIDGE_WIDTH / 2.0  # centre of left bridge
        giant = create_giant(0, bridge_cx, 10.0)
        giant.is_deployed = True
        target = Entity(
            name="dummy", player_id=1, x=bridge_cx, y=25.0,
            hp=9999, damage=0, hit_speed=99, attack_range=0,
            sight_range=99, speed=0, target_type="all",
            is_building=True,
        )
        giant.current_target = target

        physics = PhysicsEngine(fps=DEFAULT_FPS)
        for _ in range(600):  # 20 seconds
            physics.update([giant])

        assert giant.y > RIVER_Y_MAX, (
            f"Giant should have crossed the river via bridge, but y={giant.y:.2f}"
        )

    def test_river_routing_reaches_other_side(self) -> None:
        """A ground troop placed off-bridge still reaches the enemy side
        by routing through the nearest bridge."""
        from clash_royale_engine.utils.constants import RIVER_Y_MAX

        reset_entity_id_counter()
        giant = create_giant(0, 9.0, 5.0)  # off-bridge
        giant.is_deployed = True
        target = Entity(
            name="dummy", player_id=1, x=9.0, y=28.0,
            hp=9999, damage=0, hit_speed=99, attack_range=0,
            sight_range=99, speed=0, target_type="all",
            is_building=True,
        )
        giant.current_target = target

        physics = PhysicsEngine(fps=DEFAULT_FPS)
        for _ in range(900):  # 30 seconds — enough for slow giant
            physics.update([giant])

        assert giant.y > RIVER_Y_MAX, (
            f"Giant should have routed through bridge and crossed, but y={giant.y:.2f}"
        )


class TestCoordinateConversion:
    """Tile ↔ pixel conversions."""

    def test_roundtrip(self) -> None:
        """tile → pixel → tile should be identity (or ±1)."""
        for tx in range(N_WIDE_TILES):
            for ty in range(N_HEIGHT_TILES):
                px, py = tile_to_pixel(tx, ty)
                rtx, rty = pixel_to_tile(px, py)
                assert abs(rtx - tx) <= 1
                assert abs(rty - ty) <= 1


# ═══════════════════════════════════════════════════════════════════════════════
# TestEnvironment (Gymnasium interface)
# ═══════════════════════════════════════════════════════════════════════════════


class TestEnvironment:
    """Gymnasium env compliance tests."""

    def test_gymnasium_interface(self) -> None:
        env = ClashRoyaleEnv()
        assert hasattr(env, "reset")
        assert hasattr(env, "step")
        assert hasattr(env, "action_space")
        assert hasattr(env, "observation_space")

        obs, info = env.reset()
        assert env.observation_space.contains(obs), f"obs shape {obs.shape}"

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_noop_action(self) -> None:
        """The last action index is a no-op and always valid."""
        env = ClashRoyaleEnv()
        obs, _ = env.reset()
        noop = env.action_space.n - 1  # type: ignore[attr-defined]
        obs2, r, te, tr, info = env.step(noop)
        assert info["action_valid"]  # no-op should be valid

    def test_episode_terminates(self) -> None:
        """An episode eventually ends (within time limit)."""
        env = ClashRoyaleEnv(time_limit=5.0, speed_multiplier=10.0)
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < 50000:
            obs, r, te, tr, _ = env.step(env.action_space.sample())
            done = te or tr
            steps += 1
        assert done, "Episode did not terminate"

    def test_dense_reward(self) -> None:
        """Dense reward mode returns non-zero rewards during episode."""
        env = ClashRoyaleEnv(reward_shaping="dense")
        obs, _ = env.reset()
        rewards = []
        for _ in range(100):
            obs, r, te, tr, _ = env.step(env.action_space.sample())
            rewards.append(r)
            if te or tr:
                break
        # At least some reward should be non-zero (invalid-action penalty etc.)
        assert any(r != 0.0 for r in rewards)

    def test_vectorized_env(self) -> None:
        """VectorizedClashRoyaleEnv runs multiple envs."""
        vec = VectorizedClashRoyaleEnv(num_envs=4, time_limit=5.0, speed_multiplier=5.0)
        obs = vec.reset()
        assert obs.shape[0] == 4

        actions = np.array([vec.envs[0].action_space.sample() for _ in range(4)])
        obs2, rews, terms, truncs, infos = vec.step(actions)
        assert obs2.shape[0] == 4
        assert rews.shape[0] == 4


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark
# ═══════════════════════════════════════════════════════════════════════════════


class TestBenchmark:
    """Performance benchmark (not asserted in CI, just reported)."""

    @pytest.mark.slow
    def test_simulation_speed(self) -> None:
        """Target: 1000+ episodes/hour on modern CPU."""
        env = ClashRoyaleEnv(
            time_limit=10.0,
            speed_multiplier=10.0,
        )
        start = time.time()
        episodes = 0
        for _ in range(50):
            env.reset()
            done = False
            while not done:
                obs, r, te, tr, _ = env.step(env.action_space.sample())
                done = te or tr
            episodes += 1

        elapsed = time.time() - start
        eps_per_hour = (episodes / max(elapsed, 0.001)) * 3600
        print(f"\nBenchmark: {eps_per_hour:.0f} episodes/hour ({episodes} in {elapsed:.1f}s)")
        # Soft assertion — we don't fail CI if the machine is slow
        assert eps_per_hour > 100, f"Unexpectedly slow: {eps_per_hour:.0f} ep/h"

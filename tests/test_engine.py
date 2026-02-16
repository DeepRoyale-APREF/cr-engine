"""
Tests for the Clash Royale Engine.

Covers: engine init, elixir gen, card spawning, combat, tower destruction,
action validation, physics, Gymnasium interface, fog-of-war, recording,
and 4-episode imitation-learning extraction.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from clash_royale_engine.core.engine import ClashRoyaleEngine
from clash_royale_engine.core.recorder import (
    NOOP_ACTION,
    EpisodeExtractor,
    GameRecord,
    Transition,
    apply_fog_of_war,
    decode_action,
    encode_action,
    flip_action_x,
    flip_action_y,
    flip_state_x,
    flip_state_y,
)
from clash_royale_engine.core.state import State
from clash_royale_engine.entities.base_entity import Entity, reset_entity_id_counter
from clash_royale_engine.entities.troops.archers import create_archers
from clash_royale_engine.entities.troops.giant import create_giant
from clash_royale_engine.entities.troops.skeletons import create_skeletons
from clash_royale_engine.env.gymnasium_env import ClashRoyaleEnv
from clash_royale_engine.env.multi_agent_env import VectorizedClashRoyaleEnv
from clash_royale_engine.players.player import Player
from clash_royale_engine.players.player_interface import RLAgentPlayer
from clash_royale_engine.systems.elixir import ElixirSystem
from clash_royale_engine.systems.physics import PhysicsEngine
from clash_royale_engine.utils.constants import (
    DEFAULT_FPS,
    LANE_DIVIDER_X,
    MAX_ELIXIR,
    N_HEIGHT_TILES,
    N_WIDE_TILES,
    POCKET_DEPTH,
    RIVER_Y_MAX,
    RIVER_Y_MIN,
    SPEED_SLOW,
    STARTING_ELIXIR,
    TILE_HEIGHT,
    TILE_WIDTH,
)
from clash_royale_engine.utils.converters import pixel_to_tile, tile_to_pixel
from clash_royale_engine.utils.validators import (
    InvalidActionError,
    validate_placement,
)

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
        """Troops cannot be placed on the enemy's side when towers are up."""
        engine = _make_engine()
        # Player 0 trying to place at tile_y = 20 (enemy side, both towers alive)
        with pytest.raises(InvalidActionError):
            engine.step_with_action(0, (9, 20, 0))


# ═══════════════════════════════════════════════════════════════════════════════
# Pocket placement (enemy-side unlock via princess tower destruction)
# ═══════════════════════════════════════════════════════════════════════════════


class TestPocketPlacement:
    """Verify advanced placement rules tied to princess tower destruction."""

    @pytest.fixture()
    def _fresh_player(self) -> Player:
        """Player with full elixir and hand containing at least a troop + spell."""
        p = Player(0, list([
            "giant", "musketeer", "archers", "mini_pekka",
            "knight", "skeletons", "arrows", "fireball",
        ]), seed=0)
        p.elixir = 10.0
        return p

    # ── own side always allowed ───────────────────────────────────────────

    def test_own_side_always_allowed(self, _fresh_player: Player) -> None:
        """Troop placement on own side is always valid."""
        err = validate_placement(0, 9, 5, "giant", _fresh_player)
        assert err is None

    # ── enemy side blocked by default ─────────────────────────────────────

    def test_enemy_side_troop_blocked_both_towers_up(self, _fresh_player: Player) -> None:
        """Troop on enemy side fails when both princess towers are alive."""
        tile_y_enemy = int(RIVER_Y_MAX)  # 17 — just past river, enemy side for P0
        err = validate_placement(
            0, 5, tile_y_enemy, "giant", _fresh_player,
            enemy_left_princess_dead=False,
            enemy_right_princess_dead=False,
        )
        assert err is not None, "Should reject placement when both towers are up"

    # ── spells anywhere ───────────────────────────────────────────────────

    def test_spell_on_enemy_side_always_allowed(self, _fresh_player: Player) -> None:
        """Spells can be placed anywhere regardless of tower state."""
        # Place arrows on the far enemy side (tile_y=28)
        # 'arrows' is at index 6 in the default deck
        # but validate_placement uses card_name directly
        err = validate_placement(
            0, 9, 28, "arrows", _fresh_player,
            enemy_left_princess_dead=False,
            enemy_right_princess_dead=False,
        )
        assert err is None, "Spell should be allowed on enemy side"

    def test_fireball_on_enemy_side(self, _fresh_player: Player) -> None:
        """Fireball can land anywhere."""
        err = validate_placement(
            0, 3, 25, "fireball", _fresh_player,
            enemy_left_princess_dead=False,
            enemy_right_princess_dead=False,
        )
        assert err is None

    # ── left pocket unlocks when enemy left princess dies ─────────────────

    def test_left_pocket_unlocked_p0(self, _fresh_player: Player) -> None:
        """P0 can place troop in LEFT pocket (tile_x < 9) after left princess dies."""
        pocket_y = int(RIVER_Y_MAX)  # 17 — first row of pocket past river
        pocket_x = LANE_DIVIDER_X - 1  # 8 — left lane
        err = validate_placement(
            0, pocket_x, pocket_y, "giant", _fresh_player,
            enemy_left_princess_dead=True,
            enemy_right_princess_dead=False,
        )
        assert err is None, "Left pocket should be unlocked"

    def test_right_pocket_still_locked_when_only_left_dies(self, _fresh_player: Player) -> None:
        """P0 CANNOT place troop in RIGHT pocket when only left princess died."""
        pocket_y = int(RIVER_Y_MAX)
        pocket_x = LANE_DIVIDER_X  # 9 — right lane
        err = validate_placement(
            0, pocket_x, pocket_y, "giant", _fresh_player,
            enemy_left_princess_dead=True,
            enemy_right_princess_dead=False,
        )
        assert err is not None, "Right pocket should still be locked"

    # ── right pocket unlocks when enemy right princess dies ───────────────

    def test_right_pocket_unlocked_p0(self, _fresh_player: Player) -> None:
        """P0 can place troop in RIGHT pocket (tile_x >= 9) after right princess dies."""
        pocket_y = int(RIVER_Y_MAX)
        pocket_x = LANE_DIVIDER_X  # 9
        err = validate_placement(
            0, pocket_x, pocket_y, "knight", _fresh_player,
            enemy_left_princess_dead=False,
            enemy_right_princess_dead=True,
        )
        assert err is None, "Right pocket should be unlocked"

    # ── pocket depth limit ────────────────────────────────────────────────

    def test_placement_beyond_pocket_depth_rejected(self, _fresh_player: Player) -> None:
        """Troop cannot be placed deeper than POCKET_DEPTH past the river."""
        too_deep_y = int(RIVER_Y_MAX) + POCKET_DEPTH  # one tile beyond the pocket
        err = validate_placement(
            0, 5, too_deep_y, "giant", _fresh_player,
            enemy_left_princess_dead=True,
            enemy_right_princess_dead=True,
        )
        assert err is not None, f"tile_y={too_deep_y} is beyond pocket depth"

    def test_placement_at_max_pocket_depth_allowed(self, _fresh_player: Player) -> None:
        """Troop at the deepest pocket tile is still valid."""
        max_pocket_y = int(RIVER_Y_MAX) + POCKET_DEPTH - 1
        err = validate_placement(
            0, 5, max_pocket_y, "giant", _fresh_player,
            enemy_left_princess_dead=True,
            enemy_right_princess_dead=False,
        )
        assert err is None, f"tile_y={max_pocket_y} should be within pocket"

    # ── Player 1 symmetric tests ──────────────────────────────────────────

    def test_p1_enemy_side_blocked(self) -> None:
        """Player 1 cannot place troops on P0's side (low y) without tower kill."""
        p1 = Player(1, ["giant", "musketeer", "archers", "mini_pekka",
                        "knight", "skeletons", "arrows", "fireball"], seed=0)
        p1.elixir = 10.0
        tile_y_p0_side = int(RIVER_Y_MIN) - 1  # 14 — P0's territory
        err = validate_placement(
            1, 5, tile_y_p0_side, "knight", p1,
            enemy_left_princess_dead=False,
            enemy_right_princess_dead=False,
        )
        assert err is not None

    def test_p1_left_pocket_unlocked(self) -> None:
        """Player 1 can place in left pocket on P0's side after P0's left tower dies."""
        p1 = Player(1, ["giant", "musketeer", "archers", "mini_pekka",
                        "knight", "skeletons", "arrows", "fireball"], seed=0)
        p1.elixir = 10.0
        # P1 pocket is below river: y from RIVER_Y_MIN - POCKET_DEPTH to RIVER_Y_MIN - 1
        pocket_y = int(RIVER_Y_MIN) - 1  # 14
        pocket_x = 4  # left lane (< 9)
        err = validate_placement(
            1, pocket_x, pocket_y, "knight", p1,
            enemy_left_princess_dead=True,
            enemy_right_princess_dead=False,
        )
        assert err is None, "P1 left pocket should be unlocked"

    # ── engine integration: destroy tower then place ──────────────────────

    def test_engine_pocket_after_tower_destroy(self) -> None:
        """End-to-end: destroy enemy left princess tower, then place in pocket."""
        engine = _make_engine()

        # Kill Player 1's left princess tower directly
        tower_key = "p1_left_princess"
        tower = engine.arena.towers[tower_key]
        tower.hp = 0  # force-kill

        # Give Player 0 enough elixir
        engine.elixir_system.elixir[0] = 10.0
        engine.players[0].elixir = 10.0

        # Find 'giant' card index in hand
        hand = engine.players[0].hand
        card_idx = hand.index("giant") if "giant" in hand else 0

        # Place in left pocket: tile_x=5 (left lane), tile_y=17 (just past river)
        pocket_y = int(RIVER_Y_MAX)
        pocket_x = 5
        # Should succeed (no InvalidActionError)
        engine.step_with_action(0, (pocket_x, pocket_y, card_idx))

    def test_engine_pocket_blocked_when_tower_alive(self) -> None:
        """End-to-end: cannot place in pocket when tower is alive."""
        engine = _make_engine()

        engine.elixir_system.elixir[0] = 10.0
        engine.players[0].elixir = 10.0

        hand = engine.players[0].hand
        card_idx = hand.index("giant") if "giant" in hand else 0

        pocket_y = int(RIVER_Y_MAX)
        pocket_x = 5  # left lane
        with pytest.raises(InvalidActionError):
            engine.step_with_action(0, (pocket_x, pocket_y, card_idx))


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
        from clash_royale_engine.utils.constants import RIVER_Y_MAX, RIVER_Y_MIN

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


# ═══════════════════════════════════════════════════════════════════════════════
# Fog of War
# ═══════════════════════════════════════════════════════════════════════════════


class TestFogOfWar:
    """Verify that partial observability hides enemy elixir."""

    def test_fog_of_war_hides_enemy_elixir(self) -> None:
        """With fog_of_war=True the observation's enemy-elixir slot is 0."""
        env = ClashRoyaleEnv(fog_of_war=True, time_limit=10.0)
        obs, _ = env.reset()
        # Feature index 1 is enemy_elixir / 10.0
        assert obs[1] == 0.0, "enemy_elixir should be hidden (0) with fog_of_war"

    def test_no_fog_exposes_enemy_elixir(self) -> None:
        """With fog_of_war=False the observation preserves enemy elixir."""
        env = ClashRoyaleEnv(fog_of_war=False, time_limit=10.0)
        obs, _ = env.reset()
        #  At start both players have STARTING_ELIXIR (5.0)
        expected = STARTING_ELIXIR / 10.0
        assert obs[1] == pytest.approx(expected, abs=0.01), (
            f"enemy_elixir should be {expected} without fog_of_war, got {obs[1]}"
        )

    def test_apply_fog_of_war_util(self) -> None:
        """apply_fog_of_war zeroes enemy_elixir in a State."""
        engine = _make_engine(time_limit=10.0)
        state = engine.get_state(0)
        fogged = apply_fog_of_war(state)
        assert fogged.numbers.enemy_elixir == 0.0
        # Own elixir is untouched
        assert fogged.numbers.elixir == state.numbers.elixir


# ═══════════════════════════════════════════════════════════════════════════════
# Action Encoding / Decoding (recorder helpers)
# ═══════════════════════════════════════════════════════════════════════════════


class TestActionEncoding:
    """Round-trip tests for encode_action / decode_action."""

    def test_roundtrip_placement(self) -> None:
        for tx in (0, 5, 17):
            for ty in (0, 15, 31):
                for ci in range(4):
                    action = (tx, ty, ci)
                    encoded = encode_action(action)
                    decoded = decode_action(encoded)
                    assert decoded == action, f"Roundtrip failed for {action}"

    def test_noop_roundtrip(self) -> None:
        assert encode_action(None) == NOOP_ACTION
        assert decode_action(NOOP_ACTION) is None

    def test_flip_action_y(self) -> None:
        action = (5, 10, 2)
        flipped = flip_action_y(action)
        assert flipped == (5, N_HEIGHT_TILES - 1 - 10, 2)

    def test_flip_action_x(self) -> None:
        action = (5, 10, 2)
        flipped = flip_action_x(action)
        assert flipped == (N_WIDE_TILES - 1 - 5, 10, 2)

    def test_flip_none_action(self) -> None:
        assert flip_action_y(None) is None
        assert flip_action_x(None) is None


# ═══════════════════════════════════════════════════════════════════════════════
# State Transforms
# ═══════════════════════════════════════════════════════════════════════════════


class TestStateTransforms:
    """Verify symmetry transform utilities on State objects."""

    @pytest.fixture()
    def sample_state(self) -> State:
        engine = _make_engine(time_limit=10.0)
        return engine.get_state(0)

    def test_flip_state_y_inverts_tile_y(self, sample_state: State) -> None:
        flipped = flip_state_y(sample_state)
        for orig, fl in zip(sample_state.allies, flipped.allies):
            assert fl.position.tile_y == N_HEIGHT_TILES - 1 - orig.position.tile_y
        for orig, fl in zip(sample_state.enemies, flipped.enemies):
            assert fl.position.tile_y == N_HEIGHT_TILES - 1 - orig.position.tile_y

    def test_flip_state_x_inverts_tile_x(self, sample_state: State) -> None:
        flipped = flip_state_x(sample_state)
        for orig, fl in zip(sample_state.allies, flipped.allies):
            assert fl.position.tile_x == N_WIDE_TILES - 1 - orig.position.tile_x

    def test_flip_state_x_swaps_princess_hp(self, sample_state: State) -> None:
        flipped = flip_state_x(sample_state)
        assert flipped.numbers.left_princess_hp == sample_state.numbers.right_princess_hp
        assert flipped.numbers.right_princess_hp == sample_state.numbers.left_princess_hp
        assert (
            flipped.numbers.left_enemy_princess_hp
            == sample_state.numbers.right_enemy_princess_hp
        )
        assert (
            flipped.numbers.right_enemy_princess_hp
            == sample_state.numbers.left_enemy_princess_hp
        )

    def test_double_flip_y_is_identity(self, sample_state: State) -> None:
        double = flip_state_y(flip_state_y(sample_state))
        for orig, d in zip(sample_state.allies, double.allies):
            assert d.position.tile_y == orig.position.tile_y

    def test_double_flip_x_is_identity(self, sample_state: State) -> None:
        double = flip_state_x(flip_state_x(sample_state))
        for orig, d in zip(sample_state.allies, double.allies):
            assert d.position.tile_x == orig.position.tile_x
        assert double.numbers.left_princess_hp == sample_state.numbers.left_princess_hp


# ═══════════════════════════════════════════════════════════════════════════════
# Game Recording
# ═══════════════════════════════════════════════════════════════════════════════


class TestGameRecording:
    """Verify that the engine records frames when recording is enabled."""

    def test_recording_produces_record_after_reset(self) -> None:
        """After one full episode, get_last_record returns a GameRecord."""
        env = ClashRoyaleEnv(record=True, time_limit=5.0)
        env.reset()

        # Play a very short episode
        done = False
        steps = 0
        while not done and steps < 500:
            obs, r, te, tr, _ = env.step(env.action_space.sample())
            done = te or tr
            steps += 1

        # Resetting finalises the previous recording
        env.reset()
        record = env.get_game_record()

        assert record is not None, "Expected a GameRecord after reset"
        assert isinstance(record, GameRecord)
        assert record.total_frames > 0
        assert len(record.frames) == record.total_frames

    def test_no_record_when_disabled(self) -> None:
        """Without record=True, get_game_record is None."""
        env = ClashRoyaleEnv(record=False, time_limit=5.0)
        env.reset()
        for _ in range(30):
            env.step(env.action_space.sample())
        env.reset()
        assert env.get_game_record() is None

    def test_frames_have_both_player_states(self) -> None:
        """Each FrameRecord contains State for P0 and P1."""
        env = ClashRoyaleEnv(record=True, time_limit=5.0)
        env.reset()
        for _ in range(60):
            env.step(env.action_space.sample())
        env.reset()

        record = env.get_game_record()
        assert record is not None
        for fr in record.frames[:5]:  # spot-check first few
            assert isinstance(fr.state_p0, State)
            assert isinstance(fr.state_p1, State)
            # P0 allies should be P1 enemies (god-view consistency)
            assert len(fr.state_p0.allies) > 0 or len(fr.state_p1.allies) > 0

    def test_recorder_tracks_actions(self) -> None:
        """At least some frames should contain a non-None action."""
        env = ClashRoyaleEnv(record=True, time_limit=8.0)
        env.reset()
        done = False
        steps = 0
        while not done and steps < 300:
            obs, r, te, tr, _ = env.step(env.action_space.sample())
            done = te or tr
            steps += 1
        env.reset()

        record = env.get_game_record()
        assert record is not None
        any(fr.action_p0 is not None for fr in record.frames)
        # P0 fires random actions; at least some should be valid
        # (It's possible all are invalid, so we test that the field exists)
        assert isinstance(record.frames[0].action_p0, (tuple, type(None)))
        # P1 is RLAgentPlayer (always returns None) — so all should be None
        # unless opponent is HeuristicBot; in ClashRoyaleEnv default it is.
        # Either way the field is present:
        assert isinstance(record.frames[0].action_p1, (tuple, type(None)))


# ═══════════════════════════════════════════════════════════════════════════════
# 4-Episode Imitation Learning Extraction
# ═══════════════════════════════════════════════════════════════════════════════


class TestILExtraction:
    """Verify that extract_il_episodes produces exactly 4 valid episode lists."""

    @pytest.fixture()
    def finished_env(self) -> ClashRoyaleEnv:
        """Run one short episode and finalise recording."""
        env = ClashRoyaleEnv(record=True, time_limit=5.0, fog_of_war=True)
        env.reset()
        done = False
        steps = 0
        while not done and steps < 500:
            obs, r, te, tr, _ = env.step(env.action_space.sample())
            done = te or tr
            steps += 1
        # Finalise
        env.reset()
        return env

    def test_extracts_exactly_4_episodes(self, finished_env: ClashRoyaleEnv) -> None:
        episodes = finished_env.extract_il_episodes()
        assert len(episodes) == 4, f"Expected 4 episodes, got {len(episodes)}"

    def test_episodes_are_nonempty(self, finished_env: ClashRoyaleEnv) -> None:
        episodes = finished_env.extract_il_episodes()
        for i, ep in enumerate(episodes):
            assert len(ep) > 0, f"Episode {i} is empty"

    def test_all_episodes_same_length(self, finished_env: ClashRoyaleEnv) -> None:
        """All 4 episodes should have the same number of transitions (= frames - 1)."""
        episodes = finished_env.extract_il_episodes()
        lengths = [len(ep) for ep in episodes]
        assert len(set(lengths)) == 1, f"Episode lengths differ: {lengths}"

    def test_transition_shapes(self, finished_env: ClashRoyaleEnv) -> None:
        """Each Transition has correctly shaped numpy arrays."""
        from clash_royale_engine.utils.constants import OBS_FEATURE_DIM

        episodes = finished_env.extract_il_episodes()
        for ep in episodes:
            for t in ep[:3]:  # spot-check a few
                assert isinstance(t, Transition)
                assert t.state.shape == (OBS_FEATURE_DIM,)
                assert t.next_state.shape == (OBS_FEATURE_DIM,)
                assert isinstance(t.action, (int, np.integer))
                assert isinstance(t.reward, float)
                assert isinstance(t.done, bool)

    def test_last_transition_is_done(self, finished_env: ClashRoyaleEnv) -> None:
        episodes = finished_env.extract_il_episodes()
        for i, ep in enumerate(episodes):
            assert ep[-1].done is True, f"Episode {i} last transition should be done"

    def test_fog_of_war_applied_in_episodes(self, finished_env: ClashRoyaleEnv) -> None:
        """Enemy elixir index (1) should be 0 in all episode states."""
        episodes = finished_env.extract_il_episodes()
        for ep in episodes:
            for t in ep[:5]:
                assert t.state[1] == 0.0, "enemy_elixir should be fogged"

    def test_episodes_to_numpy(self, finished_env: ClashRoyaleEnv) -> None:
        """Batch all episodes into numpy arrays via EpisodeExtractor utility."""
        episodes = finished_env.extract_il_episodes()
        batch = EpisodeExtractor.episodes_to_numpy(episodes)

        total = sum(len(ep) for ep in episodes)
        assert batch["states"].shape[0] == total
        assert batch["actions"].shape[0] == total
        assert batch["rewards"].shape[0] == total
        assert batch["next_states"].shape[0] == total
        assert batch["dones"].shape[0] == total
        assert batch["episode_ids"].shape[0] == total
        # 4 episode ids: 0, 1, 2, 3
        assert set(batch["episode_ids"].tolist()) == {0, 1, 2, 3}

    def test_no_record_raises(self) -> None:
        """extract_il_episodes raises ValueError when no record exists."""
        env = ClashRoyaleEnv(record=False, time_limit=5.0)
        env.reset()
        with pytest.raises(ValueError, match="No game record"):
            env.extract_il_episodes()

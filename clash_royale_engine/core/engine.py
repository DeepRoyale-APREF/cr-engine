"""
ClashRoyaleEngine — the main game engine.

Orchestrates all sub-systems (physics, combat, targeting, elixir) and
exposes a frame-stepping API that is player-agnostic.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from clash_royale_engine.core.arena import Arena
from clash_royale_engine.core.scheduler import Scheduler
from clash_royale_engine.core.state import (
    Card,
    Numbers,
    Position,
    SpellInfo,
    State,
    Unit,
    UnitDetection,
)
from clash_royale_engine.entities.base_entity import Entity, reset_entity_id_counter
from clash_royale_engine.players.player import Player
from clash_royale_engine.players.player_interface import PlayerInterface
from clash_royale_engine.systems.combat import CombatSystem
from clash_royale_engine.systems.elixir import ElixirSystem
from clash_royale_engine.systems.physics import PhysicsEngine
from clash_royale_engine.systems.targeting import TargetingSystem
from clash_royale_engine.utils.constants import (
    CARD_STATS,
    DEFAULT_DECK,
    DEFAULT_FPS,
    DISPLAY_HEIGHT,
    DISPLAY_WIDTH,
    GAME_DURATION,
    N_HEIGHT_TILES,
    TILE_HEIGHT,
    TILE_WIDTH,
)
from clash_royale_engine.utils.converters import tile_to_pixel
from clash_royale_engine.utils.validators import InvalidActionError, validate_action

# Recorder is optional — only imported when used
try:
    from clash_royale_engine.core.recorder import GameRecord, GameRecorder
except ImportError:  # pragma: no cover
    GameRecorder = None  # type: ignore[assignment,misc]
    GameRecord = None  # type: ignore[assignment,misc]


class ClashRoyaleEngine:
    """
    Main game engine — player-agnostic.

    Parameters
    ----------
    player1, player2 : PlayerInterface
        Objects that produce actions each frame.
    deck1, deck2 : list[str]
        Card names for each player's deck.
    fps : int
        Simulation framerate.
    time_limit : float
        Game duration in seconds (before overtime).
    speed_multiplier : float
        Multiplier applied when stepping multiple frames at once.
    """

    def __init__(
        self,
        player1: PlayerInterface,
        player2: PlayerInterface,
        deck1: Optional[List[str]] = None,
        deck2: Optional[List[str]] = None,
        fps: int = DEFAULT_FPS,
        time_limit: float = GAME_DURATION,
        speed_multiplier: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.player1_interface = player1
        self.player2_interface = player2
        self.fps = fps
        self.speed_multiplier = speed_multiplier
        self.seed = seed

        # Sub-systems
        self.scheduler = Scheduler(fps=fps)
        self.scheduler.game_duration = time_limit
        self.arena = Arena(fps=fps)
        self.physics = PhysicsEngine(fps=fps)
        self.combat = CombatSystem(fps=fps)
        self.targeting = TargetingSystem()
        self.elixir_system = ElixirSystem(fps=fps)

        # Player state
        self.players: List[Player] = [
            Player(0, deck1 or list(DEFAULT_DECK), seed=seed),
            Player(1, deck2 or list(DEFAULT_DECK), seed=seed),
        ]

        # Game-over state
        self._done: bool = False
        self._winner: Optional[int] = None  # 0, 1, or None (draw)

        # Crown snapshot taken the first frame overtime begins (for sudden-death detection)
        self._regulation_crowns: Optional[List[int]] = None

        # Previous tower HP (for reward computation)
        self._prev_tower_hp: Dict[str, float] = {}

        # ── Recording (optional) ──────────────────────────────────────────
        self.recorder: Optional["GameRecorder"] = None
        self._last_record: Optional["GameRecord"] = None

        # Perform initial setup
        self.reset()

    # ══════════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════════

    @property
    def current_frame(self) -> int:
        return self.scheduler.current_frame

    @property
    def all_entities(self) -> List[Entity]:
        return self.arena.get_alive_entities()

    def enable_recording(self) -> None:
        """Enable per-frame game recording for IL / replay."""
        if GameRecorder is not None:
            self.recorder = GameRecorder()
        else:
            raise ImportError("GameRecorder not available")

    def get_last_record(self) -> Optional["GameRecord"]:
        """Return the :class:`GameRecord` from the most recent episode."""
        return self._last_record

    def reset(self) -> State:
        """Reset the engine to a fresh game state and return P0's initial State."""
        # Finalise previous recording (if any)
        if self.recorder is not None and len(self.recorder._frames) > 0:
            self._last_record = self.recorder.build_record(
                winner=self._winner,
                deck_p0=self.players[0].deck,
                deck_p1=self.players[1].deck,
            )

        reset_entity_id_counter()
        self.scheduler.reset()
        self.arena.reset()
        self.combat.reset()
        self.elixir_system.reset()
        self._done = False
        self._winner = None
        self._regulation_crowns = None

        for p in self.players:
            p.reset(seed=self.seed)

        # Sync elixir from system → player objects (validators read player.elixir)
        self._sync_elixir()

        self.player1_interface.reset()
        self.player2_interface.reset()

        self._snapshot_tower_hp()

        if self.recorder is not None:
            self.recorder.reset()

        return self._get_state(player_id=0)

    # ── stepping (both players controlled internally) ─────────────────────

    def step(self, frames: int = 1) -> Tuple[State, State, bool]:
        """
        Advance the simulation *frames* frames, querying both player
        interfaces for actions each frame.

        Returns ``(state_p0, state_p1, done)``.
        """
        effective = max(1, int(frames * self.speed_multiplier))
        for _ in range(effective):
            if self._done:
                break
            self._tick_one_frame()
        return (
            self._get_state(0),
            self._get_state(1),
            self._done,
        )

    # ── stepping (action injected externally — used by Gym env) ──────────

    def step_with_action(
        self,
        player_id: int,
        action: Optional[Tuple[int, int, int]],
    ) -> Tuple[State, State, bool]:
        """
        Step one frame. *player_id*'s action is provided; the opponent
        acts through its PlayerInterface.

        Raises :class:`InvalidActionError` if *action* is not valid.
        """
        if self._done:
            return self._get_state(0), self._get_state(1), True

        # Validate & apply the provided action
        valid_own: Optional[Tuple[int, int, int]] = None
        if action is not None:
            err = validate_action(
                player_id,
                action,
                self.players[player_id],
                **self._enemy_tower_flags(player_id),
            )
            if err is not None:
                raise InvalidActionError(err)
            self._apply_action(player_id, action)
            valid_own = action

        # Let opponent act
        opponent_id = 1 - player_id
        opp_state = self._get_state(opponent_id)
        opp_action = (
            self.player2_interface.get_action(opp_state)
            if opponent_id == 1
            else self.player1_interface.get_action(opp_state)
        )
        valid_opp: Optional[Tuple[int, int, int]] = None
        if opp_action is not None:
            err = validate_action(
                opponent_id,
                opp_action,
                self.players[opponent_id],
                **self._enemy_tower_flags(opponent_id),
            )
            if err is None:
                self._apply_action(opponent_id, opp_action)
                valid_opp = opp_action

        # Record actions
        if self.recorder is not None:
            self.recorder.record_action(
                player_id,
                valid_own,
            )
            self.recorder.record_action(
                opponent_id,
                valid_opp,
            )

        # Physics / combat tick
        self._simulate_frame()

        return self._get_state(0), self._get_state(1), self._done

    # ── queries ───────────────────────────────────────────────────────────

    def is_done(self) -> bool:
        return self._done

    def has_winner(self) -> bool:
        return self._winner is not None

    def get_winner(self) -> Optional[int]:
        """0 = player 0 wins, 1 = player 1 wins, None = draw."""
        return self._winner

    def get_state(self, player_id: int) -> State:
        return self._get_state(player_id)

    def _enemy_tower_flags(self, player_id: int) -> dict:
        """Return kwargs for *validate_action* about enemy princess towers."""
        opp = 1 - player_id
        return {
            "enemy_left_princess_dead": self.arena.tower_hp(opp, "left_princess") <= 0,
            "enemy_right_princess_dead": self.arena.tower_hp(opp, "right_princess") <= 0,
        }

    # ══════════════════════════════════════════════════════════════════════
    # Internal simulation
    # ══════════════════════════════════════════════════════════════════════

    def _tick_one_frame(self) -> None:
        """One full frame: read actions → apply → simulate."""
        # Player actions
        s0 = self._get_state(0)
        s1 = self._get_state(1)

        a0 = self.player1_interface.get_action(s0)
        a1 = self.player2_interface.get_action(s1)

        valid_a0: Optional[Tuple[int, int, int]] = None
        valid_a1: Optional[Tuple[int, int, int]] = None

        if a0 is not None:
            err = validate_action(0, a0, self.players[0], **self._enemy_tower_flags(0))
            if err is None:
                self._apply_action(0, a0)
                valid_a0 = a0

        if a1 is not None:
            err = validate_action(1, a1, self.players[1], **self._enemy_tower_flags(1))
            if err is None:
                self._apply_action(1, a1)
                valid_a1 = a1

        # Record actions before simulation
        if self.recorder is not None:
            self.recorder.record_action(0, valid_a0)
            self.recorder.record_action(1, valid_a1)

        self._simulate_frame()

    def _simulate_frame(self) -> None:
        """Physics, targeting, combat, elixir, cleanup, game-over check."""
        # Snapshot tower HP *before* combat so that delta methods
        # (get_tower_damage_per_tower, etc.) return this frame's damage.
        self._snapshot_tower_hp()

        alive = self.arena.get_alive_entities()

        # Deployment ticking
        for e in alive:
            e.tick_deploy()

        # Targeting
        self.targeting.update_targets(alive, alive)

        # Physics
        self.physics.update(alive)

        # Combat
        self.combat.process_attacks(alive, self.scheduler.current_frame)
        self.combat.update_projectiles()

        # Elixir
        self.elixir_system.update(self.scheduler.is_double_elixir)
        self._sync_elixir()

        # King tower activation when enemy crosses bridge
        self._check_king_activation()

        # Remove dead troops (keep dead buildings for state awareness)
        self.arena.cleanup_dead()

        # Tick down active spell visuals and purge expired ones
        for fx in self.arena.spell_effects:
            fx.remaining_frames -= 1
        self.arena.spell_effects = [
            fx for fx in self.arena.spell_effects if fx.remaining_frames > 0
        ]

        # Game-over
        self._check_game_over()

        # Record frame (god-view, before advancing clock)
        if self.recorder is not None:
            self.recorder.record_frame(
                frame=self.scheduler.current_frame,
                time_remaining=self.scheduler.time_remaining,
                state_p0=self._get_state(0),
                state_p1=self._get_state(1),
            )

        # Advance clock
        self.scheduler.advance()

    def _sync_elixir(self) -> None:
        """Copy elixir values from ElixirSystem into Player objects.

        Validators read ``player.elixir`` so the two must stay in sync.
        """
        for p in self.players:
            p.elixir = self.elixir_system.get(p.player_id)

    def _apply_action(self, player_id: int, action: Tuple[int, int, int]) -> None:
        tile_x, tile_y, card_idx = action
        player = self.players[player_id]
        card_name = player.hand[card_idx]
        cost = CARD_STATS[card_name]["elixir"]

        # Spend elixir
        if not self.elixir_system.spend(player_id, cost):
            return  # silently fail

        # Keep player object in sync after spending
        self._sync_elixir()

        # Play card (cycle hand)
        player.play_card(card_idx)

        # Flip y for player 1 (player 1's "own side" is top of the map)
        actual_x = float(tile_x)
        actual_y = float(tile_y) if player_id == 0 else float(N_HEIGHT_TILES - 1 - tile_y)

        is_spell = CARD_STATS[card_name].get("is_spell", False)
        if is_spell:
            self.arena.apply_spell(card_name, player_id, actual_x, actual_y)
        else:
            self.arena.spawn_troop(card_name, player_id, actual_x, actual_y)

    # ── game over ─────────────────────────────────────────────────────────

    def _check_game_over(self) -> None:  # noqa: C901
        """Check win conditions per Clash Royale rules.

        Priority:
        1. King tower destroyed → immediate win (any time).
        2. Regulation (3:00) ends → if crown counts differ, winner declared.
        3. Overtime sudden death → first new tower destroyed wins.
        4. Overtime expires (4:00) → HP tiebreaker.
        """
        # 1. King destruction — ends game immediately at any time
        for pid in (0, 1):
            king = self.arena.king_tower(pid)
            if king is None or king.is_dead:
                self._done = True
                self._winner = 1 - pid
                return

        # 2 & 3. Overtime handling
        if self.scheduler.is_overtime:
            if self._regulation_crowns is None:
                # First frame we cross 3:00 — snapshot crowns
                crowns = self._count_crowns()
                if crowns[0] != crowns[1]:
                    # Unequal at regulation end → winner now
                    self._done = True
                    self._winner = 0 if crowns[0] > crowns[1] else 1
                    return
                # Equal → overtime begins; record baseline snapshot
                self._regulation_crowns = crowns[:]

            # 3. Sudden death — first tower destroyed after overtime started wins
            current = self._count_crowns()
            if current[0] > self._regulation_crowns[0]:
                self._done = True
                self._winner = 0
                return
            if current[1] > self._regulation_crowns[1]:
                self._done = True
                self._winner = 1
                return

        # 4. Overtime expired with no sudden-death event → HP tiebreaker
        if self.scheduler.is_time_up:
            self._done = True
            self._winner = self._determine_winner_by_crowns()
            return

    def _count_crowns(self) -> List[int]:
        """Return [crowns_for_p0, crowns_for_p1] based on destroyed towers."""
        crowns = [0, 0]
        for pid in (0, 1):
            opponent = 1 - pid
            for side in ("left_princess", "right_princess"):
                if self.arena.tower_hp(opponent, side) <= 0:
                    crowns[pid] += 1
        return crowns

    def _determine_winner_by_crowns(self) -> Optional[int]:
        """HP tiebreaker used only when crown counts are equal after overtime."""
        crowns = self._count_crowns()
        if crowns[0] != crowns[1]:
            return 0 if crowns[0] > crowns[1] else 1

        # Equal crowns → lowest total HP remaining loses
        hp0 = sum(self.arena.tower_hp(0, t) for t in ("left_princess", "right_princess", "king"))
        hp1 = sum(self.arena.tower_hp(1, t) for t in ("left_princess", "right_princess", "king"))
        if hp0 > hp1:
            return 0
        elif hp1 > hp0:
            return 1

        return None  # true draw

    def _check_king_activation(self) -> None:
        """Activate king tower when a princess tower is destroyed.

        In real Clash Royale the king tower wakes up when:
        1. It takes direct damage (handled in :class:`CombatSystem`).
        2. A friendly princess tower is destroyed (handled here).
        """
        for pid in (0, 1):
            king = self.arena.king_tower(pid)
            if king is None or king.is_active:
                continue
            # Check if any of this player's princess towers have been destroyed
            for side in ("left_princess", "right_princess"):
                if self.arena.tower_hp(pid, side) <= 0:
                    king.activate()
                    break

    # ── state building ────────────────────────────────────────────────────

    def _get_state(self, player_id: int) -> State:
        """Build a :class:`State` from the perspective of *player_id*."""
        opponent_id = 1 - player_id
        player = self.players[player_id]
        elixir = self.elixir_system.get(player_id)

        allies = self._detections_for(player_id)
        enemies = self._detections_for(opponent_id)

        # Tower HPs — from player's perspective
        # King tower active flags
        own_king = self.arena.king_tower(player_id)
        opp_king = self.arena.king_tower(opponent_id)
        king_active = own_king.is_active if own_king is not None else False
        enemy_king_active = opp_king.is_active if opp_king is not None else False

        numbers = Numbers(
            elixir=elixir,
            enemy_elixir=self.elixir_system.get(opponent_id),
            left_princess_hp=self.arena.tower_hp(player_id, "left_princess"),
            right_princess_hp=self.arena.tower_hp(player_id, "right_princess"),
            king_hp=self.arena.tower_hp(player_id, "king"),
            left_enemy_princess_hp=self.arena.tower_hp(opponent_id, "left_princess"),
            right_enemy_princess_hp=self.arena.tower_hp(opponent_id, "right_princess"),
            enemy_king_hp=self.arena.tower_hp(opponent_id, "king"),
            time_remaining=self.scheduler.time_remaining,
            king_active=king_active,
            enemy_king_active=enemy_king_active,
            is_double_elixir=self.scheduler.is_double_elixir,
            is_overtime=self.scheduler.is_overtime,
            overtime_remaining=self.scheduler.overtime_remaining,
        )

        # Cards
        cards_tuple = self._build_cards(player)
        ready = player.playable_indices(elixir)

        active_spells = [
            SpellInfo(
                name=fx.name,
                tile_x=fx.center_x,
                tile_y=fx.center_y,
                radius=fx.radius,
                remaining_frames=fx.remaining_frames,
            )
            for fx in self.arena.spell_effects
        ]
        return State(
            allies=allies,
            enemies=enemies,
            numbers=numbers,
            cards=cards_tuple,
            ready=ready,
            active_spells=active_spells,
        )

    def _detections_for(self, player_id: int) -> List[UnitDetection]:
        detections: List[UnitDetection] = []
        for e in self.arena.get_entities_for_player(player_id):
            px, py = tile_to_pixel(e.x, e.y)
            half_w = int(e.hitbox_radius * TILE_WIDTH)
            half_h = int(e.hitbox_radius * TILE_HEIGHT)
            bbox = (
                max(0, int(px) - half_w),
                max(0, int(py) - half_h),
                min(DISPLAY_WIDTH, int(px) + half_w),
                min(DISPLAY_HEIGHT, int(py) + half_h),
            )
            pos = Position(bbox=bbox, conf=1.0, tile_x=e.tile_x, tile_y=e.tile_y)
            unit = Unit(
                name=e.name,
                category="building" if e.is_building else "troop",
                target=e.target_type,
                transport=e.transport,
            )
            detections.append(UnitDetection(unit=unit, position=pos, hp=e.hp, max_hp=e.max_hp))
        return detections

    @staticmethod
    def _build_cards(player: Player) -> Tuple[Card, Card, Card, Card]:
        cards: List[Card] = []
        for name in player.hand:
            stats = CARD_STATS[name]
            is_spell = stats.get("is_spell", False)
            cost = stats["elixir"]
            unit = Unit(
                name=name,
                category="spell" if is_spell else "troop",
                target=stats.get("target", "all"),
                transport=stats.get("transport", "ground"),
            )
            cards.append(Card(name=name, is_spell=is_spell, cost=cost, units=[unit]))
        # Ensure exactly 4
        while len(cards) < 4:
            cards.append(Card(name="empty", is_spell=False, cost=0, units=[]))
        return (cards[0], cards[1], cards[2], cards[3])

    # ── tower HP snapshot ─────────────────────────────────────────────────

    def _snapshot_tower_hp(self) -> None:
        self._prev_tower_hp = {}
        for pid in (0, 1):
            for t in ("left_princess", "right_princess", "king"):
                self._prev_tower_hp[f"p{pid}_{t}"] = self.arena.tower_hp(pid, t)

    def get_tower_damage_delta(self, player_id: int) -> float:
        """
        Return total HP lost by *opponent*'s towers since last snapshot.
        Positive = player dealt damage.
        """
        opponent_id = 1 - player_id
        delta = 0.0
        for t in ("left_princess", "right_princess", "king"):
            key = f"p{opponent_id}_{t}"
            prev = self._prev_tower_hp.get(key, 0.0)
            curr = self.arena.tower_hp(opponent_id, t)
            if prev > curr:
                delta += prev - curr
        return delta

    def get_tower_loss_delta(self, player_id: int) -> float:
        """HP lost by *own* towers since last snapshot (positive = bad)."""
        delta = 0.0
        for t in ("left_princess", "right_princess", "king"):
            key = f"p{player_id}_{t}"
            prev = self._prev_tower_hp.get(key, 0.0)
            curr = self.arena.tower_hp(player_id, t)
            if prev > curr:
                delta += prev - curr
        return delta

    def count_towers_destroyed(self, player_id: int) -> int:
        """Return how many of the **opponent's** towers player has destroyed."""
        opponent_id = 1 - player_id
        count = 0
        for t in ("left_princess", "right_princess", "king"):
            if self.arena.tower_hp(opponent_id, t) <= 0:
                count += 1
        return count

    def get_tower_damage_per_tower(self, player_id: int) -> Dict[str, float]:
        """Return HP lost per opponent tower since last snapshot.

        Keys: ``left_princess``, ``right_princess``, ``king``.
        """
        opponent_id = 1 - player_id
        result: Dict[str, float] = {}
        for t in ("left_princess", "right_princess", "king"):
            key = f"p{opponent_id}_{t}"
            prev = self._prev_tower_hp.get(key, 0.0)
            curr = self.arena.tower_hp(opponent_id, t)
            result[t] = max(0.0, prev - curr)
        return result

    def get_tower_loss_per_tower(self, player_id: int) -> Dict[str, float]:
        """Return HP lost per own tower since last snapshot.

        Keys: ``left_princess``, ``right_princess``, ``king``.
        """
        result: Dict[str, float] = {}
        for t in ("left_princess", "right_princess", "king"):
            key = f"p{player_id}_{t}"
            prev = self._prev_tower_hp.get(key, 0.0)
            curr = self.arena.tower_hp(player_id, t)
            result[t] = max(0.0, prev - curr)
        return result

    def get_leaked_elixir(self, player_id: int) -> float:
        """Return cumulative leaked elixir for *player_id*."""
        return self.elixir_system.get_leaked(player_id)

    def get_alive_troop_elixir_value(self, player_id: int) -> float:
        """Return total elixir cost of alive non-building troops for *player_id*."""
        from clash_royale_engine.utils.constants import CARD_STATS

        total = 0.0
        for e in self.arena.get_entities_for_player(player_id):
            if not e.is_building:
                stats = CARD_STATS.get(e.name)
                if stats is not None:
                    total += float(stats["elixir"])
        return total

    def debug_reward_signals(self, player_id: int) -> Dict[str, float]:
        """Return a snapshot of all reward-relevant signals for debugging.

        Useful for diagnosing zero-reward issues during training.

        Parameters
        ----------
        player_id : int
            The player whose perspective to report from.

        Returns
        -------
        dict[str, float]
            Keys include damage deltas, tower HPs, elixir values, troop counts.
        """
        opp = 1 - player_id
        result: Dict[str, float] = {}

        # Tower damage deltas (since last snapshot)
        result["damage_dealt_total"] = self.get_tower_damage_delta(player_id)
        result["damage_received_total"] = self.get_tower_loss_delta(player_id)

        # Per-tower damage
        for tower, hp in self.get_tower_damage_per_tower(player_id).items():
            result[f"dmg_dealt_{tower}"] = hp
        for tower, hp in self.get_tower_loss_per_tower(player_id).items():
            result[f"dmg_recv_{tower}"] = hp

        # Tower HPs
        for t in ("left_princess", "right_princess", "king"):
            result[f"own_{t}_hp"] = self.arena.tower_hp(player_id, t)
            result[f"enemy_{t}_hp"] = self.arena.tower_hp(opp, t)

        # Elixir
        result["elixir"] = self.elixir_system.get(player_id)
        result["leaked_elixir"] = self.get_leaked_elixir(player_id)
        result["troop_elixir_value"] = self.get_alive_troop_elixir_value(player_id)

        # Troop counts
        result["own_troops"] = float(len(self.arena.get_entities_for_player(player_id)))
        result["enemy_troops"] = float(len(self.arena.get_entities_for_player(opp)))

        # Towers destroyed
        result["towers_destroyed"] = float(self.count_towers_destroyed(player_id))

        # Game state
        result["time_remaining"] = self.scheduler.time_remaining
        result["frame"] = float(self.scheduler.current_frame)

        return result

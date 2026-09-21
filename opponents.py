"""
Synthetic Opponents (Table 1)
=============================
A library of opponent bots for testing and training:

    Random          Random legal actions
    Nit             Very tight, low bluff frequency
    CallingStation  High calling, low folding
    Maniac          High aggression and bluff frequency
    GTOLike         Fixed approximate GTO policy
    RigidBot        Deterministic thresholds and bet sizes
    HeroBot         Wraps the Phase 2 strategy controller

Each bot implements the BaseBot interface.
"""

from __future__ import annotations
import random
from typing import Optional

from bot_interface import BaseBot, HandStartInfo
from game_state import GameState, Action, ActionType, LegalAction, Street
from hand_evaluator import evaluate, card_from_str, HandRank, RANK_VALUE
from equity import monte_carlo_equity
from strategy import StrategyController, StrategyMode, EVConfig, StrategyConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _pick_legal(gs: GameState, seat: int, action_type: ActionType,
                amount: Optional[int] = None) -> Action:
    """Build an Action for the given type, clamping to legal bounds."""
    legals = gs.get_legal_actions(seat)
    for la in legals:
        if la.action_type == action_type:
            if amount is not None:
                amt = max(la.min_amount, min(amount, la.max_amount))
            else:
                amt = la.min_amount
            return Action(seat, action_type, amt, gs.street)

    # Fallback: if the requested type isn't legal, check or fold
    for la in legals:
        if la.action_type == ActionType.CHECK:
            return Action(seat, ActionType.CHECK, 0, gs.street)
    for la in legals:
        if la.action_type == ActionType.FOLD:
            return Action(seat, ActionType.FOLD, 0, gs.street)

    # Absolute fallback
    la = legals[0] if legals else None
    if la:
        return Action(seat, la.action_type, la.min_amount, gs.street)
    return Action(seat, ActionType.FOLD, 0, gs.street)


def _hand_strength_quick(hole: list[int], board: list[int]) -> float:
    """Fast rough hand strength 0-1 using rank only (no Monte Carlo)."""
    if not board:
        # Preflop: use a simple card-rank heuristic
        r1, r2 = sorted([hole[0] // 4, hole[1] // 4], reverse=True)
        suited = (hole[0] % 4) == (hole[1] % 4)
        score = (r1 * 2 + r2) / 36.0  # normalize to ~0-1
        if r1 == r2:
            score += 0.15  # pair bonus
        if suited:
            score += 0.03
        return min(1.0, max(0.0, score))
    else:
        # Post-flop: evaluate and map category to 0-1
        hr = evaluate(hole + board)
        # category 0-8, tiebreakers max 12
        return (hr.category * 13 + hr.tiebreakers[0]) / (8 * 13 + 12)


# ===================================================================
# RandomBot
# ===================================================================
class RandomBot(BaseBot):
    """Picks a random legal action with random sizing."""

    def __init__(self, name: str = "Random", seed: Optional[int] = None):
        super().__init__(name)
        self._rng = random.Random(seed)

    def act(self, gs: GameState) -> Action:
        legals = gs.get_legal_actions(self.seat)
        if not legals:
            return Action(self.seat, ActionType.FOLD, 0, gs.street)
        la = self._rng.choice(legals)
        if la.max_amount > la.min_amount:
            amt = self._rng.randint(la.min_amount, la.max_amount)
        else:
            amt = la.min_amount
        return Action(self.seat, la.action_type, amt, gs.street)


# ===================================================================
# NitBot
# ===================================================================
class NitBot(BaseBot):
    """
    Very tight. Only plays premium hands, rarely bluffs.
    Folds anything below a threshold, bets small with strong hands.
    """

    def __init__(self, name: str = "Nit", seed: Optional[int] = None,
                 open_threshold: float = 0.60, bet_threshold: float = 0.55):
        super().__init__(name)
        self._rng = random.Random(seed)
        self.open_threshold = open_threshold
        self.bet_threshold = bet_threshold

    def act(self, gs: GameState) -> Action:
        strength = _hand_strength_quick(gs.hole_cards[self.seat], gs.board)
        to_call = gs.max_current_bet - gs.current_bets[self.seat]

        # Preflop: only open with premium hands
        if gs.street == Street.PREFLOP and to_call > 0:
            if strength < self.open_threshold:
                return _pick_legal(gs, self.seat, ActionType.FOLD)

        # Post-flop: fold weak hands facing a bet
        if to_call > 0:
            if strength < self.bet_threshold:
                return _pick_legal(gs, self.seat, ActionType.FOLD)
            else:
                # Strong hand: raise sometimes
                if strength > 0.75 and self._rng.random() < 0.4:
                    bet_size = int(gs.pot * 0.5)
                    return _pick_legal(gs, self.seat, ActionType.RAISE,
                                       gs.current_bets[self.seat] + bet_size)
                return _pick_legal(gs, self.seat, ActionType.CALL)

        # No bet to face
        if strength > self.bet_threshold:
            bet_size = int(gs.pot * 0.5)
            return _pick_legal(gs, self.seat, ActionType.BET, bet_size)
        return _pick_legal(gs, self.seat, ActionType.CHECK)


# ===================================================================
# CallingStation
# ===================================================================
class CallingStation(BaseBot):
    """
    Calls almost everything. Rarely folds, rarely raises.
    The classic "I want to see what comes" player.
    """

    def __init__(self, name: str = "CallingStation", seed: Optional[int] = None,
                 fold_threshold: float = 0.10):
        super().__init__(name)
        self._rng = random.Random(seed)
        self.fold_threshold = fold_threshold

    def act(self, gs: GameState) -> Action:
        strength = _hand_strength_quick(gs.hole_cards[self.seat], gs.board)
        to_call = gs.max_current_bet - gs.current_bets[self.seat]

        if to_call > 0:
            # Only fold absolute trash
            if strength < self.fold_threshold:
                return _pick_legal(gs, self.seat, ActionType.FOLD)
            # Rarely raise — only with monsters
            if strength > 0.85 and self._rng.random() < 0.2:
                return _pick_legal(gs, self.seat, ActionType.RAISE,
                                   gs.current_bets[self.seat] + gs.pot)
            return _pick_legal(gs, self.seat, ActionType.CALL)

        # No bet to face: mostly check, occasionally bet strong
        if strength > 0.70 and self._rng.random() < 0.3:
            return _pick_legal(gs, self.seat, ActionType.BET, int(gs.pot * 0.4))
        return _pick_legal(gs, self.seat, ActionType.CHECK)


# ===================================================================
# ManiacBot
# ===================================================================
class ManiacBot(BaseBot):
    """
    Hyper-aggressive. Bets and raises frequently with any hand.
    High bluff frequency, big bet sizes.
    """

    def __init__(self, name: str = "Maniac", seed: Optional[int] = None,
                 aggression: float = 0.70):
        super().__init__(name)
        self._rng = random.Random(seed)
        self.aggression = aggression

    def act(self, gs: GameState) -> Action:
        to_call = gs.max_current_bet - gs.current_bets[self.seat]

        if to_call > 0:
            # Raise most of the time
            if self._rng.random() < self.aggression:
                size = int(gs.pot * self._rng.uniform(0.75, 2.0))
                return _pick_legal(gs, self.seat, ActionType.RAISE,
                                   gs.current_bets[self.seat] + size)
            # Occasionally call
            if self._rng.random() < 0.7:
                return _pick_legal(gs, self.seat, ActionType.CALL)
            return _pick_legal(gs, self.seat, ActionType.FOLD)

        # No bet to face: bet aggressively
        if self._rng.random() < self.aggression:
            size = int(gs.pot * self._rng.uniform(0.5, 1.5))
            return _pick_legal(gs, self.seat, ActionType.BET, max(size, gs.big_blind))
        return _pick_legal(gs, self.seat, ActionType.CHECK)


# ===================================================================
# GTOLikeBot
# ===================================================================
class GTOLikeBot(BaseBot):
    """
    Approximate GTO play. Uses fixed ranges, balanced bet frequencies,
    and standard bet sizing. Intentionally has the implementation
    artifacts described in Section 11:
      - fixed opening ranges
      - fixed c-bet frequency
      - deterministic thresholds
      - simplistic bluff frequencies
    """

    def __init__(self, name: str = "GTOLike", seed: Optional[int] = None):
        super().__init__(name)
        self._rng = random.Random(seed)
        # Fixed strategy parameters (exploitable artifacts)
        self.open_range = 0.40       # top 40% of hands preflop
        self.cbet_freq = 0.65        # c-bet 65% of flops
        self.bluff_freq = 0.30       # bluff 30% when betting
        self.value_threshold = 0.55  # value bet above this
        self.call_threshold = 0.30   # call above this
        self.bet_size_fraction = 0.66  # always bet 2/3 pot
        self._was_preflop_raiser = False

    def observe_hand_start(self, info):
        super().observe_hand_start(info)
        self._was_preflop_raiser = False

    def act(self, gs: GameState) -> Action:
        strength = _hand_strength_quick(gs.hole_cards[self.seat], gs.board)
        to_call = gs.max_current_bet - gs.current_bets[self.seat]
        pot = gs.pot

        # --- PREFLOP ---
        if gs.street == Street.PREFLOP:
            if to_call > 0:
                if strength >= self.open_range:
                    if strength > 0.70 and self._rng.random() < 0.5:
                        # 3-bet with premiums
                        size = gs.max_current_bet * 3
                        self._was_preflop_raiser = True
                        return _pick_legal(gs, self.seat, ActionType.RAISE, size)
                    return _pick_legal(gs, self.seat, ActionType.CALL)
                return _pick_legal(gs, self.seat, ActionType.FOLD)
            else:
                # In BB with no raise — check
                return _pick_legal(gs, self.seat, ActionType.CHECK)

        # --- POST-FLOP ---
        bet_size = max(int(pot * self.bet_size_fraction), gs.big_blind)

        if to_call > 0:
            # Facing a bet
            if strength > self.value_threshold:
                # Raise for value sometimes
                if strength > 0.75 and self._rng.random() < 0.35:
                    raise_size = gs.current_bets[self.seat] + int(pot * 0.8)
                    return _pick_legal(gs, self.seat, ActionType.RAISE, raise_size)
                return _pick_legal(gs, self.seat, ActionType.CALL)
            elif strength > self.call_threshold:
                return _pick_legal(gs, self.seat, ActionType.CALL)
            else:
                # Occasionally bluff-raise
                if self._rng.random() < self.bluff_freq * 0.15:
                    return _pick_legal(gs, self.seat, ActionType.RAISE,
                                       gs.current_bets[self.seat] + int(pot * 0.75))
                return _pick_legal(gs, self.seat, ActionType.FOLD)
        else:
            # No bet to face
            if self._was_preflop_raiser and gs.street == Street.FLOP:
                # C-bet with fixed frequency
                if self._rng.random() < self.cbet_freq:
                    return _pick_legal(gs, self.seat, ActionType.BET, bet_size)
            if strength > self.value_threshold:
                return _pick_legal(gs, self.seat, ActionType.BET, bet_size)
            elif strength < self.bluff_freq and self._rng.random() < self.bluff_freq:
                # Bluff with weak hands at fixed frequency
                return _pick_legal(gs, self.seat, ActionType.BET, bet_size)
            return _pick_legal(gs, self.seat, ActionType.CHECK)


# ===================================================================
# RigidBot
# ===================================================================
class RigidBot(BaseBot):
    """
    Deterministic thresholds and fixed bet sizes.
    No randomization at all — pure if/else policy.
    This is the easiest bot to exploit once its thresholds are identified.
    """

    def __init__(self, name: str = "Rigid", seed: Optional[int] = None):
        super().__init__(name)
        self.fold_threshold = 0.30
        self.call_threshold = 0.45
        self.raise_threshold = 0.70
        self.bet_size = 0.50  # always half pot

    def act(self, gs: GameState) -> Action:
        strength = _hand_strength_quick(gs.hole_cards[self.seat], gs.board)
        to_call = gs.max_current_bet - gs.current_bets[self.seat]
        bet_size = max(int(gs.pot * self.bet_size), gs.big_blind)

        if to_call > 0:
            if strength >= self.raise_threshold:
                return _pick_legal(gs, self.seat, ActionType.RAISE,
                                   gs.current_bets[self.seat] + bet_size)
            elif strength >= self.call_threshold:
                return _pick_legal(gs, self.seat, ActionType.CALL)
            else:
                return _pick_legal(gs, self.seat, ActionType.FOLD)
        else:
            if strength >= self.raise_threshold:
                return _pick_legal(gs, self.seat, ActionType.BET, bet_size)
            else:
                return _pick_legal(gs, self.seat, ActionType.CHECK)


# ===================================================================
# HeroBot — wraps Phase 2 strategy controller + Phase 4 spectator learning
# ===================================================================
class HeroBot(BaseBot):
    """
    The adaptive bot: Phase 1-2 engine + Phase 3 range inference +
    Phase 4 spectator learning.

    During each hand, maintains a live Bayesian range for every opponent
    that narrows as actions are observed. The inferred range feeds into
    the Monte Carlo equity estimator for more accurate EV calculations.
    """

    def __init__(self, name: str = "Hero",
                 mode: StrategyMode = StrategyMode.BALANCED,
                 ev_config: Optional[EVConfig] = None,
                 strategy_config: Optional[StrategyConfig] = None,
                 seed: Optional[int] = None,
                 learning: bool = True):
        super().__init__(name)
        self._controller = StrategyController(
            ev_config=ev_config or EVConfig(equity_simulations=500,
                                            equity_simulations_important=1000),
            strategy_config=strategy_config,
            rng_seed=seed,
        )
        self._controller.mode = mode
        self._base_mode = mode

        # Phase 4: spectator learning
        self.learning = learning
        self._learner = None
        if learning:
            from spectator import SpectatorLearner
            self._learner = SpectatorLearner()

        # Phase 3: live range tracking (per-hand)
        self._range_tracker = None
        self._current_board: list[int] = []
        self._num_players: int = 2

    def observe_hand_start(self, info):
        super().observe_hand_start(info)
        self._num_players = info.num_players
        self._current_board = []

        if self._learner:
            self._learner.on_hand_start(info)

        # Phase 3: init range tracker for this hand
        from range_inference import RangeTracker
        self._range_tracker = RangeTracker(
            hero_seat=info.seat,
            hero_cards=list(info.hole_cards),
        )
        # Build initial ranges using learned models (Phase 4 → Phase 3)
        opp_models = {}
        if self._learner:
            for seat in range(info.num_players):
                if seat != info.seat:
                    opp_models[seat] = self._learner.get_model(seat)
        self._range_tracker.init_preflop(info.num_players, opp_models)

    def observe_action(self, action, game_state):
        if self._learner:
            self._learner.on_action(action, game_state)

        # Phase 3: update opponent range based on their action
        if self._range_tracker and action.player != self.seat:
            facing_bet = (game_state.max_current_bet >
                          game_state.current_bets[action.player])
            self._range_tracker.update_action(
                action_seat=action.player,
                action_type=action.action_type,
                street=action.street,
                board=self._current_board,
                pot=game_state.pot,
                bet_amount=action.amount,
                facing_bet=facing_bet,
            )

    def observe_board(self, street, board):
        self._current_board = list(board)
        if self._learner:
            self._learner.on_board(street, board)
        # Phase 3: remove board cards from ranges
        if self._range_tracker:
            self._range_tracker.update_board(board)

    def observe_showdown(self, info):
        if self._learner:
            self._learner.on_showdown(info)

    def observe_hand_end(self, info):
        if self._learner:
            self._learner.on_hand_end(info)
            self._adapt_strategy()
        self._range_tracker = None  # reset for next hand

    def _adapt_strategy(self):
        """Switch strategy mode based on learned opponent types."""
        if not self._learner or self._base_mode != StrategyMode.BALANCED:
            return

        models = self._learner.models.all_models()
        opponents = {k: v for k, v in models.items() if k != self.seat}
        if len(opponents) == 1:
            opp_model = list(opponents.values())[0]
            if opp_model.hands_observed >= 15:
                suggestion = self._learner.suggest_strategy(opp_model.player_id)
                mode_map = {
                    "value_heavy": StrategyMode.VALUE_HEAVY,
                    "aggressive": StrategyMode.AGGRESSIVE,
                    "trap_heavy": StrategyMode.TRAP_HEAVY,
                    "balanced": StrategyMode.BALANCED,
                }
                self._controller.mode = mode_map.get(suggestion,
                                                     StrategyMode.BALANCED)

    def act(self, gs: GameState) -> Action:
        # Phase 3: feed inferred ranges to the decision engine
        opp_ranges = None
        if self._range_tracker:
            opp_ranges = self._range_tracker.get_ranges_for_equity()

        # Phase 5: build exploit profile for the primary opponent
        exploit_profile = None
        if self._learner and self.learning:
            exploit_profile = self._build_exploit_profile(gs)

        decision = self._controller.decide(
            gs, hero=self.seat,
            opponent_ranges=opp_ranges,
            exploit_profile=exploit_profile,
        )
        return StrategyController.to_action(decision, self.seat, gs.street)

    def _build_exploit_profile(self, gs: GameState):
        """Build an exploit profile for the current primary opponent."""
        from exploiter import Exploiter

        # Find the primary opponent (in HU it's the only one; multi-way
        # use the most aggressive active opponent)
        active_opps = [s for s in gs.active_players if s != self.seat]
        if not active_opps:
            return None

        target = active_opps[0]
        if len(active_opps) > 1:
            # Pick the opponent who has bet/raised most recently
            for a in reversed(gs.action_history):
                if (a.player in active_opps and
                        a.action_type in (ActionType.BET, ActionType.RAISE)):
                    target = a.player
                    break

        model = self._learner.get_model(target)
        exploiter = Exploiter()
        return exploiter.build_profile(model)

    @property
    def learner(self):
        return self._learner

    @property
    def range_tracker(self):
        return self._range_tracker


# ===================================================================
# Registry
# ===================================================================
ALL_OPPONENTS = {
    "random": RandomBot,
    "nit": NitBot,
    "calling_station": CallingStation,
    "maniac": ManiacBot,
    "gto_like": GTOLikeBot,
    "rigid": RigidBot,
}

def make_opponent(name: str, seed: Optional[int] = None) -> BaseBot:
    """Factory: create an opponent bot by name."""
    cls = ALL_OPPONENTS.get(name.lower().replace(" ", "_"))
    if cls is None:
        raise ValueError(f"Unknown opponent: {name}. "
                         f"Available: {list(ALL_OPPONENTS.keys())}")
    return cls(seed=seed)

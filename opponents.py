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
    AdaptiveBot     Changes strategy over time (switches persona)
    HeroBot         The adaptive bot (all phases)

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
# AdaptiveBot — changes strategy over time (design doc §28, Table)
# ===================================================================
class AdaptiveBot(BaseBot):
    """
    Switches persona every `switch_every` hands (e.g. nit → maniac →
    calling station).  The test opponent for drift / changepoint detection
    (§18): a model that never forgets would keep exploiting the old
    persona.
    """

    def __init__(self, name: str = "Adaptive", seed: Optional[int] = None,
                 switch_every: int = 60,
                 personas: Optional[list[str]] = None):
        super().__init__(name)
        self.switch_every = switch_every
        names = personas or ["nit", "maniac", "calling_station"]
        self._personas = [ALL_OPPONENTS[p](seed=(seed or 0) + i) for i, p in enumerate(names)]
        self._persona_names = names
        self._hands = 0

    @property
    def current_persona(self) -> str:
        return self._persona_names[(self._hands // self.switch_every) % len(self._personas)]

    def _active(self) -> BaseBot:
        return self._personas[(self._hands // self.switch_every) % len(self._personas)]

    def observe_hand_start(self, info):
        super().observe_hand_start(info)
        for p in self._personas:
            p.observe_hand_start(info)

    def observe_hand_end(self, info):
        self._hands += 1

    def act(self, gs: GameState) -> Action:
        return self._active().act(gs)


# ===================================================================
# HeroBot — the adaptive bot (all phases)
# ===================================================================
class HeroBot(BaseBot):
    """
    Poker engine + Bayesian range inference with the §12 likelihood model +
    spectator learning + MDF-loop exploitation + regret-matched randomization.

    Per hand: a RangeTracker narrows each opponent's range from their
    actions (with pre-action context); the ranges feed the decision engine,
    together with each opponent's model (response probabilities) and
    ExploitProfile (confidence-scaled fold rates by size).
    """

    def __init__(self, name: str = "Hero",
                 mode: StrategyMode = StrategyMode.BALANCED,
                 ev_config: Optional[EVConfig] = None,
                 strategy_config: Optional[StrategyConfig] = None,
                 seed: Optional[int] = None,
                 learning: bool = True,
                 adapt_mode: bool = True):
        super().__init__(name)
        self._controller = StrategyController(
            ev_config=ev_config or EVConfig(equity_simulations=400,
                                            equity_simulations_important=800),
            strategy_config=strategy_config,
            rng_seed=seed,
        )
        self._controller.mode = mode
        self._base_mode = mode
        self._seed = seed
        self.adapt_mode = adapt_mode

        from spectator import SpectatorLearner
        from exploiter import Exploiter
        from calibration import CalibrationTracker
        self.learning = learning
        self._learner = SpectatorLearner() if learning else None
        self._exploiter = Exploiter()
        self.calibration = CalibrationTracker()

        self._range_tracker = None
        self._profiles: dict = {}
        self._positions: list[str] = []
        self._current_board: list[int] = []
        self._num_players: int = 2
        self._hole: list[int] = []
        self._street_aggressor: dict[int, int] = {}
        self._decisions = 0

    # ------------------------------------------------------------------
    # Observation hooks
    # ------------------------------------------------------------------
    def observe_hand_start(self, info):
        super().observe_hand_start(info)
        self._num_players = info.num_players
        self._positions = list(info.positions)
        self._current_board = []
        self._hole = list(info.hole_cards)
        self._street_aggressor = {}
        if self._learner:
            self._learner.on_hand_start(info)

        from range_inference import RangeTracker
        self._range_tracker = RangeTracker(hero_seat=info.seat,
                                           hero_cards=list(info.hole_cards))
        models = {}
        self._profiles = {}
        if self._learner:
            for seat in range(info.num_players):
                if seat != info.seat:
                    models[seat] = self._learner.get_model(seat)
                    self._profiles[seat] = self._exploiter.build_profile(models[seat])
        sitting_out = [s for s in range(info.num_players)
                       if info.stacks and info.stacks[s] <= 0]
        self._range_tracker.init_preflop(info.num_players, models,
                                         positions=self._positions,
                                         sitting_out=sitting_out)

    def observe_action(self, action, game_state):
        if self._learner:
            self._learner.on_action(action, game_state)
        if action.is_aggressive:
            self._street_aggressor[int(action.street)] = action.player
        if self._range_tracker and action.player != self.seat:
            if action.action_type == ActionType.FOLD:
                self._range_tracker.mark_folded(action.player)
            else:
                self._range_tracker.update_action(
                    action_seat=action.player,
                    action_type=action.action_type,
                    street=action.street,
                    board=self._current_board,
                    action=action,
                )

    def observe_board(self, street, board):
        self._current_board = list(board)
        if self._learner:
            self._learner.on_board(street, board)
        if self._range_tracker:
            self._range_tracker.update_board(board)

    def observe_showdown(self, info):
        # §28 calibration: did the range posterior predict who was ahead?
        if self._range_tracker and len(info.board) == 5 and self.seat in info.revealed_cards:
            from hand_evaluator import evaluate
            mine = evaluate(self._hole + list(info.board))
            for seat, cards in info.revealed_cards.items():
                if seat == self.seat:
                    continue
                lr = self._range_tracker.get_range(seat)
                if lr is None or not lr.size:
                    continue
                w = lr.normalized()
                p_ahead = sum(p for h, p in w.items()
                              if not (set(h) & set(info.board))
                              and evaluate([h[0], h[1]] + list(info.board)) > mine)
                actual = evaluate(list(cards) + list(info.board)) > mine
                self.calibration.record(p_ahead, actual)
        if self._learner:
            self._learner.on_showdown(info)

    def observe_hand_end(self, info):
        if self._learner:
            self._learner.on_hand_end(info)
            self._adapt_strategy()
        self._range_tracker = None

    def _adapt_strategy(self):
        """Mode label from the exploiter's hysteretic classification (§21)."""
        if not self._learner or not self.adapt_mode or self._base_mode != StrategyMode.BALANCED:
            return
        from strategy import MODE_FROM_LABEL
        opps = [s for s in range(self._num_players) if s != self.seat]
        labels = [self._exploiter.build_profile(self._learner.get_model(s)).strategy_mode
                  for s in opps]
        if labels and all(l == labels[0] for l in labels):
            self._controller.mode = MODE_FROM_LABEL.get(labels[0], StrategyMode.BALANCED)
        else:
            self._controller.mode = StrategyMode.BALANCED

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------
    def decision_inputs(self, gs: GameState) -> dict:
        """Everything the engine needs beyond the game state."""
        live = [s for s in gs.active_players if s != self.seat]
        ranges = (self._range_tracker.get_ranges_for_equity(live)
                  if self._range_tracker else None)
        models = ({s: self._learner.get_model(s) for s in live}
                  if self._learner else {})
        prev = self._street_aggressor.get(int(gs.street) - 1) if gs.street > 0 else None
        return dict(opponent_ranges=ranges, models=models,
                    exploit_profiles={s: self._profiles[s] for s in live if s in self._profiles},
                    positions=self._positions, prev_aggressor=prev)

    def act(self, gs: GameState) -> Action:
        self._decisions += 1
        kw = self.decision_inputs(gs)
        seed = None if self._seed is None else self._seed * 100003 + self._decisions
        decision = self._controller.decide(gs, hero=self.seat, rng_seed=seed, **kw)
        return StrategyController.to_action(decision, self.seat, gs.street)

    @property
    def learner(self):
        return self._learner

    @property
    def exploiter(self):
        return self._exploiter

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

ALL_OPPONENTS["adaptive"] = AdaptiveBot   # wraps the entries above

def make_opponent(name: str, seed: Optional[int] = None) -> BaseBot:
    """Factory: create an opponent bot by name."""
    cls = ALL_OPPONENTS.get(name.lower().replace(" ", "_"))
    if cls is None:
        raise ValueError(f"Unknown opponent: {name}. "
                         f"Available: {list(ALL_OPPONENTS.keys())}")
    return cls(seed=seed)

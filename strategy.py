"""
Strategy Controller
===================
Implements the randomized policy from Section 6 and the strategy
controller from Section 18.

The key insight: a deterministic policy (Equation 17) is exploitable.
Instead, use softmax action selection (Equation 22):

    P(a | s) = exp(Q(s,a) / τ) / Σ exp(Q(s,a') / τ)

where τ controls exploration vs exploitation.

Phase 2 implements:
  - softmax action selection from Q-values
  - temperature scheduling based on hand context
  - basic strategy modes (balanced, value-heavy, aggressive)
  - SPR-aware strategy adjustments

Phase 3+ will add opponent-dependent strategy switching (Equations 63-68).
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from game_state import GameState, ActionType, Action, Street, Position
from decision_engine import DecisionEngine, ActionEV, EVConfig
from board_texture import analyze_board


# ---------------------------------------------------------------------------
# Strategy modes (Section 18, Equation 63)
# ---------------------------------------------------------------------------
class StrategyMode(Enum):
    BALANCED = auto()     # default: moderate temperature
    VALUE_HEAVY = auto()  # lower temp, bias toward value bets
    AGGRESSIVE = auto()   # boost bet/raise EV, bluff more
    TRAP_HEAVY = auto()   # boost check/call EV for strong hands
    TIGHT = auto()        # raise fold threshold, play fewer hands


# ---------------------------------------------------------------------------
# Strategy configuration
# ---------------------------------------------------------------------------
@dataclass
class StrategyConfig:
    """Parameters for the strategy controller."""
    # Base softmax temperature (Equation 22)
    # τ → 0: greedy,  τ → ∞: uniform random
    base_temperature: float = 8.0

    # Temperature scaling
    min_temperature: float = 1.0
    max_temperature: float = 30.0

    # Temperature adjustments by context
    river_temperature_scale: float = 0.6   # tighter play on river
    preflop_temperature_scale: float = 1.3  # more varied preflop
    large_pot_temperature_scale: float = 0.7  # tighter in big pots

    # Large pot threshold (in BBs)
    large_pot_bb: float = 20.0

    # Minimum probability floor (prevent never folding/never raising)
    min_action_probability: float = 0.02

    # Strategy mode EV adjustments (in chips, added to Q-values)
    mode_adjustments: dict = None

    def __post_init__(self):
        if self.mode_adjustments is None:
            self.mode_adjustments = {
                StrategyMode.BALANCED: {},
                StrategyMode.VALUE_HEAVY: {
                    ActionType.BET: 3.0,
                    ActionType.RAISE: 3.0,
                    ActionType.CALL: 1.0,
                    ActionType.FOLD: -2.0,
                },
                StrategyMode.AGGRESSIVE: {
                    ActionType.BET: 5.0,
                    ActionType.RAISE: 5.0,
                    ActionType.ALL_IN: 3.0,
                    ActionType.CHECK: -3.0,
                },
                StrategyMode.TRAP_HEAVY: {
                    ActionType.CHECK: 4.0,
                    ActionType.CALL: 4.0,
                    ActionType.BET: -2.0,
                    ActionType.RAISE: -3.0,
                },
                StrategyMode.TIGHT: {
                    ActionType.FOLD: 3.0,
                    ActionType.CHECK: 1.0,
                    ActionType.BET: -1.0,
                },
            }


DEFAULT_STRATEGY_CONFIG = StrategyConfig()


# ---------------------------------------------------------------------------
# Action decision — what the controller returns
# ---------------------------------------------------------------------------
@dataclass
class ActionDecision:
    """Final action decision with probability distribution."""
    chosen_action: ActionEV
    action_probabilities: list[tuple[str, float]]  # [(label, prob), ...]
    temperature: float
    strategy_mode: StrategyMode
    debug: dict = None


# ---------------------------------------------------------------------------
# Strategy Controller
# ---------------------------------------------------------------------------
class StrategyController:
    """
    Wraps the DecisionEngine with softmax randomization and strategy modes.

    Usage:
        controller = StrategyController()
        decision = controller.decide(game_state, hero=0)
        # decision.chosen_action is the ActionEV to execute
    """

    def __init__(
        self,
        ev_config: EVConfig = None,
        strategy_config: StrategyConfig = None,
        rng_seed: Optional[int] = None,
    ):
        self.engine = DecisionEngine(ev_config or EVConfig())
        self.config = strategy_config or DEFAULT_STRATEGY_CONFIG
        self._rng = random.Random(rng_seed)
        self._mode = StrategyMode.BALANCED

    @property
    def mode(self) -> StrategyMode:
        return self._mode

    @mode.setter
    def mode(self, value: StrategyMode):
        self._mode = value

    def decide(
        self,
        gs: GameState,
        hero: int,
        mode: Optional[StrategyMode] = None,
        opponent_ranges=None,
        rng_seed=None,
        exploit_profile=None,
    ) -> ActionDecision:
        """
        Evaluate all actions, apply softmax randomization, return a decision.

        Parameters
        ----------
        gs : current game state
        hero : seat index of the bot
        mode : override strategy mode (default: use self._mode)
        opponent_ranges : optional ranges for Monte Carlo
        rng_seed : seed for equity simulations (not for action selection)
        exploit_profile : Phase 5 per-opponent exploit profile
        """
        active_mode = mode or self._mode

        # --- 1. Get Q-values from decision engine ---
        action_evs = self.engine.evaluate(
            gs, hero, opponent_ranges, rng_seed,
            exploit_profile=exploit_profile,
        )
        if not action_evs:
            dummy = ActionEV(ActionType.FOLD, 0, 0.0, {}, "Fold (forced)")
            return ActionDecision(
                chosen_action=dummy,
                action_probabilities=[("Fold", 1.0)],
                temperature=0.0,
                strategy_mode=active_mode,
            )

        # --- 2. Deduplicate: keep best EV per action type ---
        best_per_type: dict[ActionType, ActionEV] = {}
        for aev in action_evs:
            at = aev.action_type
            if at not in best_per_type or aev.ev > best_per_type[at].ev:
                best_per_type[at] = aev
        candidates = list(best_per_type.values())

        # --- 3. Apply strategy mode adjustments ---
        adjusted_evs = []
        mode_adj = self.config.mode_adjustments.get(active_mode, {})
        for aev in candidates:
            bonus = mode_adj.get(aev.action_type, 0.0)
            adj_ev = aev.ev + bonus
            adjusted_evs.append((aev, adj_ev))

        # --- 4. Compute temperature ---
        temperature = self._compute_temperature(gs, hero)
        # Phase 5: exploit profile temperature scaling
        if exploit_profile and exploit_profile.confidence > 0.3:
            temperature *= exploit_profile.temperature_scale

        # --- 5. Softmax (Equation 22) ---
        probs = self._softmax(
            [ev for _, ev in adjusted_evs],
            temperature,
        )

        # --- 6. Apply minimum probability floor ---
        probs = self._apply_floor(probs)

        # --- 7. Sample an action ---
        roll = self._rng.random()
        cumulative = 0.0
        chosen_idx = len(probs) - 1
        for i, p in enumerate(probs):
            cumulative += p
            if roll < cumulative:
                chosen_idx = i
                break

        chosen = adjusted_evs[chosen_idx][0]

        # Build probability report
        action_probs = [
            (adjusted_evs[i][0].label, round(probs[i], 3))
            for i in range(len(probs))
        ]
        action_probs.sort(key=lambda x: x[1], reverse=True)

        return ActionDecision(
            chosen_action=chosen,
            action_probabilities=action_probs,
            temperature=round(temperature, 2),
            strategy_mode=active_mode,
            debug={
                "raw_evs": [(aev.label, round(aev.ev, 1)) for aev in candidates],
                "adjusted_evs": [
                    (aev.label, round(adj, 1))
                    for aev, adj in adjusted_evs
                ],
                "mode_bonuses": mode_adj,
            },
        )

    # ------------------------------------------------------------------
    # Softmax
    # ------------------------------------------------------------------
    @staticmethod
    def _softmax(values: list[float], temperature: float) -> list[float]:
        """Compute softmax probabilities (Equation 22)."""
        if temperature <= 0:
            # Greedy: all mass on max
            max_val = max(values)
            return [1.0 if v == max_val else 0.0 for v in values]

        # Subtract max for numerical stability
        max_v = max(values)
        exps = [math.exp((v - max_v) / temperature) for v in values]
        total = sum(exps)
        if total == 0:
            return [1.0 / len(values)] * len(values)
        return [e / total for e in exps]

    # ------------------------------------------------------------------
    # Probability floor
    # ------------------------------------------------------------------
    def _apply_floor(self, probs: list[float]) -> list[float]:
        """Ensure every action has at least min_action_probability."""
        n = len(probs)
        floor = self.config.min_action_probability
        if n * floor >= 1.0:
            return [1.0 / n] * n

        result = list(probs)
        deficit = 0.0
        for i in range(n):
            if result[i] < floor:
                deficit += floor - result[i]
                result[i] = floor

        # Redistribute deficit from the largest probabilities
        if deficit > 0:
            above_floor = [(i, result[i]) for i in range(n) if result[i] > floor]
            total_above = sum(p for _, p in above_floor)
            if total_above > 0:
                for i, p in above_floor:
                    result[i] -= deficit * (p / total_above)

        # Normalize
        total = sum(result)
        if total > 0:
            result = [p / total for p in result]
        return result

    # ------------------------------------------------------------------
    # Temperature scheduling
    # ------------------------------------------------------------------
    def _compute_temperature(self, gs: GameState, hero: int) -> float:
        """
        Adapt temperature to game context.
        Lower temperature = more exploitative (greedy).
        Higher temperature = more exploratory (balanced/unpredictable).
        """
        temp = self.config.base_temperature

        # Street adjustment
        if gs.street == Street.RIVER:
            temp *= self.config.river_temperature_scale
        elif gs.street == Street.PREFLOP:
            temp *= self.config.preflop_temperature_scale

        # Large pot → play tighter (lower temp)
        pot_in_bb = gs.pot / gs.big_blind if gs.big_blind > 0 else 0
        if pot_in_bb > self.config.large_pot_bb:
            temp *= self.config.large_pot_temperature_scale

        # Clamp
        temp = max(self.config.min_temperature,
                   min(self.config.max_temperature, temp))

        return temp

    # ------------------------------------------------------------------
    # Convert decision to game Action
    # ------------------------------------------------------------------
    @staticmethod
    def to_action(decision: ActionDecision, hero: int, street: Street) -> Action:
        """Convert an ActionDecision into an Action for apply_action()."""
        aev = decision.chosen_action
        return Action(
            player=hero,
            action_type=aev.action_type,
            amount=aev.amount,
            street=street,
        )

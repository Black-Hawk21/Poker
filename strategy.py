"""
Strategy Controller  (design doc §7, §21)
=========================================
Turns Q-values into a randomized action.

Corrections relative to the original softmax-with-floor policy:

* **Units.**  Q is in chips (or utility), so a fixed softmax temperature
  behaves differently in a 15-chip pot and a 1,500-chip pot.  Q-values are
  normalized by the utility value of the current pot before any
  randomization, making the temperature dimensionless.
* **No EV leakage.**  The old policy put at least 2% on *every* legal
  action — including folding the nuts.  Now only actions within ε of the
  best normalized Q are eligible; dominated actions get probability 0.
* **Balance where it matters.**  At nodes where balance matters (a value
  bet vs. check, or a bluff-catcher call vs. fold, with both options
  near-optimal), the mix comes from regret matching (Eq. 18), not from a
  temperature:
        π_{t+1}(a) = R⁺_t(a) / Σ_a' R⁺_t(a'),  R_t(a) = Σ_τ (u_τ(a) − u_τ(π_τ))
  with a uniform fallback when all regrets are non-positive.  Nodes are
  abstracted by (street, facing a bet, hero strength decile, board
  texture), so the mix is learned across the hands that share a spot.
  Elsewhere a pot-normalized softmax over the near-optimal set is used.
* **Modes are labels, not chip bonuses.**  A mode (value-heavy, aggressive,
  trap-heavy …) is chosen from the opponent classification with hysteresis
  (exploiter.py).  Its only direct effect here is a bounded tie-break among
  near-optimal actions; the substantive exploitation already lives in the
  EV (confidence-scaled q(B) from the MDF loop).  The old +3 / +5 chip
  bonuses could override real EV differences of any size.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from game_state import GameState, ActionType, Action, Street
from decision_engine import DecisionEngine, ActionEV, EVConfig
from hand_strength import hand_strength
from opponent_model import texture_class


class StrategyMode(Enum):
    BALANCED = auto()
    VALUE_HEAVY = auto()
    AGGRESSIVE = auto()
    TRAP_HEAVY = auto()
    TIGHT = auto()


MODE_FROM_LABEL = {
    "balanced": StrategyMode.BALANCED,
    "value_heavy": StrategyMode.VALUE_HEAVY,
    "aggressive": StrategyMode.AGGRESSIVE,
    "trap_heavy": StrategyMode.TRAP_HEAVY,
    "tight": StrategyMode.TIGHT,
}

# Tie-break preferences among near-optimal actions, in units of ε.
MODE_TIEBREAK = {
    StrategyMode.BALANCED: {},
    StrategyMode.VALUE_HEAVY: {"bet": 0.5, "raise": 0.5},
    StrategyMode.AGGRESSIVE: {"bet": 0.5, "raise": 0.5, "allin": 0.25},
    StrategyMode.TRAP_HEAVY: {"check": 0.5, "call": 0.5},
    StrategyMode.TIGHT: {"fold": 0.5, "check": 0.25},
}


@dataclass
class StrategyConfig:
    # Candidate set: actions within ε·(pot utility) of the best Q.
    near_optimal_epsilon: float = 0.05
    # Softmax temperature on pot-normalized Q (dimensionless).
    temperature: float = 0.02
    # Use regret matching at balance nodes.
    use_regret_matching: bool = True
    # Only these streets are treated as balance nodes.
    balance_streets: tuple = (Street.TURN, Street.RIVER)

    # --- Deprecated fields (accepted, not used) ------------------------
    base_temperature: float = 8.0
    min_temperature: float = 1.0
    max_temperature: float = 30.0
    min_action_probability: float = 0.0
    mode_adjustments: Optional[dict] = None


DEFAULT_STRATEGY_CONFIG = StrategyConfig()


@dataclass
class ActionDecision:
    chosen_action: ActionEV
    action_probabilities: list[tuple[str, float]]
    temperature: float            # effective temperature in utility units
    strategy_mode: StrategyMode
    debug: dict = field(default_factory=dict)


def action_kind(aev: ActionEV) -> str:
    """Abstract action identity used by regret matching and tie-breaks."""
    t = aev.action_type
    if t == ActionType.FOLD:
        return "fold"
    if t == ActionType.CHECK:
        return "check"
    if t == ActionType.CALL:
        return "call"
    if t == ActionType.ALL_IN or "ALL_IN" in aev.label:
        return "allin"
    frac = aev.ev_components.get("bet_fraction_of_pot", 0.66)
    base = "bet" if t == ActionType.BET else "raise"
    return base + ("_small" if frac < 0.6 else "_big")


# ---------------------------------------------------------------------------
# Regret matching (Eq. 18)
# ---------------------------------------------------------------------------
class RegretMatcher:
    """Regret matching over abstract nodes, with full-information updates."""

    def __init__(self):
        self.regret: dict[tuple, dict[str, float]] = {}
        self.strategy_sum: dict[tuple, dict[str, float]] = {}
        self.visits: dict[tuple, int] = {}

    def strategy(self, node: tuple, actions: list[str]) -> dict[str, float]:
        r = self.regret.get(node, {})
        pos = {a: max(0.0, r.get(a, 0.0)) for a in actions}
        s = sum(pos.values())
        if s <= 0:
            return {a: 1.0 / len(actions) for a in actions}
        return {a: v / s for a, v in pos.items()}

    def update(self, node: tuple, utilities: dict[str, float]):
        """R(a) += u(a) − Σ_a' π(a') u(a')  for every action a at the node."""
        actions = list(utilities)
        pi = self.strategy(node, actions)
        baseline = sum(pi[a] * utilities[a] for a in actions)
        r = self.regret.setdefault(node, {})
        for a in actions:
            r[a] = r.get(a, 0.0) + utilities[a] - baseline
        ss = self.strategy_sum.setdefault(node, {})
        for a in actions:
            ss[a] = ss.get(a, 0.0) + pi[a]
        self.visits[node] = self.visits.get(node, 0) + 1

    def average_strategy(self, node: tuple) -> dict[str, float]:
        ss = self.strategy_sum.get(node, {})
        s = sum(ss.values())
        return {a: v / s for a, v in ss.items()} if s > 0 else {}


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------
class StrategyController:
    """DecisionEngine + near-optimal randomization + regret-matched balance."""

    def __init__(self, ev_config: Optional[EVConfig] = None,
                 strategy_config: Optional[StrategyConfig] = None,
                 rng_seed: Optional[int] = None,
                 likelihood_model=None):
        self.engine = DecisionEngine(ev_config or EVConfig(), likelihood_model)
        self.config = strategy_config or DEFAULT_STRATEGY_CONFIG
        self._rng = random.Random(rng_seed)
        self._mode = StrategyMode.BALANCED
        self.regrets = RegretMatcher()

    @property
    def mode(self) -> StrategyMode:
        return self._mode

    @mode.setter
    def mode(self, value: StrategyMode):
        self._mode = value

    # ------------------------------------------------------------------
    def decide(self, gs: GameState, hero: int, mode: Optional[StrategyMode] = None,
               opponent_ranges=None, rng_seed=None, exploit_profile=None,
               **engine_kwargs) -> ActionDecision:
        active_mode = mode or self._mode
        evs = self.engine.evaluate(gs, hero, opponent_ranges, rng_seed,
                                   exploit_profile=exploit_profile, **engine_kwargs)
        if not evs:
            dummy = ActionEV(ActionType.FOLD, 0, 0.0, {}, "Fold (forced)")
            return ActionDecision(dummy, [("Fold", 1.0)], 0.0, active_mode)

        scale = self._pot_scale(gs, hero, evs)
        eps = self.config.near_optimal_epsilon
        best = max(a.ev for a in evs) / scale
        cands = [a for a in evs if a.ev / scale >= best - eps]

        # one representative (best EV) per abstract action kind
        reps: dict[str, ActionEV] = {}
        for a in cands:
            k = action_kind(a)
            if k not in reps or a.ev > reps[k].ev:
                reps[k] = a
        kinds = list(reps)
        qn = {k: reps[k].ev / scale for k in kinds}

        node = self._balance_node(gs, hero, kinds)
        if node is not None and self.config.use_regret_matching and len(kinds) > 1:
            pi = self.regrets.strategy(node, kinds)
            self.regrets.update(node, qn)
            method = "regret_matching"
        else:
            tb = MODE_TIEBREAK.get(active_mode, {})
            logits = [qn[k] + eps * tb.get(k.split("_")[0], 0.0) for k in kinds]
            probs = self._softmax(logits, self.config.temperature)
            pi = dict(zip(kinds, probs))
            method = "softmax_near_optimal"

        roll = self._rng.random()
        acc = 0.0
        chosen_kind = kinds[-1]
        for k in kinds:
            acc += pi[k]
            if roll < acc:
                chosen_kind = k
                break
        chosen = reps[chosen_kind]

        probs_report = sorted(((reps[k].label, round(pi[k], 3)) for k in kinds),
                              key=lambda x: -x[1])
        return ActionDecision(
            chosen_action=chosen,
            action_probabilities=probs_report,
            temperature=round(self.config.temperature * scale, 3),
            strategy_mode=active_mode,
            debug={
                "method": method,
                "node": node,
                "pot_scale": round(scale, 3),
                "epsilon": eps,
                "candidates": [(a.label, round(a.ev, 2)) for a in cands],
                "raw_evs": [(a.label, round(a.ev, 2)) for a in evs],
                "excluded": [a.label for a in evs if a not in cands],
            },
        )

    # ------------------------------------------------------------------
    def _pot_scale(self, gs: GameState, hero: int, evs: list[ActionEV]) -> float:
        """Utility value of the current pot (chips → the pot itself)."""
        util = self.engine.utility
        opps = [s for s in gs.active_players if s != hero]
        ref = list(gs.starting_stacks) if gs.starting_stacks else list(gs.stacks)
        ref[hero] = gs.stacks[hero]
        s = util.pot_scale(gs.stacks[hero], max(gs.pot, gs.big_blind), hero, ref, opps)
        return max(s, 1e-9)

    def _balance_node(self, gs: GameState, hero: int, kinds: list[str]) -> Optional[tuple]:
        """A spot where both an aggressive/continuing and a passive option are
        near-optimal on a late street (value-bet vs check, call vs fold)."""
        if gs.street not in self.config.balance_streets:
            return None
        ks = {k.split("_")[0] for k in kinds}
        facing = gs.max_current_bet > gs.current_bets[hero]
        if facing and not ({"call", "raise", "allin"} & ks and "fold" in ks):
            return None
        if not facing and not ({"bet", "allin"} & ks and "check" in ks):
            return None
        s = hand_strength(gs.hole_cards[hero], gs.board)
        return (int(gs.street), facing, int(min(s, 0.999) * 10), texture_class(gs.board))

    @staticmethod
    def _softmax(values: list[float], temperature: float) -> list[float]:
        if temperature <= 0:
            mx = max(values)
            hits = [1.0 if v == mx else 0.0 for v in values]
            s = sum(hits)
            return [h / s for h in hits]
        mx = max(values)
        ex = [math.exp((v - mx) / temperature) for v in values]
        s = sum(ex)
        return [e / s for e in ex]

    @staticmethod
    def to_action(decision: ActionDecision, hero: int, street: Street) -> Action:
        a = decision.chosen_action
        return Action(player=hero, action_type=a.action_type, amount=a.amount, street=street)

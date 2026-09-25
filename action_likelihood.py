"""
Action-Likelihood Model  (design doc §12)
=========================================
Range inference (§11) is only as good as P(A | h, s, M): the probability
that opponent M takes action A in state s while holding h.  This module
specifies that term, following the three-step progression of §12:

1. **Population baseline.**  A smooth hand-strength → action map: strong
   hands bet/raise, medium hands check/call, weak hands check/fold, with
   sigmoid transitions and a uniform bluff component.  Strengths are the
   percentiles of `hand_strength` (fraction of holdings beaten on this
   board), so the map is meaningful on every texture.

2. **Per-opponent specialization.**  The thresholds are *calibrated* so the
   map reproduces the opponent's measured, backoff-smoothed frequencies for
   this conditioning cell (§10.2).  If an opponent bets 70% of the time in
   this spot, the weighted average of P(bet | h) over their current range
   is 70%; if they fold 60% to a pot-sized bet, their fold region covers
   60% of the range.  Fold frequencies use q̂(B) for the bet-size bucket
   actually faced (§6).  The bluff component comes from the opponent's
   showdown-derived bluff frequency.

3. **Bet sizing.**  For bets and raises the likelihood is multiplied by
   P(size bucket | strength class, M), learned from showdowns (§12.3).
   Rigid sizers therefore leak their holdings through size; opponents whose
   sizes do not depend on strength contribute a flat term.  Deterministic
   bots (high threshold consistency, §14) get a sharper transition.

A small uniform noise floor keeps every combo alive, so the posterior never
collapses on a single guessed hand (§11, §29).
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional

from hand_strength import weighted_quantile
from opponent_model import (
    OpponentModel, ROOT_PRIORS, strength_class, UNOPENED_ACTIONS,
    FACING_ACTIONS, PREFLOP_ACTIONS, balanced_fold_rate,
)

Combo = tuple[int, int]


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-min(x, 60.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(x, -60.0))
    return z / (1.0 + z)


class ActionLikelihoodModel:
    """
    Computes P(action | h) for every combo in a range.

    Parameters
    ----------
    width : softness of the strength thresholds (in percentile units).
    rigid_width : width used for opponents fingerprinted as deterministic.
    noise : uniform floor mixed into every likelihood.
    population_bluff_share : share of bets/raises that are bluffs when no
        opponent data exists.
    """

    def __init__(self, width: float = 0.06, rigid_width: float = 0.02,
                 noise: float = 0.03, population_bluff_share: float = 0.25):
        self.width = width
        self.rigid_width = rigid_width
        self.noise = noise
        self.population_bluff_share = population_bluff_share

    # ------------------------------------------------------------------
    # Target frequencies for a cell
    # ------------------------------------------------------------------
    def target_frequencies(self, decision: str, cell: tuple,
                           model: Optional[OpponentModel],
                           size_fraction: Optional[float] = None) -> dict[str, float]:
        """Backoff-smoothed action frequencies for this opponent and cell."""
        acts = {"unopened": UNOPENED_ACTIONS, "facing": FACING_ACTIONS,
                "preflop": PREFLOP_ACTIONS}[decision]
        full = (decision,) + tuple(cell)
        if model is not None:
            freq = model.action_distribution(full)
        else:
            freq = dict(ROOT_PRIORS[decision])
        if decision in ("facing", "preflop") and size_fraction is not None:
            # Fold frequency for the size actually faced: q̂(B) (§6)
            q = (model.estimated_fold_to_bet(size_fraction) if model is not None
                 else balanced_fold_rate(size_fraction))
            rest = max(1e-6, freq.get("call", 0) + freq.get("raise", 0))
            freq = {"fold": q,
                    "call": (1 - q) * freq.get("call", 0) / rest,
                    "raise": (1 - q) * freq.get("raise", 0) / rest}
        s = sum(freq.get(a, 0.0) for a in acts)
        return {a: freq.get(a, 0.0) / s for a in acts} if s > 0 else freq

    def _bluff_share(self, model: Optional[OpponentModel]) -> float:
        if model is None:
            return self.population_bluff_share
        return min(0.6, max(0.02, model.bluff_frequency.mean))

    def _width(self, model: Optional[OpponentModel]) -> float:
        if model is not None:
            tc = model.threshold_consistency()
            if tc is not None and tc > 0.9:
                return self.rigid_width
        return self.width

    # ------------------------------------------------------------------
    # Core: calibrate thresholds so the policy reproduces `freq`
    # ------------------------------------------------------------------
    def calibrate(self, strengths: dict[Combo, float],
                  weights: dict[Combo, float], decision: str,
                  freq: dict[str, float], bluff_share: float,
                  width: float) -> "CalibratedPolicy":
        """Fit the thresholds on the range, binned by strength (100 bins).

        Value region: the top `f_agg·(1−bluff_share)` of the range bets or
        raises (sigmoid edge of `width`).  The rest bluffs at a uniform rate
        that makes the total aggression frequency equal f_agg.  Facing a
        bet, the fold threshold is solved so the expected fold rate over the
        range equals the target fold frequency.
        """
        bins = [0.0] * 101
        for h, w in weights.items():
            if w > 0:
                bins[int(min(max(strengths.get(h, 0.5), 0.0), 1.0) * 100)] += w
        tot = sum(bins) or 1.0
        pts = [(i / 100.0 + 0.005, b / tot) for i, b in enumerate(bins) if b > 0]

        agg_kind = "bet" if decision == "unopened" else "raise"
        f_agg = freq.get(agg_kind, 0.0)
        value_mass = f_agg * (1 - bluff_share)
        bluff_rate = (f_agg * bluff_share / (1 - value_mass)) if value_mass < 1 else 0.0
        bluff_rate = min(max(bluff_rate, 0.0), 1.0)
        t_v = weighted_quantile(pts, 1.0 - value_mass) if value_mass > 0 else 2.0
        pol = CalibratedPolicy(decision, t_v, bluff_rate, 2.0 if decision == "unopened" else -1.0,
                               width, value_mass)
        if decision == "unopened":
            return pol

        f_fold = freq.get("fold", 0.0)
        aggs = [(s, w, pol.p_agg(s)) for s, w in pts]

        def fold_rate(t):
            return sum(w * (1 - pa) * _sigmoid((t - s) / width) for s, w, pa in aggs)

        lo, hi = -0.5, 1.5
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if fold_rate(mid) < f_fold:
                lo = mid
            else:
                hi = mid
        pol.t_f = 0.5 * (lo + hi)
        return pol

    def action_probabilities(self, strengths: dict[Combo, float],
                             weights: dict[Combo, float], decision: str,
                             freq: dict[str, float], bluff_share: float,
                             width: float) -> dict[Combo, dict[str, float]]:
        """P(a | h) for every combo, calibrated so E_w[P(a|h)] ≈ freq[a]."""
        pol = self.calibrate(strengths, weights, decision, freq, bluff_share, width)
        return {h: pol.probs(strengths.get(h, 0.5)) for h in weights}

    def policy(self, strengths: dict[Combo, float], weights: dict[Combo, float],
               decision: str, model: Optional[OpponentModel], cell: tuple,
               faced_size: Optional[float] = None,
               fold_override: Optional[float] = None) -> "CalibratedPolicy":
        """Calibrated policy for one opponent in one spot (used by the engine).

        fold_override replaces the fold frequency target (the exploiter's
        confidence-scaled q(B), §22).
        """
        freq = self.target_frequencies(decision, cell, model, faced_size)
        if fold_override is not None and "fold" in freq:
            q = min(max(fold_override, 0.0), 1.0)
            rest = max(1e-9, 1.0 - freq["fold"])
            freq = {a: (q if a == "fold" else (1 - q) * v / rest) for a, v in freq.items()}
        return self.calibrate(strengths, weights, decision, freq,
                              self._bluff_share(model), self._width(model))

    def likelihoods(self, strengths: dict[Combo, float],
                    weights: dict[Combo, float], decision: str, kind: str,
                    model: Optional[OpponentModel] = None,
                    cell: tuple = (),
                    size_fraction: Optional[float] = None,
                    faced_size: Optional[float] = None) -> dict[Combo, float]:
        """
        P(observed action | h) for every combo in `weights`.

        decision : "unopened" (check/bet), "facing" (fold/call/raise) or
                   "preflop" (fold/call/raise; a BB check = "call").
        kind : the observed action class.
        size_fraction : size of the observed bet/raise (for §12.3).
        faced_size : size of the bet being faced (for q̂(B)).
        """
        freq = self.target_frequencies(decision, cell, model, faced_size)
        probs = self.action_probabilities(
            strengths, weights, decision, freq,
            self._bluff_share(model), self._width(model))

        size_lr = None
        if (model is not None and size_fraction is not None
                and kind in ("bet", "raise")):
            size_lr = model.size_likelihood(size_fraction)

        n = self.noise
        out: dict[Combo, float] = {}
        for h, pd in probs.items():
            if kind == "check" and decision != "unopened":
                p = 1.0 - pd["raise"]      # e.g. big blind checks its option
            else:
                p = pd.get(kind, 0.0)
            if size_lr is not None:
                p *= size_lr[strength_class(strengths.get(h, 0.5))]
            out[h] = n + (1.0 - n) * min(p, 1.0)
        return out



@dataclass
class CalibratedPolicy:
    """Thresholds fitted to one opponent's frequencies in one spot."""
    decision: str
    t_v: float            # value threshold (bet/raise above)
    bluff_rate: float     # aggression rate among non-value hands
    t_f: float            # fold threshold (fold below), facing decisions
    width: float
    value_mass: float

    def p_agg(self, s: float) -> float:
        v = _sigmoid((s - self.t_v) / self.width) if self.value_mass > 0 else 0.0
        return v + (1 - v) * self.bluff_rate

    def probs(self, s: float) -> dict[str, float]:
        pa = self.p_agg(s)
        if self.decision == "unopened":
            return {"bet": pa, "check": 1.0 - pa}
        pf = (1 - pa) * _sigmoid((self.t_f - s) / self.width)
        return {"raise": pa, "fold": pf, "call": max(0.0, 1.0 - pa - pf)}


DEFAULT_LIKELIHOOD_MODEL = ActionLikelihoodModel()

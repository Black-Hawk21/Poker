"""
Opponent Model — Sections 9, 12, 14–16
=======================================
Maintains a learned model M_i (Equation 26) for each opponent.

Components:
  - Per-opponent statistics (Section 9.1): VPIP, PFR, c-bet, aggression, etc.
  - Opponent type classification (Section 9.2): probabilistic P(T_i=k | O)
  - Online Bayesian estimation (Section 14): Beta priors on binary stats
  - Recency and strategy drift (Section 15): long-term + recent models
  - Population vs individual (Section 16): hierarchical prior → posterior
  - Policy fingerprinting (Section 12): bet-size entropy, threshold detection

Key data structure from Section 22:
    OpponentModel:
        hands_observed, vpip, pfr, three_bet, cbet_by_street,
        fold_by_street, aggression_by_street, bet_size_distribution,
        bluff_posterior, value_posterior, range_model,
        recent_model, long_term_model, opponent_type_distribution
"""

from __future__ import annotations
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from game_state import ActionType, Street


# ---------------------------------------------------------------------------
# Beta distribution helper (Section 14, Equations 49–51)
# ---------------------------------------------------------------------------
@dataclass
class BetaStat:
    """
    Bayesian estimate of a binary probability using a Beta prior.
        p ~ Beta(alpha, beta)
    After observing `successes` and `failures`:
        p | D ~ Beta(alpha + successes, beta + failures)
    Posterior mean = (alpha + s) / (alpha + beta + s + f)
    """
    alpha: float = 1.0     # prior successes (1 = uniform prior)
    beta: float = 1.0      # prior failures
    successes: int = 0
    failures: int = 0

    def update(self, success: bool):
        if success:
            self.successes += 1
        else:
            self.failures += 1

    @property
    def mean(self) -> float:
        """Posterior mean (Equation 51)."""
        a = self.alpha + self.successes
        b = self.beta + self.failures
        return a / (a + b) if (a + b) > 0 else 0.5

    @property
    def count(self) -> int:
        return self.successes + self.failures

    @property
    def confidence(self) -> float:
        """How much data we have. 0 = prior only, 1 = very confident."""
        n = self.count
        return n / (n + 10.0)  # half-life at 10 observations

    def __repr__(self):
        return f"Beta({self.mean:.2f}, n={self.count})"


# ---------------------------------------------------------------------------
# Exponentially weighted stat (Section 15, Equation 52)
# ---------------------------------------------------------------------------
@dataclass
class EWMAStat:
    """
    Exponentially weighted moving average for detecting strategy drift.
        S_t = λ S_{t-1} + (1-λ) x_t
    """
    value: float = 0.5
    count: int = 0
    decay: float = 0.95  # λ — higher = more memory

    def update(self, x: float):
        if self.count == 0:
            self.value = x
        else:
            self.value = self.decay * self.value + (1 - self.decay) * x
        self.count += 1


# ---------------------------------------------------------------------------
# Bet-size tracker for fingerprinting (Section 12, Equation 46)
# ---------------------------------------------------------------------------
class BetSizeTracker:
    """
    Tracks bet sizes as fractions of pot, per street.
    Computes bet-size entropy H(S) for fingerprinting (Equation 46).
    Low entropy = rigid sizing (exploitable).
    """

    def __init__(self, bucket_width: float = 0.1):
        self.bucket_width = bucket_width
        # street → list of (bet_fraction_of_pot)
        self.sizes: dict[int, list[float]] = defaultdict(list)

    def record(self, street: int, bet_amount: int, pot_before: int):
        if pot_before > 0:
            frac = bet_amount / pot_before
        else:
            frac = 1.0
        self.sizes[street].append(frac)

    def entropy(self, street: Optional[int] = None) -> float:
        """
        Bet-size entropy H(S) = -Σ P(s) log P(s)  (Equation 46).
        Bucketizes bet sizes and computes distribution entropy.
        """
        if street is not None:
            sizes = self.sizes.get(street, [])
        else:
            sizes = [s for vals in self.sizes.values() for s in vals]

        if not sizes:
            return 0.0

        # Bucketize
        buckets: dict[int, int] = defaultdict(int)
        for s in sizes:
            bucket = int(s / self.bucket_width)
            buckets[bucket] += 1

        total = len(sizes)
        h = 0.0
        for count in buckets.values():
            p = count / total
            if p > 0:
                h -= p * math.log2(p)
        return h

    @property
    def total_samples(self) -> int:
        return sum(len(v) for v in self.sizes.values())

    def most_common_size(self, street: Optional[int] = None) -> Optional[float]:
        """Return the most frequently used bet size fraction."""
        if street is not None:
            sizes = self.sizes.get(street, [])
        else:
            sizes = [s for vals in self.sizes.values() for s in vals]
        if not sizes:
            return None
        buckets: dict[int, list[float]] = defaultdict(list)
        for s in sizes:
            bucket = int(s / self.bucket_width)
            buckets[bucket].append(s)
        biggest = max(buckets.values(), key=len)
        return sum(biggest) / len(biggest)


# ---------------------------------------------------------------------------
# Opponent types (Section 9.2, Equations 28–31)
# ---------------------------------------------------------------------------
OPPONENT_TYPES = ["tight", "aggressive", "passive", "balanced", "gto_like", "unknown"]


def classify_opponent(model: "OpponentModel") -> dict[str, float]:
    """
    Probabilistic opponent classification P(T_i = k | O).
    Returns {type_name: probability}.
    Uses VPIP, PFR, aggression, and bet-size entropy as features.
    """
    vpip = model.vpip.mean
    pfr = model.pfr.mean
    agg = model.postflop_aggression.mean
    entropy = model.bet_sizes.entropy()
    n = model.hands_observed

    # Soft scoring — each type gets a raw score, then normalize
    scores = {}

    # Tight / nit: low VPIP, low PFR
    scores["tight"] = (1 - vpip) * 1.5 + (1 - pfr) * 0.5

    # Aggressive / maniac: high aggression, high VPIP
    scores["aggressive"] = agg * 1.5 + vpip * 0.8

    # Passive / calling station: high VPIP, low aggression, low PFR
    scores["passive"] = vpip * 1.2 + (1 - agg) * 1.0 + (1 - pfr) * 0.5

    # Balanced: moderate everything
    bal_penalty = (abs(vpip - 0.5) + abs(pfr - 0.3) + abs(agg - 0.5)) * 2
    scores["balanced"] = max(0, 2.5 - bal_penalty)

    # GTO-like: balanced + low bet-size entropy (fixed sizing)
    if entropy > 0:
        gto_signal = max(0, 2.0 - entropy) * 0.5
    else:
        gto_signal = 0.3
    scores["gto_like"] = scores["balanced"] * 0.6 + gto_signal

    # Unknown: decays as we get more data
    scores["unknown"] = max(0, 2.0 - n * 0.1)

    # Normalize to probabilities
    total = sum(scores.values())
    if total == 0:
        return {t: 1.0 / len(OPPONENT_TYPES) for t in OPPONENT_TYPES}
    return {t: scores[t] / total for t in OPPONENT_TYPES}


# ---------------------------------------------------------------------------
# Per-Opponent Model (Section 9.1, Equation 26)
# ---------------------------------------------------------------------------
class OpponentModel:
    """
    Learned model M_i for one opponent.

    All binary stats use Beta priors (Section 14) so the bot doesn't
    overreact to a single observation (Section 25.1).

    Maintains both long-term and recent models (Section 15) to detect
    strategy drift.
    """

    def __init__(self, player_id: int, prior: Optional["PopulationPrior"] = None):
        self.player_id = player_id
        self.hands_observed: int = 0

        p = prior or PopulationPrior()

        # --- Core stats (Section 9.1) ---
        # Preflop
        self.vpip = BetaStat(p.vpip_alpha, p.vpip_beta)
        self.pfr = BetaStat(p.pfr_alpha, p.pfr_beta)
        self.three_bet = BetaStat(1.0, 4.0)        # ~20% prior

        # Post-flop
        self.cbet_flop = BetaStat(p.cbet_alpha, p.cbet_beta)
        self.cbet_turn = BetaStat(1.5, 1.5)
        self.cbet_river = BetaStat(1.0, 2.0)

        self.fold_to_cbet = BetaStat(1.5, 1.5)
        self.fold_to_bet = BetaStat(p.fold_alpha, p.fold_beta)
        self.fold_to_raise = BetaStat(p.fold_raise_alpha, p.fold_raise_beta)

        self.postflop_aggression = BetaStat(1.5, 1.5)

        # Showdown
        self.bluff_frequency = BetaStat(1.0, 3.0)   # ~25% prior
        self.went_to_sd_with_winner = BetaStat(1.5, 1.5)

        # --- Bet sizing (Section 12) ---
        self.bet_sizes = BetSizeTracker()

        # --- Recency tracking (Section 15) ---
        self.recent_vpip = EWMAStat(decay=0.90)
        self.recent_aggression = EWMAStat(decay=0.90)
        self.recent_fold_to_bet = EWMAStat(decay=0.90)

        # --- Per-street action counts ---
        self._street_actions: dict[int, dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )  # street → {action_type: count}

        # --- Conditional stats: P(action | position, street) (Eq 27) ---
        self._positional_vpip: dict[str, BetaStat] = defaultdict(
            lambda: BetaStat(1.0, 1.0)
        )  # position_name → BetaStat

    # ------------------------------------------------------------------
    # Update methods — called by SpectatorLearner
    # ------------------------------------------------------------------
    def record_preflop_action(self, action_type: int, position: str,
                              is_voluntary: bool, is_raise: bool,
                              facing_raise: bool):
        """Update preflop stats from a single preflop action."""
        if is_voluntary:
            self.vpip.update(True)
            self.recent_vpip.update(1.0)
        elif action_type == ActionType.FOLD and facing_raise:
            self.vpip.update(False)
            self.recent_vpip.update(0.0)

        if is_raise:
            self.pfr.update(True)
        elif action_type in (ActionType.CALL, ActionType.FOLD) and facing_raise:
            self.pfr.update(False)

        if facing_raise and is_raise:
            self.three_bet.update(True)
        elif facing_raise and not is_raise:
            self.three_bet.update(False)

        # Positional VPIP
        if is_voluntary or action_type == ActionType.FOLD:
            self._positional_vpip[position].update(is_voluntary)

    def record_postflop_action(self, action_type: int, street: int,
                               bet_amount: int, pot_before: int,
                               is_aggressor: bool, facing_bet: bool,
                               facing_raise: bool):
        """Update post-flop stats from a single action."""
        self._street_actions[street][action_type] += 1

        is_aggressive = action_type in (
            ActionType.BET, ActionType.RAISE, ActionType.ALL_IN
        )

        # Aggression
        if action_type != ActionType.FOLD:
            self.postflop_aggression.update(is_aggressive)
            self.recent_aggression.update(1.0 if is_aggressive else 0.0)

        # C-bet tracking (bet when was preflop aggressor)
        if is_aggressor and action_type in (ActionType.BET, ActionType.CHECK):
            did_cbet = action_type == ActionType.BET
            if street == Street.FLOP:
                self.cbet_flop.update(did_cbet)
            elif street == Street.TURN:
                self.cbet_turn.update(did_cbet)
            elif street == Street.RIVER:
                self.cbet_river.update(did_cbet)

        # Fold to bet / raise
        if facing_bet:
            folded = action_type == ActionType.FOLD
            self.fold_to_bet.update(folded)
            self.recent_fold_to_bet.update(1.0 if folded else 0.0)
            if facing_raise:
                self.fold_to_raise.update(folded)
            if not facing_raise and not folded:
                self.fold_to_cbet.update(False)  # called a bet (possibly cbet)
            elif not facing_raise and folded:
                self.fold_to_cbet.update(True)

        # Bet sizing
        if is_aggressive and bet_amount > 0:
            self.bet_sizes.record(street, bet_amount, pot_before)

    def record_showdown(self, won: bool, was_bluffing: bool):
        """Update showdown-derived stats."""
        self.went_to_sd_with_winner.update(won)
        if was_bluffing is not None:
            self.bluff_frequency.update(was_bluffing)

    def finish_hand(self):
        """Increment hand counter."""
        self.hands_observed += 1

    # ------------------------------------------------------------------
    # Queries — used by decision engine
    # ------------------------------------------------------------------
    def estimated_fold_to_bet(self) -> float:
        """Best estimate of fold-to-bet frequency."""
        return self.fold_to_bet.mean

    def estimated_fold_to_raise(self) -> float:
        return self.fold_to_raise.mean

    def estimated_cbet(self, street: int) -> float:
        if street == Street.FLOP:
            return self.cbet_flop.mean
        elif street == Street.TURN:
            return self.cbet_turn.mean
        else:
            return self.cbet_river.mean

    def strategy_drift(self) -> float:
        """
        Detect strategy changes (Section 15, Equations 55-56).
        Returns a divergence score: high = opponent changed behavior.
        """
        if self.hands_observed < 20:
            return 0.0

        diffs = []
        # VPIP drift
        if self.recent_vpip.count > 5:
            diffs.append(abs(self.vpip.mean - self.recent_vpip.value))
        # Aggression drift
        if self.recent_aggression.count > 5:
            diffs.append(abs(self.postflop_aggression.mean -
                             self.recent_aggression.value))
        # Fold drift
        if self.recent_fold_to_bet.count > 5:
            diffs.append(abs(self.fold_to_bet.mean -
                             self.recent_fold_to_bet.value))

        return sum(diffs) / len(diffs) if diffs else 0.0

    def classify(self) -> dict[str, float]:
        """Return probabilistic type classification (Section 9.2)."""
        return classify_opponent(self)

    def primary_type(self) -> str:
        """Return the single most likely opponent type."""
        c = self.classify()
        return max(c, key=c.get)

    # ------------------------------------------------------------------
    # Fingerprint (Section 12, Equation 37)
    # ------------------------------------------------------------------
    def fingerprint(self) -> dict:
        """
        Policy fingerprint vector (Equations 38–45).
        """
        return {
            "vpip": round(self.vpip.mean, 3),
            "pfr": round(self.pfr.mean, 3),
            "three_bet": round(self.three_bet.mean, 3),
            "cbet_flop": round(self.cbet_flop.mean, 3),
            "postflop_aggression": round(self.postflop_aggression.mean, 3),
            "fold_to_bet": round(self.fold_to_bet.mean, 3),
            "fold_to_raise": round(self.fold_to_raise.mean, 3),
            "bluff_frequency": round(self.bluff_frequency.mean, 3),
            "bet_size_entropy": round(self.bet_sizes.entropy(), 3),
            "bet_size_common": self.bet_sizes.most_common_size(),
            "hands": self.hands_observed,
            "type": self.primary_type(),
            "type_probs": {k: round(v, 2) for k, v in self.classify().items()},
            "strategy_drift": round(self.strategy_drift(), 3),
            "data_confidence": round(self.vpip.confidence, 2),
        }

    def summary(self) -> str:
        """One-line summary."""
        fp = self.fingerprint()
        return (f"P{self.player_id}: {fp['type']}  "
                f"VPIP={fp['vpip']:.0%} PFR={fp['pfr']:.0%} "
                f"Agg={fp['postflop_aggression']:.0%} "
                f"FoldBet={fp['fold_to_bet']:.0%} "
                f"Bluff={fp['bluff_frequency']:.0%} "
                f"({fp['hands']}h)")


# ---------------------------------------------------------------------------
# Population Prior (Section 16, Equations 57–59)
# ---------------------------------------------------------------------------
@dataclass
class PopulationPrior:
    """
    Default Bayesian priors representing a "typical" opponent.
    Used when little is known about an individual (Equation 59):
        P(H | O, M_i) ≈ P(H | O)  when data is scarce.

    As evidence accumulates, individual posteriors dominate.
    """
    vpip_alpha: float = 2.0    # prior ~50% VPIP (loose default)
    vpip_beta: float = 2.0
    pfr_alpha: float = 1.5     # prior ~37% PFR
    pfr_beta: float = 2.5
    cbet_alpha: float = 2.0    # prior ~57% c-bet
    cbet_beta: float = 1.5
    fold_alpha: float = 1.5    # prior ~43% fold to bet
    fold_beta: float = 2.0
    fold_raise_alpha: float = 2.0   # prior ~50% fold to raise
    fold_raise_beta: float = 2.0


# ---------------------------------------------------------------------------
# Model collection — manages all opponents
# ---------------------------------------------------------------------------
class OpponentModelSet:
    """
    Holds OpponentModel instances for every opponent observed.
    Creates models on demand with population priors.
    """

    def __init__(self, prior: Optional[PopulationPrior] = None):
        self.prior = prior or PopulationPrior()
        self._models: dict[int, OpponentModel] = {}

    def get(self, player_id: int) -> OpponentModel:
        if player_id not in self._models:
            self._models[player_id] = OpponentModel(player_id, self.prior)
        return self._models[player_id]

    def all_models(self) -> dict[int, OpponentModel]:
        return dict(self._models)

    def report(self) -> str:
        lines = ["Opponent Models:"]
        for pid in sorted(self._models):
            lines.append(f"  {self._models[pid].summary()}")
        return "\n".join(lines)

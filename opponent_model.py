"""
Opponent Model  (design doc §10, §14, §16–§19)
==============================================
A learned model M_i for each opponent.

What changed relative to the original specification / code:

* **Posterior uncertainty (§16).**  Every binary statistic is a Beta
  posterior that reports its variance and an exact credible interval.
  `confidence` is derived from the width of that interval, so exploitation
  can be scaled by how tight the estimate really is — not by a hand count.
* **Explicit empirical Bayes (§17, §19).**  The population level supplies
  each statistic's prior as α = κ·μ_pop, β = κ·(1−μ_pop), with μ_pop and κ
  estimated from the observed population by method of moments.  The
  individual estimate is then exactly the shrinkage estimator
  p̂_i = (n_i x̄_i + κ μ_pop)/(n_i + κ).  Near-duplicate opponents (similar
  fingerprints) are partially pooled.
* **Backoff-smoothed conditional statistics (§10.2).**  Action frequencies
  are kept per conditioning cell (decision type × street × position class ×
  board texture × prior action) and read with n-gram-style backoff
  p̂(a|c) = (n_c p̄(a|c) + κ p̂(a|parent(c))) / (n_c + κ).
* **Fold-to-bet by size bucket (§6, §22).**  q̂(B) per sizing bucket, each a
  Beta shrunk toward the opponent's overall fold-to-bet, so it can be
  compared against MDF(B).
* **Sample-size-aware classification (§10.3–§10.4).**  Types are scored by
  the Beta-binomial marginal likelihood of the observed counts under each
  type's prototype.  A statistic with n = 3 barely moves the posterior;
  one with n = 300 dominates it.  No hand-count heuristics.
* **Drift (§18).**  Recent-vs-long-term comparison uses Jensen–Shannon
  divergence on Laplace-smoothed action distributions (symmetric, bounded,
  never infinite), and a CUSUM changepoint detector on key statistics.  A
  detected change discounts old evidence, which both lowers confidence in
  old assumptions and raises the adaptation rate.
* **Fingerprinting (§14).**  Bet-size entropy plus decision-threshold
  consistency measured from showdowns, and the size-given-strength table
  that lets rigid sizers leak their holdings (§12.3).
"""

from __future__ import annotations
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional

from game_state import ActionType, Street


# ---------------------------------------------------------------------------
# Beta distribution utilities
# ---------------------------------------------------------------------------
def _lbeta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the regularized incomplete beta (Lentz)."""
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-10:
            break
    return h


def beta_cdf(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lnfront = a * math.log(x) + b * math.log1p(-x) - _lbeta(a, b)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lnfront) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lnfront) * _betacf(b, a, 1.0 - x) / b


def beta_ppf(q: float, a: float, b: float) -> float:
    """Inverse CDF of Beta(a, b) by bisection (exact to ~1e-6)."""
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if beta_cdf(mid, a, b) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# BetaStat  (§16, Eq. 29 and the posterior variance)
# ---------------------------------------------------------------------------
@dataclass
class BetaStat:
    """
    Bayesian estimate of a binary probability.
        p ~ Beta(α, β);  after s successes and f failures
        p | D ~ Beta(α + s, β + f)
        E[p|D]   = (α+s)/(α+β+s+f)                                (Eq. 29)
        Var[p|D] = (α+s)(β+f) / ((α+β+s+f)² (α+β+s+f+1))
    Counts are floats so that old evidence can be discounted (§18).
    """
    alpha: float = 1.0
    beta: float = 1.0
    successes: float = 0.0
    failures: float = 0.0

    def update(self, success: bool, weight: float = 1.0):
        if success:
            self.successes += weight
        else:
            self.failures += weight

    def set_prior(self, mu: float, kappa: float):
        """Empirical-Bayes prior: α = κμ, β = κ(1−μ)  (§17)."""
        mu = min(max(mu, 1e-3), 1 - 1e-3)
        self.alpha = kappa * mu
        self.beta = kappa * (1.0 - mu)

    def discount(self, factor: float):
        """Shrink old evidence toward the prior (used after a changepoint)."""
        self.successes *= factor
        self.failures *= factor

    @property
    def a(self) -> float:
        return self.alpha + self.successes

    @property
    def b(self) -> float:
        return self.beta + self.failures

    @property
    def mean(self) -> float:
        s = self.a + self.b
        return self.a / s if s > 0 else 0.5

    @property
    def raw_mean(self) -> Optional[float]:
        """Observed frequency x̄ without the prior (None if no data)."""
        n = self.successes + self.failures
        return self.successes / n if n > 0 else None

    @property
    def count(self) -> float:
        n = self.successes + self.failures
        return int(n) if float(n).is_integer() else n

    @property
    def variance(self) -> float:
        a, b = self.a, self.b
        return a * b / ((a + b) ** 2 * (a + b + 1.0))

    @property
    def sd(self) -> float:
        return math.sqrt(self.variance)

    def credible_interval(self, level: float = 0.90) -> tuple[float, float]:
        """Equal-tailed Bayesian credible interval (exact Beta quantiles)."""
        t = (1.0 - level) / 2.0
        return beta_ppf(t, self.a, self.b), beta_ppf(1.0 - t, self.a, self.b)

    @property
    def interval_width(self) -> float:
        lo, hi = self.credible_interval(0.90)
        return hi - lo

    @property
    def confidence(self) -> float:
        """0 = the 90% interval is as wide as for a flat prior, 1 = a point.

        Tied to the credible-interval width (§16): deviate hard only when the
        interval is tight.  The width of a Uniform(0,1) 90% interval is 0.9.
        """
        return max(0.0, min(1.0, 1.0 - self.interval_width / 0.9))

    def beta_binomial_loglik(self, mu: float, conc: float) -> float:
        """log P(observed counts | p ~ Beta(conc·μ, conc·(1−μ)))."""
        a0, b0 = conc * mu, conc * (1.0 - mu)
        k, f = self.successes, self.failures
        return _lbeta(k + a0, f + b0) - _lbeta(a0, b0)

    def __repr__(self):
        return f"Beta({self.mean:.2f}, n={self.count})"


# ---------------------------------------------------------------------------
# EWMA (§18, Eq. 31)
# ---------------------------------------------------------------------------
@dataclass
class EWMAStat:
    """S_t = λ S_{t−1} + (1−λ) x_t."""
    value: float = 0.5
    count: int = 0
    decay: float = 0.95

    def update(self, x: float):
        if self.count == 0:
            self.value = x
        else:
            self.value = self.decay * self.value + (1 - self.decay) * x
        self.count += 1


# ---------------------------------------------------------------------------
# CUSUM changepoint detector (§18)
# ---------------------------------------------------------------------------
@dataclass
class CUSUM:
    """Two-sided CUSUM for a Bernoulli stream.

    Detects a shift of size ≥ `delta` away from the reference mean.  The
    threshold `h` sets the false-alarm rate (larger = fewer alarms).
    """
    delta: float = 0.30
    h: float = 3.5
    s_hi: float = 0.0
    s_lo: float = 0.0
    n: int = 0

    def update(self, x: float, reference: float) -> bool:
        k = self.delta / 2.0
        self.s_hi = max(0.0, self.s_hi + (x - reference) - k)
        self.s_lo = max(0.0, self.s_lo - (x - reference) - k)
        self.n += 1
        return self.n >= 5 and (self.s_hi > self.h or self.s_lo > self.h)

    def reset(self):
        self.s_hi = self.s_lo = 0.0
        self.n = 0


# ---------------------------------------------------------------------------
# Bet-size tracker (§14, Eq. 27)
# ---------------------------------------------------------------------------
SIZE_BUCKET_EDGES = (0.40, 0.80, 1.25)          # fractions of pot
SIZE_BUCKET_NAMES = ("small", "medium", "pot", "over")
SIZE_BUCKET_REPR = (0.33, 0.60, 1.00, 1.75)      # representative fraction


def balanced_fold_rate(size_fraction: float) -> float:
    """Fold rate of a balanced defender facing a bet of B = f·P:
    1 − MDF = B/(P+B) = f/(1+f).  At exactly this rate a 0%-equity bluff
    breaks even (§6)."""
    f = max(0.0, size_fraction)
    return f / (1.0 + f)


def size_bucket(fraction: float) -> int:
    for i, edge in enumerate(SIZE_BUCKET_EDGES):
        if fraction < edge:
            return i
    return len(SIZE_BUCKET_EDGES)


class BetSizeTracker:
    """Bet sizes as fractions of pot, per street; entropy H(S) for fingerprinting."""

    def __init__(self, bucket_width: float = 0.1):
        self.bucket_width = bucket_width
        self.sizes: dict[int, list[float]] = defaultdict(list)

    def record(self, street: int, bet_amount: float, pot_before: float):
        frac = bet_amount / pot_before if pot_before > 0 else 1.0
        self.record_fraction(street, frac)

    def record_fraction(self, street: int, fraction: float):
        self.sizes[int(street)].append(float(fraction))

    def _all(self, street: Optional[int]) -> list[float]:
        if street is not None:
            return self.sizes.get(int(street), [])
        return [s for vals in self.sizes.values() for s in vals]

    def entropy(self, street: Optional[int] = None) -> float:
        """H(S) = −Σ P(s) log₂ P(s) over bucketed sizes  (Eq. 27)."""
        sizes = self._all(street)
        if not sizes:
            return 0.0
        buckets: dict[int, int] = defaultdict(int)
        for s in sizes:
            buckets[int(s / self.bucket_width)] += 1
        total = len(sizes)
        return -sum((c / total) * math.log2(c / total) for c in buckets.values())

    @property
    def total_samples(self) -> int:
        return sum(len(v) for v in self.sizes.values())

    def most_common_size(self, street: Optional[int] = None) -> Optional[float]:
        sizes = self._all(street)
        if not sizes:
            return None
        buckets: dict[int, list[float]] = defaultdict(list)
        for s in sizes:
            buckets[int(s / self.bucket_width)].append(s)
        biggest = max(buckets.values(), key=len)
        return sum(biggest) / len(biggest)


# ---------------------------------------------------------------------------
# Backoff-smoothed conditional action statistics (§10.2, Eq. 23)
# ---------------------------------------------------------------------------
UNOPENED_ACTIONS = ("check", "bet")
FACING_ACTIONS = ("fold", "call", "raise")
PREFLOP_ACTIONS = ("fold", "call", "raise")

# Population defaults at the root of each backoff chain.
ROOT_PRIORS = {
    "unopened": {"check": 0.62, "bet": 0.38},
    "facing": {"fold": 0.40, "call": 0.45, "raise": 0.15},
    "preflop": {"fold": 0.45, "call": 0.35, "raise": 0.20},
}


def position_class(position: str) -> str:
    if position in ("SB", "BB"):
        return "blinds"
    if position in ("BTN", "CO", "HJ"):
        return "late"
    return "early"


def texture_class(board: list[int]) -> str:
    if not board:
        return "none"
    from board_texture import analyze_board
    bt = analyze_board(board)
    if bt.is_paired:
        return "paired"
    return "wet" if bt.wetness >= 0.35 else "dry"


def decision_cell(street: int, position: str, board: list[int],
                  was_aggressor: bool, facing: bool,
                  raises_before: int = 0) -> tuple[str, tuple]:
    """(decision type, conditioning cell) used consistently by the spectator
    (to record) and by range inference / the decision engine (to read).

    Preflop:  ("preflop", ("open"|"raised", position class))
    Postflop: ("facing"|"unopened", (street, position class, texture, "agg"|"non"))
    """
    pos = position_class(position)
    if int(street) == 0:
        return "preflop", ("raised" if raises_before > 0 else "open", pos)
    return ("facing" if facing else "unopened",
            (int(street), pos, texture_class(board), "agg" if was_aggressor else "non"))


def action_kind(action_type: int, is_aggressive: bool, facing: bool) -> str:
    if action_type == ActionType.FOLD:
        return "fold"
    if action_type == ActionType.CHECK:
        return "check"
    if is_aggressive:
        return "raise" if facing else "bet"
    return "call"


class ConditionalActionStats:
    """
    Counts of actions per conditioning cell with backoff smoothing.

    A cell is a tuple ordered coarse → fine, e.g.
        ("facing", FLOP, "late", "wet", "agg")
    Its parent drops the last element; the root ("facing",) backs off to
    the population prior for that decision type.  Every observation is
    added to the cell *and all its ancestors*, so parents hold the pooled
    counts of their children.
    """

    def __init__(self, kappa: float = 5.0,
                 root_priors: Optional[dict[str, dict[str, float]]] = None):
        self.kappa = kappa
        self.root_priors = root_priors or ROOT_PRIORS
        self.counts: dict[tuple, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def record(self, cell: tuple, action: str, weight: float = 1.0):
        for depth in range(1, len(cell) + 1):
            self.counts[cell[:depth]][action] += weight

    def n(self, cell: tuple) -> float:
        c = self.counts.get(cell)
        return sum(c.values()) if c else 0.0

    def prob(self, cell: tuple, action: str) -> float:
        """p̂(a|c) = (n_c p̄(a|c) + κ p̂(a|parent(c))) / (n_c + κ)   (Eq. 23)."""
        if len(cell) == 0:
            return 0.0
        if len(cell) == 1:
            prior = self.root_priors.get(cell[0], {})
            parent = prior.get(action, 1.0 / max(1, len(prior)))
        else:
            parent = self.prob(cell[:-1], action)
        c = self.counts.get(cell)
        n_c = sum(c.values()) if c else 0.0
        if n_c == 0:
            return parent
        return (c.get(action, 0.0) + self.kappa * parent) / (n_c + self.kappa)

    def distribution(self, cell: tuple, actions: Iterable[str]) -> dict[str, float]:
        d = {a: self.prob(cell, a) for a in actions}
        s = sum(d.values())
        return {a: v / s for a, v in d.items()} if s > 0 else d

    def discount(self, factor: float):
        for c in self.counts.values():
            for a in c:
                c[a] *= factor


# ---------------------------------------------------------------------------
# Jensen–Shannon divergence (§18)
# ---------------------------------------------------------------------------
def js_divergence(p: Optional[dict[str, float]], q: Optional[dict[str, float]],
                  support: Iterable[str], laplace: float = 1.0) -> float:
    """D_JS(P‖Q) in bits, ∈ [0, 1], with additive smoothing on the counts."""
    support = list(support)
    p = p or {}
    q = q or {}
    ps = {k: p.get(k, 0.0) + laplace for k in support}
    qs = {k: q.get(k, 0.0) + laplace for k in support}
    sp, sq = sum(ps.values()), sum(qs.values())
    P = {k: v / sp for k, v in ps.items()}
    Q = {k: v / sq for k, v in qs.items()}
    js = 0.0
    for k in support:
        m = 0.5 * (P[k] + Q[k])
        js += 0.5 * P[k] * math.log2(P[k] / m) + 0.5 * Q[k] * math.log2(Q[k] / m)
    return max(0.0, min(1.0, js))


# ---------------------------------------------------------------------------
# Opponent types (§10.4) — prototypes for the likelihood classifier
# ---------------------------------------------------------------------------
OPPONENT_TYPES = ["tight", "aggressive", "passive", "balanced", "gto_like", "unknown"]

# Prototype means for (vpip, pfr, postflop aggression, fold-to-bet).
TYPE_PROTOTYPES: dict[str, dict[str, float]] = {
    # nit: plays few hands, rarely bets, folds to pressure
    "tight":      {"vpip": 0.35, "pfr": 0.10, "postflop_aggression": 0.20, "fold_to_bet": 0.80},
    # maniac: plays most hands, bets/raises constantly, rarely folds
    "aggressive": {"vpip": 0.72, "pfr": 0.55, "postflop_aggression": 0.75, "fold_to_bet": 0.15},
    # calling station: plays most hands, almost never raises or folds
    "passive":    {"vpip": 0.75, "pfr": 0.06, "postflop_aggression": 0.08, "fold_to_bet": 0.15},
    # solid, mixed player
    "balanced":   {"vpip": 0.50, "pfr": 0.30, "postflop_aggression": 0.45, "fold_to_bet": 0.45},
    # approximate-GTO implementation: moderate frequencies + fixed sizes
    "gto_like":   {"vpip": 0.48, "pfr": 0.15, "postflop_aggression": 0.35, "fold_to_bet": 0.60},
}
TYPE_CONCENTRATION = 8.0     # how tightly each prototype pins a statistic
TYPE_PRIOR = {"tight": 0.18, "aggressive": 0.18, "passive": 0.18,
              "balanced": 0.18, "gto_like": 0.18, "unknown": 0.10}


def classify_opponent(model: "OpponentModel") -> dict[str, float]:
    """
    Posterior P(T_i = k | O) ∝ P(T=k) · Π_stats P(counts | T=k)   (Eq. 24).

    Each type's statistic is a Beta(c·μ_k, c·(1−μ_k)) around its prototype,
    so the per-statistic evidence is the Beta-binomial marginal likelihood
    of the *observed counts*.  With few observations every type explains
    the data about equally well and the posterior stays near the prior —
    sample-size awareness falls out of the likelihood (§10.3).  "unknown"
    uses a flat Beta(1,1) for every statistic: it wins when no prototype
    fits.  GTO-like is separated from balanced by rigid bet sizing (a few
    fixed size buckets, i.e. low bet-size entropy).
    """
    stats = {
        "vpip": model.vpip,
        "pfr": model.pfr,
        "postflop_aggression": model.postflop_aggression,
        "fold_to_bet": model.fold_to_bet,
    }
    logp: dict[str, float] = {}
    for t in OPPONENT_TYPES:
        lp = math.log(TYPE_PRIOR[t])
        for name, st in stats.items():
            if t == "unknown":
                lp += st.beta_binomial_loglik(0.5, 2.0)
            else:
                lp += st.beta_binomial_loglik(TYPE_PROTOTYPES[t][name], TYPE_CONCENTRATION)
        logp[t] = lp

    n_sizes = model.bet_sizes.total_samples
    if n_sizes >= 5:
        h = model.bet_sizes.entropy()
        w = min(1.0, n_sizes / 30.0)
        logp["gto_like"] += w * (1.5 if h < 1.2 else -1.5)
        logp["balanced"] += w * (0.8 if h >= 1.2 else -0.8)

    # "unknown" gets a head start that decays with total evidence, so a
    # data-poor model reads unknown and a data-rich one is dominated by the
    # prototype likelihoods (sample-size awareness, §10.3).
    n_eff = min(stats["vpip"].count, stats["postflop_aggression"].count) \
        + 0.25 * model.bet_sizes.total_samples
    logp["unknown"] += 2.5 * math.exp(-n_eff / 8.0)

    mx = max(logp.values())
    ex = {t: math.exp(v - mx) for t, v in logp.items()}
    z = sum(ex.values())
    return {t: ex[t] / z for t in OPPONENT_TYPES}


# ---------------------------------------------------------------------------
# Population prior  (§17)
# ---------------------------------------------------------------------------
@dataclass
class PopulationPrior:
    """
    Population-level prior for each statistic as (μ_pop, κ).
    A statistic's Beta prior is α = κ·μ_pop, β = κ·(1−μ_pop).

    The defaults describe a loose, unknown field.  OpponentModelSet
    re-estimates μ_pop and κ from the opponents actually observed.
    """
    stats: dict = field(default_factory=lambda: {
        "vpip": (0.50, 4.0),
        "pfr": (0.375, 4.0),
        "three_bet": (0.20, 5.0),
        "cbet_flop": (0.57, 3.5),
        "cbet_turn": (0.50, 3.0),
        "cbet_river": (0.33, 3.0),
        "fold_to_cbet": (0.50, 3.0),
        "fold_to_bet": (0.43, 3.5),
        "fold_to_raise": (0.50, 4.0),
        "postflop_aggression": (0.50, 3.0),
        "bluff_frequency": (0.25, 4.0),
        "value_frequency": (0.75, 4.0),
        "went_to_sd_with_winner": (0.50, 3.0),
    })

    def get(self, name: str) -> tuple[float, float]:
        return self.stats.get(name, (0.5, 2.0))

    def _ab(self, name):
        mu, k = self.get(name)
        return k * mu, k * (1 - mu)

    # Backward-compatible α/β accessors
    @property
    def vpip_alpha(self): return self._ab("vpip")[0]
    @property
    def vpip_beta(self): return self._ab("vpip")[1]
    @property
    def pfr_alpha(self): return self._ab("pfr")[0]
    @property
    def pfr_beta(self): return self._ab("pfr")[1]
    @property
    def fold_alpha(self): return self._ab("fold_to_bet")[0]
    @property
    def fold_beta(self): return self._ab("fold_to_bet")[1]


BETA_STAT_NAMES = list(PopulationPrior().stats.keys())


# ---------------------------------------------------------------------------
# Per-opponent model (§10.1, §14)
# ---------------------------------------------------------------------------
STRENGTH_CLASSES = ("weak", "medium", "strong")


def strength_class(s: float) -> int:
    return 0 if s < 0.45 else (1 if s < 0.80 else 2)


class OpponentModel:
    """Learned model M_i = (R_i, Π_i, S_i, B_i, A_i) for one opponent."""

    FOLD_SIZE_KAPPA = 6.0     # pseudo-count tying q̂(B) to overall fold-to-bet

    def __init__(self, player_id: int, prior: Optional[PopulationPrior] = None):
        self.player_id = player_id
        self.hands_observed: int = 0
        self.prior = prior or PopulationPrior()

        for name in BETA_STAT_NAMES:
            st = BetaStat()
            st.set_prior(*self.prior.get(name))
            setattr(self, name, st)

        # fold-to-bet by size bucket (§6, §22)
        self.fold_by_size: list[BetaStat] = [BetaStat() for _ in SIZE_BUCKET_NAMES]
        self.size_seen: list[list[float]] = [[] for _ in SIZE_BUCKET_NAMES]
        self._faced_ref_sum = 0.0
        self._faced_ref_n = 0
        self._refresh_fold_size_priors()

        # conditional action statistics with backoff (§10.2)
        self.actions = ConditionalActionStats()

        # bet sizing (§14) and size given showdown strength (§12.3)
        self.bet_sizes = BetSizeTracker()
        self.size_by_strength = [[0.0] * len(SIZE_BUCKET_NAMES) for _ in STRENGTH_CLASSES]
        # showdown ground truth: (street, strength, was_aggressive, facing)
        self.showdown_samples: list[tuple[int, float, bool, bool]] = []

        # recency / drift (§18)
        self.recent_vpip = EWMAStat(decay=0.90)
        self.recent_aggression = EWMAStat(decay=0.90)
        self.recent_fold_to_bet = EWMAStat(decay=0.90)
        self.recent_decay = 0.90
        self.recent_counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.long_counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.cusums = {"vpip": CUSUM(), "postflop_aggression": CUSUM(),
                       "fold_to_bet": CUSUM()}
        self.changepoints: list[int] = []
        self.changepoint_discount = 0.30

        self._positional_vpip: dict[str, BetaStat] = defaultdict(lambda: BetaStat(1.0, 1.0))

    # ------------------------------------------------------------------
    # Prior management (§17)
    # ------------------------------------------------------------------
    def beta_stats(self) -> dict[str, BetaStat]:
        return {name: getattr(self, name) for name in BETA_STAT_NAMES}

    def apply_prior(self, prior: PopulationPrior):
        self.prior = prior
        for name, st in self.beta_stats().items():
            st.set_prior(*prior.get(name))
        self._refresh_fold_size_priors()

    def _refresh_fold_size_priors(self):
        """Prior for q̂(B) in each size bucket.

        Centre: the balanced defender's fold rate for that size,
        q_ref(B) = B/(P+B) = 1 − MDF(B), shifted by how much this opponent
        over- or under-folds on average relative to q_ref at the sizes they
        have actually faced.  So a nit's large-bet bucket starts high even
        before any large bet has been observed, and the bucket's own data
        takes over as it accumulates (κ = FOLD_SIZE_KAPPA).
        """
        offset = 0.0
        if self._faced_ref_n > 0:
            offset = self.fold_to_bet.mean - self._faced_ref_sum / self._faced_ref_n
        for b, st in enumerate(self.fold_by_size):
            mu = balanced_fold_rate(SIZE_BUCKET_REPR[b]) + offset
            st.set_prior(min(max(mu, 0.02), 0.98), self.FOLD_SIZE_KAPPA)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def _track_kind(self, context: str, kind: str):
        rc = self.recent_counts[context]
        for k in list(rc.keys()):
            rc[k] *= self.recent_decay
        rc[kind] += 1.0
        self.long_counts[context][kind] += 1.0

    def _cusum(self, name: str, x: float):
        st = getattr(self, name)
        if self.cusums[name].update(x, st.mean):
            self._on_changepoint()

    def _on_changepoint(self):
        """Opponent changed strategy: discount old evidence (§18)."""
        self.changepoints.append(self.hands_observed)
        f = self.changepoint_discount
        for st in self.beta_stats().values():
            st.discount(f)
        for st in self.fold_by_size:
            st.discount(f)
        self.actions.discount(f)
        for c in self.cusums.values():
            c.reset()

    def record_preflop_hand(self, position: str, vpip: bool, pfr: bool,
                            faced_raise: bool, three_bet: bool,
                            first_kind: Optional[str] = None):
        """Per-hand preflop accounting (VPIP/PFR are per-hand statistics)."""
        self.vpip.update(vpip)
        self.pfr.update(pfr)
        self.recent_vpip.update(1.0 if vpip else 0.0)
        self._cusum("vpip", 1.0 if vpip else 0.0)
        if faced_raise:
            self.three_bet.update(three_bet)
        self._positional_vpip[position].update(vpip)
        if first_kind:
            self._track_kind("preflop", first_kind)

    def record_preflop_decision(self, cell: tuple, kind: str):
        """One preflop decision into the conditional table."""
        self.actions.record(cell, kind)

    def record_preflop_action(self, action_type: int, position: str,
                              is_voluntary: bool, is_raise: bool,
                              facing_raise: bool):
        """Legacy per-action entry point (one call ≈ one hand's decision).

        The spectator learner uses record_preflop_hand(), which counts VPIP
        once per hand.  This wrapper is kept for manual/scripted updates.
        """
        if not is_voluntary and action_type != ActionType.FOLD:
            return
        kind = "raise" if is_raise else ("call" if is_voluntary else "fold")
        self.record_preflop_hand(position, is_voluntary, is_raise,
                                 facing_raise, facing_raise and is_raise,
                                 first_kind=kind)
        self.actions.record(("preflop", "raised" if facing_raise else "open"), kind)

    def record_postflop_action(self, action_type: int, street: int,
                               bet_amount: float, pot_before: float,
                               is_aggressor: bool, facing_bet: bool,
                               facing_raise: bool,
                               size_fraction: Optional[float] = None,
                               cell: Optional[tuple] = None,
                               cbet_opportunity: Optional[bool] = None):
        """One postflop decision.

        size_fraction: for bets/raises, chips beyond a call over the pot
            after calling (Action.size_fraction).  For decisions facing a
            bet, the *faced* bet's size fraction (drives q̂(B)).
        cell: conditioning cell below the decision-type root.
        cbet_opportunity: first chance to continue-bet as the previous
            street's aggressor (defaults to is_aggressor when unopened).
        """
        is_aggressive = action_type in (ActionType.BET, ActionType.RAISE,
                                        ActionType.ALL_IN)
        kind = action_kind(action_type, is_aggressive, facing_bet)
        dtype = "facing" if facing_bet else "unopened"

        if action_type != ActionType.FOLD:
            self.postflop_aggression.update(is_aggressive)
            self.recent_aggression.update(1.0 if is_aggressive else 0.0)
            self._cusum("postflop_aggression", 1.0 if is_aggressive else 0.0)

        self.actions.record((dtype,) + tuple(cell or (int(street),)), kind)
        self._track_kind(dtype, kind)

        if cbet_opportunity is None:
            cbet_opportunity = is_aggressor and not facing_bet
        if cbet_opportunity and not facing_bet:
            {Street.FLOP: self.cbet_flop, Street.TURN: self.cbet_turn}.get(
                street, self.cbet_river).update(is_aggressive)

        if facing_bet:
            folded = action_type == ActionType.FOLD
            self.fold_to_bet.update(folded)
            self.recent_fold_to_bet.update(1.0 if folded else 0.0)
            self._cusum("fold_to_bet", 1.0 if folded else 0.0)
            if facing_raise:
                self.fold_to_raise.update(folded)
            else:
                self.fold_to_cbet.update(folded)
            if size_fraction is not None:
                self._faced_ref_sum += balanced_fold_rate(size_fraction)
                self._faced_ref_n += 1
            self._refresh_fold_size_priors()
            if size_fraction is not None:
                b = size_bucket(size_fraction)
                self.fold_by_size[b].update(folded)
                self.size_seen[b].append(size_fraction)
        elif is_aggressive:
            if size_fraction is None:
                size_fraction = bet_amount / pot_before if pot_before > 0 else 1.0
            self.bet_sizes.record_fraction(street, size_fraction)

    def record_showdown(self, won: bool, was_bluffing: Optional[bool]):
        self.went_to_sd_with_winner.update(won)
        if was_bluffing is not None:
            self.bluff_frequency.update(was_bluffing)
            self.value_frequency.update(not was_bluffing)

    def record_showdown_action(self, street: int, strength: float,
                               was_aggressive: bool, facing: bool,
                               size_fraction: Optional[float] = None):
        """Ground truth from a revealed hand: what they did with what strength."""
        self.showdown_samples.append((int(street), strength, was_aggressive, facing))
        if len(self.showdown_samples) > 500:
            self.showdown_samples = self.showdown_samples[-500:]
        if was_aggressive and size_fraction is not None:
            self.size_by_strength[strength_class(strength)][size_bucket(size_fraction)] += 1.0

    def finish_hand(self):
        self.hands_observed += 1

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def estimated_fold_to_bet(self, size_fraction: Optional[float] = None) -> float:
        if size_fraction is None:
            return self.fold_to_bet.mean
        return self.fold_by_size[size_bucket(size_fraction)].mean

    def fold_stat_for_size(self, size_fraction: float) -> BetaStat:
        return self.fold_by_size[size_bucket(size_fraction)]

    def estimated_fold_to_raise(self) -> float:
        return self.fold_to_raise.mean

    def estimated_cbet(self, street: int) -> float:
        if street == Street.FLOP:
            return self.cbet_flop.mean
        if street == Street.TURN:
            return self.cbet_turn.mean
        return self.cbet_river.mean

    def action_distribution(self, cell: tuple) -> dict[str, float]:
        """Backoff-smoothed P(action | cell); cell[0] is the decision type."""
        acts = {"unopened": UNOPENED_ACTIONS, "facing": FACING_ACTIONS,
                "preflop": PREFLOP_ACTIONS}[cell[0]]
        return self.actions.distribution(cell, acts)

    def size_likelihood(self, size_fraction: float) -> tuple[float, float, float]:
        """Likelihood ratio of a size bucket for (weak, medium, strong)  (§12.3).

        Laplace-smoothed and normalized to average 1; flat (no information)
        until showdowns reveal how this opponent sizes different strengths.
        """
        b = size_bucket(size_fraction)
        k = len(SIZE_BUCKET_NAMES)
        out = [(row[b] + 1.0) / (sum(row) + k) for row in self.size_by_strength]
        m = sum(out) / len(out)
        return tuple(x / m for x in out)

    def threshold_consistency(self, min_samples: int = 12) -> Optional[float]:
        """How well one strength threshold separates bet vs check (§14).

        1.0 = perfectly separated (deterministic threshold policy).
        None until enough showdown samples exist.
        """
        pts = sorted((s, agg) for (_, s, agg, facing) in self.showdown_samples if not facing)
        if len(pts) < min_samples:
            return None
        n = len(pts)
        total_agg = sum(1 for _, a in pts if a)
        best = max(total_agg, n - total_agg)
        below_non = below_agg = 0
        for _, agg in pts:
            if agg:
                below_agg += 1
            else:
                below_non += 1
            best = max(best, below_non + (total_agg - below_agg))
        return best / n

    # ------------------------------------------------------------------
    # Drift (§18)
    # ------------------------------------------------------------------
    def strategy_drift(self) -> float:
        """Jensen–Shannon divergence between recent and long-run behaviour.

        Weighted over decision contexts with enough data; in [0, 1].
        (KL is avoided: asymmetric and infinite on sparse counts.)
        """
        total_w, acc = 0.0, 0.0
        for ctx, support in (("preflop", PREFLOP_ACTIONS),
                             ("unopened", UNOPENED_ACTIONS),
                             ("facing", FACING_ACTIONS)):
            long = self.long_counts.get(ctx)
            if not long or sum(long.values()) < 10:
                continue
            w = min(sum(long.values()), 50.0)
            acc += w * js_divergence(self.recent_counts.get(ctx), long, support, laplace=0.5)
            total_w += w
        return acc / total_w if total_w else 0.0

    # ------------------------------------------------------------------
    # Classification & fingerprint
    # ------------------------------------------------------------------
    def classify(self) -> dict[str, float]:
        return classify_opponent(self)

    def primary_type(self) -> str:
        c = self.classify()
        return max(c, key=c.get)

    def type_entropy(self) -> float:
        return -sum(p * math.log2(p) for p in self.classify().values() if p > 0)

    def fingerprint_vector(self) -> list[float]:
        return [self.vpip.mean, self.pfr.mean, self.postflop_aggression.mean,
                self.fold_to_bet.mean, min(self.bet_sizes.entropy(), 4.0) / 4.0]

    def is_rigid(self) -> bool:
        """Deterministic-policy signature: rigid sizes and a sharp threshold."""
        tc = self.threshold_consistency()
        rigid_size = self.bet_sizes.total_samples >= 10 and self.bet_sizes.entropy() < 0.6
        return rigid_size and (tc is None or tc > 0.9)

    def fingerprint(self) -> dict:
        tc = self.threshold_consistency()
        cls = self.classify()
        return {
            "vpip": round(self.vpip.mean, 3),
            "pfr": round(self.pfr.mean, 3),
            "three_bet": round(self.three_bet.mean, 3),
            "cbet_flop": round(self.cbet_flop.mean, 3),
            "postflop_aggression": round(self.postflop_aggression.mean, 3),
            "fold_to_bet": round(self.fold_to_bet.mean, 3),
            "fold_to_bet_by_size": {SIZE_BUCKET_NAMES[i]: round(s.mean, 3)
                                    for i, s in enumerate(self.fold_by_size)},
            "fold_to_raise": round(self.fold_to_raise.mean, 3),
            "bluff_frequency": round(self.bluff_frequency.mean, 3),
            "bet_size_entropy": round(self.bet_sizes.entropy(), 3),
            "bet_size_common": self.bet_sizes.most_common_size(),
            "threshold_consistency": None if tc is None else round(tc, 3),
            "is_rigid": self.is_rigid(),
            "hands": self.hands_observed,
            "type": max(cls, key=cls.get),
            "type_probs": {k: round(v, 2) for k, v in cls.items()},
            "strategy_drift": round(self.strategy_drift(), 3),
            "changepoints": list(self.changepoints),
            "data_confidence": round(self.vpip.confidence, 2),
        }

    def summary(self) -> str:
        fp = self.fingerprint()
        return (f"P{self.player_id}: {fp['type']}  "
                f"VPIP={fp['vpip']:.0%} PFR={fp['pfr']:.0%} "
                f"Agg={fp['postflop_aggression']:.0%} "
                f"FoldBet={fp['fold_to_bet']:.0%} "
                f"Bluff={fp['bluff_frequency']:.0%} "
                f"({fp['hands']}h)")


# ---------------------------------------------------------------------------
# Model collection with empirical-Bayes population prior (§17, §19)
# ---------------------------------------------------------------------------
def estimate_population_prior(models: list[OpponentModel],
                              base: Optional[PopulationPrior] = None,
                              min_n: float = 5.0) -> PopulationPrior:
    """Method-of-moments estimate of (μ_pop, κ) for every statistic.

    μ_pop is the pooled mean.  The between-opponent variance τ² is the
    observed variance of individual rates minus the expected binomial
    sampling variance; κ = μ(1−μ)/τ² − 1.  Fewer than 3 opponents with
    data → the base prior is kept for that statistic.
    """
    base = base or PopulationPrior()
    out = dict(base.stats)
    for name in BETA_STAT_NAMES:
        data = []
        for m in models:
            st = getattr(m, name)
            n = st.successes + st.failures
            if n >= min_n:
                data.append((st.successes, n))
        if len(data) < 3:
            continue
        mu = sum(k for k, _ in data) / sum(n for _, n in data)
        rates = [k / n for k, n in data]
        mean_r = sum(rates) / len(rates)
        var_obs = sum((r - mean_r) ** 2 for r in rates) / (len(rates) - 1)
        sampling = mu * (1 - mu) * sum(1.0 / n for _, n in data) / len(data)
        tau2 = max(var_obs - sampling, 1e-4)
        kappa = mu * (1 - mu) / tau2 - 1.0
        out[name] = (min(max(mu, 0.02), 0.98), min(max(kappa, 2.0), 200.0))
    return PopulationPrior(stats=out)


class OpponentModelSet:
    """
    An OpponentModel per opponent.  Periodically re-estimates the population
    prior from all opponents (empirical Bayes) and partially pools
    near-duplicate opponents (§19).
    """

    def __init__(self, prior: Optional[PopulationPrior] = None,
                 refresh_every: int = 25, pool_distance: float = 0.12):
        self.base_prior = prior or PopulationPrior()
        self.prior = self.base_prior
        self._models: dict[int, OpponentModel] = {}
        self.refresh_every = refresh_every
        self.pool_distance = pool_distance
        self._hands_since_refresh = 0

    def get(self, player_id: int) -> OpponentModel:
        if player_id not in self._models:
            self._models[player_id] = OpponentModel(player_id, self.prior)
        return self._models[player_id]

    def all_models(self) -> dict[int, OpponentModel]:
        return dict(self._models)

    def on_hand_end(self):
        self._hands_since_refresh += 1
        if self._hands_since_refresh >= self.refresh_every:
            self.refresh_population_prior()
            self._hands_since_refresh = 0

    def refresh_population_prior(self):
        models = list(self._models.values())
        self.prior = estimate_population_prior(models, self.base_prior)
        for m in models:
            group = [o for o in models if o is not m
                     and o.hands_observed >= 30 and m.hands_observed >= 30
                     and self._distance(m, o) < self.pool_distance]
            if group:
                # Near-duplicates inform each other's prior weakly: their
                # evidence counts at ¼ weight, capped at 40 pseudo-counts.
                pooled = dict(self.prior.stats)
                for name in BETA_STAT_NAMES:
                    k = sum(getattr(o, name).successes for o in group)
                    n = sum(getattr(o, name).successes + getattr(o, name).failures
                            for o in group)
                    if n >= 10:
                        pooled[name] = (min(max(k / n, 0.02), 0.98),
                                        min(max(0.25 * n, 2.0), 40.0))
                m.apply_prior(PopulationPrior(stats=pooled))
            else:
                m.apply_prior(self.prior)

    @staticmethod
    def _distance(a: OpponentModel, b: OpponentModel) -> float:
        va, vb = a.fingerprint_vector(), b.fingerprint_vector()
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(va, vb)) / len(va))

    def report(self) -> str:
        lines = ["Opponent Models:"]
        for pid in sorted(self._models):
            lines.append(f"  {self._models[pid].summary()}")
        return "\n".join(lines)

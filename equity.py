"""
Monte Carlo Equity Estimator  (design doc §3–§6)
================================================
Estimates hand equity by simulation.

    Ê = (W + Σ_k T_k / k) / N                      (Eq. 4, Eq. 6)

Corrections relative to the original specification:

* **Multi-way ties split 1/k, not 1/2.**  A tie among k players (hero
  included) returns 1/k of the pot.  The 0.5 factor is heads-up only.
* **Joint dealing with card removal.**  Every trial deals all opponents and
  the runout from the *remaining* deck, so no card can appear twice.  A
  range is represented as concrete combos; any combo blocked by a known
  card (hero, board, dead) has posterior probability exactly zero.
* **Joint, not sequential, range sampling.**  With several ranged opponents
  the target is P(h_1..h_m) ∝ Π_i w_i(h_i) · 1[no shared cards].  Sampling
  opponent 1, filtering opponent 2's range, and so on biases toward the
  first opponent's marginal.  We use rejection sampling (exact) and fall
  back to sequential filtering only if rejection keeps failing.
* **Weighted ranges.**  Ranges may carry weights (the Bayesian posterior),
  and are sampled proportionally instead of being flattened to a list.
* **Precision.**  The standard error √(Ê(1−Ê)/N) is reported with every
  estimate (§3.2).

Also provides the pot-odds, MDF and balanced-bluff arithmetic of §4–§6.
"""

from __future__ import annotations
import bisect
import math
import random
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Union

from hand_evaluator import evaluate, HandRank

Combo = tuple[int, int]
# Accepted range formats: a list of combos (uniform, duplicates add weight),
# a list of (combo, weight) pairs, or a {combo: weight} dict.  None = random.
RangeSpec = Optional[Union[Sequence[Combo], Sequence[tuple[Combo, float]], dict]]


# ---------------------------------------------------------------------------
# Range handling
# ---------------------------------------------------------------------------
def _canon(h) -> Combo:
    a, b = int(h[0]), int(h[1])
    return (a, b) if a < b else (b, a)


def normalize_range(spec: RangeSpec, dead: Iterable[int] = ()) -> Optional[dict[Combo, float]]:
    """Turn any accepted range format into {combo: weight}, removing blocked combos.

    Returns None for "no range" (uniform random hand) and an empty dict when
    every combo in the range is blocked by known cards.
    """
    if spec is None:
        return None
    dead = set(dead)
    out: dict[Combo, float] = {}
    items = spec.items() if isinstance(spec, dict) else spec
    for item in items:
        if isinstance(spec, dict):
            h, w = item
        elif (len(item) == 2 and isinstance(item[0], (tuple, list))):
            h, w = item
        else:
            h, w = item, 1.0
        if w <= 0:
            continue
        h = _canon(h)
        if h[0] == h[1] or h[0] in dead or h[1] in dead:
            continue   # posterior is identically zero on blocked combos
        out[h] = out.get(h, 0.0) + float(w)
    return out


class WeightedRangeSampler:
    """O(log n) sampling of a combo proportional to its weight."""

    __slots__ = ("combos", "cum", "total")

    def __init__(self, weights: dict[Combo, float]):
        self.combos: list[Combo] = []
        self.cum: list[float] = []
        acc = 0.0
        for h, w in weights.items():
            acc += w
            self.combos.append(h)
            self.cum.append(acc)
        self.total = acc

    def __len__(self):
        return len(self.combos)

    def sample(self, rng: random.Random) -> Combo:
        i = bisect.bisect_right(self.cum, rng.random() * self.total)
        return self.combos[min(i, len(self.combos) - 1)]

    def sample_excluding(self, rng: random.Random, used: set[int]) -> Optional[Combo]:
        """Sequential-filter fallback: sample among combos not touching `used`."""
        cands = [(h, (self.cum[i] - (self.cum[i - 1] if i else 0.0)))
                 for i, h in enumerate(self.combos)
                 if h[0] not in used and h[1] not in used]
        if not cands:
            return None
        tot = sum(w for _, w in cands)
        x = rng.random() * tot
        for h, w in cands:
            x -= w
            if x <= 0:
                return h
        return cands[-1][0]


def build_samplers(opponent_ranges, num_opponents: int, known: set[int]):
    """Per-opponent sampler (or None for a uniformly random hand)."""
    samplers: list[Optional[WeightedRangeSampler]] = []
    for i in range(num_opponents):
        spec = opponent_ranges[i] if (opponent_ranges and i < len(opponent_ranges)) else None
        w = normalize_range(spec, known)
        if w is None or not w:
            # No range, or a range fully blocked by known cards: the
            # opponent still holds *some* two cards, so treat as random.
            samplers.append(None)
        else:
            samplers.append(WeightedRangeSampler(w))
    return samplers


def deal_joint(rng: random.Random, samplers, available: list[int],
               max_rejects: int = 64) -> tuple[list[Combo], set[int]]:
    """Deal every opponent a hand jointly, without card collisions.

    Ranged opponents are sampled by rejection from Π w_i(h_i)·1[disjoint]
    (exact).  Random opponents are then dealt uniformly from what is left,
    which leaves the ranged marginals exact because the number of ways to
    deal the random hands does not depend on which disjoint ranged hands
    were chosen.
    """
    ranged = [i for i, s in enumerate(samplers) if s is not None]
    hands: list[Optional[Combo]] = [None] * len(samplers)
    used: set[int] = set()

    if ranged:
        for _ in range(max_rejects):
            used = set()
            ok = True
            for i in ranged:
                h = samplers[i].sample(rng)
                if h[0] in used or h[1] in used:
                    ok = False
                    break
                used.add(h[0]); used.add(h[1])
                hands[i] = h
            if ok:
                break
        else:
            # Rejection kept failing (tiny overlapping ranges): sequential
            # filtering is biased but always terminates.
            used = set()
            for i in ranged:
                h = samplers[i].sample_excluding(rng, used)
                if h is None:
                    rest = [c for c in available if c not in used]
                    h = _canon(rng.sample(rest, 2))
                used.add(h[0]); used.add(h[1])
                hands[i] = h

    randoms = [i for i, s in enumerate(samplers) if s is None]
    if randoms:
        rest = [c for c in available if c not in used]
        picks = rng.sample(rest, 2 * len(randoms))
        for j, i in enumerate(randoms):
            h = _canon((picks[2 * j], picks[2 * j + 1]))
            hands[i] = h
            used.add(h[0]); used.add(h[1])
    return hands, used


# ---------------------------------------------------------------------------
# Showdown sampling (shared by equity and the decision engine)
# ---------------------------------------------------------------------------
@dataclass
class ShowdownTrial:
    """One simulated deal: every opponent's hand and the final hand ranks."""
    opp_hands: list[Combo]
    hero_rank: HandRank
    opp_ranks: list[HandRank]
    board: list[int]

    def hero_share(self, contenders: Optional[Sequence[int]] = None) -> float:
        """Hero's fraction of the pot against the given opponent indices.

        contenders=None means every opponent.  An empty list means nobody
        contests (hero wins everything).  Ties among k players pay 1/k.
        """
        idx = range(len(self.opp_ranks)) if contenders is None else contenders
        k = 1
        for i in idx:
            r = self.opp_ranks[i]
            if r > self.hero_rank:
                return 0.0
            if r == self.hero_rank:
                k += 1
        return 1.0 / k


def sample_showdowns(
    hero_cards: Sequence[int],
    board: Sequence[int],
    num_opponents: int,
    num_simulations: int,
    opponent_ranges=None,
    dead_cards: Optional[Iterable[int]] = None,
    rng: Optional[random.Random] = None,
    fixed_runout: Optional[Sequence[int]] = None,
) -> list[ShowdownTrial]:
    """Draw joint (opponent hands, runout) samples and evaluate them.

    fixed_runout, if given, forces the remaining board cards (used by the
    optional environment layer, §24); trials whose sampled hands collide with
    it are simply dealt around it.
    """
    rng = rng or random.Random()
    hero_cards = list(hero_cards)
    board = list(board)
    known = set(hero_cards) | set(board)
    if dead_cards:
        known |= set(dead_cards)
    runout_fixed = list(fixed_runout or [])[: 5 - len(board)]
    known_plus = known | set(runout_fixed)
    available = [c for c in range(52) if c not in known_plus]
    samplers = build_samplers(opponent_ranges, num_opponents, known_plus)
    n_board = 5 - len(board) - len(runout_fixed)

    trials: list[ShowdownTrial] = []
    for _ in range(num_simulations):
        opp_hands, used = deal_joint(rng, samplers, available)
        if n_board > 0:
            rest = [c for c in available if c not in used]
            run = rng.sample(rest, n_board)
        else:
            run = []
        full_board = board + runout_fixed + run
        hero_rank = evaluate(hero_cards + full_board)
        opp_ranks = [evaluate([h[0], h[1]] + full_board) for h in opp_hands]
        trials.append(ShowdownTrial(opp_hands, hero_rank, opp_ranks, full_board))
    return trials


def monte_carlo_equity(
    hero_cards: list[int],
    board: list[int],
    num_opponents: int = 1,
    num_simulations: int = 10_000,
    opponent_ranges=None,
    dead_cards: Optional[list[int]] = None,
    rng_seed: Optional[int] = None,
) -> dict:
    """
    Estimate hero's equity against `num_opponents` hands.

    Parameters
    ----------
    hero_cards : two-card list
    board : 0–5 community cards already dealt
    num_opponents : number of live opponents
    num_simulations : Monte Carlo trials
    opponent_ranges : optional list (one per opponent) of ranges.  Each range
        may be None (random hand), a list of combos, a list of
        (combo, weight) pairs, or a {combo: weight} dict.
    dead_cards : additional cards known to be unavailable
    rng_seed : for reproducibility

    Returns
    -------
    dict with equity, wins, ties, tie_equity, losses, samples, std_error.
    `ties` counts trials with any split; `tie_equity` is Σ 1/k over them.
    """
    rng = random.Random(rng_seed)
    trials = sample_showdowns(hero_cards, board, num_opponents, num_simulations,
                              opponent_ranges, dead_cards, rng)
    wins = ties = losses = 0
    tie_equity = 0.0
    for t in trials:
        s = t.hero_share()
        if s == 1.0:
            wins += 1
        elif s == 0.0:
            losses += 1
        else:
            ties += 1
            tie_equity += s
    n = max(1, len(trials))
    equity = (wins + tie_equity) / n
    return {
        "equity": round(equity, 4),
        "wins": wins,
        "ties": ties,
        "tie_equity": round(tie_equity, 4),
        "losses": losses,
        "samples": len(trials),
        "std_error": round(standard_error(equity, n), 5),
    }


def standard_error(equity: float, n: int) -> float:
    """√(Ê(1−Ê)/N)  (§3.2)."""
    if n <= 0:
        return float("inf")
    return math.sqrt(max(0.0, equity * (1.0 - equity)) / n)


# ---------------------------------------------------------------------------
# Pot odds / break-even equity  (§4)
# ---------------------------------------------------------------------------
def break_even_equity(call_cost: int, pot: int) -> float:
    """Minimum equity for an immediately profitable call: C / (P + C)  (Eq. 8)."""
    if pot + call_cost == 0:
        return 0.0
    return call_cost / (pot + call_cost)


def call_ev(equity: float, pot: int, call_cost: int) -> float:
    """Simplified call EV = E·(P+C) − C  (Eq. 9)."""
    return equity * (pot + call_cost) - call_cost


# ---------------------------------------------------------------------------
# Fold equity  (§5)
# ---------------------------------------------------------------------------
def bluff_ev(fold_probability: float, pot: int, ev_when_called: float) -> float:
    """EV = q·P + (1−q)·EV_called  (Eq. 11, general form)."""
    return fold_probability * pot + (1 - fold_probability) * ev_when_called


def bet_ev(fold_probability: float, pot: float, bet: float,
           equity_when_called: float) -> float:
    """One-street semi-bluff value with EV_called made explicit  (Eq. 12).

        EV_bet = q·P + (1−q)·[ e·(P+B) − (1−e)·B ]

    With e = 0 this is the pure-bluff value q·P − (1−q)·B.
    """
    q, e = fold_probability, equity_when_called
    return q * pot + (1 - q) * (e * (pot + bet) - (1 - e) * bet)


# ---------------------------------------------------------------------------
# Minimum defense frequency and balanced bluffing  (§6)
# ---------------------------------------------------------------------------
def minimum_defense_frequency(pot: float, bet: float) -> float:
    """MDF = P / (P + B)  (Eq. 15).  Pot-sized bet → 0.5; half-pot → 2/3."""
    if pot + bet <= 0:
        return 1.0
    return pot / (pot + bet)


def balanced_bluff_fraction(pot: float, bet: float) -> float:
    """Bluffs as a share of a balanced betting range: B / (P + 2B)  (Eq. 16).

    Pot-sized → 1/3.  Tends to 1/2 as B → ∞, which is why 0.5 is the
    theoretical ceiling on bluff frequency (Eq. 14).
    """
    if pot + 2 * bet <= 0:
        return 0.0
    return bet / (pot + 2 * bet)


def max_profitable_fold_threshold(pot: float, bet: float) -> float:
    """Fold frequency above which a 0%-equity bluff profits: B / (P + B).

    This is exactly 1 − MDF, i.e. the break-even point of Eq. 12 with e = 0.
    """
    if pot + bet <= 0:
        return 1.0
    return bet / (pot + bet)


# Short aliases used around the codebase
mdf = minimum_defense_frequency
balanced_bluff = balanced_bluff_fraction

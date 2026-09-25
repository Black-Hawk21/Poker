"""
Utility Functions  (design doc §8, §20, §30)
============================================
The decision engine maximizes E[U(outcome)], not chip EV with a bolted-on
risk penalty.  Subtracting an ad-hoc EV_risk term makes the quantity no
longer an expected value; variance is handled properly by the curvature of
U instead (§20).

* ChipUtility     — cash games: U = chips (so E[U] is ordinary chip EV).
* LogUtility      — a concave utility for risk-averse play without a prize
                    structure.
* ICMUtility      — tournaments: Malmuth–Harville independent chip model.
                    Doubling a stack less than doubles prize equity (§8).

All utilities are evaluated on hero's *final* stack for an outcome.  For
ICM the other players' stacks move by the opposite amount, split among the
opponents contesting the pot in proportion to their stacks.
"""

from __future__ import annotations
import math
from functools import lru_cache
from typing import Optional, Sequence


class Utility:
    name = "abstract"

    def value(self, hero_final: float, hero: int,
              stacks_ref: Sequence[float],
              contesting: Sequence[int] = ()) -> float:
        raise NotImplementedError

    def pot_scale(self, hero_now: float, pot: float, hero: int,
                  stacks_ref: Sequence[float], contesting: Sequence[int] = ()) -> float:
        """Utility value of winning the current pot (used to normalize Q)."""
        return abs(self.value(hero_now + pot, hero, stacks_ref, contesting)
                   - self.value(hero_now, hero, stacks_ref, contesting)) or 1.0


class ChipUtility(Utility):
    """U = chips.  E[U(a)] − U(now) is exactly the chip EV of a."""
    name = "chips"

    def value(self, hero_final, hero, stacks_ref, contesting=()):
        return float(hero_final)


class LogUtility(Utility):
    """U = log(1 + chips / scale): concave, penalizes stack-threatening variance."""
    name = "log"

    def __init__(self, scale: float = 100.0):
        self.scale = scale

    def value(self, hero_final, hero, stacks_ref, contesting=()):
        return self.scale * math.log1p(max(0.0, hero_final) / self.scale)


def icm_equities(stacks: Sequence[float], payouts: Sequence[float]) -> list[float]:
    """Malmuth–Harville prize equity for every player.

    P(player j takes the next place) = s_j / Σ remaining stacks.
    Busted players (stack 0) are excluded (they already hold the lowest
    places).  Exact via memoized recursion over subsets of players.
    """
    n = len(stacks)
    live = [i for i in range(n) if stacks[i] > 0]
    pay = list(payouts)[: len(live)]
    st = tuple(float(stacks[i]) for i in live)
    res = [0.0] * n
    for k, i in enumerate(live):
        res[i] = _icm_one(st, tuple(pay), k)
    return res


@lru_cache(maxsize=4096)
def _icm_one(stacks: tuple, payouts: tuple, hero: int) -> float:
    m = len(stacks)

    @lru_cache(maxsize=None)
    def rec(mask: int) -> float:
        place = m - bin(mask).count("1")
        if place >= len(payouts) or not (mask >> hero) & 1:
            return 0.0
        total = sum(stacks[j] for j in range(m) if (mask >> j) & 1)
        if total <= 0:
            return 0.0
        ev = 0.0
        for j in range(m):
            if (mask >> j) & 1:
                p = stacks[j] / total
                gain = payouts[place] if j == hero else 0.0
                ev += p * (gain + rec(mask & ~(1 << j)))
        return ev

    return rec((1 << m) - 1)


class ICMUtility(Utility):
    """Tournament utility: hero's ICM prize equity."""
    name = "icm"

    def __init__(self, payouts: Sequence[float] = (0.5, 0.3, 0.2)):
        self.payouts = tuple(payouts)

    def value(self, hero_final, hero, stacks_ref, contesting=()):
        stacks = [float(s) for s in stacks_ref]
        delta = hero_final - stacks[hero]
        stacks[hero] = max(0.0, hero_final)
        opp = [i for i in (contesting or range(len(stacks)))
               if i != hero and stacks[i] > 0]
        if opp and delta != 0:
            tot = sum(stacks[i] for i in opp)
            for i in opp:
                stacks[i] = max(0.0, stacks[i] - delta * stacks[i] / tot)
        return icm_equities(stacks, self.payouts)[hero]


def make_utility(kind: str = "chips", payouts: Optional[Sequence[float]] = None) -> Utility:
    if kind == "icm":
        return ICMUtility(payouts or (0.5, 0.3, 0.2))
    if kind == "log":
        return LogUtility()
    return ChipUtility()

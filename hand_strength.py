"""
Hand Strength Tables
====================
One strength scale shared by range inference (§11), the action-likelihood
model (§12), the decision engine's opponent-response model (§20) and
showdown analysis (§15).

Every strength is a **percentile in [0, 1]**: the fraction of possible
opponent holdings a combo beats on the current board.  Using percentiles
instead of absolute hand categories matters: on a paired board every
holding "has a pair", so category-based strength (the original code) says
nothing about who is ahead.  Percentiles also make frequency statistics
map directly onto thresholds — an opponent who continues 30% of the time
continues with roughly the top 30% of their range.

* Preflop: all-in equity of each of the 169 hand classes against one random
  hand (precomputed by Monte Carlo, 3,000 deals per class), converted to a
  percentile over the 1,326 combos.
* Postflop: exact made-hand percentile over every combo not blocked by the
  board or known dead cards, plus a draw-potential term on the flop/turn.

Tables are cached per (board, dead cards).
"""

from __future__ import annotations
from functools import lru_cache
from itertools import combinations
from typing import Iterable, Optional

from hand_evaluator import evaluate, RANKS, card_rank, card_suit

Combo = tuple[int, int]

# All-in equity vs a random hand, per hand class (3,000 MC deals each).
PREFLOP_EQUITY: dict[str, float] = {
    "AA": 0.84, "KK": 0.827, "QQ": 0.792, "JJ": 0.772, "TT": 0.745, "99": 0.715, "88": 0.701, "AKs": 0.673,
    "77": 0.67, "AQs": 0.662, "AJs": 0.657, "AKo": 0.65, "A9s": 0.648, "AJo": 0.645, "ATs": 0.644, "66": 0.642,
    "AQo": 0.64, "A8s": 0.634, "KJs": 0.628, "KQs": 0.627, "KTs": 0.625, "ATo": 0.623, "A8o": 0.614, "55": 0.612,
    "A7s": 0.611, "K9s": 0.608, "QJs": 0.608, "KQo": 0.607, "A5s": 0.606, "A9o": 0.601, "A6s": 0.599, "KJo": 0.599,
    "K8s": 0.593, "A7o": 0.587, "A4s": 0.587, "KTo": 0.586, "Q9s": 0.585, "A5o": 0.583, "A4o": 0.577, "QTs": 0.577,
    "QTo": 0.576, "JTs": 0.576, "QJo": 0.574, "A3s": 0.571, "44": 0.57, "K6s": 0.569, "K9o": 0.567, "K7s": 0.567,
    "A2s": 0.566, "K5s": 0.563, "K8o": 0.562, "Q9o": 0.561, "A6o": 0.559, "A3o": 0.556, "K7o": 0.556, "J8s": 0.553,
    "33": 0.547, "Q8s": 0.546, "JTo": 0.546, "Q8o": 0.543, "K3s": 0.542, "J9s": 0.541, "K5o": 0.538, "K4s": 0.536,
    "A2o": 0.535, "K6o": 0.529, "Q7s": 0.528, "Q6s": 0.528, "Q5s": 0.526, "J9o": 0.526, "K4o": 0.525, "J6s": 0.522,
    "T8s": 0.52, "T9s": 0.519, "K3o": 0.517, "K2s": 0.516, "K2o": 0.516, "J8o": 0.516, "Q3s": 0.515, "J7s": 0.514,
    "Q4s": 0.509, "Q6o": 0.507, "T9o": 0.507, "Q2s": 0.505, "98s": 0.505, "Q7o": 0.502, "T7s": 0.498, "Q4o": 0.497,
    "22": 0.495, "Q3o": 0.494, "Q5o": 0.492, "J7o": 0.492, "J5s": 0.492, "T7o": 0.49, "J3s": 0.488, "T8o": 0.487,
    "J4s": 0.483, "96s": 0.483, "T6s": 0.482, "Q2o": 0.481, "J6o": 0.476, "87s": 0.475, "98o": 0.47, "T3s": 0.469,
    "J2s": 0.468, "T4s": 0.467, "86s": 0.467, "J4o": 0.465, "76s": 0.465, "97s": 0.463, "T5s": 0.462, "97o": 0.461,
    "95s": 0.456, "75s": 0.451, "J5o": 0.45, "T4o": 0.449, "J3o": 0.445, "T5o": 0.445, "T6o": 0.442, "96o": 0.442,
    "85s": 0.441, "J2o": 0.44, "65s": 0.439, "T2s": 0.438, "87o": 0.433, "94s": 0.43, "93s": 0.428, "T3o": 0.425,
    "95o": 0.424, "86o": 0.424, "84s": 0.417, "T2o": 0.415, "64s": 0.414, "94o": 0.413, "85o": 0.412, "83s": 0.412,
    "76o": 0.411, "74s": 0.409, "54s": 0.407, "92s": 0.406, "82s": 0.406, "93o": 0.405, "63s": 0.405, "43s": 0.401,
    "65o": 0.398, "92o": 0.392, "84o": 0.39, "75o": 0.39, "62s": 0.388, "52s": 0.388, "53s": 0.385, "64o": 0.383,
    "73s": 0.381, "73o": 0.377, "74o": 0.376, "54o": 0.376, "83o": 0.371, "72s": 0.368, "42s": 0.368, "82o": 0.365,
    "32s": 0.364, "43o": 0.361, "62o": 0.356, "53o": 0.355, "63o": 0.353, "72o": 0.351, "52o": 0.343, "42o": 0.335,
    "32o": 0.328,
}


def hand_class(c1: int, c2: int) -> str:
    """'AKs', 'T9o', '77' ..."""
    r1, r2 = card_rank(c1), card_rank(c2)
    hi, lo = max(r1, r2), min(r1, r2)
    if hi == lo:
        return RANKS[hi] * 2
    return RANKS[hi] + RANKS[lo] + ("s" if card_suit(c1) == card_suit(c2) else "o")


def _build_preflop_percentiles() -> dict[str, float]:
    # weight each class by its combo count (pairs 6, suited 4, offsuit 12)
    def n_combos(name):
        return 6 if len(name) == 2 else (4 if name[2] == "s" else 12)
    items = sorted(PREFLOP_EQUITY.items(), key=lambda kv: kv[1])
    total = sum(n_combos(k) for k, _ in items)
    out, below = {}, 0
    for name, _ in items:
        k = n_combos(name)
        out[name] = (below + 0.5 * k) / total   # mid-rank percentile
        below += k
    return out


PREFLOP_PERCENTILE: dict[str, float] = _build_preflop_percentiles()


def preflop_strength(c1: int, c2: int) -> float:
    """Preflop percentile of a holding (0 = 32o, ≈1 = AA)."""
    return PREFLOP_PERCENTILE[hand_class(c1, c2)]


def preflop_equity(c1: int, c2: int) -> float:
    return PREFLOP_EQUITY[hand_class(c1, c2)]


# ---------------------------------------------------------------------------
# Draw potential (flop / turn)
# ---------------------------------------------------------------------------
def _straight_in(mask: int) -> bool:
    m = (mask << 1) | (1 if mask & (1 << 12) else 0)
    for h in range(13, 3, -1):
        w = 0b11111 << (h - 4)
        if m & w == w:
            return True
    return False


def draw_outs(hole: Iterable[int], board: list[int]) -> int:
    """Approximate clean outs to a flush or straight that uses a hole card."""
    hole = list(hole)
    if len(board) not in (3, 4):
        return 0
    cards = hole + board
    outs = 0
    flush_draw = False
    for s in range(4):
        n_all = sum(1 for c in cards if card_suit(c) == s)
        n_hole = sum(1 for c in hole if card_suit(c) == s)
        if n_all == 4 and n_hole >= 1:
            flush_draw = True
            outs += 9
    mask = 0
    for c in cards:
        mask |= 1 << card_rank(c)
    bmask = 0
    for c in board:
        bmask |= 1 << card_rank(c)
    if not _straight_in(mask):
        ranks = 0
        for r in range(13):
            if not mask & (1 << r) and _straight_in(mask | (1 << r)) \
                    and not _straight_in(bmask | (1 << r)):
                ranks += 1
        outs += 4 * min(ranks, 2) - (2 if flush_draw and ranks else 0)
    return max(0, min(outs, 15))


def draw_equity(hole: Iterable[int], board: list[int]) -> float:
    """Rule of 2 and 4: ≈4% per out with two cards to come, 2% with one."""
    outs = draw_outs(hole, board)
    per = 0.04 if len(board) == 3 else 0.02
    return min(0.6, outs * per)


# ---------------------------------------------------------------------------
# Postflop strength table
# ---------------------------------------------------------------------------
@lru_cache(maxsize=64)
def _strength_table(board: tuple, dead: frozenset) -> dict:
    blocked = set(board) | set(dead)
    avail = [c for c in range(52) if c not in blocked]
    combos = list(combinations(avail, 2))
    b = list(board)
    ranked = [(evaluate([h[0], h[1]] + b), h) for h in combos]
    ranked.sort(key=lambda x: x[0])
    n = len(ranked)
    table: dict[Combo, float] = {}
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ranked[j + 1][0] == ranked[i][0]:
            j += 1
        pct = (i + 0.5 * (j - i + 1)) / n   # mid-rank handles ties
        for k in range(i, j + 1):
            table[ranked[k][1]] = pct
        i = j + 1
    if len(b) < 5:
        for h, made in list(table.items()):
            d = draw_equity(h, b)
            if d > 0 and made < 0.9:
                # a draw is worth its chance of improving to a strong hand
                table[h] = made + d * (0.9 - made)
    return table


def strength_table(board: Iterable[int], dead: Iterable[int] = ()) -> dict[Combo, float]:
    """{combo: strength percentile} for every combo not blocked by board/dead.

    Preflop (empty board) returns preflop percentiles.
    """
    board = tuple(board)
    dead = frozenset(dead) - set(board)
    if not board:
        return _preflop_table(dead)
    return _strength_table(board, dead)


@lru_cache(maxsize=16)
def _preflop_table(dead: frozenset) -> dict:
    avail = [c for c in range(52) if c not in dead]
    return {h: preflop_strength(h[0], h[1]) for h in combinations(avail, 2)}


def hand_strength(hole: Iterable[int], board: Iterable[int],
                  dead: Iterable[int] = ()) -> float:
    """Strength percentile of one holding (hero's own hand, or a shown hand)."""
    h = tuple(sorted(hole))
    board = list(board)
    if not board:
        return preflop_strength(h[0], h[1])
    table = strength_table(board, set(dead) - set(h))
    return table.get(h, 0.5)


def weighted_quantile(values_weights: list[tuple[float, float]], q: float) -> float:
    """q-quantile of a weighted sample [(value, weight), ...]."""
    if not values_weights:
        return 0.5
    vw = sorted(values_weights)
    total = sum(w for _, w in vw)
    if total <= 0:
        return 0.5
    target = q * total
    acc = 0.0
    for v, w in vw:
        acc += w
        if acc >= target:
            return v
    return vw[-1][0]

"""
Hand Evaluator
==============
Evaluates poker hands (5–7 cards) and returns a comparable HandRank.
evaluate() scores 5-7 cards directly from rank counts and suit masks.
evaluate_bruteforce() is the reference definition (best of all C(n,5)
subsets) and is used by the tests to verify evaluate() exactly.

Hand ranks (high = better):
  8  Straight Flush
  7  Four of a Kind
  6  Full House
  5  Flush
  4  Straight
  3  Three of a Kind
  2  Two Pair
  1  One Pair
  0  High Card
"""

from __future__ import annotations
from collections import Counter
from itertools import combinations
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Card representation
# ---------------------------------------------------------------------------
RANKS = "23456789TJQKA"
SUITS = "shdc"  # spades, hearts, diamonds, clubs

RANK_VALUE = {r: i for i, r in enumerate(RANKS)}  # '2'->0 … 'A'->12


def card(rank: str, suit: str) -> int:
    """Encode a card as an integer 0–51.  rank in '23456789TJQKA', suit in 'shdc'."""
    return RANK_VALUE[rank] * 4 + SUITS.index(suit)


def card_from_str(s: str) -> int:
    """Parse 'As', 'Td', '2c' etc."""
    return card(s[0], s[1])


def card_rank(c: int) -> int:
    return c // 4


def card_suit(c: int) -> int:
    return c % 4


def card_str(c: int) -> str:
    return RANKS[card_rank(c)] + SUITS[card_suit(c)]


# ---------------------------------------------------------------------------
# HandRank — a tuple that supports direct comparison
# ---------------------------------------------------------------------------
class HandRank(NamedTuple):
    """Comparable hand ranking.  Higher is better."""
    category: int        # 0=high card … 8=straight flush
    tiebreakers: tuple   # kickers in descending importance


# ---------------------------------------------------------------------------
# Five-card evaluation (core)
# ---------------------------------------------------------------------------
def _eval5(cards: tuple[int, ...]) -> HandRank:
    """Evaluate exactly 5 cards, return HandRank."""
    ranks = sorted((card_rank(c) for c in cards), reverse=True)
    suits = [card_suit(c) for c in cards]

    is_flush = len(set(suits)) == 1

    # Check straight (including A-2-3-4-5 wheel)
    unique = sorted(set(ranks), reverse=True)
    is_straight = False
    straight_high = 0
    if len(unique) == 5:
        if unique[0] - unique[4] == 4:
            is_straight = True
            straight_high = unique[0]
        elif unique == [12, 3, 2, 1, 0]:  # A-2-3-4-5
            is_straight = True
            straight_high = 3  # 5-high straight

    # Count rank frequencies
    freq = Counter(ranks)
    groups = sorted(freq.items(), key=lambda x: (x[1], x[0]), reverse=True)
    counts = [g[1] for g in groups]
    group_ranks = tuple(g[0] for g in groups)

    if is_straight and is_flush:
        return HandRank(8, (straight_high,))
    if counts == [4, 1]:
        return HandRank(7, group_ranks)
    if counts == [3, 2]:
        return HandRank(6, group_ranks)
    if is_flush:
        return HandRank(5, tuple(ranks))
    if is_straight:
        return HandRank(4, (straight_high,))
    if counts == [3, 1, 1]:
        return HandRank(3, group_ranks)
    if counts == [2, 2, 1]:
        return HandRank(2, group_ranks)
    if counts == [2, 1, 1, 1]:
        return HandRank(1, group_ranks)
    return HandRank(0, tuple(ranks))


# ---------------------------------------------------------------------------
# Best-of-N evaluation (5, 6, or 7 cards)
# ---------------------------------------------------------------------------
# Straight windows over a 14-bit rank mask where bit (r+1) is rank r and bit 0
# is the ace played low.  Checked high to low so the first hit is the best.
_STRAIGHT_WINDOWS = [(0b11111 << (h - 4), h - 1) for h in range(13, 3, -1)]


def _straight_high(rank_mask: int) -> int:
    """Highest straight in a 13-bit rank mask, as a rank index, or -1."""
    m = rank_mask << 1
    if rank_mask & (1 << 12):  # ace also plays low
        m |= 1
    for window, high in _STRAIGHT_WINDOWS:
        if m & window == window:
            return high
    return -1


def _top_ranks(rank_mask: int, k: int) -> tuple:
    out = []
    r = 12
    while r >= 0 and len(out) < k:
        if rank_mask & (1 << r):
            out.append(r)
        r -= 1
    return tuple(out)


def evaluate(cards: list[int] | tuple[int, ...]) -> HandRank:
    """Return the best HandRank from 5-7 cards.

    Direct evaluation from rank counts and suit masks instead of scoring all
    C(7,5)=21 subsets.  Produces exactly the same HandRank (category and
    tiebreakers) as the brute-force best-of-subsets definition, which is kept
    as evaluate_bruteforce() and cross-checked in the tests.
    """
    n = len(cards)
    if n < 5:
        raise ValueError(f"Need at least 5 cards, got {n}")
    if n > 7:
        return evaluate_bruteforce(cards)

    counts = [0] * 13
    suit_masks = [0, 0, 0, 0]
    suit_counts = [0, 0, 0, 0]
    rank_mask = 0
    for c in cards:
        r = c >> 2
        s = c & 3
        counts[r] += 1
        suit_masks[s] |= 1 << r
        suit_counts[s] += 1
        rank_mask |= 1 << r

    # With at most 7 cards a flush and a full house / quads are mutually
    # exclusive, so a flush can be scored immediately.
    for s in range(4):
        if suit_counts[s] >= 5:
            sh = _straight_high(suit_masks[s])
            if sh >= 0:
                return HandRank(8, (sh,))
            return HandRank(5, _top_ranks(suit_masks[s], 5))

    quads = trips = -1
    trips2 = -1
    pairs = []
    for r in range(12, -1, -1):
        k = counts[r]
        if k == 4:
            quads = r
        elif k == 3:
            if trips < 0:
                trips = r
            elif trips2 < 0:
                trips2 = r
        elif k == 2:
            pairs.append(r)

    if quads >= 0:
        kicker = _top_ranks(rank_mask & ~(1 << quads), 1)
        return HandRank(7, (quads,) + kicker)
    if trips >= 0 and (trips2 >= 0 or pairs):
        pair_rank = max(trips2, pairs[0] if pairs else -1)
        return HandRank(6, (trips, pair_rank))

    sh = _straight_high(rank_mask)
    if sh >= 0:
        return HandRank(4, (sh,))
    if trips >= 0:
        return HandRank(3, (trips,) + _top_ranks(rank_mask & ~(1 << trips), 2))
    if len(pairs) >= 2:
        p1, p2 = pairs[0], pairs[1]
        kick = _top_ranks(rank_mask & ~(1 << p1) & ~(1 << p2), 1)
        return HandRank(2, (p1, p2) + kick)
    if pairs:
        p = pairs[0]
        return HandRank(1, (p,) + _top_ranks(rank_mask & ~(1 << p), 3))
    return HandRank(0, _top_ranks(rank_mask, 5))


def evaluate_bruteforce(cards: list[int] | tuple[int, ...]) -> HandRank:
    """Reference implementation: best _eval5 over all 5-card subsets."""
    if len(cards) < 5:
        raise ValueError(f"Need at least 5 cards, got {len(cards)}")
    if len(cards) == 5:
        return _eval5(tuple(cards))
    best = None
    for combo in combinations(cards, 5):
        hr = _eval5(combo)
        if best is None or hr > best:
            best = hr
    return best


CATEGORY_NAMES = [
    "High Card", "One Pair", "Two Pair", "Three of a Kind",
    "Straight", "Flush", "Full House", "Four of a Kind", "Straight Flush",
]


def hand_description(hr: HandRank) -> str:
    return CATEGORY_NAMES[hr.category]

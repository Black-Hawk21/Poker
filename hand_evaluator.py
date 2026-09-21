"""
Hand Evaluator
==============
Evaluates poker hands (5–7 cards) and returns a comparable HandRank.
Uses a direct combinatorial approach: for 7 cards, checks all C(7,5)=21
five-card combos and keeps the best.

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
    from collections import Counter
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
def evaluate(cards: list[int] | tuple[int, ...]) -> HandRank:
    """Return the best HandRank from any 5+ card collection."""
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

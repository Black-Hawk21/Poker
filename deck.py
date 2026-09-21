"""
Deck
====
Standard 52-card deck with shuffle and deal operations.
Cards are ints 0–51 (see hand_evaluator for encoding).
"""

from __future__ import annotations
import random
from typing import Optional

NUM_CARDS = 52


class Deck:
    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        self.reset()

    def reset(self):
        self._cards = list(range(NUM_CARDS))
        self._rng.shuffle(self._cards)
        self._idx = 0

    def deal(self, n: int = 1) -> list[int]:
        """Deal n cards off the top."""
        if self._idx + n > NUM_CARDS:
            raise RuntimeError("Deck exhausted")
        dealt = self._cards[self._idx : self._idx + n]
        self._idx += n
        return dealt

    def deal_one(self) -> int:
        return self.deal(1)[0]

    def remove(self, cards: list[int]):
        """Remove specific cards (for Monte Carlo with known cards)."""
        for c in cards:
            if c in self._cards[self._idx:]:
                self._cards.remove(c)

    @property
    def remaining(self) -> int:
        return NUM_CARDS - self._idx

    def shuffle_remaining(self):
        """Re-shuffle only the un-dealt portion."""
        rest = self._cards[self._idx:]
        self._rng.shuffle(rest)
        self._cards[self._idx:] = rest

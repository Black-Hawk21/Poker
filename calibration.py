"""
Calibration Tracking  (design doc §28)
======================================
"A bot that says '65% GTO-like' should be right about 65% of the time."
A mis-calibrated model exploits phantom tendencies, so predictions are
checked against outcomes with a reliability curve.

HeroBot records one binary prediction per revealed opponent at showdown:
the probability its range posterior assigned to "this opponent's hand
beats mine on the final board", against whether it actually did.  The test
harness uses the same tracker for the opponent-type posterior, where the
true type of each synthetic bot is known.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class CalibrationTracker:
    predictions: list[tuple[float, bool]] = field(default_factory=list)

    def record(self, p: float, outcome: bool):
        self.predictions.append((min(max(float(p), 0.0), 1.0), bool(outcome)))

    @property
    def n(self) -> int:
        return len(self.predictions)

    def brier(self) -> float:
        if not self.predictions:
            return float("nan")
        return sum((p - o) ** 2 for p, o in self.predictions) / self.n

    def reliability_curve(self, n_bins: int = 10) -> list[tuple[float, float, int]]:
        """[(mean predicted, observed frequency, count)] per non-empty bin."""
        bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
        for p, o in self.predictions:
            bins[min(int(p * n_bins), n_bins - 1)].append((p, o))
        out = []
        for b in bins:
            if b:
                out.append((sum(p for p, _ in b) / len(b),
                            sum(1 for _, o in b if o) / len(b), len(b)))
        return out

    def ece(self, n_bins: int = 10) -> float:
        """Expected calibration error: Σ_bins (n_b/N)·|mean p − observed|."""
        if not self.predictions:
            return float("nan")
        return sum(c / self.n * abs(mp - fr)
                   for mp, fr, c in self.reliability_curve(n_bins))

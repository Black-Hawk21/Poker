"""
Bayesian Range Inference  (design doc §11)
==========================================
    P(h | A, O, M) ∝ P(A | h, O, M) · P(h | O, M)                  (Eq. 25)
    P(h | A_1..A_t) ∝ P(A_t | h, A_1:t−1) · P(h | A_1:t−1)         (Eq. 26)

Corrections relative to the original code:

* The likelihood P(A | h, s, M) comes from the explicit, per-opponent
  calibrated model in `action_likelihood` (§12) instead of hand-written
  step functions on absolute hand category.
* A range is always a set of **card-consistent combos**.  Combos touching
  hero's cards or the board have weight exactly zero, so blockers shift
  the posterior automatically (§3.2).
* The full weighted posterior is handed to the equity engine.  The old
  `to_combo_list()` sorted by weight and truncated to ~200 entries, which
  silently turned every range into its strongest ~5%.
* Folded opponents are removed from the equity calculation rather than
  being re-entered as random hands.
* Pruning keeps anything above 1e-4 of the maximum weight.  The old 2%
  threshold eliminated bluffs after about three streets of betting — the
  "range narrows too early" failure mode of §29.
* Preflop ranges start from all card-consistent combos and are narrowed by
  the *actions* (with likelihoods calibrated to the opponent's VPIP/PFR per
  cell), rather than by pre-applying VPIP and then applying the action
  again, which double-counted the same evidence.
"""

from __future__ import annotations
import math
from itertools import combinations
from typing import Iterable, Optional

from hand_evaluator import card_rank, card_suit, card_str
from hand_strength import strength_table, PREFLOP_PERCENTILE, RANKS
from game_state import Action, ActionType, Street
from opponent_model import OpponentModel, decision_cell, action_kind
from action_likelihood import ActionLikelihoodModel, DEFAULT_LIKELIHOOD_MODEL

Combo = tuple[int, int]


# ---------------------------------------------------------------------------
# Helpers (kept for backward compatibility)
# ---------------------------------------------------------------------------
def _preflop_hand_class(c1: int, c2: int) -> tuple[int, int, bool]:
    r1, r2 = card_rank(c1), card_rank(c2)
    return (max(r1, r2), min(r1, r2), card_suit(c1) == card_suit(c2))


def _preflop_strength(high: int, low: int, suited: bool) -> float:
    """Preflop percentile of a hand class (see hand_strength)."""
    if high == low:
        name = RANKS[high] * 2
    else:
        name = RANKS[high] + RANKS[low] + ("s" if suited else "o")
    return PREFLOP_PERCENTILE[name]


def all_two_card_combos(exclude: Optional[set[int]] = None) -> list[Combo]:
    exclude = exclude or set()
    return list(combinations([c for c in range(52) if c not in exclude], 2))


# ---------------------------------------------------------------------------
# LiveRange
# ---------------------------------------------------------------------------
class LiveRange:
    """A weighted distribution over card-consistent opponent combos."""

    PRUNE_FRAC = 1e-4

    def __init__(self, hands_weights: Optional[dict[Combo, float]] = None):
        self._weights: dict[Combo, float] = dict(hands_weights or {})
        self.folded = False

    @classmethod
    def uniform(cls, exclude: Optional[set[int]] = None) -> "LiveRange":
        return cls({h: 1.0 for h in all_two_card_combos(exclude)})

    @classmethod
    def from_vpip(cls, vpip: float, exclude: Optional[set[int]] = None,
                  width: float = 0.04, floor: float = 0.01) -> "LiveRange":
        """Range of a player who entered with frequency `vpip`: combos above
        the (1−vpip) preflop percentile, with a smooth edge and a small
        residual weight everywhere (never exactly zero)."""
        t = 1.0 - vpip
        w = {}
        for h in all_two_card_combos(exclude):
            s = _preflop_strength(*_preflop_hand_class(*h))
            x = (s - t) / width
            sig = 1.0 / (1.0 + math.exp(-max(min(x, 60), -60)))
            w[h] = floor + (1 - floor) * sig
        return cls(w)

    # ------------------------------------------------------------------
    # Card removal
    # ------------------------------------------------------------------
    def remove_cards(self, cards: Iterable[int]):
        """Posterior is identically zero on combos blocked by known cards."""
        cs = set(cards)
        if cs:
            self._weights = {h: w for h, w in self._weights.items()
                             if h[0] not in cs and h[1] not in cs}

    # ------------------------------------------------------------------
    # Bayesian update
    # ------------------------------------------------------------------
    def apply_likelihoods(self, lk: dict[Combo, float]):
        self._weights = {h: w * lk.get(h, 0.0) for h, w in self._weights.items()}
        self._prune()

    def update_observed(self, kind: str, decision: str, board: list[int],
                        dead: Iterable[int] = (),
                        model: Optional[OpponentModel] = None,
                        cell: tuple = (),
                        size_fraction: Optional[float] = None,
                        faced_size: Optional[float] = None,
                        likelihood_model: Optional[ActionLikelihoodModel] = None):
        """One Bayesian update (Eq. 26) for an observed action class."""
        if kind == "fold":
            self.folded = True
            self._weights.clear()
            return
        if not self._weights:
            return
        lm = likelihood_model or DEFAULT_LIKELIHOOD_MODEL
        strengths = strength_table(board, dead)
        lk = lm.likelihoods(strengths, self._weights, decision, kind, model,
                            cell, size_fraction, faced_size)
        self.apply_likelihoods(lk)

    def update(self, action_type: int, street: int, board: list[int],
               model: Optional[OpponentModel] = None, pot: int = 0,
               bet_amount: int = 0, facing_bet: bool = False):
        """Legacy postflop update from raw action parameters.

        Note: a postflop FOLD is informative here (it re-weights toward the
        hands that fold) rather than emptying the range, so callers can see
        what the folding range looked like.  RangeTracker drops folded
        opponents from equity separately.
        """
        if street > Street.PREFLOP and not board:
            return
        aggressive = action_type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN)
        kind = action_kind(action_type, aggressive, facing_bet)
        if kind == "fold" and not facing_bet:
            facing_bet = True
        decision = "facing" if facing_bet else "unopened"
        if decision == "unopened" and kind in ("call", "raise"):
            kind = "bet" if kind == "raise" else "check"
        size = (bet_amount / pot) if (aggressive and pot > 0) else None
        lm = DEFAULT_LIKELIHOOD_MODEL
        strengths = strength_table(board, ())
        lk = lm.likelihoods(strengths, self._weights, decision, kind, model,
                            (int(street),), size, None)
        self.apply_likelihoods(lk)

    def update_preflop(self, action_type: int,
                       model: Optional[OpponentModel] = None,
                       is_raise: bool = False, facing_raise: bool = False):
        """Legacy preflop update: FOLD empties the range; otherwise the
        calibrated preflop likelihood is applied."""
        if action_type == ActionType.FOLD:
            self.folded = True
            self._weights.clear()
            return
        kind = "raise" if (is_raise or action_type in (
            ActionType.RAISE, ActionType.BET, ActionType.ALL_IN)) else (
            "check" if action_type == ActionType.CHECK else "call")
        cell = ("raised" if facing_raise else "open",)
        self.update_observed(kind, "preflop", [], (), model, cell)

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------
    def _prune(self, min_frac: Optional[float] = None):
        if not self._weights:
            return
        mx = max(self._weights.values())
        if mx <= 0:
            self._weights = {}
            return
        thr = mx * (self.PRUNE_FRAC if min_frac is None else min_frac)
        self._weights = {h: w for h, w in self._weights.items() if w >= thr}

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def normalized(self) -> dict[Combo, float]:
        tot = sum(self._weights.values())
        return {h: w / tot for h, w in self._weights.items()} if tot > 0 else {}

    def to_weighted_combos(self, top_n: Optional[int] = None,
                           min_weight_frac: float = 0.0
                           ) -> list[tuple[Combo, float]]:
        """[(combo, normalized weight)] sorted by weight, optionally filtered."""
        if not self._weights:
            return []
        mx = max(self._weights.values())
        items = [(h, w) for h, w in self._weights.items() if w >= mx * min_weight_frac]
        items.sort(key=lambda x: x[1], reverse=True)
        if top_n:
            items = items[:top_n]
        tot = sum(w for _, w in items)
        return [(h, w / tot) for h, w in items] if tot > 0 else []

    def to_combo_list(self, top_n: Optional[int] = None) -> list[tuple[Combo, float]]:
        """The full weighted posterior, in the (combo, weight) format the
        equity engine accepts.  `top_n` is ignored on purpose: truncating a
        range to its heaviest combos biases equity (see module docstring)."""
        return list(self.normalized().items())

    @property
    def size(self) -> int:
        return sum(1 for w in self._weights.values() if w > 0)

    @property
    def total_weight(self) -> float:
        return sum(w for w in self._weights.values() if w > 0)

    def entropy_bits(self) -> float:
        return -sum(p * math.log2(p) for p in self.normalized().values() if p > 0)

    def top_hands(self, n: int = 10) -> list[tuple[str, float]]:
        return [(f"{card_str(h[0])}{card_str(h[1])}", round(w, 4))
                for h, w in self.to_weighted_combos(top_n=n)]


# ---------------------------------------------------------------------------
# Range Tracker — per-opponent live ranges across a hand
# ---------------------------------------------------------------------------
class RangeTracker:
    """
    Maintains a LiveRange for each opponent during one hand.

        tracker = RangeTracker(hero_seat=0, hero_cards=[As, Kd])
        tracker.init_preflop(num_players=2, models={1: opp_model},
                             positions=["BTN", "BB"])
        tracker.update_action(..., action=action)   # Action carries pre-action context
        tracker.update_board(flop)
        ranges = tracker.get_ranges_for_equity()    # active opponents only
    """

    def __init__(self, hero_seat: int, hero_cards: list[int],
                 likelihood_model: Optional[ActionLikelihoodModel] = None):
        self.hero_seat = hero_seat
        self.hero_cards = list(hero_cards)
        self.lm = likelihood_model or DEFAULT_LIKELIHOOD_MODEL
        self._ranges: dict[int, LiveRange] = {}
        self._models: dict[int, OpponentModel] = {}
        self._positions: list[str] = []
        self._board: list[int] = []
        self._excluded = set(hero_cards)
        self._last_aggressor: Optional[int] = None      # current street
        self._prev_street_aggressor: Optional[int] = None
        self._street = 0

    def init_preflop(self, num_players: int,
                     models: Optional[dict[int, OpponentModel]] = None,
                     positions: Optional[list[str]] = None,
                     sitting_out: Iterable[int] = ()):
        self._models = models or {}
        self._positions = list(positions or [])
        out = set(sitting_out)
        for seat in range(num_players):
            if seat == self.hero_seat or seat in out:
                continue
            self._ranges[seat] = LiveRange.uniform(self._excluded)

    def _position(self, seat: int) -> str:
        return self._positions[seat] if seat < len(self._positions) else "MP"

    def _new_street(self, street: int):
        if street != self._street:
            self._prev_street_aggressor = self._last_aggressor
            self._last_aggressor = None
            self._street = street

    def update_action(self, action_seat: int, action_type: int,
                      street: int, board: list[int],
                      pot: int = 0, bet_amount: int = 0,
                      facing_bet: bool = False,
                      action: Optional[Action] = None):
        """Apply a Bayesian update for one opponent action.

        Pass the Action object when available: it carries what the player
        was facing *before* acting (to_call, raises_before, pot_before),
        which the post-action game state cannot tell you.
        """
        self._new_street(int(street))
        if action is not None:
            facing_bet = action.to_call > 0
            raises_before = action.raises_before
            aggressive = action.is_aggressive
            size = action.size_fraction if aggressive else None
            denom = action.pot_before - action.to_call
            faced = (action.to_call / denom) if (facing_bet and denom > 0) else None
        else:
            raises_before = 1 if (facing_bet and street == Street.PREFLOP) else 0
            aggressive = action_type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN)
            size = (bet_amount / pot) if (aggressive and pot > 0) else None
            faced = None

        if aggressive:
            self._last_aggressor = action_seat
        if action_seat == self.hero_seat or action_seat not in self._ranges:
            return
        lr = self._ranges[action_seat]
        if lr.folded:
            return

        was_agg = self._prev_street_aggressor == action_seat
        decision, cell = decision_cell(street, self._position(action_seat), board,
                                       was_agg, facing_bet, raises_before)
        kind = action_kind(action_type, aggressive, facing_bet)
        if decision == "preflop" and kind == "bet":
            kind = "raise"
        dead = self._excluded | set(board)
        lr.remove_cards(dead)
        lr.update_observed(kind, decision, board, self._excluded,
                           self._models.get(action_seat), cell, size, faced, self.lm)

    def update_board(self, board: list[int]):
        """Board cards cannot be in anyone's hand."""
        self._board = list(board)
        dead = set(board) | self._excluded
        for lr in self._ranges.values():
            lr.remove_cards(dead)

    def mark_folded(self, seat: int):
        if seat in self._ranges:
            self._ranges[seat].folded = True
            self._ranges[seat]._weights.clear()

    def active_seats(self) -> list[int]:
        return [s for s in sorted(self._ranges) if not self._ranges[s].folded]

    def get_ranges_for_equity(self, active_seats: Optional[Iterable[int]] = None
                              ) -> Optional[list[Optional[list[tuple[Combo, float]]]]]:
        """Weighted ranges for the opponents still in the hand, in seat order.

        The list is aligned with `active_seats` (default: every opponent
        that has not folded), so its length equals the number of live
        opponents the equity engine should simulate.
        """
        seats = self.active_seats() if active_seats is None else \
            [s for s in active_seats if s != self.hero_seat]
        if not seats:
            return None
        out = []
        for s in seats:
            lr = self._ranges.get(s)
            out.append(lr.to_combo_list() if (lr and lr.size) else None)
        return out

    def get_range(self, seat: int) -> Optional[LiveRange]:
        return self._ranges.get(seat)

    def range_summary(self) -> dict[int, dict]:
        return {seat: {"size": lr.size, "folded": lr.folded,
                       "entropy_bits": round(lr.entropy_bits(), 2),
                       "top_hands": lr.top_hands(5)}
                for seat, lr in self._ranges.items()}

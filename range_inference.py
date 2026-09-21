"""
Bayesian Range Inference — Section 10
======================================
The central inference problem (Equation 32):

    P(H_opp | O, M_opp)

Given opponent actions and our learned model, what hands could they hold?

Bayes' rule (Equation 33):
    P(h | A, O, M) ∝ P(A | h, O, M) · P(h | O, M)

After multiple actions (Equation 34):
    P(h | A_1,...,A_t) ∝ P(A_t | h, A_{1:t-1}) · P(h | A_{1:t-1})

This produces a live range that narrows as the hand progresses:

    Preflop range  →  observe raise  →  updated range
                   →  observe flop bet →  updated range
                   →  observe turn check → updated range
                   →  final posterior range

The range is a list of (hand, weight) pairs that feeds directly into
the Monte Carlo equity estimator (Equation 6):

    h ~ P(H_opp | O, M_opp)

Key design principles:
  - Maintain probability mass over MANY hands (Section 25.2)
  - Never collapse to a single guessed hand too early
  - Use opponent model stats for action likelihoods
  - Board texture influences which hands take which actions
"""

from __future__ import annotations
from itertools import combinations
from typing import Optional

from hand_evaluator import (
    card_rank, card_suit, evaluate, HandRank, RANK_VALUE, RANKS
)
from game_state import ActionType, Street
from board_texture import analyze_board
from opponent_model import OpponentModel


# ---------------------------------------------------------------------------
# Preflop hand strength table (used for initial range construction)
# ---------------------------------------------------------------------------
def _preflop_hand_class(c1: int, c2: int) -> tuple[int, int, bool]:
    """Return (high_rank, low_rank, suited) for a two-card hand."""
    r1, r2 = card_rank(c1), card_rank(c2)
    suited = card_suit(c1) == card_suit(c2)
    return (max(r1, r2), min(r1, r2), suited)


def _preflop_strength(high: int, low: int, suited: bool) -> float:
    """
    Rough preflop hand strength 0–1.
    Pairs occupy 0.50–0.98. Non-pairs occupy 0.02–0.499.
    Within non-pairs, suited > offsuit, connected > gapped, broadway > low.
    """
    if high == low:
        return 0.50 + (high / 12) * 0.48

    # Non-pair: base from rank sum, scaled to [0, ~0.43]
    base = (high * 2.0 + low) / 36.0 * 0.43
    bonus = 0.0
    if suited:
        bonus += 0.025
    gap = high - low
    if gap <= 2:
        bonus += 0.015
    if high >= 10:
        bonus += 0.015
    return min(0.499, max(0.02, base + bonus))


# ---------------------------------------------------------------------------
# All possible 2-card combos
# ---------------------------------------------------------------------------
def all_two_card_combos(exclude: set[int] = None) -> list[tuple[int, int]]:
    """Generate all C(52,2) = 1326 possible two-card holdings, minus excluded."""
    exclude = exclude or set()
    available = [c for c in range(52) if c not in exclude]
    return list(combinations(available, 2))


# ---------------------------------------------------------------------------
# Live Range — the core data structure
# ---------------------------------------------------------------------------
class LiveRange:
    """
    A probability distribution over possible opponent hands.

    Each hand is a (card1, card2) tuple with an associated weight.
    Weights are un-normalized log-probabilities that get normalized
    when sampling or querying.

    The range narrows as Bayesian updates are applied from observed actions.
    """

    def __init__(self, hands_weights: Optional[dict[tuple[int, int], float]] = None):
        # {(card1, card2): weight}
        self._weights: dict[tuple[int, int], float] = hands_weights or {}

    @classmethod
    def uniform(cls, exclude: set[int] = None) -> "LiveRange":
        """Start with a uniform range over all possible hands."""
        combos = all_two_card_combos(exclude)
        return cls({h: 1.0 for h in combos})

    @classmethod
    def from_vpip(cls, vpip: float, exclude: set[int] = None) -> "LiveRange":
        """
        Construct a preflop range based on opponent's VPIP.
        Hands stronger than the VPIP percentile get full weight;
        hands near the boundary get partial weight.
        """
        combos = all_two_card_combos(exclude)

        # Score every hand and sort
        scored = []
        for h in combos:
            high, low, suited = _preflop_hand_class(h[0], h[1])
            strength = _preflop_strength(high, low, suited)
            scored.append((h, strength))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Top `vpip` fraction gets full weight, next 10% tapers
        n = len(scored)
        cutoff_idx = int(n * vpip)
        taper_idx = int(n * min(vpip + 0.10, 1.0))

        weights = {}
        for i, (hand, strength) in enumerate(scored):
            if i < cutoff_idx:
                weights[hand] = 1.0
            elif i < taper_idx:
                # Linear taper
                frac = 1.0 - (i - cutoff_idx) / max(taper_idx - cutoff_idx, 1)
                weights[hand] = max(0.05, frac)
            else:
                weights[hand] = 0.01  # tiny residual (Section 25.2)

        return cls(weights)

    # ------------------------------------------------------------------
    # Bayesian update (Equations 33–34)
    # ------------------------------------------------------------------
    def update(self, action_type: int, street: int,
               board: list[int], model: Optional[OpponentModel] = None,
               pot: int = 0, bet_amount: int = 0,
               facing_bet: bool = False):
        """
        Apply one Bayesian update based on an observed action.

        P(h | action) ∝ P(action | h) · P(h)

        The action likelihood P(action | h) depends on:
          - hand strength given the board
          - opponent model stats (aggression, bluff freq, etc.)
          - bet sizing relative to pot
        """
        if not self._weights or not board and street > Street.PREFLOP:
            return  # nothing to update post-flop without a board

        new_weights = {}
        for hand, prior_w in self._weights.items():
            if prior_w <= 0:
                continue

            likelihood = self._action_likelihood(
                hand, action_type, street, board, model,
                pot, bet_amount, facing_bet
            )
            new_weights[hand] = prior_w * likelihood

        self._weights = new_weights
        self._prune()

    def _action_likelihood(
        self, hand: tuple[int, int], action_type: int, street: int,
        board: list[int], model: Optional[OpponentModel],
        pot: int, bet_amount: int, facing_bet: bool,
    ) -> float:
        """
        Estimate P(action | hand, state, model).

        Strong hands are more likely to bet/raise.
        Weak hands are more likely to fold or check.
        Draws may call or semi-bluff.
        """
        # Evaluate hand strength if we have a board
        if board:
            try:
                hr = evaluate(list(hand) + board)
                # Normalize category to 0-1 strength
                strength = (hr.category + hr.tiebreakers[0] / 13.0) / 8.5
            except Exception:
                strength = 0.5
        else:
            # Preflop
            high, low, suited = _preflop_hand_class(hand[0], hand[1])
            strength = _preflop_strength(high, low, suited)

        # Get model-based stats if available
        if model:
            aggression = model.postflop_aggression.mean
            bluff_freq = model.bluff_frequency.mean
            fold_rate = model.fold_to_bet.mean
        else:
            aggression = 0.5
            bluff_freq = 0.25
            fold_rate = 0.40

        # Compute action likelihoods based on hand strength
        if action_type == ActionType.FOLD:
            # Strong hands almost never fold; weak hands often fold
            return max(0.01, (1.0 - strength) * 0.8 + 0.1)

        elif action_type == ActionType.CHECK:
            # Medium hands check; very strong may trap; very weak may give up
            if strength > 0.8:
                return 0.3  # sometimes traps
            elif strength > 0.4:
                return 0.6  # often checks medium hands
            else:
                return 0.5  # weak hands check when no bet to face

        elif action_type == ActionType.CALL:
            # Calling range: medium-to-strong hands, draws
            if strength > 0.7:
                return 0.5  # strong hands sometimes just call (slow-play)
            elif strength > 0.3:
                return 0.7  # bread-and-butter calling range
            else:
                return 0.15 + bluff_freq * 0.3  # weak hands rarely call

        elif action_type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN):
            # Betting/raising: polarized — very strong OR bluffs
            if strength > 0.75:
                # Value bet: strong hands bet for value
                return 0.7 + aggression * 0.2
            elif strength > 0.5:
                # Medium hands: sometimes bets, less likely to raise
                if action_type == ActionType.RAISE:
                    return 0.2
                return 0.3 + aggression * 0.2
            elif strength > 0.25:
                # Draws / semi-bluffs
                if board:
                    bt = analyze_board(board)
                    if bt.flush_draw_possible or bt.straight_draw_possible:
                        return 0.25 + aggression * 0.2  # semi-bluff
                return 0.10 + bluff_freq * 0.3
            else:
                # Pure bluff territory
                return 0.05 + bluff_freq * 0.4

        return 0.3  # default

    # ------------------------------------------------------------------
    # Preflop-specific updates
    # ------------------------------------------------------------------
    def update_preflop(self, action_type: int,
                       model: Optional[OpponentModel] = None,
                       is_raise: bool = False, facing_raise: bool = False):
        """
        Specialized preflop update.
        - FOLD narrows to nothing (hand is gone)
        - CALL keeps medium+ hands
        - RAISE keeps strong hands + some bluffs
        """
        if action_type == ActionType.FOLD:
            self._weights.clear()
            return

        pfr = model.pfr.mean if model else 0.30
        three_bet = model.three_bet.mean if model else 0.10

        new_weights = {}
        for hand, prior_w in self._weights.items():
            if prior_w <= 0:
                continue

            high, low, suited = _preflop_hand_class(hand[0], hand[1])
            strength = _preflop_strength(high, low, suited)

            if action_type in (ActionType.RAISE, ActionType.BET, ActionType.ALL_IN):
                if facing_raise:
                    # 3-bet: very strong range + some bluffs
                    if strength > 0.7:
                        likelihood = 0.8
                    elif strength > 0.5 and suited:
                        likelihood = three_bet * 0.5  # suited bluff 3-bet
                    else:
                        likelihood = 0.02
                else:
                    # Open raise
                    if strength > (1.0 - pfr * 1.2):
                        likelihood = 0.8
                    else:
                        likelihood = 0.05
            elif action_type == ActionType.CALL:
                # Calling range: medium hands that didn't raise
                if strength > 0.35:
                    likelihood = 0.6
                else:
                    likelihood = 0.1
            else:
                likelihood = 0.3

            new_weights[hand] = prior_w * likelihood

        self._weights = new_weights
        self._prune()

    # ------------------------------------------------------------------
    # Pruning — remove negligible hands (Section 25.2 balance)
    # ------------------------------------------------------------------
    def _prune(self, min_frac: float = 0.02):
        """
        Remove hands whose weight is < min_frac of the maximum.
        Keeps probability mass spread (Section 25.2) but trims noise.
        """
        if not self._weights:
            return
        max_w = max(self._weights.values())
        if max_w <= 0:
            return
        threshold = max_w * min_frac
        self._weights = {h: w for h, w in self._weights.items()
                         if w >= threshold}

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def to_weighted_combos(self, top_n: Optional[int] = None,
                           min_weight_frac: float = 0.01
                           ) -> list[tuple[tuple[int, int], float]]:
        """
        Return hands with normalized weights, sorted by weight descending.
        Filters out hands below min_weight_frac of the max weight.
        """
        if not self._weights:
            return []

        max_w = max(self._weights.values())
        if max_w <= 0:
            return []

        threshold = max_w * min_weight_frac
        filtered = [
            (h, w) for h, w in self._weights.items() if w >= threshold
        ]
        filtered.sort(key=lambda x: x[1], reverse=True)

        if top_n:
            filtered = filtered[:top_n]

        # Normalize
        total = sum(w for _, w in filtered)
        if total > 0:
            filtered = [(h, w / total) for h, w in filtered]

        return filtered

    def to_combo_list(self, top_n: int = 200) -> list[tuple[int, int]]:
        """
        Return the top hands as a flat list suitable for equity.monte_carlo_equity().
        The equity estimator samples uniformly from this list, so we include
        hands proportional to their weight by repeating high-weight hands.
        """
        weighted = self.to_weighted_combos(min_weight_frac=0.02)
        if not weighted:
            return []

        # Build a list with repetitions proportional to weight
        # Aim for ~top_n total entries
        max_w = max(w for _, w in weighted)
        result = []
        for hand, w in weighted:
            copies = max(1, int((w / max_w) * 3 + 0.5))
            result.extend([hand] * copies)
            if len(result) >= top_n:
                break

        return result

    @property
    def size(self) -> int:
        """Number of hands with positive weight."""
        return sum(1 for w in self._weights.values() if w > 0)

    @property
    def total_weight(self) -> float:
        return sum(w for w in self._weights.values() if w > 0)

    def top_hands(self, n: int = 10) -> list[tuple[str, float]]:
        """Return top n hands as (description, weight) for debugging."""
        from hand_evaluator import card_str
        weighted = self.to_weighted_combos(top_n=n)
        return [(f"{card_str(h[0])}{card_str(h[1])}", round(w, 4))
                for h, w in weighted]


# ---------------------------------------------------------------------------
# Range Tracker — manages per-opponent live ranges across a hand
# ---------------------------------------------------------------------------
class RangeTracker:
    """
    Maintains a LiveRange for each opponent during a hand.

    Usage:
        tracker = RangeTracker(hero_seat=0, hero_cards=[As, Kd])
        tracker.init_preflop(num_players=2, models={1: opp_model})
        tracker.update_action(action, street, board, pot)
        ranges = tracker.get_ranges()  # for equity estimator
    """

    def __init__(self, hero_seat: int, hero_cards: list[int]):
        self.hero_seat = hero_seat
        self.hero_cards = hero_cards
        self._ranges: dict[int, LiveRange] = {}
        self._models: dict[int, OpponentModel] = {}
        self._excluded = set(hero_cards)

    def init_preflop(self, num_players: int,
                     models: Optional[dict[int, OpponentModel]] = None):
        """
        Create initial ranges for all opponents based on their VPIP.
        Uses population prior if no model is available (Section 16).
        """
        self._models = models or {}

        for seat in range(num_players):
            if seat == self.hero_seat:
                continue

            model = self._models.get(seat)
            if model and model.vpip.count > 5:
                vpip = model.vpip.mean
            else:
                vpip = 0.50  # population prior

            self._ranges[seat] = LiveRange.from_vpip(vpip, self._excluded)

    def update_action(self, action_seat: int, action_type: int,
                      street: int, board: list[int],
                      pot: int = 0, bet_amount: int = 0,
                      facing_bet: bool = False):
        """Apply a Bayesian range update for one opponent's action."""
        if action_seat == self.hero_seat:
            return
        if action_seat not in self._ranges:
            return

        live_range = self._ranges[action_seat]
        model = self._models.get(action_seat)

        if street == Street.PREFLOP:
            is_raise = action_type in (ActionType.RAISE, ActionType.BET,
                                       ActionType.ALL_IN)
            live_range.update_preflop(
                action_type, model,
                is_raise=is_raise, facing_raise=facing_bet
            )
        else:
            # Remove board cards from hands
            board_set = set(board)
            live_range._weights = {
                h: w for h, w in live_range._weights.items()
                if h[0] not in board_set and h[1] not in board_set
            }
            live_range.update(
                action_type, street, board, model,
                pot, bet_amount, facing_bet
            )

    def update_board(self, board: list[int]):
        """Remove board cards from all ranges (they can't be in anyone's hand)."""
        board_set = set(board) | self._excluded
        for seat, lr in self._ranges.items():
            lr._weights = {
                h: w for h, w in lr._weights.items()
                if h[0] not in board_set and h[1] not in board_set
            }

    def get_ranges_for_equity(self) -> Optional[list[list[tuple[int, int]]]]:
        """
        Return opponent ranges formatted for equity.monte_carlo_equity().
        One combo list per opponent, in seat order (skipping hero).
        """
        if not self._ranges:
            return None

        result = []
        for seat in sorted(self._ranges.keys()):
            combo_list = self._ranges[seat].to_combo_list()
            if combo_list:
                result.append(combo_list)
            else:
                result.append(None)

        # If all ranges are None/empty, return None (fall back to uniform)
        if all(r is None for r in result):
            return None
        return result

    def get_range(self, seat: int) -> Optional[LiveRange]:
        return self._ranges.get(seat)

    def range_summary(self) -> dict[int, dict]:
        """Debug summary of all ranges."""
        summary = {}
        for seat, lr in self._ranges.items():
            summary[seat] = {
                "size": lr.size,
                "top_hands": lr.top_hands(5),
            }
        return summary

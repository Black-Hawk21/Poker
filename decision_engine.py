"""
Decision Engine
===============
Implements the core action-value estimation from Section 17:

    Q(s, a) = EV(a | s, M1, ..., Mn)           (Equation 60)

For Phase 2, opponent models M_i are replaced with default assumptions
(population priors). The engine evaluates every legal action and returns
Q-values suitable for the strategy controller's softmax selection.

Action decomposition (Equation 62):
    EV(a) = EV_pot + EV_future + EV_fold_equity - EV_risk

Key components:
  - pot odds & immediate call EV
  - fold equity for bets/raises
  - bet-size evaluation across multiple sizes
  - SPR-aware adjustments
  - position-based adjustments
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional

from game_state import GameState, ActionType, LegalAction, Street, Position
from equity import monte_carlo_equity, break_even_equity, call_ev, bluff_ev
from board_texture import analyze_board, BoardTexture


# ---------------------------------------------------------------------------
# Configuration — tunable constants for Phase 2 (no opponent model yet)
# ---------------------------------------------------------------------------
@dataclass
class EVConfig:
    """Knobs for the EV engine. Will be overridden by opponent models later."""
    # Monte Carlo simulations per decision
    equity_simulations: int = 5_000
    equity_simulations_important: int = 15_000  # for large-pot decisions

    # Default fold-probability estimates (population prior, no opponent model)
    fold_to_bet_default: float = 0.40
    fold_to_raise_default: float = 0.50
    fold_to_allin_default: float = 0.60

    # Fold probability scaling by bet size (fraction of pot)
    # Larger bets → higher fold probability
    fold_size_slope: float = 0.30  # extra fold% per pot-sized bet

    # Position bonus (IP advantage, in BB)
    ip_bonus_bb: float = 0.5

    # SPR thresholds
    low_spr_threshold: float = 4.0
    high_spr_threshold: float = 12.0

    # Bluff threshold: don't bluff if equity > this (you have a value hand)
    bluff_equity_ceiling: float = 0.35
    # Value threshold: consider value betting above this
    value_equity_floor: float = 0.55

    # Risk aversion for tournament play (1.0 = chip-neutral, <1 = risk-averse)
    risk_factor: float = 1.0

    # Bet sizes to evaluate (as fractions of pot)
    bet_sizes: tuple = (0.33, 0.50, 0.75, 1.0)


DEFAULT_CONFIG = EVConfig()


# ---------------------------------------------------------------------------
# ActionEV — what the engine returns for each candidate action
# ---------------------------------------------------------------------------
@dataclass
class ActionEV:
    """Expected value breakdown for a single action."""
    action_type: ActionType
    amount: int           # total chips for this action (0 for fold/check)
    ev: float             # estimated EV in chips
    ev_components: dict   # breakdown for debugging
    label: str = ""       # human-readable description


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------
class DecisionEngine:
    """
    Evaluates Q(s, a) for all legal actions in a given game state.

    Usage:
        engine = DecisionEngine()
        results = engine.evaluate(game_state, hero=0)
        # results is a list of ActionEV sorted by EV descending
    """

    def __init__(self, config: EVConfig = DEFAULT_CONFIG):
        self.config = config

    def evaluate(
        self,
        gs: GameState,
        hero: int,
        opponent_ranges: Optional[list] = None,
        rng_seed: Optional[int] = None,
        exploit_profile: Optional["ExploitProfile"] = None,
    ) -> list[ActionEV]:
        """
        Evaluate all legal actions for `hero` in the current game state.
        Returns a list of ActionEV sorted by EV (best first).

        exploit_profile: Phase 5 per-opponent overrides for fold estimates,
                         bluff/value thresholds, and bet sizing.
        """
        legal = gs.get_legal_actions(hero)
        if not legal:
            return []

        ep = exploit_profile  # shorthand

        # --- compute equity once (reused across all action evaluations) ---
        num_opp = len(gs.active_players) - 1
        if num_opp < 1:
            num_opp = 1

        # Use more simulations for larger pots
        pot_significance = gs.pot / gs.big_blind if gs.big_blind > 0 else 1
        n_sims = (self.config.equity_simulations_important
                  if pot_significance > 20
                  else self.config.equity_simulations)

        eq_result = monte_carlo_equity(
            hero_cards=gs.hole_cards[hero],
            board=gs.board,
            num_opponents=num_opp,
            num_simulations=n_sims,
            opponent_ranges=opponent_ranges,
            rng_seed=rng_seed,
        )
        equity = eq_result["equity"]

        # --- board texture ---
        bt = analyze_board(gs.board)

        # --- SPR ---
        hero_stack = gs.stacks[hero]
        spr = hero_stack / gs.pot if gs.pot > 0 else float('inf')

        # --- position ---
        hero_pos = gs.positions[hero]
        in_position = self._is_in_position(gs, hero)

        # --- evaluate each legal action ---
        results: list[ActionEV] = []
        for la in legal:
            aev = self._evaluate_action(
                gs, hero, la, equity, bt, spr, in_position, hero_stack, ep
            )
            results.append(aev)

        # If BET or RAISE is legal, also evaluate multiple bet sizes
        bet_sizes = ep.preferred_bet_sizes if ep else self.config.bet_sizes
        for la in legal:
            if la.action_type in (ActionType.BET, ActionType.RAISE):
                size_evs = self._evaluate_bet_sizes(
                    gs, hero, la, equity, bt, spr, in_position, hero_stack, ep
                )
                results.extend(size_evs)

        # Sort by EV descending
        results.sort(key=lambda x: x.ev, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Per-action evaluation
    # ------------------------------------------------------------------
    def _evaluate_action(
        self, gs, hero, la: LegalAction, equity, bt, spr, in_position, stack,
        ep=None,
    ) -> ActionEV:
        """Compute EV for a single legal action."""
        pot = gs.pot
        to_call = gs.max_current_bet - gs.current_bets[hero]

        if la.action_type == ActionType.FOLD:
            return ActionEV(
                action_type=ActionType.FOLD,
                amount=0,
                ev=0.0,
                ev_components={"reason": "fold — no further investment"},
                label="Fold",
            )

        elif la.action_type == ActionType.CHECK:
            initiative_discount = 0.90 if not in_position else 0.95
            ev_check = equity * pot * initiative_discount
            if bt.wetness > 0.4 and equity > 0.6:
                ev_check *= 0.85
            return ActionEV(
                action_type=ActionType.CHECK,
                amount=0,
                ev=ev_check,
                ev_components={
                    "equity": equity, "pot": pot,
                    "initiative_discount": initiative_discount,
                    "wetness_penalty": bt.wetness > 0.4 and equity > 0.6,
                },
                label="Check",
            )

        elif la.action_type == ActionType.CALL:
            call_cost = min(to_call, stack)
            ev_immediate = call_ev(equity, pot, call_cost)

            # Phase 5: adjust calling threshold using exploit profile
            call_adjust = ep.call_threshold_adjust if ep else 0.0
            hero_call_boost = ep.hero_call_boost if ep else 0.0

            # Implied odds adjustment
            implied_bonus = 0.0
            if gs.street < Street.RIVER:
                if equity < 0.5 and equity > 0.2:
                    implied_multiplier = min(spr * 0.05, 0.3)
                    implied_bonus = implied_multiplier * call_cost

            reverse_implied = 0.0
            if equity > 0.4 and equity < 0.6 and spr > self.config.high_spr_threshold:
                reverse_implied = 0.1 * call_cost

            pos_bonus = self.config.ip_bonus_bb * gs.big_blind if in_position else 0.0

            # Phase 5: hero call boost from opponent bluff frequency
            call_boost_chips = hero_call_boost * pot

            ev_total = (ev_immediate + implied_bonus - reverse_implied
                        + pos_bonus + call_boost_chips
                        - call_adjust * call_cost)
            ev_total *= self.config.risk_factor

            return ActionEV(
                action_type=ActionType.CALL,
                amount=call_cost,
                ev=ev_total,
                ev_components={
                    "equity": equity, "pot": pot, "call_cost": call_cost,
                    "ev_immediate": ev_immediate,
                    "implied_bonus": implied_bonus,
                    "reverse_implied": reverse_implied,
                    "pos_bonus": pos_bonus,
                    "break_even": break_even_equity(call_cost, pot),
                },
                label=f"Call {call_cost}",
            )

        elif la.action_type in (ActionType.BET, ActionType.RAISE):
            return self._evaluate_bet(
                gs, hero, la, la.min_amount, equity, bt, spr, in_position, stack, ep
            )

        elif la.action_type == ActionType.ALL_IN:
            return self._evaluate_bet(
                gs, hero, la, stack + gs.current_bets[hero],
                equity, bt, spr, in_position, stack, ep
            )

        return ActionEV(
            action_type=la.action_type, amount=0, ev=0.0,
            ev_components={}, label="Unknown"
        )

    # ------------------------------------------------------------------
    # Bet / raise evaluation
    # ------------------------------------------------------------------
    def _evaluate_bet(
        self, gs, hero, la, bet_total, equity, bt, spr, in_position, stack,
        ep=None,
    ) -> ActionEV:
        """
        Evaluate a specific bet/raise amount.
        Phase 5: uses exploit_profile for fold estimates and thresholds.
        """
        pot = gs.pot
        cost = bet_total - gs.current_bets[hero]
        cost = min(cost, stack)

        if pot > 0:
            bet_fraction = cost / pot
        else:
            bet_fraction = 1.0

        # Phase 5: use exploit profile fold estimates when available
        if ep and ep.confidence > 0:
            base_fold = self._exploit_fold_probability(
                la.action_type, bet_fraction, ep
            )
        else:
            base_fold = self._estimate_fold_probability(la.action_type, bet_fraction)

        # Board texture adjustment
        fold_prob = base_fold * (1.0 - 0.2 * bt.wetness)
        fold_prob = max(0.05, min(0.90, fold_prob))

        new_pot = pot + 2 * cost
        ev_when_called = equity * new_pot - cost

        ev_bet = fold_prob * pot + (1 - fold_prob) * ev_when_called

        spr_adj = self._spr_adjustment(spr, equity, la.action_type)
        ev_bet += spr_adj

        if in_position:
            ev_bet += self.config.ip_bonus_bb * gs.big_blind * 0.5

        ev_bet *= self.config.risk_factor

        # Phase 5: bluff/value classification from exploit profile
        bluff_ceil = ep.bluff_equity_ceiling if ep else self.config.bluff_equity_ceiling
        value_floor = ep.value_equity_floor if ep else self.config.value_equity_floor

        if equity < bluff_ceil:
            label_type = "Bluff"
            # Phase 5: cap bluff EV if above bluff frequency cap
            if ep and ep.bluff_frequency_cap < 0.20:
                ev_bet *= 0.7  # discourage bluffing vs calling stations
        elif equity > value_floor:
            label_type = "Value"
        else:
            label_type = "Thin-value"

        action_name = ActionType(la.action_type).name
        pct = int(bet_fraction * 100) if pot > 0 else 0

        return ActionEV(
            action_type=la.action_type,
            amount=bet_total,
            ev=ev_bet,
            ev_components={
                "equity": equity, "pot": pot, "cost": cost,
                "bet_fraction_of_pot": round(bet_fraction, 2),
                "fold_probability": round(fold_prob, 3),
                "ev_when_called": round(ev_when_called, 1),
                "ev_fold_component": round(fold_prob * pot, 1),
                "spr_adjustment": round(spr_adj, 1),
                "bet_type": label_type,
            },
            label=f"{action_name} {bet_total} ({pct}% pot, {label_type})",
        )

    def _evaluate_bet_sizes(
        self, gs, hero, la, equity, bt, spr, in_position, stack, ep=None,
    ) -> list[ActionEV]:
        """Evaluate multiple bet sizes from config or exploit profile."""
        results = []
        pot = gs.pot
        if pot <= 0:
            return results

        sizes = ep.preferred_bet_sizes if ep and ep.confidence > 0.3 else self.config.bet_sizes
        for frac in sizes:
            bet_chips = int(pot * frac)
            # Compute total bet this street
            bet_total = gs.current_bets[hero] + bet_chips

            # Clamp to legal range
            if bet_total < la.min_amount:
                continue  # below minimum
            if bet_total > la.max_amount:
                bet_total = la.max_amount  # cap at all-in

            # Skip if this duplicates the min-amount evaluation
            if bet_total == la.min_amount:
                continue

            aev = self._evaluate_bet(
                gs, hero, la, bet_total, equity, bt, spr, in_position, stack, ep
            )
            results.append(aev)

        return results

    # ------------------------------------------------------------------
    # Fold probability estimation (population prior — Phase 2)
    # ------------------------------------------------------------------
    def _estimate_fold_probability(
        self, action_type: ActionType, bet_fraction: float
    ) -> float:
        """
        Estimate opponent fold probability given bet size.
        Phase 2 uses population defaults; Phase 3+ uses per-opponent models.

        Larger bets → higher fold probability, with diminishing returns.
        """
        if action_type == ActionType.ALL_IN:
            base = self.config.fold_to_allin_default
        elif action_type == ActionType.RAISE:
            base = self.config.fold_to_raise_default
        else:
            base = self.config.fold_to_bet_default

        # Scale by bet size: log curve (diminishing returns on huge bets)
        if bet_fraction > 0:
            size_factor = self.config.fold_size_slope * math.log1p(bet_fraction)
        else:
            size_factor = 0.0

        fold_prob = base + size_factor
        return max(0.05, min(0.90, fold_prob))

    def _exploit_fold_probability(
        self, action_type: ActionType, bet_fraction: float,
        ep: "ExploitProfile",
    ) -> float:
        """
        Phase 5: Fold probability from learned opponent stats.
        Uses the actual observed fold rates from the exploit profile,
        scaled by bet size.
        """
        if action_type == ActionType.ALL_IN:
            base = ep.fold_to_allin
        elif action_type == ActionType.RAISE:
            base = ep.fold_to_raise
        else:
            base = ep.fold_to_bet

        # Scale by bet size (bigger bets → higher fold prob), but
        # anchored to the opponent's actual observed rate
        if bet_fraction > 0:
            size_factor = 0.15 * math.log1p(bet_fraction)
        else:
            size_factor = 0.0

        fold_prob = base + size_factor
        return max(0.05, min(0.90, fold_prob))

    # ------------------------------------------------------------------
    # SPR-aware adjustment (Section 7)
    # ------------------------------------------------------------------
    def _spr_adjustment(
        self, spr: float, equity: float, action_type: ActionType
    ) -> float:
        """
        Adjust EV based on stack-to-pot ratio.

        Low SPR:
          - Strong hands gain (commit stacks profitably)
          - Weak hands lose (can't maneuver, get pot-committed)

        High SPR:
          - Drawing hands gain (room to realize implied odds)
          - Marginal made hands lose (hard to play multi-street)
        """
        adj = 0.0
        if spr < self.config.low_spr_threshold:
            # Low SPR: showdown strength matters most
            if equity > 0.6:
                # Strong hand in low SPR → push value, bonus
                adj = (0.7 - spr / self.config.low_spr_threshold) * 5.0
            elif equity < 0.3:
                # Weak hand in low SPR → bluffs are expensive, penalty
                adj = -(0.7 - spr / self.config.low_spr_threshold) * 3.0

        elif spr > self.config.high_spr_threshold:
            # High SPR: implied odds matter, marginal hands risky
            if equity > 0.4 and equity < 0.6:
                # Marginal hand deep-stacked → reverse implied odds danger
                adj = -2.0
            elif equity < 0.35 and equity > 0.15:
                # Drawing hand deep → implied odds bonus
                adj = 1.5

        return adj

    # ------------------------------------------------------------------
    # Position detection
    # ------------------------------------------------------------------
    def _is_in_position(self, gs: GameState, hero: int) -> bool:
        """
        Determine if hero acts last post-flop among active players.
        In position = informational advantage.
        """
        if gs.street == Street.PREFLOP:
            # Preflop position is complex; approximate: BTN/CO/HJ = late
            return gs.positions[hero] in (Position.BTN, Position.CO, Position.HJ)

        # Post-flop: in position = last to act among non-folded players
        active = gs.active_players
        if not active:
            return False

        # Find who acts last (closest to dealer going clockwise)
        n = gs.num_players
        last_seat = -1
        for offset in range(n, 0, -1):
            seat = (gs.dealer_seat + offset) % n
            if seat in active and not gs.all_in[seat]:
                last_seat = seat
                break

        return hero == last_seat

    # ------------------------------------------------------------------
    # Convenience: best action
    # ------------------------------------------------------------------
    def best_action(
        self,
        gs: GameState,
        hero: int,
        opponent_ranges=None,
        rng_seed=None,
    ) -> ActionEV:
        """Return the single highest-EV action."""
        results = self.evaluate(gs, hero, opponent_ranges, rng_seed)
        if not results:
            return ActionEV(ActionType.FOLD, 0, 0.0, {}, "Fold (no legal actions)")
        return results[0]

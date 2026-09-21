"""
Monte Carlo Equity Estimator
=============================
Estimates hand equity via simulation, per Equation (5) in the design doc:

    Ê = (W + 0.5T) / N

Supports:
  - uniform opponent ranges (default)
  - weighted opponent ranges (for Bayesian range inference in later phases)
  - multi-opponent equity
  - partial boards (flop, turn, or empty)
"""

from __future__ import annotations
import random
from typing import Optional
from hand_evaluator import evaluate, card_rank, HandRank


def monte_carlo_equity(
    hero_cards: list[int],
    board: list[int],
    num_opponents: int = 1,
    num_simulations: int = 10_000,
    opponent_ranges: Optional[list[list[tuple[int, int]]]] = None,
    dead_cards: Optional[list[int]] = None,
    rng_seed: Optional[int] = None,
) -> dict:
    """
    Estimate hero's equity against num_opponents random hands.

    Parameters
    ----------
    hero_cards : two-card list
    board : 0–5 community cards already dealt
    num_opponents : how many opponents to simulate
    num_simulations : Monte Carlo trials
    opponent_ranges : optional per-opponent list of (card1, card2) tuples
                      to sample from instead of the full deck
    dead_cards : cards known to be unavailable (mucked, etc.)
    rng_seed : for reproducibility

    Returns
    -------
    dict with keys: equity, wins, ties, losses, samples
    """
    rng = random.Random(rng_seed)

    known = set(hero_cards) | set(board)
    if dead_cards:
        known |= set(dead_cards)

    # Build the stub deck (all cards not known)
    full_remaining = [c for c in range(52) if c not in known]
    cards_to_deal_board = 5 - len(board)

    wins = 0
    ties = 0
    losses = 0

    for _ in range(num_simulations):
        # --- sample opponent hands ---
        used = set(known)
        opp_hands: list[list[int]] = []

        if opponent_ranges:
            for opp_idx in range(num_opponents):
                r = opponent_ranges[opp_idx] if opp_idx < len(opponent_ranges) else None
                if r:
                    # Sample from provided range, filtering used cards
                    candidates = [(a, b) for a, b in r if a not in used and b not in used]
                    if not candidates:
                        candidates = _random_hand(full_remaining, used, rng)
                        opp_hands.append(list(candidates))
                        used.add(candidates[0])
                        used.add(candidates[1])
                    else:
                        h = rng.choice(candidates)
                        opp_hands.append(list(h))
                        used.add(h[0])
                        used.add(h[1])
                else:
                    h = _random_hand(full_remaining, used, rng)
                    opp_hands.append(list(h))
                    used.add(h[0])
                    used.add(h[1])
        else:
            for _ in range(num_opponents):
                h = _random_hand(full_remaining, used, rng)
                opp_hands.append(list(h))
                used.add(h[0])
                used.add(h[1])

        # --- complete the board ---
        remaining_deck = [c for c in full_remaining if c not in used]
        rng.shuffle(remaining_deck)
        run_out = remaining_deck[:cards_to_deal_board]
        full_board = list(board) + run_out

        # --- evaluate ---
        hero_rank = evaluate(hero_cards + full_board)
        opp_ranks = [evaluate(oh + full_board) for oh in opp_hands]
        best_opp = max(opp_ranks)

        if hero_rank > best_opp:
            wins += 1
        elif hero_rank == best_opp:
            ties += 1
        else:
            losses += 1

    equity = (wins + 0.5 * ties) / num_simulations

    return {
        "equity": round(equity, 4),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "samples": num_simulations,
    }


def _random_hand(
    remaining: list[int], used: set[int], rng: random.Random
) -> tuple[int, int]:
    """Pick two random cards not in `used`."""
    available = [c for c in remaining if c not in used]
    pair = rng.sample(available, 2)
    return (pair[0], pair[1])


# ---------------------------------------------------------------------------
# Pot odds / break-even equity helpers  (Section 4 of the doc)
# ---------------------------------------------------------------------------
def break_even_equity(call_cost: int, pot: int) -> float:
    """Minimum equity needed for an immediately profitable call.
       E_be = C / (P + C)
    """
    if pot + call_cost == 0:
        return 0.0
    return call_cost / (pot + call_cost)


def call_ev(equity: float, pot: int, call_cost: int) -> float:
    """Simplified call EV = E*(P+C) - C."""
    return equity * (pot + call_cost) - call_cost


def bluff_ev(fold_probability: float, pot: int, ev_when_called: float) -> float:
    """EV_bluff = q*P + (1-q)*EV_called   (Equation 11)."""
    return fold_probability * pot + (1 - fold_probability) * ev_when_called

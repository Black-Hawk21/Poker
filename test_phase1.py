#!/usr/bin/env python3
"""
Phase 1 Test Suite
==================
Validates:
  1. Hand evaluator correctness (rankings, tie-breaking, 7-card best-of)
  2. Legal-action generation
  3. Pot & stack accounting
  4. Position assignment
  5. Monte Carlo equity (sanity checks)
"""

import sys
sys.path.insert(0, ".")

from hand_evaluator import (
    card_from_str as c, evaluate, hand_description, HandRank, CATEGORY_NAMES
)
from game_state import GameState, Action, ActionType, Street, Position
from equity import monte_carlo_equity, break_even_equity, call_ev
from deck import Deck

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✓ {name}")
    else:
        failed += 1
        print(f"  ✗ {name}  — {detail}")


# ===================================================================
print("\n═══ HAND EVALUATOR ═══")
# ===================================================================

# --- Basic category detection ---
print("\n  Category detection:")

# High card
hc = evaluate([c("2s"), c("5d"), c("7h"), c("9c"), c("Jd")])
check("High card", hc.category == 0, hand_description(hc))

# One pair
op = evaluate([c("As"), c("Ad"), c("5h"), c("9c"), c("Jd")])
check("One pair", op.category == 1, hand_description(op))

# Two pair
tp = evaluate([c("As"), c("Ad"), c("5h"), c("5c"), c("Jd")])
check("Two pair", tp.category == 2, hand_description(tp))

# Three of a kind
tok = evaluate([c("As"), c("Ad"), c("Ah"), c("5c"), c("Jd")])
check("Three of a kind", tok.category == 3, hand_description(tok))

# Straight
st = evaluate([c("5s"), c("6d"), c("7h"), c("8c"), c("9d")])
check("Straight", st.category == 4, hand_description(st))

# Wheel (A-2-3-4-5)
wh = evaluate([c("As"), c("2d"), c("3h"), c("4c"), c("5d")])
check("Wheel straight", wh.category == 4, hand_description(wh))

# Flush
fl = evaluate([c("2s"), c("5s"), c("7s"), c("9s"), c("Js")])
check("Flush", fl.category == 5, hand_description(fl))

# Full house
fh = evaluate([c("As"), c("Ad"), c("Ah"), c("Kc"), c("Kd")])
check("Full house", fh.category == 6, hand_description(fh))

# Four of a kind
fk = evaluate([c("As"), c("Ad"), c("Ah"), c("Ac"), c("Kd")])
check("Four of a kind", fk.category == 7, hand_description(fk))

# Straight flush
sf = evaluate([c("5s"), c("6s"), c("7s"), c("8s"), c("9s")])
check("Straight flush", sf.category == 8, hand_description(sf))

# Royal flush
rf = evaluate([c("Ts"), c("Js"), c("Qs"), c("Ks"), c("As")])
check("Royal flush (= straight flush)", rf.category == 8, hand_description(rf))


# --- Ordering ---
print("\n  Ordering:")
check("Pair > High card", op > hc)
check("Two pair > Pair", tp > op)
check("Trips > Two pair", tok > tp)
check("Straight > Trips", st > tok)
check("Flush > Straight", fl > st)
check("Full house > Flush", fh > fl)
check("Quads > Full house", fk > fh)
check("Straight flush > Quads", sf > fk)

# --- Tie-breaking ---
print("\n  Tie-breaking:")
pair_aces = evaluate([c("As"), c("Ad"), c("Kh"), c("Qc"), c("Jd")])
pair_kings = evaluate([c("Ks"), c("Kd"), c("Ah"), c("Qc"), c("Jd")])
check("Pair AA > Pair KK", pair_aces > pair_kings)

high_AK = evaluate([c("As"), c("Kd"), c("3h"), c("5c"), c("7d")])
high_AQ = evaluate([c("As"), c("Qd"), c("3h"), c("5c"), c("7d")])
check("A-K high > A-Q high", high_AK > high_AQ)

# --- 7-card evaluation ---
print("\n  7-card best-of-21:")
seven = [c("As"), c("Ks"), c("Qs"), c("Js"), c("Ts"), c("2d"), c("3h")]
r7 = evaluate(seven)
check("7 cards → royal flush found", r7.category == 8)

seven2 = [c("2s"), c("3d"), c("5h"), c("5c"), c("5d"), c("9s"), c("Jh")]
r72 = evaluate(seven2)
check("7 cards → trips found", r72.category == 3)


# ===================================================================
print("\n═══ GAME STATE — HEADS-UP ═══")
# ===================================================================

gs = GameState(
    num_players=2,
    starting_stacks=[1000, 1000],
    small_blind=5,
    big_blind=10,
)
gs.new_hand(
    dealer_seat=0,
    hole_cards=[[c("As"), c("Kd")], [c("Ts"), c("9s")]],
)

print("\n  Blinds & positions:")
check("Pot after blinds = 15", gs.pot == 15, f"pot={gs.pot}")
check("SB posted by BTN (seat 0)", gs.current_bets[0] == 5)
check("BB posted by seat 1", gs.current_bets[1] == 10)
check("Seat 0 = BTN", gs.positions[0] == Position.BTN)
check("Seat 1 = BB", gs.positions[1] == Position.BB)
check("BTN acts first preflop in HU", gs.actor == 0)

# --- Legal actions for BTN preflop (facing BB) ---
legal = gs.get_legal_actions(0)
types = [la.action_type for la in legal]
print("\n  Legal actions (BTN preflop):")
check("Can fold", ActionType.FOLD in types)
check("Can call", ActionType.CALL in types)
check("Can raise", ActionType.RAISE in types)
check("Cannot check (facing BB)", ActionType.CHECK not in types)

# BTN calls
gs.apply_action(Action(player=0, action_type=ActionType.CALL, amount=10, street=Street.PREFLOP))
check("Pot after BTN call = 20", gs.pot == 20, f"pot={gs.pot}")
check("Actor moves to BB (seat 1)", gs.actor == 1)

# BB checks
legal_bb = gs.get_legal_actions(1)
bb_types = [la.action_type for la in legal_bb]
check("BB can check", ActionType.CHECK in bb_types)
gs.apply_action(Action(player=1, action_type=ActionType.CHECK, street=Street.PREFLOP))
check("Street advances to FLOP", gs.street == Street.FLOP, gs.street.name)

# Deal flop
gs.board = [c("Ah"), c("7d"), c("2c")]

# Post-flop: BB (seat 1) acts first
print("\n  Post-flop:")
check("BB acts first post-flop", gs.actor == 1, f"actor={gs.actor}")

# BB bets 10
gs.apply_action(Action(player=1, action_type=ActionType.BET, amount=10, street=Street.FLOP))
check("Pot after BB bet = 30", gs.pot == 30, f"pot={gs.pot}")

# BTN raises to 30 (total this street = 30 chips committed)
gs.apply_action(Action(player=0, action_type=ActionType.RAISE, amount=30, street=Street.FLOP))
check("Pot after BTN raise = 60", gs.pot == 60, f"pot={gs.pot}")

# BB folds
gs.apply_action(Action(player=1, action_type=ActionType.FOLD, street=Street.FLOP))
check("Hand over after fold", gs.hand_over)
check("Only 1 active player", len(gs.active_players) == 1)


# ===================================================================
print("\n═══ GAME STATE — 3-PLAYER ═══")
# ===================================================================

gs3 = GameState(
    num_players=3,
    starting_stacks=[500, 500, 500],
    small_blind=5,
    big_blind=10,
)
gs3.new_hand(
    dealer_seat=0,
    hole_cards=[[c("As"), c("Kd")], [c("Ts"), c("9s")], [c("Jh"), c("Jc")]],
)

print("\n  3-player positions:")
check("Seat 0 = BTN", gs3.positions[0] == Position.BTN)
check("Seat 1 = SB", gs3.positions[1] == Position.SB)
check("Seat 2 = BB", gs3.positions[2] == Position.BB)
check("UTG (seat 0) acts first preflop", gs3.actor == 0)
check("Pot = 15", gs3.pot == 15, f"pot={gs3.pot}")


# ===================================================================
print("\n═══ MONTE CARLO EQUITY ═══")
# ===================================================================

print("\n  Pocket aces vs random (should be ~85%):")
result = monte_carlo_equity(
    hero_cards=[c("As"), c("Ad")],
    board=[],
    num_opponents=1,
    num_simulations=20_000,
    rng_seed=42,
)
eq = result["equity"]
check(f"AA equity ≈ 85% (got {eq:.1%})", 0.78 < eq < 0.92, f"{eq:.4f}")

print("\n  72o vs random (should be ~35%):")
result2 = monte_carlo_equity(
    hero_cards=[c("7s"), c("2d")],
    board=[],
    num_opponents=1,
    num_simulations=20_000,
    rng_seed=42,
)
eq2 = result2["equity"]
check(f"72o equity ≈ 35% (got {eq2:.1%})", 0.28 < eq2 < 0.42, f"{eq2:.4f}")

print("\n  AKs on A-7-2 rainbow (top pair, should be high):")
result3 = monte_carlo_equity(
    hero_cards=[c("As"), c("Ks")],
    board=[c("Ah"), c("7d"), c("2c")],
    num_opponents=1,
    num_simulations=20_000,
    rng_seed=42,
)
eq3 = result3["equity"]
check(f"AK on A72 equity > 85% (got {eq3:.1%})", eq3 > 0.85, f"{eq3:.4f}")

print("\n  Multi-opponent (AA vs 3 randoms, ~64%):")
result4 = monte_carlo_equity(
    hero_cards=[c("As"), c("Ad")],
    board=[],
    num_opponents=3,
    num_simulations=20_000,
    rng_seed=42,
)
eq4 = result4["equity"]
check(f"AA vs 3 ≈ 64% (got {eq4:.1%})", 0.55 < eq4 < 0.72, f"{eq4:.4f}")


# ===================================================================
print("\n═══ POT ODDS / EV HELPERS ═══")
# ===================================================================

be = break_even_equity(50, 100)  # call 50 into pot of 100
check(f"Break-even equity 50/150 ≈ 33% (got {be:.1%})", abs(be - 1/3) < 0.01)

ev = call_ev(equity=0.40, pot=100, call_cost=50)
check(f"Call EV with 40% equity = 10 (got {ev:.1f})", abs(ev - 10) < 0.1)


# ===================================================================
print("\n═══ DECK ═══")
# ===================================================================

d = Deck(seed=1)
hand = d.deal(2)
check("Deal 2 cards", len(hand) == 2)
check("50 remaining", d.remaining == 50)
flop = d.deal(3)
check("Deal flop", len(flop) == 3)
check("47 remaining", d.remaining == 47)


# ===================================================================
print(f"\n{'═'*50}")
print(f"  Results: {passed} passed, {failed} failed")
print(f"{'═'*50}\n")
sys.exit(1 if failed else 0)

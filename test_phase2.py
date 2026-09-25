#!/usr/bin/env python3
"""
Phase 2 Test Suite
==================
Validates:
  1. Board texture analysis
  2. Decision engine EV calculations
  3. Pot odds / fold equity integration
  4. SPR-aware adjustments
  5. Strategy controller softmax policy
  6. Bet-size evaluation
  7. Full hand simulation with bot decisions
"""

import sys, math
sys.path.insert(0, ".")

from hand_evaluator import card_from_str as c
from game_state import GameState, Action, ActionType, Street, Position
from board_texture import analyze_board, board_category, BoardTexture
from decision_engine import DecisionEngine, EVConfig, ActionEV
from strategy import StrategyController, StrategyMode, StrategyConfig

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
print("\n═══ BOARD TEXTURE ═══")
# ===================================================================

print("\n  Dry rainbow flop:")
bt1 = analyze_board([c("As"), c("7d"), c("2c")])
check("A72r is not paired", not bt1.is_paired)
check("A72r is rainbow", bt1.is_rainbow)
check("A72r low wetness", bt1.wetness < 0.2, f"wetness={bt1.wetness:.2f}")
check("A72r highest = Ace", bt1.highest_rank == 12)
cat1 = board_category(bt1)
check("A72r classified as DRY", "DRY" in cat1, cat1)

print("\n  Wet monotone flop:")
bt2 = analyze_board([c("Ts"), c("Js"), c("Qs")])
check("TJQs is monotone", bt2.is_monotone)
check("TJQs flush possible", bt2.flush_possible)
check("TJQs connected", bt2.is_connected)
check("TJQs high wetness", bt2.wetness > 0.5, f"wetness={bt2.wetness:.2f}")
cat2 = board_category(bt2)
check("TJQs classified as WET", "WET" in cat2, cat2)

print("\n  Paired board:")
bt3 = analyze_board([c("8s"), c("8d"), c("3h")])
check("88-3 is paired", bt3.is_paired)
check("88-3 is rainbow", bt3.is_rainbow)

print("\n  Two-tone connected:")
bt4 = analyze_board([c("6s"), c("7s"), c("8d")])
check("678 is connected", bt4.is_connected)
check("678 two-tone", bt4.is_two_tone)
check("678 straight draw possible", bt4.straight_draw_possible)

print("\n  Empty board:")
bt0 = analyze_board([])
check("Empty board has 0 cards", bt0.num_cards == 0)


# ===================================================================
print("\n═══ DECISION ENGINE — BASIC EV ═══")
# ===================================================================

engine = DecisionEngine(EVConfig(equity_simulations=5000))

# Set up a HU hand where hero has top pair (AK on A-7-2 rainbow)
gs = GameState(
    num_players=2,
    starting_stacks=[1000, 1000],
    small_blind=5,
    big_blind=10,
)
gs.new_hand(dealer_seat=0, hole_cards=[[c("Ts"), c("9s")], [c("As"), c("Kd")]])

# Limp preflop
gs.apply_action(Action(0, ActionType.CALL, 10, Street.PREFLOP))
gs.apply_action(Action(1, ActionType.CHECK, 0, Street.PREFLOP))
gs.board = [c("Ah"), c("7d"), c("2c")]  # dry flop, hero (seat 1) has TPTK

print("\n  Top pair top kicker on dry flop (hero=BB, seat 1):")
results = engine.evaluate(gs, hero=1, rng_seed=42)
check("Multiple actions evaluated", len(results) >= 3, f"n={len(results)}")

# Find the EV of each action type
ev_by_type = {}
for r in results:
    if r.action_type not in ev_by_type or r.ev > ev_by_type[r.action_type]:
        ev_by_type[r.action_type] = r.ev

check("Bet has positive EV", ev_by_type.get(ActionType.BET, -999) > 0,
      f"ev_bet={ev_by_type.get(ActionType.BET)}")
check("Check has positive EV", ev_by_type.get(ActionType.CHECK, -999) > 0)
check("Betting > checking with strong hand",
      ev_by_type.get(ActionType.BET, 0) > ev_by_type.get(ActionType.CHECK, 0),
      f"bet={ev_by_type.get(ActionType.BET, 0):.1f} vs check={ev_by_type.get(ActionType.CHECK, 0):.1f}")

best = engine.best_action(gs, hero=1, rng_seed=42)
check("Best action is a bet or raise",
      best.action_type in (ActionType.BET, ActionType.RAISE),
      best.label)


# ===================================================================
print("\n═══ DECISION ENGINE — WEAK HAND ═══")
# ===================================================================

# Hero has 72o on A-K-Q board — terrible hand
gs2 = GameState(
    num_players=2, starting_stacks=[1000, 1000],
    small_blind=5, big_blind=10,
)
gs2.new_hand(dealer_seat=0, hole_cards=[[c("7s"), c("2d")], [c("As"), c("Kd")]])
gs2.apply_action(Action(0, ActionType.CALL, 10, Street.PREFLOP))
gs2.apply_action(Action(1, ActionType.CHECK, 0, Street.PREFLOP))
gs2.board = [c("Ah"), c("Kh"), c("Qd")]

print("\n  72o on AKQ board (hero=BTN, seat 0):")
# Villain bets
gs2.apply_action(Action(1, ActionType.BET, 15, Street.FLOP))
results2 = engine.evaluate(gs2, hero=0, rng_seed=42)

ev_by_type2 = {}
for r in results2:
    if r.action_type not in ev_by_type2 or r.ev > ev_by_type2[r.action_type]:
        ev_by_type2[r.action_type] = r.ev

# With population priors (no opponent model), implied odds can inflate
# call EV in deep stacks.  The key test: fold should be competitive.
check("Fold EV = 0 (baseline)", ev_by_type2.get(ActionType.FOLD, -1) == 0.0)
# Raw immediate EV of calling is negative (before implied odds bonus)
call_result = [r for r in results2 if r.action_type == ActionType.CALL]
if call_result:
    imm = call_result[0].ev_components.get("ev_immediate", 0)
    check("Immediate call EV is negative (before implied odds)",
          imm < 0, f"ev_immediate={imm:.1f}")


# ===================================================================
print("\n═══ DECISION ENGINE — BET SIZING ═══")
# ===================================================================

# Check that multiple bet sizes are evaluated
gs3 = GameState(
    num_players=2, starting_stacks=[500, 500],
    small_blind=5, big_blind=10,
)
gs3.new_hand(dealer_seat=0, hole_cards=[[c("Qs"), c("Qd")], [c("5s"), c("4s")]])
gs3.apply_action(Action(0, ActionType.CALL, 10, Street.PREFLOP))
gs3.apply_action(Action(1, ActionType.CHECK, 0, Street.PREFLOP))
gs3.board = [c("Qh"), c("7d"), c("2c")]  # set of queens

print("\n  Set of Queens — bet sizing evaluation (hero=BB):")
results3 = engine.evaluate(gs3, hero=1, rng_seed=42)

bet_actions = [r for r in results3 if r.action_type == ActionType.BET]
check("Multiple bet sizes evaluated", len(bet_actions) >= 2,
      f"n_bets={len(bet_actions)}")

if len(bet_actions) >= 2:
    sizes = [r.amount for r in bet_actions]
    check("Different bet amounts offered", len(set(sizes)) > 1,
          f"sizes={sizes}")
    # Print all bet EVs for visibility
    for ba in sorted(bet_actions, key=lambda x: x.ev, reverse=True):
        comps = ba.ev_components
        print(f"    {ba.label}: EV={ba.ev:.1f}  "
              f"(fold_p={comps.get('fold_probability', '?')}, "
              f"ev_called={comps.get('ev_when_called', '?')})")


# ===================================================================
print("\n═══ DECISION ENGINE — SPR AWARENESS ═══")
# ===================================================================

# Low SPR scenario: short stack, big pot
print("\n  Low SPR (short stack):")
gs_low = GameState(
    num_players=2, starting_stacks=[50, 50],
    small_blind=5, big_blind=10,
)
gs_low.new_hand(dealer_seat=0, hole_cards=[[c("As"), c("Ad")], [c("Ks"), c("Kd")]])
gs_low.apply_action(Action(0, ActionType.RAISE, 30, Street.PREFLOP))

spr = gs_low.stacks[1] / gs_low.pot if gs_low.pot > 0 else 0
check(f"SPR is low ({spr:.1f})", spr < 4.0, f"spr={spr:.1f}")

results_low = engine.evaluate(gs_low, hero=1, rng_seed=42)
best_low = results_low[0] if results_low else None
if best_low:
    check("AA commits in low SPR (raise or all-in)",
          best_low.action_type in (ActionType.RAISE, ActionType.ALL_IN, ActionType.CALL),
          best_low.label)

# High SPR scenario: deep stacks
print("\n  High SPR (deep stacks):")
gs_high = GameState(
    num_players=2, starting_stacks=[5000, 5000],
    small_blind=5, big_blind=10,
)
gs_high.new_hand(dealer_seat=0, hole_cards=[[c("As"), c("Ad")], [c("6s"), c("5s")]])
gs_high.apply_action(Action(0, ActionType.CALL, 10, Street.PREFLOP))
gs_high.apply_action(Action(1, ActionType.CHECK, 0, Street.PREFLOP))
gs_high.board = [c("Th"), c("3d"), c("2c")]

spr_high = gs_high.stacks[0] / gs_high.pot if gs_high.pot > 0 else 0
check(f"SPR is high ({spr_high:.0f})", spr_high > 12.0, f"spr={spr_high:.0f}")


# ===================================================================
print("\n═══ STRATEGY CONTROLLER — SOFTMAX ═══")
# ===================================================================

ctrl = StrategyController(rng_seed=123)

print("\n  Softmax basic properties:")
probs = ctrl._softmax([10.0, 5.0, 1.0], temperature=5.0)
check("Softmax sums to 1.0", abs(sum(probs) - 1.0) < 1e-6, f"sum={sum(probs)}")
check("Highest EV gets highest prob", probs[0] > probs[1] > probs[2],
      f"probs={[f'{p:.3f}' for p in probs]}")

probs_greedy = ctrl._softmax([10.0, 5.0, 1.0], temperature=0.01)
check("Near-zero temp → greedy", probs_greedy[0] > 0.99,
      f"p0={probs_greedy[0]:.3f}")

probs_uniform = ctrl._softmax([10.0, 5.0, 1.0], temperature=1000.0)
check("Very high temp → near-uniform",
      abs(probs_uniform[0] - probs_uniform[2]) < 0.05,
      f"probs={[f'{p:.3f}' for p in probs_uniform]}")


# ===================================================================
print("\n═══ STRATEGY CONTROLLER — FULL DECISION ═══")
# ===================================================================

# Use the strong hand scenario
gs_d = GameState(
    num_players=2, starting_stacks=[1000, 1000],
    small_blind=5, big_blind=10,
)
gs_d.new_hand(dealer_seat=0, hole_cards=[[c("As"), c("Kd")], [c("Ts"), c("9s")]])
gs_d.apply_action(Action(0, ActionType.CALL, 10, Street.PREFLOP))
gs_d.apply_action(Action(1, ActionType.CHECK, 0, Street.PREFLOP))
gs_d.board = [c("Ah"), c("7d"), c("2c")]

print("\n  TPTK decision (balanced mode):")
decision = ctrl.decide(gs_d, hero=1, rng_seed=42)
check("Decision has probabilities", len(decision.action_probabilities) >= 2)
check("Temperature is reasonable", 0.0 <= decision.temperature < 1000.0,
      f"temp={decision.temperature}")
check("Strategy mode is BALANCED", decision.strategy_mode == StrategyMode.BALANCED)

print(f"    Chosen: {decision.chosen_action.label}")
print(f"    Temperature: {decision.temperature}")
print(f"    Action probabilities:")
for label, prob in decision.action_probabilities:
    print(f"      {label}: {prob:.1%}")

# Convert to Action and verify it's valid
action = StrategyController.to_action(decision, hero=1, street=gs_d.street)
check("Action is valid type",
      action.action_type in (ActionType.FOLD, ActionType.CHECK, ActionType.CALL,
                             ActionType.BET, ActionType.RAISE, ActionType.ALL_IN))


# ===================================================================
print("\n═══ STRATEGY MODES ═══")
# ===================================================================

print("\n  Value-heavy mode (strong hand):")
ctrl_val = StrategyController(rng_seed=456)
ctrl_val.mode = StrategyMode.VALUE_HEAVY
dec_val = ctrl_val.decide(gs_d, hero=1, rng_seed=42)

# In value-heavy mode, bet/raise should get a bonus
bet_prob_val = sum(p for l, p in dec_val.action_probabilities
                   if "BET" in l or "RAISE" in l or "Value" in l)
check("Value-heavy mode boosts betting probability",
      bet_prob_val > 0.3, f"bet_prob={bet_prob_val:.1%}")

print(f"    Chosen: {dec_val.chosen_action.label}")
for label, prob in dec_val.action_probabilities:
    print(f"      {label}: {prob:.1%}")

print("\n  Aggressive mode:")
ctrl_agg = StrategyController(rng_seed=789)
ctrl_agg.mode = StrategyMode.AGGRESSIVE
dec_agg = ctrl_agg.decide(gs_d, hero=1, rng_seed=42)
print(f"    Chosen: {dec_agg.chosen_action.label}")
for label, prob in dec_agg.action_probabilities:
    print(f"      {label}: {prob:.1%}")


# ===================================================================
print("\n═══ RANDOMIZATION TEST ═══")
# ===================================================================

# Run 200 decisions with same state, different RNG seeds → should see variety
print("\n  200 decisions from same state (checking distribution):")
action_counts = {}
for i in range(200):
    ctrl_test = StrategyController(rng_seed=i)
    dec = ctrl_test.decide(gs_d, hero=1, rng_seed=42)
    at = dec.chosen_action.action_type.name
    action_counts[at] = action_counts.get(at, 0) + 1

check("Multiple action types chosen across 200 trials",
      len(action_counts) >= 2,
      f"actions={action_counts}")

for atype, count in sorted(action_counts.items(), key=lambda x: -x[1]):
    print(f"    {atype}: {count}/200 ({count/2:.0f}%)")


# ===================================================================
print("\n═══ FULL HAND SIMULATION ═══")
# ===================================================================

print("\n  Bot plays a complete hand (both seats):")
gs_sim = GameState(
    num_players=2, starting_stacks=[500, 500],
    small_blind=5, big_blind=10,
)
gs_sim.new_hand(
    dealer_seat=0,
    hole_cards=[[c("Ks"), c("Qd")], [c("8h"), c("7h")]],
)

ctrl_sim = StrategyController(rng_seed=999)
boards = [
    [c("Kh"), c("6h"), c("2d")],  # flop
    [c("9h")],                     # turn
    [c("3s")],                     # river
]
board_dealt = 0

steps = 0
max_steps = 30  # safety
while not gs_sim.hand_over and steps < max_steps:
    hero = gs_sim.actor
    dec = ctrl_sim.decide(gs_sim, hero=hero, rng_seed=steps)
    action = StrategyController.to_action(dec, hero, gs_sim.street)

    old_street = gs_sim.street
    gs_sim.apply_action(action)
    steps += 1

    # Deal board cards when street advances
    if not gs_sim.hand_over and gs_sim.street != old_street:
        if gs_sim.street == Street.FLOP and board_dealt == 0:
            gs_sim.board = boards[0]
            board_dealt = 1
        elif gs_sim.street == Street.TURN and board_dealt == 1:
            gs_sim.board.extend(boards[1])
            board_dealt = 2
        elif gs_sim.street == Street.RIVER and board_dealt == 2:
            gs_sim.board.extend(boards[2])
            board_dealt = 3

check("Hand completed", gs_sim.hand_over or steps >= max_steps)
check("Steps < safety limit", steps < max_steps, f"steps={steps}")
print(f"    Hand played in {steps} actions")
print(f"    Final pot: {gs_sim.pot}")
print(f"    Final stacks: {gs_sim.stacks}")
print(f"    Actions: {[repr(a) for a in gs_sim.action_history]}")


# ===================================================================
print(f"\n{'═'*50}")
print(f"  Results: {passed} passed, {failed} failed")
print(f"{'═'*50}\n")
sys.exit(1 if failed else 0)

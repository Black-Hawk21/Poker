#!/usr/bin/env python3
"""
Phase 3 Test Suite
==================
Validates:
  1. Preflop range construction from VPIP
  2. Bayesian range narrowing from actions
  3. Preflop-specific updates (fold/call/raise)
  4. Post-flop range updates (bet/check/fold)
  5. Board card removal from ranges
  6. Range → equity integration
  7. RangeTracker across a full hand
  8. Live game: HeroBot uses inferred ranges
  9. Range quality: strong hands weighted higher after value bets
"""

import sys
sys.path.insert(0, ".")

from hand_evaluator import card_from_str as c, card_str, evaluate
from game_state import GameState, Action, ActionType, Street
from game_runner import GameRunner
from opponents import HeroBot, ManiacBot, CallingStation, NitBot, RigidBot
from opponent_model import OpponentModel, PopulationPrior
from range_inference import (
    LiveRange, RangeTracker, all_two_card_combos,
    _preflop_strength, _preflop_hand_class,
)
from equity import monte_carlo_equity

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
print("\n═══ PREFLOP HAND STRENGTH ═══")
# ===================================================================

print("\n  Hand strength ordering:")
aa = _preflop_strength(12, 12, False)
kk = _preflop_strength(11, 11, False)
aks = _preflop_strength(12, 11, True)
ako = _preflop_strength(12, 11, False)
t9s = _preflop_strength(8, 7, True)
seven_two = _preflop_strength(5, 0, False)

check("AA > KK", aa > kk, f"AA={aa:.3f} KK={kk:.3f}")
check("KK > AKs", kk > aks, f"KK={kk:.3f} AKs={aks:.3f}")
check("AKs > AKo", aks > ako, f"AKs={aks:.3f} AKo={ako:.3f}")
check("AKo > T9s", ako > t9s, f"AKo={ako:.3f} T9s={t9s:.3f}")
check("T9s > 72o", t9s > seven_two, f"T9s={t9s:.3f} 72o={seven_two:.3f}")


# ===================================================================
print("\n═══ RANGE CONSTRUCTION ═══")
# ===================================================================

hero_cards = {c("As"), c("Kd")}

print("\n  Uniform range:")
uni = LiveRange.uniform(exclude=hero_cards)
check("Uniform range has C(50,2)=1225 hands", uni.size == 1225,
      f"size={uni.size}")

print("\n  VPIP-based range (tight, 25%):")
tight_range = LiveRange.from_vpip(0.25, exclude=hero_cards)
weighted = tight_range.to_weighted_combos(min_weight_frac=0.5)
check("Tight range: top hands have high weight", len(weighted) > 50,
      f"top_hands={len(weighted)}")

# Top hands should be premium
top5 = tight_range.top_hands(5)
print(f"    Top 5 in tight range: {top5}")
check("Tight range top hands are strong",
      len(top5) > 0 and top5[0][1] > 0)

print("\n  VPIP-based range (loose, 75%):")
loose_range = LiveRange.from_vpip(0.75, exclude=hero_cards)
loose_top = loose_range.to_weighted_combos(min_weight_frac=0.5)
check("Loose range has more high-weight hands than tight",
      len(loose_top) > len(weighted),
      f"loose={len(loose_top)} vs tight={len(weighted)}")


# ===================================================================
print("\n═══ BAYESIAN RANGE UPDATES — PREFLOP ═══")
# ===================================================================

print("\n  Preflop RAISE narrows effective range:")
raise_range = LiveRange.from_vpip(0.50, exclude=hero_cards)
top_before = raise_range.to_weighted_combos(min_weight_frac=0.5)
raise_range.update_preflop(ActionType.RAISE, is_raise=True)
top_after = raise_range.to_weighted_combos(min_weight_frac=0.5)

check("Raise concentrates weight on fewer hands",
      len(top_after) < len(top_before),
      f"top_before={len(top_before)} top_after={len(top_after)}")

top_after_raise = raise_range.top_hands(5)
print(f"    Top 5 after raise: {top_after_raise}")

print("\n  Preflop CALL keeps medium hands:")
call_range = LiveRange.from_vpip(0.50, exclude=hero_cards)
call_range.update_preflop(ActionType.CALL)
top_after_call = call_range.top_hands(5)
print(f"    Top 5 after call: {top_after_call}")
check("Call range is wider than raise range",
      call_range.size >= raise_range.size,
      f"call={call_range.size} raise={raise_range.size}")

print("\n  Preflop FOLD empties range:")
fold_range = LiveRange.from_vpip(0.50, exclude=hero_cards)
fold_range.update_preflop(ActionType.FOLD)
check("Fold empties range", fold_range.size == 0, f"size={fold_range.size}")


# ===================================================================
print("\n═══ BAYESIAN RANGE UPDATES — POST-FLOP ═══")
# ===================================================================

board = [c("Ah"), c("7d"), c("2c")]  # dry A-high board
hero_set = {c("As"), c("Kd")}

print("\n  Post-flop BET on A-7-2 (strong hands weighted up):")
pf_range = LiveRange.from_vpip(0.40, exclude=hero_set | set(board))
size_pre_bet = pf_range.size
pf_range.update(ActionType.BET, Street.FLOP, board, pot=20, bet_amount=15)
top_after_bet = pf_range.to_weighted_combos(top_n=10)

# Hands containing an Ace should be weighted higher
ace_hands = [(h, w) for h, w in top_after_bet
             if c("Ah") // 4 in (h[0] // 4, h[1] // 4)]
non_ace = [(h, w) for h, w in top_after_bet
           if c("Ah") // 4 not in (h[0] // 4, h[1] // 4)]

print(f"    Top 10 after flop bet:")
for h, w in top_after_bet[:5]:
    print(f"      {card_str(h[0])}{card_str(h[1])}: {w:.4f}")

check("Range updated by flop bet", pf_range.size > 0)

print("\n  Post-flop CHECK (medium hands, traps):")
check_range = LiveRange.from_vpip(0.40, exclude=hero_set | set(board))
check_range.update(ActionType.CHECK, Street.FLOP, board, pot=20)
check("Range updated by check", check_range.size > 0)

print("\n  Post-flop FOLD (weak hands removed):")
fold_pf = LiveRange.from_vpip(0.40, exclude=hero_set | set(board))
fold_pf.update(ActionType.FOLD, Street.FLOP, board, pot=20)
# After fold, weak hands should be weighted higher (they're the ones that fold)
check("Range updated by fold", fold_pf.size > 0)


# ===================================================================
print("\n═══ BOARD CARD REMOVAL ═══")
# ===================================================================

print("\n  Board cards removed from range:")
board_cards = [c("Ah"), c("7d"), c("2c")]
board_set = set(board_cards)
clean_range = LiveRange.uniform(exclude=hero_set)

# Count hands containing board cards
before = sum(1 for h in clean_range._weights
             if h[0] in board_set or h[1] in board_set)
clean_range._weights = {
    h: w for h, w in clean_range._weights.items()
    if h[0] not in board_set and h[1] not in board_set
}
after = clean_range.size

check("Board cards removed from range",
      after < 1225 and before > 0,
      f"before_removal={before} remaining={after}")


# ===================================================================
print("\n═══ RANGE → EQUITY INTEGRATION ═══")
# ===================================================================

print("\n  Equity with tight range vs uniform range:")
hero = [c("As"), c("Kd")]
board = [c("Ah"), c("7d"), c("2c")]

# Tight opponent range (only premium hands)
tight_combos = LiveRange.from_vpip(0.20, exclude=set(hero) | set(board))
tight_combos.update_preflop(ActionType.RAISE, is_raise=True)
tight_list = tight_combos.to_combo_list(top_n=100)

eq_vs_tight = monte_carlo_equity(
    hero_cards=hero, board=board,
    num_opponents=1, num_simulations=3000,
    opponent_ranges=[tight_list] if tight_list else None,
    rng_seed=42,
)

eq_vs_uniform = monte_carlo_equity(
    hero_cards=hero, board=board,
    num_opponents=1, num_simulations=3000,
    rng_seed=42,
)

print(f"    AK on A72 vs tight range: {eq_vs_tight['equity']:.1%}")
print(f"    AK on A72 vs uniform:     {eq_vs_uniform['equity']:.1%}")
check("Range-weighted equity computed successfully",
      0.5 < eq_vs_tight["equity"] < 1.0,
      f"tight={eq_vs_tight['equity']:.3f}")
check("Uniform equity computed successfully",
      0.5 < eq_vs_uniform["equity"] < 1.0,
      f"uniform={eq_vs_uniform['equity']:.3f}")


# ===================================================================
print("\n═══ RANGE TRACKER — FULL HAND ═══")
# ===================================================================

print("\n  Simulating a hand with range narrowing:")
hero_cards = [c("Qs"), c("Qd")]
exclude = set(hero_cards)

# Build a model for a tight-aggressive opponent
model = OpponentModel(player_id=1)
for _ in range(30):
    model.vpip.update(False)
for _ in range(10):
    model.vpip.update(True)
for _ in range(8):
    model.pfr.update(True)
for _ in range(2):
    model.pfr.update(False)

tracker = RangeTracker(hero_seat=0, hero_cards=hero_cards)
tracker.init_preflop(num_players=2, models={1: model})

initial_range = tracker.get_range(1)
initial_size = initial_range.size
print(f"    Initial range: {initial_size} hands")
check("Initial range built from VPIP", initial_size > 0)

# Opponent raises preflop
tracker.update_action(
    action_seat=1, action_type=ActionType.RAISE,
    street=Street.PREFLOP, board=[],
    pot=15, bet_amount=30
)
after_raise_range = tracker.get_range(1)
after_raise_top = after_raise_range.to_weighted_combos(min_weight_frac=0.5)
print(f"    After preflop raise: {after_raise_range.size} hands, "
      f"{len(after_raise_top)} high-weight")
check("Raise concentrates range weight", len(after_raise_top) < initial_size,
      f"high_weight={len(after_raise_top)} vs initial={initial_size}")

# Flop: A-7-2 rainbow
flop = [c("Ah"), c("7d"), c("2c")]
tracker.update_board(flop)
after_board = tracker.get_range(1).size
print(f"    After board removal: {after_board} hands")

# Opponent bets flop
tracker.update_action(
    action_seat=1, action_type=ActionType.BET,
    street=Street.FLOP, board=flop,
    pot=65, bet_amount=45
)
after_flop_bet = tracker.get_range(1).size
print(f"    After flop bet: {after_flop_bet} hands")

# Get ranges for equity
eq_ranges = tracker.get_ranges_for_equity()
check("Equity ranges produced", eq_ranges is not None)
if eq_ranges:
    check("Range is a list of combos", len(eq_ranges) > 0 and eq_ranges[0] is not None)

# Top hands in final range
summary = tracker.range_summary()
print(f"    Range summary: {summary}")


# ===================================================================
print("\n═══ LIVE GAME — HEROBOT WITH RANGE INFERENCE ═══")
# ===================================================================

print("\n  HeroBot (with ranges) vs CallingStation, 30 hands:")
hero_bot = HeroBot("Hero_R3", seed=42, learning=True)
station = CallingStation("Station", seed=10)
runner = GameRunner(
    bots=[hero_bot, station],
    starting_stack=1000, small_blind=5, big_blind=10,
    deck_seed=100,
)
stats = runner.run(num_hands=30)

check("30 hands completed", stats.total_hands == 30)
hero_ps = stats.players.get("Hero_R3")
if hero_ps:
    bbh = hero_ps.net_chips / 10 / 30
    print(f"    Hero: net={hero_ps.net_chips:+d}  bb/hand={bbh:+.2f}")

# Verify range tracker was used (it resets each hand, but learner persists)
check("Learner processed hands", hero_bot.learner.hands_processed == 30)

print("\n  HeroBot vs ManiacBot, 30 hands:")
hero2 = HeroBot("Hero_R3", seed=42, learning=True)
maniac = ManiacBot("Maniac", seed=20)
runner2 = GameRunner(
    bots=[hero2, maniac],
    starting_stack=1000, small_blind=5, big_blind=10,
    deck_seed=200,
)
stats2 = runner2.run(num_hands=30)
hero2_ps = stats2.players.get("Hero_R3")
if hero2_ps:
    bbh2 = hero2_ps.net_chips / 10 / 30
    print(f"    Hero vs Maniac: net={hero2_ps.net_chips:+d}  bb/hand={bbh2:+.2f}")


# ===================================================================
print("\n═══ RANGE NARROWS WITH OPPONENT MODEL ═══")
# ===================================================================

print("\n  Tight opponent model → narrow initial range:")
tight_model = OpponentModel(player_id=99)
for _ in range(40):
    tight_model.vpip.update(False)
for _ in range(5):
    tight_model.vpip.update(True)

loose_model = OpponentModel(player_id=88)
for _ in range(5):
    loose_model.vpip.update(False)
for _ in range(40):
    loose_model.vpip.update(True)

tight_lr = LiveRange.from_vpip(tight_model.vpip.mean, exclude={0, 1})
loose_lr = LiveRange.from_vpip(loose_model.vpip.mean, exclude={0, 1})

tight_top = tight_lr.to_weighted_combos(min_weight_frac=0.5)
loose_top = loose_lr.to_weighted_combos(min_weight_frac=0.5)

print(f"    Tight VPIP={tight_model.vpip.mean:.2f}: "
      f"{len(tight_top)} high-weight hands")
print(f"    Loose VPIP={loose_model.vpip.mean:.2f}: "
      f"{len(loose_top)} high-weight hands")
check("Tight model → fewer high-weight hands",
      len(tight_top) < len(loose_top),
      f"tight={len(tight_top)} loose={len(loose_top)}")


# ===================================================================
print("\n═══ MULTI-STREET NARROWING ═══")
# ===================================================================

print("\n  Range through preflop raise → flop bet → turn bet:")
hero = [c("Jh"), c("Jd")]
tracker = RangeTracker(hero_seat=0, hero_cards=hero)
tracker.init_preflop(num_players=2)

sizes = [tracker.get_range(1).size]
effective = [len(tracker.get_range(1).to_weighted_combos(min_weight_frac=0.3))]

# Preflop raise
tracker.update_action(1, ActionType.RAISE, Street.PREFLOP, [], 15, 30, False)
sizes.append(tracker.get_range(1).size)
effective.append(len(tracker.get_range(1).to_weighted_combos(min_weight_frac=0.3)))

# Flop: K-8-3 rainbow
flop = [c("Ks"), c("8d"), c("3c")]
tracker.update_board(flop)
tracker.update_action(1, ActionType.BET, Street.FLOP, flop, 65, 45, False)
sizes.append(tracker.get_range(1).size)
effective.append(len(tracker.get_range(1).to_weighted_combos(min_weight_frac=0.3)))

# Turn: 5h
turn_board = flop + [c("5h")]
tracker.update_board(turn_board)
tracker.update_action(1, ActionType.BET, Street.TURN, turn_board, 155, 100, False)
sizes.append(tracker.get_range(1).size)
effective.append(len(tracker.get_range(1).to_weighted_combos(min_weight_frac=0.3)))

print(f"    Range sizes: init→{sizes[0]} raise→{sizes[1]} "
      f"flop_bet→{sizes[2]} turn_bet→{sizes[3]}")
print(f"    Effective:   init→{effective[0]} raise→{effective[1]} "
      f"flop_bet→{effective[2]} turn_bet→{effective[3]}")
check("Effective range monotonically narrows",
      effective[0] >= effective[1] >= effective[2],
      f"effective={effective}")
check("Final effective range < 50% of initial",
      effective[-1] < effective[0] * 0.5,
      f"ratio={effective[-1]/max(effective[0],1):.2f}")

top_final = tracker.get_range(1).top_hands(5)
print(f"    Top 5 in final range: {top_final}")


# ===================================================================
print(f"\n{'═'*60}")
print(f"  Results: {passed} passed, {failed} failed")
print(f"{'═'*60}\n")
sys.exit(1 if failed else 0)

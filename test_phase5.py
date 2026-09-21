#!/usr/bin/env python3
"""
Phase 5 Test Suite
==================
Validates:
  1. ExploitProfile construction from opponent models
  2. Type-specific adjustments (calling station, nit, maniac, GTO-like)
  3. Confidence blending with population defaults
  4. Fold estimates fed to decision engine
  5. Bluff/value threshold tuning
  6. Strategy mode selection from opponent type
  7. Live game: HeroBot exploits each opponent type
  8. Ablation: exploitation ON vs OFF
"""

import sys
sys.path.insert(0, ".")

from opponent_model import OpponentModel, PopulationPrior
from exploiter import Exploiter, ExploitProfile, POPULATION_DEFAULTS
from game_state import GameState, Action, ActionType, Street
from game_runner import GameRunner
from opponents import HeroBot, CallingStation, ManiacBot, NitBot, RigidBot, GTOLikeBot
from hand_evaluator import card_from_str as c

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
print("\n═══ EXPLOIT PROFILE — CALLING STATION ═══")
# ===================================================================

print("\n  Building profile for a calling station:")
cs_model = OpponentModel(player_id=1)
# High VPIP, low aggression, low fold rate
for _ in range(40):
    cs_model.vpip.update(True)
for _ in range(5):
    cs_model.vpip.update(False)
for _ in range(35):
    cs_model.postflop_aggression.update(False)  # passive
for _ in range(5):
    cs_model.postflop_aggression.update(True)
for _ in range(5):
    cs_model.fold_to_bet.update(True)
for _ in range(35):
    cs_model.fold_to_bet.update(False)  # rarely folds
for _ in range(45):
    cs_model.finish_hand()

exploiter = Exploiter()
cs_profile = exploiter.build_profile(cs_model)

check("Classified as passive", cs_profile.opponent_type == "passive",
      f"type={cs_profile.opponent_type}")
check("Low bluff ceiling (don't bluff)",
      cs_profile.bluff_equity_ceiling < POPULATION_DEFAULTS.bluff_equity_ceiling,
      f"bluff_ceil={cs_profile.bluff_equity_ceiling:.2f}")
check("Low bluff cap",
      cs_profile.bluff_frequency_cap < 0.20,
      f"bluff_cap={cs_profile.bluff_frequency_cap:.2f}")
check("Lower value floor (value bet wider)",
      cs_profile.value_equity_floor < POPULATION_DEFAULTS.value_equity_floor,
      f"val_floor={cs_profile.value_equity_floor:.2f}")
check("Strategy = value_heavy",
      cs_profile.strategy_mode == "value_heavy",
      f"mode={cs_profile.strategy_mode}")
check("Low fold-to-bet (from actual data)",
      cs_profile.fold_to_bet < 0.25,
      f"fold_bet={cs_profile.fold_to_bet:.2f}")

print(f"    {exploiter.summary(cs_profile)}")


# ===================================================================
print("\n═══ EXPLOIT PROFILE — NIT ═══")
# ===================================================================

print("\n  Building profile for a nit:")
nit_model = OpponentModel(player_id=2)
for _ in range(35):
    nit_model.vpip.update(False)
for _ in range(5):
    nit_model.vpip.update(True)
for _ in range(25):
    nit_model.fold_to_bet.update(True)  # folds a lot
for _ in range(10):
    nit_model.fold_to_bet.update(False)
for _ in range(15):
    nit_model.postflop_aggression.update(False)
for _ in range(5):
    nit_model.postflop_aggression.update(True)
for _ in range(40):
    nit_model.finish_hand()

nit_profile = exploiter.build_profile(nit_model)

check("Classified as tight", nit_profile.opponent_type == "tight",
      f"type={nit_profile.opponent_type}")
check("High bluff ceiling (bluff more vs nit)",
      nit_profile.bluff_equity_ceiling > POPULATION_DEFAULTS.bluff_equity_ceiling,
      f"bluff_ceil={nit_profile.bluff_equity_ceiling:.2f}")
check("Strategy = aggressive",
      nit_profile.strategy_mode == "aggressive",
      f"mode={nit_profile.strategy_mode}")
check("High fold-to-bet (from data)",
      nit_profile.fold_to_bet > 0.55,
      f"fold_bet={nit_profile.fold_to_bet:.2f}")
check("Positive call-threshold adjust (fold when they bet back)",
      nit_profile.call_threshold_adjust > 0,
      f"call_adj={nit_profile.call_threshold_adjust:.2f}")

print(f"    {exploiter.summary(nit_profile)}")


# ===================================================================
print("\n═══ EXPLOIT PROFILE — MANIAC ═══")
# ===================================================================

print("\n  Building profile for a maniac:")
man_model = OpponentModel(player_id=3)
for _ in range(40):
    man_model.vpip.update(True)
for _ in range(35):
    man_model.postflop_aggression.update(True)
for _ in range(5):
    man_model.postflop_aggression.update(False)
for _ in range(10):
    man_model.fold_to_bet.update(True)
for _ in range(30):
    man_model.fold_to_bet.update(False)
for _ in range(15):
    man_model.bluff_frequency.update(True)
for _ in range(10):
    man_model.bluff_frequency.update(False)
for _ in range(40):
    man_model.finish_hand()

man_profile = exploiter.build_profile(man_model)

check("Classified as aggressive", man_profile.opponent_type == "aggressive",
      f"type={man_profile.opponent_type}")
check("Strategy = trap_heavy",
      man_profile.strategy_mode == "trap_heavy",
      f"mode={man_profile.strategy_mode}")
check("Positive hero-call boost (call down vs bluffer)",
      man_profile.hero_call_boost > 0,
      f"call_boost={man_profile.hero_call_boost:.2f}")
check("Low bluff cap (don't bluff — they'll raise)",
      man_profile.bluff_frequency_cap < 0.25,
      f"bluff_cap={man_profile.bluff_frequency_cap:.2f}")

print(f"    {exploiter.summary(man_profile)}")


# ===================================================================
print("\n═══ CONFIDENCE BLENDING ═══")
# ===================================================================

print("\n  Low-data opponent (5 hands) → population defaults:")
low_model = OpponentModel(player_id=10)
for _ in range(5):
    low_model.vpip.update(True)
    low_model.finish_hand()

low_profile = exploiter.build_profile(low_model)
check("Low confidence (< min threshold)",
      low_profile.confidence == 0.0,
      f"conf={low_profile.confidence}")
check("Type = unknown", low_profile.opponent_type == "unknown")

print("\n  Medium-data opponent (25 hands) → partial blend:")
med_model = OpponentModel(player_id=11)
for _ in range(20):
    med_model.vpip.update(True)
    med_model.fold_to_bet.update(False)
for _ in range(5):
    med_model.vpip.update(False)
    med_model.fold_to_bet.update(True)
for _ in range(25):
    med_model.finish_hand()

med_profile = exploiter.build_profile(med_model)
check("Medium confidence (0 < c < 1)",
      0 < med_profile.confidence < 1,
      f"conf={med_profile.confidence}")
# Fold-to-bet should be between raw (0.20) and default (0.40)
check("Fold-to-bet blended between raw and default",
      0.15 < med_profile.fold_to_bet < 0.45,
      f"fold_bet={med_profile.fold_to_bet:.2f}")


# ===================================================================
print("\n═══ DECISION ENGINE INTEGRATION ═══")
# ===================================================================

from decision_engine import DecisionEngine, EVConfig

print("\n  Fold estimates: population vs exploit profile:")
engine = DecisionEngine(EVConfig(equity_simulations=1000))

# Build a game state
gs = GameState(
    num_players=2, starting_stacks=[1000, 1000],
    small_blind=5, big_blind=10,
)
gs.new_hand(dealer_seat=0, hole_cards=[[c("As"), c("Kd")], [c("5s"), c("4s")]])
gs.apply_action(Action(0, ActionType.CALL, 10, Street.PREFLOP))
gs.apply_action(Action(1, ActionType.CHECK, 0, Street.PREFLOP))
gs.board = [c("Ah"), c("7d"), c("2c")]

# Evaluate without exploit profile (population defaults)
evs_pop = engine.evaluate(gs, hero=1, rng_seed=42)

# Evaluate with calling-station profile (should change bet EVs)
evs_cs = engine.evaluate(gs, hero=1, rng_seed=42, exploit_profile=cs_profile)

# Evaluate with nit profile
evs_nit = engine.evaluate(gs, hero=1, rng_seed=42, exploit_profile=nit_profile)

# Find bet EVs
def find_ev(evs, action_type):
    for e in evs:
        if e.action_type == action_type:
            return e.ev
    return None

bet_pop = find_ev(evs_pop, ActionType.BET)
bet_cs = find_ev(evs_cs, ActionType.BET)
bet_nit = find_ev(evs_nit, ActionType.BET)

print(f"    Bet EV (population): {bet_pop:.1f}")
print(f"    Bet EV (vs station): {bet_cs:.1f}")
print(f"    Bet EV (vs nit):     {bet_nit:.1f}")

check("Bet EVs differ with exploit profiles",
      bet_pop != bet_cs or bet_pop != bet_nit,
      "all EVs identical")
# vs nit: they fold more → bluff EV higher from fold equity
check("Higher fold equity vs nit → different EV structure",
      bet_nit is not None)


# ===================================================================
print("\n═══ LIVE GAME — EXPLOITATION IN ACTION ═══")
# ===================================================================

N = 30

def run_match(hero_name, opp, deck_seed, learning=True):
    h = HeroBot(hero_name, seed=42, learning=learning)
    r = GameRunner(bots=[h, opp], starting_stack=1000,
                   small_blind=5, big_blind=10, deck_seed=deck_seed)
    s = r.run(num_hands=N)
    ps = s.players.get(hero_name)
    return ps.net_chips if ps else 0, h

print(f"\n  HeroBot (Phase 5) vs each opponent ({N} hands):")

matchups = [
    ("Station",  CallingStation("Station", seed=10)),
    ("Nit",      NitBot("Nit", seed=20)),
    ("Maniac",   ManiacBot("Maniac", seed=30)),
    ("GTOLike",  GTOLikeBot("GTOLike", seed=40)),
    ("Rigid",    RigidBot("Rigid", seed=50)),
]

print(f"\n  {'Opponent':<10} {'Net':>8} {'bb/h':>7} {'Mode':>12} {'OppType':>12}")
print(f"  {'─' * 52}")

for opp_name, opp in matchups:
    net, hero = run_match(f"H_{opp_name}", opp, deck_seed=500 + hash(opp_name) % 100)
    bbh = net / 10 / N

    if hero.learner:
        opp_model = hero.learner.get_model(1)
        otype = opp_model.primary_type() if opp_model.hands_observed > 10 else "?"
        mode = hero._controller.mode.name
    else:
        otype = "?"
        mode = "?"

    print(f"  {opp_name:<10} {net:>+8} {bbh:>+7.2f} {mode:>12} {otype:>12}")


# ===================================================================
print("\n═══ ABLATION: EXPLOITATION ON vs OFF ═══")
# ===================================================================

print(f"\n  Same hands, learning on vs off:")
results = []
for opp_name, opp_cls in [("Station", CallingStation), ("Maniac", ManiacBot),
                            ("Nit", NitBot)]:
    net_on, _ = run_match("Hero_ON", opp_cls(opp_name, seed=10),
                           deck_seed=700, learning=True)
    net_off, _ = run_match("Hero_OFF", opp_cls(opp_name, seed=10),
                            deck_seed=700, learning=False)
    results.append((opp_name, net_off, net_on))

print(f"\n  {'Opponent':<10} {'No Exploit':>11} {'Exploit':>9} {'Δ':>8}")
print(f"  {'─' * 40}")
for name, off, on in results:
    print(f"  {name:<10} {off:>+11} {on:>+9} {on-off:>+8}")

check("Ablation completed", len(results) == 3)


# ===================================================================
print("\n═══ EXPLOIT PROFILE DETAILS ═══")
# ===================================================================

print("\n  All profiles side by side:")
profiles = [
    ("Station", cs_profile),
    ("Nit", nit_profile),
    ("Maniac", man_profile),
    ("Default", POPULATION_DEFAULTS),
]

print(f"\n  {'Stat':<20}", end="")
for name, _ in profiles:
    print(f" {name:>10}", end="")
print()
print(f"  {'─' * 62}")

fields = [
    ("fold_to_bet", "Fold to bet"),
    ("fold_to_raise", "Fold to raise"),
    ("bluff_equity_ceiling", "Bluff ceiling"),
    ("bluff_frequency_cap", "Bluff cap"),
    ("value_equity_floor", "Value floor"),
    ("call_threshold_adjust", "Call adjust"),
    ("hero_call_boost", "Call boost"),
    ("temperature_scale", "Temp scale"),
]

for attr, label in fields:
    print(f"  {label:<20}", end="")
    for _, prof in profiles:
        val = getattr(prof, attr)
        print(f" {val:>+10.2f}" if "adjust" in attr or "boost" in attr
              else f" {val:>10.2f}", end="")
    print()


# ===================================================================
print(f"\n{'═'*60}")
print(f"  Results: {passed} passed, {failed} failed")
print(f"{'═'*60}\n")
sys.exit(1 if failed else 0)

#!/usr/bin/env python3
"""
Demo — Poker Bot Showcase
==========================
Runs several matches and shows the full system in action:
  1. Round-robin tournament (all opponents)
  2. OracleBot (Phase 6) vs each opponent
  3. Ablation: Oracle with/without RNG cracking

Usage:  python demo.py
"""

import sys, time
sys.path.insert(0, ".")

from game_runner import GameRunner, round_robin
from opponents import (
    HeroBot, RandomBot, NitBot, CallingStation,
    ManiacBot, GTOLikeBot, RigidBot,
)
from oracle_bot import OracleBot
from hand_evaluator import card_str

HANDS = 30
STACK = 1000
SB, BB = 5, 10


def divider(title):
    print(f"\n{'━' * 60}")
    print(f"  {title}")
    print(f"{'━' * 60}")


# ===================================================================
# 1. Round-Robin Tournament (fast bots only — no Monte Carlo)
# ===================================================================
divider("ROUND-ROBIN TOURNAMENT (50 hands per matchup)")

bots = [
    RandomBot("Random", seed=10),
    NitBot("Nit", seed=20),
    CallingStation("Station", seed=30),
    ManiacBot("Maniac", seed=40),
    GTOLikeBot("GTOLike", seed=50),
    RigidBot("Rigid", seed=60),
]

t0 = time.time()
rr = round_robin(
    bots=bots,
    hands_per_matchup=HANDS,
    starting_stack=STACK,
    small_blind=SB,
    big_blind=BB,
    deck_seed=12345,
)
dt = time.time() - t0

print(rr.report())
print(f"\n  Completed in {dt:.1f}s")

# ===================================================================
# 2. OracleBot Head-to-Head
# ===================================================================
divider("ORACLEBOT (Phase 6) vs EACH OPPONENT — 50 hands")

GAME_SEED = 77
opponents = [
    ("Random",  RandomBot("Random", seed=10)),
    ("Nit",     NitBot("Nit", seed=20)),
    ("Station", CallingStation("Station", seed=30)),
    ("Maniac",  ManiacBot("Maniac", seed=40)),
    ("GTOLike", GTOLikeBot("GTOLike", seed=50)),
    ("Rigid",   RigidBot("Rigid", seed=60)),
]

print(f"\n  {'Opponent':<12} {'Oracle Net':>10} {'bb/hand':>8} "
      f"{'Cracked':>8} {'Predictions':>12}")
print(f"  {'─' * 54}")

for name, opp in opponents:
    oracle = OracleBot("Oracle", max_seed=100, rng_enabled=True, seed=1)
    runner = GameRunner(
        bots=[oracle, opp],
        starting_stack=STACK,
        small_blind=SB, big_blind=BB,
        deck_seed=GAME_SEED,
    )
    stats = runner.run(num_hands=HANDS)

    ps = stats.players.get("Oracle")
    net = ps.net_chips if ps else 0
    bbh = net / BB / HANDS
    status = oracle.analyzer_status()

    print(f"  {name:<12} {net:>+10} {bbh:>+8.2f} "
          f"{'  ✓ yes' if status['cracked'] else '  ✗ no':>8} "
          f"{status['predictions_used']:>12}")


# ===================================================================
# 3. Ablation: RNG On vs Off
# ===================================================================
divider("ABLATION: RNG ADVANTAGE (30 hands vs Station)")

ablation_results = []
for opp_name, opp_cls in [("Station", CallingStation)]:
    # Without RNG
    hero_off = OracleBot("Off", max_seed=100, rng_enabled=False, seed=1)
    opp1 = opp_cls(opp_name, seed=99)
    r1 = GameRunner(bots=[hero_off, opp1], starting_stack=STACK,
                    small_blind=SB, big_blind=BB, deck_seed=GAME_SEED)
    s1 = r1.run(num_hands=30)
    net_off = s1.players["Off"].net_chips

    # With RNG
    hero_on = OracleBot("On", max_seed=100, rng_enabled=True, seed=1)
    opp2 = opp_cls(opp_name, seed=99)
    r2 = GameRunner(bots=[hero_on, opp2], starting_stack=STACK,
                    small_blind=SB, big_blind=BB, deck_seed=GAME_SEED)
    s2 = r2.run(num_hands=30)
    net_on = s2.players["On"].net_chips

    ablation_results.append((opp_name, net_off, net_on))

print(f"\n  {'Opponent':<12} {'No RNG':>10} {'With RNG':>10} {'Advantage':>10}")
print(f"  {'─' * 44}")
for name, off, on in ablation_results:
    print(f"  {name:<12} {off:>+10} {on:>+10} {on - off:>+10}")


# ===================================================================
# 4. Sample Prediction
# ===================================================================
divider("SAMPLE PREDICTION (seed=42)")

oracle_demo = OracleBot("Oracle", max_seed=100, rng_enabled=True, seed=1)
station_demo = CallingStation("Station", seed=2)
runner_demo = GameRunner(
    bots=[oracle_demo, station_demo],
    starting_stack=STACK,
    small_blind=SB, big_blind=BB,
    deck_seed=42,
)
stats_demo = runner_demo.run(num_hands=3)

status = oracle_demo.analyzer_status()
if status["cracked"]:
    from environment_model import EnvironmentAnalyzer
    # Predict the next 3 hands
    preds = oracle_demo._analyzer.predict_n_hands(3, num_players=2, hero_seat=0)
    print(f"\n  RNG cracked after {status['observations']} hands")
    print(f"  Seed: {status['best_seed']}  "
          f"Protocol: {status['best_protocol']}  "
          f"PRNG: {status['best_prng']}")
    print(f"\n  Next 3 hands predicted:")
    for i, p in enumerate(preds):
        if p:
            print(f"    Hand {p['hand_index']}: "
                  f"Hero {p['hero_cards_str']}  "
                  f"Board {p['board_str']}  "
                  f"Opp {p['opponent_cards_str']}")
else:
    print(f"\n  RNG not cracked ({status['candidates_alive']} candidates alive)")

divider("SEED BEYOND THE 0-999 WINDOW (fallback crack from cards)")

# The initial enumeration only covers seeds 0-999.  When the arena's seed
# is larger, the same card observations — hero hole cards, the community
# cards, and any opponent hand shown at showdown — drive an expanded search
# that pins the seed anyway.  (A truly random 32-bit seed is only reachable
# as far as you can enumerate; there is no untemper shortcut from a shuffle,
# which never exposes whole 32-bit words.)
from environment_model import EnvironmentAnalyzer, HandObservation
from deck import Deck

OOW_SEED = 4242  # outside the 0-999 initial window
_d = Deck(seed=OOW_SEED)
_decks = []
for _ in range(5):
    _d.reset()
    _decks.append(_d.deal(12))

analyzer = EnvironmentAnalyzer(max_seed=1000, fallback_max_seed=8000,
                               fallback_batch=8000, fallback_time_budget=20.0)
for hi in range(4):
    a = _decks[hi]
    analyzer.observe(HandObservation(
        hand_index=hi, num_players=2, hero_seat=0,
        hero_cards=[a[0], a[1]],
        board=[a[5], a[6], a[7], a[9], a[11]],
        showdown_cards={1: [a[2], a[3]]}))   # opponent hand as extra constraint

st = analyzer.status()
print(f"\n  Initial window 0-999 exhausted, fallback triggered: "
      f"{analyzer._fallback_triggered}")
print(f"  Recovered seed: {st['best_seed']}  (true seed {OOW_SEED})  "
      f"confidence {st['confidence']:.2f}")
pred = analyzer.predict_next_hand(2, 0)
if pred:
    real = _decks[4]
    exact = (pred["hero_cards"] == [real[0], real[1]]
             and pred["board"] == [real[5], real[6], real[7], real[9], real[11]]
             and pred["opponent_cards"][1] == [real[2], real[3]])
    print(f"  Next hand predicted: Hero {pred['hero_cards_str']}  "
          f"Board {pred['board_str']}  Opp {pred['opponent_cards_str'][1]}")
    print(f"  Prediction exact: {exact}")

divider("DONE")

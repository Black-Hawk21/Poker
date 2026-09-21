#!/usr/bin/env python3
"""
Phase 6 Test Suite
==================
Validates:
  1. Observation collection from card data
  2. Dealing protocol hypothesis detection
  3. Seed elimination from hero cards alone
  4. Full seed cracking with showdown data
  5. Future card prediction accuracy
  6. OracleBot vs standard opponents (advantage measurement)
  7. Ablation: OracleBot with RNG disabled vs enabled
  8. Edge cases: wrong seed space, no showdowns
"""

import sys, time
sys.path.insert(0, ".")

from hand_evaluator import card_from_str as c, card_str
from game_state import GameState, Action, ActionType, Street
from game_runner import GameRunner
from opponents import (
    RandomBot, NitBot, CallingStation, ManiacBot,
    GTOLikeBot, RigidBot, HeroBot, make_opponent,
)
from oracle_bot import OracleBot
from environment_model import (
    EnvironmentAnalyzer, HandObservation,
    PROTOCOL_STANDARD, PROTOCOL_NO_BURNS, ALL_PROTOCOLS,
)
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
print("\n═══ DEALING PROTOCOL — SLOT MAPPING ═══")
# ===================================================================

print("\n  Standard protocol (2 players):")
obs = HandObservation(
    hand_index=0, num_players=2, hero_seat=0,
    hero_cards=[10, 20], board=[30, 31, 32, 33, 34],
    showdown_cards={1: [40, 41]},
)

slots = obs.known_deck_slots(PROTOCOL_STANDARD)
# Standard 2-player: P0 cards at [0,1], P1 at [2,3]
# burn at [4], flop at [5,6,7], burn at [8], turn at [9], burn [10], river [11]
check("Hero cards at positions 0,1", slots.get(0) == 10 and slots.get(1) == 20)
check("Opponent cards at positions 2,3", slots.get(2) == 40 and slots.get(3) == 41)
check("Flop at positions 5,6,7",
      slots.get(5) == 30 and slots.get(6) == 31 and slots.get(7) == 32)
check("Turn at position 9", slots.get(9) == 33)
check("River at position 11", slots.get(11) == 34)

print("\n  No-burns protocol:")
slots_nb = obs.known_deck_slots(PROTOCOL_NO_BURNS)
check("No-burns: flop at 4,5,6",
      slots_nb.get(4) == 30 and slots_nb.get(5) == 31 and slots_nb.get(6) == 32)
check("No-burns: turn at 7", slots_nb.get(7) == 33)
check("No-burns: river at 8", slots_nb.get(8) == 34)


# ===================================================================
print("\n═══ SEED ELIMINATION — UNIT TEST ═══")
# ===================================================================

# Create a deck with known seed and extract observations
print("\n  Simulating seed=42, 3 hands:")
test_seed = 42
deck = Deck(seed=test_seed)
sim_observations = []

for hand_i in range(3):
    deck.reset()
    all_cards = deck.deal(12)  # deal enough for 2 players + burn/board

    # P0 cards: [0,1], P1 cards: [2,3]
    # burn [4], flop [5,6,7], burn [8], turn [9], burn [10], river [11]
    obs = HandObservation(
        hand_index=hand_i, num_players=2, hero_seat=0,
        hero_cards=[all_cards[0], all_cards[1]],
        board=[all_cards[5], all_cards[6], all_cards[7],
               all_cards[9], all_cards[11]],
        showdown_cards={1: [all_cards[2], all_cards[3]]},
    )
    sim_observations.append(obs)
    print(f"    Hand {hand_i}: hero=[{card_str(all_cards[0])} {card_str(all_cards[1])}]  "
          f"opp=[{card_str(all_cards[2])} {card_str(all_cards[3])}]  "
          f"board=[{' '.join(card_str(c) for c in [all_cards[5],all_cards[6],all_cards[7],all_cards[9],all_cards[11]])}]")

# Create analyzer with seed space that includes the real seed
analyzer = EnvironmentAnalyzer(max_seed=100)  # seed 42 is in range

print(f"\n  Initial: {analyzer.alive_count()} candidates")

# Feed observations one at a time
for i, obs in enumerate(sim_observations):
    t0 = time.time()
    analyzer.observe(obs)
    dt = time.time() - t0
    alive = analyzer.alive_count()
    print(f"    After hand {i}: {alive} alive  ({dt:.2f}s)  "
          f"entropy={analyzer._entropy_bits():.1f} bits")

check("Candidates eliminated", analyzer.alive_count() < 300,
      f"alive={analyzer.alive_count()}")
check("True seed (42) survives",
      any(c.alive and c.seed == test_seed
          for c in analyzer._candidates.values()),
      f"alive seeds include 42: {analyzer._best_seed}")

# Check if cracked
if analyzer.cracked:
    check("Seed cracked!", True)
    check("Correct seed found", analyzer.best_seed == test_seed,
          f"found={analyzer.best_seed}")
    check("Correct protocol found",
          analyzer.best_protocol.name == "standard",
          f"found={analyzer.best_protocol.name}")
else:
    check(f"High confidence ({analyzer.confidence:.2f})",
          analyzer.confidence > 0.3,
          f"conf={analyzer.confidence:.4f}")


# ===================================================================
print("\n═══ PREDICTION VALIDATION ═══")
# ===================================================================

if analyzer.is_confident and analyzer.best_seed is not None:
    print(f"\n  Predicting hand 3 (seed={analyzer.best_seed}):")
    pred = analyzer.predict_next_hand(num_players=2, hero_seat=0)

    if pred is not None:
        # Simulate the actual hand 3 to verify
        deck2 = Deck(seed=test_seed)
        for _ in range(3):
            deck2.reset()
            deck2.deal(12)
        deck2.reset()
        actual = deck2.deal(12)
        actual_hero = [actual[0], actual[1]]
        actual_board = [actual[5], actual[6], actual[7], actual[9], actual[11]]
        actual_opp = [actual[2], actual[3]]

        check("Predicted hero cards match",
              pred["hero_cards"] == actual_hero,
              f"pred={pred['hero_cards_str']} actual={[card_str(c) for c in actual_hero]}")
        check("Predicted board matches",
              pred["board"] == actual_board,
              f"pred={pred['board_str']} actual={[card_str(c) for c in actual_board]}")
        check("Predicted opponent cards match",
              pred["opponent_cards"].get(1) == actual_opp,
              f"pred={pred.get('opponent_cards_str',{}).get(1)} "
              f"actual={[card_str(c) for c in actual_opp]}")

        print(f"    Hero:  {pred['hero_cards_str']}")
        print(f"    Board: {pred['board_str']}")
        print(f"    Opp:   {pred['opponent_cards_str']}")

    # Multi-hand prediction
    preds = analyzer.predict_n_hands(3, num_players=2, hero_seat=0)
    check("Predict 3 future hands", len(preds) == 3 and preds[0] is not None)
else:
    print("\n  (Skipping prediction — not confident enough yet)")
    # Feed more observations if needed
    for hand_i in range(3, 8):
        deck.reset()
        all_cards = deck.deal(12)
        obs = HandObservation(
            hand_index=hand_i, num_players=2, hero_seat=0,
            hero_cards=[all_cards[0], all_cards[1]],
            board=[all_cards[5], all_cards[6], all_cards[7],
                   all_cards[9], all_cards[11]],
            showdown_cards={1: [all_cards[2], all_cards[3]]},
        )
        analyzer.observe(obs)

    alive_final = analyzer.alive_count()
    print(f"    After 8 total hands: {alive_final} alive")
    check("Eventually narrows candidates", alive_final <= 5,
          f"alive={alive_final}")


# ===================================================================
print("\n═══ HERO-CARDS-ONLY INFERENCE ═══")
# ===================================================================

# Test with NO showdown data — only hero cards (harder)
print("\n  Seed elimination using only hero hole cards (no showdowns):")
analyzer_hero_only = EnvironmentAnalyzer(max_seed=100)
deck3 = Deck(seed=42)

for hand_i in range(10):
    deck3.reset()
    all_cards = deck3.deal(12)
    obs = HandObservation(
        hand_index=hand_i, num_players=2, hero_seat=0,
        hero_cards=[all_cards[0], all_cards[1]],
        board=[],  # no board info (folded preflop)
        showdown_cards={},
    )
    analyzer_hero_only.observe(obs)

alive_ho = analyzer_hero_only.alive_count()
print(f"    After 10 hands (hero cards only): {alive_ho} alive")
check("Hero-only elimination works", alive_ho < 300,
      f"alive={alive_ho}")
check("True seed survives hero-only test",
      any(c.alive and c.seed == 42
          for c in analyzer_hero_only._candidates.values()))


# ===================================================================
print("\n═══ ORACLEBOT IN GAME RUNNER ═══")
# ===================================================================

# Run OracleBot against a CallingStation with a known seed
GAME_SEED = 77
N_HANDS = 30

print(f"\n  OracleBot vs CallingStation, {N_HANDS} hands (game seed={GAME_SEED}):")
oracle = OracleBot("Oracle", max_seed=100, rng_enabled=True, seed=1)
station = CallingStation("Station", seed=2)

runner = GameRunner(
    bots=[oracle, station],
    starting_stack=1000,
    small_blind=5,
    big_blind=10,
    deck_seed=GAME_SEED,
)
t0 = time.time()
stats = runner.run(num_hands=N_HANDS)
dt = time.time() - t0

check(f"{N_HANDS} hands completed ({dt:.1f}s)", stats.total_hands == N_HANDS)

# Check oracle status
ostatus = oracle.analyzer_status()
print(f"    Analyzer: {ostatus['candidates_alive']} alive, "
      f"confidence={ostatus['confidence']:.2f}, "
      f"cracked={ostatus['cracked']}")
print(f"    Predictions made: {ostatus['predictions_made']}, "
      f"used: {ostatus['predictions_used']}")
print(f"    Entropy: {ostatus['entropy_bits']:.1f} bits")

if ostatus['cracked']:
    check("Oracle cracked the seed", True)
    check("Correct seed found in game",
          ostatus['best_seed'] == GAME_SEED,
          f"found={ostatus['best_seed']}, expected={GAME_SEED}")
    check("Predictions were used",
          ostatus['predictions_used'] > 0,
          f"used={ostatus['predictions_used']}")
else:
    # Still should have eliminated most candidates
    check("Significant elimination occurred",
          ostatus['candidates_alive'] < 100,
          f"alive={ostatus['candidates_alive']}")

# Print performance
oracle_ps = stats.players.get("Oracle")
station_ps = stats.players.get("Station")
if oracle_ps:
    bb_h = oracle_ps.net_chips / 10 / N_HANDS
    print(f"    Oracle: net={oracle_ps.net_chips:+d}  bb/hand={bb_h:+.2f}")
if station_ps:
    bb_h2 = station_ps.net_chips / 10 / N_HANDS
    print(f"    Station: net={station_ps.net_chips:+d}  bb/hand={bb_h2:+.2f}")


# ===================================================================
print("\n═══ ABLATION: RNG ON vs OFF ═══")
# ===================================================================

print(f"\n  Same matchup, RNG enabled vs disabled:")

# RNG disabled (pure Phase 2)
hero_no_rng = OracleBot("Hero_NoRNG", max_seed=100, rng_enabled=False, seed=1)
station2 = CallingStation("Station", seed=2)
runner_off = GameRunner(
    bots=[hero_no_rng, station2],
    starting_stack=1000, small_blind=5, big_blind=10,
    deck_seed=GAME_SEED,
)
stats_off = runner_off.run(num_hands=N_HANDS)

# RNG enabled (Phase 6)
hero_rng = OracleBot("Hero_RNG", max_seed=100, rng_enabled=True, seed=1)
station3 = CallingStation("Station", seed=2)
runner_on = GameRunner(
    bots=[hero_rng, station3],
    starting_stack=1000, small_blind=5, big_blind=10,
    deck_seed=GAME_SEED,
)
stats_on = runner_on.run(num_hands=N_HANDS)

net_off = stats_off.players.get("Hero_NoRNG").net_chips if stats_off.players.get("Hero_NoRNG") else 0
net_on = stats_on.players.get("Hero_RNG").net_chips if stats_on.players.get("Hero_RNG") else 0

print(f"    Without RNG: {net_off:+d} chips")
print(f"    With RNG:    {net_on:+d} chips")
print(f"    RNG advantage: {net_on - net_off:+d} chips")

check("Ablation completed", True)

rng_status = hero_rng.analyzer_status()
print(f"    RNG bot: cracked={rng_status['cracked']}, "
      f"predictions_used={rng_status['predictions_used']}")


# ===================================================================
print("\n═══ EDGE CASE: SEED NOT IN RANGE ═══")
# ===================================================================

print("\n  Game seed=999, analyzer max_seed=50 (seed outside range):")
oracle_miss = OracleBot("OracleMiss", max_seed=50, rng_enabled=True, seed=1)
station4 = CallingStation("Station", seed=2)
runner_miss = GameRunner(
    bots=[oracle_miss, station4],
    starting_stack=1000, small_blind=5, big_blind=10,
    deck_seed=999,  # outside [0, 50)
)
stats_miss = runner_miss.run(num_hands=20)

miss_status = oracle_miss.analyzer_status()
check("All candidates eliminated (seed not in range)",
      miss_status['candidates_alive'] == 0,
      f"alive={miss_status['candidates_alive']}")
check("Confidence is 0 (correctly uncertain)",
      miss_status['confidence'] == 0.0,
      f"conf={miss_status['confidence']}")
check("Bot still plays without crashing", stats_miss.total_hands == 20)
print(f"    Oracle gracefully falls back to standard play")


# ===================================================================
print("\n═══ CONVERGENCE SPEED ═══")
# ===================================================================

print("\n  Measuring hands needed to crack seed=42 in [0,200):")
conv_analyzer = EnvironmentAnalyzer(max_seed=200)
conv_deck = Deck(seed=42)

cracked_at = None
for hand_i in range(30):
    conv_deck.reset()
    all_cards = conv_deck.deal(12)
    obs = HandObservation(
        hand_index=hand_i, num_players=2, hero_seat=0,
        hero_cards=[all_cards[0], all_cards[1]],
        board=[all_cards[5], all_cards[6], all_cards[7],
               all_cards[9], all_cards[11]],
        showdown_cards={1: [all_cards[2], all_cards[3]]},
    )
    conv_analyzer.observe(obs)

    alive = conv_analyzer.alive_count()
    if hand_i < 5 or alive <= 5 or conv_analyzer.cracked:
        print(f"    Hand {hand_i}: {alive} alive, "
              f"entropy={conv_analyzer._entropy_bits():.1f} bits")

    if conv_analyzer.cracked and cracked_at is None:
        cracked_at = hand_i
        break

if cracked_at is not None:
    check(f"Cracked in {cracked_at + 1} hands",
          cracked_at < 15,
          f"took {cracked_at + 1} hands")
else:
    alive_final = conv_analyzer.alive_count()
    check(f"Near-cracked after 30 hands ({alive_final} alive)",
          alive_final <= 3, f"alive={alive_final}")


# ===================================================================
print(f"\n{'═'*60}")
print(f"  Results: {passed} passed, {failed} failed")
print(f"{'═'*60}\n")
sys.exit(1 if failed else 0)

#!/usr/bin/env python3
"""
Environment Test Suite
======================
Validates:
  1. Bot interface and protocol
  2. All synthetic opponents can play without errors
  3. Game runner — single hands and multi-hand sessions
  4. Showdown resolution
  5. Stats tracking and reporting
  6. Round-robin tournament
  7. HeroBot integration
  8. Spectator learning protocol
"""

import sys
sys.path.insert(0, ".")

from hand_evaluator import card_from_str as c, card_str
from game_state import GameState, Action, ActionType, Street
from game_runner import GameRunner, round_robin
from opponents import (
    RandomBot, NitBot, CallingStation, ManiacBot,
    GTOLikeBot, RigidBot, HeroBot, ALL_OPPONENTS, make_opponent,
)
from bot_interface import BaseBot, HandStartInfo, ShowdownInfo, HandEndInfo
from stats import StatsTracker

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
print("\n═══ BOT INTERFACE ═══")
# ===================================================================

print("\n  Factory:")
for name in ALL_OPPONENTS:
    bot = make_opponent(name, seed=42)
    check(f"make_opponent('{name}') → {bot.name}", bot is not None)

try:
    make_opponent("nonexistent")
    check("Unknown opponent raises error", False)
except ValueError:
    check("Unknown opponent raises ValueError", True)


# ===================================================================
print("\n═══ SINGLE HAND — EACH OPPONENT VS RANDOM ═══")
# ===================================================================

for opp_name, opp_cls in ALL_OPPONENTS.items():
    print(f"\n  {opp_name} vs Random:")
    runner = GameRunner(
        bots=[opp_cls(seed=1), RandomBot(seed=2)],
        starting_stack=500,
        small_blind=5,
        big_blind=10,
        deck_seed=42,
    )
    try:
        stats = runner.run(num_hands=1)
        check(f"{opp_name} plays 1 hand OK",
              stats.total_hands == 1)
    except Exception as e:
        check(f"{opp_name} plays without error", False, str(e))


# ===================================================================
print("\n═══ MULTI-HAND SESSION ═══")
# ===================================================================

print("\n  200 hands: Maniac vs CallingStation:")
runner = GameRunner(
    bots=[ManiacBot(seed=10), CallingStation(seed=20)],
    starting_stack=1000,
    small_blind=5,
    big_blind=10,
    deck_seed=100,
)
stats = runner.run(num_hands=200)

check("200 hands completed", stats.total_hands == 200)
check("Both players tracked", len(stats.players) == 2)

# Net chips should sum to zero (zero-sum game)
names = list(stats.players.keys())
net_sum = sum(ps.net_chips for ps in stats.players.values())
check("Zero-sum: net chips sum ≈ 0", abs(net_sum) < 5,
      f"net_sum={net_sum}")

for name, ps in stats.players.items():
    check(f"{name}: hands_played=200", ps.hands_played == 200)
    check(f"{name}: win_rate in [0,1]", 0 <= ps.win_rate <= 1,
          f"wr={ps.win_rate:.2f}")
    check(f"{name}: VPIP > 0", ps.vpip > 0, f"vpip={ps.vpip:.2f}")

# Print report
report = stats.report()
check("Report generated", len(report) > 100)
print(report)


# ===================================================================
print("\n═══ STATS DETAIL ═══")
# ===================================================================

print("\n  Showdown tracking:")
total_sd = sum(ps.showdowns for ps in stats.players.values())
check("Showdowns recorded", total_sd > 0, f"total_showdowns={total_sd}")

sd_summary = stats.summary_dict()
check("Summary dict has player data", "players" in sd_summary)
check("Summary has total_hands", sd_summary["total_hands"] == 200)


# ===================================================================
print("\n═══ 3-PLAYER TABLE ═══")
# ===================================================================

print("\n  100 hands: Nit vs Maniac vs GTOLike:")
runner3 = GameRunner(
    bots=[NitBot(seed=1), ManiacBot(seed=2), GTOLikeBot(seed=3)],
    starting_stack=1000,
    small_blind=5,
    big_blind=10,
    deck_seed=200,
)
stats3 = runner3.run(num_hands=100)
check("3-player: 100 hands completed", stats3.total_hands == 100)
check("3-player: all 3 tracked", len(stats3.players) == 3)
net_sum3 = sum(ps.net_chips for ps in stats3.players.values())
check("3-player zero-sum", abs(net_sum3) < 5, f"net_sum={net_sum3}")
print(stats3.report())


# ===================================================================
print("\n═══ HEROBOT INTEGRATION ═══")
# ===================================================================

print("\n  50 hands: HeroBot vs CallingStation:")
hero = HeroBot("Hero", seed=42)
station = CallingStation("Station", seed=99)

runner_hero = GameRunner(
    bots=[hero, station],
    starting_stack=1000,
    small_blind=5,
    big_blind=10,
    deck_seed=300,
)
stats_hero = runner_hero.run(num_hands=50)
check("Hero plays 50 hands", stats_hero.total_hands == 50)

hero_ps = stats_hero.players.get("Hero")
check("Hero stats tracked", hero_ps is not None)
if hero_ps:
    bb_h = hero_ps.net_chips / 10 / 50
    print(f"    Hero: net={hero_ps.net_chips:+d}  "
          f"bb/hand={bb_h:+.2f}  "
          f"wr={hero_ps.win_rate:.1%}  "
          f"sd_wr={hero_ps.showdown_win_rate:.1%}")


# ===================================================================
print("\n═══ SPECTATOR LEARNING PROTOCOL ═══")
# ===================================================================

class SpyBot(BaseBot):
    """Bot that records all observations for testing."""
    def __init__(self):
        super().__init__("Spy")
        self.hand_starts = 0
        self.actions_seen = 0
        self.boards_seen = 0
        self.showdowns_seen = 0
        self.hand_ends = 0
        self.showdown_cards: list[dict] = []

    def act(self, gs):
        # Call or check — never fold, so we reach showdowns
        legals = gs.get_legal_actions(self.seat)
        for la in legals:
            if la.action_type == ActionType.CHECK:
                return Action(self.seat, ActionType.CHECK, 0, gs.street)
        for la in legals:
            if la.action_type == ActionType.CALL:
                return Action(self.seat, ActionType.CALL, la.min_amount, gs.street)
        return Action(self.seat, ActionType.FOLD, 0, gs.street)

    def observe_hand_start(self, info):
        super().observe_hand_start(info)
        self.hand_starts += 1

    def observe_action(self, action, gs):
        self.actions_seen += 1

    def observe_board(self, street, board):
        self.boards_seen += 1

    def observe_showdown(self, info):
        self.showdowns_seen += 1
        self.showdown_cards.append(info.revealed_cards)

    def observe_hand_end(self, info):
        self.hand_ends += 1

print("\n  Protocol validation (SpyBot + ManiacBot, 50 hands):")
spy = SpyBot()
maniac = ManiacBot("Maniac", seed=55)
runner_spy = GameRunner(
    bots=[spy, maniac],
    starting_stack=500,
    small_blind=5,
    big_blind=10,
    deck_seed=400,
)
stats_spy = runner_spy.run(num_hands=50)

check("SpyBot received 50 hand_starts", spy.hand_starts == 50)
check("SpyBot received 50 hand_ends", spy.hand_ends == 50)
check("SpyBot saw actions", spy.actions_seen > 50,
      f"actions={spy.actions_seen}")
check("SpyBot saw board cards", spy.boards_seen > 0,
      f"boards={spy.boards_seen}")
check("SpyBot saw showdowns (spectator learning!)",
      spy.showdowns_seen > 0,
      f"showdowns={spy.showdowns_seen}")

if spy.showdown_cards:
    first_sd = spy.showdown_cards[0]
    check("Showdown reveals opponent cards",
          any(seat != spy.seat for seat in first_sd),
          f"revealed_seats={list(first_sd.keys())}")
    print(f"    Showdowns observed: {spy.showdowns_seen}/50 hands")
    print(f"    Total actions observed: {spy.actions_seen}")


# ===================================================================
print("\n═══ ROUND-ROBIN TOURNAMENT ═══")
# ===================================================================

print("\n  4-bot round-robin (50 hands per matchup):")
tournament_bots = [
    HeroBot("Hero", seed=1),
    NitBot("Nit", seed=2),
    ManiacBot("Maniac", seed=3),
    CallingStation("Station", seed=4),
]
rr_stats = round_robin(
    bots=tournament_bots,
    hands_per_matchup=50,
    starting_stack=1000,
    small_blind=5,
    big_blind=10,
    deck_seed=500,
)

# 4 bots → C(4,2) = 6 matchups × 50 hands = 300 total
check("Round-robin: 300 total hands", rr_stats.total_hands == 300,
      f"total={rr_stats.total_hands}")
check("Round-robin: all 4 players tracked", len(rr_stats.players) == 4)

net_sum_rr = sum(ps.net_chips for ps in rr_stats.players.values())
check("Round-robin zero-sum", abs(net_sum_rr) < 10,
      f"net_sum={net_sum_rr}")

print(rr_stats.report())


# ===================================================================
print("\n═══ ERROR HANDLING ═══")
# ===================================================================

class BrokenBot(BaseBot):
    """Bot that raises exceptions."""
    def __init__(self):
        super().__init__("Broken")
    def act(self, gs):
        raise RuntimeError("I'm broken!")

print("\n  Broken bot is handled gracefully:")
runner_err = GameRunner(
    bots=[BrokenBot(), RandomBot(seed=1)],
    starting_stack=500,
    small_blind=5,
    big_blind=10,
    deck_seed=600,
)
try:
    stats_err = runner_err.run(num_hands=20)
    check("Broken bot: 20 hands completed without crash",
          stats_err.total_hands == 20)
except Exception as e:
    check("Broken bot handled", False, str(e))


# ===================================================================
print("\n═══ SEED FALLBACK — CRACK A SEED OUTSIDE 0–999 ═══")
# ===================================================================

import random
from deck import Deck
from environment_model import (EnvironmentAnalyzer, HandObservation,
                               MT19937Recovery)


def _observe_seed(seed, n_hands):
    d = Deck(seed=seed)
    obs = []
    for hi in range(n_hands):
        d.reset()
        a = d.deal(12)
        obs.append(HandObservation(hi, 2, 0, [a[0], a[1]],
                                   [a[5], a[6], a[7], a[9], a[11]],
                                   {1: [a[2], a[3]]}))
    return obs


print("\n  True seed 45678 (initial window is only 0–999):")
big_obs = _observe_seed(45678, 4)
an = EnvironmentAnalyzer(max_seed=1000, fallback_max_seed=100_000,
                         fallback_time_budget=30.0)
an.observe(big_obs[0])
check("Initial 0–999 window exhausted after hand 0", an.alive_count() <= 1)
check("Fallback was triggered", an._fallback_triggered)
check("Fallback cracked the out-of-window seed", an.best_seed == 45678,
      f"best={an.best_seed}")
for o in big_obs[1:]:
    an.observe(o)
check("Confident after fallback", an.is_confident)

pred = an.predict_next_hand(2, 0)
d = Deck(seed=45678)
for _ in range(5):
    d.reset()
    a = d.deal(12)
check("Fallback prediction matches actual next hand",
      pred is not None
      and pred["hero_cards"] == [a[0], a[1]]
      and pred["board"] == [a[5], a[6], a[7], a[9], a[11]]
      and pred["opponent_cards"][1] == [a[2], a[3]])

print("\n  Seed beyond the fallback range → graceful, no false crack:")
oor = _observe_seed(500_000, 3)
an_oor = EnvironmentAnalyzer(max_seed=1000, fallback_max_seed=20_000,
                            fallback_time_budget=10.0)
for o in oor:
    an_oor.observe(o)
check("Out-of-range seed: 0 candidates, 0 confidence",
      an_oor.alive_count() == 0 and an_oor.confidence == 0.0)
check("Out-of-range seed: no prediction returned",
      an_oor.predict_next_hand(2, 0) is None)


# ===================================================================
print("\n═══ MT19937 STATE RECOVERY — 624 OUTPUTS ═══")
# ===================================================================

def _temper(y):
    y ^= y >> 11
    y ^= (y << 7) & 0x9D2C5680
    y ^= (y << 15) & 0xEFC60000
    y ^= y >> 18
    return y & 0xFFFFFFFF

samples = [random.getrandbits(32) for _ in range(2000)]
check("untemper is the exact inverse of MT tempering",
      all(MT19937Recovery.untemper(_temper(v)) == v for v in samples))

src = random.Random(2024)
outs = [src.getrandbits(32) for _ in range(624)]
recovered = MT19937Recovery.from_outputs(outs)
check("State recovered from 624 outputs", recovered is not None)
check("Recovered generator predicts next 300 words exactly",
      all(recovered.getrandbits(32) == src.getrandbits(32) for _ in range(300)))

check("from_outputs needs at least 624 words",
      MT19937Recovery.from_outputs(outs[:600]) is None)

# The untemper attack is for full-word-leaking arenas; a 52-card shuffle
# leaks only the top ≤6 bits per word, so it is not recoverable this way.
class _Counter(random.Random):
    def __init__(self, seed):
        super().__init__(seed); self.widths = []
    def getrandbits(self, k):
        self.widths.append(k); return super().getrandbits(k)

cnt = _Counter(1)
deck = list(range(52)); cnt.shuffle(deck)
check("A 52-card shuffle leaks only ≤6-bit words (not untemperable)",
      cnt.widths and max(cnt.widths) <= 6)


# ===================================================================
print(f"\n{'═'*60}")
print(f"  Results: {passed} passed, {failed} failed")
print(f"{'═'*60}\n")
sys.exit(1 if failed else 0)

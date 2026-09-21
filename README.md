# Adaptive Adversarial Poker Bot

A modular, implementation-ready poker bot framework that combines Monte Carlo
equity estimation, Bayesian opponent-range inference, exploitative decision
making, and optional RNG inference — all in pure Python with no dependencies.

## Quick Start

```bash
# Run the demo (tournament + OracleBot showcase)
cd poker_bot
python demo.py

# Run all tests (166 tests across 4 suites)
python run_tests.py

# Run a single test suite
python test_phase1.py
python test_phase2.py
python test_environment.py
python test_phase6.py
```

## Architecture

```
GAME ENVIRONMENT (game_runner.py)
 │
 ├─ Deck (deck.py)              → shuffles, deals cards
 ├─ GameState (game_state.py)    → blinds, streets, legal actions, pot/stacks
 └─ HandEvaluator (hand_evaluator.py) → ranks 5-7 card poker hands
     │
     ▼
 BOT INTERFACE (bot_interface.py)
 │
 ├─ observe_hand_start()   → hole cards, positions, stacks
 ├─ act(game_state)        → return an Action
 ├─ observe_action()       → see every action at the table ──┐
 ├─ observe_board()        → see flop / turn / river         ├─→ SPECTATOR
 ├─ observe_showdown()     → see revealed cards              │   LEARNER
 └─ observe_hand_end()     → final results ──────────────────┘   (spectator.py)
     │                                                            │
     ▼                                                            ▼
 DECISION PIPELINE                                    OPPONENT MODELS
 │                                                    (opponent_model.py)
 ├─ Equity (equity.py)                → MC simulation     │
 ├─ BoardTexture (board_texture.py)   → wet/dry, draws    ├─ BetaStat priors
 ├─ DecisionEngine (decision_engine.py) → Q(s,a)         ├─ EWMA recency
 ├─ StrategyController (strategy.py)  → softmax policy ◄──┤ Type classification
 └─ EnvironmentModel (environment_model.py) → RNG crack   └─ Strategy suggestion
     │
     ▼
 ACTION → fold / check / call / bet / raise / all-in
```

## File Map

### Phase 1 — Correct Poker Engine
| File | Lines | Description |
|---|---|---|
| `hand_evaluator.py` | 136 | Card encoding (0-51), 5-7 card evaluation, all hand categories |
| `deck.py` | 50 | 52-card deck with seeded RNG, deal, shuffle |
| `game_state.py` | 390 | Full Hold'em state machine: blinds, positions, streets, legal actions, pot/stack accounting |
| `equity.py` | 152 | Monte Carlo equity estimator, pot odds, call EV, bluff EV |

### Phase 2 — EV Decision Engine
| File | Lines | Description |
|---|---|---|
| `board_texture.py` | 155 | Board feature vector: paired, suited, connected, wetness score |
| `decision_engine.py` | 483 | Q(s,a) estimation: fold equity, bet sizing, SPR, position, implied odds |
| `strategy.py` | 336 | Softmax policy, temperature scheduling, strategy modes (balanced/value/aggressive/trap) |

### Phase 4 — Spectator Learning & Opponent Modeling
| File | Lines | Description |
|---|---|---|
| `opponent_model.py` | 310 | Per-opponent Bayesian model: Beta stats, EWMA recency, bet-size entropy, type classification |
| `spectator.py` | 235 | Processes ALL hands (played + folded): action stat extraction, showdown bluff detection |

### Game Environment
| File | Lines | Description |
|---|---|---|
| `bot_interface.py` | 114 | Abstract BaseBot class with 6 observation hooks |
| `opponents.py` | 380 | 6 synthetic opponents + HeroBot wrapper |
| `game_runner.py` | 442 | Arena: deals cards, calls bots, resolves showdowns, round-robin |
| `stats.py` | 252 | Per-player tracking: bb/hand, win rate, VPIP, PFR, head-to-head matrix |

### Phase 6 — Environment/RNG Analysis
| File | Lines | Description |
|---|---|---|
| `environment_model.py` | 542 | Bayesian seed inference across multiple PRNG + protocol hypotheses |
| `oracle_bot.py` | 193 | HeroBot + RNG prediction (exact opponent hands, board runout) |

### Tests & Demo
| File | Lines | Description |
|---|---|---|
| `test_phase1.py` | 285 | 57 tests: hand rankings, game state, legal actions, equity |
| `test_phase2.py` | 386 | 40 tests: board texture, EV engine, strategy, bet sizing, SPR |
| `test_phase4.py` | 310 | 36 tests: Bayesian stats, fingerprinting, spectator learning, adaptation |
| `test_environment.py` | 312 | 41 tests: all opponents, game runner, stats, spectator protocol |
| `test_phase6.py` | 394 | 28 tests: seed elimination, prediction, OracleBot, ablation |
| `run_tests.py` | 48 | Runs all 5 suites, prints combined summary |
| `demo.py` | 170 | Round-robin tournament + OracleBot showcase + ablation |

**Total: ~6,000 lines, 202 tests, 0 external dependencies.**

## Synthetic Opponents

| Bot | VPIP | Strategy | Exploit |
|---|---|---|---|
| RandomBot | ~50% | Random legal actions | Baseline — loses to everything |
| NitBot | ~40% | Premium hands only, folds to aggression | Steal blinds, fold when they bet |
| CallingStation | ~90% | Calls almost everything | Value bet relentlessly, never bluff |
| ManiacBot | ~80% | Bets/raises 70% with any hand | Trap with strong hands, call down |
| GTOLikeBot | ~65% | Fixed ranges, 2/3-pot c-bet, balanced | Detect rigid frequencies |
| RigidBot | ~45% | Pure if/else thresholds, no randomization | Identify thresholds exactly |

## Key Design Decisions

- **Zero dependencies**: Runs on any Python 3.8+ installation
- **Modular**: Each component is independently testable and replaceable
- **Spectator learning**: Bots receive showdown data even from hands they folded
- **Bayesian inference**: Beta priors prevent overreacting to small samples
- **Softmax randomization**: Prevents exploitable deterministic policies
- **Multi-hypothesis RNG cracking**: Tries multiple PRNG implementations, dealing
  protocols, and initial-state warmup patterns simultaneously
- **Graceful degradation**: OracleBot falls back to standard play if RNG
  inference fails (seed outside search space, wrong PRNG, etc.)

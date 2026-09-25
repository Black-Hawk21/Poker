"""
Adaptive Adversarial Poker Bot
==============================
A modular poker bot framework implementing:

  Phase 1 — Correct Poker Engine
    hand_evaluator    Card encoding, fast + reference 5-7 card evaluation
    hand_strength     Percentile hand-strength tables (preflop + per board)
    deck              Shuffleable 52-card deck
    game_state        Full Hold'em state machine (blinds, streets, legal actions)
    equity            Monte Carlo equity estimator + pot odds helpers

  Phase 2 — EV Decision Engine
    board_texture     Community board feature vector (wetness, draws, pairing)
    action_likelihood Explicit P(A|h,s,M) model (§12)
    utility           Chip / concave / ICM utilities (§8, §20)
    decision_engine   Q(s,a)=Σ_r P(r|a)·E[U] response-model engine (§20)
    strategy          Near-optimal randomization + regret matching (§7)
    exploiter         MDF-loop exploitation, confidence-scaled (§6, §22)
    calibration       Reliability curve / ECE for predictions (§28)

  Phase 4 — Spectator Learning & Opponent Modeling
    opponent_model    Per-opponent Bayesian stats, fingerprinting, type classification
    spectator         Learns from ALL hands (played + folded), showdown extraction

  Game Environment
    bot_interface     Abstract BaseBot protocol (act + observation hooks)
    opponents         6 synthetic opponents (Random, Nit, CallingStation, Maniac,
                      GTOLike, RigidBot) + HeroBot wrapper
    game_runner       Arena: deals cards, runs hands, resolves showdowns
    stats             Per-player / per-matchup tracking and reporting

  Phase 6 — Environment / RNG Analysis
    environment_model Bayesian seed inference from observed cards
    oracle_bot        HeroBot + RNG prediction integration

Quick start:
    from poker_bot.game_runner import GameRunner
    from poker_bot.opponents import HeroBot, ManiacBot
    runner = GameRunner(bots=[HeroBot("Hero"), ManiacBot("Villain")])
    stats = runner.run(num_hands=200)
    print(stats.report())
"""

__version__ = "0.8.0"

"""
Adaptive Adversarial Poker Bot
==============================
A modular poker bot framework implementing:

  Phase 1 — Correct Poker Engine
    hand_evaluator    Card encoding, 5-7 card hand evaluation
    deck              Shuffleable 52-card deck
    game_state        Full Hold'em state machine (blinds, streets, legal actions)
    equity            Monte Carlo equity estimator + pot odds helpers

  Phase 2 — EV Decision Engine
    board_texture     Community board feature vector (wetness, draws, pairing)
    decision_engine   Q(s,a) estimation with fold equity, sizing, SPR awareness
    strategy          Softmax randomized policy + strategy modes

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

__version__ = "0.7.0"

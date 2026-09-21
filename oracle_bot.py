"""
OracleBot — RNG-Aware Adaptive Bot
===================================
Extends HeroBot with Phase 6 environment analysis.

When the RNG is cracked:
  - knows its own future hole cards (can pre-plan)
  - knows the exact board runout (perfect equity on future streets)
  - knows opponent hole cards (perfect range = single hand)

Section 20 caveat: knowing future cards does NOT mean knowing future
opponent ACTIONS.  The decision engine still models opponent behavior.

The bot falls back to standard HeroBot play when RNG confidence is low.
"""

from __future__ import annotations
from typing import Optional

from bot_interface import BaseBot, HandStartInfo, ShowdownInfo, HandEndInfo
from game_state import GameState, Action, ActionType, Street
from strategy import StrategyController, StrategyMode, EVConfig, StrategyConfig
from environment_model import EnvironmentAnalyzer, HandObservation
from hand_evaluator import card_str


class OracleBot(BaseBot):
    """
    Phase 1-2 adaptive bot + Phase 6 environment inference.

    Parameters
    ----------
    max_seed : upper bound of seed search space
    rng_enabled : set False to disable RNG analysis (ablation)
    confidence_threshold : min confidence to use predictions
    """

    def __init__(
        self,
        name: str = "Oracle",
        max_seed: int = 1000,
        rng_enabled: bool = True,
        confidence_threshold: float = 0.90,
        mode: StrategyMode = StrategyMode.BALANCED,
        seed: Optional[int] = None,
    ):
        super().__init__(name)
        self._controller = StrategyController(
            ev_config=EVConfig(
                equity_simulations=500,
                equity_simulations_important=1000,
            ),
            rng_seed=seed,
        )
        self._controller.mode = mode
        self.rng_enabled = rng_enabled

        # Phase 6 analyzer
        self._analyzer = EnvironmentAnalyzer(
            max_seed=max_seed,
            confidence_threshold=confidence_threshold,
        ) if rng_enabled else None

        # Current hand tracking
        self._hand_index = -1
        self._current_obs: Optional[HandObservation] = None
        self._num_players = 2
        self._prediction: Optional[dict] = None

        # Stats
        self.predictions_made = 0
        self.predictions_used = 0

        # Phase 4: spectator learning
        from spectator import SpectatorLearner
        self._learner = SpectatorLearner()

    # ------------------------------------------------------------------
    # Observation hooks — feed data to both analyzer and learner
    # ------------------------------------------------------------------
    def observe_hand_start(self, info: HandStartInfo):
        super().observe_hand_start(info)
        self._hand_index += 1
        self._num_players = info.num_players
        self._learner.on_hand_start(info)

        # Start building the observation for this hand
        self._current_obs = HandObservation(
            hand_index=self._hand_index,
            num_players=info.num_players,
            hero_seat=info.seat,
            hero_cards=list(info.hole_cards),
            board=[],
            showdown_cards={},
        )

        # Check if we have a prediction for this hand
        self._prediction = None
        if self._analyzer and self._analyzer.is_confident:
            pred = self._analyzer.predict_next_hand(
                num_players=info.num_players,
                hero_seat=info.seat,
            )
            if pred and pred["hand_index"] == self._hand_index:
                # Validate: our predicted hero cards should match actual
                if pred["hero_cards"] == list(info.hole_cards):
                    self._prediction = pred
                    self.predictions_made += 1

    def observe_board(self, street: Street, board: list[int]):
        if self._current_obs is not None:
            self._current_obs.board = list(board)
        self._learner.on_board(street, board)

    def observe_showdown(self, info: ShowdownInfo):
        if self._current_obs is not None:
            for seat, cards in info.revealed_cards.items():
                self._current_obs.showdown_cards[seat] = list(cards)
        self._learner.on_showdown(info)

    def observe_hand_end(self, info: HandEndInfo):
        # Feed completed observation to the analyzer
        if self._current_obs is not None and self._analyzer is not None:
            # Update board from hand end info (in case we folded early)
            if info.board:
                self._current_obs.board = list(info.board)
            # Include showdown info if available
            if info.showdown:
                for seat, cards in info.showdown.revealed_cards.items():
                    self._current_obs.showdown_cards[seat] = list(cards)

            self._analyzer.observe(self._current_obs)

            # Purge dead candidates periodically to save memory
            if self._hand_index % 5 == 0:
                self._analyzer.purge_dead()

        # Phase 4: spectator learning
        self._learner.on_hand_end(info)

    # ------------------------------------------------------------------
    # Decision making
    # ------------------------------------------------------------------
    def act(self, gs: GameState) -> Action:
        """
        Make a decision, using predicted cards when confident.

        When we know the board and opponent cards:
          - opponent_ranges collapses to their exact hand
          - equity becomes exact (we know if we win or lose)
          - BUT opponent actions are still uncertain (Section 20)
        """
        if self._prediction is not None:
            return self._act_with_prediction(gs)
        else:
            return self._act_standard(gs)

    def _act_standard(self, gs: GameState) -> Action:
        """Standard Phase 2 decision (no RNG knowledge)."""
        decision = self._controller.decide(gs, hero=self.seat)
        return StrategyController.to_action(decision, self.seat, gs.street)

    def _act_with_prediction(self, gs: GameState) -> Action:
        """
        Decision with known cards.

        We construct opponent_ranges as the single known hand,
        giving the equity estimator perfect information.
        """
        self.predictions_used += 1
        pred = self._prediction

        # Build exact opponent ranges (single hand per opponent)
        opponent_ranges = []
        for seat in range(self._num_players):
            if seat == self.seat:
                continue
            opp_cards = pred["opponent_cards"].get(seat)
            if opp_cards:
                # Range is a single hand
                opponent_ranges.append([(opp_cards[0], opp_cards[1])])
            else:
                opponent_ranges.append(None)

        # Use the strategy controller with the exact opponent range
        decision = self._controller.decide(
            gs,
            hero=self.seat,
            opponent_ranges=opponent_ranges if opponent_ranges else None,
        )
        return StrategyController.to_action(decision, self.seat, gs.street)

    # ------------------------------------------------------------------
    # Public status
    # ------------------------------------------------------------------
    def analyzer_status(self) -> dict:
        """Return the analyzer's current state."""
        if self._analyzer is None:
            return {"enabled": False}
        status = self._analyzer.status()
        status["predictions_made"] = self.predictions_made
        status["predictions_used"] = self.predictions_used
        return status

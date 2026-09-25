"""
OracleBot — RNG-Aware Adaptive Bot  (design doc §24, §23.6)
==========================================================
HeroBot plus the optional environment layer.  When the analyzer is
confident about the deck stream, the *board* it will run out is fed to the
decision engine as one more feature, weighted by the analyzer's confidence
(Eq. 37–38): the future-board distribution narrows toward a point mass as
confidence rises, and the engine degrades to pure poker when it is low.

Deliberately not done (§24, §23.6):

* Opponent hole cards are NOT used to collapse ranges to a single hand.
  Knowing the board does not tell you opponents' actions, and betting as if
  you can see their cards is the clearest possible tell.  Ranges are still
  inferred from actions exactly as in HeroBot.
* The predicted board is blended, never followed: acting perfectly on cards
  you did not "see" leaves a statistically detectable footprint (§23.6).

If RNG analysis is disabled or unconfident, this is just HeroBot.
"""

from __future__ import annotations
from typing import Optional

from opponents import HeroBot
from bot_interface import HandStartInfo, ShowdownInfo, HandEndInfo
from game_state import GameState, Action, Street
from strategy import StrategyController, StrategyMode, EVConfig
from environment_model import EnvironmentAnalyzer, HandObservation


class OracleBot(HeroBot):
    """HeroBot + confidence-gated environment/RNG feature."""

    def __init__(self, name: str = "Oracle", max_seed: int = 1000,
                 rng_enabled: bool = True, confidence_threshold: float = 0.90,
                 mode: StrategyMode = StrategyMode.BALANCED,
                 seed: Optional[int] = None, learning: bool = True,
                 fallback_enabled: bool = True,
                 fallback_max_seed: int = 100_000):
        super().__init__(name, mode=mode, seed=seed, learning=learning)
        self.rng_enabled = rng_enabled
        self.confidence_threshold = confidence_threshold
        # The analyzer first sweeps 0..max_seed cheaply, then — only if that
        # is exhausted and fallback_enabled — widens to fallback_max_seed
        # using accumulated card observations. The widening costs nothing when
        # the seed is small; disable it to force pure graceful fallback.
        self._analyzer = (EnvironmentAnalyzer(max_seed=max_seed,
                                              confidence_threshold=confidence_threshold,
                                              fallback_enabled=fallback_enabled,
                                              fallback_max_seed=fallback_max_seed)
                          if rng_enabled else None)
        self._hand_index = -1
        self._current_obs: Optional[HandObservation] = None
        self._predicted_board: Optional[list[int]] = None
        self._runout_weight: float = 0.0
        self.predictions_made = 0
        self.predictions_used = 0

    # ------------------------------------------------------------------
    def observe_hand_start(self, info: HandStartInfo):
        super().observe_hand_start(info)
        self._hand_index += 1
        self._current_obs = HandObservation(
            hand_index=self._hand_index, num_players=info.num_players,
            hero_seat=info.seat, hero_cards=list(info.hole_cards), board=[])
        self._predicted_board = None
        self._runout_weight = 0.0
        if self._analyzer is not None and self._analyzer.is_confident:
            pred = self._analyzer.predict_next_hand(info.num_players, info.seat)
            # Only trust the prediction if it names our actual hole cards.
            if pred and pred["hand_index"] == self._hand_index \
                    and pred["hero_cards"] == list(info.hole_cards):
                self._predicted_board = list(pred["board"])
                self._runout_weight = float(self._analyzer.confidence)
                self.predictions_made += 1

    def observe_board(self, street, board):
        super().observe_board(street, board)
        if self._current_obs is not None:
            self._current_obs.board = list(board)

    def observe_showdown(self, info: ShowdownInfo):
        super().observe_showdown(info)
        if self._current_obs is not None:
            for seat, cards in info.revealed_cards.items():
                self._current_obs.showdown_cards[seat] = list(cards)

    def observe_hand_end(self, info: HandEndInfo):
        if self._current_obs is not None and self._analyzer is not None:
            if info.board:
                self._current_obs.board = list(info.board)
            if info.showdown:
                for seat, cards in info.showdown.revealed_cards.items():
                    self._current_obs.showdown_cards[seat] = list(cards)
            self._analyzer.observe(self._current_obs)
            if self._hand_index % 5 == 0:
                self._analyzer.purge_dead()
        super().observe_hand_end(info)

    # ------------------------------------------------------------------
    def act(self, gs: GameState) -> Action:
        self._decisions += 1
        kw = self.decision_inputs(gs)
        if self._predicted_board and self._runout_weight > 0 and gs.street < Street.RIVER:
            need = 5 - len(gs.board)
            run = self._predicted_board[len(gs.board):len(gs.board) + need] \
                if len(self._predicted_board) >= 5 else []
            # Trust the prediction only if it matches the board seen so far.
            if list(gs.board) == self._predicted_board[:len(gs.board)] and len(run) == need:
                kw["predicted_runout"] = run
                kw["runout_weight"] = self._runout_weight
                self.predictions_used += 1
        seed = None if self._seed is None else self._seed * 100003 + self._decisions
        decision = self._controller.decide(gs, hero=self.seat, rng_seed=seed, **kw)
        return StrategyController.to_action(decision, self.seat, gs.street)

    # ------------------------------------------------------------------
    def analyzer_status(self) -> dict:
        if self._analyzer is None:
            return {"enabled": False, "cracked": False, "candidates_alive": 0,
                    "confidence": 0.0, "predictions_made": 0, "predictions_used": 0,
                    "best_seed": None, "entropy_bits": 0.0}
        s = self._analyzer.status()
        s["enabled"] = True
        s["predictions_made"] = self.predictions_made
        s["predictions_used"] = self.predictions_used
        return s

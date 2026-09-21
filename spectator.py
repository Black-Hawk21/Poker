"""
Spectator Learning — Section 13
================================
"Folding does not terminate information gathering."

Processes every hand the bot observes — whether it played, folded
preflop, or folded on a later street — and extracts training data
for opponent models.

Equation 47:  D_t = {positions, actions, sizes, board, showdown, outcome}
Equation 48:  M_i^{t+1} = U(M_i^t, D_t)
Equation 81:  IG(D) = H(M_i) - H(M_i | D)

Information sources ranked by value:
  1. Showdown reveals (we SEE their cards — ground truth)
  2. Actions in hands we're still in (real-time)
  3. Actions in hands we've folded from (spectator)
  4. Bet sizing patterns (always visible)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from game_state import GameState, Action, ActionType, Street
from bot_interface import HandStartInfo, ShowdownInfo, HandEndInfo
from hand_evaluator import evaluate, HandRank
from opponent_model import OpponentModel, OpponentModelSet, PopulationPrior


# ---------------------------------------------------------------------------
# Hand record (Section 22)
# ---------------------------------------------------------------------------
@dataclass
class HandRecord:
    """
    All observable data from one hand (Equation 47).
    Built incrementally as actions and events arrive.
    """
    hand_number: int = 0
    num_players: int = 2
    hero_seat: int = 0
    positions: list[str] = field(default_factory=list)
    board: list[int] = field(default_factory=list)
    pot: int = 0

    # Per-player data
    hole_cards: dict[int, list[int]] = field(default_factory=dict)
    actions: list[Action] = field(default_factory=list)

    # Tracking who did what preflop
    preflop_raiser: int = -1   # seat of last preflop raiser
    preflop_voluntary: set = field(default_factory=set)

    # Per-street pot snapshots (for bet-size ratios)
    pot_at_street: dict[int, int] = field(default_factory=dict)

    # Showdown
    went_to_showdown: bool = False
    showdown_cards: dict[int, list[int]] = field(default_factory=dict)
    winners: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Spectator Learner
# ---------------------------------------------------------------------------
class SpectatorLearner:
    """
    Observes all hands and updates opponent models.

    Wired into the BaseBot observation hooks:
      - on_hand_start()     → reset hand record
      - on_action()         → extract stats from each action
      - on_board()          → track street transitions
      - on_showdown()       → ground-truth opponent cards
      - on_hand_end()       → finalize and commit updates

    The hero's own actions are skipped for opponent modeling.
    """

    def __init__(self, hero_seat: int = -1,
                 prior: Optional[PopulationPrior] = None):
        self.hero_seat = hero_seat
        self.models = OpponentModelSet(prior)
        self._record: Optional[HandRecord] = None
        self._current_pot: int = 0
        self._hero_folded: bool = False

        # Stats
        self.hands_processed: int = 0
        self.showdowns_observed: int = 0
        self.spectator_hands: int = 0  # hands learned from after folding

    # ------------------------------------------------------------------
    # Hook: hand start
    # ------------------------------------------------------------------
    def on_hand_start(self, info: HandStartInfo):
        self.hero_seat = info.seat
        self._hero_folded = False
        self._current_pot = info.small_blind + info.big_blind

        self._record = HandRecord(
            hand_number=info.hand_number,
            num_players=info.num_players,
            hero_seat=info.seat,
            positions=list(info.positions),
        )
        self._record.hole_cards[info.seat] = list(info.hole_cards)
        self._record.pot_at_street[Street.PREFLOP] = self._current_pot

    # ------------------------------------------------------------------
    # Hook: action observed
    # ------------------------------------------------------------------
    def on_action(self, action: Action, game_state: GameState):
        if self._record is None:
            return

        self._record.actions.append(action)
        player = action.player
        self._current_pot = game_state.pot

        # Skip hero's own actions for opponent modeling
        if player == self.hero_seat:
            if action.action_type == ActionType.FOLD:
                self._hero_folded = True
            return

        model = self.models.get(player)
        position = (self._record.positions[player]
                    if player < len(self._record.positions) else "?")

        if action.street == Street.PREFLOP:
            self._process_preflop_action(model, action, game_state, position)
        else:
            self._process_postflop_action(model, action, game_state)

    def _process_preflop_action(self, model: OpponentModel, action: Action,
                                gs: GameState, position: str):
        """Extract preflop stats from one action."""
        at = action.action_type
        facing_raise = gs.max_current_bet > gs.big_blind
        is_voluntary = at in (ActionType.CALL, ActionType.RAISE,
                              ActionType.BET, ActionType.ALL_IN)
        is_raise = at in (ActionType.RAISE, ActionType.BET, ActionType.ALL_IN)

        model.record_preflop_action(
            action_type=at,
            position=position,
            is_voluntary=is_voluntary,
            is_raise=is_raise,
            facing_raise=facing_raise,
        )

        if is_raise:
            self._record.preflop_raiser = action.player
        if is_voluntary:
            self._record.preflop_voluntary.add(action.player)

    def _process_postflop_action(self, model: OpponentModel, action: Action,
                                 gs: GameState):
        """Extract post-flop stats from one action."""
        at = action.action_type
        street = action.street

        # Track pot at each new street
        if street not in self._record.pot_at_street:
            self._record.pot_at_street[street] = self._current_pot

        pot_before = self._record.pot_at_street.get(street, self._current_pot)

        # Determine if this player was the preflop aggressor
        is_aggressor = action.player == self._record.preflop_raiser

        # Determine what they're facing
        facing_bet = gs.max_current_bet > gs.current_bets[action.player]
        facing_raise = facing_bet and any(
            a.action_type in (ActionType.RAISE, ActionType.ALL_IN)
            for a in self._record.actions
            if a.street == street and a.player != action.player
        )

        # Compute bet amount relative to pot
        bet_amount = 0
        if at in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN):
            bet_amount = action.amount - gs.current_bets.get(action.player, 0) \
                if hasattr(gs.current_bets, 'get') else action.amount

        model.record_postflop_action(
            action_type=at,
            street=street,
            bet_amount=bet_amount,
            pot_before=pot_before,
            is_aggressor=is_aggressor,
            facing_bet=facing_bet,
            facing_raise=facing_raise,
        )

    # ------------------------------------------------------------------
    # Hook: board dealt
    # ------------------------------------------------------------------
    def on_board(self, street: Street, board: list[int]):
        if self._record is not None:
            self._record.board = list(board)
            self._record.pot_at_street[street] = self._current_pot

    # ------------------------------------------------------------------
    # Hook: showdown
    # ------------------------------------------------------------------
    def on_showdown(self, info: ShowdownInfo):
        if self._record is None:
            return

        self._record.went_to_showdown = True
        self._record.showdown_cards = dict(info.revealed_cards)
        self._record.winners = list(info.winners)
        self._record.board = list(info.board)
        self.showdowns_observed += 1

        # Process showdown results for each revealed opponent
        for seat, cards in info.revealed_cards.items():
            if seat == self.hero_seat:
                continue

            model = self.models.get(seat)
            won = seat in info.winners

            # Determine if the opponent was bluffing at showdown
            # A "bluff" = they were betting/raising with a losing hand
            was_bluffing = self._detect_bluff(seat, cards, info)
            model.record_showdown(won=won, was_bluffing=was_bluffing)

    def _detect_bluff(self, seat: int, cards: list[int],
                      info: ShowdownInfo) -> Optional[bool]:
        """
        Determine if the opponent was bluffing.
        A bluff = they were the aggressor (bet/raised) and lost.

        This is the high-value observation from Section 21 (Equation 82):
            P(bluff | river, board, sizing, history)
        """
        if not self._record or not self._record.actions:
            return None

        # Check if they were betting/raising in the last street before showdown
        last_street_actions = [
            a for a in self._record.actions
            if a.player == seat and a.action_type in (
                ActionType.BET, ActionType.RAISE, ActionType.ALL_IN
            )
        ]

        if not last_street_actions:
            return None  # they were passive — can't determine bluff

        # They were aggressive — did they lose?
        was_aggressor = len(last_street_actions) > 0
        lost = seat not in info.winners

        return was_aggressor and lost

    # ------------------------------------------------------------------
    # Hook: hand end
    # ------------------------------------------------------------------
    def on_hand_end(self, info: HandEndInfo):
        if self._record is None:
            return

        # Finalize all opponent models for this hand
        for seat in range(self._record.num_players):
            if seat == self.hero_seat:
                continue
            model = self.models.get(seat)
            model.finish_hand()

        if self._hero_folded:
            self.spectator_hands += 1

        self.hands_processed += 1
        self._record = None

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def get_model(self, player_id: int) -> OpponentModel:
        return self.models.get(player_id)

    def get_fold_estimate(self, player_id: int,
                          facing_raise: bool = False) -> float:
        """
        Get estimated fold probability for an opponent.
        Uses individual model when confident, falls back to population prior.
        """
        model = self.models.get(player_id)
        if facing_raise:
            return model.estimated_fold_to_raise()
        return model.estimated_fold_to_bet()

    def get_opponent_type(self, player_id: int) -> str:
        return self.models.get(player_id).primary_type()

    def suggest_strategy(self, player_id: int) -> str:
        """
        Suggest an exploitative strategy mode based on opponent type.
        Maps to the strategy modes from Section 18 (Equations 65-68).
        """
        model = self.models.get(player_id)
        otype = model.primary_type()

        # Section 18 mappings
        if otype == "passive":
            return "value_heavy"        # Equation 65: calling station → value
        elif otype == "tight":
            return "aggressive"         # Equation 66: nit → bluff-heavy
        elif otype == "aggressive":
            return "trap_heavy"         # Equation 67: maniac → trap
        elif otype in ("balanced", "gto_like"):
            return "balanced"           # Equation 68: balanced → balanced
        else:
            return "balanced"

    def report(self) -> str:
        lines = [
            f"Spectator Learner: {self.hands_processed} hands processed, "
            f"{self.showdowns_observed} showdowns, "
            f"{self.spectator_hands} spectator hands",
        ]
        lines.append(self.models.report())
        return "\n".join(lines)

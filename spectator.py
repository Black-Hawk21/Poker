"""
Spectator Learning  (design doc §15, §25)
=========================================
Folding does not stop information gathering: every observable hand —
played, folded preflop, folded later, or contested only by others — updates
the opponent models.

    D_t = {positions, actions, sizes, board, showdown, outcome}   (Eq. 28)
    M_i^(t+1) = U(M_i^(t), D_t)
    IG(D) = H(M_i) − H(M_i | D)                                   (Eq. 39)

Fixes relative to the original code:

* **Pre-action context.**  Observers receive an action after it has been
  applied, so "was the player facing a bet / a raise?" and the pot they
  faced must come from the Action's recorded pre-action fields.  The old
  code read the post-action state: every preflop open-raise was logged as a
  3-bet (the raiser "faced" its own raise), calls were logged as not facing
  a bet, and sizes were divided by the wrong pot.
* **VPIP / PFR are per-hand statistics** (one observation per player per
  hand), not per action.
* **Conditional cells.**  Every decision is recorded in the same
  (decision type × street × position × texture × prior action) cell that
  range inference reads from (§10.2).
* **Fold-to-bet by size** is recorded with the size actually faced.
* **Showdown ground truth.**  A revealed hand is replayed street by street:
  its strength percentile at each decision feeds the size-given-strength
  table (§12.3) and the threshold-consistency fingerprint (§14).  A bluff
  is a bet/raise made with a hand in the bottom half of holdings on that
  street — not "was aggressive and lost", which labelled every beaten value
  bet a bluff.
* **Information value (§25).**  The entropy drop of the type posterior is
  recorded per showdown, and the most informative observations are kept.
"""

from __future__ import annotations
import heapq
from dataclasses import dataclass, field
from typing import Optional

from game_state import GameState, Action, ActionType, Street
from bot_interface import HandStartInfo, ShowdownInfo, HandEndInfo
from hand_strength import hand_strength
from opponent_model import (
    OpponentModel, OpponentModelSet, PopulationPrior, decision_cell, action_kind,
)

BLUFF_STRENGTH = 0.50   # bets below this percentile count as bluffs


# ---------------------------------------------------------------------------
# Hand record (§26)
# ---------------------------------------------------------------------------
@dataclass
class HandRecord:
    hand_number: int = 0
    num_players: int = 2
    hero_seat: int = 0
    positions: list[str] = field(default_factory=list)
    board: list[int] = field(default_factory=list)
    boards: dict[int, list[int]] = field(default_factory=dict)   # street → board
    pot: int = 0
    stacks: list[int] = field(default_factory=list)

    hole_cards: dict[int, list[int]] = field(default_factory=dict)
    actions: list[Action] = field(default_factory=list)

    preflop_raiser: int = -1
    preflop_voluntary: set = field(default_factory=set)
    # per-seat preflop summary: seat → dict(vpip, pfr, faced_raise, three_bet, first)
    preflop: dict = field(default_factory=dict)
    # aggressor of each street (last player to bet/raise)
    street_aggressor: dict[int, int] = field(default_factory=dict)
    acted_on_street: dict[int, set] = field(default_factory=dict)

    went_to_showdown: bool = False
    showdown_cards: dict[int, list[int]] = field(default_factory=dict)
    winners: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Spectator Learner
# ---------------------------------------------------------------------------
class SpectatorLearner:
    """Observes every hand and updates opponent models (hero's own actions skipped)."""

    def __init__(self, hero_seat: int = -1,
                 prior: Optional[PopulationPrior] = None,
                 keep_top_observations: int = 50):
        self.hero_seat = hero_seat
        self.models = OpponentModelSet(prior)
        self._record: Optional[HandRecord] = None
        self._hero_folded: bool = False

        self.hands_processed: int = 0
        self.showdowns_observed: int = 0
        self.spectator_hands: int = 0          # hands learned from after folding
        self.spectator_showdowns: int = 0      # showdowns seen after hero folded

        # §25 information value bookkeeping
        self.information_gain_total: float = 0.0
        self._keep = keep_top_observations
        self._top_observations: list[tuple[float, int, int]] = []   # min-heap

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def on_hand_start(self, info: HandStartInfo):
        self.hero_seat = info.seat
        self._hero_folded = False
        self._record = HandRecord(
            hand_number=info.hand_number,
            num_players=info.num_players,
            hero_seat=info.seat,
            positions=list(info.positions),
            stacks=list(info.stacks),
        )
        self._record.hole_cards[info.seat] = list(info.hole_cards)
        self._record.boards[0] = []

    def on_action(self, action: Action, game_state: Optional[GameState] = None):
        rec = self._record
        if rec is None:
            return
        rec.actions.append(action)
        p = action.player
        street = int(action.street)
        rec.acted_on_street.setdefault(street, set())
        first_on_street = p not in rec.acted_on_street[street]
        rec.acted_on_street[street].add(p)

        aggressive = action.is_aggressive
        prev_aggressor = rec.street_aggressor.get(street - 1) if street > 0 else None
        if aggressive:
            rec.street_aggressor[street] = p
            if street == 0:
                rec.preflop_raiser = p

        if p == self.hero_seat:
            if action.action_type == ActionType.FOLD:
                self._hero_folded = True
            return

        model = self.models.get(p)
        position = rec.positions[p] if p < len(rec.positions) else "MP"
        facing = action.to_call > 0
        kind = action_kind(action.action_type, aggressive, facing)

        if street == 0:
            self._preflop(model, rec, action, position, kind)
            return

        board = rec.boards.get(street, rec.board)
        was_agg = prev_aggressor == p
        decision, cell = decision_cell(street, position, board, was_agg, facing,
                                       action.raises_before)
        denom = action.pot_before - action.to_call
        faced = (action.to_call / denom) if (facing and denom > 0) else None
        size = action.size_fraction if aggressive else faced
        cbet_opp = (was_agg and first_on_street and not facing
                    and action.raises_before == 0)
        model.record_postflop_action(
            action_type=action.action_type,
            street=street,
            bet_amount=max(0, action.amount - action.bet_before) if aggressive else action.to_call,
            pot_before=action.pot_before,
            is_aggressor=was_agg,
            facing_bet=facing,
            facing_raise=action.raises_before >= 2,
            size_fraction=size,
            cell=cell,
            cbet_opportunity=cbet_opp,
        )

    def _preflop(self, model: OpponentModel, rec: HandRecord, action: Action,
                 position: str, kind: str):
        decision, cell = decision_cell(0, position, [], False, action.to_call > 0,
                                       action.raises_before)
        if kind == "bet":
            kind = "raise"
        if kind == "check":
            kind = "call"          # BB option: stayed in without raising
        model.record_preflop_decision((decision,) + cell, kind)

        st = rec.preflop.setdefault(action.player, {
            "vpip": False, "pfr": False, "faced_raise": False,
            "three_bet": False, "first": None, "position": position})
        if st["first"] is None:
            st["first"] = kind
        if action.raises_before >= 1:
            st["faced_raise"] = True
        if action.action_type in (ActionType.CALL, ActionType.RAISE,
                                  ActionType.BET, ActionType.ALL_IN):
            st["vpip"] = True
            rec.preflop_voluntary.add(action.player)
        if action.is_aggressive:
            st["pfr"] = True
            if action.raises_before >= 1:
                st["three_bet"] = True

    def on_board(self, street: Street, board: list[int]):
        if self._record is not None:
            self._record.board = list(board)
            self._record.boards[int(street)] = list(board)

    def on_showdown(self, info: ShowdownInfo):
        rec = self._record
        if rec is None:
            return
        rec.went_to_showdown = True
        rec.showdown_cards = dict(info.revealed_cards)
        rec.winners = list(info.winners)
        rec.board = list(info.board)
        self.showdowns_observed += 1
        if self._hero_folded:
            self.spectator_showdowns += 1

        for seat, cards in info.revealed_cards.items():
            if seat == self.hero_seat:
                continue
            model = self.models.get(seat)
            h_before = model.type_entropy()
            was_bluffing = self._replay_showdown(model, seat, cards, info.board)
            model.record_showdown(won=seat in info.winners, was_bluffing=was_bluffing)
            ig = h_before - model.type_entropy()
            self._log_information(ig, rec.hand_number, seat)

    def _board_at(self, street: int, final_board: list[int]) -> list[int]:
        n = {1: 3, 2: 4, 3: 5}.get(street, 0)
        return list(final_board[:n])

    def _replay_showdown(self, model: OpponentModel, seat: int,
                         cards: list[int], final_board: list[int]) -> Optional[bool]:
        """Walk the revealed player's postflop actions with ground truth.

        Returns whether their last aggressive action was a bluff (None if
        they never bet or raised postflop).
        """
        last_aggr_strength = None
        for a in self._record.actions:
            if a.player != seat or a.street == Street.PREFLOP:
                continue
            board = self._board_at(int(a.street), final_board)
            if len(board) < 3:
                continue
            s = hand_strength(cards, board)
            model.record_showdown_action(int(a.street), s, a.is_aggressive,
                                         a.to_call > 0,
                                         a.size_fraction if a.is_aggressive else None)
            if a.is_aggressive:
                last_aggr_strength = s
        if last_aggr_strength is None:
            return None
        return last_aggr_strength < BLUFF_STRENGTH

    def _log_information(self, ig: float, hand_number: int, seat: int):
        self.information_gain_total += max(0.0, ig)
        item = (ig, hand_number, seat)
        if len(self._top_observations) < self._keep:
            heapq.heappush(self._top_observations, item)
        elif ig > self._top_observations[0][0]:
            heapq.heapreplace(self._top_observations, item)

    def on_hand_end(self, info: HandEndInfo):
        rec = self._record
        if rec is None:
            return
        for seat, st in rec.preflop.items():
            if seat == self.hero_seat:
                continue
            self.models.get(seat).record_preflop_hand(
                st["position"], st["vpip"], st["pfr"], st["faced_raise"],
                st["three_bet"], st["first"])
        for seat in range(rec.num_players):
            if seat == self.hero_seat:
                continue
            if rec.stacks and seat < len(rec.stacks) and rec.stacks[seat] <= 0 \
                    and seat not in rec.preflop:
                continue        # sat out (busted)
            self.models.get(seat).finish_hand()
        self.models.on_hand_end()

        if self._hero_folded:
            self.spectator_hands += 1
        self.hands_processed += 1
        self._record = None

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def get_model(self, player_id: int) -> OpponentModel:
        return self.models.get(player_id)

    def get_fold_estimate(self, player_id: int, facing_raise: bool = False,
                          size_fraction: Optional[float] = None) -> float:
        model = self.models.get(player_id)
        if facing_raise:
            return model.estimated_fold_to_raise()
        return model.estimated_fold_to_bet(size_fraction)

    def get_opponent_type(self, player_id: int) -> str:
        return self.models.get(player_id).primary_type()

    def top_observations(self) -> list[tuple[float, int, int]]:
        """Most informative showdowns so far: (IG bits, hand number, seat)."""
        return sorted(self._top_observations, reverse=True)

    def suggest_strategy(self, player_id: int) -> str:
        """Behavioural mode label for an opponent (§21).

        The actual exploitation is quantitative (MDF loop, exploiter.py);
        this label is descriptive and used for reporting / hysteresis.
        """
        otype = self.models.get(player_id).primary_type()
        return {"passive": "value_heavy", "tight": "aggressive",
                "aggressive": "trap_heavy"}.get(otype, "balanced")

    def report(self) -> str:
        lines = [
            f"Spectator Learner: {self.hands_processed} hands processed, "
            f"{self.showdowns_observed} showdowns ({self.spectator_showdowns} after "
            f"hero folded), {self.spectator_hands} spectator hands, "
            f"IG={self.information_gain_total:.2f} bits",
            self.models.report(),
        ]
        return "\n".join(lines)

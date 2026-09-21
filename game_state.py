"""
Game State
==========
Complete representation of a Texas Hold'em hand in progress.
Handles:
  - pot & side-pot accounting
  - stack tracking
  - position labels
  - street progression (preflop → flop → turn → river → showdown)
  - betting-round logic (who acts next, min-raise, legal actions)
  - action history
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class Street(IntEnum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3
    SHOWDOWN = 4


class ActionType(IntEnum):
    FOLD = 0
    CHECK = 1
    CALL = 2
    BET = 3      # opening bet on a street
    RAISE = 4
    ALL_IN = 5


class Position(IntEnum):
    """Canonical seats for up to 9 players. For 2-player, use BTN and BB."""
    BTN = 0    # button / dealer (= SB in heads-up)
    SB = 1
    BB = 2
    UTG = 3
    UTG1 = 4
    MP = 5
    MP1 = 6
    CO = 7
    HJ = 8


# ---------------------------------------------------------------------------
# Action record
# ---------------------------------------------------------------------------
@dataclass
class Action:
    player: int
    action_type: ActionType
    amount: int = 0       # total chips committed this action (0 for fold/check)
    street: Street = Street.PREFLOP

    def __repr__(self):
        name = ActionType(self.action_type).name
        if self.amount:
            return f"P{self.player}:{name}({self.amount})"
        return f"P{self.player}:{name}"


# ---------------------------------------------------------------------------
# Legal-action descriptor
# ---------------------------------------------------------------------------
@dataclass
class LegalAction:
    action_type: ActionType
    min_amount: int = 0   # min raise/bet size (total, not on-top)
    max_amount: int = 0   # max raise/bet size (= stack for no-limit)


# ---------------------------------------------------------------------------
# GameState
# ---------------------------------------------------------------------------
@dataclass
class GameState:
    num_players: int
    starting_stacks: list[int]
    small_blind: int
    big_blind: int

    # --- mutable per-hand state ---
    stacks: list[int] = field(default_factory=list)
    hole_cards: list[Optional[list[int]]] = field(default_factory=list)
    board: list[int] = field(default_factory=list)
    pot: int = 0
    street: Street = Street.PREFLOP
    action_history: list[Action] = field(default_factory=list)

    # positions[i] = Position enum for seat i (assigned at hand start)
    positions: list[Position] = field(default_factory=list)
    dealer_seat: int = 0

    # --- betting-round tracking ---
    current_bets: list[int] = field(default_factory=list)   # chips bet this street per player
    folded: list[bool] = field(default_factory=list)
    all_in: list[bool] = field(default_factory=list)
    actor: int = 0            # seat index of next actor
    last_raiser: int = -1
    last_raise_size: int = 0  # the *increment* of the last raise
    num_actions_this_street: int = 0
    hand_over: bool = False
    # Set of players who still need to act this street
    needs_to_act: set = field(default_factory=set)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def new_hand(self, dealer_seat: int, hole_cards: list[list[int]]):
        """Reset state for a new hand. hole_cards[i] = [c1,c2] per player."""
        self.dealer_seat = dealer_seat
        self.hole_cards = [list(h) for h in hole_cards]
        self.stacks = list(self.starting_stacks)
        self.board = []
        self.pot = 0
        self.street = Street.PREFLOP
        self.action_history = []
        self.current_bets = [0] * self.num_players
        self.folded = [False] * self.num_players
        self.all_in = [False] * self.num_players
        self.hand_over = False
        self.last_raiser = -1
        self.last_raise_size = self.big_blind
        self.num_actions_this_street = 0

        self.needs_to_act = set()

        self._assign_positions()
        self._post_blinds()

    def _assign_positions(self):
        """Map seat indices to positional labels based on dealer."""
        n = self.num_players
        self.positions = [Position.BTN] * n
        if n == 2:
            self.positions[self.dealer_seat] = Position.BTN  # BTN = SB in HU
            self.positions[(self.dealer_seat + 1) % n] = Position.BB
        else:
            order = [Position.BTN, Position.SB, Position.BB,
                     Position.UTG, Position.UTG1, Position.MP,
                     Position.MP1, Position.CO, Position.HJ]
            for i in range(n):
                seat = (self.dealer_seat + i) % n
                self.positions[seat] = order[i] if i < len(order) else Position.MP

    def _post_blinds(self):
        n = self.num_players
        if n == 2:
            sb_seat = self.dealer_seat
            bb_seat = (self.dealer_seat + 1) % n
        else:
            sb_seat = (self.dealer_seat + 1) % n
            bb_seat = (self.dealer_seat + 2) % n

        self._force_bet(sb_seat, min(self.small_blind, self.stacks[sb_seat]))
        self._force_bet(bb_seat, min(self.big_blind, self.stacks[bb_seat]))

        # First to act preflop: left of BB (or SB in HU after posting)
        if n == 2:
            self.actor = self.dealer_seat  # BTN/SB acts first preflop in HU
        else:
            self.actor = (bb_seat + 1) % n

        # Everyone who can act needs to act at least once preflop
        # BB gets an option even if no one raises (the "big blind option")
        self.needs_to_act = set(self.players_can_act)

    def _force_bet(self, seat: int, amount: int):
        self.stacks[seat] -= amount
        self.current_bets[seat] = amount
        self.pot += amount
        if self.stacks[seat] == 0:
            self.all_in[seat] = True

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------
    @property
    def active_players(self) -> list[int]:
        """Players still in the hand (not folded)."""
        return [i for i in range(self.num_players) if not self.folded[i]]

    @property
    def players_can_act(self) -> list[int]:
        """Players who can still take actions (not folded, not all-in)."""
        return [i for i in range(self.num_players)
                if not self.folded[i] and not self.all_in[i]]

    @property
    def max_current_bet(self) -> int:
        return max(self.current_bets)

    @property
    def effective_stack(self) -> int:
        """Smallest stack among active players (useful for HU)."""
        active_stacks = [self.stacks[i] + self.current_bets[i]
                         for i in self.active_players]
        return min(active_stacks) if active_stacks else 0

    @property
    def spr(self) -> float:
        """Stack-to-pot ratio for the hero (actor)."""
        if self.pot == 0:
            return float('inf')
        return self.stacks[self.actor] / self.pot

    # ------------------------------------------------------------------
    # Legal actions
    # ------------------------------------------------------------------
    def get_legal_actions(self, player: Optional[int] = None) -> list[LegalAction]:
        """Return every legal action for the given player (default: current actor)."""
        if player is None:
            player = self.actor
        if self.hand_over or self.folded[player] or self.all_in[player]:
            return []

        actions: list[LegalAction] = []
        to_call = self.max_current_bet - self.current_bets[player]
        stack = self.stacks[player]

        # Fold — always legal if there's a bet to face
        if to_call > 0:
            actions.append(LegalAction(ActionType.FOLD))

        # Check — legal only when no bet to call
        if to_call == 0:
            actions.append(LegalAction(ActionType.CHECK))

        # Call
        if to_call > 0:
            call_amt = min(to_call, stack)
            actions.append(LegalAction(ActionType.CALL, call_amt, call_amt))

        # Bet / Raise (no-limit)
        if to_call == 0:
            # Opening bet: min is 1 BB
            min_bet = min(self.big_blind, stack)
            if stack > 0:
                actions.append(LegalAction(ActionType.BET, min_bet, stack))
        else:
            # Raise: min raise = last raise size on top of current bet to match
            min_raise_total = self.max_current_bet + max(self.last_raise_size, self.big_blind)
            min_raise_cost = min_raise_total - self.current_bets[player]
            if min_raise_cost > stack:
                # Can only all-in for less than a min-raise
                if stack > to_call:
                    actions.append(LegalAction(ActionType.ALL_IN, stack, stack))
            else:
                actions.append(LegalAction(
                    ActionType.RAISE,
                    min_amount=min_raise_total,
                    max_amount=self.current_bets[player] + stack,
                ))

        return actions

    # ------------------------------------------------------------------
    # Apply action
    # ------------------------------------------------------------------
    def apply_action(self, action: Action):
        """Apply an action to the game state. Advances actor and street."""
        p = action.player
        assert p == self.actor, f"Expected actor {self.actor}, got {p}"
        assert not self.folded[p] and not self.all_in[p]

        self.action_history.append(action)
        self.num_actions_this_street += 1

        if action.action_type == ActionType.FOLD:
            self.folded[p] = True

        elif action.action_type == ActionType.CHECK:
            pass

        elif action.action_type == ActionType.CALL:
            amount = min(self.max_current_bet - self.current_bets[p], self.stacks[p])
            self.stacks[p] -= amount
            self.pot += amount
            self.current_bets[p] += amount
            if self.stacks[p] == 0:
                self.all_in[p] = True

        elif action.action_type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN):
            total = action.amount  # total bet this street
            cost = total - self.current_bets[p]
            cost = min(cost, self.stacks[p])
            actual_total = self.current_bets[p] + cost
            raise_increment = actual_total - self.max_current_bet
            if raise_increment > 0:
                self.last_raise_size = raise_increment
                self.last_raiser = p
            self.stacks[p] -= cost
            self.pot += cost
            self.current_bets[p] = actual_total
            if self.stacks[p] == 0:
                self.all_in[p] = True

        # Remove current player from needs_to_act (they just acted)
        self.needs_to_act.discard(p)

        # A raise/bet re-opens action for everyone else still in
        if action.action_type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN):
            raise_increment = self.current_bets[p] - (self.max_current_bet
                              if action.action_type == ActionType.ALL_IN
                              else 0)
            # Everyone who can act (except the raiser) needs to respond
            for seat in self.players_can_act:
                if seat != p:
                    self.needs_to_act.add(seat)

        # Check if hand is over (only one active player left)
        if len(self.active_players) == 1:
            self.hand_over = True
            return

        # Advance to next actor
        self._advance_actor()

    def _advance_actor(self):
        """Move actor to the next player who needs to act, or end the street."""
        n = self.num_players

        # Find next player clockwise who still needs to act
        start = (self.actor + 1) % n
        for offset in range(n):
            seat = (start + offset) % n
            if seat in self.needs_to_act and not self.folded[seat] and not self.all_in[seat]:
                self.actor = seat
                return

        # No one left to act — end the street
        self._end_street()

    def _end_street(self):
        """Move to the next street or showdown."""
        # Sweep bets into pot (already done incrementally, just reset tracking)
        self.current_bets = [0] * self.num_players
        self.last_raiser = -1
        self.last_raise_size = self.big_blind
        self.num_actions_this_street = 0

        if self.street == Street.RIVER or len(self.players_can_act) <= 1:
            self.street = Street.SHOWDOWN
            self.hand_over = True
        else:
            self.street = Street(self.street + 1)
            # Everyone who can act needs to act on the new street
            self.needs_to_act = set(self.players_can_act)
            # Post-flop, first to act is first active player left of dealer
            n = self.num_players
            for offset in range(1, n + 1):
                seat = (self.dealer_seat + offset) % n
                if not self.folded[seat] and not self.all_in[seat]:
                    self.actor = seat
                    return
            # Everyone is all-in — run out the board
            self.street = Street.SHOWDOWN
            self.hand_over = True

    # ------------------------------------------------------------------
    # State snapshot (for decision engine)
    # ------------------------------------------------------------------
    def snapshot(self, hero: int) -> dict:
        """Return a dict of everything the hero can observe."""
        return {
            "hero": hero,
            "hero_cards": self.hole_cards[hero] if self.hole_cards else [],
            "board": list(self.board),
            "pot": self.pot,
            "street": self.street.name,
            "stacks": list(self.stacks),
            "position": self.positions[hero].name if self.positions else "?",
            "current_bets": list(self.current_bets),
            "folded": list(self.folded),
            "all_in": list(self.all_in),
            "spr": round(self.spr, 2) if self.pot > 0 else None,
            "action_history": [repr(a) for a in self.action_history],
            "legal_actions": [
                (ActionType(la.action_type).name, la.min_amount, la.max_amount)
                for la in self.get_legal_actions(hero)
            ],
        }

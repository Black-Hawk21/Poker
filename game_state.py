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
import copy
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
    amount: int = 0       # BET/RAISE/ALL_IN: total street commitment after the
                          # action.  CALL: chips added.  FOLD/CHECK: 0.
    street: Street = Street.PREFLOP

    # --- pre-action context, filled in by GameState.apply_action() ---
    # Observers receive the action *after* it has been applied, so anything
    # that needs "what was the player facing?" (range updates, fold-to-bet,
    # 3-bet detection, bet-size ratios) must read these fields rather than
    # the post-action GameState.
    to_call: int = 0          # chips needed to call before acting
    pot_before: int = 0       # pot before this action
    bet_before: int = 0       # highest street commitment before this action
    committed_before: int = 0 # this player's street commitment before acting
    raises_before: int = 0    # voluntary bets/raises this street before this action
    stack_before: int = 0     # player's remaining stack before acting

    @property
    def facing_bet(self) -> bool:
        return self.to_call > 0

    @property
    def is_aggressive(self) -> bool:
        return self.action_type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN) \
            and self.amount > self.bet_before

    @property
    def size_fraction(self) -> float:
        """Chips added beyond a call, as a fraction of the pot after calling.

        For an opening bet this is simply bet / pot.  For a raise it is the
        raise increment over the pot the raiser would face after calling —
        the same convention the decision engine uses for hero's own sizes.
        """
        if not self.is_aggressive:
            return 0.0
        added = self.amount - self.bet_before
        denom = self.pot_before + self.to_call
        return added / denom if denom > 0 else 1.0

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
    # Players who acted since the last *full* bet/raise.  If an all-in raise
    # is smaller than a full raise, these players may only call or fold
    # (standard no-limit rule: an incomplete raise does not reopen betting).
    acted_since_full_raise: set = field(default_factory=set)
    cannot_reraise: set = field(default_factory=set)
    # Voluntary bets/raises made on the current street.
    raises_this_street: int = 0

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
        self.raises_this_street = 0
        self.acted_since_full_raise = set()
        self.cannot_reraise = set()

        self.needs_to_act = set()

        # Seats with no chips (busted in tournament play) sit the hand out.
        for i in range(self.num_players):
            if self.stacks[i] <= 0:
                self.folded[i] = True

        self._assign_positions()
        self._post_blinds()

    def _live_seats_from_dealer(self) -> list[int]:
        """Seats dealt into the hand, starting at the button and going clockwise."""
        n = self.num_players
        seats = [(self.dealer_seat + i) % n for i in range(n)]
        return [s for s in seats if not self.folded[s]]

    def _assign_positions(self):
        """Map seat indices to positional labels based on dealer."""
        n = self.num_players
        self.positions = [Position.MP] * n
        live = self._live_seats_from_dealer()
        if len(live) == 2:
            self.positions[live[0]] = Position.BTN  # BTN = SB in HU
            self.positions[live[1]] = Position.BB
        else:
            order = [Position.BTN, Position.SB, Position.BB,
                     Position.UTG, Position.UTG1, Position.MP,
                     Position.MP1, Position.CO, Position.HJ]
            for i, seat in enumerate(live):
                self.positions[seat] = order[i] if i < len(order) else Position.MP

    def _post_blinds(self):
        live = self._live_seats_from_dealer()
        if len(live) < 2:
            self.hand_over = True
            return
        if len(live) == 2:
            sb_seat, bb_seat = live[0], live[1]
            first = sb_seat               # BTN/SB acts first preflop in HU
        else:
            sb_seat, bb_seat = live[1], live[2]
            first = live[3 % len(live)]   # left of the big blind

        self._force_bet(sb_seat, min(self.small_blind, self.stacks[sb_seat]))
        self._force_bet(bb_seat, min(self.big_blind, self.stacks[bb_seat]))
        self.actor = first

        # Everyone who can act needs to act at least once preflop
        # BB gets an option even if no one raises (the "big blind option")
        self.needs_to_act = set(self.players_can_act)
        if self.all_in[self.actor] or self.folded[self.actor]:
            self._advance_actor()

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
        """Effective remaining stack for the current actor."""
        return self.effective_stack_for(self.actor)

    def effective_stack_for(self, seat: int) -> int:
        """Chips `seat` can actually win or lose from here on.

        min(own remaining stack, largest remaining stack among the other
        players still in the hand).  A deep stack facing a short stack is
        only as deep as the short stack.
        """
        others = [self.stacks[i] for i in self.active_players if i != seat]
        if not others:
            return self.stacks[seat]
        return min(self.stacks[seat], max(others))

    @property
    def spr(self) -> float:
        """Stack-to-pot ratio (Eq. 19) for the actor: effective stack / pot."""
        return self.spr_for(self.actor)

    def spr_for(self, seat: int) -> float:
        if self.pot == 0:
            return float('inf')
        return self.effective_stack_for(seat) / self.pot

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
        elif player in self.cannot_reraise or not any(
                s != player and not self.all_in[s] for s in self.active_players):
            # Facing an incomplete all-in raise after already acting, or
            # everyone else is all-in: calling/folding only.
            pass
        else:
            # Raise: min raise = last raise size on top of current bet to match
            min_raise_total = self.max_current_bet + max(self.last_raise_size, self.big_blind)
            min_raise_cost = min_raise_total - self.current_bets[player]
            if min_raise_cost > stack:
                # Can only all-in for less than a min-raise
                if stack > to_call:
                    # Amounts are street totals, like BET/RAISE.  (Previously
                    # this was the bare stack, so a player who had already
                    # put chips in this street was not actually all-in.)
                    total = self.current_bets[player] + stack
                    actions.append(LegalAction(ActionType.ALL_IN, total, total))
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

        # Record what the player was facing (observers see post-action state).
        action.street = self.street
        action.to_call = self.max_current_bet - self.current_bets[p]
        action.pot_before = self.pot
        action.bet_before = self.max_current_bet
        action.committed_before = self.current_bets[p]
        action.raises_before = self.raises_this_street
        action.stack_before = self.stacks[p]

        self.action_history.append(action)
        self.num_actions_this_street += 1
        full_raise = False
        reopened = False

        if action.action_type == ActionType.FOLD:
            self.folded[p] = True

        elif action.action_type == ActionType.CHECK:
            pass

        elif action.action_type == ActionType.CALL:
            amount = min(self.max_current_bet - self.current_bets[p], self.stacks[p])
            self.stacks[p] -= amount
            self.pot += amount
            self.current_bets[p] += amount
            action.amount = amount
            if self.stacks[p] == 0:
                self.all_in[p] = True

        elif action.action_type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN):
            total = action.amount  # total bet this street
            cost = total - self.current_bets[p]
            cost = max(0, min(cost, self.stacks[p]))
            actual_total = self.current_bets[p] + cost
            raise_increment = actual_total - self.max_current_bet
            if raise_increment > 0:
                reopened = True
                self.raises_this_street += 1
                self.last_raiser = p
                # Only a complete raise resets the minimum-raise increment
                # and re-opens betting for players who already acted.
                if raise_increment >= self.last_raise_size or self.max_current_bet == 0:
                    full_raise = True
                    self.last_raise_size = max(raise_increment, self.big_blind)
            self.stacks[p] -= cost
            self.pot += cost
            self.current_bets[p] = actual_total
            action.amount = actual_total
            if self.stacks[p] == 0:
                self.all_in[p] = True

        # Remove current player from needs_to_act (they just acted)
        self.needs_to_act.discard(p)

        if reopened:
            if full_raise:
                self.cannot_reraise = set()
                self.acted_since_full_raise = set()
            else:
                # Incomplete raise: those who already acted may only call/fold.
                self.cannot_reraise |= (self.acted_since_full_raise - {p})
            # Everyone who can act (except the raiser) needs to respond
            for seat in self.players_can_act:
                if seat != p:
                    self.needs_to_act.add(seat)
        self.acted_since_full_raise.add(p)

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
        self.raises_this_street = 0
        self.acted_since_full_raise = set()
        self.cannot_reraise = set()

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
    # Per-seat view (what a bot is allowed to see)
    # ------------------------------------------------------------------
    def view_for(self, seat: Optional[int]) -> "GameState":
        """Copy of the state with every other player's hole cards hidden.

        The arena hands bots this view rather than the live object, so a bot
        can neither read opponents' cards nor mutate the real game state
        (design doc §23.1: reading other bots' hole cards off the server is
        out of scope).  seat=None hides every hand (pure spectator view).
        """
        v = GameState(
            num_players=self.num_players,
            starting_stacks=list(self.starting_stacks),
            small_blind=self.small_blind,
            big_blind=self.big_blind,
        )
        v.stacks = list(self.stacks)
        v.hole_cards = [
            (list(h) if (i == seat and h is not None) else None)
            for i, h in enumerate(self.hole_cards)
        ]
        v.board = list(self.board)
        v.pot = self.pot
        v.street = self.street
        v.action_history = [copy.copy(a) for a in self.action_history]
        v.positions = list(self.positions)
        v.dealer_seat = self.dealer_seat
        v.current_bets = list(self.current_bets)
        v.folded = list(self.folded)
        v.all_in = list(self.all_in)
        v.actor = self.actor
        v.last_raiser = self.last_raiser
        v.last_raise_size = self.last_raise_size
        v.num_actions_this_street = self.num_actions_this_street
        v.hand_over = self.hand_over
        v.needs_to_act = set(self.needs_to_act)
        v.acted_since_full_raise = set(self.acted_since_full_raise)
        v.cannot_reraise = set(self.cannot_reraise)
        v.raises_this_street = self.raises_this_street
        return v

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
            "spr": round(self.spr_for(hero), 2) if self.pot > 0 else None,
            "action_history": [repr(a) for a in self.action_history],
            "legal_actions": [
                (ActionType(la.action_type).name, la.min_amount, la.max_amount)
                for la in self.get_legal_actions(hero)
            ],
        }

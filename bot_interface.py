"""
Bot Interface
=============
Abstract base class every poker bot must implement.

The environment calls:
  1. observe_hand_start()  — new hand begins, receive seat + hole cards
  2. act()                 — your turn, return an Action
  3. observe_action()      — someone (including you) acted
  4. observe_board()       — new community cards dealt
  5. observe_showdown()    — hand ended with a showdown; see revealed cards
  6. observe_hand_end()    — hand complete; see final result

This protocol supports spectator learning (Section 13): the bot receives
showdown data for every hand, even ones it folded from.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from game_state import GameState, Action, ActionType, LegalAction, Street


@dataclass
class HandStartInfo:
    """Information delivered at the start of each hand."""
    hand_number: int
    seat: int                 # your seat index
    hole_cards: list[int]     # your two cards
    num_players: int
    stacks: list[int]         # starting stacks for this hand
    dealer_seat: int
    small_blind: int
    big_blind: int
    positions: list[str]      # position label per seat


@dataclass
class ShowdownInfo:
    """Information delivered at showdown (Section 13 — spectator learning)."""
    revealed_cards: dict[int, list[int]]   # seat → [card1, card2] for all shown hands
    board: list[int]
    pot: int
    winners: list[int]                     # seat indices of winner(s)
    amounts_won: dict[int, int]            # seat → chips won


@dataclass
class HandEndInfo:
    """Summary delivered at the end of every hand."""
    hand_number: int
    final_stacks: list[int]
    net_won: dict[int, int]       # seat → chips gained/lost this hand
    went_to_showdown: bool
    board: list[int]
    action_history: list[Action]
    showdown: Optional[ShowdownInfo]


class BaseBot(ABC):
    """
    Abstract poker bot.  Subclass this and implement the methods below.
    """

    def __init__(self, name: str = "Bot"):
        self.name = name
        self.seat: int = -1

    @abstractmethod
    def act(self, game_state: GameState) -> Action:
        """
        Called when it is your turn.

        Parameters
        ----------
        game_state : the full GameState with your hole cards visible.
                     Call game_state.get_legal_actions(self.seat) for
                     what you can do.

        Returns
        -------
        An Action with player=self.seat.
        """
        ...

    # ------------------------------------------------------------------
    # Observation hooks (override any you care about)
    # ------------------------------------------------------------------
    def observe_hand_start(self, info: HandStartInfo):
        """New hand is starting. Store your seat and cards."""
        self.seat = info.seat

    def observe_action(self, action: Action, game_state: GameState):
        """Someone (possibly you) just acted."""
        pass

    def observe_board(self, street: Street, board: list[int]):
        """New community cards were dealt (flop/turn/river)."""
        pass

    def observe_showdown(self, info: ShowdownInfo):
        """Hand reached showdown — all revealed cards are visible.
        This is the key hook for spectator learning (Section 13):
        update opponent models even for hands you folded."""
        pass

    def observe_hand_end(self, info: HandEndInfo):
        """Hand is complete. Update any bookkeeping."""
        pass

    def __repr__(self):
        return f"{self.name}(seat={self.seat})"

"""
Game Runner (Arena)
===================
The central environment that orchestrates poker hands between bots.

Features:
  - Deals cards, manages the GameState, calls bots in turn
  - Supports 2–9 players per table
  - Rotates dealer each hand
  - Full spectator-learning protocol: every bot sees showdown data
  - Determines winners at showdown, with side pots and split pots
  - Bots receive a per-seat view: opponents' hole cards hidden
  - Tournament mode (reset_stacks_each_hand=False) carries stacks over
  - Tracks VPIP/PFR per hand for stats
  - Configurable hand count, blind levels, starting stacks
  - Verbose or silent mode
  - Safety: catches bot errors and forces a fold

Usage:
    from game_runner import GameRunner
    from opponents import HeroBot, ManiacBot, NitBot

    runner = GameRunner(
        bots=[HeroBot("Hero"), ManiacBot("Maniac")],
        starting_stack=1000,
        small_blind=5,
        big_blind=10,
    )
    stats = runner.run(num_hands=500)
    print(stats.report())
"""

from __future__ import annotations
import copy
import sys
from typing import Optional

from deck import Deck
from game_state import GameState, Action, ActionType, Street
from hand_evaluator import evaluate, card_str
from bot_interface import (
    BaseBot, HandStartInfo, ShowdownInfo, HandEndInfo
)
from stats import StatsTracker


class GameRunner:
    """
    Runs a series of poker hands between a list of bots.
    """

    def __init__(
        self,
        bots: list[BaseBot],
        starting_stack: int = 1000,
        small_blind: int = 5,
        big_blind: int = 10,
        deck_seed: Optional[int] = None,
        verbose: bool = False,
    ):
        assert len(bots) >= 2, "Need at least 2 bots"
        assert len(bots) <= 9, "Max 9 bots per table"

        self.bots = bots
        self.num_players = len(bots)
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.verbose = verbose

        self._deck = Deck(seed=deck_seed)
        self._gs = GameState(
            num_players=self.num_players,
            starting_stacks=[starting_stack] * self.num_players,
            small_blind=small_blind,
            big_blind=big_blind,
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(
        self,
        num_hands: int = 100,
        reset_stacks_each_hand: bool = True,
    ) -> StatsTracker:
        """
        Play num_hands hands and return collected stats.

        Parameters
        ----------
        num_hands : how many hands to play
        reset_stacks_each_hand : if True, all stacks reset to starting_stack
                                 each hand (ring-game style). If False,
                                 stacks carry over (tournament style).
        """
        tracker = StatsTracker(big_blind=self.big_blind)
        dealer = 0
        self._gs.starting_stacks = [self.starting_stack] * self.num_players

        for hand_num in range(1, num_hands + 1):
            if reset_stacks_each_hand:
                self._gs.starting_stacks = [self.starting_stack] * self.num_players
            live = [i for i in range(self.num_players) if self._gs.starting_stacks[i] > 0]
            if len(live) < 2:
                break                       # tournament over
            if self._gs.starting_stacks[dealer] <= 0:
                dealer = self._next_live(dealer)

            final = self._play_hand(hand_num, dealer, tracker)
            if not reset_stacks_each_hand:
                # Tournament style: stacks carry over to the next hand.
                self._gs.starting_stacks = list(final)
            dealer = self._next_live(dealer)

            if self.verbose and hand_num % 100 == 0:
                print(f"  ... {hand_num}/{num_hands} hands", file=sys.stderr)

        return tracker

    def _next_live(self, seat: int) -> int:
        stacks = self._gs.starting_stacks
        for k in range(1, self.num_players + 1):
            s = (seat + k) % self.num_players
            if stacks[s] > 0:
                return s
        return (seat + 1) % self.num_players

    # ------------------------------------------------------------------
    # Single hand
    # ------------------------------------------------------------------
    def _play_hand(self, hand_num: int, dealer: int, tracker: StatsTracker):
        """Play one complete hand."""
        gs = self._gs

        # --- Deal hole cards ---
        self._deck.reset()
        hole_cards = [self._deck.deal(2) for _ in range(self.num_players)]

        # --- Initialize hand ---
        gs.new_hand(dealer_seat=dealer, hole_cards=hole_cards)

        if self.verbose:
            names = [b.name for b in self.bots]
            print(f"\n{'─'*50}")
            print(f"Hand #{hand_num}  Dealer=P{dealer}({names[dealer]})")
            for i, bot in enumerate(self.bots):
                cards = " ".join(card_str(c) for c in hole_cards[i])
                print(f"  P{i}({bot.name}): [{cards}]  "
                      f"pos={gs.positions[i].name}  stack={gs.stacks[i]}")

        # --- Notify bots of hand start ---
        for i, bot in enumerate(self.bots):
            bot.observe_hand_start(HandStartInfo(
                hand_number=hand_num,
                seat=i,
                hole_cards=list(hole_cards[i]),
                num_players=self.num_players,
                stacks=list(gs.starting_stacks),
                dealer_seat=dealer,
                small_blind=self.small_blind,
                big_blind=self.big_blind,
                positions=[gs.positions[j].name for j in range(self.num_players)],
            ))

        # --- Track VPIP / PFR ---
        vpip_seats: set[int] = set()
        pfr_seats: set[int] = set()

        # Stacks before the hand (before blinds)
        stacks_before = list(gs.starting_stacks)

        # --- Betting rounds ---
        boards_dealt = 0
        max_actions = 200  # safety
        action_count = 0

        while not gs.hand_over and action_count < max_actions:
            actor = gs.actor
            bot = self.bots[actor]

            # Get action from bot (with error handling)
            try:
                # Each bot sees only its own hole cards, via a copy it
                # cannot use to mutate the real game (design doc §23.1).
                action = bot.act(gs.view_for(actor))
                action = copy.copy(action)
                action = self._validate_action(action, gs, actor)
            except Exception as e:
                if self.verbose:
                    print(f"  !! {bot.name} error: {e} → forced fold")
                action = self._force_fold_or_check(gs, actor)

            # Track VPIP / PFR (preflop voluntary actions)
            if gs.street == Street.PREFLOP:
                if action.action_type in (ActionType.CALL, ActionType.RAISE,
                                          ActionType.BET, ActionType.ALL_IN):
                    vpip_seats.add(actor)
                if action.action_type in (ActionType.RAISE, ActionType.BET,
                                          ActionType.ALL_IN):
                    pfr_seats.add(actor)

            if self.verbose:
                self._log_action(action, bot)

            old_street = gs.street
            gs.apply_action(action)
            action_count += 1

            # Notify all bots of the action (each gets its own copies)
            for seat, b in enumerate(self.bots):
                b.observe_action(copy.copy(action), gs.view_for(seat))

            # Deal board cards when street advances
            if not gs.hand_over and gs.street != old_street:
                self._deal_board(gs, boards_dealt)
                boards_dealt = int(gs.street)  # FLOP=1, TURN=2, RIVER=3

                # Notify bots of new board cards
                for b in self.bots:
                    b.observe_board(gs.street, list(gs.board))

        # --- Run out the board if players are all-in (same burn protocol) ---
        if len(gs.active_players) > 1:
            while len(gs.board) < 5:
                nxt = {0: Street.FLOP, 3: Street.TURN, 4: Street.RIVER}[len(gs.board)]
                self._deal_street(gs, nxt)
                for b in self.bots:
                    b.observe_board(nxt, list(gs.board))

        # --- Determine winners (with side pots) ---
        winners, amounts = self._resolve_winners(gs)

        # --- Net result per seat: final stack − stack before the hand ---
        net_won = {}
        for i in range(self.num_players):
            final = gs.stacks[i] + amounts.get(i, 0)
            net_won[i] = final - stacks_before[i]

        went_to_showdown = len(gs.active_players) > 1
        showdown_seats = set(gs.active_players) if went_to_showdown else set()

        # --- Showdown info ---
        showdown_info = None
        if went_to_showdown:
            revealed = {}
            for seat in gs.active_players:
                if gs.hole_cards[seat]:
                    revealed[seat] = list(gs.hole_cards[seat])
            showdown_info = ShowdownInfo(
                revealed_cards=revealed,
                board=list(gs.board),
                pot=gs.pot,
                winners=winners,
                amounts_won=amounts,
            )
            # Notify ALL bots (spectator learning)
            for b in self.bots:
                b.observe_showdown(showdown_info)

        # --- Hand end notification ---
        end_info = HandEndInfo(
            hand_number=hand_num,
            final_stacks=[gs.stacks[i] + amounts.get(i, 0)
                          for i in range(self.num_players)],
            net_won=net_won,
            went_to_showdown=went_to_showdown,
            board=list(gs.board),
            action_history=list(gs.action_history),
            showdown=showdown_info,
        )
        for b in self.bots:
            b.observe_hand_end(end_info)

        if self.verbose:
            self._log_result(winners, amounts, net_won, went_to_showdown, gs)

        # --- Record stats ---
        tracker.record_hand(
            player_names=[b.name for b in self.bots],
            net_won=net_won,
            went_to_showdown=went_to_showdown,
            winners=winners,
            pot=gs.pot,
            vpip_seats=vpip_seats,
            pfr_seats=pfr_seats,
            showdown_seats=showdown_seats,
            dealt_seats={i for i in range(self.num_players) if stacks_before[i] > 0},
        )
        return [gs.stacks[i] + amounts.get(i, 0) for i in range(self.num_players)]

    # ------------------------------------------------------------------
    # Board dealing
    # ------------------------------------------------------------------
    def _deal_board(self, gs: GameState, boards_dealt: int):
        """Deal community cards for the street the game just advanced to."""
        target = {Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5}.get(gs.street, 0)
        while len(gs.board) < target:
            nxt = {0: Street.FLOP, 3: Street.TURN, 4: Street.RIVER}[len(gs.board)]
            self._deal_street(gs, nxt)
        if self.verbose and gs.board:
            print(f"  Board: [{' '.join(card_str(c) for c in gs.board)}]")

    def _deal_street(self, gs: GameState, street: Street):
        """One burn card, then the street's cards.

        Every path (normal play and the all-in run-out) uses this, so the
        deck order is identical regardless of how the hand ends.  The old
        run-out dealt burn+1 card three times for a flop, a different card
        order from normal play.
        """
        self._deck.deal_one()  # burn
        if street == Street.FLOP:
            gs.board = self._deck.deal(3)
        else:
            gs.board.append(self._deck.deal_one())

    # ------------------------------------------------------------------
    # Winner resolution with side pots
    # ------------------------------------------------------------------
    def _resolve_winners(self, gs: GameState) -> tuple[list[int], dict[int, int]]:
        """
        Distribute the pot, layer by layer (main pot + side pots).

        contribution[i] = chips player i put in this hand.  Each distinct
        contribution level of a live player defines a layer; a layer is
        contested only by live players who contributed at least that much.
        Uncalled excess returns to its owner as a layer they alone contest.
        Odd chips go to the first eligible winner left of the dealer.
        """
        n = gs.num_players
        active = gs.active_players
        contrib = [gs.starting_stacks[i] - gs.stacks[i] for i in range(n)]

        if len(active) == 0:
            return [], {}
        if len(active) == 1:
            return [active[0]], {active[0]: gs.pot}

        ranks = {s: evaluate(gs.hole_cards[s] + gs.board) for s in active}
        order = [(gs.dealer_seat + k) % n for k in range(1, n + 1)]
        amounts: dict[int, int] = {}
        levels = sorted({contrib[s] for s in active if contrib[s] > 0})
        prev = 0
        distributed = 0
        last_winners: list[int] = []
        for level in levels:
            layer = sum(min(contrib[i], level) - min(contrib[i], prev) for i in range(n))
            eligible = [s for s in active if contrib[s] >= level]
            if layer <= 0 or not eligible:
                prev = level
                continue
            best = max(ranks[s] for s in eligible)
            win = [s for s in order if s in eligible and ranks[s] == best]
            share, rem = divmod(layer, len(win))
            for w in win:
                amounts[w] = amounts.get(w, 0) + share
            amounts[win[0]] += rem
            distributed += layer
            last_winners = win
            prev = level
        leftover = gs.pot - distributed
        if leftover > 0 and last_winners:
            amounts[last_winners[0]] = amounts.get(last_winners[0], 0) + leftover
        winners = [s for s in order if amounts.get(s, 0) > 0]
        return winners, amounts

    # ------------------------------------------------------------------
    # Action validation / safety
    # ------------------------------------------------------------------
    def _validate_action(self, action: Action, gs: GameState, seat: int) -> Action:
        """Ensure the bot's action is legal. Fix or force fold if not."""
        action.player = seat
        action.street = gs.street

        legals = gs.get_legal_actions(seat)
        legal_types = {la.action_type for la in legals}

        if action.action_type not in legal_types:
            return self._force_fold_or_check(gs, seat)

        # Clamp amount to legal range
        for la in legals:
            if la.action_type == action.action_type:
                if la.max_amount > 0:
                    action.amount = max(la.min_amount,
                                        min(int(action.amount), la.max_amount))
                else:
                    action.amount = la.min_amount
                break

        return action

    @staticmethod
    def _force_fold_or_check(gs: GameState, seat: int) -> Action:
        """Fallback: check if possible, otherwise fold."""
        legals = gs.get_legal_actions(seat)
        for la in legals:
            if la.action_type == ActionType.CHECK:
                return Action(seat, ActionType.CHECK, 0, gs.street)
        return Action(seat, ActionType.FOLD, 0, gs.street)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _log_action(self, action: Action, bot: BaseBot):
        print(f"    {bot.name}: {action}")

    def _log_result(self, winners, amounts, net_won, showdown, gs):
        if showdown:
            print(f"  Showdown!")
            for seat in gs.active_players:
                cards = " ".join(card_str(c) for c in gs.hole_cards[seat])
                hr = evaluate(gs.hole_cards[seat] + gs.board)
                from hand_evaluator import hand_description
                print(f"    P{seat}({self.bots[seat].name}): [{cards}] "
                      f"→ {hand_description(hr)}")
        w_names = [self.bots[w].name for w in winners]
        print(f"  Winners: {w_names}  Pot: {gs.pot}")
        for i in range(self.num_players):
            print(f"    P{i}({self.bots[i].name}): net={net_won.get(i, 0):+d}")


# ===================================================================
# Convenience: round-robin tournament
# ===================================================================
def round_robin(
    bots: list[BaseBot],
    hands_per_matchup: int = 200,
    starting_stack: int = 1000,
    small_blind: int = 5,
    big_blind: int = 10,
    deck_seed: Optional[int] = None,
    verbose: bool = False,
) -> StatsTracker:
    """
    Run a round-robin: every pair of bots plays a heads-up match.
    Results are aggregated into one StatsTracker.
    """
    from itertools import combinations

    combined_tracker = StatsTracker(big_blind=big_blind)
    pairs = list(combinations(range(len(bots)), 2))

    for idx, (i, j) in enumerate(pairs):
        b1, b2 = bots[i], bots[j]
        seed = (deck_seed or 0) + idx * 10000

        runner = GameRunner(
            bots=[b1, b2],
            starting_stack=starting_stack,
            small_blind=small_blind,
            big_blind=big_blind,
            deck_seed=seed,
            verbose=verbose,
        )
        local = runner.run(num_hands=hands_per_matchup)

        # Merge into combined tracker
        for name, ps in local.players.items():
            cp = combined_tracker._get(name)
            cp.hands_played += ps.hands_played
            cp.hands_won += ps.hands_won
            cp.hands_lost += ps.hands_lost
            cp.total_won += ps.total_won
            cp.total_lost += ps.total_lost
            cp.net_chips += ps.net_chips
            cp.showdowns += ps.showdowns
            cp.showdown_wins += ps.showdown_wins
            cp.non_showdown_wins += ps.non_showdown_wins
            cp.non_showdown_chips += ps.non_showdown_chips
            cp.vpip_hands += ps.vpip_hands
            cp.pfr_hands += ps.pfr_hands
            cp.pots_won_total += ps.pots_won_total
            cp.pots_lost_total += ps.pots_lost_total
            cp.hand_results.extend(ps.hand_results)

        combined_tracker.total_hands += local.total_hands
        for k, v in local.matchup_net.items():
            combined_tracker.matchup_net[k] += v

    return combined_tracker

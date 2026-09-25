"""
Stats Tracker
=============
Collects per-player and per-matchup statistics for evaluation.

Tracks the metrics from Section 24:
  - chip EV (bb/hand)
  - win rate
  - showdown win rate
  - non-showdown winnings
  - bluff success rate (when observable)
  - average pot won / lost
  - per-opponent breakdown

Provides formatted reports and raw data export.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional
import json


@dataclass
class PlayerStats:
    """Accumulated stats for one player."""
    name: str = ""
    hands_played: int = 0
    hands_won: int = 0
    hands_lost: int = 0

    # Chip tracking
    total_won: int = 0       # total chips gained across all hands
    total_lost: int = 0      # total chips lost across all hands
    net_chips: int = 0       # cumulative profit/loss

    # Showdown stats
    showdowns: int = 0
    showdown_wins: int = 0

    # Non-showdown (won by fold)
    non_showdown_wins: int = 0
    non_showdown_chips: int = 0

    # Voluntary play
    vpip_hands: int = 0       # hands where player voluntarily put money in
    pfr_hands: int = 0        # hands where player raised preflop

    # Pot sizes
    pots_won_total: int = 0
    pots_lost_total: int = 0

    # Per-hand net tracking for variance
    hand_results: list[int] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.hands_won / self.hands_played if self.hands_played else 0.0

    @property
    def bb_per_hand(self) -> float:
        """Needs big blind size to be set externally."""
        return 0.0  # computed by StatsTracker with bb context

    @property
    def showdown_win_rate(self) -> float:
        return self.showdown_wins / self.showdowns if self.showdowns else 0.0

    @property
    def vpip(self) -> float:
        return self.vpip_hands / self.hands_played if self.hands_played else 0.0

    @property
    def pfr(self) -> float:
        return self.pfr_hands / self.hands_played if self.hands_played else 0.0

    @property
    def avg_pot_won(self) -> float:
        return self.pots_won_total / self.hands_won if self.hands_won else 0.0

    @property
    def avg_pot_lost(self) -> float:
        return self.pots_lost_total / self.hands_lost if self.hands_lost else 0.0

    @property
    def std_dev(self) -> float:
        if len(self.hand_results) < 2:
            return 0.0
        mean = sum(self.hand_results) / len(self.hand_results)
        var = sum((x - mean) ** 2 for x in self.hand_results) / (len(self.hand_results) - 1)
        return var ** 0.5


class StatsTracker:
    """
    Central stats collector for tournament / session runs.
    """

    def __init__(self, big_blind: int = 10):
        self.big_blind = big_blind
        self.players: dict[str, PlayerStats] = {}
        self.matchup_net: dict[tuple[str, str], int] = defaultdict(int)
        self.total_hands: int = 0
        self._net_history: dict[str, list[int]] = defaultdict(list)

    def _get(self, name: str) -> PlayerStats:
        if name not in self.players:
            self.players[name] = PlayerStats(name=name)
        return self.players[name]

    # ------------------------------------------------------------------
    # Record a completed hand
    # ------------------------------------------------------------------
    def record_hand(
        self,
        player_names: list[str],
        net_won: dict[int, int],      # seat → net chips
        went_to_showdown: bool,
        winners: list[int],           # seat indices
        pot: int,
        vpip_seats: set[int],         # seats that voluntarily entered
        pfr_seats: set[int],          # seats that raised preflop
        showdown_seats: Optional[set[int]] = None,  # seats that reached showdown
        dealt_seats: Optional[set[int]] = None,     # seats dealt in (not busted)
    ):
        """Record one completed hand's results.

        showdown_seats should list every player still in at showdown.  (The
        old code guessed with "net < 0 and not a winner ⇒ folded", which
        classified every showdown *loser* as having folded, so showdown
        win rates were ~100%.)  If omitted, it is approximated (exact heads-up).
        """
        self.total_hands += 1
        if showdown_seats is None:
            # Legacy callers: approximate as "everyone who won or lost chips"
            # (exact heads-up; the runner always passes showdown_seats).
            showdown_seats = ({s for s in range(len(player_names))
                               if s in winners or net_won.get(s, 0) != 0}
                              if went_to_showdown else set())

        for seat, name in enumerate(player_names):
            if dealt_seats is not None and seat not in dealt_seats:
                continue
            ps = self._get(name)
            ps.hands_played += 1
            net = net_won.get(seat, 0)
            ps.net_chips += net
            ps.hand_results.append(net)

            if net > 0:
                ps.hands_won += 1
                ps.total_won += net
                ps.pots_won_total += pot
            elif net < 0:
                ps.hands_lost += 1
                ps.total_lost += abs(net)
                ps.pots_lost_total += pot

            if seat in showdown_seats:
                ps.showdowns += 1
                if seat in winners:
                    ps.showdown_wins += 1

            if seat in winners and seat not in showdown_seats:
                ps.non_showdown_wins += 1
                ps.non_showdown_chips += net

            if seat in vpip_seats:
                ps.vpip_hands += 1
            if seat in pfr_seats:
                ps.pfr_hands += 1

        # Matchup tracking (pairwise net)
        for i, name_i in enumerate(player_names):
            for j, name_j in enumerate(player_names):
                if i < j:
                    net_i = net_won.get(i, 0)
                    self.matchup_net[(name_i, name_j)] += net_i

        # Running net history
        for seat, name in enumerate(player_names):
            running = (self._net_history[name][-1] if self._net_history[name] else 0)
            self._net_history[name].append(running + net_won.get(seat, 0))

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------
    def report(self) -> str:
        """Generate a formatted stats report."""
        lines = []
        lines.append(f"{'═' * 70}")
        lines.append(f"  POKER SESSION REPORT — {self.total_hands} hands")
        lines.append(f"{'═' * 70}")

        # Sort by net chips
        sorted_players = sorted(
            self.players.values(),
            key=lambda p: p.net_chips,
            reverse=True,
        )

        # Leaderboard
        lines.append(f"\n  {'Player':<20} {'Net':>8} {'bb/hand':>8} {'Win%':>7} "
                     f"{'SD Win%':>8} {'VPIP':>6} {'PFR':>6} {'StdDev':>8}")
        lines.append(f"  {'─' * 68}")

        for ps in sorted_players:
            bb_h = ps.net_chips / self.big_blind / ps.hands_played if ps.hands_played else 0
            lines.append(
                f"  {ps.name:<20} {ps.net_chips:>+8} {bb_h:>+8.2f} "
                f"{ps.win_rate:>6.1%} {ps.showdown_win_rate:>7.1%} "
                f"{ps.vpip:>6.1%} {ps.pfr:>5.1%} {ps.std_dev:>8.1f}"
            )

        # Matchup matrix
        if self.matchup_net:
            lines.append(f"\n  Head-to-Head (net chips, row perspective):")
            names = [p.name for p in sorted_players]
            header = f"  {'':>15}" + "".join(f"{n:>12}" for n in names)
            lines.append(header)
            for n1 in names:
                row = f"  {n1:>15}"
                for n2 in names:
                    if n1 == n2:
                        row += f"{'—':>12}"
                    else:
                        key = (n1, n2) if (n1, n2) in self.matchup_net else (n2, n1)
                        val = self.matchup_net.get(key, 0)
                        if key[0] != n1:
                            val = -val
                        row += f"{val:>+12}"
                lines.append(row)

        lines.append(f"\n{'═' * 70}")
        return "\n".join(lines)

    def summary_dict(self) -> dict:
        """Export stats as a dictionary (for JSON serialization)."""
        return {
            "total_hands": self.total_hands,
            "big_blind": self.big_blind,
            "players": {
                name: {
                    "net_chips": ps.net_chips,
                    "bb_per_hand": round(ps.net_chips / self.big_blind / ps.hands_played, 3)
                                  if ps.hands_played else 0,
                    "hands": ps.hands_played,
                    "win_rate": round(ps.win_rate, 3),
                    "showdown_win_rate": round(ps.showdown_win_rate, 3),
                    "vpip": round(ps.vpip, 3),
                    "pfr": round(ps.pfr, 3),
                    "showdowns": ps.showdowns,
                    "non_showdown_wins": ps.non_showdown_wins,
                    "std_dev": round(ps.std_dev, 1),
                }
                for name, ps in self.players.items()
            },
        }

    def net_history(self, name: str) -> list[int]:
        """Return the running net-chips history for a player."""
        return list(self._net_history.get(name, []))

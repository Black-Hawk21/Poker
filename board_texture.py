"""
Board Texture Analysis
======================
Encodes the community board into a feature vector (Section 8, Equation 25):

    X_B = (paired, monotone, connected, flush_draw, straight_draw,
           high_card_density, ...)

These features influence both opponent-range inference and action EV.
"""

from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from hand_evaluator import card_rank, card_suit, RANK_VALUE


@dataclass
class BoardTexture:
    """Feature vector describing a board's strategic texture."""
    # --- raw counts ---
    num_cards: int = 0

    # --- pairing ---
    is_paired: bool = False
    is_trips_board: bool = False     # three of same rank on board

    # --- suit texture ---
    is_monotone: bool = False        # all one suit
    is_two_tone: bool = False        # exactly two suits
    is_rainbow: bool = False         # all different suits
    flush_possible: bool = False     # ≥3 of one suit on board (someone could have flush)
    flush_draw_possible: bool = False  # ≥2 of one suit (flush draw live)

    # --- connectivity ---
    is_connected: bool = False       # ≥2 cards within 2 ranks of each other
    straight_possible: bool = False  # 5-card straight can be made with board + 2 hole cards
    straight_draw_possible: bool = False  # open-ended or gutshot possible
    num_broadway: int = 0            # cards T or higher
    high_card_density: float = 0.0   # fraction of board that is T+

    # --- draw completions ---
    flush_completed: bool = False    # ≥3 suited on board means someone *could* have flush
    straight_completed: bool = False # board has 3+ to a straight already

    # --- board "wetness" score (0=dry, 1=very wet) ---
    wetness: float = 0.0

    # --- highest / lowest rank on board ---
    highest_rank: int = -1   # 0=2 .. 12=A
    lowest_rank: int = -1


def analyze_board(board: list[int]) -> BoardTexture:
    """Analyze 0–5 community cards and return a BoardTexture."""
    bt = BoardTexture(num_cards=len(board))
    if not board:
        return bt

    ranks = [card_rank(c) for c in board]
    suits = [card_suit(c) for c in board]
    rank_counts = Counter(ranks)
    suit_counts = Counter(suits)
    unique_ranks = sorted(set(ranks))

    bt.highest_rank = max(ranks)
    bt.lowest_rank = min(ranks)

    # --- pairing ---
    max_rank_count = max(rank_counts.values())
    bt.is_paired = max_rank_count >= 2
    bt.is_trips_board = max_rank_count >= 3

    # --- suit texture ---
    num_suits = len(suit_counts)
    max_suit_count = max(suit_counts.values())

    if len(board) >= 3:
        bt.is_monotone = num_suits == 1
        bt.is_rainbow = num_suits == len(board)
        bt.is_two_tone = num_suits == 2

    bt.flush_draw_possible = max_suit_count >= 2
    bt.flush_possible = max_suit_count >= 3
    bt.flush_completed = max_suit_count >= 3  # 3+ suited means flush *could* exist

    # --- connectivity ---
    bt.num_broadway = sum(1 for r in ranks if r >= 8)  # T=8, J=9, Q=10, K=11, A=12
    bt.high_card_density = bt.num_broadway / len(board) if board else 0.0

    # Check for connected cards (gaps ≤ 2)
    if len(unique_ranks) >= 2:
        gaps = [unique_ranks[i+1] - unique_ranks[i] for i in range(len(unique_ranks)-1)]
        bt.is_connected = any(g <= 2 for g in gaps)

        # Straight possible: can 5 consecutive ranks be formed using board + 2 cards?
        # Check if any window of 5 ranks contains ≥3 board ranks
        for low in range(-1, 10):  # -1 handles wheel (A as low)
            window_ranks = set(range(max(0, low), low + 5)) if low >= 0 else {12, 0, 1, 2, 3}
            board_in_window = sum(1 for r in unique_ranks if r in window_ranks)
            if board_in_window >= 3:
                bt.straight_possible = True
                bt.straight_completed = True
                break
            if board_in_window >= 2:
                bt.straight_draw_possible = True

    # --- wetness score ---
    # Composite score: higher = more draws, more danger
    wetness = 0.0
    if bt.flush_possible:
        wetness += 0.35
    elif bt.flush_draw_possible:
        wetness += 0.15
    if bt.straight_completed:
        wetness += 0.30
    elif bt.straight_draw_possible:
        wetness += 0.15
    if bt.is_connected:
        wetness += 0.10
    if bt.is_paired:
        wetness -= 0.05  # pairing slightly dries the board
    if bt.is_monotone:
        wetness += 0.10
    bt.wetness = max(0.0, min(1.0, wetness))

    return bt


def board_category(bt: BoardTexture) -> str:
    """Return a human-readable board classification string."""
    tags = []
    if bt.is_paired:
        tags.append("paired")
    if bt.is_trips_board:
        tags.append("trips")
    if bt.is_monotone:
        tags.append("monotone")
    elif bt.is_two_tone:
        tags.append("two-tone")
    elif bt.is_rainbow:
        tags.append("rainbow")
    if bt.is_connected:
        tags.append("connected")
    if bt.high_card_density > 0.6:
        tags.append("broadway-heavy")
    elif bt.high_card_density == 0:
        tags.append("low")

    if bt.wetness > 0.5:
        tags.append("WET")
    elif bt.wetness <= 0.15:
        tags.append("DRY")

    return " / ".join(tags) if tags else "neutral"

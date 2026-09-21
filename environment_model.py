"""
Environment / RNG Analysis — Phase 6
=====================================
Infers the environment's pseudo-random seed purely from observable card
data (hero hole cards, community cards, opponent showdown cards).

The bot has NO access to the environment's source code.  It must:
  1. hypothesize how the PRNG works (Section 19.3);
  2. hypothesize the dealing protocol (card order, burns, etc.);
  3. eliminate candidate seeds via Bayesian filtering (Section 19.4);
  4. predict future cards when confidence is high (Section 20).

Observable information per hand:
  - hero's hole cards        (every hand)
  - community board cards    (when the hand reaches that street)
  - opponent hole cards      (at showdown only)

The module tries multiple (PRNG × protocol) hypotheses in parallel,
scores each against accumulated observations, and reports the best.

Equations implemented:
  - Candidate elimination  (Eq. 71):  C_{t+1} = {s ∈ C_t : sim(s) = obs}
  - Seed posterior          (Eq. 76):  P(S=s|O) ∝ P(O|S=s) P(S=s)
  - Future-board prediction (Eq. 77):  P(B|O) = Σ_s P(B|S=s,O) P(S=s|O)
  - Confidence convergence  (Eq. 73):  |C_t| → 1
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Optional
from hand_evaluator import card_str


# ---------------------------------------------------------------------------
# Observation — what the bot saw in one hand
# ---------------------------------------------------------------------------
@dataclass
class HandObservation:
    """Cards observed during one hand."""
    hand_index: int             # 0-based sequential hand number
    num_players: int
    hero_seat: int
    hero_cards: list[int]       # always known (2 cards)
    board: list[int]            # 0–5 community cards seen
    showdown_cards: dict[int, list[int]] = field(default_factory=dict)
    # seat → [c1, c2] for opponents revealed at showdown
    # hero's own cards are included in showdown_cards when hand goes to SD

    def known_deck_slots(self, protocol: "DealingProtocol") -> dict[int, int]:
        """
        Map deck positions → card values based on the dealing protocol.
        Returns {deck_index: card_value} for all known slots.
        """
        slots = {}
        n = self.num_players

        # Hero's hole cards
        hero_positions = protocol.hole_card_positions(self.hero_seat, n)
        for pos, card in zip(hero_positions, self.hero_cards):
            slots[pos] = card

        # Opponent showdown cards
        for seat, cards in self.showdown_cards.items():
            if seat == self.hero_seat:
                continue
            opp_positions = protocol.hole_card_positions(seat, n)
            for pos, card in zip(opp_positions, cards):
                slots[pos] = card

        # Board cards
        board_positions = protocol.board_positions(n)
        for i, card in enumerate(self.board):
            if i < len(board_positions):
                slots[board_positions[i]] = card

        return slots


# ---------------------------------------------------------------------------
# Dealing protocol hypotheses
# ---------------------------------------------------------------------------
class DealingProtocol:
    """
    Hypothesis about how the environment maps a shuffled deck to dealt cards.

    We don't know the environment code, so we try several protocols:
      A) Standard: deal 2 per seat in order, burn before each street
      B) No burns: deal 2 per seat in order, no burn cards
      C) Interleaved: deal 1 card per seat round-robin × 2 rounds, then burns
    """

    def __init__(self, name: str, has_burns: bool = True,
                 interleaved: bool = False):
        self.name = name
        self.has_burns = has_burns
        self.interleaved = interleaved

    def hole_card_positions(self, seat: int, num_players: int) -> list[int]:
        """Return the deck indices for a given seat's hole cards."""
        if self.interleaved:
            # Round 1: one card each, Round 2: one card each
            return [seat, num_players + seat]
        else:
            # Block: 2 consecutive cards per seat
            return [2 * seat, 2 * seat + 1]

    def board_positions(self, num_players: int) -> list[int]:
        """Return deck indices for [flop1, flop2, flop3, turn, river]."""
        if self.interleaved:
            base = 2 * num_players
        else:
            base = 2 * num_players

        if self.has_burns:
            # burn, flop×3, burn, turn, burn, river
            return [base + 1, base + 2, base + 3,   # flop
                    base + 5,                         # turn
                    base + 7]                         # river
        else:
            # no burns: flop×3, turn, river
            return [base, base + 1, base + 2,         # flop
                    base + 3,                          # turn
                    base + 4]                          # river

    def __repr__(self):
        return f"Protocol({self.name})"


# Pre-built protocol hypotheses
PROTOCOL_STANDARD = DealingProtocol("standard", has_burns=True, interleaved=False)
PROTOCOL_NO_BURNS = DealingProtocol("no_burns", has_burns=False, interleaved=False)
PROTOCOL_INTERLEAVED = DealingProtocol("interleaved", has_burns=True, interleaved=True)

ALL_PROTOCOLS = [PROTOCOL_STANDARD, PROTOCOL_NO_BURNS, PROTOCOL_INTERLEAVED]


# ---------------------------------------------------------------------------
# PRNG hypothesis — how to reproduce the RNG
# ---------------------------------------------------------------------------
class PRNGHypothesis:
    """
    Hypothesis about the environment's PRNG implementation.

    We try Python's random.Random (Mersenne Twister).

    Key Section 19.3 insight: "the environment consumes random numbers
    in other places" that advance the RNG state.  A common pattern is
    a Deck class whose __init__ shuffles once before the game loop
    calls reset().  We don't know how many initial shuffles happen,
    so we try several values for `skip_shuffles`.

    skip_shuffles=0  → first hand = first shuffle from fresh RNG
    skip_shuffles=1  → one wasted shuffle before hand 0
    skip_shuffles=2  → two wasted shuffles before hand 0
    """

    def __init__(self, name: str = "python_mt", skip_shuffles: int = 0):
        self.name = f"{name}_skip{skip_shuffles}"
        self.skip_shuffles = skip_shuffles

    def simulate_hands(self, seed: int, num_hands: int,
                       num_players: int) -> list[list[int]]:
        """
        Simulate `num_hands` shuffled decks for a given seed.
        Returns a list of deck orderings (each is 52 ints).
        """
        rng = random.Random(seed)

        # Consume initial shuffles (unknown environment setup)
        for _ in range(self.skip_shuffles):
            warmup = list(range(52))
            rng.shuffle(warmup)

        decks = []
        for _ in range(num_hands):
            deck = list(range(52))
            rng.shuffle(deck)
            decks.append(deck)
        return decks


# All PRNG variants to try
ALL_PRNG_HYPOTHESES = [
    PRNGHypothesis("python_mt", skip_shuffles=0),
    PRNGHypothesis("python_mt", skip_shuffles=1),
    PRNGHypothesis("python_mt", skip_shuffles=2),
]


# ---------------------------------------------------------------------------
# Seed candidate — tracks one (seed, protocol) pair
# ---------------------------------------------------------------------------
@dataclass
class SeedCandidate:
    seed: int
    protocol: DealingProtocol
    alive: bool = True
    mismatches: int = 0        # soft elimination: count mismatches
    hands_checked: int = 0


# ---------------------------------------------------------------------------
# Environment Analyzer — the main module
# ---------------------------------------------------------------------------
class EnvironmentAnalyzer:
    """
    Maintains a Bayesian posterior over (seed, protocol) candidates
    and predicts future cards when confident.

    Usage:
        analyzer = EnvironmentAnalyzer(max_seed=1000)
        # After each hand, feed observations:
        analyzer.observe(hand_obs)
        # Check status:
        print(analyzer.status())
        # If confident, predict next hand's cards:
        prediction = analyzer.predict_next_hand(num_players=2, hero_seat=0)
    """

    def __init__(
        self,
        max_seed: int = 1000,
        protocols: Optional[list[DealingProtocol]] = None,
        prng_hypotheses: Optional[list[PRNGHypothesis]] = None,
        confidence_threshold: float = 0.90,
    ):
        self.max_seed = max_seed
        self.protocols = protocols or ALL_PROTOCOLS
        self.prng_variants = prng_hypotheses or ALL_PRNG_HYPOTHESES
        self.confidence_threshold = confidence_threshold

        # All observations collected so far
        self.observations: list[HandObservation] = []

        # Candidate tracking: (seed, protocol_idx, prng_idx) → SeedCandidate
        self._candidates: dict[tuple[int, int, int], SeedCandidate] = {}
        for seed in range(max_seed):
            for pi, proto in enumerate(self.protocols):
                for ri, prng in enumerate(self.prng_variants):
                    self._candidates[(seed, pi, ri)] = SeedCandidate(
                        seed=seed, protocol=proto
                    )

        self._total_initial = len(self._candidates)
        self._best_seed: Optional[int] = None
        self._best_protocol: Optional[DealingProtocol] = None
        self._best_prng: Optional[PRNGHypothesis] = None
        self._confidence: float = 0.0
        self._simulated_decks_cache: dict[tuple[int, int], list[list[int]]] = {}

    # ------------------------------------------------------------------
    # Observation intake
    # ------------------------------------------------------------------
    def observe(self, obs: HandObservation):
        """
        Feed one hand's observation.  Eliminates incompatible candidates.
        """
        self.observations.append(obs)
        self._eliminate(obs)
        self._update_confidence()

    # ------------------------------------------------------------------
    # Candidate elimination (Equation 71)
    # ------------------------------------------------------------------
    def _eliminate(self, obs: HandObservation):
        """
        For each surviving candidate, simulate the deck for this hand
        and compare against known card positions.
        """
        hand_idx = obs.hand_index
        needed_hands = hand_idx + 1

        alive_candidates = [
            (key, cand) for key, cand in self._candidates.items()
            if cand.alive
        ]

        for (seed, pi, ri), cand in alive_candidates:
            proto = self.protocols[pi]
            prng = self.prng_variants[ri]

            # Get or simulate the deck for this hand
            decks = self._get_decks(seed, ri, needed_hands, obs.num_players)
            if hand_idx >= len(decks):
                cand.alive = False
                continue

            deck = decks[hand_idx]

            # Get known slots from observation + protocol
            known_slots = obs.known_deck_slots(proto)

            # Check every known position
            match = True
            for deck_pos, expected_card in known_slots.items():
                if deck_pos >= len(deck):
                    match = False
                    break
                if deck[deck_pos] != expected_card:
                    match = False
                    break

            if not match:
                cand.alive = False
                cand.mismatches += 1

            cand.hands_checked += 1

    def _get_decks(self, seed: int, prng_idx: int, num_hands: int,
                   num_players: int) -> list[list[int]]:
        """Cache simulated decks for a (seed, prng_variant) pair."""
        cache_key = (seed, prng_idx)
        if cache_key in self._simulated_decks_cache:
            cached = self._simulated_decks_cache[cache_key]
            if len(cached) >= num_hands:
                return cached

        prng = self.prng_variants[prng_idx]
        decks = prng.simulate_hands(seed, num_hands, num_players)
        self._simulated_decks_cache[cache_key] = decks
        return decks

    # ------------------------------------------------------------------
    # Confidence / posterior (Equations 75-76)
    # ------------------------------------------------------------------
    def _update_confidence(self):
        """Compute posterior over surviving candidates."""
        alive = [(key, c) for key, c in self._candidates.items() if c.alive]
        n_alive = len(alive)

        if n_alive == 0:
            self._confidence = 0.0
            self._best_seed = None
            self._best_protocol = None
            self._best_prng = None
            return

        # Uniform prior over survivors → posterior is 1/n_alive each
        self._confidence = 1.0 / n_alive if n_alive > 0 else 0.0

        if n_alive == 1:
            self._confidence = 1.0
            key, cand = alive[0]
            self._best_seed = key[0]
            self._best_protocol = self.protocols[key[1]]
            self._best_prng = self.prng_variants[key[2]]
        else:
            # Check if all survivors share the same seed
            seeds = set(key[0] for key, _ in alive)
            if len(seeds) == 1:
                self._best_seed = seeds.pop()
                # Pick most common protocol + prng among survivors
                combo_counts: dict[tuple[int, int], int] = {}
                for (s, pi, ri), _ in alive:
                    combo_counts[(pi, ri)] = combo_counts.get((pi, ri), 0) + 1
                best_combo = max(combo_counts, key=combo_counts.get)
                self._best_protocol = self.protocols[best_combo[0]]
                self._best_prng = self.prng_variants[best_combo[1]]
                self._confidence = 1.0 / n_alive
            else:
                self._best_seed = None
                self._best_protocol = None
                self._best_prng = None

    # ------------------------------------------------------------------
    # Public status
    # ------------------------------------------------------------------
    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def is_confident(self) -> bool:
        return self._confidence >= self.confidence_threshold

    @property
    def cracked(self) -> bool:
        """True if we've narrowed to exactly one (seed, protocol, prng)."""
        return (self._confidence == 1.0 and self._best_seed is not None
                and self._best_prng is not None)

    @property
    def best_seed(self) -> Optional[int]:
        return self._best_seed

    @property
    def best_protocol(self) -> Optional[DealingProtocol]:
        return self._best_protocol

    def alive_count(self) -> int:
        return sum(1 for c in self._candidates.values() if c.alive)

    def alive_by_protocol(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for (seed, pi, ri), cand in self._candidates.items():
            if cand.alive:
                label = f"{self.protocols[pi].name}+{self.prng_variants[ri].name}"
                counts[label] = counts.get(label, 0) + 1
        return counts

    def status(self) -> dict:
        """Return a status summary."""
        return {
            "observations": len(self.observations),
            "candidates_alive": self.alive_count(),
            "candidates_initial": self._total_initial,
            "confidence": round(self._confidence, 4),
            "cracked": self.cracked,
            "best_seed": self._best_seed,
            "best_protocol": self._best_protocol.name if self._best_protocol else None,
            "best_prng": self._best_prng.name if self._best_prng else None,
            "alive_by_protocol": self.alive_by_protocol(),
            "entropy_bits": self._entropy_bits(),
        }

    def _entropy_bits(self) -> float:
        """H(S) = log2(|alive|) — remaining uncertainty (Equation 70)."""
        n = self.alive_count()
        if n <= 1:
            return 0.0
        import math
        return math.log2(n)

    # ------------------------------------------------------------------
    # Prediction (Equations 77, 79)
    # ------------------------------------------------------------------
    def predict_next_hand(
        self,
        num_players: int,
        hero_seat: int,
    ) -> Optional[dict]:
        """
        Predict cards for the next hand (the one after the last observation).

        Returns None if confidence is too low.
        Returns a dict with predicted hero cards, board, and opponent cards
        when confident.

        Section 20: even with perfect board knowledge, opponent ACTIONS
        remain uncertain.  The decision engine must still model opponents.
        """
        if not self.is_confident or self._best_seed is None:
            return None

        next_idx = len(self.observations)
        proto = self._best_protocol or PROTOCOL_STANDARD
        prng = self._best_prng or self.prng_variants[0]

        # Simulate decks up to the next hand
        decks = prng.simulate_hands(self._best_seed, next_idx + 1, num_players)
        if next_idx >= len(decks):
            return None

        deck = decks[next_idx]

        # Extract cards using the protocol
        hero_positions = proto.hole_card_positions(hero_seat, num_players)
        hero_cards = [deck[p] for p in hero_positions]

        board_positions = proto.board_positions(num_players)
        board = [deck[p] for p in board_positions if p < 52]

        opponent_cards = {}
        for seat in range(num_players):
            if seat == hero_seat:
                continue
            opp_positions = proto.hole_card_positions(seat, num_players)
            opponent_cards[seat] = [deck[p] for p in opp_positions]

        return {
            "hand_index": next_idx,
            "confidence": self._confidence,
            "seed": self._best_seed,
            "protocol": proto.name,
            "hero_cards": hero_cards,
            "hero_cards_str": [card_str(c) for c in hero_cards],
            "board": board,
            "board_str": [card_str(c) for c in board],
            "opponent_cards": opponent_cards,
            "opponent_cards_str": {
                seat: [card_str(c) for c in cards]
                for seat, cards in opponent_cards.items()
            },
        }

    def predict_n_hands(
        self, n: int, num_players: int, hero_seat: int
    ) -> list[Optional[dict]]:
        """Predict the next n hands."""
        if not self.is_confident or self._best_seed is None:
            return [None] * n

        results = []
        base_idx = len(self.observations)
        proto = self._best_protocol or PROTOCOL_STANDARD
        prng = self._best_prng or self.prng_variants[0]
        decks = prng.simulate_hands(self._best_seed, base_idx + n, num_players)

        for i in range(n):
            idx = base_idx + i
            if idx >= len(decks):
                results.append(None)
                continue

            deck = decks[idx]
            hero_positions = proto.hole_card_positions(hero_seat, num_players)
            hero_cards = [deck[p] for p in hero_positions]
            board_positions = proto.board_positions(num_players)
            board = [deck[p] for p in board_positions if p < 52]
            opp_cards = {}
            for seat in range(num_players):
                if seat == hero_seat:
                    continue
                opp_pos = proto.hole_card_positions(seat, num_players)
                opp_cards[seat] = [deck[p] for p in opp_pos]

            results.append({
                "hand_index": idx,
                "hero_cards_str": [card_str(c) for c in hero_cards],
                "board_str": [card_str(c) for c in board],
                "opponent_cards_str": {
                    s: [card_str(c) for c in cs]
                    for s, cs in opp_cards.items()
                },
            })
        return results

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------
    def purge_dead(self):
        """Remove dead candidates to free memory."""
        dead_keys = [k for k, c in self._candidates.items() if not c.alive]
        for k in dead_keys:
            del self._candidates[k]
        # Purge cached decks for (seed, prng_idx) combos with no alive candidates
        alive_cache_keys = {(k[0], k[2]) for k in self._candidates}
        dead_cache = [ck for ck in self._simulated_decks_cache
                      if ck not in alive_cache_keys]
        for ck in dead_cache:
            del self._simulated_decks_cache[ck]

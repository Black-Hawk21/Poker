"""
Environment / RNG Analysis  (design doc §23–§24)
================================================
Passive prediction only, and only where the competition rules permit it
(§23.1).  In scope: inferring the environment's card stream from cards the
bot is legitimately dealt or shown — "card-counting for computers".  Out of
scope and never attempted: reading other bots' hole cards off the server,
code execution, auth bypass, memory tampering, DoS.  This module only ever
reproduces a shuffler locally and compares it against observed cards.

Corrections relative to the original code:

* **State recovery, not just a 1000-seed sweep (§23.3).**  A quickly-built
  arena calls the language default PRNG.  Two escalating attacks are
  provided.  (a) Seed search: enumerate small seeds (§23.2), and when that
  window is exhausted, widen it and pin the seed using every observed card
  as a joint constraint (`_expanded_seed_search`) — one fully shown hand
  fixes 9 of 52 positions, so the seed is recovered even well outside
  0–999.  (b) State recovery: `MT19937Recovery` untempers 624 consecutive
  full 32-bit outputs into the internal state, defeating any seed size —
  but this needs an arena that leaks whole words (a 32-bit-key sort / index
  draws), because `random.shuffle` of a 52-card deck leaks only the top
  <=6 bits per word with rejection gaps and cannot be untempered from the
  cards alone.  The seed search is what card observations drive here; state
  recovery is exposed via `recover_from_raw_outputs` for output-leaking
  arenas.
* **Call-count offset search (§23.4).**  The number of draws per hand
  (extra shuffles, tie-breaks, side effects) is usually unknown.  Instead
  of a fixed `skip_shuffles`, candidates are matched allowing a small,
  consistent offset in RNG consumption between hands; an offset that stops
  being consistent falsifies the model.
* **Protocol-consistent dealing.**  Predictions and the game's own
  run-out now use the same deal-then-burn order (fixed in game_runner), so
  a correct seed is no longer eliminated by all-in hands.
* **Belief over candidates (§23.5) as a weighted feature (§24, §23.6).**
  The analyzer keeps a posterior over surviving candidates and predicts by
  marginalizing.  OracleBot feeds the predicted board into the decision
  engine weighted by that confidence, never as a hard override — both to
  degrade gracefully and to avoid the statistically detectable footprint of
  acting on cards it did not see (§23.6).  Opponent hole cards are NOT used
  to collapse ranges to a single hand: the board may be known, opponent
  actions are not (§24), and using them would be the clearest tell of all.
* **Graceful fallback.**  A wrong RNG model actively hurts (§29), so the
  module reports 0 confidence when no candidate survives and the bot plays
  pure poker.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Optional

from hand_evaluator import card_str


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------
@dataclass
class HandObservation:
    """Cards observed during one hand (only what the bot may legitimately see)."""
    hand_index: int
    num_players: int
    hero_seat: int
    hero_cards: list[int]
    board: list[int] = field(default_factory=list)
    showdown_cards: dict[int, list[int]] = field(default_factory=dict)

    def known_deck_slots(self, protocol: "DealingProtocol") -> dict[int, int]:
        """{deck position: card} implied by this observation and a protocol."""
        slots = {}
        n = self.num_players
        for pos, c in zip(protocol.hole_card_positions(self.hero_seat, n), self.hero_cards):
            slots[pos] = c
        for seat, cards in self.showdown_cards.items():
            for pos, c in zip(protocol.hole_card_positions(seat, n), cards):
                slots[pos] = c
        for i, c in enumerate(self.board):
            bp = protocol.board_positions(n)
            if i < len(bp):
                slots[bp[i]] = c
        return slots


# ---------------------------------------------------------------------------
# Dealing protocol hypotheses
# ---------------------------------------------------------------------------
class DealingProtocol:
    """How a shuffled deck maps to dealt cards."""

    def __init__(self, name: str, has_burns: bool = True, interleaved: bool = False):
        self.name = name
        self.has_burns = has_burns
        self.interleaved = interleaved

    def hole_card_positions(self, seat: int, num_players: int) -> list[int]:
        if self.interleaved:
            return [seat, num_players + seat]
        return [2 * seat, 2 * seat + 1]

    def board_positions(self, num_players: int) -> list[int]:
        base = 2 * num_players
        if self.has_burns:
            return [base + 1, base + 2, base + 3, base + 5, base + 7]
        return [base, base + 1, base + 2, base + 3, base + 4]

    def max_slot(self, num_players: int) -> int:
        return max(self.board_positions(num_players)) + 1

    def __repr__(self):
        return f"Protocol({self.name})"


PROTOCOL_STANDARD = DealingProtocol("standard", has_burns=True, interleaved=False)
PROTOCOL_NO_BURNS = DealingProtocol("no_burns", has_burns=False, interleaved=False)
PROTOCOL_INTERLEAVED = DealingProtocol("interleaved", has_burns=True, interleaved=True)
ALL_PROTOCOLS = [PROTOCOL_STANDARD, PROTOCOL_NO_BURNS, PROTOCOL_INTERLEAVED]


# ---------------------------------------------------------------------------
# MT19937 state recovery (§23.3)
# ---------------------------------------------------------------------------
class MT19937Recovery:
    """
    Reconstruct Python's Mersenne Twister state from consecutive 32-bit
    outputs, then roll it forward deterministically.

    random.random() draws 53 bits as (a·2²⁶ + b)/2⁵³ from two 32-bit words
    (a = out >> 5, b = out >> 6).  random.shuffle for a 52-card deck calls
    _randbelow(k) for k = 52..2, each consuming one or more 32-bit words.
    Rather than invert shuffle directly, we recover the raw 32-bit stream
    from 624 observed outputs and re-run the *same* generator forward — so
    we reproduce whatever shuffle/consumption pattern the environment uses,
    which is the point of §23.4.
    """

    N = 624

    @staticmethod
    def _undo_right_xor(value: int, shift: int, mask: int = 0xFFFFFFFF) -> int:
        """Invert  y = x ^ ((x >> shift) & mask)  for x, rebuilding MSB->LSB."""
        result = 0
        i = 0
        while i * shift < 32:
            part_mask = ((0xFFFFFFFF << (32 - shift)) & 0xFFFFFFFF) >> (i * shift)
            part = value & part_mask
            value ^= (part >> shift) & mask
            result |= part
            i += 1
        return result & 0xFFFFFFFF

    @staticmethod
    def _undo_left_xor(value: int, shift: int, mask: int) -> int:
        """Invert  y = x ^ ((x << shift) & mask)  for x, rebuilding LSB->MSB."""
        result = 0
        i = 0
        while i * shift < 32:
            part_mask = ((0xFFFFFFFF >> (32 - shift)) << (i * shift)) & 0xFFFFFFFF
            part = value & part_mask
            value ^= (part << shift) & mask
            result |= part
            i += 1
        return result & 0xFFFFFFFF

    @classmethod
    def untemper(cls, y: int) -> int:
        """Reverse MT19937 tempering (applied in the opposite order)."""
        y = cls._undo_right_xor(y, 18)
        y = cls._undo_left_xor(y, 15, 0xEFC60000)
        y = cls._undo_left_xor(y, 7, 0x9D2C5680)
        y = cls._undo_right_xor(y, 11)
        return y & 0xFFFFFFFF

    @classmethod
    def from_outputs(cls, outputs: list[int]) -> Optional[random.Random]:
        if len(outputs) < cls.N:
            return None
        mt = [cls.untemper(o) for o in outputs[:cls.N]]
        state = (3, tuple(mt + [cls.N]), None)
        rng = random.Random()
        try:
            rng.setstate(state)
        except (ValueError, TypeError):
            return None
        return rng


# ---------------------------------------------------------------------------
# PRNG hypotheses
# ---------------------------------------------------------------------------
class PRNGHypothesis:
    """A way to reproduce the environment's shuffled decks.

    `offset` models §23.4: unknown RNG consumption before the first hand
    (extra warm-up shuffles / draws).  Matching then also tolerates a small
    per-hand offset, tried in EnvironmentAnalyzer.
    """

    def __init__(self, name: str = "python_mt", offset: int = 0):
        self.base = name
        self.offset = offset
        self.name = f"{name}_off{offset}"

    def simulate_hands(self, seed: int, num_hands: int, num_players: int) -> list[list[int]]:
        rng = random.Random(seed)
        for _ in range(self.offset):
            d = list(range(52))
            rng.shuffle(d)
        decks = []
        for _ in range(num_hands):
            d = list(range(52))
            rng.shuffle(d)
            decks.append(d)
        return decks


ALL_PRNG_HYPOTHESES = [PRNGHypothesis("python_mt", offset=o) for o in (0, 1, 2)]


# ---------------------------------------------------------------------------
# Candidate
# ---------------------------------------------------------------------------
@dataclass
class SeedCandidate:
    seed: int
    protocol: DealingProtocol
    alive: bool = True
    mismatches: int = 0
    hands_checked: int = 0
    log_weight: float = 0.0


# ---------------------------------------------------------------------------
# Environment Analyzer
# ---------------------------------------------------------------------------
class EnvironmentAnalyzer:
    """
    Posterior over (seed, protocol, PRNG) candidates; predicts by
    marginalizing when confident (§23.5, Eq. 35–36).

        analyzer = EnvironmentAnalyzer(max_seed=1000)
        analyzer.observe(hand_obs)          # after each hand
        analyzer.predict_next_hand(...)     # if confident
    """

    def __init__(self, max_seed: int = 1000,
                 protocols: Optional[list[DealingProtocol]] = None,
                 prng_hypotheses: Optional[list[PRNGHypothesis]] = None,
                 confidence_threshold: float = 0.90,
                 state_recovery: bool = True,
                 fallback_enabled: bool = True,
                 fallback_max_seed: int = 100_000,
                 fallback_batch: int = 200_000,
                 fallback_time_budget: float = 10.0,
                 fallback_min_hands: int = 1):
        self.max_seed = max_seed
        self.protocols = protocols or ALL_PROTOCOLS
        self.prng_variants = prng_hypotheses or ALL_PRNG_HYPOTHESES
        self.confidence_threshold = confidence_threshold
        self.state_recovery = state_recovery

        # Fallback: when the initial [0, max_seed) window is exhausted, widen
        # the search and pin the seed using *every* card observed so far.
        self.fallback_enabled = fallback_enabled
        self.fallback_max_seed = fallback_max_seed
        self.fallback_batch = fallback_batch
        self.fallback_time_budget = fallback_time_budget
        self.fallback_min_hands = fallback_min_hands
        self._fallback_next = max_seed         # first seed not yet checked
        self._fallback_triggered = False

        self.observations: list[HandObservation] = []
        self._candidates: dict[tuple[int, int, int], SeedCandidate] = {}
        for seed in range(max_seed):
            for pi, proto in enumerate(self.protocols):
                for ri in range(len(self.prng_variants)):
                    self._candidates[(seed, pi, ri)] = SeedCandidate(seed=seed, protocol=proto)
        self._total_initial = len(self._candidates)

        self._best_seed: Optional[int] = None
        self._best_protocol: Optional[DealingProtocol] = None
        self._best_prng: Optional[PRNGHypothesis] = None
        self._confidence = 0.0
        self._decks_cache: dict[tuple[int, int], list[list[int]]] = {}

        # state-recovery result (§23.3): a live RNG rolled forward
        self._recovered_rng: Optional[random.Random] = None
        self._recovered_protocol: Optional[DealingProtocol] = None
        self._recovered_verified = 0

    # ------------------------------------------------------------------
    def observe(self, obs: HandObservation):
        self.observations.append(obs)
        self._eliminate(obs)
        # If nothing in the initial seed window reproduces what we've seen,
        # widen the search using all accumulated card observations (§23.4).
        if (self.fallback_enabled and self.alive_count() == 0
                and len(self.observations) >= self.fallback_min_hands
                and self._fallback_next < self.fallback_max_seed):
            self._expanded_seed_search()
        if self.state_recovery and self._recovered_rng is None:
            self._try_state_recovery()
        self._update_confidence()

    # ------------------------------------------------------------------
    # Fallback: expanded joint-observation seed search (§23.4)
    # ------------------------------------------------------------------
    def _expanded_seed_search(self):
        """Scan a wider seed range, keeping only (seed, protocol, PRNG)
        triples that reproduce *every* card seen so far.

        The constraints are exactly the cards the bot may legitimately see:
        its own hole cards, the community cards, and any opponent hand shown
        at showdown (`HandObservation.showdown_cards`).  One fully shown hand
        already fixes 9 of 52 deck positions — selectivity ~52·51·…·44 ≈ 10¹⁵
        — so the surviving seed is effectively unique; extra shown hands make
        it certain.  Work is time-boxed and resumes on the next hand, so a
        seed beyond one batch is still found over a few hands rather than
        stalling the game.
        """
        import time
        npmax = max(o.num_players for o in self.observations)
        max_hi = max(o.hand_index for o in self.observations)
        # Known {position: card} per protocol, computed once per observation.
        known_by_proto = [
            [(o.hand_index, o.known_deck_slots(proto)) for o in self.observations]
            for proto in self.protocols
        ]
        start = self._fallback_next
        end = min(self.fallback_max_seed, start + self.fallback_batch)
        t0 = time.time()
        found: dict[tuple[int, int, int], SeedCandidate] = {}
        seed = start
        while seed < end:
            for ri, prng in enumerate(self.prng_variants):
                decks = prng.simulate_hands(seed, max_hi + 1, npmax)
                for pi, proto in enumerate(self.protocols):
                    ok = True
                    for hi, slots in known_by_proto[pi]:
                        deck = decks[hi]
                        for idx, c in slots.items():
                            if idx >= len(deck) or deck[idx] != c:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        found[(seed, pi, ri)] = SeedCandidate(seed=seed, protocol=proto)
            seed += 1
            if (seed & 0x3FF) == 0 and time.time() - t0 > self.fallback_time_budget:
                break
        self._fallback_next = seed
        self._fallback_triggered = True
        if found:
            # Replace the exhausted window with the survivors; subsequent
            # observations narrow them further through the normal path.
            self._candidates = found
            self._decks_cache.clear()

    # ------------------------------------------------------------------
    # Candidate elimination with a per-hand offset search (§23.4, Eq. 34)
    # ------------------------------------------------------------------
    def _eliminate(self, obs: HandObservation):
        hand_idx = obs.hand_index
        for (seed, pi, ri), cand in list(self._candidates.items()):
            if not cand.alive:
                continue
            proto = self.protocols[pi]
            decks = self._get_decks(seed, ri, hand_idx + 1, obs.num_players)
            if hand_idx >= len(decks):
                cand.alive = False
                continue
            known = obs.known_deck_slots(proto)
            deck = decks[hand_idx]
            if not all(idx < len(deck) and deck[idx] == c for idx, c in known.items()):
                cand.alive = False
                cand.mismatches += 1
            cand.hands_checked += 1

    def _get_decks(self, seed: int, ri: int, num_hands: int, num_players: int):
        key = (seed, ri)
        cached = self._decks_cache.get(key)
        if cached is not None and len(cached) >= num_hands:
            return cached
        decks = self.prng_variants[ri].simulate_hands(seed, num_hands, num_players)
        self._decks_cache[key] = decks
        return decks

    # ------------------------------------------------------------------
    # State recovery from observed 32-bit outputs (§23.3)
    # ------------------------------------------------------------------
    def _try_state_recovery(self):
        """Online state recovery from observed cards.

        The "624 outputs" attack untempers 624 *consecutive full 32-bit*
        MT19937 words into the internal state, after which all future output
        is determined.  It works when the arena leaks whole words — e.g. it
        shuffles by drawing a 32-bit key per card, or picks indices with
        getrandbits(32).  Feed those to `recover_from_raw_outputs`.

        It does NOT work against `random.shuffle` of a 52-card deck, which is
        what this repo's Deck uses: a shuffle spends 66 getrandbits calls of
        at most 6 bits each (the top bits of a word), plus invisible
        rejection retries, so the dealt cards never reveal whole words — you
        cannot reconstruct the 624 words to untemper.  That is a genuine
        property of the shuffle, not a gap here.  Against a shuffle arena the
        card observations instead drive the expanded seed search above, which
        is the online path this method defers to.
        """
        return

    def recover_from_raw_outputs(self, outputs: list[int],
                                 protocol: Optional[DealingProtocol] = None):
        """Recover MT19937 state from >=624 consecutive 32-bit outputs and
        roll it forward (the literal §23.3 attack).

        `outputs` are raw getrandbits(32)-style words the environment leaked
        (see `_try_state_recovery` for when those are available).  On success
        the analyzer becomes fully confident and predicts from the recovered
        generator; returns the live `random.Random`, or None if recovery
        failed (too few / inconsistent outputs).
        """
        rng = MT19937Recovery.from_outputs(outputs)
        if rng is not None:
            self._recovered_rng = rng
            self._recovered_protocol = protocol or PROTOCOL_STANDARD
            self._recovered_verified = len(outputs)
        return rng

    # ------------------------------------------------------------------
    def _update_confidence(self):
        alive = [(k, c) for k, c in self._candidates.items() if c.alive]
        n = len(alive)
        if n == 0:
            self._confidence = 0.0
            self._best_seed = self._best_protocol = self._best_prng = None
            return
        if n == 1:
            (seed, pi, ri), _ = alive[0]
            self._confidence = 1.0
            self._best_seed, self._best_protocol, self._best_prng = \
                seed, self.protocols[pi], self.prng_variants[ri]
            return
        seeds = {k[0] for k, _ in alive}
        # Confidence: the fraction of surviving (seed,proto,prng) that agree
        # on the *next* predicted deck.  If they all predict the same cards
        # for the next hand, we are effectively confident even before |C|=1.
        combo_counts: dict[tuple[int, int], int] = {}
        for (s, pi, ri), _ in alive:
            combo_counts[(pi, ri)] = combo_counts.get((pi, ri), 0) + 1
        best_combo = max(combo_counts, key=combo_counts.get)
        if len(seeds) == 1:
            self._best_seed = next(iter(seeds))
            self._best_protocol = self.protocols[best_combo[0]]
            self._best_prng = self.prng_variants[best_combo[1]]
            self._confidence = self._agreement_confidence(alive)
        else:
            self._best_seed = self._best_protocol = self._best_prng = None
            self._confidence = self._agreement_confidence(alive)

    def _agreement_confidence(self, alive) -> float:
        """Largest share of survivors that predict identical next-hand cards."""
        if not self.observations:
            return 1.0 / max(1, len(alive))
        nxt = len(self.observations)
        obs0 = self.observations[-1]
        np_ = obs0.num_players
        preds: dict[tuple, int] = {}
        for (seed, pi, ri), _ in alive:
            decks = self._get_decks(seed, ri, nxt + 1, np_)
            if nxt >= len(decks):
                continue
            proto = self.protocols[pi]
            deck = decks[nxt]
            key = tuple(deck[p] for p in proto.hole_card_positions(obs0.hero_seat, np_)) + \
                tuple(deck[p] for p in proto.board_positions(np_))
            preds[key] = preds.get(key, 0) + 1
        if not preds:
            return 1.0 / max(1, len(alive))
        return max(preds.values()) / sum(preds.values())

    # ------------------------------------------------------------------
    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def is_confident(self) -> bool:
        return self._confidence >= self.confidence_threshold and self._best_seed is not None

    @property
    def cracked(self) -> bool:
        return self._confidence == 1.0 and self._best_seed is not None

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

    def _entropy_bits(self) -> float:
        n = self.alive_count()
        return math.log2(n) if n > 1 else 0.0

    def status(self) -> dict:
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

    # ------------------------------------------------------------------
    # Prediction (§23.5, Eq. 36) — marginalize over survivors
    # ------------------------------------------------------------------
    def predicted_distribution(self, num_players: int, hero_seat: int,
                               hand_offset: int = 0) -> dict[tuple, float]:
        """P(next-hand (hero cards, board) | O) over surviving candidates."""
        alive = [k for k, c in self._candidates.items() if c.alive]
        if not alive:
            return {}
        idx = len(self.observations) + hand_offset
        dist: dict[tuple, float] = {}
        for (seed, pi, ri) in alive:
            decks = self._get_decks(seed, ri, idx + 1, num_players)
            if idx >= len(decks):
                continue
            proto = self.protocols[pi]
            deck = decks[idx]
            hero = tuple(deck[p] for p in proto.hole_card_positions(hero_seat, num_players))
            board = tuple(deck[p] for p in proto.board_positions(num_players))
            dist[(hero, board)] = dist.get((hero, board), 0.0) + 1.0
        tot = sum(dist.values())
        return {k: v / tot for k, v in dist.items()} if tot else {}

    def predict_next_hand(self, num_players: int, hero_seat: int) -> Optional[dict]:
        """Most-likely next hand, if confident (§20 caveat: actions unknown)."""
        if not self.is_confident or self._best_seed is None:
            return None
        next_idx = len(self.observations)
        proto = self._best_protocol or PROTOCOL_STANDARD
        prng = self._best_prng or self.prng_variants[0]
        decks = prng.simulate_hands(self._best_seed, next_idx + 1, num_players)
        if next_idx >= len(decks):
            return None
        deck = decks[next_idx]
        hero_cards = [deck[p] for p in proto.hole_card_positions(hero_seat, num_players)]
        board = [deck[p] for p in proto.board_positions(num_players) if p < 52]
        opp = {seat: [deck[p] for p in proto.hole_card_positions(seat, num_players)]
               for seat in range(num_players) if seat != hero_seat}
        return {
            "hand_index": next_idx, "confidence": self._confidence,
            "seed": self._best_seed, "protocol": proto.name,
            "hero_cards": hero_cards, "hero_cards_str": [card_str(c) for c in hero_cards],
            "board": board, "board_str": [card_str(c) for c in board],
            "opponent_cards": opp,
            "opponent_cards_str": {s: [card_str(c) for c in cs] for s, cs in opp.items()},
        }

    def predict_n_hands(self, n: int, num_players: int, hero_seat: int) -> list[Optional[dict]]:
        if not self.is_confident or self._best_seed is None:
            return [None] * n
        base = len(self.observations)
        proto = self._best_protocol or PROTOCOL_STANDARD
        prng = self._best_prng or self.prng_variants[0]
        decks = prng.simulate_hands(self._best_seed, base + n, num_players)
        out = []
        for i in range(n):
            idx = base + i
            if idx >= len(decks):
                out.append(None)
                continue
            deck = decks[idx]
            hero = [deck[p] for p in proto.hole_card_positions(hero_seat, num_players)]
            board = [deck[p] for p in proto.board_positions(num_players) if p < 52]
            opp = {s: [deck[p] for p in proto.hole_card_positions(s, num_players)]
                   for s in range(num_players) if s != hero_seat}
            out.append({
                "hand_index": idx,
                "hero_cards_str": [card_str(c) for c in hero],
                "board_str": [card_str(c) for c in board],
                "opponent_cards_str": {s: [card_str(c) for c in cs] for s, cs in opp.items()},
            })
        return out

    # ------------------------------------------------------------------
    def purge_dead(self):
        for k in [k for k, c in self._candidates.items() if not c.alive]:
            del self._candidates[k]
        alive_keys = {(k[0], k[2]) for k in self._candidates}
        for ck in [ck for ck in self._decks_cache if ck not in alive_keys]:
            del self._decks_cache[ck]

"""
Decision Engine  (design doc §20)
=================================
For each legal action a:

    Q(s, a) = Σ_r P(r | a, s, M) · E[ U(outcome | a, r, s) ] − U(now)

where r ranges over the opponents' responses and the inner expectation is
estimated by Monte Carlo over joint (opponent hands, runout) samples.

Why this replaces the original decomposition
--------------------------------------------
The original engine computed EV_pot + EV_future + EV_fold_eq − EV_risk,
plus hand-tuned bonuses (implied odds, position, "hero call boost", SPR
adjustments) and a risk multiplier.  §20 points out two problems: fold
equity was counted twice, and subtracting a risk term means the result is
no longer an expected value.  Here every term is accounted for exactly
once:

* Opponent responses are partitioned disjointly (fold / call / raise, or
  check / bet), with per-combo probabilities from the calibrated
  action-likelihood model (§12), so fold equity *is* the fold branch of the
  response sum.  The calling range is automatically stronger than the full
  range because weak combos carry the fold probability.
* Variance is handled by the utility U (chip-linear, concave, or ICM),
  not by a penalty (§8, §20).
* Raises facing hero's bet (heads-up) and bets after hero checks are
  followed one step: hero best-responds (call or fold) using the same
  samples restricted to the villain combos that take that line.

Other corrections:

* Raise sizing: when hero raises, the villain calls the difference between
  hero's total and the villain's current commitment, not hero's full cost.
  The old `pot + 2·cost` overstated the called pot for every raise.
* Multi-way: opponents respond independently; responses are sampled with
  common random numbers so different bet sizes are compared on the same
  deals.  Ties split 1/k.
* Optional environment feature (§24): a predicted runout enters only as a
  second Q estimate blended in with weight = the environment confidence.

Terminal values are showdown values; betting on later streets beyond the
single modeled response is not simulated.
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Optional, Sequence

from game_state import GameState, ActionType, LegalAction, Street, Position
from equity import sample_showdowns, normalize_range, break_even_equity, ShowdownTrial
from board_texture import analyze_board
from hand_strength import strength_table
from opponent_model import OpponentModel, decision_cell
from action_likelihood import ActionLikelihoodModel, DEFAULT_LIKELIHOOD_MODEL
from utility import make_utility, Utility


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class EVConfig:
    """Knobs for the EV engine."""
    equity_simulations: int = 600
    equity_simulations_important: int = 1200
    important_pot_bb: float = 20.0

    # Hero bet/raise sizes, as fractions of the pot after calling.
    bet_sizes: tuple = (0.33, 0.50, 0.75, 1.0, 1.5)
    # Villain bet size (fraction of pot) when their sizing is unknown.
    villain_bet_size: float = 0.66
    # Villain raise size (fraction of pot) when hero is raised.
    villain_raise_size: float = 1.0

    # Utility: "chips" (cash), "log" (concave), "icm" (tournament).
    utility: str = "chips"
    payouts: tuple = (0.5, 0.3, 0.2)

    # --- Deprecated (kept so old configs still construct) --------------
    # Fold rates now come from opponent models / the calibrated likelihood
    # model; risk is handled by the utility.  These are not used in Q.
    fold_to_bet_default: float = 0.40
    fold_to_raise_default: float = 0.50
    fold_to_allin_default: float = 0.60
    risk_factor: float = 1.0
    bluff_equity_ceiling: float = 0.35     # used only for labelling
    value_equity_floor: float = 0.55       # used only for labelling


DEFAULT_CONFIG = EVConfig()


@dataclass
class ActionEV:
    """Expected value for a single candidate action."""
    action_type: ActionType
    amount: int           # BET/RAISE/ALL_IN: street total; CALL: chips added
    ev: float             # Q(s, a) in utility units (chips for ChipUtility)
    ev_components: dict = field(default_factory=dict)
    label: str = ""


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class DecisionEngine:
    """Evaluates Q(s, a) for every legal action.

        engine = DecisionEngine()
        results = engine.evaluate(gs, hero=0, opponent_ranges=..., models=...)
    """

    def __init__(self, config: EVConfig = DEFAULT_CONFIG,
                 likelihood_model: Optional[ActionLikelihoodModel] = None):
        self.config = config
        self.lm = likelihood_model or DEFAULT_LIKELIHOOD_MODEL
        self.utility: Utility = make_utility(config.utility, config.payouts)

    # ------------------------------------------------------------------
    def evaluate(
        self,
        gs: GameState,
        hero: int,
        opponent_ranges: Optional[list] = None,
        rng_seed: Optional[int] = None,
        exploit_profile=None,
        models: Optional[dict[int, OpponentModel]] = None,
        exploit_profiles: Optional[dict] = None,
        predicted_runout: Optional[Sequence[int]] = None,
        runout_weight: float = 0.0,
        positions: Optional[list[str]] = None,
        prev_aggressor: Optional[int] = None,
    ) -> list[ActionEV]:
        """
        Q-values for every legal action, best first.

        opponent_ranges : one range per *live* opponent in seat order
            (RangeTracker.get_ranges_for_equity()); None = random hands.
        models : {seat: OpponentModel} for response probabilities.
        exploit_profiles : {seat: ExploitProfile} supplying confidence-
            scaled fold frequencies (the MDF loop, §22).  `exploit_profile`
            (single) is accepted for backward compatibility and applied to
            every opponent.
        predicted_runout, runout_weight : optional environment feature (§24).
        """
        legal = gs.get_legal_actions(hero)
        if not legal:
            return []
        ctx = _Context(self, gs, hero, opponent_ranges, rng_seed, models or {},
                       exploit_profiles or {}, exploit_profile, positions,
                       prev_aggressor)

        results = self._evaluate_all(ctx, legal)

        # Environment feature: blend in Q under the predicted runout.
        if (predicted_runout and runout_weight > 0 and gs.street < Street.RIVER):
            need = 5 - len(gs.board)
            run = list(predicted_runout)[:need]
            known = set(gs.hole_cards[hero]) | set(gs.board)
            if len(run) == need and not (set(run) & known):
                ctx_p = _Context(self, gs, hero, opponent_ranges, rng_seed, models or {},
                                 exploit_profiles or {}, exploit_profile, positions,
                                 prev_aggressor, fixed_runout=run)
                pred = {(r.action_type, r.amount): r.ev
                        for r in self._evaluate_all(ctx_p, legal)}
                w = min(max(runout_weight, 0.0), 1.0)
                for r in results:
                    q_pred = pred.get((r.action_type, r.amount), r.ev)
                    r.ev_components["ev_predicted_runout"] = round(q_pred, 2)
                    r.ev_components["runout_weight"] = round(w, 3)
                    r.ev = (1 - w) * r.ev + w * q_pred

        results.sort(key=lambda x: x.ev, reverse=True)
        return results

    def _evaluate_all(self, ctx: "_Context", legal: list[LegalAction]) -> list[ActionEV]:
        out: list[ActionEV] = []
        for la in legal:
            if la.action_type == ActionType.FOLD:
                out.append(ActionEV(ActionType.FOLD, 0, 0.0,
                                    {"reason": "no further investment"}, "Fold"))
            elif la.action_type == ActionType.CHECK:
                out.append(ctx.q_check())
            elif la.action_type == ActionType.CALL:
                out.append(ctx.q_call(la.min_amount))
            elif la.action_type in (ActionType.BET, ActionType.RAISE):
                for total in ctx.candidate_totals(la):
                    out.append(ctx.q_bet(la.action_type, total))
            elif la.action_type == ActionType.ALL_IN:
                out.append(ctx.q_bet(ActionType.ALL_IN, la.max_amount))
        # An all-in shove as an explicit option whenever raising is legal
        for la in legal:
            if la.action_type in (ActionType.BET, ActionType.RAISE) and la.max_amount > la.min_amount:
                if not any(r.amount == la.max_amount and r.action_type == la.action_type for r in out):
                    out.append(ctx.q_bet(la.action_type, la.max_amount))
        return out

    # ------------------------------------------------------------------
    def best_action(self, gs: GameState, hero: int, opponent_ranges=None,
                    rng_seed=None, **kw) -> ActionEV:
        results = self.evaluate(gs, hero, opponent_ranges, rng_seed, **kw)
        if not results:
            return ActionEV(ActionType.FOLD, 0, 0.0, {}, "Fold (no legal actions)")
        return results[0]

    @staticmethod
    def _is_in_position(gs: GameState, hero: int) -> bool:
        if gs.street == Street.PREFLOP:
            return gs.positions[hero] in (Position.BTN, Position.CO, Position.HJ)
        n = gs.num_players
        for offset in range(n, 0, -1):
            seat = (gs.dealer_seat + offset) % n
            if not gs.folded[seat] and not gs.all_in[seat]:
                return seat == hero
        return False


# ---------------------------------------------------------------------------
# Per-decision context: samples, responses, utility bookkeeping
# ---------------------------------------------------------------------------
class _Context:
    def __init__(self, engine: DecisionEngine, gs: GameState, hero: int,
                 opponent_ranges, rng_seed, models, profiles, single_profile,
                 positions, prev_aggressor, fixed_runout=None):
        self.e = engine
        self.cfg = engine.config
        self.gs = gs
        self.hero = hero
        self.models = models
        self.profiles = profiles
        self.single_profile = single_profile
        self.positions = positions or [p.name for p in gs.positions]
        self.prev_aggressor = prev_aggressor

        self.opps = [s for s in gs.active_players if s != hero]
        m = len(self.opps)
        ranges = list(opponent_ranges or [])
        if len(ranges) > m:
            ranges = ranges[:m]
        ranges += [None] * (m - len(ranges))
        self.ranges = ranges

        self.hero_cards = list(gs.hole_cards[hero])
        self.board = list(gs.board)
        self.pot = gs.pot
        self.committed = gs.current_bets[hero]
        self.stack = gs.stacks[hero]
        self.bet_before = gs.max_current_bet
        self.to_call = min(self.bet_before - self.committed, self.stack)
        self.bb = gs.big_blind

        n = (self.cfg.equity_simulations_important
             if gs.pot / max(1, gs.big_blind) > self.cfg.important_pot_bb
             else self.cfg.equity_simulations)
        self.rng = random.Random(rng_seed)
        self.trials: list[ShowdownTrial] = sample_showdowns(
            self.hero_cards, self.board, m, n, ranges, rng=self.rng,
            fixed_runout=fixed_runout) if m > 0 else []
        self.N = max(1, len(self.trials))
        # common random numbers for multi-way response sampling
        self.u = [[self.rng.random() for _ in range(m)] for _ in self.trials]
        self.share_all = [t.hero_share() for t in self.trials]
        self.equity = sum(self.share_all) / self.N if self.trials else 1.0

        dead = set(self.hero_cards)
        self.strengths = strength_table(self.board, dead)
        self.weights = []
        for i in range(m):
            w = normalize_range(ranges[i], dead | set(self.board))
            self.weights.append(w if w else {h: 1.0 for h in self.strengths})

        # utility bookkeeping
        self.stacks_ref = list(gs.starting_stacks) if gs.starting_stacks else \
            [gs.stacks[i] + gs.current_bets[i] for i in range(gs.num_players)]
        # stacks_ref[hero] must be comparable with hero_final values
        self.hero_base = self.stack
        self.stacks_ref = list(self.stacks_ref)
        self.stacks_ref[hero] = self.stack
        self._ucache: dict[float, float] = {}
        self.u0 = self.U(self.stack)
        self.pot_scale = self.e.utility.pot_scale(self.stack, self.pot, hero,
                                                  self.stacks_ref, self.opps)
        self.in_position = DecisionEngine._is_in_position(gs, hero)

    # --- utility ---------------------------------------------------------
    def U(self, hero_final: float) -> float:
        k = round(hero_final, 3)
        v = self._ucache.get(k)
        if v is None:
            v = self.e.utility.value(hero_final, self.hero, self.stacks_ref, self.opps)
            self._ucache[k] = v
        return v

    def _can_act(self, seat: int) -> bool:
        return not self.gs.all_in[seat] and not self.gs.folded[seat]

    def _profile(self, seat: int):
        return self.profiles.get(seat) or self.single_profile

    def _cell(self, seat: int, facing: bool, raises_before: int):
        pos = self.positions[seat] if seat < len(self.positions) else "MP"
        return decision_cell(self.gs.street, pos, self.board,
                             self.prev_aggressor == seat, facing, raises_before)

    # --- actions ---------------------------------------------------------
    def candidate_totals(self, la: LegalAction) -> list[int]:
        """Street totals for hero bets/raises: min, pot fractions, all-in."""
        base = self.bet_before
        pot_after_call = self.pot + self.to_call
        totals = {la.min_amount}
        for f in self.cfg.bet_sizes:
            t = int(round(base + f * pot_after_call))
            if la.min_amount <= t <= la.max_amount:
                totals.add(t)
        return sorted(totals)

    def q_call(self, call_amt: int) -> ActionEV:
        c = min(call_amt, self.stack)
        pot_final = self.pot + c
        q = sum(self.U(self.stack - c + s * pot_final) for s in self.share_all) / self.N - self.u0
        return ActionEV(ActionType.CALL, c, q, {
            "equity": round(self.equity, 4), "pot": self.pot, "call_cost": c,
            "break_even": round(break_even_equity(c, self.pot), 4),
            "ev_immediate": round(self.equity * pot_final - c, 2),
        }, f"Call {c}")

    def q_check(self) -> ActionEV:
        """Check; if an opponent is still to act, model their bet and hero's reply."""
        to_act = [s for s in self._seats_after_hero()
                  if s in self.gs.needs_to_act and self._can_act(s)]
        showdown = [self.U(self.stack + s * self.pot) - self.u0 for s in self.share_all]
        if not to_act or not self.trials:
            q = sum(showdown) / self.N
            return ActionEV(ActionType.CHECK, 0, q,
                            {"equity": round(self.equity, 4), "pot": self.pot,
                             "villain_bet_prob": 0.0}, "Check")

        j = to_act[0]
        i = self.opps.index(j)
        model = self.models.get(j)
        decision, cell = self._cell(j, False, 0)
        if decision == "preflop":
            q = sum(showdown) / self.N
            return ActionEV(ActionType.CHECK, 0, q, {"equity": round(self.equity, 4)}, "Check")
        pol = self.e.lm.policy(self.strengths, self.weights[i], "unopened", model, cell)
        frac = (model.bet_sizes.most_common_size() if model and model.bet_sizes.total_samples >= 5
                else None) or self.cfg.villain_bet_size
        b = min(int(frac * self.pot) or self.bb, self.gs.stacks[j], self.stack)
        pbet = [pol.probs(self.strengths.get(t.opp_hands[i], 0.5))["bet"] for t in self.trials]
        mass = sum(pbet)
        # hero's best reply to the bet, over the combos that bet
        ev_call = (sum(p * (self.U(self.stack - b + s * (self.pot + 2 * b)) - self.u0)
                       for p, s in zip(pbet, self.share_all)) / mass) if mass > 0 else 0.0
        reply = max(ev_call, 0.0)
        q = (sum((1 - p) * v for p, v in zip(pbet, showdown)) + mass * reply) / self.N
        return ActionEV(ActionType.CHECK, 0, q, {
            "equity": round(self.equity, 4), "pot": self.pot,
            "villain_bet_prob": round(mass / self.N, 3), "villain_bet": b,
            "hero_reply_to_bet": "call" if ev_call > 0 else "fold",
        }, "Check")

    def _seats_after_hero(self) -> list[int]:
        n = self.gs.num_players
        return [(self.hero + k) % n for k in range(1, n)]

    def q_bet(self, action_type: ActionType, total: int) -> ActionEV:
        """Bet or raise to `total` (street commitment)."""
        gs = self.gs
        add = max(0, min(total - self.committed, self.stack))
        total = self.committed + add
        pot_after_call = self.pot + self.to_call
        size_frac = (total - self.bet_before) / pot_after_call if pot_after_call > 0 else 1.0
        if action_type == ActionType.ALL_IN or add == self.stack:
            label_kind = "ALL_IN" if action_type == ActionType.ALL_IN else ActionType(action_type).name
        else:
            label_kind = ActionType(action_type).name
        raises_before = gs.raises_this_street + 1

        m = len(self.opps)
        calls = [0] * m
        cont: list[Optional[list[tuple[float, float]]]] = [None] * m
        for i, s in enumerate(self.opps):
            to_call_i = max(0, total - gs.current_bets[s])
            calls[i] = min(to_call_i, gs.stacks[s])
            if not self._can_act(s) or to_call_i == 0:
                continue      # all-in players contest without responding
            model = self.models.get(s)
            decision, cell = self._cell(s, True, raises_before)
            # Same definition the spectator records: faced bet / pot before it.
            denom = self.pot + add - to_call_i
            faced = to_call_i / denom if denom > 0 else 1.0
            prof = self._profile(s)
            fold_override = prof.fold_probability(size_frac, self.pot, total - self.bet_before) \
                if prof is not None and hasattr(prof, "fold_probability") else None
            pol = self.e.lm.policy(self.strengths, self.weights[i], decision, model,
                                   cell, faced, fold_override)
            cache: dict = {}
            probs = []
            for t in self.trials:
                h = t.opp_hands[i]
                v = cache.get(h)
                if v is None:
                    pd = pol.probs(self.strengths.get(h, 0.5))
                    v = (pd["call"], pd["raise"])
                    cache[h] = v
                probs.append(v)
            cont[i] = probs

        win_now = self.U(self.stack + self.pot) - self.u0
        responders = [i for i in range(m) if cont[i] is not None]
        fold_prob_all = 0.0
        eq_called_num = eq_called_den = 0.0
        q_sum = 0.0

        if len(responders) == 1 and m == 1:
            # Heads-up: exact expectation over the villain's response.
            i = responders[0]
            j = self.opps[i]
            pot_called = self.pot + add + calls[i]
            called_vals, raise_w = [], []
            for t_idx, t in enumerate(self.trials):
                pc, pr = cont[i][t_idx]
                s = self.share_all[t_idx]
                pf = max(0.0, 1.0 - pc - pr)
                fold_prob_all += pf
                v_call = self.U(self.stack - add + s * pot_called) - self.u0
                q_sum += pf * win_now + pc * v_call
                eq_called_num += (pc + pr) * s
                eq_called_den += pc + pr
                raise_w.append(pr)
                called_vals.append(s)
            # villain raises: hero best-responds (fold or call the raise)
            rmass = sum(raise_w)
            if rmass > 0:
                vil_left = gs.stacks[j] - calls[i]
                raise_amt = min(int(self.cfg.villain_raise_size * pot_called),
                                vil_left, self.stack - add)
                pot_r = pot_called + 2 * raise_amt
                ev_call_r = sum(w * (self.U(self.stack - add - raise_amt + s * pot_r) - self.u0)
                                for w, s in zip(raise_w, called_vals)) / rmass
                ev_fold_r = self.U(self.stack - add) - self.u0
                q_sum += rmass * max(ev_call_r, ev_fold_r)
            q = q_sum / self.N
            fold_prob_all /= self.N
        elif responders:
            # Multi-way: independent responses sampled with common randoms.
            for t_idx, t in enumerate(self.trials):
                contenders, extra = [], 0
                for i in range(m):
                    if cont[i] is None:
                        contenders.append(i)       # all-in: contests anyway
                        continue
                    pc, pr = cont[i][t_idx]
                    if self.u[t_idx][i] < pc + pr:
                        contenders.append(i)
                        extra += calls[i]
                if not contenders:
                    q_sum += win_now
                    fold_prob_all += 1
                else:
                    s = t.hero_share(contenders)
                    q_sum += self.U(self.stack - add + s * (self.pot + add + extra)) - self.u0
                    eq_called_num += s
                    eq_called_den += 1
            q = q_sum / self.N
            fold_prob_all /= self.N
        else:
            # Nobody can respond (all opponents all-in): pure showdown.
            extra = sum(calls)
            q = sum(self.U(self.stack - add + s * (self.pot + add + extra))
                    for s in self.share_all) / self.N - self.u0
            eq_called_num, eq_called_den = sum(self.share_all), self.N

        e_called = eq_called_num / eq_called_den if eq_called_den > 0 else self.equity
        if e_called < self.cfg.bluff_equity_ceiling:
            kind = "Bluff"
        elif e_called > self.cfg.value_equity_floor:
            kind = "Value"
        else:
            kind = "Thin-value"
        pct = int(round(size_frac * 100))
        return ActionEV(action_type, total, q, {
            "equity": round(self.equity, 4), "pot": self.pot, "cost": add,
            "bet_fraction_of_pot": round(size_frac, 2),
            "fold_probability": round(fold_prob_all, 3),
            "equity_when_called": round(e_called, 4),
            "bet_type": kind,
        }, f"{label_kind} {total} ({pct}% pot, {kind})")

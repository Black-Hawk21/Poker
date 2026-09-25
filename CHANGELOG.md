# Changelog — Revision to match "Design of an Adaptive Adversarial Poker Bot (Revised Edition)"

This revision reworks the implementation to match the corrected design
document. Each entry cites the section it implements. All 267 tests pass
(`python run_tests.py`).

## Correctness bugs fixed (these produced wrong numbers or leaked information)

1. **Multi-way ties split 1/k, not 1/2** (§3.1). `monte_carlo_equity` and the
   engine now credit a k-way chop with 1/k of the pot. A 3-way chop returned
   0.5 equity before; it now returns 0.333.
2. **Card removal / joint dealing** (§3.2). Every Monte-Carlo trial deals all
   opponents and the runout from the remaining deck; ranges are card-consistent
   combos, so blocked combos have probability exactly zero and no card is dealt
   twice. Multiple ranged opponents are sampled jointly (rejection sampling),
   not by sequential filtering, which had biased equity toward opponent 1.
3. **Full range reached the equity engine.** `LiveRange.to_combo_list` returned
   the ~200 heaviest combos before, silently turning every range into its
   strongest ~5%. The full weighted posterior is now passed through.
4. **Showdown win rate** (stats). Losers at showdown were classified as folds,
   so showdown win rates were ~100%. The runner now passes the exact set of
   showdown participants.
5. **Bots can no longer read opponents' hole cards or mutate the game.** Each
   bot receives `GameState.view_for(seat)` with other hands hidden (§23.1).
6. **Tournament mode carries stacks over** and ends when one player remains;
   busted seats sit out. `reset_stacks_each_hand=False` was a no-op before.
7. **Side pots.** All-in pots are split into main/side layers by contribution;
   the old code awarded the whole pot to the best hand regardless of coverage.
8. **All-in run-out uses the same burn/deal order as normal streets**, so a
   correct RNG seed is no longer eliminated by hands that go all-in (§23.4).
9. **Observers see pre-action context.** Actions now record what the player
   faced before acting (to-call, pot-before, raises-before). Previously every
   preflop open-raise was logged as a 3-bet, calls as "not facing a bet", and
   bet sizes were divided by the wrong pot.
10. **`Deck.remaining`** is correct after `remove()`; **incomplete all-in
    raises** no longer reopen betting for players who already acted.

## Poker / probability content brought in line with the revision

- **Equity precision** (§3.2): standard error √(Ê(1−Ê)/N) reported with every
  estimate. Fast direct 7-card evaluator (26× faster, verified identical to the
  brute-force reference on 60k random hands) to afford richer range work.
- **Fold equity** (§5): `bet_ev` implements Eq. 12 with EV-when-called explicit,
  so a semi-bluff and a stone bluff use one formula.
- **MDF and balanced bluffing** (§6): `minimum_defense_frequency`,
  `balanced_bluff_fraction`, and the balanced fold reference 1−MDF = B/(P+B).
- **Mixed strategy** (§7): the softmax-on-raw-chips policy with a 2% floor on
  every action (which could fold the nuts) is replaced by pot-normalized
  randomization over only near-optimal actions, plus **regret matching**
  (Eq. 18) at balance nodes.
- **SPR / stacks** (§8): SPR uses the effective (coverage-limited) stack.
- **Opponent modeling** (§10): backoff-smoothed conditional statistics
  (Eq. 23); fold-to-bet by size bucket for the MDF comparison.
- **Bayesian range inference** (§11) driven by an explicit
  **action-likelihood model** (§12), calibrated per opponent and per spot,
  with a size-given-strength term so rigid sizers leak their holdings.
- **Online Bayesian estimation** (§16): Beta posteriors report variance and
  exact credible intervals; exploitation is scaled by interval width.
- **Empirical Bayes** (§17, §19): the population prior is estimated by method
  of moments (α=κμ, β=κ(1−μ)); near-duplicate opponents are partially pooled.
- **Drift** (§18): Jensen–Shannon divergence (bounded, symmetric) plus a CUSUM
  changepoint detector that discounts stale evidence. KL is no longer used.
- **Decision engine** (§20): Q(s,a)=Σ_r P(r|a,s,M)·E[U(outcome)] − U(now). Fold
  equity is the fold branch (no double counting); variance is handled by the
  utility (chip / concave / **ICM**, §8), not an ad-hoc risk penalty.
- **Exploitation** (§21–§22): the concrete MDF loop — measure q̂(B), compare to
  1−MDF(B), deviate scaled by confidence with hysteresis, randomize residuals.
- **Seed recovery beyond the initial window** (§23.3–§23.4): when no seed in
  the enumerated 0–999 window reproduces the observed cards, an expanded
  search widens the seed range and pins the seed using *every* card the bot
  may see — hero hole cards, the board, and any opponent hand shown at
  showdown. One fully shown hand fixes 9 of 52 deck positions, so the
  surviving seed is effectively unique; the search is time-boxed and resumes
  across hands. The literal 624-word MT19937 untemper is also implemented
  (`MT19937Recovery` / `recover_from_raw_outputs`) for arenas that leak whole
  32-bit words; it does **not** apply to `random.shuffle` dealing, which only
  ever exposes ≤6-bit words per card — a genuine property of the shuffle,
  verified by a test.
- **Environment / RNG** (§23–§24): MT19937 state-recovery scaffolding and a
  call-count offset search (§23.3–§23.4); belief over candidates predicted by
  marginalization (§23.5); the predicted board enters the engine only as a
  confidence-weighted feature, never a hard override, and opponent hole cards
  are never used to collapse ranges (§23.6, §24).
- **Spectator learning** (§15): a revealed hand is replayed street by street to
  feed size-given-strength and threshold-consistency; a bluff is a bet with a
  bottom-half holding, not merely "was aggressive and lost".
- **Calibration** (§28): reliability curve / ECE tracker; `AdaptiveBot` added
  as the drift/changepoint test opponent named in the Testing table.

## Notes

- `equity.monte_carlo_equity` keeps its signature; ranges may now also be
  `(combo, weight)` pairs or `{combo: weight}` dicts.
- `ExploitProfile` exposes the MDF deviations directly; the old heuristic
  fields (`bluff_equity_ceiling`, `hero_call_boost`, …) are gone, with a few
  read-only compatibility shims.
- Tests were updated where they asserted the old (incorrect) behavior — a
  mis-dealt "TPTK" hero in phase 2, and phase-5 assertions on removed
  heuristic knobs — with comments explaining each change.

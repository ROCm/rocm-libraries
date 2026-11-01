# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Search strategies for the monokernel's knob spaces.

:class:`SignedLookback` is the case-study method of the Qwen3.8 work (DESIGN
9.2, ``climb_b3.py --lookback signed``) as an ask/tell strategy of the rocKE
search engine, so it runs alone or as a member of the engine's Thompson
portfolio and shares every observation with the other members:

1. **Climb**: one lever at a time on the current best body. Every other level
   of the lever is an arm; the arms are asked together (one interleaved
   batch). The best arm that beats the current best beyond the noise
   threshold is APPLIED; the rest stay RETAINED-OFF knobs. Every arm adds a
   *signed gain* ``1 - value / best`` to the ledger of its (lever, level).
2. **Structural re-sweep**: when an applied lever is ``structural`` (the base
   changed: NUMA mailboxes, a split, a hop mode), the climb runs again over
   the retained levers on the new base (bounded by ``passes``).
3. **Combination lookback** around the climb's best (PLAN decision 13: a
   lever that loses alone can win in combination, so NOTHING is pruned from
   solo results). The pool is every knob and every level of the space, solo
   losers included; the combination space is explored model-guided, never
   as a capped grid: (a) a pairwise covering screen over the whole pool
   (greedy random combinations that cover the most uncovered pairs of
   changed levels); (b) an interaction-aware GP on log time over every
   measured config (kernel with main, pairwise and three-way interaction
   terms, scales chosen by marginal likelihood; prior mean from the solo
   gains and recorded pair terms); (c) expected improvement over samples of
   the full product space plus every 1-, 2- and 3-change move around the
   best measured combinations and their crossovers (densification), best EI
   plus the most uncertain points per round, until no combination is
   expected to beat the incumbent by the gain floor. When the lookback moves
   the base, the climber climbs again. The ledger keeps solo gains, measured
   combinations and fitted pairwise interaction terms
   (:meth:`SignedLookback.interactions`, :meth:`SignedLookback.combos`) for
   later lookbacks.

The driver's confirmation (``run_search(confirm=k)``) re-times the finalists
together with the baseline, interleaved, which removes the winner's curse of
one noisy batch; :mod:`.gate` applies the DESIGN 9.1 win rule on top.

Also here: :func:`mono_portfolio` (signed lookback + hill climbing + SHADE DE
+ TPE / GP-EI under the engine's discounted Thompson bandit), the
:class:`GainPrior` (an additive model from recorded signed gains, the "table
prior" of a lever ledger) and :func:`transfer_warm_starts` (nearest-variant
warm starts from a tuning DB).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence

import numpy as np

from rocke.dispatch.tuning.search.api import make_strategy
from rocke.dispatch.tuning.search.params import Genome, SearchSpace
from rocke.dispatch.tuning.search.portfolio import Portfolio, WarmStart
from rocke.dispatch.tuning.search.strategy import Strategy
from rocke.dispatch.tuning.search.trial import Observation, Trial

__all__ = [
    "GainPrior",
    "Phased",
    "SignedLookback",
    "default_strategy",
    "phased",
    "mono_portfolio",
    "transfer_warm_starts",
]

_CLIMB, _LOOKBACK, _DONE = "climb", "lookback", "done"


class SignedLookback(Strategy):
    """Hill climbing one lever at a time, then a model-guided combination
    lookback over every knob and level (no solo pruning).

    ``start``: the genome to climb from (default: the space's base genome,
    i.e. the shipped build). ``order``: lever (param) names in climb order
    (default: by best prior signed gain, then space order). ``structural``:
    param names whose application re-opens the climb (``passes`` bounds the
    passes). ``prior_gains``: ``{(param, level label): [signed gains]}`` from
    earlier climbs of the same variant (climb order and the GP's prior mean
    only; never a filter). ``prior_pairs``: ``{"a=x|b=y": [log-time
    interaction terms]}`` from earlier lookbacks (GP prior mean).
    ``min_gain``: relative improvement floor of an APPLY and of the EI stop
    (the engine's noise model raises it to twice the learned per-call
    noise). ``lookback_batch``: points per lookback round. ``screen_size``:
    the covering screen (default 1.5 x the knob count in 6..24; at most a
    quarter of ``budget`` when given). ``max_rounds``: acquisition rounds
    before handing over. ``stop_margin``: kept for the state format."""

    name = "lookback"

    def __init__(
        self,
        space: SearchSpace,
        seed: int = 0,
        *,
        start: Sequence[int] | None = None,
        order: Sequence[str] | None = None,
        structural: Sequence[str] = (),
        prior_gains: Mapping[tuple[str, str], Sequence[float]] | None = None,
        prior_pairs: Mapping[str, Sequence[float]] | None = None,
        min_gain: float = 0.002,
        lookback_batch: int = 8,
        stop_margin: float = 0.004,
        screen_size: int | None = None,
        screen_changes: int | None = 3,
        screen_factor: float = 1.0,
        budget: int | None = None,
        max_rounds: int = 16,
        passes: int = 2,
        ridge: float = 0.3,
        repair_arms: bool = True,
        reclimb: int = 2,
        retest: bool = True,
        revisit: bool = True,
        order_restarts: int = 1,
        lookback: bool = True,
        handoff_at: float = 0.0,
        retest_screen_share: float = 0.0,
        solo_combo_screen: bool = False,
    ):
        super().__init__(space, seed)
        self.start = None if start is None else list(space.check(start))
        self.order_names = list(order) if order else None
        self.structural = set(structural)
        self.prior = {tuple(k): list(v) for k, v in (prior_gains or {}).items()}
        self.prior_pairs = {str(k): list(v) for k, v in (prior_pairs or {}).items()}
        self.min_gain = float(min_gain)
        self.screen_size = None if screen_size is None else int(screen_size)
        #: levers changed per screen point (None: each lever with probability
        #: 0.2-0.5); a few changes per point still cover every pair of levels
        #: (losers included) while staying near the climb's best
        self.screen_changes = None if screen_changes is None else int(screen_changes)
        self.screen_factor = float(screen_factor)
        self.budget = None if budget is None else int(budget)
        self.max_rounds = int(max_rounds)
        self.lookback_batch = int(lookback_batch)
        self.stop_margin = float(stop_margin)
        self.passes = int(passes)
        self.ridge = float(ridge)
        self.repair_arms = bool(repair_arms)
        #: climb again from the lookback's pick when the lookback moved the
        #: base (the base changed, so every retained lever is re-swept), at
        #: most this many times
        self.reclimb = int(reclimb)
        self.reclimbs = 0
        #: PLAN decision 15: a verdict holds only on the base it was measured
        #: on. After every accepted move every rejected level is re-tested on
        #: the new base (``retest``), the lookback ends with a "revisit
        #: rejects" pass (``revisit``) and the climb is repeated from the start
        #: in ``order_restarts`` other lever orders.
        self.retest = bool(retest)
        self.do_lookback = bool(lookback)  # False: a pure climb (re-tests, revisit, restarts still apply)
        self.revisit = bool(revisit)
        self.order_restarts = int(order_restarts)
        #: share of a known budget after which the re-tests the model does not
        #: put in front are handed to the lookback (its expected-improvement
        #: pool, optionally forced screen points) instead of being queued as
        #: climb arms ahead of it; 0 (default): from the start. Replays: forcing
        #: them as climb arms or screen points cost the covering screen
        #: evaluations the GP needed (NUMA grids at 32: hit 0.30 vs 0.53)
        self.handoff_at = float(handoff_at)
        #: share of the lookback screen the handed-over re-tests may take as
        #: forced points (0: none; they are always in the expected-improvement
        #: candidate pool, every single change around the best combinations,
        #: and the revisit pass re-tests what is still open)
        self.retest_screen_share = float(retest_screen_share)
        #: measure the main-effects pick (start + every lever at its best solo
        #: level) as the lookback's first screen point. Off by default: in the
        #: replays it helped at 16 evaluations and cost hits at 32-48 (a noisy
        #: lucky measurement of it can become the incumbent); a retained knob
        self.solo_combo_screen = bool(solo_combo_screen)
        self.restarts_done = 0
        self.revisited = False
        self.verdicts: list[dict] = []  # every verdict, tagged with its base
        self.rejected: dict[str, str | None] = {}  # "i:lv" -> base key it lost (or tied) on
        self.moves: list[dict] = []  # accepted moves, in order
        self.retests = 0
        #: rejects handed to the lookback screen (re-tested there on its
        #: center, model order) once ``handoff_at`` of a known budget is spent
        self.screen_retests: list[list[int]] = []
        #: lever -> the only levels its next climb arm set tests (a re-test of
        #: rejected levels; absent: every level)
        self.lever_levels: dict[int, list[int]] = {}
        self._lookback_from: str | None = None
        self.phase = _CLIMB
        self.best_g: list[int] | None = None
        self.best_key: str | None = None
        self.queue_levers: list[int] = []
        self.pass_no = 0
        self.struct_changed = False
        self.arms: dict[str, tuple[int, int]] = {}  # key -> (param, level) of the open lever
        self.arm_pending: list[list[int]] = []
        self.open_lever: int | None = None
        self.gains: dict[str, list[float]] = {}  # "param=label" -> signed gains of this climb
        self.ledger: list[dict] = []
        self.applied: list[str] = []
        self.center: list[int] | None = None  # the lookback's base (the climb's best)
        self.start_g: list[int] | None = None  # the climb's start
        self.start_key: str | None = None
        #: signed gains of the first climb pass: every lever tried ALONE on the
        #: climb's current base (the case study's round 1 / 2): the solo
        #: picture, kept apart from later passes and lookback combinations
        self.solo: dict[str, list[float]] = {}
        self._screen_queue: list[list[int]] = []
        self.screened = 0
        self.lookback_rounds = 0
        self._theta = None
        # levers with no legal arm at the base they were tried on: re-opened
        # once the base has moved (a lever may become legal on the new base)
        self.skipped: list[tuple[int, str | None]] = []

    # ------------------------------------------------------------ helpers
    def _label(self, i: int, level: int) -> str:
        return f"{self.space.params[i].name}={self.space.params[i].label(level)}"

    def _all_gains(self, i: int, level: int) -> list[float]:
        p = self.space.params[i]
        own = self.gains.get(self._label(i, level), [])
        return own + list(self.prior.get((p.name, p.label(level)), []))

    def _lever_order(self) -> list[int]:
        idx = list(range(self.space.dims))
        if self.order_names:
            pos = {n: k for k, n in enumerate(self.order_names)}
            idx = [i for i in idx if self.space.params[i].name in pos]
            idx.sort(key=lambda i: pos[self.space.params[i].name])
            return idx

        def promise(i):
            best = -math.inf
            for lv in range(self.space.params[i].size):
                g = self.prior.get((self.space.params[i].name, self.space.params[i].label(lv)))
                if g:
                    best = max(best, sum(g) / len(g))
            return best

        # most promising (by prior signed gain) first; unknown levers in space
        # order, structural ones last (the case-study rounds: levers on the
        # base first, the structural lever then becomes the base and the
        # retained levers are re-swept on it)
        known = [i for i in idx if promise(i) > -math.inf]
        unknown = [i for i in idx if promise(i) == -math.inf]
        known.sort(key=lambda i: -promise(i))
        unknown.sort(key=lambda i: (self.space.params[i].name in self.structural, i))
        return known + unknown

    def _value(self, key: str | None) -> float:
        return math.inf if key is None else self.history.get(key, math.inf)

    def _ensure_start(self) -> bool:
        if self.best_g is not None:
            return True
        g = self.start if self.start is not None else list(self.space.encode({}))
        r = self.space.repair(g, self.rng)
        if r is None:
            self.phase = _DONE
            return False
        self.best_g = list(r)
        self.start_g = list(r)
        self.best_key = self.space.key(r)
        self.start_key = self.best_key
        self.queue_levers = self._lever_order()
        self.pass_no = 1
        return True

    # ------------------------------------------------------------- climb
    def _open_next_lever(self) -> bool:
        while self.queue_levers:
            i = self.queue_levers.pop(0)
            arms = []
            levels = range(self.space.params[i].size)
            only = self.lever_levels.pop(i, None)
            if only is not None:
                levels = [lv for lv in only if lv < self.space.params[i].size]
            elif self._seq_tuples(i) is not None:
                levels = self._order_moves(i, self.best_g[i])  # an order knob: one move at a time
            for lv in levels:
                if lv == self.best_g[i]:
                    continue
                g = list(self.best_g)
                g[i] = lv
                if not self.space.active(g)[i]:
                    continue
                k = self.space.key(g)
                if k is None and self.repair_arms:
                    # a companion move: the nearest legal config that keeps
                    # this lever's level (e.g. m4_interleave=True brings
                    # m4_guard=none; in a recorded table, the nearest member)
                    r = self._companion(g, i)
                    if r is not None:
                        g, k = list(r), self.space.key(r)
                if k is None or k == self.best_key or any(k == a[0] for a in arms):
                    continue
                arms.append((k, g, lv))
            if not arms:
                self.skipped.append((i, self.best_key))
                continue
            self.open_lever = i
            self.arms = {k: (i, lv) for k, _g, lv in arms}
            self.arm_pending = [g for k, g, _lv in arms if k not in self.history]
            return True
        return False

    def _companion(self, g: list[int], i: int) -> Genome | None:
        """The nearest legal genome with ``g[i]`` kept: single changes of the
        other knobs first (closest level first), then pairs; ties broken by
        the base's value order (deterministic)."""
        sizes = self.space.sizes
        others = [j for j in range(self.space.dims) if j != i]

        def levels(j):
            return sorted(range(sizes[j]), key=lambda lv: (abs(lv - g[j]), lv))

        for j in others:
            for lv in levels(j):
                if lv == g[j]:
                    continue
                h = list(g)
                h[j] = lv
                if self.space.key(h) is not None:
                    return tuple(h)
        for a, b in itertools.combinations(others, 2):
            for la in levels(a)[1:]:
                for lb in levels(b)[1:]:
                    h = list(g)
                    h[a], h[b] = la, lb
                    if self.space.key(h) is not None:
                        return tuple(h)
        return None

    def _base_labels(self, g: Sequence[int] | None) -> list[str]:
        """The base a verdict was measured on, as its changes from the
        climb's start."""
        if g is None or self.start_g is None:
            return []
        act = self.space.active(g)
        return sorted(self._label(i, int(g[i])) for i in range(self.space.dims) if act[i] and g[i] != self.start_g[i])

    def _rank_rejects(self, levers: Sequence[int], scores: dict | None = None,
                      level_scores: dict | None = None) -> list[int]:  # fmt: skip
        """``levers`` ordered by the interaction model: the best predicted
        log time of any of the lever's levels on the CURRENT base first (the
        GP over every measurement, pairwise / three-way terms included);
        space order when there is too little data for a model. ``scores``
        (out): lever -> predicted log time of its best level;
        ``level_scores`` (out): lever -> {level: predicted log time}."""
        levers = list(dict.fromkeys(levers))
        if not levers or sum(math.isfinite(v) for v in self.history.values()) < 4:
            return levers
        saved = self.center
        if self.center is None:
            self.center = list(self.best_g)
        try:
            predict, _best = self._fit()
            if predict is None:
                return levers
            score = {}
            for i in levers:
                gs, lvs = [], []
                for lv in range(self.space.params[i].size):
                    if lv == self.best_g[i]:
                        continue
                    g = list(self.best_g)
                    g[i] = lv
                    gs.append(self._features_of(g))
                    lvs.append(lv)
                if gs:
                    mu, _sd = predict(gs)
                    score[i] = float(np.min(mu))
                    if level_scores is not None:
                        level_scores[i] = {lv: float(m) for lv, m in zip(lvs, mu)}
            if scores is not None:
                scores.update(score)
            return sorted(levers, key=lambda i: (score.get(i, math.inf), i))
        finally:
            self.center = saved

    def _expected_first_gain(self) -> float:
        """The mean best first-test gain of the levers tested so far (a
        lever's verdicts on the first base it was tested on; losses count as
        zero): what a lever not yet tested is expected to bring."""
        first: dict[str, tuple[str, float]] = {}
        for vd in self.verdicts:
            b, g = first.get(vd["lever"], (vd["base"], 0.0))
            if vd["base"] == b:
                first[vd["lever"]] = (b, max(g, float(vd["gain"])))
        return sum(g for _b, g in first.values()) / len(first) if first else 0.0

    def _accepted(self, what: str, *, skip_lever: int | None = None) -> None:
        """An accepted move: every level rejected on another base is queued
        for a re-test on the new base (PLAN decision 15). The interaction
        model decides when, never whether: the levels it predicts to beat the
        new base by the gain floor (and, while levers of this pass are still
        untested, by the mean first-test gain such a lever is expected to
        bring) go to the FRONT of the queue, only those levels re-tested now;
        every other reject is re-queued after the untested levers (or into the
        lookback screen late in a known budget). A level not re-tested yet
        stays in ``rejected`` and comes back at the next accepted move and in
        the revisit pass: nothing is dropped."""
        self.moves.append({"move": what, "base": self.best_key, "pass": self.pass_no,
                           "changes": self._base_labels(self.best_g)})  # fmt: skip
        if not self.retest:
            return
        stale = sorted({int(k.split(":")[0]) for k, b in self.rejected.items() if b != self.best_key}
                       - ({skip_lever} if skip_lever is not None else set()))  # fmt: skip
        if not stale:
            return
        scores: dict[int, float] = {}
        lscores: dict[int, dict[int, float]] = {}
        ranked = self._rank_rejects(stale, scores, lscores)
        self.retests += len(ranked)
        v = self._value(self.best_key)
        rest = [i for i in self.queue_levers if i not in ranked]
        # the front gate: a re-test goes ahead of the levers not yet tested in
        # this pass only when the model predicts it beats the new base by more
        # than such a lever is expected to (the mean first-test gain so far)
        need = self.gain_threshold(self.min_gain, 1)
        if rest:
            need = max(need, self._expected_first_gain())
        bar = math.log(v) - math.log1p(need) if math.isfinite(v) and v > 0 else -math.inf
        front = [i for i in ranked if scores.get(i, math.inf) < bar]
        back = [i for i in ranked if i not in front]
        queued = set(self.queue_levers)
        for i in ranked:
            if i in queued:
                continue  # already due for a full test in this pass
            rej = sorted(lv for k, b in self.rejected.items() if b != self.best_key
                         for li, lv in [tuple(int(x) for x in k.split(":"))] if li == i)  # fmt: skip
            if i in front:
                # the model-promising levels now; the lever's other rejects stay
                # open (re-queued at the next accepted move, and in the revisit
                # pass at the latest)
                ls = lscores.get(i, {})
                good = sorted((lv for lv in rej if ls.get(lv, math.inf) < bar), key=lambda lv: (ls[lv], lv))
                self.lever_levels[i] = good or rej
            else:
                self.lever_levels[i] = rej
        if self.budget is not None and self._uid >= self.handoff_at * self.budget and self.do_lookback:
            # late in a known budget: the non-promising rejects are re-tested as
            # the first points of the lookback screen (on its center, model
            # order) instead of as extra climb arms ahead of it; what the screen
            # has no room for stays open (next accepted move, revisit pass)
            handed = []
            for k, b in self.rejected.items():
                i, lv = (int(x) for x in k.split(":"))
                if i in back and b != self.best_key and [i, lv] not in self.screen_retests:
                    handed.append([i, lv])
            handed.sort(key=lambda x: (lscores.get(x[0], {}).get(x[1], math.inf), x[0], x[1]))
            self.screen_retests = sorted(
                self.screen_retests + handed,
                key=lambda x: (lscores.get(x[0], {}).get(x[1], math.inf), x[0], x[1]))  # fmt: skip
            self.queue_levers = front + rest
            self.moves[-1]["retest_screen"] = [self.space.params[i].name for i in back]
            for i in back:
                self.lever_levels.pop(i, None)
            back = []
        else:
            self.queue_levers = front + rest + back
        self.moves[-1]["retest_front"] = [self.space.params[i].name for i in front]
        self.moves[-1]["retest_back"] = [self.space.params[i].name for i in back]

    def set_base(self, genome: Sequence[int], reason: str = "external") -> None:
        """An externally supplied new base (a restructure, another track's
        pick): the climb continues from it, every rejected level is re-tested
        on it first (it is measured first if it is new)."""
        self._ensure_start()
        g = self.space.repair(list(genome), self.rng)
        if g is None:
            raise ValueError("set_base: the space refuses the new base and no legal neighbour exists")
        self.best_g = list(self.space.canonical_genome(g))
        self.best_key = self.space.key(g)
        self.phase = _CLIMB
        self.center = None
        self.open_lever = None
        self.arms = {}
        self.arm_pending = []
        self.pass_no += 1
        self.queue_levers = self._lever_order()
        self._accepted(f"base:{reason}")

    def _seq_tuples(self, i: int) -> list[tuple] | None:
        """The orderings of a sequence param (levels ``"a>b>c"``), else
        None."""
        vals = [lv[0][1] if lv else None for lv in self.space.params[i].levels]
        if not all(isinstance(v, str) and ">" in v for v in vals):
            return None
        tups = [tuple(v.split(">")) for v in vals]
        if len({tuple(sorted(t)) for t in tups}) != 1:
            return None
        return tups

    def _order_moves(self, i: int, cur: int) -> list[int]:
        """Levels of a sequence param one ORDER move away from the current
        ordering (one item moved to another position, which includes every
        adjacent swap): the climb's arms for an order knob."""
        tups = self._seq_tuples(i)
        index = {t: k for k, t in enumerate(tups)}
        c = list(tups[cur])
        out = []
        for a in range(len(c)):
            for b in range(len(c)):
                if a == b:
                    continue
                q = list(c)
                x = q.pop(a)
                q.insert(b, x)
                k = index.get(tuple(q))
                if k is not None and k != cur and k not in out:
                    out.append(k)
        return out

    def _close_lever(self) -> None:
        i = self.open_lever
        base_v = self._value(self.best_key)
        base_key = self.best_key
        base_labels = self._base_labels(self.best_g)
        best_arm = None
        gain_floor = self.gain_threshold(self.min_gain, 1)
        for k, (pi, lv) in self.arms.items():
            v = self._value(k)
            if not math.isfinite(v):
                continue
            if not math.isfinite(base_v):
                # the base itself failed: any working arm is progress
                if best_arm is None or v < best_arm[0]:
                    best_arm = (v, k, lv)
                continue
            g = 1.0 - v / base_v
            self.gains.setdefault(self._label(pi, lv), []).append(g)
            if self.pass_no == 1:
                self.solo.setdefault(self._label(pi, lv), []).append(g)
            self.verdicts.append({"lever": self.space.params[pi].name, "level": self.space.params[pi].label(lv),
                                  "gain": g, "base": base_key, "base_changes": base_labels,
                                  "pass": self.pass_no, "verdict": "rejected"})  # fmt: skip
            self.rejected[f"{pi}:{lv}"] = base_key
            self.ledger.append(
                {"lever": self.space.params[pi].name, "level": self.space.params[pi].label(lv),
                 "key": k, "ratio_vs_best": v / base_v, "pass": self.pass_no, "status": "retained-off"}
            )  # fmt: skip
            if g > gain_floor and (best_arm is None or v < best_arm[0]):
                best_arm = (v, k, lv)
        if best_arm is not None:
            _v, k, lv = best_arm
            self.best_g = list(self.space.canonical_genome(self.genomes.get(k, self.best_g)))
            self.best_g[i] = lv
            self.best_key = k
            name = self.space.params[i].name
            self.applied.append(f"{name}={self.space.params[i].label(lv)}")
            for e in self.ledger:
                if e.get("key") == k and e.get("pass") == self.pass_no:
                    e["status"] = "applied"
            for e in reversed(self.verdicts):
                if e["lever"] == name and e["level"] == self.space.params[i].label(lv) and e["base"] == base_key:
                    e["verdict"] = "applied"
                    break
            self.rejected.pop(f"{i}:{lv}", None)
            if name in self.structural:
                self.struct_changed = True
            self._accepted(f"climb {name}={self.space.params[i].label(lv)}", skip_lever=i)
        self.open_lever = None
        self.arms = {}

    def _climb_ask(self, n: int) -> list[Trial]:
        out: list[Trial] = []
        taken = self.pending_keys()
        while len(out) < n:
            if self.open_lever is not None:
                while self.arm_pending and len(out) < n:
                    g = self.arm_pending.pop(0)
                    k = self.space.key(g)
                    if k in self.history or k in taken:
                        continue
                    taken.add(k)
                    out.append(self.trial(g))
                if self.arm_pending or any(k not in self.history for k in self.arms):
                    break  # the batch is full, or the lever's arms are not all told yet
                self._close_lever()
                continue
            if self._open_next_lever():
                continue
            # pass complete
            retry = sorted({i for i, k in self.skipped if k != self.best_key})
            if retry and self.pass_no < self.passes + 1:
                self.skipped = []
                self.pass_no += 1
                self.queue_levers = retry
                continue
            if self.struct_changed and self.pass_no < self.passes:
                self.struct_changed = False
                self.pass_no += 1
                self.queue_levers = self._lever_order()  # every lever again: no solo pruning
                continue
            self.phase = _LOOKBACK
            break
        return out

    # ---------------------------------------------------------- lookback
    # The lookback never prunes a lever or a level from solo results (PLAN
    # decision 13: ten levers can each lose alone while a combination of them
    # wins by half). Its pool is every knob and every level of the space; the
    # combination space is explored model-guided: a pairwise covering screen,
    # then an interaction-aware GP over all of it, then expected improvement
    # over samples of the full product space densified around the best
    # measured combinations. Only measured combination data moves the model.

    def _features_of(self, g: Sequence[int]) -> frozenset:
        """Changed (param, level) pairs of ``g`` against the lookback center
        (inactive gated genes count as unchanged)."""
        act = self.space.active(g)
        c = self.center
        return frozenset((i, int(g[i])) for i in range(self.space.dims) if act[i] and g[i] != c[i])

    def _kernel(self, fa: Sequence[frozenset], fb: Sequence[frozenset], th: Sequence[float]) -> np.ndarray:
        """ANOVA-style interaction kernel on log time: with ``s`` the number
        of changed (param, level) pairs two configs share, ``k = a0 + a1 s +
        a2 C(s, 2) + a3 C(s, 3)``: main effects, every pairwise and every
        three-way interaction term, each a GP prior with its own scale."""
        a0, a1, a2, a3 = th
        s = np.asarray([[len(x & y) for y in fb] for x in fa], dtype=float)
        return a0 + a1 * s + a2 * s * (s - 1) / 2.0 + a3 * s * (s - 1) * (s - 2) / 6.0

    def _prior_mean(self, f: frozenset) -> float:
        """Prior log-time offset of a combination: every changed level's solo
        effect (its own and recorded signed gains, losers' positive offsets
        included) plus recorded pairwise interaction terms."""
        m = 0.0
        labels = []
        for i, lv in f:
            gs = self._all_gains(i, lv)
            if gs:
                m += math.log(max(1e-3, 1.0 - sum(gs) / len(gs)))
            labels.append(self._label(i, lv))
        for a, b in itertools.combinations(sorted(labels), 2):
            v = self.prior_pairs.get(f"{a}|{b}")
            if v:
                m += sum(v) / len(v)
        return m

    def _fit(self):
        """The GP posterior over every finite observation of the space (climb
        arms, screen, acquisitions, other members' finds): ``(predict,
        log_best)``. Hyperparameters by log marginal likelihood over a small
        grid (interaction order weights and noise)."""
        keys = [k for k, v in self.history.items() if math.isfinite(v) and v > 0 and k in self.genomes]
        if not keys:
            return None, math.inf
        F = [self._features_of(self.genomes[k]) for k in keys]
        y = np.log([self.history[k] for k in keys])
        lb = self._value(self._lookback_from)
        base = math.log(lb) if math.isfinite(lb) else float(np.median(y))
        m0 = np.asarray([base + self._prior_mean(f) for f in F])
        r = y - m0
        sn = max(0.002, (self.noise.median_sigma(3) if self.noise is not None else 0.003))
        best = None
        for a1 in (0.02**2, 0.05**2):
            for ratio2 in (0.5, 1.0):
                for ratio3 in (0.0, 0.5):
                    th = (0.03**2, a1, a1 * ratio2, a1 * ratio3)
                    K = self._kernel(F, F, th) + (sn**2 + 1e-8) * np.eye(len(F))
                    try:
                        L = np.linalg.cholesky(K)
                    except np.linalg.LinAlgError:
                        continue
                    al = np.linalg.solve(L.T, np.linalg.solve(L, r))
                    lml = -0.5 * float(r @ al) - float(np.sum(np.log(np.diag(L))))
                    if best is None or lml > best[0]:
                        best = (lml, th, L, al)
        if best is None:
            return None, float(np.min(y))
        _lml, th, L, al = best
        self._theta = th

        def predict(cands: Sequence[frozenset]):
            Ks = self._kernel(cands, F, th)
            mu = np.asarray([base + self._prior_mean(f) for f in cands]) + Ks @ al
            v = np.linalg.solve(L, Ks.T)
            n_ch = np.asarray([len(f) for f in cands], dtype=float)
            kss = th[0] + th[1] * n_ch + th[2] * n_ch * (n_ch - 1) / 2.0 + th[3] * n_ch * (n_ch - 1) * (n_ch - 2) / 6.0
            var = np.maximum(kss - np.sum(v * v, axis=0), 1e-12)
            return mu, np.sqrt(var)

        return predict, float(np.min(y))

    def _random_combo(self, q: float) -> list[int]:
        g = list(self.center)
        for i in range(self.space.dims):
            if self.rng.random() < q:
                others = [lv for lv in range(self.space.params[i].size) if lv != self.center[i]]
                if others:
                    g[i] = self.rng.choice(others)
        return g

    def _k_combo(self, k: int) -> list[int]:
        """The center with ``k`` random levers moved to a random other level."""
        g = list(self.center)
        movable = [i for i in range(self.space.dims) if self.space.params[i].size > 1]
        for i in self.rng.sample(movable, min(k, len(movable))):
            g[i] = self.rng.choice([lv for lv in range(self.space.params[i].size) if lv != self.center[i]])
        return g

    def _legal(self, g: Sequence[int]) -> Genome | None:
        if self.space.key(g) is not None:
            return tuple(g)
        return self.space.repair(list(g), self.rng)

    def _screen(self, size: int) -> list[Genome]:
        """A pairwise covering design over the whole pool (every knob, every
        level, losers included): greedy picks of random combinations that
        cover the most not-yet-covered pairs of changed (param, level)."""
        covered: set = set()
        for k in self.history:
            if k in self.genomes:
                covered |= set(itertools.combinations(sorted(self._features_of(self.genomes[k])), 2))
        q = min(0.5, max(0.2, 4.0 / max(1, self.space.dims)))
        out: list[Genome] = []
        seen = set(self.history) | self.pending_keys()
        for _ in range(size):
            best = None
            for _t in range(64):
                raw = self._random_combo(q) if self.screen_changes is None else self._k_combo(self.screen_changes)
                g = self._legal(raw)
                if g is None:
                    continue
                k = self.space.key(g)
                if k in seen:
                    continue
                pairs = set(itertools.combinations(sorted(self._features_of(g)), 2))
                gain = len(pairs - covered)
                if best is None or gain > best[0]:
                    best = (gain, g, k, pairs)
            if best is None:
                break
            _gain, g, k, pairs = best
            covered |= pairs
            seen.add(k)
            out.append(g)
        return out

    def _candidates(self, n_random: int = 384) -> list[Genome]:
        """Samples of the full product space plus densification: every
        single, pair and triple change around the best measured combinations,
        and crossovers of them."""
        cands: dict[str, Genome] = {}

        def add(g):
            gl = self._legal(g)
            if gl is None:
                return
            k = self.space.key(gl)
            if k not in self.history and k not in cands:
                cands[k] = gl

        q = min(0.5, max(0.15, 3.0 / max(1, self.space.dims)))
        for _ in range(n_random // 2):
            add(self._random_combo(q))
        for _ in range(n_random // 2):  # near the climb's best: 2-4 levers moved
            add(self._k_combo(self.rng.choice((2, 3, 4))))
        top = sorted((v, k) for k, v in self.history.items() if math.isfinite(v) and k in self.genomes)[:5]
        tops = [list(self.genomes[k]) for _v, k in top]
        for g0 in tops:
            for i in range(self.space.dims):
                for lv in range(self.space.params[i].size):
                    if lv != g0[i]:
                        h = list(g0)
                        h[i] = lv
                        add(h)
            for _ in range(48):  # random 2- and 3-change moves around a good combination
                h = list(g0)
                for i in self.rng.sample(range(self.space.dims), min(self.space.dims, self.rng.choice((2, 3)))):
                    h[i] = self.rng.randrange(self.space.params[i].size)
                add(h)
        for a, b in itertools.combinations(tops, 2):  # crossovers of good combinations
            for _ in range(4):
                add([x if self.rng.random() < 0.5 else y for x, y in zip(a, b)])
        return list(cands.values())

    def _solo_winners_combo(self) -> list[int] | None:
        """The start plus, per lever, its level with the best mean first-pass
        (solo) gain when that gain is positive: the combination a pure
        main-effects model picks (the live runs' "solo-winners-only"
        reference); repaired to the nearest legal config."""
        if self.start_g is None:
            return None
        g = list(self.start_g)
        for i in range(self.space.dims):
            best = None
            for lv in range(self.space.params[i].size):
                gs = self.solo.get(self._label(i, lv))
                if gs and sum(gs) / len(gs) > 0 and (best is None or sum(gs) / len(gs) > best[0]):
                    best = (sum(gs) / len(gs), lv)
            if best is not None:
                g[i] = best[1]
        gl = self._legal(g)
        return None if gl is None else list(gl)

    def _lookback_ask(self, n: int) -> list[Trial]:
        if self.pending:
            return []  # one round at a time
        if not self.do_lookback:
            self._lookback_from = self.best_key
            return self._finish_lookback()
        if self.center is None:
            self._lookback_from = self.best_key
            self.center = list(self.best_g)
            size = self.screen_size or int(min(24, max(4, self.screen_factor * self.space.dims)))
            if self.budget is not None:
                size = min(size, max(4, self.budget // 4))
            first = []
            for i, lv in self.screen_retests:  # handed-over rejects: single changes on the center
                g = list(self.center)
                g[i] = lv
                gl = self._legal(g)
                if gl is not None and self.space.key(gl) not in self.history and list(gl) not in first:
                    first.append(list(gl))
            self.screen_retests = []
            first = first[: int(size * self.retest_screen_share)]
            # the additive model's pick: the start with every lever at its best
            # first-pass (solo) level, when the climb's path did not measure it
            sw = self._solo_winners_combo() if self.solo_combo_screen else None
            if sw is not None and self.space.key(sw) not in self.history and sw not in first:
                first.insert(0, sw)
            self._screen_queue = first + [list(g) for g in self._screen(max(1, size - len(first)))]
            self.screened = len(self._screen_queue)
        out: list[Trial] = []
        taken = self.pending_keys()
        if self._screen_queue:
            while self._screen_queue and len(out) < max(1, n):
                g = self._screen_queue.pop(0)
                k = self.space.key(g)
                if k in self.history or k in taken:
                    continue
                taken.add(k)
                out.append(self.trial(g))
            if out:
                return out
        predict, log_best = self._fit()
        cands = self._candidates()
        if predict is None or not cands:
            return self._finish_lookback()
        F = [self._features_of(g) for g in cands]
        mu, sd = predict(F)
        xi = math.log1p(self.min_gain)
        z = (log_best - xi - mu) / sd
        Phi = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
        phi = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        ei = (log_best - xi - mu) * Phi + sd * phi
        # stop: no combination is still expected to beat the incumbent by the
        # gain floor (after the screen and at least two model rounds)
        if self.lookback_rounds >= 2 and float(np.max(ei)) < 0.1 * xi:
            return self._finish_lookback()
        if self.lookback_rounds >= self.max_rounds:
            return self._finish_lookback()
        m = max(1, min(n, self.lookback_batch))
        order = list(np.argsort(-ei))
        picked = order[: max(1, m - m // 4)]
        explore = [j for j in np.argsort(-sd) if j not in picked][: m - len(picked)]
        for j in list(picked) + list(explore):
            g = cands[int(j)]
            k = self.space.key(g)
            if k in taken:
                continue
            taken.add(k)
            out.append(self.trial(g))
        self.lookback_rounds += 1
        return out

    def interactions(self, top: int = 40) -> dict:
        """Measured pairwise interaction terms (log time; negative = the pair
        is faster together than its two solo effects predict) from a ridge
        fit of main + pair effects over every measured combination: the
        ledger entry later lookbacks warm-start from."""
        keys = [k for k, v in self.history.items() if math.isfinite(v) and v > 0 and k in self.genomes]
        ref = self.start_g or self.center or self.best_g
        if len(keys) < 4 or ref is None:
            return {}
        saved, self.center = self.center, list(ref)  # changes are counted from the climb's start
        try:
            Fs = [sorted(self._features_of(self.genomes[k])) for k in keys]
        finally:
            self.center = saved
        mains = sorted({f for fs in Fs for f in fs})
        pairs = sorted({p for fs in Fs for p in itertools.combinations(fs, 2)})
        cols = {("m",) + f: j for j, f in enumerate(mains)}
        cols.update({("p",) + p: len(mains) + j for j, p in enumerate(pairs)})
        X = np.zeros((len(keys), 1 + len(cols)))
        X[:, 0] = 1.0
        for r, fs in enumerate(Fs):
            for f in fs:
                X[r, 1 + cols[("m",) + f]] = 1.0
            for p in itertools.combinations(fs, 2):
                X[r, 1 + cols[("p",) + p]] = 1.0
        y = np.log([self.history[k] for k in keys])
        lam = np.full(X.shape[1], self.ridge)
        lam[0] = 1e-6
        w = np.linalg.solve(X.T @ X + np.diag(lam), X.T @ y)
        out = {}
        for p in pairs:
            (i, a), (j, b) = p
            out[f"{self._label(i, a)}|{self._label(j, b)}"] = float(w[1 + cols[("p",) + p]])
        return dict(sorted(out.items(), key=lambda kv: kv[1])[:top])

    def combos(self) -> dict:
        """Measured combinations vs the solo picture: the solo losers (levels
        whose solo gain in the first climb pass is negative: they lost when
        tried alone on the climb's base), the best measured
        combination that contains one, and the best that contains only solo
        winners and neutral levels."""
        def solo(i, lv):
            return self.solo.get(self._label(i, lv), [])

        losers = {(i, lv) for i in range(self.space.dims) for lv in range(self.space.params[i].size)
                  if solo(i, lv) and all(x < 0 for x in solo(i, lv))}  # fmt: skip
        start = self.start_g or self.center or self.best_g
        best_with, best_without = None, None
        for k, v in self.history.items():
            if not math.isfinite(v) or k not in self.genomes:
                continue
            g = self.genomes[k]
            act = self.space.active(g)
            ch = {(i, int(g[i])) for i in range(self.space.dims) if act[i] and g[i] != start[i]}
            rec = (v, k, sorted(self._label(i, lv) for i, lv in ch))
            if ch & losers:
                if best_with is None or v < best_with[0]:
                    best_with = rec + (sorted(self._label(i, lv) for i, lv in ch & losers),)
            elif best_without is None or v < best_without[0]:
                best_without = rec
        return {
            "solo_losers": sorted(self._label(i, lv) for i, lv in losers),
            "best_with_solo_loser": None if best_with is None else
            {"value": best_with[0], "changes": best_with[2], "solo_losers_in_it": best_with[3]},
            "best_solo_winners_only": None if best_without is None else
            {"value": best_without[0], "changes": best_without[2]},
        }  # fmt: skip

    def _finish_lookback(self) -> list[Trial]:
        """End of a lookback: climb again from its pick when it moved the
        base, else done."""
        best_k = min((k for k in self.history if math.isfinite(self.history[k])),
                     key=lambda k: (self.history[k], k), default=None)  # fmt: skip
        moved = best_k is not None and best_k != self._lookback_from and self._value(best_k) < self._value(
            self._lookback_from) * (1.0 - self.gain_threshold(self.min_gain, 1))  # fmt: skip
        if moved and self.reclimbs < self.reclimb:
            self.reclimbs += 1
            self.best_key = best_k
            self.best_g = list(self.space.canonical_genome(self.genomes[best_k]))
            self.phase = _CLIMB
            self.center = None
            self.lookback_rounds = 0
            self.pass_no += 1
            self.queue_levers = self._lever_order()
            self.skipped = []
            self._accepted("lookback pick")
            return []
        if best_k is not None and best_k != self.best_key:
            self.best_key = best_k
            self.best_g = list(self.space.canonical_genome(self.genomes[best_k]))
        self.center = None
        self.lookback_rounds = 0
        if self.revisit and not self.revisited:
            # explicit "revisit rejects" pass: every level that was rejected on
            # another base is re-tested on the final base, model-ranked
            self.revisited = True
            stale = sorted({int(k.split(":")[0]) for k, b in self.rejected.items() if b != self.best_key})
            if stale:
                self.phase = _CLIMB
                self.pass_no += 1
                self.queue_levers = self._rank_rejects(stale)
                self.moves.append({"move": "revisit rejects", "base": self.best_key, "pass": self.pass_no,
                                   "levers": [self.space.params[i].name for i in self.queue_levers]})  # fmt: skip
                return []
        budget_left = self.budget is None or self._uid < 0.6 * self.budget
        if self.restarts_done < self.order_restarts and self.start_g is not None and budget_left:
            # climb-order diversity: climb again from the start in another
            # lever order (known configs cost nothing; history is shared)
            self.restarts_done += 1
            self.revisited = False
            order = self._lever_order()
            if self.restarts_done % 2 == 1:
                order = list(reversed(order))
            else:
                self.rng.shuffle(order)
            self.best_g = list(self.start_g)
            self.best_key = self.start_key
            self.phase = _CLIMB
            self.pass_no += 1
            self.queue_levers = order
            self.skipped = []
            self.moves.append({"move": f"order restart {self.restarts_done}", "base": self.best_key,
                               "pass": self.pass_no, "order": [self.space.params[i].name for i in order]})  # fmt: skip
            return []
        self.phase = _DONE
        return []

    # ---------------------------------------------------------- protocol
    def ask(self, n: int = 1) -> list[Trial]:
        if self.phase == _DONE or not self._ensure_start():
            return []
        if self.best_key not in self.history:
            if self.pending:
                return []
            return [self.trial(self.best_g)]
        for _ in range(4):  # phase changes within one ask (climb -> lookback -> climb)
            if self.phase == _CLIMB:
                out = self._climb_ask(n)
                if out or self.phase == _CLIMB:
                    return out
            if self.phase == _LOOKBACK:
                out = self._lookback_ask(n)
                if out or self.phase == _LOOKBACK:
                    return out
            if self.phase == _DONE:
                return []
        return []

    def observe(self, ob: Observation, own: bool) -> None:
        if not ob.ok or not ob.full or ob.key is None or self.best_key is None:
            return
        # another member's (or the lookback's) better find moves the base the
        # next lever climbs on
        if self.phase != _CLIMB and ob.value < self._value(self.best_key) * (
            1.0 - self.gain_threshold(self.min_gain, len(ob.samples))
        ):
            self.best_key = ob.key
            self.best_g = list(self.space.canonical_genome(ob.trial.genome))

    @property
    def done(self) -> bool:
        return self.phase == _DONE

    def report(self) -> dict:
        return {
            "applied": list(self.applied),
            "ledger": list(self.ledger),
            "lookback_pool": {"knobs": self.space.dims, "levels": int(sum(self.space.sizes)),
                              "pruned": 0},
            "lookback_screen": self.screened,
            "lookback_rounds": self.lookback_rounds,
            "gp_theta": None if self._theta is None else [float(x) for x in self._theta],
            "interactions": self.interactions(),
            "combos": self.combos(),
            "solo_gains": {k: list(v) for k, v in self.solo.items()},
            "reclimbs": self.reclimbs,
            "verdicts": list(self.verdicts),
            "moves": list(self.moves),
            "retests": self.retests,
            "order_restarts_done": self.restarts_done,
            "open_rejects": sorted(f"{self.space.params[int(k.split(':')[0])].name}="
                                   f"{self.space.params[int(k.split(':')[0])].label(int(k.split(':')[1]))}"
                                   for k in self.rejected),
            "signed_gains": {k: list(v) for k, v in self.gains.items()},
        }

    # -------------------------------------------------------------- state
    def _state(self) -> dict:
        return {
            "phase": self.phase,
            "best_g": self.best_g,
            "best_key": self.best_key,
            "queue_levers": self.queue_levers,
            "pass_no": self.pass_no,
            "struct_changed": self.struct_changed,
            "arms": {k: list(v) for k, v in self.arms.items()},
            "arm_pending": self.arm_pending,
            "open_lever": self.open_lever,
            "gains": self.gains,
            "ledger": self.ledger,
            "applied": self.applied,
            "center": self.center,
            "start_g": self.start_g,
            "start_key": self.start_key,
            "solo": self.solo,
            "screen_queue": self._screen_queue,
            "screened": self.screened,
            "lookback_rounds": self.lookback_rounds,
            "skipped": [list(x) for x in self.skipped],
            "reclimbs": self.reclimbs,
            "lookback_from": self._lookback_from,
            "verdicts": self.verdicts,
            "rejected": self.rejected,
            "moves": self.moves,
            "retests": self.retests,
            "restarts_done": self.restarts_done,
            "revisited": self.revisited,
            "screen_retests": self.screen_retests,
            "lever_levels": {str(i): lv for i, lv in self.lever_levels.items()},
        }

    def _load(self, d: Mapping) -> None:
        self.phase = d["phase"]
        self.best_g = d["best_g"]
        self.best_key = d["best_key"]
        self.queue_levers = list(d["queue_levers"])
        self.pass_no = int(d["pass_no"])
        self.struct_changed = bool(d["struct_changed"])
        self.arms = {k: tuple(v) for k, v in d["arms"].items()}
        self.arm_pending = [list(g) for g in d["arm_pending"]]
        self.open_lever = d["open_lever"]
        self.gains = {k: list(v) for k, v in d["gains"].items()}
        self.ledger = list(d["ledger"])
        self.applied = list(d["applied"])
        self.center = d.get("center")
        self.start_g = d.get("start_g")
        self.start_key = d.get("start_key")
        self.solo = {k: list(v) for k, v in d.get("solo", {}).items()}
        self._screen_queue = [list(g) for g in d.get("screen_queue", [])]
        self.screened = int(d.get("screened", 0))
        self.lookback_rounds = int(d["lookback_rounds"])
        self.skipped = [(int(i), k) for i, k in d.get("skipped", [])]
        self.reclimbs = int(d.get("reclimbs", 0))
        self._lookback_from = d.get("lookback_from")
        self.verdicts = list(d.get("verdicts", []))
        self.rejected = dict(d.get("rejected", {}))
        self.moves = list(d.get("moves", []))
        self.retests = int(d.get("retests", 0))
        self.restarts_done = int(d.get("restarts_done", 0))
        self.revisited = bool(d.get("revisited", False))
        self.screen_retests = [list(x) for x in d.get("screen_retests", [])]
        self.lever_levels = {int(i): [int(x) for x in lv] for i, lv in d.get("lever_levels", {}).items()}


# ------------------------------------------------------------------- priors
class GainPrior:
    """An additive model from a lever ledger: predicted time of a config is
    ``base * prod over its changed knobs of (1 - mean signed gain)``. Knobs
    without a recorded gain contribute nothing (``nan`` only when nothing at
    all is known). A table prior for TPE / GP-EI and the selector's
    cold-start fallback."""

    scale = "log"

    def __init__(self, gains: Mapping[tuple[str, str], Sequence[float]], base: Mapping[str, object],
                 label=repr):  # fmt: skip
        self.gains = {tuple(k): list(v) for k, v in gains.items() if v}
        self.base = dict(base)
        self.label = label

    def predict(self, configs: Sequence[Mapping[str, object]]) -> list[float]:
        out = []
        for c in configs:
            t, known = 1.0, False
            for k, v in c.items():
                if self.base.get(k, object()) == v:
                    continue
                g = self.gains.get((k, self.label(v)))
                if g:
                    known = True
                    t *= max(1e-3, 1.0 - sum(g) / len(g))
            out.append(t if known or not self.gains else 1.0)
        return out


class Phased(Strategy):
    """Members in sequence: each asks until it is done or has spent its
    share (``shares``: own fresh asks per member, ``None`` = until done);
    every member is told every observation, so a later member (TPE / GP-EI)
    starts with the earlier member's (the climber's) data as its model. The
    replay of the recorded B3 grids and schedule rounds chose
    ``lookback -> tpe``: the climber is the best strategy at small budgets,
    TPE at larger ones, and a bandit that splits a small budget from the
    start was worse than either."""

    name = "phased"

    def __init__(self, space: SearchSpace, seed: int = 0, *, members: Sequence[Strategy],
                 shares: Sequence[int | None] = ()):  # fmt: skip
        super().__init__(space, seed)
        if not members:
            raise ValueError("Phased needs at least one member")
        self.members = list(members)
        self.shares = list(shares) + [None] * (len(self.members) - len(shares))
        self.cur = 0
        self.asked = [0] * len(self.members)

    def ask(self, n: int = 1) -> list[Trial]:
        while self.cur < len(self.members):
            m = self.members[self.cur]
            share = self.shares[self.cur]
            if m.done or (share is not None and self.asked[self.cur] >= share):
                if m.pending:
                    return []  # let the member's last round come back first
                self.cur += 1
                continue
            k = int(n) if share is None else min(int(n), share - self.asked[self.cur])
            out = m.ask(k)
            if out:
                self.asked[self.cur] += len(out)
                for t in out:
                    self.pending[t.uid] = t.genome
                return out
            if m.pending:
                return []
            self.cur += 1
        return []

    def tell(self, observations: Sequence[Observation]) -> None:
        super().tell(observations)
        for m in self.members:
            m.tell(observations)

    @property
    def done(self) -> bool:
        return self.cur >= len(self.members)

    def _state(self) -> dict:
        return {"cur": self.cur, "asked": self.asked, "members": [m.state_dict() for m in self.members]}

    def _load(self, d: Mapping) -> None:
        self.cur = int(d["cur"])
        self.asked = [int(x) for x in d["asked"]]
        for m, st in zip(self.members, d["members"]):
            m.load_state_dict(st)


def phased(space: SearchSpace, seed: int = 0, *, members: Sequence[str] = ("lookback", "tpe"),
           start: Sequence[int] | None = None, structural: Sequence[str] = (), prior=None,
           warm: Sequence[WarmStart] = (), shares: Sequence[int | None] = (),
           lookback_options: Mapping | None = None) -> Phased:  # fmt: skip
    """The mono tuner's default: the climber with its combination lookback,
    then TPE (or any engine strategy) on everything it measured.
    ``lookback_options``: extra :class:`SignedLookback` arguments
    (``budget``, ``prior_gains``, ``prior_pairs``, ...)."""
    built: list[Strategy] = []
    for i, m in enumerate(members):
        if m == "lookback":
            built.append(SignedLookback(space, seed * 101 + i + 1, start=start, structural=structural,
                                        **dict(lookback_options or {})))  # fmt: skip
        else:
            built.append(make_strategy(m, space, seed * 101 + i + 1, prior=prior, warm=warm,
                                       options={"n_init": 0} if m == "hill" else None))  # fmt: skip
    return Phased(space, seed, members=built, shares=shares)


def default_strategy(space: SearchSpace, budget: int, seed: int = 0, **kw) -> Strategy:
    """The tuner's default, chosen by replay of the recorded B3 lookback
    grids and schedule rounds: enumerate a space whose legal points fit the
    budget, else :func:`phased` (signed lookback, then TPE)."""
    listed = space.listed()
    if listed is not None and len(listed) <= int(budget):
        return make_strategy("exhaustive", space, seed)
    return phased(space, seed, **kw)


def mono_portfolio(
    space: SearchSpace,
    seed: int = 0,
    *,
    members: Sequence[str] = ("lookback", "hill", "de", "tpe"),
    prior=None,
    warm: Sequence[WarmStart] = (),
    lookback_options: Mapping | None = None,
    discount: float = 0.9,
) -> Portfolio:
    """The mono's default portfolio: the signed-lookback climber with the
    engine's hill climber, SHADE DE and TPE (or ``gp``) under the discounted
    Thompson bandit, warm-started (``warm``: dispatcher / BEST tables /
    nearest variants / selector top-k)."""
    built: list[Strategy] = []
    for i, m in enumerate(members):
        if m == "lookback":
            built.append(SignedLookback(space, seed * 101 + i + 1, **dict(lookback_options or {})))
        else:
            built.append(
                make_strategy(m, space, seed * 101 + i + 1, prior=prior,
                              options={"n_init": 0} if m == "hill" else None)
            )  # fmt: skip
    return Portfolio(space, seed, members=built, warm=warm, discount=discount)


def transfer_warm_starts(db, family: str, candidate: str, arch: str, problem: str, features: Mapping[str, float],
                         *, k: int = 3) -> list[WarmStart]:  # fmt: skip
    """Warm starts from a :class:`~rocke.dispatch.tuning.search.db.TuningDB`:
    this slot's own record (a re-tune), the ``k`` nearest variants' winners
    (distance over log2 of :func:`.catalog.variant_features`) and other
    archs'."""
    return db.warm_starts(family, arch, problem, features, candidate=candidate, k=k)


def genome_distance(space: SearchSpace, a: Genome, b: Genome) -> float:
    return space.distance(a, b)

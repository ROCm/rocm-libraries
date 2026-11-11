# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Trust-region shrinkage-coupled multivariate EI-TPE over discrete catalogs."""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .params import ORDINAL, Genome, SearchSpace
from .surrogate import POOL_REPAIR_CHECKS, Prior, _ModelStrategy, prior_ranked


class CoupledTPE(_ModelStrategy):
    """Coupled multivariate Parzen search with local trust-region intensification.

    Combines four mechanisms for mixed categorical/ordinal kernel catalogs:
    1. Empirical-Bayes shrinkage between factorized univariate log-ratios and
       a joint mixture-of-product-kernels KDE (log-geometric ordinal kernels
       plus dimension-scaled Aitchison-Aitken categorical kernels).
    2. Expected-improvement linear weights on the good quantile instead of
       uniform probability-of-improvement weights.
    3. Joint-only pending repulsion: in-flight batch trials enter only the
       multivariate bad density, avoiding pipeline-wide marginal penalties.
    4. Surrogate-ranked local trust-region moves (single-knob neighbors and
       competitive two-knob combinations) interleaved with global proposals.
    """

    name = "coupled_tpe"

    def __init__(
        self,
        space: SearchSpace,
        seed: int = 0,
        *,
        n_startup: int = 10,
        gamma: float = 0.25,
        kappa: float = 8.0,
        tr_budget: int = 8,
        combo_knobs: int = 4,
        combo_tol: float = 0.02,
        max_catalog: int = 1536,
        n_candidates: int = 96,
        prior: Optional[Prior] = None,
        prior_weight: float = 1.0,
        prior_pool: int = 2048,
        init: Sequence[Sequence[int]] = (),
    ):
        super().__init__(
            space,
            seed,
            n_startup=n_startup,
            prior=prior,
            prior_pool=prior_pool,
            init=init,
        )
        if not 0.0 < gamma < 1.0:
            raise ValueError(f"coupled_tpe gamma={gamma}, expected 0 < gamma < 1")
        if kappa < 0.0:
            raise ValueError(f"coupled_tpe kappa={kappa}, expected >= 0")
        if int(tr_budget) < 0:
            raise ValueError(f"coupled_tpe tr_budget={tr_budget}, expected >= 0")
        if int(combo_knobs) < 0 or combo_tol < 0:
            raise ValueError("coupled_tpe combo_knobs and combo_tol must be >= 0")
        if int(max_catalog) < 1 or int(n_candidates) < 1:
            raise ValueError("coupled_tpe max_catalog and n_candidates must be >= 1")
        self.gamma = float(gamma)
        self.kappa = float(kappa)
        self.tr_budget = int(tr_budget)
        self.combo_knobs = int(combo_knobs)
        self.combo_tol = float(combo_tol)
        self.max_catalog = int(max_catalog)
        self.n_candidates = int(n_candidates)
        self.prior_weight = float(prior_weight)
        self._tr_probes: Dict[str, int] = {}
        self._ask_count: int = 0

        self._active_axes: List[int] = [
            i for i, p in enumerate(space.params) if p.size > 1
        ]
        self._has_gates = any(g is not None for g in space._gates)
        self._ord_coords: Dict[int, np.ndarray] = {}
        for i in self._active_axes:
            p = space.params[i]
            if p.kind == ORDINAL:
                vals = [lvl[0][1] if len(lvl) == 1 else None for lvl in p.levels]
                if all(
                    isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0
                    for v in vals
                ):
                    lo = math.log(float(vals[0]))
                    hi = math.log(float(vals[-1]))
                    span = hi - lo if hi > lo else 1.0
                    self._ord_coords[i] = np.array(
                        [(math.log(float(v)) - lo) / span for v in vals], dtype=float
                    )
                else:
                    self._ord_coords[i] = np.linspace(0.0, 1.0, p.size, dtype=float)

        listed = space.listed()
        if listed:
            seen_k: Dict[str, Genome] = {}
            for g in listed:
                cg = space.canonical_genome(g)
                k = space.key(cg)
                if k is not None and k not in seen_k:
                    seen_k[k] = cg
            self._listed_keys: Optional[Tuple[str, ...]] = tuple(seen_k.keys())
            self._listed_genomes: Optional[Tuple[Genome, ...]] = tuple(seen_k.values())
            self._listed_arr: Optional[np.ndarray] = np.asarray(
                self._listed_genomes, dtype=int
            )
        else:
            self._listed_keys = None
            self._listed_genomes = None
            self._listed_arr = None

    def _build_startup(self) -> None:
        picks: List[Genome] = []
        keys: set = set(self.history)

        def add(g) -> None:
            k = self.space.key(g)
            if k is not None and k not in keys:
                keys.add(k)
                picks.append(g)

        for g in self._seeds:
            r = self.space.repair(g, self.rng)
            if r is not None:
                add(r)
        n_seeded = len(picks)

        if self.prior is not None:
            ranked = prior_ranked(
                self.prior_scale, self.space, self.rng, pool=self.prior_pool
            )
            self._prior_top = [list(g) for g in ranked[:32]]
            cap = (self.n_startup + 1) // 2 + n_seeded
            for g in ranked:
                if len(picks) >= cap:
                    break
                add(g)

        # Avoid spending multiple extra blind probes when >= 2 warm seeds exist.
        target = (
            max(6, n_seeded + 1)
            if (n_seeded >= 2 and self.prior is None)
            else self.n_startup
        )
        if len(picks) < target:
            for g in self.space.stratified(self.rng, self.n_startup + 2):
                if len(picks) >= target:
                    break
                add(g)
        self._startup = [list(g) for g in picks]

    def _extra_state(self) -> dict:
        return {"tr_probes": dict(self._tr_probes), "ask_count": self._ask_count}

    def _load(self, state: Mapping) -> None:
        super()._load(state)
        self.uid_prefix = f"{self.name}:{self.seed}"

    def _load_extra(self, d: Mapping) -> None:
        self._tr_probes = {str(k): int(v) for k, v in d.get("tr_probes", {}).items()}
        self._ask_count = int(d.get("ask_count", 0))

    def _genome_of(self, key: str) -> Optional[Genome]:
        for g in self.pending.values():
            if self.space.key(g) == key:
                return g
        return None

    def _axis_kernel_tables(self, n_obs: int, tau_d: float) -> Dict[int, np.ndarray]:
        tables: Dict[int, np.ndarray] = {}
        for i in self._active_axes:
            p = self.space.params[i]
            c = p.size
            if i in self._ord_coords:
                coords = self._ord_coords[i]
                bw = max(0.14, 0.32 * (max(1, n_obs) ** (-0.2)) / tau_d)
                diff = coords[:, None] - coords[None, :]
                ker = np.exp(-0.5 * (diff / bw) ** 2)
                ker /= ker.sum(axis=0, keepdims=True)
                tables[i] = ker
            else:
                b = max(
                    0.05,
                    min(
                        (c - 1.0) / c,
                        (c - 1.0) / (c * (1.0 + tau_d * math.sqrt(max(1, n_obs)))),
                    ),
                )
                ker = np.full((c, c), b / max(1, c - 1), dtype=float)
                np.fill_diagonal(ker, 1.0 - b)
                tables[i] = ker
        return tables

    def _eval_densities(
        self,
        cand_arr: np.ndarray,
        cand_act: Optional[np.ndarray],
        obs_genomes: Sequence[Genome],
        weights: np.ndarray,
        tau_mv: float,
        prior_mass: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        m = cand_arr.shape[0]
        n_obs = len(obs_genomes)
        if n_obs == 0 or not self._active_axes:
            z = np.zeros(m, dtype=float)
            return z, z
        obs_arr = np.asarray(obs_genomes, dtype=int)
        w = weights / weights.sum()

        w_prior = prior_mass / (n_obs + prior_mass)
        w_obs_total = 1.0 - w_prior

        tables_univ = self._axis_kernel_tables(n_obs, tau_d=1.0)
        tables_mv = self._axis_kernel_tables(n_obs, tau_d=tau_mv)

        log_univ = np.zeros(m, dtype=float)
        log_comp = np.zeros((m, n_obs + 1), dtype=float)
        log_w_all = np.empty(n_obs + 1, dtype=float)
        log_w_all[0] = math.log(w_prior)
        log_w_all[1:] = math.log(w_obs_total) + np.log(np.maximum(w, 1e-12))

        if not self._has_gates:
            for i in self._active_axes:
                c = self.space.params[i].size
                ku = tables_univ[i]
                obs_lvls = obs_arr[:, i]
                marg = w_prior * (1.0 / c) + w_obs_total * (ku[:, obs_lvls] @ w)
                log_univ += np.log(np.maximum(marg[cand_arr[:, i]], 1e-15))

                km = np.log(np.maximum(tables_mv[i], 1e-15))
                log_comp[:, 0] += -math.log(c)
                log_comp[:, 1:] += km[cand_arr[:, i][:, None], obs_lvls[None, :]]
        else:
            obs_act = np.asarray(
                [self.space.active(g) for g in obs_genomes], dtype=bool
            )
            for i in self._active_axes:
                c = self.space.params[i].size
                ku = tables_univ[i]
                km = np.log(np.maximum(tables_mv[i], 1e-15))
                act_o = obs_act[:, i]
                act_c = cand_act[:, i]
                if np.any(act_o):
                    w_a = w[act_o]
                    w_a = w_a / w_a.sum()
                    wa_prior = prior_mass / (float(act_o.sum()) + prior_mass)
                    marg = wa_prior * (1.0 / c) + (1.0 - wa_prior) * (
                        ku[:, obs_arr[act_o, i]] @ w_a
                    )
                else:
                    marg = np.full(c, 1.0 / c, dtype=float)
                log_univ += np.where(
                    act_c, np.log(np.maximum(marg[cand_arr[:, i]], 1e-15)), 0.0
                )
                log_comp[:, 0] += np.where(act_c, -math.log(c), 0.0)
                pair_km = km[cand_arr[:, i][:, None], obs_arr[:, i][None, :]]
                contrib = np.where(act_o[None, :], pair_km, -math.log(c))
                log_comp[:, 1:] += np.where(act_c[:, None], contrib, 0.0)

        tot = log_comp + log_w_all[None, :]
        mx = tot.max(axis=1, keepdims=True)
        log_mv = mx[:, 0] + np.log(np.exp(tot - mx).sum(axis=1))
        return log_univ, log_mv

    def _local_neighbors(
        self, center_key: str, center_val: float, taken: set
    ) -> List[Genome]:
        cg = self.genomes[center_key]
        raw = list(self.space.moves(cg, ordinal_radius=1))
        if self.combo_knobs >= 2 and math.isfinite(center_val):
            best_per_param: Dict[int, Tuple[float, int]] = {}
            for m in raw:
                mk = self.space.key(m)
                mv = self.history.get(mk) if mk is not None else None
                if (
                    mv is None
                    or not math.isfinite(mv)
                    or mv > center_val * (1.0 + self.combo_tol)
                ):
                    continue
                diffs = [idx for idx in range(len(cg)) if m[idx] != cg[idx]]
                if len(diffs) == 1:
                    ax = diffs[0]
                    if ax not in best_per_param or (mv, m[ax]) < best_per_param[ax]:
                        best_per_param[ax] = (mv, m[ax])
            top = sorted(best_per_param.items(), key=lambda t: (t[1][0], t[0]))[
                : self.combo_knobs
            ]
            for r in range(2, len(top) + 1):
                for subset in itertools.combinations(top, r):
                    g = list(cg)
                    for ax, (_v, lvl) in subset:
                        g[ax] = lvl
                    tg = tuple(g)
                    if self.space.valid(tg):
                        raw.append(self.space.canonical_genome(tg))

        self.rng.shuffle(raw)
        out: List[Genome] = []
        seen: set = set()
        for m in raw:
            mk = self.space.key(m)
            if mk is None or mk in seen or mk in self.history or mk in taken:
                continue
            seen.add(mk)
            out.append(self.space.canonical_genome(m))
        return out

    def _score_candidates(
        self,
        cands: Sequence[Genome],
        good: Sequence[Genome],
        w_good: np.ndarray,
        bad_hist: Sequence[Genome],
        w_bad_univ: np.ndarray,
        bad_mv: Sequence[Genome],
        w_bad_mv: np.ndarray,
        n_ok: int,
    ) -> List[float]:
        d_eff = max(1, len(self._active_axes))
        tau_mv = min(1.0, math.sqrt(4.0 / d_eff))
        cand_arr = np.asarray(cands, dtype=int)
        cand_act = (
            np.asarray([self.space.active(g) for g in cands], dtype=bool)
            if self._has_gates
            else None
        )

        lu_g, lmv_g = self._eval_densities(
            cand_arr, cand_act, good, w_good, tau_mv, prior_mass=1.0
        )
        lu_b, _ = self._eval_densities(
            cand_arr, cand_act, bad_hist, w_bad_univ, tau_mv, prior_mass=2.0
        )
        _, lmv_b = self._eval_densities(
            cand_arr, cand_act, bad_mv, w_bad_mv, tau_mv, prior_mass=2.0
        )

        s_univ = lu_g - lu_b
        s_mv = (lmv_g - lmv_b) * math.sqrt(float(d_eff))
        alpha = float(n_ok) / (float(n_ok) + self.kappa)
        scores = ((1.0 - alpha) * s_univ + alpha * s_mv).tolist()

        if self.prior is not None and self.prior_weight > 0:
            pv = self.prior_scale.values(self.space, cands)
            fin = [v for v in pv if math.isfinite(v)]
            if len(fin) >= 2:
                mu = sum(fin) / len(fin)
                sd = math.sqrt(sum((v - mu) ** 2 for v in fin) / len(fin)) or 1.0
                scale_f = float(d_eff) * 0.35
                w = scale_f * self.prior_weight * self.n_startup / (self.n_startup + n_ok)
                scores = [
                    s - w * ((v - mu) / sd if math.isfinite(v) else 0.0)
                    for s, v in zip(scores, pv)
                ]
        return scores

    def _propose(self, taken: set, batch_pos: int) -> Optional[Genome]:
        ranked = sorted(
            (v, k) for k, v in self.history.items() if math.isfinite(v) and v > 0
        )
        if len(ranked) < 2:
            return self._random_unseen(taken)

        n_ok = len(ranked)
        n_good = min(
            n_ok - 1,
            max(
                1 if n_ok == 2 else 2,
                math.ceil(
                    self.gamma * n_ok if n_ok <= 16 else 4.0 + 0.15 * (n_ok - 16)
                ),
            ),
        )
        good_y = np.array([math.log(v) for v, _k in ranked[:n_good]], dtype=float)
        y_ref = math.log(ranked[n_good][0])
        gaps = np.maximum(y_ref - good_y, 0.0)
        span = float(gaps.max()) if float(gaps.max()) > 1e-9 else 1.0
        w_good = 0.25 + (gaps / span)
        good = [self.genomes[k] for _v, k in ranked[:n_good]]

        bad_ok = [self.genomes[k] for _v, k in ranked[n_good:]]
        fails = [tuple(self.failed[k]) for k in sorted(self.failed)]
        bad_hist = bad_ok + fails
        if not bad_hist:
            bad_hist = [good[-1]]
        w_bad_univ = np.array(
            [1.0] * len(bad_ok) + [0.5] * len(fails) if (bad_ok or fails) else [1.0],
            dtype=float,
        )

        pend = [
            self.genomes.get(k) or self._genome_of(k)
            for k in sorted(taken)
            if k not in self.history
        ]
        pend = [tuple(g) for g in pend if g is not None]
        bad_mv = list(bad_hist) + pend
        w_bad_mv = (
            np.concatenate([w_bad_univ, np.full(len(pend), 2.0, dtype=float)])
            if pend
            else w_bad_univ
        )

        slot = self._ask_count
        self._ask_count += 1

        # 1. Surrogate-ranked local trust-region intensification around incumbent.
        best_val, best_key = ranked[0]
        used_tr = self._tr_probes.get(best_key, 0)
        do_local = (slot % 2 == 0) or (n_ok < 10 and slot % 3 == 1)
        if self.tr_budget > 0 and used_tr < self.tr_budget and do_local:
            tr_cands = self._local_neighbors(best_key, best_val, taken)
            if len(tr_cands) >= 2:
                self._tr_probes[best_key] = used_tr + 1
                tr_scores = self._score_candidates(
                    tr_cands,
                    good,
                    w_good,
                    bad_hist,
                    w_bad_univ,
                    bad_mv,
                    w_bad_mv,
                    n_ok,
                )
                best_i = max(range(len(tr_cands)), key=lambda i: (tr_scores[i], -i))
                return (
                    tr_cands[best_i]
                    if self.build_preference is None
                    else self.build_preference.choose(
                        self.space, self.rng, tr_cands, tr_scores, best_i
                    )
                )

        # 2. Global shrinkage-coupled multivariate EI-TPE acquisition.
        cands: List[Genome] = []
        seen: set = set()

        def add_cand(g) -> None:
            k = self.space.key(g)
            if k is None or k in seen or k in self.history or k in taken:
                return
            seen.add(k)
            cands.append(self.space.canonical_genome(g))

        if self._listed_genomes is not None:
            m_cat = len(self._listed_genomes)
            idxs = list(range(m_cat))
            self.rng.shuffle(idxs)
            unseen_idxs = [
                idx
                for idx in idxs
                if self._listed_keys[idx] not in self.history
                and self._listed_keys[idx] not in taken
            ]
            if m_cat <= self.max_catalog:
                cands = [self._listed_genomes[idx] for idx in unseen_idxs]
            elif unseen_idxs:
                pool_k = min(
                    len(unseen_idxs),
                    int(96 * (2.0 ** (max(0, n_ok - 6) / 8.0))),
                    self.max_catalog,
                )
                d_eff = max(1, len(self._active_axes))
                tau_mv = min(1.0, math.sqrt(4.0 / d_eff))
                sub_arr = self._listed_arr[unseen_idxs]
                sub_act = (
                    np.asarray(
                        [
                            self.space.active(self._listed_genomes[idx])
                            for idx in unseen_idxs
                        ],
                        dtype=bool,
                    )
                    if self._has_gates
                    else None
                )
                lu_sub, lmv_sub = self._eval_densities(
                    sub_arr, sub_act, good, w_good, tau_mv, prior_mass=1.0
                )
                alpha = float(n_ok) / (float(n_ok) + self.kappa)
                log_l = (1.0 - alpha) * lu_sub + alpha * lmv_sub
                u_rand = np.array(
                    [
                        max(1e-15, min(1.0 - 1e-15, self.rng.random()))
                        for _ in range(len(unseen_idxs))
                    ]
                )
                gumbel = -np.log(-np.log(u_rand))
                top_order = np.argsort(-(log_l + gumbel), kind="mergesort")[:pool_k]
                for pos in top_order:
                    add_cand(self._listed_genomes[unseen_idxs[int(pos)]])
                for _v, k in ranked[:3]:
                    for mv in self.space.moves(self.genomes[k]):
                        add_cand(mv)
                for g in self._prior_top[:16]:
                    add_cand(tuple(g))
        else:
            for g in self.space.stratified(
                self.rng, max(256, self.n_candidates), repair_checks=POOL_REPAIR_CHECKS
            ):
                add_cand(g)
            for _v, k in ranked[:3]:
                for mv in self.space.moves(self.genomes[k]):
                    add_cand(mv)
            for g in self._prior_top[:16]:
                add_cand(tuple(g))

        if not cands:
            return self._random_unseen(taken)

        scores = self._score_candidates(
            cands, good, w_good, bad_hist, w_bad_univ, bad_mv, w_bad_mv, n_ok
        )
        best_i = max(range(len(cands)), key=lambda i: (scores[i], -i))
        return (
            cands[best_i]
            if self.build_preference is None
            else self.build_preference.choose(
                self.space, self.rng, cands, scores, best_i
            )
        )

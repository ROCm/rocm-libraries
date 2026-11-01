# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Learned per-shape gemm_skinny configuration choice.

The bin tables (:mod:`.skinny_table`, :mod:`.skinny_native_table`) pick one
configuration per (log2 extent, layout) bin at the bin-centre shape. Within a
bin the best configuration moves (split-K degree, wave split along K, tile):
neighbouring configurations win single shapes by a wide margin. This module ranks a
configuration menu for the request's exact shape with a LightGBM model of the
measured kernel time (log microseconds) of every screened gemm_skinny
configuration (``selector_models/skinny_cfg_<arch>.*``), and replaces the
table plan by the menu's best predicted configuration when

* the main model ranks it among the menu's :data:`SKINNY_SELECT_TOPK` best,
* the main model and every guard bag (models fitted on bootstrap resamples
  of the training cells) predict it faster than the table plan, and
* their mean predicted gain is at least :data:`SKINNY_SELECT_MARGIN`.

The menu: the base configurations of the model sidecar (atoms, wave split,
K chunk, prefetch: the configurations that were among the best measured
ones of some training cell) plus the table plan's, each at the split-K
degrees of :data:`SKINNY_SELECT_WAVES` (wave targets, as the tables), in the
request's layout (:func:`.skinny.skinny_native`); only valid, launchable
configurations are ranked. Knobs: :data:`SKINNY_SELECT` (off = the tables
alone), the margin, the wave targets.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SKINNY_SELECT = True
SKINNY_SELECT_ARCHES: Tuple[str, ...] = ("gfx950", "gfx942")
# mean predicted gain (fraction) the menu's pick needs over the table plan
SKINNY_SELECT_MARGIN = 0.03
# split-K wave targets of every menu configuration (0 = no split-K)
SKINNY_SELECT_WAVES: Tuple[int, ...] = (0, 256, 512, 1024, 2048, 4096)
# menu entries the guard bags re-score (the main model's best ones)
SKINNY_SELECT_TOPK = 8
# requests with min(M, N) below this keep the table plan (decode-like shapes:
# on gfx950 with min(M, N) <= 16 the model's pick was slower than the table's)
SKINNY_SELECT_MIN_SMALL = 17
# ... and above this many 128x128 output tiles (outside the training cells:
# the measured gemm_skinny cells have at most ~1k tiles on gfx950 and ~300 on
# gfx942)
SKINNY_SELECT_MAX_TILES = {"gfx950": 1024, "gfx942": 256}
FEATURE_VERSION = "r3sk-v1"
MODEL_DIR_ENV = "ROCKE_GEMM_SKINNY_MODEL_DIR"
SKINNY_SELECT_ENV = "ROCKE_GEMM_SKINNY_SELECT"
_CUS = {"gfx950": 256, "gfx942": 304}
_DECISIONS: Dict[Tuple[Any, ...], Any] = {}
_DECISIONS_MAX = 8192

FEATURES: Tuple[str, ...] = (
    "a942",
    "bf16",
    "a_mn",
    "b_mn",
    "ld_odd",
    "k_odd",
    "lM",
    "lN",
    "lK",
    "lsmall",
    "lbig",
    "m_le_n",
    "ma",
    "na",
    "wm",
    "wn",
    "wk",
    "kc",
    "pf",
    "lsplit",
    "tile_m",
    "tile_n",
    "ltiles",
    "lwaves",
    "waves_cu",
    "lsteps",
    "pad_m",
    "pad_n",
    "lload",
    "lws",
    "regs",
    "lds_kb",
    "u16",
    "mnp",
    "ru8",
    "lflop",
    "lbytes",
)


def _l2(x: float) -> float:
    return math.log2(max(1.0, float(x)))


def features(arch: str, M: int, N: int, K: int, spec: Any) -> List[float]:
    """The model's feature row of ``spec`` (a SkinnyGemmSpec in the request's
    layout) at ``M x N x K`` (the order of :data:`FEATURES`)."""
    from ...instances.common.gemm_skinny import _lds_bytes

    M, N, K = int(M), int(N), int(K)
    cus = _CUS.get(arch, 256)
    a_mn, b_mn = spec.mn_operands
    tm, tn = spec.tile_m, spec.tile_n
    gx, gy = -(-N // tn), -(-M // tm)
    tiles = gx * gy
    split = max(1, int(spec.split))
    waves = tiles * spec.waves * split
    steps = -(-K // spec.step_k)
    sps = max(1.0, -(-steps // split) / max(1, spec.warp_k))
    load = (
        tiles
        * spec.warp_m
        * spec.warp_n
        * 16
        * (spec.wave_m_atoms + spec.wave_n_atoms)
        * K
        * 2
    )
    ws = split * tiles * tm * tn * 4 if split > 1 else 0
    acc = spec.wave_m_atoms * spec.wave_n_atoms * 4
    frag = (spec.wave_m_atoms + spec.wave_n_atoms) * (spec.kc // 2)
    regs = acc + (spec.prefetch + 1) * frag
    return [
        1.0 if arch == "gfx942" else 0.0,
        1.0 if spec.dtype == "bf16" else 0.0,
        1.0 if a_mn else 0.0,
        1.0 if b_mn else 0.0,
        1.0 if int(spec.mnalign) < 8 else 0.0,
        1.0 if int(spec.kalign) < 8 else 0.0,
        _l2(M),
        _l2(N),
        _l2(K),
        _l2(min(M, N)),
        _l2(max(M, N)),
        1.0 if M <= N else 0.0,
        float(spec.wave_m_atoms),
        float(spec.wave_n_atoms),
        float(spec.warp_m),
        float(spec.warp_n),
        float(spec.warp_k),
        float(spec.kc),
        float(spec.prefetch),
        _l2(split),
        float(tm),
        float(tn),
        _l2(tiles),
        _l2(waves),
        waves / cus,
        math.log2(sps),
        gy * tm / M,
        gx * tn / N,
        _l2(load),
        _l2(1 + ws),
        float(regs),
        _lds_bytes(spec) / 1024.0,
        1.0 if spec.mn_lds == "u16" and (a_mn or b_mn) else 0.0,
        float(spec.mn_pad),
        1.0 if int(spec.red_unroll) == 8 else 0.0,
        _l2(2.0 * M * N * K),
        _l2(2.0 * (M * K + N * K + M * N)),
    ]


class _Packed:
    """A LightGBM text model flattened for row x tree vectorised walks."""

    def __init__(self, ens: Any):
        import numpy as np

        trees = [t for t in ens.trees]
        nt = len(trees)
        ni = max(1, max((len(t.split_feature) for t in trees), default=1))
        nl = max(1, max((len(t.leaf_value) for t in trees), default=1))
        self.sf = np.zeros((nt, ni), dtype=np.int64)
        self.th = np.zeros((nt, ni), dtype=np.float64)
        self.dl = np.zeros((nt, ni), dtype=bool)
        self.mz = np.zeros((nt, ni), dtype=bool)
        self.lc = np.full((nt, ni), -1, dtype=np.int64)
        self.rc = np.full((nt, ni), -1, dtype=np.int64)
        self.lv = np.zeros((nt, nl), dtype=np.float64)
        self.const = 0.0
        self.single = np.zeros(nt, dtype=bool)
        for i, t in enumerate(trees):
            if t.num_leaves <= 1:
                self.single[i] = True
                self.const += t.leaf_value[0] if t.leaf_value else 0.0
                continue
            k = len(t.split_feature)
            self.sf[i, :k] = t.split_feature
            self.th[i, :k] = t.threshold
            dt = np.asarray(t.decision_type, dtype=np.int64)
            self.dl[i, :k] = (dt & 2) != 0
            self.mz[i, :k] = ((dt >> 2) & 3) == 1
            self.lc[i, :k] = t.left
            self.rc[i, :k] = t.right
            self.lv[i, : len(t.leaf_value)] = t.leaf_value

    def predict(self, X: Any) -> Any:
        import numpy as np

        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        live = np.nonzero(~self.single)[0]
        node = np.zeros((n, live.size), dtype=np.int64)
        ti = np.broadcast_to(live, (n, live.size))
        ri = np.broadcast_to(np.arange(n)[:, None], (n, live.size))
        act = node >= 0
        while act.any():
            r, c = np.nonzero(act)
            t = ti[r, c]
            nd = node[r, c]
            v = X[ri[r, c], self.sf[t, nd]]
            zero = self.mz[t, nd] & (np.abs(v) <= 1e-35)
            left = np.where(zero, self.dl[t, nd], v <= self.th[t, nd])
            node[r, c] = np.where(left, self.lc[t, nd], self.rc[t, nd])
            act = node >= 0
        out = self.lv[ti, ~node].sum(axis=1) + self.const
        return out


def _model_dir() -> Path:
    env = os.environ.get(MODEL_DIR_ENV, "").strip()
    return Path(env) if env else Path(__file__).with_name("selector_models")


@lru_cache(maxsize=None)
def _load(
    arch: str, model_dir: str
) -> Optional[Tuple[Dict[str, Any], Tuple[_Packed, ...]]]:
    from .selector import TreeEnsemble

    base = Path(model_dir) / f"skinny_cfg_{arch}"
    meta_p = base.with_suffix(".meta.json")
    if not meta_p.is_file():
        return None
    meta = json.loads(meta_p.read_text())
    if meta.get("feature_version") != FEATURE_VERSION or meta.get("arch") != arch:
        return None
    if list(meta.get("features", ())) != list(FEATURES):
        raise ValueError(f"{meta_p}: feature list disagrees with this tree")
    models = []
    for nm in [meta.get("model", "")] + list(meta.get("bags", ())):
        p = Path(model_dir) / nm
        if not nm or not p.is_file():
            raise ValueError(f"{p}: model file named by the sidecar is missing")
        ens = TreeEnsemble.from_text(p.read_text())
        if ens.feature_names and ens.feature_names != list(FEATURES):
            raise ValueError(f"{p}: feature names disagree with this tree")
        models.append(_Packed(ens))
    if len(models) < 2:
        return None
    return meta, models[0], tuple(models[1:])


def load(arch: str):
    """``(meta, model, bags)`` of the arch's configuration model, or None."""
    return _load(arch, str(_model_dir()))


def _base_key(spec: Any) -> Tuple[int, ...]:
    return (
        int(spec.wave_m_atoms),
        int(spec.wave_n_atoms),
        int(spec.warp_m),
        int(spec.warp_n),
        int(spec.warp_k),
        int(spec.kc),
        int(spec.prefetch),
    )


def menu(req: Any, plan: Any, meta: Dict[str, Any]) -> List[Any]:
    """The configurations ranked for ``req``: the sidecar's base menu plus the
    plan's base, each at the :data:`SKINNY_SELECT_WAVES` split degrees, in the
    request's layout; valid and launchable only (the plan itself first)."""
    from ...instances.common.gemm_skinny import SkinnyGemmSpec
    from . import skinny as SKM

    M, N, K = int(req.M), int(req.N), int(req.K)
    dt = "fp16" if req.dtype in ("fp16", "f16") else "bf16"
    out = [plan]
    seen = {(_base_key(plan), int(plan.split))}
    bases = [tuple(b) for b in meta.get("menu", ())]
    if _base_key(plan) not in bases:
        bases.append(_base_key(plan))
    for ma, na, wm, wn, wk, kc, pf in bases:
        rcr = SkinnyGemmSpec(
            dtype=dt,
            wave_m_atoms=ma,
            wave_n_atoms=na,
            warp_m=wm,
            warp_n=wn,
            warp_k=wk,
            kc=kc,
            prefetch=pf,
            red_sync="sc1",
            kalign=SKM._kalign(K),
        )
        for tw in SKINNY_SELECT_WAVES:
            s = SKM.skinny_split(rcr, M, N, K, tw) if tw else 1
            sp = replace(rcr, split=s, red_unroll=8 if s >= 16 else 4)
            sp = SKM._native_fit(SKM.skinny_native(sp, req), req.arch, M, N, K)
            key = (_base_key(sp), int(sp.split))
            if key in seen or not SKM._plan_ok(sp, req.arch, M, N, K):
                continue
            seen.add(key)
            out.append(sp)
    return out


def enabled(arch: str = "") -> bool:
    """:data:`SKINNY_SELECT` for ``arch``, unless ``ROCKE_GEMM_SKINNY_SELECT``
    is set (``0`` / ``off``: the tables alone; ``1`` / ``on``: the model)."""
    env = os.environ.get(SKINNY_SELECT_ENV, "").strip().lower()
    on = SKINNY_SELECT
    if env in ("0", "off", "false", "no"):
        on = False
    elif env in ("1", "on", "true", "yes"):
        on = True
    return on and (not arch or arch in SKINNY_SELECT_ARCHES)


def select(req: Any, plan: Any) -> Tuple[Any, str]:
    """``(spec, why)``: the model's configuration for ``req`` or ``plan``."""
    if not enabled(req.arch):
        return plan, "SKINNY_SELECT off"
    if min(int(req.M), int(req.N)) < SKINNY_SELECT_MIN_SMALL:
        return plan, f"min(M, N) < {SKINNY_SELECT_MIN_SMALL}: the table plan"
    tiles = -(-int(req.M) // 128) * -(-int(req.N) // 128)
    if tiles > int(SKINNY_SELECT_MAX_TILES.get(req.arch, 0)):
        return plan, f"{tiles} 128x128 tiles: outside the model's cells, the table plan"
    loaded = load(req.arch)
    if loaded is None:
        return plan, f"no gemm_skinny configuration model for {req.arch}"
    key = (
        req.arch,
        str(req.dtype),
        str(getattr(req, "layout", "RCR")).upper(),
        int(req.M),
        int(req.N),
        int(req.K),
        plan.kernel_name(),
        int(plan.split),
        SKINNY_SELECT_MARGIN,
        SKINNY_SELECT_WAVES,
        SKINNY_SELECT_TOPK,
        str(_model_dir()),
    )
    hit = _DECISIONS.get(key)
    if hit is not None:
        return hit
    meta, model, bags = loaded
    cands = menu(req, plan, meta)
    res: Tuple[Any, str] = (plan, "the table plan is the menu's only entry")
    if len(cands) > 1:
        import numpy as np

        X = np.asarray(
            [features(req.arch, req.M, req.N, req.K, c) for c in cands],
            dtype=np.float64,
        )
        # the main model ranks the menu; the guard bags re-score its
        # SKINNY_SELECT_TOPK best entries against the plan (row 0)
        p0 = model.predict(X)
        order = [0] + [
            int(i) for i in np.argsort(p0[1:], kind="stable")[:SKINNY_SELECT_TOPK] + 1
        ]
        Xs = X[order]
        P = np.stack([p0[order]] + [b.predict(Xs) for b in bags])  # models x top
        gain = P[:, :1] - P  # > 0: predicted faster than the plan
        mean = gain.mean(axis=0)
        ok = (gain > 0).all(axis=0) & (mean >= math.log1p(SKINNY_SELECT_MARGIN))
        ok[0] = False
        if ok.any():
            j = int(np.argmax(np.where(ok, mean, -np.inf)))
            c = cands[order[j]]
            res = (
                c,
                f"model: {c.kernel_name()}|s{c.split} predicted "
                f"{math.exp(mean[j]):.2f}x faster than the table plan (every model)",
            )
        else:
            res = (plan, "model: no menu entry beats the table plan in every model")
    if len(_DECISIONS) >= _DECISIONS_MAX:
        _DECISIONS.clear()
    _DECISIONS[key] = res
    return res


def clear_cache() -> None:
    _DECISIONS.clear()
    _load.cache_clear()


__all__ = [
    "SKINNY_SELECT",
    "SKINNY_SELECT_ARCHES",
    "SKINNY_SELECT_MARGIN",
    "SKINNY_SELECT_WAVES",
    "SKINNY_SELECT_TOPK",
    "SKINNY_SELECT_MIN_SMALL",
    "SKINNY_SELECT_MAX_TILES",
    "FEATURES",
    "FEATURE_VERSION",
    "features",
    "menu",
    "select",
    "enabled",
    "load",
    "clear_cache",
]

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Learned time model of the gfx942 native NN / TN / TT forms of the g5 register-staged GEMM body over the
g5 plan-model menu (gfx942 non-NT).

The g5 body (``vgpr_prefetch``) reads an MN-contiguous operand as micro-blocks of ``vp_mn_kv`` K rows x
``mn_vec`` M / N elements and transposes them in registers. ``r2_native_fit.R2_NATIVE_LIN`` is a linear time
model per (layout, geometry, micro-block) for a small geometry menu; the plan-model menu of the RCR body
(``g5_plan``, its geometries x split-K) has no linear native cost model, so without this module NN / TN / TT reach it only
through guarded heuristics (``common.MERGE2_NATIVE_G5P``). This module scores every *native form* (plan x
micro-block x lane wrap, the dispatcher's native trait settings) of the plan model's best-ranked RCR plans:

* **features** (:func:`form_features`): the plan model's own features (``g5_plan.plan_features``) + the layout
  (which operands are MN-contiguous), the micro-block, the lane wrap and its repeated fraction, micro-blocks per
  thread, the native trait settings, the MN leading-dimension alignment and the plan model's predicted RCR time;
  RCR rows (the same plans behind the transpose pre-pass, GEMM-only time) share the model with layout "RCR";
* **model** (``selector_models/g5_native_<arch>.model.txt`` + ``.meta.json``): LightGBM regression of ln(us),
  fitted on interleaved gfx942 timings (every measured kernel checked against an f64 reference);
* **ranking** (:func:`rank_native`): the ``G5N_TOP`` best RCR plans of the plan model (no serial split-K: the
  native forms are validated with the atomic split), each with every launchable micro-block form, by predicted
  time.

``common.g5_vgpr_tile_trait`` / ``common.r2_vp_mn_form`` take the best form when :func:`active` holds (knob
``G5N_ON``, the arch, the layout and a shipped model); ``plan.py`` takes its predicted time as the native
estimate and the model's RCR prediction of the pre-pass body. ``G5N_ON = False`` restores the dispatch
without the native-form model exactly.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import g5_plan as _gp

G5N_ON = True
G5N_ARCHES = ("gfx942",)
# data layouts served natively by the model (RRR = NN, CRR = TN, CCR = TT)
G5N_LAYOUTS: Tuple[str, ...] = ("RRR", "CRR", "CCR")
# how many of the plan model's ranked RCR plans get native forms scored
G5N_TOP = 10
# a serial split-K plan (split_k_serial, the in-launch reduction) of the ranking is scored with its own
# reduction too (False: only its atomic-reduction twin); on measured mid-size problems the best serial form
# beat the best atomic / split-free form in most cases where both were timed
G5N_SERIAL = True
# micro-blocks (vp_mn_kv, mn_vec) scored per plan (each with the lane wrap only when the plain form does not
# launch); 2-element MN loads ran much slower than their RCR body in every layout
G5N_MICRO: Tuple[Tuple[int, int], ...] = ((8, 8), (4, 8), (8, 4), (4, 4), (2, 8))
# also score the forms without ``vp_unroll`` of a plan that unrolls (g5_plan.plan_unroll: the RCR body's
# unrolled loop was measured on NT only)
G5N_UNROLL_BOTH = True
# also score the tuned trait set of a plan whose body is a tuned (layout, geometry) of
# common.R2_VP_MN_TUNED
G5N_TUNED_FORMS = True
# extra trait sets scored on top of the dispatcher's native settings, each with every G5N_MICRO micro-block
# (e.g. {"vp_mn_pipe": 0}: the transpose of a row piece in its own slot; {"vp_mn_worder": 1}: A then B
# writes); () = none
G5N_TRAIT_VARIANTS: Tuple[Dict[str, Any], ...] = (
    {"vp_mn_pipe": 0},
    {"vp_mn_worder": 1},
    {"vp_mn_order": 2},
)
# the dispatcher takes the model's best form only when it is predicted at least this factor faster than the
# previous native pick's predicted time (1.0 = always the model's argmin)
G5N_MARGIN = 1.0

# predict_spec of an RCR body with the g5_plan main-loop traits applies G5P_MAINLOOP_SCALE (False:
# the model's raw time, which was fitted on bodies without the traits)
R3M_G5N_MAINLOOP_SCALE = True
# feature value for "not applicable"
G5N_NA = -1.0
# device class feature of the run-time prediction: the training runs mix gfx942 device classes (a class
# indicator absorbs their clock / bandwidth difference); 1.0 = the higher-bandwidth class, 0.0 = the other
G5N_SKU = 1.0

MODEL_DIR = Path(__file__).with_name("selector_models")

NATIVE_FEATURES: Tuple[str, ...] = (
    "a_mn", "b_mn", "n_mn", "kv", "mv", "mb", "wrap", "rep", "rounds_a", "rounds_b", "minreg", "pipe",
    "order", "worder", "lswr", "ld_a8", "ld_b8", "ld_a16", "ld_b16", "ln_rcr", "sku", "unroll",
)  # fmt: skip
FEATURES: Tuple[str, ...] = tuple(_gp.FEATURES) + NATIVE_FEATURES


def active(req: Any) -> bool:
    """Whether the native model serves ``req`` (knob, arch, layout, shipped model + menu)."""
    lay = str(getattr(req, "layout", "RCR") or "RCR").upper()
    return (
        bool(G5N_ON)
        and req.arch in G5N_ARCHES
        and lay in tuple(G5N_LAYOUTS)
        and bool(_gp.load_menu(req.arch))
        and load_model(req.arch) is not None
    )


@lru_cache(maxsize=8)
def load_model(arch: str):
    p = MODEL_DIR / f"g5_native_{arch}.model.txt"
    m = MODEL_DIR / f"g5_native_{arch}.meta.json"
    if not p.exists() or not m.exists():
        return None
    from .selector import TreeEnsemble

    meta = json.loads(m.read_text())
    if tuple(meta.get("features", ())) != FEATURES:
        raise ValueError(f"{p.name}: feature list does not match g5_native.FEATURES")
    return TreeEnsemble.from_text(p.read_text())


def _rounds(
    tile_mn: int, tile_k: int, kv: int, mv: int, threads: int
) -> Tuple[int, float]:
    """(micro-blocks per thread, repeated fraction of the last round) of one MN operand."""
    n = (tile_mn // mv) * (tile_k // kv)
    r = -(-n // threads)
    return r, (r * threads - n) / max(1, n)


def form_features(
    M: int,
    N: int,
    K: int,
    bf16: bool,
    layout: str,
    plan: "_gp.G5Plan",
    tr: Dict[str, Any],
    menu: Dict[str, Dict[str, float]],
    cus: int,
    ln_rcr: Optional[float],
    sku: Optional[float] = None,
) -> List[float]:
    """Feature row (order: FEATURES) of the form of ``plan`` with native trait values ``tr`` (keys vp_mn_kv,
    mn_vec, vp_mn_wrap, sched_strategy, vp_mn_pipe, vp_mn_order, vp_mn_worder, lds_swizzle_r, vp_unroll) in data layout
    ``layout`` ("RCR" = the RCR body itself)."""
    NA = G5N_NA
    base = _gp.plan_features(M, N, K, bf16, plan, menu, cus)
    lay = str(layout).upper()
    a_mn, b_mn = lay != "RCR" and lay[0] == "C", lay != "RCR" and lay[1] == "R"
    tm, tn, tk, wm, wn, _mode, swr = _gp.parse_geo(plan.geo)
    f: Dict[str, float] = {
        "a_mn": float(a_mn),
        "b_mn": float(b_mn),
        "n_mn": float(a_mn) + float(b_mn),
        "ld_a8": float(M % 8 == 0),
        "ld_b8": float(N % 8 == 0),
        "ld_a16": float(M % 16 == 0),
        "ld_b16": float(N % 16 == 0),
        "ln_rcr": float(ln_rcr) if ln_rcr is not None else NA,
        "lswr": float(int(tr.get("lds_swizzle_r", swr))),
        "sku": float(G5N_SKU if sku is None else sku),
        "unroll": float(int(tr.get("vp_unroll", 0))),
    }
    if a_mn or b_mn:
        kv, mv = int(tr.get("vp_mn_kv", 0)), int(tr.get("mn_vec", 8))
        th = 64 * wm * wn
        ra = _rounds(tm, tk, kv, mv, th) if a_mn and kv else (0, 0.0)
        rb = _rounds(tn, tk, kv, mv, th) if b_mn and kv else (0, 0.0)
        f.update(
            kv=float(kv),
            mv=float(mv),
            mb=float(kv * mv),
            wrap=float(int(tr.get("vp_mn_wrap", 0))),
            rep=max(ra[1], rb[1]),
            rounds_a=float(ra[0]),
            rounds_b=float(rb[0]),
            minreg=float(str(tr.get("sched_strategy", "")) == "iterative-minreg"),
            pipe=float(int(tr.get("vp_mn_pipe", 0))),
            order=float(int(tr.get("vp_mn_order", 0))),
            worder=float(int(tr.get("vp_mn_worder", 0))),
        )
    return base + [float(f.get(k, NA)) for k in NATIVE_FEATURES]


def _trait_values(trait: Any) -> Dict[str, Any]:
    return {
        "vp_mn_kv": int(getattr(trait, "vp_mn_kv", 0)),
        "mn_vec": int(getattr(trait, "mn_vec", 8)),
        "vp_mn_wrap": int(getattr(trait, "vp_mn_wrap", 0)),
        "sched_strategy": str(getattr(trait, "sched_strategy", "") or ""),
        "vp_mn_pipe": int(getattr(trait, "vp_mn_pipe", 0)),
        "vp_mn_order": int(getattr(trait, "vp_mn_order", 0)),
        "vp_mn_worder": int(getattr(trait, "vp_mn_worder", 0)),
        "lds_swizzle_r": int(getattr(trait, "lds_swizzle_r", 0)),
        "vp_unroll": int(getattr(trait, "vp_unroll", 0)),
    }


def plan_forms(req: Any, plan: "_gp.G5Plan") -> List[Any]:
    """Every launchable native form (UniversalGemmSpec) of menu plan ``plan`` for the non-RCR ``req``: the
    G5N_MICRO micro-blocks (the lane wrap only when the plain form does not launch) with the dispatcher's native
    trait settings, + the tuned trait set of a tuned (layout, geometry)."""
    from ...instances.common.gemm_universal import (
        DataSpec,
        UniversalGemmSpec,
        is_valid_spec,
        launch_shape_error,
    )
    from .common import R2_VP_MN_SCHED, R2_VP_MN_TRAIT, R2_VP_MN_TUNED, _old_geo_of_menu

    lay = str(req.layout).upper()
    M, N, K = int(req.M), int(req.N), int(req.K)
    dt = str(getattr(req, "dtype", "fp16"))
    dt = dt if dt in ("fp16", "bf16") else "fp16"
    tile, trait = _gp.plan_tile_trait(plan, M, N, K)
    tr0 = replace(
        trait,
        sched_strategy=R2_VP_MN_SCHED.get(lay, trait.sched_strategy),
        **R2_VP_MN_TRAIT,
    )
    data = DataSpec(dtype_a=dt, dtype_b=dt, dtype_c=dt, layout=lay)
    trs = [(tr0, list(G5N_MICRO))]
    for tv in G5N_TRAIT_VARIANTS:
        trs.append((replace(tr0, **dict(tv)), list(G5N_MICRO)))
    old = _old_geo_of_menu(plan.geo)
    tuned = R2_VP_MN_TUNED.get((lay, old)) if (G5N_TUNED_FORMS and old) else None
    if tuned is not None and plan.split == 1:
        trs.append((replace(tr0, **tuned[1]), [tuple(tuned[0])]))
    if G5N_UNROLL_BOTH and int(getattr(tr0, "vp_unroll", 0)):
        trs += [(replace(t_, vp_unroll=0), m_) for t_, m_ in trs]
    out = []
    seen = set()
    for trb, micros in trs:
        for kv, mv in micros:
            for wrap in (0, 1):
                s = UniversalGemmSpec(
                    name="g5n",
                    tile=tile,
                    trait=replace(
                        trb, vp_mn_kv=int(kv), mn_vec=int(mv), vp_mn_wrap=wrap
                    ),
                    data=data,
                    wave_size=64,
                )
                if is_valid_spec(s, arch=req.arch)[0] and not launch_shape_error(
                    s, M, N, K
                ):
                    key = body_key(s)
                    if key not in seen:
                        seen.add(key)
                        out.append(s)
                    break
    return out


def _plan_ln_rcr(M, N, K, bf16, plans, menu, cus, arch) -> List[Optional[float]]:
    """The plan model's raw ln-time predictions of ``plans`` (the ``ln_rcr`` feature; None without the model)."""
    pm = _gp.load_model(arch)
    if pm is None or not plans:
        return [None] * len(plans)
    ys = pm.predict([_gp.plan_features(M, N, K, bf16, p, menu, cus) for p in plans])
    return [float(y) for y in ys]


def ln_rcr_of(t: Optional[float], plan: "_gp.G5Plan") -> Optional[float]:
    """The plan model's raw ln-time prediction of ``plan`` from its ranked time ``t`` (``g5_plan.rank_plans``
    scales an unrolled plan by ``G5P_VP_UNROLL_SCALE``; the feature is the unscaled prediction).
    """
    if not t or t <= 0:
        return None
    return math.log(t / (_gp.G5P_VP_UNROLL_SCALE if _gp.plan_unroll(plan) else 1.0))


def body_key(spec: Any) -> tuple:
    """Identity of a GEMM body independent of the spec name (tile, trait, data)."""
    return (repr(spec.tile), repr(spec.trait), repr(spec.data))


_RANK_CACHE: dict = {}


def _knob_key() -> tuple:
    from . import common as _c

    return (
        G5N_ON, tuple(G5N_LAYOUTS), G5N_TOP, tuple(G5N_MICRO), G5N_TUNED_FORMS, G5N_UNROLL_BOTH, G5N_SKU,
        G5N_SERIAL, tuple(tuple(sorted(dict(t).items())) for t in G5N_TRAIT_VARIANTS),
        tuple(sorted(_c.R2_VP_MN_TRAIT.items())), tuple(sorted(_c.R2_VP_MN_SCHED.items())),
        len(_c.R2_VP_MN_TUNED), _gp.G5P_SPLITS, _gp.G5P_SERIAL_MODES, _gp.G5P_SPLIT_MAX_WAVES,
        _gp.G5P_PRESELECT, _gp.G5P_MODEL, _gp.G5P_VP_UNROLL_MAX_ACC,
    )  # fmt: skip


def rank_native(req: Any) -> List[Tuple[float, "_gp.G5Plan", Any]]:
    """(predicted us, plan, native form spec) of the scored native forms of ``req``, cheapest first (empty when
    :func:`active` does not hold)."""
    if not active(req):
        return []
    lay = str(req.layout).upper()
    M, N, K = int(req.M), int(req.N), int(req.K)
    dt = str(getattr(req, "dtype", "fp16"))
    key = (M, N, K, dt, req.arch, lay, _knob_key())
    hit = _RANK_CACHE.get(key)
    if hit is not None:
        return list(hit)
    from .common import _g5p_ranked

    rreq = replace(req, layout="RCR", trans_a=False, trans_b=True)
    # the plan model's ranked RCR plans; a serial split-K plan contributes its atomic-reduction twin (and itself
    # with G5N_SERIAL)
    plans: List["_gp.G5Plan"] = []
    # rank without the RCR main-loop traits' scale (g5_plan.G5P_MAINLOOP_SCALE): the native forms run
    # without them (the model was trained on the unscaled ranking)
    for _t, p in _g5p_ranked(rreq, mainloop=False):
        if p.geo == "skinny":
            continue
        for q in ([p] if (p.serial and G5N_SERIAL) else []) + [
            _gp.G5Plan(p.geo, p.split, 0) if p.serial else p
        ]:
            if q not in plans:
                plans.append(q)
    plans = plans[: max(1, int(G5N_TOP))]
    menu = _gp.load_menu(req.arch)
    cus = _gp.G5P_CUS.get(req.arch, 304)
    bf16 = dt == "bf16"
    lnr = _plan_ln_rcr(M, N, K, bf16, plans, menu, cus, req.arch)
    rows, meta = [], []
    for p, ln_rcr in zip(plans, lnr):
        for f in plan_forms(req, p):
            rows.append(
                form_features(
                    M, N, K, bf16, lay, p, _trait_values(f.trait), menu, cus, ln_rcr
                )
            )
            meta.append((p, f))
    out = []
    if rows:
        ys = load_model(req.arch).predict(rows)
        out = sorted(
            ((math.exp(y), p, f) for y, (p, f) in zip(ys, meta)),
            key=lambda x: (x[0], x[1].geo, x[1].split, repr(body_key(x[2]))),
        )
    if len(_RANK_CACHE) > 4096:
        _RANK_CACHE.clear()
    _RANK_CACHE[key] = tuple(out)
    return out


def best_form(req: Any) -> Optional[Tuple[float, "_gp.G5Plan", Any]]:
    """The cheapest scored native form of ``req`` (None: model inactive / no form)."""
    r = rank_native(req)
    return r[0] if r else None


def form_for(req: Any, spec: Any) -> Optional[Any]:
    """The best-scored native form of the plan whose body is ``spec`` (its menu plan: tile, waves, interleave
    mode, LDS swizzle row shift, split-K degree and reduction; :func:`spec_plan`) among :func:`rank_native`'s
    forms, or None. A form may change trait fields of the body (a tuned trait set's LDS swizzle row shift), so
    forms are matched by their plan, not by their own trait."""
    want = spec_plan(spec, req.arch, exact=True)
    if want is None:
        return None
    for _u, p, f in rank_native(req):
        if p == want:
            return f
    return None


def predicted_us(req: Any, spec: Any) -> Optional[float]:
    """The model's time of the native form ``spec`` when it is one of :func:`rank_native`'s forms."""
    name = body_key(spec)
    for u, _p, f in rank_native(req):
        if body_key(f) == name:
            return u
    return None


def spec_plan(spec: Any, arch: str, exact: bool = False) -> Optional["_gp.G5Plan"]:
    """The plan-menu plan of a g5 body ``spec`` (any data layout): the menu geometry of its tile, waves,
    interleave mode and LDS swizzle row shift (another shift of the same body when the exact one is not on the
    menu unless ``exact``; the geometry name even when the body is off the menu), its split-K degree and
    reduction (None: not a g5 body)."""
    tr = getattr(spec, "trait", None)
    if tr is None or not int(getattr(tr, "vgpr_prefetch", 0)):
        return None
    t = spec.tile
    menu = _gp.load_menu(arch)
    base = f"{t.tile_m}x{t.tile_n}x{t.tile_k}_w{t.warp_m}x{t.warp_n}_m{int(tr.vgpr_prefetch)}"
    key = f"{base}_r{int(tr.lds_swizzle_r)}"
    if key not in menu and not exact:
        key = next((f"{base}_r{r}" for r in (0, 1, 2) if f"{base}_r{r}" in menu), key)
    return _gp.G5Plan(key, int(tr.split_k), int(getattr(tr, "split_k_serial", 0)))


def predict_spec(req: Any, spec: Any) -> Optional[float]:
    """The model's time (us) of the g5 body ``spec`` (RCR or a native form, its own data layout) at ``req``'s
    shape and dtype, or None (no model / not a g5 body)."""
    if not bool(G5N_ON) or req.arch not in G5N_ARCHES:
        return None
    if str(getattr(spec.data, "layout", "RCR")).upper() == str(req.layout).upper():
        # one of the ranked forms (its plan, not the trait-derived one: a tuned form's swizzle shift)
        u = predicted_us(req, spec)
        if u is not None:
            return u
    model = load_model(req.arch)
    plan = spec_plan(spec, req.arch)
    if model is None or plan is None:
        return None
    menu = _gp.load_menu(req.arch)
    cus = _gp.G5P_CUS.get(req.arch, 304)
    M, N, K = int(req.M), int(req.N), int(req.K)
    bf16 = str(getattr(req, "dtype", "fp16")) == "bf16"
    ln_rcr = _plan_ln_rcr(M, N, K, bf16, [plan], menu, cus, req.arch)[0]
    lay = str(getattr(spec.data, "layout", "RCR")).upper()
    row = form_features(
        M, N, K, bf16, lay, plan, _trait_values(spec.trait), menu, cus, ln_rcr
    )
    u = math.exp(model.predict([row])[0])
    if R3M_G5N_MAINLOOP_SCALE and (
        int(getattr(spec.trait, "vp_pair_reads", 0)) > 0
        or int(getattr(spec.trait, "vp_mfma_first", 0)) > 0
    ):
        # an RCR body running the main-loop traits (g5_plan.G5P_MAINLOOP; the model was
        # trained without them) gets the plan model's main-loop time factor
        u *= float(_gp.G5P_MAINLOOP_SCALE)
    return u

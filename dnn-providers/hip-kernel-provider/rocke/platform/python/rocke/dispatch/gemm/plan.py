# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Plan-level layout choice for NN / TN / TT GEMM: native vs transpose pre-pass.

A non-RCR request can run as

* ``native``: the dispatcher's NN / TN / TT kernel reads the MN-contiguous
  operands in place (``dispatch_gemm_fp16`` / ``dispatch_gemm_bf16``);
* ``prepass``: rocKE ``transpose2d`` kernels write each MN-contiguous operand
  K-contiguous into a workspace, then the RCR (NT) dispatcher pick runs;
* ``prepass_a`` / ``prepass_b`` (TN only, the gfx942 chooser): one operand is
  transposed and the other native layout runs (NN after A, TT after B).

:func:`plan_gemm` (``plan_gemm_fp16`` / ``plan_gemm_bf16``) returns a
:class:`GemmPlan` with the chosen steps. Two choosers, by arch:

* the native selector chooser (gfx950): the native-form selector model's time
  prediction of the native pick
  (``exp(score + ln roofline)``) vs the prediction of the RCR pick plus a
  transpose cost model (``TRANSPOSE_COST``); the pre-pass must be predicted
  faster by ``GEMM_PLAN_PREPASS_MARGIN`` (log-time); only shapes the transpose
  kernel serves without padding (both extents multiples of its 64x64 tile,
  grid within the HIP limits) are eligible;
* the g5 cost-model chooser (gfx942; :func:`gemm_plan`, arches
  :data:`R2_PLAN_ARCHES`): the g5 register-staged
  body's fitted cost models (``common.g5_vgpr_cost`` / ``r2_native_cost`` /
  ``r2_native_fit``) plus a transpose cost model with a slow-copy rate for
  extents that are not multiples of 64, the partial TN pre-pass, and a slow-
  fallback factor for native picks without a model. On gfx942 the native
  selector model does not know the g5 native forms, so this chooser serves it.

Knob ``ROCKE_GEMM_PLAN_PREPASS`` / :data:`GEMM_PLAN_PREPASS`: ``auto``
(default), ``off`` (always native), ``always`` (full pre-pass wherever the
transpose kernel is eligible). Python-only planning (no kernel emission): the
plan names existing kernels.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Optional, Tuple

from . import selector as _sel

GEMM_PLAN_PREPASS_ENV = "ROCKE_GEMM_PLAN_PREPASS"
GEMM_PLAN_PREPASS = "auto"
GEMM_PLAN_PREPASS_POLICIES = ("auto", "off", "always")
# log-time margin the pre-pass must be predicted faster by (model + cost-model
# noise; the selector margins use a smaller noise floor)
GEMM_PLAN_PREPASS_MARGIN = 0.05
# transpose2d cost per arch: (launch floor in microseconds, sustained bytes per
# microsecond); fitted on timed rocKE transpose2d arms (64x64 tile, vec 8, lds
# pad 8) in log space
TRANSPOSE_COST: Dict[str, Tuple[float, float]] = {
    "gfx950": (1.9, 6.05e6),
    "gfx942": (1.9, 4.80e6),
}
TRANSPOSE_TILE = 64
_HIP_GRID_YZ_MAX = 65535
# native-selector chooser: a native gemm_skinny pick on an MN leading
# dimension that is not a multiple of 8 (no vector MN loads; transpose2d cannot
# serve the extent) with at least this many 128x128 output tiles takes the
# pre-pass through the caller's copy (TransposeStep.spec None) and the RCR pick.
# Chosen from paired gfx950 odd-ld non-NT timings (same process), where the
# pre-pass was faster on most large-tile-count shapes. Per arch; 0 / absent = off.
GEMM_PLAN_ODD_SKINNY_PREPASS_TILES: Dict[str, int] = {"gfx950": 256}

# --- the g5 cost-model chooser (gfx942) -------------------------------------
# the chooser on these targets (elsewhere: the native selector chooser)
R2_PLAN = True
R2_PLAN_ARCHES = ("gfx942",)
# transpose cost (microseconds): fixed + bytes moved (read + write) / rate;
# rocKE transpose2d 64x64 v8 pad8 (both sides multiples of 64), fitted on gfx942
# timings over a wide range of operand sizes; other sizes: the caller's copy
# (torch copy_ of the transposed view), a much slower rate
R2_TP_FIXED_US = 1.7
R2_TP_BYTES_PER_US = 4.8e6
R2_TP_SLOW_FIXED_US = 10.5
R2_TP_SLOW_BYTES_PER_US = 0.78e6
# transpose2d tile sizes the plan may use (64: the calibrated tile)
R2_TP_TILES = (64,)
# pre-pass chosen when its estimate x margin is below the native estimate
# (1.2: the model under-predicts the pre-pass's separate launches at small
# sizes; chosen by a grid search over gfx942 non-NT shapes, where it picks the
# better plan in nearly every case)
R2_PLAN_MARGIN = 1.2
# the pre-pass's RCR GEMM estimated with r2_native_fit.R2_RCR_LIN (the native
# fit's calibration) instead of the g5 dispatch model
R2_PLAN_FITTED_RCR = True
# native picks with no cost model that are known to be slow on these targets
# (the VGPR-staged transposing fallback): their estimate is the RCR GEMM's x
# this factor (the fallback is several times slower than the pre-pass on these
# shapes)
R2_PLAN_SLOW_NATIVE = ("universal_fallback", "cdna_cshuffle")
R2_PLAN_SLOW_FACTOR = 8.0
# the gemm_skinny fallback configuration counts as a slow native pick when
# min(M, N) is at least this (gfx942 non-NT mid / big shapes are far from the
# best available kernel on the skinny fallback; 0 = never)
R2_PLAN_SLOW_SKINNY_FALLBACK_MIN_MN = 1024
# the RCR GEMM of the pre-pass is the g5 plan model's pick (g5_plan, the gfx942
# NT geometry menu), which the native forms' fitted models (R2_RCR_LIN, a
# smaller geometry menu) do not cover. Its estimate:
# "ratio" (default) = the g5 plan model's prediction of the pick, rescaled by
# fitted / predicted time of the cheapest fitted old-menu plan that is also a
# menu geometry (the two models were timed differently; only their ratio is
# used); "abs" = the g5 plan model's prediction as is; "off" = the fitted
# estimate of the fixed-geometry g5 plan (an upper bound: the pick is at least as
# fast by the plan model)
R2_PLAN_G5P_RCR = "ratio"
# a slow native fallback against an RCR pick the model does not cover (e.g.
# gemm_skinny): take the pre-pass (False: keep the native pick)
R2_PLAN_FAST_RCR_OVER_SLOW_NATIVE = True
# TN (both operands MN-contiguous): also consider transposing ONE operand and
# running the other native layout ("prepass_a": A transposed -> NN form,
# "prepass_b": B transposed -> TT form); False: full pre-pass or native only
R2_PLAN_PARTIAL = True
# g5 bodies (native forms, the pre-pass's RCR body, the TN
# partial forms) estimated by the native-form model (g5_native.predict_spec)
# when it is active for the request -- one model for both sides of the choice;
# the pre-pass is then taken when it is predicted R3_PLAN_MARGIN x faster
R3_PLAN_G5N = True
R3_PLAN_MARGIN = 1.2


@dataclass(frozen=True)
class TransposeStep:
    """Transpose operand ``operand`` (stored ``[rows, cols]`` = ``[K, M|N]``)
    into a ``[cols, rows]`` workspace with ``spec`` on ``grid`` (``spec`` None:
    the caller's copy; ``grid`` None: not computed by the g5 chooser)."""

    operand: str
    rows: int
    cols: int
    spec: Any
    grid: Optional[Tuple[int, int, int]] = None

    @property
    def bytes_moved(self) -> int:
        return 2 * 2 * self.rows * self.cols


# the g5 chooser's name for the same step
GemmTranspose = TransposeStep


@dataclass(frozen=True)
class GemmPlan:
    # "native" | "prepass" (every MN operand transposed, the RCR pick) |
    # "prepass_a" / "prepass_b" (TN: only A / B transposed, the NN / TT pick)
    kind: str
    gemm: Any  # DispatchResult of the GEMM step
    transposes: Tuple[TransposeStep, ...] = ()
    predicted_us: Dict[str, float] = field(default_factory=dict)
    explanation: Tuple[str, ...] = ()
    # the request the GEMM step serves (None: the caller's request when native)
    request: Any = None
    # the g5 chooser's estimates of the chosen / the other plan
    est_us: Optional[float] = None
    alt_est_us: Optional[float] = None

    @property
    def workspace_bytes(self) -> int:
        """Bytes of the K-contiguous operand copies a pre-pass needs."""
        return sum(2 * t.rows * t.cols for t in self.transposes)

    @property
    def dispatch(self) -> Any:
        return self.gemm

    @property
    def reason(self) -> str:
        return "; ".join(self.explanation)


def plan_policy() -> str:
    v = os.environ.get(GEMM_PLAN_PREPASS_ENV, "").strip().lower()
    p = v or GEMM_PLAN_PREPASS
    if p not in GEMM_PLAN_PREPASS_POLICIES:
        raise ValueError(
            f"{GEMM_PLAN_PREPASS_ENV}={p!r}: expected one of {GEMM_PLAN_PREPASS_POLICIES}"
        )
    return p


def _rcr_request(req: Any) -> Any:
    return _sel._rcr_view(req)


def transpose_steps(req: Any) -> Optional[Tuple[TransposeStep, ...]]:
    """The pre-pass transposes of ``req`` (``()`` for RCR), or None when the
    transpose kernel cannot serve an operand without padding."""
    from ...instances.common.gemm_mn_operand import mn_layout_letters
    from ...instances.common.transpose import Transpose2DSpec, transpose2d_grid

    a_mn, b_mn = mn_layout_letters(str(req.layout))
    dt = "bf16" if str(req.dtype) in ("bf16", "bfloat16") else "f16"
    spec = Transpose2DSpec(
        tile_m=TRANSPOSE_TILE, tile_n=TRANSPOSE_TILE, vec=8, dtype=dt, lds_pad=8
    )
    steps = []
    for name, is_mn, cols in (("A", a_mn, int(req.M)), ("B", b_mn, int(req.N))):
        if not is_mn:
            continue
        rows = int(req.K)
        if rows % TRANSPOSE_TILE or cols % TRANSPOSE_TILE:
            return None
        grid = transpose2d_grid(rows, cols, spec)
        if grid[1] > _HIP_GRID_YZ_MAX or grid[2] > _HIP_GRID_YZ_MAX:
            return None
        steps.append(TransposeStep(name, rows, cols, spec, tuple(int(g) for g in grid)))
    return tuple(steps)


def transpose_cost_us(arch: str, steps) -> Optional[float]:
    """The native selector chooser's transpose cost (us) of ``steps``."""
    c = TRANSPOSE_COST.get(str(arch))
    if c is None:
        return None
    c0, bw = c
    return sum(c0 + t.bytes_moved / bw for t in steps)


def predicted_us(req: Any, spec: Any) -> Optional[float]:
    """The native selector model's time prediction (us) of ``spec`` for the
    problem of ``req`` (any layout; the spec's own layout is a feature)."""
    from ...instances.common.gemm_universal import UniversalGemmSpec

    if not isinstance(spec, UniversalGemmSpec):
        return None
    m = _sel.load_model(str(req.arch), "native")
    if m is None:
        return None
    dtype = "bf16" if str(req.dtype) in ("bf16", "bfloat16") else "fp16"
    M, N, K = int(req.M), int(req.N), int(req.K)
    s = m.score(dtype, M, N, K, [_sel.flatten_spec(spec)])[0]
    return math.exp(s + _sel._ln(_sel.roofline_cycles(str(req.arch), M, N, K)))


def tail_split_rows(req: Any, spec: Any, cus: Optional[int] = None) -> int:
    """Rows of the partial-wave tail of ``spec`` on ``req`` (0: none): the
    head keeps the whole tile rows that fit in the full waves of the grid
    (``cus`` workgroups per wave, default the arch's CU count), the tail is
    the rest of M. A plan can run the tail as its own launch (the
    dispatcher's pick for the tail's ``rows x N x K``, whose smaller tiles
    spread the remainder over the device) instead of a mostly idle last wave."""
    from ...helpers.split_k import arch_num_cus

    t = spec.tile
    M, N = int(req.M), int(req.N)
    P = int(cus or arch_num_cus(str(req.arch)))
    tn = -(-N // t.tile_n)
    R = -(-M // t.tile_m)
    total = R * tn
    W = total // P
    if W < 1 or total % P == 0:
        return 0
    Rh = (W * P) // tn
    if Rh <= 0 or Rh >= R:
        return 0
    return M - Rh * t.tile_m


def plan_gemm(req: Any, dispatch: Callable[[Any], Any]) -> GemmPlan:
    """The plan for ``req``; ``dispatch`` is the family's dispatch function
    (``dispatch_gemm_fp16`` / ``dispatch_gemm_bf16``)."""
    if _sel._rcr(req):
        return GemmPlan("native", dispatch(req), (), {}, ("RCR request",), req)
    pol = plan_policy()
    if pol == "off":
        return GemmPlan(
            "native", dispatch(req), (), {}, (f"{GEMM_PLAN_PREPASS_ENV}=off",), req
        )
    if pol == "auto" and R2_PLAN and str(req.arch) in R2_PLAN_ARCHES:
        return gemm_plan(req, dispatch)
    native = dispatch(req)
    steps = transpose_steps(req)
    if steps is None and _odd_skinny_prepass(req, native):
        rreq = _rcr_request(req)
        cps = _copy_steps(req)
        return GemmPlan(
            "prepass",
            dispatch(rreq),
            cps,
            {},
            (
                "native gemm_skinny on an odd MN leading dimension with many tiles: "
                "pre-pass through the caller's copy (GEMM_PLAN_ODD_SKINNY_PREPASS_TILES)",
            ),
            rreq,
        )
    if steps is None:
        return GemmPlan(
            "native",
            native,
            (),
            {},
            ("pre-pass not eligible (transpose tiles)",),
            req,
        )
    rreq = _rcr_request(req)
    rcr = dispatch(rreq)
    if pol == "always":
        return GemmPlan(
            "prepass", rcr, steps, {}, (f"{GEMM_PLAN_PREPASS_ENV}=always",), rreq
        )
    t_nat = predicted_us(req, native.spec)
    t_rcr = predicted_us(rreq, rcr.spec)
    t_tp = transpose_cost_us(str(req.arch), steps)
    pred = {"native": t_nat, "rcr_gemm": t_rcr, "transposes": t_tp}
    if None in (t_nat, t_rcr, t_tp):
        return GemmPlan(
            "native", native, (), pred, ("no time model for one of the plans",), req
        )
    t_pp = t_rcr + t_tp
    pred["prepass"] = t_pp
    if math.log(t_nat) - math.log(t_pp) > GEMM_PLAN_PREPASS_MARGIN:
        return GemmPlan(
            "prepass",
            rcr,
            steps,
            pred,
            (f"pre-pass predicted {t_pp:.1f} us vs native {t_nat:.1f} us",),
            rreq,
        )
    return GemmPlan(
        "native",
        native,
        (),
        pred,
        (f"native predicted {t_nat:.1f} us vs pre-pass {t_pp:.1f} us",),
        req,
    )


def _odd_skinny_prepass(req: Any, native: Any) -> bool:
    lim = int(GEMM_PLAN_ODD_SKINNY_PREPASS_TILES.get(str(req.arch), 0) or 0)
    if not lim or type(native.spec).__name__ != "SkinnyGemmSpec":
        return False
    from ...instances.common.gemm_mn_operand import mn_layout_letters

    a_mn, b_mn = mn_layout_letters(str(req.layout))
    M, N = int(req.M), int(req.N)
    odd = any(x % 8 for x, m in ((M, a_mn), (N, b_mn)) if m)
    return odd and -(-M // 128) * -(-N // 128) >= lim


def _copy_steps(req: Any) -> Tuple[TransposeStep, ...]:
    """Pre-pass steps through the caller's copy (no transpose2d spec)."""
    from ...instances.common.gemm_mn_operand import mn_layout_letters

    a_mn, b_mn = mn_layout_letters(str(req.layout))
    return tuple(
        TransposeStep(name, int(req.K), cols, None)
        for name, is_mn, cols in (("A", a_mn, int(req.M)), ("B", b_mn, int(req.N)))
        if is_mn
    )


def plan_gemm_fp16(req: Any) -> GemmPlan:
    from .fp16_rcr import dispatch_gemm_fp16

    return plan_gemm(req, dispatch_gemm_fp16)


def plan_gemm_bf16(req: Any) -> GemmPlan:
    from .bf16_rcr import dispatch_gemm_bf16

    return plan_gemm(req, dispatch_gemm_bf16)


# ---------------------------------------------------------------------------
# The g5 cost-model chooser (gfx942)
# ---------------------------------------------------------------------------


def _tp_spec(rows: int, cols: int, dtype: str):
    from ...instances.common.transpose import Transpose2DSpec

    dt = "f16" if dtype == "fp16" else "bf16"
    for t in R2_TP_TILES:
        if rows % t == 0 and cols % t == 0:
            return Transpose2DSpec(
                tile_m=t, tile_n=t, vec=8, dtype=dt, lds_pad=8, grid_order="row"
            )
    return None


def r2_transpose_cost_us(rows: int, cols: int, fast: bool = True) -> float:
    """The g5 chooser's modelled time of one 16-bit transpose of a ``rows x
    cols`` operand (``fast``: rocKE transpose2d; else the caller's copy)."""
    nbytes = 2.0 * 2.0 * rows * cols
    if fast:
        return R2_TP_FIXED_US + nbytes / R2_TP_BYTES_PER_US
    return R2_TP_SLOW_FIXED_US + nbytes / R2_TP_SLOW_BYTES_PER_US


def _dispatch(req: Any):
    from . import dispatch_gemm_bf16, dispatch_gemm_fp16

    return (dispatch_gemm_fp16 if req.dtype == "fp16" else dispatch_gemm_bf16)(req)


def _g5_est(req: Any, res) -> Optional[float]:
    """Modelled time of a dispatched g5 body (None: not a g5 body)."""
    from ...helpers.split_k import arch_num_cus
    from .common import (
        G5_VGPR_A_MIN,
        G5_VGPR_GEOS,
        G5_VGPR_SPLIT_COST,
        G5_VGPR_TRAFFIC_BW,
        g5_vgpr_cost,
        r2_native_cost,
    )
    from .r2_native_fit import R2_RCR_LIN

    spec = res.spec
    tr = getattr(spec, "trait", None)
    if tr is None or not int(getattr(tr, "vgpr_prefetch", 0)):
        return None
    t = spec.tile
    geo = None
    for name, g in G5_VGPR_GEOS.items():
        if (g[0], g[1], g[2], g[3], g[4]) == (
            t.tile_m,
            t.tile_n,
            t.tile_k,
            t.warp_m,
            t.warp_n,
        ):
            geo = name
            break
    if geo is None:
        # a native form of a g5 plan-model body (outside the fitted menu)
        return _g5p_native_est(req, res)
    M, N, K = int(req.M), int(req.N), int(req.K)
    sk = int(tr.split_k)
    est = g5_vgpr_cost(M, N, K, req.arch, geo, sk)
    lay = str(req.layout).upper()
    if lay != "RCR":
        return r2_native_cost(req, geo, sk, spec, est)
    return _geo_est(req, geo, sk)


def _geo_est(req: Any, geo: str, sk: int) -> Optional[float]:
    """Modelled time of the RCR g5 body of geometry ``geo`` (fixed menu,
    ``common.G5_VGPR_GEOS``) with split ``sk``."""
    from ...helpers.split_k import arch_num_cus
    from .common import (
        G5_VGPR_A_MIN,
        G5_VGPR_GEOS,
        G5_VGPR_SPLIT_COST,
        G5_VGPR_TRAFFIC_BW,
        g5_vgpr_cost,
    )
    from .r2_native_fit import R2_RCR_LIN

    M, N, K = int(req.M), int(req.N), int(req.K)
    est = g5_vgpr_cost(M, N, K, req.arch, geo, sk)
    ab = R2_RCR_LIN.get(geo) if R2_PLAN_FITTED_RCR else None
    if ab is None:
        return est
    # the RCR body from the same calibration as the native fit
    tm, tn, tk = G5_VGPR_GEOS[geo][0], G5_VGPR_GEOS[geo][1], G5_VGPR_GEOS[geo][2]
    tiles = -(-M // tm) * -(-N // tn)
    ks = -(-K // (tk * sk)) * tk
    cost = -(-(tiles * sk) // arch_num_cus(req.arch)) * (
        max(ab[0], G5_VGPR_A_MIN) + ab[1] * ks
    )
    cost = max(cost, tiles * sk * (tm + tn) * ks * 2 / G5_VGPR_TRAFFIC_BW)
    if sk > 1:
        fixed, per_byte = G5_VGPR_SPLIT_COST
        cost += fixed + per_byte * M * N * 4 * sk
    return cost


def _old_menu_key(geo: str) -> str:
    from .common import G5_VGPR_GEOS

    g = G5_VGPR_GEOS[geo]
    return f"{g[0]}x{g[1]}x{g[2]}_w{g[3]}x{g[4]}_m{g[9]}_r{g[10]}"


def _g5p_cal(req: Any) -> Optional[float]:
    """Calibration of the g5 plan model's predictions for the RCR request ``req``
    (:data:`R2_PLAN_G5P_RCR`): 1.0 for "abs"; for "ratio" fitted / predicted time
    of the cheapest fitted old-menu plan that is also a menu geometry; None when
    off or no plan calibrates."""
    from .common import _g5p_on

    mode = str(R2_PLAN_G5P_RCR).lower()
    if mode in ("off", "false", "0") or not _g5p_on(req):
        return None
    if mode == "abs":
        return 1.0
    from . import g5_plan as _gp
    from .common import g5_vgpr_plans

    menu = _gp.load_menu(req.arch)
    M, N, K = int(req.M), int(req.N), int(req.K)
    for _c, geo, s in g5_vgpr_plans(M, N, K, req.arch):
        key = _old_menu_key(geo)
        if key not in menu:
            continue
        fitted = _geo_est(req, geo, s)
        # the fitted estimate is of the body without the main-loop traits: unscaled
        pr = _gp.rank_plans(
            M, N, K, str(req.dtype), req.arch, [_gp.G5Plan(key, s)], mainloop=False
        )
        pred = next((float(t) for t, p in pr if p.geo == key and p.split == s), None)
        if fitted is None or not pred:
            continue
        return fitted / pred
    return None


def _g5p_plan_pred(rreq: Any, spec: Any) -> Optional[float]:
    """The g5 plan model's predicted time of the ranked plan whose body is
    ``spec`` (tile, waves, mode, swizzle, split-K; any data layout), or None."""
    from .common import _g5p_ranked
    from .g5_plan import plan_tile_trait

    t, tr = spec.tile, spec.trait
    want = (t.tile_m, t.tile_n, t.tile_k, t.warp_m, t.warp_n, int(tr.vgpr_prefetch),
            int(tr.lds_swizzle_r), int(tr.split_k), int(getattr(tr, "split_k_serial", 0)))  # fmt: skip
    M, N, K = int(rreq.M), int(rreq.N), int(rreq.K)
    # a body that runs the main-loop traits (RCR only) is ranked with
    # their scale; a native form without them is not
    ml = (
        int(getattr(tr, "vp_pair_reads", 0)) > 0
        or int(getattr(tr, "vp_mfma_first", 0)) > 0
    )
    for tm, p in _g5p_ranked(rreq, mainloop=ml):
        if p.geo == "skinny":
            continue
        pt, ptr = plan_tile_trait(p, M, N, K)
        have = (pt.tile_m, pt.tile_n, pt.tile_k, pt.warp_m, pt.warp_n, int(ptr.vgpr_prefetch),
                int(ptr.lds_swizzle_r), int(ptr.split_k), int(getattr(ptr, "split_k_serial", 0)))  # fmt: skip
        if have == want:
            return float(tm)
    return None


def _g5p_est(req: Any, res) -> Optional[float]:
    """The pre-pass RCR estimate from the g5 plan model (:data:`R2_PLAN_G5P_RCR`)
    for the dispatched RCR pick (its g5 body or the gemm_skinny plan it
    ranked), or None (model off for ``req`` / knob off / no calibration)."""
    from .common import _g5p_ranked

    if str(getattr(req, "layout", "RCR")).upper() != "RCR":
        return None
    cal = _g5p_cal(req)
    if cal is None:
        return None
    sk = type(res.spec).__name__ == "SkinnyGemmSpec"
    tr = getattr(res.spec, "trait", None)
    if not sk and (tr is None or not int(getattr(tr, "vgpr_prefetch", 0))):
        return None
    pick = next(
        (float(t) for t, p in _g5p_ranked(req) if (p.geo == "skinny") == sk), None
    )
    return None if pick is None else pick * cal


def _g5p_native_est(req: Any, res) -> Optional[float]:
    """Modelled time of a native (NN / TN / TT) g5 form whose body is a plan of
    the g5 plan model (common.MERGE2_NATIVE_G5P): the calibrated plan-model time
    of its RCR body x common.r2_native_cost_factor."""
    from .common import _r2_lay, r2_native_cost_factor

    lay = _r2_lay(req)
    if lay == "RCR":
        return None
    rreq = replace(req, layout="RCR", trans_a=False, trans_b=True)
    cal = _g5p_cal(rreq)
    if cal is None:
        return None
    pred = _g5p_plan_pred(rreq, res.spec)
    if pred is None:
        return None
    return pred * cal * r2_native_cost_factor(lay, res.spec)


def _is_vgpr(res) -> bool:
    tr = getattr(res.spec, "trait", None)
    return tr is not None and bool(int(getattr(tr, "vgpr_prefetch", 0)))


def _slow_native(req: Any, cname: str) -> bool:
    if any(s in cname for s in R2_PLAN_SLOW_NATIVE):
        return True
    lim = int(R2_PLAN_SLOW_SKINNY_FALLBACK_MIN_MN)
    return bool(
        lim and "skinny_fallback" in cname and min(int(req.M), int(req.N)) >= lim
    )


def _g5n_plan_on(req: Any) -> bool:
    """Whether the native-form model estimates the chooser's g5 plans for req."""
    from . import g5_native as _gn

    return bool(R3_PLAN_G5N) and _gn.active(req)


def _g5n_est(req: Any, res) -> Optional[float]:
    """The native-form model's time of a dispatched g5 body (None: not a g5
    body / no model)."""
    from . import g5_native as _gn

    if not _is_vgpr(res):
        return None
    return _gn.predict_spec(req, res.spec)


def _rcr_est(rreq: Any, rcr) -> Optional[float]:
    """Modelled time of the dispatched RCR pick ``rcr`` of ``rreq``."""
    est_r = _g5p_est(rreq, rcr)
    if est_r is None:
        est_r = _g5_est(rreq, rcr)
    if _is_vgpr(rcr) and str(R2_PLAN_G5P_RCR).lower() != "abs":
        # a g5 plan-model pick (outside the fitted menu): at most the fitted
        # estimate of the fixed-geometry plan (the plan model ranks its pick ahead)
        from .common import g5_vgpr_plan

        old = g5_vgpr_plan(int(rreq.M), int(rreq.N), int(rreq.K), rreq.arch)
        old_est = _geo_est(rreq, *old) if old is not None else None
        if old_est is not None:
            est_r = old_est if est_r is None else min(est_r, old_est)
    return est_r


def _plan(kind, preq, res, tps, est, alt, why) -> GemmPlan:
    pred = {}
    if est is not None:
        pred["chosen"] = est
    if alt is not None:
        pred["other"] = alt
    return GemmPlan(kind, res, tuple(tps), pred, (why,), preq, est, alt)


def gemm_plan(req: Any, dispatch: Optional[Callable[[Any], Any]] = None) -> GemmPlan:
    """The g5 cost-model chooser: the cheaper of the native kernel and the
    transpose pre-pass for ``req`` (see the module doc)."""
    dsp = dispatch if dispatch is not None else _dispatch
    lay = str(getattr(req, "layout", "RCR")).upper()
    nat = dsp(req)
    if lay == "RCR" or not R2_PLAN or req.arch not in R2_PLAN_ARCHES:
        return _plan("native", req, nat, (), None, None, "RCR / chooser off")
    a_mn, b_mn = lay[0] == "C", lay[1] == "R"
    rreq = replace(req, layout="RCR", trans_a=False, trans_b=True)
    try:
        rcr = dsp(rreq)
    except ValueError as e:  # pragma: no cover - RCR refusing what NN takes
        return _plan("native", req, nat, (), None, None, f"no RCR pick ({e})")
    M, N, K = int(req.M), int(req.N), int(req.K)
    tps = []
    tp_us = 0.0
    for is_mn, nm, cols in ((a_mn, "A", M), (b_mn, "B", N)):
        if is_mn:
            sp = _tp_spec(K, cols, req.dtype)
            tps.append(TransposeStep(nm, K, cols, sp))
            tp_us += r2_transpose_cost_us(K, cols, fast=sp is not None)
    est_r = _rcr_est(rreq, rcr)
    est_n = _g5_est(req, nat)
    g5n = _g5n_plan_on(req)
    margin = R2_PLAN_MARGIN
    use_m = False
    if g5n:
        mn, mr = _g5n_est(req, nat), _g5n_est(rreq, rcr)
        if mn is not None and mr is not None:
            est_n, est_r, margin, use_m = mn, mr, R3_PLAN_MARGIN, True
    cname = getattr(nat.candidate, "name", "")
    rname = getattr(rcr.candidate, "name", "")
    slow_n = _slow_native(req, cname)
    slow_r = _slow_native(rreq, rname)
    if est_n is None and est_r is not None and slow_n:
        est_n = est_r * R2_PLAN_SLOW_FACTOR
    if est_r is None and slow_n and not slow_r and R2_PLAN_FAST_RCR_OVER_SLOW_NATIVE:
        # no model for the RCR pick (e.g. gemm_skinny), but the native pick is
        # the slow transposing fallback: the pre-pass
        return _plan(
            "prepass",
            rreq,
            rcr,
            tps,
            None,
            None,
            f"native {cname} is a slow fallback; RCR pick {rname}",
        )
    if est_n is None or est_r is None:
        return _plan(
            "native", req, nat, (), est_n, None, "no cost model for both plans"
        )
    est_p = est_r + tp_us
    best = ("prepass", rreq, rcr, tuple(tps), est_p)
    if lay == "CRR" and R2_PLAN_PARTIAL:
        # TN: transpose one operand and run the other native layout (NN after
        # transposing A, TT after transposing B)
        for nm, lay2, keep in (("A", "RRR", "B"), ("B", "CCR", "A")):
            preq = replace(
                req, layout=lay2, trans_a=lay2[0] == "C", trans_b=lay2[1] == "C"
            )
            try:
                pres = dsp(preq)
            except ValueError:
                continue
            est_q = _g5n_est(preq, pres) if use_m else None
            if est_q is None:
                est_q = _g5_est(preq, pres)
            if est_q is None:
                continue
            tq = [t for t in tps if t.operand == nm]
            est_q += sum(
                r2_transpose_cost_us(t.rows, t.cols, fast=t.spec is not None)
                for t in tq
            )
            if est_q < best[4]:
                best = ("prepass_" + nm.lower(), preq, pres, tuple(tq), est_q)
    kind, preq, pres, ptps, est_p = best
    if est_p * margin < est_n:
        return _plan(
            kind, preq, pres, ptps, est_p, est_n,
            f"{kind} {est_p:.1f} us < native {est_n:.1f} us",
        )  # fmt: skip
    return _plan(
        "native", req, nat, (), est_n, est_p,
        f"native {est_n:.1f} us <= {kind} {est_p:.1f} us",
    )  # fmt: skip

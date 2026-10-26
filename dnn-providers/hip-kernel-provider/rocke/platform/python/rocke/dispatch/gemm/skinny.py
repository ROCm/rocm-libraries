# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Skinny / decode / long-K few-tile RCR GEMM candidate (``gemm_skinny`` body).

The register-direct split-K body (:mod:`rocke.instances.common.gemm_skinny`)
loads MFMA fragments straight from global memory and reduces its split-K
partials inside the same launch (ordered, deterministic, self-cleaning tile
counters). It serves fp16 and bf16 on gfx942 / gfx950.

Selection has two parts:

* :func:`skinny_admit_region` -- the shape region where this body is the
  dispatcher's pick (``SKINNY_REGION``);
* :func:`skinny_plan` -- the configuration (tile, wave split, K chunk,
  prefetch depth, split-K degree) for an admitted shape: the surrogate-derived
  table :data:`rocke.dispatch.gemm.skinny_table.SKINNY_TABLE` (bins of
  log2 small side x log2 big side x log2 K, per orientation) when it has an
  entry, else the closed-form fallback :func:`_fallback_plan`.

Both are knobs (``SKINNY_DISPATCH`` turns the candidate off).
"""

from __future__ import annotations

import struct
from dataclasses import replace
from typing import Any, Optional, Tuple

from ...instances.common.gemm_skinny import (
    SKINNY_ARCHES,
    SkinnyGemmSpec,
    build_gemm_skinny,
    is_valid_spec,
    skinny_counter_ints,
    skinny_grid,
    skinny_kps,
    skinny_launch_error,
    skinny_signature,
    skinny_workspace_floats,
)
from ..core import (
    Capability,
    KernelCandidate,
    OperatorRequest,
    ProblemBinding,
    selector_matches,
)
from ...instances.common.gemm_mn_operand import GEMM_LAYOUTS
from .common import GemmRequest, rcr_request_errors

# Knobs -----------------------------------------------------------------------
SKINNY_DISPATCH = True
# the same body registered a second time at the lowest priority with no
# region gate: every gfx942 / gfx950 RCR shape that no other candidate admits
# still gets a correct kernel (M, N, K >= 1, any alignment)
SKINNY_FALLBACK = True
SKINNY_FALLBACK_PRIORITY = 99
# priority among the RCR candidates (lower wins); admitted only inside
# SKINNY_REGION, so it does not shadow the other bodies elsewhere
SKINNY_PRIORITY = 3
# admission region: always when min(M, N) <= always_small; above that (up to
# max_small, the calibrated range of the table) when the surrogate table predicts
# the body within min_ratio (percent) of hipBLASLt's best
SKINNY_REGION = {
    "gfx950": {
        "always_small": 32,
        "min_ratio": 95,
        "max_small": 256,
        "defer_ratio": 100,
        "mid_max_small": 1023,
        "mid_min_ratio": 105,
        "mid_defer_ratio": 110,
    },
    "gfx942": {
        "always_small": 32,
        "min_ratio": 95,
        "max_small": 256,
        "defer_ratio": 100,
        "mid_max_small": 1023,
        "mid_min_ratio": 105,
        "mid_defer_ratio": 110,
    },
}
# (mid band max_small < min(M, N) <= mid_max_small: sparser training data, so
# only confident predicted wins are admitted, and a ratio below
# mid_defer_ratio defers to the small DTL ring when it admits the shape)
# (always_small region: a predicted ratio below defer_ratio defers to the small
# DTL ring when that candidate admits the shape: on held-out gfx950 shapes
# where the ring is the faster body, the table ratio is low)
# NN / TN / TT (layouts RRR / CRR / CCR) run the body natively (MN-contiguous
# operands staged through LDS, transpose reads), with the RCR configuration of
# the same shape; False = RCR only
SKINNY_NATIVE = True
# LDS row pad (bytes) of the native form's staged MN operand (bank spread of
# the transpose reads; gemm_skinny.SkinnyGemmSpec.mn_pad)
SKINNY_MN_PAD = 16
# native requests on the arches of the native table: configuration and
# admission from the native table (skinny_native_table: the surrogate fitted
# on the native-layout searches); False = the RCR table's configuration and
# region for every layout
SKINNY_NATIVE_TABLE_RULE = True
SKINNY_NATIVE_TABLE_ARCHES = ("gfx950", "gfx942")
# admit a native request when the table's predicted gemm_skinny ratio is at
# least this percentage of the predicted ratio of the dispatcher's
# non-gemm_skinny pick (100: predicted at least as fast)
SKINNY_NATIVE_MARGIN = 100
# native admission rule per arch: "tiles" (on gfx950 the native
# gemm_skinny beats the dispatcher's native DTL bodies when an MN-contiguous
# leading dimension is not a multiple of 8 (no DTL ring reads it in place),
# and on few output tiles: below SKINNY_NATIVE_A8_MAX_TILES 128x128 tiles when
# M or N is not a multiple of SKINNY_NATIVE_ALIGN, below
# SKINNY_NATIVE_A64_MAX_TILES when both are and min(M, N) <
# SKINNY_NATIVE_A64_MAX_SMALL); "table": the native table's predictions
# (r >= rx x SKINNY_NATIVE_MARGIN)
SKINNY_NATIVE_RULE = {"gfx950": "tiles", "gfx942": "table"}
SKINNY_NATIVE_ALIGN = 64
# thresholds fitted against the native-form selector's non-skinny picks (paired
# gemm_skinny vs SKINNY_NATIVE-off timings on gfx950 non-NT shapes, same
# process, cross-validated): admit on an odd leading dimension or fewer than 16
# 128x128 tiles
SKINNY_NATIVE_A8_MAX_TILES = 16
SKINNY_NATIVE_A64_MAX_TILES = 16
SKINNY_NATIVE_A64_MAX_SMALL = 1 << 30
# fallback configuration constants
SKINNY_WAVES_TARGET = 2048
SKINNY_MIN_STEPS_PER_WAVE = 8

_FAMILY_ALGO = "gemm_skinny"


def _small_tile(s: int) -> Tuple[int, int]:
    """(atoms, warps) along the small side."""
    if s <= 16:
        return 1, 1
    if s <= 32:
        return 2, 1
    if s <= 64:
        return 4, 1
    if s <= 128:
        return 4, 2
    return 4, 4


def _kalign(K: int) -> int:
    return 8 if K % 8 == 0 else (2 if K % 2 == 0 else 1)


def skinny_split(
    spec: SkinnyGemmSpec, M: int, N: int, K: int, waves_target: int
) -> int:
    """Split-K degree that brings the launch to about ``waves_target`` waves
    without slicing K below one step per K-wave; 1 when no target."""
    if waves_target <= 0:
        return 1
    gx, gy, _ = skinny_grid(spec, M, N, 1)
    waves = gx * gy * spec.waves
    steps = -(-int(K) // spec.step_k)
    s = round(waves_target / max(1, waves))
    return int(max(1, min(256, s, max(1, steps // spec.warp_k))))


def _fallback_plan(M: int, N: int, K: int, dtype: str) -> SkinnyGemmSpec:
    s, g = min(M, N), max(M, N)
    if s > 128:
        # outside the skinny regime (the fallback registration): 128 x 128
        # workgroup tiles of 2 x 2 waves, split-K only for few tiles
        spec = SkinnyGemmSpec(
            dtype=dtype,
            wave_m_atoms=4,
            wave_n_atoms=4,
            warp_m=2,
            warp_n=2,
            kc=16,
            prefetch=2,
            red_sync="sc1",
            kalign=_kalign(K),
        )
        tiles = -(-M // 128) * -(-N // 128)
        split = 1
        if tiles < 256:
            steps = -(-K // spec.step_k)
            split = max(
                1, min(64, 256 // tiles, steps // SKINNY_MIN_STEPS_PER_WAVE or 1)
            )
        return replace(spec, split=split, red_unroll=8 if split >= 16 else 4)
    sa, sw = _small_tile(s)
    kc = 32 if K <= 384 else 16
    steps = -(-K // (4 * kc))
    pf = 4 if steps <= 16 else 2
    # register budget of is_valid_spec: acc + (pf + 1) x fragments <= 384
    while pf > 1 and sa * 4 * 4 + (pf + 1) * (sa + 4) * kc // 2 > 384:
        pf -= 1
    stiles = -(-s // (16 * sa * sw))
    ab = 1
    for c in (2, 4):
        if stiles * -(-g // (16 * c)) * sw >= 512:
            ab = c
    tiles = stiles * -(-g // (16 * ab))
    wk = 1
    if tiles * sw < 256 and steps >= 16:
        wk = 2 if tiles * sw >= 64 else 4
    ma, wm, na, wn = (sa, sw, ab, 1) if M <= N else (ab, 1, sa, sw)
    spec = SkinnyGemmSpec(
        dtype=dtype,
        wave_m_atoms=ma,
        wave_n_atoms=na,
        warp_m=wm,
        warp_n=wn,
        warp_k=wk,
        kc=kc,
        prefetch=pf,
        red_sync="sc1",
        kalign=_kalign(K),
    )
    # at least SKINNY_MIN_STEPS_PER_WAVE steps per K-wave when K is long
    target = SKINNY_WAVES_TARGET if tiles * sw * wk < SKINNY_WAVES_TARGET // 2 else 0
    split = skinny_split(spec, M, N, K, target)
    split = max(1, min(split, steps // max(1, wk * SKINNY_MIN_STEPS_PER_WAVE) or 1))
    return replace(spec, split=split, red_unroll=8 if split >= 16 else 4)


def skinny_native(spec: SkinnyGemmSpec, req: GemmRequest) -> SkinnyGemmSpec:
    """``spec`` (an RCR configuration) in ``req``'s layout: the MN-contiguous
    operands' alignment class, the LDS pad and the transpose-read mode of the
    target (``tr`` on gfx950, 16-bit reads elsewhere). Identity for RCR."""
    lay = str(getattr(req, "layout", "RCR")).upper()
    if lay == "RCR":
        return spec
    a_mn, b_mn = lay[0] == "C", lay[1] == "R"
    lds = [int(x) for x, m in ((req.M, a_mn), (req.N, b_mn)) if m]
    mna = (
        8
        if all(x % 8 == 0 for x in lds)
        else (2 if all(x % 2 == 0 for x in lds) else 1)
    )
    return replace(
        spec,
        layout=lay,
        mnalign=mna,
        mn_pad=SKINNY_MN_PAD,
        mn_lds="tr" if req.arch == "gfx950" else "u16",
        kalign=8 if (a_mn and b_mn) else spec.kalign,
        kmap="mfma",
        b_load="buffer" if b_mn else spec.b_load,
    )


def _ld_odd(req: GemmRequest) -> bool:
    """An MN-contiguous operand's leading dimension (M of A, N of B) is not a
    multiple of 8 (the native DTL bodies cannot read it in place)."""
    lay = str(getattr(req, "layout", "RCR")).upper()
    a_mn, b_mn = lay[0] == "C", lay[1] == "R"
    return any(int(x) % 8 for x, m in ((req.M, a_mn), (req.N, b_mn)) if m)


def _native_table_on(req: GemmRequest) -> bool:
    return (
        SKINNY_NATIVE_TABLE_RULE
        and str(getattr(req, "layout", "RCR")).upper() != "RCR"
        and req.arch in SKINNY_NATIVE_TABLE_ARCHES
    )


def skinny_native_table_plan(req: GemmRequest) -> Optional[SkinnyGemmSpec]:
    """The native table's configuration for a non-RCR ``req`` (``None`` when
    the table has no entry or the entry does not plan a valid launch)."""
    from .skinny_native_table import skinny_native_entry

    M, N, K = int(req.M), int(req.N), int(req.K)
    e = skinny_native_entry(req.arch, M, N, K, req.layout, _ld_odd(req))
    if e is None:
        return None
    ma, na, wm, wn, wk, kc, pf, tw, _r, _rx = e
    spec = SkinnyGemmSpec(
        dtype="fp16" if req.dtype in ("fp16", "f16") else "bf16",
        wave_m_atoms=ma,
        wave_n_atoms=na,
        warp_m=wm,
        warp_n=wn,
        warp_k=wk,
        kc=kc,
        prefetch=pf,
        red_sync="sc1",
        kalign=_kalign(K),
    )
    split = 1
    if tw:
        tiles = -(-M // spec.tile_m) * -(-N // spec.tile_n)
        steps = -(-K // spec.step_k)
        split = max(
            1, min(256, round(tw / max(1, tiles * spec.waves)), max(1, steps // wk))
        )
    spec = replace(spec, split=split, red_unroll=8 if split >= 16 else 4)
    spec = _native_fit(skinny_native(spec, req), req.arch, M, N, K)
    return spec if _plan_ok(spec, req.arch, M, N, K) else None


def skinny_plan(req: GemmRequest) -> SkinnyGemmSpec:
    """The gemm_skinny launch configuration for ``req`` (table, else fallback)."""
    from .skinny_table import skinny_table_plan

    M, N, K = int(req.M), int(req.N), int(req.K)
    dtype = "fp16" if req.dtype in ("fp16", "f16") else "bf16"
    spec = skinny_native_table_plan(req) if _native_table_on(req) else None
    if spec is None:
        spec = skinny_table_plan(req.arch, M, N, K, dtype)
    if spec is not None:
        spec = _native_fit(skinny_native(spec, req), req.arch, M, N, K)
    if spec is None or not _plan_ok(spec, req.arch, M, N, K):
        spec = _native_fit(
            skinny_native(_fallback_plan(M, N, K, dtype), req), req.arch, M, N, K
        )
    for prefetch in (2, 1):
        if _plan_ok(spec, req.arch, M, N, K):
            break
        # last resort: one 16 x 16 wave tile, no K split inside the workgroup
        # (valid on every CDNA target whenever the launch limits hold)
        spec = replace(
            spec,
            wave_m_atoms=1,
            wave_n_atoms=1,
            warp_m=1,
            warp_n=1,
            warp_k=1,
            kc=16,
            prefetch=prefetch,
        )
    return spec


def _native_fit(
    spec: SkinnyGemmSpec, arch: str, M: int, N: int, K: int
) -> SkinnyGemmSpec:
    """A native (MN-contiguous) configuration that does not fit: drop the LDS
    pad, then halve the K chunk (one 32-K piece per step) when the staging of
    the MN operands exceeds the LDS (e.g. 64 KiB on gfx942); raise the split
    when one K slice spans more MN-operand bytes than the i32 offsets hold."""
    if spec.layout.upper() == "RCR" or _plan_ok(spec, arch, M, N, K):
        return spec
    for kv in ({}, {"mn_pad": 0}, {"mn_pad": 0, "kc": 8}):
        alt = replace(spec, **kv) if kv else spec
        if not is_valid_spec(alt, arch)[0]:
            continue
        alt = _native_split_fit(alt, M, N, K)
        if _plan_ok(alt, arch, M, N, K):
            return alt
    return spec


def _native_split_fit(spec: SkinnyGemmSpec, M: int, N: int, K: int) -> SkinnyGemmSpec:
    """The smallest split >= spec.split whose K slice keeps every MN-contiguous
    operand's offsets in i32 (``skinny_launch_error``'s slice rule)."""
    a_mn, b_mn = spec.mn_operands
    ld = max([int(x) for x, m in ((M, a_mn), (N, b_mn)) if m] or [1])
    kps_max = (2**31 - 1) // (2 * spec.step_k * ld) - 2
    if kps_max < 1:
        return spec
    steps = -(-int(K) // spec.step_k)
    split = max(int(spec.split), -(-steps // kps_max))
    if split == spec.split:
        return spec
    return replace(spec, split=split, red_unroll=8 if split >= 16 else spec.red_unroll)


def _plan_ok(spec: SkinnyGemmSpec, arch: str, M: int, N: int, K: int) -> bool:
    return (
        is_valid_spec(spec, arch)[0]
        and skinny_launch_error(spec, M, N, K, spec.split) is None
    )


def skinny_admit_region(req: GemmRequest) -> Tuple[bool, str]:
    from .skinny_table import skinny_table_entry

    reg = SKINNY_REGION.get(req.arch)
    if reg is None:
        return False, f"no gemm_skinny region for {req.arch}"
    s = min(int(req.M), int(req.N))
    if _native_table_on(req):
        return _native_admit(req, reg, s)
    if skinny_takes_over_vgpr(req):
        return True, "ok (ahead of the g5 register-staged body: SKINNY_OVER_VGPR)"
    if s <= reg["always_small"]:
        e = skinny_table_entry(req.M, req.N, req.K, req.dtype)
        if e is not None and e[-1] < reg["defer_ratio"] and _ring_small_takes(req):
            return False, (
                f"min(M, N)={s}: predicted gemm_skinny ratio {e[-1] / 100:.2f} < "
                f"{reg['defer_ratio'] / 100:.2f} and the small DTL ring admits it"
            )
        return True, "ok"
    mid = s > reg["max_small"]
    if s > reg["mid_max_small"]:
        return False, (
            f"min(M, N)={s} > {reg['mid_max_small']} (the measured range of the table)"
        )
    e = skinny_table_entry(req.M, req.N, req.K, req.dtype)
    if e is None:
        return False, f"min(M, N)={s} is outside the gemm_skinny table"
    bound = reg["mid_min_ratio"] if mid else reg["min_ratio"]
    if e[-1] < bound:
        return False, (
            f"min(M, N)={s}: predicted gemm_skinny ratio {e[-1] / 100:.2f} of "
            f"hipBLASLt is below the admission bound {bound / 100:.2f}"
        )
    if mid and e[-1] < reg["mid_defer_ratio"] and _ring_small_takes(req):
        return False, (
            f"min(M, N)={s}: predicted ratio {e[-1] / 100:.2f} < "
            f"{reg['mid_defer_ratio'] / 100:.2f} and the small DTL ring admits it"
        )
    return True, "ok"


# gfx942 NN / TN / TT: the native gemm_skinny ahead of the g5 register-staged
# native forms (common.R2_VP_MN, priority 2) below max_tiles 128x128 output
# tiles, chosen from paired gfx942 non-NT timings (gemm_skinny vs the dispatcher
# pick, same process, cross-validated). Per arch; empty = g5 first.
SKINNY_OVER_VGPR_NATIVE = {"gfx942": {"max_tiles": 32}}


def skinny_native_over_vgpr(req: GemmRequest) -> bool:
    """Whether a non-RCR ``req`` goes to the native gemm_skinny ahead of the g5
    native forms (:data:`SKINNY_OVER_VGPR_NATIVE`; only where the plan launches)."""
    rule = SKINNY_OVER_VGPR_NATIVE.get(getattr(req, "arch", ""))
    if not rule or not SKINNY_DISPATCH or not SKINNY_NATIVE:
        return False
    if str(getattr(req, "layout", "RCR")).upper() == "RCR":
        return False
    M, N, K = int(req.M), int(req.N), int(req.K)
    if -(-M // 128) * -(-N // 128) >= int(rule["max_tiles"]):
        return False
    spec = skinny_plan(req)
    return _plan_ok(spec, req.arch, M, N, K)


def _native_admit(req: GemmRequest, reg: dict, s: int) -> Tuple[bool, str]:
    """Native-layout region: always for the decode-like small side;
    else :data:`SKINNY_NATIVE_RULE` of the arch -- the fitted tile-count
    rule, or where the native table predicts gemm_skinny at least as fast as
    the dispatcher's non-gemm_skinny pick (both relative to hipBLASLt)."""
    from .skinny_native_table import skinny_native_entry

    if s <= reg["always_small"]:
        return True, "ok"
    if skinny_native_over_vgpr(req):
        return True, "ok (few tiles: ahead of the g5 native forms, SKINNY_OVER_VGPR_NATIVE)"
    if SKINNY_NATIVE_RULE.get(req.arch, "table") == "tiles":
        return _native_tiles_admit(req, s)
    e = skinny_native_entry(req.arch, req.M, req.N, req.K, req.layout, _ld_odd(req))
    if e is None:
        return False, f"min(M, N)={s} is outside the native gemm_skinny table"
    r, rx = e[-2], e[-1]
    if r * 100 < rx * SKINNY_NATIVE_MARGIN:
        return False, (
            f"layout {req.layout}: predicted gemm_skinny ratio {r / 100:.2f} < "
            f"{SKINNY_NATIVE_MARGIN / 100:.2f} x the non-gemm_skinny pick's {rx / 100:.2f}"
        )
    return True, "ok"


def _native_tiles_admit(req: GemmRequest, s: int) -> Tuple[bool, str]:
    """The measured native rule (``SKINNY_NATIVE_RULE`` "tiles")."""
    if _ld_odd(req):
        return True, "ok (MN-contiguous leading dimension not a multiple of 8)"
    M, N = int(req.M), int(req.N)
    tiles = -(-M // 128) * -(-N // 128)
    al = int(SKINNY_NATIVE_ALIGN)
    if M % al or N % al:
        if tiles < SKINNY_NATIVE_A8_MAX_TILES:
            return True, "ok"
        return False, (
            f"{tiles} 128x128 tiles >= {SKINNY_NATIVE_A8_MAX_TILES}: the native DTL "
            "bodies serve it"
        )
    if tiles < SKINNY_NATIVE_A64_MAX_TILES and s < SKINNY_NATIVE_A64_MAX_SMALL:
        return True, "ok"
    return False, (
        f"M, N multiples of {al} with {tiles} 128x128 tiles and min(M, N)={s}: the "
        "native DTL bodies serve it"
    )


# gfx942 NT shapes with 16 <= min(M, N) < 256: gemm_skinny is preferred over
# the g5 register-staged body (priority 2) for min(M, N) < max_small with
# max(M, N) < max_big; the g5 body keeps the rest. Per arch; empty = the
# default order (g5 first).
SKINNY_OVER_VGPR = {"gfx942": {"max_small": 64, "max_big": 16384}}


def skinny_takes_over_vgpr(req: GemmRequest) -> bool:
    """Whether the g5 register-staged candidate defers ``req`` to gemm_skinny
    (:data:`SKINNY_OVER_VGPR`; only where gemm_skinny admits the request)."""
    rule = SKINNY_OVER_VGPR.get(getattr(req, "arch", ""))
    if not rule or not SKINNY_DISPATCH:
        return False
    if str(getattr(req, "layout", "RCR")).upper() != "RCR":
        return False
    M, N = int(req.M), int(req.N)
    if min(M, N) >= rule["max_small"] or max(M, N) >= rule["max_big"]:
        return False
    spec = skinny_plan(req)
    return _plan_ok(spec, req.arch, M, N, int(req.K))


def _ring_small_takes(req: GemmRequest) -> bool:
    """Whether the registered small DTL ring candidate admits ``req`` (the same
    checks as its registry entry: geometry, support predicates, admission)."""
    from ...core.arch import ArchTarget
    from ...instances.common.gemm_universal import DataSpec, UniversalGemmSpec
    from .common import (
        DTL_RING_SMALL_ARCHES,
        dtl_ring_small_admit,
        dtl_ring_small_tile_trait,
    )
    from .support import request_shape_supported, universal_spec_supported

    if req.arch not in DTL_RING_SMALL_ARCHES:
        return False
    dt = "fp16" if req.dtype in ("fp16", "f16") else "bf16"
    try:
        tile, trait = dtl_ring_small_tile_trait(req.arch, req.K, req.M, req.N, dtype=dt)
        spec = UniversalGemmSpec(
            name="ring_probe",
            tile=tile,
            trait=trait,
            data=DataSpec(dtype_a=dt, dtype_b=dt, dtype_c=dt, layout="RCR"),
            wave_size=ArchTarget.from_gfx(req.arch).wave_size,
        )
        return (
            universal_spec_supported(spec, arch=req.arch)[0]
            and request_shape_supported(req, spec)[0]
            and dtl_ring_small_admit(req, spec)[0]
        )
    except Exception:  # noqa: BLE001  (a probe: no ring, no deferral)
        return False


def skinny_binding(result: Any, verify: bool, *, dtype: str) -> ProblemBinding:
    """Bind a gemm_skinny selection to a runnable problem.

    Same corpus, reference and tolerance as the UniversalGemm RCR binding
    (:func:`rocke.dispatch.gemm.binding.gemm_rcr_binding`); the argument
    buffer adds the split-K workspace ``W``, the zeroed tile counters ``Cnt``
    and the ``SPLIT`` / ``KPS`` scalars. ``device_ptrs`` = (A, B, C, W, Cnt).
    """
    from .binding import gemm_rcr_binding

    base = gemm_rcr_binding(result, verify, dtype=dtype)
    req = result.request
    spec = result.spec
    M, N, K = int(req.M), int(req.N), int(req.K)
    wsz = max(16, 4 * skinny_workspace_floats(spec, M, N, spec.split))
    csz = max(16, 4 * skinny_counter_ints(spec, M, N))

    def make_args(rt: Any):
        _packed, (a_dev, b_dev, c_dev) = base.make_args(rt)
        w_dev = rt.alloc(wsz)
        n_dev = rt.alloc(csz)
        rt.memset(n_dev, 0, csz)
        packed = struct.pack(
            "<QQQQQiiiii",
            a_dev,
            b_dev,
            c_dev,
            w_dev,
            n_dev,
            M,
            N,
            K,
            int(spec.split),
            skinny_kps(spec, K, spec.split),
        )
        return packed, (a_dev, b_dev, c_dev, w_dev, n_dev)

    def check(rt: Any, ptrs):
        return base.check(rt, tuple(ptrs[:3]))

    return ProblemBinding(
        grid=base.grid,
        block=base.block,
        make_args=make_args,
        check=check,
        flop=base.flop,
        bytes_moved=base.bytes_moved,
    )


def skinny_grid_for(spec: SkinnyGemmSpec, req: OperatorRequest) -> Tuple[int, int, int]:
    assert isinstance(req, GemmRequest)
    err = skinny_launch_error(spec, req.M, req.N, req.K, spec.split)
    if err is not None:
        raise ValueError(f"{spec.kernel_name()}: {err}")
    return skinny_grid(spec, req.M, req.N, spec.split)


def make_skinny_candidate(
    *, name: str, family: str, abi_version: str, dtype: str, fallback: bool = False
) -> KernelCandidate:
    """The gemm_skinny candidate of one RCR registry (``dtype`` fp16 / bf16).

    ``fallback``: the lowest-priority registration without the region gate
    (``SKINNY_FALLBACK``)."""

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        if not (SKINNY_FALLBACK if fallback else SKINNY_DISPATCH):
            return (
                False,
                f"{'SKINNY_FALLBACK' if fallback else 'SKINNY_DISPATCH'} is off",
            )
        errors = rcr_request_errors(req, dtype=dtype)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, GemmRequest)
        if str(req.layout).upper() != "RCR" and not SKINNY_NATIVE:
            return False, "SKINNY_NATIVE is off (RCR only)"
        ok, why = selector_matches(req, candidate)
        if not ok:
            return False, why
        if not fallback:
            ok, why = skinny_admit_region(req)
            if not ok:
                return False, why
        spec = skinny_plan(req)
        ok, why = is_valid_spec(spec, req.arch)
        if not ok:
            return False, why
        err = skinny_launch_error(spec, req.M, req.N, req.K, spec.split)
        if err is not None:
            return False, err
        return True, "ok"

    def select(req: OperatorRequest) -> SkinnyGemmSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, GemmRequest)
        return skinny_plan(req)

    def sweep(req: OperatorRequest):
        if not candidate.admits(req)[0]:
            return ()
        base = select(req)
        out = [base]
        for s in (max(1, base.split // 2), base.split * 2):
            if s != base.split and is_valid_spec(replace(base, split=s), req.arch)[0]:
                out.append(replace(base, split=s, red_unroll=8 if s >= 16 else 4))
        return tuple(out)

    candidate = KernelCandidate(
        name=name,
        family=family,
        algorithm=_FAMILY_ALGO,
        spec_id="cdna_skinny_fallback" if fallback else "cdna_skinny_splitk",
        abi_version=abi_version + "+skinny/v1",
        priority=SKINNY_FALLBACK_PRIORITY if fallback else SKINNY_PRIORITY,
        capability=Capability(
            arches=SKINNY_ARCHES, dtypes=(dtype,), layouts=GEMM_LAYOUTS
        ),
        _supports=support,
        select_spec=select,
        # the ABI depends on the dtype only (not on the configuration)
        signature=lambda _spec: skinny_signature(SkinnyGemmSpec(dtype=dtype)),
        grid=skinny_grid_for,
        block=lambda spec: (int(spec.block_size), 1, 1),
        sweep_space=sweep,
        build=build_gemm_skinny,
        bind=lambda result, verify: skinny_binding(result, verify, dtype=dtype),
    )
    return candidate


# arches whose shipped selector model ranks gemm_skinny configurations (feature
# version r2c-v2 with the family in its training data); the opt-in selected
# skinny candidate declares only these (a v1 model never picks the family)
SELECTED_SKINNY_ARCHES: Tuple[str, ...] = ("gfx950",)


def make_selected_skinny_candidate(
    *, name: str, family: str, abi_version: str, dtype: str, registry: Any
) -> KernelCandidate:
    """The learned selector's gemm_skinny pick when it is not a registered
    skinny candidate's own configuration (another split-K degree or a one-step
    lever mutation; ``selector.SELECTOR_SKINNY``). Opt-in: priority selection
    never sees it; ``dispatch_gemm_<dtype>`` reaches it through
    ``selector.dispatch_override``, and a request pinned to its ``spec_id``
    reselects the same configuration. ``registry``: a zero-argument callable
    returning the family's registry (the selector decides against it)."""
    from . import selector as _selector

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = rcr_request_errors(req, dtype=dtype)
        if errors:
            return False, "; ".join(errors)
        ok, why = selector_matches(req, candidate)
        if not ok:
            return False, why
        return _selector.selected_supports(req, registry(), family="skinny")

    def select(req: OperatorRequest) -> SkinnyGemmSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        return _selector.selected_spec(req, registry(), family="skinny")

    candidate = KernelCandidate(
        name=name,
        family=family,
        algorithm=_FAMILY_ALGO,
        spec_id="cdna_selected_skinny",
        abi_version=abi_version + "+skinny/v1",
        priority=1,
        capability=Capability(
            arches=SELECTED_SKINNY_ARCHES, dtypes=(dtype,), layouts=("RCR",)
        ),
        _supports=support,
        select_spec=select,
        signature=lambda _spec: skinny_signature(SkinnyGemmSpec(dtype=dtype)),
        grid=skinny_grid_for,
        block=lambda spec: (int(spec.block_size), 1, 1),
        sweep_space=lambda req: (
            _selector.selected_sweep(req, registry(), family="skinny")
            if candidate.admits(req)[0]
            else ()
        ),
        build=build_gemm_skinny,
        bind=lambda result, verify: skinny_binding(result, verify, dtype=dtype),
        opt_in=True,
    )
    return candidate


# ---------------------------------------------------------------------------
# small-tile DTL ring outside its tuned admission: for
# min(M, N) >= RING_ANY_MIN_SMALL the small ring with pads (+ exact DTL
# bounds) and a few-tile split-K is faster than the generic cshuffle / mem
# bodies the dispatcher otherwise falls to (on held-out gfx950 shapes, mostly
# on the aligned mid shapes) and than the register-direct body where the skinny
# table does not admit the shape. Its priority sits just above the cshuffle
# default, below every tuned DTL / beat candidate and the skinny region.
# ---------------------------------------------------------------------------
RING_ANY = True
RING_ANY_PRIORITY = 9
RING_ANY_MIN_SMALL = 48
# split-K of the ring (atomic f32 epilogue): only for long K (or very few
# tiles); the degree aims at about RING_ANY_SPLIT_CTAS workgroups, each slice
# at least RING_ANY_SPLIT_MIN_SLICE deep, a divisor of the K tiles
RING_ANY_SPLIT_MIN_K = 3072
RING_ANY_SPLIT_FEW_TILES = 8
RING_ANY_SPLIT_FEW_MIN_K = 1024
RING_ANY_SPLIT_CTAS = 160
RING_ANY_SPLIT_MIN_SLICE = 512
RING_ANY_SPLIT_MAX = 16


def ring_any_split(
    M: int, N: int, K: int, tile_m: int, tile_n: int, tile_k: int
) -> int:
    """Split-K degree of the ring_small_any spec (fitted on split-K sweeps:
    best degree vs tiles and K)."""
    tiles = -(-int(M) // tile_m) * -(-int(N) // tile_n)
    K = int(K)
    if K % tile_k:
        return 1
    if not (
        K >= RING_ANY_SPLIT_MIN_K
        or (tiles <= RING_ANY_SPLIT_FEW_TILES and K >= RING_ANY_SPLIT_FEW_MIN_K)
    ):
        return 1
    want = min(
        RING_ANY_SPLIT_MAX,
        int(round(RING_ANY_SPLIT_CTAS / max(1, tiles))),
        K // RING_ANY_SPLIT_MIN_SLICE,
    )
    kt = K // tile_k
    for d in range(max(1, want), 0, -1):
        if kt % d == 0:
            return d
    return 1


def ring_small_any_spec(req: GemmRequest, spec: Any) -> Any:
    """The small ring's spec for ``req`` with the pads it needs and the
    few-tile split-K degree (``pad_fast_tile`` only without split-K)."""
    t = spec.tile
    M, N, K = int(req.M), int(req.N), int(req.K)
    tr = replace(
        spec.trait,
        pad_m=spec.trait.pad_m or M % t.tile_m != 0,
        pad_n=spec.trait.pad_n or N % t.tile_n != 0,
    )
    split = ring_any_split(M, N, K, t.tile_m, t.tile_n, t.tile_k)
    if split > 1:
        tr = replace(tr, split_k=split, pad_fast_tile=False)
    elif tr.pad_fast_tile and not (tr.pad_m or tr.pad_n):
        tr = replace(tr, pad_fast_tile=False)
    return replace(spec, trait=tr)


def _ring_k_floor(req: GemmRequest, spec: Any) -> Optional[str]:
    """The small ring's K floor where the default candidate takes an
    aligned shape: ring_small_any keeps out of it."""
    from .common import (
        DTL_RING_ADMIT_RULES,
        DTL_RING_SMALL_K32_AGPR_MIN_K,
        DTL_RING_SMALL_K32_MIN_K,
        DTL_RING_SMALL_RAGGED_MAX_K,
        _aligned_to_default,
    )
    from .common import normalize_dtype

    if not _aligned_to_default(req):
        return None
    t = spec.tile
    if t.tile_k == 64:
        rule = DTL_RING_ADMIT_RULES.get((req.arch, normalize_dtype(req.dtype)))
        k_min = rule.k_lo if rule is not None else DTL_RING_SMALL_RAGGED_MAX_K
    elif spec.trait.acc_agpr:
        k_min = DTL_RING_SMALL_K32_AGPR_MIN_K
    else:
        k_min = DTL_RING_SMALL_K32_MIN_K
    if int(req.K) < k_min:
        return (
            f"K={req.K} < {k_min}: the measured small-ring K floor of an aligned "
            "shape (the default candidate serves it)"
        )
    return None


def ring_small_any_admit(req: GemmRequest, spec: Any) -> Tuple[bool, str]:
    from ...instances.common.gemm_universal import is_valid_spec as uv_valid
    from ...instances.common.gemm_universal import launch_shape_error

    if not RING_ANY:
        return False, "RING_ANY is off"
    s = min(int(req.M), int(req.N))
    if s < RING_ANY_MIN_SMALL:
        return False, f"min(M, N)={s} < {RING_ANY_MIN_SMALL}: the skinny body serves it"
    why = _ring_k_floor(req, spec)
    if why is not None:
        return False, why
    ok, why = uv_valid(spec, arch=req.arch)
    if not ok:
        return False, why
    err = launch_shape_error(spec, req.M, req.N, req.K)
    if err is not None:
        return False, err
    return True, "ok"


__all__ = [
    "SKINNY_DISPATCH",
    "SKINNY_FALLBACK",
    "SKINNY_PRIORITY",
    "SKINNY_REGION",
    "SKINNY_NATIVE",
    "SKINNY_NATIVE_TABLE_RULE",
    "SKINNY_NATIVE_MARGIN",
    "SKINNY_NATIVE_RULE",
    "SKINNY_OVER_VGPR",
    "skinny_takes_over_vgpr",
    "make_skinny_candidate",
    "skinny_native_table_plan",
    "skinny_native",
    "skinny_admit_region",
    "skinny_plan",
    "skinny_split",
    "RING_ANY",
    "ring_small_any_spec",
    "ring_small_any_admit",
    "ring_any_split",
]

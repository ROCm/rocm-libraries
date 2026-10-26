# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Common helpers for GEMM-family dispatcher cases."""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, replace
from typing import Optional

from ...core.arch import ArchTarget
from ...helpers.split_k import arch_num_cus, select_split_k
from ...instances.common.gemm_universal import (
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
)
from ...instances.common.gemm_mn_operand import GEMM_LAYOUTS
from ..core import OperatorRequest, normalize_selector

# native NN / TN / TT: a non-RCR request whose RCR body uses
# K-contiguous-only levers (the LDS row interleave family) runs on the B3
# persistent ring; False leaves such requests to the caller's transpose pass.
NATIVE_LAYOUT_B3_FALLBACK = True
# the B3 fallback's grid: persistent on every CU (the RCR B3 form) or one
# workgroup per output tile (0, the default: the persistent form is slower
# with an M-contiguous A)
NATIVE_LAYOUT_B3_PERSISTENT = False
# scheduler strategy of the native LDS-row-interleave (beat) forms ("": the
# backend default; see TraitSpec.sched_strategy)
NATIVE_LAYOUT_RILV_SCHED = "iterative-minreg"
# the persistent (non-interleave) ring with an N-contiguous B: keep the
# persistent grid (True) or prefer one workgroup per output tile (False)
NATIVE_LAYOUT_PERSISTENT_B_MN = False
# the non-persistent interleave (beat) body with an N-contiguous B: prefer the
# cshuffle epilogue form (no_rve_csh) over the default-epilogue form (no_rve)
NATIVE_LAYOUT_NN_CSHUFFLE = True
# leading-dimension granularity (elements) an MN-contiguous operand needs for
# the DTL ring: its 16-byte LDS-DMA rows must start 16-byte aligned (gfx950
# returned wrong elements for 2-byte-aligned rows, e.g. 3001x2003x1024 TN /
# TT); others take the VGPR-staged fallback body. Not a tuning knob.
NATIVE_LAYOUT_DTL_LD_ALIGN = 8
# offer the 16-byte swapped-accumulator epilogue form
# ("no_rve_sv16": mfma_swap_ab + TraitSpec.sab_vec16) of the interleave (beat)
# bodies with an N-contiguous B. It sits after the other forms in
# NATIVE_LAYOUT_ORDER, so the rule pick is unchanged; the native selector ranks
# it. False: not generated.
NATIVE_LAYOUT_SAB_VEC16 = True
# ... and the same 16-byte epilogue on every other DTL-ring form with an
# N-contiguous B ("<form>_sv16": the B3 / flipped / 128 / same rings, a cshuffle
# epilogue becoming the direct one). Off: slower than the base form on the B3
# ring and at parity on the same / flip rings -- the swapped MFMA operand
# order costs the main loop more than the wider stores save.
NATIVE_LAYOUT_SAB_VEC16_RINGS = False
# streaming (non-temporal) C stores on the sv16 forms (TraitSpec.sab_store_nt).
# Off: slower than plain stores on the beat and B3 forms.
NATIVE_LAYOUT_SV16_NT = False
# native_layout_spec returns the first form that launches for the request's
# shape (False: the first valid form)
NATIVE_LAYOUT_LAUNCHABLE_FIRST = True
# preference order of the native forms per layout (native_layout_alternatives):
# the RCR body itself, the beat body without its vector epilogue, the B3 ring
NATIVE_LAYOUT_ORDER = {
    "RRR": ("grid", "same", "no_rve", "no_rve_csh", "no_rve_sab", "flip", "b3", "r128", "no_rve_sv16"),
    "CRR": ("grid", "same", "no_rve", "no_rve_csh", "no_rve_sab", "flip", "b3", "r128", "no_rve_sv16"),
    "CCR": ("grid", "same", "no_rve", "no_rve_csh", "no_rve_sab", "flip", "b3", "r128", "no_rve_sv16"),
}


@dataclass(frozen=True)
class GemmRequest(OperatorRequest):
    """Normalized GEMM request shared by GEMM-family dispatchers.

    Phase 1 uses this for FP16 RCR UniversalGemm. Later GEMM-family cases can
    either extend this type or define more specialized request types beside
    their case module.
    """

    M: int
    N: int
    K: int
    arch: str
    op: str = "gemm"
    dtype: str = "fp16"
    layout: str = "RCR"
    trans_a: bool = False
    trans_b: bool = True
    algorithm: str = "auto"
    spec_id: str = "auto"

    def normalized(self) -> dict:
        d = asdict(self)
        d["dtype"] = normalize_dtype(self.dtype)
        d["layout"] = self.layout.upper()
        d["algorithm"] = normalize_selector(self.algorithm)
        d["spec_id"] = normalize_selector(self.spec_id)
        return d

    def dims(self) -> dict[str, int]:
        return {"M": int(self.M), "N": int(self.N), "K": int(self.K)}


GEMM_DIM_VOCABULARY = ("M", "N", "K")


def normalize_dtype(dtype: str) -> str:
    d = dtype.lower()
    if d in ("f16", "half"):
        return "fp16"
    return d


def basic_gemm_request_errors(req: OperatorRequest) -> list[str]:
    """Common shape/op checks shared by GEMM-family dispatchers."""
    if not isinstance(req, GemmRequest):
        return [f"expected GemmRequest, got {type(req).__name__}"]
    errors: list[str] = []
    if req.op != "gemm":
        errors.append(f"unsupported op {req.op!r}")
    for field in ("M", "N", "K"):
        if int(getattr(req, field)) <= 0:
            errors.append(f"{field} must be positive")
    try:
        ArchTarget.from_gfx(req.arch)
    except KeyError as e:
        errors.append(str(e))
    return errors


def rcr_request_errors(req: OperatorRequest, *, dtype: str) -> list[str]:
    """Shared RCR-layout request validation parametrized by element dtype.

    Used by every RCR UniversalGemm family module (fp16, bf16, ...) so the
    layout/transpose checks and the single-dtype gate live in one place. The
    family passes the canonical dtype it implements (e.g. ``"fp16"``, ``"bf16"``).
    """
    errors = basic_gemm_request_errors(req)
    if errors:
        return errors
    assert isinstance(req, GemmRequest)
    if normalize_dtype(req.dtype) != dtype:
        errors.append(
            f"unsupported dtype {req.dtype!r}; this family supports {dtype} only"
        )
    lay = req.layout.upper()
    if lay not in GEMM_LAYOUTS:
        errors.append(f"unsupported layout {req.layout!r}; one of {GEMM_LAYOUTS}")
    elif (bool(req.trans_a), bool(req.trans_b)) != (lay[0] == "C", lay[1] == "C"):
        # RCR: A row-major, B logically transposed; RRR (NN): B row-major
        # [K, N]; CRR (TN): A stored [K, M] too; CCR (TT): A [K, M], B [N, K]
        errors.append(
            f"layout {lay} expects trans_a={lay[0] == 'C'}, trans_b={lay[1] == 'C'}"
        )
    return errors


def native_layout_alternatives(
    req: "GemmRequest", spec: UniversalGemmSpec
) -> tuple[list[tuple[str, UniversalGemmSpec]], str]:
    """Every native form of a non-RCR ``req`` derived from the RCR candidate's
    ``spec`` (native NN / TN / TT: the MN-contiguous operands are read in place,
    no transpose pass; see ``instances.common.gemm_mn_operand``), in
    preference order, each validity-checked:

    * ``same``: the RCR body with the request's layout;
    * ``no_rve``: the LDS-row-interleave (beat) body without its 16-byte vector
      epilogue (that epilogue needs B's interleaved column order, which an
      N-contiguous B read by transpose loads cannot provide);
    * ``b3``: the B3 ring (256x256x32, 3-stage fragment-prefetch interleave),
      one workgroup per tile unless :data:`NATIVE_LAYOUT_B3_PERSISTENT`.

    Returns ``(alternatives, reason the list is empty)``."""
    from ...instances.common.gemm_mn_operand import mn_layout_letters
    from ...instances.common.gemm_universal import is_valid_spec

    from ...instances.common.gemm_mn_operand import MN_OPERAND_ARCHS, uses_mn_dtl

    lay = req.layout.upper()
    if lay == "RCR":
        return [("same", spec)], "ok"
    if (
        R2_VP_MN
        and int(getattr(spec.trait, "vgpr_prefetch", 0))
        and req.arch in R2_VP_MN_ARCHES
    ):
        # the register-staged (vgpr_prefetch) body reads the MN-contiguous
        # operand itself (micro-block register transpose, TraitSpec.vp_mn_kv)
        f = r2_vp_mn_form(req, spec)
        if f is None:
            return [], f"layout {lay}: no vp_mn micro-block fits the g5 body"
        return [("vp_mn", f)], "ok"
    a_mn, b_mn = mn_layout_letters(lay)
    data = replace(spec.data, layout=lay)
    tr = spec.trait
    _al = max(8, int(NATIVE_LAYOUT_DTL_LD_ALIGN))
    ld8 = not ((a_mn and int(req.M) % _al) or (b_mn and int(req.N) % _al))
    if not uses_mn_dtl(spec) or not ld8 or req.arch not in MN_OPERAND_ARCHS:
        # the VGPR-staged transposing loader: the body itself when it has no
        # DTL ring, else the generic cshuffle body (an MN leading dimension
        # that is not a multiple of 8, or a target without transpose reads)
        base = spec if not uses_mn_dtl(spec) else native_generic_spec(req, spec)
        if base is None:
            return [], f"layout {lay}: no VGPR-staged body for this request"
        v = native_mn_vec(req, base)
        if v is None:
            return (
                [],
                f"layout {lay}: no MN load width fits the tile / leading dimension",
            )
        g = replace(base, data=data, trait=replace(base.trait, mn_vec=v))
        ok, why = is_valid_spec(g, arch=req.arch)
        return ([("generic", g)], "ok") if ok else ([], f"generic: {why}")
    if tr.lds_row_interleave and NATIVE_LAYOUT_RILV_SCHED and not tr.sched_strategy:
        # the unrolled interleave main loop with transpose-read fragments
        # needs the register-minimising scheduler (TraitSpec.sched_strategy)
        tr = replace(tr, sched_strategy=NATIVE_LAYOUT_RILV_SCHED)
    cands = [("same", replace(spec, data=data, trait=tr))]
    if (
        tr.persistent_ctas
        and b_mn
        and not tr.lds_row_interleave
        and not NATIVE_LAYOUT_PERSISTENT_B_MN
    ):
        # an N-contiguous B on the persistent ring: one workgroup per tile
        # first (the persistent tile walk is slower for native layouts)
        cands.insert(
            0, ("grid", replace(spec, data=data, trait=replace(tr, persistent_ctas=0)))
        )
    if tr.lds_row_interleave and tr.rilv_vec_epilogue and b_mn:
        nr = replace(tr, rilv_vec_epilogue=False, rve_store_nt=False, rve_pad=0)
        cands.append(("no_rve", replace(spec, data=data, trait=nr)))
        # the LDS-staged cshuffle epilogue (wide row stores; one workgroup per
        # tile: the persistent loop requires the default epilogue)
        cands.append(
            (
                "no_rve_csh",
                replace(
                    spec,
                    data=data,
                    trait=replace(
                        nr, epilogue="cshuffle", persistent_ctas=0, pad_fast_tile=False
                    ),
                ),
            )
        )
        if not nr.pad_n and not nr.sia_sched:
            # the accumulator as C^T (mfma_swap_ab): 4 consecutive N per lane
            # -> 8-byte row stores from the natural N-contiguous B fragments
            cands.append(
                (
                    "no_rve_sab",
                    replace(spec, data=data, trait=replace(nr, mfma_swap_ab=True)),
                )
            )
            if NATIVE_LAYOUT_SAB_VEC16:
                # the same with 16-byte stores (permuted B quads +
                # v_permlane32_swap; TraitSpec.sab_vec16)
                cands.append(
                    (
                        "no_rve_sv16",
                        replace(
                            spec,
                            data=data,
                            trait=replace(nr, mfma_swap_ab=True, sab_vec16=True),
                        ),
                    )
                )
    from ...instances.common.gemm_mn_operand import spec_mn_layouts

    try:
        _cf = all(
            lay_.conflict_free
            for lay_ in spec_mn_layouts(cands[0][1])
            if lay_ is not None
        )
    except ValueError:
        _cf = False
    if (tr.lds_row_interleave or not _cf) and NATIVE_LAYOUT_B3_FALLBACK:
        tile, trait = agpr_ilv_big_grid_tile_trait(req.arch)
        if not NATIVE_LAYOUT_B3_PERSISTENT:
            trait = replace(trait, persistent_ctas=0)
        M, N = int(req.M), int(req.N)
        if M % tile.tile_m or N % tile.tile_n:
            trait = replace(
                trait,
                pad_m=bool(M % tile.tile_m),
                pad_n=bool(N % tile.tile_n),
                dtl_tight_rsrc=True,
            )
        cands.append(
            (
                "b3",
                UniversalGemmSpec(
                    name=spec.name,
                    tile=tile,
                    trait=trait,
                    data=data,
                    wave_size=spec.wave_size,
                ),
            )
        )
    if not _cf and not tr.lds_row_interleave:
        # the body's MN tile has no conflict-free transpose layout (e.g. a
        # 192-wide side): the same body with its tile flipped (the 128 side
        # on the MN operand), the B3 ring and the 128x128x64 ring
        t0 = spec.tile
        ft = replace(
            t0,
            tile_m=t0.tile_n,
            tile_n=t0.tile_m,
            warp_m=t0.warp_n,
            warp_n=t0.warp_m,
        )
        M_, N_ = int(req.M), int(req.N)
        pm, pn = bool(M_ % ft.tile_m), bool(N_ % ft.tile_n)
        ftr = replace(tr, pad_m=pm, pad_n=pn)
        if pm or pn:
            ftr = replace(
                ftr,
                dtl_tight_rsrc=True,
                pad_fast_tile=True,
                dtl_soff0=False,
                epilogue="default",
            )
        fspec = replace(spec, tile=ft, trait=ftr, data=data)
        try:
            if all(
                lay_.conflict_free
                for lay_ in spec_mn_layouts(fspec)
                if lay_ is not None
            ):
                cands.append(("flip", fspec))
        except ValueError:
            pass
        rt, rtr = (
            TileSpec(
                tile_m=128,
                tile_n=128,
                tile_k=64,
                warp_m=2,
                warp_n=4,
                warp_k=1,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=32,
            ),
            _ring_pad_trait(),
        )
        cands.append(
            (
                "r128",
                UniversalGemmSpec(
                    name=spec.name,
                    tile=rt,
                    trait=rtr,
                    data=data,
                    wave_size=spec.wave_size,
                ),
            )
        )
    if NATIVE_LAYOUT_SAB_VEC16 and NATIVE_LAYOUT_SAB_VEC16_RINGS and b_mn:
        for nm, c in list(cands):
            ctr = c.trait
            if (
                ctr.lds_row_interleave
                or ctr.mfma_swap_ab
                or ctr.pad_n
                or ctr.split_k > 1
                or not uses_mn_dtl(c)
            ):
                continue
            # a cshuffle (LDS-staged) epilogue becomes the direct 16-byte one
            cands.append(
                (
                    nm + "_sv16",
                    replace(
                        c,
                        trait=replace(
                            ctr,
                            epilogue="default",
                            mfma_swap_ab=True,
                            sab_vec16=True,
                            pad_fast_tile=False,
                        ),
                    ),
                )
            )
    if NATIVE_LAYOUT_SV16_NT:
        cands = [
            (nm, replace(c, trait=replace(c.trait, sab_store_nt=True)))
            if c.trait.sab_vec16
            else (nm, c)
            for nm, c in cands
        ]
    order = NATIVE_LAYOUT_ORDER.get(lay, ())
    cands.sort(key=lambda c: order.index(c[0]) if c[0] in order else len(order))
    if (
        NATIVE_LAYOUT_NN_CSHUFFLE
        and b_mn
        and tr.lds_row_interleave
        and not tr.persistent_ctas
        and not tr.pad_n
    ):
        # the one-workgroup-per-tile interleave body with an N-contiguous B:
        # the LDS-staged cshuffle epilogue ahead of the scattered default one
        cands.sort(key=lambda c: c[0] != "no_rve_csh")
    if not _cf:
        # conflict-free forms first: the flipped tile, then by grid fill the
        # B3 ring (at least one wave of its 256x256 tiles) or the 128 ring
        big = -(-int(req.M) // 256) * -(-int(req.N) // 256) >= arch_num_cus(req.arch)
        rank = {"flip": 0, "b3": 1 if big else 2, "r128": 2 if big else 1}
        cands.sort(key=lambda c: rank.get(c[0], 3))
    out, whys = [], []
    for nm, c in cands:
        # every form gets the per-request adaptation of g1_adapt_spec (the K tail
        # of the B3 / flipped / 128 rings built here; identity on exact tiles)
        c = g1_adapt_spec(req, c)
        ok, why = is_valid_spec(c, arch=req.arch)
        if ok:
            out.append((nm, c))
        else:
            whys.append(f"{nm}: {why}")
    return out, "; ".join(whys)


def native_mn_vec(req: "GemmRequest", spec: UniversalGemmSpec) -> Optional[int]:
    """Widest MN load (8 / 4 / 2 / 1 elements) of the VGPR-staged MN operand
    loader that divides each MN-contiguous operand's leading dimension and
    gives every thread whole loads of its tile."""
    from ...instances.common.gemm_mn_operand import mn_layout_letters

    a_mn, b_mn = mn_layout_letters(req.layout)
    t = spec.tile
    for v in (8, 4, 2, 1):
        ok = True
        for is_mn, ld, tile_mn in (
            (a_mn, int(req.M), t.tile_m),
            (b_mn, int(req.N), t.tile_n),
        ):
            if is_mn and (
                ld % v or tile_mn % v or (tile_mn * t.tile_k // v) % spec.block_size
            ):
                ok = False
        if ok:
            return v
    return None


# the universal padded fallback candidate (lowest priority,
# every layout): the generic cshuffle body (VGPR-staged loads) padded where the
# shape is ragged, with guarded K-contiguous rows (TraitSpec.kc_row_guard) and
# the transposing MN loader for NN / TN / TT. Serves every request no faster
# candidate admits. False: no fallback (those requests are refused).
UNIVERSAL_FALLBACK = True
UNIVERSAL_FALLBACK_ARCHES = ("gfx942", "gfx950")


def universal_fallback_tile_trait(
    arch: str, M: int, N: int, K: int
) -> Optional[tuple[TileSpec, TraitSpec]]:
    """Tile / trait of the universal padded fallback body on ``arch``."""
    if arch == "gfx942":
        tile = TileSpec(
            tile_m=128,
            tile_n=128,
            tile_k=16,
            warp_m=2,
            warp_n=2,
            warp_k=1,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=8,
        )
    elif arch == "gfx950":
        tile = TileSpec(
            tile_m=128,
            tile_n=128,
            tile_k=32,
            warp_m=2,
            warp_n=2,
            warp_k=1,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
        )
    else:
        return None
    M, N, K = int(M), int(N), int(K)
    trait = TraitSpec(
        pipeline="compv4",
        scheduler="intrawave",
        epilogue="cshuffle",
        pad_m=bool(M % tile.tile_m),
        pad_n=bool(N % tile.tile_n),
        pad_k=bool(K % tile.tile_k),
        kc_row_guard=True,
    )
    return tile, trait


def native_generic_spec(
    req: "GemmRequest", spec: UniversalGemmSpec
) -> Optional[UniversalGemmSpec]:
    """The universal padded fallback body (:func:`universal_fallback_tile_trait`)
    for ``req``: the native-layout fallback of a DTL candidate that cannot read
    an MN-contiguous operand through its DTL ring."""
    tt = universal_fallback_tile_trait(req.arch, req.M, req.N, req.K)
    if tt is None:
        return None
    return UniversalGemmSpec(
        name=spec.name,
        tile=tt[0],
        trait=tt[1],
        data=spec.data,
        wave_size=spec.wave_size,
    )


# ring_any and ring_small_any (the DTL rings past their own admission rules)
# also serve NN / TN / TT natively, as the registered ring candidates do.
# False: RCR only.
MERGE_NATIVE_RING_ANY = True


def native_any_admit(admit):
    """``admit`` for a candidate registered in every layout whose non-RCR
    service is the knob :data:`MERGE_NATIVE_RING_ANY`."""

    def _admit(req, spec):
        if str(getattr(req, "layout", "RCR")).upper() != "RCR" and not (
            MERGE_NATIVE_RING_ANY
        ):
            return False, "MERGE_NATIVE_RING_ANY is off (RCR only)"
        return admit(req, spec)

    return _admit


# ring_any ahead of ring_small_any where both admit (ring_any is the faster
# body there; see the registries). False: the registry priority order.
MERGE_RING_ANY_FIRST = True


def ring_any_first_admit(admit, small_any):
    """``admit`` (the ring_any rule) restricted to the requests the
    ring_small_any candidate (``small_any()``, looked up at call time) also
    admits, under :data:`MERGE_RING_ANY_FIRST`."""

    def _admit(req, spec):
        if not MERGE_RING_ANY_FIRST:
            return False, "MERGE_RING_ANY_FIRST is off"
        ok, why = admit(req, spec)
        if not ok:
            return ok, why
        c = small_any()
        if c is None:
            return False, "no ring_small_any candidate registered"
        # (unpinned: a probe pinned to this candidate's own spec_id must still
        # see whether ring_small_any would take the request in auto mode)
        probe = req
        if hasattr(req, "spec_id") and hasattr(req, "algorithm"):
            probe = replace(req, spec_id="auto", algorithm="auto")
        ok2, why2 = c.admits(probe)
        if not ok2:
            return False, f"ring_small_any does not admit it ({why2})"
        return True, "ok (ring_any ahead of ring_small_any)"

    return _admit


def native_layout_spec(
    req: "GemmRequest", spec: UniversalGemmSpec
) -> tuple[Optional[UniversalGemmSpec], str]:
    """The preferred native form of a non-RCR ``req`` (the first of
    :func:`native_layout_alternatives`), or ``(None, reason)`` when the layout
    cannot be served natively (the caller transposes)."""
    alts, why = native_layout_alternatives(req, spec)
    if not alts:
        return None, why
    if NATIVE_LAYOUT_LAUNCHABLE_FIRST and req.layout.upper() != "RCR":
        # the first form the launch rules accept for this shape
        # (a later form may launch where the preferred one cannot, e.g. a K tail)
        from ...instances.common.gemm_universal import launch_shape_error

        for _nm, s in alts:
            if launch_shape_error(s, int(req.M), int(req.N), int(req.K)) is None:
                return s, "ok"
    return alts[0][1], "ok"


def apply_split_k(req: GemmRequest, spec: UniversalGemmSpec) -> UniversalGemmSpec:
    """Return ``spec`` with a split-K degree chosen for ``req`` on its arch.

    Split-K is engaged only for skinny / tall-N decode shapes whose base grid
    leaves the CU-rich CDNA device idle, and only on the MFMA (CDNA) family the
    kernel's atomic-add epilogue supports. For any shape that already fills the
    device -- and on RDNA, where the split-K epilogue is not wired -- the chosen
    degree is ``1`` and this returns the spec **unchanged** (so the default /
    square-GEMM path stays byte-identical). The ``ROCKE_GEMM_SPLIT_K`` env flag
    overrides the heuristic (see :mod:`rocke.helpers.split_k`).
    """
    target = ArchTarget.from_gfx(req.arch)
    if target.family != "cdna":
        return spec
    t = spec.tile
    decision = select_split_k(
        M=req.M,
        N=req.N,
        K=req.K,
        tile_m=t.tile_m,
        tile_n=t.tile_n,
        tile_k=t.tile_k,
    )
    if decision.split_k <= 1:
        # a pad_k body splits any K with whole-tile slices
        return g1_aligned_split_k(req, spec)
    if G1_SPLIT_K_ALIGNED and spec.trait.pad_k and not spec.trait.direct_to_lds:
        # a far shallower exact factor than the slice-depth target (e.g. 2 of
        # 52 for K = 26816): the whole-tile slices of the target degree
        alt = g1_aligned_split_k(req, spec)
        if alt.trait.split_k >= 2 * decision.split_k:
            return alt
    return replace(spec, trait=replace(spec.trait, split_k=decision.split_k))


# ---------------------------------------------------------------------------
# DTL ping-pong candidate geometry, shared by the fp16 / bf16 RCR registries
# (see examples/gfx950/gemm_perf_square_warpspec): a 256x256 block of 16 waves
# (4x4, 64x64 per wave) whose A/B tiles
# are DMA'd straight to LDS (buffer_load ... lds), ping-ponged with a depth-1
# prefetch, read through the element-granular XOR swizzle, with A and B cached
# in L2 (CACHE_ALL) and the chiplet-aware grid swizzle.
#   * gfx950 (160 KiB LDS, 4-dword DTL): 256x256x64, 16x16x32 atoms -> 128 KiB.
#   * gfx942 (64 KiB LDS, 1-dword DTL):  256x256x32, 16x16x16 atoms ->  64 KiB.
# ---------------------------------------------------------------------------

# Arches the ping-pong candidate is dispatched on: those with a geometry
# (``dtl_pingpong_tile_trait``) AND a cost model calibrated on that hardware
# (``DTL_PINGPONG_COST_MODELS``), every calibration point checked against an
# f64 full-output reference before and after timing.
DTL_PINGPONG_ARCHES = ("gfx942", "gfx950")
# Arches with a ping-pong geometry (sweeps / explicit selection).
DTL_PINGPONG_GEOMETRY_ARCHES = ("gfx942", "gfx950")

# Ring body levers of the dispatched candidate, per arch (each checked on
# device against a full-output f64 reference): the alias-scoped two-stage DTL ring (``dtl_stages=2``: per-stage A/B LDS
# allocations + ``lds_alias_scopes``, which removes the compiler-inserted
# ``s_waitcnt vmcnt(0)`` in front of every K tile's first ds_read) with the
# MFMA-cluster priority bookends (``interwave_setprio``). An arch absent here
# keeps the ping-pong body.
DTL_PINGPONG_RING: dict = {
    "gfx950": {"dtl_stages": 2, "interwave_setprio": True},
}


def dtl_pingpong_tile_trait(
    arch: str, *, chiplet_chunk_size: int = 64, ring: Optional[bool] = None
) -> tuple[TileSpec, TraitSpec]:
    """The ping-pong tile and trait for ``arch`` (gfx942 / gfx950).

    ``ring``: ``None`` (default) applies the arch's :data:`DTL_PINGPONG_RING`
    levers, ``False`` gives the ping-pong body, ``True`` forces the gfx950 ring
    levers on any arch with a geometry."""
    if arch == "gfx950":
        tile_k, wtk = 64, 32
    elif arch == "gfx942":
        tile_k, wtk = 32, 16
    else:
        raise ValueError(f"no DTL ping-pong GEMM geometry for {arch!r}")
    tile = TileSpec(
        tile_m=256,
        tile_n=256,
        tile_k=tile_k,
        warp_m=4,
        warp_n=4,
        warp_k=1,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=wtk,
    )
    trait = TraitSpec(
        pipeline="compv4",
        scheduler="intrawave",
        epilogue="default",
        direct_to_lds=True,
        dtl_prefetch=True,
        lds_swizzle=True,
        dtl_cache_a=0,
        dtl_cache_b=0,
        chiplet_swizzle=True,
        chiplet_wgm=8,
        chiplet_chunk_size=chiplet_chunk_size,
    )
    if ring is None:
        levers = DTL_PINGPONG_RING.get(arch, {})
    else:
        levers = DTL_PINGPONG_RING["gfx950"] if ring else {}
    if levers:
        trait = replace(trait, **levers)
    return tile, trait


def dtl_pingpong_sweep_variants(
    spec: UniversalGemmSpec, arch: str
) -> tuple[UniversalGemmSpec, ...]:
    """The ping-pong candidate's sweep neighbourhood (selected spec first).

    Chiplet chunk 32 / off and the cshuffle epilogue (valid on gfx950, where the
    LDS gate counts the C tile aliased onto A/B; over the 64 KiB gfx942 budget,
    so dropped there), so a registry sweep sees the relevant variants instead of
    the single auto pick. Every variant is validity-checked.
    """
    from ...instances.common.gemm_universal import is_valid_spec

    tr = spec.trait
    out = [spec]
    variants = [
        replace(tr, chiplet_chunk_size=32),
        replace(tr, chiplet_swizzle=False),
        replace(tr, epilogue="cshuffle"),
    ]
    if tr.dtl_stages >= 2:
        # ring neighbourhood: the ping-pong body, the persistent overlapped
        # epilogue (one workgroup per CU), and the in-stream barrier
        variants += [
            replace(tr, dtl_stages=0, interwave_setprio=False),
            replace(tr, persistent_ctas=arch_num_cus(arch), chiplet_chunk_size=32),
            replace(tr, dtl_stream_barrier=True, interwave_setprio=False),
            # the persistent ring with the DTL resource hoist (short-K
            # shapes) and iglp_opt(1) (large-K shapes); see
            # examples/gfx950/gemm_perf_square_warpspec/DTL_RING_SHAPES_CASE_STUDY.md
            replace(
                tr,
                persistent_ctas=arch_num_cus(arch),
                chiplet_chunk_size=32,
                dtl_hoist=True,
            ),
            replace(tr, iglp_opt=1),
            # narrower chiplet super-tile groups (wide long-K grids; a sweep
            # variant here, the big-grid rule below selects it)
            replace(tr, iglp_opt=1, chiplet_wgm=4),
        ]
    for t2 in variants:
        if t2 == tr:
            continue
        cand = replace(spec, trait=t2)
        if is_valid_spec(cand, arch=arch)[0]:
            out.append(cand)
    if tr.dtl_stages >= 2 and arch == "gfx950":
        # the 4-wave AGPR interleave body (2x2 waves of 128x128, tile_k 32,
        # 3-stage ring) with chiplet wgm 8 (mid-size grids) and wgm 4 (very
        # large grids)
        ilv_tile = replace(spec.tile, tile_k=32, warp_m=2, warp_n=2)
        ilv = replace(
            tr,
            dtl_stages=3,
            dtl_frag_prefetch=True,
            acc_agpr=True,
            mfma_interleave=True,
            ilv_merge=True,
            ilv_barrier_after=2,
            lds_swizzle_r=1,
            dtl_hoist=True,
            chiplet_chunk_size=64,
        )
        for wgm in (8, 4):
            cand = replace(
                spec,
                tile=ilv_tile,
                trait=replace(ilv, chiplet_wgm=wgm),
                block_size=4 * spec.wave_size,
            )
            if is_valid_spec(cand, arch=arch)[0]:
                out.append(cand)
    if tr.dtl_stages >= 2 and arch in RILV_ARCHES:
        # row-interleave body: both K-tile schedules on the grid mappings the
        # dispatcher selects for some shape class
        grids = (
            dict(chiplet_wgm=16, chiplet_chunk_size=8, persistent_ctas=0),
            dict(
                chiplet_wgm=16, chiplet_chunk_size=8, persistent_ctas=arch_num_cus(arch)
            ),
            dict(
                chiplet_wgm=8, chiplet_chunk_size=16, persistent_ctas=arch_num_cus(arch)
            ),
            dict(
                chiplet_wgm=8, chiplet_chunk_size=32, persistent_ctas=arch_num_cus(arch)
            ),
            dict(chiplet_wgm=8, chiplet_chunk_size=64, persistent_ctas=0),
        )
        for hb, sia in ((False, False), (True, False), (True, True)):
            b_tile, b_trait = rilv_tile_trait(arch, hb, sia=sia)
            for g in grids:
                cand = replace(
                    spec,
                    tile=b_tile,
                    trait=replace(b_trait, **g),
                    block_size=4 * spec.wave_size,
                )
                if cand not in out and is_valid_spec(cand, arch=arch)[0]:
                    out.append(cand)
        # the hand-ordered (asm_loop) body on the same grids and on the
        # chunk-4 long-K grid
        a_tile, a_base = rilv_tile_trait(arch, False)
        a_trait = rilv_asm_loop_trait(a_base, False)
        for g in grids + (
            dict(chiplet_wgm=16, chiplet_chunk_size=4, persistent_ctas=0),
        ):
            cand = replace(
                spec,
                tile=a_tile,
                trait=replace(a_trait, **g),
                block_size=4 * spec.wave_size,
            )
            if cand not in out and is_valid_spec(cand, arch=arch)[0]:
                out.append(cand)
        # the dispatched tuned body (dtl_soff0 + dtl_addr_spread + hb_dtl_delay
        # + kloop_unroll on hb_sched) on the same grids, both unrolls
        for ku in (RILV_KLOOP_UNROLL_SHORT, RILV_KLOOP_UNROLL_LONG):
            x_trait = rilv_tuned_trait(a_base, kloop_unroll=ku)
            for g in grids:
                cand = replace(
                    spec,
                    tile=a_tile,
                    trait=replace(x_trait, **g),
                    block_size=4 * spec.wave_size,
                )
                if cand not in out and is_valid_spec(cand, arch=arch)[0]:
                    out.append(cand)
    return tuple(out)


# ---------------------------------------------------------------------------
# DTL-ring candidates for the shapes the 256x256 ping-pong / ring candidate
# serves badly or not at all (gfx950). The rules and thresholds below were
# calibrated on gfx950 timings (every kernel checked against a full-output f64
# reference) and then checked on held-out shapes; see
# examples/gfx950/gemm_perf_square_warpspec/DTL_RING_SHAPES_CASE_STUDY.md.
#   * a small-tile ring (128x128x64, 8 waves of 64x32, 2-stage ring + MFMA
#     priority bookends + DTL resource hoist) when the 256x256 tiles fill the
#     last wave poorly (exactly one wave of 256x256 tiles stays with the
#     aligned ring); when its tiles fit one wave (one workgroup per CU) the
#     3-stage fragment-prefetch interleave form of it, which hides the DMA
#     latency no co-resident workgroup can; for K % 64 == 32 a 128x128x32 /
#     256x128x32 interleave ring;
#   * a pad ring (the 16-wave persistent ring, P = #CUs, chiplet chunk 32)
#     for ragged M / N.
# Both carry pad_m / pad_n with exact DTL buffer bounds (``dtl_tight_rsrc``)
# and the full-tile epilogue fast path (``pad_fast_tile``), so any M / N with
# K % tile_k == 0 is admitted by the shape gate.
#
# K gates. Where the aligned default candidate can take the
# shape (M, N, K multiples of its tile) the small ring is held to the same K
# floor as the aligned ring (``DTL_RING_ADMIT_RULES`` k_lo), and the
# K % 64 == 32 bodies to their own floors (no ring body pays at the smallest
# K). Ragged shapes have no other candidate, so they are not K-gated; at small
# K on a moderate grid the small ring takes them instead of the pad ring.
# ---------------------------------------------------------------------------
DTL_RING_SMALL_ARCHES = ("gfx950",)
DTL_RING_PAD_ARCHES = ("gfx950",)
# the small-tile ring takes over below this last-wave fill of 256x256 tiles
DTL_RING_SMALL_MAX_FILL = 0.75
# ... and only from half a wave of its own tiles on (smaller problems are
# latency-bound and were not calibrated: they keep the cshuffle default)
DTL_RING_SMALL_MIN_TILES_PER_CU = 0.5
# K % 64 == 32: the 128x128x32 body on up to this many waves of 256x128 tiles,
# else the 256x128x32 AGPR body; their K floors (aligned shapes)
DTL_RING_SMALL_K32_C128_MAX_WAVES = 2
DTL_RING_SMALL_K32_MIN_K = 96
DTL_RING_SMALL_K32_AGPR_MIN_K = 160
# ragged shapes: the small ring instead of the pad ring up to this K and below
# this many 256x256 output tiles
DTL_RING_SMALL_RAGGED_MAX_K = 512
DTL_RING_SMALL_RAGGED_MAX_TILES = 1280
# the K % 64 == 0 small ring as the 3-stage fragment-prefetch interleave body
# when its 128x128 tiles fit one wave (one workgroup per CU) and K >= this
DTL_RING_SMALL_S3_MIN_K = 128
DTL_RING_SMALL_S3_SINGLE_WAVE = True

# Very large grids (>= DTL_BIG_GRID_TILES 256x256 tiles; the rule was checked
# on shapes held out when it was written). Chiplet wgm 4 (narrower super-tile
# groups) pays at small K, on very large tile counts at any K, once the A + B
# footprint passes about three quarters of the MALL, and on wide grids from a
# moderate K on. There the aligned candidate takes the 16-wave ring with wgm 4
# + iglp_opt(1) while A + B fit the MALL, and the persistent 4-wave AGPR
# interleave body beyond it and at small K; ragged shapes take the AGPR
# interleave pad body, on a slightly lower tile count than the aligned regime.
# Elsewhere (moderate tile count, a small footprint) the 16-wave wgm-8 ring
# stays. The effect is not L2 reuse (the L2 hit rate drops with wgm 4): the
# narrower groups spread the concurrent DTL streams over the XCDs, which
# shortens the DTL latency per K tile.
DTL_BIG_GRID_TILES = 9000
DTL_BIG_GRID_ALL_K_TILES = 30000
DTL_BIG_GRID_FOOTPRINT_BYTES = 192 << 20
DTL_BIG_GRID_MALL_BYTES = 256 << 20
DTL_BIG_GRID_SMALL_K = 128
DTL_BIG_GRID_WIDE_TILES = 20000
DTL_BIG_GRID_RAGGED_WIDE_TILES = 12000
DTL_BIG_GRID_WIDE_MIN_K = 512


def _tiles(M: int, N: int, tm: int, tn: int) -> int:
    return -(-int(M) // tm) * -(-int(N) // tn)


def dtl_big_grid_body(req: GemmRequest, *, ragged: bool = False) -> Optional[str]:
    """The big-grid body of the 256x256 ring candidates for ``req``:
    ``"agpr_ilv"`` (persistent AGPR interleave body, wgm 4), ``"ring_w4"`` (16-wave ring, wgm 4 + iglp_opt(1);
    aligned only) or ``None`` (the candidate's own ring body). See the
    :data:`DTL_BIG_GRID_TILES` block."""
    if req.arch != "gfx950":
        return None
    t = _tiles(req.M, req.N, 256, 256)
    if t < DTL_BIG_GRID_TILES:
        return None
    K = int(req.K)
    footprint = (int(req.M) + int(req.N)) * K * 2
    if K <= DTL_BIG_GRID_SMALL_K:
        return "agpr_ilv"
    wide = DTL_BIG_GRID_RAGGED_WIDE_TILES if ragged else DTL_BIG_GRID_WIDE_TILES
    if not (
        t >= DTL_BIG_GRID_ALL_K_TILES
        or footprint > DTL_BIG_GRID_FOOTPRINT_BYTES
        or (t >= wide and K >= DTL_BIG_GRID_WIDE_MIN_K)
    ):
        return None
    if ragged or footprint > DTL_BIG_GRID_MALL_BYTES:
        return "agpr_ilv"
    return "ring_w4"


def dtl_big_grid(req: GemmRequest) -> bool:
    """gfx950 problem whose aligned big-grid body is the persistent AGPR
    interleave body
    (:func:`dtl_big_grid_body`)."""
    return dtl_big_grid_body(req) == "agpr_ilv"


def agpr_ilv_big_grid_tile_trait(arch: str, **extra) -> tuple[TileSpec, TraitSpec]:
    """The AGPR interleave body (2x2 waves of 128x128, 16x16x32 atoms, tile_k 32, 3-stage ring,
    fragment prefetch + merged MFMA interleave, barrier after 2 MFMAs, swizzle
    R = 1, DTL resource hoist, AGPR accumulators) persistent on every CU with
    chiplet wgm 4 / chunk 64."""
    tile = TileSpec(
        tile_m=256,
        tile_n=256,
        tile_k=32,
        warp_m=2,
        warp_n=2,
        warp_k=1,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=32,
    )
    _, base = dtl_pingpong_tile_trait(arch)
    trait = replace(
        base,
        chiplet_wgm=4,
        chiplet_chunk_size=64,
        dtl_stages=3,
        interwave_setprio=True,
        dtl_frag_prefetch=True,
        acc_agpr=True,
        mfma_interleave=True,
        ilv_merge=True,
        ilv_barrier_after=2,
        lds_swizzle_r=1,
        dtl_hoist=True,
        persistent_ctas=arch_num_cus(arch),
        **extra,
    )
    return tile, trait


# ---------------------------------------------------------------------------
# Row-interleave ("rilv") body: the 4-wave AGPR body (2x2 waves of 128x128,
# 16x16x32 atoms, tile_k 64, alias-scoped 2-stage DTL ring, fragment prefetch,
# DTL resource hoist) with an LDS row interleave in hipBLASLt's layout
# (``lds_row_interleave``: unswizzled 128-byte rows, one base VGPR per fragment
# stream, no VALU in the K loop) and the 16-byte non-temporal vector epilogue
# straight from the accumulators (``rilv_vec_epilogue`` + ``rve_store_nt``).
# Two K-tile schedules, picked by K:
#   * K >= RILV_HB_MIN_K, or >= RILV_HB_BIG_TILES output tiles with
#     K >= RILV_HB_BIG_MIN_K: a slot schedule in hipBLASLt's order
#     (``hb_sched``: one fragment read every 2 MFMAs, release / acquire
#     barriers), chiplet wgm 16 / chunk 8 (long K / big grids);
#   * otherwise the merged MFMA interleave (``ilv_merge``), persistent on every
#     CU with chiplet chunk 32 for K <= RILV_PERSIST_MAX_K, else chiplet wgm 16
#     / chunk 8.
# Aligned problems only (the vector epilogue has no M / N bounds path; padded
# shapes take ``rilv_pad_tile_trait`` or the ring bodies) with at least
# RILV_MIN_TILES 256x256 output tiles and K >= RILV_MIN_K.
#
# Threshold provenance: the RILV_* / SMALL_TILE_SHORTK_* K and tile-count
# thresholds were calibrated on gfx950 timings of a fixed shape set (every
# kernel checked against a full-output f64 reference); several sit on K values
# or tile counts of that set and were not validated on held-out shapes between
# them. They are policy defaults, not derived bounds: re-calibrate them when
# the shape population changes (dsl_docs gemm-optimization-rocke.md, section
# "Row-interleave dispatch classes").
# ---------------------------------------------------------------------------
RILV_ARCHES = ("gfx950",)
RILV_HB_MIN_K = 12288
RILV_HB_BIG_TILES = 4096
RILV_HB_BIG_MIN_K = 4096
RILV_PERSIST_MAX_K = 2304
# at least one full wave of 256x256 tiles; smaller grids keep the ring bodies
RILV_MIN_TILES = 256
# smallest calibrated K; below it the short-K / ring bodies keep the shape
RILV_MIN_K = 768
# a 32-byte LDS pad per KiB (``rilv_pad``, the hipBLASLt LDS layout). With the
# 16-byte pad every ds_read_b128 takes LDS bank-conflict cycles on gfx950; with
# 32 bytes the reads are conflict free.
RILV_LDS_PAD = 32
# The sia K-tile schedule (``sia_sched`` + ``sia_plr`` + ``sia_dtl_spread``:
# operand-grouped next-atom reads every 2 MFMAs, one release, the acquire
# RILV_SIA_ACQ_GAP MFMAs into the last atom, DTL issues on the free MFMA
# slots) as an alternative to ``hb_sched`` on the long-K class. It is used by
# the ``"sia"`` RILV_BODY (bf16 only) and by the ragged long-K pad class when
# RILV_PAD_HB_TUNED is off.
RILV_SIA_DTYPES = ("bf16",)
RILV_SIA_ACQ_GAP = 12


def _beat_k_bad(K: int) -> bool:
    """K the beat bodies cannot take: not whole 64-element tiles, unless the
    peeled K tail (``G1_BEAT_K_TAIL``; K % 8 == 0) serves it."""
    K = int(K)
    if K % 64 == 0:
        return False
    return not (
        G1_BEAT_K_TAIL
        and G1_ADAPT
        and G1_DTL_K_TAIL
        and K % G1_DTL_K_TAIL_ALIGN == 0
        and -(-K // 64) >= G1_DTL_K_TAIL_MIN_TILES
    )


def rilv_admits(req: GemmRequest) -> bool:
    """gfx950 problem the row-interleave body serves (aligned, enough tiles)."""
    if req.arch not in RILV_ARCHES:
        return False
    M, N, K = int(req.M), int(req.N), int(req.K)
    if M % 256 or N % 256 or _beat_k_bad(K) or K < RILV_MIN_K:
        return False
    return (M // 256) * (N // 256) >= RILV_MIN_TILES


def rilv_hb_schedule(M: int, N: int, K: int) -> bool:
    """True where the row-interleave body takes the ``hb_sched`` K-tile
    schedule (long K, or a big grid at moderate K; see the block comment)."""
    tiles = -(-int(M) // 256) * -(-int(N) // 256)
    return int(K) >= RILV_HB_MIN_K or (
        tiles >= RILV_HB_BIG_TILES and int(K) >= RILV_HB_BIG_MIN_K
    )


def rilv_tile_trait(
    arch: str, hb: bool, persistent: bool = True, sia: bool = False
) -> tuple[TileSpec, TraitSpec]:
    """The row-interleave tile / trait: the ``hb_sched`` body (``hb``; with
    ``sia`` the ``sia_sched`` schedule of the same class) or the
    merged-interleave body, persistent with chunk 32 (``persistent``) or with
    the wgm 16 / chunk 8 grid (see the block comment)."""
    tile = TileSpec(
        tile_m=256,
        tile_n=256,
        tile_k=64,
        warp_m=2,
        warp_n=2,
        warp_k=1,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=32,
    )
    _, base = dtl_pingpong_tile_trait(arch)
    common = dict(
        lds_swizzle=False,
        interwave_setprio=False,
        dtl_stages=2,
        acc_agpr=True,
        dtl_frag_prefetch=True,
        mfma_interleave=True,
        dtl_hoist=True,
        lds_row_interleave=True,
        rilv_pad=RILV_LDS_PAD,
        rilv_vec_epilogue=True,
        rve_store_nt=True,
    )
    if hb and sia:
        trait = replace(
            base,
            chiplet_wgm=16,
            chiplet_chunk_size=8,
            sia_sched=True,
            sia_plr=True,
            sia_dtl_spread=True,
            sia_acq_gap=RILV_SIA_ACQ_GAP,
            **common,
        )
    elif hb:
        trait = replace(
            base,
            chiplet_wgm=16,
            chiplet_chunk_size=8,
            hb_sched=True,
            hb_read_every=2,
            **common,
        )
    elif persistent:
        trait = replace(
            base,
            chiplet_wgm=8,
            chiplet_chunk_size=32,
            ilv_merge=True,
            persistent_ctas=arch_num_cus(arch),
            **common,
        )
    else:
        trait = replace(
            base,
            chiplet_wgm=16,
            chiplet_chunk_size=8,
            ilv_merge=True,
            **common,
        )
    return tile, trait


def rilv_spec_tile_trait(req: GemmRequest) -> tuple[TileSpec, TraitSpec]:
    """The row-interleave tile / trait the dispatcher selects for an aligned
    ``req`` (:func:`rilv_body`: the tuned body by default; the other bodies
    stay selectable)."""
    hb = rilv_hb_schedule(req.M, req.N, req.K)
    persistent = int(req.K) <= RILV_PERSIST_MAX_K
    body = rilv_body()
    if body == "tuned":
        tile, trait = rilv_tuned_tile_trait(req.arch, req.M, req.N, req.K)
        if rilv_hb_loop() == "asm_loop":
            trait = rilv_asm_loop_tuned_trait(trait)
        return tile, trait
    if body == "asm_loop":
        tile, trait = rilv_tile_trait(req.arch, hb, persistent=persistent)
        return tile, rilv_asm_loop_trait(trait, hb)
    return rilv_tile_trait(
        req.arch,
        hb,
        persistent=persistent,
        sia=body == "sia" and req.dtype in RILV_SIA_DTYPES,
    )


# ---------------------------------------------------------------------------
# Tuned row-interleave body: one body for every aligned row-interleave class.
# The hb_sched K-tile schedule (one fragment read every 2 MFMAs, release /
# acquire barriers) on the row-interleave body with the conflict-free 32-byte
# LDS pad (``rilv_pad``), the DTL origin in the buffer resource base with an
# inline soffset of 0 (``dtl_soff0``: one VMEM issue cycle per DTL instead of
# two), the per-tile DTL address SALU spread over free MFMA slots
# (``dtl_addr_spread``), the first DTL one slot after the release
# (``hb_dtl_delay``) and an unrolled K loop (``kloop_unroll``). Grid:
# persistent on every CU (chiplet wgm 8 / chunk 32) when K <=
# RILV_PERSIST_MAX_K and the grid has more than RILV_PERSIST_MIN_WAVES waves
# of 256x256 tiles (the workgroup turnover the persistent loop saves pays only
# beyond a couple of waves); else non-persistent chiplet wgm 16 / chunk 8.
# kloop_unroll RILV_KLOOP_UNROLL_LONG for long K (> RILV_PERSIST_MAX_K),
# RILV_KLOOP_UNROLL_SHORT otherwise (a longer unroll leaves too long a tail at
# short K), and two main-loop copies per SIMD parity (``hb_skew``).
#
# Body selection (``rilv_body``; ROCKE_GEMM_RILV_BODY overrides RILV_BODY):
#   "tuned"    the body above (default);
#   "asm_loop" the hand-ordered asm-loop body (``rilv_asm_loop_trait``);
#   "sia"      the base body with the sia schedule on the hb class (bf16);
#   "base"     the base row-interleave body (``rilv_tile_trait``).
# The non-default bodies are retained as selectable alternatives; they are not
# dispatched by default.
# ---------------------------------------------------------------------------
RILV_BODY_ENV = "ROCKE_GEMM_RILV_BODY"
RILV_BODIES = ("tuned", "asm_loop", "sia", "base")
RILV_BODY = "tuned"
RILV_PERSIST_MIN_WAVES = 2
RILV_KLOOP_UNROLL_LONG = 4
RILV_KLOOP_UNROLL_SHORT = 2
# two main-loop copies per SIMD parity on the hb_sched body (the odd-SIMD copy
# reads / issues one MFMA later, so SIMD pairs do not hit LDS in the same
# slot; ``hb_skew``)
RILV_HB_SKEW = 1


def rilv_body() -> str:
    """The row-interleave body the dispatcher builds: ``ROCKE_GEMM_RILV_BODY``
    if set, else :data:`RILV_BODY`. An unknown value raises."""
    v = os.environ.get(RILV_BODY_ENV, "").strip().lower() or RILV_BODY
    if v not in RILV_BODIES:
        raise ValueError(f"{RILV_BODY_ENV}={v!r}: expected one of {RILV_BODIES}")
    return v


def rilv_persistent(M: int, N: int, K: int, arch: str) -> bool:
    """True where the tuned row-interleave body runs persistent (see the
    block)."""
    tiles = -(-int(M) // 256) * -(-int(N) // 256)
    return int(
        K
    ) <= RILV_PERSIST_MAX_K and tiles > RILV_PERSIST_MIN_WAVES * arch_num_cus(arch)


# ---------------------------------------------------------------------------
# Instruction economy on the tuned body: one explicit lgkmcnt(0) at the end of
# every K tile (LLVM then drops its per-MFMA lgkmcnt(N) waits in the tile;
# ``hb_tile_lgkm0``) and the per-tile DTL address steps placed after the
# tile's last fragment read (``das_late``; otherwise LLVM floats them into the
# read gaps between two MFMAs). Both combine with ``mfma_serpentine``.
# ---------------------------------------------------------------------------
RILV_HB_TILE_LGKM0 = 1
RILV_DAS_LATE = True


def rilv_tuned_trait(trait: TraitSpec, *, kloop_unroll: int) -> TraitSpec:
    """The tuned body on a row-interleave trait (grid fields kept), with the
    instruction-economy levers (``RILV_HB_TILE_LGKM0`` / ``RILV_DAS_LATE``)."""
    return replace(
        trait,
        ilv_merge=False,
        sia_sched=False,
        sia_plr=False,
        sia_dtl_spread=False,
        sia_acq_gap=3,
        hb_sched=True,
        hb_read_every=2,
        dtl_soff0=True,
        dtl_addr_spread=True,
        hb_dtl_delay=1,
        kloop_unroll=kloop_unroll,
        hb_skew=RILV_HB_SKEW,
        hb_tile_lgkm0=RILV_HB_TILE_LGKM0,
        das_late=RILV_DAS_LATE,
    )


# Optional (off by default): the long-K aligned class on a different chiplet
# group (``chiplet_wgm`` / ``chiplet_chunk_size``) for mid-size grids of
# 256x256 tiles; () = the wgm 16 / chunk 8 grid everywhere.
RILV_LONGK_GRID: tuple = ()
RILV_LONGK_MIN_K = 8192
RILV_LONGK_TILES = (2048, 6400)  # inclusive 256x256-tile range


def rilv_longk_grid(M: int, N: int, K: int) -> bool:
    """Whether the long-K aligned body takes ``RILV_LONGK_GRID``."""
    if not RILV_LONGK_GRID or int(K) < RILV_LONGK_MIN_K:
        return False
    tiles = (int(M) // 256) * (int(N) // 256)
    return RILV_LONGK_TILES[0] <= tiles <= RILV_LONGK_TILES[1]


def rilv_tuned_tile_trait(
    arch: str, M: int, N: int, K: int
) -> tuple[TileSpec, TraitSpec]:
    """Tile / trait of the tuned row-interleave body for an aligned M x N x K."""
    tile, base = rilv_tile_trait(arch, False, persistent=False)
    if rilv_persistent(M, N, K, arch):
        grid = dict(
            chiplet_wgm=8, chiplet_chunk_size=32, persistent_ctas=arch_num_cus(arch)
        )
    elif rilv_longk_grid(M, N, K):
        grid = dict(
            chiplet_wgm=RILV_LONGK_GRID[0],
            chiplet_chunk_size=RILV_LONGK_GRID[1],
            persistent_ctas=0,
        )
    else:
        grid = dict(chiplet_wgm=16, chiplet_chunk_size=8, persistent_ctas=0)
    ku = (
        RILV_KLOOP_UNROLL_LONG
        if int(K) > RILV_PERSIST_MAX_K
        else RILV_KLOOP_UNROLL_SHORT
    )
    trait = rilv_tuned_trait(replace(base, **grid), kloop_unroll=ku)
    if RILV_SERPENTINE:
        trait = replace(trait, mfma_serpentine=True)
    return tile, trait


# ---------------------------------------------------------------------------
# XCD-local stream-K tail on the tuned row-interleave body (opt-in helpers,
# not dispatched). Aligned long-K grids with many waves of 256x256 tiles and an
# almost empty last wave lose most of a wave to quantisation; the persistent
# stream-K loop (``streamk`` + ``sk_xcd_local``: the owner and partners of a
# split tile on one XCD, partials through that XCD's L2) spreads the tail over
# every CU. ``rilv_streamk_tail_admits`` is the narrow class where that paid on
# gfx950: K >= RILV_STREAMK_TAIL_MIN_K, at least RILV_STREAMK_TAIL_MIN_WAVES
# waves and a last wave filled to at most RILV_STREAMK_TAIL_MAX_FILL. Outside
# it the stream-K loop lost to the data-parallel body. The thresholds are fitted
# on a small set of shapes, the gain varied between nodes, and a held-out K
# below RILV_STREAMK_TAIL_MIN_K also gained, so the rule is not validated: the
# dispatcher does not call these helpers. A stream-K kernel also takes two
# workspace arguments (``streamk_workspace_bytes``: flags zeroed once, and
# partials) that the dispatch binding does not allocate; a caller that uses
# the helpers must allocate them.
# ---------------------------------------------------------------------------
RILV_STREAMK_TAIL_MIN_K = 12288
RILV_STREAMK_TAIL_MIN_WAVES = 12
RILV_STREAMK_TAIL_MAX_FILL = 0.25


def rilv_streamk_tail_admits(M: int, N: int, K: int, arch: str) -> bool:
    """True for the aligned long-K grids with a near-empty last wave that the
    XCD-local stream-K tail targets (see the block; not dispatched)."""
    if arch not in RILV_ARCHES:
        return False
    M, N, K = int(M), int(N), int(K)
    if M % 256 or N % 256 or K % 64 or K < RILV_STREAMK_TAIL_MIN_K:
        return False
    cus = arch_num_cus(arch)
    tiles = (M // 256) * (N // 256)
    if tiles < RILV_STREAMK_TAIL_MIN_WAVES * cus:
        return False
    last = tiles % cus  # tiles of the last (partial) wave; 0 = whole waves
    return 0 < last <= RILV_STREAMK_TAIL_MAX_FILL * cus


def rilv_streamk_tail_trait(trait: TraitSpec, arch: str) -> TraitSpec:
    """``trait`` (the tuned row-interleave body) as a persistent XCD-local
    stream-K loop on every CU. Stream-K requires kloop_unroll 1; hb_skew is
    off on the stream-K loop."""
    return replace(
        trait,
        persistent_ctas=arch_num_cus(arch),
        streamk=True,
        sk_xcd_local=True,
        kloop_unroll=1,
        hb_skew=0,
    )


# ---------------------------------------------------------------------------
# K-tile emission of the tuned body (``rilv_hb_loop``; ROCKE_GEMM_RILV_HB_LOOP
# overrides RILV_HB_LOOP): "ir" (default) emits the hb_sched K tile through the
# IR scheduler; "asm_loop" emits it as a hand-ordered loop in the Tensile order
# (``asm_loop`` = "tensile", the two SIMD-parity loop copies via
# ``asm_loop_skew``), optionally with the address-step spread / DTL delay
# native to the asm table (RILV_ASM_LOOP_ADDR_SPREAD) and a StaggerU start
# offset on non-persistent grids (RILV_ASM_LOOP_STAGGER_U). "asm_loop" and both
# options are retained, default-off alternatives: on gfx950 they did not beat
# the "ir" emission.
# ---------------------------------------------------------------------------
RILV_HB_LOOP_ENV = "ROCKE_GEMM_RILV_HB_LOOP"
RILV_HB_LOOPS = ("ir", "asm_loop")
RILV_HB_LOOP = "ir"
# The boustrophedon MFMA order (``mfma_serpentine``: odd A rows walk the B
# fragments backwards, so no MFMA switches both operands at a row change) on
# the tuned body, the pad bodies and the short-K merged-interleave body: the
# same work with less operand-path toggling per MFMA.
RILV_SERPENTINE = True
RILV_ASM_LOOP_ADDR_SPREAD = False  # dtl_addr_spread + hb_dtl_delay 1 in the asm table
RILV_ASM_LOOP_STAGGER_U = 0  # StaggerU on non-persistent asm-loop grids (0 = off)
RILV_ASM_LOOP_STAGGER_STRIDE = 1


def rilv_hb_loop() -> str:
    """K-tile emission of the tuned row-interleave body:
    ``ROCKE_GEMM_RILV_HB_LOOP`` if set, else :data:`RILV_HB_LOOP`. An unknown
    value raises."""
    v = os.environ.get(RILV_HB_LOOP_ENV, "").strip().lower() or RILV_HB_LOOP
    if v not in RILV_HB_LOOPS:
        raise ValueError(f"{RILV_HB_LOOP_ENV}={v!r}: expected one of {RILV_HB_LOOPS}")
    return v


def rilv_asm_loop_tuned_trait(trait: TraitSpec) -> TraitSpec:
    """The asm-loop emission of a tuned row-interleave trait (same grid and
    kloop_unroll; see the block above)."""
    trait = replace(
        trait,
        hb_skew=0,
        hb_read_every=1,
        asm_loop="tensile",
        asm_loop_skew=1,
        dtl_addr_spread=RILV_ASM_LOOP_ADDR_SPREAD,
        hb_dtl_delay=1 if RILV_ASM_LOOP_ADDR_SPREAD else 0,
        # the instruction-economy levers apply to the IR hb_sched loop only
        hb_tile_lgkm0=0,
        das_late=False,
    )
    if RILV_ASM_LOOP_STAGGER_U and not trait.persistent_ctas:
        trait = replace(
            trait,
            stagger_u=RILV_ASM_LOOP_STAGGER_U,
            stagger_u_stride=RILV_ASM_LOOP_STAGGER_STRIDE,
        )
    return trait


# ---------------------------------------------------------------------------
# Hand-ordered main loop (the "asm_loop" RILV_BODY): the base row-interleave
# body with every K tile emitted in the Tensile loop order (``asm_loop`` =
# "tensile": per-operand release AND acquire, 4 barriers per tile, every
# fragment read / DTL issue / wait pinned to its MFMA slot) and two loop copies
# per SIMD parity (``asm_loop_skew`` = 1: odd-SIMD waves read one MFMA later,
# so SIMD pairs do not hit the LDS / TA in the same slot). The long-K class
# uses chiplet chunk RILV_ASM_LOOP_HB_CHUNK.
# ---------------------------------------------------------------------------
RILV_ASM_LOOP_HB_CHUNK = 4


def rilv_asm_loop_trait(trait: TraitSpec, hb: bool) -> TraitSpec:
    """The asm-loop version of a row-interleave trait (same grid; the long-K
    class with chiplet chunk ``RILV_ASM_LOOP_HB_CHUNK``)."""
    trait = replace(
        trait,
        ilv_merge=False,
        hb_sched=True,
        hb_read_every=1,
        asm_loop="tensile",
        asm_loop_skew=1,
    )
    if hb:
        trait = replace(trait, chiplet_chunk_size=RILV_ASM_LOOP_HB_CHUNK)
    return trait


# ---------------------------------------------------------------------------
# The row-interleave body on two more shape classes:
#   * ragged N (or M): the vector epilogue's branch-free bounds path
#     (``rve_pad`` 2: buffer stores over C, out-of-range lanes dropped by the
#     range check) with pad + exact DTL bounds (``dtl_tight_rsrc``) and the
#     full-tile fast path, in place of the pad ring / AGPR interleave bodies;
#   * aligned short K (RILV_SHORTK_MIN_K <= K < RILV_MIN_K) with >= one wave of
#     256x256 tiles filling the last wave to >= RILV_SHORTK_MIN_FILL: the
#     merged interleave body, non-persistent chiplet wgm 16 / chunk 8, in place
#     of the small-tile ring / the 16-wave ring.
# ---------------------------------------------------------------------------
RILV_SHORTK_MIN_K = 384
RILV_SHORTK_MIN_FILL = 0.6
RILV_SHORTK_DTYPES = ("fp16", "bf16")


def rilv_shortk_admits(req: GemmRequest) -> bool:
    """gfx950 aligned short-K problem the row-interleave body serves (see the
    block)."""
    if req.arch not in RILV_ARCHES:
        return False
    if normalize_dtype(req.dtype) not in RILV_SHORTK_DTYPES:
        return False
    M, N, K = int(req.M), int(req.N), int(req.K)
    if M % 256 or N % 256 or _beat_k_bad(K) or not RILV_SHORTK_MIN_K <= K < RILV_MIN_K:
        return False
    if (M // 256) * (N // 256) < RILV_MIN_TILES:
        return False
    return wave_fill(M, N, 256, 256, req.arch) >= RILV_SHORTK_MIN_FILL


def rilv_shortk_tile_trait(req: GemmRequest) -> tuple[TileSpec, TraitSpec]:
    """The merged-interleave row-interleave body, non-persistent wgm 16 /
    chunk 8; the persistent tuned body where ``rilv_persistent`` holds (many
    waves of short tiles)."""
    if rilv_body() == "tuned" and rilv_persistent(req.M, req.N, req.K, req.arch):
        return rilv_tuned_tile_trait(req.arch, req.M, req.N, req.K)
    if (
        RILV_SHORTK_SMALL_TILE
        and int(req.M) % 128 == 0
        and int(req.N) % 192 == 0
        and wave_fill(req.M, req.N, 256, 256, req.arch)
        < RILV_SHORTK_SMALL_TILE_MAX_FILL
    ):
        # the small-tile 128x192 body (exact tiles)
        return small_tile_128x192_exact_tile_trait()
    tile, trait = rilv_tile_trait(req.arch, False, persistent=False)
    if RILV_SERPENTINE and rilv_body() == "tuned":
        # the serpentine MFMA order on the merged-interleave body too
        trait = replace(trait, mfma_serpentine=True)
    return tile, trait


# The short-K row-interleave class (non-persistent) takes the small-tile body
# (128x192x64, 4x2 waves, 2-stage DTL ring, no chiplet swizzle, exact tiles:
# dtl_soff0 + cshuffle epilogue) where its tiles divide M and N and the
# 256x256 tiles fill their waves poorly (below
# ``RILV_SHORTK_SMALL_TILE_MAX_FILL``; with better-filled 256x256 waves the
# bigger tile stays ahead), instead of the merged-interleave 256x256 body.
RILV_SHORTK_SMALL_TILE = True
RILV_SHORTK_SMALL_TILE_MAX_FILL = 0.7


def small_tile_128x192_exact_tile_trait() -> tuple[TileSpec, TraitSpec]:
    """The small-tile 128x192x64 body with exact tiles (see
    ``small_tile_shortk_tile_trait``)."""
    trait = replace(
        _ring_pad_trait(),
        chiplet_swizzle=False,
        pad_m=False,
        pad_n=False,
        dtl_tight_rsrc=False,
        pad_fast_tile=False,
        dtl_soff0=True,
        epilogue="cshuffle",
    )
    return _ring_tile(128, 192, 64, 4, 2), trait


def rilv_pad_admits(req: GemmRequest) -> bool:
    """gfx950 ragged-M/N problem for the padded row-interleave body: N % 8 == 0 (16-byte
    vector stores), K % 64 == 0, K >= RILV_MIN_K, >= RILV_MIN_TILES
    tiles filling the last wave to >= DTL_RING_SMALL_MAX_FILL (else the
    small-tile ring), and the 32-bit DTL / C buffer ranges."""
    if req.arch not in RILV_ARCHES:
        return False
    M, N, K = int(req.M), int(req.N), int(req.K)
    if not (M % 256 or N % 256) or N % 8 or _beat_k_bad(K) or K < RILV_MIN_K:
        return False
    if max(M, N) * K * 2 >= 2**31 or M * N * 2 >= 2**32:
        return False
    if -(-M // 256) * -(-N // 256) < RILV_MIN_TILES:
        return False
    return wave_fill(M, N, 256, 256, req.arch) >= DTL_RING_SMALL_MAX_FILL


def rilv_pad_tile_trait(req: GemmRequest) -> tuple[TileSpec, TraitSpec]:
    """The row-interleave pick for ``req`` with pad on the ragged
    dimension(s), exact DTL bounds, the full-tile fast path and the
    branch-free vector epilogue bounds (``rve_pad`` 2)."""
    M, N, K = int(req.M), int(req.N), int(req.K)
    hb = rilv_hb_schedule(M, N, K)
    tiles = -(-M // 256) * -(-N // 256)
    # persistent chunk 32 also on big grids with K up to the hb threshold
    persistent = K <= RILV_PERSIST_MAX_K or tiles >= RILV_HB_BIG_TILES
    tuned = rilv_body() == "tuned"
    if tuned:
        # the hb_sched body (persistent, or the sia schedule on the long-K
        # class); dtl_soff0 does not combine with the exact DTL bounds the pad
        # path needs (which already issue with soffset 0)
        tile, trait = rilv_tile_trait(req.arch, hb, persistent=persistent, sia=hb)
        if hb and RILV_PAD_HB_GRID:
            # the long-K pad class on the wgm 32 / small-chunk grid
            trait = replace(
                trait,
                chiplet_wgm=RILV_PAD_HB_GRID[0],
                chiplet_chunk_size=RILV_PAD_HB_GRID[1],
            )
        if hb and RILV_PAD_HB_TUNED:
            # the tuned body (hb_sched + its levers, exact bounds on the padded
            # operand only) instead of the sia schedule, non-persistent
            trait = replace(
                trait,
                sia_sched=False,
                sia_plr=False,
                sia_dtl_spread=False,
                sia_acq_gap=3,
                hb_sched=True,
                hb_read_every=2,
                dtl_addr_spread=True,
                dtl_tight_ragged_only=True,
                hb_dtl_delay=1,
                hb_skew=RILV_HB_SKEW,
                kloop_unroll=RILV_KLOOP_UNROLL_LONG,
            )
        if not hb:
            trait = replace(
                trait,
                ilv_merge=False,
                hb_sched=True,
                hb_read_every=2,
                chiplet_wgm=8 if persistent else 16,
                chiplet_chunk_size=32 if persistent else 8,
                persistent_ctas=arch_num_cus(req.arch) if persistent else 0,
            )
            if persistent and RILV_PAD_PERSIST_TUNED:
                # the tuned body's levers on the persistent pad class
                # (dtl_addr_spread also builds the exact bounds; exact bounds
                # only on the padded operand)
                trait = replace(
                    trait,
                    dtl_addr_spread=True,
                    dtl_tight_ragged_only=True,
                    hb_dtl_delay=1,
                    hb_skew=RILV_HB_SKEW,
                    kloop_unroll=(
                        RILV_KLOOP_UNROLL_LONG
                        if K > RILV_PERSIST_MAX_K
                        else RILV_KLOOP_UNROLL_SHORT
                    ),
                )
    else:
        tile, trait = rilv_tile_trait(req.arch, hb, persistent=persistent)
    trait = replace(
        trait,
        pad_m=bool(int(req.M) % 256),
        pad_n=bool(int(req.N) % 256),
        dtl_tight_rsrc=True,
        pad_fast_tile=tiles >= RILV_PAD_FAST_TILE_MIN_TILES,
        rve_pad=2,
    )
    if tuned and RILV_SERPENTINE:
        # the serpentine MFMA order on the pad bodies too (hb_sched / sia)
        trait = replace(trait, mfma_serpentine=True)
    if tuned and trait.hb_sched and not trait.asm_loop:
        # the instruction-economy levers on the hb_sched pad bodies
        # (``RILV_PAD_HB_TILE_LGKM0`` / ``RILV_PAD_DAS_LATE``, default off)
        if RILV_PAD_HB_TILE_LGKM0:
            trait = replace(trait, hb_tile_lgkm0=RILV_HB_TILE_LGKM0)
        if RILV_PAD_DAS_LATE and trait.dtl_addr_spread:
            trait = replace(trait, das_late=RILV_DAS_LATE)
    return tile, trait


# ---------------------------------------------------------------------------
# Ragged / pad row-interleave class:
#   * the long-K pad class (non-persistent) is grid bound: chiplet wgm 32 with
#     a small chunk (RILV_PAD_HB_GRID) instead of wgm 16 / chunk 8;
#   * the persistent pad class takes the tuned body's levers (dtl_addr_spread
#     with exact bounds on the padded operand only, hb_dtl_delay 1, hb_skew,
#     kloop_unroll; RILV_PAD_PERSIST_TUNED).
# ---------------------------------------------------------------------------
RILV_PAD_HB_GRID = (32, 2)  # (chiplet_wgm, chiplet_chunk_size); () = wgm 16 / chunk 8
RILV_PAD_PERSIST_TUNED = True
# long-K pad class: the tuned body (hb_sched + dtl_addr_spread +
# dtl_tight_ragged_only + hb_dtl_delay + hb_skew + kloop_unroll) instead of the
# sia schedule
RILV_PAD_HB_TUNED = True
# The instruction-economy levers on the hb_sched pad bodies, default off.
# ``das_late`` with the exact DTL bounds (``dtl_tight_rsrc`` /
# ``dtl_tight_ragged_only``) makes LLVM spill the accumulators to scratch in
# the main loop (much slower, results still exact), so it stays off there (as
# does ``hb_read_every`` 1, same failure); ``hb_tile_lgkm0`` alone does not
# pay on these bodies.
RILV_PAD_HB_TILE_LGKM0 = False
RILV_PAD_DAS_LATE = False

# the full-tile fast path (``pad_fast_tile``) from this many 256x256 tiles on
# (it pays on large ragged grids and costs on small ones; the threshold lies
# between the calibrated grid sizes)
RILV_PAD_FAST_TILE_MIN_TILES = 2048


def ring_w4_tile_trait(arch: str) -> tuple[TileSpec, TraitSpec]:
    """The aligned ring body with chiplet wgm 4 + iglp_opt(1) (its ISA does not
    depend on what the process compiled before)."""
    tile, trait = dtl_pingpong_tile_trait(arch)
    return tile, replace(trait, chiplet_wgm=4, iglp_opt=1)


def big_grid_tile_trait(req: GemmRequest) -> Optional[tuple[TileSpec, TraitSpec]]:
    """Tile and trait of the aligned candidate's big-grid body, or ``None``."""
    body = dtl_big_grid_body(req)
    if body == "agpr_ilv":
        return agpr_ilv_big_grid_tile_trait(req.arch)
    if body == "ring_w4":
        return ring_w4_tile_trait(req.arch)
    return None


def wave_fill(M: int, N: int, tile_m: int, tile_n: int, arch: str) -> float:
    """Fill of the last wave of ``tile_m x tile_n`` output tiles over the CUs
    (1.0 = whole waves; 0.53 = 1.05 waves rounded up to 2)."""
    tiles = _tiles(M, N, tile_m, tile_n)
    waves = tiles / arch_num_cus(arch)
    return waves / max(1, -(-tiles // arch_num_cus(arch)))


def _ring_pad_trait(**extra) -> TraitSpec:
    return TraitSpec(
        pipeline="compv4",
        scheduler="intrawave",
        epilogue="default",
        pad_m=True,
        pad_n=True,
        direct_to_lds=True,
        dtl_prefetch=True,
        lds_swizzle=True,
        dtl_cache_a=0,
        dtl_cache_b=0,
        chiplet_swizzle=True,
        chiplet_wgm=8,
        chiplet_chunk_size=32,
        dtl_stages=2,
        interwave_setprio=True,
        dtl_hoist=True,
        dtl_tight_rsrc=True,
        pad_fast_tile=True,
        **extra,
    )


_S3ILV = dict(
    dtl_stages=3,
    dtl_frag_prefetch=True,
    mfma_interleave=True,
    ilv_merge=True,
    ilv_barrier_after=2,
)


# ---------------------------------------------------------------------------
# Non-power-of-two small tiles for the small ring's short-K regime (few output
# tiles, short K): the DTL ring bodies at 192x128 / 128x192 (8 waves of 96x32 /
# 32x96) and 256x128x32 (4 waves) replace the 128x128 / 256x128 small-ring
# geometries where SMALL_TILE_SHORTK_MIN_K <= K <= SMALL_TILE_SHORTK_MAX_K and
# at most SMALL_TILE_SHORTK_MAX_WAVES waves of its own tiles:
#   * K % 64 == 0, tiles of 192x128 fit one wave: 192x128x64 2x4 waves, the
#     3-stage fragment-prefetch interleave (merged, no ilv_barrier_after);
#   * K % 64 == 0, more than one wave: 128x192x64 4x2 waves, 2-stage ring,
#     no chiplet swizzle, when the last wave of its tiles is at least
#     SMALL_TILE_SHORTK_MIN_FILL full;
#   * K % 64 == 32: 256x128x32 4x1 waves, AGPR 3-stage interleave, no
#     interwave_setprio;
# When the tile divides M and N the pad / dtl_tight_rsrc / pad_fast_tile
# bounds are dropped, the DTLs issue with soffset 0 (``dtl_soff0``) and C is
# written by the cshuffle (LDS-staged, wide-store) epilogue. Every body is a
# combination of existing levers (no new kernel code);
# ``SMALL_TILE_SHORTK = False`` restores the 128x128 / 256x128 geometries.
# ---------------------------------------------------------------------------
SMALL_TILE_SHORTK = True
SMALL_TILE_SHORTK_MIN_K = 256
SMALL_TILE_SHORTK_MAX_K = 768
SMALL_TILE_SHORTK_MAX_WAVES = 3
# more than one wave of 128x192 tiles: only with a last-wave fill >= this
SMALL_TILE_SHORTK_MIN_FILL = 0.75


def _ring_tile(tm: int, tn: int, tk: int, wm: int, wn: int) -> TileSpec:
    return TileSpec(
        tile_m=tm,
        tile_n=tn,
        tile_k=tk,
        warp_m=wm,
        warp_n=wn,
        warp_k=1,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=32,
    )


def small_tile_shortk_tile_trait(
    arch: str, M: int, N: int, K: int, dtype: Optional[str] = None
) -> Optional[tuple[TileSpec, TraitSpec]]:
    """The short-K small-tile body for a small-ring problem, or ``None``
    (outside its calibrated K range, off, or not gfx950)."""
    if not SMALL_TILE_SHORTK or arch not in RILV_ARCHES:
        return None
    M, N, K = int(M), int(N), int(K)
    if not SMALL_TILE_SHORTK_MIN_K <= K <= SMALL_TILE_SHORTK_MAX_K:
        return None
    cus = arch_num_cus(arch)
    if K % 64 == 0:
        if _tiles(M, N, 192, 128) <= cus:
            if M % 192 or N % 128:
                if M % 128 == 0 and N % 192 == 0:
                    tile = _ring_tile(128, 192, 64, 4, 2)
                else:
                    tile = _ring_tile(192, 128, 64, 2, 4)
            else:
                tile = _ring_tile(192, 128, 64, 2, 4)
            trait = replace(_ring_pad_trait(), **dict(_S3ILV, ilv_barrier_after=0))
        else:
            if M % 128 or N % 192:
                tile = (
                    _ring_tile(192, 128, 64, 2, 4)
                    if M % 192 == 0 and N % 128 == 0
                    else _ring_tile(128, 192, 64, 4, 2)
                )
            else:
                tile = _ring_tile(128, 192, 64, 4, 2)
            trait = replace(_ring_pad_trait(), chiplet_swizzle=False)
            if (
                wave_fill(M, N, tile.tile_m, tile.tile_n, arch)
                < SMALL_TILE_SHORTK_MIN_FILL
            ):
                # a poorly filled second wave of 128x192 tiles does not pay
                # (checked on held-out shapes): keep the 128x128 small ring
                return None
    elif K % 32 == 0:
        tile = _ring_tile(256, 128, 32, 4, 1)
        trait = replace(
            _ring_pad_trait(),
            acc_agpr=True,
            interwave_setprio=False,
            **_S3ILV,
        )
    else:
        return None
    if _tiles(M, N, tile.tile_m, tile.tile_n) > SMALL_TILE_SHORTK_MAX_WAVES * cus:
        # calibrated for the few-tile class only
        return None
    if _tiles(M, N, tile.tile_m, tile.tile_n) < DTL_RING_SMALL_MIN_TILES_PER_CU * cus:
        # fewer than half a wave of its tiles: keep the 128x128 geometry (whose
        # smaller tiles still give the small ring half a wave)
        return None
    if M % tile.tile_m == 0 and N % tile.tile_n == 0:
        # exact tiles: no bounds, the DTLs issue with soffset 0 (dtl_soff0) and
        # C leaves through the cshuffle (LDS-staged, wide-store) epilogue
        trait = replace(
            trait,
            pad_m=False,
            pad_n=False,
            dtl_tight_rsrc=False,
            pad_fast_tile=False,
            dtl_soff0=True,
            epilogue="cshuffle",
        )
    return tile, trait


def dtl_ring_small_tile_trait(
    arch: str,
    K: Optional[int] = None,
    M: Optional[int] = None,
    N: Optional[int] = None,
    dtype: Optional[str] = None,
) -> tuple[TileSpec, TraitSpec]:
    """The small-tile ring: 128x128x64, 2x4 waves of 64x32 (16x16x32 atoms),
    2-stage ring -- as the 3-stage fragment-prefetch interleave body when its
    tiles fit one wave and K >= :data:`DTL_RING_SMALL_S3_MIN_K`. For
    K % 64 != 0: a 128x128x32 2x2-wave interleave ring on up to
    :data:`DTL_RING_SMALL_K32_C128_MAX_WAVES` waves of 256x128 tiles, else a
    256x128x32 4-wave body (2x2 waves of 128x64) with AGPR accumulators, a
    3-stage ring and the merged MFMA interleave."""
    if arch not in DTL_RING_SMALL_ARCHES:
        raise ValueError(f"no small-tile DTL ring geometry for {arch!r}")
    if K is not None and M is not None and N is not None:
        r3 = small_tile_shortk_tile_trait(arch, M, N, K, dtype)
        if r3 is not None:
            return r3
    if K is not None and int(K) % 64:
        poor = (
            M is not None
            and N is not None
            and _tiles(M, N, 256, 128)
            <= DTL_RING_SMALL_K32_C128_MAX_WAVES * arch_num_cus(arch)
        )
        if poor:
            tile = TileSpec(
                tile_m=128,
                tile_n=128,
                tile_k=32,
                warp_m=2,
                warp_n=2,
                warp_k=1,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=32,
            )
            return tile, replace(_ring_pad_trait(), **_S3ILV)
        tile = TileSpec(
            tile_m=256,
            tile_n=128,
            tile_k=32,
            warp_m=2,
            warp_n=2,
            warp_k=1,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=32,
        )
        return tile, replace(_ring_pad_trait(), acc_agpr=True, **_S3ILV)
    tile = TileSpec(
        tile_m=128,
        tile_n=128,
        tile_k=64,
        warp_m=2,
        warp_n=4,
        warp_k=1,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=32,
    )
    if (
        DTL_RING_SMALL_S3_SINGLE_WAVE
        and K is not None
        and M is not None
        and N is not None
        and int(K) >= DTL_RING_SMALL_S3_MIN_K
        and _tiles(M, N, 128, 128) <= arch_num_cus(arch)
    ):
        return tile, replace(_ring_pad_trait(), **_S3ILV)
    return tile, _ring_pad_trait()


def dtl_ring_small_sweep_variants(
    spec: UniversalGemmSpec, arch: str
) -> tuple[UniversalGemmSpec, ...]:
    """The small ring's sweep neighbourhood (selected spec first): the 2-stage
    and the 3-stage interleave form of the 128x128x64 body, and for tile_k 32
    the other K % 64 == 32 body; every variant validity-checked."""
    from ...instances.common.gemm_universal import is_valid_spec

    t, tr = spec.tile, spec.trait
    base = replace(
        tr,
        dtl_stages=2,
        dtl_frag_prefetch=False,
        mfma_interleave=False,
        ilv_merge=False,
        ilv_barrier_after=0,
        acc_agpr=False,
    )
    cands = []
    if t.tile_k == 64:
        cands = [
            replace(spec, trait=base),
            replace(spec, trait=replace(base, **_S3ILV)),
        ]
    else:
        c128 = replace(t, tile_m=128, tile_n=128, warp_m=2, warp_n=2)
        c256 = replace(t, tile_m=256, tile_n=128, warp_m=2, warp_n=2)
        cands = [
            replace(spec, tile=c128, trait=replace(base, **_S3ILV)),
            replace(spec, tile=c256, trait=replace(base, acc_agpr=True, **_S3ILV)),
        ]
    out = [spec]
    for c in cands:
        if c != spec and c not in out and is_valid_spec(c, arch=arch)[0]:
            out.append(c)
    return tuple(out)


def dtl_ring_pad_tile_trait(
    arch: str, big_grid: bool = False
) -> tuple[TileSpec, TraitSpec]:
    """The pad ring: the 16-wave 256x256x64 ring, persistent on every CU; on a
    very large grid the AGPR interleave big-grid body
    (:func:`agpr_ilv_big_grid_tile_trait`) with the same pad + exact DTL bounds
    + full-tile fast path."""
    if arch not in DTL_RING_PAD_ARCHES:
        raise ValueError(f"no padded DTL ring geometry for {arch!r}")
    if big_grid:
        return agpr_ilv_big_grid_tile_trait(
            arch, pad_m=True, pad_n=True, dtl_tight_rsrc=True, pad_fast_tile=True
        )
    tile, _ = dtl_pingpong_tile_trait(arch)
    return tile, _ring_pad_trait(persistent_ctas=arch_num_cus(arch))


def _default_tile(req: GemmRequest) -> Optional[TileCostModel]:
    """The default (cshuffle) candidate's tile as calibrated for the dispatched
    ring body of ``req``'s (arch, dtype) (``DTL_PINGPONG_COST_MODELS``)."""
    models = DTL_PINGPONG_COST_MODELS.get((req.arch, normalize_dtype(req.dtype)))
    return None if models is None else models[1]


def _ring_split_k_reason(req: GemmRequest) -> Optional[str]:
    """Why a ring candidate must not take ``req`` because the default candidate
    engages split-K there (the decode regime no ring was calibrated in), or
    ``None``. Calls :func:`select_split_k` with the default candidate's tile, as
    :func:`dtl_pingpong_admit` does, instead of relying on a tile-count floor
    that happens to equal its gate."""
    d = _default_tile(req)
    if d is None:
        return f"no default-candidate calibration for {req.arch} {req.dtype}"
    sk = select_split_k(
        M=req.M, N=req.N, K=req.K, tile_m=d.tile_m, tile_n=d.tile_n, tile_k=d.tile_k
    )
    if sk.split_k > 1:
        return (
            f"the default candidate engages split-K {sk.split_k} for "
            f"{req.M}x{req.N}x{req.K} (decode regime; the ring is not calibrated there)"
        )
    return None


def _aligned_to_default(req: GemmRequest) -> bool:
    d = _default_tile(req)
    return (
        d is not None
        and int(req.M) % d.tile_m == 0
        and int(req.N) % d.tile_n == 0
        and int(req.K) % d.tile_k == 0
    )


def _ragged_small_k(req: GemmRequest) -> bool:
    """Ragged M / N at small K on a moderate grid: the small ring's regime even
    with a well-filled 256x256 wave."""
    return (
        not (int(req.M) % 256 == 0 and int(req.N) % 256 == 0)
        and int(req.K) <= DTL_RING_SMALL_RAGGED_MAX_K
        and _tiles(req.M, req.N, 256, 256) < DTL_RING_SMALL_RAGGED_MAX_TILES
    )


def dtl_ring_small_admit(req: GemmRequest, spec: UniversalGemmSpec) -> tuple[bool, str]:
    """Small-tile ring: the 256x256 tiles fill the last wave poorly (or a
    ragged small-K problem, :func:`_ragged_small_k`), its own tiles still give
    at least half a wave, the default candidate does not split K, and -- where
    the aligned default candidate can take the problem -- K is above the
    calibrated floor of the selected body."""
    if rilv_shortk_admits(req):
        return False, "the aligned short-K row-interleave body serves it"
    why = _ring_split_k_reason(req)
    if why is not None:
        return False, why
    fill = wave_fill(req.M, req.N, 256, 256, req.arch)
    ragged_small_k = _ragged_small_k(req)
    if fill >= DTL_RING_SMALL_MAX_FILL and not ragged_small_k:
        return False, (
            f"256x256 tiles fill the last wave to {fill:.2f} >= "
            f"{DTL_RING_SMALL_MAX_FILL} for {req.M}x{req.N} on {req.arch}"
        )
    t = spec.tile
    tiles = _tiles(req.M, req.N, t.tile_m, t.tile_n)
    need = DTL_RING_SMALL_MIN_TILES_PER_CU * arch_num_cus(req.arch)
    if tiles < need:
        return False, (
            f"{tiles} {t.tile_m}x{t.tile_n} tiles < {need:.0f} (half a wave): "
            "latency-bound, not calibrated"
        )
    if _aligned_to_default(req):
        if t.tile_k == 64:
            rule = DTL_RING_ADMIT_RULES.get((req.arch, normalize_dtype(req.dtype)))
            k_min = rule.k_lo if rule is not None else DTL_RING_SMALL_RAGGED_MAX_K
        elif spec.trait.acc_agpr:
            k_min = DTL_RING_SMALL_K32_AGPR_MIN_K
        else:
            k_min = DTL_RING_SMALL_K32_MIN_K
        if int(req.K) < k_min:
            return False, (
                f"K={req.K} < {k_min}: the {t.tile_m}x{t.tile_n}x{t.tile_k} small "
                f"ring loses to the default candidate there ({req.M}x{req.N} is "
                "aligned to it)"
            )
    return True, (
        f"ok (256x256 last-wave fill {fill:.2f}"
        + (", ragged small K)" if ragged_small_k else ")")
    )


def dtl_ring_pad_admit(req: GemmRequest, spec: UniversalGemmSpec) -> tuple[bool, str]:
    """Pad ring: ragged M / N (the aligned shapes belong to the ping-pong / ring
    candidate) with a well-filled last wave (else the small-tile ring), outside
    the ragged small-K regime of the small ring, and no split-K default."""
    t = spec.tile
    if req.M % t.tile_m == 0 and req.N % t.tile_n == 0:
        return False, (
            f"{req.M}x{req.N} is a multiple of {t.tile_m}x{t.tile_n}: the "
            "aligned DTL candidate covers it"
        )
    why = _ring_split_k_reason(req)
    if why is not None:
        return False, why
    fill = wave_fill(req.M, req.N, t.tile_m, t.tile_n, req.arch)
    if fill < DTL_RING_SMALL_MAX_FILL:
        return False, (
            f"{t.tile_m}x{t.tile_n} tiles fill the last wave to {fill:.2f} < "
            f"{DTL_RING_SMALL_MAX_FILL} (the small-tile ring's regime)"
        )
    if _ragged_small_k(req):
        return False, (
            f"K={req.K} <= {DTL_RING_SMALL_RAGGED_MAX_K} on fewer than "
            f"{DTL_RING_SMALL_RAGGED_MAX_TILES} 256x256 tiles: the small-tile "
            "ring's ragged small-K regime"
        )
    return True, "ok"


def fused_epilogue_spec(spec: UniversalGemmSpec) -> Optional[UniversalGemmSpec]:
    """The body to attach a ``helpers.fuse`` epilogue to, for a dispatched
    ``spec``: ``spec`` itself when its epilogue has the fused hook, else the same
    tile / main loop without the levers whose epilogue has none (the persistent
    tile loop becomes one workgroup per tile; ``mfma_swap_ab`` stores the
    accumulator directly). ``None`` when the body writes f32 partial sums
    (split-K / stream-K): re-dispatch without split-K there.

    The returned spec carries no epilogue; attach it, then check
    ``is_valid_spec`` (the caller's epilogue decides the rest)."""
    tr = spec.trait
    if tr.split_k > 1 or tr.streamk:
        return None
    if not (tr.persistent_ctas or tr.mfma_swap_ab):
        return spec
    return replace(spec, trait=replace(tr, persistent_ctas=0, mfma_swap_ab=False))


def persistent_grid(spec: UniversalGemmSpec, M: int, N: int) -> tuple[int, int, int]:
    """Launch grid of a ``persistent_ctas = P`` body: ``min(P, tiles)``
    workgroups on x. The kernel's flat workgroup id is ``y * tiles_n + x``, so
    x alone covers ids ``0 .. P-1``; workgroups ``>= P`` would only exit, so
    the full 2D grid is correct but launches (and reserves LDS for) thousands of
    empty workgroups."""
    t = spec.tile
    if spec.trait.streamk:
        # stream-K: the in-kernel partition spreads K tiles over all P
        # workgroups, so exactly P must run (and be co-resident)
        return (int(spec.trait.persistent_ctas), 1, 1)
    tiles = -(-int(M) // t.tile_m) * -(-int(N) // t.tile_n)
    return (max(1, min(int(spec.trait.persistent_ctas), tiles)), 1, 1)


def fills_device(req: GemmRequest, spec: UniversalGemmSpec) -> tuple[bool, str]:
    """Big-tile admission by device fill (kept as a selectable policy).

    Admits the ping-pong candidate only when ``ceil(M/tile_m) * ceil(N/tile_n)
    >= num_CUs``. On gfx950 this is both too strict (the candidate already pays
    well below one full wave) and too loose (it admits tiny-K shapes the
    candidate loses), so the default policy is :func:`dtl_pingpong_admit`.
    """
    t = spec.tile
    tiles = -(-int(req.M) // t.tile_m) * -(-int(req.N) // t.tile_n)
    cus = arch_num_cus(req.arch)
    if tiles < cus:
        return False, (
            f"{tiles} output tiles of {t.tile_m}x{t.tile_n} < {cus} CUs on "
            f"{req.arch} (big-tile candidate needs a full device wave)"
        )
    return True, "ok"


# ---------------------------------------------------------------------------
# Ping-pong candidate admission: per-CU wave-efficiency cost model.
#
# The ping-pong kernel (256x256 tile, 1 workgroup per CU) competes with the
# 128x128 cshuffle default (4x the tiles, several co-resident workgroups per
# CU). Which one is faster depends on how many tiles each CU gets, how well
# co-resident tiles overlap, how full the device is, and how much fixed
# per-tile work (prologue, epilogue) there is relative to K. For a candidate c:
#
#     T_c = ceil(M / tile_m) * ceil(N / tile_n)   output tiles
#     n_c = ceil(T_c / CUs)                         tiles on the busiest CU
#     u_c = min(1, T_c / CUs)                       fraction of CUs with a tile
#     t_c = a_c * n_c + b_c * K * n_c**theta_c * u_c**gamma_c
#
# ``a`` is the per-tile fixed cost, ``b`` the per-K cost of one tile alone on
# a CU, ``theta`` <= 1 how much co-resident tiles on one CU overlap and
# ``gamma`` the speed-up of a partially filled device (less shared-bandwidth
# contention). The coefficients are dimensionless relative costs (normalised so
# the default kernel's ``b`` is 1), least-squares fitted per (arch, dtype) to
# on-device timings (gfx950: squares including non-multiples of the CU count,
# a wide K range and skinny M / N; gfx942: fewer shapes over the same ranges;
# every point checked against an f64 full-output reference). The CU count is
# the static per-arch value (``arch_num_cus``), so gfx942 parts with fewer CUs
# use the full-device model. The candidate is admitted when
# ``t_F <= DTL_PINGPONG_COST_MARGIN * t_default``; the margin was chosen by
# leave-one-out cross-validation over that data (it absorbs the model's error
# in the small-K regime where the two kernels are close). Recalibrate when
# either kernel or the target changes. Arches without coefficients never admit
# the candidate.
#
# Split-K: the ping-pong candidate is selected WITHOUT split-K (its select skips
# ``apply_split_k``), and it is not admitted where the default candidate would
# engage split-K (the decode / GEMV regime the model was not calibrated in), so
# both sides of the comparison are the single-K-pass kernels that were calibrated.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TileCostModel:
    """``t = a * n + b * K * n**theta * u**gamma`` (see the block comment)."""

    tile_m: int
    tile_n: int
    tile_k: int
    a: float
    b: float
    theta: float
    gamma: float

    def cost(self, M: int, N: int, K: int, num_cus: int) -> float:
        tiles = -(-int(M) // self.tile_m) * -(-int(N) // self.tile_n)
        n = -(-tiles // num_cus)
        u = min(1.0, tiles / num_cus)
        return self.a * n + self.b * int(K) * n**self.theta * u**self.gamma


# (arch, dtype, body) -> (256x256 candidate model, default-candidate model).
# The 256x256 candidate's body decides the coefficients (``dtl_pingpong_body``):
# ``"pingpong"`` is the DTL ping-pong body (``dtl_stages == 0``), ``"ring"``
# the alias-scoped DTL ring (``dtl_stages >= 2``, the gfx950 dispatch pick via
# ``DTL_PINGPONG_RING``). Both gfx950 bodies stay calibrated, so turning the
# ring off (``DTL_PINGPONG_RING`` without gfx950, or ``ring=False``) judges the
# ping-pong body with its own coefficients. gfx942 was calibrated on fewer
# shapes and without a held-out set; the gfx950 ping-pong rows were checked on
# held-out shapes. On gfx950 the ring's cost model is the fallback
# policy -- the threshold rule below predicts the faster kernel better there
# (the ring wins at any tile count once K is large, which the model's per-tile
# fixed cost does not capture).
DTL_PINGPONG_COST_MODELS_BY_BODY = {
    ("gfx942", "fp16", "pingpong"): (
        TileCostModel(256, 256, 32, a=281.1, b=2.2663, theta=0.95, gamma=0.30),
        TileCostModel(128, 128, 16, a=56.1, b=1.0, theta=1.00, gamma=0.0),
    ),
    ("gfx942", "bf16", "pingpong"): (
        TileCostModel(256, 256, 32, a=248.6, b=1.7940, theta=0.90, gamma=0.30),
        TileCostModel(128, 128, 16, a=47.6, b=1.0, theta=0.95, gamma=0.0),
    ),
    ("gfx950", "fp16", "pingpong"): (
        TileCostModel(256, 256, 64, a=422.5, b=1.6093, theta=0.90, gamma=0.20),
        TileCostModel(128, 128, 32, a=88.4, b=1.0, theta=0.80, gamma=0.0),
    ),
    ("gfx950", "bf16", "pingpong"): (
        TileCostModel(256, 256, 64, a=413.2, b=1.5272, theta=0.90, gamma=0.20),
        TileCostModel(128, 128, 32, a=102.3, b=1.0, theta=0.85, gamma=0.0),
    ),
    ("gfx950", "fp16", "ring"): (
        TileCostModel(256, 256, 64, a=399.6, b=1.1499, theta=0.95, gamma=0.15),
        TileCostModel(128, 128, 32, a=83.8, b=1.0, theta=0.75, gamma=0.0),
    ),
    ("gfx950", "bf16", "ring"): (
        TileCostModel(256, 256, 64, a=375.6, b=1.0786, theta=0.95, gamma=0.15),
        TileCostModel(128, 128, 32, a=91.3, b=1.0, theta=0.80, gamma=0.0),
    ),
}


def dtl_pingpong_body(spec: UniversalGemmSpec) -> str:
    """``"ring"`` for the alias-scoped DTL ring body (``dtl_stages >= 2``),
    ``"pingpong"`` for the DTL ping-pong body."""
    return "ring" if spec.trait.dtl_stages >= 2 else "pingpong"


def _dispatched_body(arch: str) -> str:
    return (
        "ring"
        if DTL_PINGPONG_RING.get(arch, {}).get("dtl_stages", 0) >= 2
        else "pingpong"
    )


# (arch, dtype) -> the models of the body each arch DISPATCHES (a read-only
# view of ``DTL_PINGPONG_COST_MODELS_BY_BODY``; kept for callers that ask about
# the dispatched candidate without a spec).
DTL_PINGPONG_COST_MODELS = {
    (a, d): m
    for (a, d, body), m in DTL_PINGPONG_COST_MODELS_BY_BODY.items()
    if body == _dispatched_body(a)
}

# Per-(arch, body) cost-model margin (chosen by leave-one-out over each fit's
# data): admit when ``t_F <= margin * t_default``.
DTL_PINGPONG_COST_MARGINS_BY_BODY = {
    ("gfx942", "pingpong"): 0.95,
    ("gfx950", "pingpong"): 0.95,
    ("gfx950", "ring"): 1.0,
}
# arch -> margin of the dispatched body (read-only view).
DTL_PINGPONG_COST_MARGINS = {
    a: m
    for (a, body), m in DTL_PINGPONG_COST_MARGINS_BY_BODY.items()
    if body == _dispatched_body(a)
}


@dataclass(frozen=True)
class RingAdmitRule:
    """Threshold admission of the 256x256 candidate: admit iff ``K >= k_hi``
    or (``K >= k_lo`` and ``ceil(M/256) * ceil(N/256) >= t_min``)."""

    k_lo: int
    k_hi: int
    t_min: int

    def admits(self, M: int, N: int, K: int, tile_m: int, tile_n: int) -> bool:
        tiles = -(-int(M) // tile_m) * -(-int(N) // tile_n)
        return int(K) >= self.k_hi or (int(K) >= self.k_lo and tiles >= self.t_min)


# (arch, dtype) -> threshold rule for the RING body (it is applied only when
# the candidate is the ring, ``dtl_pingpong_body == "ring"``; the ping-pong
# body always uses its cost model). gfx950: grid-searched on squares, a wide K
# range and skinny / few-tile shapes down to one output tile, every point
# checked against an f64 full-output reference; under leave-one-out it
# predicted the faster kernel better than the refitted cost model. ``t_min``
# is an absolute 256x256 tile count calibrated on a full gfx950 device; like
# every CU count in this dispatcher (``arch_num_cus`` is static), it is not
# recalibrated for partitioned modes (CPX / DPX) or other CU counts.
DTL_RING_ADMIT_RULES = {
    ("gfx950", "fp16"): RingAdmitRule(k_lo=128, k_hi=1024, t_min=100),
    ("gfx950", "bf16"): RingAdmitRule(k_lo=64, k_hi=768, t_min=144),
}

# Admission policies for the ping-pong candidate: "auto" (default: the threshold
# rule where one is calibrated for the (arch, dtype), else the cost model),
# "rule", "cost_model", "fills_device" (the previous one-wave rule), "always",
# "never". Selected by the ROCKE_GEMM_DTL_ADMIT environment variable (like
# ROCKE_GEMM_SPLIT_K, an explicit operator override; it never changes kernel
# validity).
DTL_PINGPONG_COST_MARGIN = 0.95  # default for an arch absent from the per-arch map

DTL_ADMIT_ENV = "ROCKE_GEMM_DTL_ADMIT"
DTL_ADMIT_POLICIES = ("auto", "rule", "cost_model", "fills_device", "always", "never")


def dtl_admit_policy() -> str:
    v = os.environ.get(DTL_ADMIT_ENV, "").strip().lower() or "auto"
    if v not in DTL_ADMIT_POLICIES:
        raise ValueError(
            f"{DTL_ADMIT_ENV}={v!r}; expected one of {', '.join(DTL_ADMIT_POLICIES)}"
        )
    return v


def dtl_pingpong_admit(req: GemmRequest, spec: UniversalGemmSpec) -> tuple[bool, str]:
    """Ping-pong candidate admission: the calibrated threshold rule (``auto`` /
    ``rule``) or the cost model says it beats the default candidate."""
    policy = dtl_admit_policy()
    if policy == "always":
        return True, "ok (policy always)"
    if policy == "auto" and rilv_shortk_admits(req):
        return True, "ok (row-interleave short-K body)"
    if policy == "never":
        return False, f"ping-pong admission disabled ({DTL_ADMIT_ENV}=never)"
    if policy == "fills_device":
        return fills_device(req, spec)
    key = (req.arch, normalize_dtype(req.dtype))
    rule = DTL_RING_ADMIT_RULES.get(key)
    if policy == "rule" and rule is None:
        return False, f"no ping-pong threshold rule for {req.arch} {req.dtype}"
    body = dtl_pingpong_body(spec)
    if policy == "rule" and body != "ring":
        return False, f"the threshold rule is calibrated for the ring body, not {body}"
    use_rule = rule is not None and body == "ring" and policy in ("auto", "rule")
    # the calibration (also of the rule: its split-K regime check uses the
    # default candidate's tile) exists only where a cost model does
    models = DTL_PINGPONG_COST_MODELS_BY_BODY.get(key + (body,))
    if models is None:
        return False, f"no {body} cost model for {req.arch} {req.dtype}"
    f_model, d_model = models
    t = spec.tile
    if (t.tile_m, t.tile_n) != (f_model.tile_m, f_model.tile_n):
        return False, (
            f"ping-pong cost model is calibrated for {f_model.tile_m}x{f_model.tile_n}, "
            f"spec tile is {t.tile_m}x{t.tile_n}"
        )
    sk = select_split_k(
        M=req.M,
        N=req.N,
        K=req.K,
        tile_m=d_model.tile_m,
        tile_n=d_model.tile_n,
        tile_k=d_model.tile_k,
    )
    if sk.split_k > 1:
        return False, (
            f"the default candidate engages split-K {sk.split_k} for "
            f"{req.M}x{req.N}x{req.K} (decode regime; ping-pong not calibrated there)"
        )
    if use_rule:
        if not rule.admits(req.M, req.N, req.K, t.tile_m, t.tile_n):
            return False, (
                f"threshold rule: K={req.K} < {rule.k_hi} and not (K >= {rule.k_lo} and "
                f">= {rule.t_min} output tiles) for {req.M}x{req.N}x{req.K} on {req.arch}"
            )
        return True, "ok (threshold rule)"
    cus = arch_num_cus(req.arch)
    margin = DTL_PINGPONG_COST_MARGINS_BY_BODY.get(
        (req.arch, body), DTL_PINGPONG_COST_MARGIN
    )
    t_f = f_model.cost(req.M, req.N, req.K, cus)
    t_d = d_model.cost(req.M, req.N, req.K, cus)
    if t_f > margin * t_d:
        return False, (
            f"cost model: ping-pong {t_f:.0f} > {margin} x default "
            f"{t_d:.0f} (relative units) for {req.M}x{req.N}x{req.K} on {req.arch}"
        )
    return True, "ok"


# ---------------------------------------------------------------------------
# coverage and correctness (every RCR shape gets a correct
# kernel). Knobs (module attributes; False restores the previous dispatch):
#   * G1_ADAPT: every candidate's spec is adapted to the request's ragged
#     dimensions (``g1_adapt_spec``): global-load pad bodies get
#     ``pad_row_clamp`` (no reads past A / B) and ``pad_k_fast`` (unmasked whole
#     K tiles); DTL ring bodies take K % tile_k != 0 (K % 8 == 0) with
#     ``dtl_k_tail`` instead of being refused.
#   * G1_SPLIT_K_ALIGNED: split-K (decode regime) on pad_k bodies for any K
#     (``split_k_aligned``), at the slice-depth degree of ``select_split_k``.
#   * G1_PAD_FALLBACK: the lowest-priority padded candidate admits every shape.
#   * G1_F16_DECODE: the 16-row decode candidate for f16 as for bf16.
# ---------------------------------------------------------------------------
G1_ADAPT = True
G1_DTL_K_TAIL = True
G1_PAD_ROW_CLAMP = True
G1_PAD_K_FAST = True
G1_SPLIT_K_ALIGNED = True
G1_PAD_FALLBACK = True
G1_F16_DECODE = True
# the DTL ring bodies (small ring / pad ring) past their tuned admission
# rules, ahead of the global-load fallback (ragged / small / decode-regime
# shapes the rings refuse)
G1_RING_ANY = True
# few output tiles and a long K: the split-K padded fallback beats one ring
# pass (a rule fitted on gfx950 shapes of the region the rings used to refuse):
# below this many 128x128 tiles with K at or above this, the ring yields
G1_RING_ANY_SPLIT_RULE = (32, 4096)
G1_DTL_K_TAIL_ALIGN = 8
# 2: peeled tail tile after the unchanged K loop; 1: per-DMA-issue select (the
# ping-pong bodies, which have no peeled form)
G1_DTL_K_TAIL_MODE = 2
# a body that needs the K tail drops iglp_opt (compile cliff: iglp bodies with
# either K-tail mode exceed the comgr compile-time budget). False: keep
# iglp_opt and take K-tail mode 1.
G1_K_TAIL_DROP_IGLP = True
# the beat bodies (aligned M / N classes) also take K % 64 != 0 with the K tail
G1_BEAT_K_TAIL = True
# the K tail only where it wastes little: at least this many K tiles (a 2-tile
# K = 96 on tile_k 64 keeps the faster exact-tile pick)
G1_DTL_K_TAIL_MIN_TILES = 4
# levers that wrap K at a whole-tile span (not with a K tail)
_G1_K_WRAP_LEVERS = (
    ("k_stagger", 0),
    ("k_stagger_tiles", 0),
    ("k_stagger_rc", 0),
    ("stagger_u", 0),
    ("dtl_clamp_k", False),
)


def g1_adapt_spec(req: GemmRequest, spec: UniversalGemmSpec) -> UniversalGemmSpec:
    """``spec`` made safe and fast for ``req``'s ragged dimensions.

    Returns ``spec`` unchanged for shapes the body already tiles exactly (so
    every aligned dispatch stays byte-identical) and when ``G1_ADAPT`` is off.
    """
    if not G1_ADAPT:
        return spec
    t, tr = spec.tile, spec.trait
    M, N, K = int(req.M), int(req.N), int(req.K)
    upd = {}
    if tr.direct_to_lds:
        if (
            G1_DTL_K_TAIL
            and K % t.tile_k
            and K % G1_DTL_K_TAIL_ALIGN == 0
            and -(-K // t.tile_k) >= G1_DTL_K_TAIL_MIN_TILES
            and int(tr.dtl_stages) >= 2
            and int(tr.pingpong) != 7
            and not tr.asm_loop
            and not tr.streamk
            and int(tr.split_k) == 1
            and not spec.batched
        ):
            # (iglp_opt bodies: the peeled tile doubles the scheduled region and
            # comgr compile time runs past minutes; ping-pong: no peeled form)
            iglp = tr.iglp_opt is not None
            if iglp and G1_K_TAIL_DROP_IGLP:
                # mode 1 with iglp_opt 1 also exceeds the compile-time budget
                # (e.g. a 256x256x64 beat body at K = 392, which the selector
                # can dispatch): the K-tail form of an iglp body is the same
                # body without the scheduling hint
                upd["iglp_opt"] = None
                iglp = False
            pp_or_iglp = int(tr.pingpong) or iglp
            mode = 1 if pp_or_iglp else int(G1_DTL_K_TAIL_MODE)
            upd.update(pad_k=True, dtl_k_tail=mode)
            if spec.data.layout.upper() != "RCR":
                # an MN-contiguous operand's K-tail rows read
                # zeros through its exact-bounds resource (launch rule)
                # (dtl_soff0 is the untight form of the same hoisted base)
                upd.update(
                    dtl_tight_rsrc=True, dtl_tight_ragged_only=False, dtl_soff0=False
                )
            for f, off in _G1_K_WRAP_LEVERS:
                if hasattr(tr, f) and getattr(tr, f) != off:
                    upd[f] = off
    elif not int(getattr(tr, "vgpr_prefetch", 0)):
        # (the register-staged bodies bound their rows with the buffer
        # range and mask the K tail themselves)
        rag = (tr.pad_m and M % t.tile_m) or (tr.pad_n and N % t.tile_n)
        if G1_PAD_ROW_CLAMP and rag and not tr.preshuffle_b:
            upd["pad_row_clamp"] = True
        if G1_PAD_K_FAST and tr.pad_k and K % t.tile_k:
            upd["pad_k_fast"] = True
    if not upd:
        return spec
    return replace(spec, trait=replace(tr, **upd))


def g1_aligned_split_k(req: GemmRequest, spec: UniversalGemmSpec) -> UniversalGemmSpec:
    """Split-K for a decode-regime pad_k body whose K has no whole-tile factor:
    the slice-depth degree with whole-tile slices (``split_k_aligned``)."""
    tr = spec.trait
    t = spec.tile
    if not (G1_SPLIT_K_ALIGNED and tr.pad_k) or tr.direct_to_lds or tr.split_k > 1:
        return spec
    from ...helpers.split_k import TARGET_SLICE_K, _parse_env_override

    M, N, K = int(req.M), int(req.N), int(req.K)
    mode, forced = _parse_env_override(os.environ)
    if mode == "off":
        return spec
    base = -(-M // t.tile_m) * -(-N // t.tile_n)
    if mode != "force" and base >= arch_num_cus(req.arch) // 2:
        return spec
    want = forced if mode == "force" else round(K / TARGET_SLICE_K)
    sk = max(1, min(int(want), -(-K // t.tile_k)))
    if sk <= 1:
        return spec
    upd = dict(split_k=sk, split_k_aligned=True)
    if K % t.tile_k and G1_PAD_K_FAST:
        upd["pad_k_fast"] = True
    return replace(spec, trait=replace(tr, **upd))


def g1_decode_tile_trait(arch: str, dtype: str) -> tuple[TileSpec, TraitSpec]:
    """The 16-row decode body (bf16; f16 with G1_F16_DECODE):
    16 x 64 tile, 1 x 4 waves of 16x16 atoms, padded M / N / K."""
    deep = arch == "gfx950"
    wk = 32 if deep else 16
    return (
        TileSpec(
            tile_m=16,
            tile_n=64,
            tile_k=32,
            warp_m=1,
            warp_n=4,
            warp_k=1,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=wk,
        ),
        TraitSpec(
            pipeline="compv4",
            scheduler="intrawave",
            epilogue="default",
            pad_m=True,
            pad_n=True,
            pad_k=True,
        ),
    )


G1_PAD_FALLBACK_ARCHES = ("gfx942", "gfx950")
# skinny side at or below this many rows takes a 16-row tile (decode-like)
G1_PAD_SKINNY = 32


def g1_pad_tile_trait(req: GemmRequest) -> tuple[TileSpec, TraitSpec]:
    """Fallback body for any RCR shape: the global-load compv4 body padded only
    on the ragged dimensions (rows clamped, whole K tiles unmasked).

    * min(M, N) <= G1_PAD_SKINNY: a 16 x 64 (M skinny) / 64 x 16 (N skinny)
      tile of 16x16 atoms;
    * else the 128 x 128 x 32 tile of the cshuffle default (32x32 atoms).
    """
    M, N, K = int(req.M), int(req.N), int(req.K)
    deep = req.arch == "gfx950"
    if M <= G1_PAD_SKINNY or N <= G1_PAD_SKINNY:
        wk = 32 if deep else 16
        if M <= N:
            tile = TileSpec(16, 64, 32, 1, 4, 1, 16, 16, wk)
        else:
            tile = TileSpec(64, 16, 32, 4, 1, 1, 16, 16, wk)
    else:
        tile = (
            TileSpec(128, 128, 32, 2, 2, 1, 32, 32, 16)
            if deep
            else TileSpec(128, 128, 16, 2, 2, 1, 32, 32, 8)
        )
    trait = TraitSpec(
        pipeline="compv4",
        scheduler="intrawave",
        epilogue="default",
        pad_m=M % tile.tile_m != 0,
        pad_n=N % tile.tile_n != 0,
        pad_k=K % tile.tile_k != 0,
    )
    return tile, trait


def g1_pad_fallback_admit(
    req: GemmRequest, spec: UniversalGemmSpec
) -> tuple[bool, str]:
    if not G1_PAD_FALLBACK:
        return False, "G1_PAD_FALLBACK is off"
    return True, "ok (padded fallback)"


def g1_gemm_signature(spec: Optional[UniversalGemmSpec], dtype: str) -> list:
    """The RCR GEMM args signature of ``spec``: split-K bodies take the f32
    workspace ``Cf32`` (atomic-add target) in C's slot."""
    from ...helpers.manifest import gemm_args_signature

    sig = gemm_args_signature(dtype=dtype)
    if (
        spec is not None
        and int(spec.trait.split_k) > 1
        and int(getattr(spec.trait, "split_k_serial", 0))
    ):
        # split_k_serial: C, then the f32 partial slots W and
        # the i32 counters Cnt (split_k_serial_workspace)
        out = []
        for a in sig:
            out.append(a)
            if a["name"] == "C":
                out.append({"name": "W", "type": "ptr<f32, global>", "size_bytes": 8})
                out.append({"name": "Cnt", "type": "ptr<i32, global>", "size_bytes": 8})
        return out
    if spec is not None and int(spec.trait.split_k) > 1:
        sig = [
            (
                {"name": "Cf32", "type": "ptr<f32, global>", "size_bytes": 8}
                if a["name"] == "C"
                else a
            )
            for a in sig
        ]
    return sig


def g1_sweep_filter(req: GemmRequest, specs) -> tuple:
    """The sweep variants of a candidate adapted to ``req``'s ragged
    dimensions (``g1_adapt_spec``), keeping only those the body can launch: a
    variant generator that rebuilds the trait would otherwise hand back a
    spec without the K tail / row clamp its admitted base needed."""
    from ...instances.common.gemm_universal import is_valid_spec, launch_shape_error

    out = []
    for s in specs:
        s = g1_adapt_spec(req, s)
        if launch_shape_error(s, int(req.M), int(req.N), int(req.K)) is not None:
            continue
        if not is_valid_spec(s, arch=req.arch)[0]:
            continue
        out.append(s)
    return tuple(out)


def g1_ring_any_admit(req: GemmRequest, spec: UniversalGemmSpec) -> tuple[bool, str]:
    """The ring bodies for any shape they can launch: the
    validity and launch rules are the candidate's support check; this only
    gates the knob and the K alignment the DMA needs."""
    if not G1_RING_ANY:
        return False, "G1_RING_ANY is off"
    if int(req.K) % G1_DTL_K_TAIL_ALIGN:
        return False, f"K={req.K} is not a multiple of {G1_DTL_K_TAIL_ALIGN} (DMA rows)"
    if G1_RING_ANY_SPLIT_RULE:
        tmax, kmin = G1_RING_ANY_SPLIT_RULE
        t128 = _tiles(req.M, req.N, 128, 128)
        if t128 < tmax and int(req.K) >= kmin:
            return False, (
                f"{t128} 128x128 tiles < {tmax} with K={req.K} >= {kmin}: the split-K "
                "fallback is faster than one ring pass"
            )
    return True, "ok (ring, any shape)"


def g1_ring_any_uses_pad_ring(req: GemmRequest) -> bool:
    """Pad ring (256x256) when its tiles fill the device with a full last
    wave, else the small ring."""
    return wave_fill(
        req.M, req.N, 256, 256, req.arch
    ) >= DTL_RING_SMALL_MAX_FILL and _tiles(req.M, req.N, 256, 256) >= arch_num_cus(
        req.arch
    )


# ``dtl_ring_any`` output tile: the wave-quantised cost ceil(tiles / CUs) *
# tile area / relative per-FLOP efficiency (fitted on gfx950 ring-any tuning
# data; see the table below); () keeps
# the small ring's own tile.
G1_RING_ANY_TILES = ((128, 128, 1.0), (128, 64, 0.6), (64, 128, 0.6), (64, 64, 0.4))


def g1_ring_any_retile(req: GemmRequest, spec: UniversalGemmSpec) -> UniversalGemmSpec:
    """The ring-any spec on the tile of least wave-quantised cost (a valid
    warp layout kept or re-chosen); ``spec`` itself when no option is valid
    or the knob is off."""
    from ...instances.common.gemm_universal import is_valid_spec

    if not G1_RING_ANY_TILES:
        return spec
    M, N = int(req.M), int(req.N)
    cus = arch_num_cus(req.arch)
    t = spec.tile
    best = None
    for tm, tn, eff in G1_RING_ANY_TILES:
        tiles = -(-M // tm) * -(-N // tn)
        cost = -(-tiles // cus) * tm * tn / eff
        if best is not None and cost >= best[0]:
            continue
        cand = None
        for wm, wn in ((t.warp_m, t.warp_n), (2, 4), (2, 2), (4, 2), (1, 4), (4, 1)):
            if (tm, tn, wm, wn) == (t.tile_m, t.tile_n, t.warp_m, t.warp_n):
                cand = spec
                break
            try:
                s2 = replace(
                    spec,
                    tile=replace(t, tile_m=tm, tile_n=tn, warp_m=wm, warp_n=wn),
                    block_size=0,
                )
            except Exception:  # noqa: BLE001 - an impossible warp layout
                continue
            if is_valid_spec(s2, arch=req.arch)[0]:
                cand = s2
                break
        if cand is not None:
            best = (cost, cand)
    return spec if best is None else best[1]


# ---------------------------------------------------------------------------
# gfx942: register-staged prefetch bodies.
#
# gfx942's direct-to-LDS moves one dword per lane (a quarter of gfx950's), so
# every DTL body there issues four times the load instructions per byte. The
# ``vgpr_prefetch`` bodies load with ``buffer_load_dwordx4`` into VGPRs one or
# more K tiles ahead and write LDS from the registers (hipBLASLt's gfx942
# path). The candidate picks one geometry per request with a wave-
# quantization cost: ceil(tiles / CUs) * tile area / relative efficiency
# (fitted per geometry on gfx942 with a full-output f64 check of every timed
# kernel).
# ---------------------------------------------------------------------------

G5_VGPR = True
G5_VGPR_ARCHES = ("gfx942",)
# registry priority of the candidate. 2 puts it ahead of
# gemm_skinny (3; its region table is gfx950-fitted) on gfx942, where both
# admit min(M, N) >= G5_VGPR_MIN_MN; gemm_skinny keeps the smaller shapes.
G5_VGPR_PRIORITY = 2
# vgpr_prefetch depth of the dispatched bodies
G5_VGPR_DEPTH = 1
# name -> (tile_m, tile_n, tile_k, warp_m, warp_n, warp_tile_m, warp_tile_n,
#          warp_tile_k, acc_agpr, vgpr_prefetch mode, lds_swizzle_r,
#          relative per-tile efficiency). Mode 4 = one LDS buffer (tile_k 64 /
#          128 fit the 64 KiB LDS), mode 3 = the double buffer (tile_k 32, the
#          K % 64 == 32 fallback). The efficiencies are the fitted
#          ``eff`` of t ~ ceil(tiles / CUs) * tile_m * tile_n * (K + G5_VGPR_K0)
#          / eff over interleaved gfx942 timings (every timed kernel
#          f64-checked).
G5_VGPR_GEOS = {
    "256x256x64": (256, 256, 64, 2, 2, 16, 16, 16, True, 4, 0, (18.0, 0.0490)),
    "256x256x64w8": (256, 256, 64, 2, 4, 16, 16, 16, True, 4, 0, (13.5, 0.0415)),
    "256x224x64": (256, 224, 64, 2, 2, 16, 16, 16, True, 4, 0, (14.3, 0.0393)),
    "224x256x64": (224, 256, 64, 2, 2, 16, 16, 16, True, 4, 0, (17.3, 0.0397)),
    "256x192x64": (256, 192, 64, 2, 2, 16, 16, 16, True, 4, 0, (7.5, 0.0392)),
    "192x256x64": (192, 256, 64, 2, 2, 16, 16, 16, True, 4, 0, (10.1, 0.0392)),
    "192x256x64w8": (192, 256, 64, 2, 4, 16, 16, 16, False, 4, 0, (10.3, 0.0331)),
    "128x128x128": (128, 128, 128, 2, 2, 16, 16, 16, False, 4, 1, (4.78, 0.0151)),
    "128x128x64": (128, 128, 64, 2, 2, 16, 16, 16, False, 4, 0, (4.11, 0.0170)),
    "256x64x64": (256, 64, 64, 4, 1, 16, 16, 16, False, 4, 0, (1.18, 0.0168)),
    "64x256x64": (64, 256, 64, 1, 4, 16, 16, 16, False, 4, 0, (0.09, 0.0179)),
    "128x64x128": (128, 64, 128, 2, 2, 16, 16, 16, False, 4, 0, (3.50, 0.00904)),
    "64x128x128": (64, 128, 128, 2, 2, 16, 16, 16, False, 4, 0, (1.92, 0.00971)),
    "64x64x128": (64, 64, 128, 2, 2, 16, 16, 16, False, 4, 0, (2.42, 0.00587)),
    "128x32x128": (128, 32, 128, 4, 1, 16, 16, 16, False, 4, 0, (2.57, 0.00682)),
    "32x128x128": (32, 128, 128, 1, 4, 16, 16, 16, False, 4, 0, (2.24, 0.00681)),
    "256x256x32": (256, 256, 32, 2, 2, 16, 16, 16, True, 3, 1, (16.6, 0.0565)),
    "128x128x32": (128, 128, 32, 2, 4, 16, 16, 16, False, 3, 1, (0.95, 0.0227)),
    # 256x128 / 128x256 (4 waves, mode 4; ~320 VGPRs, no
    # spills); planned only with R2_EXTRA_GEOS_ON (calibration pending)
    "256x128x64": (256, 128, 64, 2, 2, 16, 16, 16, False, 4, 0, (9.0, 0.0245)),
    "128x256x64": (128, 256, 64, 2, 2, 16, 16, 16, False, 4, 0, (9.0, 0.0245)),
    "256x128x64w41": (256, 128, 64, 4, 1, 16, 16, 16, False, 4, 0, (9.0, 0.0245)),
    "128x256x64w14": (128, 256, 64, 1, 4, 16, 16, 16, False, 4, 0, (9.0, 0.0245)),
    # hipBLASLt-like non-power-of-2 tiles (2x2 waves, mode 4;
    # no spills on RCR); placeholder models until calibrated
    "320x192x64": (320, 192, 64, 2, 2, 16, 16, 16, True, 4, 0, (16.88, 0.0459)),
    "192x320x64": (192, 320, 64, 2, 2, 16, 16, 16, True, 4, 0, (16.88, 0.0459)),
    "256x160x64": (256, 160, 64, 2, 2, 16, 16, 16, True, 4, 0, (11.25, 0.0306)),
    "160x256x64": (160, 256, 64, 2, 2, 16, 16, 16, True, 4, 0, (11.25, 0.0306)),
    "160x192x64": (160, 192, 64, 2, 2, 16, 16, 16, False, 4, 0, (8.44, 0.0230)),
    "192x160x64": (192, 160, 64, 2, 2, 16, 16, 16, False, 4, 0, (8.44, 0.0230)),
    "128x160x64": (128, 160, 64, 2, 2, 16, 16, 16, False, 4, 0, (5.62, 0.0153)),
    "160x128x64": (160, 128, 64, 2, 2, 16, 16, 16, False, 4, 0, (5.62, 0.0153)),
    "128x96x64": (128, 96, 64, 2, 2, 16, 16, 16, False, 4, 0, (3.38, 0.0092)),
    "96x128x64": (96, 128, 64, 2, 2, 16, 16, 16, False, 4, 0, (3.38, 0.0092)),
    "96x96x64": (96, 96, 64, 2, 2, 16, 16, 16, False, 4, 0, (2.53, 0.0069)),
    "320x128x64": (320, 128, 64, 2, 2, 16, 16, 16, True, 4, 0, (11.25, 0.0306)),
    "128x320x64": (128, 320, 64, 2, 2, 16, 16, 16, True, 4, 0, (11.25, 0.0306)),
    "192x192x64": (192, 192, 64, 2, 2, 16, 16, 16, True, 4, 0, (10.12, 0.0276)),
    "64x96x128": (64, 96, 128, 2, 2, 16, 16, 16, False, 4, 0, (2.50, 0.0046)),
    "96x64x128": (96, 64, 128, 2, 2, 16, 16, 16, False, 4, 0, (2.50, 0.0046)),
    "96x96x128": (96, 96, 128, 2, 2, 16, 16, 16, False, 4, 0, (2.53, 0.0069)),
    "128x96x128": (128, 96, 128, 2, 2, 16, 16, 16, False, 4, 0, (3.38, 0.0092)),
    "96x128x128": (96, 128, 128, 2, 2, 16, 16, 16, False, 4, 0, (3.38, 0.0092)),
}
# geometries the planner skips unless R2_EXTRA_GEOS_ON
R2_EXTRA_GEOS = (
    "256x128x64",
    "128x256x64",
    "256x128x64w41",
    "128x256x64w14",
    "320x192x64",
    "192x320x64",
    "256x160x64",
    "160x256x64",
    "160x192x64",
    "192x160x64",
    "128x160x64",
    "160x128x64",
    "128x96x64",
    "96x128x64",
    "96x96x64",
    "320x128x64",
    "128x320x64",
    "192x192x64",
    "64x96x128",
    "96x64x128",
    "96x96x128",
    "128x96x128",
    "96x128x128",
)
R2_EXTRA_GEOS_ON = False
# the last field: the per-tile time model (a fixed term and b per K element,
# both in microseconds): one wave of tiles takes a + b * slice_K, fitted per
# geometry on gfx942 interleaved timings with every timed kernel f64-checked
# smallest M / N the candidate takes (skinny / decode shapes go elsewhere)
G5_VGPR_MIN_MN = 16
# split-K for grids that leave most CUs idle (``select_split_k`` with the
# arch's CU count as the CTA target); the slices are whole tiles
G5_VGPR_SPLIT_K = True
G5_VGPR_SPLITS = (2, 4, 8, 16)
# sweep space of the candidate: the N cheapest plans of the cost model
G5_VGPR_SWEEP = 6
# split-K extra cost in microseconds: (fixed: zero-fill + f32 -> 16-bit pass
# + launches, per byte of f32 partial traffic)
G5_VGPR_SPLIT_COST = (4.0, 2.5e-6)
# smallest per-wave fixed time (microseconds; the fitted intercepts of the
# wide-tile bodies fall below it where only short K was fitted) and the
# operand-traffic ceiling (bytes per microsecond; the fitted envelope of the
# L2 / MALL-fed streaming rate on gfx942)
G5_VGPR_A_MIN = 2.5
G5_VGPR_TRAFFIC_BW = 9.0e6
# K % tile_k != 0 (K % 8 == 0): the body masks the tail tile's columns past K
# (``pad_k`` on the vgpr_prefetch body; no split-K then)
G5_VGPR_K_TAIL = True
# pad_fast_tile on the padded bodies (off: it doubles the epilogue, and the
# 256-accumulator bodies then exceed the comgr compile budget)
G5_VGPR_PAD_FAST_TILE = False


def _g5_tile_trait(geo: str, pad_m: bool, pad_n: bool) -> tuple[TileSpec, TraitSpec]:
    tm, tn, tk, wm, wn, wtm, wtn, wtk, agpr, mode, swr, _eff = G5_VGPR_GEOS[geo]
    tile = TileSpec(
        tile_m=tm,
        tile_n=tn,
        tile_k=tk,
        warp_m=wm,
        warp_n=wn,
        warp_k=1,
        warp_tile_m=wtm,
        warp_tile_n=wtn,
        warp_tile_k=wtk,
    )
    trait = TraitSpec(
        pipeline="compv4",
        scheduler="intrawave",
        epilogue="default",
        ab_double_buffer=mode != 4,
        chiplet_swizzle=True,
        chiplet_wgm=8,
        chiplet_chunk_size=64,
        lds_swizzle=True,
        lds_swizzle_r=swr,
        acc_agpr=agpr,
        vgpr_prefetch=mode,
        vgpr_prefetch_depth=G5_VGPR_DEPTH,
        pad_m=pad_m,
        pad_n=pad_n,
        pad_fast_tile=G5_VGPR_PAD_FAST_TILE and (pad_m or pad_n),
    )
    return tile, trait


def g5_vgpr_cost(
    M: int, N: int, K: int, arch: str, geo: str, split_k: int = 1
) -> float:
    """Modelled time (microseconds) of ``geo`` with ``split_k`` K slices for
    ``M x N x K``: waves of tiles x (a + b * slice K), plus the split-K cost
    (``G5_VGPR_SPLIT_COST``)."""
    tm, tn, tk, (a, b) = (G5_VGPR_GEOS[geo][i] for i in (0, 1, 2, -1))
    cus = arch_num_cus(arch)
    tiles = -(-M // tm) * -(-N // tn)
    ks = -(-K // (tk * split_k)) * tk
    cost = -(-(tiles * split_k) // cus) * (max(a, G5_VGPR_A_MIN) + b * ks)
    # operand traffic floor: every tile streams (tile_m + tile_n) x K elements
    cost = max(cost, tiles * split_k * (tm + tn) * ks * 2 / G5_VGPR_TRAFFIC_BW)
    if split_k > 1:
        fixed, per_byte = G5_VGPR_SPLIT_COST
        cost += fixed + per_byte * M * N * 4 * split_k
    return cost


def g5_vgpr_plan(M: int, N: int, K: int, arch: str) -> Optional[tuple[str, int]]:
    """(geometry, split_k) of least modelled cost (None: no geometry applies)."""
    best, best_cost = None, None
    for name, g in G5_VGPR_GEOS.items():
        tk = g[2]
        if name in R2_EXTRA_GEOS and not R2_EXTRA_GEOS_ON:
            continue
        if K % tk and (not G5_VGPR_K_TAIL or K % 8):
            continue
        splits = [1]
        if G5_VGPR_SPLIT_K and not K % tk:
            splits += [s for s in G5_VGPR_SPLITS if K % (s * tk) == 0]
        for s in splits:
            cost = g5_vgpr_cost(M, N, K, arch, name, s)
            if best_cost is None or cost < best_cost:
                best, best_cost = (name, s), cost
    return best


def g5_vgpr_plans(M: int, N: int, K: int, arch: str, any_k_tail: bool = False) -> list:
    """Every applicable (cost, geometry, split_k), cheapest first
    (``any_k_tail``: the K tail for any K, the native TN form whose
    operands are both MN-contiguous)."""
    out = []
    for name, g in G5_VGPR_GEOS.items():
        tk = g[2]
        if name in R2_EXTRA_GEOS and not R2_EXTRA_GEOS_ON:
            continue
        if K % tk and (not G5_VGPR_K_TAIL or (K % 8 and not any_k_tail)):
            continue
        splits = [1]
        if G5_VGPR_SPLIT_K and not K % tk:
            splits += [s for s in G5_VGPR_SPLITS if K % (s * tk) == 0]
        for s in splits:
            out.append((g5_vgpr_cost(M, N, K, arch, name, s), name, s))
    return sorted(out)


def g5_vgpr_sweep_tile_traits(req: GemmRequest) -> list:
    """The ``G5_VGPR_SWEEP`` cheapest (tile, trait) of the cost model (the
    candidate's sweep space: a registry sweep / the dispatcher sweep measures
    them next to the pick)."""
    if (
        not G5_VGPR
        or req.arch not in G5_VGPR_ARCHES
        or min(req.M, req.N) < G5_VGPR_MIN_MN
    ):
        return []
    if _g5p_on(req):
        from .g5_plan import plan_tile_trait

        return [
            plan_tile_trait(p, req.M, req.N, req.K)
            for _t, p in _g5p_ranked(req)
            if p.geo != "skinny"
        ][:G5_VGPR_SWEEP]
    out = []
    if _r2_lay(req) != "RCR":
        # the cheapest plans that have a native form
        if not (R2_VP_MN and req.arch in R2_VP_MN_ARCHES):
            return []
        for _c, geo, s in r2_vp_mn_plans(req)[:G5_VGPR_SWEEP]:
            out.append(_g5_plan_tile_trait(req, geo, s))
        return out
    for _c, geo, s in g5_vgpr_plans(req.M, req.N, req.K, req.arch)[:G5_VGPR_SWEEP]:
        tm, tn, tk = G5_VGPR_GEOS[geo][0], G5_VGPR_GEOS[geo][1], G5_VGPR_GEOS[geo][2]
        tile, trait = _g5_tile_trait(geo, bool(req.M % tm), bool(req.N % tn))
        if req.K % tk:
            trait = replace(trait, pad_k=True)
        elif s > 1:
            trait = replace(trait, split_k=s, pad_fast_tile=False)
        out.append((tile, trait))
    return out


def g5_vgpr_geometry(M: int, N: int, K: int, arch: str) -> Optional[str]:
    """The geometry of :func:`g5_vgpr_plan` (None: no geometry applies)."""
    plan = g5_vgpr_plan(M, N, K, arch)
    return plan[0] if plan else None


# NN / TN / TT through the native form of the gfx942 register-staged plan
# model's RCR pick (g5_plan's full geometry menu) instead of the cheapest
# fitted plan of the smaller geometry menu (r2_vp_mn_plans), for the layouts
# listed (data layouts; () = off). On for RRR and CCR, where the plan model's
# pick is the faster one; TN (CRR) keeps the fitted plan.
# MERGE2_NATIVE_G5P_TOP = how many of the plan model's ranked plans are tried
# for a launchable native form.
MERGE2_NATIVE_G5P: Tuple[str, ...] = ("RRR", "CCR")
MERGE2_NATIVE_G5P_TOP = 4


def merge2_g5p_native_tile_trait(
    req: GemmRequest,
) -> Optional[tuple[TileSpec, TraitSpec]]:
    """The RCR (tile, trait) of the best-ranked plan-model plan of ``req``'s
    RCR view whose native form launches for ``req`` (None: none / model off)."""
    rreq = replace(req, layout="RCR", trans_a=False, trans_b=True)
    if not _g5p_on(rreq):
        return None
    from ...instances.common.gemm_universal import DataSpec
    from .g5_plan import plan_tile_trait

    dt = str(getattr(req, "dtype", "fp16"))
    dt = dt if dt in ("fp16", "bf16") else "fp16"
    plans = [p for _t, p in _g5p_ranked(rreq) if p.geo != "skinny"]
    for p in plans[: max(1, int(MERGE2_NATIVE_G5P_TOP))]:
        tile, trait = plan_tile_trait(p, int(req.M), int(req.N), int(req.K))
        if int(getattr(trait, "split_k_serial", 0)):
            continue  # the native forms are validated with the atomic split only
        spec = UniversalGemmSpec(
            name="merge2_probe",
            tile=tile,
            trait=trait,
            data=DataSpec(dtype_a=dt, dtype_b=dt, dtype_c=dt, layout="RCR"),
            wave_size=64,
        )
        if r2_vp_mn_form(req, spec) is not None:
            return tile, trait
    return None


def g5_vgpr_tile_trait(req: GemmRequest) -> Optional[tuple[TileSpec, TraitSpec]]:
    """The register-staged body for ``req`` (None when not applicable)."""
    if not G5_VGPR or req.arch not in G5_VGPR_ARCHES:
        return None
    if min(req.M, req.N) < G5_VGPR_MIN_MN:
        return None
    if _r2_lay(req) != "RCR":
        # the cheapest plan with a native NN / TN / TT form
        if not (R2_VP_MN and req.arch in R2_VP_MN_ARCHES):
            return None
        if _r2_lay(req) in tuple(MERGE2_NATIVE_G5P):
            tt = merge2_g5p_native_tile_trait(req)
            if tt is not None:
                return tt
        plans = r2_vp_mn_plans(req)
        if not plans:
            return None
        return _g5_plan_tile_trait(req, plans[0][1], plans[0][2])
    if _g5p_on(req):
        from .g5_plan import plan_tile_trait

        g5 = [p for _t, p in _g5p_ranked(req) if p.geo != "skinny"]
        return plan_tile_trait(g5[0], req.M, req.N, req.K) if g5 else None
    plan = g5_vgpr_plan(req.M, req.N, req.K, req.arch)
    if plan is None:
        return None
    geo, split_k = plan
    tm, tn, tk = G5_VGPR_GEOS[geo][0], G5_VGPR_GEOS[geo][1], G5_VGPR_GEOS[geo][2]
    tile, trait = _g5_tile_trait(geo, bool(req.M % tm), bool(req.N % tn))
    if req.K % tk:
        trait = replace(trait, pad_k=True)
    elif split_k > 1:
        trait = replace(trait, split_k=split_k, pad_fast_tile=False)
    return tile, trait


# the region rule (gemm_skinny ahead of the register-staged body for small min(M, N),
# skinny.SKINNY_OVER_VGPR) and the learned gfx942 plan model (which ranks the gemm_skinny
# plan among the register-staged plans, g5_plan.G5P_SKINNY_ARB) both arbitrate the two on
# gfx942 RCR. Default: the model decides when it is on; True = the region rule applies too.
MERGE2_SKINNY_OVER_VGPR_WITH_MODEL = False


def g5_vgpr_admit(req: GemmRequest, spec: UniversalGemmSpec) -> tuple[bool, str]:
    from ...instances.common.gemm_universal import launch_shape_error

    if g5_vgpr_tile_trait(req) is None:
        return False, (
            f"g5 register-staged body: not applicable to {req.M}x{req.N}x{req.K} "
            f"on {req.arch} (G5_VGPR={G5_VGPR}, min(M, N) >= {G5_VGPR_MIN_MN}, "
            "K a multiple of a geometry's tile_k)"
        )
    if _r2_lay(req) != "RCR":
        # the launch rules of the native form
        if r2_vp_mn_form(req, spec) is None:
            return False, f"layout {_r2_lay(req)}: no launchable vp_mn form"
        from .skinny import skinny_native_over_vgpr

        if skinny_native_over_vgpr(req):
            return False, "gemm_skinny serves it (skinny.SKINNY_OVER_VGPR_NATIVE)"
        return True, "ok"
    err = launch_shape_error(spec, req.M, req.N, req.K)
    if err:
        return False, err
    if _g5p_on(req):
        from . import g5_plan as _gp

        r = _g5p_ranked(req)
        if _gp.G5P_SKINNY_ARB and r and r[0][1].geo == "skinny":
            g5t = next((t for t, p in r if p.geo != "skinny"), None)
            if g5t is None or math.log(g5t / r[0][0]) > _gp.G5P_SKINNY_MARGIN:
                return False, (
                    "g5 register-staged body steps aside: the gfx942 plan model "
                    "predicts the gemm_skinny plan faster"
                )
    if not _g5p_on(req) or MERGE2_SKINNY_OVER_VGPR_WITH_MODEL:
        # the region rule (skinny.SKINNY_OVER_VGPR); with the gfx942 plan model on,
        # the model arbitrates (above) unless MERGE2_SKINNY_OVER_VGPR_WITH_MODEL
        from .skinny import skinny_takes_over_vgpr

        if skinny_takes_over_vgpr(req):
            return False, "gemm_skinny serves it (skinny.SKINNY_OVER_VGPR)"
    return True, "ok"


# gfx942 NT: the register-staged plan of the geometry menu and the
# learned plan model (``g5_plan``); cached per request and knob state
_G5P_CACHE: dict = {}


def _g5p_on(req: GemmRequest) -> bool:
    from . import g5_plan as _gp

    return (
        _gp.G5P_ON
        and req.arch in _gp.G5P_ARCHES
        and str(getattr(req, "layout", "RCR")).upper() == "RCR"
        and bool(_gp.load_menu(req.arch))
    )


def _g5p_ranked(req: GemmRequest) -> list:
    """(predicted time, G5Plan) of the register-staged plans for ``req`` cheapest first, with
    the gemm_skinny plan ranked among them when that candidate admits the
    request (``G5P_SKINNY_ARB``)."""
    from . import g5_plan as _gp

    knobs = (
        _gp.G5P_SPLITS, _gp.G5P_SERIAL_MODES, _gp.G5P_SPLIT_MAX_WAVES,
        _gp.G5P_PRESELECT, _gp.G5P_MODEL, _gp.G5P_SKINNY_ARB,
        tuple(sorted(_gp.G5P_SPLIT_COST.items())), _gp.G5P_TRAFFIC_BW,
        _gp.G5P_VP_UNROLL_MAX_ACC, _gp.G5P_VP_UNROLL_SCALE,
    )  # fmt: skip
    key = (int(req.M), int(req.N), int(req.K), str(req.dtype), req.arch, knobs)
    if key in _G5P_CACHE:
        return _G5P_CACHE[key]
    extra = []
    if _gp.G5P_SKINNY_ARB and _gp.G5P_MODEL:
        from .skinny import SKINNY_DISPATCH, skinny_admit_region, skinny_plan

        try:
            if SKINNY_DISPATCH and skinny_admit_region(req)[0]:
                extra.append(_gp.G5Plan("skinny", skinny=skinny_plan(req)))
        except Exception:  # noqa: BLE001 - no skinny plan: g5 decides alone
            extra = []
    out = _gp.rank_plans(req.M, req.N, req.K, str(req.dtype), req.arch, extra)
    if len(_G5P_CACHE) > 4096:
        _G5P_CACHE.clear()
    _G5P_CACHE[key] = out
    return out


# ---------------------------------------------------------------------------
# gfx942: native NN / TN / TT in the register-staged body
# ---------------------------------------------------------------------------
# The register-staged candidate (universal_gemm_*_cdna_vgpr) serves the non-RCR layouts
# itself: an MN-contiguous operand is read as micro-blocks of vp_mn_kv K rows
# x mn_vec M / N elements and transposed in registers into the K-contiguous
# LDS tile (TraitSpec.vp_mn_kv; gfx942 has no ds_read_b64_tr_b16). False: the
# candidate stays RCR-only (non-RCR requests take the other candidates'
# native forms).
R2_VP_MN = True
R2_VP_MN_ARCHES = ("gfx942",)
# (vp_mn_kv, mn_vec) micro-blocks, preferred first (the first one the tile,
# the thread count and the leading dimension admit)
R2_VP_MN_MICRO = (
    (8, 8),
    (8, 4),
    (4, 8),
    (4, 4),
    (8, 2),
    (2, 8),
    (4, 2),
    (2, 4),
    (2, 2),
)
# micro-block counts that are not a multiple of the thread count
# (TraitSpec.vp_mn_wrap: surplus threads repeat micro-blocks) after every
# whole-round micro-block (True) or never (False)
R2_VP_MN_WRAP = True
# trait settings of the native forms (the faster settings on gfx942)
R2_VP_MN_TRAIT = {"vp_mn_pipe": 1}
# scheduler per layout (both operands MN-contiguous: the default scheduler
# shuffles the AGPR accumulators in the K loop; iterative-minreg does not)
R2_VP_MN_SCHED = {"CRR": "iterative-minreg"}
# modelled time of a native form over its RCR body: per layout, and per
# micro-block size (elements)
R2_VP_MN_COST = {"RRR": 1.04, "CCR": 1.02, "CRR": 1.06}
R2_VP_MN_MICRO_COST = {64: 1.0, 32: 1.02, 16: 1.06, 8: 1.12, 4: 1.25}
R2_VP_MN_WRAP_COST = 0.5  # x the fraction of repeated micro-blocks
# TN (both operands MN-contiguous): K rows load one by one, so the K tail
# needs no K alignment (K % 8 != 0 too); False: K % 8 == 0 as for RCR
R2_VP_MN_TN_ANY_K = True
# per (layout, register-staged geometry): the fastest micro-block (vp_mn_kv,
# mn_vec; vp_mn_wrap where needed) and its time relative to the RCR body
# (geomean over a shape set per layout on gfx942, every micro-block of every
# geometry timed next to the RCR body in one process). A form with
# this micro-block gets this factor instead of the generic one above; the
# table's micro-block is tried first.
R2_VP_MN_TABLE = {
    "RRR": {
        "256x192x64": (8, 8, 1.193),
        "256x224x64": (4, 8, 1.081),
        "256x256x32": (4, 8, 1.133),
        "256x256x64": (4, 8, 1.058),
        "256x256x64w8": (4, 8, 1.248),
        "256x64x64": (4, 4, 1.179),
        "224x256x64": (4, 8, 1.055),
        "192x256x64": (4, 8, 1.073),
        "192x256x64w8": (4, 8, 1.047),
        "128x128x128": (8, 8, 1.136),
        "128x128x32": (4, 4, 1.749),
        "128x128x64": (4, 8, 1.134),
        "128x32x128": (4, 4, 1.335),
        "128x64x128": (4, 8, 1.157),
        "64x128x128": (8, 8, 1.238),
        "64x256x64": (4, 8, 1.252),
        "64x64x128": (4, 8, 1.245),
        "32x128x128": (8, 8, 1.348),
    },
    "CRR": {
        "256x192x64": (4, 8, 1.286),
        "256x224x64": (4, 8, 1.2),
        "256x256x32": (4, 8, 1.134),
        "256x256x64": (8, 8, 1.099),
        "256x256x64w8": (4, 8, 1.147),
        "256x64x64": (4, 8, 1.506),
        "224x256x64": (4, 8, 1.28),
        "192x256x64": (4, 8, 1.315),
        "192x256x64w8": (4, 8, 1.191),
        "128x128x128": (8, 8, 1.287),
        "128x128x32": (4, 4, 2.92),
        "128x128x64": (4, 8, 1.269),
        "128x32x128": (4, 8, 1.829),
        "128x64x128": (4, 8, 1.495),
        "64x128x128": (4, 8, 1.472),
        "64x256x64": (4, 8, 1.499),
        "64x64x128": (4, 8, 1.563),
        "32x128x128": (4, 8, 1.789),
    },
    "CCR": {
        "256x192x64": (4, 8, 1.078),
        "256x224x64": (4, 8, 1.066),
        "256x256x32": (4, 8, 1.128),
        "256x256x64": (4, 8, 1.049),
        "256x256x64w8": (4, 8, 1.25),
        "256x64x64": (4, 8, 1.284),
        "224x256x64": (4, 8, 1.077),
        "192x256x64": (8, 8, 1.153),
        "192x256x64w8": (4, 8, 1.11),
        "128x128x128": (8, 8, 1.135),
        "128x128x32": (4, 4, 1.719),
        "128x128x64": (4, 8, 1.142),
        "128x32x128": (8, 8, 1.369),
        "128x64x128": (8, 8, 1.26),
        "64x128x128": (4, 8, 1.152),
        "64x256x64": (4, 4, 1.177),
        "64x64x128": (4, 8, 1.261),
        "32x128x128": (4, 4, 1.306),
    },
}
R2_VP_MN_USE_TABLE = True
# successive-halving tuned forms (tune-engine SuccessiveHalving over a
# growing shape set per (layout, geometry), each config timed against the
# dispatcher's form in one process on gfx942): (micro-block, trait
# overrides, time relative to that form). The form takes
# the trait overrides and tries the micro-block first; its cost is the
# fitted model x the factor.
R2_VP_MN_TUNED = {
    ("CCR", "128x128x32"): (
        (4, 4),
        {
            "ab_double_buffer": True,
            "lds_swizzle_r": 1,
            "sched_strategy": "",
            "vgpr_prefetch": 3,
            "vp_mn_order": 2,
            "vp_mn_pipe": 1,
            "vp_mn_worder": 2,
        },
        0.898,
    ),
    ("CCR", "128x128x64"): (
        (4, 8),
        {
            "lds_swizzle_r": 0,
            "sched_strategy": "",
            "vp_mn_order": 2,
            "vp_mn_pipe": 1,
            "vp_mn_worder": 0,
        },
        0.922,
    ),
    ("CCR", "256x224x64"): (
        (4, 8),
        {
            "lds_swizzle_r": 0,
            "sched_strategy": "",
            "vp_mn_order": 2,
            "vp_mn_pipe": 0,
            "vp_mn_worder": 1,
        },
        0.982,
    ),
    ("CCR", "256x256x64"): (
        (8, 8),
        {
            "lds_swizzle_r": 0,
            "sched_strategy": "",
            "vp_mn_order": 2,
            "vp_mn_pipe": 1,
            "vp_mn_worder": 3,
        },
        0.987,
    ),
    ("CCR", "64x64x128"): (
        (4, 8),
        {
            "ab_double_buffer": True,
            "lds_swizzle_r": 0,
            "sched_strategy": "",
            "vgpr_prefetch": 3,
            "vp_mn_order": 2,
            "vp_mn_pipe": 0,
            "vp_mn_worder": 1,
        },
        0.863,
    ),
    ("CRR", "192x256x64w8"): (
        (4, 8),
        {
            "lds_swizzle_r": 0,
            "sched_strategy": "",
            "vp_mn_order": 2,
            "vp_mn_pipe": 0,
            "vp_mn_worder": 1,
        },
        0.902,
    ),
    ("CRR", "256x256x64w8"): (
        (4, 8),
        {"sched_strategy": "", "vp_mn_order": 0, "vp_mn_pipe": 1, "vp_mn_worder": 3},
        0.991,
    ),
    ("CRR", "64x128x128"): (
        (4, 8),
        {
            "lds_swizzle_r": 1,
            "sched_strategy": "",
            "vp_mn_order": 2,
            "vp_mn_pipe": 1,
            "vp_mn_worder": 3,
        },
        0.737,
    ),
    ("RRR", "128x128x32"): (
        (4, 4),
        {
            "ab_double_buffer": True,
            "lds_swizzle_r": 1,
            "sched_strategy": "",
            "vgpr_prefetch": 3,
            "vp_mn_order": 2,
            "vp_mn_pipe": 1,
            "vp_mn_worder": 2,
        },
        0.929,
    ),
    ("RRR", "128x128x64"): (
        (4, 8),
        {
            "lds_swizzle_r": 0,
            "sched_strategy": "",
            "vp_mn_order": 2,
            "vp_mn_pipe": 1,
            "vp_mn_worder": 0,
        },
        0.915,
    ),
    ("RRR", "256x256x64"): (
        (4, 8),
        {
            "lds_swizzle_r": 0,
            "sched_strategy": "",
            "vp_mn_order": 2,
            "vp_mn_pipe": 1,
            "vp_mn_worder": 0,
        },
        0.984,
    ),
    ("RRR", "64x64x128"): (
        (4, 8),
        {
            "lds_swizzle_r": 0,
            "sched_strategy": "",
            "vp_mn_order": 2,
            "vp_mn_pipe": 1,
            "vp_mn_worder": 0,
        },
        0.792,
    ),
    # (the top configs of the targets whose halving state lost its last
    # rung, re-timed on the full shape set)
    ("CCR", "192x256x64w8"): (
        (4, 8),
        {
            "lds_swizzle_r": 0,
            "sched_strategy": "",
            "vp_mn_order": 2,
            "vp_mn_pipe": 1,
            "vp_mn_worder": 0,
        },
        0.941,
    ),
    ("CCR", "64x128x128"): (
        (4, 8),
        {
            "lds_swizzle_r": 0,
            "sched_strategy": "",
            "vp_mn_order": 2,
            "vp_mn_pipe": 0,
            "vp_mn_worder": 1,
        },
        0.921,
    ),
    ("CRR", "128x128x64"): (
        (4, 8),
        {
            "sched_strategy": "iterative-minreg",
            "vp_mn_order": 2,
            "vp_mn_pipe": 0,
            "vp_mn_worder": 0,
        },
        0.85,
    ),
    ("CRR", "64x64x128"): (
        (4, 8),
        {
            "lds_swizzle_r": 0,
            "sched_strategy": "",
            "vp_mn_order": 2,
            "vp_mn_pipe": 1,
            "vp_mn_worder": 0,
        },
        0.67,
    ),
    ("RRR", "128x64x128"): (
        (4, 8),
        {
            "lds_swizzle_r": 0,
            "sched_strategy": "",
            "vp_mn_order": 2,
            "vp_mn_pipe": 1,
            "vp_mn_worder": 0,
        },
        0.929,
    ),
}
R2_VP_MN_USE_TUNED = True
# (fit data: r2_native_fit.py; with vp_mn_wrap where the micro-block needs it)
from .r2_native_fit import R2_NATIVE_LIN  # noqa: E402

# the fitted per (layout, geometry, micro-block) time models
# (r2_native_fit.R2_NATIVE_LIN) for the native plans' cost and micro-block
# choice; False: the RCR model x r2_native_cost_factor
R2_VP_MN_USE_FIT = True


def g5_geo_of(spec: UniversalGemmSpec) -> Optional[str]:
    """The G5_VGPR_GEOS name of a g5 body (None: not a g5 geometry)."""
    t = spec.tile
    for name, g in G5_VGPR_GEOS.items():
        if (g[0], g[1], g[2], g[3], g[4]) == (
            t.tile_m,
            t.tile_n,
            t.tile_k,
            t.warp_m,
            t.warp_n,
        ):
            return name
    return None


def _r2_lay(req) -> str:
    return str(getattr(req, "layout", "RCR") or "RCR").upper()


def r2_vp_mn_form(
    req: GemmRequest, spec: UniversalGemmSpec
) -> Optional[UniversalGemmSpec]:
    """The native NN / TN / TT form of a g5 vgpr_prefetch spec for req
    (None: no micro-block fits the tile / thread count / leading dimension,
    or the request is RCR)."""
    from ...instances.common.gemm_universal import is_valid_spec, launch_shape_error

    lay = _r2_lay(req)
    if lay == "RCR" or not int(getattr(spec.trait, "vgpr_prefetch", 0)):
        return None
    tr0 = replace(
        spec.trait,
        sched_strategy=R2_VP_MN_SCHED.get(lay, spec.trait.sched_strategy),
        **R2_VP_MN_TRAIT,
    )
    data = replace(spec.data, layout=lay)
    M, N, K = int(req.M), int(req.N), int(req.K)
    geo = g5_geo_of(spec) or ""
    order = []
    tuned = R2_VP_MN_TUNED.get((lay, geo)) if R2_VP_MN_USE_TUNED else None
    if tuned is not None and not int(spec.trait.split_k) > 1:
        tr_t = replace(tr0, **tuned[1])
        for wrap in (0, 1):
            s = replace(
                spec,
                data=data,
                trait=replace(
                    tr_t, vp_mn_kv=tuned[0][0], mn_vec=tuned[0][1], vp_mn_wrap=wrap
                ),
            )
            if is_valid_spec(s, arch=req.arch)[0] and not launch_shape_error(
                s, M, N, K
            ):
                return s
        # the tuned micro-block does not fit this leading dimension: the
        # tuned traits with the next micro-block that does
        tr0 = tr_t
    if R2_VP_MN_USE_FIT:
        # the fitted micro-blocks of this (layout, geometry), cheapest first
        # for this K (one wave's a + b * K_slice)
        ks = -(-K // (spec.tile.tile_k * int(spec.trait.split_k))) * spec.tile.tile_k
        fits = [
            (ab[0] + ab[1] * ks, mic)
            for (L_, g_, mic), ab in R2_NATIVE_LIN.items()
            if L_ == lay and g_ == geo
        ]
        order += [(mic, w) for _c, mic in sorted(fits) for w in (0, 1)]
    ent = R2_VP_MN_TABLE.get(lay, {}).get(geo) if R2_VP_MN_USE_TABLE else None
    if ent is not None:
        # the measured micro-block of this (layout, geometry)
        order += [(ent[:2], 0), (ent[:2], 1)]
    order += [
        (mic, w) for w in ((0, 1) if R2_VP_MN_WRAP else (0,)) for mic in R2_VP_MN_MICRO
    ]
    seen = set()
    for (kv, mv), wrap in order:
        if (kv, mv, wrap) in seen:
            continue
        seen.add((kv, mv, wrap))
        s = replace(
            spec,
            data=data,
            trait=replace(tr0, vp_mn_kv=kv, mn_vec=mv, vp_mn_wrap=wrap),
        )
        if is_valid_spec(s, arch=req.arch)[0] and not launch_shape_error(s, M, N, K):
            return s
    return None


def _g5_plan_tile_trait(
    req: GemmRequest, geo: str, split_k: int
) -> tuple[TileSpec, TraitSpec]:
    """The RCR (tile, trait) of g5 plan (geo, split_k) for req."""
    tm, tn, tk = G5_VGPR_GEOS[geo][0], G5_VGPR_GEOS[geo][1], G5_VGPR_GEOS[geo][2]
    tile, trait = _g5_tile_trait(geo, bool(req.M % tm), bool(req.N % tn))
    if req.K % tk:
        trait = replace(trait, pad_k=True)
    elif split_k > 1:
        trait = replace(trait, split_k=split_k, pad_fast_tile=False)
    return tile, trait


def r2_native_cost_factor(lay: str, spec: UniversalGemmSpec) -> float:
    """Modelled time of the native form spec over its RCR body."""
    t, tr = spec.tile, spec.trait
    ent = (
        R2_VP_MN_TABLE.get(lay, {}).get(g5_geo_of(spec) or "")
        if R2_VP_MN_USE_TABLE
        else None
    )
    if ent is not None and (int(tr.vp_mn_kv), int(tr.mn_vec)) == ent[:2]:
        return float(ent[2])
    f = R2_VP_MN_COST.get(lay, 1.0)
    f *= R2_VP_MN_MICRO_COST.get(int(tr.vp_mn_kv) * int(tr.mn_vec), 1.3)
    if ent is not None:
        # not the measured micro-block (the leading dimension refused it)
        f = max(f, float(ent[2]) * 1.15)
    if int(tr.vp_mn_wrap):
        a_mn, b_mn = lay[0] == "C", lay[1] == "R"
        rep = 0.0
        for is_mn, tmn in ((a_mn, t.tile_m), (b_mn, t.tile_n)):
            if is_mn:
                n = (tmn // int(tr.mn_vec)) * (t.tile_k // int(tr.vp_mn_kv))
                th = spec.block_size
                rep = max(rep, (-(-n // th) * th - n) / max(1, n))
        f *= 1.0 + R2_VP_MN_WRAP_COST * rep
    return f


def r2_native_cost(
    req: GemmRequest, geo: str, split_k: int, form: UniversalGemmSpec, rcr_cost: float
) -> float:
    """Modelled time (us) of the native form ``form`` of g5 plan (geo,
    split_k): the fitted model of its (layout, geometry, micro-block) when there
    is one (waves x (a + b * K_slice), the g5 traffic floor and split cost as
    in g5_vgpr_cost), else ``rcr_cost`` x :func:`r2_native_cost_factor`."""
    lay = _r2_lay(req)
    tr = form.trait
    ab = (
        R2_NATIVE_LIN.get((lay, geo, (int(tr.vp_mn_kv), int(tr.mn_vec))))
        if R2_VP_MN_USE_FIT
        else None
    )
    tuned = R2_VP_MN_TUNED.get((lay, geo)) if R2_VP_MN_USE_TUNED else None
    t_factor = 1.0
    if ab is not None and tuned is not None and R2_VP_MN_USE_FIT:
        # the tuning baseline was the fitted-model form: the cheapest fitted
        # micro-block for this K, x the measured tuning factor
        tk_ = G5_VGPR_GEOS[geo][2]
        ks_ = -(-int(req.K) // (tk_ * split_k)) * tk_
        cands = [
            v for (L_, g_, _m), v in R2_NATIVE_LIN.items() if L_ == lay and g_ == geo
        ]
        ab = min(cands, key=lambda v: v[0] + v[1] * ks_)
        t_factor = float(tuned[2])
    if ab is None:
        return rcr_cost * r2_native_cost_factor(lay, form)
    M, N, K = int(req.M), int(req.N), int(req.K)
    tm, tn, tk = G5_VGPR_GEOS[geo][0], G5_VGPR_GEOS[geo][1], G5_VGPR_GEOS[geo][2]
    tiles = -(-M // tm) * -(-N // tn)
    ks = -(-K // (tk * split_k)) * tk
    cost = -(-(tiles * split_k) // arch_num_cus(req.arch)) * (
        max(ab[0], G5_VGPR_A_MIN) + ab[1] * ks
    )
    cost = max(cost, tiles * split_k * (tm + tn) * ks * 2 / G5_VGPR_TRAFFIC_BW)
    if split_k > 1:
        fixed, per_byte = G5_VGPR_SPLIT_COST
        cost += fixed + per_byte * M * N * 4 * split_k
    return cost * t_factor


_R2_PLAN_CACHE: dict = {}


def r2_vp_mn_plans(req: GemmRequest) -> list:
    """Every (cost, geometry, split_k) of the g5 cost model whose body has a
    native form for the non-RCR req (cost x r2_native_cost_factor),
    cheapest first."""
    lay = _r2_lay(req)
    key = (
        int(req.M),
        int(req.N),
        int(req.K),
        req.arch,
        lay,
        str(getattr(req, "dtype", "fp16")),
        R2_VP_MN,
        R2_VP_MN_WRAP,
        tuple(R2_VP_MN_MICRO),
        tuple(sorted(R2_VP_MN_TRAIT.items())),
        tuple(sorted(R2_VP_MN_SCHED.items())),
        tuple(sorted(R2_VP_MN_COST.items())),
        tuple(sorted(R2_VP_MN_MICRO_COST.items())),
        R2_VP_MN_WRAP_COST,
        R2_VP_MN_USE_TABLE,
        R2_VP_MN_USE_FIT,
        len(R2_NATIVE_LIN),
        R2_VP_MN_TN_ANY_K,
        R2_VP_MN_USE_TUNED,
        len(R2_VP_MN_TUNED),
        R2_EXTRA_GEOS_ON,
        tuple(sorted((k, tuple(sorted(v.items()))) for k, v in R2_VP_MN_TABLE.items())),
        tuple(sorted((k, v[:3] + v[-1:]) for k, v in G5_VGPR_GEOS.items())),
        G5_VGPR_SPLIT_K,
        tuple(G5_VGPR_SPLITS),
        G5_VGPR_K_TAIL,
    )
    hit = _R2_PLAN_CACHE.get(key)
    if hit is not None:
        return list(hit)
    from ...instances.common.gemm_universal import DataSpec

    dt = str(getattr(req, "dtype", "fp16"))
    dt = dt if dt in ("fp16", "bf16") else "fp16"
    out = []
    for cost, geo, s in g5_vgpr_plans(
        req.M, req.N, req.K, req.arch, any_k_tail=lay == "CRR" and R2_VP_MN_TN_ANY_K
    ):
        tile, trait = _g5_plan_tile_trait(req, geo, s)
        spec = UniversalGemmSpec(
            name="r2_probe",
            tile=tile,
            trait=trait,
            data=DataSpec(dtype_a=dt, dtype_b=dt, dtype_c=dt, layout="RCR"),
            wave_size=64,
        )
        f = r2_vp_mn_form(req, spec)
        if f is None:
            continue
        out.append((r2_native_cost(req, geo, s, f, cost), geo, s))
    out.sort()
    if len(_R2_PLAN_CACHE) > 8192:
        _R2_PLAN_CACHE.clear()
    _R2_PLAN_CACHE[key] = tuple(out)
    return out

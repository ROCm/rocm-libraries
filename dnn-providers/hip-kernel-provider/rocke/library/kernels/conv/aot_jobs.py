# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The implicit-GEMM convolution AOT grid: which kernels the cache builds, and
how one of them is built.

A :class:`BuildJob` is one kernel: a :class:`KernelIdentity` plus the spec
kwargs and capability axes that produce it. The generators below walk the
grid (``CACHE_*``) and yield jobs; :func:`job_for_identity` inverts them for one
identity and :func:`kernel_for_job` turns a job into kernel IR. Nothing here
compiles, caches or launches -- that is :mod:`benchmarks.common.kernel_sweep`
-- so the dispatcher (:mod:`dispatch.grouped_convolution`) can build any
registered instance without reaching above its layer.

What the implicit-GEMM cache sweeps
-----------------------------------
One binary per point of the grid below that passes its direction's spec
validator (``--compile-all`` prints the per-direction breakdown of what
survived). Values are the ``CACHE_*`` constants of this module; the dtype is
fp16 or bf16 (one cache pass per dtype, A/B/D all of that dtype).

Base tile geometry, every direction (wgrad adds to it, see below)::

    tile_m, tile_n   16, 32, 64, 128, 256
    tile_k           16, 32, 64
    warp_m, warp_n   1, 2, 4, 8   (warp_m * warp_tile must divide tile_m, same for n)
    warp_tile_m/n    16, 32       (square; warp_tile_k = the widest MMA atom
                                   for the dtype, which must divide tile_k)
    accumulator      (tile_m/warp_m) * (tile_n/warp_n) / wave_size <= 256
                     f32 registers per lane (CACHE_MAX_ACC_REGS; above that
                     the kernel spills and is the slowest to compile)
    epilogue         default, cshuffle
    pipeline         mem, compv3, compv4, wavelet (wavelet: gfx1250 WMMA only;
                     "basic" is never cached -- it emits mem's code)

Per direction, on top of that:

* **fwd** -- vector widths: a = b in {1, 2, 4, 8} (X and W run along cpg),
  c in {1, 2, 4, 8} (Y along kpg). K loop: plain, ``unroll_k`` (double-
  buffered 2x loop), and ``async_dma`` (direct-to-LDS; built once per
  geometry, under pipeline "mem", since it ignores the pipeline); on the
  wave64 MFMA pipelines also ``reg_prefetch`` 1 / 2 (register-prefetch loop,
  one / two LDS buffers), ``uniform_k_decode`` (per-tile filter-tap decode
  with the plain loop as its fallback) and the two together; the
  ``uniform_k_decode`` + ``reg_prefetch`` = 1 loop also in the XCD-contiguous
  tile orders of :data:`CACHE_FWD_XCD_REMAP` (``xcd_remap``) and on
  XOR-swizzled unpadded LDS tiles (``lds_xor``, tile_k >= 32). The plain
  and ``unroll_k`` loops are also built with ``prefetch_regs`` (register-
  staged prefetch); the single-tile ones (plain, ``prefetch_regs``) on the
  cshuffle epilogue also as the split-K binary (``split_k`` recorded as 2,
  degree swept at launch), and the 3-D sync-load ones also with
  ``split_addr_k_decode`` (per-tile decode, cpg % tile_k == 0 only). Plus
  group-merged depthwise kernels (``group_merge`` in 4..64, cshuffle, plain
  and ``unroll_k`` loops, load widths derived by the builder, on the tile
  subset the ``CACHE_FWD_GROUP_MERGE_*`` constants allow), served only to
  depthwise problems whose merged GEMM-N fits the tile.
* **wgrad** -- tile_m/tile_n also take 144 and 192, and warp_m/warp_n also
  take 3 and 6 (``CACHE_WGRAD_EXTRA_*``): wgrad's GEMM N is Z*Y*X*cpg, which
  carries a factor of 9 for a 3x3 filter, and the base ladder steps by 2x, so
  it has nothing to offer between 128 and 256. Vector widths: a in {1, 2, 4, 8} (dY along kpg), b = c in
  {1, 2, 4, 8} (X and dW along cpg); with the default epilogue (all
  two-stage kernels) c = 1 and b in {1, 2, 4, 8}. pipeline "wavelet" is not cached (wgrad
  builds it as mem). Both the non-split kernel (``split_k`` = 1) and the
  split-K kernel (``split_k`` > 1, recorded as 2; the degree is a kernarg and
  the benchmark sweeps it at launch) are cached, the latter as atomic or
  two-stage (f32 scratch + Stage-2 reduce, ``ws_replicas`` = 8). Plus
  group-merged depthwise kernels (``group_merge`` in 2..64, split-K
  two-stage only, load widths derived by the builder), served only to
  depthwise problems whose merged GEMM fits the tile. ``lds_k_outer`` and
  ``prefetch_addr`` follow ``WgradConvSpec.default_lds_k_outer`` /
  ``default_prefetch_addr`` per geometry.
  Opt-in (``streamk_modes``; ``--compile-all --streamk``): stream-K
  kernels, default epilogue only, ``split_k`` = 1, c = 1. At the widest
  vector width W of the dtype (a = W; b in {W, 1}): one per launch mode
  (``dp_sk``, ``persistent``) x reduction (``linear``, ``workspace`` with
  ``ws_replicas`` = 8, and ``atomic`` where dW is fp32). Fully scalar
  (a = b = 1), on tiles with tile_n >= 128 only: ``dp_sk`` x (``tree``,
  ``workspace``), for large-filter depthwise layers. The CTA pool is a
  kernarg, so it is not swept here.
* **dgrad** -- vector widths: a in {1, 2, 4, 8} (dY along kpg), b = c in
  {1, 2, 4, 8} (W and dX along cpg); with the default epilogue c = 1 and b in
  {1, 2, 4, 8}. One tilde binary per configuration serves every stride and
  dilation: the tilde decomposition is computed on the host and passed in
  the sub-GEMM record buffer, up to 1024 sub-GEMMs (stride 32x32). Next to
  it, on wave64 MFMA targets outside the wavelet pipeline, a stride-1 /
  dilation-1 binary of the same configuration (``CACHE_DGRAD_STRIDE1``),
  which rebuilds its single sub-GEMM record from the kernargs instead of
  searching and loading the buffer, stores through the unit-stride
  epilogue, and runs the tap-outer K loop (``tap_outer_k``, in place of
  ``uniform_k_decode``; served to problems whose kpg is a multiple of
  tile_k). A stride-1 binary with the flat K loop is not built: it serves
  the same problems as the tilde binary of its configuration with the same
  K loop. On gfx950 a capped sub-grid adds the dY halo mode of the tap-outer
  loop (``dy_halo`` = 2): the three halo tiles (``CACHE_DGRAD_HALO_TILES``)
  x square filters 3/5/7 (baked, ``filter_h/w``) x the widest output row
  served (``halo_max_wo`` in ``CACHE_DGRAD_HALO_MAX_WO``) x the
  ``dy_halo_setprio`` / ``dy_halo_kouter_pad`` pairs of the tile
  (``_dgrad_halo_knobs``), K-outer only (b = c in {2, 4, 8}, a = 8). The B
  tile takes the layout the JIT sweep
  picks (``DgradConvSpec.default_lds_k_outer``): K-outer wherever the arch,
  dtype, atom and pipeline allow it and the B load is wider than one element,
  M-outer otherwise. ``uniform_k_decode`` follows
  ``DgradConvSpec.default_uniform_k_decode``.

Every direction builds the grouped kernel only: it serves groups == 1 as well
(``CACHE_GROUPED``), so there is no separate ungrouped binary.

Not swept (spec defaults): fwd ``lds_k_outer`` (off), wgrad ``async_dma``,
``chiplet_swizzle``, ``lds_layout``, ``waves_per_eu``, accumulator epilogues,
``cshuffle_no_alias``, the pointwise fast path (1x1 problems run the general
kernels) and 3-D convolution. The direct-conv cache sweeps its own grid: see
:mod:`benchmarks.common.direct_kernel_sweep`.

Two more are not swept but are not left at the spec default either -- both are
pure functions of the geometry, so they are derived rather than searched.
wgrad ``lds_k_outer`` comes from
:meth:`~kernels.common.conv_implicit_gemm_wgrad.WgradConvSpec.default_lds_k_outer`,
and ``lds_k_pad`` from
:meth:`~kernels.common.conv_implicit_gemm_wgrad.WgradConvSpec.default_lds_k_pad`,
which supplies the bank-conflict-free row stride for the M-outer tile and
stands aside (returning ``None``) wherever ``default_lds_k_outer`` chose the
K-outer one, since that layout takes its stride from the builder.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from kernels.conv.kernel_identity import KernelIdentity, current_llvm_flavor


# ---------------------------------------------------------------------------
# Implicit-GEMM AOT configuration grid
#
# These axes match the JIT sweep in benchmark_implicit_gemm_conv.py (minus the
# gfx1250-only extremes). An AOT cache is compiled once, so every axis value
# that can produce the fastest kernel on the target must be present --
# a value omitted here is permanently missing from --run-from-cache regardless
# of what the benchmark found.
#
# GFX1250-only extensions (tile=512, warp=16) are omitted: they require a
# minimum register file that gfx950 and gfx942 do not have, so those kernels
# fail spec validation and are silently skipped on those targets. The grid
# below does build for gfx1250 (WMMA: "mem" and "wavelet" pipelines, group-
# merged depthwise kernels included); only those extensions are left out.
# ---------------------------------------------------------------------------

CACHE_TILE_MN: Tuple[int, ...] = (16, 32, 64, 128, 256)
CACHE_TILE_K: Tuple[int, ...] = (16, 32, 64)
CACHE_WARP_MN: Tuple[int, ...] = (1, 2, 4, 8)
CACHE_WARP_TILE_MN: Tuple[int, ...] = (16, 32)

# ---- wgrad-only geometry extension ----------------------------------------
# wgrad is the one direction whose GEMM dims are filter-shaped rather than
# tensor-shaped: N = Z*Y*X*cpg and M = kpg. For a 3x3 filter N carries a
# factor of 9, which the power-of-two ladder has no way to represent, and the
# ladder's steps are 2x apart -- so between 128 and 256 it offers nothing at
# all. 144 and 192 fill that gap with the multiples of 16 that a 9*cpg N
# actually lands on.
#
# Both failure modes this fixes were measured, and they are not the same one:
#
#  * cpg = 48 (N = 432): no power of two divides it, so every base-grid
#    kernel runs a partial tile on the N edge and the waves on it do
#    full-tile work for a fraction of a tile of output.
#  * cpg = 128 (N = 1152): 128 divides it exactly and the edge is clean, yet
#    192 still wins -- 6 fatter tiles beat 9 thinner ones on grid shape and
#    per-workgroup arithmetic intensity, which divisibility says nothing
#    about.
#
# So do not read these as "edge tiles for indivisible shapes". The ladder is
# simply too coarse for this direction, and clean division is one of several
# reasons a 1.5x step can win.
#
# Warps 3 and 6 come with them rather than separately: a 144 tile is only
# reachable with warp_m in (1, 3) at warp_tile 16 (144 % (2*16) != 0), so the
# odd splits are what makes the new edges usable at all. Adding the warps
# without the tiles changes nothing -- no power-of-two tile divides by 3.
#
# fwd and dgrad keep the base grid: their GEMM dims come from N*Ho*Wo and C,
# which have no factor of 9 to chase, and widening all three directions costs
# several times the disk for no measured gain.
CACHE_WGRAD_EXTRA_TILE_MN: Tuple[int, ...] = (144, 192)
CACHE_WGRAD_EXTRA_WARP_MN: Tuple[int, ...] = (3, 6)
# Cap on the f32 accumulator registers each lane holds:
# (tile_m / warp_m) * (tile_n / warp_n) / wave_size. Above 256 the
# accumulators no longer fit the 256 AGPRs MFMA accumulates into, so the
# kernel spills -- it is not a contender, and those are the slowest kernels in
# the grid to compile (several seconds each, against a fraction of a second
# for the rest). Not a validity rule: the JIT sweep still builds them.
CACHE_MAX_ACC_REGS = 256
CACHE_VECS: Tuple[int, ...] = (1, 2, 4, 8)
CACHE_PIPELINES: Tuple[str, ...] = ("mem", "compv3", "compv4", "wavelet", "basic")
CACHE_EPILOGUES: Tuple[str, ...] = ("default", "cshuffle")
# Forward split-K (ImplicitGemmConvSpec.split_k): 1 is the classic kernel, 2
# marks the split binary (the degree is a kernarg, swept at launch). Built on
# the cshuffle epilogue and the single-tile K loops only (plain and
# prefetch_regs), which is what the split binary supports.
CACHE_FWD_SPLIT_KS: Tuple[int, ...] = (1, 2)
# Forward per-tile reduction decode of the split-addressing path
# (ImplicitGemmConvSpec.split_addr_k_decode), 3-D sync-load binaries only.
CACHE_FWD_SPLIT_ADDR_K_DECODE: Tuple[bool, ...] = (False, True)
# Forward K-loop register prefetch (ImplicitGemmConvSpec.prefetch_regs): both
# values are built, on the plain and the unroll_k sync drivers, so the cache
# offers the prefetching loop next to the classic one.
CACHE_FWD_PREFETCH_REGS: Tuple[bool, ...] = (False, True)

# Pipeline names that emit exactly the code of another one, so the AOT cache
# skips them (the JIT sweep shares CACHE_PIPELINES and still walks them).
#
# * "basic" in every direction: its K loop is the shared scf.for_iter body of
#   "mem" and its schedule policy is mem's (no hints, no setprio), so the
#   binary differs only in its name.
# * "wavelet" in wgrad: the wgrad builder has no load/math wave split, and the
#   wavelet schedule policy is mem's too. (fwd/dgrad have a real wavelet
#   kernel, gfx1250 WMMA only; their validators reject it elsewhere.)
_AOT_ALIAS_PIPELINES: Dict[str, Tuple[str, ...]] = {
    "fwd": ("basic",),
    "wgrad": ("basic", "wavelet"),
    "dgrad": ("basic",),
}

# ---- capability axes ------------------------------------------------------
# These are NOT tuning knobs: they change which problems a binary can serve.
# Grouped convolution takes a different code path in every direction (the
# contraction index only spans one group). An ungrouped binary gives wrong
# numbers -- not an error -- on a grouped problem, but the grouped binary
# computes groups == 1 correctly too (group index 0, cpg == C), so the cache
# builds the grouped one only and offers it to both; KernelCache keeps
# honouring ungrouped entries of older caches for ungrouped problems only.
#
# Implicit-GEMM dgrad reads its stride and dilation from the tilde sub-GEMM
# records at run time, so one tilde binary (stride/dilation recorded as 0,
# runtime) serves every stride and dilation within max_sub_gemms. A stride-1
# binary is a second capability: at stride 1 the kernel rebuilds its single
# sub-GEMM record from the kernargs instead of searching and loading the
# record buffer (DgradConvSpec.folds_sub_gemm_record), and with dilation 1 it
# also takes the unit-stride epilogue. Its identity records stride 1 and
# dilation 1, so the cache offers it to stride-1, dilation-1 problems only;
# the tilde binary of the same configuration still serves those too. The
# stride-1 capability is built with the tap-outer K loop only (see
# _dgrad_tap_outer_variants).
CACHE_GROUPED: Tuple[bool, ...] = (True,)
# Spatial rank of the cached binaries: False = 2-D (NHWC), True = 3-D (NDHWC).
# The 3-D kernels decode a depth axis, so they are distinct binaries. Only 2-D
# is built by default; ROCKE_AOT_SPATIAL=3d (or 2d,3d) adds the 3-D grid.
import os as _os
import re as _re
CACHE_SPATIAL_3D: Tuple[bool, ...] = tuple(
    {"2d": False, "3d": True}[v.strip()]
    for v in _os.environ.get("ROCKE_AOT_SPATIAL", "2d").split(",")
    if v.strip()
)
# Software pipelining of the stride-1 tap-outer dgrad loop
# (DgradConvSpec.tapk_prefetch) built per tap-outer configuration: 0 = the
# plain loop, 1 = register prefetch, 2 = LDS ping-pong. ROCKE_AOT_DGRAD_TPF
# (e.g. "0,1,2") overrides it.
CACHE_DGRAD_TAPK_PREFETCH: Tuple[int, ...] = tuple(
    int(v) for v in _os.environ.get("ROCKE_AOT_DGRAD_TPF", "0").split(",") if v.strip()
)
# The x-halo mode of the stride-1 tap-outer dgrad loop
# (DgradConvSpec.tapk_xhalo; filter width baked, pW == 0 served) next to the
# plain tap-outer binary, one per tapk_prefetch value listed in
# ROCKE_AOT_DGRAD_XH (e.g. "0,1,3"; unset = none). Built for the probe's
# 3-wide filter.
CACHE_DGRAD_XHALO_TPF: Tuple[int, ...] = tuple(
    int(v) for v in _os.environ.get("ROCKE_AOT_DGRAD_XH", "").split(",") if v.strip()
)
# k values per B (W) LDS write of the stride-1 tap-outer dgrad binaries
# (DgradConvSpec.b_kpack; 0 = the element-wise transpose-on-store), built per
# tap-outer variant where the tile admits it. ROCKE_AOT_DGRAD_BKP (e.g.
# "0,2,4") overrides it.
CACHE_DGRAD_B_KPACK: Tuple[int, ...] = tuple(
    int(v) for v in _os.environ.get("ROCKE_AOT_DGRAD_BKP", "0").split(",") if v.strip()
)
# Knob sets of the x-halo dgrad binaries (tapk_xhalo), each built per x-halo
# variant it applies to: ROCKE_AOT_DGRAD_XHK is a ";"-separated list of
# ","-separated key=value sets over ilv (DgradConvSpec.tapk_ilv), lda
# (tapk_lds_ahead), lsb (tapk_lds_barrier), sp (tapk_setprio) -- those four
# on tpf 1/3 only --, xhx (xhalo_exact), xcd (xcd_remap), kpad (lds_k_pad),
# lxor (lds_xor, with lds_k_pad 0), mch (cshuffle_m_chunks; cshuffle only),
# wpe (waves_per_eu), sk (split_k; cshuffle tpf 1/3 only, serves
# groups == 1), bw (xh_b_wide), bpad (xh_b_kpad), ak4 (xh_a_k4, with lxor),
# xo (xh_order bits) and xdf (xh_defer; xo and xdf on tpf 1/3 only), e.g. "ilv=4,xhx=1,xcd=6;ilv=8,xhx=1,xcd=6". An item
# "@<regex>" gives its set its own scope (see ROCKE_AOT_DGRAD_XHK_SCOPE). An
# empty set is the plain binary; unset = that alone.
def _parse_dgrad_xh_knobs(text: str) -> Tuple[Tuple[Tuple[str, int], ...], ...]:
    keys = {"ilv", "xhx", "xcd", "kpad", "mch", "sk", "lda", "wpe", "lsb", "sp", "lxor", "bw", "bpad", "ak4", "xo", "xdf", "xfa", "xep", "xhp", "mb"}
    out = []
    for part in text.split(";"):
        kv = []
        for item in part.split(","):
            item = item.strip()
            if not item:
                continue
            if item.startswith("@"):
                # This set's own scope (overrides ROCKE_AOT_DGRAD_XHK_SCOPE).
                kv.append(("@", item[1:]))
                continue
            k, _, v = item.partition("=")
            if k not in keys:
                raise ValueError(f"ROCKE_AOT_DGRAD_XHK: unknown knob {k!r}")
            kv.append((k, int(v)))
        out.append(tuple(sorted(kv)))
    # The plain binary is always built; the sets add to it.
    return tuple(dict.fromkeys([(), *out]))


CACHE_DGRAD_XHALO_KNOBS = _parse_dgrad_xh_knobs(
    _os.environ.get("ROCKE_AOT_DGRAD_XHK", "")
)
# The x-halo binaries the non-empty knob sets apply to: a regular expression
# searched in the plain binary's label (ROCKE_AOT_DGRAD_XHK_SCOPE; unset =
# every x-halo binary), so a cache can carry the knobs where they pay only.
CACHE_DGRAD_XHALO_KNOB_SCOPE = _os.environ.get("ROCKE_AOT_DGRAD_XHK_SCOPE", "")


def _dgrad_xh_knob_cfg(knobs, *, tpf: int, grouped: bool, epilogue: str):
    """The spec kwargs of one CACHE_DGRAD_XHALO_KNOBS set on an x-halo
    variant, or None where the set does not apply."""
    d = dict(knobs)
    d.pop("@", None)
    cfg = {}
    if d.get("ilv"):
        if tpf not in (1, 3):
            return None
        cfg["tapk_ilv"] = d["ilv"]
    if d.get("lda"):
        if tpf not in (1, 3):
            return None
        cfg["tapk_lds_ahead"] = d["lda"]
    if d.get("lsb"):
        if tpf not in (1, 3):
            return None
        cfg["tapk_lds_barrier"] = True
    if d.get("bpad"):
        cfg["xh_b_kpad"] = d["bpad"]
    if d.get("ak4"):
        cfg["xh_a_k4"] = True
    if d.get("xo"):
        if tpf not in (1, 3):
            return None
        cfg["xh_order"] = d["xo"]
    if d.get("xdf"):
        if tpf not in (1, 3):
            return None
        cfg["xh_defer"] = d["xdf"]
    if d.get("xfa"):
        cfg["xh_fast_a"] = d["xfa"]
    if d.get("xhp"):
        cfg["xh_persist"] = d["xhp"]
    if d.get("xep"):
        if epilogue != "cshuffle":
            return None
        cfg["xh_epi"] = d["xep"]
    if d.get("bw"):
        if tpf not in (1, 3):
            return None
        cfg["xh_b_wide"] = True
    if d.get("lxor"):
        # Swizzled instead of padded rows.
        cfg["lds_xor"] = True
        cfg["lds_k_pad"] = 0
    if d.get("sp"):
        if tpf not in (1, 3):
            return None
        cfg["tapk_setprio"] = d["sp"]
    if d.get("xhx"):
        cfg["xhalo_exact"] = True
    if d.get("xcd"):
        cfg["xcd_remap"] = d["xcd"]
    if d.get("mb"):
        cfg["m_band"] = d["mb"]
    if d.get("kpad"):
        cfg["lds_k_pad"] = d["kpad"]
    if d.get("wpe"):
        cfg["waves_per_eu"] = d["wpe"]
    if d.get("mch", 1) > 1:
        if epilogue != "cshuffle":
            return None
        cfg["cshuffle_m_chunks"] = d["mch"]
    if d.get("sk", 1) > 1:
        if epilogue != "cshuffle" or tpf not in (1, 3):
            return None
        cfg["split_k"] = d["sk"]
    return cfg


# Knob sets of the plain (not x-halo) tapk_prefetch=1 dgrad binaries:
# ROCKE_AOT_DGRAD_PPK, same syntax as ROCKE_AOT_DGRAD_XHK, keys bpp
# (DgradConvSpec.tapk_pingpong), pps (tapk_pp_sched), xcd, kpad, lxor, mch, wpe; every set needs
# an "@<regex>" scope on the plain binary's label (unscoped sets are
# ignored, so the plain grid never grows by accident). Unset = none.
def _parse_dgrad_pp_knobs(text: str):
    keys = {"bpp", "pps", "ppe", "ppp", "ppx", "zt", "pbs", "xcd", "kpad", "lxor", "mch", "wpe", "sk", "sapl", "mb"}
    out = []
    for part in text.split(";"):
        kv = []
        for item in part.split(","):
            item = item.strip()
            if not item:
                continue
            if item.startswith("@"):
                kv.append(("@", item[1:]))
                continue
            k, _, v = item.partition("=")
            if k not in keys:
                raise ValueError(f"ROCKE_AOT_DGRAD_PPK: unknown knob {k!r}")
            kv.append((k, int(v)))
        if kv and dict(kv).get("@"):
            out.append(tuple(sorted(kv)))
    return tuple(dict.fromkeys([(), *out]))


CACHE_DGRAD_PLAIN_KNOBS = _parse_dgrad_pp_knobs(
    _os.environ.get("ROCKE_AOT_DGRAD_PPK", "")
)
# Extra dgrad tile extents (e.g. "512") built ONLY as ROCKE_AOT_DGRAD_PPK
# variants of the plain tpf1 tap-outer binary (no base binary, no other loop):
# ROCKE_AOT_DGRAD_PP_TILE_MN. Unset = none.
CACHE_DGRAD_PP_EXTRA_TILE_MN: Tuple[int, ...] = tuple(
    int(v) for v in _os.environ.get("ROCKE_AOT_DGRAD_PP_TILE_MN", "").split(",") if v.strip()
)


def _dgrad_pp_knob_cfg(knobs, *, epilogue: str):
    """The spec kwargs of one CACHE_DGRAD_PLAIN_KNOBS set, or None."""
    d = dict(knobs)
    d.pop("@", None)
    cfg = {}
    if d.get("bpp"):
        cfg["tapk_pingpong"] = d["bpp"]
    if d.get("pps"):
        cfg["tapk_pp_sched"] = d["pps"]
    if d.get("ppe"):
        cfg["tapk_pp_epi"] = d["ppe"]
    if d.get("ppp"):
        cfg["tapk_pp_persist"] = d["ppp"]
    if d.get("ppx"):
        cfg["tapk_pp_xhalo"] = d["ppx"]
    if d.get("zt"):
        cfg["tapk_ztrim"] = d["zt"]
    if d.get("pbs"):
        cfg["tapk_pp_bslab"] = d["pbs"]
    if d.get("lxor"):
        cfg["lds_xor"] = True
        cfg["lds_k_pad"] = 0
    if d.get("kpad"):
        cfg["lds_k_pad"] = d["kpad"]
    if d.get("xcd"):
        cfg["xcd_remap"] = d["xcd"]
    if d.get("mb"):
        cfg["m_band"] = d["mb"]
    if d.get("wpe"):
        cfg["waves_per_eu"] = d["wpe"]
    if d.get("mch", 1) > 1:
        if epilogue != "cshuffle":
            return None
        cfg["cshuffle_m_chunks"] = d["mch"]
    if d.get("sk", 1) > 1:
        if epilogue != "cshuffle" or not d.get("bpp"):
            return None
        cfg["split_k"] = d["sk"]
        if d.get("sapl"):
            # lane-contiguous split-K atomic pairs (DgradConvSpec.split_atomic_pairs)
            cfg["split_atomic_pairs"] = True
    return cfg


# Whether the stride-1 dgrad binaries are built next to the tilde ones.
CACHE_DGRAD_STRIDE1: tuple[bool, ...] = (False, True)
# dY halo binaries (DgradConvSpec.dy_halo): a capped sub-grid, not the
# geometry product. The tiles are the halo tiles of the gfx950 dgrad pick, as
# (tile_m, tile_n, tile_k, warp_m, warp_n, warp tile): 64x64 on 2x2 waves and
# 128x64 / 256x64 on 4x1 waves, 32x32 atoms. The filter is baked (the taps
# are unrolled), square 3, 5 and 7 (stride 1 and Ho == Hi fix the pad). The
# staged halo is tile_m + (Y-1)*halo_max_wo + (X-1) LDS rows, so a binary
# serves Wo <= halo_max_wo and a wider bucket costs LDS on a narrower image;
# that costs time only where the extra rows cross an LDS occupancy step. Powers
# of two keep the over-size under 2x, and 48 / 96 bound it at 1.5x between
# the wider ones, where the steps lie closest together. The loads are K-outer
# (B = D width > 1); A loads 8 wide, which every halo problem admits (kpg is a
# multiple of tile_k). dy_halo = 2 (the double-buffered B tile) only.
CACHE_DGRAD_HALO_TILES: tuple[tuple[int, ...], ...] = (
    (64, 64, 64, 2, 2, 32),
    (128, 64, 64, 4, 1, 32),
    (256, 64, 64, 4, 1, 32),
)
CACHE_DGRAD_HALO_FILTERS: tuple[int, ...] = (3, 5, 7)
CACHE_DGRAD_HALO_MAX_WO: tuple[int, ...] = (8, 16, 32, 48, 64, 96, 128)
CACHE_DGRAD_HALO_VEC_BC: tuple[int, ...] = (2, 4, 8)
CACHE_DGRAD_HALO_VEC_A = 8
CACHE_DGRAD_HALO_MODES: tuple[int, ...] = (2,)
# Sub-GEMM records an implicit-GEMM dgrad binary dispatches over (its tilde
# decomposition has stride_h * stride_w of them at most): 32x32, so the
# patch-embedding strides 14 and 16 of vision transformers fit. The CTA's
# search runs only as many steps as a problem's records need, so the bound
# costs the common stride-1 / stride-2 problems nothing.
CACHE_DGRAD_MAX_SUB_GEMMS = 1024
# The patch-embedding dgrad (DgradConvSpec.patch): one GEMM over the filter's
# (y, x, c) for stride >= filter and no padding. Built on these ungrouped
# cshuffle geometries only -- (tile_m, tile_n, tile_k, warp_m, warp_n, warp
# tile, pipeline), the ones that led a sweep of ViT/SigLIP/ConvNeXt stems --
# at every B/D width, since the X*C run of a stem is often odd or 2 mod 4.
CACHE_DGRAD_PATCH_GEOMETRIES: frozenset = frozenset(
    (tm, tn, tk, wm, wn, wt, pipe)
    for (tm, tn, tk, wm, wn, wt), pipes in (
        ((128, 256, 64, 1, 4, 16), ("compv3", "compv4")),
        ((128, 256, 64, 2, 4, 16), ("mem", "compv3", "compv4")),
        ((128, 256, 64, 2, 2, 16), ("compv3", "compv4")),
        ((256, 128, 64, 2, 2, 16), ("compv3", "compv4")),
        ((256, 128, 64, 4, 1, 16), ("compv3", "compv4")),
        ((256, 128, 32, 4, 2, 16), ("mem", "compv3")),
        ((128, 128, 64, 2, 2, 16), ("mem", "compv3", "compv4")),
        ((128, 128, 64, 2, 4, 16), ("mem",)),
        ((128, 128, 64, 1, 4, 32), ("mem",)),
        ((128, 128, 64, 2, 2, 32), ("mem", "compv3", "compv4")),
        ((128, 64, 64, 4, 1, 32), ("mem", "compv3", "compv4")),
        ((128, 64, 64, 4, 2, 32), ("compv4",)),
        ((64, 64, 64, 2, 2, 16), ("mem", "compv3", "compv4")),
    )
    for pipe in pipes
)
# wgrad group merging (WgradConvSpec.group_merge): Gm depthwise groups share
# one workgroup. The degree is baked in and the binary serves depthwise
# problems only (groups divisible by Gm, Y*X*Gm within the tile). Mirrors
# _GROUP_MERGE_DEGREES of the wgrad instance.
CACHE_WGRAD_GROUP_MERGES: Tuple[int, ...] = (2, 4, 8, 16, 32, 64)
# fwd group merging (ImplicitGemmConvSpec.group_merge): the same idea on the
# forward GEMM -- Gm depthwise groups fold onto GEMM-N and GEMM-K. Baked in;
# the binary serves depthwise problems with groups divisible by Gm and
# Gm <= tile_n. _GROUP_MERGE_DEGREES of the fwd instance minus Gm = 2, which
# was never the best merged degree on the depthwise shapes the merged path was
# tuned on. Depthwise problems whose group count is even but not divisible by
# 4 therefore get no merged binary and run the unmerged kernels.
CACHE_FWD_GROUP_MERGES: Tuple[int, ...] = (4, 8, 16, 32, 64)
# The merged fwd kernels are built on a subset of the shared tile geometry.
# The merged GEMM-N is exactly Gm <= 64, so a wider tile only adds idle
# N-lanes. tile_k = 16, warp_n = 4 and tile_m = 16 were never needed for the
# best merged kernel on the depthwise shapes the merged path was tuned on.
# Together these drop most of the merged binaries; unroll_k stays whole.
CACHE_FWD_GROUP_MERGE_MAX_TILE_N = 64
CACHE_FWD_GROUP_MERGE_EXCLUDED_TILE_K: Tuple[int, ...] = (16,)
CACHE_FWD_GROUP_MERGE_EXCLUDED_WARP_N: Tuple[int, ...] = (4,)
CACHE_FWD_GROUP_MERGE_EXCLUDED_TILE_M: Tuple[int, ...] = (16,)

# ---- wgrad split-K axis ---------------------------------------------------
# The split degree is never compiled in: it is passed as kernargs
# (``ks_count`` / ``ks``), so the one split-K binary -- any spec split_k > 1
# builds the same kernel -- handles every degree > 1 at launch time. The cache
# records it as split_k=2 ("splits"); the benchmark sweeps the real degrees.
#
# split_k=1 (direct store, no atomics) is a structurally different kernel
# (different epilogue, dW is writeonly, vec_c can be > 1), so it is a separate
# set of binaries. The JIT sweep builds it for every non-depthwise problem and
# it wins where the grid already fills the device without splitting K, so the
# cache carries it too. Group-merged (depthwise) kernels skip it, as the JIT
# sweep does: one workgroup per merged group cannot fill the device.
CACHE_WGRAD_SPLIT_KS: Tuple[int, ...] = (1, 2)

# ---- wgrad K-loop variants (per arch) ----------------------------------------
# Built next to every plain (synchronous, M-outer) wgrad kernel, as extra
# binaries the benchmark ranks like any other: WgradConvSpec.prefetch_regs
# (register-prefetch K loop), lds_kpack (packed transpose-on-store) and
# sep_addr=2 (operand addresses carried across the K loop). Together they are
# the gfx942 answer to what lds_k_outer / async_dma do on gfx950, which need
# the LDS transpose read gfx942 lacks: kp2 replaces the bank-conflicting
# ds_write_b16 transpose-on-store, the prefetch overlaps a workgroup's own
# global reads with its MFMAs, and the carried addressing takes the address
# decode (magic division, stride multiplies) out of the loop.
CACHE_WGRAD_KLOOP_VARIANTS: Dict[str, Tuple[dict, ...]] = {
    "gfx942": ({"prefetch_regs": 1, "lds_kpack": 2, "sep_addr": 2},),
}
# Unsplit kernels also get the two-tiles-ahead prefetch (prefetch_regs=3):
# their K loop runs the whole reduction, long enough for a second tile in
# flight to pay for its registers. The split-K slices are short, and there
# the deeper queue only adds VGPRs.
CACHE_WGRAD_KLOOP_UNSPLIT_VARIANTS: Dict[str, Tuple[dict, ...]] = {
    "gfx942": ({"prefetch_regs": 3, "lds_kpack": 2, "sep_addr": 2},),
}
# ROCKE_AOT_WGRAD_XCD=<n> additionally builds each split-K variant with
# WgradConvSpec.xcd_remap=n (the whole-grid XCD remap), for a cache scoped to
# a device with n XCDs. The XCD count is a property of the part, not of the
# arch (gfx942 ships with 6 and with 8), so it is opt-in per cache build.
_WGRAD_XCD = int(_os.environ.get("ROCKE_AOT_WGRAD_XCD", "0") or 0)
# ROCKE_AOT_WGRAD_PFS=<list> (e.g. "1,4") additionally builds each
# register-prefetch variant (of a split-K kernel the xcd ones only, when
# ROCKE_AOT_WGRAD_XCD is set) with WgradConvSpec.pfr_sched=<v>: 1 = early issue of the next tile's
# reads, N > 1 = reads interleaved with N MFMAs each (mem pipeline only).
# Which one wins is per kernel and per shape, so it is opt-in and swept.
_WGRAD_PFS: Tuple[int, ...] = tuple(
    int(v) for v in _os.environ.get("ROCKE_AOT_WGRAD_PFS", "").split(",") if v.strip()
)
# ROCKE_AOT_WGRAD_PFS_MIN_MN=<v> limits the pfr_sched variants to tile_m >= v,
# tile_n >= v and tile_k >= 32 (where they were measured to win: the large
# 1-2 wave/SIMD tiles; the M=64 tiles gain nothing from them).
_WGRAD_PFS_MIN_MN = int(_os.environ.get("ROCKE_AOT_WGRAD_PFS_MIN_MN", "0") or 0)
# ROCKE_AOT_WGRAD_BSWZ=1 additionally builds each register-prefetch variant
# above (pfr_sched ones included) with WgradConvSpec.lds_bswz=1 (pad-free
# XOR-swizzled B tile; the validator rejects tiles it does not cover).
# ROCKE_AOT_WGRAD_BSWZ_MAX_M=<m> limits it to tile_m <= m: the pad it saves is
# what moves the M=64 tiles over an LDS occupancy step.
# ROCKE_AOT_WGRAD_KPL=8 doubles every k-packed register-prefetch variant above
# (pfr_sched ones included) with WgradConvSpec.kp_line=8 (line-major decode of
# the packed global reads); applied before the lds_bswz axis, so it stacks.
_WGRAD_KPL = int(_os.environ.get("ROCKE_AOT_WGRAD_KPL", "0") or 0)
# ROCKE_AOT_WGRAD_KPL_MAX_M=<m> limits it to tile_m <= m (measured: it helps the
# M=64, C=64 wgrad tiles; the 128/256-row tiles are unchanged).
_WGRAD_KPL_MAX_M = int(_os.environ.get("ROCKE_AOT_WGRAD_KPL_MAX_M", "0") or 0)
# ROCKE_AOT_WGRAD_M64_MIN_N=<n> further limits the kp_line and lds_bswz axes
# to tile_n >= n (the M=64 problems are served by 64 x 256 tiles).
_WGRAD_M64_MIN_N = int(_os.environ.get("ROCKE_AOT_WGRAD_M64_MIN_N", "0") or 0)
# ROCKE_AOT_WGRAD_M64_MIN_M=<m>: and tile_m >= m; ROCKE_AOT_WGRAD_M64_SPLIT_ONLY=1:
# and split-K binaries only (the M=64 problems have huge reductions).
_WGRAD_M64_MIN_M = int(_os.environ.get("ROCKE_AOT_WGRAD_M64_MIN_M", "0") or 0)
_WGRAD_M64_SPLIT_ONLY = int(_os.environ.get("ROCKE_AOT_WGRAD_M64_SPLIT_ONLY", "0") or 0)
_WGRAD_BSWZ = int(_os.environ.get("ROCKE_AOT_WGRAD_BSWZ", "0") or 0)
_WGRAD_BSWZ_MAX_M = int(_os.environ.get("ROCKE_AOT_WGRAD_BSWZ_MAX_M", "0") or 0)
# ROCKE_AOT_WGRAD_BSWZ_WPE=<n> also builds each tile_k=32 lds_bswz variant with
# waves_per_eu=<n>: the pad-free tile fits 3 workgroups per CU by LDS, and the
# 16x16 kernels need the VGPR cap (<= 168) to fit them by registers too.
_WGRAD_BSWZ_WPE = int(_os.environ.get("ROCKE_AOT_WGRAD_BSWZ_WPE", "0") or 0)
# ROCKE_AOT_WGRAD_XWIN=<X> (the filter width, 3) builds an X-window twin
# (WgradConvSpec.xwin=X, sep_addr=0) of every k-packed prefetch_regs=1 split-K
# two-stage variant above (pfr_sched ones included; xcd_remap ones only when
# ROCKE_AOT_WGRAD_XCD is set) of the tiles whose tile_n is X channel blocks.
# ROCKE_AOT_WGRAD_XWIN_TK=<k>: only tile_k == k (measured: 32 wins, 16 and 64
# lose). ROCKE_AOT_WGRAD_XWIN_N384_MIN_M=<m>: also a tile_n=384 twin of the
# tile_n=256 jobs with tile_m >= m (the 256-row tiles of the C >= 256
# problems). ROCKE_AOT_WGRAD_XWIN_WPE=<n>[,<n>...] with
# ROCKE_AOT_WGRAD_XWIN_WPE_MAX_M=<m>: also a waves_per_eu=<n> copy of the twins
# with tile_m <= m (the 16x16x16 twins of the 64/128-row tiles need the VGPR
# cap for 2-3 waves/SIMD).
_WGRAD_XWIN = int(_os.environ.get("ROCKE_AOT_WGRAD_XWIN", "0") or 0)
_WGRAD_XWIN_TK = int(_os.environ.get("ROCKE_AOT_WGRAD_XWIN_TK", "0") or 0)
_WGRAD_XWIN_N384_MIN_M = int(_os.environ.get("ROCKE_AOT_WGRAD_XWIN_N384_MIN_M", "0") or 0)
_WGRAD_XWIN_WPE: Tuple[int, ...] = tuple(
    int(v) for v in (_os.environ.get("ROCKE_AOT_WGRAD_XWIN_WPE", "") or "").split(",") if v
)
_WGRAD_XWIN_WPE_MAX_M = int(_os.environ.get("ROCKE_AOT_WGRAD_XWIN_WPE_MAX_M", "0") or 0)
# ROCKE_AOT_WGRAD_XWIN_M=<m>[,<m>...]: only these tile_m;
# ROCKE_AOT_WGRAD_XWIN_CB=<c>: only channel blocks (tile_n / X) that are
# multiples of c (64: the blocks that divide the power-of-two channel counts);
# ROCKE_AOT_WGRAD_XWIN_N384_WT=<w>: the tile_n=384 twins only with
# warp_tile_m == w (32: the 16x16 ones spill at 256 rows).
_WGRAD_XWIN_M: Tuple[int, ...] = tuple(
    int(v) for v in (_os.environ.get("ROCKE_AOT_WGRAD_XWIN_M", "") or "").split(",") if v
)
_WGRAD_XWIN_N384_WT = int(_os.environ.get("ROCKE_AOT_WGRAD_XWIN_N384_WT", "0") or 0)
_WGRAD_XWIN_CB = int(_os.environ.get("ROCKE_AOT_WGRAD_XWIN_CB", "0") or 0)
# ROCKE_AOT_WGRAD_XWIN_WSR=<r>[,<r>...] with ROCKE_AOT_WGRAD_XWIN_WSR_MIN_M=<m>:
# also a ws_replicas=<r> copy of the X-window twins (after the WPE copies)
# with tile_m >= m. Their Stage 1 atomics go to fewer scratch slabs, so the
# per-launch scratch clear and the Stage 2 fold touch r instead of 8
# dW-sized f32 slabs -- what the medium shapes (dW of 0.4-3.5 M elements at
# split 6-48) pay outside Stage 1. Default unset: no extra jobs.
_WGRAD_XWIN_WSR: Tuple[int, ...] = tuple(
    int(v) for v in (_os.environ.get("ROCKE_AOT_WGRAD_XWIN_WSR", "") or "").split(",") if v
)
_WGRAD_XWIN_WSR_MIN_M = int(_os.environ.get("ROCKE_AOT_WGRAD_XWIN_WSR_MIN_M", "0") or 0)
# ROCKE_AOT_WGRAD_XWIN_TK2=<k> with ROCKE_AOT_WGRAD_XWIN_TK2_MIN_M=<m> and
# ROCKE_AOT_WGRAD_XWIN_TK2_WSR=<r>: also the X-window twins at tile_k == k
# (16: one K tile per output row of a Wo=16 problem) with tile_m >= m, only
# as their waves_per_eu copies (ROCKE_AOT_WGRAD_XWIN_WPE) with ws_replicas=r.
_WGRAD_XWIN_TK2 = int(_os.environ.get("ROCKE_AOT_WGRAD_XWIN_TK2", "0") or 0)
_WGRAD_XWIN_TK2_MIN_M = int(_os.environ.get("ROCKE_AOT_WGRAD_XWIN_TK2_MIN_M", "0") or 0)
_WGRAD_XWIN_TK2_WSR = int(_os.environ.get("ROCKE_AOT_WGRAD_XWIN_TK2_WSR", "0") or 0)
# ROCKE_AOT_WGRAD_XWIN_MROW=<n>[,<n>...] with ROCKE_AOT_WGRAD_XWIN_MROW_MIN_M=<m>
# and ROCKE_AOT_WGRAD_XWIN_MROW_WSR=<r>: also xwin_mrow=<n> copies (K tile = n
# output rows of Wo = tile_k / n: the Wo 16 / 8 shapes) of the X-window
# twins with tile_m >= m, only of their waves_per_eu copies, with
# ws_replicas=r. Unset: no jobs.
_WGRAD_XWIN_MROW: Tuple[int, ...] = tuple(
    int(v) for v in (_os.environ.get("ROCKE_AOT_WGRAD_XWIN_MROW", "") or "").split(",") if v
)
_WGRAD_XWIN_MROW_MIN_M = int(_os.environ.get("ROCKE_AOT_WGRAD_XWIN_MROW_MIN_M", "0") or 0)
_WGRAD_XWIN_MROW_WSR = int(_os.environ.get("ROCKE_AOT_WGRAD_XWIN_MROW_WSR", "0") or 0)
# ROCKE_AOT_WGRAD_XWIN_PP=<n>[,<n>...]: also an 8-wave block ping-pong copy
# (xwin_groups=8/waves, xwin_pp=<n>, waves_per_eu=2) of the 2-wave 64-row
# (w1x2), 4-wave 128-row and 8-wave 256-row (w4x2, one tile) xwin twins on
# 16x16 atoms (plain "mem" pipeline, no pfr_sched). Launch degrees must be
# multiples of xwin_groups (4 / 2 / 1).
# Unset: no jobs.
_WGRAD_XWIN_PP: Tuple[int, ...] = tuple(
    int(v) for v in (_os.environ.get("ROCKE_AOT_WGRAD_XWIN_PP", "") or "").split(",") if v
)

# ---- wgrad stream-K axis (opt-in) -------------------------------------------
# Stream-K (WgradConvSpec.streamk) bakes its launch mode and its partial-sum
# reduction into the kernel; the CTA pool -- and with it the whole iteration
# partition -- is a kernarg, so one binary per (mode, reduction) serves every
# shape and pool. It is off unless a caller asks for it (--streamk): it adds
# a default-epilogue kernel per (mode, reduction) to every wgrad geometry,
# which more than doubles the wgrad grid. "atomic" accumulates into dW with
# f32 atomics and the validator keeps it to fp32 dW only.
#
# Only a slice of the (mode, reduction) x vector-width grid is cached (see
# _wgrad_streamk_builds). Stream-K pays off where the shape loads at the
# widest vector width, so the requested variants are built there, minus
# "tree" ("linear" is the other deterministic fixup). Narrower widths are left
# to the split-K kernels, which are cached at every width and serve those
# shapes as well -- except for one case: large-filter depthwise layers, whose
# per-group reduction is long and whose GEMM is wide. They load scalar, and
# stream-K wins there, so a scalar dp_sk slice is kept on the wide tiles.
CACHE_WGRAD_STREAMK_MODES: Tuple[str, ...] = ("dp_sk", "persistent")
CACHE_WGRAD_STREAMK_REDUCTIONS: Tuple[str, ...] = (
    "linear",
    "tree",
    "workspace",
    "atomic",
)
# Bytes per stream-K operand load at the widest vector width: 8 elements for
# fp16/bf16, 4 for fp32.
CACHE_WGRAD_STREAMK_LOAD_BYTES = 16
# Reductions not cached at the widest vector widths.
CACHE_WGRAD_STREAMK_WIDE_EXCLUDED_REDUCTIONS: Tuple[str, ...] = ("tree",)
# The scalar (vector width 1) slice for large-filter depthwise layers.
CACHE_WGRAD_STREAMK_SCALAR_MODES: Tuple[str, ...] = ("dp_sk",)
CACHE_WGRAD_STREAMK_SCALAR_REDUCTIONS: Tuple[str, ...] = ("tree", "workspace")
CACHE_WGRAD_STREAMK_SCALAR_MIN_TILE_N = 128

# Private aliases for internal use -- the generators below refer to these.
_TILE_MN = CACHE_TILE_MN
_TILE_K = CACHE_TILE_K
_WARP_MN = CACHE_WARP_MN
_WGRAD_EXTRA_TILE_MN = CACHE_WGRAD_EXTRA_TILE_MN
_WGRAD_EXTRA_WARP_MN = CACHE_WGRAD_EXTRA_WARP_MN
_WARP_TILE_MN = CACHE_WARP_TILE_MN
_MAX_ACC_REGS = CACHE_MAX_ACC_REGS
# ROCKE_AOT_VECS (e.g. "8") narrows the vector widths a scoped cache builds:
# --run-from-cache only offers the widest widths a problem admits unless
# --all-vector-sizes, so a cache for known shapes needs no narrower ones.
_VECS = tuple(
    int(v) for v in _os.environ.get("ROCKE_AOT_VECS", "").split(",") if v.strip()
) or CACHE_VECS
_PIPELINES = CACHE_PIPELINES
_EPILOGUES = CACHE_EPILOGUES
_GROUPED = CACHE_GROUPED
_WGRAD_GROUP_MERGES = CACHE_WGRAD_GROUP_MERGES
_FWD_GROUP_MERGES = CACHE_FWD_GROUP_MERGES
_WGRAD_SPLIT_KS = CACHE_WGRAD_SPLIT_KS


@dataclass(frozen=True)
class BuildJob:
    """One kernel to build: an identity plus the spec kwargs that produce it."""

    identity: KernelIdentity
    direction: str
    spec_kwargs: dict
    # Capability axes this job is built for. They pick the probe problem
    # (see _probe_problem); they are not spec kwargs.
    caps: dict


def _dtype_triple(dtype: str) -> Tuple[str, str, str]:
    return dtype, dtype, dtype


def _grid_axes(direction: Optional[str]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """``(tile_mn, warp_mn)`` for a direction, sorted so order is stable.

    ``None`` means the base grid, which is what a caller that walks an
    explicit geometry list already holds.
    """
    if direction == "dgrad" and CACHE_DGRAD_PP_EXTRA_TILE_MN:
        return tuple(sorted(set(_TILE_MN) | set(CACHE_DGRAD_PP_EXTRA_TILE_MN))), _WARP_MN
    if direction != "wgrad":
        return _TILE_MN, _WARP_MN
    return (
        tuple(sorted(set(_TILE_MN) | set(_WGRAD_EXTRA_TILE_MN))),
        tuple(sorted(set(_WARP_MN) | set(_WGRAD_EXTRA_WARP_MN))),
    )


def _geometries(
    target,
    mma_family: str,
    da: str,
    db: str,
    direction: Optional[str] = None,
) -> Iterator[tuple]:
    """The tile/warp/atom geometries a direction's generator walks.

    Yields ``(tile_m, tile_n, tile_k, warp_m, warp_n, wt, pipeline, epilogue,
    atom)`` for each combination whose warp tiling fits the block tile, whose
    K tile is a multiple of the selected MMA atom, and whose per-lane
    accumulator stays within :data:`CACHE_MAX_ACC_REGS`. Shared by the
    generators and :func:`count_jobs`, so the progress total cannot drift from
    what the generators actually yield.

    ``direction`` selects the grid: wgrad gets the filter-shaped tile and warp
    extensions (see :data:`CACHE_WGRAD_EXTRA_TILE_MN`), everything else the
    base grid. It must be threaded through wherever a geometry list is built
    for a direction, or the chunk bounds stop matching the list the worker
    walks.
    """
    tile_mn, warp_mn = _grid_axes(direction)
    for (
        tile_m,
        tile_n,
        tile_k,
        warp_m,
        warp_n,
        wt,
        pipeline,
        epilogue,
    ) in itertools.product(
        tile_mn,
        tile_mn,
        _TILE_K,
        warp_mn,
        warp_mn,
        _WARP_TILE_MN,
        _PIPELINES,
        _EPILOGUES,
    ):
        if warp_m * wt > tile_m or warp_n * wt > tile_n:
            continue
        if tile_m % (warp_m * wt) or tile_n % (warp_n * wt):
            continue
        acc_regs = (tile_m // warp_m) * (tile_n // warp_n) // target.wave_size
        if acc_regs > _MAX_ACC_REGS:
            continue
        atom = target.mma.select_largest_k(
            family=mma_family, a_dtype=da, b_dtype=db, c_dtype="fp32", m=wt, n=wt
        )
        if atom is None or tile_k % atom.k:
            continue
        yield tile_m, tile_n, tile_k, warp_m, warp_n, wt, pipeline, epilogue, atom


def _aot_pipeline(direction: str, pipeline: str) -> bool:
    """Whether the AOT cache builds ``pipeline`` for ``direction``."""
    return pipeline not in _AOT_ALIAS_PIPELINES[direction]


# XCD-contiguous tile orders (ImplicitGemmConvSpec.xcd_remap) built per arch:
# the XCD count is baked into the binary, and one gfx target spans parts with
# different counts (gfx942: MI300A has 6 XCDs, MI300X / MI325X 8), so each
# count is its own binary and the sweep picks the one that runs fastest on the
# device. Built only on the uniform_k_decode + reg_prefetch=1 loop -- the loop
# that won every large-spatial 3-D shape it was measured on -- so the grid grows
# by two kernels per geometry rather than doubling.
CACHE_FWD_XCD_REMAP: Dict[str, Tuple[int, ...]] = {"gfx942": (6, 8), "gfx950": (8,)}


def _fwd_k_loops(pipeline: str, arch: Optional[str] = None) -> List[dict]:
    """K-loop driver kwargs emitted per forward geometry.

    async_dma replaces the K-loop driver entirely and ignores the pipeline
    string, so it is only emitted once per geometry instead of as identical
    binaries under different pipeline labels.

    The register-staged prefetch loop (``prefetch_regs``,
    :data:`CACHE_FWD_PREFETCH_REGS`) is built on both sync drivers:
    single-buffered and, with ``unroll_k``, ping-pong. These and the plain
    loops also get the split-K and ``split_addr_k_decode`` variants
    (:func:`_fwd_small_spatial_variants`).

    ``reg_prefetch`` (the next tile's global reads in flight across the
    MFMAs; one or two LDS buffers) and ``uniform_k_decode`` (per-tile filter-tap
    decode, with the plain loop kept as the fallback for cpg % tile_k != 0)
    are built on the wave64 MFMA pipelines, alone and together.
    """
    k_loops = [
        dict(
            unroll_k=unroll,
            async_dma=False,
            **({"prefetch_regs": True} if pfr else {}),
        )
        for pfr in CACHE_FWD_PREFETCH_REGS
        for unroll in (False, True)
    ]
    if pipeline == "mem":
        k_loops.append(dict(unroll_k=False, async_dma=True))
    if pipeline != "wavelet":
        for ukd, rpf in ((False, 1), (False, 2), (True, 0), (True, 1), (True, 2)):
            k_loops.append(
                dict(
                    unroll_k=False,
                    async_dma=False,
                    uniform_k_decode=ukd,
                    reg_prefetch=rpf,
                )
            )
        # The same loop on XOR-swizzled, unpadded A/B LDS tiles (lds_xor;
        # the validator drops tile_k < 32): the saved pad is what lets a
        # 128x128x64 tile run two workgroups per CU.
        k_loops.append(
            dict(
                unroll_k=False,
                async_dma=False,
                uniform_k_decode=True,
                reg_prefetch=1,
                lds_xor=True,
            )
        )
        for xcds in CACHE_FWD_XCD_REMAP.get(arch, ()):
            for lx in (False, True):
                k_loops.append(
                    dict(
                        unroll_k=False,
                        async_dma=False,
                        uniform_k_decode=True,
                        reg_prefetch=1,
                        xcd_remap=xcds,
                        lds_xor=lx,
                    )
                )
        # The tap-fastest order of the same loops (uk_tap_fast; 3-D only, the
        # validator drops it for 2-D): lds_xor, launch and XCD orders.
        for xcds in (0,) + tuple(CACHE_FWD_XCD_REMAP.get(arch, ())):
            k_loops.append(
                dict(
                    unroll_k=False,
                    async_dma=False,
                    uniform_k_decode=True,
                    reg_prefetch=1,
                    xcd_remap=xcds,
                    lds_xor=True,
                    uk_tap_fast=True,
                )
            )
        # Opt-in (ROCKE_AOT_FWD_ADIR=1): the same loop with the A operand
        # loaded straight into the MFMA registers (a_direct), on padded and
        # XOR-swizzled B tiles, in launch and XCD-contiguous orders.
        if _os.environ.get("ROCKE_AOT_FWD_ADIR", "0") not in ("", "0"):
            for xcds in (0,) + tuple(CACHE_FWD_XCD_REMAP.get(arch, ())):
                for lx in (False, True):
                    k_loops.append(
                        dict(
                            unroll_k=False,
                            async_dma=False,
                            uniform_k_decode=True,
                            reg_prefetch=1,
                            lds_xor=lx,
                            xcd_remap=xcds,
                            a_direct=True,
                        )
                    )
    return _fwd_sched_k_loops(k_loops) + _fwd_pingpong_k_loops(k_loops, pipeline)


# Opt-in 8-wave block ping-pong K loops (ImplicitGemmConvSpec.rpf_pingpong),
# ROCKE_AOT_FWD_PP="p<mode>[m<pre>][s<prio>][v<vmi>][w<wpe>][b],..." (e.g. "p4,p5v4b"; b = pp_b_double):
# each token is built on every plain uniform_k_decode + reg_prefetch=1 lds_xor
# loop (any tap order / XCD order / split) with k-packed reads and without the
# per-tap bounds checks (lds_kpack_read + uk_nopad), pp_pre_mfma <pre>,
# rpf_setprio <prio>, vmem_interleave <vmi> (pp5) and waves_per_eu <wpe>, on the pipelines of
# ROCKE_AOT_FWD_PP_PIPES (default "mem": the loop emits no scheduler hints, so
# compv3/compv4 would only rebuild it) and only on the geometries of
# ROCKE_AOT_FWD_PP_SCOPE ("<M>x<N>x<K>[w<wm>x<wn>][a<atom>],...", required).
_FWD_PP_RE = re.compile(r"^p([1-5])(?:m(\d+))?(?:s([0-3]))?(?:v(\d))?(?:w(\d))?(b)?$")
_FWD_PP_PLAIN_OFF = ("lds_ahead", "defer_mfma", "vmem_interleave", "rpf_fetch_late",
                     "rpf_lds_barrier", "a_direct", "uk_xhalo", "uk_chunk_group",
                     "lds_kpack_read", "uk_nopad", "rpf_setprio", "waves_per_eu",
                     "defer_mfma_per_write", "xh_dpp")


def _fwd_pp_variants() -> Tuple[dict, ...]:
    out = []
    for tok in _os.environ.get("ROCKE_AOT_FWD_PP", "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        m = _FWD_PP_RE.match(tok)
        if m is None:
            raise ValueError(f"ROCKE_AOT_FWD_PP: {tok!r} is not p<1-5>[m<n>][s<0-3>][v<n>][w<n>][b]")
        v = dict(rpf_pingpong=int(m.group(1)), lds_kpack_read=True, uk_nopad=True)
        if m.group(2) and int(m.group(2)):
            v["pp_pre_mfma"] = int(m.group(2))
        if m.group(3) and int(m.group(3)):
            v["rpf_setprio"] = int(m.group(3))
        if m.group(4):
            v["vmem_interleave"] = int(m.group(4))
        if m.group(5):
            v["waves_per_eu"] = int(m.group(5))
        if m.group(6):
            v["pp_b_double"] = True
        if v not in out:
            out.append(v)
    return tuple(out)


def _fwd_pp_scope() -> List[Tuple[int, ...]]:
    raw = _os.environ.get("ROCKE_AOT_FWD_PP_SCOPE", "").strip()
    out = []
    for tok in (t.strip() for t in raw.split(",")):
        if not tok:
            continue
        m = _FWD_R2_SCOPE_RE.match(tok)
        if m is None:
            raise ValueError(f"ROCKE_AOT_FWD_PP_SCOPE: {tok!r} is not MxNxK[wAxB][aT]")
        out.append(tuple(int(g) if g else 0 for g in m.groups()))
    return out


def _fwd_pp_geometry_in_scope(tm: int, tn: int, tk: int, wm: int, wn: int, wt: int) -> bool:
    for m_, n_, k_, wm_, wn_, a_ in _fwd_pp_scope():
        if (m_, n_, k_) == (tm, tn, tk) and (not wm_ or (wm_, wn_) == (wm, wn)) and (
            not a_ or a_ == wt
        ):
            return True
    return False


def _fwd_pingpong_k_loops(k_loops: List[dict], pipeline: str) -> List[dict]:
    variants = _fwd_pp_variants()
    pipes = _os.environ.get("ROCKE_AOT_FWD_PP_PIPES", "mem").split(",")
    if not variants or pipeline not in pipes:
        return []
    out = []
    for kl in k_loops:
        if not (kl.get("uniform_k_decode") and kl.get("reg_prefetch") == 1
                and kl.get("lds_xor")) or any(kl.get(k) for k in _FWD_PP_PLAIN_OFF):
            continue
        for v in variants:
            d = dict(kl, **v)
            if d not in out:
                out.append(d)
    # ROCKE_AOT_FWD_PP_MCH=2[,4]: the ping-pong loops also with the C tile
    # staged in m-chunks (cshuffle_m_chunks; built only where the C tile
    # does not fit the LDS, see _fwd_r2_in_scope), with the row-keyed
    # swizzle when ROCKE_AOT_FWD_PP_MCH_LXRK=1.
    _mch = [int(t) for t in _os.environ.get("ROCKE_AOT_FWD_PP_MCH", "").split(",") if t.strip()]
    _mx = _os.environ.get("ROCKE_AOT_FWD_PP_MCH_LXRK", "0") not in ("", "0")
    out = out + [
        dict(d, cshuffle_m_chunks=m, **({"lds_xor_rowkey": True} if _mx else {}))
        for m in _mch for d in out if not d.get("pp_b_double")
    ]
    # ROCKE_AOT_FWD_ESF=1: the ping-pong loops also with the constant-step
    # in-bounds cshuffle store (epi_store_fast).
    if _os.environ.get("ROCKE_AOT_FWD_ESF", "0") not in ("", "0"):
        out = out + [dict(d, epi_store_fast=True) for d in out if not d.get("cshuffle_m_chunks")]
    # ROCKE_AOT_FWD_PP_CG=2[,4]: the tap-fastest ping-pong loops also with the
    # channel chunks grouped (uk_chunk_group; split binaries need uk_nopad).
    for g in (int(t) for t in _os.environ.get("ROCKE_AOT_FWD_PP_CG", "").split(",") if t.strip()):
        out = out + [
            dict(d, uk_chunk_group=g) for d in out
            if d.get("uk_tap_fast") and not d.get("uk_chunk_group") and not d.get("uk_xhalo")
        ]
    # ROCKE_AOT_FWD_PSF=3[,4]: the pp5 loops also with their global reads
    # split between [mem A] and [compute B] (pp_split_fetch).
    for n in (int(t) for t in _os.environ.get("ROCKE_AOT_FWD_PSF", "").split(",") if t.strip()):
        out = out + [
            dict(d, pp_split_fetch=n)
            for d in out
            if d.get("rpf_pingpong") == 5 and not d.get("uk_xhalo") and not d.get("pp_split_fetch")
        ]
    # ROCKE_AOT_FWD_FBSE=1: the variants the two axes above add
    # (pp_split_fetch / uk_chunk_group) skip the empty fallback loop's
    # prologue (fb_skip_empty).
    if _os.environ.get("ROCKE_AOT_FWD_FBSE", "0") not in ("", "0"):
        out = [dict(d, fb_skip_empty=True)
               if d.get("pp_split_fetch") or d.get("uk_chunk_group") else d for d in out]
    # ROCKE_AOT_FWD_ELB=1: ... and LDS-only epilogue barriers (epi_lds_barriers).
    if _os.environ.get("ROCKE_AOT_FWD_ELB", "0") not in ("", "0"):
        out = [dict(d, epi_lds_barriers=True)
               if d.get("pp_split_fetch") or d.get("uk_chunk_group") else d for d in out]
    # ROCKE_AOT_FWD_XSUB=32[,..]: the pp5 loops (no pp_b_double) also as the
    # x-halo sub-tile ping-pong (xh_subtile) staging tile_m + n pixels.
    for n in (int(t) for t in _os.environ.get("ROCKE_AOT_FWD_XSUB", "").split(",") if t.strip()):
        out = out + [
            dict(d, uk_xhalo=n, uk_nopad=False, xh_subtile=True)
            for d in out
            if d.get("rpf_pingpong") == 5 and not d.get("pp_b_double")
            and not d.get("uk_xhalo")
        ]
    return out


# Opt-in forward K-loop scheduling variants (ImplicitGemmConvSpec.lds_ahead /
# defer_mfma / vmem_interleave / rpf_setprio), as
# comma-separated ``a<ahead>d<defer>v<interleave>[p<prio>]`` tokens (ahead: 0
# off, 1 lds_ahead), e.g. ``ROCKE_AOT_FWD_SCHED=a1d1v2,a1d1v4p1``: each
# triple is built, besides the unscheduled loop, on every reg_prefetch=1 loop
# and (when it defers nothing) every single-buffer prefetch_regs and
# uniform_k_decode + reg_prefetch=2 (ping-pong) loop; the
# validator drops the combinations a loop or pipeline does not implement
# (vmem_interleave needs compv3/compv4). Unset: none (the default grid).
# ROCKE_AOT_FWD_LXOR=1 adds the XOR-swizzled unpadded LDS tiles (lds_xor) to
# the single-buffer prefetch_regs loops (the small-spatial split-K family)
# and the uniform_k_decode ping-pong loop (two 16 KB tile_k=32 buffers).
_FWD_SCHED_RE = re.compile(r"^a([01])d(\d+)v(\d+)(?:p([0-3]))?$")


def _fwd_sched_variants() -> Tuple[dict, ...]:
    out = []
    for tok in _os.environ.get("ROCKE_AOT_FWD_SCHED", "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        m = _FWD_SCHED_RE.match(tok)
        if m is None:
            raise ValueError(
                f"ROCKE_AOT_FWD_SCHED: {tok!r} is not a<0|1>d<n>v<n>[p<1-3>] "
                f"(e.g. a1d1v2, a1d1v4p1)"
            )
        lah, dfr, vmi = int(m.group(1)), int(m.group(2)), int(m.group(3))
        prio = int(m.group(4) or 0)
        v = {}
        if lah == 1:
            v["lds_ahead"] = True
        if prio:
            v["rpf_setprio"] = prio
        if dfr:
            v["defer_mfma"] = dfr
        if vmi:
            v["vmem_interleave"] = vmi
        if v and v not in out:
            out.append(v)
    return tuple(out)


# The opt-in K-loop knobs below (x-halo, k-packed reads, chunk groups, LDS-only
# barriers, late fetch, DPP shift, occupancy target): with
# ROCKE_AOT_FWD_R2_SCOPE="<M>x<N>x<K>[w<wm>x<wn>][a<atom>],..." (e.g.
# "256x64x32w4x1a16") a K loop carrying any of them is only built on those
# geometries, and with ROCKE_AOT_FWD_R2_XCD="6" only in those tile orders, so a
# cache can carry them where they win without multiplying the whole grid.
_FWD_R2_KNOBS = ("uk_xhalo", "lds_kpack_read", "uk_chunk_group", "rpf_lds_barrier",
                 "rpf_fetch_late", "xh_dpp", "waves_per_eu", "rpf_setprio", "xh_wait_smem",
                 "lds_xor_rowkey", "epi_quad_xpose", "no_agpr", "m_band", "xh_stride_m")
_FWD_R2_SCOPE_RE = re.compile(r"^(\d+)x(\d+)x(\d+)(?:w(\d+)x(\d+))?(?:a(\d+))?$")


def _fwd_r2_scope() -> Optional[List[Tuple[int, ...]]]:
    raw = _os.environ.get("ROCKE_AOT_FWD_R2_SCOPE", "").strip()
    if not raw:
        return None
    out = []
    for tok in raw.split(","):
        m = _FWD_R2_SCOPE_RE.match(tok.strip())
        if m is None:
            raise ValueError(f"ROCKE_AOT_FWD_R2_SCOPE: {tok!r} is not MxNxK[wAxB][aT]")
        out.append(tuple(int(g) if g else 0 for g in m.groups()))
    return out


# The opt-in K-loop knobs ROCKE_AOT_FWD_R2_SCOPE confines: opt-fwdl's, plus
# the scheduling knobs (ROCKE_AOT_FWD_SCHED) and uk_nopad. (ukd split-K and
# split_tail are confined by _fwd_job_variants on the same scope.)
_FWD_SCOPED_KNOBS = _FWD_R2_KNOBS + (
    "lds_ahead", "defer_mfma", "vmem_interleave", "uk_nopad", "defer_mfma_per_write",
)


def _fwd_r2_in_scope(kloop: dict, tm: int, tn: int, tk: int, wm: int, wn: int, wt: int) -> bool:
    if kloop.get("cshuffle_m_chunks", 1) > 1 and tm * tn * 2 <= 65536:
        # (ROCKE_AOT_FWD_PP_MCH) m-chunks only where the bf16 C tile does not
        # fit the LDS.
        return False
    if kloop.get("rpf_pingpong"):
        # The ping-pong loops have their own scope (ROCKE_AOT_FWD_PP_SCOPE).
        return _fwd_pp_geometry_in_scope(tm, tn, tk, wm, wn, wt)
    if not any(kloop.get(k) for k in _FWD_SCOPED_KNOBS):
        return True
    return _fwd_r2_geometry_in_scope(tm, tn, tk, wm, wn, wt)


def _fwd_r2_geometry_in_scope(tm: int, tn: int, tk: int, wm: int, wn: int, wt: int) -> bool:
    """Whether a geometry is in ROCKE_AOT_FWD_R2_SCOPE (unset: every one)."""
    scope = _fwd_r2_scope()
    if scope is None:
        return True
    for m_, n_, k_, wm_, wn_, a_ in scope:
        if (m_, n_, k_) == (tm, tn, tk) and (not wm_ or (wm_, wn_) == (wm, wn)) and (
            not a_ or a_ == wt
        ):
            return True
    return False


def _fwd_sched_k_loops(k_loops: List[dict]) -> List[dict]:
    """``k_loops`` plus the opt-in lds_xor and scheduling variants (see
    :func:`_fwd_sched_variants`)."""
    def pfr_single(kl: dict) -> bool:
        return bool(kl.get("prefetch_regs")) and not kl.get("unroll_k")

    def ukd_rpf2(kl: dict) -> bool:
        return bool(kl.get("uniform_k_decode")) and kl.get("reg_prefetch") == 2

    if _os.environ.get("ROCKE_AOT_FWD_LXOR", "0") not in ("", "0"):
        k_loops = k_loops + [
            dict(kl, lds_xor=True)
            for kl in k_loops
            if (pfr_single(kl) or ukd_rpf2(kl)) and not kl.get("lds_xor")
        ]
    # ROCKE_AOT_FWD_CG=2[,4]: the tap-fastest loops with the channel chunks
    # grouped (uk_chunk_group).
    for g in (int(t) for t in _os.environ.get("ROCKE_AOT_FWD_CG", "").split(",") if t.strip()):
        k_loops = k_loops + [
            dict(kl, uk_chunk_group=g) for kl in k_loops
            if kl.get("uk_tap_fast") and not kl.get("uk_chunk_group")
        ]
    # ROCKE_AOT_FWD_LBAR=1: LDS-only barriers (rpf_lds_barrier) on the
    # reg_prefetch=1 loops.
    if _os.environ.get("ROCKE_AOT_FWD_LBAR", "0") not in ("", "0"):
        k_loops = k_loops + [
            dict(kl, rpf_lds_barrier=True) for kl in k_loops if kl.get("reg_prefetch") == 1
        ]
    # ROCKE_AOT_FWD_FLATE=1: the next tile's global reads behind the first LDS
    # fragment reads (rpf_fetch_late) on the reg_prefetch=1 loops.
    if _os.environ.get("ROCKE_AOT_FWD_FLATE", "0") not in ("", "0"):
        k_loops = k_loops + [
            dict(kl, rpf_fetch_late=True) for kl in k_loops if kl.get("reg_prefetch") == 1
        ]
    # ROCKE_AOT_FWD_XHALO=4[,8]: the x-halo input reuse (uk_xhalo = the
    # staged range's rows past tile_m) on the 3-D lds_xor uniform_k_decode +
    # reg_prefetch=1 loops.
    for xh in (int(t) for t in _os.environ.get("ROCKE_AOT_FWD_XHALO", "").split(",") if t.strip()):
        k_loops = k_loops + [
            dict(kl, uk_xhalo=xh) for kl in k_loops
            if kl.get("lds_xor") and kl.get("uniform_k_decode") and kl.get("reg_prefetch") == 1
            and not kl.get("a_direct") and not kl.get("uk_xhalo")
            and not kl.get("rpf_fetch_late")
        ]
    # ROCKE_AOT_FWD_XWS=1: every x-halo loop with the scalar loads retired
    # before it (xh_wait_smem; partial lgkmcnt waits in its body).
    # ROCKE_AOT_FWD_XWS=only: in place of the loops without it.
    _xws = _os.environ.get("ROCKE_AOT_FWD_XWS", "0")
    if _xws == "only":
        k_loops = [dict(kl, xh_wait_smem=True) if kl.get("uk_xhalo") else kl for kl in k_loops]
    elif _xws not in ("", "0"):
        k_loops = k_loops + [dict(kl, xh_wait_smem=True) for kl in k_loops if kl.get("uk_xhalo")]
    # ROCKE_AOT_FWD_EQX=<n>[n][,...]: the x-halo loops also with the register-only
    # epi_quad_xpose=<n> store, a trailing "n" adding no_agpr (accumulators in
    # arch VGPRs) (=only:...: in place of the cshuffle store); jobs the store
    # does not apply to (split-K, 32x32 atom) are rejected.
    _eqx = _os.environ.get("ROCKE_AOT_FWD_EQX", "").strip()
    if _eqx:
        _only = _eqx.startswith("only:")
        _toks = [t.strip() for t in _eqx.split(":")[-1].split(",") if t.strip()]
        _vs = [dict(epi_quad_xpose=int(t.rstrip("n")), **({"no_agpr": True} if t.endswith("n") else {}))
               for t in _toks]
        _xh = [dict(kl, **v) for kl in k_loops if kl.get("uk_xhalo") for v in _vs]
        k_loops = ([kl for kl in k_loops if not kl.get("uk_xhalo")] if _only else k_loops) + _xh
    # ROCKE_AOT_FWD_MB=<n>[,...]: those register-store (epi_quad_xpose) x-halo
    # loops also with the output-plane band tile order m_band=<n> (needs
    # xcd_remap; the jobs without it are rejected).
    for _mb in (int(t) for t in _os.environ.get("ROCKE_AOT_FWD_MB", "").split(",") if t.strip()):
        k_loops = k_loops + [
            dict(kl, m_band=_mb) for kl in k_loops
            if kl.get("uk_xhalo") and kl.get("epi_quad_xpose") and not kl.get("m_band")
        ]
    # ROCKE_AOT_FWD_XSM=1: those register-store (epi_quad_xpose 3/4) x-halo
    # loops also with M-strided wave rows (xh_stride_m).
    if _os.environ.get("ROCKE_AOT_FWD_XSM", "").strip() == "1":
        k_loops = k_loops + [
            dict(kl, xh_stride_m=True) for kl in k_loops
            if kl.get("uk_xhalo") and kl.get("epi_quad_xpose") in (3, 4)
            and not kl.get("xh_dpp") and not kl.get("xh_stride_m")
        ]
    # ROCKE_AOT_FWD_XH_VMI=2[,4]: those (xh_wait_smem) x-halo loops also with
    # the next tile's global reads spread over the first tap's MFMAs
    # (vmem_interleave, any pipeline).
    for vi in (int(t) for t in _os.environ.get("ROCKE_AOT_FWD_XH_VMI", "").split(",") if t.strip()):
        k_loops = k_loops + [
            dict(kl, vmem_interleave=vi) for kl in k_loops
            if kl.get("xh_wait_smem") and not kl.get("vmem_interleave")
        ]
    # ROCKE_AOT_FWD_XDPP=1: the x-halo loops with DPP-shifted A fragments
    # (xh_dpp).
    if _os.environ.get("ROCKE_AOT_FWD_XDPP", "0") not in ("", "0"):
        k_loops = k_loops + [dict(kl, xh_dpp=True) for kl in k_loops if kl.get("uk_xhalo")]
    # ROCKE_AOT_FWD_PRIO=1[,2]: s_setprio over the load issue + MFMA phase of
    # the reg_prefetch=1 loops (rpf_setprio).
    for pr in (int(t) for t in _os.environ.get("ROCKE_AOT_FWD_PRIO", "").split(",") if t.strip()):
        k_loops = k_loops + [
            dict(kl, rpf_setprio=pr) for kl in k_loops
            if kl.get("reg_prefetch") == 1 and not kl.get("rpf_setprio") and not kl.get("a_direct")
        ]
    # ROCKE_AOT_FWD_KPRD=1: k-packed 16-byte LDS fragment reads
    # (lds_kpack_read) on every lds_xor uniform_k_decode / reg_prefetch loop.
    if _os.environ.get("ROCKE_AOT_FWD_KPRD", "0") not in ("", "0"):
        k_loops = k_loops + [
            dict(kl, lds_kpack_read=True)
            for kl in k_loops
            if kl.get("lds_xor")
            and (kl.get("uniform_k_decode") or kl.get("reg_prefetch"))
            and not kl.get("a_direct")
        ]
    # ROCKE_AOT_FWD_LXRK=1|only: the k-packed x-halo loops also (only) with the
    # row-keyed LDS swizzle (lds_xor_rowkey; the x-halo geometries are tile_k 32).
    _lxrk = _os.environ.get("ROCKE_AOT_FWD_LXRK", "0")
    if _lxrk not in ("", "0"):
        _xk = [dict(kl, lds_xor_rowkey=True) for kl in k_loops
               if kl.get("uk_xhalo") and kl.get("lds_kpack_read")]
        if _lxrk == "only":
            k_loops = [kl for kl in k_loops
                       if not (kl.get("uk_xhalo") and kl.get("lds_kpack_read"))]
        k_loops = k_loops + _xk
    # ROCKE_AOT_FWD_WPE=3[,4]: the reg_prefetch=1 lds_xor loops with an
    # occupancy target (waves_per_eu), e.g. to hold a 126-VGPR 8-wave tile
    # at 4 waves/SIMD once k-packed reads add registers.
    for w in (int(t) for t in _os.environ.get("ROCKE_AOT_FWD_WPE", "").split(",") if t.strip()):
        k_loops = k_loops + [
            dict(kl, waves_per_eu=w) for kl in k_loops
            if kl.get("reg_prefetch") == 1 and kl.get("lds_xor") and not kl.get("waves_per_eu")
        ]
    xcds = _os.environ.get("ROCKE_AOT_FWD_R2_XCD", "").strip()
    if xcds:
        keep = {int(t) for t in xcds.split(",") if t.strip()}
        k_loops = [
            kl for kl in k_loops
            if not any(kl.get(k) for k in _FWD_R2_KNOBS) or kl.get("xcd_remap", 0) in keep
        ]
    sched = _fwd_sched_variants()
    if not sched:
        return _fwd_nopad_k_loops(k_loops)
    extra = []
    for kl in k_loops:
        rpf1 = kl.get("reg_prefetch") == 1
        if not (rpf1 or pfr_single(kl) or ukd_rpf2(kl)):
            continue
        for v in sched:
            if v.get("defer_mfma") and not rpf1:
                continue
            # vmem_interleave on the x-halo loop is its own axis
            # (ROCKE_AOT_FWD_XH_VMI), so the sched sets keep their grid.
            if v.get("vmem_interleave") and kl.get("uk_xhalo"):
                continue
            extra.append(dict(kl, **v))
    # ROCKE_AOT_FWD_DPW=1[,3]: the defer_mfma loops also with that many MFMAs
    # per deferred ds_write (defer_mfma_per_write; rpf1, not x-halo).
    for dpw in (int(t) for t in _os.environ.get("ROCKE_AOT_FWD_DPW", "").split(",") if t.strip()):
        extra = extra + [
            dict(kl, defer_mfma_per_write=dpw)
            for kl in extra
            if kl.get("defer_mfma") and kl.get("reg_prefetch") == 1
            and not kl.get("uk_xhalo") and not kl.get("defer_mfma_per_write")
        ]
    return _fwd_nopad_k_loops(k_loops + extra)


def _fwd_nopad_k_loops(k_loops: List[dict]) -> List[dict]:
    """``ROCKE_AOT_FWD_NOPAD=1``: every uniform_k_decode loop also without
    its per-tap bounds checks (``uk_nopad``; serves unpadded problems)."""
    if _os.environ.get("ROCKE_AOT_FWD_NOPAD", "0") in ("", "0"):
        return k_loops
    return k_loops + [
        dict(kl, uk_nopad=True)
        for kl in k_loops
        if kl.get("uniform_k_decode") and not kl.get("a_direct")
    ]


def _fwd_split_addr_k_decodes(caps: dict, async_dma: bool) -> Tuple[bool, ...]:
    """Forward ``split_addr_k_decode`` values built for one variant: both on
    the 3-D sync-load binaries (the per-tile decode serves only problems with
    cpg % tile_k == 0, so the per-element binary stays for the rest), the
    per-element decode alone elsewhere."""
    if caps.get("is_3d", False) and not async_dma:
        return CACHE_FWD_SPLIT_ADDR_K_DECODE
    return (False,)


def _fwd_small_spatial_variants(
    kloop: dict, epilogue: str, caps: dict
) -> List[Tuple[int, bool]]:
    """``(split_k, split_addr_k_decode)`` pairs built on one forward K loop.

    The plain and ``prefetch_regs`` loops take both. The ``uniform_k_decode``
    + ``reg_prefetch=1`` loop also takes split-K when
    ``ROCKE_AOT_FWD_UKD_SPLIT=1`` (opt-in: the small-spatial grids that need
    it, s7/s8/s10-like, are a corner of the problem space); the other
    ``uniform_k_decode`` / ``reg_prefetch`` loops take neither.
    """
    if kloop.get("uniform_k_decode") or kloop.get("reg_prefetch"):
        if (
            kloop.get("uniform_k_decode")
            and kloop.get("reg_prefetch") == 1
            and not kloop.get("a_direct")
            and _os.environ.get("ROCKE_AOT_FWD_UKD_SPLIT", "0") not in ("", "0")
        ):
            return [(sk, False) for sk in _fwd_split_ks(epilogue, False, False)]
        return [(1, False)]
    unroll_k = kloop.get("unroll_k", False)
    async_dma = kloop.get("async_dma", False)
    return list(
        itertools.product(
            _fwd_split_ks(epilogue, unroll_k, async_dma),
            _fwd_split_addr_k_decodes(caps, async_dma),
        )
    )


def _fwd_job_variants(
    kloop: dict, geo: Tuple[int, ...], epilogue: str, caps: dict
) -> List[Tuple[int, bool, bool]]:
    """``(split_k, split_addr_k_decode, split_tail)`` built on one forward K
    loop of one geometry ``(tile_m, tile_n, tile_k, warp_m, warp_n,
    warp_tile)``; split-K on the uniform_k_decode loops and split_tail only
    on the geometries of ``ROCKE_AOT_FWD_R2_SCOPE`` (when set)."""
    in_scope = _fwd_r2_geometry_in_scope(*geo) or (
        bool(kloop.get("rpf_pingpong")) and _fwd_pp_geometry_in_scope(*geo)
    )
    ukd = bool(kloop.get("uniform_k_decode") or kloop.get("reg_prefetch"))
    out = []
    for sk, sd in _fwd_small_spatial_variants(kloop, epilogue, caps):
        if sk > 1 and ukd and not in_scope:
            continue
        for spt in _fwd_split_tails(kloop, sk):
            if spt and not in_scope:
                continue
            out.append((sk, sd, spt))
    return out


_FWD_SAA_TOKENS = {"a": dict(split_atomic_agent=True), "p": dict(split_atomic_pairs=True),
                   "ap": dict(split_atomic_agent=True, split_atomic_pairs=True),
                   "pf": dict(split_atomic_pairs=True, split_atomic_fast=True)}


def _fwd_split_atomic_variants(kloop: dict, variant: tuple) -> Tuple[dict, ...]:
    """Split-K store/order variants built for one forward split variant
    ``(split_k, split_addr_k_decode, split_tail)``: ``{}`` plus, on the
    ping-pong split binaries, with ``ROCKE_AOT_FWD_SAA="a,p,ap"``
    device-scope atomics (a = split_atomic_agent), lane-contiguous pairs
    (p = split_atomic_pairs) or both (ap); with ``ROCKE_AOT_FWD_SKX=6`` each
    of those also with the slice-major XCD remap (split_xcd_remap) where the
    loop has no xcd_remap and the variant no split tail; with
    ``ROCKE_AOT_FWD_FSK=1`` (same condition) each also as stream-K
    (split_streamk)."""
    split_k = variant[0]
    out = [{}]
    if split_k > 1 and kloop.get("rpf_pingpong"):
        for tok in _os.environ.get("ROCKE_AOT_FWD_SAA", "").split(","):
            tok = tok.strip()
            if not tok:
                continue
            if tok not in _FWD_SAA_TOKENS:
                raise ValueError(f"ROCKE_AOT_FWD_SAA: {tok!r} is not one of a, p, ap, pf")
            if _FWD_SAA_TOKENS[tok] not in out:
                out.append(_FWD_SAA_TOKENS[tok])
        # ROCKE_AOT_FWD_SFS=1: each also with the flag-ordered split store
        # (split_flag_store: slice 0 stores, no zeroed D).
        if _os.environ.get("ROCKE_AOT_FWD_SFS", "0") not in ("", "0"):
            out = out + [dict(v, split_flag_store=True) for v in out]
        # ROCKE_AOT_FWD_SZF=1: each also zeroing its own D tile (split_zero_first).
        if _os.environ.get("ROCKE_AOT_FWD_SZF", "0") not in ("", "0"):
            out = out + [dict(v, split_zero_first=True) for v in out
                         if not v.get("split_flag_store")]
        flat = not variant[2] and not kloop.get("xcd_remap")
        # ROCKE_AOT_FWD_FSK=1: each also as a stream-K binary (split_streamk).
        fsk = _os.environ.get("ROCKE_AOT_FWD_FSK", "0") not in ("", "0")
        skx = [int(t) for t in _os.environ.get("ROCKE_AOT_FWD_SKX", "").split(",") if t.strip()]
        base = list(out)
        base = [v for v in base
                if not v.get("split_flag_store") and not v.get("split_zero_first")]
        if fsk and flat:
            out = out + [dict(v, split_streamk=True) for v in base]
        # ROCKE_AOT_FWD_SHY=24[,32,..]: each also as a split_hybrid binary
        # (aligned slices + whole-tile remainders on the stream-K grid).
        shy = [int(t) for t in _os.environ.get("ROCKE_AOT_FWD_SHY", "").split(",") if t.strip()]
        if shy and flat:
            out = out + [dict(v, split_hybrid=e) for e in shy for v in base]
        if skx and flat:
            out = out + [dict(v, split_xcd_remap=n) for n in skx for v in base]
    return tuple(out)


def _fwd_split_tails(kloop: dict, split_k: int) -> Tuple[bool, ...]:
    """``split_tail`` values built for one forward split variant: the
    tail-only split (ImplicitGemmConvSpec.split_tail) is opt-in
    (``ROCKE_AOT_FWD_SPLIT_TAIL=1``), on the uniform_k_decode + reg_prefetch=1
    split binaries -- the loop the few-wave grids (s5/s6-like) run on."""
    if (
        split_k > 1
        and kloop.get("uniform_k_decode")
        and kloop.get("reg_prefetch") == 1
        and _os.environ.get("ROCKE_AOT_FWD_SPLIT_TAIL", "0") not in ("", "0")
    ):
        return (False, True)
    return (False,)


def _fwd_split_ks(epilogue: str, unroll_k: bool, async_dma: bool) -> Tuple[int, ...]:
    """Forward split-K values built for one K-loop driver (see
    :data:`CACHE_FWD_SPLIT_KS`)."""
    if epilogue != "cshuffle" or unroll_k or async_dma:
        return (1,)
    return CACHE_FWD_SPLIT_KS


def _wgrad_two_stages(split_k: int) -> Tuple[bool, ...]:
    """Two-stage variants for a split-K degree.

    Two-stage applies whenever the reduction is split (split_k > 1). It is
    what reaches problems the packed 16-bit atomic cannot address (an odd dW
    row, e.g. a 3-channel stem conv).
    """
    return (False, True) if split_k > 1 else (False,)


def _layout_caps() -> List[dict]:
    """The grouped capability variants fwd and wgrad build."""
    return [dict(grouped=g, is_3d=d) for d in CACHE_SPATIAL_3D for g in _GROUPED]


def _dgrad_caps() -> List[dict]:
    """dgrad's capability variants: grouped x (tilde, stride 1 / dilation 1)."""
    return [
        (
            {"grouped": g, "stride": 1, "dilation": 1, "is_3d": d}
            if s1
            else {"grouped": g, "tilde": True, "is_3d": d}
        )
        for d in CACHE_SPATIAL_3D
        for s1 in CACHE_DGRAD_STRIDE1
        for g in _GROUPED
    ]


def _dgrad_tap_outer_variants(
    caps: dict, wave_size: int, pipeline: str
) -> tuple[bool, ...]:
    """``tap_outer_k`` values built for one dgrad configuration.

    The tap-outer K loop (DgradConvSpec.tap_outer_k) is a stride-1,
    dilation-1 loop of the wave64 MFMA kernel, so only the stride-1
    capability gets it, and not under the wavelet pipeline (WMMA, its own K
    loop). The probe's kpg is a multiple of every cached tile_k; the cache
    checks kpg % tile_k on the problem.

    The stride-1 capability is built with that loop only, so it yields
    nothing where the loop does not apply: a stride-1 binary with the flat K
    loop only folds the sub-GEMM record, and the tilde binary of the same
    configuration (same K loop, record searched and loaded) serves the same
    problems at no measurable cost, so it would double the stride-1 slice of
    the catalog for nothing. The fold itself stays: the tap-outer (and dY
    halo) binaries are built on it.
    """
    stride1 = caps.get("stride") == 1 and caps.get("dilation") == 1
    if not stride1:
        return (False,)
    if wave_size == 64 and pipeline != "wavelet":
        return (True,)
    return ()


def _dgrad_halo_knobs(warp_n: int) -> tuple[tuple[int, int], ...]:
    """``(dy_halo_setprio, dy_halo_kouter_pad)`` pairs built per halo tile.

    The 4x1-wave tiles carry the wave priority and the wider K-outer B row
    pad as options (each with and without); the 2x2 tile neither.
    """
    if warp_n == 1:
        return tuple((prio, kpad) for prio in (0, 1) for kpad in (0, 32))
    return ((0, 0),)


def _dgrad_halo_jobs_for(
    arch: str, geometry: tuple, wave_size: int, db: str
) -> list[tuple[int, int, int, int, int, int, int]]:
    """The dY halo variants of one dgrad geometry, as ``(dy_halo, vec_bc,
    filter, halo_max_wo, setprio, kpad, vec_c)``; empty off the halo grid.

    Shared by :func:`_dgrad_jobs` and :func:`count_jobs`.
    """
    from kernels.common.conv_implicit_gemm_dgrad import DgradConvSpec

    tile_m, tile_n, tile_k, warp_m, warp_n, wt, pipeline, epilogue, _ = geometry
    if (tile_m, tile_n, tile_k, warp_m, warp_n, wt) not in CACHE_DGRAD_HALO_TILES:
        return []
    if pipeline != "mem" or wave_size != 64:
        return []
    out = []
    for vec_bc in CACHE_DGRAD_HALO_VEC_BC:
        if not DgradConvSpec.default_lds_k_outer(
            arch=arch,
            dtype_b=db,
            warp_tile_n=wt,
            cpg=vec_bc,
            wave_size=wave_size,
            pipeline=pipeline,
        ):
            continue
        vec_c = 1 if epilogue == "default" else vec_bc
        for mode, filt, max_wo, (prio, kpad) in itertools.product(
            CACHE_DGRAD_HALO_MODES,
            CACHE_DGRAD_HALO_FILTERS,
            CACHE_DGRAD_HALO_MAX_WO,
            _dgrad_halo_knobs(warp_n),
        ):
            out.append((mode, vec_bc, filt, max_wo, prio, kpad, vec_c))
    return out


def _is_patch_geometry(geometry: tuple) -> bool:
    """Is ``geometry`` (a :func:`_geometries` entry) one the patch dgrad builds?

    MFMA only: the patch store is not implemented for WMMA.
    """
    tile_m, tile_n, tile_k, warp_m, warp_n, wt, pipeline, epilogue, atom = geometry
    return (
        atom.family == "mma"
        and epilogue == "cshuffle"
        and (tile_m, tile_n, tile_k, warp_m, warp_n, wt, pipeline)
        in CACHE_DGRAD_PATCH_GEOMETRIES
    )


def _job_flavor(llvm_flavor: Optional[str]) -> str:
    """The LLVM flavor recorded in every identity of one enumeration.

    Resolved once per walk instead of per identity: ``KernelIdentity`` would
    otherwise call :func:`current_llvm_flavor` from its ``default_factory`` for
    each of millions of candidates, and that lookup globs the filesystem for
    the comgr library every time -- it was most of the enumeration's runtime.
    The parallel enumeration resolves it in the parent and passes it down, so
    every worker records the same flavor the parent would.
    """
    return llvm_flavor if llvm_flavor is not None else current_llvm_flavor()


def _fwd_jobs(
    arch: str,
    dtype: str,
    wave_size: int,
    mma_family: str,
    target,
    *,
    geometries: Optional[Sequence[tuple]] = None,
    llvm_flavor: Optional[str] = None,
) -> Iterator[BuildJob]:
    """Every forward implicit-GEMM variant worth caching for this arch/dtype.

    ``geometries`` restricts the walk to a slice of :func:`_geometries` (the
    parallel enumeration hands each worker one); ``llvm_flavor`` pins the
    identity's flavor (see :func:`_job_flavor`).
    """
    da, db, dd = _dtype_triple(dtype)
    flavor = _job_flavor(llvm_flavor)
    for (
        tile_m,
        tile_n,
        tile_k,
        warp_m,
        warp_n,
        wt,
        pipeline,
        epilogue,
        atom,
    ) in (
        _geometries(target, mma_family, da, db, "fwd")
        if geometries is None
        else geometries
    ):
        if not _aot_pipeline("fwd", pipeline):
            continue
        # fwd's A (X) and B (W) are both contiguous along cpg, so they share
        # one width; D (Y) runs along kpg. The default epilogue stores scalar
        # (c = 1 is its only width), so a cache narrowed by ROCKE_AOT_VECS
        # still builds it at c = 1 -- on the lds_xor loops only, where dropping
        # the cshuffle C tile is what lets a tile run one more workgroup per CU
        # (e.g. 256x64x32: 20 KB instead of the 32 KB C tile -> 3 WG/CU).
        narrowed = 1 not in _VECS
        vec_cs = (1,) if (epilogue == "default" and narrowed) else _VECS
        for vec_ab, vec_c in itertools.product(_VECS, vec_cs):
            for kloop, caps in itertools.product(
                _fwd_k_loops(pipeline, arch), _layout_caps()
            ):
                if narrowed and epilogue == "default" and not kloop.get("lds_xor"):
                    continue
                if not _fwd_r2_in_scope(kloop, tile_m, tile_n, tile_k, warp_m, warp_n, wt):
                    continue
                grouped = caps["grouped"]
                cfg = dict(
                    tile_m=tile_m,
                    tile_n=tile_n,
                    tile_k=tile_k,
                    warp_m=warp_m,
                    warp_n=warp_n,
                    warp_tile_m=wt,
                    warp_tile_n=wt,
                    warp_tile_k=atom.k,
                    pipeline=pipeline,
                    epilogue=epilogue,
                    wave_size=wave_size,
                    vector_size_a=vec_ab,
                    vector_size_b=vec_ab,
                    vector_size_c=vec_c,
                    **kloop,
                )
                for (split_k, sakd, spt), saa in (
                    (v, saa)
                    for v in _fwd_job_variants(
                        kloop, (tile_m, tile_n, tile_k, warp_m, warp_n, wt), epilogue, caps
                    )
                    for saa in _fwd_split_atomic_variants(kloop, v)
                ):
                    job_cfg = dict(cfg, split_k=split_k) if split_k > 1 else cfg
                    if sakd:
                        job_cfg = dict(job_cfg, split_addr_k_decode=True)
                    if spt:
                        job_cfg = dict(job_cfg, split_tail=True)
                    if saa:
                        job_cfg = dict(job_cfg, **saa)
                    # The identity's prefetch_regs is an int shared with
                    # wgrad (fwd's bool True is 1).
                    id_cfg = (
                        dict(job_cfg, prefetch_regs=1)
                        if job_cfg.get("prefetch_regs")
                        else job_cfg
                    )
                    yield BuildJob(
                        identity=KernelIdentity(
                            arch=arch,
                            direction="fwd",
                            algorithm="implicit_gemm",
                            llvm_flavor=flavor,
                            dtype_a=da,
                            dtype_b=db,
                            dtype_d=dd,
                            grouped=grouped,
                            is_3d=caps.get("is_3d", False),
                            **_async_chunks(job_cfg, (da, db, dd), caps, arch),
                            **id_cfg,
                        ),
                        direction="fwd",
                        spec_kwargs=job_cfg,
                        caps=caps,
                    )
        yield from _fwd_merged_jobs(
            arch,
            da,
            db,
            dd,
            flavor,
            wave_size,
            (tile_m, tile_n, tile_k, warp_m, warp_n, wt, pipeline, epilogue, atom),
        )


def _fwd_merged_k_loops() -> Tuple[bool, ...]:
    """``unroll_k`` values built for a group-merged forward geometry.

    async_dma is excluded: its chunk-granular predicate cannot carry the
    per-element diagonal mask of the merged B load, so the gate rejects it.
    """
    return (False, True)


def _fwd_merged_geometry(geometry: tuple) -> bool:
    """Whether group-merged forward kernels are built for ``geometry``.

    Only the cshuffle epilogue: the merged store runs Gm channels wide, which
    the default (scalar-store) epilogue rejects. On top of that the tile is
    restricted per the ``CACHE_FWD_GROUP_MERGE_*`` constants.
    """
    tile_m, tile_n, tile_k, _warp_m, warp_n, _wt, _pipeline, epilogue, _atom = geometry
    return (
        epilogue == "cshuffle"
        and tile_n <= CACHE_FWD_GROUP_MERGE_MAX_TILE_N
        and tile_k not in CACHE_FWD_GROUP_MERGE_EXCLUDED_TILE_K
        and warp_n not in CACHE_FWD_GROUP_MERGE_EXCLUDED_WARP_N
        and tile_m not in CACHE_FWD_GROUP_MERGE_EXCLUDED_TILE_M
    )


def _fwd_merged_jobs(
    arch: str,
    da: str,
    db: str,
    dd: str,
    flavor: str,
    wave_size: int,
    geometry: tuple,
) -> Iterator[BuildJob]:
    """Group-merged depthwise forward kernels for one geometry.

    Nothing is built for a geometry :func:`_fwd_merged_geometry` excludes.
    The vector widths are left to
    the builder: it derives them from the Gm-wide merged channel run -- exactly
    Gm for every problem the binary serves (cpg = kpg = 1) -- while an explicit
    width is validated against the per-group run of 1 and could only be 1.
    """
    tile_m, tile_n, tile_k, warp_m, warp_n, wt, pipeline, epilogue, atom = geometry
    if not _fwd_merged_geometry(geometry):
        return
    for gm, unroll_k in itertools.product(_FWD_GROUP_MERGES, _fwd_merged_k_loops()):
        cfg = dict(
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            warp_m=warp_m,
            warp_n=warp_n,
            warp_tile_m=wt,
            warp_tile_n=wt,
            warp_tile_k=atom.k,
            pipeline=pipeline,
            epilogue=epilogue,
            wave_size=wave_size,
            unroll_k=unroll_k,
            group_merge=gm,
        )
        yield BuildJob(
            identity=KernelIdentity(
                arch=arch,
                direction="fwd",
                algorithm="implicit_gemm",
                llvm_flavor=flavor,
                dtype_a=da,
                dtype_b=db,
                dtype_d=dd,
                grouped=True,
                # 0 = derived by the builder (from the Gm merged run).
                vector_size_a=0,
                vector_size_b=0,
                vector_size_c=0,
                **cfg,
            ),
            direction="fwd",
            spec_kwargs=cfg,
            caps=dict(grouped=True, group_merge=gm),
        )


def _async_chunks(cfg: dict, dtypes, caps: dict, arch: str) -> dict:
    """The fwd async loaders' chunk widths, for the identity.

    They are chosen from the build-time cpg -- the probe problem's -- so they
    are a capability of the binary and have to be recorded. Taken from the
    builder's own loader construction so the two cannot disagree. An invalid
    spec records 0; the job is dropped by the validity filter anyway.

    ``arch`` matters because the width is also capped by what the target's
    DRAM->LDS DMA can move per lane, so the same ``cfg`` records different
    chunk widths on gfx942 and gfx950.
    """
    if not cfg.get("async_dma"):
        return {}
    from kernels.common._conv_implicit_gemm_common import ConvDataSpec
    from kernels.common.conv_implicit_gemm import (
        ImplicitGemmConvSpec,
        async_tile_loaders,
    )

    da, db, dd = dtypes
    try:
        spec = ImplicitGemmConvSpec(
            problem=_probe_problem("fwd", caps),
            data=ConvDataSpec(dtype_a=da, dtype_b=db, dtype_d=dd),
            **cfg,
        )
        a_loader, b_loader = async_tile_loaders(spec, arch)
    except ValueError:
        return {}
    return dict(
        async_chunk_a=a_loader.elems_per_chunk,
        async_chunk_b=b_loader.elems_per_chunk,
    )


def _wgrad_streamk_variants(
    epilogue: str,
    streamk_modes: Sequence[str],
    streamk_reductions: Sequence[str],
) -> List[Tuple[str, str]]:
    """``(mode, reduction)`` stream-K variants built per wgrad geometry.

    Stream-K runs its epilogue once per tile inside the stream-K tile loop and
    only the default epilogue supports that, so cshuffle geometries get none.
    """
    if epilogue != "default":
        return []
    return list(itertools.product(streamk_modes, streamk_reductions))


def _wgrad_streamk_builds(
    epilogue: str,
    tile_n: int,
    dtype: str,
    streamk_modes: Sequence[str],
    streamk_reductions: Sequence[str],
) -> List[Tuple[str, str, int, int]]:
    """``(mode, reduction, vector_size_a, vector_size_b)`` built per geometry.

    Of the requested variants: every one but the
    ``CACHE_WGRAD_STREAMK_WIDE_EXCLUDED_REDUCTIONS`` at the widest width W of
    the dtype on both operands, and at W on dY with a scalar X for the
    narrow-channel layers (C = 3 stems) that cannot load X wide; plus the
    ``CACHE_WGRAD_STREAMK_SCALAR_*`` slice, fully scalar, on tiles at least
    ``CACHE_WGRAD_STREAMK_SCALAR_MIN_TILE_N`` wide. Every other combination
    falls to the split-K kernels.
    """
    variants = _wgrad_streamk_variants(epilogue, streamk_modes, streamk_reductions)
    w = CACHE_WGRAD_STREAMK_LOAD_BYTES // {"fp16": 2, "bf16": 2, "fp32": 4}[dtype]
    builds = [
        (mode, reduction, vec_a, vec_b)
        for mode, reduction in variants
        if reduction not in CACHE_WGRAD_STREAMK_WIDE_EXCLUDED_REDUCTIONS
        for vec_a, vec_b in ((w, w), (w, 1))
    ]
    if tile_n >= CACHE_WGRAD_STREAMK_SCALAR_MIN_TILE_N:
        builds += [
            (mode, reduction, 1, 1)
            for mode, reduction in variants
            if mode in CACHE_WGRAD_STREAMK_SCALAR_MODES
            and reduction in CACHE_WGRAD_STREAMK_SCALAR_REDUCTIONS
        ]
    return builds


def _wgrad_jobs(
    arch,
    dtype,
    wave_size,
    mma_family,
    target,
    split_ks,
    *,
    geometries: Optional[Sequence[tuple]] = None,
    llvm_flavor: Optional[str] = None,
    streamk_modes: Sequence[str] = (),
    streamk_reductions: Sequence[str] = CACHE_WGRAD_STREAMK_REDUCTIONS,
) -> Iterator[BuildJob]:
    from kernels.common.conv_implicit_gemm_wgrad import (
        _DEFAULT_WS_REPLICAS,
        WgradConvSpec,
    )

    da, db, dd = _dtype_triple(dtype)
    flavor = _job_flavor(llvm_flavor)
    for (
        tile_m,
        tile_n,
        tile_k,
        warp_m,
        warp_n,
        wt,
        pipeline,
        epilogue,
        atom,
    ) in (
        _geometries(target, mma_family, da, db, "wgrad")
        if geometries is None
        else geometries
    ):
        if not _aot_pipeline("wgrad", pipeline):
            continue
        # The K-outer LDS tile, picked per geometry by the predicate the JIT
        # sweep and dispatch use. It is a pure function of arch/dtype/atom, so
        # a shape-generic binary can carry it; leaving it at the spec default
        # (M-outer) caches only the slower layout on gfx950.
        lds_k_outer = WgradConvSpec.default_lds_k_outer(
            arch=arch,
            dtype_a=da,
            dtype_b=db,
            warp_tile_m=wt,
            warp_tile_n=wt,
            wave_size=wave_size,
        )
        # The row stride is the LDS bank-conflict lever on the M-outer tile.
        # Left unset the spec falls back to a flat pad of 8, which is not
        # conflict-free; derive it instead. This has to be told which layout
        # the geometry above settled on: a K-outer tile takes its stride from
        # the builder and rejects an explicit pad, so the derivation returns
        # None there -- as it does wherever else it does not apply (see
        # default_lds_k_pad). The key is then omitted so the spec and the
        # identity both keep their defaults.
        _k_pad = WgradConvSpec.default_lds_k_pad(
            arch=arch,
            tile_k=tile_k,
            dtype_a=da,
            dtype_b=db,
            lds_k_outer=lds_k_outer,
            wave_size=wave_size,
        )
        _pad_cfg = {} if _k_pad is None else {"lds_k_pad": _k_pad}
        # wgrad's B (X) and D (dW) are both contiguous along cpg, so they
        # share one width; A (dY) runs along kpg. The default epilogue (and so
        # every two-stage kernel) stores scalar, so there D is 1 and B takes
        # its own width -- tied to D it could only ever load scalar.
        for vec_a, vec_bc, split_k in itertools.product(_VECS, _VECS, split_ks):
            vec_c = 1 if epilogue == "default" else vec_bc
            for two_stage, caps in itertools.product(
                _wgrad_two_stages(split_k), _layout_caps()
            ):
                grouped = caps["grouped"]
                cfg = dict(
                    tile_m=tile_m,
                    tile_n=tile_n,
                    tile_k=tile_k,
                    warp_m=warp_m,
                    warp_n=warp_n,
                    warp_tile_m=wt,
                    warp_tile_n=wt,
                    warp_tile_k=atom.k,
                    pipeline=pipeline,
                    epilogue=epilogue,
                    wave_size=wave_size,
                    vector_size_a=vec_a,
                    vector_size_b=vec_bc,
                    vector_size_c=vec_c,
                    split_k=split_k,
                    two_stage=two_stage,
                    lds_k_outer=lds_k_outer,
                    **_pad_cfg,
                )
                if two_stage:
                    # Pinned in the spec and the identity alike, so the host
                    # sizes the scratch and Stage 2 from what was compiled.
                    cfg["ws_replicas"] = _DEFAULT_WS_REPLICAS
                cfg["prefetch_addr"] = _wgrad_prefetch_addr(arch, cfg)
                for _kloop in ({},) + _wgrad_kloop_variants(arch, cfg):
                    vcfg = dict(cfg, **_kloop)
                    yield BuildJob(
                        identity=KernelIdentity(
                            arch=arch,
                            direction="wgrad",
                            algorithm="implicit_gemm",
                            llvm_flavor=flavor,
                            dtype_a=da,
                            dtype_b=db,
                            dtype_d=dd,
                            grouped=grouped,
                            is_3d=caps.get("is_3d", False),
                            **vcfg,
                        ),
                        direction="wgrad",
                        spec_kwargs=vcfg,
                        caps=caps,
                    )
        # Stream-K kernels (opt-in). Default epilogue only, so the store is
        # scalar and B takes its own width; split_k stays 1 (stream-K owns the
        # K partition). The workspace reduction reuses the two-stage scratch,
        # so its replica count is pinned exactly as for two-stage.
        for (mode, reduction, vec_a, vec_b), caps in itertools.product(
            _wgrad_streamk_builds(
                epilogue, tile_n, dtype, streamk_modes, streamk_reductions
            ),
            _layout_caps(),
        ):
            cfg = dict(
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                warp_m=warp_m,
                warp_n=warp_n,
                warp_tile_m=wt,
                warp_tile_n=wt,
                warp_tile_k=atom.k,
                pipeline=pipeline,
                epilogue=epilogue,
                wave_size=wave_size,
                vector_size_a=vec_a,
                vector_size_b=vec_b,
                vector_size_c=1,
                streamk=mode,
                streamk_reduction=reduction,
                lds_k_outer=lds_k_outer,
                **_pad_cfg,
            )
            if reduction == "workspace":
                cfg["ws_replicas"] = _DEFAULT_WS_REPLICAS
            yield BuildJob(
                identity=KernelIdentity(
                    arch=arch,
                    direction="wgrad",
                    algorithm="implicit_gemm",
                    llvm_flavor=flavor,
                    dtype_a=da,
                    dtype_b=db,
                    dtype_d=dd,
                    grouped=caps["grouped"],
                    is_3d=caps.get("is_3d", False),
                    **cfg,
                ),
                direction="wgrad",
                spec_kwargs=cfg,
                caps=caps,
            )
        # Group-merged depthwise kernels. Merged split-K must take the
        # two-stage path (the packed atomic cannot drop off-diagonal pairs).
        # The vector widths are left to the builder: it derives them from the
        # Gm-wide merged channel run -- exactly Gm for every problem the binary
        # serves (cpg = kpg = 1) -- while an explicit width is validated
        # against the per-group run of 1 and could only be 1.
        for gm, split_k in itertools.product(
            _WGRAD_GROUP_MERGES, [sk for sk in split_ks if sk > 1]
        ):
            cfg = dict(
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                warp_m=warp_m,
                warp_n=warp_n,
                warp_tile_m=wt,
                warp_tile_n=wt,
                warp_tile_k=atom.k,
                pipeline=pipeline,
                epilogue=epilogue,
                wave_size=wave_size,
                split_k=split_k,
                two_stage=split_k > 1,
                group_merge=gm,
                lds_k_outer=lds_k_outer,
                **_pad_cfg,
            )
            if split_k > 1:
                cfg["ws_replicas"] = _DEFAULT_WS_REPLICAS
            cfg["prefetch_addr"] = _wgrad_prefetch_addr(arch, cfg)
            yield BuildJob(
                identity=KernelIdentity(
                    arch=arch,
                    direction="wgrad",
                    algorithm="implicit_gemm",
                    llvm_flavor=flavor,
                    dtype_a=da,
                    dtype_b=db,
                    dtype_d=dd,
                    grouped=True,
                    # 0 = derived by the builder (from the Gm merged run).
                    vector_size_a=0,
                    vector_size_b=0,
                    vector_size_c=0,
                    **cfg,
                ),
                direction="wgrad",
                spec_kwargs=cfg,
                caps=dict(grouped=True, group_merge=gm),
            )


def _wgrad_kloop_variants(arch: str, cfg: dict) -> Tuple[dict, ...]:
    """Extra K-loop variants built next to each plain wgrad job.

    ``CACHE_WGRAD_KLOOP_VARIANTS[arch]``, minus what cannot apply to the
    job: the register-prefetch loop and the packed transpose-on-store both
    rewrite the synchronous M-outer K loop, so a K-outer, async or
    prefetch_addr job gets none. (A variant the tile does not admit is
    rejected by the spec validator like any other grid point.)
    """
    if cfg.get("lds_k_outer") or cfg.get("async_dma") or cfg.get("unroll_k"):
        return ()
    if cfg.get("prefetch_addr"):
        return ()
    variants = CACHE_WGRAD_KLOOP_VARIANTS.get(arch, ())
    if cfg.get("split_k", 1) == 1:
        variants = variants + CACHE_WGRAD_KLOOP_UNSPLIT_VARIANTS.get(arch, ())
    if _WGRAD_XCD > 1 and cfg.get("split_k", 1) > 1:
        variants = variants + tuple(dict(v, xcd_remap=_WGRAD_XCD) for v in variants)
    if _WGRAD_PFS and (
        not _WGRAD_PFS_MIN_MN
        or (
            cfg.get("tile_m", 0) >= _WGRAD_PFS_MIN_MN
            and cfg.get("tile_n", 0) >= _WGRAD_PFS_MIN_MN
            and cfg.get("tile_k", 0) >= 32
        )
    ):
        split = cfg.get("split_k", 1) > 1
        variants = variants + tuple(
            dict(v, pfr_sched=p)
            for v in variants
            for p in _WGRAD_PFS
            if v.get("prefetch_regs") in (1, 3)
            and (_WGRAD_XCD <= 1 or not split or v.get("xcd_remap"))
            and (p == 1 or cfg.get("pipeline") == "mem")
        )
    _n_ok = (
        cfg.get("tile_n", 0) >= _WGRAD_M64_MIN_N
        and cfg.get("tile_m", 0) >= _WGRAD_M64_MIN_M
        and (not _WGRAD_M64_SPLIT_ONLY or cfg.get("split_k", 1) > 1)
    )
    if _WGRAD_KPL and _n_ok and (
        not _WGRAD_KPL_MAX_M or cfg.get("tile_m", 0) <= _WGRAD_KPL_MAX_M
    ):
        split = cfg.get("split_k", 1) > 1
        variants = variants + tuple(
            dict(v, kp_line=_WGRAD_KPL)
            for v in variants
            if v.get("prefetch_regs") in (1, 3)
            and v.get("lds_kpack", 1) > 1
            and (_WGRAD_XCD <= 1 or not split or v.get("xcd_remap"))
        )
    if _WGRAD_BSWZ and _n_ok and (
        not _WGRAD_BSWZ_MAX_M or cfg.get("tile_m", 0) <= _WGRAD_BSWZ_MAX_M
    ):
        split = cfg.get("split_k", 1) > 1
        bswz = tuple(
            dict(v, lds_bswz=_WGRAD_BSWZ)
            for v in variants
            if v.get("prefetch_regs") in (1, 3)
            and (_WGRAD_XCD <= 1 or not split or v.get("xcd_remap"))
        )
        if _WGRAD_BSWZ_WPE and cfg.get("tile_k") == 32:
            bswz = bswz + tuple(dict(v, waves_per_eu=_WGRAD_BSWZ_WPE) for v in bswz)
        variants = variants + bswz
    _tk2 = bool(
        _WGRAD_XWIN_TK2
        and cfg.get("tile_k") == _WGRAD_XWIN_TK2
        and cfg.get("tile_m", 0) >= _WGRAD_XWIN_TK2_MIN_M
        and _WGRAD_XWIN_TK2_WSR
        and _WGRAD_XWIN_WPE
    )
    if (
        _WGRAD_XWIN
        and cfg.get("split_k", 1) > 1
        and cfg.get("two_stage")
        and (not _WGRAD_XWIN_TK or cfg.get("tile_k") == _WGRAD_XWIN_TK or _tk2)
        and (not _WGRAD_XWIN_M or cfg.get("tile_m", 0) in _WGRAD_XWIN_M)
    ):
        tn = cfg.get("tile_n", 0)
        over = None
        if tn % _WGRAD_XWIN == 0 and (
            not _WGRAD_XWIN_CB or (tn // _WGRAD_XWIN) % _WGRAD_XWIN_CB == 0
        ):
            over = {}
        elif (
            tn == 256
            and _WGRAD_XWIN_N384_MIN_M
            and cfg.get("tile_m", 0) >= _WGRAD_XWIN_N384_MIN_M
            and (
                not _WGRAD_XWIN_N384_WT
                or cfg.get("warp_tile_m") == _WGRAD_XWIN_N384_WT
            )
        ):
            over = {"tile_n": 384}
        if over is not None:
            xw = tuple(
                dict(v, xwin=_WGRAD_XWIN, sep_addr=0, **over)
                for v in variants
                if v.get("prefetch_regs") == 1
                and v.get("lds_kpack", 1) == 2
                and not v.get("kp_line")
                and not v.get("lds_bswz")
                and not v.get("waves_per_eu")
                and (_WGRAD_XCD <= 1 or v.get("xcd_remap"))
            )
            if _WGRAD_XWIN_WPE and cfg.get("tile_m", 0) <= _WGRAD_XWIN_WPE_MAX_M:
                xw = xw + tuple(
                    dict(v, waves_per_eu=w) for w in _WGRAD_XWIN_WPE for v in xw
                )
            if _tk2 and not (_WGRAD_XWIN_TK and cfg.get("tile_k") == _WGRAD_XWIN_TK):
                xw = tuple(
                    dict(v, ws_replicas=_WGRAD_XWIN_TK2_WSR)
                    for v in xw
                    if v.get("waves_per_eu")
                )
                variants = variants + xw
                return _wgrad_m_band_variants(cfg, variants)
            _nw = cfg.get("warp_m", 0) * cfg.get("warp_n", 0)
            if (
                _WGRAD_XWIN_PP
                and (
                    (cfg.get("tile_m") == 64 and cfg.get("warp_m") == 1 and _nw == 2)
                    or (cfg.get("tile_m") == 128 and _nw == 4)
                    or (cfg.get("tile_m") == 256 and cfg.get("warp_m") == 4 and _nw == 8)
                )
                and cfg.get("warp_tile_m") == 16
                and cfg.get("pipeline") == "mem"
            ):
                _pp = tuple(
                    dict(v, xwin_groups=8 // _nw, xwin_pp=n, waves_per_eu=2)
                    if _nw < 8
                    else dict(v, xwin_pp=n, waves_per_eu=2)
                    for n in _WGRAD_XWIN_PP
                    for v in xw
                    if not v.get("pfr_sched") and not v.get("waves_per_eu")
                )
                # + the groups-over-consecutive-tiles twins of the multi-group
                # ones (xwin_gn 1; 2 = with the dY tile shared in LDS)
                _pp = _pp + tuple(
                    dict(v, xwin_gn=gn)
                    for gn in (1, 2)
                    for v in _pp
                    if v.get("xwin_groups", 1) > 1
                )
                xw = xw + _pp
            if _WGRAD_XWIN_MROW and cfg.get("tile_m", 0) >= _WGRAD_XWIN_MROW_MIN_M:
                mrow = tuple(
                    dict(v, xwin_mrow=n, **(
                        {"ws_replicas": _WGRAD_XWIN_MROW_WSR} if _WGRAD_XWIN_MROW_WSR else {}
                    ))
                    for n in _WGRAD_XWIN_MROW
                    for v in xw
                    if v.get("waves_per_eu") and not v.get("xwin_pp")
                    and cfg.get("tile_k", 0) % n == 0 and (cfg.get("tile_k", 0) // n) % 4 == 0
                )
            else:
                mrow = ()
            if _WGRAD_XWIN_WSR and cfg.get("tile_m", 0) >= _WGRAD_XWIN_WSR_MIN_M:
                xw = xw + tuple(
                    dict(v, ws_replicas=r)
                    for r in _WGRAD_XWIN_WSR
                    if r != cfg.get("ws_replicas")
                    for v in xw
                )
            variants = variants + xw + mrow
    return _wgrad_m_band_variants(cfg, variants)


def _wgrad_m_band_variants(cfg: dict, variants: tuple) -> tuple:
    """ROCKE_AOT_WGRAD_MB=<n>[,...]: the split-K xwin variants with xcd_remap
    also with the split-K slice band order m_band=<n> (WgradConvSpec.m_band)."""
    _wmb = [int(t) for t in _os.environ.get("ROCKE_AOT_WGRAD_MB", "").split(",") if t.strip()]
    if _wmb and cfg.get("split_k", 1) > 1:
        variants = variants + tuple(
            dict(v, m_band=m)
            for m in _wmb
            for v in variants
            if v.get("xwin") and v.get("xcd_remap", 0) > 1 and not v.get("m_band")
        )
    return variants


def _wgrad_prefetch_addr(arch: str, cfg: dict) -> bool:
    """``prefetch_addr`` for a wgrad job, from the policy the JIT sweep uses.

    A pure function of the binary's own configuration, so a shape-generic
    kernel can carry it; the identity records it through ``cfg``.
    """
    from kernels.common.conv_implicit_gemm_wgrad import WgradConvSpec

    return WgradConvSpec.default_prefetch_addr(
        arch=arch,
        tile_m=cfg["tile_m"],
        tile_n=cfg["tile_n"],
        tile_k=cfg["tile_k"],
        lds_k_outer=cfg.get("lds_k_outer", False),
        wave_size=cfg["wave_size"],
        async_dma=cfg.get("async_dma", False),
        unroll_k=cfg.get("unroll_k", False),
    )


def _dgrad_jobs(
    arch,
    dtype,
    wave_size,
    mma_family,
    target,
    max_sub_gemms,
    *,
    geometries: Optional[Sequence[tuple]] = None,
    llvm_flavor: Optional[str] = None,
) -> Iterator[BuildJob]:
    from kernels.common.conv_implicit_gemm_dgrad import DgradConvSpec

    da, db, dd = _dtype_triple(dtype)
    flavor = _job_flavor(llvm_flavor)
    for (
        tile_m,
        tile_n,
        tile_k,
        warp_m,
        warp_n,
        wt,
        pipeline,
        epilogue,
        atom,
    ) in (
        _geometries(target, mma_family, da, db, "dgrad")
        if geometries is None
        else geometries
    ):
        if not _aot_pipeline("dgrad", pipeline):
            continue
        # The tilde binary takes stride and dilation at run time (the tilde
        # records); its identity records them as 0. The stride-1 binary
        # records the stride and dilation it serves.
        # dgrad's B (W, KYXC) and D (dX) are both contiguous along cpg, so
        # they share one width; A (dY) runs along kpg. The default epilogue
        # stores scalar, so there D is 1 and B takes its own width.
        for vec_a, vec_bc, caps in itertools.product(_VECS, _VECS, _dgrad_caps()):
            vec_c = 1 if epilogue == "default" else vec_bc
            # The JIT sweep's B-tile layout. Its predicate keys on the parity
            # of cpg, which is runtime here; the B load width stands in for
            # it: a width > 1 only serves problems whose cpg it divides (an
            # even channel run), and width 1 is the odd-cpg kernel, where
            # K-outer has no scatter to remove.
            lds_k_outer = DgradConvSpec.default_lds_k_outer(
                arch=arch,
                dtype_b=db,
                warp_tile_n=wt,
                cpg=vec_bc,
                wave_size=wave_size,
                pipeline=pipeline,
            )
            grouped = caps["grouped"]
            cfg = dict(
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                warp_m=warp_m,
                warp_n=warp_n,
                warp_tile_m=wt,
                warp_tile_n=wt,
                warp_tile_k=atom.k,
                pipeline=pipeline,
                epilogue=epilogue,
                wave_size=wave_size,
                vector_size_a=vec_a,
                vector_size_b=vec_bc,
                vector_size_c=vec_c,
                lds_k_outer=lds_k_outer,
                max_sub_gemms=max_sub_gemms,
                # A pure function of the binary's own configuration (the
                # kernel picks between its two K loops from the kernargs), so
                # a shape-generic binary can carry it. The policy the JIT
                # sweep uses; the identity records it through cfg.
                uniform_k_decode=DgradConvSpec.default_uniform_k_decode(
                    arch=arch, wave_size=wave_size, pipeline=pipeline
                ),
            )
            # The stride-1 binary is the tap-outer K loop, which replaces the
            # uniform_k_decode loop and serves only problems whose kpg is a
            # multiple of tile_k (supports_problem).
            for tapk, tpf, bkp in (
                (t, f, k)
                for t in _dgrad_tap_outer_variants(caps, wave_size, pipeline)
                for f in (
                    (
                        CACHE_DGRAD_TAPK_PREFETCH
                        + tuple(("xh", x) for x in CACHE_DGRAD_XHALO_TPF)
                    )
                    if t
                    else (0,)
                )
                for k in (CACHE_DGRAD_B_KPACK if t else (0,))
            ):
                # Extra (ping-pong only) tile extents: the PPK sets of the
                # plain tpf1 binary alone.
                pp_only = tile_m not in _TILE_MN or tile_n not in _TILE_MN
                if pp_only and not (tapk and tpf == 1):
                    continue
                filt = {}
                job_caps = caps
                if tapk:
                    cfg = {
                        **cfg,
                        "uniform_k_decode": False,
                        "tap_outer_k": True,
                        "tapk_prefetch": tpf[1] if isinstance(tpf, tuple) else tpf,
                        "tapk_xhalo": isinstance(tpf, tuple),
                    }
                    if not cfg["tapk_prefetch"]:
                        del cfg["tapk_prefetch"]
                    if not cfg["tapk_xhalo"]:
                        del cfg["tapk_xhalo"]
                    else:
                        # The x taps are unrolled: the probe's filter is baked,
                        # and its W pad is 0 (the problems the binary serves).
                        filt = {"filter_w": 3}
                        job_caps = {**caps, "xhalo": True}
                    if bkp:
                        cfg["b_kpack"] = bkp
                    else:
                        cfg.pop("b_kpack", None)
                knob_sets = (((),) if not (tapk and isinstance(tpf, tuple)) else CACHE_DGRAD_XHALO_KNOBS)
                knob_cfgs = [
                    (kc, dict(k).get("@", CACHE_DGRAD_XHALO_KNOB_SCOPE))
                    for kc, k in (
                        (
                            _dgrad_xh_knob_cfg(
                                k, tpf=tpf[1] if isinstance(tpf, tuple) else tpf,
                                grouped=grouped, epilogue=epilogue,
                            ),
                            k,
                        )
                        for k in knob_sets
                    )
                    if kc is not None
                ]
                if pp_only:
                    knob_cfgs = []
                if tapk and tpf == 1:
                    # The plain loop's knob sets (ROCKE_AOT_DGRAD_PPK).
                    knob_cfgs += [
                        (kc, dict(k)["@"])
                        for kc, k in (
                            (_dgrad_pp_knob_cfg(k, epilogue=epilogue), k)
                            for k in CACHE_DGRAD_PLAIN_KNOBS
                            if k
                        )
                        if kc is not None
                    ]
                for kcfg, kscope in knob_cfgs:
                    jcfg = {**cfg, **kcfg}
                    # The x-halo ping-pong (ppx) bakes the filter width and
                    # serves pW == 0, as the x-halo loop does.
                    _ppx = bool(kcfg.get("tapk_pp_xhalo"))
                    ident = KernelIdentity(
                        arch=arch,
                        direction="dgrad",
                        algorithm="implicit_gemm",
                        llvm_flavor=flavor,
                        dtype_a=da,
                        dtype_b=db,
                        dtype_d=dd,
                        grouped=grouped,
                        is_3d=caps.get("is_3d", False),
                        stride_h=caps.get("stride", 0),
                        stride_w=caps.get("stride", 0),
                        dilation_h=caps.get("dilation", 0),
                        dilation_w=caps.get("dilation", 0),
                        **({**filt, "filter_w": 3} if _ppx else filt),
                        **jcfg,
                    )
                    if kcfg and kscope:
                        base_label = KernelIdentity(
                            **{
                                **ident.to_dict(),
                                **{
                                    k: KernelIdentity.__dataclass_fields__[k].default
                                    for k in kcfg
                                },
                                **({"filter_w": filt.get("filter_w", 0)} if _ppx else {}),
                            }
                        ).label()
                        if not _re.search(kscope, base_label):
                            continue
                    yield BuildJob(
                        identity=ident,
                        direction="dgrad",
                        spec_kwargs=jcfg,
                        caps={**job_caps, "xhalo": True} if _ppx else job_caps,
                    )

        # The dY halo mode of the tap-outer loop (stride 1 / dilation 1, the
        # filter and the widest Wo baked), on the halo tiles only.
        geometry = (
            tile_m,
            tile_n,
            tile_k,
            warp_m,
            warp_n,
            wt,
            pipeline,
            epilogue,
            atom,
        )
        for mode, vec_bc, filt, max_wo, prio, kpad, vec_c in _dgrad_halo_jobs_for(
            arch, geometry, wave_size, db
        ):
            caps = {
                "grouped": True,
                "stride": 1,
                "dilation": 1,
                "halo_filter": filt,
                "halo_max_wo": max_wo,
            }
            cfg = {
                "tile_m": tile_m,
                "tile_n": tile_n,
                "tile_k": tile_k,
                "warp_m": warp_m,
                "warp_n": warp_n,
                "warp_tile_m": wt,
                "warp_tile_n": wt,
                "warp_tile_k": atom.k,
                "pipeline": pipeline,
                "epilogue": epilogue,
                "wave_size": wave_size,
                "vector_size_a": CACHE_DGRAD_HALO_VEC_A,
                "vector_size_b": vec_bc,
                "vector_size_c": vec_c,
                "lds_k_outer": True,
                "max_sub_gemms": max_sub_gemms,
                "uniform_k_decode": False,
                "tap_outer_k": True,
                "dy_halo": mode,
                "dy_halo_setprio": prio,
                "dy_halo_kouter_pad": kpad,
                "halo_max_wo": max_wo,
            }
            yield BuildJob(
                identity=KernelIdentity(
                    arch=arch,
                    direction="dgrad",
                    algorithm="implicit_gemm",
                    llvm_flavor=flavor,
                    dtype_a=da,
                    dtype_b=db,
                    dtype_d=dd,
                    grouped=True,
                    stride_h=1,
                    stride_w=1,
                    dilation_h=1,
                    dilation_w=1,
                    filter_h=filt,
                    filter_w=filt,
                    **cfg,
                ),
                direction="dgrad",
                spec_kwargs=cfg,
                caps=caps,
            )

        # The patch-embedding GEMM (ungrouped, on the tilde record path) on its
        # geometries only, at every B/D width; A (dY) runs along K.
        if not _is_patch_geometry(
            (tile_m, tile_n, tile_k, warp_m, warp_n, wt, pipeline, epilogue, atom)
        ):
            continue
        caps = dict(grouped=False, tilde=True, patch=True)
        for vec_bc in _VECS:
            cfg = dict(
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                warp_m=warp_m,
                warp_n=warp_n,
                warp_tile_m=wt,
                warp_tile_n=wt,
                warp_tile_k=atom.k,
                pipeline=pipeline,
                epilogue=epilogue,
                wave_size=wave_size,
                vector_size_a=8,
                vector_size_b=vec_bc,
                vector_size_c=vec_bc,
                lds_k_outer=DgradConvSpec.default_lds_k_outer(
                    arch=arch,
                    dtype_b=db,
                    warp_tile_n=wt,
                    cpg=vec_bc,
                    wave_size=wave_size,
                    pipeline=pipeline,
                ),
                max_sub_gemms=max_sub_gemms,
                uniform_k_decode=DgradConvSpec.default_uniform_k_decode(
                    arch=arch, wave_size=wave_size, pipeline=pipeline
                ),
                patch=True,
            )
            yield BuildJob(
                identity=KernelIdentity(
                    arch=arch,
                    direction="dgrad",
                    algorithm="implicit_gemm",
                    llvm_flavor=flavor,
                    dtype_a=da,
                    dtype_b=db,
                    dtype_d=dd,
                    grouped=False,
                    **cfg,
                ),
                direction="dgrad",
                spec_kwargs=cfg,
                caps=caps,
            )


def job_for_identity(identity: KernelIdentity) -> BuildJob:
    """The job ``--compile-all`` builds ``identity`` from.

    The inverse of the generators above, obtained by running them: the
    identity's geometry is fed to its direction's generator (with the
    identity's own split-K degree, sub-GEMM cap and LLVM flavor) and the job
    whose identity is equal is returned. So a rebuilt job is exactly the one
    the cache build made, never a reconstruction that could drift from it.

    Raises ``LookupError`` for an identity the generators of this checkout do
    not produce (a different grid, or a field they never set).
    """
    from rocke.core.arch import ArchTarget

    if identity.algorithm != "implicit_gemm" or identity.direction not in (
        "fwd",
        "wgrad",
        "dgrad",
    ):
        raise LookupError(
            f"not an implicit-GEMM conv identity: {identity.direction}/"
            f"{identity.algorithm}"
        )
    if not identity.dtype_a == identity.dtype_b == identity.dtype_d:
        raise LookupError("the AOT grid builds A, B and D in one dtype")
    if identity.warp_tile_m != identity.warp_tile_n:
        raise LookupError("the AOT grid builds square warp tiles only")
    target = ArchTarget.from_gfx(identity.arch)
    if target.wave_size != identity.wave_size:
        raise LookupError(
            f"{identity.arch} runs wave{target.wave_size}, the identity records "
            f"wave{identity.wave_size}"
        )
    dtype = identity.dtype_a
    wt = identity.warp_tile_m
    mma_family = "wmma" if target.wave_size == 32 else "mma"
    atom = target.mma.select_largest_k(
        family=mma_family, a_dtype=dtype, b_dtype=dtype, c_dtype="fp32", m=wt, n=wt
    )
    if atom is None or atom.k != identity.warp_tile_k:
        raise LookupError(
            f"no {wt}x{wt}x{identity.warp_tile_k} {dtype} MMA atom on {identity.arch}"
        )
    geometry = (
        identity.tile_m,
        identity.tile_n,
        identity.tile_k,
        identity.warp_m,
        identity.warp_n,
        wt,
        identity.pipeline,
        identity.epilogue,
        atom,
    )
    want = identity.stable_hash()
    streamk = (
        {
            "streamk_modes": (identity.streamk,),
            "streamk_reductions": (identity.streamk_reduction,),
        }
        if identity.direction == "wgrad" and identity.streamk != "off"
        else {}
    )
    for job in _direction_jobs(
        identity.direction,
        identity.arch,
        dtype,
        target,
        (identity.split_k,),
        identity.max_sub_gemms,
        geometries=[geometry],
        llvm_flavor=identity.llvm_flavor,
        **streamk,
    ):
        if job.identity.stable_hash() == want:
            return job
    raise LookupError(
        f"{identity.label()} is not produced by this checkout's "
        f"{identity.direction} generator (another grid, or a field it never sets)"
    )


# Forward LDS budget per arch for the AOT grid, below the arch's capacity.
# gfx1250 has 320 KB of LDS per workgroup; its catalog value was a 160 KB
# placeholder until the WMMA direct dgrad. The forward configurations between
# the two (256x256 wavelet/cshuffle tiles) have not been run with --verify on
# gfx1250 yet, so its forward grid stays at the old budget until they have.
_AOT_FWD_LDS_BUDGET = {"gfx1250": 160 * 1024}

# Configurations that validate but are left out of the AOT grid, per
# (arch, direction): each entry is matched against the job's spec_kwargs.
_AOT_EXCLUDED = {
    # Fails --verify on the C=3 stem dgrad rows: the pre-existing
    # vector_size_a=8 C=3 implicit-GEMM dgrad bug. The 320 KB gfx1250 LDS
    # admits this tile; the bug's other configurations predate it.
    ("gfx1250", "dgrad"): (
        dict(
            tile_m=256,
            tile_n=256,
            tile_k=64,
            warp_m=1,
            warp_n=8,
            pipeline="wavelet",
            epilogue="cshuffle",
            vector_size_a=8,
            vector_size_b=1,
            vector_size_c=1,
        ),
    ),
}


def _aot_excluded(job: BuildJob, arch: str) -> bool:
    return any(
        all(job.spec_kwargs.get(k) == v for k, v in entry.items())
        for entry in _AOT_EXCLUDED.get((arch, job.direction), ())
    )


def _spec_is_valid(job: BuildJob, arch: str, dtype: str) -> bool:
    """Would this configuration build at all?

    The grid is deliberately over-generated, and the per-direction validators
    reject most of it on LDS budget, fragment widths, atomic pairing rules and
    so on. Running that check here costs microseconds; discovering it inside
    ``compile_kernel`` costs an LLVM invocation, so the prefilter is the
    difference between a cache build that finishes and one that does not.

    It also applies the grid's own limits on top of spec validity
    (:data:`_AOT_FWD_LDS_BUDGET`, :data:`_AOT_EXCLUDED`).
    """
    if _aot_excluded(job, arch):
        return False
    from kernels.common._conv_implicit_gemm_common import ConvDataSpec

    data = ConvDataSpec(dtype_a=dtype, dtype_b=dtype, dtype_d=dtype)
    problem = _probe_problem(job.direction, job.caps)
    try:
        if job.direction == "fwd":
            from kernels.common.conv_implicit_gemm import (
                ImplicitGemmConvSpec,
                implicit_gemm_lds_bytes,
                is_valid_spec,
            )

            spec = ImplicitGemmConvSpec(problem=problem, data=data, **job.spec_kwargs)
            spec.validate()
            budget = _AOT_FWD_LDS_BUDGET.get(arch)
            if budget is not None and implicit_gemm_lds_bytes(spec)[0] > budget:
                return False
            return is_valid_spec(spec, arch=arch)[0]
        if job.direction == "wgrad":
            from kernels.common.conv_implicit_gemm_wgrad import (
                WgradConvSpec,
                is_valid_wgrad_spec,
            )

            spec = WgradConvSpec(problem=problem, data=data, **job.spec_kwargs)
            spec.validate()
            return is_valid_wgrad_spec(spec, arch=arch)[0]
        from kernels.common.conv_implicit_gemm_dgrad import (
            DgradConvSpec,
            is_valid_dgrad_spec,
        )

        spec = DgradConvSpec(problem=problem, data=data, **job.spec_kwargs)
        spec.validate()
        return is_valid_dgrad_spec(spec, arch=arch)[0]
    except Exception:  # noqa: BLE001 - an invalid combination, not an error
        return False


def count_jobs(
    *,
    dtype: str,
    target,
    directions: Sequence[str],
    split_ks: Sequence[int] = CACHE_WGRAD_SPLIT_KS,
    streamk_modes: Sequence[str] = (),
    streamk_reductions: Sequence[str] = CACHE_WGRAD_STREAMK_REDUCTIONS,
) -> int:
    """Raw candidate count ``kernel_sweep.enumerate_jobs`` walks, before
    dedup/validation.

    Cheap: it runs only the geometry filter and multiplies out the inner axes,
    without building a single identity -- which is where enumeration spends
    its time. Used as the denominator of the enumeration progress.

    Counted per direction rather than once over a shared grid: wgrad walks a
    wider one (see :func:`_grid_axes`).
    """
    da, db, _ = _dtype_triple(dtype)
    mma_family = "wmma" if target.wave_size == 32 else "mma"
    vecs = len(_VECS) * len(_VECS)
    layouts = len(_layout_caps())
    merged = sum(1 for sk in split_ks if sk > 1) * len(_WGRAD_GROUP_MERGES)
    wave_size = target.wave_size
    total = 0
    for direction in directions:
        if direction not in _AOT_ALIAS_PIPELINES:
            continue
        for geo in _geometries(target, mma_family, da, db, direction):
            pipeline = geo[6]
            if not _aot_pipeline(direction, pipeline):
                continue
            if direction == "fwd":
                # As _fwd_jobs: a narrowed cache builds the default epilogue
                # at c = 1, on the lds_xor loops only.
                lean = geo[7] == "default" and 1 not in _VECS
                g6 = geo[:6]
                total += (len(_VECS) if lean else vecs) * sum(
                    sum(len(_fwd_split_atomic_variants(kl, v))
                        for v in _fwd_job_variants(kl, g6, geo[7], caps))
                    for kl in _fwd_k_loops(pipeline, target.gfx)
                    if (not lean or kl.get("lds_xor")) and _fwd_r2_in_scope(kl, *g6)
                    for caps in _layout_caps()
                )
                if _fwd_merged_geometry(geo):
                    total += len(_FWD_GROUP_MERGES) * len(_fwd_merged_k_loops())
            elif direction == "wgrad":
                total += (
                    vecs * sum(len(_wgrad_two_stages(sk)) for sk in split_ks) * layouts
                )
                total += merged
                total += layouts * len(
                    _wgrad_streamk_builds(
                        geo[7], geo[1], dtype, streamk_modes, streamk_reductions
                    )
                )
            elif direction == "dgrad":
                total += vecs * sum(
                    (
                        len(CACHE_DGRAD_TAPK_PREFETCH)
                        + len(CACHE_DGRAD_XHALO_TPF) * len(CACHE_DGRAD_XHALO_KNOBS)
                    )
                    * len(CACHE_DGRAD_B_KPACK)
                    if t
                    else 1
                    for caps in _dgrad_caps()
                    for t in _dgrad_tap_outer_variants(caps, wave_size, pipeline)
                )
                total += len(_dgrad_halo_jobs_for(target.gfx, geo, wave_size, db))
                if _is_patch_geometry(geo):
                    total += len(_VECS)
    return total


def _direction_jobs(
    direction: str,
    arch: str,
    dtype: str,
    target,
    split_ks: Sequence[int],
    max_sub_gemms: int,
    *,
    streamk_modes: Sequence[str] = (),
    streamk_reductions: Sequence[str] = CACHE_WGRAD_STREAMK_REDUCTIONS,
    **kw,
) -> Iterator[BuildJob]:
    jobs = _direction_jobs_all(
        direction,
        arch,
        dtype,
        target,
        split_ks,
        max_sub_gemms,
        streamk_modes=streamk_modes,
        streamk_reductions=streamk_reductions,
        **kw,
    )
    # The group-merged, stream-K, halo and patch generators build 2-D
    # kernels only; drop whatever rank CACHE_SPATIAL_3D did not ask for.
    return (j for j in jobs if j.identity.is_3d in CACHE_SPATIAL_3D)


def _direction_jobs_all(
    direction: str,
    arch: str,
    dtype: str,
    target,
    split_ks: Sequence[int],
    max_sub_gemms: int,
    *,
    streamk_modes: Sequence[str] = (),
    streamk_reductions: Sequence[str] = CACHE_WGRAD_STREAMK_REDUCTIONS,
    **kw,
) -> Iterator[BuildJob]:
    wave_size = target.wave_size
    mma_family = "wmma" if wave_size == 32 else "mma"
    if direction == "fwd":
        return _fwd_jobs(arch, dtype, wave_size, mma_family, target, **kw)
    if direction == "wgrad":
        return _wgrad_jobs(
            arch,
            dtype,
            wave_size,
            mma_family,
            target,
            split_ks,
            streamk_modes=streamk_modes,
            streamk_reductions=streamk_reductions,
            **kw,
        )
    if direction == "dgrad":
        return _dgrad_jobs(
            arch, dtype, wave_size, mma_family, target, max_sub_gemms, **kw
        )
    raise ValueError(f"unknown direction {direction!r}")


def _geometry_list(target, dtype: str, direction: Optional[str] = None) -> List[tuple]:
    mma_family = "wmma" if target.wave_size == 32 else "mma"
    da, db, _ = _dtype_triple(dtype)
    return list(_geometries(target, mma_family, da, db, direction))


# ---------------------------------------------------------------------
# building one job
# ---------------------------------------------------------------------
#
# A build needs a ConvProblem because the spec dataclasses still carry one for
# validation and grid sizing. The *emitted IR does not depend on it* -- that is
# what test_conv_abi.py's shape-invariance cases assert -- so any problem
# that satisfies the spec's own validity rules produces the cacheable binary.
# We use a small canonical one.


def _probe_problem(direction: str, caps: Optional[dict] = None):
    """A canonical problem to build against.

    The emitted IR no longer depends on the extents, so any shape produces the
    same binary -- but it *does* depend on the capability axes, so the probe
    has to match the ones this job is being built for. Getting that wrong is
    not a build error: it silently produces a binary that the cache then
    offers for shapes it cannot compute.
    """
    from kernels.common._conv_implicit_gemm_common import ConvProblem

    caps = caps or {}
    if caps.get("patch"):
        # The patch dgrad's record is the patch embedding's own.
        return ConvProblem(N=1, Hi=16, Wi=16, C=64, K=64, Y=2, X=2, sH=2, sW=2)
    groups = 2 if caps.get("grouped") else 1
    # The tilde dgrad kernel is the strided one: any stride > 1 selects its
    # code path, and the binary then serves every stride at run time.
    stride = 2 if caps.get("tilde") else int(caps.get("stride", 1))
    dilation = int(caps.get("dilation", 1))
    # Keep the dilated filter inside the image so Ho/Wo stay positive.
    extent = 16 + 2 * (dilation - 1)
    channels = 64
    if caps.get("halo_filter"):
        # A dY halo binary bakes its filter and is sized for halo_max_wo: the
        # probe is a same-size output (pad (F-1)/2) that wide.
        f = int(caps["halo_filter"])
        return ConvProblem(
            N=1,
            Hi=max(f, 8),
            Wi=int(caps["halo_max_wo"]),
            C=channels * 2,
            K=channels * 2,
            Y=f,
            X=f,
            pH=(f - 1) // 2,
            pW=(f - 1) // 2,
            groups=2,
        )
    if caps.get("group_merge", 1) > 1:
        # Merging is depthwise only: cpg = kpg = 1, and 64 groups divide by
        # every merge degree.
        groups, channels = 64, 1
    depth = {}
    if caps.get("is_3d"):
        # NDHWC: same rules along depth, so the binary decodes a real D axis.
        depth = dict(Di=extent, Z=3, sD=stride, dD=dilation, pD=dilation)
    return ConvProblem(
        N=1,
        Hi=extent,
        Wi=extent,
        C=channels * groups,
        K=channels * groups,
        Y=3,
        X=3,
        sH=stride,
        sW=stride,
        dH=dilation,
        dW=dilation,
        pH=dilation,
        # An x-halo dgrad binary serves pW == 0 only.
        pW=0 if caps.get("xhalo") else dilation,
        groups=groups,
        **depth,
    )


def kernel_for_job(job: BuildJob, arch: str, dtype: str):
    """Build one job's kernel IR. Returns ``(kernel, meta)``; nothing compiled.

    Raises on an invalid configuration; the caller counts those as skipped
    rather than failed -- the grid is deliberately over-generated and most
    rejections are ordinary spec-validity rules.
    """
    from kernels.common._conv_implicit_gemm_common import ConvDataSpec

    data = ConvDataSpec(dtype_a=dtype, dtype_b=dtype, dtype_d=dtype)
    problem = _probe_problem(job.direction, job.caps)

    if job.direction == "fwd":
        from kernels.common.conv_implicit_gemm import (
            ImplicitGemmConvSpec,
            build_implicit_gemm_conv,
        )

        spec = ImplicitGemmConvSpec(problem=problem, data=data, **job.spec_kwargs)
        kernel = build_implicit_gemm_conv(spec, arch=arch)
    elif job.direction == "wgrad":
        from kernels.common.conv_implicit_gemm_wgrad import (
            WgradConvSpec,
            build_implicit_gemm_conv_wgrad,
        )

        spec = WgradConvSpec(problem=problem, data=data, **job.spec_kwargs)
        kernel = build_implicit_gemm_conv_wgrad(spec, arch=arch)
    elif job.direction == "dgrad":
        from kernels.common.conv_implicit_gemm_dgrad import (
            DgradConvSpec,
            build_implicit_gemm_conv_dgrad,
        )

        spec = DgradConvSpec(problem=problem, data=data, **job.spec_kwargs)
        kernel = build_implicit_gemm_conv_dgrad(spec, arch=arch)
    else:
        raise ValueError(f"unknown direction {job.direction!r}")

    meta = {
        "spec_kernel_name": spec.kernel_name(),
        # launch_block_size only exists where a pipeline appends extra waves
        # (wavelet); wgrad has no such pipeline and exposes block_size alone.
        "block_size": getattr(spec, "launch_block_size", spec.block_size),
    }
    return kernel, meta

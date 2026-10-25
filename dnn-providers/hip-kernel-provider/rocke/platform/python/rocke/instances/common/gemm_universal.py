# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Universal GEMM kernel instance builder.

This is the DSL-side counterpart of CK's
`dispatcher/codegen/unified_gemm_codegen.py`: given the exact same
config schema CK's dispatcher uses to enumerate kernels (see
`dispatcher/codegen/default_config.json`), produce a Python IR
`KernelDef` that lowers to AMDGPU LLVM IR and (via libamd_comgr) to a
HSA code object.

The schema is intentionally identical to CK's so a sweep driver can
walk the same cartesian product CK walks and produce the matching DSL
kernel for every entry. The instance space we cover today is the FP16
RCR family — the dispatcher's hero family for compute-bound large
GEMMs (`preselected_fp16_rcr_compute`).

Geometry conventions (mirror CK Tile's `BlockGemmShape`):

    Block tile:   tile_m  x tile_n  x tile_k
    Warp grid:    warp_m  x warp_n          (warp_k=1)
    MFMA atom:    warp_tile_m x warp_tile_n x warp_tile_k

Each warp owns a `(tile_m / warp_m) x (tile_n / warp_n)` output sub-tile;
that's `mfmas_per_warp_m = (tile_m / warp_m) / warp_tile_m` MFMA tiles
along M and similarly along N. The accumulator length per lane is
`warp_tile_m * warp_tile_n / wave_size` (4 for 16x16; 16 for 32x32 on
wave64).

What this file implements *now*:
  - tile geometries 64..256 x 64..256 x 16..128
  - warp grids 1x1, 2x1, 1x2, 2x2, 4x1, 1x4, 2x4, 4x2, 4x4
  - MFMA atoms 16x16x16, 16x16x32, 32x32x8, 32x32x16 f16
  - pipeline `mem` (single buffer, sync), `compv3` (single buffer +
    scheduler hints), `compv4` (double-buffer LDS + sched_group_barrier
    interleave between MFMA/DS_read/DS_write/VMEM)
  - epilogue `default` (vectorised direct global stores) and
    `cshuffle` (LDS-staged C with the wide-store distribution)
  - layout RCR (row(A), col(B), row(C)) — the production layout

  - the DTL bodies (``direct_to_lds``: ping-pong, alias-scoped ring
    ``dtl_stages``, row interleave ``lds_row_interleave``) and their
    schedule / epilogue levers (see :class:`TraitSpec`); ``persistent_ctas``
    gives a real persistent tile loop (with optional stream-K) on the ring
    body.

What is left out for now (called out explicitly):
  - fp8 input dtypes on the universal body (f16 / bf16 have atoms)
  - ``persistent=True`` is ACCEPTED but is a no-op on the emitted body (the
    flag only adds ``_pers`` to the kernel name, and
    :func:`build_universal_gemm` warns once); it is kept accepted so published
    experiments that build persistent variants keep working. The persistent
    tile loop is the ``persistent_ctas`` lever.
  - padding (`pad_m/n/k`) — the standard configs in default_config.json
    use `pad_*=false`; the dispatcher tries pad-on variants in the
    preselect set; we accept those as input but emit the same body
    (the bounds are statically expected to divide).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace as _dc_replace
from typing import Dict, Iterator, List, Literal, Optional, Sequence, Tuple

from ...core.ir import (
    BF16,
    F16,
    F32,
    I32,
    IRBuilder,
    KernelDef,
    PtrType,
    Type,
    Value,
    VectorType,
    type_bits,
)
from ...helpers.io import io_ir_type
from ...helpers.spec import WarpTileBlockSizeMixin, choose_load_vec
from ...helpers.tensor_view import make_global_view, make_tile_window
from . import gemm_asm_loop as _asm


def _asm_name_tag(tr) -> str:
    """Kernel-name flag of ``TraitSpec.asm_loop`` (only used when it is set)."""
    return _asm.name_tag(tr) if tr.asm_loop else "asm"


# ---------------------------------------------------------------------
# Spec dataclasses
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class TileSpec:
    """Mirror of CK's `TileConfig`."""

    tile_m: int
    tile_n: int
    tile_k: int
    warp_m: int
    warp_n: int
    warp_k: int = 1
    warp_tile_m: int = 32
    warp_tile_n: int = 32
    warp_tile_k: int = 16

    @property
    def mfmas_per_warp_m(self) -> int:
        out, rem = divmod(self.tile_m, self.warp_m * self.warp_tile_m)
        if rem:
            raise ValueError(
                f"tile_m {self.tile_m} not divisible by warp_m * warp_tile_m "
                f"= {self.warp_m * self.warp_tile_m}"
            )
        return out

    @property
    def mfmas_per_warp_n(self) -> int:
        out, rem = divmod(self.tile_n, self.warp_n * self.warp_tile_n)
        if rem:
            raise ValueError(
                f"tile_n {self.tile_n} not divisible by warp_n * warp_tile_n "
                f"= {self.warp_n * self.warp_tile_n}"
            )
        return out

    @property
    def k_atoms_per_tile_k(self) -> int:
        out, rem = divmod(self.tile_k, self.warp_tile_k)
        if rem:
            raise ValueError(
                f"tile_k {self.tile_k} not divisible by warp_tile_k {self.warp_tile_k}"
            )
        return out


Pipeline = Literal["mem", "compv3", "compv4", "wsp3", "wmma_v1"]
Scheduler = Literal["intrawave", "interwave"]
Epilogue = Literal["default", "cshuffle"]


@dataclass(frozen=True)
class TraitSpec:
    """Mirror of CK's `TraitConfig`.

    Extra traits beyond CK's defaults:

    * ``chiplet_swizzle``: opt into the chiplet-aware grid swizzle
      that remaps WGIDs so every contiguous stripe of workgroups
      lands on the same XCD (improves L2 reuse on multi-die GPUs).
      Composes a XCD-round-robin reverse (chunk_size WGs per XCD)
      with a WGM super-tile reordering. The kernel still launches
      with the standard ``(N_tiles, M_tiles[, batch])`` grid; the
      remap happens at kernel entry from the flattened blockIdx.

    * ``waves_per_eu``: AMDGPU occupancy hint emitted as
      ``"amdgpu-waves-per-eu"`` on the kernel attribute list. Default
      ``None`` keeps the LLVM backend's heuristic choice; set to 2
      (or a ``(min, max)`` tuple) when targeting two workgroups per CU.
    """

    pipeline: Pipeline = "compv4"
    scheduler: Scheduler = "intrawave"
    epilogue: Epilogue = "cshuffle"
    pad_m: bool = False
    pad_n: bool = False
    pad_k: bool = False
    persistent: bool = False
    chiplet_swizzle: bool = False
    chiplet_wgm: int = 8
    chiplet_num_xcds: int = 8
    chiplet_chunk_size: int = 64
    waves_per_eu: Optional[int] = None
    # P40: when True, the kernel expects the B operand pre-shuffled by
    # the host into the layout :func:`rocke.helpers.preshuffle.
    # host_preshuffle_layout` produces, and the per-lane B-load uses
    # :func:`emit_preshuffleb_offset` to compute the byte offset
    # (one ``buffer_load_dwordx4`` per K-tile vs the per-K-element
    # strided scalar loads of the column-major path). The flag-free
    # default keeps the canonical strided-scalar load.
    preshuffle_b: bool = False
    # P41: DirectToLDS (DTLA/DTLB). When True, the global -> LDS load
    # phase emits ``raw.ptr.buffer.load.lds`` (one instruction per chunk;
    # hardware writes the dword payload straight into LDS) instead of
    # the round-trip ``buffer_load -> VGPR -> ds_write_b128`` pair the
    # default path uses. Matches Tensile's ``DTLA1_DTLB1`` token in
    # tuned kernels like ``MT16x16x512`` on gfx950. Constraints inherited
    # from :class:`AsyncTileLoader`: per-lane payload must be 4 / 12 / 16
    # bytes (dwords in {1,3,4}; dwords=2 is rejected by the intrinsic).
    direct_to_lds: bool = False
    # Per-operand cache hints when direct_to_lds is True. Default mirrors
    # rocBLAS skinny-M behaviour: A is small (M*K halves) and reused
    # across CTAs through L2 -> CACHE_ALL; B is the 32 MiB weight matrix,
    # one-shot streamed -> CACHE_STREAM.
    dtl_cache_a: int = 0  # CACHE_ALL
    dtl_cache_b: int = 2  # CACHE_STREAM
    # Prefetch ping-pong on top of direct_to_lds. Allocates two LDS buffers
    # per operand and issues next-iter DTLA loads while current-iter MFMAs
    # run, then waits with ``s_waitcnt vmcnt(loads_in_flight)`` so only the
    # current tile's loads have to complete before MFMAs start. Requires
    # ``direct_to_lds=True``. Doubles LDS usage (knocks 2 WGs/CU down to 1
    # at typical tile sizes) but hides the global-load latency that
    # otherwise serialises every K-tile.
    dtl_prefetch: bool = False
    # MoE active-tile early-exit. When True (only honored in
    # ``batched=True`` mode), the kernel takes two extra args
    # (``SortedTokenIds: ptr<i32>``, ``slot_size: i32``) and at CTA
    # entry computes a wave-uniform ``do_work`` predicate from
    # ``SortedTokenIds[block_id_z * slot_size + block_id_y * tile_m]``.
    # The K-loop + epilogue are wrapped in ``scf.if(do_work)`` so an
    # inactive expert tile (``first_row_token == -1``) skips all
    # MFMAs, LDS reads, and HBM stores. Enables CK Tile-style
    # "expert-by-expert" MoE dispatch over a static grid that's
    # padded to the worst-case ``experts`` size.
    active_tile_skip: bool = False
    # LDS K-padding: extra columns added to each AB LDS row so the per-row
    # byte stride is not a power-of-two multiple of the bank count, breaking
    # the stride-based bank-conflict pattern on the MFMA ds_reads (gfx950 has
    # 64 banks; an unpadded block_k=32 f16 row = 64 B stride aliases banks
    # every 4 rows). Default 0 = no change (golden-gate-safe). Use a multiple
    # of the load vector width (e.g. 16 halves) so the wide vector ds_read /
    # ds_write stay naturally aligned -- pad=4 breaks 16-byte alignment and
    # regresses hard. Measured NEUTRAL-to-NEGATIVE on square fp16 (the extra
    # LDS costs occupancy and outweighs the conflict reduction) -- prefer
    # ``lds_swizzle`` (zero LDS cost). Kept as an opt-in knob for shapes/dtypes
    # where the trade may pay. Non-DTL only (the direct-to-LDS path computes
    # flat byte offsets that assume a contiguous block_k stride).
    lds_k_pad: int = 0
    # LDS XOR swizzle (st_16x32-style): toggle the 32-byte column group on
    # rows with bit 3 set so consecutive-row MFMA ds_reads hit different banks.
    # ZERO LDS overhead (unlike lds_k_pad). Applied identically to the LDS
    # store and ds_read columns -> bit-exact. Measured ~+3% on square fp16/bf16.
    # Default off (golden-gate-safe). Non-DTL only; 2-byte dtypes.
    lds_swizzle: bool = False
    # Split-K over the production universal body. When > 1, the kernel:
    #   * takes a third grid dim ``block_id_z`` in [0, split_k) selecting
    #     a K-slice ``[z*ks, (z+1)*ks)`` (``ks = K // split_k``) for the
    #     CTA's K-loop, instead of the full ``[0, K)``;
    #   * replaces the ``C`` output param with an f32 workspace
    #     ``Cf32[M, N]`` and the epilogue atomic-adds each warp's f32
    #     accumulator into it (instead of the direct/cshuffle bf16 store).
    # The grid becomes ``(N_tiles, M_tiles, split_k)``. The caller must
    # zero-init the f32 workspace before launch and cast it to the target
    # dtype afterwards (Python-side finalisation, matching the v1 StreamK
    # atomic-strategy contract). This re-uses the fast vectorized +
    # LDS-double-buffered ``compv4`` load/MFMA inner verbatim; only the
    # K-loop bound and the epilogue change. ``split_k == 1`` (default)
    # keeps the canonical single-K-pass body byte-identical.
    split_k: int = 1
    # compv4/compv3 schedule directives: the per-cluster s_setprio(1/0) +
    # sched_barrier(0) fences (_mma_cluster) AND the two-stage
    # sched_group_barrier HotLoop interleave (_emit_hotloop_schedule). On gfx950
    # these OVER-CONSTRAIN comgr's backend scheduler -- removing them lets the
    # hardware scheduler pack MFMAs tighter. Measured +1.9-2.5% (MfmaUtil
    # 63->68%) on square fp16/bf16 GEMM, 200/200 GPU-event-timed cycles, relerr=0
    # (see optimization/utilities/skills/empirical-case-studies.md, Case Study 7).
    # None (default): arch-resolved -> hints OFF on gfx950 (take the uplift), ON
    # elsewhere (preserve the historical emission). True/False forces the choice.
    emit_sched_hints: Optional[bool] = None
    # cshuffle epilogue LDS aliasing. By default (False) the smem-pool packer
    # aliases the cshuffle C tile onto the A/B staging bytes (pool = max(ab, c)),
    # which needs a barrier between the main loop's last A/B read and the first
    # C write (the epilogue "step-0 reuse barrier"). Setting True gives C its own
    # LDS bytes (pool = ab + c) and elides that barrier -> lower latency for
    # small tiles, at the cost of more LDS (lower occupancy) for large tiles.
    # Only affects the cshuffle epilogue; False keeps byte-identical output.
    cshuffle_no_alias: bool = False
    # compv4 (VGPR-staged, non-DTL) AB LDS double buffer, made explicit.
    # ``None`` (default) keeps the historical automatic rule byte-identical:
    # double-buffer only with the direct epilogue AND only when the doubled AB
    # still fits 2 WG/CU (``4 * ab_single <= lds_cap``). That rule silently
    # single-buffers every compv4 tile >= 256x128x64 f16 on gfx950 and every
    # compv4 + cshuffle tile (the shipped dispatcher pick) although the cshuffle
    # C tile aliases the A/B bytes. ``True`` forces the ping-pong double buffer
    # (also with cshuffle; rejected by ``is_valid_spec`` when the doubled AB does
    # not fit one WG), ``False`` forces the single buffer. Only meaningful for
    # ``pipeline="compv4"`` without ``direct_to_lds`` (DTL has ``dtl_prefetch``);
    # setting it anywhere else is rejected rather than silently ignored.
    ab_double_buffer: Optional[bool] = None
    # MFMA-cluster priority bookends (opt-in, default off). When True, every
    # K-atom step's whole ``mfmas_m x mfmas_n`` MFMA cluster is emitted as
    # ``sched_barrier(0); s_setprio(1); <cluster>; s_setprio(0);
    # sched_barrier(0)`` (CK Tile's Interwave bookends) INSTEAD of the default
    # per-m-row bracketing (swizzle + sched-hints mode) / no bracketing. It is a
    # separate knob rather than the meaning of ``scheduler="interwave"``: that
    # value has always emitted the intrawave body, and shipped, tuned kernels
    # (qwen3_30b_a3b decode, gemm_perf_skinny_decode) set it, so giving it new
    # emission would silently change them. MFMA (wave64 CDNA) only.
    interwave_setprio: bool = False
    # ---- DTL ring levers (all default-off; the defaults keep every existing
    # kernel byte-identical; mirrored in the C++ engine) ----
    # ``dtl_stages``: alias-scoped DTL LDS ring. 0 (default) keeps the
    # historical ``dtl_prefetch`` body (one ``[2*block_m, block_k]`` allocation
    # per operand, runtime parity). N >= 2 gives every stage its own A and B
    # smem allocation, sets the kernel attr ``lds_alias_scopes`` and unrolls the
    # K-loop by N so every LDS address is static: SIInsertWaitcnts then no
    # longer has to assume that a ds_read of the current stage may alias the
    # in-flight next-stage ``buffer_load ... lds`` (the compiler-inserted
    # ``s_waitcnt vmcnt(0)`` in front of the first ds_read, a large share of the
    # ping-pong body's stall). N = 2 is the ping-pong (one tile in flight under
    # the MFMAs);
    # N >= 3 keeps N-1 tiles in flight with a partial ``vmcnt((N-2)*loads)``
    # wait (two-tiles-deep prefetch at N = 3). Requires direct_to_lds +
    # dtl_prefetch; LDS = N x AB.
    dtl_stages: int = 0
    # ``dtl_issue_spread`` (dtl_stages >= 2): issue the next tile's DTL loads
    # one at a time between the MFMAs of the current tile instead of as one
    # burst right after the barrier.
    dtl_issue_spread: bool = False
    # ``dtl_frag_prefetch`` (dtl_stages >= 2): register-prefetch the next K
    # atom's operand fragments one atom ahead, and move the per-tile wait +
    # barrier (+ the next DTL issue and the next tile's first fragment reads)
    # INSIDE the MFMA stream, in front of the last K atom's MFMA cluster, so
    # the rendezvous and the ds_read latency of the next tile overlap matrix
    # work instead of draining the pipe at the tile boundary.
    dtl_frag_prefetch: bool = False
    # ``dtl_stream_barrier`` (dtl_stages >= 2): the in-stream wait + barrier +
    # DTL issue of ``dtl_frag_prefetch`` WITHOUT the fragment register
    # prefetch (no extra VGPRs; fits the 128-VGPR 16-wave body). Implied by
    # ``dtl_frag_prefetch``.
    dtl_stream_barrier: bool = False
    # ``mfma_swap_ab``: feed the MFMA (B, A) instead of (A, B) so the
    # accumulator holds C^T: each lane then owns 4 consecutive N columns of
    # one row and the direct epilogue writes 8-byte vector stores instead of
    # 2-byte scalar stores (MFMA, default epilogue, no pad_n / split-K).
    mfma_swap_ab: bool = False
    # ``iglp_opt``: emit ``llvm.amdgcn.iglp.opt(level)`` once per K tile of the
    # dtl_stages body (None = off). It combines with the staged body's other
    # scheduling fences: those are ``sched_barrier(0)`` + ``s_setprio``
    # (``interwave_setprio``, or the per-m-row sched hints), never a
    # ``sched_group_barrier`` (the HotLoop interleave is not emitted by the
    # staged body), and LLVM's IGroupLP drops an ``iglp_opt`` only when a
    # ``sched_group_barrier`` shares its scheduling region. Checked on the gfx950
    # ISA: iglp_opt=1 reorders the first MFMA region's ds_reads both with
    # ``interwave_setprio`` and with ``emit_sched_hints=True``.
    iglp_opt: Optional[int] = None
    # ``persistent_ctas`` (dtl_stages >= 2, default epilogue, non-batched,
    # split_k == 1; 0 = off): a persistent tile loop. Workgroup w < P owns the
    # output tiles w, w + P, ... (w >= P exits, so the standard grid stays
    # correct; a launcher should use ``persistent_grid`` = min(P, tiles)
    # workgroups instead). The next tile's DTL prologue is issued before the
    # current tile's epilogue stores, overlapping the store with the next
    # tile's global-load latency. P = the CU count gives one resident
    # workgroup per CU. With ``chiplet_swizzle`` P must be a multiple of
    # ``chiplet_num_xcds`` (workgroup w always runs on XCD w mod #XCDs, so
    # only then do its tiles w + kP stay on the XCD the swizzle assigned).
    persistent_ctas: int = 0
    # ---- AGPR accumulator / MFMA interleave levers (default-off; defaults keep
    # every existing kernel byte-identical; mirrored in the C++ engine) ----
    # ``acc_agpr`` (MFMA, CDNA): keep the MFMA accumulators in AGPRs. Sets the
    # kernel attr ``amdgpu-agpr-alloc`` = (n, n) with n = the per-lane
    # accumulator registers (MFMA selection then uses the AGPR dst / srcC
    # form) and seeds every accumulator from an opaque zero
    # (``asm "", "=a,0"(zeroinitializer)``) instead of the zero constant. The
    # constant seed makes AMDGPUCodeGenPrepare break the loop-carried
    # ``<16 x float>`` phis into scalar phis, and the per-element copies that
    # reassemble them around every MFMA (v_accvgpr_read / write / mov, AGPR
    # spills) are what kept the 4-wave 2x2 32x32x16 body (128x128 per wave,
    # 256 accumulator registers) from compiling cleanly. With it that body's
    # K loop holds only MFMAs, ds_reads and DTL issues.
    acc_agpr: bool = False
    # ``mfma_interleave`` (dtl_frag_prefetch body): emit every next-K-atom
    # fragment ds_read and every DTL issue between two MFMAs, spread evenly
    # over the K atom's MFMA cluster (in MFMA use order), each MFMA followed
    # by ``sched_barrier(0)`` so the backend keeps exactly this order -- the
    # instruction-level interleave of a hand-scheduled (Tensile-style) loop,
    # instead of bursts of reads / DMAs that the scheduler places.
    mfma_interleave: bool = False
    # ``acc_agpr_pin`` (acc_agpr): also route every MFMA result through an
    # ``asm "", "=a,0"`` so the accumulator is pinned to the AGPR file after
    # each MFMA, not only at the loop entry. A knob for backends that treat the
    # MFMA operands as AGPR-or-VGPR and mix the two files.
    acc_agpr_pin: bool = False
    # ``ilv_dtl_every`` (mfma_interleave): 0 (default) spreads a cluster's DTL
    # issues evenly over its MFMAs; k >= 1 places them one every k MFMAs from
    # the first MFMA on (k = 1: back to back, one behind each MFMA).
    ilv_dtl_every: int = 0
    # ``ilv_barrier_after`` (mfma_interleave): issue the first k MFMAs of the
    # K tile's last cluster (they read fragments already in registers) BEFORE
    # its ``s_waitcnt`` + ``s_barrier``, so the barrier wait overlaps matrix
    # work instead of draining the MFMA pipe; that cluster's ds_reads and DTL
    # issues spread over the remaining MFMAs. 0 (default): barrier first.
    ilv_barrier_after: int = 0
    # ``ilv_merge`` (mfma_interleave): place the ds_reads and the DTL issues on
    # ONE schedule (merged by their position in their own list) so that no
    # two of them share an MFMA slot while slots are left; the default places
    # each list on its own even spread, which stacks a read and a DTL issue
    # (+ its SALU) behind the same MFMA whenever their counts share factors.
    ilv_merge: bool = False
    # ``lds_swizzle_r`` (lds_swizzle): the row shift R of the XOR key
    # ``col ^= ((row >> R) % 2^W) << L``; -1 (default) keeps the geometry rule
    # of :func:`lds_swizzle_params` (R = 0). That rule is conflict-free for the
    # 16x16x32 atom at tile_k 64 but leaves 2-way ds_read_b128 conflicts on
    # gfx950 for 16x16x32 at tile_k 32 (R = 1 clears them) and for 32x32x16 at
    # tile_k 64 (R = 1) / 4-way at tile_k 32 (R = 2), per rocke.analysis.lds.
    lds_swizzle_r: int = -1
    # ``dtl_hoist`` (dtl_stages >= 2): build the buffer resource and the
    # wave-uniform ``soffset`` of a K tile's DTL loads once per operand (when
    # the tile's issue set is created) instead of once per DTL instruction;
    # fewer SALU moves between the loads.
    dtl_hoist: bool = False
    # ---- persistent / stream-K / pad levers (default-off; defaults keep every
    # existing kernel byte-identical; streamk, dtl_tight_rsrc and pad_fast_tile
    # are mirrored in the C++ engine, the others are Python-engine only, see
    # core.backend._PYTHON_ONLY_GEMM_LEVERS) ----
    # ``streamk`` (persistent_ctas = P): stream-K tail balancing in the
    # persistent tile loop (CK Tile's "DP + 2-tile SK" partition, resolved in
    # the kernel from M / N / K and P). When the output tiles do not divide P,
    # one full wave of tiles plus the remainder are split by MAC iterations
    # (K tiles) evenly over the P workgroups; the rest run data-parallel. A
    # workgroup that covers the first K tile of an output tile owns it: it adds
    # the partial sums of the later contributors (linear fixup, one i32 ready
    # flag per (workgroup, wave), reset by the owner after use, so the flag
    # table only has to be zeroed once when it is allocated) and stores C.
    # Extra kernel args ``SkFlags`` (i32) and ``SkPart`` (f32), sized by
    # :func:`streamk_workspace_bytes`. Launch exactly P workgroups, all
    # co-resident (a waiting owner spins on higher-index workgroups):
    # :func:`is_valid_spec` rejects P above the static co-residency bound
    # :func:`streamk_resident_ctas` (CUs x workgroups per CU at the LDS use and
    # the worst-case register budget); a launcher on a partitioned (CPX / DPX)
    # device or with fewer CUs must also check P against the live device
    # (:func:`streamk_launch_error`).
    streamk: bool = False
    # ``sk_xcd_local`` (streamk + chiplet_swizzle): number the
    # stream-K ranks XCD-major (workgroup w runs on XCD w mod chiplet_num_xcds)
    # so a tile's owner and its partners share one L2. Same-XCD hand-offs then
    # use a workgroup-scope fence + vmcnt(0) instead of the agent-scope L2
    # write-back / invalidate (which also evicts every other workgroup's A / B
    # lines on that XCD); the X - 1 cross-XCD hand-offs keep the agent fences.
    sk_xcd_local: bool = False
    # ``dtl_tight_rsrc`` (dtl_stages >= 2): build every DTL buffer resource
    # from the tile's own origin (``ptr + (tile0 + k) * 2``, soffset 0) with
    # ``num_records`` = the operand bytes left from there, so the hardware
    # range check returns zeros for rows past M (A) / N (B) instead of reading
    # past the allocation. Makes ``pad_m`` / ``pad_n`` safe on the ring body.
    dtl_tight_rsrc: bool = False
    # ``pad_fast_tile`` (pad_m / pad_n, default epilogue): branch once per
    # output tile on whether it lies fully inside M x N; full tiles run the
    # unguarded epilogue (no per-element bounds compares), only the edge tiles
    # the guarded one.
    pad_fast_tile: bool = False
    # ``dtl_tight_ragged_only`` (dtl_tight_rsrc, ragged shapes):
    # exact DTL bounds only on the operand whose row dimension is padded (A
    # with pad_m, B with pad_n); the other operand takes the unbounded
    # base-origin resource (the ``dtl_soff0`` form: origin in the base, inline
    # soffset 0, num_records = the big constant or 0 past K), which needs no
    # per-tile bytes-left SALU. Interior and edge tiles run the same code.
    dtl_tight_ragged_only: bool = False
    # ``k_stagger_rc`` (direct_to_lds, dtl_stages >= 2): a StaggerU-style K
    # start offset. Output tile (r, c) walks its K tiles starting at tile
    # ((r + c) mod k_stagger_rc) mod T and wraps around (power of two, 0 = off), so
    # the workgroups that share an A row block (same r) or a B row block (same
    # c) and run at the same time read different K columns -- different HBM
    # channels / L2 sets -- instead of all streaming the same addresses in
    # lock step. Changes only the f32 accumulation order. The range predicate
    # of each DTL stays on the unwrapped index (prologue / tail issues past K
    # stay out of range).
    k_stagger_rc: int = 0
    # ---- early-release interleave levers (default-off; defaults keep every
    # existing kernel byte-identical; mirrored in the C++ engine) ----
    # ``ilv_early_release`` (mfma_interleave, >= 2 K atoms per tile): split the
    # per-tile sync point in two (release / acquire, as in hipBLASLt's
    # 256x256x64 loop). The
    # last K atom's fragments are read during the previous atom, so the stage
    # is free once those reads land: a RELEASE point (``s_waitcnt lgkmcnt(0);
    # s_barrier`` -- no vmcnt wait) at MFMA ``ilv_barrier_after`` of the last
    # atom, followed by the DTL issues of tile i+S into that stage, then an
    # ACQUIRE point (``s_waitcnt vmcnt((S-1)*L); s_barrier``, tile i+1 landed)
    # at MFMA ``ilv_acquire_at`` followed by the next tile's first fragment
    # reads. The DTL of tile i+S is issued before (not after) the wait for tile
    # i+1, so the vmcnt wait leaves one more tile in flight: with S = 2 the load
    # latency hidden grows from one tile to about one and a half.
    # 1 = that split inside the last atom (release at MFMA ilv_barrier_after,
    # acquire at ilv_acquire_at); 2 = release in the second-to-last atom: the
    # last atom's
    # fragments are read at the start of the atom BEFORE it, the release sits
    # at MFMA ``ilv_release_at`` of that atom (DTL issues spread over the rest
    # of it), the acquire at MFMA ``ilv_acquire_at`` of the last atom (auto
    # for both: 1.5x the atom's fragment reads); 3 = mode 2 with the DTL
    # issues spread over [release, atom end) + [0, acquire) of the last atom
    # (auto acquire: the atom's MFMAs minus 1.5x its fragment reads).
    # 0 (default): off.
    ilv_early_release: int = 0
    # ``ilv_release_at`` (ilv_early_release = 2): see above; 0 = auto.
    ilv_release_at: int = 0
    # ``acc_out_vgpr`` (acc_agpr): the epilogue's per-atom ordered hand-out is
    # ``asm "", "=v,0"`` (the accumulator leaves the AGPRs straight into
    # VGPRs) instead of ``"=a,0"`` (an AGPR -> AGPR move per register, plus the
    # hazard s_nops before its v_accvgpr_read).
    acc_out_vgpr: bool = False
    # ``ilv_read_span`` / ``ilv_dtl_span`` (mfma_interleave, percent, 0 = off):
    # spread a cluster's fragment ds_reads over the first ``ilv_read_span`` %
    # of its MFMAs and its DTL issues over the first ``ilv_dtl_span`` %
    # (independently; a read and an issue may share a slot) instead of an
    # even (or merged) spread over the whole cluster -- the reads land well
    # before the next atom's first MFMA and the DMAs leave earlier.
    ilv_read_span: int = 0
    ilv_dtl_span: int = 0
    # ``ilv_acquire_at`` (ilv_early_release): MFMA index of the last K atom at
    # which the acquire wait + barrier sits (ilv_barrier_after < k < the
    # atom's MFMAs); 0 (default) = auto: the atom's MFMAs minus twice its
    # fragment reads, at least ilv_barrier_after + 1.
    ilv_acquire_at: int = 0
    # ---- early stage-free / split-wait levers (default-off; defaults keep
    # every existing kernel byte-identical; Python-engine only) ----
    # ``ilv_early_free=Q`` (mfma_interleave, >= 2 K atoms per tile): free the
    # tile's LDS stage early and decouple the two rendezvous of the in-stream
    # barrier (a PGR2-style order). The LAST K atom's fragment ds_reads are
    # packed into the first Q - 4 MFMA slots of the second-to-last atom; at slot
    # Q a ``s_waitcnt lgkmcnt(0); s_barrier`` (every wave has read the whole
    # stage) is followed by the DTL issues of tile i+S into that stage, spread
    # over the remaining slots of the atom. The last atom keeps its wait +
    # barrier for tile i+1 (now ``vmcnt`` = the in-flight loads of tile i+S, a
    # partial wait) and reads the next tile's first fragments. The DTL issues
    # then no longer share MFMA slots with the next-tile ds_reads, and each
    # load gets about half a tile more lead time. 0 (default): off.
    ilv_early_free: int = 0
    # ``ilv_ef_spread`` (ilv_early_free): spread the DTL issues of tile i+S
    # evenly from the stage-free barrier to the END of the tile (over both the
    # second-to-last and the last K atom) instead of only over the rest of the
    # second-to-last atom, so the four waves of a CU keep the texture
    # addresser at about half load instead of saturating it for half a tile.
    # The last atom's wait for tile i+1 then keeps only the issues placed
    # before it in flight.
    ilv_ef_spread: bool = False
    # ``ilv_split_wait=Q`` (mfma_interleave, not with ilv_early_free): split
    # the last K atom's wait for tile i+1 by operand. The first rendezvous
    # (slot ``ilv_barrier_after``) waits only for tile i+1's A loads
    # (``vmcnt`` keeps its B loads in flight) and is followed by the A
    # fragment reads; at MFMA slot Q a second ``vmcnt`` + barrier waits for the
    # B loads (keeping the tile i+S loads issued so far in flight) before the
    # B fragment reads. The B loads -- issued last -- gain Q MFMAs of lead time
    # (two partial-vmcnt barriers per tile). 0 (default): off.
    ilv_split_wait: int = 0
    # ---- row-interleave LDS layout and slot-schedule levers (Python-engine
    # only, default-off; the defaults keep every kernel byte-identical) ----
    # ``lds_row_interleave`` (dtl_stages >= 2, gfx950 16-byte DTL, 16-bit data,
    # tile_k 64 = one 128-byte LDS row, 16x16x32 atom, mfmas_m / mfmas_n
    # multiples of 8, no ``lds_swizzle``): the LDS tile layout hipBLASLt uses
    # for its gfx950 256x256x64 kernels. LDS row R of a warp's 16*mfmas block holds
    # the global row ``(R % mfmas) * 16 + R // mfmas`` (the DTL lanes pick their
    # global row; the MFMA / epilogue mapping is unchanged), the rows are
    # unswizzled and every 1 KiB (8 rows = one DTL instruction per wave) is
    # followed by 16 bytes of padding. A lane's fragment of atom mi is then
    # ``base(lane) + mi * 128 (+ pad) + kk * 64`` bytes: every ds_read of a K
    # tile shares ONE address VGPR per stage / operand (immediate offsets, no
    # per-read XOR / v_add), and ds_read_b128 is bank-conflict free (the
    # 16-byte pad shifts consecutive lane rows by one 16-byte bank group).
    lds_row_interleave: bool = False
    # ``rilv_pad`` (lds_row_interleave; bytes, 16 / 32 / 48 / 64): the pad after
    # every 1 KiB LDS block (8 rows of 128 bytes). 16 shifts consecutive lane
    # rows by one 16-byte bank group, so the 8 rows of a ds_read_b128 lane
    # group fill banks 0..31 and the next K group (+16 bytes) of the same
    # pass overlaps the next row: bank conflicts on every ds_read_b128 on
    # gfx950. 32 (the hipBLASLt layout) gives rows a 32-byte bank step, so the
    # second K group lands in the gaps.
    rilv_pad: int = 16
    # ``hb_sched`` (dtl_frag_prefetch + mfma_interleave): a slot K-tile
    # schedule in hipBLASLt's order. One MFMA stream per K tile with every
    # other instruction pinned
    # to an MFMA slot (sched_barrier(0) after each MFMA):
    #   * the next K atom's fragments are read front-loaded, one ds_read every
    #     ``hb_read_every`` MFMAs from the atom's first MFMA;
    #   * RELEASE: after the last such read (+ a gap), ``s_waitcnt lgkmcnt(0);
    #     s_barrier`` -- this K tile's stage is fully read -- and the DTL issues
    #     of tile i+S into it follow, spread over the rest of the tile
    #     (``hb_dtl_every`` MFMAs apart, 0 = evenly);
    #   * ACQUIRE: a few MFMAs into the last K atom, ``s_waitcnt vmcnt(n)``
    #     (n = the loads issued after tile i+1's) + ``s_barrier``, then the next
    #     tile's first-atom fragments are read from its stage, spread.
    # The release / acquire split gives the DTL of tile i+S (issued early in
    # tile i) more than one tile of latency budget, and the stage is freed as
    # soon as its last fragment is in registers (2 barriers per K tile, a
    # PGR2 + PLR1 loop). ``hb_rel_slot`` / ``hb_acq_slot``: the
    # MFMA index (within the K tile) after which the release / acquire sits
    # (-1 = auto: release 4 MFMAs after the last intra-tile read, acquire after
    # the 4th MFMA of the last K atom).
    hb_sched: bool = False
    hb_rel_slot: int = -1
    hb_acq_slot: int = -1
    hb_read_every: int = 1
    hb_dtl_every: int = 0
    # ``hb_acq_b_slot`` (hb_sched; -1 = one acquire for both operands): split
    # the acquire per operand -- wait for the next stage's A loads only at
    # ``hb_acq_slot`` (vmcnt also leaves the B loads in flight) and read the A
    # fragments, then wait for B at this slot and read the B fragments, so
    # the later-issued B loads get more latency budget.
    hb_acq_b_slot: int = -1
    # ``hb_split_rel`` (hb_sched, >= 2 K atoms): per-operand release -- read
    # all next-atom A fragments first, then
    # all B fragments; release (lgkmcnt(0) + s_barrier) A ``hb_rel_gap`` MFMAs
    # after its last read and start its DTL issues right away (one every
    # ``hb_dtl_every`` MFMAs, 3 when 0), then release B after its reads and
    # issue B. The A loads leave about one tile earlier than with one release.
    hb_split_rel: bool = False
    # ``hb_rel_gap`` (hb_sched): MFMAs between the last fragment read and the
    # (auto-placed) release.
    hb_rel_gap: int = 4
    # ---- SIA-style schedule and vector-epilogue levers (Python-engine only,
    # default-off) ----
    # ``sia_sched`` (dtl_frag_prefetch + mfma_interleave + lds_row_interleave,
    # 2 K atoms per tile): a Tensile-SIA3-style K-tile schedule family. One MFMA
    # stream per K tile, every other instruction pinned after an MFMA slot
    # (sched_barrier(0) after each MFMA). With every sub-lever off it is the
    # hb_sched shape (A-first next-atom reads every ``sia_read_every`` MFMAs,
    # one release lgkmcnt(0) + s_barrier ``sia_rel_gap`` MFMAs after the last
    # read, DTL issues spread evenly over the rest of the tile, one acquire
    # vmcnt(n) + s_barrier, then the next tile's first-atom reads). Sub-levers
    # (each its own knob):
    #   * ``sia_b_first``: operand order B before A everywhere -- next-atom
    #     reads, releases, DTL issues (also in the prologue), acquires. B is
    #     the MFMA-inner operand (all of its fragments are used by the first
    #     mfmas_n MFMAs of an atom), so reading it first gives the A fragments
    #     (first used at MFMA mi * mfmas_n) the remaining lead.
    #   * ``sia_plr``: PLR1 lead -- the acquire sits ``sia_acq_gap`` MFMAs into
    #     the last K atom so the next tile's first-atom fragments are read a
    #     full atom before their use; off: the acquire sits half an atom
    #     later (reads denser, shorter lead).
    #   * ``sia_single_lgkm``: one explicit ``s_waitcnt lgkmcnt(0)`` after the
    #     tile's first MFMA, before its first next-atom read, so the compiler
    #     emits no per-MFMA lgkmcnt(N) for the fragments read in the previous
    #     tile (the counter then holds only this tile's reads).
    #   * ``sia_split_rel``: per-operand release (two barriers): the first
    #     operand's reads, lgkmcnt(0) + s_barrier, its DTL issues start while
    #     the second operand is read, then lgkmcnt(0) + s_barrier for it.
    #   * ``sia_split_acq``: per-operand acquire (two partial vmcnt waits +
    #     barriers): first operand's stage, its reads, later the second's.
    #   * ``sia_dtl_spread``: DTL issues never share an MFMA slot with a
    #     ds_read or a sync and are spread evenly over the free slots after
    #     their operand's release, at least ``sia_dtl_stride`` MFMAs apart
    #     when room allows; off: evenly over every slot after the release.
    sia_sched: bool = False
    sia_b_first: bool = False
    sia_plr: bool = False
    sia_single_lgkm: bool = False
    sia_split_rel: bool = False
    sia_split_acq: bool = False
    sia_dtl_spread: bool = False
    sia_read_every: int = 2
    sia_rel_gap: int = 4
    sia_acq_gap: int = 3
    sia_dtl_stride: int = 3
    # ``sia_tensile`` (sia_sched, 8x8 MFMAs per atom, 8 + 8 DTL issues per
    # tile): a fixed slot table in the Tensile SIA3 bf16 256x256x64 order
    # instead of the computed one; sia_b_first picks the
    # operand order, sia_single_lgkm still adds its wait, the other position
    # knobs are ignored.
    sia_tensile: bool = False
    # ``sia_use_order`` (sia_sched, no split release / acquire): read every
    # atom's fragments in first-use order (A0, B0..B[n-1], A1..A[m-1]) instead
    # of operand by operand.
    sia_use_order: bool = False
    # ``sia_merge_sync`` (sia_sched, no split release / acquire): no separate
    # release; the acquire waits vmcnt(n) AND lgkmcnt(0) and its one barrier
    # also frees this tile's stage (the ilv_merge sync point), the DTL issues
    # follow it. With it ``sia_acq_gap`` may be negative (down to -mfmas_n):
    # the sync then sits that many MFMAs before the last K atom.
    sia_merge_sync: bool = False
    # ``rilv_vec_epilogue`` (lds_row_interleave, default epilogue, no pad_m /
    # pad_n, split_k 1): a column-major C mapping instead of a row permutation in
    # the DTL. The DTL rows stay linear and the MFMA atom (mi, lane row r)
    # computes C row ``wm + r * mfmas_m + mi`` (column ``wn + c * mfmas_n +
    # ni``), so for a fixed (mi, slot) a lane holds mfmas_n CONSECUTIVE
    # columns: the epilogue writes 16-byte vector stores straight from the
    # accumulators (no LDS round trip, no 2-byte scalar stores).
    rilv_vec_epilogue: bool = False
    # ``rve_store_nt`` (rilv_vec_epilogue): write C with the streaming
    # (non-temporal) hint (the NT store bits).
    rve_store_nt: bool = False
    # ---- DTL issue-economy levers (Python-engine only, default-off) ----
    # ``dtl_rsrc_word3`` (direct_to_lds, dtl_stages >= 2): DWORD3 (flags) of
    # the DTL buffer resources; 0 (default) = the target default (CDNA
    # 0x00027000: DATA_FORMAT 32, NUM_FORMAT FLOAT). hipBLASLt / CK use
    # 0x00020000 (NUM_FORMAT 0) on gfx94x / gfx950.
    dtl_rsrc_word3: int = 0
    # ``dtl_soff0`` (dtl_hoist, no dtl_tight_rsrc): put the per-tile global
    # origin into the DTL resource BASE (64-bit SGPR add per operand and
    # tile) and issue the DTLs with an inline ``soffset`` of 0
    # (``buffer_load_dwordx4 v, s[..], 0 offen lds``). With an SGPR soffset
    # every DTL occupies the VMEM issue port for 2 cycles instead of 1 on
    # gfx950 (per the SQ VMEM issue counters).
    # Out-of-range tiles keep num_records 0 (every lane out of range).
    dtl_soff0: bool = False
    # ``kloop_unroll`` (dtl_stages >= 2, no pingpong): the staged K loop body
    # covers kloop_unroll * dtl_stages tiles (default 1 = dtl_stages), which
    # halves the per-tile loop overhead (counter / compare / branch) at 2.
    # The tail (T % (kloop_unroll * S) tiles) then also issues its DTLs
    # (tile i + S may still be in range), with the same partial waits as the
    # main loop.
    kloop_unroll: int = 1
    # ``dtl_addr_spread`` (hb_sched + dtl_hoist, no dtl_tight_rsrc): emit the
    # per-tile DTL address math (K offset, range predicate, resource /
    # soffset per operand: a block of SALU) in small steps pinned to free MFMA
    # slots of the tile's first K atom instead of as one block in front of
    # the tile's first MFMA, where it idles the MFMA pipe (visible as a gap in
    # a WaveScope trace).
    dtl_addr_spread: bool = False
    # ``hb_dtl_delay`` (hb_sched): MFMA slots between the release barrier
    # and the first DTL issue (0 = the DTL right behind the barrier).
    hb_dtl_delay: int = 0
    # ---- SIMD-parity loop copies (Python-engine only, default-off) ----
    # ``hb_skew`` (hb_sched, no asm_loop / pingpong): two copies of the main
    # loop (as ``asm_loop_skew``) on the computed hb_sched schedule: the
    # odd copy issues every fragment read, DTL issue and DTL address step one
    # MFMA later (release / acquire slots unchanged), so the two SIMDs of a pair
    # do not hit the LDS / TA in the same MFMA slot. 1: the copy is picked by the
    # SIMD parity (``s_getreg hwreg(HW_ID, 4, 1)``); 2: by the wave-index parity.
    hb_skew: int = 0
    # ---- K start stagger and MFMA order (Python-engine only, default-off) ----
    # ``stagger_u`` (direct_to_lds, non-persistent, no split-K / stream-K /
    # pingpong; 0 = off, else a power of two 2..64): Tensile's StaggerU. Each
    # workgroup starts its K loop at a different K tile, ``((idx mod SU) *
    # stagger_u_stride) mod T`` (T = K tiles), and wraps around at K, so the
    # 256 concurrently running workgroups do not stream the same K column
    # range of A / B at the same moment (power-of-two row strides put those
    # rows on the same memory / cache channels). Only the DTL address moves
    # (the logical K index, range predicate and LDS stage are unchanged), so
    # the result is the same sum in another order. ``idx`` per
    # ``stagger_u_map``: 0 = the output tile's row (M) index (Tensile mapping
    # 0), 1 = its column (N) index, 2 = the flat workgroup id. A K that is not
    # a multiple of tile_k disables the stagger at run time (offset 0).
    stagger_u: int = 0
    stagger_u_stride: int = 1
    stagger_u_map: int = 0
    # ``mfma_serpentine`` (hb_sched / asm_loop / sia_sched bodies, and the
    # ``mfma_interleave`` atom emitters of the other bodies): within each
    # K atom the odd A-fragment rows issue their MFMAs over the B fragments in
    # reverse order (boustrophedon), so at every row change the B operand
    # (src1) of two consecutive MFMAs is the same register quad: one B-operand
    # switch per row change disappears (less operand-path toggling per
    # FLOP). Only the order of independent MFMAs changes (every accumulator
    # still gets its K atoms in order, one atom apart); the event slots (reads,
    # DTL issues, waits, barriers) stay on the same MFMA issue positions.
    mfma_serpentine: bool = False
    # ``mfma_order`` (default 0; hb_sched / sia_sched / mfma_interleave, no
    # asm_loop, not with ``mfma_serpentine``): alternative MFMA issue orders
    # inside a K atom (:func:`mfma_order_perm`; ``mfma_serpentine`` is order
    # 1). 2 = serpentine rows with the odd K atoms running the reversed
    # sequence, so the last MFMA of an atom and the first MFMA of the next one
    # update the same accumulator (a dependent back-to-back pair; hb_sched /
    # sia_sched only). 3 = serpentine columns (ni outer: the B operand repeats
    # for mfmas_m MFMAs, A changes every MFMA). Results are bit-identical to
    # the default order. Retained default-off alternatives: on gfx950 neither
    # was faster than ``mfma_serpentine`` (order 2 forces register copies).
    mfma_order: int = 0
    # ---- hand-ordered main loop (Python-engine only, default-off) ----
    # ``asm_loop`` (hb_sched with no hb_* slot knob, lds_row_interleave,
    # 256x256x64 tile, 2x2 waves of 8x8 16x16x32 atoms, dtl_stages 2): emit
    # every K tile as ONE explicit instruction order taken from an event table
    # (``gemm_asm_loop``): each fragment ds_read, DTL issue, s_waitcnt,
    # s_barrier and s_setprio is pinned after a given MFMA. "tensile" = the
    # Tensile SIA3 loop order (per-operand release AND acquire, 4 barriers
    # per tile), "custom" = a hand-written 256x256x64 f16 loop order
    # (per-operand release, one acquire), "table" = ``asm_loop_table``. The
    # DTL issues of a tile (and the prologue) go inner operand (B) first.
    asm_loop: str = ""
    asm_loop_table: str = ""
    # ``asm_loop_prio`` (asm_loop, 0 = off): ``s_setprio N`` over the MFMA
    # stream, dropped to 0 around every s_barrier.
    asm_loop_prio: int = 0
    # ``asm_loop_skew`` (asm_loop; 0 = off): run a second copy
    # of the main loop on half of the waves, whose table moves the fragment
    # reads / DTL issues by one MFMA (``<mode>_odd``; a "table" gives it after
    # ``||``), so the SIMD pairs do not hit the LDS / TA in the same cycle.
    # 1: the group is the SIMD parity (``s_getreg hwreg(HW_ID, 4, 1)``);
    # 2: the wave-index parity. Both loop copies are emitted and a
    # wave runs one of them (the other's trip count is 0 for it).
    asm_loop_skew: int = 0
    # ``fr0_loop_order`` (dtl_frag_prefetch + mfma_interleave, no asm_loop):
    # read the first tile's fragments before the main loop in the loop's own
    # read order (A0, B0 .. B[n-1], A1 ..) instead of all A then all B. LLVM
    # merges the waitcnt state of the loop entry and the back edge; with the
    # entry's B fragments younger than its A ones, every first-atom MFMA of
    # a tile waited for a B fragment (``lgkmcnt(N)`` per MFMA).
    fr0_loop_order: bool = False
    # ``asm_loop_rawrd`` (asm_loop; DIAGNOSTIC, not for production): emit the
    # fragment ds_reads as inline asm, which LLVM's waitcnt insertion does not
    # see, so the only LDS waits are the table's (``wl``, ``l<N>``), and the
    # next tile's first fragments are consumed with no wait at all.
    # Correctness then rests on the read latency being covered;
    # only the on-GPU full-output check guards it.
    asm_loop_rawrd: bool = False
    # ``rve_pad`` (rilv_vec_epilogue + pad_m / pad_n + dtl_tight_rsrc): the
    # vector epilogue on ragged M / N. Each 16-byte store (8
    # consecutive columns of one row) is guarded by row < M and col < N; the
    # launch contract requires N % 8 == 0, so a store is wholly inside or wholly
    # outside C. With ``pad_fast_tile`` only the edge tiles run the guards.
    # 1: control-flow guards (one per-lane column branch around the epilogue,
    # a per-store row branch); 2: branch-free -- buffer stores over C whose
    # out-of-range lanes get voffset = num_records (dropped by the hardware
    # range check; no non-temporal hint; needs M * N * 2 < 2^32, non-batched).
    rve_pad: int = 0
    # ---- wave-specialized ping-pong (Python-engine only, default-off) ----
    # ``pingpong`` (dtl_stages >= 3, plain staged order): 8-wave-style ping-pong.
    # The waves split into two equal groups (group = warp_id >= waves / 2; with
    # the round-robin SIMD placement every SIMD holds one wave of each group).
    # Every K tile is a MEMORY phase (this tile's operand fragments -> registers,
    # the DTL issue of tile i+S-1, ``s_waitcnt vmcnt((S-2)*L) lgkmcnt(0)``) and
    # a COMPUTE phase (all MFMAs of the tile, ``s_setprio(1)`` with
    # ``interwave_setprio``), separated by full-CTA ``s_barrier``s. Group 1
    # executes one extra barrier before the loop and group 0 one after it, so
    # the groups run one phase apart: while one group's MFMAs fill a SIMD's
    # matrix pipe, the other group's ds_reads / DTL issues / waits run beside
    # them. 1: DTL issues first, then the ds_reads; 2: ds_reads first; 3: the DTL
    # issues spread over the compute phase's MFMAs (the stage they write was read
    # by both groups before this phase), the memory phase only reads; 4
    # (dtl_stages == 2, e.g. tile_k 64 with full-line DMAs): group 0 issues
    # tile i+1 at the start of its memory phase and waits for it after its
    # compute phase, group 1 issues tile i+2 at the start of its compute phase
    # (right after the barrier that ends every wave's reads of tile i) and
    # waits for it at the end of its next memory phase (both windows ~ two
    # phases); 5: as 4, but group 1 spreads those DTL issues over its compute
    # phase. (A first pingpong 4 issued group 1's DTLs at the END of its memory
    # phase, before that barrier: other group-1 waves could still be reading
    # tile i -> a WAR race that failed the full-output check.) 6: dtl_stages 2
    # for A plus a 3-stage B ring (2 A + 3 B stages must fit the LDS, e.g.
    # 256x256x64 f16 = 160 KiB): group 0 issues A(i+1) at the start of its
    # memory phase and spreads B(i+2) over its compute phase, group 1 issues
    # B(i+2) at the start of its memory phase and spreads A(i+2) over its
    # compute phase -- each phase carries half of the tile's DMA bytes. 7: the
    # same rings, but group 0's waves load the whole A tile (A(i+1), start of
    # their memory phase) and group 1's the whole B tile (B(i+2), start of
    # theirs): the compute phases carry no DMA at all.
    pingpong: int = 0
    # ``pp_wait_late`` (pingpong 1..3): drop the memory-phase vmcnt wait and wait
    # after the compute phase instead, for tile i+2 (a wait there for tile i+1
    # would come one phase late for group 1, whose compute phase ends after
    # group 0 started reading tile i+1 -- the first version did that and
    # failed the full-output check).
    pp_wait_late: bool = False
    # ``pp_mix_reads`` (pingpong 4 / 5: group 0; 6: both groups): interleave the
    # memory phase's DTL issues with its fragment ds_reads (one DTL after every
    # k reads, the first DTL first) instead of issuing the DTL burst before the
    # reads. 0 = off.
    pp_mix_reads: int = 0
    # ``pp_spread_lo`` (pingpong 6): spread the compute phase's DTL issues over
    # MFMAs [k, n) instead of [0, n), i.e. after the other group's memory-phase
    # DMA burst has drained from the TA. 0 = whole phase.
    pp_spread_lo: int = 0
    # ---- instruction-economy levers (Python-engine only, default-off) ----
    # ``hb_tile_lgkm0`` (hb_sched + dtl_frag_prefetch, no asm_loop): one
    # explicit ``s_waitcnt lgkmcnt(0)`` after the last MFMA slot of every K
    # tile (and after the first fragments read before the main loop). The next
    # tile's first-atom fragments were read many MFMA slots earlier (right
    # after the acquire), so the wait is free, and LLVM's waitcnt insertion
    # then sees no outstanding LDS read at the tile start: the per-MFMA
    # ``s_waitcnt lgkmcnt(N)`` chain of every tile's first atom disappears.
    # LLVM-visible, so always safe.
    hb_tile_lgkm0: int = 0
    # ``dtl_base_fold`` (dtl_hoist + dtl_soff0, non-persistent): fold the output
    # tile's operand origin into a per-operand 64-bit base pointer once per
    # kernel, so a K tile's resource base is base + k * 2 (one shared shift for
    # both operands) instead of ptr + (tile0 + k) * 2 per operand: two s_add_i32
    # and one s_lshl_b32 fewer per K tile.
    dtl_base_fold: bool = False
    # ``hb_skew_dtl`` (hb_skew): MFMA-slot shift of the odd loop copy's DTL
    # issues (0 = the read shift, 1 slot). The DTLs of a tile are several slots
    # apart, so a shift of half that spacing interleaves the two SIMD parities'
    # DMA issues on the TA instead of issuing them one slot apart.
    hb_skew_dtl: int = 0
    # ``das_anchor`` (dtl_addr_spread): tie every DTL address step to its MFMA
    # slot with a side-effecting empty asm that reads the step's SGPR result
    # (K offset / range predicate, origin, 64-bit resource base). Without it
    # LLVM's DAG scheduler floats the pure SALU of a step into the next gap,
    # which already holds a fragment read: more instructions between two
    # 16-cycle MFMAs than the wave can issue under them (one instruction per 4
    # cycles), which idles the matrix pipe.
    das_anchor: bool = False
    # ``das_late`` (dtl_addr_spread): place the address steps AFTER the
    # tile's last fragment read instead of in the read slots' neighbours: the
    # K offset and A's origin / resource in the free slots between the last
    # read and A's first DTL, B's origin / resource between A's first DTL and
    # the acquire (B's first DTL comes later). LLVM keeps chained ops (reads,
    # DTLs, waits) in order but floats the pure SALU and MFMAs between them, so
    # a step next to a read slot ends up in the read's gap (too many
    # instructions between two MFMAs); away from the reads it cannot.
    das_late: bool = False
    # ``k_stagger`` (direct_to_lds, no tight rsrc / stream-K / split-K / pad_k /
    # pingpong): Tensile's StaggerU. Every CTA starts its K loop at a different
    # K tile, ``((bx + by) % k_stagger) * max(1, k_stagger_tiles)`` (mod the
    # tile count), and wraps around at K: the DTL address of logical tile t is
    # that of tile (t + offset) mod T (range predicate on the logical t, so the
    # prefetches past the end stay zero-size). Without it all CTAs walk K in
    # lockstep and with a power-of-two leading dimension the DMAs of one K
    # offset per 1 KiB land on the same memory channels (visible in a WaveScope
    # capture as periodic long acquire waits).
    # Values 2 / 4 / 8 / 16 / 32 / 64; 0 = off.
    k_stagger: int = 0
    k_stagger_tiles: int = 0
    # ``k_stagger_map`` (k_stagger): which CTA index picks the offset. 0: bx + by
    # (neighbouring CTAs differ; also breaks the L2 sharing of CTAs that read
    # the same A / B panel at the same time); 1: the launch-order linear id
    # (by * grid_x + bx) mod k_stagger -- with k_stagger = the XCD count every
    # XCD (one L2) stays in lockstep and only different XCDs are staggered.
    k_stagger_map: int = 0
    # ``dtl_clamp_k`` (dtl_soff0 + dtl_hoist, non-persistent, no tight rsrc /
    # stream-K / split-K / pad_k / pingpong): the prefetches past the last K
    # tile re-load the last tile (K address clamped to K - tile_k) instead of
    # using a zero-size resource, so num_records is a constant: no per-tile
    # range compare / select and no copy of it into the second resource. The
    # extra LDS writes land in stages no later tile reads (the CTA ends after
    # its epilogue, which does not use LDS).
    dtl_clamp_k: bool = False


@dataclass(frozen=True)
class DataSpec:
    """Element / accumulator / layout choice. Today: f16 in, f16 out, f32 acc, RCR."""

    dtype_a: str = "fp16"
    dtype_b: str = "fp16"
    dtype_c: str = "fp16"
    dtype_acc: str = "fp32"
    layout: str = "RCR"


def mono_data_spec(dtype: str) -> DataSpec:
    """Canonical single-element-type :class:`DataSpec` (A == B == C == ``dtype``).

    The ``f16`` / ``fp16`` aliases collapse to ``fp16``; the accumulator
    (``fp32``) and layout (``RCR``) keep the :class:`DataSpec` defaults. The
    GEMM-family wrappers (batched / grouped / fused-MoE) all share this exact
    mapping, so routing them through one helper keeps the produced
    :class:`DataSpec` -- and therefore the kernel name and emitted IR --
    byte-identical to the hand-rolled bodies it replaces.
    """
    dt = "fp16" if dtype in ("f16", "fp16") else dtype
    return DataSpec(dtype_a=dt, dtype_b=dt, dtype_c=dt)


@dataclass(frozen=True)
class UniversalGemmSpec(WarpTileBlockSizeMixin):
    name: str
    tile: TileSpec
    trait: TraitSpec = field(default_factory=TraitSpec)
    data: DataSpec = field(default_factory=DataSpec)
    wave_size: int = 64
    # If None, derived from `warp_m * warp_n * warp_k * wave_size`
    # (the only valid value per CK's `gemm_validation_utils.py` line 605:
    # `BlockSize = NumWarps * warp_size`). We expose it so an over-rider
    # can force a specific block_size for autotuning experiments.
    block_size: int = 0
    # When True, the kernel reads ``block_id_z`` as the batch index and
    # picks up three extra i64 stride args (``stride_a``, ``stride_b``,
    # ``stride_c``) that scale the per-batch pointer offset. The grid
    # then becomes ``(N_tiles, M_tiles, batch_count)``. This is the only
    # difference between the non-batched ``build_universal_gemm`` and the
    # batched form -- the MFMA / LDS body is shared verbatim so the
    # batched kernel inherits the same correctness + perf as the base.
    batched: bool = False

    def __post_init__(self) -> None:
        self._init_block_size()

    def kernel_name(self) -> str:
        from ...helpers.spec import kernel_name_join

        t = self.tile
        tr = self.trait
        return kernel_name_join(
            self.name,
            self.data.dtype_a,
            f"t{t.tile_m}x{t.tile_n}x{t.tile_k}",
            f"w{t.warp_m}x{t.warp_n}x{t.warp_k}",
            f"wt{t.warp_tile_m}x{t.warp_tile_n}x{t.warp_tile_k}",
            f"{tr.pipeline}_{tr.scheduler}_{tr.epilogue}",
            flags={
                "pad": any([tr.pad_m, tr.pad_n, tr.pad_k]),
                "pers": tr.persistent,
                "bat": self.batched,
                "preb": tr.preshuffle_b,
                "dtl": tr.direct_to_lds,
                "pref": tr.dtl_prefetch,
                "actt": tr.active_tile_skip,
                f"spk{tr.split_k}": tr.split_k > 1,
                "noalc": tr.cshuffle_no_alias,
                "abdb": tr.ab_double_buffer is True,
                "absb": tr.ab_double_buffer is False,
                # Levers that change the emitted body but used to share one
                # symbol (lever-bundle / sweep-space variants were ambiguous).
                # Encoded only when they take effect, so a kernel that does not
                # use them keeps its historical name.
                "swz": tr.lds_swizzle,
                (
                    f"chip{tr.chiplet_wgm}x{tr.chiplet_num_xcds}"
                    f"c{tr.chiplet_chunk_size}"
                ): tr.chiplet_swizzle,
                f"ca{tr.dtl_cache_a}cb{tr.dtl_cache_b}": (
                    tr.direct_to_lds and (tr.dtl_cache_a, tr.dtl_cache_b) != (0, 2)
                ),
                "iwp": tr.interwave_setprio,
                f"stg{tr.dtl_stages}": tr.dtl_stages >= 2,
                "isp": tr.dtl_issue_spread,
                "frp": tr.dtl_frag_prefetch,
                "sbar": tr.dtl_stream_barrier,
                "sab": tr.mfma_swap_ab,
                f"iglp{tr.iglp_opt}": tr.iglp_opt is not None,
                f"pc{tr.persistent_ctas}": tr.persistent_ctas > 0,
                "agpr": tr.acc_agpr,
                "ilv": tr.mfma_interleave,
                "apin": tr.acc_agpr_pin,
                f"de{tr.ilv_dtl_every}": tr.ilv_dtl_every > 0,
                f"ba{tr.ilv_barrier_after}": tr.ilv_barrier_after > 0,
                "mrg": tr.ilv_merge,
                f"swr{tr.lds_swizzle_r}": tr.lds_swizzle_r >= 0,
                "dh": tr.dtl_hoist,
                "sk": tr.streamk,
                "skx": tr.sk_xcd_local,
                "trs": tr.dtl_tight_rsrc,
                "pft": tr.pad_fast_tile,
                "tro": tr.dtl_tight_ragged_only,
                f"ksr{tr.k_stagger_rc}": tr.k_stagger_rc > 0,
                f"er{tr.ilv_early_release}": tr.ilv_early_release > 0,
                f"rl{tr.ilv_release_at}": tr.ilv_release_at > 0,
                f"aq{tr.ilv_acquire_at}": tr.ilv_acquire_at > 0,
                "aov": tr.acc_out_vgpr,
                f"rs{tr.ilv_read_span}": tr.ilv_read_span > 0,
                f"ds{tr.ilv_dtl_span}": tr.ilv_dtl_span > 0,
                f"ef{tr.ilv_early_free}": tr.ilv_early_free > 0,
                "efs": tr.ilv_ef_spread,
                f"sw{tr.ilv_split_wait}": tr.ilv_split_wait > 0,
                "rilv": tr.lds_row_interleave,
                f"rp{tr.rilv_pad}": tr.lds_row_interleave and tr.rilv_pad != 16,
                "hbs": tr.hb_sched,
                f"hr{tr.hb_rel_slot}": tr.hb_sched and tr.hb_rel_slot >= 0,
                f"hq{tr.hb_acq_slot}": tr.hb_sched and tr.hb_acq_slot >= 0,
                f"he{tr.hb_read_every}": tr.hb_sched and tr.hb_read_every != 1,
                f"hd{tr.hb_dtl_every}": tr.hb_sched and tr.hb_dtl_every > 0,
                f"hqb{tr.hb_acq_b_slot}": tr.hb_sched and tr.hb_acq_b_slot >= 0,
                "hsr": tr.hb_sched and tr.hb_split_rel,
                f"hg{tr.hb_rel_gap}": tr.hb_sched and tr.hb_rel_gap != 4,
                "sia": tr.sia_sched,
                "sbf": tr.sia_sched and tr.sia_b_first,
                "splr": tr.sia_sched and tr.sia_plr,
                "slg": tr.sia_sched and tr.sia_single_lgkm,
                "ssr": tr.sia_sched and tr.sia_split_rel,
                "ssq": tr.sia_sched and tr.sia_split_acq,
                "sds": tr.sia_sched and tr.sia_dtl_spread,
                f"sre{tr.sia_read_every}": tr.sia_sched and tr.sia_read_every != 2,
                f"srg{tr.sia_rel_gap}": tr.sia_sched and tr.sia_rel_gap != 4,
                f"sag{tr.sia_acq_gap}": tr.sia_sched and tr.sia_acq_gap != 3,
                f"sst{tr.sia_dtl_stride}": tr.sia_sched and tr.sia_dtl_stride != 3,
                "stt": tr.sia_sched and tr.sia_tensile,
                "suo": tr.sia_sched and tr.sia_use_order,
                "sms": tr.sia_sched and tr.sia_merge_sync,
                "rve": tr.rilv_vec_epilogue,
                f"rvp{tr.rve_pad}": tr.rve_pad > 0,
                "rvnt": tr.rve_store_nt,
                f"w3x{tr.dtl_rsrc_word3:x}": tr.dtl_rsrc_word3 != 0,
                "so0": tr.dtl_soff0,
                f"ku{tr.kloop_unroll}": tr.kloop_unroll != 1,
                "das": tr.dtl_addr_spread,
                f"hdd{tr.hb_dtl_delay}": tr.hb_dtl_delay > 0,
                f"hsk{tr.hb_skew}": tr.hb_skew > 0,
                f"htl{tr.hb_tile_lgkm0}": tr.hb_tile_lgkm0 > 0,
                "dbf": tr.dtl_base_fold,
                f"hsd{tr.hb_skew_dtl}": tr.hb_skew_dtl > 0,
                "dsa": tr.das_anchor,
                "dsl": tr.das_late,
                f"kst{tr.k_stagger}": tr.k_stagger > 0,
                f"kss{tr.k_stagger_tiles}": tr.k_stagger_tiles > 0,
                f"ksm{tr.k_stagger_map}": tr.k_stagger_map > 0,
                "dck": tr.dtl_clamp_k,
                f"su{tr.stagger_u}s{tr.stagger_u_stride}m{tr.stagger_u_map}": tr.stagger_u
                > 0,
                "msp": tr.mfma_serpentine,
                f"mo{tr.mfma_order}": tr.mfma_order > 0,
                _asm_name_tag(tr): bool(tr.asm_loop),
                f"ap{tr.asm_loop_prio}": tr.asm_loop_prio > 0,
                f"ask{tr.asm_loop_skew}": tr.asm_loop_skew > 0,
                "araw": tr.asm_loop_rawrd,
                "f0o": tr.fr0_loop_order,
                f"pp{tr.pingpong}": tr.pingpong > 0,
                "ppwl": tr.pp_wait_late,
                f"ppm{tr.pp_mix_reads}": tr.pp_mix_reads > 0,
                f"ppl{tr.pp_spread_lo}": tr.pp_spread_lo > 0,
            },
        )


# ---------------------------------------------------------------------
# Validity rules (a subset of CK's `arch_filter.py` for gfx950 fp16)
# ---------------------------------------------------------------------


# Per-lever target admission of the AGPR / interleave levers: a lever is
# admitted only on the targets where its bodies passed GPU full-output numerics
# (the C++ mirror, gemm_spec.cpp, uses the same lists). gfx90a also has an AGPR
# file but no numerics yet.
#   acc_agpr, mfma_interleave (+ ilv_*): gfx950 and gfx942 (AGPR / interleave
#     bodies, f16 + bf16).
#   lds_swizzle_r, dtl_hoist: gfx950 and gfx942 (ring, AGPR and interleave
#     bodies with R = 0 / 1 / 2 and the hoist).
_AGPR_ARCHS = ("gfx942", "gfx950")
_ILV_ARCHS = ("gfx942", "gfx950")
_SWR_DH_ARCHS = ("gfx942", "gfx950")


def hb_last_read_slot(spec: "UniversalGemmSpec") -> int:
    """``hb_sched``: the MFMA index (within a K tile) of the last intra-tile
    fragment read (-1 when the tile has a single K atom)."""
    t = spec.tile
    n = t.mfmas_per_warp_m * t.mfmas_per_warp_n
    ka = t.k_atoms_per_tile_k
    if ka < 2:
        return -1
    nrd = t.mfmas_per_warp_m + t.mfmas_per_warp_n
    return (ka - 2) * n + min(n - 1, (nrd - 1) * max(1, spec.trait.hb_read_every))


def hb_slots(spec: "UniversalGemmSpec") -> Tuple[int, int]:
    """``hb_sched``: (release, acquire) MFMA slot of a K tile (auto-resolved)."""
    t = spec.tile
    n = t.mfmas_per_warp_m * t.mfmas_per_warp_n
    ntot = t.k_atoms_per_tile_k * n
    tr = spec.trait
    rel = (
        tr.hb_rel_slot
        if tr.hb_rel_slot >= 0
        else min(ntot - 1, hb_last_read_slot(spec) + max(1, tr.hb_rel_gap))
    )
    acq = (
        tr.hb_acq_slot
        if tr.hb_acq_slot >= 0
        else min(ntot - 1, (t.k_atoms_per_tile_k - 1) * n + 3)
    )
    return rel, acq


def sia_table(spec: "UniversalGemmSpec") -> Dict[str, Any]:
    """``sia_sched``: the MFMA-slot table of one K tile (2 K atoms).

    Every entry is placed AFTER the MFMA with that index (0 .. 2n-1, n =
    mfmas_m * mfmas_n). Returns a dict with

    * ``rd1``: [(slot, op, idx)] next-atom (kk 1) fragment reads, this stage;
    * ``rel``: [(slot, ops)] releases (lgkmcnt(0) + s_barrier), ops a string
      of operands ("A", "B" or "AB") whose stage region becomes free;
    * ``acq``: [(slot, ops)] acquires (vmcnt + s_barrier) of the next stage;
    * ``rd0``: [(slot, op, idx)] next tile's first-atom reads, next stage;
    * ``dtl``: [(slot, op, pass)] DTL issues in issue order;
    * ``lgkm``: slot of the explicit lgkmcnt(0) or -1;
    * ``order``: the operand order ("AB" or "BA").

    Raises ValueError when the table does not fit the tile.
    """
    t, tr = spec.tile, spec.trait
    mm, mn = t.mfmas_per_warp_m, t.mfmas_per_warp_n
    n = mm * mn
    ntot = 2 * n
    e = max(1, int(tr.sia_read_every))
    cnt = {"A": mm, "B": mn}
    bs = max(1, spec.block_size)
    npass = {
        "A": -(-(t.tile_m * t.tile_k // 8) // bs),
        "B": -(-(t.tile_n * t.tile_k // 8) // bs),
    }
    X, Y = ("B", "A") if tr.sia_b_first else ("A", "B")
    if tr.sia_tensile:
        if (mm, mn, npass["A"], npass["B"]) != (8, 8, 8, 8):
            raise ValueError(
                "sia_tensile needs 8x8 MFMAs per atom and 8 + 8 DTL issues"
            )
        rd1 = [(2 * j, X, j) for j in range(8)]
        rd1 += [(s_, Y, j) for j, s_ in enumerate((24, 27, 30, 33, 36, 38, 40, 42))]
        rd0 = [(69 + 2 * j, X, j) for j in range(8)] + [
            (106 + 2 * j, Y, j) for j in range(8)
        ]
        xs = (22, 25, 28, 31, 34, 52, 55, 58)
        ys = (61, 64, 85, 87, 89, 96, 100, 124)
        return {
            "n": n,
            "ntot": ntot,
            "rd1": rd1,
            "rel": [(19, X), (50, Y)],
            "acq": [(67, X), (104, Y)],
            "rd0": rd0,
            "dtl": [(s_, X, p) for p, s_ in enumerate(xs)]
            + [(s_, Y, p) for p, s_ in enumerate(ys)],
            "lgkm": 0 if tr.sia_single_lgkm else -1,
            "order": X + Y,
            "npass": npass,
            "merged": False,
        }
    gap = max(1, int(tr.sia_rel_gap))
    if tr.sia_use_order:
        useq = (
            [("A", 0)]
            + [("B", j) for j in range(mn)]
            + [("A", j) for j in range(1, mm)]
        )
    # ---- next-atom reads (this stage) + releases ----
    rd1 = [(j * e, X, j) for j in range(cnt[X])]
    last_x = rd1[-1][0]
    if tr.sia_split_rel:
        rel_x = last_x + gap
        y0 = rel_x + 1
    else:
        y0 = last_x + e
    rd1 += [(y0 + j * e, Y, j) for j in range(cnt[Y])]
    if tr.sia_use_order:
        rd1 = [(j * e, op, i) for j, (op, i) in enumerate(useq)]
    last_y = rd1[-1][0]
    rel_y = last_y + gap
    rel = [(rel_x, X), (rel_y, Y)] if tr.sia_split_rel else [(rel_y, X + Y)]
    # ---- acquires + next tile's first-atom reads (next stage) ----
    acq_x = n + int(tr.sia_acq_gap) if tr.sia_plr else n + n // 2
    if tr.sia_merge_sync:
        acq_x = max(acq_x, last_y + 1)
        rel = []
    elif acq_x <= rel_y:
        acq_x = rel_y + 1
    rd0: List[Tuple[int, str, int]] = []
    acq: List[Tuple[int, str]] = []
    nread0 = cnt[X] + cnt[Y]

    def _fit(start: int, k: int) -> int:
        # read spacing that keeps k reads from ``start`` inside the tile
        room = ntot - start
        return max(1, min(e, room // max(1, k)))

    if tr.sia_split_acq:
        acq.append((acq_x, X))
        ex = _fit(acq_x + 1, nread0)
        rd0 += [(acq_x + 1 + j * ex, X, j) for j in range(cnt[X])]
        acq_y = rd0[-1][0] + 1
        acq.append((acq_y, Y))
        ey = _fit(acq_y + 1, cnt[Y])
        rd0 += [(acq_y + 1 + j * ey, Y, j) for j in range(cnt[Y])]
    else:
        acq.append((acq_x, X + Y))
        ex = _fit(acq_x + 1, nread0)
        rd0 += [(acq_x + 1 + j * ex, X, j) for j in range(cnt[X])]
        st = rd0[-1][0] + ex
        rd0 += [(st + j * ex, Y, j) for j in range(cnt[Y])]
        if tr.sia_merge_sync:
            # the merged sync leaves no release in front: read right away
            rd0 = [(s_ - 1, op, i) for s_, op, i in rd0]
        if tr.sia_use_order:
            st0 = acq_x if tr.sia_merge_sync else acq_x + 1
            rd0 = [(st0 + j * ex, op, i) for j, (op, i) in enumerate(useq)]
    if rd0[-1][0] >= ntot:
        raise ValueError(
            f"next-tile reads overflow the tile (last at {rd0[-1][0]} >= {ntot})"
        )
    if last_y >= n:
        raise ValueError(
            f"next-atom reads must finish inside the first atom (last at {last_y})"
        )
    # ---- DTL issue slots: X passes after X's release, then Y's ----
    rel_of = {op: s for s, ops in rel for op in ops}
    if tr.sia_merge_sync:
        rel_of = {X: acq_x, Y: acq_x}
    busy = {s for s, _, _ in rd1 + rd0} | {s for s, _ in rel + acq}
    lgkm = 0 if tr.sia_single_lgkm else -1
    dtl: List[Tuple[int, str, int]] = []
    stride = max(1, int(tr.sia_dtl_stride))
    seq = [(X, p) for p in range(npass[X])] + [(Y, p) for p in range(npass[Y])]
    lo0 = rel_of[X] + 1
    if tr.sia_dtl_spread:
        # free slots (no read / sync) after the first release; the issues
        # take evenly spaced ones, never before their operand's release,
        # at least ``stride`` apart while the rest still fits
        free = [s for s in range(lo0, ntot) if s not in busy]
        prev = -1
        for j, (op, p) in enumerate(seq):
            left = len(seq) - j
            tgt = free[min(len(free) - 1, (j * len(free)) // len(seq))] if free else lo0
            want = max(tgt, rel_of[op] + 1, prev + 1)
            if prev >= 0 and (ntot - (prev + stride)) >= left * 1:
                want = max(want, prev + stride)
            nxt = [s for s in free if s >= want]
            # snap to a free slot when the remaining issues still fit after it
            if nxt and ntot - nxt[0] >= left:
                want = nxt[0]
            want = min(want, ntot - left)
            want = max(want, rel_of[op] + 1, prev + 1)
            dtl.append((want, op, p))
            prev = want
    else:
        # evenly over every slot after the first release (hb_sched style)
        span = ntot - lo0
        prev = -1
        for j, (op, p) in enumerate(seq):
            want = max(lo0 + (j * span) // len(seq), rel_of[op] + 1, prev + 1)
            dtl.append((want, op, p))
            prev = want
    if dtl and max(s for s, _, _ in dtl) >= ntot:
        raise ValueError("DTL issues overflow the tile")
    return {
        "n": n,
        "ntot": ntot,
        "rd1": rd1,
        "rel": rel,
        "acq": acq,
        "rd0": rd0,
        "dtl": dtl,
        "lgkm": lgkm,
        "order": X + Y,
        "npass": npass,
        "merged": bool(tr.sia_merge_sync),
    }


def sia_acq_vmcnt(tb: Dict[str, Any], with_load: bool = True) -> List[int]:
    """``sia_sched``: the vmcnt of every acquire of a ``sia_table`` (without
    the older tiles' ``vm_wait``): the previous tile's DTL issues after its
    last load of the acquired operand(s), plus this tile's issues before the
    acquire (an issue in the acquire's own slot runs after it). A tile
    without loads (the K tail) only keeps the first part."""
    dtl = tb["dtl"]

    def _after(op: str) -> int:
        last = max(j for j, (_s, o, _p) in enumerate(dtl) if o == op)
        return len(dtl) - 1 - last

    out = []
    for sl, ops in tb["acq"]:
        before = sum(1 for d, _o, _p in dtl if d < sl) if with_load else 0
        out.append(min(_after(o) for o in ops) + before)
    return out


def launch_shape_error(
    spec: "UniversalGemmSpec", M: int, N: int, K: int
) -> Optional[str]:
    """Why a launch with problem size ``(M, N, K)`` is outside ``spec``'s
    kernel contract, or ``None``.

    A body without ``pad_m`` / ``pad_n`` / ``pad_k`` has no bounds checks on
    that dimension: a partial M / N tile reads and writes out of bounds, and a
    partial K tile (e.g. K % 64 == 32 on a tile_k 64 body) silently drops or
    double-counts K slices. ``split_k`` K-slices must be whole tiles too.
    """
    t = spec.tile
    tr = spec.trait
    for name, dim, tile, padded in (
        ("M", M, t.tile_m, tr.pad_m),
        ("N", N, t.tile_n, tr.pad_n),
        ("K", K, t.tile_k, tr.pad_k),
    ):
        if not padded and int(dim) % tile:
            return (
                f"{name}={dim} is not a multiple of tile_{name.lower()}={tile} "
                f"and pad_{name.lower()} is off (the body has no bounds checks "
                f"on {name}: wrong results)"
            )
    if tr.direct_to_lds and int(K) % t.tile_k:
        # The K-tail masking (pad_k) lives on the global-load path only; the
        # DRAM->LDS loads copy whole K tiles, so a partial K tile reads past the
        # row (the next row's elements) and the outputs are wrong. A pad_k
        # body with K a tile multiple has no tail and stays legal.
        return (
            f"direct_to_lds: K={K} is not a multiple of tile_k={t.tile_k}; the "
            f"DRAM->LDS loads do not mask the K tail (wrong results even with "
            f"pad_k)"
        )
    if tr.dtl_tight_rsrc and max(int(M), int(N)) * int(K) * 2 >= 2**31:
        return (
            f"dtl_tight_rsrc: operand bytes max(M, N) * K * 2 = "
            f"{max(int(M), int(N)) * int(K) * 2} do not fit the i32 buffer range"
        )
    if tr.rve_pad and tr.pad_n and int(N) % 8:
        return (
            f"rve_pad: N={N} is not a multiple of 8 (each lane stores 8 "
            "consecutive 16-bit columns with one 16-byte store)"
        )
    if tr.rve_pad == 2 and int(M) * int(N) * 2 >= 2**32:
        return (
            f"rve_pad 2: C bytes M * N * 2 = {int(M) * int(N) * 2} do not fit "
            "the u32 buffer range"
        )
    if tr.split_k > 1 and int(K) % tr.split_k:
        return f"K={K} is not a multiple of split_k={tr.split_k}"
    if tr.split_k > 1 and not tr.pad_k and int(K) % (tr.split_k * t.tile_k):
        return (
            f"K={K} is not a multiple of split_k * tile_k = "
            f"{tr.split_k * t.tile_k} (each K-slice must be whole tiles)"
        )
    return None


def check_launch_shape(spec: "UniversalGemmSpec", M: int, N: int, K: int) -> None:
    """Raise ``ValueError`` when :func:`launch_shape_error` reports a problem.
    Call it before launching a universal-GEMM kernel with a runtime shape."""
    err = launch_shape_error(spec, M, N, K)
    if err is not None:
        raise ValueError(f"{spec.kernel_name()}: {err}")


def streamk_workspace_bytes(spec: "UniversalGemmSpec") -> Tuple[int, int]:
    """``(flags_bytes, partials_bytes)`` of a ``streamk`` body: one i32 ready
    flag per (workgroup of the P pool, wave), padded to 256 bytes, then one f32
    partial tile per workgroup. Pass the base as ``SkFlags`` and ``base +
    flags_bytes`` as ``SkPart``; zero the flag region once at allocation (the
    kernel leaves it zeroed again)."""
    if not spec.trait.streamk:
        return (0, 0)
    t = spec.tile
    if t.warp_k != 1:
        # the kernel strides a partial slot by len(accs) x c_per_lane x threads
        # floats, which equals tile_m x tile_n only for warp_k == 1 (the
        # validity rule "streamk requires warp_k == 1")
        raise ValueError(
            f"streamk_workspace_bytes: warp_k {t.warp_k} != 1 (the partial slot "
            "would be smaller than the kernel writes)"
        )
    pool = int(spec.trait.persistent_ctas)
    waves = t.warp_m * t.warp_n * t.warp_k
    flags = -(-4 * pool * waves // 256) * 256
    return (flags, pool * t.tile_m * t.tile_n * 4)


# CDNA: 4 SIMDs per CU, and one wave per SIMD at the 512-register (VGPR + AGPR)
# maximum -- the register-limited occupancy a kernel is guaranteed before its
# register count is known.
STREAMK_WORST_CASE_WAVES_PER_CU = 4


def streamk_resident_ctas(
    spec: "UniversalGemmSpec", arch: str, lds_bytes: int
) -> Tuple[int, int]:
    """``(resident, per_cu)``: a lower bound of the workgroups of ``spec``
    that are co-resident on a whole ``arch`` device -- ``arch_num_cus(arch)`` x
    ``per_cu``, ``per_cu = min(LDS capacity // lds_bytes, max(1, 4 // waves))``
    (LDS-limited, and register-limited at the worst case of one wave per SIMD).
    A stream-K owner spins on higher-index workgroups, so ``persistent_ctas``
    above this bound can hang the GPU instead of failing."""
    from ...core.arch import ArchTarget
    from ...helpers.split_k import arch_num_cus

    t = spec.tile
    waves = t.warp_m * t.warp_n * t.warp_k
    cap = ArchTarget.from_gfx(arch).lds_capacity_bytes
    per_lds = cap // max(1, int(lds_bytes))
    per_reg = max(1, STREAMK_WORST_CASE_WAVES_PER_CU // max(1, waves))
    per_cu = min(per_lds, per_reg)
    return arch_num_cus(arch) * per_cu, per_cu


def streamk_launch_error(
    spec: "UniversalGemmSpec", device_cus: int, wgs_per_cu: int
) -> Optional[str]:
    """Runtime co-residency check of a ``streamk`` launch on the LIVE device
    (``device_cus`` = the multiprocessor count the runtime reports, e.g. 32 per
    partition in CPX mode; ``wgs_per_cu`` = the occupancy of the compiled kernel,
    e.g. ``hipModuleOccupancyMaxActiveBlocksPerMultiprocessor``). ``None`` when
    all ``persistent_ctas`` workgroups fit at once, else the reason."""
    if not spec.trait.streamk:
        return None
    p = int(spec.trait.persistent_ctas)
    fit = int(device_cus) * int(wgs_per_cu)
    if p > fit:
        return (
            f"streamk pool {p} > {fit} co-resident workgroups ({device_cus} CUs x "
            f"{wgs_per_cu} per CU): the fixup would spin on workgroups that never start"
        )
    return None


def acc_agpr_regs(spec: "UniversalGemmSpec") -> int:
    """Per-lane f32 accumulator registers of one wave (wave64 MFMA):
    ``mfmas_m * mfmas_n * warp_tile_m * warp_tile_n / 64``."""
    t = spec.tile
    c_per = (t.warp_tile_m * t.warp_tile_n) // 64
    return t.mfmas_per_warp_m * t.mfmas_per_warp_n * c_per


_F16_WARP_TILE_SHAPES_GFX950 = {
    (16, 16, 16),
    (16, 16, 32),
    (32, 32, 8),
    (32, 32, 16),
}

_BF16_WARP_TILE_SHAPES_GFX950 = {
    (16, 16, 16),
    (16, 16, 32),
}


def _dtype_ir(name: str) -> Type:
    """Resolve GEMM storage dtype strings to IR types.

    Universal GEMM currently supports homogeneous A/B/C storage dtypes
    for f16 and bf16 with f32 accumulation.
    """
    return io_ir_type(name)


def _mma_family(arch: str) -> str:
    """The MMA atom family this target uses: ``"wmma"`` for the RDNA
    wave32 targets (gfx11xx), ``"mma"`` (MFMA) for CDNA.

    The single body in :func:`build_universal_gemm` selects the atom from
    the target's catalog using this family, so the same spec resolves to an
    MFMA op_id on gfx942/gfx950 and a WMMA op_id on gfx1151.
    """
    from ...core.arch import ArchTarget

    return "wmma" if ArchTarget.from_gfx(arch).wave_size == 32 else "mma"


def _resolve_mma_op(spec: "UniversalGemmSpec", arch: str):
    """Resolve the :class:`~rocke.core.arch.MmaOp` for ``spec`` on ``arch``.

    The op carries the backend ``op_id`` (consumed by the target-neutral
    :meth:`IRBuilder.mma`), the per-lane fragment vector lengths
    (``a_frag_len`` / ``b_frag_len`` / ``c_frag_len``) and the
    lane/slot -> tile-coordinate layout maps that drive the fragment loads and
    the accumulator scatter. Returns ``None`` if the target has no atom for the
    spec's warp-tile shape + dtype (callers turn that into a structured reject).
    """
    from ...core.arch import ArchTarget

    target = ArchTarget.from_gfx(arch)
    t = spec.tile
    a_name = spec.data.dtype_a  # homogeneous A/B/C (validated in _storage_dtype)
    return target.mma.op_for_shape(
        family=_mma_family(arch),
        a_dtype=a_name,
        b_dtype=a_name,
        c_dtype="fp32",
        m=t.warp_tile_m,
        n=t.warp_tile_n,
        k=t.warp_tile_k,
    )


def _storage_dtype(spec: UniversalGemmSpec) -> Type:
    d = spec.data
    if d.dtype_a != d.dtype_b or d.dtype_a != d.dtype_c:
        raise ValueError(
            "UniversalGemmSpec currently requires homogeneous A/B/C dtypes; "
            f"got A={d.dtype_a}, B={d.dtype_b}, C={d.dtype_c}"
        )
    if d.dtype_acc not in ("fp32", "f32"):
        raise ValueError(
            f"UniversalGemmSpec only supports fp32 accumulation, got {d.dtype_acc!r}"
        )
    if d.layout != "RCR":
        raise ValueError(
            f"UniversalGemmSpec only supports RCR layout, got {d.layout!r}"
        )
    return _dtype_ir(d.dtype_a)


# ``num_records`` of the buffer resources the direct-to-LDS load path builds for
# A and B (bytes, just under 2 GiB). The per-lane offset is an i32 BYTE offset,
# and a buffer load past ``num_records`` returns 0 without faulting, so a DTL
# kernel is only correct while each operand fits in this range. The GEMM
# dispatcher enforces it per request (``dispatch.gemm.support.
# operand_range_supported``); direct users of ``direct_to_lds=True`` must too.
DTL_BUFFER_NUM_RECORDS = 0x7FFF0000


def ab_double_buffer_decision(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    *,
    pipeline: str,
    epilogue: str,
    direct_to_lds: bool,
    ab_double_buffer: Optional[bool],
    lds_cap: int,
) -> bool:
    """Whether the compv4 VGPR-staged AB LDS double buffer (``_emit_kloop_db``)
    is emitted. Pure ints/bools, shared by the validity gate, the emitter and
    the dispatcher support predicate (and mirrored by the C++ engine's
    ``rocke_gemm_ab_db_decision``).

    * ``ab_double_buffer is None`` -- the historical automatic rule, kept
      byte-identical: compv4, direct epilogue, not DTL, and the doubled AB still
      fits 2 WG/CU (``2 * (2 * ab_single) <= lds_cap``).
    * ``True`` / ``False`` -- explicit (only legal on compv4 without DTL; the
      validity gate rejects it elsewhere so it can never be a silent no-op).
    """
    if pipeline != "compv4" or direct_to_lds:
        return False
    if ab_double_buffer is not None:
        return bool(ab_double_buffer)
    ab_single = ((tile_m * tile_k) + (tile_n * tile_k)) * 2
    db_fits_2wg = (2 * ab_single) * 2 <= lds_cap
    return epilogue != "cshuffle" and db_fits_2wg


def universal_lds_bytes(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    *,
    pipeline: str,
    epilogue: str,
    direct_to_lds: bool,
    dtl_prefetch: bool,
    ab_double_buffer: Optional[bool],
    cshuffle_no_alias: bool,
    lds_k_pad: int,
    lds_cap: int,
    elem_bytes_a: int = 2,
    elem_bytes_b: int = 2,
    elem_bytes_c: int = 2,
    dtl_stages: int = 0,
) -> Tuple[int, int, int]:
    """LDS bytes the universal body really allocates: ``(total, ab, c)``.

    * AB: ``(tile_m * eA + tile_n * eB) * (tile_k + pad)`` per buffer (the
      non-DTL ``lds_k_pad`` widens every row), times 2 when the AB is
      ping-ponged (``dtl_prefetch`` or the compv4 double buffer).
    * C (cshuffle only): ``tile_m * tile_n * eC``. The smem-pool packer aliases
      it onto the A/B bytes (they are not live at the same time) unless
      ``cshuffle_no_alias`` marks it exclusive, so the pool is ``max(ab, c)``
      by default and ``ab + c`` only with ``cshuffle_no_alias``.

    The element sizes default to 2 bytes (the f16 / bf16 body this module
    emits; the validity gate always uses the defaults). The dispatcher support
    predicate passes the request's sizes so 1- / 4-byte queries are sized
    rather than rejected.
    """
    pad = 0 if direct_to_lds else int(lds_k_pad)
    ab_single = (tile_m * elem_bytes_a + tile_n * elem_bytes_b) * (tile_k + pad)
    db = ab_double_buffer_decision(
        tile_m,
        tile_n,
        tile_k,
        pipeline=pipeline,
        epilogue=epilogue,
        direct_to_lds=direct_to_lds,
        ab_double_buffer=ab_double_buffer,
        lds_cap=lds_cap,
    )
    ab = ab_single * (2 if (bool(dtl_prefetch) or db) else 1)
    if direct_to_lds and dtl_prefetch and dtl_stages >= 2:
        # alias-scoped DTL ring: one A + one B allocation per stage
        ab = ab_single * int(dtl_stages)
    c = tile_m * tile_n * elem_bytes_c if epilogue == "cshuffle" else 0
    total = ab + c if cshuffle_no_alias else max(ab, c)
    return total, ab, c


def _ab_lds_plan(spec: UniversalGemmSpec, arch: str) -> Tuple[int, bool, bool]:
    """AB LDS double-buffer plan, shared by the validity gate and emitter.

    Returns ``(ab_single, db, two_buf)`` as pure Python ints/bools:

    * ``ab_single`` — bytes for one (single-buffered, unpadded) AB LDS region,
      ``(tile_m*tile_k + tile_n*tile_k) * 2``.
    * ``db`` — whether the compv4 software-pipelined double buffer is
      enabled (:func:`ab_double_buffer_decision`).
    * ``two_buf`` — whether the AB LDS is double-buffered at all
      (``dtl_prefetch`` ping-pong OR ``db``).

    Keeping ``is_valid_spec`` and ``build_universal_gemm`` in lock-step
    through this single source avoids the class of bug where the gate
    reserves 2x AB but the emitter single-buffers (or vice-versa).
    """
    from ...core.arch import ArchTarget

    t = spec.tile
    ab_single = ((t.tile_m * t.tile_k) + (t.tile_n * t.tile_k)) * 2
    lds_cap = ArchTarget.from_gfx(arch).lds_capacity_bytes
    db = ab_double_buffer_decision(
        t.tile_m,
        t.tile_n,
        t.tile_k,
        pipeline=spec.trait.pipeline,
        epilogue=spec.trait.epilogue,
        direct_to_lds=spec.trait.direct_to_lds,
        ab_double_buffer=spec.trait.ab_double_buffer,
        lds_cap=lds_cap,
    )
    two_buf = bool(spec.trait.dtl_prefetch) or db
    return ab_single, db, two_buf


def _ilog2_exact(x: int) -> Optional[int]:
    """``log2(x)`` for a positive power of two, else ``None``."""
    return x.bit_length() - 1 if x and (x & (x - 1)) == 0 else None


def lds_swizzle_params(
    a_per_lane: int, b_per_lane: int, block_k: int
) -> Tuple[int, int, int]:
    """Geometry-derived ``(R, W, L)`` of the ``lds_swizzle`` permutation
    ``col ^= ((row >> R) % 2^W) << L`` (pure; mirrored by the C++ engine's
    ``rocke_gemm_lds_swizzle_params``).

    Single source for the emitter (:func:`build_universal_gemm`, which lets the
    ``CK_SWZ_R/W/L`` debug env vars override the result) and the validity gate
    (which never reads the environment, so validity and dispatch are
    environment-independent). The permutation moves whole ``2^L``-half groups.

    * ``L = log2(elem)`` -- swizzle whole ``elem``-half fragment slots;
    * ``W = log2(block_k / elem)`` -- use all element slots of a row;
    * ``R = 0`` -- key on the low row bits.

    Fallback (``elem`` / slots not a clean power-of-two split): the 2-slot
    toggle ``(3, 1, 4)``.
    """
    elem = a_per_lane if a_per_lane == b_per_lane else 0
    slots = (block_k // elem) if (elem and block_k % elem == 0) else 0
    auto_l, auto_w = _ilog2_exact(elem), _ilog2_exact(slots)
    if auto_l is not None and auto_w is not None and auto_w >= 1:
        return 0, auto_w, auto_l
    return 3, 1, 4


def is_valid_spec(spec: UniversalGemmSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    """Return `(ok, reason)`. The same predicate CK's dispatcher uses
    to drop unbuildable configs from a sweep.

    Architecture facts (legal MFMA atoms, LDS capacity, max threads/block)
    are sourced from :class:`rocke.core.arch.ArchTarget`, so this predicate
    is arch-aware: a warp-tile atom that exists on gfx950 but not gfx942
    (e.g. the wide ``16x16x32`` f16 atom) is rejected for gfx942 with a
    structured reason instead of crashing comgr at lower time.
    """
    from ...core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)

    t = spec.tile
    try:
        _storage_dtype(spec)  # validates homogeneous dtypes / fp32 acc / RCR
    except ValueError as e:
        return False, str(e)

    # Bodies whose epilogue has no fused-epilogue hook (helpers.fuse attaches
    # one through the ``_fused_epilogue`` side channel): reject here instead of
    # raising in build_universal_gemm.
    if getattr(spec, "_fused_epilogue", None) is not None:
        for _lever, _on in (
            ("rilv_vec_epilogue", spec.trait.rilv_vec_epilogue),
            ("persistent_ctas", spec.trait.persistent_ctas > 0),
            ("mfma_swap_ab", spec.trait.mfma_swap_ab),
        ):
            if _on:
                return False, f"{_lever} does not support a fused epilogue"

    # MMA atom must be in the target's catalog for this dtype combo. The
    # family is selected from the target's wave size: CDNA (wave64) uses MFMA,
    # the RDNA wave32 targets (gfx11xx) use WMMA. The same warp-tile shape thus
    # resolves to an MFMA op_id on gfx942/gfx950 and a WMMA op_id on gfx1151.
    family = _mma_family(arch)
    a_name = spec.data.dtype_a  # homogeneous A/B/C (checked above)
    atom = (t.warp_tile_m, t.warp_tile_n, t.warp_tile_k)
    if not target.supports_dtype_combo(a_name, a_name, "fp32", family=family):
        return False, f"unsupported GEMM dtype {a_name!r} on {arch}"
    if not target.mma.has_shape(
        family=family,
        a_dtype=a_name,
        b_dtype=a_name,
        c_dtype="fp32",
        m=t.warp_tile_m,
        n=t.warp_tile_n,
        k=t.warp_tile_k,
    ):
        return False, f"unsupported {a_name} warp_tile {atom} on {arch}"

    # The spec's wave size must match the target's. ``spec.wave_size`` is baked
    # into ``block_size`` at construction, so a wave64 spec cannot launch on a
    # wave32 target (and vice versa); reject early with a structured reason
    # rather than emitting a kernel whose lane geometry the hardware rejects.
    if spec.wave_size != target.wave_size:
        return False, (
            f"spec wave_size {spec.wave_size} != {arch} wave_size {target.wave_size}"
        )

    # WMMA coverage is intentionally narrower than the full CDNA MFMA matrix:
    # gfx11/gfx12 RDNA supports the 16x16x16 atom and gfx1250 supports the
    # gfx1250-class 16x16x32 atom, both through the simple ``mem`` pipeline +
    # ``default`` epilogue. The richer pipelines (compv3 / compv4 scheduler
    # interleave, cshuffle LDS-staged C, DTLA, preshuffle) encode MFMA-shaped
    # assumptions and are gated off until ported. CDNA MFMA keeps the full
    # matrix.
    if family == "wmma":
        supported_atoms = {(16, 16, 16)}
        if arch == "gfx1250":
            supported_atoms = {(16, 16, 32)}
        if atom not in supported_atoms:
            supported = ", ".join(
                f"{m}x{n}x{k}" for (m, n, k) in sorted(supported_atoms)
            )
            return False, f"WMMA path supports only {supported} (got {atom}) on {arch}"
        if spec.trait.pipeline not in ("mem", "wmma_v1"):
            return False, (
                f"WMMA path supports only the 'mem' or 'wmma_v1' pipeline "
                f"(got {spec.trait.pipeline!r}) on {arch}"
            )
        if spec.trait.epilogue != "default":
            return False, (
                f"WMMA path supports only the 'default' epilogue "
                f"(got {spec.trait.epilogue!r}) on {arch}"
            )
        for flag, label in (
            (spec.trait.preshuffle_b, "preshuffle_b"),
            (spec.trait.direct_to_lds, "direct_to_lds"),
            (spec.trait.dtl_prefetch, "dtl_prefetch"),
            (spec.trait.active_tile_skip, "active_tile_skip"),
            (spec.trait.chiplet_swizzle, "chiplet_swizzle"),
        ):
            if flag:
                return False, f"WMMA path does not support {label} on {arch}"

    # DirectToLDS: the emitter bakes a fixed per-lane chunk width into the
    # kernel at build time (see build_universal_gemm), capped at the arch's
    # own buffer_load_lds width (ArchTarget.async_lds_max_dwords). Asking for
    # a wider one than the arch has is not a build-time error -- the AMDGPU
    # backend aborts the whole process -- so an arch with too little width to
    # satisfy tile_k, or none at all, must be rejected here.
    if spec.trait.direct_to_lds:
        if target.async_lds_max_dwords < 1:
            return False, f"direct_to_lds: {arch} has no DRAM->LDS DMA instruction"
        _dtl_halves = min(4, target.async_lds_max_dwords) * 2
        if t.tile_k % _dtl_halves != 0:
            return False, (
                f"direct_to_lds requires tile_k % {_dtl_halves} == 0 (got "
                f"{t.tile_k}) with {target.async_lds_max_dwords} dword(s)/lane "
                f"on {arch}"
            )

    # Geometry divisibility.
    if t.tile_m % (t.warp_m * t.warp_tile_m):
        return False, "tile_m not divisible by warp_m * warp_tile_m"
    if t.tile_n % (t.warp_n * t.warp_tile_n):
        return False, "tile_n not divisible by warp_n * warp_tile_n"
    if t.tile_k % t.warp_tile_k:
        return False, "tile_k not divisible by warp_tile_k"

    # block_size = warp_m * warp_n * wave_size.
    expected_bs = t.warp_m * t.warp_n * spec.wave_size
    if expected_bs != spec.block_size:
        return False, (
            f"block_size {spec.block_size} != warp_m*warp_n*wave_size = {expected_bs}"
        )

    # Trait values.
    #   * ``persistent``: accepted (historical behaviour); the body has no
    #     persistent tile loop, so it only renames the kernel
    #     (:func:`build_universal_gemm` warns).
    #   * ``scheduler``: "intrawave" / "interwave" both emit the intrawave
    #     body (historical behaviour, kept so tuned kernels that set
    #     "interwave" do not change); any other value is a typo -> rejected.
    #   * ``interwave_setprio``: the opt-in MFMA-cluster priority bookends;
    #     the WMMA body has no such schedule -> rejected there.
    #   * ``ab_double_buffer``: only the compv4 VGPR-staged body has the AB
    #     double buffer it controls.
    if spec.trait.scheduler not in ("intrawave", "interwave"):
        return False, f"unknown scheduler {spec.trait.scheduler!r}"
    if spec.trait.interwave_setprio and family != "mma":
        return (
            False,
            f"interwave_setprio is MFMA-only (got family {family!r} on {arch})",
        )
    if spec.trait.ab_double_buffer is not None and (
        spec.trait.pipeline != "compv4" or spec.trait.direct_to_lds
    ):
        return False, (
            "ab_double_buffer applies only to the compv4 VGPR-staged pipeline "
            f"(pipeline={spec.trait.pipeline!r}, direct_to_lds="
            f"{spec.trait.direct_to_lds}); DTL uses dtl_prefetch"
        )

    # DTL ring levers: each is only implemented by one body, so asking
    # for it anywhere else is rejected rather than silently ignored.
    tr = spec.trait
    if tr.dtl_stages not in (0, 2, 3, 4):
        return False, f"dtl_stages must be 0, 2, 3 or 4 (got {tr.dtl_stages})"
    if tr.dtl_stages and not (family == "mma" and tr.direct_to_lds and tr.dtl_prefetch):
        return False, "dtl_stages requires the MFMA direct_to_lds + dtl_prefetch body"
    for _flag, _nm in (
        (tr.dtl_issue_spread, "dtl_issue_spread"),
        (tr.dtl_frag_prefetch, "dtl_frag_prefetch"),
        (tr.dtl_stream_barrier, "dtl_stream_barrier"),
        (tr.iglp_opt is not None, "iglp_opt"),
    ):
        if _flag and tr.dtl_stages < 2:
            return False, f"{_nm} requires dtl_stages >= 2"
    if tr.persistent_ctas:
        if tr.persistent_ctas < 0:
            return False, f"persistent_ctas must be >= 0 (got {tr.persistent_ctas})"
        if tr.dtl_stages < 2:
            return False, "persistent_ctas requires dtl_stages >= 2"
        if (
            tr.epilogue != "default"
            or spec.batched
            or tr.split_k > 1
            or tr.active_tile_skip
        ):
            return False, (
                "persistent_ctas requires the default epilogue, a non-batched "
                "body and split_k == 1"
            )
        if tr.chiplet_swizzle and tr.persistent_ctas % max(1, tr.chiplet_num_xcds):
            return False, (
                f"persistent_ctas {tr.persistent_ctas} with chiplet_swizzle must be a "
                f"multiple of chiplet_num_xcds {tr.chiplet_num_xcds} (else the tiles "
                "of a workgroup leave the XCD the swizzle assigned them to)"
            )
    if tr.streamk and tr.persistent_ctas <= 0:
        return False, "streamk requires persistent_ctas > 0 (the stream-K pool)"
    if tr.streamk and t.warp_k != 1:
        return False, (
            f"streamk requires warp_k == 1 (got {t.warp_k}): the partial-tile "
            "slot is sized tile_m x tile_n"
        )
    if tr.sk_xcd_local and not (tr.streamk and tr.chiplet_swizzle):
        return False, (
            "sk_xcd_local requires streamk and chiplet_swizzle (the XCD-major "
            "rank order assumes workgroup w on XCD w mod chiplet_num_xcds)"
        )
    if tr.dtl_tight_rsrc and (tr.dtl_stages < 2 or spec.batched):
        return False, "dtl_tight_rsrc requires dtl_stages >= 2 and a non-batched body"
    if tr.pad_fast_tile and (
        not (tr.pad_m or tr.pad_n)
        or tr.epilogue != "default"
        or tr.split_k > 1
        or tr.mfma_swap_ab
    ):
        return False, (
            "pad_fast_tile requires pad_m or pad_n, the default epilogue, "
            "split_k == 1 and no mfma_swap_ab"
        )
    if tr.iglp_opt is not None and tr.iglp_opt not in (0, 1, 2, 3):
        return False, f"iglp_opt must be 0..3 (got {tr.iglp_opt})"
    if tr.acc_agpr and not (family == "mma" and arch in _AGPR_ARCHS):
        return False, (
            f"acc_agpr needs MFMA accumulators in AGPRs and is admitted on "
            f"gfx942 / gfx950 only (GPU numerics); got family {family!r} on {arch}"
        )
    if tr.mfma_interleave and arch not in _ILV_ARCHS:
        return False, (
            f"mfma_interleave is admitted on gfx942 / gfx950 only (GPU numerics); "
            f"got {arch}"
        )
    if (tr.lds_swizzle_r != -1 or tr.dtl_hoist) and arch not in _SWR_DH_ARCHS:
        return False, (
            f"lds_swizzle_r / dtl_hoist are admitted on gfx942 / gfx950 only "
            f"(GPU numerics); got {arch}"
        )
    if tr.ilv_dtl_every and (tr.ilv_dtl_every < 0 or not tr.mfma_interleave):
        return False, (
            f"ilv_dtl_every must be >= 0 and requires mfma_interleave "
            f"(got {tr.ilv_dtl_every})"
        )
    if tr.ilv_barrier_after and (
        not tr.mfma_interleave
        or not 0 < tr.ilv_barrier_after < t.mfmas_per_warp_m * t.mfmas_per_warp_n
    ):
        return False, (
            "ilv_barrier_after requires mfma_interleave and 0 < k < the MFMAs of "
            f"one K atom ({t.mfmas_per_warp_m * t.mfmas_per_warp_n}); got "
            f"{tr.ilv_barrier_after}"
        )
    if tr.ilv_merge and (not tr.mfma_interleave or tr.ilv_dtl_every):
        return False, "ilv_merge requires mfma_interleave and excludes ilv_dtl_every"
    _n_atom = t.mfmas_per_warp_m * t.mfmas_per_warp_n
    if (
        tr.ilv_early_release or tr.ilv_read_span or tr.ilv_dtl_span or tr.acc_out_vgpr
    ) and arch != "gfx950":
        return False, (
            "ilv_early_release / ilv_read_span / ilv_dtl_span / acc_out_vgpr are "
            f"admitted on gfx950 only (GPU numerics); got {arch}"
        )
    if tr.ilv_early_release and (not tr.mfma_interleave or t.k_atoms_per_tile_k < 2):
        return False, (
            "ilv_early_release requires mfma_interleave and >= 2 K atoms per "
            f"tile (got {t.k_atoms_per_tile_k})"
        )
    if tr.ilv_early_release not in (0, 1, 2, 3) or (
        tr.ilv_early_release >= 2 and tr.ilv_barrier_after
    ):
        return False, (
            "ilv_early_release must be 0..3; modes 2 / 3 exclude "
            f"ilv_barrier_after (got {tr.ilv_early_release})"
        )
    if tr.ilv_release_at and (
        tr.ilv_early_release < 2 or not 0 < tr.ilv_release_at < _n_atom
    ):
        return False, (
            "ilv_release_at requires ilv_early_release 2 / 3 and 0 < k < the MFMAs "
            f"of one K atom ({_n_atom}); got {tr.ilv_release_at}"
        )
    if tr.ilv_acquire_at and (
        not tr.ilv_early_release
        or not tr.ilv_barrier_after < tr.ilv_acquire_at < _n_atom
    ):
        return False, (
            "ilv_acquire_at requires ilv_early_release and ilv_barrier_after < k "
            f"< the MFMAs of one K atom ({_n_atom}); got {tr.ilv_acquire_at}"
        )
    if tr.lds_swizzle_r != -1 and not (tr.lds_swizzle and 0 <= tr.lds_swizzle_r <= 4):
        return False, (
            f"lds_swizzle_r must be -1 (auto) or 0..4 with lds_swizzle "
            f"(got {tr.lds_swizzle_r})"
        )
    if tr.dtl_hoist and tr.dtl_stages < 2:
        return False, "dtl_hoist requires dtl_stages >= 2"
    for _nm, _pc in (
        ("ilv_read_span", tr.ilv_read_span),
        ("ilv_dtl_span", tr.ilv_dtl_span),
    ):
        if _pc and (not tr.mfma_interleave or not 0 < _pc <= 100 or tr.ilv_dtl_every):
            return False, (
                f"{_nm} requires mfma_interleave (without ilv_dtl_every) and "
                f"0 < pct <= 100; got {_pc}"
            )
    if tr.acc_out_vgpr and not tr.acc_agpr:
        return False, "acc_out_vgpr requires acc_agpr"
    if tr.lds_row_interleave:
        if arch != "gfx950" or tr.dtl_stages < 2 or tr.lds_swizzle:
            return False, (
                "lds_row_interleave requires gfx950 (16-byte DTL), dtl_stages >= 2 "
                "and lds_swizzle off (the interleave replaces the XOR swizzle)"
            )
        if (
            spec.data.dtype_a not in ("fp16", "bf16")
            or spec.data.dtype_b != spec.data.dtype_a
            or t.tile_k != 64
            or (t.warp_tile_m, t.warp_tile_n, t.warp_tile_k) != (16, 16, 32)
            or t.mfmas_per_warp_m % 8
            or t.mfmas_per_warp_n % 8
            or spec.wave_size != 64
        ):
            return False, (
                "lds_row_interleave requires 16-bit A/B, tile_k 64 (128-byte LDS "
                "rows), the 16x16x32 atom, wave64 and mfmas_m / mfmas_n multiples "
                f"of 8 (got {spec.data.dtype_a}/{spec.data.dtype_b}, tile_k "
                f"{t.tile_k}, atom {t.warp_tile_m}x{t.warp_tile_n}x{t.warp_tile_k}, "
                f"mfmas {t.mfmas_per_warp_m}x{t.mfmas_per_warp_n})"
            )
    if tr.rilv_vec_epilogue and not (
        tr.lds_row_interleave
        and tr.epilogue == "default"
        and not tr.mfma_swap_ab
        and (tr.rve_pad or not (tr.pad_m or tr.pad_n))
        and tr.split_k == 1
    ):
        return False, (
            "rilv_vec_epilogue requires lds_row_interleave, the default epilogue, "
            "no mfma_swap_ab, split_k == 1 and no pad_m / pad_n unless rve_pad"
        )
    if tr.rve_pad not in (0, 1, 2):
        return False, f"rve_pad must be 0, 1 or 2 (got {tr.rve_pad})"
    if tr.rve_pad and not (
        tr.rilv_vec_epilogue and (tr.pad_m or tr.pad_n) and tr.dtl_tight_rsrc
    ):
        return False, (
            "rve_pad requires rilv_vec_epilogue, pad_m or pad_n and dtl_tight_rsrc "
            "(exact DTL bounds for the rows past M / N)"
        )
    if tr.rve_pad == 2 and spec.batched:
        return False, "rve_pad 2 (buffer stores over C) requires a non-batched body"
    if tr.rve_store_nt and not tr.rilv_vec_epilogue:
        return False, "rve_store_nt requires rilv_vec_epilogue"
    if tr.dtl_rsrc_word3 and not (
        tr.direct_to_lds
        and tr.dtl_stages >= 2
        and 0 < int(tr.dtl_rsrc_word3) < (1 << 32)
    ):
        return False, (
            "dtl_rsrc_word3 requires direct_to_lds with dtl_stages >= 2 and a "
            f"u32 value (got {tr.dtl_rsrc_word3})"
        )
    if tr.dtl_addr_spread and not (tr.hb_sched and tr.dtl_hoist):
        return False, "dtl_addr_spread requires hb_sched and dtl_hoist"
    if tr.k_stagger_rc and not (
        tr.k_stagger_rc in (2, 4, 8, 16, 32, 64)
        and tr.direct_to_lds
        and tr.dtl_stages >= 2
        and not tr.pingpong
        and not tr.streamk
        and not tr.asm_loop
        and tr.split_k == 1
    ):
        return False, (
            "k_stagger_rc requires a power of two 2..64, direct_to_lds, dtl_stages >= 2, "
            f"split_k 1 and no pingpong / streamk / asm_loop (got {tr.k_stagger_rc})"
        )
    if tr.dtl_tight_ragged_only and not (
        tr.dtl_tight_rsrc and tr.dtl_hoist and (tr.pad_m or tr.pad_n)
    ):
        return False, (
            "dtl_tight_ragged_only requires dtl_tight_rsrc, dtl_hoist and pad_m / pad_n"
        )
    if tr.hb_skew and not (
        tr.hb_skew in (1, 2) and tr.hb_sched and not tr.asm_loop and not tr.pingpong
    ):
        return False, (
            f"hb_skew requires 1 or 2 with hb_sched, no asm_loop / pingpong "
            f"(got {tr.hb_skew})"
        )
    if tr.hb_tile_lgkm0 and not (
        tr.hb_tile_lgkm0 == 1
        and tr.hb_sched
        and tr.dtl_frag_prefetch
        and tr.mfma_interleave
        and not tr.asm_loop
        and not tr.pingpong
    ):
        return False, (
            "hb_tile_lgkm0 requires 1 with hb_sched + dtl_frag_prefetch + "
            f"mfma_interleave, no asm_loop / pingpong (got {tr.hb_tile_lgkm0})"
        )
    if tr.k_stagger and not (
        tr.k_stagger in (2, 4, 8, 16, 32, 64)
        and tr.direct_to_lds
        and tr.dtl_stages >= 2
        and not tr.dtl_tight_rsrc
        and not tr.streamk
        and int(tr.split_k) == 1
        and not tr.pad_k
        and not tr.pingpong
    ):
        return False, (
            "k_stagger requires 2..64 (power of two) with direct_to_lds, "
            "dtl_stages >= 2, no dtl_tight_rsrc / streamk / split_k / pad_k / "
            f"pingpong (got {tr.k_stagger})"
        )
    if tr.dtl_clamp_k and not (
        tr.dtl_soff0
        and tr.dtl_hoist
        and not tr.dtl_tight_rsrc
        and not tr.persistent_ctas
        and not tr.streamk
        and int(tr.split_k) == 1
        and not tr.pad_k
        and not tr.pingpong
        and not tr.epilogue == "cshuffle"
    ):
        return False, (
            "dtl_clamp_k requires dtl_soff0 + dtl_hoist, a non-persistent grid, "
            "no dtl_tight_rsrc / streamk / split_k / pad_k / pingpong / cshuffle"
        )
    if tr.k_stagger_map and not (tr.k_stagger and tr.k_stagger_map == 1):
        return False, "k_stagger_map requires k_stagger and 1"
    if tr.k_stagger_tiles and not (tr.k_stagger and 0 < tr.k_stagger_tiles <= 16):
        return False, "k_stagger_tiles requires k_stagger and 1..16"
    if tr.das_late and not (tr.dtl_addr_spread and tr.hb_sched):
        return False, "das_late requires dtl_addr_spread (hb_sched)"
    if tr.das_anchor and not tr.dtl_addr_spread:
        return False, "das_anchor requires dtl_addr_spread"
    if tr.hb_skew_dtl and not (tr.hb_skew and 0 < tr.hb_skew_dtl <= 8):
        return False, (
            f"hb_skew_dtl requires hb_skew and 1..8 slots (got {tr.hb_skew_dtl})"
        )
    if tr.dtl_base_fold and not (
        tr.dtl_soff0
        and tr.dtl_hoist
        and not tr.dtl_tight_rsrc
        and not tr.persistent_ctas
        and not tr.streamk
    ):
        return False, (
            "dtl_base_fold requires dtl_soff0 + dtl_hoist, no dtl_tight_rsrc and "
            "a non-persistent grid"
        )
    if (tr.stagger_u or tr.stagger_u_stride != 1 or tr.stagger_u_map) and not (
        tr.stagger_u in (2, 4, 8, 16, 32, 64)
        and 1 <= tr.stagger_u_stride <= 64
        and tr.stagger_u_map in (0, 1, 2)
        and (tr.stagger_u_map != 2 or tr.chiplet_swizzle)
        and tr.direct_to_lds
        and tr.dtl_stages >= 2
        and not tr.persistent_ctas
        and not tr.streamk
        and not tr.pingpong
        and int(tr.split_k) == 1
    ):
        return False, (
            "stagger_u requires a power of two 2..64 (stride 1..64 tiles, map 0..2; "
            "map 2 needs chiplet_swizzle) "
            "with direct_to_lds and dtl_stages >= 2, no persistent_ctas / streamk / pingpong / split-K "
            f"(got {tr.stagger_u}/{tr.stagger_u_stride}/{tr.stagger_u_map})"
        )
    if bool(tr.k_stagger) + bool(tr.stagger_u) + bool(tr.k_stagger_rc) > 1:
        return False, (
            "k_stagger, stagger_u and k_stagger_rc are alternative StaggerU "
            "forms (at most one)"
        )
    if tr.stagger_u and tr.dtl_clamp_k:
        return False, "stagger_u does not combine with dtl_clamp_k"
    if tr.mfma_serpentine and not (tr.hb_sched or tr.sia_sched or tr.mfma_interleave):
        return False, (
            "mfma_serpentine requires hb_sched / asm_loop / sia_sched / "
            "mfma_interleave"
        )
    if tr.mfma_order and not (
        tr.mfma_order in (2, 3)
        and (tr.hb_sched or tr.sia_sched or tr.mfma_interleave)
        and (tr.mfma_order != 2 or tr.hb_sched or tr.sia_sched)
        and not tr.asm_loop
        and not tr.mfma_serpentine
    ):
        return False, (
            "mfma_order requires 2 or 3 (order 1 is mfma_serpentine) with "
            "hb_sched / sia_sched / mfma_interleave (2: hb_sched / sia_sched "
            f"only), no asm_loop and no mfma_serpentine (got {tr.mfma_order})"
        )
    if tr.hb_dtl_delay and not (tr.hb_sched and 0 < tr.hb_dtl_delay <= 16):
        return False, (
            f"hb_dtl_delay requires hb_sched and 1..16 slots (got {tr.hb_dtl_delay})"
        )
    if tr.kloop_unroll != 1 and not (
        tr.kloop_unroll in (2, 4)
        and tr.direct_to_lds
        and tr.dtl_stages >= 2
        and not tr.pingpong
        and not tr.streamk
    ):
        return False, (
            "kloop_unroll requires 2 or 4 with direct_to_lds, dtl_stages >= 2, "
            f"no pingpong / streamk (got {tr.kloop_unroll})"
        )
    if tr.dtl_soff0 and not (
        tr.direct_to_lds
        and tr.dtl_stages >= 2
        and tr.dtl_hoist
        and not tr.dtl_tight_rsrc
        and not tr.pingpong
    ):
        return False, (
            "dtl_soff0 requires direct_to_lds, dtl_stages >= 2 and dtl_hoist, "
            "without dtl_tight_rsrc / pingpong"
        )
    if tr.rilv_pad != 16 and not (
        tr.lds_row_interleave and tr.rilv_pad in (32, 48, 64)
    ):
        return False, (
            "rilv_pad requires lds_row_interleave and one of 16, 32, 48, 64 bytes "
            f"(got {tr.rilv_pad})"
        )
    if tr.hb_sched:
        if not (tr.dtl_frag_prefetch and tr.mfma_interleave):
            return False, "hb_sched requires dtl_frag_prefetch and mfma_interleave"
        if (
            tr.ilv_merge
            or tr.ilv_barrier_after
            or tr.ilv_dtl_every
            or tr.iglp_opt is not None
        ):
            return False, (
                "hb_sched is its own MFMA-slot schedule: it excludes ilv_merge, "
                "ilv_barrier_after, ilv_dtl_every and iglp_opt"
            )
        _ntot = t.k_atoms_per_tile_k * t.mfmas_per_warp_m * t.mfmas_per_warp_n
        if tr.hb_split_rel and (tr.hb_rel_slot >= 0 or t.k_atoms_per_tile_k < 2):
            return False, (
                "hb_split_rel places both releases itself (hb_rel_slot must stay "
                "-1) and needs >= 2 K atoms per tile"
            )
        if tr.hb_rel_gap < 1:
            return False, f"hb_rel_gap must be >= 1 (got {tr.hb_rel_gap})"
        if tr.hb_acq_b_slot >= 0 and not (hb_slots(spec)[1] < tr.hb_acq_b_slot < _ntot):
            return False, (
                f"hb_acq_b_slot {tr.hb_acq_b_slot} must lie after the A acquire "
                f"({hb_slots(spec)[1]}) and below {_ntot}"
            )
        if not (-1 <= tr.hb_rel_slot < _ntot and -1 <= tr.hb_acq_slot < _ntot):
            return False, (
                f"hb_rel_slot / hb_acq_slot must be -1 (auto) or an MFMA index "
                f"< {_ntot} (got {tr.hb_rel_slot} / {tr.hb_acq_slot})"
            )
        if tr.hb_read_every < 1 or tr.hb_dtl_every < 0:
            return False, (
                f"hb_read_every must be >= 1 and hb_dtl_every >= 0 (got "
                f"{tr.hb_read_every} / {tr.hb_dtl_every})"
            )
        _last_rd = hb_last_read_slot(spec)
        if 0 <= tr.hb_rel_slot <= _last_rd:
            return False, (
                f"hb_rel_slot {tr.hb_rel_slot} must come after the last intra-tile "
                f"fragment read (MFMA {_last_rd}): the release frees the stage"
            )
    elif (
        tr.hb_rel_slot,
        tr.hb_acq_slot,
        tr.hb_read_every,
        tr.hb_dtl_every,
        tr.hb_acq_b_slot,
        tr.hb_split_rel,
        tr.hb_rel_gap,
    ) != (-1, -1, 1, 0, -1, False, 4):
        return (
            False,
            "hb_rel_slot / hb_acq_slot / hb_read_every / hb_dtl_every require hb_sched",
        )
    if tr.sia_sched:
        if not (tr.dtl_frag_prefetch and tr.mfma_interleave and tr.lds_row_interleave):
            return False, (
                "sia_sched requires dtl_frag_prefetch, mfma_interleave and "
                "lds_row_interleave"
            )
        if (
            tr.ilv_merge
            or tr.ilv_barrier_after
            or tr.ilv_dtl_every
            or tr.iglp_opt is not None
            or tr.ilv_read_span
            or tr.ilv_dtl_span
            or tr.mfma_swap_ab
            or tr.streamk
        ):
            return False, (
                "sia_sched is its own MFMA-slot schedule: it excludes ilv_merge, "
                "ilv_barrier_after, ilv_dtl_every, ilv_read_span, ilv_dtl_span, "
                "iglp_opt, mfma_swap_ab and streamk"
            )
        if t.k_atoms_per_tile_k != 2:
            return False, "sia_sched needs exactly 2 K atoms per tile"
        if (
            tr.sia_read_every < 1
            or tr.sia_rel_gap < 1
            or tr.sia_acq_gap < (-t.mfmas_per_warp_n if tr.sia_merge_sync else 0)
            or tr.sia_dtl_stride < 1
        ):
            return False, (
                "sia_read_every / sia_rel_gap / sia_dtl_stride must be >= 1 and "
                "sia_acq_gap >= 0 (>= -mfmas_n with sia_merge_sync)"
            )
        if (tr.sia_use_order or tr.sia_merge_sync) and (
            tr.sia_split_rel or tr.sia_split_acq or tr.sia_tensile
        ):
            return False, (
                "sia_use_order / sia_merge_sync need one release and one acquire "
                "(no sia_split_rel / sia_split_acq / sia_tensile)"
            )
        try:
            sia_table(spec)
        except ValueError as e:
            return False, f"sia_sched: {e}"
    elif (
        tr.sia_b_first,
        tr.sia_plr,
        tr.sia_single_lgkm,
        tr.sia_split_rel,
        tr.sia_split_acq,
        tr.sia_dtl_spread,
        tr.sia_read_every,
        tr.sia_rel_gap,
        tr.sia_acq_gap,
        tr.sia_dtl_stride,
        tr.sia_tensile,
        tr.sia_use_order,
        tr.sia_merge_sync,
    ) != (False, False, False, False, False, False, 2, 4, 3, 3, False, False, False):
        return False, "sia_* sub-levers require sia_sched"
    if tr.fr0_loop_order and (
        tr.asm_loop or not (tr.dtl_frag_prefetch and tr.mfma_interleave)
    ):
        return False, (
            "fr0_loop_order requires dtl_frag_prefetch + mfma_interleave and no "
            "asm_loop (asm_loop orders the first fragments itself)"
        )
    if tr.asm_loop not in _asm.ASM_LOOP_MODES:
        return (
            False,
            f"asm_loop must be one of {_asm.ASM_LOOP_MODES} (got {tr.asm_loop!r})",
        )
    if not tr.asm_loop and (
        tr.asm_loop_table or tr.asm_loop_prio or tr.asm_loop_skew or tr.asm_loop_rawrd
    ):
        return False, "asm_loop_table / asm_loop_prio / asm_loop_skew require asm_loop"
    if tr.asm_loop:
        if tr.asm_loop_table and tr.asm_loop != "table":
            return False, "asm_loop_table requires asm_loop='table'"
        if not (tr.hb_sched and tr.lds_row_interleave and tr.dtl_stages == 2):
            return (
                False,
                "asm_loop requires hb_sched, lds_row_interleave and dtl_stages 2",
            )
        if (
            tr.hb_rel_slot,
            tr.hb_acq_slot,
            tr.hb_read_every,
            tr.hb_dtl_every,
            tr.hb_acq_b_slot,
            tr.hb_split_rel,
            tr.hb_rel_gap,
        ) != (-1, -1, 1, 0, -1, False, 4):
            return False, "asm_loop places every event itself (no hb_* slot knob)"
        # dtl_addr_spread pins the DTL address
        # steps to the table's free MFMA slots before its first DTL issue, and
        # hb_dtl_delay moves every DTL issue of the table that many slots later
        # (both native to the asm table; see _tile_asm)
        if (
            (t.tile_m, t.tile_n, t.tile_k) != (256, 256, 64)
            or (t.warp_m, t.warp_n, t.warp_k) != (2, 2, 1)
            or (t.warp_tile_m, t.warp_tile_n, t.warp_tile_k) != (16, 16, 32)
        ):
            return False, (
                "asm_loop tables are written for the 256x256x64 tile, 2x2x1 waves "
                "and 16x16x32 atoms (8 x 8 atoms, 2 K atoms, 8 + 8 DTL per tile)"
            )
        if tr.streamk or tr.interwave_setprio:
            return False, "asm_loop excludes streamk and interwave_setprio"
        if not 0 <= tr.asm_loop_prio <= 3:
            return False, f"asm_loop_prio must be 0..3 (got {tr.asm_loop_prio})"
        try:
            _why = _asm.check_table(_asm.table_events(tr))
        except (ValueError, KeyError) as _e:
            _why = str(_e)
        if _why:
            return False, f"asm_loop table: {_why}"
        if tr.asm_loop_skew not in (0, 1, 2):
            return False, f"asm_loop_skew must be 0, 1 or 2 (got {tr.asm_loop_skew})"
        if tr.asm_loop_skew:
            try:
                _why = _asm.check_skew(tr)
            except (ValueError, KeyError) as _e:
                _why = str(_e)
            if _why:
                return False, f"asm_loop_skew: {_why}"
    if tr.pp_wait_late and not tr.pingpong:
        return False, "pp_wait_late requires pingpong"
    if tr.pp_spread_lo and not (
        tr.pingpong == 6
        and 0
        < tr.pp_spread_lo
        < t.mfmas_per_warp_m * t.mfmas_per_warp_n * t.k_atoms_per_tile_k
    ):
        return False, (
            "pp_spread_lo requires pingpong 6 and 0 < k < the MFMAs of one K tile "
            f"(got {tr.pp_spread_lo})"
        )
    if tr.pp_mix_reads and (tr.pp_mix_reads < 0 or tr.pingpong < 4):
        return False, "pp_mix_reads must be >= 0 and requires pingpong 4 / 5 / 6"
    if tr.pingpong:
        _nw = t.warp_m * t.warp_n * t.warp_k
        if tr.pingpong not in (1, 2, 3, 4, 5, 6, 7):
            return False, f"pingpong must be 0..7 (got {tr.pingpong})"
        if tr.pingpong == 7 and (tr.dtl_hoist or tr.dtl_tight_rsrc):
            return (
                False,
                "pingpong 7 builds its own DMA resources (no dtl_hoist / dtl_tight_rsrc)",
            )
        if tr.pingpong in (6, 7):
            _eb = 2 if spec.data.dtype_a in ("fp16", "bf16") else 4
            _l6 = (2 * t.tile_m + 3 * t.tile_n) * t.tile_k * _eb
            if _l6 > 163840:
                return (
                    False,
                    f"pingpong 6 / 7 need 2 A + 3 B stages = {_l6} B of LDS > 163840",
                )
        if tr.pingpong >= 4 and tr.dtl_stages != 2:
            return False, "pingpong 4 / 5 / 6 requires dtl_stages == 2"
        if tr.pingpong < 4 and tr.dtl_stages < 3:
            return False, "pingpong 1..3 requires dtl_stages >= 3"
        if tr.pingpong >= 4 and tr.pp_wait_late:
            return False, "pingpong 4 / 5 place their own waits (exclude pp_wait_late)"
        if tr.pingpong >= 4 and (
            tr.epilogue != "default" or tr.split_k > 1 or spec.batched
        ):
            return False, (
                "pingpong 4 emits one K loop + epilogue per wave group: it needs the "
                "default epilogue (a cshuffle C tile per copy would not fit the LDS), "
                "split_k == 1 and a non-batched body"
            )
        if t.warp_k != 1 or _nw < 2 or _nw % 2:
            return False, "pingpong requires warp_k == 1 and an even wave count"
        if (
            tr.dtl_frag_prefetch
            or tr.dtl_stream_barrier
            or tr.dtl_issue_spread
            or tr.mfma_interleave
            or tr.iglp_opt is not None
            or (tr.persistent_ctas and tr.pingpong < 4)
            or tr.streamk
        ):
            return False, (
                "pingpong is its own staged order: it excludes dtl_frag_prefetch, "
                "dtl_stream_barrier, dtl_issue_spread, mfma_interleave, iglp_opt "
                "and streamk (persistent_ctas: pingpong 4 / 5 only)"
            )
    # ---- the K-tile schedule families are separate bodies; at most one of
    # them per kernel ----
    _sched_fams = [
        n_
        for n_, on in (
            ("ilv_early_release", tr.ilv_early_release),
            ("ilv_early_free", tr.ilv_early_free),
            ("ilv_split_wait", tr.ilv_split_wait),
            ("hb_sched", tr.hb_sched),
            ("sia_sched", tr.sia_sched),
            ("pingpong", tr.pingpong),
        )
        if on
    ]
    if len(_sched_fams) > 1:
        return False, (
            "at most one K-tile schedule family per kernel (got "
            f"{', '.join(_sched_fams)})"
        )
    if tr.hb_sched and (tr.ilv_read_span or tr.ilv_dtl_span):
        return (
            False,
            "hb_sched places its own reads / DTL issues (no ilv_read_span / ilv_dtl_span)",
        )
    if tr.pingpong and tr.lds_row_interleave:
        return False, "pingpong builds its own DTL lane layout (no lds_row_interleave)"
    if tr.acc_agpr_pin and not tr.acc_agpr:
        return False, "acc_agpr_pin requires acc_agpr"
    if tr.mfma_interleave and not tr.dtl_frag_prefetch:
        return False, "mfma_interleave requires dtl_frag_prefetch (dtl_stages >= 2)"
    if tr.ilv_split_wait:
        _sw_n = (t.warp_tile_m and (t.tile_m // t.warp_m) // t.warp_tile_m) * (
            t.warp_tile_n and (t.tile_n // t.warp_n) // t.warp_tile_n
        )
        if not tr.mfma_interleave or tr.ilv_early_free:
            return (
                False,
                "ilv_split_wait requires mfma_interleave without ilv_early_free",
            )
        if not tr.ilv_barrier_after < tr.ilv_split_wait < _sw_n:
            return False, (
                f"ilv_split_wait must be in (ilv_barrier_after, MFMAs per K atom = "
                f"{_sw_n}) (got {tr.ilv_split_wait})"
            )
    if tr.ilv_ef_spread and not tr.ilv_early_free:
        return False, "ilv_ef_spread requires ilv_early_free"
    if tr.ilv_early_free:
        _ef_n = (t.warp_tile_m and (t.tile_m // t.warp_m) // t.warp_tile_m) * (
            t.warp_tile_n and (t.tile_n // t.warp_n) // t.warp_tile_n
        )
        _ef_ka = t.warp_tile_k and t.tile_k // t.warp_tile_k
        if not tr.mfma_interleave or _ef_ka < 2:
            return False, (
                "ilv_early_free requires mfma_interleave and >= 2 K atoms per tile"
            )
        if not 4 <= tr.ilv_early_free < _ef_n:
            return False, (
                f"ilv_early_free must be in [4, MFMAs per K atom = {_ef_n}) "
                f"(got {tr.ilv_early_free})"
            )
    if tr.mfma_interleave and tr.dtl_issue_spread:
        return False, (
            "mfma_interleave already spreads the DTL issues; it excludes "
            "dtl_issue_spread"
        )
    if tr.acc_agpr and (spec.wave_size != 64 or acc_agpr_regs(spec) > 256):
        return False, (
            f"acc_agpr needs wave64 and <= 256 accumulator registers per lane "
            f"(got {acc_agpr_regs(spec)})"
        )
    if tr.mfma_swap_ab:
        if family != "mma":
            return False, f"mfma_swap_ab is MFMA-only (got family {family!r} on {arch})"
        if tr.epilogue != "default" or tr.pad_n or tr.split_k > 1:
            return False, (
                "mfma_swap_ab requires the default (direct) epilogue without "
                "pad_n or split-K"
            )
        if t.warp_tile_m != t.warp_tile_n:
            return False, "mfma_swap_ab requires warp_tile_m == warp_tile_n"

    # LDS write-granularity rules (found by a lever-combination sweep on
    # gfx950: many DTL and compv4-DB points this gate used to accept faulted or
    # computed wrong results).
    #   * DTL: every DMA pass must be full. A tile with fewer chunks than
    #     ``block_size`` (or a non-multiple) lets the surplus lanes write past
    #     the LDS tile (into the other ping-pong half / the B tile) and read
    #     past the global tile (illegal address on the last row of blocks).
    #   * lds_swizzle: the XOR moves whole ``2^L``-half groups; the LDS write
    #     chunk (the DTL chunk, or the VGPR path's ``load_vec`` store) must not
    #     be wider than a group, or one chunk is scattered across groups (the
    #     16x16x16 / 32x32x8 atoms give L=2 against 8-half chunks).
    if family == "mma":
        _dtl_h = min(4, target.async_lds_max_dwords) * 2
        if spec.trait.direct_to_lds:
            for _nm, _rows in (("A", t.tile_m), ("B", t.tile_n)):
                _chunks = (_rows * t.tile_k) // _dtl_h
                if _chunks % spec.block_size:
                    return False, (
                        f"direct_to_lds: the {_nm} tile has {_chunks} {_dtl_h}-half "
                        f"chunks, not a multiple of block_size {spec.block_size} "
                        "(a partial DMA pass writes past the LDS tile)"
                    )
        if spec.trait.lds_swizzle:
            _op = _resolve_mma_op(spec, arch)
            _swz_l = lds_swizzle_params(_op.a_frag_len, _op.b_frag_len, t.tile_k)[2]
            if spec.trait.direct_to_lds:
                _chunk = _dtl_h
            else:
                try:
                    _chunk = _choose_load_vec(spec)
                except ValueError as e:
                    return False, str(e)
            if _swz_l < 0 or (1 << _swz_l) % _chunk:
                return False, (
                    f"lds_swizzle: XOR group of {1 << max(_swz_l, 0)} halves is "
                    f"narrower than the {_chunk}-half LDS write chunk (atom "
                    f"{atom}); the swizzle would split a chunk"
                )

    # LDS budget. The cap is the target's per-WG LDS capacity (160 KiB on
    # gfx950 / CDNA4, 64 KiB on gfx942 / CDNA3). The bytes are what the body
    # really allocates (:func:`universal_lds_bytes`): the AB staging (doubled
    # when ping-ponged, row-padded by ``lds_k_pad``) and, for the cshuffle
    # epilogue, the C tile -- which the smem-pool packer ALIASES onto the A/B
    # bytes unless ``cshuffle_no_alias`` (pool = max(ab, c), not ab + c). The
    # previous additive formula rejected e.g. the 256x256x64 DTL-prefetch
    # cshuffle tile (claimed 256 KiB, really 128 KiB).
    bytes_lds, ab_bytes, c_bytes = universal_lds_bytes(
        t.tile_m,
        t.tile_n,
        t.tile_k,
        pipeline=spec.trait.pipeline,
        epilogue=spec.trait.epilogue,
        direct_to_lds=spec.trait.direct_to_lds,
        dtl_prefetch=spec.trait.dtl_prefetch,
        ab_double_buffer=spec.trait.ab_double_buffer,
        cshuffle_no_alias=spec.trait.cshuffle_no_alias,
        lds_k_pad=spec.trait.lds_k_pad,
        lds_cap=target.lds_capacity_bytes,
        dtl_stages=spec.trait.dtl_stages,
    )
    if spec.trait.lds_row_interleave:
        # rilv_pad bytes of padding per 8 LDS rows, per operand and stage
        _rpad = (
            ((t.tile_m + t.tile_n) // 8)
            * int(spec.trait.rilv_pad)
            * int(spec.trait.dtl_stages)
        )
        ab_bytes += _rpad
        bytes_lds = (
            ab_bytes + c_bytes
            if spec.trait.cshuffle_no_alias
            else max(ab_bytes, c_bytes)
        )
    if not target.fits_lds(bytes_lds):
        return False, (
            f"LDS budget {bytes_lds} > {target.lds_capacity_bytes} cap "
            f"(AB={ab_bytes}, C={c_bytes}) on {arch}"
        )

    # Stream-K co-residency: an owner spins on
    # higher-index workgroups of the pool, so all P must be resident at once.
    if spec.trait.streamk:
        _res, _per_cu = streamk_resident_ctas(spec, arch, bytes_lds)
        if spec.trait.persistent_ctas > _res:
            return False, (
                f"streamk needs all persistent_ctas = {spec.trait.persistent_ctas} "
                f"workgroups co-resident, but at most {_res} fit on {arch} "
                f"({_res // _per_cu} CUs x {_per_cu} per CU: LDS {bytes_lds} of "
                f"{target.lds_capacity_bytes} B, "
                f"{t.warp_m * t.warp_n * t.warp_k} waves at the 512-register worst case)"
            )

    # Per-WG thread cap (hardware limit from the target).
    if spec.block_size > target.max_threads_per_block:
        return False, (
            f"block_size {spec.block_size} > {target.max_threads_per_block} "
            f"(hardware cap) on {arch}"
        )

    # Global -> LDS vectorised load divisibility: at least one vec
    # element per thread per phase.
    threads = spec.block_size
    a_total = t.tile_m * t.tile_k
    b_total = t.tile_n * t.tile_k
    if a_total < threads or b_total < threads:
        return False, "block too small for one element/thread/phase"

    # Split-K (over the production body): the K-slice each CTA processes
    # is ``ks = K // split_k`` and must itself be a whole number of
    # K-tiles. We can only validate the K-slice divisibility at build
    # time when K is a compile-time fact, which it is not in the
    # universal body (K is a runtime arg). So we only check the
    # static invariants here: split_k >= 1, and the atomic-add epilogue
    # is only wired for the MFMA (CDNA) family. The K % split_k and
    # ks % tile_k divisibility are the caller's responsibility (mirrors
    # the v1 StreamK contract, where the partitioner enforces it).
    sk = spec.trait.split_k
    if sk < 1:
        return False, f"split_k must be >= 1 (got {sk})"
    if sk > 1 and family != "mma":
        return False, f"split_k > 1 is CDNA-only (got family {family!r} on {arch})"

    return True, "ok"


# ---------------------------------------------------------------------
# IR builders for the pieces
# ---------------------------------------------------------------------


def _mfma_atom_widths(spec: UniversalGemmSpec) -> Tuple[int, int, int]:
    """Return (a_per_lane, b_per_lane, c_per_lane) for the spec's MFMA atom.

    Kept for the MFMA-only callers (``moe_gemm_fused``) that compute the
    per-lane widths straight from the wave64 geometry. The contract-driven
    GEMM body uses :func:`_atom_frag_lengths` (op-sourced) instead, which is
    equal to this for MFMA but correct for WMMA too.
    """
    t = spec.tile
    waves = spec.wave_size
    a_per = (t.warp_tile_m * t.warp_tile_k) // waves
    b_per = (t.warp_tile_k * t.warp_tile_n) // waves
    c_per = (t.warp_tile_m * t.warp_tile_n) // waves
    return a_per, b_per, c_per


def mfma_order_perm(mo: int, mfmas_m: int, mfmas_n: int) -> Tuple[List[int], List[int]]:
    """The MFMA issue order of a K atom: the accumulator index
    (``mi * mfmas_n + ni``) issued at each MFMA slot, for even and odd K atoms.

    0 = row-major, 1 = serpentine rows (``TraitSpec.mfma_serpentine``), 2 =
    serpentine rows with odd atoms reversed, 3 = serpentine columns (2 and 3:
    ``TraitSpec.mfma_order``). Each list is a permutation of
    ``range(mfmas_m * mfmas_n)``, so every accumulator is updated exactly once
    per K atom and its K atoms are summed in the same order for every mode.
    """
    if mo in (1, 2):
        even = [
            mi * mfmas_n + (ni if mi % 2 == 0 else mfmas_n - 1 - ni)
            for mi in range(mfmas_m)
            for ni in range(mfmas_n)
        ]
    elif mo == 3:
        even = [
            (mi if ni % 2 == 0 else mfmas_m - 1 - mi) * mfmas_n + ni
            for ni in range(mfmas_n)
            for mi in range(mfmas_m)
        ]
    else:
        even = list(range(mfmas_m * mfmas_n))
    return even, (even[::-1] if mo == 2 else list(even))


def _emit_mfma(
    b: IRBuilder, spec: UniversalGemmSpec, a: Value, bb: Value, c: Value
) -> Value:
    """Emit the MFMA for ``spec``'s warp-tile atom (MFMA-only callers).

    Routes through the same op_id the contract resolver picks, so the emission
    is byte-identical to the prior ISA-named helpers. The unified GEMM body
    uses :func:`_emit_mma` (op-driven, works for WMMA too).
    """
    t = spec.tile
    key = (t.warp_tile_m, t.warp_tile_n, t.warp_tile_k)
    dtype = _storage_dtype(spec)
    if dtype == F16:
        if key == (16, 16, 16):
            return b.mfma_f32_16x16x16_f16(a, bb, c)
        if key == (16, 16, 32):
            return b.mfma_f32_16x16x32_f16(a, bb, c)
        if key == (32, 32, 8):
            return b.mfma_f32_32x32x8_f16(a, bb, c)
        if key == (32, 32, 16):
            return b.mfma_f32_32x32x16_f16(a, bb, c)
    if dtype == BF16:
        if key == (16, 16, 16):
            return b.mfma_f32_16x16x16_bf16(a, bb, c)
        if key == (16, 16, 32):
            return b.mfma_f32_16x16x32_bf16(a, bb, c)
    raise NotImplementedError(f"no MFMA emitter for {dtype.name} warp_tile {key}")


def _emit_zero_acc(b: IRBuilder, spec: UniversalGemmSpec) -> Value:
    """Zero accumulator vector sized from the spec geometry (MFMA-only callers)."""
    _, _, c_per = _mfma_atom_widths(spec)
    return b.zero_vec_f32(c_per)


def _atom_frag_lengths(op) -> Tuple[int, int, int]:
    """Return ``(a_frag_len, b_frag_len, c_frag_len)`` — the per-lane fragment
    vector lengths of the resolved :class:`~rocke.core.arch.MmaOp`.

    These come from the MMA contract, **not** from the ``shape / wave`` formula:
    the two are equal for MFMA (wave64) but diverge for WMMA, where the
    hardware replicates each operand fragment across the two half-waves
    (``a_frag_len == 16`` for the 16x16x16 atom, where ``16*16/32 == 8`` would
    be wrong). Driving everything off ``op.*_frag_len`` is what lets one body
    emit both ISAs.
    """
    return op.a_frag_len, op.b_frag_len, op.c_frag_len


def _emit_mma(b: IRBuilder, op, a: Value, bb: Value, c: Value) -> Value:
    """Target-neutral matmul: ``D = a * bb + c`` via :meth:`IRBuilder.mma`.

    The backend dispatches ``op.op_id`` to the matching MFMA call on CDNA or
    the WMMA call on RDNA. This reproduces the prior ISA-named helper emission
    byte-for-byte on CDNA (the helpers were already thin wrappers over
    :meth:`mma`).
    """
    return b.mma(op, a, bb, c)


def _emit_zero_acc_op(b: IRBuilder, op) -> Value:
    """Zero accumulator vector sized from the resolved op's ``c_frag_len``."""
    return b.zero_vec_f32(op.c_frag_len)


def _choose_load_vec(spec: UniversalGemmSpec) -> int:
    """Choose the widest naturally-aligned global-load width for this
    block shape. f16 -> we can vectorise up to 8 halves per lane.

    Thin adapter over :func:`rocke.helpers.spec.choose_load_vec`, the shared
    single-source picker; kept so callers (incl. ``moe_gemm_fused``) that
    pass a whole spec stay unchanged."""
    t = spec.tile
    return choose_load_vec(t.tile_m, t.tile_n, t.tile_k, spec.block_size)


def _emit_smem_load(
    b: IRBuilder, smem: Value, row: Value, col: Value, n: int, dtype: Type
) -> Value:
    if dtype == F16 and n == 4:
        return b.smem_load_v4_f16(smem, row, col)
    return b.smem_load_vN(smem, row, col, dtype=dtype, n=n)


def _hotloop_inst_list(spec: UniversalGemmSpec, load_vec: int):
    """Build the per-K-tile :class:`HotLoopInstList` for ``spec``.

    Pure arithmetic from the block tile geometry + the spec's MFMA atom
    timing (``MfmaAtom.mfma_cycle`` / ``k_per_xdlops``), mirroring CK's
    ``BlockwiseGemmXdlops_pipeline_hotloop_inst``. The global buffer-load
    width is the kernel's coalesced ``load_vec`` (elements/lane/load); the
    LDS read/write widths default to the atom K-pack inside
    :meth:`HotLoopInstList.from_geometry` (the comp_v4 ``*_LDS_*_Width =
    KPerXDL`` convention).
    """
    from ...helpers.atoms import mfma_atom
    from ...helpers.schedule import HotLoopInstList

    t = spec.tile
    atom = mfma_atom(spec.data.dtype_a, t.warp_tile_m, t.warp_tile_n, t.warp_tile_k)
    return HotLoopInstList.from_geometry(
        atom=atom,
        block_size=spec.block_size,
        m_per_block=t.tile_m,
        n_per_block=t.tile_n,
        k_per_block=t.tile_k,
        m_repeat=t.mfmas_per_warp_m,
        n_repeat=t.mfmas_per_warp_n,
        a_buffer_load_width=load_vec,
        b_buffer_load_width=load_vec,
    )


def _hotloop_well_formed(il, pipeline: str) -> bool:
    """Whether the v3/v4 schedule for ``il`` yields only non-negative counts.

    The CK HotLoopScheduler is only instantiated for tiles large enough that
    every emitted ``sched_group_barrier`` count is >= 0. For very small / K-
    light tiles the integer-divided LDS counts collapse to 0 and the v4
    trailing ``C_MFMA / num_issue - 3`` group can go negative, which is not a
    legal hint. In those degenerate cases the caller keeps the flat hint
    (still a pure scheduling choice -> numerically identical either way).
    """
    num_buffer_load = il.a_buffer_load_inst_num + il.b_buffer_load_inst_num
    if num_buffer_load <= 0:
        return False
    if pipeline == "compv3":
        num_mfma_stage1 = il.c_mfma_inst_num - (
            il.num_dsread_a_mfma + il.num_dsread_b_mfma
        )
        if num_mfma_stage1 < 0:
            return False
        num_mfma_per_issue = num_mfma_stage1 // num_buffer_load
        # Stage-1 groups emit (num_mfma_per_issue - num_dswrite_per_issue_*)
        # MFMAs; that count must stay non-negative.
        if il.a_buffer_load_inst_num <= 0 or il.b_buffer_load_inst_num <= 0:
            return False
        dswr_a = il.a_lds_write_inst_num // il.a_buffer_load_inst_num
        dswr_b = il.b_lds_write_inst_num // il.b_buffer_load_inst_num
        return (num_mfma_per_issue - dswr_a) >= 0 and (num_mfma_per_issue - dswr_b) >= 0
    # compv4: the trailing MFMA group is C_MFMA / num_issue - 3.
    return (il.c_mfma_inst_num // num_buffer_load - 3) >= 0


def _emit_hotloop_schedule(b: IRBuilder, spec: UniversalGemmSpec, load_vec: int):
    """Emit the comp_v3 / comp_v4 two-stage HotLoop schedule once per K-tile.

    Replaces the old flat per-kk ``sched_group_barrier`` hint. Falls back to
    the flat hint for degenerate small tiles whose derived counts would
    produce an illegal (negative) group (see :func:`_hotloop_well_formed`).
    WMMA never reaches here (gated to the ``mem`` pipeline upstream).
    """
    from ...helpers.schedule import SchedulePolicy

    t = spec.tile
    pipeline = spec.trait.pipeline
    il = _hotloop_inst_list(spec, load_vec)
    policy = SchedulePolicy.for_pipeline(pipeline)
    if _hotloop_well_formed(il, pipeline):
        if pipeline == "compv3":
            policy.emit_compv3_hotloop(b, il)
        else:
            policy.emit_compv4_hotloop(b, il)
        return
    # Degenerate tile: keep the prior flat hint (numerically identical;
    # both are pure scheduling hints).
    b.sched_group_barrier(0x100, 1, 0)  # one DS read
    b.sched_group_barrier(0x008, t.mfmas_per_warp_m * t.mfmas_per_warp_n, 0)


# ---------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------


def build_universal_gemm(spec: UniversalGemmSpec, arch: str = "gfx950") -> KernelDef:
    """Build the IR for one universal GEMM instance on ``arch``.

    The body is **MMA-contract driven**: it resolves the warp-tile atom from
    the target's catalog (an MFMA op_id on gfx942/gfx950, a WMMA op_id on
    gfx1151), then drives the per-lane fragment loads, the matmul, and the
    accumulator scatter through that op's layout maps / fragment lengths and the
    target-neutral :meth:`IRBuilder.mma`. The same source thus emits MFMA on
    CDNA (byte-identical to the prior ISA-named emission) and WMMA on RDNA.

    On CDNA this dispatches on (pipeline, epilogue):

      | pipeline | epilogue | implementation |
      |---|---|---|
      | mem      | default  | single-buffer + direct vector stores (the dsl/01-05 family) |
      | compv3   | default  | single-buffer + sched_group_barrier interleave |
      | compv3   | cshuffle | single-buffer + LDS-staged f16 epilogue |
      | compv4   | default  | double-buffer LDS + sched_group_barrier + direct vector stores |
      | compv4   | cshuffle | double-buffer LDS + sched_group_barrier + LDS-staged cshuffle |
      | mem      | cshuffle | single-buffer + LDS-staged epilogue |

    All five CDNA paths share the same IR-construction subroutines below. The
    RDNA/WMMA target (gfx1151) is gated to the ``mem`` pipeline + ``default``
    epilogue + 16x16x16 atom subset (see :func:`is_valid_spec`); the richer
    pipelines/epilogues remain CDNA-only until ported.
    """

    # Warp-specialized producer/consumer 3-stage pipeline.
    # Lives in a SEPARATE emitter (gemm_wsp3.py) and produces new kernel
    # names, so the byte-identical golden gate over the existing
    # mem/compv3/compv4 paths is untouched (no existing emission edited).
    if spec.trait.pipeline == "wsp3":
        from .gemm_wsp3 import build_wsp3_gemm

        return build_wsp3_gemm(spec, arch=arch)

    ok, why = is_valid_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid GEMM spec: {why}")
    if spec.trait.persistent:
        warnings.warn(
            "TraitSpec.persistent=True is a no-op for the universal GEMM body "
            "(no persistent tile loop; only the kernel name changes). Use "
            "streamk_gemm for a persistent body.",
            stacklevel=2,
        )

    op = _resolve_mma_op(spec, arch)
    if op is None:  # pragma: no cover - is_valid_spec already rejects this
        raise ValueError(f"no MMA atom for spec on {arch}")

    # fused operand prologue (Python-only side channel ``spec._prologue``, see
    # gemm_prologue)
    from .gemm_prologue import PrologueEmitter as _ProEm, get_prologue as _get_pro

    _pro = _get_pro(spec)
    if _pro is not None:
        if op.family == "wmma":
            raise ValueError("prologue: MFMA bodies only")
        if _pro.out_f32 and spec.trait.epilogue != "default":
            raise ValueError("prologue out_f32 needs the default epilogue")
        if spec.trait.split_k > 1:
            raise ValueError("prologue: split_k unsupported")
        if spec.trait.sia_sched or spec.trait.asm_loop:
            # these K-tile emitters were never verified with a prologue
            raise ValueError("prologue: sia_sched / asm_loop bodies unsupported")
    b = IRBuilder(spec.kernel_name() + (_pro.suffix() if _pro is not None else ""))
    # IMPORTANT: AMDGPU bakes `amdgpu-flat-work-group-size` into the
    # kernel descriptor; if we launch with more threads than that, HIP
    # returns `unspecified launch failure` *before* the kernel body
    # runs. Pin the upper bound to this spec's `block_size`.
    b.kernel.attrs["max_workgroup_size"] = spec.block_size
    if spec.trait.waves_per_eu is not None:
        b.kernel.attrs["waves_per_eu"] = spec.trait.waves_per_eu
    # EXPERIMENT (gated, Python engine only): ROCKE_EXP_AGPR_ALLOC=min,max sets
    # ``amdgpu-agpr-alloc`` on a kernel without ``acc_agpr``. Without an
    # accumulator layout that wants AGPRs (zero constant seed, no AGPR MFMA
    # form) gfx950 keeps the accumulators in arch-VGPRs, so for such kernels it
    # changes nothing there. ``acc_agpr`` (below) is the supported route: it sets
    # the attribute, the opaque AGPR seed and the AGPR MFMA form together, and
    # the env var is rejected for it rather than silently replacing the
    # reservation under a kernel name that still says ``_agpr``.
    # Default off -> byte-identical.
    import os as _os_agpr

    _agpr_alloc = _os_agpr.environ.get("ROCKE_EXP_AGPR_ALLOC")
    if _agpr_alloc and spec.trait.acc_agpr:
        raise ValueError(
            "ROCKE_EXP_AGPR_ALLOC conflicts with TraitSpec.acc_agpr (which sets "
            "amdgpu-agpr-alloc itself); unset the env var"
        )
    if _agpr_alloc:
        b.kernel.attrs["agpr_alloc"] = _agpr_alloc
    elif spec.trait.acc_agpr:
        # reserve exactly the accumulator registers as AGPRs
        # (the attr's minimum also selects the AGPR form of every MFMA).
        _n_acc = acc_agpr_regs(spec)
        b.kernel.attrs["agpr_alloc"] = (_n_acc, _n_acc)
        # compile option ``-amdgpu-mfma-vgpr-form=0`` (helpers.compile): LLVM 23
        # backends otherwise pick the VGPR MFMA form despite the reservation
        b.kernel.attrs["mfma_agpr_form"] = True
    storage_dtype = _storage_dtype(spec)
    _split_k = spec.trait.split_k
    _is_split_k = _split_k > 1
    # ``_mono`` (Python-only hooks: kernels/gfx950/gemm_rich_epilogue and the
    # persistent contraction monokernel, kernels/gfx950/hiptensor_mono): a side
    # channel consumed below; absent -> byte-identical emission. A hook whose
    # producer tasks write the A / B workspaces in the same launch sets
    # ``writes_operands``, which drops readonly / noalias on A and B.
    _mono = getattr(spec, "_mono", None)
    _ro = _mono is None or not getattr(_mono, "writes_operands", False)
    A = b.param(
        "A", PtrType(storage_dtype, "global"), noalias=_ro, readonly=_ro, align=16
    )
    Bp = b.param(
        "B", PtrType(storage_dtype, "global"), noalias=_ro, readonly=_ro, align=16
    )
    # In split-K mode the output is the f32 accumulation workspace the
    # epilogue atomic-adds into (the caller casts it to the target dtype
    # afterwards). It is read+write (atomicrmw), so drop the writeonly
    # attribute. Outside split-K the C param is byte-identical to before.
    if _is_split_k:
        from ...core.ir import F32 as _F32

        C = b.param("Cf32", PtrType(_F32, "global"), noalias=True, align=4)
    else:
        C = b.param(
            "C",
            PtrType(
                F32 if (_pro is not None and _pro.out_f32) else storage_dtype, "global"
            ),
            noalias=True,
            writeonly=True,
            align=16,
        )
    M = b.param("M", I32)
    N = b.param("N", I32)
    K = b.param("K", I32)
    if _mono is not None:
        _mono.add_params(b, A=A, B=Bp, C=C, M=M, N=N, K=K)
    # ``_gett`` (native GETT, ``rocke.instances.common.tensor.gett``; Python
    # only): a side channel like ``_fused_epilogue``. Absent -> nothing below
    # reads it and every kernel stays byte-identical. Present: both operands
    # are MN-contiguous tensors read in place through per-lane mode offsets
    # (DTL), the LDS stages are K-outer in the hook's padded block layout, the
    # MFMA fragments come from ds_read_b64_tr_b16, and the hook owns the
    # epilogue (alpha / beta, permuted output offsets).
    _gett = getattr(spec, "_gett", None)
    if _gett is not None:
        _gett.add_params(b)
    if spec.trait.streamk:
        # stream-K fixup workspace (streamk_workspace_bytes): ready flags and
        # one f32 partial tile per workgroup of the pool
        sk_flags = b.param("SkFlags", PtrType(I32, "global"), align=4)
        sk_part = b.param("SkPart", PtrType(F32, "global"), noalias=True, align=16)
    if spec.batched:
        # Per-batch strides (in elements, not bytes). The kernel uses
        # ``block_id_z`` as the batch index and adds
        # ``z * stride_X`` to the X-load/store base offset for X in
        # {A, B, C}. Strides are i32 to match the rest of the index
        # arithmetic; the LLVM zext/sext folds them into the gep idx.
        stride_a = b.param("stride_a", I32)
        stride_b = b.param("stride_b", I32)
        stride_c = b.param("stride_c", I32)
        # two-mode batch (Python side channel: ``spec._batch2 = True`` on a batched
        # spec, set with object.__setattr__): grid z = L1 * L2, z -> (z / L2, z % L2),
        # offset = z1 * stride_X + z2 * stride_X2. Two non-mergeable Hadamard (batch)
        # modes in one launch, no permute pass.
        _batch2 = bool(getattr(spec, "_batch2", False))
        if _batch2:
            stride_a2 = b.param("stride_a2", I32)
            stride_b2 = b.param("stride_b2", I32)
            stride_c2 = b.param("stride_c2", I32)
            batch_l2 = b.param("L2", I32)
    if spec.batched and spec.trait.active_tile_skip:
        # MoE active-tile gate. ``SortedTokenIds`` carries the
        # bucket -> token-id map; ``-1`` marks an inactive padded
        # row. ``slot_size`` is the per-expert padded row count.
        sorted_token_ids = b.param(
            "SortedTokenIds",
            PtrType(I32, "global"),
            noalias=True,
            readonly=True,
            align=4,
        )
        slot_size_p = b.param("slot_size", I32)

    t = spec.tile
    a_per_lane, b_per_lane, c_per_lane = _atom_frag_lengths(op)

    block_m = t.tile_m
    block_n = t.tile_n
    block_k = t.tile_k

    # Common geometry.
    c0 = b.const_i32(0)
    # Split-K K-slice bounds. Each CTA along ``block_id_z`` owns the slice
    # ``[z * ks, (z+1) * ks)`` where ``ks = K // split_k``; the K-loop runs
    # over that slice instead of ``[0, K)``. ``block_id_z`` is CTA-uniform so
    # the slice base is SGPR-pinned. For ``split_k == 1`` these collapse to
    # ``k_lo = 0`` / ``k_hi = K`` (the bound expressions below substitute them
    # verbatim, so the non-split body stays byte-identical).
    if _is_split_k:
        ks = b.div(K, b.const_i32(_split_k))
        k_lo = b.to_sgpr_u32(b.mul(b.block_id_z(), ks))
        k_hi = b.to_sgpr_u32(b.add(k_lo, ks))
    else:
        k_lo = c0
        k_hi = None  # use K directly so the bound SSA is unchanged
    c_wave = b.const_i32(spec.wave_size)
    c_warps_n = b.const_i32(t.warp_n)
    c_block_m = b.const_i32(block_m)
    c_block_n = b.const_i32(block_n)
    c_block_k = b.const_i32(block_k)

    tid = b.thread_id_x()
    warp_id = b.div(tid, c_wave)
    warp_m_idx = b.div(warp_id, c_warps_n)
    warp_n_idx = b.mod(warp_id, c_warps_n)
    lane = b.mod(tid, c_wave)

    # LDS XOR swizzle (st_16x32-style, zero LDS overhead). For 2-byte
    # elements the 32-byte (16-half) column group is toggled on rows whose
    # bit 3 is set: ``col ^= ((row>>3)&1)<<4``. Applied identically to the
    # LDS store column and the MFMA ds_read column, so the physical address
    # for any logical (row,col) agrees on both sides (correctness preserved)
    # while consecutive rows hit different banks. Opt-in via CK_SWIZZLE; the
    # 16-half toggle preserves the 8-half load-vector alignment.
    _SWZ = spec.trait.lds_swizzle
    # compv4 schedule-hint policy: explicit trait override wins; the default
    # (None) takes the measured gfx950 uplift (hints OFF) and preserves the
    # historical emission on every other arch. Gates both the per-cluster
    # s_setprio/sched_barrier fences and the sched_group_barrier HotLoop.
    _sched_hints = (
        spec.trait.emit_sched_hints
        if spec.trait.emit_sched_hints is not None
        else (arch != "gfx950")
    )
    # interwave_setprio: per-K-atom MFMA cluster priority bookends (see
    # ``_mma_cluster``). Off (default) keeps the historical emission for both
    # scheduler values.
    _interwave = spec.trait.interwave_setprio
    # EXPERIMENT (ROCKE_EXP_HOIST_SWZ=1): feed the ds_read swizzle mask a
    # PARITY-FREE row so it is loop-invariant and LLVM/LICM hoists the
    # lshr/mod/shl out of the K-loop (only the per-read xor with col_base
    # remains). Measured bit-exact +~2% (750->765 TFLOPS) on the grouped bf16
    # 256x256x64 DTL kernel. Default OFF keeps emission byte-identical; correct
    # iff the double-buffer parity offset adds an even multiple to the swizzle
    # bit (block_m>>R even) -- verify before promoting to a TraitSpec lever.
    import os as _os_hsw

    _hoist_swz = _SWZ and _os_hsw.environ.get("ROCKE_EXP_HOIST_SWZ", "0") == "1"
    # EXPERIMENT (ROCKE_EXP_UNROLL_K=<K>): fully Python-unroll the DTL-prefetch
    # K-loop for a fixed, compile-time K (= the env value, a multiple of
    # tile_k), eliminating the scf.for backedge and giving static double-buffer
    # parity offsets. Only valid for split_k==1 and when the launch K matches.
    # 0 = off (runtime scf.for, default).
    _unroll_k = int(_os_hsw.environ.get("ROCKE_EXP_UNROLL_K", "0"))
    # EXPERIMENT (ROCKE_EXP_PREFETCH_DEPTH=D>1): D+1-buffer LDS ring in the
    # fully-unrolled fixed-K path (requires ROCKE_EXP_UNROLL_K). Issues loads D
    # tiles ahead. NOTE: buffer_load_lds (DTL) completes OUT OF ORDER, so the
    # per-tile drain must stay vmcnt(0) for correctness -> deeper buffering keeps
    # no extra loads usefully in flight (verified in gemm_wsp3). Kept as a gated
    # lever for measurement / the non-DTL ordered path. Costs D+1x AB LDS.
    _pf_depth = int(_os_hsw.environ.get("ROCKE_EXP_PREFETCH_DEPTH", "1"))
    # LDS XOR swizzle: ``col ^= ((row >> R) % 2^W) << L``. XOR of any
    # deterministic function of ``row`` into ``col`` is correctness-preserving
    # as long as it is applied identically to the LDS store-source + ds_read
    # columns (it is) and stays in [0, block_k).
    #
    # The optimal (measured 0.0% LDSBankConflict) parameters are derived from
    # the geometry, not guessed:
    #   * L = log2(elem)  -- swizzle whole ``elem``-half (b128) vector slots so
    #                        every target is naturally 16B-aligned.
    #   * W = log2(block_k/elem) -- use ALL element slots in the row.
    #   * R = 0           -- key the swizzle on the LOW row bits, i.e. the
    #                        ``m_in_atom`` (0..warp_tile_m-1) that actually
    #                        aliases (every M-row of an atom reads the same col,
    #                        and the row*stride term vanishes mod 32 banks, so
    #                        only a low-bit element permutation decorrelates them).
    # The earlier default (R=3,W=1,L=4) keyed on HIGH row bits / 2 slots and so
    # floored at 25% (4-way). Env vars override for sweeps; geometry fallback to
    # the 2-slot toggle if elem/slots aren't a clean power-of-2 split.
    # (``lds_swizzle_params`` is the env-free single source shared with the
    # validity gate; the CK_SWZ_* env vars are debug overrides that bypass the
    # gate's write-granularity rule.)
    import os as _os

    _def_r, _def_w, _def_l = lds_swizzle_params(a_per_lane, b_per_lane, block_k)
    if spec.trait.lds_swizzle_r >= 0:
        _def_r = spec.trait.lds_swizzle_r
    _swz_r = int(_os.environ.get("CK_SWZ_R", str(_def_r)))
    _swz_w = int(_os.environ.get("CK_SWZ_W", str(_def_w)))
    _swz_l = int(_os.environ.get("CK_SWZ_L", str(_def_l)))
    _c_swr, _c_swmod, _c_swl = (
        b.const_i32(_swz_r),
        b.const_i32(1 << _swz_w),
        b.const_i32(_swz_l),
    )

    def _swz_col(col, row):
        if not _SWZ:
            return col
        return b.xor(col, b.shl(b.mod(b.lshr(row, _c_swr), _c_swmod), _c_swl))

    if spec.batched:
        # P57: every wave-uniform i32 derived from the batch axis is
        # pinned into an SGPR via ``to_sgpr_u32`` so the per-tile
        # address arithmetic stays in scalar registers rather than
        # being re-materialised in VGPRs at every consumer (one
        # ``v_readfirstlane_b32`` saved per use across the K-loop +
        # epilogue). Mirrors CK Tile ``batched_gemm_kernel.hpp:215-230``
        # ``amd_wave_read_first_lane`` pattern.
        batch_idx = b.to_sgpr_u32(b.block_id_z())
        if _batch2:
            _z1 = b.to_sgpr_u32(b.div(batch_idx, batch_l2))
            _z2 = b.to_sgpr_u32(b.sub(batch_idx, b.mul(_z1, batch_l2)))
            batch_off_a = b.to_sgpr_u32(
                b.add(b.mul(_z1, stride_a), b.mul(_z2, stride_a2))
            )
            batch_off_b = b.to_sgpr_u32(
                b.add(b.mul(_z1, stride_b), b.mul(_z2, stride_b2))
            )
            batch_off_c = b.to_sgpr_u32(
                b.add(b.mul(_z1, stride_c), b.mul(_z2, stride_c2))
            )
        else:
            batch_off_a = b.to_sgpr_u32(b.mul(batch_idx, stride_a))
            batch_off_b = b.to_sgpr_u32(b.mul(batch_idx, stride_b))
            batch_off_c = b.to_sgpr_u32(b.mul(batch_idx, stride_c))
    else:
        batch_off_a = c0
        batch_off_b = c0
        batch_off_c = c0

    # Tile-index assignment. By default ``block_id_x`` maps to the
    # N-tile and ``block_id_y`` to the M-tile (the host launcher uses
    # ``grid = (N_tiles, M_tiles, batch?)``). With
    # ``trait.chiplet_swizzle=True`` we instead flatten the 2D grid
    # into a linear WGID and run it through the chiplet-aware
    # super-tile remap so consecutive workgroups land on the same XCD.
    if spec.trait.chiplet_swizzle:
        from ...helpers.grid import chiplet_aware_super_tile_dynamic

        # Compute M_tiles / N_tiles at runtime from the dynamic M/N args.
        n_pid_m = b.div(b.add(M, b.const_i32(block_m - 1)), c_block_m)
        n_pid_n = b.div(b.add(N, b.const_i32(block_n - 1)), c_block_n)
        # Flatten (bx, by) -> wgid_flat using the actual launch grid's
        # X-extent (= n_pid_n) so wgid_flat mirrors a 1D dispatch
        # walking ``for by: for bx:`` order.
        wgid_flat = b.add(
            b.mul(b.block_id_y(), n_pid_n),
            b.block_id_x(),
        )
        swz = chiplet_aware_super_tile_dynamic(
            b,
            wgid_flat,
            num_pid_m=n_pid_m,
            num_pid_n=n_pid_n,
            wgm=spec.trait.chiplet_wgm,
            num_xcds=spec.trait.chiplet_num_xcds,
            chunk_size=spec.trait.chiplet_chunk_size,
        )
        # P57: pin chiplet-swizzled offsets in SGPR. ``swz.row`` / ``swz.col``
        # are wave-uniform (function of CTA-uniform inputs), so
        # ``readfirstlane`` is safe.
        block_m_off = b.to_sgpr_u32(b.mul(swz.row, c_block_m))
        block_n_off = b.to_sgpr_u32(b.mul(swz.col, c_block_n))
    else:
        # P57: pin the non-swizzled batched-GEMM offsets in SGPR too.
        block_m_off = b.to_sgpr_u32(b.mul(b.block_id_y(), c_block_m))
        block_n_off = b.to_sgpr_u32(b.mul(b.block_id_x(), c_block_n))

    # stagger_u: this workgroup's K start offset (elements)
    _su_off = None
    if spec.trait.stagger_u:
        _su_map = int(spec.trait.stagger_u_map)
        if _su_map == 0:
            _su_idx = swz.row if spec.trait.chiplet_swizzle else b.block_id_y()
        elif _su_map == 1:
            _su_idx = swz.col if spec.trait.chiplet_swizzle else b.block_id_x()
        else:
            # the flat (pre-swizzle) workgroup id (validity: chiplet_swizzle)
            _su_idx = wgid_flat
        _su_kt = b.div(K, b.const_i32(t.tile_k))
        _su_t = b.mul(
            b.mod(_su_idx, b.const_i32(int(spec.trait.stagger_u))),
            b.const_i32(int(spec.trait.stagger_u_stride)),
        )
        # (idx mod SU) * stride mod T; 0 when K is not a multiple of tile_k
        _su_t = b.mod(_su_t, b.smax(_su_kt, b.const_i32(1)))
        _su_ok = b.cmp_eq(b.mul(_su_kt, b.const_i32(t.tile_k)), K)
        _su_off = b.to_sgpr_u32(
            b.select(_su_ok, b.mul(_su_t, b.const_i32(t.tile_k)), c0)
        )

    # LDS allocation. compv4 (non-DTL, direct epilogue) double-buffers AB
    # so the next K-tile's global->LDS copy overlaps the current tile's
    # MFMA/ds_read work (real software pipelining; see ``_emit_kloop_db``).
    # Both halves live in one ``[2*block_m, block_k]`` / ``[2*block_n,
    # block_k]`` smem region and the K-loop alternates the parity row
    # origin. The cshuffle epilogue allocates a separate tile_m*tile_n C
    # staging buffer; with the default (``ab_double_buffer=None``) it stays
    # single-buffered. NOTE: the C tile is aliased onto the A/B bytes by the
    # smem-pool packer (pool = max(ab, c)), so ``ab_double_buffer=True`` makes
    # the cshuffle variant double-buffered at no extra LDS for 128x128x32.
    #
    # When ``dtl_prefetch`` is on, we likewise double the M/N dimension of
    # each LDS buffer; the ``parity`` (0 or 1) selects which half is the
    # current write target / read source, and the K-loop alternates so
    # next-iter DTLA writes go to the buffer the MFMAs aren't reading.
    _prefetch = bool(spec.trait.dtl_prefetch)
    if _prefetch and not spec.trait.direct_to_lds:
        raise ValueError("dtl_prefetch requires direct_to_lds=True")
    # compv4 (non-DTL) double-buffers the AB LDS so the next K-tile's
    # global->LDS copy overlaps the current K-tile's MFMA/ds_read work
    # (a real software-pipelined prefetch). The validity gate already
    # reserves 2x AB LDS for compv4, so this consumes budget that was
    # previously reserved-but-unused. DTL keeps its own ping-pong path.
    # Double-buffer only when the epilogue does NOT also stage C through
    # LDS: the cshuffle epilogue allocates a separate tile_m*tile_n C
    # buffer, so doubling AB on top of it overflows the LDS budget and
    # crushes occupancy (a net regression). The direct ("default")
    # epilogue stores straight to global, leaving the whole LDS budget
    # for the AB ping-pong.
    # Occupancy guard: doubling AB only pays off if the doubled buffer
    # still leaves room for the same number of workgroups per CU. On
    # gfx950 the LDS-limited occupancy step is at half the per-CU LDS
    # (2 WG/CU needs <= cap/2 bytes each). If the doubled AB would push a
    # config from 2 WG/CU down to 1, the lost occupancy outweighs the
    # prefetch overlap (measured: 256x128x64 regresses -3% when it drops
    # to 1 WG/CU). In that case we keep the single-buffer pipeline.
    # AB LDS double-buffer plan, shared with the validity gate via
    # :func:`_ab_lds_plan` so the reserved/used budget stays in lock-step.
    _, _db, _two_buf = _ab_lds_plan(spec, arch)
    # depth-N prefetch ring needs (depth+1) AB buffers (only in the unrolled
    # fixed-K path); otherwise the usual 2 (ping-pong) or 1 (single-buffer).
    if _pf_depth > 1 and _unroll_k > 0:
        _nbuf = _pf_depth + 1
    else:
        _nbuf = 2 if _two_buf else 1
    _A_LDS_M = _nbuf * block_m
    _B_LDS_N = _nbuf * block_n
    # LDS K-padding (non-DTL only): widen each row's stride to break the
    # bank-conflict alias. The logical column range stays [0, block_k); only
    # the row stride grows, so the read GEP (alloc shape[1]) and the
    # store_vec TensorView (with_strides below) both pick up the padded stride.
    _lds_pad = spec.trait.lds_k_pad if not spec.trait.direct_to_lds else 0
    _lds_k = block_k + _lds_pad
    # dtl_stages >= 2: one A and one B allocation per stage +
    # LDS alias scopes, so every ds_read / buffer_load...lds is attributable to
    # one stage and SIInsertWaitcnts only waits for DMA into the stage it reads.
    _stages = (
        int(spec.trait.dtl_stages) if (_prefetch and spec.trait.direct_to_lds) else 0
    )
    _rilv = bool(spec.trait.lds_row_interleave)
    if _stages >= 2 and _gett is not None:
        # GETT: K-outer stages in the hook's padded block layout
        A_stg = [
            b.smem_alloc(storage_dtype, _gett.lds_shape("A"), name_hint=f"A_smem_s{j}")
            for j in range(_stages)
        ]
        B_stg = [
            b.smem_alloc(storage_dtype, _gett.lds_shape("B"), name_hint=f"B_smem_s{j}")
            for j in range(_stages)
        ]
        A_smem, B_smem = A_stg[0], B_stg[0]
        b.kernel.attrs["lds_alias_scopes"] = True
    elif _stages >= 2 and _rilv:
        # lds_row_interleave: rows of 8 x block_k elements + rilv_pad
        # bytes of pad (one DTL instruction per wave fills exactly one such row)
        _rilv_cols = 8 * block_k + int(spec.trait.rilv_pad) // (
            type_bits(storage_dtype) // 8
        )
        A_stg = [
            b.smem_alloc(
                storage_dtype, [block_m // 8, _rilv_cols], name_hint=f"A_smem_s{j}"
            )
            for j in range(_stages)
        ]
        B_stg = [
            b.smem_alloc(
                storage_dtype, [block_n // 8, _rilv_cols], name_hint=f"B_smem_s{j}"
            )
            for j in range(_stages)
        ]
        A_smem, B_smem = A_stg[0], B_stg[0]
        b.kernel.attrs["lds_alias_scopes"] = True
    elif _stages >= 2:
        A_stg = [
            b.smem_alloc(storage_dtype, [block_m, block_k], name_hint=f"A_smem_s{j}")
            for j in range(_stages)
        ]
        # pingpong 6: the B operand gets a 3-stage ring (A keeps 2)
        _stages_b = 3 if int(spec.trait.pingpong) in (6, 7) else _stages
        B_stg = [
            b.smem_alloc(storage_dtype, [block_n, block_k], name_hint=f"B_smem_s{j}")
            for j in range(_stages_b)
        ]
        A_smem, B_smem = A_stg[0], B_stg[0]
        b.kernel.attrs["lds_alias_scopes"] = True
    else:
        A_smem = b.smem_alloc(storage_dtype, [_A_LDS_M, _lds_k], name_hint="A_smem")
        B_smem = b.smem_alloc(storage_dtype, [_B_LDS_N, _lds_k], name_hint="B_smem")
    # When ``_two_buf`` (compv4 DB or dtl_prefetch) the two halves share
    # this one ``2*block_m``/``2*block_n``-tall region; the K-loop's
    # ``lds_parity`` shifts the load-store and MFMA-read row origin by
    # ``parity * block_m`` / ``parity * block_n`` between them. Single-
    # buffer pipelines (``mem``, ``compv3``, cshuffle-compv4) use the
    # bottom half only.

    # Per-warp MFMA tile (mfmas_per_warp_m * mfmas_per_warp_n MFMAs per K-step).
    mfmas_m = t.mfmas_per_warp_m
    mfmas_n = t.mfmas_per_warp_n
    # MFMA issue order inside a K atom (``mfma_serpentine`` = order 1, else
    # ``mfma_order``): MFMA slot -> accumulator index, per K-atom parity; the
    # identity for the default order 0
    _mo_perm = mfma_order_perm(
        1 if spec.trait.mfma_serpentine else int(spec.trait.mfma_order),
        mfmas_m,
        mfmas_n,
    )
    _mo_ilv = _mo_perm[0]
    k_atoms = t.k_atoms_per_tile_k
    _kmul = 1
    _pe = None
    if _pro is not None:
        _kmul = _pro.kmul
        if k_atoms % _kmul:
            raise ValueError(
                f"prologue src=f32: tile_k ({block_k} halves) must hold whole f32 atoms"
            )
        k_atoms //= _kmul
        if a_per_lane != b_per_lane:
            raise ValueError("prologue: a_per_lane != b_per_lane")
        _pe = _ProEm(b, _pro, storage_dtype, a_per_lane)
    # complex B (real embedding, see Prologue.cplx_b): tile row r of B' is complex row r >> 1
    _cplx = _pe is not None and bool(_pro.cplx_b)
    if _cplx:
        if not spec.trait.direct_to_lds:
            raise ValueError("cplx_b needs the direct-to-LDS ring")
        if int(spec.trait.pingpong) == 7:
            raise ValueError("cplx_b does not support pingpong 7")
        if _rilv and not spec.trait.rilv_vec_epilogue:
            raise ValueError(
                "cplx_b + lds_row_interleave needs rilv_vec_epilogue (linear rows)"
            )
        if _rilv and (t.mfmas_per_warp_n % 2):
            raise ValueError(
                "cplx_b + lds_row_interleave needs an even mfmas_per_warp_n"
            )
        if _rilv:
            _pe.cplx_mode = "ni"
        else:
            _pe.cplx_mode = "lane"
            _pe.cplx_lane_odd = b.cmp_ne(b.land(lane, b.const_i32(1)), b.const_i32(0))

    def _brow(r: Value) -> Value:
        """Global B row of B' row ``r`` (complex B: r >> 1)."""
        return b.div(r, b.const_i32(2)) if _cplx else r

    def _frag_ld(src, row, col_base, key, n, which, swz=True):
        """One operand fragment (``_emit_smem_load`` + the prologue hook)."""
        if _pe is None:
            col = _swz_col(col_base, key) if swz else col_base
            return _emit_smem_load(b, src, row, col, n, storage_dtype)
        n_raw = _pe.raw_len()
        raw = None
        for off in range(0, n_raw, 8):
            col = col_base if off == 0 else b.add(col_base, b.const_i32(off))
            if swz:
                col = _swz_col(col, key)
            ch = _emit_smem_load(b, src, row, col, min(8, n_raw), storage_dtype)
            raw = ch if raw is None else b.vec_concat(raw, ch)
        return _pe.after_load(raw, which)

    # Accumulators: one `<c_per_lane x float>` per warp-local MFMA tile.
    acc_init = _emit_zero_acc_op(b, op)
    if spec.trait.acc_agpr:
        # opaque AGPR zero seed (see TraitSpec.acc_agpr): keeps the
        # loop-carried accumulator phis whole vectors
        acc_init = b.inline_asm("", "=a,0", [acc_init], acc_init.type, sideeffect=False)
    accs = [
        (f"acc_m{mi}_n{ni}", acc_init) for mi in range(mfmas_m) for ni in range(mfmas_n)
    ]

    # Global -> LDS coalesced copy plan.
    threads = spec.block_size
    load_vec = _choose_load_vec(spec)
    a_total = block_m * block_k
    b_total = block_n * block_k
    a_vec_total = a_total // load_vec
    b_vec_total = b_total // load_vec
    a_vecs_per_thread = a_vec_total // threads
    b_vecs_per_thread = b_vec_total // threads
    c_threads = b.const_i32(threads)
    c_load_vec = b.const_i32(load_vec)
    c_block_k_div_vec = b.const_i32(block_k // load_vec)

    # CK Tile-style data views. The A and B global tensors are
    # modelled as 3D views ``(batch, M_or_N, K)`` with element strides
    # ``(1, K, 1)``. The batch dim's stride of 1 lets us pre-compute
    # ``batch_off_a`` (in elements) once per CTA and pass it as the
    # batch-axis origin; the descriptor's offset formula then yields
    #
    #   offset = batch_off_a + (block_m_off + local_row) * K
    #          + (k_off + local_col)
    #
    # which matches the prior hand-rolled IR byte-for-byte after
    # constant folding. ``batch_off_a == 0`` in non-batched mode, so
    # the lowered IR collapses to the unchanged 2D form.
    a_view = make_global_view(
        A, shape=(1, 1, 1), dtype=storage_dtype, strides=(1, K, 1)
    )
    b_view = make_global_view(
        Bp, shape=(1, 1, 1), dtype=storage_dtype, strides=(1, K, 1)
    )
    # LDS views are 2D packed (block_m, block_k) / (block_n, block_k).
    from ...helpers.tensor_view import TensorDescriptor, TensorView

    # When LDS K-padding is on, the logical shape stays (block_*, block_k) but
    # the row stride is the padded ``_lds_k`` so the store_vec addressing
    # agrees with the read GEP (which uses the padded alloc shape).
    if _lds_pad:
        _a_lds_desc = TensorDescriptor.with_strides(
            (block_m, block_k), (_lds_k, 1), storage_dtype
        )
        _b_lds_desc = TensorDescriptor.with_strides(
            (block_n, block_k), (_lds_k, 1), storage_dtype
        )
    else:
        _a_lds_desc = TensorDescriptor.packed((block_m, block_k), storage_dtype)
        _b_lds_desc = TensorDescriptor.packed((block_n, block_k), storage_dtype)
    a_lds_view = TensorView(base=A_smem, desc=_a_lds_desc, addr_space="lds")
    b_lds_view = TensorView(base=B_smem, desc=_b_lds_desc, addr_space="lds")

    # DirectToLDS (DTLA/DTLB) plumbing. We issue
    # ``async_buffer_load_lds_addr`` directly because
    # :class:`AsyncTileLoader` assumes a tile shape where the row dimension
    # wraps at ``halves_per_chunk`` — that's an attention-specific layout
    # and does not match a [block_m, block_k] GEMM tile with block_k >>
    # halves_per_chunk.
    #
    # The intrinsic writes ``dwords * 4`` bytes per lane lane-contiguous
    # starting at the wave-uniform ``lds_dst``. For our tile we lay
    # ``block_size`` lanes across the tile such that each lane covers one
    # chunk of ``dwords * 2`` halves (bf16 elements). Passes cover the
    # remaining chunks_total / block_size iterations.
    if spec.trait.direct_to_lds:
        from ...core.arch import ArchTarget
        from ...core.ir import I64 as _I64

        # Capped at the arch's own buffer_load_lds width (is_valid_spec has
        # already checked tile_k divides it): CDNA4 (gfx950) has the full
        # dword=4 (16 bytes/lane) form, CDNA3 (gfx942) only dword=1.
        _DTL_DWORDS = min(4, ArchTarget.from_gfx(arch).async_lds_max_dwords)
        _DTL_HALVES = _DTL_DWORDS * 2  # dwords*2 halves (bf16 elements) per lane chunk
        _DTL_BYTES_PER_LANE = _DTL_DWORDS * 4
        if (block_k % _DTL_HALVES) != 0:
            raise ValueError(
                f"direct_to_lds requires block_k % {_DTL_HALVES} == 0 (got {block_k})"
            )
        _dtl_a_chunks = (block_m * block_k) // _DTL_HALVES
        _dtl_b_chunks = (block_n * block_k) // _DTL_HALVES
        _dtl_a_passes = (_dtl_a_chunks + spec.block_size - 1) // spec.block_size
        _dtl_b_passes = (_dtl_b_chunks + spec.block_size - 1) // spec.block_size
        # block_size lanes write block_size * BYTES_PER_LANE bytes per pass
        # to LDS. Single workgroup writes share the LDS base.
        _dtl_pass_bytes = spec.block_size * _DTL_BYTES_PER_LANE

        _dtl_big_bytes = b.const_i32(DTL_BUFFER_NUM_RECORDS)
        _dtl_a_rsrc = b.buffer_rsrc(A, _dtl_big_bytes)
        _dtl_b_rsrc = b.buffer_rsrc(Bp, _dtl_big_bytes)
        _dtl_a_lds_base = b.smem_addr_of(A_smem)
        _dtl_b_lds_base = b.smem_addr_of(B_smem)
        _dtl_zero_soff = b.const_i32(0)
        # chunks-per-row in halves: block_k halves wide / chunk width
        _dtl_chunks_per_row = block_k // _DTL_HALVES
        _dtl_c_chunks_per_row = b.const_i32(_dtl_chunks_per_row)
        _dtl_c_halves_per_chunk = b.const_i32(_DTL_HALVES)
        _dtl_c_block_size = b.const_i32(spec.block_size)

    def emit_load_phase(A_dst: Value, B_dst: Value, k_off: Value, lds_parity=0) -> None:
        """Coalesced global -> LDS copy for one K tile.

        Driven by :class:`TileWindow`: the A/B global views carry the
        ``(1, K, 1)`` descriptor; the per-call ``k_off`` shifts each
        tile's column origin. ``batch_off_a`` / ``batch_off_b`` ride
        in the batch-axis origin and the descriptor's `mul-by-1`
        folds away in LLVM, yielding identical lowered IR to the
        pre-helpers version.

        ``lds_parity`` (DTLA + prefetch only): 0 writes the first LDS
        half-buffer; 1 writes the second. May be a Python int (0/1) or
        an i32 :class:`Value` for runtime parity. Ignored when prefetch
        is off.
        """
        if spec.trait.direct_to_lds:
            # For each pass, every lane copies one chunk of HALVES halves
            # from global -> LDS via the hardware DTLA/DTLB intrinsic.
            #   chunk_idx_global = tid + pass * block_size
            #   tile_row         = chunk_idx_global / chunks_per_row
            #   tile_col_halves  = (chunk_idx_global % chunks_per_row) * HALVES
            # The LDS destination address advances by pass_bytes per pass.
            #
            # Parity offset (prefetch ping-pong): adds the size of one
            # half-buffer in bytes to the LDS base when writing the second
            # buffer; size = block_m*block_k*2 for A, block_n*block_k*2 for B.
            # ``lds_parity`` may be a Python int (compile-time) or an i32
            # Value (runtime, carried as a scf.for iter-arg).
            _parity_is_value = isinstance(lds_parity, Value)
            a_half_bytes = block_m * block_k * 2
            b_half_bytes = block_n * block_k * 2
            if _prefetch and _parity_is_value:
                # Runtime parity: precompute a single i64 byte offset and
                # add it once to the LDS base per pass.
                _a_par_v = b.zext(b.mul(lds_parity, b.const_i32(a_half_bytes)), _I64)
                _b_par_v = b.zext(b.mul(lds_parity, b.const_i32(b_half_bytes)), _I64)
                a_lds_par_base = b.smem_ptr_add(_dtl_a_lds_base, _a_par_v)
                b_lds_par_base = b.smem_ptr_add(_dtl_b_lds_base, _b_par_v)
            else:
                a_lds_par_base = _dtl_a_lds_base
                b_lds_par_base = _dtl_b_lds_base
            a_parity_bytes_static = (
                lds_parity * a_half_bytes if _prefetch and not _parity_is_value else 0
            )
            b_parity_bytes_static = (
                lds_parity * b_half_bytes if _prefetch and not _parity_is_value else 0
            )
            c2 = b.const_i32(2)
            # Per-wave LDS offset. ``async_buffer_load_lds_addr`` is a
            # wave-level intrinsic: every wave writes
            # ``wave_size * BYTES_PER_LANE`` contiguous bytes starting
            # at the wave-uniform ``lds_dst``. With multiple waves per
            # WG, each wave must target a different LDS slice or they
            # stomp on each other. The block-wide ``chunk_idx`` already
            # uses ``tid``, so wave w covers chunks [w*wave .. (w+1)*wave),
            # which in the LDS layout (block_m x block_k row-major)
            # corresponds to a ``wave_size * BYTES_PER_LANE`` slice
            # starting at ``w * wave_size * BYTES_PER_LANE`` within the
            # per-pass region. We compute the per-wave LDS base once.
            _wave_bytes = spec.wave_size * _DTL_BYTES_PER_LANE
            if t.warp_m * t.warp_n * t.warp_k > 1:
                _warp_id = b.div(tid, c_wave)
                _wave_par_off = b.zext(b.mul(_warp_id, b.const_i32(_wave_bytes)), _I64)
                a_lds_wave_base = b.smem_ptr_add(a_lds_par_base, _wave_par_off)
                b_lds_wave_base = b.smem_ptr_add(b_lds_par_base, _wave_par_off)
            else:
                a_lds_wave_base = a_lds_par_base
                b_lds_wave_base = b_lds_par_base
            for p in range(_dtl_a_passes):
                pass_off_bytes = p * _dtl_pass_bytes + a_parity_bytes_static
                pass_lds_a = (
                    b.smem_ptr_add(
                        a_lds_wave_base, b.zext(b.const_i32(pass_off_bytes), _I64)
                    )
                    if pass_off_bytes > 0
                    else a_lds_wave_base
                )
                chunk_idx = b.add(tid, b.const_i32(p * spec.block_size))
                row = b.div(chunk_idx, _dtl_c_chunks_per_row)
                col_v = b.mod(chunk_idx, _dtl_c_chunks_per_row)
                col = b.mul(col_v, _dtl_c_halves_per_chunk)
                # element offset for A row in halves. With lds_swizzle, load the
                # global element at col^f(row) into the lane-linear LDS slot so
                # the swizzled ds_read (col_base^f(row)) fetches it back (XOR is
                # its own inverse) -- the LDS dest stays wave-contiguous.
                off_elems = b.add(
                    batch_off_a,
                    b.add(
                        b.mul(b.add(block_m_off, row), K),
                        b.add(k_off, _swz_col(col, row)),
                    ),
                )
                off_bytes = b.mul(off_elems, c2)
                b.async_buffer_load_lds_addr(
                    _dtl_a_rsrc,
                    pass_lds_a,
                    off_bytes,
                    _dtl_zero_soff,
                    _DTL_DWORDS,
                    coherency=spec.trait.dtl_cache_a,
                )
            for p in range(_dtl_b_passes):
                pass_off_bytes = p * _dtl_pass_bytes + b_parity_bytes_static
                pass_lds_b = (
                    b.smem_ptr_add(
                        b_lds_wave_base, b.zext(b.const_i32(pass_off_bytes), _I64)
                    )
                    if pass_off_bytes > 0
                    else b_lds_wave_base
                )
                chunk_idx = b.add(tid, b.const_i32(p * spec.block_size))
                row = b.div(chunk_idx, _dtl_c_chunks_per_row)
                col_v = b.mod(chunk_idx, _dtl_c_chunks_per_row)
                col = b.mul(col_v, _dtl_c_halves_per_chunk)
                off_elems = b.add(
                    batch_off_b,
                    b.add(
                        b.mul(_brow(b.add(block_n_off, row)), K),
                        b.add(k_off, _swz_col(col, row)),
                    ),
                )
                off_bytes = b.mul(off_elems, c2)
                b.async_buffer_load_lds_addr(
                    _dtl_b_rsrc,
                    pass_lds_b,
                    off_bytes,
                    _dtl_zero_soff,
                    _DTL_DWORDS,
                    coherency=spec.trait.dtl_cache_b,
                )
            # The following ``b.sync()`` (in the caller) lowers to
            # ``s_waitcnt vmcnt(0) lgkmcnt(0) ; s_barrier``, which drains
            # the in-flight DTLA writes before any wave reads LDS.
            return

        a_global_tile = make_tile_window(
            a_view,
            lengths=(1, block_m, block_k),
            origin=(batch_off_a, block_m_off, k_off),
        )
        b_global_tile = make_tile_window(
            b_view,
            lengths=(1, block_n, block_k),
            origin=(batch_off_b, block_n_off, k_off),
        )
        a_lds_tile = make_tile_window(
            a_lds_view,
            lengths=(block_m, block_k),
            origin=(b.const_i32(0), b.const_i32(0)),
        )
        b_lds_tile = make_tile_window(
            b_lds_view,
            lengths=(block_n, block_k),
            origin=(b.const_i32(0), b.const_i32(0)),
        )

        # Double-buffer parity row offset (compv4 non-DTL): the LDS
        # ``store_vec``/``store_scalar`` paths index the smem base
        # directly (origin is not applied for the lds addr space), so we
        # fold the parity offset into the destination ``row`` here. Half 1
        # lands at row block_m / block_n in the [2*block_m, block_k] /
        # [2*block_n, block_k] alloc. ``lds_parity`` is a Python int
        # (compile-time prologue) or an i32 Value (runtime loop iter-arg).
        def _ld_a_row(r: Value) -> Value:
            if not _db:
                return r
            if isinstance(lds_parity, Value):
                return b.add(r, b.mul(lds_parity, b.const_i32(block_m)))
            return b.add(r, b.const_i32(lds_parity * block_m)) if lds_parity else r

        def _ld_b_row(r: Value) -> Value:
            if not _db:
                return r
            if isinstance(lds_parity, Value):
                return b.add(r, b.mul(lds_parity, b.const_i32(block_n)))
            return b.add(r, b.const_i32(lds_parity * block_n)) if lds_parity else r

        # Coalesced (row, col_v, col) decode of a per-thread vec index,
        # shared by the A / B-canonical / B-preshuffle copy loops. Takes
        # the already-built ``vec_idx`` so the emitted SSA matches the
        # inline order verbatim (callers that also need ``vec_idx`` for a
        # global offset compute it before calling).
        def _vec_rc(vec_idx: Value) -> Tuple[Value, Value, Value]:
            row = b.div(vec_idx, c_block_k_div_vec)
            col_v = b.mod(vec_idx, c_block_k_div_vec)
            col = b.mul(col_v, c_load_vec) if load_vec > 1 else col_v
            return row, col_v, col

        def _pad_k_valid(elem_col: Value) -> Value:
            return b.cmp_lt(b.add(k_off, elem_col), K)

        def _zero_storage_scalar() -> Value:
            return b.cast_f32_to(b.const_f32(0.0), storage_dtype)

        def _mask_storage_scalar(value: Value, valid: Value) -> Value:
            return b.select(valid, value, _zero_storage_scalar())

        def _load_a_pad_k(row: Value, col: Value, n: int) -> Value:
            vals: List[Value] = []
            for i in range(n):
                elem_col = b.add(col, b.const_i32(i)) if i else col
                valid = _pad_k_valid(elem_col)
                safe_col = b.select(valid, elem_col, b.const_i32(0))
                raw = a_global_tile.load_scalar(b, b.const_i32(0), row, safe_col)
                vals.append(_mask_storage_scalar(raw, valid))
            return vals[0] if n == 1 else b.vec_pack(vals, storage_dtype)

        def _load_b_pad_k(row: Value, col: Value, n: int) -> Value:
            vals: List[Value] = []
            for i in range(n):
                elem_col = b.add(col, b.const_i32(i)) if i else col
                valid = _pad_k_valid(elem_col)
                safe_col = b.select(valid, elem_col, b.const_i32(0))
                raw = b_global_tile.load_scalar(b, b.const_i32(0), row, safe_col)
                vals.append(_mask_storage_scalar(raw, valid))
            return vals[0] if n == 1 else b.vec_pack(vals, storage_dtype)

        def _load_preshuffled_b_pad_k(
            base_off: Value, vec_idx: Value, col: Value, n: int
        ) -> Value:
            vals: List[Value] = []
            for i in range(n):
                elem_col = b.add(col, b.const_i32(i)) if i else col
                valid = _pad_k_valid(elem_col)
                raw_off = b.add(
                    base_off, b.add(b.mul(vec_idx, c_load_vec), b.const_i32(i))
                )
                safe_off = b.select(valid, raw_off, base_off)
                raw = b.global_load(Bp, safe_off, storage_dtype)
                vals.append(_mask_storage_scalar(raw, valid))
            return vals[0] if n == 1 else b.vec_pack(vals, storage_dtype)

        for e in range(a_vecs_per_thread):
            vec_idx = b.add(b.mul(b.const_i32(e), c_threads), tid)
            row, col_v, col = _vec_rc(vec_idx)
            if load_vec == 1:
                a_val = (
                    _load_a_pad_k(row, col, 1)
                    if spec.trait.pad_k
                    else a_global_tile.load_scalar(b, b.const_i32(0), row, col)
                )
                a_lds_tile.store_scalar(
                    b, _ld_a_row(row), _swz_col(col, row), value=a_val
                )
            else:
                a_val = (
                    _load_a_pad_k(row, col, load_vec)
                    if spec.trait.pad_k
                    else a_global_tile.load_vec(b, b.const_i32(0), row, col, n=load_vec)
                )
                a_lds_tile.store_vec(
                    b, _ld_a_row(row), _swz_col(col, row), value=a_val, n=load_vec
                )
        # B-load branch.
        #
        # Canonical (row-major B): each lane reads ``load_vec`` halves at
        # `(n_global_row, k_global_col)`; address arithmetic is strided
        # (one row of B is K halves apart). The wave's lanes hit
        # different rows -> multiple discontiguous VMEM transactions per
        # K-tile.
        #
        # ``preshuffle_b`` (B is pre-shuffled to ``(k_tiles, n_tiles,
        # block_n, block_k)`` per batch by the host): the (block_n x
        # block_k) tile for the current `(n_tile, k_tile)` lives as
        # ``block_n * block_k`` consecutive elements in B. Loading it
        # contiguously into ``B_smem`` (which is row-major
        # ``(block_n, block_k)``) yields **the same** in-LDS layout as
        # the canonical path, so the MFMA inner loop is unchanged. The
        # win is that the per-K global loads collapse to one wide
        # ``buffer_load_dwordxN`` burst per warp.
        if spec.trait.preshuffle_b:
            n_tile_idx = b.div(block_n_off, c_block_n)
            k_tile_idx = b.div(k_off, c_block_k)
            # ceil(N / block_n); pre-condition: caller pads N to a
            # multiple of block_n so the ceiling is N / block_n exactly.
            n_tile_count = b.div(N, c_block_n)
            tile_offset_elements = b.mul(
                b.add(b.mul(k_tile_idx, n_tile_count), n_tile_idx),
                b.const_i32(block_n * block_k),
            )
            base_off = b.add(batch_off_b, tile_offset_elements)
            for e in range(b_vecs_per_thread):
                vec_idx = b.add(b.mul(b.const_i32(e), c_threads), tid)
                glob_off = b.add(base_off, b.mul(vec_idx, c_load_vec))
                row, col_v, col = _vec_rc(vec_idx)
                if load_vec == 1:
                    b_val = (
                        _load_preshuffled_b_pad_k(base_off, vec_idx, col, 1)
                        if spec.trait.pad_k
                        else b.global_load(Bp, glob_off, storage_dtype)
                    )
                    b_lds_tile.store_scalar(
                        b, _ld_b_row(row), _swz_col(col, row), value=b_val
                    )
                else:
                    b_val = (
                        _load_preshuffled_b_pad_k(base_off, vec_idx, col, load_vec)
                        if spec.trait.pad_k
                        else b.global_load_vN(Bp, glob_off, storage_dtype, load_vec)
                    )
                    b_lds_tile.store_vec(
                        b, _ld_b_row(row), _swz_col(col, row), value=b_val, n=load_vec
                    )
        else:
            for e in range(b_vecs_per_thread):
                vec_idx = b.add(b.mul(b.const_i32(e), c_threads), tid)
                row, col_v, col = _vec_rc(vec_idx)
                if load_vec == 1:
                    b_val = (
                        _load_b_pad_k(row, col, 1)
                        if spec.trait.pad_k
                        else b_global_tile.load_scalar(b, b.const_i32(0), row, col)
                    )
                    b_lds_tile.store_scalar(
                        b, _ld_b_row(row), _swz_col(col, row), value=b_val
                    )
                else:
                    b_val = (
                        _load_b_pad_k(row, col, load_vec)
                        if spec.trait.pad_k
                        else b_global_tile.load_vec(
                            b, b.const_i32(0), row, col, n=load_vec
                        )
                    )
                    b_lds_tile.store_vec(
                        b, _ld_b_row(row), _swz_col(col, row), value=b_val, n=load_vec
                    )

    def _emit_frag_smem_load(
        src: Value,
        mn_in_atom: Value,
        k_in_atom: Value,
        atom_mn_base: Value,
        k_tile_base: Value,
        frag_len: int,
    ) -> Value:
        """Load one ``frag_len``-wide operand fragment from a row-major LDS tile.

        Both the A LDS tile ``(block_m, block_k)`` and the B LDS tile
        ``(block_n, block_k)`` are row-major with the M/N index as the row and K
        as the column. The fragment for one lane occupies a single tile row
        (``atom_mn_base + mn_in_atom``) and ``frag_len`` contiguous K columns
        starting at ``k_tile_base + k_in_atom`` — true for both the MFMA and
        WMMA layout maps, whose A/B fragment slots are K-contiguous. fp16/bf16
        smem loads cap at 8 lanes, so a wider fragment (WMMA ``<16 x half>``) is
        assembled from 8-wide chunks.
        """
        lds_row = b.add(atom_mn_base, mn_in_atom)
        lds_col = b.add(k_tile_base, k_in_atom)
        max_vec = 8 if storage_dtype in (F16, BF16) else frag_len
        if frag_len <= max_vec:
            return _emit_smem_load(b, src, lds_row, lds_col, frag_len, storage_dtype)
        frag = None
        for off in range(0, frag_len, max_vec):
            chunk = _emit_smem_load(
                b,
                src,
                lds_row,
                b.add(lds_col, b.const_i32(off)),
                max_vec,
                storage_dtype,
            )
            frag = chunk if frag is None else b.vec_concat(frag, chunk)
        return frag

    def _emit_wmma_phase(
        A_src: Value,
        B_src: Value,
        iter_vars: Sequence[Value],
    ) -> List[Value]:
        """One K-tile of WMMA atoms, fully MMA-contract driven.

        Operand fragments come straight from the op's A/B layout maps and the
        matmul is emitted target-neutrally (the backend picks the WMMA
        intrinsic). The A map yields ``(row, k)`` and the B map ``(k, col)``;
        in both LDS tiles the M/N index is the row and K the column, so we read
        the M-coord from the A map and the N-coord (= ``col``) from the B map.
        """
        a_map = op.a_layout()  # (row, k):   row = lane % 16, k = slot
        b_map = op.b_layout()  # (k, col):   col = lane % 16, k = slot
        a_row_in_atom, a_k_in_atom = a_map.coord(b, lane, 0)
        b_k_in_atom, b_col_in_atom = b_map.coord(b, lane, 0)
        warp_m_off = b.mul(warp_m_idx, b.const_i32(mfmas_m * t.warp_tile_m))
        warp_n_off = b.mul(warp_n_idx, b.const_i32(mfmas_n * t.warp_tile_n))
        new_accs: List[Value] = list(iter_vars)
        for kk in range(k_atoms):
            k_tile_base = b.const_i32(kk * t.warp_tile_k)
            a_rows = []
            for mi in range(mfmas_m):
                atom_row = b.add(warp_m_off, b.const_i32(mi * t.warp_tile_m))
                a_rows.append(
                    _emit_frag_smem_load(
                        A_src,
                        a_row_in_atom,
                        a_k_in_atom,
                        atom_row,
                        k_tile_base,
                        a_per_lane,
                    )
                )
            b_cols = []
            for ni in range(mfmas_n):
                atom_row = b.add(warp_n_off, b.const_i32(ni * t.warp_tile_n))
                b_cols.append(
                    _emit_frag_smem_load(
                        B_src,
                        b_col_in_atom,
                        b_k_in_atom,
                        atom_row,
                        k_tile_base,
                        b_per_lane,
                    )
                )
            flat = 0
            for mi in range(mfmas_m):
                for ni in range(mfmas_n):
                    new_accs[flat] = _emit_mma(
                        b, op, a_rows[mi], b_cols[ni], new_accs[flat]
                    )
                    flat += 1
        # WMMA intrawave compute schedule (the simple/mem k-loop keeps the LDS
        # store + global load in a separate barrier region, so this region holds
        # only the operand ds_reads + WMMAs). Opt-in via the 'wmma_v1' pipeline.
        if spec.trait.pipeline == "wmma_v1":
            from ...helpers.schedule import SchedulePolicy, WmmaHotLoopInstList

            il = WmmaHotLoopInstList.from_geometry(
                block_size=spec.block_size,
                m_per_block=t.tile_m,
                n_per_block=t.tile_n,
                k_per_block=t.tile_k,
                m_repeat=mfmas_m,
                n_repeat=mfmas_n,
                m_per_wmma=t.warp_tile_m,
                n_per_wmma=t.warp_tile_n,
                k_per_wmma=t.warp_tile_k,
                a_frag_len=a_per_lane,
                b_frag_len=b_per_lane,
                a_dtype_bytes=2,
                b_dtype_bytes=2,
            )
            SchedulePolicy.for_pipeline("wmma_v1").emit_wmma_compute_schedule(b, il)
        return new_accs

    _swap_ab = bool(spec.trait.mfma_swap_ab)

    def _mma_ab(a_frag: Value, b_frag: Value, acc: Value, ni=None) -> Value:
        # mfma_swap_ab: D' = B * A^T = C^T in the MFMA output layout (the A and B
        # operand lane layouts of the square 16x16 / 32x32 atoms are identical).
        if _pe is not None:
            return _pe.mma(
                lambda x, y, c: _emit_mma(b, op, x, y, c),
                a_frag,
                b_frag,
                acc,
                _swap_ab,
                (
                    (lambda d: b.inline_asm("", "=a,0", [d], d.type, sideeffect=False))
                    if spec.trait.acc_agpr_pin
                    else None
                ),
                ni=ni,
            )
        if _swap_ab:
            d = _emit_mma(b, op, b_frag, a_frag, acc)
        else:
            d = _emit_mma(b, op, a_frag, b_frag, acc)
        if spec.trait.acc_agpr_pin:
            # pin the accumulator to the AGPR file (``"=a,0"`` after every MFMA)
            d = b.inline_asm("", "=a,0", [d], d.type, sideeffect=False)
        return d

    def emit_mfma_phase(
        A_src: Value,
        B_src: Value,
        iter_vars: Sequence[Value],
        lds_parity=0,
    ) -> List[Value]:
        """One K-tile worth of MMAs across all per-warp atom positions
        and every K atom step inside this K-tile.

        On RDNA (WMMA) this delegates to the fully contract-driven
        :func:`_emit_wmma_phase`. On CDNA (MFMA) it keeps the byte-identical
        arch-formula fragment load below; the matmul + accumulator length are
        contract-driven (``_emit_mma`` / ``op.c_frag_len``).

        ``lds_parity`` (DTLA + prefetch only): shifts every A row by
        ``parity * block_m`` and every B row by ``parity * block_n`` so
        the reads target the half-buffer that holds the freshly-loaded
        K-tile. Ignored when prefetch is off. May be a Python int or
        runtime i32 Value.
        """
        if op.family == "wmma":
            return _emit_wmma_phase(A_src, B_src, iter_vars)
        _mp_is_val = isinstance(lds_parity, Value)
        if _prefetch or _db:
            if _mp_is_val:
                a_par_row_v = b.mul(lds_parity, b.const_i32(block_m))
                b_par_row_v = b.mul(lds_parity, b.const_i32(block_n))
                a_row_parity_off = 0  # carried as a Value addition below
                b_row_parity_off = 0
            else:
                a_par_row_v = None
                b_par_row_v = None
                a_row_parity_off = lds_parity * block_m
                b_row_parity_off = lds_parity * block_n
        else:
            a_par_row_v = None
            b_par_row_v = None
            a_row_parity_off = 0
            b_row_parity_off = 0
        # Lane mapping into LDS: A wants per-lane `a_per_lane` K-elements
        # starting at K = k_blk * a_per_lane. The decode is identical for the
        # 16x16 and 32x32 atoms — both put the M-in-atom in ``lane %
        # warp_tile_m``, the N-in-atom in ``lane % warp_tile_n``, and the
        # K-sub-block in ``lane / warp_tile_m`` — so there is no per-shape
        # branch. (The op's A/B layout maps encode the same arithmetic; we
        # keep the inline form here because ``emit_mfma_phase`` also needs the
        # raw ``k_blk`` quotient, which the slot-indexed layout map folds into
        # its ``k`` coordinate.)
        m_in_atom = b.mod(lane, b.const_i32(t.warp_tile_m))
        k_blk = b.div(lane, b.const_i32(t.warp_tile_m))
        n_in_atom = b.mod(lane, b.const_i32(t.warp_tile_n))

        warp_m_off = b.mul(warp_m_idx, b.const_i32(mfmas_m * t.warp_tile_m))
        warp_n_off = b.mul(warp_n_idx, b.const_i32(mfmas_n * t.warp_tile_n))

        # k_blk * a_per_lane is the lane-local K offset inside the
        # current K-tile; reused across every kk to form ``col_base``.
        # Hoisting saves a mul + add per kk and matches CK Tile's
        # block_gemm where the lane-K base is computed once outside
        # the warp_gemm_dispatcher inner loop.
        k_blk_kbase = b.mul(k_blk, b.const_i32(a_per_lane * _kmul))

        new_accs: List[Value] = list(iter_vars)

        # ---- per-fragment LDS read helpers (shared by the legacy and PLR
        # emission paths). Each returns one ds_read'd operand fragment for the
        # given K-atom step ``kk`` and atom-row/col index. Pure address arith +
        # one ``_emit_smem_load``; identical math to the inlined legacy form. ----
        def _read_a(kk: int, mi: int) -> Value:
            col_base = b.add(k_blk_kbase, b.const_i32(kk * t.warp_tile_k * _kmul))
            a_row = b.add(
                warp_m_off,
                b.add(b.const_i32(mi * t.warp_tile_m + a_row_parity_off), m_in_atom),
            )
            if a_par_row_v is not None:
                a_row = b.add(a_row, a_par_row_v)
            # Parity-free row for the swizzle mask (loop-invariant -> LICM hoists
            # the lshr/mod/shl); the address still uses the parity-full a_row.
            swz_row = (
                b.add(warp_m_off, b.add(b.const_i32(mi * t.warp_tile_m), m_in_atom))
                if _hoist_swz
                else a_row
            )
            return _frag_ld(A_src, a_row, col_base, swz_row, a_per_lane, "A")

        def _read_b(kk: int, ni: int) -> Value:
            col_base = b.add(k_blk_kbase, b.const_i32(kk * t.warp_tile_k * _kmul))
            b_row = b.add(
                warp_n_off,
                b.add(b.const_i32(ni * t.warp_tile_n + b_row_parity_off), n_in_atom),
            )
            if b_par_row_v is not None:
                b_row = b.add(b_row, b_par_row_v)
            swz_row = (
                b.add(warp_n_off, b.add(b.const_i32(ni * t.warp_tile_n), n_in_atom))
                if _hoist_swz
                else b_row
            )
            return _frag_ld(B_src, b_row, col_base, swz_row, b_per_lane, "B")

        def _mma_cluster(a_rows, b_cols) -> None:
            # Reference-faithful MFMA clusters: bracket each m-row's MFMA
            # cluster with s_setprio(1)/(0) + a sched_barrier(0) fence so the
            # post-RA scheduler keeps the matrix pipe at high priority through
            # the cluster and cannot float the surrounding ds_read/buffer_load
            # across it. Enabled with lds_swizzle (the reference-faithful mode).
            if _interwave:
                # interwave_setprio (CK Tile's Interwave bookends): fence the
                # K-atom step's operand ds_reads off, then issue the whole
                # mfmas_m x mfmas_n MFMA cluster at raised wave priority so a
                # wave that holds its operands keeps the matrix pipe while the
                # other waves of the SIMD wait on LDS / VMEM; drop the priority
                # and fence again before the next step's reads.
                b.sched_barrier(0)
                b.s_setprio(1)
                for mi in range(mfmas_m):
                    for ni in range(mfmas_n):
                        flat = mi * mfmas_n + ni
                        new_accs[flat] = _mma_ab(
                            a_rows[mi], b_cols[ni], new_accs[flat], ni=ni
                        )
                b.s_setprio(0)
                b.sched_barrier(0)
                return
            for mi in range(mfmas_m):
                if _SWZ and _sched_hints:
                    b.s_setprio(1)
                for ni in range(mfmas_n):
                    flat = mi * mfmas_n + ni
                    new_accs[flat] = _mma_ab(
                        a_rows[mi], b_cols[ni], new_accs[flat], ni=ni
                    )
                if _SWZ and _sched_hints:
                    b.s_setprio(0)
                    b.sched_barrier(0)

        for kk in range(k_atoms):
            # We compute one A load per atom-row and reuse across N.
            a_rows = [_read_a(kk, mi) for mi in range(mfmas_m)]
            # B load per atom-col, reused across M.
            b_cols = [_read_b(kk, ni) for ni in range(mfmas_n)]
            _mma_cluster(a_rows, b_cols)

        # Quantitative two-stage HotLoop schedule (E3, E6 L1).
        #
        # The flat per-kk hint (one DS-read group + one all-MFMA group) left
        # ds_write / buffer-load placement entirely to the backend. The
        # comp_v3 / comp_v4 HotLoopScheduler instead pins the whole K-tile's
        # MFMA / ds_read / ds_write / VMEM interleave to the measured MFMA
        # shadow, derived purely from the tile geometry + dtype + the spec
        # atom's timing (``MfmaAtom.mfma_cycle`` / ``k_per_xdlops``). It is
        # issued ONCE per K-tile (matching the C++, which calls
        # ``HotLoopScheduler()`` once per hot iteration) rather than per kk.
        #
        # ``sched_group_barrier`` is a pure scheduling hint -- changing only
        # its operands cannot change the kernel's numeric result, so this is
        # numerically identical to the flat-hint emission.
        if _sched_hints and spec.trait.pipeline in ("compv3", "compv4"):
            _emit_hotloop_schedule(b, spec, load_vec)

        return new_accs

    # ---- active-tile gate ----
    # When ``trait.active_tile_skip`` is on, the K-loop + epilogue
    # are wrapped in ``scf.if(do_work)``. The condition is wave-
    # uniform (CTA-uniform inputs + a single CTA-uniform global
    # load) so the AMDGPU backend collapses the whole if-then to
    # a scalar branch around the body; inactive tiles do one
    # ``buffer_load_dword`` of the bucket head and exit.
    do_work_cond: Optional[Value] = None
    if spec.batched and spec.trait.active_tile_skip:
        # Bucket head index for THIS CTA's (expert, m-tile) slot:
        # ``block_id_z * slot_size + block_m_off``. Using
        # ``block_m_off`` (already chiplet-swizzle-aware and
        # tile_m-aligned) keeps the gate consistent with the actual
        # rows the address arithmetic below will read.
        bucket_head = b.add(b.mul(b.block_id_z(), slot_size_p), block_m_off)
        first_token = b.global_load_i32(sorted_token_ids, bucket_head)
        do_work_cond = b.cmp_ge(first_token, c0)

    def _ep_default(
        vals: Sequence[Value],
        bm: Value,
        bn: Value,
        batch_off_c=None,
        fused_epilogue=None,
    ) -> None:
        """``_emit_epilogue_default`` of one output tile; with ``pad_fast_tile``
        a uniform branch sends tiles fully inside M x N to the unguarded body."""
        if _gett is not None:
            _gett.epilogue(
                b,
                vals,
                bm,
                bn,
                C=C,
                warp_m_idx=warp_m_idx,
                warp_n_idx=warp_n_idx,
                lane=lane,
                acc_out=lambda v: _acc_agpr_out(b, spec, v),
            )
            return
        args = (warp_m_idx, warp_n_idx, lane, bm, bn, M, N, C, c_per_lane)
        kw = dict(batch_off_c=batch_off_c, fused_epilogue=fused_epilogue)
        if not spec.trait.pad_fast_tile:
            _emit_epilogue_default(b, spec, op, vals, *args, **kw)
            return
        full = None
        if spec.trait.pad_m:
            full = b.cmp_le(b.add(bm, c_block_m), M)
        if spec.trait.pad_n:
            fn = b.cmp_le(b.add(bn, c_block_n), N)
            full = fn if full is None else b.land(full, fn)
        nopad = _dc_replace(
            spec, trait=_dc_replace(spec.trait, pad_m=False, pad_n=False)
        )
        with b.scf_if_else(full) as (fast, edge):
            with fast:
                _emit_epilogue_default(b, nopad, op, vals, *args, **kw)
            with edge:
                _emit_epilogue_default(b, spec, op, vals, *args, **kw)

    def emit_compute_and_epilogue() -> None:
        if _stages >= 2:
            _emit_kloop_staged()
            if _for_results is None:  # persistent: the epilogue ran per tile
                return
        elif _prefetch:
            _emit_kloop_prefetch()
        elif _db:
            _emit_kloop_db()
        else:
            _emit_kloop_simple()
        _emit_epilogue()

    def _emit_kloop_db() -> None:
        """compv4 double-buffered K-loop (non-DTL, VGPR-staged).

        Two LDS half-buffers per operand. Each iteration issues the
        *next* K-tile's global->VGPR->LDS copy into the half the MFMAs
        are not reading, so the global-load + ds_write traffic overlaps
        the current tile's ds_reads + MFMAs. ``parity`` is carried as an
        i32 iter-arg and flips each step.

        Barrier discipline (single barrier per iter, placed at the
        *start*, half ``p`` = current read, ``p^1`` = next write):
          * sync() : drains the previous iter's ds_writes (so half p,
                     loaded last iter, is RAW-safe to read) AND the
                     previous iter's ds_reads (so half p^1 is WAR-safe to
                     overwrite), then barriers all waves.
          * issue load -> half(p^1)   (next tile, overlaps this MFMA)
          * MFMA <- half(p)
        The prologue issues tile 0's load into half 0 (no barrier — the
        loop's first start-sync covers it); the epilogue runs the final
        tile's MFMA from the last-written half after one drain barrier.
        """
        c1_i32 = b.const_i32(1)
        K_minus_one_tile = b.sub(_k_upper, c_block_k)

        # Prologue: issue tile 0's load into half 0. The first iteration's
        # start-sync drains these ds_writes before the MFMA reads half 0.
        # In split-K mode tile 0 is the slice base ``k_lo``.
        emit_load_phase(A_smem, B_smem, k_lo, lds_parity=0)

        loop_args = [("par", c0)] + list(accs)
        for_op = b.scf_for_iter(
            k_lo, K_minus_one_tile, c_block_k, loop_args, iv_name="k0"
        )
        with for_op as (k0, iter_vars):
            parity = iter_vars[0]
            acc_iter = iter_vars[1:]
            next_parity = b.sub(c1_i32, parity)
            k_next = b.add(k0, c_block_k)
            # Start-of-iter barrier: makes half(parity) RAW-safe to read
            # (its load was issued last iter / prologue) and half(next)
            # WAR-safe to overwrite (its reads finished last iter).
            b.sync()
            # Issue next-tile global->VGPR->LDS copy into the other half;
            # this VMEM + ds_write stream overlaps the MFMAs below.
            emit_load_phase(A_smem, B_smem, k_next, lds_parity=next_parity)
            new_accs = emit_mfma_phase(A_smem, B_smem, acc_iter, lds_parity=parity)
            b.scf_yield(next_parity, *new_accs)

        # Epilogue: drain the last in-loop load (it wrote the K-1 tile
        # into half ``final_parity``), barrier, then MFMA that tile.
        final_parity = for_op.results[0]
        b.sync()
        epi_accs = emit_mfma_phase(
            A_smem, B_smem, for_op.results[1:], lds_parity=final_parity
        )
        nonlocal _for_results
        _for_results = epi_accs

    # K-loop upper bound: full ``K`` (byte-identical) or the split-K slice
    # end ``k_hi``. Computed lazily so the non-split path's SSA is unchanged.
    _k_upper = K if k_hi is None else k_hi

    def _emit_kloop_simple() -> None:
        for_op = b.scf_for_iter(k_lo, _k_upper, c_block_k, accs, iv_name="k0")
        with for_op as (k0, iter_vars):
            # Single-buffer, load-then-compute pipeline (``mem``,
            # ``compv3``, and the cshuffle-epilogue ``compv4`` that opts
            # out of AB double-buffering). compv4 with the direct epilogue
            # instead routes through ``_emit_kloop_db``; the scheduler
            # hints in ``emit_mfma_phase`` carry the in-tile interleave.
            emit_load_phase(A_smem, B_smem, k0)
            b.sync()

            new_accs = emit_mfma_phase(A_smem, B_smem, iter_vars)

            b.sync()
            b.scf_yield(*new_accs)
        nonlocal _for_results
        _for_results = for_op.results

    def _emit_kloop_prefetch() -> None:
        """Software-pipelined K-loop with DTLA ping-pong.

        Structure:
          prologue:        DTLA load tile 0 -> half 0
          for k in [0, K-block_k) step block_k:
              parity = (k / block_k) & 1   ; next_parity = parity ^ 1
              DTLA load tile k+block_k -> half (parity ^ 1)
              s_waitcnt vmcnt(loads_in_flight_for_next_tile) ; barrier
              MFMA from half parity
              ; no end-barrier — half (parity ^ 1) has its own LDS region
          epilogue:        ; last tile already loaded into the final parity
              s_waitcnt vmcnt(0) ; barrier
              MFMA from final parity

        The post-issue ``s_waitcnt vmcnt(N)`` keeps N loads (= next
        tile's count) in flight while the current tile drains, so the
        loop's MFMA work overlaps the next tile's HBM transfers.
        """
        loads_per_tile = _dtl_a_passes + _dtl_b_passes
        # vmcnt is 6 bits on gfx950 (max 63). If next-tile loads exceed
        # that, we'd saturate and the prefetch buys nothing extra.
        if loads_per_tile > 63:
            # Fall back to the non-prefetched path; the constant would
            # have to be encoded as 63 either way.
            _emit_kloop_simple()
            return

        nonlocal _for_results

        # Fully-unrolled fixed-K variant (ROCKE_EXP_UNROLL_K): no scf.for
        # backedge; Python-unroll the K-tiles with static (compile-time) parity
        # so the double-buffer LDS offsets fold to constants. Same single-barrier
        # software pipeline (prologue load tile0; per tile: drain + barrier +
        # load next + MFMA current; epilogue MFMA last). split_k==1 only.
        if _unroll_k > 0 and not _is_split_k:
            tk = t.tile_k
            trip = _unroll_k // tk
            D = max(1, _pf_depth)  # tiles loaded ahead
            nbuf = _nbuf  # ring size (= D+1 when depth>1, else 2)
            acc = [a for (_n, a) in accs]
            # Prologue: issue the first D tiles' loads into rings 0..D-1.
            for j in range(min(D, trip)):
                emit_load_phase(
                    A_smem, B_smem, b.const_i32(j * tk), lds_parity=j % nbuf
                )
            # Steady state: per tile i, drain (vmcnt(0): buffer_load_lds is
            # out-of-order, so a partial drain would be incorrect), barrier,
            # issue tile i+D's load, then MFMA tile i from ring i%nbuf. With the
            # mandatory full drain, depth>1 keeps no extra loads usefully in
            # flight -- the empirical confirmation of the wsp3 finding.
            for i in range(trip):
                b.s_waitcnt(vmcnt=0, lgkmcnt=0)
                b.s_barrier_bare()
                nj = i + D
                if nj < trip:
                    emit_load_phase(
                        A_smem, B_smem, b.const_i32(nj * tk), lds_parity=nj % nbuf
                    )
                acc = emit_mfma_phase(A_smem, B_smem, acc, lds_parity=i % nbuf)
            _for_results = acc
            return

        # Prologue: load tile 0 into half 0 (the slice base in split-K mode).
        emit_load_phase(A_smem, B_smem, k_lo, lds_parity=0)

        # Loop bounds: iterate k in [k_lo, k_hi - block_k), so the last tile
        # is handled by the epilogue (which doesn't need to issue a
        # next-tile load).
        K_minus_one_tile = b.sub(_k_upper, c_block_k)
        # iter_args: (parity_i32, acc...). parity flips each iter.
        c1_i32 = b.const_i32(1)
        loop_args = [("par", c0)] + list(accs)
        for_op = b.scf_for_iter(
            k_lo, K_minus_one_tile, c_block_k, loop_args, iv_name="k0"
        )
        with for_op as (k0, iter_vars):
            parity = iter_vars[0]
            acc_iter = iter_vars[1:]
            next_parity = b.sub(c1_i32, parity)  # 1 - parity (0/1 only)
            k_next = b.add(k0, c_block_k)
            # Single-barrier software pipeline: ONE s_waitcnt + ONE WG barrier
            # per K-tile (vs the prior WAR+RAW two-barrier form, which halved
            # the available MFMA-shadow time at 1 WG/CU). The single barrier
            # serves both hazards because we issue the next-tile write AFTER it:
            #   * vmcnt(0)  -> the current tile's DTL loads (issued last iter
            #                  into half(parity)) have landed: RAW-safe to read.
            #   * lgkmcnt(0)-> the previous iter's ds_reads of half(next_parity)
            #                  have drained: WAR-safe to overwrite that half.
            #   * s_barrier -> WG rendezvous so the freshly-loaded current half
            #                  is visible to every wave before any MFMA reads it.
            # The async next-tile load is issued AFTER the barrier (cannot race
            # the just-drained reads -> no second barrier needed) but BEFORE the
            # MFMAs (its HBM transfer overlaps the matrix work). The prior
            # structure issued that write BEFORE draining, which both raced and
            # forced the extra barrier this collapses away.
            b.s_waitcnt(vmcnt=0, lgkmcnt=0)
            b.s_barrier_bare()
            emit_load_phase(A_smem, B_smem, k_next, lds_parity=next_parity)
            new_accs = emit_mfma_phase(A_smem, B_smem, acc_iter, lds_parity=parity)
            b.scf_yield(next_parity, *new_accs)

        # Epilogue: drain the final tile's loads, rendezvous, MFMA last tile.
        final_parity = for_op.results[0]
        b.s_waitcnt(vmcnt=0, lgkmcnt=0)
        b.s_barrier_bare()
        epi_accs = emit_mfma_phase(
            A_smem, B_smem, for_op.results[1:], lds_parity=final_parity
        )
        _for_results = epi_accs

    def _emit_kloop_staged() -> None:
        """Alias-scoped DTL LDS ring (``dtl_stages = S >= 2``).

        Every stage j has its own A / B allocation (``A_stg[j]`` / ``B_stg[j]``)
        and the kernel carries ``lds_alias_scopes``, so the compiler can tell the
        stage a ds_read reads from the stage an in-flight ``buffer_load ... lds``
        writes. The K-loop is unrolled by S so every stage index is static. With
        ``T = (k_hi - k_lo) / block_k`` tiles and ``L`` DTL loads per tile (per
        wave):

        default order (per tile i, stage s = i % S)::

            s_waitcnt vmcnt((S-2)*L) lgkmcnt(0); s_barrier      # tile i landed
            DTL tile i+S-1 -> stage (s-1) % S                   # freed by tile i-1
            MFMA tile i from stage s

        ``dtl_frag_prefetch`` order: operand fragments are register-prefetched
        one K atom ahead, and the wait + barrier + DTL issue + the first
        fragment reads of tile i+1 sit in front of the LAST K atom's MFMA
        cluster of tile i (inside the MFMA stream)::

            for kk < last: frags(kk+1) <- stage s;  MFMA(kk)
            kk == last:    s_waitcnt vmcnt((S-2)*L) lgkmcnt(0); s_barrier
                           DTL tile i+S -> stage s              # fully read
                           frags(0) of tile i+1 <- stage (s+1) % S
                           MFMA(last)

        Loads past the K range are issued with an out-of-range buffer offset
        (the DMA returns zeros into a dead stage, no memory traffic), so every
        tile issues exactly L loads and the partial ``vmcnt`` counts stay
        static. The main loop covers ``(T // S) * S`` tiles; the remaining
        ``T % S`` tiles run in 0/1-trip loops with static stages.
        """
        nonlocal _for_results
        S = _stages
        if _gett is not None:
            L = _gett.passes("A") + _gett.passes("B")
        else:
            L = _dtl_a_passes + _dtl_b_passes
        keep = (S - 2) * L
        # dtl_rsrc_word3: DWORD3 of every DTL resource (None = target default)
        _w3 = int(spec.trait.dtl_rsrc_word3) or None
        fprf = bool(spec.trait.dtl_frag_prefetch)
        # in-stream barrier: the wait + barrier for tile i+1 (and the DTL issue
        # of tile i+S) sit in front of the last K atom's MFMA cluster of tile i
        instream = fprf or bool(spec.trait.dtl_stream_barrier)
        spread = bool(spec.trait.dtl_issue_spread)
        iglp = spec.trait.iglp_opt
        ilv = bool(spec.trait.mfma_interleave)
        k_up = _k_upper
        c2s = b.const_i32(2)
        # k_stagger (StaggerU): per-CTA K start offset (elements) and wrap
        _kst = int(spec.trait.k_stagger)
        if _kst:
            _kspan = b.to_sgpr_u32(b.sub(k_up, k_lo))
            if int(spec.trait.k_stagger_map) == 1:
                # launch-order linear id (grid x = the N tile count)
                _gx = b.div(b.add(N, b.const_i32(block_n - 1)), c_block_n)
                _lin = b.add(b.mul(b.block_id_y(), _gx), b.block_id_x())
            else:
                _lin = b.add(b.block_id_x(), b.block_id_y())
            _sidx = b.mod(b.to_sgpr_u32(_lin), b.const_i32(_kst))
            _koff = b.to_sgpr_u32(
                b.mul(
                    b.mod(
                        b.mul(
                            _sidx, b.const_i32(max(1, int(spec.trait.k_stagger_tiles)))
                        ),
                        b.div(_kspan, c_block_k),
                    ),
                    c_block_k,
                )
            )

        _clampk = bool(spec.trait.dtl_clamp_k)
        if _clampk:
            _kmax = b.to_sgpr_u32(b.sub(k_up, c_block_k))

        def _ka(k: Value) -> Value:
            """k_stagger: the K element whose DMA logical K ``k`` issues
            (dtl_clamp_k: past the end, the last tile's)."""
            if not _kst:
                return _kp(b.smin(k, _kmax) if _clampk else k)
            t_ = b.add(k, _koff)
            return b.select(b.cmp_lt(t_, k_up), t_, b.sub(t_, _kspan))

        from ...core.ir import I64 as _I64s

        def _kp(k: Value) -> Value:
            """``stagger_u``: the physical K offset of logical K offset ``k``
            (this workgroup's start offset added, wrapped at K)."""
            if _su_off is None:
                return k
            kk = b.add(k, _su_off)
            return b.select(b.cmp_lt(kk, k_up), kk, b.sub(kk, k_up))

        # ---- loop-invariant DTL addressing. The wave-uniform parts live in
        # SGPRs: the per-wave LDS slice (-> M0 without a readfirstlane per
        # load) and the per-tile global origin (``soffset``); only the per-lane
        # (row * K + col) byte offset of each pass is a VGPR, and it is
        # loop-invariant. Tiles past the K range use a zero-size buffer
        # resource (every lane out of range -> the DMA writes zeros into the
        # dead stage and moves no memory). ----
        _wave_bytes = spec.wave_size * _DTL_BYTES_PER_LANE
        # lds_row_interleave: every wave's 1 KiB DTL block is followed by the
        # rilv_pad-byte (16) row pad (one padded alloc row per wave and pass)
        _pass_bytes = _dtl_pass_bytes
        if _rilv:
            _wave_bytes += int(spec.trait.rilv_pad)
            _pass_bytes = (spec.block_size // spec.wave_size) * _wave_bytes
        if t.warp_m * t.warp_n * t.warp_k > 1:
            _wv = b.to_sgpr_u32(warp_id)
            _wpo = b.zext(b.to_sgpr_u32(b.mul(_wv, b.const_i32(_wave_bytes))), _I64s)
        else:
            _wpo = None

        def _stage_base(smem: Value) -> Value:
            base = b.smem_addr_of(smem)
            return b.smem_ptr_add(base, _wpo) if _wpo is not None else base

        if _gett is not None:
            # per-operand DTL width: the per-wave LDS slice is 64 lanes x the
            # operand's bytes per lane
            def _gett_base(smem: Value, nm: str) -> Value:
                base = b.smem_addr_of(smem)
                if t.warp_m * t.warp_n * t.warp_k == 1:
                    return base
                wb = _gett.wave_bytes(nm)
                off = b.zext(b.to_sgpr_u32(b.mul(_wv, b.const_i32(wb))), _I64s)
                return b.smem_ptr_add(base, off)

            a_wbase = [_gett_base(A_stg[j], "A") for j in range(S)]
            b_wbase = [_gett_base(B_stg[j], "B") for j in range(S)]
            _gett.bind(
                b, tid=tid, lane=lane, warp_m_idx=warp_m_idx, warp_n_idx=warp_n_idx
            )
        else:
            a_wbase = [_stage_base(A_stg[j]) for j in range(S)]
            b_wbase = [_stage_base(B_stg[j]) for j in range(len(B_stg))]

        def _lane_offs(n_pass: int, mf: int = 0, isb: bool = False) -> List[Value]:
            offs = []
            _rr = _brow if isb else (lambda r_: r_)
            for p in range(n_pass):
                chunk_idx = b.add(tid, b.const_i32(p * spec.block_size))
                row = b.div(chunk_idx, _dtl_c_chunks_per_row)
                col = b.mul(
                    b.mod(chunk_idx, _dtl_c_chunks_per_row), _dtl_c_halves_per_chunk
                )
                if _rilv and spec.trait.rilv_vec_epilogue:
                    # linear rows: the C mapping carries the interleave
                    offs.append(b.mul(b.add(b.mul(_rr(row), K), col), c2s))
                    continue
                if _rilv:
                    # LDS row R of a warp block of 16*mf rows holds global row
                    # (R % mf) * 16 + R // mf of that block (no column swizzle)
                    wblk = b.const_i32(16 * mf)
                    r_in = b.mod(row, wblk)
                    row = b.add(
                        b.mul(b.div(row, wblk), wblk),
                        b.add(
                            b.mul(b.mod(r_in, b.const_i32(mf)), b.const_i32(16)),
                            b.div(r_in, b.const_i32(mf)),
                        ),
                    )
                    offs.append(b.mul(b.add(b.mul(row, K), col), c2s))
                    continue
                offs.append(b.mul(b.add(b.mul(_rr(row), K), _swz_col(col, row)), c2s))
            return offs

        if _gett is not None:
            a_voff = b_voff = None
        else:
            a_voff = _lane_offs(_dtl_a_passes, mfmas_m)
            b_voff = _lane_offs(_dtl_b_passes, mfmas_n, isb=True)
        # prologue stage='vgpr': where each lane's DMA chunk would land (alloc row, col)
        _vg = _pe is not None and _pe.vgpr_stage
        _vg_pend: List = []
        if _vg:
            if spec.trait.dtl_tight_rsrc:
                raise ValueError(
                    "prologue stage='vgpr' does not support dtl_tight_rsrc"
                )
            if int(spec.trait.pingpong) == 7:
                raise ValueError("prologue stage='vgpr' does not support pingpong 7")
            if _DTL_HALVES != 8:
                raise ValueError("prologue stage='vgpr' needs 16-byte DTL chunks")
            _nw = spec.block_size // spec.wave_size
            _lane64 = b.mod(tid, c_wave)

            def _vg_rc(n_pass: int):
                out_ = []
                for p_ in range(n_pass):
                    if _rilv:
                        if 8 * block_k * 2 + 16 != _wave_bytes:
                            raise ValueError(
                                "stage='vgpr' + lds_row_interleave: block_k 64 only"
                            )
                        out_.append(
                            (
                                b.add(b.const_i32(p_ * _nw), warp_id),
                                b.mul(_lane64, b.const_i32(8)),
                            )
                        )
                    else:
                        ci = b.add(tid, b.const_i32(p_ * spec.block_size))
                        out_.append(
                            (
                                b.div(ci, _dtl_c_chunks_per_row),
                                b.mul(b.mod(ci, _dtl_c_chunks_per_row), b.const_i32(8)),
                            )
                        )
                return out_

            _vg_a_rc = _vg_rc(_dtl_a_passes)
            _vg_b_rc = _vg_rc(_dtl_b_passes)
            _vg_a_vo = [b.mul(v_, c2s) for v_ in a_voff]
            _vg_b_vo = [b.mul(v_, c2s) for v_ in b_voff]
            _c16b = b.const_i32(16)

        def _vg_commit() -> None:
            """stage='vgpr': convert + ds_write every pending chunk."""
            while _vg_pend:
                x0, x1, alloc, (r_, c_), which_ = _vg_pend.pop(0)
                x = b.vec_concat(x0, x1)
                b.smem_store_vN(alloc, [r_, c_], _pe.stage_convert(x, which_), 8)

        if int(spec.trait.pingpong) == 7:
            # pingpong 7: one group (half the waves) loads a whole operand tile
            _nthr_g = spec.block_size // 2
            _tid_g = b.mod(tid, b.const_i32(_nthr_g))
            _gpass_bytes = _nthr_g * _DTL_BYTES_PER_LANE

            def _lane_offs_g(n_pass: int) -> List[Value]:
                offs = []
                for p_ in range(n_pass):
                    chunk_idx = b.add(_tid_g, b.const_i32(p_ * _nthr_g))
                    row = b.div(chunk_idx, _dtl_c_chunks_per_row)
                    col = b.mul(
                        b.mod(chunk_idx, _dtl_c_chunks_per_row), _dtl_c_halves_per_chunk
                    )
                    offs.append(b.mul(b.add(b.mul(row, K), _swz_col(col, row)), c2s))
                return offs

            _gp_a = 2 * _dtl_a_passes
            _gp_b = 2 * _dtl_b_passes
            a_voff_g = _lane_offs_g(_gp_a)
            b_voff_g = _lane_offs_g(_gp_b)
            _wv_g = b.mod(b.to_sgpr_u32(warp_id), b.const_i32(t.warp_m * t.warp_n // 2))
            _wpo_g = b.zext(
                b.to_sgpr_u32(b.mul(_wv_g, b.const_i32(_wave_bytes))), _I64s
            )
            a_wbase_g = [
                b.smem_ptr_add(b.smem_addr_of(A_stg[j]), _wpo_g) for j in range(S)
            ]
            b_wbase_g = [
                b.smem_ptr_add(b.smem_addr_of(B_stg[j]), _wpo_g)
                for j in range(len(B_stg))
            ]

            def _dtl_issues_g(op: str, stage: int, k_ld: Value, tile0s=None, kup=None):
                """pingpong 7: the whole ``op`` tile ``k_ld`` into ``stage``, issued by
                this wave's group (2x the per-wave passes of the block-wide split)."""
                ta0, tb0 = tile0s if tile0s is not None else (a_tile0, b_tile0)
                ok = b.cmp_lt(k_ld, k_up if kup is None else kup)
                nrec = b.select(ok, c_big, c_zero_rec)
                if op == "A":
                    ptr, n_pass, voffs, wb, tile0, coh = (
                        A,
                        _gp_a,
                        a_voff_g,
                        a_wbase_g,
                        ta0,
                        spec.trait.dtl_cache_a,
                    )
                else:
                    ptr, n_pass, voffs, wb, tile0, coh = (
                        Bp,
                        _gp_b,
                        b_voff_g,
                        b_wbase_g,
                        tb0,
                        spec.trait.dtl_cache_b,
                    )
                rsrc = b.buffer_rsrc(ptr, nrec, word3=_w3)
                soff = b.to_sgpr_u32(
                    b.select(ok, b.mul(b.add(tile0, k_ld), c2s), c_zero_rec)
                )
                out = []
                for p_ in range(n_pass):

                    def _one(p_=p_):
                        lds = wb[stage]
                        if p_:
                            lds = b.smem_ptr_add(
                                lds, b.zext(b.const_i32(p_ * _gpass_bytes), _I64s)
                            )
                        b.async_buffer_load_lds_addr(
                            rsrc, lds, voffs[p_], soff, _DTL_DWORDS, coherency=coh
                        )

                    out.append(_one)
                return out

        pers = int(spec.trait.persistent_ctas)
        if _gett is not None and not pers:
            # GETT: the "tile origin" of an operand is its list of per-pass
            # per-lane byte offsets (row modes of this output tile + the
            # in-tile K modes); the K tile origin goes to soffset
            a_tile0 = _gett.tile_voffs(b, "A", block_m_off)
            b_tile0 = _gett.tile_voffs(b, "B", block_n_off)
        elif not pers:
            a_tile0 = b.to_sgpr_u32(b.add(batch_off_a, b.mul(block_m_off, K)))
            b_tile0 = b.to_sgpr_u32(b.add(batch_off_b, b.mul(_brow(block_n_off), K)))
        else:
            a_tile0 = b_tile0 = None
        # dtl_base_fold: per-operand base pointer at the output tile's origin
        _fold = {}
        if spec.trait.dtl_base_fold and not pers:
            _fold["A"] = b.global_ptr_add(A, b.to_sgpr_u32(b.mul(a_tile0, c2s)))
            _fold["B"] = b.global_ptr_add(Bp, b.to_sgpr_u32(b.mul(b_tile0, c2s)))
        c_big = b.const_i32(DTL_BUFFER_NUM_RECORDS)
        c_zero_rec = b.const_i32(0)
        tight = bool(spec.trait.dtl_tight_rsrc)
        if tight:
            # operand bytes (launch_shape_error keeps them below 2^31)
            a_bytes = b.to_sgpr_u32(b.mul(b.mul(M, K), c2s))
            b_bytes = b.to_sgpr_u32(b.mul(b.mul(_brow(N), K), c2s))
        else:
            a_bytes = b_bytes = None
        # per operand (A, B): exact bounds? (``dtl_tight_ragged_only``: only the
        # operand whose row dimension is padded)
        if tight and spec.trait.dtl_tight_ragged_only:
            _tight_ops = (bool(spec.trait.pad_m), bool(spec.trait.pad_n))
        else:
            _tight_ops = (tight, tight)
        # ``k_stagger_rc``: the current output tile's K start (elements, SGPR);
        # set by the tile loops before they emit that tile's prologue / K loop
        _ksr: List[Optional[Value]] = [None]

        def _k_wrap(k_ld: Value) -> Value:
            """``k_stagger_rc``: K offset of the (unwrapped) tile at ``k_ld``."""
            if _ksr[0] is None:
                return k_ld
            ke = b.add(k_ld, _ksr[0])
            return b.select(b.cmp_lt(ke, k_up), ke, b.sub(ke, b.sub(k_up, k_lo)))

        def _tight_rs(ptr: Value, tile0: Value, k_ld: Value, ok: Value, nbytes: Value):
            """``dtl_tight_rsrc``: (resource, soffset 0) whose base is this tile's
            origin and whose range ends at the operand's last byte."""
            org = b.to_sgpr_u32(b.mul(b.add(tile0, k_ld), c2s))
            left = b.smax(b.sub(nbytes, org), c_zero_rec)
            rs = b.buffer_rsrc(
                b.global_ptr_add(ptr, org),
                b.select(ok, left, c_zero_rec),
                word3=_w3,
            )
            return rs, c_zero_rec

        def _dtl_issues(
            stage: int,
            k_ld: Value,
            tile0s=None,
            valid=None,
            kup=None,
            which=(True, True),
            stage_b=None,
        ) -> List:
            """One closure per DTL instruction of tile ``k_ld`` into ``stage``.
            ``tile0s`` (persistent body): the (A, B) global element origin of the
            output tile; ``valid``: an extra wave-uniform predicate (else the
            zero-size resource); ``kup`` (stream-K item): the K bound of the
            item (else the kernel's)."""
            ta0, tb0 = tile0s if tile0s is not None else (a_tile0, b_tile0)
            ok = b.cmp_lt(k_ld, k_up if kup is None else kup)
            if valid is not None:
                ok = b.land(ok, valid)
            if _gett is not None:
                # one resource + soffset (= the K tile's mode offset) per
                # operand; tiles past K: zero-size resource, soffset 0
                g_out = []
                nrec_g = b.select(ok, c_big, c_zero_rec)
                for nm, ptr, voffs, wbase, coh in (
                    ("A", A, ta0, a_wbase, spec.trait.dtl_cache_a),
                    ("B", Bp, tb0, b_wbase, spec.trait.dtl_cache_b),
                ):
                    rs_g = b.buffer_rsrc(ptr, nrec_g)
                    so_g = b.to_sgpr_u32(
                        b.select(ok, _gett.ktile_off_bytes(b, nm, k_ld), c_zero_rec)
                    )
                    pb = _gett.pass_bytes(nm)
                    dw = _gett.lane_bytes(nm) // 4
                    for p in range(_gett.passes(nm)):

                        def _one_g(
                            p=p,
                            rs_g=rs_g,
                            so_g=so_g,
                            voffs=voffs,
                            wbase=wbase,
                            coh=coh,
                            pb=pb,
                            dw=dw,
                            nm=nm,
                        ):
                            lds = wbase[stage]
                            ps = _gett.pass_soff_bytes(nm, p)
                            if ps:
                                so_g = b.to_sgpr_u32(b.add(so_g, b.const_i32(ps)))
                            if p:
                                lds = b.smem_ptr_add(
                                    lds, b.zext(b.const_i32(p * pb), _I64s)
                                )
                            b.async_buffer_load_lds_addr(
                                rs_g, lds, voffs[p], so_g, dw, coherency=coh
                            )

                        g_out.append(_one_g)
                return g_out
            # out of range: num_records 0 AND soffset 0, so every lane is out of
            # range whether or not the range check includes soffset
            nrec = c_big if _clampk else b.select(ok, c_big, c_zero_rec)
            # the DMA's physical K: k_stagger_rc wrap, then k_stagger /
            # dtl_clamp_k / stagger_u (``_ka``); the range predicate stays logical
            k_ad = _ka(_k_wrap(k_ld))
            out = []
            _ops = (
                (
                    A,
                    _dtl_a_passes,
                    a_voff,
                    a_wbase,
                    ta0,
                    spec.trait.dtl_cache_a,
                    a_bytes,
                    stage,
                ),
                (
                    Bp,
                    _dtl_b_passes,
                    b_voff,
                    b_wbase,
                    tb0,
                    spec.trait.dtl_cache_b,
                    b_bytes,
                    stage if stage_b is None else stage_b,
                ),
            )
            for (
                (ptr, n_pass, voffs, wbase, tile0, coh, nbytes, stg_),
                _use,
                _rag,
            ) in zip(_ops, which, _tight_ops):
                if not _use:
                    continue
                h_rsrc = h_soff = None
                if spec.trait.dtl_hoist and tight and _rag:
                    h_rsrc, h_soff = _tight_rs(ptr, tile0, k_ad, ok, nbytes)
                elif spec.trait.dtl_hoist and (spec.trait.dtl_soff0 or tight):
                    # origin in the resource base, inline soffset 0
                    _fk = "A" if ptr is A else "B"
                    if _fold and tile0s is None:
                        org = b.to_sgpr_u32(b.mul(k_ad, c2s))
                        h_rsrc = b.buffer_rsrc(
                            b.global_ptr_add(_fold[_fk], org), nrec, word3=_w3
                        )
                    else:
                        org = b.to_sgpr_u32(b.mul(b.add(tile0, k_ad), c2s))
                        h_rsrc = b.buffer_rsrc(
                            b.global_ptr_add(ptr, org), nrec, word3=_w3
                        )
                    h_soff = c_zero_rec
                elif spec.trait.dtl_hoist:
                    h_rsrc = b.buffer_rsrc(ptr, nrec, word3=_w3)
                    h_soff = b.to_sgpr_u32(
                        b.select(ok, b.mul(b.add(tile0, k_ad), c2s), c_zero_rec)
                    )
                for p in range(n_pass):

                    def _one(
                        p=p,
                        ptr=ptr,
                        voffs=voffs,
                        wbase=wbase,
                        tile0=tile0,
                        coh=coh,
                        h_rsrc=h_rsrc,
                        h_soff=h_soff,
                        nbytes=nbytes,
                        stg_=stg_,
                    ):
                        if h_rsrc is not None:
                            rsrc, soff = h_rsrc, h_soff
                        elif tight:
                            rsrc, soff = _tight_rs(ptr, tile0, k_ad, ok, nbytes)
                        else:
                            rsrc = b.buffer_rsrc(ptr, nrec, word3=_w3)
                            soff = b.to_sgpr_u32(
                                b.select(ok, b.mul(b.add(tile0, k_ad), c2s), c_zero_rec)
                            )
                        if _vg:
                            isA = ptr is A
                            vo = (_vg_a_vo if isA else _vg_b_vo)[p]
                            so = b.to_sgpr_u32(b.mul(soff, c2s))
                            x0 = b.buffer_load_vN(rsrc, vo, so, F32, 4)
                            x1 = b.buffer_load_vN(rsrc, b.add(vo, _c16b), so, F32, 4)
                            _vg_pend.append(
                                (
                                    x0,
                                    x1,
                                    (A_stg if isA else B_stg)[stg_],
                                    (_vg_a_rc if isA else _vg_b_rc)[p],
                                    "A" if isA else "B",
                                )
                            )
                            return
                        lds = wbase[stg_]
                        if p:
                            lds = b.smem_ptr_add(
                                lds, b.zext(b.const_i32(p * _pass_bytes), _I64s)
                            )
                        b.async_buffer_load_lds_addr(
                            rsrc, lds, voffs[p], soff, _DTL_DWORDS, coherency=coh
                        )

                    out.append(_one)
            if spec.trait.asm_loop and all(which):
                # asm_loop: the inner operand (B) is released, loaded and
                # acquired first (the tables' order; the prologue follows it)
                out = out[_dtl_a_passes:] + out[:_dtl_a_passes]
            return out

        def _dtl_issues_deferred(stage: int, k_i: Value, tile0s=None, kup=None):
            """``dtl_addr_spread``: (prep steps, issue closures) of tile
            ``k_i + S * block_k`` into ``stage``. The address math is split
            into small steps (K offset + range predicate, then per operand the
            origin and the resource / soffset) that the caller pins to free
            MFMA slots; an issue closure runs any step not yet emitted."""
            ta0, tb0 = tile0s if tile0s is not None else (a_tile0, b_tile0)
            hold: Dict[str, Value] = {}
            # dtl_tight_rsrc: the origin goes into the base as with dtl_soff0;
            # num_records = the operand bytes left (``_tight_rs``) on the
            # operands ``_tight_ops`` marks
            soff0 = bool(spec.trait.dtl_soff0) or tight
            _tgt = {"A": (_tight_ops[0], a_bytes), "B": (_tight_ops[1], b_bytes)}
            anchor = bool(spec.trait.das_anchor)

            def _anc(v: Value) -> None:
                # das_anchor: a side-effecting read of v keeps its SALU here
                if anchor:
                    b.inline_asm("", "s", [v], None, sideeffect=True)

            def _s_k():
                # pinned (side-effecting no-op asm) so the SALU stays in
                # this slot instead of sinking to its first use
                k_ld = b.pin_sgpr(b.add(k_i, b.const_i32(S * block_k)))
                ok = b.cmp_lt(k_ld, k_up if kup is None else kup)
                hold["k"] = k_ld
                hold["ok"] = ok
                if _clampk:
                    hold["nrec"] = c_big
                else:
                    hold["nrec"] = b.pin_sgpr(b.select(ok, c_big, c_zero_rec))
                    _anc(hold["nrec"])
                # k_stagger / dtl_clamp_k: the DMA's K (the range predicate
                # stays logical)
                # (stagger_u folds in through _ka; k_stagger_rc through
                # ``_k_wrap``)
                hold["ka"] = (
                    b.pin_sgpr(_ka(_k_wrap(k_ld)))
                    if (_kst or _clampk or _su_off is not None or _ksr[0] is not None)
                    else k_ld
                )

            def _s_org(nm, tile0):
                def f():
                    if "k" not in hold:
                        _s_k()
                    if _fold and tile0s is None and soff0:
                        o = b.mul(hold["ka"], c2s)
                    else:
                        o = b.mul(b.add(tile0, hold["ka"]), c2s)
                    if soff0:
                        hold[nm + "org"] = b.to_sgpr_u32(o)
                        _anc(hold[nm + "org"])
                    else:
                        hold[nm + "soff"] = b.to_sgpr_u32(
                            b.select(hold["ok"], o, c_zero_rec)
                        )
                        _anc(hold[nm + "soff"])

                return f

            def _s_rsrc(nm, ptr):
                def f():
                    if "k" not in hold:
                        _s_k()
                    if soff0:
                        if nm + "org" not in hold:
                            _s_org(nm, ta0 if nm == "A" else tb0)()
                        _tg, _nb = _tgt[nm]
                        if _tg:
                            _left = b.smax(b.sub(_nb, hold[nm + "org"]), c_zero_rec)
                            _nr = b.select(hold["ok"], _left, c_zero_rec)
                        else:
                            _nr = hold["nrec"]
                        _bp = _fold[nm] if (_fold and tile0s is None) else ptr
                        _gp = b.global_ptr_add(_bp, hold[nm + "org"])
                        if anchor:
                            _anc(b.global_ptr_to_i64(_gp))
                        hold[nm + "rs"] = b.buffer_rsrc(
                            _gp,
                            _nr,
                            word3=_w3,
                        )
                        hold[nm + "soff"] = c_zero_rec
                    else:
                        hold[nm + "rs"] = b.buffer_rsrc(ptr, hold["nrec"], word3=_w3)

                return f

            prep = [
                _s_k,
                _s_org("A", ta0),
                _s_rsrc("A", A),
                _s_org("B", tb0),
                _s_rsrc("B", Bp),
            ]
            out = []
            for nm, ptr, n_pass, voffs, wbase, coh in (
                ("A", A, _dtl_a_passes, a_voff, a_wbase, spec.trait.dtl_cache_a),
                ("B", Bp, _dtl_b_passes, b_voff, b_wbase, spec.trait.dtl_cache_b),
            ):
                for p in range(n_pass):

                    def _one(p=p, nm=nm, ptr=ptr, voffs=voffs, wbase=wbase, coh=coh):
                        if nm + "soff" not in hold:
                            _s_org(nm, ta0 if nm == "A" else tb0)()
                        if nm + "rs" not in hold:
                            _s_rsrc(nm, ptr)()
                        lds = wbase[stage]
                        if p:
                            lds = b.smem_ptr_add(
                                lds, b.zext(b.const_i32(p * _pass_bytes), _I64s)
                            )
                        b.async_buffer_load_lds_addr(
                            hold[nm + "rs"],
                            lds,
                            voffs[p],
                            hold[nm + "soff"],
                            _DTL_DWORDS,
                            coherency=coh,
                        )

                    out.append(_one)
            return prep, out

        # ---- loop-invariant fragment addressing ----
        m_in_atom = b.mod(lane, b.const_i32(t.warp_tile_m))
        k_blk = b.div(lane, b.const_i32(t.warp_tile_m))
        n_in_atom = b.mod(lane, b.const_i32(t.warp_tile_n))
        # The warp offsets are wave-uniform: pin them in SGPRs so every ds_read
        # address is SGPR(stage, operand, warp) + VGPR(lane, kk) + imm(mi / ni)
        # (one short-lived v_add per stage / operand / K atom) instead of a
        # loop-invariant VGPR per (stage, operand, K atom, mi / ni), which with
        # S stages does not fit the 128-VGPR budget of a 16-wave block.
        warp_m_off = b.to_sgpr_u32(
            b.mul(warp_m_idx, b.const_i32(mfmas_m * t.warp_tile_m))
        )
        warp_n_off = b.to_sgpr_u32(
            b.mul(warp_n_idx, b.const_i32(mfmas_n * t.warp_tile_n))
        )
        k_blk_kbase = b.mul(k_blk, b.const_i32(a_per_lane * _kmul))
        a_row = [
            b.add(warp_m_off, b.add(b.const_i32(mi * t.warp_tile_m), m_in_atom))
            for mi in range(mfmas_m)
        ]
        b_row = [
            b.add(warp_n_off, b.add(b.const_i32(ni * t.warp_tile_n), n_in_atom))
            for ni in range(mfmas_n)
        ]
        # The swizzle key ((row >> R) mod 2^W) depends only on the in-atom row
        # when every warp / atom row origin is a multiple of 2^(R+W): key it on
        # the lane's in-atom row then (same value, but lane-only, so the
        # address splits into uniform + per-lane parts).
        _swz_span = 1 << (_swz_r + _swz_w)
        _lane_key = (t.warp_tile_m % _swz_span == 0) and (
            t.warp_tile_n % _swz_span == 0
        )
        a_key = [m_in_atom if _lane_key else a_row[mi] for mi in range(mfmas_m)]
        b_key = [n_in_atom if _lane_key else b_row[ni] for ni in range(mfmas_n)]

        # ds_read immediate offsets are 16-bit: for a stage allocation the pool
        # places at >= 64 KiB, LLVM merges the allocation base with the
        # mi / ni row constant into one unencodable constant and spends one
        # loop-invariant VGPR per (stage, K atom, mi / ni). Fold the 64 KiB
        # multiple of the base into an opaque per-lane row instead
        # (``row = opaque(row0 + shift) - shift + mi * wtm``: the same value,
        # but the remaining constant fits the immediate). The base offsets are
        # those of the smem-pool packer for these allocations (all live for the
        # whole loop -> placed back to back in definition order); a different
        # placement only costs registers, never correctness.
        _elem = type_bits(storage_dtype) // 8
        _row_bytes = block_k * _elem
        _offs, _o = {}, 0
        for _nm, _rows in (("A", block_m), ("B", block_n)):
            for _j in range(S if _nm == "A" else len(B_stg)):
                _offs[(_nm, _j)] = _o
                _o += _rows * _row_bytes

        def _shift_rows(nm: str, j: int) -> int:
            return ((_offs[(nm, j)] // 65536) * 65536) // _row_bytes

        # created here (dominating the prologue, the loop and the tail)
        _opq: Dict[Tuple[str, int], Value] = {}
        if _lane_key and not _rilv and _gett is None:
            for _nm in ("A", "B"):
                for _j in range(S if _nm == "A" else len(B_stg)):
                    _sh = _shift_rows(_nm, _j)
                    if _sh and (_nm, _sh) not in _opq:
                        w_off = warp_m_off if _nm == "A" else warp_n_off
                        in_atom = m_in_atom if _nm == "A" else n_in_atom
                        _opq[(_nm, _sh)] = b.optimization_barrier(
                            b.add(b.add(w_off, in_atom), b.const_i32(_sh))
                        )

        def _row_for(nm: str, j: int, idx: int) -> Value:
            sh = _shift_rows(nm, j)
            base_rows = a_row if nm == "A" else b_row
            if sh == 0 or not _lane_key:
                return base_rows[idx]
            wt = t.warp_tile_m if nm == "A" else t.warp_tile_n
            return b.add(_opq[(nm, sh)], b.const_i32(idx * wt - sh))

        # ---- lds_row_interleave fragment addressing: LDS alloc row (8 tile
        # rows + pad) R8 = (warp_off + mf * in_atom + idx) // 8, column
        # (idx % 8) * block_k + k: one lane-dependent row per (operand, 64 KiB
        # window); idx / kk only add immediates ----
        _rbase: Dict[Tuple[str, int], Value] = {}
        if _rilv:
            _rb = _rilv_cols * _elem
            _roffs, _ro = {}, 0
            for _nm, _rows in (("A", block_m // 8), ("B", block_n // 8)):
                for _j in range(S):
                    _roffs[(_nm, _j)] = _ro
                    _ro += _rows * _rb

            def _rsh(nm: str, j: int) -> int:
                return ((_roffs[(nm, j)] // 65536) * 65536) // _rb

            for _nm in ("A", "B"):
                _mf = mfmas_m if _nm == "A" else mfmas_n
                _w_off = warp_m_off if _nm == "A" else warp_n_off
                _in = m_in_atom if _nm == "A" else n_in_atom
                _lr8 = b.add(
                    b.div(_w_off, b.const_i32(8)), b.mul(_in, b.const_i32(_mf // 8))
                )
                for _j in range(S):
                    _sh = _rsh(_nm, _j)
                    if (_nm, _sh) not in _rbase:
                        _rbase[(_nm, _sh)] = (
                            b.optimization_barrier(b.add(_lr8, b.const_i32(_sh)))
                            if _sh
                            else _lr8
                        )

        # asm_loop_rawrd: one lane address per (operand, stage), created here
        # (dominating the prologue, the loop and the tail)
        _raw_base: Dict[Tuple[str, int], Value] = {}
        if _rilv and spec.trait.asm_loop_rawrd:
            for _nm in ("A", "B"):
                _mf = mfmas_m if _nm == "A" else mfmas_n
                _w_off = warp_m_off if _nm == "A" else warp_n_off
                _in = m_in_atom if _nm == "A" else n_in_atom
                _lr8r = b.add(
                    b.div(_w_off, b.const_i32(8)), b.mul(_in, b.const_i32(_mf // 8))
                )
                _lane_b = b.add(
                    b.mul(_lr8r, b.const_i32(_rilv_cols * _elem)),
                    b.mul(k_blk_kbase, b.const_i32(_elem)),
                )
                for _j in range(S):
                    _stg = A_stg[_j] if _nm == "A" else B_stg[_j]
                    _raw_base[(_nm, _j)] = b.add(
                        b.trunc(b.smem_addr_of(_stg), I32), _lane_b
                    )

        def _rilv_ld(nm: str, stage: int, idx: int, kk: int) -> Value:
            sh = _rsh(nm, stage)
            row = _rbase[(nm, sh)]
            if sh or idx // 8:
                row = b.add(row, b.const_i32(idx // 8 - sh))
            col = b.add(
                k_blk_kbase,
                b.const_i32((idx % 8) * block_k + kk * t.warp_tile_k * _kmul),
            )
            if nm == "A":
                return _frag_ld(
                    A_stg[stage], row, col, None, a_per_lane, "A", swz=False
                )
            return _frag_ld(B_stg[stage], row, col, None, b_per_lane, "B", swz=False)

        def _rilv_ld_raw(nm: str, stage: int, idx: int, kk: int) -> Value:
            """``asm_loop_rawrd``: the fragment as an inline-asm ds_read_b128
            (lane address in a VGPR, the (idx, kk) part as the immediate)."""
            key = (nm, stage)
            imm = (idx // 8) * _rilv_cols * _elem + (
                (idx % 8) * block_k + kk * t.warp_tile_k
            ) * _elem
            per = a_per_lane if nm == "A" else b_per_lane
            return b.inline_asm(
                f"ds_read_b128 $0, $1 offset:{imm}",
                "=v,v",
                [_raw_base[key]],
                VectorType(storage_dtype, per),
                sideeffect=True,
                result_name_hint="rawfr",
            )

        def _frags(stage: int, kk: int, stage_b=None) -> List[Value]:
            sb_ = stage if stage_b is None else stage_b
            if _gett is not None:
                return [
                    _gett.frag(b, A_stg[stage], "A", stage, mi, kk)
                    for mi in range(mfmas_m)
                ] + [
                    _gett.frag(b, B_stg[sb_], "B", sb_, ni, kk) for ni in range(mfmas_n)
                ]
            if _rilv:
                return [_rilv_ld("A", stage, mi, kk) for mi in range(mfmas_m)] + [
                    _rilv_ld("B", sb_, ni, kk) for ni in range(mfmas_n)
                ]
            col_base = b.add(k_blk_kbase, b.const_i32(kk * t.warp_tile_k * _kmul))
            fa = [
                _frag_ld(
                    A_stg[stage],
                    _row_for("A", stage, mi),
                    col_base,
                    a_key[mi],
                    a_per_lane,
                    "A",
                )
                for mi in range(mfmas_m)
            ]
            fb = [
                _frag_ld(
                    B_stg[sb_],
                    _row_for("B", sb_, ni),
                    col_base,
                    b_key[ni],
                    b_per_lane,
                    "B",
                )
                for ni in range(mfmas_n)
            ]
            return fa + fb

        def _frag_loaders(stage: int, kk: int, stage_b=None):
            """``mfma_interleave``: the fragments of ``_frags(stage, kk)`` as one
            load closure each, in MFMA use order (fa0, fb0..fb[n-1], fa1..);
            each fills its slot of the returned list (fa then fb, as _frags)."""
            out: List[Optional[Value]] = [None] * (mfmas_m + mfmas_n)
            cb: List[Value] = []

            def _col_base() -> Value:
                if not cb:
                    cb.append(
                        b.add(k_blk_kbase, b.const_i32(kk * t.warp_tile_k * _kmul))
                    )
                return cb[0]

            def _ld_a(mi: int) -> None:
                if _gett is not None:
                    out[mi] = _gett.frag(b, A_stg[stage], "A", stage, mi, kk)
                    return
                if _rilv and spec.trait.asm_loop_rawrd:
                    out[mi] = _rilv_ld_raw("A", stage, mi, kk)
                    return
                if _rilv:
                    out[mi] = _rilv_ld("A", stage, mi, kk)
                    return
                out[mi] = _frag_ld(
                    A_stg[stage],
                    _row_for("A", stage, mi),
                    _col_base(),
                    a_key[mi],
                    a_per_lane,
                    "A",
                )

            def _ld_b(ni: int) -> None:
                if _gett is not None:
                    sb_ = stage if stage_b is None else stage_b
                    out[mfmas_m + ni] = _gett.frag(b, B_stg[sb_], "B", sb_, ni, kk)
                    return
                if _rilv and spec.trait.asm_loop_rawrd:
                    out[mfmas_m + ni] = _rilv_ld_raw("B", stage, ni, kk)
                    return
                if _rilv:
                    out[mfmas_m + ni] = _rilv_ld("B", stage, ni, kk)
                    return
                out[mfmas_m + ni] = _frag_ld(
                    B_stg[stage if stage_b is None else stage_b],
                    _row_for("B", stage if stage_b is None else stage_b, ni),
                    _col_base(),
                    b_key[ni],
                    b_per_lane,
                    "B",
                )

            order = [lambda: _ld_a(0)]
            order += [lambda ni=ni: _ld_b(ni) for ni in range(mfmas_n)]
            order += [lambda mi=mi: _ld_a(mi) for mi in range(1, mfmas_m)]
            return out, order

        def _ilv_slots(loads: List, issues: List, lo: int, n: int) -> Dict[int, List]:
            """Slot (MFMA index) of every load / DTL issue: spread evenly over
            MFMAs [lo, n), loads first at a shared slot."""
            slot: Dict[int, List] = {}
            span = n - lo
            rp, ip = int(spec.trait.ilv_read_span), int(spec.trait.ilv_dtl_span)
            if rp or ip:
                # front-loaded spreads: the reads over the first rp % of the
                # cluster's MFMAs, the DTL issues over the first ip %
                for lst, pct in ((loads, rp or 100), (issues, ip or 100)):
                    sp = max(1, (span * pct) // 100)
                    for j, f in enumerate(lst):
                        slot.setdefault(lo + (j * sp) // len(lst), []).append(f)
                return slot
            if spec.trait.ilv_merge:
                # one schedule: rank every op by its fractional position in its
                # own list ((2j+1) / 2len, integer cross-multiplied; reads first
                # on a tie), then spread the ranks evenly over the slots
                nl, ni = len(loads), len(issues)
                ev = [((2 * j + 1) * max(ni, 1), 0, j) for j in range(nl)]
                ev += [((2 * j + 1) * max(nl, 1), 1, j) for j in range(ni)]
                ev.sort()
                for r, (_k, kind, j) in enumerate(ev):
                    f = loads[j] if kind == 0 else issues[j]
                    slot.setdefault(lo + (r * span) // len(ev), []).append(f)
                return slot
            for j, f in enumerate(loads):
                slot.setdefault(lo + (j * span) // len(loads), []).append(f)
            every = int(spec.trait.ilv_dtl_every)
            for j, f in enumerate(issues):
                if every:
                    at = lo + min(span - 1, j * every)
                else:
                    at = lo + (j * span) // len(issues)
                slot.setdefault(at, []).append(f)
            return slot

        def _ilv_cluster(
            acc: List[Value],
            fr: List[Value],
            loads: List,
            issues: List,
            pre: int = 0,
            at_barrier=None,
        ) -> List[Value]:
            """``mfma_interleave``: one K atom's MFMAs, each followed by its
            share of the next fragments' ds_reads and of the DTL issues (both
            spread evenly over the cluster, loads first at a shared slot), and
            a ``sched_barrier(0)`` that pins the order. With ``at_barrier``
            (``ilv_barrier_after`` = pre): MFMAs [0, pre) first, then
            ``at_barrier()`` (wait + barrier; returns the loads and issues),
            spread over MFMAs [pre, n)."""
            fa, fb = fr[:mfmas_m], fr[mfmas_m:]
            n = mfmas_m * mfmas_n
            slot = _ilv_slots(loads, issues, 0, n) if at_barrier is None else {}
            if _interwave:
                b.sched_barrier(0)
                b.s_setprio(1)
            for pos, flat in enumerate(_mo_ilv):
                mi, ni = divmod(flat, mfmas_n)
                if at_barrier is not None and pos == pre:
                    ld_, is_ = at_barrier()
                    slot = _ilv_slots(ld_, is_, pre, n)
                acc[flat] = _mma_ab(fa[mi], fb[ni], acc[flat], ni=ni)
                for f in slot.get(pos, ()):
                    f()
                b.sched_barrier(0)
            if _interwave:
                b.s_setprio(0)
                b.sched_barrier(0)
            return acc

        def _ilv_cluster_ev(
            acc: List[Value], fr: List[Value], events: List
        ) -> List[Value]:
            """``ilv_early_release``: one K atom's MFMAs with sync events. Each
            event ``(pos, fn)`` runs ``fn()`` in front of MFMA ``pos`` (after the
            MFMAs before it); ``fn`` returns (loads, issues), spread over the
            MFMAs [pos, next event's pos)."""
            fa, fb = fr[:mfmas_m], fr[mfmas_m:]
            n = mfmas_m * mfmas_n
            bounds = [p for p, _ in events] + [n]
            evs = {p: (fn, bounds[j + 1]) for j, (p, fn) in enumerate(events)}
            slot: Dict[int, List] = {}
            if _interwave:
                b.sched_barrier(0)
                b.s_setprio(1)
            for pos, flat in enumerate(_mo_ilv):
                mi, ni = divmod(flat, mfmas_n)
                if pos in evs:
                    fn, hi = evs[pos]
                    ld_, is_ = fn()
                    if ld_ or is_:
                        for k_, v_ in _ilv_slots(ld_, is_, pos, hi).items():
                            slot.setdefault(k_, []).extend(v_)
                    b.sched_barrier(0)
                acc[flat] = _mma_ab(fa[mi], fb[ni], acc[flat], ni=ni)
                for f in slot.get(pos, ()):
                    f()
                b.sched_barrier(0)
            if _interwave:
                b.s_setprio(0)
                b.sched_barrier(0)
            return acc

        n_mfma_tile = k_atoms * mfmas_m * mfmas_n

        def _cluster(
            acc: List[Value], fr: List[Value], kk: int, pos: Dict[int, List]
        ) -> List[Value]:
            fa, fb = fr[:mfmas_m], fr[mfmas_m:]
            if _interwave:
                b.sched_barrier(0)
                b.s_setprio(1)
            for mi in range(mfmas_m):
                if _SWZ and _sched_hints and not _interwave:
                    b.s_setprio(1)
                for ni in range(mfmas_n):
                    flat = mi * mfmas_n + ni
                    idx = kk * mfmas_m * mfmas_n + flat
                    for f in pos.get(idx, ()):
                        f()
                    acc[flat] = _mma_ab(fa[mi], fb[ni], acc[flat], ni=ni)
                if _SWZ and _sched_hints and not _interwave:
                    b.s_setprio(0)
                    b.sched_barrier(0)
            if _interwave:
                b.s_setprio(0)
                b.sched_barrier(0)
            return acc

        def _spread_pos(issues: List, first: int, span: int) -> Dict[int, List]:
            """Spread the issues evenly over MFMA indices [first, first+span)."""
            pos: Dict[int, List] = {}
            n = len(issues)
            for j, f in enumerate(issues):
                pos.setdefault(first + (j * span) // n, []).append(f)
            return pos

        _hb_rel, _hb_acq = hb_slots(spec) if spec.trait.hb_sched else (0, 0)
        # hb_skew: MFMA-slot shift of the reads / DTL issues of the loop copy
        # being emitted (0: the even copy, prologue and tail; 1: the odd copy)
        _hb_shift = [0]

        # the table the next _tile_asm emits (asm_loop_skew: the odd copy
        # while its main loop is emitted)
        _asm_cur = [_asm.table_events(spec.trait) if spec.trait.asm_loop else []]

        _serp = bool(spec.trait.mfma_serpentine)

        def _serp_n(mi: int, ni: int) -> int:
            """``mfma_serpentine``: the B fragment (column) of issue slot
            (mi, ni): odd A rows walk the B fragments backwards, so the B
            operand stays the same across every row change."""
            return mfmas_n - 1 - ni if _serp and mi % 2 else ni

        def _tile_asm(s, k_i, acc, fr, vm_wait, with_load, tile0s=None, kup=None):
            """``asm_loop``: one K tile as the event table's explicit order (see
            ``gemm_asm_loop``): MFMA x, then every event with K = x + 1, then a
            ``sched_barrier(0)``. Returns (acc, frags-for-next-tile)."""
            acc = list(acc)
            _asm_evs = _asm_cur[0]
            n = mfmas_m * mfmas_n
            out1, ord1 = _frag_loaders(s, 1)
            out0, ord0 = _frag_loaders((s + 1) % S, 0)

            def _by_op(order):
                # _frag_loaders order: A0, B0 .. B[n-1], A1 .. A[m-1]
                m_ = {("A", 0): order[0]}
                for ni in range(mfmas_n):
                    m_[("B", ni)] = order[1 + ni]
                for mi in range(1, mfmas_m):
                    m_[("A", mi)] = order[mfmas_n + mi]
                return m_

            rd = {"r1": _by_op(ord1), "r0": _by_op(ord0)}
            prep_steps: List = []
            if with_load and spec.trait.dtl_addr_spread:
                # the address math as small pinned steps in the
                # table's free slots; issues / steps reordered B-first
                prep_steps, issues = _dtl_issues_deferred(s, k_i, tile0s, kup)
                issues = issues[_dtl_a_passes:] + issues[:_dtl_a_passes]
                prep_steps = [prep_steps[0], prep_steps[3], prep_steps[4]] + prep_steps[
                    1:3
                ]
            else:
                issues = (
                    _dtl_issues(
                        s, b.add(k_i, b.const_i32(S * block_k)), tile0s, kup=kup
                    )
                    if with_load
                    else []
                )
            # hb_dtl_delay: every DTL issue that many slots later
            _dly = int(spec.trait.hb_dtl_delay)
            if _dly:
                _asm_evs = sorted(
                    [
                        (
                            ((min(_asm.N_TILE, e[0] + _dly),) + tuple(e[1:]))
                            if e[1] == "d"
                            else e
                        )
                        for e in _asm_evs
                    ],
                    key=lambda e: e[0],
                )
            # issues are B-first (asm_loop); the table names operand + index
            nb_ = _dtl_b_passes
            # loads of the PREVIOUS tile (or the prologue: B-first too) that
            # may stay in flight once operand set O has landed: the ones
            # issued after O's last load, in the table's and the prologue's
            # issue order (the smaller of the two)
            d_tbl = [(op, j) for (_k, kind, op, j) in _asm_evs if kind == "d"]
            d_pro = [("B", j) for j in range(nb_)] + [
                ("A", j) for j in range(_dtl_a_passes)
            ]

            def _after_last(order, ops):
                q = max(i for i, (o, _j) in enumerate(order) if o in ops)
                return len(order) - 1 - q

            pr = int(spec.trait.asm_loop_prio)
            n_iss = [0]

            def _run(ev):
                if ev[0] == "prep":
                    ev[1]()
                    return
                _k, kind, op, idx = ev
                if kind in ("r1", "r0"):
                    rd[kind][(op, idx)]()
                elif kind == "d":
                    if issues:
                        issues[idx if op == "B" else nb_ + idx]()
                        n_iss[0] += 1
                elif kind == "wl":
                    b.s_waitcnt(lgkmcnt=0)
                elif kind == "l":
                    b.s_waitcnt(lgkmcnt=idx)
                elif kind == "v":
                    allow = min(_after_last(d_tbl, op), _after_last(d_pro, op))
                    b.s_waitcnt(vmcnt=vm_wait + n_iss[0] + allow)
                elif kind == "bar":
                    if pr:
                        b.s_setprio(0)
                    b.s_barrier_bare()
                    if pr:
                        b.s_setprio(pr)
                elif kind == "p":
                    b.s_setprio(idx)

            at: Dict[int, List] = {}
            for ev in _asm_evs:
                at.setdefault(ev[0], []).append(ev)
            if prep_steps:
                # one address step per free slot (no event there) before the
                # first DTL issue, every other free slot first
                first_d = min(
                    (e[0] for e in _asm_evs if e[1] == "d"), default=n * k_atoms
                )
                free = [x for x in range(1, first_d) if x not in at]
                free = free[::2] + free[1::2]
                for j, f in enumerate(prep_steps):
                    if j < len(free):
                        at.setdefault(free[j], []).insert(0, ("prep", f))
            b.sched_barrier(0)
            if pr:
                b.s_setprio(pr)
            for ev in at.get(0, ()):
                _run(ev)
            b.sched_barrier(0)
            for kk in range(k_atoms):
                cur = fr if kk == 0 else out1
                fa, fb = cur[:mfmas_m], cur[mfmas_m:]
                for mi in range(mfmas_m):
                    for ni in range(mfmas_n):
                        flat = mi * mfmas_n + ni
                        nj = _serp_n(mi, ni)
                        af = mi * mfmas_n + nj
                        acc[af] = _mma_ab(fa[mi], fb[nj], acc[af], ni=nj)
                        for ev in at.get(kk * n + flat + 1, ()):
                            _run(ev)
                        b.sched_barrier(0)
            if pr:
                b.s_setprio(0)
                b.sched_barrier(0)
            return acc, out0

        def _tile_hb(s, k_i, acc, fr, vm_wait, with_load, tile0s=None, kup=None):
            """``hb_sched``: one K tile as a single MFMA stream with every
            ds_read, DTL issue, wait and barrier pinned after an MFMA slot
            (see TraitSpec.hb_sched). Returns (acc, frags-for-next-tile)."""
            if spec.trait.asm_loop:
                return _tile_asm(s, k_i, acc, fr, vm_wait, with_load, tile0s, kup)
            acc = list(acc)
            n = mfmas_m * mfmas_n
            ntot = k_atoms * n
            ops: Dict[int, List] = {}
            frs: List = [fr] + [None] * (k_atoms - 1)
            e_rd = max(1, int(spec.trait.hb_read_every))
            split = bool(spec.trait.hb_split_rel)
            last_rd = last_a = last_b = -1
            for kk in range(k_atoms - 1):
                out, order = _frag_loaders(s, kk + 1)
                frs[kk + 1] = out
                if split:
                    # all A fragments first, then all B
                    order = [order[0]] + order[1 + mfmas_n :] + order[1 : 1 + mfmas_n]
                for j, f in enumerate(order):
                    at = kk * n + min(n - 1, j * e_rd + _hb_shift[0])
                    ops.setdefault(at, []).append(f)
                    last_rd = max(last_rd, at)
                    if j < mfmas_m:
                        last_a = max(last_a, at)
                    else:
                        last_b = max(last_b, at)
            rel, acq = _hb_rel, _hb_acq
            if split:
                gap = max(1, int(spec.trait.hb_rel_gap))
                rel = min(ntot - 1, last_a + gap)
                rel_b = min(ntot - 1, last_b + gap)
            if not split and rel <= last_rd:
                raise ValueError(
                    f"hb_sched: release slot {rel} must follow the last fragment "
                    f"read (MFMA {last_rd})"
                )
            prep_steps: List = []
            if with_load and spec.trait.dtl_addr_spread:
                prep_steps, issues = _dtl_issues_deferred(s, k_i, tile0s, kup)
            else:
                issues = (
                    _dtl_issues(
                        s, b.add(k_i, b.const_i32(S * block_k)), tile0s, kup=kup
                    )
                    if with_load
                    else []
                )
            # DTL issue slots: from the release on, evenly or every hb_dtl_every
            every = int(spec.trait.hb_dtl_every)
            dly = int(spec.trait.hb_dtl_delay)
            span = ntot - rel - dly
            iss_at = []
            if split and issues:
                ev = every or 3
                na = _dtl_a_passes
                for j in range(len(issues)):
                    if j < na:
                        iss_at.append(min(ntot - 1, rel + j * ev))
                    else:
                        st = max(rel_b, iss_at[na - 1] + ev) if na else rel_b
                        iss_at.append(min(ntot - 1, st + (j - na) * ev))
            for j in range(0 if split else len(issues)):
                if every:
                    iss_at.append(rel + dly + min(span - 1, j * every))
                else:
                    iss_at.append(rel + dly + (j * span) // max(1, len(issues)))
            if _hb_shift[0]:
                _dsh = int(spec.trait.hb_skew_dtl) or _hb_shift[0]
                iss_at = [min(ntot - 1, a + _dsh) for a in iss_at]
            n_before_acq = sum(1 for a in iss_at if a < acq)
            acq_b = int(spec.trait.hb_acq_b_slot)
            nxt_holder: List = []

            def _release():
                b.s_waitcnt(lgkmcnt=0)
                b.s_barrier_bare()

            def _acquire():
                if acq_b < 0:
                    b.s_waitcnt(vmcnt=vm_wait + n_before_acq, lgkmcnt=0)
                    b.s_barrier_bare()
                    nx, ld = _frag_loaders((s + 1) % S, 0)
                    nxt_holder.append(nx)
                    for j, f in enumerate(ld):
                        ops.setdefault(
                            min(ntot - 1, acq + 1 + j * e_rd + _hb_shift[0]), []
                        ).append(f)
                    return
                # split: A now (the previous tile's B loads may stay in flight)
                b.s_waitcnt(vmcnt=vm_wait + _dtl_b_passes + n_before_acq, lgkmcnt=0)
                b.s_barrier_bare()
                nx, ld = _frag_loaders((s + 1) % S, 0)
                nxt_holder.append(nx)
                ld_a = [ld[0]] + ld[1 + mfmas_n :]
                ld_b = ld[1 : 1 + mfmas_n]
                for j, f in enumerate(ld_a):
                    ops.setdefault(min(ntot - 1, acq + 1 + j * e_rd), []).append(f)
                n_before_b = sum(1 for a in iss_at if a < acq_b)

                def _acquire_b():
                    b.s_waitcnt(vmcnt=vm_wait + n_before_b)
                    b.s_barrier_bare()
                    for j, f in enumerate(ld_b):
                        ops.setdefault(min(ntot - 1, acq_b + 1 + j * e_rd), []).append(
                            f
                        )

                pre.setdefault(acq_b, []).insert(0, _acquire_b)

            # per slot, in order: release, acquire, DTL issues, then reads
            pre: Dict[int, List] = {}
            if issues:
                pre.setdefault(rel, []).append(_release)
                if split and rel_b != rel:
                    pre.setdefault(rel_b, []).append(_release)
            pre.setdefault(acq, []).append(_acquire)
            for j, a in enumerate(iss_at):
                pre.setdefault(a, []).append(issues[j])
            if prep_steps:
                # dtl_addr_spread: one address step per free MFMA slot (no
                # fragment read, wait or DTL there) before the first DTL
                first_iss = min(iss_at) if iss_at else ntot
                free = [
                    x
                    for x in range(1 + _hb_shift[0], first_iss)
                    if x not in ops and x not in pre
                ]
                free = free[::2] + free[1::2]  # every other free slot first
                if spec.trait.das_late and len(prep_steps) == 5:
                    # das_late: k + A steps after the last read, B steps
                    # between A's first DTL and the acquire
                    na_ = _dtl_a_passes

                    def _fr(lo_, hi_):
                        f_ = [
                            x for x in range(lo_, hi_) if x not in ops and x not in pre
                        ]
                        return f_[::2] + f_[1::2]

                    fa_ = _fr(last_rd + 1, first_iss)
                    fb_end = iss_at[na_] if len(iss_at) > na_ else ntot
                    fb_ = _fr(first_iss + 1, min(fb_end, acq))
                    if len(fa_) >= 3 and len(fb_) >= 2:
                        free = sorted(fa_[:3]) + sorted(fb_[:2])
                for j, f in enumerate(prep_steps):
                    if j < len(free):
                        pre.setdefault(free[j], []).append(f)
            if _interwave:
                b.sched_barrier(0)
                b.s_setprio(1)
            for kk in range(k_atoms):
                fa, fb = frs[kk][:mfmas_m], frs[kk][mfmas_m:]
                # MFMA issue order: slot x = kk * n + pos runs accumulator flat
                for pos, flat in enumerate(_mo_perm[kk % 2]):
                    mi, ni = divmod(flat, mfmas_n)
                    x = kk * n + pos
                    acc[flat] = _mma_ab(fa[mi], fb[ni], acc[flat], ni=ni)
                    for f in pre.get(x, ()):
                        f()
                    for f in ops.get(x, ()):
                        f()
                    b.sched_barrier(0)
            if _interwave:
                b.s_setprio(0)
                b.sched_barrier(0)
            if spec.trait.hb_tile_lgkm0:
                # hb_tile_lgkm0: the next tile's first-atom fragments (read
                # after the acquire) have landed; LLVM then needs no per-MFMA
                # lgkmcnt(N) at the next tile's start
                b.s_waitcnt(lgkmcnt=0)
                b.sched_barrier(0)
            return acc, nxt_holder[0]

        _sia = sia_table(spec) if spec.trait.sia_sched else None

        def _tile_sia(s, k_i, acc, fr, vm_wait, with_load, tile0s=None, kup=None):
            """``sia_sched``: one K tile from the ``sia_table`` slot table.
            Returns (acc, frags-for-next-tile)."""
            tb = _sia
            acc = list(acc)
            n = mfmas_m * mfmas_n
            out1, _ = _frag_loaders(s, 1)
            out0, _ = _frag_loaders((s + 1) % S, 0)

            def _rd(stage, kk, op, idx, out):
                def _f():
                    if op == "A":
                        out[idx] = _rilv_ld("A", stage, idx, kk)
                    else:
                        out[mfmas_m + idx] = _rilv_ld("B", stage, idx, kk)

                return _f

            slots: Dict[int, List] = {}

            def _at(slot, f):
                slots.setdefault(slot, []).append(f)

            if tb["lgkm"] >= 0:
                _at(tb["lgkm"], lambda: b.s_waitcnt(lgkmcnt=0))

            def _release():
                b.s_waitcnt(lgkmcnt=0)
                b.s_barrier_bare()

            for sl, _ops in tb["rel"]:
                if with_load:
                    _at(sl, _release)
            # DTL issues: closures per operand, in table order
            iss_by: Dict[str, List] = {"A": [], "B": []}
            if with_load:
                k_ld = b.add(k_i, b.const_i32(S * block_k))
                for op in tb["order"]:
                    iss_by[op] = _dtl_issues(
                        s, k_ld, tile0s, kup=kup, which=(op == "A", op == "B")
                    )
            # loads still allowed in flight at each acquire: older tiles
            # (vm_wait) + sia_acq_vmcnt
            for (sl, _ops), vmc in zip(tb["acq"], sia_acq_vmcnt(tb, with_load)):

                def _acquire(vmc=vmc):
                    if tb["merged"]:
                        # also the release: this tile's reads are done
                        b.s_waitcnt(vmcnt=vm_wait + vmc, lgkmcnt=0)
                    else:
                        b.s_waitcnt(vmcnt=vm_wait + vmc)
                    b.s_barrier_bare()

                _at(sl, _acquire)
            if with_load:
                for sl, op, p in tb["dtl"]:
                    _at(sl, iss_by[op][p])
            for sl, op, idx in tb["rd1"]:
                _at(sl, _rd(s, 1, op, idx, out1))
            for sl, op, idx in tb["rd0"]:
                _at(sl, _rd((s + 1) % S, 0, op, idx, out0))
            frs = [fr, out1]
            if _interwave:
                b.sched_barrier(0)
                b.s_setprio(1)
            for kk in range(2):
                fa, fb = frs[kk][:mfmas_m], frs[kk][mfmas_m:]
                # MFMA issue order: slot x = kk * n + pos runs accumulator flat
                for pos, flat in enumerate(_mo_perm[kk % 2]):
                    mi, ni = divmod(flat, mfmas_n)
                    x = kk * n + pos
                    acc[flat] = _mma_ab(fa[mi], fb[ni], acc[flat], ni=ni)
                    for f in slots.get(x, ()):
                        f()
                    b.sched_barrier(0)
            if _interwave:
                b.s_setprio(0)
                b.sched_barrier(0)
            return acc, out0

        def _tile(
            s: int,
            k_i: Value,
            acc: List[Value],
            fr,
            vm_wait: int,
            with_load: bool,
            tile0s=None,
            kup=None,
        ):
            """One K tile from stage s. Returns (acc, frags-for-next-tile)."""
            if spec.trait.sia_sched:
                return _tile_sia(s, k_i, acc, fr, vm_wait, with_load, tile0s, kup)
            if spec.trait.hb_sched:
                return _tile_hb(s, k_i, acc, fr, vm_wait, with_load, tile0s, kup)
            acc = list(acc)
            if iglp is not None:
                b.iglp_opt(iglp)
            if instream and not fprf:
                # barrier inside the MFMA stream, no extra fragment registers
                for kk in range(k_atoms):
                    fr_k = _frags(s, kk)
                    pos = {}
                    if kk + 1 == k_atoms:
                        b.s_waitcnt(vmcnt=vm_wait, lgkmcnt=0)
                        b.s_barrier_bare()
                        issues = (
                            _dtl_issues(
                                s, b.add(k_i, b.const_i32(S * block_k)), tile0s, kup=kup
                            )
                            if with_load
                            else []
                        )
                        if spread and issues:
                            pos = _spread_pos(
                                issues, kk * mfmas_m * mfmas_n, mfmas_m * mfmas_n
                            )
                        else:
                            for f in issues:
                                f()
                    acc = _cluster(acc, fr_k, kk, pos)
                return acc, None
            if not fprf:
                b.s_waitcnt(vmcnt=vm_wait, lgkmcnt=0)
                b.s_barrier_bare()
                issues = (
                    _dtl_issues(
                        (s + S - 1) % S,
                        b.add(k_i, b.const_i32((S - 1) * block_k)),
                        tile0s,
                        kup=kup,
                    )
                    if with_load
                    else []
                )
                if spread and issues:
                    pos = _spread_pos(issues, 0, n_mfma_tile)
                else:
                    for f in issues:
                        f()
                    pos = {}
                for kk in range(k_atoms):
                    acc = _cluster(acc, _frags(s, kk), kk, pos)
                return acc, None
            efq = int(spec.trait.ilv_early_free)
            if ilv and efq:
                # ilv_early_free: the stage-free barrier sits inside the
                # second-to-last atom (after its packed reads of the last
                # atom's fragments); the DTL issues of tile i+S follow it.
                n_at = mfmas_m * mfmas_n
                n_iss = 0
                late: List = []
                for kk in range(k_atoms):
                    pre = int(spec.trait.ilv_barrier_after)
                    if kk + 2 < k_atoms:
                        nxt, loads = _frag_loaders(s, kk + 1)
                        acc = _ilv_cluster(acc, fr, loads, [])
                        fr = nxt
                        continue
                    if kk + 2 == k_atoms:
                        nxt, loads = _frag_loaders(s, kk + 1)
                        slot: Dict[int, List] = {}
                        rspan = max(1, efq - 4)
                        for j, f in enumerate(loads):
                            slot.setdefault((j * rspan) // len(loads), []).append(f)
                        iss_box: List = []
                        late_box: List = []

                        def _free(k_i=k_i, s=s, iss_box=iss_box):
                            b.s_waitcnt(lgkmcnt=0)
                            b.s_barrier_bare()
                            iss = (
                                _dtl_issues(
                                    s,
                                    b.add(k_i, b.const_i32(S * block_k)),
                                    tile0s,
                                    kup=kup,
                                )
                                if with_load
                                else []
                            )
                            if spec.trait.ilv_ef_spread and iss:
                                # global slot of issue j over [efq, 2 n): the
                                # ones past this atom go to the last atom
                                tot = 2 * n_at - efq
                                pos = [
                                    efq + (j * tot) // len(iss) for j in range(len(iss))
                                ]
                                now = [f for f, q in zip(iss, pos) if q < n_at]
                                iss_box.extend(now)
                                late_box.extend(
                                    f for f, q in zip(iss, pos) if q >= n_at
                                )
                                return now
                            iss_box.extend(iss)
                            return iss

                        fa, fb = fr[:mfmas_m], fr[mfmas_m:]
                        if _interwave:
                            b.sched_barrier(0)
                            b.s_setprio(1)
                        for mi in range(mfmas_m):
                            for ni in range(mfmas_n):
                                flat = mi * mfmas_n + ni
                                if flat == efq:
                                    iss = _free()
                                    if spec.trait.ilv_ef_spread:
                                        _tot = 2 * n_at - efq
                                        for j, f in enumerate(iss):
                                            slot.setdefault(
                                                efq
                                                + (j * _tot)
                                                // (len(iss) + len(late_box)),
                                                [],
                                            ).append(f)
                                    else:
                                        for j, f in enumerate(iss):
                                            slot.setdefault(
                                                efq + (j * (n_at - efq)) // len(iss), []
                                            ).append(f)
                                acc[flat] = _mma_ab(fa[mi], fb[ni], acc[flat], ni=ni)
                                for f in slot.get(flat, ()):
                                    f()
                                b.sched_barrier(0)
                        if _interwave:
                            b.s_setprio(0)
                            b.sched_barrier(0)
                        n_iss = len(iss_box)
                        late = list(late_box)
                        fr = nxt
                        continue
                    # last atom: wait for tile i+1 (the n_iss loads of tile
                    # i+S stay in flight) + barrier, then the next tile's
                    # first fragments
                    held: List = []

                    def _at_barrier2(s=s, held=held, n_iss=n_iss, late=late):
                        b.s_waitcnt(vmcnt=vm_wait + n_iss, lgkmcnt=0)
                        b.s_barrier_bare()
                        nx, ld = _frag_loaders((s + 1) % S, 0)
                        held.append(nx)
                        return ld, list(late)

                    if pre:
                        acc = _ilv_cluster(acc, fr, [], [], pre, _at_barrier2)
                    else:
                        ld, lt = _at_barrier2()
                        acc = _ilv_cluster(acc, fr, ld, lt)
                    fr = held[0]
                return acc, fr
            swq = int(spec.trait.ilv_split_wait)
            if ilv and swq:
                # ilv_split_wait: the last atom waits for tile i+1's A loads,
                # reads the A fragments, then (slot swq) waits for its B loads
                n_at = mfmas_m * mfmas_n
                pre = int(spec.trait.ilv_barrier_after)
                for kk in range(k_atoms):
                    if kk + 1 < k_atoms:
                        nxt, loads = _frag_loaders(s, kk + 1)
                        acc = _ilv_cluster(acc, fr, loads, [])
                        fr = nxt
                        continue
                    nxt, _order = _frag_loaders((s + 1) % S, 0)
                    # _order = [fa0, fb0..fb(n-1), fa1..fa(m-1)]
                    lda = [_order[0]] + _order[1 + mfmas_n :]
                    ldb = _order[1 : 1 + mfmas_n]
                    fa, fb = fr[:mfmas_m], fr[mfmas_m:]
                    slot: Dict[int, List] = {}
                    n_before = [0]
                    if _interwave:
                        b.sched_barrier(0)
                        b.s_setprio(1)
                    for mi in range(mfmas_m):
                        for ni in range(mfmas_n):
                            flat = mi * mfmas_n + ni
                            if flat == pre:
                                b.s_waitcnt(vmcnt=vm_wait + _dtl_b_passes, lgkmcnt=0)
                                b.s_barrier_bare()
                                iss = (
                                    _dtl_issues(
                                        s,
                                        b.add(k_i, b.const_i32(S * block_k)),
                                        tile0s,
                                        kup=kup,
                                    )
                                    if with_load
                                    else []
                                )
                                for j, f in enumerate(lda):
                                    slot.setdefault(
                                        pre + (j * (swq - pre)) // len(lda), []
                                    ).append(f)
                                for j, f in enumerate(ldb):
                                    slot.setdefault(
                                        swq + (j * (n_at - swq)) // len(ldb), []
                                    ).append(f)
                                for j, f in enumerate(iss):
                                    at = pre + (j * (n_at - pre)) // len(iss)
                                    if at < swq:
                                        n_before[0] += 1
                                    slot.setdefault(at, []).append(f)
                            if flat == swq:
                                b.s_waitcnt(vmcnt=vm_wait + n_before[0])
                                b.s_barrier_bare()
                            acc[flat] = _mma_ab(fa[mi], fb[ni], acc[flat], ni=ni)
                            for f in slot.get(flat, ()):
                                f()
                            b.sched_barrier(0)
                    if _interwave:
                        b.s_setprio(0)
                        b.sched_barrier(0)
                    fr = nxt
                return acc, fr
            if ilv:
                # frag-prefetch order with every ds_read / DTL issue pinned
                # between two MFMAs (Tensile-style interleave)
                for kk in range(k_atoms):
                    issues = []
                    pre = int(spec.trait.ilv_barrier_after)
                    er = int(spec.trait.ilv_early_release)
                    n_at = mfmas_m * mfmas_n
                    n_ld = mfmas_m + mfmas_n
                    if er == 3 and kk + 2 == k_atoms:
                        # mode 3: as mode 2, but the DTL issues spread over
                        # [release, end of this atom) + [0, acquire) of the
                        # last atom (about one DTL per 5 MFMAs)
                        nxt, loads = _frag_loaders(s, kk + 1)
                        rel = int(spec.trait.ilv_release_at) or min(
                            n_at - 1, n_ld + n_ld // 2
                        )
                        aq3 = int(spec.trait.ilv_acquire_at) or max(
                            1, n_at - n_ld - n_ld // 2
                        )
                        carry: List = []

                        def _rel3(k_i=k_i, s=s, rel=rel, aq3=aq3):
                            b.s_waitcnt(lgkmcnt=0)
                            b.s_barrier_bare()
                            iss = (
                                _dtl_issues(
                                    s,
                                    b.add(k_i, b.const_i32(S * block_k)),
                                    tile0s,
                                    kup=kup,
                                )
                                if with_load
                                else []
                            )
                            n1 = (
                                len(iss) * (n_at - rel) + (n_at - rel + aq3) // 2
                            ) // (n_at - rel + aq3)
                            carry.extend(iss[n1:])
                            return [], iss[:n1]

                        acc = _ilv_cluster_ev(
                            acc, fr, [(0, lambda ld=loads: (ld, [])), (rel, _rel3)]
                        )
                        fr = nxt
                        er3_carry = carry
                        continue
                    if er == 3 and kk + 1 == k_atoms:
                        held_e3: List = []
                        aq3 = int(spec.trait.ilv_acquire_at) or max(
                            1, n_at - n_ld - n_ld // 2
                        )
                        vm_aq3 = vm_wait + (L if with_load else 0)

                        def _acq3(s=s, vm_aq3=vm_aq3):
                            b.s_waitcnt(vmcnt=vm_aq3)
                            b.s_barrier_bare()
                            nx, ld = _frag_loaders((s + 1) % S, 0)
                            held_e3.append(nx)
                            return ld, []

                        cr = list(er3_carry)
                        acc = _ilv_cluster_ev(
                            acc, fr, [(0, lambda cr=cr: ([], cr)), (aq3, _acq3)]
                        )
                        fr = held_e3[0]
                        continue
                    if er == 2 and kk + 2 == k_atoms:
                        # early release: the last atom's fragments are read
                        # first, then the release point inside THIS atom
                        nxt, loads = _frag_loaders(s, kk + 1)
                        rel = int(spec.trait.ilv_release_at) or min(
                            n_at - 1, n_ld + n_ld // 2
                        )

                        def _rel2(k_i=k_i, s=s):
                            b.s_waitcnt(lgkmcnt=0)
                            b.s_barrier_bare()
                            iss = (
                                _dtl_issues(
                                    s,
                                    b.add(k_i, b.const_i32(S * block_k)),
                                    tile0s,
                                    kup=kup,
                                )
                                if with_load
                                else []
                            )
                            return [], iss

                        acc = _ilv_cluster_ev(
                            acc, fr, [(0, lambda ld=loads: (ld, [])), (rel, _rel2)]
                        )
                        fr = nxt
                        continue
                    if er == 2 and kk + 1 == k_atoms:
                        held_e2: List = []
                        aq2 = int(spec.trait.ilv_acquire_at) or min(
                            n_at - 1, n_ld + n_ld // 2
                        )
                        vm_aq2 = vm_wait + (L if with_load else 0)

                        def _acq2(s=s, vm_aq2=vm_aq2):
                            b.s_waitcnt(vmcnt=vm_aq2)
                            b.s_barrier_bare()
                            nx, ld = _frag_loaders((s + 1) % S, 0)
                            held_e2.append(nx)
                            return ld, []

                        acc = _ilv_cluster_ev(acc, fr, [(aq2, _acq2)])
                        fr = held_e2[0]
                        continue
                    if kk + 1 < k_atoms:
                        nxt, loads = _frag_loaders(s, kk + 1)
                    elif er == 1:
                        # release (stage s fully read) -> DTL tile i+S into s
                        # -> acquire (tile i+1 landed) -> next tile's frags
                        held_er: List = []
                        aq = int(spec.trait.ilv_acquire_at) or max(
                            pre + 1, n_at - 2 * (mfmas_m + mfmas_n)
                        )
                        vm_aq = vm_wait + (L if with_load else 0)

                        def _release(k_i=k_i, s=s):
                            b.s_waitcnt(lgkmcnt=0)
                            b.s_barrier_bare()
                            iss = (
                                _dtl_issues(
                                    s,
                                    b.add(k_i, b.const_i32(S * block_k)),
                                    tile0s,
                                    kup=kup,
                                )
                                if with_load
                                else []
                            )
                            return [], iss

                        def _acquire(s=s, vm_aq=vm_aq):
                            b.s_waitcnt(vmcnt=vm_aq)
                            b.s_barrier_bare()
                            nx, ld = _frag_loaders((s + 1) % S, 0)
                            held_er.append(nx)
                            return ld, []

                        acc = _ilv_cluster_ev(
                            acc, fr, [(pre, _release), (aq, _acquire)]
                        )
                        fr = held_er[0]
                        continue
                    elif pre:
                        held: List = []

                        def _at_barrier(k_i=k_i, s=s):
                            b.s_waitcnt(vmcnt=vm_wait, lgkmcnt=0)
                            b.s_barrier_bare()
                            iss = (
                                _dtl_issues(
                                    s,
                                    b.add(k_i, b.const_i32(S * block_k)),
                                    tile0s,
                                    kup=kup,
                                )
                                if with_load
                                else []
                            )
                            nx, ld = _frag_loaders((s + 1) % S, 0)
                            held.append(nx)
                            return ld, iss

                        acc = _ilv_cluster(acc, fr, [], [], pre, _at_barrier)
                        fr = held[0]
                        continue
                    else:
                        b.s_waitcnt(vmcnt=vm_wait, lgkmcnt=0)
                        b.s_barrier_bare()
                        if with_load:
                            issues = _dtl_issues(
                                s, b.add(k_i, b.const_i32(S * block_k)), tile0s, kup=kup
                            )
                        nxt, loads = _frag_loaders((s + 1) % S, 0)
                    acc = _ilv_cluster(acc, fr, loads, issues)
                    fr = nxt
                return acc, fr
            # frag-prefetch order
            for kk in range(k_atoms):
                pos = {}
                if kk + 1 < k_atoms:
                    nxt = _frags(s, kk + 1)
                else:
                    b.s_waitcnt(vmcnt=vm_wait, lgkmcnt=0)
                    b.s_barrier_bare()
                    issues = (
                        _dtl_issues(
                            s, b.add(k_i, b.const_i32(S * block_k)), tile0s, kup=kup
                        )
                        if with_load
                        else []
                    )
                    if spread and issues:
                        pos = _spread_pos(
                            issues, kk * mfmas_m * mfmas_n, mfmas_m * mfmas_n
                        )
                    else:
                        for f in issues:
                            f()
                    nxt = _frags((s + 1) % S, 0)
                acc = _cluster(acc, fr, kk, pos)
                fr = nxt
            return acc, fr

        pp = int(spec.trait.pingpong)
        if pp:
            _nw = t.warp_m * t.warp_n * t.warp_k
            _wsg = b.to_sgpr_u32(warp_id)
            pp_grp1 = b.cmp_ge(_wsg, b.const_i32(_nw // 2))
            pp_grp0 = b.cmp_lt(_wsg, b.const_i32(_nw // 2))
        pp_late = bool(spec.trait.pp_wait_late)
        _ppg: List[Optional[int]] = [None]  # pingpong 4: the group being emitted

        def _pp_barrier() -> None:
            b.sched_barrier(0)
            b.s_barrier_bare()
            b.sched_barrier(0)

        def _tile_pp4(s, k_i, acc, with_load, tile0s=None, kup=None):
            """``pingpong = 4 / 5`` (S = 2): group 0 issues tile i+1 at the start
            of its memory phase and waits for it after its compute phase; group 1
            issues tile i+2 at the start of its compute phase (4) / spread over
            it (5) and waits for it at the end of its next memory phase. Both
            windows span about two phases (see TraitSpec)."""
            acc = list(acc)
            g = _ppg[0]
            mix = int(spec.trait.pp_mix_reads)
            if g == 0 and with_load and mix:
                # pp_mix_reads: one DTL issue after every `mix` fragment reads
                iss = _dtl_issues((s + 1) % S, b.add(k_i, c_block_k), tile0s, kup=kup)
                outs, lds_ = [], []
                for kk in range(k_atoms):
                    o_, ld = _frag_loaders(s, kk)
                    outs.append(o_)
                    lds_ += ld
                iss[0]()
                b.sched_barrier(0)
                j = 1
                for r, f in enumerate(lds_):
                    f()
                    if (r + 1) % mix == 0 and j < len(iss):
                        b.sched_barrier(0)
                        iss[j]()
                        b.sched_barrier(0)
                        j += 1
                for f in iss[j:]:
                    f()
                frs = outs
            else:
                if g == 0 and with_load:
                    for f in _dtl_issues(
                        (s + 1) % S, b.add(k_i, c_block_k), tile0s, kup=kup
                    ):
                        f()
                    b.sched_barrier(0)
                frs = [_frags(s, kk) for kk in range(k_atoms)]
            b.s_waitcnt(lgkmcnt=0)
            nx2 = []
            if g == 1 and with_load:
                # tile i+2 overwrites tile i's stage: only after the barrier that
                # ends this memory phase have ALL waves (the other group-1 waves
                # included) finished reading tile i
                nx2 = _dtl_issues(
                    s, b.add(k_i, b.const_i32(2 * block_k)), tile0s, kup=kup
                )
            if g == 1:
                # tile i+1 (issued in the previous compute phase) has landed
                b.s_waitcnt(vmcnt=0)
            _pp_barrier()
            if g == 1 and pp == 4:
                for f in nx2:
                    f()
                b.sched_barrier(0)
            if _interwave:
                b.s_setprio(1)
            n_mm = k_atoms * mfmas_m * mfmas_n
            at = {}
            if pp == 5:
                for j, f in enumerate(nx2):
                    at.setdefault(((2 * j + 1) * n_mm) // (2 * len(nx2)), []).append(f)
            for kk in range(k_atoms):
                fa, fb = frs[kk][:mfmas_m], frs[kk][mfmas_m:]
                for mi in range(mfmas_m):
                    for ni in range(mfmas_n):
                        flat = mi * mfmas_n + ni
                        acc[flat] = _mma_ab(fa[mi], fb[ni], acc[flat], ni=ni)
                        fs = at.get(kk * mfmas_m * mfmas_n + flat, ())
                        for f in fs:
                            f()
                        if fs:
                            b.sched_barrier(0)
            if _interwave:
                b.sched_barrier(0)
                b.s_setprio(0)
            b.sched_barrier(0)
            if g == 0:
                b.s_waitcnt(vmcnt=0)
            _pp_barrier()
            return acc, None

        def _tile_pp6(t_, k_i, acc, tile0s=None, kup=None):
            """``pingpong = 6``: A ring of 2, B ring of 3 (static stages t % 2 /
            t % 3 of tile i, loop unrolled by 6). Group 0: A(i+1) at the start of
            its memory phase, B(i+2) spread over its compute phase, then
            ``vmcnt(LB)`` (A(i+1) and B(i+1) landed). Group 1: B(i+2) at the start
            of its memory phase, ``vmcnt(LB)`` at its end (A(i+1), B(i+1) landed),
            A(i+2) spread over its compute phase. Every DMA targets a stage whose
            last reads ended before the barrier that opened the issuing phase;
            both groups' DMAs split evenly over the two phases."""
            acc = list(acc)
            g = _ppg[0]
            sa, sb = t_ % 2, t_ % 3
            LB = _dtl_b_passes
            k1 = b.add(k_i, c_block_k)
            k2 = b.add(k_i, b.const_i32(2 * block_k))
            mix = int(spec.trait.pp_mix_reads)

            def _mem(burst):
                """the memory phase's DMA burst (to a stage nobody reads now) and
                the fragment reads of tile i; pp_mix_reads interleaves them"""
                if not mix:
                    for f in burst:
                        f()
                    b.sched_barrier(0)
                    return [_frags(sa, kk, sb) for kk in range(k_atoms)]
                outs, lds_ = [], []
                for kk in range(k_atoms):
                    o_, ld = _frag_loaders(sa, kk, sb)
                    outs.append(o_)
                    lds_ += ld
                j = 0
                if burst:
                    burst[0]()
                    b.sched_barrier(0)
                    j = 1
                for r, f in enumerate(lds_):
                    f()
                    if (r + 1) % mix == 0 and j < len(burst):
                        b.sched_barrier(0)
                        burst[j]()
                        b.sched_barrier(0)
                        j += 1
                for f in burst[j:]:
                    f()
                return outs

            if g == 0:
                frs = _mem(
                    _dtl_issues((t_ + 1) % 2, k1, tile0s, kup=kup, which=(True, False))
                )
                b.s_waitcnt(lgkmcnt=0)
                spread = _dtl_issues(
                    0, k2, tile0s, kup=kup, which=(False, True), stage_b=(t_ + 2) % 3
                )
            else:
                frs = _mem(
                    _dtl_issues(
                        0,
                        k2,
                        tile0s,
                        kup=kup,
                        which=(False, True),
                        stage_b=(t_ + 2) % 3,
                    )
                )
                b.s_waitcnt(lgkmcnt=0)
                b.s_waitcnt(vmcnt=LB)
                spread = None
            _pp_barrier()
            if g == 1:
                # tile i's A stage: every read of tile i ended before that barrier
                spread = _dtl_issues(sa, k2, tile0s, kup=kup, which=(True, False))
            if _interwave:
                b.s_setprio(1)
            n_mm = k_atoms * mfmas_m * mfmas_n
            at = {}
            lo_ = int(spec.trait.pp_spread_lo)
            for j, f in enumerate(spread):
                at.setdefault(
                    lo_ + ((2 * j + 1) * (n_mm - lo_)) // (2 * len(spread)), []
                ).append(f)
            for kk in range(k_atoms):
                fa, fb = frs[kk][:mfmas_m], frs[kk][mfmas_m:]
                for mi in range(mfmas_m):
                    for ni in range(mfmas_n):
                        flat = mi * mfmas_n + ni
                        acc[flat] = _mma_ab(fa[mi], fb[ni], acc[flat], ni=ni)
                        fs = at.get(kk * mfmas_m * mfmas_n + flat, ())
                        for f in fs:
                            f()
                        if fs:
                            b.sched_barrier(0)
            if _interwave:
                b.sched_barrier(0)
                b.s_setprio(0)
            b.sched_barrier(0)
            if g == 0:
                b.s_waitcnt(vmcnt=LB)
            _pp_barrier()
            return acc, None

        def _tile_pp7(t_, k_i, acc, tile0s=None, kup=None):
            """``pingpong = 7``: group 0 loads A(i+1) (whole tile) at the start of
            its memory phase and waits ``vmcnt(0)`` after its compute phase; group
            1 loads B(i+2) (whole tile, 3-stage B ring) at the start of its memory
            phase and waits ``vmcnt(LBg)`` after its reads (B(i+1) landed, B(i+2)
            in flight). Both targets were last read before the barrier that opened
            the issuing phase; compute phases are pure MFMA."""
            acc = list(acc)
            g = _ppg[0]
            sa, sb = t_ % 2, t_ % 3
            mix = int(spec.trait.pp_mix_reads)
            if g == 0:
                burst = _dtl_issues_g(
                    "A", (t_ + 1) % 2, b.add(k_i, c_block_k), tile0s, kup
                )
            else:
                burst = _dtl_issues_g(
                    "B", (t_ + 2) % 3, b.add(k_i, b.const_i32(2 * block_k)), tile0s, kup
                )
            if not mix:
                for f in burst:
                    f()
                b.sched_barrier(0)
                frs = [_frags(sa, kk, sb) for kk in range(k_atoms)]
            else:
                frs, lds_ = [], []
                for kk in range(k_atoms):
                    o_, ld = _frag_loaders(sa, kk, sb)
                    frs.append(o_)
                    lds_ += ld
                burst[0]()
                b.sched_barrier(0)
                j = 1
                for r, f in enumerate(lds_):
                    f()
                    if (r + 1) % mix == 0 and j < len(burst):
                        b.sched_barrier(0)
                        burst[j]()
                        b.sched_barrier(0)
                        j += 1
                for f in burst[j:]:
                    f()
            b.s_waitcnt(lgkmcnt=0)
            if g == 1:
                b.s_waitcnt(vmcnt=len(burst))
            _pp_barrier()
            if _interwave:
                b.s_setprio(1)
            for kk in range(k_atoms):
                fa, fb = frs[kk][:mfmas_m], frs[kk][mfmas_m:]
                for mi in range(mfmas_m):
                    for ni in range(mfmas_n):
                        flat = mi * mfmas_n + ni
                        acc[flat] = _mma_ab(fa[mi], fb[ni], acc[flat], ni=ni)
            if _interwave:
                b.sched_barrier(0)
                b.s_setprio(0)
            b.sched_barrier(0)
            if g == 0:
                b.s_waitcnt(vmcnt=0)
            _pp_barrier()
            return acc, None

        def _tile_pp(s, k_i, acc, vm_wait, with_load, tile0s=None, kup=None):
            """``pingpong``: memory phase | barrier | compute phase | barrier."""
            if pp == 7:
                return _tile_pp7(s, k_i, acc, tile0s, kup)
            if pp == 6:
                return _tile_pp6(s, k_i, acc, tile0s, kup)
            if pp in (4, 5):
                return _tile_pp4(s, k_i, acc, with_load, tile0s, kup)
            acc = list(acc)
            issues = (
                _dtl_issues(
                    (s + S - 1) % S,
                    b.add(k_i, b.const_i32((S - 1) * block_k)),
                    tile0s,
                    kup=kup,
                )
                if with_load
                else []
            )
            if pp == 1:
                for f in issues:
                    f()
            frs = [_frags(s, kk) for kk in range(k_atoms)]
            if pp == 2:
                for f in issues:
                    f()
            # pp_wait_late: tile i+2 after the compute phase (the latest DTL,
            # tile i+S-1, is issued by then in every mode)
            vm_late = max(0, vm_wait - L)
            if pp == 3 and with_load:
                # this tile's DTL is issued in the compute phase below
                vm_wait = max(0, vm_wait - L)
            if pp_late:
                b.s_waitcnt(lgkmcnt=0)
            else:
                b.s_waitcnt(vmcnt=vm_wait, lgkmcnt=0)
            _pp_barrier()
            if _interwave:
                b.s_setprio(1)
            n_mm = k_atoms * mfmas_m * mfmas_n
            at = {}
            if pp == 3:
                for j, f in enumerate(issues):
                    at.setdefault(((2 * j + 1) * n_mm) // (2 * len(issues)), []).append(
                        f
                    )
            for kk in range(k_atoms):
                fa, fb = frs[kk][:mfmas_m], frs[kk][mfmas_m:]
                for mi in range(mfmas_m):
                    for ni in range(mfmas_n):
                        flat = mi * mfmas_n + ni
                        acc[flat] = _mma_ab(fa[mi], fb[ni], acc[flat], ni=ni)
                        fs = at.get(kk * mfmas_m * mfmas_n + flat, ())
                        for f in fs:
                            f()
                        if fs:
                            b.sched_barrier(0)
            if _interwave:
                b.sched_barrier(0)
                b.s_setprio(0)
            if pp_late:
                # tile i+2 (one tile further than the memory-phase wait, which
                # it replaces): group 1 reaches this point only after group 0
                # has started reading tile i+1, so waiting here for tile i+1
                # would be one phase late
                b.sched_barrier(0)
                b.s_waitcnt(vmcnt=vm_late)
            _pp_barrier()
            return acc, None

        T = b.to_sgpr_u32(b.div(b.sub(k_up, k_lo), c_block_k))
        U = 6 if pp in (6, 7) else S  # pingpong 6 / 7: lcm of the A (2) and B (3) rings
        _ku = int(spec.trait.kloop_unroll)
        if not pp:
            U = S * _ku  # kloop_unroll: U tiles per main-loop trip
        n_main = b.mul(b.div(T, b.const_i32(U)), b.const_i32(U))
        n_pro = S if instream else S - 1

        def _stag_of(bm: Value, bn: Value) -> Value:
            """``k_stagger_rc``: K start (elements) of output tile (bm, bn)."""
            rc = b.add(b.div(bm, c_block_m), b.div(bn, c_block_n))
            st = b.mod(b.mod(rc, b.const_i32(int(spec.trait.k_stagger_rc))), T)
            return b.to_sgpr_u32(b.mul(st, c_block_k))

        def _prologue(tile0s=None, valid=None, klo=None, kup=None) -> None:
            kl_ = k_lo if klo is None else klo
            for j in range(n_pro):
                k_j = b.add(kl_, b.const_i32(j * block_k))
                if _sia is not None:
                    # sia_sched: the table's operand order (the acquire
                    # vmcnt counts assume it)
                    for op in _sia["order"]:
                        for f in _dtl_issues(
                            j, k_j, tile0s, valid, kup, which=(op == "A", op == "B")
                        ):
                            f()
                    continue
                for f in _dtl_issues(j, k_j, tile0s, valid, kup):
                    f()

        if _vg:
            _tile_raw = _tile
            _prologue_raw = _prologue

            def _tile(*a_, **k_):
                r_ = _tile_raw(*a_, **k_)
                _vg_commit()
                return r_

            def _prologue(*a_, **k_):
                _prologue_raw(*a_, **k_)
                _vg_commit()

        def _kloop(
            acc_init: List[Value],
            tile0s=None,
            tag: str = "",
            klo=None,
            Tn=None,
            kup=None,
        ) -> List[Value]:
            """Main loop + tail of one output tile (its prologue already issued).
            Stream-K item: ``klo`` / ``kup`` bound its K range and ``Tn`` counts
            its K tiles (else the kernel's ``k_lo`` / ``k_up`` / ``T``)."""
            kl_ = k_lo if klo is None else klo
            T_ = T if Tn is None else Tn
            nm_ = (
                n_main
                if Tn is None
                else b.mul(b.div(Tn, b.const_i32(U)), b.const_i32(U))
            )
            fr0 = None
            if instream:
                b.s_waitcnt(vmcnt=(S - 1) * L, lgkmcnt=0)
                b.s_barrier_bare()
            if pp:
                # tile 0 landed (S-1 tiles issued; pp_wait_late: tiles 0 and 1);
                # group 1 then lags one phase
                b.s_waitcnt(vmcnt=(S - (3 if pp_late else 2)) * L, lgkmcnt=0)
                _pp_barrier()
                if pp == 7:
                    if _ppg[0] == 1:
                        for f in _dtl_issues_g(
                            "B", 1, b.add(kl_, c_block_k), tile0s, kup
                        ):
                            f()
                        b.s_barrier_bare()
                elif pp == 6:
                    # steady-state issues that tile 0 would have inherited
                    if _ppg[0] == 0:
                        for f in _dtl_issues(
                            0,
                            b.add(kl_, c_block_k),
                            tile0s,
                            kup=kup,
                            which=(False, True),
                            stage_b=1,
                        ):
                            f()
                    else:
                        for f in _dtl_issues(1, b.add(kl_, c_block_k), tile0s, kup=kup):
                            f()
                        b.s_barrier_bare()
                elif pp in (4, 5):
                    if _ppg[0] == 1:
                        # group 1's share of tile 1 (later tiles: end of its
                        # memory phases), then its one-phase lag
                        for f in _dtl_issues(1, b.add(kl_, c_block_k), tile0s, kup=kup):
                            f()
                        b.s_barrier_bare()
                else:
                    with b.scf_if(pp_grp1):
                        b.s_barrier_bare()
                b.sched_barrier(0)
            if fprf and spec.trait.asm_loop:
                # asm_loop: the first fragments in the table's next-tile read
                # order, so the waitcnt state at the loop header matches the
                # back edge's (else LLVM waits for fragments the entry read
                # later than the loop does, at every first-atom MFMA)
                _o0, _ld0 = _frag_loaders(0, 0)
                _pos = {("A", 0): 0}
                _pos.update({("B", j): 1 + j for j in range(mfmas_n)})
                _pos.update({("A", j): mfmas_n + j for j in range(1, mfmas_m)})
                for _k, _kind, _op, _j in _asm_cur[0]:
                    if _kind == "r0":
                        _ld0[_pos[(_op, _j)]]()
                if spec.trait.asm_loop_rawrd:
                    b.s_waitcnt(lgkmcnt=0)
                fr0 = list(_o0)
            elif fprf and spec.trait.fr0_loop_order:
                # fr0_loop_order: the first fragments in the loop's read
                # order (A0, B0.., A1..): the loop-header waitcnt state then
                # matches the back edge's
                _o0, _ld0 = _frag_loaders(0, 0)
                for _f in _ld0:
                    _f()
                fr0 = list(_o0)
            elif fprf:
                fr0 = _frags(0, 0)
                if spec.trait.hb_tile_lgkm0:
                    # the loop header's waitcnt state then matches the back
                    # edge's (no outstanding LDS read)
                    b.s_waitcnt(lgkmcnt=0)
            # ---- main loop: (T // S) groups of S tiles, static stages ----
            args = [(f"{tag}sa{i}_{accs[i][0]}", v) for i, v in enumerate(acc_init)]
            if fprf:
                args += [(f"{tag}sf{i}", v) for i, v in enumerate(fr0)]
            k_main_hi = b.add(kl_, b.mul(nm_, c_block_k))
            # asm_loop_skew: two copies of the main loop; a wave runs the
            # even copy or the odd one (the other has an empty range for it)
            _loops = [(None, k_main_hi, "")]
            _skew = int(spec.trait.asm_loop_skew or spec.trait.hb_skew)
            if _skew:
                if _skew == 1:
                    _par = b.inline_asm(
                        "s_getreg_b32 $0, hwreg(HW_REG_HW_ID, 4, 1)",
                        "=s",
                        [],
                        I32,
                        sideeffect=True,
                        result_name_hint="simdpar",
                    )
                else:
                    _par = b.mod(b.to_sgpr_u32(warp_id), b.const_i32(2))
                _odd = b.cmp_ne(_par, c0)
                _loops = [
                    (
                        (
                            _asm.table_events(spec.trait)
                            if spec.trait.asm_loop
                            else ("hbs", 0)
                        ),
                        b.to_sgpr_u32(b.select(_odd, kl_, k_main_hi)),
                        "",
                    ),
                    (
                        (
                            _asm.table_events(spec.trait, odd=True)
                            if spec.trait.asm_loop
                            else ("hbs", 1)
                        ),
                        b.to_sgpr_u32(b.select(_odd, k_main_hi, kl_)),
                        "o",
                    ),
                ]
            for _evs_l, _hi_l, _sfx in _loops:
                if isinstance(_evs_l, tuple) and _evs_l and _evs_l[0] == "hbs":
                    _hb_shift[0] = _evs_l[1]
                elif _evs_l is not None:
                    _asm_cur[0] = _evs_l
                if _sfx:
                    args = [
                        (f"{tag}{_sfx}{nm_a}", v_a)
                        for (nm_a, _v0), v_a in zip(args, loop.results)
                    ]
                loop = b.scf_for_iter(
                    kl_, _hi_l, b.const_i32(U * block_k), args, iv_name=f"{tag}{_sfx}kg"
                )
                with loop as (kg, iv):
                    acc = list(iv[: len(accs)])
                    fr = list(iv[len(accs) :]) if fprf else None
                    for s_ in range(U):
                        k_i = b.add(kg, b.const_i32(s_ * block_k)) if s_ else kg
                        if pp:
                            acc, fr = _tile_pp(s_, k_i, acc, keep, True, tile0s, kup)
                            continue
                        acc, fr = _tile(s_ % S, k_i, acc, fr, keep, True, tile0s, kup)
                        if _gett is not None:
                            acc = _gett.tile_end(b, acc)
                    b.scf_yield(*(acc + (fr if fprf else [])))
                if spec.trait.asm_loop_rawrd:
                    # inline-asm reads are invisible to LLVM: land them before
                    # any copy of the loop results
                    b.s_waitcnt(lgkmcnt=0)
            if spec.trait.asm_loop_skew:
                _asm_cur[0] = _asm.table_events(spec.trait)
            _hb_shift[0] = 0
            res = list(loop.results)
            acc = res[: len(accs)]
            fr = res[len(accs) :] if fprf else None
            # ---- tail: T % S tiles (< S), each a 0/1-trip loop, static stage s ----
            rem = b.sub(T_, nm_)
            for s_ in range(U - 1):
                targs = [(f"{tag}st{s_}a{i}", v) for i, v in enumerate(acc)]
                if fprf:
                    targs += [(f"{tag}st{s_}f{i}", v) for i, v in enumerate(fr)]
                tl = b.scf_for_iter(
                    b.const_i32(s_), rem, b.const_i32(U), targs, iv_name=f"{tag}kt{s_}"
                )
                with tl as (_kt, tv):
                    a_ = list(tv[: len(acc)])
                    f_ = list(tv[len(acc) :]) if fprf else None
                    k_i = b.add(kl_, b.mul(b.add(nm_, b.const_i32(s_)), c_block_k))
                    if pp:
                        # waits for tile i+1 (no loads are issued in the tail)
                        a_, f_ = _tile_pp(
                            s_, k_i, a_, max(0, S - 3 - s_) * L, False, tile0s, kup
                        )
                    elif _ku > 1:
                        # tail tiles up to U - 1 > S - 1: tile i + S can still
                        # be in range, so issue (the main loop's waits apply)
                        a_, f_ = _tile(s_ % S, k_i, a_, f_, keep, True, tile0s, kup)
                    else:
                        a_, f_ = _tile(
                            s_, k_i, a_, f_, (S - 2 - s_) * L, False, tile0s, kup
                        )
                    b.scf_yield(*(a_ + (f_ if fprf else [])))
                if spec.trait.asm_loop_rawrd:
                    b.s_waitcnt(lgkmcnt=0)
                r = list(tl.results)
                acc = r[: len(acc)]
                fr = r[len(acc) :] if fprf else None
            if pp:
                b.sched_barrier(0)
                if pp in (4, 5, 6, 7):
                    if _ppg[0] == 0:
                        b.s_barrier_bare()
                else:
                    with b.scf_if(pp_grp0):
                        b.s_barrier_bare()
                if pp in (4, 5, 6, 7):
                    # group 1's out-of-range tail DMAs still write (zeros) into LDS
                    b.s_waitcnt(vmcnt=0)
                b.sched_barrier(0)
            return acc

        def _streamk_loop(total, wg, c_p, _tile_offs) -> None:
            """``streamk``: the persistent loop over this
            workgroup's work items. CK Tile's DP + 2-tile SK partition is resolved
            here from the runtime tile count, the K tiles per output tile (``T``) and
            the pool P (= ``persistent_ctas``; launch exactly P co-resident
            workgroups): when P does not divide the tiles, one full wave of tiles
            plus the remainder are split by K tiles evenly over the pool, the rest
            stay data-parallel (workgroup w owns tiles w, w + P, ... first). An
            item = one output tile + the K-tile range [lo, hi) this workgroup owns.
            The next item's DTL prologue is issued before the current item's fixup
            and stores (as in the plain persistent loop).

            Linear fixup (helpers.streamk; the wgrad conv twin): the workgroup that
            covers an output tile's first K tile owns it, adds the partial tiles of
            the later contributors (all higher-index workgroups, each publishing
            its partial as its FIRST stream-K item) in order and stores C; a
            contributor stores its partial (f32, ``SkPart`` slot w) and then sets
            its ready flag ``SkFlags[w * waves + wave]`` (release). The owner waits
            for the flag (acquire), reads the partial and resets the flag to 0, so
            the flag table is zero again after every launch."""
            from ...helpers.streamk import StreamKIterArgs, emit_streamk_iter_range

            ipt = T
            c1 = b.const_i32(1)
            rem_t = b.mod(total, c_p)
            two_t = b.select(b.cmp_lt(c_p, total), b.add(c_p, rem_t), total)
            sk_t = b.select(b.cmp_eq(rem_t, c0), c0, two_t)
            sk_t = b.to_sgpr_u32(b.select(b.cmp_lt(b.mul(sk_t, ipt), c_p), c0, sk_t))
            dp_t = b.to_sgpr_u32(b.sub(total, sk_t))
            sk_iters = b.mul(sk_t, ipt)
            per = b.to_sgpr_u32(b.smax(b.div(sk_iters, c_p), c1))
            extra = b.to_sgpr_u32(b.mod(sk_iters, c_p))
            dp_iters = b.to_sgpr_u32(b.mul(dp_t, ipt))
            sk_args = StreamKIterArgs(
                iters_per_tile=ipt,
                dp_tiles=dp_t,
                total_dp_iters=dp_iters,
                iters_per_sk_cta=per,
                extra_iters=extra,
            )
            in_pool = b.cmp_lt(wg, c_p)
            # data-parallel tiles wg, wg + P, ... < dp_t (wg < P keeps it >= 0)
            n_dp = b.select(
                in_pool, b.div(b.add(b.sub(dp_t, wg), b.sub(c_p, c1)), c_p), c0
            )
            # ``sk_xcd_local``: the stream-K rank of this workgroup. Workgroup w
            # runs on XCD w mod X (the chiplet swizzle's dispatch model), so rank
            # (w mod X) * (P / X) + w / X puts consecutive ranks -- a tile's owner
            # and its partners -- on one XCD (one L2) except at the X - 1 rank
            # boundaries between XCD groups. Without the knob rank = wg.
            xcd_local = bool(spec.trait.sk_xcd_local)
            if xcd_local:
                c_nx = b.const_i32(spec.trait.chiplet_num_xcds)
                c_px = b.const_i32(pers // spec.trait.chiplet_num_xcds)
                me = b.to_sgpr_u32(b.add(b.mul(b.mod(wg, c_nx), c_px), b.div(wg, c_nx)))
            else:
                me = wg
            rng = emit_streamk_iter_range(b, me, sk_args, dp=False)
            first_tile = b.to_sgpr_u32(b.div(rng.start, ipt))
            last_tile = b.div(b.sub(rng.end, c1), ipt)
            n_sk = b.add(b.sub(last_tile, first_tile), c1)
            n_sk = b.select(b.land(in_pool, b.cmp_lt(c0, sk_t)), n_sk, c0)
            n_dp = b.to_sgpr_u32(n_dp)
            n_items = b.to_sgpr_u32(b.add(n_dp, n_sk))

            def _item(it: Value):
                is_dp = b.cmp_lt(it, n_dp)
                dp_tile = b.add(wg, b.mul(it, c_p))
                sk_tile = b.add(first_tile, b.sub(it, n_dp))
                tl = b.select(is_dp, dp_tile, sk_tile)
                tl = b.to_sgpr_u32(b.select(b.cmp_lt(tl, total), tl, c0))
                ts = b.mul(tl, ipt)
                st_ = b.select(is_dp, ts, rng.start)
                en_ = b.select(is_dp, b.add(ts, ipt), rng.end)
                lo = b.to_sgpr_u32(b.sub(b.smax(st_, ts), ts))
                hi = b.to_sgpr_u32(b.sub(b.smin(en_, b.add(ts, ipt)), ts))
                return tl, lo, hi

            def _kb(v: Value) -> Value:
                return b.to_sgpr_u32(b.mul(v, c_block_k))

            t = spec.tile
            n_waves = t.warp_m * t.warp_n * t.warp_k
            c_waves = b.const_i32(n_waves)
            nq = c_per_lane // 4 if c_per_lane % 4 == 0 else 0
            slot = len(accs) * c_per_lane * threads  # floats per partial tile

            def _flag_idx(w: Value) -> Value:
                return b.add(b.mul(w, c_waves), warp_id)

            # [w][atom][quad][tid][4] f32: one 16-byte buffer access per lane
            # (coalesced); the slot base is the resource, the (atom, quad)
            # offset an SGPR constant, the lane offset one loop-invariant VGPR
            voff_p = b.mul(tid, b.const_i32(16))

            def _part_rsrc(w: Value) -> Value:
                base = b.to_sgpr_u32(b.mul(w, b.const_i32(slot * 4)))
                return b.buffer_rsrc(
                    b.global_ptr_add(sk_part, base), b.const_i32(slot * 4)
                )

            def _soff(ai: int, q: int) -> Value:
                return b.const_i32((ai * nq + q) * threads * 16)

            def _quads(v: Value) -> List[Value]:
                if c_per_lane == 4:
                    return [v]
                return [
                    b.vec_pack([b.vec_extract(v, 4 * q + i) for i in range(4)], F32)
                    for q in range(nq)
                ]

            def _store_partial(w: Value, vals: Sequence[Value], tl_=None) -> None:
                rs = _part_rsrc(w)
                for ai, v in enumerate(vals):
                    for q, v4 in enumerate(_quads(_acc_agpr_out(b, spec, v))):
                        b.buffer_store_vN_f32(rs, voff_p, _soff(ai, q), v4, 4)
                if xcd_local:
                    # owner on this XCD (same L2): the partial stores only have
                    # to reach the L2 -- no write-back of the whole XCD L2
                    owner = _sk_cta_of(b.mul(tl_, ipt))
                    cross = b.cmp_ne(b.div(owner, c_px), b.div(w, c_px))
                    with b.scf_if_else(cross) as (far, near):
                        with far:
                            b.fence(scope="agent", ordering="release")
                        with near:
                            b.s_waitcnt(vmcnt=0)
                            b.fence(scope="workgroup", ordering="release")
                else:
                    b.fence(scope="agent", ordering="release")
                b.global_flag_store(sk_flags, _flag_idx(w), c1, ordering="monotonic")

            def _load_atom(rs: Value, ai: int) -> Value:
                qs = [
                    b.buffer_load_vN(rs, voff_p, _soff(ai, q), F32, 4)
                    for q in range(nq)
                ]
                if c_per_lane == 4:
                    return qs[0]
                return b.vec_pack(
                    [b.vec_extract(qs[i // 4], i % 4) for i in range(c_per_lane)], F32
                )

            def _sk_cta_of(it: Value) -> Value:
                rel = b.sub(it, dp_iters)
                wide = b.add(per, c1)
                boundary = b.mul(extra, wide)
                lead = b.div(rel, wide)
                tail = b.add(extra, b.div(b.sub(rel, boundary), per))
                return b.select(b.cmp_lt(rel, boundary), lead, tail)

            def _sk_fixup(acc, hi_p, bm, bn) -> None:
                """Owner of a tile with partners wg + 1 .. wg + hi_p - 1: wait for
                each partner's flag (a relaxed spin -- an acquire load per spin
                would invalidate the XCD's L2 every time -- then one acquire
                fence), fold the partials in a group of atoms at a time (16
                floats in flight without acc_agpr, 64 with; with acc_agpr each
                folded atom goes back to the AGPR file until the epilogue hands
                it out), reset the flags for the next launch, store C."""
                with b.scf_for(c1, hi_p, c1, iv_name="skw") as pi:
                    b.global_flag_wait_eq(
                        sk_flags, _flag_idx(b.add(me, pi)), c1, ordering="monotonic"
                    )
                    if xcd_local:
                        # a partner on this XCD wrote its partial into the shared
                        # L2: no invalidation of the XCD's L2 (every other
                        # workgroup's A / B lines) unless it sits on another XCD
                        cross = b.cmp_ne(b.div(b.add(me, pi), c_px), b.div(me, c_px))
                        with b.scf_if_else(cross) as (far, near):
                            with far:
                                b.fence(scope="agent", ordering="acquire")
                            with near:
                                b.fence(scope="workgroup", ordering="acquire")
                    else:
                        b.fence(scope="agent", ordering="acquire")
                vals = list(acc)
                grp = max(1, (64 if spec.trait.acc_agpr else 16) // c_per_lane)
                for g0 in range(0, len(accs), grp):
                    idx = list(range(g0, min(g0 + grp, len(accs))))
                    fold = b.scf_for_iter(
                        c1,
                        hi_p,
                        c1,
                        [
                            (f"skf_{accs[i][0]}", _acc_agpr_out(b, spec, vals[i]))
                            for i in idx
                        ],
                        iv_name=f"skp{g0}",
                    )
                    with fold as (pi, folded):
                        rs_p = _part_rsrc(b.add(me, pi))
                        parts = [_load_atom(rs_p, i) for i in idx]
                        b.scf_yield(
                            *[b.vector_add(v, q) for v, q in zip(folded, parts)]
                        )
                    for i, r in zip(idx, fold.results):
                        vals[i] = (
                            b.inline_asm("", "=a,0", [r], r.type, sideeffect=True)
                            if spec.trait.acc_agpr
                            else r
                        )
                with b.scf_for(c1, hi_p, c1, iv_name="skz") as pi:
                    b.global_flag_store(
                        sk_flags, _flag_idx(b.add(me, pi)), c0, ordering="monotonic"
                    )
                _ep_default(vals, bm, bn)

            tl0, lo0, hi0 = _item(c0)
            _bm0, _bn0, t00 = _tile_offs(tl0)
            _prologue(t00, b.cmp_lt(c0, n_items), _kb(lo0), _kb(hi0))
            acc_zero = [a for (_n, a) in accs]
            with b.scf_for(c0, n_items, c1, iv_name="skitem") as it:
                tl, lo, hi = _item(it)
                bm, bn, t0 = _tile_offs(tl)
                acc = _kloop(
                    acc_zero, t0, "p", _kb(lo), b.to_sgpr_u32(b.sub(hi, lo)), _kb(hi)
                )
                nxt = b.add(it, c1)
                has_next = b.cmp_lt(nxt, n_items)
                tl1, lo1, hi1 = _item(b.select(has_next, nxt, it))
                _bm1, _bn1, t01 = _tile_offs(tl1)
                # every wave has finished reading the ring before it is refilled
                b.s_waitcnt(lgkmcnt=0)
                b.s_barrier_bare()
                _prologue(t01, has_next, _kb(lo1), _kb(hi1))
                started = b.cmp_eq(lo, c0)
                ended = b.cmp_eq(hi, ipt)
                with b.scf_if_else(started) as (owner, contrib):
                    with owner:
                        tile_end = b.add(b.mul(tl, ipt), ipt)
                        last = _sk_cta_of(b.sub(tile_end, c1))
                        n_part = b.select(ended, c0, b.sub(last, me))
                        with b.scf_if_else(b.cmp_lt(c0, n_part)) as (fix, plain):
                            with fix:
                                _sk_fixup(acc, b.to_sgpr_u32(b.add(n_part, c1)), bm, bn)
                            with plain:
                                # data-parallel tile / no partner: store as is
                                _ep_default(acc, bm, bn)
                    with contrib:
                        _store_partial(me, acc, tl)

        if not pers:
            if spec.trait.k_stagger_rc:
                _ksr[0] = _stag_of(block_m_off, block_n_off)
            _prologue()
            if pp in (4, 5, 6, 7):
                # one specialised K loop + epilogue per wave group (no branch
                # inside the loop: a CFG split there moves the accumulators out
                # of the AGPR file); the barrier counts of the two match
                with b.scf_if_else(pp_grp0) as (_th, _el):
                    for _g, _ctx in ((0, _th), (1, _el)):
                        with _ctx:
                            _ppg[0] = _g
                            _for_results = _kloop([a for (_n, a) in accs], tag=f"g{_g}")
                            _emit_epilogue()
                _for_results = None
                return
            _for_results = _kloop([a for (_n, a) in accs])
            return

        # ---- persistent body (persistent_ctas = P): workgroup w < P owns the
        # output tiles w, w + P, w + 2P, ... (workgroups >= P exit, so the
        # standard 2D grid stays correct; launch P workgroups to save the empty
        # ones). The next tile's DTL prologue is issued right after the current
        # tile's last MFMA and BEFORE its epilogue stores, so the stores overlap
        # the next tile's global-load latency. The XCD round-robin of the
        # chiplet swizzle is preserved because is_valid_spec requires P to be
        # a multiple of the XCD count.
        #
        # vmcnt with S >= 3 (gfx9 has no separate store counter, so the
        # epilogue stores share vmcnt with the DTL loads): when tile j+1's first
        # wait ``vmcnt((S-2)*L)`` executes, the issue order is [leftover
        # out-of-range loads of tile j] [(S-1)*L prologue loads of tile j+1]
        # [n epilogue stores of tile j]. The wait returns once at most (S-2)*L
        # are outstanding, i.e. at least leftover + L + n have retired. The DTL
        # loads retire in issue order among themselves (the same assumption the
        # non-persistent partial wait makes; hardware-checked at every K-tile
        # residue), so whichever stores have or have not retired, at least
        # leftover + L loads have -- the first L prologue loads included. Stores
        # retiring ahead of (or behind) earlier loads therefore never lets the
        # wait pass early; it can only make it wait longer. ----
        n_pid_m = b.div(b.add(M, b.const_i32(block_m - 1)), c_block_m)
        n_pid_n = b.div(b.add(N, b.const_i32(block_n - 1)), c_block_n)
        total = b.to_sgpr_u32(b.mul(n_pid_m, n_pid_n))
        wg = b.to_sgpr_u32(b.add(b.mul(b.block_id_y(), n_pid_n), b.block_id_x()))
        c_p = b.const_i32(pers)

        def _tile_offs(tid_flat: Value):
            if spec.trait.chiplet_swizzle:
                from ...helpers.grid import chiplet_aware_super_tile_dynamic

                sw = chiplet_aware_super_tile_dynamic(
                    b,
                    tid_flat,
                    num_pid_m=n_pid_m,
                    num_pid_n=n_pid_n,
                    wgm=spec.trait.chiplet_wgm,
                    num_xcds=spec.trait.chiplet_num_xcds,
                    chunk_size=spec.trait.chiplet_chunk_size,
                )
                row, col = sw.row, sw.col
            else:
                row, col = b.div(tid_flat, n_pid_n), b.mod(tid_flat, n_pid_n)
            bm = b.to_sgpr_u32(b.mul(row, c_block_m))
            bn = b.to_sgpr_u32(b.mul(col, c_block_n))
            if _gett is not None:
                t0 = (_gett.tile_voffs(b, "A", bm), _gett.tile_voffs(b, "B", bn))
            else:
                t0 = (b.to_sgpr_u32(b.mul(bm, K)), b.to_sgpr_u32(b.mul(_brow(bn), K)))
            return bm, bn, t0

        if getattr(spec, "_fused_epilogue", None) is not None:
            raise ValueError("persistent_ctas does not support a fused epilogue")
        if spec.trait.streamk:
            _streamk_loop(total, wg, c_p, _tile_offs)
            _for_results = None
            return
        mine = b.land(b.cmp_lt(wg, c_p), b.cmp_lt(wg, total))
        first = b.to_sgpr_u32(b.select(b.cmp_lt(wg, total), wg, c0))
        _bm0, _bn0, t00 = _tile_offs(first)
        if spec.trait.k_stagger_rc:
            _ksr[0] = _stag_of(_bm0, _bn0)
        if _mono is not None:
            _mono.pre_tasks(b, wg=wg, nwg=c_p, tid=tid)
            _mono.wait_tile(b, _bm0, _bn0, mine, tid=tid)
        _prologue(t00, mine)
        hi = b.select(mine, total, wg)  # zero-trip loop for workgroups that own no tile
        acc_zero = [a for (_n, a) in accs]

        def _persistent_tiles(tag: str) -> None:
            with b.scf_for(wg, hi, c_p, iv_name=f"{tag}tile") as tcur:
                bm, bn, t0 = _tile_offs(tcur)
                if spec.trait.k_stagger_rc:
                    _ksr[0] = _stag_of(bm, bn)
                acc = _kloop(acc_zero, t0, tag=tag)
                tnext = b.add(tcur, c_p)
                has_next = b.cmp_lt(tnext, total)
                _bm1, _bn1, t01 = _tile_offs(b.select(has_next, tnext, tcur))
                if spec.trait.k_stagger_rc:
                    _ksr[0] = _stag_of(_bm1, _bn1)
                # every wave has finished reading the ring before it is refilled
                b.s_waitcnt(lgkmcnt=0)
                b.s_barrier_bare()
                if _mono is not None:
                    # dependency wait of the next tile (e.g. the persistent
                    # contraction monokernel's operand-pack counter / flags)
                    _mono.wait_tile(b, _bm1, _bn1, has_next, tid=tid)
                _prologue(t01, has_next)
                _ep_default(acc, bm, bn)
                if _mono is not None:
                    _mono.post_tile(b, bm, bn, tid=tid)

        if pp in (4, 5, 6, 7):
            # pingpong 4 .. 7: one specialised persistent loop per wave group
            # (equal barrier counts per output tile in both)
            with b.scf_if_else(pp_grp0) as (_th, _el):
                for _g, _ctx in ((0, _th), (1, _el)):
                    with _ctx:
                        _ppg[0] = _g
                        _persistent_tiles(f"p{_g}")
        else:
            _persistent_tiles("p")
        _for_results = None

    _for_results: Sequence[Value] = ()

    def _emit_epilogue() -> None:
        # ---- epilogue ----
        fused_ep = getattr(spec, "_fused_epilogue", None)
        if _is_split_k:
            # Split-K: atomic-add each warp's f32 accumulator into the
            # Cf32[M, N] workspace. Reuses the same MFMA acc -> (row, col)
            # scatter as the direct epilogue, but the per-cell write is an
            # f32 atomicrmw fadd instead of a bf16 global store. The caller
            # zero-inits the workspace and casts it to bf16 afterwards.
            _emit_epilogue_split_k(
                b,
                spec,
                _for_results,
                warp_m_idx,
                warp_n_idx,
                lane,
                block_m_off,
                block_n_off,
                M,
                N,
                C,
                c_per_lane,
            )
            return
        if spec.trait.epilogue == "cshuffle":
            _emit_epilogue_cshuffle(
                b,
                spec,
                A_smem,
                _for_results,
                warp_m_idx,
                warp_n_idx,
                lane,
                block_m_off,
                block_n_off,
                M,
                N,
                C,
                a_per_lane,
                b_per_lane,
                c_per_lane,
                batch_off_c=batch_off_c,
                fused_epilogue=fused_ep,
            )
        else:
            _ep_default(
                _for_results,
                block_m_off,
                block_n_off,
                batch_off_c=batch_off_c,
                fused_epilogue=fused_ep,
            )

    if do_work_cond is None:
        emit_compute_and_epilogue()
    else:
        with b.scf_if(do_work_cond):
            emit_compute_and_epilogue()

    return b.kernel


# ---------------------------------------------------------------------
# Epilogues
# ---------------------------------------------------------------------


def _acc_agpr_out(b: IRBuilder, spec: UniversalGemmSpec, acc: Value) -> Value:
    """``acc_agpr`` epilogue: hand one accumulator out of the AGPRs at the point
    of use. The ordered (side-effecting) ``asm "", "=a,0"`` keeps the value in
    AGPRs until its own stores, so the AGPR -> VGPR copies happen one MFMA atom
    at a time instead of all at the loop exit (which spills the epilogue's
    address registers to scratch). Identity without ``acc_agpr``."""
    if not spec.trait.acc_agpr:
        return acc
    if spec.trait.acc_out_vgpr:
        # the tied VGPR operand makes the AGPR -> VGPR copy (v_accvgpr_read)
        # the asm's own input copy: no AGPR -> AGPR move (+ its hazard nops)
        return b.inline_asm("", "=v,0", [acc], acc.type, sideeffect=True)
    return b.inline_asm("", "=a,0", [acc], acc.type, sideeffect=True)


def _emit_mfma_acc_scatter(
    b: IRBuilder,
    spec: UniversalGemmSpec,
    lane: Value,
    accs: Sequence[Value],
    m_base_off: Value,
    n_base_off: Value,
    c_per_lane: int,
    storage_dtype: Type,
    per_cell,
    *,
    n_base_first: bool = False,
) -> None:
    """Walk an MFMA warp tile's per-lane accumulator slots in output order.

    This is the single source of the MFMA accumulator -> (row, col) scatter
    that the *default* (direct global store) and the *cshuffle* (LDS staging)
    epilogues both need. It reproduces the canonical AMD MFMA output layout
    (``MfmaAtom.lane_to_output`` / the ``c_layout`` map in
    :mod:`rocke.core.arch`) with the same hoisting the two epilogues used to
    open-code in four near-identical blocks (default-16/32, cshuffle-16/32):

    * 16x16 atom: ``row = m_blk*c_per_lane + i``, ``col = n_in_atom``.
    * 32x32 atom: ``row = (i//4)*8 + m_blk*4 + (i%4)``, ``col = n_in_atom``,
      where the per-slot ``(i//4)*8 + (i%4)`` ramp is folded to one host
      constant exactly as before.

    ``m_base_off`` / ``n_base_off`` are the warp-tile base offsets the caller
    already holds: the default epilogue passes ``block + warp`` offsets (the
    store targets global), the cshuffle epilogue passes ``warp`` offsets (the
    block offset is added later at the wide global store). The per-cell
    callback ``per_cell(c_m, c_n, acc_h, i)`` owns the actual write.

    ``n_base_first`` selects the column-offset association so each call site's
    prior IR is reproduced: the default epilogue grouped it as
    ``n_base + (ni*wtn + n_in_atom)`` (``False``); the cshuffle epilogue
    grouped it as ``(n_base + ni*wtn) + n_in_atom`` (``True``). Both are
    arithmetically equal; the flag only chooses the column grouping.

    The MFMA C-fragment ``(row_in_atom, col_in_atom)`` decode is now sourced
    from the atom's CK-Tile ``CWarpDstrEncoding`` (a
    :class:`~rocke.helpers.distribution.TileDistribution`) instead of the
    open-coded ``m_blk``/``lane_m``/``row_ramp`` lane arithmetic. The single
    lane is split into the distribution's P sub-sequence ``[m_blk, n_in_atom]
    = [lane // kCNLane, lane % kCNLane]`` and the per-lane accumulator slot
    ``i`` into the two Y dims ``(i // kCM1PerLane, i % kCM1PerLane)``;
    ``calculate_x`` then yields ``row_in_atom = lane_m + row_ramp`` and
    ``col_in_atom = n_in_atom``, identical to the prior formulas for every
    supported atom (16x16 and 32x32). This is Tier-2: the same div/mod and
    add/mul values are produced, but the SSA emission order differs from the
    hand-hoisted form.
    """
    from ...helpers.atoms import c_warp_params, make_c_warp_dstr_encoding, mfma_atom
    from ...helpers.distribution import make_static_tile_distribution

    t = spec.tile
    mfmas_m = t.mfmas_per_warp_m
    mfmas_n = t.mfmas_per_warp_n
    # The CWarpDstrEncoding is keyed on the warp-tile (m, n) only (the lane/
    # slot -> (row, col) layout is dtype-independent), so resolve the atom by
    # the spec's input dtype + warp-tile shape.
    atom = mfma_atom(
        spec.data.dtype_a,
        t.warp_tile_m,
        t.warp_tile_n,
        t.warp_tile_k,
    )
    _, _kc_mlane, kc_m1, kc_nlane = c_warp_params(atom)
    c_dist = make_static_tile_distribution(make_c_warp_dstr_encoding(atom))

    c_nlane = b.const_i32(kc_nlane)
    # Single P (the lane) -> [m_blk, n_in_atom] = [lane // kCNLane, lane %
    # kCNLane]. n_in_atom is also the column coordinate (col_in_atom).
    n_in_atom = b.mod(lane, c_nlane)
    m_blk = b.div(lane, c_nlane)
    p_lane = [m_blk, n_in_atom]

    # Per accumulator slot ``i`` the two Y dims decompose row-major over
    # (kCM0PerLane, kCM1PerLane): y0 = i // kCM1PerLane, y1 = i % kCM1PerLane.
    # calculate_x reconstructs the full row_in_atom (lane + slot) and the
    # col_in_atom (= n_in_atom) from these.
    row_in_atom: List[Value] = []
    col_in_atom: Optional[Value] = None
    for i in range(c_per_lane):
        ys = [b.const_i32(i // kc_m1), b.const_i32(i % kc_m1)]
        x_row, x_col = c_dist.calculate_x(b, ys=ys, ps=[p_lane])
        row_in_atom.append(x_row)
        col_in_atom = x_col  # constant across i (depends only on n_in_atom)

    flat = 0
    for mi in range(mfmas_m):
        base_m = b.add(m_base_off, b.const_i32(mi * t.warp_tile_m))
        for ni in range(mfmas_n):
            acc = _acc_agpr_out(b, spec, accs[flat])
            flat += 1
            acc_h = (
                acc if storage_dtype == F32 else b.vec_cast_f32_to(acc, storage_dtype)
            )
            if n_base_first:
                c_n = b.add(
                    b.add(n_base_off, b.const_i32(ni * t.warp_tile_n)),
                    col_in_atom,
                )
            else:
                c_n = b.add(
                    n_base_off,
                    b.add(b.const_i32(ni * t.warp_tile_n), col_in_atom),
                )
            for i in range(c_per_lane):
                c_m = b.add(base_m, row_in_atom[i])
                per_cell(c_m, c_n, acc_h, i)


def _emit_epilogue_default(
    b: IRBuilder,
    spec: UniversalGemmSpec,
    op,
    accs: Sequence[Value],
    warp_m_idx: Value,
    warp_n_idx: Value,
    lane: Value,
    block_m_off: Value,
    block_n_off: Value,
    M: Value,
    N: Value,
    C: Value,
    c_per_lane: int,
    *,
    batch_off_c: Optional[Value] = None,
    fused_epilogue=None,  # Optional[FusedEpilogue] from helpers.fuse
) -> None:
    """Direct vector-store epilogue.

    Per-lane accumulator layout for an `m x n x k` MFMA atom on wave64:
      - 16x16 atoms: lane = (m_blk * 16 + n_in_atom), c_per_lane = 4
                     -> lane stores `(m_base + i, n_in_atom)` for i=0..3
        where m_base = m_blk * 4.
      - 32x32 atoms: c_per_lane = 16, accumulator is divided into 4
                     row-blocks of 4 floats each; runtime layout is
                     ((m_block_within_warp, row_in_block, n_in_atom))
                     with m_block_within_warp = lane / 32,
                     row_in_block coming from the accumulator index
                     i and row_block = i / 4.

    We use the canonical AMD layout per ROCm docs:
      For mfma_f32_32x32x8f16: each lane holds 16 fp32; the per-lane
      layout maps acc[i] to output element
      (row, col) = ((i//4)*8 + (lane/32)*4 + (i%4), lane%32)
    so the per-lane row stride between 4-element groups is 8 (not 4).
    """
    t = spec.tile
    storage_dtype = _storage_dtype(spec)
    _pro_ep = getattr(spec, "_prologue", None)
    if _pro_ep is not None and _pro_ep.out_f32:
        storage_dtype = (
            F32  # prologue out_f32: f32 output (no rounding of the accumulator)
        )
    _st_align = type_bits(storage_dtype) // 8
    mfmas_m = t.mfmas_per_warp_m
    mfmas_n = t.mfmas_per_warp_n
    pad_m = bool(spec.trait.pad_m)
    pad_n = bool(spec.trait.pad_n)

    warp_m_off = b.mul(warp_m_idx, b.const_i32(mfmas_m * t.warp_tile_m))
    warp_n_off = b.mul(warp_n_idx, b.const_i32(mfmas_n * t.warp_tile_n))

    # A fused epilogue (BiasAdd, ReLU, residual, ...) declares its extra
    # kernel params once before the scatter, then transforms each per-lane
    # scalar at its (c_m, c_n) below. Guarded by ``None`` so the matmul-only
    # default path (and the byte-identical CDNA MFMA store) is unchanged.
    if fused_epilogue is not None:
        fused_epilogue.declare_params(b)
        record_runtime = getattr(fused_epilogue, "record_runtime", None)
        if record_runtime is not None:
            record_runtime(b, N=N)

    def _store_masked(c_m: Value, c_n: Value, c_off: Value, h: Value) -> None:
        """Per-element global store with optional ``pad_m`` / ``pad_n`` guard.

        Lane-level scatter does not benefit from the vec-aligned check
        used in the cshuffle path; per-element ``c_n < N`` is the
        natural mask here and matches what the partially-tiled output
        column requires.
        """
        if not (pad_m or pad_n):
            b.global_store(C, c_off, h, align=_st_align)
            return
        checks: List[Value] = []
        if pad_m:
            checks.append(b.cmp_lt(c_m, M))
        if pad_n:
            checks.append(b.cmp_lt(c_n, N))
        in_bounds = checks[0] if len(checks) == 1 else b.land(checks[0], checks[1])
        with b.scf_if(in_bounds):
            b.global_store(C, c_off, h, align=_st_align)

    # ---- WMMA (RDNA) accumulator scatter: fully MMA-contract driven. ----
    # The WMMA accumulator distributes the M x N tile across wave32 lanes
    # differently from MFMA, so we ask the op's accumulator layout map for the
    # (row, col) of every per-lane slot rather than hard-coding the lane math.
    # One slot -> one f16 store. (The supported WMMA subset is the single
    # 16x16x16 atom -> mfmas_m == mfmas_n == 1.)
    if op.family == "wmma":
        c_map = op.c_layout()
        block_warp_m_off = b.add(block_m_off, warp_m_off)
        block_warp_n_off = b.add(block_n_off, warp_n_off)
        flat = 0
        for mi in range(mfmas_m):
            atom_m = b.add(block_warp_m_off, b.const_i32(mi * t.warp_tile_m))
            for ni in range(mfmas_n):
                acc = accs[flat]
                flat += 1
                atom_n = b.add(block_warp_n_off, b.const_i32(ni * t.warp_tile_n))
                acc_h = (
                    acc
                    if storage_dtype == F32
                    else b.vec_cast_f32_to(acc, storage_dtype)
                )
                for i in range(c_per_lane):
                    row_in_atom, col_in_atom = c_map.coord(b, lane, i)
                    c_m = b.add(atom_m, row_in_atom)
                    c_n = b.add(atom_n, col_in_atom)
                    c_off = b.add(b.mul(c_m, N), c_n)
                    if batch_off_c is not None:
                        c_off = b.add(batch_off_c, c_off)
                    h = b.vec_extract(acc_h, i)
                    if fused_epilogue is not None:
                        h = fused_epilogue.apply_scalar(b, h, c_m, c_n)
                    _store_masked(c_m, c_n, c_off, h)
        return

    # Compile-time invariants that do not depend on (mi, ni, i).
    # Hoisting them avoids re-emitting the same add chain per
    # accumulator slot; the IR's constant folder collapses them
    # downstream, but building them once keeps the IR small and the
    # lowered LLVM readable. The accumulator -> (row, col) scatter (16x16
    # and 32x32 layouts, with the per-mi base hoist + the single-constant
    # per-slot ramp) lives in :func:`_emit_mfma_acc_scatter` so the default
    # and cshuffle epilogues share one copy of the MFMA output-layout math.
    block_warp_m_off = b.add(block_m_off, warp_m_off)
    block_warp_n_off = b.add(block_n_off, warp_n_off)

    if spec.trait.mfma_swap_ab:
        if fused_epilogue is not None:
            raise ValueError("mfma_swap_ab does not support a fused epilogue")
        _emit_store_swapped_acc(
            b,
            spec,
            lane,
            accs,
            block_warp_m_off,
            block_warp_n_off,
            c_per_lane,
            storage_dtype,
            M,
            N,
            C,
            batch_off_c=batch_off_c,
        )
        return

    if spec.trait.rilv_vec_epilogue:
        # atom (mi, ni), lane (q = lane // 16, c = lane % 16), slot v holds
        # C[wm + (4q + v) * mfmas_m + mi, wn + c * mfmas_n + ni]: 8 consecutive
        # columns per (mi, v) -> one 16-byte store each
        if fused_epilogue is not None:
            raise ValueError("rilv_vec_epilogue does not support a fused epilogue")
        _rv_pad_m = bool(spec.trait.rve_pad == 1 and spec.trait.pad_m)
        _rv_pad_n = bool(spec.trait.rve_pad == 1 and spec.trait.pad_n)
        _rv_buf = spec.trait.rve_pad == 2 and (spec.trait.pad_m or spec.trait.pad_n)
        if _rv_buf:
            # branch-free bounds: one buffer resource over C (M * N * 2 bytes);
            # an out-of-range store gets voffset = num_records and is dropped
            _rv_nbytes = b.to_sgpr_u32(b.mul(b.mul(M, N), b.const_i32(2)))
            _rv_rsrc = b.buffer_rsrc(C, _rv_nbytes)
            _rv_st = (
                b.buffer_store_vN_bf16
                if storage_dtype.name == "bf16"
                else b.buffer_store_vN_f16
            )

        def _rve_store(c_off: Value, vec: Value) -> None:
            if spec.trait.rve_store_nt:
                from ...core.ir import TemporalHint as _TH

                b.global_store_vN(
                    C, c_off, vec, 8, align=16, temporal_hint=_TH.STREAMING
                )
            else:
                b.global_store_vN(C, c_off, vec, 8, align=16)

        def _rve_guarded_store(
            c_m: Value, c_n: Value, c_off: Value, vec: Value
        ) -> None:
            # rve_pad: N % 8 == 0 (launch contract), so the 8 columns of a store
            # are all inside or all outside C. The column bound is per lane
            # (one guard around the whole epilogue, below); the row bound is
            # per store.
            if _rv_buf:
                ok = None
                if spec.trait.pad_m:
                    ok = b.cmp_lt(c_m, M)
                if spec.trait.pad_n:
                    okn = b.cmp_lt(c_n, N)
                    ok = okn if ok is None else b.land(ok, okn)
                voff = b.select(ok, b.mul(c_off, b.const_i32(2)), _rv_nbytes)
                _rv_st(_rv_rsrc, voff, b.const_i32(0), vec, 4)
                return
            if not _rv_pad_m:
                _rve_store(c_off, vec)
                return
            with b.scf_if(b.cmp_lt(c_m, M)):
                _rve_store(c_off, vec)

        _c16 = b.const_i32(16)
        _q = b.div(lane, _c16)
        _cn = b.mod(lane, _c16)
        _row0 = b.add(block_warp_m_off, b.mul(_q, b.const_i32(c_per_lane * mfmas_m)))
        _col0 = b.add(block_warp_n_off, b.mul(_cn, b.const_i32(mfmas_n)))
        _mono = getattr(spec, "_mono", None)
        if _mono is not None:
            # rich-epilogue hook (``_mono``): the hook owns the whole tile store. Lane
            # (q, cn) holds rows _row0 + v * mfmas_m + mi (v < c_per_lane,
            # mi < mfmas_m) x columns _col0 + ni (ni < mfmas_n) as f32
            # accumulators accf[mi * mfmas_n + ni][v].
            accf = [_acc_agpr_out(b, spec, a) for a in accs]
            _mono.epilogue(
                b,
                accf=accf,
                row0=_row0,
                col0=_col0,
                mfmas_m=mfmas_m,
                mfmas_n=mfmas_n,
                c_per_lane=c_per_lane,
                C=C,
                M=M,
                N=N,
                block_m_off=block_m_off,
                block_n_off=block_n_off,
                warp_n_idx=warp_n_idx,
                lane=lane,
            )
            return
        _rv_colguard = None
        if _rv_pad_n:
            # every store of this lane covers columns _col0 + n0 .. + 7 with
            # n0 < mfmas_n: all inside C iff _col0 + mfmas_n <= N (N % 8 == 0
            # and mfmas_n == 8), else all outside -- one guard per lane
            assert mfmas_n == 8, mfmas_n
            _rv_colguard = b.scf_if(b.cmp_lt(_col0, N))
            _rv_colguard.__enter__()
        for mi in range(mfmas_m):
            hs = [
                (
                    _acc_agpr_out(b, spec, accs[mi * mfmas_n + ni])
                    if storage_dtype == F32
                    else b.vec_cast_f32_to(
                        _acc_agpr_out(b, spec, accs[mi * mfmas_n + ni]), storage_dtype
                    )
                )
                for ni in range(mfmas_n)
            ]
            for v in range(c_per_lane):
                c_m = b.add(_row0, b.const_i32(v * mfmas_m + mi))
                for n0 in range(0, mfmas_n, 8):
                    if storage_dtype == F32:
                        # prologue out_f32: 8 consecutive f32 = two 16-byte stores
                        from ...core.ir import TemporalHint as _TH

                        c_off = b.add(b.mul(c_m, N), b.add(_col0, b.const_i32(n0)))
                        if batch_off_c is not None:
                            c_off = b.add(batch_off_c, c_off)
                        for h4 in (0, 4):
                            v4 = b.vec_pack(
                                [
                                    b.vec_extract(hs[ni], v)
                                    for ni in range(n0 + h4, n0 + h4 + 4)
                                ],
                                F32,
                            )
                            o4 = c_off if h4 == 0 else b.add(c_off, b.const_i32(4))
                            if spec.trait.rve_store_nt:
                                b.global_store_vN(
                                    C, o4, v4, 4, align=16, temporal_hint=_TH.STREAMING
                                )
                            else:
                                b.global_store_vN(C, o4, v4, 4, align=16)
                        continue
                    vec = b.vec_pack(
                        [b.vec_extract(hs[ni], v) for ni in range(n0, n0 + 8)],
                        storage_dtype,
                    )
                    c_n = b.add(_col0, b.const_i32(n0))
                    c_off = b.add(b.mul(c_m, N), c_n)
                    if batch_off_c is not None:
                        c_off = b.add(batch_off_c, c_off)
                    _rve_guarded_store(c_m, c_n, c_off, vec)
        if _rv_colguard is not None:
            _rv_colguard.__exit__(None, None, None)
        return

    def _store_cell(c_m: Value, c_n: Value, acc_h: Value, i: int) -> None:
        c_off = b.add(b.mul(c_m, N), c_n)
        if batch_off_c is not None:
            c_off = b.add(batch_off_c, c_off)
        h = b.vec_extract(acc_h, i)
        if fused_epilogue is not None:
            h = fused_epilogue.apply_scalar(b, h, c_m, c_n)
        _store_masked(c_m, c_n, c_off, h)

    _emit_mfma_acc_scatter(
        b,
        spec,
        lane,
        accs,
        block_warp_m_off,
        block_warp_n_off,
        c_per_lane,
        storage_dtype,
        _store_cell,
    )


def _emit_store_swapped_acc(
    b: IRBuilder,
    spec: UniversalGemmSpec,
    lane: Value,
    accs: Sequence[Value],
    m_base_off: Value,
    n_base_off: Value,
    c_per_lane: int,
    storage_dtype: Type,
    M: Value,
    N: Value,
    C: Value,
    *,
    batch_off_c: Optional[Value] = None,
) -> None:
    """Direct epilogue for ``mfma_swap_ab`` (the accumulator holds C^T).

    The MFMA output layout's row index (``row_in_atom``, the per-lane slot
    ramp) is now the N coordinate and its column (``lane % kCNLane``) the M
    coordinate, so each run of ``kCM1PerLane`` consecutive slots is
    ``kCM1PerLane`` consecutive N columns of one C row: one
    ``kCM1PerLane``-wide vector store (8 bytes for f16 / bf16) per run instead
    of ``kCM1PerLane`` scalar stores (the alignment is derived from the
    storage dtype's width). ``pad_n`` / split-K are rejected
    by :func:`is_valid_spec`; ``N % tile_n == 0`` keeps every run 8-byte
    aligned. ``pad_m`` guards the row.
    """
    from ...helpers.atoms import c_warp_params, make_c_warp_dstr_encoding, mfma_atom
    from ...helpers.distribution import make_static_tile_distribution

    t = spec.tile
    mfmas_m = t.mfmas_per_warp_m
    mfmas_n = t.mfmas_per_warp_n
    atom = mfma_atom(spec.data.dtype_a, t.warp_tile_m, t.warp_tile_n, t.warp_tile_k)
    _, _kc_mlane, kc_m1, kc_nlane = c_warp_params(atom)
    c_dist = make_static_tile_distribution(make_c_warp_dstr_encoding(atom))
    c_nlane = b.const_i32(kc_nlane)
    lane_mn = b.mod(lane, c_nlane)  # MFMA output column -> C row (M)
    p_lane = [b.div(lane, c_nlane), lane_mn]
    ramp: List[Value] = []
    for i in range(c_per_lane):
        ys = [b.const_i32(i // kc_m1), b.const_i32(i % kc_m1)]
        x_row, _x_col = c_dist.calculate_x(b, ys=ys, ps=[p_lane])
        ramp.append(x_row)  # MFMA output row -> C column (N)
    elem_bytes = type_bits(storage_dtype) // 8
    flat = 0
    for mi in range(mfmas_m):
        c_m = b.add(m_base_off, b.add(b.const_i32(mi * t.warp_tile_m), lane_mn))
        row_off = b.mul(c_m, N)
        if batch_off_c is not None:
            row_off = b.add(batch_off_c, row_off)
        for ni in range(mfmas_n):
            # acc_agpr: hand the atom out of the AGPRs at its use (as the
            # other epilogues do); identity without acc_agpr
            acc_h = (
                _acc_agpr_out(b, spec, accs[flat])
                if storage_dtype == F32
                else b.vec_cast_f32_to(
                    _acc_agpr_out(b, spec, accs[flat]), storage_dtype
                )
            )
            flat += 1
            n_tile = b.add(n_base_off, b.const_i32(ni * t.warp_tile_n))
            for g in range(c_per_lane // kc_m1):
                c_n = b.add(n_tile, ramp[g * kc_m1])
                vals = [b.vec_extract(acc_h, g * kc_m1 + r) for r in range(kc_m1)]
                vec = b.vec_pack(vals, storage_dtype)
                off = b.add(row_off, c_n)
                if spec.trait.pad_m:
                    with b.scf_if(b.cmp_lt(c_m, M)):
                        b.global_store_vN(C, off, vec, kc_m1, align=kc_m1 * elem_bytes)
                else:
                    b.global_store_vN(C, off, vec, kc_m1, align=kc_m1 * elem_bytes)


def _emit_epilogue_split_k(
    b: IRBuilder,
    spec: UniversalGemmSpec,
    accs: Sequence[Value],
    warp_m_idx: Value,
    warp_n_idx: Value,
    lane: Value,
    block_m_off: Value,
    block_n_off: Value,
    M: Value,
    N: Value,
    Cf32: Value,
    c_per_lane: int,
) -> None:
    """Split-K atomic-add epilogue.

    Each warp owns a per-lane ``<c_per_lane x f32>`` accumulator that is the
    partial product over this CTA's K-slice. We scatter every slot to its
    output ``(c_m, c_n)`` (the canonical MFMA layout, identical to the direct
    epilogue) and atomic-add the raw f32 value into ``Cf32[c_m, c_n]``. The
    split-K reduction across the ``split_k`` CTAs that share a ``(m_tile,
    n_tile)`` converges to the full f32 GEMM; the caller casts the workspace
    to the target dtype.

    The MFMA C-fragment ``(row_in_atom, col_in_atom)`` decode reuses the
    atom's ``CWarpDstrEncoding`` distribution exactly as
    :func:`_emit_mfma_acc_scatter` does, so the scattered output coords match
    the direct epilogue cell-for-cell -- only the per-cell write differs
    (f32 atomicrmw fadd vs bf16 global store).
    """
    from ...helpers.atoms import c_warp_params, make_c_warp_dstr_encoding, mfma_atom
    from ...helpers.distribution import make_static_tile_distribution

    t = spec.tile
    mfmas_m = t.mfmas_per_warp_m
    mfmas_n = t.mfmas_per_warp_n

    warp_m_off = b.mul(warp_m_idx, b.const_i32(mfmas_m * t.warp_tile_m))
    warp_n_off = b.mul(warp_n_idx, b.const_i32(mfmas_n * t.warp_tile_n))
    block_warp_m_off = b.add(block_m_off, warp_m_off)
    block_warp_n_off = b.add(block_n_off, warp_n_off)

    atom = mfma_atom(spec.data.dtype_a, t.warp_tile_m, t.warp_tile_n, t.warp_tile_k)
    _, _kc_mlane, kc_m1, kc_nlane = c_warp_params(atom)
    c_dist = make_static_tile_distribution(make_c_warp_dstr_encoding(atom))

    c_nlane = b.const_i32(kc_nlane)
    n_in_atom = b.mod(lane, c_nlane)
    m_blk = b.div(lane, c_nlane)
    p_lane = [m_blk, n_in_atom]

    row_in_atom: List[Value] = []
    col_in_atom: Optional[Value] = None
    for i in range(c_per_lane):
        ys = [b.const_i32(i // kc_m1), b.const_i32(i % kc_m1)]
        x_row, x_col = c_dist.calculate_x(b, ys=ys, ps=[p_lane])
        row_in_atom.append(x_row)
        col_in_atom = x_col

    pad_m = bool(spec.trait.pad_m)
    pad_n = bool(spec.trait.pad_n)

    flat = 0
    for mi in range(mfmas_m):
        base_m = b.add(block_warp_m_off, b.const_i32(mi * t.warp_tile_m))
        for ni in range(mfmas_n):
            # acc_agpr: one atom out of the AGPRs at a time (_acc_agpr_out)
            acc = _acc_agpr_out(b, spec, accs[flat])
            flat += 1
            c_n = b.add(
                block_warp_n_off,
                b.add(b.const_i32(ni * t.warp_tile_n), col_in_atom),
            )
            for i in range(c_per_lane):
                c_m = b.add(base_m, row_in_atom[i])
                c_off = b.add(b.mul(c_m, N), c_n)
                val = b.vec_extract(acc, i)
                checks: List[Value] = []
                if pad_m:
                    checks.append(b.cmp_lt(c_m, M))
                if pad_n:
                    checks.append(b.cmp_lt(c_n, N))
                if checks:
                    in_bounds = (
                        checks[0] if len(checks) == 1 else b.land(checks[0], checks[1])
                    )
                    with b.scf_if(in_bounds):
                        b.global_atomic_add_f32(Cf32, c_off, val)
                else:
                    b.global_atomic_add_f32(Cf32, c_off, val)


def _emit_epilogue_cshuffle(
    b: IRBuilder,
    spec: UniversalGemmSpec,
    _smem_unused: Value,  # placeholder for future reuse
    accs: Sequence[Value],
    warp_m_idx: Value,
    warp_n_idx: Value,
    lane: Value,
    block_m_off: Value,
    block_n_off: Value,
    M: Value,
    N: Value,
    C: Value,
    a_per_lane: int,
    b_per_lane: int,
    c_per_lane: int,
    *,
    batch_off_c: Optional[Value] = None,
    fused_epilogue=None,  # Optional[FusedEpilogue] from helpers.fuse
) -> None:
    """LDS-staged cshuffle epilogue.

    Pattern (matches CK's `cshuffle_epilogue.hpp`):
      1. Each warp converts its per-warp-tile accumulators (`<c_per_lane
         x float>`) to `<c_per_lane x half>`.
      2. Each warp stores them to LDS in a layout where consecutive
         lanes hold consecutive N-direction elements (the *output*
         layout, not the MFMA layout).
      3. Barrier.
      4. A subset of `STORE_VECS = (tile_m * tile_n) / store_vec_width`
         threads each read `<store_vec_width x half>` from LDS and
         issue one `<store_vec_width x half>` global store.

    For now we implement the 16x16 case and the 32x32 case using a
    distribution where every thread writes its own block-local row of
    `c_per_lane` halves into LDS at the canonical MFMA position, then
    a flat distribution of threads issues 4-wide global stores.

    The MFMA->LDS index math matches what we used in the default
    epilogue (which keeps the implementation honest: same lane->output
    mapping, just an extra LDS pass).
    """
    t = spec.tile
    storage_dtype = _storage_dtype(spec)
    mfmas_m = t.mfmas_per_warp_m
    mfmas_n = t.mfmas_per_warp_n

    # If a fused epilogue is attached (e.g. BiasAdd, ReLU), give it a
    # chance to declare any extra kernel params (bias pointer, …). It
    # caches the SSA values internally so apply_vec / apply_scalar
    # can reach them at the per-element transform site below. The
    # ``record_runtime`` call lets residual-style ops use the
    # contiguous-(M, N) layout's row stride (= N) without an extra
    # i32 kernel param.
    if fused_epilogue is not None:
        fused_epilogue.declare_params(b)
        record_runtime = getattr(fused_epilogue, "record_runtime", None)
        if record_runtime is not None:
            record_runtime(b, N=N)

    # LDS staging tile: tile_m x tile_n of output storage dtype.
    # ``cshuffle_no_alias`` marks it exclusive so the smem-pool packer gives it
    # its own byte range instead of aliasing the A/B staging bytes.
    Cs = b.smem_alloc(
        storage_dtype,
        [t.tile_m, t.tile_n],
        name_hint="C_smem",
        exclusive=spec.trait.cshuffle_no_alias,
    )

    # ---- step 0: reuse barrier. ----
    # The common-LDS packer aliases this C staging tile onto the A/B
    # staging bytes (they are non-interfering in program order, so the
    # liveness packer places C at the A/B pool offset). The double-buffered
    # and prefetched mainloops (``_emit_kloop_db`` / ``_emit_kloop_prefetch``)
    # end with the tail-tile MFMA reading A/B from LDS *after* their last
    # drain barrier and emit no trailing barrier, so without a barrier here a
    # fast wave's first C ``ds_write`` would clobber A/B bytes a slow wave is
    # still reading for its tail MFMA -- a cross-wave WAR on the aliased pool.
    # (The single-buffer ``_emit_kloop_simple`` already has a trailing in-loop
    # barrier, which makes this one redundant-but-harmless there.)
    #
    # With ``cshuffle_no_alias`` the C tile has its own exclusive LDS bytes that
    # never overlap A/B, so this cross-wave WAR cannot occur and the barrier is
    # elided -- the small-tile latency win. The step-3 C-write->C-read barrier
    # below is a genuine intra-epilogue RAW and always stays.
    if not spec.trait.cshuffle_no_alias:
        b.sync()

    warp_m_off = b.mul(warp_m_idx, b.const_i32(mfmas_m * t.warp_tile_m))
    warp_n_off = b.mul(warp_n_idx, b.const_i32(mfmas_n * t.warp_tile_n))

    # ---- step 1+2: warp accs -> LDS at the MFMA layout. ----
    #
    # For both 16x16 and 32x32 we hoist the per-tile (warp + mi *
    # warp_tile_m, warp + ni * warp_tile_n + n_in_atom) bases outside
    # the per-element loop. The MFMA output layout pins consecutive
    # lanes to consecutive N columns and scatters the per-lane
    # c_per_lane slots across M rows, so the smem store itself stays
    # scalar (n=1). What we can hoist is the address arithmetic:
    # every (mi, ni, i) re-derived the warp + mi offset in the
    # original form.
    def _smem_cell(ld_m: Value, ld_n: Value, acc_h: Value, i: int) -> None:
        h = b.vec_extract(acc_h, i)
        b.smem_store_vN(Cs, [ld_m, ld_n], h, n=1)

    # Same MFMA accumulator -> (row, col) layout as the default epilogue, but
    # the base offsets are warp-relative (the block offset is applied at the
    # wide global store in step 4) and each cell writes to the LDS staging
    # tile instead of global. The shared scatter keeps the 16x16 / 32x32 row
    # math + hoisting in one place.
    _emit_mfma_acc_scatter(
        b,
        spec,
        lane,
        accs,
        warp_m_off,
        warp_n_off,
        c_per_lane,
        storage_dtype,
        _smem_cell,
        n_base_first=True,
    )

    # ---- step 3: barrier. ----
    b.sync()

    # ---- step 4: wide global stores. ----
    # STORE_VECS = (tile_m * tile_n) / store_vec. We pick store_vec as
    # wide as we can naturally align (8 halves = 16 B).
    threads = spec.block_size
    store_vec = 8
    while store_vec > 1 and (
        (t.tile_n % store_vec != 0)
        or ((t.tile_m * t.tile_n) // store_vec < threads)
        or (((t.tile_m * t.tile_n) // store_vec) % threads)
    ):
        store_vec //= 2

    if store_vec == 1:
        # Pathological: fall back to scalar stores.
        store_vec = 1

    tid = b.thread_id_x()
    c_threads = b.const_i32(threads)
    c_tile_n_div_vec = b.const_i32(t.tile_n // store_vec)
    vecs_per_thread = (t.tile_m * t.tile_n // store_vec) // threads
    pad_m = bool(spec.trait.pad_m)
    pad_n = bool(spec.trait.pad_n)

    # Coalesced (row, col_v, col) decode of a per-thread vec index for the
    # wide C store. Takes the already-built ``vec_idx`` so the emitted SSA
    # matches the inline order verbatim.
    def _vec_rc(vec_idx: Value) -> Tuple[Value, Value, Value]:
        row = b.div(vec_idx, c_tile_n_div_vec)
        col_v = b.mod(vec_idx, c_tile_n_div_vec)
        col = b.mul(col_v, b.const_i32(store_vec)) if store_vec > 1 else col_v
        return row, col_v, col

    for e in range(vecs_per_thread):
        vec_idx = b.add(b.mul(b.const_i32(e), c_threads), tid)
        row, col_v, col = _vec_rc(vec_idx)

        c_m = b.add(block_m_off, row)
        c_n = b.add(block_n_off, col)
        c_off = b.add(b.mul(c_m, N), c_n)
        if batch_off_c is not None:
            c_off = b.add(batch_off_c, c_off)

        # Out-of-bounds guard: when ``pad_m`` / ``pad_n`` is set the
        # caller may launch with grid dims that round up past the
        # logical ``M`` / ``N``. The cshuffle staging is sized to the
        # full tile (always in-bounds for LDS), but the global store
        # must skip rows / columns past the logical extent. For
        # non-vector-aligned N tails we fall back to element-granular
        # guarded stores so valid columns at the start of the final
        # vector are not dropped.
        in_bounds: Optional[Value] = None
        if pad_m or pad_n:
            checks: List[Value] = []
            if pad_m:
                checks.append(b.cmp_lt(c_m, M))
            if pad_n:
                if store_vec == 1:
                    checks.append(b.cmp_lt(c_n, N))
                else:
                    c_n_last = b.add(c_n, b.const_i32(store_vec - 1))
                    checks.append(b.cmp_lt(c_n_last, N))
            in_bounds = checks[0] if len(checks) == 1 else b.land(checks[0], checks[1])

        if store_vec == 1:
            h = _load_smem_scalar(b, Cs, row, col, storage_dtype)
            if fused_epilogue is not None:
                # Treat the scalar as a length-1 vector for op uniformity.
                # The fused epilogue may transform the value (bias-add,
                # activation, scale, ...); we then unpack back to scalar
                # before the global store.
                h = fused_epilogue.apply_scalar(b, h, c_m, c_n)
            if in_bounds is not None:
                with b.scf_if(in_bounds):
                    b.global_store(C, c_off, h, align=2)
            else:
                b.global_store(C, c_off, h, align=2)
        else:
            hv = _load_smem_vec(b, Cs, row, col, store_vec, storage_dtype)
            if pad_n:
                for i in range(store_vec):
                    c_n_i = b.add(c_n, b.const_i32(i)) if i else c_n
                    c_off_i = b.add(c_off, b.const_i32(i)) if i else c_off
                    h = b.vec_extract(hv, i)
                    if fused_epilogue is not None:
                        h = fused_epilogue.apply_scalar(b, h, c_m, c_n_i)
                    checks: List[Value] = []
                    if pad_m:
                        checks.append(b.cmp_lt(c_m, M))
                    checks.append(b.cmp_lt(c_n_i, N))
                    elem_in_bounds = (
                        checks[0] if len(checks) == 1 else b.land(checks[0], checks[1])
                    )
                    with b.scf_if(elem_in_bounds):
                        b.global_store(C, c_off_i, h, align=2)
            else:
                if fused_epilogue is not None:
                    hv = fused_epilogue.apply_vec(b, hv, c_m, c_n, n_elems=store_vec)
                if in_bounds is not None:
                    with b.scf_if(in_bounds):
                        b.global_store_vN(C, c_off, hv, store_vec)
                else:
                    b.global_store_vN(C, c_off, hv, store_vec)


def _load_smem_scalar(
    b: IRBuilder, smem: Value, row: Value, col: Value, dtype: Type
) -> Value:
    # We expose vector loads but not a scalar half load from smem yet;
    # the v=1 path is rare. Emit as a 2-half vector and extract index 0.
    v = b.smem_load_vN(smem, row, col, dtype=dtype, n=2)
    return b.vec_extract(v, 0)


def _load_smem_vec(
    b: IRBuilder, smem: Value, row: Value, col: Value, n: int, dtype: Type
) -> Value:
    if dtype == F16 and n == 4:
        return b.smem_load_v4_f16(smem, row, col)
    return b.smem_load_vN(smem, row, col, dtype=dtype, n=n)


# ---------------------------------------------------------------------
# Cartesian-product enumeration matching CK's default_config.json
# ---------------------------------------------------------------------


# Named lever bundles the config enumerator crosses with every tile / warp /
# atom / pipeline / epilogue point. On gfx950 the levers interact: DTL alone
# does not pay, while DTL + dtl_prefetch + lds_swizzle + dtl_cache_b=ALL +
# chiplet swizzle does. A funnel that sweeps tiles with no levers and then
# adds one lever at a time to the global top-k never reaches the combination,
# so the bundles are BASE axes here: each one is a whole lever set.
# Invalid combinations (e.g. a compv4-only bundle on compv3) are dropped by
# ``is_valid_spec`` like any other point.
LEVER_BUNDLES = {
    # Plain body (the historical enumeration).
    "none": {},
    # Element-granular LDS XOR swizzle on the VGPR-staged body.
    "swizzle": {"lds_swizzle": True},
    # compv4 VGPR-staged AB double buffer made explicit (+ swizzle).
    "compv4_db_swizzle": {"ab_double_buffer": True, "lds_swizzle": True},
    # The DTL ping-pong bundle:
    # direct-to-LDS + depth-1 prefetch + swizzle + A and B CACHE_ALL.
    "dtl_pingpong": {
        "direct_to_lds": True,
        "dtl_prefetch": True,
        "lds_swizzle": True,
        "dtl_cache_a": 0,
        "dtl_cache_b": 0,
    },
    # The DTL ping-pong + chiplet-aware grid swizzle (wgm 8, 64 WGs per XCD chunk).
    "dtl_pingpong_chiplet": {
        "direct_to_lds": True,
        "dtl_prefetch": True,
        "lds_swizzle": True,
        "dtl_cache_a": 0,
        "dtl_cache_b": 0,
        "chiplet_swizzle": True,
        "chiplet_wgm": 8,
        "chiplet_chunk_size": 64,
    },
}

DEFAULT_LEVER_BUNDLES: Tuple[str, ...] = tuple(LEVER_BUNDLES)


def all_dispatcher_configs(
    *,
    tile_m: Sequence[int] = (128, 256),
    tile_n: Sequence[int] = (128, 256),
    tile_k: Sequence[int] = (32, 64),
    warp_m: Sequence[int] = (2, 4),
    warp_n: Sequence[int] = (2, 4),
    warp_k: Sequence[int] = (1,),
    warp_tile: Sequence[Tuple[int, int, int]] = (
        (16, 16, 16),
        (32, 32, 8),
        (32, 32, 16),
        (16, 16, 32),
    ),
    pipeline: Sequence[Pipeline] = ("compv3", "compv4"),
    scheduler: Sequence[Scheduler] = ("intrawave",),
    epilogue: Sequence[Epilogue] = ("default", "cshuffle"),
    pad: Sequence[bool] = (False,),
    persistent: Sequence[bool] = (False,),
    levers: Sequence[str] = ("none",),
    wave_size: int = 64,
    name_prefix: str = "rocke_universal",
    arch: str = "gfx950",
) -> Iterator[UniversalGemmSpec]:
    """Yield every valid `(TileSpec, TraitSpec)` combo on this arch.

    Defaults mirror `dispatcher/codegen/default_config.json` for fp16
    (which uses `pipeline=[compv4]`, `epilogue=[cshuffle]`). We accept
    the broader space so a sweep can compare CK's choices against
    alternatives (e.g. `mem` for memory-bound shapes).

    ``levers`` names :data:`LEVER_BUNDLES` entries crossed with every point
    (default ``("none",)``: the historical plain-body enumeration, so existing
    callers keep their config count). Pass ``levers=DEFAULT_LEVER_BUNDLES`` (or
    any subset) to cross the lever bundles as base axes, so a sweep cannot miss
    a lever combination that only pays off as a set. A bundle's specs are named
    ``<name_prefix>_<bundle>`` (the plain ``none`` bundle keeps
    ``name_prefix``) so bundle variants stay distinguishable in labels and run
    plans.
    """
    for lv in levers:
        if lv not in LEVER_BUNDLES:
            raise ValueError(
                f"unknown lever bundle {lv!r}; known: {sorted(LEVER_BUNDLES)}"
            )
    for tm in tile_m:
        for tn in tile_n:
            for tk in tile_k:
                for wm in warp_m:
                    for wn in warp_n:
                        for wk in warp_k:
                            for wt in warp_tile:
                                for pl in pipeline:
                                    for sc in scheduler:
                                        for ep in epilogue:
                                            for p in pad:
                                                for pers in persistent:
                                                    for lv in levers:
                                                        spec = UniversalGemmSpec(
                                                            name=(
                                                                name_prefix
                                                                if lv == "none"
                                                                else f"{name_prefix}_{lv}"
                                                            ),
                                                            tile=TileSpec(
                                                                tile_m=tm,
                                                                tile_n=tn,
                                                                tile_k=tk,
                                                                warp_m=wm,
                                                                warp_n=wn,
                                                                warp_k=wk,
                                                                warp_tile_m=wt[0],
                                                                warp_tile_n=wt[1],
                                                                warp_tile_k=wt[2],
                                                            ),
                                                            trait=TraitSpec(
                                                                pipeline=pl,
                                                                scheduler=sc,
                                                                epilogue=ep,
                                                                pad_m=p,
                                                                pad_n=p,
                                                                pad_k=p,
                                                                persistent=pers,
                                                                **LEVER_BUNDLES[lv],
                                                            ),
                                                            wave_size=wave_size,
                                                        )
                                                        ok, _ = is_valid_spec(
                                                            spec, arch=arch
                                                        )
                                                        if ok:
                                                            yield spec

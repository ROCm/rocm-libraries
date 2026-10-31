# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Implicit-GEMM convolution kernel instance (NHWC × KYXC -> NHWK).

This is the standard implicit-GEMM convolution problem authored entirely
through the rocke IR + coordinate-transform DAG. It mirrors how CK
Tile expresses convolution (`tensor_descriptor` with
`unmerge_transform` for `(m -> n, ho, wo)`, `embed_transform` for
`(ho, y -> hi)`, and `pad_transform` for boundary checks), and the
GEMM body is the same compv4-style pipeline we use for square GEMM
in `gemm_universal.py`. The only kernel-authoring change vs GEMM is
how the A tile's address is computed.

Authoring style (this is what the kernel writer types):

    spec = ImplicitGemmConvSpec(
        problem=ConvProblem(N=8, Hi=56, Wi=56, C=64,
                            K=64, Y=3, X=3,
                            sH=1, sW=1, pH=1, pW=1, dH=1, dW=1),
        tile_m=64, tile_n=64, tile_k=64,
        warp_m=2, warp_n=2,
        warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
    )
    kernel = build_implicit_gemm_conv(spec)

Internally, A's per-element address is computed via a transform DAG:

    A_desc = TensorDescriptor.naive('A', [N, Hi, Wi, C], coord_names=
        ['n', 'hi', 'wi', 'c'])
        .transform(unmerge('m', into=['n','ho','wo'], dims=[N,Ho,Wo]))
        .transform(embed(['ho','y'], 'hi', strides=(sH,dH), offset=-pH,
                          lo=0, hi=Hi))
        .transform(embed(['wo','x'], 'wi', strides=(sW,dW), offset=-pW,
                          lo=0, hi=Wi))
        .transform(unmerge('k', into=['y','x','c'], dims=[Y,X,C]))

At every per-thread A-tile load we call `A_desc.offset(b, m=m_val,
k=k_val)` which emits the bounds-checked NHWC offset for that
implicit-GEMM (m, k) point — the same SSA dataflow a hand-written
implicit-GEMM kernel would compute.

B (KYXC weight) and D (NHWK output) use naive descriptors with extra
`unmerge` transforms for the output store (so the epilogue can write
NHWK from MFMA's (m_in_tile, n_in_tile) layout).

Target: CK Tile's `cktile_fixed_lean` reference for this shape
(`N=8, Hi=Wi=56, C=K=64, Y=X=3`) reaches ~250 TFLOPS in CUDA-graph
mode. We aim to beat that on the same shape.

Shared dataclasses and low-level helpers live in
:mod:`._conv_implicit_gemm_common` (underscore-prefixed, internal).
This module re-exports the shared helpers to keep existing callers stable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace as dc_replace
from typing import Callable, List, Optional, Sequence, Tuple

from rocke.core.ir import (
    VectorType,
    BF16,
    F16 as _F16,
    F32,
    I32,
    IRBuilder,
    KernelDef,
    PtrType,
    Type,
    Value,
)
from rocke.helpers.atoms import MfmaAtom, mfma_atom
from rocke.helpers.epilogues import CShuffleEpilogue, DirectEpilogue
from rocke.helpers.geometry import WarpGrid
from rocke.helpers.layouts import LdsLayout
from rocke.helpers.loads import AsyncTileLoader, CoalescedTileLoader
from rocke.helpers.mfma_gemm_inner import decode_mfma_lanes
from rocke.helpers.pipeline import SoftwarePipeline
from rocke.helpers.schedule import DS_READ, DS_WRITE, MFMA, VMEM_READ, SchedulePolicy
from rocke.helpers.tensor_view import (
    make_buffer_resource,
)
from rocke.helpers.transforms import (
    TensorDescriptor,
    pad,
    unmerge_magic,
)
from kernels.common.conv_m_band import emit_m_band_order
from kernels.common.conv_abi import conv_fwd_problem_block

# ImplicitGemmConvSpec.sched_strategy -> AMDGPU machine-scheduler strategy.
_SCHED_STRATEGIES = {1: "max-ilp", 2: "max-memory-clause", 3: "iterative-ilp",
                     4: "iterative-minreg", 5: "iterative-maxocc"}
from kernels.common._conv_implicit_gemm_common import (  # noqa: F401 — re-exported for callers
    ConvAccumulatorEpilogue,
    ConvDataSpec,
    ConvProblem,
    _apply_accumulator_epilogue,
    _choose_load_vec_for,
    coalesced_load_reason,
    vector_width_reason,
    _emit_frag_smem_load,
    _emit_mfma,
    _emit_smem_load,
    _ir_dtype,
    _build_wavelet_loaders,
    compute_wavelet_epi_barriers,
    emit_param_block,
    emit_direct_epilogue_wmma,
    emit_wavelet_kloop_dynamic,
    make_a_descriptor,
    make_a_descriptor_dynamic,
    make_b_descriptor_dynamic,
    make_d_descriptor_dynamic,
    magic_divmod,
    mul_u24,
)


def _choose_load_vec(spec: ImplicitGemmConvSpec) -> int:
    """Thin shim keeping the forward-conv call sites unchanged."""
    return _choose_load_vec_for(
        spec.tile_m, spec.tile_n, spec.tile_k, spec.block_size, spec.data.dtype_a
    )


def _sync_load_vecs(spec: "ImplicitGemmConvSpec") -> Tuple[int, int]:
    """``(load_vec_a, load_vec_b)`` of the sync (CoalescedTileLoader) path.

    An explicit ``vector_size_a`` / ``vector_size_b`` is used verbatim;
    otherwise the per-group channel default, clamped by the tile-geometry
    maximum so the loader's ``(tile_rows * tile_cols / vec) % block_size == 0``
    invariant holds (e.g. when tile_n is small relative to block_size). Shared
    by the builder and :func:`is_valid_spec`, which rejects an explicit width
    that breaks that invariant.

    Under ``group_merge`` the defaults come from the *merged* problem: on
    depthwise that reads (Gm, Gm) instead of (1, 1), so A's default goes from 1
    to min(Gm, 8) -- the merged A tile really does address Gm consecutive NHWC
    channels, because the merged group's ``c_in_group`` axis *is* the group
    index and NHWC stores groups contiguously. B is the exception, see below.
    """
    p = spec.merged_problem
    _def_vec_a, _def_vec_b, _ = ImplicitGemmConvSpec.default_vector_sizes(
        p.cpg, p.kpg, spec.data.dtype_a
    )
    _tile_vec = _choose_load_vec(spec)
    _def_vec_a = min(_def_vec_a, _tile_vec)
    _def_vec_b = min(_def_vec_b, _tile_vec)
    if spec.group_merge > 1:
        # B does NOT widen under merge, and taking the (Gm, Gm) default above
        # would be actively wrong. The merged B tile is only 1/Gm dense: along
        # the reduction axis consecutive ``k_gemm`` differ in the merged group
        # ``g_k``, and all but the diagonal one are masked off. A vector load
        # would fetch a whole run under a single predicate, so it cannot express
        # a per-element mask -- it would either drop the one valid element or
        # keep Gm-1 invalid ones. Weights are negligible for depthwise (K*Y*X
        # elements against N*H*W*C of activations), so scalar B costs nothing.
        _def_vec_b = 1
    load_vec_a = spec.vector_size_a if spec.vector_size_a is not None else _def_vec_a
    load_vec_b = spec.vector_size_b if spec.vector_size_b is not None else _def_vec_b
    return load_vec_a, load_vec_b


@dataclass(frozen=True)
class ImplicitGemmConvSpec:
    """One concrete implicit-GEMM convolution kernel configuration.

    The geometry conventions match `gemm_universal.UniversalGemmSpec`:
      - `tile_m x tile_n x tile_k` is the per-block tile.
      - `warp_m x warp_n` is the warp grid.
      - `warp_tile_m x warp_tile_n x warp_tile_k` is the MFMA atom.

    Pipeline options (mirror CK's compv4 family):
      - `pipeline="mem"`      : single-buffer LDS, no scheduler hints
      - `pipeline="compv3"`   : single-buffer LDS + sched_group_barrier
                                interleave hints (overlap MFMA/DS_read)
      - `pipeline="compv4"`   : double-buffer LDS (ping-pong A_smem/B_smem)
                                + sched hints + s_setprio to push the K-loop
                                into compute steady state
      - `pipeline="wavelet"`  : load/math wave specialization (CK Tile PR
                                #8009). **gfx1250/WMMA only** — rejected by
                                ``is_valid_spec`` on MFMA/CDNA targets.
                                Appends ``num_load_waves`` extra waves to
                                the workgroup; those waves handle all
                                DRAM→LDS transfers while the standard
                                warp_m × warp_n waves run WMMA exclusively.
                                Single-buffer LDS shared by both roles.
                                Uses scf_if_else (LLVM br i1); separate
                                VMEM/WMMA issue slots provide hardware
                                concurrency without exec-masking.
    Epilogue options:
      - `epilogue="default"`  : per-lane scalar dtype_d stores via the D
                                descriptor; correctness-first.
      - `epilogue="cshuffle"` : LDS-stage the accumulators in MFMA layout,
                                then re-read in coalesced (row-major)
                                layout for wide-vector global stores
                                (runbook §9.3 — the single largest perf
                                lever once the K loop is bandwidth-bound).

    Memory-pipeline options:
      - `async_dma=False`     : (default) classic
                                `buffer_load_vN_f16 -> register ->
                                smem_store_vN_f16` path. Adds K-padded
                                LDS (`K_pad = block_k + 8`) to avoid
                                ds_read bank conflicts.
      - `async_dma=True`      : direct DRAM->LDS via
                                `raw_ptr_buffer_load_lds` (runbook §6.3).
                                The intrinsic writes lane-contiguous LDS,
                                so the LDS layout must be plain
                                `[block_m, block_k]` (no K-pad). Consumers
                                still emit standard 2D ds_reads; LDS
                                bank-conflict avoidance moves into the
                                consumer's address arithmetic (XOR
                                swizzle) if it becomes the next
                                bottleneck. Place `s_waitcnt(vmcnt=0)`
                                before the MFMA phase.
      - `lds_k_pad=None`      : default LDS K-stride policy. Sync path
                                uses `+8` when `block_k >= 16`; async
                                path uses `+0` because
                                `raw_ptr_buffer_load_lds` writes a
                                lane-contiguous packed tile. Set this
                                explicitly to tune or isolate LDS bank
                                conflict effects in sweeps.
    """

    problem: ConvProblem
    name: str = "conv_igemm"
    data: ConvDataSpec = field(default_factory=ConvDataSpec)

    tile_m: int = 64
    tile_n: int = 64
    # tile_k=64 + the 32x32x16 atom is the measured-best default on gfx950:
    # for the bake-off shape (N8 C64 K64) it ties the old tk128/16x16x32
    # default, and for compute-bound shapes (e.g. N16 C256 K256) it is ~1.4x
    # faster (619 vs 445 TFLOPS, gfx950). All shipped callers (bake_off,
    # tests, probes, hip_lowering_parity, verify_dsl_docs) already override to
    # this config; the stale tk128/16x16x32 default only penalised callers who
    # relied on the dataclass defaults.
    tile_k: int = 64

    warp_m: int = 2
    warp_n: int = 2

    warp_tile_m: int = 32
    warp_tile_n: int = 32
    warp_tile_k: int = 16

    wave_size: int = 64

    pipeline: str = "mem"
    epilogue: str = "default"
    async_dma: bool = False
    unroll_k: bool = False  # NEW: Clean Python-level K-loop unrolling
    lds_k_pad: Optional[int] = None
    # Per-operand vector widths (elements). ``None`` means auto-select via the
    # shared ``choose_load_vec`` / ``CShuffleEpilogue.from_grid`` heuristics.
    vector_size_a: Optional[int] = None
    vector_size_b: Optional[int] = None
    vector_size_c: Optional[int] = None
    lds_layout: Optional[LdsLayout] = None
    # Chiplet-aware grid swizzle (multi-XCD L2 locality). When True,
    # the kernel flattens its 2D blockIdx into a linear WGID, runs it
    # through ``chiplet_aware_super_tile`` (compile-time variant — conv
    # tile counts are derived from the problem shape so they are known
    # statically), and uses the remapped (pid_m, pid_n) for tile offsets.
    chiplet_swizzle: bool = False
    chiplet_wgm: int = 8
    chiplet_num_xcds: int = 8
    chiplet_chunk_size: int = 64
    # AMDGPU occupancy hint: emits ``amdgpu-waves-per-eu`` on the
    # kernel attribute list. ``None`` keeps the backend's default.
    waves_per_eu: Optional[int] = None
    # P87: K0/K1 split for implicit-GEMM conv. Set ``k0_k1_split=True``
    # to drive :class:`rocke.helpers.loads.CoalescedTileLoader`'s
    # ``inner_dim`` parameter (P33) so the loader processes whole C
    # rows contiguously and the MFMA loop iterates ``kk`` over K1
    # only. ``None`` (default) keeps the legacy flat-K behaviour.
    k0_k1_split: bool = False
    # P86: grouped convolution. ``groups > 1`` uses the descriptor
    # DAG's ``unmerge('group', into=...)`` to recover the per-group
    # `(C/groups, K/groups)` slabs so each group's GEMM stays
    # within its own slab. The implementation mirrors CK Tile's
    # ``GroupedConvolutionForward`` shape; for ``groups == 1`` (the
    # default) the descriptor reduces to the flat-conv form and the
    # kernel emits the same code as before.
    groups: int = 1
    # Static accumulator epilogue used by the gfx950 deep-fusion prototype.
    # It composes simple fp32 VALU transforms directly on MFMA accumulator
    # fragments before the existing direct/cshuffle store path.
    acc_epilogue: ConvAccumulatorEpilogue = ConvAccumulatorEpilogue()
    # cshuffle epilogue LDS aliasing (same knob as UniversalGemmSpec.trait).
    # False (default): the smem-pool packer aliases the cshuffle C tile onto the
    # A/B staging bytes (pool = max(ab, c)) with a step-0 reuse barrier. True:
    # C gets its own LDS bytes (pool = ab + c) and the barrier is elided ->
    # lower small-tile latency, more LDS. Only affects epilogue == "cshuffle".
    cshuffle_no_alias: bool = False
    # Number of extra load waves appended to the workgroup for pipeline="wavelet".
    # Load waves handle all DRAM→LDS transfers; math waves (warp_m × warp_n) handle
    # LDS reads + MFMA only. The launch block size becomes
    # block_size + num_load_waves * wave_size. Ignored for all other pipelines.
    num_load_waves: int = 4

    # Number of conv groups a single workgroup computes ("Gm"; CK calls it
    # NumGroupsToMerge). 1 is grid-per-group, the historical behaviour.
    #
    # A depthwise group has cpg == kpg == 1, so its per-group forward GEMM is
    # (N*Ho*Wo) x 1 x (Y*X) over single channels of an NHWC tensor. The free
    # axis (GEMM-N = kpg) is one element wide and the stride-1 axis (c) sits
    # *inside* the reduction, so both the A load and the D store degenerate to
    # one element and every MFMA wastes all but one of its N lanes.
    #
    # Merging Gm adjacent groups makes the group index the fastest-varying
    # factor of GEMM-N and GEMM-K:
    #
    #     M      = N*Ho*Wo          (unchanged -- the group does NOT go on M)
    #     N_gemm = Gm               (the GEMM-N index *is* the merged group g_n)
    #     K_gemm = Y*X*Gm           (decodes to (y, x, g_k), g_k innermost)
    #
    # A then reads ``in[n][hi][wi][gm_grp*Gm + k % Gm]`` -- a run of Gm
    # consecutive NHWC channels -- and D writes ``out[n][ho][wo][gm_grp*Gm +
    # g_n]``, likewise contiguous. Only the diagonal ``g_k == g_n`` is real
    # work; the mask rides on the **B (weight) load**, where an invalid element
    # becomes the loader's OOB sentinel and the buffer resource returns a
    # hardware zero. So the off-diagonal costs no memory traffic and the store
    # needs no mask at all -- unlike wgrad (and CK), whose merged output is a
    # diagonal gather. This is why fwd puts Gm on N and K rather than copying
    # wgrad's "Gm on M and N".
    #
    # Total MFMA issue is unchanged vs Gm == 1: grid.z shrinks by Gm exactly as
    # K_gemm grows by Gm. The redundant MACs land in the previously idle N lanes.
    #
    # Default 1 emits no merge-specific IR and keeps the kernel name. (The
    # cshuffle store width derived from kpg is a separate Gm=1 change.)
    group_merge: int = 1
    # Register-staged global prefetch (CK's "prefetch stage 2"): the next K
    # tile's global loads are issued into registers *before* the current
    # tile's ds_read + MFMA work and written to LDS after it, so the DRAM/L2
    # latency hides behind the MFMAs of the same wave instead of only behind
    # other resident workgroups. Without it, every load phase fuses
    # buffer_load -> s_waitcnt -> ds_write and the wave stalls on the load.
    # With ``unroll_k`` the next tile goes to the alternate LDS buffer (one
    # barrier per tile); without it a single buffer is reused behind a second
    # barrier (half the A/B LDS, so more resident workgroups). Costs the
    # registers of one A + B tile slice per thread. Sync-load path only.
    prefetch_regs: bool = False
    # Split-K over the reduction (Z*Y*X*cpg). ``split_k > 1`` builds the
    # split binary: the degree is a launch parameter (``ks`` = slice width, a
    # multiple of tile_k, and ``ks_count`` = number of slices, appended to the
    # kernargs), gridDim.z = groups * ks_count, and every slice atomic-adds its
    # partial tile into D, which the caller zeroes first. For problems whose
    # output has too few tiles to fill the device (small spatial extent, long
    # reduction). 1 (default) builds the classic kernel, unchanged.
    split_k: int = 1
    # Decode the reduction index once per K tile on the 3-D split-addressing
    # path instead of once per loaded element: when cpg % tile_k == 0 a tile
    # never straddles a filter tap, so (z, y, x) and the tile's first channel
    # come from the wave-uniform tile origin (scalar ALU) and each element
    # only adds its column. Unlike ``uniform_k_decode`` (below) the binary
    # carries no fallback loop: it is only correct for problems with
    # cpg % tile_k == 0 (the AOT cache checks it per problem). Pairs with the
    # ``prefetch_regs`` / ``split_k`` loops.
    split_addr_k_decode: bool = False
    # Split-K on the tail only (``split_k > 1`` binaries): a ``full_tiles``
    # kernarg says how many output tiles, in launch order, run the whole
    # reduction in slice 0 (their other slices exit at once; plain store);
    # only the tiles past it are split ``ks_count`` ways and atomic-added
    # into D. Fills the last, partial wave of workgroups of a grid that is a
    # few waves deep (s5/s6: 1024 tiles over 456 slots). ``xcd_remap``
    # permutes the full tiles only, so the tail stays the last output tiles
    # in row-major tile order and the host zeroes only the D rows from tile
    # row ``full_tiles // num_pid_n`` on (:func:`fwd_split_tail_zero_rows`).
    split_tail: bool = False
    # split_tail on a compact grid: (full_tiles + tail_tiles * ks_count, 1,
    # groups) workgroups instead of (tiles, 1, groups * ks_count), so the
    # full_tiles * (ks_count - 1) slices that would only exit are never
    # dispatched (they sit in front of the tail's slices). The kernel maps
    # its flat id to the same (tile, slice) as the default launch: id <
    # full_tiles -> full tile id, else tail slice-major. Tags ``stc``.
    split_tail_compact: bool = False
    # K-loop instruction scheduling (wave64 MFMA loops; all default off, which
    # leaves the emitted kernel and its name unchanged):
    # * ``lds_ahead``: the LDS fragment reads of kk step kk+1 issue before the
    #   MFMAs of step kk, so their latency hides behind those MFMAs instead of
    #   an lgkmcnt(0) wait per step. reg_prefetch / prefetch_regs loops.
    # * ``defer_mfma``: the MFMAs of the last ``defer_mfma`` kk steps of a tile
    #   are emitted after the barrier that retires the tile's LDS reads,
    #   interleaved 2 MFMAs : 1 ds_write with the next tile's LDS store (CK V3
    #   "write phase"); both barriers become LDS-only so the store waits per
    #   ds_write for its own load. reg_prefetch=1 loop only.
    # * ``vmem_interleave``: the next tile's buffer_loads are spread one per
    #   ``vmem_interleave`` MFMAs over the MFMA phase (sched_group_barrier)
    #   instead of issuing back to back ahead of it, where at 1-2 waves/SIMD
    #   they back up in the TA queue in front of the first MFMA.
    #   reg_prefetch (1, 2) and single-buffer prefetch_regs loops, hint-emitting
    #   pipelines (compv3/compv4).
    lds_ahead: bool = False
    defer_mfma: int = 0
    vmem_interleave: int = 0
    # k-packed LDS fragment reads (CK's KPack): the MFMA k order inside a tile
    # is permuted so that each lane's A row / B column elements of two
    # consecutive kk steps are contiguous, and both are read with one 16-byte
    # ds_read_b128 instead of two 8-byte reads. With lds_xor the XOR key is
    # taken on the row's 128-byte (32-bank) span, so the 8 lanes of one
    # b128 read phase reach 8 distinct 16-byte bank groups. Same reduction,
    # another summation order. uniform_k_decode / reg_prefetch loops, an even
    # number of kk steps per tile, not with a_direct. Default False; tags
    # ``kprd``.
    lds_kpack_read: bool = False
    # reg_prefetch=1 loop: both per-tile barriers drain LDS only (lgkmcnt(0) +
    # s_barrier) instead of also waiting for every in-flight global load
    # (vmcnt(0)): a wave reaches the barrier that retires the tile's ds_reads
    # with its prefetch still in flight, and each LDS write waits only for its
    # own load. Default False; tags ``lbar``.
    rpf_lds_barrier: bool = False
    # reg_prefetch=1 loop: issue the next tile's global reads after the MFMA
    # phase's first LDS fragment reads instead of before them, so the first
    # MFMAs' operands are not queued behind the buffer_loads (Triton's order).
    # Default False; tags ``flate``.
    rpf_fetch_late: bool = False
    # Input reuse across the filter's x taps (fwd x-halo; 0 = off). On a 3-D
    # problem with pD == pH == pW == 0, sW == dW == 1 and X == 3, output pixel
    # m at tap (z, y, x) reads input pixel lin(m) + x + tap(z, y), and a tile
    # of consecutive output pixels inside one (n, d) plane reads one
    # contiguous range of input pixels at tap (z, y, 0): its span is tile_m
    # plus (Wi - Wo) per output row it crosses. So the loop runs over
    # (chunk, z, y) and stages that range (at most tile_m + uk_xhalo pixels)
    # once, together with the X taps' W tiles side by side, and the X taps
    # read it at a row shift of x: a third of the A loads, A LDS writes and
    # barriers per MFMA. Every workgroup checks the conditions from its
    # kernargs (and that its span fits) and otherwise runs the plain
    # uniform_k_decode loop, so the binary serves every problem the plain
    # kernel does. uniform_k_decode + reg_prefetch=1 + lds_xor only; a
    # multiple of 4. Tags ``uxh<n>``.
    uk_xhalo: int = 0
    # With uk_xhalo: the x = 1, 2 taps take the A fragments of the previous
    # tap shifted one row through the lanes (v_mov_b32 DPP row_shl:1, the
    # 16th row from the next fragment's first lane via row_shr:15) instead of
    # re-reading them from LDS; only each wave's last fragment is read. Needs
    # every output row boundary to fall on a wave's M extent (Wo a multiple
    # of it), checked per workgroup like the other x-halo conditions.
    # Tags ``xdpp``.
    xh_dpp: bool = False
    # reg_prefetch (1, 2) / single-buffer prefetch_regs / uk_xhalo loops:
    # raise the wave priority (s_setprio
    # <level>) over the next tile's load issue and the MFMA phase and drop it
    # before the barrier, so a wave that reached its MFMAs keeps the pipe
    # (Triton / CK intrawave do this). 0 = off. Tags ``prio<n>``.
    rpf_setprio: int = 0
    # ``uk_nopad`` (uniform_k_decode loop): the problem has no padding, so
    # every filter tap of an output pixel is inside the input and the
    # per-row, per-tap bounds checks go (only the tap-range check of the
    # prefetch past the end stays; rows past M read zeros through the buffer
    # bounds). The identity serves only unpadded problems (the halo form).
    uk_nopad: bool = False
    # defer_mfma's store phase (rpf1 loop): MFMAs per ds_write of the next
    # tile, one group per staged write. 0 = the original pattern (8 groups
    # of 1 ds_write : 2 MFMAs, tuned for defer_mfma=1). Tags ``dpw<n>``.
    defer_mfma_per_write: int = 0

    # Per-tile reduction decode (the forward counterpart of
    # ``DgradConvSpec.uniform_k_decode``). When cpg is a multiple of tile_k
    # every K tile lies inside one filter tap ([z,] y, x) and one run of
    # tile_k channels, so the tap is carried across the loop as uniform
    # counters (no per-element division chain), each load offset splits into a
    # loop-invariant per-element part (the output pixel) plus a uniform
    # per-tile part (the tap and channel base), and the bounds test compares
    # the invariant pixel origin against the tap's shifted extents. The binary
    # also carries the plain K loop and picks one of the two per problem from
    # the kernargs, so it serves every problem the plain kernel serves.
    # Default False: strictly additive (byte-identical IR, same name).
    uniform_k_decode: bool = False
    # Register-prefetch K loop: the global reads of K tile t+1 are issued
    # into registers *before* the MFMAs of tile t and written to LDS after
    # them, so their DRAM/L2 latency hides behind the MFMA phase of the same
    # wave. The plain ``unroll_k`` loop writes each prefetched tile to LDS
    # before the MFMAs, which makes every wave wait for its loads first.
    #   1: one LDS buffer -- a barrier retires the tile's ds_reads before the
    #      write and a second publishes it (no extra LDS, so large tiles fit);
    #   2: two LDS buffers, x2-unrolled ping-pong, one barrier per tile.
    # Replaces the ``unroll_k`` loop (setting both is rejected). 0 = off.
    reg_prefetch: int = 0
    # XCD-contiguous tile order: the hardware hands workgroups to the XCDs
    # round-robin in launch order and every XCD has its own L2, so in launch
    # order the neighbouring M tiles -- which read overlapping input rows
    # (the filter's y/z taps) -- all land on different L2s. With
    # ``xcd_remap = n`` (the GPU's XCD count) each XCD instead walks one
    # contiguous run of ``num_wgs / n`` tiles (M-major, the N tiles of one M
    # tile adjacent), so the input-row reuse between neighbouring tiles stays
    # inside one L2. A bijection of the launch grid (the tail past the last
    # full block keeps launch order). 0 (default) = launch order.
    xcd_remap: int = 0
    # XOR-swizzled A/B LDS tiles in place of the +8 row pad: the 16-byte chunk
    # ``j`` of tile row ``r`` is stored at chunk ``j ^ (r % (tile_k / 8))``, so
    # the MFMA fragment reads of consecutive rows still spread over the banks
    # while the row stride stays tile_k. Saves the pad's LDS (1/9 of the A/B
    # tile at tile_k = 64), which is what lets e.g. a 128x128x64 tile run two
    # workgroups per CU. Applies to the uniform_k_decode / reg_prefetch loops
    # (their LDS writes and the MFMA reads); tile_k >= 32. Default False.
    lds_xor: bool = False
    # A operand straight from global memory into the MFMA operand registers
    # (no A LDS tile): each lane loads the k-contiguous slice of its fragment
    # rows, and the MFMA k order inside a tile is permuted so a lane's
    # a_per_lane * k_atoms elements are contiguous in memory (B is read from
    # LDS in the same order). Only the uniform_k_decode + reg_prefetch=1 loop.
    # Measured slower on gfx942 (MI300A 3-D shapes): the MFMA lane order puts
    # 16 pixels in 16 consecutive lanes, which the TA cannot coalesce (TA busy
    # 30% -> 78%). Opt-in AOT axis only (ROCKE_AOT_FWD_ADIR=1).
    a_direct: bool = False
    # uniform_k_decode loop order: the filter tap (x, then y, then z) advances
    # fastest and the channel chunk outermost, instead of the chunk fastest.
    # A tile then revisits an input plane after Z*Y*X tiles rather than after
    # Z*Y*X*cpg/tile_k / Z, so the dz taps of one tile and of the tiles of the
    # neighbouring planes in flight hit the input in L2 (MI300A 3-D C=128:
    # L2 misses 37% -> see notes). Same reduction, other summation order.
    # 3-D uniform_k_decode loop only. Default False; tags ``tapf``.
    uk_tap_fast: bool = False
    # With uk_tap_fast: the channel chunks advanced fastest within groups of
    # ``uk_chunk_group`` chunks (tile_k * group channels, e.g. one 128-byte
    # L2 line of bf16 at tile_k 32, group 2), then the filter tap, then the
    # next group. Plain tapf reads half of every input line it touches at
    # tile_k 32 and the other half only Z*Y*X tiles later. Power of two,
    # 0 = off (plain tapf). Tags ``cg<n>``.
    uk_chunk_group: int = 0
    # 8-wave block ping-pong (Triton's BlockPingpong form) of the
    # uniform_k_decode + reg_prefetch=1 loop; 0 = off. The waves of the high
    # half (warp_id >= n_waves / 2) run one extra s_barrier before the loop
    # and the low half one after it, so the two waves sharing a SIMD run half
    # an iteration apart. Every K tile is four phases separated by barriers:
    # [mem A] all LDS fragment reads of the tile + the next tile's global
    # reads, [compute A] the first half of the MFMAs, [mem B] the next tile's
    # LDS store, [compute B] the second half -- one wave's memory phase hides
    # under its SIMD partner's MFMAs. The compute phases run at s_setprio
    # ``rpf_setprio``. 1: the barrier closing [mem A] waits for the wave's
    # fragment reads (lgkmcnt(0)); 2: it does not (the reads complete under
    # the partner's MFMAs; Triton's form); 3: as 2 without the compiler
    # fence on that barrier, so each MFMA waits for its own fragments;
    # 4: as 3 with the global reads of tile t+2 issued in [mem B] after the
    # LDS store of t+1 (loop-carried registers; [mem A] ends with
    # lgkmcnt(0) as in 1, modes 2/3 rely on in-order LDS across waves and
    # are experiments only), so [mem A] holds the
    # fragment reads only; 5: as 4 with those reads (and their address math)
    # inside [compute B], one buffer_load per ``vmem_interleave`` MFMAs.
    # Tags ``pp<n>``.
    rpf_pingpong: int = 0
    # rpf_pingpong: MFMAs of each compute phase issued ahead of the barrier
    # that opens it (operands in registers; hides the barrier). Tags ``ppm<n>``.
    pp_pre_mfma: int = 0
    # rpf_pingpong: B in two LDS buffers (alternating per tile), so the B
    # fragments of the second half of a tile's kk steps are read in [mem B]
    # instead of [mem A] (which is LDS-read bound). Tags ``ppbd``.
    pp_b_double: bool = False
    # lds_xor + lds_kpack_read at tile_k 32: XOR key = row % 4 instead of
    # (row / 2) % 4 (see lds_swz). Tags ``lxrk``.
    lds_xor_rowkey: bool = False
    # cshuffle epilogue: stage the C tile through LDS in this many row slabs
    # (CShuffleEpilogue.m_chunks), so a tile whose C does not fit the LDS
    # (e.g. 512x128 / 256x256 bf16 = 128 KB) can still take the cshuffle
    # store and the split-K atomic store. bf16 output, MFMA. Tags ``mch<n>``.
    cshuffle_m_chunks: int = 1
    # uk_xhalo loop: s_waitcnt lgkmcnt(0) before the loop, so no scalar load
    # is pending at its header and the first MFMA of every iteration waits
    # for its own fragments only (partial lgkmcnt) instead of all of them.
    # Tags ``xws``.
    xh_wait_smem: bool = False
    # split_k > 1: the packed bf16/fp16 atomic-adds of the split-K store at
    # device ("agent") scope instead of LLVM's default system scope. On
    # gfx94x the system-scope atomic carries SC1 and is performed past the
    # L2; at agent scope it is performed in the L2. D is device memory read
    # after the kernel, so device scope is the scope the reduction needs.
    # Tags ``saa``.
    split_atomic_agent: bool = False
    # split_k > 1: the split-K atomic store with one bf16 pair per lane,
    # lanes contiguous along N (CShuffleEpilogue.atomic_pair_lanes), instead
    # of store_vec/2 pairs per lane at a store_vec stride. Tags ``sapl``.
    split_atomic_pairs: bool = False
    # split_atomic_pairs: the in-bounds fast path of that store
    # (CShuffleEpilogue.atomic_fast_full). Tags ``sapf``.
    split_atomic_fast: bool = False
    # cshuffle bf16 store: a workgroup whose tile is in bounds, with a packed
    # N[D]HWK D, stores at a constant row step (CShuffleEpilogue.fast_ok)
    # instead of decoding each vector's output pixel. Tags ``esf``.
    epi_store_fast: bool = False
    # split_k > 1 (not split_tail): XCD-contiguous remap of the whole
    # (slice, tile) launch, slice-major, over this many XCDs -- the
    # workgroups of one XCD then run the same K slice of neighbouring tiles
    # and share its A/B blocks in their L2 (xcd_remap remaps the tiles of a
    # slice only, so a slice's blocks are read into every XCD's L2).
    # Tags ``skx<n>``.
    split_xcd_remap: int = 0
    # split_k > 1: stream-K over the split binary. The launch is a flat grid
    # of ``ks_count`` workgroups (the split degree kernarg is the workgroup
    # count) that share the (tile, K tile) iterations evenly; each runs the
    # one or two tile segments its range covers and atomic-adds each into the
    # zeroed D. Groups == 1. Tags ``fsk``.
    split_streamk: bool = False
    # split_k > 1: no zeroed D. Slice 0 of a split tile stores its partial
    # plainly, releases it (agent scope) and raises the tile's flag; the
    # other slices wait for the flag, then atomic-add. A counter beside the
    # flag lets the last slice reset both, so the ``sk_flags`` buffer (two
    # i32 per output tile, zeroed once at allocation) needs no per-launch
    # clear. A waiting workgroup only waits on a lower-z (earlier-dispatched)
    # one. Appends the ``sk_flags`` kernarg. Tags ``sfs``.
    split_flag_store: bool = False
    # uk_xhalo + rpf_pingpong 5: the x-halo loop as single-tap ping-pong
    # sub-tiles. The staged input range (A) is stored once per (chunk, z, y)
    # and serves three consecutive ping-pong tiles, one per x tap, each with
    # its own W tile (one tap's B, so the LDS holds A range + one B tile and
    # tile_k 64 fits). The next range's A loads issue in the second tap's
    # [compute B]. Also runs the split-K slices (a slice is a range of
    # (chunk, z, y) steps). Tags ``xsub``.
    xh_subtile: bool = False
    # split_k > 1: no zeroed D from the host. Slice 0 of a split tile zeroes
    # its D tile before its K loop, releases it (agent scope) and raises the
    # tile's flag; every slice checks the flag (long set by then) before its
    # atomic-add, and the last of them resets the flag pair (the sk_flags
    # buffer is zeroed once at allocation). Saves the per-launch memset of D.
    # Appends the ``sk_flags`` kernarg. Tags ``szf``.
    split_zero_first: bool = False
    # rpf_pingpong 4/5 (plain loop): the global reads run one tile further
    # ahead -- tile t+3's reads issue in [compute B] of tile t (two staged
    # register sets, loop-carried, alternating), so a load has a whole extra
    # K tile to arrive before its [mem B] store. Costs one more A + B slice of
    # registers per thread. Tags ``ppfd``.
    pp_fetch_deep: bool = False
    # rpf_pingpong 5 (plain loop): split the next tile's global reads between
    # two phases -- 1: the B reads of tile t+1 issue in [mem A] of tile t
    # (after its fragment reads) and only the A reads of tile t+2 in
    # [compute B]; 2: the other way round (A in [mem A], B in [compute B]).
    # The [mem A] phase runs under the partner's [compute B] with slack, so
    # fewer buffer_load issue stalls land between the MFMAs. 3 / 4: as 1 / 2
    # but the [mem A] reads are tile t+2's (a second register set for that
    # operand, so they have a whole tile more to arrive). 5: as 3, and tile
    # t+2's A reads issue in [mem B] after the LDS store (no reads left in
    # [compute B]). 6: as 3 with tile t+2's A reads split between
    # [compute A] (the first half of the thread's vectors) and [compute B].
    # 7: as 3 with the first half of tile t+2's A reads also in [mem A].
    # 8: as 3 with all of tile t+2's A reads in [compute A] (none in
    # [compute B]): every phase pair {mem A, compute B} / {compute A, mem B}
    # then carries one operand's reads. 9: as 3 with the [mem A] B reads
    # issued ahead of the tile's LDS fragment reads. 10: all of tile t+2's
    # reads (A and B) in [mem B] after the LDS store, none in [mem A] /
    # [compute B] (t256x256x32: [mem A]'s 12 fragment reads are the long
    # phase). Tags ``psf<n>``.
    pp_split_fetch: int = 0
    # uniform_k_decode kernels: the plain fallback loop's prologue (one
    # tile's address decode, global reads, LDS store and barrier) runs only
    # when its range is not empty, as the x-halo kernel already does. Every
    # workgroup the uniform loop serves otherwise pays it after its K loop
    # (~5k cycles at the end of a split-K slice, ATT). Tags ``fbse``.
    fb_skip_empty: bool = False
    # rpf_pingpong loop: the global reads of a tile past the loop's end (the
    # last iterations' prefetch) are masked out (buffer OOB: no memory
    # traffic) instead of reading the next split-K slice's data, which the
    # epilogue's first vmcnt wait then has to sit out. Tags ``ppeg``.
    pp_end_guard: bool = False
    # cshuffle epilogue: LDS-only barriers (CShuffleEpilogue.lds_barriers), so
    # an m-chunk does not wait for the previous chunk's atomics / stores and
    # the first chunk not for the K loop's last prefetch. Tags ``elb``.
    epi_lds_barriers: bool = False
    # split_k > 1: data-parallel slices + a stream-K tail (the stream-K
    # binary's flat grid of G = ks_count workgroups). With T tiles, the first
    # S*T workgroups (S = G // T) each run one aligned slice [s*P, (s+1)*P)
    # K tiles of one tile (slice-major, so concurrent slices share their K
    # offsets in L2); the H = G - S*T others each take the remainders
    # [S*P, Kt) of q = ceil(T / H) whole tiles. P balances a slice against q
    # remainders plus q-1 extra epilogues of ``split_hybrid`` K tiles each.
    # For grids that leave CUs idle at a whole slice count (s7: 64 tiles x 3 =
    # 192 of 228 CUs). Tags ``shy<n>``.
    split_hybrid: int = 0
    # AMDGPU machine-scheduler strategy (the kernel's "amdgpu-sched-strategy"
    # attribute): 0 = LLVM's default, 1 max-ilp, 2 max-memory-clause,
    # 3 iterative-ilp, 4 iterative-minreg, 5 iterative-maxocc. Tags ``mss<n>``.
    sched_strategy: int = 0
    # cshuffle epilogue replaced by a register-only store (16x16 MFMA atom,
    # bf16/f16 output, no split-K): each quad of lanes transposes its 4x4
    # block of the accumulator (DPP quad_perm + v_perm_b32), so every lane
    # holds 4 consecutive output channels of one output pixel and stores
    # them with one 8-byte buffer_store -- no LDS staging, no barriers, each
    # wave finishes on its own. 2: the B tile's columns are read in the
    # order that makes a lane's channels of two neighbouring N atoms one
    # contiguous run of 8 (atom ni, column j -> channel 32*(ni/2) +
    # 8*(j/4) + 4*(ni%2) + j%4), stored with one 16-byte buffer_store.
    # 3: the transpose across lanes l ^ 4 / l ^ 8 instead (B channel order
    # 8*(j%4) + j/4), so lanes l..l+3 store 64 contiguous bytes of one
    # output pixel (coalesced). 4: as 3 with a 4-VALU bf16 round-to-nearest-
    # even (NaN -> canonical quiet NaN) instead of fptrunc's 5.
    # 0 = off. Tags ``eqx<n>``.
    epi_quad_xpose: int = 0
    # Keep the MFMA accumulators in arch VGPRs (LLVM "amdgpu-agpr-alloc"="0,0"):
    # no v_accvgpr_write zeroing / v_accvgpr_read before the epilogue. Only
    # for kernels whose total register need fits the VGPR file. Tags ``nag``.
    no_agpr: bool = False
    # Output-plane band order of the M tiles (with ``xcd_remap``): each XCD's
    # contiguous run of tiles walks bands of ``m_band`` tiles of one output
    # plane (Ho*Wo pixels) through all planes (z) before the next band, so
    # the tiles an XCD runs at one time cover a compact z-y block whose input
    # rows (the z/y taps) stay in that XCD's L2. A plane need not be a whole
    # number of tiles (rows of round(Ho*Wo / tile_m) tiles, a narrower last
    # band; kernels/common/conv_m_band.py). 0 = off. Tags ``mb<n>``.
    m_band: int = 0
    # x-halo M-strided wave rows: MFMA fragment mi, atom row r of a wave holds
    # output pixel wave_m0 + mfmas_m * r + mi (not mi * 16 + r). Tap x of
    # fragment mi is then tap x - 1 of fragment mi + 1 (the same registers):
    # per kk step the three x taps read mfmas_m + 2 A fragments from LDS
    # instead of 3 * mfmas_m, with no DPP. The x-halo A tile gets its own LDS
    # layout (rows rotated within groups of 4, chunk key (r>>2)+(r>>4)) so the
    # 4-strided fragment rows stay conflict-free; the eqx epilogue stores
    # with the strided row map when the workgroup ran the x-halo loop (its
    # fallback loops keep the default map). Needs Wo % mfmas_m == 0 (per
    # workgroup, with the other x-halo conditions). Tags ``xsm``.
    xh_stride_m: bool = False

    @property
    def block_size(self) -> int:
        return self.warp_m * self.warp_n * self.wave_size

    @property
    def launch_block_size(self) -> int:
        """Total threads launched per workgroup.

        For ``pipeline="wavelet"`` this is ``block_size + num_load_waves * wave_size``;
        for all other pipelines it equals ``block_size``.
        """
        if self.pipeline == "wavelet":
            return self.block_size + self.num_load_waves * self.wave_size
        return self.block_size

    @property
    def k_atoms_per_tile_k(self) -> int:
        return self.tile_k // self.warp_tile_k

    @property
    def mfmas_per_warp_m(self) -> int:
        return self.tile_m // (self.warp_m * self.warp_tile_m)

    @property
    def mfmas_per_warp_n(self) -> int:
        return self.tile_n // (self.warp_n * self.warp_tile_n)

    @property
    def atom(self) -> MfmaAtom:
        return mfma_atom(
            self.data.dtype_a, self.warp_tile_m, self.warp_tile_n, self.warp_tile_k
        )

    # ---- group merging (Gm) ----
    #
    # ``problem.M`` / ``problem.N_gemm`` / ``problem.K_gemm`` stay the TRUE
    # per-group dims: they size the tensors, which do not merge. The ``grid_*``
    # properties below are the dims the *tile* covers. At group_merge == 1 the
    # two are equal, which is what keeps the default path byte-identical.

    @property
    def merged_problem(self) -> ConvProblem:
        """``problem`` with Gm groups folded into one.

        ``cpg``/``kpg`` are derived as ``C // groups`` / ``K // groups``, so
        dividing the group count multiplies both -- and nothing else moves.
        That is the whole trick: every consumer that asks the problem how wide
        a channel run is then sees ``Gm`` instead of 1.
        """
        if self.group_merge <= 1:
            return self.problem
        return dc_replace(self.problem, groups=self.problem.groups // self.group_merge)

    @property
    def grid_M(self) -> int:
        # GEMM-M the tile covers: N*Ho*Wo, unchanged by merging.
        return self.merged_problem.M

    @property
    def grid_N_gemm(self) -> int:
        # GEMM-N the tile covers: kpg*Gm (== Gm on depthwise).
        return self.merged_problem.N_gemm

    @property
    def grid_K_gemm(self) -> int:
        # GEMM-K the tile reduces over: [Z*]Y*X*cpg*Gm.
        return self.merged_problem.K_gemm

    @property
    def grid_groups(self) -> int:
        # Conv groups actually launched; gridDim.z is grid_groups.
        return self.problem.groups // max(1, self.group_merge)

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        p = self.problem
        return kernel_name_join(
            self.name,
            p.short(),
            f"t{self.tile_m}x{self.tile_n}x{self.tile_k}",
            f"w{self.warp_m}x{self.warp_n}",
            f"a{self.warp_tile_m}x{self.warp_tile_n}x{self.warp_tile_k}",
            f"{self.pipeline}_{self.epilogue}",
            self.acc_epilogue.tag(),
            flags={
                "async": self.async_dma,
                "noalc": self.cshuffle_no_alias,
                # unroll_k hand-rolls a double-buffered K-loop: a different body
                # that would otherwise share the plain kernel's name. Only
                # tagged when set, so every other kernel keeps its name.
                "unroll": self.unroll_k,
                "ukd": self.uniform_k_decode,
                f"rpf{self.reg_prefetch}": self.reg_prefetch > 0,
                f"xcd{self.xcd_remap}": self.xcd_remap > 0,
                "lxor": self.lds_xor,
                "pfr": self.prefetch_regs,
                "spk": self.split_k > 1,
                "spt": self.split_tail,
                "sakd": self.split_addr_k_decode,
                "adir": self.a_direct,
                "tapf": self.uk_tap_fast,
                f"cg{self.uk_chunk_group}": self.uk_chunk_group > 0,
                "lah": self.lds_ahead,
                f"dfr{self.defer_mfma}": self.defer_mfma > 0,
                f"vmi{self.vmem_interleave}": self.vmem_interleave > 0,
                "kprd": self.lds_kpack_read,
                "lbar": self.rpf_lds_barrier,
                "flate": self.rpf_fetch_late,
                f"uxh{self.uk_xhalo}": self.uk_xhalo > 0,
                "xdpp": self.xh_dpp,
                f"prio{self.rpf_setprio}": self.rpf_setprio > 0,
                f"pp{self.rpf_pingpong}": self.rpf_pingpong > 0,
                f"ppm{self.pp_pre_mfma}": self.pp_pre_mfma > 0,
                "ppbd": self.pp_b_double,
                "lxrk": self.lds_xor_rowkey,
                f"mch{self.cshuffle_m_chunks}": self.cshuffle_m_chunks > 1,
                "xws": self.xh_wait_smem,
                "saa": self.split_atomic_agent,
                "sapl": self.split_atomic_pairs,
                "sapf": self.split_atomic_fast,
                "esf": self.epi_store_fast,
                f"skx{self.split_xcd_remap}": self.split_xcd_remap > 0,
                "fsk": self.split_streamk,
                "sfs": self.split_flag_store,
                "szf": self.split_zero_first,
                "xsub": self.xh_subtile,
                "ppfd": self.pp_fetch_deep,
                "fbse": self.fb_skip_empty,
                "ppeg": self.pp_end_guard,
                "elb": self.epi_lds_barriers,
                f"shy{self.split_hybrid}": self.split_hybrid > 0,
                f"mss{self.sched_strategy}": self.sched_strategy > 0,
                "stc": self.split_tail_compact,
                f"psf{self.pp_split_fetch}": self.pp_split_fetch > 0,
                "nag": self.no_agpr,
                f"mb{self.m_band}": self.m_band > 0,
                "xsm": self.xh_stride_m,
                f"eqx{self.epi_quad_xpose}": self.epi_quad_xpose > 0,
                "nopad": self.uk_nopad,
                f"dpw{self.defer_mfma_per_write}": self.defer_mfma_per_write > 0,
                # The occupancy target changes the register allocation.
                f"wpe{self.waves_per_eu}": self.waves_per_eu is not None,
                # Gm changes the emitted body, so it must reach the symbol:
                # without it two degrees collide on one name and the compile
                # cache hands the second the first's binary.
                f"gm{self.group_merge}": self.group_merge > 1,
            },
        )

    def validate(self) -> None:
        # ``ImplicitGemmConvSpec.groups`` (the legacy P86 field) must not
        # contradict the authoritative ``problem.groups``. The kernel body reads
        # the problem's grouping; the spec field is kept for back-compat and, if
        # set, must agree.
        if self.groups != 1 and self.groups != self.problem.groups:
            raise ValueError(
                f"spec.groups={self.groups} contradicts problem.groups="
                f"{self.problem.groups}; set them consistently (problem.groups "
                f"is authoritative)"
            )
        if self.tile_m % (self.warp_m * self.warp_tile_m) != 0:
            raise ValueError(
                f"tile_m {self.tile_m} not divisible by warp_m * warp_tile_m "
                f"({self.warp_m} * {self.warp_tile_m})"
            )
        if self.tile_n % (self.warp_n * self.warp_tile_n) != 0:
            raise ValueError(
                f"tile_n {self.tile_n} not divisible by warp_n * warp_tile_n "
                f"({self.warp_n} * {self.warp_tile_n})"
            )
        if self.tile_k % self.warp_tile_k != 0:
            raise ValueError(
                f"tile_k {self.tile_k} not divisible by warp_tile_k {self.warp_tile_k}"
            )
        if self.block_size > 1024:
            raise ValueError(f"block_size {self.block_size} > 1024")
        layout = self.effective_lds_layout()
        if self.async_dma:
            layout.validate_for_async()
        if self.async_dma and self.lds_k_pad not in (None, 0):
            raise ValueError(
                "async_dma requires lds_k_pad to be 0/None because "
                "raw_ptr_buffer_load_lds writes a packed lane-contiguous tile"
            )
        if (
            self.acc_epilogue.clamp_min is not None
            and self.acc_epilogue.clamp_max is not None
            and self.acc_epilogue.clamp_min > self.acc_epilogue.clamp_max
        ):
            raise ValueError(
                "acc_epilogue clamp_min must be <= clamp_max "
                f"(got {self.acc_epilogue.clamp_min} > {self.acc_epilogue.clamp_max})"
            )
        _gm_ok, _gm_why = fwd_group_merge_available(self)
        if not _gm_ok:
            raise ValueError(_gm_why)
        _kl_why = _fwd_kloop_conflict(self)
        if _kl_why:
            raise ValueError(_kl_why)
        _spk_why = fwd_split_k_conflict(self)
        if _spk_why:
            raise ValueError(_spk_why)
        if self.split_addr_k_decode:
            p_ = self.problem
            if not p_.is_3d or p_.is_pointwise or self.group_merge > 1:
                raise ValueError(
                    "split_addr_k_decode is built on the 3-D split-addressing path "
                    "(no pointwise, no group_merge)"
                )
            if p_.cpg % self.tile_k:
                raise ValueError(
                    f"split_addr_k_decode needs cpg % tile_k == 0 (a K tile must not "
                    f"straddle a filter tap); cpg={p_.cpg}, tile_k={self.tile_k}"
                )
        if self.prefetch_regs and (self.async_dma or self.pipeline == "wavelet"):
            raise ValueError(
                "prefetch_regs stages the sync loader's registers; async_dma and "
                "the wavelet pipeline have their own load path"
            )

    def effective_lds_layout(self) -> LdsLayout:
        if self.lds_layout is not None:
            layout = self.lds_layout
        elif self.lds_xor:
            layout = LdsLayout.padded_k(self.tile_k, 0)
        elif self.lds_k_pad is not None:
            layout = LdsLayout.padded_k(self.tile_k, self.lds_k_pad)
        elif self.async_dma:
            layout = LdsLayout.packed_async(self.tile_k)
        else:
            layout = LdsLayout.padded_k(self.tile_k, 8 if self.tile_k >= 16 else 0)
        layout.validate()
        return layout

    @staticmethod
    def default_vector_sizes(C: int, K: int, dtype: str) -> "Tuple[int, int, int]":
        """Return ``(vec_a, vec_b, vec_c)`` for a forward-pass problem.

        Forward pass memory layout:
          A (input):  NHWC → last dim C → vec_a
          B (filter): KYXC → last dim C → vec_b
          D (output): NHWK → last dim K → vec_c
        """
        sizes = [8, 4, 2, 1] if dtype != "fp32" else [4, 2, 1]

        def _vec(n: int) -> int:
            return next(v for v in sizes if n % v == 0)

        vec_c = _vec(C)
        return vec_c, vec_c, _vec(K)


def _fwd_pfr_single_buffer(spec: "ImplicitGemmConvSpec") -> bool:
    """Whether the spec runs the single-buffer ``prefetch_regs`` K loop."""
    return (
        spec.prefetch_regs
        and not spec.unroll_k
        and not spec.async_dma
        and spec.pipeline != "wavelet"
        and not (spec.uniform_k_decode or spec.reg_prefetch)
    )


def _fwd_kloop_sched_conflict(spec: "ImplicitGemmConvSpec") -> str:
    """Why ``lds_ahead`` / ``defer_mfma`` / ``vmem_interleave`` cannot be
    honoured, or ``""`` (each knob is rejected where it would be a no-op, so
    no two names share one binary)."""
    if spec.uk_nopad and not spec.uniform_k_decode:
        return "uk_nopad drops the bounds checks of the uniform_k_decode loop only"
    if spec.uk_nopad and spec.uk_xhalo:
        return "uk_nopad is a uniform_k_decode loop variant; the x-halo loop has its own"
    if not (
        spec.lds_ahead or spec.defer_mfma or spec.vmem_interleave
        or spec.defer_mfma_per_write
    ):
        return ""
    if spec.wave_size != 64 or spec.pipeline == "wavelet" or spec.async_dma:
        return "lds_ahead/defer_mfma/vmem_interleave schedule the sync wave64 MFMA K loop"
    if spec.a_direct:
        return "lds_ahead/defer_mfma/vmem_interleave are not implemented in the a_direct loop"
    if spec.lds_ahead and not (spec.reg_prefetch or spec.prefetch_regs):
        return "lds_ahead is implemented in the reg_prefetch / prefetch_regs loops only"
    if spec.defer_mfma < 0 or spec.vmem_interleave < 0:
        return "defer_mfma and vmem_interleave must be >= 0"
    if spec.defer_mfma_per_write and (
        spec.defer_mfma_per_write < 0 or not spec.defer_mfma or spec.uk_xhalo
    ):
        return "defer_mfma_per_write shapes the rpf1 loop's defer_mfma store phase"
    if spec.defer_mfma:
        if spec.reg_prefetch != 1:
            return "defer_mfma is implemented in the reg_prefetch=1 loop only"
        # The x-halo loop defers inside its last tap's phase, so the whole
        # phase may go (the other taps' MFMAs stay before the barrier).
        _kk = spec.tile_k // spec.warp_tile_k
        if spec.defer_mfma > _kk or (spec.defer_mfma == _kk and not spec.uk_xhalo):
            return "defer_mfma must leave at least one kk step before the barrier"
    if spec.vmem_interleave:
        if not (spec.reg_prefetch in (1, 2) or _fwd_pfr_single_buffer(spec)):
            return (
                "vmem_interleave is implemented in the reg_prefetch and "
                "single-buffer prefetch_regs loops only"
            )
        if not (SchedulePolicy.for_pipeline(spec.pipeline).emit_hints or spec.uk_xhalo
                or spec.rpf_pingpong):
            # (The x-halo loop places its loads itself on every pipeline.)
            return "vmem_interleave needs a hint-emitting pipeline (compv3/compv4)"
    return ""


def _fwd_kloop_conflict(spec: "ImplicitGemmConvSpec") -> str:
    """Why ``uniform_k_decode`` / ``reg_prefetch`` cannot be honoured, or ``""``.

    Both rewrite the synchronous wave64 MFMA K loop; every K-loop driver that
    never reaches it (async DMA, the wavelet split, the hand-unrolled
    ``unroll_k`` loop) is rejected rather than silently renamed. Shared by
    ``validate`` and ``is_valid_spec``.
    """
    if spec.xcd_remap < 0:
        return "xcd_remap must be >= 0"
    _sched_why = _fwd_kloop_sched_conflict(spec)
    if _sched_why:
        return _sched_why
    if spec.xcd_remap and spec.chiplet_swizzle:
        return "xcd_remap and chiplet_swizzle both remap the grid; set one"
    if spec.m_band < 0 or (spec.m_band and (not spec.xcd_remap or spec.split_tail)):
        return "m_band needs xcd_remap (and no split_tail)"
    if spec.lds_xor:
        if not (spec.uniform_k_decode or spec.reg_prefetch or _fwd_pfr_single_buffer(spec)):
            return (
                "lds_xor is implemented in the uniform_k_decode / reg_prefetch and "
                "single-buffer prefetch_regs loops only"
            )
        if spec.tile_k < 32 or spec.lds_layout is not None or spec.lds_k_pad:
            return "lds_xor needs tile_k >= 32 and the default (unpadded) LDS layout"
    if spec.uk_tap_fast and not (spec.uniform_k_decode and spec.problem.is_3d):
        return "uk_tap_fast reorders the 3-D uniform_k_decode loop only"
    if spec.uk_chunk_group:
        g = spec.uk_chunk_group
        if not spec.uk_tap_fast:
            return "uk_chunk_group groups the chunks of the uk_tap_fast order only"
        if g < 2 or g & (g - 1):
            return "uk_chunk_group must be a power of two >= 2"
        if spec.split_k > 1 and not (spec.uk_nopad and spec.rpf_pingpong):
            # A split slice decodes its start in the grouped order (whole
            # groups only; the binary falls back to the plain loop otherwise).
            # Built and verified for the unpadded (halo form) ping-pong
            # binaries (ROCKE_AOT_FWD_PP_CG); the other loops keep rejecting
            # it, so the default AOT job set is unchanged.
            return "uk_chunk_group with split_k: the uk_nopad rpf_pingpong loops only"
        if spec.split_k > 1 and (spec.split_streamk or spec.uk_xhalo):
            return "uk_chunk_group with split_k: not with stream-K / x-halo"
    if spec.xh_subtile and not (
        spec.uk_xhalo and spec.rpf_pingpong == 5 and not spec.pp_b_double
        and not spec.xh_dpp and not spec.split_streamk and not spec.split_xcd_remap
    ):
        return "xh_subtile: the uk_xhalo pp5 loop (no pp_b_double / xh_dpp / stream-K / skx)"
    if spec.uk_xhalo and spec.split_k > 1 and not spec.xh_subtile:
        return "uk_xhalo does not combine with split_k (the slices run the uk loop)"
    if spec.uk_xhalo:
        if spec.uk_xhalo < 0 or spec.uk_xhalo % 4:
            return "uk_xhalo must be a positive multiple of 4"
        if not (spec.uniform_k_decode and spec.reg_prefetch == 1 and spec.lds_xor
                and spec.problem.is_3d):
            return "uk_xhalo is implemented in the 3-D uniform_k_decode + reg_prefetch=1 lds_xor loop only"
        if spec.a_direct or spec.rpf_fetch_late:
            return "uk_xhalo does not combine with a_direct or rpf_fetch_late"
    if spec.rpf_setprio and (
        not (spec.reg_prefetch in (1, 2) or _fwd_pfr_single_buffer(spec))
        or spec.rpf_setprio > 3
        or spec.a_direct
    ):
        return (
            "rpf_setprio (1..3) is implemented in the reg_prefetch and "
            "single-buffer prefetch_regs loops"
        )
    if spec.rpf_pingpong:
        _nw = spec.warp_m * spec.warp_n
        if spec.rpf_pingpong not in (1, 2, 3, 4, 5):
            return "rpf_pingpong must be 0..5"
        if (spec.rpf_pingpong == 5) != bool(spec.vmem_interleave):
            return "rpf_pingpong=5 places its loads one per vmem_interleave MFMAs (and only it)"
        _half_mf = (spec.tile_k // spec.warp_tile_k
                    * (spec.tile_m // spec.warp_m // spec.warp_tile_m)
                    * (spec.tile_n // spec.warp_n // spec.warp_tile_n)) // 2
        if spec.pp_pre_mfma > _half_mf:
            return "pp_pre_mfma must not exceed a compute phase's MFMAs"
        if spec.rpf_pingpong >= 4 and not spec.lds_xor:
            return "rpf_pingpong>=4 carries the staged tile in the lds_xor store form"
        if not (spec.uniform_k_decode and spec.reg_prefetch == 1):
            return "rpf_pingpong is implemented in the uniform_k_decode + reg_prefetch=1 loop only"
        if _nw != 8:
            return "rpf_pingpong pairs the two waves of each SIMD: it needs 8 waves"
        if (spec.lds_ahead or spec.defer_mfma or spec.rpf_fetch_late
                or spec.rpf_lds_barrier or spec.a_direct
                or (spec.uk_xhalo and spec.rpf_pingpong < 4)):
            return "rpf_pingpong has its own schedule (no lah/dfr/vmi/flate/lbar/adir/uxh)"
    if spec.cshuffle_m_chunks != 1:
        _mpw = spec.tile_m // spec.warp_m // spec.warp_tile_m
        if (spec.cshuffle_m_chunks < 1 or spec.epilogue != "cshuffle"
                or spec.data.dtype_d != "bf16" or spec.wave_size != 64
                or spec.pipeline == "wavelet" or _mpw % spec.cshuffle_m_chunks):
            return ("cshuffle_m_chunks needs the bf16 wave64 MFMA cshuffle epilogue and "
                    "must divide the MFMAs per warp along M")
        if spec.split_k > 1 and spec.warp_n > 2 and not spec.split_atomic_pairs:
            # Measured: the chunked atomic store of t256x256 w2x4 fails the
            # fp32 reference at every split degree (w4x2 passes); not root-
            # caused, so the combination is not built.
            return "cshuffle_m_chunks with split_k is verified for warp_n <= 2 only"
    if spec.lds_xor_rowkey and not (spec.lds_xor and spec.lds_kpack_read and spec.tile_k == 32):
        return "lds_xor_rowkey re-keys the k-packed lds_xor swizzle of tile_k 32 only"
    if spec.xh_wait_smem and not spec.uk_xhalo:
        return "xh_wait_smem shapes the uk_xhalo loop"
    if spec.epi_quad_xpose and not (
        spec.epilogue == "cshuffle" and spec.split_k <= 1 and spec.pipeline != "wavelet"
        and spec.warp_tile_m == spec.warp_tile_n == 16 and spec.data.dtype_d in ("bf16", "fp16")
    ):
        return ("epi_quad_xpose replaces the unsplit cshuffle store of the 16x16 MFMA "
                "atom (bf16/f16 output)")
    if spec.epi_quad_xpose not in (0, 1, 2, 3, 4) or (
        spec.epi_quad_xpose >= 2 and (
            (spec.tile_n // spec.warp_n) % 32 or not spec.uniform_k_decode
            or spec.reg_prefetch != 1 or spec.a_direct or spec.rpf_pingpong)
    ):
        return ("epi_quad_xpose is 1..4 (2-4: the uniform_k_decode reg_prefetch=1 "
                "loops, an even number of N atoms per wave)")
    if spec.pp_b_double and not (
        spec.rpf_pingpong and (
            (spec.lds_kpack_read and (spec.tile_k // spec.warp_tile_k // 2) % 2 == 0)
            # unpacked reads: the kk steps split in halves, one B read each
            or (not spec.lds_kpack_read and (spec.tile_k // spec.warp_tile_k) % 2 == 0)
        )
        if not spec.uk_xhalo else
        spec.rpf_pingpong in (4, 5) and spec.lds_kpack_read and not spec.xh_dpp
        and (spec.tile_n // spec.warp_n // spec.warp_tile_n) % 2 == 0
    ):
        return ("pp_b_double: the plain rpf_pingpong loop with k-packed reads and an even "
                "number of kk pairs per half")
    if spec.pp_pre_mfma and not spec.rpf_pingpong:
        return "pp_pre_mfma shapes the rpf_pingpong loop"
    if spec.pp_end_guard and not (spec.rpf_pingpong and not spec.uk_xhalo and not spec.a_direct):
        return "pp_end_guard masks the plain rpf_pingpong loop's prefetch past its end"
    if spec.epi_lds_barriers and spec.epilogue != "cshuffle":
        return "epi_lds_barriers shapes the cshuffle epilogue"
    if spec.fb_skip_empty and not spec.uniform_k_decode:
        return "fb_skip_empty guards the uniform_k_decode kernels' fallback loop"
    if spec.pp_fetch_deep and not (spec.rpf_pingpong in (4, 5) and not spec.uk_xhalo):
        return "pp_fetch_deep shapes the plain rpf_pingpong 4/5 loop"
    if spec.pp_split_fetch and not (spec.pp_split_fetch in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10) and spec.rpf_pingpong == 5
                                    and not spec.uk_xhalo and not spec.a_direct
                                    and (not spec.pp_fetch_deep or spec.pp_split_fetch in (3, 4))):
        return ("pp_split_fetch (1-10) shapes the plain rpf_pingpong 5 loop (with "
                "pp_fetch_deep: 3 / 4 only)")
    if spec.pp_pre_mfma < 0:
        return "pp_pre_mfma must be >= 0"
    if spec.xh_dpp and not spec.uk_xhalo:
        return "xh_dpp shifts the x-halo A fragments; it needs uk_xhalo"
    if spec.xh_stride_m and not (
        spec.uk_xhalo and spec.uk_xhalo % 4 == 0 and spec.epi_quad_xpose in (3, 4)
        and not spec.xh_dpp
        and not spec.rpf_pingpong and spec.mfmas_per_warp_m >= _XHALO_TAPS - 1
    ):
        return ("xh_stride_m needs uk_xhalo (a multiple of 4), epi_quad_xpose 3/4, mfmas_per_warp_m >= 2 "
                "and no xh_dpp / rpf_pingpong")
    if spec.rpf_fetch_late and (
        spec.reg_prefetch != 1 or spec.vmem_interleave or spec.a_direct
    ):
        return "rpf_fetch_late is implemented in the reg_prefetch=1 loop without vmem_interleave"
    if spec.rpf_lds_barrier and (spec.reg_prefetch != 1 or spec.defer_mfma or spec.a_direct):
        return (
            "rpf_lds_barrier is implemented in the reg_prefetch=1 loop only "
            "(defer_mfma and a_direct already use LDS-only barriers)"
        )
    if spec.lds_kpack_read:
        if not (spec.uniform_k_decode or spec.reg_prefetch):
            return "lds_kpack_read is implemented in the uniform_k_decode / reg_prefetch loops only"
        if spec.a_direct:
            return "lds_kpack_read and a_direct are exclusive"
        if (spec.tile_k // spec.warp_tile_k) % 2:
            return "lds_kpack_read needs an even number of kk steps per tile"
    if spec.a_direct:
        if not (spec.uniform_k_decode and spec.reg_prefetch == 1):
            return "a_direct is implemented in the uniform_k_decode + reg_prefetch=1 loop only"
    if spec.reg_prefetch not in (0, 1, 2):
        return f"reg_prefetch must be 0, 1 or 2, got {spec.reg_prefetch}"
    if not (spec.uniform_k_decode or spec.reg_prefetch):
        return ""
    # The small-spatial K loop (prefetch_regs, split_addr_k_decode) is a
    # separate driver; the two are not combined. split_k runs each slice's
    # K range through the same loops (the tap state starts at the slice).
    if spec.prefetch_regs or spec.split_addr_k_decode:
        return (
            "uniform_k_decode/reg_prefetch do not combine with prefetch_regs "
            "or split_addr_k_decode; set one K-loop family"
        )
    if spec.async_dma:
        return "uniform_k_decode/reg_prefetch rewrite the synchronous K loop; async_dma has its own"
    if spec.pipeline == "wavelet" or spec.wave_size != 64:
        return "uniform_k_decode/reg_prefetch are wave64 MFMA K loops"
    if spec.unroll_k:
        return "reg_prefetch/uniform_k_decode replace the unroll_k loop; set one of them"
    if spec.group_merge > 1:
        return "uniform_k_decode/reg_prefetch are not implemented with group_merge"
    if spec.uniform_k_decode and spec.problem.is_pointwise:
        return "uniform_k_decode: the pointwise fast path has no filter tap to decode"
    return ""


# ---------------------------------------------------------------------
# Group merging (Gm)
# ---------------------------------------------------------------------

# Group-merge degrees the merged-tile index math is written for. Powers of two
# keep the group split (``kg & (Gm-1)``, ``kg >> log2(Gm)``) and the diagonal
# test to a mask, a shift and a compare; CK constrains its own Gm to the same
# set (its xor transform runs with ApplyModulo=false, which requires it).
_GROUP_MERGE_DEGREES = (2, 4, 8, 16, 32, 64)


def fwd_group_merge_available(
    spec: ImplicitGemmConvSpec, arch: str = "gfx950"
) -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for ``spec.group_merge``.

    Single source of truth for :meth:`ImplicitGemmConvSpec.validate` and
    :func:`is_valid_spec`. Two copies of a wgrad gate is exactly how the
    dispatcher came to hand the builder a spec the builder then rejected (see
    :func:`~.conv_implicit_gemm_wgrad.wgrad_group_merge_available`), so both
    delegate here rather than re-deriving.
    """
    gm = spec.group_merge
    if gm == 1:
        return True, "ok"
    if gm not in _GROUP_MERGE_DEGREES:
        return False, (
            f"group_merge must be 1 or one of {_GROUP_MERGE_DEGREES}; got {gm}"
        )
    p = spec.problem
    if p.groups % gm != 0:
        return False, (
            f"group_merge {gm} does not divide groups {p.groups}: a workgroup "
            f"would own a partial group"
        )
    if p.cpg != 1 or p.kpg != 1:
        return False, (
            f"group_merge is implemented for depthwise only (cpg == kpg == 1); "
            f"got cpg={p.cpg} kpg={p.kpg}. Wider groups need the merged group "
            f"index to ride alongside a real channel index on both N and K, "
            f"which is not covered, so it is gated off."
        )
    # No wave-size term: the merge lives in the A/B addressing, the diagonal
    # mask on the B load and the epilogue's N bound, none of which depend on
    # the MMA family, so MFMA (wave64) and WMMA (wave32) build it alike.
    if spec.async_dma:
        return False, (
            "group_merge is incompatible with async_dma=True: AsyncTileLoader "
            "takes no load_vec and its predicate is chunk-granular (>= 1 dword), "
            "so it cannot express the per-element diagonal mask the merged B "
            "load needs"
        )
    if spec.vector_size_b not in (None, 1):
        # The builder forces B scalar under merge (the merged B tile is 1/Gm
        # dense and the mask is per-element). Rejecting here rather than
        # silently overriding keeps the spec an honest description of the
        # emitted kernel -- a spec that says vec_b=8 and emits vec_b=1 makes
        # every sweep row and every ISA review a lie.
        return False, (
            f"group_merge forces a scalar B load; got vector_size_b="
            f"{spec.vector_size_b}. Leave it None (or 1)."
        )
    # The merged A load and D store both run over exactly Gm consecutive NHWC
    # channels -- a build-time extent, so an explicit width can be checked
    # here rather than per problem. A width that does not divide Gm straddles
    # the merged run: the A load crosses into the next filter tap and the
    # store fails the epilogue's whole-vector bound and is dropped.
    for _name, _vec in (("a", spec.vector_size_a), ("c", spec.vector_size_c)):
        if _vec is not None and gm % _vec != 0:
            return False, (
                f"vector_size_{_name}={_vec} does not divide group_merge {gm}: "
                f"the merged channel run is Gm wide"
            )
    if p.is_pointwise:
        return False, (
            "group_merge is not implemented for the pointwise (1x1) fast path: "
            "it builds its bounds predicate from scratch and has no slot for "
            "the diagonal term"
        )
    if p.is_3d:
        return False, (
            "group_merge is not implemented for 3-D convolution: the merged "
            "addressing lives in the 2-D split address form, and the 3-D "
            "descriptor path has no slot for the merged channel decode or the "
            "diagonal mask"
        )
    # The merged GEMM-N must fit one tile, or a tile would straddle group pairs
    # the diagonal mask cannot separate. Mirrors CK's GemmN <= NPerBlock gate.
    if spec.grid_N_gemm > spec.tile_n:
        return False, (
            f"merged GEMM-N {spec.grid_N_gemm} (kpg*{gm}) exceeds tile_n "
            f"{spec.tile_n}"
        )
    return True, "ok"


# ---------------------------------------------------------------------
# Arch-aware spec validation
# ---------------------------------------------------------------------


def is_valid_spec_for_problem(
    spec: ImplicitGemmConvSpec, problem: ConvProblem, arch: str = "gfx950"
) -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for ``spec`` paired with a specific ``problem`` on ``arch``.

    Extends :func:`is_valid_spec` with problem-aware checks — e.g. tile
    divisibility against the actual M / N_gemm / K_gemm dimensions,
    minimum occupancy constraints, or problem-specific LDS pressure.
    """
    ok, reason = is_valid_spec(spec, arch)
    if not ok:
        return False, reason

    # uk_nopad drops the per-tap bounds checks: the binary is only correct
    # on an unpadded problem (the AOT build problem may be padded; the
    # runtime problem decides, as identity_supports_problem does).
    if getattr(spec, "uk_nopad", False) and (
        problem.pH or problem.pW or (problem.is_3d and problem.pD)
    ):
        return False, "uk_nopad serves unpadded problems only"

    # caps gridDim.y (and .x/.z) at 65535. The M-tile axis maps
    # to gridDim.y so reject specs whose grid would silently truncate and
    # leave output tiles unwritten.
    _MAX_GRID_DIM = 65535
    _grid_m = (problem.M + spec.tile_m - 1) // spec.tile_m
    _grid_n = (problem.N_gemm + spec.tile_n - 1) // spec.tile_n
    if _grid_m > _MAX_GRID_DIM:
        return False, (
            f"grid_m {_grid_m} > {_MAX_GRID_DIM} (hardware gridDim.y cap): "
            f"M={problem.M} tile_m={spec.tile_m}"
        )
    if _grid_n > _MAX_GRID_DIM:
        return False, (
            f"grid_n {_grid_n} > {_MAX_GRID_DIM} (hardware gridDim.x cap): "
            f"N_gemm={problem.N_gemm} tile_n={spec.tile_n}"
        )

    return True, "ok"


def fwd_split_tail_zero_rows(gemm_m: int, tile_m: int, num_pid_n: int, full_tiles: int) -> int:
    """First D row (GEMM-M index) a ``split_tail`` launch accumulates into:
    the host must zero D from this row on (every row of every group's
    output channels) before the launch; the rows above are stored plainly."""
    return min(gemm_m, (max(0, full_tiles) // max(1, num_pid_n)) * tile_m)


def fwd_split_k_conflict(spec: "ImplicitGemmConvSpec") -> str:
    """Why ``spec.split_k`` cannot be honoured, or ``""`` if it can.

    The split binary reduces through packed atomics out of the cshuffle tile
    (:meth:`CShuffleEpilogue.atomic_store`), so it needs that epilogue, the
    wave64 MFMA path, and a K loop that starts at an arbitrary tile: the
    plain loop or the single-buffer ``prefetch_regs`` loop. (The ``unroll_k``
    loops step two tiles and would run a slice of odd length into the next
    slice.) A 16-bit D pairs adjacent output channels, so kpg must be even.
    """
    if spec.split_k == 1:
        if (spec.split_xcd_remap or spec.split_streamk or spec.split_hybrid or spec.split_flag_store
                or spec.split_zero_first):
            return "split_xcd_remap / split_streamk / split_flag_store need the split-K binary"
        if spec.split_atomic_agent or spec.split_atomic_pairs:
            return "split_atomic_agent/split_atomic_pairs need the split-K binary (split_k > 1)"
        return "split_tail needs the split-K binary (split_k > 1)" if spec.split_tail else ""
    if spec.split_hybrid and (spec.split_streamk or spec.split_hybrid < 0):
        return "split_hybrid: one segment scheme (not with split_streamk), E >= 0"
    if (spec.split_streamk or spec.split_hybrid) and (
        spec.split_tail or spec.xcd_remap or spec.chiplet_swizzle
        or (spec.split_xcd_remap and not spec.split_hybrid)
    ):
        # (One group at launch: the host launches it on groups == 1 only.)
        return "split_streamk: no split tail, no tile remap"
    if (spec.split_flag_store or spec.split_zero_first) and (
        spec.split_streamk or spec.split_xcd_remap
        or (spec.split_flag_store and spec.split_zero_first)
    ):
        return "split_flag_store / split_zero_first: the plain split grid (slice = z), one of them"
    if spec.split_xcd_remap and (spec.split_tail or spec.xcd_remap or spec.chiplet_swizzle):
        return "split_xcd_remap replaces xcd_remap / chiplet_swizzle and keeps no split tail"
    if spec.epi_store_fast and (spec.epilogue != "cshuffle" or spec.cshuffle_m_chunks > 1
                                or spec.data.dtype_d != "bf16" or spec.wave_size != 64):
        return "epi_store_fast: the unchunked bf16 wave64 cshuffle store only"
    if spec.split_atomic_fast and (not spec.split_atomic_pairs or spec.cshuffle_m_chunks > 1):
        return "split_atomic_fast is the unchunked split_atomic_pairs store"
    if spec.split_atomic_pairs and spec.data.dtype_d != "bf16":
        return "split_atomic_pairs is the bf16 atomic store only"
    if spec.split_tail_compact and not spec.split_tail:
        return "split_tail_compact is a split_tail launch"
    if spec.split_tail and spec.chiplet_swizzle:
        return "split_tail keeps the tail in launch order; use xcd_remap, not chiplet_swizzle"
    if spec.split_k < 1:
        return f"split_k must be >= 1 (got {spec.split_k})"
    if spec.wave_size != 64 or spec.pipeline == "wavelet":
        return "fwd split_k > 1 is the wave64 MFMA path only"
    if spec.epilogue != "cshuffle":
        return "fwd split_k > 1 needs the cshuffle epilogue (atomic store)"
    if spec.unroll_k or spec.async_dma:
        return "fwd split_k > 1 needs the single-tile K loop (no unroll_k / async_dma)"
    if spec.group_merge > 1:
        return "fwd split_k > 1 does not support group_merge"
    if spec.data.dtype_d not in ("fp32", "bf16", "fp16"):
        return f"fwd split_k > 1 needs dtype_d fp32/bf16/fp16 (got {spec.data.dtype_d})"
    if spec.data.dtype_d != "fp32" and spec.problem.kpg % 2:
        return f"fwd split_k > 1 with a 16-bit D needs even kpg (got {spec.problem.kpg})"
    if spec.acc_epilogue.tag():
        return "fwd split_k > 1 does not support an accumulator epilogue"
    return ""


# uk_xhalo: the x taps a staged input range serves (the filter width it runs).
_XHALO_TAPS = 3


def _xhalo_a_rows(spec: "ImplicitGemmConvSpec") -> int:
    """A LDS rows: tile_m (or, with uk_xhalo, the staged range of tile_m +
    uk_xhalo pixels plus one scratch row the loader's surplus chunks write)."""
    if spec.uk_xhalo and spec.xh_stride_m:
        # whole groups of 4 rows (the xh_stride_m layout rotates rows in them)
        return -(-(spec.tile_m + spec.uk_xhalo + 1) // 4) * 4
    return spec.tile_m + spec.uk_xhalo + 1 if spec.uk_xhalo else spec.tile_m


def _xhalo_b_rows(spec: "ImplicitGemmConvSpec") -> int:
    """B LDS rows: tile_n (with uk_xhalo the X taps' tiles side by side, plus
    a scratch row when they do not split evenly over the threads)."""
    if not spec.uk_xhalo:
        return spec.tile_n
    rows = (1 if spec.xh_subtile else _XHALO_TAPS) * spec.tile_n
    _vb = _sync_load_vecs(spec)[1]
    return rows + (1 if (rows * (spec.tile_k // _vb)) % spec.block_size else 0)


def implicit_gemm_lds_bytes(spec: ImplicitGemmConvSpec) -> Tuple[int, int, int, bool]:
    """LDS a forward spec allocates: ``(total, a_b_bytes, c_bytes, double)``.

    ``a_b_bytes`` is one A/B staging copy, doubled in ``total`` when
    ``double``; ``c_bytes`` is the cshuffle C tile. Shared by
    :func:`is_valid_spec` and the AOT grid's per-arch LDS budget.
    """
    # LDS budget: A/B staging (×2 for double-buffer) + optional cshuffle C.
    _ab_dtype_bytes = 4 if spec.data.dtype_a in ("fp32",) else 2
    _lds_layout = spec.effective_lds_layout()
    _a_shape = _lds_layout.storage_shape(_xhalo_a_rows(spec))
    _b_shape = _lds_layout.storage_shape(_xhalo_b_rows(spec))
    _ab_bytes = (
        _a_shape[0] * _a_shape[1] + _b_shape[0] * _b_shape[1]
    ) * _ab_dtype_bytes
    # Only async_dma and unroll_k reach a K-loop that actually alternates between
    # the two LDS buffers: async_dma takes the SoftwarePipeline branch and
    # unroll_k hand-rolls a ping-pong. "compv4" on its own shares the plain
    # single-buffer scf.for_iter body with "mem"/"compv3" -- it differs only in
    # scheduling hints -- so charging it for a second A/B tile rejected specs for
    # LDS the kernel never allocates.
    _double = spec.async_dma or spec.unroll_k or spec.reg_prefetch == 2
    _ab_lds = _ab_bytes * (2 if _double else 1)
    if spec.pp_b_double:
        _ab_lds += _b_shape[0] * _b_shape[1] * _ab_dtype_bytes
    # cshuffle stages tile_m×tile_n elements at dtype_d (fp16/bf16 = 2B, fp32 = 4B).
    _c_dtype_bytes = 4 if spec.data.dtype_d == "fp32" else 2
    _c_lds = (
        spec.tile_m // max(1, spec.cshuffle_m_chunks) * spec.tile_n * _c_dtype_bytes
        if spec.epilogue == "cshuffle" else 0
    )
    # cshuffle_no_alias decides whether the cshuffle C tile shares the A/B pool
    # bytes, so the LDS budget must be computed per case:
    #   OFF (default, aliased): C reuses the A/B bytes   -> pool = max(ab, c)
    #   ON  (exclusive):        C gets its own LDS bytes -> pool = ab + c
    # wavelet also forces additive: A/B stay live across the whole scf_if_else
    # body (both branches reference them), so the liveness packer cannot alias
    # C onto the A/B region even when cshuffle_no_alias=False.
    _no_alias = spec.cshuffle_no_alias or spec.pipeline == "wavelet"
    _total_lds = (_ab_lds + _c_lds) if _no_alias else max(_ab_lds, _c_lds)
    return _total_lds, _ab_bytes, _c_lds, _double


def is_valid_spec(spec: ImplicitGemmConvSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for ``spec`` on ``arch``.

    The MFMA warp-tile atom (``warp_tile_m x warp_tile_n x
    warp_tile_k``) and the per-WG LDS capacity are sourced from
    :class:`rocke.core.arch.ArchTarget`, so this predicate is
    arch-aware: a warp-tile atom that exists on gfx950 but not gfx942
    (e.g. the default wide ``16x16x32`` f16 atom) is rejected for gfx942
    with a structured reason instead of crashing comgr at lower time
    (``LLVM ERROR: Cannot select intrinsic ...``).
    """
    from rocke.core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)

    # Geometry divisibility (mirrors ``spec.validate``).
    if spec.tile_m % (spec.warp_m * spec.warp_tile_m):
        return False, "tile_m not divisible by warp_m * warp_tile_m"
    if spec.tile_n % (spec.warp_n * spec.warp_tile_n):
        return False, "tile_n not divisible by warp_n * warp_tile_n"
    if spec.tile_k % spec.warp_tile_k:
        return False, "tile_k not divisible by warp_tile_k"
    _launch_block = spec.launch_block_size
    if _launch_block > target.max_threads_per_block:
        return False, (
            f"launch_block_size {_launch_block} > {target.max_threads_per_block} "
            f"(hardware cap) on {arch}"
        )

    _gm_ok, _gm_why = fwd_group_merge_available(spec, arch)
    if not _gm_ok:
        return False, _gm_why
    _kl_why = _fwd_kloop_conflict(spec)
    if _kl_why:
        return False, _kl_why

    # Check global store vector size and disable default epilogue for
    # vec_size_c > 1 — whether set explicitly or auto-derived from kpg.
    #
    # Read the *merged* problem: under group_merge the store runs over kpg*Gm
    # consecutive output channels, so a depthwise spec that looks scalar here
    # (kpg == 1) really stores Gm-wide. Using spec.problem would make this rule
    # never fire on exactly the specs merging exists for, and the validator
    # would then disagree with the emitter about which epilogue is legal.
    _merged = spec.merged_problem
    _eff_vec_c = (
        spec.vector_size_c
        if spec.vector_size_c is not None
        else ImplicitGemmConvSpec.default_vector_sizes(
            _merged.cpg, _merged.kpg, spec.data.dtype_d
        )[2]
    )
    _is_wmma_arch = target.wave_size == 32
    if _eff_vec_c > 1 and spec.epilogue == "default":
        return False, (
            f"default epilogue is not supported with vector size c: {_eff_vec_c}"
        )
    _why = vector_width_reason(
        (
            ("a", spec.vector_size_a, spec.data.dtype_a),
            ("b", spec.vector_size_b, spec.data.dtype_b),
            ("c", spec.vector_size_c, spec.data.dtype_d),
        )
    )
    if _why is not None:
        return False, _why

    # The MMA *family* is selected from the target's wave size: CDNA (wave64)
    # uses MFMA, the RDNA wave32 targets (gfx11xx) use WMMA. The same warp-tile
    # shape thus resolves to an MFMA op_id on gfx942/gfx950 and a WMMA op_id on
    # gfx1151. ``spec.wave_size`` is baked into ``block_size``, so it must match
    # the target's wave size or the lane geometry is wrong on hardware.
    family = "wmma" if _is_wmma_arch else "mma"
    if spec.wave_size != target.wave_size:
        return False, (
            f"spec wave_size {spec.wave_size} != {arch} wave_size {target.wave_size}"
        )

    # MMA atom must be in the target's catalog for the requested dtype.
    atom = (spec.warp_tile_m, spec.warp_tile_n, spec.warp_tile_k)
    if not target.mma.has_shape(
        family=family,
        a_dtype=spec.data.dtype_a,
        b_dtype=spec.data.dtype_b,
        c_dtype="fp32",
        m=spec.warp_tile_m,
        n=spec.warp_tile_n,
        k=spec.warp_tile_k,
    ):
        return False, f"unsupported {spec.data.dtype_a} warp_tile {atom} on {arch}"

    _total_lds, _ab_bytes, _c_lds, _double = implicit_gemm_lds_bytes(spec)
    if not target.fits_lds(_total_lds):
        return False, (
            f"LDS budget {_total_lds} bytes "
            f"(A/B={'x2 ' if _double else ''}{_ab_bytes}, C={_c_lds}) "
            f"> {target.lds_capacity_bytes} cap on {arch}"
        )

    if spec.pipeline == "wavelet":
        if spec.num_load_waves < 1:
            return False, "pipeline='wavelet' requires num_load_waves >= 1"
        if family != "wmma":
            return False, (
                f"pipeline='wavelet' is WMMA/gfx1250 only: on MFMA targets "
                f"the single-buffer LDS is overwritten each K iteration and load/math "
                f"waves execute sequentially rather than truly concurrently."
            )
        if spec.async_dma:
            return False, (
                "pipeline='wavelet' is incompatible with async_dma=True: "
                "the wavelet loaders are only constructed in the non-async branch "
                "and a_wavelet_loader/b_wavelet_loader would be None at fetch time."
            )
        # Guard against kernel IR so large that comgr takes minutes to compile.
        # The WMMA K-loop is fully unrolled: each iteration emits
        #   mfmas_per_warp_m × mfmas_per_warp_n WMMA calls,
        # and the loop runs K_iters = ceil(K_gemm / tile_k) times.
        # Empirically, cost > 512 causes comgr_relocatable to take excessively
        # long (e.g. t512x512x32 w1x1 with cost=36864 never completes in
        # practice).  The cut-off is conservative so all practical tile shapes
        # pass.
        _mfmas_m = spec.tile_m // (spec.warp_m * spec.warp_tile_m)
        _mfmas_n = spec.tile_n // (spec.warp_n * spec.warp_tile_n)
        # Merged K_gemm is Gm x larger, so reading spec.problem here would
        # under-count by exactly Gm and admit specs over the compile-time limit.
        _k_iters = (spec.grid_K_gemm + spec.tile_k - 1) // spec.tile_k
        _wmma_cost = _k_iters * _mfmas_m * _mfmas_n
        _WMMA_COST_LIMIT = 512
        if _wmma_cost > _WMMA_COST_LIMIT:
            return False, (
                f"pipeline='wavelet' unrolled WMMA count {_wmma_cost} "
                f"(K_iters={_k_iters} × mfmas={_mfmas_m}×{_mfmas_n}) "
                f"exceeds compile-time limit {_WMMA_COST_LIMIT}; "
                f"reduce tile_k, tile_m, or tile_n"
            )
    # basic pipeline uses the split emit_global_read/emit_lds_write path which
    # is only available on the sync (non-async-DMA) CoalescedTileLoader path.
    # async_dma uses raw_ptr_buffer_load_lds which atomically loads directly
    # into LDS with no VGPR staging, making the split impossible.
    if spec.pipeline == "basic" and spec.async_dma:
        return False, "pipeline='basic' is incompatible with async_dma=True"

    # WMMA (wave32) coverage: the 16x16x4 / 16x16x16 (gfx11/gfx12) or 16x16x32
    # (gfx1250) atom with the simple ``mem`` pipeline + ``default``/``cshuffle``
    # epilogue and synchronous descriptor-driven loads. The generic catalog check
    # above already rejects an atom the *specific* arch lacks (e.g. 16x16x32 on
    # gfx1151). The richer MFMA-shaped paths (compv3/compv4 scheduler interleave,
    # async DMA, K-unroll, chiplet swizzle) remain gated off on WMMA until ported;
    # grouped conv IS supported (the grid-per-group group index + group-aware
    # epilogue are family-neutral).
    if family == "wmma":
        if atom not in ((16, 16, 4), (16, 16, 16), (16, 16, 32)):
            return (
                False,
                f"WMMA conv supports only 16x16x4, 16x16x16, or 16x16x32 (got {atom}) on {arch}",
            )
        if spec.pipeline not in ("mem", "wavelet"):
            return False, (
                f"WMMA conv supports only 'mem', or 'wavelet' pipeline "
                f"(got {spec.pipeline!r}) on {arch}"
            )
        if spec.epilogue not in ("default", "cshuffle"):
            return False, (
                f"WMMA conv supports only 'default' or 'cshuffle' epilogue "
                f"(got {spec.epilogue!r}) on {arch}"
            )
        for flag, label in (
            (spec.async_dma, "async_dma"),
            (spec.unroll_k, "unroll_k"),
            (spec.prefetch_regs, "prefetch_regs"),
            (spec.chiplet_swizzle, "chiplet_swizzle"),
        ):
            if flag:
                return False, f"WMMA conv does not support {label} on {arch}"

    # The tile loaders. An explicit vector width (or the async chunk width the
    # tile admits) has to split the tile evenly over the block's threads; the
    # builder would otherwise only find out halfway through a build. The
    # wavelet loaders pick their own width, so they cannot fail this way.
    if spec.async_dma:
        if target.async_lds_max_dwords < 1:
            return False, f"async_dma: {arch} has no DRAM->LDS DMA instruction"
        try:
            async_tile_loaders(spec, arch)
        except ValueError:
            # Includes the arch width cap: on an arch whose buffer_load_lds
            # only moves a dword per lane, a tile that needs a wider chunk to
            # divide evenly over the block has no valid async configuration.
            return False, (
                f"async_dma: no usable chunk width for the A/B tiles with "
                f"block_size {spec.block_size} and at most "
                f"{target.async_lds_max_dwords} dword(s) per lane on {arch}"
            )
    elif spec.pipeline != "wavelet":
        try:
            load_vec_a, load_vec_b = _sync_load_vecs(spec)
        except ValueError:
            return False, (
                f"no usable load width for tile {spec.tile_m}x{spec.tile_n}x"
                f"{spec.tile_k} with block_size {spec.block_size}"
            )
        for operand, rows, vec in (
            ("A", spec.tile_m, load_vec_a),
            ("B", spec.tile_n, load_vec_b),
        ):
            why = coalesced_load_reason(
                operand, rows, spec.tile_k, spec.block_size, vec
            )
            if why is not None:
                return False, why

    return True, "ok"


def _conv_mma_family(arch: str) -> str:
    from rocke.core.arch import ArchTarget

    return "wmma" if ArchTarget.from_gfx(arch).wave_size == 32 else "mma"


def _resolve_conv_op(spec: ImplicitGemmConvSpec, arch: str):
    """Resolve the :class:`~rocke.core.arch.MmaOp` for ``spec`` on ``arch``.

    Returns the MFMA op on CDNA and the WMMA op on gfx1151. The op carries the
    backend ``op_id`` (consumed by :meth:`IRBuilder.mma`), the per-lane fragment
    lengths, and the lane/slot -> tile-coordinate layout maps that drive the
    fragment loads and the accumulator scatter — the cross-arch contract that
    lets one conv body emit both ISAs.
    """
    from rocke.core.arch import ArchTarget

    target = ArchTarget.from_gfx(arch)
    op = target.mma.op_for_shape(
        family=_conv_mma_family(arch),
        a_dtype=spec.data.dtype_a,
        b_dtype=spec.data.dtype_b,
        c_dtype="fp32",
        m=spec.warp_tile_m,
        n=spec.warp_tile_n,
        k=spec.warp_tile_k,
    )
    if op is None:
        raise ValueError(
            f"no MMA atom for conv warp_tile "
            f"({spec.warp_tile_m},{spec.warp_tile_n},{spec.warp_tile_k}) on {arch}"
        )
    return op


def implicit_gemm_conv_grid(spec: ImplicitGemmConvSpec) -> Tuple[int, int, int]:
    """Launch grid for one implicit-GEMM conv instance: ``(gn, gm, groups)``.

    ``x`` = N-tiles over the *per-group* GEMM N (``N_gemm = kpg``), ``y`` =
    M-tiles, ``z`` = group (CK-style grid-per-group; the kernel reads the group
    from ``blockIdx.z``). ``groups == 1`` gives the historical 2-D grid with a
    trivial ``z = 1``. When ``tile_n > kpg`` (e.g. cardinality-grouped g32/cpg8)
    ``gn == 1`` and the surplus N lanes are masked by the epilogue bound check.

    Under ``group_merge > 1`` one workgroup owns ``Gm`` conv groups, so ``z``
    shrinks by ``Gm`` and ``x`` is taken over the *merged* GEMM N. Computing
    either from the true problem would launch ``Gm x`` redundant workgroups all
    writing the same output.
    """
    gm = (spec.grid_M + spec.tile_m - 1) // spec.tile_m
    gn = (spec.grid_N_gemm + spec.tile_n - 1) // spec.tile_n
    return (gn, gm, spec.grid_groups)


# ---------------------------------------------------------------------
# Descriptor builders (the user-visible "transform-DAG" surface)
# ---------------------------------------------------------------------


def make_b_descriptor(p: ConvProblem, dtype: str = "fp16") -> TensorDescriptor:
    """Build the (n_gemm, k_gemm) -> K[Z]YXC linear-offset descriptor for the weight.

    KYXC (2-D) / KZYXC (3-D) is a flat row-major layout, and the
    implicit-GEMM treats ``n_gemm = k_out`` and
    ``k_gemm = y*X*C + x*C + c`` (2-D) or ``k_gemm = z*Y*X*C + y*X*C + x*C + c`` (3-D).
    The descriptor is a renaming + unmerge:

      2-D:
        naive(KYXC)  (k_out, y, x, c)
        + unmerge('k_gemm' -> y, x, c)
        + pad('y' lo=0 hi=Y)   boundary check for partial K-tile
        + pad('x' lo=0 hi=X)   boundary check for partial K-tile

      3-D:
        naive(KZYXC)  (k_out, z, y, x, c)
        + unmerge('k_gemm' -> z, y, x, c)
        + pad('z'), pad('y'), pad('x')

    The ``pad`` transforms catch the ``k_gemm >= K_gemm`` case (when the
    K-loop's last tile is partial): without them, the naive offset computation
    produces a value that's still in-buffer-bounds but indexes into the *next*
    ``k_out``'s weights, contaminating the accumulator.

    A also has ``pad('y')`` / ``pad('x')``, so when A's mask is 0 for the
    partial K-tile the MFMA contribution is 0 regardless of B. Padding B here
    is defense-in-depth so a future A-load change doesn't silently regress.
    """
    # Weight is stored per-group: KYXC / KZYXC with the channel extent equal to
    # ``cpg = C / groups`` (== C when groups == 1, so byte-identical to the
    # pre-groups kernel). The absolute output filter ``k_out = g*kpg + k_in_group``
    # is threaded in by the caller, so the group is folded into the ``k_out`` row
    # index and needs no descriptor coord here (mirrors CK's per-group weight
    # slab GKYXC == KYXC with C = cpg).
    if p.is_3d:
        return TensorDescriptor.naive(
            "B_kzyxc",
            lengths=[p.K, p.Z, p.Y, p.X, p.cpg],
            dtype=_ir_dtype(dtype),
            coord_names=["k_out", "z", "y", "x", "c"],
        ).transform(
            unmerge_magic(
                "k_gemm", into=["z", "y", "x", "c"], dims=[p.Z, p.Y, p.X, p.cpg]
            ),
            pad("z", lo=0, hi=p.Z),
            pad("y", lo=0, hi=p.Y),
            pad("x", lo=0, hi=p.X),
        )
    return TensorDescriptor.naive(
        "B_kyxc",
        lengths=[p.K, p.Y, p.X, p.cpg],
        dtype=_ir_dtype(dtype),
        coord_names=["k_out", "y", "x", "c"],
    ).transform(
        unmerge_magic(upper="k_gemm", into=["y", "x", "c"], dims=[p.Y, p.X, p.cpg]),
        pad("y", lo=0, hi=p.Y),
        pad("x", lo=0, hi=p.X),
    )


def make_d_descriptor(p: ConvProblem, dtype: str = "fp16") -> TensorDescriptor:
    """Build the (m, k_out) -> N[D]HWK linear-offset descriptor for the output.

    2-D:  naive(NHWK):  (n, ho, wo, k_out)
          + unmerge('m' -> n, ho, wo):  user-facing = (m, k_out)

    3-D:  naive(NDHWK): (n, do, ho, wo, k_out)
          + unmerge('m' -> n, do, ho, wo): user-facing = (m, k_out)
    """
    if p.is_3d:
        return TensorDescriptor.naive(
            "D_ndhwk",
            lengths=[p.N, p.Do, p.Ho, p.Wo, p.K],
            dtype=_ir_dtype(dtype),
            coord_names=["n", "do", "ho", "wo", "k_out"],
        ).transform(
            unmerge_magic(
                "m", into=["n", "do", "ho", "wo"], dims=[p.N, p.Do, p.Ho, p.Wo]
            ),
        )
    return TensorDescriptor.naive(
        "D_nhwk",
        lengths=[p.N, p.Ho, p.Wo, p.K],
        dtype=_ir_dtype(dtype),
        coord_names=["n", "ho", "wo", "k_out"],
    ).transform(
        unmerge_magic(upper="m", into=["n", "ho", "wo"], dims=[p.N, p.Ho, p.Wo]),
    )


# ---------------------------------------------------------------------
# Kernel body
# ---------------------------------------------------------------------


def _build_implicit_gemm_conv_impl(
    spec: ImplicitGemmConvSpec,
    *,
    arch: str = "gfx950",
    extra_params: Optional[Callable[[IRBuilder], object]] = None,
    m_index_fn: Optional[Callable[[IRBuilder, Value, WarpGrid], Value]] = None,
    a_mhw_index_fn: Optional[
        Callable[[IRBuilder, Value, WarpGrid], Tuple[Value, Value, Value]]
    ] = None,
    input_cache_setup: Optional[
        Callable[[IRBuilder, ImplicitGemmConvSpec, WarpGrid, Value], object]
    ] = None,
    a_load_override: Optional[
        Callable[
            [IRBuilder, ImplicitGemmConvSpec, Value, Value, WarpGrid, object], None
        ]
    ] = None,
    a_operand_override: Optional[
        Callable[
            [
                IRBuilder,
                ImplicitGemmConvSpec,
                Value,
                Value,
                Value,
                int,
                WarpGrid,
                object,
            ],
            Value,
        ]
    ] = None,
    epilogue_override: Optional[
        Callable[
            [IRBuilder, ImplicitGemmConvSpec, Sequence[Value], WarpGrid, Value, object],
            None,
        ]
    ] = None,
) -> KernelDef:
    """Build the IR for one implicit-GEMM conv instance.

    Shape:
        M = N * Ho * Wo,
        N_gemm = K,
        K_gemm = Y * X * C.
    Block tile: tile_m x tile_n x tile_k MFMA atoms at warp_tile_m x
        warp_tile_n x warp_tile_k.
    Pipeline: single-buffer LDS, sync barriers, direct vector global
    stores. (compv4 + cshuffle is a follow-on.)

    ``arch`` (``"gfx942"`` / ``"gfx950"``) selects the target GPU. The
    warp-tile MFMA atom is validated against the
    :class:`rocke.core.arch.ArchTarget` catalog (via
    :func:`is_valid_spec`) before lowering, so requesting an atom that
    only exists on gfx950 (e.g. the default wide ``16x16x32`` f16 atom)
    for ``gfx942`` fails with a clean structured error instead of an
    ``LLVM ERROR`` from comgr.
    """
    spec.validate()
    ok, why = is_valid_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid conv_igemm spec for {arch}: {why}")
    p = spec.problem
    # Group merging (Gm). The kernargs describe the TRUE problem -- the host
    # packs them from the tensors, which do not merge -- so every merged extent
    # the tile covers is derived in-kernel from the build-time degree. Merging
    # is depthwise-only (``fwd_group_merge_available``), so the merged per-group
    # channel run is exactly Gm: ``cpg*Gm`` and ``kpg*Gm`` fold to the constant
    # ``c_gm`` and only the reduction extent (``K_gemm*Gm``) stays a runtime
    # product. At ``group_merge == 1`` none of this is emitted.
    merged = spec.group_merge > 1
    p_load = spec.merged_problem
    if merged and a_mhw_index_fn is not None:
        # The merged A/B addressing lives in the split address form below; the
        # descriptor-DAG path a caller-supplied (n, ho, wo) decode forces has no
        # slot for the merged channel decode or the diagonal mask. (3-D, the
        # other descriptor-DAG path, is rejected by fwd_group_merge_available.)
        raise ValueError(
            "group_merge is not implemented with an a_mhw_index_fn override: "
            "it needs the 2-D split address form"
        )
    ir_dtype_a = _ir_dtype(spec.data.dtype_a)
    ir_dtype_b = _ir_dtype(spec.data.dtype_b)
    ir_dtype_d = _ir_dtype(spec.data.dtype_d)

    b = IRBuilder(spec.kernel_name())
    if spec.waves_per_eu is not None:
        b.kernel.attrs["waves_per_eu"] = spec.waves_per_eu
    if spec.sched_strategy:
        b.kernel.attrs["scheduler_strategy"] = _SCHED_STRATEGIES[spec.sched_strategy]
    if spec.no_agpr:
        b.kernel.attrs["agpr_alloc"] = "0,0"

    A = b.param(
        "A", PtrType(ir_dtype_a, "global"), noalias=True, readonly=True, align=16
    )
    Bp = b.param(
        "B", PtrType(ir_dtype_b, "global"), noalias=True, readonly=True, align=16
    )
    _split = spec.split_k > 1
    D = b.param(
        "D",
        PtrType(ir_dtype_d, "global"),
        noalias=True,
        # The split binary atomic-adds into D (read-modify-write).
        writeonly=not _split,
        align=16,
    )
    extra_context = extra_params(b) if extra_params is not None else None
    A_bytes = b.param("A_bytes", I32)
    B_bytes = b.param("B_bytes", I32)
    D_bytes = b.param("D_bytes", I32)

    # ---- Runtime problem-dimension parameters (AOT) -------------------------
    # Every problem dimension that was previously baked as b.const_i32(p.X) is
    # now a kernel argument, so one compiled HSACO serves any shape with the
    # same tile/pipeline configuration. The ConvProblem in spec.problem is
    # still used for validation, grid sizing and vectorisation checks --
    # host-side Python only; it is not emitted as IR.
    #
    # The block is emitted straight from the ordered ABI list so the kernel's
    # parameter order and conv_args_signature() cannot drift apart.
    params = emit_param_block(b, conv_fwd_problem_block(is_3d=p.is_3d))
    if _split:
        # Split-K launch block, appended after the problem block (conv_abi).
        p_ks = b.param("ks", I32)
        p_ks_count = b.param("ks_count", I32)
        if spec.split_tail:
            p_full_tiles = b.param("full_tiles", I32)
        if spec.split_flag_store or spec.split_zero_first:
            p_sk_flags = b.param("sk_flags", PtrType(I32, "global"), noalias=True)
    p_N = params["p_N"]
    p_Hi = params["p_Hi"]
    p_Wi = params["p_Wi"]
    p_C = params["p_C"]
    p_K = params["p_K"]
    p_Y = params["p_Y"]
    p_X = params["p_X"]
    p_sH = params["p_sH"]
    p_sW = params["p_sW"]
    p_pH = params["p_pH"]
    p_pW = params["p_pW"]
    p_dH = params["p_dH"]
    p_dW = params["p_dW"]
    p_groups = params["p_groups"]
    p_Ho = params["p_Ho"]
    p_Wo = params["p_Wo"]
    p_cpg = params["p_cpg"]
    p_kpg = params["p_kpg"]
    p_K_gemm = params["p_K_gemm"]  # [Z*]Y*X*cpg (reduction)
    p_M = params["p_M"]  # N*[Do*]Ho*Wo (output spatial)
    # 3-D depth axis; absent from the 2-D parameter list entirely.
    p_Z = params.get("p_Z")
    p_Di = params.get("p_Di")
    p_Do = params.get("p_Do")
    p_sD = params.get("p_sD")
    p_pD = params.get("p_pD")
    p_dD = params.get("p_dD")
    # Strides (row-major, in elements).
    p_A_stride_n = params["p_A_stride_n"]
    p_A_stride_di = params.get("p_A_stride_di")
    p_A_stride_hi = params["p_A_stride_hi"]
    p_A_stride_wi = params["p_A_stride_wi"]
    p_B_stride_k = params["p_B_stride_k"]
    p_B_stride_z = params.get("p_B_stride_z")
    p_B_stride_y = params["p_B_stride_y"]
    p_B_stride_x = params["p_B_stride_x"]
    p_D_stride_n = params["p_D_stride_n"]
    p_D_stride_do = params.get("p_D_stride_do")
    p_D_stride_ho = params["p_D_stride_ho"]
    p_D_stride_wo = params["p_D_stride_wo"]
    # Magic-number pairs, one per unmerge divisor (host-precomputed).
    p_num_pid_m = params["p_num_pid_m"]
    p_num_pid_n = params["p_num_pid_n"]
    if _split and spec.split_tail and spec.split_tail_compact:
        # Compact split_tail grid: decode the flat workgroup id into the
        # (x, y, z) the default launch would have given this (tile, slice),
        # and serve every later block_id_* from it.
        _stc_w = b.block_id_x()
        _stc_g = b.block_id_z()
        _stc_tiles = b.mul(p_num_pid_m, p_num_pid_n)
        _stc_tail = b.smax(b.sub(_stc_tiles, p_full_tiles), b.const_i32(1))
        _stc_r = b.smax(b.sub(_stc_w, p_full_tiles), b.const_i32(0))
        _stc_is_full = b.cmp_lt(_stc_w, p_full_tiles)
        _stc_slice = b.select(_stc_is_full, b.const_i32(0), b.div(_stc_r, _stc_tail))
        _stc_tile = b.select(
            _stc_is_full, _stc_w,
            b.add(p_full_tiles, b.mod(_stc_r, _stc_tail)),
        )
        _stc_vx = b.to_sgpr_u32(b.mod(_stc_tile, p_num_pid_n))
        _stc_vy = b.to_sgpr_u32(b.div(_stc_tile, p_num_pid_n))
        _stc_vz = b.to_sgpr_u32(b.add(b.mul(_stc_g, p_ks_count), _stc_slice))
        b.block_id_x = lambda: _stc_vx
        b.block_id_y = lambda: _stc_vy
        b.block_id_z = lambda: _stc_vz

    def _magic(prefix: str, name: str):
        """The ``(mult, shift)`` SSA pair for one unmerge divisor."""
        return (
            params[f"p_magic_{prefix}{name}_mult"],
            params[f"p_magic_{prefix}{name}_shift"],
        )

    # -------------------------------------------------------------------------

    # Resolve the MMA atom from the target catalog: an MFMA op on CDNA, a WMMA
    # op on gfx1151. ``op`` carries the per-lane fragment lengths and the
    # lane/slot -> coordinate layout maps that drive both the fragment loads and
    # the accumulator scatter, so the single conv body emits both ISAs. The
    # legacy ``MfmaAtom`` (``spec.atom``) is only used by the MFMA-specific
    # phases; it is ``None`` on the WMMA path.
    op = _resolve_conv_op(spec, arch)
    atom = spec.atom if op.family == "mma" else None
    a_per_lane = op.a_frag_len
    b_per_lane = op.b_frag_len
    # LDS operand element type must match the MFMA operand type.
    # bf16 → BF16, fp32 → F32; fp16 and fp8 go through the default f16 path.
    _smem_dtype: Optional[Type] = (
        BF16 if op.a_dtype == "bf16" else F32 if op.a_dtype == "fp32" else None
    )
    c_per_lane = op.c_frag_len

    block_m, block_n, block_k = spec.tile_m, spec.tile_n, spec.tile_k

    # ---- Tile-level block/warp/lane decomposition (CK Tile ``BlockGemmShape``) ----
    # ``WarpGrid.from_atom(...).bind(b)`` emits ``thread_id_x`` /
    # ``lane`` / ``warp_id`` / ``warp_m_idx`` / ``warp_n_idx`` /
    # ``block_m_off`` / ``block_n_off`` as one named decomposition,
    # along with ``max_workgroup_size`` on the kernel attribute list.
    # Replaces ~8 lines of hand-rolled ``b.div``/``b.mod`` lane arithmetic
    # plus the manual ``block_m_off = block_id_y * block_m`` math.
    grid = WarpGrid.from_atom(
        op,
        tile_m=block_m,
        tile_n=block_n,
        tile_k=block_k,
        warp_m=spec.warp_m,
        warp_n=spec.warp_n,
        wave_size=spec.wave_size,
    ).bind(b, block_m_axis="y", block_n_axis="x")
    tid = grid.tid
    lane = grid.lane
    warp_id = grid.warp_id
    # The mfma cshuffle/direct epilogues consume warp_m_idx/warp_n_idx via the
    # bound ``grid``; the WMMA direct epilogue still takes them explicitly.
    warp_m_idx = grid.warp_m_idx
    warp_n_idx = grid.warp_n_idx

    c0 = b.const_i32(0)
    c_block_k = b.const_i32(block_k)
    if merged:
        # Gm and its power-of-two split constants, materialised once. Gm is a
        # power of two (enforced by the gate), so splitting the merged
        # reduction index into its (filter position, merged group) parts is a
        # shift and a mask rather than a division. Not just cheaper: div/mod
        # lower to *signed* ops, and the compiler cannot see that k >= 0, so
        # it would emit the sign-correction sequence for a quantity that is
        # non-negative by construction.
        c_gm = b.const_i32(spec.group_merge)
        c_gm_mask = b.const_i32(spec.group_merge - 1)
        c_gm_shift = b.const_i32(spec.group_merge.bit_length() - 1)
    # Merged: [Z*]Y*X*Gm. The kernargs describe the TRUE problem (K_gemm is
    # [Z*]Y*X on depthwise), so the merged reduction is scaled in-kernel by the
    # build-time degree. The loop must cover it or Gm-1 of every Gm group pairs
    # never get their diagonal iteration.
    if merged:
        c_K_gemm = b.mul(p_K_gemm, c_gm)
    else:
        c_K_gemm = p_K_gemm  # runtime Value (was b.const_i32(p.K_gemm))

    # Grid: (block_n_idx, block_m_idx, 1). We follow gemm_universal:
    # block.x indexes N tile, block.y indexes M tile.
    #
    # With ``spec.chiplet_swizzle=True`` the (bx, by) is first flattened
    # into a linear WGID and run through the chiplet-aware super-tile
    # remap so consecutive WGs share an XCD (and an L2 slice). Conv
    # tile counts are derived from the problem shape and known at
    # build time, so we use the compile-time variant. We override the
    # ``WarpGrid``'s default block offsets via ``dataclasses.replace``
    # so the downstream helpers (loaders, epilogues) automatically
    # pick up the remapped origins.
    if spec.chiplet_swizzle:
        from rocke.helpers.grid import chiplet_aware_super_tile_dynamic

        wgid_flat = b.add(b.mul(b.block_id_y(), p_num_pid_n), b.block_id_x())
        swz = chiplet_aware_super_tile_dynamic(
            b,
            wgid_flat,
            num_pid_m=p_num_pid_m,
            num_pid_n=p_num_pid_n,
            wgm=spec.chiplet_wgm,
            num_xcds=spec.chiplet_num_xcds,
            chunk_size=spec.chiplet_chunk_size,
        )
        block_m_off_v = b.mul(swz.row, b.const_i32(block_m))
        block_n_off_v = b.mul(swz.col, b.const_i32(block_n))
        grid = dc_replace(grid, block_m_off=block_m_off_v, block_n_off=block_n_off_v)
    elif (spec.split_streamk or spec.split_hybrid) and spec.split_k > 1:
        # Stream-K: workgroup w of the flat grid of G = ks_count runs the
        # global iterations [w*T/G, (w+1)*T/G) of the T = tiles * K-tiles
        # (tile-major) -- one segment per tile it touches. Everything from
        # here to the epilogue is emitted inside the segment loop (closed
        # after the epilogue).
        _c1 = b.const_i32(1)
        _fsk_tiles = b.mul(p_num_pid_m, p_num_pid_n)
        _fsk_kt = b.div(b.add(c_K_gemm, b.const_i32(block_k - 1)), c_block_k)
        _fsk_T = b.mul(_fsk_tiles, _fsk_kt)
        _fsk_w = b.block_id_x()
        if spec.split_hybrid:
            # split_hybrid: S aligned slices per tile + whole-tile remainders.
            _shy_S = b.div(p_ks_count, _fsk_tiles)
            _shy_np = b.mul(_shy_S, _fsk_tiles)
            _shy_H = b.sub(p_ks_count, _shy_np)
            _shy_q = b.select(
                b.cmp_lt(c0, _shy_H),
                b.div(b.add(_fsk_tiles, b.sub(_shy_H, _c1)), b.smax(_shy_H, _c1)), c0)
            _shy_E = b.const_i32(spec.split_hybrid)
            _shy_Pb = b.div(
                b.add(b.mul(_shy_q, _fsk_kt), b.mul(b.smax(b.sub(_shy_q, _c1), c0), _shy_E)),
                b.add(_c1, b.mul(_shy_q, _shy_S)))
            _shy_S1 = b.smax(_shy_S, _c1)
            _shy_P = b.to_sgpr_u32(b.select(
                b.cmp_lt(c0, _shy_H),
                b.smin(_shy_Pb, b.div(_fsk_kt, _shy_S1)),
                b.div(b.add(_fsk_kt, b.sub(_shy_S1, _c1)), _shy_S1)))
            _shy_prim = b.cmp_lt(_fsk_w, _shy_np)
            _shy_wl = _fsk_w
            if spec.split_xcd_remap:
                # split_hybrid + split_xcd_remap: the aligned slices (slice-
                # major) XCD-contiguously, so an XCD runs neighbouring tiles of
                # one slice; the remainder workgroups keep launch order.
                from rocke.helpers.grid import chiplet_transform_chunked_runtime

                _shy_wl = b.select(_shy_prim, chiplet_transform_chunked_runtime(
                    b, _fsk_w, num_wgs=_shy_np, num_xcds=spec.split_xcd_remap), _fsk_w)
            _shy_s = b.div(_shy_wl, _fsk_tiles)
            _shy_tp = b.mod(_shy_wl, _fsk_tiles)
            _shy_r0 = b.mul(b.sub(_fsk_w, _shy_np), _shy_q)
            _shy_r1 = b.smin(b.add(_shy_r0, _shy_q), _fsk_tiles)
            _fsk_nseg = b.to_sgpr_u32(
                b.select(_shy_prim, _c1, b.smax(b.sub(_shy_r1, _shy_r0), c0)))
        else:
            _fsk_lo = b.to_sgpr_u32(b.div(b.mul(_fsk_w, _fsk_T), p_ks_count))
            _fsk_hi = b.to_sgpr_u32(b.div(b.mul(b.add(_fsk_w, _c1), _fsk_T), p_ks_count))
            _fsk_t0 = b.div(_fsk_lo, _fsk_kt)
            _fsk_t1 = b.div(b.sub(_fsk_hi, _c1), _fsk_kt)
            _fsk_nseg = b.select(
                b.cmp_lt(_fsk_lo, _fsk_hi), b.add(b.sub(_fsk_t1, _fsk_t0), _c1), c0
            )
        _fsk_loop = b.scf_for(c0, _fsk_nseg, _c1, iv_name="fsk_seg")
        # Every segment rewrites the LDS it reads (a barrier ends the body).
        _fsk_loop.op.attrs["smem_iter_local"] = True
        _fsk_seg = _fsk_loop.__enter__()
        if spec.split_hybrid:
            _fsk_tile = b.to_sgpr_u32(b.select(_shy_prim, _shy_tp, b.add(_shy_r0, _fsk_seg)))
            _shy_SP = b.mul(_shy_S, _shy_P)
            _fsk_k_lo = b.to_sgpr_u32(b.mul(
                b.select(_shy_prim, b.mul(_shy_s, _shy_P), _shy_SP), c_block_k))
            _fsk_k_hi = b.to_sgpr_u32(b.mul(
                b.select(_shy_prim, b.smin(b.mul(b.add(_shy_s, _c1), _shy_P), _fsk_kt),
                         _fsk_kt), c_block_k))
        else:
            _fsk_tile = b.to_sgpr_u32(b.add(_fsk_t0, _fsk_seg))
            _fsk_base = b.mul(_fsk_tile, _fsk_kt)
            _fsk_k_lo = b.to_sgpr_u32(
                b.mul(b.sub(b.smax(_fsk_lo, _fsk_base), _fsk_base), c_block_k)
            )
            _fsk_k_hi = b.to_sgpr_u32(
                b.mul(b.sub(b.smin(_fsk_hi, b.add(_fsk_base, _fsk_kt)), _fsk_base), c_block_k)
            )
        block_m_off_v = b.to_sgpr_u32(
            b.mul(b.div(_fsk_tile, p_num_pid_n), b.const_i32(block_m))
        )
        block_n_off_v = b.to_sgpr_u32(
            b.mul(b.mod(_fsk_tile, p_num_pid_n), b.const_i32(block_n))
        )
        # The thread's coordinates through an opaque per-segment copy: every
        # per-thread value derived from them stays inside the segment loop
        # instead of being hoisted out of it (LICM) and kept live -- spilled
        # -- across the K loop (the epilogue alone derives ~100 of them).
        tid = b.inline_asm("", "=v,0", [grid.tid], I32, sideeffect=True)
        _wave = b.const_i32(spec.wave_size)
        lane = b.mod(tid, _wave)
        warp_id = b.div(tid, _wave)
        warp_m_idx = b.div(warp_id, b.const_i32(spec.warp_n))
        warp_n_idx = b.mod(warp_id, b.const_i32(spec.warp_n))
        grid = dc_replace(
            grid, block_m_off=block_m_off_v, block_n_off=block_n_off_v,
            tid=tid, lane=lane, warp_id=warp_id, warp_m_idx=warp_m_idx,
            warp_n_idx=warp_n_idx,
        )
    elif spec.split_xcd_remap and spec.split_k > 1:
        from rocke.helpers.grid import chiplet_transform_chunked_runtime

        # The whole launch of one group, slice-major (hardware order: x, y,
        # then z = group * ks_count + slice), remapped XCD-contiguously.
        _tiles = b.mul(p_num_pid_m, p_num_pid_n)
        _skx_hw = b.add(
            b.mul(
                b.add(b.mul(b.mod(b.block_id_z(), p_ks_count), p_num_pid_m), b.block_id_y()),
                p_num_pid_n,
            ),
            b.block_id_x(),
        )
        _skx_log = chiplet_transform_chunked_runtime(
            b, _skx_hw, num_wgs=b.mul(_tiles, p_ks_count), num_xcds=spec.split_xcd_remap
        )
        _skx_slice = b.div(_skx_log, _tiles)
        _skx_tile = b.sub(_skx_log, b.mul(_skx_slice, _tiles))
        block_m_off_v = b.mul(b.div(_skx_tile, p_num_pid_n), b.const_i32(block_m))
        block_n_off_v = b.mul(b.mod(_skx_tile, p_num_pid_n), b.const_i32(block_n))
        grid = dc_replace(grid, block_m_off=block_m_off_v, block_n_off=block_n_off_v)
    elif spec.xcd_remap:
        from rocke.helpers.grid import chiplet_transform_chunked_runtime

        num_wgs = b.mul(p_num_pid_m, p_num_pid_n)
        wgid_flat = b.add(b.mul(b.block_id_y(), p_num_pid_n), b.block_id_x())
        if spec.split_tail:
            # Remap the full tiles among themselves only; the split tail keeps
            # launch order, so it is the last tiles of the output (the host
            # zeroes just those rows).
            logical = b.select(
                b.cmp_lt(wgid_flat, p_full_tiles),
                chiplet_transform_chunked_runtime(
                    b, wgid_flat, num_wgs=p_full_tiles, num_xcds=spec.xcd_remap
                ),
                wgid_flat,
            )
        else:
            logical = chiplet_transform_chunked_runtime(
                b, wgid_flat, num_wgs=num_wgs, num_xcds=spec.xcd_remap
            )
        _mt = b.div(logical, p_num_pid_n)
        if spec.m_band:
            # Output-plane band order (kernels/common/conv_m_band.py).
            _mt = emit_m_band_order(
                b, _mt, num_m_tiles=p_num_pid_m, plane=b.mul(p_Ho, p_Wo),
                tile_m=block_m, m_band=spec.m_band,
            )
        block_m_off_v = b.mul(_mt, b.const_i32(block_m))
        block_n_off_v = b.mul(b.mod(logical, p_num_pid_n), b.const_i32(block_n))
        grid = dc_replace(grid, block_m_off=block_m_off_v, block_n_off=block_n_off_v)
    else:
        block_m_off_v = grid.block_m_off
        block_n_off_v = grid.block_n_off

    # Grouped conv (groups > 1): the group index lives on grid.z (CK Tile puts
    # the group on the grid and offsets per-group). Each block computes one
    # group's per-group GEMM: N_gemm = kpg, K_gemm = [Z]*Y*X*cpg. The group
    # selects A's input-channel slab (via the descriptor's ``group`` coord) and
    # the absolute output-filter base ``k_out = g*kpg + n`` for B and D. For
    # groups == 1 nothing is emitted, keeping the kernel byte-identical.
    #
    # Under merge "group" means *merged* group: grid.z runs 0..groups/Gm and
    # the base advances by kpg*Gm == Gm output channels. The grouped path stays
    # engaged even when Gm == groups (one merged group): an AOT binary serves
    # every group count Gm divides, so the build-time problem's count cannot
    # decide whether block_id_z is needed.
    grouped = p.groups > 1 or merged
    if _split:
        # gridDim.z = groups * ks_count: group = z / ks_count, slice = z % ks_count.
        _z = b.block_id_z()
        if spec.split_streamk or spec.split_hybrid:
            k_lo, k_hi = _fsk_k_lo, _fsk_k_hi
        else:
            _slice = _skx_slice if spec.split_xcd_remap else b.mod(_z, p_ks_count)
            k_lo = b.mul(_slice, p_ks)
            k_hi = b.add(k_lo, p_ks)
        if spec.split_tail:
            # Launch-order tile index (before any grid remap): the first
            # full_tiles tiles run whole in slice 0; slices > 0 of them exit.
            _wg_lin = b.add(b.mul(b.block_id_y(), p_num_pid_n), b.block_id_x())
            _is_full = b.cmp_lt(_wg_lin, p_full_tiles)
            with b.scf_if(b.land(_is_full, b.cmp_ne(_slice, c0))):
                b.ret()
            k_lo = b.select(_is_full, c0, k_lo)
            k_hi = b.select(_is_full, c_K_gemm, k_hi)
    else:
        k_lo = c0
        k_hi = c_K_gemm
    if grouped:
        group_idx = b.div(b.block_id_z(), p_ks_count) if _split else b.block_id_z()
        # Merged: the base advances by the merged kpg, which is Gm exactly
        # (depthwise only), so it is the build-time constant, not p_kpg.
        k_out_group_base = b.mul(group_idx, c_gm if merged else p_kpg)
    else:
        group_idx = None
        k_out_group_base = None

    # LDS bank-conflict avoidance for the sync path: pad each K-row
    # by 8 halves so the stride is `block_k + 8` not `block_k`.
    # Adjacent lanes reading `ds_read_b128` (16 bytes = 4 banks each)
    # at the same K offset but different M rows would otherwise hit the
    # same banks; the +8 half pad (=16 bytes) shifts each row by 1
    # bank cycle. The pad-by-8 trick is a standard remedy: e.g. a
    # `K_PAD = 136 = 128 + 8` row stride for a `block_k = 128` tile.
    #
    # Async path (runbook §6.3) writes lane-contiguous LDS via
    # `raw_ptr_buffer_load_lds`, so K-pad would break the layout the
    # intrinsic produces. Use plain `[block_m, block_k]` instead;
    # bank conflicts (if they bind) move into the consumer's
    # ds_read distribution.
    lds_layout = spec.effective_lds_layout()
    if spec.async_dma:
        lds_layout.validate_for_async()
    A_smem = b.smem_alloc(
        ir_dtype_a, lds_layout.storage_shape(_xhalo_a_rows(spec)), name_hint="A_smem"
    )
    B_smem = b.smem_alloc(
        ir_dtype_b, lds_layout.storage_shape(_xhalo_b_rows(spec)), name_hint="B_smem"
    )
    # Async DMA only buys overlap when there is a second buffer to
    # write into while the MFMA phase reads from the first. Force
    # double-buffering whenever the pipeline opts into async DMA,
    # regardless of the chosen `compv*` flag.
    # See the LDS budget note in is_valid_*_spec: "compv4" alone does not reach a
    # buffer-alternating K-loop, so allocating a second tile for it produced a
    # dead allocation that the LDS pool then stripped anyway.
    double_buffer = spec.async_dma or spec.unroll_k or spec.reg_prefetch == 2
    if spec.pp_b_double:
        # rpf_pingpong: B alone in two buffers (A stays single).
        A_smem2 = A_smem
        B_smem2 = b.smem_alloc(
            ir_dtype_b, lds_layout.storage_shape(_xhalo_b_rows(spec)), name_hint="B_smem2"
        )
    elif double_buffer:
        A_smem2 = b.smem_alloc(
            ir_dtype_a, lds_layout.storage_shape(block_m), name_hint="A_smem2"
        )
        B_smem2 = b.smem_alloc(
            ir_dtype_b, lds_layout.storage_shape(block_n), name_hint="B_smem2"
        )
    else:
        A_smem2 = A_smem
        B_smem2 = B_smem

    mfmas_m = spec.mfmas_per_warp_m
    mfmas_n = spec.mfmas_per_warp_n
    k_atoms = spec.k_atoms_per_tile_k

    acc_init = b.zero_vec_f32(c_per_lane)
    accs = [
        (f"acc_m{mi}_n{ni}", acc_init) for mi in range(mfmas_m) for ni in range(mfmas_n)
    ]

    threads = spec.block_size
    load_vec_a, load_vec_b = _sync_load_vecs(spec)
    # ``CoalescedTileLoader`` derives ``vecs_per_thread`` /
    # ``cols_per_vec`` internally from ``(tile_rows, tile_cols,
    # block_size, load_vec)`` and re-emits the per-iter constants
    # (``c_threads``, ``c_load_vec``, ``c_cols_per_vec``) once per
    # ``load()`` invocation, which the AMDGPU backend constant-folds.

    # Build the descriptors against the AOT kernarg block instead of folding
    # the problem shape in as constants. The is_pointwise fast path stays a
    # compile-time decision (it is baked into the spec), so it still skips the
    # transform DAG entirely and only its bounds become runtime values.
    if p.is_pointwise:
        A_desc = None
        B_desc = None
        _c_C_ir = p_cpg  # runtime Value (was b.const_i32(p.cpg))
        _c_K_ir = p_kpg  # runtime Value (was b.const_i32(p.kpg))
        _c_M_ir = p_M  # runtime Value (was b.const_i32(p.M))
        _always_valid = b.const_i32(1)
    else:
        A_desc = make_a_descriptor_dynamic(
            b,
            params,
            is_3d=p.is_3d,
            grouped=grouped,
            decompose_m=(a_mhw_index_fn is None),
        )
        B_desc = make_b_descriptor_dynamic(b, params, is_3d=p.is_3d)
        _c_M_ir = _c_C_ir = _c_K_ir = _always_valid = None

    # CK Tile-style buffer views over A / B / D. ``make_buffer_resource``
    # wraps ``b.buffer_rsrc(ptr, num_bytes)`` and pre-binds a zero
    # ``soffset``; the resulting :class:`BufferResource` carries
    # everything ``raw_ptr_buffer_load`` needs. The buffer's DW3 flags
    # silently clamp OOB byte offsets to zero on loads / drop them on
    # stores -- the canonical AMDGPU tail-safe load idiom used for
    # both the conv padding-zone reads and the tail-of-grid epilogue
    # stores.
    a_buf_rsrc = make_buffer_resource(b, A, num_bytes=A_bytes)
    b_buf_rsrc = make_buffer_resource(b, Bp, num_bytes=B_bytes)
    d_buf_rsrc = make_buffer_resource(b, D, num_bytes=D_bytes)
    a_rsrc = a_buf_rsrc.rsrc
    b_rsrc = b_buf_rsrc.rsrc
    d_rsrc = d_buf_rsrc.rsrc
    input_cache_context = (
        input_cache_setup(b, spec, grid, a_rsrc)
        if input_cache_setup is not None
        else None
    )

    # Descriptor callbacks shared by both sync and async paths.
    # `(row, col)` are in the (tile_local M, tile_local K halves)
    # coordinate system. The descriptor returns
    # `(element_offset, valid_predicate)`.
    # Grouped conv threads the group index as A's ``group`` upper coord (selects
    # the input-channel slab) and folds it into B/D's absolute output filter
    # ``k_out``. ``_a_group_kw`` is empty for groups == 1 so the offset call is
    # unchanged (byte-identical).
    _a_group_kw = {"group": group_idx} if grouped else {}

    def a_descriptor(b_: IRBuilder, row: Value, col: Value):
        k_val = b_.add(k_off_capture[0], col)
        if p.is_pointwise:
            # Flat: offset = m * C + c,  valid = (m < M) & (c < C)
            m_val = b_.add(block_m_off_v, row)
            off = b_.add(b_.mul(m_val, _c_C_ir), k_val)
            m_ok = b_.cmp_lt(m_val, _c_M_ir)
            c_ok = b_.cmp_lt(k_val, _c_C_ir)
            return off, b_.land(m_ok, c_ok)
        if a_mhw_index_fn is not None:
            n_v, ho_v, wo_v = a_mhw_index_fn(b_, row, grid)
            return A_desc.offset(b_, n=n_v, ho=ho_v, wo=wo_v, k=k_val, **_a_group_kw)
        m_val = (
            m_index_fn(b_, row, grid)
            if m_index_fn is not None
            else b_.add(block_m_off_v, row)
        )
        if split_ab:
            if p.is_3d:
                return a_offset_split3(b_, m_val, k_decode3_at(b_, col))
            return a_offset_split(b_, m_val, k_val)
        return A_desc.offset(b_, m=m_val, k=k_val, **_a_group_kw)

    def b_descriptor_merged(b_: IRBuilder, row: Value, col: Value):
        """B load for ``group_merge > 1``: split k_gemm, mask the off-diagonal.

        The merged GEMM indexes ``n`` by the merged group ``g_n`` and ``k`` by
        ``(y, x, g_k)`` with ``g_k`` innermost. Only ``g_k == g_n`` is real
        work; every other (n, k) pair is a redundant MAC that must contribute
        zero.

        Putting that mask *here* -- on the weight load -- rather than on the
        store is the one substantive departure from CK Tile, and it is what
        makes this arrangement cheap. An invalid element is handed to the
        loader as ``valid = False``, which routes it to the buffer descriptor's
        OOB sentinel offset, and the hardware returns a zero without issuing
        any memory traffic. A zero weight makes the redundant MAC a no-op, so
        the accumulator is correct with no masking in the epilogue at all --
        and the store stays a dense contiguous run of Gm real outputs. CK, which
        puts the merged group on M and N, has to gather a diagonal out of its C
        tile instead, which is why its Gm=32 instance ships VectorSizeC = 1.
        """
        # g_n: which merged group this tile column owns. Self-bounding -- for
        # row >= Gm the diagonal is false for every k, so no N-axis clamp is
        # needed here (the epilogue still bounds the store).
        g_n = b_.add(block_n_off_v, row)
        kg = b_.add(k_off_capture[0], col)
        y, x, g_k = k_decode(b_, kg)
        # k_out is the ABSOLUTE output filter index into the physical weight
        # tensor: merged_group * Gm + g_n.
        k_out = b_.add(k_out_group_base, g_n)
        # The weight tensor is physical and does not merge: on depthwise it
        # really is W[K][Y][X][1], so the channel term is always 0 and the
        # offset is the filter row plus the (y, x) tap.
        off = b_.add(
            b_.add(b_.mul(k_out, p_B_stride_k), mul_u24(b_, y, p_B_stride_y)),
            mul_u24(b_, x, p_B_stride_x),
        )
        diag = b_.cmp_eq(g_k, g_n)
        return off, b_.land(b_.cmp_lt(y, p_Y), diag)

    def b_descriptor(b_: IRBuilder, row: Value, col: Value):
        k_out = b_.add(block_n_off_v, row)
        if grouped:
            k_out = b_.add(k_out_group_base, k_out)
        kg = b_.add(k_off_capture[0], col)
        if p.is_pointwise:
            # Flat: offset = k_out * C + c,  valid = (k_out < K) & (c < C)
            off = b_.add(b_.mul(k_out, _c_C_ir), kg)
            k_ok = b_.cmp_lt(k_out, _c_K_ir)
            c_ok = b_.cmp_lt(kg, _c_C_ir)
            return off, b_.land(k_ok, c_ok)
        if split_ab:
            if p.is_3d:
                return b_offset_split3(b_, k_out, k_decode3_at(b_, col))
            return b_offset_split(b_, k_out, kg)
        return B_desc.offset(b_, k_out=k_out, k_gemm=kg)

    if spec.group_merge > 1:
        # Swap once, here, rather than at the four consumer sites below: the
        # two closures are interchangeable by signature, and a per-site
        # conditional is how one of them (e.g. the wavelet path at the bottom)
        # quietly keeps the unmerged loader.
        b_descriptor = b_descriptor_merged

    # `k_off_capture` lets the closures pick up the current k0 from
    # the K-loop body without recompiling the loaders per iteration.
    # The list is mutated by `emit_load_phase`.
    k_off_capture: List[Optional[Value]] = [None]

    if spec.async_dma:
        # Async DRAM -> LDS via `raw_ptr_buffer_load_lds`. Each wave
        # writes lane-contiguous LDS at the wave-uniform base computed
        # by AsyncTileLoader. Consumers (the MFMA phase) must place an
        # `s_waitcnt(vmcnt=0)` before the first ds_read. The chunk width is
        # capped by the arch (CDNA3 moves only a dword per lane to LDS).
        a_loader, b_loader = async_tile_loaders(spec, arch)
        a_sync_loader = None
        b_sync_loader = None
    else:
        a_loader = None
        b_loader = None
        # Sync path: ``CoalescedTileLoader`` encapsulates the
        # "pick load_vec, per-thread div/mod into (row, col),
        # buffer_load_vN_f16 -> smem_store_vN_f16" pattern. Same per-iter
        # IR shape as the prior hand-rolled loop, but the per-thread
        # chunk math (``cols_per_vec``, ``vecs_per_thread``, OOB sentinel
        # routing via the descriptor's ``valid`` predicate) is centralised
        # so a future tile-shape sweep picks it up uniformly across
        # GEMM, conv, and attention loads.
        a_sync_loader = CoalescedTileLoader(
            tile_rows=block_m,
            tile_cols=block_k,
            block_size=threads,
            load_vec=load_vec_a,
            elem_dtype=ir_dtype_a,
        )
        b_sync_loader = CoalescedTileLoader(
            tile_rows=block_n,
            tile_cols=block_k,
            block_size=threads,
            load_vec=load_vec_b,
            elem_dtype=ir_dtype_b,
        )
        # Wavelet pipeline: load waves use a separate loader sized to the
        # number of load threads (num_load_waves * wave_size), not to
        # spec.block_size (math threads only).  Using the math-wave
        # block_size here causes vec_idx = e*math_threads + load_tid to
        # exceed the tile dimensions, writing LDS at wrong addresses and
        # producing NaN outputs.
        if spec.pipeline == "wavelet":
            a_wavelet_loader, b_wavelet_loader = _build_wavelet_loaders(
                num_load_waves=spec.num_load_waves,
                wave_size=spec.wave_size,
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
                load_vec_a=load_vec_a,
                load_vec_b=load_vec_b,
                ir_dtype_a=ir_dtype_a,
                ir_dtype_b=ir_dtype_b,
            )
        else:
            a_wavelet_loader = None
            b_wavelet_loader = None

    # ---- A/B addressing, split into a K-invariant and a K-varying part ----
    # The descriptor DAG forms A's offset as n*s_n + hi*s_hi + wi*s_wi + c with
    # hi = ho*sH + y*dH - pH: every stride multiplies a sum of an output-position
    # term (fixed per loader row) and a filter-position term (moving with k), so
    # no product in it is loop-invariant. Distributing the strides separates a
    # row part -- the Ho/Wo magic divisions and the n/hi/wi products, which LLVM
    # hoists out of the K loop -- from a k part of one stride step per filter
    # axis. The k decode's remainder products and B's filter strides (both
    # bounded by the reduction extent) use 24-bit multiplies; the host enforces
    # the bound (conv_args.MUL24_REDUCTION_LIMIT).
    split_ab = not p.is_pointwise and a_mhw_index_fn is None and not (
        p.is_3d and merged
    )
    if split_ab and p.is_3d:
        # 3-D: the same row/k split with the depth axis added. A K tile moves
        # only the k part (one 24-bit decode of k into (z, y, x, c) per load
        # plus three stride steps); the (n, do, ho, wo) magic divisions and
        # the (di, hi, wi) origin are per loader row and hoisted.
        s_di = params["p_A_stride_di"]
        s_hi = params["p_A_stride_hi"]
        s_wi = params["p_A_stride_wi"]
        neg_pD = b.sub(c0, p_pD)
        neg_pH = b.sub(c0, p_pH)
        neg_pW = b.sub(c0, p_pW)
        a_zstep = b.mul(p_dD, s_di)
        a_ystep = b.mul(p_dH, s_hi)
        a_xstep = b.mul(p_dW, s_wi)
        a_group_base = b.mul(group_idx, p_cpg) if grouped else None
    elif split_ab:
        s_hi = params["p_A_stride_hi"]
        s_wi = params["p_A_stride_wi"]
        neg_pH = b.sub(c0, p_pH)
        neg_pW = b.sub(c0, p_pW)
        # One filter row / column step in A's offset.
        a_ystep = b.mul(p_dH, s_hi)
        a_xstep = b.mul(p_dW, s_wi)
        # Merged: the A channel slab is the merged group's, Gm channels wide.
        a_group_base = b.mul(group_idx, c_gm if merged else p_cpg) if grouped else None

    def k_decode(b_: IRBuilder, k_val: Value) -> Tuple[Value, Value, Value]:
        """k -> (y, x, c) through the runtime magic pairs for cpg and X.

        Under ``group_merge`` the innermost field of k is the merged group
        (the merged per-group channel run is Gm wide), split off with a shift
        and a mask; ``c`` is then the in-merge group index, which is both A's
        channel within the merged slab and the diagonal coordinate of B.
        """
        if merged:
            q_c = b_.lshr(k_val, c_gm_shift)
            c = b_.land(k_val, c_gm_mask)
            y, x = magic_divmod(
                b_,
                q_c,
                params["p_magic_k_X_mult"],
                params["p_magic_k_X_shift"],
                p_X,
                u24=True,
            )
            return y, x, c
        q_c, c = magic_divmod(
            b_,
            k_val,
            params["p_magic_k_cpg_mult"],
            params["p_magic_k_cpg_shift"],
            p_cpg,
            u24=True,
        )
        y, x = magic_divmod(
            b_,
            q_c,
            params["p_magic_k_X_mult"],
            params["p_magic_k_X_shift"],
            p_X,
            u24=True,
        )
        return y, x, c

    def k_decode3(b_: IRBuilder, k_val: Value) -> Tuple[Value, Value, Value, Value]:
        """3-D k -> (z, y, x, c) through the runtime magic pairs for cpg, X, Y."""
        q_c, c = magic_divmod(
            b_,
            k_val,
            params["p_magic_k_cpg_mult"],
            params["p_magic_k_cpg_shift"],
            p_cpg,
            u24=True,
        )
        q_x, x = magic_divmod(
            b_,
            q_c,
            params["p_magic_k_X_mult"],
            params["p_magic_k_X_shift"],
            p_X,
            u24=True,
        )
        z, y = magic_divmod(
            b_,
            q_x,
            params["p_magic_k_Y_mult"],
            params["p_magic_k_Y_shift"],
            p_Y,
            u24=True,
        )
        return z, y, x, c

    def k_decode3_at(b_: IRBuilder, col: Value) -> Tuple[Value, Value, Value, Value]:
        """``(z, y, x, c)`` of reduction index ``k_off + col`` of the current tile.

        With ``split_addr_k_decode`` the tile lies inside one filter tap
        (cpg % tile_k == 0, a host-checked precondition), so the tap is
        decoded once from the wave-uniform tile origin -- scalar work -- and
        the channel is the origin's plus ``col``. Otherwise every element
        decodes its own index.
        """
        k_off = k_off_capture[0]
        if spec.split_addr_k_decode:
            z, y, x, c_t = k_decode3(b_, k_off)
            return z, y, x, b_.add(c_t, col)
        return k_decode3(b_, b_.add(k_off, col))

    def a_offset_split3(b_: IRBuilder, m_val: Value, zyxc):
        # Row part: m -> (n, do, ho, wo), the (di, hi, wi) origin and the base.
        q_wo, wo = magic_divmod(
            b_,
            m_val,
            params["p_magic_m_Wo_mult"],
            params["p_magic_m_Wo_shift"],
            p_Wo,
        )
        q_ho, ho = magic_divmod(
            b_,
            q_wo,
            params["p_magic_m_Ho_mult"],
            params["p_magic_m_Ho_shift"],
            p_Ho,
        )
        n, do_ = magic_divmod(
            b_,
            q_ho,
            params["p_magic_m_Do_mult"],
            params["p_magic_m_Do_shift"],
            p_Do,
        )
        di0 = b_.add(b_.mul(do_, p_sD), neg_pD)
        hi0 = b_.add(b_.mul(ho, p_sH), neg_pH)
        wi0 = b_.add(b_.mul(wo, p_sW), neg_pW)
        base = b_.add(
            b_.add(
                b_.add(b_.mul(n, params["p_A_stride_n"]), b_.mul(di0, s_di)),
                b_.mul(hi0, s_hi),
            ),
            b_.mul(wi0, s_wi),
        )
        if grouped:
            base = b_.add(base, a_group_base)
        # k part.
        z, y, x, c = zyxc
        di = b_.add(di0, b_.mul(z, p_dD))
        hi = b_.add(hi0, b_.mul(y, p_dH))
        wi = b_.add(wi0, b_.mul(x, p_dW))
        ok = b_.land(
            b_.land(
                b_.land(b_.cmp_ge(di, c0), b_.cmp_lt(di, p_Di)),
                b_.land(b_.cmp_ge(hi, c0), b_.cmp_lt(hi, p_Hi)),
            ),
            b_.land(b_.cmp_ge(wi, c0), b_.cmp_lt(wi, p_Wi)),
        )
        ok = b_.land(ok, b_.cmp_lt(z, p_Z))
        off = b_.add(
            b_.add(
                b_.add(b_.add(base, b_.mul(z, a_zstep)), b_.mul(y, a_ystep)),
                b_.mul(x, a_xstep),
            ),
            c,
        )
        return off, ok

    def b_offset_split3(b_: IRBuilder, k_out: Value, zyxc):
        base = b_.mul(k_out, params["p_B_stride_k"])
        z, y, x, c = zyxc
        off = b_.add(
            b_.add(
                b_.add(
                    b_.add(base, mul_u24(b_, z, params["p_B_stride_z"])),
                    mul_u24(b_, y, params["p_B_stride_y"]),
                ),
                mul_u24(b_, x, params["p_B_stride_x"]),
            ),
            c,
        )
        return off, b_.cmp_lt(z, p_Z)

    def a_offset_split(b_: IRBuilder, m_val: Value, k_val: Value):
        # Row part: m -> (n, ho, wo), the (hi, wi) origin and the base offset.
        q_wo, wo = magic_divmod(
            b_,
            m_val,
            params["p_magic_m_Wo_mult"],
            params["p_magic_m_Wo_shift"],
            p_Wo,
        )
        n, ho = magic_divmod(
            b_,
            q_wo,
            params["p_magic_m_Ho_mult"],
            params["p_magic_m_Ho_shift"],
            p_Ho,
        )
        hi0 = b_.add(b_.mul(ho, p_sH), neg_pH)
        wi0 = b_.add(b_.mul(wo, p_sW), neg_pW)
        base = b_.add(
            b_.add(b_.mul(n, params["p_A_stride_n"]), b_.mul(hi0, s_hi)),
            b_.mul(wi0, s_wi),
        )
        if grouped:
            base = b_.add(base, a_group_base)
        # k part.
        y, x, c = k_decode(b_, k_val)
        # Plain multiplies on purpose: masking these two kept extra values
        # live and pushed large tiles past an occupancy step.
        hi = b_.add(hi0, b_.mul(y, p_dH))
        wi = b_.add(wi0, b_.mul(x, p_dW))
        ok = b_.land(
            b_.land(b_.cmp_ge(hi, c0), b_.cmp_lt(hi, p_Hi)),
            b_.land(b_.cmp_ge(wi, c0), b_.cmp_lt(wi, p_Wi)),
        )
        ok = b_.land(ok, b_.cmp_lt(y, p_Y))
        off = b_.add(b_.add(b_.add(base, b_.mul(y, a_ystep)), b_.mul(x, a_xstep)), c)
        return off, ok

    def b_offset_split(b_: IRBuilder, k_out: Value, kg: Value):
        base = b_.mul(k_out, params["p_B_stride_k"])
        y, x, c = k_decode(b_, kg)
        off = b_.add(
            b_.add(
                b_.add(base, mul_u24(b_, y, params["p_B_stride_y"])),
                mul_u24(b_, x, params["p_B_stride_x"]),
            ),
            c,
        )
        return off, b_.cmp_lt(y, p_Y)

    schedule = SchedulePolicy.for_pipeline(
        "async_dma" if spec.async_dma else spec.pipeline
    )
    schedule.emit_prologue(b)

    def emit_load_phase(k_off: Value, A_dst: Value, B_dst: Value) -> None:
        """Global -> LDS copy for one K tile via the descriptor DAG.

        For A: each (a_row, a_col) inside the block maps to
            m = block_m_off + a_row
            k = k_off + a_col
        and the A descriptor turns (m, k) -> NHWC linear offset with
        the convolution bounds check (pad on y/x, embed for hi/wi).
        The buffer rsrc is created with flag 0x00027000 which encodes
        proper bounds checking; OOB byte offsets silently return 0
        (the runbook §6.1 lever for tail-safe loads).

        For B: same (b_row, b_col) -> (k_out, k_gemm) -> KYXC linear
        offset, also bounds-checked via the pad on y/x (catches the
        partial-last-tile case).

        `spec.async_dma=True` switches the load path to
        ``AsyncTileLoader`` + ``raw_ptr_buffer_load_lds`` (runbook §6.3).
        Otherwise we use ``CoalescedTileLoader`` which emits the
        register-staged ``buffer_load_vN -> smem_store_vN`` pipeline.
        """
        k_off_capture[0] = k_off

        if spec.async_dma:
            # ``CACHE_STREAM`` (SLC=1) is correct here: each K-tile is
            # consumed exactly once by the MFMA phase of the same iter
            # and then overwritten by the next iter's prefetch. Marking
            # the loads as streaming keeps them from evicting useful
            # cache lines (e.g. the B matrix's columns that *will* be
            # re-read across multiple M-tiles within the same XCD).
            from rocke.core.ir import CACHE_STREAM

            a_slot = a_loader.bind(b, smem_dst=A_dst, wave_id=warp_id)
            a_slot.issue(
                b,
                tid=tid,
                rsrc=a_rsrc,
                descriptor=a_descriptor,
                coherency=CACHE_STREAM,
            )
            b_slot = b_loader.bind(b, smem_dst=B_dst, wave_id=warp_id)
            b_slot.issue(
                b,
                tid=tid,
                rsrc=b_rsrc,
                descriptor=b_descriptor,
                coherency=CACHE_STREAM,
            )
            return

        # Sync path: ``CoalescedTileLoader.load`` emits the per-thread
        # ``buffer_load_vN_f16 -> smem_store_vN_f16`` chunks. Each
        # callback gets ``(row, col)`` inside the tile-local frame and
        # returns the global element offset + validity predicate, so
        # the conv-coord-transform DAG drives the address arithmetic
        # while the loader owns the thread distribution.
        if a_load_override is not None:
            a_load_override(b, spec, k_off, A_dst, grid, input_cache_context)
        else:
            a_sync_loader.load(
                b,
                tid=tid,
                smem_dst=A_dst,
                descriptor=a_descriptor,
                rsrc=a_rsrc,
            )
        b_sync_loader.load(
            b,
            tid=tid,
            smem_dst=B_dst,
            descriptor=b_descriptor,
            rsrc=b_rsrc,
        )

    def emit_wmma_phase(
        A_src: Value, B_src: Value, iter_vars: Sequence[Value]
    ) -> List[Value]:
        """One K-tile of WMMA atoms, fully MMA-contract driven (gfx1151).

        Mirrors ``gemm_universal._emit_wmma_phase``: operand fragments come from
        the op's A/B layout maps and the matmul is emitted target-neutrally via
        :meth:`IRBuilder.mma` (the backend selects the WMMA intrinsic). The A
        map yields ``(row, k)`` and the B map ``(k, col)``; both LDS tiles store
        the M/N index as the row and K as the column, so we read the M-coord
        from the A map and the N-coord (= ``col``) from the B map.
        """
        a_map = op.a_layout()
        b_map = op.b_layout()
        a_row_in_atom, a_k_in_atom = a_map.coord(b, lane, 0)
        b_k_in_atom, b_col_in_atom = b_map.coord(b, lane, 0)
        warp_m_off = grid.warp_m_off(b)
        warp_n_off = grid.warp_n_off(b)
        new_accs: List[Value] = list(iter_vars)
        for kk in range(k_atoms):
            k_tile_base = b.const_i32(kk * spec.warp_tile_k)
            a_rows = []
            for mi in range(mfmas_m):
                atom_row = b.add(warp_m_off, b.const_i32(mi * spec.warp_tile_m))
                a_rows.append(
                    _emit_frag_smem_load(
                        b,
                        A_src,
                        a_row_in_atom,
                        a_k_in_atom,
                        atom_row,
                        k_tile_base,
                        a_per_lane,
                        smem_dtype=_smem_dtype,
                    )
                )
            b_cols = []
            for ni in range(mfmas_n):
                atom_row = b.add(warp_n_off, b.const_i32(ni * spec.warp_tile_n))
                b_cols.append(
                    _emit_frag_smem_load(
                        b,
                        B_src,
                        b_col_in_atom,
                        b_k_in_atom,
                        atom_row,
                        k_tile_base,
                        b_per_lane,
                        smem_dtype=_smem_dtype,
                    )
                )
            flat = 0
            for mi in range(mfmas_m):
                for ni in range(mfmas_n):
                    new_accs[flat] = b.mma(op, a_rows[mi], b_cols[ni], new_accs[flat])
                    flat += 1
        return new_accs

    def lds_swz(b_: IRBuilder, row: Value, col: Value) -> Value:
        """``lds_xor`` column of tile element ``(row, col)``: its 16-byte
        chunk XOR ``row % (tile_k / 8)`` (the in-chunk offset is kept)."""
        # Key the XOR on the row's 256-byte (64-bank) line, not the row: rows
        # of tile_k < 128 elements share a line (2 of 128 B, 4 of 64 B), and
        # keyed by row % (tile_k/8) rows r and r + 256/row_bytes of one
        # 16-row MFMA fragment read hit the same banks.
        if spec.lds_kpack_read:
            # 16-byte reads: key on the 128-byte (32-bank) span.
            # lds_xor_rowkey: key on the row itself at tile_k 32 too (64-byte
            # rows): measured conflict-free for the x-halo loop's shifted
            # A rows, where the 128-byte key leaves ~21% conflict cycles.
            sh = 0 if spec.lds_xor_rowkey else {32: 1}.get(block_k, 0)
        else:
            sh = {32: 2, 64: 1}.get(block_k, 0)
        if sh:
            row = b_.lshr(row, b_.const_i32(sh))
        rmask = b_.land(row, b_.const_i32(block_k // 8 - 1))
        return b_.xor(col, b_.shl(rmask, b_.const_i32(3)))

    def _xa(b_: IRBuilder, row: Value, col: Value):
        """xh_stride_m: LDS (row, col) of x-halo A element (row, col): row bit 0
        flipped by row bit 2 (which half of a 128-byte bank line the row
        takes) and the 16-byte chunk XOR (row / 4) % 4. Searched to be
        conflict-free for ds_read_b128 of 4-strided fragment rows (8-lane
        phases {0-3, 20-23}, {4-7, 16-19}, ...) at every row offset."""
        hi = b_.lshr(row, b_.const_i32(2))
        prow = b_.xor(row, b_.land(hi, b_.const_i32(1)))
        key = b_.land(hi, b_.const_i32(3))
        return prow, b_.xor(col, b_.shl(key, b_.const_i32(3)))

    # a_direct: per lane, the a_per_lane * k_atoms k-contiguous elements of
    # each fragment row, loaded straight into registers; MFMA step kk takes
    # elements [kk * a_per_lane, (kk + 1) * a_per_lane) of them.
    adir_span = a_per_lane * k_atoms
    adir_w = min(8, adir_span)

    def adir_operand(a_regs, mi: int, kk: int) -> Value:
        lo = kk * a_per_lane
        piece = a_regs[mi * (adir_span // adir_w) + lo // adir_w]
        off = lo % adir_w
        if adir_w == a_per_lane:
            return piece
        return b.vec_pack(
            [b.vec_extract(piece, off + i) for i in range(a_per_lane)], ir_dtype_a
        )

    # ``defer_mfma``: while ``on``, emit_mfma_phase reads the fragments of its
    # last ``n`` kk steps but leaves their MFMAs to the caller (``fns``).
    _mfma_defer = {"on": False, "n": 0, "fns": []}
    # rpf_pingpong + pp_b_double: which fragments a pp_read reads.
    _pp_part = {"b_only": False, "b_from": None, "kk_from": 0, "b_ni": None}
    # ``vmem_interleave``: buffer_loads issued just before the next
    # emit_mfma_phase that it spreads over its MFMAs.
    _vmi_pending = [0]

    _xh_lane15 = []
    # xh_stride_m: whether this workgroup ran the x-halo loop (the epilogue's
    # row map), set by _emit_xhalo_loop.
    _xsm_ok = []

    def _eqx_chan(b_: IRBuilder, r: Value) -> Value:
        """epi_quad_xpose 2/3: the output channel (tile-relative) whose
        weights the B LDS row ``r`` holds, so that MFMA column j of N atom ni
        (row 16*ni + j) is channel 32*(ni/2) + 4*(ni%2) + 8*(j/4) + j%4 (2)
        or 32*(ni/2) + 4*(ni%2) + 8*(j%4) + j/4 (3); identity otherwise."""
        if spec.epi_quad_xpose not in (2, 3, 4):
            return r
        p32 = b_.land(r, b_.const_i32(~31))
        ni_lo = b_.land(r, b_.const_i32(16))     # 16 * (ni % 2)
        j = b_.land(r, b_.const_i32(15))
        if spec.epi_quad_xpose == 2:
            jj = b_.add(b_.mul(b_.land(j, b_.const_i32(12)), b_.const_i32(2)),
                        b_.land(j, b_.const_i32(3)))
        else:
            jj = b_.add(b_.mul(b_.land(j, b_.const_i32(3)), b_.const_i32(8)),
                        b_.div(j, b_.const_i32(4)))
        return b_.add(b_.add(p32, b_.div(ni_lo, b_.const_i32(4))), jj)


    def _xh_shift(cur: Value, nxt: Value) -> Value:
        """xh_dpp: fragment ``cur`` one row down (lane i <- lane i+1 of its
        16-lane row; lane 15 <- lane 0 of ``nxt``), per 32-bit word."""
        if not _xh_lane15:
            _xh_lane15.append(
                b.cmp_eq(b.land(lane, b.const_i32(15)), b.const_i32(15))
            )
        n = cur.type.count
        wt = VectorType(I32, n // 2)
        cw = b.vec_bitcast(cur, wt)
        nw = b.vec_bitcast(nxt, wt)
        words = []
        for i in range(n // 2):
            a_ = b.mov_dpp(b.vec_extract(cw, i), row_shl=1)
            n_ = b.mov_dpp(b.vec_extract(nw, i), row_shr=15)
            words.append(b.select(_xh_lane15[0], n_, a_))
        return b.vec_bitcast(b.vec_pack(words, I32), cur.type)

    def _prio_on() -> None:
        # rpf_setprio: raise the wave priority over load issue + MFMAs.
        if spec.rpf_setprio:
            b.s_setprio(spec.rpf_setprio)

    def _prio_off() -> None:
        if spec.rpf_setprio:
            b.s_setprio(0)

    def emit_mfma_phase(
        A_src: Value, B_src: Value, iter_vars: Sequence[Value], a_regs=None,
        after_first_reads=None, a_lds_rows=None, b_row_off: int = 0,
        a_shift_prev=None, a_rec=None, pp_read=None, pp_use=None,
    ) -> List[Value]:
        """One K-tile worth of MFMAs across all per-warp atom positions.

        For the `compv4`/`compv3` pipelines we interleave sched_group_barrier
        hints inside the K-atom loop so the AMDGPU backend overlaps DS_read
        + MFMA + VMEM traffic. The hints don't reorder our SSA — they tell
        the post-RA scheduler what groups to keep together.
        """
        if op.family == "wmma":
            return emit_wmma_phase(A_src, B_src, iter_vars)
        # Tile-level lane decode (CK Tile ``BlockGemmAdaptor`` analogue):
        #   16x16:  m_in_atom = lane % 16,  k_blk = lane / 16,  n_in_atom = lane % 16
        #   32x32:  m_in_atom = lane % 32,  k_blk = lane / 32,  n_in_atom = lane % 32
        # ``decode_mfma_lanes`` returns a frozen :class:`LaneDecode`
        # carrying these as named SSA fields, so the MFMA loop never
        # mis-derives them (a perennial copy-paste hazard).
        decoded = decode_mfma_lanes(b, atom, lane)
        m_in_atom = decoded.m_in_atom
        n_in_atom = decoded.n_in_atom
        k_blk = decoded.k_blk

        warp_m_off = grid.warp_m_off(b)
        warp_n_off = grid.warp_n_off(b)

        new_accs: List[Value] = list(iter_vars)

        _kp_cache = {}

        def load_frags(kk: int):
            """The A rows and B columns one kk step of MFMAs consumes."""
            if spec.lds_kpack_read and a_regs is None and a_operand_override is None:
                # k-packed: lane group k_blk owns 2*per_lane contiguous k of
                # the kk pair (2p, 2p+1); one 16-byte read, split in halves.
                if kk % 2:
                    return _kp_cache.pop(kk)
                cb = b.add(
                    b.mul(k_blk, b.const_i32(2 * a_per_lane)),
                    b.const_i32(kk * spec.warp_tile_k),
                )
                pairs = ([], [])
                _skip_a = _pp_part["b_only"]
                _skip_b = _pp_part["b_from"] is not None and kk >= _pp_part["b_from"]
                for mi in range(0 if _skip_a else mfmas_m):
                    if a_shift_prev is not None and mi < mfmas_m - 1:
                        prev = a_shift_prev[("p", kk)]
                        pairs[0].append(prev[mi + 1] if spec.xh_stride_m
                                        else _xh_shift(prev[mi], prev[mi + 1]))
                        continue
                    a_row = a_lds_rows[mi] if a_lds_rows is not None else b.add(
                        warp_m_off, b.add(b.const_i32(mi * spec.warp_tile_m), m_in_atom)
                    )
                    if spec.xh_stride_m and a_lds_rows is not None:
                        v = _emit_smem_load(b, A_src, *_xa(b, a_row, cb),
                                            2 * a_per_lane, smem_dtype=_smem_dtype)
                    else:
                        v = _emit_smem_load(
                            b, A_src, a_row,
                            lds_swz(b, a_row, cb) if spec.lds_xor else cb,
                            2 * a_per_lane, smem_dtype=_smem_dtype,
                        )
                    pairs[0].append(v)
                if a_rec is not None:
                    a_rec[("p", kk)] = list(pairs[0])
                b_pairs = []
                _b_lo, _b_hi = _pp_part["b_ni"] or (0, mfmas_n)
                for ni in range(_b_lo, _b_lo if _skip_b else _b_hi):
                    b_row = b.add(
                        warp_n_off,
                        b.add(b.const_i32(ni * spec.warp_tile_n + b_row_off), n_in_atom),
                    )
                    b_pairs.append(_emit_smem_load(
                        b, B_src, b_row,
                        lds_swz(b, b_row, cb) if spec.lds_xor else cb,
                        2 * b_per_lane, smem_dtype=_smem_dtype,
                    ))

                def half(v, n, h):
                    return b.vec_pack(
                        [b.vec_extract(v, h * n + i) for i in range(n)], ir_dtype_a
                    )

                _kp_cache[kk + 1] = (
                    [half(v, a_per_lane, 1) for v in pairs[0]],
                    [half(v, b_per_lane, 1) for v in b_pairs],
                )
                return (
                    [half(v, a_per_lane, 0) for v in pairs[0]],
                    [half(v, b_per_lane, 0) for v in b_pairs],
                )
            if a_regs is not None:
                # Permuted k order (see adir_operand): lane group k_blk owns
                # the adir_span contiguous elements starting at k_blk*span.
                col_base = b.add(
                    b.mul(k_blk, b.const_i32(adir_span)),
                    b.const_i32(kk * a_per_lane),
                )
            else:
                col_base = b.add(
                    b.mul(k_blk, b.const_i32(a_per_lane)),
                    b.const_i32(kk * spec.warp_tile_k),
                )
            a_rows = []
            # pp_b_double without k-packed reads: [mem B] reads the B columns
            # only (b_only); [mem A] the B columns of kk < b_from only.
            _np_skip_a = _pp_part["b_only"] and a_regs is None and a_operand_override is None
            _np_skip_b = _pp_part["b_from"] is not None and kk >= _pp_part["b_from"]
            for mi in range(0 if _np_skip_a else mfmas_m):
                a_row = a_lds_rows[mi] if a_lds_rows is not None else b.add(
                    warp_m_off, b.add(b.const_i32(mi * spec.warp_tile_m), m_in_atom)
                )
                if a_regs is not None:
                    a_rows.append(adir_operand(a_regs, mi, kk))
                elif a_operand_override is not None:
                    a_rows.append(
                        a_operand_override(
                            b,
                            spec,
                            a_row,
                            k_off_capture[0],
                            col_base,
                            a_per_lane,
                            grid,
                            input_cache_context,
                        )
                    )
                elif a_shift_prev is not None and mi < mfmas_m - 1:
                    prev = a_shift_prev[("k", kk)]
                    a_rows.append(prev[mi + 1] if spec.xh_stride_m
                                  else _xh_shift(prev[mi], prev[mi + 1]))
                elif spec.xh_stride_m and a_lds_rows is not None:
                    a_rows.append(_emit_smem_load(b, A_src, *_xa(b, a_row, col_base),
                                                  a_per_lane, smem_dtype=_smem_dtype))
                else:
                    a_rows.append(
                        _emit_smem_load(
                            b,
                            A_src,
                            a_row,
                            lds_swz(b, a_row, col_base) if spec.lds_xor else col_base,
                            a_per_lane,
                            smem_dtype=_smem_dtype,
                        )
                    )
            if a_rec is not None:
                a_rec[("k", kk)] = list(a_rows)

            b_cols = []
            for ni in range(0 if _np_skip_b else mfmas_n):
                b_row = b.add(
                    warp_n_off,
                    b.add(b.const_i32(ni * spec.warp_tile_n + b_row_off), n_in_atom),
                )
                b_cols.append(
                    _emit_smem_load(
                        b,
                        B_src,
                        b_row,
                        lds_swz(b, b_row, col_base) if spec.lds_xor else col_base,
                        b_per_lane,
                        smem_dtype=_smem_dtype,
                    )
                )
            return a_rows, b_cols

        if pp_read is not None:
            # rpf_pingpong [mem A]: every fragment of the tile, no MFMA
            # (with pp_b_double: A all + B of the first half of the kk steps;
            # the other B columns in [mem B], _pp_part["b_only"]).
            for kk in range(_pp_part["kk_from"], k_atoms):
                pp_read.append(load_frags(kk))
            return new_accs
        if pp_use is not None:
            # rpf_pingpong [compute A/B]: MFMAs [i0, i1) of the tile, counted
            # kk-major (then mi, ni).
            frags_pp, i0, i1 = pp_use
            per_kk = mfmas_m * mfmas_n
            for i in range(i0, i1):
                kk, flat = divmod(i, per_kk)
                mi, ni = divmod(flat, mfmas_n)
                a_rows, b_cols = frags_pp[kk]
                new_accs[flat] = _emit_mfma(
                    b, atom, a_rows[mi], b_cols[ni], new_accs[flat]
                )
            return new_accs

        # ``lds_ahead`` (identity field, AOT axis ROCKE_AOT_FWD_SCHED; it
        # replaces the former ROCKE_FWD_LDS_AHEAD env gate): the LDS fragment
        # reads run one kk step ahead -- step kk+1's ds_reads issue before
        # step kk's MFMAs, so the LDS latency hides behind them instead of an
        # lgkmcnt(0) wait per step.
        _lds_ahead = spec.lds_ahead and a_operand_override is None
        _kprd = spec.lds_kpack_read and a_regs is None and a_operand_override is None

        def _n_dsr(kk: int) -> int:
            # DS reads one kk step issues (k-packed: both steps' at even kk).
            _na = 1 if (spec.xh_stride_m and a_shift_prev is not None) else mfmas_m
            if _kprd:
                return 0 if kk % 2 else _na + mfmas_n
            return mfmas_n if a_regs is not None else _na + mfmas_n
        frags = load_frags(0) if _lds_ahead else None
        for kk in range(k_atoms):
            if _lds_ahead:
                a_rows, b_cols = frags
                if kk + 1 < k_atoms:
                    frags = load_frags(kk + 1)
            else:
                a_rows, b_cols = load_frags(kk)
            if kk == 0 and after_first_reads is not None:
                # rpf_fetch_late: the next tile's global reads go here, behind
                # the first LDS fragment reads and ahead of every MFMA.
                b.sched_barrier(0)
                after_first_reads()

            if _mfma_defer["on"] and kk >= k_atoms - _mfma_defer["n"]:
                # defer_mfma: the fragments are read now; their MFMAs are
                # emitted by the caller after the barrier, interleaved with the
                # next tile's LDS store.
                def _deferred(accs_in, a_rows=a_rows, b_cols=b_cols):
                    out = list(accs_in)
                    flat_ = 0
                    for mi_ in range(mfmas_m):
                        for ni_ in range(mfmas_n):
                            out[flat_] = _emit_mfma(
                                b, atom, a_rows[mi_], b_cols[ni_], out[flat_]
                            )
                            flat_ += 1
                    return out

                _mfma_defer["fns"].append(_deferred)
                continue

            flat = 0
            for mi in range(mfmas_m):
                for ni in range(mfmas_n):
                    acc = _emit_mfma(b, atom, a_rows[mi], b_cols[ni], new_accs[flat])
                    new_accs[flat] = acc
                    flat += 1

            # sched_group_barrier hints for `compv3`/`compv4`. The group
            # masks follow the runbook §7.3 convention:
            #   0x100 = DS_READ, 0x008 = MFMA, 0x020 = DS_WRITE, 0x040 = VMEM
            #
            # We tell the scheduler: one group of (mfmas_m + mfmas_n)
            # DS_READs (one per row of A and one per col of B inside
            # this kk step), then one group of mfmas_m * mfmas_n
            # MFMAs. Forcing DS_READs ahead of MFMAs gives the
            # backend latitude to issue the LDS reads early so the
            # MFMA pipeline doesn't stall waiting for operands.
            #
            # Tried alternative patterns:
            # - 1+1 paired (DS+MFMA+DS+MFMA...): 175 TFLOPS (worse).
            # - none:                            tested below.
            # - 1 DS group + 1 MFMA group:       186 TFLOPS (best).
            n_ld = _vmi_pending[0]
            if n_ld and (schedule.emit_hints or spec.uk_xhalo):
                # vmem_interleave: the next tile's buffer_loads, issued just
                # before this phase, go one per ``vmem_interleave`` MFMAs.
                b.sched_group_barrier(
                    DS_READ, _n_dsr(kk), 0
                )
                n_mf = mfmas_m * mfmas_n
                while n_ld and n_mf:
                    b.sched_group_barrier(VMEM_READ, 1, 0)
                    take = min(spec.vmem_interleave, n_mf)
                    b.sched_group_barrier(MFMA, take, 0)
                    n_mf -= take
                    n_ld -= 1
                if n_mf:
                    b.sched_group_barrier(MFMA, n_mf, 0)
                _vmi_pending[0] = n_ld
            else:
                schedule.emit_after_mfma_step(
                    b,
                    ds_read_count=_n_dsr(kk),
                    mfma_count=mfmas_m * mfmas_n,
                )

        _vmi_pending[0] = 0
        return new_accs

    # ---- the K loop ----
    # Three code paths:
    #
    # 1) Sync path (`async_dma=False`, `unroll_k=False`): emit a
    #    single `scf.for_iter` body (load + barrier + MFMA + barrier).
    #    No software pipelining; each iter waits for its own load.
    # The K-loop *structure* determines the branch — not the pipeline string.
    # "mem", "compv3", and "compv4" all share the same scf.for_iter shape;
    # their differences (scheduling hints, double-buffering) are handled inside
    # emit_mfma_phase and the LDS allocation, not by the K-loop itself.
    # A new branch is only introduced when the loop structure itself changes.
    #
    # 2) unroll_k: double-buffer ping-pong, the scf.for_iter body unrolled
    #    twice. Stage tile t+1 into the alternate buffer while MFMA runs on
    #    tile t.
    #
    # 3) not async_dma (mem/compv3/compv4/basic): single scf.for_iter with
    #    emit_load_phase -> sync -> emit_mfma_phase -> sync per tile.
    #

    # ---- D descriptor + epilogue dispatch (shared by every K-loop driver) ----
    # Built before the K-loop so the wavelet branch, which emits its epilogue
    # from inside the math-wave region, runs exactly the same store code as
    # the straight-line drivers instead of a second compile-time copy.
    if p.is_pointwise:
        D_desc = None
        _always_valid_d = b.const_i32(1)

        def d_addr(b_: IRBuilder, m_val: Value, n_val: Value):
            # Pointwise D is a flat [M, kpg] matrix; no pad guards apply.
            return b_.add(b_.mul(m_val, p_kpg), n_val), _always_valid_d

    else:
        D_desc = make_d_descriptor_dynamic(b, params, is_3d=p.is_3d)
        # Grouped conv: the per-warp N coord is within-group (n_val < kpg);
        # recover the absolute N[D]HWK output filter k_out = g*kpg + n_val.
        # Merged: g is the merged group and the merged kpg is Gm, so the same
        # expression yields k_out = merged_group*Gm + g_n.
        _k_out_grp_base = (
            (
                b.mul(group_idx, p_kpg)
                if _split
                else b.mul(b.block_id_z(), c_gm if merged else p_kpg)
            )
            if grouped
            else None
        )

        def d_addr(b_: IRBuilder, m_val: Value, n_val: Value):
            k_out = n_val if _k_out_grp_base is None else b_.add(_k_out_grp_base, n_val)
            return D_desc.offset(b_, m=m_val, k_out=k_out)

    # The GEMM-N extent the tile covers: kpg, or kpg*Gm == Gm under merge. The
    # true kpg (1 on depthwise) would clamp away Gm-1 of every Gm real outputs
    # -- silently, as a wrong answer rather than a crash.
    c_N_bound = c_gm if merged else p_kpg
    # The cshuffle store width when the spec leaves it open: the widest power of
    # two dividing the per-group output channel run, the same width
    # ``is_valid_spec`` derives (and the C++ engine emits). Under merge that run
    # is the merged kpg, Gm. A fixed width wider than the run would fail the
    # epilogue's whole-vector bound check and drop the outputs it covers.
    if spec.vector_size_c is not None:
        _max_store_vec = spec.vector_size_c
    else:
        _max_store_vec = ImplicitGemmConvSpec.default_vector_sizes(
            p_load.cpg, p_load.kpg, spec.data.dtype_d
        )[2]

    def _emit_fwd_epilogue_qx(final_accs_in):
        """epi_quad_xpose: register-only store of the 16x16-atom accumulators.

        Atom layout: lane l, slot i holds C[4*(l/16) + i][l % 16]. Within a
        quad (lanes 4j..4j+3, q = l % 4) the four lanes hold four consecutive
        columns of the same four rows; a 4x4 transpose (two DPP quad_perm
        exchanges) leaves lane q with row 4*(l/16) + q, columns
        (l % 16) & ~3 .. +3, which one 8-byte store writes."""
        mfmas_m = spec.mfmas_per_warp_m
        mfmas_n = spec.mfmas_per_warp_n
        _dt = BF16 if spec.data.dtype_d == "bf16" else _F16
        # Mode 3 transposes across lanes {l, l^4, l^8, l^12} instead of the
        # quad: the "q" bits are lane bits 2-3, so lanes l..l+3 end on one
        # output pixel and consecutive 16-byte chunks (coalesced stores).
        x3 = spec.epi_quad_xpose in (3, 4)
        fast_cvt = spec.epi_quad_xpose == 4 and spec.data.dtype_d == "bf16"
        c1_ = b.const_i32(4 if x3 else 1)
        c2_ = b.const_i32(8 if x3 else 2)
        q_hi = b.cmp_ne(b.land(lane, c2_), c0)       # q in {2, 3}
        q_odd = b.cmp_ne(b.land(lane, c1_), c0)      # q in {1, 3}

        def _xchg1(v):
            if not x3:
                return b.dpp_xor(v, 1)
            # lane ^ 4 inside the 16-lane row: row_shl:4 / row_shr:4.
            return b.select(q_odd, b.mov_dpp(v, row_shr=4), b.mov_dpp(v, row_shl=4))
        # odd lanes send (A.lo, B.lo), even (A.hi, B.hi); then
        # even: o0 = (A.lo, T.lo), o1 = (B.lo, T.hi);
        # odd:  o0 = (T.lo, A.hi), o1 = (T.hi, B.hi).
        sel_s = b.select(q_odd, b.const_i32(0x05040100), b.const_i32(0x07060302))
        sel_o0 = b.select(q_odd, b.const_i32(0x03020504), b.const_i32(0x05040100))
        sel_o1 = b.select(q_odd, b.const_i32(0x03020706), b.const_i32(0x07060100))
        if x3:
            row_in = b.add(b.mul(b.div(lane, b.const_i32(16)), b.const_i32(4)),
                           b.land(b.div(lane, b.const_i32(4)), b.const_i32(3)))
            # lane's channels: 32*p + 8*(l & 3) + 0..7 -> col_in + col_in = 8*(l & 3)
            col_in = b.mul(b.land(lane, b.const_i32(3)), b.const_i32(4))
        else:
            row_in = b.add(b.mul(b.div(lane, b.const_i32(16)), b.const_i32(4)), b.land(lane, b.const_i32(3)))
            col_in = b.land(lane, b.const_i32(12))
        m_base = b.add(b.add(block_m_off_v, grid.warp_m_off(b)), row_in)
        n_base = b.add(b.add(block_n_off_v, grid.warp_n_off(b)), col_in)
        c_eb_d = b.const_i32(2)
        oob = b.const_i32((1 << 31) - 1)
        wt2 = VectorType(I32, 2)
        wide = spec.epi_quad_xpose in (2, 3, 4)
        c_lsb_sh = b.const_i32(16)
        c1_one = b.const_i32(1)
        c_7fff = b.const_i32(0x7FFF)
        c_qnan = b.const_i32(0x7FC00000)
        s_hi2 = b.const_i32(0x07060302)

        def _cvt_pair(acc, i0):
            # bf16 round-to-nearest-even of elements i0, i0+1, packed (lo =
            # i0): u + 0x7fff + lsb, NaN -> canonical qNaN.
            ts = []
            for i in (i0, i0 + 1):
                f = b.vec_extract(acc, i)
                u = b.bitcast(f, I32)
                lsb = b.land(b.lshr(u, c_lsb_sh), c1_one)
                t = b.add(b.add(u, c_7fff), lsb)
                ts.append(b.select(b.fcmp("uno", f, f), c_qnan, t))
            return b.perm_b32(ts[1], ts[0], s_hi2)
        if wide:
            # lane channels of the atom pair: 32*p + 2*(l & 12) + 0..7
            n_base = b.add(b.add(block_n_off_v, grid.warp_n_off(b)), b.add(col_in, col_in))
        if spec.xh_stride_m:
            m_wave = b.add(block_m_off_v, grid.warp_m_off(b))
            m_str = b.add(m_wave, b.mul(row_in, b.const_i32(mfmas_m)))
        for mi in range(mfmas_m):
            m_val = b.add(m_base, b.const_i32(mi * 16))
            if spec.xh_stride_m:
                m_val = b.select(_xsm_ok[0], b.add(m_str, b.const_i32(mi)), m_val)
            off0, valid = d_addr(b, m_val, n_base)
            m_ok = b.cmp_lt(m_val, p_M)
            if valid is not None:
                m_ok = b.land(m_ok, valid)
            _pend = []
            for ni in range(mfmas_n):
                acc = final_accs_in[mi * mfmas_n + ni]
                if fast_cvt:
                    w0 = _cvt_pair(acc, 0)
                    w1 = _cvt_pair(acc, 2)
                else:
                    w = b.vec_bitcast(b.vec_cast_f32_to(acc, _dt), wt2)
                    w0 = b.vec_extract(w, 0)   # rows 0, 1 (lo, hi)
                    w1 = b.vec_extract(w, 1)   # rows 2, 3
                # stage 1 (lanes q ^ 2): q < 2 keeps rows 0,1, q >= 2 rows 2,3.
                snd = b.select(q_hi, w0, w1)
                rcv = b.dpp_xor(snd, 8 if x3 else 2)
                a_ = b.select(q_hi, rcv, w0)    # column with bit 1 clear
                bv = b.select(q_hi, w1, rcv)    # column with bit 1 set
                # stage 2 (lanes q ^ 1): even lanes keep row lo, odd row hi;
                # the per-lane byte selectors pick the halves (v_perm_b32:
                # bytes 0-3 = src1, 4-7 = src0).
                t_ = _xchg1(b.perm_b32(bv, a_, sel_s))
                o0 = b.perm_b32(t_, a_, sel_o0)
                o1 = b.perm_b32(t_, bv, sel_o1)
                if wide and ni % 2 == 0:
                    _pend = [o0, o1]
                    continue
                words = _pend + [o0, o1] if wide else [o0, o1]
                nw = len(words)
                n_rel = 32 * (ni // 2) if wide else 16 * ni
                n_val = b.add(n_base, b.const_i32(n_rel))
                ok = b.land(m_ok, b.cmp_le(b.add(n_val, b.const_i32(2 * nw)), c_N_bound))
                off = b.add(off0, b.const_i32(n_rel))
                safe = b.select(ok, b.mul(off, c_eb_d), oob)
                v = b.vec_bitcast(b.vec_pack(words, I32), VectorType(_dt, 2 * nw))
                if _dt == BF16:
                    b.buffer_store_vN_bf16(d_rsrc, safe, c0, v, nw)
                else:
                    b.buffer_store_vN_f16(d_rsrc, safe, c0, v, nw)

    def _emit_fwd_epilogue(final_accs_in):
        """Store the accumulators through ``d_addr`` with runtime bounds."""
        if spec.epi_quad_xpose:
            return _emit_fwd_epilogue_qx(final_accs_in)
        if spec.epilogue == "cshuffle":
            cshuffle_kwargs = {
                "out_dtype": spec.data.dtype_d,
                "max_store_vec": _max_store_vec,
                "no_alias": spec.cshuffle_no_alias,
            }
            if op.family == "wmma":
                epi = CShuffleEpilogue.from_grid_op(op=op, grid=grid, **cshuffle_kwargs)
            else:
                epi = CShuffleEpilogue.from_grid(
                    atom=spec.atom, grid=grid, **cshuffle_kwargs
                )
            # Wavelet needs two WAR barriers, not one: the load waves overwrite
            # the A/B LDS that the cshuffle store then reuses, so a single
            # barrier before the store leaves the re-read racing them.
            epi = dc_replace(epi, war_barriers=(2 if spec.pipeline == "wavelet" else 1))
            if spec.cshuffle_m_chunks > 1:
                epi = dc_replace(epi, m_chunks=spec.cshuffle_m_chunks)
            if _split and spec.split_atomic_agent:
                epi = dc_replace(epi, atomic_syncscope="agent")
            if _split and spec.split_atomic_pairs:
                epi = dc_replace(epi, atomic_pair_lanes=True)
            if spec.epi_lds_barriers:
                epi = dc_replace(epi, lds_barriers=True)
            if _split and spec.split_atomic_fast:
                epi = dc_replace(epi, atomic_fast_full=True)
            if spec.epi_store_fast and not p.is_pointwise:
                # Packed N[D]HWK: offset = m * K + k_out.
                _pk = b.land(
                    b.cmp_eq(p_D_stride_wo, p_K),
                    b.cmp_eq(p_D_stride_ho, b.mul(p_Wo, p_K)),
                )
                if p.is_3d:
                    _pk = b.land(_pk, b.cmp_eq(p_D_stride_do, b.mul(p_Ho, p_D_stride_ho)))
                    _pk = b.land(_pk, b.cmp_eq(params["p_D_stride_n"], b.mul(p_Do, p_D_stride_do)))
                else:
                    _pk = b.land(_pk, b.cmp_eq(params["p_D_stride_n"], b.mul(p_Ho, p_D_stride_ho)))
                _pk = b.land(_pk, b.cmp_le(b.add(block_m_off_v, b.const_i32(block_m)), p_M))
                _pk = b.land(
                    _pk, b.cmp_le(b.add(block_n_off_v, b.const_i32(block_n)), c_N_bound)
                )
                epi = dc_replace(
                    epi, fast_ok=_pk, fast_row_pitch=p_K, fast_col_base=_k_out_grp_base
                )
            if _split:
                # Each K slice adds its partial tile into the zeroed D. D's
                # row (output position) stride is the total K; the group's
                # output-channel slab starts at group * kpg.
                def _atomic_plain():
                    epi.atomic_store(
                        b,
                        accs=final_accs_in,
                        dw_ptr=D,
                        wg_N=p_K,
                        bounds=(p_M, c_N_bound),
                        col_offset=_k_out_grp_base,
                    )

                def _atomic():
                    if spec.split_zero_first:
                        # The tile was zeroed by its slice 0 at kernel start:
                        # wait for that (normally long done), add, and the
                        # last adder resets the flag pair.
                        c_1 = b.const_i32(1)
                        f_idx = _szf_idx[0]
                        with b.scf_if(b.cmp_eq(tid, c0)):
                            b.global_flag_wait_eq(p_sk_flags, f_idx, c_1, ordering="monotonic")
                            b.fence(scope="agent", ordering="acquire")
                        b.sync()
                        _atomic_plain()
                        with b.scf_if(b.cmp_eq(tid, c0)):
                            n_idx = b.add(f_idx, c_1)
                            old = b.global_atomic_add(p_sk_flags, n_idx, c_1)
                            with b.scf_if(b.cmp_eq(old, b.sub(p_ks_count, c_1))):
                                b.global_flag_store(p_sk_flags, n_idx, c0, ordering="monotonic")
                                b.global_flag_store(p_sk_flags, f_idx, c0, ordering="monotonic")
                        return
                    if not spec.split_flag_store:
                        _atomic_plain()
                        return
                    # Slice 0 stores and raises the flag; the rest wait, add.
                    c_1 = b.const_i32(1)
                    t_lin = b.add(
                        b.mul(b.div(block_m_off_v, b.const_i32(block_m)), p_num_pid_n),
                        b.div(block_n_off_v, b.const_i32(block_n)),
                    )
                    if grouped:
                        t_lin = b.add(t_lin, b.mul(group_idx, b.mul(p_num_pid_m, p_num_pid_n)))
                    f_idx = b.to_sgpr_u32(b.mul(t_lin, b.const_i32(2)))
                    n_idx = b.add(f_idx, c_1)
                    with b.scf_if_else(b.cmp_eq(_slice, c0)) as (first_ctx, rest_ctx):
                        with first_ctx:
                            epi.store(
                                b,
                                accs=final_accs_in,
                                addr_fn=d_addr,
                                d_rsrc=d_rsrc,
                                bounds=(p_M, c_N_bound),
                            )
                            # Every wave's stores are issued; release them.
                            b.fence(scope="agent", ordering="release")
                            b.sync()
                            with b.scf_if(b.cmp_eq(tid, c0)):
                                b.global_flag_store(p_sk_flags, f_idx, c_1)
                        with rest_ctx:
                            # One lane polls relaxed, then one acquire; the
                            # block waits for it at the barrier.
                            with b.scf_if(b.cmp_eq(tid, c0)):
                                b.global_flag_wait_eq(
                                    p_sk_flags, f_idx, c_1, ordering="monotonic"
                                )
                                b.fence(scope="agent", ordering="acquire")
                                old = b.global_atomic_add(p_sk_flags, n_idx, c_1)
                                # The last of the ks_count - 1 adders resets.
                                with b.scf_if(b.cmp_eq(old, b.sub(p_ks_count, b.const_i32(2)))):
                                    b.global_flag_store(p_sk_flags, n_idx, c0, ordering="monotonic")
                                    b.global_flag_store(p_sk_flags, f_idx, c0, ordering="monotonic")
                            b.sync()
                            _atomic_plain()

                if not spec.split_tail:
                    _atomic()
                    return
                # split_tail: a whole-K tile stores plainly; a tail slice adds.
                with b.scf_if_else(_is_full) as (then_ctx, else_ctx):
                    with then_ctx:
                        epi.store(
                            b,
                            accs=final_accs_in,
                            addr_fn=d_addr,
                            d_rsrc=d_rsrc,
                            bounds=(p_M, c_N_bound),
                        )
                    with else_ctx:
                        _atomic()
                return
            epi.store(
                b,
                accs=final_accs_in,
                addr_fn=d_addr,
                d_rsrc=d_rsrc,
                bounds=(p_M, c_N_bound),
            )
        elif op.family == "wmma":
            # RDNA/WMMA keeps the explicit per-warp/lane decomposition; it
            # predates the helper-based path.
            emit_direct_epilogue_wmma(
                b,
                spec,
                op,
                final_accs_in,
                warp_m_idx,
                warp_n_idx,
                lane,
                block_m_off_v,
                block_n_off_v,
                d_rsrc,
                c0,
                addr_fn=d_addr,
                p_M=p_M,
                p_kpg=c_N_bound,
            )
        else:
            DirectEpilogue(
                atom=spec.atom, grid=grid, out_dtype=spec.data.dtype_d
            ).store(
                b,
                accs=final_accs_in,
                addr_fn=d_addr,
                d_rsrc=d_rsrc,
                bounds=(p_M, c_N_bound),
            )

    if spec.pipeline == "wavelet":
        # ----------------------------------------------------------------
        # WAVELET pipeline — load/math wave specialization (CK Tile #8009).
        #
        # Reference: gemm_pipeline_ag_bg_cr_wavelet.hpp
        #
        # The workgroup is split into two roles identified by warp_id:
        #   math waves  [0,          n_math_warps)  — LDS reads + MFMA + epilogue
        #   load waves  [n_math_warps, total_warps)  — DRAM→register fetch + LDS write
        #
        # Both roles share a single LDS buffer (no ping-pong).
        #
        # Barrier protocol (must be bit-identical in both branches):
        #
        #   MATH branch:
        #     barrier_0
        #     for i in 0..K-2:
        #       MFMA(LDS)
        #       barrier_A          ← math done reading LDS
        #       barrier_B          ← wait for load to write next tile
        #     MFMA(LDS)            ← tail, no barriers
        #     [epilogue — emits N_epi barriers]
        #
        #   LOAD branch:
        #     fetch tile 0 → regs
        #     store regs → LDS
        #     barrier_0
        #     for i in 0..K-2:
        #       fetch tile i+1 → regs   ← overlaps math MFMA
        #       barrier_A                ← wait for math to release LDS
        #       store regs → LDS
        #       barrier_B                ← signal LDS ready
        #     [epilogue stub — N_epi bare barriers, no stores]
        #
        # Total barriers per branch: 1 + 2*(K-1) + N_epi
        #
        # N_epi = 3 for cshuffle (step-0 WAR#1 + step-1 WAR#2 + step-2 LDS RAW),
        #         1 for cshuffle_no_alias (step-2 RAW only),
        #         0 for default/wmma direct epilogues.
        # epilogue_override is incompatible with wavelet (unknown N_epi).
        # ----------------------------------------------------------------
        if epilogue_override is not None:
            raise ValueError(
                "pipeline='wavelet' does not support epilogue_override "
                "(barrier count of the override is unknown)"
            )
        if a_operand_override is not None:
            raise ValueError(
                "pipeline='wavelet' does not support a_operand_override on WMMA: "
                "emit_wmma_phase uses fragment loads from LDS directly and does "
                "not consult a_operand_override."
            )

        if op.family == "mma":
            # The exec-mask MFMA wavelet path has an unfixable data race:
            # the math section reads a single-buffer LDS that the load section
            # overwrites every K iteration. True concurrent execution (different
            # SIMD units) is required, but rocke emits flat sequential code with
            # no cross-section barriers, so math reads garbage after iteration 0.
            # is_valid_spec already blocks this via the WMMA-only guard, so this
            # branch is unreachable in normal use. Raise here to make any direct
            # call fail loudly instead of silently producing wrong results.
            raise ValueError(
                "pipeline='wavelet' is not supported for MFMA/CDNA targets: "
                "the exec-mask split path has a data race (single LDS buffer "
                "overwritten each K iteration with no cross-section barriers). "
                "Use is_valid_spec() to reject this combination before building."
            )

        # WMMA path: scf_if_else (gfx1250).
        #
        # gfx1250 has separate VMEM and WMMA issue slots, so load and math
        # waves truly overlap without exec-mask interleaving. The LLVM br i1
        # divergent branch is fine here — the hardware concurrency comes
        # from the distinct issue queues, not from hardware scheduling at
        # s_barrier. Keep scf_if_else to avoid changing the validated path.
        n_math_warps = spec.warp_m * spec.warp_n
        launch_block = spec.launch_block_size
        b.kernel.attrs["max_workgroup_size"] = launch_block

        _epi_barriers = compute_wavelet_epi_barriers(
            spec.epilogue, spec.cshuffle_no_alias
        )

        def _fwd_epilogue_with_acc_transform(accs_in):
            _emit_fwd_epilogue(
                _apply_accumulator_epilogue(b, spec.acc_epilogue, accs_in)
            )

        emit_wavelet_kloop_dynamic(
            b=b,
            warp_id=warp_id,
            tid=tid,
            n_math_warps=n_math_warps,
            math_block_size=spec.block_size,
            k_lo=c0,
            k_hi=c_K_gemm,
            block_k=block_k,
            A_smem=A_smem,
            B_smem=B_smem,
            a_wavelet_loader=a_wavelet_loader,
            b_wavelet_loader=b_wavelet_loader,
            a_descriptor=a_descriptor,
            b_descriptor=b_descriptor,
            a_rsrc=a_rsrc,
            b_rsrc=b_rsrc,
            k_off_capture=k_off_capture,
            accs=accs,
            emit_mfma_phase=emit_mfma_phase,
            emit_epilogue_fn=_fwd_epilogue_with_acc_transform,
            epi_barriers=_epi_barriers,
        )
        return b.kernel

    def _emit_fwd_kloops() -> List[Value]:
        """The ``uniform_k_decode`` / ``reg_prefetch`` K loops (final accs).

        Up to two loops over the same reduction, exactly one of which has a
        non-empty range for a given problem:

        * the uniform-tap loop (``uniform_k_decode``), taken when cpg is a
          multiple of tile_k. It carries the tile's channel base and filter tap
          ``(c, x, y[, z])`` as uniform counters and advances them by one tile
          per iteration, so the per-element address is a loop-invariant pixel
          part plus a uniform tap part and the per-element ``k -> (z, y, x, c)``
          division chain disappears. Its bounds test is an unsigned compare of
          the invariant pixel origin shifted by the uniform tap offset.
        * the plain loop over the descriptor DAG otherwise (or always, when
          only ``reg_prefetch`` is set).

        With ``reg_prefetch`` the global reads of tile t+1 are issued into
        registers ahead of the MFMAs of tile t (pinned there with
        ``sched_barrier``) and written to LDS after them. ``reg_prefetch=1``
        keeps one LDS buffer (a barrier before the write retires the tile's
        ds_reads, a second publishes it); ``reg_prefetch=2`` ping-pongs two
        buffers in an x2-unrolled loop, one barrier per tile, with an odd last
        tile in a 0/1-trip tail loop. A prefetch past the end of the reduction
        addresses an out-of-range tap / k and loads zeros without touching
        memory.
        """
        if (
            a_load_override is not None
            or a_operand_override is not None
            or m_index_fn is not None
            or a_mhw_index_fn is not None
        ):
            raise ValueError(
                "uniform_k_decode/reg_prefetch do not support the builder's "
                "A-operand override hooks"
            )
        c1 = b.const_i32(1)
        is3 = p.is_3d
        state0: List[Tuple[str, Value]] = []
        uk_cap: List[Optional[Tuple[Value, ...]]] = [None]

        def adir_load(desc_a) -> list:
            # A straight into the MFMA operand registers: lane (m_in_atom,
            # k_blk) loads, for each of its fragment rows, the adir_span
            # k-contiguous elements at k_blk * adir_span (adir_w per load).
            dec = decode_mfma_lanes(b, atom, lane)
            wm = grid.warp_m_off(b)
            col0 = b.mul(dec.k_blk, b.const_i32(adir_span))
            c_eb = b.const_i32(2)
            c_oob = b.const_i32(a_sync_loader.oob_sentinel)
            c0_ = b.const_i32(0)
            regs = []
            for mi in range(mfmas_m):
                row = b.add(wm, b.add(b.const_i32(mi * spec.warp_tile_m), dec.m_in_atom))
                for pc in range(adir_span // adir_w):
                    col = b.add(col0, b.const_i32(pc * adir_w)) if pc else col0
                    off, ok = desc_a(b, row, col)
                    safe = b.select(ok, b.mul(off, c_eb), c_oob)
                    regs.append(b.buffer_load_vN(a_rsrc, safe, c0_, ir_dtype_a, adir_w))
            return regs

        def fetch_adir(desc_a, desc_b) -> Tuple[list, list]:
            # B first: its LDS write waits for it alone, while A's loads stay
            # in flight across the barrier into the next MFMA phase.
            b_st = b_sync_loader.load_global(
                b, tid=tid, descriptor=desc_b, rsrc=b_rsrc
            )
            return adir_load(desc_a), b_st

        _fetch_part = {"a": True, "b": True, "a_range": None, "hi": None}

        def fetch(desc_a, desc_b) -> Tuple[list, list]:
            # pp_split_fetch: one operand's reads only (the other list empty).
            a_st = (a_sync_loader.load_global(
                b, tid=tid, descriptor=desc_a, rsrc=a_rsrc
            ) if _fetch_part["a_range"] is None else a_sync_loader.load_global(
                b, tid=tid, descriptor=desc_a, rsrc=a_rsrc, vec_range=_fetch_part["a_range"]
            )) if _fetch_part["a"] else []
            b_st = b_sync_loader.load_global(
                b, tid=tid, descriptor=desc_b, rsrc=b_rsrc
            ) if _fetch_part["b"] else []
            return a_st, b_st

        def store(staged, A_dst: Value, B_dst: Value, adir: bool = False) -> None:
            if adir:
                if not spec.lds_xor:
                    b_sync_loader.store_lds(b, smem_dst=B_dst, staged=staged[1])
                else:
                    for row, col, v in staged[1]:
                        b.smem_store_vN(B_dst, [row, lds_swz(b, row, col)], v, load_vec_b)
                return
            if not spec.lds_xor:
                a_sync_loader.store_lds(b, smem_dst=A_dst, staged=staged[0])
                b_sync_loader.store_lds(b, smem_dst=B_dst, staged=staged[1])
                return
            for dst, st, vec in ((A_dst, staged[0], load_vec_a), (B_dst, staged[1], load_vec_b)):
                for row, col, v in st:
                    b.smem_store_vN(dst, [row, lds_swz(b, row, col)], v, vec)

        def run_pingpong(lo, hi, it1, n_acc, fetch_tile, advance, tag):
            """The rpf_pingpong loop (see ImplicitGemmConvSpec.rpf_pingpong).
            Entered with tile ``lo`` in the LDS buffer, published.

            Every phase boundary is pinned with ``sched_barrier`` on both
            sides of its ``s_barrier``; only ``pp_pre_mfma`` MFMAs of each
            compute phase are placed ahead of the barrier that opens it (they
            read registers only, so the barrier does not order them)."""
            nw = spec.warp_m * spec.warp_n
            wid = b.readfirstlane(warp_id)
            hi_half = b.lshr(wid, b.const_i32((nw // 2).bit_length() - 1))
            lo_half = b.sub(c1, hi_half)
            h_mf = k_atoms * mfmas_m * mfmas_n // 2
            pre = spec.pp_pre_mfma
            late = spec.rpf_pingpong in (4, 5)
            # pp5: tile t+2's address math and global reads move from [mem B]
            # into [compute B], one buffer_load per vmem_interleave MFMAs.
            in_comp = spec.rpf_pingpong == 5

            def pin():
                b.sched_barrier(0)

            def prio(on: bool) -> None:
                if spec.rpf_setprio:
                    b.s_setprio(spec.rpf_setprio if on else 0)

            def mfmas(frags, accs_, i0, i1):
                return emit_mfma_phase(A_smem, B_smem, accs_, pp_use=(frags, i0, i1))

            def sched_loads(n_ld, n_m):
                # One buffer_load per vmem_interleave MFMAs of the phase.
                vi = spec.vmem_interleave
                for _ in range(n_ld):
                    take = min(vi, n_m)
                    if take:
                        b.sched_group_barrier(MFMA, take, 0)
                        n_m -= take
                    b.sched_group_barrier(VMEM_READ, 1, 0)
                if n_m:
                    b.sched_group_barrier(MFMA, n_m, 0)

            def open_compute(frags, accs_, i0, barrier):
                # The first ``pre`` MFMAs of the phase, then its barrier.
                prio(True)
                accs_ = mfmas(frags, accs_, i0, i0 + pre) if pre else accs_
                pin()
                barrier()
                pin()
                return accs_

            def close_compute(frags, accs_, i0, i1):
                accs_ = mfmas(frags, accs_, i0, i1)
                pin()
                prio(False)
                b.s_barrier_bare(fence="compiler")
                pin()
                return accs_

            def bar_mem_a():
                if spec.rpf_pingpong in (1, 4, 5):
                    # The wave's fragment reads complete before the partner
                    # half may start overwriting the buffer: a read still
                    # queued at the LDS when the barrier releases is not
                    # ordered before another SIMD's later write (pp3 measured
                    # nondeterministic results on t128x128 w2x4).
                    b.sync_lds_only(fence="compiler")
                elif spec.rpf_pingpong == 2:
                    b.s_barrier_bare(fence="compiler")
                else:
                    # No compiler fence: the compute phase waits for each
                    # fragment with its own (descending) lgkmcnt.
                    b.s_barrier_bare(fence="none")

            def bar_mem_b():
                b.sync_lds_only(fence="compiler")

            if spec.pp_end_guard:
                _fetch_part.update(hi=hi)
            # The high half runs one barrier ahead: its barrier j pairs with
            # the low half's barrier j + 1 from here to the loop's end.
            b.sync_half_block(hi_half)
            # Retire every scalar load before the loop: a scalar load still
            # pending at the loop header makes every LDS wait in the body a
            # full lgkmcnt(0) (SMEM returns out of order).
            b.s_waitcnt(lgkmcnt=0)
            it = list(it1)
            n_st = len(it1) - n_acc
            deep = late and spec.pp_fetch_deep
            # pp_split_fetch: operand index fetched in [mem A] (sf_m) and the
            # one carried from [compute B] (sf_c).
            psf = spec.pp_split_fetch if late and in_comp else 0
            sf_m, sf_c = {0: (None, None), 1: (1, 0), 2: (0, 1), 3: (1, 0), 4: (0, 1),
                          5: (1, 0), 6: (1, 0), 7: (1, 0), 8: (1, 0), 9: (1, 0),
                          10: (None, None)}[psf]
            # psf 3/4: the [mem A] reads are tile t+2's; the carried registers
            # hold both operands of tile t+1 as in plain pp5.
            psf_ahead = psf in (3, 4, 5, 6, 7, 8, 9, 10)
            # psf 6: the A reads of tile t+2 in two halves, [compute A] / [compute B].
            psf_asplit = psf in (6, 7)
            n_av = a_sync_loader.vecs_per_thread if psf_asplit else 0
            # psf 5: the carried operand's reads issue in [mem B].
            # psf 10: both operands' reads of tile t+2 issue in [mem B] (after
            # the LDS store), none in [mem A] or [compute B] -- for loops whose
            # [mem A] fragment reads are the long phase (t256x256x32).
            psf_memb = psf in (5, 10)

            def part(which):
                _fetch_part.update(a=which in (None, 0), b=which in (None, 1))

            if late:
                # pp4: the global reads of tile t+2 issue in [mem B] of tile
                # t, after its LDS store of tile t+1, so [mem A] holds only the
                # fragment reads. The staged registers are loop-carried; their
                # LDS coordinates are the prologue's (thread-only values).
                st1 = advance([v for _, v in it1[n_acc:]])
                part(None if psf_ahead else sf_c)
                staged0 = fetch_tile(b.add(lo, c_block_k), st1)
                part(None)
                n_a = len(staged0[0])
                regs0 = [v for _, _, v in staged0[0]] + [v for _, _, v in staged0[1]]
                n_rg = len(regs0)
                regs1 = []
                if deep:
                    # pp_fetch_deep: tile lo+2 is in flight too; the carried
                    # state is tile t+2's and the registers hold t+1 and t+2.
                    st1 = advance(st1)
                    staged1 = fetch_tile(b.add(lo, b.const_i32(2 * block_k)), st1)
                    regs1 = [v for _, _, v in staged1[0]] + [v for _, _, v in staged1[1]]
                it = (list(it1[:n_acc]) + [(f"{tag}_ps{i}", v) for i, v in enumerate(st1)]
                      + [(f"{tag}_pr{i}", v) for i, v in enumerate(regs0)]
                      + [(f"{tag}_pq{i}", v) for i, v in enumerate(regs1)])
            c_2bk = b.const_i32(2 * block_k)
            c_fbk = b.const_i32(3 * block_k) if deep else c_2bk
            # (an x-halo kernel's plain fallback loop keeps one B buffer)
            bdb = spec.pp_b_double and not spec.uk_xhalo
            half_kk = k_atoms // 2

            def body(kv, iv, B_cur, B_nxt):
                """One K tile; returns the values the loop carries on."""
                accs_t = list(iv[:n_acc])
                st = list(iv[n_acc:n_acc + n_st])
                # [mem A] every fragment of tile t (+ tile t+1's global reads);
                # pp_b_double: the B columns of the second half later.
                frags = []
                k_off_capture[0] = kv
                if psf == 9:
                    # pp_split_fetch 9: tile t+2's B reads first.
                    st_ahead = advance(st)
                    part(sf_m)
                    staged_m = fetch_tile(b.add(kv, c_2bk), st_ahead)
                    part(None)
                    pin()
                if bdb:
                    _pp_part.update(b_from=half_kk)
                emit_mfma_phase(A_smem, B_cur, accs_t, pp_read=frags)
                _pp_part.update(b_from=None)
                pin()
                if psf == 9:
                    pass
                elif psf == 10:
                    st_ahead = advance(st)
                elif psf_ahead:
                    # pp_split_fetch 3/4: tile t+2's [mem A] operand.
                    st_ahead = advance(st)
                    part(sf_m)
                    staged_m = fetch_tile(b.add(kv, c_2bk), st_ahead)
                    part(None)
                    if psf == 7:
                        # ... and the first half of its A reads.
                        _fetch_part.update(a_range=(0, n_av // 2))
                        part(0)
                        nxt_a1 = fetch_tile(b.add(kv, c_2bk), st_ahead)
                        part(None)
                        _fetch_part.update(a_range=None)
                    pin()
                elif psf:
                    # pp_split_fetch: tile t+1's other operand (st is its state).
                    part(sf_m)
                    staged_m = fetch_tile(b.add(kv, c_block_k), st)
                    part(None)
                    pin()
                if late:
                    st_n = st
                else:
                    st_n = advance(st)
                    staged = fetch_tile(b.add(kv, c_block_k), st_n)
                    pin()
                acc_n = open_compute(frags, accs_t, 0, bar_mem_a)
                # [compute A]
                if psf in (6, 8):
                    acc_n = mfmas(frags, acc_n, pre, h_mf)
                    _fetch_part.update(a_range=(0, n_av // 2) if psf == 6 else None)
                    part(0)
                    nxt_a1 = fetch_tile(b.add(kv, c_2bk), st_ahead)
                    part(None)
                    _fetch_part.update(a_range=None)
                    sched_loads(len(nxt_a1[0]), h_mf - pre)
                    pin()
                    prio(False)
                    b.s_barrier_bare(fence="compiler")
                    pin()
                else:
                    acc_n = close_compute(frags, acc_n, pre, h_mf)
                # [mem B] tile t+1 into the buffer (every wave's reads of tile
                # t are retired: the partner half closed its [mem A] before
                # the barrier that opened this phase).
                if late:
                    regs = list(iv[n_acc + n_st:n_acc + n_st + n_rg])
                    staged = (
                        [(r, c, v) for (r, c, _), v in zip(staged0[0], regs[:n_a])],
                        [(r, c, v) for (r, c, _), v in zip(staged0[1], regs[n_a:])],
                    )
                    if psf and not psf_ahead:
                        staged = tuple(staged_m[i] if i == sf_m else staged[i] for i in (0, 1))
                store(staged, A_smem, B_nxt)
                pin()
                if psf_memb:
                    part(None if psf == 10 else sf_c)
                    nxt_c = fetch_tile(b.add(kv, c_2bk), st_ahead)
                    part(None)
                    pin()
                if bdb:
                    # The second half's B columns of tile t, from the buffer
                    # nobody writes before tile t+2's [mem B].
                    bf = []
                    _pp_part.update(b_only=True, kk_from=half_kk)
                    emit_mfma_phase(A_smem, B_cur, acc_n, pp_read=bf)
                    _pp_part.update(b_only=False, kk_from=0)
                    for i, (_, bc) in enumerate(bf):
                        frags[half_kk + i] = (frags[half_kk + i][0], bc)
                    pin()
                if late and not in_comp:
                    st_n = advance(st)
                    nxt = fetch_tile(b.add(kv, c_fbk), st_n)
                    pin()
                acc_n = open_compute(frags, acc_n, h_mf, bar_mem_b)
                # [compute B]
                if in_comp:
                    acc_n = mfmas(frags, acc_n, h_mf + pre, 2 * h_mf)
                    st_n = st_ahead if psf_ahead else advance(st)
                    if psf_memb:
                        nxt, n_ld = nxt_c, 0
                    elif psf == 8:
                        nxt, n_ld = (nxt_a1[0], []), 0
                    else:
                        part(sf_c)
                        if psf_asplit:
                            _fetch_part.update(a_range=(n_av // 2, n_av))
                        nxt = fetch_tile(b.add(kv, c_fbk), st_n)
                        _fetch_part.update(a_range=None)
                        part(None)
                        n_ld = len(nxt[0]) + len(nxt[1])
                        if psf_asplit:
                            nxt = (nxt_a1[0] + nxt[0], nxt[1])
                    if psf_ahead:
                        nxt = tuple(staged_m[i] if i == sf_m else nxt[i] for i in (0, 1))
                    sched_loads(n_ld, h_mf - pre)
                    pin()
                    prio(False)
                    b.s_barrier_bare(fence="compiler")
                    pin()
                else:
                    acc_n = close_compute(frags, acc_n, h_mf + pre, 2 * h_mf)
                if late:
                    return [*acc_n, *st_n, *list(iv[n_acc + n_st + n_rg:]),
                            *[v for _, _, v in nxt[0]], *[v for _, _, v in nxt[1]]]
                return [*acc_n, *st_n]

            if not bdb:
                for_op = b.scf_for_iter(lo, hi, c_block_k, it, iv_name=f"{tag}_k")
                with for_op as (kv, iv):
                    b.scf_yield(*body(kv, list(iv), B_smem, B_smem))
                b.sync_half_block(lo_half)
                return list(for_op.results)[:n_acc]
            # pp_b_double: B alternates between two buffers, so the loop runs
            # whole pairs of tiles and an odd last tile in a 0/1-trip tail.
            n_tiles = b.div(b.add(b.sub(hi, lo), b.const_i32(block_k - 1)), c_block_k)
            pair_hi = b.add(lo, b.mul(b.div(n_tiles, b.const_i32(2)), c_2bk))
            tail_hi = b.add(lo, b.mul(n_tiles, c_block_k))
            for_op = b.scf_for_iter(lo, pair_hi, c_2bk, it, iv_name=f"{tag}_k")
            with for_op as (kv, iv):
                iv1 = body(kv, list(iv), B_smem, B_smem2)
                b.scf_yield(*body(b.add(kv, c_block_k), iv1, B_smem2, B_smem))
            it_t = [(f"{n}_t", v) for (n, _), v in zip(it, for_op.results)]
            for_t = b.scf_for_iter(pair_hi, tail_hi, c_block_k, it_t, iv_name=f"{tag}_kt")
            with for_t as (kt, tv):
                b.scf_yield(*body(kt, list(tv), B_smem, B_smem2))
            b.sync_half_block(lo_half)
            return list(for_t.results)[:n_acc]

        def run_loop(lo, hi, accs_in, state_in, fetch_tile, advance, tag, adir=False):
            """One K loop over [lo, hi). ``fetch_tile(k, state)`` issues the
            global reads of the tile at ``k`` and returns them staged;
            ``advance(state)`` steps the carried tile state by one tile."""
            n_acc = len(accs_in)
            names = [f"{tag}_{n}" for n, _ in state_in]
            if adir:
                # a_direct (reg_prefetch=1 shape): tile t's A operand is
                # loop-carried in registers; B goes through one LDS buffer.
                # Unrolled x2 so the two A register sets alternate in place
                # (a single-tile body would copy the prefetched set into the
                # carried one, and the copy waits for the loads).
                n_st = len(state_in)
                st0 = [v for _, v in state_in]
                a0, b0 = fetch_tile(lo, st0)
                store((None, b0), A_smem, B_smem, adir=True)
                b.sync()

                def half(kv_t, st_t, a_cur, accs_t):
                    st_n = advance(st_t)
                    a_nx, b_nx = fetch_tile(b.add(kv_t, c_block_k), st_n)
                    b.sched_barrier(0)
                    k_off_capture[0] = kv_t
                    acc_o = emit_mfma_phase(A_smem, B_smem, accs_t, a_regs=a_cur)
                    b.sched_barrier(0)
                    # LDS-only barriers: the A loads stay in flight across them.
                    b.sync_lds_only()
                    store((None, b_nx), A_smem, B_smem, adir=True)
                    b.sync_lds_only()
                    return st_n, a_nx, acc_o

                c_2bk_a = b.const_i32(2 * block_k)
                n_tiles_a = b.div(b.add(b.sub(hi, lo), b.const_i32(block_k - 1)), c_block_k)
                pair_hi_a = b.add(lo, b.mul(b.div(n_tiles_a, b.const_i32(2)), c_2bk_a))
                tail_hi_a = b.add(lo, b.mul(n_tiles_a, c_block_k))
                it1 = (
                    [(f"{tag}_{n}", v) for (n, _), v in zip(accs, accs_in)]
                    + [(nm, v) for nm, (_, v) in zip(names, state_in)]
                    + [(f"{tag}_ar{i}", v) for i, v in enumerate(a0)]
                )
                for_op = b.scf_for_iter(lo, pair_hi_a, c_2bk_a, it1, iv_name=f"{tag}_k")
                with for_op as (kv, iv):
                    st = list(iv[n_acc:n_acc + n_st])
                    a_x = list(iv[n_acc + n_st:])
                    st1, a_y, acc1 = half(kv, st, a_x, list(iv[:n_acc]))
                    st2, a_x2, acc2 = half(b.add(kv, c_block_k), st1, a_y, acc1)
                    b.scf_yield(*acc2, *st2, *a_x2)
                res = list(for_op.results)
                a_last = res[n_acc + n_st:]
                t_it = [(f"{tag}_t_{n}", v) for (n, _), v in zip(accs, res[:n_acc])]
                for_t = b.scf_for_iter(pair_hi_a, tail_hi_a, c_block_k, t_it, iv_name=f"{tag}_kt")
                with for_t as (kt, tv):
                    k_off_capture[0] = kt
                    acc_t = emit_mfma_phase(A_smem, B_smem, list(tv), a_regs=a_last)
                    b.sync()
                    b.scf_yield(*acc_t)
                return list(for_t.results)
            if not spec.reg_prefetch:
                it = [(f"{tag}_{n}", v) for (n, _), v in zip(accs, accs_in)] + [
                    (nm, v) for nm, (_, v) in zip(names, state_in)
                ]
                for_op = b.scf_for_iter(lo, hi, c_block_k, it, iv_name=f"{tag}_k")
                with for_op as (kv, iv):
                    st = list(iv[n_acc:])
                    staged = fetch_tile(kv, st)
                    store(staged, A_smem, B_smem)
                    b.sync()
                    new_accs = emit_mfma_phase(A_smem, B_smem, list(iv[:n_acc]))
                    b.sync()
                    b.scf_yield(*new_accs, *advance(st))
                return list(for_op.results)[:n_acc]
            # Prologue: tile `lo` into buffer 0, published. Guarded only by
            # the loads' own validity: an empty range loads one zero tile that
            # no MFMA reads.
            st0 = [v for _, v in state_in]
            if spec.uk_xhalo or (spec.fb_skip_empty and tag == "fb"):
                # The x-halo kernel carries the plain loops as a fallback for
                # the workgroups the x-halo loop does not serve; their
                # prologue only runs when the range is not empty
                # (fb_skip_empty: the uniform_k_decode kernels' fallback too).
                with b.scf_if(b.cmp_lt(lo, hi)):
                    store(fetch_tile(lo, st0), A_smem, B_smem)
                    b.sync()
            else:
                store(fetch_tile(lo, st0), A_smem, B_smem)
                b.sync()
            it1 = [(f"{tag}_{n}", v) for (n, _), v in zip(accs, accs_in)] + [
                (nm, v) for nm, (_, v) in zip(names, state_in)
            ]
            if spec.reg_prefetch == 1 and spec.rpf_pingpong and tag == "uk":
                return run_pingpong(lo, hi, it1, n_acc, fetch_tile, advance, tag)
            if spec.reg_prefetch == 1:
                # One buffer: read tile t+1 into registers, MFMA tile t, retire
                # every wave's ds_reads, then overwrite the buffer and publish.
                # The last iteration reads a tile past the end (zeros, no
                # memory traffic) that no MFMA consumes.
                for_op = b.scf_for_iter(lo, hi, c_block_k, it1, iv_name=f"{tag}_k")
                with for_op as (kv, iv):
                    st = list(iv[n_acc:])
                    st1 = advance(st)
                    _late = {}
                    if spec.rpf_fetch_late:
                        def _fetch_cb(kv=kv, st1=st1):
                            _late["staged"] = fetch_tile(b.add(kv, c_block_k), st1)
                        staged = None
                    else:
                        _fetch_cb = None
                        staged = fetch_tile(b.add(kv, c_block_k), st1)
                    if spec.vmem_interleave:
                        # The MFMA phase places the loads (vmem_interleave).
                        _vmi_pending[0] = len(staged[0]) + len(staged[1])
                    else:
                        b.sched_barrier(0)
                    k_off_capture[0] = kv
                    if spec.defer_mfma:
                        _mfma_defer.update(on=True, n=spec.defer_mfma, fns=[])
                        # (An x-halo kernel's fallback loop keeps its priority:
                        # rpf_setprio there is the x-halo loop's.)
                        if not spec.uk_xhalo:
                            _prio_on()
                        acc_n = emit_mfma_phase(
                            A_smem, B_smem, list(iv[:n_acc]), after_first_reads=_fetch_cb
                        )
                        staged = _late.get("staged", staged)
                        _mfma_defer["on"] = False
                        b.sched_barrier(0)
                        if not spec.uk_xhalo:
                            _prio_off()
                        b.sync_lds_only()
                        store(staged, A_smem, B_smem)
                        for fn_ in _mfma_defer["fns"]:
                            acc_n = fn_(acc_n)
                        if spec.defer_mfma_per_write:
                            # One group per staged write: 1 ds_write : n MFMAs.
                            for _ in range(len(staged[0]) + len(staged[1])):
                                b.sched_group_barrier(DS_WRITE, 1, 0)
                                b.sched_group_barrier(
                                    MFMA, spec.defer_mfma_per_write, 0
                                )
                        else:
                            # 1 ds_write : 2 MFMAs (measured best of 1:1, 1:2,
                            # 2:4 and 1:6 on MI300A at defer_mfma=1).
                            for _ in range(8):
                                b.sched_group_barrier(DS_WRITE, 1, 0)
                                b.sched_group_barrier(MFMA, 2, 0)
                        b.sched_barrier(0)
                        b.sync_lds_only()
                    else:
                        if spec.rpf_setprio:
                            b.s_setprio(spec.rpf_setprio)
                        acc_n = emit_mfma_phase(
                            A_smem, B_smem, list(iv[:n_acc]), after_first_reads=_fetch_cb
                        )
                        staged = _late.get("staged", staged)
                        b.sched_barrier(0)
                        if spec.rpf_setprio:
                            b.s_setprio(0)
                        _bar = b.sync_lds_only if spec.rpf_lds_barrier else b.sync
                        _bar()
                        store(staged, A_smem, B_smem)
                        _bar()
                    b.scf_yield(*acc_n, *st1)
                return list(for_op.results)[:n_acc]
            c_2bk = b.const_i32(2 * block_k)
            it = [(f"{tag}_{n}", v) for (n, _), v in zip(accs, accs_in)] + [
                (nm, v) for nm, (_, v) in zip(names, state_in)
            ]
            # Whole pairs of tiles in the unrolled loop; an odd last tile was
            # prefetched into buffer 0 by the last pair and runs in a 0/1-trip
            # tail loop, so no MFMA phase is spent on a tile past the end.
            n_tiles = b.div(b.add(b.sub(hi, lo), b.const_i32(block_k - 1)), c_block_k)
            pair_hi = b.add(lo, b.mul(b.div(n_tiles, b.const_i32(2)), c_2bk))
            tail_hi = b.add(lo, b.mul(n_tiles, c_block_k))
            def _pin_or_interleave(staged_):
                # Pin the reads ahead of the MFMAs and the LDS writes behind
                # them: left alone, the scheduler sinks the buffer loads next
                # to their ds_write and the wave waits on them again. With
                # vmem_interleave the MFMA phase places them instead.
                if spec.vmem_interleave:
                    _vmi_pending[0] = len(staged_[0]) + len(staged_[1])
                else:
                    b.sched_barrier(0)

            for_op = b.scf_for_iter(lo, pair_hi, c_2bk, it, iv_name=f"{tag}_k")
            with for_op as (kv, iv):
                st = list(iv[n_acc:])
                k1 = b.add(kv, c_block_k)
                st1 = advance(st)
                _prio_on()
                staged = fetch_tile(k1, st1)
                _pin_or_interleave(staged)
                k_off_capture[0] = kv
                acc_a = emit_mfma_phase(A_smem, B_smem, list(iv[:n_acc]))
                b.sched_barrier(0)
                _prio_off()
                store(staged, A_smem2, B_smem2)
                b.sync()
                k2 = b.add(kv, c_2bk)
                st2 = advance(st1)
                _prio_on()
                staged = fetch_tile(k2, st2)
                _pin_or_interleave(staged)
                k_off_capture[0] = k1
                acc_b = emit_mfma_phase(A_smem2, B_smem2, acc_a)
                b.sched_barrier(0)
                _prio_off()
                store(staged, A_smem, B_smem)
                b.sync()
                b.scf_yield(*acc_b, *st2)
            pair_accs = list(for_op.results)[:n_acc]
            t_it = [(f"{tag}_t_{n}", v) for (n, _), v in zip(accs, pair_accs)]
            for_t = b.scf_for_iter(pair_hi, tail_hi, c_block_k, t_it, iv_name=f"{tag}_kt")
            with for_t as (kt, tv):
                k_off_capture[0] = kt
                acc_t = emit_mfma_phase(A_smem, B_smem, list(tv))
                b.sync()
                b.scf_yield(*acc_t)
            return list(for_t.results)

        def plain_fetch(k_val, _st):
            k_off_capture[0] = k_val
            return fetch(a_descriptor, b_descriptor)

        def plain_adv(st):
            return list(st)

        if not spec.uniform_k_decode:
            return run_loop(k_lo, k_hi, [v for _, v in accs], state0,
                            plain_fetch, plain_adv, "k")

        def _emit_xhalo_pingpong(accs_in, st0, xh_ok, xh_hi, xh_fetch, xh_store, xh_adv,
                                 base_rows, XT):
            """The x-halo loop in the rpf_pingpong form (modes 4 and 5): per
            staged tile, [mem A] reads every fragment of the X taps, the X
            taps' MFMAs are split in two halves around [mem B] (the LDS store
            of the next tile), and the global reads of the tile after it
            issue in [mem B] (pp4) or inside [compute B] (pp5). The two halves
            of the waves run half an iteration apart as in run_pingpong."""
            nw = spec.warp_m * spec.warp_n
            wid = b.readfirstlane(warp_id)
            hi_half = b.lshr(wid, b.const_i32((nw // 2).bit_length() - 1))
            lo_half = b.sub(c1, hi_half)
            per_tap = k_atoms * mfmas_m * mfmas_n
            n_all = XT * per_tap
            h_mf = n_all // 2
            pre = spec.pp_pre_mfma
            in_comp = spec.rpf_pingpong == 5
            n_acc = len(accs_in)

            def pin():
                b.sched_barrier(0)

            def prio(on: bool) -> None:
                if spec.rpf_setprio:
                    b.s_setprio(spec.rpf_setprio if on else 0)

            def mfmas(frags, accs_, i0, i1):
                i = i0
                while i < i1:
                    x, j = divmod(i, per_tap)
                    j1 = min(per_tap, j + (i1 - i))
                    accs_ = emit_mfma_phase(A_smem, B_smem, accs_, pp_use=(frags[x], j, j1))
                    i += j1 - j
                return accs_

            with b.scf_if(xh_ok):
                xh_store(xh_fetch(st0))
                b.sync()
            # Tile 1 into registers (a workgroup that does not run the loop
            # loads it and never stores it).
            st1 = xh_adv(st0)
            staged0 = xh_fetch(st1)
            n_a = len(staged0[0])
            regs0 = [v for _, _, v in staged0[0]] + [v for _, _, v in staged0[1]]
            b.sync_half_block(hi_half)
            b.s_waitcnt(lgkmcnt=0)
            it = ([(f"xp_{n}", v) for (n, _), v in zip(accs, accs_in)]
                  + [("xp_c", st1[0]), ("xp_y", st1[1]), ("xp_z", st1[2])]
                  + [(f"xp_r{i}", v) for i, v in enumerate(regs0)])
            bdb = spec.pp_b_double
            nh = mfmas_n // 2

            def mf_part(frags, accs_, part):
                # pp_b_double: compute A = the B columns [0, n/2) of every
                # tap, compute B = [n/2, n) (their B fragments are read in
                # [mem B]).
                out = list(accs_)
                for x in range(XT):
                    for kk in range(k_atoms):
                        a_rows, b_cols = frags[x][kk]
                        for mi in range(mfmas_m):
                            for ni in range(part * nh, (part + 1) * nh):
                                f = mi * mfmas_n + ni
                                out[f] = _emit_mfma(b, atom, a_rows[mi], b_cols[ni], out[f])
                return out

            def body(kv, iv, B_cur, B_nxt):
                accs_t = list(iv[:n_acc])
                st_t1 = list(iv[n_acc:n_acc + 3])
                regs = list(iv[n_acc + 3:])
                # [mem A] the fragments of every x tap of the staged tile.
                frags = []
                prev = None
                if bdb:
                    _pp_part.update(b_ni=(0, nh))
                for x in range(XT):
                    rows = base_rows if x == 0 else [b.add(r, b.const_i32(x)) for r in base_rows]
                    f = []
                    rec = {} if spec.xh_dpp else None
                    emit_mfma_phase(A_smem, B_cur, accs_t, pp_read=f, a_lds_rows=rows,
                                    b_row_off=x * block_n,
                                    a_shift_prev=prev if spec.xh_dpp else None, a_rec=rec)
                    prev = rec
                    frags.append(f)
                _pp_part.update(b_ni=None)
                prio(True)
                if bdb:
                    acc_n = accs_t
                else:
                    acc_n = mfmas(frags, accs_t, 0, pre) if pre else accs_t
                pin()
                b.sync_lds_only(fence="compiler")  # see run_pingpong's bar_mem_a
                pin()
                # [compute A]
                acc_n = mf_part(frags, acc_n, 0) if bdb else mfmas(frags, acc_n, pre, h_mf)
                pin()
                prio(False)
                b.s_barrier_bare(fence="compiler")
                pin()
                # [mem B] the next staged tile (registers) into the buffers.
                xh_store((
                    [(r, c_, v) for (r, c_, _), v in zip(staged0[0], regs[:n_a])],
                    [(r, c_, v) for (r, c_, _), v in zip(staged0[1], regs[n_a:])],
                ), B_nxt)
                pin()
                if bdb:
                    # The other B columns of every tap of tile t (B_cur is not
                    # written before tile t+2's [mem B]).
                    _pp_part.update(b_only=True, b_ni=(nh, mfmas_n))
                    for x in range(XT):
                        bf = []
                        emit_mfma_phase(A_smem, B_cur, accs_t, pp_read=bf,
                                        a_lds_rows=base_rows, b_row_off=x * block_n)
                        for kk in range(k_atoms):
                            lo_b = frags[x][kk][1]
                            frags[x][kk] = (frags[x][kk][0], list(lo_b) + list(bf[kk][1]))
                    _pp_part.update(b_only=False, b_ni=None)
                    pin()
                if not in_comp:
                    st_t2 = xh_adv(st_t1)
                    nxt = xh_fetch(st_t2)
                    pin()
                prio(True)
                if pre and not bdb:
                    acc_n = mfmas(frags, acc_n, h_mf, h_mf + pre)
                pin()
                b.sync_lds_only(fence="compiler")
                pin()
                # [compute B]
                if bdb:
                    acc_n = mf_part(frags, acc_n, 1)
                else:
                    acc_n = mfmas(frags, acc_n, h_mf + pre, n_all)
                if in_comp:
                    st_t2 = xh_adv(st_t1)
                    nxt = xh_fetch(st_t2)
                    n_ld = len(nxt[0]) + len(nxt[1])
                    n_m = n_all - h_mf - (0 if bdb else pre)
                    for _ in range(n_ld):
                        take = min(spec.vmem_interleave, n_m)
                        if take:
                            b.sched_group_barrier(MFMA, take, 0)
                            n_m -= take
                        b.sched_group_barrier(VMEM_READ, 1, 0)
                    if n_m:
                        b.sched_group_barrier(MFMA, n_m, 0)
                pin()
                prio(False)
                b.s_barrier_bare(fence="compiler")
                pin()
                return [*acc_n, *st_t2, *[v for _, _, v in nxt[0]], *[v for _, _, v in nxt[1]]]

            if not bdb:
                for_op = b.scf_for_iter(c0, xh_hi, c_block_k, it, iv_name="xp_k")
                with for_op as (kv, iv):
                    b.scf_yield(*body(kv, list(iv), B_smem, B_smem))
                b.sync_half_block(lo_half)
                return list(for_op.results)[:n_acc]
            c_2bk = b.const_i32(2 * block_k)
            n_tiles = b.div(b.add(xh_hi, b.const_i32(block_k - 1)), c_block_k)
            pair_hi = b.mul(b.div(n_tiles, b.const_i32(2)), c_2bk)
            tail_hi = b.mul(n_tiles, c_block_k)
            for_op = b.scf_for_iter(c0, pair_hi, c_2bk, it, iv_name="xp_k")
            with for_op as (kv, iv):
                iv1 = body(kv, list(iv), B_smem, B_smem2)
                b.scf_yield(*body(b.add(kv, c_block_k), iv1, B_smem2, B_smem))
            it_t = [(f"{n}_t", v) for (n, _), v in zip(it, for_op.results)]
            for_t = b.scf_for_iter(pair_hi, tail_hi, c_block_k, it_t, iv_name="xp_kt")
            with for_t as (kt, tv):
                b.scf_yield(*body(kt, list(tv), B_smem, B_smem2))
            b.sync_half_block(lo_half)
            return list(for_t.results)[:n_acc]

        def _emit_xhalo_sub_pp(accs_in, xh_ok, lo, hi, st0, xh_uniforms, xh_adv,
                               a_slots, a_base0, base_rows, chunks, c_oob_a, c_eb):
            """The x-halo loop as single-tap ping-pong sub-tiles (xh_subtile).

            One iteration = one staged (chunk, z, y) step = three run_pingpong
            style tiles x = 0, 1, 2 (A rows shifted by x, the tap's own B tile).
            Entered with A(j) and B(j, 0) published and B(j, 1) in registers.
            [mem B] of tile x stores what tile x + 1 reads: B(j, 1), B(j, 2),
            then A(j + 1) + B(j + 1, 0); [compute B] of tile x issues the
            global reads of tile x + 2: B(j, 2), A(j + 1) + B(j + 1, 0),
            B(j + 1, 1)."""
            nw = spec.warp_m * spec.warp_n
            wid = b.readfirstlane(warp_id)
            hi_half = b.lshr(wid, b.const_i32((nw // 2).bit_length() - 1))
            lo_half = b.sub(c1, hi_half)
            per_tap = k_atoms * mfmas_m * mfmas_n
            h_mf = per_tap // 2
            pre = spec.pp_pre_mfma
            n_acc = len(accs_in)
            b1_slots = chunks(block_n, load_vec_b, block_n)
            s_wi_ = params["p_A_stride_wi"]

            def pin():
                b.sched_barrier(0)

            def prio(on: bool) -> None:
                if spec.rpf_setprio:
                    b.s_setprio(spec.rpf_setprio if on else 0)

            def fetch_a(st):
                a_off, _bo, t_ok = xh_uniforms(st)
                a_base = b.add(a_base0, a_off)
                out = []
                for row, col, live in a_slots:
                    off = b.add(b.add(a_base, b.mul(row, s_wi_)), col)
                    ok = t_ok if live is None else b.land(t_ok, live)
                    safe = b.select(ok, b.mul(off, c_eb), c_oob_a)
                    out.append(b.buffer_load_vN(a_rsrc, safe, c0, ir_dtype_a, load_vec_a))
                return out

            def fetch_b(st, x):
                _ao, b_off, t_ok = xh_uniforms(st)
                out = []
                for row, col, live in b1_slots:
                    k_out = b.add(block_n_off_v, row)
                    if grouped:
                        k_out = b.add(k_out_group_base, k_out)
                    off = b.add(
                        b.add(b.mul(k_out, p_B_stride_k), b.const_i32(0) if x == 0
                              else b.mul(b.const_i32(x), p_B_stride_x)),
                        b.add(b_off, col),
                    )
                    ok = t_ok if live is None else b.land(t_ok, live)
                    safe = b.select(ok, b.mul(off, c_eb), c_oob_a)
                    out.append(b.buffer_load_vN(b_rsrc, safe, c0, ir_dtype_b, load_vec_b))
                return out

            def store_a(regs):
                for (row, col, _l), v in zip(a_slots, regs):
                    b.smem_store_vN(A_smem, [row, lds_swz(b, row, col)], v, load_vec_a)

            def store_b(regs):
                for (row, col, _l), v in zip(b1_slots, regs):
                    b.smem_store_vN(B_smem, [row, lds_swz(b, row, col)], v, load_vec_b)

            def mfmas(frags, accs_, i0, i1):
                return emit_mfma_phase(A_smem, B_smem, accs_, pp_use=(frags, i0, i1))

            def sub(x, accs_t, store_fn, load_fn):
                # [mem A] every fragment of tap x.
                frags = []
                rows = base_rows if x == 0 else [b.add(r, b.const_i32(x)) for r in base_rows]
                emit_mfma_phase(A_smem, B_smem, accs_t, pp_read=frags, a_lds_rows=rows,
                                b_row_off=0)
                prio(True)
                acc = mfmas(frags, accs_t, 0, pre) if pre else accs_t
                pin()
                b.sync_lds_only(fence="compiler")  # see run_pingpong's bar_mem_a
                pin()
                # [compute A]
                acc = mfmas(frags, acc, pre, h_mf)
                pin()
                prio(False)
                b.s_barrier_bare(fence="compiler")
                pin()
                # [mem B] what tap x + 1 reads.
                store_fn()
                pin()
                prio(True)
                if pre:
                    acc = mfmas(frags, acc, h_mf, h_mf + pre)
                pin()
                b.sync_lds_only(fence="compiler")
                pin()
                # [compute B] + the global reads of tap x + 2.
                acc = mfmas(frags, acc, h_mf + pre, per_tap)
                nxt = load_fn()
                n_m = per_tap - h_mf - pre
                for _ in range(len(nxt)):
                    take = min(spec.vmem_interleave, n_m)
                    if take:
                        b.sched_group_barrier(MFMA, take, 0)
                        n_m -= take
                    b.sched_group_barrier(VMEM_READ, 1, 0)
                if n_m:
                    b.sched_group_barrier(MFMA, n_m, 0)
                pin()
                prio(False)
                b.s_barrier_bare(fence="compiler")
                pin()
                return acc, nxt

            with b.scf_if(xh_ok):
                store_a(fetch_a(st0))
                store_b(fetch_b(st0, 0))
                b.sync()
            rb1_0 = fetch_b(st0, 1)
            b.sync_half_block(hi_half)
            b.s_waitcnt(lgkmcnt=0)
            it = ([(f"xs_{n}", v) for (n, _), v in zip(accs, accs_in)]
                  + [("xs_c", st0[0]), ("xs_y", st0[1]), ("xs_z", st0[2])]
                  + [(f"xs_r{i}", v) for i, v in enumerate(rb1_0)])
            n_a = len(a_slots)
            for_op = b.scf_for_iter(lo, hi, c_block_k, it, iv_name="xs_k")
            with for_op as (kv, iv):
                iv = list(iv)
                acc = iv[:n_acc]
                st = iv[n_acc:n_acc + 3]
                rb1 = iv[n_acc + 3:]
                st_n = xh_adv(st)
                acc, rb2 = sub(0, acc, lambda: store_b(rb1), lambda: fetch_b(st, 2))
                acc, nx = sub(1, acc, lambda: store_b(rb2),
                              lambda: fetch_a(st_n) + fetch_b(st_n, 0))
                acc, rb1n = sub(2, acc, lambda: (store_a(nx[:n_a]), store_b(nx[n_a:])),
                                lambda: fetch_b(st_n, 1))
                b.scf_yield(*acc, *st_n, *rb1n)
            b.sync_half_block(lo_half)
            return list(for_op.results)[:n_acc]

        def _emit_xhalo_loop(uk_ok_, accs_in, xs_range=None):
            """The uk_xhalo K loop (see ImplicitGemmConvSpec.uk_xhalo).

            Returns ``(xh_ok, accs)``: whether this workgroup ran it, and the
            accumulators after it (unchanged when it did not)."""
            XT = _XHALO_TAPS
            R = block_m + spec.uk_xhalo  # staged input pixels (rows)
            c_R = b.const_i32(R)
            c_XT = b.const_i32(XT)
            m0 = block_m_off_v

            def lin_of(b_, m_val):
                # Input pixel index of output pixel m at tap 0 (pad 0, sW 1).
                q_wo, wo = magic_divmod(b_, m_val, params["p_magic_m_Wo_mult"],
                                        params["p_magic_m_Wo_shift"], p_Wo)
                q_ho, ho = magic_divmod(b_, q_wo, params["p_magic_m_Ho_mult"],
                                        params["p_magic_m_Ho_shift"], p_Ho)
                n_, do_ = magic_divmod(b_, q_ho, params["p_magic_m_Do_mult"],
                                       params["p_magic_m_Do_shift"], p_Do)
                d_ = b_.add(b_.mul(n_, p_Di), b_.mul(do_, p_sD))
                h_ = b_.add(b_.mul(d_, p_Hi), b_.mul(ho, p_sH))
                return b_.add(b_.mul(h_, p_Wi), wo)

            c_bm = b.const_i32(block_m)
            m_end = b.add(m0, c_bm)
            m_last = b.sub(b.select(b.cmp_lt(m_end, p_M), m_end, p_M), c1)
            lin0 = lin_of(b, m0)
            span = b.add(b.sub(lin_of(b, m_last), lin0), c_XT)
            conds = [
                uk_ok_,
                b.cmp_le(span, c_R),
                b.cmp_eq(p_pD, c0), b.cmp_eq(p_pH, c0), b.cmp_eq(p_pW, c0),
                b.cmp_eq(p_sW, c1), b.cmp_eq(p_dW, c1), b.cmp_eq(p_X, c_XT),
                # packed input: pixel lin's channel 0 is at lin * s_wi
                b.cmp_eq(s_hi, b.mul(p_Wi, s_wi)),
                b.cmp_eq(s_di, b.mul(p_Hi, s_hi)),
                b.cmp_eq(s_n, b.mul(p_Di, s_di)),
            ]
            if spec.xh_dpp:
                # Output row boundaries only at a wave's M extent.
                conds.append(b.cmp_eq(
                    b.mod(p_Wo, b.const_i32(mfmas_m * spec.warp_tile_m)), c0))
            if spec.xh_stride_m:
                # pixels 4r + mi, mi < mfmas_m - 1, never end an output row.
                conds.append(b.cmp_eq(b.mod(p_Wo, b.const_i32(mfmas_m)), c0))
            xh_ok = conds[0]
            for c_ in conds[1:]:
                xh_ok = b.land(xh_ok, c_)
            _xsm_ok[:] = [xh_ok]
            # (chunk, z, y) tiles: Z*Y*cpg / tile_k of them.
            xh_hi = b.select(xh_ok, b.mul(b.mul(p_Z, p_Y), p_cpg), c0)
            a_base0 = b.mul(lin0, s_wi)
            if grouped:
                a_base0 = b.add(a_base0, a_gbase)

            def xh_uniforms(st):
                c_t, y_t, z_t = st
                a_off = b.add(b.add(b.mul(y_t, a_ystep), b.mul(z_t, a_zstep)), c_t)
                b_off = b.add(b.add(b.mul(y_t, p_B_stride_y), b.mul(z_t, p_B_stride_z)), c_t)
                return a_off, b_off, b.cmp_lt(z_t, p_Z)

            c_oob_a = b.const_i32((1 << 31) - 1)
            c_eb = b.const_i32(2 if spec.data.dtype_a in ("fp16", "bf16") else 4)

            def chunks(rows, vec, cap_row):
                # Per-thread (row, col) of the rows x tile_k tile in vec-wide
                # chunks; surplus chunks of the last round go to cap_row.
                cpv = block_k // vec
                total = rows * cpv
                n_e = -(-total // threads)
                out = []
                for e in range(n_e):
                    v = b.add(b.const_i32(e * threads), tid)
                    surplus = (e + 1) * threads > total
                    row = b.div(v, b.const_i32(cpv))
                    if spec.xh_stride_m and cap_row == R and cpv == 4:
                        # xh_stride_m: lanes 4t..4t+3 store row t ^ ((t/4) & 1)
                        # of their 16-row block, so each 8-lane write phase
                        # pairs two rows on different bank-line halves.
                        row = b.xor(row, b.land(b.lshr(row, b.const_i32(2)), c1))
                    col = b.mul(b.mod(v, b.const_i32(cpv)), b.const_i32(vec))
                    live = None
                    if surplus:
                        live = b.cmp_lt(v, b.const_i32(total))
                        row = b.select(live, row, b.const_i32(cap_row))
                    out.append((row, col, live))
                return out

            a_slots = chunks(R, load_vec_a, R)
            b_slots = chunks(XT * block_n, load_vec_b, XT * block_n)

            def xh_fetch(st):
                a_off, b_off, t_ok = xh_uniforms(st)
                a_base = b.add(a_base0, a_off)
                a_st = []
                for row, col, live in a_slots:
                    off = b.add(b.add(a_base, b.mul(row, s_wi)), col)
                    ok = t_ok if live is None else b.land(t_ok, live)
                    safe = b.select(ok, b.mul(off, c_eb), c_oob_a)
                    a_st.append((row, col, b.buffer_load_vN(a_rsrc, safe, c0, ir_dtype_a, load_vec_a)))
                b_st = []
                c_tn = b.const_i32(block_n)
                for row, col, live in b_slots:
                    x_ = b.div(row, c_tn)
                    k_out = b.add(block_n_off_v, _eqx_chan(b, b.mod(row, c_tn)))
                    if grouped:
                        k_out = b.add(k_out_group_base, k_out)
                    off = b.add(b.add(b.mul(k_out, p_B_stride_k), b.mul(x_, p_B_stride_x)),
                                b.add(b_off, col))
                    ok = t_ok if live is None else b.land(t_ok, live)
                    safe = b.select(ok, b.mul(off, c_eb), c_oob_a)
                    b_st.append((row, col, b.buffer_load_vN(b_rsrc, safe, c0, ir_dtype_b, load_vec_b)))
                return a_st, b_st

            def xh_store(staged, B_dst=None):
                B_dst = B_smem if B_dst is None else B_dst
                for dst, st, vec in ((A_smem, staged[0], load_vec_a), (B_dst, staged[1], load_vec_b)):
                    for row, col, v in st:
                        if spec.xh_stride_m and dst is A_smem:
                            b.smem_store_vN(dst, list(_xa(b, row, col)), v, vec)
                        else:
                            b.smem_store_vN(dst, [row, lds_swz(b, row, col)], v, vec)

            def xh_adv(st):
                c_t, y_t, z_t = st
                if spec.uk_tap_fast:
                    g = spec.uk_chunk_group
                    if g:
                        c_gspan = b.const_i32(g * block_k)
                        c_gmask = b.const_i32(g * block_k - 1)
                        nc = b.add(c_t, c_block_k)
                        w_c = b.lor(b.cmp_eq(b.land(nc, c_gmask), c0), b.cmp_ge(nc, p_cpg))
                        nc = b.select(w_c, b.sub(c_t, b.land(c_t, c_gmask)), nc)
                        ny = b.add(y_t, b.select(w_c, c1, c0))
                    else:
                        c_gspan = c_block_k
                        nc = c_t
                        ny = b.add(y_t, c1)
                    w_y = b.cmp_ge(ny, p_Y)
                    ny = b.select(w_y, c0, ny)
                    nz = b.add(z_t, b.select(w_y, c1, c0))
                    w_z = b.cmp_ge(nz, p_Z)
                    nz = b.select(w_z, c0, nz)
                    nc = b.add(nc, b.select(w_z, c_gspan, c0))
                    nz = b.select(b.cmp_ge(nc, p_cpg), p_Z, nz)
                    return [nc, ny, nz]
                nc = b.add(c_t, c_block_k)
                w_c = b.cmp_ge(nc, p_cpg)
                nc = b.select(w_c, c0, nc)
                ny = b.add(y_t, b.select(w_c, c1, c0))
                w_y = b.cmp_ge(ny, p_Y)
                ny = b.select(w_y, c0, ny)
                nz = b.add(z_t, b.select(w_y, c1, c0))
                return [nc, ny, nz]

            # The MFMA A rows: lin(m0 + row) - lin0, then + x per tap.
            dec = decode_mfma_lanes(b, atom, lane)
            wm_off = grid.warp_m_off(b)
            base_rows = []
            for mi in range(mfmas_m):
                if spec.xh_stride_m:
                    a_row = b.add(wm_off, b.add(b.mul(dec.m_in_atom, b.const_i32(mfmas_m)),
                                                b.const_i32(mi)))
                else:
                    a_row = b.add(wm_off, b.add(b.const_i32(mi * spec.warp_tile_m), dec.m_in_atom))
                base_rows.append(b.sub(lin_of(b, b.add(m0, a_row)), lin0))

            def xh_mfma(accs_t, defer_last: bool = False):
                prev = None
                for x in range(XT):
                    if defer_last and x == XT - 1:
                        # defer_mfma: the last tap's last kk MFMAs run after the
                        # barrier, interleaved with the next tile's LDS store.
                        _mfma_defer.update(on=True, n=spec.defer_mfma, fns=[])
                    rows = base_rows if x == 0 else [b.add(r, b.const_i32(x)) for r in base_rows]
                    _shr = spec.xh_dpp or spec.xh_stride_m
                    rec = {} if _shr else None
                    accs_t = emit_mfma_phase(A_smem, B_smem, accs_t, a_lds_rows=rows,
                                             b_row_off=x * block_n,
                                             a_shift_prev=prev if _shr else None,
                                             a_rec=rec)
                    prev = rec
                _mfma_defer["on"] = False
                return accs_t

            _bar = b.sync_lds_only if spec.rpf_lds_barrier else b.sync
            n_acc = len(accs_in)
            st0 = [c0, c0, c0]
            if spec.xh_subtile:
                if xs_range is None:
                    xs_lo, xs_hi, xs_st = c0, xh_hi, st0
                else:
                    xs_lo, xs_hi_r, xs_st = xs_range
                    xs_hi = b.select(xh_ok, xs_hi_r, xs_lo)
                return xh_ok, _emit_xhalo_sub_pp(
                    accs_in, xh_ok, xs_lo, xs_hi, xs_st, xh_uniforms, xh_adv,
                    a_slots, a_base0, base_rows, chunks, c_oob_a, c_eb,
                )
            if spec.rpf_pingpong:
                return xh_ok, _emit_xhalo_pingpong(
                    accs_in, st0, xh_ok, xh_hi, xh_fetch, xh_store, xh_adv, base_rows, XT
                )
            with b.scf_if(xh_ok):
                xh_store(xh_fetch(st0))
                b.sync()
            it = [(f"xh_{n}", v) for (n, _), v in zip(accs, accs_in)] + [
                ("xh_c", c0), ("xh_y", c0), ("xh_z", c0)
            ]
            if spec.xh_wait_smem:
                # A scalar load still pending at the loop header makes the
                # first LDS wait of every iteration a full lgkmcnt(0) (SMEM
                # returns out of order); retire it once before the loop.
                b.s_waitcnt(lgkmcnt=0)
            for_op = b.scf_for_iter(c0, xh_hi, c_block_k, it, iv_name="xh_k")
            with for_op as (kv, iv):
                st = list(iv[n_acc:])
                st1 = xh_adv(st)
                if spec.rpf_setprio:
                    b.s_setprio(spec.rpf_setprio)
                staged = xh_fetch(st1)
                if spec.vmem_interleave:
                    # The first tap's MFMA phase spreads the loads, one per
                    # vmem_interleave MFMAs (they back up in the TA queue
                    # when issued back to back; ATT: 46-54 stall cycles each).
                    _vmi_pending[0] = len(staged[0]) + len(staged[1])
                else:
                    b.sched_barrier(0)
                acc_n = xh_mfma(list(iv[:n_acc]), defer_last=bool(spec.defer_mfma))
                b.sched_barrier(0)
                if spec.rpf_setprio:
                    b.s_setprio(0)
                if spec.defer_mfma:
                    b.sync_lds_only()
                    xh_store(staged)
                    for fn_ in _mfma_defer["fns"]:
                        acc_n = fn_(acc_n)
                    n_w = len(staged[0]) + len(staged[1])
                    n_m = spec.defer_mfma * mfmas_m * mfmas_n
                    # 1 ds_write : 2 MFMAs (measured best for defer 1 and 2;
                    # with defer 2 the other half of the MFMAs follow freely).
                    per_w = min(2, max(1, n_m // max(1, n_w)))
                    for _ in range(min(n_w, n_m)):
                        b.sched_group_barrier(DS_WRITE, 1, 0)
                        b.sched_group_barrier(MFMA, per_w, 0)
                    b.sched_barrier(0)
                    b.sync_lds_only()
                else:
                    _bar()
                    xh_store(staged)
                    _bar()
                b.scf_yield(*acc_n, *st1)
            return xh_ok, list(for_op.results)[:n_acc]

        # ---- uniform-tap loop ----
        c_bk_mask = b.const_i32(block_k - 1)
        uk_ok = b.cmp_eq(b.land(p_cpg, c_bk_mask), c0)
        if _split and spec.uk_chunk_group:
            # The split start decode below assumes whole channel groups.
            uk_ok = b.land(uk_ok, b.cmp_eq(
                b.land(p_cpg, b.const_i32(spec.uk_chunk_group * block_k - 1)), c0))
        uk_hi = b.select(uk_ok, c_K_gemm, c0)
        s_n = params["p_A_stride_n"]
        s_hi = params["p_A_stride_hi"]
        s_wi = params["p_A_stride_wi"]
        a_ystep = b.mul(p_dH, s_hi)
        a_xstep = b.mul(p_dW, s_wi)
        if is3:
            s_di = params["p_A_stride_di"]
            a_zstep = b.mul(p_dD, s_di)
        a_gbase = b.mul(group_idx, p_cpg) if grouped else None

        def tile_uniforms(st):
            # st = (c, x, y[, z]): channel base and filter tap of the tile.
            if is3:
                c_t, x_t, y_t, z_t = st
            else:
                c_t, x_t, y_t = st
            hsh = b.mul(y_t, p_dH)
            wsh = b.mul(x_t, p_dW)
            a_off = b.add(b.add(b.mul(y_t, a_ystep), b.mul(x_t, a_xstep)), c_t)
            b_off = b.add(
                b.add(b.mul(y_t, params["p_B_stride_y"]),
                      b.mul(x_t, params["p_B_stride_x"])),
                c_t,
            )
            if is3:
                dsh = b.mul(z_t, p_dD)
                a_off = b.add(a_off, b.mul(z_t, a_zstep))
                b_off = b.add(b_off, b.mul(z_t, params["p_B_stride_z"]))
                t_ok = b.cmp_lt(z_t, p_Z)
            else:
                dsh = None
                t_ok = b.cmp_lt(y_t, p_Y)
            return (hsh, wsh, dsh, a_off, b_off, t_ok)

        def a_uk(b_: IRBuilder, row: Value, col: Value):
            hsh, wsh, dsh, a_off, _b_off, t_ok = uk_cap[0]
            m_val = b_.add(block_m_off_v, row)
            q_wo, wo = magic_divmod(
                b_, m_val, params["p_magic_m_Wo_mult"],
                params["p_magic_m_Wo_shift"], p_Wo,
            )
            q_ho, ho = magic_divmod(
                b_, q_wo, params["p_magic_m_Ho_mult"],
                params["p_magic_m_Ho_shift"], p_Ho,
            )
            if is3:
                n, do_ = magic_divmod(
                    b_, q_ho, params["p_magic_m_Do_mult"],
                    params["p_magic_m_Do_shift"], p_Do,
                )
            else:
                n = q_ho
            hi0 = b_.sub(b_.mul(ho, p_sH), p_pH)
            wi0 = b_.sub(b_.mul(wo, p_sW), p_pW)
            base = b_.add(
                b_.add(b_.mul(n, s_n), b_.mul(hi0, s_hi)), b_.mul(wi0, s_wi)
            )
            if spec.uk_nopad:
                # Unpadded: every tap of an output pixel is in range; only the
                # tap-range check of the prefetch past the end stays.
                if is3:
                    di0 = b_.sub(b_.mul(do_, p_sD), p_pD)
                    base = b_.add(base, b_.mul(di0, s_di))
                if grouped:
                    base = b_.add(base, a_gbase)
                return b_.add(b_.add(base, col), a_off), t_ok
            # (u32)(origin + shift) < extent  <=>  0 <= origin + shift < extent
            ok = b_.land(
                b_.cmp_ult(b_.add(hi0, hsh), p_Hi),
                b_.cmp_ult(b_.add(wi0, wsh), p_Wi),
            )
            if is3:
                di0 = b_.sub(b_.mul(do_, p_sD), p_pD)
                base = b_.add(base, b_.mul(di0, s_di))
                ok = b_.land(ok, b_.cmp_ult(b_.add(di0, dsh), p_Di))
            if grouped:
                base = b_.add(base, a_gbase)
            ok = b_.land(ok, t_ok)
            return b_.add(b_.add(base, col), a_off), ok

        def b_uk(b_: IRBuilder, row: Value, col: Value):
            _h, _w, _d, _a, b_off, t_ok = uk_cap[0]
            k_out = b_.add(block_n_off_v, _eqx_chan(b_, row))
            if grouped:
                k_out = b_.add(k_out_group_base, k_out)
            off = b_.add(b_.add(b_.mul(k_out, params["p_B_stride_k"]), col), b_off)
            return off, t_ok

        def uk_fetch(_k_val, st):
            uk_cap[0] = tile_uniforms(st)
            if _fetch_part["hi"] is not None:
                # pp_end_guard: no reads for a tile at or past the loop's end.
                _u = uk_cap[0]
                uk_cap[0] = _u[:-1] + (b.land(_u[-1], b.cmp_lt(_k_val, _fetch_part["hi"])),)
            if spec.a_direct:
                return fetch_adir(a_uk, b_uk)
            return fetch(a_uk, b_uk)

        def uk_adv(st):
            if is3:
                c_t, x_t, y_t, z_t = st
            else:
                c_t, x_t, y_t = st
            if spec.uk_tap_fast and spec.uk_chunk_group:
                # Chunks fastest inside a group of uk_chunk_group chunks,
                # then the tap (x, y, z), then the next group.
                c_gspan = b.const_i32(spec.uk_chunk_group * block_k)
                c_gmask = b.const_i32(spec.uk_chunk_group * block_k - 1)
                nc = b.add(c_t, c_block_k)
                # End of the group, or of the channels (cpg % group != 0).
                w_c = b.lor(b.cmp_eq(b.land(nc, c_gmask), c0), b.cmp_ge(nc, p_cpg))
                nc = b.select(w_c, b.sub(c_t, b.land(c_t, c_gmask)), nc)
                nx = b.add(x_t, b.select(w_c, c1, c0))
                w_x = b.cmp_ge(nx, p_X)
                nx = b.select(w_x, c0, nx)
                ny = b.add(y_t, b.select(w_x, c1, c0))
                w_y = b.cmp_ge(ny, p_Y)
                ny = b.select(w_y, c0, ny)
                nz = b.add(z_t, b.select(w_y, c1, c0))
                w_z = b.cmp_ge(nz, p_Z)
                nz = b.select(w_z, c0, nz)
                nc = b.add(nc, b.select(w_z, c_gspan, c0))
                nz = b.select(b.cmp_ge(nc, p_cpg), p_Z, nz)
                return [nc, nx, ny, nz]
            if spec.uk_tap_fast:
                # Filter taps fastest, channel chunk outermost (see
                # ImplicitGemmConvSpec.uk_tap_fast).
                nx = b.add(x_t, c1)
                w_x = b.cmp_ge(nx, p_X)
                nx = b.select(w_x, c0, nx)
                ny = b.add(y_t, b.select(w_x, c1, c0))
                w_y = b.cmp_ge(ny, p_Y)
                ny = b.select(w_y, c0, ny)
                nz = b.add(z_t, b.select(w_y, c1, c0))
                w_z = b.cmp_ge(nz, p_Z)
                nz = b.select(w_z, c0, nz)
                nc = b.add(c_t, b.select(w_z, c_block_k, c0))
                # Past the end: park on an out-of-range tap (z = Z) so the
                # prefetch past the last tile loads zeros.
                nz = b.select(b.cmp_ge(nc, p_cpg), p_Z, nz)
                return [nc, nx, ny, nz]
            nc = b.add(c_t, c_block_k)
            w_c = b.cmp_ge(nc, p_cpg)
            nc = b.select(w_c, c0, nc)
            nx = b.add(x_t, b.select(w_c, c1, c0))
            w_x = b.cmp_ge(nx, p_X)
            nx = b.select(w_x, c0, nx)
            ny = b.add(y_t, b.select(w_x, c1, c0))
            if not is3:
                return [nc, nx, ny]
            w_y = b.cmp_ge(ny, p_Y)
            ny = b.select(w_y, c0, ny)
            nz = b.add(z_t, b.select(w_y, c1, c0))
            return [nc, nx, ny, nz]

        if not _split:
            uk_state = [("uc", c0), ("ux", c0), ("uy", c0)] + (
                [("uz", c0)] if is3 else []
            )
            accs_in0 = [v for _, v in accs]
            if spec.uk_xhalo:
                xh_ok, accs_in0 = _emit_xhalo_loop(uk_ok, accs_in0)
                uk_hi = b.select(b.land(uk_ok, b.lnot(xh_ok)), c_K_gemm, c0)
            uk_accs = run_loop(c0, uk_hi, accs_in0, uk_state,
                               uk_fetch, uk_adv, "uk", adir=spec.a_direct)
            fb_lo = b.select(uk_ok, c_K_gemm, c0)
            return run_loop(fb_lo, c_K_gemm, uk_accs, state0, plain_fetch,
                            plain_adv, "fb")
        # split_k: this slice runs K tiles [k_lo, k_hi) / block_k (the slice
        # length is a whole number of tiles), so the uniform tap state starts
        # at tile k_lo / block_k in the loop's own tile order. A slice past
        # the end starts on an out-of-range tap and loads zeros.
        t_lo = b.div(k_lo, c_block_k)
        n_taps_xy = b.mul(p_X, p_Y)
        if spec.uk_tap_fast and spec.uk_chunk_group:
            # Grouped tap-fast order: uk_chunk_group chunks fastest, then the
            # tap, then the next group (cpg a multiple of the group span).
            c_g = b.const_i32(spec.uk_chunk_group)
            n_taps = b.mul(n_taps_xy, p_Z)
            g_tiles = b.mul(n_taps, c_g)
            grp = b.div(t_lo, g_tiles)
            g_r = b.mod(t_lo, g_tiles)
            tap = b.div(g_r, c_g)
            c_s = b.mul(b.add(b.mul(grp, c_g), b.mod(g_r, c_g)), c_block_k)
            x_s = b.mod(tap, p_X)
            y_s = b.mod(b.div(tap, p_X), p_Y)
            z_s = b.div(tap, n_taps_xy)
            z_s = b.select(b.cmp_ge(c_s, p_cpg), p_Z, z_s)
        elif spec.uk_tap_fast:
            n_taps = b.mul(n_taps_xy, p_Z)
            tap = b.mod(t_lo, n_taps)
            c_s = b.mul(b.div(t_lo, n_taps), c_block_k)
            x_s = b.mod(tap, p_X)
            y_s = b.mod(b.div(tap, p_X), p_Y)
            z_s = b.div(tap, n_taps_xy)
            z_s = b.select(b.cmp_ge(c_s, p_cpg), p_Z, z_s)
        else:
            tap = b.div(k_lo, p_cpg)
            c_s = b.sub(k_lo, b.mul(tap, p_cpg))
            x_s = b.mod(tap, p_X)
            y_s = b.mod(b.div(tap, p_X), p_Y) if is3 else b.div(tap, p_X)
            z_s = b.div(tap, n_taps_xy) if is3 else None
        uk_state = [("uc", c_s), ("ux", x_s), ("uy", y_s)] + (
            [("uz", z_s)] if is3 else []
        )
        uk_hi_s = b.select(uk_ok, k_hi, k_lo)
        accs_s = [v for _, v in accs]
        uk_run_hi = uk_hi_s
        if spec.uk_xhalo:
            # xh_subtile: a slice of the x-halo loop = a range of its
            # (chunk, z, y) steps (the x-halo order), ks_count equal parts;
            # a split_tail full tile runs them all.
            c_bk = c_block_k
            xs_J = b.mul(b.mul(p_Z, p_Y), p_cpg)
            xs_jt = b.div(xs_J, c_bk)
            xs_w = b.div(b.add(xs_jt, b.sub(p_ks_count, c1)), p_ks_count)
            xs_lo = b.mul(b.mul(_slice, xs_w), c_bk)
            xs_hi = b.smin(b.add(xs_lo, b.mul(xs_w, c_bk)), xs_J)
            if spec.split_tail:
                xs_lo = b.select(_is_full, c0, xs_lo)
                xs_hi = b.select(_is_full, xs_J, xs_hi)
            xs_lo = b.to_sgpr_u32(xs_lo)
            xs_hi = b.to_sgpr_u32(xs_hi)
            xs_i = b.div(xs_lo, c_bk)
            if spec.uk_tap_fast:
                xs_y = b.mod(xs_i, p_Y)
                xs_z = b.mod(b.div(xs_i, p_Y), p_Z)
                xs_c = b.mul(b.div(xs_i, b.mul(p_Y, p_Z)), c_bk)
                xs_z = b.select(b.cmp_ge(xs_c, p_cpg), p_Z, xs_z)
            else:
                xs_nc = b.div(p_cpg, c_bk)
                xs_c = b.mul(b.mod(xs_i, xs_nc), c_bk)
                xs_y = b.mod(b.div(xs_i, xs_nc), p_Y)
                xs_z = b.div(xs_i, b.mul(xs_nc, p_Y))
            xh_ok_s, accs_s = _emit_xhalo_loop(
                uk_ok, accs_s, xs_range=(xs_lo, xs_hi, [xs_c, xs_y, xs_z])
            )
            uk_run_hi = b.select(xh_ok_s, k_lo, uk_hi_s)
        uk_accs = run_loop(k_lo, uk_run_hi, accs_s, uk_state,
                           uk_fetch, uk_adv, "uk", adir=spec.a_direct)
        return run_loop(uk_hi_s, k_hi, uk_accs, state0, plain_fetch, plain_adv, "fb")

    def emit_fetch_phase(k_off: Value, interleave: bool = False) -> Tuple[list, list]:
        """Global -> registers for one K tile (``prefetch_regs``).

        ``interleave``: leave the loads to the MFMA phase that follows to place
        (``vmem_interleave``) instead of pinning them ahead of it."""
        k_off_capture[0] = k_off
        a_regs = a_sync_loader.fetch(b, tid=tid, descriptor=a_descriptor, rsrc=a_rsrc)
        b_regs = b_sync_loader.fetch(b, tid=tid, descriptor=b_descriptor, rsrc=b_rsrc)
        if interleave and spec.vmem_interleave:
            _vmi_pending[0] = len(a_regs) + len(b_regs)
            return a_regs, b_regs
        # Pin the loads ahead of the MFMA phase: left free, the backend sinks
        # them next to their ds_write consumers and the latency is exposed
        # again.
        b.sched_barrier(0)
        return a_regs, b_regs

    def emit_store_phase(regs: Tuple[list, list], A_dst: Value, B_dst: Value) -> None:
        """Registers -> LDS for a tile fetched by :func:`emit_fetch_phase`."""
        b.sched_barrier(0)
        if spec.lds_xor:
            for dst, fetched, vec in (
                (A_dst, regs[0], a_sync_loader.load_vec),
                (B_dst, regs[1], b_sync_loader.load_vec),
            ):
                for row, col, v in fetched:
                    b.smem_store_vN(dst, [row, lds_swz(b, row, col)], v, vec)
            return
        a_sync_loader.store_fetched(b, smem_dst=A_dst, fetched=regs[0])
        b_sync_loader.store_fetched(b, smem_dst=B_dst, fetched=regs[1])

    _szf_idx = [None]
    if _split and spec.split_zero_first:
        # Slice 0 of a split tile zeroes the tile's D (bounds-checked wide
        # stores), releases it and raises the flag the slices' atomic stores
        # wait for.
        _t_lin = b.add(
            b.mul(b.div(block_m_off_v, b.const_i32(block_m)), p_num_pid_n),
            b.div(block_n_off_v, b.const_i32(block_n)),
        )
        if grouped:
            _t_lin = b.add(_t_lin, b.mul(group_idx, b.mul(p_num_pid_m, p_num_pid_n)))
        _szf_idx[0] = b.to_sgpr_u32(b.mul(_t_lin, b.const_i32(2)))
        _zero_cond = b.cmp_eq(_slice, c0)
        if spec.split_tail:
            _zero_cond = b.land(_zero_cond, b.lnot(_is_full))
        with b.scf_if(_zero_cond):
            _zsv = 8
            _zvpr = block_n // _zsv
            _zthreads = spec.block_size
            _zbf = b.vec_trunc_f32_to_bf16(b.zero_vec_f32(_zsv))
            _c_eb = b.const_i32(2)
            _oob = b.const_i32((1 << 31) - 1)
            for e in range(-(-block_m * _zvpr // _zthreads)):
                vidx = b.add(b.const_i32(e * _zthreads), tid)
                row = b.div(vidx, b.const_i32(_zvpr))
                col = b.mul(b.mod(vidx, b.const_i32(_zvpr)), b.const_i32(_zsv))
                m_val = b.add(block_m_off_v, row)
                n_val = b.add(block_n_off_v, col)
                ok = b.land(
                    b.land(b.cmp_lt(row, b.const_i32(block_m)), b.cmp_lt(m_val, p_M)),
                    b.cmp_le(b.add(n_val, b.const_i32(_zsv)), c_N_bound),
                )
                off, valid = d_addr(b, m_val, n_val)
                if valid is not None:
                    ok = b.land(ok, valid)
                safe = b.select(ok, b.mul(off, _c_eb), _oob)
                b.buffer_store_vN_bf16(d_rsrc, safe, c0, _zbf, _zsv // 2)
            b.fence(scope="agent", ordering="release")
            b.sync()
            with b.scf_if(b.cmp_eq(tid, c0)):
                b.global_flag_store(p_sk_flags, _szf_idx[0], b.const_i32(1))
    if spec.uniform_k_decode or spec.reg_prefetch:
        final_accs = _emit_fwd_kloops()
    elif spec.prefetch_regs and a_load_override is None:
        # Register-staged prefetch. Tile t+1 is fetched into registers before
        # tile t's MFMAs and stored to LDS after them, so the load latency
        # overlaps the MFMAs of the issuing wave. As in the unroll_k loop, the
        # fetch past the last tile addresses k >= K_gemm, which the descriptor
        # bounds turn into zero-filled loads, so no tail guard is needed.
        if spec.lds_xor:
            emit_store_phase(emit_fetch_phase(k_lo), A_smem, B_smem)
        else:
            emit_load_phase(k_lo, A_smem, B_smem)
        b.sync()
        if spec.unroll_k:
            # Ping-pong: one barrier per tile publishes the stored tile and
            # retires the reads of the buffer the next store overwrites.
            c_2block_k = b.const_i32(2 * block_k)
            for_op = b.scf_for_iter(
                c0, c_K_gemm, c_2block_k, accs, iv_name="k_unroll"
            )
            with for_op as (k_unroll, iter_accs):
                k_odd = b.add(k_unroll, c_block_k)
                k_nxt_pair = b.add(k_unroll, c_2block_k)
                regs = emit_fetch_phase(k_odd)
                k_off_capture[0] = k_unroll
                accs_a = emit_mfma_phase(A_smem, B_smem, list(iter_accs))
                emit_store_phase(regs, A_smem2, B_smem2)
                b.sync()
                regs = emit_fetch_phase(k_nxt_pair)
                k_off_capture[0] = k_odd
                accs_b = emit_mfma_phase(A_smem2, B_smem2, accs_a)
                emit_store_phase(regs, A_smem, B_smem)
                b.sync()
                b.scf_yield(*accs_b)
        else:
            # Single buffer: the store waits behind a barrier that retires
            # every wave's reads of the current tile.
            for_op = b.scf_for_iter(k_lo, k_hi, c_block_k, accs, iv_name="k0")
            with for_op as (k0, iter_vars):
                _prio_on()
                regs = emit_fetch_phase(b.add(k0, c_block_k), interleave=True)
                k_off_capture[0] = k0
                new_accs = emit_mfma_phase(A_smem, B_smem, list(iter_vars))
                _prio_off()
                b.sync()
                emit_store_phase(regs, A_smem, B_smem)
                b.sync()
                b.scf_yield(*new_accs)
        final_accs = for_op.results
    elif spec.unroll_k:
        # Double-buffered K-loop software pipeline.
        #
        # Stage tile it+1 into the alternate LDS buffer while the MFMA for
        # tile it reads the current buffer. The buffers are disjoint, so the
        # next tile's global->LDS writes overlap the current tile's ds_read +
        # MFMA work instead of being serialized behind a barrier.
        #
        # AOT: the trip count is the runtime ``p_K_gemm``, so the buffer for
        # an iteration cannot be picked by ``bufs[it % 2]`` — an LDS
        # allocation is a build-time SSA value. Instead the body is unrolled
        # twice and steps by 2*block_k, which binds each phase to a
        # Python-time buffer while still alternating them.
        #
        # One barrier per tile does double duty, exactly as in the
        # compile-time form: it publishes the tile just prefetched into the
        # other buffer, and it orders the current tile's ds_reads ahead of
        # the prefetch that reuses that buffer two tiles later.
        #
        # An odd tile count needs no guard: the trailing phase addresses
        # k >= K_gemm, whose coords fall outside the descriptor's padded
        # bounds, so the buffer resource returns zero and the MFMA adds
        # nothing. That is the same zero-fill the single-buffer path already
        # relies on for a partial last tile.
        c_2block_k = b.const_i32(2 * block_k)

        # Prologue: stage tile 0 into buf0 and publish it.
        emit_load_phase(c0, A_smem, B_smem)
        b.sync()

        for_op = b.scf_for_iter(c0, c_K_gemm, c_2block_k, accs, iv_name="k_unroll")
        with for_op as (k_unroll, iter_accs):
            k_odd = b.add(k_unroll, c_block_k)
            k_nxt_pair = b.add(k_unroll, c_2block_k)

            # Phase A: prefetch tile k+1 into buf1, MFMA tile k out of buf0.
            emit_load_phase(k_odd, A_smem2, B_smem2)
            k_off_capture[0] = k_unroll
            accs_a = emit_mfma_phase(A_smem, B_smem, list(iter_accs))
            # Publishes buf1 and drains buf0's ds_reads so the next
            # emit_load_phase may overwrite buf0.
            b.sync()

            # Phase B: the buffers swap roles.
            emit_load_phase(k_nxt_pair, A_smem, B_smem)
            k_off_capture[0] = k_odd
            accs_b = emit_mfma_phase(A_smem2, B_smem2, accs_a)
            b.sync()

            b.scf_yield(*accs_b)

        final_accs = for_op.results
    elif not spec.async_dma:
        # mem / compv3 / compv4 / basic: all use scf_for_iter — only bound changes to runtime.
        for_op = b.scf_for_iter(k_lo, k_hi, c_block_k, accs, iv_name="k0")
        with for_op as (k0, iter_vars):
            emit_load_phase(k0, A_smem, B_smem)
            b.sync()
            new_accs = emit_mfma_phase(A_smem, B_smem, iter_vars)
            b.sync()
            b.scf_yield(*new_accs)
        final_accs = for_op.results
    else:
        # async_dma path: double-buffered ping-pong over the runtime K extent.
        bufs = [(A_smem, B_smem), (A_smem2, B_smem2)]

        pipeline = SoftwarePipeline(
            num_iters=0,  # unused by the dynamic driver; K comes from p_K_gemm
            double_buffer=double_buffer,
            wait_vmcnt=True,
            sync_after_wait=True,
            sync_before_issue=True,
            overlap_vmcnt=True,
        )

        def issue_load_dyn(k_offset_val, buf_pair):
            emit_load_phase(k_offset_val, buf_pair[0], buf_pair[1])

        def compute_dyn(k_offset_val, buf_pair, state):
            k_off_capture[0] = k_offset_val
            return emit_mfma_phase(buf_pair[0], buf_pair[1], state)

        final_accs = pipeline.run_ping_pong_dynamic(
            b,
            k_extent=c_K_gemm,
            block_k=block_k,
            # The kernel's own c0, not the helper's default fresh const 0: the
            # C++ k-loop driver passes ctx->c0, and the builder does not dedupe
            # constants, so the two engines must name the same value.
            k_lo=c0,
            buffers=bufs,
            iter_args=accs,
            issue_load_fn=issue_load_dyn,
            compute_fn=compute_dyn,
            schedule=schedule,
        )

    # ---- epilogue ----
    final_accs = _apply_accumulator_epilogue(b, spec.acc_epilogue, final_accs)
    if epilogue_override is not None:
        epilogue_override(b, spec, final_accs, grid, d_rsrc, extra_context)
    else:
        _emit_fwd_epilogue(final_accs)
    if (spec.split_streamk or spec.split_hybrid) and spec.split_k > 1:
        # The next segment's prologue rewrites the LDS the epilogue staged C in.
        b.sync()
        _fsk_loop.__exit__(None, None, None)
    return b.kernel


def async_tile_loaders(
    spec: "ImplicitGemmConvSpec", arch: str = "gfx950"
) -> Tuple[AsyncTileLoader, AsyncTileLoader]:
    """The A and B loaders of the ``async_dma`` path, built from ``spec``.

    contig_cols: the tile's col axis is the reduction index (y, x, c) and only
    the inner ``c`` is stride-1, so a chunk wider than cpg -- or one that does
    not divide it -- would straddle a filter position and fetch the wrong
    elements with no diagnostic. The chunk width is therefore chosen from the
    build-time cpg and baked into the ISA: the binary is only correct for a
    cpg its chunk widths divide. That makes them a capability of the binary,
    which is why this is a function the kernel cache can call too rather than
    inline builder code.

    ``arch`` caps the chunk width at what the target's ``buffer_load_lds`` can
    actually move per lane (``ArchTarget.async_lds_max_dwords``): CDNA3 has
    only the dword form, the b96/b128 forms arrived with CDNA4. Passing a
    wider one is not a compile error -- the AMDGPU backend aborts the process
    with ``LLVM ERROR: Do not know how to expand this operator's operand!`` --
    so the cap has to be applied here, where the width is chosen.
    """
    from rocke.core.arch import ArchTarget

    p = spec.problem
    max_dwords = ArchTarget.from_gfx(arch).async_lds_max_dwords
    a_loader = AsyncTileLoader.from_tile(
        tile_rows=spec.tile_m,
        tile_cols=spec.tile_k,
        block_size=spec.block_size,
        wave_size=spec.wave_size,
        elem_dtype=_ir_dtype(spec.data.dtype_a),
        max_dwords=max_dwords,
        contig_cols=p.cpg,
    )
    b_loader = AsyncTileLoader.from_tile(
        tile_rows=spec.tile_n,
        tile_cols=spec.tile_k,
        block_size=spec.block_size,
        wave_size=spec.wave_size,
        elem_dtype=_ir_dtype(spec.data.dtype_b),
        max_dwords=max_dwords,
        contig_cols=p.cpg,
    )
    return a_loader, b_loader


def build_implicit_gemm_conv(
    spec: "ImplicitGemmConvSpec", *, arch: str = "gfx950"
) -> "KernelDef":
    """Build the IR for one implicit-GEMM forward convolution kernel.

    Public API: takes only ``spec`` and ``arch``.  For internal extension via
    descriptor callbacks (``extra_params``, ``m_index_fn``, etc.) call
    :func:`_build_implicit_gemm_conv_impl` directly.
    """
    return _build_implicit_gemm_conv_impl(spec, arch=arch)

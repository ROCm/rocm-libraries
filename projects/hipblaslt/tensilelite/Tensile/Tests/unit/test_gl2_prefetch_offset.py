#!/usr/bin/env python3
################################################################################
# GL2-prefetch address verification test (gfx1250).
#
# Runs the *production* GL2 prefetch address computation
# (GL2PrefetchLoad.init / setIncrement / calculateStartAddr / incrementAddr)
# inside a real assembled kernel on a gfx1250 GPU. The prefetch
# (global_prefetch_b8, which returns no data) is replaced by exporting each
# computed address as a byte offset from the tensor base. The host then checks
# that, across all cooperative threads and all per-thread loads, the computed
# offsets tile the coalesced x perpendicular block exactly once in steps of
# GlobalPrefetchSize.
#
# Each config describes a problem and a list of tensors to prefetch. A single
# kernel computes the addresses for *all* of them in one round -- exactly like
# production gl2PrefetchCalcAddr (setIncrement then calculateStartAddr per
# tensor) -- so A, B, MXSA and MXSB are exercised together, catching
# register/sgpr aliasing across tensors.
#
# Coverage:
#   - data tensors A and B (non-MX), TLU and non-TLU layouts
#   - MX scale tensors MXSA / MXSB (mxUnit = MatrixInstK / MXBlock)
#   - FP8 (bpe=1) and FP4 (bpe=0.5) element sizes, including mixed A/B dtypes
#   - ClusterDim != [1,1]: gl2-prefetch is emitted whenever PrefetchGL2 is set,
#     but the cooperative fan-out only engages for a real cluster, so every config
#     runs a [cx, cy] grid (shapes vary, incl. [4,4]). Each
#     WG self-identifies via ttmp (gfx12 carries the workgroup id in ttmp, not
#     s2): wg_x -> WorkGroup0, wg_y -> WorkGroup1. A 2D cluster drives A and B
#     (and MXSA/MXSB) cooperatively at the same time. The *whole* cluster
#     cooperates: A is macro-tile-selected by WorkGroup0 and cooperative across
#     the rest of the cluster, B is the mirror, and together the cluster's
#     workgroups cover every macro-tile the cluster consumes (contiguous along
#     the MT-selector axis). Each WG writes to its own output region; the host
#     aggregates across all cx*cy.
#   - StridedBatched: a batch dim (index 2) maps to WorkGroup2, and
#     calculateStartAddr folds WorkGroup2 * Stride{tc}K into the base address.
#     Batched configs launch a 3D [cx, cy, num_batches] grid (wg_z from
#     ttmp7[31:16]); each batch b is verified against its footprint shifted by
#     b * Stride{tc}K * bpe.
#   - Address increment (PGR=2, PGL=2): with PrefetchGlobalRead>1 the start
#     address is pre-skipped by PGR*inc inside calculateStartAddr, and each
#     prefetched-ahead iteration advances every address by `inc` (incrementAddr).
#     The kernel re-exports all addresses across n_inc+1 stages; stage s must be
#     the base footprint shifted by (PGR+s)*inc along the summation (K) axis.
#   - GlobalSplitU: each workgroup prefetches only its own slice of K, so
#     calculateGSUIterOffset/applyGSUChunk shift the start address by
#     startIter*inc and widen the per-iteration step to the chunk stride. Both
#     chunk layouts are covered: interleaved (GSUC=0, group g starts at
#     iteration g and steps G at a time) and contiguous (GSUC=1, group g starts
#     after every lower group's run and steps one at a time, with the first
#     numIter%G groups getting an extra iteration). Each group is verified
#     against its *own* chunk rather than aggregated, so a group landing on the
#     wrong K slice fails. The group index is fed from the grid z axis rather
#     than split out of workgroup y as production does; see build_kernel.
#   - Non-power-of-2 MacroTile (e.g. 384 for A/B, 192/96 for MX scales): exercises
#     MT offset, non-POT gl2ncc (vectorStaticDivideAndRemainder), and non-POT
#     perpendicular/coalesced extents. DepthU remains a multiple of MatrixInstK.
# Verification is set-based: the union of each tensor's computed byte offsets
# (per stage) must equal the cluster's contiguous prefetch footprint
# {perp*perp_stride + c*GPS} (the mt_tiles macro-tiles the cluster spans folded
# into one block), shifted by the stage's K increment. The cluster's workgroups
# jointly enumerate this footprint (the host aggregates across all cx*cy). It
# tolerates the benign replication of the whole-cluster scheme (overlapping
# cooperative-thread slices, and nc < cooperative threads, e.g. MX scales).
#
# Usage:
#   pytest test_gl2_prefetch_offset.py -v -s
#   python test_gl2_prefetch_offset.py --debug
################################################################################

import os
import sys
import struct
import tempfile
import types
from dataclasses import dataclass
from math import ceil
from types import SimpleNamespace

import pytest
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, TENSILE_ROOT)

# Reuse the shared GPU test harness (target detection, rocIsa init, assembly,
# register scanning, kernel launch) instead of re-implementing it here. The
# wave32-specific bits (gfx1250) are passed as args to the wave-size-aware
# helpers.
from gpu_test_helpers import (  # noqa: E402
    GFX_TARGET,
    run_on_gpu,
    init_rocisa,
    assemble_kernel,
    _scan_register_indices,
)


# ---------------------------------------------------------------------------
# GPU target (this test is gfx1250-only)
# ---------------------------------------------------------------------------
HAS_GFX1250 = GFX_TARGET == "gfx1250"
WAVESIZE = 32
GLOBAL_PREFETCH_SIZE = 256


# ---------------------------------------------------------------------------
# Test configurations
# ---------------------------------------------------------------------------
@dataclass
class TensorSpec:
    tc: str            # "A", "B", "MXSA", "MXSB", "Metadata"
    tlu: bool
    mt: int            # MacroTile (A-type tensors share MacroTileA; B-type MacroTileB)
    bpe: float = 1     # bytes/elem: FP8/scale=1 (int), FP4=0.5
    is_m: bool = False       # sparse metadata tensor (isM in GL2Prefetch)
    sparse_side: str = None  # for is_m: "A" or "B" -- which sparse data tensor this
                             # metadata mirrors (Sparse==1 -> A, Sparse==2 -> B); tc
                             # itself is always literally "Metadata"

    @property
    def subtc(self):
        return self.sparse_side if self.is_m else self.tc[-1]  # 'A' or 'B'

    @property
    def idx(self):
        return 0 if self.subtc == "A" else 1

    @property
    def is_mx(self):
        return self.tc.startswith("MX")

    @property
    def ia(self):
        return [self.idx, 3]


@dataclass
class GL2Config:
    name: str
    tensors: list             # list[TensorSpec] computed together in one kernel
    depth_u: int = 256
    num_threads: int = 256
    cluster: tuple = (1, 1)   # ClusterDim [x, y]. A-type tensors are cooperative
                              # along y (WorkGroup1) and macro-tile-selected by x
                              # (WorkGroup0); B-type is the mirror image. So a 2D
                              # cluster exercises A and B cooperatively at once.
    matrix_inst_k: int = 128
    mx_block: int = 32
    size_i: int = None        # free-dim size (M) override for A-type; None => clean M*MT
    size_j: int = None        # free-dim size (N) override for B-type; None => clean N*MT
    pgr: int = 2              # PrefetchGlobalRead. PGR>1 makes calculateStartAddr
                              # pre-skip PGR*inc (the addr-increment fast path); 0
                              # would skip the increment logic entirely.
    pgl: int = 2              # PrefetchGL2 (>=2 prefetches ahead, advancing the
                              # address by `inc` each iteration via incrementAddr).
    n_inc: int = 2            # extra incrementAddr stages to verify after the start
                              # addr, i.e. exercise the per-iteration advance.
    batched: bool = False     # StridedBatched: add a batch dim (index 2 -> wg_z).
                              # calculateStartAddr then folds WorkGroup2 * Stride{tc}K
                              # into the base address (the batch-offset path).
    num_batches: int = 1      # batch extent (grid z). Each batch b shifts the whole
                              # footprint by b * Stride{tc}K * bpe.
    sparse: int = 0           # 0 = dense; 1 = A is the 2:4-compressed sparse operand;
                              # 2 = B is. Halves _DepthU{A,B} for that data tensor
                              # (GL2Prefetch reads _DepthU{A,B}, not the raw DepthU,
                              # so this must be modeled here too) and is required
                              # whenever cfg.tensors includes a Metadata (_M) spec.
    depth_u_metadata: int = 0 # _DepthUMetadata: metadata's own (already-compressed)
                              # unroll extent; independent of DepthU/_DepthU{A,B}.
    gsu: int = 0              # GlobalSplitU group count; 0 leaves the GSU paths off
                              # entirely (GlobalSplitU=0), 1 exercises them with a
                              # single group. Production splits the y axis to derive
                              # the group index; this harness feeds it from z instead
                              # (see build_kernel for why that is equivalent here), so
                              # a GSU config cannot also be batched.
    gsuc: bool = False        # GlobalSplitUCoalesced: False = interleaved chunks
                              # (group g starts at iteration g, steps G at a time),
                              # True = contiguous chunks (group g starts after the
                              # lower groups' runs, steps one at a time).
    k_iters: int = 8          # unroll iterations in the summation loop; programmed as
                              # SizesSum = k_iters * DepthU. The contiguous GSU layout
                              # reads it (numIter = SizesSum / DepthU), where
                              # k_iters % gsu picks how many groups get an extra
                              # iteration, and it is the numIter the StaggerU rotation
                              # wraps at (per group; see num_iters).
    stagger: int = None       # StaggerUIter, the number of unroll iterations to rotate
                              # the summation loop by. None leaves the StaggerU paths
                              # out entirely; an int (0 included) emits them, so 0
                              # covers "stagger code generated, StaggerU off at
                              # runtime" -- what every non-cluster GL2 kernel now
                              # builds. Must be < the smallest group's numIter, which
                              # declareStaggerParms guarantees in production.

    @property
    def n_wg(self):
        return self.cluster[0] * self.cluster[1]

    @property
    def gsu_on(self):
        return self.gsu > 0

    @property
    def stagger_on(self):
        return self.stagger is not None

    @property
    def n_groups(self):
        """GSU groups the launch splits K across (1 when the GSU paths are off)."""
        return max(1, self.gsu)

    @property
    def grid_z(self):
        # batches and GSU groups share the z axis; a config uses at most one.
        return self.num_batches * self.n_groups

    @property
    def n_regions(self):
        # distinct output regions = cooperative cluster wgs * grid z
        return self.n_wg * self.grid_z


def num_cooperative_threads(cfg, subtc):
    """Cooperative thread count = the *whole* cluster (matches GL2Prefetch.init).
    Every workgroup in the cluster cooperates on the prefetch, so the pool is
    ClusterDim[0]*ClusterDim[1]*NumThreads regardless of the tensor."""
    return cfg.cluster[0] * cfg.cluster[1] * cfg.num_threads


def depth_u_side(side, cfg):
    """_DepthU{A,B}: the per-tensor unroll extent GL2Prefetch.init actually reads
    for a plain data tensor (matches Solution.py). Equal to DepthU, except for
    the 2:4-compressed sparse data tensor (A when Sparse==1, B when Sparse==2),
    which is physically halved along K."""
    if (cfg.sparse == 1 and side == "A") or (cfg.sparse == 2 and side == "B"):
        return cfg.depth_u // 2
    return cfg.depth_u


def data_depth_u(spec, cfg):
    return depth_u_side(spec.subtc, cfg)


def tensor_dims(spec, cfg):
    """(coal_dim, perp_dim, ncc, nc) for a tensor, matching GL2Prefetch.init.
    The prefetched block spans *all* the macro-tiles the cluster consumes along
    the MT-selector axis (mt_tiles of them, contiguous in memory), so the tile
    dimension is scaled by mt_tiles: it is the coalesced dim for TLU/MX and the
    perpendicular dim for non-TLU."""
    M = mt_tiles(spec, cfg)
    if spec.is_m:
        # metadata uses its own compressed unroll extent (_DepthUMetadata), not
        # DepthU/_DepthU{A,B}; bpe is always 1 (already byte-granular)
        coal, perp = (spec.mt * M, cfg.depth_u_metadata) if spec.tlu else (cfg.depth_u_metadata, spec.mt * M)
    elif spec.is_mx:
        coal = spec.mt * M * cfg.matrix_inst_k // cfg.mx_block
        perp = cfg.depth_u // cfg.matrix_inst_k
    else:
        du = data_depth_u(spec, cfg)
        coal, perp = (spec.mt * M, du) if spec.tlu else (du, spec.mt * M)
    ncc = max(1, round(coal * spec.bpe) // GLOBAL_PREFETCH_SIZE)
    return coal, perp, ncc, perp * ncc


def tensor_gl2nl(spec, cfg):
    """Loads per thread (gl2nl), matching GL2Prefetch.init."""
    nc = tensor_dims(spec, cfg)[3]
    return max(1, ceil(nc / num_cooperative_threads(cfg, spec.subtc)))


def free_dim_size(cfg, subtc):
    """Programmed SizeI/SizeJ (the GEMM free-dim, used for the edge-limit clamp).
    Defaults to a clean tiling (cluster_extent * MacroTile); an explicit size_i/
    size_j makes the last macro-tile partial so the edge clamp fires."""
    mt = next((t.mt for t in cfg.tensors if t.subtc == subtc and not t.is_m), 1)
    if subtc == "A":
        return cfg.size_i if cfg.size_i is not None else cfg.cluster[0] * mt
    return cfg.size_j if cfg.size_j is not None else cfg.cluster[1] * mt


def gsu_start_iter(cfg, group):
    """Unroll iteration where `group`'s K chunk starts, matching
    GL2Prefetch.calculateGSUIterOffset.

    Interleaved chunks put group g at iteration g. Contiguous chunks put it
    after every lower group's run: with numIter = q*G + r, the first r groups
    get q+1 iterations and the rest get q."""
    if not cfg.gsu_on:
        return 0
    if not cfg.gsuc:
        return group
    q, r = divmod(cfg.k_iters, cfg.n_groups)
    return group * q + min(group, r)


def gsu_iter_stride(cfg):
    """Unroll iterations one prefetch increment covers, matching the increment
    scaling in GL2Prefetch.applyGSUChunk: a whole GSU round for interleaved
    chunks, a single iteration for contiguous ones (and with GSU off)."""
    return cfg.n_groups if (cfg.gsu_on and not cfg.gsuc) else 1


def num_iters(cfg, group):
    """Unroll iterations in `group`'s chunk -- the LoopCounterL the kernel runs
    with, and the modulus the StaggerU rotation wraps at.

    With numIter = q*G + r the first r groups get q+1 iterations and the rest
    get q, in both chunk layouts: the contiguous one hands out the longer runs
    first, and the interleaved one leaves the extra iterations with the low
    groups because they start earlier."""
    if not cfg.gsu_on:
        return cfg.k_iters
    q, r = divmod(cfg.k_iters, cfg.n_groups)
    return q + (1 if group < r else 0)


def stagger_chunk_idx(cfg, stage, group):
    """Index into `group`'s chunk that stage `stage` prefetches.

    Stage s sits at prefetch position PGR + s of the stream: PGR from the
    calculateStartAddr pre-skip, plus one incrementAddr per later stage. Two
    things bend that into a chunk index:
      - the end-of-K freeze pins the position at the last one in the chunk, and
      - StaggerU rotates the chunk, so a position lands on
        (StaggerUIter + position) % numIter rather than on the position itself.
    Without StaggerU the position is the index and the freeze never engages
    within the stages a config runs, so this collapses to PGR + stage."""
    pos = cfg.pgr + stage
    if not cfg.stagger_on:
        return pos
    n = num_iters(cfg, group)
    return (cfg.stagger + min(pos, n - 1)) % n


def mt_tiles(spec, cfg):
    """Number of distinct macro-tiles this tensor sweeps = the launch extent
    along its macro-tile-selector axis (WorkGroup{tIdx}). A-type tensors are
    MT-selected by WorkGroup0 (ClusterDim[0]); B-type by WorkGroup1
    (ClusterDim[1]). The other axis is the cooperative one."""
    return cfg.cluster[0] if spec.subtc == "A" else cfg.cluster[1]


def _A(tlu, mt, bpe=1):   return TensorSpec("A", tlu, mt, bpe)
def _B(tlu, mt, bpe=1):   return TensorSpec("B", tlu, mt, bpe)
def _MXSA(mt):            return TensorSpec("MXSA", True, mt, 1)
def _MXSB(mt):            return TensorSpec("MXSB", True, mt, 1)
def _M(side, tlu, mt):    return TensorSpec("Metadata", tlu, mt, 1, is_m=True, sparse_side=side)


# gl2-prefetch is emitted whenever PrefetchGL2 is set (KernelWriter guards
# gl2PrefetchCalcAddr on kernel["PrefetchGL2"] only, not ClusterDim). The
# cooperative fan-out only kicks in for a real cluster, so most configs run with
# ClusterDim != [1,1] to exercise it (one [1,1] case covers the degenerate path).
# ClusterDim = [cx, cy]: A/MXSA cooperate along cy and span cx macro-tiles; B/MXSB are the mirror. Shapes
# include power-of-2 and non-POT MacroTile / cluster extents
# (scalarStaticRemainder, ceil(gl2nl), ncc divide).
CONFIGS = [
    # ---- ClusterDim [1,1]: gl2-prefetch is still emitted (guard is PrefetchGL2,
    # not ClusterDim), but there is no cooperative fan-out -- numTileWGs ==
    # numShareWGs == 1, so every scalarStaticRemainder divides by 1 and the whole
    # footprint is covered by this single WG's threads. Guards the degenerate
    # single-workgroup path. ----
    GL2Config("ab_fp8_tlu_nocluster", [_A(True, 256), _B(True, 256)], cluster=(1, 1)),
    # ---- A + B together, FP8 TLU; MT=384 (non-POT) -> gl2ncc==2 ----
    GL2Config("ab_fp8_tlu",          [_A(True, 384),  _B(True, 384)],  cluster=(2, 2)),
    # ---- A + B non-TLU; MT=384 (non-POT) on perpendicular dim ----
    GL2Config("ab_fp8_ntlu",         [_A(False, 384), _B(False, 384)], cluster=(4, 4)),
    # batched=True also exercises the StridedBatched path (WorkGroup2 * Stride{tc}K
    # folded into the base addr): batch 0 reproduces the non-batched footprint, and
    # batch >0 verifies the per-batch shift. The grid gains a z extent (wg_z from
    # ttmp7[31:16]). FP8 (integer bpe) so the batch stride * bpe stays integral.
    GL2Config("ab_fp8_mixed_layout", [_A(True, 256),  _B(False, 256)], cluster=(2, 4),
              batched=True, num_batches=3),
    # gl2ncc == 2 for both (coal*bpe == 2*GPS)
    GL2Config("ab_fp8_ncc2",         [_A(True, 512),  _B(True, 512)], depth_u=128, cluster=(4, 2)),
    # ---- A + B with mixed dtypes (F8 x F4) ----
    GL2Config("ab_f8f4_mixed", [_A(True, 256, bpe=1), _B(True, 512, bpe=0.5)],
              depth_u=512, cluster=(1, 2)),
    # ---- A + B + MXSA + MXSB together (full MX problem) ----
    # batched=True here also covers the StridedBatched path for MX scales (Stride{MXSx}K).
    # ---- A + B + MXSA + MXSB together; MT=192 (non-POT) -> MX gl2ncc==3 ----
    GL2Config("abmx_fp8",      [_A(True, 192),  _B(True, 192),  _MXSA(192), _MXSB(192)],
              depth_u=256, mx_block=32, cluster=(2, 2), batched=True, num_batches=2),
    # ---- full MX problem, non-TLU data; MT=384 (non-POT) ----
    GL2Config("abmx_fp8_ntlu", [_A(False, 384), _B(False, 384), _MXSA(384), _MXSB(384)],
              depth_u=256, mx_block=32, cluster=(2, 1)),
    # ---- FP4 (bpe=0.5) on A and B, TLU, ncc==1 and (coal*bpe==2*GPS) ncc==2 ----
    GL2Config("ab_fp4_tlu",      [_A(True, 512, bpe=0.5),  _B(True, 512, bpe=0.5)],
              depth_u=256, cluster=(1, 4)),
    GL2Config("ab_fp4_tlu_ncc2", [_A(True, 1024, bpe=0.5), _B(True, 1024, bpe=0.5)],
              depth_u=128, cluster=(4, 1)),
    # ---- non-TLU: FP4 tile-split on A + FP8 coalesced-split ncc2 on B (coal==DepthU) ----
    GL2Config("ab_ntlu_f4f8", [_A(False, 256, bpe=0.5), _B(False, 128, bpe=1)],
              depth_u=512, cluster=(2, 2)),
    # ---- MX scales together: MXSA ncc==3, MXSB ncc==2 (non-POT MT 192 / 96) ----
    GL2Config("mxab_ncc", [_MXSA(192), _MXSB(96)], depth_u=1024, num_threads=16,
              mx_block=32, cluster=(2, 4)),
    # ---- Edge clamp: SizeI/SizeJ is NOT a clean multiple of the tiling, so the
    # last macro-tile is partial and the edge-limit clamp min(idx, Size-1) fires.
    # non-TLU clamps the perpendicular index; TLU/MX clamp the coalesced index.
    GL2Config("ab_ntlu_edge", [_A(False, 256), _B(False, 256)],
              cluster=(2, 2), size_i=384, size_j=384),
    GL2Config("ab_tlu_edge",  [_A(True, 512), _B(True, 512)],
              depth_u=128, cluster=(2, 2), size_i=700, size_j=700),
    GL2Config("mx_edge", [_MXSA(128), _MXSB(64)], depth_u=1024, num_threads=16,
              mx_block=32, cluster=(2, 2), size_i=150, size_j=80),
    # ---- gl2nl > 1: nc > cooperative threads (stride-add path); DU is MIK-aligned ----
    GL2Config("ab_tlu_nl2", [_A(True, 256), _B(True, 256)], depth_u=640, cluster=(2, 2)),
    # ---- gl2nl >> 1 with an uneven nc/nl: exercises the per-inst index stride
    # ncPerInst = ceil(nc/nl). A floor(nc/nl) stride under-tiles the top of the
    # footprint here (nc=1536, T=144, nl=11 -> floor stride 139 leaves the last
    # two cache lines uncovered; ceil stride 140 covers them). Needs a small
    # thread pool so nc/T is large; DU stays MIK-aligned (512 % 128 == 0). ----
    GL2Config("ab_tlu_nl_ceil", [_A(True, 256), _B(True, 256)], depth_u=512,
              num_threads=16, cluster=(3, 3)),
    # ---- non-POT cooperative cluster extent (scalarStaticRemainder non-POT path) ----
    GL2Config("ab_cluster_cy3", [_A(True, 256), _B(True, 256)], cluster=(2, 3)),
    # ---- non-POT cluster on a non-TLU layout: both cluster axes are non-POT, so
    # every scalarStaticRemainder (tile-selector and share) hits the non-POT path
    # for both A and B, while the MT offset/folded tile dim land on the perp dim ----
    GL2Config("ab_ntlu_cluster3", [_A(False, 256), _B(False, 256)], cluster=(3, 3)),
    # ---- Sparse metadata (isM) coverage. Sparse=1 -> A is the 2:4-compressed
    # data tensor (_DepthUA halved) and Metadata mirrors A's tile axis (idx=0);
    # Sparse=2 is the mirror on B. MetadataLayout is independent of the data
    # tensor's TLU (real kernels support both), so both are exercised. ----
    # ---- Sparse=1, data TLU, metadata non-TLU (MetadataLayout=0) ----
    GL2Config("a_sparse_tlu_mlayout0", [_A(True, 256), _B(True, 256), _M("A", False, 256)],
              cluster=(2, 2), sparse=1, depth_u_metadata=64),
    # ---- Sparse=1, data TLU, metadata also TLU (MetadataLayout=1) ----
    GL2Config("a_sparse_tlu_mlayout1", [_A(True, 256), _B(True, 256), _M("A", True, 256)],
              cluster=(2, 2), sparse=1, depth_u_metadata=64),
    # ---- Sparse=2 (mirror on B), non-TLU data; MT=384 (non-POT) on both data and
    # metadata -> exercises non-POT gl2ncc/scalarStaticRemainder for isM too ----
    GL2Config("b_sparse_ntlu_nonpot", [_A(False, 384), _B(False, 384), _M("B", True, 384)],
              cluster=(3, 3), sparse=2, depth_u_metadata=96),
    # ---- Sparse=1 + StridedBatched: exercises the WorkGroup2*Stride{tc}K batch
    # offset for Metadata too (AddressMetadata/StrideMetadataK) ----
    GL2Config("a_sparse_batched", [_A(True, 256), _B(True, 256), _M("A", False, 256)],
              cluster=(2, 2), sparse=1, depth_u_metadata=64, batched=True, num_batches=2),
    # ---- Sparse=1 + gl2nl > 1 for the metadata tensor (small thread pool, larger
    # DepthUMetadata) -> exercises the per-inst stride-add path on isM ----
    GL2Config("a_sparse_nl2", [_A(True, 256), _B(True, 256), _M("A", True, 256)],
              cluster=(2, 2), num_threads=16, sparse=1, depth_u_metadata=256),
    # ---- GlobalSplitU. Each group prefetches its own K chunk, so the start
    # address gains startIter*inc and the per-iteration step widens to the chunk
    # stride. The group index is the grid z axis and every group is verified
    # against its own chunk (never aggregated), so a group landing on the wrong
    # slice of K fails. GSU rides alongside the cluster, so the cooperative
    # fan-out is exercised at the same time. DepthU stays a power of 2 and the
    # thread count a whole number of waves (see the asserts in build_kernel). ----
    # gsu=1: the identity case. The GSU paths are emitted but there is one group,
    # so startIter==0 and the stride is unscaled -- guards the GSU codegen against
    # perturbing a single-group launch.
    GL2Config("gsu1_tlu", [_A(True, 256), _B(True, 256)], cluster=(2, 2), gsu=1),
    # Interleaved chunks (GSUC=0): group g starts at iteration g and every
    # increment steps a whole GSU round. Mixed TLU/non-TLU so the chunk offset is
    # checked against both K-axis layouts (K perpendicular vs K coalesced).
    GL2Config("gsu4_interleaved", [_A(True, 256), _B(False, 256)], cluster=(2, 2),
              gsu=4, k_iters=10),
    # Contiguous chunks (GSUC=1) with an uneven split: 10 iterations over 4 groups
    # is q=2 r=2, so groups 0/1 own 3 iterations and start at 0/3 while groups 2/3
    # own 2 and start at 6/8. Exercises the (q+1)*g vs q*g+r select.
    GL2Config("gsu4_contiguous_rem", [_A(True, 256), _B(False, 256)], cluster=(2, 2),
              gsu=4, gsuc=True, k_iters=10),
    # Non-POT group count with remainder 1 (10 = 3*3 + 1): only group 0 gets the
    # extra iteration, so the select flips for exactly one group. Non-POT MT too.
    GL2Config("gsu3_contiguous_rem", [_A(True, 384), _B(True, 384)], cluster=(2, 1),
              gsu=3, gsuc=True, k_iters=10),
    # Exact split (no remainder, 9 = 3*3) on a non-POT group count: every group
    # gets q iterations and the select must never take the (q+1) side.
    GL2Config("gsu3_contiguous_exact", [_A(False, 256), _B(False, 256)], cluster=(1, 3),
              gsu=3, gsuc=True, k_iters=9),
    # MX scales under GSU: the chunk offset is startIter * that tensor's own
    # increment, so MXSA/MXSB must shift by SizeFree*(DepthU/MXBlock) per iteration
    # while A/B shift by their (much larger) stride. A shared shift would fail here.
    GL2Config("gsu2_mx", [_A(True, 192), _B(True, 192), _MXSA(192), _MXSB(192)],
              depth_u=256, mx_block=32, cluster=(2, 2), gsu=2, gsuc=True, k_iters=7),
    # Sparse metadata under GSU: _DepthUA is halved and _DepthUMetadata differs
    # from DepthU, so each of the three tensors needs its own chunk offset even
    # though they all share one start iteration.
    GL2Config("gsu2_sparse", [_A(True, 256), _B(True, 256), _M("A", False, 256)],
              cluster=(2, 2), sparse=1, depth_u_metadata=64, gsu=2, k_iters=6),
    # Edge clamp + GSU: the K-direction chunk shift must stay orthogonal to the
    # free-dim clamp (it translates the clamped footprint, it does not re-clamp).
    GL2Config("gsu2_ntlu_edge", [_A(False, 256), _B(False, 256)], cluster=(2, 2),
              size_i=384, size_j=384, gsu=2, gsuc=True, k_iters=5),
    # GSU without a cluster: the degenerate single-workgroup fan-out combined with
    # a 4-way K split, so the chunk offset is the only thing distinguishing the wgs.
    GL2Config("gsu4_nocluster", [_A(True, 256), _B(True, 256)], cluster=(1, 1),
              gsu=4, k_iters=12),

    # ---- StaggerU. The prefetch has to follow the same rotated K order as the
    # real load stream, which splits into a start shifted by StaggerUIter and a
    # one-off wrap back to iteration 0 partway through. Each config below picks
    # its numIter and rotation so the wrap falls on a stage the kernel actually
    # exports, since a prefetch that never rolls over is indistinguishable from
    # a plain shift. Stage s prefetches (StaggerUIter + PGR + s) % numIter, so
    # the wrap lands on stage numIter - StaggerUIter - PGR. ----
    # StaggerU off at runtime (StaggerUIter==0) with the rotation code emitted
    # anyway -- what every non-cluster GL2 kernel now builds, and the case where
    # a stray rotation or an early wrap would be pure regression.
    GL2Config("su_off", [_A(True, 256), _B(True, 256)], cluster=(2, 2), stagger=0),
    # Plain rotation, wrap on stage 3 of 0..4 (8 - 3 - 2), so stages before and
    # after the roll-over are both checked. Mixed layouts: the wrap is one step
    # of each tensor's own increment, not a shared byte count.
    GL2Config("su_wrap", [_A(True, 256), _B(False, 256)], cluster=(2, 2),
              stagger=3, k_iters=8, n_inc=4),
    # Wrap on stage 1, which is the prologue increment rather than an in-loop
    # one: it runs before the loop counter starts stepping and is not guarded by
    # the end-of-K freeze, so it compares against a different counter value.
    GL2Config("su_wrap_prologue", [_A(True, 384), _B(True, 384)], cluster=(2, 1),
              stagger=3, k_iters=6, n_inc=3),
    # Rotation running into the end-of-K freeze: numIter 6 wraps on stage 3 and
    # freezes from stage 4 (counter <= PGR+PGL2), so the last stages must all sit
    # on the final iteration instead of walking off the end of the chunk.
    GL2Config("su_freeze", [_A(False, 256), _B(False, 256)], cluster=(2, 2),
              stagger=1, k_iters=6, n_inc=5),
    # PGR 3: calculateStagger shifts StaggerUIter by PGR instead of 2, and the
    # start pre-skip grows to match, so both halves have to move together.
    GL2Config("su_pgr3", [_A(True, 256), _B(True, 256)], cluster=(2, 2),
              pgr=3, stagger=2, k_iters=8, n_inc=4),
    # PGR 3 with StaggerU off: from PGR 3 up, calculateStagger skips the rewrite
    # when the rotation is 0, which parks the wrap-target at a counter value the
    # freeze has to swallow. Getting this wrong wraps a non-staggered kernel.
    GL2Config("su_pgr3_off", [_A(True, 256), _B(False, 256)], cluster=(2, 2),
              pgr=3, stagger=0, k_iters=8, n_inc=4),
    # ---- PGR 1. calculateStagger still shifts StaggerUIter by 2, so pf no longer
    # equals the PGR the prefetch stream is actually offset by and the roll-over
    # sits one counter step off the plain PrefetchGL2 lead (gl2StaggerWrapOffset).
    # Driving the offset straight off PrefetchGL2, as PGR>=2 can, rolls these over
    # one stage early. ----
    # Wrap on stage 3 of 0..4 (8 - 4 - 1), an in-loop increment with stages either
    # side of it.
    GL2Config("su_pgr1", [_A(True, 256), _B(False, 256)], cluster=(2, 2),
              pgr=1, stagger=4, k_iters=8, n_inc=4),
    # Wrap on stage 1, the prologue increment: its offset is PGR+1-pf, which is 0
    # at PGR 1 rather than the 1 that PGR>=2 uses.
    GL2Config("su_pgr1_prologue", [_A(True, 384), _B(True, 384)], cluster=(2, 1),
              pgr=1, stagger=4, k_iters=6, n_inc=3),
    # Rotation running into the end-of-K freeze at PGR 1: wrap on stage 4, freeze
    # from stage 5, so the roll-over and the clamp are both exercised.
    GL2Config("su_pgr1_freeze", [_A(False, 256), _B(False, 256)], cluster=(2, 2),
              pgr=1, stagger=1, k_iters=6, n_inc=5),
    # PGR 1 with StaggerU off. The offset parks the wrap-target exactly on the
    # freeze boundary (counter PGR+PGL2), so the stages run right onto it and the
    # freeze has to swallow the roll-over -- otherwise an unrotated stream jumps
    # back to the start of K. n_inc reaches that counter, which the shorter
    # StaggerU-off configs above do not.
    GL2Config("su_pgr1_off", [_A(True, 256), _B(False, 256)], cluster=(2, 2),
              pgr=1, stagger=0, k_iters=8, n_inc=7),

    # MX scales and sparse metadata under rotation: every tensor rotates by the
    # same iteration count but wraps by its own increment, so a wrap value shared
    # across tensors fails here even though the unrotated stages pass.
    GL2Config("su_mx", [_A(True, 192), _B(True, 192), _MXSA(192), _MXSB(192)],
              depth_u=256, mx_block=32, cluster=(2, 2), stagger=2, k_iters=7, n_inc=4),
    GL2Config("su_sparse", [_A(True, 256), _B(True, 256), _M("A", False, 256)],
              cluster=(2, 2), sparse=1, depth_u_metadata=64, stagger=2, k_iters=7, n_inc=4),
    # Edge clamp + rotation: the K rotation translates the clamped free-dim
    # footprint, it does not re-clamp it.
    GL2Config("su_ntlu_edge", [_A(False, 256), _B(False, 256)], cluster=(2, 2),
              size_i=384, size_j=384, stagger=2, k_iters=7, n_inc=4),

    # ---- StaggerU x GlobalSplitU. The rotation is per group, wrapping inside
    # that group's chunk, so numIter (and with it the wrap stage) differs group
    # to group. Both chunk layouts are covered because the rotation composes
    # with the chunk stride, which is a GSU round when interleaved and a single
    # iteration when contiguous. ----
    # Uneven interleaved split, 13 over 2 groups: group 0 owns 7 iterations and
    # wraps on stage 3, group 1 owns 6 and wraps on stage 2. A wrap driven off
    # the total instead of the group's own count fails on one of the two.
    GL2Config("su_gsu2_interleaved_rem", [_A(True, 256), _B(False, 256)], cluster=(2, 2),
              gsu=2, stagger=2, k_iters=13, n_inc=4),
    # Contiguous chunks: the rotation rides on top of the group's start
    # iteration, so start and rotation have to compose rather than replace.
    GL2Config("su_gsu3_contiguous", [_A(True, 256), _B(True, 256)], cluster=(2, 1),
              gsu=3, gsuc=True, stagger=2, k_iters=15, n_inc=4),
    # Single group: the GSU and StaggerU paths are both emitted but the chunk is
    # the whole loop, pinning down that neither perturbs the other.
    GL2Config("su_gsu1", [_A(True, 256), _B(True, 256)], cluster=(2, 2),
              gsu=1, stagger=2, k_iters=8, n_inc=4),
]


def batch_stride_elems(spec, cfg):
    """Per-tensor batch stride in *elements* (the programmed Stride{tc}K). Arbitrary
    but fixed and distinct per tensor so the verifier can reproduce the
    WorkGroup2 * Stride{tc}K * bpe shift and so a cross-tensor stride mixup fails.
    Chosen even so that stride * bpe is integral for fractional bpe (FP4)."""
    return {"A": 1_000_002, "B": 2_000_006, "MXSA": 3_000_010, "MXSB": 4_000_014,
            "Metadata": 5_000_018}[spec.tc]

# ---------------------------------------------------------------------------
# Kernel + writer construction
# ---------------------------------------------------------------------------

def _subtc_attr(cfg, sub, attr, default):
    for t in cfg.tensors:
        if t.subtc == sub and not t.is_m:
            return getattr(t, attr)
    return default


def _make_kernel(cfg):
    has_mxa = any(t.tc == "MXSA" for t in cfg.tensors)
    has_mxb = any(t.tc == "MXSB" for t in cfg.tensors)
    m_spec = next((t for t in cfg.tensors if t.is_m), None)
    kernel = {
        "ProblemType": {
            "Batched": cfg.batched,
            "StridedBatched": cfg.batched,
            "IndicesBatch": [2] if cfg.batched else [],
            "IndicesFree": [0, 1],
            "IndicesSummation": [3],
            "IndexAssignmentsA": [0, 3, 2] if cfg.batched else [0, 3],
            "IndexAssignmentsB": [1, 3, 2] if cfg.batched else [1, 3],
            "UseInitialStridesAB": True,
            "MXBlockA": cfg.mx_block if has_mxa else 0,
            "MXBlockB": cfg.mx_block if has_mxb else 0,
            "TLUA": _subtc_attr(cfg, "A", "tlu", True),
            "TLUB": _subtc_attr(cfg, "B", "tlu", True),
            "Sparse": cfg.sparse,
        },
        "MacroTileA": _subtc_attr(cfg, "A", "mt", 256),
        "MacroTileB": _subtc_attr(cfg, "B", "mt", 256),
        # _DepthU{A,B}: per-tensor unroll extent (== DepthU for dense; GL2Prefetch
        # reads these instead of the plain DepthU so it matches the sparse-halved
        # layout when a data tensor is the compressed (2:4) sparse operand).
        "_DepthUA": depth_u_side("A", cfg),
        "_DepthUB": depth_u_side("B", cfg),
        "MatrixInstK": cfg.matrix_inst_k,
        "ClusterDim": list(cfg.cluster),
        "NumThreads": cfg.num_threads,
        "DepthU": cfg.depth_u,
        "PrefetchGlobalRead": cfg.pgr,
        "WavefrontSize": WAVESIZE,
        "PrefetchGL2": cfg.pgl,
        "GlobalSplitU": cfg.gsu,
    }
    if m_spec is not None:
        kernel["MacroTileMetadata"] = m_spec.mt
        kernel["_DepthUMetadata"] = cfg.depth_u_metadata
    return kernel


def _make_writer(kernel):
    """Mock writer that binds the real KernelWriterAssembly methods GL2Prefetch needs."""
    from Tensile.Common import INDEX_CHARS
    from Tensile.KernelWriterAssembly import KernelWriterAssembly as KWA
    from rocisa.label import LabelManager
    from rocisa.register import RegisterPool
    from rocisa.enum import RegisterType

    w = SimpleNamespace()
    w.vgprPool = RegisterPool(0, RegisterType.Vgpr, defaultPreventOverflow=False, printRP=False)
    w.sgprPool = RegisterPool(0, RegisterType.Sgpr, defaultPreventOverflow=False, printRP=False)
    w.sgprs = {}
    w.labels = LabelManager()
    w.db = {"AssertOnSgprOverflow": False}
    w.states = SimpleNamespace(
        kernel=kernel,
        indexChars=INDEX_CHARS,
        regCaps={"MaxSgpr": 106, "MaxVgpr": 1024, "PhysicalMaxVgpr": 1024,
                 "GlobalPrefetchSize": GLOBAL_PREFETCH_SIZE},
        asmCaps={"HasSMulHi": True, "HasGlobalPrefetch": True},
        unrollIdx=0,
        overflowedResources=0,
        a=SimpleNamespace(), b=SimpleNamespace(),
    )
    # gsuMaskHex/calculateLoopNumIterOffsetGsu back the contiguous-chunk branch of
    # calculateGSUIterOffset; loopChar/loopCounterName/loopCounter resolve the
    # LoopCounterL the StaggerU rotation reads. All come from the real writer so
    # the test cannot drift from production's arithmetic.
    for m in ["strideRef", "allocTmpSgpr", "s_mul_u64_u32", "s_mul_i64_i32",
              "gsuMaskHex", "calculateLoopNumIterOffsetGsu",
              "loopChar", "loopCounterName", "loopCounter"]:
        setattr(w, m, types.MethodType(getattr(KWA, m), w))
    w.sgprPool.checkOut(6)  # reserve hardware sgprs (s0:1 kernarg ptr, etc.)
    return w


# ---------------------------------------------------------------------------
# Kernel assembly generation
# ---------------------------------------------------------------------------

def build_kernel(cfg):
    """Build a kernel computing GL2 addresses for all of cfg.tensors at once.

    The kernel emits cfg.n_inc+1 "stages": stage 0 is the start address (which,
    because PGR>1, already includes the calculateStartAddr PGR*inc pre-skip),
    and each later stage calls incrementAddr once more, so stage s is shifted by
    (PGR + s) * inc from the base footprint. This exercises both addr-increment
    paths (the PGR pre-skip and the per-iteration incrementAddr).

    Returns (asm, layout, n_out) where layout is a list of
    (TensorSpec, num_loads, stage, region_start) describing the output partition.
    """
    from rocisa.code import Module, TextBlock
    from rocisa.container import sgpr, ContinuousRegister
    from rocisa.instruction import (SAddU32, SCMovB32, SCmpGtU32, SCmpLtU32, SCSelectB32,
                                    SMovB32, SSubBU32, SSubU32)
    from Tensile.KernelWriterAssembly import GL2PrefetchLoad
    from Tensile.KernelWriter import gl2StaggerWrapOffset, staggerPrefetchFactor

    init_rocisa(wavesize=WAVESIZE)
    kernel = _make_kernel(cfg)
    w = _make_writer(kernel)
    comp = GL2PrefetchLoad()

    if cfg.gsu_on:
        # The GSU group index rides the grid z axis, which the batch index also
        # uses. Not a limitation worth engineering around: the batch offset and the
        # GSU chunk offset are added to the same base accumulator, so they compose
        # additively and batching is already covered on its own.
        assert not cfg.batched, f"{cfg.name}: a GSU config cannot also be batched"
        # calculateLoopNumIterOffsetGsu's divide resets exec to all lanes, which is
        # only correct when every wave is full -- as it always is in production.
        assert cfg.num_threads % WAVESIZE == 0, \
            f"{cfg.name}: GSU needs full waves (num_threads % {WAVESIZE} == 0)"
        # the contiguous branch derives numIter with a shift, like computeLoadSrd
        assert cfg.depth_u & (cfg.depth_u - 1) == 0, \
            f"{cfg.name}: GSU needs a power-of-2 DepthU, got {cfg.depth_u}"

    if cfg.stagger_on:
        smallest = min(num_iters(cfg, g) for g in range(cfg.n_groups))
        # declareStaggerParms picks the rotation from a mask below the iteration
        # count, so a rotation that does not fit in the chunk cannot occur.
        assert cfg.stagger < smallest, \
            f"{cfg.name}: stagger {cfg.stagger} must be < numIter {smallest}"
        # Stage 1 goes through the prologue increment, which has no end-of-K
        # freeze (production branches around it instead), so the chunk has to be
        # long enough for position PGR+1 to be a real one.
        assert smallest >= cfg.pgr + 2, \
            f"{cfg.name}: stagger needs numIter >= PGR+2, got {smallest}"

    subtcs = {t.subtc for t in cfg.tensors}

    # ---- named sgprs (resolved via .set; values assigned in the prologue) ----
    w.sgprs["OutPtr"] = w.sgprPool.checkOutAligned(2, 2, "OutPtr", preventOverflow=False)
    shared = ["WorkGroup0", "WorkGroup1", "WorkGroup2"]
    if cfg.n_regions > 1:
        shared += ["WGOUT"]   # per-region output shift = linear_wg_id * n_out
    if "A" in subtcs:
        shared += ["StrideAI", "StrideAL", "SizeI"]
    if "B" in subtcs:
        shared += ["StrideBJ", "StrideBL", "SizeJ"]
    for t in cfg.tensors:
        if t.is_m:                        # StrideMetadata{I,J} + StrideMetadataL
            idxChar = "I" if t.idx == 0 else "J"
            shared += [f"StrideMetadata{idxChar}", "StrideMetadataL"]
    if cfg.gsu_on:
        # GSU packs the group count and the GSUC bit; SizesSum feeds numIter in the
        # contiguous branch.
        shared += ["GSU", "SizesSum"]
    if cfg.stagger_on:
        # StaggerUIter is the rotation; LoopCounterL is this group's numIter, and
        # doubles as the clock the wrap and freeze compares run off (the epilogue
        # reprograms it per stage to the value the unroll loop would hold there).
        # NumIterL is harness-only: it keeps numIter live for that reprogramming.
        shared += ["StaggerUIter", "LoopCounterL", "NumIterL"]
    for n in shared:
        w.sgprs[n] = w.sgprPool.checkOut(1, n, preventOverflow=False)
    if cfg.stagger_on:
        # WrapU{tc} is calculateStagger's roll-over distance, which the prefetch
        # reuses rather than deriving its own. Production computes it after the
        # rotation phase; the prologue here programs it directly.
        for t in cfg.tensors:
            w.sgprs[f"WrapU{t.tc}"] = w.sgprPool.checkOutAligned(
                2, 2, f"WrapU{t.tc}", preventOverflow=False)
    if cfg.gsu_on:
        # 2 registers: calculateLoopNumIterOffsetGsu uses GSUSumIdx+1 as the
        # divide's remainder scratch. Allocated before the Address{tc} pairs so the
        # .set-based sgpr count (which only sees the base index) still covers +1.
        w.sgprs["GSUSumIdx"] = w.sgprPool.checkOut(2, "GSUSumIdx", preventOverflow=False)
    for t in cfg.tensors:
        w.sgprs[f"Address{t.tc}"] = w.sgprPool.checkOutAligned(2, 2, f"Address{t.tc}", preventOverflow=False)
        w.sgprs[f"GL2PrefetchInc{t.tc}"] = w.sgprPool.checkOut(1, f"GL2PrefetchInc{t.tc}", preventOverflow=False)
        if cfg.batched:    # batch stride Stride{tc}K (index 2 -> 'K')
            w.sgprs[f"Stride{t.tc}K"] = w.sgprPool.checkOut(1, f"Stride{t.tc}K", preventOverflow=False)

    w.vgprPool.checkOut(1)  # v0 = Serial (workitem id)

    # ---- init each tp, allocate its per-load address vgprs ----
    tps = []
    vgpr_sets = {}
    for t in cfg.tensors:
        ia = t.ia + [2] if cfg.batched else t.ia   # batch index 2 must be in ia
        tp = {"tensorChar": t.tc, "idx": t.idx, "tlu": t.tlu, "bpeGR": t.bpe, "ia": ia, "isM": t.is_m}
        comp.init(w, kernel, tp)
        assert tp["gl2nc"] == tensor_dims(t, cfg)[3], \
            f"{t.tc}: gl2nc {tp['gl2nc']} != expected {tensor_dims(t, cfg)[3]}"
        assert tp["gl2nl"] == tensor_gl2nl(t, cfg), \
            f"{t.tc}: gl2nl {tp['gl2nl']} != expected {tensor_gl2nl(t, cfg)}"
        for i in range(tp["gl2nl"]):
            name = f"GL2PrefetchAddr{t.tc}_{i}"
            vgpr_sets[name] = w.vgprPool.checkOutAligned(2, 2, name, preventOverflow=False)
        tps.append((t, tp))

    # output elements written by one workgroup (used to shift per-wg regions);
    # each of the n_stages re-exports every tensor's loads.
    n_stages = cfg.n_inc + 1
    n_out_per_wg = n_stages * sum(cfg.num_threads * tp["gl2nl"] for _, tp in tps)

    # ---- body: setIncrement (all), then calculateStartAddr (each).
    # calculateStartAddr folds in the base Address{tc}, the GSU chunk offset and
    # the PGR pre-skip itself (SGPR-accumulated), so there is no separate
    # gsuOffset step. Under GSU this mirrors production gl2PrefetchCalcAddr: the
    # chunk start iteration is tensor independent, so it is derived once and each
    # tensor scales it by its own per-iteration increment. ----
    body = Module("body")
    if cfg.gsu_on:
        with w.allocTmpSgpr(3, tag="gl2_gsu") as tmpSgprRes:
            gsu_iter_sgpr = tmpSgprRes.idx
            body.add(comp.calculateGSUIterOffset(
                w, kernel, gsu_iter_sgpr,
                ContinuousRegister(idx=tmpSgprRes.idx + 1, size=2)))
            for t, tp in tps:
                body.add(comp.setIncrement(w, kernel, tp))
                body.add(comp.calculateStartAddr(w, kernel, tp, gsu_iter_sgpr))
    else:
        for t, tp in tps:
            body.add(comp.setIncrement(w, kernel, tp))
            body.add(comp.calculateStartAddr(w, kernel, tp))

    # ---- StaggerU rotation, in production's order: gl2PrefetchApplyStagger runs
    # after declareStaggerParms (StaggerUIter still the plain rotation amount)
    # and before calculateStagger rewrites StaggerUIter into the loop-counter
    # value the wrap compares against. The rewrite is replayed here so the two
    # halves are exercised against the same register the kernel would see. ----
    if cfg.stagger_on:
        with w.allocTmpSgpr(4, 2, tag="gl2_stagger") as tmpSgprRes:
            body.add(comp.staggerStartIterDelta(w, kernel, tmpSgprRes.idx, tmpSgprRes.idx + 1))
            for t, tp in tps:
                body.add(comp.applyStaggerStart(w, kernel, tp, tmpSgprRes.idx, tmpSgprRes.idx + 2))
        # WrapU{tc}, as calculateStagger would leave it. Production derives it
        # from GlobalReadIncs{tc}; there is no real load stream here, so it comes
        # off GL2PrefetchInc{tc} instead -- which is the premise that lets the
        # prefetch share the register in the first place, and is why this is
        # spelled out rather than called into the component.
        for t, tp in tps:
            wrap, inc = f"WrapU{t.tc}", f"GL2PrefetchInc{t.tc}"
            body.addModuleAsFlatItems(w.s_mul_i64_i32(
                sgpr(f"{wrap}+0"), sgpr(f"{wrap}+1"),
                sgpr("LoopCounterL"), sgpr(inc), "bytes accessed by the unroll loop"))
            body.add(SSubU32(dst=sgpr(f"{wrap}+0"), src0=sgpr(inc), src1=sgpr(f"{wrap}+0"),
                             comment="remove one iteration"))
            body.add(SSubBU32(dst=sgpr(f"{wrap}+1"), src0=0, src1=sgpr(f"{wrap}+1"),
                              comment="remove one iteration"))
        # calculateStagger converts the rotation S' into the loop-counter value S
        # the real stream's wrap compares against, S = S' + pf. From PGR 3 up the
        # rewrite is skipped when StaggerU is off at runtime, which moves where a
        # StaggerUIter of 0 wraps; below that it is unconditional.
        pf = staggerPrefetchFactor(kernel)
        if cfg.pgr >= 3:
            with w.allocTmpSgpr(1, tag="gl2_stagger_pf") as t:
                body.add(SAddU32(dst=sgpr(t.idx), src0=sgpr("StaggerUIter"), src1=pf))
                body.add(SCmpGtU32(src0=sgpr("StaggerUIter"), src1=0, comment="StaggerU > 0?"))
                body.add(SCMovB32(dst=sgpr("StaggerUIter"), src=sgpr(t.idx),
                                  comment="calculateStagger: StaggerUIter -> wrap-target iteration"))
        else:
            body.add(SAddU32(dst=sgpr("StaggerUIter"), src0=sgpr("StaggerUIter"), src1=pf,
                             comment="calculateStagger: StaggerUIter -> wrap-target iteration"))

    # ---- prologue ----
    prologue = Module("prologue")
    for t in cfg.tensors:                       # every tensor shares the same dummy base buffer
        ab = w.sgprs[f"Address{t.tc}"]
        prologue.add(TextBlock("  s_load_b64 s[%d:%d], s[0:1], 0x0\n" % (ab, ab + 1)))
    prologue.add(TextBlock("  s_load_b64 s[%d:%d], s[0:1], 0x8\n"
                           % (w.sgprs["OutPtr"], w.sgprs["OutPtr"] + 1)))
    prologue.add(TextBlock("  s_wait_kmcnt 0x0\n"))

    consts = []
    if "A" in subtcs:
        coal_a = _data_coal(cfg, "A")
        # free-dim size spans all MT-selector tiles (clean tiling), unless an
        # explicit size_i makes the last tile partial to exercise the edge clamp.
        consts += [("StrideAI", coal_a), ("StrideAL", coal_a),
                   ("SizeI", free_dim_size(cfg, "A"))]
    if "B" in subtcs:
        coal_b = _data_coal(cfg, "B")
        consts += [("StrideBJ", coal_b), ("StrideBL", coal_b),
                   ("SizeJ", free_dim_size(cfg, "B"))]
    for t in cfg.tensors:
        if t.is_m:
            # StrideMetadata{I,J}/StrideMetadataL are both programmed to the
            # metadata's own (folded) coalesced extent, mirroring how StrideAI==
            # StrideAL / StrideBJ==StrideBL are set for a plain data tensor.
            coal_m = tensor_dims(t, cfg)[0]
            idxChar = "I" if t.idx == 0 else "J"
            consts += [(f"StrideMetadata{idxChar}", coal_m), ("StrideMetadataL", coal_m)]
    consts += [("WorkGroup0", 0), ("WorkGroup1", 0), ("WorkGroup2", 0)]
    if cfg.batched:                              # programmed batch stride Stride{tc}K
        consts += [(f"Stride{t.tc}K", batch_stride_elems(t, cfg)) for t in cfg.tensors]
    if cfg.gsu_on:
        # packed GSU kernel argument: group count in the low bits, GSUC in bit 15
        consts += [("GSU", cfg.n_groups | (0x8000 if cfg.gsuc else 0)),
                   ("SizesSum", cfg.k_iters * cfg.depth_u)]
    for n, v in consts:
        prologue.add(SMovB32(dst=sgpr(n), src=v))

    # gfx1250 carries the workgroup id in ttmp (not s2): wg_x in ttmp9, wg_y in
    # ttmp7[15:0], wg_z in ttmp7[31:16] (matching the production non-cluster
    # decode). The cooperative cluster drives WorkGroup0/1; the z axis drives
    # WorkGroup2. Each region's linear id is wg_z*(cx*cy) + wg_y*cx + wg_x.
    cx, cy = cfg.cluster
    if cfg.n_wg > 1:
        prologue.add(TextBlock("  s_mov_b32 s%d, ttmp9\n" % w.sgprs["WorkGroup0"]))
        prologue.add(TextBlock("  s_and_b32 s%d, 0xFFFF, ttmp7\n" % w.sgprs["WorkGroup1"]))
    if cfg.grid_z > 1:
        prologue.add(TextBlock("  s_lshr_b32 s%d, ttmp7, 16\n" % w.sgprs["WorkGroup2"]))
    if cfg.gsu_on:
        # NB: production does not get the group index from z. GSUOn.graWorkGroup
        # launches cy*GSU workgroups along *y* and splits them, WorkGroup1 = wg_y /
        # GSU and GSUSumIdx = wg_y % GSU (or the GSUWGMRR round-robin variant). That
        # split happens before gl2PrefetchCalcAddr, so by the time the prefetch code
        # runs, GSUSumIdx and the already-divided WorkGroup1 are simply inputs to it
        # -- the prefetch never participates in the derivation.
        # Driving the group index off z instead enumerates exactly the same
        # (WorkGroup0, WorkGroup1, GSUSumIdx) tuples, without reimplementing the
        # divide in the harness and without tying the group index to the tile index.
        # WorkGroup2 holds the raw wg_z and is otherwise unused here (a GSU config is
        # never batched, so no Stride{tc}K is programmed).
        prologue.add(SMovB32(dst=sgpr("GSUSumIdx"), src=sgpr("WorkGroup2")))
    if cfg.stagger_on:
        prologue.add(SMovB32(dst=sgpr("StaggerUIter"), src=cfg.stagger))
        # LoopCounterL = this group's numIter. calculateLoopNumIterGsu derives it
        # in production; here the group index is already in hand, so pick the
        # longer run directly for the first (k_iters % groups) groups.
        # NumIterL keeps numIter around after the epilogue starts overwriting
        # LoopCounterL to walk the stages.
        q, r = divmod(cfg.k_iters, cfg.n_groups)
        prologue.add(SMovB32(dst=sgpr("NumIterL"), src=q if cfg.gsu_on else cfg.k_iters))
        if cfg.gsu_on and r:
            prologue.add(SCmpLtU32(src0=sgpr("GSUSumIdx"), src1=r, comment="group gets a longer run?"))
            prologue.add(SCSelectB32(dst=sgpr("NumIterL"), src0=q + 1, src1=q,
                                     comment="numIter of this group's chunk"))
        prologue.add(SMovB32(dst=sgpr("LoopCounterL"), src=sgpr("NumIterL")))
    if cfg.n_regions > 1:
        # WGOUT = (wg_z*cy + wg_y)*cx + wg_x, then * n_out_per_wg. WorkGroup2 is 0
        # when not batched and WorkGroup0/1 are 0 without a cluster, so this one
        # chain covers cluster-only, batch-only, and combined launches.
        wgout = w.sgprs["WGOUT"]
        prologue.add(TextBlock("  s_mul_i32 s%d, s%d, %d\n" % (wgout, w.sgprs["WorkGroup2"], cy)))
        prologue.add(TextBlock("  s_add_u32 s%d, s%d, s%d\n" % (wgout, wgout, w.sgprs["WorkGroup1"])))
        prologue.add(TextBlock("  s_mul_i32 s%d, s%d, %d\n" % (wgout, wgout, cx)))
        prologue.add(TextBlock("  s_add_u32 s%d, s%d, s%d\n" % (wgout, wgout, w.sgprs["WorkGroup0"])))
        prologue.add(TextBlock("  s_mul_i32 s%d, s%d, %d\n" % (wgout, wgout, n_out_per_wg)))

    # ---- epilogue: for each stage, export (addr - base) per tensor into its own
    # output region, then incrementAddr to advance to the next stage. ----
    epi = Module("epi")
    off = w.vgprPool.checkOut(1, "off", preventOverflow=False)
    a_lo = w.vgprPool.checkOutAligned(2, 2, "outaddr", preventOverflow=False)
    a_hi = a_lo + 1
    val = w.vgprPool.checkOut(1, "val", preventOverflow=False)

    def export_tensor(t, tp, region):
        num_loads = tp["gl2nl"]
        base = w.sgprs[f"Address{t.tc}"]
        k = 0
        for i in range(tp["gl2nl"]):
            addr = vgpr_sets[f"GL2PrefetchAddr{t.tc}_{i}"]
            epi.add(TextBlock("  v_sub_co_u32 v%d, vcc_lo, v%d, s%d\n" % (val, addr, base)))
            # output element index = region + Serial*num_loads + k
            if num_loads == 1:
                epi.add(TextBlock("  v_add_nc_u32 v%d, %d, v0\n" % (off, region + k)))
            else:
                epi.add(TextBlock("  v_mul_u32_u24 v%d, v0, %d\n" % (off, num_loads)))
                epi.add(TextBlock("  v_add_nc_u32 v%d, %d, v%d\n" % (off, region + k, off)))
            if cfg.n_regions > 1:          # shift this region's results into its own slice
                epi.add(TextBlock("  v_add_nc_u32 v%d, s%d, v%d\n"
                                  % (off, w.sgprs["WGOUT"], off)))
            epi.add(TextBlock("  v_lshlrev_b32 v%d, 2, v%d\n" % (off, off)))
            epi.add(TextBlock("  v_add_co_u32 v%d, vcc_lo, s%d, v%d\n"
                              % (a_lo, w.sgprs["OutPtr"], off)))
            epi.add(TextBlock("  v_mov_b32 v%d, s%d\n" % (a_hi, w.sgprs["OutPtr"] + 1)))
            epi.add(TextBlock("  v_add_co_ci_u32 v%d, vcc_lo, v%d, 0, vcc_lo\n" % (a_hi, a_hi)))
            epi.add(TextBlock("  flat_store_b32 v[%d:%d], v%d\n" % (a_lo, a_hi, val)))
            k += 1

    def set_loop_counter(stage):
        """Drive LoopCounterL to the value the unroll loop would hold at `stage`.

        The counter is what the wrap and freeze compares run off, so the stages
        only mean anything if it steps the way production's does. Anchoring on
        the real stream: it wraps at LoopCounterL == StaggerUIter, i.e. when its
        own next position would be numIter, and the prefetch sits PrefetchGL2
        positions ahead of it. That fixes the counter at prefetch position j to
        numIter + PGR + PGL2 - j, and stage s prefetches position PGR + s.
        Stage 1 is the odd one out: it is the prologue increment, which runs
        before the loop starts stepping the counter at all, so it still sees the
        initial numIter."""
        delta = 0 if stage <= 1 else cfg.pgl - stage
        if delta == 0:
            epi.add(SMovB32(dst=sgpr("LoopCounterL"), src=sgpr("NumIterL")))
        elif delta > 0:
            epi.add(SAddU32(dst=sgpr("LoopCounterL"), src0=sgpr("NumIterL"), src1=delta))
        else:
            epi.add(SSubU32(dst=sgpr("LoopCounterL"), src0=sgpr("NumIterL"), src1=-delta))

    layout = []
    region = 0
    for stage in range(n_stages):
        if stage > 0:                          # advance every tensor by one inc
            tpList = [tp for _, tp in tps]
            if not cfg.stagger_on:
                epi.add(comp.incrementAddr(w, kernel, tpList))
            else:
                set_loop_counter(stage)
                # Stage 1 replays the PrefetchGL2==2 prologue increment (rolls
                # over one counter step later than the loop's, and is guarded by
                # a branch rather than the freeze); later stages replay the
                # in-loop one.
                prologueStep = (stage == 1)
                steps = 1 if prologueStep else cfg.pgl
                epi.add(comp.incrementAddr(
                    w, kernel, tpList,
                    staggerWrapOffset=gl2StaggerWrapOffset(kernel, steps),
                    freezeIter=None if prologueStep else cfg.pgr + cfg.pgl))
        for t, tp in tps:
            num_loads = tp["gl2nl"]
            export_tensor(t, tp, region)
            layout.append((t, num_loads, stage, region))
            region += cfg.num_threads * num_loads
    epi.add(TextBlock("  s_wait_storecnt 0x0\n"))
    n_out = region

    inner = "\n".join([str(prologue), str(body), str(epi)])

    set_lines = [".set vgprSerial, 0"]
    set_lines += [".set sgpr%s, %d" % (n, i) for n, i in w.sgprs.items()]
    set_lines += [".set vgpr%s, %d" % (n, i) for n, i in vgpr_sets.items()]
    set_dir = "\n".join(set_lines)

    text = inner + "\n" + set_dir
    vgprs, _, sgprs = _scan_register_indices(text)
    max_v = max((((max(vgprs | {0}) + 1) + 3) // 4) * 4, 4)
    max_s = max(sgprs | {0}) + 1

    asm = f"""\
.amdgcn_target "amdgcn-amd-amdhsa--{GFX_TARGET}"
{set_dir}
.text
.protected test_kernel
.globl test_kernel
.p2align 8
.type test_kernel,@function
.section .rodata,#alloc
.p2align 6
.amdhsa_kernel test_kernel
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_next_free_vgpr {max_v}
  .amdhsa_next_free_sgpr {max_s}
  .amdhsa_group_segment_fixed_size 0
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 1
  .amdhsa_system_sgpr_workgroup_id_z 1
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_wavefront_size32 1
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel
.text
test_kernel:
{inner}
  s_endpgm
.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: test_kernel
    .symbol: 'test_kernel.kd'
    .kernarg_segment_size: 16
    .kernarg_segment_align: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: {WAVESIZE}
    .sgpr_count: {max_s}
    .vgpr_count: {max_v}
    .max_flat_workgroup_size: {cfg.num_threads}
    .args:
      - {{.name: addrT,   .size: 8, .offset: 0,  .value_kind: global_buffer, .address_space: global, .value_type: u8}}
      - {{.name: outptr,  .size: 8, .offset: 8,  .value_kind: global_buffer, .address_space: global, .value_type: u32}}
...
.end_amdgpu_metadata
"""
    return asm, layout, n_out


def _data_coal(cfg, sub):
    """Leading (coalesced) extent of the data tensor for subtc, used as the
    contiguous stride. MX tensors derive their stride in-kernel, so any value
    works there; fall back to whatever tensor is present. Excludes Metadata:
    the data tensor's stride is programmed independently of StrideMetadataL."""
    for t in cfg.tensors:
        if t.subtc == sub and not t.is_mx and not t.is_m:
            return tensor_dims(t, cfg)[0]
    for t in cfg.tensors:
        if t.subtc == sub:
            return tensor_dims(t, cfg)[0]
    return 1


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def inc_bytes(spec, cfg):
    """Per-iteration K (summation) address increment in bytes, matching
    GL2Prefetch.setIncrement. Advancing the prefetch by one iteration moves a
    full DepthU along the summation axis; in bytes this is:
      - MX:       SizeFree * (DepthU // MXBlock)              (* bpe == 1)
      - Metadata: StrideMetadataL(=coal_m) * DepthUMetadata if TLUMetadata,
                  else DepthUMetadata                          (bpe == 1)
      - TLU:      StrideUnroll(=coal) * (_DepthU{A,B} * bpe)
      - non-TLU:  (_DepthU{A,B} * bpe)                        (K is the coalesced axis)
    """
    bpe = spec.bpe
    if spec.is_mx:
        return free_dim_size(cfg, spec.subtc) * round(cfg.depth_u // cfg.mx_block * bpe)
    if spec.is_m:
        coal, _, _, _ = tensor_dims(spec, cfg)
        return round(coal * cfg.depth_u_metadata) if spec.tlu else round(cfg.depth_u_metadata)
    if spec.tlu:
        return _data_coal(cfg, spec.subtc) * round(data_depth_u(spec, cfg) * bpe)
    return round(data_depth_u(spec, cfg) * bpe)


def expected_offsets(spec, cfg, stage=0, batch=0, group=0):
    """Geometric prefetch footprint: the *set* of byte offsets a tensor's
    prefetch must cover, independent of how threads are allocated to addresses.

    `stage` shifts the whole footprint along the K axis: stage 0 is the start
    address (the calculateStartAddr PGR pre-skip already advanced it by PGR
    increments), and each later stage adds one incrementAddr. The shift is
    orthogonal to the free-dim edge clamp, so it just translates the set.

    `group` is the GSU group, which shifts the footprint onto that group's K
    chunk. Both the chunk start and the stage stride are whole multiples of the
    one-DepthU increment, so the K shift is
        (startIter(group) + chunkIdx(stage, group) * iterStride) * inc
    with GSU off collapsing to the plain chunkIdx * inc. chunkIdx is PGR + stage
    unless StaggerU rotates the chunk; see stagger_chunk_idx.

    `batch` adds the StridedBatched shift batch * Stride{tc}K * bpe (the
    WorkGroup2 * batchStride term calculateStartAddr folds into the base
    address); like the stage shift it is a pure translation of the set.
    The whole cluster cooperates, so the footprint spans all mt_tiles macro-tiles
    the cluster consumes as one contiguous block (folded into tensor_dims): ncc
    coalesced GPS-chunks x `perp` perpendicular rows, with the edge-limit clamp
    min(index, SizeFree-1) applied to the coalesced index (TLU/MX) or the
    perpendicular index (non-TLU).

    We deliberately do NOT model the thread<->address mapping (cooperative-WG
    fan-out, inactive-bit shifts, per-thread load counts): those are an
    implementation detail. Any allocation that yields the same footprint passes;
    only a coverage bug (a missing/extra/out-of-bounds address) fails."""
    GPS = GLOBAL_PREFETCH_SIZE
    bpe = spec.bpe
    coal, perp, ncc, _ = tensor_dims(spec, cfg)   # tile dim folded over the cluster
    size_free = free_dim_size(cfg, spec.subtc)
    if spec.is_mx:
        mx_unit = cfg.matrix_inst_k // cfg.mx_block
        perp_stride = size_free * mx_unit
        edge = (size_free - 1) * mx_unit
    else:
        perp_stride = coal            # StrideAL (TLU) / StrideAI (nTLU): the folded leading dim
        edge = size_free - 1
    coal_to_mt = (spec.is_mx or spec.tlu)    # MT offset & clamp land in coal (else perp)
    gps_elems = round(GPS / bpe)
    k_iter = gsu_start_iter(cfg, group) + stagger_chunk_idx(cfg, stage, group) * gsu_iter_stride(cfg)
    shift = k_iter * inc_bytes(spec, cfg)
    if cfg.batched:
        shift += batch * round(batch_stride_elems(spec, cfg) * bpe)
    out = set()
    for c in range(ncc):
        for p in range(perp):
            if coal_to_mt:
                coal_idx = min(c * gps_elems, edge)
                perp_idx = p
            else:
                perp_idx = min(p, edge)
                coal_idx = c * gps_elems
            out.add(round((perp_idx * perp_stride + coal_idx) * bpe) + shift)
    return out


def verify_tensor(offsets, spec, cfg, stage, batch=0, group=0, debug=False):
    """Compare the union of GPU-computed byte offsets for one
    (stage, batch, GSU group) against the geometric prefetch footprint
    (set-based, edge-clamp aware, shifted by the group's K chunk, the stage's K
    increment and the batch's Stride{tc}K offset)."""
    expected = expected_offsets(spec, cfg, stage, batch, group)
    got = set(offsets)
    errors = []
    tag = f"{spec.tc}[s{stage}b{batch}" + (f"g{group}" if cfg.gsu_on else "") + "]"
    missing = sorted(expected - got)
    extra = sorted(got - expected)
    if missing:
        errors.append(f"{tag}: missing {missing[:6]}")
    if extra:
        errors.append(f"{tag}: unexpected {extra[:6]}")
    if debug:
        _, _, ncc, nc = tensor_dims(spec, cfg)
        M = mt_tiles(spec, cfg)
        clamped = "" if free_dim_size(cfg, spec.subtc) == M * spec.mt else " EDGE"
        gsu = f" startIter={gsu_start_iter(cfg, group)}x{gsu_iter_stride(cfg)}" if cfg.gsu_on else ""
        print(f"  {tag:12s}: ncc={ncc} nc={nc} mt_tiles={M}{clamped}{gsu} "
              f"inc={inc_bytes(spec, cfg)} expect={len(expected)} unique={len(got)} "
              f"total={len(offsets)} max={max(got) if got else 0}")
    return errors


def run_config(cfg, tmp_dir, debug=False):
    asm, layout, n_out = build_kernel(cfg)
    co_path = os.path.join(tmp_dir, f"gl2_{cfg.name}.co")
    if debug:
        with open(os.path.join(tmp_dir, f"gl2_{cfg.name}.s"), "w") as f:
            f.write(asm)
    # gfx1250 is natively wave32; its assembler rejects -mwavefrontsize32, so
    # omit the flag and let the target default apply.
    assemble_kernel(asm, co_path, wavefront_size=None)

    base = np.zeros(64 * 1024 * 1024, dtype=np.uint8)   # valid base pointer (contents unused)
    # Launch a real [cx, cy, num_batches] grid in one shot. Each wg self-identifies
    # via ttmp and writes its offsets into region [lin*n_out, (lin+1)*n_out), with
    # lin = wg_z*(cx*cy) + wg_y*cx + wg_x in [0, n_regions). The batch index is
    # lin // (cx*cy); offsets are aggregated per (tensor, stage, batch) and each
    # batch is checked against its own Stride{tc}K-shifted footprint.
    n_wg = cfg.n_wg
    n_groups = cfg.n_groups
    n_regions = cfg.n_regions
    raw = run_on_gpu(co_path, n_regions * n_out * 4, inputs=(base,),
                     num_threads=cfg.num_threads,
                     grid=(cfg.cluster[0], cfg.cluster[1], cfg.grid_z))
    vals = struct.unpack(f"{n_regions * n_out}I", raw)
    # Aggregate each (tensor, stage, batch, GSU group)'s offsets across the
    # cooperative cluster wgs -- but *not* across GSU groups: those do not
    # cooperate, each owns a different K chunk and is checked against it. Batches
    # and groups share the z axis and a config uses at most one, so divmod picks
    # out whichever is active.
    per = {(t.tc, stage, b, g): []
           for t, _, stage, _ in layout
           for b in range(cfg.num_batches) for g in range(n_groups)}
    for lin in range(n_regions):
        b, g = divmod(lin // n_wg, n_groups)
        lin_base = lin * n_out
        for t, num_loads, stage, region in layout:
            start = lin_base + region
            per[(t.tc, stage, b, g)].extend(vals[start: start + cfg.num_threads * num_loads])

    errors = []
    for t, _, stage, _ in layout:
        for b in range(cfg.num_batches):
            for g in range(n_groups):
                errors += verify_tensor(per[(t.tc, stage, b, g)], t, cfg, stage, b, g, debug=debug)
    return errors


# ---------------------------------------------------------------------------
# Pytest
# ---------------------------------------------------------------------------
@pytest.mark.gfx1250
@pytest.mark.skipif(not HAS_GFX1250, reason=f"GL2 prefetch tests require gfx1250, found {GFX_TARGET}")
class TestGL2PrefetchOffset:

    @pytest.fixture(params=CONFIGS, ids=lambda c: c.name)
    def cfg(self, request):
        return request.param

    def test_gl2_prefetch_offset(self, cfg, tmp_path):
        errors = run_config(cfg, str(tmp_path))
        assert not errors, f"Config {cfg.name}: " + "; ".join(errors)


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GL2 prefetch address verification (gfx1250)")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if not HAS_GFX1250:
        print(f"SKIP: requires gfx1250, found {GFX_TARGET}")
        sys.exit(0)

    total = 0
    with tempfile.TemporaryDirectory() as tmp:
        for cfg in CONFIGS:
            print(f"\n=== {cfg.name} ===")
            errs = run_config(cfg, tmp, debug=args.debug)
            if errs:
                print("  FAIL:", "; ".join(errs))
                total += len(errs)
            else:
                print("  PASS")
    print(f"\nResult: {total} errors")
    sys.exit(1 if total else 0)

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
#   - ClusterDim != [1,1]: gl2-prefetch is only emitted for cooperative clusters,
#     so every config runs a real [cx, cy] grid (shapes vary, incl. [4,4]). Each
#     WG self-identifies via ttmp (gfx12 carries the workgroup id in ttmp, not
#     s2): wg_x -> WorkGroup0, wg_y -> WorkGroup1. A 2D cluster drives A and B
#     (and MXSA/MXSB) cooperatively at the same time, since A is cooperative along
#     WorkGroup1 / macro-tile-selected by WorkGroup0 and B is the mirror. Each WG
#     writes to its own output region; the host aggregates across all cx*cy.
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
# Verification is set-based: the union of each tensor's computed byte offsets
# (per stage) must equal M macro-tiles {m*mt_stride + c*GPS} shifted by the
# stage's K increment, where M is the launch extent along the tensor's
# MT-selector axis (1 for the non-cluster case). This tolerates the benign
# replication when a tensor's nc < cooperative threads (e.g. MX scales).
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
    tc: str            # "A", "B", "MXSA", "MXSB"
    tlu: bool
    mt: int            # MacroTile (A-type tensors share MacroTileA; B-type MacroTileB)
    bpe: float = 1     # bytes/elem: FP8/scale=1 (int), FP4=0.5

    @property
    def subtc(self):
        return self.tc[-1]            # 'A' or 'B'

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

    @property
    def n_wg(self):
        return self.cluster[0] * self.cluster[1]

    @property
    def n_regions(self):
        # distinct output regions = cooperative cluster wgs * batches (grid z)
        return self.n_wg * self.num_batches


def tensor_dims(spec, cfg):
    """(coal_dim, perp_dim, ncc, nc) for a tensor, matching GL2Prefetch.init."""
    if spec.is_mx:
        coal = spec.mt * cfg.matrix_inst_k // cfg.mx_block
        perp = cfg.depth_u // cfg.matrix_inst_k
    else:
        coal, perp = (spec.mt, cfg.depth_u) if spec.tlu else (cfg.depth_u, spec.mt)
    ncc = max(1, round(coal * spec.bpe) // GLOBAL_PREFETCH_SIZE)
    return coal, perp, ncc, perp * ncc


def free_dim_size(cfg, subtc):
    """Programmed SizeI/SizeJ (the GEMM free-dim, used for the edge-limit clamp).
    Defaults to a clean tiling (cluster_extent * MacroTile); an explicit size_i/
    size_j makes the last macro-tile partial so the edge clamp fires."""
    mt = next((t.mt for t in cfg.tensors if t.subtc == subtc), 1)
    if subtc == "A":
        return cfg.size_i if cfg.size_i is not None else cfg.cluster[0] * mt
    return cfg.size_j if cfg.size_j is not None else cfg.cluster[1] * mt


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


# gl2-prefetch is only emitted for ClusterDim != [1,1], so every config runs a
# (power-of-2) cooperative cluster. ClusterDim = [cx, cy]: A/MXSA cooperate along
# cy and span cx macro-tiles; B/MXSB are the mirror. Cluster shapes are varied
# across configs (and include [4,4]) to exercise both cooperative axes.
CONFIGS = [
    # ---- A + B together, FP8, assorted cluster shapes (incl. [4,4]) ----
    GL2Config("ab_fp8_tlu",          [_A(True, 256),  _B(True, 256)],  cluster=(2, 2)),
    GL2Config("ab_fp8_ntlu",         [_A(False, 256), _B(False, 256)], cluster=(4, 4)),
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
    GL2Config("abmx_fp8",      [_A(True, 256),  _B(True, 256),  _MXSA(256), _MXSB(256)],
              depth_u=256, mx_block=32, cluster=(2, 2), batched=True, num_batches=2),
    GL2Config("abmx_fp8_ntlu", [_A(False, 256), _B(False, 256), _MXSA(256), _MXSB(256)],
              depth_u=256, mx_block=32, cluster=(2, 1)),
    # ---- FP4 (bpe=0.5) on A and B, TLU, ncc==1 and (coal*bpe==2*GPS) ncc==2 ----
    GL2Config("ab_fp4_tlu",      [_A(True, 512, bpe=0.5),  _B(True, 512, bpe=0.5)],
              depth_u=256, cluster=(1, 4)),
    GL2Config("ab_fp4_tlu_ncc2", [_A(True, 1024, bpe=0.5), _B(True, 1024, bpe=0.5)],
              depth_u=128, cluster=(4, 1)),
    # ---- non-TLU: FP4 tile-split on A + FP8 coalesced-split ncc2 on B (coal==DepthU) ----
    GL2Config("ab_ntlu_f4f8", [_A(False, 256, bpe=0.5), _B(False, 128, bpe=1)],
              depth_u=512, cluster=(2, 2)),
    # ---- MX scales together: MXSA at ncc==2, MXSB at ncc==1 ----
    GL2Config("mxab_ncc", [_MXSA(128), _MXSB(64)], depth_u=1024, num_threads=16,
              mx_block=32, cluster=(2, 4)),
    # ---- Edge clamp: SizeI/SizeJ is NOT a clean multiple of the tiling, so the
    # last macro-tile is partial and the edge-limit clamp min(idx, Size-1) fires.
    # non-TLU clamps the perpendicular index; TLU/MX clamp the coalesced index.
    GL2Config("ab_ntlu_edge", [_A(False, 256), _B(False, 256)],
              cluster=(2, 2), size_i=384, size_j=384),
    GL2Config("ab_tlu_edge",  [_A(True, 512), _B(True, 512)],
              depth_u=128, cluster=(2, 2), size_i=700, size_j=700),
    GL2Config("mx_edge", [_MXSA(128), _MXSB(64)], depth_u=1024, num_threads=16,
              mx_block=32, cluster=(2, 2), size_i=150, size_j=50),
]


def batch_stride_elems(spec, cfg):
    """Per-tensor batch stride in *elements* (the programmed Stride{tc}K). Arbitrary
    but fixed and distinct per tensor so the verifier can reproduce the
    WorkGroup2 * Stride{tc}K * bpe shift and so a cross-tensor stride mixup fails.
    Chosen even so that stride * bpe is integral for fractional bpe (FP4)."""
    return {"A": 1_000_002, "B": 2_000_006, "MXSA": 3_000_010, "MXSB": 4_000_014}[spec.tc]

# ---------------------------------------------------------------------------
# Kernel + writer construction
# ---------------------------------------------------------------------------

def _subtc_attr(cfg, sub, attr, default):
    for t in cfg.tensors:
        if t.subtc == sub:
            return getattr(t, attr)
    return default


def _make_kernel(cfg):
    has_mxa = any(t.tc == "MXSA" for t in cfg.tensors)
    has_mxb = any(t.tc == "MXSB" for t in cfg.tensors)
    return {
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
        },
        "MacroTileA": _subtc_attr(cfg, "A", "mt", 256),
        "MacroTileB": _subtc_attr(cfg, "B", "mt", 256),
        "MatrixInstK": cfg.matrix_inst_k,
        "ClusterDim": list(cfg.cluster),
        "NumThreads": cfg.num_threads,
        "DepthU": cfg.depth_u,
        "PrefetchGlobalRead": cfg.pgr,
        "WavefrontSize": WAVESIZE,
        "PrefetchGL2": cfg.pgl,
    }


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
        regCaps={"MaxSgpr": 102, "MaxVgpr": 1024, "PhysicalMaxVgpr": 1024,
                 "GlobalPrefetchSize": GLOBAL_PREFETCH_SIZE},
        asmCaps={"HasSMulHi": True, "HasGlobalPrefetch": True},
        unrollIdx=0,
        overflowedResources=0,
        a=SimpleNamespace(), b=SimpleNamespace(),
    )
    for m in ["strideRef", "allocTmpSgpr", "s_mul_u64_u32"]:
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
    from rocisa.container import sgpr
    from rocisa.instruction import SMovB32
    from Tensile.KernelWriterAssembly import GL2PrefetchLoad

    init_rocisa(wavesize=WAVESIZE)
    kernel = _make_kernel(cfg)
    w = _make_writer(kernel)
    comp = GL2PrefetchLoad()

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
    for n in shared:
        w.sgprs[n] = w.sgprPool.checkOut(1, n, preventOverflow=False)
    for t in cfg.tensors:
        w.sgprs[f"Address{t.tc}"] = w.sgprPool.checkOutAligned(2, 2, f"Address{t.tc}", preventOverflow=False)
        w.sgprs[f"GL2PrefetchInc{t.tc}"] = w.sgprPool.checkOutAligned(2, 2, f"GL2PrefetchInc{t.tc}", preventOverflow=False)
        if cfg.batched:    # batch stride Stride{tc}K (index 2 -> 'K')
            w.sgprs[f"Stride{t.tc}K"] = w.sgprPool.checkOut(1, f"Stride{t.tc}K", preventOverflow=False)

    w.vgprPool.checkOut(1)  # v0 = Serial (workitem id)

    # ---- init each tp, allocate its per-load address vgprs ----
    tps = []
    vgpr_sets = {}
    for t in cfg.tensors:
        ia = t.ia + [2] if cfg.batched else t.ia   # batch index 2 must be in ia
        tp = {"tensorChar": t.tc, "idx": t.idx, "tlu": t.tlu, "bpeGR": t.bpe, "ia": ia}
        comp.init(w, kernel, tp)
        assert tp["gl2nc"] == tensor_dims(t, cfg)[3], \
            f"{t.tc}: gl2nc {tp['gl2nc']} != expected {tensor_dims(t, cfg)[3]}"
        for i in range(tp["gl2nlp"]):
            for j in range(tp["gl2nlc"]):
                name = f"GL2PrefetchAddr{t.tc}_{i}_{j}"
                vgpr_sets[name] = w.vgprPool.checkOutAligned(2, 2, name, preventOverflow=False)
        tps.append((t, tp))

    # output elements written by one workgroup (used to shift per-wg regions);
    # each of the n_stages re-exports every tensor's loads.
    n_stages = cfg.n_inc + 1
    n_out_per_wg = n_stages * sum(cfg.num_threads * tp["gl2nlp"] * tp["gl2nlc"] for _, tp in tps)

    # ---- body: setIncrement (all), then calculateStartAddr (each).
    # calculateStartAddr now folds in the base Address{tc} and the PGR pre-skip
    # itself (SGPR-accumulated), so there is no separate gsuOffset step. ----
    body = Module("body")
    for t, tp in tps:
        body.add(comp.setIncrement(w, kernel, tp))
        body.add(comp.calculateStartAddr(w, kernel, tp))

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
    consts += [("WorkGroup0", 0), ("WorkGroup1", 0), ("WorkGroup2", 0)]
    if cfg.batched:                              # programmed batch stride Stride{tc}K
        consts += [(f"Stride{t.tc}K", batch_stride_elems(t, cfg)) for t in cfg.tensors]
    for n, v in consts:
        prologue.add(SMovB32(dst=sgpr(n), src=v))

    # gfx1250 carries the workgroup id in ttmp (not s2): wg_x in ttmp9, wg_y in
    # ttmp7[15:0], wg_z in ttmp7[31:16] (matching the production non-cluster
    # decode). The cooperative cluster drives WorkGroup0/1; the batch dim drives
    # WorkGroup2. Each region's linear id is wg_z*(cx*cy) + wg_y*cx + wg_x.
    cx, cy = cfg.cluster
    if cfg.n_wg > 1:
        prologue.add(TextBlock("  s_mov_b32 s%d, ttmp9\n" % w.sgprs["WorkGroup0"]))
        prologue.add(TextBlock("  s_and_b32 s%d, 0xFFFF, ttmp7\n" % w.sgprs["WorkGroup1"]))
    if cfg.num_batches > 1:
        prologue.add(TextBlock("  s_lshr_b32 s%d, ttmp7, 16\n" % w.sgprs["WorkGroup2"]))
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
        num_loads = tp["gl2nlp"] * tp["gl2nlc"]
        base = w.sgprs[f"Address{t.tc}"]
        k = 0
        for i in range(tp["gl2nlp"]):
            for j in range(tp["gl2nlc"]):
                addr = vgpr_sets[f"GL2PrefetchAddr{t.tc}_{i}_{j}"]
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

    layout = []
    region = 0
    for stage in range(n_stages):
        if stage > 0:                          # advance every tensor by one inc
            for t, tp in tps:
                epi.add(comp.incrementAddr(w, kernel, tp))
        for t, tp in tps:
            num_loads = tp["gl2nlp"] * tp["gl2nlc"]
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
    works there; fall back to whatever tensor is present."""
    for t in cfg.tensors:
        if t.subtc == sub and not t.is_mx:
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
      - MX:      SizeFree * (DepthU // MXBlock)              (* bpe == 1)
      - TLU:     StrideUnroll(=coal) * (DepthU * bpe)
      - non-TLU: (DepthU * bpe)                              (K is the coalesced axis)
    """
    bpe = spec.bpe
    if spec.is_mx:
        return free_dim_size(cfg, spec.subtc) * round(cfg.depth_u // cfg.mx_block * bpe)
    if spec.tlu:
        return _data_coal(cfg, spec.subtc) * round(cfg.depth_u * bpe)
    return round(cfg.depth_u * bpe)


def expected_offsets(spec, cfg, stage=0, batch=0):
    """Geometric prefetch footprint: the *set* of byte offsets a tensor's
    prefetch must cover, independent of how threads are allocated to addresses.

    `stage` shifts the whole footprint by (PGR + stage) * inc along the K axis:
    stage 0 is the start address (the calculateStartAddr PGR pre-skip already
    advanced it by PGR*inc), and each later stage adds one incrementAddr. The
    shift is orthogonal to the free-dim edge clamp, so it just translates the set.

    `batch` adds the StridedBatched shift batch * Stride{tc}K * bpe (the
    WorkGroup2 * batchStride term calculateStartAddr folds into the base
    address); like the stage shift it is a pure translation of the set.
    Per tensor this is M macro-tiles (MT-selector axis) x ncc coalesced GPS-chunks
    x `perp` perpendicular rows, with the edge-limit clamp min(index, SizeFree-1)
    applied to the coalesced index (TLU/MX) or the perpendicular index (non-TLU).

    We deliberately do NOT model the thread<->address mapping (cooperative-WG
    fan-out, inactive-bit shifts, per-thread load counts): those are an
    implementation detail. Any allocation that yields the same footprint passes;
    only a coverage bug (a missing/extra/out-of-bounds address) fails."""
    GPS = GLOBAL_PREFETCH_SIZE
    bpe = spec.bpe
    coal, perp, ncc, _ = tensor_dims(spec, cfg)
    M = mt_tiles(spec, cfg)
    size_free = free_dim_size(cfg, spec.subtc)
    if spec.is_mx:
        mx_unit = cfg.matrix_inst_k // cfg.mx_block
        perp_stride = size_free * mx_unit
        edge = (size_free - 1) * mx_unit
        mt_off = mx_unit * spec.mt
    else:
        perp_stride = coal            # StrideAL (TLU) == mt; StrideAI (nTLU) == DepthU
        edge = size_free - 1
        mt_off = spec.mt
    coal_to_mt = (spec.is_mx or spec.tlu)    # MT offset & clamp land in coal (else perp)
    gps_elems = round(GPS / bpe)
    shift = (cfg.pgr + stage) * inc_bytes(spec, cfg)
    if cfg.batched:
        shift += batch * round(batch_stride_elems(spec, cfg) * bpe)
    out = set()
    for m in range(M):
        for c in range(ncc):
            for p in range(perp):
                if coal_to_mt:
                    coal_idx = min(m * mt_off + c * gps_elems, edge)
                    perp_idx = p
                else:
                    perp_idx = min(p + m * mt_off, edge)
                    coal_idx = c * gps_elems
                out.add(round((perp_idx * perp_stride + coal_idx) * bpe) + shift)
    return out


def verify_tensor(offsets, spec, cfg, stage, batch=0, debug=False):
    """Compare the union of GPU-computed byte offsets for one (stage, batch)
    against the geometric prefetch footprint (set-based, edge-clamp aware,
    shifted by the stage's K increment and the batch's Stride{tc}K offset)."""
    expected = expected_offsets(spec, cfg, stage, batch)
    got = set(offsets)
    errors = []
    tag = f"{spec.tc}[s{stage}b{batch}]"
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
        print(f"  {tag:9s}: ncc={ncc} nc={nc} mt_tiles={M}{clamped} "
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
    n_regions = cfg.n_regions
    raw = run_on_gpu(co_path, n_regions * n_out * 4, inputs=(base,),
                     num_threads=cfg.num_threads,
                     grid=(cfg.cluster[0], cfg.cluster[1], cfg.num_batches))
    vals = struct.unpack(f"{n_regions * n_out}I", raw)
    # aggregate each (tensor, stage, batch)'s offsets across all cooperative wgs
    per = {(t.tc, stage, b): [] for t, _, stage, _ in layout for b in range(cfg.num_batches)}
    for lin in range(n_regions):
        b = lin // n_wg
        lin_base = lin * n_out
        for t, num_loads, stage, region in layout:
            start = lin_base + region
            per[(t.tc, stage, b)].extend(vals[start: start + cfg.num_threads * num_loads])

    errors = []
    for t, _, stage, _ in layout:
        for b in range(cfg.num_batches):
            errors += verify_tensor(per[(t.tc, stage, b)], t, cfg, stage, b, debug=debug)
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

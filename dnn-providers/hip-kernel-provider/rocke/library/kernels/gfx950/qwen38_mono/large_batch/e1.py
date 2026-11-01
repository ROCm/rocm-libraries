# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""WP-L2a: the productized large-batch dense decode GEMM core (DESIGN-LB E1), gfx950.

A fork of probe P-L3's :mod:`.gemm` (``gemm.py`` stays byte-identical: ``insitu.py`` subclasses it and P-L3's
measurements name it). With every new knob at its default and ``hz=False`` a build here lowers to the same LLVM IR as
the matching :mod:`.gemm` build apart from the kernel name (``q38lb_e1_*`` here, ``q38lb_gemm_*`` there); the CPU
tests check that, so P-L3's numerics and verdicts carry over to the default arm. ``hz`` (on by default) pads every
inline-asm block that issues VMEM on an SGPR operand with ``s_nop 4`` (alola rules 176 / 182 / 191; WP-C5 F1).

Everything P-L3 documents holds (``Y[token, row] = sum_k X[token, k] W[row, k]``, weights as the MFMA A operand,
16-token tiles as N, units = (layer, row group, token block, K split) walked by a persistent grid, ``mx`` = MXFP4 W x
MXFP8 X on ``v_mfma_scale_f32_16x16x128_f8f6f4`` and ``bf16`` = FlyTape's bf16 operand map; see ``gemm.py``'s
docstring for layouts and arguments). New default-off levers (each in the kernel name when set):

* ``kw`` (K split over the waves of a CTA): the ``waves`` of a CTA form ``kw`` groups of ``wr = waves / kw`` waves;
  group ``g`` multiplies chunks ``[g kc, (g + 1) kc)`` of the unit's K range (``kc = nq / (splits kw)``) for the
  unit's ``16 wr mt`` rows, and the groups' accumulators are added in LDS in group order (0, 1, .., kw - 1, fixed:
  deterministic) before the epilogue. No cross-CTA fold for the K the CTA covers: the hierarchical (wave -> CU)
  reduction of XCD-SCHEDULING lever 5, and the restructure for latency-bound small GEMMs (out-proj, HC down) whose
  split-K fold costs FlyTape 0.06-0.42 of its phase (P-L3 5.4). ``xsrc="lds"`` stages one chunk per group
  (``kw x 2`` LDS buffers); ``xsrc="global"`` lets every wave read its own B operands (no staging, no barriers).
* ``sop`` for short units: op_sel scales were legal only when a unit's chunks are a multiple of 4 (P-L3). A unit of 1
  or 2 chunks (``kc`` in (1, 2): FlyTape's ``OUT_SPLITS`` / ``MIX_SPLITS`` plans) now loads the 4-chunk scale dwords
  once and shifts them by ``8 (q0 mod 4)`` once per unit (never per MFMA), then selects byte ``c`` with op_sel.
  ``sop`` with ``xsrc="global"`` is new too (B codes straight from global, scales as before).
* ``act="fp4"`` (W4A4, mx only): MXFP4 activations ([rows, K / 2] codes, even k in the low nibble, E8M0 scales as for
  fp8) on the fp4 x fp4 scaled MFMA: half the activation bytes and LDS, the double-rate MFMA; accuracy-gated (FP4
  activations cost ATOM 3.3 GSM8K points: a tolerance knob, not a default).
* ``bar="persist"``: the phase metric's grid barrier as W5's striped barrier (``persist.sync``: 8 counter words per XCD
  on their own 256-byte lines, one non-returning agent-scope arrival per CTA, a 64-lane ballot poll of all words with
  ``s_nop`` pauses and an iteration bound; a timeout adds 1 to ``gs`` word 0 and releases), cumulative per launch and
  reset to zero by the launch's last CTA (one returning atomic per CTA at the launch end). ``probe`` = P-L3's barrier
  (8 words, a returning arrival, ``s_sleep 1`` polls, ``s_memrealtime`` deadline). See :func:`gs_bytes`.
* ``fg``: f32 partial loads in flight per fold step (P-L3: 8; FlyTape's ``FOLD_LOADS`` keeps every split's loads of
  a tile in flight).
* ``kseg`` (segment scales: the HC per-stream re-association, DESIGN-LB S1 / E2): K is ``kseg`` equal segments
  (streams); every wave multiplies its accumulators by ``seg[segment][token]`` (f32) before any cross-wave or
  cross-CTA sum, so ``Y = sum_s seg_s (X_s W_s^T)``: FlyTape's TwoStage source normalisation (``x rrms_s (1 + g)``,
  applied while staging) re-associated so the producer writes the plain residual (``(1 + g)`` folded into W once,
  host-side). Needs every unit's and every wave group's K range inside one segment (``splits kw`` a multiple of
  ``kseg``). Kernel argument ``seg`` (after ``gs``).
* ``yq`` (producer epilogue in the op_sel layout): besides bf16 Y, the unit that completes an output block (the unit
  itself at one split, else the fold's last arriver) quantizes ``bf16(Y)`` to MXFP8 by the decode rule of
  ``rocke.helpers.mx_quant`` (blocks of 32 rows of one token) and writes the codes ``yqc[layer][token][M]`` and the
  scales in the consumer GEMM's ``sop`` layout ``yqs[layer][nq4][bucket16][4 g][4 j]`` (``nq4 = ceil(ceil(M / 128) / 4)``,
  byte ``16 G + 4 j + g`` of a 512-row group): the next E1 GEMM (K = M) reads them as its X with no repack.
  ``sc1`` (write-through) stores. Needs ``16 wr mt`` a multiple of 32; ``M`` must be a multiple of 32 (host).
* ``chain`` (protocol test mode, needs ``yq`` and ``M == K``): layer ``l > 0`` reads its X from layer ``l - 1``'s
  ``yq`` output, so the grid barrier carries a real dependency; after every barrier each wave acquires
  (``buffer_inv sc1``). Run it with ``gsync`` on (the stream walk would race).

Mutants (must fail): P-L3's ``kskip`` / ``scale`` / ``fold`` / ``sopsel``; ``kwred`` (the LDS reduction drops wave group
``kw - 1``), ``seg`` (every wave uses segment 0's scale), ``bardrop`` (``chain``: the barrier waits for one arrival less
on CTA 0's word), ``barnone`` (``chain``: no wait at all); in both CTA 0 sleeps about 60 us at the start of every unit,
so its rows are read stale deterministically; ``nowait`` (``fold=spread``: no group wait, CTA 0 late: its partials are
read stale; the bench poisons the partials before each check launch).

* ``mfma=32``: the 32x32x64 scaled MFMA (``v_mfma_scale_f32_32x32x64_f8f6f4``): 32-row weight tiles read from the same
  tile layout (two 16-byte loads a chunk a lane), two scale dwords per weight / token tile per 4-chunk group, 32-token B
  tiles, two K steps a chunk. Same MAC rate as 16x16x128 on gfx950; fewer, longer MFMAs. Needs ``sop`` and an even number
  of token tiles per unit.
* ``fold="spread"``: the split-K fold distributed over the group's units (FlyTape folds its HC down projection in a
  separate reduce phase over every CTA; P-L3's last arriver reads all S partials through one CU: 0.3 MB at 64 rows for
  HC down). Every unit arrives on its group's cumulative counter, waits for the group (only when every CTA holds at
  most one unit of the layer, a grid-uniform runtime test, else the last arriver folds everything) and folds 1 / S of
  the outputs in split order: the same bits as ``fold=last``; arrival and departure counters (2 RG ntb words a layer),
  zeroed by the last departer. ``noacq`` (``chain``: no acquire after the barrier) is the informational coherence twin (a miss proves
nothing, alola rule 182). Python engine only; names ``q38lb_e1_*``; nothing served imports it.
Measurements: ``experiments/qwen38-true-mono/large-batch/WP-L2a.md``.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, fields

from rocke.core.ir import (
    BF16,
    CPOL_NT,
    CPOL_SC1,
    F32,
    I32,
    I64,
    I8,
    IRBuilder,
    KernelDef,
    PtrType,
    Value,
    VectorType,
)
from rocke.helpers import device_ops as D
from rocke.helpers import mx_quant as MQ
from rocke.helpers.lds_arena import ONE_CTA_PER_CU_WORDS

from ..persist import sync as PS

__all__ = [
    "CHUNK",
    "TILE",
    "E1Knobs",
    "build_e1_kernel",
    "e1_params",
    "gs_bytes",
    "kernel_name",
    "ksp",
    "mpad",
    "pack_act_scales_sop",
    "pack_weights_tile_sop",
    "yq_bytes",
]

TILE = 16
CHUNK = 128
XCDS = 8
OOB = 0x7FFFFFF0  # buffer offset past any record a layer buffer holds (layers < 2 GiB): loads 0, stores dropped
MUTANTS = ("none", "kskip", "scale", "fold", "sopsel", "kwred", "seg", "bardrop", "barnone", "noacq", "nowait")
WCPS = {"default": 0, "nt": CPOL_NT}
BAR_LINE = 256  # bytes between persist-barrier counter words
BAR_WORDS = 64  # 8 per XCD
BAR_MAXIT = 1 << 21  # poll iterations before a persist-barrier wait gives up (seconds)
LDS_HALF = (ONE_CTA_PER_CU_WORDS - 64) * 4  # bytes: above this one CTA per CU holds without the occupancy pad
LDS_MAX = 160 * 1024 - 256


def _c(b: IRBuilder, v: int) -> Value:
    return b.const_i32(int(v))


def ksp(k: int) -> int:
    """Scale row stride in bytes (MXFP4 weights and MXFP8 activations): one byte per 32 K, rounded up to 16."""
    return -(-(-(-k // 32)) // 16) * 16


def mpad(m: int, knobs: "E1Knobs") -> int:
    """Y / partial row stride: ``m`` rounded up to the row group."""
    r = knobs.rows_per_unit
    return -(-m // r) * r


def gs_bytes(layers: int) -> int:
    """Bytes of the ``gs`` buffer for either barrier (zeroed once by the host, never again): ``probe`` uses words
    ``0 .. 8 (layers + 1) + 7``; ``persist`` word 0 (timeouts), 64 counters at ``256 (1 + w)`` and the launch-end word
    at ``256 * 65``."""
    return max(4 * (8 * (layers + 2) + 64), BAR_LINE * (BAR_WORDS + 2))


def yq_bytes(m: int, bucket: int) -> tuple[int, int]:
    """Per-layer bytes of the ``yq`` outputs: codes ``[bucket16][m]`` and op_sel scales ``[nq4][bucket16][16]``."""
    t16 = -(-bucket // TILE) * TILE
    nq4 = -(-(-(-m // CHUNK)) // 4)
    return t16 * m, nq4 * t16 * 16


@dataclass(frozen=True)
class E1Knobs:
    """One build. Defaults = P-L3's default arm; the kernel name lists only deviations from them."""

    w: str = "mx"  # "mx" (MXFP4 W x MXFP8 X) | "bf16" (bf16 W x bf16 X)
    k: int = 2560  # K
    bucket: int = 128  # rows bucket (token tiles = ceil(bucket / 16))
    waves: int = 8  # waves per CTA (4 | 8)
    mt: int = 1  # 16-row weight tiles per wave (1 | 2 | 4)
    splits: int = 1  # K splits over CTAs
    tb: int = 0  # token tiles per unit (0 = the whole bucket)
    depth: int = 2  # chunks of weight loads in flight
    xa: int = 1  # activation chunks in flight in registers (xsrc="lds")
    xsrc: str = "lds"  # "lds" (staged per CTA) | "global" (every wave loads its B operands, no LDS, no barrier)
    xpad: int = 0  # LDS row pad in bytes (0 = 16)
    wcp: str = "default"  # weight cache policy: "default" | "nt"
    xslay: str = "row"  # mx X scale layout (gemm.py)
    xscale: str = "lds"  # mx X scales: "lds" | "glb" (gemm.py)
    xsb: str = "shift"  # mx X scale operand (gemm.py)
    wsb: str = "shift"  # mx weight scale operand (gemm.py)
    sop: bool = False  # mx op_sel scales (gemm.py; here also kc in (1, 2) and xsrc="global")
    wlay: str = "row"  # weight layout: "row" | "tile"
    map: str = "rr"  # unit -> CTA map: "rr" | "xcd"
    xu: bool = True  # prefetch across unit boundaries
    grid: int = 256  # persistent CTAs
    occ1: bool = True  # pad LDS so one CTA fits a CU
    sb: bool = True  # sched_barrier fences around each chunk's issue block
    bpre: int = 0  # token tiles whose B operands are read ahead of the MFMAs
    fold: str = "last"  # splits > 1: "last" | "none" | "spread" (every unit of a group folds 1 / splits of its outputs)
    tstamp: bool = False  # diagnosis (gemm.py)
    diag: str = "none"  # diagnosis only (gemm.py)
    stamp: bool = False
    mutant: str = "none"
    tag: str = ""  # recompiled-control name suffix
    # ---- WP-L2a levers (default off)
    kw: int = 1  # K split over wave groups inside a CTA (1 | 2 | 4 | 8; divides waves)
    act: str = "fp8"  # mx activations: "fp8" (W4A8) | "fp4" (W4A4, tolerance-gated)
    bar: str = "probe"  # phase barrier: "probe" (P-L3) | "persist" (W5's striped barrier)
    fg: int = 8  # f32 partial loads in flight per fold step
    kseg: int = 0  # K segments with per-(segment, token) f32 scales (0 = off)
    yq: bool = False  # also write MXFP8(bf16 Y) in the consumer's op_sel layout
    chain: bool = False  # protocol test: layer l > 0 reads X from layer l - 1's yq output
    hz: bool = True  # s_nop 4 before inline-asm VMEM with SGPR operands
    xjunk: bool = False  # branch-free activation staging: threads past the chunk's pieces write an LDS junk row
    mfma: int = 16  # scaled MFMA shape: 16 (16x16x128) | 32 (32x32x64: 32-row weight tiles x 32-token tiles, 2 K steps a chunk)

    # ---------------------------------------------------------------- derived
    @property
    def tiles(self) -> int:
        return -(-self.bucket // TILE)

    @property
    def tbt(self) -> int:
        return self.tb or self.tiles

    @property
    def ntb(self) -> int:
        return -(-self.tiles // self.tbt)

    @property
    def nq(self) -> int:
        return -(-self.k // CHUNK)

    @property
    def span(self) -> int:
        """Chunks of one unit (all wave groups)."""
        return self.nq // self.splits

    @property
    def kc(self) -> int:
        """Chunks a wave multiplies per unit."""
        return self.nq // (self.splits * self.kw)

    @property
    def wr(self) -> int:
        """Waves per K group (row waves)."""
        return self.waves // self.kw

    @property
    def ragged(self) -> bool:
        return self.k % CHUNK != 0

    @property
    def rt(self) -> int:
        """Rows of one weight tile (the MFMA's M)."""
        return 32 if self.mfma == 32 else TILE

    @property
    def rows_per_unit(self) -> int:
        return self.rt * self.wr * self.mt

    @property
    def ntt(self) -> int:
        """Accumulator token tiles per unit (16 tokens, or 32 with mfma=32)."""
        return self.tbt // 2 if self.mfma == 32 else self.tbt

    @property
    def threads(self) -> int:
        return 64 * self.waves

    @property
    def pad(self) -> int:
        return self.xpad or 16

    @property
    def fp4x(self) -> bool:
        return self.w == "mx" and self.act == "fp4"

    @property
    def xrow(self) -> int:
        """Bytes of one staged activation row per chunk (without the pad)."""
        if self.w == "bf16":
            return CHUNK * 2
        return CHUNK // 2 if self.fp4x else CHUNK

    @property
    def sop_short(self) -> bool:
        """op_sel scales on 1- or 2-chunk wave ranges (one group dword, shifted once per unit)."""
        return self.sop and self.kc in (1, 2) and self.splits * self.kw > 1

    @property
    def sop_funnel(self) -> bool:
        """op_sel scales on wave ranges that may start inside a 4-chunk group and cross into the next (kc not a
        multiple of 4, not 1 or 2): every 4 chunks the dwords of groups G and G + 1 are funnel-shifted by the range's
        start (``8 (q0 mod 4)``) once per unit, so byte c of the result is chunk q0 + c's scale."""
        return self.sop and self.kc % 4 != 0 and self.kc not in (1, 2) and self.splits * self.kw > 1

    def check(self) -> None:
        if self.w not in ("mx", "bf16"):
            raise ValueError("w mx | bf16")
        if self.k <= 0 or self.k % 32:
            raise ValueError("k must be a positive multiple of 32")
        if not 1 <= self.bucket <= 128:
            raise ValueError("bucket 1..128")
        if self.waves not in (4, 8):
            raise ValueError("waves 4 | 8")
        if self.mt not in (1, 2, 4):
            raise ValueError("mt 1 | 2 | 4")
        if self.kw not in (1, 2, 4, 8) or self.waves % self.kw:
            raise ValueError("kw 1 | 2 | 4 | 8 dividing waves")
        if self.splits < 1 or self.nq % (self.splits * self.kw):
            raise ValueError(f"splits {self.splits} x kw {self.kw} must divide the {self.nq} K chunks")
        if self.tb < 0 or self.tb > self.tiles:
            raise ValueError("tb 0..tiles")
        if not 1 <= self.depth <= 8:
            raise ValueError("depth 1..8")
        if not 1 <= self.xa <= 4:
            raise ValueError("xa 1..4")
        if self.xsrc not in ("lds", "global"):
            raise ValueError("xsrc lds | global")
        if self.xpad % 16 or self.xpad < 0:
            raise ValueError("xpad a multiple of 16 bytes")
        if self.act not in ("fp8", "fp4") or (self.act == "fp4" and self.w != "mx"):
            raise ValueError("act fp8 | fp4 (fp4: w=mx)")
        if self.xslay not in ("row", "chunk"):
            raise ValueError("xslay row | chunk")
        if self.xslay == "chunk" and (self.w != "mx" or self.xsrc != "lds"):
            raise ValueError("xslay chunk: w=mx, xsrc=lds")
        if self.xsb not in ("shift", "byte", "dword"):
            raise ValueError("xsb shift | byte | dword")
        if self.xsb == "dword" and (self.xscale != "glb"):
            raise ValueError("xsb dword: xscale glb (X scales pre-unpacked to one dword per (chunk, token, block))")
        if self.xsb == "byte" and (self.w != "mx" or self.xsrc != "lds"):
            raise ValueError("xsb byte: w=mx, xsrc=lds")
        if self.wsb not in ("shift", "byte") or (self.wsb == "byte" and (self.w != "mx" or self.mutant == "scale")):
            raise ValueError("wsb shift | byte (mx, not with the scale mutant)")
        if self.xscale not in ("lds", "glb"):
            raise ValueError("xscale lds | glb")
        if self.xscale == "glb" and (self.xslay != "chunk" or self.diag == "noxs"):
            raise ValueError("xscale glb needs xslay chunk")
        if self.sop:
            glb_ok = self.xscale == "glb" if self.xsrc == "lds" else self.xslay == "row"
            kc_ok = True  # any kc: aligned groups, short ranges (shift) or funnel-shifted group pairs
            if not (self.w == "mx" and self.wlay == "tile" and glb_ok and self.xsb == "shift" and kc_ok
                    and self.diag == "none" and self.mutant != "scale" and self.wsb == "shift"):
                raise ValueError("sop: w=mx, wlay=tile, xscale=glb (lds) / xslay=row (global), xsb/wsb shift")
        elif self.xsrc == "global" and self.w == "mx" and (self.kw > 1 or self.fp4x):
            raise ValueError("xsrc=global with kw > 1 or act=fp4 needs sop (mx)")
        if (self.kw > 1 or self.fp4x) and self.w == "mx" and not self.sop:
            raise ValueError("kw > 1 / act=fp4 (mx): needs sop")
        if self.kw > 1 and self.diag != "none":
            raise ValueError("kw > 1: no diag builds")
        if self.mfma not in (16, 32):
            raise ValueError("mfma 16 | 32")
        if self.mfma == 32 and not (self.w == "mx" and self.sop and self.tbt % 2 == 0 and not self.yq and not self.kseg
                                    and not self.chain and self.bpre == 0 and self.diag == "none" and not self.tstamp):
            raise ValueError("mfma=32: mx, sop, an even number of token tiles per unit, no yq / kseg / chain / bpre / diag / tstamp")
        if self.wlay not in ("row", "tile"):
            raise ValueError("wlay row | tile")
        if self.wcp not in WCPS:
            raise ValueError(f"wcp in {sorted(WCPS)}")
        if self.map not in ("rr", "xcd"):
            raise ValueError("map rr | xcd")
        if self.grid < XCDS or self.grid % XCDS:
            raise ValueError("grid a multiple of 8")
        if self.bar not in ("probe", "persist"):
            raise ValueError("bar probe | persist")
        if self.bar == "persist" and self.grid % BAR_WORDS:
            raise ValueError("bar persist: grid a multiple of 64")
        if not 1 <= self.fg <= 32:
            raise ValueError("fg 1..32")
        if self.kseg:
            if self.kseg < 1 or (self.splits * self.kw) % self.kseg or self.nq % self.kseg:
                raise ValueError("kseg must divide nq and splits x kw")
        if self.yq:
            if self.rows_per_unit % 32:
                raise ValueError("yq: 16 wr mt must be a multiple of 32")
            if self.fold != "last" and self.splits > 1:
                raise ValueError("yq: fold=last")
            if self.ybuf_bytes() > 64 * 1024:
                raise ValueError("yq: rows per unit x token block too large for the LDS staging")
        if self.chain and not (self.yq and self.w == "mx" and self.act == "fp8" and self.sop):
            raise ValueError("chain: yq, mx, fp8, sop")
        if self.mutant not in MUTANTS:
            raise ValueError(f"mutant in {MUTANTS}")
        if self.mutant == "scale" and self.w != "mx":
            raise ValueError("the scale mutant needs w=mx")
        if self.mutant == "fold" and self.splits < 2:
            raise ValueError("the fold mutant needs splits > 1")
        if self.fold not in ("last", "none", "spread"):
            raise ValueError("fold last | none | spread")
        if self.fold == "spread" and (self.splits < 2 or self.yq):
            raise ValueError("fold=spread: splits > 1, not with yq")
        if self.mutant == "nowait" and self.fold != "spread":
            raise ValueError("the nowait mutant needs fold=spread")
        if self.diag not in ("none", "nomfma", "nob", "noxs", "nows", "xs0", "xsu", "xsk"):
            raise ValueError("diag none | nomfma | nob | noxs | nows | xs0 | xsu | xsk")
        if self.mutant == "fold" and self.fold not in ("last", "spread"):
            raise ValueError("the fold mutant needs fold=last | spread")
        if self.mutant == "sopsel" and not self.sop:
            raise ValueError("the sopsel mutant needs sop")
        if self.mutant == "kwred" and self.kw < 2:
            raise ValueError("the kwred mutant needs kw > 1")
        if self.mutant == "seg" and self.kseg < 2:
            raise ValueError("the seg mutant needs kseg > 1")
        if self.mutant in ("bardrop", "barnone", "noacq") and not self.chain:
            raise ValueError("the barrier / acquire mutants need chain")
        if not 0 <= self.bpre <= self.tbt:
            raise ValueError("bpre 0..tb")
        if self.tstamp and (self.stamp or 3 + 2 * self.kc > 64):
            raise ValueError("tstamp: not with stamp, kc <= 30")
        if self.kc > 80:
            raise ValueError("kc > 80 chunks per unit (unrolled): raise splits")
        if self.lds_bytes() > LDS_MAX:
            raise ValueError(f"LDS {self.lds_bytes()} bytes too large")
        if self.occ1 and not self.occ_pad() and self.lds_bytes() <= LDS_HALF + 1024:
            raise ValueError(f"LDS {self.lds_bytes()} bytes: neither the occupancy pad fits nor is it above half a CU")

    def ybuf_bytes(self) -> int:
        return self.rows_per_unit * self.tbt * TILE * 4 if self.yq else 0

    def red_bytes(self) -> int:
        return (self.kw - 1) * self.wr * self.mt * self.tbt * 64 * 16 * (2 if self.mfma == 32 else 1) if self.kw > 1 else 0

    def lds_bytes(self) -> int:
        """LDS of everything but the occupancy pad (gemm.py's count for the P-L3 arms, plus the new buffers)."""
        rows = self.tbt * TILE
        if self.xsrc == "global":
            base = 0
        elif self.w == "bf16":
            base = 2 * self.kw * (rows + self.xjunk) * (CHUNK * 2 + self.pad)
        else:
            base = 2 * self.kw * (rows + self.xjunk) * (self.xrow + self.pad) + 2 * rows * 4
        return base + self.red_bytes() + self.ybuf_bytes()

    def occ_pad(self) -> bool:
        """The occupancy pad is allocated (gemm.py always does with occ1) when it fits next to the build's own LDS
        (1 KiB slack for the small words); a build whose own LDS is above half a CU needs none."""
        return self.occ1 and self.lds_bytes() + ONE_CTA_PER_CU_WORDS * 4 <= LDS_MAX - 768

    def suffix(self) -> str:
        parts = []
        d = E1Knobs()
        for f in fields(self):
            if f.name in ("w", "k", "bucket", "tag"):
                continue
            v = getattr(self, f.name)
            if v == getattr(d, f.name):
                continue
            parts.append((f.name if v else f"no{f.name}") if isinstance(v, bool) else f"{f.name}{v}")
        if self.tag:
            parts.append(self.tag)
        return "_".join(parts)


def kernel_name(k: E1Knobs) -> str:
    s = k.suffix()
    return f"q38lb_e1_{k.w}_k{k.k}_b{k.tiles * TILE}" + (f"_{s}" if s else "")


def e1_params(k: E1Knobs) -> list[str]:
    """Parameter order: i64 ``w, ws, x, xs, y, part, cnt, gs`` (+ ``seg`` with kseg, + ``yqc, yqs`` with yq, + ``stamps``
    with stamp / tstamp), then i32 ``m, rows, layers, gsync`` (host packs ``<Q...iiii``)."""
    p = ["w", "ws", "x", "xs", "y", "part", "cnt", "gs"]
    if k.kseg:
        p.append("seg")
    if k.yq:
        p += ["yqc", "yqs"]
    if k.stamp or k.tstamp:
        p.append("stamps")
    return p + ["m", "rows", "layers", "gsync"]


# ------------------------------------------------------------------ host-side layouts (one-time repacks)


def pack_weights_tile_sop(codes, scales, m: int, k: int):
    """One-time checkpoint repack (bytes unchanged up to padding) of MXFP4 weights for ``wlay="tile"`` + ``sop``.

    ``codes``: uint8 ``[m, k / 2]`` (E2M1, even k in the low nibble); ``scales``: uint8 ``[m, >= ceil(k / 32)]`` (E8M0).
    Returns (``codes_t`` ``[ceil(m / 16)][nq][16][64]``, ``scales_op`` ``[ceil(m / 16)][nq4][16][4 g][4 j]``) as flat
    uint8 tensors (pad rows / K zero codes, pad scales 127). torch tensors in, torch tensors out (any device)."""
    import torch

    mt_, nq = -(-m // 16), -(-k // CHUNK)
    nq4 = -(-nq // 4)
    dev = codes.device
    c = torch.zeros(mt_ * 16, nq * 64, dtype=torch.uint8, device=dev)
    c[:m, : k // 2] = codes[:m, : k // 2]
    ct = c.view(mt_, 16, nq, 64).permute(0, 2, 1, 3).contiguous().view(-1)
    nsc = -(-k // 32)
    so = torch.full((mt_ * 16, nq4 * 16), 127, dtype=torch.uint8, device=dev)
    so[:m, :nsc] = scales[:m, :nsc]
    st = so.view(mt_, 16, nq4, 4, 4).permute(0, 2, 1, 4, 3).contiguous().view(-1)
    return ct, st


def pack_act_scales_sop(scales, bucket: int, k: int):
    """Activation (MXFP8 / MXFP4) scales ``[rows, >= ceil(k / 32)]`` uint8 -> the ``sop`` layout
    ``[nq4][bucket16][4 g][4 j]`` (what a ``yq`` producer writes directly). Pad scales 127."""
    import torch

    t16 = -(-bucket // TILE) * TILE
    nq4 = -(-(-(-k // CHUNK)) // 4)
    nsc = -(-k // 32)
    xp = torch.full((t16, nq4 * 16), 127, dtype=torch.uint8, device=scales.device)
    r = min(scales.shape[0], t16)
    xp[:r, :nsc] = scales[:r, :nsc]
    return xp.view(t16, nq4, 4, 4).permute(1, 0, 3, 2).contiguous().view(-1)


# ------------------------------------------------------------------ kernel


def _maybe_if(b: IRBuilder, cond):
    return contextlib.nullcontext() if cond is None else b.scf_if(cond)


def _ptr(b: IRBuilder, addr: Value, t=I32) -> Value:
    return b.int_to_ptr(addr, PtrType(t, "global"))


def _rsrc(b: IRBuilder, addr: Value, nbytes: Value) -> Value:
    return b.buffer_rsrc(_ptr(b, b.readfirstlane(addr)), nbytes)


def _i64(b: IRBuilder, v: Value) -> Value:
    return b.zext(v, I64)


def _cdiv(b: IRBuilder, a: Value, d: Value) -> Value:
    return b.div(b.add(a, b.sub(d, _c(b, 1))), d)


def _udiv(b: IRBuilder, x: Value, d: int) -> Value:
    if d & (d - 1) == 0:
        return b.lshr(x, _c(b, d.bit_length() - 1))
    return b.div(x, _c(b, d))


def _umod(b: IRBuilder, x: Value, d: int) -> Value:
    if d & (d - 1) == 0:
        return b.land(x, _c(b, d - 1))
    return b.mod(x, _c(b, d))


class _Unit:
    """Uniform scalars of this CTA's unit ``n`` (any ``n``; past the end it is invalid and every load is OOB)."""

    def __init__(self, g: "_E1", n: Value):
        b, k = g.b, g.k
        self.valid = b.cmp_lt(n, g.ncta)
        nn = b.select(self.valid, n, _c(b, 0))
        layer = b.div(nn, g.cnt1)
        self.i = b.readfirstlane(b.sub(nn, b.mul(layer, g.cnt1)))
        o = b.add(g.obase, b.mul(self.i, g.ostride))
        self.layer = b.readfirstlane(layer)
        o = b.readfirstlane(o)
        self.tbk = b.mod(o, g.NTBl)
        r2 = b.div(o, g.NTBl)
        self.rg = b.mod(r2, g.RG)
        self.s = b.div(r2, g.RG)
        if k.kw == 1:
            self.q0 = b.mul(self.s, _c(b, k.kc))  # first global chunk
        else:  # this wave's group: chunks [s span + kg kc, + kc)
            self.q0 = b.add(b.mul(self.s, _c(b, k.span)), g.kgc)
        self.tok0 = b.mul(self.tbk, _c(b, k.tbt * TILE))
        lay = _i64(b, self.layer)
        self.w_rs = _rsrc(b, b.add(g.a["w"], b.mul(lay, g.wl)), g.wl32)
        if k.w == "mx":
            self.s_rs = _rsrc(b, b.add(g.a["ws"], b.mul(lay, g.sl)), g.sl32)
            if k.wsb == "byte":
                self.s_ptr = b.int_to_ptr(b.readfirstlane(b.add(g.a["ws"], b.mul(lay, g.sl))), PtrType(I8, "global"))
        self.y_a = b.add(g.a["y"], b.mul(lay, g.yl))
        if k.splits > 1:
            self.p_a = b.add(g.a["part"], b.mul(lay, g.pl))
            self.c_a = b.add(g.a["cnt"], b.mul(lay, g.cl))
        if k.chain:  # X of layer l > 0 = layer l - 1's yq output
            prev = _i64(b, b.smax(b.sub(self.layer, _c(b, 1)), _c(b, 0)))
            first = b.cmp_eq(self.layer, _c(b, 0))
            xa = b.select(first, g.a["x"], b.add(g.a["yqc"], b.mul(prev, g.yqcl)))
            xsa = b.select(first, g.a["xs"], b.add(g.a["yqs"], b.mul(prev, g.yqsl)))
            self.x_rs = _rsrc(b, xa, g.xbytes)
            self.xs_rs = _rsrc(b, xsa, g.xsbytes)
        else:
            self.x_rs = g.x_rs
            self.xs_rs = getattr(g, "xs_rs", None)
        if k.yq:
            self.yqc_a = b.add(g.a["yqc"], b.mul(lay, g.yqcl))
            self.yqs_a = b.add(g.a["yqs"], b.mul(lay, g.yqsl))


class _E1:
    def __init__(self, k: E1Knobs):
        k.check()
        self.k = k
        self.b = b = IRBuilder(kernel_name(k))
        b.kernel.attrs["max_workgroup_size"] = k.threads
        names = e1_params(k)
        self.a = {n: b.param(n, I64) for n in names[:-4]}
        self.M = b.readfirstlane(b.param("m", I32))
        self.rows = b.readfirstlane(b.param("rows", I32))
        self.L = b.readfirstlane(b.param("layers", I32))
        gs_ = b.readfirstlane(b.param("gsync", I32))
        self.gsync = b.cmp_ne(gs_, _c(b, 0))
        self.nogs = b.cmp_eq(gs_, _c(b, 0))
        self.cta = b.readfirstlane(b.block_id_x())
        if k.stamp:
            self._stamp(0, None)
            self._stamp(2, D.xcc_id(b))
        self.tid = b.thread_id_x()
        self.lane = b.lane_id()
        self.wave = D.wave_id(b)
        self.g = b.lshr(self.lane, _c(b, 4))  # lane group 0..3
        self.l16 = b.land(self.lane, _c(b, 15))
        if k.mfma == 32:
            self.l32 = b.land(self.lane, _c(b, 31))
            self.h = b.lshr(self.lane, _c(b, 5))  # lane half: K block h of each 64-K step
        if k.kw > 1:
            self.kg = _udiv(b, self.wave, k.wr)  # K group of this wave
            self.wi = _umod(b, self.wave, k.wr)  # row wave within the group
            self.kgc = b.mul(self.kg, _c(b, k.kc))
            self.gtid = _umod(b, self.tid, 64 * k.wr)  # thread index within the group
        R, T = k.rows_per_unit, k.tiles
        TP = T * TILE
        # runtime geometry
        self.RG = b.readfirstlane(_cdiv(b, self.M, _c(b, R)))
        self.Mpad = b.mul(self.RG, _c(b, R))
        Tl = _cdiv(b, self.rows, _c(b, TILE))
        self.NTBl = b.readfirstlane(_cdiv(b, Tl, _c(b, k.tbt)))
        self.U = b.readfirstlane(b.mul(b.mul(self.RG, self.NTBl), _c(b, k.splits)))
        # units of one layer: CTA c takes o = obase + i ostride, i < cnt (the same count in every layer)
        if k.map == "rr":
            self.obase = self.cta
            self.ostride = _c(b, k.grid)
            cnt = b.smax(_cdiv(b, b.sub(self.U, self.cta), _c(b, k.grid)), _c(b, 0))
        else:
            GX = k.grid // XCDS
            x = _umod(b, self.cta, XCDS)
            xj = _udiv(b, self.cta, XCDS)
            lo = b.div(b.mul(x, self.U), _c(b, XCDS))
            hi = b.div(b.mul(b.add(x, _c(b, 1)), self.U), _c(b, XCDS))
            self.obase = b.add(lo, xj)
            self.ostride = _c(b, GX)
            cnt = b.smax(_cdiv(b, b.sub(b.sub(hi, lo), xj), _c(b, GX)), _c(b, 0))
        self.cnt = b.readfirstlane(cnt)
        self.cnt1 = b.readfirstlane(b.smax(cnt, _c(b, 1)))
        self.ncta = b.readfirstlane(b.mul(cnt, self.L))
        # per-layer strides (bytes)
        Mi = _i64(b, self.M)
        MT = _i64(b, _cdiv(b, self.M, _c(b, TILE)))
        if k.wlay == "tile":  # [M / 16][nq][16 rows x 128 K] (mx: 1 KiB codes + 64 B scales per tile-chunk)
            if k.w == "mx":
                self.wl = b.mul(MT, b.const_i64(k.nq * 1024))
                self.sl = b.mul(MT, b.const_i64((-(-k.nq // 4) * 256) if k.sop else (k.nq * 64)))
                self.sl32 = b.trunc(self.sl, I32)
            else:
                self.wl = b.mul(MT, b.const_i64(k.nq * 4096))
        elif k.w == "mx":
            self.wl = b.mul(Mi, b.const_i64(k.k // 2))
            self.sl = b.mul(Mi, b.const_i64(ksp(k.k)))
            self.sl32 = b.trunc(self.sl, I32)
        else:
            self.wl = b.mul(Mi, b.const_i64(k.k * 2))
        self.wl32 = b.trunc(self.wl, I32)
        Mp = _i64(b, self.Mpad)
        self.yl = b.mul(Mp, b.const_i64(TP * 2))
        self.yl32 = b.trunc(self.yl, I32)
        if k.splits > 1:
            self.pl = b.mul(Mp, b.const_i64(TP * 4 * k.splits))
            self.pl32 = b.trunc(self.pl, I32)
            self.cl = _i64(b, b.mul(b.mul(self.RG, _c(b, k.ntb)), _c(b, 8 if k.fold == "spread" else 4)))
        if k.yq:  # yq outputs: codes [bucket16][M], op_sel scales [nq4(M)][bucket16][16] per layer
            nq4m = b.lshr(b.add(_cdiv(b, self.M, _c(b, CHUNK)), _c(b, 3)), _c(b, 2))
            self.yqcl = b.mul(Mi, b.const_i64(TP))
            self.yqsl = _i64(b, b.mul(nq4m, _c(b, TP * 16)))
            self.yqcl32 = b.trunc(self.yqcl, I32)
            self.yqsl32 = b.trunc(self.yqsl, I32)
        # activations (shared by every layer)
        rl = b.smax(self.rows, _c(b, 1))
        self.rlast = b.sub(rl, _c(b, 1))
        if k.w == "mx":
            if k.chain:  # every layer's X is a full bucket (its producer's yq output)
                self.xbytes = _c(b, TP * k.k)
                self.xsbytes = _c(b, -(-k.nq // 4) * TP * 16)
            self.x_rs = _rsrc(b, self.a["x"], b.mul(rl, _c(b, k.k // 2 if k.fp4x else k.k)))
            if k.sop:
                self.xs_rs = _rsrc(b, self.a["xs"], _c(b, -(-k.nq // 4) * k.tiles * TILE * 16))
            elif k.xsb == "dword":
                self.xs_rs = _rsrc(b, self.a["xs"], _c(b, k.nq * k.tiles * TILE * 16))
            elif k.xslay == "chunk":
                self.xs_rs = _rsrc(b, self.a["xs"], _c(b, k.nq * k.tiles * TILE * 4))
                self.xs_ptr = b.int_to_ptr(b.readfirstlane(self.a["xs"]), PtrType(I8, "global"))
            else:
                self.xs_rs = _rsrc(b, self.a["xs"], b.mul(rl, _c(b, ksp(k.k))))
        else:
            self.x_rs = _rsrc(b, self.a["x"], b.mul(rl, _c(b, k.k * 2)))
        if k.kseg:
            self.seg_rs = _rsrc(b, self.a["seg"], _c(b, k.kseg * TP * 4))
        self.aux = WCPS[k.wcp]
        # LDS
        if k.bar == "persist":
            self.lds0 = PS.pad_lds(b, name_hint="e1pad") if k.occ_pad() else None
        elif k.occ_pad():
            pad = b.smem_alloc(F32, [ONE_CTA_PER_CU_WORDS], name_hint="occpad")
            a0 = b.trunc(b.smem_addr_of(pad), I32)
            with b.scf_if(D.is_thread0(b)):
                b.inline_asm(
                    "ds_write_b32 $0, $1\ns_waitcnt lgkmcnt(0)",
                    "v,v,~{memory}",
                    [b.add(a0, _c(b, 4 * (ONE_CTA_PER_CU_WORDS - 1))), _c(b, 0)],
                    None,
                )
        rows = k.tbt * TILE
        if k.xsrc == "lds":
            if k.w == "bf16":
                self.xs = b.smem_alloc(BF16, [2 * k.kw, rows + k.xjunk, CHUNK + k.pad // 2], name_hint="xs")
            else:
                self.xs = b.smem_alloc(I8, [2 * k.kw, rows + k.xjunk, k.xrow + k.pad], name_hint="xs8")
                # f32-typed LDS words (loaded with smem_load_vN_f32): an ``av1`` name hint of a 1-element smem_load_vN
                # collides with ``av16`` + counter in kernels this large (duplicate SSA names, invalid IR)
                self.xss = b.smem_alloc(F32, [2, rows * (4 if k.xsb == "byte" else 1)], name_hint="xss")
        if k.splits > 1:
            self.flag = b.smem_alloc(F32, [2 if k.fold == "spread" else 1], name_hint="fold")
        self.gsw = b.smem_alloc(F32, [1], name_hint="gsw")
        if k.kw > 1:
            self.red = b.smem_alloc(F32, [k.kw - 1, k.wr * k.mt * k.tbt * 64 * (2 if k.mfma == 32 else 1), 4], name_hint="kwred")
        if k.yq:
            self.ybuf = b.smem_alloc(F32, [rows, k.rows_per_unit], name_hint="ybuf")
        # this lane's weight row within a unit, per weight tile
        if k.mfma == 32:
            wv = self.wave if k.kw == 1 else self.wi
            self.wrow = [b.add(b.mul(b.add(_c(b, mi * k.wr), wv), _c(b, 32)), self.l32) for mi in range(k.mt)]
        elif k.kw == 1:
            self.wrow = [b.add(b.mul(b.add(_c(b, mi * k.waves), self.wave), _c(b, TILE)), self.l16) for mi in range(k.mt)]
        else:
            self.wrow = [b.add(b.mul(b.add(_c(b, mi * k.wr), self.wi), _c(b, TILE)), self.l16) for mi in range(k.mt)]
        self.sh8g = b.mul(self.g, _c(b, 8))
        if k.diag == "xsk":  # one loop-invariant (opaque) X scale VGPR per token tile: distinct registers, no producer
            self.xsk = [D.fresh(b, _c(b, 120 + (t % 4))) for t in range(k.tbt)]

    # ------------------------------------------------------------------ helpers
    def _stamp(self, col: int, val) -> None:
        b = self.b
        with b.scf_if(D.is_thread0(b)):
            p = b.int_to_ptr(self.a["stamps"], PtrType(I64, "global"))
            v = D.memrealtime(b) if val is None else b.zext(val, I64)
            b.store(v, p, b.add(b.mul(self.cta, _c(b, 4)), _c(b, col)))

    def tst(self, n: Value, p: int) -> None:
        if not self.k.tstamp:
            return
        b = self.b
        with b.scf_if(b.land(D.is_thread0(b), b.cmp_lt(n, _c(b, 4)))):
            ptr = b.int_to_ptr(self.a["stamps"], PtrType(I64, "global"))
            b.store(D.memtime(b), ptr, b.add(b.mul(b.add(b.mul(self.cta, _c(b, 4)), n), _c(b, 64)), _c(b, p)))

    def xbuf(self, parity: Value) -> Value:
        """LDS staging buffer of this wave's K group for ``parity``."""
        if self.k.kw == 1:
            return parity
        return self.b.add(b_mul2(self.b, self.kg), parity)

    # ------------------------------------------------------------------ loads
    def wissue(self, u: _Unit, c: int, xonly_gate=None, xonly: bool = False) -> list:
        """Weight loads of chunk ``c`` of unit ``u`` (and with xsrc=global its X operands). ``sop`` order:
        weight codes, then at a 4-chunk group's first chunk the weight scale dwords, then the X operands (codes when
        xsrc=global, then at the group's first chunk the X scale dwords): the X part is the list's tail."""
        b, k = self.b, self.k
        if k.mfma == 32:
            return self._wissue32(u, c, xonly_gate, xonly)
        q = b.add(u.q0, _c(b, c))
        out = []
        kl = b.add(b.mul(q, _c(b, CHUNK)), b.mul(self.g, _c(b, 32)))  # first K of the lane's 32
        kok = b.cmp_lt(kl, _c(b, k.k)) if k.ragged else None
        for mi in range(0 if xonly else k.mt):
            row = b.add(b.mul(u.rg, _c(b, k.rows_per_unit)), self.wrow[mi])
            ok = b.land(u.valid, b.cmp_lt(row, self.M))
            if k.wlay == "tile":
                tq = b.add(b.mul(b.lshr(row, _c(b, 4)), _c(b, k.nq)), q)  # (row tile, chunk) block
                r16 = b.land(row, _c(b, 15))
            if k.w == "mx":
                okc = ok if kok is None else b.land(ok, kok)
                if k.wlay == "tile":
                    off = b.add(b.add(b.mul(tq, _c(b, 1024)), b.mul(r16, _c(b, 64))), b.mul(self.g, _c(b, 16)))
                else:
                    off = b.add(b.mul(row, _c(b, k.k // 2)), b.lshr(kl, _c(b, 1)))
                out.append(b.buffer_load_cp(u.w_rs, b.select(okc, off, _c(b, OOB)), _c(b, 0), dwords=4, aux=self.aux))
                if k.sop:
                    continue  # the scales of 4 chunks come after the codes (below)
                if k.wlay == "tile":
                    soff = b.add(b.mul(tq, _c(b, 64)), b.mul(r16, _c(b, 4)))
                else:
                    soff = b.add(b.mul(row, _c(b, ksp(k.k))), b.mul(q, _c(b, 4)))
                if k.diag == "nows":
                    out.append(_c(b, 0x7F7F7F7F))
                elif k.wsb == "byte":  # the lane's own block byte; invalid rows read byte 0 (their codes are 0)
                    out.append(b.zext(b.global_load_i8(u.s_ptr, b.select(ok, b.add(soff, self.g), _c(b, 0))), I32))
                else:
                    out.append(b.buffer_load_cp(u.s_rs, b.select(ok, soff, _c(b, OOB)), _c(b, 0), dwords=1, aux=self.aux))
            else:
                base = b.add(b.mul(row, _c(b, k.k)), kl)
                if k.wlay == "tile":
                    tbase = b.add(b.add(b.mul(tq, _c(b, 4096)), b.mul(r16, _c(b, 256))), b.mul(self.g, _c(b, 64)))
                for m in range(4):
                    okm = ok
                    if k.ragged:
                        okm = b.land(ok, b.cmp_lt(b.add(kl, _c(b, 8 * m)), _c(b, k.k)))
                    if k.wlay == "tile":
                        off = b.add(tbase, _c(b, 16 * m))
                    else:
                        off = b.mul(b.add(base, _c(b, 8 * m)), _c(b, 2))
                    out.append(b.buffer_load_cp(u.w_rs, b.select(okm, off, _c(b, OOB)), _c(b, 0), dwords=4, aux=self.aux))
        if k.sop:
            nq4 = -(-k.nq // 4)
            grps = []
            if c % 4 == 0:
                grp = b.lshr(q, _c(b, 2))
                grps = [grp, b.add(grp, _c(b, 1))] if k.sop_funnel else [grp]
                for gq in grps:
                    for mi in range(0 if xonly else k.mt):
                        row = b.add(b.mul(u.rg, _c(b, k.rows_per_unit)), self.wrow[mi])
                        ok = b.land(u.valid, b.cmp_lt(row, self.M))
                        if gq is not grp:
                            ok = b.land(ok, b.cmp_lt(gq, _c(b, nq4)))
                        blk = b.add(b.mul(b.lshr(row, _c(b, 4)), _c(b, nq4)), gq)
                        soff = b.add(b.add(b.mul(blk, _c(b, 256)), b.mul(b.land(row, _c(b, 15)), _c(b, 16))), b.mul(self.g, _c(b, 4)))
                        out.append(b.buffer_load_cp(u.s_rs, b.select(ok, soff, _c(b, OOB)), _c(b, 0), dwords=1, aux=self.aux))
            if c % 4 and k.xsrc != "global":
                return out
            gate = self.xgate(u) if xonly_gate is None else xonly_gate
            if k.xsrc == "global":
                out += self.xglobal(u, q, gate)
            if c % 4:
                return out
            for gq in grps:
                gg = gate if gq is grp else b.land(gate, b.cmp_lt(gq, _c(b, nq4)))
                for t in range(k.tbt):
                    tok = b.add(u.tok0, b.add(_c(b, t * TILE), self.l16))
                    xoff = b.mul(b.add(b.mul(b.add(b.mul(gq, _c(b, k.tiles * TILE)), tok), _c(b, 4)), self.g), _c(b, 4))
                    out.append(b.buffer_load_cp(u.xs_rs, b.select(gg, xoff, _c(b, OOB)), _c(b, 0), dwords=1))
            return out
        if k.xsrc == "global":
            out += self.xglobal(u, q, self.xgate(u) if xonly_gate is None else xonly_gate)
        if k.xscale == "glb":
            gate = self.xgate(u) if xonly_gate is None else xonly_gate
            for t in range(k.tbt):
                tok = b.add(u.tok0, b.add(_c(b, t * TILE), self.l16))
                soff = b.mul(b.add(b.mul(q, _c(b, k.tiles * TILE)), tok), _c(b, 4))
                if k.xsb == "dword":  # [nq][bucket16][4] dwords: the lane's block scale, no VALU op before the MFMA
                    doff = b.mul(b.add(b.mul(b.add(b.mul(q, _c(b, k.tiles * TILE)), tok), _c(b, 4)), self.g), _c(b, 4))
                    out.append(b.buffer_load_cp(u.xs_rs, b.select(gate, doff, _c(b, OOB)), _c(b, 0), dwords=1))
                elif k.xsb == "byte":  # gated loads read byte 0 (a valid scale; re-issued at the crossing)
                    out.append(b.zext(b.global_load_i8(self.xs_ptr, b.select(gate, b.add(soff, self.g), _c(b, 0))), I32))
                else:
                    out.append(b.buffer_load_cp(u.xs_rs, b.select(gate, soff, _c(b, OOB)), _c(b, 0), dwords=1))
        return out

    def xgate(self, u: _Unit) -> Value:
        """X of unit ``u`` may be loaded now: no grid sync, or ``u`` is in the current unit's layer (an activation of a
        later layer exists only after the barrier; those loads go OOB and are re-issued at the crossing)."""
        b = self.b
        return b.lor(self.nogs, b.cmp_eq(u.layer, self.cur_layer))

    def xglobal(self, u: _Unit, q: Value, gate=None) -> list:
        """xsrc=global: this lane's B operands of every tile of chunk ``q`` (the LDS layout's values); with ``sop`` the
        codes only (fp8: two 16-byte halves 64 K apart; fp4: one 16-byte load of K ``32 g .. 32 g + 31``)."""
        b, k = self.b, self.k
        out = []
        for t in range(k.tbt):
            tok = b.add(u.tok0, b.add(_c(b, t * TILE), self.l16))
            tok = b.smin(tok, self.rlast)
            if k.w == "mx" and k.fp4x:
                kk = b.add(b.mul(q, _c(b, CHUNK)), b.mul(self.g, _c(b, 32)))
                ok = b.cmp_lt(kk, _c(b, k.k))
                if gate is not None:
                    ok = b.land(ok, gate)
                off = b.select(ok, b.add(b.mul(tok, _c(b, k.k // 2)), b.lshr(kk, _c(b, 1))), _c(b, OOB))
                out.append(b.buffer_load_cp(u.x_rs, off, _c(b, 0), dwords=4))
            elif k.w == "mx":
                for h in range(2):
                    kk = b.add(b.mul(q, _c(b, CHUNK)), b.add(b.mul(self.g, _c(b, 16)), _c(b, 64 * h)))
                    ok = b.cmp_lt(kk, _c(b, k.k))
                    if gate is not None:
                        ok = b.land(ok, gate)
                    off = b.select(ok, b.add(b.mul(tok, _c(b, k.k)), kk), _c(b, OOB))
                    out.append(b.buffer_load_cp(u.x_rs, off, _c(b, 0), dwords=4))
                if k.sop:
                    continue
                soff = b.add(b.mul(tok, _c(b, ksp(k.k))), b.mul(q, _c(b, 4)))
                if gate is not None:
                    soff = b.select(gate, soff, _c(b, OOB))
                out.append(b.buffer_load_cp(u.xs_rs, soff, _c(b, 0), dwords=1))
            else:
                for m in range(4):
                    kk = b.add(b.mul(q, _c(b, CHUNK)), b.add(b.mul(self.g, _c(b, 32)), _c(b, 8 * m)))
                    ok = b.cmp_lt(kk, _c(b, k.k))
                    if gate is not None:
                        ok = b.land(ok, gate)
                    off = b.select(ok, b.mul(b.add(b.mul(tok, _c(b, k.k)), kk), _c(b, 2)), _c(b, OOB))
                    out.append(b.buffer_load_cp(u.x_rs, off, _c(b, 0), dwords=4))
        return out

    def _xthreads(self):
        """(thread index within the staging group, threads of the group): kw groups each stage their own chunk."""
        k = self.k
        if k.kw == 1:
            return self.tid, k.threads
        return self.gtid, 64 * k.wr

    def xissue(self, u: _Unit, c: int, gate=None) -> list:
        """xsrc=lds: this thread's share of the activation chunk ``c`` of unit ``u`` (codes, then scales)."""
        b, k = self.b, self.k
        q = b.add(u.q0, _c(b, c))
        rows = k.tbt * TILE
        per = 16 if k.w == "bf16" else (4 if k.fp4x else 8)  # 16-byte pieces per token
        total = rows * per
        tid, nthr = self._xthreads()
        out = []
        for it in range(-(-total // nthr)):
            v = b.add(tid, _c(b, it * nthr))
            live = b.cmp_lt(v, _c(b, total))
            v = b.select(live, v, _c(b, total - 1))
            tl = _udiv(b, v, per)
            tok = b.smin(b.add(u.tok0, tl), self.rlast)
            if k.fp4x:  # piece p holds K 32 p .. 32 p + 31 (16 bytes)
                kk = b.add(b.mul(q, _c(b, CHUNK)), b.mul(_umod(b, v, per), _c(b, 32)))
            else:
                kk = b.add(b.mul(q, _c(b, CHUNK)), b.mul(_umod(b, v, per), _c(b, 16 // (2 if k.w == "bf16" else 1))))
            ok = b.cmp_lt(kk, _c(b, k.k)) if k.ragged else None
            g_ = self.xgate(u) if gate is None else gate
            ok = g_ if ok is None else b.land(ok, g_)
            if k.fp4x:
                off = b.add(b.mul(tok, _c(b, k.k // 2)), b.lshr(kk, _c(b, 1)))
            else:
                eb = 2 if k.w == "bf16" else 1
                off = b.mul(b.add(b.mul(tok, _c(b, k.k)), kk), _c(b, eb)) if eb > 1 else b.add(b.mul(tok, _c(b, k.k)), kk)
            if ok is not None:
                off = b.select(ok, off, _c(b, OOB))
            out.append(b.buffer_load_cp(u.x_rs, off, _c(b, 0), dwords=4))
        if k.w == "mx" and k.diag not in ("noxs", "xsk") and k.xslay == "chunk" and k.xscale == "lds":
            n4 = rows // 4
            for it in range(-(-n4 // k.threads)):
                v = b.add(self.tid, _c(b, it * k.threads))
                ok = b.land(b.cmp_lt(v, _c(b, n4)), self.xgate(u) if gate is None else gate)
                soff = b.mul(b.add(b.add(b.mul(q, _c(b, k.tiles * TILE)), u.tok0), b.mul(v, _c(b, 4))), _c(b, 4))
                out.append(b.buffer_load_cp(u.xs_rs, b.select(ok, soff, _c(b, OOB)), _c(b, 0), dwords=4))
        elif k.w == "mx" and k.diag not in ("noxs", "xsk") and k.xscale == "lds":
            for it in range(-(-rows // k.threads)):
                v = b.add(self.tid, _c(b, it * k.threads))
                vv = b.smin(v, _c(b, rows - 1))
                tok = b.smin(b.add(u.tok0, vv), self.rlast)
                soff = b.add(b.mul(tok, _c(b, ksp(k.k))), b.mul(q, _c(b, 4)))
                soff = b.select(self.xgate(u) if gate is None else gate, soff, _c(b, OOB))
                out.append(b.buffer_load_cp(u.xs_rs, soff, _c(b, 0), dwords=1))
        return out

    def xput(self, regs: list, buf: Value) -> None:
        """Write a staged activation chunk (``xissue``'s registers) to LDS buffer ``buf`` (the parity; kw > 1: this
        thread's group buffer is derived here)."""
        b, k = self.b, self.k
        rows = k.tbt * TILE
        per = 16 if k.w == "bf16" else (4 if k.fp4x else 8)
        total = rows * per
        tid, nthr = self._xthreads()
        n_it = -(-total // nthr)
        bufg = self.xbuf(buf)
        for it in range(n_it):
            v = b.add(tid, _c(b, it * nthr))
            live = b.cmp_lt(v, _c(b, total))
            tl = _udiv(b, v, per)
            col = b.mul(_umod(b, v, per), _c(b, 8 if k.w == "bf16" else 16))
            r = regs[it]
            pred = live if total % nthr else None
            if pred is not None and k.xjunk:  # no branch: the idle threads' pieces land in the junk row
                tl = b.select(live, tl, _c(b, rows))
                pred = None
            with _maybe_if(b, pred):
                if k.w == "bf16":
                    b.smem_store_vN(self.xs, [bufg, tl, col], b.vec_bitcast(r, VectorType(BF16, 8)), 8)
                else:
                    b.smem_store_vN(self.xs, [bufg, tl, col], b.vec_bitcast(r, VectorType(I8, 16)), 16)
        if k.w == "mx" and k.diag not in ("noxs", "xsk") and k.xslay == "chunk" and k.xscale == "lds":
            n4 = rows // 4
            for j, it in enumerate(range(-(-n4 // k.threads))):
                v = b.add(self.tid, _c(b, it * k.threads))
                r = b.vec_bitcast(regs[n_it + j], VectorType(F32, 4))
                with _maybe_if(b, b.cmp_lt(v, _c(b, n4)) if n4 % k.threads else None):
                    if k.xsb == "byte":
                        for e in range(4):
                            self._xs_bytes(buf, b.add(b.mul(v, _c(b, 4)), _c(b, e)), b.bitcast(b.vec_extract(r, e), I32))
                    else:
                        b.smem_store_vN(self.xss, [buf, b.mul(v, _c(b, 4))], r, 4)
        elif k.w == "mx" and k.diag not in ("noxs", "xsk") and k.xscale == "lds":
            for j, it in enumerate(range(-(-rows // k.threads))):
                v = b.add(self.tid, _c(b, it * k.threads))
                r = regs[n_it + j]
                with _maybe_if(b, b.cmp_lt(v, _c(b, rows)) if rows % k.threads else None):
                    if k.xsb == "byte":
                        self._xs_bytes(buf, v, r)
                    else:
                        b.smem_store_vN(self.xss, [buf, v], b.bitcast(r, F32), 1)

    def _xs_bytes(self, buf: Value, tok: Value, w: Value) -> None:
        """xsb=byte: token ``tok``'s 4 block scales (bytes of ``w``) as 4 LDS dwords (16 bytes, one store)."""
        b = self.b
        parts = [b.land(b.lshr(w, _c(b, 8 * e)), _c(b, 255)) for e in range(4)]
        vec = b.vec_bitcast(b.vec_pack(parts, I32), VectorType(F32, 4))
        b.smem_store_vN(self.xss, [buf, b.mul(tok, _c(b, 4))], vec, 4)

    # ------------------------------------------------------------------ math
    def bop(self, t: int, buf: Value | None, xg: list | None, m: int = 0):
        """B operand of token tile ``t`` (mx: (codes <8 x i32> / fp4 <4 x i32>, scale dword shifted); bf16: slice ``m``)."""
        b, k = self.b, self.k
        if k.xsrc == "global":
            if k.w == "mx":
                if k.sop:
                    if k.fp4x:
                        return xg[t], None
                    return b.vec_concat(xg[2 * t], xg[2 * t + 1]), None
                base = 3 * t
                v = b.vec_concat(xg[base], xg[base + 1])
                return v, b.lshr(xg[base + 2], self.sh8g)
            return b.vec_bitcast(xg[4 * t + m], VectorType(BF16, 8))
        row = b.add(_c(b, t * TILE), self.l16)
        bufg = self.xbuf(buf)
        if k.w == "mx":
            if k.fp4x:
                hv = b.smem_load_vN(self.xs, bufg, row, b.mul(self.g, _c(b, 16)), dtype=I8, n=16)
                return b.vec_bitcast(hv, VectorType(I32, 4)), None
            halves = []
            for h in range(2):
                col = b.add(b.mul(self.g, _c(b, 16)), _c(b, 64 * h))
                hv = b.smem_load_vN(self.xs, bufg, row, col, dtype=I8, n=16)
                halves.append(b.vec_bitcast(hv, VectorType(I32, 4)))
            if k.diag == "noxs":
                return b.vec_concat(halves[0], halves[1]), _c(b, 127)
            if k.diag == "xsk":
                return b.vec_concat(halves[0], halves[1]), self.xsk[t]
            if k.sop:
                return b.vec_concat(halves[0], halves[1]), None
            if k.xscale == "glb":
                return b.vec_concat(halves[0], halves[1]), (xg[t] if k.xsb in ("byte", "dword") else b.lshr(xg[t], self.sh8g))
            if k.diag == "xs0":
                row = self.l16
            if k.xsb == "byte":
                sw = b.bitcast(b.vec_extract(b.smem_load_vN_f32(self.xss, buf, b.add(b.mul(row, _c(b, 4)), self.g), n=1), 0), I32)
                return b.vec_concat(halves[0], halves[1]), sw
            sw = b.bitcast(b.vec_extract(b.smem_load_vN_f32(self.xss, buf, row, n=1), 0), I32)
            if k.diag == "xsu":
                return b.vec_concat(halves[0], halves[1]), b.readfirstlane(b.lshr(sw, self.sh8g))
            return b.vec_concat(halves[0], halves[1]), b.lshr(sw, self.sh8g)
        col = b.add(b.mul(self.g, _c(b, 32)), _c(b, 8 * m))
        return b.smem_load_vN(self.xs, bufg, row, col, dtype=BF16, n=8)

    def chunk_math(self, acc, wl: list, buf: Value | None, skip: bool, j: int = 0, u: _Unit | None = None) -> list:
        b, k = self.b, self.k
        if k.mfma == 32:
            return self._sop_math32(acc, wl, buf, skip, j, u)
        if k.sop:
            return self._sop_math(acc, wl, buf, skip, j, u)
        xg = None
        if k.xsrc == "global" or k.xscale == "glb":
            nw = k.mt * (2 if k.w == "mx" else 4)
            xg = wl[nw:]
        acc = [list(r) for r in acc]

        def ops_of(t):
            if k.diag == "nob":
                return None
            if k.w == "mx":
                return self.bop(t, buf, xg)
            return [self.bop(t, buf, xg, m) for m in range(4)]

        ahead = {}
        if k.bpre:
            for t in range(k.bpre):
                ahead[t] = ops_of(t)
            if k.sb:
                b.sched_barrier(0)
        if k.w == "mx":
            A, SA = [], []
            for mi in range(k.mt):
                A.append(wl[2 * mi])
                sw = wl[2 * mi + 1]
                SA.append(sw if (k.mutant == "scale" or k.wsb == "byte") else b.lshr(sw, self.sh8g))
        for t in range(k.tbt):
            if k.bpre:
                if t + k.bpre < k.tbt:
                    ahead[t + k.bpre] = ops_of(t + k.bpre)
                cur = ahead.pop(t)
            else:
                cur = ops_of(t)
            # mutant kskip: the chunk's weights are replaced by zeros (every value stays live)
            zw = b.vec_pack([_c(b, 0)] * 4, I32) if skip else None
            if k.diag in ("nomfma", "nob"):
                acc = self._diag_math(acc, t, cur, wl, SA if k.w == "mx" else None)
                continue
            if k.w == "mx":
                bv, bs = cur
                for mi in range(k.mt):
                    acc[mi][t] = b.mfma_scale_mx(
                        zw if skip else A[mi], bv, acc[mi][t], SA[mi], bs, shape="16x16x128", fmt_a="fp4", fmt_b="fp8"
                    )
            else:
                for m in range(4):
                    for mi in range(k.mt):
                        a = b.vec_bitcast(zw if skip else wl[4 * mi + m], VectorType(BF16, 8))
                        acc[mi][t] = b.mfma_f32_16x16x32_bf16(a, cur[m], acc[mi][t])
            if k.bpre and k.sb:
                b.sched_barrier(0)
        return acc

    def _sop_math(self, acc, wl, buf, skip, j, u):
        """sop: the chunk's weight codes (one per weight tile), the current 4-chunk scale dwords (``self.sop_sc``:
        weight tiles then token tiles; shifted by ``8 (q0 mod 4)`` once per short unit), byte ``j`` by op_sel."""
        b, k = self.b, self.k
        acc = [list(r) for r in acc]
        nx = (k.tbt if k.fp4x else 2 * k.tbt) if k.xsrc == "global" else 0
        if j == 0:  # first chunk of a 4-chunk group (or of a short unit): new scale dwords
            nws = k.mt * (2 if k.sop_funnel else 1)
            ws = wl[k.mt : k.mt + nws]
            xs = wl[k.mt + nws + nx :]
            if k.sop_funnel:  # (group G, group G + 1) pairs -> the 4 bytes from chunk q0 + c on
                sh = b.mul(b.land(u.q0, _c(b, 3)), _c(b, 8))
                sc = [_funnel(b, ws[i], ws[k.mt + i], sh) for i in range(k.mt)]
                sc += [_funnel(b, xs[t], xs[k.tbt + t], sh) for t in range(k.tbt)]
            else:
                sc = ws + xs
            if k.sop_short:
                sh = b.mul(b.land(u.q0, _c(b, 3)), _c(b, 8))
                sc = [b.lshr(v, sh) for v in sc]
            self.sop_sc = sc
            xg = wl[k.mt + nws : k.mt + nws + nx]
        else:
            xg = wl[k.mt : k.mt + nx]
        SA = self.sop_sc[: k.mt]
        SB = self.sop_sc[k.mt :]
        zw = b.vec_pack([_c(b, 0)] * 4, I32) if skip else None
        js = (j + 1) % 4 if k.mutant == "sopsel" else j  # mutant sopsel: the next chunk's scale byte
        ahead = {}
        nb = k.bpre or 0
        for t in range(nb):
            ahead[t] = self.bop(t, buf, xg)
        fb = "fp4" if k.fp4x else "fp8"
        for t in range(k.tbt):
            if nb:
                if t + nb < k.tbt:
                    ahead[t + nb] = self.bop(t + nb, buf, xg)
                bv = ahead.pop(t)[0]
            else:
                bv = self.bop(t, buf, xg)[0]
            for mi in range(k.mt):
                acc[mi][t] = b.mfma_scale_mx(
                    zw if skip else wl[mi], bv, acc[mi][t], SA[mi], SB[t], shape="16x16x128", fmt_a="fp4", fmt_b=fb,
                    opsel_a=js, opsel_b=js,
                )
        return acc

    # ------------------------------------------------------------------ mfma=32 (32x32x64 scaled MFMA)
    def _wissue32(self, u: _Unit, c: int, xonly_gate=None, xonly: bool = False) -> list:
        """mfma=32: lane (row l mod 32 of the 32-row tile, half h) needs, per 64-K step j of the chunk, K
        ``64 j + 32 h .. + 31`` of its row: 16 bytes at ``32 j + 16 h`` of the row's 64-byte chunk slice in the tile layout
        (two 16-row tiles per 32 rows). Scales: block ``2 j + h`` of the chunk, from the same op_sel dwords. Order: codes
        [mi][j], scales [group][mi][j], X codes (global) [t2][j][half], X scales [group][t2][j]."""
        b, k = self.b, self.k
        q = b.add(u.q0, _c(b, c))
        nq4 = -(-k.nq // 4)
        out = []
        rows_ = []
        for mi in range(0 if xonly else k.mt):
            row = b.add(b.mul(u.rg, _c(b, k.rows_per_unit)), self.wrow[mi])
            ok = b.land(u.valid, b.cmp_lt(row, self.M))
            tq = b.add(b.mul(b.lshr(row, _c(b, 4)), _c(b, k.nq)), q)
            r16 = b.land(row, _c(b, 15))
            rows_.append((row, ok, r16))
            for j in range(2):
                okj = ok
                if k.ragged:
                    kk = b.add(b.mul(q, _c(b, CHUNK)), b.add(_c(b, 64 * j), b.mul(self.h, _c(b, 32))))
                    okj = b.land(ok, b.cmp_lt(kk, _c(b, k.k)))
                off = b.add(b.add(b.mul(tq, _c(b, 1024)), b.mul(r16, _c(b, 64))), b.add(_c(b, 32 * j), b.mul(self.h, _c(b, 16))))
                out.append(b.buffer_load_cp(u.w_rs, b.select(okj, off, _c(b, OOB)), _c(b, 0), dwords=4, aux=self.aux))
        grps = []
        if c % 4 == 0:
            grp = b.lshr(q, _c(b, 2))
            grps = [grp, b.add(grp, _c(b, 1))] if k.sop_funnel else [grp]
            for gq in grps:
                for (row, ok, r16) in rows_:
                    okg = ok if gq is grp else b.land(ok, b.cmp_lt(gq, _c(b, nq4)))
                    blk = b.add(b.mul(b.lshr(row, _c(b, 4)), _c(b, nq4)), gq)
                    for j in range(2):
                        gi = b.add(self.h, _c(b, 2 * j))
                        soff = b.add(b.add(b.mul(blk, _c(b, 256)), b.mul(r16, _c(b, 16))), b.mul(gi, _c(b, 4)))
                        out.append(b.buffer_load_cp(u.s_rs, b.select(okg, soff, _c(b, OOB)), _c(b, 0), dwords=1, aux=self.aux))
        if c % 4 and k.xsrc != "global":
            return out
        gate = self.xgate(u) if xonly_gate is None else xonly_gate
        if k.xsrc == "global":
            out += self._xglobal32(u, q, gate)
        if c % 4:
            return out
        for gq in grps:
            gg = gate if gq is grp else b.land(gate, b.cmp_lt(gq, _c(b, nq4)))
            for t2 in range(k.ntt):
                tok = b.add(u.tok0, b.add(_c(b, 32 * t2), self.l32))
                for j in range(2):
                    gi = b.add(self.h, _c(b, 2 * j))
                    xoff = b.mul(b.add(b.mul(b.add(b.mul(gq, _c(b, k.tiles * TILE)), tok), _c(b, 4)), gi), _c(b, 4))
                    out.append(b.buffer_load_cp(u.xs_rs, b.select(gg, xoff, _c(b, OOB)), _c(b, 0), dwords=1))
        return out

    def _xglobal32(self, u: _Unit, q: Value, gate) -> list:
        """mfma=32, xsrc=global: B codes of token tile ``t2`` (token l mod 32), K step j: fp8 = two 16-byte halves at chunk
        bytes ``64 j + 16 h`` and ``64 j + 32 + 16 h``; fp4 = 16 bytes at ``32 j + 16 h``."""
        b, k = self.b, self.k
        out = []
        for t2 in range(k.ntt):
            tok = b.smin(b.add(u.tok0, b.add(_c(b, 32 * t2), self.l32)), self.rlast)
            for j in range(2):
                if k.fp4x:
                    kk = b.add(b.mul(q, _c(b, CHUNK)), b.add(_c(b, 64 * j), b.mul(self.h, _c(b, 32))))
                    ok = b.land(b.cmp_lt(kk, _c(b, k.k)), gate)
                    off = b.select(ok, b.add(b.mul(tok, _c(b, k.k // 2)), b.lshr(kk, _c(b, 1))), _c(b, OOB))
                    out.append(b.buffer_load_cp(u.x_rs, off, _c(b, 0), dwords=4))
                else:
                    for hh in range(2):
                        kk = b.add(b.mul(q, _c(b, CHUNK)), b.add(_c(b, 64 * j + 32 * hh), b.mul(self.h, _c(b, 16))))
                        ok = b.land(b.cmp_lt(kk, _c(b, k.k)), gate)
                        off = b.select(ok, b.add(b.mul(tok, _c(b, k.k)), kk), _c(b, OOB))
                        out.append(b.buffer_load_cp(u.x_rs, off, _c(b, 0), dwords=4))
        return out

    def _bop32(self, t2: int, j: int, buf: Value | None, xg: list):
        b, k = self.b, self.k
        if k.xsrc == "global":
            if k.fp4x:
                return xg[t2 * 2 + j]
            return b.vec_concat(xg[(t2 * 2 + j) * 2], xg[(t2 * 2 + j) * 2 + 1])
        row = b.add(_c(b, 32 * t2), self.l32)
        bufg = self.xbuf(buf)
        if k.fp4x:
            hv = b.smem_load_vN(self.xs, bufg, row, b.add(_c(b, 32 * j), b.mul(self.h, _c(b, 16))), dtype=I8, n=16)
            return b.vec_bitcast(hv, VectorType(I32, 4))
        halves = []
        for hh in range(2):
            col = b.add(_c(b, 64 * j + 32 * hh), b.mul(self.h, _c(b, 16)))
            hv = b.smem_load_vN(self.xs, bufg, row, col, dtype=I8, n=16)
            halves.append(b.vec_bitcast(hv, VectorType(I32, 4)))
        return b.vec_concat(halves[0], halves[1])

    def _sop_math32(self, acc, wl, buf, skip, jc, u):
        """mfma=32: per chunk two K steps (j = 0, 1, in order) of ``v_mfma_scale_f32_32x32x64_f8f6f4`` per (weight tile,
        32-token tile); scale byte ``jc`` (the chunk within its 4-chunk group) by op_sel."""
        b, k = self.b, self.k
        acc = [list(r) for r in acc]
        mt, nt = k.mt, k.ntt
        nW = 2 * mt
        nx = (nt * 2 * (1 if k.fp4x else 2)) if k.xsrc == "global" else 0
        if jc == 0:
            ng = 2 if k.sop_funnel else 1
            nws = ng * mt * 2
            ws = wl[nW : nW + nws]
            xg = wl[nW + nws : nW + nws + nx]
            xs = wl[nW + nws + nx :]
            wsw = {(mi, j): [ws[(g * mt + mi) * 2 + j] for g in range(ng)] for mi in range(mt) for j in range(2)}
            xsw = {(t2, j): [xs[(g * nt + t2) * 2 + j] for g in range(ng)] for t2 in range(nt) for j in range(2)}
            sh = b.mul(b.land(u.q0, _c(b, 3)), _c(b, 8)) if (k.sop_funnel or k.sop_short) else None

            def fix(v):
                if k.sop_funnel:
                    return _funnel(b, v[0], v[1], sh)
                if k.sop_short:
                    return b.lshr(v[0], sh)
                return v[0]

            self.sop_sc32 = ({key: fix(v) for key, v in wsw.items()}, {key: fix(v) for key, v in xsw.items()})
        else:
            xg = wl[nW : nW + nx]
        SA, SB = self.sop_sc32
        zw = b.vec_pack([_c(b, 0)] * 4, I32) if skip else None
        js = (jc + 1) % 4 if k.mutant == "sopsel" else jc
        fb = "fp4" if k.fp4x else "fp8"
        for t2 in range(nt):
            for j in range(2):
                bv = self._bop32(t2, j, buf, xg)
                for mi in range(mt):
                    acc[mi][t2] = b.mfma_scale_mx(
                        zw if skip else wl[mi * 2 + j], bv, acc[mi][t2], SA[(mi, j)], SB[(t2, j)], shape="32x32x64",
                        fmt_a="fp4", fmt_b=fb, opsel_a=js, opsel_b=js,
                    )
        return acc

    def _diag_math(self, acc, t, cur, A, SA):
        """diag=nomfma / nob (gemm.py): wrong results by design (timing diagnosis only)."""
        b, k = self.b, self.k
        if k.diag == "nob":
            if k.w == "mx":
                bv = b.vec_pack([_c(b, 0x38383838)] * 8, I32)
                bs = _c(b, 127)
                for mi in range(k.mt):
                    acc[mi][t] = b.mfma_scale_mx(A[2 * mi], bv, acc[mi][t], SA[mi], bs, shape="16x16x128", fmt_a="fp4", fmt_b="fp8")
            else:
                bv = b.vec_bitcast(b.vec_pack([_c(b, 0x3F803F80)] * 4, I32), VectorType(BF16, 8))
                for m in range(4):
                    for mi in range(k.mt):
                        a = b.vec_bitcast(A[4 * mi + m], VectorType(BF16, 8))
                        acc[mi][t] = b.mfma_f32_16x16x32_bf16(a, bv, acc[mi][t])
            return acc
        if k.w == "mx":
            bv, bs = cur
            x = b.xor(b.xor(b.vec_extract(bv, 0), b.vec_extract(bv, 7)), bs)
        else:
            x = b.vec_extract(b.vec_bitcast(cur[0], VectorType(I32, 4)), 0)
            for m in range(1, 4):
                x = b.xor(x, b.vec_extract(b.vec_bitcast(cur[m], VectorType(I32, 4)), 3))
        for mi in range(k.mt):
            w0 = b.vec_extract(A[2 * mi] if k.w == "mx" else A[4 * mi], 0)
            if k.w == "mx":
                w0 = b.xor(w0, SA[mi])
            f = b.bitcast(b.land(b.xor(x, w0), _c(b, 0x3FFFFFFF)), F32)
            acc[mi][t] = b.vec_insert(acc[mi][t], b.fadd(b.vec_extract(acc[mi][t], 0), f), 0)
        return acc

    # ------------------------------------------------------------------ grid sync (phase metric)
    def gbar(self, bnd: Value) -> None:
        if self.k.bar == "persist":
            self._gbar_persist(bnd)
        else:
            self._gbar_probe(bnd)
        if self.k.chain and self.k.mutant != "noacq":
            D.acquire_l2(self.b, "agent")  # layer l's X (layer l - 1's yq stores) are on other XCDs' lines

    def _gbar_probe(self, bnd: Value) -> None:
        """P-L3's grid barrier at layer boundary ``bnd`` (gemm.py ``gbar``; the chain mutants as in the persist form)."""
        b, k = self.b, self.k
        from kernels.gfx950.qwen38_mono.ccl import wire as W

        per = k.grid // XCDS
        b.s_waitcnt(vmcnt=0)
        b.sync_lds_only()
        gsa = b.readfirstlane(self.a["gs"])
        row = b.mul(b.add(_c(b, 1), bnd), _c(b, 8))
        with b.scf_if(D.is_thread0(b)):
            mine = b.mul(b.add(row, _umod(b, self.cta, XCDS)), _c(b, 4))
            old = D.atomic_add_scoped(b, gsa, mine, _c(b, 1), scope="agent", returns=True)
            target = b.mul(b.add(b.div(old, _c(b, per)), _c(b, 1)), _c(b, per))
            b.smem_store_vN(self.gsw, [_c(b, 0)], b.bitcast(target, F32), 1)
        if k.mutant != "barnone":
            # lanes 0..7 of wave 0 poll the 8 stripes together (one load round per poll iteration)
            with b.scf_if(b.cmp_lt(self.tid, _c(b, XCDS))):
                b.s_waitcnt(lgkmcnt=0)
                target = b.bitcast(b.vec_extract(b.smem_load_vN_f32(self.gsw, _c(b, 0), n=1), 0), I32)
                if k.mutant == "bardrop":
                    target = b.sub(target, b.select(b.cmp_eq(self.tid, _c(b, 0)), _c(b, 1), _c(b, 0)))
                dl = b.add(D.memrealtime(b), b.const_i64(200_000_000))
                st = W.poll_count_ge(b, gsa, b.mul(b.add(row, self.tid), _c(b, 4)), target, dl, pause="s_sleep1")
                with b.scf_if(b.land(b.cmp_ne(st, _c(b, 0)), D.is_thread0(b))):
                    D.atomic_add_scoped(b, gsa, _c(b, 0), _c(b, 1), scope="agent")
        b.sync_lds_only()

    def _gbar_persist(self, bnd: Value) -> None:
        """W5's striped barrier (``persist.sync``): drain, CTA barrier, one non-returning agent-scope arrival per CTA on
        word ``(c mod 8) 8 + (c / 8) mod 8`` (its own 256-byte line), wave 0 polls all 64 words (lane w: word w) with a
        ballot per retry until each reached ``(bnd + 1) grid / 64``; a wait past ``BAR_MAXIT`` retries adds 1 to word 0."""
        b, k = self.b, self.k
        gs = PS.GridSync(grid=k.grid, stripes=8)
        b.s_waitcnt(vmcnt=0)
        b.sync_lds_only()
        gsa = b.readfirstlane(self.a["gs"])
        with b.scf_if(D.is_thread0(b)):
            woff = b.add(b.mul(gs.word_of(b, self.cta), _c(b, BAR_LINE)), _c(b, BAR_LINE))
            D.atomic_add_scoped(b, gsa, woff, _c(b, 1), scope="agent")
        if k.mutant != "barnone":
            with b.scf_if(D.is_wave0(b)):
                lane = self.lane
                voff = b.add(b.mul(lane, _c(b, BAR_LINE)), _c(b, BAR_LINE))
                exp = b.mul(b.add(bnd, _c(b, 1)), _c(b, gs.count))
                if k.mutant == "bardrop":
                    exp = b.sub(exp, b.select(b.cmp_eq(lane, _c(b, 0)), _c(b, 1), _c(b, 0)))
                _seen, it = PS.poll_counter(b, voff, gsa, exp, _c(b, BAR_MAXIT), "s_nop")
                with b.scf_if(b.land(D.is_thread0(b), b.cmp_ge(it, _c(b, BAR_MAXIT)))):
                    D.atomic_add_scoped(b, gsa, _c(b, 0), _c(b, 1), scope="agent")
        b.sync_lds_only()

    def _bar_end(self) -> None:
        """bar=persist: the launch's last CTA (one returning arrival per CTA after its last wait) resets the counters."""
        b, k = self.b, self.k
        gsa = b.readfirstlane(self.a["gs"])
        endo = BAR_LINE * (BAR_WORDS + 1)
        b.sync_lds_only()
        with b.scf_if(b.land(self.gsync, D.is_thread0(b))):
            old = D.atomic_add_scoped(b, gsa, _c(b, endo), _c(b, 1), scope="agent", returns=True)
            with b.scf_if(b.cmp_eq(old, _c(b, k.grid - 1))):
                rs = b.buffer_rsrc(_ptr(b, gsa), _c(b, BAR_LINE * (BAR_WORDS + 2)))
                for w_ in range(BAR_WORDS):
                    b.buffer_store_cp(_c(b, 0), rs, _c(b, BAR_LINE * (1 + w_)), _c(b, 0), aux=CPOL_SC1)
                b.buffer_store_cp(_c(b, 0), rs, _c(b, endo), _c(b, 0), aux=CPOL_SC1)

    # ------------------------------------------------------------------ epilogue
    def seg_loads(self, u: _Unit) -> list:
        """kseg: this lane's segment scale per token tile (issued at the unit's start, after any crossing)."""
        b, k = self.b, self.k
        if not k.kseg:
            return []
        segs = k.nq // k.kseg
        si = _c(b, 0) if k.mutant == "seg" else b.div(u.q0, _c(b, segs))
        out = []
        for t in range(k.tbt):
            tok = b.add(u.tok0, b.add(_c(b, t * TILE), self.l16))
            off = b.mul(b.add(b.mul(si, _c(b, k.tiles * TILE)), tok), _c(b, 4))
            out.append(b.buffer_load_cp(self.seg_rs, b.select(self.xgate(u), off, _c(b, OOB)), _c(b, 0), dwords=1))
        return out

    def kw_reduce(self, acc) -> list:
        """kw > 1: groups 1..kw-1 put their accumulators in LDS, group 0 adds them in group order."""
        b, k = self.b, self.k
        if k.kw == 1:
            return acc
        if k.mfma == 32:  # reduce the 16-element accumulators as 4 vec4 pieces each, then put them back together
            pieces = [[b.vec_pack([b.vec_extract(acc[mi][t2], 4 * qd + e) for e in range(4)], F32)
                       for t2 in range(k.ntt) for qd in range(4)] for mi in range(k.mt)]
            red = self._kw_reduce4(pieces, k.ntt * 4)
            return [[b.vec_pack([b.vec_extract(red[mi][t2 * 4 + qd], e) for qd in range(4) for e in range(4)], F32)
                     for t2 in range(k.ntt)] for mi in range(k.mt)]
        return self._kw_reduce4(acc, k.tbt)

    def _kw_reduce4(self, acc, nper) -> list:
        """The LDS reduction of ``kw`` groups over ``acc[mi][i]`` vec4 pieces (``nper`` a weight tile)."""
        b, k = self.b, self.k
        n = k.mt * nper
        slot = lambda mi, t: b.add(b.mul(b.add(b.mul(self.wi, _c(b, n)), _c(b, mi * nper + t)), _c(b, 64)), self.lane)  # noqa: E731
        with b.scf_if(b.cmp_ne(self.kg, _c(b, 0))):
            gi = b.sub(self.kg, _c(b, 1))
            for mi in range(k.mt):
                for t in range(nper):
                    b.smem_store_vN(self.red, [gi, slot(mi, t), _c(b, 0)], acc[mi][t], 4)
        b.sync_lds_only()
        out = [list(r) for r in acc]
        top = k.kw - 1 if k.mutant == "kwred" else k.kw
        for gi in range(1, top):
            for mi in range(k.mt):
                for t in range(nper):
                    v = b.smem_load_vN_f32(self.red, _c(b, gi - 1), slot(mi, t), _c(b, 0), n=4)
                    out[mi][t] = b.vec_fadd(out[mi][t], v) if hasattr(b, "vec_fadd") else b.vec_pack(
                        [b.fadd(b.vec_extract(out[mi][t], e), b.vec_extract(v, e)) for e in range(4)], F32)
        b.sync_lds_only()  # the next unit's group > 0 waves may overwrite the buffer only after group 0 read it
        return out

    def epilogue(self, u: _Unit, acc, segv=None) -> None:
        b, k = self.b, self.k
        TP = k.tiles * TILE
        if k.kseg:
            acc = [[b.vec_pack([b.fmul(b.vec_extract(acc[mi][t], e), b.bitcast(segv[t], F32)) for e in range(4)], F32)
                    for t in range(k.tbt)] for mi in range(k.mt)]
        acc = self.kw_reduce(acc)
        g0 = b.cmp_eq(self.kg, _c(b, 0)) if k.kw > 1 else None
        if k.mfma == 32:
            self._epilogue32(u, acc, g0)
            return
        if k.splits == 1:
            y_rs = _rsrc(b, u.y_a, self.yl32)
            for mi in range(k.mt):
                row = b.add(b.add(b.mul(u.rg, _c(b, k.rows_per_unit)), b.sub(self.wrow[mi], self.l16)), b.mul(self.g, _c(b, 4)))
                for t in range(k.tbt):
                    tok = b.add(u.tok0, b.add(_c(b, t * TILE), self.l16))
                    ok = b.land(u.valid, b.cmp_lt(tok, self.rows))
                    if g0 is not None:
                        ok = b.land(ok, g0)
                    off = b.mul(b.add(b.mul(tok, self.Mpad), row), _c(b, 2))
                    v = b.vec_bitcast(b.vec_trunc_f32_to_bf16(acc[mi][t]), VectorType(I32, 2))
                    b.buffer_store_cp(v, y_rs, b.select(ok, off, _c(b, OOB)), _c(b, 0))
                    if k.yq:
                        lrow = b.sub(row, b.mul(u.rg, _c(b, k.rows_per_unit)))
                        with _maybe_if(b, g0):
                            self._ybuf_put(b.add(_c(b, t * TILE), self.l16), lrow, v)
            if k.yq:
                self.yq_pass(u)
            return
        p_rs = _rsrc(b, u.p_a, self.pl32)
        for mi in range(k.mt):
            row = b.add(b.add(b.mul(u.rg, _c(b, k.rows_per_unit)), b.sub(self.wrow[mi], self.l16)), b.mul(self.g, _c(b, 4)))
            for t in range(k.tbt):
                tok = b.add(u.tok0, b.add(_c(b, t * TILE), self.l16))
                ok = b.land(u.valid, b.cmp_lt(tok, self.rows))
                if g0 is not None:
                    ok = b.land(ok, g0)
                off = b.mul(b.add(b.mul(b.add(b.mul(u.s, _c(b, TP)), tok), self.Mpad), row), _c(b, 4))
                v = b.vec_bitcast(acc[mi][t], VectorType(I32, 4))
                b.buffer_store_cp(v, p_rs, b.select(ok, off, _c(b, OOB)), _c(b, 0), aux=CPOL_SC1)
        if k.fold == "none":
            return
        if k.fold == "spread":
            self._fold_spread(u, p_rs)
            return
        # arrive: every wave's partial stores complete (sc1 write-through), then one agent-scope add
        b.s_waitcnt(vmcnt=0)
        b.sync_lds_only()
        idx = b.add(b.mul(u.rg, _c(b, k.ntb)), u.tbk)
        cbase = b.readfirstlane(u.c_a)
        with b.scf_if(D.is_thread0(b)):
            b.smem_store_vN(self.flag, [_c(b, 0)], b.bitcast(_c(b, -1), F32), 1)
            with b.scf_if(u.valid):
                old = D.atomic_add_scoped(b, cbase, b.mul(idx, _c(b, 4)), _c(b, 1), scope="agent", returns=True)
                b.smem_store_vN(self.flag, [_c(b, 0)], b.bitcast(old, F32), 1)
        b.sync_lds_only()
        seen = b.readfirstlane(b.bitcast(b.vec_extract(b.smem_load_vN_f32(self.flag, _c(b, 0), n=1), 0), I32))
        with b.scf_if(b.cmp_eq(seen, _c(b, k.splits - 1))):
            D.acquire_l2(b, "agent")
            self.fold(u, p_rs)
            with b.scf_if(D.is_thread0(b)):
                cp = _ptr(b, cbase)
                b.store(_c(b, 0), cp, idx)
            if k.yq:
                self.yq_pass(u)

    def _epilogue32(self, u: _Unit, acc, g0) -> None:
        """mfma=32: C element i of lane l is (row 8 (i / 4) + 4 h + i mod 4, token l mod 32); stores of 4 consecutive rows
        (bf16 Y at one split, f32 partials otherwise), then the split-K protocol of the 16-shape (``_fold_tail``)."""
        b, k = self.b, self.k
        TP = k.tiles * TILE
        if k.splits == 1:
            dst = _rsrc(b, u.y_a, self.yl32)
        else:
            dst = _rsrc(b, u.p_a, self.pl32)
        for mi in range(k.mt):
            base = b.add(b.mul(u.rg, _c(b, k.rows_per_unit)), b.add(b.sub(self.wrow[mi], self.l32), b.mul(self.h, _c(b, 4))))
            for t2 in range(k.ntt):
                tok = b.add(u.tok0, b.add(_c(b, 32 * t2), self.l32))
                ok = b.land(u.valid, b.cmp_lt(tok, self.rows))
                if g0 is not None:
                    ok = b.land(ok, g0)
                for qd in range(4):
                    row = b.add(base, _c(b, 8 * qd))
                    v4 = b.vec_pack([b.vec_extract(acc[mi][t2], 4 * qd + e) for e in range(4)], F32)
                    if k.splits == 1:
                        off = b.mul(b.add(b.mul(tok, self.Mpad), row), _c(b, 2))
                        v = b.vec_bitcast(b.vec_trunc_f32_to_bf16(v4), VectorType(I32, 2))
                        b.buffer_store_cp(v, dst, b.select(ok, off, _c(b, OOB)), _c(b, 0))
                    else:
                        off = b.mul(b.add(b.mul(b.add(b.mul(u.s, _c(b, TP)), tok), self.Mpad), row), _c(b, 4))
                        b.buffer_store_cp(b.vec_bitcast(v4, VectorType(I32, 4)), dst, b.select(ok, off, _c(b, OOB)), _c(b, 0),
                                          aux=CPOL_SC1)
        if k.splits > 1:
            self._fold_tail(u, dst)

    def _fold_tail(self, u: _Unit, p_rs: Value) -> None:
        """The split-K protocol after the partial stores (fold=none / spread / last), shared by both MFMA shapes."""
        b, k = self.b, self.k
        if k.fold == "none":
            return
        if k.fold == "spread":
            self._fold_spread(u, p_rs)
            return
        b.s_waitcnt(vmcnt=0)
        b.sync_lds_only()
        idx = b.add(b.mul(u.rg, _c(b, k.ntb)), u.tbk)
        cbase = b.readfirstlane(u.c_a)
        with b.scf_if(D.is_thread0(b)):
            b.smem_store_vN(self.flag, [_c(b, 0)], b.bitcast(_c(b, -1), F32), 1)
            with b.scf_if(u.valid):
                old = D.atomic_add_scoped(b, cbase, b.mul(idx, _c(b, 4)), _c(b, 1), scope="agent", returns=True)
                b.smem_store_vN(self.flag, [_c(b, 0)], b.bitcast(old, F32), 1)
        b.sync_lds_only()
        seen = b.readfirstlane(b.bitcast(b.vec_extract(b.smem_load_vN_f32(self.flag, _c(b, 0), n=1), 0), I32))
        with b.scf_if(b.cmp_eq(seen, _c(b, k.splits - 1))):
            D.acquire_l2(b, "agent")
            self.fold(u, p_rs)
            with b.scf_if(D.is_thread0(b)):
                cp = _ptr(b, cbase)
                b.store(_c(b, 0), cp, idx)

    def _fold_spread(self, u: _Unit, p_rs: Value) -> None:
        """fold=spread: every unit arrives (agent-scope add on its group's arrival counter), then, when every CTA holds
        at most one unit of the layer (``U <= grid``, the same for the whole grid: no unit can wait for a unit queued
        behind it), waits until all ``splits`` units arrived, folds slice ``s`` of the group's outputs (split order: the
        bits of ``fold=last``) and departs (add on the group's departure counter, the word ``RG ntb`` after the arrival
        one); the last departer zeroes both words, so the counters are zero at rest, as with ``fold=last``. With more
        units than CTAs the last arriver folds everything and zeroes the arrival word (``fold=last``'s protocol)."""
        b, k = self.b, self.k
        S = k.splits
        b.s_waitcnt(vmcnt=0)
        b.sync_lds_only()
        idx = b.add(b.mul(u.rg, _c(b, k.ntb)), u.tbk)
        didx = b.add(idx, b.mul(self.RG, _c(b, k.ntb)))
        cbase = b.readfirstlane(u.c_a)
        with b.scf_if(D.is_thread0(b)):
            b.smem_store_vN(self.flag, [_c(b, 0)], b.bitcast(_c(b, -1), F32), 1)
            with b.scf_if(u.valid):
                old = D.atomic_add_scoped(b, cbase, b.mul(idx, _c(b, 4)), _c(b, 1), scope="agent", returns=True)
                b.smem_store_vN(self.flag, [_c(b, 0)], b.bitcast(old, F32), 1)
        b.sync_lds_only()
        old = b.readfirstlane(b.bitcast(b.vec_extract(b.smem_load_vN_f32(self.flag, _c(b, 0), n=1), 0), I32))
        valid = b.cmp_ge(old, _c(b, 0))
        spread = b.cmp_le(self.U, _c(b, k.grid))
        with b.scf_if(b.land(valid, spread)):
            tgt = _c(b, 0 if k.mutant == "nowait" else S)  # mutant nowait: no group wait (CTA 0 also starts late)
            with b.scf_if(D.is_thread0(b)):
                _seen, it = PS.poll_counter(b, b.mul(idx, _c(b, 4)), cbase, tgt, _c(b, BAR_MAXIT), "s_nop")
                with b.scf_if(b.cmp_ge(it, _c(b, BAR_MAXIT))):
                    D.atomic_add_scoped(b, b.readfirstlane(self.a["gs"]), _c(b, 0), _c(b, 1), scope="agent")
            b.sync_lds_only()
            D.acquire_l2(b, "agent")
            self.fold(u, p_rs, part=u.s)
            with b.scf_if(D.is_thread0(b)):
                dep = D.atomic_add_scoped(b, cbase, b.mul(didx, _c(b, 4)), _c(b, 1), scope="agent", returns=True)
                with b.scf_if(b.cmp_eq(dep, _c(b, S - 1))):  # every unit of the group passed its wait
                    cp = _ptr(b, cbase)
                    b.store(_c(b, 0), cp, idx)
                    b.store(_c(b, 0), cp, didx)
        with b.scf_if(b.land(b.land(valid, b.lnot(spread)), b.cmp_eq(old, _c(b, S - 1)))):
            D.acquire_l2(b, "agent")
            self.fold(u, p_rs)
            with b.scf_if(D.is_thread0(b)):
                b.store(_c(b, 0), _ptr(b, cbase), idx)

    def fold(self, u: _Unit, p_rs: Value, part: Value | None = None) -> None:
        """Last arriver (``part`` None): Y = sum over splits (split order) of the partials of (row group, token block);
        ``part`` = s (fold=spread): only pieces ``[s P / S, (s + 1) P / S)`` of the group's P pieces."""
        b, k = self.b, self.k
        TP = k.tiles * TILE
        R = k.rows_per_unit
        q4 = R // 4
        pieces = k.tbt * TILE * q4
        y_rs = _rsrc(b, u.y_a, self.yl32)
        S = k.splits - (1 if k.mutant == "fold" else 0)
        span = pieces if part is None else -(-pieces // k.splits)
        lo = None if part is None else b.mul(part, _c(b, span))
        for it in range(-(-span // k.threads)):
            v = b.add(self.tid, _c(b, it * k.threads))
            if lo is not None:
                inslice = b.cmp_lt(v, _c(b, span))
                v = b.add(v, lo)
            tl = _udiv(b, v, q4)
            row = b.add(b.mul(u.rg, _c(b, R)), b.mul(_umod(b, v, q4), _c(b, 4)))
            tok = b.add(u.tok0, tl)
            ok = b.land(b.cmp_lt(v, _c(b, pieces)), b.cmp_lt(tok, self.rows))
            if lo is not None:
                ok = b.land(ok, inslice)
            tot = None
            for s0 in range(0, S, k.fg):
                loads = []
                for s in range(s0, min(S, s0 + k.fg)):
                    off = b.mul(b.add(b.mul(b.add(_c(b, s * TP), tok), self.Mpad), row), _c(b, 4))
                    loads.append(b.buffer_load_cp(p_rs, b.select(ok, off, _c(b, OOB)), _c(b, 0), dwords=4, aux=CPOL_SC1))
                for ld in loads:
                    f = b.vec_bitcast(ld, VectorType(F32, 4))
                    if tot is None:
                        tot = [b.vec_extract(f, e) for e in range(4)]
                    else:
                        tot = [b.fadd(tot[e], b.vec_extract(f, e)) for e in range(4)]
            yv = b.vec_bitcast(b.vec_trunc_f32_to_bf16(b.vec_pack(tot, F32)), VectorType(I32, 2))
            yoff = b.mul(b.add(b.mul(tok, self.Mpad), row), _c(b, 2))
            b.buffer_store_cp(yv, y_rs, b.select(ok, yoff, _c(b, OOB)), _c(b, 0))
            if k.yq:
                with b.scf_if(b.cmp_lt(v, _c(b, pieces))):
                    self._ybuf_put(tl, b.mul(_umod(b, v, q4), _c(b, 4)), yv)

    # ------------------------------------------------------------------ yq: MXFP8 of bf16(Y) in the op_sel layout
    def _ybuf_put(self, tl: Value, lrow: Value, ybf: Value) -> None:
        """4 consecutive rows ``lrow ..`` of local token ``tl``: the bf16 values (``<2 x i32>``) as f32 in LDS."""
        b = self.b
        vals = []
        for d in range(2):
            w = b.vec_extract(ybf, d)
            vals.append(b.bitcast(b.shl(w, _c(b, 16)), F32))
            vals.append(b.bitcast(b.land(w, _c(b, -65536)), F32))
        b.smem_store_vN(self.ybuf, [tl, lrow], b.vec_pack(vals, F32), 4)

    def yq_pass(self, u: _Unit) -> None:
        """Quantize the unit's ``R x 16 tbt`` bf16 outputs (LDS ``ybuf``) to MXFP8 per (token, 32-row block): 4 lanes x 8
        rows a block (``mx_quant.quantize_mxfp8``); codes ``yqc[token][row]`` (8 bytes a lane), the E8M0 byte at
        ``yqs[(G bucket16 + token) 16 + 4 g + j]`` for chunk ``q = row / 128`` (``G = q / 4``, ``j = q mod 4``, block
        ``g`` of the chunk). Write-through (``sc1``) stores; every lane of every wave runs the shuffles."""
        b, k = self.b, self.k
        R = k.rows_per_unit
        TPT = k.tbt * TILE
        TP = k.tiles * TILE
        nb = R // 32
        total = TPT * nb * 4
        b.sync_lds_only()
        yc_rs = _rsrc(b, u.yqc_a, self.yqcl32)
        ys_a = b.readfirstlane(u.yqs_a)
        for it in range(-(-total // k.threads)):
            v = b.add(self.tid, _c(b, it * k.threads))
            vin = b.cmp_lt(v, _c(b, total))
            vv = b.select(vin, v, _c(b, 0))
            p = _umod(b, vv, 4)
            blk = _umod(b, _udiv(b, vv, 4), nb)
            tl = _udiv(b, vv, 4 * nb)
            lrow = b.add(b.mul(blk, _c(b, 32)), b.mul(p, _c(b, 8)))
            lo = b.smem_load_vN_f32(self.ybuf, tl, lrow, n=4)
            hi = b.smem_load_vN_f32(self.ybuf, tl, b.add(lrow, _c(b, 4)), n=4)
            vals = [b.vec_extract(lo, e) for e in range(4)] + [b.vec_extract(hi, e) for e in range(4)]
            words, E = MQ.quantize_mxfp8(b, vals, 4)
            tok = b.add(u.tok0, tl)
            grow = b.add(b.mul(u.rg, _c(b, R)), lrow)  # global output row (= the consumer's K)
            ok = b.land(b.land(vin, u.valid), b.land(b.cmp_lt(tok, self.rows), b.cmp_lt(grow, self.M)))
            coff = b.add(b.mul(tok, self.M), grow)
            b.buffer_store_cp(b.vec_pack(words, I32), yc_rs, b.select(ok, coff, _c(b, OOB)), _c(b, 0), aux=CPOL_SC1)
            q = _udiv(b, grow, CHUNK)
            gq = _umod(b, _udiv(b, grow, 32), 4)
            soff = b.add(b.mul(b.add(b.mul(b.add(b.mul(_udiv(b, q, 4), _c(b, TP)), tok), _c(b, 4)), gq), _c(b, 4)), _umod(b, q, 4))
            with b.scf_if(b.land(ok, b.cmp_eq(p, _c(b, 0)))):
                addr = b.add(ys_a, _i64(b, soff))
                b.inline_asm("global_store_byte $0, $1, off sc1", "v,v,~{memory}", [addr, E], None)
        b.sync_lds_only()  # ybuf is reused by the next unit

    def bardrop_delay(self) -> None:
        """mutants bardrop / barnone: CTA 0 sleeps (about 60 us) at the start of every unit, so its rows of every layer
        land late and a consumer released early reads them stale (deterministic, whatever the unit count)."""
        b = self.b
        with b.scf_if(b.cmp_eq(self.cta, _c(b, 0))):  # straight-line sleeps (a loop here made the build spill)
            b.inline_asm("\n".join(["s_sleep 127"] * 16), "~{memory}", [], None)

    # ------------------------------------------------------------------ main loop
    def build(self) -> KernelDef:
        b, k = self.b, self.k
        D_, XA, KC = k.depth, k.xa, k.kc
        lds = k.xsrc == "lds"
        units = {}

        def unit(n: Value, d: int) -> _Unit:
            key = d
            if key not in units:
                units[key] = _Unit(self, b.add(n, _c(b, d)) if d else n)
            return units[key]

        def wpos(n: Value, p: int, xg=None) -> list:
            d, c = divmod(p, KC)
            return self.wissue(unit(n, d), c, xg)

        def xpos(n: Value, p: int, gate=None) -> list:
            d, c = divmod(p, KC)
            return self.xissue(unit(n, d), c, gate)

        def sel(cond, new, old):
            return [b.select(cond, x, y) for x, y in zip(new, old)]

        def parity(n: Value, c: int) -> Value:
            if KC % 2 == 0 or not k.xu:
                return _c(b, c % 2)
            return b.land(b.add(n, _c(b, c)), _c(b, 1))

        zero = _c(b, 0)
        # prologue: unit 0's first chunks
        n0 = zero
        units.clear()
        self.cur_layer = unit(n0, 0).layer
        wq = [wpos(n0, p) for p in range(D_)] if k.xu else []
        xq = []
        if lds and k.xu:
            first = xpos(n0, 0)
            self.xput(first, zero)
            xq = [xpos(n0, p) for p in range(1, XA + 1)]
            b.sync_lds_only()
        lens_w = [len(g) for g in wq]
        lens_x = [len(g) for g in xq]
        carry = [v for g in wq for v in g] + [v for g in xq for v in g]
        iters = [(f"c{i}", v) for i, v in enumerate(carry)]
        loop = b.scf_for_iter(zero, self.ncta, _c(b, 1), iters, iv_name="un")
        with loop as lv:
            n, vals = lv if iters else (lv, [])
            units.clear()
            vals = list(vals)
            u = unit(n, 0)
            self.cur_layer = u.layer
            self.tst(n, 0)
            # grid sync at a layer crossing (phase metric), then the activations of this layer (re-)issued
            cross = b.land(self.gsync, b.land(b.cmp_eq(u.i, zero), b.cmp_gt(u.layer, zero)))
            with b.scf_if(cross):
                self.gbar(b.sub(u.layer, _c(b, 1)))
            pos = 0
            if k.xu:
                wq = []
                for ln in lens_w:
                    wq.append(vals[pos : pos + ln])
                    pos += ln
                xq = []
                for ln in lens_x:
                    xq.append(vals[pos : pos + ln])
                    pos += ln
                if lds:
                    f0 = xpos(n, 0, cross)
                    with b.scf_if(cross):
                        self.xput(f0, parity(n, 0))
                        b.sync_lds_only()
                    xq = [sel(cross, xpos(n, p, b.land(cross, self.xgate(unit(n, p // KC)))), xq[p - 1]) for p in range(1, XA + 1)]
                if k.sop:
                    for p in range(D_):
                        d_, c_ = divmod(p, KC)
                        if c_ % 4 and k.xsrc != "global":
                            continue
                        fresh = self.wissue(unit(n, d_), c_, b.land(cross, self.xgate(unit(n, d_))), xonly=True)
                        nx = len(fresh)
                        if nx:
                            wq[p] = wq[p][: len(wq[p]) - nx] + sel(cross, fresh, wq[p][len(wq[p]) - nx :])
                elif k.xsrc == "global" or k.xscale == "glb":
                    nw = k.mt * (2 if k.w == "mx" else 4)
                    for p in range(D_):
                        d_, c_ = divmod(p, KC)
                        fresh = self.wissue(unit(n, d_), c_, b.land(cross, self.xgate(unit(n, d_))), xonly=True)
                        wq[p] = wq[p][:nw] + sel(cross, fresh, wq[p][nw:])
            else:
                wq = [wpos(n, p) for p in range(min(D_, KC))]
                if lds:
                    first = xpos(n, 0)
                    b.sync_lds_only()  # every wave is done with the previous unit's last LDS buffer
                    self.xput(first, zero)
                    xq = [xpos(n, p) for p in range(1, min(XA, KC - 1) + 1)]
                    b.sync_lds_only()
            if k.mutant in ("bardrop", "barnone", "nowait"):
                self.bardrop_delay()
            segv = self.seg_loads(u)
            acc = [[b.zero_vec_f32(16 if k.mfma == 32 else 4) for _ in range(k.ntt)] for _ in range(k.mt)]
            self.tst(n, 1)
            for c in range(KC):
                if k.sb:
                    b.sched_barrier(0)
                if k.xu or c + D_ < KC:
                    wq.append(wpos(n, c + D_))
                if lds and (k.xu or c + 1 + XA < KC):
                    xq.append(xpos(n, c + 1 + XA))
                if k.sb:
                    b.sched_barrier(0)
                cur = wq.pop(0)
                skip = k.mutant == "kskip" and c == KC - 1
                acc = self.chunk_math(acc, cur, parity(n, c) if lds else None, skip, c % 4, u)
                self.tst(n, 2 + 2 * c)
                if lds and (k.xu or c + 1 < KC):
                    nxt = xq.pop(0)
                    self.xput(nxt, parity(n, c + 1))
                    b.sync_lds_only()
                self.tst(n, 3 + 2 * c)
            if k.sb:
                b.sched_barrier(0)
            self.epilogue(u, acc, segv)
            self.tst(n, 2 + 2 * KC)
            carry = [v for g in wq for v in g] + [v for g in xq for v in g]
            if not k.xu:
                carry = []
            b.scf_yield(*carry)
        # a CTA without units in a layer still takes part in every boundary's grid sync
        with b.scf_if(b.land(self.gsync, b.cmp_eq(self.cnt, zero))):
            with b.scf_for(zero, b.sub(self.L, _c(b, 1)), _c(b, 1), iv_name="gb") as bnd:
                self.gbar(bnd)
        if k.bar == "persist":
            self._bar_end()
        if k.stamp:
            b.sync_lds_only()
            self._stamp(1, None)
        return b.kernel


def b_mul2(b: IRBuilder, v: Value) -> Value:
    return b.shl(v, _c(b, 1))


def _funnel(b: IRBuilder, lo: Value, hi: Value, sh8: Value) -> Value:
    """Low dword of ``(hi:lo) >> sh8`` (``v_alignbyte_b32``-shaped; ``sh8`` in 0, 8, 16, 24)."""
    w = b.lor(b.shl(b.zext(hi, I64), b.const_i64(32)), b.zext(lo, I64))
    return b.trunc(b.lshr(w, b.zext(sh8, I64)), I32)


def build_e1_kernel(k: E1Knobs) -> KernelDef:
    kd = _E1(k).build()
    if k.hz:
        from .insitu import pad_asm_vmem

        pad_asm_vmem(kd)
    return kd

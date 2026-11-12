# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx950 MXFP6 x MXFP6 GEMM: ``C[M, N] (bf16) = A[M, K] * B[N, K]^T``, fp32 accumulate.

Operands are OCP MX: 6-bit element codes (E2M3 by default, E3M2 with ``elem_fmt="e3m2"``)
and one E8M0 scale per 32 K values.
The codes are read either in the canonical dense stream (``operand_layout=
"canonical"``) or in a host pre-tiled image of the LDS stage (``"c0c1"``, see
``layout.pack_operand_c0c1``); the scales are repacked once on the host into the
lane-native op_sel dword layout (``layout.pack_scales``).

Body (every performance-relevant choice is a defaulted :class:`Mxfp6GemmSpec`
field, listed with its legal values in :data:`KNOBS`):

* one CTA per ``tile_m x tile_n`` output tile, ``waves_m x waves_n`` waves, each
  wave a ``tile_m / waves_m x tile_n / waves_n`` sub-tile of scaled MFMAs
  (``atom``: ``v_mfma_scale_f32_16x16x128_f8f6f4`` or ``..._32x32x64_...``,
  fp6 x fp6, six-VGPR operands, f32 accumulators in AGPRs);
* a ring of ``stages`` LDS stages, one 128-K tile each (96 bytes per row),
  filled by direct-to-LDS ``buffer_load_dwordx4 ... lds``. Canonical layout:
  every 16-byte unit comes straight from the dense stream (rows 8..15 of every
  16 rotated by 48 bytes, ``swizzle="rot48"``, so the three ``ds_read_b64`` of a
  24-byte fragment are bank-conflict free). c0c1 layout: one contiguous block
  per operand and stage, fragments read as ``ds_read_b128`` + ``ds_read_b64``;
* the K-tile schedule: one MFMA stream per K tile, every other instruction
  pinned to an MFMA slot (``sched_barrier(0)`` after each MFMA), one acquire
  (``s_waitcnt vmcnt(n) lgkmcnt(0); s_barrier``) per tile. ``frag_mode``:
  ``prefetch`` reads the next tile's fragments during this tile (two fragment
  sets, refill distance ``stages``); ``jit`` reads this tile's fragments in it,
  each ``read_lead`` MFMAs before its first use (one fragment set, refill
  distance ``stages - 1``; a large lead is the aiter "burst" step);
* scales: one 16-byte load per lane per four 16-row tiles per four K steps; the
  MFMA ``op_sel`` picks the byte (a scale VGPR is written once per four MFMAs
  of an accumulator);
* XCD-aware tile order (``xcd_block``);
* bf16 epilogue straight from the accumulators: with ``swap_ab`` (B is the
  MFMA's src0) a lane holds four consecutive output columns, one 8-byte store.

``probe`` builds measurement variants that are NOT GEMMs: ``mfma_only`` (the
same MFMA stream on register-resident tile-0 data: the data-matched compute
ceiling) and ``fill_only`` (the same DTL ring, waits, barriers and LDS reads
with the MFMAs removed: the fill floor).

Python engine only (``tile.mfma_scale_mx`` is a Python-only op): compile with
``compile_kernel(kernel, arch="gfx950", backend="python")``.

Kernel arguments: ``a, b`` (i8 operand images), ``sa, sb`` (i32 packed scales),
``c`` (bf16 output). Grid ``(M / tile_m) * (N / tile_n)`` CTAs of
``64 * waves_m * waves_n`` threads; no dynamic LDS.
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import json
from dataclasses import asdict, dataclass, fields, replace
from typing import Callable, Dict, List, Optional, Tuple

from rocke.core.ir import BF16, F32, I8, I32, I64, IRBuilder, KernelDef, PtrType, Value, VectorType

from .layout import ROW_BYTES, SWIZZLES, UNITS, atom_geometry
from .splitk import MASK_MUTANTS, SK_MUTANTS

__all__ = [
    "KNOBS",
    "MUTANTS",
    "PRESETS",
    "Mxfp6GemmSpec",
    "build_mxfp6_gemm",
    "check_compiler",
    "preset",
    "descriptor",
    "grid_of",
    "kernel_name",
    "lds_bytes",
    "plan",
    "scale_layout_of",
    "tile_of_wg",
    "validate_spec",
]

ARCH = "gfx950"
#: Test-only must-fail variants (each breaks the result in a different way).
#: ``scale_sel``: K tiles with op_sel 1 read byte 0 (wrong E8M0 scale);
#: ``k_order``: one K tile's fragments are read from the wrong ring stage;
#: ``drop_kstep``: the last K tile's MFMAs are skipped;
#: ``wrong_fmt``: the MFMA decodes the codes as the other 6-bit format (E2M3 <-> E3M2).
MUTANTS = ("none", "scale_sel", "k_order", "drop_kstep", "wrong_fmt")
#: Must-fail variants of the tile-library structures (not in MUTANTS: they need a batched / M-contiguous spec):
#: ``batch_swap``: entry z reads entry z ^ 1's A; ``mc_noswz``: the M-contiguous A is read without its slot swizzle;
#: ``scale_lane``: every lane reads the wtile scale dwords of lane l ^ 16 (the neighbouring 32-K block; unlike
#: scale_sel it also fails a wave with one MFMA tile, whose op_sel is always 0).
LIB_MUTANTS = ("batch_swap", "mc_noswz", "scale_lane", "persist_drop", "wper_rot", "mc_copyswap")
#: ``mc_copyswap`` (am lane): with ``mc_copy`` other than auto, the explicit second-half copies swap dwords 1 and 2.
#: line_merge's must-fail hazard: the loop's line refills one tile further ahead (they overwrite the slot of a line the
#: next tile still reads, after it landed)
LINE_MUTANTS = ("lmerge_lead",)
#: Must-fail variants of ``group_table`` (grouped GEMM; not in LIB_MUTANTS: they need a group-table spec):
#: ``gt_zero``: every tile row multiplies group 0 (the table is ignored); ``gt_swap``: tile row r multiplies group
#: gtab[r] ^ 1 (the neighbouring expert; needs an even group count).
GROUP_MUTANTS = ("gt_zero", "gt_swap")
#: Element formats (problem contract, not a tuning knob) -> MFMA operand format name (cbsz / blgp 2 / 3).
ELEM_FMTS = {"e2m3": "fp6", "e3m2": "bf6"}
#: ``required_compiler`` values: "any" (LLVM or the rocKE direct compiler) | "rust" (direct compiler only).
COMPILERS = ("any", "rust")
XCD_BLOCKS = ("none", "4x8", "8x4", "2x16", "16x2",
              # smaller CTA grids (overlap.py's M-chunk launches: M = 2048 / 1024 / 512 / 256 rows of a 4096^2 C)
              "2x8", "4x4", "8x2", "1x8", "2x4", "4x2", "1x4", "2x2", "1x2")
FRAG_MODES = ("prefetch", "jit")
OPERAND_LAYOUTS = ("canonical", "c0c1", "c0c1p")
PROBES = ("none", "mfma_only", "fill_only", "dtl_only", "lds_only", "stamps", "xcd_trim", "canon_align", "canon_line",
          "line_lead3", "l2m_nomini", "l2m_nodual", "l2m_norelease", "l2m_early", "l2m_nocopy",
          "l2m_lead4", "l2m_copyread", "l2m_copywrite", "l2m_copyvm0",
          "ldtl_noA", "ldtl_noB", "l2m_nocopyB", "ldtl_contigB", "l2m_earlyB",
          # a_scale_src="ring" must-fail hazards: the next period's scale DTL into the slot still being read
          # (as_hazard), the reads from the other slot (as_stale)
          "as_hazard", "as_stale",
          # a_scale_src="ring" timing probes (not GEMMs): the ring DTLs fetch 1 KiB contiguous per instruction (as_contig:
          # the cost of 64 lines per DTL), the A scale dwords read like the tile image (as_lite: one b64, no v_perm),
          # no ring DTLs at all (as_nodtl); as_ideal = as_contig + as_lite, as_none = as_nodtl + as_lite (the floor of the
          # LDS / refill-plan changes alone)
          "as_contig", "as_lite", "as_nodtl", "as_ideal", "as_none",
          # tile-scale timing probes (not GEMMs): no per-tile scale DTL for B / for A and B (the scales go stale)
          "sc_noB", "sc_noAB", "bs_hazard", "bs_stale")
A_SCALE_SRCS = ("tile", "ring", "pring")
B_SCALE_SRCS = ("tile", "pring")
AS_PROBES = ("as_hazard", "as_stale", "as_contig", "as_lite", "as_nodtl", "as_ideal", "as_none")
AS_RAW_PROBES = ("as_contig", "as_lite", "as_nodtl", "as_ideal", "as_none")  # the raw ring's timing probes
BS_PROBES = ("bs_hazard", "bs_stale")  # b_scale_src pring must-fail hazards (as_hazard / as_stale for B)
PRING_PERIODS = (3, 4)  # pring: K tiles per period = dwords per lane of its DTL (dwordx3 / dwordx4)
AS_PERIOD = 4  # a_scale_src ring: K tiles per ring slot (16 scale bytes = one dwordx4 lane per row)
AS_SLOTS = 2  # a_scale_src ring: slots (periods) in the ring
SCALE_LAYOUTS = ("kstep", "tile", "raw", "wtile")
#: A operand layouts (``a_layout``): "" (= operand_layout), "canonical" (K-contiguous dense codes), "c0c1",
#: "mcontig" (M-contiguous activations: [K][M] dense codes, read with ds_read_b96_tr_b6)
A_LAYOUTS = ("", "canonical", "c0c1", "mcontig")
CTA_ORDERS = ("xcd_block", "grouped", "linear", "mfast")
DEPTH_US = (128, 256, 512)
TAIL_MODES = ("static", "clamp")
SCALE_SRCS = ("vgpr", "lds")
NUM_XCDS = 8
DTL_BYTES = 16  # dwordx4 direct-to-LDS unit
#: sched_barrier mask letting every class but VMEM cross (ALU, VALU, SALU, MFMA, DS, transcendental)
VM_FENCE_MASK = 0x0001 | 0x0002 | 0x0004 | 0x0008 | 0x0080 | 0x0100 | 0x0200 | 0x0400
ROWS_PER_GROUP = 32  # canonical: 32 rows x 6 units = 3 DTL instructions of 64 lanes
MC_ROW_BYTES = 128  # mcontig A: LDS bytes per tile row and 128-K tile (8 K rows x one 16-byte slot per 16 rows)
LINE_BYTES = 128  # canon_ring line2: one ring slot of a row (an L2 line)


@dataclass(frozen=True)
class Mxfp6GemmSpec:
    """One build. Every field has a default (the problem shape included)."""

    m: int = 4096
    n: int = 4096
    k: int = 65536
    elem_fmt: str = "e2m3"  # OCP MX 6-bit element format of A and B | "e3m2" (problem contract, not a knob)
    required_compiler: str = "any"  # | "rust": structures the LLVM build spills on (see PRESETS / check_compiler)
    dlevers: str = ""  # the kernel's own direct-compiler levers (kernel attribute ``dlevers``; "a,b" on, "-a" off, each
    #   deciding that lever over ROCKE_DIRECT_LEVERS): a self-describing build (``with_dlevers``). Not in the kernel
    #   name; an empty string emits nothing. Needs a direct compiler that reads it (rustgemm crate-up); others refuse
    # ---- geometry ----
    atom: str = "16x16x128"  # | "32x32x64"
    tile_m: int = 256
    tile_n: int = 256
    waves_m: int = 2
    waves_n: int = 2
    stages: int = 3  # LDS ring depth (128-K tiles)
    swizzle: str = "rot48"  # canonical layout: LDS row rotation | "none"
    canon_ring: str = "stage"  # canonical operands: 96-byte rows in the stage ring | "line2" / "line3": a ring of
    #   2 / 3 whole 128-byte lines per row (one new line per row on 3 of 4 tiles: 0.75 L2 requests per row and tile
    #   instead of 1.5); refill 2 / 3 tiles ahead; line2 needs a split release barrier at ``line_release`` on every
    #   4th tile, line3 none (3 lines cover the next 3 tiles) | "line2m": 2 slots + a 32-byte-per-row remnant
    #   buffer (the head line's tail, read by the group's second tile): every refill 3 tiles ahead, after a release
    ring_ops: str = "all"  # line ring: every canonical operand | "A" (a canonical B keeps the stage ring)
    mini_copy_gap: int = 8  # line2m: MFMA slots between the remnant copy's LDS reads and writes
    mini_src: str = "copy"  # line2m remnant: copied in LDS after the head line lands | "dtl": fetched from L2 by its
    #   own LDS DMAs (dwordx4, 32 rows x 32 B each) with the head line's refill; needs mini_layout v4
    mini_copy_at: int = -1  # line2m: MFMA slot of the remnant copy's reads (-1: the acquire slot); before the
    #   (first) line release: the head line's slot is refilled after it
    mini_copy_at_b: int = -1  # line2m + line_release_b: B's remnant copy reads at their own slot (before B's release;
    #   writes mini_copy_gap later); -1: with A's
    mini_layout: str = "v2"  # line2m remnant buffer: group stride rows / 16 * 128 + 8, one row per lane in the copy
    #   (2-way LDS bank conflicts in the remnant reads and the copy reads, PMC job 68318010) | "v3": group offsets on
    #   8-byte bank slots 0 / 2 / 16 / 18 and a two-lanes-per-row copy (conflict-free: ``mini_offsets``) | "v4": v3's
    #   offsets (16-byte aligned) with v2's one-row-per-lane copy and two 16-byte writes (PMC job 68319782: the copy's
    #   cost is its writes -- 4 ds_write_b64 per lane -- not its reads)
    ring_kib: bool = False  # line ring: each ring allocation padded to whole KiB (the direct compiler's LDS DMA tags
    #   are KiB ranges: two rings sharing a KiB make every read of one wait for the DMAs into the other, vmcnt(0))
    b_reg: bool = False  # c0c1 B through VGPRs: global loads 3 tiles ahead, ds_write 2 tiles ahead into a 2-stage
    #   LDS ring (B keeps a 2-tile lead in 2 stages: room for a 3-slot A line ring)
    line_spread: int = 0  # line2: MFMA slots between consecutive line DTLs (0: spread over the rest of the tile)
    line_reads_first: bool = False  # line ring: prefetch the ring operands' fragments first (their release / refills
    #   can then come before the stage-ring operand's reads end)
    line_early: bool = False  # line2: refills that overwrite no line the next tile reads go right after the acquire
    #   (else after the tile's last ring read: the direct compiler waits for a ring DTL before every later ring read)
    line_release: int = 26  # line2: MFMA slot of the release barrier before the refill that overwrites a line the
    #   next tile still reads (every fragment read of the next tile must precede it)
    line_release_b: int = -1  # both operands in a line2 / line2m ring: B's own release slot (A's fragments are read
    #   first, A's release at ``line_release`` and A's refills follow them, then B's reads, release and refills); -1:
    #   one release for both
    canon_rsrc_once: bool = False  # canonical operands: one buffer descriptor (built once), the K tile's 96-byte
    #                                step in soffset (rsrc_once for the dense stream)
    canon_read: str = "b64x3"  # stage-ring canonical fragment reads: three ds_read_b64 | "b96x2": two ds_read_b96
    canon_dtl: str = "band"  # canonical DTL order: wave w fills its own 32-row groups | "interleave": instruction
    #                          i of wave w copies tile bytes [1 KiB (i nw + w), +1 KiB) (HipKittens' order)
    operand_layout: str = "c0c1"  # host pre-tiled stage images | "canonical" (dense stream)
    a_layout: str = ""  # A's layout when it differs ("canonical": live activations next to pre-tiled weights)
    mc_dma: str = "x4"  # mcontig A DMA: 16 bytes per lane (the slot's 12 code bytes + 4 overlap) | "x3" (am lane):
    #   buffer_load_dwordx3 lds, the 12 bytes only (gfx950 writes a b96 DMA lane at a 16-byte LDS stride, so the
    #   stage image is unchanged; -25 % A bytes through TA / L2). Needs a direct compiler with BufLoadLdsB96.
    mc_copy: str = "auto"  # mcontig A second-half copy (am lane, mid-tile core loop): "auto" = the compiler's
    #   alignment copy (gfx950 VGPR tuples are even-aligned, so the ds_read_b96_tr_b6 of K 16..31 lands in a temp and
    #   moves to dwords 3..5 right behind the read, behind an s_waitcnt lgkmcnt(0) per fragment) | "late": three
    #   explicit v_mov_b32 of the half at the fragment's first MFMA (or the K loop's yield), so the wait sits there |
    #   "fence": "late" pinned there (``~{memory}``). Rust compiler only. (Explicit v_movs at the read itself compile
    #   to the auto code: the placement, not the copy kind, is what matters.)
    scale_layout: str = "tile"  # op_sel = MFMA tile, per-tile loads | "kstep" (op_sel = K step, 4 tiles per load)
    #                             | "raw": the canonical [rows, K/32] bytes (no scale packing): one dword per row
    #                               and tile DTL'd into the stage, assembled into the tile op_sel dwords by v_perm
    scale_src: str = "lds"  # tile scales DTL'd into the ring stage, ds_read | "vgpr" (VGPR loads one tile ahead)
    a_scale_src: str = "tile"  # A's scales in ``scale_layout`` (tile: a per-call raw -> tile repack for live
    #   activations) | "ring": the canonical [M, K/32] bytes as produced (no repack; B keeps the tile layout): each
    #   wave DTLs 64 rows x 16 bytes (4 K tiles) per 4 tiles (dwordx4) into a 2-slot LDS ring, the tile op_sel
    #   dwords are assembled from it with 8 ds_read_b32 + 6 v_perm per tile and wave
    as_lead: int = 4  # a_scale_src ring: tiles from a period's scale DTL to the period's first tile (2..5; 5 issues
    #   it in the tile after the slot's last read: after that tile's acquire)
    as_slot: int = 1  # a_scale_src ring: MFMA slot of that DTL in its tile
    as_perm: int = -1  # a_scale_src ring: MFMA slot of the v_perm assembly of the next tile's A scale dwords (-1:
    #   right after their ring reads at the acquire, which then wait for the LDS there)
    #   a_scale_src "pring": A's tile scales pre-packed per period of as_period K tiles (layout.pack_scales_for
    #   "period{P}x{tile_m}": the CTA row block's tile images of P consecutive K tiles contiguous; a light per-call
    #   repack like the tile layout), one dwords=P DTL per wave and period into a 2-slot LDS ring (instead of one
    #   dword DTL per tile), read like the stage image (one b64 per tile)
    as_period: int = 4  # a_scale_src pring: K tiles per period (3 | 4: dwordx3 / dwordx4 DTLs); ring: 4
    b_scale_src: str = "tile"  # B's tile scales in the stage image | "pring": as a_scale_src pring (weights: packed
    #   offline)
    bs_period: int = 4  # b_scale_src pring: K tiles per period (3 | 4)
    bs_lead: int = 4  # b_scale_src pring: tiles from a period's DTL to the period's first tile (2..5)
    bs_slot: int = 1  # b_scale_src pring: MFMA slot of that DTL
    # ---- MFMA ----
    swap_ab: bool = True  # B fragment is src0: a lane holds 4 consecutive C columns
    out_exp2: int = 0  # epilogue multiplies C by 2^out_exp2 before bf16 RNE (exact; pairs with host scale biasing)
    acc_agpr: bool = True  # accumulators in AGPRs (+ the AGPR MFMA form)
    line_dtl_order: str = "strided"  # line rings: line DTL c of wave w = i nw + w ("strided") | w n + i ("wave": one
    #   wave's DTLs write consecutive row classes, RB bytes apart, so m0share can give them one M0 + offsets) | "block"
    #   (chunk counts per wave that do not divide 16, e.g. 384 rows on 4 waves: wave w takes row classes
    #   [w 16 / nw, (w + 1) 16 / nw) of every 16-class block -- one LDS base per wave + constants as "wave")
    acc_vgpr_spill: bool = False  # acc_agpr with more than 256 accumulator registers per lane: atoms 0..63 in the 256
    #   AGPRs, the rest in VGPRs (a 4-wave layout of the tiles that needed 8 waves to hold their accumulators)
    acc_pin: bool = False  # route every MFMA result through asm "=a,0" (pin to the AGPR file)
    peel_first: bool = False  # first scale period static, first MFMAs with srcC = 0 (no accumulator seed)
    serpentine: bool = True  # odd rows walk the inner fragments backwards
    m_outer: bool = True  # MFMA order: A-fragment (m) outer, B (n) inner
    pin_mfma: bool = True  # sched_barrier(0) after every MFMA
    read_split: bool = True  # sched_barrier between a fragment's LDS reads (no ds_read2 merge)
    lds_alias_scopes: bool = True  # per-stage LDS alias scopes: refill DMAs do not force vmcnt(0) before reads
    # ---- K-tile schedule (slots = MFMA index within the tile; -1 = before the first) ----
    frag_mode: str = "jit"  # | "prefetch"
    acq_slot: int = -1  # acquire: s_waitcnt vmcnt(n) lgkmcnt(0) + s_barrier (jit: must be -1)
    read_every: int = 2  # prefetch: one fragment every N MFMAs after the acquire
    read_delay: int = 0  # prefetch: MFMAs between the acquire and the first fragment read
    read_lead: int = 4  # jit: a fragment is read this many MFMAs before its first use (>= 64: all at the top)
    dtl_delay: int = 0  # MFMAs between the acquire and the first refill DTL
    dtl_spread: int = 0  # MFMAs between DTL issues (0 = evenly over the rest of the tile)
    scale_slot: int = 40  # slot of the next period's scale loads (first tile of a period)
    setprio: int = 0  # s_setprio level over the MFMA stream (0 = off)
    tile_lgkm0: bool = False  # explicit lgkmcnt(0) at the end of every tile
    unroll: int = 1  # K-loop body = unroll x lcm(stages, scale period) tiles
    reg_stages: int = 0  # register-staged fill: 0 = direct-to-LDS; R >= 1 = global->VGPR loads R tiles further
    #   ahead, written to the freed stage with ds_write (more bytes in flight than the LDS ring holds)
    reg_slot: int = 0  # MFMA slot of a tile's first register-fill global load
    reg_every: int = 4  # MFMA slots between register-fill global loads
    release_slot: int = -1  # jit split barrier: release (lgkmcnt(0) + s_barrier) after this MFMA slot, then refill
    #   THIS tile's stage (stages tiles ahead instead of stages - 1); all fragment reads move before it; -1 = merged
    skew_dtl: int = 0  # odd waves run a second loop copy whose refill DTLs sit this many MFMA slots later
    skew_read: int = 0  # ... and whose fragment reads lead by this many more MFMAs (desynchronize LDS / TA bursts)
    carry: int = 0  # jit: the first N fragments (first-use order) + the scales of the next tile are read at
    #   carry_slot of the current tile, after a moved acquire (hides the tile-top read latency)
    carry_slot: int = 56  # carry: MFMA slot of the moved acquire (every own-tile read must issue before it)
    scale_dtl: str = "dword"  # scale_src lds: every wave copies 256 B per operand with dword DTLs | "x4w": one
    #   1 KiB dwordx4 DTL per operand (wave 0: A, wave nw/2: B), issued first in the tile's refill | "split" (wtile):
    #   the images' 256-byte chunks over the waves without duplicates (chunk c of A then B: wave c % waves), first in
    #   the refill, outside the vmcnt model
    k_rot: int = 0  # fabric probe: XCD x starts its K loop at tile x * k_rot * T / 8 (mod T) -- 1 de-aligns the
    #   eight XCDs along K (cross-XCD panel reuse in MALL), 0 = all XCDs walk K in step (default)
    k_phase: int = 2  # k_rot 3 / 4: CTA loc of its XCD block starts at K tile (loc % k_phase) * k_phase_tiles (3: the CTAs
    #   of one B panel in step) or ((loc / bn) % k_phase) * k_phase_tiles (4: the CTAs of one A panel in step) -- a small
    k_phase_tiles: int = 8  # per-CTA phase that keeps the panels' L2 reuse window (k_rot 2 spreads over all of K)
    mma_order: str = "rows"  # MFMA order in a tile: "rows" (m_outer / serpentine) | "pairs" (two A rows
    #   interleaved: each B fragment feeds two MFMAs back to back) | "blk2" (2x2 blocks, serpentine)
    rsrc_once: bool = False  # pre-tiled operands: one buffer descriptor per operand (built once); the K tile's
    #   byte offset rides in the DTL soffset (no per-tile descriptor rebuild in the loop)
    dtl_imm: bool = False  # c0c1: blocks of four 1 KiB DTLs share M0 / soffset via the instruction offset
    dtl_aux: int = 0  # cache-policy bits of the operand refill DTLs (0 default, 1 glc/sc0, 2 slc/nt, 3 both)
    b_aux: int = -1  # B's (the weights') refill DTLs only (-1: dtl_aux): the streamed weights can load non-temporal
    #   while A keeps its L2 reuse (MoE lane: nt streaming reads +13 % in the HBM probe); with b_direct, a
    #   b_aux >= 0 sets the direct B VGPR loads' cache policy instead (-1 leaves them at 0)
    b_direct: bool = False  # B fragments straight from the c0c1 global image into VGPRs (no LDS for B)
    a_fill: str = "dtl"  # canonical A in the stage ring: "dtl" (the 96-byte rows as stored) | "c0c1": the DMAs build the
    #   c0c1 image (c0 plane: each fragment's first 16 bytes, one dwordx4 lane from the dword-aligned global offset
    #   24 kg; c1 plane: the last 8 bytes, two dword lanes) so A is read as one ds_read_b128 + one ds_read_b64 per
    #   fragment, like a pre-tiled A, from the canonical activations (12 DMA instructions per 256 rows and wave vs 6)
    a_direct: bool = False  # tile-library ak specs: A fragments straight from the canonical rows into VGPRs (no LDS
    #   for A; b_depth / b_slot / b_every steer the direct loads of whichever operand is direct)
    b_depth: int = 1  # K tiles the direct B loads run ahead (VGPR fragment sets in flight = b_depth + 1)
    b_slot: int = 0  # MFMA slot of a tile's first direct B fragment load
    b_every: int = 2  # MFMA slots between direct B fragment loads
    tail_mode: str = "static"  # | "clamp": the loop runs to the last tile, refills past K re-load the last tile
    # ---- grid ----
    xcd_block: str = "4x8"  # CTA tile block per XCD (rows x cols of tiles) | "none"
    # ---- epilogue ----
    store_nt: bool = False  # non-temporal C stores (buffer store slc)
    line_merge: bool = False  # line2m: no line release barrier -- each tile issues the lines of the tile two ahead (not
    #   three) after its own acquire barrier (which already orders every wave's reads of the slots they overwrite): one
    #   barrier per tile instead of ~1.75, one tile less of line lead (line_early: issued right after the acquire)
    epi_lds: bool = False  # swap_ab False: C staged through LDS per 16-row block of the wave tile (the bf16 pairs as
    #   ds_write_b32, read back as 16-byte row pieces) and stored as buffer_store_dwordx4 (fm 16 rows x 2 fn pieces /
    #   64 per wave: 4 fm stores instead of 2 fm fn dword stores); store_nt applies to them
    # ---- tile-config library (every field default = off: the IR of every spec without them is unchanged) ----
    runtime_shape: bool = False  # M, N, K, lda, ldb, ldc and per-entry batch strides are kernel arguments (after c);
    #   m / n / k above are the build's reference shape: the static prologue / tail are built for k, the main loop
    #   runs (K / 128 - k / 128) / body more iterations (``runtime_problem_ok`` lists the host-side predicates)
    batch: int = 1  # strided-batched entries on grid z: entry z reads A + z sA, B + z sB, ... (static: packed
    #   strides of the reference shape; runtime_shape: the stride arguments); grid_of() launches ``batch`` entries
    depth_u: int = 128  # K per barrier period (hipBLASLt DepthU): 128 | 256 | 512. One acquire barrier per depth_u /
    #   128 K tiles; ``stages`` counts 128-K tiles of the ring and must hold >= 2 depth_u periods
    cta_order: str = "xcd_block"  # CTA -> output tile: "xcd_block" (static xcd_block blocks) | "grouped": XCD x gets
    #   the contiguous tile range x * T / 8 .. (when the CTA count per entry is a multiple of 8), walked in groups of
    #   ``group_m`` tile rows (runtime grid) | "linear": row-major tiles
    group_m: int = 8  # cta_order grouped: tile rows per group
    occupancy: int = 1  # target work-groups per CU (waves_per_eu hint = ceil(waves x occupancy / 4); LDS must fit)
    stage_kib: bool = False  # stage-ring allocations (operand and scale images) padded to whole KiB: the direct
    #   compiler's LDS DMA tags are KiB ranges, so a read of one stage never waits for a refill of another
    schedule: str = "dp"  # work decomposition: "dp" = one CTA per output tile | "persistent": a fixed grid (kernel
    #   arguments nbatch, ngrid after the strides; ``persistent_grid``), CTA w on XCD w % 8 walks that XCD's contiguous
    #   range of (entry, tile) work units; runtime_shape only
    magic_div: bool = False  # runtime_shape: the CTA map's runtime divisions (grouped span / last group / persistent
    #   tiles per entry; linear: N tiles) by host-computed magic numbers (``magic_u31``; 6 i32 kernel arguments last)
    kernarg_preload: bool = False  # every parameter ``inreg``: the leading 14 kernarg dwords preloaded into SGPRs
    persist_overlap: bool = False  # schedule persistent: unit u's prologue DMAs issue before unit u-1's epilogue
    #   (accumulators, C pointer and tile origin carried across units; the epilogue skips its vmcnt(0))
    k_tail: bool = False  # runtime_shape: K a multiple of 32 (MX block), not of 128: ceil(K / 128) K tiles. The host pads
    #   B and both scale images to whole 128-K tiles with zeros (``layout.pad_k``); A stays as produced -- its tail
    #   bytes multiply B's zero codes, and past the matrix they read 0 (exact-range A descriptor)
    edge_tiles: bool = False  # runtime_shape: M / N need not be tile multiples -- the last tile row / column starts at
    #   M - tile_m / N - tile_n (it recomputes, bit-identically, rows its neighbour stores too: no masks, no
    #   out-of-range access); the host packers build the last pre-tiled B block / scale image from those rows
    scale_raw: bool = False  # scale_layout wtile + scale_src vgpr: the canonical [rows, K/32] scale bytes as produced
    #   (no host packing): per 4 K tiles a lane loads its row's 16 bytes (one dwordx4 per wave-side dword) and the
    #   wtile op_sel dwords come from a 4x4 byte transpose across the 16-lane groups (raw_scale_transpose:
    #   v_permlane32_swap + v_perm, v_permlane16_swap + v_perm per K tile and dword; no LDS)
    wtile_period: int = 1  # scale_layout wtile + scale_src vgpr: K tiles per scale load -- the lane's D dwords of P
    #   consecutive K tiles are contiguous (layout "wtile{R}x{Wr}p{P}") and arrive in one buffer_load of P D <= 4
    #   dwords (P = 4 / D: a quarter of the scale VMEM instructions at D = 1)
    rt_tail: str = "fixed"  # runtime_shape: one static tail, K = k + 128 U j (U = loop body tiles; a dispatcher keeps
    #   one build per K residue) | "switch": any K = k + 128 i, one static tail per residue mod U (a uniform branch
    #   each, own vmcnt model and epilogue; doubles the tail's fragment registers: spills the 256x256 tiles)
    group_table: bool = False  # grouped (MoE) GEMM, runtime_shape: one packed A / C problem of M rows (each group's
    #   rows padded to a multiple of tile_m), B / its scales per group at the batch strides; CTA tile row r multiplies
    #   group gtab[r] (int32 device table, one entry per tile_m rows; kernel argument ``gtab`` after every other
    #   argument; scalar-loaded once per tile). Groups with no rows get no tile rows: their weights are never read.
    #   M-contiguous A (am family): one [K][tile_m] slab per tile row (lda = tile_m 6 / 8; slab r at r K lda bytes),
    #   so a tile's tokens stay one dense block whatever Mpad is
    splitk: str = "none"  # | "fused": split-K / stream-K work units (splitk.py): DP tiles + tiles split along K into
    #   sk_split units, fp32 partials in a workspace, the last arriving wave of a tile sums them in split order (no
    #   waits: deadlock-free; deterministic). Kernel arguments ws, cnt, sk_dp, sk_sc (s | c << 16) after every other one
    sk_split: int = 1  # splitk: units per split tile (host plan; not in the kernel IR / name)
    sk_hybrid: bool = False  # splitk: full waves of tiles data-parallel, only the last partial wave split (host plan)
    sk_wave: int = 0  # splitk hybrid: tiles per wave (0 = 256 CUs x the LDS occupancy; host plan)
    sk_fix: str = "last"  # splitk: the last arriving wave reduces (wait-free) | "owner": split s - 1 polls its peers'
    #   flags (co-resident units only: problem_ok)
    sk_static: bool = False  # splitk: sk_split baked into the kernel (unit map by constants, the reducer's loads of all
    #   s partials issued at once, then summed in split order); in the IR / name, unlike the host-only sk_split
    sk_release: str = "fence"  # splitk: partials released by an agent-scope fence | "wt": write-through stores + vmcnt(0)
    sk_kargs: str = "packed"  # splitk: kernel arguments end at 128 B (s | c << 16 in one dword: one kernarg round trip)
    #   | "wide" (s, c in two dwords: 132 B, the prologue's second, serialized kernarg load; kept for A/B)
    m_mask: bool = False  # splitk, runtime_shape: any M >= 1 -- A rows past M read as zeros (exact-range descriptor),
    #   ceil(M / tile_m) tile rows without the edge shift, C stores of rows >= M predicated off (decode)
    # ---- measurement / test only ----
    trim_xcd: int = 3  # probe xcd_trim: the XCD whose CTAs skip ``trim_iters`` K-loop iterations (wrong result:
    trim_iters: int = 1  #   the timing upper bound of moving that much K work off the slowest XCD)
    probe: str = "none"  # | "mfma_only" | "fill_only" | "dtl_only" | "lds_only" (not GEMMs: ceiling / fill floors)
    mutant: str = "none"


#: Every tunable knob with its legal values (the search space; first value = default).
#: ``validate_spec`` decides which combinations build.
KNOBS: Dict[str, tuple] = {
    "atom": ("16x16x128", "32x32x64"),
    "tile_m": (256, 128),
    "tile_n": (256, 128),
    "waves_m": (2, 1, 4),
    "waves_n": (2, 1, 4),
    "stages": (3, 2, 4),
    "swizzle": ("rot48", "none"),
    "canon_dtl": ("band", "interleave"),
    "canon_read": ("b64x3", "b96x2", "b128b64"),
    "operand_layout": ("c0c1", "canonical", "c0c1p"),
    "a_layout": ("", "canonical", "mcontig"),
    "mc_dma": ("x4", "x3"),
    "mc_copy": ("auto", "late", "fence"),
    "scale_layout": ("tile", "kstep", "raw", "wtile"),
    "depth_u": DEPTH_US,
    "cta_order": CTA_ORDERS,
    "group_m": (8, 1, 2, 4, 16),
    "occupancy": (1, 2, 3, 4),
    "stage_kib": (False, True),
    "runtime_shape": (False, True),
    "rt_tail": ("fixed", "switch"),
    "schedule": ("dp", "persistent"),
    "wtile_period": (1, 2, 4),
    "acc_vgpr_spill": (False, True),
    "line_dtl_order": ("strided", "wave", "block"),
    "scale_raw": (False, True),
    "edge_tiles": (False, True),
    "k_tail": (False, True),
    "persist_overlap": (False, True),
    "splitk": ("none", "fused"),
    "sk_split": (1, 2, 3, 4, 6, 8, 16),
    "sk_hybrid": (False, True),
    "sk_static": (False, True),
    "sk_fix": ("last", "owner"),
    "sk_release": ("fence", "wt", "wt1"),
    "sk_kargs": ("packed", "wide"),
    "m_mask": (False, True),
    "magic_div": (False, True),
    "kernarg_preload": (False, True),
    "scale_src": ("lds", "vgpr"),
    "a_scale_src": ("tile", "ring", "pring"),
    "b_scale_src": ("tile", "pring"),
    "as_period": (4, 3),
    "bs_period": (4, 3),
    "bs_lead": (4, 2, 3, 5),
    "bs_slot": (1, 8, 16, 32, 48),
    "as_lead": (4, 2, 3, 5),
    "as_slot": (1, 8, 16, 32, 48),
    "as_perm": (-1, 8, 16, 32, 56),
    "acc_pin": (False, True),
    "acc_agpr": (True,),
    "peel_first": (False, True),
    "serpentine": (True, False),
    "m_outer": (True, False),
    "pin_mfma": (True, False),
    "read_split": (True, False),
    "lds_alias_scopes": (True, False),
    "frag_mode": ("jit", "prefetch"),
    "acq_slot": (-1, 0, 1, 2, 4, 8),
    "read_every": (2, 1, 3, 4),
    "read_delay": (0, 2, 4, 8),
    "read_lead": (4, 2, 3, 6, 8, 12, 16, 64),
    "dtl_delay": (0, 1, 2, 4, 8),
    "dtl_spread": (0, 2, 3, 4, 5),
    "scale_slot": (40, 2, 16, 32, 56),
    "setprio": (0, 1, 2, 3),
    "tile_lgkm0": (False, True),
    "unroll": (1, 2, 3, 4, 8),
    "reg_stages": (0, 1, 2),
    "reg_slot": (0, 8, 16),
    "reg_every": (4, 2, 3),
    "release_slot": (-1, 8, 16, 24, 32),
    "skew_dtl": (0, 1, 2, 3, 4),
    "skew_read": (0, 1, 2, 4),
    "carry": (0, 1, 2, 4, 9),
    "carry_slot": (56, 48, 52, 60),
    "scale_dtl": ("dword", "x4w", "split"),
    "k_rot": (0, 1, 2, 3, 4),
    "k_phase": (2, 4, 8),
    "k_phase_tiles": (8, 16, 24),
    "swap_ab": (True, False),
    "store_nt": (False, True),
    "epi_lds": (False, True),
    "line_merge": (False, True),
    "mma_order": ("rows", "pairs", "blk2"),
    "rsrc_once": (False, True),
    "canon_rsrc_once": (False, True),
    "canon_ring": ("stage", "line2", "line3", "line2m"),
    "line_release": (26, 18, 22, 30, 34, 40),
    "line_spread": (0, 1, 2, 3),
    "ring_ops": ("all", "A"),
    "b_reg": (False, True),
    "mini_copy_gap": (8, 0, 4, 12, 16),
    "mini_copy_at": (-1, 16, 24, 30),
    "mini_copy_at_b": (-1, 30, 40, 48),
    "ring_kib": (False, True),
    "mini_layout": ("v2", "v3", "v4"),
    "mini_src": ("copy", "dtl"),
    "line_release_b": (-1, 30, 34, 40, 48),
    "line_early": (False, True),
    "line_reads_first": (False, True),
    "dtl_imm": (False, True),
    "dtl_aux": (0, 1, 2, 3),
    "b_direct": (False, True),
    "a_direct": (False, True),
    "a_fill": ("dtl", "c0c1"),
    "b_depth": (1, 2),
    "b_slot": (0, 8, 16, 32),
    "b_every": (2, 1, 4),
    "tail_mode": TAIL_MODES,
    "xcd_block": ("4x8", "none", "8x4", "2x16", "16x2"),
}


def _defaults() -> Dict[str, object]:
    return {f.name: f.default for f in fields(Mxfp6GemmSpec)}


def _adv(spec: Mxfp6GemmSpec) -> int:
    """Tiles by which fragment reads run ahead of their MFMAs (prefetch 1, jit 0)."""
    return 1 if spec.frag_mode == "prefetch" else 0


def validate_spec(spec: Mxfp6GemmSpec) -> Tuple[bool, str]:
    """``(ok, reason)``: structural validity of ``spec``."""
    try:
        geo = atom_geometry(spec.atom)
    except ValueError as e:
        return False, str(e)
    for name, legal in (("swizzle", SWIZZLES), ("xcd_block", XCD_BLOCKS),
                        ("mutant", MUTANTS + LIB_MUTANTS + GROUP_MUTANTS + SK_MUTANTS + MASK_MUTANTS + LINE_MUTANTS),
                        ("frag_mode", FRAG_MODES), ("operand_layout", OPERAND_LAYOUTS), ("probe", PROBES),
                        ("scale_layout", SCALE_LAYOUTS), ("tail_mode", TAIL_MODES),
                        ("scale_src", SCALE_SRCS), ("elem_fmt", tuple(ELEM_FMTS)),
                        ("required_compiler", COMPILERS)):
        if getattr(spec, name) not in legal:
            return False, f"{name} must be one of {legal}"
    nw = spec.waves_m * spec.waves_n
    if nw not in (1, 2, 4, 8, 16):
        return False, "waves_m * waves_n must be 1, 2, 4, 8 or 16"
    if spec.m % spec.tile_m or spec.n % spec.tile_n:
        return False, "M / N must be multiples of tile_m / tile_n"
    if spec.k % 512 and not ((spec.splitk != "none" or _library_mode(spec)) and spec.scale_layout == "wtile"
                             and spec.k % (128 * (4 if spec.scale_raw else max(1, spec.wtile_period))) == 0):
        # (splitk: an exact-K short build of per-tile wtile scales may stop on any whole scale period)
        return False, "K must be a multiple of 512 (four 128-K tiles per scale period)"
    if spec.tile_m % spec.waves_m or spec.tile_n % spec.waves_n:
        return False, "tile must split evenly over the waves"
    wt_m, wt_n = spec.tile_m // spec.waves_m, spec.tile_n // spec.waves_n
    wtile = spec.scale_layout == "wtile"
    if wtile:  # per-wave scale images: any whole number of 16-row MFMA tiles per wave (16x16x128)
        if geo.rows != 16 or wt_m % 16 or wt_n % 16:
            return False, "scale_layout wtile needs the 16x16x128 atom and wave tiles of whole 16-row MFMA tiles"
        if max(wtile_dwords(wt_m), wtile_dwords(wt_n)) > 4:
            return False, "scale_layout wtile: at most 16 MFMA tiles per wave side (4 scale dwords per lane)"
    elif wt_m % (4 * geo.rows) or wt_n % (4 * geo.rows):
        return False, f"wave tile must be a multiple of {4 * geo.rows} (scale groups of 4 MFMA tiles)"
    if spec.a_layout not in A_LAYOUTS:
        return False, f"a_layout must be one of {A_LAYOUTS} ('' = operand_layout)"
    if spec.a_layout and spec.a_layout != spec.operand_layout and spec.operand_layout not in ("c0c1", "c0c1p"):
        return False, "a mixed layout needs B pre-tiled (c0c1 / c0c1p)"
    if operand_layout_of(spec, "B") == "c0c1p" and (spec.tile_n // spec.waves_n) % 32:
        return False, "c0c1p B: wave sides of whole 32-row fragment pairs"
    for rows, X in ((spec.tile_m, "A"), (spec.tile_n, "B")):
        lay = operand_layout_of(spec, X)
        if lay == "mcontig":
            if rows % 32:
                return False, "mcontig: tile_m a multiple of 32 (an even number of 16-row blocks)"
        elif lay == "canonical":
            if rows % ROWS_PER_GROUP or (spec.canon_dtl == "band" and (rows // ROWS_PER_GROUP) % nw):
                return False, "tile rows must give every wave whole 32-row DTL groups"
            if spec.canon_dtl == "interleave" and (rows * ROW_BYTES) % (1024 * nw) and not (
                    _library_mode(spec) and spec.canon_ring == "stage" and (rows * ROW_BYTES) % 1024 == 0):
                # an uneven split (waves share the last 1 KiB chunk, ``dtl_chunks``): library structures only
                return False, "canon_dtl interleave: a stage image must split into whole 1 KiB chunks per wave"
        elif (rows * ROW_BYTES) % 1024:
            return False, "c0c1: a stage image of whole 1 KiB DTL instructions (tile rows a multiple of 32)"
        elif (rows * ROW_BYTES // 1024) % nw and (spec.dtl_imm or not _library_mode(spec)):
            return False, "c0c1: a stage image must split into whole 1 KiB DTL instructions per wave"
    KS = spec.depth_u // 128
    if spec.depth_u not in DEPTH_US:
        return False, f"depth_u must be one of {DEPTH_US}"
    if KS == 1 and not 2 <= spec.stages <= (8 if _library_mode(spec) else 4):
        return False, "stages must be 2..4 (tile-library specs: 2..8, deep rings for the latency-bound small tiles)"
    if KS > 1 and (spec.stages % KS or not 2 * KS <= spec.stages <= 4 * KS):
        return False, f"depth_u {spec.depth_u}: stages (128-K tiles) a multiple of {KS}, 2 to 4 depth_u periods"
    if not 1 <= spec.occupancy <= 8:
        return False, "occupancy must be 1..8"
    if lds_bytes(spec) * spec.occupancy > 160 * 1024:
        return False, f"LDS {lds_bytes(spec)} B x occupancy {spec.occupancy} exceeds 160 KiB"
    if spec.acc_agpr and acc_regs(spec) > 256 and not spec.acc_vgpr_spill:
        return False, "accumulators exceed 256 AGPRs (acc_vgpr_spill puts the rest in VGPRs)"
    if spec.acc_vgpr_spill and (not spec.acc_agpr or spec.acc_pin or spec.persist_overlap or acc_regs(spec) > 384):
        return False, "acc_vgpr_spill: acc_agpr, no acc_pin / persist_overlap, at most 384 accumulator registers"
    if spec.k // 128 < spec.stages + 2:
        return False, "K too small for the ring"
    if spec.read_every < 1 or spec.read_delay < 0 or spec.dtl_delay < 0 or spec.read_lead < 1:
        return False, "bad read / DTL slot knobs"
    if spec.unroll not in (1, 2, 3, 4, 8):
        return False, "unroll must be 1..4 or 8 (8: a 24-tile body for the line ring next to 3 stages)"
    n_mfma = mfmas_per_tile(spec)
    if not -1 <= spec.acq_slot < n_mfma - 1:
        return False, "acq_slot out of range"
    if spec.frag_mode == "jit" and spec.acq_slot != -1:
        return False, "jit reads start at the top of the tile: acq_slot must be -1"
    if not 0 <= spec.setprio <= 3:
        return False, "setprio must be 0..3"
    if spec.cta_order not in CTA_ORDERS:
        return False, f"cta_order must be one of {CTA_ORDERS}"
    if spec.cta_order == "xcd_block" and spec.xcd_block != "none":
        bm, bn = (int(x) for x in spec.xcd_block.split("x"))
        gm, gn = spec.m // spec.tile_m, spec.n // spec.tile_n
        if gm % bm or gn % bn or (gm // bm) * (gn // bn) != NUM_XCDS:
            return False, f"xcd_block {spec.xcd_block} does not tile the {gm}x{gn} CTA grid into {NUM_XCDS}"
    if spec.cta_order != "xcd_block" and (spec.xcd_block != "none" or spec.group_m < 1):
        return False, "cta_order grouped / linear: xcd_block none, group_m >= 1"
    if spec.scale_layout == "raw" and (spec.scale_src != "lds" or spec.scale_dtl != "dword" or geo.rows != 16):
        return False, "scale_layout raw needs scale_src lds, scale_dtl dword and the 16x16x128 atom"
    if spec.scale_layout in ("tile", "raw"):
        legal = (64, 128) if geo.rows == 16 else (128,)
        if wt_m not in legal or wt_n not in legal:
            return False, f"scale_layout tile needs wave tiles of {legal} rows"
        if spec.scale_src == "lds" and (wt_m != 128 or wt_n != 128):
            return False, "scale_src lds needs 128-row wave tiles (use scale_src vgpr)"
    if spec.scale_src == "lds" and spec.scale_layout not in ("tile", "raw", "wtile"):
        return False, "scale_src lds needs scale_layout tile, raw or wtile"
    if spec.scale_src == "lds" and not wtile and any(r * 4 % (256 * nw) for r in (spec.tile_m, spec.tile_n)):
        return False, "scale_src lds: whole 256-byte scale DTLs per wave and operand"
    if wtile and (spec.scale_dtl not in ("dword", "split") or scale_rings(spec)):
        return False, "scale_layout wtile: dword / split scale DTLs, no scale rings"
    if spec.scale_src == "lds" and spec.probe == "mfma_only":
        return False, "the mfma_only probe loads its scales to VGPRs (scale_src vgpr)"
    if spec.scale_dtl not in ("dword", "x4w", "split"):
        return False, "scale_dtl must be dword, x4w or split"
    if spec.scale_dtl == "split" and (not wtile or spec.scale_src != "lds"):
        return False, "scale_dtl split: wtile scales in LDS"
    if spec.scale_dtl == "x4w" and (spec.scale_src != "lds" or nw != 4 or spec.tile_m * 4 != 1024
                                    or spec.tile_n * 4 != 1024 or spec.reg_stages or spec.b_direct):
        return False, "scale_dtl x4w needs LDS scales, 4 waves, 256-row tiles (1 KiB per operand), no reg / direct-B"
    if spec.scale_dtl == "x4w" and len(line_operands(spec)) == 2:
        # its vmcnt safety needs operand DTLs after it in the same refill (both operands in line rings: none)
        return False, "scale_dtl x4w needs a stage-ring operand (not both operands in line rings)"
    if spec.a_scale_src not in A_SCALE_SRCS or spec.b_scale_src not in B_SCALE_SRCS:
        return False, f"a_scale_src must be one of {A_SCALE_SRCS}, b_scale_src one of {B_SCALE_SRCS}"
    if spec.probe in AS_RAW_PROBES and spec.a_scale_src != "ring":
        return False, f"probe {spec.probe} needs a_scale_src ring"
    if spec.probe in ("as_hazard", "as_stale") and spec.a_scale_src == "tile":
        return False, f"probe {spec.probe} needs an A scale ring (a_scale_src ring / pring)"
    if spec.probe in BS_PROBES and spec.b_scale_src != "pring":
        return False, f"probe {spec.probe} needs b_scale_src pring"
    srings = scale_rings(spec)
    if srings:
        if (spec.scale_src != "lds" or spec.scale_layout != "tile" or spec.scale_dtl != "dword" or geo.rows != 16
                or any((wt_m if X == "A" else wt_n) != 128 for X in srings)):
            return False, ("scale rings (a_scale_src ring / pring, b_scale_src pring) need LDS tile scales (scale_dtl "
                           "dword), the 16x16x128 atom and 128-row wave tiles")
        if (spec.frag_mode != "prefetch" or spec.carry or spec.reg_stages or spec.b_direct or spec.b_reg
                or spec.tail_mode != "static" or spec.k_rot or spec.release_slot >= 0
                or spec.probe in ("mfma_only", "fill_only", "dtl_only", "lds_only")):
            return False, ("scale rings need prefetch fragments, a static tail, no carry / register fill / direct or "
                           "register B / k_rot / split release / fill probes")
        u_body = spec.stages * spec.unroll  # tile scales: scale period 1
        for X, (kind, P, lead, slot) in srings.items():
            rows = spec.tile_m if X == "A" else spec.tile_n
            if kind == "raw" and (rows * 16) % (1024 * nw):
                return False, "a_scale_src ring: every wave one whole 1 KiB DTL (64 rows x 16 bytes) per period"
            if kind == "packed" and (P not in PRING_PERIODS or rows != 64 * nw):
                return False, (f"{X} scale pring: period 3 or 4 and {64 * nw} tile rows (one dwords=P DTL lane per "
                               "wave row)")
            if u_body % (P * AS_SLOTS):
                return False, f"{X} scale ring: the loop body (stages x unroll) a multiple of {2 * P} (static ring slot)"
            if lead not in (2, 3, 4, 5) or not 0 <= slot < mfmas_per_tile(spec) - 1:
                return False, f"{X} scale ring: lead 2..5, DTL slot within the tile"
            if lead == 5 and slot <= spec.acq_slot:
                return False, f"{X} scale ring: lead 5 overwrites a slot read up to this tile's acquire: DTL after it"
        if spec.a_scale_src == "ring" and spec.as_period != AS_PERIOD:
            return False, "a_scale_src ring (raw scales): as_period 4 (16 bytes per row)"
    if not -1 <= spec.as_perm < mfmas_per_tile(spec) or (spec.as_perm >= 0 and spec.a_scale_src != "ring"):
        return False, "as_perm: within the tile, a_scale_src ring only"
    if spec.k_rot in (3, 4) and (spec.k_phase < 2 or spec.k_phase_tiles < 1
                            or spec.k_phase * spec.k_phase_tiles > spec.k // 128
                            or (ring_slots(spec) and spec.k_phase_tiles % line_period(spec))):
        return False, ("k_rot 3 / 4: k_phase >= 2, k_phase x k_phase_tiles within K, k_phase_tiles a multiple of the line "
                       "period")
    if spec.k_rot not in (0, 1, 2, 3, 4) or (spec.k_rot and (spec.scale_src != "lds" or spec.b_direct or spec.reg_stages
                                                    or spec.xcd_block == "none")):
        return False, "k_rot needs LDS scales, an XCD block order, no direct-B / register fill"
    if spec.mma_order not in ("rows", "pairs", "blk2"):
        return False, "mma_order must be rows, pairs or blk2"
    if spec.mma_order != "rows" and (atom_geometry(spec.atom).rows != 16
                                     or (spec.tile_m // spec.waves_m // 16) % 2 or (spec.tile_n // spec.waves_n // 16) % 2):
        return False, "mma_order pairs / blk2 need the 16x16x128 atom and an even number of MFMA tiles per wave side"
    if spec.rsrc_once and spec.operand_layout == "canonical":
        return False, "rsrc_once needs pre-tiled operands (c0c1 / c0c1p)"
    if spec.operand_layout == "c0c1p" and (geo.rows != 16 or spec.b_direct):
        return False, "c0c1p (paired c1 reads) needs the 16x16x128 atom and no direct B"
    if spec.a_direct and (not _library_mode(spec) or spec.b_direct or operand_layout_of(spec, "A") != "canonical"
                          or spec.canon_ring != "stage" or geo.rows != 16 or spec.probe != "none" or spec.m_mask):
        return False, "a_direct: tile-library spec, canonical A in the stage ring, no b_direct / m_mask, 16x16x128, no probe"
    if spec.b_direct and (spec.operand_layout != "c0c1" or geo.rows != 16 or spec.probe != "none"
                          or (spec.frag_mode != "jit" and not _library_mode(spec) and spec.canon_ring != "line2m")):
        return False, ("b_direct needs a c0c1 B, the 16x16x128 atom and no probe (static presets: jit fragments or a line2m A "
                       "ring; "
                       "tile-library specs: prefetch or jit)")
    if not spec.swap_ab and geo.rows != 16:
        return False, "swap_ab=False (A as MFMA src0, DPP-paired dword stores) is built for the 16x16x128 atom only"
    if spec.canon_read not in ("b64x3", "b96x2", "b128b64"):
        return False, "canon_read must be b64x3, b96x2 or b128b64"
    if spec.canon_read == "b128b64" and (spec.required_compiler != "rust" or not _library_mode(spec)):
        return False, "canon_read b128b64 (inline-asm LDS reads, lever trtag): direct compiler, tile-library specs"
    if spec.b_reg and (spec.frag_mode != "prefetch" or spec.stages != 2 or operand_layout_of(spec, "B") != "c0c1"
                       or spec.b_direct or spec.reg_stages or spec.carry or spec.tail_mode != "static"
                       or spec.probe not in ("none", "stamps") or geo.rows != 16):
        return False, "b_reg needs prefetch fragments, 2 stages, a c0c1 B, the 16x16x128 atom and a static tail"
    if spec.ring_ops not in ("all", "A"):
        return False, "ring_ops must be all or A"
    if spec.mini_copy_gap < 0:
        return False, "mini_copy_gap must be >= 0"
    if spec.mini_copy_at != -1 and (spec.canon_ring != "line2m"
                                    or not spec.acq_slot <= spec.mini_copy_at < spec.line_release):
        return False, ("mini_copy_at needs line2m and a slot from the acquire to before the line release (the head "
                       "line's slot is refilled after it)")
    if spec.mini_copy_at_b != -1 and (spec.canon_ring != "line2m" or spec.line_release_b < 0
                                      or not spec.acq_slot <= spec.mini_copy_at_b < spec.line_release_b):
        return False, "mini_copy_at_b needs line2m, line_release_b and a slot from the acquire to before B's release"
    if spec.mini_layout not in ("v2", "v3", "v4"):
        return False, "mini_layout must be v2, v3 or v4"
    if spec.mini_src not in ("copy", "dtl") or (spec.mini_src == "dtl" and (spec.canon_ring != "line2m"
                                                                          or spec.mini_layout != "v4" or nw != 4)):
        return False, "mini_src must be copy or dtl; dtl needs line2m, mini_layout v4 and 4 waves"
    if spec.line_spread < 0:
        return False, "line_spread must be >= 0"
    if spec.canon_ring not in ("stage", "line2", "line3", "line2m"):
        return False, "canon_ring must be stage, line2, line3 or line2m"
    if spec.canon_ring == "line2m" and geo.rows != 16:
        return False, "canon_ring line2m needs the 16x16x128 atom"
    if spec.canon_ring == "line2m" and (spec.mini_layout == "v3" or spec.mini_src == "dtl") and any(
            (spec.tile_m if X == "A" else spec.tile_n) != 64 * nw for X in line_operands(spec)):
        return False, "line2m remnant layout v3 / DMA remnant: tile rows = 64 x waves (v2 / v4 copy any rows)"
    if spec.canon_ring in ("line2", "line3", "line2m"):
        R = ring_slots(spec)
        lines = list(line_operands(spec))
        P_sc = 1 if spec.scale_layout in ("tile", "raw", "wtile") else 4 // (128 // geo.k)  # tiles per scale period
        U = spec.stages * P_sc // _gcd(spec.stages, P_sc) * spec.unroll  # loop body
        if not lines:
            return False, f"canon_ring {spec.canon_ring} needs a canonical operand"
        if (spec.frag_mode != "prefetch" or spec.carry or spec.reg_stages or (spec.b_direct and "B" in lines)
                or spec.release_slot >= 0 or spec.probe not in ("none", "dtl_only", "stamps", "line_lead3",
                                                                       "l2m_nomini", "l2m_nodual", "l2m_norelease",
                                                                       "l2m_early", "l2m_nocopy", "l2m_lead4",
                                                                       "l2m_copyread", "l2m_copywrite",
                                                                       "l2m_copyvm0", "ldtl_noA", "ldtl_noB",
                                                                       "l2m_nocopyB", "ldtl_contigB", "l2m_earlyB",
                                                                       *AS_PROBES, *BS_PROBES, "sc_noB", "sc_noAB")
                or spec.tail_mode != "static"):
            return False, (f"canon_ring {spec.canon_ring} needs prefetch fragments, "
                           "a static "
                           "tail and no carry / register fill / direct B / split release / fill probes")
        per = line_period(spec)
        if (spec.k // 128) % per or U % per:
            return False, f"canon_ring {spec.canon_ring}: K / 128 and the loop body (stages x unroll) multiples of {per}"
        if spec.k_rot and (spec.k // 128 // NUM_XCDS) % per:
            return False, f"canon_ring {spec.canon_ring} with k_rot: the per-XCD K offset a multiple of {per} tiles"
        if any((spec.tile_m if X == "A" else spec.tile_n) % max(128, 8 * nw) for X in lines):
            return False, (f"canon_ring {spec.canon_ring}: tile rows a multiple of 128 (16 row classes x 8-row DTL "
                           "blocks) per wave")
        if spec.line_release_b >= 0:
            if R != 2 or tuple(lines) != ("A", "B") or not spec.line_reads_first:
                return False, "line_release_b needs both operands in a line2 / line2m ring and line_reads_first"
            last_b = last_ring_read(spec, "B")
            if not last_b < spec.line_release_b < n_mfma - 1:
                return False, (f"line_release_b {spec.line_release_b} must follow B's last fragment read (slot "
                               f"{last_b}) and precede the tile's last MFMA")
        last_read = last_ring_read(spec, "A" if spec.line_release_b >= 0 else None)
        if R == 2 and not last_read < spec.line_release < n_mfma - 1:
            return False, (f"line_release {spec.line_release} must follow the last fragment read (slot {last_read}) "
                           f"and precede the tile's last MFMA")
    if spec.reg_stages and (spec.frag_mode != "jit" or spec.operand_layout == "canonical" or spec.b_direct
                            or spec.probe not in ("none",) or not 0 <= spec.reg_stages <= 3 or spec.reg_every < 1):
        return False, "reg_stages needs jit fragments, c0c1 operands, no direct B / probe, 1..3 stages"
    if spec.release_slot >= 0 and (spec.frag_mode != "jit" or spec.release_slot >= n_mfma - 1):
        return False, "release_slot needs frag_mode jit and a slot before the tile's last MFMA"
    if (spec.skew_dtl or spec.skew_read) and (spec.frag_mode != "jit" or spec.skew_dtl < 0 or spec.skew_read < 0
                                              or spec.tail_mode != "static"):
        return False, "skew_dtl / skew_read need frag_mode jit, tail_mode static, non-negative shifts"
    if spec.carry:
        if (spec.frag_mode != "jit" or spec.release_slot >= 0 or spec.reg_stages or spec.b_direct
                or spec.skew_dtl or spec.skew_read or spec.tail_mode != "static" or spec.probe != "none"
                or spec.scale_src != "lds"):
            return False, "carry needs plain jit fragments (no release / reg / direct-B / skew / probe), LDS scales, static tail"
        order = mfma_order(spec)
        first: Dict = {}
        for x, (t, mi, ni) in enumerate(order):
            first.setdefault(("A", mi, t), x)
            first.setdefault(("B", ni, t), x)
        keys = sorted(first, key=lambda key: (first[key], key[0]))
        if not 1 <= spec.carry <= len(keys) or not 0 <= spec.carry_slot < n_mfma - 1:
            return False, "carry / carry_slot out of range"
        own = [max(-1, first[key] - spec.read_lead) for key in keys[spec.carry:]]
        if own and max(own) >= spec.carry_slot:
            return False, "carry_slot must follow every own-tile fragment read (they share the moved barrier)"
    if not 0 <= spec.dtl_aux <= 3:
        return False, "dtl_aux must be 0..3"
    if not -1 <= spec.b_aux <= 3:
        return False, "b_aux must be -1..3"
    if not 1 <= spec.b_depth <= 3 or spec.b_slot < 0 or spec.b_every < 1:
        return False, "bad direct-B knobs"
    if spec.tail_mode == "clamp":
        geo_p = 1 if spec.scale_layout in ("tile", "raw", "wtile") else 4 // (128 // geo.k)
        u_body = spec.stages * geo_p // _gcd(spec.stages, geo_p) * spec.unroll
        if (spec.k // 128) % u_body % geo_p:
            return False, "clamp: the static head (K tiles mod loop body) must be whole scale periods"
    if spec.probe != "none" and spec.mutant != "none":
        return False, "probe and mutant are exclusive"
    if _library_mode(spec):
        return _validate_library(spec, geo, nw)
    if spec.mutant in LIB_MUTANTS + GROUP_MUTANTS + SK_MUTANTS + MASK_MUTANTS:
        return False, f"mutant {spec.mutant} needs a tile-library spec"
    return True, ""


def _library_mode(spec: Mxfp6GemmSpec) -> bool:
    """A spec that uses any tile-library field (runtime shape, batch, depth_u, CTA order, occupancy, per-wave scale
    images, M-contiguous A): such specs may also split DTL images unevenly over the waves (``dtl_chunks``)."""
    return (spec.runtime_shape or spec.batch != 1 or spec.depth_u != 128 or spec.cta_order != "xcd_block"
            or spec.occupancy != 1 or spec.scale_layout == "wtile" or spec.a_layout == "mcontig"
            or spec.schedule != "dp" or spec.persist_overlap or spec.magic_div or spec.kernarg_preload
            or spec.edge_tiles or spec.group_table or spec.splitk != "none" or spec.m_mask)


def _validate_library(spec: Mxfp6GemmSpec, geo, nw: int) -> Tuple[bool, str]:
    """The structures the tile-library fields are built for (a subset of the knob space; the rest stays refused
    until it is built and verified)."""
    if not isinstance(spec.batch, int) or spec.batch < 1:
        return False, "batch must be an int >= 1"
    if spec.splitk != "none" or spec.m_mask or spec.mutant in SK_MUTANTS + MASK_MUTANTS:
        from .splitk import validate as _sk_validate

        ok, why = _sk_validate(spec)
        if not ok:
            return ok, why
    exotic = (spec.carry or spec.reg_stages or spec.b_reg or spec.release_slot >= 0
              or spec.skew_dtl or spec.skew_read or spec.k_rot or spec.tail_mode != "static" or scale_rings(spec)
              or spec.scale_layout == "raw" or spec.scale_dtl not in ("dword", "split"))
    if exotic:
        return False, ("tile-library fields: no carry / register fill / register B / split release / skew / "
                       "k_rot / clamp tail / scale rings / raw scales / x4w scale DTLs (not built for them)")
    if spec.probe not in ("none", "fill_only", "mfma_only", "l2m_nocopy", "l2m_lead4"):
        return False, "tile-library fields: probe none, fill_only, mfma_only (or the line2m hazard must-fails)"
    if spec.a_layout == "mcontig" and spec.canon_rsrc_once and spec.mc_dma != "x3":
        # the 16-byte DMAs read 4 bytes past each 12-byte slot: the last K tile's past the entry, so an exact-range
        # descriptor per tile (reads 0 there); the 12-byte b96 DMAs (mc_dma x3) end at the entry: one descriptor
        return False, "a_layout mcontig: one exact-range buffer descriptor per tile (canon_rsrc_once needs mc_dma x3)"
    if spec.depth_u > 128 and spec.canon_ring != "stage":
        return False, "depth_u > 128 needs the stage ring (the line rings release per 128-K tile)"
    if spec.a_layout == "mcontig":
        if geo.rows != 16 or spec.operand_layout not in ("c0c1", "c0c1p"):
            return False, "a_layout mcontig needs the 16x16x128 atom and a pre-tiled (c0c1 / c0c1p) B"
    if spec.mutant == "batch_swap" and not (spec.runtime_shape or (spec.batch > 1 and spec.batch % 2 == 0)):
        return False, "mutant batch_swap needs a batched spec (static: an even batch)"
    if spec.mutant == "mc_copyswap" and spec.mc_copy == "auto":
        return False, "mutant mc_copyswap needs mc_copy late | fence"
    if spec.mutant == "mc_noswz" and spec.a_layout != "mcontig":
        return False, "mutant mc_noswz needs a_layout mcontig"
    if spec.mc_dma not in ("x4", "x3") or (spec.mc_dma != "x4" and spec.a_layout != "mcontig"):
        return False, "mc_dma x4 | x3, and x3 needs a_layout mcontig"
    if spec.mc_copy not in ("auto", "late", "fence") or (
            spec.mc_copy != "auto" and (spec.a_layout != "mcontig" or spec.required_compiler != "rust")):
        return False, "mc_copy auto | late | fence, and a non-auto copy needs a_layout mcontig + the rust compiler"
    if spec.mutant == "persist_drop" and spec.schedule != "persistent":
        return False, "mutant persist_drop needs schedule persistent"
    if spec.mutant == "scale_lane" and spec.scale_layout != "wtile":
        return False, "mutant scale_lane needs scale_layout wtile"
    if spec.wtile_period not in (1, 2, 4):
        return False, "wtile_period must be 1, 2 or 4"
    if spec.wtile_period > 1:
        D = max(wtile_dwords(spec.tile_m // spec.waves_m), wtile_dwords(spec.tile_n // spec.waves_n))
        if spec.scale_layout != "wtile" or spec.scale_src != "vgpr" or D * spec.wtile_period > 4:
            return False, "wtile_period > 1: scale_layout wtile, scale_src vgpr, period x dwords per lane <= 4"
        if spec.rt_tail == "switch" or spec.probe == "mfma_only" or (spec.k // 128) % spec.wtile_period:
            return False, "wtile_period > 1: rt_tail fixed, no mfma_only probe, K / 128 a multiple of the period"
    if spec.scale_raw and (spec.scale_layout != "wtile" or spec.scale_src != "vgpr" or spec.wtile_period != 1
                           or spec.rt_tail != "fixed" or spec.probe == "mfma_only"):
        return False, "scale_raw: scale_layout wtile, scale_src vgpr, wtile_period 1, rt_tail fixed, no mfma_only"
    if spec.mutant in LINE_MUTANTS and not spec.line_merge:
        return False, "mutant lmerge_lead needs line_merge"
    if spec.epi_lds and not epi_lds_fits(spec):
        return False, "epi_lds: the C staging region (waves x 16 rows) exceeds the largest operand LDS allocation"
    if spec.a_fill not in ("dtl", "c0c1"):
        return False, "a_fill must be dtl or c0c1"
    if spec.a_fill == "c0c1" and (operand_layout_of(spec, "A") != "canonical" or spec.canon_ring != "stage"
                                  or geo.rows != 16 or spec.a_direct or spec.m_mask or spec.k_tail
                                  or (spec.tile_m // 16) % (spec.waves_m * spec.waves_n) or spec.reg_stages
                                  or spec.probe not in ("none", "stamps", "mfma_only")):
        return False, ("a_fill c0c1: canonical A in the stage ring, 16x16x128, tile_m / 16 a multiple of the waves, "
                       "no a_direct / m_mask / k_tail / reg_stages / probes")
    if spec.line_merge and (spec.canon_ring != "line2m" or spec.line_release_b >= 0
                            or spec.probe not in ("none", "stamps")):
        return False, "line_merge: canon_ring line2m, no line_release_b, no line-ring probes"
    if spec.epi_lds and (spec.swap_ab or geo.rows != 16 or spec.tile_n // spec.waves_n // 16 not in (2, 4, 8, 16, 32) or
                         spec.m_mask or spec.splitk != "none" or spec.persist_overlap or spec.group_table):
        return False, ("epi_lds: swap_ab False, 16x16 atom, 2-32 (power of 2) MFMA tiles per wave along N, no m_mask / "
                       "splitk / persist_overlap / group_table")
    if spec.line_dtl_order not in ("strided", "wave", "block"):
        return False, "line_dtl_order must be strided, wave or block"
    if spec.mutant == "wper_rot" and spec.wtile_period < 2:
        return False, "mutant wper_rot needs wtile_period >= 2"
    if spec.schedule not in ("dp", "persistent"):
        return False, "schedule must be dp or persistent"
    if spec.schedule == "persistent" and (not spec.runtime_shape or spec.probe != "none"
                                          or spec.mutant == "batch_swap"):
        return False, "schedule persistent: runtime_shape, no probe / batch_swap"
    if spec.group_table and (not spec.runtime_shape or spec.mutant == "batch_swap"):
        return False, "group_table: runtime_shape, no batch_swap (one packed A / C problem, batch 1)"
    if spec.mutant in GROUP_MUTANTS and not spec.group_table:
        return False, f"mutant {spec.mutant} needs group_table"
    if spec.edge_tiles and not spec.runtime_shape:
        return False, "edge_tiles: runtime_shape"
    if spec.k_tail and (not spec.runtime_shape or spec.rt_tail != "fixed" or spec.a_direct or spec.scale_raw
                        or (operand_layout_of(spec, "A") == "canonical"
                            and (spec.canon_ring != "stage" or not spec.canon_rsrc_once))):
        return False, ("k_tail: runtime_shape, rt_tail fixed, no a_direct / scale_raw, canonical A in the stage "
                       "ring with one exact-range descriptor (canon_rsrc_once) or M-contiguous A")
    if spec.magic_div and not spec.runtime_shape:
        return False, "magic_div: runtime_shape (a static shape divides by constants)"
    if spec.persist_overlap and (spec.schedule != "persistent" or spec.rt_tail != "fixed" or spec.swap_ab):
        return False, "persist_overlap: schedule persistent, rt_tail fixed, swap_ab False"
    if spec.runtime_shape:
        if spec.cta_order == "xcd_block":
            return False, "runtime_shape: cta_order grouped or linear (the xcd_block orders are built for one grid)"
        stage_canon = [X for X in ("A", "B") if operand_layout_of(spec, X) == "canonical"
                       and X not in line_operands(spec)]
        if spec.canon_dtl != "interleave" and stage_canon:
            return False, "runtime_shape: canonical stage-ring operands with interleaved DTLs (per-operand row strides)"
    U, H, L, T = loop_plan(spec)
    if spec.rt_tail not in ("switch", "fixed"):
        return False, "rt_tail must be switch or fixed"
    if spec.runtime_shape and spec.rt_tail == "switch" and spec.scale_layout not in ("tile", "wtile"):
        return False, "rt_tail switch: per-tile scales (scale period 1)"
    if spec.runtime_shape and L < 2 and not (spec.splitk != "none" and T >= spec.stages + 2):
        # (splitk: a short build whose units all run exactly the reference K -- splitk.problem_ok enforces it)
        return False, (f"runtime_shape: the reference K must give >= 2 main-loop iterations (body {U} tiles, head "
                       f"{H}, got {L}): use k >= {library_k_ref(spec)}")
    if U % (spec.depth_u // 128):
        return False, "the loop body must hold whole depth_u periods"
    return True, ""


def _kib(spec: Mxfp6GemmSpec, nbytes: int) -> int:
    """An allocation's bytes: rounded up to whole KiB under ``stage_kib``."""
    return -(-nbytes // 1024) * 1024 if spec.stage_kib else nbytes


def wtile_dwords(wave_rows: int) -> int:
    """scale_layout wtile: scale dwords per lane of a wave side of ``wave_rows`` rows: ceil(f / 4) for f = wave_rows /
    16 MFMA tiles (op_sel = f % 4), rounded up to 1, 2 or 4 (one ds_read_b32 / b64 / b128)."""
    d = -(-(wave_rows // 16) // 4)
    return 1 if d <= 1 else (2 if d == 2 else 4 * -(-d // 4))


def scale_img_bytes(spec: Mxfp6GemmSpec, X: str) -> int:
    """Bytes of operand ``X``'s scale image per (K tile, CTA tile): rows x 4 (tile / raw), or (wtile) one
    [wave side][64 lanes][D dwords] block per wave side (``layout.pack_scales_wtile``)."""
    rows = spec.tile_m if X == "A" else spec.tile_n
    if spec.scale_layout != "wtile":
        return rows * 4
    ws = spec.waves_m if X == "A" else spec.waves_n
    return ws * 64 * 4 * wtile_dwords(rows // ws)


def mc_swizzle(nblk: int) -> Tuple[int, int]:
    """mcontig A: (a, sh) of the slot swizzle s(k) = (k >> sh) & (2^a - 1), a = min(v2(nblk), 4), sh = 4 - a: slot
    (k, b) sits at k nblk + (b ^ s(k)). For 16 consecutive K rows k0 .. k0 + 15 (k0 a multiple of 16) and any block b
    the 16-byte slots (k nblk + (b ^ s(k))) mod 16 are distinct: k nblk mod 16 takes 2^(4 - a) values (multiples of
    2^a, period 2^(4 - a) in k) and s(k) covers the remaining 2^a classes (``tests``: bank model)."""
    a = 0
    while nblk % (2 << a) == 0 and a < 4:
        a += 1
    return a, 4 - a


def dtl_chunks(n_chunks: int, nw: int) -> List[List[int]]:
    """Chunk of each wave's DTL instruction q: chunk q nw + w, the last chunk again where the count does not split
    over the waves (every wave issues ceil(n / nw) instructions, so the static vmcnt model holds for each wave; the
    duplicated DMA writes the same bytes)."""
    per = -(-n_chunks // nw)
    return [[min(q * nw + w, n_chunks - 1) for q in range(per)] for w in range(nw)]


def loop_plan(spec: Mxfp6GemmSpec) -> Tuple[int, int, int, int]:
    """(U, H, L, T) of the K loop as ``_mainloop`` builds it: loop body U tiles, static head H, L loop iterations,
    T = k / 128 tiles (the static tail is T - H - L U)."""
    geo = atom_geometry(spec.atom)
    P = 1 if spec.scale_layout in ("tile", "raw", "wtile") else 4 // (128 // geo.k)
    if spec.scale_layout == "wtile" and spec.wtile_period > 1:
        P = spec.wtile_period
    if spec.scale_layout == "wtile" and spec.scale_raw:
        P = 4
    S, A = spec.stages, _adv(spec)
    U = S * P // _gcd(S, P) * spec.unroll
    T = spec.k // 128
    clamp = spec.tail_mode == "clamp"
    if spec.probe in ("fill_only", "dtl_only", "lds_only"):
        H = T % U if clamp else 0
    elif clamp and (T % U or spec.peel_first):
        H = T % U or U
    elif spec.peel_first:
        H = P
    else:
        H = 0
    L = (T - H) // U if clamp else max(0, (T - H - S - A) // U)
    return U, H, L, T


def library_k_ref(spec: Mxfp6GemmSpec, min_iters: int = 2, k_target: Optional[int] = None) -> int:
    """The smallest reference K (a multiple of 512) whose build gives ``min_iters`` main-loop iterations; with
    ``k_target``: also in ``k_target``'s residue class (rt_tail fixed serves K = k + 128 U j)."""
    # multiples of 512 first (the shipped reference K's); then of 128 for the K residues no multiple of 512 reaches
    # (wtile scales need only whole 128-K tiles x the scale period)
    steps = (512,) if spec.scale_layout != "wtile" else (512, 128 * (4 if spec.scale_raw else max(1, spec.wtile_period)))
    for step in dict.fromkeys(steps):
        k = step
        while k_target is None or k <= k_target:
            U, H, L, T = loop_plan(replace(spec, k=k))
            if (L >= min_iters and k // 128 >= spec.stages + 2
                    and (k_target is None or (k_target - k) % (128 * U) == 0)
                    and (not ring_slots(spec) or (k // 128) % line_period(spec) == 0)):
                return k
            k += step
    raise ValueError(f"no reference K <= {k_target} with {min_iters} loop iterations in its residue class")


def runtime_problem_ok(spec: Mxfp6GemmSpec, m: int, n: int, k: int, batch: int = 1) -> Tuple[bool, str]:
    """Host-side predicates of a runtime-shape launch (``runtime_shape=True``): M / N multiples of the tile (partial
    tiles are a later step), K a multiple of 128 with K / 128 = k / 128 + j U (j >= 0, U = the loop body)."""
    if spec.mutant == "batch_swap" and (batch < 2 or batch % 2):
        # K-BUG-1: entry z of this mutant reads entry z ^ 1's A -- past the buffer for an odd entry count
        return False, "mutant batch_swap needs an even batch >= 2 (entry z reads entry z ^ 1)"
    if not spec.runtime_shape:
        ok = (m, n, k, batch) == (spec.m, spec.n, spec.k, spec.batch)
        return ok, "" if ok else "a static-shape build runs its own shape only"
    if spec.splitk != "none":  # split-K units: their own K rule (splitk.problem_ok)
        from .splitk import problem_ok as _sk_problem_ok

        return _sk_problem_ok(spec, m, n, k, batch)
    U, H, L, T = loop_plan(spec)
    if spec.edge_tiles:
        if m < spec.tile_m or n < spec.tile_n:
            return False, f"edge_tiles: M >= tile_m, N >= tile_n ({spec.tile_m} / {spec.tile_n})"
        if operand_layout_of(spec, "A") == "mcontig" and m % 16:
            return False, "edge_tiles with M-contiguous A: M a multiple of 16 (dword-aligned tile origins)"
    elif m < 1 or n < 1 or m % spec.tile_m or n % spec.tile_n:
        return False, f"M / N must be positive multiples of tile_m / tile_n ({spec.tile_m} / {spec.tile_n})"
    if spec.rt_tail == "switch" and (k % 128 or k // 128 < T):
        return False, f"K must be a multiple of 128, >= {spec.k}"
    if spec.k_tail and spec.rt_tail == "fixed":  # K a multiple of 32: ceil(K / 128) tiles in the loop's classes
        kt = -(-k // 128)
        if k % 32 or kt < T or (kt - T) % U:
            return False, f"k_tail: K a multiple of 32 with ceil(K / 128) = {T} + j {U} (j >= 0)"
    elif spec.rt_tail == "fixed" and (k % 128 or k // 128 < T or (k // 128 - T) % U):
        return False, f"K must be {spec.k} + j {128 * U} (j >= 0)"
    if not 1 <= batch <= 65535:
        return False, "batch must be 1..65535 (grid z)"
    if spec.group_table and (batch != 1 or m % spec.tile_m):
        return False, "group_table: batch 1 (the groups come from the table), M a multiple of tile_m (padded groups)"
    if (m // spec.tile_m) * (n // spec.tile_n) > 2 ** 31 - 1:
        return False, "too many tiles"
    return True, ""


def operand_layout_of(spec: Mxfp6GemmSpec, X: str) -> str:
    """Device layout of operand ``X`` ("A" / "B"): ``a_layout`` overrides ``operand_layout`` for A."""
    return spec.a_layout if (X == "A" and spec.a_layout) else spec.operand_layout


def acc_regs(spec: Mxfp6GemmSpec) -> int:
    return (spec.tile_m // spec.waves_m) * (spec.tile_n // spec.waves_n) // 64


def ring_slots(spec: Mxfp6GemmSpec) -> int:
    """Line-ring slots per row (``canon_ring`` line2 / line3), 0 for the stage ring."""
    return {"line2": 2, "line3": 3, "line2m": 2}.get(spec.canon_ring, 0)


def line_period(spec: Mxfp6GemmSpec) -> int:
    """Tiles after which a line ring's slot / address pattern repeats: line3 4 (3 lines = 4 tiles = 3 slots),
    line2 8 (the slot parity of line 3 (u / 4) + u % 4 flips every 4 tiles)."""
    return 4 if ring_slots(spec) == 3 else 8


def line_operands(spec: Mxfp6GemmSpec) -> Tuple[str, ...]:
    """Operands held in the line ring (``canon_ring`` line2 / line3): the canonical ones (``ring_ops="A"``: A only, a
    canonical B stays in the stage ring)."""
    if not ring_slots(spec):
        return ()
    return tuple(X for X in ("A", "B") if operand_layout_of(spec, X) == "canonical" and (spec.ring_ops == "all"
                                                                                       or X == "A"))


def lds_bytes(spec: Mxfp6GemmSpec) -> int:
    """LDS of one work-group as the code object allocates it (group segment size: the allocations' sum rounded up to
    16 bytes, as both the direct compiler and LLVM emit it)."""
    sr = scale_rings(spec)
    sc = ((0 if "A" in sr else spec.tile_m) + (0 if "B" in sr else spec.tile_n)) * 4 if spec.scale_src == "lds" else 0
    if spec.scale_src == "lds" and spec.scale_layout == "wtile":
        sc = scale_img_bytes(spec, "A") + scale_img_bytes(spec, "B")
    if spec.stage_kib:  # every stage allocation on whole KiB (``_kib``)
        n = 0
        for X, rows in (("A", 0 if spec.a_direct else spec.tile_m), ("B", 0 if spec.b_direct else spec.tile_n)):
            if rows and X not in line_operands(spec):
                n += _kib(spec, rows * (MC_ROW_BYTES if operand_layout_of(spec, X) == "mcontig" else ROW_BYTES))
            if spec.scale_src == "lds" and X not in sr:
                n += _kib(spec, scale_img_bytes(spec, X) if spec.scale_layout == "wtile" else
                          (spec.tile_m if X == "A" else spec.tile_n) * 4)
        n = spec.stages * n + sum(ring_alloc_bytes(spec, spec.tile_m if X == "A" else spec.tile_n)
                                  for X in line_operands(spec))
        return -(-(n + sum(scale_ring_bytes(spec, X) for X in sr)) // 16) * 16
    rows = {"A": 0 if spec.a_direct else spec.tile_m, "B": 0 if spec.b_direct else spec.tile_n}
    lines = line_operands(spec)
    ops = sum(r for X, r in rows.items() if X not in lines)
    op_bytes = ops * ROW_BYTES
    if operand_layout_of(spec, "A") == "mcontig":  # 128 K rows x tile_m / 16 slots of 16 bytes (12 + 4 overlap)
        op_bytes += spec.tile_m * (MC_ROW_BYTES - ROW_BYTES)
    n = spec.stages * (op_bytes + sc) + sum(ring_alloc_bytes(spec, rows[X]) for X in lines)
    n += sum(scale_ring_bytes(spec, X) for X in sr)
    return -(-n // 16) * 16


def scale_rings(spec: Mxfp6GemmSpec) -> Dict[str, Tuple[str, int, int, int]]:
    """Operands whose scales come through a 2-slot LDS scale ring: ``X -> (kind, period P, lead, DTL slot)``; kind
    "raw" (a_scale_src ring: the canonical bytes, 16 per row and period) or "packed" (pring: tile images of P
    consecutive K tiles)."""
    out: Dict[str, Tuple[str, int, int, int]] = {}
    if spec.a_scale_src == "ring":
        out["A"] = ("raw", AS_PERIOD, spec.as_lead, spec.as_slot)
    elif spec.a_scale_src == "pring":
        out["A"] = ("packed", spec.as_period, spec.as_lead, spec.as_slot)
    if spec.b_scale_src == "pring":
        out["B"] = ("packed", spec.bs_period, spec.bs_lead, spec.bs_slot)
    return out


def scale_ring_bytes(spec: Mxfp6GemmSpec, X: str) -> int:
    """LDS bytes of operand ``X``'s scale ring (2 slots x P K tiles x rows x 4 bytes; raw: 16 bytes per row), else 0."""
    sr = scale_rings(spec)
    if X not in sr:
        return 0
    return AS_SLOTS * sr[X][1] * 4 * (spec.tile_m if X == "A" else spec.tile_n)


def as_ring_bytes(spec: Mxfp6GemmSpec) -> int:
    """``a_scale_src="ring"`` / ``"pring"``: A's scale ring bytes, else 0."""
    return scale_ring_bytes(spec, "A")


def scale_layout_of(spec: Mxfp6GemmSpec, X: str) -> str:
    """Device layout of operand ``X``'s scales (``layout.pack_scales_for``): "raw" for A under ``a_scale_src="ring"``
    (the canonical bytes as produced), "period{P}x{tile rows}" under pring, else ``scale_layout``."""
    sr = scale_rings(spec)
    if X in sr:
        kind, P = sr[X][:2]
        return "raw" if kind == "raw" else f"period{P}x{spec.tile_m if X == 'A' else spec.tile_n}"
    if spec.scale_layout == "wtile":
        rows, ws = (spec.tile_m, spec.waves_m) if X == "A" else (spec.tile_n, spec.waves_n)
        if spec.scale_raw:  # the canonical bytes as produced
            return "raw"
        return f"wtile{rows}x{rows // ws}" + (f"p{spec.wtile_period}" if spec.wtile_period > 1 else "")
    return spec.scale_layout


def ring_alloc_bytes(spec: Mxfp6GemmSpec, rows: int) -> int:
    """Bytes of a line-ring allocation of ``rows`` rows: R slots (+ the line2m remnant buffer); ``ring_kib``: rounded
    up to whole KiB."""
    T = atom_geometry(spec.atom).rows
    n = ring_slots(spec) * line_geometry(rows, T)[0] + (mini_geometry(rows, spec.mini_layout)[0]
                                                         if spec.canon_ring == "line2m" else 0)
    return -(-n // 1024) * 1024 if spec.ring_kib else n


def line_geometry(rows: int, T: int = 16) -> Tuple[int, int]:
    """canon_ring line2 / line3: (slot bytes S, row-class stride RB) of a ``rows``-row ring: RB = (rows / 16) 128 + 16,
    S = 16 RB (slot ``s`` of row r, line byte x at s S + (r % 16) RB + (r / 16) 128 + x). In 8-byte bank slots
    (256 bytes = 32 slots; a ds_read_b64 serves 32 lanes per pass) lane (r16, kg) of a fragment read sits at
    2 r16 + 3 kg (+ a common offset) mod 32: kg 0 / 2 on the even slots, kg 1 / 3 on the odd ones -> conflict-free
    (RB = ... + 8 gave r16 + 3 kg: 2-way conflicts, PMC job 68314415); S = 0 mod 256 and the -128 of a wrapping lane
    keep the parity."""
    # 32x32x64 (T = 32): a read pass is the 32 rows of one chunk group, two row-blocks 128 bytes apart -> slot r16 +
    # 16 (r / 16) with RB = ... + 8
    rb = (rows // 16) * LINE_BYTES + (16 if T == 16 else 8)
    return 16 * rb, rb


def mfmas_per_tile(spec: Mxfp6GemmSpec) -> int:
    geo = atom_geometry(spec.atom)
    fm = spec.tile_m // spec.waves_m // geo.rows
    fn = spec.tile_n // spec.waves_n // geo.rows
    return fm * fn * (128 // geo.k)


def grid_of(spec: Mxfp6GemmSpec) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """``((grid_x, 1, batch), (block_x, 1, 1))`` of the spec's (reference) shape."""
    nwg = (spec.m // spec.tile_m) * (spec.n // spec.tile_n)
    if spec.batch != 1:
        return (nwg, 1, spec.batch), (64 * spec.waves_m * spec.waves_n, 1, 1)
    return (nwg, 1, 1), (64 * spec.waves_m * spec.waves_n, 1, 1)


def grid_for(spec: Mxfp6GemmSpec, m: int, n: int, batch: int = 1) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Launch grid of a ``runtime_shape`` build for problem (m, n, batch) (every K: one CTA per output tile)."""
    if spec.schedule == "persistent":
        return persistent_grid(spec, m, n, batch)
    if spec.splitk != "none":  # split-K: data-parallel units + split units (splitk.plan; the grid needs no K)
        from .splitk import grid as _sk_grid

        return _sk_grid(spec, m, n, 0, batch)
    nwg = n_tiles(spec, "A", m) * n_tiles(spec, "B", n)
    return (nwg, 1, batch), (64 * spec.waves_m * spec.waves_n, 1, 1)


def mcontig_rows(m: int) -> int:
    """Codes per K row of an M-contiguous A (``a_layout="mcontig"``): M, or with M % 16 != 0 the M-padded producer
    layout's ceil16(M) (whole 12-byte row blocks; rows past M are don't-care codes that feed only C rows the fused
    path's ``m_mask`` does not store: ``splitk.mask_needed``)."""
    return -(-m // 16) * 16


def n_tiles(spec: Mxfp6GemmSpec, X: str, rows: int) -> int:
    """CTA tiles along operand ``X``'s rows: ceil under ``edge_tiles`` (the last one shifted), else rows / tile."""
    t = spec.tile_m if X == "A" else spec.tile_n
    return -(-rows // t) if (spec.edge_tiles or (X == "A" and spec.m_mask)) else rows // t


def runtime_kargs(spec: Mxfp6GemmSpec, ptrs, m: int, n: int, k: int, batch: int = 1, *, lda: int = 0, ldb: int = 0,
                  ldc: int = 0, strides=None, ngrid: int = 0, gtab: int = 0, ws: int = 0, cnt: int = 0) -> bytes:
    """Kernel arguments of a ``runtime_shape`` build: a, b, sa, sb, c (device pointers), then M, N, K, lda, ldb, ldc
    (i32; lda / ldb = bytes per canonical row (A K-contiguous: K 6 / 8, mcontig: bytes per K row = M 6 / 8), ldc =
    C elements per row; 0 = packed) and the per-entry batch strides of a, b, sa, sb, c in bytes (i64; None = packed:
    ``packed_strides``)."""
    import struct

    if spec.group_table and operand_layout_of(spec, "A") == "mcontig":  # grouped: one [K][tile_m] slab per tile row
        lda = lda or spec.tile_m * 6 // 8
    lda = lda or (mcontig_rows(m) * 6 // 8 if operand_layout_of(spec, "A") == "mcontig" else k * 6 // 8)
    ldb = ldb or k * 6 // 8
    ldc = ldc or n
    st = strides if strides is not None else packed_strides(spec, m, n, k)
    if spec.schedule == "persistent":
        ngrid = ngrid or persistent_grid(spec, m, n, batch)[0][0]
        units = n_tiles(spec, "A", m) * n_tiles(spec, "B", n) * batch
        if ngrid < 1 or (units % NUM_XCDS == 0 and ngrid % NUM_XCDS):
            raise ValueError(f"persistent: ngrid {ngrid} must be >= 1 and a multiple of {NUM_XCDS} when the "
                             f"{units} work units are")
        out = struct.pack("<QQQQQiiiiiiqqqqqii", *ptrs, m, n, k, lda, ldb, ldc, *st, batch, ngrid)
    else:
        out = struct.pack("<QQQQQiiiiiiqqqqq", *ptrs, m, n, k, lda, ldb, ldc, *st)
    if spec.magic_div:
        out += struct.pack("<IIIIII", *[v for d in magic_divisors(spec, m, n) for v in magic_u31(d)])
    if spec.group_table:  # the group table's device pointer: last (every argument before it ends 8-byte aligned)
        if not gtab:
            raise ValueError("group_table: pass the device pointer of the int32 group table (gtab)")
        out += struct.pack("<Q", gtab)
    if spec.splitk != "none":  # split-K: workspace, counters (zeroed once), D, s, K tiles per split (splitk.plan)
        from .splitk import kargs_tail as _sk_kargs

        out += _sk_kargs(spec, m, n, k, batch, ws, cnt)
    return out


def raw_transpose_selectors(b, lane: Value) -> Tuple[Value, Value]:
    """v_perm selectors of ``raw_scale_transpose`` (per lane): stage 1 lanes 0-31 bytes (x0, x1, y0, y1), lanes
    32-63 (x2, x3, y2, y3); stage 2 even 16-lane groups (x0, y0, x2, y2), odd (x1, y1, x3, y3)."""
    lo = b.cmp_lt(lane, b.const_i32(32))
    even = b.cmp_eq(b.land(b.lshr(lane, b.const_i32(4)), b.const_i32(1)), b.const_i32(0))
    return (b.select(lo, b.const_i32(0x05040100), b.const_i32(0x07060302)),
            b.select(even, b.const_i32(0x06020400), b.const_i32(0x07030501)))


def raw_scale_transpose(b, w: Value, sel1: Value, sel2: Value) -> Value:
    """4x4 byte transpose across the 16-lane groups: lane 16 g + r gets byte g of the words of lanes 16 i + r, i =
    0..3, as its byte i (``layout.raw_transpose_model``). Stage 1 swaps the 32-lane halves (v_permlane32_swap) and
    keeps 2 bytes of each word, stage 2 swaps neighbouring 16-lane groups (v_permlane16_swap) the same way."""
    x, y = b.permlane32_swap(w, w)
    z = b.perm_b32(y, x, sel1)
    # tile.permlane16_swap (v_permlane16_swap_b32; the direct compiler's op, no Python IRBuilder method yet)
    x2, y2 = b._op("tile.permlane16_swap", [z, z], [I32, I32], result_name_hint="psw").results
    return b.perm_b32(y2, x2, sel2)


def magic_u31(d: int) -> Tuple[int, int]:
    """(m, s) with n // d == ((2 n * m) >> 32) >> s for every 0 <= n < 2^31: s = ceil(log2 d), m = ceil(2^(31 + s) /
    d) (the error n (m d - 2^(31 + s)) < 2^31 d <= 2^(31 + s) stays below one quotient step)."""
    if not 1 <= d < 2 ** 31:
        raise ValueError("magic_u31: divisor 1 .. 2^31 - 1")
    s = (d - 1).bit_length()
    m = -(-(1 << (31 + s)) // d)
    if m >= 1 << 32:
        raise AssertionError("magic_u31: multiplier past 32 bits")
    return m, s


def magic_divisors(spec: Mxfp6GemmSpec, m: int, n: int) -> Tuple[int, int, int]:
    """magic_div: the divisors (span, rem, units) of the CTA map at problem m x n: grouped: group_m x N tiles, the
    last group's M tiles (gm % group_m, else group_m), tiles per entry; linear: N tiles in ``span``."""
    gm, gn = n_tiles(spec, "A", m), n_tiles(spec, "B", n)
    span = gn if spec.cta_order == "linear" else spec.group_m * gn
    return span, gm % spec.group_m or spec.group_m, gm * gn


def persistent_units(units: int, ngrid: int, drop: bool = False) -> List[List[int]]:
    """Host model of the ``schedule="persistent"`` walk: the work units (entry x tiles + tile) CTA w = 0..ngrid-1
    processes, in order (``drop``: the persist_drop mutant)."""
    out = []
    for w in range(ngrid):
        if units % NUM_XCDS == 0:
            x, j, g8, c = w % NUM_XCDS, w // NUM_XCDS, ngrid // NUM_XCDS, units // NUM_XCDS
            trips = -(-(c - j) // g8) if j < c else 0
            us = [x * c + j + i * g8 for i in range(trips)]
        else:
            trips = -(-(units - w) // ngrid) if w < units else 0
            us = [w + i * ngrid for i in range(trips)]
        out.append(us[:-1] if drop and us else us)
    return out


def persistent_grid(spec: Mxfp6GemmSpec, m: int, n: int, batch: int = 1, wg_per_cu: Optional[int] = None,
                    n_cu: int = 256) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Launch grid of a ``schedule="persistent"`` build: min(work units, CUs x WGs per CU), a multiple of 8 (the
    XCDs) when there are >= 8 units (``wg_per_cu`` default: the LDS occupancy)."""
    units = n_tiles(spec, "A", m) * n_tiles(spec, "B", n) * batch
    occ = wg_per_cu or max(1, (160 * 1024) // max(1, lds_bytes(spec)))
    g = min(units, n_cu * occ)
    g = max(8, g - g % 8) if units >= 8 else units
    return (g, 1, 1), (64 * spec.waves_m * spec.waves_n, 1, 1)


def kernargs(spec: Mxfp6GemmSpec, a: int, b: int, sa: int, sb: int, c: int, m: int, n: int, k: int, batch: int = 1,
             stride_a: int = 0, stride_b: int = 0, stride_sa: int = 0, stride_sb: int = 0, stride_c: int = 0,
             gtab: int = 0, ws: int = 0, cnt: int = 0) -> bytes:
    """Kernel arguments of any build: a static build takes the five pointers (its shape and batch are baked in and
    must equal m / n / k / batch); a ``runtime_shape`` build adds M, N, K, packed leading dimensions and the per-entry
    strides (0 = the packed image sizes of one entry, ``packed_strides``)."""
    import struct

    if not spec.runtime_shape:
        ok, why = runtime_problem_ok(spec, m, n, k, batch)
        if not ok:
            raise ValueError(why)
        return struct.pack("<QQQQQ", a, b, sa, sb, c)
    ok, why = runtime_problem_ok(spec, m, n, k, batch)
    if not ok:
        raise ValueError(why)
    st = packed_strides(spec, m, n, k)
    given = (stride_a, stride_b, stride_sa, stride_sb, stride_c)
    return runtime_kargs(spec, (a, b, sa, sb, c), m, n, k, batch, strides=tuple(g or p for g, p in zip(given, st)),
                         gtab=gtab, ws=ws, cnt=cnt)


def packed_strides(spec: Mxfp6GemmSpec, m: int, n: int, k: int) -> Tuple[int, int, int, int, int]:
    """Bytes between batch entries of a, b, sa, sb, c when every entry is packed (the device images of
    ``layout.pack_operand`` / ``layout.pack_scales_for`` of one entry, C m x n bf16)."""
    nb_rows = n_tiles(spec, "B", n) * spec.tile_n if operand_layout_of(spec, "B") != "canonical" else n
    kp = -(-k // 128) * 128 if spec.k_tail else k  # k_tail: B and the scale images padded to whole 128-K tiles
    ma = mcontig_rows(m) if operand_layout_of(spec, "A") == "mcontig" else m
    return (ma * k * 6 // 8, nb_rows * kp * 6 // 8, scale_image_bytes(spec, "A", m, kp),
            scale_image_bytes(spec, "B", n, kp), m * n * 2)


def scale_image_bytes(spec: Mxfp6GemmSpec, X: str, rows: int, k: int) -> int:
    """Bytes of one entry's packed scale image of operand ``X`` (``rows`` x ``k``)."""
    lay = scale_layout_of(spec, X)
    if lay.startswith("wtile"):
        return (k // 128) * n_tiles(spec, X, rows) * scale_img_bytes(spec, X)
    return rows * k // 32


def tile_of_wg(spec: Mxfp6GemmSpec, wg: int, m: Optional[int] = None, n: Optional[int] = None) -> Tuple[int, int]:
    """Host model of the CTA -> (tile row, tile column) map (``cta_order`` / ``xcd_block``; ``m`` / ``n``: a runtime
    shape, else the spec's)."""
    gm, gn = n_tiles(spec, "A", spec.m if m is None else m), n_tiles(spec, "B", spec.n if n is None else n)
    if spec.cta_order in ("grouped", "mfast"):
        nwg = gm * gn
        t = (wg % NUM_XCDS) * (nwg // NUM_XCDS) + wg // NUM_XCDS if nwg % NUM_XCDS == 0 else wg
        if spec.cta_order == "mfast":
            return t % gm, t // gm
        G = spec.group_m
        span = G * gn
        first = (t // span) * G
        gsz = min(gm - first, G)
        return first + (t % span) % gsz, (t % span) // gsz
    if spec.xcd_block == "none" or spec.cta_order == "linear":
        return wg // gn, wg % gn
    bm, bn = (int(x) for x in spec.xcd_block.split("x"))
    nbn = gn // bn
    x, loc = wg % NUM_XCDS, wg // NUM_XCDS
    return (x // nbn) * bm + loc // bn, (x % nbn) * bn + loc % bn


def kernel_name(spec: Mxfp6GemmSpec) -> str:
    """Full format contract + geometry + a digest of the non-default knobs."""
    d = _defaults()
    diff = {k: v for k, v in asdict(spec).items()
            if k not in ("m", "n", "k", "elem_fmt", "required_compiler", "a_layout", "dlevers", "sk_split", "sk_hybrid",
                         "sk_wave")
            and v != d[k]}
    if spec.sk_static:  # the baked split count is part of the kernel
        diff["sk_split"] = spec.sk_split
    f = spec.elem_fmt
    lay = "ccr" if spec.a_layout == "mcontig" else "rcr"  # A M-contiguous: column-major A
    base = (
        f"rocke_gemm_mx_{f}_{f}_e8m0b32_f32acc_bf16_{lay}_"
        f"{spec.tile_m}x{spec.tile_n}x{spec.depth_u}_w{spec.waves_m}x{spec.waves_n}_{spec.atom}_s{spec.stages}"
        f"_{spec.m}x{spec.n}x{spec.k}"
    )
    if spec.operand_layout != "canonical":
        base += f"_{spec.operand_layout}"
    if spec.a_layout and spec.a_layout != spec.operand_layout:
        base += f"_A{spec.a_layout}"
    if spec.probe != "none":
        base += f"_PROBE_{spec.probe}"
    if spec.mutant != "none":
        base += f"_MUT_{spec.mutant}"
    if diff:
        h = hashlib.sha1(json.dumps(diff, sort_keys=True).encode()).hexdigest()[:10]
        base += f"_k{h}"
    return base


def descriptor(spec: Mxfp6GemmSpec) -> Dict[str, object]:
    """JSON sidecar: format contract, launch geometry and every spec field."""
    grid, block = grid_of(spec)
    abt = {
        "canonical": "{f} [rows,K] row-major dense 6-bit stream (layout.pack_operand, 'canonical')",
        "c0c1": "{f} c0c1 pre-tiled [rows/tile_rows][K/128][tile_rows*96] (layout.pack_operand, 'c0c1')",
        "c0c1p": "{f} c0c1p paired-plane pre-tiled [rows/tile_rows][K/128][tile_rows*96] "
                 "(layout.pack_operand, 'c0c1p')",
        "mcontig": "{f} [K,rows] M-contiguous dense 6-bit stream, rows * 6 / 8 bytes per K row "
                   "(layout.pack_operand, 'mcontig')",
    }
    a_s = abt[operand_layout_of(spec, "A")].replace("{f}", spec.elem_fmt)
    ab = abt[operand_layout_of(spec, "B")].replace("{f}", spec.elem_fmt)
    if spec.scale_layout == "wtile":
        scales = ("e8m0 per 32 K, uint8 [rows, K/32] packed per (K tile, CTA tile) into per-wave lane images by "
                  "layout.pack_scales_wtile(scales, tile rows, wave rows) (= layout.pack_scales_for(scales, atom, "
                  "'wtile{tile rows}x{wave rows}')); scale_layout=wtile, read via "
                  f"{'the LDS ring' if spec.scale_src == 'lds' else 'VGPR loads'} (scale_src={spec.scale_src})")
    elif spec.scale_layout == "raw":
        scales = ("e8m0 per 32 K, uint8 [rows, K/32] row-major as produced (no packing; "
                  "= layout.pack_scales_for(scales, atom, 'raw')); scale_layout=raw, read via the LDS ring")
    else:
        packer = {"tile": "layout.pack_scales_tile", "kstep": "layout.pack_scales"}[spec.scale_layout]
        scales = (f"e8m0 per 32 K, uint8 [rows, K/32] packed by {packer}(scales, atom) "
                  f"(= layout.pack_scales_for(scales, atom, '{spec.scale_layout}')); scale_layout={spec.scale_layout}, "
                  f"read via {'the LDS ring' if spec.scale_src == 'lds' else 'VGPR loads'} (scale_src={spec.scale_src})")
    if spec.a_scale_src == "ring":
        scales = ("A: e8m0 per 32 K, uint8 [M, K/32] row-major as produced (no packing; "
                  "= layout.pack_scales_for(scales, atom, 'raw')), read in-kernel through a 2-slot LDS ring "
                  "(a_scale_src=ring); B: " + scales)
    for X, nm in (("A", "a_scale_src"), ("B", "b_scale_src")):
        if getattr(spec, nm) == "pring":
            scales += (f"; {X}: tile scales re-laid per K-tile period (= layout.pack_scales_for(scales, atom, "
                       f"'{scale_layout_of(spec, X)}')), {nm}=pring, read via a 2-slot LDS scale ring")
    return {
        "kernel": kernel_name(spec),
        "arch": ARCH,
        "op": "gemm_mx",
        "elem_fmt": spec.elem_fmt,
        "required_compiler": spec.required_compiler,
        "contract": {
            "a": a_s.replace("rows", "M").replace("tile_M", "tile_m"),
            "b": ab.replace("rows", "N").replace("tile_N", "tile_n"),
            "scales": scales,
            "acc": "f32",
            "c": "bf16 [M,N] row-major, RNE",
            "layout": "RCR",
        },
        "params": ["a", "b", "sa", "sb", "c"] + (["M", "N", "K", "lda", "ldb", "ldc", "batch_stride_a",
                                                  "batch_stride_b", "batch_stride_sa", "batch_stride_sb",
                                                  "batch_stride_c"] if spec.runtime_shape else [])
                  + (["gtab"] if spec.group_table else [])
                  + ((["ws", "cnt", "sk_dp"] + (["sk_sc"] if spec.sk_kargs == "packed" else ["sk_ns", "sk_kc"]))
                     if spec.splitk != "none" else []),
        "grid": list(grid),
        "block": list(block),
        "lds_bytes": lds_bytes(spec),
        "backend": "python",
        "probe": spec.probe,
        "spec": asdict(spec),
    }


# ----------------------------------------------------------------- static plan


@dataclass
class TilePlan:
    """Slot schedule of one K tile (static; the same for every tile)."""

    n_mfma: int
    acq: int
    read_slots: List[int]  # prefetch: one per fragment, in first-use order
    dtl_slots: List[int]  # one per DTL instruction of the refill
    scale_slot: int


def _n_mini_dtl(spec: Mxfp6GemmSpec, X: Optional[str] = None) -> int:
    """Remnant DTLs per wave of a head-line refill (``mini_src="dtl"``: 32 rows x 32 B each; of operand ``X`` only if
    given), else 0 (the remnant is copied in LDS)."""
    if spec.canon_ring != "line2m" or spec.mini_src != "dtl":
        return 0
    nw = spec.waves_m * spec.waves_n
    return sum((spec.tile_m if Y == "A" else spec.tile_n) // (32 * nw) for Y in line_operands(spec)
               if X is None or Y == X)


def mini_offsets(rows: int, layout: str = "v2") -> List[int]:
    """Byte offsets O_i of the line2m remnant buffer's 4 row classes i = r16 / 4 (``mini_geometry``). v2: i G with
    G = rows / 16 * 128 + 8. v3: i rows / 16 * 128 + (0, 16, 128, 144)[i] -> 8-byte bank slots 0, 2, 16, 18 (mod 32):
    with the (r16 % 4) 32 term every remnant lane of a read pass sits on an even slot + h, distinct, and the slot-ring
    lanes of the same pass on the other parity."""
    gs = (rows // 16) * LINE_BYTES
    if layout in ("v3", "v4"):
        return [i * gs + d for i, d in enumerate((0, 16, 128, 144))]
    return [i * (gs + 8) for i in range(4)]


def mini_geometry(rows: int, layout: str = "v2") -> Tuple[int, int]:
    """canon_ring line2m remnant buffer: (bytes, v2 group stride G). Byte x in [96, 128) of row r's head line sits at
    O_(r16 / 4) + (r / 16) 128 + (r16 % 4) 32 + (x - 96), r16 = r % 16 (``mini_offsets``): the slots' 128-byte
    row-block step (one ds_read serves slot and remnant lanes with a per-lane address), 4 row classes per 128-byte
    block."""
    g = (rows // 16) * LINE_BYTES + 8
    return mini_offsets(rows, layout)[3] + (rows // 16) * LINE_BYTES, g


def _n_line_dtl(spec: Mxfp6GemmSpec, ops: Optional[Tuple[str, ...]] = None) -> int:
    """canon_ring line2: line DTLs per wave of a refill (8 rows x 128 bytes each) of the line operands (``ops``: of
    those only), 0 without line operands."""
    nw = spec.waves_m * spec.waves_n
    return sum((spec.tile_m if X == "A" else spec.tile_n) // (8 * nw) for X in line_operands(spec)
               if ops is None or X in ops)


def last_ring_read(spec: Mxfp6GemmSpec, X: Optional[str] = None) -> int:
    """MFMA slot of a tile's last prefetch read of a line-ring fragment (``line_reads_first``: the ring operands'
    fragments are read first, the stage-ring ones after; else every fragment counts). ``X == "A"``
    (``line_release_b``: A's fragments first, then B's): the last read of A."""
    geo = atom_geometry(spec.atom)
    fa = spec.tile_m // spec.waves_m // geo.rows
    fb = spec.tile_n // spec.waves_n // geo.rows
    lines = line_operands(spec)
    n = ((fa if "A" in lines else 0) + (fb if "B" in lines else 0)) if spec.line_reads_first else fa + fb
    if X == "A":
        n = fa
    return spec.acq_slot + 1 + spec.read_delay + (max(1, n) - 1) * spec.read_every


def line_slot_plan(spec: Mxfp6GemmSpec, late: bool) -> List[Tuple[int, str]]:
    """(MFMA slot, operand) of a refill's line DTLs in issue order: ``line_slots`` for every operand, or
    (``line_release_b``) A's after ``line_release`` and B's after ``line_release_b``."""
    if spec.line_release_b < 0:
        ops = [X for X in line_operands(spec) for _ in range(_n_line_dtl(spec, (X,)))]
        return list(zip(line_slots(spec, late), ops))
    return [(s_, X) for X in ("A", "B") for s_ in line_slots(spec, late, X)]


def line_slots(spec: Mxfp6GemmSpec, late: bool, X: Optional[str] = None) -> List[int]:
    """MFMA slots of a tile's line DTLs: after the tile's last fragment read, or (``late``: the refill that overwrites a
    line the next tile reads) after the ``line_release`` barrier; spread like the stage DTLs. (A line DTL issued
    before a ring read makes the direct compiler wait for it there: its LDS tags are per allocation, and the ring is
    one allocation because a fragment read spans both slots.)"""
    n = mfmas_per_tile(spec)
    k = _n_line_dtl(spec, None if X is None else (X,))
    last_read = last_ring_read(spec, X)
    early = spec.acq_slot if spec.line_early else min(n - 2, last_read)
    rel = spec.line_release_b if X == "B" else spec.line_release
    lo = (rel if late else early) + 1 + spec.dtl_delay
    if spec.line_spread:
        return [min(n - 1, lo + j * spec.line_spread) for j in range(k)]
    span = max(1, n - lo)
    return [min(n - 1, lo + (j * span) // max(1, k)) for j in range(k)]


def _n_dtl(spec: Mxfp6GemmSpec) -> int:
    nw = spec.waves_m * spec.waves_n
    lines = line_operands(spec)
    if _library_mode(spec):  # per operand: ceil(chunks / waves) (the uneven splits repeat the last chunk)
        n = 0
        for X, rows in (("A", spec.tile_m), ("B", spec.tile_n)):
            if X in lines or (X == "B" and (spec.b_direct or spec.b_reg)) or (X == "A" and spec.a_direct):
                continue
            if X == "A" and spec.a_fill == "c0c1":  # c0 dwordx4 chunks + c1 dword DMAs (validated: even splits)
                n += rows // 16 // nw + rows // 8 // nw
                continue
            ch = rows // 8 if operand_layout_of(spec, X) == "mcontig" else rows * UNITS // 64
            n += -(-ch // nw)
        if spec.scale_src == "lds" and spec.scale_dtl == "dword":
            n += sum(-(-(scale_img_bytes(spec, X) // 256) // nw) for X in ("A", "B"))
        return n
    rows = (0 if "A" in lines else spec.tile_m) + (0 if (spec.b_direct or spec.b_reg or "B" in lines) else spec.tile_n)
    n = rows * UNITS // (64 * nw)
    if spec.a_fill == "c0c1":  # A: c0 dwordx4 chunks + c1 dword DMAs instead of its rows x 96 B in 1 KiB chunks
        n += spec.tile_m // 16 // nw + spec.tile_m // 8 // nw - spec.tile_m * UNITS // (64 * nw)
    if spec.scale_src == "lds" and spec.scale_dtl == "dword":  # per operand: rows * 4 B, 256 B per dword DTL
        sr = scale_rings(spec)  # scale-ring operands have no per-tile scale DTL
        n += ((0 if "A" in sr or spec.probe == "sc_noAB" else spec.tile_m)
              + (0 if "B" in sr or spec.probe in ("sc_noB", "sc_noAB") else spec.tile_n)) * 4 // 256 // nw
    return n


def n_breg(spec: Mxfp6GemmSpec) -> int:
    """b_reg: B global->VGPR loads (16 bytes per lane) per wave and tile."""
    return spec.tile_n * ROW_BYTES // 1024 // (spec.waves_m * spec.waves_n) if spec.b_reg else 0


def breg_slots(spec: Mxfp6GemmSpec) -> List[int]:
    """b_reg: MFMA slots of a tile's B loads, spread over the tile after the acquire."""
    n = mfmas_per_tile(spec)
    k = n_breg(spec)
    lo = spec.acq_slot + 1 + spec.dtl_delay
    span = max(1, n - lo)
    return [min(n - 1, lo + (j * span) // max(1, k)) for j in range(k)]


def plan(spec: Mxfp6GemmSpec) -> TilePlan:
    geo = atom_geometry(spec.atom)
    n = mfmas_per_tile(spec)
    fm = spec.tile_m // spec.waves_m // geo.rows
    fn = spec.tile_n // spec.waves_n // geo.rows
    ksteps = 128 // geo.k
    n_frag = (fm + fn) * ksteps
    acq = spec.acq_slot
    first = acq + 1 + spec.read_delay
    reads = [min(n - 1, first + j * spec.read_every) for j in range(n_frag)]
    n_dtl = _n_dtl(spec)
    lo = (spec.release_slot if spec.release_slot >= 0 else acq) + 1 + spec.dtl_delay
    span = max(1, n - lo)
    if spec.dtl_spread:
        dtl = [min(n - 1, lo + j * spec.dtl_spread) for j in range(n_dtl)]
    else:
        dtl = [min(n - 1, lo + (j * span) // n_dtl) for j in range(n_dtl)]
    sc = min(n - 1, max(acq + 1, spec.scale_slot))
    return TilePlan(n, acq, reads, dtl, sc)


def mfma_order(spec: Mxfp6GemmSpec) -> List[Tuple[int, int, int]]:
    """``(t, mi, ni)`` of every MFMA of a K tile in issue order."""
    geo = atom_geometry(spec.atom)
    fm = spec.tile_m // spec.waves_m // geo.rows
    fn = spec.tile_n // spec.waves_n // geo.rows
    o_n, i_n = (fm, fn) if spec.m_outer else (fn, fm)
    out = []
    if spec.mma_order == "pairs":  # rows 2r, 2r+1 interleaved: (2r, n), (2r+1, n), serpentine over n
        for r in range(fm // 2):
            cols = range(fn) if r % 2 == 0 else reversed(range(fn))
            for ni in cols:
                out += [(0, 2 * r, ni), (0, 2 * r + 1, ni)] if ni % 2 == 0 else [(0, 2 * r + 1, ni), (0, 2 * r, ni)]
        return out
    if spec.mma_order == "blk2":  # 2x2 blocks in serpentine block order, each block (a0b0, a0b1, a1b1, a1b0)
        for br in range(fm // 2):
            bcols = range(fn // 2) if br % 2 == 0 else reversed(range(fn // 2))
            for bc in bcols:
                m0, n0 = 2 * br, 2 * bc
                out += [(0, m0, n0), (0, m0, n0 + 1), (0, m0 + 1, n0 + 1), (0, m0 + 1, n0)]
        return out
    for t in range(128 // geo.k):
        for oi in range(o_n):
            inner = range(i_n)
            if spec.serpentine and oi % 2 == 1:
                inner = reversed(range(i_n))
            for ii in inner:
                out.append((t, oi, ii) if spec.m_outer else (t, ii, oi))
    return out


# ----------------------------------------------------------------- builder


#: Named configurations. The defaults stay safe for a pure-LLVM build; "best" is the measured best configuration and
#: needs the rocKE direct (Rust) compiler (its LLVM build spills / waits on vmcnt(0) in the K loop: < 0.5x).
PRESETS: Dict[str, Dict[str, object]] = {
    "default": {},
    # prefetch fragments (acquire at slot 2, first refill DTL one slot later, reads from slot 7 every 3 MFMAs),
    # c0c1 operands, a 12-tile loop body, an lgkmcnt(0) at each tile end (search c5) and A as MFMA src0 (v5)
    "best": {"operand_layout": "c0c1", "frag_mode": "prefetch", "acq_slot": 2, "read_every": 3, "read_delay": 4,
             "dtl_delay": 1, "unroll": 4, "tile_lgkm0": True, "dtl_imm": True, "rsrc_once": True, "swap_ab": False,
             "required_compiler": "rust"},
    # search c6 (rounds 11-13): "best" with the acquire at slot 0, built with the sreassoc/m0share/ldsgroup levers
    # (PRESET_LEVERS): +0.25-0.84 % over "best" on its old levers (twin CI excluded on 2 GPUs, also on E3M2); the
    # gain is the levers', acq_slot 0 alone is within noise
    "best_v6": {"operand_layout": "c0c1", "frag_mode": "prefetch", "acq_slot": 0, "read_every": 3, "read_delay": 4,
                "dtl_delay": 1, "unroll": 4, "tile_lgkm0": True, "dtl_imm": True, "rsrc_once": True,
                "swap_ab": False, "required_compiler": "rust"},
    # v4 (packaged): as "best" with B as MFMA src0 (swap_ab); 0.2-0.3 % behind
    "best_v4": {"operand_layout": "c0c1", "frag_mode": "prefetch", "acq_slot": 2, "read_every": 3, "read_delay": 4,
                "dtl_delay": 1, "unroll": 4, "tile_lgkm0": True, "dtl_imm": True, "rsrc_once": True,
                "required_compiler": "rust"},
    # v3 (packaged): paired-plane operands, unroll 2; 0.2-0.8 % behind "best"
    "best_v3": {"operand_layout": "c0c1p", "frag_mode": "prefetch", "acq_slot": 2, "read_every": 3, "read_delay": 4,
                "dtl_delay": 1, "unroll": 2, "dtl_imm": True, "rsrc_once": True, "required_compiler": "rust"},
    # the previous "best" (packaged as v2): acquire at slot 4, read_delay 2, dtl_delay 0
    "best_v2": {"operand_layout": "c0c1p", "frag_mode": "prefetch", "acq_slot": 4, "read_every": 3, "read_delay": 2,
                "unroll": 2, "dtl_imm": True, "rsrc_once": True, "required_compiler": "rust"},
    # the previous best (c0c1 operands, b128 + b64 per fragment): ~0.8 % behind "best"
    "best_c0c1": {"frag_mode": "prefetch", "acq_slot": 4, "read_every": 3, "read_delay": 2, "unroll": 3,
                  "dtl_imm": True, "rsrc_once": True, "required_compiler": "rust"},
    # no operand prepack: canonical dense codes for A and B (tile scales: a light per-call raw -> tile repack, or
    # scale_layout="raw" at a cost). A in the line ring (whole 128-byte lines, 2 slots; its fragments read first,
    # the release and the refills right after them), B in the stage ring with HipKittens' DTL order (interleave), a
    # per-XCD K offset; early refills (line_early + the dmamanualring lever, PRESET_LEVERS): +8-9 % over the
    # stage-ring version (2 GPUs, jobs 68313971/2, 68314200/1, 68314497/9)
    "best_canonical_line2": {"operand_layout": "canonical", "canon_dtl": "interleave", "k_rot": 1,
                             "frag_mode": "prefetch", "canon_ring": "line2", "ring_ops": "A", "stages": 3,
                             "unroll": 8, "read_every": 1, "read_delay": 0, "line_reads_first": True,
                             "line_release": 12, "dtl_delay": 0, "line_spread": 1, "line_early": True,
                             "acq_slot": 2, "tile_lgkm0": True, "swap_ab": False, "required_compiler": "rust"},
    # both canonical operands in line2m rings (whole-line fetches for B too: half the stage ring's L2 requests),
    # KiB-aligned rings (ring_kib: no cross-ring DMA waits), A's reads / release / refills first then B's
    # (line_release_b), reads every 3rd MFMA slot, refills 2 slots apart, the search's can_f1 knobs (acquire slot 1,
    # read delay 2, setprio 2, paired MMA order, one canonical descriptor): +5.8-7.2 % over can_f1, +3-4 % over
    # packAB + v6 (2 GPUs, jobs 68317884 / 85, 68318003 / 07, 68318519); the remnant copies out of the post-acquire
    # read burst (reads at slot 20, writes 30 slots later): +4.8 / +2.9 % (jobs 68319177 / 78)
    "best_canonical_k3": {"operand_layout": "canonical", "canon_dtl": "interleave", "k_rot": 0, "frag_mode": "prefetch",
                          "canon_ring": "line2m", "ring_ops": "all", "ring_kib": True, "stages": 3, "unroll": 8,
                          "acq_slot": 1, "read_every": 3, "read_delay": 2, "line_reads_first": True,
                          "line_release": 27, "line_release_b": 52, "dtl_delay": 0, "line_spread": 2,
                          "line_early": True, "mini_copy_at": 20, "mini_copy_gap": 30, "setprio": 2, "mma_order": "pairs",
                          "canon_rsrc_once": True, "tile_lgkm0": True, "swap_ab": False, "required_compiler": "rust"},
    # search r30f8 (c20, job 68319756, 3 GPUs): best_canonical_k3 + scale_slot 16, store_nt, line_release 26, setprio
    # 0, canon_rsrc_once off; levers - m0cse - dmafree + litreuse + ldsalias: +0.6-0.9 %
    "best_canonical_r30": {"operand_layout": "canonical", "canon_dtl": "interleave", "frag_mode": "prefetch",
                       "canon_ring": "line2m", "ring_kib": True, "unroll": 8, "acq_slot": 1, "read_every": 3,
                       "read_delay": 2, "line_reads_first": True, "line_release_b": 52, "line_spread": 2,
                       "line_early": True, "mini_copy_at": 20, "mini_copy_gap": 30, "mma_order": "pairs",
                       "scale_slot": 16, "store_nt": True, "tile_lgkm0": True, "swap_ab": False,
                       "required_compiler": "rust"},
    # search r31f0 (c21, job 68320083, 3 GPUs): best_canonical_r30 + read_every 2, line_release 22, mini_copy_gap 16,
    # mini_copy_at_b 30: +0.6-0.9 % (here +0.74 %, job 68320428; gate job 68320423)
    "best_canonical_r31": {"operand_layout": "canonical", "canon_dtl": "interleave", "frag_mode": "prefetch",
                       "canon_ring": "line2m", "ring_kib": True, "unroll": 8, "acq_slot": 1, "read_every": 2,
                       "read_delay": 2, "line_reads_first": True, "line_release": 22, "line_release_b": 52,
                       "line_spread": 2, "line_early": True, "mini_copy_at": 20, "mini_copy_at_b": 30,
                       "mini_copy_gap": 16, "mma_order": "pairs", "scale_slot": 16, "store_nt": True,
                       "tile_lgkm0": True, "swap_ab": False, "required_compiler": "rust"},
    # search r32f0 (c22, job 68320350, 3 GPUs): best_canonical_r31 + acq_slot 0, dtl_delay 1, line_release 18,
    # mini_copy_at 16, mini_copy_gap 20: +0.7-0.9 %
    "best_canonical": {"operand_layout": "canonical", "canon_dtl": "interleave", "frag_mode": "prefetch",
                       "canon_ring": "line2m", "ring_kib": True, "unroll": 8, "acq_slot": 0, "read_every": 2,
                       "read_delay": 2, "dtl_delay": 1, "line_reads_first": True, "line_release": 18,
                       "line_release_b": 52, "line_spread": 2, "line_early": True, "mini_copy_at": 16,
                       "mini_copy_at_b": 30, "mini_copy_gap": 20, "mma_order": "pairs", "scale_slot": 16,
                       "store_nt": True, "tile_lgkm0": True, "swap_ab": False, "required_compiler": "rust"},
    # the stage-ring canonical kernel (HK parity, 1.05x HK on the same GPU)
    "best_canonical_stage": {"operand_layout": "canonical", "canon_dtl": "interleave", "k_rot": 1,
                             "frag_mode": "prefetch", "acq_slot": 2, "read_every": 3, "read_delay": 4, "dtl_delay": 1,
                             "unroll": 4, "tile_lgkm0": True, "swap_ab": False, "required_compiler": "rust"},
    # live activations next to pre-packed weights: A canonical in the line2m ring (2 line slots + the 4-tile group
    # head line's tail copied into a remnant buffer in LDS -> every line refill 3 tiles ahead, after the reads and a
    # release barrier; its fragments read first, acquire at slot 0), B c0c1 in a 3-stage ring. Needs the
    # dmamanualring + dmamanualringw levers (PRESET_LEVERS; direct compiler >= out-68317154): +5 % over the line2
    # version, beats packA + best_v6 (2 GPUs, jobs 68317237 / 39)
    "best_mixed_line2m": {"operand_layout": "c0c1", "a_layout": "canonical", "canon_dtl": "interleave",
                   "frag_mode": "prefetch", "canon_ring": "line2m", "stages": 3, "unroll": 8, "acq_slot": 0, "read_every": 2, "read_delay": 0,
                   "line_reads_first": True, "line_release": 17, "dtl_delay": 0, "line_spread": 1,
                   "mini_copy_gap": 32, "tile_lgkm0": True, "dtl_imm": True, "rsrc_once": True, "swap_ab": False,
                   "required_compiler": "rust"},
    # search round 22 / c14 (job 68317922, f2): best_mixed_line2m + early refills, release 34, copy gap 4, b96x2 B
    # reads, no swizzle, scale slot 32, non-temporal stores, DTL delay 1: +3.3-3.7 % (2 GPUs); GPU gate incl. its own
    # mutants and line-ring hazard mutants (jobs 68318185, 68318263, 68318514)
    "best_mixed_f2": {"operand_layout": "c0c1", "a_layout": "canonical", "canon_dtl": "interleave",
                      "frag_mode": "prefetch", "canon_ring": "line2m", "stages": 3, "unroll": 8, "acq_slot": 0,
                      "read_every": 2, "read_delay": 0, "line_reads_first": True, "line_early": True,
                      "line_release": 34, "dtl_delay": 1, "mini_copy_gap": 4,
                      "canon_read": "b96x2", "swizzle": "none", "scale_slot": 32, "store_nt": True,
                      "tile_lgkm0": True, "dtl_imm": True, "rsrc_once": True, "swap_ab": False,
                      "required_compiler": "rust"},
    # search round 25 / c17 (r25f5): best_mixed_f2 with line_early off, peel_first, m_outer False, dtl_spread 4;
    # levers - licm + litshare, soffshare: +0.7-1.1 % over f2 (search, 3 GPUs; here +0.1 / +0.6 %, jobs
    # 68318917 / 18); the same object under out-68317511 and out-68318380; GPU gate incl. its own mutants and the
    # line-ring hazard mutants (job 68318916)
    "best_mixed": {"operand_layout": "c0c1", "a_layout": "canonical", "frag_mode": "prefetch",
                         "canon_ring": "line2m", "stages": 3, "unroll": 8, "acq_slot": 0, "line_reads_first": True,
                         "line_release": 34, "dtl_delay": 1, "dtl_spread": 4, "mini_copy_gap": 4,
                         "canon_read": "b96x2", "swizzle": "none", "scale_slot": 32, "store_nt": True,
                         "peel_first": True, "m_outer": False, "tile_lgkm0": True, "dtl_imm": True, "rsrc_once": True,
                         "swap_ab": False, "required_compiler": "rust"},
    # the previous best_mixed (line2 ring, refills up to ~1 tile ahead)
    "best_mixed_line2": {"operand_layout": "c0c1", "a_layout": "canonical", "canon_dtl": "interleave",
                         "frag_mode": "prefetch", "canon_ring": "line2", "stages": 3, "unroll": 8, "acq_slot": 0,
                         "read_every": 2, "read_delay": 0, "line_reads_first": True, "line_release": 17,
                         "dtl_delay": 0, "line_spread": 1, "line_early": True, "tile_lgkm0": True, "dtl_imm": True,
                         "rsrc_once": True, "swap_ab": False, "required_compiler": "rust"},
    # the stage-ring mixed kernel
    "best_mixed_stage": {"operand_layout": "c0c1", "a_layout": "canonical", "canon_dtl": "interleave",
                         "frag_mode": "prefetch", "acq_slot": 2, "read_every": 3, "read_delay": 4, "dtl_delay": 1,
                         "unroll": 4, "tile_lgkm0": True, "dtl_imm": True, "rsrc_once": True, "swap_ab": False,
                         "required_compiler": "rust"},
}


#: Direct-compiler levers a preset is measured with (``ROCKE_DIRECT_LEVERS``; others: bench.RUST_LEVERS). The line-ring
#: presets refill early (``line_early``): without ``dmamanualring`` (rocKE direct compiler >= out-68314386: LDS
#: allocations named ``ring*`` are synchronised by the kernel's own vmcnt model, not by the compiler) they stay
#: correct but every ring read after an early refill waits for it.
PRESET_LEVERS: Dict[str, str] = {
    "best_canonical_line2": "scalesep,dsrebase,dshoist,m0fill,m0cse,dmamanualring",
    "best_canonical_k3": "scalesep,dsrebase,dshoist,m0fill,m0cse,dmamanualring,dmamanualringw,sreassoc,m0share,"
                         "bufoffany,dmafree",
    "best_canonical_r30": "scalesep,dsrebase,dshoist,m0fill,dmamanualring,dmamanualringw,sreassoc,m0share,"
                          "bufoffany,litreuse,ldsalias",
    "best_canonical_r31": "scalesep,dsrebase,dshoist,m0fill,dmamanualring,dmamanualringw,sreassoc,m0share,"
                          "bufoffany,litreuse,ldsalias",
    "best_canonical": "scalesep,dsrebase,dshoist,m0fill,dmamanualring,dmamanualringw,sreassoc,m0share,bufoffany,"
                      "litreuse,ldsalias",
    "best_mixed_line2m": "scalesep,dsrebase,dshoist,m0fill,m0cse,dmamanualring,dmamanualringw",
    "best_mixed": "scalesep,dsrebase,dshoist,m0fill,m0cse,dmamanualringw,m0share,ldsgroup,litshare,soffshare",
    "best_mixed_line2": "scalesep,dsrebase,dshoist,m0fill,m0cse,dmamanualring",
    "best_mixed_f2": "scalesep,dsrebase,dshoist,m0fill,m0cse,dmamanualringw,m0share,ldsgroup,licm",
    "best_v6": "scalesep,dsrebase,dshoist,m0fill,m0cse,sreassoc,m0share,ldsgroup",
}


def preset(name: str, **over) -> Mxfp6GemmSpec:
    """``Mxfp6GemmSpec`` of preset ``name`` (+ overrides, e.g. the shape or ``elem_fmt``)."""
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r} (expected one of {tuple(PRESETS)})")
    return replace(Mxfp6GemmSpec(), **{**PRESETS[name], **over})


def with_dlevers(spec: Mxfp6GemmSpec, levers: str) -> Mxfp6GemmSpec:
    """``spec`` carrying direct-compiler ``levers`` in its IR (kernel attribute ``dlevers``): the same code object
    as building ``spec`` with ``ROCKE_DIRECT_LEVERS=levers``, whatever the environment then says about those levers.
    ``with_dlevers(preset(name), PRESET_LEVERS[name])`` makes a preset self-describing."""
    return replace(spec, dlevers=",".join(x.strip() for x in levers.split(",") if x.strip()))


def check_compiler(spec: Mxfp6GemmSpec, compiler: str) -> None:
    """Refuse building ``spec`` with ``compiler`` ("llvm" | "rust") when the spec requires the other one."""
    if spec.required_compiler == "rust" and compiler != "rust":
        raise ValueError(
            f"this Mxfp6GemmSpec requires the rocKE direct (Rust) compiler (required_compiler='rust'); its LLVM "
            f"build is catastrophically slow (spills / vmcnt(0) in the K loop). Build it with the direct compiler, "
            f"or use the LLVM-safe defaults (preset 'default'). Got compiler={compiler!r}."
        )


def build_mxfp6_gemm(spec: Mxfp6GemmSpec, *, arch: str = ARCH) -> KernelDef:
    """Build the kernel for ``spec`` (gfx950 only)."""
    if arch != ARCH:
        raise ValueError(f"mxfp6 GEMM is gfx950-only, got {arch!r}")
    ok, why = validate_spec(spec)
    if not ok:
        raise ValueError(f"invalid Mxfp6GemmSpec: {why}")
    with no_gc():
        return _Builder(spec).build()


def epi_lds_fits(spec: Mxfp6GemmSpec) -> bool:
    """epi_lds: the per-wave C staging blocks (16 rows x (2 fn T + 16) bytes per wave) fit the largest operand LDS
    allocation the builder makes (a line ring, or one stage of an operand held in LDS)."""
    nw = spec.waves_m * spec.waves_n
    fn = spec.tile_n // spec.waves_n // 16
    need = nw * 16 * (fn * 32 + 16)
    lines = line_operands(spec)
    sizes = [ring_alloc_bytes(spec, spec.tile_m if X == "A" else spec.tile_n) for X in lines]
    for X, rows in (("A", spec.tile_m), ("B", spec.tile_n)):
        if X in lines or (X == "A" and spec.a_direct) or (X == "B" and spec.b_direct):
            continue
        rb = MC_ROW_BYTES if (X == "A" and spec.a_layout == "mcontig") else ROW_BYTES
        sizes.append(_kib(spec, rows * rb))
    return need <= max(sizes + [0])


@contextlib.contextmanager
def no_gc():
    """The cyclic garbage collector paused (restored on exit): building a kernel allocates ~10^4 ops with Value <-> Op
    cycles, and the collections the allocations trigger rescan the growing IR -- more than half of the build time
    (2.3x faster builds; the IR is the same object graph). Wrap ``serialize`` in it too (``build_ir_text``)."""
    was = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was:
            gc.enable()


def build_ir_text(spec: Mxfp6GemmSpec, *, arch: str = ARCH) -> str:
    """``serialize(build_mxfp6_gemm(spec))`` with the garbage collector paused for both steps (build services)."""
    from rocke.core.ir_serialize import serialize

    with no_gc():
        return serialize(build_mxfp6_gemm(spec, arch=arch))


class _McFrag:
    """An mcontig A fragment whose second half (``ds_read_b96_tr_b6`` of K 16..31) is copied at its first MFMA
    (``mc_copy`` late / fence): the first half's three dwords, the second half's 3-dword read, the packed operand."""

    def __init__(self, parts, half):
        self.parts, self.half, self.packed = parts, half, None


class _Builder:
    def __init__(self, spec: Mxfp6GemmSpec):
        self.spec = spec
        self.geo = atom_geometry(spec.atom)
        self.b = IRBuilder(kernel_name(spec))
        g = self.geo
        self.T = g.rows
        self.wt_m = spec.tile_m // spec.waves_m
        self.wt_n = spec.tile_n // spec.waves_n
        self.fm = self.wt_m // g.rows
        self.fn = self.wt_n // g.rows
        self.ksteps = 128 // g.k  # MFMA K steps per 128-K tile
        self.n_tiles = spec.k // 128
        self.row_bytes = spec.k * 6 // 8
        self.nw = spec.waves_m * spec.waves_n
        self.S = spec.stages
        self.adv = _adv(spec)
        self.vm_trace: List[Tuple[object, str]] = []
        self.acq_off = 1 if spec.carry else self.adv  # tiles between an acquire's tile and the tile it acquires
        # scale period: tiles per scale load (kstep: four K steps; tile: one tile)
        self.tile_scales = spec.scale_layout in ("tile", "raw", "wtile")  # op_sel = MFMA tile (raw: in-kernel)
        self.raw_scales = spec.scale_layout == "raw"
        self.clamp = spec.tail_mode == "clamp"
        self.sc_lds = spec.scale_src == "lds"
        # a_scale_src ring: raw A scales, 4-tile periods in a 2-slot LDS ring; DTLs per wave and period
        self.as_ring = spec.a_scale_src == "ring"
        # scale rings (raw A ring, packed period rings): X -> (kind, P, lead, slot); DTLs per wave and period; periods
        self.srings = scale_rings(spec)
        nw_ = spec.waves_m * spec.waves_n
        self.sr_n = {X: (spec.tile_m * 16 // 1024 // nw_ if v[0] == "raw" else 1) for X, v in self.srings.items()}
        self.sr_np = {X: -(-(spec.k // 128) // v[1]) for X, v in self.srings.items()}
        # the raw ring's no-DTL timing probes issue none
        self.sr_on = {X: not (X == "A" and spec.probe in ("as_nodtl", "as_none")) for X in self.srings}
        self.n_as = self.sr_n.get("A", 0)
        self.n_per = self.sr_np.get("A", spec.k // 128 // AS_PERIOD)
        self.P = 1 if self.tile_scales else 4 // self.ksteps
        if spec.scale_layout == "wtile" and spec.wtile_period > 1:  # one scale load per wtile_period K tiles
            self.P = spec.wtile_period
        if spec.scale_layout == "wtile" and spec.scale_raw:  # one dwordx4 = 4 K tiles of a raw scale row
            self.P = 4
        # persist_overlap: the epilogue's store instructions per wave (swap_ab False: 2 dwords per MFMA tile)
        self.n_epi_stores = 2 * (spec.tile_m // spec.waves_m // 16) * (spec.tile_n // spec.waves_n // 16)
        self.plan = plan(spec)
        self.order = mfma_order(spec)
        self.sc_groups = {"A": 1, "B": 1} if self.tile_scales else {"A": self.fm // 4, "B": self.fn // 4}
        if spec.scale_src == "lds":
            self.sc_groups = {"A": 0, "B": 0}  # no VGPR scale loads: the scales ride the stage DTLs
        self.rows = {"A": spec.tile_m, "B": spec.tile_n}
        self.line_ops = line_operands(spec)
        self.R = ring_slots(spec)  # line-ring slots per row (0: stage ring)
        # tiles between a line refill's tile and its target (probe line_lead3: 3 with 2 slots -- overwrites live
        # lines, a timing probe of the lead a third slot would give)
        self.mini = spec.canon_ring == "line2m"
        self.LA = 3 if (spec.probe == "line_lead3" or self.mini) else self.R
        if spec.probe == "l2m_lead4":  # must-fail hazard mutant: every refill one tile too far ahead (it overwrites
            self.LA = 4  # the slot of a line the next tile still reads, after that line landed: deterministic)
        # target offset of the loop's line refills (line_merge: two ahead, issued after the tile's acquire)
        self.LI = self.LA - 1 if spec.line_merge else self.LA
        if spec.mutant == "lmerge_lead":
            self.LI = self.LA + 1
        self.LP = line_period(spec)  # tiles per line-ring address pattern
        # first use (MFMA index in the tile) of every fragment key (X, f, t)
        self.first_use: Dict[Tuple[str, int, int], int] = {}
        for x, (t, mi, ni) in enumerate(self.order):
            self.first_use.setdefault(("A", mi, t), x)
            self.first_use.setdefault(("B", ni, t), x)
        # tile-library fields (all inert at their defaults)
        self.KS = spec.depth_u // 128  # 128-K tiles per barrier period
        self.rt = spec.runtime_shape
        self.lib = _library_mode(spec)
        self.tshift: Optional[Value] = None  # runtime_shape: K-tile shift of the static tail (set while emitting it)
        self.tshift_u: Optional[Value] = None  # ... in scale periods
        self.wtile = spec.scale_layout == "wtile"
        self.dop = "B" if spec.b_direct else ("A" if spec.a_direct else None)  # the operand loaded straight to VGPRs
        self.dn = self.fn if self.dop == "B" else self.fm  # its fragments per wave and K tile
        self.mc = operand_layout_of(spec, "A") == "mcontig"

    # ------------------------------------------------------------- tile-library helpers
    def _ti(self, v: int, period: bool = False) -> Value:
        """The K tile (``period``: scale period) ``v`` of the static schedule: the constant, or (runtime_shape, while
        the static tail is emitted) shifted by the runtime K's extra tiles."""
        sh = self.tshift_u if period else self.tshift
        return self.c32(v) if sh is None else self.b.add(self.c32(v), sh)

    def _rb(self, X: str) -> Value:
        """Bytes per row of canonical operand ``X`` (the K-contiguous stream; runtime_shape: lda / ldb)."""
        if not self.rt:
            return self.c32(self.row_bytes)
        return self.p_ld[X]

    def _nqv(self, X: str) -> Value:
        """128-row (tile scales) / 4T-row (kstep) groups of operand ``X``'s rows (runtime_shape: from M / N)."""
        if not self.rt:
            return self.c32(self.nq[X])
        return self.b.div(self.p_dim[X], self.c32(self.grp_rows))

    def _ntile_rows(self, X: str) -> Value:
        """CTA tiles along operand ``X``'s rows (M / tile_m, N / tile_n)."""
        if not self.rt:
            return self.c32((self.spec.m if X == "A" else self.spec.n) // self.rows[X])
        if self.spec.edge_tiles or (X == "A" and self.spec.m_mask):  # ceil (edge: the last tile shifted; m_mask:
            #   rows past M masked)
            return self.b.div(self.b.add(self.p_dim[X], self.c32(self.rows[X] - 1)), self.c32(self.rows[X]))
        return self.b.div(self.p_dim[X], self.c32(self.rows[X]))

    def _cnb(self) -> Value:
        """Bytes per C row (runtime_shape: ldc * 2)."""
        if not self.rt:
            return self.c32(self.spec.n * 2)
        return self.b.mul(self.p_ldc, self.c32(2))

    def _cta_map(self, wg: Value) -> Tuple[Value, Value]:
        """``cta_order`` grouped / linear: (tile row, tile column) of CTA ``wg`` (``tile_of_wg`` is the host model).
        The work decomposition is this one function: a persistent / stream-K schedule replaces it (and the K range,
        see notes/tile-library.md "stream-K hook points")."""
        b, spec = self.b, self.spec
        gm, gn = self._ntile_rows("A"), self._ntile_rows("B")
        if spec.cta_order == "linear":
            if getattr(self, "persist_t", None) is not None:
                wg = self.persist_t
            if spec.magic_div:  # "span" holds N tiles here
                q = self._mdiv(wg, "span")
                return q, b.sub(wg, b.mul(q, gn))
            return b.div(wg, gn), b.mod(wg, gn)
        nx = self.c32(NUM_XCDS)
        nwg = b.mul(gm, gn)
        if getattr(self, "persist_t", None) is not None:  # persistent: the work unit's tile (XCD order is global)
            t = self.persist_t
        else:
            chunked = b.add(b.mul(b.mod(wg, nx), b.div(nwg, nx)), b.div(wg, nx))  # XCD x: tiles x nwg / 8 .. on
            t = b.select(b.cmp_eq(b.mod(nwg, nx), self.c32(0)), chunked, wg)
        if spec.cta_order == "mfast":  # weight-panel major: t -> (t % gm, t / gm), XCD-chunked like grouped
            return b.mod(t, gm), b.div(t, gm)
        G = self.c32(spec.group_m)
        span = b.mul(G, gn)
        if spec.magic_div:  # gsz is group_m (a constant) or the last group's gm % group_m (host magic)
            qs = self._mdiv(t, "span")
            first = b.mul(qs, G)
            r = b.sub(t, b.mul(qs, span))
            last = b.cmp_lt(b.sub(gm, first), G)
            qr = b.select(last, self._mdiv(r, "rem"), b.div(r, G))
            gsz = b.select(last, b.sub(gm, first), G)
            return b.add(first, b.sub(r, b.mul(qr, gsz))), qr
        first = b.mul(b.div(t, span), G)
        gsz = b.smin(b.sub(gm, first), G)
        r = b.mod(t, span)
        return b.add(first, b.mod(r, gsz)), b.div(r, gsz)

    def _mdiv(self, n: Value, nm: str) -> Value:
        """magic_div: n / d (0 <= n < 2^31) = mulhi(2 n, m) >> s for the host's (m, s) = ``magic_u31(d)``."""
        b = self.b
        m_, s_ = self.p_magic[nm]
        hi = b.inline_asm("s_mul_hi_u32 $0, $1, $2", "=s,s,s", [b.to_sgpr_u32(b.add(n, n)), m_], I32,
                          sideeffect=False)
        return b.to_sgpr_u32(b.lshr(hi, s_))

    # ------------------------------------------------------------- helpers
    def c32(self, v: int) -> Value:
        return self.b.const_i32(int(v))

    def _vm_fence(self, kind: str = "dtl", n: int = 1) -> None:
        """Keep vector-memory issue in program order (everything else may cross).

        The acquire vmcnt counts come from a static model of the VM issue order;
        without this fence the machine scheduler reorders independent DTLs and
        scale / fragment loads that share an MFMA slot (or a tile, with
        ``pin_mfma=False``) and a vmcnt(n) can then release before the awaited
        fill landed (a race found by the search: DTL spread with clamped slots).
        """
        self.b.sched_barrier(VM_FENCE_MASK)
        # emission trace (tile tag, kind) checked against the static model by the tests
        self.vm_trace.extend([(getattr(self, "_trace_tile", None), kind)] * n)

    def build(self) -> KernelDef:
        b, spec = self.b, self.spec
        k = b.kernel
        k.attrs["max_workgroup_size"] = 64 * self.nw
        if spec.dlevers:  # the kernel carries its direct-compiler levers (default off: nothing emitted)
            k.attrs["dlevers"] = spec.dlevers
        if spec.lds_alias_scopes:
            k.attrs["lds_alias_scopes"] = True
        if spec.acc_agpr:
            na = min(acc_regs(spec), 256)  # acc_vgpr_spill: the atoms past the 256th register live in VGPRs
            k.attrs["agpr_alloc"] = (na, na)
            k.attrs["mfma_agpr_form"] = True
        if self.nw > 4 or spec.occupancy > 1:
            # two (or more) waves per SIMD: budget the registers (occupancy: that many work-groups per CU)
            k.attrs["waves_per_eu"] = -(-(self.nw * spec.occupancy) // 4)
        self.pa = b.param("a", PtrType(I8, "global"))
        self.pb = b.param("b", PtrType(I8, "global"))
        self.psa = b.param("sa", PtrType(I32, "global"))
        self.psb = b.param("sb", PtrType(I32, "global"))
        self.pc = b.param("c", PtrType(BF16, "global"))
        if self.rt:  # runtime problem: M, N, K, leading dimensions, per-entry batch strides (bytes)
            self.p_dim = {"A": b.param("M", I32), "B": b.param("N", I32)}
            self.p_k = b.param("K", I32)
            self.p_ld = {"A": b.param("lda", I32), "B": b.param("ldb", I32)}
            self.p_ldc = b.param("ldc", I32)
            self.p_bs = [b.param(f"bstride_{x}", I64) for x in ("a", "b", "sa", "sb", "c")]
            self.k_tiles = b.to_sgpr_u32(b.lshr(b.add(self.p_k, self.c32(127)) if spec.k_tail else self.p_k,
                                                self.c32(7)))  # k_tail: ceil(K / 128)
        if spec.schedule == "persistent":
            self.p_nbatch = b.param("nbatch", I32)
            self.p_ngrid = b.param("ngrid", I32)
        if spec.magic_div:  # (m, s) pairs of the CTA map's runtime divisors (magic_u31)
            self.p_magic = {nm: (b.param(f"mg_{nm}_m", I32), b.param(f"mg_{nm}_s", I32))
                            for nm in ("span", "rem", "units")}
        if spec.group_table:  # grouped GEMM: int32 group of each tile row (gtab[r]), the last argument
            self.p_gtab = b.param("gtab", PtrType(I32, "global"))
        if spec.kernarg_preload:
            for p_ in k.params:
                p_.attrs["inreg"] = True
        if spec.splitk != "none":  # split-K / stream-K units with the fused deterministic fix-up (splitk.py)
            from .splitk import emit as _sk_emit

            _sk_emit(self)
            b.ret()
            return k
        if spec.schedule == "persistent":
            self._persistent()
            b.ret()
            return k
        if self.rt or spec.batch != 1:  # entry z of a strided batch: every pointer moves by z x its stride
            z = b.zext(b.block_id_z(), I64)
            st = self.p_bs if self.rt else [b.const_i64(v) for v in packed_strides(spec, spec.m, spec.n, spec.k)]
            za = z
            if spec.mutant == "batch_swap":  # must-fail: entry z multiplies entry z ^ 1's A
                za = b.zext(b.xor(b.block_id_z(), self.c32(1)), I64)
            zs = (za, z, z, z, z)
            self.pa, self.pb, self.psa, self.psb, self.pc = (
                b.global_ptr_add(p, b.mul(zz, s))
                for p, s, zz in zip((self.pa, self.pb, self.psa, self.psb, self.pc), st, zs))
        self._setup()
        if spec.probe == "stamps":  # realtime clock (100 MHz) at kernel start, written with the end time
            self.t_start = b.inline_asm("s_memrealtime $0\n\ts_waitcnt lgkmcnt(0)", "=s", [], I64)
            self.c_start = b.inline_asm("s_memtime $0\n\ts_waitcnt lgkmcnt(0)", "=s", [], I64)
        if spec.probe == "mfma_only":
            self._mfma_only()
        else:
            self._mainloop()
        if spec.probe == "stamps":  # realtime at the end of the K loop (all MFMAs issued), before the epilogue
            self.t_epi = b.inline_asm("s_memrealtime $0\n\ts_waitcnt lgkmcnt(0)", "=s", [], I64)
        if not getattr(self, "epilogue_done", False):  # (runtime_shape switch tail: one epilogue per K residue)
            self._epilogue()
        if spec.probe == "stamps":
            self._stamps()
        b.ret()
        return k

    def _persistent(self) -> None:
        """schedule persistent: CTA w (XCD x = w % 8, j = w / 8 of G8 = ngrid / 8 per XCD) walks the work units
        x C + j + i G8 (C = N / 8, N = nbatch x tiles per entry; N not a multiple of 8: w + i ngrid). Each unit: a
        barrier (every wave left the previous unit's LDS stages), the entry's pointers, the tile setup, the K loop and
        the epilogue. The work-decomposition hook of stream-K (notes/tile-library.md)."""
        b = self.b
        T = b.to_sgpr_u32(b.mul(self._ntile_rows("A"), self._ntile_rows("B")))
        Nu = b.to_sgpr_u32(b.mul(self.p_nbatch, T))
        wg = b.block_id_x()
        nx = self.c32(NUM_XCDS)
        x, j = b.mod(wg, nx), b.div(wg, nx)
        g8 = b.to_sgpr_u32(b.div(self.p_ngrid, nx))
        C = b.to_sgpr_u32(b.div(Nu, nx))
        even = b.cmp_eq(b.mod(Nu, nx), self.c32(0))
        one = self.c32(1)
        t_even = b.select(b.cmp_lt(j, C), b.div(b.add(b.sub(C, j), b.sub(g8, one)), g8), self.c32(0))
        t_odd = b.select(b.cmp_lt(wg, Nu), b.div(b.add(b.sub(Nu, wg), b.sub(self.p_ngrid, one)), self.p_ngrid),
                         self.c32(0))
        trips = b.select(even, t_even, t_odd)
        if self.spec.mutant == "persist_drop":  # must-fail: every CTA skips its last work unit
            trips = b.select(b.cmp_lt(self.c32(0), trips), b.sub(trips, one), trips)
        trips = b.to_sgpr_u32(trips)
        base_e = b.add(b.mul(x, C), j)
        orig = (self.pa, self.pb, self.psa, self.psb, self.pc)
        if self.spec.persist_overlap:
            self._persistent_overlap(trips, even, base_e, g8, wg, T, orig)
            return
        with b.scf_for(self.c32(0), trips, one, iv_name="unit") as it:
            g = b.to_sgpr_u32(b.select(even, b.add(base_e, b.mul(it, g8)), b.add(wg, b.mul(it, self.p_ngrid))))
            entry, self.persist_t = self._unit_split(g, T)
            ze = b.zext(entry, I64)
            self.pa, self.pb, self.psa, self.psb, self.pc = (b.global_ptr_add(p, b.mul(ze, st))
                                                           for p, st in zip(orig, self.p_bs))
            b.s_waitcnt(lgkmcnt=0)  # every wave finished reading the previous unit's stages
            b.s_barrier_bare()
            self.epilogue_done = False
            self._setup()
            self._mainloop()
            if not self.epilogue_done:
                self._epilogue()
        self.persist_t = None

    def _unit_split(self, g: Value, T: Value) -> Tuple[Value, Value]:
        """Work unit g -> (batch entry, tile of the entry) (magic_div: the host's magic for T)."""
        b = self.b
        if self.spec.magic_div:
            e = self._mdiv(g, "units")
            return e, b.to_sgpr_u32(b.sub(g, b.mul(e, T)))
        return b.div(g, T), b.to_sgpr_u32(b.mod(g, T))

    def _persistent_overlap(self, trips, even, base_e, g8, wg, T, orig) -> None:
        """persist_overlap: unit u = barrier, setup, unit u's prologue DMAs, unit u-1's epilogue (u > 0; its
        accumulators / C pointer / tile origin are loop-carried), prologue waits, K loop; the last unit's epilogue
        after the loop. The epilogue's stores sit between the prologue DMAs and the first acquire: the static vmcnt
        counts (which do not know them) only get stricter."""
        b = self.b
        n_acc = self.fm * self.fn
        # a whole KiB: the stages after it stay KiB-aligned (the compiler's LDS DMA tags are KiB ranges)
        self.dummy_lds = b.smem_alloc(I32, [256], name_hint="ovdummy")
        zero = b.vec_bitcast(b.vec_trunc_f32_to_bf16(b.zero_vec_f32(self.geo.c_len)), VectorType(I32, 2))
        z32 = self.c32(0)
        # carried: the previous unit's accumulators as packed bf16 (2 VGPRs per atom), C pointer, tile origin
        if self.spec.acc_agpr:  # AGPRs are free during the next unit's prologue: carry the packed values there
            zero = b.inline_asm("", "=a,0", [zero], zero.type, sideeffect=False)
        init = [(f"pacc{i}", zero) for i in range(n_acc)] + [("ppc", orig[4]), ("prA", z32), ("prB", z32)]
        loop = b.scf_for_iter(self.c32(0), trips, self.c32(1), init, iv_name="unit", elide_trailing_barrier=False)
        with loop as (it, vs):
            prev_acc, prev_pc, prev_ra, prev_rb = list(vs[:n_acc]), vs[n_acc], vs[n_acc + 1], vs[n_acc + 2]
            g = b.to_sgpr_u32(b.select(even, b.add(base_e, b.mul(it, g8)), b.add(wg, b.mul(it, self.p_ngrid))))
            entry, self.persist_t = self._unit_split(g, T)
            ze = b.zext(entry, I64)
            self.pa, self.pb, self.psa, self.psb, self.pc = (b.global_ptr_add(p, b.mul(ze, st))
                                                           for p, st in zip(orig, self.p_bs))
            b.s_waitcnt(vmcnt=0, lgkmcnt=0)  # every wave finished the previous unit's loads and stage reads
            b.s_barrier_bare()
            self.epilogue_done = False
            self._setup()

            def between():
                # exactly n_epi_stores VM ops on either branch: the static vmcnt model counts them (_vm_events)
                with b.scf_if_else(b.cmp_lt(self.c32(0), it)) as (then_ctx, else_ctx):
                    with then_ctx:
                        cur = (self.pc, dict(self.row0))
                        self.pc, self.row0 = prev_pc, {"A": prev_ra, "B": prev_rb}
                        self._epilogue(wait=False, packed=prev_acc)
                        self.pc, self.row0 = cur
                    with else_ctx:  # first unit: dummy 4-byte LDS loads of A's first 256 bytes into a scratch word
                        rs = b.buffer_rsrc(self.pa, self.c32(256))
                        voff = b.mul(self.lane, self.c32(4))
                        lds = b.smem_addr_of(self.dummy_lds)
                        for _ in range(self.n_epi_stores):
                            b.async_buffer_load_lds_addr(rs, lds, voff, self.c32(0), 1)
                            self._vm_fence("store")

            self._mainloop(between=between)
            packed = self._pack_acc()
            if self.spec.acc_agpr:
                packed = [b.inline_asm("", "=a,0", [v], v.type, sideeffect=False) for v in packed]
            b.scf_yield(*(packed + [self.pc, self.row0["A"], self.row0["B"]]))
        res = loop.results
        with b.scf_if(b.cmp_lt(self.c32(0), trips)):
            # the loop body's lane / wave values are out of scope here: derive them again
            self.lane = b.lane_id()
            wave = b.to_sgpr_u32(b.lshr(b.thread_id_x(), self.c32(6)))
            self.wrow0 = {"A": b.to_sgpr_u32(b.mul(b.div(wave, self.c32(self.spec.waves_n)), self.c32(self.wt_m))),
                          "B": b.to_sgpr_u32(b.mul(b.mod(wave, self.c32(self.spec.waves_n)), self.c32(self.wt_n)))}
            self.pc = res[n_acc]
            self.row0 = {"A": res[n_acc + 1], "B": res[n_acc + 2]}
            self._epilogue(packed=list(res[:n_acc]))
        self.persist_t = None

    def _aux(self, X: str) -> int:
        """Cache-policy bits of operand ``X``'s refill DTLs (``b_aux`` for B when set, else ``dtl_aux``)."""
        return self.spec.b_aux if (X == "B" and self.spec.b_aux >= 0) else self.spec.dtl_aux

    def _group_b(self, tm: Value) -> None:
        """group_table: B and its scales move to group gtab[tm] (tile row tm of the packed A / C problem): one scalar
        load (s_load_dword + lgkmcnt(0) in one asm block, before any of the tile's DMAs: no VM counter is touched),
        then the batch strides of b / sb. Mutants: ``gt_zero`` (group 0), ``gt_swap`` (group ^ 1)."""
        b, spec = self.b, self.spec
        if spec.mutant == "gt_zero":
            g = self.c32(0)
        else:
            ptr = b.global_ptr_add(self.p_gtab, b.zext(b.to_sgpr_u32(b.mul(tm, self.c32(4))), I64))
            g = b.inline_asm("s_load_dword $0, $1, 0x0\n\ts_waitcnt lgkmcnt(0)", "=s,s", [ptr], I32)
            if spec.mutant == "gt_swap":
                g = b.xor(g, self.c32(1))
        ze = b.zext(b.to_sgpr_u32(g), I64)
        sb_b, sb_sb = self.p_bs[1], self.p_bs[3]
        self.pb = b.global_ptr_add(self.pb, b.mul(ze, sb_b))
        self.psb = b.global_ptr_add(self.psb, b.mul(ze, sb_sb))

    # ------------------------------------------------------------- setup
    def _setup(self) -> None:
        b, spec, g = self.b, self.spec, self.geo
        tid = b.thread_id_x()
        self.lane = b.lane_id()
        wave = b.to_sgpr_u32(b.lshr(tid, self.c32(6)))
        self.wave = wave
        wm = b.div(wave, self.c32(spec.waves_n))
        wn = b.mod(wave, self.c32(spec.waves_n))
        self.wsel = {"A": wm, "B": wn}  # the wave's row / column index in the wave grid
        # CTA tile (XCD-aware order)
        wg = b.block_id_x()
        gm, gn = spec.m // spec.tile_m, spec.n // spec.tile_n
        if spec.cta_order != "xcd_block":
            tm, tn = self._cta_map(wg)
        elif spec.xcd_block == "none":
            tm, tn = b.div(wg, self.c32(gn)), b.mod(wg, self.c32(gn))
        else:
            bm, bn = (int(x) for x in spec.xcd_block.split("x"))
            nbn = gn // bn
            x, loc = b.mod(wg, self.c32(NUM_XCDS)), b.div(wg, self.c32(NUM_XCDS))
            if spec.k_rot == 1:  # per XCD
                self.krot_off = b.to_sgpr_u32(b.mul(x, self.c32(self.n_tiles // NUM_XCDS)))
            elif spec.k_rot == 2:  # per CTA within its XCD block (same offsets in every XCD)
                nloc = (gm * gn) // NUM_XCDS
                self.krot_off = b.to_sgpr_u32(b.mul(loc, self.c32(self.n_tiles // nloc)))
            elif spec.k_rot in (3, 4):  # small per-CTA phase: one B panel's (3) / one A panel's (4) CTAs in step
                grp = loc if spec.k_rot == 3 else b.div(loc, self.c32(bn))
                self.krot_off = b.to_sgpr_u32(b.mul(b.mod(grp, self.c32(spec.k_phase)),
                                                    self.c32(spec.k_phase_tiles)))
            tm = b.add(b.mul(b.div(x, self.c32(nbn)), self.c32(bm)), b.div(loc, self.c32(bn)))
            tn = b.add(b.mul(b.mod(x, self.c32(nbn)), self.c32(bn)), b.mod(loc, self.c32(bn)))
        self.tile_idx = {"A": tm, "B": tn}
        self.row0 = {"A": b.to_sgpr_u32(b.mul(tm, self.c32(spec.tile_m))),
                     "B": b.to_sgpr_u32(b.mul(tn, self.c32(spec.tile_n)))}
        if spec.edge_tiles:  # the last tile row / column ends at the matrix edge
            self.row0 = {X: (self.row0[X] if (X == "A" and spec.m_mask) else  # m_mask: rows past M masked instead
                             b.to_sgpr_u32(b.smin(self.row0[X], b.sub(self.p_dim[X], self.c32(self.rows[X])))))
                         for X in ("A", "B")}
        if spec.group_table:
            self._group_b(tm)
        # wave sub-tile origin (rows of A / rows of B within the CTA tile)
        self.wrow0 = {"A": b.to_sgpr_u32(b.mul(wm, self.c32(self.wt_m))),
                      "B": b.to_sgpr_u32(b.mul(wn, self.c32(self.wt_n)))}
        # LDS ring: one allocation per (stage, operand) so LLVM's alias scopes
        # keep refill DMAs and fragment reads of different stages apart
        self.stg = {}
        for s in range(self.S):
            for X in ("A", "B"):
                if not (X == "B" and spec.b_direct) and not (X == "A" and spec.a_direct) and X not in self.line_ops:
                    rb = MC_ROW_BYTES if (X == "A" and self.mc) else ROW_BYTES
                    self.stg[(s, X)] = b.smem_alloc(I32, [_kib(spec, self.rows[X] * rb) // 4], name_hint=f"stg{X}{s}")
                if spec.scale_src == "lds" and X not in self.srings:  # rows * 4 bytes: the tile's scale image
                    n_sc = scale_img_bytes(spec, X) if self.wtile else self.rows[X] * 4
                    self.stg[(s, "s" + X)] = b.smem_alloc(I32, [_kib(spec, n_sc) // 4], name_hint=f"sst{X}{s}")
        # line ring: one allocation per operand, 2 slots x rows x 128 B (a fragment read may span both slots)
        # the line2m remnant buffer sits after the R slots in the same allocation (one ds_read addresses both)
        # scale rings (raw A: [slot][row][16 B]; packed: [slot][P tile images]); named ring* (the direct compiler's
        # dmamanualring(w) levers leave the waits for their DMAs to the kernel's vmcnt model, as for the line rings);
        # allocated before the line rings so they start and end on whole KiB (the compiler's LDS DMA tags are KiB
        # ranges: sharing a KiB with a line ring would make every ring-scale read wait for the line refills)
        self.sr_lds = {X: b.smem_alloc(I32, [scale_ring_bytes(spec, X) // 4], name_hint=f"ring{X}S")
                       for X in self.srings}
        self.as_lds = self.sr_lds.get("A")
        self.ring = {X: b.smem_alloc(I32, [ring_alloc_bytes(spec, self.rows[X]) // 4], name_hint=f"ring{X}")
                     for X in self.line_ops}
        if spec.epi_lds:  # the C staging region lives in the largest operand allocation (free after the K loop)
            cands = [(self.ring[X], ring_alloc_bytes(spec, self.rows[X]) // 4) for X in self.line_ops]
            cands += [(arr, _kib(spec, self.rows[X] * (MC_ROW_BYTES if (X == "A" and self.mc) else ROW_BYTES)) // 4)
                      for (s_, X), arr in self.stg.items() if X in ("A", "B")]
            self.epi_buf = max(cands, key=lambda c: c[1])
        lane = self.lane
        self.layout = {X: operand_layout_of(spec, X) for X in ("A", "B")}
        self.afill = spec.a_fill == "c0c1"
        if self.afill:  # A read as a c0c1 stage image (built by _setup_afill's DMAs from the canonical rows)
            self.layout["A"] = "c0c1"
        if self.layout["A"] == self.layout["B"] == "canonical":
            self._setup_canonical(wave, lane)
        elif "canonical" not in self.layout.values():
            self._setup_c0c1(wave, lane)
        else:  # mixed: the canonical operand's DTLs / base / tile step from the canonical setup, the rest c0c1
            self._setup_canonical(wave, lane)
            can = {"dtl_list": dict(self.dtl_list), "base_ptr": dict(self.base_ptr), "tile_bytes": dict(self.tile_bytes)}
            self._setup_c0c1(wave, lane)
            for X in ("A", "B"):
                if self.layout[X] == "canonical":
                    self.dtl_list[X] = can["dtl_list"][X]
                    self.base_ptr[X] = can["base_ptr"][X]
                    self.tile_bytes[X] = can["tile_bytes"][X]
        self.dtl_dwx: Dict[str, List[int]] = {}
        if self.afill:
            self._setup_afill(wave, lane)
        if self.mc:  # M-contiguous A: its own DTLs / base / tile step / fragment reads (the c0c1 setup made B's)
            self._setup_mcontig(wave, lane)
        if spec.b_reg:  # B goes global -> VGPR -> ds_write (_breg_loads / _breg_writes), not by DTL
            self.dtl_list["B"] = []
            self.breg_rsrc = b.buffer_rsrc(self.base_ptr["B"], self.c32(0x7FFFFFF0))
            self.breg_voff = b.mul(lane, self.c32(16))
            self.breg_w = b.to_sgpr_u32(b.mul(wave, self.c32(n_breg(spec) * 1024)))
        # --- scale addressing -------------------------------------------------
        T = self.T
        grp_rows = 128 if self.tile_scales else 4 * T
        self.grp_rows = grp_rows
        self.nq = {"A": spec.m // grp_rows, "B": spec.n // grp_rows}
        self.sc_q0 = {X: b.to_sgpr_u32(b.div(b.add(self.row0[X], self.wrow0[X]), self.c32(grp_rows)))
                      for X in ("A", "B")}
        self.sc_rsrc = {
            "A": b.buffer_rsrc(self.psa, self.c32(0x7FFFFFF0)),
            "B": b.buffer_rsrc(self.psb, self.c32(0x7FFFFFF0)),
        }
        self.sc_voff = b.mul(lane, self.c32(8 if self.tile_scales else 16))
        # tile scales: dwords a lane loads per operand (one per 64 wave rows) and the wave's first dword
        self.sc_nd = {X: (max(1, wt // 64) if self.tile_scales and T == 16 else (2 if self.tile_scales else 4))
                      for X, wt in (("A", self.wt_m), ("B", self.wt_n))}
        self.sc_voff_x = {}
        for X in ("A", "B"):
            if self.tile_scales and T == 16 and self.sc_nd[X] == 1:
                half = b.div(b.mod(self.wrow0[X], self.c32(128)), self.c32(64))
                self.sc_voff_x[X] = b.add(self.sc_voff, b.mul(half, self.c32(4)))
            else:
                self.sc_voff_x[X] = self.sc_voff
        if spec.scale_src == "lds":
            # wave w copies bytes [256 w, 256 w + 256) of each operand's rows * 4 byte scale image
            self.sc_dtl_voff = b.mul(lane, self.c32(4))
            if self.raw_scales:
                # raw [rows, K/32]: lane l of a scale DTL copies row (image row + l)'s dword of the tile
                self.sc_nb = spec.k // 32
                self.sc_dtl_voff = b.mul(lane, self.c32(self.sc_nb))
                # v_perm selectors: byte kg (= lane / 16, the lane's 32-block of the 128-K tile) of two raw dwords
                # -> bytes 0, 1 (rest zero: selector 0x0C)
                kg = b.div(lane, self.c32(16))
                self.sc_perm_sel = b.lor(b.lor(kg, b.shl(b.add(kg, self.c32(4)), self.c32(8))), self.c32(0x0C0C0000))
            if spec.scale_dtl == "x4w":
                self.sc4_voff = b.mul(lane, self.c32(16))
            self.sc_dtl_w = b.to_sgpr_u32(b.mul(wave, self.c32(256)))
            self.sc_dtl_per_wave = {X: self.rows[X] * 4 // 256 // self.nw for X in ("A", "B")}
            if spec.probe in ("sc_noB", "sc_noAB"):  # timing probes: no per-tile scale DTL (stale scales)
                self.sc_dtl_per_wave["B"] = 0
                if spec.probe == "sc_noAB":
                    self.sc_dtl_per_wave["A"] = 0
            self.sr_soff0, self.sr_voff, self.sr_lds_w = {}, {}, {}
            for X, (kind, P, _lead, _slot) in self.srings.items():
                self.sc_dtl_per_wave[X] = 0  # X's scales come through its scale ring, not the stage image
                if kind == "packed":  # period block of row block q, period p at ((q NP + p) P) image bytes
                    img = self.rows[X] * 4
                    self.sr_soff0[X] = b.to_sgpr_u32(b.add(b.mul(self.tile_idx[X], self.c32(self.sr_np[X] * P * img)),
                                                           b.mul(self.wave, self.c32(P * img // self.nw))))
                    self.sr_voff[X] = b.mul(lane, self.c32(4 * P))
                    self.sr_lds_w[X] = b.to_sgpr_u32(b.mul(self.wave, self.c32(P * img // self.nw)))
            if self.as_ring:
                nb = spec.k // 32  # raw row stride (bytes)
                # DTL j of wave w: lane l copies the 16 bytes (4 K tiles) of row 64 (w + nw j) + l of the period
                self.as_voff = b.mul(lane, self.c32(nb))
                self.as_soff0 = b.to_sgpr_u32(b.mul(b.add(self.row0["A"], b.mul(self.wave, self.c32(64))),
                                                    self.c32(nb)))
                self.as_lds_w = b.to_sgpr_u32(b.mul(self.wave, self.c32(1024)))
                # v_perm selectors: byte kg (= lane / 16, the lane's 32-block of the 128-K tile) of two raw dwords
                kg = b.div(lane, self.c32(16))
                self.as_perm_sel = b.lor(b.lor(kg, b.shl(b.add(kg, self.c32(4)), self.c32(8))), self.c32(0x0C0C0000))
                # the lane's raw ring word of its wave row r0 = wave row 0 + l % 16 (16 bytes = 4 words per row)
                self.as_row_w = b.mul(b.add(self.wrow0["A"], b.mod(lane, self.c32(16))), self.c32(AS_PERIOD))
            # first 128-row scale chunk of the CTA (global) and of the wave (local)
            self.sc_cta_q = {X: b.to_sgpr_u32(b.div(self.row0[X], self.c32(128))) for X in ("A", "B")}
            self.sc_wq = {X: b.div(self.wrow0[X], self.c32(128)) for X in ("A", "B")}
        if self.wtile:
            self._setup_wtile(wave, lane)

    def _setup_wtile(self, wave: Value, lane: Value) -> None:
        """scale_layout wtile: per (K tile, CTA tile) one [wave side][64 lanes][D dwords] image of ``img`` bytes per
        operand (``layout.pack_scales_wtile``). LDS: copied by ceil(img / 256 / waves) dword DTLs per wave (the last
        chunk repeated, ``dtl_chunks``); the lane's D dwords sit at word (side 64 + lane) D. VGPR: the lane loads its
        D dwords itself."""
        b, spec = self.b, self.spec
        self.wsc_img = {X: scale_img_bytes(spec, X) for X in ("A", "B")}
        self.wsc_D = {X: wtile_dwords(wt) for X, wt in (("A", self.wt_m), ("B", self.wt_n))}
        self.wsc_nch = {X: self.wsc_img[X] // 256 for X in ("A", "B")}
        # loop-invariant image addressing: K tile t of CTA tile q at t (tiles per K tile) img + q img
        self.wsc_tstride = {X: b.to_sgpr_u32(b.mul(self._ntile_rows(X), self.c32(self.wsc_img[X]))) for X in ("A", "B")}
        self.wsc_off0 = {X: b.to_sgpr_u32(b.mul(self.tile_idx[X], self.c32(self.wsc_img[X]))) for X in ("A", "B")}
        sl = b.xor(lane, self.c32(16)) if spec.mutant == "scale_lane" else lane  # must-fail: the neighbour block
        self.wsc_ridx = {X: b.mul(b.add(b.mul(self.wsel[X], self.c32(64)), sl), self.c32(self.wsc_D[X]))
                         for X in ("A", "B")}
        if spec.scale_src == "lds":
            self.sc_dtl_per_wave = {X: -(-self.wsc_nch[X] // self.nw) for X in ("A", "B")}
            if spec.scale_dtl == "split":  # (X, chunk byte offset, issuing wave): A's chunks, then B's, round robin
                ch = [(X, q * 256) for X in ("A", "B") for q in range(self.wsc_nch[X])]
                self.wsc_split = [(X, off, c % self.nw) for c, (X, off) in enumerate(ch)]
            self.wsc_chunk = {X: [b.to_sgpr_u32(b.mul(b.smin(b.add(self.c32(q * self.nw), wave),
                                                              self.c32(self.wsc_nch[X] - 1)), self.c32(256)))
                                  for q in range(self.sc_dtl_per_wave[X])] for X in ("A", "B")}
        elif spec.scale_raw:
            # lane (q, r) = (l / 16, l % 16) loads raw row wave row 16 (4 d + q) + r (clamped into the matrix) for
            # each of its D dwords; the transpose selectors depend on the lane's half / 16-lane group parity
            nb = b.lshr(self.p_k, self.c32(5)) if self.rt else self.c32(spec.k // 32)
            q, r = b.div(sl, self.c32(16)), b.mod(sl, self.c32(16))
            self.sc_groups = {X: self.wsc_D[X] for X in ("A", "B")}
            self.raw_voff = {}
            for X in ("A", "B"):
                last = b.sub(self.p_dim[X] if self.rt else self.c32(spec.m if X == "A" else spec.n), self.c32(1))
                base_row = b.add(b.add(self.row0[X], self.wrow0[X]), r)
                self.raw_voff[X] = [b.mul(b.smin(b.add(base_row, b.mul(b.add(q, self.c32(4 * d)), self.c32(16))),
                                                 last), nb) for d in range(self.wsc_D[X])]
            self.raw_sel = raw_transpose_selectors(b, lane)
            self.raw_cache = {}
        else:
            Pw = spec.wtile_period  # period images: [period][CTA tile][side][64 lanes][P][D]
            if Pw > 1:
                self.wsc_tstride = {X: b.to_sgpr_u32(b.mul(self._ntile_rows(X), self.c32(self.wsc_img[X] * Pw)))
                                    for X in ("A", "B")}
                self.wsc_off0 = {X: b.to_sgpr_u32(b.mul(self.tile_idx[X], self.c32(self.wsc_img[X] * Pw)))
                                 for X in ("A", "B")}
            self.wsc_voff = {X: b.mul(sl, self.c32(4 * self.wsc_D[X] * Pw)) for X in ("A", "B")}
            self.wsc_woff = {X: b.to_sgpr_u32(b.mul(self.wsel[X], self.c32(256 * self.wsc_D[X] * Pw)))
                             for X in ("A", "B")}

    def _wsc_base(self, X: str, tile: Value) -> Value:
        """Global byte offset of operand ``X``'s wtile scale image of K tile ``tile`` for this CTA tile."""
        b = self.b
        return b.add(b.mul(tile, self.wsc_tstride[X]), self.wsc_off0[X])

    def _setup_mcontig(self, wave: Value, lane: Value) -> None:
        """A M-contiguous ([K][M] dense 6-bit codes, ``lda`` bytes per K row). Stage image: 128 K rows x nblk = tile_m /
        16 slots of 16 bytes; slot (k, b) holds the 12 code bytes of rows 16 b .. 16 b + 15 at K row k (+ the next 4
        bytes: the DMA fetches 16 overlapping bytes per lane, so the 96-bit rows sit 128-bit aligned with no padding
        in memory), at slot index k nblk + (b ^ s(k)) with the XOR swizzle s of ``mc_swizzle`` (bank-conflict free
        ``ds_read_b96_tr_b6``: the 16 lanes of a group read 16 consecutive K rows of one block). The 1 KiB DMA chunk c
        (2 nblk per tile, ``dtl_chunks`` over the waves) lane L fills slot 64 c + L.

        Fragment (f) read, half h: lane (g = l / 16, j = l % 16) addresses slot (32 g + 16 h + j, wb0 + f) (wb0 = the
        wave's first 16-row block); the transposing read hands lane 16 g + t row 16 (wb0 + f) + t's K values 32 g +
        16 h .. +15 -- the 16x16x128 MFMA's lane layout. Per lane: e = f mod 2^a VGPRs (wb0 + e) ^ s(j), the rest of f
        and h in the instruction offset."""
        b, spec = self.b, self.spec
        nblk = spec.tile_m // 16
        a2, sh = mc_swizzle(nblk)
        msk = (1 << a2) - 1
        nch = 2 * nblk
        lda = self.p_ld["A"] if self.rt else self.c32(spec.m * 6 // 8)
        per = -(-nch // self.nw)
        lst = []
        for q in range(per):
            chunk = b.to_sgpr_u32(b.smin(b.add(self.c32(q * self.nw), wave), self.c32(nch - 1)))
            s = b.add(b.mul(chunk, self.c32(64)), lane)
            k = b.div(s, self.c32(nblk))
            bp = b.mod(s, self.c32(nblk))
            blk = b.xor(bp, b.land(b.lshr(k, self.c32(sh)), self.c32(msk))) if a2 else bp
            voff = b.add(b.mul(k, lda), b.mul(blk, self.c32(12)))
            lst.append((voff, self.c32(0), b.to_sgpr_u32(b.mul(chunk, self.c32(1024))), 0))
        self.dtl_list["A"] = lst
        self.base_ptr["A"] = b.global_ptr_add(self.pa, b.lshr(b.mul(self.row0["A"], self.c32(3)), self.c32(2)))
        self.tile_bytes["A"] = b.to_sgpr_u32(b.mul(lda, self.c32(128))) if self.rt else spec.m * 6 // 8 * 128
        entry = b.mul(self.p_k, lda) if self.rt else self.c32(spec.k * (spec.m * 6 // 8))  # bytes of one entry
        self.mc_left = b.to_sgpr_u32(b.sub(entry, b.lshr(b.mul(self.row0["A"], self.c32(3)), self.c32(2))))
        if spec.group_table:  # grouped: A is one [K][tile_m] slab per tile row (lda = tile_m 6 / 8), slab r at r K lda
            slab = b.to_sgpr_u32(b.mul(self.p_k, lda))
            self.base_ptr["A"] = b.global_ptr_add(self.pa, b.mul(b.zext(self.tile_idx["A"], I64), b.zext(slab, I64)))
            self.mc_left = slab
        if getattr(self, "sk_a_shift32", None) is not None:  # splitk: A's base moved to the unit's first K tile
            self.mc_left = b.to_sgpr_u32(b.sub(self.mc_left, self.sk_a_shift32))
        if spec.canon_rsrc_once:
            self.canon_rsrc = {"A": b.buffer_rsrc(self.base_ptr["A"], self.c32(0x7FFFFFF0))}
        j = b.mod(lane, self.c32(16))
        g = b.div(lane, self.c32(16))
        sw = b.land(b.lshr(j, self.c32(sh)), self.c32(msk))
        if spec.mutant == "mc_noswz":  # must-fail: the reads ignore the slot swizzle the DMAs wrote
            sw = self.c32(0)
        wb0 = b.div(self.wrow0["A"], self.c32(16))
        lane_part = b.mul(b.add(b.mul(g, self.c32(32)), j), self.c32(16 * nblk))
        self.mc_ne = min(self.fm, 1 << a2)
        self.mc_lane = [b.add(lane_part, b.mul(b.xor(b.add(wb0, self.c32(e)), sw), self.c32(16)))
                        for e in range(self.mc_ne)]
        self.mc_geo = (nblk, a2)

    def _mc_read(self, stage: int, f: int) -> Value:
        """mcontig fragment ``f`` of the tile in ``stage``: two ds_read_b96_tr_b6 (K values 0..15, 16..31 of the lane's
        32-block) -> the six MFMA operand dwords."""
        b, spec = self.b, self.spec
        nblk, a2 = self.mc_geo
        e, q = f % (1 << a2), f // (1 << a2)  # (fm <= 2^a: one lane VGPR per fragment, q = 0)
        base = b.smem_addr_of(self.stg[(stage, "A")])
        addr = b.trunc(b.smem_ptr_add(base, b.zext(self.mc_lane[e], I64)), I32)
        parts = []
        for h in range(2):
            if h and spec.read_split:
                b.sched_barrier(0)
            off = 16 * (1 << a2) * q + 256 * nblk * h
            tmpl = f"ds_read_b96_tr_b6 $0, $1 offset:{off}" if off else "ds_read_b96_tr_b6 $0, $1"
            if spec.required_compiler != "rust":  # LLVM does not track an inline-asm LDS read: wait for it here
                tmpl += "\n\ts_waitcnt lgkmcnt(0)"
            v = b.inline_asm(tmpl, "=v,v", [addr], VectorType(I32, 3), sideeffect=True)
            if h and spec.mc_copy in ("late", "fence"):  # the copy waits for the fragment's first MFMA (``_mfma``)
                return _McFrag(parts, v)
            parts += [b.vec_extract(v, x) for x in range(3)]
        return b.vec_pack(parts, I32)

    def _mc_movs(self, v: Value, fence: bool) -> List[Value]:
        """mc_copy: the three dwords of a second-half ``ds_read_b96_tr_b6`` result through explicit ``v_mov_b32`` (one
        dword each, so the 6-dword MFMA operand takes them at its odd offset 3 without an alignment copy)."""
        b = self.b
        mut = self.spec.mutant == "mc_copyswap"  # must-fail: dwords 1 and 2 of the half swapped
        order = (0, 2, 1) if mut else (0, 1, 2)
        return [b.inline_asm("v_mov_b32 $0, $1", "=v,v" + (",~{memory}" if fence else ""), [b.vec_extract(v, x)], I32,
                             sideeffect=fence) for x in order]

    def _mc_resolve(self, f):
        """A deferred mcontig fragment (``mc_copy`` late / fence) packed at its MFMA; anything else as is."""
        if isinstance(f, _McFrag):
            if f.packed is None:
                f.packed = self.b.vec_pack(f.parts + self._mc_movs(f.half, fence=self.spec.mc_copy == "fence"), I32)
            return f.packed
        return f

    def _setup_canonical(self, wave: Value, lane: Value) -> None:
        b, spec, g = self.b, self.spec, self.geo
        # LDS unit U of a 32-row group: lane l of instruction q writes unit
        # U = 64 q + l -> row U // 6, unit U % 6; its source is global unit
        # dtl_source_unit(row, unit).
        self.dtl_voff = []
        for q in range(3):
            u = b.add(lane, self.c32(64 * q))
            r = b.div(u, self.c32(UNITS))
            c = b.mod(u, self.c32(UNITS))
            if spec.swizzle == "rot48":
                rot = b.land(b.lshr(r, self.c32(3)), self.c32(1))
                c = b.mod(b.add(c, b.mul(rot, self.c32(3))), self.c32(UNITS))
            self.dtl_voff.append(b.add(b.mul(r, self.c32(self.row_bytes)), b.mul(c, self.c32(DTL_BYTES))))
        # per (operand, wave-local issue j): voffset index, soffset and LDS byte offset (SGPRs, loop invariant)
        self.dtl_list: Dict[str, List[Tuple[Value, Value, Value]]] = {}
        if spec.canon_dtl == "interleave":
            # the stage image of an operand is rows x 96 B, linear; instruction i of wave w copies its 1 KiB chunk
            # i nw + w: lane l's 16-byte unit at tile byte 1024 (i nw + w) + 16 l -> row, unit (rot48 like band)
            for X in ("A", "B"):
                lst = []
                nch = self.rows[X] * ROW_BYTES // 1024
                for i in range(-(-nch // self.nw) if nch % self.nw else nch // self.nw):
                    chunk = b.add(self.c32(i * self.nw), wave)
                    if nch % self.nw:  # uneven split (library specs): the last chunk repeated (``dtl_chunks``)
                        chunk = b.to_sgpr_u32(b.smin(chunk, self.c32(nch - 1)))
                    tb = b.add(b.mul(chunk, self.c32(1024)), b.mul(lane, self.c32(DTL_BYTES)))
                    r, c = b.div(tb, self.c32(ROW_BYTES)), b.div(b.mod(tb, self.c32(ROW_BYTES)), self.c32(DTL_BYTES))
                    if spec.swizzle == "rot48":
                        rot = b.land(b.lshr(r, self.c32(3)), self.c32(1))
                        c = b.mod(b.add(c, b.mul(rot, self.c32(3))), self.c32(UNITS))
                    voff = b.add(b.mul(r, self._rb(X)), b.mul(c, self.c32(DTL_BYTES)))
                    loff = b.to_sgpr_u32(b.mul(chunk, self.c32(1024)))
                    lst.append((voff, self.c32(0), loff, 0))
                self.dtl_list[X] = lst
        for X in ("A", "B") if spec.canon_dtl == "band" else ():
            per_wave = self.rows[X] // ROWS_PER_GROUP // self.nw
            g0 = b.mul(wave, self.c32(per_wave))
            lst = []
            for gi in range(per_wave):
                grp = b.add(g0, self.c32(gi))
                soff = b.to_sgpr_u32(b.mul(grp, self.c32(ROWS_PER_GROUP * self.row_bytes)))
                loff = b.to_sgpr_u32(b.mul(grp, self.c32(ROWS_PER_GROUP * ROW_BYTES)))
                for q in range(3):
                    lst.append((self.dtl_voff[q], soff, loff, q * 64 * DTL_BYTES))
            self.dtl_list[X] = lst
        self.base_ptr = {X: b.global_ptr_add(p, b.mul(self.row0[X], self._rb(X)))
                         for X, p in (("A", self.pa), ("B", self.pb))}
        self.tile_bytes = {"A": 96, "B": 96}  # global bytes per 128-K tile step
        if spec.a_direct:  # A straight to VGPRs: no A DTLs; lane (r16, kg) reads row wrow0 + 16 f + r16 at 24 kg
            self.dtl_list["A"] = []
            self.ad_voff = b.add(b.mul(b.add(self.wrow0["A"], b.mod(lane, self.c32(16))), self._rb("A")),
                                 b.mul(b.div(lane, self.c32(16)), self.c32(24)))
            self.ad_soff = [b.to_sgpr_u32(b.mul(self.c32(16 * f), self._rb("A"))) for f in range(self.fm)]
        if self.line_ops:
            self._setup_lines(lane)
            for X in self.line_ops:
                self.dtl_list[X] = []  # filled by _line_issues
        if spec.canon_rsrc_once:
            self.canon_rsrc = {X: b.buffer_rsrc(self.base_ptr[X], self.c32(0x7FFFFFF0)) for X in ("A", "B")}
            if spec.m_mask or spec.k_tail:  # A's descriptor ends at row M: the rows past it read as zeros (the K
                #   offset rides in soffset, the row in voffset; k_tail: the last row's tail past the matrix too)
                self.canon_rsrc["A"] = b.buffer_rsrc(self.base_ptr["A"], b.to_sgpr_u32(
                    b.mul(b.sub(self.p_dim["A"], self.row0["A"]), self._rb("A"))))
        # --- fragment read addressing (i32 word index within a stage allocation)
        T = self.T
        rr = b.mod(lane, self.c32(T))
        grp = b.div(lane, self.c32(T))
        rot = b.land(b.lshr(rr, self.c32(3)), self.c32(1)) if spec.swizzle == "rot48" else self.c32(0)
        self.frag_word = {}
        for t in range(self.ksteps):
            j = b.add(grp, self.c32(t * g.groups))  # 24-byte chunk of the 96-byte row
            slot_words = b.mod(b.add(b.mul(j, self.c32(6)), b.mul(rot, self.c32(12))), self.c32(24))
            lw = b.add(b.mul(rr, self.c32(24)), slot_words)
            for X in ("A", "B"):
                self.frag_word[(t, X)] = b.add(lw, b.mul(self.wrow0[X], self.c32(24)))

    def _setup_lines(self, lane: Value) -> None:
        """canon_ring line2 addressing (``line_geometry``). Ring slot ``l % 2`` of row r holds the row's 128-byte line
        l. LDS byte of (slot, row, byte x of the line) = slot S + (r % 16) RB + (r / 16) 128 + x with RB = (rows / 16)
        128 + 16: the 32 lanes of a read pass hit distinct 8-byte bank slots (``line_geometry``) and the 16 rows
        r % 16 == c of a block of 8 row-groups are contiguous (one 1 KiB DTL).

        * line DTL c (c = i nw + w): rows 16 (8 (c / 16) + L / 8) + c % 16, unit L % 8 for lane L -> one voffset
          (16 (L / 8) row_bytes + 16 (L % 8)) for every instruction, the rest in soffset / M0;
        * fragment reads: lane (r16, kg), part h of tile u reads row byte p = 96 u + 24 kg + 8 h; n = p / 8 mod 32 =
          c + 3 kg with c = (12 u + h) mod 32 = 16 a + b: slot (n / 16) % 2, byte 8 (n % 16). For b + 3 kg < 16 the
          address is linear (lane part r16 RB + 24 kg + the wave rows, immediate a S + 8 b + 128 f); lanes with
          b + 3 kg >= 16 move to the other slot, one line back: + (1 - 2 a) S - 128 for kg >= kmin(b) -> one VGPR per
          (kmin, a), 7 per operand."""
        b = self.b
        if self.rt:  # runtime row strides: one lane offset per line operand (lda / ldb)
            self.line_voff = {X: b.add(b.mul(b.div(lane, self.c32(8)), b.mul(self._rb(X), self.c32(16))),
                                       b.mul(b.mod(lane, self.c32(8)), self.c32(DTL_BYTES))) for X in self.line_ops}
        else:
            self.line_voff = b.add(b.mul(b.div(lane, self.c32(8)), self.c32(16 * self.row_bytes)),
                                   b.mul(b.mod(lane, self.c32(8)), self.c32(DTL_BYTES)))
        # lane = (fragment row l % T, chunk group g = l / T); chunk j = G t + g of the tile's four 24-byte chunks
        T = self.T
        G = 64 // T
        rl = b.mod(lane, self.c32(T))
        kg = b.div(lane, self.c32(T))
        r16 = b.mod(rl, self.c32(16))
        rh = b.div(rl, self.c32(16))
        if self.mini:
            # remnant DTL lane L: row 32 j + L % 32, line bytes 96 + 16 (L / 32) .. +16

            # remnant read of part h: lane (r16, kg) reads line byte 96 + 24 kg + 8 h (lanes past the line read a
            # clamped in-range address, their value is replaced by the slot read)
            self.kg_ge = {k: b.cmp_ge(kg, self.c32(k)) for k in (1, 2, 3)}
            # remnant lane byte relative to the slot read's immediate a S + 8 b + 128 f (b = 12 + h on the group's
            # second tile): R S + (r16 / 4) G + (r16 % 4) 32 + 24 kg - 96 + 8 wrow0 - a S
            self.mini_rel = {}
            for X in self.line_ops:
                S_, _ = line_geometry(self.rows[X], self.T)
                v = b.add(b.add(self._mini_class_off(X, b.div(r16, self.c32(4))),
                                b.mul(b.mod(r16, self.c32(4)), self.c32(32))),
                          b.add(b.mul(kg, self.c32(24)), b.mul(self.wrow0[X], self.c32(LINE_BYTES // 16))))
                for a_ in (0, 1):  # + the buffer's offset R S in the ring allocation
                    self.mini_rel[(X, a_)] = b.add(v, self.c32(self.R * S_ - 96 - a_ * S_))
            # remnant copy (head line bytes 96..127 of the slot -> the buffer): wave w copies rows 64 w + lane
            if self.spec.mini_layout == "v3":
                # two lanes per row: lane l -> row 64 w + 32 it + 16 (l / 32) + l % 16, 8-byte half p = (l / 16) % 2
                rc16 = b.mod(lane, self.c32(16))
                rcb = b.add(b.mul(self.wave, self.c32(4)), b.div(lane, self.c32(32)))
                pb = b.mul(b.mod(b.div(lane, self.c32(16)), self.c32(2)), self.c32(8))
                self.mini_copy = {}
                for X in self.line_ops:
                    S_, RB = line_geometry(self.rows[X], self.T)
                    src = b.add(b.add(b.mul(rc16, self.c32(RB)), b.mul(rcb, self.c32(LINE_BYTES))),
                                b.add(pb, self.c32(96)))
                    dst = b.add(b.add(self._mini_class_off(X, b.div(rc16, self.c32(4))),
                                      b.mul(rcb, self.c32(LINE_BYTES))),
                                b.add(b.add(b.mul(b.mod(rc16, self.c32(4)), self.c32(32)), pb),
                                      self.c32(self.R * S_)))
                    self.mini_copy[X] = (b.lshr(src, self.c32(2)), b.lshr(dst, self.c32(2)))
        self.mini_copy_x: Dict[str, List[Tuple[Value, Value]]] = {}  # further copy passes (rows != 64 waves)
        if self.mini and self.spec.mini_layout in ("v2", "v4"):
            rc = b.add(b.mul(self.wave, self.c32(64)), lane)
            rc16, rcb = b.mod(rc, self.c32(16)), b.div(rc, self.c32(16))
            self.mini_copy = {}
            for X in self.line_ops:
                S_, RB = line_geometry(self.rows[X], self.T)
                Gm = mini_geometry(self.rows[X])[1]

                def copy_addr(rc16, rcb, X=X, S_=S_, RB=RB, Gm=Gm):
                    src = b.add(b.add(b.mul(rc16, self.c32(RB)), b.mul(rcb, self.c32(LINE_BYTES))), self.c32(96))
                    cls = (self._mini_class_off(X, b.div(rc16, self.c32(4))) if self.spec.mini_layout == "v4"
                           else b.mul(b.div(rc16, self.c32(4)), self.c32(Gm)))
                    dst = b.add(b.add(cls, b.mul(rcb, self.c32(LINE_BYTES))),
                                b.add(b.mul(b.mod(rc16, self.c32(4)), self.c32(32)), self.c32(self.R * S_)))
                    return b.lshr(src, self.c32(2)), b.lshr(dst, self.c32(2))

                rows = self.rows[X]
                if rows == 64 * self.nw:  # one row per lane and wave (the record kernels)
                    self.mini_copy[X] = copy_addr(rc16, rcb)
                    continue
                # pass p of wave w, lane l: row (p nw + w) 64 + l, clamped to the last row (rows < 64 waves: the
                # spare lanes copy the last row again -- the same bytes)
                passes = []
                for p_ in range(-(-rows // (64 * self.nw))):
                    rcp = b.smin(b.add(rc, self.c32(p_ * 64 * self.nw)), self.c32(rows - 1))
                    passes.append(copy_addr(b.mod(rcp, self.c32(16)), b.div(rcp, self.c32(16))))
                self.mini_copy[X] = passes[0]
                self.mini_copy_x[X] = passes[1:]
        # line DTL chunk c = i nw + w: rows 16 (8 (c / 16) + L / 8) + c % 16 -> (row byte offset, LDS offset), SGPRs
        self.line_chunk = {}
        for X in self.line_ops:
            _, RB_ = line_geometry(self.rows[X], self.T)
            n_i = self.rows[X] // (8 * self.nw)
            blk = (self.spec.line_dtl_order == "block" and (n_i * self.nw) % 16 == 0 and 16 % self.nw == 0)
            qpw = 16 // self.nw
            for i in range(n_i):
                if blk and i:
                    # block order: chunk i = row class w qpw + i % qpw of 16-class block i / qpw (rows 128 per block):
                    # chunk 0's offsets + (128 (i / qpw) + i % qpw) rows; LDS chunk 0's base + a constant
                    s0, l0 = self.line_chunk[(X, 0)][:2]
                    bi, j = divmod(i, qpw)
                    self.line_chunk[(X, i)] = (b.to_sgpr_u32(b.add(s0, b.mul(self.c32(128 * bi + j), self._rb(X)))), l0,
                                               j * RB_ + 8 * bi * LINE_BYTES)
                    continue
                if blk:  # i = 0: row class w qpw of block 0
                    r16_ = b.mul(self.wave, self.c32(qpw))
                    self.line_chunk[(X, 0)] = (b.to_sgpr_u32(b.mul(r16_, self._rb(X))),
                                               b.to_sgpr_u32(b.mul(r16_, self.c32(RB_))))
                    continue
                if self.spec.line_dtl_order == "wave" and 16 % n_i == 0 and i:
                    # consecutive row classes of one 16-class block: chunk 0's offsets + i rows / i RB (the LDS
                    # base differs by a constant, which m0share folds into one M0 + instruction offsets)
                    s0, l0 = self.line_chunk[(X, 0)][:2]
                    self.line_chunk[(X, i)] = (b.to_sgpr_u32(b.add(s0, b.mul(self.c32(i), self._rb(X)))), l0,
                                               i * RB_)  # LDS: chunk 0's base + a constant (one M0, offsets)
                    continue
                if self.spec.line_dtl_order == "wave":  # this wave's DTLs on consecutive row classes
                    c = b.add(b.mul(self.wave, self.c32(n_i)), self.c32(i))
                else:
                    c = b.add(self.c32(i * self.nw), self.wave)
                r16_ = b.mod(c, self.c32(16))
                jb = b.mul(b.div(c, self.c32(16)), self.c32(8))
                row0 = b.add(b.mul(jb, self.c32(16)), r16_)
                self.line_chunk[(X, i)] = (
                    b.to_sgpr_u32(b.mul(row0, self._rb(X))),
                    b.to_sgpr_u32(b.add(b.mul(r16_, self.c32(RB_)), b.mul(jb, self.c32(LINE_BYTES)))))
        if self.mini and self.spec.mini_src == "dtl":
            rl_ = b.add(b.mul(b.div(lane, self.c32(8)), self.c32(16)), b.div(b.mod(lane, self.c32(8)), self.c32(2)))
            if self.rt:
                self.mini_voff = {X: b.add(b.mul(rl_, self._rb(X)),
                                           b.add(self.c32(96), b.mul(b.mod(lane, self.c32(2)), self.c32(16))))
                                  for X in self.line_ops}
            else:
                self.mini_voff = b.add(b.mul(rl_, self.c32(self.row_bytes)),
                                       b.add(self.c32(96), b.mul(b.mod(lane, self.c32(2)), self.c32(16))))
        self.line_lane = {}  # (X, kmin, a) -> byte offset VGPR (kmin 0: no wrap)
        self.mini_sel = {}  # line2m: (X, kmin, a) -> word base: lanes kg >= kmin the next slot, the rest the remnant
        for X in self.line_ops:
            S_, RB = line_geometry(self.rows[X], T)
            v0 = b.add(b.add(b.add(b.mul(r16, self.c32(RB)), b.mul(rh, self.c32(LINE_BYTES))),
                             b.mul(kg, self.c32(24))), b.mul(self.wrow0[X], self.c32(LINE_BYTES // 16)))
            self.line_lane[(X, 0, 0)] = b.lshr(v0, self.c32(2))  # word indices (all offsets are 8-byte multiples)
            for kmin in range(1, G):
                ge = b.cmp_ge(kg, self.c32(kmin))
                for last in (0, 1):  # the wrapping lanes go one slot on (last: back to slot 0), one line back
                    corr = (-(self.R - 1) * S_ if last else S_) - LINE_BYTES
                    self.line_lane[(X, kmin, last)] = b.lshr(b.select(ge, b.add(v0, self.c32(corr)), v0),
                                                             self.c32(2))
            if self.mini:  # the group's second tile: non-wrapped lanes (still in the head line) -> remnant buffer
                for kmin in range(1, G):
                    ge = b.cmp_ge(kg, self.c32(kmin))
                    for a_ in (0, 1):
                        last = int(a_ == self.R - 1)
                        S_, _ = line_geometry(self.rows[X], T)
                        corr = (-(self.R - 1) * S_ if last else S_) - LINE_BYTES
                        self.mini_sel[(X, kmin, a_)] = b.lshr(
                            b.select(ge, b.add(v0, self.c32(corr)), self.mini_rel[(X, a_)]), self.c32(2))

    def _mini_class_off(self, X: str, i: Value) -> Value:
        """O_i (``mini_offsets``) of row class ``i`` (a VGPR, 0..3) of operand ``X``'s remnant buffer."""
        b = self.b
        gs = (self.rows[X] // 16) * LINE_BYTES
        if self.spec.mini_layout in ("v3", "v4"):  # i gs + 16 (i % 2) + 128 (i / 2)
            return b.add(b.mul(i, self.c32(gs)), b.add(b.mul(b.mod(i, self.c32(2)), self.c32(16)),
                                                      b.mul(b.div(i, self.c32(2)), self.c32(128))))
        return b.mul(i, self.c32(gs + 8))

    def _mini_copy_read(self, slot: int, out: Dict, ops: Optional[Tuple[str, ...]] = None) -> None:
        """line2m: read the head line's bytes 96..127 (slot ``slot``) of the wave's rows (v2: one row per lane, two
        16-byte reads; v3: two lanes per row, four 8-byte reads over two 32-row passes) into ``out`` -- written by
        :meth:`_mini_copy_write` a few MFMA slots later (no wait stall)."""
        b = self.b
        if self.spec.mini_layout == "v3":
            for X in (self.line_ops if ops is None else ops):
                S_, _ = line_geometry(self.rows[X], self.T)
                src, _ = self.mini_copy[X]
                for it in range(2):
                    for k in range(2):
                        off = (slot * S_ + 2 * LINE_BYTES * it + 16 * k) // 4
                        out[(X, it, k)] = b.smem_load_vN(self.ring[X], b.add(src, self.c32(off)), dtype=I32, n=2)
            return
        for X in (self.line_ops if ops is None else ops):
            S_, _ = line_geometry(self.rows[X], self.T)
            src, _ = self.mini_copy[X]
            for k in range(2):
                out[(X, k)] = b.smem_load_vN(self.ring[X], b.add(src, self.c32((slot * S_ + 16 * k) // 4)),
                                             dtype=I32, n=4)
            for p_, (src, _) in enumerate(self.mini_copy_x.get(X, ())):
                for k in range(2):
                    out[(X, "x", p_, k)] = b.smem_load_vN(self.ring[X], b.add(src, self.c32((slot * S_ + 16 * k) // 4)),
                                                          dtype=I32, n=4)

    def _mini_copy_write(self, vals: Dict, ops: Optional[Tuple[str, ...]] = None) -> None:
        """line2m: four 8-byte writes per lane into the remnant buffer (its group stride is 8-byte aligned)."""
        b = self.b
        if self.spec.mini_layout == "v3":
            for X in (self.line_ops if ops is None else ops):
                _, dst = self.mini_copy[X]
                for it in range(2):
                    for k in range(2):
                        off = (2 * LINE_BYTES * it + 16 * k) // 4
                        b.smem_store_vN(self.ring[X], [b.add(dst, self.c32(off))], vals[(X, it, k)], 2)
            return
        if self.spec.mini_layout == "v4":  # 16-byte aligned entries: two 16-byte writes per lane
            for X in (self.line_ops if ops is None else ops):
                _, dst = self.mini_copy[X]
                for k in range(2):
                    b.smem_store_vN(self.ring[X], [b.add(dst, self.c32(4 * k))], vals[(X, k)], 4)
                for p_, (_, dst) in enumerate(self.mini_copy_x.get(X, ())):
                    for k in range(2):
                        b.smem_store_vN(self.ring[X], [b.add(dst, self.c32(4 * k))], vals[(X, "x", p_, k)], 4)
            return
        for X in (self.line_ops if ops is None else ops):
            pas = [(self.mini_copy[X][1], lambda k: (X, k))]
            pas += [(d, lambda k, p_=p_: (X, "x", p_, k)) for p_, (_, d) in enumerate(self.mini_copy_x.get(X, ()))]
            for dst, key in pas:
                for k in range(2):
                    v = vals[key(k)]
                    for e in range(2):
                        w = b.vec_pack([b.vec_extract(v, 2 * e), b.vec_extract(v, 2 * e + 1)], I32)
                        b.smem_store_vN(self.ring[X], [b.add(dst, self.c32((16 * k + 8 * e) // 4))], w, 2)

    def _mini_copy_dummy(self, ops) -> Dict:
        """Constant stand-ins for the copy's read values (probe l2m_copywrite)."""
        b = self.b
        z = b.vec_pack([self.c32(0), self.c32(0)], I32)
        z4 = b.vec_pack([self.c32(0)] * 4, I32)
        out: Dict = {}
        for X in ops:
            for k in range(2):
                out[(X, k)] = z4
                for it in range(2):
                    out[(X, it, k)] = z
        return out

    def _mini_copy(self, slot: int) -> None:
        cp: Dict = {}
        self._mini_copy_read(slot, cp)
        self._mini_copy_write(cp)

    def _breg_loads(self, tile: Value, out: Dict) -> List[Callable[[], None]]:
        """b_reg: the wave's B loads of 128-K tile ``tile`` into ``out[("BR", q)]`` (1 KiB per load and wave)."""
        b = self.b
        if self.spec.k_rot:  # the XCD's K offset, as the DTL refills
            t2 = b.add(tile, self.krot_off)
            tile = b.select(b.cmp_lt(t2, self.c32(self.n_tiles)), t2, b.sub(t2, self.c32(self.n_tiles)))
        kb = b.mul(tile, self.c32(self.tile_bytes["B"]))
        res = []
        for q in range(n_breg(self.spec)):

            def ld(q=q):
                soff = b.add(b.add(kb, self.breg_w), self.c32(q * 1024))
                out[("BR", q)] = b.buffer_load_vN(self.breg_rsrc, self.breg_voff, soff, I32, 4)
                self._vm_fence("breg")

            res.append(ld)
        return res

    def _breg_writes(self, stage: int, vals: Dict) -> None:
        """b_reg: ds_write the staged B tile (``vals[("BR", q)]``) into ``stage`` (the DTL image layout)."""
        b = self.b
        base = b.lshr(self.breg_w, self.c32(2))
        for q in range(n_breg(self.spec)):
            idx = b.add(b.add(base, self.c32(q * 256)), b.mul(self.lane, self.c32(4)))
            b.smem_store_vN(self.stg[(stage, "B")], [idx], vals[("BR", q)], 4)

    def _line_kmin(self, u8: int, h: int, t: int = 0) -> int:
        """First chunk group whose part ``h`` of tile ``u % LP == u8`` lies in the window's next line (0: none)."""
        G = 64 // self.T
        b_ = ((12 * u8 + 3 * G * t + h) % (16 * self.R)) % 16
        return next((g for g in range(1, G) if b_ + 3 * g >= 16), 0)

    def _line_read_addr(self, X: str, u8: int, h: int, f: int, t: int = 0) -> Value:
        """Word index of fragment ``(f, t)`` part ``h`` of a tile ``u % LP == u8`` in operand ``X``'s ring: 8-byte unit
        n = (12 u + 3 G t + h + 3 g) mod 16 R of the R-line window (chunk group g = lane / T, G = 64 / T), slot
        n / 16, byte 8 (n % 16)."""
        b = self.b
        T = self.T
        G = 64 // T
        S_, _ = line_geometry(self.rows[X], T)
        c = (12 * u8 + 3 * G * t + h) % (16 * self.R)
        a_, b_ = divmod(c, 16)
        kmin = next((g for g in range(1, G) if b_ + 3 * g >= 16), 0)  # first chunk group past the line end
        base = self.line_lane[(X, kmin, int(a_ == self.R - 1)) if kmin else (X, 0, 0)]
        if self.mini and u8 % 4 == 1 and self.spec.probe != "l2m_nodual":
            assert kmin, "the group's second tile always straddles its head line"
            base = self.mini_sel[(X, kmin, a_)]
        imm = a_ * S_ + 8 * b_ + LINE_BYTES * (f * T // 16)  # f: T rows = T / 16 row-block steps
        return b.add(base, self.c32(imm // 4))  # the immediate folds into the ds_read offset

    def _line_issues(self, tile: Value, u8: int, ops: Optional[Tuple[str, ...]] = None) -> List[Callable[[], None]]:
        """canon_ring line2: the DTLs of the line that 128-K tile ``tile`` (``tile % 8 == u8``) brings in (none when
        ``tile % 4 == 3``: its 96 bytes lie in lines already loaded)."""
        b = self.b
        q = u8 % 4
        if q == 3:
            return []
        out: List[Callable[[], None]] = []
        if self.spec.k_rot:  # global K tile of this XCD's step (offset a multiple of 8 tiles: u mod 8 unchanged)
            t2 = b.add(tile, self.krot_off)
            tile = b.select(b.cmp_lt(t2, self.c32(self.n_tiles)), t2, b.sub(t2, self.c32(self.n_tiles)))
        line = b.add(b.mul(b.div(tile, self.c32(4)), self.c32(3)), self.c32(q))
        # slot = line % R, line = 3 (tile / 4) + q: line3 -> q; line2 -> parity, u8 / 4 = (tile / 4) % 2
        slot = q if self.R == 3 else (3 * (u8 // 4) + q) % 2
        lbyte = b.mul(line, self.c32(LINE_BYTES))
        for X in self.line_ops:
            if (ops is not None and X not in ops) or self.spec.probe == f"ldtl_no{X}":  # probe: no fetch of X's lines
                continue
            rows = self.rows[X]
            S_, RB = line_geometry(rows, self.T)
            holder: List[Value] = []

            def rsrc(X=X, holder=holder):
                if self.spec.canon_rsrc_once:
                    return self.canon_rsrc[X]
                if not holder:
                    holder.append(b.buffer_rsrc(self.base_ptr[X], self.c32(0x7FFFFFF0)))
                return holder[0]

            lds0 = b.smem_addr_of(self.ring[X])
            for i in range(rows // (8 * self.nw)):

                def issue(X=X, i=i, lds0=lds0, rsrc=rsrc, S_=S_, RB=RB):
                    ch = self.line_chunk[(X, i)]  # loop-invariant row / LDS offsets of the chunk
                    soff0, ldsb = ch[0], ch[1]
                    extra = ch[2] if len(ch) > 2 else 0  # line_dtl_order wave: constant past chunk 0's LDS base
                    soff = b.add(soff0, lbyte)
                    lds_b = b.add(ldsb, self.c32(slot * S_ + extra))  # slot < R
                    lds = b.smem_ptr_add(lds0, b.zext(lds_b, I64))
                    voff = self.line_voff[X] if self.rt else self.line_voff
                    if self.spec.probe == "ldtl_contigB" and X == "B":  # timing probe: one contiguous 1 KiB run
                        voff = b.mul(self.lane, self.c32(16))
                        soff = b.add(b.mul(b.add(self.c32(i * self.nw), self.wave), self.c32(1024)), lbyte)
                    b.async_buffer_load_lds_addr(rsrc(), lds, voff, soff, 4, coherency=self._aux(X))
                    self._vm_fence("ldtl")

                out.append(issue)
        if self.mini and q == 0 and _n_mini_dtl(self.spec) and self.spec.probe != "l2m_nomini":  # the head line's tail -> the remnant buffer (v4 layout)
            for X in self.line_ops:
                if (ops is not None and X not in ops) or self.spec.probe == f"ldtl_no{X}":
                    continue
                rows = self.rows[X]
                S_, _ = line_geometry(rows, self.T)
                gs = (rows // 16) * LINE_BYTES
                lds0m = b.smem_addr_of(self.ring[X])

                def rsrc_m(X=X):
                    return self.canon_rsrc[X] if self.spec.canon_rsrc_once else \
                        b.buffer_rsrc(self.base_ptr[X], self.c32(0x7FFFFFF0))

                for i in range(rows // (32 * self.nw)):

                    def mini_issue(X=X, i=i, lds0m=lds0m, rsrc_m=rsrc_m, S_=S_, gs=gs):
                        # DTL c = i nw + w: row class ci = c / 2, row blocks 8 (c % 2) .. + 8 -> 1 KiB contiguous
                        c = b.add(self.c32(i * self.nw), self.wave)
                        ci, h2 = b.div(c, self.c32(2)), b.mod(c, self.c32(2))
                        rowc = b.add(b.mul(h2, self.c32(128)), b.mul(ci, self.c32(4)))  # 16 (8 h2) + 4 ci
                        soff = b.add(b.mul(rowc, self._rb(X)), lbyte)
                        ldsb = b.add(b.add(self._mini_class_off(X, ci), b.mul(h2, self.c32(8 * LINE_BYTES))),
                                     self.c32(self.R * S_))
                        lds = b.smem_ptr_add(lds0m, b.zext(b.to_sgpr_u32(ldsb), I64))
                        b.async_buffer_load_lds_addr(rsrc_m(), lds, self.mini_voff[X] if self.rt else self.mini_voff,
                                                     soff, 4,
                                                     coherency=self._aux(X))
                        self._vm_fence("ldtl")

                    out.append(mini_issue)
        return out

    def _setup_afill(self, wave: Value, lane: Value) -> None:
        """a_fill c0c1: canonical A rows in global, the c0c1 stage image in LDS (reads: _setup_c0c1's).

        c0 plane, 1 KiB chunk q (rows 16 q .. 16 q + 15): lane l = 16 kg + r16 fetches the 16 bytes at row byte 24 kg
        (dwordx4, dword-aligned source) into slot 64 q + l. c1 plane (at 64 R): dword DMA p (256 B, slots 32 p ..
        32 p + 31) lane j fetches the 4 bytes at row byte 24 kg + 16 + 4 (j % 2) of slot 32 p + j / 2, i.e. row
        16 (p / 2) + (j / 2) % 16, kg = 2 (p % 2) + j / 32."""
        b = self.b
        R, nw = self.rows["A"], self.nw
        rb = self._rb("A")
        self.base_ptr["A"] = b.global_ptr_add(self.pa, b.mul(self.row0["A"], rb))
        self.tile_bytes["A"] = 96  # global bytes per 128-K tile step of a row
        v0 = b.add(b.mul(b.mod(lane, self.c32(16)), rb), b.mul(b.div(lane, self.c32(16)), self.c32(24)))
        v1 = b.add(b.add(b.mul(b.mod(b.div(lane, self.c32(2)), self.c32(16)), rb),
                         b.mul(b.div(lane, self.c32(32)), self.c32(24))),
                   b.add(self.c32(16), b.mul(b.mod(lane, self.c32(2)), self.c32(4))))
        lst, dwx = [], []
        for i in range(R // 16 // nw):
            q = b.add(self.c32(i * nw), wave)
            lst.append((v0, b.to_sgpr_u32(b.mul(b.mul(q, self.c32(16)), rb)),
                        b.to_sgpr_u32(b.mul(q, self.c32(1024))), 0))
            dwx.append(4)
        for i in range(R // 8 // nw):
            p_ = b.add(self.c32(i * nw), wave)
            soff = b.add(b.mul(b.mul(b.div(p_, self.c32(2)), self.c32(16)), rb),
                         b.mul(b.mod(p_, self.c32(2)), self.c32(48)))
            lst.append((v1, b.to_sgpr_u32(soff), b.to_sgpr_u32(b.add(b.mul(p_, self.c32(256)), self.c32(64 * R))), 0))
            dwx.append(1)
        self.dtl_list["A"] = lst
        self.dtl_dwx["A"] = dwx
        if self.spec.canon_rsrc_once:
            if not hasattr(self, "canon_rsrc"):
                self.canon_rsrc = {}
            self.canon_rsrc["A"] = b.buffer_rsrc(self.base_ptr["A"], self.c32(0x7FFFFFF0))

    def _setup_c0c1(self, wave: Value, lane: Value) -> None:
        b, spec, g = self.b, self.spec, self.geo
        # DTL: the stage image of an operand is one contiguous block; lane l of the
        # wave's instruction q copies bytes [1 KiB (wave * per_wave + q) + 16 l, +16)
        self.dtl_voff = [b.mul(lane, self.c32(16))]
        self.dtl_list = {}
        for X in ("A", "B"):
            if X == "A" and self.mc:  # _setup_mcontig builds A's
                continue
            nch = self.rows[X] * ROW_BYTES // 1024
            if nch % self.nw:  # uneven split (library specs): wave w copies chunks q nw + w, the last one repeated
                lst = []
                for q in range(-(-nch // self.nw)):
                    cb = b.to_sgpr_u32(b.mul(b.smin(b.add(self.c32(q * self.nw), wave), self.c32(nch - 1)),
                                             self.c32(1024)))
                    lst.append((self.dtl_voff[0], cb, cb, 0))
                self.dtl_list[X] = [] if (X == "B" and spec.b_direct) else lst
                continue
            per_wave = self.rows[X] * ROW_BYTES // 1024 // self.nw
            base = b.to_sgpr_u32(b.mul(wave, self.c32(per_wave * 1024)))
            lst = []
            for q in range(per_wave):
                lst.append((self.dtl_voff[0], base, base, q * 1024))
            self.dtl_list[X] = [] if (X == "B" and spec.b_direct) else lst
        nt = self.n_tiles
        if self.rt:  # the c0c1 image of a row block is K / 128 tile images
            self.base_ptr = {
                X: b.global_ptr_add(p, b.mul(self.tile_idx[X], b.mul(getattr(self, "k_tiles_full", self.k_tiles),
                                                                     self.c32(self.rows[X] * ROW_BYTES))))
                for X, p in (("A", self.pa), ("B", self.pb))
            }
        else:
            self.base_ptr = {
                X: b.global_ptr_add(p, b.mul(self.tile_idx[X], self.c32(nt * self.rows[X] * ROW_BYTES)))
                for X, p in (("A", self.pa), ("B", self.pb))
            }
        self.tile_bytes = {X: self.rows[X] * ROW_BYTES for X in ("A", "B")}
        # reads: slot o = 64 rb + 16 kg + r16; c0 at 16 o, c1 at 64 R + 8 o (bytes)
        if self.T == 16:
            o_lane = lane  # rb within the fragment's 16-row block = 0, kg = l // 16, r16 = l % 16
            o_t = [0]
        else:  # 32x32x64: row l % 32 -> rb' = (l % 32) // 16, kg = 2 t + l // 32
            o_lane = b.add(b.add(b.mul(b.div(b.mod(lane, self.c32(32)), self.c32(16)), self.c32(64)),
                                 b.mul(b.div(lane, self.c32(32)), self.c32(16))), b.mod(lane, self.c32(16)))
            o_t = [0, 32]
        self.c0c1_lane = o_lane
        self.c0c1_t = o_t
        self.c0c1_wrb = {X: b.div(self.wrow0[X], self.c32(16)) for X in ("A", "B")}
        if spec.operand_layout == "c0c1p":
            # dword index of plane 0 of the wave's first fragment pair (wave rows start on a 32-row pair):
            # 768 (wrb / 2) + 4 lane, computed once; a pair / plane adds only immediates
            self.c0c1p_base = {X: b.add(b.mul(b.div(self.c0c1_wrb[X], self.c32(2)), self.c32(768)),
                                        b.mul(o_lane, self.c32(4))) for X in ("A", "B")}
        if spec.rsrc_once:
            self.op_rsrc = {X: b.buffer_rsrc(self.base_ptr[X], self.c32(0x7FFFFFF0)) for X in ("A", "B")}
        if spec.b_direct:
            # c0 of fragment f: bytes 1024 (wrb + f) + 16 l; c1: 64 R + 512 (wrb + f) + 8 l
            self.bd_voff = (b.mul(lane, self.c32(16)), b.mul(lane, self.c32(8)))
            self.bd_soff = (b.to_sgpr_u32(b.mul(self.c0c1_wrb["B"], self.c32(1024))),
                            b.to_sgpr_u32(b.add(b.mul(self.c0c1_wrb["B"], self.c32(512)),
                                                self.c32(64 * self.rows["B"]))))

    # ------------------------------------------------------------- building blocks
    def _dtl_issues(self, tile: Value, stage: int) -> List[Callable[[], None]]:
        """One closure per DTL instruction filling ``stage`` with 128-K tile ``tile``."""
        b = self.b
        out: List[Callable[[], None]] = []
        if self.spec.k_rot:  # global K tile of this XCD's step ``tile``: (tile + offset) mod T
            t2 = b.add(tile, self.krot_off)
            tile = b.select(b.cmp_lt(t2, self.c32(self.n_tiles)), t2, b.sub(t2, self.c32(self.n_tiles)))
        if self.sc_lds and self.spec.scale_dtl == "split":
            for X, off, w in self.wsc_split:
                lds0 = b.smem_addr_of(self.stg[(stage, "s" + X)])
                base = self._wsc_base(X, tile)

                def scs(X=X, off=off, w=w, lds0=lds0, base=base):
                    with b.scf_if(b.cmp_eq(self.wave, self.c32(w))):
                        lds = b.smem_ptr_add(lds0, b.zext(self.c32(off), I64))
                        b.async_buffer_load_lds_addr(self.sc_rsrc[X], lds, self.sc_dtl_voff, b.add(base, self.c32(off)),
                                                     1)
                    self._vm_fence("sdtl")

                out.append(scs)
        if self.sc_lds and self.spec.scale_dtl == "x4w":
            # one 1 KiB dwordx4 DTL per operand from one wave (A: wave 0, B: wave nw / 2), first in the refill so
            # it is never a tile's last awaited VM op; the other waves' vmcnt model (no scale op) is the strictest
            for X, w in (("A", 0), ("B", self.nw // 2)):
                lds0 = b.smem_addr_of(self.stg[(stage, "s" + X)])
                base = b.mul(b.add(b.mul(tile, self.c32(self.nq[X])), self.sc_cta_q[X]), self.c32(512))

                def sc4(X=X, w=w, lds0=lds0, base=base):
                    with b.scf_if(b.cmp_eq(self.wave, self.c32(w))):
                        b.async_buffer_load_lds_addr(self.sc_rsrc[X], lds0, self.sc4_voff, base, 4)
                    self._vm_fence("sdtl")

                out.append(sc4)
        for X in ("A", "B"):
            if not self.dtl_list[X]:
                continue
            tb = self.tile_bytes[X]  # (mcontig + runtime_shape: a Value, 128 lda)
            kbyte = b.mul(tile, tb if isinstance(tb, Value) else self.c32(tb))
            if self.spec.probe == "canon_align" and self.layout[X] == "canonical":
                # timing probe (not a GEMM): every canonical 96-byte row segment starts 128-byte aligned (the
                # tile steps 128 bytes, wrapping inside the row) -> the cost of the 96-byte segment misalignment
                kbyte = b.mul(b.mod(tile, self.c32(self.row_bytes // 128)), self.c32(128))
            holder: List[Value] = []

            canon = self.layout[X] in ("canonical", "mcontig") or (X == "A" and self.afill)  # mcontig / a_fill: per-lane
            #   voffsets like the canonical DTLs
            once = self.spec.canon_rsrc_once if canon else self.spec.rsrc_once

            def rsrc(X=X, holder=holder, kbyte=kbyte, once=once, canon=canon):
                if once:
                    return self.canon_rsrc[X] if canon else self.op_rsrc[X]
                if not holder:
                    # mcontig: the descriptor ends at the entry's end, so the last chunk's 4 overlap bytes past the
                    # final K row read 0 instead of past the buffer (no tail padding needed)
                    gp = b.global_ptr_add(self.base_ptr[X], kbyte)  # (op order kept: the IR of every other spec)
                    nrec = b.sub(self.mc_left, kbyte) if (X == "A" and self.mc) else self.c32(0x7FFFFFF0)
                    holder.append(b.buffer_rsrc(gp, nrec))
                return holder[0]

            kadd = kbyte if once else None  # the tile offset joins the soffset

            lds0 = b.smem_addr_of(self.stg[(stage, X)])
            if self.spec.probe == "canon_line" and canon:
                # timing probe (not a GEMM): the global side of a line-granular ring -- per tile, 3/4 of the rows
                # (all but the 64-row block tile % 4) fetch one whole 128-byte line (line 3 tile / 4), 8 lanes per
                # row: the same instruction count, 0.75 requests per row and tile instead of 1.5
                nchunk = self.rows[X] * ROW_BYTES // 1024
                q4 = b.mod(tile, self.c32(4))
                lbyte = b.mul(b.div(b.mul(tile, self.c32(3)), self.c32(4)), self.c32(128))
                for i in range(nchunk // self.nw):

                    def line_issue(i=i, X=X, lds0=lds0, q4=q4, lbyte=lbyte):
                        c = b.add(self.c32(i * self.nw), self.wave)
                        r0 = b.mul(c, self.c32(8))
                        r0 = b.select(b.cmp_lt(r0, b.mul(q4, self.c32(64))), r0, b.add(r0, self.c32(64)))
                        row = b.add(r0, b.div(self.lane, self.c32(8)))
                        voff = b.add(b.mul(row, self.c32(self.row_bytes)), b.mul(b.mod(self.lane, self.c32(8)),
                                                                                 self.c32(DTL_BYTES)))
                        lds = b.smem_ptr_add(lds0, b.zext(b.mul(c, self.c32(1024)), I64))
                        rs = b.buffer_rsrc(self.base_ptr[X], self.c32(0x7FFFFFF0))
                        b.async_buffer_load_lds_addr(rs, lds, voff, lbyte, 4, coherency=self.spec.dtl_aux)
                        self._vm_fence()

                    out.append(line_issue)
                continue
            # mc_dma x3: the mcontig slot's 12 bytes only (b96 DMA, 16-byte LDS lane stride)
            dwx0 = 3 if (X == "A" and self.mc and self.spec.mc_dma == "x3") else 4
            for di, (voff, soff, loff, lconst) in enumerate(self.dtl_list[X]):
                dwx = self.dtl_dwx[X][di] if X in self.dtl_dwx else dwx0

                def issue(voff=voff, soff=soff, loff=loff, lconst=lconst, rsrc=rsrc, lds0=lds0, kadd=kadd,
                          canon=canon, dwx=dwx, aux=self._aux(X)):
                    sbase = soff if kadd is None else b.add(soff, kadd)
                    if self.spec.dtl_imm and not canon:
                        blk, imm = (lconst // 4096) * 4096, lconst % 4096
                        lds = b.smem_ptr_add(lds0, b.zext(b.add(loff, self.c32(blk)), I64))
                        b.async_buffer_load_lds_imm(rsrc(), lds, voff, b.add(sbase, self.c32(blk)), 4, imm,
                                                    coherency=aux)
                        self._vm_fence()
                        return
                    lds = b.smem_ptr_add(lds0, b.zext(b.add(loff, self.c32(lconst)), I64))
                    if not canon:
                        # the instruction's 1 KiB block offset rides in soffset (global side only)
                        b.async_buffer_load_lds_addr(rsrc(), lds, voff, b.add(sbase, self.c32(lconst)), 4,
                                                     coherency=aux)
                    else:
                        b.async_buffer_load_lds_addr(rsrc(), lds, voff, sbase, dwx, coherency=aux)
                    self._vm_fence()

                out.append(issue)
        if self.sc_lds and self.spec.scale_dtl == "dword":  # (split: issued above)
            for X in ("A", "B"):
                if not self.sc_dtl_per_wave[X]:  # a_scale_src ring: A's scales are not in the stage image
                    continue
                lds0 = b.smem_addr_of(self.stg[(stage, "s" + X)])
                if self.wtile:  # per-wave images: chunk c of the CTA tile's image, last chunk repeated
                    base = self._wsc_base(X, tile)
                    for j in range(self.sc_dtl_per_wave[X]):

                        def wsc_issue(X=X, j=j, lds0=lds0, base=base):
                            off = self.wsc_chunk[X][j]
                            lds = b.smem_ptr_add(lds0, b.zext(off, I64))
                            b.async_buffer_load_lds_addr(self.sc_rsrc[X], lds, self.sc_dtl_voff, b.add(base, off), 1)
                            self._vm_fence()

                        out.append(wsc_issue)
                    continue
                # global byte offset of the CTA's scale image of this tile (pack_scales_tile layout)
                base = b.mul(b.add(b.mul(tile, self._nqv(X)), self.sc_cta_q[X]), self.c32(512))
                if self.raw_scales:  # raw: dword 4 tile of row row0 + image row
                    base = b.add(b.mul(self.row0[X], self.c32(self.sc_nb)), b.mul(tile, self.c32(4)))
                for j in range(self.sc_dtl_per_wave[X]):

                    def sc_issue(X=X, j=j, lds0=lds0, base=base):
                        off = b.add(self.sc_dtl_w, self.c32(j * 256 * self.nw))
                        lds = b.smem_ptr_add(lds0, b.zext(off, I64))
                        # raw: the image row off / 4 starts nb bytes per row further (the lane adds l nb)
                        goff = b.add(base, b.mul(b.lshr(off, self.c32(2)), self.c32(self.sc_nb))) if self.raw_scales \
                            else b.add(base, off)
                        b.async_buffer_load_lds_addr(self.sc_rsrc[X], lds, self.sc_dtl_voff, goff, 1)
                        self._vm_fence()

                    out.append(sc_issue)
        return out

    def _fill_units(self) -> List[tuple]:
        """Every 16-byte (operand) / 4-byte (scale) per-lane unit of a stage fill, in DTL order."""
        units = []
        for X in ("A", "B"):
            for voff, soff, loff, lconst in self.dtl_list[X]:
                units.append(("op", X, voff, soff, loff, lconst))
        if self.sc_lds:
            for X in ("A", "B"):
                for j in range(self.sc_dtl_per_wave[X]):
                    units.append(("sc", X, j))
        return units

    def _reg_loads(self, tile: Value, d: int, out: Dict) -> List[Callable[[], None]]:
        """Register fill: one global->VGPR load per fill unit of 128-K tile ``tile`` into ``out[("RG", u, d)]``."""
        b = self.b
        rs: Dict[str, Value] = {}

        def rsrc(X):
            if X not in rs:
                kb = b.mul(tile, self.c32(self.tile_bytes[X]))
                rs[X] = b.buffer_rsrc(b.global_ptr_add(self.base_ptr[X], kb), self.c32(0x7FFFFFF0))
            return rs[X]

        res = []
        for u, unit in enumerate(self._fill_units()):

            def ld(u=u, unit=unit):
                if unit[0] == "op":
                    _, X, voff, soff, _, lconst = unit
                    out[("RG", u, d)] = b.buffer_load_vN(rsrc(X), voff, b.add(soff, self.c32(lconst)), I32, 4)
                else:
                    _, X, j = unit
                    base = b.mul(b.add(b.mul(tile, self.c32(self.nq[X])), self.sc_cta_q[X]), self.c32(512))
                    off = b.add(self.sc_dtl_w, self.c32(j * 256 * self.nw))
                    out[("RG", u, d)] = b.buffer_load_vN(self.sc_rsrc[X], self.sc_dtl_voff, b.add(base, off), I32, 1)
                self._vm_fence("reg")

            res.append(ld)
        return res

    def _reg_writes(self, stage: int, vals: Dict) -> List[Callable[[], None]]:
        """Register fill: ds_write every unit of a staged tile (``vals[("RG", u, 0)]``) into ``stage``."""
        b = self.b
        res = []
        for u, unit in enumerate(self._fill_units()):

            def wr(u=u, unit=unit):
                v = vals[("RG", u, 0)]
                if unit[0] == "op":
                    _, X, _, _, loff, lconst = unit
                    idx = b.add(b.lshr(b.add(loff, self.c32(lconst)), self.c32(2)), b.mul(self.lane, self.c32(4)))
                    b.smem_store_vN(self.stg[(stage, X)], [idx], v, 4)
                else:
                    _, X, j = unit
                    off = b.add(self.sc_dtl_w, self.c32(j * 256 * self.nw))
                    idx = b.add(b.lshr(off, self.c32(2)), self.lane)
                    b.smem_store_vN(self.stg[(stage, "s" + X)], [idx], b.vec_extract(v, 0), 1)

            res.append(wr)
        return res

    def _bdir_loads(self, tile: Value, d: int, out: Dict) -> List[Callable[[], None]]:
        """Direct operand: one closure per fragment of 128-K tile ``tile``, filling ``out[(X, f, 0, d)]`` (B: the
        c0c1 image; A (a_direct): the canonical row bytes 96 tile + 24 kg .. + 24 of row wrow0 + 16 f + l % 16)."""
        b = self.b
        holder: List[Value] = []
        if self.dop == "A":
            def rsrc_a():
                if not holder:
                    kb = b.mul(tile, self.c32(96))
                    holder.append(b.buffer_rsrc(b.global_ptr_add(self.base_ptr["A"], kb), self.c32(0x7FFFFFF0)))
                return holder[0]

            res = []
            for f in range(self.fm):

                def lda_(f=f):
                    r = rsrc_a()
                    soff = self.ad_soff[f]
                    v0 = b.buffer_load_vN(r, self.ad_voff, soff, I32, 4)
                    v1 = b.buffer_load_vN(r, b.add(self.ad_voff, self.c32(16)), soff, I32, 2)
                    self._vm_fence("bdir", 2)
                    parts = [b.vec_extract(v0, e) for e in range(4)] + [b.vec_extract(v1, e) for e in range(2)]
                    out[("A", f, 0, d)] = b.vec_pack(parts, I32)

                res.append(lda_)
            return res

        def rsrc():
            if not holder:
                kb = b.mul(tile, self.c32(self.tile_bytes["B"]))
                holder.append(b.buffer_rsrc(b.global_ptr_add(self.base_ptr["B"], kb), self.c32(0x7FFFFFF0)))
            return holder[0]

        res = []
        for f in range(self.fn):

            def ld(f=f):
                r = rsrc()
                # an explicit b_aux streams the weights' VGPR loads with those cache-policy bits too (-1: none, as before)
                cp = max(self.spec.b_aux, 0)
                v0 = b.buffer_load_vN(r, self.bd_voff[0], b.add(self.bd_soff[0], self.c32(1024 * f)), I32, 4, coherency=cp)
                v1 = b.buffer_load_vN(r, self.bd_voff[1], b.add(self.bd_soff[1], self.c32(512 * f)), I32, 2, coherency=cp)
                self._vm_fence("bdir", 2)
                parts = [b.vec_extract(v0, e) for e in range(4)] + [b.vec_extract(v1, e) for e in range(2)]
                out[("B", f, 0, d)] = b.vec_pack(parts, I32)

            res.append(ld)
        return res

    def _frag_load(self, stage: int, X: str, f: int, t: int, out: Dict, u8: int = 0) -> Callable[[], None]:
        """Read fragment ``(X, f, t)`` of the tile in ``stage`` (line ring: of the tile ``u`` with ``u % 8 == u8``)."""
        b, T, spec = self.b, self.T, self.spec

        def ld():
            parts = []
            if X in self.line_ops:  # line ring: three 8-byte reads (lane VGPR per wrap class + immediate)
                smem = self.ring[X]
                for h in range(3):
                    if h and spec.read_split:
                        b.sched_barrier(0)
                    v = b.smem_load_vN(smem, self._line_read_addr(X, u8, h, f, t), dtype=I32, n=2)
                    parts += [b.vec_extract(v, 0), b.vec_extract(v, 1)]
                out[(X, f, t)] = b.vec_pack(parts, I32)
                return
            if X == "A" and self.mc:  # two transposing reads of the M-contiguous stage image
                out[(X, f, t)] = self._mc_read(stage, f)
                return
            smem = self.stg[(stage, X)]
            if self.layout[X] == "canonical" and spec.canon_read == "b128b64":
                # one ds_read_b128 + one ds_read_b64 of the fragment's 24 contiguous bytes (8-byte aligned: gfx950 reads
                # unaligned b128, as rocRoller's FP6 kernels do); inline asm tagged by its allocation (lever trtag)
                lane_b = b.mul(self.frag_word[(t, X)], self.c32(4))
                addr = b.trunc(b.smem_ptr_add(b.smem_addr_of(smem), b.zext(lane_b, I64)), I32)
                off = f * T * 96
                for h, (nd, o) in enumerate(((4, 0), (2, 16))):
                    if h and spec.read_split:
                        b.sched_barrier(0)
                    tmpl = f"ds_read_b{32 * nd} $0, $1" + (f" offset:{off + o}" if off + o else "")
                    v = b.inline_asm(tmpl, "=v,v", [addr], VectorType(I32, nd), sideeffect=True)
                    parts += [b.vec_extract(v, e) for e in range(nd)]
                out[(X, f, t)] = b.vec_pack(parts, I32)
                return
            if self.layout[X] == "canonical":
                base = b.add(self.frag_word[(t, X)], self.c32(f * T * 24))
                if spec.canon_read == "b96x2":  # the 24 bytes are contiguous in the stage (rot48 moves 48-byte halves)
                    for h in range(2):
                        if h and spec.read_split:
                            b.sched_barrier(0)
                        v = b.smem_load_vN(smem, b.add(base, self.c32(3 * h)), dtype=I32, n=3)
                        parts += [b.vec_extract(v, e) for e in range(3)]
                    out[(X, f, t)] = b.vec_pack(parts, I32)
                    return
                for h in range(3):
                    if h and spec.read_split:
                        b.sched_barrier(0)
                    v = b.smem_load_vN(smem, b.add(base, self.c32(2 * h)), dtype=I32, n=2)
                    parts += [b.vec_extract(v, 0), b.vec_extract(v, 1)]
            elif self.layout[X] == "c0c1p":
                # fragments 2j, 2j+1 = three 1 KiB planes read as three b128 into 12 consecutive
                # registers: the even key loads both, the odd key is a no-op
                if f % 2:
                    return
                base = b.add(self.c0c1p_base[X], self.c32(768 * (f // 2)))  # dwords (hoisted + immediate)
                vs = []
                for p in range(3):
                    if p and spec.read_split:
                        b.sched_barrier(0)
                    v = b.smem_load_vN(smem, b.add(base, self.c32(256 * p)), dtype=I32, n=4)
                    vs += [b.vec_extract(v, e) for e in range(4)]
                # one 12-register tuple first (each load = an aligned 4-register slice of it), then the two
                # fragments as its 6-register halves: lets the allocator place the three loads contiguously
                pair = b.vec_pack(vs, I32)
                out[(X, f, t)] = b.vec_pack([b.vec_extract(pair, e) for e in range(6)], I32)
                out[(X, f + 1, t)] = b.vec_pack([b.vec_extract(pair, e) for e in range(6, 12)], I32)
                return
            else:
                rb = b.add(self.c0c1_wrb[X], self.c32(f * T // 16))
                o = b.add(b.add(b.mul(rb, self.c32(64)), self.c0c1_lane), self.c32(self.c0c1_t[t]))
                v0 = b.smem_load_vN(smem, b.mul(o, self.c32(4)), dtype=I32, n=4)
                if spec.read_split:
                    b.sched_barrier(0)
                v1 = b.smem_load_vN(smem, b.add(b.mul(o, self.c32(2)), self.c32(self.rows[X] * 16)), dtype=I32, n=2)
                parts = [b.vec_extract(v0, e) for e in range(4)] + [b.vec_extract(v1, e) for e in range(2)]
            out[(X, f, t)] = b.vec_pack(parts, I32)

        return ld

    def _frag_keys_in_order(self) -> List[Tuple[str, int, int]]:
        return sorted(self.first_use, key=lambda key: (self.first_use[key], key[0]))

    def _scale_loads(self, u: Value, out: Dict[Tuple[str, int], Value]) -> List[Callable[[], None]]:
        b = self.b
        res = []
        for X in ("A", "B"):
            for qi in range(self.sc_groups[X]):

                def ld(X=X, qi=qi):
                    if self.wtile and self.spec.scale_raw:  # raw row bytes of K tiles 4 u .. 4 u + 3
                        out[(X, qi)] = b.buffer_load_vN(self.sc_rsrc[X], self.raw_voff[X][qi], b.mul(u, self.c32(16)),
                                                        I32, 4)
                        self._vm_fence("scale")
                        return
                    if self.wtile:  # the lane's D dwords of its wave side of the CTA tile's image of tile u
                        soff = b.add(self._wsc_base(X, u), self.wsc_woff[X])
                        out[(X, qi)] = b.buffer_load_vN(self.sc_rsrc[X], self.wsc_voff[X], soff, I32,
                                                        self.wsc_D[X] * self.spec.wtile_period)
                        self._vm_fence("scale")
                        return
                    q = b.add(self.sc_q0[X], self.c32(qi))
                    w = self.sc_nd[X]  # dwords per lane
                    chunk = 512 if self.tile_scales else 1024  # bytes per (period, row chunk)
                    soff = b.mul(b.add(b.mul(u, self._nqv(X)), q), self.c32(chunk))
                    out[(X, qi)] = b.buffer_load_vN(self.sc_rsrc[X], self.sc_voff_x[X], soff, I32, w)
                    self._vm_fence("scale")

                res.append(ld)
        return res

    def _as_issues(self, per: Value, slot: int) -> List[Callable[[], None]]:
        """a_scale_src ring: one closure per DTL of the raw A scales of 4-tile period ``per`` into ring ``slot``: DTL j
        of wave w, lane l copies row 64 (w + nw j) + l's 16 bytes [16 per, 16 per + 16) to slot byte 16 x that row."""
        b = self.b
        nb = self.spec.k // 32
        base = b.add(self.as_soff0, b.mul(per, self.c32(AS_PERIOD * 4)))
        lds0 = b.smem_addr_of(self.as_lds)
        out = []
        for j in range(self.n_as):

            def issue(j=j):
                soff = b.add(base, self.c32(j * 64 * self.nw * nb)) if j else base
                off = b.add(self.as_lds_w, self.c32(slot * self.spec.tile_m * 16 + j * 1024 * self.nw))
                voff = self.as_voff if self.spec.probe not in ("as_contig", "as_ideal") else b.mul(self.lane, self.c32(16))
                b.async_buffer_load_lds_addr(self.sc_rsrc["A"], b.smem_ptr_add(lds0, b.zext(off, I64)), voff, soff, 4)
                self._vm_fence("asdtl")

            out.append(issue)
        return out

    def _sr_issues(self, X: str, per: Value, slot: int) -> List[Callable[[], None]]:
        """Scale ring of operand ``X``: one closure per DTL of period ``per`` into ring ``slot`` (raw: ``_as_issues``;
        packed: wave w copies bytes [w P img / nw, +P img / nw) of the period block, P dwords per lane)."""
        kind, P = self.srings[X][:2]
        if kind == "raw":
            return self._as_issues(per, slot)
        b = self.b
        img = self.rows[X] * 4
        base = b.add(self.sr_soff0[X], b.mul(per, self.c32(P * img)))
        lds0 = b.smem_addr_of(self.sr_lds[X])

        def issue():
            off = b.add(self.sr_lds_w[X], self.c32(slot * P * img))
            b.async_buffer_load_lds_addr(self.sc_rsrc[X], b.smem_ptr_add(lds0, b.zext(off, I64)), self.sr_voff[X],
                                         base, P)
            self._vm_fence("asdtl" if X == "A" else "bsdtl")

        return [issue]

    def _pr_read(self, X: str, u: int) -> Value:
        """Packed scale ring: the tile image of tile u (static, mod 2 P) of operand ``X``, read like the stage image."""
        b = self.b
        P = self.srings[X][1]
        slot, t = (u // P) % AS_SLOTS, u % P
        if self.spec.probe == ("as_stale" if X == "A" else "bs_stale"):  # must-fail hazard: the other slot
            slot = 1 - slot
        idx = b.add(b.add(b.mul(self.sc_wq[X], self.c32(128)), b.mul(self.lane, self.c32(2))),
                    self.c32((slot * P + t) * self.rows[X]))
        return b.smem_load_vN(self.sr_lds[X], idx, dtype=I32, n=2)

    def _as_read(self, u8: int) -> Value:
        """a_scale_src ring: the wave's tile-op_sel A scale dwords (``_sc_lds_read``'s contract) of the tile u8 mod 8,
        assembled from the raw ring: word (slot, row, tile % 4) = row's 4 scale bytes (blocks 0..3) of that tile."""
        if self.spec.probe in ("as_lite", "as_ideal", "as_none"):  # timing probe: one b64 per tile, like the tile
            # image (wrong scales)
            b = self.b
            idx = b.add(b.mul(self.lane, self.c32(2)), self.c32(((u8 // AS_PERIOD) % AS_SLOTS) * self.spec.tile_m
                                                                * AS_PERIOD + (u8 % AS_PERIOD) * 128))
            return b.smem_load_vN(self.as_lds, b.add(idx, b.mul(self.sc_wq["A"], self.c32(512))), dtype=I32, n=2)
        return self._as_assemble(self._as_read_raw(u8))

    def _as_read_raw(self, u8: int) -> List[Value]:
        """a_scale_src ring: the 4 sc_nd raw ring words of the lane for the tile u8 mod 8 (rows 64 d + 16 i)."""
        b = self.b
        slot = (u8 // AS_PERIOD) % AS_SLOTS
        if self.spec.probe == "as_stale":  # must-fail hazard: the other slot (a stale / not yet landed period)
            slot = 1 - slot
        w0 = b.add(self.as_row_w, self.c32(slot * self.spec.tile_m * AS_PERIOD + u8 % AS_PERIOD))
        return [b.vec_extract(b.smem_load_vN(self.as_lds, b.add(w0, self.c32((64 * d + 16 * i) * AS_PERIOD)),
                                             dtype=I32, n=1), 0)
                for d in range(self.sc_nd["A"]) for i in range(4)]

    def _as_assemble(self, raw: List[Value]) -> Value:
        """a_scale_src ring: dword d byte i = byte kg (= lane / 16) of raw word (d, i)."""
        b = self.b
        words = []
        for d in range(len(raw) // 4):
            r = raw[4 * d: 4 * d + 4]
            p01 = b.perm_b32(r[1], r[0], self.as_perm_sel)  # bytes: raw0[kg], raw1[kg], 0, 0
            p23 = b.perm_b32(r[3], r[2], self.as_perm_sel)
            words.append(b.perm_b32(p23, p01, self.c32(0x05040100)))  # raw0[kg], raw1[kg], raw2[kg], raw3[kg]
        return b.vec_pack(words, I32)

    def _sc_lds_read(self, stage: int, X: str, u8: Optional[int] = None) -> Value:
        """The wave's tile-op_sel scale dwords of operand ``X`` from ``stage``'s scale image (``sc_nd`` dwords:
        dword d byte i = scale of wave row 64 d + 16 i + l % 16, 32-block l / 16 of the tile). ``a_scale_src`` ring:
        A's from the raw ring (``u8`` = the read tile mod 8)."""
        b = self.b
        if X in self.srings:
            if self.srings[X][0] == "raw":
                return self._as_read(u8)
            return self._pr_read(X, u8)
        smem = self.stg[(stage, "s" + X)]
        if self.wtile:  # the lane's D dwords of its wave side
            return b.smem_load_vN(smem, self.wsc_ridx[X], dtype=I32, n=self.wsc_D[X])
        if not self.raw_scales:
            idx = b.add(b.mul(self.sc_wq[X], self.c32(128)), b.mul(self.lane, self.c32(2)))
            return b.smem_load_vN(smem, idx, dtype=I32, n=2)
        # raw image: word r = row r's 4 scale bytes (blocks 0..3 of the tile); 4 rows -> one op_sel dword
        r0 = b.add(self.wrow0[X], b.mod(self.lane, self.c32(16)))
        words = []
        for d in range(self.sc_nd[X]):
            raw = [b.smem_load_vN(smem, b.add(r0, self.c32(64 * d + 16 * i)), dtype=I32, n=1) for i in range(4)]
            raw = [b.vec_extract(v, 0) for v in raw]
            p01 = b.perm_b32(raw[1], raw[0], self.sc_perm_sel)  # bytes: raw0[kg], raw1[kg], 0, 0
            p23 = b.perm_b32(raw[3], raw[2], self.sc_perm_sel)
            words.append(b.perm_b32(p23, p01, self.c32(0x05040100)))  # raw0[kg], raw1[kg], raw2[kg], raw3[kg]
        return b.vec_pack(words, I32)

    def _scale_of(self, sc: Dict, X: str, f: int, t: int, kstep: int) -> Tuple[Value, int]:
        """(scale dword, op_sel) of MFMA tile ``f`` of operand ``X`` at K step ``t`` of the tile."""
        b = self.b
        if self.tile_scales:
            d = f // 4 if self.T == 16 else t
            if self.wtile and self.spec.scale_raw:  # transpose of the period load's word of this K tile
                src = sc[(X, d)]
                key = (id(src), kstep % 4)
                if key not in self.raw_cache:
                    self.raw_cache[key] = raw_scale_transpose(b, b.vec_extract(src, kstep % 4), *self.raw_sel)
                return self.raw_cache[key], f % 4
            if self.wtile and self.spec.wtile_period > 1:  # the K tile's D dwords inside the period load
                Pw = self.spec.wtile_period
                tp = (kstep + 1) % Pw if self.spec.mutant == "wper_rot" else kstep % Pw  # must-fail: next tile's
                d += tp * self.wsc_D[X]
            return b.vec_extract(sc[(X, 0)], d), f % 4
        return b.vec_extract(sc[(X, f // 4)], f % 4), kstep

    def _mfma(self, acc, fa, fb, sa, sb, ja, jb):
        b = self.b
        fa, fb = self._mc_resolve(fa), self._mc_resolve(fb)
        if self.spec.swap_ab:
            src0, src1, s0, s1, j0, j1 = fb, fa, sb, sa, jb, ja
        else:
            src0, src1, s0, s1, j0, j1 = fa, fb, sa, sb, ja, jb
        fmt = ELEM_FMTS[self.spec.elem_fmt]
        if self.spec.mutant == "wrong_fmt":  # decode the codes as the other 6-bit format
            fmt = "bf6" if fmt == "fp6" else "fp6"
        d = b.mfma_scale_mx(src0, src1, acc, s0, s1, shape=self.spec.atom, fmt_a=fmt, fmt_b=fmt,
                            opsel_a=j0, opsel_b=j1)
        if self.spec.acc_agpr and self.spec.acc_pin:
            d = b.inline_asm("", "=a,0", [d], d.type, sideeffect=False)
        return d

    # ------------------------------------------------------------- vmcnt model
    def _vm_events(self) -> List[Tuple[int, int, str, int]]:
        """Static VM issue order of the whole kernel: ``(tile, slot, kind, target)``.

        ``tile`` -1 is the prologue. kind "dtl" (target = K tile filled) or
        "scale" (target = scale period).
        """
        ev: List[Tuple[int, int, str, int]] = []
        n_dtl = len(self.plan.dtl_slots)
        sc_n = self.sc_groups["A"] + self.sc_groups["B"]
        S, A = self.S, self.adv
        ev += [(-1, -3, "scale", 0)] * sc_n
        for X, (_k, P, lead, _sl) in self.srings.items():  # scale ring periods due before the first tiles
            if self.sr_on[X]:
                for p_ in range(self.sr_np[X]):
                    if P * p_ - lead < 0:
                        ev += [(-1, -3, "asdtl" if X == "A" else "bsdtl", P * p_)] * self.sr_n[X]
        n_line = _n_line_dtl(self.spec)
        for t in range(max(self.ahead, self.LI if self.line_ops else 0)):
            if self.line_ops and t < self.LI and t % 4 != 3 and not (self.mini and t == 2 and self.LI == self.LA):
                ev += [(-1, -2, "ldtl", t)] * n_line
                if self.mini and t % 4 == 0 and self.spec.probe != "l2m_nomini":
                    ev += [(-1, -2, "ldtl", t + 1)] * _n_mini_dtl(self.spec)
            if t < self.ahead:
                ev += [(-1, -2, "dtl", t)] * n_dtl
        if self.spec.persist_overlap:  # the previous unit's epilogue stores (first unit: as many dummy LDS loads)
            ev += [(-1, -2, "store", 0)] * self.n_epi_stores
        if self.dop:
            for d in range(self.spec.b_depth):
                ev += [(-1, -1, "bdir", d)] * (2 * self.dn)
        nb = n_breg(self.spec)
        for t in range(3 if nb else 0):  # b_reg: B(0), B(1), B(2) after the prologue fills
            ev += [(-1, -1, "breg", t)] * nb
        if self.mini and self.LI == self.LA:  # line2m: tile 2's line, at the end of the prologue (after tile
            #   0's acquire and reads); line_merge: the loop's tile 0 issues it
            ev += [(-1, 1, "ldtl", 2)] * n_line
        n_u = self.n_tiles // self.P
        for i in range(self.n_tiles):
            items = []
            if self.clamp or i + self.ahead < self.n_tiles:
                items += [(s, "dtl", i + self.ahead) for s in self.plan.dtl_slots]
            if i % self.P == 0 and (self.clamp or i // self.P + 1 < n_u):
                items += [(self.plan.scale_slot, "scale", i // self.P + 1)] * sc_n
            if self.dop and (self.clamp or i + self.spec.b_depth < self.n_tiles):
                for j in range(self.dn):
                    items += [(self._bdir_slot(j), "bdir", i + self.spec.b_depth)] * 2
            R, LA = self.R, self.LI
            if self.line_ops and i + LA < self.n_tiles and (i + LA) % 4 != 3:
                late = ((R == 2 and LA == 2 and (i + LA) % 4 == 2 and not self.spec.line_merge)
                        or (self.mini and self.spec.probe != "l2m_early" and not self.spec.line_merge))
                sl = line_slots(self.spec, late)
                head = self.mini and (i + LA) % 4 == 0 and self.spec.probe != "l2m_nomini"
                if self.spec.line_release_b >= 0:  # per operand: its lines, then its remnant DTLs (_tile's order)
                    plan_l = line_slot_plan(self.spec, late)
                    for X in self.line_ops:
                        slx = [s_ for s_, Y in plan_l if Y == X]
                        items += [(s_, "ldtl", i + LA) for s_ in slx]
                        if head:
                            items += [(slx[-1], "ldtl", i + LA + 1)] * _n_mini_dtl(self.spec, X)
                else:
                    items += [(s_, "ldtl", i + LA) for s_, _X in line_slot_plan(self.spec, late)]
                    if head:  # the remnant
                        items += [(sl[-1], "ldtl", i + LA + 1)] * _n_mini_dtl(self.spec)
            for X, (_k, P, lead, sl_) in self.srings.items():
                if self.sr_on[X] and (i + lead) % P == 0 and (i + lead) // P < self.sr_np[X]:
                    items += [(sl_, "asdtl" if X == "A" else "bsdtl", i + lead)] * self.sr_n[X]
            if nb and i + 3 < self.n_tiles:
                items += [(s_, "breg", i + 3) for s_ in breg_slots(self.spec)]
            # same-slot issue order of _tile: stage DTLs, line DTLs, raw A scale DTLs, B register loads, scale loads,
            # direct-B loads
            items.sort(key=lambda x: (x[0], {"dtl": 0, "ldtl": 1, "asdtl": 2, "bsdtl": 3, "breg": 4, "scale": 5,
                                             "bdir": 6}[x[1]]))
            ev += [(i, s, kd, tg) for s, kd, tg in items]
        return ev

    @property
    def ahead(self) -> int:
        """K tiles between a tile and the tile its refill DTLs fetch (depth_u: the stage refilled during a barrier
        period was last read before the period's acquire: S - KS + adv)."""
        return self.S if self.spec.release_slot >= 0 else self.S - self.KS + self.adv

    def _bdir_slot(self, j: int) -> int:
        return min(self.plan.n_mfma - 1, self.spec.b_slot + j * self.spec.b_every)

    def _acquire_counts(self) -> Dict[int, int]:
        """vmcnt for the acquire of tile ``j`` (at tile ``j - adv``'s acquire slot; prologue for j < adv)."""
        if self.spec.reg_stages:  # ds_write fill: the acquire is lgkmcnt(0) + barrier, no vmcnt
            return {j: -1 for j in range(self.n_tiles)}
        ev = self._vm_events()
        last_dtl: Dict[int, int] = {}
        for idx, e in enumerate(ev):
            if e[2] in ("dtl", "ldtl", "asdtl", "bsdtl"):  # a/bsdtl: target = the first tile of its scale period
                last_dtl[e[3]] = idx
        # tile j's data = every fill with a target <= j (line ring: tile j may bring no new line of its own)
        run = -1
        for j in range(self.n_tiles):
            run = max(run, last_dtl.get(j, -1))
            last_dtl[j] = run
        # b_reg: the acquire of tile j precedes the ds_write of B(j + 1) -> its loads must have landed too
        last_br: Dict[int, int] = {}
        for idx, e in enumerate(ev):
            if e[2] == "breg":
                last_br[e[3]] = idx
        run = -1
        for j in range(self.n_tiles + 1):
            run = max(run, last_br.get(j, -1))
            last_br[j] = run
        if any(e[2] == "breg" for e in ev):
            # prologue: the acquire of tile 0 precedes the ds_write of B(0), B(1)
            last_dtl[0] = max(last_dtl[0], last_br.get(1, -1))
            # tile i's ds_write of B(i + 2) at its last MFMA slot: every event issued before it, minus those after
            # B(i + 2)'s last load; one (strictest) count for every tile
            waits = []
            idx = sum(1 for e in ev if e[0] == -1)
            n_last = self.plan.n_mfma - 1
            for i in range(self.n_tiles - 2):
                while idx < len(ev) and (ev[idx][0] < i or (ev[idx][0] == i and ev[idx][1] < n_last)):
                    idx += 1
                waits.append(idx - 1 - last_br[i + 2])
            if waits and min(waits) < 0:
                raise AssertionError("b_reg: a B ds_write precedes its loads")
            self.breg_waits = waits  # per tile; the loop body takes the strictest of each position (_mainloop)
        # position of the acquire in tile i = index of the first event at/after its slot
        first_at: Dict[int, int] = {}
        n_pro = sum(1 for e in ev if e[0] == -1)
        n_pre = sum(1 for e in ev if e[0] == -1 and e[1] < 0)  # issued before the prologue's acquire of tile 0
        idx = n_pro
        acq_slot = self.spec.carry_slot if self.spec.carry else self.plan.acq
        for i in range(self.n_tiles):
            while idx < len(ev) and ev[idx][0] == i and ev[idx][1] < acq_slot:
                idx += 1
            first_at[i] = idx
            while idx < len(ev) and ev[idx][0] == i:
                idx += 1
        counts: Dict[int, int] = {}
        for j in range(self.n_tiles):
            pos = n_pre if j - self.acq_off < 0 else first_at[j - self.acq_off]
            cover = j
            if self.KS > 1 and j - self.acq_off >= 0:
                if (j - self.acq_off) % self.KS:  # no acquire in this tile (depth_u: one per KS tiles)
                    counts[j] = 0
                    continue
                cover = min(self.n_tiles - 1, j + self.KS - 1)  # every tile read until the next acquire
            counts[j] = pos - 1 - last_dtl[cover]
            if counts[j] < 0:
                raise AssertionError(f"tile {j} acquired before its DTLs were issued")
            if self.lib and counts[j] > 63:  # the 6-bit vmcnt field: waiting for more than needed is safe
                counts[j] = 63
        return counts

    # ------------------------------------------------------------- one K tile
    def _tile(self, i_dyn: Value, i_static: Optional[int], pos: int, acc: List,
              frags_in: Dict, scales_cur: Dict, scales_next: Dict, vm_wait: int) -> Tuple[List, Dict]:
        """Emit K tile ``i`` (``pos`` = its static index mod the loop body length).

        ``frags_in``: this tile's fragments read in the previous tile (prefetch).
        Returns (acc, fragments for the next tile). Fills ``scales_next`` when this
        tile issues the next period's scale loads.
        """
        b, spec, pl = self.b, self.spec, self.plan
        self._trace_tile = i_static if i_static is not None else ("loop", pos)
        lead = spec.read_lead + getattr(self, "_lead_extra", 0)
        S, P, A = self.S, self.P, self.adv
        T_total = self.n_tiles
        rd_stage = (pos + A) % S  # stage holding the tile whose fragments are read now
        dtl_stage = (pos + self.ahead) % S  # stage refilled during this tile
        has_read = i_static is None or i_static + A < T_total or self.clamp
        has_dtl = i_static is None or i_static + self.ahead < T_total or self.clamp
        if spec.mutant == "k_order" and pos == 1:
            rd_stage = (rd_stage + 1) % S  # read the fragments from the wrong stage
        pre: Dict[int, List[Callable[[], None]]] = {}
        read_now: Dict = {}

        def acquire():
            b.s_waitcnt(vmcnt=vm_wait, lgkmcnt=0)
            b.s_barrier_bare()

        carry_next: Dict = {}
        if spec.carry:
            # this tile's scales and first fragments were read in the previous tile (or the prologue)
            scales_cur = dict(scales_cur)
            for X in ("A", "B"):
                scales_cur[(X, 0)] = frags_in[("S", X, 0)]
            if i_static is None or i_static + 1 < T_total:
                nxt_stage = (pos + 1) % S

                def carry_acquire():  # every wave's DTLs into the next stage landed; this stage's reads done
                    b.s_waitcnt(vmcnt=vm_wait, lgkmcnt=0)
                    b.s_barrier_bare()
                    for X in ("A", "B"):
                        carry_next[("S", X, 0)] = self._sc_lds_read(nxt_stage, X)
                    for key in self._frag_keys_in_order()[:spec.carry]:
                        self._frag_load(nxt_stage, *key, carry_next)()

                pre.setdefault(spec.carry_slot, []).append(carry_acquire)
        elif (has_read or has_dtl) and pos % self.KS == 0:  # depth_u: one acquire per barrier period
            pre.setdefault(pl.acq, []).append(acquire)
        if self.mini and (pos + 1) % 4 == 0 and (i_static is None or i_static + 1 < T_total)                 and spec.probe != "l2m_nocopy" and spec.mini_src == "copy":
            # tile 4m + 4's head line landed (this acquire) and is overwritten after this tile's release: read it
            # right after the acquire, write the remnant ``mini_copy_gap`` MFMA slots later (before the release)
            rs = pl.acq if spec.mini_copy_at < 0 else spec.mini_copy_at
            groups = [(rs, self.line_ops)]
            if spec.mini_copy_at_b >= 0:  # B's copy at its own slot (before B's release)
                groups = [(rs, tuple(X for X in self.line_ops if X != "B")), (spec.mini_copy_at_b, ("B",))]
            if spec.probe == "l2m_nocopyB":  # timing probe: B's remnant never copied
                groups = [(r_, tuple(X for X in o_ if X != "B")) for r_, o_ in groups]
                groups = [(r_, o_) for r_, o_ in groups if o_]
            for rs_, ops_ in groups:
                cp: Dict = {}
                ws = min(pl.n_mfma - 1, rs_ + spec.mini_copy_gap)
                if spec.probe == "l2m_copywrite":  # timing / PMC probe: the copy's writes only (constant data)
                    pre.setdefault(ws, []).append(lambda cp=cp, ops_=ops_: self._mini_copy_write(
                        self._mini_copy_dummy(ops_), ops_))
                    continue
                pre.setdefault(rs_, []).append(
                    lambda u8n=(pos + 1) % self.LP, cp=cp, ops_=ops_: self._mini_copy_read(u8n // 4, cp, ops_))
                if spec.probe == "l2m_copyread":  # timing / PMC probe: the copy's reads only (values kept live)
                    pre.setdefault(ws, []).append(lambda cp=cp: [b.inline_asm("", "v", [v], None, sideeffect=True)
                                                                for v in cp.values()])
                    continue
                if spec.probe == "l2m_copyvm0":  # probe: the writes with no DTL in flight (a correct, slower GEMM)
                    pre.setdefault(ws, []).append(lambda: b.s_waitcnt(vmcnt=0))
                pre.setdefault(ws, []).append(lambda cp=cp, ops_=ops_: self._mini_copy_write(cp, ops_))
        br_next: Dict = {}
        if spec.b_reg:
            if i_static is None or i_static + 2 < T_total:  # B(i + 2) -> stage (i + 2) % 2 = this tile's (read)

                wv = self.breg_waits[i_static] if i_static is not None else self.breg_wait_pos[pos]

                def bw(st=(pos + 2) % S, wv=wv):  # at the tile's end: the loads (previous tile) had a tile to land
                    b.s_waitcnt(vmcnt=wv)
                    self._breg_writes(st, frags_in)

                pre.setdefault(pl.n_mfma - 1, []).append(bw)
        if self.sc_lds and A == 1:  # prefetch: this tile's scales were read with its fragments (previous tile)
            scales_cur = dict(scales_cur)
            for X in ("A", "B"):
                scales_cur[(X, 0)] = frags_in[("S", X, 0)]
        if self.sc_lds and has_read and not spec.carry:
            sc_stage = rd_stage
            sc_out = read_now if A == 1 else scales_cur
            sc_key = (lambda X: ("S", X, 0)) if A == 1 else (lambda X: (X, 0))

            u8_rd = pos + A  # the read tile (mod the loop body: every ring reduces it mod 2 P)
            split = self.as_ring and spec.as_perm >= 0 and spec.probe not in ("as_lite", "as_ideal",
                                                                                           "as_none")  # raw ring words read now,
            # assembled at slot as_perm
            raw_a: List = []

            def sc_read():
                for X in ("A", "B"):
                    if X == "A" and split:
                        raw_a.extend(self._as_read_raw(u8_rd))
                        continue
                    sc_out[sc_key(X)] = self._sc_lds_read(sc_stage, X, u8=u8_rd)

            pre.setdefault(pl.acq, []).append(sc_read)
            if split:
                pre.setdefault(max(pl.acq, spec.as_perm), []).append(
                    lambda: sc_out.__setitem__(sc_key("A"), self._as_assemble(raw_a)))
        if spec.probe == "dtl_only":
            has_read = False
        if spec.probe == "lds_only":
            has_dtl = False
        bq_next: Dict = {}
        bdir_issue: List = []
        bdir_now: Dict = {}
        if self.dop:
            D = spec.b_depth
            for key, v in frags_in.items():
                if len(key) != 4:  # prefetch: this tile's LDS (A) fragments
                    continue
                X, f, t, d = key
                if d == 0:
                    (bdir_now if A == 1 else read_now)[(X, f, t)] = v
                else:
                    bq_next[(X, f, t, d - 1)] = v
            if i_static is None or i_static + D < T_total or self.clamp:
                if i_static is not None:
                    tgt_b = self._ti(min(i_static + D, T_total - 1))  # runtime K: the static tail is shifted
                elif self.clamp:  # past K: re-load the last tile (in range; never consumed)
                    tgt_b = b.smin(b.add(i_dyn, self.c32(D)), self.c32(T_total - 1))
                else:
                    tgt_b = b.add(i_dyn, self.c32(D))
                # registered after the DTLs / scale loads below: within one MFMA slot the VM issue order is
                # dtl, scale, bdir (= the static vmcnt model's order)
                bdir_issue = list(enumerate(self._bdir_loads(tgt_b, D - 1, bq_next)))
        if has_read:
            keys = self._frag_keys_in_order()
            if spec.line_reads_first:  # the line-ring fragments first (stable): the release / refills follow them
                keys = sorted(keys, key=lambda key: key[0] not in self.line_ops)
            if spec.line_release_b >= 0:  # A's fragments, then B's (each operand's release follows its reads)
                keys = sorted(keys, key=lambda key: key[0] != "A")
            if self.dop:  # the direct operand has no LDS image
                keys = [key for key in keys if key[0] != self.dop]
            if spec.carry:
                keys = keys[spec.carry:]
            rd_u8 = (pos + A) % self.LP  # line ring: the read tile mod LP (static: the loop body is a multiple)
            if A == 1:
                for key, s_ in zip(keys, pl.read_slots):
                    pre.setdefault(s_, []).append(self._frag_load(rd_stage, *key, read_now, u8=rd_u8))
            else:
                for key in keys:
                    s_ = max(pl.acq, self.first_use[key] - lead)
                    if spec.release_slot >= 0:
                        s_ = min(s_, spec.release_slot)  # every read of this stage before its release
                    pre.setdefault(s_, []).append(self._frag_load(rd_stage, *key, read_now, u8=rd_u8))
        if has_dtl and spec.release_slot >= 0:

            def release():  # every wave finished reading this tile's stage: it may be refilled
                b.s_waitcnt(lgkmcnt=0)
                b.s_barrier_bare()

            pre.setdefault(spec.release_slot, []).append(release)
        if has_dtl:
            if i_static is not None:
                tgt = self._ti(min(i_static + self.ahead, T_total - 1))
            elif self.clamp:  # refills past K re-load the last tile (in range; its stage is never read)
                tgt = b.smin(b.add(i_dyn, self.c32(self.ahead)), self.c32(T_total - 1))
            else:
                tgt = b.add(i_dyn, self.c32(self.ahead))
            if spec.reg_stages:
                for f, s_ in zip(self._reg_writes(dtl_stage, frags_in), pl.dtl_slots):
                    pre.setdefault(s_, []).append(f)
            else:
                slots = list(pl.dtl_slots)
                if self.sc_lds and spec.scale_dtl == "x4w":
                    slots = [slots[0]] * 2 + slots  # the two scale DTLs lead the refill
                if self.sc_lds and spec.scale_dtl == "split":
                    slots = [slots[0]] * len(self.wsc_split) + slots  # the split scale chunks lead the refill
                issues = self._dtl_issues(tgt, dtl_stage)
                if self.lib and len(issues) != len(slots):
                    raise AssertionError(f"{len(issues)} refill DTLs vs {len(slots)} planned slots (vmcnt model)")
                for f, s_ in zip(issues, slots):
                    pre.setdefault(s_, []).append(f)
        R, LA = self.R, self.LI
        if self.line_ops and (i_static is None or i_static + LA < T_total):
            lu8 = (pos + LA) % self.LP  # the refilled tile (LA ahead) mod LP
            if lu8 % 4 != 3:  # tiles 4m + 3 bring no new line
                # line2: the new line of tile 4m + 2 overwrites the slot of a line the next tile still reads
                # (line_merge: two ahead after this tile's acquire -- no release)
                late = not spec.line_merge and ((R == 2 and LA == 2 and lu8 % 4 == 2)
                                                or (self.mini and spec.probe != "l2m_early"))

                if late and spec.probe != "l2m_norelease":

                    def line_release():  # every wave finished reading the next tile's fragments
                        b.s_waitcnt(lgkmcnt=0)
                        b.s_barrier_bare()

                    pre.setdefault(spec.line_release, []).append(line_release)
                    if spec.line_release_b >= 0 and spec.probe != "l2m_earlyB":  # B's own release, after B's reads
                        pre.setdefault(spec.line_release_b, []).append(line_release)
                tgt_l = self._ti(i_static + LA) if i_static is not None else b.add(i_dyn, self.c32(LA))
                if spec.line_release_b >= 0:
                    plan_l = line_slot_plan(spec, late)
                    for X in self.line_ops:
                        sl = [s_ for s_, Y in plan_l if Y == X]
                        if X == "B" and spec.probe == "l2m_earlyB":  # timing probe: B refilled right after acquire
                            sl = line_slots(spec, False, "B") if spec.line_early else sl
                            sl = [min(pl.n_mfma - 1, pl.acq + 1 + j * max(1, spec.line_spread)) for j in range(len(sl))]
                        sl = sl + [sl[-1]] * _n_mini_dtl(spec, X)  # X's remnant DTLs after its last line DTL
                        for f, s_ in zip(self._line_issues(tgt_l, lu8, (X,)), sl):
                            pre.setdefault(s_, []).append(f)
                else:
                    sl = line_slots(spec, late)
                    sl = sl + [sl[-1]] * _n_mini_dtl(spec)  # the remnant DTLs follow the head line's last DTL
                    for f, s_ in zip(self._line_issues(tgt_l, lu8), sl):
                        pre.setdefault(s_, []).append(f)
        for X, (_k, P, lead, sl_) in self.srings.items():
            if not (self.sr_on[X] and (pos + lead) % P == 0 and (
                    i_static is None or (i_static + lead) // P < self.sr_np[X])):
                continue
            # X's scales of the period starting lead tiles ahead, into its ring slot (free: its previous period's last
            # read preceded this tile's acquire, or an earlier one)
            if i_static is not None:
                per = self.c32((i_static + lead) // P)
            else:
                per = b.div(b.add(i_dyn, self.c32(lead)), self.c32(P))
            slot = ((pos + lead) // P) % AS_SLOTS  # static: the loop body is a multiple of 2 P tiles
            if spec.probe == ("as_hazard" if X == "A" else "bs_hazard"):  # must-fail: into the slot still being read
                slot = 1 - slot
            for f in self._sr_issues(X, per, slot):
                pre.setdefault(sl_, []).append(f)
        if spec.b_reg and (i_static is None or i_static + 3 < T_total):
            tgt_b = self.c32(i_static + 3) if i_static is not None else b.add(i_dyn, self.c32(3))
            for f, s_ in zip(self._breg_loads(tgt_b, br_next), breg_slots(spec)):
                pre.setdefault(s_, []).append(f)
        rg_next: Dict = {}
        if spec.reg_stages:
            R = spec.reg_stages
            for key, v in frags_in.items():
                if key[0] == "RG" and key[2] > 0:
                    rg_next[("RG", key[1], key[2] - 1)] = v
            if i_static is None or i_static + self.ahead + R < T_total:
                tgt_r = b.add(i_dyn, self.c32(self.ahead + R))
                for j, f in enumerate(self._reg_loads(tgt_r, R - 1, rg_next)):
                    pre.setdefault(min(pl.n_mfma - 1, spec.reg_slot + j * spec.reg_every), []).append(f)
        n_u = T_total // P
        issue_scales = (pos % P == 0) and (i_static is None or i_static // P + 1 < n_u or self.clamp)
        issue_scales = issue_scales and not self.sc_lds
        if issue_scales:
            if i_static is not None:
                u_next = self._ti(min(i_static // P + 1, n_u - 1), period=True)
            elif self.clamp:
                u_next = b.smin(b.add(b.div(i_dyn, self.c32(P)), self.c32(1)), self.c32(n_u - 1))
            else:
                u_next = b.add(b.div(i_dyn, self.c32(P)), self.c32(1))
            for f in self._scale_loads(u_next, scales_next):
                pre.setdefault(pl.scale_slot, []).append(f)
        for j, f_ in bdir_issue:
            pre.setdefault(self._bdir_slot(j), []).append(f_)
        frags = frags_in if A == 1 else read_now
        if A == 1 and self.dop:  # prefetched LDS fragments + this tile's direct fragments
            frags = {**{k: v for k, v in frags_in.items() if len(k) == 3}, **bdir_now}
        if spec.carry:  # read_now fills as the jit reads issue: seed it with the carried fragments
            read_now.update(frags_in)
            frags = read_now
        for f in pre.get(-1, ()):
            f()
        if spec.setprio:
            b.sched_barrier(0)
            b.s_setprio(spec.setprio)
        acc = list(acc)
        drop = spec.mutant == "drop_kstep" and i_static is not None and i_static == T_total - 1
        mfma_on = spec.probe not in ("fill_only", "dtl_only", "lds_only")
        for x, (t, mi, ni) in enumerate(self.order):
            a_idx = mi * self.fn + ni
            kstep = (pos % P) * self.ksteps + t  # K step within the scale period
            if not mfma_on:
                if x == 0:
                    for v in scales_cur.values():
                        b.inline_asm("", "v", [v], None, sideeffect=True)
                for key in (("A", mi, t), ("B", ni, t)):
                    if self.first_use[key] == x and key in frags:
                        b.inline_asm("", "v", [frags[key]], None, sideeffect=True)
            if mfma_on and not drop:
                sa, ja = self._scale_of(scales_cur, "A", mi, t, kstep)
                sb, jb = self._scale_of(scales_cur, "B", ni, t, kstep)
                if spec.mutant == "scale_sel":
                    ja = 0 if ja == 1 else ja
                    jb = 0 if jb == 1 else jb
                c_in = acc[a_idx]
                if c_in is None:
                    c_in = b.zero_vec_f32(self.geo.c_len)
                acc[a_idx] = self._mfma(c_in, frags[("A", mi, t)], frags[("B", ni, t)], sa, sb, ja, jb)
            for f in pre.get(x, ()):
                f()
            if spec.pin_mfma:
                b.sched_barrier(0)
        if spec.setprio:
            b.s_setprio(0)
            b.sched_barrier(0)
        if spec.tile_lgkm0:
            b.s_waitcnt(lgkmcnt=0)
            b.sched_barrier(0)
        if spec.carry:
            return acc, carry_next
        if A == 1:  # prefetch: fragments not re-read here (dtl_only probe, last tile) pass through unchanged
            if self.dop:  # the direct queue advances (bq_next), stale direct keys are dropped
                return acc, {**{k: v for k, v in frags_in.items() if len(k) == 3}, **read_now, **bq_next}
            return acc, {**frags_in, **read_now, **br_next}
        return acc, ({**bq_next, **rg_next} if spec.reg_stages else bq_next)

    # ------------------------------------------------------------- main loop
    def _prologue(self, between: Optional[Callable[[], None]] = None) -> Tuple[Dict, Dict, Dict[int, int]]:
        b, S, A = self.b, self.S, self.adv
        counts = self._acquire_counts()
        sc_cur: Dict = {}
        for f in self._scale_loads(self.c32(0), sc_cur):
            f()
        for X, (_k, P, lead, _sl) in self.srings.items():  # scale ring periods due before the loop issues them
            if self.sr_on[X]:
                for p_ in range(self.sr_np[X]):
                    if P * p_ - lead < 0:
                        for f in self._sr_issues(X, self.c32(p_), p_ % AS_SLOTS):
                            f()
        for t in range(max(self.ahead, self.LI if self.line_ops else 0)):
            # line2m: tile 2's line goes into tile 0's head-line slot -> issued after tile 0 is read (below)
            if self.line_ops and t < self.LI and not (self.mini and t == 2 and self.LI == self.LA):
                for f in self._line_issues(self.c32(t), t % self.LP):
                    f()
            if t < self.ahead:
                for f in self._dtl_issues(self.c32(t), t % S):
                    f()
        if between is not None:  # persist_overlap: the previous unit's epilogue while the fills fly
            between()
        fr: Dict = {}
        b0: Dict = {}
        b1: Dict = {}
        if self.spec.b_reg:  # tiles 0, 1 straight into the 2 stages after the acquire; tile 2 carried
            for f in self._breg_loads(self.c32(0), b0) + self._breg_loads(self.c32(1), b1) \
                    + self._breg_loads(self.c32(2), fr):
                f()
        if self.spec.reg_stages:
            b.s_waitcnt(vmcnt=0)  # the DTL-filled first tiles landed (the loop's acquire waits on LDS only)
            for d in range(self.spec.reg_stages):
                for f in self._reg_loads(self.c32(self.ahead + d), d, fr):
                    f()
        if self.dop:
            for d in range(self.spec.b_depth):
                for f in self._bdir_loads(self.c32(d), d, fr):
                    f()
        if A == 1:
            b.s_waitcnt(vmcnt=counts[0], lgkmcnt=0)
            if self.spec.b_reg:
                self._breg_writes(0, b0)
                self._breg_writes(1, b1)
                b.s_waitcnt(lgkmcnt=0)
            b.s_barrier_bare()
            if self.sc_lds:
                for X in ("A", "B"):
                    fr[("S", X, 0)] = self._sc_lds_read(0, X, u8=0)
            for key in self._frag_keys_in_order():
                if self.dop and key[0] == self.dop:  # direct operand: loaded above, no LDS image
                    continue
                self._frag_load(0, *key, fr, u8=0)()
            if self.mini:  # every wave read tile 0: its head line's slot may take tile 2's line (model: tile 0)
                if self.spec.mini_src == "copy":
                    self._mini_copy(0)  # group 0's remnant (line 0, slot 0); dtl: fetched with line 0
                b.s_waitcnt(lgkmcnt=0)
                b.s_barrier_bare()
                if self.LI == self.LA:
                    for f in self._line_issues(self.c32(2), 2):
                        f()
        if self.spec.carry:
            b.s_waitcnt(vmcnt=counts[0], lgkmcnt=0)
            b.s_barrier_bare()
            for X in ("A", "B"):
                fr[("S", X, 0)] = self._sc_lds_read(0, X)
            for key in self._frag_keys_in_order()[:self.spec.carry]:
                self._frag_load(0, *key, fr)()
        return sc_cur, fr, counts

    def _mainloop(self, between: Optional[Callable[[], None]] = None) -> None:
        b, spec = self.b, self.spec
        S, P, A = self.S, self.P, self.adv
        U = S * P // _gcd(S, P) * spec.unroll
        T = self.n_tiles
        sc_cur, fr, counts = self._prologue(between)
        if spec.probe == "stamps":  # realtime after the prologue (fill issued; prefetch: first tile landed and read)
            self.t_loop = b.inline_asm("s_memrealtime $0\n\ts_waitcnt lgkmcnt(0)", "=s", [], I64)
        n_acc = self.fm * self.fn
        if spec.probe in ("fill_only", "dtl_only", "lds_only"):
            acc: List = [b.zero_vec_f32(self.geo.c_len)] * n_acc
            H = T % U if self.clamp else 0
        elif self.clamp and (T % U or spec.peel_first):
            acc = [None] * n_acc  # the static head's first tile takes srcC = 0
            H = T % U or U
        elif spec.peel_first:
            acc = [None] * n_acc
            H = P
        else:
            zero = b.zero_vec_f32(self.geo.c_len)
            n_ag = 256 // self.geo.c_len if spec.acc_vgpr_spill else n_acc  # atoms held in AGPRs
            acc = [b.inline_asm("", "=a,0", [zero], zero.type, sideeffect=False)
                   if spec.acc_agpr and i_ < n_ag else zero for i_ in range(n_acc)]
            H = 0
        pending = None

        def static_tile(i, acc, fr, sc_cur, pending, counts=counts):
            sc_next: Dict = {}
            acc, fr = self._tile(self._ti(i), i, i % U, acc, fr, sc_cur, sc_next, counts.get(i + self.acq_off, 0))
            if sc_next:
                pending = sc_next
            if (i + 1) % P == 0 and pending is not None:
                sc_cur, pending = pending, None
            return acc, fr, sc_cur, pending

        for i in range(H):
            acc, fr, sc_cur, pending = static_tile(i, acc, fr, sc_cur, pending)
        # main loop: L iterations of U tiles from tile H; every tile in it has its reads and refills
        L = (T - H) // U if self.clamp else max(0, (T - H - S - A) // U)
        while L > 0 and any((i + ld) % P_ == 0 and (i + ld) // P_ >= self.sr_np[X]
                            for X, (_k, P_, ld, _s) in self.srings.items() for i in range(H + (L - 1) * U, H + L * U)):
            L -= 1  # every loop tile issues its scale ring DTL (no period past K): those tiles go to the static tail
        if spec.b_reg:  # b_reg ds_write waits: the strictest of the loop's tiles at each position
            self.breg_wait_pos = {p: min([self.breg_waits[H + j * U + p] for j in range(L)
                                          if H + j * U + p < len(self.breg_waits)] or [0]) for p in range(U)}
        # the loop body uses one vmcnt per position: the strictest (smallest) of its
        # iterations (only the first iteration can differ, by the prologue's order)
        body_wait = {}
        for p in range(U):
            vals = [counts[H + j * U + p + self.acq_off] for j in range(L) if H + j * U + p + self.acq_off < T]
            body_wait[p] = min(vals) if vals else 0
        if L > 0:
            if pending is not None:
                raise AssertionError("loop must start on a scale-period boundary")
            if any(v is None for v in acc):
                raise AssertionError("loop-carried accumulators need a value")
            frag_keys = sorted(fr.keys())
            sc_keys = sorted(sc_cur.keys())
            copies = [("", self.plan, 0, self.c32(L))]
            if self.rt:  # the runtime K's extra tiles are (K / 128 - k / 128) / U more iterations of the same body
                extra = b.div(b.sub(self.k_tiles, self.c32(T)), self.c32(U))
                copies = [("", self.plan, 0, b.to_sgpr_u32(b.add(extra, self.c32(L))))]
            if spec.probe == "xcd_trim":
                on = b.cmp_eq(b.mod(b.block_id_x(), self.c32(NUM_XCDS)), self.c32(spec.trim_xcd))
                copies = [("", self.plan, 0, b.select(on, self.c32(max(0, L - spec.trim_iters)), self.c32(L)))]
            if spec.skew_dtl or spec.skew_read:
                # odd waves run the skewed copy, even waves the plain one (the other's trip count is 0)
                odd = b.cmp_eq(b.land(self.wave, self.c32(1)), self.c32(1))
                alt = plan(spec)
                alt.dtl_slots = [min(alt.n_mfma - 1, x + spec.skew_dtl) for x in alt.dtl_slots]
                copies = [("", self.plan, 0, b.select(odd, self.c32(0), self.c32(L))),
                          ("s", alt, spec.skew_read, b.select(odd, self.c32(L), self.c32(0)))]
            for tag, pl_c, lead_x, trips in copies:
                base_plan = self.plan
                self.plan, self._lead_extra = pl_c, lead_x
                if pl_c is not base_plan:  # the skewed copy's VM order may differ: its own vmcnt model
                    counts_c = self._acquire_counts()
                    for k_, v_ in counts_c.items():  # the static tail follows either copy: strictest count
                        counts[k_] = min(counts[k_], v_)
                    bw = {p: min([counts_c[H + j * U + p + A] for j in range(L) if H + j * U + p + A < T] or [0])
                          for p in range(U)}
                else:
                    bw = body_wait
                acc, fr, sc_cur = self._emit_loop(tag, trips, U, H, P, A, acc, fr, sc_cur, frag_keys, sc_keys, bw)
                self.plan, self._lead_extra = base_plan, 0
        if self.rt and spec.rt_tail == "switch":
            # runtime K = k + (U j + r) 128: the loop ran j more iterations; the static tail of the residue r (one
            # uniform branch per r, each built and vmcnt-modelled for K / 128 = k / 128 + r, then the epilogue)
            rres = b.to_sgpr_u32(b.mod(b.sub(self.k_tiles, self.c32(T)), self.c32(U)))
            for rr in range(U):
                with b.scf_if(b.cmp_eq(rres, self.c32(rr))):
                    self.n_tiles = T + rr
                    counts_r = self._acquire_counts()
                    self.tshift = b.to_sgpr_u32(b.sub(self.k_tiles, self.c32(T + rr)))
                    self.tshift_u = self.tshift if P == 1 else b.to_sgpr_u32(b.div(self.tshift, self.c32(P)))
                    st = (acc, fr, sc_cur, pending)
                    for i in range(H + L * U, T + rr):
                        st = static_tile(i, *st, counts=counts_r)
                    self.acc = st[0]
                    self._epilogue()
                    self.tshift = self.tshift_u = None
                    self.n_tiles = T
            self.epilogue_done = True
            return
        if self.rt:  # the static tail's K tiles move by the runtime K's extra tiles (a multiple of U)
            self.tshift = b.to_sgpr_u32(b.sub(self.k_tiles, self.c32(T)))
            self.tshift_u = self.tshift if P == 1 else b.to_sgpr_u32(b.div(self.tshift, self.c32(P)))
        for i in range(H + L * U, T):
            acc, fr, sc_cur, pending = static_tile(i, acc, fr, sc_cur, pending)
        self.tshift = self.tshift_u = None
        self.acc = acc

    def _emit_loop(self, tag, trips, U, H, P, A, acc, fr, sc_cur, frag_keys, sc_keys, body_wait):
        """One K-loop copy of ``trips`` iterations of U tiles from tile H."""
        b = self.b
        iter_args = [(f"acc{tag}{i}", v) for i, v in enumerate(acc)]
        iter_args += [("fr" + tag + "_" + "_".join(str(x) for x in key), self._mc_resolve(fr[key])) for key in frag_keys]
        iter_args += [(f"sc{tag}_{X}{q}", sc_cur[(X, q)]) for (X, q) in sc_keys]
        na, nf = len(acc), len(frag_keys)
        if True:
            loop = b.scf_for_iter(self.c32(0), trips, self.c32(1), iter_args, iv_name="kt" + tag,
                                  elide_trailing_barrier=False)
            with loop as (iv, vs):
                acc_l = list(vs[:na])
                fr_l = {key: vs[na + i] for i, key in enumerate(frag_keys)}
                sc_l = {key: vs[na + nf + i] for i, key in enumerate(sc_keys)}
                i0 = b.add(b.mul(iv, self.c32(U)), self.c32(H))
                pend = None
                for p in range(U):
                    i_dyn = b.add(i0, self.c32(p)) if p else i0
                    sc_next: Dict = {}
                    acc_l, fr_l = self._tile(i_dyn, None, (H + p) % U, acc_l, fr_l, sc_l, sc_next,
                                             body_wait[p])
                    if sc_next:
                        pend = sc_next
                    if (H + p + 1) % P == 0:
                        if pend is None and not self.sc_lds:
                            raise AssertionError("scale period without a prefetch")
                        if pend is not None:
                            sc_l, pend = pend, None
                b.scf_yield(*(acc_l + [self._mc_resolve(fr_l[key]) for key in frag_keys] + [sc_l[key] for key in sc_keys]))
            res = loop.results
            acc = list(res[:na])
            fr = {key: res[na + i] for i, key in enumerate(frag_keys)}
            sc_cur = {key: res[na + nf + i] for i, key in enumerate(sc_keys)}
        return acc, fr, sc_cur

    def _mfma_only(self) -> None:
        """Probe: the tile's MFMA stream on register-resident tile-0 operands and period-0 scales."""
        b, spec = self.b, self.spec
        S = self.S
        sc: Dict = {}
        for f in self._scale_loads(self.c32(0), sc):
            f()
        for f in self._dtl_issues(self.c32(0), 0):
            f()
        b.s_waitcnt(vmcnt=0, lgkmcnt=0)
        b.s_barrier_bare()
        fr: Dict = {}
        for key in self._frag_keys_in_order():
            self._frag_load(0, *key, fr)()
        zero = b.zero_vec_f32(self.geo.c_len)
        acc = [b.inline_asm("", "=a,0", [zero], zero.type, sideeffect=False) for _ in range(self.fm * self.fn)]
        U = 4 // self.ksteps * spec.unroll  # one scale period per body (all op_sel bytes)
        L = self.n_tiles // U
        lo = self.c32(0)
        trips = b.to_sgpr_u32(b.div(self.k_tiles, self.c32(U))) if self.rt else self.c32(L)  # runtime K: K / 128 / U
        loop = b.scf_for_iter(lo, trips, self.c32(1), [(f"acc{i}", v) for i, v in enumerate(acc)],
                              iv_name="kt", elide_trailing_barrier=False)
        with loop as (iv, vs):
            acc_l = list(vs)
            for p in range(U):
                for t, mi, ni in self.order:
                    a_idx = mi * self.fn + ni
                    kstep = (p % (4 // self.ksteps)) * self.ksteps + t
                    sa, ja = self._scale_of(sc, "A", mi, t, kstep)
                    sb, jb = self._scale_of(sc, "B", ni, t, kstep)
                    acc_l[a_idx] = self._mfma(acc_l[a_idx], fr[("A", mi, t)], fr[("B", ni, t)], sa, sb, ja, jb)
                    if spec.pin_mfma:
                        b.sched_barrier(0)
            b.scf_yield(*acc_l)
        self.acc = list(loop.results)
        del S

    # ------------------------------------------------------------- epilogue
    def _pack_acc(self) -> List[Value]:
        """persist_overlap: the accumulators (out_exp2 applied) as bf16 pairs, one i32x2 per atom (the epilogue's
        first step for swap_ab False)."""
        b, g = self.b, self.geo
        out = []
        for v in self.acc:
            v = v if v is not None else b.zero_vec_f32(4)
            if self.spec.out_exp2:
                f = b.const_f32(2.0 ** self.spec.out_exp2)
                v = b.vec_pack([b.fmul(b.vec_extract(v, e), f) for e in range(g.c_len)], F32)
            out.append(b.vec_bitcast(b.vec_trunc_f32_to_bf16(v), VectorType(I32, 2)))
        return out

    def _epilogue(self, wait: bool = True, packed: Optional[List[Value]] = None) -> None:
        b, spec, g = self.b, self.spec, self.geo
        if wait:
            b.s_waitcnt(vmcnt=0)
        lane, T = self.lane, self.T
        if spec.schedule == "persistent":
            # the unit loop would hoist the epilogue's lane-only address math above the K loop (LICM), keeping it
            # live through the loop's register peak: an opaque copy of the lane id pins it here
            lane = b.inline_asm("v_mov_b32 $0, $1", "=v,v", [lane], I32, sideeffect=True)
        if spec.out_exp2 and packed is None:  # exact power-of-two rescale of every accumulator (host scale bias D = -out_exp2)
            f = b.const_f32(2.0 ** spec.out_exp2)
            self.acc = [None if v is None else
                        b.vec_pack([b.fmul(b.vec_extract(v, e), f) for e in range(g.c_len)], F32) for v in self.acc]
        rr = b.mod(lane, self.c32(T))
        grp = b.div(lane, self.c32(T))
        n_bytes = spec.n * 2
        rsrc = b.buffer_rsrc(self.pc, self.c32(0x7FFFFFF0))
        m0 = b.add(self.row0["A"], self.wrow0["A"])
        n0 = b.add(self.row0["B"], self.wrow0["B"])
        if not spec.swap_ab:  # A is src0: a lane holds C[m0 + mi T + 4 grp + (0..3)][n0 + ni T + rr]
            if g.c_len != 4:
                raise NotImplementedError("swap_ab=False epilogue for the 16x16 atom only")
            if spec.epi_lds:
                self._epilogue_lds(lane, rr, grp, rsrc, m0, n0, packed)
                return
            for mi in range(self.fm):
                soff = b.to_sgpr_u32(b.add(b.mul(b.add(m0, self.c32(mi * T)), self._cnb()),
                                           b.mul(n0, self.c32(2))))
                # lanes rr, rr ^ 1 hold adjacent columns: swap halves with a DPP xor-1 so the even lane stores
                # rows 0-1 and the odd lane rows 2-3, each as (even column, odd column) bf16 pairs = one dword
                par = b.land(rr, self.c32(1))
                odd = b.cmp_eq(par, self.c32(1))
                row0 = b.add(b.mul(grp, self.c32(4)), b.mul(par, self.c32(2)))
                col_b = b.mul(b.sub(rr, par), self.c32(2))
                lo_m, hi_m = self.c32(0xFFFF), self.c32(-65536)
                for ni in range(self.fn):
                    if packed is not None:  # persist_overlap: converted at the end of the unit
                        h = packed[mi * self.fn + ni]
                    else:
                        v = self.acc[mi * self.fn + ni]
                        h = b.vec_bitcast(b.vec_trunc_f32_to_bf16(v if v is not None else b.zero_vec_f32(4)),
                                          VectorType(I32, 2))
                    d0, d1 = b.vec_extract(h, 0), b.vec_extract(h, 1)
                    x0, x1 = b.dpp_xor(d0, 1), b.dpp_xor(d1, 1)  # the partner lane's rows 0-1 / 2-3
                    lo = b.select(odd, x1, d0)
                    hi = b.select(odd, d1, x0)
                    words = (b.lor(b.land(lo, lo_m), b.shl(hi, self.c32(16))),
                             b.lor(b.lshr(lo, self.c32(16)), b.land(hi, hi_m)))
                    for j, w in enumerate(words):
                        off = b.add(b.add(b.mul(b.add(row0, self.c32(j)), self._cnb()), col_b),
                                    self.c32(ni * T * 2))
                        b.buffer_store_vN(rsrc, off, soff, b.vec_pack([w], I32))
                        if packed is not None and not wait:  # persist_overlap: modelled VM ops
                            self._vm_fence("store")
            return
        # lane holds C[m0 + mi*T + rr][n0 + ni*T + 4*grp + 8*q4 + (0..3)]
        lane_off = b.add(b.mul(rr, self._cnb()), b.mul(grp, self.c32(8)))
        nq4 = g.c_len // 4
        zero = b.zero_vec_f32(4)
        for mi in range(self.fm):
            soff = b.to_sgpr_u32(b.add(b.mul(b.add(m0, self.c32(mi * T)), self._cnb()), b.mul(n0, self.c32(2))))
            for ni in range(self.fn):
                v = self.acc[mi * self.fn + ni]
                for q4 in range(nq4):
                    if v is None:
                        four = zero
                    elif nq4 > 1:
                        four = b.vec_pack([b.vec_extract(v, 4 * q4 + e) for e in range(4)], F32)
                    else:
                        four = v
                    h = b.vec_trunc_f32_to_bf16(four)
                    off = b.add(lane_off, self.c32(ni * T * 2 + q4 * 16))
                    if spec.store_nt:  # streaming C stores (slc)
                        b.buffer_store_vN_bf16(rsrc, off, soff, h, 2, cachepolicy=2)
                    else:
                        b.buffer_store_vN(rsrc, off, soff, b.vec_bitcast(h, VectorType(I32, 2)))

    def _epilogue_lds(self, lane, rr, grp, rsrc, m0, n0, packed) -> None:
        """epi_lds: per 16-row block mi of the wave tile, the bf16 pairs of the dword epilogue go to a per-wave LDS
        block (row pitch 2 fn T bytes + 16), and come back as 16-byte row pieces stored with buffer_store_dwordx4."""
        b, spec, T = self.b, self.spec, self.T
        pitch = self.fn * T * 2 // 4 + 4  # words per staged row (+16 B: rows start on different banks)
        cpr = self.fn * T * 2 // 16  # 16-byte pieces per row
        rows_rd = 64 // cpr  # rows per ds_read_b128 / store instruction
        region = 16 * pitch
        arr, words = self.epi_buf
        if words < self.nw * region:
            raise ValueError(f"epi_lds: no LDS allocation holds {self.nw} x {region * 4} bytes")
        base = b.to_sgpr_u32(b.mul(self.wave, self.c32(region)))
        par = b.land(rr, self.c32(1))
        odd = b.cmp_eq(par, self.c32(1))
        row0 = b.add(b.mul(grp, self.c32(4)), b.mul(par, self.c32(2)))
        colw = b.div(b.sub(rr, par), self.c32(2))  # word column of the pair (even column / 2)
        wr = [b.add(b.add(b.mul(b.add(row0, self.c32(j)), self.c32(pitch)), colw), base) for j in range(2)]
        lr, lc = b.div(lane, self.c32(cpr)), b.mod(lane, self.c32(cpr))
        rd = b.add(b.add(b.mul(lr, self.c32(pitch)), b.mul(lc, self.c32(4))), base)
        gofs = b.add(b.mul(lr, self._cnb()), b.mul(lc, self.c32(16)))
        lo_m, hi_m = self.c32(0xFFFF), self.c32(-65536)
        b.s_waitcnt(lgkmcnt=0)
        b.s_barrier_bare()  # every wave is past its last fragment read of the ring
        for mi in range(self.fm):
            soff = b.to_sgpr_u32(b.add(b.mul(b.add(m0, self.c32(mi * T)), self._cnb()), b.mul(n0, self.c32(2))))
            for ni in range(self.fn):
                if packed is not None:
                    h = packed[mi * self.fn + ni]
                else:
                    v = self.acc[mi * self.fn + ni]
                    h = b.vec_bitcast(b.vec_trunc_f32_to_bf16(v if v is not None else b.zero_vec_f32(4)),
                                      VectorType(I32, 2))
                d0, d1 = b.vec_extract(h, 0), b.vec_extract(h, 1)
                x0, x1 = b.dpp_xor(d0, 1), b.dpp_xor(d1, 1)
                lo = b.select(odd, x1, d0)
                hi = b.select(odd, d1, x0)
                wds = (b.lor(b.land(lo, lo_m), b.shl(hi, self.c32(16))),
                       b.lor(b.lshr(lo, self.c32(16)), b.land(hi, hi_m)))
                for j in range(2):
                    b.smem_store_vN(arr, [b.add(wr[j], self.c32(ni * T // 2))], wds[j], 1)
            b.s_waitcnt(lgkmcnt=0)  # the block's pairs are in LDS before any lane reads a row piece
            for q in range(16 // rows_rd):
                v4 = b.smem_load_vN(arr, b.add(rd, self.c32(q * rows_rd * pitch)), dtype=I32, n=4)
                off = b.add(gofs, b.mul(self.c32(q * rows_rd), self._cnb()))
                if spec.store_nt:
                    b.buffer_store_vN_bf16(rsrc, off, soff, b.vec_bitcast(v4, VectorType(BF16, 8)), 4,
                                           cachepolicy=2)
                else:
                    b.buffer_store_vN(rsrc, off, soff, v4)
            b.s_waitcnt(lgkmcnt=0)  # the reads are done before the next block overwrites the region
        if spec.schedule == "persistent":
            b.s_barrier_bare()  # the next unit's ring fills come after every wave's last read of the region

    def _stamps(self) -> None:
        """Probe: wave 0 lane 0 writes (start, end) realtime stamps over the first 8 bf16 of its CTA's C tile."""
        b = self.b
        t_end = b.inline_asm("s_waitcnt vmcnt(0)\n\ts_memrealtime $0\n\ts_waitcnt lgkmcnt(0)", "=s", [], I64)
        c_end = b.inline_asm("s_memtime $0\n\ts_waitcnt lgkmcnt(0)", "=s", [], I64)

        def w4(x, y):
            vx, vy = b.bitcast(x, VectorType(I32, 2)), b.bitcast(y, VectorType(I32, 2))
            return b.vec_pack([b.vec_extract(vx, 0), b.vec_extract(vx, 1), b.vec_extract(vy, 0), b.vec_extract(vy, 1)],
                              I32)

        words, cwords = w4(self.t_start, t_end), w4(self.c_start, c_end)
        pwords = w4(getattr(self, "t_loop", self.t_start), getattr(self, "t_epi", t_end))
        rsrc = b.buffer_rsrc(self.pc, self.c32(0x7FFFFFF0))
        soff = b.to_sgpr_u32(b.add(b.mul(self.row0["A"], self.c32(self.spec.n * 2)), b.mul(self.row0["B"], self.c32(2))))
        first = b.land(b.cmp_eq(self.wave, self.c32(0)), b.cmp_eq(self.lane, self.c32(0)))
        with b.scf_if(first):  # bf16 slots 0..7: realtime (start, end); 8..15: shader clock counter (start, end)
            b.buffer_store_vN(rsrc, self.c32(0), soff, words)
            b.buffer_store_vN(rsrc, self.c32(16), soff, cwords)
            b.buffer_store_vN(rsrc, self.c32(32), soff, pwords)  # bf16 slots 16..23: (after prologue, before epilogue)


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a

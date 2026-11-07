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

import hashlib
import json
from dataclasses import asdict, dataclass, fields, replace
from typing import Callable, Dict, List, Optional, Tuple

from rocke.core.ir import BF16, F32, I8, I32, I64, IRBuilder, KernelDef, PtrType, Value, VectorType

from .layout import ROW_BYTES, SWIZZLES, UNITS, atom_geometry

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
#: Element formats (problem contract, not a tuning knob) -> MFMA operand format name (cbsz / blgp 2 / 3).
ELEM_FMTS = {"e2m3": "fp6", "e3m2": "bf6"}
#: ``required_compiler`` values: "any" (LLVM or the rocKE direct compiler) | "rust" (direct compiler only).
COMPILERS = ("any", "rust")
XCD_BLOCKS = ("none", "4x8", "8x4", "2x16", "16x2")
FRAG_MODES = ("prefetch", "jit")
OPERAND_LAYOUTS = ("canonical", "c0c1", "c0c1p")
PROBES = ("none", "mfma_only", "fill_only", "dtl_only", "lds_only", "stamps", "xcd_trim")
SCALE_LAYOUTS = ("kstep", "tile")
TAIL_MODES = ("static", "clamp")
SCALE_SRCS = ("vgpr", "lds")
NUM_XCDS = 8
DTL_BYTES = 16  # dwordx4 direct-to-LDS unit
#: sched_barrier mask letting every class but VMEM cross (ALU, VALU, SALU, MFMA, DS, transcendental)
VM_FENCE_MASK = 0x0001 | 0x0002 | 0x0004 | 0x0008 | 0x0080 | 0x0100 | 0x0200 | 0x0400
ROWS_PER_GROUP = 32  # canonical: 32 rows x 6 units = 3 DTL instructions of 64 lanes


@dataclass(frozen=True)
class Mxfp6GemmSpec:
    """One build. Every field has a default (the problem shape included)."""

    m: int = 4096
    n: int = 4096
    k: int = 65536
    elem_fmt: str = "e2m3"  # OCP MX 6-bit element format of A and B | "e3m2" (problem contract, not a knob)
    required_compiler: str = "any"  # | "rust": structures the LLVM build spills on (see PRESETS / check_compiler)
    # ---- geometry ----
    atom: str = "16x16x128"  # | "32x32x64"
    tile_m: int = 256
    tile_n: int = 256
    waves_m: int = 2
    waves_n: int = 2
    stages: int = 3  # LDS ring depth (128-K tiles)
    swizzle: str = "rot48"  # canonical layout: LDS row rotation | "none"
    operand_layout: str = "c0c1"  # host pre-tiled stage images | "canonical" (dense stream)
    a_layout: str = ""  # A's layout when it differs ("canonical": live activations next to pre-tiled weights)
    scale_layout: str = "tile"  # op_sel = MFMA tile, per-tile loads | "kstep" (op_sel = K step, 4 tiles per load)
    scale_src: str = "lds"  # tile scales DTL'd into the ring stage, ds_read | "vgpr" (VGPR loads one tile ahead)
    # ---- MFMA ----
    swap_ab: bool = True  # B fragment is src0: a lane holds 4 consecutive C columns
    out_exp2: int = 0  # epilogue multiplies C by 2^out_exp2 before bf16 RNE (exact; pairs with host scale biasing)
    acc_agpr: bool = True  # accumulators in AGPRs (+ the AGPR MFMA form)
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
    #   1 KiB dwordx4 DTL per operand (wave 0: A, wave nw/2: B), issued first in the tile's refill
    k_rot: int = 0  # fabric probe: XCD x starts its K loop at tile x * k_rot * T / 8 (mod T) -- 1 de-aligns the
    #   eight XCDs along K (cross-XCD panel reuse in MALL), 0 = all XCDs walk K in step (default)
    mma_order: str = "rows"  # MFMA order in a tile: "rows" (m_outer / serpentine) | "pairs" (two A rows
    #   interleaved: each B fragment feeds two MFMAs back to back) | "blk2" (2x2 blocks, serpentine)
    rsrc_once: bool = False  # pre-tiled operands: one buffer descriptor per operand (built once); the K tile's
    #   byte offset rides in the DTL soffset (no per-tile descriptor rebuild in the loop)
    dtl_imm: bool = False  # c0c1: blocks of four 1 KiB DTLs share M0 / soffset via the instruction offset
    dtl_aux: int = 0  # cache-policy bits of the operand refill DTLs (0 default, 1 glc/sc0, 2 slc/nt, 3 both)
    b_direct: bool = False  # B fragments straight from the c0c1 global image into VGPRs (no LDS for B)
    b_depth: int = 1  # K tiles the direct B loads run ahead (VGPR fragment sets in flight = b_depth + 1)
    b_slot: int = 0  # MFMA slot of a tile's first direct B fragment load
    b_every: int = 2  # MFMA slots between direct B fragment loads
    tail_mode: str = "static"  # | "clamp": the loop runs to the last tile, refills past K re-load the last tile
    # ---- grid ----
    xcd_block: str = "4x8"  # CTA tile block per XCD (rows x cols of tiles) | "none"
    # ---- epilogue ----
    store_nt: bool = False  # non-temporal C stores (buffer store slc)
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
    "operand_layout": ("c0c1", "canonical", "c0c1p"),
    "a_layout": ("", "canonical"),
    "scale_layout": ("tile", "kstep"),
    "scale_src": ("lds", "vgpr"),
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
    "unroll": (1, 2, 3, 4),
    "reg_stages": (0, 1, 2),
    "reg_slot": (0, 8, 16),
    "reg_every": (4, 2, 3),
    "release_slot": (-1, 8, 16, 24, 32),
    "skew_dtl": (0, 1, 2, 3, 4),
    "skew_read": (0, 1, 2, 4),
    "carry": (0, 1, 2, 4, 9),
    "carry_slot": (56, 48, 52, 60),
    "scale_dtl": ("dword", "x4w"),
    "k_rot": (0, 1, 2),
    "swap_ab": (True, False),
    "store_nt": (False, True),
    "mma_order": ("rows", "pairs", "blk2"),
    "rsrc_once": (False, True),
    "dtl_imm": (False, True),
    "dtl_aux": (0, 1, 2, 3),
    "b_direct": (False, True),
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
    for name, legal in (("swizzle", SWIZZLES), ("xcd_block", XCD_BLOCKS), ("mutant", MUTANTS),
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
    if spec.k % 512:
        return False, "K must be a multiple of 512 (four 128-K tiles per scale period)"
    if spec.tile_m % spec.waves_m or spec.tile_n % spec.waves_n:
        return False, "tile must split evenly over the waves"
    wt_m, wt_n = spec.tile_m // spec.waves_m, spec.tile_n // spec.waves_n
    if wt_m % (4 * geo.rows) or wt_n % (4 * geo.rows):
        return False, f"wave tile must be a multiple of {4 * geo.rows} (scale groups of 4 MFMA tiles)"
    if spec.a_layout not in ("", "canonical", "c0c1"):
        return False, "a_layout must be '' (= operand_layout), canonical or c0c1"
    if spec.a_layout and spec.a_layout != spec.operand_layout and spec.operand_layout != "c0c1":
        return False, "a mixed layout needs B in c0c1"
    for rows, X in ((spec.tile_m, "A"), (spec.tile_n, "B")):
        if operand_layout_of(spec, X) == "canonical":
            if rows % ROWS_PER_GROUP or (rows // ROWS_PER_GROUP) % nw:
                return False, "tile rows must give every wave whole 32-row DTL groups"
        elif (rows * ROW_BYTES // 1024) % nw:
            return False, "c0c1: a stage image must split into whole 1 KiB DTL instructions per wave"
    if not 2 <= spec.stages <= 4:
        return False, "stages must be 2..4"
    if lds_bytes(spec) > 160 * 1024:
        return False, f"LDS {lds_bytes(spec)} B exceeds 160 KiB"
    if spec.acc_agpr and acc_regs(spec) > 256:
        return False, "accumulators exceed 256 AGPRs"
    if spec.k // 128 < spec.stages + 2:
        return False, "K too small for the ring"
    if spec.read_every < 1 or spec.read_delay < 0 or spec.dtl_delay < 0 or spec.read_lead < 1:
        return False, "bad read / DTL slot knobs"
    if spec.unroll not in (1, 2, 3, 4):
        return False, "unroll must be 1..4"
    n_mfma = mfmas_per_tile(spec)
    if not -1 <= spec.acq_slot < n_mfma - 1:
        return False, "acq_slot out of range"
    if spec.frag_mode == "jit" and spec.acq_slot != -1:
        return False, "jit reads start at the top of the tile: acq_slot must be -1"
    if not 0 <= spec.setprio <= 3:
        return False, "setprio must be 0..3"
    if spec.xcd_block != "none":
        bm, bn = (int(x) for x in spec.xcd_block.split("x"))
        gm, gn = spec.m // spec.tile_m, spec.n // spec.tile_n
        if gm % bm or gn % bn or (gm // bm) * (gn // bn) != NUM_XCDS:
            return False, f"xcd_block {spec.xcd_block} does not tile the {gm}x{gn} CTA grid into {NUM_XCDS}"
    if spec.scale_layout == "tile":
        legal = (64, 128) if geo.rows == 16 else (128,)
        if wt_m not in legal or wt_n not in legal:
            return False, f"scale_layout tile needs wave tiles of {legal} rows"
        if spec.scale_src == "lds" and (wt_m != 128 or wt_n != 128):
            return False, "scale_src lds needs 128-row wave tiles (use scale_src vgpr)"
    if spec.scale_src == "lds" and spec.scale_layout != "tile":
        return False, "scale_src lds needs scale_layout tile"
    if spec.scale_src == "lds" and any(r * 4 % (256 * nw) for r in (spec.tile_m, spec.tile_n)):
        return False, "scale_src lds: whole 256-byte scale DTLs per wave and operand"
    if spec.scale_src == "lds" and spec.probe == "mfma_only":
        return False, "the mfma_only probe loads its scales to VGPRs (scale_src vgpr)"
    if spec.scale_dtl not in ("dword", "x4w"):
        return False, "scale_dtl must be dword or x4w"
    if spec.scale_dtl == "x4w" and (spec.scale_src != "lds" or nw != 4 or spec.tile_m * 4 != 1024
                                    or spec.tile_n * 4 != 1024 or spec.reg_stages or spec.b_direct):
        return False, "scale_dtl x4w needs LDS scales, 4 waves, 256-row tiles (1 KiB per operand), no reg / direct-B"
    if spec.k_rot not in (0, 1, 2) or (spec.k_rot and (spec.scale_src != "lds" or spec.b_direct or spec.reg_stages
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
    if spec.b_direct and (spec.operand_layout != "c0c1" or spec.frag_mode != "jit" or geo.rows != 16
                          or spec.probe != "none"):
        return False, "b_direct needs c0c1 operands, jit fragments, the 16x16x128 atom and no probe"
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
    if not 1 <= spec.b_depth <= 3 or spec.b_slot < 0 or spec.b_every < 1:
        return False, "bad direct-B knobs"
    if spec.tail_mode == "clamp":
        geo_p = 1 if spec.scale_layout == "tile" else 4 // (128 // geo.k)
        u_body = spec.stages * geo_p // _gcd(spec.stages, geo_p) * spec.unroll
        if (spec.k // 128) % u_body % geo_p:
            return False, "clamp: the static head (K tiles mod loop body) must be whole scale periods"
    if spec.probe != "none" and spec.mutant != "none":
        return False, "probe and mutant are exclusive"
    return True, ""


def operand_layout_of(spec: Mxfp6GemmSpec, X: str) -> str:
    """Device layout of operand ``X`` ("A" / "B"): ``a_layout`` overrides ``operand_layout`` for A."""
    return spec.a_layout if (X == "A" and spec.a_layout) else spec.operand_layout


def acc_regs(spec: Mxfp6GemmSpec) -> int:
    return (spec.tile_m // spec.waves_m) * (spec.tile_n // spec.waves_n) // 64


def lds_bytes(spec: Mxfp6GemmSpec) -> int:
    sc = (spec.tile_m + spec.tile_n) * 4 if spec.scale_src == "lds" else 0
    ops = spec.tile_m + (0 if spec.b_direct else spec.tile_n)
    return spec.stages * (ops * ROW_BYTES + sc)


def mfmas_per_tile(spec: Mxfp6GemmSpec) -> int:
    geo = atom_geometry(spec.atom)
    fm = spec.tile_m // spec.waves_m // geo.rows
    fn = spec.tile_n // spec.waves_n // geo.rows
    return fm * fn * (128 // geo.k)


def grid_of(spec: Mxfp6GemmSpec) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """``((grid_x, 1, 1), (block_x, 1, 1))``."""
    nwg = (spec.m // spec.tile_m) * (spec.n // spec.tile_n)
    return (nwg, 1, 1), (64 * spec.waves_m * spec.waves_n, 1, 1)


def tile_of_wg(spec: Mxfp6GemmSpec, wg: int) -> Tuple[int, int]:
    """Host model of the CTA -> (tile row, tile column) map (``xcd_block``)."""
    gm, gn = spec.m // spec.tile_m, spec.n // spec.tile_n
    if spec.xcd_block == "none":
        return wg // gn, wg % gn
    bm, bn = (int(x) for x in spec.xcd_block.split("x"))
    nbn = gn // bn
    x, loc = wg % NUM_XCDS, wg // NUM_XCDS
    return (x // nbn) * bm + loc // bn, (x % nbn) * bn + loc % bn


def kernel_name(spec: Mxfp6GemmSpec) -> str:
    """Full format contract + geometry + a digest of the non-default knobs."""
    d = _defaults()
    diff = {k: v for k, v in asdict(spec).items()
            if k not in ("m", "n", "k", "elem_fmt", "required_compiler", "a_layout") and v != d[k]}
    f = spec.elem_fmt
    base = (
        f"rocke_gemm_mx_{f}_{f}_e8m0b32_f32acc_bf16_rcr_"
        f"{spec.tile_m}x{spec.tile_n}x128_w{spec.waves_m}x{spec.waves_n}_{spec.atom}_s{spec.stages}"
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
    }
    a_s = abt[operand_layout_of(spec, "A")].replace("{f}", spec.elem_fmt)
    ab = abt[operand_layout_of(spec, "B")].replace("{f}", spec.elem_fmt)
    packer = {"tile": "layout.pack_scales_tile", "kstep": "layout.pack_scales"}[spec.scale_layout]
    scales = (f"e8m0 per 32 K, uint8 [rows, K/32] packed by {packer}(scales, atom) "
              f"(= layout.pack_scales_for(scales, atom, '{spec.scale_layout}')); scale_layout={spec.scale_layout}, "
              f"read via {'the LDS ring' if spec.scale_src == 'lds' else 'VGPR loads'} (scale_src={spec.scale_src})")
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
        "params": ["a", "b", "sa", "sb", "c"],
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


def _n_dtl(spec: Mxfp6GemmSpec) -> int:
    nw = spec.waves_m * spec.waves_n
    n = (spec.tile_m + (0 if spec.b_direct else spec.tile_n)) * UNITS // (64 * nw)
    if spec.scale_src == "lds" and spec.scale_dtl == "dword":  # per operand: rows * 4 B, 256 B per dword DTL
        n += (spec.tile_m + spec.tile_n) * 4 // 256 // nw
    return n


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
}


def preset(name: str, **over) -> Mxfp6GemmSpec:
    """``Mxfp6GemmSpec`` of preset ``name`` (+ overrides, e.g. the shape or ``elem_fmt``)."""
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r} (expected one of {tuple(PRESETS)})")
    return replace(Mxfp6GemmSpec(), **{**PRESETS[name], **over})


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
    return _Builder(spec).build()


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
        self.tile_scales = spec.scale_layout == "tile"
        self.clamp = spec.tail_mode == "clamp"
        self.sc_lds = spec.scale_src == "lds"
        self.P = 1 if self.tile_scales else 4 // self.ksteps
        self.plan = plan(spec)
        self.order = mfma_order(spec)
        self.sc_groups = {"A": 1, "B": 1} if self.tile_scales else {"A": self.fm // 4, "B": self.fn // 4}
        if spec.scale_src == "lds":
            self.sc_groups = {"A": 0, "B": 0}  # no VGPR scale loads: the scales ride the stage DTLs
        self.rows = {"A": spec.tile_m, "B": spec.tile_n}
        # first use (MFMA index in the tile) of every fragment key (X, f, t)
        self.first_use: Dict[Tuple[str, int, int], int] = {}
        for x, (t, mi, ni) in enumerate(self.order):
            self.first_use.setdefault(("A", mi, t), x)
            self.first_use.setdefault(("B", ni, t), x)

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
        if spec.lds_alias_scopes:
            k.attrs["lds_alias_scopes"] = True
        if spec.acc_agpr:
            k.attrs["agpr_alloc"] = (acc_regs(spec), acc_regs(spec))
            k.attrs["mfma_agpr_form"] = True
        if self.nw > 4:
            k.attrs["waves_per_eu"] = self.nw // 4  # two (or more) waves per SIMD: budget the registers
        self.pa = b.param("a", PtrType(I8, "global"))
        self.pb = b.param("b", PtrType(I8, "global"))
        self.psa = b.param("sa", PtrType(I32, "global"))
        self.psb = b.param("sb", PtrType(I32, "global"))
        self.pc = b.param("c", PtrType(BF16, "global"))
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
        self._epilogue()
        if spec.probe == "stamps":
            self._stamps()
        b.ret()
        return k

    # ------------------------------------------------------------- setup
    def _setup(self) -> None:
        b, spec, g = self.b, self.spec, self.geo
        tid = b.thread_id_x()
        self.lane = b.lane_id()
        wave = b.to_sgpr_u32(b.lshr(tid, self.c32(6)))
        self.wave = wave
        wm = b.div(wave, self.c32(spec.waves_n))
        wn = b.mod(wave, self.c32(spec.waves_n))
        # CTA tile (XCD-aware order)
        wg = b.block_id_x()
        gm, gn = spec.m // spec.tile_m, spec.n // spec.tile_n
        if spec.xcd_block == "none":
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
            tm = b.add(b.mul(b.div(x, self.c32(nbn)), self.c32(bm)), b.div(loc, self.c32(bn)))
            tn = b.add(b.mul(b.mod(x, self.c32(nbn)), self.c32(bn)), b.mod(loc, self.c32(bn)))
        self.tile_idx = {"A": tm, "B": tn}
        self.row0 = {"A": b.to_sgpr_u32(b.mul(tm, self.c32(spec.tile_m))),
                     "B": b.to_sgpr_u32(b.mul(tn, self.c32(spec.tile_n)))}
        # wave sub-tile origin (rows of A / rows of B within the CTA tile)
        self.wrow0 = {"A": b.to_sgpr_u32(b.mul(wm, self.c32(self.wt_m))),
                      "B": b.to_sgpr_u32(b.mul(wn, self.c32(self.wt_n)))}
        # LDS ring: one allocation per (stage, operand) so LLVM's alias scopes
        # keep refill DMAs and fragment reads of different stages apart
        self.stg = {}
        for s in range(self.S):
            for X in ("A", "B"):
                if not (X == "B" and spec.b_direct):
                    self.stg[(s, X)] = b.smem_alloc(I32, [self.rows[X] * ROW_BYTES // 4], name_hint=f"stg{X}{s}")
                if spec.scale_src == "lds":  # rows * 4 bytes: the tile's scale image of this operand
                    self.stg[(s, "s" + X)] = b.smem_alloc(I32, [self.rows[X]], name_hint=f"sst{X}{s}")
        lane = self.lane
        self.layout = {X: operand_layout_of(spec, X) for X in ("A", "B")}
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
        # --- scale addressing -------------------------------------------------
        T = self.T
        grp_rows = 128 if self.tile_scales else 4 * T
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
            if spec.scale_dtl == "x4w":
                self.sc4_voff = b.mul(lane, self.c32(16))
            self.sc_dtl_w = b.to_sgpr_u32(b.mul(wave, self.c32(256)))
            self.sc_dtl_per_wave = {X: self.rows[X] * 4 // 256 // self.nw for X in ("A", "B")}
            # first 128-row scale chunk of the CTA (global) and of the wave (local)
            self.sc_cta_q = {X: b.to_sgpr_u32(b.div(self.row0[X], self.c32(128))) for X in ("A", "B")}
            self.sc_wq = {X: b.div(self.wrow0[X], self.c32(128)) for X in ("A", "B")}

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
        for X in ("A", "B"):
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
        self.base_ptr = {X: b.global_ptr_add(p, b.mul(self.row0[X], self.c32(self.row_bytes)))
                         for X, p in (("A", self.pa), ("B", self.pb))}
        self.tile_bytes = {"A": 96, "B": 96}  # global bytes per 128-K tile step
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

    def _setup_c0c1(self, wave: Value, lane: Value) -> None:
        b, spec, g = self.b, self.spec, self.geo
        # DTL: the stage image of an operand is one contiguous block; lane l of the
        # wave's instruction q copies bytes [1 KiB (wave * per_wave + q) + 16 l, +16)
        self.dtl_voff = [b.mul(lane, self.c32(16))]
        self.dtl_list = {}
        for X in ("A", "B"):
            per_wave = self.rows[X] * ROW_BYTES // 1024 // self.nw
            base = b.to_sgpr_u32(b.mul(wave, self.c32(per_wave * 1024)))
            lst = []
            for q in range(per_wave):
                lst.append((self.dtl_voff[0], base, base, q * 1024))
            self.dtl_list[X] = [] if (X == "B" and spec.b_direct) else lst
        nt = self.n_tiles
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
            kbyte = b.mul(tile, self.c32(self.tile_bytes[X]))
            holder: List[Value] = []

            def rsrc(X=X, holder=holder, kbyte=kbyte):
                if self.spec.rsrc_once and self.layout[X] != "canonical":
                    return self.op_rsrc[X]
                if not holder:
                    holder.append(b.buffer_rsrc(b.global_ptr_add(self.base_ptr[X], kbyte), self.c32(0x7FFFFFF0)))
                return holder[0]

            canon = self.layout[X] == "canonical"
            kadd = kbyte if (self.spec.rsrc_once and not canon) else None  # the tile offset joins the soffset

            lds0 = b.smem_addr_of(self.stg[(stage, X)])
            for voff, soff, loff, lconst in self.dtl_list[X]:

                def issue(voff=voff, soff=soff, loff=loff, lconst=lconst, rsrc=rsrc, lds0=lds0, kadd=kadd,
                          canon=canon):
                    sbase = soff if kadd is None else b.add(soff, kadd)
                    if self.spec.dtl_imm and not canon:
                        blk, imm = (lconst // 4096) * 4096, lconst % 4096
                        lds = b.smem_ptr_add(lds0, b.zext(b.add(loff, self.c32(blk)), I64))
                        b.async_buffer_load_lds_imm(rsrc(), lds, voff, b.add(sbase, self.c32(blk)), 4, imm,
                                                    coherency=self.spec.dtl_aux)
                        self._vm_fence()
                        return
                    lds = b.smem_ptr_add(lds0, b.zext(b.add(loff, self.c32(lconst)), I64))
                    if not canon:
                        # the instruction's 1 KiB block offset rides in soffset (global side only)
                        b.async_buffer_load_lds_addr(rsrc(), lds, voff, b.add(sbase, self.c32(lconst)), 4,
                                                     coherency=self.spec.dtl_aux)
                    else:
                        b.async_buffer_load_lds_addr(rsrc(), lds, voff, soff, 4, coherency=self.spec.dtl_aux)
                    self._vm_fence()

                out.append(issue)
        if self.sc_lds and self.spec.scale_dtl == "dword":
            for X in ("A", "B"):
                lds0 = b.smem_addr_of(self.stg[(stage, "s" + X)])
                # global byte offset of the CTA's scale image of this tile (pack_scales_tile layout)
                base = b.mul(b.add(b.mul(tile, self.c32(self.nq[X])), self.sc_cta_q[X]), self.c32(512))
                for j in range(self.sc_dtl_per_wave[X]):

                    def sc_issue(X=X, j=j, lds0=lds0, base=base):
                        off = b.add(self.sc_dtl_w, self.c32(j * 256 * self.nw))
                        lds = b.smem_ptr_add(lds0, b.zext(off, I64))
                        b.async_buffer_load_lds_addr(self.sc_rsrc[X], lds, self.sc_dtl_voff, b.add(base, off), 1)
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
        """Direct B: one closure per fragment of 128-K tile ``tile``, filling ``out[("B", f, 0, d)]``."""
        b = self.b
        holder: List[Value] = []

        def rsrc():
            if not holder:
                kb = b.mul(tile, self.c32(self.tile_bytes["B"]))
                holder.append(b.buffer_rsrc(b.global_ptr_add(self.base_ptr["B"], kb), self.c32(0x7FFFFFF0)))
            return holder[0]

        res = []
        for f in range(self.fn):

            def ld(f=f):
                r = rsrc()
                v0 = b.buffer_load_vN(r, self.bd_voff[0], b.add(self.bd_soff[0], self.c32(1024 * f)), I32, 4)
                v1 = b.buffer_load_vN(r, self.bd_voff[1], b.add(self.bd_soff[1], self.c32(512 * f)), I32, 2)
                self._vm_fence("bdir", 2)
                parts = [b.vec_extract(v0, e) for e in range(4)] + [b.vec_extract(v1, e) for e in range(2)]
                out[("B", f, 0, d)] = b.vec_pack(parts, I32)

            res.append(ld)
        return res

    def _frag_load(self, stage: int, X: str, f: int, t: int, out: Dict) -> Callable[[], None]:
        b, T, spec = self.b, self.T, self.spec

        def ld():
            parts = []
            smem = self.stg[(stage, X)]
            if self.layout[X] == "canonical":
                base = b.add(self.frag_word[(t, X)], self.c32(f * T * 24))
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
                    q = b.add(self.sc_q0[X], self.c32(qi))
                    w = self.sc_nd[X]  # dwords per lane
                    chunk = 512 if self.tile_scales else 1024  # bytes per (period, row chunk)
                    soff = b.mul(b.add(b.mul(u, self.c32(self.nq[X])), q), self.c32(chunk))
                    out[(X, qi)] = b.buffer_load_vN(self.sc_rsrc[X], self.sc_voff_x[X], soff, I32, w)
                    self._vm_fence("scale")

                res.append(ld)
        return res

    def _scale_of(self, sc: Dict, X: str, f: int, t: int, kstep: int) -> Tuple[Value, int]:
        """(scale dword, op_sel) of MFMA tile ``f`` of operand ``X`` at K step ``t`` of the tile."""
        b = self.b
        if self.tile_scales:
            d = f // 4 if self.T == 16 else t
            return b.vec_extract(sc[(X, 0)], d), f % 4
        return b.vec_extract(sc[(X, f // 4)], f % 4), kstep

    def _mfma(self, acc, fa, fb, sa, sb, ja, jb):
        b = self.b
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
        for t in range(self.ahead):
            ev += [(-1, -2, "dtl", t)] * n_dtl
        if self.spec.b_direct:
            for d in range(self.spec.b_depth):
                ev += [(-1, -1, "bdir", d)] * (2 * self.fn)
        n_u = self.n_tiles // self.P
        for i in range(self.n_tiles):
            items = []
            if self.clamp or i + self.ahead < self.n_tiles:
                items += [(s, "dtl", i + self.ahead) for s in self.plan.dtl_slots]
            if i % self.P == 0 and (self.clamp or i // self.P + 1 < n_u):
                items += [(self.plan.scale_slot, "scale", i // self.P + 1)] * sc_n
            if self.spec.b_direct and (self.clamp or i + self.spec.b_depth < self.n_tiles):
                for j in range(self.fn):
                    items += [(self._bdir_slot(j), "bdir", i + self.spec.b_depth)] * 2
            items.sort(key=lambda x: (x[0], {"dtl": 0, "scale": 1, "bdir": 2}[x[1]]))
            ev += [(i, s, kd, tg) for s, kd, tg in items]
        return ev

    @property
    def ahead(self) -> int:
        """K tiles between a tile and the tile its refill DTLs fetch."""
        return self.S if self.spec.release_slot >= 0 else self.S - 1 + self.adv

    def _bdir_slot(self, j: int) -> int:
        return min(self.plan.n_mfma - 1, self.spec.b_slot + j * self.spec.b_every)

    def _acquire_counts(self) -> Dict[int, int]:
        """vmcnt for the acquire of tile ``j`` (at tile ``j - adv``'s acquire slot; prologue for j < adv)."""
        if self.spec.reg_stages:  # ds_write fill: the acquire is lgkmcnt(0) + barrier, no vmcnt
            return {j: -1 for j in range(self.n_tiles)}
        ev = self._vm_events()
        last_dtl: Dict[int, int] = {}
        for idx, e in enumerate(ev):
            if e[2] == "dtl":
                last_dtl[e[3]] = idx
        # position of the acquire in tile i = index of the first event at/after its slot
        first_at: Dict[int, int] = {}
        n_pro = sum(1 for e in ev if e[0] == -1)
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
            pos = n_pro if j - self.acq_off < 0 else first_at[j - self.acq_off]
            counts[j] = pos - 1 - last_dtl[j]
            if counts[j] < 0:
                raise AssertionError(f"tile {j} acquired before its DTLs were issued")
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
                        idx = b.add(b.mul(self.sc_wq[X], self.c32(128)), b.mul(self.lane, self.c32(2)))
                        carry_next[("S", X, 0)] = b.smem_load_vN(self.stg[(nxt_stage, "s" + X)], idx, dtype=I32, n=2)
                    for key in self._frag_keys_in_order()[:spec.carry]:
                        self._frag_load(nxt_stage, *key, carry_next)()

                pre.setdefault(spec.carry_slot, []).append(carry_acquire)
        elif has_read or has_dtl:
            pre.setdefault(pl.acq, []).append(acquire)
        if self.sc_lds and A == 1:  # prefetch: this tile's scales were read with its fragments (previous tile)
            scales_cur = dict(scales_cur)
            for X in ("A", "B"):
                scales_cur[(X, 0)] = frags_in[("S", X, 0)]
        if self.sc_lds and has_read and not spec.carry:
            sc_stage = rd_stage
            sc_out = read_now if A == 1 else scales_cur
            sc_key = (lambda X: ("S", X, 0)) if A == 1 else (lambda X: (X, 0))

            def sc_read():
                for X in ("A", "B"):
                    idx = b.add(b.mul(self.sc_wq[X], self.c32(128)), b.mul(self.lane, self.c32(2)))
                    sc_out[sc_key(X)] = b.smem_load_vN(self.stg[(sc_stage, "s" + X)], idx, dtype=I32, n=2)

            pre.setdefault(pl.acq, []).append(sc_read)
        if spec.probe == "dtl_only":
            has_read = False
        if spec.probe == "lds_only":
            has_dtl = False
        bq_next: Dict = {}
        bdir_issue: List = []
        if spec.b_direct:
            D = spec.b_depth
            for (X, f, t, d), v in frags_in.items():
                if d == 0:
                    read_now[(X, f, t)] = v
                else:
                    bq_next[(X, f, t, d - 1)] = v
            if i_static is None or i_static + D < T_total or self.clamp:
                if i_static is not None:
                    tgt_b = self.c32(min(i_static + D, T_total - 1))
                elif self.clamp:  # past K: re-load the last tile (in range; never consumed)
                    tgt_b = b.smin(b.add(i_dyn, self.c32(D)), self.c32(T_total - 1))
                else:
                    tgt_b = b.add(i_dyn, self.c32(D))
                # registered after the DTLs / scale loads below: within one MFMA slot the VM issue order is
                # dtl, scale, bdir (= the static vmcnt model's order)
                bdir_issue = list(enumerate(self._bdir_loads(tgt_b, D - 1, bq_next)))
        if has_read:
            keys = self._frag_keys_in_order()
            if spec.b_direct:
                keys = [key for key in keys if key[0] == "A"]
            if spec.carry:
                keys = keys[spec.carry:]
            if A == 1:
                for key, s_ in zip(keys, pl.read_slots):
                    pre.setdefault(s_, []).append(self._frag_load(rd_stage, *key, read_now))
            else:
                for key in keys:
                    s_ = max(pl.acq, self.first_use[key] - lead)
                    if spec.release_slot >= 0:
                        s_ = min(s_, spec.release_slot)  # every read of this stage before its release
                    pre.setdefault(s_, []).append(self._frag_load(rd_stage, *key, read_now))
        if has_dtl and spec.release_slot >= 0:

            def release():  # every wave finished reading this tile's stage: it may be refilled
                b.s_waitcnt(lgkmcnt=0)
                b.s_barrier_bare()

            pre.setdefault(spec.release_slot, []).append(release)
        if has_dtl:
            if i_static is not None:
                tgt = self.c32(min(i_static + self.ahead, T_total - 1))
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
                for f, s_ in zip(self._dtl_issues(tgt, dtl_stage), slots):
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
                u_next = self.c32(min(i_static // P + 1, n_u - 1))
            elif self.clamp:
                u_next = b.smin(b.add(b.div(i_dyn, self.c32(P)), self.c32(1)), self.c32(n_u - 1))
            else:
                u_next = b.add(b.div(i_dyn, self.c32(P)), self.c32(1))
            for f in self._scale_loads(u_next, scales_next):
                pre.setdefault(pl.scale_slot, []).append(f)
        for j, f_ in bdir_issue:
            pre.setdefault(self._bdir_slot(j), []).append(f_)
        frags = frags_in if A == 1 else read_now
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
            return acc, {**frags_in, **read_now}
        return acc, ({**bq_next, **rg_next} if spec.reg_stages else bq_next)

    # ------------------------------------------------------------- main loop
    def _prologue(self) -> Tuple[Dict, Dict, Dict[int, int]]:
        b, S, A = self.b, self.S, self.adv
        counts = self._acquire_counts()
        sc_cur: Dict = {}
        for f in self._scale_loads(self.c32(0), sc_cur):
            f()
        for t in range(self.ahead):
            for f in self._dtl_issues(self.c32(t), t % S):
                f()
        fr: Dict = {}
        if self.spec.reg_stages:
            b.s_waitcnt(vmcnt=0)  # the DTL-filled first tiles landed (the loop's acquire waits on LDS only)
            for d in range(self.spec.reg_stages):
                for f in self._reg_loads(self.c32(self.ahead + d), d, fr):
                    f()
        if self.spec.b_direct:
            for d in range(self.spec.b_depth):
                for f in self._bdir_loads(self.c32(d), d, fr):
                    f()
        if A == 1:
            b.s_waitcnt(vmcnt=counts[0], lgkmcnt=0)
            b.s_barrier_bare()
            if self.sc_lds:
                for X in ("A", "B"):
                    idx = b.add(b.mul(self.sc_wq[X], self.c32(128)), b.mul(self.lane, self.c32(2)))
                    fr[("S", X, 0)] = b.smem_load_vN(self.stg[(0, "s" + X)], idx, dtype=I32, n=2)
            for key in self._frag_keys_in_order():
                self._frag_load(0, *key, fr)()
        if self.spec.carry:
            b.s_waitcnt(vmcnt=counts[0], lgkmcnt=0)
            b.s_barrier_bare()
            for X in ("A", "B"):
                idx = b.add(b.mul(self.sc_wq[X], self.c32(128)), b.mul(self.lane, self.c32(2)))
                fr[("S", X, 0)] = b.smem_load_vN(self.stg[(0, "s" + X)], idx, dtype=I32, n=2)
            for key in self._frag_keys_in_order()[:self.spec.carry]:
                self._frag_load(0, *key, fr)()
        return sc_cur, fr, counts

    def _mainloop(self) -> None:
        b, spec = self.b, self.spec
        S, P, A = self.S, self.P, self.adv
        U = S * P // _gcd(S, P) * spec.unroll
        T = self.n_tiles
        sc_cur, fr, counts = self._prologue()
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
            acc = [b.inline_asm("", "=a,0", [zero], zero.type, sideeffect=False) if spec.acc_agpr else zero
                   for _ in range(n_acc)]
            H = 0
        pending = None

        def static_tile(i, acc, fr, sc_cur, pending):
            sc_next: Dict = {}
            acc, fr = self._tile(self.c32(i), i, i % U, acc, fr, sc_cur, sc_next, counts.get(i + self.acq_off, 0))
            if sc_next:
                pending = sc_next
            if (i + 1) % P == 0 and pending is not None:
                sc_cur, pending = pending, None
            return acc, fr, sc_cur, pending

        for i in range(H):
            acc, fr, sc_cur, pending = static_tile(i, acc, fr, sc_cur, pending)
        # main loop: L iterations of U tiles from tile H; every tile in it has its reads and refills
        L = (T - H) // U if self.clamp else max(0, (T - H - S - A) // U)
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
        for i in range(H + L * U, T):
            acc, fr, sc_cur, pending = static_tile(i, acc, fr, sc_cur, pending)
        self.acc = acc

    def _emit_loop(self, tag, trips, U, H, P, A, acc, fr, sc_cur, frag_keys, sc_keys, body_wait):
        """One K-loop copy of ``trips`` iterations of U tiles from tile H."""
        b = self.b
        iter_args = [(f"acc{tag}{i}", v) for i, v in enumerate(acc)]
        iter_args += [("fr" + tag + "_" + "_".join(str(x) for x in key), fr[key]) for key in frag_keys]
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
                b.scf_yield(*(acc_l + [fr_l[key] for key in frag_keys] + [sc_l[key] for key in sc_keys]))
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
        loop = b.scf_for_iter(self.c32(0), self.c32(L), self.c32(1), [(f"acc{i}", v) for i, v in enumerate(acc)],
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
    def _epilogue(self) -> None:
        b, spec, g = self.b, self.spec, self.geo
        b.s_waitcnt(vmcnt=0)
        lane, T = self.lane, self.T
        if spec.out_exp2:  # exact power-of-two rescale of every accumulator (host scale bias D = -out_exp2)
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
            for mi in range(self.fm):
                soff = b.to_sgpr_u32(b.add(b.mul(b.add(m0, self.c32(mi * T)), self.c32(n_bytes)),
                                           b.mul(n0, self.c32(2))))
                # lanes rr, rr ^ 1 hold adjacent columns: swap halves with a DPP xor-1 so the even lane stores
                # rows 0-1 and the odd lane rows 2-3, each as (even column, odd column) bf16 pairs = one dword
                par = b.land(rr, self.c32(1))
                odd = b.cmp_eq(par, self.c32(1))
                row0 = b.add(b.mul(grp, self.c32(4)), b.mul(par, self.c32(2)))
                col_b = b.mul(b.sub(rr, par), self.c32(2))
                lo_m, hi_m = self.c32(0xFFFF), self.c32(-65536)
                for ni in range(self.fn):
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
                        off = b.add(b.add(b.mul(b.add(row0, self.c32(j)), self.c32(n_bytes)), col_b),
                                    self.c32(ni * T * 2))
                        b.buffer_store_vN(rsrc, off, soff, b.vec_pack([w], I32))
            return
        # lane holds C[m0 + mi*T + rr][n0 + ni*T + 4*grp + 8*q4 + (0..3)]
        lane_off = b.add(b.mul(rr, self.c32(n_bytes)), b.mul(grp, self.c32(8)))
        nq4 = g.c_len // 4
        zero = b.zero_vec_f32(4)
        for mi in range(self.fm):
            soff = b.to_sgpr_u32(b.add(b.mul(b.add(m0, self.c32(mi * T)), self.c32(n_bytes)), b.mul(n0, self.c32(2))))
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

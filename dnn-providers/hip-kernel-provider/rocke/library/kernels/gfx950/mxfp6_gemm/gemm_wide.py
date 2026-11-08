# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx950 MXFP6 x MXFP6 GEMM with a wide CTA tile: ``C[M, N] (bf16) = A[M, K] * B[N, K]^T``, fp32 accumulate.

Same problem contract as :mod:`.gemm` (OCP MX 6-bit codes, E8M0 scale per 32 K, bf16 RNE output); a different
CTA tile and fragment schedule that move fewer bytes per MFMA:

* CTA tile ``256 x tile_n`` (``tile_n`` 384 by default) with 2 x 2 waves of ``128 x tile_n / 2``; the ``N``
  columns that do not fill a wide tile run as ``256 x 256`` tiles in the same launch (``N = q tile_n + r 256``:
  a CTA-uniform branch picks the wide or the narrow body). Per 128-K step a wide CTA moves 640 rows (L2 -> LDS)
  for 384 MFMAs and each wave reads 20 fragments for 96 MFMAs: 1/6 fewer bytes per MFMA on both paths than the
  256 x 256 tile (512 rows / 256 MFMAs, 16 fragments / 64 MFMAs).
* accumulators: ``128 x 192 / 64`` = 384 fp32 per lane (the direct compiler puts 256 in AGPRs, the rest in VGPRs),
  so the operands get the remaining 128 VGPRs: the wave **holds** its B fragments for a whole K tile and **streams**
  its A fragments (MFMA order: A rows outer, B columns inner, serpentine; the row direction alternates with the
  tile parity so the next tile's first row walks the columns in the order the last row frees them). B(t+1)[c] is
  read right after the last MFMA that uses B(t)[c] (``fn`` MFMAs of lead); A(t)[r+1] is read at the start of row
  ``r`` (two A buffers).
* LDS: separate rings for A (``stages_a`` slots) and B (``stages_b``), c0c1 stage images (one contiguous block per
  operand, ``ds_read_b128`` + ``ds_read_b64`` per fragment), scales DTL'd with the operands (tile op_sel layout).
  Per tile two barriers: the *release* barrier ``R`` after the first row (every wave has consumed the B fragments it
  read from this tile's B slot, which is then refilled) and the *acquire* ``Q`` at the start of the last row
  (``s_waitcnt vmcnt(n)`` for the next tile's fills + ``s_barrier``; every wave has consumed this tile's A
  fragments, whose slot is then refilled). A static VM-issue model gives ``n`` (:func:`vm_events`).
* scales: A in :func:`layout.pack_scales_tile` (one ``ds_read_b64`` per tile and wave), B in the wide layout of
  :func:`pack_scales_b` (four dwords per lane and wave half, one ``ds_read_b128``; 192-row wave halves need three).
* XCD order ``band``: XCD ``x`` takes a band of ``gm / 4`` tile rows and one half of the tile columns (an odd
  middle column split by rows), so each XCD's CTAs share their A and B panels in L2.

Host images (:func:`host_images`): A as in :mod:`.gemm` (c0c1, 256-row blocks; tile scales); B as c0c1 blocks of the
column widths (``tile_n`` rows, then 256) and :func:`pack_scales_b`. Rust (direct) compiler only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields, replace
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from rocke.core.ir import BF16, F32, I8, I32, I64, IRBuilder, KernelDef, PtrType, Value, VectorType

from . import layout as L
from .gemm import ELEM_FMTS, NUM_XCDS, VM_FENCE_MASK

__all__ = [
    "KNOBS",
    "MUTANTS",
    "PRESETS",
    "WideGemmSpec",
    "build_mxfp6_gemm_wide",
    "sk_role",
    "workspace_bytes",
    "col_widths",
    "grid_of",
    "host_images",
    "image_key",
    "kernel_name",
    "lds_bytes",
    "pack_scales_b",
    "path_plan",
    "preset",
    "tile_of_wg",
    "validate_spec",
    "vm_events",
]

ARCH = "gfx950"
ROW_BYTES = L.ROW_BYTES  # 96 bytes per row per 128-K tile
T16 = 16  # MFMA tile rows (16x16x128)
FM = 8  # A fragments per wave (128 wave rows)
SB_DWORDS = 4  # B scale dwords per lane and wave half (wide layout)
MUTANTS = ("none", "scale_sel", "k_order", "drop_kstep", "wrong_fmt")
XCD_MAPS = ("band", "none")
PROBES = ("none", "mfma_only", "stamps")
STAMP_OFF, STAMP_BYTES = 1024, 64  # stamps probe: CTA role rho writes 16 dwords at flags + 1024 + 64 rho


@dataclass(frozen=True)
class WideGemmSpec:
    """One build of the wide-tile kernel. Every field has a default."""

    m: int = 4096
    n: int = 4096
    k: int = 65536
    elem_fmt: str = "e2m3"
    tile_n: int = 384  # wide column width (256: every tile narrow -- the schedule's own 256 x 256 control)
    head_rows: int = 1  # the tile's first h A rows run column-interleaved (each B fragment feeds h MFMAs back to back):
    #                     the next tile's B refill, read during the last row, gets h x the MFMA time to land
    stages_a: int = 3  # A ring slots
    stages_b: int = 2  # B ring slots (3 fits the LDS only with 256-wide tiles)
    release: bool = True  # R barrier after the first row (refill this tile's B slot early)
    release_delay: int = 1  # R after MFMA fn - 1 + release_delay
    acq_delay: int = 0  # Q after MFMA (fm - 1) fn + acq_delay (<= 0; < 0 moves the acquire earlier)
    a_delay: int = 0  # A(t)[r + 1] read after MFMA r fn - 1 + a_delay
    dtl_delay: int = 1  # MFMAs between a barrier and the first refill DTL it releases
    b_dtl_end: str = "acq"  # B refill DTLs spread up to the acquire ("acq") or the tile end ("end")
    a_dtl: str = "late"  # A refill DTLs: "spread" over the next tile (from its start to a_dtl_end, interleaved with
    #                        the B refill) | "late": right after the releasing barrier (the acquire), in the last row
    a_dtl_end: str = "acq"  # last slot of a spread A refill: before the acquire ("acq") or the tile end ("end")
    b_dtl: str = "release"  # B refill DTLs right after their releasing barrier | "spread" over the next tile (needs
    #                        stages_b 3 for lead; with release=False: one barrier per tile)
    sc_exact: bool = True  # read only the B scale dwords the path uses (b64 / b64 + b32): no dead register a following
    #                        LDS load can reuse (which costs an lgkmcnt(0) after the acquire)
    unroll: int = 1  # loop body = unroll x lcm(2, stages_a, stages_b) tiles
    dtl_imm: bool = True  # four 1 KiB DTLs share M0 / soffset (instruction offset)
    setprio: int = 0
    xcd_map: str = "band"
    k_phase: int = 0  # fabric probe: CTAs with (tile row + tile column) odd walk K rotated by this many K tiles (two
    #                   K phases per XCD, as a stream-K split of the wide tiles would run; result exact, order differs)
    stream_k: bool = False  # 4096 x 4096 only: every CU busy -- 160 "head" CTAs (K tiles [0, k_head) of one wide tile,
    #   XCDs 0-4), 80 "tail" CTAs (K [k_head, K) of a rows-0-7 tile, then of a rows-8-15 tile; XCDs 5, 6, 7), 16 narrow
    #   CTAs (XCD 7); fp32 partials through a workspace + agent-scope flags (kernel params ws, flags)
    k_head: int = 0  # head K tiles (0: 2/3 of K, rounded)
    probe: str = "none"
    mutant: str = "none"


KNOBS: Dict[str, tuple] = {
    "tile_n": (384, 256),
    "head_rows": (1, 2),
    "stages_a": (3, 2),
    "stages_b": (2, 3),
    "release": (True, False),
    "release_delay": (1, 0, 2, 4, 8),
    "acq_delay": (0, -1, -2, -4, -8),
    "a_delay": (0, 1, 2, 4),
    "dtl_delay": (1, 0, 2, 4),
    "b_dtl_end": ("acq", "end"),
    "a_dtl": ("late", "spread"),
    "b_dtl": ("release", "spread"),
    "sc_exact": (True, False),
    "a_dtl_end": ("acq", "end"),
    "unroll": (1, 2),
    "dtl_imm": (True, False),
    "setprio": (0, 1),
    "xcd_map": ("band", "none"),
    "stream_k": (False, True),
}

PRESETS: Dict[str, Dict[str, object]] = {
    "wide": {},
    "narrow": {"tile_n": 256},  # the hold-B / stream-A schedule on 256 x 256 tiles only (control)
}


def preset(name: str, **over) -> WideGemmSpec:
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r} (expected one of {tuple(PRESETS)})")
    return replace(WideGemmSpec(), **{**PRESETS[name], **over})


def _defaults() -> Dict[str, object]:
    return {f.name: f.default for f in fields(WideGemmSpec)}


# ----------------------------------------------------------------- geometry


def col_widths(spec: WideGemmSpec) -> List[int]:
    """Widths of the CTA tile columns: ``q`` wide ones, then 256-wide ones."""
    q = spec.n // spec.tile_n
    r = spec.n - q * spec.tile_n
    return [spec.tile_n] * q + [256] * (r // 256)


def n_wide(spec: WideGemmSpec) -> int:
    return sum(1 for w in col_widths(spec) if w != 256) if spec.tile_n != 256 else 0


def grid_of(spec: WideGemmSpec) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    if spec.stream_k:
        return (256, 1, 1), (256, 1, 1)
    return ((spec.m // 256) * len(col_widths(spec)), 1, 1), (256, 1, 1)


SK_WIDE_COLS, SK_ROWS = 10, 16  # the stream-K schedule's grid: 16 tile rows x (10 wide + 1 narrow) columns
SK_HEADS, SK_TAILS = 160, 80
WS_BYTES_PER_TILE = 4 * 96 * 1024  # fp32 partial of one 256 x 384 tile (4 waves x 96 accumulators x 64 lanes x 16 B)


def k_head_of(spec: WideGemmSpec) -> int:
    return spec.k_head or round(spec.k // 128 * 2 / 3)


def sk_role(wg: int) -> Tuple[str, List[Tuple[int, int]]]:
    """Stream-K role of CTA ``wg`` (host model of the kernel's decode): ("head", [tile]), ("tail", [T1, T2]) or
    ("narrow", [tile]); tiles as (tile row, tile column). XCD x = wg % 8 takes roles 32 x .. 32 x + 31."""
    rho = (wg % NUM_XCDS) * 32 + wg // NUM_XCDS
    if rho < SK_HEADS:
        blk, loc = rho // 32, rho % 32
        if blk < 4:
            return "head", [(8 * (blk % 2) + loc // 4, 4 * (blk // 2) + loc % 4)]
        return "head", [(loc // 2, 8 + loc % 2)]
    if rho < SK_HEADS + SK_TAILS:
        j = rho - SK_HEADS
        blk, jj = j // 32, j % 32
        r, c = (jj // 4, 4 * blk + jj % 4) if blk < 2 else (jj // 2, 8 + jj % 2)
        return "tail", [(r, c), (8 + r, c)]
    return "narrow", [(rho - SK_HEADS - SK_TAILS, SK_WIDE_COLS)]


def workspace_bytes(spec: WideGemmSpec) -> Tuple[int, int]:
    """(partial workspace bytes, flag bytes) a stream-K launch needs (flags zeroed once; the kernel resets them)."""
    if not spec.stream_k:
        return 0, 0
    return SK_HEADS * WS_BYTES_PER_TILE, (STAMP_OFF + 256 * STAMP_BYTES if spec.probe == "stamps" else SK_HEADS * 4)


def tile_of_wg(spec: WideGemmSpec, wg: int) -> Tuple[int, int]:
    """Host model of the CTA -> (tile row, tile column) map."""
    gm, gc = spec.m // 256, len(col_widths(spec))
    if spec.xcd_map == "none":
        return wg // gc, wg % gc
    x, loc = wg % NUM_XCDS, wg // NUM_XCDS
    rb = gm // 4
    h = gc // 2
    band, half = x // 2, x % 2
    if loc < rb * h:
        r = band * rb + loc // h
        c = (gc - h + loc % h) if half else loc % h
        return r, c
    j = loc - rb * h
    return band * rb + half * (rb // 2) + j, h


def lds_bytes(spec: WideGemmSpec) -> int:
    wb = max(col_widths(spec))
    return spec.stages_a * (256 * ROW_BYTES + 256 * 4) + spec.stages_b * (wb * ROW_BYTES + 2 * 64 * SB_DWORDS * 4)


def validate_spec(spec: WideGemmSpec) -> Tuple[bool, str]:
    if spec.elem_fmt not in ELEM_FMTS:
        return False, f"elem_fmt must be one of {tuple(ELEM_FMTS)}"
    if spec.tile_n not in (256, 384):
        return False, "tile_n must be 256 or 384"
    if spec.m % 256 or spec.k % 512 or spec.k // 128 < 8:
        return False, "M must be a multiple of 256, K of 512 (>= 1024)"
    q = spec.n // spec.tile_n
    if (spec.n - q * spec.tile_n) % 256:
        return False, "N must be q tile_n + a multiple of 256"
    if spec.stages_a not in (2, 3) or spec.stages_b not in (2, 3):
        return False, "stages_a / stages_b must be 2 or 3"
    if lds_bytes(spec) > 160 * 1024:
        return False, f"LDS {lds_bytes(spec)} B exceeds 160 KiB"
    if spec.xcd_map not in XCD_MAPS or spec.mutant not in MUTANTS or spec.probe not in PROBES:
        return False, "bad xcd_map / mutant / probe"
    if spec.probe != "none" and spec.mutant != "none":
        return False, "probe and mutant are exclusive"
    if spec.probe == "stamps" and not spec.stream_k:
        return False, "the stamps probe is for the stream-K schedule"
    if spec.xcd_map == "band":
        gm, gc = spec.m // 256, len(col_widths(spec))
        if gm % 4 or (gm // 4) % 2 or (gm * gc) % NUM_XCDS:
            return False, "xcd_map band needs M / 256 a multiple of 8 tile rows and the grid a multiple of 8"
    if not 0 <= spec.k_phase < spec.k // 128:
        return False, "k_phase must be 0 .. K / 128 - 1"
    if spec.stream_k:
        if (spec.m, spec.n, spec.tile_n) != (4096, 4096, 384) or spec.k_phase or spec.probe not in ("none", "stamps"):
            return False, "stream_k is the 4096 x 4096 schedule with 384-wide tiles (no k_phase / probe but stamps)"
        kh = k_head_of(spec)
        if not 8 <= kh <= spec.k // 128 - 8:
            return False, "k_head must leave >= 8 K tiles to both head and tail"
    if spec.unroll not in (1, 2) or spec.setprio not in (0, 1, 2, 3):
        return False, "unroll 1..2, setprio 0..3"
    if (spec.b_dtl_end not in ("acq", "end") or spec.a_dtl_end not in ("acq", "end")
            or spec.a_dtl not in ("spread", "late") or spec.b_dtl not in ("release", "spread")):
        return False, "bad a_dtl / b_dtl / dtl end"
    for fn in sorted({w // 2 // T16 for w in col_widths(spec)}):
        try:
            vm_events(path_plan(spec, fn), spec.k // 128)
        except ValueError as e:
            return False, str(e)
    return True, ""


def kernel_name(spec: WideGemmSpec) -> str:
    d = _defaults()
    diff = {k: v for k, v in asdict(spec).items() if k not in ("m", "n", "k", "elem_fmt") and v != d[k]}
    f = spec.elem_fmt
    base = (f"rocke_gemm_mx_{f}_{f}_e8m0b32_f32acc_bf16_rcr_256x{spec.tile_n}x128_w2x2_16x16x128_wide"
            f"_{spec.m}x{spec.n}x{spec.k}")
    if spec.probe != "none":
        base += f"_PROBE_{spec.probe}"
    if spec.mutant != "none":
        base += f"_MUT_{spec.mutant}"
    if diff:
        base += "_k" + hashlib.sha1(json.dumps(diff, sort_keys=True).encode()).hexdigest()[:10]
    return base


# ----------------------------------------------------------------- static plan of one path (fn B fragments)


@dataclass
class PathPlan:
    fn: int  # B fragments per wave (12 wide, 8 narrow)
    rows_b: int  # B rows of the CTA tile
    n_mfma: int
    q: int  # acquire after MFMA q
    r: int  # release barrier after MFMA r (-1: none)
    d: Dict[str, int]  # tiles between the issuing tile and the tile a refill fills
    issue_bar: Dict[str, int]  # slot of the barrier each operand's refill follows
    dtl: List[Tuple[int, str, str, int]]  # (slot, operand, "op" | "sc", index) in emission order, full tile
    a_read: Dict[int, int]  # slot of the read of A(t)[r], r = h..fm-1 (start of the unit before r's unit)
    a0_next: int  # slot of the reads of A(t + 1)[0..h-1]
    h: int  # head rows
    U: int  # loop body tiles

    def n_dtl(self, X: str) -> int:
        return sum(1 for e in self.dtl if e[1] == X)


def units(h: int) -> List[List[int]]:
    """A rows of each unit of a tile: the first ``h`` rows column-interleaved, then one row per unit."""
    return [list(range(h))] + [[r] for r in range(h, FM)]


def unit_starts(fn: int, h: int) -> List[int]:
    out, x = [], 0
    for rows in units(h):
        out.append(x)
        x += len(rows) * fn
    return out


def mfma_order(fn: int, parity: int, h: int = 1) -> List[Tuple[int, int]]:
    """(A row, B column) of every MFMA of a tile: units outer (the h head rows interleaved per column, then single
    rows), columns serpentine; with an even number of units the first unit's direction flips with the tile parity, so
    the next tile's first unit always walks the columns in the order the last row frees them."""
    us = units(h)
    flip = parity if len(us) % 2 == 0 else 0
    out = []
    for u, rows in enumerate(us):
        cols = range(fn) if (u + flip) % 2 == 0 else reversed(range(fn))
        for c in cols:
            out += [(mi, c) for mi in rows]
    return out


def _lcm(a: int, b: int) -> int:
    x, y = a, b
    while y:
        x, y = y, x % y
    return a * b // x


def path_plan(spec: WideGemmSpec, fn: int) -> PathPlan:
    n = FM * fn
    h = spec.head_rows
    if h not in (1, 2):
        raise ValueError("head_rows must be 1 or 2")
    rows_b = 2 * fn * T16
    st = unit_starts(fn, h)
    last = st[-1]  # first MFMA of the last row
    q = last + spec.acq_delay
    if not h * fn - 1 <= q <= last:
        raise ValueError("acquire slot must lie after the head unit and before the last row (acq_delay too negative)")
    r = h * fn - 1 + spec.release_delay if spec.release else -1
    if spec.release and not h * fn - 1 <= r < q:
        raise ValueError("release barrier must follow the head unit and precede the acquire")
    bars = sorted([q] + ([r] if r >= 0 else []))
    consumed = {"B": h * fn - 1, "A": last}  # last MFMA consuming a tile's LDS reads of the operand
    S = {"A": spec.stages_a, "B": spec.stages_b}
    d, issue_bar, lo_of = {}, {}, {}
    for X in ("B", "A"):
        # the first barrier after every wave consumed the reads of the slot a refill overwrites: in the consuming tile
        # (refill issued there: d = S) or the first barrier of the next tile (d = S - 1)
        same = [s for s in bars if s >= consumed[X]]
        if (X == "A" and spec.a_dtl == "spread") or (X == "B" and spec.b_dtl == "spread"):
            # issued over the tile after the consuming one: from its start when the releasing barrier closed the
            # consuming tile, else after the next tile's first barrier
            d[X] = S[X] - 1
            issue_bar[X] = same[0] - n if same else bars[0]  # (< 0: the previous tile's barrier)
        elif same:
            d[X], issue_bar[X] = S[X], same[0]
        else:
            d[X], issue_bar[X] = S[X] - 1, bars[0]
        lo_of[X] = max(0, issue_bar[X] + 1) + spec.dtl_delay
        if d[X] < 1:
            raise ValueError(f"{X}: ring too shallow for this barrier placement")
    per_wave = {"A": 256 * ROW_BYTES // 1024 // 4, "B": rows_b * ROW_BYTES // 1024 // 4}
    sc_per_wave = {"A": 1, "B": 2}  # dword (256 B) scale DTLs per wave: A 1 KiB, B 2 KiB per CTA tile
    dtl: List[Tuple[int, str, str, int]] = []
    for X in ("B", "A"):
        lo = lo_of[X]
        end_mode = spec.b_dtl_end if X == "B" else spec.a_dtl_end
        hi = (q if (end_mode == "acq" and lo <= q - 1) else n) - 1
        items = [(X, "op", j) for j in range(per_wave[X])] + [(X, "sc", j) for j in range(sc_per_wave[X])]
        span = max(1, hi - lo + 1)
        for j, it in enumerate(items):
            dtl.append((min(n - 1, lo + (j * span) // len(items)), *it))
    dtl.sort(key=lambda e: e[0])  # stable: B before A at equal slots
    a_read = {mi: max(-1, st[mi - h] - 1 + spec.a_delay) for mi in range(h, FM)}  # unit of row mi is mi - h + 1
    a0_next = max(q, last - 1 + spec.a_delay)
    U = _lcm(_lcm(2, spec.stages_a), spec.stages_b) * spec.unroll
    return PathPlan(fn=fn, rows_b=rows_b, n_mfma=n, q=q, r=r, d=d, issue_bar=issue_bar, dtl=dtl, a_read=a_read,
                    a0_next=a0_next, h=h, U=U)


def vm_events(pp: PathPlan, n_tiles: int) -> Tuple[List[Tuple[int, int, str, int]], Dict[int, int]]:
    """Static VM issue order of one path ``(tile, slot, operand, target tile)`` (tile -1 = prologue) and the
    ``vmcnt`` of each acquire: ``counts[u]`` = the wait that makes every fill of tile ``u`` land (at ``Q_{u-1}``,
    the prologue for ``u = 0``)."""
    ev: List[Tuple[int, int, str, int]] = []
    per = {X: [e for e in pp.dtl if e[1] == X] for X in ("A", "B")}
    for u in range(max(pp.d.values())):
        for X in ("A", "B"):
            if u < pp.d[X]:
                ev += [(-1, -1, X, u)] * len(per[X])
    for t in range(n_tiles):
        for slot, X, _, _ in pp.dtl:
            if t + pp.d[X] < n_tiles:
                ev.append((t, slot, X, t + pp.d[X]))
    last: Dict[int, int] = {}
    for i, e in enumerate(ev):
        last[e[3]] = i
    run = -1
    for u in range(n_tiles):
        run = max(run, last.get(u, -1))
        last[u] = run
    n_pro = sum(1 for e in ev if e[0] == -1)
    # position of Q_t = index of the first event of tile t at / after slot q (DTLs at slot q follow the acquire)
    first_at: Dict[int, int] = {}
    i = n_pro
    for t in range(n_tiles):
        while i < len(ev) and ev[i][0] == t and ev[i][1] < pp.q:
            i += 1
        first_at[t] = i
        while i < len(ev) and ev[i][0] == t:
            i += 1
    counts: Dict[int, int] = {}
    for u in range(n_tiles):
        pos = n_pro if u == 0 else first_at[u - 1]
        counts[u] = pos - 1 - last[u]
        if counts[u] < 0:
            raise ValueError(f"tile {u} acquired before its fills were issued (raise the ring depth)")
    return ev, counts


# ----------------------------------------------------------------- host images


def pack_scales_b(scales: np.ndarray, widths: List[int]) -> np.ndarray:
    """B scales in the wide layout (uint32, flat): per column block ``c`` (``widths[c]`` rows), K tile ``s``, wave half
    ``g`` (``widths[c] / 2`` rows), lane ``l`` four dwords ``d``: byte ``i`` of dword ``d`` (``d < rows_g / 64``) =
    ``scale[row0_c + g rows_g + 64 d + 16 i + l % 16, 4 s + l // 16]``; unused dwords 0.
    dword index = ((((c T + s) 2 + g) 64 + l) 4 + d), T = K / 128 (per block: T x 2 KiB, contiguous)."""
    scales = np.asarray(scales, dtype=np.uint8)
    rows, nkb = scales.shape
    if sum(widths) != rows or nkb % 4:
        raise ValueError("widths must cover the rows; K/32 a multiple of 4")
    ns = nkb // 4
    out = []
    r0 = 0
    for w in widths:
        hg = w // 2
        nd = hg // 64
        x = scales[r0:r0 + w].reshape(2, nd, 4, 16, ns, 4)  # g d i r16 s kg
        t = x.transpose(4, 0, 5, 3, 1, 2)  # s g kg r16 d i -> lane = 16 kg + r16
        blk = np.zeros((ns, 2, 64, SB_DWORDS, 4), np.uint8)
        blk[:, :, :, :nd, :] = t.reshape(ns, 2, 64, nd, 4)
        out.append(blk.reshape(-1))
        r0 += w
    return np.ascontiguousarray(np.concatenate(out)).view(np.uint32)


def pack_b_operand(op: L.Operand, widths: List[int]) -> np.ndarray:
    """B codes: c0c1 blocks of the column widths, concatenated ([block][K/128][rows_c * 96])."""
    out, r0 = [], 0
    for w in widths:
        sub = L.Operand(w, op.k, op.codes[r0:r0 + w], op.scales[r0:r0 + w], op.fmt)
        out.append(L.pack_operand_c0c1(sub, w))
        r0 += w
    return np.ascontiguousarray(np.concatenate(out))


def host_images(spec: WideGemmSpec, a: L.Operand, b: L.Operand):
    """(A codes, B codes, A scales, B scales) device images of ``spec``."""
    w = col_widths(spec)
    return (L.pack_operand_c0c1(a, 256), pack_b_operand(b, w), L.pack_scales_tile(a.scales, "16x16x128"),
            pack_scales_b(b.scales, w))


def image_key(spec: WideGemmSpec) -> tuple:
    return (spec.m, spec.n, tuple(col_widths(spec)))


# ----------------------------------------------------------------- builder


def build_mxfp6_gemm_wide(spec: WideGemmSpec, *, arch: str = ARCH) -> KernelDef:
    if arch != ARCH:
        raise ValueError(f"wide mxfp6 GEMM is gfx950-only, got {arch!r}")
    ok, why = validate_spec(spec)
    if not ok:
        raise ValueError(f"invalid WideGemmSpec: {why}")
    return _Builder(spec).build()


class _Builder:
    def __init__(self, spec: WideGemmSpec):
        self.spec = spec
        self.b = IRBuilder(kernel_name(spec))
        self.T = spec.k // 128
        self.widths = col_widths(spec)
        self.q = n_wide(spec)  # wide columns
        self.vm_trace: List[Tuple[object, str]] = []

    def c32(self, v: int) -> Value:
        return self.b.const_i32(int(v))

    def sg(self, v: Value) -> Value:
        return self.b.to_sgpr_u32(v)

    def _stamp(self) -> None:
        """stamps probe: realtime (100 MHz, low 32 bits) appended to the role's stamp list."""
        if self.spec.probe == "stamps":
            t = self.b._op("tile.memrealtime", [], [I64], result_name_hint="rt").result
            self.st.append(self.b.vec_extract(self.b.bitcast(t, VectorType(I32, 2)), 0))

    def _stamps_store(self, rho: Value, code: int) -> None:
        """stamps probe: wave 0 lane 0 stores (code, n, stamps...) as 16 dwords at flags + 1024 + 64 rho."""
        if self.spec.probe != "stamps":
            return
        b, c32 = self.b, self.c32
        words = [c32(code), c32(len(self.st))] + self.st
        words += [c32(0)] * (16 - len(words))
        rsrc = b.buffer_rsrc(self.pflags, c32(0x7FFFFFF0))
        soff = self.sg(b.add(b.mul(rho, c32(STAMP_BYTES)), c32(STAMP_OFF)))
        first = b.land(b.cmp_eq(self.wave, c32(0)), b.cmp_eq(self.lane, c32(0)))
        with b.scf_if(first):
            for q4 in range(4):
                b.buffer_store_vN(rsrc, c32(16 * q4), soff, b.vec_pack(words[4 * q4:4 * q4 + 4], I32))

    def _fence(self, kind: str) -> None:
        self.b.sched_barrier(VM_FENCE_MASK)
        self.vm_trace.append((getattr(self, "_trace", None), kind))

    def build(self) -> KernelDef:
        b, spec = self.b, self.spec
        k = b.kernel
        k.attrs["max_workgroup_size"] = 256
        k.attrs["lds_alias_scopes"] = True
        k.attrs["agpr_alloc"] = (256, 256)
        k.attrs["mfma_agpr_form"] = True
        self.pa = b.param("a", PtrType(I8, "global"))
        self.pb = b.param("b", PtrType(I8, "global"))
        self.psa = b.param("sa", PtrType(I32, "global"))
        self.psb = b.param("sb", PtrType(I32, "global"))
        self.pc = b.param("c", PtrType(BF16, "global"))
        if spec.stream_k:
            self.pws = b.param("ws", PtrType(F32, "global"))
            self.pflags = b.param("flags", PtrType(I32, "global"))
        self._setup()
        fns = sorted({w // 2 // T16 for w in self.widths}, reverse=True)
        if spec.stream_k:
            self._stream_k()
        elif len(fns) == 1:
            self._path(fns[0])
        else:
            is_wide = b.cmp_lt(self.tc, self.c32(self.q))
            with b.scf_if_else(is_wide) as (then_ctx, else_ctx):
                with then_ctx:
                    self._path(fns[0])
                with else_ctx:
                    self._path(fns[1])
        b.ret()
        return k

    # ------------------------------------------------------------- setup (both paths)
    def _setup(self) -> None:
        b, spec = self.b, self.spec
        tid = b.thread_id_x()
        self.lane = b.lane_id()
        self.wave = self.sg(b.lshr(tid, self.c32(6)))
        self.wm = self.sg(b.div(self.wave, self.c32(2)))
        self.wn = self.sg(b.mod(self.wave, self.c32(2)))
        wg = b.block_id_x()
        gm, gc = spec.m // 256, len(self.widths)
        if spec.xcd_map == "none":
            tm, tc = b.div(wg, self.c32(gc)), b.mod(wg, self.c32(gc))
        else:
            x, loc = b.mod(wg, self.c32(NUM_XCDS)), b.div(wg, self.c32(NUM_XCDS))
            rb, h = gm // 4, gc // 2
            band, half = b.div(x, self.c32(2)), b.mod(x, self.c32(2))
            main = b.cmp_lt(loc, self.c32(rb * h))
            r_main = b.add(b.mul(band, self.c32(rb)), b.div(loc, self.c32(h)))
            c_lo = b.mod(loc, self.c32(h))
            c_main = b.add(c_lo, b.mul(half, self.c32(gc - h)))
            j = b.sub(loc, self.c32(rb * h))
            r_mid = b.add(b.add(b.mul(band, self.c32(rb)), b.mul(half, self.c32(rb // 2))), j)
            tm = b.select(main, r_main, r_mid)
            tc = b.select(main, c_main, self.c32(h))
        self.tm, self.tc = self.sg(tm), self.sg(tc)
        self.row0_a = self.sg(b.mul(self.tm, self.c32(256)))
        # LDS: A ring (256-row images + 1 KiB scales), B ring (widest images + 2 KiB scales)
        wb = max(self.widths)
        self.stg = {}
        for s in range(spec.stages_a):
            self.stg[("A", s)] = b.smem_alloc(I32, [256 * ROW_BYTES // 4], name_hint=f"stgA{s}")
            self.stg[("sA", s)] = b.smem_alloc(I32, [256], name_hint=f"sstA{s}")
        for s in range(spec.stages_b):
            self.stg[("B", s)] = b.smem_alloc(I32, [wb * ROW_BYTES // 4], name_hint=f"stgB{s}")
            self.stg[("sB", s)] = b.smem_alloc(I32, [2 * 64 * SB_DWORDS], name_hint=f"sstB{s}")
        lane = self.lane
        self.dtl_voff = b.mul(lane, self.c32(16))
        self.sc_dtl_voff = b.mul(lane, self.c32(4))
        self.sc_rsrc = {"A": b.buffer_rsrc(self.psa, self.c32(0x7FFFFFF0)),
                        "B": b.buffer_rsrc(self.psb, self.c32(0x7FFFFFF0))}
        self.nq_a = spec.m // 128
        self.sc_cta_q_a = self.sg(b.mul(self.tm, self.c32(2)))

    # ------------------------------------------------------------- one path
    def _path(self, fn: int) -> None:
        _Path(self, fn).run()

    def _stream_k(self) -> None:
        """Role decode (= :func:`sk_role`) and the three bodies."""
        b, spec, c32, sg = self.b, self.spec, self.c32, self.sg
        T, kh = self.T, k_head_of(spec)
        wg = b.block_id_x()
        rho = sg(b.add(b.mul(b.mod(wg, c32(NUM_XCDS)), c32(32)), b.div(wg, c32(NUM_XCDS))))
        with b.scf_if_else(b.cmp_lt(rho, c32(SK_HEADS))) as (head_ctx, rest_ctx):
            with head_ctx:
                self.st = []
                self._stamp()
                blk, loc = sg(b.div(rho, c32(32))), sg(b.mod(rho, c32(32)))
                lo = b.cmp_lt(blk, c32(4))
                tm = b.select(lo, b.add(b.mul(b.mod(blk, c32(2)), c32(8)), b.div(loc, c32(4))), b.div(loc, c32(2)))
                tc = b.select(lo, b.add(b.mul(b.div(blk, c32(2)), c32(4)), b.mod(loc, c32(4))),
                              b.add(c32(8), b.mod(loc, c32(2))))
                _Path(self, 12, tm=sg(tm), tc=sg(tc), k0=0, nseg=kh, epi="head").run()
                self._stamp()
                self._stamps_store(rho, 1)
            with rest_ctx:
                with b.scf_if_else(b.cmp_lt(rho, c32(SK_HEADS + SK_TAILS))) as (tail_ctx, nar_ctx):
                    with tail_ctx:
                        self.st = []
                        self._stamp()
                        j = sg(b.sub(rho, c32(SK_HEADS)))
                        blk, jj = sg(b.div(j, c32(32))), sg(b.mod(j, c32(32)))
                        lo = b.cmp_lt(blk, c32(2))
                        r = sg(b.select(lo, b.div(jj, c32(4)), b.div(jj, c32(2))))
                        c = sg(b.select(lo, b.add(b.mul(blk, c32(4)), b.mod(jj, c32(4))),
                                        b.add(c32(8), b.mod(jj, c32(2)))))
                        _Path(self, 12, tm=r, tc=c, k0=kh, nseg=T - kh, epi="partial").run()
                        b.s_waitcnt(lgkmcnt=0)
                        b.s_barrier_bare()  # every wave is past segment 1's LDS reads before segment 2 refills
                        _Path(self, 12, tm=sg(b.add(r, c32(8))), tc=c, k0=kh, nseg=T - kh, epi="finalize").run()
                        self._stamp()
                        self._stamps_store(rho, 2)
                    with nar_ctx:
                        self.st = []
                        self._stamp()
                        i = sg(b.sub(rho, c32(SK_HEADS + SK_TAILS)))
                        _Path(self, 8, tm=i, tc=c32(SK_WIDE_COLS), k0=0, nseg=T, epi="store_c").run()
                        self._stamp()
                        self._stamps_store(rho, 3)


class _Path:
    """The wide (fn = 12) or narrow (fn = 8) body for one CTA."""

    def __init__(self, W: _Builder, fn: int, *, tm: Optional[Value] = None, tc: Optional[Value] = None, k0: int = 0,
                 nseg: Optional[int] = None, epi: str = "store_c"):
        self.W, self.b, self.spec = W, W.b, W.spec
        self.fn = fn
        self.pp = path_plan(W.spec, fn)
        self.rows = {"A": 256, "B": self.pp.rows_b}
        self.T = nseg or W.T  # K tiles of this segment (the loop)
        self.k0 = k0  # first K tile of the segment
        self.epi = epi
        spec, b = self.spec, self.b
        c32 = W.c32
        nT = W.T  # K tiles of the whole problem (image addressing)
        q = W.q
        tm = W.tm if tm is None else tm
        tc = W.tc if tc is None else tc
        self.tm, self.tc = tm, tc
        self.row0_a = W.sg(b.mul(tm, c32(256)))
        self.sc_cta_q_a = W.sg(b.mul(tm, c32(2)))
        self.tile_id = W.sg(b.add(b.mul(tm, c32(SK_WIDE_COLS)), tc)) if spec.stream_k else None
        # B column block of this CTA: byte offsets of its c0c1 image and its scale block, first output column
        if fn * 2 * T16 == 256 and q:  # narrow path of a mixed grid
            j = b.sub(tc, c32(q))
            col0 = b.add(c32(q * spec.tile_n), b.mul(j, c32(256)))
            bbase = b.add(c32(q * nT * spec.tile_n * ROW_BYTES), b.mul(j, c32(nT * 256 * ROW_BYTES)))
        else:
            col0 = b.mul(tc, c32(self.pp.rows_b))
            bbase = b.mul(tc, c32(nT * self.pp.rows_b * ROW_BYTES))
        self.col0 = W.sg(col0)
        abase = b.mul(tm, c32(nT * 256 * ROW_BYTES))
        self.base_ptr = {"A": b.global_ptr_add(W.pa, abase), "B": b.global_ptr_add(W.pb, bbase)}
        self.rsrc = {X: b.buffer_rsrc(self.base_ptr[X], c32(0x7FFFFFF0)) for X in ("A", "B")}
        self.tile_bytes = {X: self.rows[X] * ROW_BYTES for X in ("A", "B")}
        self.sc_b_blk = W.sg(b.mul(tc, c32(nT * 2048)))
        # per-wave DTL base (bytes within the stage image) and wave sub-tile row offsets
        self.dtl_base = {X: W.sg(b.mul(W.wave, c32(self.rows[X] * ROW_BYTES // 4))) for X in ("A", "B")}
        self.wrow0 = {"A": W.sg(b.mul(W.wm, c32(128))), "B": W.sg(b.mul(W.wn, c32(fn * T16)))}
        self.wrb = {X: b.div(self.wrow0[X], c32(16)) for X in ("A", "B")}
        self.sc_dtl_w = W.sg(b.mul(W.wave, c32(256)))
        if spec.k_phase:
            odd = b.land(b.add(tm, tc), c32(1))
            self.k_off = W.sg(b.mul(odd, c32(spec.k_phase)))

    # ---------------------------------------------------------- building blocks
    def _dtl_closure(self, X: str, kind: str, j: int, tile: Value, slot: int) -> Callable[[], None]:
        b, W, c32 = self.b, self.W, self.W.c32
        lds_key = (X if kind == "op" else "s" + X, slot)

        if self.spec.k_phase:  # rotated K walk: tile -> (tile + off) mod T
            t2 = b.add(tile, self.k_off)
            tile = b.select(b.cmp_lt(t2, c32(self.T)), t2, b.sub(t2, c32(self.T)))
        if self.k0:  # a stream-K segment starts at K tile k0
            tile = b.add(tile, c32(self.k0))

        def issue():
            lds0 = b.smem_addr_of(W.stg[lds_key])
            if kind == "op":
                kbyte = b.mul(tile, c32(self.tile_bytes[X]))
                lconst = j * 1024
                sbase = b.add(self.dtl_base[X], kbyte)
                if self.spec.dtl_imm:
                    blk, imm = (lconst // 4096) * 4096, lconst % 4096
                    lds = b.smem_ptr_add(lds0, b.zext(b.add(self.dtl_base[X], c32(blk)), I64))
                    b.async_buffer_load_lds_imm(self.rsrc[X], lds, W.dtl_voff, b.add(sbase, c32(blk)), 4, imm)
                else:
                    lds = b.smem_ptr_add(lds0, b.zext(b.add(self.dtl_base[X], c32(lconst)), I64))
                    b.async_buffer_load_lds_addr(self.rsrc[X], lds, W.dtl_voff, b.add(sbase, c32(lconst)), 4)
            else:
                off = b.add(self.sc_dtl_w, c32(j * 1024))  # wave w copies bytes [256 (4 j + w), +256)
                lds = b.smem_ptr_add(lds0, b.zext(off, I64))
                if X == "A":
                    base = b.mul(b.add(b.mul(tile, c32(W.nq_a)), self.sc_cta_q_a), c32(512))
                else:
                    base = b.add(self.sc_b_blk, b.mul(tile, c32(2048)))
                b.async_buffer_load_lds_addr(W.sc_rsrc[X], lds, W.sc_dtl_voff, b.add(base, off), 1)
            W._fence(f"{X}{kind}")

        return issue

    def _frag(self, X: str, slot: int, f: int) -> Value:
        """c0c1 fragment ``f`` of the wave's operand ``X`` from ring slot ``slot``."""
        b, c32 = self.b, self.W.c32
        smem = self.W.stg[(X, slot)]
        rb = b.add(self.wrb[X], c32(f))
        o = b.add(b.mul(rb, c32(64)), self.W.lane)
        v0 = b.smem_load_vN(smem, b.mul(o, c32(4)), dtype=I32, n=4)
        b.sched_barrier(0)
        v1 = b.smem_load_vN(smem, b.add(b.mul(o, c32(2)), c32(self.rows[X] * 16)), dtype=I32, n=2)
        parts = [b.vec_extract(v0, e) for e in range(4)] + [b.vec_extract(v1, e) for e in range(2)]
        return b.vec_pack(parts, I32)

    def _scales(self, a_slot: int, b_slot: int) -> Tuple[Value, Value]:
        b, c32, W = self.b, self.W.c32, self.W
        ia = b.add(b.mul(W.wm, c32(128)), b.mul(W.lane, c32(2)))
        sa = b.smem_load_vN(W.stg[("sA", a_slot)], ia, dtype=I32, n=2)
        ib = b.add(b.mul(W.wn, c32(64 * SB_DWORDS)), b.mul(W.lane, c32(SB_DWORDS)))
        nd = self.fn // 4  # B scale dwords this path uses (2 narrow, 3 wide)
        if not self.spec.sc_exact:
            sb = b.smem_load_vN(W.stg[("sB", b_slot)], ib, dtype=I32, n=4)
        elif nd == 2:
            sb = b.smem_load_vN(W.stg[("sB", b_slot)], ib, dtype=I32, n=2)
        else:
            lo = b.smem_load_vN(W.stg[("sB", b_slot)], ib, dtype=I32, n=2)
            hi = b.smem_load_vN(W.stg[("sB", b_slot)], b.add(ib, c32(2)), dtype=I32, n=1)
            sb = b.vec_pack([b.vec_extract(lo, 0), b.vec_extract(lo, 1), b.vec_extract(hi, 0)], I32)
        return sa, sb

    def _mfma(self, acc, fa, fb, sa, sb, ja, jb):
        b, spec = self.b, self.spec
        fmt = ELEM_FMTS[spec.elem_fmt]
        if spec.mutant == "wrong_fmt":
            fmt = "bf6" if fmt == "fp6" else "fp6"
        return b.mfma_scale_mx(fa, fb, acc, sa, sb, shape="16x16x128", fmt_a=fmt, fmt_b=fmt, opsel_a=ja, opsel_b=jb)

    # ---------------------------------------------------------- one K tile
    def _tile(self, t_dyn: Value, t_st: Optional[int], pos: int, acc: List[Value], cur: Dict, wait: int):
        """Emit K tile t (``pos`` = static index mod the loop body). ``cur``: this tile's B fragments ("B", c),
        A(t)[0] ("A0") and scales ("sA", "sB"), read in the previous tile. Returns (acc, next tile's ``cur``)."""
        b, spec, pp, W, c32 = self.b, self.spec, self.pp, self.W, self.W.c32
        fn, n, T = self.fn, pp.n_mfma, self.T
        W._trace = t_st if t_st is not None else ("loop", pos)
        parity = pos % 2
        h = pp.h
        order = mfma_order(fn, parity, h)
        a_slot = pos % spec.stages_a
        if spec.mutant == "k_order" and pos == 1:
            a_slot = (a_slot + 1) % spec.stages_a
        a_slot_nx = (pos + 1) % spec.stages_a
        b_slot_nx = (pos + 1) % spec.stages_b
        has_next = t_st is None or t_st + 1 < T
        pre: Dict[int, List[Callable[[], None]]] = {}
        nxt: Dict = {}
        afr: Dict[int, Value] = {i: cur[f"A{i}"] for i in range(h)}

        def acquire():
            b.s_waitcnt(vmcnt=wait)
            b.s_barrier_bare()
            nxt["sA"], nxt["sB"] = self._scales(a_slot_nx, b_slot_nx)

        if has_next:
            pre.setdefault(pp.q, []).append(acquire)
        if pp.r >= 0 and (t_st is None or any(t_st + pp.d[X] < T for X in ("A", "B"))):
            pre.setdefault(pp.r, []).append(lambda: b.s_barrier_bare())
        # A stream: A(t)[mi] at the start of the unit before row mi's unit
        for mi in range(h, FM):

            def rd_a(mi=mi):
                afr[mi] = self._frag("A", a_slot, mi)

            pre.setdefault(pp.a_read[mi], []).append(rd_a)
        if has_next:
            # B(t+1)[c] right after the last MFMA using B(t)[c] (the last row); A(t+1)[0..h-1] at a0_next
            for x in range(n - fn, n):
                c = order[x][1]

                def rd_b(c=c):
                    nxt[("B", c)] = self._frag("B", b_slot_nx, c)

                pre.setdefault(x, []).append(rd_b)

            def rd_a0():
                for i in range(h):
                    nxt[f"A{i}"] = self._frag("A", a_slot_nx, i)

            pre.setdefault(pp.a0_next, []).append(rd_a0)
        # refill DTLs
        for slot, X, kind, j in pp.dtl:
            if t_st is not None and t_st + pp.d[X] >= T:
                continue
            tgt = c32(t_st + pp.d[X]) if t_st is not None else b.add(t_dyn, c32(pp.d[X]))
            ring = (pos + pp.d[X]) % (spec.stages_a if X == "A" else spec.stages_b)
            pre.setdefault(slot, []).append(self._dtl_closure(X, kind, j, tgt, ring))
        for f in pre.get(-1, ()):
            f()
        if spec.setprio:
            b.sched_barrier(0)
            b.s_setprio(spec.setprio)
        acc = list(acc)
        drop = spec.mutant == "drop_kstep" and t_st == T - 1
        sa_v, sb_v = cur["sA"], cur["sB"]
        for x, (mi, c) in enumerate(order):
            if not drop:
                ja, jb = mi % 4, c % 4
                if spec.mutant == "scale_sel":
                    ja = 0 if ja == 1 else ja
                    jb = 0 if jb == 1 else jb
                sa = b.vec_extract(sa_v, mi // 4)
                sb = b.vec_extract(sb_v, c // 4)
                acc[mi * fn + c] = self._mfma(acc[mi * fn + c], afr[mi], cur[("B", c)], sa, sb, ja, jb)
            for f in pre.get(x, ()):
                f()
            b.sched_barrier(0)
        if spec.setprio:
            b.s_setprio(0)
            b.sched_barrier(0)
        if not has_next:
            return acc, cur
        return acc, nxt

    # ---------------------------------------------------------- whole path
    def run(self) -> None:
        b, spec, pp, W, c32 = self.b, self.spec, self.pp, self.W, self.W.c32
        fn, T = self.fn, self.T
        _, counts = vm_events(pp, T)
        self.counts = counts
        if spec.probe == "mfma_only":
            self._mfma_only()
            self._epilogue()
            return
        # prologue: fills of the first d_X tiles, acquire tile 0, read its B fragments, A(0)[0], scales
        W._trace = -1
        for u in range(max(pp.d.values())):
            for X in ("A", "B"):
                if u < pp.d[X]:
                    for slot, X2, kind, j in pp.dtl:
                        if X2 == X:
                            self._dtl_closure(X, kind, j, c32(u), u % (spec.stages_a if X == "A" else spec.stages_b))()
        b.s_waitcnt(vmcnt=counts[0])
        b.s_barrier_bare()
        cur: Dict = {}
        cur["sA"], cur["sB"] = self._scales(0, 0)
        for c in range(fn):
            cur[("B", c)] = self._frag("B", 0, c)
        for i in range(pp.h):
            cur[f"A{i}"] = self._frag("A", 0, i)
        zero = b.zero_vec_f32(4)
        acc = [zero for _ in range(FM * fn)]
        U = pp.U
        maxd = max(pp.d.values())
        L_ = max(0, (T - maxd) // U)
        body_wait = {}
        for p in range(U):
            vals = [counts[j * U + p + 1] for j in range(L_) if j * U + p + 1 < T]
            body_wait[p] = min(vals) if vals else 0
        keys = sorted(cur.keys(), key=str)
        if L_ > 0:
            iter_args = [(f"acc{i}", v) for i, v in enumerate(acc)] + [(f"cur{i}", cur[kk]) for i, kk in enumerate(keys)]
            na = len(acc)
            loop = b.scf_for_iter(c32(0), c32(L_), c32(1), iter_args, iv_name="kt", elide_trailing_barrier=False)
            with loop as (iv, vs):
                acc_l = list(vs[:na])
                cur_l = {kk: vs[na + i] for i, kk in enumerate(keys)}
                i0 = b.mul(iv, c32(U))
                for p in range(U):
                    t_dyn = b.add(i0, c32(p)) if p else i0
                    acc_l, cur_l = self._tile(t_dyn, None, p, acc_l, cur_l, body_wait[p])
                b.scf_yield(*(acc_l + [cur_l[kk] for kk in keys]))
            res = loop.results
            acc = list(res[:na])
            cur = {kk: res[na + i] for i, kk in enumerate(keys)}
        for t in range(L_ * U, T):
            acc, cur = self._tile(c32(t), t, t % U, acc, cur, counts.get(t + 1, 0))
        self.acc = acc
        W._stamp()
        if self.epi == "store_c":
            self._epilogue()
        elif self.epi == "partial":
            self._write_partial()
        elif self.epi == "finalize":
            self._finalize()
        else:  # head: rows 0-7 finalize (their tail partial landed mid-run), rows 8-15 hand theirs to the tail
            with b.scf_if_else(b.cmp_lt(self.tm, c32(SK_ROWS // 2))) as (fin_ctx, part_ctx):
                with fin_ctx:
                    self._finalize()
                with part_ctx:
                    self._write_partial()

    # ---------------------------------------------------------- stream-K partial handoff
    def _ws(self):
        b, W, c32 = self.b, self.W, self.W.c32
        rsrc = b.buffer_rsrc(W.pws, c32(0x7FFFFFF0))
        base = W.sg(b.mul(b.add(b.mul(self.tile_id, c32(4)), W.wave), c32(96 * 1024)))
        return rsrc, base, b.mul(W.lane, c32(16))

    def _write_partial(self) -> None:
        """fp32 accumulators -> workspace (lane-native, 1 KiB per accumulator and wave), release, flag = 1."""
        b, W, c32 = self.b, self.W, self.W.c32
        b.s_waitcnt(vmcnt=0)
        rsrc, base, voff = self._ws()
        for i, v in enumerate(self.acc):
            b.buffer_store_vN(rsrc, voff, W.sg(b.add(base, c32(i * 1024))), v)
        if self.epi == "partial":  # stamps (not inside the head's branch)
            b.s_waitcnt(vmcnt=0)
            W._stamp()
        b.fence(scope="agent", ordering="release")
        if self.epi == "partial":
            W._stamp()
        b.s_barrier_bare()  # every wave's part is released before the flag
        b.global_flag_store(W.pflags, self.tile_id, c32(1), scope="agent", ordering="release")

    def _finalize(self) -> None:
        """Wait for the other segment's partial, add it, store C, reset the flag."""
        b, W, c32 = self.b, self.W, self.W.c32
        b.global_flag_wait_eq(W.pflags, self.tile_id, c32(1), scope="agent")
        if self.epi == "finalize":  # (not inside the head's branch: its values could not leave it)
            W._stamp()
        self._epilogue(partial=self._ws())  # (self.acc stays: the head's other branch stores the K-loop values)
        b.s_waitcnt(vmcnt=0)
        b.s_barrier_bare()  # every wave read the partial
        b.global_flag_store(W.pflags, self.tile_id, c32(0), scope="agent", ordering="release")

    def _mfma_only(self) -> None:
        """Probe: the tile's MFMA stream on register-resident tile-0 operands and scales (energy ceiling)."""
        b, spec, W, c32 = self.b, self.spec, self.W, self.W.c32
        fn, T = self.fn, self.T
        for slot, X, kind, j in self.pp.dtl:
            self._dtl_closure(X, kind, j, c32(0), 0)()
        b.s_waitcnt(vmcnt=0)
        b.s_barrier_bare()
        sa_v, sb_v = self._scales(0, 0)
        fa = [self._frag("A", 0, mi) for mi in range(FM)]
        fbv = [self._frag("B", 0, c) for c in range(fn)]
        zero = b.zero_vec_f32(4)
        acc = [zero for _ in range(FM * fn)]
        loop = b.scf_for_iter(c32(0), c32(T // 2), c32(1), [(f"acc{i}", v) for i, v in enumerate(acc)], iv_name="kt",
                              elide_trailing_barrier=False)
        with loop as (iv, vs):
            acc_l = list(vs)
            for p in range(2):
                for mi, c in mfma_order(fn, p, self.pp.h):
                    acc_l[mi * fn + c] = self._mfma(acc_l[mi * fn + c], fa[mi], fbv[c], b.vec_extract(sa_v, mi // 4),
                                                    b.vec_extract(sb_v, c // 4), mi % 4, c % 4)
                    b.sched_barrier(0)
            b.scf_yield(*acc_l)
        self.acc = list(loop.results)

    # ---------------------------------------------------------- epilogue (A is MFMA src0)
    def _epilogue(self, partial=None) -> None:
        """A lane holds C[m0 + 16 mi + 4 grp + (0..3)][n0 + 16 c + rr]; lanes rr, rr ^ 1 swap halves with a DPP
        xor-1 so every store is one dword = (even column, odd column) bf16 of one row.

        ``partial`` (stream-K finalize: (rsrc, soffset base, voffset) of the other segment's fp32 partial): each
        accumulator gets its partial added just before its stores; the partial loads run one group of 8
        accumulators ahead, every VM op in program order (VM fences), so the waitcnt pass keeps the previous
        group's stores in flight and the register file never holds more than two groups of partials."""
        b, spec, W, c32 = self.b, self.spec, self.W, self.W.c32
        b.s_waitcnt(vmcnt=0)
        G, loaded = 8, {}

        def load_group(g):
            if partial is None or g * G >= len(self.acc) or g * G in loaded:
                return
            prs, pbase, pvoff = partial
            for i in range(g * G, min(g * G + G, len(self.acc))):
                loaded[i] = b.buffer_load_vN(prs, pvoff, W.sg(b.add(pbase, c32(i * 1024))), F32, 4)
                W._fence("partial")

        def value(i):
            v = self.acc[i]
            if partial is None:
                return v
            if i % G == 0:
                load_group(i // G)
                load_group(i // G + 1)
            p_ = loaded.pop(i)
            return b.vec_pack([b.fadd(b.vec_extract(v, e), b.vec_extract(p_, e)) for e in range(4)], F32)

        lane = W.lane
        rr = b.mod(lane, c32(16))
        grp = b.div(lane, c32(16))
        n_bytes = spec.n * 2
        rsrc = b.buffer_rsrc(W.pc, c32(0x7FFFFFF0))
        m0 = b.add(self.row0_a, self.wrow0["A"])
        n0 = b.add(self.col0, self.wrow0["B"])
        par = b.land(rr, c32(1))
        odd = b.cmp_eq(par, c32(1))
        row0 = b.add(b.mul(grp, c32(4)), b.mul(par, c32(2)))
        col_b = b.mul(b.sub(rr, par), c32(2))
        lo_m, hi_m = c32(0xFFFF), c32(-65536)
        for mi in range(FM):
            soff = W.sg(b.add(b.mul(b.add(m0, c32(mi * T16)), c32(n_bytes)), b.mul(n0, c32(2))))
            for c in range(self.fn):
                v = value(mi * self.fn + c)
                h = b.vec_bitcast(b.vec_trunc_f32_to_bf16(v), VectorType(I32, 2))
                d0, d1 = b.vec_extract(h, 0), b.vec_extract(h, 1)
                x0, x1 = b.dpp_xor(d0, 1), b.dpp_xor(d1, 1)
                lo = b.select(odd, x1, d0)
                hi = b.select(odd, d1, x0)
                words = (b.lor(b.land(lo, lo_m), b.shl(hi, c32(16))), b.lor(b.lshr(lo, c32(16)), b.land(hi, hi_m)))
                for jj, w in enumerate(words):
                    off = b.add(b.add(b.mul(b.add(row0, c32(jj)), c32(n_bytes)), col_b), c32(c * T16 * 2))
                    b.buffer_store_vN(rsrc, off, soff, b.vec_pack([w], I32))
                    if partial is not None:
                        W._fence("cstore")

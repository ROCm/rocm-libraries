# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The gfx950 MXFP6 GEMM tile-config library: configs, work model and benchmark problems.

Pure Python, no GPU, no measured numbers (measured ceilings / bandwidths are inputs: see ``libbench``).

* :data:`LIBRARY` -- every library tile ``MT0 x MT1 x DepthU`` in hipBLASLt / TensileLite terms: MT0 tiles free
  index 0 = hipBLASLt m = the pre-shuffled weights (our N, ``Mxfp6GemmSpec.tile_n``), MT1 tiles free index 1 =
  hipBLASLt n = the activations / tokens (our M, ``tile_m``); DepthU = K per main-loop iteration.
* :data:`FAMILIES` -- the two kernel families per tile, keyed by the activation operand (hipBLASLt opB):
  activation K-contiguous (``ak``: NN / TN) and activation M-contiguous (``am``: NT / TT). opA (the weights) only
  changes the offline pre-shuffle, never the kernel.
* :func:`wave_layouts` / :func:`default_layout` -- wave decompositions of a tile (MFMA atom, waves_m x waves_n).
* :func:`work` -- the per-work-group work model (FLOP, global / LDS bytes per DepthU, register and LDS estimates);
  :func:`roofline` combines it with measured per-clock bandwidths into per-level bounds.
* :func:`problems` -- the benchmark problem sets (strided-batched, chip filling).

Notation: ``B_CODE = 6/8`` bytes per element, ``B_SCALE = 1/32`` bytes per element (one E8M0 byte per 32 along K).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

# ------------------------------------------------------------------ the library (user spec 2026-11-11)

_DU256 = (
    "32x128 32x256 32x384 32x512 64x64 64x128 64x192 64x256 64x320 64x384 64x448 64x512 96x128 96x256 96x384 "
    "128x32 128x64 128x96 128x128 128x192 128x224 128x256 128x320 128x384 128x448 160x256 160x384 192x64 192x128 "
    "192x192 192x256 192x320 192x384 224x256 256x32 256x64 256x96 256x128 256x160 256x192 256x256 256x288 256x320 "
    "288x256 320x64 320x128 320x192 320x256 384x32 384x64 384x96 384x128 384x192 448x64 448x128 512x32 512x64"
)
_DU512 = "32x32 32x128 64x64 64x128 64x192 96x128 128x32 128x64 128x96 128x128 128x160 160x128 192x64"

FORMATS = ("e2m3", "e3m2")
#: kernel family -> (BLAS transposes it serves, activation contiguous dimension)
FAMILIES: Dict[str, Tuple[Tuple[str, ...], str]] = {"ak": (("NN", "TN"), "K"), "am": (("NT", "TT"), "M")}

B_CODE = 6.0 / 8.0
B_SCALE = 1.0 / 32.0
B_ELEM = B_CODE + B_SCALE  # 0.78125 bytes of A / B traffic per element (codes + E8M0 scales)
B_OUT = 2.0  # bf16 C

# gfx950 (MI355X) machine constants (architectural, not measured)
N_CU = 256
N_XCD = 8
SIMD_PER_CU = 4
LDS_BYTES = 160 * 1024
REGS_PER_LANE = 512  # VGPR + AGPR file per lane at 1 wave / SIMD
AGPR_MAX = 256
FLOP_CLK_CU = 16384  # dense MXFP6 (f8f6f4 scaled MFMA) FLOP per clock per CU
MALL_BYTES = 256 * 1024 * 1024

ATOMS = {  # name -> (m, n, k, accumulator regs per lane, fragment regs per lane (one operand, MXFP6))
    "16x16x128": (16, 16, 128, 4, 6),
    "32x32x64": (32, 32, 64, 16, 6),
}


@dataclass(frozen=True, order=True)
class Tile:
    mt0: int
    mt1: int
    du: int

    @property
    def name(self) -> str:
        return f"{self.mt0}x{self.mt1}x{self.du}"

    @property
    def tile_m(self) -> int:  # token / activation rows (our M) = MT1
        return self.mt1

    @property
    def tile_n(self) -> int:  # weight rows (our N) = MT0
        return self.mt0

    @classmethod
    def parse(cls, s: str) -> "Tile":
        a, b, c = (int(x) for x in s.lower().split("x"))
        return cls(a, b, c)


LIBRARY: Tuple[Tile, ...] = tuple(
    [Tile(*map(int, t.split("x")), 256) for t in _DU256.split()] + [Tile(*map(int, t.split("x")), 512) for t in _DU512.split()]
)


# ------------------------------------------------------------------ wave layouts and the work model


@dataclass(frozen=True)
class Layout:
    atom: str = "16x16x128"
    waves_m: int = 2
    waves_n: int = 2

    @property
    def waves(self) -> int:
        return self.waves_m * self.waves_n

    @property
    def name(self) -> str:
        return f"{self.atom}_w{self.waves_m}x{self.waves_n}"


def layout_ok(t: Tile, lay: Layout) -> bool:
    am, an, ak, _, _ = ATOMS[lay.atom]
    return t.tile_m % (lay.waves_m * am) == 0 and t.tile_n % (lay.waves_n * an) == 0 and t.du % ak == 0


def wave_layouts(t: Tile, atoms: Iterable[str] = ("16x16x128",), waves: Iterable[int] = (1, 2, 4, 8)) -> List[Layout]:
    out = []
    for a in atoms:
        for w in waves:
            for wm in (1, 2, 4, 8):
                if w % wm:
                    continue
                lay = Layout(a, wm, w // wm)
                if layout_ok(t, lay):
                    out.append(lay)
    return out


@dataclass
class Work:
    """Per work-group work of one DepthU iteration (and the per-wave register tile)."""

    tile: str
    layout: str
    atom: str
    waves: int
    tm: int  # MFMA atoms per wave along M
    tn: int  # ... along N
    ks: int  # MFMA K steps per DepthU
    flop: float  # per WG per DepthU
    glob_bytes: float  # global -> LDS (A + B codes + scales) per WG per DepthU
    lds_read_bytes: float  # LDS -> VGPR per WG per DepthU (every wave reads its own A rows and B columns)
    acc_regs: int  # accumulator registers per lane
    frag_regs: int  # one K step of A + B fragments per lane
    vgpr_est: int  # estimate: one K step of fragments + 4 read-ahead fragments + 48 address / scale / loop registers
    lds_est: int  # estimate: LDS ring of (DepthU + 128) K (one 128-K tile of prefetch), else DepthU
    occ_est: int  # WGs per CU from lds_est and the register estimate (0: does not fit)
    ai_glob: float  # FLOP per global byte
    ai_lds: float  # FLOP per LDS-read byte
    mfma_clk_wave: float  # MFMA issue clocks per wave per DepthU at the spec rate (16x16x128: 16 clk, 32x32x64: 32)


def work(t: Tile, lay: Layout) -> Work:
    am, an, ak, acc_per, frag_per = ATOMS[lay.atom]
    tm, tn, ks = t.tile_m // (lay.waves_m * am), t.tile_n // (lay.waves_n * an), t.du // ak
    flop = 2.0 * t.mt0 * t.mt1 * t.du
    glob = (t.mt0 + t.mt1) * t.du * B_ELEM
    lds_read = (lay.waves_n * t.tile_m + lay.waves_m * t.tile_n) * t.du * B_ELEM
    acc = tm * tn * acc_per
    frag = (tm + tn) * frag_per
    vgpr = frag + 4 * frag_per + 48
    lds = int((t.du + 128) * (t.mt0 + t.mt1) * B_ELEM)
    if lds > LDS_BYTES:
        lds = int(t.du * (t.mt0 + t.mt1) * B_ELEM)
    regs = acc + vgpr  # accumulators beyond the AGPR file live in VGPRs (MFMA srcC / vdst take either file)
    waves_simd = 0 if regs > REGS_PER_LANE else REGS_PER_LANE // (8 * math.ceil(regs / 8))
    occ_regs = (waves_simd * SIMD_PER_CU) // lay.waves
    occ_lds = LDS_BYTES // lds if lds <= LDS_BYTES else 0
    occ = min(occ_regs, occ_lds, 8)
    clk = 16 if lay.atom == "16x16x128" else 32
    return Work(t.name, lay.name, lay.atom, lay.waves, tm, tn, ks, flop, glob, lds_read, acc, frag, vgpr, lds, occ,
                flop / glob, flop / lds_read, tm * tn * ks * clk)


def default_layout(t: Tile, atom: str = "16x16x128") -> Layout:
    """The emulation default: the fewest of 4 / 8 waves whose registers fit (occupancy >= 1 WG per CU), then the
    split with the least LDS read traffic (ties: the squarer per-wave tile)."""
    best = None
    for w in (4, 8):
        cands = [lay for lay in wave_layouts(t, (atom,), (w,)) if work(t, lay).occ_est >= 1]
        if not cands:
            continue
        for lay in cands:
            wk = work(t, lay)
            key = (wk.lds_read_bytes, abs(math.log2(wk.tm / wk.tn)))
            if best is None or key < best[0]:
                best = (key, lay)
        if best:
            return best[1]
    raise ValueError(f"no wave layout for {t.name} with atom {atom}")


def candidate_layouts(t: Tile) -> List[Layout]:
    """Layouts the ceiling sweep measures for tile ``t``: the default; for tiles whose default per-wave tile has
    fewer than 4 accumulator atoms, the 1- and 2-wave layouts (small per-wave tiles may not saturate the MFMA pipe);
    for tiles whose default per-wave accumulators exceed the AGPR file, the 8-wave layouts that fit; the 32x32x64
    atom's default where it divides."""
    d = default_layout(t)
    out = [d]
    wd = work(t, d)
    if wd.tm * wd.tn < 4:
        for lay in wave_layouts(t, ("16x16x128",), (1, 2)):
            if work(t, lay).occ_est >= 1 and lay not in out:
                out.append(lay)
    if wd.acc_regs > AGPR_MAX:
        for lay in wave_layouts(t, ("16x16x128",), (8,)):
            if work(t, lay).occ_est >= 1 and lay not in out:
                out.append(lay)
    try:
        d32 = default_layout(t, "32x32x64")
        if d32 not in out:
            out.append(d32)
    except ValueError:
        pass
    return out


def family_key(w: Work, occ: int) -> str:
    """Ceiling-probe family: per-wave register tile + K steps + waves per WG + WGs per CU (what the MFMA-only probe
    emulates). Tiles with the same key share an MFMA-only ceiling."""
    return f"{w.atom}_t{w.tm}x{w.tn}_k{w.ks}_w{w.waves}_o{occ}"


# ------------------------------------------------------------------ roofline


def roofline(w: Work, bw: Dict[str, float], occ: int, problem: Optional["Problem"] = None) -> Dict[str, float]:
    """Per-level throughput bounds as a fraction of the per-clock MFMA spec rate (1.0 = 16384 FLOP/clk/CU).

    ``bw``: measured per-CU bytes per clock, keys ``lds`` (LDS -> VGPR), ``l2`` (L2 -> LDS / VGPR, all CUs streaming
    L2-resident data), ``mall`` and ``hbm`` (chip bytes per second / (N_CU * clock of that measurement)); and
    ``clk_ghz`` of each (``*_clk``) to convert chip-level rates.
    The MALL / HBM bounds use the problem's traffic: ``fabric`` = every XCD fetches the A / B panels of the tiles it
    holds at once (``occ`` WGs on each of its 32 CUs, as a square-ish block), ``hbm`` = the problem's unique bytes."""
    out = {}
    out["lds"] = bw["lds"] * w.ai_lds / FLOP_CLK_CU
    out["l2"] = bw["l2"] * w.ai_glob / FLOP_CLK_CU
    if problem is not None:
        wg_xcd = (N_CU // N_XCD) * max(occ, 1)
        tt = Tile.parse(w.tile)
        tm0, tm1, du = tt.tile_m, tt.tile_n, tt.du  # M (token) and N (weight) tile extents
        gm, gn = problem.m // tm0, problem.n // tm1
        # square-ish block of tiles per XCD inside one batch entry (>= 1 x 1)
        best = None
        for bm in range(1, wg_xcd + 1):
            if wg_xcd % bm:
                continue
            bn = wg_xcd // bm
            bm_, bn_ = min(bm, gm), min(bn, gn)
            nblk = math.ceil(wg_xcd / (bm_ * bn_))  # blocks in flight (several batch entries when one is small)
            byt = nblk * (bm_ * tm0 + bn_ * tm1) * du * B_ELEM
            if best is None or byt < best:
                best = byt
        fl = wg_xcd * w.flop
        ai_fab = fl / best
        out["mall"] = bw["mall"] * ai_fab / FLOP_CLK_CU
        ai_hbm = problem.flop / problem.unique_bytes
        out["hbm"] = bw["hbm"] * ai_hbm / FLOP_CLK_CU
        out["ai_fabric"] = ai_fab
        out["ai_hbm"] = ai_hbm
    out["ai_l2"] = w.ai_glob
    out["ai_lds"] = w.ai_lds
    return out


# ------------------------------------------------------------------ benchmark problems


@dataclass
class Problem:
    """A strided-batched GEMM: C[b] (m x n, bf16) = A[b] (m x k) * B[b]^T (n x k), MXFP6 in, fp32 accumulation."""

    set: str
    tile: str
    batch: int
    m: int
    n: int
    k: int
    tiles: int = 0
    rotate: int = 1  # distinct input sets cycled by the warm loop (footprint >= 2 x MALL)
    note: str = ""

    @property
    def flop(self) -> float:
        return 2.0 * self.batch * self.m * self.n * self.k

    @property
    def in_bytes(self) -> float:
        return self.batch * (self.m + self.n) * self.k * B_ELEM

    @property
    def out_bytes(self) -> float:
        return self.batch * self.m * self.n * B_OUT

    @property
    def unique_bytes(self) -> float:
        return self.in_bytes + self.out_bytes

    @property
    def name(self) -> str:
        return f"{self.set}:{self.batch}x{self.m}x{self.n}x{self.k}"


#: tiles per launch of the fill sets: a multiple of 256 CUs x {1, 2, 4, 8} WGs per CU (no partial last wave),
#: >= 4 waves of the chip at up to 4 WGs per CU
FILL_TILES = 4096
K_FILL = 8192
K_SWEEP = (4096, 8192, 16384)


def _pow2_near(x: float) -> int:
    return max(1, 2 ** round(math.log2(max(x, 1.0))))


def _rotate(p: Problem) -> int:
    return max(1, math.ceil(2 * MALL_BYTES / p.in_bytes))


def problems(t: Tile, k: int = K_FILL, tiles: int = FILL_TILES) -> List[Problem]:
    """The benchmark problem sets of tile ``t``.

    * ``S`` (fill-square, the headline): per-batch M, N ~ 2048 (power-of-two tile counts per batch), K = 8192,
      batch = ``tiles`` / tiles-per-batch -> exactly ``tiles`` tiles.
    * ``G`` (fill-grouped, MoE-like): one token tile per batch entry (M = MT1 tokens per expert), N ~ 4096 weight
      rows, K = 8192: the grouped / expert regime where the weight stream dominates (bandwidth-bound for small MT1
      by construction).
    """
    out = []
    gm, gn = _pow2_near(2048 / t.tile_m), _pow2_near(2048 / t.tile_n)
    while gm * gn > tiles:
        if gm >= gn:
            gm //= 2
        else:
            gn //= 2
    p = Problem("S", t.name, tiles // (gm * gn), gm * t.tile_m, gn * t.tile_n, k, tiles)
    p.rotate = _rotate(p)
    out.append(p)
    gn = _pow2_near(4096 / t.tile_n)
    g = Problem("G", t.name, tiles // gn, t.tile_m, gn * t.tile_n, k, tiles, note="one token tile per batch entry")
    g.rotate = _rotate(g)
    out.append(g)
    return out


def k_sweep(t: Tile) -> List[Problem]:
    out = []
    for k in K_SWEEP:
        p = problems(t, k)[0]
        p.set = f"S-k{k}"
        out.append(p)
    return out


#: known per-expert / per-layer shapes for the "realistic" set (N x K of the weight; M = tokens routed). Proposed,
#: pending the user's confirmation in notes/tile-library.md.
REALISTIC_WEIGHTS = {
    "dsv3-moe-gateup": (4096, 7168),  # DeepSeek-V3 / V3.1 expert: 2 x 2048 x hidden 7168, 256 experts, top-8
    "dsv3-moe-down": (7168, 2048),
    "qwen3-235b-gateup": (3072, 4096),  # Qwen3-235B-A22B expert: 2 x 1536 x 4096, 128 experts, top-8
    "qwen3-235b-down": (4096, 1536),
    "gptoss-120b-gateup": (5760, 2880),  # gpt-oss-120b expert: 2 x 2880 x 2880, 128 experts, top-4
    "gptoss-120b-down": (2880, 2880),
    "llama3-70b-qkv": (10240, 8192),  # dense
    "llama3-70b-up": (57344, 8192),
    "llama3-70b-down": (8192, 28672),
}


def realistic(tokens_per_expert: Iterable[int] = (32, 64, 128, 256, 512)) -> List[Problem]:
    """Grouped (MoE) problems: batch = experts, M = tokens per expert (uniform routing). The user's per-config
    shapes replace these once known."""
    out = []
    experts = {"dsv3": 256, "qwen3": 128, "gptoss": 128}
    for nm, (n, kk) in REALISTIC_WEIGHTS.items():
        e = next((v for key, v in experts.items() if nm.startswith(key)), 1)
        for mt in tokens_per_expert if e > 1 else (2048, 8192):
            out.append(Problem("R", nm, e, mt, n, kk, note=nm))
    return out


def library_table() -> List[dict]:
    rows = []
    for t in LIBRARY:
        for lay in candidate_layouts(t):
            w = work(t, lay)
            rows.append({**asdict(w), "default": lay == default_layout(t), "family": family_key(w, max(w.occ_est, 1))})
    return rows

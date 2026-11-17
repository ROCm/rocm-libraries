# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Decode-shaped bf16 GEMMs on gfx950: ``Y[M, N] = X[M, K] @ W[N, K]^T``.

X ``[M, K]``, W ``[N, K]`` and Y ``[M, N]`` are row-major bf16; sums are fp32.
The output equals ``torch.nn.functional.linear(X, W)`` (CK-Tile RCR). Every
output row past M stays untouched, so any M (17 included) is legal.

At small M the weights are read once and dominate the bytes, so both kernel
families are built around one goal: keep enough weight bytes in flight on
every CU, without spilling, from the first instruction to the last.

``family="gemv"``
    One wave owns ``ytile`` weight rows over all of K. Lane ``l`` takes the
    8-element K chunks at ``8 l + 512 j``, so each wave-wide ``dwordx4`` load
    is 1 KB of one row. X is staged once per workgroup in LDS (``stage="dma"``)
    or read from global by every wave (``stage="global"``). Each lane's partial
    dot products come from ``v_mfma_f32_4x4x4_16b_bf16`` (the diagonal of each
    4x4 block) and a DPP reduction ends each row. The grid is persistent:
    ``grid`` workgroups walk the row groups. For M up to a few rows.

``family="smallm"``
    ``v_mfma_f32_16x16x32_bf16`` with the weights as the A operand (16 rows
    of W) and X as the B operand (16 rows of X, padded with zeros past M). A
    wave owns ``16 * row_groups`` rows of W over ``K / (split_k * k_waves)``
    of K; a workgroup holds ``n_waves`` such row blocks side by side and
    ``k_waves`` of them along K, whose partial tiles meet in LDS. With
    ``    split_k > 1`` the workgroups of one row block share K: each adds its
    partial tile into an fp32 workspace with device-scope atomics, and the
    last of them (an atomic ticket per row block) converts the tile to bf16,
    stores it, and clears the workspace and its ticket for the next call, so
    one launch does the whole GEMM.

``family="skinny"``
    The same MFMA with ``block_n`` W rows per workgroup and ``k_waves`` waves
    along K, each streaming ``block_k``-element stages of X and W through a
    ring of ``stages`` slots in its own LDS region (no barrier in the K
    loop); the partial tiles meet in LDS. With ``m_waves x n_waves`` waves
    the workgroup instead shares one ring of ``block_m x block_n`` tiles
    (one barrier per stage) and the grid also tiles M. See
    :mod:`rocke.instances.gfx950.decode_gemm_skinny`.

Both families issue the weight loads of ``pf`` K steps ahead of the step being
computed (``pf=-1``: all of the wave's loads up front), with the cache-policy
bits ``w_cache`` (``CPOL_SC0`` 1, ``CPOL_NT`` 2, ``CPOL_SC1`` 16), and pin each
step's load group and MFMA group with ``sched_barrier`` (``pin``) so LLVM does
not hoist the loads into spills (``pin=2``: also each step of smallm's window
prologue, which LLVM otherwise may reorder). ``preload=1`` preloads the kernel arguments
into user SGPRs (``inreg``; the direct compiler's ``kernarg_preload``), and
``dlevers`` names direct-compiler levers for this kernel alone.

Kernel arguments (all five always present, 8-byte device addresses): ``W``,
``X``, ``Y``, ``WS`` (fp32 ``[M, N]`` workspace, zero before the first call)
and ``CNT`` (int32 ``[n_tiles]`` tickets, zero before the first call). ``WS``
and ``CNT`` are read only when ``split_k > 1``; :func:`decode_gemm_launch`
gives their sizes. The kernels use Python-engine-only ops (cache-policy buffer
loads), so they lower with ``backend="python"``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, fields, replace

from ...core.ir import BF16, F32, I32, I64, IRBuilder, PtrType, VectorType
from ...helpers.spec import SignatureBuilder
from .decode_gemm_skinny import (
    build_skinny,
    skinny_auto_split,
    skinny_plan,
    skinny_shared,
    valid_skinny,
)

FAMILIES = ("gemv", "smallm", "skinny")
LDS_BYTES = 160 * 1024
# A buffer offset past every tensor this builder accepts: loads there return
# zero without a memory request and stores there are dropped.
OOB = 0x7FFF0000
W_CACHE_BITS = (0, 1, 2, 3, 16, 17, 18, 19)
SK_SYNCS = ("fence", "relaxed", "xcd", "xcdrr")
# K steps per iteration of sk_sync="xcd"'s fallback (recompute) loop.
RECOMPUTE_UNROLL = 8
# gemv: VGPRs of load destinations a wave may keep in flight (more spills or
# runs out of registers next to the accumulators and addresses).
GEMV_LOAD_VGPRS = 128
# K elements one MFMA step of family "smallm" consumes, and one gemv chunk.
SMALLM_STEP_K = 32
GEMV_CHUNK_K = 512

_COMMON = (
    "M",
    "N",
    "K",
    "family",
    "pf",
    "w_cache",
    "pin",
    "prio",
    "wpe",
    "backend",
    "preload",
    "dlevers",
)
_FAMILY_FIELDS = {
    "smallm": (
        "row_groups",
        "n_waves",
        "k_waves",
        "split_k",
        "x_stage",
        "lpf",
        "epi",
        "wg_order",
        "block_rows",
        "reduce",
        "sk_sync",
        "w_stage",
        "w_piece",
        "w_ring",
        "w_ilv",
        "x_stream",
        "sk_finish",
        "lds_swz",
        "x_ring",
        "acc_split",
        "w_tail",
        "cus",
        "xccs",
    ),
    "gemv": (
        "ytile",
        "unrl",
        "waves",
        "weff",
        "grid",
        "cus",
        "gmap",
        "stage",
        "pf0",
        "xpf",
        "lpf",
        "lorder",
        "morder",
    ),
    "skinny": (
        "block_m",
        "block_n",
        "block_k",
        "stages",
        "k_waves",
        "m_waves",
        "n_waves",
        "xcd",
        "refill",
        "split_k",
        "sk_sync",
        "m0fill",
        "cus",
        "xccs",
        "sk_finish",
    ),
}
# Short spellings for kernel names.
_ABBR = {
    "pf": "pf",
    "w_cache": "wc",
    "pin": "pin",
    "prio": "pr",
    "wpe": "wpe",
    "backend": "be",
    "preload": "kp",
    "dlevers": "dl",
    "row_groups": "rg",
    "n_waves": "nw",
    "k_waves": "kw",
    "split_k": "sk",
    "x_stage": "xs",
    "lpf": "lpf",
    "epi": "epi",
    "wg_order": "wo",
    "block_rows": "br",
    "reduce": "rd",
    "sk_sync": "ss",
    "w_stage": "wst",
    "w_piece": "wpc",
    "w_ring": "wrg",
    "w_ilv": "wil",
    "x_stream": "xsm",
    "sk_finish": "skf",
    "lds_swz": "swz",
    "x_ring": "xrg",
    "acc_split": "acs",
    "w_tail": "wtl",
    "ytile": "y",
    "unrl": "u",
    "waves": "w",
    "weff": "we",
    "grid": "g",
    "cus": "cu",
    "xccs": "xc",
    "gmap": "gm",
    "stage": "st",
    "pf0": "pfz",
    "xpf": "xpf",
    "lorder": "lo",
    "morder": "mo",
    "block_m": "bm",
    "block_n": "bn",
    "block_k": "bk",
    "stages": "stg",
    "m_waves": "mw",
    "xcd": "xcd",
    "refill": "rf",
    "m0fill": "mf",
}


@dataclass(frozen=True)
class DecodeGemmSpec:
    """One decode GEMM: the problem (``M``, ``N``, ``K``), the kernel family
    and its knobs. Only the problem is required; every knob defaults to the
    value the family was written around, and knobs of the other family are
    ignored (and left out of the kernel name)."""

    M: int
    N: int
    K: int
    family: str = "smallm"
    # Weight-load window: K steps (smallm: 32 elements, gemv: ``unrl`` chunks)
    # issued ahead of the step being computed; -1 issues all of them up front.
    pf: int = -1
    w_cache: int = 2  # weight-load cache bits: 1 sc0, 2 nt, 16 sc1 (sums allowed)
    # sched_barrier(0) around each step's load and MFMA groups; 2 (smallm, direct
    # weight loads) also after each step of the window prologue.
    pin: int = 1
    prio: int = 0  # s_setprio(1): 1 around MFMA groups, 2 around load groups
    wpe: int = 0  # amdgpu-waves-per-eu hint (0: none)
    backend: str = "llvm"  # code-object compiler: "llvm" (COMGR) or "direct"
    preload: int = 0  # 1: kernel arguments preloaded into user SGPRs (inreg)
    # Direct-compiler levers of this kernel (its ``dlevers`` attribute, ``,``
    # joined); the LLVM lowering ignores them.
    dlevers: str = ""
    # ---- smallm ----
    row_groups: int = 1  # 16-row groups of W per wave
    n_waves: int = 1  # waves side by side along N in a workgroup
    k_waves: int = 1  # waves along K in a workgroup (partials meet in LDS)
    split_k: int = 1  # workgroups along K (partials meet in the workspace)
    x_stage: str = "lds"  # X: "lds" (staged once per workgroup) or "global"
    lpf: int = 0  # X reads from LDS issued this many steps ahead (both families)
    epi: str = "auto"  # "direct" (stores from registers), "lds" (through LDS), "auto"
    # Workgroup order: "split" (split index fastest), "tile" (row block fastest)
    # or "skew" ("tile" with the last split's row blocks shifted by one, so one
    # partner of each block sits on another XCD under round-robin dispatch).
    wg_order: str = "split"
    # W rows per workgroup (0: n_waves * 16 * row_groups). Fewer rows than the
    # waves cover mask the rest inside the 16-row MFMA tiles (no memory traffic),
    # so a row block can match the rows per CU exactly.
    block_rows: int = 0
    # split_k > 1: "atomic" (fp32 atomics into WS, the finisher clears it) or
    # "slots" (each split stores its partial tile into its own WS slot).
    reduce: str = "atomic"
    # How the finisher sees the partials: "fence" (agent-scope release fence,
    # ticket, acquire fence), "relaxed" (each wave waits for its own partial
    # writes, device-coherent sc1 stores and loads on the partials only),
    # "xcd" (plain partial stores, an agent-scope ticket that also sums its
    # workgroup's XCC id and its square: when the sums show every partner on
    # the finisher's XCC, the finisher reads the partials with plain loads from
    # that XCC's L2; otherwise it computes the row block again from W and X.
    # wg_order="tile" with a row-block count divisible by ``xccs`` co-locates
    # the partners under round-robin dispatch, the fast path), "xcdrr" (plain
    # partials and loads without the check: correct only while the partners
    # share an XCD; a reference, left out of the knob moves) or "auto" ("xcd"
    # when the order co-locates the partners on ``xccs`` XCCs, else "relaxed").
    sk_sync: str = "fence"
    # W path: "direct" (each lane loads its 16x16x32 A-operand slice: a wave load
    # is 16 rows x 64 B) or "lds" (LDS DMA of whole lines of one row at a time
    # into a wave-private ring, then ds_read_b128 in the operand layout).
    w_stage: str = "direct"
    w_piece: int = 512  # K elements of one ring slot (512: one row per DMA, 1 KB)
    w_ring: int = 2  # ring slots (pieces in flight) per wave
    # w_piece < 512 (R = 512 / w_piece rows per 1 KB DMA): 0 puts the R rows of
    # a DMA block back to back (their A-operand reads share banks); 1 interleaves
    # them at 16 B (DMA lane l: row l % R, chunk l / R) in blocks padded by 16 R
    # bytes, so the 16 rows of a ds_read_b128 land on distinct banks.
    w_ilv: int = 0
    # x_stage="lds" with w_stage="lds": 0 stages the workgroup's whole X slice
    # before the first piece; 1 stages it in 1 KB K phases (one LDS allocation
    # each), the first before the weight ring and the rest after it, and each
    # piece waits only for the phase it reads; 2 (k_waves=1, lds_swz=1) stages
    # X per W piece: phase p holds the X rows at piece p's K range, each wave
    # issues its share of X(p) just before W(p), and piece p waits for both and
    # passes one barrier (the other waves' X rows of the phase).
    x_stream: int = 0
    # x_stream=2: LDS slots the X phases cycle through (0: one per phase). A
    # slot is refilled with the W piece that is w_ring pieces ahead, so it needs
    # more slots than the W ring.
    x_ring: int = 0
    # Who sums the split-K slots: "last" (the workgroup that takes the last
    # ticket sums the whole row block) or "shared" (reduce="slots",
    # sk_sync="relaxed": every partner waits on the ticket until all have
    # stored, then sums 1 / split_k of the block. The ticket counts up across
    # calls (no reset), so split_k must be a power of two; the partners of a
    # row block must be resident together, so the grid stays within one
    # workgroup per CU of the device.) "shared_sc" is "shared" with the ticket
    # polled at system scope (sc0 sc1) instead of device scope (sc1). Skinny's
    # split ring takes "last", "shared" (the same protocol over the tile) and
    # "epoch" ("last" with the ticket counting up across calls, no reset store).
    sk_finish: str = "last"
    # LDS layout of the staged X rows and the W ring: 0 pads each X row (and
    # each 1 KB W block) by 16 B; 1 packs the rows densely and stores the 16 B
    # chunk c of row r at chunk c ^ (r & 15) of the row. The DMA keeps its
    # linear LDS write and fetches the permuted chunk from global memory, and
    # the operand reads apply the same XOR, so the 16 rows of a ds_read_b128
    # cover all banks once.
    lds_swz: int = 0
    # Accumulator chains (1 or 2, at most two 16x16 tiles per wave): with 2, K
    # steps alternate between two sets of accumulators, added before the
    # epilogue, so consecutive MFMAs do not wait on each other's results.
    acc_split: int = 1
    # w_stage="lds", w_piece=128, lds_swz=1: 1 stages the last piece as two
    # 64-element halves in LDS buffers of their own (8 dense 128 B rows per
    # DMA, chunk c of row r at c ^ ((r >> 1) & 7)); the first half's two steps
    # run while the second half is still in flight.
    w_tail: int = 0
    # ---- gemv ----
    ytile: int = 1  # weight rows per wave per row group
    unrl: int = 1  # 512-element K chunks per step
    waves: int = 8  # waves per workgroup
    weff: int = 0  # waves that compute (0: the fewest that keep the rounds; -1: all)
    grid: int = 0  # persistent workgroups (0: cus)
    # Compute units of the device or partition: gemv's persistent grid, skinny's
    # split_k=-1 and the resident-grid bound of smallm's sk_finish="shared".
    cus: int = 256
    gmap: str = "wave"  # row-group order: "wave" (wave * grid + block) or "block"
    stage: str = "dma"  # X: "dma" (global_load_lds into LDS) or "global"
    pf0: int = -2  # steps issued before the X staging wait (-2: pf)
    xpf: int = 1  # issue the next row group's first pf steps during this one
    lorder: str = "ky"  # loads in a step: "ky" (chunk-major) or "yk" (row-major)
    morder: str = "hn"  # MFMAs of a chunk: "hn" (half-major) or "nh"
    # ---- skinny (also k_waves) ----
    block_m: int = 0  # X rows per stage: 8, 16, 24 or 32 (0: the smallest >= M)
    block_n: int = 32  # W rows per workgroup (16-row MFMA tiles)
    block_k: int = 64  # K elements per stage
    stages: int = 4  # ring slots per wave (stages in flight)
    m_waves: int = 1  # waves along M sharing one ring with the n_waves (k_waves = 1)
    xcd: int = 0  # 1: workgroup b runs tile (b % xccs) * (grid / xccs) + b / xccs (one tile range per XCD)
    refill: int = 0  # shared ring: 0 refills right after the barrier, 1 after the step's reads
    m0fill: int = 0  # LDS DMA M0 hazard slot: 1 shares M0 writes (immediate offsets), 2 refills among the MFMAs, 3 both
    # ---- device (smallm, skinny; also cus) ----
    # XCCs consecutive workgroups go round-robin to: skinny's xcd=1 modulus and
    # the co-location rule of smallm's sk_sync "xcdrr" and "auto".
    xccs: int = 8

    def relevant_fields(self) -> tuple[str, ...]:
        return _COMMON + _FAMILY_FIELDS.get(self.family, ())

    def kernel_name(self) -> str:
        """A stable symbol per distinct kernel: the problem, then every knob of
        the family that differs from its default."""
        parts = [f"dg_{self.family}_m{self.M}n{self.N}k{self.K}"]
        defaults = DecodeGemmSpec(self.M, self.N, self.K, family=self.family)
        for name in self.relevant_fields()[4:]:
            value = getattr(self, name)
            if value != getattr(defaults, name):
                text = re.sub(r"[,+]", "x", f"{_ABBR[name]}{value}")
                parts.append(text.replace("-", "n"))
        return "_".join(parts)


def _ceil(a: int, b: int) -> int:
    return -(-a // b)


def _mindiv(rows: int, div1: int, waves: int) -> int:
    """The fewest waves per workgroup that keep the number of rounds."""
    rnds = [_ceil(rows, div1 * (waves - i)) for i in range(waves)]
    for i in range(len(rnds) - 1, -1, -1):
        if rnds[0] == rnds[i]:
            return waves - i
    return waves


# ---------------------------------------------------------------- plans


def gemv_plan(spec: DecodeGemmSpec) -> dict:
    """The concrete tiling of a gemv spec (no IR)."""
    Y, U, W = spec.ytile, spec.unrl, spec.waves
    G = spec.grid or spec.cus
    WE = W if spec.weff == -1 else (spec.weff or _mindiv(spec.N, G * Y, W))
    WE = min(WE, W)
    C = spec.K // GEMV_CHUNK_K
    steps = [list(range(s0, min(C, s0 + U))) for s0 in range(0, C, U)]
    S = len(steps)
    D = S if spec.pf < 0 else min(spec.pf, S)
    D0 = D if spec.pf0 == -2 else (S if spec.pf0 < 0 else min(spec.pf0, S))
    rounds = _ceil(_ceil(spec.N, Y), G * WE)
    lds = spec.M * spec.K * 2 if spec.stage == "dma" else 0
    # VGPRs of dwordx4 load destinations in flight per wave: the window's steps,
    # each U chunks of the Y weight rows (and of every X row from global).
    per_step = Y + (0 if spec.stage == "dma" else spec.M)
    load_vgprs = 4 * per_step * sum(len(steps[s]) for s in range(max(D, D0)))
    return {
        "Y": Y,
        "U": U,
        "W": W,
        "WE": WE,
        "G": G,
        "C": C,
        "steps": steps,
        "S": S,
        "D": D,
        "D0": D0,
        "rounds": rounds,
        "lds": lds,
        "load_vgprs": load_vgprs,
        "threads": 64 * W,
        "grid": G,
    }


def smallm_plan(spec: DecodeGemmSpec) -> dict:
    """The concrete tiling of a smallm spec (no IR)."""
    MG = _ceil(spec.M, 16)
    rows_w = 16 * spec.row_groups
    block_n = rows_w * spec.n_waves  # rows the waves of a workgroup cover
    block_rows = spec.block_rows or block_n  # rows a workgroup owns (the tile stride)
    n_tiles = _ceil(spec.N, block_rows)
    k_wg = spec.K // spec.split_k
    kw_len = k_wg // spec.k_waves
    S = kw_len // SMALLM_STEP_K
    D = S if spec.pf < 0 else min(spec.pf, S)
    epi = spec.epi
    if epi == "auto":
        direct = spec.k_waves == 1 and spec.split_k == 1 and block_rows % 4 == 0
        epi = "direct" if direct else "lds"
    # Consecutive output columns a thread stores at once in the epilogue.
    vec = 4 if block_rows % 4 == 0 else (2 if block_rows % 2 == 0 else 1)
    x_pitch = k_wg * 2 + (0 if spec.lds_swz else 16)  # bytes per staged X row
    x_lds = spec.M * x_pitch if spec.x_stage == "lds" else 0
    e_pitch = block_n + 4  # floats per row of the epilogue tile
    e_lds = spec.k_waves * 16 * MG * e_pitch * 4 if epi == "lds" else 0
    waves = spec.n_waves * spec.k_waves
    # LDS-staged W: a DMA covers 1 KB = w_rows rows x (2 w_piece) B; a slot holds
    # a wave's 16 x row_groups rows of one piece, one 1 KB block (+16 B pad) per DMA.
    w_rows = max(1, 512 // spec.w_piece) if spec.w_piece > 0 else 1
    w_dmas = 16 * spec.row_groups // w_rows  # DMAs per piece per wave
    pieces = kw_len // spec.w_piece if spec.w_piece > 0 else 0
    ring = min(spec.w_ring, pieces) if spec.w_stage == "lds" else 0
    w_pitch = (
        1024 + 16 * w_rows if spec.w_ilv else (1024 if spec.lds_swz else 1040)
    )  # bytes per DMA block in a slot
    w_lds = ring * waves * w_dmas * w_pitch if spec.w_stage == "lds" else 0
    if spec.w_tail:  # the last piece's two halves, w_dmas / 2 DMAs each
        w_lds += waves * w_dmas * 1024
    x_phases = kw_len // GEMV_CHUNK_K if spec.x_stream else 1
    if spec.x_stream == 1 and spec.x_stage == "lds":
        x_lds = (
            x_phases * spec.k_waves * spec.M * 1040
        )  # per phase: [k_wave][m] rows of 1 KB + 16 B
    # x_stream=2: a phase per W piece, dense rows of 2 w_piece B written by
    # 1 KB DMAs of x_rows rows (the last DMA's spare rows repeat row M - 1).
    x_rows = max(1, 512 // spec.w_piece) if spec.w_piece > 0 else 1
    x_dmas = _ceil(spec.M, x_rows)
    x_slots = 0
    if spec.x_stream == 2:
        x_phases = pieces
        x_slots = spec.x_ring or pieces
        x_lds = x_slots * x_dmas * 1024
    return {
        "MG": MG,
        "rows_w": rows_w,
        "block_n": block_n,
        "block_rows": block_rows,
        "vec": vec,
        "n_tiles": n_tiles,
        "k_wg": k_wg,
        "kw_len": kw_len,
        "S": S,
        "D": D,
        "epi": epi,
        "x_pitch": x_pitch,
        "x_lds": x_lds,
        "e_pitch": e_pitch,
        "e_lds": e_lds,
        "w_rows": w_rows,
        "w_dmas": w_dmas,
        "w_pitch": w_pitch,
        "x_phases": x_phases,
        "x_rows": x_rows,
        "x_dmas": x_dmas,
        "x_slots": x_slots,
        "pieces": pieces,
        "ring": ring,
        "w_lds": w_lds,
        "lds": x_lds + e_lds + w_lds + 16,
        "waves": waves,
        "threads": 64 * waves,
        "grid": n_tiles * spec.split_k,
    }


def _xcd_sync_error(spec: DecodeGemmSpec) -> str:
    """Why smallm's sk_sync="xcd" cannot sync ``spec`` ("" when it can)."""
    br = spec.block_rows or 16 * spec.row_groups * spec.n_waves
    if not 2 <= spec.split_k <= 63 or br % 4 or (spec.K // SMALLM_STEP_K) % RECOMPUTE_UNROLL:
        return (
            "sk_sync='xcd' needs 2 <= split_k <= 63 (its ticket fields), block_rows "
            f"% 4 == 0 and K / 32 a multiple of {RECOMPUTE_UNROLL} (its fallback)"
        )
    return ""


def smallm_colocated(spec: DecodeGemmSpec) -> bool:
    """Whether round-robin dispatch over ``xccs`` XCCs puts every split partner
    of a row block on one XCC (wg_order="tile": partners lie a row-block count
    apart)."""
    br = spec.block_rows or 16 * spec.row_groups * spec.n_waves
    return spec.split_k > 1 and spec.wg_order == "tile" and _ceil(spec.N, br) % spec.xccs == 0


def resolved_spec(spec: DecodeGemmSpec) -> DecodeGemmSpec:
    """``spec`` with its device-derived knobs made concrete (the kernel keeps
    the name of ``spec``): skinny's shared-ring split_k=-1 becomes
    :func:`skinny_auto_split`, smallm's sk_sync="auto" becomes "xcd" when the
    partners co-locate and the xcd protocol applies, else "relaxed"."""
    if spec.family == "skinny" and spec.split_k == -1 and skinny_shared(spec):
        return replace(spec, split_k=skinny_auto_split(spec))
    if spec.family == "smallm" and spec.sk_sync == "auto":
        xcd = smallm_colocated(spec) and not _xcd_sync_error(spec)
        return replace(spec, sk_sync="xcd" if xcd else "relaxed")
    return spec


def decode_gemm_launch(spec: DecodeGemmSpec) -> dict:
    """Launch geometry and buffers: ``grid`` and ``block`` (3-tuples),
    ``ws_bytes`` (fp32 workspace) and ``cnt_ints`` (tickets), both to be
    zeroed once before the first launch (0 when the kernel does not use
    them), and the static ``lds_bytes``."""
    spec = resolved_spec(spec)
    if spec.family == "gemv":
        p = gemv_plan(spec)
        return {
            "grid": (p["grid"], 1, 1),
            "block": (p["threads"], 1, 1),
            "ws_bytes": 0,
            "cnt_ints": 0,
            "lds_bytes": p["lds"],
        }
    if spec.family == "skinny":
        p = skinny_plan(spec)
        sk = spec.split_k > 1
        return {
            "grid": (p["grid"], 1, 1),
            "block": (p["threads"], 1, 1),
            "ws_bytes": spec.split_k * spec.M * spec.N * 4 if sk else 0,
            # sk_finish="shared": (unused, ticket) pairs, the ticket an odd word
            "cnt_ints": p["tiles"] * (2 if spec.sk_finish == "shared" else 1) if sk else 0,
            "lds_bytes": p["lds"],
        }
    p = smallm_plan(spec)
    sk = spec.split_k > 1
    slots = spec.split_k if spec.reduce == "slots" else 1
    return {
        "grid": (p["grid"], 1, 1),
        "block": (p["threads"], 1, 1),
        "ws_bytes": slots * spec.M * spec.N * 4 if sk else 0,
        # sk_finish="shared": (unused, ticket) pairs, the ticket an odd word
        "cnt_ints": p["n_tiles"] * (2 if spec.sk_finish != "last" else 1)
        if sk
        else 0,
        "lds_bytes": p["lds"],
    }


def decode_gemm_signature(spec: DecodeGemmSpec):
    signature = SignatureBuilder()
    for name, dtype in (
        ("W", "bf16"),
        ("X", "bf16"),
        ("Y", "bf16"),
        ("WS", "f32"),
        ("CNT", "i32"),
    ):
        signature.ptr(name, dtype)
    return signature.build()


def is_valid_spec(spec: DecodeGemmSpec, arch: str = "gfx950") -> tuple[bool, str]:
    if arch != "gfx950":
        return False, "decode GEMM targets gfx950"
    if spec.family not in FAMILIES:
        return False, f"family must be one of {FAMILIES}"
    M, N, K = spec.M, spec.N, spec.K
    if M < 1 or N < 16 or K < SMALLM_STEP_K:
        return False, "M >= 1, N >= 16 and K >= 32 required"
    if N % 16 or K % SMALLM_STEP_K:
        return False, "N must be a multiple of 16 and K of 32"
    if max(N * K * 2, M * K * 2, M * N * 4) >= OOB:
        return False, "tensors must stay below the out-of-bounds offset"
    if spec.pf < -1:
        return False, "pf must be -1 (all) or >= 0"
    if spec.w_cache not in W_CACHE_BITS:
        return False, f"w_cache must be one of {W_CACHE_BITS}"
    if (
        spec.pin not in (0, 1, 2)
        or spec.prio not in (0, 1, 2)
        or not 0 <= spec.wpe <= 8
    ):
        return False, "pin in {0, 1, 2}, prio in {0, 1, 2}, wpe in 0..8"
    if spec.pin == 2 and (spec.family != "smallm" or spec.w_stage != "direct"):
        return False, "pin=2 fences the prologue of smallm's direct weight-load window"
    if spec.backend not in ("llvm", "direct"):
        return False, "backend must be 'llvm' or 'direct'"
    if spec.preload not in (0, 1):
        return False, "preload in {0, 1}"
    if not re.fullmatch(r"[a-z0-9,+-]*", spec.dlevers):
        return False, "dlevers is a ',' or '+' joined list of lever names"
    if spec.lpf < 0:
        return False, "lpf must be >= 0"
    if spec.family != "gemv" and (spec.cus < 1 or not 1 <= spec.xccs <= 16):
        return False, "cus >= 1 and xccs in 1..16 (XCC ids are 4 bits)"
    spec = resolved_spec(spec)
    if spec.family == "gemv":
        return _valid_gemv(spec)
    if spec.family == "skinny":
        return valid_skinny(spec, LDS_BYTES)
    return _valid_smallm(spec)


def _valid_gemv(spec: DecodeGemmSpec) -> tuple[bool, str]:
    if spec.K % GEMV_CHUNK_K:
        return False, "gemv needs K a multiple of 512"
    if not 1 <= spec.M <= 8:
        return False, "gemv keeps one accumulator per X row: M <= 8"
    if not 1 <= spec.ytile <= 4 or not 1 <= spec.unrl <= 4:
        return False, "ytile and unrl in 1..4"
    if not 1 <= spec.waves <= 16 or spec.weff < -1 or spec.weff > spec.waves:
        return False, "waves in 1..16, weff in -1..waves"
    if spec.grid < 0 or spec.cus < 1:
        return False, "grid >= 0 and cus >= 1"
    if spec.gmap not in ("wave", "block") or spec.stage not in ("dma", "global"):
        return False, "gmap in (wave, block), stage in (dma, global)"
    if spec.lorder not in ("ky", "yk") or spec.morder not in ("hn", "nh"):
        return False, "lorder in (ky, yk), morder in (hn, nh)"
    if spec.pf0 < -2 or spec.xpf not in (0, 1):
        return False, "pf0 >= -2, xpf in {0, 1}"
    if gemv_plan(spec)["lds"] > LDS_BYTES:
        return False, "X does not fit LDS: use stage='global'"
    if gemv_plan(spec)["load_vgprs"] > GEMV_LOAD_VGPRS:
        return (
            False,
            (
                f"gemv keeps at most {GEMV_LOAD_VGPRS} VGPRs of loads in flight per wave "
                "(4 x window chunks x (ytile + X rows from global)): lower pf or use stage='dma'"
            ),
        )
    return True, "ok"


def _valid_smallm(spec: DecodeGemmSpec) -> tuple[bool, str]:
    if spec.M > 128 or (spec.M > 64 and spec.x_stage != "lds"):
        return False, "smallm keeps at most eight 16-row X groups (four with x_stage='global'): M <= 128"
    if not 1 <= spec.row_groups <= 4:
        return False, "row_groups in 1..4"
    if spec.n_waves < 1 or spec.k_waves < 1 or spec.n_waves * spec.k_waves > 16:
        return False, "n_waves, k_waves >= 1 and at most 16 waves per workgroup"
    if spec.split_k < 1 or spec.K % (spec.split_k * spec.k_waves * SMALLM_STEP_K):
        return (
            False,
            "K must split into split_k * k_waves pieces of whole 32-element steps",
        )
    if spec.x_stage not in ("lds", "global"):
        return False, "x_stage in (lds, global)"
    if spec.x_stage == "lds" and (spec.K // spec.split_k) % GEMV_CHUNK_K:
        return False, "x_stage='lds' stages 1 KB rows: K / split_k a multiple of 512"
    if spec.epi not in ("auto", "direct", "lds"):
        return False, "epi in (auto, direct, lds)"
    block_n = 16 * spec.row_groups * spec.n_waves
    if not 0 <= spec.block_rows <= block_n:
        return False, "block_rows in 0..n_waves * 16 * row_groups (0: all of them)"
    br = spec.block_rows or block_n
    if spec.epi == "direct" and (spec.k_waves > 1 or spec.split_k > 1 or br % 4):
        return (
            False,
            "epi='direct' needs k_waves == 1, split_k == 1 and block_rows % 4 == 0",
        )
    if spec.wg_order not in ("split", "tile", "skew"):
        return False, "wg_order in (split, tile, skew)"
    if spec.wg_order == "skew" and spec.split_k < 2:
        return False, "wg_order='skew' moves split partners: split_k > 1"
    if spec.reduce not in ("atomic", "slots") or spec.sk_sync not in SK_SYNCS:
        return False, f"reduce in (atomic, slots), sk_sync in {SK_SYNCS}"
    if spec.sk_sync == "xcd" and _xcd_sync_error(spec):
        return False, _xcd_sync_error(spec)
    if spec.sk_sync == "xcdrr" and not smallm_colocated(spec):
        return (
            False,
            (
                "sk_sync='xcdrr' needs split_k > 1, wg_order='tile' and a row-block "
                "count divisible by xccs (the split partners of a block on one XCD)"
            ),
        )
    if spec.split_k * spec.M * spec.N * 4 >= OOB:
        return False, "split-K slots must stay below the out-of-bounds offset"
    if spec.w_stage not in ("direct", "lds"):
        return False, "w_stage in (direct, lds)"
    if spec.w_stage == "lds":
        kw_len = spec.K // (spec.split_k * spec.k_waves)
        if spec.w_piece not in (128, 256, 512) or kw_len % spec.w_piece:
            return (
                False,
                "w_stage='lds': w_piece in (128, 256, 512) dividing the wave's K range",
            )
        if spec.w_ring < 1 or spec.x_stage != "lds" or spec.w_cache not in (0, 1, 2, 3):
            return (
                False,
                "w_stage='lds' needs w_ring >= 1, x_stage='lds' and w_cache in 0..3",
            )
    if spec.w_ilv not in (0, 1) or spec.x_stream not in (0, 1, 2):
        return False, "w_ilv in {0, 1} and x_stream in {0, 1, 2}"
    if spec.x_stream == 2:
        plan = smallm_plan(spec)
        if (
            spec.w_stage != "lds"
            or spec.k_waves != 1
            or not spec.lds_swz
            or (spec.x_ring and not plan["ring"] < spec.x_ring <= plan["pieces"])
        ):
            return (
                False,
                (
                    "x_stream=2 needs w_stage='lds', k_waves=1, lds_swz=1 and x_ring 0 or "
                    "above the W ring and at most the pieces"
                ),
            )
    elif spec.x_ring:
        return False, "x_ring needs x_stream=2"
    if spec.w_ilv and (spec.w_stage != "lds" or spec.w_piece == 512):
        return (
            False,
            "w_ilv=1 interleaves the rows of a DMA block: w_stage='lds', w_piece < 512",
        )
    if spec.x_stream == 1 and (
        spec.w_stage != "lds"
        or (spec.K // (spec.split_k * spec.k_waves)) % GEMV_CHUNK_K
        or spec.K // (spec.split_k * spec.k_waves) < 2 * GEMV_CHUNK_K
    ):
        return (
            False,
            "x_stream=1 needs w_stage='lds' and a wave K range of two or more 512-element phases",
        )
    if spec.lds_swz not in (0, 1):
        return False, "lds_swz in {0, 1}"
    if spec.lds_swz and (
        spec.x_stage != "lds"
        or spec.x_stream == 1
        or spec.w_ilv
        or (spec.K // (spec.split_k * spec.k_waves)) % 128
    ):
        return (
            False,
            (
                "lds_swz=1 needs x_stage='lds', x_stream 0 or 2, w_ilv=0 and a wave K range "
                "of whole 128-element (16-chunk) groups"
            ),
        )
    if spec.sk_finish not in ("last", "shared", "shared_sc"):
        return False, "sk_finish in (last, shared, shared_sc)"
    if spec.sk_finish != "last" and (
        spec.split_k < 2
        or spec.split_k & (spec.split_k - 1)
        or spec.reduce != "slots"
        or spec.sk_sync != "relaxed"
        or smallm_plan(spec)["grid"] > spec.cus
    ):
        # The finisher spins until every partner has stored: all must be
        # resident at once, so the grid stays within one workgroup per CU.
        return (
            False,
            (
                "sk_finish='shared' needs a power-of-two split_k > 1, reduce='slots', "
                "sk_sync='relaxed' and a grid of at most cus workgroups"
            ),
        )
    if spec.acc_split not in (1, 2) or (
        spec.acc_split == 2 and spec.row_groups * _ceil(spec.M, 16) > 2
    ):
        return False, "acc_split in {1, 2}; 2 needs at most two 16x16 tiles per wave"
    if spec.w_tail not in (0, 1) or (
        spec.w_tail
        and (
            spec.w_stage != "lds"
            or spec.w_piece != 128
            or not spec.lds_swz
            or spec.x_stream
        )
    ):
        return (
            False,
            "w_tail in {0, 1}; 1 needs w_stage='lds', w_piece=128, lds_swz=1 and x_stream=0",
        )
    if smallm_plan(spec)["lds"] > LDS_BYTES:
        return False, "LDS exceeds the workgroup limit"
    return True, "ok"


def default_spec(M: int, N: int, K: int, cus: int = 256, xccs: int = 8) -> DecodeGemmSpec:
    """A valid starting configuration for one problem (the search starts here)
    on a device or partition of ``cus`` CUs and ``xccs`` XCCs.

    gemv for one to four rows when X fits LDS; otherwise smallm with the row
    block and K split that give the grid closest to (and not above) a whole
    number of waves of ``cus`` workgroups, the fewest split-K partials first."""
    if M <= 4 and K % GEMV_CHUNK_K == 0 and M * K * 2 <= LDS_BYTES // 2:
        rows_cu = _ceil(N, cus)
        ytile = 1 if rows_cu <= 16 else 2
        waves = max(1, min(16, _ceil(rows_cu, ytile)))
        spec = DecodeGemmSpec(M, N, K, family="gemv", ytile=ytile, waves=waves, cus=cus)
        if is_valid_spec(spec)[0]:
            return spec
    tiles16 = N // 16
    best = None
    for n_waves in range(1, 17):
        if tiles16 % n_waves:
            continue
        for split_k in (1, 2, 4, 8, 16, 32):
            spec = DecodeGemmSpec(M, N, K, n_waves=n_waves, split_k=split_k, cus=cus, xccs=xccs)
            if not is_valid_spec(spec)[0]:
                continue
            grid = (tiles16 // n_waves) * split_k
            fill = grid / (_ceil(grid, cus) * cus)
            key = (-fill, split_k, -n_waves)
            if best is None or key < best[0]:
                best = (key, spec)
    if best is None:
        return DecodeGemmSpec(M, N, K, x_stage="global", cus=cus, xccs=xccs)
    return best[1]


# ---------------------------------------------------------------- builder


def build_decode_gemm(spec: DecodeGemmSpec, *, arch: str = "gfx950"):
    """The kernel of ``spec`` (a :class:`~rocke.core.ir.KernelDef`)."""
    ok, reason = is_valid_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid decode GEMM spec: {reason}")
    b = IRBuilder(spec.kernel_name())
    spec = resolved_spec(spec)
    if spec.wpe:
        b.kernel.attrs["waves_per_eu"] = spec.wpe
    if spec.preload:
        b.kernel.attrs["kernarg_preload"] = True
    if spec.dlevers:
        b.kernel.attrs["dlevers"] = spec.dlevers
    ptrs = {}
    for name, elem, ro in (
        ("W", BF16, True),
        ("X", BF16, True),
        ("Y", BF16, False),
        ("WS", F32, False),
        ("CNT", I32, False),
    ):
        attrs = {"noalias": True, "align": 16}
        if ro:
            attrs["readonly"] = True
        if spec.preload:
            attrs["inreg"] = True
        ptrs[name] = b.param(name, PtrType(elem, "global"), **attrs)
    if spec.family == "gemv":
        _build_gemv(b, spec, ptrs)
    elif spec.family == "skinny":
        build_skinny(b, spec, ptrs)
    else:
        _build_smallm(b, spec, ptrs)
    b.ret()
    return b.kernel


def _half4(b, words, h):
    """bf16 elements 4h .. 4h + 3 of a ``<4 x i32>`` (eight bf16) as ``<4 x bf16>``."""
    pair = b.vec_pack(
        [b.vec_extract(words, 2 * h), b.vec_extract(words, 2 * h + 1)], I32
    )
    return b.bitcast(pair, VectorType(BF16, 4))


def _dpp(b, v, shl):
    vi = b.bitcast(v, I32)
    if shl == 3:  # outside mov_dpp's row-shift set; the lowering encodes it directly
        got = b._op(
            "tile.mov_dpp",
            [vi],
            [I32],
            attrs={"bound_ctrl": True, "row_shl": 3},
            result_name_hint="dpp",
        ).result
    else:
        got = b.mov_dpp(vi, row_shl=shl, bound_ctrl=True)
    return b.bitcast(got, F32)


def _build_gemv(b, spec, ptrs):
    p = gemv_plan(spec)
    Y, W, WE, G, C = p["Y"], p["W"], p["WE"], p["G"], p["C"]
    steps, S, D, D0, rounds = p["steps"], p["S"], p["D"], p["D0"], p["rounds"]
    R, K, A = spec.N, spec.K, spec.M
    staged = spec.stage == "dma"
    xpf = bool(spec.xpf) and rounds > 1 and D > 0
    b.kernel.attrs["max_workgroup_size"] = 64 * W
    i = b.const_i32
    rw = b.buffer_rsrc(ptrs["W"], i(R * K * 2))
    rx = b.buffer_rsrc(ptrs["X"], i(A * K * 2))
    ry = b.buffer_rsrc(ptrs["Y"], i(A * R * 2))
    tid = b.thread_id_x()
    lane = b.land(tid, i(63))
    lane16 = b.mul(lane, i(16))
    lane4 = b.mul(lane, i(4))
    wave = b.readfirstlane(b.lshr(tid, i(6)))
    bid = b.block_id_x()
    rg0 = (
        b.add(b.mul(wave, i(G)), bid)
        if spec.gmap == "wave"
        else b.add(b.mul(bid, i(WE)), wave)
    )
    # waves past WE compute nothing: their rows sit past N (loads out of bounds, no stores)
    m0 = b.select(b.cmp_lt(wave, i(WE)), b.mul(rg0, i(Y)), i(R))
    mstep = G * WE * Y
    smem = None
    if staged:
        smem = b.smem_alloc(I32, [A * C * 256], name_hint="dg_x")
        lds_base = b.smem_addr_of(smem)

    def row_voffs(m):
        vo = []
        for y in range(Y):
            row = b.add(m, i(y))
            off = b.select(b.cmp_lt(row, i(R)), b.mul(row, i(K * 2)), i(OOB))
            vo.append(b.add(lane16, off))
        return vo

    def issue(vo, s):
        js = steps[s]
        got = [[None] * len(js) for _ in range(Y)]
        order = [(u, y) for u in range(len(js)) for y in range(Y)]
        if spec.lorder == "yk":
            order = [(u, y) for y in range(Y) for u in range(len(js))]
        if spec.prio == 2:
            b.s_setprio(1)
        for u, y in order:
            got[y][u] = b.buffer_load_cp(
                rw, vo[y], i(js[u] * 1024), dwords=4, aux=spec.w_cache
            )
        acts = None
        if not staged:
            acts = [
                [
                    b.buffer_load_cp(
                        rx, lane16, i(n * K * 2 + j * 1024), dwords=4, aux=0
                    )
                    for j in js
                ]
                for n in range(A)
            ]
        if spec.prio == 2:
            b.s_setprio(0)
        return (got, acts)

    def nloads(s):
        return len(steps[s]) * (Y + (0 if staged else A))

    def act_reads(s):
        return [
            [
                b.smem_load_vN(smem, b.add(lane4, i((n * C + j) * 256)), dtype=I32, n=4)
                for j in steps[s]
            ]
            for n in range(A)
        ]

    def stage_x(extra):
        """X into LDS (1 KB chunks spread over the waves); ``extra()`` issues
        weight loads that overlap the copy. Every wave must call it."""
        total = A * C
        for sl in range(_ceil(total, W)):
            c = b.add(i(sl * W), wave)
            n = b.div(c, i(C))
            jj = b.sub(c, b.mul(n, i(C)))
            src = b.add(b.mul(n, i(K * 2)), b.mul(jj, i(1024)))
            dst = b.smem_ptr_add(
                lds_base, b.sext(b.add(b.mul(n, i(C * 1024)), b.mul(jj, i(1024))), I64)
            )
            if total % W:
                with b.scf_if(b.cmp_lt(c, i(total))):
                    b.global_load_lds(ptrs["X"], b.add(src, lane16), dst, 16, 0)
            else:
                b.global_load_lds(ptrs["X"], b.add(src, lane16), dst, 16, 0)
        if spec.pin:
            b.sched_barrier(0)
        n_extra = extra()
        if spec.pin:
            b.sched_barrier(0)
        b.s_waitcnt(vmcnt=min(63, n_extra))
        b.sync_lds_only(fence="compiler")

    def mfmas(acc, wv, av):
        if spec.prio == 1:
            b.s_setprio(1)
        for u in range(len(wv[0])):
            if spec.morder == "hn":
                order = [
                    (n, y, h) for h in range(2) for n in range(A) for y in range(Y)
                ]
            else:
                order = [
                    (n, y, h) for n in range(A) for y in range(Y) for h in range(2)
                ]
            for n, y, h in order:
                acc[n][y] = b.mfma_f32_4x4x4_bf16(
                    _half4(b, av[n][u], h), _half4(b, wv[y][u], h), acc[n][y]
                )
        if spec.prio == 1:
            b.s_setprio(0)

    def reduce_store(acc, m):
        for n in range(A):
            for y in range(Y):
                e = [b.vec_extract(acc[n][y], q) for q in range(4)]
                a = e[0]
                for q in (
                    1,
                    2,
                    3,
                ):  # the diagonal of each 4x4 block into its first lane
                    a = b.fadd(a, _dpp(b, e[q], q))
                a = b.fadd(a, _dpp(b, a, 4))
                a = b.fadd(a, _dpp(b, a, 8))
                s_ = [b.readlane(a, i(16 * q)) for q in range(4)]
                a = b.fadd(b.fadd(s_[0], s_[1]), b.fadd(s_[2], s_[3]))
                row = b.add(m, i(y))
                with b.scf_if(b.land(b.cmp_eq(lane, i(0)), b.cmp_lt(row, i(R)))):
                    b.buffer_store_bf16(
                        ry,
                        b.mul(b.add(i(n * R), row), i(2)),
                        i(0),
                        b.trunc_f32_to_bf16(a),
                    )

    def body(m, carried):
        """One row group (rows m .. m + Y - 1) over all of K; returns the
        loads issued for the next row group (xpf)."""
        vo = row_voffs(m)
        q = dict(carried or {})
        for s in range(D):
            if s not in q:
                q[s] = issue(vo, s)
        zero = b.vec_pack([b.const_f32(0.0)] * 4, F32)
        acc = [[zero] * Y for _ in range(A)]
        nxt, vo_n, aq = {}, [None], {}

        def ahead(t):
            if t < S:
                if t not in q:
                    q[t] = issue(vo, t)
                return
            if xpf and t - S < D and (t - S) not in nxt:
                if vo_n[0] is None:
                    vo_n[0] = row_voffs(b.add(m, i(mstep)))
                nxt[t - S] = issue(vo_n[0], t - S)

        for s in range(S):
            ahead(s + D)
            if staged:
                for s2 in range(s, min(S, s + spec.lpf + 1)):
                    if s2 not in aq:
                        aq[s2] = act_reads(s2)
            if spec.pin:
                b.sched_barrier(0)
            wv, av = q.pop(s)
            mfmas(acc, wv, aq.pop(s) if staged else av)
            if spec.pin:
                b.sched_barrier(0)
        reduce_store(acc, m)
        return nxt

    def flat(d):
        vals = []
        for s in range(D):
            got, acts = d[s]
            vals += [v for row in got for v in row]
            vals += [v for row in (acts or []) for v in row]
        return vals

    def unflat(vals):
        d, k = {}, 0
        for s in range(D):
            nj = len(steps[s])
            got = [vals[k + y * nj : k + (y + 1) * nj] for y in range(Y)]
            k += Y * nj
            acts = None
            if not staged:
                acts = [vals[k + n * nj : k + (n + 1) * nj] for n in range(A)]
                k += A * nj
            d[s] = (got, acts)
        return d

    vo0 = row_voffs(m0)
    pre = {}
    if staged:

        def prologue():
            for s in range(D0):
                pre[s] = issue(vo0, s)
            return sum(nloads(s) for s in range(D0))

        stage_x(prologue)
    else:
        pre = {s: issue(vo0, s) for s in range(D0)}

    def rows():
        nxt = body(m0, pre)
        if rounds > 1:
            if xpf:
                loop = b.scf_for_iter(
                    b.add(m0, i(mstep)),
                    i(R),
                    i(mstep),
                    [(f"pf{k}", v) for k, v in enumerate(flat(nxt))],
                    iv_name="mrow",
                )
                with loop as (m, carried):
                    b.scf_yield(*flat(body(m, unflat(carried))))
            else:
                with b.scf_for(
                    b.add(m0, i(mstep)), i(R), i(mstep), iv_name="mrow"
                ) as m:
                    body(m, None)

    # Every wave runs the body, idle ones too (waves past WE start at row N:
    # their loads are out of bounds and their stores guarded). No branch around
    # it: LLVM would sink the prologue's weight loads into the branch, past the
    # X staging wait, which then no longer covers the X DMA.
    rows()


def _build_smallm(b, spec, ptrs):
    p = smallm_plan(spec)
    M, N, K = spec.M, spec.N, spec.K
    RG, MG, NW, KWV, SK = (
        spec.row_groups,
        p["MG"],
        spec.n_waves,
        spec.k_waves,
        spec.split_k,
    )
    rows_w, block_n, n_tiles = p["rows_w"], p["block_n"], p["n_tiles"]
    k_wg, kw_len = p["k_wg"], p["kw_len"]
    waves, threads = p["waves"], p["threads"]
    b.kernel.attrs["max_workgroup_size"] = threads
    i = b.const_i32
    rw = b.buffer_rsrc(ptrs["W"], i(N * K * 2))
    rx = b.buffer_rsrc(ptrs["X"], i(M * K * 2))
    ry = b.buffer_rsrc(ptrs["Y"], i(M * N * 2))
    tid = b.thread_id_x()
    lane = b.land(tid, i(63))
    r16 = b.land(lane, i(15))  # row of the A (W) operand, column of the B (X) operand
    g4 = b.lshr(lane, i(4))  # which 8-element quarter of the 32-element step
    wave = b.readfirstlane(b.lshr(tid, i(6)))
    nw = b.mod(wave, i(NW)) if NW > 1 else i(0)
    kw = b.div(wave, i(NW)) if KWV > 1 else i(0)
    bid = b.block_id_x()
    if SK == 1:
        tile, ks = bid, i(0)
    elif spec.wg_order == "split":
        tile, ks = b.div(bid, i(SK)), b.mod(bid, i(SK))
    else:
        tile, ks = b.mod(bid, i(n_tiles)), b.div(bid, i(n_tiles))
        if spec.wg_order == "skew":  # the last split's row blocks shifted by one
            shifted = b.select(b.cmp_eq(tile, i(n_tiles - 1)), i(0), b.add(tile, i(1)))
            tile = b.select(b.cmp_eq(ks, i(SK - 1)), shifted, tile)
    kbase = b.add(
        b.mul(ks, i(k_wg)), b.mul(kw, i(kw_len))
    )  # first K element of this wave
    block_rows = p["block_rows"]
    masked = block_rows < block_n  # some wave rows fall past the workgroup's block
    row0 = b.add(
        b.mul(tile, i(block_rows)), b.mul(nw, i(rows_w))
    )  # first W row of this wave
    loc0 = b.mul(nw, i(rows_w)) if masked else None  # its row within the block
    g4x16 = b.mul(g4, i(16))
    # lds_swz: byte offset of a lane's 16 B in a dense row for steps t with
    # t % 4 == q (chunk (4 t + g4) ^ r16; steps t + 4 add 256 B).
    swz = (
        [b.mul(b.xor(b.xor(g4, r16), i(4 * q)), i(16)) for q in range(4)]
        if spec.lds_swz
        else None
    )
    lds_w = spec.w_stage == "lds"
    voff_w = []
    for rg in range(0 if lds_w else RG):
        row = b.add(row0, b.add(i(rg * 16), r16))
        off = b.add(b.mul(row, i(K * 2)), b.add(b.mul(kbase, i(2)), g4x16))
        ok = b.cmp_lt(row, i(N))
        if masked:
            ok = b.land(
                ok, b.cmp_lt(b.add(loc0, b.add(i(rg * 16), r16)), i(block_rows))
            )
        voff_w.append(b.select(ok, off, i(OOB)))
    m_l = [b.add(i(mg * 16), r16) for mg in range(MG)]
    m_ok = [b.cmp_lt(m, i(M)) for m in m_l]
    zero_w = b.vec_pack([i(0)] * 4, I32)

    xs = None
    if spec.x_stage == "global":
        voff_x = []
        for mg in range(MG):
            off = b.add(b.mul(m_l[mg], i(K * 2)), b.add(b.mul(kbase, i(2)), g4x16))
            voff_x.append(b.select(m_ok[mg], off, i(OOB)))
    elif spec.x_stream == 2:
        xph = [
            b.smem_alloc(I32, [p["x_dmas"] * 256], name_hint=f"dg_x{f}")
            for f in range(p["x_slots"])
        ]
        xidx = []
        for mg in range(MG):
            m_safe = b.select(m_ok[mg], m_l[mg], i(0)) if (mg + 1) * 16 > M else m_l[mg]
            row = b.mul(m_safe, i(spec.w_piece // 2))
            xidx.append([b.add(row, b.lshr(o, i(2))) for o in swz])
    elif spec.x_stream:
        # One allocation per 512-element phase of the wave K range: [k_wave][m]
        # rows of 1 KB + 16 B, so LLVM waits for one phase's DMAs at a time.
        xph = [
            b.smem_alloc(I32, [KWV * M * 260], name_hint=f"dg_x{f}")
            for f in range(p["x_phases"])
        ]
        xidx = []
        for mg in range(MG):
            m_safe = b.select(m_ok[mg], m_l[mg], i(0)) if (mg + 1) * 16 > M else m_l[mg]
            row = b.add(b.mul(kw, i(M)), m_safe) if KWV > 1 else m_safe
            xidx.append(b.add(b.mul(row, i(260)), b.mul(g4, i(4))))
    else:
        pitch_w = p["x_pitch"] // 4
        xs = b.smem_alloc(I32, [M * pitch_w], name_hint="dg_x")
        lds_x = b.smem_addr_of(xs)
        kin = b.lshr(
            b.add(b.mul(kw, i(kw_len)), b.mul(g4, i(8))), i(1)
        )  # words into a staged row
        xidx = []
        for mg in range(MG):
            m_safe = b.select(m_ok[mg], m_l[mg], i(0)) if (mg + 1) * 16 > M else m_l[mg]
            if swz is None:
                xidx.append(b.add(b.mul(m_safe, i(pitch_w)), kin))
                continue
            row = b.add(b.mul(m_safe, i(pitch_w)), b.mul(kw, i(kw_len // 2)))
            xidx.append([b.add(row, b.lshr(o, i(2))) for o in swz])

    def wload(rg, s):
        return b.buffer_load_cp(rw, voff_w[rg], i(s * 64), dwords=4, aux=spec.w_cache)

    def issue(s):
        if spec.prio == 2:
            b.s_setprio(1)
        w = [wload(rg, s) for rg in range(RG)]
        x = None
        if spec.x_stage == "global":
            x = [
                b.buffer_load_cp(rx, voff_x[mg], i(s * 64), dwords=4, aux=0)
                for mg in range(MG)
            ]
        if spec.prio == 2:
            b.s_setprio(0)
        return w, x

    def xread(s):
        out = []
        for mg in range(MG):
            if spec.x_stream == 2:
                f, t = divmod(s, spec.w_piece // SMALLM_STEP_K)
                v = b.smem_load_vN(
                    xph[f % len(xph)], b.add(xidx[mg][t % 4], i(t // 4 * 64)), dtype=I32, n=4
                )
            elif spec.x_stream:
                f, sl = divmod(s, GEMV_CHUNK_K // SMALLM_STEP_K)
                v = b.smem_load_vN(xph[f], b.add(xidx[mg], i(sl * 16)), dtype=I32, n=4)
            elif swz is not None:
                v = b.smem_load_vN(xs, b.add(xidx[mg][s % 4], i(s // 4 * 64)), dtype=I32, n=4)
            else:
                v = b.smem_load_vN(xs, b.add(xidx[mg], i(s * 16)), dtype=I32, n=4)
            if (mg + 1) * 16 > M:
                v = b.select(m_ok[mg], v, zero_w)
            out.append(v)
        return out

    def stage_x(extra):
        """The workgroup's X rows (K range ``k_wg``) into LDS, 1 KB per DMA,
        spread over the waves; ``extra()`` issues weight loads that overlap
        the copy and returns how many."""
        chunks = k_wg // 512
        total = M * chunks
        lane16 = b.mul(lane, i(16))
        for sl in range(_ceil(total, waves)):
            c = b.add(i(sl * waves), wave)
            m = b.div(c, i(chunks)) if chunks > 1 else c
            jj = b.sub(c, b.mul(m, i(chunks))) if chunks > 1 else i(0)
            src = b.add(
                b.add(b.mul(m, i(K * 2)), b.mul(ks, i(k_wg * 2))), b.mul(jj, i(1024))
            )
            dst = b.smem_ptr_add(
                lds_x, b.sext(b.add(b.mul(m, i(p["x_pitch"])), b.mul(jj, i(1024))), I64)
            )
            if swz is not None:  # lane l fetches chunk l ^ (m & 15) of its 1 KB
                lane16 = b.mul(b.xor(lane, b.land(m, i(15))), i(16))
            if total % waves:
                with b.scf_if(b.cmp_lt(c, i(total))):
                    b.global_load_lds(ptrs["X"], b.add(src, lane16), dst, 16, 0)
            else:
                b.global_load_lds(ptrs["X"], b.add(src, lane16), dst, 16, 0)
        if spec.pin:
            b.sched_barrier(0)
        n_extra = extra()
        if spec.pin:
            b.sched_barrier(0)
        b.s_waitcnt(vmcnt=min(63, n_extra))
        b.sync_lds_only(fence="compiler")

    def stage_x_phase(f):
        """Phase ``f`` of X (K elements ``512 f`` on of every k-wave's range)
        into its allocation, 1 KB per DMA over the waves, no wait. Returns the
        DMAs every wave issued (the fewest, for a safe vmcnt)."""
        total = KWV * M
        lane16 = b.mul(lane, i(16))
        lds = b.smem_addr_of(xph[f])
        for sl in range(_ceil(total, waves)):
            c = b.add(i(sl * waves), wave)  # [k_wave][m] row of the phase
            kq = b.div(c, i(M)) if KWV > 1 else i(0)
            m = b.sub(c, b.mul(kq, i(M))) if KWV > 1 else c
            src = b.add(
                b.add(b.mul(m, i(K * 2)), b.mul(ks, i(k_wg * 2))),
                b.mul(b.add(b.mul(kq, i(kw_len)), i(f * GEMV_CHUNK_K)), i(2)),
            )
            dst = b.smem_ptr_add(lds, b.sext(b.mul(c, i(1040)), I64))
            if (sl + 1) * waves > total:
                with b.scf_if(b.cmp_lt(c, i(total))):
                    b.global_load_lds(ptrs["X"], b.add(src, lane16), dst, 16, 0)
            else:
                b.global_load_lds(ptrs["X"], b.add(src, lane16), dst, 16, 0)
        return total // waves

    def stage_x_piece(f):
        """x_stream=2: phase ``f`` (the K range of W piece ``f``) into slot
        ``f % x_slots``, 1 KB DMAs of x_rows dense rows (chunk c of row r
        fetched into chunk c ^ (r & 15)) spread over the waves, every wave the
        same count (the spare DMAs repeat the last one), no wait."""
        R, nd = p["x_rows"], p["x_dmas"]
        per_row = 64 // R
        lds = b.smem_addr_of(xph[f % len(xph)])
        r_in = b.lshr(lane, i(per_row.bit_length() - 1)) if R > 1 else i(0)
        chunk = b.land(lane, i(per_row - 1))
        k0 = b.mul(b.add(kbase, i(f * spec.w_piece)), i(2))
        for sl in range(_ceil(nd, waves)):
            d = b.add(i(sl * waves), wave)
            if (sl + 1) * waves > nd:
                d = b.select(b.cmp_lt(d, i(nd)), d, i(nd - 1))
            r = b.add(b.mul(d, i(R)), r_in)
            src = b.add(
                b.add(b.mul(b.select(b.cmp_lt(r, i(M)), r, i(M - 1)), i(K * 2)), k0),
                b.mul(b.xor(chunk, b.land(r, i(15))), i(16)),
            )
            dst = b.smem_ptr_add(lds, b.sext(b.mul(d, i(1024)), I64))
            b.global_load_lds(ptrs["X"], src, dst, 16, 0)
        return _ceil(nd, waves)

    if lds_w:
        acc = _smallm_lds_w_loop(
            b,
            spec,
            p,
            stage_x,
            xread,
            rw,
            wave,
            lane,
            r16,
            g4,
            row0,
            loc0,
            kbase,
            {0: None, 1: stage_x_phase, 2: stage_x_piece}[spec.x_stream],
            swz,
        )
    else:
        acc = _smallm_direct_loop(b, spec, p, stage_x, xread, issue)
    _smallm_epilogue(
        b,
        spec,
        p,
        ptrs,
        acc,
        i,
        tid,
        g4,
        nw,
        kw,
        tile,
        ks,
        ry,
        m_l,
        m_ok,
        row0,
        loc0,
        masked,
        rw,
        rx,
    )


def _smallm_direct_loop(b, spec, p, stage_x, xread, issue):
    """The K loop with direct W loads: ``pf`` steps of loads ahead of the MFMAs."""
    RG, MG, S, D = spec.row_groups, p["MG"], p["S"], p["D"]
    q = {}

    def prologue():
        for s in range(D):
            q[s] = issue(s)
            if spec.pin == 2 or (spec.pin and spec.x_stage == "global" and MG > 1):
                # Without a fence per step LLVM may sink an early step's load
                # behind later steps of the window (seen with two X groups from
                # global, and with 32 weight loads), and that step waits for most
                # of the window.
                b.sched_barrier(0)
        return D * (RG + (MG if spec.x_stage == "global" else 0))

    if spec.x_stage == "lds":
        stage_x(prologue)
    else:
        prologue()
    zero = b.vec_pack([b.const_f32(0.0)] * 4, F32)
    accs = [[[zero] * MG for _ in range(RG)] for _ in range(spec.acc_split)]
    xq = {}
    for s in range(S):
        acc = accs[s % spec.acc_split]
        if s + D < S and (s + D) not in q:
            q[s + D] = issue(s + D)
        if spec.x_stage == "lds":
            for s2 in range(s, min(S, s + spec.lpf + 1)):
                if s2 not in xq:
                    xq[s2] = xread(s2)
        if spec.pin:
            b.sched_barrier(0)
        w, x = q.pop(s)
        if x is None:
            x = xq.pop(s)
        if spec.prio == 1:
            b.s_setprio(1)
        for mg in range(MG):
            xb = b.bitcast(x[mg], VectorType(BF16, 8))
            for rg in range(RG):
                acc[rg][mg] = b.mfma_f32_16x16x32_bf16(
                    b.bitcast(w[rg], VectorType(BF16, 8)), xb, acc[rg][mg]
                )
        if spec.prio == 1:
            b.s_setprio(0)
        if spec.pin:
            b.sched_barrier(0)
    return _sum_accs(b, accs)


def _sum_accs(b, accs):
    """acc_split's chains (one per K step parity) added into one before the epilogue."""
    acc = accs[0]
    for more in accs[1:]:
        acc = [[b.fadd(a, m) for a, m in zip(ar, mr)] for ar, mr in zip(acc, more)]
    return acc


def _smallm_lds_w_loop(
    b, spec, p, stage_x, xread, rw, wave, lane, r16, g4, row0, loc0, kbase, stage_xp, swz=None
):
    """The K loop with LDS-staged W: per piece of ``w_piece`` K elements a wave
    DMAs its rows into its own region of ring slot ``piece % ring`` (each DMA
    one 1 KB block of whole lines; rows past N or the block load nothing),
    waits for that piece only, then reads the 16x16x32 A operand with
    ``ds_read_b128`` (1040 B blocks: rows of a quarter-wave on distinct banks).
    No barrier: a wave reads only its own region."""
    i = b.const_i32
    N, K, RG, MG = spec.N, spec.K, spec.row_groups, p["MG"]
    R, ndma, P, ring = p["w_rows"], p["w_dmas"], p["pieces"], p["ring"]
    block_rows, waves = p["block_rows"], p["waves"]
    spp = spec.w_piece // SMALLM_STEP_K
    b.kernel.attrs["lds_alias_scopes"] = (
        True  # LLVM then waits only for DMAs into the slot read
    )
    ilv = bool(spec.w_ilv)
    pitch = p["w_pitch"]  # bytes per DMA block
    slots = [
        b.smem_alloc(I32, [waves * ndma * (pitch // 4)], name_hint=f"dg_w{k}")
        for k in range(ring)
    ]
    bases = [b.smem_addr_of(s) for s in slots]
    wave_off = b.mul(wave, i(ndma * pitch))
    per_row = 64 // R  # lanes per row in one DMA
    if ilv:  # lane l: row l % R, 16 B chunk l / R (LDS: lane order)
        lane_row = b.land(lane, i(R - 1))
        lane_col = b.mul(b.lshr(lane, i(R.bit_length() - 1)), i(16))
    else:  # lane l: row l / per_row, chunk l % per_row
        lane_row = b.lshr(lane, i(per_row.bit_length() - 1)) if R > 1 else None
        lane_chunk = b.land(lane, i(per_row - 1))
        lane_col = b.mul(lane_chunk, i(16))
    dma_voff = []
    for j in range(ndma):
        w = b.add(i(j * R), lane_row) if R > 1 else i(j)  # row of the wave
        grow = b.add(row0, w)
        ok = b.cmp_lt(grow, i(N))
        if loc0 is not None:
            ok = b.land(ok, b.cmp_lt(b.add(loc0, w), i(block_rows)))
        col = lane_col
        if swz is not None:  # LDS chunk l % per_row of row w holds chunk (l % per_row) ^ (w & 15)
            col = b.mul(b.xor(lane_chunk, b.land(w, i(15))), i(16))
        off = b.add(b.mul(grow, i(K * 2)), b.add(b.mul(kbase, i(2)), col))
        dma_voff.append(b.select(ok, off, i(OOB)))
    abase = []  # word index of a lane's 16 B of step 0 in its region, per row group
    for rg in range(RG):
        w = b.add(i(rg * 16), r16)
        if swz is not None:  # dense rows; one index per step t % 4 (w & 15 == r16)
            row = b.add(wave_off, b.mul(w, i(2 * spec.w_piece)))
            abase.append([b.lshr(b.add(row, o), i(2)) for o in swz])
            continue
        blk = b.lshr(w, i(R.bit_length() - 1)) if R > 1 else w
        byte = b.add(wave_off, b.mul(blk, i(pitch)))
        if ilv:  # chunk c of row w at (c R + w % R) 16 B; a step is 4 chunks
            byte = b.add(byte, b.mul(b.land(w, i(R - 1)), i(16)))
            abase.append(b.lshr(b.add(byte, b.mul(g4, i(16 * R))), i(2)))
            continue
        if R > 1:
            byte = b.add(byte, b.mul(b.land(w, i(R - 1)), i(2 * spec.w_piece)))
        abase.append(b.lshr(b.add(byte, b.mul(g4, i(16))), i(2)))
    step_words = 16 * R if ilv else 16
    tail = P - 1 if spec.w_tail else -1  # piece staged as two 64-element halves
    half = ndma // 2  # DMAs per half: 8 rows of 128 B each
    if tail >= 0:
        tbufs = [
            b.smem_alloc(I32, [waves * half * 256], name_hint=f"dg_wt{h}")
            for h in range(2)
        ]
        tptrs = [b.smem_addr_of(s) for s in tbufs]
        t_wave_off = b.mul(wave, i(half * 1024))
        t_row, t_chunk = b.lshr(lane, i(3)), b.land(lane, i(7))
        tail_voff = []
        for j in range(half):
            w = b.add(i(j * 8), t_row)
            grow = b.add(row0, w)
            ok = b.cmp_lt(grow, i(N))
            if loc0 is not None:
                ok = b.land(ok, b.cmp_lt(b.add(loc0, w), i(block_rows)))
            col = b.mul(b.xor(t_chunk, b.land(b.lshr(w, i(1)), i(7))), i(16))
            off = b.add(b.mul(grow, i(K * 2)), b.add(b.mul(kbase, i(2)), col))
            tail_voff.append(b.select(ok, off, i(OOB)))
        t_swz = b.land(b.lshr(r16, i(1)), i(7))
        tbase = [  # word index of a lane's 16 B of half step tt, per row group
            [
                b.lshr(
                    b.add(
                        b.add(t_wave_off, b.mul(b.add(i(rg * 16), r16), i(128))),
                        b.mul(b.xor(b.add(g4, i(4 * tt)), t_swz), i(16)),
                    ),
                    i(2),
                )
                for tt in range(2)
            ]
            for rg in range(RG)
        ]

    def a_index(rg, t):
        if swz is not None:
            return b.add(abase[rg][t % 4], i(t // 4 * 64))
        return b.add(abase[rg], i(step_words * t))

    def issue_piece(pc):
        if pc == tail:
            for h in range(2):
                for j in range(half):
                    lds = b.smem_ptr_add(
                        tptrs[h], b.sext(b.add(t_wave_off, i(j * 1024)), I64)
                    )
                    b.async_buffer_load_lds_addr(
                        rw,
                        lds,
                        tail_voff[j],
                        i(pc * spec.w_piece * 2 + h * 128),
                        dwords=4,
                        coherency=spec.w_cache & 3,
                    )
            return ndma
        for j in range(ndma):
            lds = b.smem_ptr_add(
                bases[pc % ring], b.sext(b.add(wave_off, i(j * pitch)), I64)
            )
            b.async_buffer_load_lds_addr(
                rw,
                lds,
                dma_voff[j],
                i(pc * spec.w_piece * 2),
                dwords=4,
                coherency=spec.w_cache & 3,
            )
        return ndma

    log = None  # x_stream: (kind, index, DMAs per wave) in issue order
    per_piece = spec.x_stream == 2
    if stage_xp is None:
        stage_x(lambda: sum(issue_piece(pc) for pc in range(ring)))
    elif per_piece:
        log = []
        for pc in range(ring):
            log += [("x", pc, stage_xp(pc)), ("w", pc, issue_piece(pc))]
    else:
        log = [("x", 0, stage_xp(0))]
        log += [("w", pc, issue_piece(pc)) for pc in range(ring)]
        log += [("x", f, stage_xp(f)) for f in range(1, p["x_phases"])]
    x_ready = set()
    zero = b.vec_pack([b.const_f32(0.0)] * 4, F32)
    accs = [[[zero] * MG for _ in range(RG)] for _ in range(spec.acc_split)]
    for pc in range(P):
        if pc == tail:  # the first half has landed
            b.s_waitcnt(vmcnt=half)
        elif log is None:
            b.s_waitcnt(
                vmcnt=min(63, (min(pc + ring, P) - pc - 1) * ndma)
            )  # this piece has landed
        else:
            f = pc if per_piece else pc * spec.w_piece // GEMV_CHUNK_K
            need = {("w", pc)} | (set() if f in x_ready else {("x", f)})
            last = max(n for n, (k, j, _) in enumerate(log) if (k, j) in need)
            b.s_waitcnt(vmcnt=min(63, sum(c for _, _, c in log[last + 1 :])))
            if f not in x_ready:  # every wave's DMAs of the phase have landed
                b.sync_lds_only(fence="compiler")
                x_ready.add(f)

        def reads(t, pc=pc):
            if pc == tail:
                a = [
                    b.smem_load_vN(
                        tbufs[t // 2], tbase[rg][t % 2], dtype=I32, n=4
                    )
                    for rg in range(RG)
                ]
                return xread(pc * spp + t), a
            a = [
                b.smem_load_vN(slots[pc % ring], a_index(rg, t), dtype=I32, n=4)
                for rg in range(RG)
            ]
            return xread(pc * spp + t), a

        rq = {}
        for t in range(spp):
            end = spp
            if pc == tail:
                if t == spp // 2:  # the second half has landed
                    b.s_waitcnt(vmcnt=0)
                end = spp // 2 if t < spp // 2 else spp
            # LDS reads lpf steps ahead, within the piece (the next piece waits first)
            for t2 in range(t, min(end, t + spec.lpf + 1)):
                if t2 not in rq:
                    rq[t2] = reads(t2)
            if spec.pin:
                b.sched_barrier(0)
            x, a = rq.pop(t)
            acc = accs[(pc * spp + t) % spec.acc_split]
            if spec.prio == 1:
                b.s_setprio(1)
            for mg in range(MG):
                xb = b.bitcast(x[mg], VectorType(BF16, 8))
                for rg in range(RG):
                    acc[rg][mg] = b.mfma_f32_16x16x32_bf16(
                        b.bitcast(a[rg], VectorType(BF16, 8)), xb, acc[rg][mg]
                    )
            if spec.prio == 1:
                b.s_setprio(0)
            if spec.pin:
                b.sched_barrier(0)
        if pc + ring < P:
            b.s_waitcnt(
                lgkmcnt=0
            )  # every read of this slot is done before it is refilled
            if per_piece:  # its X slot last held phase pc + ring - x_slots < pc (past this piece's barrier)
                log.append(("x", pc + ring, stage_xp(pc + ring)))
            n = issue_piece(pc + ring)
            if log is not None:
                log.append(("w", pc + ring, n))
    return _sum_accs(b, accs)


def _smallm_epilogue(
    b,
    spec,
    p,
    ptrs,
    acc,
    i,
    tid,
    g4,
    nw,
    kw,
    tile,
    ks,
    ry,
    m_l,
    m_ok,
    row0,
    loc0,
    masked,
    rw,
    rx,
):
    """Direct stores from the accumulators, or the LDS epilogue (k-waves, split-K)."""
    N, RG, MG = spec.N, spec.row_groups, p["MG"]
    block_rows = p["block_rows"]
    n_l = [
        b.add(row0, b.add(i(rg * 16), b.mul(g4, i(4)))) for rg in range(RG)
    ]  # first of 4 n per lane
    if p["epi"] == "direct":
        for rg in range(RG):
            for mg in range(MG):
                ok = b.land(m_ok[mg], b.cmp_lt(n_l[rg], i(N)))
                if masked:  # block_rows % 4 == 0: a lane's 4 rows are all in or all out
                    loc = b.add(loc0, b.add(i(rg * 16), b.mul(g4, i(4))))
                    ok = b.land(ok, b.cmp_lt(loc, i(block_rows)))
                voff = b.select(
                    ok, b.mul(b.add(b.mul(m_l[mg], i(N)), n_l[rg]), i(2)), i(OOB)
                )
                b.buffer_store_vN_bf16(
                    ry, voff, i(0), b.vec_trunc_f32_to_bf16(acc[rg][mg]), dwords=2
                )
        return
    regs = {
        "i": i,
        "tid": tid,
        "g4": g4,
        "nw": nw,
        "kw": kw,
        "tile": tile,
        "ks": ks,
        "ry": ry,
        "m_l": m_l,
        "rw": rw,
        "rx": rx,
    }
    _smallm_lds_epilogue(b, spec, p, ptrs, acc, regs)


def _smallm_lds_epilogue(b, spec, p, ptrs, acc, v):
    """Partial tiles of the k-waves into LDS ([k_wave][m][n], fp32), summed by
    the whole workgroup in output order; split_k == 1 stores bf16, else adds
    into the workspace and the last workgroup of the row block finishes it."""
    i, tid, g4, nw, kw, tile, ry = (
        v["i"],
        v["tid"],
        v["g4"],
        v["nw"],
        v["kw"],
        v["tile"],
        v["ry"],
    )
    M, N = spec.M, spec.N
    RG, MG, KWV, SK = spec.row_groups, p["MG"], spec.k_waves, spec.split_k
    rows_w, ep, threads = p["rows_w"], p["e_pitch"], p["threads"]
    mp = 16 * MG
    et = b.smem_alloc(F32, [KWV * mp * ep], name_hint="dg_epi")
    if spec.x_stage == "lds":
        # The LDS layout may place this tile over the staged X, whose last reads
        # end the K loop: every wave must be past them before any wave writes.
        b.sync_lds_only(fence="compiler")
    for rg in range(RG):
        n_loc = b.add(b.mul(nw, i(rows_w)), b.add(i(rg * 16), b.mul(g4, i(4))))
        for mg in range(MG):
            row = b.add(b.mul(kw, i(mp)), v["m_l"][mg])
            b.smem_store_vN(et, [b.add(b.mul(row, i(ep)), n_loc)], acc[rg][mg], 4)
    b.sync_lds_only(fence="compiler")
    block_rows, slots = p["block_rows"], spec.reduce == "slots"
    xcd = spec.sk_sync in ("xcd", "xcdrr")  # agent-scope ticket, plain partials
    relaxed = spec.sk_sync == "relaxed" or xcd
    # Device-coherent partial stores and loads.
    sc1 = 16 if spec.sk_sync == "relaxed" else 0
    rws = b.buffer_rsrc(ptrs["WS"], i(SK * M * N * 4)) if SK > 1 and slots else None
    vec = p["vec"] if SK == 1 or slots else 1
    per_row = block_rows // vec
    total = M * per_row
    n_base = b.mul(tile, i(block_rows))
    ks = v["ks"]
    for q in range(_ceil(total, threads)):
        e = b.add(tid, i(q * threads))
        m = b.div(e, i(per_row))
        n_loc = b.mul(b.sub(e, b.mul(m, i(per_row))), i(vec))
        ok = b.cmp_lt(b.add(n_base, n_loc), i(N))
        if total % threads:
            ok = b.land(ok, b.cmp_lt(e, i(total)))
            m = b.select(ok, m, i(0))
        parts = []
        for k in range(KWV):
            idx = b.add(b.mul(b.add(i(k * mp), m), i(ep)), n_loc)
            got = b.smem_load_vN(et, idx, dtype=F32, n=vec)
            parts.append([b.vec_extract(got, j) for j in range(vec)])
        sums = parts[0]
        for k in range(1, KWV):
            sums = [b.fadd(sa, sb) for sa, sb in zip(sums, parts[k])]
        n = b.add(n_base, n_loc)
        if SK == 1:
            voff = b.select(ok, b.mul(b.add(b.mul(m, i(N)), n), i(2)), i(OOB))
            _store_bf16(b, ry, voff, sums)
        elif slots:
            flat = b.add(b.mul(b.add(b.mul(ks, i(M)), m), i(N)), n)
            voff = b.select(ok, b.mul(flat, i(4)), i(OOB))
            _store_f32(b, rws, voff, sums, sc1)
        else:
            with b.scf_if(ok):
                b.global_atomic_add_f32(ptrs["WS"], b.add(b.mul(m, i(N)), n), sums[0])
    if SK == 1:
        return
    # Every wave's partial writes are performed before the ticket is taken.
    if relaxed:
        b.s_waitcnt(vmcnt=0)
    else:
        b.fence(scope="agent", ordering="release")
    if spec.sk_finish != "last":
        _shared_finish(b, spec, p, ptrs, tid, tile, ks, n_base, rws, ry)
        return
    ticket = b.smem_alloc(I32, [1], name_hint="dg_ticket")
    inc = None
    if spec.sk_sync == "xcd":
        # Ticket fields: partners (bits 0-5), sum of their XCC ids (6-15) and of
        # the squares (16-31). Both sums equal split_k times the finisher's id
        # (or its square) exactly when every partner ran on the finisher's XCC.
        xcc = b.land(b.s_getreg("XCC_ID"), i(0xF))
        inc = b.add(b.add(i(1), b.shl(xcc, i(6))), b.shl(b.mul(xcc, xcc), i(16)))
    b.sync_lds_only(fence="compiler")
    with b.scf_if(b.cmp_eq(tid, i(0))):
        if xcd:  # agent scope (no sc1)
            cnt = b.readfirstlane(b.global_ptr_to_i64(ptrs["CNT"]))
            old = b.global_atomic_add_scoped(
                cnt,
                b.mul(tile, i(4)),
                i(1) if inc is None else inc,
                scope="agent",
                returns=True,
            )
        else:
            old = b.global_atomic_add(ptrs["CNT"], tile, i(1), ordering="monotonic")
        b.smem_store_vN(ticket, [i(0)], old, 1)
    b.sync_lds_only(fence="compiler")
    last = b.readfirstlane(
        b.vec_extract(b.smem_load_vN(ticket, i(0), dtype=I32, n=1), 0)
    )
    colocated = None
    if spec.sk_sync == "xcd":
        full = b.add(last, inc)
        colocated = b.land(
            b.cmp_eq(b.land(b.lshr(full, i(6)), i(1023)), b.mul(xcc, i(SK))),
            b.cmp_eq(b.lshr(full, i(16)), b.mul(b.mul(xcc, xcc), i(SK))),
        )
        last = b.land(last, i(63))
    vf = p["vec"]
    per_row4 = block_rows // vf
    total4 = M * per_row4

    def finish(rws, load_aux):
        zeros = None if slots else _zero_words(b, vf)
        for q in range(_ceil(total4, threads)):
            e = b.add(tid, i(q * threads))
            m = b.div(e, i(per_row4))
            n = b.add(n_base, b.mul(b.sub(e, b.mul(m, i(per_row4))), i(vf)))
            ok = b.cmp_lt(n, i(N))
            if total4 % threads:
                ok = b.land(ok, b.cmp_lt(e, i(total4)))
            flat = b.add(b.mul(m, i(N)), n)
            voff4 = b.select(ok, b.mul(flat, i(4)), i(OOB))
            got = _load_f32(b, rws, voff4, vf, load_aux)
            for s in range(1, SK if slots else 1):  # the other splits' slots
                more = _load_f32(b, rws, b.add(voff4, i(s * M * N * 4)), vf, load_aux)
                got = b.fadd(got, more)
            voff2 = b.select(ok, b.mul(flat, i(2)), i(OOB))
            _store_bf16(b, ry, voff2, got if vf > 1 else [got])
            if not slots:  # clear the accumulator for the next call
                b.buffer_store_cp(zeros, rws, voff4, i(0), aux=sc1)

    with b.scf_if(b.cmp_eq(last, i(SK - 1))):
        if not relaxed:
            b.fence(scope="agent", ordering="acquire")
        if rws is None:
            rws = b.buffer_rsrc(ptrs["WS"], i(M * N * 4))
        if colocated is None:
            finish(rws, sc1)
        else:
            with b.scf_if_else(colocated) as (same_xcc, other_xcc):
                with same_xcc:
                    finish(rws, 0)
                with other_xcc:
                    # Partials of another XCC's L2 are not visible here: compute
                    # the row block again from W and X (and clear the
                    # accumulator with device-coherent stores).
                    _recompute_tile(b, spec, p, v, n_base)
                    if not slots:
                        _clear_tile(b, spec, p, rws, tid, n_base)
        with b.scf_if(b.cmp_eq(tid, i(0))):
            b.global_store(ptrs["CNT"], tile, i(0), align=4)


def _recompute_tile(b, spec, p, v, n_base):
    """The row block's whole-K product, the fallback of sk_sync="xcd": wave g
    (mod the waves) takes 16-row group g with direct 16x16x32 MFMA operand
    loads of W and X from global, ``RECOMPUTE_UNROLL`` steps per iteration of
    a K loop, and stores bf16 from the accumulators."""
    i = b.const_i32
    M, N, K = spec.M, spec.N, spec.K
    MG, waves, block_rows = p["MG"], p["waves"], p["block_rows"]
    tid, rw, rx, ry = v["tid"], v["rw"], v["rx"], v["ry"]
    lane = b.land(tid, i(63))
    r16, g4 = b.land(lane, i(15)), b.lshr(lane, i(4))
    wave = b.readfirstlane(b.lshr(tid, i(6)))
    m_l = [b.add(i(mg * 16), r16) for mg in range(MG)]
    m_ok = [b.cmp_lt(m, i(M)) for m in m_l]
    voff_x = [
        b.select(m_ok[mg], b.add(b.mul(m_l[mg], i(K * 2)), b.mul(g4, i(16))), i(OOB))
        for mg in range(MG)
    ]
    zero = b.vec_pack([b.const_f32(0.0)] * 4, F32)
    U = RECOMPUTE_UNROLL
    for g in range(_ceil(block_rows, 16)):
        with b.scf_if(b.cmp_eq(wave, i(g % waves))):
            loc = b.add(i(g * 16), r16)
            row = b.add(n_base, loc)
            ok = b.land(b.cmp_lt(row, i(N)), b.cmp_lt(loc, i(block_rows)))
            voff_w = b.select(ok, b.add(b.mul(row, i(K * 2)), b.mul(g4, i(16))), i(OOB))
            loop = b.scf_for_iter(
                i(0),
                i(K * 2),
                i(64 * U),
                [(f"rc{g}_{mg}", zero) for mg in range(MG)],
                iv_name=f"rck{g}",
            )
            with loop as (kb, accs):
                accs = list(accs)
                ws = [
                    b.buffer_load_cp(rw, voff_w, b.add(kb, i(64 * u)), dwords=4, aux=0)
                    for u in range(U)
                ]
                xs = [
                    [
                        b.buffer_load_cp(
                            rx, voff_x[mg], b.add(kb, i(64 * u)), dwords=4, aux=0
                        )
                        for mg in range(MG)
                    ]
                    for u in range(U)
                ]
                for u in range(U):
                    wb = b.bitcast(ws[u], VectorType(BF16, 8))
                    for mg in range(MG):
                        accs[mg] = b.mfma_f32_16x16x32_bf16(
                            wb, b.bitcast(xs[u][mg], VectorType(BF16, 8)), accs[mg]
                        )
                b.scf_yield(*accs)
            n_loc = b.add(
                i(g * 16), b.mul(g4, i(4))
            )  # 4 n per lane (block_rows % 4 == 0)
            n = b.add(n_base, n_loc)
            for mg in range(MG):
                okn = b.land(
                    m_ok[mg],
                    b.land(b.cmp_lt(n, i(N)), b.cmp_lt(n_loc, i(block_rows))),
                )
                voff = b.select(
                    okn, b.mul(b.add(b.mul(m_l[mg], i(N)), n), i(2)), i(OOB)
                )
                b.buffer_store_vN_bf16(
                    ry,
                    voff,
                    i(0),
                    b.vec_trunc_f32_to_bf16(loop.results[mg]),
                    dwords=2,
                )


def _clear_tile(b, spec, p, rws, tid, n_base):
    """Zero the row block's fp32 accumulator (reduce="atomic") with sc1 stores."""
    i = b.const_i32
    M, N, vf = spec.M, spec.N, p["vec"]
    per_row4 = p["block_rows"] // vf
    total4 = M * per_row4
    zeros = _zero_words(b, vf)
    for q in range(_ceil(total4, p["threads"])):
        e = b.add(tid, i(q * p["threads"]))
        m = b.div(e, i(per_row4))
        n = b.add(n_base, b.mul(b.sub(e, b.mul(m, i(per_row4))), i(vf)))
        ok = b.land(b.cmp_lt(n, i(N)), b.cmp_lt(e, i(total4)))
        voff4 = b.select(ok, b.mul(b.add(b.mul(m, i(N)), n), i(4)), i(OOB))
        b.buffer_store_cp(zeros, rws, voff4, i(0), aux=16)


def _shared_finish(b, spec, p, ptrs, tid, tile, ks, n_base, rws, ry):
    """sk_finish="shared": take the ticket (it counts up across calls: this
    call's partners bring it to the next multiple of split_k), spin until it
    gets there, then sum this partner's 1 / split_k of the row block over
    every slot (device-coherent loads) and store it."""
    i = b.const_i32
    M, N, SK, threads = spec.M, spec.N, spec.split_k, p["threads"]
    block_rows, vf = p["block_rows"], p["vec"]
    with b.scf_if(b.cmp_eq(tid, i(0))):
        cnt = b.readfirstlane(b.global_ptr_to_i64(ptrs["CNT"]))
        voff = b.add(b.mul(tile, i(8)), i(4))
        old = b.global_atomic_add_scoped(cnt, voff, i(1), scope="agent", returns=True)
        target = b.add(b.land(old, i(-SK)), i(SK))
        rcnt = b.buffer_rsrc(ptrs["CNT"], i(p["n_tiles"] * 8))
        b.tagged_poll(
            target,
            [(rcnt, b.mul(tile, i(8)), 2)],
            aux=17 if spec.sk_finish == "shared_sc" else 16,
            pause="s_sleep 1",
            reload="launder",
        )
    b.sync_lds_only(fence="compiler")
    per_row4 = block_rows // vf
    total4 = M * per_row4
    share = _ceil(total4, SK)  # vectors of the block this partner sums
    for q in range(_ceil(share, threads)):
        e_loc = b.add(tid, i(q * threads))
        e = b.add(b.mul(ks, i(share)), e_loc)
        m = b.div(e, i(per_row4))
        n = b.add(n_base, b.mul(b.sub(e, b.mul(m, i(per_row4))), i(vf)))
        ok = b.land(b.cmp_lt(n, i(N)), b.cmp_lt(e, i(total4)))
        if share % threads:
            ok = b.land(ok, b.cmp_lt(e_loc, i(share)))
        flat = b.add(b.mul(m, i(N)), n)
        voff4 = b.select(ok, b.mul(flat, i(4)), i(OOB))
        got = _load_f32(b, rws, voff4, vf, 16)
        for s in range(1, SK):
            more = _load_f32(b, rws, b.add(voff4, i(s * M * N * 4)), vf, 16)
            got = b.fadd(got, more)
        voff2 = b.select(ok, b.mul(flat, i(2)), i(OOB))
        _store_bf16(b, ry, voff2, got if vf > 1 else [got])


def _zero_words(b, n):
    """Zero data for an ``n``-dword buffer store (i32, or ``<n x i32>``)."""
    zero = b.const_i32(0)
    return zero if n == 1 else b.vec_pack([zero] * n, I32)


def _load_f32(b, rsrc, voff, n, aux):
    """``n`` (1, 2 or 4) fp32 words from a buffer: an f32, or ``<n x f32>``."""
    raw = b.buffer_load_cp(rsrc, voff, b.const_i32(0), dwords=n, aux=aux)
    return b.bitcast(raw, F32 if n == 1 else VectorType(F32, n))


def _store_f32(b, rsrc, voff, vals, aux):
    """``vals`` (a list of 1, 2 or 4 f32 values) into a buffer at byte offset ``voff``."""
    words = [b.bitcast(x, I32) for x in vals]
    data = words[0] if len(words) == 1 else b.vec_pack(words, I32)
    b.buffer_store_cp(data, rsrc, voff, b.const_i32(0), aux=aux)


def _store_bf16(b, rsrc, voff, vals):
    """bf16 of ``vals`` (a list of 1, 2 or 4 f32 values, or an ``<n x f32>``) at byte offset ``voff``."""
    zero = b.const_i32(0)
    if isinstance(vals, list) and len(vals) == 1:
        b.buffer_store_bf16(rsrc, voff, zero, b.trunc_f32_to_bf16(vals[0]))
        return
    vec = b.vec_pack(vals, F32) if isinstance(vals, list) else vals
    b.buffer_store_vN_bf16(
        rsrc, voff, zero, b.vec_trunc_f32_to_bf16(vec), dwords=vec.type.count // 2
    )


def knob_moves(spec: DecodeGemmSpec) -> list[DecodeGemmSpec]:
    """Every spec one knob step away from ``spec`` (for searches and tests):
    integer knobs up and down by one step of their natural scale, string
    knobs to each other value. Invalid moves are included; filter them with
    :func:`is_valid_spec`."""
    choices = {
        "family": FAMILIES,
        "x_stage": ("lds", "global"),
        "epi": ("auto", "direct", "lds"),
        "wg_order": ("split", "tile"),
        "gmap": ("wave", "block"),
        "stage": ("dma", "global"),
        "lorder": ("ky", "yk"),
        "morder": ("hn", "nh"),
        "backend": ("llvm", "direct"),
        "w_cache": W_CACHE_BITS,
        "preload": (0, 1),
        "dlevers": ("", "loadfirst", "vmemfirst", "loadorder", "storewait"),
        "reduce": ("atomic", "slots"),
        "sk_sync": ("fence", "relaxed", "xcd"),
        "sk_finish": ("last", "shared", "shared_sc", "epoch"),
        "w_stage": ("direct", "lds"),
        "w_piece": (128, 256, 512),
    }
    doubling = {"split_k", "grid", "cus", "xccs"}
    out = []
    for name in spec.relevant_fields()[3:]:
        cur = getattr(spec, name)
        if name in choices:
            vals = [c for c in choices[name] if c != cur]
        elif name in doubling:
            vals = [cur * 2, cur // 2] if cur > 0 else [1]
        else:
            vals = [cur + 1, cur - 1]
        for val in vals:
            if val != cur:
                out.append(replace(spec, **{name: val}))
    return out


__all__ = [
    "FAMILIES",
    "DecodeGemmSpec",
    "build_decode_gemm",
    "decode_gemm_launch",
    "decode_gemm_signature",
    "default_spec",
    "gemv_plan",
    "is_valid_spec",
    "knob_moves",
    "resolved_spec",
    "skinny_auto_split",
    "skinny_plan",
    "smallm_colocated",
    "smallm_plan",
]
# Every field of the spec is either the problem, common, or one family's knob.
assert set(_COMMON) | {f for fs in _FAMILY_FIELDS.values() for f in fs} == {
    f.name for f in fields(DecodeGemmSpec)
}
assert math.gcd(SMALLM_STEP_K, GEMV_CHUNK_K) == SMALLM_STEP_K

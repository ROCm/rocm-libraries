# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Register-direct split-K GEMM for skinny / decode / long-K RCR problems.

``C[M, N] = A[M, K] . B[N, K]^T`` (A and B K-contiguous, C row-major), 16-bit
inputs and output, f32 accumulation.

Why a separate body: when ``min(M, N)`` is small (decode / GEMV, skinny
projections) or the output has few tiles but K is long, the LDS-staged
universal body leaves most CUs idle (few output tiles) or spends its time in
tile prologue / epilogue. This body

* loads both operand fragments **straight from global memory into the MFMA
  operand registers** (no LDS): A and B are K-contiguous, so every lane's MFMA
  fragment (8 consecutive K of one row on gfx950, 4 on gfx942) is one 16-byte
  buffer load. ``kc`` > 8 lets a lane load ``kc`` contiguous K elements and
  feed ``kc / 8`` MFMAs with them: the K order inside a step is permuted, but
  A and B are permuted identically, so the dot product is unchanged;
* software-pipelines the loads ``prefetch`` steps ahead in registers;
* splits K three ways: ``warp_k`` waves of one workgroup (reduced through LDS),
  and ``SPLIT`` workgroups along grid z (a runtime argument);
* reduces the split-K partials **inside the same launch**: every workgroup
  writes its f32 partial tile to the workspace ``W``, releases it with an
  agent-scope fence and bumps a per-tile counter in ``Cnt``; the last
  workgroup to arrive sums the ``SPLIT`` partials in a FIXED order
  (z = 0 .. SPLIT-1, so the result is deterministic), rounds once to the
  output dtype, stores C and resets the counter to 0. ``Cnt`` therefore only
  has to be zeroed once when it is allocated; there is no separate memset or
  finalize kernel.
* puts B in the MFMA "A" operand slot so each lane's accumulator holds 4
  consecutive N of one output row (one 8-byte store per lane).

Bounds: every operand load goes through a buffer resource whose range ends at
the last valid row of this workgroup's tile, so rows past M / N read zeros;
the partial last K step is masked element-wise on both operands (``K`` need not
be a multiple of anything). Output stores are guarded per element at the N
edge. Unaligned rows (``K % 8 != 0``) rely on the CDNA unaligned buffer-access
mode (default under ROCm).

MN-contiguous operands (``layout`` ``RRR`` / ``CRR`` / ``CCR``, BLAS NN / TN /
TT): an operand stored ``[K, M]`` / ``[K, N]`` is read
in place, without a transpose pass. Its 16-byte loads take 8 consecutive M /
N elements of one K row (a K row of a wave's extent is one contiguous
segment, so the loads stay coalesced); each wave stages its loaded vectors in
its own LDS area (no barrier: no other wave reads it) and reads the MFMA
fragments back transposed, with ``ds_read_b64_tr_b16`` on gfx950 (two per
fragment) or eight 16-bit LDS reads on gfx942 (``mn_lds="u16"``). The
fragments then have exactly the K-contiguous lane layout, so the MFMA
stream, the split-K reduction and the epilogue are unchanged. Bounds: the
operand's buffer range ends at the last element this workgroup's K slice may
read; M / N positions past the extent read the next K row (feeding only
outputs that are never stored) and K rows past K are zeroed in the masked
partial step.

Python-only kernel family (no C++ engine mirror; it does not change the
emission of any existing family).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from ...core.ir import BF16, F16, F32, I32, I64, IRBuilder, KernelDef, PtrType, Value
from ...core.ir import I16, TemporalHint, VectorType
from ...helpers.spec import SignatureBuilder, kernel_name_join

__all__ = [
    "SkinnyGemmSpec",
    "build_gemm_skinny",
    "is_valid_spec",
    "skinny_launch_error",
    "skinny_grid",
    "skinny_signature",
    "skinny_steps",
    "skinny_kps",
    "skinny_workspace_floats",
    "skinny_counter_ints",
    "SKINNY_ARCHES",
]

SKINNY_ARCHES = ("gfx942", "gfx950")

# red_sync "sc1": partial-tile stores / loads per inline-asm group (each group
# ends with its own vmcnt(0))
_ASM_GROUP = 8


@dataclass(frozen=True)
class SkinnyGemmSpec:
    """One register-direct split-K GEMM configuration.

    Workgroup tile: ``tile_m = 16 * wave_m_atoms * warp_m`` rows of A (M) by
    ``tile_n = 16 * wave_n_atoms * warp_n`` rows of B (N); ``warp_k`` waves
    split the workgroup's K slice step by step. One K step covers
    ``4 * kc`` K elements (four 16-lane groups x ``kc``).
    """

    dtype: str = "bf16"  # "fp16" | "bf16": A, B and C
    wave_m_atoms: int = 1  # 16-row MFMA atoms per wave along M
    wave_n_atoms: int = 4  # 16-row MFMA atoms per wave along N
    warp_m: int = 1
    warp_n: int = 1
    warp_k: int = 1
    kc: int = 16  # contiguous K elements per lane per step (multiple of 8)
    prefetch: int = 2  # K steps loaded ahead (register pipeline depth)
    # ---- levers (default-off unless noted) ----
    waves_per_eu: int = 0  # amdgpu-waves-per-eu min (0 = compiler default)
    # ``nt_ws``: nontemporal (streaming) stores / loads of the f32 partials
    nt_ws: bool = False
    # ``b_load``: main-loop B loads: "buffer" (bounds-checked raw buffer
    # loads), "global", "global_nt" (global loads, nt = streaming hint)
    b_load: str = "buffer"
    # ``kmap``: K order of the lane loads inside a step ("mfma" or "lane")
    kmap: str = "mfma"
    # ``kalign``: K alignment the operand loads assume. 8: K % 8 == 0, one
    # 16-byte load per 8 K elements (fast path); 2: K % 2 == 0, the same
    # loads at 4-byte-aligned row starts; 1: any K (dword loads + a ushort,
    # realigned with v_alignbyte).
    kalign: int = 8
    # ``red_sync``: how the split-K partials cross workgroups. "agent":
    # agent-scope release / acquire fences (LLVM memory model; on gfx942 /
    # gfx950 these write back / invalidate the whole XCD L2); "sc1": the
    # partials are stored and loaded with agent-scope (sc1) cache policy,
    # so only a vmcnt(0) orders them before the ticket atomic.
    red_sync: str = "agent"
    # ``red_unroll``: partial tiles loaded per trip of the final reduction
    red_unroll: int = 4
    # ``layout``: operand storage (``RCR`` = A [M, K], B [N, K], both
    # K-contiguous; ``RRR`` = B [K, N]; ``CRR`` = A [K, M], B [K, N];
    # ``CCR`` = A [K, M], B [N, K]; see ``gemm_mn_operand.mn_layout_letters``)
    layout: str = "RCR"
    # ``mnalign``: alignment of an MN-contiguous operand's leading dimension
    # (M for A, N for B) the loads assume: 8 (16-byte loads), 2 (the same
    # loads at 4-byte boundaries), 1 (any; dword loads realigned with
    # v_alignbyte). Unused (must stay 8) without an MN-contiguous operand.
    mnalign: int = 8
    # ``mn_pad``: bytes of padding after every staged K row of an
    # MN-contiguous operand's LDS area (0 / 16 / 32: bank spread of the
    # transpose reads)
    mn_pad: int = 0
    # ``mn_lds``: how the MN-contiguous fragments are read back from LDS:
    # "tr" (ds_read_b64_tr_b16, gfx950) or "u16" (eight 16-bit reads per
    # fragment, any target)
    mn_lds: str = "tr"
    name: str = "rocke_gemm_skinny"
    # ``split``: the launch's split-K degree (grid z, the ``SPLIT`` argument).
    # Launch-only: it does not change the IR or the kernel name; a dispatcher
    # selection carries it so the result is a complete launch configuration.
    split: int = 1

    # ---- derived ----
    @property
    def tile_m(self) -> int:
        return 16 * self.wave_m_atoms * self.warp_m

    @property
    def tile_n(self) -> int:
        return 16 * self.wave_n_atoms * self.warp_n

    @property
    def step_k(self) -> int:
        return 4 * self.kc

    @property
    def waves(self) -> int:
        return self.warp_m * self.warp_n * self.warp_k

    @property
    def block_size(self) -> int:
        return 64 * self.waves

    def kernel_name(self) -> str:
        return kernel_name_join(
            self.name,
            "f16" if self.dtype == "fp16" else "bf16",
            f"a{self.wave_m_atoms}x{self.wave_n_atoms}",
            f"w{self.warp_m}x{self.warp_n}x{self.warp_k}",
            f"kc{self.kc}",
            f"pf{self.prefetch}",
            f"wpe{self.waves_per_eu}" if self.waves_per_eu else "",
            f"ru{self.red_unroll}" if self.red_unroll != 4 else "",
            f"ka{self.kalign}" if self.kalign != 8 else "",
            "kml" if self.kmap == "lane" else "",
            "rsc1" if self.red_sync == "sc1" else "",
            {"buffer": "", "global": "bgl", "global_nt": "bgnt"}[self.b_load],
            f"l{self.layout.lower()}" if self.layout.upper() != "RCR" else "",
            f"mna{self.mnalign}" if self.mnalign != 8 else "",
            f"mnp{self.mn_pad}" if self.mn_pad else "",
            "mnu16" if self.mn_lds == "u16" else "",
            flags={"ntws": self.nt_ws},
        )

    @property
    def mn_operands(self) -> Tuple[bool, bool]:
        """``(a_mn, b_mn)``: which operands are MN-contiguous."""
        lay = self.layout.upper()
        return lay[0] == "C", lay[1] == "R"


def _atom_k(arch: str) -> int:
    """K elements per lane per MFMA atom (16x16x32 on gfx950, 16x16x16 on gfx942)."""
    return 8 if arch == "gfx950" else 4


def is_valid_spec(spec: SkinnyGemmSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    """Static validity of ``spec`` on ``arch`` (shape-independent)."""
    if arch not in SKINNY_ARCHES:
        return False, f"gemm_skinny is built for {SKINNY_ARCHES}, not {arch}"
    if spec.dtype not in ("fp16", "bf16"):
        return False, f"dtype must be fp16 or bf16 (got {spec.dtype!r})"
    for nm in ("wave_m_atoms", "wave_n_atoms", "warp_m", "warp_n", "warp_k"):
        v = getattr(spec, nm)
        if not 1 <= int(v) <= 16:
            return False, f"{nm}={v} must be in [1, 16]"
    if spec.kc not in (8, 16, 32):
        return False, f"kc={spec.kc} must be 8, 16 or 32"
    if spec.b_load not in ("buffer", "global", "global_nt"):
        return False, f"b_load={spec.b_load!r} must be buffer, global or global_nt"
    if spec.red_sync not in ("agent", "sc1"):
        return False, f"red_sync={spec.red_sync!r} must be 'agent' or 'sc1'"
    if spec.kmap not in ("mfma", "lane"):
        return False, f"kmap={spec.kmap!r} must be 'mfma' or 'lane'"
    if spec.kalign not in (1, 2, 8):
        return False, f"kalign={spec.kalign} must be 1, 2 or 8"
    if spec.red_unroll not in (1, 2, 4, 8):
        return False, f"red_unroll={spec.red_unroll} must be 1, 2, 4 or 8"
    if not 1 <= spec.prefetch <= 4:
        return False, f"prefetch={spec.prefetch} must be in [1, 4]"
    if not 1 <= int(spec.split) <= 65535:
        return False, f"split={spec.split} must be in [1, 65535]"
    lay = str(spec.layout).upper()
    if lay not in ("RCR", "RRR", "CRR", "CCR"):
        return False, f"layout={spec.layout!r} must be RCR, RRR, CRR or CCR"
    a_mn, b_mn = spec.mn_operands
    if spec.mnalign not in (1, 2, 8):
        return False, f"mnalign={spec.mnalign} must be 1, 2 or 8"
    if spec.mn_pad not in (0, 16, 32):
        return False, f"mn_pad={spec.mn_pad} must be 0, 16 or 32"
    if spec.mn_lds not in ("tr", "u16"):
        return False, f"mn_lds={spec.mn_lds!r} must be 'tr' or 'u16'"
    if not (a_mn or b_mn):
        # canonical RCR form (no unused MN levers in the kernel name)
        if spec.mnalign != 8 or spec.mn_pad or spec.mn_lds != "tr":
            return False, "mnalign / mn_pad / mn_lds need an MN-contiguous operand"
    else:
        if spec.kmap != "mfma":
            return False, "an MN-contiguous operand needs kmap='mfma'"
        if b_mn and spec.b_load != "buffer":
            return False, "an N-contiguous B needs b_load='buffer'"
        if a_mn and b_mn and spec.kalign != 8:
            return False, "kalign is unused with both operands MN-contiguous (keep 8)"
        if spec.mn_lds == "tr" and arch != "gfx950":
            return False, f"mn_lds='tr' needs ds_read_b64_tr_b16 (gfx950), not {arch}"
    if spec.block_size > 1024:
        return False, f"block_size {spec.block_size} > 1024"
    # registers: accumulators + prefetch stages of fragments (VGPR budget 512
    # incl. AGPRs; leave headroom for addresses / epilogue)
    acc = spec.wave_m_atoms * spec.wave_n_atoms * 4
    frag = (spec.wave_m_atoms + spec.wave_n_atoms) * (spec.kc // 2)
    regs = acc + (spec.prefetch + 1) * frag
    if regs > 384:
        return False, f"register estimate {regs} > 384 (acc {acc}, frag/stage {frag})"
    from ...core.arch import ArchTarget

    lds = _lds_bytes(spec)
    cap = ArchTarget.from_gfx(arch).lds_capacity_bytes
    if lds > cap:
        return False, f"LDS {lds} B > {cap} B"
    return True, "ok"


def _lds_pad(spec: SkinnyGemmSpec) -> int:
    return 4  # f32 pad per LDS row (bank spread of the float4 rows)


def _mn_row_elems(spec: SkinnyGemmSpec, atoms: int) -> int:
    """Elements per staged K row of an MN-contiguous operand (the wave's MN
    extent plus the pad)."""
    return 16 * atoms + spec.mn_pad // 2


def _mn_stage_elems(spec: SkinnyGemmSpec) -> Tuple[int, int]:
    """Per-wave LDS staging elements of (A, B) (0 for a K-contiguous one):
    every piece of a step holds 32 K rows of the wave's MN extent."""
    a_mn, b_mn = spec.mn_operands
    pieces = spec.kc // 8
    ea = pieces * 32 * _mn_row_elems(spec, spec.wave_m_atoms) if a_mn else 0
    eb = pieces * 32 * _mn_row_elems(spec, spec.wave_n_atoms) if b_mn else 0
    return ea, eb


def _lds_bytes(spec: SkinnyGemmSpec) -> int:
    red = 0
    if spec.warp_k > 1:
        red = spec.warp_k * spec.tile_m * (spec.tile_n + _lds_pad(spec)) * 4
    ea, eb = _mn_stage_elems(spec)
    mn = spec.waves * (ea + eb) * 2
    return red + mn + 16  # + the ticket broadcast slot


def skinny_steps(spec: SkinnyGemmSpec, K: int) -> int:
    """K steps of ``spec`` for a K extent (the last one may be partial)."""
    return -(-int(K) // spec.step_k)


def skinny_kps(spec: SkinnyGemmSpec, K: int, split: int) -> int:
    """K steps per split slice."""
    return -(-skinny_steps(spec, K) // max(1, int(split)))


def skinny_grid(
    spec: SkinnyGemmSpec, M: int, N: int, split: int
) -> Tuple[int, int, int]:
    return (
        -(-int(N) // spec.tile_n),
        -(-int(M) // spec.tile_m),
        max(1, int(split)),
    )


def skinny_workspace_floats(spec: SkinnyGemmSpec, M: int, N: int, split: int) -> int:
    """f32 elements of the partial-tile workspace ``W`` (0 when split == 1)."""
    if int(split) <= 1:
        return 0
    gx, gy, _ = skinny_grid(spec, M, N, split)
    return int(split) * gx * gy * spec.tile_m * spec.tile_n


def skinny_counter_ints(spec: SkinnyGemmSpec, M: int, N: int) -> int:
    """i32 tile counters (zero them once at allocation; the kernel resets them)."""
    gx, gy, _ = skinny_grid(spec, M, N, 1)
    return gx * gy


def skinny_launch_error(
    spec: SkinnyGemmSpec, M: int, N: int, K: int, split: int
) -> Optional[str]:
    """Why ``(M, N, K, split)`` is outside the kernel contract, or ``None``."""
    M, N, K, split = int(M), int(N), int(K), int(split)
    if min(M, N, K) < 1:
        return f"M, N, K must be >= 1 (got {M}, {N}, {K})"
    if split < 1 or split > 65535:
        return f"split={split} must be in [1, 65535]"
    a_mn, b_mn = spec.mn_operands
    if K % spec.kalign and not (a_mn and b_mn):
        return f"K={K} is not a multiple of kalign={spec.kalign} (use kalign 1)"
    for is_mn, ld, nm in ((a_mn, M, "M"), (b_mn, N, "N")):
        if is_mn and ld % spec.mnalign:
            return (
                f"{nm}={ld} (MN-contiguous leading dimension) is not a multiple "
                f"of mnalign={spec.mnalign} (use mnalign 1)"
            )
    gx, gy, _ = skinny_grid(spec, M, N, split)
    if gy > 65535:
        return f"grid.y {gy} > 65535"
    # per-workgroup buffer ranges and offsets are i32 byte counts
    kc_rows = [t for t, mn in ((spec.tile_m, a_mn), (spec.tile_n, b_mn)) if not mn]
    if kc_rows and max(kc_rows) * K * 2 >= 2**31:
        return "tile rows x K x 2 bytes must fit in i32"
    if a_mn or b_mn:
        # an MN-contiguous operand's offsets span the workgroup's K slice:
        # (K steps of the slice + 1) x step x leading dimension x 2 bytes
        rows = (skinny_kps(spec, K, split) + 1) * spec.step_k
        for is_mn, ld in ((a_mn, M), (b_mn, N)):
            if is_mn and (rows * ld + 64) * 2 >= 2**31:
                return (
                    "K slice x MN leading dimension x 2 bytes must fit in i32 "
                    "(raise the split)"
                )
    if spec.tile_m * N * 2 >= 2**31:
        return "tile_m x N x 2 bytes must fit in i32"
    if skinny_workspace_floats(spec, M, N, split) >= 2**31:
        return "workspace too large for i32 indexing"
    if -(-K // spec.step_k) >= 2**31 // spec.step_k:
        return "K too large"
    return None


def skinny_signature(spec: SkinnyGemmSpec):
    dt = spec.dtype
    return (
        SignatureBuilder()
        .ptr("A", dt)
        .ptr("B", dt)
        .ptr("C", dt)
        .ptr("W", "f32")
        .ptr("Cnt", "i32")
        .scalar("M", "i32")
        .scalar("N", "i32")
        .scalar("K", "i32")
        .scalar("SPLIT", "i32")
        .scalar("KPS", "i32")
        .build()
    )


def build_gemm_skinny(spec: SkinnyGemmSpec, arch: str = "gfx950") -> KernelDef:
    """Build the IR of one register-direct split-K GEMM.

    Kernel arguments ``(A, B, C, W, Cnt, M, N, K, SPLIT, KPS)``; launch
    :func:`skinny_grid` x ``(block_size, 1, 1)``. ``KPS`` = K steps per split
    slice (:func:`skinny_kps`). With ``SPLIT == 1`` ``W`` / ``Cnt`` are not
    touched (any valid pointer). With ``SPLIT > 1`` ``W`` holds
    :func:`skinny_workspace_floats` floats (no init) and ``Cnt``
    :func:`skinny_counter_ints` i32 zeros (kept zero by the kernel).
    """
    ok, why = is_valid_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid gemm_skinny spec for {arch}: {why}")

    elem = BF16 if spec.dtype == "bf16" else F16
    akl = _atom_k(arch)  # K elements per lane per MFMA
    subs = 8 // akl  # MFMAs per 16-byte piece
    pieces = spec.kc // 8  # 16-byte pieces per fragment per step
    na, ma = spec.wave_n_atoms, spec.wave_m_atoms
    WM, WN, WK = spec.warp_m, spec.warp_n, spec.warp_k
    TM, TN = spec.tile_m, spec.tile_n
    STEP = spec.step_k
    PF = spec.prefetch
    mfma_name = (
        ("mfma_f32_16x16x32_bf16" if spec.dtype == "bf16" else "mfma_f32_16x16x32_f16")
        if arch == "gfx950"
        else (
            "mfma_f32_16x16x16_bf16"
            if spec.dtype == "bf16"
            else "mfma_f32_16x16x16_f16"
        )
    )

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block_size
    if spec.waves_per_eu:
        b.kernel.attrs["waves_per_eu"] = int(spec.waves_per_eu)

    A = b.param("A", PtrType(elem, "global"), noalias=True, readonly=True, align=2)
    Bp = b.param("B", PtrType(elem, "global"), noalias=True, readonly=True, align=2)
    C = b.param("C", PtrType(elem, "global"), noalias=True, align=2)
    W = b.param("W", PtrType(F32, "global"), align=16)
    Cnt = b.param("Cnt", PtrType(I32, "global"), align=4)
    M = b.param("M", I32)
    N = b.param("N", I32)
    K = b.param("K", I32)
    SPLIT = b.param("SPLIT", I32)
    KPS = b.param("KPS", I32)

    c0 = b.const_i32(0)
    c1 = b.const_i32(1)
    c2 = b.const_i32(2)

    tid = b.thread_id_x()
    lane = b.mod(tid, b.const_i32(64))
    wave = b.to_sgpr_u32(b.div(tid, b.const_i32(64)))
    # wave -> (wk, wm, wn): wn fastest
    wn = b.to_sgpr_u32(b.mod(wave, b.const_i32(WN)))
    wm = b.to_sgpr_u32(b.mod(b.div(wave, b.const_i32(WN)), b.const_i32(WM)))
    wk = b.to_sgpr_u32(b.div(wave, b.const_i32(WN * WM)))

    bx = b.block_id_x()
    by = b.block_id_y()
    bz = b.block_id_z()
    n_tile0 = b.to_sgpr_u32(b.mul(bx, b.const_i32(TN)))
    m_tile0 = b.to_sgpr_u32(b.mul(by, b.const_i32(TM)))

    lrow = b.mod(lane, b.const_i32(16))  # row within an atom (operand load)
    lgrp = b.div(lane, b.const_i32(16))  # K lane group 0..3

    # ---- per-workgroup operand resources (range = this tile's valid rows) ----
    K2 = b.mul(K, c2)
    c3 = b.const_i32(3)
    cm4 = b.const_i32(-4)

    def _rsrc(ptr: Value, row0: Value, rows_total: Value, tile_rows: int) -> Value:
        base = b.global_ptr_add(ptr, b.mul(b.zext(row0, I64), b.zext(K2, I64)))
        rows_left = b.smin(b.sub(rows_total, row0), b.const_i32(tile_rows))
        nbytes = b.mul(rows_left, K2)
        if spec.kalign == 1:
            # odd K: the last element of the range shares a dword with the 2
            # bytes after it; the buffer range check drops a dword that is
            # not entirely in range, so round the range up to whole dwords
            # (the extra 2 bytes sit in the same aligned dword, hence the
            # same page; they are element K of the last row, which the
            # partial-step mask zeroes)
            nbytes = b.land(b.add(nbytes, c3), cm4)
        return b.buffer_rsrc(base, nbytes)

    a_mn, b_mn = spec.mn_operands
    # (an MN-contiguous operand gets its resource below, once the K slice of
    # this workgroup is known)
    rs_b = _rsrc(Bp, n_tile0, N, TN) if not b_mn else None
    rs_a = _rsrc(A, m_tile0, M, TM) if not a_mn else None
    # ``b_load`` "global" / "global_nt": main-loop B fragments through global
    # loads (row index clamped to the tile's last valid row: a clamped row only
    # feeds outputs that are never stored), optionally with the streaming (nt)
    # hint; the masked partial step keeps the bounds-checked buffer loads
    g_b = spec.b_load != "buffer" and spec.kalign >= 2 and not b_mn
    if g_b:
        gb_base = b.global_ptr_add(Bp, b.mul(b.zext(n_tile0, I64), b.zext(K2, I64)))
        gb_rows = b.smin(b.sub(N, n_tile0), b.const_i32(TN))
        gb_hint = (
            TemporalHint.STREAMING
            if spec.b_load == "global_nt"
            else TemporalHint.DEFAULT
        )

    # lane voffsets (bytes, relative to the tile resources), one per
    # (atom, piece): row * K * 2 + (lgrp * kc + piece * 8) * 2
    # K element of lane group g, piece p inside a step:
    #   kmap "mfma" (default): p * 32 + g * 8 -- each piece is one natural
    #     32-wide MFMA K block, so one load instruction covers 64 contiguous
    #     bytes of each of its 16 rows;
    #   kmap "lane": g * kc + p * 8 -- each lane reads kc contiguous elements
    #     (a permuted K order inside the step, identical for A and B).
    if spec.kmap == "lane":
        kl = b.mul(lgrp, b.const_i32(spec.kc))
        kpiece = [8 * p for p in range(pieces)]
    else:
        kl = b.mul(lgrp, b.const_i32(8))
        kpiece = [32 * p for p in range(pieces)]

    def _voffs(wave_row0: Value, atoms: int) -> List[List[Value]]:
        out = []
        for a_ in range(atoms):
            row = b.add(wave_row0, b.add(b.const_i32(16 * a_), lrow))
            base = b.add(b.mul(row, K2), b.mul(kl, c2))
            out.append([b.add(base, b.const_i32(2 * kpiece[p])) for p in range(pieces)])
        return out

    vo_b = _voffs(b.mul(wn, b.const_i32(16 * na)), na) if not b_mn else None
    vo_a = _voffs(b.mul(wm, b.const_i32(16 * ma)), ma) if not a_mn else None
    if g_b:
        # element index of each (atom, piece) relative to gb_base
        gi_b = []
        for a_ in range(na):
            row = b.add(
                b.mul(wn, b.const_i32(16 * na)), b.add(b.const_i32(16 * a_), lrow)
            )
            row = b.smin(row, b.sub(gb_rows, c1))
            base_e = b.add(b.mul(row, K), kl)
            gi_b.append([b.add(base_e, b.const_i32(kpiece[p])) for p in range(pieces)])

    # ---- K range of this workgroup / wave ----
    n_full = b.div(K, b.const_i32(STEP))  # full steps
    s_lo = b.to_sgpr_u32(b.mul(bz, KPS))
    steps_total = b.div(b.add(K, b.const_i32(STEP - 1)), b.const_i32(STEP))
    s_hi = b.to_sgpr_u32(b.smin(steps_total, b.add(s_lo, KPS)))
    s_main_hi = b.to_sgpr_u32(b.smax(s_lo, b.smin(s_hi, n_full)))
    # wave wk runs steps s_lo + wk + j * WK, j < cnt
    rem = b.sub(s_main_hi, b.add(s_lo, wk))
    cnt = b.to_sgpr_u32(
        b.select(
            b.cmp_gt(rem, c0),
            b.div(b.add(rem, b.const_i32(WK - 1)), b.const_i32(WK)),
            c0,
        )
    )
    last_full = b.smax(b.sub(n_full, c1), c0)

    def _soff(step: Value) -> Value:
        # clamp into the valid full-step range so a prefetch never leaves the
        # operand rows (soffset is not range-checked)
        s = b.smax(b.smin(step, last_full), c0)
        return b.to_sgpr_u32(b.mul(s, b.const_i32(STEP * 2)))

    def _piece(rs: Value, vo: Value, so: Value, align: Optional[int] = None) -> Value:
        """8 consecutive elements of one row at byte offset vo + so (K of a
        K-contiguous operand at ``kalign``, or M / N of an MN-contiguous one
        at ``align`` = ``mnalign``)."""
        if (spec.kalign if align is None else align) >= 2:
            return b.buffer_load_vN(rs, vo, so, elem, 8)
        # kalign 1 (any K): the row may start at a 2-byte boundary. Load the
        # four dwords at the dword-aligned offset plus the 2 bytes after them
        # (one ushort, so nothing past the element range is read) and shift
        # the 18 bytes down by (vo & 3) with v_alignbyte. so is a multiple of
        # 4 (whole K steps), so the alignment is the lane's own.
        va = b.land(vo, cm4)
        sh = b.land(vo, c3)
        ds = [
            b.buffer_load(rs, b.add(va, b.const_i32(4 * i)), so, I32) for i in range(4)
        ]
        hi = b.buffer_load(rs, b.add(vo, b.const_i32(14)), so, elem)
        d4 = b.zext(b.bitcast(hi, I16), I32)
        ds.append(d4)
        words = [b.alignbyte(ds[i + 1], ds[i], sh) for i in range(4)]
        return b.vec_bitcast(b.vec_pack(words, I32), VectorType(elem, 8))

    # ---- MN-contiguous operands: slice resource, lane offsets, LDS staging ----
    mn_op = {}
    if a_mn or b_mn:
        k_lo = b.to_sgpr_u32(b.mul(s_lo, b.const_i32(STEP)))
        slice_rows = b.to_sgpr_u32(b.mul(KPS, b.const_i32(STEP)))
        rows_all = b.sub(K, k_lo)
        capped = b.cmp_gt(rows_all, slice_rows)
        rows_c = b.smax(b.smin(rows_all, slice_rows), c0)
        q16 = b.div(lane, b.const_i32(16))
        j16 = b.mod(lane, b.const_i32(16))

        def _mn_setup(
            ptr: Value, L: Value, mn_tile0: Value, atoms: int, widx: Value, nm: str
        ):
            E = 16 * atoms
            CH = E // 8  # 16-byte chunks per K row of the wave's extent
            rowE = _mn_row_elems(spec, atoms)
            # range: through the last element this K slice may read (K rows
            # past K / the slice and M / N past the operand's end read 0)
            nel = b.sub(b.mul(rows_c, L), b.select(capped, c0, mn_tile0))
            nbytes = b.mul(b.smax(nel, c0), c2)
            if spec.mnalign == 1:
                # odd leading dimension: whole dwords only (see _rsrc)
                nbytes = b.land(b.add(nbytes, c3), cm4)
            base = b.global_ptr_add(
                ptr,
                b.mul(
                    b.add(
                        b.mul(b.zext(k_lo, I64), b.zext(L, I64)), b.zext(mn_tile0, I64)
                    ),
                    b.const_i64(2),
                ),
            )
            rs = b.buffer_rsrc(base, nbytes)
            wmn0 = b.mul(widx, b.const_i32(E))
            lo, krs, wofs = [], [], []
            for i in range(atoms):
                f = b.add(lane, b.const_i32(64 * i))
                kr = b.div(f, b.const_i32(CH))
                ch8 = b.mul(b.mod(f, b.const_i32(CH)), b.const_i32(8))
                lo.append(b.mul(b.add(b.mul(kr, L), b.add(wmn0, ch8)), c2))
                krs.append(kr)
                wofs.append(b.add(b.mul(kr, b.const_i32(rowE)), ch8))
            sm = b.smem_alloc(
                elem,
                [spec.waves, (spec.kc // 8) * 32 * rowE],
                name_hint=nm,
                exclusive=True,
            )
            if spec.mn_lds == "tr":
                # transpose read: lane j of a 16-lane group addresses K row
                # j // 4 (of the 4-row half) and M / N 4 (j % 4) .. + 4
                rd = b.add(
                    b.mul(
                        b.add(b.mul(q16, b.const_i32(8)), b.div(j16, b.const_i32(4))),
                        b.const_i32(rowE),
                    ),
                    b.mul(b.mod(j16, b.const_i32(4)), b.const_i32(4)),
                )
            else:
                # 16-bit reads: K rows 8q .. 8q + 7 of M / N row j
                rd = b.add(b.mul(q16, b.const_i32(8 * rowE)), j16)
            return {
                "rs": rs,
                "L": L,
                "atoms": atoms,
                "rowE": rowE,
                "lo": lo,
                "krs": krs,
                "wofs": wofs,
                "sm": sm,
                "rd": rd,
            }

        if b_mn:
            mn_op["b"] = _mn_setup(Bp, N, n_tile0, na, wn, "skmnb")
        if a_mn:
            mn_op["a"] = _mn_setup(A, M, m_tile0, ma, wm, "skmna")
        s_hi_m1 = b.sub(s_hi, c1)

    def _mn_loads(op: dict, krow0: Value, mask_k: bool = False) -> List[Value]:
        """Raw 8-element M / N vectors of one MN operand for the step whose
        first K row (relative to the slice) is ``krow0``, ordered
        ``[i * pieces + p]`` (load i of piece p); ``mask_k`` zeroes the
        vectors of K rows past K (the masked partial step)."""
        out: List[Optional[Value]] = [None] * (op["atoms"] * pieces)
        for p in range(pieces):
            kp = b.add(krow0, b.const_i32(32 * p))
            sp = b.to_sgpr_u32(b.mul(b.mul(kp, op["L"]), c2))
            for i in range(op["atoms"]):
                v = _piece(op["rs"], b.add(op["lo"][i], sp), c0, align=spec.mnalign)
                if mask_k:
                    krow = b.add(b.add(k_lo, kp), op["krs"][i])
                    v = b.vector_select(
                        b.vector_splat(b.cmp_lt(krow, K), 8), v, b.zero_vec(elem, 8)
                    )
                out[i * pieces + p] = v
        return out

    def _mn_step_row(step: Value) -> Value:
        # the step clamped into this slice (a prefetch past the wave's last
        # step re-reads a step of the slice), relative first K row
        s = b.smax(b.smin(step, s_hi_m1), s_lo)
        return b.mul(b.sub(s, s_lo), b.const_i32(STEP))

    def _mn_frags(op: dict, raw: List[Value]) -> List[Value]:
        """Stage one MN operand's raw vectors in this wave's LDS area and read
        the MFMA fragments back transposed (``[atom * pieces + p]``)."""
        rowE = op["rowE"]
        out: List[Optional[Value]] = [None] * (op["atoms"] * pieces)
        for p in range(pieces):
            pb = p * 32 * rowE
            for i in range(op["atoms"]):
                off = b.add(op["wofs"][i], b.const_i32(pb)) if pb else op["wofs"][i]
                b.smem_store_vN(op["sm"], [wave, off], raw[i * pieces + p], 8)
            for a_ in range(op["atoms"]):
                if spec.mn_lds == "tr":
                    parts = []
                    for r in range(2):
                        imm = pb + 4 * r * rowE + 16 * a_
                        addr = b.add(op["rd"], b.const_i32(imm)) if imm else op["rd"]
                        parts.append(
                            b.ds_read_tr16_b64(op["sm"], wave, addr, dtype=elem)
                        )
                    out[a_ * pieces + p] = b.vec_concat(parts[0], parts[1])
                else:
                    vals = []
                    for e in range(8):
                        imm = pb + e * rowE + 16 * a_
                        addr = b.add(op["rd"], b.const_i32(imm)) if imm else op["rd"]
                        vals.append(
                            b.vec_extract(
                                b.smem_load_vN(op["sm"], wave, addr, dtype=elem, n=1), 0
                            )
                        )
                    out[a_ * pieces + p] = b.vec_pack(vals, elem)
        return out

    def _load_stage(step: Value) -> List[Value]:
        if a_mn or b_mn:
            krow0 = _mn_step_row(step)
        so = _soff(step) if not (a_mn and b_mn) else None
        frs = []
        if b_mn:
            frs += _mn_loads(mn_op["b"], krow0)
        elif g_b:
            ke = b.to_sgpr_u32(b.div(so, c2))  # step offset in elements
            for a_ in range(na):
                for p in range(pieces):
                    frs.append(
                        b.global_load_vN(
                            gb_base,
                            b.add(gi_b[a_][p], ke),
                            elem,
                            8,
                            align=16 if spec.kalign == 8 else 4,
                            temporal_hint=gb_hint,
                        )
                    )
        else:
            for a_ in range(na):
                for p in range(pieces):
                    frs.append(_piece(rs_b, vo_b[a_][p], so))
        if a_mn:
            frs += _mn_loads(mn_op["a"], krow0)
        else:
            for a_ in range(ma):
                for p in range(pieces):
                    frs.append(_piece(rs_a, vo_a[a_][p], so))
        return frs

    def _sub(v: Value, s: int) -> Value:
        if subs == 1:
            return v
        return b.vec_pack([b.vec_extract(v, s * akl + i) for i in range(akl)], elem)

    def _mma(accs: List[Value], frs: List[Value]) -> List[Value]:
        accs = list(accs)
        if a_mn or b_mn:
            fb_, fa_ = list(frs[: na * pieces]), list(frs[na * pieces :])
            if b_mn:
                fb_ = _mn_frags(mn_op["b"], fb_)
            if a_mn:
                fa_ = _mn_frags(mn_op["a"], fa_)
            frs = fb_ + fa_
        for p in range(pieces):
            for s in range(subs):
                for i in range(na):
                    fb = _sub(frs[i * pieces + p], s)
                    for j in range(ma):
                        fa = _sub(frs[(na + j) * pieces + p], s)
                        accs[i * ma + j] = b.mma(mfma_name, fb, fa, accs[i * ma + j])
        return accs

    accs0 = [b.zero_vec(F32, 4) for _ in range(na * ma)]
    first = b.add(s_lo, wk)
    # the wave's last step: prefetches past it re-read it (cache hits) instead
    # of pulling the next slice's bytes from memory
    s_last = b.add(first, b.mul(b.smax(b.sub(cnt, c1), c0), b.const_i32(WK)))

    def _step_of(t: Value) -> Value:
        return b.smin(b.add(first, b.mul(t, b.const_i32(WK))), s_last)

    stages0: List[List[Value]] = [
        _load_stage(_step_of(b.const_i32(d))) for d in range(PF)
    ]
    nfr = (na + ma) * pieces
    # Modulo-expanded register pipeline: one loop trip runs PF steps, step d
    # of the trip consumes stage slot d and refills the same slot with the
    # step PF ahead, so no fragment is copied between slots (a rotating
    # iter-arg pipeline makes LLVM copy the in-flight loads at the back edge
    # and wait for them there).
    iters = [(f"acc{i}", v) for i, v in enumerate(accs0)]
    for d in range(PF):
        iters += [(f"st{d}_{q}", v) for q, v in enumerate(stages0[d])]
    groups = b.to_sgpr_u32(b.div(cnt, b.const_i32(PF)))
    loop = b.scf_for_iter(c0, groups, c1, iters, iv_name="j")
    with loop as (j, vals):
        accs = vals[: na * ma]
        stg = [vals[na * ma + d * nfr : na * ma + (d + 1) * nfr] for d in range(PF)]
        t0 = b.mul(j, b.const_i32(PF))
        new_stg = []
        for d in range(PF):
            accs = _mma(accs, stg[d])
            new_stg.append(_load_stage(_step_of(b.add(t0, b.const_i32(d + PF)))))
        out = list(accs)
        for d in range(PF):
            out += new_stg[d]
        b.scf_yield(*out)
    accs = loop.results[: na * ma]
    rest = loop.results[na * ma :]
    # remainder: cnt % PF steps, held in slots 0 .. PF-2
    remc = b.sub(cnt, b.mul(groups, b.const_i32(PF)))
    for r in range(PF - 1):
        rl = b.scf_for_iter(
            c0,
            b.select(b.cmp_gt(remc, b.const_i32(r)), c1, c0),
            c1,
            [(f"racc{r}_{i}", v) for i, v in enumerate(accs)],
            iv_name=f"rr{r}",
        )
        with rl as (_rr, rv):
            b.scf_yield(*_mma(list(rv), rest[r * nfr : (r + 1) * nfr]))
        accs = rl.results

    # ---- masked partial last K step (only the slice holding it, wave 0) ----
    # the slice that holds step n_full (the partial one) runs it, on wave 0;
    # empty slices past the end (s_lo > n_full) must not
    has_tail = b.land(
        b.land(b.cmp_gt(s_hi, n_full), b.cmp_le(s_lo, n_full)), b.cmp_eq(wk, c0)
    )
    tail_k0 = b.mul(n_full, b.const_i32(STEP))
    iota = [b.const_i32(i) for i in range(8)]

    def _tail_accs_masked(accs_in: List[Value]) -> List[Value]:
        kvoff = b.mul(tail_k0, c2)
        frs = []
        for vo_set, rs, opk in ((vo_b, rs_b, "b"), (vo_a, rs_a, "a")):
            if opk in mn_op:
                frs += _mn_loads(mn_op[opk], b.sub(tail_k0, k_lo), mask_k=True)
                continue
            for a_ in range(len(vo_set)):
                for p in range(pieces):
                    v = _piece(rs, b.add(vo_set[a_][p], kvoff), c0)
                    kbase = b.add(tail_k0, b.add(kl, b.const_i32(kpiece[p])))
                    lim = b.sub(K, kbase)  # elements valid: i < lim
                    mask = b.vector_cmp(
                        "lt",
                        b.vec_pack(iota, I32),
                        b.vector_splat(lim, 8),
                    )
                    frs.append(b.vector_select(mask, v, b.zero_vec(elem, 8)))
        return _mma(accs_in, frs)

    # scf_if has no results: a wave-uniform loop of 0 / 1 trips carries the
    # accumulators through the (cheap, one-step) tail of the last slice.
    tail_trips = b.select(has_tail, c1, c0)
    tl = b.scf_for_iter(
        c0, tail_trips, c1, [(f"tacc{i}", v) for i, v in enumerate(accs)], iv_name="t"
    )
    with tl as (_t, tv):
        b.scf_yield(*_tail_accs_masked(list(tv)))
    accs = tl.results

    # ---- per-thread output fragments: (m_local, n_local, <4 x f32>) ----
    # MFMA(a = B frag, b = A frag): lane holds n = 4 * (lane / 16) + v of atom
    # i, m = lane % 16 of atom j.
    n_lane = b.mul(lgrp, b.const_i32(4))
    frags: List[Tuple[Value, Value, Value]] = []
    if WK == 1:
        for i in range(na):
            for jj in range(ma):
                ml = b.add(
                    b.mul(wm, b.const_i32(16 * ma)), b.add(b.const_i32(16 * jj), lrow)
                )
                nl = b.add(
                    b.mul(wn, b.const_i32(16 * na)), b.add(b.const_i32(16 * i), n_lane)
                )
                frags.append((ml, nl, accs[i * ma + jj]))
    else:
        TNP = TN + _lds_pad(spec)
        red = b.smem_alloc(F32, [WK * TM, TNP], name_hint="skred", exclusive=True)
        for i in range(na):
            for jj in range(ma):
                ml = b.add(
                    b.mul(wm, b.const_i32(16 * ma)), b.add(b.const_i32(16 * jj), lrow)
                )
                nl = b.add(
                    b.mul(wn, b.const_i32(16 * na)), b.add(b.const_i32(16 * i), n_lane)
                )
                row = b.add(b.mul(wk, b.const_i32(TM)), ml)
                b.smem_store_vN_f32(red, [row, nl], accs[i * ma + jj], 4)
        b.sync()
        nf4 = TM * TN // 4
        per = -(-nf4 // spec.block_size)
        q4 = TN // 4
        for r in range(per):
            f = b.add(tid, b.const_i32(r * spec.block_size))
            if nf4 % spec.block_size:
                # fewer float4s than threads in the last round: the extra
                # threads redo the last float4 (identical values to the same
                # addresses), so no store needs a mask
                f = b.smin(f, b.const_i32(nf4 - 1))
            ml = b.div(f, b.const_i32(q4))
            nl = b.mul(b.mod(f, b.const_i32(q4)), b.const_i32(4))
            s = b.smem_load_vN_f32(red, ml, nl, n=4)
            for w_ in range(1, WK):
                s = b.vector_add(
                    s, b.smem_load_vN_f32(red, b.add(b.const_i32(w_ * TM), ml), nl, n=4)
                )
            frags.append((ml, nl, s))

    # ---- output store helpers ----
    # this workgroup's C rows start at a 64-bit offset (M x N may exceed the
    # i32 element range); the in-tile index stays below tile_m x N
    c_tile = b.global_ptr_add(
        C, b.mul(b.mul(b.zext(m_tile0, I64), b.zext(N, I64)), b.const_i64(2))
    )

    def _store_c(ml: Value, nl: Value, v4: Value) -> None:
        m = b.add(m_tile0, ml)
        n = b.add(n_tile0, nl)
        h = b.vec_cast_f32_to(v4, elem)
        in_m = b.cmp_lt(m, M)
        full = b.land(in_m, b.cmp_le(b.add(n, b.const_i32(4)), N))
        idx = b.add(b.mul(ml, N), n)
        with b.scf_if_else(full) as (then_ctx, else_ctx):
            with then_ctx:
                b.global_store_vN(c_tile, idx, h, 4, align=2)
            with else_ctx:
                for e in range(4):
                    ok_e = b.land(in_m, b.cmp_lt(b.add(n, b.const_i32(e)), N))
                    with b.scf_if(ok_e):
                        b.global_store(
                            c_tile,
                            b.add(idx, b.const_i32(e)),
                            b.vec_extract(h, e),
                            align=2,
                        )

    tiles = b.mul(
        b.div(b.add(N, b.const_i32(TN - 1)), b.const_i32(TN)),
        b.div(b.add(M, b.const_i32(TM - 1)), b.const_i32(TM)),
    )
    tile_id = b.to_sgpr_u32(
        b.add(b.mul(by, b.div(b.add(N, b.const_i32(TN - 1)), b.const_i32(TN))), bx)
    )
    TE = TM * TN

    def _ws_idx(z: Value, ml: Value, nl: Value) -> Value:
        return b.add(
            b.mul(b.add(b.mul(z, tiles), tile_id), b.const_i32(TE)),
            b.add(b.mul(ml, b.const_i32(TN)), nl),
        )

    _wsh = TemporalHint.STREAMING if spec.nt_ws else TemporalHint.DEFAULT
    is_split = b.cmp_gt(SPLIT, c1)
    flag = b.smem_alloc(I32, [4], name_hint="sktk", exclusive=True)
    with b.scf_if_else(is_split) as (then_ctx, else_ctx):
        with else_ctx:
            for ml, nl, v in frags:
                _store_c(ml, nl, v)
        with then_ctx:
            sc1 = spec.red_sync == "sc1"
            w64 = b.global_ptr_to_i64(W) if sc1 else None

            def _w_addr(idx: Value) -> Value:
                return b.add(w64, b.mul(b.zext(idx, I64), b.const_i64(4)))

            if sc1:
                # agent-scope (sc1) stores: written through past this XCD's L2,
                # so no whole-L2 write-back fence is needed. LLVM cannot see
                # into an asm store (no store-data hazard handling, no
                # vmcnt), so each asm group ends with its own vmcnt(0): once
                # it returns, the data registers are free and the partials
                # are complete before the ticket atomic.
                for g0 in range(0, len(frags), _ASM_GROUP):
                    grp = frags[g0 : g0 + _ASM_GROUP]
                    tmpl = []
                    ops = []
                    for q, (ml, nl, v) in enumerate(grp):
                        tmpl.append(
                            f"global_store_dwordx4 ${2 * q}, ${2 * q + 1}, off sc1"
                        )
                        ops += [_w_addr(_ws_idx(bz, ml, nl)), v]
                    tmpl.append("s_waitcnt vmcnt(0)")
                    b.inline_asm("\n".join(tmpl), ",".join(["v"] * len(ops)), ops)
            else:
                for ml, nl, v in frags:
                    b.global_store_vN(
                        W, _ws_idx(bz, ml, nl), v, 4, align=16, temporal_hint=_wsh
                    )
                b.fence(scope="agent", ordering="release")
            b.sync()
            with b.scf_if(b.cmp_eq(tid, c0)):
                old = b.global_atomic_add(
                    Cnt, tile_id, c1, ordering="monotonic" if sc1 else "acq_rel"
                )
                b.smem_store_vN(flag, [c0], old, 1)
            b.sync()
            ticket = b.smem_load_vN(flag, c0, dtype=I32, n=1)
            ticket = b.to_sgpr_u32(b.vec_extract(ticket, 0))
            last = b.cmp_eq(ticket, b.sub(SPLIT, c1))
            with b.scf_if(last):
                if not sc1:
                    b.fence(scope="agent", ordering="acquire")
                # fixed-order sum over z = 0 .. SPLIT-1, RU partials per trip
                # loaded together (independent loads in flight); z >= SPLIT
                # loads a clamped slot and adds +0.0
                RU = spec.red_unroll
                trips = b.div(b.add(SPLIT, b.const_i32(RU - 1)), b.const_i32(RU))
                zl = b.scf_for_iter(
                    c0,
                    trips,
                    c1,
                    [(f"ws{fi}", b.zero_vec(F32, 4)) for fi in range(len(frags))],
                    iv_name="zt",
                )
                with zl as (zt, wv):
                    z0 = b.mul(zt, b.const_i32(RU))
                    oks = []
                    addrs = []
                    for u in range(RU):
                        z = b.add(z0, b.const_i32(u))
                        zc = b.smin(z, b.sub(SPLIT, c1))
                        oks.append(b.vector_splat(b.cmp_lt(z, SPLIT), 4))
                        addrs.append([_ws_idx(zc, ml, nl) for ml, nl, _v in frags])
                    raw = [[None] * len(frags) for _ in range(RU)]
                    if sc1:
                        # agent-scope (sc1) loads: they miss any stale line a
                        # previous launch left in this XCD's L2. LLVM cannot
                        # see an asm load is in flight, so each asm group
                        # issues its loads into early-clobber outputs and
                        # ends with vmcnt(0): the outputs are complete when
                        # the asm returns
                        flat = [(u, fi) for u in range(RU) for fi in range(len(frags))]
                        for g0 in range(0, len(flat), _ASM_GROUP):
                            grp = flat[g0 : g0 + _ASM_GROUP]
                            n_ = len(grp)
                            tmpl = [
                                f"global_load_dwordx4 ${q}, ${n_ + q}, off sc1"
                                for q in range(n_)
                            ]
                            tmpl.append("s_waitcnt vmcnt(0)")
                            res_ = b.inline_asm_multi(
                                "\n".join(tmpl),
                                ",".join(["=&v"] * n_ + ["v"] * n_),
                                [_w_addr(addrs[u][fi]) for u, fi in grp],
                                result_types=[VectorType(F32, 4)] * n_,
                            )
                            for (u, fi), r_ in zip(grp, res_):
                                raw[u][fi] = r_
                    else:
                        for u in range(RU):
                            for fi in range(len(frags)):
                                raw[u][fi] = b.global_load_vN(
                                    W,
                                    addrs[u][fi],
                                    F32,
                                    4,
                                    align=16,
                                    temporal_hint=_wsh,
                                )
                    outv = []
                    for fi in range(len(frags)):
                        acc_ = wv[fi]
                        for u in range(RU):
                            acc_ = b.vector_add(
                                acc_,
                                b.vector_select(oks[u], raw[u][fi], b.zero_vec(F32, 4)),
                            )
                        outv.append(acc_)
                    b.scf_yield(*outv)
                for fi, (ml, nl, _v) in enumerate(frags):
                    _store_c(ml, nl, zl.results[fi])
                with b.scf_if(b.cmp_eq(tid, c0)):
                    b.global_store(Cnt, tile_id, c0, align=4)
    b.ret()
    return b.kernel

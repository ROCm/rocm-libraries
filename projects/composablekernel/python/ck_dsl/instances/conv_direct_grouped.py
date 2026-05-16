# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Direct grouped convolution kernel — streaming row-by-row pipeline.

A DSL-native direct grouped-convolution kernel: each output row is
computed by streaming the input row through MFMAs without ever
materialising an im2col or implicit-GEMM tile. The 16-channel
(`cpg=kpg=16`) and 4-channel (`cpg=kpg=4`) variants share the
authoring surface; they differ only in `BLOCK_GROUPS` and the choice
of MFMA atom.

Correctness note: the earlier apparent correctness drift was a host
reference bug. The shared launcher compared grouped convolution output
against a dense convolution reference. `ck_dsl.run_manifest` now verifies
with a grouped NumPy fp32-accum reference, and both 16c and 4c paths pass
with `bad=0` at the bake-off tolerance.

Why direct conv (vs implicit GEMM) for small channels:
  - For `C=K=4` or `C=K=16` 3x3 group conv, the implicit-GEMM packs
    the work as a `M = N*Ho*Wo, N_gemm = K, K_gemm = R*S*C = {36, 144}`
    GEMM. `N_gemm` is far below the natural 16x16 MFMA tile shape;
    most of the MFMA's M dimension is wasted. The shape is also
    extremely elongated and spatially structured.
  - Direct conv keeps the spatial structure and the small channels
    aligned to MFMA naturally: per wave, process one group, with
    `M = K_filter = cpg`, `N = BLOCK_Q`, `K = cpg`.

Kernel structure (16c variant):
  - 8 waves per workgroup (`BLOCK_GROUPS = 8`), each handling one
    group. `BLOCK_GROUPS * WAVE = 512` threads per block.
  - `BLOCK_Q = 16` output W positions per block; the kernel iterates
    H output rows in series.
  - LDS double-buffered: at row `y`, wave reads from `lds_a` while
    threads prefetch row `y+1` into `lds_b` (and ping-pong).
  - 3-accumulator circular pipeline along H: accumulator slot
    `(y - r) % 3` holds the contribution from output row
    `y - r`, with `r ∈ {0, 1, 2}` for a 3x3 conv. After 3
    rows fill, the oldest slot is *flushed* to D and reset to zero.

Coordinate-transform DAG (described as CK Tile transforms — kept here
as documentation, not as a runtime object, because direct conv's
addressing is structurally per-row rather than per-(M, K) point):

    A_nhwc (input):
      naive: (n, h, w, c)
      pad(h, lo=0, hi=H), pad(w, lo=0, hi=W)        boundary
      embed(("y", "r") -> "h", strides=(1, 1),       row-row
            offset=-pad, lo=0, hi=H)
      embed(("q", "s") -> "w", strides=(1, 1),       col-col
            offset=-pad, lo=0, hi=W)
      unmerge(c -> (group, ch_block, channel),       chan unpack
              dims=(groups, cpg/load_vec, load_vec))

    B_krsc (weight):
      naive: (k_out, r, s, c)
      unmerge(k_out -> (group, k_in_group), dims=(groups, kpg))

    D_nhwk (output):
      naive: (n, h, w, k_out)
      unmerge(k_out -> (group, k_in_group), dims=(groups, kpg))

The 16c kernel uses `mfma_f32_16x16x16_f16` once per (R, S). The 4c
kernel uses `mfma_f32_4x4x4_f16` which emits 16 independent 4x4x4
matmuls per wave — letting one wave process 16 groups simultaneously
(perfect fit for cpg=4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..core.ir import (
    F16,
    I32,
    IRBuilder,
    KernelDef,
    PtrType,
    Value,
)
from ..transforms import TensorDescriptor


@dataclass(frozen=True)
class DirectConvProblem:
    """The grouped direct-conv shape parameters.

    Layouts:
      A: NHWC fp16, `[N, H, W, groups*cpg]`
      B: KRSC fp16, `[groups*kpg, KH, KW, cpg]`
      D: NHWK fp16, `[N, H, W, groups*kpg]`
    """

    N: int
    H: int
    W: int
    groups: int
    cpg: int  # channels per group
    kpg: int  # filters per group (= cpg in the bake-off)
    KH: int = 3
    KW: int = 3
    PAD: int = 1
    stride: int = 1

    @property
    def total_c(self) -> int:
        return self.groups * self.cpg

    @property
    def total_k(self) -> int:
        return self.groups * self.kpg

    @property
    def flops(self) -> int:
        return (
            2
            * self.N
            * self.H
            * self.W
            * self.groups
            * self.kpg
            * self.KH
            * self.KW
            * self.cpg
        )

    def short(self) -> str:
        return f"N{self.N}H{self.H}W{self.W}_g{self.groups}_c{self.cpg}k{self.kpg}"


@dataclass(frozen=True)
class DirectConv16cSpec:
    """Direct grouped convolution kernel for `cpg = kpg = 16`.

    Block geometry:
      - `BLOCK_Q = 16` output W positions per block (one MFMA's N tile).
      - `BLOCK_GROUPS = 8` groups per workgroup.
      - `WAVE = 64` threads per wave, `BLOCK_GROUPS * WAVE = 512`
        threads per block.
      - Each wave owns one group.

    MFMA atom: `mfma_f32_16x16x16_f16` with per-warp tile
      M = K_filter = kpg = 16,
      N = BLOCK_Q       = 16,
      K = cpg           = 16
    so the inner loop is exactly 9 MFMAs (R*S) per output row.

    Pipeline knobs:
      - `double_buffer`: ping-pong two LDS regions; prefetch input row
        y+1 while computing on row y.
      - `accumulator_pipeline_depth`: number of circular accumulators
        (KH for a 3x3 conv).
    """

    problem: DirectConvProblem
    name: str = "direct_conv_16c"
    block_q: int = 16
    block_groups: int = 8
    wave_size: int = 64
    double_buffer: bool = True
    fold_k32: bool = True

    @property
    def threads_per_block(self) -> int:
        return self.block_groups * self.wave_size

    @property
    def n_acc_slots(self) -> int:
        return self.problem.KH

    def kernel_name(self) -> str:
        from ..helpers.spec import kernel_name_join

        p = self.problem
        return kernel_name_join(
            self.name,
            p.short(),
            f"bq{self.block_q}",
            f"bg{self.block_groups}",
            "db" if self.double_buffer else "sb",
            flags={"k32": self.fold_k32},
        )

    def validate(self) -> None:
        p = self.problem
        if p.cpg != 16 or p.kpg != 16:
            raise ValueError(
                f"DirectConv16cSpec expects cpg=kpg=16 (got {p.cpg}, {p.kpg})"
            )
        if p.groups % self.block_groups != 0:
            raise ValueError(
                f"groups {p.groups} not divisible by block_groups {self.block_groups}"
            )


def build_direct_conv_16c(spec: DirectConv16cSpec) -> KernelDef:
    """Build the IR for one direct conv 16c kernel instance.

    See the module docstring for the kernel structure. The Python
    builder unrolls every Python `for` loop at IR-build time; the
    only runtime loop is the H-row streaming `scf.for`.
    """
    spec.validate()
    p = spec.problem
    BLOCK_Q = spec.block_q
    BLOCK_GROUPS = spec.block_groups
    WAVE = spec.wave_size
    THREADS = spec.threads_per_block
    LDS_W = BLOCK_Q + p.KW - 1
    LDS_ROW_FP16 = LDS_W * BLOCK_GROUPS * p.cpg
    LOAD_VEC = 4
    NUM_VEC4 = LDS_ROW_FP16 // LOAD_VEC

    if NUM_VEC4 == 0:
        raise ValueError("LDS row too small for one vec4 per thread")
    PASSES = (NUM_VEC4 + THREADS - 1) // THREADS

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = THREADS

    A = b.param("A", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    Bp = b.param("B", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    D = b.param("D", PtrType(F16, "global"), noalias=True, writeonly=True, align=16)
    A_bytes = b.param("A_bytes", I32)
    B_bytes = b.param("B_bytes", I32)
    D_bytes = b.param("D_bytes", I32)

    c0 = b.const_i32(0)
    c_wave = b.const_i32(WAVE)
    c_BG = b.const_i32(BLOCK_GROUPS)
    c_BQ = b.const_i32(BLOCK_Q)
    c_cpg = b.const_i32(p.cpg)
    c_kpg = b.const_i32(p.kpg)
    c_PAD = b.const_i32(p.PAD)
    c_W = b.const_i32(p.W)
    c_H = b.const_i32(p.H)

    # The address constants previously hand-rolled here
    # (``c_W_totalC``, ``c_H_W_totalC``, …) are now folded into the
    # per-axis ``TensorDescriptor`` lookups below. Keep the LDS
    # geometry constant (``c_BG_cpg``) because that one names a
    # workgroup-shaped LDS stride and isn't part of any DRAM
    # descriptor.
    c_BG_cpg = b.const_i32(BLOCK_GROUPS * p.cpg)

    tid = b.thread_id_x()
    wave_id = b.div(tid, c_wave)
    lane = b.mod(tid, c_wave)
    c4 = b.div(lane, b.const_i32(16))  # 0..3
    q_in_lane = b.mod(lane, b.const_i32(16))  # 0..15
    # K=32 folded direct-conv mapping:
    #   c4=0,1 -> S=0 with channel blocks 0..7 and 8..15
    #   c4=2,3 -> S=1 with channel blocks 0..7 and 8..15
    # S=2 remains a residual K=16 MFMA using the original c4*4 mapping.
    s_lane_k32 = b.div(c4, b.const_i32(2))
    ch_lane_k32 = b.mul(b.mod(c4, b.const_i32(2)), b.const_i32(8))
    ch_lane_k16 = b.mul(c4, b.const_i32(4))

    # Grid layout:
    #   bx = Q-tile index (0..ceil(W/BQ)-1)
    #   by = group-tile index (0..groups/BG - 1)
    #   bz = batch index n
    bx = b.block_id_x()
    by = b.block_id_y()
    n = b.block_id_z()

    g_tile = by
    g = b.add(b.mul(g_tile, c_BG), wave_id)  # absolute group for this wave
    q_tile_start = b.mul(bx, c_BQ)

    # LDS: two ping-pong rows for the input. Use 2D shape `[1, ROW]`
    # to keep the smem_load/store_vN_f16 ABIs happy (they always emit
    # a 2D GEP — `[i32 0, i32 row, i32 col]`).
    #
    # IMPORTANT: the LDS is sized to fit *every* chunk a thread might
    # ever address, not just the in-bounds chunks. With THREADS=512
    # and NUM_VEC4=576 we have PASSES=2 passes; the second pass has
    # 448 threads whose `chunk_idx >= NUM_VEC4` and would write past
    # the end of a `[ROW]`-sized allocation. Even though those threads
    # write zeros (after the validity mask), an LDS store past the
    # allocation is undefined behaviour and gets either dropped or
    # miscompiled. We over-allocate to `PASSES * THREADS * LOAD_VEC`
    # halves so the OOB-zeroed writes land in the slack region of the
    # allocation and never alias a valid chunk.
    lds_total_fp16 = PASSES * THREADS * LOAD_VEC
    A_smem = b.smem_alloc(F16, [1, lds_total_fp16], name_hint="lds_a")
    B_smem = (
        b.smem_alloc(F16, [1, lds_total_fp16], name_hint="lds_b")
        if spec.double_buffer
        else A_smem
    )

    # Buffer rsrcs.
    a_rsrc = b.buffer_rsrc(A, A_bytes)
    b_rsrc = b.buffer_rsrc(Bp, B_bytes)
    d_rsrc = b.buffer_rsrc(D, D_bytes)

    c_half_bytes = b.const_i32(2)
    oob_sentinel = b.const_i32((1 << 31) - 1)
    fp16x4_zero = b.zero_vec_f16(4)
    zero_acc = b.zero_vec_f32(4)

    # ---- weight loads (constant across H-loop) ----
    # Build a `TensorDescriptor` for B[K_OUT, KH, KW, CPG] -- the
    # weight layout. Lower coords (k_out, r, s, c) compose into the
    # naive linear offset
    #   k_out * KH * KW * cpg + r * KW * cpg + s * cpg + c
    # which is exactly what the hand-rolled math computed below. Using
    # the transform DAG instead of stringing together ``add``/``mul``
    # SSA ops keeps the addressing in one place and makes future
    # fusion / boundary-check additions easier.
    b_desc = TensorDescriptor.naive(
        "B",
        lengths=[p.total_k, p.KH, p.KW, p.cpg],
        coord_names=("k_out", "r", "s", "c"),
    )
    k_out_val = b.add(b.mul(g, c_kpg), q_in_lane)
    weights: List[Value] = []
    weights_k32: List[Value] = []
    weights_k16: List[Value] = []
    if spec.fold_k32:
        for r_const in range(p.KH):
            r_i = b.const_i32(r_const)
            # Fold S=0 and S=1 into one K=32 MFMA. Each lane reads
            # <8 x half> at s_lane_k32*cpg + ch_lane_k32.
            w_off_k32, _ = b_desc.offset(
                b,
                k_out=k_out_val,
                r=r_i,
                s=s_lane_k32,
                c=ch_lane_k32,
            )
            weights_k32.append(
                b.buffer_load_vN_f16(b_rsrc, b.mul(w_off_k32, c_half_bytes), c0, 4)
            )
            # Residual S=2 uses K=16, 4 channels per lane.
            w_off_k16, _ = b_desc.offset(
                b,
                k_out=k_out_val,
                r=r_i,
                s=b.const_i32(2),
                c=ch_lane_k16,
            )
            weights_k16.append(
                b.buffer_load_vN_f16(b_rsrc, b.mul(w_off_k16, c_half_bytes), c0, 2)
            )
    else:
        for r_const in range(p.KH):
            for s_const in range(p.KW):
                r_i = b.const_i32(r_const)
                s_i = b.const_i32(s_const)
                w_off, _ = b_desc.offset(
                    b,
                    k_out=k_out_val,
                    r=r_i,
                    s=s_i,
                    c=ch_lane_k16,
                )
                weights.append(
                    b.buffer_load_vN_f16(b_rsrc, b.mul(w_off, c_half_bytes), c0, 2)
                )

    # ---- LDS load helper ----
    # Each thread loads a vec4 of `cpg=16` halves of input from DRAM
    # at (n, hi, wi, c) -> LDS at index `chunk_idx * 4`.
    # The per-thread chunk decomposition is:
    #   chunk_idx = tid + pass_idx * THREADS
    #   ch_block  = chunk_idx % 4
    #   gw_idx    = chunk_idx // 4
    #   group_in_wg = gw_idx % BLOCK_GROUPS
    #   W_lds       = gw_idx // BLOCK_GROUPS
    chunk_meta = []
    for pass_idx in range(PASSES):
        chunk_idx = b.add(tid, b.const_i32(pass_idx * THREADS))
        ch_block = b.mod(chunk_idx, b.const_i32(4))
        gw_idx = b.div(chunk_idx, b.const_i32(4))
        group_in_wg = b.mod(gw_idx, c_BG)
        W_lds = b.div(gw_idx, c_BG)
        W_in = b.sub(b.add(q_tile_start, W_lds), c_PAD)
        W_ok = b.land(b.cmp_ge(W_in, c0), b.cmp_lt(W_in, c_W))
        in_bounds = b.cmp_lt(chunk_idx, b.const_i32(NUM_VEC4))
        abs_group = b.add(b.mul(g_tile, c_BG), group_in_wg)
        chunk_meta.append(
            {
                "chunk_idx": chunk_idx,
                "ch_block": ch_block,
                "group_in_wg": group_in_wg,
                "W_lds": W_lds,
                "W_in": W_in,
                "W_ok": W_ok,
                "in_bounds": in_bounds,
                "abs_group": abs_group,
            }
        )

    # Input descriptor: A[N, H, W, total_c] in NHWC. The channel axis
    # is sub-divided into (group, ch_block, 4) at the per-thread load
    # site via an :class:`Unmerge` over the existing ``c`` coord, but
    # we compute the leaf offset directly on the (group * cpg + 4*block)
    # composite to avoid an extra coord transform inside the hot loop.
    # The descriptor still encodes the (N, H, W, C) layout cleanly so
    # future migrations can add a Pad/Embed for boundary handling.
    a_desc = TensorDescriptor.naive(
        "A",
        lengths=[p.N, p.H, p.W, p.total_c],
        coord_names=("n", "h", "w", "c"),
    )

    def issue_dram_load(hi_in: Value, hi_ok: Value):
        """Per-thread DRAM read of one vec4 of A.

        Returns `(vec4, lds_idx)` pairs; the caller decides when to
        store them to LDS. This is important for the v6 pipeline:
        issue DRAM reads for row y+1 before the MFMAs on row y, then
        write those prefetched registers to the next LDS buffer after
        the MFMAs. That preserves the read-before-write ordering on
        the current buffer while overlapping the VMEM latency with
        compute.
        """
        out = []
        for cm in chunk_meta:
            valid = b.land(hi_ok, b.land(cm["W_ok"], cm["in_bounds"]))
            c_val = b.add(
                b.mul(cm["abs_group"], c_cpg),
                b.mul(cm["ch_block"], b.const_i32(4)),
            )
            a_off_elems, _ = a_desc.offset(
                b,
                n=n,
                h=hi_in,
                w=cm["W_in"],
                c=c_val,
            )
            a_off_bytes = b.mul(a_off_elems, c_half_bytes)
            safe_off = b.select(valid, a_off_bytes, oob_sentinel)
            a_vec = b.buffer_load_vN_f16(a_rsrc, safe_off, c0, 2)
            a_vec = b.select(valid, a_vec, fp16x4_zero)
            # LDS index in halves: chunk_idx * 4. Allocation is 2D
            # `[1, ROW]` so we pass (row=0, col=lds_idx).
            lds_idx = b.mul(cm["chunk_idx"], b.const_i32(4))
            out.append((a_vec, lds_idx))
        return out

    def store_to_lds(loads, lds: Value) -> None:
        for a_vec, lds_idx in loads:
            b.smem_store_vN_f16(lds, [c0, lds_idx], a_vec, 4)

    q_subtiles = BLOCK_Q // 16

    def lds_read_input(q_subtile: int, s_const: int, lds: Value) -> Value:
        """Per-lane <4 x half> read from LDS for the s-th column of the
        3-wide input row. Lane c4=0..3 picks 4 channels of one group
        (each group has cpg=16 channels, split into 4 c4-blocks of 4).

        The LDS is laid out (W_lds, group_in_wg, channel) row-major,
        so the read index is:
            W_lds_idx * (BG * cpg) + wave_id * cpg + c4 * 4
        with W_lds_idx = q_in_lane + s. Note the stride per W_lds is
        `BG * cpg`, NOT `LDS_W * BG * cpg` — the LDS row is laid out
        flat across (W, G, C) so the stride is just `G * C` halves.
        """
        W_lds_idx = b.add(q_in_lane, b.const_i32(q_subtile * 16 + s_const))
        lds_idx = b.add(
            b.add(
                b.mul(W_lds_idx, c_BG_cpg),
                b.mul(wave_id, c_cpg),
            ),
            b.mul(c4, b.const_i32(4)),
        )
        return b.smem_load_vN_f16(lds, c0, lds_idx, n=4)

    def lds_read_input_k32(q_subtile: int, lds: Value) -> Value:
        """Per-lane <8 x half> read for the folded K=32 MFMA.

        The lane's `c4` selects S=0/1 and channel block 0/8:
            W_lds = q_in_lane + (c4 // 2)
            channel = (c4 % 2) * 8
        """
        W_lds_idx = b.add(b.add(q_in_lane, b.const_i32(q_subtile * 16)), s_lane_k32)
        lds_idx = b.add(
            b.add(
                b.mul(W_lds_idx, c_BG_cpg),
                b.mul(wave_id, c_cpg),
            ),
            ch_lane_k32,
        )
        return b.smem_load_vN_f16(lds, c0, lds_idx, n=8)

    # ---- prologue: prefetch row 0 (= -PAD..-PAD+1 = -1) into A_smem ----
    # The first iter's input row is hi = 0 - PAD = -1 for PAD=1, which
    # is invalid (above the image). The descriptor masks it to zero,
    # so the prologue effectively zero-fills A_smem for iter 0.
    hi0 = b.sub(c0, c_PAD)
    hi0_ok = b.land(b.cmp_ge(hi0, c0), b.cmp_lt(hi0, c_H))
    store_to_lds(issue_dram_load(hi0, hi0_ok), A_smem)
    b.sync()

    # ---- the H-row streaming loop ----
    # We use a constant Python loop for all H+KH-1 row positions.
    # This unrolls the entire kernel body, which gives the AMDGPU
    # backend the most freedom to schedule. With H=200, KH=3 this
    # is 202 iterations; the per-iter body is small (10-15 ds
    # reads, 9 MFMAs, optional epilogue store) so the unrolled IR
    # stays small.
    n_iters = p.H + p.KH - 1
    acc_tiles: List[List[Value]] = [
        [zero_acc, zero_acc, zero_acc] for _ in range(q_subtiles)
    ]

    for y in range(n_iters):
        cur = A_smem if (y % 2 == 0 or not spec.double_buffer) else B_smem
        nxt = B_smem if (y % 2 == 0 or not spec.double_buffer) else A_smem

        # Read inputs from the current buffer first; no writes to
        # `cur` are issued until the next time it becomes `nxt`.
        if spec.fold_k32:
            inputs_by_q = [
                (lds_read_input_k32(qt, cur), lds_read_input(qt, 2, cur))
                for qt in range(q_subtiles)
            ]
        else:
            inputs_by_q = [
                [lds_read_input(qt, s, cur) for s in range(p.KW)]
                for qt in range(q_subtiles)
            ]

        # Issue DRAM reads for the next row into registers before
        # the MFMAs. Store those registers to the next LDS buffer
        # after MFMAs to overlap the next-row load with this-row compute.
        loads_next = None
        if y + 1 < n_iters:
            hi_next = b.sub(b.const_i32(y + 1), c_PAD)
            hi_next_ok = b.land(b.cmp_ge(hi_next, c0), b.cmp_lt(hi_next, c_H))
            loads_next = issue_dram_load(hi_next, hi_next_ok)

        for qt in range(q_subtiles):
            accs = acc_tiles[qt]
            for r_const in range(p.KH):
                p_idx = (y - r_const) % p.KH
                acc_in = accs[p_idx]
                if spec.fold_k32:
                    input_k32, input_k16 = inputs_by_q[qt]
                    acc_in = b.mfma_f32_16x16x32_f16(
                        weights_k32[r_const], input_k32, acc_in
                    )
                    acc_in = b.mfma_f32_16x16x16_f16(
                        weights_k16[r_const], input_k16, acc_in
                    )
                else:
                    inputs = inputs_by_q[qt]
                    for s_const in range(p.KW):
                        w_idx = r_const * p.KW + s_const
                        acc_in = b.mfma_f32_16x16x16_f16(
                            weights[w_idx], inputs[s_const], acc_in
                        )
                accs[p_idx] = acc_in

        if loads_next is not None:
            store_to_lds(loads_next, nxt)
        b.sync()

        # Flush output for row p_flush = y - (KH-1) when in range,
        # then ALWAYS reset accs[P_FLUSH = p_flush_val % KH] to zero.
        #
        # The unconditional reset (NOT inside the `if`) is the key
        # correctness fix. Without it, iters y=0..KH-2 (whose
        # p_flush_val is negative) leak their r=KH-1 contributions
        # into accs[(-y-1)%KH], which the next flush of that slot
        # (for a valid output row) accidentally includes.
        # Concretely: y=1, r=2 leaks `weight[r=2] * input[hi=0]`
        # into accs[2]; later acc[2] is flushed for ho=2 with three
        # correct contributions, *plus* the leak, producing a wrong
        # answer. The unconditional `accs[P_FLUSH] = zero_acc` reset
        # ensures every flushed slot starts from a clean accumulator.
        p_flush_val = y - (p.KH - 1)
        P_FLUSH = p_flush_val % p.KH
        if 0 <= p_flush_val < p.H:
            # Output descriptor: D[N, H, W, total_k] in NHWK. Computing
            # the flush address through the descriptor keeps the
            # multi-axis stride math centralised.
            d_desc = TensorDescriptor.naive(
                "D",
                lengths=[p.N, p.H, p.W, p.total_k],
                coord_names=("n", "h", "w", "k"),
            )
            for qt in range(q_subtiles):
                acc_to_flush = acc_tiles[qt][P_FLUSH]
                out_q = b.add(b.add(q_tile_start, b.const_i32(qt * 16)), q_in_lane)
                out_q_valid = b.cmp_lt(out_q, c_W)
                k_val = b.add(b.mul(g, c_kpg), b.mul(c4, b.const_i32(4)))
                d_base, _ = d_desc.offset(
                    b,
                    n=n,
                    h=b.const_i32(p_flush_val),
                    w=out_q,
                    k=k_val,
                )
                d_base_bytes = b.mul(d_base, c_half_bytes)
                safe_d_off = b.select(out_q_valid, d_base_bytes, oob_sentinel)
                # The 4 per-lane output elements are contiguous in NHWK:
                # k_out = g*kpg + c4*4 + [0..3].  Store them as one
                # 64-bit vector instead of four scalar buffer_store_short
                # ops. This is the direct-epilogue analogue of the
                # cshuffle wide-store lever.
                acc_h = b.vec_trunc_f32_to_f16(acc_to_flush)
                b.buffer_store_vN_f16(d_rsrc, safe_d_off, c0, acc_h, 2)
        # Unconditional slot reset — also kills early-iter leaks
        # before they pollute a later output row.
        for qt in range(q_subtiles):
            acc_tiles[qt][P_FLUSH] = zero_acc

    return b.kernel


@dataclass(frozen=True)
class DirectConv4cSpec:
    """Direct grouped convolution kernel for `cpg = kpg = 4`.

    Uses `mfma_f32_4x4x4_f16`, whose wave64 form computes 16 independent
    4x4x4 matmuls per wave. We map those 16 independent batches to 16
    convolution groups, so a single wave processes 16 groups at once.
    """

    problem: DirectConvProblem
    name: str = "direct_conv_4c"
    block_q: int = 4
    block_groups: int = 16
    wave_size: int = 64

    @property
    def threads_per_block(self) -> int:
        return (self.block_groups // 16) * self.wave_size

    def kernel_name(self) -> str:
        from ..helpers.spec import kernel_name_join

        p = self.problem
        return kernel_name_join(
            self.name, p.short(), f"bq{self.block_q}", f"bg{self.block_groups}"
        )

    def validate(self) -> None:
        p = self.problem
        if p.cpg != 4 or p.kpg != 4:
            raise ValueError(
                f"DirectConv4cSpec expects cpg=kpg=4 (got {p.cpg}, {p.kpg})"
            )
        if self.block_groups % 16 != 0:
            raise ValueError("DirectConv4cSpec block_groups must be a multiple of 16")
        if self.block_q % 4 != 0:
            raise ValueError("DirectConv4cSpec block_q must be a multiple of 4")
        if p.groups % self.block_groups != 0:
            raise ValueError(
                f"groups {p.groups} not divisible by block_groups {self.block_groups}"
            )


def build_direct_conv_4c(spec: DirectConv4cSpec) -> KernelDef:
    """Build the direct grouped 4c kernel using MFMA 4x4x4.

    Each lane has:
      - batch = lane / 4 -> group within the workgroup (0..15)
      - lane_q = lane % 4 -> output W position and output channel row

    The MFMA output vector `<4 x f32>` maps to output channels
    `k_in_group = 0..3` at fixed output W position `lane_q`.
    """
    spec.validate()
    p = spec.problem
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.threads_per_block

    A = b.param("A", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    Bp = b.param("B", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    D = b.param("D", PtrType(F16, "global"), noalias=True, writeonly=True, align=16)
    A_bytes = b.param("A_bytes", I32)
    B_bytes = b.param("B_bytes", I32)
    D_bytes = b.param("D_bytes", I32)

    c0 = b.const_i32(0)
    c_W = b.const_i32(p.W)
    c_H = b.const_i32(p.H)
    c_PAD = b.const_i32(p.PAD)
    c_cpg = b.const_i32(p.cpg)
    c_kpg = b.const_i32(p.kpg)
    # Same addressing convention as the 16c kernel: the per-axis
    # strides are encoded in the input/weight/output ``TensorDescriptor``s
    # below, so the per-iteration body no longer carries pre-multiplied
    # i32 constants like ``c_W_totalC``.
    c_half_bytes = b.const_i32(2)
    oob_sentinel = b.const_i32((1 << 31) - 1)

    tid = b.thread_id_x()
    wave_id = b.div(tid, b.const_i32(spec.wave_size))
    lane = b.mod(tid, b.const_i32(spec.wave_size))
    batch = b.div(lane, b.const_i32(4))
    lane_q = b.mod(lane, b.const_i32(4))

    bx = b.block_id_x()
    by = b.block_id_y()
    n = b.block_id_z()
    q_tile_start = b.mul(bx, b.const_i32(spec.block_q))
    group_in_wg = b.add(b.mul(wave_id, b.const_i32(16)), batch)
    g = b.add(b.mul(by, b.const_i32(spec.block_groups)), group_in_wg)

    a_rsrc = b.buffer_rsrc(A, A_bytes)
    b_rsrc = b.buffer_rsrc(Bp, B_bytes)
    d_rsrc = b.buffer_rsrc(D, D_bytes)
    fp16x4_zero = b.zero_vec_f16(4)
    zero_acc = b.zero_vec_f32(4)

    # Weights: per (r, s), per lane: B[g*kpg + lane_q, r, s, 0:4].
    # Same descriptor algebra as the 16c kernel but with the kpg=4
    # layout; the leading channel coord is fixed at 0 because the
    # 4c kernel's MFMA 4x4x4 atom processes all 4 channels of one
    # group per lane.
    b_desc = TensorDescriptor.naive(
        "B",
        lengths=[p.total_k, p.KH, p.KW, p.cpg],
        coord_names=("k_out", "r", "s", "c"),
    )
    k_out_val = b.add(b.mul(g, c_kpg), lane_q)
    weights: List[Value] = []
    for r_const in range(p.KH):
        for s_const in range(p.KW):
            w_off, _ = b_desc.offset(
                b,
                k_out=k_out_val,
                r=b.const_i32(r_const),
                s=b.const_i32(s_const),
                c=c0,
            )
            weights.append(
                b.buffer_load_vN_f16(b_rsrc, b.mul(w_off, c_half_bytes), c0, 2)
            )

    q_tiles_per_wave = spec.block_q // 4
    acc_tiles: List[List[Value]] = [
        [zero_acc, zero_acc, zero_acc] for _ in range(q_tiles_per_wave)
    ]
    n_iters = p.H + p.KH - 1

    for y in range(n_iters):
        hi_in = b.sub(b.const_i32(y), c_PAD)
        hi_ok = b.land(b.cmp_ge(hi_in, c0), b.cmp_lt(hi_in, c_H))

        # Input descriptor: A[N, H, W, total_c] in NHWC.
        a_desc = TensorDescriptor.naive(
            "A",
            lengths=[p.N, p.H, p.W, p.total_c],
            coord_names=("n", "h", "w", "c"),
        )
        inputs_by_qtile: List[List[Value]] = []
        for qt in range(q_tiles_per_wave):
            q_base = b.add(q_tile_start, b.const_i32(qt * 4))
            inputs: List[Value] = []
            for s_const in range(p.KW):
                wi = b.sub(b.add(q_base, b.add(lane_q, b.const_i32(s_const))), c_PAD)
                wi_ok = b.land(b.cmp_ge(wi, c0), b.cmp_lt(wi, c_W))
                valid = b.land(hi_ok, wi_ok)
                a_off, _ = a_desc.offset(
                    b,
                    n=n,
                    h=hi_in,
                    w=wi,
                    c=b.mul(g, c_cpg),
                )
                safe_a = b.select(valid, b.mul(a_off, c_half_bytes), oob_sentinel)
                vec = b.buffer_load_vN_f16(a_rsrc, safe_a, c0, 2)
                vec = b.select(valid, vec, fp16x4_zero)
                inputs.append(vec)
            inputs_by_qtile.append(inputs)

        for qt in range(q_tiles_per_wave):
            accs = acc_tiles[qt]
            inputs = inputs_by_qtile[qt]
            for r_const in range(p.KH):
                p_idx = (y - r_const) % p.KH
                acc = accs[p_idx]
                for s_const in range(p.KW):
                    acc = b.mfma_f32_4x4x4_f16(
                        weights[r_const * p.KW + s_const], inputs[s_const], acc
                    )
                accs[p_idx] = acc

        p_flush = y - (p.KH - 1)
        P_FLUSH = p_flush % p.KH
        if 0 <= p_flush < p.H:
            d_desc = TensorDescriptor.naive(
                "D",
                lengths=[p.N, p.H, p.W, p.total_k],
                coord_names=("n", "h", "w", "k"),
            )
            for qt in range(q_tiles_per_wave):
                acc = acc_tiles[qt][P_FLUSH]
                q_base = b.add(q_tile_start, b.const_i32(qt * 4))
                out_q = b.add(q_base, lane_q)
                out_q_ok = b.cmp_lt(out_q, c_W)
                d_base, _ = d_desc.offset(
                    b,
                    n=n,
                    h=b.const_i32(p_flush),
                    w=out_q,
                    k=b.mul(g, c_kpg),
                )
                safe_d = b.select(out_q_ok, b.mul(d_base, c_half_bytes), oob_sentinel)
                # MFMA 4x4x4 wave64 per-lane output layout is
                #   acc[i]  ->  D[n, p_flush, out_q, g*kpg + i]  for i in 0..3
                # so the 4 elements are at consecutive halves in k_out
                # (kpg=4 means the 4 elements *fill* one group's k-channels).
                # Fuse the four `buffer_store_short` stores into a single
                # `buffer_store_dwordx2` (= 2 dwords = 4 halves) — runbook
                # §6.2's "vectorise the epilogue" lever.
                acc_h = b.vec_trunc_f32_to_f16(acc)
                b.buffer_store_vN_f16(d_rsrc, safe_d, c0, acc_h, 2)
        for qt in range(q_tiles_per_wave):
            acc_tiles[qt][P_FLUSH] = zero_acc

    return b.kernel

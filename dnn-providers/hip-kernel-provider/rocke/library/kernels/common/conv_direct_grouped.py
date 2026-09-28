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
against a dense convolution reference. `rocke.run_manifest` now verifies
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
from typing import List, Tuple

from rocke.core.ir import (
    BF16,
    F16,
    F32,
    I32,
    IRBuilder,
    KernelDef,
    PtrType,
    Value,
)
from rocke.helpers.transforms import TensorDescriptor, embed, unmerge_magic


def _io_type(dtype: str):
    """Return the IR type for a given ``dtype`` string (``"fp16"`` or ``"bf16"``)."""
    if dtype == "bf16":
        return BF16
    if dtype == "fp16":
        return F16
    raise ValueError(
        f"unsupported direct_conv dtype: {dtype!r}; expected 'fp16' or 'bf16'"
    )


def _buf_load_vN(
    b: IRBuilder, dtype: str, rsrc: Value, voff: Value, soff: Value, dwords: int
) -> Value:
    """Dtype-dispatch for vectorised buffer load.

    ``dwords`` matches the ``dwords`` parameter of ``buffer_load_vN_f16`` /
    ``buffer_load_vN_bf16``: each dword holds two 16-bit elements (dwords=1 ->
    2 elements, dwords=2 -> 4 elements, dwords=4 -> 8 elements).
    Uses the type-specific op names to stay byte-identical with the C++ engine.
    """
    if dtype == "bf16":
        return b.buffer_load_vN_bf16(rsrc, voff, soff, dwords)
    return b.buffer_load_vN_f16(rsrc, voff, soff, dwords)


def _buf_store_vN(
    b: IRBuilder,
    dtype: str,
    rsrc: Value,
    voff: Value,
    soff: Value,
    val: Value,
    dwords: int,
) -> None:
    """Dtype-dispatch for vectorised buffer store.

    ``dwords`` matches the ``dwords`` parameter of ``buffer_store_vN_f16`` /
    ``buffer_store_vN_bf16``: each dword holds two 16-bit elements.
    """
    if dtype == "bf16":
        b.buffer_store_vN_bf16(rsrc, voff, soff, val, dwords)
    else:
        b.buffer_store_vN_f16(rsrc, voff, soff, val, dwords)


def _trunc_f32(b: IRBuilder, dtype: str, val: Value) -> Value:
    """Truncate a vector of f32 accumulators to the output dtype."""
    if dtype == "bf16":
        return b.vec_trunc_f32_to_bf16(val)
    return b.vec_trunc_f32_to_f16(val)


def _mfma(
    b: IRBuilder,
    dtype: str,
    shape: str,
    a: Value,
    b_val: Value,
    acc: Value,
) -> Value:
    """Dtype-dispatch for a single MFMA call.

    ``shape`` is the size suffix without the dtype, e.g. ``"16x16x16"``,
    ``"16x16x32"``, or ``"32x32x8"``.
    """
    if dtype == "bf16":
        fn = getattr(b, f"mfma_f32_{shape}_bf16")
    else:
        fn = getattr(b, f"mfma_f32_{shape}_f16")
    return fn(a, b_val, acc)


@dataclass(frozen=True)
class DirectConvProblem:
    """The grouped direct-conv shape parameters.

    Layouts:
      A: NHWC, `[N, H, W, groups*cpg]`
      B: KRSC, `[groups*kpg, KH, KW, cpg]`
      D: NHWK, `[N, H, W, groups*kpg]`
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
    dtype: str = "fp16"  # "fp16" or "bf16"

    @property
    def total_c(self) -> int:
        return self.groups * self.cpg

    @property
    def total_k(self) -> int:
        return self.groups * self.kpg

    @property
    def Ho(self) -> int:
        """Output height for a strided convolution."""
        return (self.H + 2 * self.PAD - self.KH) // self.stride + 1

    @property
    def Wo(self) -> int:
        """Output width for a strided convolution."""
        return (self.W + 2 * self.PAD - self.KW) // self.stride + 1

    @property
    def flops(self) -> int:
        return (
            2
            * self.N
            * self.Ho
            * self.Wo
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
        from rocke.helpers.spec import kernel_name_join

        p = self.problem
        return kernel_name_join(
            self.name,
            p.short(),
            f"bq{self.block_q}",
            f"bg{self.block_groups}",
            "db" if self.double_buffer else "sb",
            flags={"k32": self.fold_k32, "bf16": p.dtype == "bf16"},
        )

    def validate(self) -> None:
        p = self.problem
        if p.dtype not in ("fp16", "bf16"):
            raise ValueError(f"DirectConv16cSpec: unsupported dtype {p.dtype!r}")
        if p.cpg != 16 or p.kpg != 16:
            raise ValueError(
                f"DirectConv16cSpec expects cpg=kpg=16 (got {p.cpg}, {p.kpg})"
            )
        if p.groups % self.block_groups != 0:
            raise ValueError(
                f"groups {p.groups} not divisible by block_groups {self.block_groups}"
            )


def is_valid_spec_16c(
    spec: DirectConv16cSpec, arch: str = "gfx950"
) -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for a 16c spec on ``arch``.

    The 16c kernel's inner MFMA shape depends on ``fold_k32``:
      - ``fold_k32=True`` (default) folds S=0/1 into one ``16x16x32``
        f16 MFMA (the wide K-packed atom). That atom only exists on
        gfx950; requesting it on gfx942 would crash comgr
        (``LLVM ERROR: Cannot select intrinsic
        ...mfma.f32.16x16x32.f16``), so it is rejected here with a clean
        structured reason. Use ``fold_k32=False`` for a gfx942-capable
        kernel (it issues only ``16x16x16`` f16 MFMAs).
      - ``fold_k32=False`` uses only the ``16x16x16`` f16 atom, which is
        present on both gfx942 and gfx950.
    The atom legality is sourced from
    :class:`rocke.core.arch.ArchTarget`.
    """
    from rocke.core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)
    p = spec.problem
    if p.dtype not in ("fp16", "bf16"):
        return False, f"unsupported dtype {p.dtype!r}; expected 'fp16' or 'bf16'"
    if p.stride != 1:
        return False, f"stride > 1 is not supported (got {p.stride})"
    if p.cpg != 16 or p.kpg != 16:
        return False, f"DirectConv16cSpec expects cpg=kpg=16 (got {p.cpg}, {p.kpg})"
    if p.groups % spec.block_groups != 0:
        return False, (
            f"groups {p.groups} not divisible by block_groups {spec.block_groups}"
        )
    ab_dtype = "bf16" if p.dtype == "bf16" else "f16"
    if not target.mma.has_shape(
        a_dtype=ab_dtype, b_dtype=ab_dtype, c_dtype="fp32", m=16, n=16, k=16
    ):
        return False, f"missing 16x16x16 {ab_dtype} MFMA atom on {arch}"
    if spec.fold_k32 and not target.mma.has_shape(
        a_dtype=ab_dtype, b_dtype=ab_dtype, c_dtype="fp32", m=16, n=16, k=32
    ):
        return False, (
            f"fold_k32=True needs the 16x16x32 {ab_dtype} MFMA atom, absent on "
            f"{arch}; use fold_k32=False for a {arch}-capable kernel"
        )
    return True, "ok"


def build_direct_conv_16c(
    spec: DirectConv16cSpec, *, arch: str = "gfx950"
) -> KernelDef:
    """Build the IR for one direct conv 16c kernel instance.

    See the module docstring for the kernel structure. The Python
    builder unrolls every Python `for` loop at IR-build time; the
    only runtime loop is the H-row streaming `scf.for`.

    ``arch`` (``"gfx942"`` / ``"gfx950"``) selects the target GPU. When
    ``spec.fold_k32`` is True the inner loop emits the wide
    ``16x16x32`` f16 MFMA, which only exists on gfx950; requesting
    ``gfx942`` then fails with a clean structured error (via
    :func:`is_valid_spec_16c`) instead of crashing comgr. Set
    ``fold_k32=False`` for a gfx942-capable instance.
    """
    spec.validate()
    ok, why = is_valid_spec_16c(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid direct_conv_16c spec for {arch}: {why}")
    p = spec.problem
    io_type = _io_type(p.dtype)
    BLOCK_Q = spec.block_q
    BLOCK_GROUPS = spec.block_groups
    WAVE = spec.wave_size
    THREADS = spec.threads_per_block
    Ho = p.Ho
    Wo = p.Wo
    LDS_W = (BLOCK_Q - 1) * p.stride + p.KW
    LDS_ROW_FP16 = LDS_W * BLOCK_GROUPS * p.cpg
    LOAD_VEC = 4
    NUM_VEC4 = LDS_ROW_FP16 // LOAD_VEC

    if NUM_VEC4 == 0:
        raise ValueError("LDS row too small for one vec4 per thread")
    PASSES = (NUM_VEC4 + THREADS - 1) // THREADS
    c_stride = p.stride  # Python int, used in emit-time guards below

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = THREADS

    A = b.param("A", PtrType(io_type, "global"), noalias=True, readonly=True, align=16)
    Bp = b.param("B", PtrType(io_type, "global"), noalias=True, readonly=True, align=16)
    D = b.param("D", PtrType(io_type, "global"), noalias=True, writeonly=True, align=16)
    A_bytes = b.param("A_bytes", I32)
    B_bytes = b.param("B_bytes", I32)
    D_bytes = b.param("D_bytes", I32)

    c0 = b.const_i32(0)
    c_wave = b.const_i32(WAVE)
    c_BG = b.const_i32(BLOCK_GROUPS)
    c_BQ = b.const_i32(BLOCK_Q)
    c_cpg = b.const_i32(p.cpg)
    c_kpg = b.const_i32(p.kpg)
    c_W = b.const_i32(Wo)

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
    A_smem = b.smem_alloc(io_type, [1, lds_total_fp16], name_hint="lds_a")
    B_smem = (
        b.smem_alloc(io_type, [1, lds_total_fp16], name_hint="lds_b")
        if spec.double_buffer
        else A_smem
    )

    # Buffer rsrcs.
    a_rsrc = b.buffer_rsrc(A, A_bytes)
    b_rsrc = b.buffer_rsrc(Bp, B_bytes)
    d_rsrc = b.buffer_rsrc(D, D_bytes)

    c_half_bytes = b.const_i32(2)
    oob_sentinel = b.const_i32((1 << 31) - 1)
    fp16x4_zero = b.zero_vec(io_type, 4)
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
    weights_s2_k32: List[Value] = []
    # ``lane_in_lo_half`` is true for the two lane groups (c4 in {0, 1})
    # that carry the low 16 K of a folded K=32 atom. The S=2 residual is
    # promoted to a *second* wide K=32 atom whose upper 16 K (lane groups
    # c4 in {2, 3}) are zero-padded, so its accumulator chain stays the
    # same width as the S=0/1 atom (see the MFMA comment below).
    lane_in_lo_half = b.cmp_lt(c4, b.const_i32(2))
    fp16x8_zero = b.zero_vec(io_type, 8)
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
                _buf_load_vN(b, p.dtype, b_rsrc, b.mul(w_off_k32, c_half_bytes), c0, 4)
            )
            # Residual S=2 promoted to a zero-padded K=32 atom. The low
            # half (c4 in {0,1}) carries B[k_out, r, 2, 0:8] / [8:16]; the
            # high half (c4 in {2,3}) is zeroed so it contributes nothing.
            w_off_s2, _ = b_desc.offset(
                b,
                k_out=k_out_val,
                r=r_i,
                s=b.const_i32(2),
                c=ch_lane_k32,
            )
            w_s2 = _buf_load_vN(
                b, p.dtype, b_rsrc, b.mul(w_off_s2, c_half_bytes), c0, 4
            )
            weights_s2_k32.append(b.select(lane_in_lo_half, w_s2, fp16x8_zero))
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
                    _buf_load_vN(b, p.dtype, b_rsrc, b.mul(w_off, c_half_bytes), c0, 2)
                )

    # ---- LDS load helper ----
    # Each thread loads a vec4 of `cpg=16` halves of input from DRAM
    # at (n, hi, wi, c) -> LDS at index `chunk_idx * 4`.
    # The per-thread chunk decomposition used to be five hand-rolled
    # div/mod/mul/add chains:
    #   ch_block    = chunk_idx % 4
    #   gw_idx      = chunk_idx // 4
    #   group_in_wg = gw_idx % BLOCK_GROUPS
    #   W_lds       = gw_idx // BLOCK_GROUPS
    #   W_in        = q_tile_start + W_lds - PAD
    #   abs_group   = g_tile * BLOCK_GROUPS + group_in_wg
    #   c_val       = abs_group * cpg + ch_block * 4
    # which is exactly the CK Tile pattern "unmerge then embed" — a
    # flat per-wave chunk index split into (W_lds, group_in_wg,
    # ch_block) via ``unmerge``, then the per-axis embed maps
    # (group_in_wg, ch_block) -> c and (q_tile_start, W_lds) -> w (with
    # the -PAD shift folded in). We use that algebra via a
    # :class:`TensorDescriptor` chain so the ad-hoc SSA disappears
    # behind one ``a_desc.offset(...)`` call per chunk.
    # The per-wave chunk index splits into (W_lds, group_in_wg,
    # ch_block) -- a ``merge((LDS_W, BLOCK_GROUPS, 4))`` whose inverse is
    # the CK Tile default magic-division unmerge
    # (``merge_v2_magic_division`` -> :class:`UnmergeMagicDiv`). Driving
    # the split through the descriptor's :meth:`unmerge_lower` (instead
    # of the prior inline ``b.div`` / ``b.mod`` chain) removes the two
    # integer divisions per chunk from the loader's address path and
    # turns the documentation-only ``chunk_desc`` into the live decode.
    chunk_desc = TensorDescriptor.naive(
        "chunk_unmerge",
        lengths=[LDS_W, BLOCK_GROUPS, 4],
        coord_names=("W_lds", "group_in_wg", "ch_block"),
    ).transform(
        unmerge_magic(
            "chunk_idx",
            into=("W_lds", "group_in_wg", "ch_block"),
            dims=[LDS_W, BLOCK_GROUPS, 4],
        ),
    )
    chunk_meta = []
    for pass_idx in range(PASSES):
        chunk_idx = b.add(tid, b.const_i32(pass_idx * THREADS))
        decoded = chunk_desc.unmerge_lower(b, chunk_idx=chunk_idx)
        ch_block = decoded["ch_block"]
        group_in_wg = decoded["group_in_wg"]
        W_lds = decoded["W_lds"]
        in_bounds = b.cmp_lt(chunk_idx, b.const_i32(NUM_VEC4))
        abs_group = b.add(b.mul(g_tile, c_BG), group_in_wg)
        chunk_meta.append(
            {
                "chunk_idx": chunk_idx,
                "ch_block": ch_block,
                "group_in_wg": group_in_wg,
                "W_lds": W_lds,
                "in_bounds": in_bounds,
                "abs_group": abs_group,
            }
        )

    # Input descriptor: A[N, H, W, total_c] in NHWC. Two embeds fold
    # the conv-spatial coord algebra into the descriptor so the loader
    # body no longer carries hand-rolled add/sub chains for h and w:
    #
    #   * ``embed(("y_iter",) -> "h", strides=(1,), offset=-PAD,
    #            lo=0, hi=H)``  — folds the per-iter ``hi = y - PAD``
    #     and the (0 <= hi < H) boundary check that used to live in
    #     the ``pad("h", ...)`` transform.
    #   * ``embed(("q_pos","W_lds_pos") -> "w", strides=(1,1),
    #            offset=-PAD, lo=0, hi=W)`` — folds ``wi =
    #     q_tile_start + W_lds - PAD`` plus the (0 <= wi < W) check
    #     that used to live in ``pad("w", ...)``. The lifted scalar
    #     ``W_in = q_tile_start + W_lds - PAD`` chain in the previous
    #     version was redundant once the descriptor carried this.
    #
    # The remaining ``c`` coord stays manual: ``c = abs_group * cpg +
    # ch_block * 4`` is in [0, total_c) by construction (abs_group <
    # groups, ch_block < 4), so wrapping it in an ``embed`` with
    # ``lo=0, hi=total_c`` would add a redundant bounds-check
    # (``cmp_ge`` / ``cmp_lt`` / ``land``) per chunk — the
    # transforms.Embed always emits its bounds AND, regardless of how
    # trivially provable the range is. We skip the embed and pass
    # ``c=c_val`` directly to keep the SSA count tight on a hot path.
    a_desc = TensorDescriptor.naive(
        "A",
        lengths=[p.N, p.H, p.W, p.total_c],
        coord_names=("n", "h", "w", "c"),
    ).transform(
        embed(
            upper=("y_iter",),
            into="h",
            strides=(1,),
            offset=-p.PAD,
            lo=0,
            hi=p.H,
        ),
        embed(
            upper=("q_pos", "W_lds_pos"),
            into="w",
            strides=(p.stride, 1),
            offset=-p.PAD,
            lo=0,
            hi=p.W,
        ),
    )

    def issue_dram_load(y_iter_val: Value):
        """Per-thread DRAM read of one vec4 of A.

        Returns `(vec4, lds_idx)` pairs; the caller decides when to
        store them to LDS. This is important for the v6 pipeline:
        issue DRAM reads for row y+1 before the MFMAs on row y, then
        write those prefetched registers to the next LDS buffer after
        the MFMAs. That preserves the read-before-write ordering on
        the current buffer while overlapping the VMEM latency with
        compute.

        ``y_iter_val`` is the input-row index (descriptor embed folds
        the ``- PAD`` and the (0 <= h < H) boundary check). The
        per-thread spatial coords (``q_pos``, ``W_lds_pos``) flow
        through the ``w`` embed; only the ``c`` coord (cheap mul-add
        with statically-known range) is computed inline to keep the
        descriptor from emitting a redundant bounds AND.
        """
        out = []
        for cm in chunk_meta:
            c_val = b.add(
                b.mul(cm["abs_group"], c_cpg),
                b.mul(cm["ch_block"], b.const_i32(4)),
            )
            a_off_elems, addr_valid = a_desc.offset(
                b,
                n=n,
                y_iter=y_iter_val,
                q_pos=q_tile_start,
                W_lds_pos=cm["W_lds"],
                c=c_val,
            )
            valid = b.land(addr_valid, cm["in_bounds"])
            a_off_bytes = b.mul(a_off_elems, c_half_bytes)
            safe_off = b.select(valid, a_off_bytes, oob_sentinel)
            a_vec = _buf_load_vN(b, p.dtype, a_rsrc, safe_off, c0, 2)
            a_vec = b.select(valid, a_vec, fp16x4_zero)
            # LDS index in halves: chunk_idx * 4. Allocation is 2D
            # `[1, ROW]` so we pass (row=0, col=lds_idx).
            lds_idx = b.mul(cm["chunk_idx"], b.const_i32(4))
            out.append((a_vec, lds_idx))
        return out

    def store_to_lds(loads, lds: Value) -> None:
        for a_vec, lds_idx in loads:
            b.smem_store_vN(lds, [c0, lds_idx], a_vec, 4)

    q_subtiles = BLOCK_Q // 16

    def lds_read_input(q_subtile: int, s_const: int, lds: Value) -> Value:
        """Per-lane <4 x half> read from LDS for the s-th filter column.

        LDS layout: (W_lds, group_in_wg, channel) row-major, stride BG*cpg
        per W_lds position. Lane ``q_in_lane`` owns output column
        ``q_tile_start + q_subtile*16 + q_in_lane``; the input column for
        filter tap ``s`` is at LDS offset ``q_in_lane * stride + s`` (within
        the subtile block starting at ``q_subtile * 16 * stride``).
        """
        W_lds_idx = b.add(
            b.mul(b.add(q_in_lane, b.const_i32(q_subtile * 16)), b.const_i32(c_stride)),
            b.const_i32(s_const),
        )
        lds_idx = b.add(
            b.add(
                b.mul(W_lds_idx, c_BG_cpg),
                b.mul(wave_id, c_cpg),
            ),
            b.mul(c4, b.const_i32(4)),
        )
        return b.smem_load_vN(lds, c0, lds_idx, dtype=io_type, n=4)

    def lds_read_input_k32(q_subtile: int, lds: Value) -> Value:
        """Per-lane <8 x half> read for the folded K=32 MFMA.

        ``s_lane_k32 = c4 // 2`` selects filter column 0 or 1 within the
        folded pair. The LDS offset for output lane ``q_in_lane`` and filter
        tap ``s_lane_k32`` is ``(q_in_lane + q_subtile*16) * stride + s_lane_k32``.
        """
        W_lds_idx = b.add(
            b.mul(b.add(q_in_lane, b.const_i32(q_subtile * 16)), b.const_i32(c_stride)),
            s_lane_k32,
        )
        lds_idx = b.add(
            b.add(
                b.mul(W_lds_idx, c_BG_cpg),
                b.mul(wave_id, c_cpg),
            ),
            ch_lane_k32,
        )
        return b.smem_load_vN(lds, c0, lds_idx, dtype=io_type, n=8)

    def lds_read_input_s2_k32(q_subtile: int, lds: Value) -> Value:
        """Per-lane <8 x half> input read for the S=2 residual, promoted to
        a zero-padded K=32 atom.

        The low half (c4 in {0,1}) reads filter column s=2 at LDS offset
        ``(q_in_lane + q_subtile*16) * stride + 2``; the high half (c4 in
        {2,3}) is zeroed so the wide atom's upper 16 K contribute nothing.
        Promoting S=2 to a wide atom keeps the per-(r) MFMA chain
        homogeneous-width (wide -> wide on one accumulator), which avoids
        the cross-width MFMA read-after-write accumulator hazard.
        """
        W_lds_idx = b.add(
            b.mul(b.add(q_in_lane, b.const_i32(q_subtile * 16)), b.const_i32(c_stride)),
            b.const_i32(2),
        )
        lds_idx = b.add(
            b.add(
                b.mul(W_lds_idx, c_BG_cpg),
                b.mul(wave_id, c_cpg),
            ),
            ch_lane_k32,
        )
        vec = b.smem_load_vN(lds, c0, lds_idx, dtype=io_type, n=8)
        return b.select(lane_in_lo_half, vec, fp16x8_zero)

    # ---- prologue: prefetch row 0 (= -PAD..-PAD+1 = -1) into A_smem ----
    # The first iter's input row is hi = 0 - PAD = -1 for PAD=1, which
    # is invalid (above the image). The descriptor's embed("y_iter",
    # offset=-PAD, lo=0, hi=H) flips the validity to false; the loader
    # then replaces the byte offset with the OOB sentinel + zero-fill
    # so the prologue effectively zero-fills A_smem for iter 0.
    store_to_lds(issue_dram_load(c0), A_smem)
    b.sync()

    # ---- the H-row streaming loop ----
    # Iterates over input rows y = 0 .. H + KH - 2. For each input row
    # the kernel accumulates KH contributions; accumulator slot
    # p_flush_val = y - (KH-1) is flushed when it becomes valid and
    # (for stride > 1) when it aligns to an output row.
    n_iters = p.H + p.KH - 1
    acc_tiles: List[List[Value]] = [
        [zero_acc, zero_acc, zero_acc] for _ in range(q_subtiles)
    ]

    # Output descriptor: D[N, Ho, Wo, total_k] in NHWK. Built ONCE
    # outside the H-loop so each iter only pays one ``d_desc.offset``
    # SSA emission rather than reconstructing the descriptor object.
    d_desc = TensorDescriptor.naive(
        "D",
        lengths=[p.N, Ho, Wo, p.total_k],
        coord_names=("n", "h", "w", "k"),
    )

    for y in range(n_iters):
        cur = A_smem if (y % 2 == 0 or not spec.double_buffer) else B_smem
        nxt = B_smem if (y % 2 == 0 or not spec.double_buffer) else A_smem

        # Read inputs from the current buffer first; no writes to
        # `cur` are issued until the next time it becomes `nxt`.
        if spec.fold_k32:
            inputs_by_q = [
                (lds_read_input_k32(qt, cur), lds_read_input_s2_k32(qt, cur))
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
        # ``y_iter`` is the unshifted row index; the A_desc embed folds
        # the -PAD and the (0 <= h < H) check.
        loads_next = None
        if y + 1 < n_iters:
            loads_next = issue_dram_load(b.const_i32(y + 1))

        for qt in range(q_subtiles):
            accs = acc_tiles[qt]
            for r_const in range(p.KH):
                p_idx = (y - r_const) % p.KH
                acc_in = accs[p_idx]
                if spec.fold_k32:
                    input_k32, input_s2 = inputs_by_q[qt]
                    # CORRECTNESS-CRITICAL: both folded MFMAs are the *same*
                    # width (wide K=32). S=0/1 fold into one 16x16x32 atom;
                    # the S=2 residual is promoted to a SECOND 16x16x32 atom
                    # with its upper 16 K zero-padded (``weights_s2_k32`` /
                    # ``lds_read_input_s2_k32`` zero the c4 in {2,3} lane
                    # groups). Chaining two same-width atoms on one
                    # accumulator -- ``acc = k32(s2pad, k32(s01, acc))`` --
                    # matches the mfma_gemm hero path that runs the wide atom
                    # correctly. The earlier fold mixed a 16x16x16 residual
                    # into the same accumulator as the 16x16x32 atom; a narrow
                    # MFMA whose C-operand is the just-written result of a wide
                    # MFMA (or vice versa) is a read-after-write accumulator
                    # hazard that BOTH the comgr LLVM-direct backend AND hipcc
                    # miscompile in this fully-unrolled kernel (the wide atom's
                    # longer accumulation latency is dropped when its result
                    # feeds the next, different-width MFMA's C input), silently
                    # corrupting accumulator slots on the H-edge output rows in
                    # a SHAPE-DEPENDENT way (~0.5-0.8% bad, max_abs ~360).
                    # Keeping both atoms the same width removes the hazard and
                    # keeps a single accumulator per slot (no occupancy hit
                    # from a second accumulator triple). Verified bad=0 across
                    # shapes on gfx950 via both comgr and hipcc.
                    #
                    # NOTE: this builder still rides the legacy hand-rolled
                    # MFMA lane math (s_lane_k32 / ch_lane_k32 magic constants)
                    # rather than the unified ``op_for_shape`` +
                    # ``op.c_layout().coord(...)`` contract that mfma_gemm is
                    # migrating to (refactor_opportunities.md items 1-4).
                    # Migrating the C-accumulator readout + A/B K-pack to
                    # c_layout().coord would delete this whole hazard class at
                    # the source; tracked as a follow-up.
                    acc_in = _mfma(
                        b, p.dtype, "16x16x32", weights_k32[r_const], input_k32, acc_in
                    )
                    acc_in = _mfma(
                        b,
                        p.dtype,
                        "16x16x32",
                        weights_s2_k32[r_const],
                        input_s2,
                        acc_in,
                    )
                else:
                    inputs = inputs_by_q[qt]
                    for s_const in range(p.KW):
                        w_idx = r_const * p.KW + s_const
                        acc_in = _mfma(
                            b,
                            p.dtype,
                            "16x16x16",
                            weights[w_idx],
                            inputs[s_const],
                            acc_in,
                        )
                accs[p_idx] = acc_in

        if loads_next is not None:
            # Single-buffer correctness barrier. When ``double_buffer`` is
            # False, ``cur`` and ``nxt`` are the SAME LDS allocation, so
            # the ``store_to_lds`` below overwrites the row this iteration
            # just read via ``lds_read_input``. With more than one wave per
            # workgroup (``block_groups > 1``) the only barrier used to be
            # the one at the end of the iteration, so a fast wave could
            # begin storing row y+1 into LDS while a slower wave was still
            # issuing its ds_reads for row y -- a read-after-write race that
            # corrupted the slower waves' inputs (seen as *nondeterministic*
            # wrong outputs concentrated in the interior waves/groups and
            # near the H/W edges). The next-row DRAM loads were already
            # issued into registers above, so this barrier only forces every
            # wave to finish reading the current LDS row before any wave
            # overwrites it; the MFMAs above overlap the ds_read latency.
            # The double-buffer path doesn't need it (the store targets the
            # other ping-pong buffer).
            if not spec.double_buffer:
                b.sync()
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
        if 0 <= p_flush_val < p.H and p_flush_val % c_stride == 0:
            # ``p_flush_val`` is the input row that produced a complete set
            # of KH contributions. For stride > 1 only rows that align to
            # an output position (p_flush_val % stride == 0) generate a
            # write; the output row index is p_flush_val // stride.
            ho_row = p_flush_val // c_stride
            for qt in range(q_subtiles):
                acc_to_flush = acc_tiles[qt][P_FLUSH]
                out_q = b.add(b.add(q_tile_start, b.const_i32(qt * 16)), q_in_lane)
                out_q_valid = b.cmp_lt(out_q, c_W)
                k_val = b.add(b.mul(g, c_kpg), b.mul(c4, b.const_i32(4)))
                d_base, _ = d_desc.offset(
                    b,
                    n=n,
                    h=b.const_i32(ho_row),
                    w=out_q,
                    k=k_val,
                )
                d_base_bytes = b.mul(d_base, c_half_bytes)
                safe_d_off = b.select(out_q_valid, d_base_bytes, oob_sentinel)
                # The 4 per-lane output elements are contiguous in NHWK:
                # k_out = g*kpg + c4*4 + [0..3].  Store them as one
                # 64-bit vector instead of four scalar buffer_store_short
                # ops.
                acc_h = _trunc_f32(b, p.dtype, acc_to_flush)
                _buf_store_vN(b, p.dtype, d_rsrc, safe_d_off, c0, acc_h, 2)
        # Unconditional slot reset - kills early-iter leaks before they
        # pollute a later output row.
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
        from rocke.helpers.spec import kernel_name_join

        p = self.problem
        return kernel_name_join(
            self.name, p.short(), f"bq{self.block_q}", f"bg{self.block_groups}"
        )

    def validate(self) -> None:
        p = self.problem
        if p.dtype != "fp16":
            raise ValueError(
                f"DirectConv4cSpec: bf16 is not supported - the mfma_f32_4x4x4 atom "
                f"is fp16-only on CDNA; use fp16 dtype or a different cpg variant"
            )
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


def is_valid_spec_4c(spec: DirectConv4cSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for a 4c spec on ``arch``.

    The 4c kernel uses the tiny ``mfma_f32_4x4x4_f16`` atom (16
    independent 4x4x4 matmuls per wave). That intrinsic is selectable on
    both gfx942 and gfx950, so the kernel is arch-neutral: ``arch`` is
    validated against :class:`rocke.core.arch.ArchTarget` (unknown gfx
    names rejected) but does not change the emitted MFMA. The 4x4x4 atom
    is deliberately not gated through the MMA catalog ``has_shape`` check
    because the catalog lists only the warp-tile (16x16 / 32x32) shapes,
    while comgr selects the 4x4x4 intrinsic directly on both targets.
    bf16 is not supported — no 4x4x4 bf16 MFMA atom exists on CDNA.
    """
    from rocke.core.arch import ArchTarget

    try:
        ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)
    p = spec.problem
    if p.dtype != "fp16":
        return False, (
            f"DirectConv4cSpec: bf16 not supported - no mfma_f32_4x4x4_bf16 atom on CDNA"
        )
    if p.stride != 1:
        return False, f"stride > 1 is not supported (got {p.stride})"
    if p.cpg != 4 or p.kpg != 4:
        return False, f"DirectConv4cSpec expects cpg=kpg=4 (got {p.cpg}, {p.kpg})"
    if spec.block_groups % 16 != 0:
        return False, "DirectConv4cSpec block_groups must be a multiple of 16"
    if spec.block_q % 4 != 0:
        return False, "DirectConv4cSpec block_q must be a multiple of 4"
    if p.groups % spec.block_groups != 0:
        return False, (
            f"groups {p.groups} not divisible by block_groups {spec.block_groups}"
        )
    return True, "ok"


def build_direct_conv_4c(spec: DirectConv4cSpec, *, arch: str = "gfx950") -> KernelDef:
    """Build the direct grouped 4c kernel using MFMA 4x4x4.

    Each lane has:
      - batch = lane / 4 -> group within the workgroup (0..15)
      - lane_q = lane % 4 -> output W position and output channel row

    The MFMA output vector `<4 x f32>` maps to output channels
    `k_in_group = 0..3` at fixed output W position `lane_q`.

    ``arch`` (``"gfx942"`` / ``"gfx950"``) selects the target GPU. The
    ``mfma_f32_4x4x4_f16`` atom this kernel uses is selectable on both
    targets, so the kernel is arch-neutral; ``arch`` is validated (via
    :func:`is_valid_spec_4c`) but does not change the emitted IR.
    """
    spec.validate()
    ok, why = is_valid_spec_4c(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid direct_conv_4c spec for {arch}: {why}")
    p = spec.problem
    Ho = p.Ho
    Wo = p.Wo
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.threads_per_block

    A = b.param("A", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    Bp = b.param("B", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    D = b.param("D", PtrType(F16, "global"), noalias=True, writeonly=True, align=16)
    A_bytes = b.param("A_bytes", I32)
    B_bytes = b.param("B_bytes", I32)
    D_bytes = b.param("D_bytes", I32)

    c0 = b.const_i32(0)
    c_W = b.const_i32(Wo)
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
    c_stride_4c = p.stride  # Python int used in emit-time flush guard

    # Input descriptor: A[N, H, W, total_c] in NHWC.
    # y_iter is the input row index; wo is the output W position.
    # h = y_iter - PAD  (stride-1 in H: the loop walks input rows)
    # w = wo * stride + s - PAD  (stride in W from output column)
    a_desc = TensorDescriptor.naive(
        "A",
        lengths=[p.N, p.H, p.W, p.total_c],
        coord_names=("n", "h", "w", "c"),
    ).transform(
        embed(
            upper=("y_iter",),
            into="h",
            strides=(1,),
            offset=-p.PAD,
            lo=0,
            hi=p.H,
        ),
        embed(
            upper=("wo", "s"),
            into="w",
            strides=(p.stride, 1),
            offset=-p.PAD,
            lo=0,
            hi=p.W,
        ),
    )

    # Output descriptor: D[N, Ho, Wo, total_k] in NHWK.
    d_desc = TensorDescriptor.naive(
        "D",
        lengths=[p.N, Ho, Wo, p.total_k],
        coord_names=("n", "h", "w", "k"),
    )

    c_val_groupc = b.mul(g, c_cpg)
    # ``q_pos = q_base + lane_q`` (= ``wo`` for the embed) is the same
    # across all KW values within a qt iter, and ``q_base`` only
    # depends on the unrolled Python ``qt`` index, so we precompute it
    # per qt outside the s-loop. Per-(qt, s) the loader then passes
    # ``wo=q_pos`` and ``s=const`` straight to the descriptor.
    s_consts = [b.const_i32(s) for s in range(p.KW)]

    for y in range(n_iters):
        y_iter = b.const_i32(y)

        inputs_by_qtile: List[List[Value]] = []
        for qt in range(q_tiles_per_wave):
            q_base = b.add(q_tile_start, b.const_i32(qt * 4))
            q_pos = b.add(q_base, lane_q)
            inputs: List[Value] = []
            for s_idx, s_val in enumerate(s_consts):
                a_off, valid = a_desc.offset(
                    b,
                    n=n,
                    y_iter=y_iter,
                    wo=q_pos,
                    s=s_val,
                    c=c_val_groupc,
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
        if 0 <= p_flush < p.H and p_flush % c_stride_4c == 0:
            ho_row = p_flush // c_stride_4c
            k_out_base = b.mul(g, c_kpg)
            for qt in range(q_tiles_per_wave):
                acc = acc_tiles[qt][P_FLUSH]
                q_base = b.add(q_tile_start, b.const_i32(qt * 4))
                out_q = b.add(q_base, lane_q)
                out_q_ok = b.cmp_lt(out_q, c_W)
                d_base, _ = d_desc.offset(
                    b,
                    n=n,
                    h=b.const_i32(ho_row),
                    w=out_q,
                    k=k_out_base,
                )
                safe_d = b.select(out_q_ok, b.mul(d_base, c_half_bytes), oob_sentinel)
                # MFMA 4x4x4 wave64 per-lane output layout:
                #   acc[i] -> D[n, ho_row, out_q, g*kpg + i]  for i in 0..3
                acc_h = _trunc_f32(b, p.dtype, acc)
                _buf_store_vN(b, p.dtype, d_rsrc, safe_d, c0, acc_h, 2)
        for qt in range(q_tiles_per_wave):
            acc_tiles[qt][P_FLUSH] = zero_acc

    return b.kernel


# ---------------------------------------------------------------------------
# 8c kernel — cpg = kpg = 8, mfma_f32_16x16x16_f16, one group per wave
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DirectConv8cSpec:
    """Direct grouped convolution kernel for ``cpg = kpg = 8``.

    Uses the same ``mfma_f32_16x16x16_f16`` atom as the 16c kernel, but
    with ``cpg = kpg = 8``.  Because cpg (8) is half of the K tile width (16),
    two filter columns ``s=0`` and ``s=1`` are folded into the K=16 dimension
    of a single MFMA call, and the residual ``s=2`` column is handled as a
    zero-padded K=16 atom (upper 8 K lanes carry zeros).

    Lane layout (wave64, ``mfma_f32_16x16x16_f16``):
      ``q_in_lane = lane % 16`` — N (output W position) and M (k_out row within group)
      ``c4        = lane // 16`` — K-block selector (0..3)

    K-fold mapping:
      c4 = 0 → s = 0, ch = 0..3   (valid)
      c4 = 1 → s = 0, ch = 4..7   (valid)
      c4 = 2 → s = 1, ch = 0..3   (valid for main atom; zero for s=2 residual)
      c4 = 3 → s = 1, ch = 4..7   (valid for main atom; zero for s=2 residual)

    Output (M dimension): M rows 0..kpg-1 (``q_in_lane < 8``) are valid.
    Rows 8..15 are produced by the MFMA but correspond to out-of-group k_out
    values and are never stored (gated by ``c4 < kpg // 4``).

    Block geometry:
      ``BLOCK_Q = 16`` (N=16 output W positions per block).
      ``BLOCK_GROUPS`` waves per block, each wave owns one group.
      ``threads_per_block = block_groups * 64``.

    Architecture: gfx942 and gfx950 (uses only ``mfma_f32_16x16x16_f16``).
    """

    problem: DirectConvProblem
    name: str = "direct_conv_8c"
    block_q: int = 16
    block_groups: int = 8
    wave_size: int = 64
    double_buffer: bool = True

    @property
    def threads_per_block(self) -> int:
        return self.block_groups * self.wave_size

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        p = self.problem
        return kernel_name_join(
            self.name,
            p.short(),
            f"bq{self.block_q}",
            f"bg{self.block_groups}",
            "db" if self.double_buffer else "sb",
            flags={"bf16": p.dtype == "bf16"},
        )

    def validate(self) -> None:
        p = self.problem
        if p.dtype not in ("fp16", "bf16"):
            raise ValueError(f"DirectConv8cSpec: unsupported dtype {p.dtype!r}")
        if p.cpg != 8 or p.kpg != 8:
            raise ValueError(
                f"DirectConv8cSpec expects cpg=kpg=8 (got {p.cpg}, {p.kpg})"
            )
        if p.groups % self.block_groups != 0:
            raise ValueError(
                f"groups {p.groups} not divisible by block_groups {self.block_groups}"
            )
        if self.block_q % 16 != 0:
            raise ValueError("DirectConv8cSpec block_q must be a multiple of 16")


def is_valid_spec_8c(spec: DirectConv8cSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for an 8c spec on ``arch``.

    The 8c kernel folds two S-positions into the K=16 dimension of
    ``mfma_f32_16x16x16_f16`` (or the bf16 counterpart), which is present on
    both gfx942 and gfx950.
    """
    from rocke.core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)
    p = spec.problem
    if p.dtype not in ("fp16", "bf16"):
        return False, f"unsupported dtype {p.dtype!r}; expected 'fp16' or 'bf16'"
    if p.stride != 1:
        return False, f"stride > 1 is not supported (got {p.stride})"
    if p.cpg != 8 or p.kpg != 8:
        return False, f"DirectConv8cSpec expects cpg=kpg=8 (got {p.cpg}, {p.kpg})"
    if p.groups % spec.block_groups != 0:
        return (
            False,
            f"groups {p.groups} not divisible by block_groups {spec.block_groups}",
        )
    if spec.block_q % 16 != 0:
        return False, "DirectConv8cSpec block_q must be a multiple of 16"
    ab_dtype = "bf16" if p.dtype == "bf16" else "f16"
    if not target.mma.has_shape(
        a_dtype=ab_dtype, b_dtype=ab_dtype, c_dtype="fp32", m=16, n=16, k=16
    ):
        return False, f"missing 16x16x16 {ab_dtype} MFMA atom on {arch}"
    return True, "ok"


def build_direct_conv_8c(spec: DirectConv8cSpec, arch: str = "gfx950") -> KernelDef:
    """Build the IR for one direct conv 8c kernel instance.

    Kernel structure:
      - ``BLOCK_Q = 16``, ``BLOCK_GROUPS`` waves per block, each wave owns
        one convolution group (cpg = kpg = 8).
      - Inner MFMA atom: ``mfma_f32_16x16x16_f16``.  Since cpg=8 < K=16, two
        filter columns (s=0 and s=1) are folded into the K=16 dimension; the
        residual s=2 column is handled as a zero-padded K=16 atom (same trick
        as the fold_k32 S=2 residual in the 16c kernel, but at K=16 scale).
      - Output M rows 8..15 are wasted (kpg=8 < M=16); only c4 in {0,1}
        produce valid k_out addresses and are stored.
      - LDS double-buffered (same ping-pong scheme as the 16c kernel).

    MFMA lane layout (``mfma_f32_16x16x16_f16``, wave64):
      ``q_in_lane = lane % 16``:
        - A operand: M row (k_out within group, valid 0..7; rows 8..15 → zeros)
        - B operand: N column (output W position)
        - C output:  N column
      ``c4 = lane // 16``:
        - A/B operand: K block (c4 in {0,1} → s=0 channels; c4 in {2,3} → s=1 channels)
        - C output: selects 4 consecutive M rows (c4*4 .. c4*4+3); only c4 in {0,1}
          correspond to valid k_out (0..3 and 4..7); c4 in {2,3} are never stored.
    """
    spec.validate()
    ok, why = is_valid_spec_8c(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid direct_conv_8c spec for {arch}: {why}")
    p = spec.problem
    io_type = _io_type(p.dtype)

    BLOCK_Q = spec.block_q
    BLOCK_GROUPS = spec.block_groups
    WAVE = spec.wave_size
    THREADS = spec.threads_per_block
    Ho = p.Ho
    Wo = p.Wo
    LDS_W = (BLOCK_Q - 1) * p.stride + p.KW
    # LDS row: (LDS_W positions) × (BLOCK_GROUPS groups) × (cpg=8 channels)
    LDS_ROW_FP16 = LDS_W * BLOCK_GROUPS * p.cpg
    LOAD_VEC = 4
    NUM_VEC4 = LDS_ROW_FP16 // LOAD_VEC
    PASSES = (NUM_VEC4 + THREADS - 1) // THREADS

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = THREADS

    A = b.param("A", PtrType(io_type, "global"), noalias=True, readonly=True, align=16)
    Bp = b.param("B", PtrType(io_type, "global"), noalias=True, readonly=True, align=16)
    D = b.param("D", PtrType(io_type, "global"), noalias=True, writeonly=True, align=16)
    A_bytes = b.param("A_bytes", I32)
    B_bytes = b.param("B_bytes", I32)
    D_bytes = b.param("D_bytes", I32)

    c0 = b.const_i32(0)
    c_wave = b.const_i32(WAVE)
    c_BG = b.const_i32(BLOCK_GROUPS)
    c_BQ = b.const_i32(BLOCK_Q)
    c_cpg = b.const_i32(p.cpg)
    c_kpg = b.const_i32(p.kpg)
    c_W = b.const_i32(Wo)
    c_BG_cpg = b.const_i32(BLOCK_GROUPS * p.cpg)
    c_half_bytes = b.const_i32(2)
    oob_sentinel = b.const_i32((1 << 31) - 1)

    tid = b.thread_id_x()
    wave_id = b.div(tid, c_wave)
    lane = b.mod(tid, c_wave)
    # Lane decomposition: identical to 16c kernel.
    # c4: K-block index (0..3); selects s-position and channel block.
    # q_in_lane: M row (k_out within group) and N column (output W position).
    c4 = b.div(lane, b.const_i32(16))
    q_in_lane = b.mod(lane, b.const_i32(16))

    # K-fold mapping (mirrors fold_k32 in 16c but at K=16 scale):
    #   c4 in {0,1} → s=0, ch_block = (c4 % 2) * 4 ∈ {0, 4}
    #   c4 in {2,3} → s=1, ch_block = (c4 % 2) * 4 ∈ {0, 4}
    s_lane = b.div(c4, b.const_i32(2))  # 0, 0, 1, 1
    ch_lane = b.mul(b.mod(c4, b.const_i32(2)), b.const_i32(4))  # 0, 4, 0, 4

    # Lanes in the "low half" (c4 ∈ {0,1}) carry valid data for the s=2
    # residual atom; lanes in the "high half" (c4 ∈ {2,3}) are zeroed.
    lane_in_lo_half = b.cmp_lt(c4, b.const_i32(2))

    # Grid: bx=Q-tile, by=group-tile, bz=batch
    bx = b.block_id_x()
    by = b.block_id_y()
    n = b.block_id_z()
    g_tile = by
    g = b.add(b.mul(g_tile, c_BG), wave_id)
    q_tile_start = b.mul(bx, c_BQ)

    # LDS allocation (same over-allocation scheme as 16c to absorb OOB writes).
    lds_total_fp16 = PASSES * THREADS * LOAD_VEC
    A_smem = b.smem_alloc(io_type, [1, lds_total_fp16], name_hint="lds_a")
    B_smem = (
        b.smem_alloc(io_type, [1, lds_total_fp16], name_hint="lds_b")
        if spec.double_buffer
        else A_smem
    )

    a_rsrc = b.buffer_rsrc(A, A_bytes)
    b_rsrc = b.buffer_rsrc(Bp, B_bytes)
    d_rsrc = b.buffer_rsrc(D, D_bytes)

    fp16x4_zero = b.zero_vec(io_type, 4)
    zero_acc = b.zero_vec_f32(4)

    # Weight loads (constant across the H-loop).
    # For the folded main atom (s=0 + s=1 → K=16):
    #   each lane loads 4 f16 at weight[k_out_val, r, s_lane, ch_lane..ch_lane+3].
    # For the s=2 residual (zero-padded K=16 atom):
    #   c4 in {0,1}: load 4 f16 at weight[k_out_val, r, 2, ch_lane..ch_lane+3].
    #   c4 in {2,3}: load zero vec4.
    #
    # k_out_val = g*kpg + q_in_lane, with q_in_lane ∈ {0..15}.
    # For q_in_lane ≥ kpg=8 the address is out of range for this group, but
    # the corresponding C rows (c4 in {2,3}) are never stored, so the garbage
    # load values never reach the output.
    b_desc = TensorDescriptor.naive(
        "B",
        lengths=[p.total_k, p.KH, p.KW, p.cpg],
        coord_names=("k_out", "r", "s", "c"),
    )
    k_out_val = b.add(b.mul(g, c_kpg), q_in_lane)

    weights_main: List[Value] = []  # one per r: s=0+s=1 folded into K=16
    weights_s2: List[Value] = []  # one per r: s=2 zero-padded residual

    for r_const in range(p.KH):
        r_i = b.const_i32(r_const)
        # Main atom: s_lane selects s=0 (c4 ∈ {0,1}) or s=1 (c4 ∈ {2,3}).
        w_off_main, _ = b_desc.offset(b, k_out=k_out_val, r=r_i, s=s_lane, c=ch_lane)
        weights_main.append(
            _buf_load_vN(b, p.dtype, b_rsrc, b.mul(w_off_main, c_half_bytes), c0, 2)
        )
        # Residual s=2: valid only for c4 ∈ {0,1} (lower half of K).
        w_off_s2, _ = b_desc.offset(
            b, k_out=k_out_val, r=r_i, s=b.const_i32(2), c=ch_lane
        )
        w_s2 = _buf_load_vN(b, p.dtype, b_rsrc, b.mul(w_off_s2, c_half_bytes), c0, 2)
        weights_s2.append(b.select(lane_in_lo_half, w_s2, fp16x4_zero))

    # LDS loader (same chunk-decomposition algebra as 16c but with cpg=8).
    # chunk_idx decomposes into (W_lds, group_in_wg, ch_block) where ch_block
    # selects 4 out of 8 channels — hence 2 ch_blocks (not 4 as in 16c).
    chunk_desc = TensorDescriptor.naive(
        "chunk_unmerge",
        lengths=[LDS_W, BLOCK_GROUPS, 2],
        coord_names=("W_lds", "group_in_wg", "ch_block"),
    ).transform(
        unmerge_magic(
            "chunk_idx",
            into=("W_lds", "group_in_wg", "ch_block"),
            dims=[LDS_W, BLOCK_GROUPS, 2],
        ),
    )
    chunk_meta = []
    for pass_idx in range(PASSES):
        chunk_idx = b.add(tid, b.const_i32(pass_idx * THREADS))
        decoded = chunk_desc.unmerge_lower(b, chunk_idx=chunk_idx)
        ch_block = decoded["ch_block"]
        group_in_wg = decoded["group_in_wg"]
        W_lds = decoded["W_lds"]
        in_bounds = b.cmp_lt(chunk_idx, b.const_i32(NUM_VEC4))
        abs_group = b.add(b.mul(g_tile, c_BG), group_in_wg)
        chunk_meta.append(
            {
                "chunk_idx": chunk_idx,
                "ch_block": ch_block,
                "group_in_wg": group_in_wg,
                "W_lds": W_lds,
                "in_bounds": in_bounds,
                "abs_group": abs_group,
            }
        )

    a_desc = TensorDescriptor.naive(
        "A",
        lengths=[p.N, p.H, p.W, p.total_c],
        coord_names=("n", "h", "w", "c"),
    ).transform(
        embed(
            upper=("y_iter",),
            into="h",
            strides=(1,),
            offset=-p.PAD,
            lo=0,
            hi=p.H,
        ),
        embed(
            upper=("q_pos", "W_lds_pos"),
            into="w",
            strides=(p.stride, 1),
            offset=-p.PAD,
            lo=0,
            hi=p.W,
        ),
    )

    c_stride_8c = p.stride

    def issue_dram_load(y_iter_val):
        out = []
        for cm in chunk_meta:
            c_val = b.add(
                b.mul(cm["abs_group"], c_cpg),
                b.mul(cm["ch_block"], b.const_i32(4)),
            )
            a_off_elems, addr_valid = a_desc.offset(
                b,
                n=n,
                y_iter=y_iter_val,
                q_pos=q_tile_start,
                W_lds_pos=cm["W_lds"],
                c=c_val,
            )
            valid = b.land(addr_valid, cm["in_bounds"])
            a_off_bytes = b.mul(a_off_elems, c_half_bytes)
            safe_off = b.select(valid, a_off_bytes, oob_sentinel)
            a_vec = _buf_load_vN(b, p.dtype, a_rsrc, safe_off, c0, 2)
            a_vec = b.select(valid, a_vec, fp16x4_zero)
            lds_idx = b.mul(cm["chunk_idx"], b.const_i32(4))
            out.append((a_vec, lds_idx))
        return out

    def store_to_lds(loads, lds):
        for a_vec, lds_idx in loads:
            b.smem_store_vN(lds, [c0, lds_idx], a_vec, 4)

    q_subtiles = BLOCK_Q // 16

    def lds_read_input_main(q_subtile: int, lds) -> Value:
        """Per-lane <4 x half> read from LDS for the s-folded K=16 main atom.

        Lane ``q_in_lane`` owns output column ``q_subtile*16 + q_in_lane``.
        ``s_lane`` (0 or 1) selects the filter column encoded in the K-fold.
        LDS offset: ``(q_in_lane + q_subtile*16) * stride + s_lane``.
        """
        W_lds_idx = b.add(
            b.mul(
                b.add(q_in_lane, b.const_i32(q_subtile * 16)), b.const_i32(c_stride_8c)
            ),
            s_lane,
        )
        ch_block_idx = b.div(ch_lane, b.const_i32(4))
        lds_idx = b.add(
            b.add(
                b.mul(W_lds_idx, c_BG_cpg),
                b.mul(wave_id, c_cpg),
            ),
            b.mul(ch_block_idx, b.const_i32(4)),
        )
        return b.smem_load_vN(lds, c0, lds_idx, dtype=io_type, n=4)

    def lds_read_input_s2(q_subtile: int, lds) -> Value:
        """Per-lane <4 x half> read from LDS for the s=2 residual.

        LDS offset: ``(q_in_lane + q_subtile*16) * stride + 2``.
        Valid only for c4 ∈ {0,1}; upper K half is zeroed.
        """
        W_lds_idx = b.add(
            b.mul(
                b.add(q_in_lane, b.const_i32(q_subtile * 16)), b.const_i32(c_stride_8c)
            ),
            b.const_i32(2),
        )
        ch_block_idx = b.div(ch_lane, b.const_i32(4))
        lds_idx = b.add(
            b.add(
                b.mul(W_lds_idx, c_BG_cpg),
                b.mul(wave_id, c_cpg),
            ),
            b.mul(ch_block_idx, b.const_i32(4)),
        )
        vec = b.smem_load_vN(lds, c0, lds_idx, dtype=io_type, n=4)
        return b.select(lane_in_lo_half, vec, fp16x4_zero)

    # Prologue: prefetch row 0 into A_smem.
    store_to_lds(issue_dram_load(c0), A_smem)
    b.sync()

    n_iters = p.H + p.KH - 1
    acc_tiles: List[List[Value]] = [
        [zero_acc, zero_acc, zero_acc] for _ in range(q_subtiles)
    ]

    d_desc = TensorDescriptor.naive(
        "D",
        lengths=[p.N, Ho, Wo, p.total_k],
        coord_names=("n", "h", "w", "k"),
    )

    for y in range(n_iters):
        cur = A_smem if (y % 2 == 0 or not spec.double_buffer) else B_smem
        nxt = B_smem if (y % 2 == 0 or not spec.double_buffer) else A_smem

        inputs_main_by_q = [lds_read_input_main(qt, cur) for qt in range(q_subtiles)]
        inputs_s2_by_q = [lds_read_input_s2(qt, cur) for qt in range(q_subtiles)]

        loads_next = None
        if y + 1 < n_iters:
            loads_next = issue_dram_load(b.const_i32(y + 1))

        for qt in range(q_subtiles):
            accs = acc_tiles[qt]
            inp_main = inputs_main_by_q[qt]
            inp_s2 = inputs_s2_by_q[qt]
            for r_const in range(p.KH):
                p_idx = (y - r_const) % p.KH
                acc_in = accs[p_idx]
                # Main atom: s=0 and s=1 folded into K=16.
                acc_in = _mfma(
                    b, p.dtype, "16x16x16", weights_main[r_const], inp_main, acc_in
                )
                # Residual atom: s=2, zero-padded to K=16 (upper lanes carry zeros).
                acc_in = _mfma(
                    b, p.dtype, "16x16x16", weights_s2[r_const], inp_s2, acc_in
                )
                accs[p_idx] = acc_in

        if loads_next is not None:
            if not spec.double_buffer:
                b.sync()
            store_to_lds(loads_next, nxt)
        b.sync()

        p_flush_val = y - (p.KH - 1)
        P_FLUSH = p_flush_val % p.KH
        if 0 <= p_flush_val < p.H and p_flush_val % c_stride_8c == 0:
            ho_row = p_flush_val // c_stride_8c
            for qt in range(q_subtiles):
                acc_to_flush = acc_tiles[qt][P_FLUSH]
                out_q = b.add(b.add(q_tile_start, b.const_i32(qt * 16)), q_in_lane)
                out_q_valid = b.cmp_lt(out_q, c_W)
                # C output layout for mfma_f32_16x16x16_f16 (wave64):
                #   row = c4 * 4 + slot (slot ∈ {0..3}), col = q_in_lane.
                # Valid k_out rows: c4 * 4 ∈ {0,4} (c4 ∈ {0,1}, since kpg=8).
                # c4 ∈ {2,3} → rows 8..15 are out-of-group; never stored.
                k_val = b.add(b.mul(g, c_kpg), b.mul(c4, b.const_i32(4)))
                c4_valid = b.cmp_lt(c4, b.const_i32(p.kpg // 4))
                d_base, _ = d_desc.offset(
                    b,
                    n=n,
                    h=b.const_i32(ho_row),
                    w=out_q,
                    k=k_val,
                )
                d_base_bytes = b.mul(d_base, c_half_bytes)
                store_valid = b.land(out_q_valid, c4_valid)
                safe_d_off = b.select(store_valid, d_base_bytes, oob_sentinel)
                acc_h = _trunc_f32(b, p.dtype, acc_to_flush)
                _buf_store_vN(b, p.dtype, d_rsrc, safe_d_off, c0, acc_h, 2)
        for qt in range(q_subtiles):
            acc_tiles[qt][P_FLUSH] = zero_acc

    return b.kernel


# ---------------------------------------------------------------------------
# 32c kernel — cpg = kpg = 32, mfma_f32_32x32x8_f16, one group per wave
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DirectConv32cSpec:
    """Direct grouped convolution kernel for ``cpg = kpg = 32``.

    Uses ``mfma_f32_32x32x8_f16`` (gfx950 only, wave64).  Each wave processes
    one convolution group:
      M = kpg = 32  — fully covers all output channels per group.
      N = BLOCK_Q   — output W positions per block (default 32).
      K = 8 per atom — cpg=32 channels split across 4 atoms per (r, s) step.

    Lane layout (``mfma_f32_32x32x8_f16``, wave64):
      ``q_in_lane = lane % 32`` — M row (k_out within group) and N column (output W).
      ``k_blk     = lane // 32`` — K-block selector (0 or 1).

    Per (r, s), 4 MFMA calls cover all 32 input channels:
      atom 0: ch = 0..7   (k_blk=0 → ch=0..3, k_blk=1 → ch=4..7)
      atom 1: ch = 8..15  (k_blk=0 → ch=8..11, k_blk=1 → ch=12..15)
      atom 2: ch = 16..23 (k_blk=0 → ch=16..19, k_blk=1 → ch=20..23)
      atom 3: ch = 24..31 (k_blk=0 → ch=24..27, k_blk=1 → ch=28..31)

    C output (16 f32 per lane, wave64 32×32 layout):
      For slot i ∈ {0..15}:
        row = (i // 4) * 8 + (lane // 32) * 4 + (i % 4)
        col = lane % 32
      Slots 0..3 produce rows 0,1,2,3 (k_blk=0) or 4,5,6,7 (k_blk=1) per octant.

    Architecture: gfx942 and gfx950 (``mfma_f32_32x32x8_f16`` is in the MMA
    catalog for both).
    """

    problem: DirectConvProblem
    name: str = "direct_conv_32c"
    block_q: int = 32
    block_groups: int = 4
    wave_size: int = 64
    double_buffer: bool = True

    @property
    def threads_per_block(self) -> int:
        return self.block_groups * self.wave_size

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        p = self.problem
        return kernel_name_join(
            self.name,
            p.short(),
            f"bq{self.block_q}",
            f"bg{self.block_groups}",
            "db" if self.double_buffer else "sb",
            flags={"bf16": p.dtype == "bf16"},
        )

    def validate(self) -> None:
        p = self.problem
        if p.dtype not in ("fp16", "bf16"):
            raise ValueError(f"DirectConv32cSpec: unsupported dtype {p.dtype!r}")
        if p.cpg != 32 or p.kpg != 32:
            raise ValueError(
                f"DirectConv32cSpec expects cpg=kpg=32 (got {p.cpg}, {p.kpg})"
            )
        if p.groups % self.block_groups != 0:
            raise ValueError(
                f"groups {p.groups} not divisible by block_groups {self.block_groups}"
            )
        if self.block_q % 32 != 0:
            raise ValueError("DirectConv32cSpec block_q must be a multiple of 32")


def is_valid_spec_32c(
    spec: DirectConv32cSpec, arch: str = "gfx950"
) -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for a 32c spec on ``arch``.

    The 32c kernel uses ``mfma_f32_32x32x8_f16`` or ``mfma_f32_32x32x8_bf16``,
    which are present in the rocke MMA catalog for both gfx942 and gfx950.
    """
    from rocke.core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)
    p = spec.problem
    if p.dtype not in ("fp16", "bf16"):
        return False, f"unsupported dtype {p.dtype!r}; expected 'fp16' or 'bf16'"
    if p.stride != 1:
        return False, f"stride > 1 is not supported (got {p.stride})"
    if p.cpg != 32 or p.kpg != 32:
        return False, f"DirectConv32cSpec expects cpg=kpg=32 (got {p.cpg}, {p.kpg})"
    if p.groups % spec.block_groups != 0:
        return (
            False,
            f"groups {p.groups} not divisible by block_groups {spec.block_groups}",
        )
    if spec.block_q % 32 != 0:
        return False, "DirectConv32cSpec block_q must be a multiple of 32"
    ab_dtype = "bf16" if p.dtype == "bf16" else "f16"
    if not target.mma.has_shape(
        a_dtype=ab_dtype, b_dtype=ab_dtype, c_dtype="fp32", m=32, n=32, k=8
    ):
        return False, f"missing 32x32x8 {ab_dtype} MFMA atom on {arch}"
    return True, "ok"


def build_direct_conv_32c(spec: DirectConv32cSpec, arch: str = "gfx950") -> KernelDef:
    """Build the IR for one direct conv 32c kernel instance.

    Kernel structure:
      - BLOCK_Q = 32 (N=32 output W positions), BLOCK_GROUPS waves per block,
        each wave owns one convolution group (cpg = kpg = 32).
      - MFMA atom: ``mfma_f32_32x32x8_f16`` — M=32 matches kpg exactly.
      - Per (r, s): 4 consecutive MFMA calls cover all cpg=32 channels in
        K-chunks of 8 (ch_start = 0, 8, 16, 24).
      - C output: 16 f32 per lane (32×32 / 64 lanes).
        For slot i: row = (i//4)*8 + (lane//32)*4 + (i%4), col = lane%32.
        The 16 slots are stored as 8 pairs of consecutive k_out values
        (2 slots per store → 4 halves = 1 dwordx2 store).
      - LDS double-buffered (same ping-pong scheme as 16c).

    MFMA lane layout (``mfma_f32_32x32x8_f16``, wave64):
      ``q_in_lane = lane % 32``: M row and N column simultaneously.
      ``k_blk     = lane // 32``: K-block (0 → ch=0..3, 1 → ch=4..7 within atom).
    """
    spec.validate()
    ok, why = is_valid_spec_32c(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid direct_conv_32c spec for {arch}: {why}")
    p = spec.problem
    io_type = _io_type(p.dtype)

    BLOCK_Q = spec.block_q
    BLOCK_GROUPS = spec.block_groups
    WAVE = spec.wave_size
    THREADS = spec.threads_per_block
    Ho = p.Ho
    Wo = p.Wo
    LDS_W = (BLOCK_Q - 1) * p.stride + p.KW
    LDS_ROW_FP16 = LDS_W * BLOCK_GROUPS * p.cpg
    LOAD_VEC = 4
    NUM_VEC4 = LDS_ROW_FP16 // LOAD_VEC
    PASSES = (NUM_VEC4 + THREADS - 1) // THREADS
    N_CH_BLOCKS = p.cpg // LOAD_VEC  # = 8 for cpg=32

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = THREADS

    A = b.param("A", PtrType(io_type, "global"), noalias=True, readonly=True, align=16)
    Bp = b.param("B", PtrType(io_type, "global"), noalias=True, readonly=True, align=16)
    D = b.param("D", PtrType(io_type, "global"), noalias=True, writeonly=True, align=16)
    A_bytes = b.param("A_bytes", I32)
    B_bytes = b.param("B_bytes", I32)
    D_bytes = b.param("D_bytes", I32)

    c0 = b.const_i32(0)
    c_wave = b.const_i32(WAVE)
    c_BG = b.const_i32(BLOCK_GROUPS)
    c_BQ = b.const_i32(BLOCK_Q)
    c_cpg = b.const_i32(p.cpg)
    c_kpg = b.const_i32(p.kpg)
    c_W = b.const_i32(Wo)
    c_BG_cpg = b.const_i32(BLOCK_GROUPS * p.cpg)
    c_half_bytes = b.const_i32(2)
    oob_sentinel = b.const_i32((1 << 31) - 1)

    tid = b.thread_id_x()
    wave_id = b.div(tid, c_wave)
    lane = b.mod(tid, c_wave)
    # Lane decomposition for mfma_f32_32x32x8_{f16,bf16} (wave64):
    #   q_in_lane = lane % 32 → M row (k_out within group) and N col (output W)
    #   k_blk     = lane // 32 → K-block (0 → ch=0..3, 1 → ch=4..7 within each atom)
    q_in_lane = b.mod(lane, b.const_i32(32))
    k_blk = b.div(lane, b.const_i32(32))

    bx = b.block_id_x()
    by = b.block_id_y()
    n = b.block_id_z()
    g_tile = by
    g = b.add(b.mul(g_tile, c_BG), wave_id)
    q_tile_start = b.mul(bx, c_BQ)

    lds_total_fp16 = PASSES * THREADS * LOAD_VEC
    A_smem = b.smem_alloc(io_type, [1, lds_total_fp16], name_hint="lds_a")
    B_smem = (
        b.smem_alloc(io_type, [1, lds_total_fp16], name_hint="lds_b")
        if spec.double_buffer
        else A_smem
    )

    a_rsrc = b.buffer_rsrc(A, A_bytes)
    b_rsrc = b.buffer_rsrc(Bp, B_bytes)
    d_rsrc = b.buffer_rsrc(D, D_bytes)

    fp16x4_zero = b.zero_vec(io_type, 4)
    zero_acc = b.zero_vec_f32(16)

    # Weight loads: per (r, s, atom_idx), each lane loads 4 elements at
    #   weight[k_out_val, r, s, ch_start + k_blk*4 .. ch_start + k_blk*4 + 3]
    # where ch_start = atom_idx * 8, k_out_val = g*kpg + q_in_lane.
    b_desc = TensorDescriptor.naive(
        "B",
        lengths=[p.total_k, p.KH, p.KW, p.cpg],
        coord_names=("k_out", "r", "s", "c"),
    )
    k_out_val = b.add(b.mul(g, c_kpg), q_in_lane)
    ch_in_atom = b.mul(k_blk, b.const_i32(4))  # 0 or 4 within the 8-ch atom

    weights: List[List[List[Value]]] = []
    for r_const in range(p.KH):
        weights_r = []
        for s_const in range(p.KW):
            weights_rs = []
            for atom_idx in range(4):
                ch_start = atom_idx * 8
                ch_off = b.add(b.const_i32(ch_start), ch_in_atom)
                w_off, _ = b_desc.offset(
                    b,
                    k_out=k_out_val,
                    r=b.const_i32(r_const),
                    s=b.const_i32(s_const),
                    c=ch_off,
                )
                weights_rs.append(
                    _buf_load_vN(b, p.dtype, b_rsrc, b.mul(w_off, c_half_bytes), c0, 2)
                )
            weights_r.append(weights_rs)
        weights.append(weights_r)

    # LDS loader: chunk_idx → (W_lds, group_in_wg, ch_block) with ch_block ∈ {0..7}.
    chunk_desc = TensorDescriptor.naive(
        "chunk_unmerge",
        lengths=[LDS_W, BLOCK_GROUPS, N_CH_BLOCKS],
        coord_names=("W_lds", "group_in_wg", "ch_block"),
    ).transform(
        unmerge_magic(
            "chunk_idx",
            into=("W_lds", "group_in_wg", "ch_block"),
            dims=[LDS_W, BLOCK_GROUPS, N_CH_BLOCKS],
        ),
    )
    chunk_meta = []
    for pass_idx in range(PASSES):
        chunk_idx = b.add(tid, b.const_i32(pass_idx * THREADS))
        decoded = chunk_desc.unmerge_lower(b, chunk_idx=chunk_idx)
        ch_block = decoded["ch_block"]
        group_in_wg = decoded["group_in_wg"]
        W_lds = decoded["W_lds"]
        in_bounds = b.cmp_lt(chunk_idx, b.const_i32(NUM_VEC4))
        abs_group = b.add(b.mul(g_tile, c_BG), group_in_wg)
        chunk_meta.append(
            {
                "chunk_idx": chunk_idx,
                "ch_block": ch_block,
                "group_in_wg": group_in_wg,
                "W_lds": W_lds,
                "in_bounds": in_bounds,
                "abs_group": abs_group,
            }
        )

    a_desc = TensorDescriptor.naive(
        "A",
        lengths=[p.N, p.H, p.W, p.total_c],
        coord_names=("n", "h", "w", "c"),
    ).transform(
        embed(
            upper=("y_iter",),
            into="h",
            strides=(1,),
            offset=-p.PAD,
            lo=0,
            hi=p.H,
        ),
        embed(
            upper=("q_pos", "W_lds_pos"),
            into="w",
            strides=(p.stride, 1),
            offset=-p.PAD,
            lo=0,
            hi=p.W,
        ),
    )

    c_stride_32c = p.stride

    def issue_dram_load(y_iter_val):
        out = []
        for cm in chunk_meta:
            c_val = b.add(
                b.mul(cm["abs_group"], c_cpg),
                b.mul(cm["ch_block"], b.const_i32(4)),
            )
            a_off_elems, addr_valid = a_desc.offset(
                b,
                n=n,
                y_iter=y_iter_val,
                q_pos=q_tile_start,
                W_lds_pos=cm["W_lds"],
                c=c_val,
            )
            valid = b.land(addr_valid, cm["in_bounds"])
            a_off_bytes = b.mul(a_off_elems, c_half_bytes)
            safe_off = b.select(valid, a_off_bytes, oob_sentinel)
            a_vec = _buf_load_vN(b, p.dtype, a_rsrc, safe_off, c0, 2)
            a_vec = b.select(valid, a_vec, fp16x4_zero)
            lds_idx = b.mul(cm["chunk_idx"], b.const_i32(4))
            out.append((a_vec, lds_idx))
        return out

    def store_to_lds(loads, lds):
        for a_vec, lds_idx in loads:
            b.smem_store_vN(lds, [c0, lds_idx], a_vec, 4)

    def lds_read_input(q_subtile: int, s_const: int, atom_idx: int, lds) -> Value:
        """Per-lane <4 x half/bfloat> read from LDS for one K-atom of the 32c kernel.

        Lane ``q_in_lane`` owns output column ``q_subtile*32 + q_in_lane``.
        LDS offset: ``(q_in_lane + q_subtile*32) * stride + s_const``.
        atom_idx selects ch_start = atom_idx*8; k_blk selects the 4-ch half.
        """
        ch_start = atom_idx * 8
        W_lds_idx = b.add(
            b.mul(
                b.add(q_in_lane, b.const_i32(q_subtile * 32)), b.const_i32(c_stride_32c)
            ),
            b.const_i32(s_const),
        )
        ch_off = b.add(b.const_i32(ch_start), b.mul(k_blk, b.const_i32(4)))
        lds_idx = b.add(
            b.add(
                b.mul(W_lds_idx, c_BG_cpg),
                b.mul(wave_id, c_cpg),
            ),
            ch_off,
        )
        return b.smem_load_vN(lds, c0, lds_idx, dtype=io_type, n=4)

    q_subtiles = BLOCK_Q // 32

    store_to_lds(issue_dram_load(c0), A_smem)
    b.sync()

    n_iters = p.H + p.KH - 1
    acc_tiles: List[List[Value]] = [
        [zero_acc, zero_acc, zero_acc] for _ in range(q_subtiles)
    ]

    d_desc = TensorDescriptor.naive(
        "D",
        lengths=[p.N, Ho, Wo, p.total_k],
        coord_names=("n", "h", "w", "k"),
    )

    for y in range(n_iters):
        cur = A_smem if (y % 2 == 0 or not spec.double_buffer) else B_smem
        nxt = B_smem if (y % 2 == 0 or not spec.double_buffer) else A_smem

        inputs_by_q = [
            [
                [lds_read_input(qt, s_const, atom_idx, cur) for atom_idx in range(4)]
                for s_const in range(p.KW)
            ]
            for qt in range(q_subtiles)
        ]

        loads_next = None
        if y + 1 < n_iters:
            loads_next = issue_dram_load(b.const_i32(y + 1))

        for qt in range(q_subtiles):
            accs = acc_tiles[qt]
            for r_const in range(p.KH):
                p_idx = (y - r_const) % p.KH
                acc_in = accs[p_idx]
                for s_const in range(p.KW):
                    for atom_idx in range(4):
                        acc_in = _mfma(
                            b,
                            p.dtype,
                            "32x32x8",
                            weights[r_const][s_const][atom_idx],
                            inputs_by_q[qt][s_const][atom_idx],
                            acc_in,
                        )
                accs[p_idx] = acc_in

        if loads_next is not None:
            if not spec.double_buffer:
                b.sync()
            store_to_lds(loads_next, nxt)
        b.sync()

        p_flush_val = y - (p.KH - 1)
        P_FLUSH = p_flush_val % p.KH
        if 0 <= p_flush_val < p.H and p_flush_val % c_stride_32c == 0:
            ho_row = p_flush_val // c_stride_32c
            for qt in range(q_subtiles):
                acc_to_flush = acc_tiles[qt][P_FLUSH]
                out_q = b.add(b.add(q_tile_start, b.const_i32(qt * 32)), q_in_lane)
                out_q_valid = b.cmp_lt(out_q, c_W)

                # C output layout for mfma_f32_32x32x8_f16 (wave64), 16 slots:
                #   slot i: row = (i//4)*8 + k_blk*4 + (i%4),  col = q_in_lane
                k_base = b.add(b.mul(g, c_kpg), b.mul(k_blk, b.const_i32(4)))
                for octant in range(4):
                    octant_row_base = octant * 8
                    for half in range(2):
                        slot0 = octant * 4 + half * 2
                        slot1 = slot0 + 1
                        row_off = octant_row_base + half * 2
                        k_val = b.add(k_base, b.const_i32(row_off))
                        d_base, _ = d_desc.offset(
                            b,
                            n=n,
                            h=b.const_i32(ho_row),
                            w=out_q,
                            k=k_val,
                        )
                        d_base_bytes = b.mul(d_base, c_half_bytes)
                        safe_d_off = b.select(out_q_valid, d_base_bytes, oob_sentinel)
                        e0 = b.vec_extract(acc_to_flush, slot0)
                        e1 = b.vec_extract(acc_to_flush, slot1)
                        pair_acc = b.vec_pack([e0, e1], F32)
                        pair_h = _trunc_f32(b, p.dtype, pair_acc)
                        _buf_store_vN(b, p.dtype, d_rsrc, safe_d_off, c0, pair_h, 1)
        for qt in range(q_subtiles):
            acc_tiles[qt][P_FLUSH] = zero_acc

    return b.kernel


# ---------------------------------------------------------------------------
# Generic parametric kernel — any cpg that is a multiple of 4
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DirectConvSpec:
    """Direct grouped convolution kernel for any ``cpg`` that is a multiple of 4.

    Uses ``mfma_f32_16x16x16_f16`` (M=16, N=16, K=16) on gfx942 and gfx950.
    The inner K-reduction runs as a runtime ``scf.for`` loop over
    ``N_K_ATOMS = ceil(cpg / 16)`` atoms, each covering 16 input channels.
    This allows a single kernel builder to handle cpg values beyond the
    fixed {4, 8, 16, 32} set of the specialised variants.

    For ``cpg < 16`` (cpg ∈ {4, 8, 12}), ``N_K_ATOMS = 1`` and lanes whose
    channel offset exceeds ``cpg`` are zero-masked before the MFMA call so
    the arithmetic is correct at the cost of partial K utilisation.

    Block geometry:
      - One wave per convolution group (``WAVE = 64``).
      - ``block_groups`` waves share one workgroup.
      - ``block_q`` output W positions per workgroup (multiple of 16).

    Accumulator layout:
      - ``q_subtiles = block_q // 16`` output W sub-tiles.
      - ``N_M_TILES = ceil(kpg / 16)`` M-tiles per sub-tile.
      - ``KH`` circular pipeline slots per (q_subtile, M-tile).
      - Total: ``q_subtiles × N_M_TILES × KH`` accumulators, each ``<4 × f32>``.
    """

    problem: DirectConvProblem
    name: str = "direct_conv"
    block_q: int = 16
    block_groups: int = 8
    wave_size: int = 64
    double_buffer: bool = True
    block_h: int = 0  # 0 = no H-tiling; > 0 = tile H (rows per block)
    waves_q: int = 1  # waves along W: all waves process the same Q-tile cooperatively
    waves_k: int = 1  # waves along K-reduction: each wave preloads its K-atom slice
    runtime_k_loop: bool = (
        False  # True = runtime scf.for over K-atoms (low peak VGPR usage)
    )
    persistent_grid: bool = False  # True = always 256 blocks, each iterates over cells
    fold_k32: bool = (
        False  # True = use mfma_f32_16x16x32_f16 (2× fewer MFMAs) + LOAD_VEC=8
    )

    @property
    def threads_per_block(self) -> int:
        return self.block_groups * self.waves_q * self.waves_k * self.wave_size

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        p = self.problem
        bh_flag = f"bh{self.block_h}" if self.block_h > 0 else ""
        wq_flag = f"wq{self.waves_q}" if self.waves_q > 1 else ""
        wk_flag = f"wk{self.waves_k}" if self.waves_k > 1 else ""
        rk_flag = "rk" if self.runtime_k_loop else ""
        k32_flag = "k32" if self.fold_k32 else ""
        bf16_flag = "bf16" if p.dtype == "bf16" else ""
        return kernel_name_join(
            self.name,
            p.short(),
            f"bq{self.block_q}",
            f"bg{self.block_groups}",
            "db" if self.double_buffer else "sb",
            bh_flag,
            wq_flag,
            wk_flag,
            rk_flag,
            k32_flag,
            bf16_flag,
        )

    def validate(self) -> None:
        p = self.problem
        if p.dtype not in ("fp16", "bf16"):
            raise ValueError(f"DirectConvSpec: unsupported dtype {p.dtype!r}")
        if p.cpg % 4 != 0 or p.cpg < 4:
            raise ValueError(
                f"DirectConvSpec requires cpg to be a positive multiple of 4 "
                f"(got cpg={p.cpg})"
            )
        if p.kpg < 1:
            raise ValueError(f"DirectConvSpec requires kpg >= 1 (got {p.kpg})")
        if p.groups % self.block_groups != 0:
            raise ValueError(
                f"groups {p.groups} not divisible by block_groups {self.block_groups}"
            )
        if self.block_q % 16 != 0:
            raise ValueError("DirectConvSpec block_q must be a multiple of 16")
        if self.block_h < 0:
            raise ValueError("DirectConvSpec block_h must be >= 0")
        if self.fold_k32 and p.cpg % 32 != 0:
            raise ValueError(
                f"DirectConvSpec fold_k32 requires cpg to be a multiple of 32 (got {p.cpg})"
            )
        if self.block_groups > 1 and self.waves_k > 1:
            raise ValueError(
                f"block_groups={self.block_groups} > 1 combined with waves_k={self.waves_k} > 1 "
                f"is not supported: the LDS reduction row index does not account for "
                f"wave_group_idx, causing cross-group partial-sum corruption"
            )
        N_K_ATOMS = (p.cpg + 15) // 16
        if N_K_ATOMS % self.waves_k != 0:
            raise ValueError(
                f"N_K_ATOMS={N_K_ATOMS} (ceil(cpg/16)) must be divisible by waves_k={self.waves_k}"
            )
        if self.block_q // self.waves_q < 16:
            raise ValueError(
                f"block_q//waves_q must be >= 16 (got {self.block_q}//{self.waves_q}={self.block_q//self.waves_q})"
            )


def is_valid_spec(spec: "DirectConvSpec", arch: str = "gfx950") -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for a :class:`DirectConvSpec` on ``arch``.

    Checks cpg divisibility, block geometry, and MFMA atom availability
    (``mfma_f32_16x16x16_f16`` must be present on the target).

    ``cpg`` and ``kpg`` need not be equal — this enables the transposed-fprop
    pass used by the dgrad pipeline (where the transposed weight tensor has
    cpg_new = kpg_orig and kpg_new = cpg_orig, which differ for non-square
    channel counts).
    """
    from rocke.core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)

    p = spec.problem
    if p.dtype not in ("fp16", "bf16"):
        return False, f"unsupported dtype {p.dtype!r}; expected 'fp16' or 'bf16'"
    if p.stride != 1:
        return False, f"stride > 1 is not supported (got {p.stride})"
    if p.cpg % 4 != 0 or p.cpg < 4:
        return False, f"cpg must be a positive multiple of 4 (got {p.cpg})"
    if p.kpg < 1:
        return False, f"kpg must be >= 1 (got {p.kpg})"
    if p.groups % spec.block_groups != 0:
        return (
            False,
            f"groups {p.groups} not divisible by block_groups {spec.block_groups}",
        )
    if spec.block_q % 16 != 0:
        return False, "block_q must be a multiple of 16"
    ab_dtype = "bf16" if p.dtype == "bf16" else "f16"
    if not target.mma.has_shape(
        a_dtype=ab_dtype, b_dtype=ab_dtype, c_dtype="fp32", m=16, n=16, k=16
    ):
        return False, f"missing mfma_f32_16x16x16_{ab_dtype} on {arch}"
    if spec.fold_k32 and not target.mma.has_shape(
        a_dtype=ab_dtype, b_dtype=ab_dtype, c_dtype="fp32", m=16, n=16, k=32
    ):
        return False, f"fold_k32 requires mfma_f32_16x16x32_{ab_dtype} on {arch}"
    return True, "ok"


def build_direct_conv(spec: "DirectConvSpec", arch: str = "gfx950") -> KernelDef:
    """Build the IR for a parametric direct grouped convolution kernel.

    Supports any ``cpg`` that is a positive multiple of 4.  The inner reduction
    over input channels runs as a runtime ``scf.for`` loop over
    ``N_K_ATOMS = ceil(cpg / 16)`` atoms (each covering 16 channels).
    The filter-column loop (KW) is also a runtime ``scf.for`` loop.

    Kernel structure (streaming row-by-row pipeline):
      H-loop (Python-level unroll, H + KH - 1 iterations):
        1. Load input row into LDS via chunk-based DRAM loader.
        2. For each (q_subtile, r):
             for s in [0, KW):              # runtime scf.for
               for atom in [0, N_K_ATOMS): # runtime scf.for with carried acc
                 w = B[g*kpg + m*16 + q_in_lane, r, s, atom*16 + c4*4]
                 x = LDS[q+s, group, atom*16 + c4*4]
                 acc = mfma_f32_16x16x16_f16(w, x, acc)
        3. Flush accumulator to D when p_flush = y - (KH-1) is valid.
        4. Reset flushed slot to zero.

    For ``cpg < 16``: N_K_ATOMS = 1; lanes where c4*4 >= cpg are zero-masked.
    For ``kpg % 16 != 0``: excess M-tiles are suppressed by Python-level guards.
    """
    spec.validate()
    ok, why = is_valid_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid DirectConvSpec for {arch}: {why}")

    p = spec.problem
    io_type = _io_type(p.dtype)

    BLOCK_Q = spec.block_q
    BLOCK_GROUPS = spec.block_groups
    WAVES_Q = spec.waves_q  # waves along W (more LDS loading threads)
    WAVES_K = spec.waves_k  # waves along K-reduction (preload per wave)
    WAVE = spec.wave_size
    THREADS = spec.threads_per_block  # = BLOCK_GROUPS * WAVES_Q * WAVES_K * WAVE
    Ho = p.Ho
    Wo = p.Wo

    # fold_k32: use mfma_f32_16x16x32_f16 — 2× fewer K-atoms, 8 halves per atom operand.
    FOLD_K32 = spec.fold_k32
    K_ATOM_SIZE = 32 if FOLD_K32 else 16  # K-channels per atom
    LOAD_VEC = 8 if FOLD_K32 else 4  # halves per LDS chunk (8 = dwordx4)
    # For fold_k32=True cpg must be divisible by K_ATOM_SIZE (validated); use floor div.
    # For fold_k32=False use ceil to support partial K-atoms (e.g. cpg=24 → 2 atoms of 16,
    # second atom covers only channels 16..23 and is OOB-masked in the inner loop).
    if FOLD_K32:
        N_K_ATOMS = p.cpg // K_ATOM_SIZE
    else:
        N_K_ATOMS = (
            p.cpg + K_ATOM_SIZE - 1
        ) // K_ATOM_SIZE  # ceil — restores original behaviour
    N_K_LOCAL = N_K_ATOMS // WAVES_K  # K-atoms per wave (each wave preloads this slice)
    N_M_TILES = (p.kpg + 15) // 16  # M-tiles per q_subtile
    N_CH_PER_VEC = LOAD_VEC  # channels per LDS vec load (4 or 8)
    # For LDS chunk loading: use exact chunks (cpg must be divisible by LOAD_VEC).
    # LOAD_VEC=4 always divides standard cpg values; LOAD_VEC=8 requires cpg%8==0.
    N_VECS = p.cpg // N_CH_PER_VEC  # vec-chunks per group (for chunk_desc)
    N_CH4 = p.cpg // 4  # kept for LDS address math (unchanged)
    # Each wave handles block_q // WAVES_Q W positions.
    BLOCK_Q_WAVE = BLOCK_Q // WAVES_Q
    q_subtiles = BLOCK_Q_WAVE // 16  # q_subtiles per wave
    LDS_W = (BLOCK_Q - 1) * p.stride + p.KW

    # LDS loading: all THREADS cooperate to load LDS_W × BLOCK_GROUPS × cpg halves.
    NUM_CHUNKS = LDS_W * BLOCK_GROUPS * N_VECS
    PASSES = (NUM_CHUNKS + THREADS - 1) // THREADS
    lds_total_fp16 = PASSES * THREADS * LOAD_VEC

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = THREADS

    A = b.param("A", PtrType(io_type, "global"), noalias=True, readonly=True, align=16)
    Bp = b.param("B", PtrType(io_type, "global"), noalias=True, readonly=True, align=16)
    D = b.param("D", PtrType(io_type, "global"), noalias=True, writeonly=True, align=16)
    A_bytes = b.param("A_bytes", I32)
    B_bytes = b.param("B_bytes", I32)
    D_bytes = b.param("D_bytes", I32)

    c0 = b.const_i32(0)
    c1 = b.const_i32(1)
    c_wave = b.const_i32(WAVE)
    c_BG = b.const_i32(BLOCK_GROUPS)
    c_BQ = b.const_i32(BLOCK_Q)
    c_cpg = b.const_i32(p.cpg)
    c_kpg = b.const_i32(p.kpg)
    c_W = b.const_i32(Wo)
    c_KW = b.const_i32(p.KW)
    # Runtime K-atom loop now runs only N_K_LOCAL iterations per wave (its slice).
    c_N_K_LOCAL = b.const_i32(N_K_LOCAL)
    c_BG_cpg = b.const_i32(BLOCK_GROUPS * p.cpg)
    c_half_bytes = b.const_i32(2)
    oob_sentinel = b.const_i32((1 << 31) - 1)

    fp16x4_zero = b.zero_vec(io_type, 4)
    zero_acc = b.zero_vec_f32(4)  # mfma_f32_16x16x16_{f16,bf16}: 4 f32 per lane

    tid = b.thread_id_x()
    # Wave decomposition: wave_id encodes (group, waves_q, waves_k) as:
    #   wave_id = (group_idx * WAVES_Q * WAVES_K) + (wq * WAVES_K) + wk
    waves_per_group = WAVES_Q * WAVES_K
    wave_id_full = b.div(tid, c_wave)
    wave_id_in_group = b.mod(wave_id_full, b.const_i32(waves_per_group))
    wave_group_idx = b.div(
        wave_id_full, b.const_i32(waves_per_group)
    )  # group within block
    # Q-wave index and K-wave index within the group's wave block.
    wave_id_q = b.div(wave_id_in_group, b.const_i32(WAVES_K))
    wave_id_k = b.mod(wave_id_in_group, b.const_i32(WAVES_K))
    # K-atom base for this wave's slice: wave_id_k * N_K_LOCAL K-atoms.
    k_atom_base = b.mul(wave_id_k, b.const_i32(N_K_LOCAL))

    lane = b.mod(tid, c_wave)
    # mfma_f32_16x16x16_f16 (wave64) lane layout:
    #   c4        = lane // 16 → K-block in A/B; 4 M-rows in C (c4*4 .. c4*4+3)
    #   q_in_lane = lane % 16  → N column in B/C (output W position within tile)
    c4 = b.div(lane, b.const_i32(16))
    q_in_lane = b.mod(lane, b.const_i32(16))

    bx = b.block_id_x()
    by = b.block_id_y()
    bz = b.block_id_z()

    BLOCK_H = spec.block_h
    PERSISTENT = spec.persistent_grid

    # ---- Persistent grid: decode XCD and workgroup positions ----
    # 256 = 8 XCDs × 32 blocks per XCD.
    # Each block processes multiple cells in a runtime loop, keeping weights
    # in registers across all cells (weights are the SAME for groups=1).
    NUM_XCD = 8
    BLOCKS_PER_XCD = 32
    TOTAL_PERSISTENT = NUM_XCD * BLOCKS_PER_XCD  # = 256

    if PERSISTENT:
        assert BLOCK_H > 0, "persistent_grid requires block_h > 0"
        n_h_tiles = (p.H + BLOCK_H - 1) // BLOCK_H
        q_tiles_p = (p.Wo + BLOCK_Q - 1) // BLOCK_Q
        g_tiles_p = p.groups // BLOCK_GROUPS
        n_cells_p = p.N * n_h_tiles * q_tiles_p * g_tiles_p  # total cells

        # XCD-aware cell assignment.
        cells_per_xcd = (n_cells_p + NUM_XCD - 1) // NUM_XCD
        rounds_per_block = (cells_per_xcd + BLOCKS_PER_XCD - 1) // BLOCKS_PER_XCD

        xcd_id = b.mod(bx, b.const_i32(NUM_XCD))
        wg_in_xcd = b.div(bx, b.const_i32(NUM_XCD))
        xcd_start = b.mul(xcd_id, b.const_i32(cells_per_xcd))

        # Placeholders — actual values set inside cell loop body
        n = b.const_i32(0)
        h_tile_start = b.const_i32(0)
        q_tile_start_cell = b.const_i32(0)
        g_tile_cell = b.const_i32(0)
    else:
        # H-tiling: block_h > 0 encodes (n, h_tile) in bz = n*n_h_tiles + h_tile.
        if BLOCK_H > 0:
            n_h_tiles = (p.H + BLOCK_H - 1) // BLOCK_H
            c_n_h_tiles = b.const_i32(n_h_tiles)
            n = b.div(bz, c_n_h_tiles)
            h_tile_idx = b.mod(bz, c_n_h_tiles)
            h_tile_start = b.mul(h_tile_idx, b.const_i32(BLOCK_H))
        else:
            n = bz
            n_h_tiles = 1
            h_tile_start = None

    if not PERSISTENT:
        g_tile = by

    # wave_group_idx: which group within block_groups this wave cluster belongs to.
    if PERSISTENT:
        g = b.const_i32(0)  # updated inside cell loop
    else:
        g = b.add(b.mul(g_tile, c_BG), wave_group_idx)
    # Q-tile: block-level base + wave_id_q sub-offset (waves_q partitions W).
    block_q_start = b.mul(bx, c_BQ)
    q_tile_start = b.add(block_q_start, b.mul(wave_id_q, b.const_i32(BLOCK_Q_WAVE)))
    # LDS loader uses the full block Q range (all waves cooperate on same LDS row).
    q_tile_start_lds = block_q_start

    A_smem = b.smem_alloc(io_type, [1, lds_total_fp16], name_hint="lds_a")
    B_smem = (
        b.smem_alloc(io_type, [1, lds_total_fp16], name_hint="lds_b")
        if spec.double_buffer
        else A_smem
    )

    # LDS reduction buffer for waves_k > 1:
    # 2D shape [WAVES_Q * WAVES_K, WAVE * 4] f32:
    #   row = wave_id_q * WAVES_K + wave_id_k (unique per (wq, wk) pair)
    #   col = lane * 4 (each lane's 4 f32 are contiguous)
    # Each (wave_id_q, wave_id_k) wave writes to its own unique row so that waves
    # with different wave_id_q but the same wave_id_k don't overwrite each other.
    # The "master" wave for W-subtile wq (wave_id_q=wq, wave_id_k=0) then reads
    # rows [wq*WAVES_K .. wq*WAVES_K + WAVES_K - 1] and sums them.
    N_RED_ROWS = WAVES_Q * WAVES_K
    if WAVES_K > 1:
        red_lds = b.smem_alloc_f32([N_RED_ROWS, WAVE * 4], name_hint="red_lds")
        # Row index for this wave: unique per (wave_id_q, wave_id_k) pair.
        lds_row_idx = b.add(b.mul(wave_id_q, b.const_i32(WAVES_K)), wave_id_k)

    a_rsrc = b.buffer_rsrc(A, A_bytes)
    b_rsrc = b.buffer_rsrc(Bp, B_bytes)
    d_rsrc = b.buffer_rsrc(D, D_bytes)

    # A[N, H, W, total_c] NHWC with h and w embeds for padding and boundary.
    a_desc = TensorDescriptor.naive(
        "A",
        lengths=[p.N, p.H, p.W, p.total_c],
        coord_names=("n", "h", "w", "c"),
    ).transform(
        embed(
            upper=("y_iter",),
            into="h",
            strides=(1,),
            offset=-p.PAD,
            lo=0,
            hi=p.H,
        ),
        embed(
            upper=("q_pos", "W_lds_pos"),
            into="w",
            strides=(p.stride, 1),
            offset=-p.PAD,
            lo=0,
            hi=p.W,
        ),
    )
    c_stride_gen = p.stride

    b_desc = TensorDescriptor.naive(
        "B",
        lengths=[p.total_k, p.KH, p.KW, p.cpg],
        coord_names=("k_out", "r", "s", "c"),
    )

    d_desc = TensorDescriptor.naive(
        "D",
        lengths=[p.N, Ho, Wo, p.total_k],
        coord_names=("n", "h", "w", "k"),
    )

    # LDS chunk loader: chunk_idx → (W_lds, group_in_wg, ch_block).
    # ch_block selects LOAD_VEC (4 or 8) channels within the group.
    chunk_desc = TensorDescriptor.naive(
        "chunk_unmerge",
        lengths=[LDS_W, BLOCK_GROUPS, N_VECS],
        coord_names=("W_lds", "group_in_wg", "ch_block"),
    ).transform(
        unmerge_magic(
            "chunk_idx",
            into=("W_lds", "group_in_wg", "ch_block"),
            dims=[LDS_W, BLOCK_GROUPS, N_VECS],
        ),
    )

    chunk_meta = []
    for pass_idx in range(PASSES):
        chunk_idx = b.add(tid, b.const_i32(pass_idx * THREADS))
        decoded = chunk_desc.unmerge_lower(b, chunk_idx=chunk_idx)
        in_bounds = b.cmp_lt(chunk_idx, b.const_i32(NUM_CHUNKS))
        chunk_meta.append(
            {
                "chunk_idx": chunk_idx,
                "ch_block": decoded["ch_block"],
                "W_lds": decoded["W_lds"],
                "in_bounds": in_bounds,
                "group_in_wg": decoded["group_in_wg"],
            }
        )

    def issue_dram_load(y_iter_val: Value, g_tile_val=None):
        # g_tile_val: for persistent grid, the per-cell decoded g_tile (pg_gt_v);
        # for non-persistent, None (uses the static by value).
        _g_tile = g_tile_val if g_tile_val is not None else by
        out = []
        for cm in chunk_meta:
            abs_group = b.add(b.mul(_g_tile, c_BG), cm["group_in_wg"])
            c_val = b.add(
                b.mul(abs_group, c_cpg),
                b.mul(
                    cm["ch_block"], b.const_i32(LOAD_VEC)
                ),  # LOAD_VEC halves per chunk
            )
            # LDS loading uses the full block Q range (q_tile_start_lds) so that
            # all waves_q × waves_k waves cooperate on loading the same LDS row.
            a_off, addr_valid = a_desc.offset(
                b,
                n=n,
                y_iter=y_iter_val,
                q_pos=q_tile_start_lds,
                W_lds_pos=cm["W_lds"],
                c=c_val,
            )
            valid = b.land(addr_valid, cm["in_bounds"])
            safe_off = b.select(valid, b.mul(a_off, c_half_bytes), oob_sentinel)
            # LOAD_VEC=4 → dwordx2 (4 elements); LOAD_VEC=8 → dwordx4 (8 elements).
            _n_dwords = LOAD_VEC // 2  # dwords = elements / 2
            a_vec = _buf_load_vN(b, p.dtype, a_rsrc, safe_off, c0, _n_dwords)
            _zero_vec = b.zero_vec(io_type, LOAD_VEC)
            a_vec = b.select(valid, a_vec, _zero_vec)
            lds_idx = b.mul(cm["chunk_idx"], b.const_i32(LOAD_VEC))
            out.append((a_vec, lds_idx))
        return out

    def store_to_lds(loads, lds) -> None:
        for a_vec, lds_idx in loads:
            b.smem_store_vN(lds, [c0, lds_idx], a_vec, LOAD_VEC)

    # ---- Persistent cell loop (when persistent_grid=True) ----
    # Each of 256 blocks iterates over its assigned cells: (n, h_tile, q_tile).
    # Weights (preloaded below) are valid for ALL cells → loaded ONCE per block.
    # The H-streaming LDS loop runs once per cell (inside cell loop body).
    if PERSISTENT:
        # Carry cell_idx as meaningful state so the loop isn't optimized away.
        cell_loop = b.scf_for_iter(
            c0,
            b.const_i32(rounds_per_block),
            c1,
            [
                ("pg_cell_idx_carry", xcd_start)
            ],  # starts at xcd_start, updated per round
            iv_name="pg_round",
            elide_trailing_barrier=False,
        )
        # Enter the cell loop body (will be exited with b.scf_yield at end).
        _pg_ctx = cell_loop.__enter__()
        pg_round_iv, (pg_prev_cell,) = _pg_ctx

        # Decode cell index → (g_tile, n, h_tile, q_tile)
        cell_idx = b.add(
            xcd_start, b.add(wg_in_xcd, b.mul(pg_round_iv, b.const_i32(BLOCKS_PER_XCD)))
        )
        pg_in_bounds = b.cmp_lt(cell_idx, b.const_i32(n_cells_p))

        c_qt_p = b.const_i32(q_tiles_p)
        c_nht = b.const_i32(n_h_tiles)
        c_nhq = b.const_i32(n_h_tiles * q_tiles_p)
        c_nhqg = b.const_i32(n_h_tiles * q_tiles_p * g_tiles_p)

        pg_g_tile = b.div(cell_idx, c_nhq)  # groups axis (outermost)
        pg_rem1 = b.mod(cell_idx, c_nhq)
        pg_n = b.div(pg_rem1, c_nhq)  # actually n is here
        # Correct decode: cell = n*n_h_tiles*q_tiles + h_tile*q_tiles + q_tile
        pg_n_v = b.div(cell_idx, b.const_i32(n_h_tiles * q_tiles_p * g_tiles_p))
        pg_rem_n = b.mod(cell_idx, b.const_i32(n_h_tiles * q_tiles_p * g_tiles_p))
        pg_gt_v = b.div(pg_rem_n, b.const_i32(n_h_tiles * q_tiles_p))
        pg_rem_gt = b.mod(pg_rem_n, b.const_i32(n_h_tiles * q_tiles_p))
        pg_ht_v = b.div(pg_rem_gt, c_qt_p)
        pg_qt_v = b.mod(pg_rem_gt, c_qt_p)

        # For OOB rounds: use cell 0 coordinates (produces zeros, guarded by store_ok).
        n = b.select(pg_in_bounds, pg_n_v, c0)
        h_tile_start = b.select(pg_in_bounds, b.mul(pg_ht_v, b.const_i32(BLOCK_H)), c0)
        g = b.select(pg_in_bounds, b.add(b.mul(pg_gt_v, c_BG), wave_group_idx), c0)
        q_tile_start = b.select(
            pg_in_bounds,
            b.add(b.mul(pg_qt_v, c_BQ), b.mul(wave_id_q, b.const_i32(BLOCK_Q_WAVE))),
            c0,
        )
        q_tile_start_lds = b.select(pg_in_bounds, b.mul(pg_qt_v, c_BQ), c0)
        # pg_in_bounds is ANDed into every output store via the persistent flush guard below.

    # Prologue: zero-fill LDS for the first (padded) row of this cell/tile.
    # For persistent grid, pass the per-cell g_tile so abs_group is correct.
    _load_g_tile = pg_gt_v if PERSISTENT else None
    prologue_y = c0 if BLOCK_H == 0 else h_tile_start
    store_to_lds(issue_dram_load(prologue_y, g_tile_val=_load_g_tile), A_smem)
    b.sync()

    # acc_tiles[qt][m][slot]: <4 x f32> per (q_subtile, M-tile, pipeline slot).
    acc_tiles: List[List[List[Value]]] = [
        [[zero_acc] * p.KH for _ in range(N_M_TILES)] for _ in range(q_subtiles)
    ]

    # ---- Weight preloading (WAVES_K > 1 path) --------------------------------
    # When waves_k > 1 each wave handles only N_K_LOCAL K-atoms. Preloading all
    # weights into registers before the H-loop eliminates n_iters × KH × KW × N_K_LOCAL
    # weight DRAM loads from inside the loop — the dominant bottleneck.
    #
    # The B parameter carries W_coa (coalesced reorganized format from a second-pass
    # reorganize kernel). W_coa layout: [groups, KH, KW, N_K_ATOMS, N_M_TILES, 64, 4].
    # For block_idx = r*KW*N_K_ATOMS*N_M_TILES + s*N_K_ATOMS*N_M_TILES + atom*N_M_TILES + m
    # and lane_id = c4*16+q_in_lane:
    #   W_coa[block_idx][lane_id][e] = W_T[m*16+q_in_lane, r', s', atom*16+c4*4+e]
    # Load: elem_off = block_idx * 64 * 4 + lane_id * 4 → stride 4 between lanes = COALESCED!
    preloaded_w: dict = {}
    if WAVES_K > 1:
        lane_id_pw = b.mod(tid, b.const_i32(WAVE))  # = c4*16 + q_in_lane
        # W_coa block size: WAVE × LOAD_VEC elements per (r,s,atom,m) block.
        # fold_k32: LOAD_VEC=8 → 64×8=512 elements; normal: 64×4=256 elements.
        _pw_elems_per_block = WAVE * LOAD_VEC
        c_block_sz = b.const_i32(_pw_elems_per_block)
        _pw_n_dwords = LOAD_VEC // 2  # for buffer_load_vN_f16 n parameter
        # W_coa layout: [groups, KH, KW, N_K_ATOMS, N_M_TILES, WAVE, LOAD_VEC]
        # group-level block offset uses the absolute group index g (not the g_tile)
        # so that each wave-group loads its own group's filters when block_groups > 1.
        _pw_blocks_per_group = p.KH * p.KW * N_K_ATOMS * N_M_TILES
        _pw_g_abs = g if not PERSISTENT else b.const_i32(0)
        _pw_group_base = b.mul(_pw_g_abs, b.const_i32(_pw_blocks_per_group))
        for r_const in range(p.KH):
            for s_const in range(p.KW):
                for local_atom in range(N_K_LOCAL):
                    atom_global_val = b.add(k_atom_base, b.const_i32(local_atom))
                    rs_base_pw = b.add(
                        _pw_group_base,
                        b.add(
                            b.mul(atom_global_val, b.const_i32(N_M_TILES)),
                            b.const_i32(
                                (r_const * p.KW * N_K_ATOMS + s_const * N_K_ATOMS)
                                * N_M_TILES
                            ),
                        ),
                    )
                    for m in range(N_M_TILES):
                        if m * 16 >= p.kpg:
                            preloaded_w[(r_const, s_const, local_atom, m)] = None
                            continue
                        block_idx = b.add(rs_base_pw, b.const_i32(m))
                        # Coalesced: elem_off = block_idx * block_sz + lane_id * LOAD_VEC
                        elem_off = b.add(
                            b.mul(block_idx, c_block_sz),
                            b.mul(lane_id_pw, b.const_i32(LOAD_VEC)),
                        )
                        w_frag_pw = _buf_load_vN(
                            b,
                            p.dtype,
                            b_rsrc,
                            b.mul(elem_off, c_half_bytes),
                            c0,
                            _pw_n_dwords,
                        )
                        preloaded_w[(r_const, s_const, local_atom, m)] = w_frag_pw

    # H-loop iteration count.
    # Without H-tiling: iterate all H+KH-1 rows.
    # With H-tiling: iterate BLOCK_H+KH-1 rows for this tile (OOB → embed zeros).
    n_iters = (BLOCK_H + p.KH - 1) if BLOCK_H > 0 else (p.H + p.KH - 1)

    for y_local in range(n_iters):
        # Global y index for this iteration.
        if BLOCK_H > 0:
            # h_tile_start is a runtime Value; y_local is a Python constant.
            y = b.add(h_tile_start, b.const_i32(y_local))
        else:
            y = b.const_i32(y_local)  # static y for no-tile path
        y = y  # alias for readability; now a runtime or static Value
        cur = A_smem if (y_local % 2 == 0 or not spec.double_buffer) else B_smem
        nxt = B_smem if (y_local % 2 == 0 or not spec.double_buffer) else A_smem

        loads_next = None
        if y_local + 1 < n_iters:
            if BLOCK_H > 0:
                next_y = b.add(h_tile_start, b.const_i32(y_local + 1))
            else:
                next_y = b.const_i32(y_local + 1)
            loads_next = issue_dram_load(next_y, g_tile_val=_load_g_tile)

        for qt in range(q_subtiles):
            qt_w_base = qt * 16

            for r_const in range(p.KH):
                p_idx = (y_local - r_const) % p.KH
                r_i = b.const_i32(r_const)

                if WAVES_K > 1:
                    # ---- Preloaded-weight path (Python-unrolled s & atom loops) ----
                    # Weights already in registers; inner loops are fully unrolled here
                    # so there are no runtime loops and no weight DRAM loads.
                    for s_const in range(p.KW):
                        s_val = b.const_i32(s_const)
                        for local_atom in range(N_K_LOCAL):
                            # ch_off for LDS read: atom_global × K_ATOM_SIZE + c4 × (K_ATOM_SIZE//4)
                            # K_ATOM_SIZE=32 for fold_k32 (c4 selects 8-channel blocks)
                            # K_ATOM_SIZE=16 for normal (c4 selects 4-channel blocks)
                            _c4_step = K_ATOM_SIZE // 4  # = 8 (fold_k32) or 4 (normal)
                            ch_off = b.add(
                                b.add(
                                    b.mul(k_atom_base, b.const_i32(K_ATOM_SIZE)),
                                    b.const_i32(local_atom * K_ATOM_SIZE),
                                ),
                                b.mul(c4, b.const_i32(_c4_step)),
                            )
                            # LDS read for this wave's Q-subtile
                            W_lds_idx_pw = b.add(
                                b.mul(
                                    b.add(q_in_lane, b.const_i32(qt_w_base)),
                                    b.const_i32(c_stride_gen),
                                ),
                                s_val,
                            )
                            if WAVES_Q > 1:
                                W_lds_idx_pw = b.add(
                                    b.mul(wave_id_q, b.const_i32(BLOCK_Q_WAVE)),
                                    W_lds_idx_pw,
                                )
                            lds_idx_pw = b.add(
                                b.add(
                                    b.mul(W_lds_idx_pw, c_BG_cpg),
                                    b.mul(wave_group_idx, c_cpg),
                                ),
                                ch_off,
                            )
                            # For fold_k32: load 8 halves from LDS (K=32 atom).
                            x_frag_pw = b.smem_load_vN(
                                cur, c0, lds_idx_pw, dtype=io_type, n=LOAD_VEC
                            )
                            if p.cpg % K_ATOM_SIZE != 0:
                                c4_oob_pw = b.cmp_ge(ch_off, b.const_i32(p.cpg))
                                _zero_pw = b.zero_vec(io_type, LOAD_VEC)
                                x_frag_pw = b.select(c4_oob_pw, _zero_pw, x_frag_pw)

                            for m in range(N_M_TILES):
                                if m * 16 >= p.kpg:
                                    continue
                                w_frag_pw = preloaded_w[
                                    (r_const, s_const, local_atom, m)
                                ]
                                mfma_shape = "16x16x32" if FOLD_K32 else "16x16x16"
                                acc_tiles[qt][m][p_idx] = _mfma(
                                    b,
                                    p.dtype,
                                    mfma_shape,
                                    w_frag_pw,
                                    x_frag_pw,
                                    acc_tiles[qt][m][p_idx],
                                )
                    continue  # skip the runtime s_loop/atom_loop below

                # ---- Runtime K-atom loop path (runtime_k_loop=True) ----
                # Load 1 K-atom at a time from W_coa (coalesced).  Only the
                # current K-atom's weights are live in registers simultaneously,
                # giving ~55 total VGPRs vs ~200 for the preloaded path.  With
                # 256 threads, this allows 4 blocks/CU instead of 1, giving 4×
                # better occupancy and ~4× more throughput.
                if spec.runtime_k_loop:
                    loop_tag_rk = f"y{y_local}_qt{qt}_r{r_const}"
                    # Carry M-tile accumulators through the K-atom loop.
                    k_iter_args_rk = [
                        (f"rk_acc_m{m}_{loop_tag_rk}", acc_tiles[qt][m][p_idx])
                        for m in range(N_M_TILES)
                    ]
                    k_loop_rk = b.scf_for_iter(
                        c0,
                        c_N_K_LOCAL,
                        c1,
                        k_iter_args_rk,
                        iv_name=f"rk_iv_{loop_tag_rk}",
                        elide_trailing_barrier=False,
                    )
                    with k_loop_rk as (k_iv_rk, k_accs_rk):
                        # Effective global K-atom index: k_atom_base + k_iv_rk.
                        k_atom_global = b.add(k_atom_base, k_iv_rk)
                        _rk_c4_step = K_ATOM_SIZE // 4  # 8 for fold_k32, 4 for normal
                        ch_off_rk = b.add(
                            b.mul(k_atom_global, b.const_i32(K_ATOM_SIZE)),
                            b.mul(c4, b.const_i32(_rk_c4_step)),
                        )
                        new_k_accs_rk = list(k_accs_rk)
                        # Python-unrolled s-loop: 1 LDS read + N_M_TILES MFMAs per s.
                        for s_const in range(p.KW):
                            s_val_rk = b.const_i32(s_const)
                            # LDS read for this (s, K-atom).
                            W_lds_idx_rk = b.add(
                                b.mul(
                                    b.add(q_in_lane, b.const_i32(qt_w_base)),
                                    b.const_i32(c_stride_gen),
                                ),
                                s_val_rk,
                            )
                            if WAVES_Q > 1:
                                W_lds_idx_rk = b.add(
                                    b.mul(wave_id_q, b.const_i32(BLOCK_Q_WAVE)),
                                    W_lds_idx_rk,
                                )
                            lds_idx_rk = b.add(
                                b.add(
                                    b.mul(W_lds_idx_rk, c_BG_cpg),
                                    b.mul(wave_group_idx, c_cpg),
                                ),
                                ch_off_rk,
                            )
                            x_frag_rk = b.smem_load_vN(
                                cur, c0, lds_idx_rk, dtype=io_type, n=LOAD_VEC
                            )
                            if p.cpg % K_ATOM_SIZE != 0:
                                c4_oob_rk = b.cmp_ge(ch_off_rk, b.const_i32(p.cpg))
                                _zero_rk = b.zero_vec(io_type, LOAD_VEC)
                                x_frag_rk = b.select(c4_oob_rk, _zero_rk, x_frag_rk)
                            for m in range(N_M_TILES):
                                if m * 16 >= p.kpg:
                                    continue
                                rs_const_base = (
                                    r_const * p.KW * N_K_ATOMS + s_const * N_K_ATOMS
                                ) * N_M_TILES
                                # W_coa layout: [groups, KH, KW, N_K_ATOMS, N_M_TILES, ...].
                                # Use absolute group g so each wave-group loads its own
                                # group's filters when block_groups > 1.
                                _rk_blocks_per_group = (
                                    p.KH * p.KW * N_K_ATOMS * N_M_TILES
                                )
                                _rk_g_abs = g if not PERSISTENT else b.const_i32(0)
                                _rk_group_base = b.mul(
                                    _rk_g_abs, b.const_i32(_rk_blocks_per_group)
                                )
                                block_idx_rk = b.add(
                                    _rk_group_base,
                                    b.add(
                                        b.const_i32(rs_const_base + m),
                                        b.mul(k_atom_global, b.const_i32(N_M_TILES)),
                                    ),
                                )
                                elem_off_rk = b.add(
                                    b.mul(block_idx_rk, b.const_i32(WAVE * LOAD_VEC)),
                                    b.mul(
                                        b.mod(tid, b.const_i32(WAVE)),
                                        b.const_i32(LOAD_VEC),
                                    ),
                                )
                                w_frag_rk = _buf_load_vN(
                                    b,
                                    p.dtype,
                                    b_rsrc,
                                    b.mul(elem_off_rk, c_half_bytes),
                                    c0,
                                    LOAD_VEC // 2,
                                )
                                mfma_shape = "16x16x32" if FOLD_K32 else "16x16x16"
                                new_k_accs_rk[m] = _mfma(
                                    b,
                                    p.dtype,
                                    mfma_shape,
                                    w_frag_rk,
                                    x_frag_rk,
                                    new_k_accs_rk[m],
                                )
                        b.scf_yield(*new_k_accs_rk)
                    for m in range(N_M_TILES):
                        acc_tiles[qt][m][p_idx] = k_loop_rk.results[m]
                    continue  # skip the existing runtime s_loop/atom_loop

                # ---- Runtime loop path (WAVES_K == 1, existing code) ----
                # Each (y_local, qt, r) combination emits a new scf.for loop.
                loop_tag = f"y{y_local}_qt{qt}_r{r_const}"

                # Thread all M-tile accumulators through the S and K-atom loops.
                s_iter_args = [
                    (f"siv_acc_m{m}_{loop_tag}", acc_tiles[qt][m][p_idx])
                    for m in range(N_M_TILES)
                ]
                s_loop = b.scf_for_iter(
                    c0,
                    c_KW,
                    c1,
                    s_iter_args,
                    iv_name=f"s_iv_{loop_tag}",
                    elide_trailing_barrier=False,
                )
                with s_loop as (s_iv, s_accs):
                    atom_iter_args = [
                        (f"katom_acc_m{m}_{loop_tag}", s_accs[m])
                        for m in range(N_M_TILES)
                    ]
                    # K-atom loop: runs over this wave's slice [k_atom_base, k_atom_base+N_K_LOCAL).
                    # k_atom_base offsets the loop into the correct K-slice.
                    atom_loop = b.scf_for_iter(
                        c0,
                        c_N_K_LOCAL,
                        c1,
                        atom_iter_args,
                        iv_name=f"atom_iv_{loop_tag}",
                        elide_trailing_barrier=False,
                    )
                    with atom_loop as (atom_iv_local, atom_accs):
                        # Global K-atom index: k_atom_base + atom_iv_local.
                        atom_iv_global = b.add(k_atom_base, atom_iv_local)
                        # Channel offset: atom_global × K_ATOM_SIZE + c4 × (K_ATOM_SIZE//4)
                        _std_c4_step = K_ATOM_SIZE // 4
                        ch_off = b.add(
                            b.mul(atom_iv_global, b.const_i32(K_ATOM_SIZE)),
                            b.mul(c4, b.const_i32(_std_c4_step)),
                        )

                        # LDS read: the LDS stores the full block Q × BG × cpg.
                        # q_tile_start_lds was used for loading; the wave reads
                        # at its own W-slice offset (qt_w_base within wave's tile).
                        W_lds_idx = b.add(
                            b.mul(
                                b.add(q_in_lane, b.const_i32(qt_w_base)),
                                b.const_i32(c_stride_gen),
                            ),
                            s_iv,
                        )
                        # When waves_q > 1, each wave's W-slice starts at
                        # wave_id_q * BLOCK_Q_WAVE within the block's LDS.
                        if WAVES_Q > 1:
                            wave_q_lds_base = b.mul(
                                wave_id_q, b.const_i32(BLOCK_Q_WAVE)
                            )
                            W_lds_idx_full = b.add(wave_q_lds_base, W_lds_idx)
                        else:
                            W_lds_idx_full = W_lds_idx
                        lds_idx = b.add(
                            b.add(
                                b.mul(W_lds_idx_full, c_BG_cpg),
                                b.mul(wave_group_idx, c_cpg),
                            ),
                            ch_off,
                        )
                        x_frag = b.smem_load_vN(
                            cur, c0, lds_idx, dtype=io_type, n=LOAD_VEC
                        )

                        if p.cpg % K_ATOM_SIZE != 0:
                            c4_oob = b.cmp_ge(ch_off, b.const_i32(p.cpg))
                            _zero_std = b.zero_vec(io_type, LOAD_VEC)
                            x_frag = b.select(c4_oob, _zero_std, x_frag)

                        new_atom_accs = []
                        for m in range(N_M_TILES):
                            k_out_m = b.add(
                                b.mul(g, c_kpg),
                                b.add(b.const_i32(m * 16), q_in_lane),
                            )
                            w_off, _ = b_desc.offset(
                                b,
                                k_out=k_out_m,
                                r=r_i,
                                s=s_iv,
                                c=ch_off,
                            )
                            w_frag = _buf_load_vN(
                                b,
                                p.dtype,
                                b_rsrc,
                                b.mul(w_off, c_half_bytes),
                                c0,
                                LOAD_VEC // 2,
                            )
                            if p.cpg % K_ATOM_SIZE != 0:
                                w_frag = b.select(c4_oob, _zero_std, w_frag)

                            mfma_shape = "16x16x32" if FOLD_K32 else "16x16x16"
                            new_acc = _mfma(
                                b, p.dtype, mfma_shape, w_frag, x_frag, atom_accs[m]
                            )
                            new_atom_accs.append(new_acc)

                        b.scf_yield(*new_atom_accs)

                    b.scf_yield(*atom_loop.results)

                for m in range(N_M_TILES):
                    acc_tiles[qt][m][p_idx] = s_loop.results[m]

        if loads_next is not None:
            if not spec.double_buffer:
                # Single-buffer: barrier to prevent overwriting the LDS row
                # that slower waves are still reading.
                b.sync()
            store_to_lds(loads_next, nxt)
        b.sync()

        # Flush accumulator for the output row that completes at this y_local.
        # p_flush_local is a Python int (tile-local offset from h_tile_start).
        p_flush_local = y_local - (p.KH - 1)
        P_FLUSH = p_flush_local % p.KH

        if BLOCK_H > 0:
            # With H-tiling: p_flush_global = h_tile_start + p_flush_local (runtime).
            # Flush guard is a runtime comparison.
            do_flush_python = p_flush_local >= 0  # tile-local lower bound
            if do_flush_python:
                p_flush_global = b.add(h_tile_start, b.const_i32(p_flush_local))
                # Guard: 0 <= p_flush_global < H (runtime).
                in_h_range = b.land(
                    b.cmp_ge(p_flush_global, c0),
                    b.cmp_lt(p_flush_global, b.const_i32(p.H)),
                )
                # For persistent grid: also gate on this round being in-bounds.
                if PERSISTENT:
                    in_h_range = b.land(in_h_range, pg_in_bounds)
                if c_stride_gen > 1:
                    stride_ok = b.cmp_eq(
                        b.mod(p_flush_global, b.const_i32(c_stride_gen)), c0
                    )
                    in_h_range = b.land(in_h_range, stride_ok)
                ho_row_val = b.div(p_flush_global, b.const_i32(c_stride_gen))
                should_flush = True  # yes, emit the store (guarded by in_h_range)
            else:
                should_flush = False  # negative local flush index, skip
        else:
            # No H-tiling: original Python-time bounds check.
            p_flush_val = p_flush_local
            should_flush = 0 <= p_flush_val < p.H and p_flush_val % c_stride_gen == 0
            if should_flush:
                ho_row_val = b.const_i32(p_flush_val // c_stride_gen)
                in_h_range = None  # no runtime guard needed

        if should_flush:
            for qt in range(q_subtiles):
                qt_w_base = qt * 16
                out_q = b.add(b.add(q_tile_start, b.const_i32(qt_w_base)), q_in_lane)
                out_q_ok = b.cmp_lt(out_q, c_W)

                for m in range(N_M_TILES):
                    if m * 16 >= p.kpg:
                        continue

                    acc_to_flush = acc_tiles[qt][m][P_FLUSH]
                    k_val = b.add(
                        b.mul(g, c_kpg),
                        b.add(b.const_i32(m * 16), b.mul(c4, b.const_i32(4))),
                    )
                    rows_in_tile = p.kpg - m * 16
                    if rows_in_tile < 16:
                        c4_ok = b.cmp_lt(
                            b.mul(c4, b.const_i32(4)), b.const_i32(rows_in_tile)
                        )
                        store_ok = b.land(out_q_ok, c4_ok)
                    else:
                        store_ok = out_q_ok

                    if BLOCK_H > 0 and in_h_range is not None:
                        store_ok = b.land(store_ok, in_h_range)

                    d_base, _ = d_desc.offset(b, n=n, h=ho_row_val, w=out_q, k=k_val)

                    if WAVES_K > 1:
                        # LDS reduction: each (wave_id_q, wave_id_k) wave writes its
                        # partial <4 x f32> to red_lds[wave_id_q*WAVES_K+wave_id_k, lane*4].
                        # After a barrier, wave_id_k==0 of each wave_id_q reads the
                        # WAVES_K rows belonging to its wq-group and sums them.
                        lane_col = b.mul(lane, b.const_i32(4))
                        b.smem_store_vN_f32(
                            red_lds, [lds_row_idx, lane_col], acc_to_flush, 4
                        )
                        b.sync()

                        # Master wave per W-subtile: wave_id_k == 0.
                        is_k0 = b.cmp_eq(wave_id_k, c0)
                        with b.scf_if(is_k0):
                            # Base row for this wave's Q-group: wave_id_q * WAVES_K.
                            row_base = b.mul(wave_id_q, b.const_i32(WAVES_K))
                            rows_f32 = [
                                b.smem_load_vN_f32(
                                    red_lds,
                                    b.add(row_base, b.const_i32(wk)),
                                    lane_col,
                                    n=4,
                                )
                                for wk in range(WAVES_K)
                            ]
                            sum_slots = []
                            for slot in range(4):
                                s = b.vec_extract(rows_f32[0], slot)
                                for wk in range(1, WAVES_K):
                                    s = b.fadd(s, b.vec_extract(rows_f32[wk], slot))
                                sum_slots.append(s)
                            partial = b.vec_pack(sum_slots, F32)
                            safe_d = b.select(
                                store_ok, b.mul(d_base, c_half_bytes), oob_sentinel
                            )
                            acc_h = _trunc_f32(b, p.dtype, partial)
                            _buf_store_vN(b, p.dtype, d_rsrc, safe_d, c0, acc_h, 2)
                        b.sync()  # allow red_lds reuse by next flush
                    else:
                        safe_d = b.select(
                            store_ok, b.mul(d_base, c_half_bytes), oob_sentinel
                        )
                        acc_h = _trunc_f32(b, p.dtype, acc_to_flush)
                        _buf_store_vN(b, p.dtype, d_rsrc, safe_d, c0, acc_h, 2)

        for qt in range(q_subtiles):
            for m in range(N_M_TILES):
                acc_tiles[qt][m][P_FLUSH] = zero_acc

    # Close the persistent cell loop if open.
    if PERSISTENT:
        # Yield cell_idx (changes each round) to prevent loop elimination.
        b.scf_yield(cell_idx)
        cell_loop.__exit__(None, None, None)

    return b.kernel


# ---------------------------------------------------------------------------
# Weight transpose kernel for dgrad  (W[K,r,s,C] → W_T[C,r',s',K] flipped)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DirectTransposeWeightsDgradSpec:
    """Spec for :func:`build_direct_transpose_weights_dgrad`."""

    problem: "DirectConvProblem"


@dataclass(frozen=True)
class DirectReorganizeWeightsSpec:
    """Spec for :func:`build_direct_reorganize_weights`."""

    problem: "DirectConvProblem"
    fold_k32: bool = False


@dataclass(frozen=True)
class DirectCoalescedWeightsDgradSpec:
    """Spec for :func:`build_direct_coalesced_weights_dgrad`."""

    problem: "DirectConvProblem"


@dataclass(frozen=True)
class DirectMfmaDgradSpec:
    """Spec for :func:`build_direct_mfma_dgrad`; wraps a fprop spec."""

    problem: "DirectConvSpec"


def build_direct_transpose_weights_dgrad(
    spec: "DirectTransposeWeightsDgradSpec", arch: str = "gfx950"
) -> "KernelDef":
    """Transpose W from [total_K, KH, KW, cpg] to [total_C, KH, KW, kpg] with
    spatial flip: ``W_T[c, r', s', k] = W[k, KH-1-r', KW-1-s', c]`` per group.

    After transposition the fprop streaming MFMA kernel can be called unchanged
    with dY as the "input" and W_T as the "weight".

    Tensor roles:
      A param — W:   source weights,     shape [total_K, KH, KW, cpg]
      D param — W_T: transposed weights, shape [total_C, KH, KW, kpg]

    Grid: (KH * KW * groups, ceil(kpg / 64), ceil(cpg / 64))
    Block: (64, 1, 1)
    """
    p = spec.problem
    io_type = _io_type(p.dtype)
    BLOCK = 64

    b = IRBuilder(f"direct_transpose_weights_dgrad_{p.short()}")
    b.kernel.attrs["max_workgroup_size"] = BLOCK

    A = b.param("A", PtrType(io_type, "global"), noalias=True, readonly=True, align=16)
    D = b.param("D", PtrType(io_type, "global"), noalias=True, writeonly=True, align=16)
    A_bytes = b.param("A_bytes", I32)
    D_bytes = b.param("D_bytes", I32)

    c0 = b.const_i32(0)
    c_half_bytes = b.const_i32(2)
    oob_sentinel = b.const_i32((1 << 31) - 1)

    tid = b.thread_id_x()

    # Grid:
    #   bx = flat (group * KH * KW) index:  g = bx // (KH*KW),  rs = bx % (KH*KW)
    #   by = k tile: k_in_g = by*64 + lane
    #   bz = c tile: c_in_g = bz*64 + some offset (here bz not used; tid covers c)
    # Simpler: bx = (group, r', s') flattened, by = k_in_g tile, bz = c_in_g tile.
    bx = b.block_id_x()
    by = b.block_id_y()
    bz = b.block_id_z()

    n_rs = p.KH * p.KW
    c_KH = b.const_i32(p.KH)
    c_KW = b.const_i32(p.KW)
    c_n_rs = b.const_i32(n_rs)

    grp = b.div(bx, c_n_rs)
    rs = b.mod(bx, c_n_rs)
    r_prime = b.div(rs, c_KW)
    s_prime = b.mod(rs, c_KW)
    # Flipped filter positions.
    r_flip = b.sub(b.const_i32(p.KH - 1), r_prime)
    s_flip = b.sub(b.const_i32(p.KW - 1), s_prime)

    k_in_g = b.add(b.mul(by, b.const_i32(BLOCK)), tid)
    c_in_g = bz  # one c per block in z-dim (scalar dispatch)

    k_abs = b.add(b.mul(grp, b.const_i32(p.kpg)), k_in_g)
    c_abs = b.add(b.mul(grp, b.const_i32(p.cpg)), c_in_g)

    k_ok = b.cmp_lt(k_in_g, b.const_i32(p.kpg))
    c_ok = b.cmp_lt(c_in_g, b.const_i32(p.cpg))
    valid = b.land(k_ok, c_ok)

    # Source: W[k_abs, r_flip, s_flip, c_in_g]
    src_desc = TensorDescriptor.naive(
        "A", lengths=[p.total_k, p.KH, p.KW, p.cpg], coord_names=("k", "r", "s", "c")
    )
    src_off, _ = src_desc.offset(b, k=k_abs, r=r_flip, s=s_flip, c=c_in_g)
    src_safe = b.select(valid, b.mul(src_off, c_half_bytes), oob_sentinel)
    a_rsrc = b.buffer_rsrc(A, A_bytes)
    d_rsrc = b.buffer_rsrc(D, D_bytes)
    if p.dtype == "bf16":
        val = b.buffer_load_bf16(a_rsrc, src_safe, c0)
    else:
        val = b.buffer_load_f16(a_rsrc, src_safe, c0)

    # Destination: W_T[c_abs, r', s', k_in_g]
    dst_desc = TensorDescriptor.naive(
        "D", lengths=[p.total_c, p.KH, p.KW, p.kpg], coord_names=("c", "r", "s", "k")
    )
    dst_off, _ = dst_desc.offset(b, c=c_abs, r=r_prime, s=s_prime, k=k_in_g)
    dst_safe = b.select(valid, b.mul(dst_off, c_half_bytes), oob_sentinel)
    if p.dtype == "bf16":
        b.buffer_store_bf16(d_rsrc, dst_safe, c0, val)
    else:
        b.buffer_store_f16(d_rsrc, dst_safe, c0, val)

    return b.kernel


def direct_dgrad_workspace_bytes(problem: "DirectConvProblem") -> int:
    """Bytes for the old (non-coalesced) transposed-weight workspace."""
    return problem.total_c * problem.KH * problem.KW * problem.kpg * 2  # fp16


def direct_dgrad_coalesced_workspace_bytes(
    problem: "DirectConvProblem", waves_k: int = 1, fold_k32: bool = False
) -> int:
    """Bytes for the coalesced weight workspace used by the MFMA dgrad with preloading.

    Layout: [groups, KH, KW, N_K_ATOMS, N_M_TILES, 64, 4] fp16 where:
      N_K_ATOMS = ceil(kpg / 16)  (kpg of original problem = cpg of transposed)
      N_M_TILES = ceil(cpg / 16)  (cpg of original problem = kpg of transposed)
      64  = one wave (lane dimension, coalesced)
      4   = one MFMA A-operand slot (4 consecutive halves per lane per atom)

    Each (r,s,atom,m) block of 64×4=256 halves is loaded coalesced by one wave.
    """
    p = problem
    K_ATOM_SZ = 32 if fold_k32 else 16
    ELEMS_PER_LANE = 8 if fold_k32 else 4
    if fold_k32 and p.kpg % 32 != 0:
        raise ValueError(
            f"DirectConvSpec fold_k32 requires cpg to be a multiple of 32 (got {p.kpg})"
        )
    N_K_ATOMS = (
        p.kpg + K_ATOM_SZ - 1
    ) // K_ATOM_SZ  # ceil; kpg_orig = cpg of transposed fprop
    N_M_TILES = (p.cpg + 15) // 16  # cpg_orig = kpg of transposed fprop
    return (
        p.groups * p.KH * p.KW * N_K_ATOMS * N_M_TILES * 64 * ELEMS_PER_LANE * 2
    )  # fp16


def build_direct_reorganize_weights(
    spec: "DirectReorganizeWeightsSpec", arch: str = "gfx950"
) -> "KernelDef":
    """Reorganize W_T[total_C, KH, KW, kpg] → W_coa[blocks, 64, 4] for coalesced preload.

    This is the SECOND pass (after ``build_direct_transpose_weights_dgrad``).
    It reads from the simple W_T format where kpg is the last (contiguous)
    dimension, giving perfectly contiguous vec4 reads per lane.  The output
    W_coa is organised so that consecutive lanes write to consecutive positions
    (stride = 4 halves = 8 bytes between lanes) → fully coalesced stores.

    After this kernel the compute-kernel preload reads W_coa at
    ``block_idx * 256 + lane_id * 4``.  All 64 lanes issue a single coalesced
    256-half (512-byte) read per (r, s, atom, m) triplet.

    Source access pattern (reading W_T):
      lane l = c4*16 + q_in_lane reads W_T[m*16+q_in_lane, r', s', atom*16+c4*4 .. +3]
      → 4 CONTIGUOUS halves along kpg (last dim of W_T) ✓
    Destination pattern (writing W_coa):
      lane l writes to W_coa[bx*256 + l*4 .. l*4+3]
      → stride 4 halves between consecutive l → coalesced ✓

    Grid: (groups * KH * KW * N_K_ATOMS * N_M_TILES, 1, 1)
    Block: (64, 1, 1)
    """
    p = spec.problem
    fold_k32 = spec.fold_k32
    K_ATOM_SZ = 32 if fold_k32 else 16
    ELEMS_PER_LANE = 8 if fold_k32 else 4
    if fold_k32 and p.kpg % 32 != 0:
        raise ValueError(
            f"DirectConvSpec fold_k32 requires cpg to be a multiple of 32 (got {p.kpg})"
        )
    N_K_ATOMS = (p.kpg + K_ATOM_SZ - 1) // K_ATOM_SZ  # ceil; matches workspace sizing
    N_M_TILES = (p.cpg + 15) // 16  # cpg = kpg of transposed fprop
    WAVE = 64

    b = IRBuilder(f"direct_reorg_wt{'32' if fold_k32 else ''}_{p.short()}")
    b.kernel.attrs["max_workgroup_size"] = WAVE

    io_type = _io_type(p.dtype)
    A = b.param("A", PtrType(io_type, "global"), noalias=True, readonly=True, align=16)
    D = b.param("D", PtrType(io_type, "global"), noalias=True, writeonly=True, align=16)
    A_bytes = b.param("A_bytes", I32)
    D_bytes = b.param("D_bytes", I32)

    c0 = b.const_i32(0)
    c_half_bytes = b.const_i32(2)
    oob_sentinel = b.const_i32((1 << 31) - 1)

    tid = b.thread_id_x()
    bx = b.block_id_x()

    n_per_group = p.KH * p.KW * N_K_ATOMS * N_M_TILES
    grp = b.div(bx, b.const_i32(n_per_group))
    rem = b.mod(bx, b.const_i32(n_per_group))
    r_prime = b.div(rem, b.const_i32(p.KW * N_K_ATOMS * N_M_TILES))
    rem2 = b.mod(rem, b.const_i32(p.KW * N_K_ATOMS * N_M_TILES))
    s_prime = b.div(rem2, b.const_i32(N_K_ATOMS * N_M_TILES))
    rem3 = b.mod(rem2, b.const_i32(N_K_ATOMS * N_M_TILES))
    atom_idx = b.div(rem3, b.const_i32(N_M_TILES))
    m_idx = b.mod(rem3, b.const_i32(N_M_TILES))

    q_in_lane = b.mod(tid, b.const_i32(16))
    c4 = b.div(tid, b.const_i32(16))

    # W_T layout: [total_C = p.cpg, KH, KW, kpg = p.kpg]
    # kpg is the last dim → the 4 elements per lane ARE contiguous.
    a_rsrc = b.buffer_rsrc(A, A_bytes)
    d_rsrc = b.buffer_rsrc(D, D_bytes)

    wt_desc = TensorDescriptor.naive(
        "A",
        lengths=[p.cpg, p.KH, p.KW, p.kpg],
        coord_names=("k_new", "r", "s", "c_new"),
    )
    k_new_val = b.add(
        b.mul(grp, b.const_i32(p.cpg)),
        b.add(b.mul(m_idx, b.const_i32(16)), q_in_lane),
    )
    c_new_base = b.add(
        b.mul(grp, b.const_i32(p.kpg)),
        b.add(b.mul(atom_idx, b.const_i32(16)), b.mul(c4, b.const_i32(4))),
    )
    src_ok = b.land(
        b.cmp_lt(b.add(b.mul(m_idx, b.const_i32(16)), q_in_lane), b.const_i32(p.cpg)),
        b.cmp_lt(
            b.add(
                b.mul(atom_idx, b.const_i32(K_ATOM_SZ)),
                b.mul(c4, b.const_i32(ELEMS_PER_LANE)),
            ),
            b.const_i32(p.kpg),
        ),
    )
    # c_new_base: start of this lane's K-slice within the current atom.
    # fold_k32: c4 selects groups of ELEMS_PER_LANE=8; fold_k16: groups of 4.
    c_new_base = b.add(
        b.mul(grp, b.const_i32(p.kpg)),
        b.add(
            b.mul(atom_idx, b.const_i32(K_ATOM_SZ)),
            b.mul(c4, b.const_i32(ELEMS_PER_LANE)),
        ),
    )
    src_off, _ = wt_desc.offset(
        b, k_new=k_new_val, r=r_prime, s=s_prime, c_new=c_new_base
    )
    safe_src = b.select(src_ok, b.mul(src_off, c_half_bytes), oob_sentinel)
    # Load ELEMS_PER_LANE consecutive elements: n_dwords = ELEMS_PER_LANE // 2
    val = _buf_load_vN(b, p.dtype, a_rsrc, safe_src, c0, ELEMS_PER_LANE // 2)

    # Destination: stride ELEMS_PER_LANE between consecutive lanes → coalesced.
    dst_off = b.add(
        b.mul(bx, b.const_i32(WAVE * ELEMS_PER_LANE)),
        b.mul(tid, b.const_i32(ELEMS_PER_LANE)),
    )
    safe_dst = b.select(src_ok, b.mul(dst_off, c_half_bytes), oob_sentinel)
    _buf_store_vN(b, p.dtype, d_rsrc, safe_dst, c0, val, ELEMS_PER_LANE // 2)

    return b.kernel


def build_direct_coalesced_weights_dgrad(
    spec: "DirectCoalescedWeightsDgradSpec", arch: str = "gfx950"
) -> "KernelDef":
    """Build a weight-transpose kernel that writes W in the coalesced MFMA preload format.

    Layout W_coalesced[groups, KH, KW, N_K_ATOMS, N_M_TILES, 64, 4] fp16:
      W_coa[g, r', s', atom, m, lane_id, elem] =
          W[g*kpg + m*16 + (lane_id%16), KH-1-r', KW-1-s', g*cpg + atom*16 + (lane_id//16)*4 + elem]

    The lane-index encodes both the M-tile row (q_in_lane = lane_id % 16) and the
    K-chunk within the atom (c4 = lane_id // 16, selecting 4 elements).  With this
    layout, weight preloading in the compute kernel is perfectly coalesced: all 64
    lanes of one wave issue a contiguous 512-byte read per (r', s', atom, m) triplet
    (stride = 4 halves = 8 bytes between consecutive lanes).

    Grid: (groups * KH * KW * N_K_ATOMS * N_M_TILES, 1, 1)
    Block: (64, 1, 1)
    """
    p = spec.problem
    N_K_ATOMS = (p.kpg + 15) // 16  # kpg_orig → cpg of transposed fprop
    N_M_TILES = (p.cpg + 15) // 16  # cpg_orig → kpg of transposed fprop
    WAVE = 64

    b = IRBuilder(f"direct_coa_wt_dgrad_{p.short()}")
    b.kernel.attrs["max_workgroup_size"] = WAVE

    io_type = _io_type(p.dtype)
    # A = W source, D = W_coalesced destination
    A = b.param("A", PtrType(io_type, "global"), noalias=True, readonly=True, align=16)
    D = b.param("D", PtrType(io_type, "global"), noalias=True, writeonly=True, align=16)
    A_bytes = b.param("A_bytes", I32)
    D_bytes = b.param("D_bytes", I32)

    c0 = b.const_i32(0)
    c_half_bytes = b.const_i32(2)
    oob_sentinel = b.const_i32((1 << 31) - 1)

    tid = b.thread_id_x()  # 0..63 = lane_id

    # Decode bx → (group, r', s', atom, m).
    bx = b.block_id_x()
    n_per_group = p.KH * p.KW * N_K_ATOMS * N_M_TILES
    c_npg = b.const_i32(n_per_group)
    grp = b.div(bx, c_npg)
    rem = b.mod(bx, c_npg)
    c_KW = b.const_i32(p.KW)
    c_natoms = b.const_i32(N_K_ATOMS)
    c_nmtiles = b.const_i32(N_M_TILES)
    r_prime = b.div(rem, b.const_i32(p.KW * N_K_ATOMS * N_M_TILES))
    rem2 = b.mod(rem, b.const_i32(p.KW * N_K_ATOMS * N_M_TILES))
    s_prime = b.div(rem2, b.const_i32(N_K_ATOMS * N_M_TILES))
    rem3 = b.mod(rem2, b.const_i32(N_K_ATOMS * N_M_TILES))
    atom_idx = b.div(rem3, c_nmtiles)
    m_idx = b.mod(rem3, c_nmtiles)

    # Flipped filter positions.
    r_flip = b.sub(b.const_i32(p.KH - 1), r_prime)
    s_flip = b.sub(b.const_i32(p.KW - 1), s_prime)

    # Lane decomposition: lane_id = c4*16 + q_in_lane
    q_in_lane = b.mod(tid, b.const_i32(16))
    c4 = b.div(tid, b.const_i32(16))

    # Source W[g*kpg + m*16 + q_in_lane, r_flip, s_flip, g*cpg + atom*16 + c4*4 + e]
    # Source descriptor in KRSC: [total_k, KH, KW, cpg]
    src_desc = TensorDescriptor.naive(
        "A",
        lengths=[p.total_k, p.KH, p.KW, p.cpg],
        coord_names=("k", "r", "s", "c"),
    )
    # For dgrad: W_T[k_new=c_orig, r', s', c_new=k_orig]
    #   = W_orig[k_orig=c_new, KH-1-r', KW-1-s', c_orig=k_new]
    # k_orig (first dim of W_orig, size kpg_orig=p.kpg) = atom*16 + c4*4 (K-reduction)
    # c_orig (last  dim of W_orig, size cpg_orig=p.cpg) = m*16 + q_in_lane (M-tile)
    k_src = b.add(
        b.mul(grp, b.const_i32(p.kpg)),
        b.add(b.mul(atom_idx, b.const_i32(16)), b.mul(c4, b.const_i32(4))),
    )
    c_src = b.add(
        b.mul(grp, b.const_i32(p.cpg)),
        b.add(b.mul(m_idx, b.const_i32(16)), q_in_lane),
    )
    a_rsrc = b.buffer_rsrc(A, A_bytes)
    d_rsrc = b.buffer_rsrc(D, D_bytes)

    # Validity: k_orig must be < total_k, c_orig must be < cpg.
    k_ok = b.land(
        b.cmp_lt(
            b.add(b.mul(atom_idx, b.const_i32(16)), b.mul(c4, b.const_i32(4))),
            b.const_i32(p.kpg),
        ),
        b.cmp_lt(b.add(b.mul(m_idx, b.const_i32(16)), q_in_lane), b.const_i32(p.cpg)),
    )
    src_off, _ = src_desc.offset(b, k=k_src, r=r_flip, s=s_flip, c=c_src)
    safe_src = b.select(k_ok, b.mul(src_off, c_half_bytes), oob_sentinel)
    val = _buf_load_vN(b, p.dtype, a_rsrc, safe_src, c0, 2)

    # Destination: W_coa[bx * WAVE * 4 + tid * 4]  (coalesced: tid*4 stride)
    dst_off = b.add(
        b.mul(bx, b.const_i32(WAVE * 4)),
        b.mul(tid, b.const_i32(4)),
    )
    safe_dst = b.select(k_ok, b.mul(dst_off, c_half_bytes), oob_sentinel)
    _buf_store_vN(b, p.dtype, d_rsrc, safe_dst, c0, val, 2)

    return b.kernel


def build_direct_mfma_dgrad(
    spec: "DirectMfmaDgradSpec", arch: str = "gfx950"
) -> "Tuple[KernelDef, KernelDef]":
    """Build the two kernels for the MFMA dgrad pipeline.

    Returns ``(transpose_kernel, fprop_kernel)`` where:
      - ``transpose_kernel`` converts W → W_T (workspace) via
        :func:`build_direct_transpose_weights_dgrad`.
      - ``fprop_kernel`` runs the unmodified MFMA streaming fprop on
        ``(dY, W_T) → dX``.  The caller passes the workspace as the
        "B" (weight) argument.

    The ``fprop_spec`` must describe the *transposed* problem:
      N = N,  H = Ho,  W = Wo  (dY spatial dimensions)
      cpg = kpg_orig,  kpg = cpg_orig  (swapped channel counts)
      KH, KW, PAD, stride = 1  (same filter, same PAD for symmetric case)

    Use :func:`make_dgrad_fprop_spec` to build the spec from the original
    conv problem automatically.
    """
    fprop_spec = spec.problem
    orig_p = fprop_spec.problem
    # Reconstruct original problem from the transposed fprop spec.
    # orig.cpg = fprop.kpg, orig.kpg = fprop.cpg,
    # orig.H = fprop.Ho (fprop streams dY rows),
    # orig.KH = fprop.KH (same filter), etc.
    # For the transpose kernel we need the ORIGINAL problem dimensions.
    # We recover them: original cpg = fprop.kpg, kpg = fprop.cpg.
    orig_problem = DirectConvProblem(
        N=orig_p.N,
        H=orig_p.Ho,  # original H was fprop's Ho (dY height)
        W=orig_p.Wo,  # original W was fprop's Wo
        groups=orig_p.groups,
        cpg=orig_p.kpg,  # original cpg = fprop.kpg
        kpg=orig_p.cpg,  # original kpg = fprop.cpg
        KH=orig_p.KH,
        KW=orig_p.KW,
        PAD=orig_p.KH - 1 - orig_p.PAD,  # undo the PAD swap
        stride=1,
        dtype=orig_p.dtype,
    )
    transpose_kernel = build_direct_transpose_weights_dgrad(
        DirectTransposeWeightsDgradSpec(problem=orig_problem), arch=arch
    )
    fprop_kernel = build_direct_conv(fprop_spec, arch=arch)
    return transpose_kernel, fprop_kernel


def make_dgrad_fprop_spec(
    problem: "DirectConvProblem",
    block_q: int = 16,
    block_groups: int = 1,
    block_h: int = 0,
    double_buffer: bool = True,
    waves_q: int = 1,
    waves_k: int = 1,
    runtime_k_loop: bool = False,
    persistent_grid: bool = False,
) -> "DirectConvSpec":
    """Build the ``DirectConvSpec`` for the transposed-fprop pass of dgrad.

    For the original conv (N, H, W, groups, cpg, kpg, KH, KW, PAD, stride=1)
    the transposed-fprop problem is:
      N = N,  H = Ho,  W = Wo,  groups = groups
      cpg_new = kpg  (MFMA K-reduction = original output channels)
      kpg_new = cpg  (MFMA output = original input channels = dX channels)
      PAD_new = KH - 1 - PAD  (for symmetric PAD=(KH-1)/2 this equals PAD)
      stride  = 1

    The weight W_T (in workspace) has shape [total_C, KH, KW, kpg] which
    matches [total_K_new, KH, KW, cpg_new] expected by the fprop kernel.

    ``block_h > 0`` enables H-tiling: each block processes ``block_h`` output
    rows, giving ``ceil(Ho/block_h)`` more blocks in the Z-grid dimension.
    H-tiling increases block count for large H: ``block_h = 16`` gives
    ``ceil(Ho/16)`` extra blocks in the Z-grid dimension.
    """
    p = problem
    assert p.stride == 1, "make_dgrad_fprop_spec requires stride=1"
    pad_new = p.KH - 1 - p.PAD

    transposed_problem = DirectConvProblem(
        N=p.N,
        H=p.Ho,
        W=p.Wo,
        groups=p.groups,
        cpg=p.kpg,  # K-reduction axis = original output channels
        kpg=p.cpg,  # output axis       = original input channels (dX)
        KH=p.KH,
        KW=p.KW,
        PAD=pad_new,
        stride=1,
        dtype=p.dtype,
    )
    return DirectConvSpec(
        problem=transposed_problem,
        name="direct_mfma_dgrad",
        block_q=block_q,
        block_groups=block_groups,
        double_buffer=double_buffer,
        block_h=block_h,
        waves_q=waves_q,
        waves_k=waves_k,
        runtime_k_loop=runtime_k_loop,
        persistent_grid=persistent_grid,
    )


# ---------------------------------------------------------------------------
# Direct grouped convolution — backward data (dgrad, scalar FMA fallback)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DirectConvDgradSpec:
    """Direct grouped dgrad kernel: computes the input gradient dX.

    Computes::

        dX[n, hi, wi, c] = sum_{r, s, k} dY[n, ho, wo, k] * W[k, r, s, c]

    where ho = (hi + PAD - r) / stride, wo = (wi + PAD - s) / stride.

    Algorithm — per-(n, wi_tile, c_in_tile) workgroup:
      The workgroup loops over all H rows (hi_iv) and for each (r, s) tap
      reduces over all K output channels via scalar FMA.  No weight transpose
      prepass is needed: W is accessed as W[k, r, s, c_in] with a strided
      pointer step per k.

    Block geometry:
      - ``block_groups`` waves per workgroup, one group per wave.
      - ``block_q = 16`` input W positions per block.
      - Scalar FMA over K (no MFMA); supports any cpg/kpg alignment and stride >= 1.
      - Grid: (ceil(Wi / block_q), ceil(total_c / block_ch), N)
        where block_ch = block_groups * wave_size; each workgroup loops over all H.

    Supported: cpg >= 1, kpg >= 1, stride >= 1, PAD >= 0, groups divisible by block_groups.
    """

    problem: DirectConvProblem
    name: str = "direct_conv_dgrad"
    block_q: int = 16
    block_groups: int = 8
    wave_size: int = 64

    @property
    def threads_per_block(self) -> int:
        return self.block_groups * self.wave_size

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        p = self.problem
        return kernel_name_join(
            self.name,
            p.short(),
            f"bq{self.block_q}",
            f"bg{self.block_groups}",
            flags={"bf16": p.dtype == "bf16"},
        )

    def validate(self) -> None:
        p = self.problem
        if p.dtype not in ("fp16", "bf16"):
            raise ValueError(f"DirectConvDgradSpec: unsupported dtype {p.dtype!r}")
        if p.cpg < 1:
            raise ValueError(f"DirectConvDgradSpec requires cpg >= 1 (got {p.cpg})")
        if p.kpg < 1:
            raise ValueError(f"DirectConvDgradSpec requires kpg >= 1 (got {p.kpg})")
        if p.groups % self.block_groups != 0:
            raise ValueError(
                f"groups {p.groups} not divisible by block_groups {self.block_groups}"
            )


def is_valid_dgrad_spec(
    spec: DirectConvDgradSpec, arch: str = "gfx950"
) -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for a dgrad spec on ``arch``.

    The dgrad kernel uses scalar FMA (no MFMA), so there are no MFMA-atom
    constraints on cpg or kpg alignment.  stride > 1 is supported.
    """
    from rocke.core.arch import ArchTarget

    try:
        ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)
    p = spec.problem
    if p.cpg < 1:
        return False, f"cpg must be >= 1 (got {p.cpg})"
    if p.kpg < 1:
        return False, f"kpg must be >= 1 (got {p.kpg})"
    if p.groups % spec.block_groups != 0:
        return (
            False,
            f"groups {p.groups} not divisible by block_groups {spec.block_groups}",
        )
    return True, "ok"


def build_direct_conv_dgrad(
    spec: DirectConvDgradSpec, arch: str = "gfx950"
) -> KernelDef:
    """Build the IR for the direct grouped convolution dgrad kernel.

    Computes dX[n, hi, wi, c] = sum_{r, s, k} dY[n, ho, wo, k] * W[k, r, s, c]
    where ho = (hi + PAD - r) / stride, wo = (wi + PAD - s) / stride.

    Algorithm — scalar FMA over (r, s, k_out):
      Each thread owns one (c_in, wi) output element and reduces over all
      (r, s) filter taps and k_out output channels via scalar FMA.  This
      avoids the need for a weight-transpose prepass: W is accessed as
      W[k_out, r, s, c_in] using a strided offset per k_out step.

      dY is loaded as vec4 (4 consecutive k values) per outer k_out block
      to amortise the DRAM-load cost over the c_in reduction.  W is loaded
      as 4 individual scalar loads at stride KH*KW*cpg between k_out values.

    Tensor roles:
      A param — dY: output gradient, shape [N, Ho, Wo, total_k], NHWK
      B param — W:  weights,          shape [total_k, KH, KW, cpg], KRSC
      D param — dX: input gradient,   shape [N, H, W, total_c], NHWC

    Grid: (ceil(Wi / block_w), ceil(total_c / block_ch), N * Hi)
    Block: (block_waves * 64, 1, 1)
    """
    spec.validate()
    ok, why = is_valid_dgrad_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid DirectConvDgradSpec for {arch}: {why}")

    p = spec.problem
    io_type = _io_type(p.dtype)
    BLOCK_W = spec.block_q  # reuse block_q field as input-W tile
    BLOCK_WAVES = spec.block_groups  # reuse block_groups as wave count per block
    WAVE = spec.wave_size
    THREADS = BLOCK_WAVES * WAVE
    BLOCK_CH = BLOCK_WAVES * WAVE  # channels per workgroup tile

    Ho = p.Ho
    Wo = p.Wo
    c_stride = p.stride

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = THREADS

    A = b.param("A", PtrType(io_type, "global"), noalias=True, readonly=True, align=16)
    Bp = b.param("B", PtrType(io_type, "global"), noalias=True, readonly=True, align=16)
    D = b.param("D", PtrType(io_type, "global"), noalias=True, writeonly=True, align=16)
    A_bytes = b.param("A_bytes", I32)
    B_bytes = b.param("B_bytes", I32)
    D_bytes = b.param("D_bytes", I32)

    c0 = b.const_i32(0)
    c1 = b.const_i32(1)
    c_total_c = b.const_i32(p.total_c)
    c_total_k = b.const_i32(p.total_k)
    c_half_bytes = b.const_i32(2)
    c_wave = b.const_i32(WAVE)
    oob_sentinel = b.const_i32((1 << 31) - 1)
    zero_f32 = b.const_f32(0.0)

    tid = b.thread_id_x()
    wave_id = b.div(tid, c_wave)
    lane = b.mod(tid, c_wave)

    # Grid: bx=Wi-tile, by=c_in-tile, bz=n.
    # Each workgroup iterates over all H rows in the scf_for below, so grid.z = N.
    bx = b.block_id_x()
    by = b.block_id_y()
    bz = b.block_id_z()
    n = bz

    wi_tile_start = b.mul(bx, b.const_i32(BLOCK_W))
    # Absolute c_in index for this thread.
    c_in = b.add(
        b.mul(by, b.const_i32(BLOCK_CH)),
        b.add(b.mul(wave_id, c_wave), lane),
    )
    c_in_ok = b.cmp_lt(c_in, c_total_c)

    a_rsrc = b.buffer_rsrc(A, A_bytes)
    b_rsrc = b.buffer_rsrc(Bp, B_bytes)
    d_rsrc = b.buffer_rsrc(D, D_bytes)

    # dY descriptor: A[N, Ho, Wo, total_k] NHWK (k is contiguous).
    dy_desc = TensorDescriptor.naive(
        "A", lengths=[p.N, Ho, Wo, p.total_k], coord_names=("n", "ho", "wo", "k")
    )
    # W descriptor: B[total_k, KH, KW, cpg] KRSC (c is contiguous).
    b_desc = TensorDescriptor.naive(
        "B", lengths=[p.total_k, p.KH, p.KW, p.cpg], coord_names=("k", "r", "s", "c")
    )
    # dX descriptor: D[N, H, W, total_c] NHWC.
    d_desc = TensorDescriptor.naive(
        "D", lengths=[p.N, p.H, p.W, p.total_c], coord_names=("n", "h", "w", "c")
    )

    # Stride between consecutive k_out values in W (in bytes):
    #   W[k+1, r, s, c] - W[k, r, s, c] = KH*KW*cpg * 2 bytes
    k_stride_bytes = b.const_i32(p.KH * p.KW * p.cpg * 2)
    c_Wi = b.const_i32(p.W)
    c_kpg = b.const_i32(p.kpg)
    c_st = b.const_i32(c_stride) if c_stride > 1 else None

    # Runtime H-loop: each hi iteration is self-contained.
    # A dummy i32 iter_arg threads through the loop.
    hi_loop = b.scf_for_iter(
        c0,
        b.const_i32(p.H),
        c1,
        [("dg_hi_dummy", b.const_i32(0))],
        iv_name="dg_hi",
        elide_trailing_barrier=False,
    )
    with hi_loop as (hi_iv, (dummy_in,)):
        for j in range(BLOCK_W):
            wi = b.add(wi_tile_start, b.const_i32(j))
            wi_ok = b.cmp_lt(wi, c_Wi)

            acc = zero_f32

            for r_const in range(p.KH):
                hi_p_r = b.add(hi_iv, b.const_i32(p.PAD - r_const))
                if c_stride > 1:
                    ho = b.div(hi_p_r, c_st)
                    r_valid = b.land(
                        b.cmp_ge(hi_p_r, c0),
                        b.land(
                            b.cmp_eq(b.mod(hi_p_r, c_st), c0),
                            b.cmp_lt(ho, b.const_i32(Ho)),
                        ),
                    )
                else:
                    ho = hi_p_r
                    r_valid = b.land(
                        b.cmp_ge(hi_p_r, c0), b.cmp_lt(ho, b.const_i32(Ho))
                    )

                for s_const in range(p.KW):
                    wi_p_s = b.add(wi, b.const_i32(p.PAD - s_const))
                    if c_stride > 1:
                        wo = b.div(wi_p_s, c_st)
                        s_valid = b.land(
                            b.cmp_ge(wi_p_s, c0),
                            b.land(
                                b.cmp_eq(b.mod(wi_p_s, c_st), c0),
                                b.cmp_lt(wo, b.const_i32(Wo)),
                            ),
                        )
                    else:
                        wo = wi_p_s
                        s_valid = b.land(
                            b.cmp_ge(wi_p_s, c0), b.cmp_lt(wo, b.const_i32(Wo))
                        )

                    tap_valid = b.land(b.land(r_valid, s_valid), b.land(c_in_ok, wi_ok))

                    # Precompute W base offset at k_out=0 for this (r, s, c_in).
                    # W offset step per k_out: KH*KW*cpg*2 bytes (stride along k dim).
                    # c_in_in_group = c_in % cpg (c index within the filter's last dim).
                    c_in_in_grp = b.mod(c_in, b.const_i32(p.cpg))
                    # Determine group: group = c_in // cpg → k_out base = group * kpg.
                    grp = b.div(c_in, b.const_i32(p.cpg))
                    k_base = b.mul(grp, c_kpg)

                    # W base offset at (k=k_base, r, s, c_in_in_grp) in bytes.
                    w_off0, _ = b_desc.offset(
                        b,
                        k=k_base,
                        r=b.const_i32(r_const),
                        s=b.const_i32(s_const),
                        c=c_in_in_grp,
                    )
                    w_off0_bytes = b.mul(w_off0, c_half_bytes)

                    # dY base offset at (n, ho, wo, k=k_base) in bytes.
                    # dy_desc is a naive descriptor (no embed), so dy_valid = None;
                    # the ho/wo boundary is already encoded in tap_valid above.
                    dy_off0, _ = dy_desc.offset(b, n=n, ho=ho, wo=wo, k=k_base)
                    dy_off0_bytes = b.mul(dy_off0, c_half_bytes)

                    # Inner loop over k_out within the group (runtime scf.for).
                    loop_tag = f"dg_rs_r{r_const}_s{s_const}_j{j}"
                    k_loop = b.scf_for_iter(
                        c0,
                        c_kpg,
                        c1,
                        [(f"k_acc_{loop_tag}", acc)],
                        iv_name=f"dg_k_{loop_tag}",
                        elide_trailing_barrier=False,
                    )
                    with k_loop as (k_iv, (acc_k,)):
                        # W[k_base + k_iv, r, s, c_in_in_grp]: strided k_out access.
                        k_byte_off = b.mul(k_iv, k_stride_bytes)
                        w_byte = b.add(w_off0_bytes, k_byte_off)
                        safe_w = b.select(tap_valid, w_byte, oob_sentinel)
                        if p.dtype == "bf16":
                            w_h = b.buffer_load_bf16(b_rsrc, safe_w, c0)
                        else:
                            w_h = b.buffer_load_f16(b_rsrc, safe_w, c0)
                        w_f32 = b.select(tap_valid, b.cast_to_f32(w_h), zero_f32)

                        # dY[n, ho, wo, k_base + k_iv]: k is contiguous in NHWK.
                        dy_byte = b.add(dy_off0_bytes, b.mul(k_iv, c_half_bytes))
                        safe_dy = b.select(tap_valid, dy_byte, oob_sentinel)
                        if p.dtype == "bf16":
                            dy_h = b.buffer_load_bf16(a_rsrc, safe_dy, c0)
                        else:
                            dy_h = b.buffer_load_f16(a_rsrc, safe_dy, c0)
                        dy_f32 = b.select(tap_valid, b.cast_to_f32(dy_h), zero_f32)

                        new_acc = b.fma(w_f32, dy_f32, acc_k)
                        b.scf_yield(new_acc)

                    acc = k_loop.results[0]

            # Store dX[n, hi, wi, c_in].
            d_off, _ = d_desc.offset(b, n=n, h=hi_iv, w=wi, c=c_in)
            safe_d = b.select(
                b.land(c_in_ok, wi_ok), b.mul(d_off, c_half_bytes), oob_sentinel
            )
            if p.dtype == "bf16":
                b.buffer_store_bf16(d_rsrc, safe_d, c0, b.trunc_f32_to_bf16(acc))
            else:
                b.buffer_store_f16(d_rsrc, safe_d, c0, b.trunc_f32_to_f16(acc))

        b.scf_yield(dummy_in)

    return b.kernel


# ---------------------------------------------------------------------------
# Depthwise dgrad kernel — cpg = kpg = 1, groups = C = K, any stride
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DirectDepthwiseDgradSpec:
    """Direct depthwise dgrad kernel for ``cpg = kpg = 1`` (groups == C == K).

    Computes::

        dX[n, hi, wi, ch] = sum_{r, s} dY[n, ho, wo, ch] * W[ch, r, s, 0]

    where ``ho = (hi + PAD - r) / stride`` and ``wo = (wi + PAD - s) / stride``
    (the quotients must be exact integers lying in ``[0, Ho)``/``[0, Wo)``).

    For stride > 1 most (r, s) positions are invalid for a given (hi, wi) —
    the embed boundary check handles this automatically.

    Algorithm: scalar FMA over (r, s) filter taps, identical in structure to
    the fprop depthwise kernel but iterating over input rows hi (not output
    rows ho). Weights are preloaded into f32 registers.

    Block geometry:
      ``threads_per_block = block_waves * 64``
      Grid: ``(ceil(Wi / block_w), ceil(C / block_ch), N)``
    """

    problem: DirectConvProblem
    name: str = "direct_depthwise_dgrad"
    block_w: int = 8
    block_waves: int = 1
    wave_size: int = 64

    @property
    def threads_per_block(self) -> int:
        return self.block_waves * self.wave_size

    @property
    def block_ch(self) -> int:
        return self.block_waves * self.wave_size

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        p = self.problem
        return kernel_name_join(
            self.name,
            p.short(),
            f"bw{self.block_w}",
            f"bw{self.block_waves}wv",
            flags={"bf16": p.dtype == "bf16"},
        )

    def validate(self) -> None:
        p = self.problem
        if p.dtype not in ("fp16", "bf16"):
            raise ValueError(
                f"DirectDepthwiseDgradSpec: unsupported dtype {p.dtype!r}; expected fp16 or bf16"
            )
        if p.cpg != 1 or p.kpg != 1:
            raise ValueError(
                f"DirectDepthwiseDgradSpec requires cpg=kpg=1 (got {p.cpg}, {p.kpg})"
            )


def is_valid_depthwise_dgrad_spec(
    spec: DirectDepthwiseDgradSpec, arch: str = "gfx950"
) -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for a depthwise dgrad spec on ``arch``."""
    from rocke.core.arch import ArchTarget

    try:
        ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)
    p = spec.problem
    if p.dtype not in ("fp16", "bf16"):
        return (
            False,
            f"DirectDepthwiseDgradSpec: unsupported dtype {p.dtype!r}; expected fp16 or bf16",
        )
    if p.cpg != 1 or p.kpg != 1:
        return False, f"requires cpg=kpg=1 (got {p.cpg}, {p.kpg})"
    return True, "ok"


def build_direct_depthwise_dgrad(
    spec: DirectDepthwiseDgradSpec, arch: str = "gfx950"
) -> KernelDef:
    """Build the IR for the scalar depthwise dgrad kernel.

    Computes dX from dY and W using a scalar FMA loop over (r, s) filter taps.
    Supports any stride and padding.

    Tensor roles:
      A param — dY: output gradient, shape [N, Ho, Wo, groups], NHWK
      B param — W:  weights,          shape [groups, KH, KW, 1], KRSC
      D param — dX: input gradient,   shape [N, H, W, groups],   NHWC

    Grid: (ceil(Wi / block_w), ceil(groups / block_ch), N)
    Block: (block_waves * 64, 1, 1)

    The kernel uses a runtime H-loop (scf.for over hi = 0..H-1). Each hi
    iteration is self-contained: accumulate over valid (r, s) taps, then store
    dX[hi]. A dummy iteration argument threads through the loop so no actual
    state crosses iteration boundaries.
    """
    spec.validate()
    ok, why = is_valid_depthwise_dgrad_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid DirectDepthwiseDgradSpec for {arch}: {why}")

    p = spec.problem
    BLOCK_W = spec.block_w
    BLOCK_WAVES = spec.block_waves
    WAVE = spec.wave_size
    THREADS = spec.threads_per_block
    BLOCK_CH = spec.block_ch
    Ho = p.Ho
    Wo = p.Wo
    c_stride = p.stride  # Python int — used in build-time divisibility tests

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = THREADS

    io_type = _io_type(p.dtype)
    A = b.param("A", PtrType(io_type, "global"), noalias=True, readonly=True, align=16)
    Bp = b.param("B", PtrType(io_type, "global"), noalias=True, readonly=True, align=16)
    D = b.param("D", PtrType(io_type, "global"), noalias=True, writeonly=True, align=16)
    A_bytes = b.param("A_bytes", I32)
    B_bytes = b.param("B_bytes", I32)
    D_bytes = b.param("D_bytes", I32)

    c0 = b.const_i32(0)
    c1 = b.const_i32(1)
    c_groups = b.const_i32(p.groups)
    c_half_bytes = b.const_i32(2)
    c_wave = b.const_i32(WAVE)
    oob_sentinel = b.const_i32((1 << 31) - 1)
    zero_f32 = b.const_f32(0.0)

    tid = b.thread_id_x()
    wave_id = b.div(tid, c_wave)
    lane = b.mod(tid, c_wave)

    bx = b.block_id_x()
    by = b.block_id_y()
    n = b.block_id_z()
    wi_tile_start = b.mul(bx, b.const_i32(BLOCK_W))
    ch = b.add(
        b.mul(by, b.const_i32(BLOCK_CH)),
        b.add(b.mul(wave_id, c_wave), lane),
    )
    ch_in_range = b.cmp_lt(ch, c_groups)

    a_rsrc = b.buffer_rsrc(A, A_bytes)
    b_rsrc = b.buffer_rsrc(Bp, B_bytes)
    d_rsrc = b.buffer_rsrc(D, D_bytes)

    dy_desc = TensorDescriptor.naive(
        "A", lengths=[p.N, Ho, Wo, p.groups], coord_names=("n", "ho", "wo", "ch")
    )
    b_desc = TensorDescriptor.naive(
        "B", lengths=[p.groups, p.KH, p.KW, 1], coord_names=("k", "r", "s", "c")
    )
    d_desc = TensorDescriptor.naive(
        "D", lengths=[p.N, p.H, p.W, p.groups], coord_names=("n", "h", "w", "ch")
    )

    # Preload W[ch, r, s, 0] into registers (KH * KW f32 per lane).
    weights_f32: List[List[Value]] = []
    for r_const in range(p.KH):
        row: List[Value] = []
        for s_const in range(p.KW):
            w_off, _ = b_desc.offset(
                b, k=ch, r=b.const_i32(r_const), s=b.const_i32(s_const), c=c0
            )
            safe_w = b.select(ch_in_range, b.mul(w_off, c_half_bytes), oob_sentinel)
            w_h = (
                b.buffer_load_bf16(b_rsrc, safe_w, c0)
                if p.dtype == "bf16"
                else b.buffer_load_f16(b_rsrc, safe_w, c0)
            )
            row.append(b.select(ch_in_range, b.cast_to_f32(w_h), zero_f32))
        weights_f32.append(row)

    c_Wi = b.const_i32(p.W)
    c_st = b.const_i32(c_stride) if c_stride > 1 else None

    # Runtime H-loop: each iteration independently accumulates and stores dX[hi].
    # A dummy i32 iter_arg threads through to satisfy scf_for_iter requirements
    # (no actual state crosses iterations — each hi is self-contained).
    hi_loop = b.scf_for_iter(
        c0,
        b.const_i32(p.H),
        c1,
        [("dg_dw_dummy", b.const_i32(0))],
        iv_name="dg_dw_hi",
        elide_trailing_barrier=False,
    )
    with hi_loop as (hi_iv, (dummy_in,)):
        for j in range(BLOCK_W):
            wi = b.add(wi_tile_start, b.const_i32(j))
            wi_ok = b.cmp_lt(wi, c_Wi)

            acc = zero_f32
            for r_const in range(p.KH):
                # ho = (hi + PAD - r) / stride — check non-negative, in-range, divisible.
                hi_p_r = b.add(hi_iv, b.const_i32(p.PAD - r_const))
                if c_stride > 1:
                    ho = b.div(hi_p_r, c_st)
                    r_valid = b.land(
                        b.cmp_ge(hi_p_r, c0),
                        b.land(
                            b.cmp_eq(b.mod(hi_p_r, c_st), c0),
                            b.cmp_lt(ho, b.const_i32(Ho)),
                        ),
                    )
                else:
                    ho = hi_p_r
                    r_valid = b.land(
                        b.cmp_ge(hi_p_r, c0), b.cmp_lt(ho, b.const_i32(Ho))
                    )

                for s_const in range(p.KW):
                    wi_p_s = b.add(wi, b.const_i32(p.PAD - s_const))
                    if c_stride > 1:
                        wo = b.div(wi_p_s, c_st)
                        s_valid = b.land(
                            b.cmp_ge(wi_p_s, c0),
                            b.land(
                                b.cmp_eq(b.mod(wi_p_s, c_st), c0),
                                b.cmp_lt(wo, b.const_i32(Wo)),
                            ),
                        )
                    else:
                        wo = wi_p_s
                        s_valid = b.land(
                            b.cmp_ge(wi_p_s, c0), b.cmp_lt(wo, b.const_i32(Wo))
                        )

                    valid = b.land(b.land(r_valid, s_valid), b.land(ch_in_range, wi_ok))

                    dy_off, _ = dy_desc.offset(b, n=n, ho=ho, wo=wo, ch=ch)
                    safe_dy = b.select(valid, b.mul(dy_off, c_half_bytes), oob_sentinel)
                    dy_h = (
                        b.buffer_load_bf16(a_rsrc, safe_dy, c0)
                        if p.dtype == "bf16"
                        else b.buffer_load_f16(a_rsrc, safe_dy, c0)
                    )
                    dy_f32 = b.select(valid, b.cast_to_f32(dy_h), zero_f32)
                    acc = b.fma(weights_f32[r_const][s_const], dy_f32, acc)

            # Store dX[n, hi, wi, ch].
            d_off, _ = d_desc.offset(b, n=n, h=hi_iv, w=wi, ch=ch)
            safe_d = b.select(
                b.land(ch_in_range, wi_ok), b.mul(d_off, c_half_bytes), oob_sentinel
            )
            if p.dtype == "bf16":
                b.buffer_store_bf16(d_rsrc, safe_d, c0, b.trunc_f32_to_bf16(acc))
            else:
                b.buffer_store_f16(d_rsrc, safe_d, c0, b.trunc_f32_to_f16(acc))

        b.scf_yield(dummy_in)

    return b.kernel


# ---------------------------------------------------------------------------
# Depthwise dgrad — ho-streaming with circular accumulator slots
#
# Algorithm: stream dY rows (ho=0..Ho-1) like the fprop kernel streams hi rows.
# For each ho, every r-tap contributes to hi = ho*stride + r - PAD.  A circular
# buffer of n_slots = stride*KH accumulator slots carries each hi value until all
# its (ho, r) contributions are received, then flushes it to dX.
#
#   - dY is streamed forward one row at a time (cache-friendly, no scatter reads)
#   - W[r, s, ch] is preloaded into registers once before the ho-loop
#   - Each lane holds one (ch, wi) pair and accumulates in f32 registers
#
# Supports any stride.  For stride=1 the flush logic degenerates to the fprop
# mirror (flush one hi per ho).  For stride=2, up to stride hi values are flushed
# per ho (the last ho flushes the remaining odd hi values).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DirectDepthwiseDgradStreamSpec:
    """Ho-streaming depthwise dgrad kernel for any stride.

    Streams dY rows (ho), accumulates into circular dX slots (hi), and flushes
    each hi value when all its contributing (ho, r) pairs have been processed.
    W[ch, r, s] is preloaded into registers before the ho-loop.

    Block geometry:
      ``threads_per_block = block_waves * 64``
      Grid: ``(ceil(Wo / block_w), ceil(C / block_ch), N)``
      (Note: grid uses Wo/Wo — the dY spatial dims — but block_w tiles dX Wi too.)
    """

    problem: DirectConvProblem
    name: str = "direct_depthwise_dgrad_stream"
    block_w: int = 8  # dX Wi positions per block (also controls LDS width)
    block_waves: int = 1
    wave_size: int = 64

    @property
    def threads_per_block(self) -> int:
        return self.block_waves * self.wave_size

    @property
    def block_ch(self) -> int:
        return self.block_waves * self.wave_size

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        p = self.problem
        return kernel_name_join(
            self.name,
            p.short(),
            f"bw{self.block_w}",
            f"bw{self.block_waves}wv",
            flags={"bf16": p.dtype == "bf16"},
        )

    def validate(self) -> None:
        p = self.problem
        if p.dtype not in ("fp16", "bf16"):
            raise ValueError(
                f"DirectDepthwiseDgradStreamSpec: unsupported dtype {p.dtype!r}; expected fp16 or bf16"
            )
        if p.cpg != 1 or p.kpg != 1:
            raise ValueError(
                f"DirectDepthwiseDgradStreamSpec requires cpg=kpg=1 (got {p.cpg}, {p.kpg})"
            )


def is_valid_depthwise_dgrad_stream_spec(
    spec: DirectDepthwiseDgradStreamSpec, arch: str = "gfx950"
) -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for a ho-streaming depthwise dgrad spec."""
    from rocke.core.arch import ArchTarget

    try:
        ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)
    p = spec.problem
    if p.dtype not in ("fp16", "bf16"):
        return (
            False,
            f"DirectDepthwiseDgradStreamSpec: unsupported dtype {p.dtype!r}; expected fp16 or bf16",
        )
    if p.cpg != 1 or p.kpg != 1:
        return False, f"requires cpg=kpg=1 (got {p.cpg}, {p.kpg})"
    return True, "ok"


def build_direct_depthwise_dgrad_streaming(
    spec: DirectDepthwiseDgradStreamSpec, arch: str = "gfx950"
) -> KernelDef:
    """Build the ho-streaming depthwise dgrad kernel.

    Streams dY rows (ho) in a Python-unrolled loop.  For each ho, all r-taps
    are evaluated (KH iterations, fully unrolled), contributing to hi values
    via circular accumulator slots.  W is preloaded into registers once before
    the loop.  The flush condition is computed at Python build time for each ho.

    Tensor roles:
      A param — dY: output gradient, shape [N, Ho, Wo, groups], NHWK
      B param — W:  weights,          shape [groups, KH, KW, 1], KRSC
      D param — dX: input gradient,   shape [N, H, W, groups],   NHWC

    Grid: (ceil(Wi / block_w), ceil(C / block_ch), N)
    """
    spec.validate()
    ok, why = is_valid_depthwise_dgrad_stream_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid DirectDepthwiseDgradStreamSpec for {arch}: {why}")

    p = spec.problem
    BLOCK_W = spec.block_w
    WAVE = spec.wave_size
    BLOCK_WAVES = spec.block_waves
    THREADS = BLOCK_WAVES * WAVE
    BLOCK_CH = BLOCK_WAVES * WAVE
    Ho = p.Ho
    Wo = p.Wo
    stride = p.stride

    # Circular accumulator depth: stride * KH slots covers all in-flight hi values.
    n_slots = stride * p.KH

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = THREADS

    io_type = _io_type(p.dtype)
    A = b.param("A", PtrType(io_type, "global"), noalias=True, readonly=True, align=16)
    Bp = b.param("B", PtrType(io_type, "global"), noalias=True, readonly=True, align=16)
    D = b.param("D", PtrType(io_type, "global"), noalias=True, writeonly=True, align=16)
    A_bytes = b.param("A_bytes", I32)
    B_bytes = b.param("B_bytes", I32)
    D_bytes = b.param("D_bytes", I32)

    c0 = b.const_i32(0)
    c_groups = b.const_i32(p.groups)
    c_half_bytes = b.const_i32(2)
    c_wave = b.const_i32(WAVE)
    oob_sentinel = b.const_i32((1 << 31) - 1)
    zero_f32 = b.const_f32(0.0)

    tid = b.thread_id_x()
    wave_id = b.div(tid, c_wave)
    lane = b.mod(tid, c_wave)

    bx = b.block_id_x()
    by = b.block_id_y()
    n = b.block_id_z()
    wi_tile_start = b.mul(bx, b.const_i32(BLOCK_W))
    ch = b.add(
        b.mul(by, b.const_i32(BLOCK_CH)),
        b.add(b.mul(wave_id, c_wave), lane),
    )
    ch_ok = b.cmp_lt(ch, c_groups)

    a_rsrc = b.buffer_rsrc(A, A_bytes)
    b_rsrc = b.buffer_rsrc(Bp, B_bytes)
    d_rsrc = b.buffer_rsrc(D, D_bytes)

    # dY descriptor: A[N, Ho, Wo, groups] NHWK.
    dy_desc = TensorDescriptor.naive(
        "A", lengths=[p.N, Ho, Wo, p.groups], coord_names=("n", "ho", "wo", "ch")
    )
    # W descriptor: B[groups, KH, KW, 1] KRSC.
    b_desc = TensorDescriptor.naive(
        "B", lengths=[p.groups, p.KH, p.KW, 1], coord_names=("k", "r", "s", "c")
    )
    # dX descriptor: D[N, H, W, groups] NHWC.
    d_desc = TensorDescriptor.naive(
        "D", lengths=[p.N, p.H, p.W, p.groups], coord_names=("n", "h", "w", "ch")
    )

    # Preload W[ch, r, s, 0] into f32 registers (KH*KW values per thread).
    weights_f32: List[List[Value]] = []
    for r_const in range(p.KH):
        row: List[Value] = []
        for s_const in range(p.KW):
            w_off, _ = b_desc.offset(
                b, k=ch, r=b.const_i32(r_const), s=b.const_i32(s_const), c=c0
            )
            safe_w = b.select(ch_ok, b.mul(w_off, c_half_bytes), oob_sentinel)
            w_h = (
                b.buffer_load_bf16(b_rsrc, safe_w, c0)
                if p.dtype == "bf16"
                else b.buffer_load_f16(b_rsrc, safe_w, c0)
            )
            row.append(b.select(ch_ok, b.cast_to_f32(w_h), zero_f32))
        weights_f32.append(row)

    # Circular accumulator slots: acc_slots[slot][j] for slot in [0, n_slots) and j in [0, BLOCK_W).
    # Each slot corresponds to hi % n_slots, accumulating one (hi, wi) pair's partial dX sum.
    acc_slots: List[List[Value]] = [[zero_f32] * BLOCK_W for _ in range(n_slots)]

    c_Wi = b.const_i32(p.W)
    c_Wo = b.const_i32(Wo)

    # Python-unrolled ho-streaming loop (Ho iterations = half of H for stride=2).
    for y in range(Ho):
        y_ho = y  # Python int — ho_iter

        # For each (r_const): compute hi = y*stride + r_const - PAD.
        # If hi is in [0, H): pre-load dY[n, y, *, ch] values needed for this (ho, r).
        # Then for each (s_const): find wo = (wi + PAD - s_const)//stride, check divisibility.

        for r_const in range(p.KH):
            hi_int = y_ho * stride + r_const - p.PAD  # Python int
            if not (0 <= hi_int < p.H):
                continue
            slot = hi_int % n_slots

            for s_const in range(p.KW):
                for j in range(BLOCK_W):
                    # wi = wi_tile_start + j (runtime value)
                    # wo = (wi + PAD - s_const) / stride — check divisibility
                    # Since wi_tile_start = bx * BLOCK_W (even when BLOCK_W even),
                    # parity of wi is determined by j.
                    # (wi + PAD - s_const) % stride: we check at runtime.
                    wi_rel_PAD_s = (
                        j + p.PAD - s_const
                    )  # Python int relative to wi_tile_start
                    # wi_tile_start is bx*BLOCK_W; we need (wi_tile_start + wi_rel_PAD_s) % stride.
                    # This is (bx*BLOCK_W + wi_rel_PAD_s) % stride.
                    # Since BLOCK_W must be divisible by stride for parity-based tiling to work,
                    # we check at runtime to handle all cases.

                    wi = b.add(wi_tile_start, b.const_i32(j))
                    wi_ok = b.cmp_lt(wi, c_Wi)

                    # Runtime divisibility and range check for wo.
                    wi_p_s = b.add(wi, b.const_i32(p.PAD - s_const))
                    if stride > 1:
                        c_st = b.const_i32(stride)
                        div_ok = b.cmp_eq(b.mod(wi_p_s, c_st), c0)
                        wo = b.div(wi_p_s, c_st)
                    else:
                        div_ok = None
                        wo = wi_p_s

                    wo_ok = b.land(b.cmp_ge(wi_p_s, c0), b.cmp_lt(wo, c_Wo))
                    if stride > 1:
                        tap_valid = b.land(b.land(div_ok, wo_ok), b.land(ch_ok, wi_ok))
                    else:
                        tap_valid = b.land(wo_ok, b.land(ch_ok, wi_ok))

                    dy_off, _ = dy_desc.offset(
                        b, n=n, ho=b.const_i32(y_ho), wo=wo, ch=ch
                    )
                    safe_dy = b.select(
                        tap_valid, b.mul(dy_off, c_half_bytes), oob_sentinel
                    )
                    dy_h = (
                        b.buffer_load_bf16(a_rsrc, safe_dy, c0)
                        if p.dtype == "bf16"
                        else b.buffer_load_f16(a_rsrc, safe_dy, c0)
                    )
                    dy_f32 = b.select(tap_valid, b.cast_to_f32(dy_h), zero_f32)

                    acc_slots[slot][j] = b.fma(
                        weights_f32[r_const][s_const], dy_f32, acc_slots[slot][j]
                    )

        # Flush complete hi values after this ho.
        # hi is complete when y (= ho) equals y_last(hi) = min(Ho-1, (hi+PAD)//stride).
        # Case 1 (y < Ho-1): flush hi in [y*stride - PAD, (y+1)*stride - 1 - PAD] ∩ [0, H).
        # Case 2 (y == Ho-1): flush hi in [(Ho-1)*stride - PAD, H).
        if y_ho < Ho - 1:
            flush_start = y_ho * stride - p.PAD
            flush_end = min(p.H, (y_ho + 1) * stride - p.PAD)
        else:
            flush_start = (Ho - 1) * stride - p.PAD
            flush_end = p.H

        for hi_flush in range(max(0, flush_start), flush_end):
            slot = hi_flush % n_slots
            for j in range(BLOCK_W):
                wi = b.add(wi_tile_start, b.const_i32(j))
                wi_ok = b.cmp_lt(wi, c_Wi)
                d_off, _ = d_desc.offset(b, n=n, h=b.const_i32(hi_flush), w=wi, ch=ch)
                safe_d = b.select(
                    b.land(ch_ok, wi_ok), b.mul(d_off, c_half_bytes), oob_sentinel
                )
                if p.dtype == "bf16":
                    b.buffer_store_bf16(
                        d_rsrc, safe_d, c0, b.trunc_f32_to_bf16(acc_slots[slot][j])
                    )
                else:
                    b.buffer_store_f16(
                        d_rsrc, safe_d, c0, b.trunc_f32_to_f16(acc_slots[slot][j])
                    )
                # Reset slot for future use.
                acc_slots[slot][j] = zero_f32

    return b.kernel


# ---------------------------------------------------------------------------
# Depthwise convolution kernel — cpg = kpg = 1, groups = C = K
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DirectDepthwiseSpec:
    """Direct depthwise convolution kernel for ``cpg = kpg = 1`` (groups == C == K).

    Each lane owns one channel for the duration of the kernel; no cross-channel
    communication or MFMA is needed.  The inner loop is a scalar
    multiply-accumulate (``fma``) over ``KH × KW`` filter taps.

    Kernel structure:
      - ``block_waves`` waves of 64 threads share one workgroup.
      - Lane ``l`` in wave ``w`` handles channel
        ``by * block_ch + w * 64 + l``.
      - Weights (``KH × KW`` values per channel) are preloaded into registers
        before the H-streaming loop.
      - H-streaming loop (Python-level unroll, ``H + KH - 1`` iterations):
        circular ``KH``-slot accumulators per output W position, flushed to
        global memory one output row at a time.  For stride > 1 only input
        rows that align to an output position produce a write.

    Block geometry:
      ``threads_per_block = block_waves * 64``
      Grid: ``(ceil(Wo / block_w), ceil(C / block_ch), N)``
    """

    problem: DirectConvProblem
    name: str = "direct_depthwise"
    block_w: int = 8  # output W positions per block
    block_waves: int = 1  # waves per block
    wave_size: int = 64

    @property
    def threads_per_block(self) -> int:
        return self.block_waves * self.wave_size

    @property
    def block_ch(self) -> int:
        return self.block_waves * self.wave_size

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        p = self.problem
        return kernel_name_join(
            self.name,
            p.short(),
            f"bw{self.block_w}",
            f"bw{self.block_waves}wv",
            flags={"bf16": p.dtype == "bf16"},
        )

    def validate(self) -> None:
        p = self.problem
        if p.dtype not in ("fp16", "bf16"):
            raise ValueError(
                f"DirectDepthwiseSpec: unsupported dtype {p.dtype!r}; expected fp16 or bf16"
            )
        if p.cpg != 1 or p.kpg != 1:
            raise ValueError(
                f"DirectDepthwiseSpec requires cpg=kpg=1 (got cpg={p.cpg}, kpg={p.kpg})"
            )


def is_valid_depthwise_spec(
    spec: "DirectDepthwiseSpec", arch: str = "gfx950"
) -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for a :class:`DirectDepthwiseSpec` on ``arch``.

    Only validates geometry constraints; no MFMA atom check is needed because
    the kernel uses only scalar ``fma`` operations.
    """
    from rocke.core.arch import ArchTarget

    try:
        ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)

    p = spec.problem
    if p.dtype not in ("fp16", "bf16"):
        return (
            False,
            f"DirectDepthwiseSpec: unsupported dtype {p.dtype!r}; expected fp16 or bf16",
        )
    if p.cpg != 1 or p.kpg != 1:
        return False, f"cpg and kpg must both be 1 (got cpg={p.cpg}, kpg={p.kpg})"
    return True, "ok"


# Shared unroll threshold for both depthwise builders.  When the static cost
# (loop iterations × filter taps) exceeds this, a runtime scf.for_iter loop
# is emitted instead of fully unrolling; see _use_unroll below.  Note that
# build_direct_depthwise multiplies by BLOCK_W (each thread covers multiple
# output W positions) while build_direct_depthwise_spatial does not (each
# thread owns exactly one W position).
_DW_UNROLL_THRESH = 20_000


def build_direct_depthwise(
    spec: "DirectDepthwiseSpec", arch: str = "gfx950"
) -> KernelDef:
    """Build the IR for a scalar depthwise convolution kernel.

    Supports any ``groups = C = K`` shape where ``cpg = kpg = 1``.

    Each wave of 64 lanes processes 64 channels independently.  Weights
    (``KH × KW`` fp16 values per lane/channel) are hoisted into registers
    before the H-streaming loop.  Per output position the kernel issues
    ``KH × KW`` scalar ``buffer_load_f16`` + ``fma`` operations.

    The H-streaming pipeline is identical in structure to the grouped direct
    conv kernels: ``KH`` circular accumulator slots per output W position are
    drained one row at a time as the outer H loop advances.
    """
    spec.validate()
    ok, why = is_valid_depthwise_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid DirectDepthwiseSpec for {arch}: {why}")

    from rocke.core.ir import F32

    p = spec.problem
    BLOCK_W = spec.block_w
    BLOCK_WAVES = spec.block_waves
    WAVE = spec.wave_size
    THREADS = spec.threads_per_block
    BLOCK_CH = spec.block_ch
    Ho = p.Ho
    Wo = p.Wo
    c_stride_dw = p.stride

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = THREADS

    io_type = _io_type(p.dtype)
    A = b.param("A", PtrType(io_type, "global"), noalias=True, readonly=True, align=16)
    Bp = b.param("B", PtrType(io_type, "global"), noalias=True, readonly=True, align=16)
    D = b.param("D", PtrType(io_type, "global"), noalias=True, writeonly=True, align=16)
    A_bytes = b.param("A_bytes", I32)
    B_bytes = b.param("B_bytes", I32)
    D_bytes = b.param("D_bytes", I32)

    c0 = b.const_i32(0)
    c_wave = b.const_i32(WAVE)
    c_W = b.const_i32(Wo)
    c_groups = b.const_i32(p.groups)
    c_half_bytes = b.const_i32(2)
    oob_sentinel = b.const_i32((1 << 31) - 1)
    zero_f32 = b.const_f32(0.0)

    tid = b.thread_id_x()
    wave_id = b.div(tid, c_wave)
    lane = b.mod(tid, c_wave)

    # Grid layout: bx = W-tile, by = channel-tile, bz = batch.
    bx = b.block_id_x()
    by = b.block_id_y()
    n = b.block_id_z()
    q_tile_start = b.mul(bx, b.const_i32(BLOCK_W))
    # Absolute channel for this lane: by*BLOCK_CH + wave_id*WAVE + lane.
    ch = b.add(
        b.mul(by, b.const_i32(BLOCK_CH)),
        b.add(b.mul(wave_id, c_wave), lane),
    )
    # Guard: lanes beyond groups are inactive (partial last tile).
    ch_in_range = b.cmp_lt(ch, c_groups)

    a_rsrc = b.buffer_rsrc(A, A_bytes)
    b_rsrc = b.buffer_rsrc(Bp, B_bytes)
    d_rsrc = b.buffer_rsrc(D, D_bytes)

    # A[N, H, W, C] NHWC descriptor with h and w boundary embeds.
    a_desc = TensorDescriptor.naive(
        "A",
        lengths=[p.N, p.H, p.W, p.total_c],
        coord_names=("n", "h", "w", "c"),
    ).transform(
        embed(
            upper=("y_iter",),
            into="h",
            strides=(1,),
            offset=-p.PAD,
            lo=0,
            hi=p.H,
        ),
        embed(
            upper=("wo", "s_off"),
            into="w",
            strides=(p.stride, 1),
            offset=-p.PAD,
            lo=0,
            hi=p.W,
        ),
    )

    # B[total_k, KH, KW, 1] KRSC descriptor (cpg=1: last dim is always 0).
    b_desc = TensorDescriptor.naive(
        "B",
        lengths=[p.total_k, p.KH, p.KW, 1],
        coord_names=("k", "r", "s", "c"),
    )

    # D[N, Ho, Wo, total_k] NHWK descriptor.
    d_desc = TensorDescriptor.naive(
        "D",
        lengths=[p.N, Ho, Wo, p.total_k],
        coord_names=("n", "h", "w", "k"),
    )

    # ---- Preload weights into registers (KH * KW fp16 → f32 values per lane) ----
    # Each lane owns one channel (ch), so weight[ch, r, s, 0] is a scalar.
    # Lanes beyond groups (partial last tile) load from OOB sentinel → zero.
    weights_f32: List[List[Value]] = []
    for r_const in range(p.KH):
        row: List[Value] = []
        for s_const in range(p.KW):
            w_off, _ = b_desc.offset(
                b,
                k=ch,
                r=b.const_i32(r_const),
                s=b.const_i32(s_const),
                c=c0,
            )
            safe_w_off = b.select(ch_in_range, b.mul(w_off, c_half_bytes), oob_sentinel)
            w_h = (
                b.buffer_load_bf16(b_rsrc, safe_w_off, c0)
                if p.dtype == "bf16"
                else b.buffer_load_f16(b_rsrc, safe_w_off, c0)
            )
            w_f32 = b.select(ch_in_range, b.cast_to_f32(w_h), zero_f32)
            row.append(w_f32)
        weights_f32.append(row)

    # ---- Accumulator array ----
    acc: List[List[Value]] = [[zero_f32] * p.KH for _ in range(BLOCK_W)]

    # ---- H-streaming loop ----
    # Below _DW_UNROLL_THRESH the loop is Python-unrolled (best codegen).
    # Above it a runtime grouped-period scf.for is used: the outer loop
    # runs n_groups = ceil(n_iters/KH) times; the inner KH steps are
    # Python-unrolled with STATIC slot indices so preloaded weights are
    # referenced directly (no scatter, no runtime weight loads).
    n_iters = p.H + p.KH - 1
    _use_unroll = n_iters * BLOCK_W * p.KH * p.KW <= _DW_UNROLL_THRESH

    if _use_unroll:
        for y in range(n_iters):
            y_i = b.const_i32(y)
            for w_out in range(BLOCK_W):
                w_pos = b.add(q_tile_start, b.const_i32(w_out))
                for s_const in range(p.KW):
                    a_off, valid = a_desc.offset(
                        b,
                        n=n,
                        y_iter=y_i,
                        wo=w_pos,
                        s_off=b.const_i32(s_const),
                        c=ch,
                    )
                    load_ok = b.land(valid, ch_in_range)
                    safe_off = b.select(
                        load_ok, b.mul(a_off, c_half_bytes), oob_sentinel
                    )
                    a_h = (
                        b.buffer_load_bf16(a_rsrc, safe_off, c0)
                        if p.dtype == "bf16"
                        else b.buffer_load_f16(a_rsrc, safe_off, c0)
                    )
                    a_f32 = b.select(load_ok, b.cast_to_f32(a_h), zero_f32)
                    for r_const in range(p.KH):
                        p_idx = (y - r_const + p.KH) % p.KH
                        acc[w_out][p_idx] = b.fma(
                            weights_f32[r_const][s_const], a_f32, acc[w_out][p_idx]
                        )

            p_flush_val = y - (p.KH - 1)
            P_FLUSH = p_flush_val % p.KH
            if 0 <= p_flush_val < p.H and p_flush_val % c_stride_dw == 0:
                ho_row = p_flush_val // c_stride_dw
                if ho_row >= Ho:
                    continue
                for w_out in range(BLOCK_W):
                    out_q = b.add(q_tile_start, b.const_i32(w_out))
                    out_q_ok = b.land(b.cmp_lt(out_q, c_W), ch_in_range)
                    d_off, _ = d_desc.offset(
                        b, n=n, h=b.const_i32(ho_row), w=out_q, k=ch
                    )
                    safe_d = b.select(
                        out_q_ok, b.mul(d_off, c_half_bytes), oob_sentinel
                    )
                    if p.dtype == "bf16":
                        b.buffer_store_bf16(
                            d_rsrc, safe_d, c0, b.trunc_f32_to_bf16(acc[w_out][P_FLUSH])
                        )
                    else:
                        b.buffer_store_f16(
                            d_rsrc, safe_d, c0, b.trunc_f32_to_f16(acc[w_out][P_FLUSH])
                        )
            for w_out in range(BLOCK_W):
                acc[w_out][P_FLUSH] = zero_f32

    else:
        c1 = b.const_i32(1)
        c_KH = b.const_i32(p.KH)
        c_stride_rv = b.const_i32(c_stride_dw)
        n_groups = (n_iters + p.KH - 1) // p.KH

        dw_iter_args = [
            (f"dw_acc_kh{kh}_w{w}", zero_f32)
            for kh in range(p.KH)
            for w in range(BLOCK_W)
        ]
        group_loop = b.scf_for_iter(
            c0,
            b.const_i32(n_groups),
            c1,
            dw_iter_args,
            iv_name="dw_grp",
            elide_trailing_barrier=False,
        )
        with group_loop as (grp_iv, loop_accs):
            new_accs = list(loop_accs)

            for j in range(p.KH):
                y_j = b.add(b.mul(grp_iv, c_KH), b.const_i32(j))
                j_valid = b.cmp_lt(y_j, b.const_i32(n_iters))

                for w_out in range(BLOCK_W):
                    w_pos = b.add(q_tile_start, b.const_i32(w_out))
                    for s_const in range(p.KW):
                        a_off, valid = a_desc.offset(
                            b,
                            n=n,
                            y_iter=y_j,
                            wo=w_pos,
                            s_off=b.const_i32(s_const),
                            c=ch,
                        )
                        ok = b.land(b.land(valid, j_valid), ch_in_range)
                        safe_off = b.select(
                            ok, b.mul(a_off, c_half_bytes), oob_sentinel
                        )
                        a_h = (
                            b.buffer_load_bf16(a_rsrc, safe_off, c0)
                            if p.dtype == "bf16"
                            else b.buffer_load_f16(a_rsrc, safe_off, c0)
                        )
                        a_f32 = b.select(ok, b.cast_to_f32(a_h), zero_f32)
                        for r_const in range(p.KH):
                            p_idx = (j - r_const + p.KH) % p.KH  # STATIC slot
                            idx = p_idx * BLOCK_W + w_out
                            new_accs[idx] = b.fma(
                                weights_f32[r_const][s_const], a_f32, new_accs[idx]
                            )

                P_FLUSH_j = (j + 1) % p.KH  # STATIC
                p_flush_rv = b.add(y_j, b.const_i32(-(p.KH - 1)))

                if j >= p.KH - 1:
                    flush_ge = j_valid
                else:
                    flush_ge = b.land(b.cmp_lt(c0, grp_iv), j_valid)

                if c_stride_dw == 1:
                    should_flush = flush_ge
                else:
                    flush_stride = b.cmp_eq(b.mod(p_flush_rv, c_stride_rv), c0)
                    should_flush = b.land(flush_ge, flush_stride)

                ho_row_j = b.div(p_flush_rv, c_stride_rv)

                for w_out in range(BLOCK_W):
                    out_q = b.add(q_tile_start, b.const_i32(w_out))
                    out_q_ok = b.land(b.cmp_lt(out_q, c_W), ch_in_range)
                    store_ok = b.land(out_q_ok, should_flush)
                    acc_val = new_accs[P_FLUSH_j * BLOCK_W + w_out]  # STATIC index
                    d_off, _ = d_desc.offset(b, n=n, h=ho_row_j, w=out_q, k=ch)
                    safe_d = b.select(
                        store_ok, b.mul(d_off, c_half_bytes), oob_sentinel
                    )
                    if p.dtype == "bf16":
                        b.buffer_store_bf16(
                            d_rsrc, safe_d, c0, b.trunc_f32_to_bf16(acc_val)
                        )
                    else:
                        b.buffer_store_f16(
                            d_rsrc, safe_d, c0, b.trunc_f32_to_f16(acc_val)
                        )

                for w_out in range(BLOCK_W):
                    new_accs[P_FLUSH_j * BLOCK_W + w_out] = zero_f32

            b.scf_yield(*new_accs)

    return b.kernel


# ---------------------------------------------------------------------------
# Depthwise spatial kernel — groups ≤ wave_size: threads map to both
# channel AND output W-position within one wavefront.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DirectDepthwiseSpatialSpec:
    """Depthwise kernel for groups ≤ wave_size (small-group variant).

    Thread layout within each wavefront:
      ch        = t_in_wave % groups   → which channel this thread owns
      w_in_wave = t_in_wave // groups  → W-position offset within the wave

    Each wave covers ``n_w_per_wave = wave_size // groups`` output W
    positions for ALL channels simultaneously.  Thread utilisation:
    ``floor(wave_size/groups) * groups / wave_size``.

    For groups=3, wave_size=64: n_w=21, utilisation 63/64 = 98.4%.

    Block geometry:
      ``threads_per_block = block_waves * wave_size``
      ``block_w = block_waves * n_w_per_wave``
      Grid: ``(ceil(Wo / block_w), 1, N)``  — no channel tile.
    """

    problem: DirectConvProblem
    name: str = "direct_depthwise_spatial"
    block_waves: int = 1
    wave_size: int = 64

    @property
    def n_w_per_wave(self) -> int:
        return self.wave_size // self.problem.groups

    @property
    def block_w(self) -> int:
        return self.block_waves * self.n_w_per_wave

    @property
    def threads_per_block(self) -> int:
        return self.block_waves * self.wave_size

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        p = self.problem
        return kernel_name_join(
            self.name,
            p.short(),
            f"bwv{self.block_waves}",
            flags={"bf16": p.dtype == "bf16"},
        )

    def validate(self) -> None:
        p = self.problem
        if p.dtype not in ("fp16", "bf16"):
            raise ValueError(
                f"DirectDepthwiseSpatialSpec: unsupported dtype {p.dtype!r}; expected fp16 or bf16"
            )
        if p.cpg != 1 or p.kpg != 1:
            raise ValueError(
                f"DirectDepthwiseSpatialSpec requires cpg=kpg=1 "
                f"(got cpg={p.cpg}, kpg={p.kpg})"
            )
        if p.groups > self.wave_size:
            raise ValueError(
                f"groups {p.groups} > wave_size {self.wave_size}: "
                f"use DirectDepthwiseSpec instead"
            )
        if self.n_w_per_wave == 0:
            raise ValueError(
                f"groups={p.groups} == wave_size={self.wave_size}: no W positions per wave"
            )


def is_valid_depthwise_spatial_spec(
    spec: "DirectDepthwiseSpatialSpec", arch: str = "gfx950"
) -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for a :class:`DirectDepthwiseSpatialSpec`."""
    from rocke.core.arch import ArchTarget

    try:
        ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)

    p = spec.problem
    if p.dtype not in ("fp16", "bf16"):
        return (
            False,
            f"DirectDepthwiseSpatialSpec: unsupported dtype {p.dtype!r}; expected fp16 or bf16",
        )
    if p.cpg != 1 or p.kpg != 1:
        return False, f"cpg and kpg must both be 1 (got cpg={p.cpg}, kpg={p.kpg})"
    if p.groups > spec.wave_size:
        return False, f"groups {p.groups} > wave_size {spec.wave_size}"
    if spec.n_w_per_wave == 0:
        return False, f"groups={p.groups} == wave_size: no W positions per wave"
    return True, "ok"


def build_direct_depthwise_spatial(
    spec: "DirectDepthwiseSpatialSpec", arch: str = "gfx950"
) -> KernelDef:
    """Build the small-group depthwise spatial kernel.

    Thread layout: ``ch = t % groups``, ``w_local = t // groups``.
    Weights preloaded into registers.  Input loaded once per (j, s_const)
    and reused across all KH filter rows — no redundant memory traffic.
    """
    spec.validate()
    ok, why = is_valid_depthwise_spatial_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid DirectDepthwiseSpatialSpec for {arch}: {why}")

    p = spec.problem
    WAVE = spec.wave_size
    BLOCK_WAVES = spec.block_waves
    THREADS = spec.threads_per_block
    n_w = spec.n_w_per_wave
    BLOCK_W = spec.block_w
    Ho = p.Ho
    Wo = p.Wo
    c_stride_dw = p.stride

    n_iters = p.H + p.KH - 1
    _use_unroll = n_iters * p.KH * p.KW <= _DW_UNROLL_THRESH

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = THREADS

    io_type = _io_type(p.dtype)
    A = b.param("A", PtrType(io_type, "global"), noalias=True, readonly=True, align=16)
    Bp = b.param("B", PtrType(io_type, "global"), noalias=True, readonly=True, align=16)
    D = b.param("D", PtrType(io_type, "global"), noalias=True, writeonly=True, align=16)
    A_bytes = b.param("A_bytes", I32)
    B_bytes = b.param("B_bytes", I32)
    D_bytes = b.param("D_bytes", I32)

    c0 = b.const_i32(0)
    c_wave = b.const_i32(WAVE)
    c_Wo = b.const_i32(Wo)
    c_half_bytes = b.const_i32(2)
    oob_sentinel = b.const_i32((1 << 31) - 1)
    zero_f32 = b.const_f32(0.0)

    tid = b.thread_id_x()
    wave_id = b.div(tid, c_wave)
    t_in_wave = b.mod(tid, c_wave)

    ch = b.mod(t_in_wave, b.const_i32(p.groups))
    w_in_wave = b.div(t_in_wave, b.const_i32(p.groups))

    bx = b.block_id_x()
    n = b.block_id_z()

    q_out = b.add(
        b.mul(bx, b.const_i32(BLOCK_W)),
        b.add(b.mul(wave_id, b.const_i32(n_w)), w_in_wave),
    )

    # Guard: wasted threads when groups * n_w < wave_size
    w_valid = b.cmp_lt(w_in_wave, b.const_i32(n_w))
    q_ok = b.land(b.cmp_lt(q_out, c_Wo), w_valid)

    a_rsrc = b.buffer_rsrc(A, A_bytes)
    b_rsrc = b.buffer_rsrc(Bp, B_bytes)
    d_rsrc = b.buffer_rsrc(D, D_bytes)

    a_desc = TensorDescriptor.naive(
        "A",
        lengths=[p.N, p.H, p.W, p.total_c],
        coord_names=("n", "h", "w", "c"),
    ).transform(
        embed(upper=("y_iter",), into="h", strides=(1,), offset=-p.PAD, lo=0, hi=p.H),
        embed(
            upper=("wo", "s_off"),
            into="w",
            strides=(p.stride, 1),
            offset=-p.PAD,
            lo=0,
            hi=p.W,
        ),
    )
    b_desc = TensorDescriptor.naive(
        "B", lengths=[p.total_k, p.KH, p.KW, 1], coord_names=("k", "r", "s", "c")
    )
    d_desc = TensorDescriptor.naive(
        "D", lengths=[p.N, Ho, Wo, p.total_k], coord_names=("n", "h", "w", "k")
    )

    # Preload weights: KH * KW f32 per thread (one channel each).
    weights_f32: List[List[Value]] = []
    for r_const in range(p.KH):
        row: List[Value] = []
        for s_const in range(p.KW):
            w_off, _ = b_desc.offset(
                b, k=ch, r=b.const_i32(r_const), s=b.const_i32(s_const), c=c0
            )
            safe_w = b.select(w_valid, b.mul(w_off, c_half_bytes), oob_sentinel)
            w_h = (
                b.buffer_load_bf16(b_rsrc, safe_w, c0)
                if p.dtype == "bf16"
                else b.buffer_load_f16(b_rsrc, safe_w, c0)
            )
            row.append(b.select(w_valid, b.cast_to_f32(w_h), zero_f32))
        weights_f32.append(row)

    if _use_unroll:
        acc: List[Value] = [zero_f32] * p.KH

        for y in range(n_iters):
            y_i = b.const_i32(y)
            for s_const in range(p.KW):
                a_off, valid = a_desc.offset(
                    b, n=n, y_iter=y_i, wo=q_out, s_off=b.const_i32(s_const), c=ch
                )
                ok = b.land(valid, q_ok)
                safe_off = b.select(ok, b.mul(a_off, c_half_bytes), oob_sentinel)
                a_h = (
                    b.buffer_load_bf16(a_rsrc, safe_off, c0)
                    if p.dtype == "bf16"
                    else b.buffer_load_f16(a_rsrc, safe_off, c0)
                )
                a_f32 = b.select(ok, b.cast_to_f32(a_h), zero_f32)
                for r_const in range(p.KH):
                    p_idx = (y - r_const + p.KH) % p.KH
                    acc[p_idx] = b.fma(weights_f32[r_const][s_const], a_f32, acc[p_idx])

            p_flush_val = y - (p.KH - 1)
            P_FLUSH = p_flush_val % p.KH
            if 0 <= p_flush_val < p.H and p_flush_val % c_stride_dw == 0:
                ho_row = p_flush_val // c_stride_dw
                if ho_row < Ho:
                    d_off, _ = d_desc.offset(
                        b, n=n, h=b.const_i32(ho_row), w=q_out, k=ch
                    )
                    safe_d = b.select(q_ok, b.mul(d_off, c_half_bytes), oob_sentinel)
                    if p.dtype == "bf16":
                        b.buffer_store_bf16(
                            d_rsrc, safe_d, c0, b.trunc_f32_to_bf16(acc[P_FLUSH])
                        )
                    else:
                        b.buffer_store_f16(
                            d_rsrc, safe_d, c0, b.trunc_f32_to_f16(acc[P_FLUSH])
                        )
            acc[P_FLUSH] = zero_f32

    else:
        c1 = b.const_i32(1)
        c_KH = b.const_i32(p.KH)
        c_stride_rv = b.const_i32(c_stride_dw)
        n_groups = (n_iters + p.KH - 1) // p.KH

        iter_args = [(f"sp_acc_{kh}", zero_f32) for kh in range(p.KH)]
        group_loop = b.scf_for_iter(
            c0,
            b.const_i32(n_groups),
            c1,
            iter_args,
            iv_name="sp_grp",
            elide_trailing_barrier=False,
        )
        with group_loop as (grp_iv, loop_accs):
            new_accs = list(loop_accs)

            for j in range(p.KH):
                y_j = b.add(b.mul(grp_iv, c_KH), b.const_i32(j))
                j_valid = b.cmp_lt(y_j, b.const_i32(n_iters))

                for s_const in range(p.KW):
                    a_off, valid = a_desc.offset(
                        b, n=n, y_iter=y_j, wo=q_out, s_off=b.const_i32(s_const), c=ch
                    )
                    ok = b.land(b.land(valid, j_valid), q_ok)
                    safe_off = b.select(ok, b.mul(a_off, c_half_bytes), oob_sentinel)
                    a_h = (
                        b.buffer_load_bf16(a_rsrc, safe_off, c0)
                        if p.dtype == "bf16"
                        else b.buffer_load_f16(a_rsrc, safe_off, c0)
                    )
                    a_f32 = b.select(ok, b.cast_to_f32(a_h), zero_f32)
                    for r_const in range(p.KH):
                        p_idx = (j - r_const + p.KH) % p.KH  # STATIC
                        new_accs[p_idx] = b.fma(
                            weights_f32[r_const][s_const], a_f32, new_accs[p_idx]
                        )

                P_FLUSH_j = (j + 1) % p.KH  # STATIC
                p_flush_rv = b.add(y_j, b.const_i32(-(p.KH - 1)))

                if j >= p.KH - 1:
                    flush_ge = j_valid
                else:
                    flush_ge = b.land(b.cmp_lt(c0, grp_iv), j_valid)

                if c_stride_dw == 1:
                    should_flush = flush_ge
                else:
                    flush_stride = b.cmp_eq(b.mod(p_flush_rv, c_stride_rv), c0)
                    should_flush = b.land(flush_ge, flush_stride)

                ho_row_j = b.div(p_flush_rv, c_stride_rv)
                store_ok = b.land(q_ok, should_flush)

                acc_val = new_accs[P_FLUSH_j]  # STATIC index
                d_off, _ = d_desc.offset(b, n=n, h=ho_row_j, w=q_out, k=ch)
                safe_d = b.select(store_ok, b.mul(d_off, c_half_bytes), oob_sentinel)
                if p.dtype == "bf16":
                    b.buffer_store_bf16(
                        d_rsrc, safe_d, c0, b.trunc_f32_to_bf16(acc_val)
                    )
                else:
                    b.buffer_store_f16(d_rsrc, safe_d, c0, b.trunc_f32_to_f16(acc_val))

                new_accs[P_FLUSH_j] = zero_f32  # unconditional static reset

            b.scf_yield(*new_accs)

    return b.kernel

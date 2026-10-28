# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Configurable gfx1250 MX FP4 GEMM with double-buffered TDM/LDS staging."""

from __future__ import annotations

from dataclasses import dataclass

from ...core.arch import ArchTarget
from ...core.ir import (
    F32,
    FP8E4M3,
    I8,
    I32,
    IRBuilder,
    KernelDef,
    PtrType,
)
from ...helpers.spec import SignatureBuilder, ceil_div_grid, kernel_name_join

_ATOM_M = 32
_ATOM_N = 16
_ATOM_K = 128
_WAVE = 32
_ACCUMULATORS_PER_ATOM = 16
_OP_ID = "wmma_scale_f32_32x16x128_f4"


class _WmmaScaleF4Atom:
    op_id = _OP_ID
    c_frag_len = _ACCUMULATORS_PER_ATOM
    c_dtype = "fp32"


@dataclass(frozen=True)
class MxFp4TdmGemmSpec:
    """RCR MX FP4 GEMM: ``D = alpha * A @ B.T + beta * C``."""

    name: str
    M: int
    N: int
    K: int
    tile_m: int = 128
    tile_n: int = 64
    tile_k: int = 128
    warp_m: int = 2
    warp_n: int = 2
    warp_k: int = 1
    warp_tile_m: int = _ATOM_M
    warp_tile_n: int = _ATOM_N
    warp_tile_k: int = _ATOM_K
    block_k: int = 32
    tdm_lds: bool = True
    tdm_prefetch: bool = True
    tdm_prefetch_depth: int = 2
    # Step 6 hipBLASLt scheduling flags (default off). Each lands in its own
    # substep; see mxfp4_tdm_gemm_variants.MXFP4_GEMM_VARIANTS.
    local_read_prefetch: bool = False
    prefetch_gl2: bool = False
    scale_double_buffer: bool = False
    wave_separated_tdm: bool = False
    scale_tdm: bool = False
    lds_pad: bool = False
    lds_segment_interleave: bool = False
    lds_segment_b_rotate: bool = False
    tdm_plus_lds_buf: bool = False
    init_c_wmma: bool = False
    cluster_dim: tuple[int, int, int] | None = None

    @property
    def wave_repeat_m(self) -> int:
        return self.tile_m // (self.warp_m * self.warp_tile_m)

    @property
    def wave_repeat_n(self) -> int:
        return self.tile_n // (self.warp_n * self.warp_tile_n)

    @property
    def block_size(self) -> int:
        return _WAVE * self.warp_m * self.warp_n * self.warp_k

    @property
    def accumulator_values_per_lane(self) -> int:
        return self.wave_repeat_m * self.wave_repeat_n * _ACCUMULATORS_PER_ATOM

    @property
    def packed_k(self) -> int:
        return self.K // 2

    @property
    def a_tile_bytes(self) -> int:
        return self.tile_m * self.a_lds_row_stride_bytes

    @property
    def b_tile_bytes(self) -> int:
        return self.tile_n * self.b_lds_row_stride_bytes

    @property
    def a_scale_tile_bytes(self) -> int:
        """Row-major rocKE staging: tile_m × (tile_k/32) E8M0 bytes."""
        return self.tile_m * (self.tile_k // 32)

    @property
    def b_scale_tile_bytes(self) -> int:
        """Row-major rocKE staging: (tile_k/32) × tile_n E8M0 bytes."""
        return (self.tile_k // 32) * self.tile_n

    @property
    def effective_prefetch_depth(self) -> int:
        if self.tdm_plus_lds_buf and self.tdm_prefetch_depth < 3:
            return 3
        return self.tdm_prefetch_depth

    @property
    def lds_bytes(self) -> int:
        depth = self.effective_prefetch_depth
        ab = depth * (self.a_tile_bytes + self.b_tile_bytes)
        if self.scale_tdm:
            ab += depth * (self.a_scale_tile_bytes + self.b_scale_tile_bytes)
        return ab

    @property
    def a_lds_row_stride_bytes(self) -> int:
        """Packed-K bytes per A row in LDS, including optional pad."""
        base = self.tile_k // 2
        return base + (16 if self.lds_pad else 0)

    @property
    def b_lds_row_stride_bytes(self) -> int:
        base = self.tile_k // 2
        return base + (16 if self.lds_pad else 0)

    def kernel_name(self) -> str:
        cluster_on = self.cluster_dim is not None
        return kernel_name_join(
            self.name,
            "mxfp4_tdm",
            f"M{self.M}N{self.N}K{self.K}",
            f"t{self.tile_m}x{self.tile_n}x{self.tile_k}",
            f"w{self.warp_m}x{self.warp_n}x{self.warp_k}",
            f"r{self.wave_repeat_m}x{self.wave_repeat_n}",
            flags={
                "db": self.effective_prefetch_depth == 2,
                "db3": self.effective_prefetch_depth == 3,
                "lrp": self.local_read_prefetch,
                "pgl2": self.prefetch_gl2,
                "sdb": self.scale_double_buffer,
                "wstdm": self.wave_separated_tdm,
                "stdm": self.scale_tdm,
                "ldsp": self.lds_pad,
                "ldssi": self.lds_segment_interleave,
                "ldsbr": self.lds_segment_b_rotate,
                "initc": self.init_c_wmma,
                "clu": cluster_on,
            },
        )


# Step 6 flags that are declared but not yet wired into the builder.
# Each landed substep removes its flag from this frozenset.
_UNIMPLEMENTED_STEP6_FLAGS = frozenset(
    {
        "lds_pad",
        "lds_segment_interleave",
        "lds_segment_b_rotate",
        "tdm_plus_lds_buf",
        "init_c_wmma",
        "cluster_dim",
    }
)


def _unimplemented_step6_flags(spec: MxFp4TdmGemmSpec) -> str | None:
    active: list[str] = []
    for name in sorted(_UNIMPLEMENTED_STEP6_FLAGS):
        value = getattr(spec, name)
        if name == "cluster_dim":
            if value is not None:
                active.append(name)
        elif value:
            active.append(name)
    if not active:
        return None
    return ", ".join(active)


def is_valid_spec(spec: MxFp4TdmGemmSpec, arch: str = "gfx1250") -> tuple[bool, str]:
    if arch != "gfx1250":
        return False, f"mxfp4_tdm_gemm is gfx1250-only (got {arch!r})"
    if min(spec.M, spec.N, spec.K) <= 0:
        return False, "M, N, and K must be positive"
    if not spec.tdm_lds or not spec.tdm_prefetch:
        return False, "Step 4 requires TDM/LDS staging with prefetch enabled"
    if spec.tdm_prefetch_depth not in (2, 3):
        return False, "tdm_prefetch_depth must be 2 or 3"
    if spec.tdm_prefetch_depth == 3 and not spec.tdm_plus_lds_buf:
        return False, "depth 3 requires tdm_plus_lds_buf"
    if spec.tdm_plus_lds_buf and spec.tdm_prefetch_depth not in (2, 3):
        return False, "tdm_plus_lds_buf expects depth 2 (auto-escalates) or 3"
    unimplemented = _unimplemented_step6_flags(spec)
    if unimplemented:
        return False, f"Step 6 flag not implemented yet: {unimplemented}"
    if spec.lds_segment_b_rotate and not spec.lds_segment_interleave:
        return False, "lds_segment_b_rotate requires lds_segment_interleave"
    if spec.scale_tdm and not spec.scale_double_buffer:
        return False, "scale_tdm requires scale_double_buffer"
    if spec.wave_separated_tdm:
        num_waves = spec.warp_m * spec.warp_n * spec.warp_k
        if spec.tile_m % num_waves or spec.tile_n % num_waves:
            return False, (
                "wave_separated_tdm requires tile_m and tile_n divisible by "
                f"num_waves ({num_waves})"
            )
    if spec.cluster_dim is not None:
        if len(spec.cluster_dim) != 3 or any(d < 1 for d in spec.cluster_dim):
            return False, "cluster_dim must be a positive (x, y, z) triple"
        target_probe = ArchTarget.from_gfx("gfx1250")
        if not target_probe.memory.has_cluster_launch:
            return False, "gfx1250 reports no cluster launch support"
    if spec.warp_k != 1:
        return False, "K-split waves are not supported (warp_k must be 1)"
    if (spec.warp_tile_m, spec.warp_tile_n, spec.warp_tile_k) != (
        _ATOM_M,
        _ATOM_N,
        _ATOM_K,
    ):
        return False, "the WMMA atom is fixed at 32x16x128"
    if spec.block_k != 32:
        return False, "E8M0 scales require block_k=32"
    if min(spec.tile_m, spec.tile_n, spec.tile_k) <= 0:
        return False, "tile dimensions must be positive"
    if spec.tile_m % (spec.warp_m * _ATOM_M):
        return False, "tile_m must be divisible by warp_m * 32"
    if spec.tile_n % (spec.warp_n * _ATOM_N):
        return False, "tile_n must be divisible by warp_n * 16"
    if spec.tile_k % _ATOM_K:
        return False, "tile_k must be divisible by 128"
    if spec.M % spec.tile_m or spec.N % spec.tile_n or spec.K % spec.tile_k:
        return False, "M/N/K must divide their selected workgroup tile"
    if spec.block_size > 1024:
        return False, f"workgroup has too many threads ({spec.block_size})"
    target = ArchTarget.from_gfx("gfx1250")
    vgpr_cap = target.limits.vgprs
    # Accumulators alone must fit the arch VGPR file (gfx1250 MaxVgpr=1024).
    if spec.accumulator_values_per_lane > vgpr_cap:
        return False, (
            "accumulator tile exceeds gfx1250 VGPR budget: "
            f"{spec.accumulator_values_per_lane} FP32 values per lane "
            f"> {vgpr_cap} MaxVgpr"
        )
    lds_cap = target.lds_capacity_bytes
    if spec.lds_bytes > lds_cap:
        return False, (
            f"double-buffered A/B tiles need too much LDS "
            f"({spec.lds_bytes} B > {lds_cap} cap)"
        )
    return True, "ok"


def mxfp4_tdm_gemm_signature(spec: MxFp4TdmGemmSpec) -> list[dict]:
    return (
        SignatureBuilder()
        .ptr("A", "i8")
        .ptr("B", "i8")
        .ptr("A_scale", "i8")
        .ptr("B_scale", "i8")
        .ptr("C", "fp8e4m3")
        .ptr("D", "fp8e4m3")
        .scalar("M", "i32")
        .scalar("N", "i32")
        .scalar("K", "i32")
        .scalar("alpha", "f32")
        .scalar("beta", "f32")
        .build()
    )


def mxfp4_tdm_gemm_grid(spec: MxFp4TdmGemmSpec) -> tuple[int, int, int]:
    return ceil_div_grid((spec.N, spec.tile_n), (spec.M, spec.tile_m))


def build_mxfp4_tdm_gemm(spec: MxFp4TdmGemmSpec, *, arch: str = "gfx1250") -> KernelDef:
    """Build the double-buffered TDM/LDS MX FP4 GEMM."""
    ok, reason = is_valid_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid mxfp4_tdm_gemm spec: {reason}")

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block_size

    A = b.param("A", PtrType(I8, "global"), noalias=True, readonly=True, align=16)
    B = b.param("B", PtrType(I8, "global"), noalias=True, readonly=True, align=16)
    AS = b.param("A_scale", PtrType(I8, "global"), noalias=True, readonly=True, align=4)
    BS = b.param("B_scale", PtrType(I8, "global"), noalias=True, readonly=True, align=4)
    C = b.param("C", PtrType(FP8E4M3, "global"), noalias=True, readonly=True, align=1)
    D = b.param("D", PtrType(FP8E4M3, "global"), noalias=True, writeonly=True, align=1)
    M = b.param("M", I32)
    N = b.param("N", I32)
    K = b.param("K", I32)
    alpha = b.param("alpha", F32)
    beta = b.param("beta", F32)

    c0 = b.const_i32(0)
    c1 = b.const_i32(1)
    c16 = b.const_i32(16)
    c32 = b.const_i32(32)
    packed_k = b.div(K, b.const_i32(2))
    scale_groups = b.div(K, c32)

    tid = b.thread_id_x()
    lane = b.mod(tid, c32)
    lane_col = b.mod(lane, c16)
    lane_half = b.div(lane, c16)
    wave = b.div(tid, c32)
    wave_n = b.mod(wave, b.const_i32(spec.warp_n))
    wave_m = b.div(wave, b.const_i32(spec.warp_n))

    block_m = b.mul(b.block_id_y(), b.const_i32(spec.tile_m))
    block_n = b.mul(b.block_id_x(), b.const_i32(spec.tile_n))

    shared = b.smem_alloc(I8, [spec.lds_bytes], name_hint="ab_tdm")
    shared_addr = b.smem_addr_of(shared)
    depth = spec.effective_prefetch_depth
    b_region = depth * spec.a_tile_bytes
    scale_region = depth * (spec.a_tile_bytes + spec.b_tile_bytes)
    b_scale_region = scale_region + depth * spec.a_scale_tile_bytes

    def _buffer_offsets(parity):
        a_offset = b.mul(parity, b.const_i32(spec.a_tile_bytes))
        b_offset = b.add(
            b.const_i32(b_region),
            b.mul(parity, b.const_i32(spec.b_tile_bytes)),
        )
        if not spec.scale_tdm:
            return a_offset, b_offset, None, None
        a_scale_offset = b.add(
            b.const_i32(scale_region),
            b.mul(parity, b.const_i32(spec.a_scale_tile_bytes)),
        )
        b_scale_offset = b.add(
            b.const_i32(b_scale_region),
            b.mul(parity, b.const_i32(spec.b_scale_tile_bytes)),
        )
        return a_offset, b_offset, a_scale_offset, b_scale_offset

    def _issue_tdm(k0, parity) -> None:
        a_lds_offset, b_lds_offset, a_scale_lds, b_scale_lds = _buffer_offsets(parity)
        k_byte = b.div(k0, b.const_i32(2))
        remaining_k = b.sub(packed_k, k_byte)
        a_row_stride = b.const_i32(spec.a_lds_row_stride_bytes)
        scale_k_groups = spec.tile_k // 32
        group0 = b.div(k0, c32)
        remaining_groups = b.sub(scale_groups, group0)

        def _issue_ab_strip(a_row0, a_rows, b_row0, b_rows) -> None:
            a_lds = b.smem_ptr_add(
                shared_addr,
                b.zext(
                    b.add(a_lds_offset, b.mul(a_row0, a_row_stride)),
                    shared_addr.type,
                ),
            )
            b_lds = b.smem_ptr_add(
                shared_addr,
                b.zext(
                    b.add(b_lds_offset, b.mul(b_row0, a_row_stride)),
                    shared_addr.type,
                ),
            )
            a_element = b.add(
                b.mul(b.add(block_m, a_row0), packed_k),
                k_byte,
            )
            b_element = b.add(
                b.mul(b.add(block_n, b_row0), packed_k),
                k_byte,
            )
            b.tensor_load_to_lds(
                *b.tdm_descriptor_2d(
                    b.global_ptr_add(A, a_element),
                    a_lds,
                    elem_bytes=1,
                    tensor_dim0=remaining_k,
                    tensor_dim1=b.sub(M, b.add(block_m, a_row0)),
                    row_stride=packed_k,
                    tile_dim0=spec.tile_k // 2,
                    tile_dim1=a_rows,
                )
            )
            b.tensor_load_to_lds(
                *b.tdm_descriptor_2d(
                    b.global_ptr_add(B, b_element),
                    b_lds,
                    elem_bytes=1,
                    tensor_dim0=remaining_k,
                    tensor_dim1=b.sub(N, b.add(block_n, b_row0)),
                    row_stride=packed_k,
                    tile_dim0=spec.tile_k // 2,
                    tile_dim1=b_rows,
                )
            )

        def _issue_scale_strip(a_row0, a_rows, b_col0, b_cols) -> None:
            assert a_scale_lds is not None and b_scale_lds is not None
            a_scale_dst = b.smem_ptr_add(
                shared_addr,
                b.zext(
                    b.add(
                        a_scale_lds,
                        b.mul(a_row0, b.const_i32(scale_k_groups)),
                    ),
                    shared_addr.type,
                ),
            )
            b_scale_dst = b.smem_ptr_add(
                shared_addr,
                b.zext(
                    b.add(b_scale_lds, b_col0),
                    shared_addr.type,
                ),
            )
            a_scale_elem = b.add(
                b.mul(b.add(block_m, a_row0), scale_groups),
                group0,
            )
            b_scale_elem = b.add(
                b.mul(group0, N),
                b.add(block_n, b_col0),
            )
            # rocKE staging layout (not Tensile InMemorySwizzle bit-exact).
            b.tensor_load_to_lds(
                *b.tdm_descriptor_2d(
                    b.global_ptr_add(AS, a_scale_elem),
                    a_scale_dst,
                    elem_bytes=1,
                    tensor_dim0=remaining_groups,
                    tensor_dim1=b.sub(M, b.add(block_m, a_row0)),
                    row_stride=scale_groups,
                    tile_dim0=scale_k_groups,
                    tile_dim1=a_rows,
                )
            )
            b.tensor_load_to_lds(
                *b.tdm_descriptor_2d(
                    b.global_ptr_add(BS, b_scale_elem),
                    b_scale_dst,
                    elem_bytes=1,
                    tensor_dim0=b.sub(N, b.add(block_n, b_col0)),
                    tensor_dim1=remaining_groups,
                    row_stride=N,
                    tile_dim0=b_cols,
                    tile_dim1=scale_k_groups,
                )
            )

        if spec.wave_separated_tdm:
            num_waves = spec.warp_m * spec.warp_n * spec.warp_k
            a_rows = spec.tile_m // num_waves
            b_rows = spec.tile_n // num_waves
            a_row0 = b.mul(wave, b.const_i32(a_rows))
            b_row0 = b.mul(wave, b.const_i32(b_rows))
            _issue_ab_strip(a_row0, a_rows, b_row0, b_rows)
        else:
            with b.scf_if(b.cmp_eq(wave, c0)):
                _issue_ab_strip(c0, spec.tile_m, c0, spec.tile_n)
        # Scale tiles are small; wave 0 stages the full MacroTile into LDS
        # (row-major rocKE layout, not Tensile InMemorySwizzle).
        if spec.scale_tdm:
            with b.scf_if(b.cmp_eq(wave, c0)):
                _issue_scale_strip(c0, spec.tile_m, c0, spec.tile_n)

    def _prefetch_gl2(k0) -> None:
        """PrefetchGL2=2: hint A/B rows two DepthU tiles ahead into GL2."""
        if not spec.prefetch_gl2:
            return
        ahead = b.add(k0, b.const_i32(2 * spec.tile_k))
        with b.scf_if(b.cmp_lt(ahead, K)):
            k_byte = b.div(ahead, b.const_i32(2))
            # One lane of each wave prefetches its local A/B row head.
            a_row = b.add(block_m, b.add(atom_rows[0], lane))
            b_row = b.add(block_n, b.add(atom_cols[0], lane_col))
            a_elem = b.add(b.mul(a_row, packed_k), k_byte)
            b_elem = b.add(b.mul(b_row, packed_k), k_byte)
            b.global_prefetch(b.global_ptr_add(A, a_elem))
            b.global_prefetch(b.global_ptr_add(B, b_elem))

    def _load_i32x4(byte_offset):
        return b.smem_load_vN(shared, byte_offset, dtype=I32, n=4)

    def _load_a_fragment(a_buffer, atom_row, inner_k):
        row0 = b.add(atom_row, lane_col)
        row1 = b.add(row0, c16)
        lane_k = b.mul(lane_half, c16)

        def row_fragment(row):
            row_base = b.add(
                a_buffer,
                b.mul(row, b.const_i32(spec.tile_k // 2)),
            )
            first = b.add(row_base, b.add(b.const_i32(inner_k // 2), lane_k))
            second = b.add(first, c32)
            return b.vec_concat(_load_i32x4(first), _load_i32x4(second))

        return b.vec_concat(row_fragment(row0), row_fragment(row1))

    def _load_b_fragment(b_buffer, atom_col, inner_k):
        row = b.add(atom_col, lane_col)
        row_base = b.add(
            b_buffer,
            b.mul(row, b.const_i32(spec.tile_k // 2)),
        )
        lane_k = b.mul(lane_half, c16)
        first = b.add(row_base, b.add(b.const_i32(inner_k // 2), lane_k))
        return b.vec_concat(_load_i32x4(first), _load_i32x4(b.add(first, c32)))

    def _pack_a_scale_global(global_row, group0):
        index = b.add(b.mul(global_row, scale_groups), group0)
        raw = b.global_load_vN(AS, index, I8, 4, align=1)
        return b.bitcast(raw, I32)

    def _pack_b_scale_global(global_col, group0):
        values = []
        for group in range(4):
            index = b.add(
                b.mul(b.add(group0, b.const_i32(group)), N),
                global_col,
            )
            values.append(b.global_load(BS, index, I8, align=1))
        return b.bitcast(b.vec_pack(values, I8), I32)

    def _pack_a_scale_lds(a_scale_buf, local_row, inner_k):
        byte_off = b.add(
            a_scale_buf,
            b.add(
                b.mul(local_row, b.const_i32(spec.tile_k // 32)),
                b.const_i32(inner_k // 32),
            ),
        )
        raw = b.smem_load_vN(shared, byte_off, dtype=I8, n=4)
        return b.bitcast(raw, I32)

    def _pack_b_scale_lds(b_scale_buf, local_col, inner_k):
        values = []
        group_base = inner_k // 32
        for group in range(4):
            byte_off = b.add(
                b_scale_buf,
                b.add(
                    b.mul(
                        b.const_i32(group_base + group),
                        b.const_i32(spec.tile_n),
                    ),
                    local_col,
                ),
            )
            loaded = b.smem_load_vN(shared, byte_off, dtype=I8, n=1)
            values.append(b.vec_extract(loaded, 0))
        return b.bitcast(b.vec_pack(values, I8), I32)

    atom_rows = [
        b.add(
            b.mul(wave_m, b.const_i32(spec.wave_repeat_m * _ATOM_M)),
            b.const_i32(repeat * _ATOM_M),
        )
        for repeat in range(spec.wave_repeat_m)
    ]
    atom_cols = [
        b.add(
            b.mul(wave_n, b.const_i32(spec.wave_repeat_n * _ATOM_N)),
            b.const_i32(repeat * _ATOM_N),
        )
        for repeat in range(spec.wave_repeat_n)
    ]

    def _a_scales_for_step(a_scale_buf, k0, inner_k):
        if spec.scale_tdm:
            assert a_scale_buf is not None
            return [
                _pack_a_scale_lds(a_scale_buf, b.add(atom_row, lane), inner_k)
                for atom_row in atom_rows
            ]
        group0 = b.div(b.add(k0, b.const_i32(inner_k)), c32)
        return [
            _pack_a_scale_global(b.add(block_m, b.add(atom_row, lane)), group0)
            for atom_row in atom_rows
        ]

    def _b_scales_for_step(b_scale_buf, k0, inner_k):
        if spec.scale_tdm:
            assert b_scale_buf is not None
            return [
                _pack_b_scale_lds(b_scale_buf, b.add(atom_col, lane_col), inner_k)
                for atom_col in atom_cols
            ]
        group0 = b.div(b.add(k0, b.const_i32(inner_k)), c32)
        return [
            _pack_b_scale_global(b.add(block_n, b.add(atom_col, lane_col)), group0)
            for atom_col in atom_cols
        ]

    def _compute_tile(k0, parity, accumulators):
        a_buffer, b_buffer, a_scale_buf, b_scale_buf = _buffer_offsets(parity)
        out = list(accumulators)
        inner_ks = list(range(0, spec.tile_k, _ATOM_K))
        # HalfPLR-style A local-read prefetch: load the next K-step A
        # fragments before the current WMMA cluster consumes them.
        a_prefetch = None
        if spec.local_read_prefetch and inner_ks:
            a_prefetch = [
                _load_a_fragment(a_buffer, atom_row, inner_ks[0])
                for atom_row in atom_rows
            ]
        # Scale double-buffer: prefetch next E8M0 packs into VGPR while the
        # current WMMA cluster runs (global or LDS depending on scale_tdm).
        a_scale_prefetch = None
        b_scale_prefetch = None
        if spec.scale_double_buffer and inner_ks:
            a_scale_prefetch = _a_scales_for_step(a_scale_buf, k0, inner_ks[0])
            b_scale_prefetch = _b_scales_for_step(b_scale_buf, k0, inner_ks[0])
        for step_i, inner_k in enumerate(inner_ks):
            group0 = b.div(b.add(k0, b.const_i32(inner_k)), c32)
            next_inner = inner_ks[step_i + 1] if step_i + 1 < len(inner_ks) else None
            if spec.local_read_prefetch:
                a_frags = a_prefetch
                if next_inner is not None:
                    a_prefetch = [
                        _load_a_fragment(a_buffer, atom_row, next_inner)
                        for atom_row in atom_rows
                    ]
            else:
                a_frags = None
            if spec.scale_double_buffer:
                a_scales = a_scale_prefetch
                b_scales = b_scale_prefetch
                if next_inner is not None:
                    a_scale_prefetch = _a_scales_for_step(a_scale_buf, k0, next_inner)
                    b_scale_prefetch = _b_scales_for_step(b_scale_buf, k0, next_inner)
            else:
                a_scales = None
                b_scales = None
            for rm, atom_row in enumerate(atom_rows):
                a_frag = (
                    a_frags[rm]
                    if a_frags is not None
                    else _load_a_fragment(a_buffer, atom_row, inner_k)
                )
                if a_scales is not None:
                    a_scale = a_scales[rm]
                elif spec.scale_tdm:
                    a_scale = _pack_a_scale_lds(
                        a_scale_buf, b.add(atom_row, lane), inner_k
                    )
                else:
                    a_scale = _pack_a_scale_global(
                        b.add(block_m, b.add(atom_row, lane)), group0
                    )
                for rn, atom_col in enumerate(atom_cols):
                    index = rm * spec.wave_repeat_n + rn
                    b_frag = _load_b_fragment(b_buffer, atom_col, inner_k)
                    if b_scales is not None:
                        b_scale = b_scales[rn]
                    elif spec.scale_tdm:
                        b_scale = _pack_b_scale_lds(
                            b_scale_buf,
                            b.add(atom_col, lane_col),
                            inner_k,
                        )
                    else:
                        b_scale = _pack_b_scale_global(
                            b.add(block_n, b.add(atom_col, lane_col)), group0
                        )
                    out[index] = b.mma(
                        _WmmaScaleF4Atom(),
                        a_frag,
                        b_frag,
                        out[index],
                        a_scale,
                        b_scale,
                    )
        return out

    accumulators = [
        b.zero_vec_f32(_ACCUMULATORS_PER_ATOM)
        for _ in range(spec.wave_repeat_m * spec.wave_repeat_n)
    ]

    _issue_tdm(c0, c0)
    _prefetch_gl2(c0)
    upper = b.sub(K, b.const_i32(spec.tile_k))
    loop_args = [("parity", c0)] + [
        (f"acc{i}", value) for i, value in enumerate(accumulators)
    ]
    loop = b.scf_for_iter(
        c0,
        upper,
        b.const_i32(spec.tile_k),
        loop_args,
        iv_name="k0",
        elide_trailing_barrier=False,
    )
    with loop as (k0, carried):
        parity = carried[0]
        next_parity = b.sub(c1, parity)
        b.s_wait_tensorcnt(0)
        b.s_waitcnt(lgkmcnt=0)
        b.s_barrier_bare()
        _issue_tdm(b.add(k0, b.const_i32(spec.tile_k)), next_parity)
        _prefetch_gl2(b.add(k0, b.const_i32(spec.tile_k)))
        updated = _compute_tile(k0, parity, carried[1:])
        b.scf_yield(next_parity, *updated)

    final_parity = loop.results[0]
    b.s_wait_tensorcnt(0)
    b.s_waitcnt(lgkmcnt=0)
    b.s_barrier_bare()
    final_accumulators = _compute_tile(upper, final_parity, loop.results[1:])

    for rm, atom_row in enumerate(atom_rows):
        for rn, atom_col in enumerate(atom_cols):
            acc = final_accumulators[rm * spec.wave_repeat_n + rn]
            for slot in range(_ACCUMULATORS_PER_ATOM):
                row = b.add(
                    block_m,
                    b.add(
                        atom_row,
                        b.add(
                            b.const_i32((slot // 8) * 16 + slot % 8),
                            b.mul(lane_half, b.const_i32(8)),
                        ),
                    ),
                )
                col = b.add(block_n, b.add(atom_col, lane_col))
                index = b.add(b.mul(row, N), col)
                value = b.fmul(alpha, b.vec_extract(acc, slot))
                c_value = b.cvt_fp8_to_f32(b.global_load(C, index, FP8E4M3, align=1))
                value = b.fadd(value, b.fmul(beta, c_value))
                b.global_store(D, index, b.cvt_f32_to_fp8(value), align=1)

    b.ret()
    return b.kernel

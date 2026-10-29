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
    Value,
)
from ...helpers.spec import SignatureBuilder, ceil_div_grid, kernel_name_join

_ATOM_M = 32
_ATOM_N = 16
_ATOM_K = 128
_WAVE = 32
_ACCUMULATORS_PER_ATOM = 16
# Bytes one global_prefetch covers in TensileLite's GL2Prefetch.
_GL2_PREFETCH_BYTES = 256
# A-scale LDS pad: 32 lanes read rows 16 B apart, so rows r and r+16 share
# a bank; 8 B after every 256 B moves the second half onto other banks.
_A_SCALE_PAD_BLOCK_BYTES = 256
_A_SCALE_PAD_BYTES = 8
_OP_ID = "wmma_scale_f32_32x16x128_f4"


class _WmmaScaleF4Atom:
    op_id = _OP_ID
    c_frag_len = _ACCUMULATORS_PER_ATOM
    c_dtype = "fp32"


@dataclass(frozen=True)
class _SegmentInterleaveLayout:
    """bcontig LDS layout for the supported gfx1250 four-wave configuration.

    A is split into two M components. B stays contiguous because the rocKE
    WMMA B-fragment mapping spans both N components. The A component stride
    crosses a 64 KiB segment boundary; a buffer may use more than one segment.
    """

    component_rows: int
    a_component_bytes: int
    a_component_stride_bytes: int
    b_base_bytes: int
    a_scale_base_bytes: int | None
    b_scale_base_bytes: int | None
    buffer_bytes: int


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
    # B_scale is [N][K/32] in global memory (same as A_scale) instead of
    # [K/32][N], so each lane's four E8M0 groups are one 4-byte load.
    b_scale_n_major: bool = False
    # Cluster TDM multicast: workgroups that share an A (or B) tile issue the
    # same load with a workgroup_mask so it can be fetched once per cluster.
    tdm_multicast: bool = False
    # Barrier, TDM refill and the next tile's first-step operand reads run
    # before the last K-step's WMMAs instead of at the top of the next tile.
    cross_tile_prefetch: bool = False
    # The workgroups of a cluster row (A) or column (B) split the GL2
    # prefetch of the shared tile, one cache line per thread, instead of
    # each hinting the same row heads.
    cluster_gl2_prefetch: bool = False
    # Pad the LDS A-scale tile by _A_SCALE_PAD_BYTES per
    # _A_SCALE_PAD_BLOCK_BYTES (TDM pad fields plus matching reads).
    a_scale_lds_pad: bool = False

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
        """Row-major rocKE staging: tile_m × (tile_k/32) E8M0 bytes, plus pad."""
        data = self.tile_m * (self.tile_k // 32)
        if not self.a_scale_lds_pad:
            return data
        return data + data // _A_SCALE_PAD_BLOCK_BYTES * _A_SCALE_PAD_BYTES

    @property
    def b_scale_tile_bytes(self) -> int:
        """(tile_k/32) × tile_n E8M0 bytes; tile_n × (tile_k/32) when N-major."""
        return (self.tile_k // 32) * self.tile_n

    @property
    def segment_interleave_layout(self) -> _SegmentInterleaveLayout:
        """Return rocKE's bcontig layout for one LDS prefetch buffer.

        The existing TDM descriptor padding is row-based (+16 B per row), so
        these sizes intentionally describe rocKE's layout rather than copying
        TensileLite's block-padding byte counts. Both layouts place A0 and A1
        in separate 64 KiB segments while B remains contiguous.
        """
        component_rows = self.tile_m // 2
        a_component_bytes = component_rows * self.a_lds_row_stride_bytes
        b_base_bytes = a_component_bytes
        b_bytes = self.b_tile_bytes
        packed_stride = a_component_bytes + b_bytes
        component_stride = max(packed_stride, _LDS_SEGMENT_BYTES)
        ab_end = max(
            b_base_bytes + b_bytes,
            component_stride + a_component_bytes,
        )
        a_scale_base = ab_end if self.scale_tdm else None
        b_scale_base = (
            a_scale_base + self.a_scale_tile_bytes
            if a_scale_base is not None
            else None
        )
        buffer_bytes = (
            b_scale_base + self.b_scale_tile_bytes
            if b_scale_base is not None
            else ab_end
        )
        return _SegmentInterleaveLayout(
            component_rows=component_rows,
            a_component_bytes=a_component_bytes,
            a_component_stride_bytes=component_stride,
            b_base_bytes=b_base_bytes,
            a_scale_base_bytes=a_scale_base,
            b_scale_base_bytes=b_scale_base,
            buffer_bytes=buffer_bytes,
        )

    @property
    def _bytes_per_prefetch_slot(self) -> int:
        if self.lds_segment_interleave:
            return self.segment_interleave_layout.buffer_bytes
        total = self.a_tile_bytes + self.b_tile_bytes
        if self.scale_tdm:
            total += self.a_scale_tile_bytes + self.b_scale_tile_bytes
        return total

    @property
    def effective_prefetch_depth(self) -> int:
        """Depth 2 by default; ``tdm_plus_lds_buf`` escalates to 3 when it fits.

        Escalation requires ``3 * slot <= LDS cap``. Segment-interleaved
        buffers can span multiple 64 KiB segments, so the capacity gate is
        their complete bcontig buffer size.
        """
        if self.tdm_prefetch_depth == 3:
            return 3
        if not self.tdm_plus_lds_buf:
            return self.tdm_prefetch_depth
        slot = self._bytes_per_prefetch_slot
        lds_cap = ArchTarget.from_gfx("gfx1250").lds_capacity_bytes
        if 3 * slot <= lds_cap:
            return 3
        return 2

    @property
    def lds_bytes(self) -> int:
        depth = self.effective_prefetch_depth
        if self.lds_segment_interleave:
            return depth * self.segment_interleave_layout.buffer_bytes
        return depth * self._bytes_per_prefetch_slot

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
                "bsn": self.b_scale_n_major,
                "mcast": self.tdm_multicast,
                "xtile": self.cross_tile_prefetch,
                "cgl2": self.cluster_gl2_prefetch,
                "aspad": self.a_scale_lds_pad,
            },
        )


# Step 6 flags that are declared but not yet wired into the builder.
# Each landed substep removes its flag from this frozenset.
_LDS_SEGMENT_BYTES = 65536

_UNIMPLEMENTED_STEP6_FLAGS: frozenset[str] = frozenset()


def _tdm_lds_pad_fields(row_bytes: int, pad_bytes: int = 16) -> dict[str, int]:
    """Encode ``pad_bytes`` of LDS pad after every ``row_bytes`` of contiguous data.

    Matches CK Tile ``EncodedTdmPadding``: ``pad_interval = log2(row/4)-1``,
    ``pad_amount = pad/4-1``.
    """
    if row_bytes < 4 or row_bytes & (row_bytes - 1):
        raise ValueError(f"LDS pad row_bytes must be a power of two >= 4, got {row_bytes}")
    if pad_bytes < 4 or pad_bytes % 4:
        raise ValueError(f"LDS pad_bytes must be a positive multiple of 4, got {pad_bytes}")
    pad_interval = (row_bytes // 4).bit_length() - 2
    return {"pad_interval": pad_interval, "pad_amount": pad_bytes // 4 - 1}


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
    if spec.lds_segment_interleave:
        if not spec.wave_separated_tdm:
            return False, (
                "lds_segment_interleave requires wave_separated_tdm"
            )
        if (spec.warp_m, spec.warp_n, spec.warp_k) != (2, 2, 1):
            return False, (
                "lds_segment_interleave currently supports a 2x2x1 wave group"
            )
        if spec.lds_segment_b_rotate:
            return False, (
                "lds_segment_b_rotate is unsupported by the bcontig "
                "segment-interleave layout"
            )
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
        cx, cy, cz = spec.cluster_dim
        gx, gy, gz = mxfp4_tdm_gemm_grid(spec)
        if gx % cx or gy % cy or gz % cz:
            return False, (
                f"grid {gx}x{gy}x{gz} must be divisible by cluster_dim "
                f"{cx}x{cy}x{cz}"
            )
    if spec.tdm_multicast:
        if spec.cluster_dim is None:
            return False, "tdm_multicast requires cluster_dim"
        cx, cy, cz = spec.cluster_dim
        if cz != 1 or cx * cy > 16:
            return False, "tdm_multicast needs a 2D cluster of at most 16 workgroups"
        if not ArchTarget.from_gfx("gfx1250").memory.has_multicast_load:
            return False, "gfx1250 reports no multicast load support"
    if spec.cross_tile_prefetch and not (
        spec.init_c_wmma and spec.local_read_prefetch and spec.scale_double_buffer
    ):
        return False, (
            "cross_tile_prefetch requires init_c_wmma, local_read_prefetch "
            "and scale_double_buffer"
        )
    if spec.cluster_gl2_prefetch:
        if not spec.prefetch_gl2 or spec.cluster_dim is None:
            return False, "cluster_gl2_prefetch requires prefetch_gl2 and cluster_dim"
        if spec.cluster_dim[2] != 1:
            return False, "cluster_gl2_prefetch needs a 2D cluster"
        if (spec.tile_k // 2) % _GL2_PREFETCH_BYTES:
            return False, (
                f"cluster_gl2_prefetch needs tile_k/2 to be a multiple of "
                f"{_GL2_PREFETCH_BYTES} B"
            )
    if spec.a_scale_lds_pad:
        if not spec.scale_tdm:
            return False, "a_scale_lds_pad requires scale_tdm"
        if _A_SCALE_PAD_BLOCK_BYTES % (spec.tile_k // 32):
            return False, (
                f"a_scale_lds_pad needs tile_k/32 to divide {_A_SCALE_PAD_BLOCK_BYTES}"
            )
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
    if spec.cluster_dim is not None:
        b.set_cluster_dims(*spec.cluster_dim)

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

    # Grid x walks N tiles and y walks M tiles, so the workgroups of one
    # cluster row share A and those of one cluster column share B.
    a_mcast_mask: int | Value = 0
    b_mcast_mask: int | Value = 0
    if spec.tdm_multicast:
        assert spec.cluster_dim is not None
        cluster_x, cluster_y, _ = spec.cluster_dim
        row_bits = (1 << cluster_x) - 1
        col_bits = sum(1 << (row * cluster_x) for row in range(cluster_y))
        a_mcast_mask = b.shl(
            b.const_i32(row_bits),
            b.mul(b.cluster_workgroup_id("y"), b.const_i32(cluster_x)),
        )
        b_mcast_mask = b.shl(b.const_i32(col_bits), b.cluster_workgroup_id("x"))

    shared = b.smem_alloc(I8, [spec.lds_bytes], name_hint="ab_tdm")
    shared_addr = b.smem_addr_of(shared)
    depth = spec.effective_prefetch_depth
    b_region = depth * spec.a_tile_bytes
    scale_region = depth * (spec.a_tile_bytes + spec.b_tile_bytes)
    b_scale_region = scale_region + depth * spec.a_scale_tile_bytes

    def _buffer_offsets(parity):
        if spec.lds_segment_interleave:
            # bcontig layout: [A0][B-full][A1][A-scales][B-scales].
            # A's second M component is component_stride bytes after A0;
            # B stays contiguous at b_base. Each parity selects a full buffer.
            layout = spec.segment_interleave_layout
            buffer = b.mul(parity, b.const_i32(layout.buffer_bytes))
            a_offset = buffer
            b_offset = b.add(buffer, b.const_i32(layout.b_base_bytes))
            if not spec.scale_tdm:
                return a_offset, b_offset, None, None
            a_scale_offset = b.add(
                buffer, b.const_i32(layout.a_scale_base_bytes)
            )
            b_scale_offset = b.add(
                buffer, b.const_i32(layout.b_scale_base_bytes)
            )
            return a_offset, b_offset, a_scale_offset, b_scale_offset
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
        b_row_stride = b.const_i32(spec.b_lds_row_stride_bytes)
        scale_k_groups = spec.tile_k // 32
        group0 = b.div(k0, c32)
        remaining_groups = b.sub(scale_groups, group0)
        ab_pad = (
            _tdm_lds_pad_fields(spec.tile_k // 2) if spec.lds_pad else {}
        )
        a_scale_pad = (
            _tdm_lds_pad_fields(_A_SCALE_PAD_BLOCK_BYTES, _A_SCALE_PAD_BYTES)
            if spec.a_scale_lds_pad
            else {}
        )

        def _issue_ab_strip(a_row0, a_rows, b_row0, b_rows) -> None:
            if spec.lds_segment_interleave:
                layout = spec.segment_interleave_layout
                a_component = b.div(
                    a_row0, b.const_i32(layout.component_rows)
                )
                a_component_row = b.mod(
                    a_row0, b.const_i32(layout.component_rows)
                )
                a_lds_row = b.add(
                    a_lds_offset,
                    b.add(
                        b.mul(
                            a_component,
                            b.const_i32(layout.a_component_stride_bytes),
                        ),
                        b.mul(a_component_row, a_row_stride),
                    ),
                )
            else:
                a_lds_row = b.add(a_lds_offset, b.mul(a_row0, a_row_stride))
            a_lds = b.smem_ptr_add(
                shared_addr,
                b.zext(
                    a_lds_row,
                    shared_addr.type,
                ),
            )
            b_lds = b.smem_ptr_add(
                shared_addr,
                b.zext(
                    b.add(b_lds_offset, b.mul(b_row0, b_row_stride)),
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
                    workgroup_mask=a_mcast_mask,
                    **ab_pad,
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
                    workgroup_mask=b_mcast_mask,
                    **ab_pad,
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
            if spec.b_scale_n_major:
                b_scale_dst_off = b.add(
                    b_scale_lds, b.mul(b_col0, b.const_i32(scale_k_groups))
                )
                b_scale_elem = b.add(
                    b.mul(b.add(block_n, b_col0), scale_groups),
                    group0,
                )
            else:
                b_scale_dst_off = b.add(b_scale_lds, b_col0)
                b_scale_elem = b.add(
                    b.mul(group0, N),
                    b.add(block_n, b_col0),
                )
            b_scale_dst = b.smem_ptr_add(
                shared_addr,
                b.zext(b_scale_dst_off, shared_addr.type),
            )
            a_scale_elem = b.add(
                b.mul(b.add(block_m, a_row0), scale_groups),
                group0,
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
                    workgroup_mask=a_mcast_mask,
                    **a_scale_pad,
                )
            )
            if spec.b_scale_n_major:
                b_scale_desc = b.tdm_descriptor_2d(
                    b.global_ptr_add(BS, b_scale_elem),
                    b_scale_dst,
                    elem_bytes=1,
                    tensor_dim0=remaining_groups,
                    tensor_dim1=b.sub(N, b.add(block_n, b_col0)),
                    row_stride=scale_groups,
                    tile_dim0=scale_k_groups,
                    tile_dim1=b_cols,
                    workgroup_mask=b_mcast_mask,
                )
            else:
                b_scale_desc = b.tdm_descriptor_2d(
                    b.global_ptr_add(BS, b_scale_elem),
                    b_scale_dst,
                    elem_bytes=1,
                    tensor_dim0=b.sub(N, b.add(block_n, b_col0)),
                    tensor_dim1=remaining_groups,
                    row_stride=N,
                    tile_dim0=b_cols,
                    tile_dim1=scale_k_groups,
                    workgroup_mask=b_mcast_mask,
                )
            b.tensor_load_to_lds(*b_scale_desc)

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

    def _prefetch_gl2_cluster(k_byte) -> None:
        """Split the cluster's A and B blocks at ``k_byte`` over all its threads.

        The cluster consumes ``cy`` A tiles and ``cx`` B tiles. Their rows are
        numbered A first, then B, in ``_GL2_PREFETCH_BYTES`` chunks; cluster
        thread ``flat * block_size + tid`` prefetches every ``pool``-th chunk.
        """
        assert spec.cluster_dim is not None
        cx, cy, _ = spec.cluster_dim
        chunks = spec.tile_k // 2 // _GL2_PREFETCH_BYTES
        a_chunks = cy * spec.tile_m * chunks
        total = a_chunks + cx * spec.tile_n * chunks
        pool = cx * cy * spec.block_size
        cwx = b.cluster_workgroup_id("x")
        cwy = b.cluster_workgroup_id("y")
        cluster_m0 = b.sub(block_m, b.mul(cwy, b.const_i32(spec.tile_m)))
        cluster_n0 = b.sub(block_n, b.mul(cwx, b.const_i32(spec.tile_n)))
        flat = b.add(cwx, b.mul(cwy, b.const_i32(cx)))
        thread = b.add(b.mul(flat, b.const_i32(spec.block_size)), tid)
        chunk_bytes = b.const_i32(_GL2_PREFETCH_BYTES)
        chunks_v = b.const_i32(chunks)

        def _row_chunk_ptr(base, row0, index):
            row = b.add(row0, b.div(index, chunks_v))
            offset = b.add(k_byte, b.mul(b.mod(index, chunks_v), chunk_bytes))
            return b.global_ptr_add(base, b.add(b.mul(row, packed_k), offset))

        def _prefetch_chunk(index):
            # Branch-free: in the YAML cluster the A/B choice is uniform per
            # workgroup, and a divergent branch here costs exec-mask code.
            is_a = b.cmp_lt(index, b.const_i32(a_chunks))
            b_index = b.sub(index, b.const_i32(a_chunks))
            a_index = b.select(is_a, index, c0)
            b_index = b.select(is_a, c0, b_index)
            b.global_prefetch(
                b.select(
                    is_a,
                    _row_chunk_ptr(A, cluster_m0, a_index),
                    _row_chunk_ptr(B, cluster_n0, b_index),
                )
            )

        for start in range(0, total, pool):
            index = b.add(thread, b.const_i32(start))
            if start + pool <= total:
                _prefetch_chunk(index)
            else:
                with b.scf_if(b.cmp_lt(index, b.const_i32(total))):
                    _prefetch_chunk(index)

    def _prefetch_gl2(k0) -> None:
        """PrefetchGL2=2: hint A/B rows two DepthU tiles ahead into GL2."""
        if not spec.prefetch_gl2:
            return
        ahead = b.add(k0, b.const_i32(2 * spec.tile_k))
        if spec.cluster_gl2_prefetch:
            with b.scf_if(b.cmp_lt(ahead, K)):
                _prefetch_gl2_cluster(b.div(ahead, b.const_i32(2)))
            return
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
        a_stride = b.const_i32(spec.a_lds_row_stride_bytes)

        def row_fragment(row):
            if spec.lds_segment_interleave:
                layout = spec.segment_interleave_layout
                component = b.div(row, b.const_i32(layout.component_rows))
                component_row = b.mod(row, b.const_i32(layout.component_rows))
                row_base = b.add(
                    a_buffer,
                    b.add(
                        b.mul(
                            component,
                            b.const_i32(layout.a_component_stride_bytes),
                        ),
                        b.mul(component_row, a_stride),
                    ),
                )
            else:
                row_base = b.add(a_buffer, b.mul(row, a_stride))
            first = b.add(row_base, b.add(b.const_i32(inner_k // 2), lane_k))
            second = b.add(first, c32)
            return b.vec_concat(_load_i32x4(first), _load_i32x4(second))

        return b.vec_concat(row_fragment(row0), row_fragment(row1))

    def _b_local_col(atom_col):
        """B LDS / store column; odd waves rotate by half the N atom when enabled."""
        col = b.add(atom_col, lane_col)
        if not spec.lds_segment_b_rotate:
            return col
        rotated = b.add(atom_col, b.mod(b.add(lane_col, b.const_i32(_ATOM_N // 2)), c16))
        return b.select(b.cmp_eq(b.mod(wave, b.const_i32(2)), c1), rotated, col)

    def _load_b_fragment(b_buffer, atom_col, inner_k):
        row = _b_local_col(atom_col)
        row_base = b.add(
            b_buffer,
            b.mul(row, b.const_i32(spec.b_lds_row_stride_bytes)),
        )
        lane_k = b.mul(lane_half, c16)
        first = b.add(row_base, b.add(b.const_i32(inner_k // 2), lane_k))
        return b.vec_concat(_load_i32x4(first), _load_i32x4(b.add(first, c32)))

    def _pack_a_scale_global(global_row, group0):
        index = b.add(b.mul(global_row, scale_groups), group0)
        raw = b.global_load_vN(AS, index, I8, 4, align=1)
        return b.bitcast(raw, I32)

    def _pack_b_scale_global(global_col, group0):
        if spec.b_scale_n_major:
            index = b.add(b.mul(global_col, scale_groups), group0)
            raw = b.global_load_vN(BS, index, I8, 4, align=1)
            return b.bitcast(raw, I32)
        values = []
        for group in range(4):
            index = b.add(
                b.mul(b.add(group0, b.const_i32(group)), N),
                global_col,
            )
            values.append(b.global_load(BS, index, I8, align=1))
        return b.bitcast(b.vec_pack(values, I8), I32)

    def _pack_a_scale_lds(a_scale_buf, local_row, inner_k):
        offset = b.add(
            b.mul(local_row, b.const_i32(spec.tile_k // 32)),
            b.const_i32(inner_k // 32),
        )
        if spec.a_scale_lds_pad:
            block = b.lshr(
                offset,
                b.const_i32(_A_SCALE_PAD_BLOCK_BYTES.bit_length() - 1),
            )
            offset = b.add(offset, b.mul(block, b.const_i32(_A_SCALE_PAD_BYTES)))
        byte_off = b.add(a_scale_buf, offset)
        raw = b.smem_load_vN(shared, byte_off, dtype=I8, n=4)
        return b.bitcast(raw, I32)

    def _pack_b_scale_lds(b_scale_buf, local_col, inner_k):
        if spec.b_scale_n_major:
            byte_off = b.add(
                b_scale_buf,
                b.add(
                    b.mul(local_col, b.const_i32(spec.tile_k // 32)),
                    b.const_i32(inner_k // 32),
                ),
            )
            raw = b.smem_load_vN(shared, byte_off, dtype=I8, n=4)
            return b.bitcast(raw, I32)
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

    def _load_first_step(k0, parity):
        """Operands of a tile's first K-step: A/B fragments, A/B scales."""
        a_buffer, b_buffer, a_scale_buf, b_scale_buf = _buffer_offsets(parity)
        return (
            [_load_a_fragment(a_buffer, atom_row, 0) for atom_row in atom_rows],
            _a_scales_for_step(a_scale_buf, k0, 0),
            _b_scales_for_step(b_scale_buf, k0, 0),
            [_load_b_fragment(b_buffer, atom_col, 0) for atom_col in atom_cols],
        )

    def _compute_tile(
        k0, parity, accumulators, *, seed_zero_c=False, first=None, boundary=None
    ):
        """Compute one DepthU tile.

        With ``seed_zero_c`` (InitC WMMA), the first K-step's WMMAs use
        ``zero_vec_f32`` as C instead of ``accumulators``. ``first`` supplies
        this tile's first-step operands (from :func:`_load_first_step`).
        ``boundary`` runs after every LDS read of this tile has been issued
        and before the last K-step's WMMAs; its return value is returned
        alongside the accumulators.
        """
        a_buffer, b_buffer, a_scale_buf, b_scale_buf = _buffer_offsets(parity)
        out = list(accumulators)
        inner_ks = list(range(0, spec.tile_k, _ATOM_K))
        use_seed = seed_zero_c
        boundary_result = None
        # HalfPLR-style A local-read prefetch: load the next K-step A
        # fragments before the current WMMA cluster consumes them.
        a_prefetch = None
        b_frags_first = None
        if first is not None:
            a_prefetch, a_scale_first, b_scale_first, b_frags_first = first
        elif spec.local_read_prefetch and inner_ks:
            a_prefetch = [
                _load_a_fragment(a_buffer, atom_row, inner_ks[0])
                for atom_row in atom_rows
            ]
        # Scale double-buffer: prefetch next E8M0 packs into VGPR while the
        # current WMMA cluster runs (global or LDS depending on scale_tdm).
        a_scale_prefetch = None
        b_scale_prefetch = None
        if first is not None:
            a_scale_prefetch, b_scale_prefetch = a_scale_first, b_scale_first
        elif spec.scale_double_buffer and inner_ks:
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
            b_frags = b_frags_first if step_i == 0 else None
            if boundary is not None and next_inner is None:
                b_frags = [
                    _load_b_fragment(b_buffer, atom_col, inner_k)
                    for atom_col in atom_cols
                ]
                boundary_result = boundary()
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
                    b_frag = (
                        b_frags[rn]
                        if b_frags is not None
                        else _load_b_fragment(b_buffer, atom_col, inner_k)
                    )
                    b_col = _b_local_col(atom_col)
                    if b_scales is not None and not spec.lds_segment_b_rotate:
                        b_scale = b_scales[rn]
                    elif spec.scale_tdm:
                        b_scale = _pack_b_scale_lds(b_scale_buf, b_col, inner_k)
                    else:
                        b_scale = _pack_b_scale_global(b.add(block_n, b_col), group0)
                    if use_seed:
                        c_in = b.zero_vec_f32(_ACCUMULATORS_PER_ATOM)
                    else:
                        c_in = out[index]
                    out[index] = b.mma(
                        _WmmaScaleF4Atom(),
                        a_frag,
                        b_frag,
                        c_in,
                        a_scale,
                        b_scale,
                    )
            # Only the first K-step seeds C.
            use_seed = False
        if boundary is not None:
            return out, boundary_result
        return out

    n_acc = spec.wave_repeat_m * spec.wave_repeat_n
    tile_k_v = b.const_i32(spec.tile_k)

    def _next_parity(parity):
        if depth == 3:
            return b.mod(b.add(parity, c1), b.const_i32(3))
        return b.sub(c1, parity)

    def _wait_tile_ready() -> None:
        b.s_wait_tensorcnt(0)
        b.s_waitcnt(lgkmcnt=0)
        if spec.tdm_multicast:
            # Peers write this workgroup's LDS: no one may refill a buffer
            # until every workgroup of the cluster has finished reading it.
            b.cluster_barrier(fences=False)
        else:
            b.s_barrier_bare()

    if spec.tdm_multicast:
        b.cluster_barrier(fences=False)
    _issue_tdm(c0, c0)
    _prefetch_gl2(c0)
    if spec.cross_tile_prefetch:
        # The barrier and the TDM refill of the buffer being read move ahead
        # of the last K-step's WMMAs, and the next tile's first-step
        # operands are read there, so the WMMA stream crosses the tile
        # boundary without an LDS-read bubble.
        two_tiles = b.const_i32(2 * spec.tile_k)

        def _boundary(k0, parity):
            def run():
                next_parity = _next_parity(parity)
                # The barrier and wait intrinsics do not access memory, so
                # LLVM would hoist the next tile's LDS reads above them.
                b.inline_asm("", "~{memory}")
                _wait_tile_ready()
                b.inline_asm("", "~{memory}")
                ahead = b.add(k0, two_tiles)
                with b.scf_if(b.cmp_lt(ahead, K)):
                    _issue_tdm(ahead, _next_parity(next_parity))
                    _prefetch_gl2(ahead)
                return _load_first_step(b.add(k0, tile_k_v), next_parity)

            return run

        def _flatten(first):
            return [v for group in first for v in group]

        n_a, n_b = spec.wave_repeat_m, spec.wave_repeat_n

        def _unflatten(values):
            values = list(values)
            return (
                values[:n_a],
                values[n_a : 2 * n_a],
                values[2 * n_a : 2 * n_a + n_b],
                values[2 * n_a + n_b :],
            )

        _wait_tile_ready()
        with b.scf_if(b.cmp_lt(tile_k_v, K)):
            _issue_tdm(tile_k_v, c1)
            _prefetch_gl2(tile_k_v)
        undef = [b.undef_vec(F32, _ACCUMULATORS_PER_ATOM) for _ in range(n_acc)]
        accumulators, first = _compute_tile(
            c0,
            c0,
            undef,
            seed_zero_c=True,
            first=_load_first_step(c0, c0),
            boundary=_boundary(c0, c0),
        )
        loop_args = (
            [("parity", c1)]
            + [(f"acc{i}", value) for i, value in enumerate(accumulators)]
            + [(f"first{i}", value) for i, value in enumerate(_flatten(first))]
        )
        loop = b.scf_for_iter(
            tile_k_v,
            K,
            tile_k_v,
            loop_args,
            iv_name="k0",
            elide_trailing_barrier=False,
        )
        with loop as (k0, carried):
            parity = carried[0]
            updated, first = _compute_tile(
                k0,
                parity,
                carried[1 : 1 + n_acc],
                first=_unflatten(carried[1 + n_acc :]),
                boundary=_boundary(k0, parity),
            )
            b.scf_yield(_next_parity(parity), *updated, *_flatten(first))
        final_accumulators = loop.results[1 : 1 + n_acc]
    elif spec.init_c_wmma:
        # InitCIterWmma: peel the first K tile so its WMMAs read C = 0
        # directly; the steady-state loop carries no seed select.
        _wait_tile_ready()
        with b.scf_if(b.cmp_lt(tile_k_v, K)):
            _issue_tdm(tile_k_v, c1)
            _prefetch_gl2(tile_k_v)
        undef = [b.undef_vec(F32, _ACCUMULATORS_PER_ATOM) for _ in range(n_acc)]
        accumulators = _compute_tile(c0, c0, undef, seed_zero_c=True)
        loop_args = [("parity", c1)] + [
            (f"acc{i}", value) for i, value in enumerate(accumulators)
        ]
        loop = b.scf_for_iter(
            tile_k_v,
            K,
            tile_k_v,
            loop_args,
            iv_name="k0",
            elide_trailing_barrier=False,
        )
        with loop as (k0, carried):
            parity = carried[0]
            next_parity = _next_parity(parity)
            _wait_tile_ready()
            next_k = b.add(k0, tile_k_v)
            with b.scf_if(b.cmp_lt(next_k, K)):
                _issue_tdm(next_k, next_parity)
                _prefetch_gl2(next_k)
            updated = _compute_tile(k0, parity, carried[1:])
            b.scf_yield(next_parity, *updated)
        final_accumulators = loop.results[1:]
    else:
        accumulators = [b.zero_vec_f32(_ACCUMULATORS_PER_ATOM) for _ in range(n_acc)]
        upper = b.sub(K, tile_k_v)
        loop_args = [("parity", c0)] + [
            (f"acc{i}", value) for i, value in enumerate(accumulators)
        ]
        loop = b.scf_for_iter(
            c0,
            upper,
            tile_k_v,
            loop_args,
            iv_name="k0",
            elide_trailing_barrier=False,
        )
        with loop as (k0, carried):
            parity = carried[0]
            next_parity = _next_parity(parity)
            _wait_tile_ready()
            _issue_tdm(b.add(k0, tile_k_v), next_parity)
            _prefetch_gl2(b.add(k0, tile_k_v))
            updated = _compute_tile(k0, parity, carried[1:])
            b.scf_yield(next_parity, *updated)

        final_parity = loop.results[0]
        _wait_tile_ready()
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
                # Un-rotate B's odd-wave N mapping so stores land on logical cols.
                col = b.add(block_n, _b_local_col(atom_col))
                index = b.add(b.mul(row, N), col)
                value = b.fmul(alpha, b.vec_extract(acc, slot))
                c_value = b.cvt_fp8_to_f32(b.global_load(C, index, FP8E4M3, align=1))
                value = b.fadd(value, b.fmul(beta, c_value))
                b.global_store(D, index, b.cvt_f32_to_fp8(value), align=1)

    b.ret()
    return b.kernel

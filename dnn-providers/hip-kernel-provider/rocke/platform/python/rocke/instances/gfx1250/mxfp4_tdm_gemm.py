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
_OP_ID = "wmma_scale_f32_32x16x128_f4"

# YAML winner LDS pads (Tensile names relabeled to the rocKE operand).
_PAD_BYTES = 16
_A_PAD_PERIOD = 1024  # LdsBlockSizePerPadB
_B_PAD_PERIOD = 2048  # LdsBlockSizePerPadA; needs TDM iterate mode
_B_SCALE_PAD_PERIOD = 256  # LdsPadMXSA
_B_SCALE_REGION_ALIGN = 512
# CK Tile comp_tdm rule (UniversalGemmBasePolicy::GetLdsPaddingConfig): one
# 16 B pad per sweep of the 64 LDS banks, or per row if a row is longer.
_CK_PAD_SWEEP = 64 * 4
_TDM_PAD_PERIOD_MAX = 1024
_SCALE_GROUPS_PER_STEP = _ATOM_K // 32
_GL2_LINE_BYTES = 256
# th:TH_LOAD_NT (bits 0..2) | scope:SCOPE_SE (bits 3..4), as in the winner.
_GL2_PREFETCH_CPOL = 1 | (1 << 3)
_FP8_MAX = 448.0
# sched_barrier mask: VALU (0x2), SALU (0x4) and TRANS (0x400) may cross;
# WMMAs and memory operations may not.
_SCHED_ALU_ONLY = 0x2 | 0x4 | 0x400
# The expansion of llvm.amdgcn.s.cluster.barrier (no fences): a workgroup
# barrier, then the first wave to arrive signals the cluster barrier. As
# inline asm the signal's skip branch is invisible to LLVM. LLVM expands the
# intrinsic into a real branch instead, which splits the main-loop block and
# drains the LDS-read pipeline.
_CLUSTER_BARRIER_ASM = (
    "s_barrier_signal_isfirst -1\n"
    "s_barrier_wait -1\n"
    "s_cbranch_scc0 1\n"
    "s_barrier_signal -3\n"
    "s_barrier_wait -3"
)


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
    # Step 6 flags (default off), one per YAML-winner feature; see
    # mxfp4_tdm_gemm_variants.MXFP4_GEMM_VARIANTS for the cumulative chain.
    vector_epilogue: bool = False
    local_read_prefetch: bool = False
    wave_parity_tdm: bool = False
    scale_tdm: bool = False
    lds_pad: bool = False
    lds_pad_ck: bool = False
    lds_segment_interleave: bool = False
    init_c_wmma: bool = False
    cluster_dim: tuple[int, int, int] | None = None
    tdm_multicast: bool = False
    prefetch_gl2: bool = False
    tile_refill_mid: bool = False

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
        return self.tile_m * (self.tile_k // 2)

    @property
    def b_tile_bytes(self) -> int:
        return self.tile_n * (self.tile_k // 2)

    @property
    def effective_prefetch_depth(self) -> int:
        return self.tdm_prefetch_depth

    @property
    def k_steps(self) -> int:
        return self.tile_k // _ATOM_K

    @property
    def uses_step6(self) -> bool:
        return any(getattr(self, name) for name in _STEP6_BOOL_FLAGS) or (
            self.cluster_dim is not None
        )

    @property
    def lds_layout(self) -> MxFp4LdsLayout:
        return mxfp4_lds_layout(self)

    @property
    def lds_bytes(self) -> int:
        return self.effective_prefetch_depth * self.lds_layout.buffer_bytes

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
                "vepi": self.vector_epilogue,
                "lrp": self.local_read_prefetch,
                "wptdm": self.wave_parity_tdm,
                "stdm": self.scale_tdm,
                "ldsp": self.lds_pad,
                "ldspck": self.lds_pad_ck,
                "ldssi": self.lds_segment_interleave,
                "initc": self.init_c_wmma,
                "clu": cluster_on,
                "mcast": self.tdm_multicast,
                "pgl2": self.prefetch_gl2,
                "rmid": self.tile_refill_mid,
            },
        )


_STEP6_BOOL_FLAGS = (
    "vector_epilogue",
    "local_read_prefetch",
    "wave_parity_tdm",
    "scale_tdm",
    "lds_pad",
    "lds_pad_ck",
    "lds_segment_interleave",
    "init_c_wmma",
    "tdm_multicast",
    "prefetch_gl2",
    "tile_refill_mid",
)


def _is_pow2(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def tdm_pad_fields(period_bytes: int, pad_bytes: int) -> dict[str, int]:
    """Raw TDM ``pad_interval`` / ``pad_amount`` for a pad of ``pad_bytes``
    after every ``period_bytes`` written (the TensileLite encoding)."""
    dwords = period_bytes // 4
    if not _is_pow2(dwords) or dwords < 2 or pad_bytes % 4 or pad_bytes <= 0:
        raise ValueError(f"unencodable TDM pad {pad_bytes} B every {period_bytes} B")
    return {
        "pad_interval": dwords.bit_length() - 2,
        "pad_amount": pad_bytes // 4 - 1,
    }


@dataclass(frozen=True)
class MxFp4LdsLayout:
    """Byte layout of one LDS buffer (the kernel double-buffers it).

    Each operand is split into ``*_components`` TDM transfers of
    ``*_component_rows`` rows. Component ``c`` of an operand starts at
    ``*_base + c * *_component_stride``. The ``*_offset`` methods take Python
    ints or IR values through ``ops`` so the host tests and the builder use
    the same address math.
    """

    row_bytes: int
    k_steps: int
    a_components: int
    a_component_rows: int
    a_base: int
    a_component_stride: int
    a_pad_period: int
    b_components: int
    b_component_rows: int
    b_base: int
    b_component_stride: int
    b_pad_period: int
    b_iterate_rows: int
    b_iterate_stride: int
    a_scale_base: int
    a_scale_component_stride: int
    b_scale_base: int
    b_scale_component_stride: int
    b_scale_pad_period: int
    buffer_bytes: int

    @property
    def a_scale_row_bytes(self) -> int:
        return self.a_component_rows * 4

    @property
    def b_scale_row_bytes(self) -> int:
        return self.b_component_rows * 4

    def a_component_base(self, ops, component):
        return ops.add(self.a_base, ops.mul(component, self.a_component_stride))

    def b_component_base(self, ops, component):
        return ops.add(self.b_base, ops.mul(component, self.b_component_stride))

    def a_scale_component_base(self, ops, component):
        return ops.add(
            self.a_scale_base, ops.mul(component, self.a_scale_component_stride)
        )

    def b_scale_component_base(self, ops, component):
        return ops.add(
            self.b_scale_base, ops.mul(component, self.b_scale_component_stride)
        )

    def a_row_offset(self, ops, row):
        """LDS byte offset of rocKE A tile row ``row`` (packed-K byte 0)."""
        rows = self.a_component_rows
        component = ops.div(row, rows)
        linear = ops.mul(ops.mod(row, rows), self.row_bytes)
        if self.a_pad_period:
            linear = ops.add(
                linear, ops.mul(ops.div(linear, self.a_pad_period), _PAD_BYTES)
            )
        return ops.add(self.a_component_base(ops, component), linear)

    def b_row_offset(self, ops, row):
        """LDS byte offset of rocKE B tile row ``row`` (packed-K byte 0)."""
        rows = self.b_component_rows
        component = ops.div(row, rows)
        local = ops.mod(row, rows)
        if self.b_iterate_rows:
            linear = ops.add(
                ops.mul(ops.div(local, self.b_iterate_rows), self.b_iterate_stride),
                ops.mul(ops.mod(local, self.b_iterate_rows), self.row_bytes),
            )
        else:
            linear = ops.mul(local, self.row_bytes)
            if self.b_pad_period:
                linear = ops.add(
                    linear, ops.mul(ops.div(linear, self.b_pad_period), _PAD_BYTES)
                )
        return ops.add(self.b_component_base(ops, component), linear)

    def a_scale_offset(self, ops, step, row):
        """LDS byte offset of the four E8M0 scales of A row ``row`` at K-step ``step``."""
        rows = self.a_component_rows
        linear = ops.add(
            ops.mul(step, self.a_scale_row_bytes), ops.mul(ops.mod(row, rows), 4)
        )
        return ops.add(self.a_scale_component_base(ops, ops.div(row, rows)), linear)

    def b_scale_offset(self, ops, step, col):
        """LDS byte offset of the four E8M0 scales of B row ``col`` at K-step ``step``."""
        rows = self.b_component_rows
        linear = ops.add(
            ops.mul(step, self.b_scale_row_bytes), ops.mul(ops.mod(col, rows), 4)
        )
        if self.b_scale_pad_period:
            linear = ops.add(
                linear,
                ops.mul(ops.div(linear, self.b_scale_pad_period), _PAD_BYTES),
            )
        return ops.add(self.b_scale_component_base(ops, ops.div(col, rows)), linear)


def ck_pad_period(row_bytes: int) -> int:
    """CK Tile's A/B pad period for ``row_bytes``-byte LDS rows."""
    return max(_CK_PAD_SWEEP, row_bytes)


def mxfp4_lds_layout(spec: MxFp4TdmGemmSpec) -> MxFp4LdsLayout:
    """LDS buffer layout for ``spec``: ``[B][A][B scales][A scales]``, or with
    ``lds_segment_interleave`` ``[B0][A][B1][B scales][A scales]``."""
    row_bytes = spec.tile_k // 2
    components = 2 if spec.wave_parity_tdm else 1
    a_rows = spec.tile_m // components
    b_rows = spec.tile_n // components
    if spec.lds_pad_ck:
        a_pad_period = b_pad_period = ck_pad_period(row_bytes)
    else:
        a_pad_period = _A_PAD_PERIOD if spec.lds_pad else 0
        b_pad_period = 0
    a_component = a_rows * row_bytes
    if a_pad_period:
        a_component += a_component // a_pad_period * _PAD_BYTES
    if spec.lds_pad and row_bytes and _B_PAD_PERIOD % row_bytes == 0:
        b_iterate_rows = _B_PAD_PERIOD // row_bytes
        b_iterate_stride = _B_PAD_PERIOD + _PAD_BYTES
        b_component = -(-b_rows // b_iterate_rows) * b_iterate_stride
    else:
        b_iterate_rows = 0
        b_iterate_stride = 0
        b_component = b_rows * row_bytes
        if b_pad_period:
            b_component += b_component // b_pad_period * _PAD_BYTES
    steps = spec.k_steps
    a_scale_component = steps * a_rows * 4 if spec.scale_tdm else 0
    b_scale_pad_period = _B_SCALE_PAD_PERIOD if spec.scale_tdm and spec.lds_pad else 0
    b_scale_component = steps * b_rows * 4 if spec.scale_tdm else 0
    if b_scale_pad_period:
        b_scale_component += b_scale_component // b_scale_pad_period * _PAD_BYTES
    a_total = components * a_component
    if spec.lds_segment_interleave and components == 2:
        b_base, a_base = 0, b_component
        b_stride = b_component + a_total
        data_end = a_total + 2 * b_component
    else:
        b_base, b_stride = 0, b_component
        a_base = components * b_component
        data_end = a_base + a_total
    b_scale_region = components * b_scale_component
    if b_scale_pad_period:
        b_scale_region = _round_up(b_scale_region, _B_SCALE_REGION_ALIGN)
    b_scale_base = data_end
    a_scale_base = b_scale_base + b_scale_region
    buffer_bytes = a_scale_base + components * a_scale_component
    return MxFp4LdsLayout(
        row_bytes=row_bytes,
        k_steps=steps,
        a_components=components,
        a_component_rows=a_rows,
        a_base=a_base,
        a_component_stride=a_component,
        a_pad_period=a_pad_period,
        b_components=components,
        b_component_rows=b_rows,
        b_base=b_base,
        b_component_stride=b_stride,
        b_pad_period=b_pad_period,
        b_iterate_rows=b_iterate_rows,
        b_iterate_stride=b_iterate_stride,
        a_scale_base=a_scale_base,
        a_scale_component_stride=a_scale_component,
        b_scale_base=b_scale_base,
        b_scale_component_stride=b_scale_component,
        b_scale_pad_period=b_scale_pad_period,
        buffer_bytes=buffer_bytes,
    )


class IntOps:
    """Address arithmetic over Python ints (host model of the IR math)."""

    @staticmethod
    def add(x, y):
        return x + y

    @staticmethod
    def mul(x, y):
        return x * y

    @staticmethod
    def div(x, d: int):
        return x // d

    @staticmethod
    def mod(x, d: int):
        return x % d


class _IrOps:
    """Address arithmetic on non-negative i32 IR values, folding Python ints."""

    def __init__(self, b: IRBuilder):
        self.b = b

    def v(self, x):
        return self.b.const_i32(x) if isinstance(x, int) else x

    def add(self, x, y):
        if isinstance(x, int) and isinstance(y, int):
            return x + y
        if isinstance(y, int) and y == 0:
            return x
        if isinstance(x, int) and x == 0:
            return y
        return self.b.add(self.v(x), self.v(y))

    def mul(self, x, y):
        if isinstance(x, int) and isinstance(y, int):
            return x * y
        if isinstance(y, int):
            if y == 0:
                return 0
            if y == 1:
                return x
            if _is_pow2(y):
                return self.b.shl(x, self.b.const_i32(y.bit_length() - 1))
        if isinstance(x, int):
            return self.mul(y, x)
        return self.b.mul(self.v(x), self.v(y))

    def div(self, x, d: int):
        if isinstance(x, int):
            return x // d
        if d == 1:
            return x
        if _is_pow2(d):
            return self.b.lshr(x, self.b.const_i32(d.bit_length() - 1))
        return self.b.div(x, self.b.const_i32(d))

    def mod(self, x, d: int):
        if isinstance(x, int):
            return x % d
        if d == 1:
            return 0
        if _is_pow2(d):
            return self.b.land(x, self.b.const_i32(d - 1))
        return self.b.mod(x, self.b.const_i32(d))


def _step6_reason(spec: MxFp4TdmGemmSpec) -> str | None:
    """Flag dependencies and geometry limits of the Step 6 features."""
    if spec.vector_epilogue and spec.wave_repeat_n not in (4, 8, 16):
        return "vector_epilogue needs wave_repeat_n in (4, 8, 16)"
    if spec.local_read_prefetch and spec.wave_repeat_n % 2:
        return "local_read_prefetch (HalfPLR on B) needs an even wave_repeat_n"
    if spec.wave_parity_tdm and (spec.warp_m, spec.warp_n) != (2, 2):
        return "wave_parity_tdm needs 2x2 waves (two components per operand)"
    if spec.scale_tdm:
        if not spec.wave_parity_tdm:
            return "scale_tdm requires wave_parity_tdm"
        if not spec.vector_epilogue:
            return "scale_tdm requires vector_epilogue (swizzled B scales follow its N mapping)"
        if spec.wave_repeat_m != 4 or spec.wave_repeat_n % 4:
            return "scale_tdm needs wave_repeat_m == 4 and wave_repeat_n % 4 == 0"
    if spec.lds_pad:
        row_bytes = spec.tile_k // 2
        if _A_PAD_PERIOD % row_bytes or _B_PAD_PERIOD % row_bytes:
            return "lds_pad needs tile_k / 2 to divide the 1024 B and 2048 B pad periods"
    if spec.lds_pad_ck:
        row_bytes = spec.tile_k // 2
        if spec.lds_pad:
            return "lds_pad_ck and lds_pad are alternative LDS paddings"
        if not _is_pow2(row_bytes) or ck_pad_period(row_bytes) > _TDM_PAD_PERIOD_MAX:
            return "lds_pad_ck needs tile_k / 2 to be a power of two of at most 1024 B"
    if spec.lds_segment_interleave and not (spec.lds_pad and spec.wave_parity_tdm):
        return "lds_segment_interleave requires lds_pad and wave_parity_tdm"
    if spec.init_c_wmma and spec.K < 2 * spec.tile_k:
        return "init_c_wmma peels the first and last K tiles (needs K >= 2 * tile_k)"
    if spec.cluster_dim is not None:
        if len(spec.cluster_dim) != 3 or any(d < 1 for d in spec.cluster_dim):
            return "cluster_dim must be a positive (x, y, z) triple"
        cx, cy, cz = spec.cluster_dim
        if cz != 1 or cx * cy > 16:
            return "cluster_dim must be 2D with at most 16 workgroups"
        if not ArchTarget.from_gfx("gfx1250").memory.has_cluster_launch:
            return "gfx1250 reports no cluster launch support"
        gx, gy, _ = mxfp4_tdm_gemm_grid(spec)
        if gx % cx or gy % cy:
            return f"grid {gx}x{gy} must be divisible by cluster_dim {cx}x{cy}"
    if spec.tdm_multicast and (spec.cluster_dim is None or not spec.wave_parity_tdm):
        return "tdm_multicast requires cluster_dim and wave_parity_tdm"
    if spec.prefetch_gl2:
        if spec.cluster_dim is None:
            return "prefetch_gl2 requires cluster_dim for the cooperative split"
        if (spec.tile_k // 2) % _GL2_LINE_BYTES:
            return "prefetch_gl2 needs tile_k / 2 to be a multiple of 256 B"
    if spec.tile_refill_mid and not (
        spec.local_read_prefetch and spec.init_c_wmma and spec.scale_tdm
    ):
        return "tile_refill_mid requires local_read_prefetch, init_c_wmma and scale_tdm"
    return None


def is_valid_spec(spec: MxFp4TdmGemmSpec, arch: str = "gfx1250") -> tuple[bool, str]:
    if arch != "gfx1250":
        return False, f"mxfp4_tdm_gemm is gfx1250-only (got {arch!r})"
    if min(spec.M, spec.N, spec.K) <= 0:
        return False, "M, N, and K must be positive"
    if not spec.tdm_lds or not spec.tdm_prefetch:
        return False, "Step 4 requires TDM/LDS staging with prefetch enabled"
    if spec.tdm_prefetch_depth != 2:
        return False, "tdm_prefetch_depth must be 2 (double-buffered LDS)"
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
    reason = _step6_reason(spec)
    if reason:
        return False, reason
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
    if spec.uses_step6:
        return _build_step6(spec)
    return _build_base(spec)


def _build_base(spec: MxFp4TdmGemmSpec) -> KernelDef:
    """The Step 5 kernel; the C++ engine mirrors it byte for byte."""
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
    b_region = spec.tdm_prefetch_depth * spec.a_tile_bytes

    def _buffer_offsets(parity):
        a_offset = b.mul(parity, b.const_i32(spec.a_tile_bytes))
        b_offset = b.add(
            b.const_i32(b_region),
            b.mul(parity, b.const_i32(spec.b_tile_bytes)),
        )
        return a_offset, b_offset

    def _issue_tdm(k0, parity) -> None:
        a_lds_offset, b_lds_offset = _buffer_offsets(parity)
        a_lds = b.smem_ptr_add(shared_addr, b.zext(a_lds_offset, shared_addr.type))
        b_lds = b.smem_ptr_add(shared_addr, b.zext(b_lds_offset, shared_addr.type))
        k_byte = b.div(k0, b.const_i32(2))
        a_element = b.add(b.mul(block_m, packed_k), k_byte)
        b_element = b.add(b.mul(block_n, packed_k), k_byte)
        remaining_k = b.sub(packed_k, k_byte)
        with b.scf_if(b.cmp_eq(wave, c0)):
            b.tensor_load_to_lds(
                *b.tdm_descriptor_2d(
                    b.global_ptr_add(A, a_element),
                    a_lds,
                    elem_bytes=1,
                    tensor_dim0=remaining_k,
                    tensor_dim1=b.sub(M, block_m),
                    row_stride=packed_k,
                    tile_dim0=spec.tile_k // 2,
                    tile_dim1=spec.tile_m,
                )
            )
            b.tensor_load_to_lds(
                *b.tdm_descriptor_2d(
                    b.global_ptr_add(B, b_element),
                    b_lds,
                    elem_bytes=1,
                    tensor_dim0=remaining_k,
                    tensor_dim1=b.sub(N, block_n),
                    row_stride=packed_k,
                    tile_dim0=spec.tile_k // 2,
                    tile_dim1=spec.tile_n,
                )
            )

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

    def _pack_a_scale(global_row, group0):
        index = b.add(b.mul(global_row, scale_groups), group0)
        raw = b.global_load_vN(AS, index, I8, 4, align=1)
        return b.bitcast(raw, I32)

    def _pack_b_scale(global_col, group0):
        values = []
        for group in range(4):
            index = b.add(
                b.mul(b.add(group0, b.const_i32(group)), N),
                global_col,
            )
            values.append(b.global_load(BS, index, I8, align=1))
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

    def _compute_tile(k0, parity, accumulators):
        a_buffer, b_buffer = _buffer_offsets(parity)
        out = list(accumulators)
        for inner_k in range(0, spec.tile_k, _ATOM_K):
            group0 = b.div(b.add(k0, b.const_i32(inner_k)), c32)
            for rm, atom_row in enumerate(atom_rows):
                a_frag = _load_a_fragment(a_buffer, atom_row, inner_k)
                a_scale = _pack_a_scale(b.add(block_m, b.add(atom_row, lane)), group0)
                for rn, atom_col in enumerate(atom_cols):
                    index = rm * spec.wave_repeat_n + rn
                    b_frag = _load_b_fragment(b_buffer, atom_col, inner_k)
                    b_scale = _pack_b_scale(
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


def _build_step6(spec: MxFp4TdmGemmSpec) -> KernelDef:
    """The base kernel plus the Step 6 YAML-winner features selected in ``spec``.

    Tile-local rows: rocKE A row ``m`` and rocKE B row ``n`` (an N index).
    The WMMA's 32 M positions and 16 N positions are mapped to tile rows by
    :func:`m_row` / :func:`n_row`; the WMMA itself does not care which rows
    it gets, so the mapping only has to agree between the A/B fragments, the
    scales and the epilogue.
    """
    layout = spec.lds_layout
    rm_count = spec.wave_repeat_m
    rn_count = spec.wave_repeat_n
    steps = spec.k_steps
    row_bytes = layout.row_bytes

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block_size
    if spec.cluster_dim is not None:
        b.set_cluster_dims(*spec.cluster_dim)
    ops = _IrOps(b)

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
    tile_k_v = b.const_i32(spec.tile_k)
    packed_k = ops.div(K, 2)
    scale_groups = ops.div(K, 32)

    tid = b.thread_id_x()
    lane = ops.mod(tid, _WAVE)
    lane_col = ops.mod(lane, 16)
    lane_half = ops.div(lane, 16)
    # Wave-uniform, so the TDM descriptors built from it stay scalar.
    wave = b.readfirstlane(ops.div(tid, _WAVE))
    wave_n = ops.mod(wave, spec.warp_n)
    wave_m = ops.div(wave, spec.warp_n)

    block_m = ops.mul(b.block_id_y(), spec.tile_m)
    block_n = ops.mul(b.block_id_x(), spec.tile_n)

    def m_row(rm, m_local):
        """Tile row of rocKE A for M-repeat ``rm`` and WMMA M position ``m_local``.

        With ``scale_tdm`` the four M-repeats of a WMMA position are adjacent
        rows, so one lane's A scales for a K-step are 16 contiguous bytes.
        """
        base = ops.mul(wave_m, rm_count * _ATOM_M)
        if spec.scale_tdm:
            return ops.add(base, ops.add(ops.mul(m_local, rm_count), rm))
        return ops.add(base, ops.add(rm * _ATOM_M, m_local))

    def n_row(rn, n_local):
        """Tile row of rocKE B (an N index) for N-repeat ``rn`` and WMMA N position.

        With ``vector_epilogue`` the N-repeats of a WMMA position are adjacent
        columns of D, so a lane's accumulators for one slot are one vector store.
        """
        base = ops.mul(wave_n, rn_count * _ATOM_N)
        if spec.vector_epilogue:
            return ops.add(base, ops.add(ops.mul(n_local, rn_count), rn))
        return ops.add(base, ops.add(rn * _ATOM_N, n_local))

    a_mask: int | Value = 0
    b_mask: int | Value = 0
    cluster_wx = cluster_wy = None
    if spec.cluster_dim is not None:
        cluster_x, cluster_y, _ = spec.cluster_dim
        cluster_wx = b.cluster_workgroup_id("x")
        cluster_wy = b.cluster_workgroup_id("y")
        if spec.tdm_multicast:
            # Grid x walks N tiles and y walks M tiles: workgroups with the
            # same cluster x share rocKE B, those with the same y share rocKE A.
            # Bit i of a mask is cluster flat id i = x + cluster_x * y.
            col_bits = sum(1 << (row * cluster_x) for row in range(cluster_y))
            row_bits = (1 << cluster_x) - 1
            b_mask = b.shl(b.const_i32(col_bits), cluster_wx)
            a_mask = b.shl(
                b.const_i32(row_bits), ops.mul(cluster_wy, cluster_x)
            )

    shared = b.smem_alloc(I8, [spec.lds_bytes], name_hint="ab_tdm")
    shared_addr = b.smem_addr_of(shared)

    def buffer_base(parity):
        return ops.mul(parity, layout.buffer_bytes)

    def lds_ptr(offset):
        return b.smem_ptr_add(shared_addr, b.zext(ops.v(offset), shared_addr.type))

    def data_descriptor(operand: str, parity, component, k_byte, remaining_k):
        if operand == "a":
            rows, block, extent, base_ptr, mask = (
                layout.a_component_rows,
                block_m,
                M,
                A,
                a_mask,
            )
            lds = ops.add(buffer_base(parity), layout.a_component_base(ops, component))
        else:
            rows, block, extent, base_ptr, mask = (
                layout.b_component_rows,
                block_n,
                N,
                B,
                b_mask,
            )
            lds = ops.add(buffer_base(parity), layout.b_component_base(ops, component))
        row0 = ops.add(block, ops.mul(component, rows))
        element = ops.add(ops.mul(row0, packed_k), k_byte)
        extra: dict = {}
        tile_rows = rows
        if operand == "a" and layout.a_pad_period:
            extra.update(tdm_pad_fields(layout.a_pad_period, _PAD_BYTES))
        if operand == "b" and layout.b_pad_period:
            extra.update(tdm_pad_fields(layout.b_pad_period, _PAD_BYTES))
        if operand == "b" and layout.b_iterate_rows:
            tile_rows = layout.b_iterate_rows
            extra.update(
                iterate_count=rows // tile_rows,
                lds_increment=layout.b_iterate_stride,
                global_increment=ops.v(ops.mul(packed_k, tile_rows)),
            )
        return b.tdm_descriptor_2d(
            b.global_ptr_add(base_ptr, ops.v(element)),
            lds_ptr(lds),
            elem_bytes=1,
            tensor_dim0=remaining_k,
            tensor_dim1=b.sub(extent, ops.v(row0)),
            row_stride=packed_k,
            tile_dim0=row_bytes,
            tile_dim1=tile_rows,
            workgroup_mask=mask,
            **extra,
        )

    def scale_descriptor(operand: str, parity, component, k_step):
        """Swizzled scales ``[K/128][MN][4]``: one row per K-step, 4 B per MN."""
        if operand == "a":
            rows, block, extent, base_ptr, mask = (
                layout.a_component_rows,
                block_m,
                M,
                AS,
                a_mask,
            )
            lds = ops.add(
                buffer_base(parity), layout.a_scale_component_base(ops, component)
            )
            extra: dict = {}
        else:
            rows, block, extent, base_ptr, mask = (
                layout.b_component_rows,
                block_n,
                N,
                BS,
                b_mask,
            )
            lds = ops.add(
                buffer_base(parity), layout.b_scale_component_base(ops, component)
            )
            extra = (
                tdm_pad_fields(layout.b_scale_pad_period, _PAD_BYTES)
                if layout.b_scale_pad_period
                else {}
            )
        row_stride = ops.mul(extent, 4)
        col0 = ops.mul(ops.add(block, ops.mul(component, rows)), 4)
        element = ops.add(ops.mul(k_step, row_stride), col0)
        return b.tdm_descriptor_2d(
            b.global_ptr_add(base_ptr, ops.v(element)),
            lds_ptr(lds),
            elem_bytes=1,
            tensor_dim0=b.sub(ops.v(row_stride), ops.v(col0)),
            tensor_dim1=b.sub(ops.div(K, _ATOM_K), ops.v(k_step)),
            row_stride=ops.v(row_stride),
            tile_dim0=rows * 4,
            tile_dim1=steps,
            workgroup_mask=mask,
            **extra,
        )

    def issue_tdm(k0, parity) -> None:
        k_byte = ops.div(k0, 2)
        remaining_k = b.sub(packed_k, k_byte)
        if spec.wave_parity_tdm:
            # Even waves copy rocKE B, odd waves rocKE A; wave // 2 picks the
            # 128-row component. One branch-free load per wave. 6.4 and 6.5
            # share this select; the scale load is the only 6.5 addition.
            even = b.cmp_eq(ops.mod(wave, 2), c0)
            component = ops.div(wave, 2)

            def load_selected(b_desc, a_desc) -> None:
                b.tensor_load_to_lds(
                    *[b.select(even, x, y) for x, y in zip(b_desc, a_desc)]
                )

            load_selected(
                data_descriptor("b", parity, component, k_byte, remaining_k),
                data_descriptor("a", parity, component, k_byte, remaining_k),
            )
            if spec.scale_tdm:
                k_step = ops.div(k0, _ATOM_K)
                load_selected(
                    scale_descriptor("b", parity, component, k_step),
                    scale_descriptor("a", parity, component, k_step),
                )
            return
        with b.scf_if(b.cmp_eq(wave, c0)):
            b.tensor_load_to_lds(*data_descriptor("a", parity, 0, k_byte, remaining_k))
            b.tensor_load_to_lds(*data_descriptor("b", parity, 0, k_byte, remaining_k))

    def prefetch_gl2(k_ahead) -> None:
        """PrefetchGL2: the cluster's A/B (and scale) lines of the tile at
        ``k_ahead``, split over every thread of the cluster, 256 B each."""
        if not spec.prefetch_gl2:
            return
        assert spec.cluster_dim is not None
        cluster_x, cluster_y, _ = spec.cluster_dim
        # Branch-free: any branch in the loop body makes LLVM drain the LDS
        # read pipeline. Past the end, the last tile is prefetched again.
        k_ahead = b.select(b.cmp_lt(k_ahead, K), k_ahead, b.sub(K, tile_k_v))
        k_byte = ops.div(k_ahead, 2)
        data_lines = row_bytes // _GL2_LINE_BYTES
        cluster_m0 = b.sub(block_m, ops.v(ops.mul(cluster_wy, spec.tile_m)))
        cluster_n0 = b.sub(block_n, ops.v(ops.mul(cluster_wx, spec.tile_n)))

        def data_line(base_ptr, row0):
            def address(index):
                row = ops.add(row0, ops.div(index, data_lines))
                offset = ops.add(
                    k_byte, ops.mul(ops.mod(index, data_lines), _GL2_LINE_BYTES)
                )
                return b.global_ptr_add(
                    base_ptr, ops.v(ops.add(ops.mul(row, packed_k), offset))
                )

            return address

        def scale_line(base_ptr, extent, mn0, mn_count):
            lines_per_row = mn_count * 4 // _GL2_LINE_BYTES
            k_step = ops.div(k_ahead, _ATOM_K)

            def address(index):
                row = ops.add(k_step, ops.div(index, lines_per_row))
                offset = ops.add(
                    ops.mul(mn0, 4),
                    ops.mul(ops.mod(index, lines_per_row), _GL2_LINE_BYTES),
                )
                return b.global_ptr_add(
                    base_ptr,
                    ops.v(ops.add(ops.mul(row, ops.mul(extent, 4)), offset)),
                )

            return address, steps * lines_per_row

        categories = [
            (cluster_y * spec.tile_m * data_lines, data_line(A, cluster_m0)),
            (cluster_x * spec.tile_n * data_lines, data_line(B, cluster_n0)),
        ]
        if spec.scale_tdm and (cluster_x * spec.tile_n * 4) % _GL2_LINE_BYTES == 0:
            b_scale_address, b_scale_lines = scale_line(
                BS, N, cluster_n0, cluster_x * spec.tile_n
            )
            a_scale_address, a_scale_lines = scale_line(
                AS, M, cluster_m0, cluster_y * spec.tile_m
            )
            categories += [
                (b_scale_lines, b_scale_address),
                (a_scale_lines, a_scale_address),
            ]
        starts = []
        total = 0
        for count, _ in categories:
            starts.append(total)
            total += count
        pool = cluster_x * cluster_y * spec.block_size
        flat = ops.add(cluster_wx, ops.mul(cluster_wy, cluster_x))
        thread = ops.add(ops.mul(flat, spec.block_size), tid)

        def prefetch_round(round_start: int, index) -> None:
            live = [
                (start, count, address)
                for start, (count, address) in zip(starts, categories)
                if start < round_start + pool and start + count > round_start
            ]
            # Ranges are visited last to first, so each lane ends up with
            # the first range whose end lies above its index.
            pointer = None
            for start, count, address in reversed(live):
                local = b.sub(ops.v(index), ops.v(start)) if start else index
                if pointer is None:
                    pointer = address(local)
                    continue
                in_range = b.cmp_lt(ops.v(index), ops.v(start + count))
                pointer = b.select(in_range, address(local), pointer)
            b.global_prefetch(pointer, cachepolicy=_GL2_PREFETCH_CPOL)

        for round_start in range(0, total, pool):
            remaining = total - round_start
            if remaining >= pool:
                index = ops.add(thread, round_start)
            else:
                # Every lane covers one of the remaining lines; some
                # lines are prefetched more than once.
                index = ops.add(ops.mod(thread, remaining), round_start)
            prefetch_round(round_start, index)

    def load_i32x4(offset):
        return b.smem_load_vN(shared, ops.v(offset), dtype=I32, n=4)

    def load_a_fragment(buffer, rm, step):
        lane_k = ops.add(step * (_ATOM_K // 2), ops.mul(lane_half, 16))
        rows = []
        for m_local in (lane_col, ops.add(lane_col, 16)):
            base = ops.add(
                ops.add(buffer, layout.a_row_offset(ops, m_row(rm, m_local))), lane_k
            )
            rows.append(b.vec_concat(load_i32x4(base), load_i32x4(ops.add(base, 32))))
        return b.vec_concat(rows[0], rows[1])

    def load_b_fragment(buffer, rn, step):
        lane_k = ops.add(step * (_ATOM_K // 2), ops.mul(lane_half, 16))
        base = ops.add(
            ops.add(buffer, layout.b_row_offset(ops, n_row(rn, lane_col))), lane_k
        )
        return b.vec_concat(load_i32x4(base), load_i32x4(ops.add(base, 32)))

    def load_scales(buffer, k0, step):
        """Per-lane E8M0 words for one K-step: ``rm_count`` for A, ``rn_count`` for B."""
        if spec.scale_tdm:
            a_words = load_i32x4(
                ops.add(buffer, layout.a_scale_offset(ops, step, m_row(0, lane)))
            )
            a_scales = [b.vec_extract(a_words, rm) for rm in range(rm_count)]
            b_base = ops.add(buffer, layout.b_scale_offset(ops, step, n_row(0, lane_col)))
            b_scales = []
            for chunk in range(rn_count // 4):
                words = load_i32x4(ops.add(b_base, chunk * 16))
                b_scales += [b.vec_extract(words, i) for i in range(4)]
            return a_scales, b_scales
        group0 = ops.div(ops.add(k0, step * _ATOM_K), 32)
        a_scales = []
        for rm in range(rm_count):
            index = ops.add(
                ops.mul(ops.add(block_m, m_row(rm, lane)), scale_groups), group0
            )
            raw = b.global_load_vN(AS, ops.v(index), I8, 4, align=1)
            a_scales.append(b.bitcast(raw, I32))
        b_scales = []
        for rn in range(rn_count):
            col = ops.add(block_n, n_row(rn, lane_col))
            values = [
                b.global_load(
                    BS, ops.v(ops.add(ops.mul(ops.add(group0, g), N), col)), I8, align=1
                )
                for g in range(_SCALE_GROUPS_PER_STEP)
            ]
            b_scales.append(b.bitcast(b.vec_pack(values, I8), I32))
        return a_scales, b_scales

    def compute_tile(k0, parity, accumulators, *, seed_zero=False, refill=None):
        """One DepthU tile. ``seed_zero`` gives the first K-step's WMMAs C = 0.

        ``refill`` runs before the last K-step's WMMAs, after the tile's last
        LDS read (local_read_prefetch only)."""
        buffer = buffer_base(parity)
        out = list(accumulators)
        zero = b.zero_vec_f32(_ACCUMULATORS_PER_ATOM) if seed_zero else None

        def mma(rm, rn, a_frag, b_frag, a_scale, b_scale, first_step):
            index = rm * rn_count + rn
            c_in = zero if (seed_zero and first_step) else out[index]
            out[index] = b.mma(
                _WmmaScaleF4Atom(), a_frag, b_frag, c_in, a_scale, b_scale
            )

        if not spec.local_read_prefetch:
            for step in range(steps):
                a_scales, b_scales = load_scales(buffer, k0, step)
                for rm in range(rm_count):
                    a_frag = load_a_fragment(buffer, rm, step)
                    for rn in range(rn_count):
                        b_frag = load_b_fragment(buffer, rn, step)
                        mma(rm, rn, a_frag, b_frag, a_scales[rm], b_scales[rn], step == 0)
            return out

        # PrefetchLocalRead=1 + HalfPLR: everything for K-step s+1 is read
        # during K-step s. A and the scales get a second full register set;
        # B is read in halves so only three half-sets are live: the next
        # step's first half is read before this step's WMMAs, its second
        # half after this step's first half has been consumed.
        half = rn_count // 2

        def load_head(step):
            a_scales, b_scales = load_scales(buffer, k0, step)
            a_frags = [load_a_fragment(buffer, rm, step) for rm in range(rm_count)]
            return a_frags, a_scales, b_scales

        def load_b_half(step, which):
            return [load_b_fragment(buffer, rn, step) for rn in range(which * half, (which + 1) * half)]

        head = load_head(0)
        b_halves = [load_b_half(0, 0), load_b_half(0, 1)]
        if refill is not None:
            # Otherwise LLVM mixes K-step 1's reads into K-step 0's and the
            # first WMMA waits for twice as many reads.
            b.sched_barrier(_SCHED_ALU_ONLY)
        for step in range(steps):
            a_frags, a_scales, b_scales = head
            last = step + 1 == steps
            if not last:
                head = load_head(step + 1)
                next_first = load_b_half(step + 1, 0)
            elif refill is not None:
                refill()
            for which in (0, 1):
                if which == 1 and not last:
                    next_second = load_b_half(step + 1, 1)
                for rm in range(rm_count):
                    for j in range(half):
                        rn = which * half + j
                        mma(
                            rm,
                            rn,
                            a_frags[rm],
                            b_halves[which][j],
                            a_scales[rm],
                            b_scales[rn],
                            step == 0,
                        )
            if not last:
                b_halves = [next_first, next_second]
        return out

    def cluster_sync() -> None:
        b.inline_asm(_CLUSTER_BARRIER_ASM, "~{scc},~{memory}")

    def wait_tile_ready() -> None:
        b.s_wait_tensorcnt(0)
        b.s_waitcnt(lgkmcnt=0)
        if spec.tdm_multicast:
            # Peers write this workgroup's LDS: no buffer may be refilled until
            # every workgroup of the cluster has finished reading it.
            cluster_sync()
        else:
            b.s_barrier_bare()

    def refill_mid_pipeline():
        """PrefetchGlobalRead=2: the TDM runs two tiles ahead. Each tile
        refills its own buffer with tile k+2 once its last LDS read is done,
        before its last K-step's WMMAs. Tile k+1 is waited for at the tile
        boundary, with tile k+2's loads still in flight.
        SuppressNoLoadLoop: past the end the last tile is loaded again into
        a buffer nobody reads, so the loop needs no guard or peeled tail."""
        loads_per_tile = 2 if spec.scale_tdm else 1
        # PGR + PrefetchGL2 = 4 tiles ahead, as TensileLite's skipPGR.
        prefetch_ahead = 4

        def clamp_k(k):
            return b.select(b.cmp_lt(k, K), k, b.sub(K, tile_k_v))

        def fence() -> None:
            # Barrier and wait intrinsics do not access memory, and WMMAs
            # are register-only: keep LLVM from moving LDS reads or WMMAs
            # across the refill, or it reads the whole tile up front and
            # refills after the first K-step. Scalar and vector ALU work
            # (descriptor and address math) may cross.
            b.inline_asm("", "~{memory}")
            b.sched_barrier(_SCHED_ALU_ONLY)

        def refill(k0, parity):
            def run() -> None:
                prefetch_gl2(ops.v(ops.add(k0, prefetch_ahead * spec.tile_k)))
                fence()
                b.s_waitcnt(lgkmcnt=0)
                if spec.tdm_multicast:
                    cluster_sync()
                else:
                    b.s_barrier_bare()
                issue_tdm(clamp_k(ops.v(ops.add(k0, two_tiles))), parity)
                fence()

            return run

        def wait_next_tile() -> None:
            b.s_wait_tensorcnt(loads_per_tile)
            b.s_barrier_bare()
            b.inline_asm("", "~{memory}")

        if spec.tdm_multicast:
            cluster_sync()
        issue_tdm(c0, c0)
        issue_tdm(tile_k_v, c1)
        for tile in range(2, prefetch_ahead):
            prefetch_gl2(ops.v(tile * spec.tile_k))
        wait_next_tile()
        undef = [b.undef_vec(F32, _ACCUMULATORS_PER_ATOM) for _ in range(n_acc)]
        accumulators = compute_tile(
            c0, c0, undef, seed_zero=True, refill=refill(c0, c0)
        )
        loop = b.scf_for_iter(
            tile_k_v,
            K,
            tile_k_v,
            [("parity", c1)]
            + [(f"acc{i}", value) for i, value in enumerate(accumulators)],
            iv_name="k0",
            elide_trailing_barrier=False,
        )
        with loop as (k0, carried):
            parity = carried[0]
            wait_next_tile()
            updated = compute_tile(k0, parity, carried[1:], refill=refill(k0, parity))
            b.scf_yield(b.sub(c1, parity), *updated)
        # No workgroup may exit while a TDM load into its LDS is in flight:
        # with multicast, peers' loads write here too.
        b.s_wait_tensorcnt(0)
        if spec.tdm_multicast:
            cluster_sync()
        return loop.results[1:]

    n_acc = rm_count * rn_count
    two_tiles = 2 * spec.tile_k
    if spec.tile_refill_mid:
        final = refill_mid_pipeline()
    else:
        if spec.tdm_multicast:
            cluster_sync()
        issue_tdm(c0, c0)
        if spec.init_c_wmma:
            # InitCIterWmma: peel the first K tile so its first WMMAs read C = 0;
            # the steady-state loop carries no accumulator clear.
            # As in the winner, the GL2 prefetch is issued ahead of the cluster
            # wait, so it overlaps the barrier.
            prefetch_gl2(ops.v(two_tiles))
            wait_tile_ready()
            issue_tdm(tile_k_v, c1)
            undef = [b.undef_vec(F32, _ACCUMULATORS_PER_ATOM) for _ in range(n_acc)]
            accumulators = compute_tile(c0, c0, undef, seed_zero=True)
            start, start_parity = tile_k_v, c1
        else:
            accumulators = [
                b.zero_vec_f32(_ACCUMULATORS_PER_ATOM) for _ in range(n_acc)
            ]
            start, start_parity = c0, c0
        upper = b.sub(K, tile_k_v)
        loop = b.scf_for_iter(
            start,
            upper,
            tile_k_v,
            [("parity", start_parity)]
            + [(f"acc{i}", value) for i, value in enumerate(accumulators)],
            iv_name="k0",
            elide_trailing_barrier=False,
        )
        with loop as (k0, carried):
            parity = carried[0]
            next_parity = b.sub(c1, parity)
            prefetch_gl2(ops.v(ops.add(k0, two_tiles)))
            wait_tile_ready()
            issue_tdm(ops.add(k0, spec.tile_k), next_parity)
            updated = compute_tile(k0, parity, carried[1:])
            b.scf_yield(next_parity, *updated)
        wait_tile_ready()
        final = compute_tile(upper, loop.results[0], loop.results[1:])

    def saturate(value):
        # hipBLASLt's FP8 store: +-inf -> NaN (x * 0 + x), finite values
        # clamped to +-448, NaN kept.
        value = b.fma(value, b.const_f32(0.0), value)
        clamped = b.clamp_f32(value, b.const_f32(-_FP8_MAX), b.const_f32(_FP8_MAX))
        return b.select(b.fcmp("uno", value, value), value, clamped)

    def store_vector(read_c: bool) -> None:
        col = ops.add(block_n, n_row(0, lane_col))
        for rm in range(rm_count):
            for slot in range(_ACCUMULATORS_PER_ATOM):
                m_local = ops.add((slot // 8) * 16 + slot % 8, ops.mul(lane_half, 8))
                row = ops.add(block_m, m_row(rm, m_local))
                index = ops.v(ops.add(ops.mul(row, N), col))
                values = [
                    b.fmul(alpha, b.vec_extract(final[rm * rn_count + rn], slot))
                    for rn in range(rn_count)
                ]
                if read_c:
                    c_words = b.global_load_vN(C, index, FP8E4M3, rn_count, align=rn_count)
                    values = [
                        b.fadd(
                            value,
                            b.fmul(beta, b.cvt_fp8_to_f32(b.vec_extract(c_words, i))),
                        )
                        for i, value in enumerate(values)
                    ]
                values = [saturate(value) for value in values]
                packed = None
                for chunk in range(rn_count // 4):
                    quad = b.cvt_pk_fp8_f32x4(
                        b.vec_pack(values[chunk * 4 : chunk * 4 + 4], F32)
                    )
                    packed = quad if packed is None else b.vec_concat(packed, quad)
                b.global_store_vN(D, index, packed, rn_count, align=rn_count)

    if spec.vector_epilogue:
        # GW_B0 / GW_B1: the beta = 0 path never reads C.
        with b.scf_if_else(b.fcmp("oeq", beta, b.const_f32(0.0))) as (then_, else_):
            with then_:
                store_vector(read_c=False)
            with else_:
                store_vector(read_c=True)
    else:
        for rm in range(rm_count):
            for rn in range(rn_count):
                acc = final[rm * rn_count + rn]
                col = ops.add(block_n, n_row(rn, lane_col))
                for slot in range(_ACCUMULATORS_PER_ATOM):
                    m_local = ops.add((slot // 8) * 16 + slot % 8, ops.mul(lane_half, 8))
                    row = ops.add(block_m, m_row(rm, m_local))
                    index = ops.v(ops.add(ops.mul(row, N), col))
                    value = b.fmul(alpha, b.vec_extract(acc, slot))
                    c_value = b.cvt_fp8_to_f32(b.global_load(C, index, FP8E4M3, align=1))
                    value = b.fadd(value, b.fmul(beta, c_value))
                    b.global_store(D, index, b.cvt_f32_to_fp8(value), align=1)

    b.ret()
    return b.kernel

# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Flag-free gfx1250 MX FP4 GEMM: ``D = alpha * A @ B.T + beta * C`` (RCR).

The Step 6.14 kernel of :mod:`.mxfp4_tdm_gemm` with its feature flags turned
into structure. Always on: double-buffered TDM staging split over wave
parity, scales staged by TDM, padded and segment-interleaved LDS, hoisted
descriptors, the tile k+2 refill inside tile k, the split cluster barrier,
coarse LDS waits, a loop unrolled to one tile per LDS buffer and the vector
epilogue. The tile, wave group, cluster and the scheduling choices below are
tuning parameters; LDS padding, multicast, convert width, WMMA order, refill
position and wait counts are derived from them.

Tile-local rows: rocKE A row ``m`` and rocKE B row ``n`` (an N index). The
WMMA's 32 M and 16 N positions map to tile rows through :class:`RowMaps`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...core.arch import ArchTarget
from ...core.ir import (
    BF16,
    F16,
    F32,
    FP8E4M3,
    I8,
    I32,
    I64,
    IRBuilder,
    KernelDef,
    PtrType,
    Value,
)
from ...helpers.compile import EXPERT_SCHEDULING_ATTR, SOURCE_ORDER_ATTR
from ...helpers.spec import SignatureBuilder, ceil_div_grid, kernel_name_join
from .mxfp4_tdm_gemm import IntOps, _IrOps, tdm_pad_fields

__all__ = [
    "IntOps",
    "LdsPad",
    "MxFp4GemmLdsLayout",
    "MxFp4GemmSpec",
    "build_mxfp4_gemm",
    "derived_lds_pads",
    "is_valid_spec",
    "mxfp4_gemm_grid",
    "mxfp4_gemm_lds_layout",
    "mxfp4_gemm_signature",
    "validate",
]

_ATOM_M = 32
_ATOM_N = 16
_ATOM_K = 128
_WAVE = 32
_WAVES = 4
_ACCUMULATORS_PER_ATOM = 16
_OP_ID = "wmma_scale_f32_32x16x128_f4"
_SCALE_GROUPS_PER_STEP = _ATOM_K // 32

TILE_MN = (128, 256)
TILE_K = (256, 512)
WAVE_GROUPS = ((2, 2), (4, 1), (1, 4))
OUT_DTYPES = ("fp8e4m3", "fp16", "bf16")
_OUT_TYPES = {"fp8e4m3": FP8E4M3, "fp16": F16, "bf16": BF16}
_OUT_BYTES = {"fp8e4m3": 1, "fp16": 2, "bf16": 2}
# The tile 6.14 was tuned at; the wave-0 cluster signal is its default.
_BENCH_TILE = (256, 256, 512)
_DEFAULT_SIGNAL_WMMAS = 6

# TensileLite (calcLdsPad / calcLdsBlockSizePerPad / get_mxs_mt_config) for a
# K-major FP4 operand on gfx1250: pad max(GlobalReadVectorWidth 2,
# LocalReadVectorWidth 32) FP4 elements = 16 B, one pass of the 64 banks.
_DATA_PAD_BYTES = 16
_BANK_CYCLE_BYTES = 256
_MXS_BLOCK_BYTES = 256
_MXS_PAD_BYTES = 16
_B_SCALE_REGION_ALIGN = 512
# Longest pad block the TDM pad_interval field encodes; A/B blocks above it
# use TDM iterate mode (one iteration per block).
_TDM_PAD_PERIOD_MAX = 1024
_TDM_PAD_MAX_BYTES = 128 * 4
_LDS_PAD_STEP_BYTES = 8
# s_wait_dscnt immediates are 6 bits on gfx1250.
_DSCNT_MAX = 63

_GL2_LINE_BYTES = 256
# th:TH_LOAD_NT (bits 0..2) | scope:SCOPE_SE (bits 3..4), as in the winner.
_GL2_PREFETCH_CPOL = 1 | (1 << 3)
_FP8_MAX = 448.0
# sched_barrier mask: VALU (0x2), SALU (0x4) and TRANS (0x400) may cross;
# WMMAs and memory operations may not.
_SCHED_ALU_ONLY = 0x2 | 0x4 | 0x400
# llvm.amdgcn.s.cluster.barrier as inline asm (see mxfp4_tdm_gemm), whole
# and split into its signal and wait halves.
_CLUSTER_BARRIER_ASM = (
    "s_barrier_signal_isfirst -1\n"
    "s_barrier_wait -1\n"
    "s_cbranch_scc0 1\n"
    "s_barrier_signal -3\n"
    "s_barrier_wait -3"
)
_CLUSTER_SIGNAL_ASM = (
    "s_barrier_signal_isfirst -1\n"
    "s_barrier_wait -1\n"
    "s_cbranch_scc0 1\n"
    "s_barrier_signal -3"
)
_CLUSTER_WAIT_ASM = "s_barrier_wait -3"
# Wave 0 alone signals, with no LDS drain. $0 is the wave index.
_CLUSTER_SIGNAL_WAVE0_ASM = (
    "s_cmp_eq_u32 $0, 0\n" "s_cbranch_scc0 1\n" "s_barrier_signal -3"
)
# $0 = this wave's SE/AID id word; bits 18:16 are the XCC.
_AID_ID_ASM = "s_sendmsg_rtn_b32 $0, sendmsg(MSG_RTN_GET_SE_AID_ID)\ns_wait_kmcnt 0"
_CLUSTERS_PER_AID_GROUP = 8
_XCCS_PER_AID = 4


class _WmmaScaleF4Atom:
    op_id = _OP_ID
    c_frag_len = _ACCUMULATORS_PER_ATOM
    c_dtype = "fp32"


def _is_pow2(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LdsPad:
    """``pad_bytes`` of LDS padding after every ``block_bytes`` written
    (TensileLite LdsPad / LdsBlockSizePerPad, in bytes)."""

    pad_bytes: int = 0
    block_bytes: int = 0

    @property
    def on(self) -> bool:
        return self.pad_bytes > 0


@dataclass(frozen=True)
class MxFp4GemmSpec:
    """Problem and tuning parameters. M, N and K are compile-time: the loop's
    tile-pair count comes from ``K``.

    ``*_lds_pad`` override the derived padding of one operand (``None``:
    derived, ``LdsPad()``: none). ``cluster_signal_wmmas`` and
    ``cluster_signal`` default (``None``) to 6, clamped to the tile, and to
    ``"wave0"`` at the 256x256x512 tile, ``"drained"`` otherwise.
    """

    name: str
    M: int
    N: int
    K: int
    out_dtype: str = "fp8e4m3"
    tile_m: int = 256
    tile_n: int = 256
    tile_k: int = 512
    wave_group: tuple[int, int] = (2, 2)
    cluster_dim: tuple[int, int, int] = (4, 4, 1)
    gl2_prefetch_distance: int = 4
    cluster_signal_wmmas: int | None = None
    cluster_signal: str | None = None
    a_lds_pad: LdsPad | None = None
    b_lds_pad: LdsPad | None = None
    a_scale_lds_pad: LdsPad | None = None
    b_scale_lds_pad: LdsPad | None = None
    init_c_peel: bool = True
    cluster_order: str = "hw"
    local_prefetch: int = 1
    half_plr: str = "b"
    block_k: int = field(default=32, init=False)

    @property
    def warp_m(self) -> int:
        return self.wave_group[0]

    @property
    def warp_n(self) -> int:
        return self.wave_group[1]

    @property
    def wave_repeat_m(self) -> int:
        return self.tile_m // (self.warp_m * _ATOM_M)

    @property
    def wave_repeat_n(self) -> int:
        return self.tile_n // (self.warp_n * _ATOM_N)

    @property
    def block_size(self) -> int:
        return _WAVE * _WAVES

    @property
    def k_steps(self) -> int:
        return self.tile_k // _ATOM_K

    @property
    def packed_k(self) -> int:
        return self.K // 2

    @property
    def has_cluster(self) -> bool:
        return tuple(self.cluster_dim[:2]) != (1, 1)

    @property
    def launch_cluster(self) -> tuple[int, int, int] | None:
        return tuple(self.cluster_dim) if self.has_cluster else None

    @property
    def half_b(self) -> bool:
        return self.local_prefetch == 1 and self.half_plr == "b"

    @property
    def signal_limit(self) -> int:
        per_step = self.wave_repeat_m * self.wave_repeat_n
        return per_step // 2 if self.half_b else per_step

    @property
    def signal_wmmas(self) -> int:
        if self.cluster_signal_wmmas is not None:
            return self.cluster_signal_wmmas
        return min(_DEFAULT_SIGNAL_WMMAS, self.signal_limit)

    @property
    def signal_form(self) -> str | None:
        """``"wave0"`` / ``"drained"``, or ``None`` without a cluster."""
        if not self.has_cluster:
            return None
        if self.cluster_signal is not None:
            return self.cluster_signal
        tile = (self.tile_m, self.tile_n, self.tile_k)
        return "wave0" if tile == _BENCH_TILE else "drained"

    @property
    def fp8_convert_width(self) -> int:
        return 8 if self.wave_repeat_n % 8 == 0 else 4

    @property
    def a_scale_reads(self) -> int:
        return max(1, self.wave_repeat_m // 4)

    @property
    def b_scale_reads(self) -> int:
        return self.wave_repeat_n // 4

    @property
    def reads_per_step(self) -> int:
        """LDS reads of one K-step: scales, four per A and two per B fragment."""
        return (
            self.a_scale_reads
            + self.b_scale_reads
            + 4 * self.wave_repeat_m
            + 2 * self.wave_repeat_n
        )

    @property
    def lds_layout(self) -> MxFp4GemmLdsLayout:
        return mxfp4_gemm_lds_layout(self)

    @property
    def lds_bytes(self) -> int:
        return 2 * self.lds_layout.buffer_bytes

    def kernel_name(self) -> str:
        cx, cy, _ = self.cluster_dim
        parts = [
            f"M{self.M}N{self.N}K{self.K}",
            f"t{self.tile_m}x{self.tile_n}x{self.tile_k}",
            f"c{cx}x{cy}",
        ]
        if self.out_dtype != "fp8e4m3":
            parts.append(self.out_dtype)
        if self.wave_group != (2, 2):
            parts.append(f"wg{self.warp_m}x{self.warp_n}")
        if self.gl2_prefetch_distance != 4:
            parts.append(f"pf{self.gl2_prefetch_distance}")
        if self.cluster_signal_wmmas is not None:
            parts.append(f"cs{self.cluster_signal_wmmas}")
        if self.cluster_signal is not None:
            parts.append(f"sig{self.cluster_signal}")
        for tag, pad in (
            ("pa", self.a_lds_pad),
            ("pb", self.b_lds_pad),
            ("psa", self.a_scale_lds_pad),
            ("psb", self.b_scale_lds_pad),
        ):
            if pad is not None:
                parts.append(f"{tag}{pad.pad_bytes}x{pad.block_bytes}")
        return kernel_name_join(
            self.name,
            "mxfp4",
            *parts,
            flags={
                "noinitc": not self.init_c_peel,
                "aid": self.cluster_order == "aid",
                "lp0": self.local_prefetch == 0,
                "hplrnone": self.half_plr == "none",
            },
        )


# ---------------------------------------------------------------------------
# LDS layout
# ---------------------------------------------------------------------------


def _tensile_data_block(k_bytes: int, vector_width: int) -> int:
    """calcLdsBlockSizePerPad for an UnrollMajorLDS operand: the K row,
    rounded to a bank cycle, times VectorWidth once that is longer."""
    block = _round_up(k_bytes, _BANK_CYCLE_BYTES)
    if k_bytes * vector_width > _BANK_CYCLE_BYTES:
        block = _round_up(k_bytes * vector_width, _BANK_CYCLE_BYTES)
    return block


def _tensile_scale_pad(vector_width: int) -> LdsPad:
    """get_mxs_mt_config(MatrixInstK=128, MXBlock=32, VectorWidth)."""
    per_block = (_ATOM_K // 32) * vector_width
    if vector_width < 4 or per_block % 16 or (per_block // 16) & 1:
        return LdsPad()
    return LdsPad(_MXS_PAD_BYTES, _MXS_BLOCK_BYTES)


def derived_lds_pads(spec: MxFp4GemmSpec) -> dict[str, LdsPad]:
    """TensileLite's padding with its A/B relabeled to rocKE B/A and
    VectorWidth taken as the operand's wave repeat, as in the benchmark
    solution (MIWaveTile [8, 4], VectorWidthA 8, VectorWidthB 4)."""
    k_bytes = spec.tile_k // 2
    rm, rn = spec.wave_repeat_m, spec.wave_repeat_n
    return {
        "a": LdsPad(_DATA_PAD_BYTES, _tensile_data_block(k_bytes, rm)),
        "b": LdsPad(_DATA_PAD_BYTES, _tensile_data_block(k_bytes, rn)),
        "a_scale": _tensile_scale_pad(rm),
        "b_scale": _tensile_scale_pad(rn),
    }


def resolved_lds_pads(spec: MxFp4GemmSpec) -> dict[str, LdsPad]:
    derived = derived_lds_pads(spec)
    overrides = {
        "a": spec.a_lds_pad,
        "b": spec.b_lds_pad,
        "a_scale": spec.a_scale_lds_pad,
        "b_scale": spec.b_scale_lds_pad,
    }
    return {k: derived[k] if overrides[k] is None else overrides[k] for k in derived}


@dataclass(frozen=True)
class _OperandLayout:
    rows: int
    base: int
    stride: int
    pad: LdsPad
    iterate_rows: int
    iterate_stride: int


@dataclass(frozen=True)
class MxFp4GemmLdsLayout:
    """One LDS buffer: ``[B0][A0 A1][B1][B scales][A scales]``.

    Each operand is two TDM components (one per wave pair) of ``rows`` rows;
    component ``c`` starts at ``base + c * stride``. Data pads above 1024 B
    are TDM iterate mode: ``iterate_rows`` rows per iteration, iterations
    ``iterate_stride`` apart. The offset methods take ints or IR values
    through ``ops`` so host tests and the builder share the address math.
    """

    row_bytes: int
    k_steps: int
    a: _OperandLayout
    b: _OperandLayout
    a_scale: _OperandLayout
    b_scale: _OperandLayout
    buffer_bytes: int

    def operand(self, name: str) -> _OperandLayout:
        return getattr(self, name)

    @staticmethod
    def component_base(ops, layout: _OperandLayout, component):
        return ops.add(layout.base, ops.mul(component, layout.stride))

    def row_offset(self, ops, name: str, row):
        """LDS byte offset of row ``row`` of operand ``"a"`` / ``"b"``."""
        op = self.operand(name)
        component = ops.div(row, op.rows)
        local = ops.mod(row, op.rows)
        if op.iterate_rows:
            linear = ops.add(
                ops.mul(ops.div(local, op.iterate_rows), op.iterate_stride),
                ops.mul(ops.mod(local, op.iterate_rows), self.row_bytes),
            )
        else:
            linear = ops.mul(local, self.row_bytes)
            if op.pad.on:
                linear = ops.add(
                    linear,
                    ops.mul(ops.div(linear, op.pad.block_bytes), op.pad.pad_bytes),
                )
        return ops.add(self.component_base(ops, op, component), linear)

    def scale_offset(self, ops, name: str, step, row):
        """LDS byte offset of the four E8M0 scales of row ``row`` at K-step
        ``step`` (``name`` is ``"a_scale"`` / ``"b_scale"``)."""
        op = self.operand(name)
        linear = ops.add(ops.mul(step, op.rows * 4), ops.mul(ops.mod(row, op.rows), 4))
        if op.pad.on:
            linear = ops.add(
                linear, ops.mul(ops.div(linear, op.pad.block_bytes), op.pad.pad_bytes)
            )
        return ops.add(self.component_base(ops, op, ops.div(row, op.rows)), linear)


def _data_component(rows: int, row_bytes: int, pad: LdsPad) -> tuple[int, int, int]:
    """(component bytes, iterate rows, iterate stride) of one data component."""
    if pad.on and pad.block_bytes > _TDM_PAD_PERIOD_MAX:
        iterate_rows = pad.block_bytes // row_bytes
        stride = pad.block_bytes + pad.pad_bytes
        return -(-rows // iterate_rows) * stride, iterate_rows, stride
    size = rows * row_bytes
    if pad.on:
        size += size // pad.block_bytes * pad.pad_bytes
    return size, 0, 0


def mxfp4_gemm_lds_layout(spec: MxFp4GemmSpec) -> MxFp4GemmLdsLayout:
    pads = resolved_lds_pads(spec)
    row_bytes = spec.tile_k // 2
    steps = spec.k_steps
    a_rows, b_rows = spec.tile_m // 2, spec.tile_n // 2
    a_comp, a_it, a_it_stride = _data_component(a_rows, row_bytes, pads["a"])
    b_comp, b_it, b_it_stride = _data_component(b_rows, row_bytes, pads["b"])

    def scale_component(rows: int, pad: LdsPad) -> int:
        size = steps * rows * 4
        return size + (size // pad.block_bytes * pad.pad_bytes if pad.on else 0)

    a_scale_comp = scale_component(a_rows, pads["a_scale"])
    b_scale_comp = scale_component(b_rows, pads["b_scale"])
    a_total = 2 * a_comp
    b_base, a_base = 0, b_comp
    b_stride = b_comp + a_total
    data_end = a_total + 2 * b_comp
    b_scale_region = 2 * b_scale_comp
    if pads["b_scale"].on:
        b_scale_region = _round_up(b_scale_region, _B_SCALE_REGION_ALIGN)
    b_scale_base = data_end
    a_scale_base = b_scale_base + b_scale_region
    return MxFp4GemmLdsLayout(
        row_bytes=row_bytes,
        k_steps=steps,
        a=_OperandLayout(a_rows, a_base, a_comp, pads["a"], a_it, a_it_stride),
        b=_OperandLayout(b_rows, b_base, b_stride, pads["b"], b_it, b_it_stride),
        a_scale=_OperandLayout(
            a_rows, a_scale_base, a_scale_comp, pads["a_scale"], 0, 0
        ),
        b_scale=_OperandLayout(
            b_rows, b_scale_base, b_scale_comp, pads["b_scale"], 0, 0
        ),
        buffer_bytes=a_scale_base + 2 * a_scale_comp,
    )


# ---------------------------------------------------------------------------
# Validation, signature, grid
# ---------------------------------------------------------------------------


def _pad_reason(
    name: str, pad: LdsPad, *, row_bytes: int, rows: int, chunk: int
) -> str | None:
    if pad.pad_bytes < 0 or pad.block_bytes < 0:
        return f"{name} LDS pad must be non-negative"
    if not pad.on:
        return None
    if pad.pad_bytes % _LDS_PAD_STEP_BYTES or pad.pad_bytes > _TDM_PAD_MAX_BYTES:
        return f"{name} LDS pad must be a multiple of 8 B up to {_TDM_PAD_MAX_BYTES} B"
    block = pad.block_bytes
    if block <= _TDM_PAD_PERIOD_MAX:
        if block < 8 or not _is_pow2(block):
            return f"{name} LDS pad block must be a power of two of 8..1024 B"
    elif row_bytes == 0:
        return f"{name} LDS pad block must be at most {_TDM_PAD_PERIOD_MAX} B"
    elif block % row_bytes or rows % (block // row_bytes):
        return f"{name} LDS pad block above 1024 B must be whole rows dividing the component"
    if block % chunk:
        return f"{name} LDS pad block must be a multiple of {chunk} B (one lane's read)"
    return None


def validate(spec: MxFp4GemmSpec, arch: str = "gfx1250") -> str | None:
    """``None`` if ``spec`` builds, else the reason it does not."""
    if arch != "gfx1250":
        return f"mxfp4_gemm is gfx1250-only (got {arch!r})"
    if min(spec.M, spec.N, spec.K) <= 0:
        return "M, N and K must be positive"
    if spec.out_dtype not in OUT_DTYPES:
        return f"out_dtype must be one of {OUT_DTYPES}"
    if spec.tile_m not in TILE_MN or spec.tile_n not in TILE_MN:
        return f"tile_m and tile_n must be in {TILE_MN}"
    if spec.tile_k not in TILE_K:
        return f"tile_k must be in {TILE_K}"
    if tuple(spec.wave_group) not in WAVE_GROUPS:
        return f"wave_group must be one of {WAVE_GROUPS} (4 waves)"
    if spec.tile_m % (spec.warp_m * _ATOM_M) or spec.tile_n % (spec.warp_n * _ATOM_N):
        return "the tile must hold whole 32x16 WMMA atoms per wave"
    rm, rn = spec.wave_repeat_m, spec.wave_repeat_n
    if rn < 4:
        return "wave_repeat_n must be at least 4 (vector epilogue, B scale reads)"
    if spec.M % spec.tile_m or spec.N % spec.tile_n or spec.K % spec.tile_k:
        return "M, N and K must divide their tile"
    if spec.K < 2 * spec.tile_k:
        return "K must hold at least two tiles (double buffer, init-C peel)"
    if spec.local_prefetch not in (0, 1):
        return "local_prefetch must be 0 or 1"
    if spec.half_plr not in ("b", "none"):
        return "half_plr must be 'b' or 'none'"
    if spec.half_plr == "b" and (spec.local_prefetch != 1 or rn % 2):
        return "half_plr 'b' needs local_prefetch 1 and an even wave_repeat_n"
    if spec.cluster_order not in ("hw", "aid"):
        return "cluster_order must be 'hw' or 'aid'"
    if spec.cluster_signal not in (None, "wave0", "drained"):
        return "cluster_signal must be 'wave0' or 'drained'"
    if not 0 <= spec.signal_wmmas <= spec.signal_limit:
        return f"cluster_signal_wmmas must be in 0..{spec.signal_limit}"
    if spec.gl2_prefetch_distance != 0 and not 2 <= spec.gl2_prefetch_distance <= 8:
        return "gl2_prefetch_distance must be 0 (off) or 2..8 tiles"
    cluster = tuple(spec.cluster_dim)
    if len(cluster) != 3 or any(d < 1 for d in cluster):
        return "cluster_dim must be a positive (x, y, z) triple"
    cx, cy, cz = cluster
    if cz != 1 or cx * cy > 16:
        return "cluster_dim must be 2D with at most 16 workgroups"
    gx, gy, _ = mxfp4_gemm_grid(spec)
    if gx % cx or gy % cy:
        return f"grid {gx}x{gy} must be divisible by cluster_dim {cx}x{cy}"
    if (
        spec.has_cluster
        and not ArchTarget.from_gfx("gfx1250").memory.has_cluster_launch
    ):
        return "gfx1250 reports no cluster launch support"
    pads = resolved_lds_pads(spec)
    row_bytes = spec.tile_k // 2
    for name, rows, chunk, data in (
        ("a", spec.tile_m // 2, 16, True),
        ("b", spec.tile_n // 2, 16, True),
        ("a_scale", spec.tile_m // 2, min(16, 4 * rm), False),
        ("b_scale", spec.tile_n // 2, 16, False),
    ):
        pad = pads[name]
        reason = _pad_reason(
            name, pad, row_bytes=row_bytes if data else 0, rows=rows, chunk=chunk
        )
        if reason:
            return reason
        if (
            data
            and pad.on
            and pad.block_bytes <= _TDM_PAD_PERIOD_MAX
            and pad.block_bytes % row_bytes
        ):
            return f"{name} LDS pad block must be whole {row_bytes} B rows"
    if spec.reads_per_step > _DSCNT_MAX:
        return (
            f"{spec.reads_per_step} LDS reads per K-step exceed the s_wait_dscnt range"
        )
    target = ArchTarget.from_gfx("gfx1250")
    accumulators = rm * rn * _ACCUMULATORS_PER_ATOM
    if accumulators > target.limits.vgprs:
        return (
            f"{accumulators} accumulators per lane exceed {target.limits.vgprs} VGPRs"
        )
    if spec.lds_bytes > target.lds_capacity_bytes:
        return f"LDS {spec.lds_bytes} B exceeds {target.lds_capacity_bytes} B"
    return None


def is_valid_spec(spec: MxFp4GemmSpec, arch: str = "gfx1250") -> tuple[bool, str]:
    reason = validate(spec, arch)
    return (False, reason) if reason else (True, "ok")


def mxfp4_gemm_signature(spec: MxFp4GemmSpec) -> list[dict]:
    return (
        SignatureBuilder()
        .ptr("A", "i8")
        .ptr("B", "i8")
        .ptr("A_scale", "i8")
        .ptr("B_scale", "i8")
        .ptr("C", spec.out_dtype)
        .ptr("D", spec.out_dtype)
        .scalar("M", "i32")
        .scalar("N", "i32")
        .scalar("K", "i32")
        .scalar("alpha", "f32")
        .scalar("beta", "f32")
        .build()
    )


def mxfp4_gemm_grid(spec: MxFp4GemmSpec) -> tuple[int, int, int]:
    return ceil_div_grid((spec.N, spec.tile_n), (spec.M, spec.tile_m))


# ---------------------------------------------------------------------------
# Kernel roles
# ---------------------------------------------------------------------------


@dataclass
class _Kernel:
    """Values every role reads: the builder, spec, layout and the kernel's
    parameters and per-lane indices."""

    b: IRBuilder
    ops: _IrOps
    spec: MxFp4GemmSpec
    layout: MxFp4GemmLdsLayout
    params: dict
    c0: Value = None
    c1: Value = None
    tile_k_v: Value = None
    packed_k: Value = None
    tid: Value = None
    lane: Value = None
    lane_col: Value = None
    lane_half: Value = None
    wave: Value = None
    wave_m: Value = None
    wave_n: Value = None
    block_m: Value = None
    block_n: Value = None
    cluster_wx: Value = None
    cluster_wy: Value = None
    shared: Value = None
    shared_addr: Value = None

    def buffer_base(self, parity):
        return self.ops.mul(parity, self.layout.buffer_bytes)

    def fence(self) -> None:
        # Keep LLVM from moving LDS reads or WMMAs across the refill; ALU
        # work (descriptor and address math) may cross.
        self.b.inline_asm("", "~{memory}")
        self.b.sched_barrier(_SCHED_ALU_ONLY)

    def cluster_sync(self) -> None:
        if self.spec.has_cluster:
            self.b.inline_asm(_CLUSTER_BARRIER_ASM, "~{scc},~{memory}")
        else:
            self.b.s_barrier_bare()

    def clamp_k(self, k):
        b = self.b
        return b.select(
            b.cmp_lt(k, self.params["K"]), k, b.sub(self.params["K"], self.tile_k_v)
        )


def _aid_remapped_blocks(k: _Kernel, block_x, block_y):
    """ClusterAIDMapping: renumber clusters so the clusters on one AID's XCCs
    cover one contiguous half of the grid. Grids that are not whole clusters
    in groups of eight keep the hardware order."""
    b, ops, spec = k.b, k.ops, k.spec
    gx, gy, _ = mxfp4_gemm_grid(spec)
    cx, cy, _ = spec.cluster_dim
    clusters_x, clusters_y = gx // cx, gy // cy
    if clusters_x * clusters_y % _CLUSTERS_PER_AID_GROUP:
        return block_x, block_y
    cluster = ops.add(ops.div(block_x, cx), ops.mul(ops.div(block_y, cy), clusters_x))
    aid_word = b.inline_asm(_AID_ID_ASM, "=s", result_type=I32)
    xcc = ops.mod(ops.div(aid_word, 1 << 16), _CLUSTERS_PER_AID_GROUP)
    slot = ops.add(
        ops.add(
            ops.mul(ops.div(xcc, _XCCS_PER_AID), clusters_x * clusters_y // 2),
            ops.mul(ops.div(cluster, _CLUSTERS_PER_AID_GROUP), _XCCS_PER_AID),
        ),
        ops.mod(xcc, _XCCS_PER_AID),
    )
    split_rows = clusters_y % 2 == 0 and (
        clusters_x % 2 == 1 or gx * spec.tile_n <= gy * spec.tile_m
    )
    if split_rows:
        new_x, new_y = ops.mod(slot, clusters_x), ops.div(slot, clusters_x)
    else:
        new_x, new_y = ops.div(slot, clusters_y), ops.mod(slot, clusters_y)
    return (
        ops.add(ops.mul(new_x, cx), ops.mod(block_x, cx)),
        ops.add(ops.mul(new_y, cy), ops.mod(block_y, cy)),
    )


class RowMaps:
    """WMMA positions to tile rows. A lane's M-repeats are adjacent A rows,
    so its A scales of a K-step are contiguous; its N-repeats are adjacent
    D columns, so one accumulator slot is one vector store."""

    def __init__(self, k: _Kernel):
        self.k = k

    def m_row(self, rm, m_local):
        ops, rm_count = self.k.ops, self.k.spec.wave_repeat_m
        base = ops.mul(self.k.wave_m, rm_count * _ATOM_M)
        return ops.add(base, ops.add(ops.mul(m_local, rm_count), rm))

    def n_row(self, rn, n_local):
        ops, rn_count = self.k.ops, self.k.spec.wave_repeat_n
        base = ops.mul(self.k.wave_n, rn_count * _ATOM_N)
        return ops.add(base, ops.add(ops.mul(n_local, rn_count), rn))


class TdmLoader:
    """Tile loads: even waves copy one rocKE B component, odd waves one A
    component (data, then scales); descriptors are built once and each tile
    patches only its LDS address, global address and remaining sizes."""

    def __init__(self, k: _Kernel):
        self.k = k
        b, ops, spec = k.b, k.ops, k.spec
        self.a_mask: int | Value = 0
        self.b_mask: int | Value = 0
        if spec.has_cluster:
            cluster_x, cluster_y, _ = spec.cluster_dim
            # Grid x walks N tiles and y walks M tiles: workgroups with the
            # same cluster x share rocKE B, those with the same y share A.
            col_bits = sum(1 << (row * cluster_x) for row in range(cluster_y))
            row_bits = (1 << cluster_x) - 1
            self.b_mask = b.shl(b.const_i32(col_bits), k.cluster_wx)
            self.a_mask = b.shl(b.const_i32(row_bits), ops.mul(k.cluster_wy, cluster_x))

    def lds_ptr(self, offset):
        k = self.k
        return k.b.smem_ptr_add(
            k.shared_addr, k.b.zext(k.ops.v(offset), k.shared_addr.type)
        )

    def _operand(self, operand: str):
        k, p = self.k, self.k.params
        if operand == "a":
            return k.layout.a, k.block_m, p["M"], self.a_mask
        return k.layout.b, k.block_n, p["N"], self.b_mask

    def data_descriptor(self, operand: str, parity, component, k_byte, remaining_k):
        k = self.k
        b, ops, p = k.b, k.ops, k.params
        layout, block, extent, mask = self._operand(operand)
        base_ptr = p["A"] if operand == "a" else p["B"]
        lds = ops.add(
            k.buffer_base(parity), k.layout.component_base(ops, layout, component)
        )
        rows = layout.rows
        row0 = ops.add(block, ops.mul(component, rows))
        element = ops.add(ops.mul(row0, k.packed_k), k_byte)
        extra: dict = {}
        tile_rows = rows
        if layout.iterate_rows:
            tile_rows = layout.iterate_rows
            extra.update(
                iterate_count=rows // tile_rows,
                lds_increment=layout.iterate_stride,
                global_increment=ops.v(ops.mul(k.packed_k, tile_rows)),
            )
        elif layout.pad.on:
            extra.update(tdm_pad_fields(layout.pad.block_bytes, layout.pad.pad_bytes))
        return b.tdm_descriptor_2d(
            b.global_ptr_add(base_ptr, ops.v(element)),
            self.lds_ptr(lds),
            elem_bytes=1,
            tensor_dim0=remaining_k,
            tensor_dim1=b.sub(extent, ops.v(row0)),
            row_stride=k.packed_k,
            tile_dim0=k.layout.row_bytes,
            tile_dim1=tile_rows,
            workgroup_mask=mask,
            **extra,
        )

    def scale_descriptor(self, operand: str, parity, component, k_step):
        """Swizzled scales ``[K/128][MN][4]``: one row per K-step, 4 B per MN."""
        k = self.k
        b, ops, p = k.b, k.ops, k.params
        _, block, extent, mask = self._operand(operand)
        layout = k.layout.a_scale if operand == "a" else k.layout.b_scale
        base_ptr = p["A_scale"] if operand == "a" else p["B_scale"]
        lds = ops.add(
            k.buffer_base(parity), k.layout.component_base(ops, layout, component)
        )
        extra = (
            tdm_pad_fields(layout.pad.block_bytes, layout.pad.pad_bytes)
            if layout.pad.on
            else {}
        )
        rows = layout.rows
        row_stride = ops.mul(extent, 4)
        col0 = ops.mul(ops.add(block, ops.mul(component, rows)), 4)
        element = ops.add(ops.mul(k_step, row_stride), col0)
        return b.tdm_descriptor_2d(
            b.global_ptr_add(base_ptr, ops.v(element)),
            self.lds_ptr(lds),
            elem_bytes=1,
            tensor_dim0=b.sub(ops.v(row_stride), ops.v(col0)),
            tensor_dim1=b.sub(ops.div(p["K"], _ATOM_K), ops.v(k_step)),
            row_stride=ops.v(row_stride),
            tile_dim0=rows * 4,
            tile_dim1=k.layout.k_steps,
            workgroup_mask=mask,
            **extra,
        )

    def hoist(self) -> None:
        """Both descriptors for tile 0, buffer 0, selected by wave parity."""
        k = self.k
        b, ops, c0, p = k.b, k.ops, k.c0, k.params
        even0 = b.cmp_eq(ops.mod(k.wave, 2), c0)
        component0 = ops.div(k.wave, 2)

        def selected(b_desc, a_desc):
            return [b.select(even0, x, y) for x, y in zip(b_desc, a_desc)]

        self.data = selected(
            self.data_descriptor("b", c0, component0, c0, b.sub(k.packed_k, c0)),
            self.data_descriptor("a", c0, component0, c0, b.sub(k.packed_k, c0)),
        )
        self.scale = selected(
            self.scale_descriptor("b", c0, component0, c0),
            self.scale_descriptor("a", c0, component0, c0),
        )
        self.scale_stride = b.select(
            even0, ops.v(ops.mul(p["N"], 4)), ops.v(ops.mul(p["M"], 4))
        )

    def _patched(self, desc, parity, byte_offset, words):
        """``desc`` with its LDS address moved to buffer ``parity``, its
        global address advanced by ``byte_offset`` and the group-1 words in
        ``words`` (index -> value) replaced."""
        k = self.k
        b, ops = k.b, k.ops
        d0, d1, *rest = desc
        lds = b.add(b.vec_extract(d0, 1), ops.v(k.buffer_base(parity)))
        lo, hi = b.vec_extract(d0, 2), b.vec_extract(d0, 3)
        address = b.lor(
            b.shl(b.zext(b.land(hi, b.const_i32(0x1FFFFFF)), I64), b.const_i64(32)),
            b.zext(lo, I64),
        )
        address = b.add(address, b.zext(ops.v(byte_offset), I64))
        new_hi = b.lor(
            b.land(
                b.trunc(b.lshr(address, b.const_i64(32)), I32), b.const_i32(0x1FFFFFF)
            ),
            b.land(hi, b.const_i32(-0x2000000)),
        )
        d0 = b.vec_pack([b.vec_extract(d0, 0), lds, b.trunc(address, I32), new_hi], I32)
        d1 = b.vec_pack([words.get(i, b.vec_extract(d1, i)) for i in range(8)], I32)
        return (d0, d1, *rest)

    def issue(self, k0, parity) -> None:
        """This wave's data and scale loads of the tile at ``k0`` into buffer ``parity``."""
        k = self.k
        b, ops = k.b, k.ops
        k_byte = ops.div(k0, 2)
        remaining_k = b.sub(k.packed_k, ops.v(k_byte))
        c16 = b.const_i32(16)
        data = self.data
        w2 = b.vec_extract(data[1], 2)
        b.tensor_load_to_lds(
            *self._patched(
                data,
                parity,
                k_byte,
                {
                    1: b.shl(remaining_k, c16),
                    2: b.lor(
                        b.land(w2, b.const_i32(-0x10000)), b.lshr(remaining_k, c16)
                    ),
                },
            )
        )
        k_step = ops.v(ops.div(k0, _ATOM_K))
        scale = self.scale
        steps_left = b.sub(ops.v(ops.div(k.params["K"], _ATOM_K)), k_step)
        s2, s3 = b.vec_extract(scale[1], 2), b.vec_extract(scale[1], 3)
        b.tensor_load_to_lds(
            *self._patched(
                scale,
                parity,
                b.mul(k_step, self.scale_stride),
                {
                    2: b.lor(b.land(s2, b.const_i32(0xFFFF)), b.shl(steps_left, c16)),
                    3: b.lor(
                        b.land(s3, b.const_i32(-0x10000)), b.lshr(steps_left, c16)
                    ),
                },
            )
        )


class Gl2Prefetcher:
    """PrefetchGL2: the cluster's A/B and scale lines of a tile ahead, split
    over every thread of the cluster, one 256 B line each."""

    def __init__(self, k: _Kernel):
        self.k = k

    def prefetch(self, k_ahead, ahead_parity: int) -> None:
        """``ahead_parity`` is the parity of the prefetched tile's index: with
        128 B rows one line holds an even tile and the next one, so the data
        lines are only prefetched for even tiles."""
        k = self.k
        b, ops, spec, p = k.b, k.ops, k.spec, k.params
        if not spec.gl2_prefetch_distance:
            return
        row_bytes = k.layout.row_bytes
        cluster_x, cluster_y, _ = spec.cluster_dim
        # Branch-free: any branch in the loop body makes LLVM drain the LDS
        # read pipeline. Past the end, the last tile is prefetched again.
        k_ahead = b.select(
            b.cmp_lt(k_ahead, p["K"]), k_ahead, b.sub(p["K"], k.tile_k_v)
        )
        k_byte = ops.div(k_ahead, 2)
        data_lines = row_bytes // _GL2_LINE_BYTES
        if not data_lines and ahead_parity == 0:
            data_lines = 1
        if spec.has_cluster:
            cluster_m0 = b.sub(k.block_m, ops.v(ops.mul(k.cluster_wy, spec.tile_m)))
            cluster_n0 = b.sub(k.block_n, ops.v(ops.mul(k.cluster_wx, spec.tile_n)))
        else:
            cluster_m0, cluster_n0 = k.block_m, k.block_n

        def data_line(base_ptr, row0):
            def address(index):
                row = ops.add(row0, ops.div(index, data_lines))
                offset = ops.add(
                    k_byte, ops.mul(ops.mod(index, data_lines), _GL2_LINE_BYTES)
                )
                return b.global_ptr_add(
                    base_ptr, ops.v(ops.add(ops.mul(row, k.packed_k), offset))
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
                    base_ptr, ops.v(ops.add(ops.mul(row, ops.mul(extent, 4)), offset))
                )

            return address, k.layout.k_steps * lines_per_row

        categories = []
        if data_lines:
            categories += [
                (cluster_y * spec.tile_m * data_lines, data_line(p["A"], cluster_m0)),
                (cluster_x * spec.tile_n * data_lines, data_line(p["B"], cluster_n0)),
            ]
        if (cluster_x * spec.tile_n * 4) % _GL2_LINE_BYTES == 0:
            b_scale_address, b_scale_lines = scale_line(
                p["B_scale"], p["N"], cluster_n0, cluster_x * spec.tile_n
            )
            a_scale_address, a_scale_lines = scale_line(
                p["A_scale"], p["M"], cluster_m0, cluster_y * spec.tile_m
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
        if spec.has_cluster:
            flat = ops.add(k.cluster_wx, ops.mul(k.cluster_wy, cluster_x))
            thread = ops.add(ops.mul(flat, spec.block_size), k.tid)
        else:
            thread = k.tid

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
                # Every lane covers one of the remaining lines; some lines
                # are prefetched more than once.
                index = ops.add(ops.mod(thread, remaining), round_start)
            prefetch_round(round_start, index)


class LdsReader:
    """LDS to registers: 16 B reads (scales of one K-step, A fragments as
    two rows of two reads, B fragments as two reads)."""

    def __init__(self, k: _Kernel, rows: RowMaps):
        self.k = k
        self.rows = rows

    def load_i32(self, offset, n: int = 4):
        return self.k.b.smem_load_vN(
            self.k.shared, self.k.ops.v(offset), dtype=I32, n=n
        )

    def a_fragment(self, buffer, rm, step):
        ops = self.k.ops
        lane_k = ops.add(step * (_ATOM_K // 2), ops.mul(self.k.lane_half, 16))
        frags = []
        for m_local in (self.k.lane_col, ops.add(self.k.lane_col, 16)):
            base = ops.add(
                ops.add(
                    buffer,
                    self.k.layout.row_offset(ops, "a", self.rows.m_row(rm, m_local)),
                ),
                lane_k,
            )
            frags.append(
                self.k.b.vec_concat(
                    self.load_i32(base), self.load_i32(ops.add(base, 32))
                )
            )
        return self.k.b.vec_concat(frags[0], frags[1])

    def b_fragment(self, buffer, rn, step):
        ops = self.k.ops
        lane_k = ops.add(step * (_ATOM_K // 2), ops.mul(self.k.lane_half, 16))
        base = ops.add(
            ops.add(
                buffer,
                self.k.layout.row_offset(
                    ops, "b", self.rows.n_row(rn, self.k.lane_col)
                ),
            ),
            lane_k,
        )
        return self.k.b.vec_concat(
            self.load_i32(base), self.load_i32(ops.add(base, 32))
        )

    def scales(self, buffer, step):
        """Per-lane E8M0 words of one K-step: one per M- and per N-repeat."""
        k = self.k
        b, ops, spec = k.b, k.ops, k.spec
        rm_count, rn_count = spec.wave_repeat_m, spec.wave_repeat_n
        a_base = ops.add(
            buffer,
            k.layout.scale_offset(ops, "a_scale", step, self.rows.m_row(0, k.lane)),
        )
        a_scales = []
        if rm_count >= 4:
            for chunk in range(rm_count // 4):
                words = self.load_i32(ops.add(a_base, chunk * 16))
                a_scales += [b.vec_extract(words, i) for i in range(4)]
        else:
            words = self.load_i32(a_base, n=rm_count)
            a_scales = [b.vec_extract(words, i) for i in range(rm_count)]
        b_base = ops.add(
            buffer,
            k.layout.scale_offset(ops, "b_scale", step, self.rows.n_row(0, k.lane_col)),
        )
        b_scales = []
        for chunk in range(rn_count // 4):
            words = self.load_i32(ops.add(b_base, chunk * 16))
            b_scales += [b.vec_extract(words, i) for i in range(4)]
        return a_scales, b_scales

    def head(self, buffer, step):
        a_scales, b_scales = self.scales(buffer, step)
        a_frags = [
            self.a_fragment(buffer, rm, step) for rm in range(self.k.spec.wave_repeat_m)
        ]
        return a_frags, a_scales, b_scales

    def b_group(self, buffer, step, first: int, count: int):
        return [self.b_fragment(buffer, rn, step) for rn in range(first, first + count)]


@dataclass
class _Hooks:
    """Refill events inside one tile (see :class:`Pipeline`)."""

    refill: object = None
    signal: object = None
    late_refill: object = None


class TileCompute:
    """One K tile of WMMAs with its LDS reads and refill events.

    ``local_prefetch`` 1: K-step s+1 is read during K-step s; the last K-step
    reads the next tile's K-step 0 from the other buffer (``head``). With
    ``half_plr`` ``"b"`` B is read in halves, the second after the first
    half's WMMAs: three B half-sets live, two waits per K-step, B-outer
    WMMAs. Each wait leaves exactly one K-step of reads in flight.
    """

    def __init__(self, k: _Kernel, reader: LdsReader):
        self.k = k
        self.reader = reader

    def _mma_fn(self, out, seed_zero):
        k = self.k
        rn_count = k.spec.wave_repeat_n
        zero = k.b.zero_vec_f32(_ACCUMULATORS_PER_ATOM) if seed_zero else None

        def mma(rm, rn, a_frag, b_frag, a_scale, b_scale, first_step):
            index = rm * rn_count + rn
            c_in = zero if (seed_zero and first_step) else out[index]
            out[index] = k.b.mma(
                _WmmaScaleF4Atom(), a_frag, b_frag, c_in, a_scale, b_scale
            )

        return mma

    def _wait(self, count: int) -> None:
        self.k.b.inline_asm(f"s_wait_dscnt {count}", "~{memory}")

    def head_parts(self) -> list[int]:
        spec = self.k.spec
        rm, rn = spec.wave_repeat_m, spec.wave_repeat_n
        if spec.half_b:
            return [rm, rm, rn, rn // 2, rn // 2]
        return [rm, rm, rn, rn]

    def first_head(self, parity):
        """Tile 0's K-step 0 reads, issued before the first tile."""
        k = self.k
        a_frags, a_scales, b_scales = self.reader.head(k.buffer_base(parity), 0)
        if k.spec.half_b:
            half = k.spec.wave_repeat_n // 2
            groups = [
                self.reader.b_group(k.buffer_base(parity), 0, w * half, half)
                for w in (0, 1)
            ]
        else:
            groups = [
                self.reader.b_group(k.buffer_base(parity), 0, 0, k.spec.wave_repeat_n)
            ]
        return a_frags, a_scales, b_scales, groups

    @staticmethod
    def flatten(head) -> list:
        a_frags, a_scales, b_scales, groups = head
        return [*a_frags, *a_scales, *b_scales, *(v for g in groups for v in g)]

    def unflatten(self, values):
        values = list(values)
        parts = []
        for count in self.head_parts():
            parts.append(values[:count])
            values = values[count:]
        return parts[0], parts[1], parts[2], parts[3:]

    def run(
        self,
        parity,
        accumulators,
        *,
        hooks: _Hooks,
        head=None,
        next_parity=None,
        seed_zero=False,
    ):
        """Returns ``(accumulators, head)``; ``head`` is the next tile's
        K-step 0 reads (``None`` with ``local_prefetch`` 0)."""
        if self.k.spec.local_prefetch == 0:
            return self._run_direct(parity, accumulators, seed_zero, hooks)
        if self.k.spec.half_b:
            return self._run_half_b(
                parity, accumulators, head, next_parity, seed_zero, hooks
            )
        return self._run_full(parity, accumulators, head, next_parity, seed_zero, hooks)

    def _run_half_b(self, parity, accumulators, head, next_parity, seed_zero, hooks):
        k, reader = self.k, self.reader
        spec = k.spec
        rm_count, rn_count, steps = spec.wave_repeat_m, spec.wave_repeat_n, spec.k_steps
        half = rn_count // 2
        buffer = k.buffer_base(parity)
        out = list(accumulators)
        mma = self._mma_fn(out, seed_zero)
        *head, b_halves = head
        next_buffer = k.buffer_base(next_parity)
        for step in range(steps):
            a_frags, a_scales, b_scales = head
            last = step + 1 == steps
            if not last:
                head = reader.head(buffer, step + 1)
                next_first = reader.b_group(buffer, step + 1, 0, half)
            else:
                hooks.refill()
                head = reader.head(next_buffer, 0)
                next_first = reader.b_group(next_buffer, 0, 0, half)
            for which in (0, 1):
                if which == 1 and not last:
                    next_second = reader.b_group(buffer, step + 1, half, half)
                elif which == 1:
                    next_second = reader.b_group(next_buffer, 0, half, half)
                if which == 1 and last:
                    hooks.late_refill()
                signal_here = (
                    hooks.signal is not None and which == 1 and step + 2 == steps
                )
                self._wait(spec.reads_per_step)
                order = [(rm, j) for j in range(half) for rm in range(rm_count)]
                issued = 0
                for rm, j in order:
                    rn = which * half + j
                    if signal_here and issued == spec.signal_wmmas:
                        hooks.signal()
                    issued += 1
                    mma(
                        rm,
                        rn,
                        a_frags[rm],
                        b_halves[which][j],
                        a_scales[rm],
                        b_scales[rn],
                        step == 0,
                    )
                if signal_here and issued == spec.signal_wmmas:
                    hooks.signal()
            if not last:
                b_halves = [next_first, next_second]
        return out, (*head, [next_first, next_second])

    def _run_full(self, parity, accumulators, head, next_parity, seed_zero, hooks):
        k, reader = self.k, self.reader
        spec = k.spec
        rm_count, rn_count, steps = spec.wave_repeat_m, spec.wave_repeat_n, spec.k_steps
        buffer = k.buffer_base(parity)
        out = list(accumulators)
        mma = self._mma_fn(out, seed_zero)
        a_frags, a_scales, b_scales, (b_frags,) = head
        next_buffer = k.buffer_base(next_parity)
        for step in range(steps):
            last = step + 1 == steps
            if not last:
                ahead = (
                    *reader.head(buffer, step + 1),
                    reader.b_group(buffer, step + 1, 0, rn_count),
                )
            else:
                hooks.refill()
                ahead = (
                    *reader.head(next_buffer, 0),
                    reader.b_group(next_buffer, 0, 0, rn_count),
                )
                hooks.late_refill()
            signal_here = hooks.signal is not None and step + 2 == steps
            self._wait(spec.reads_per_step)
            issued = 0
            for rm in range(rm_count):
                for rn in range(rn_count):
                    if signal_here and issued == spec.signal_wmmas:
                        hooks.signal()
                    issued += 1
                    mma(
                        rm,
                        rn,
                        a_frags[rm],
                        b_frags[rn],
                        a_scales[rm],
                        b_scales[rn],
                        step == 0,
                    )
            if signal_here and issued == spec.signal_wmmas:
                hooks.signal()
            a_frags, a_scales, b_scales, b_frags = ahead
        return out, (a_frags, a_scales, b_scales, [b_frags])

    def _run_direct(self, parity, accumulators, seed_zero, hooks):
        """local_prefetch 0: each K-step reads, drains and computes. The
        tile's last read is the last K-step's: its refill (GL2 prefetch, the
        signal) follows the drain, the cluster wait and TDM come
        ``cluster_signal_wmmas`` WMMAs later (at once without a cluster)."""
        k, reader = self.k, self.reader
        spec = k.spec
        rm_count, rn_count, steps = spec.wave_repeat_m, spec.wave_repeat_n, spec.k_steps
        buffer = k.buffer_base(parity)
        out = list(accumulators)
        mma = self._mma_fn(out, seed_zero)
        late_at = spec.signal_wmmas if spec.has_cluster else 0
        for step in range(steps):
            last = step + 1 == steps
            a_frags, a_scales, b_scales = reader.head(buffer, step)
            b_frags = reader.b_group(buffer, step, 0, rn_count)
            self._wait(0)
            if last:
                hooks.refill()
                if hooks.signal is not None:
                    hooks.signal()
            issued = 0
            for rm in range(rm_count):
                for rn in range(rn_count):
                    if last and issued == late_at:
                        hooks.late_refill()
                    issued += 1
                    mma(
                        rm,
                        rn,
                        a_frags[rm],
                        b_frags[rn],
                        a_scales[rm],
                        b_scales[rn],
                        step == 0,
                    )
        return out, None


class Pipeline:
    """PrefetchGlobalRead=2: the TDM runs two tiles ahead. Tile k refills its
    own buffer with tile k+2 once every workgroup of the cluster has read
    it; tile k+1 is waited for (and published) at the start of tile k's last
    K-step, with tile k+2's loads still in flight. Past the end the last
    tile is loaded again into a buffer nobody reads (SuppressNoLoadLoop).
    The loop runs tile pairs, one tile per buffer, so buffer offsets fold;
    an odd tile count peels one tile."""

    def __init__(
        self, k: _Kernel, tdm: TdmLoader, gl2: Gl2Prefetcher, tile: TileCompute
    ):
        self.k = k
        self.tdm = tdm
        self.gl2 = gl2
        self.tile = tile

    def _prefetch_ahead(self, k0, parity: int) -> None:
        distance = self.k.spec.gl2_prefetch_distance
        if distance:
            self.gl2.prefetch(
                self.k.ops.v(self.k.ops.add(k0, distance * self.k.spec.tile_k)),
                (parity + distance) % 2,
            )

    def hooks(self, k0, parity, static_parity: int) -> _Hooks:
        k = self.k
        b, ops, spec = k.b, k.ops, k.spec
        two_tiles = 2 * spec.tile_k
        form = spec.signal_form

        def signal() -> None:
            k.fence()
            if form == "wave0":
                b.inline_asm(_CLUSTER_SIGNAL_WAVE0_ASM, "s,~{scc},~{memory}", [k.wave])
            else:
                b.s_waitcnt(lgkmcnt=0)
                b.inline_asm(_CLUSTER_SIGNAL_ASM, "~{scc},~{memory}")
            k.fence()

        def refill() -> None:
            self._prefetch_ahead(k0, static_parity)
            k.fence()
            if spec.local_prefetch:
                # Tile k+1 is read next, from this point: it is the only TDM
                # load in flight, and the barrier below publishes it.
                b.s_wait_tensorcnt(0)
                b.s_barrier_bare()
                k.fence()

        def late() -> None:
            k.fence()
            if spec.has_cluster:
                b.inline_asm(_CLUSTER_WAIT_ASM, "~{memory}")
            if form != "drained":
                # No wave drained its reads of this buffer before the signal;
                # the workgroup barrier publishes them here.
                pending = spec.reads_per_step if spec.local_prefetch else 0
                b.inline_asm(f"s_wait_dscnt {pending}", "~{memory}")
                b.s_barrier_bare()
            self.tdm.issue(k.clamp_k(ops.v(ops.add(k0, two_tiles))), parity)
            k.fence()

        return _Hooks(
            refill=refill, signal=signal if spec.has_cluster else None, late_refill=late
        )

    def wait_next_tile(self) -> None:
        b = self.k.b
        b.s_wait_tensorcnt(2)
        b.s_barrier_bare()
        b.inline_asm("", "~{memory}")

    def run(self):
        k = self.k
        b, ops, spec, c0, c1 = k.b, k.ops, k.spec, k.c0, k.c1
        n_acc = spec.wave_repeat_m * spec.wave_repeat_n
        read_ahead = spec.local_prefetch == 1
        tile = self.tile
        k.cluster_sync()
        self.tdm.issue(c0, c0)
        self.tdm.issue(k.tile_k_v, c1)
        for ahead in range(2, spec.gl2_prefetch_distance):
            self.gl2.prefetch(ops.v(ahead * spec.tile_k), ahead % 2)
        peel = spec.init_c_peel
        if peel or read_ahead:
            self.wait_next_tile()
        undef = [b.undef_vec(F32, _ACCUMULATORS_PER_ATOM) for _ in range(n_acc)]
        first_head = tile.first_head(c0) if read_ahead else None
        if not peel:
            accumulators = [
                b.zero_vec_f32(_ACCUMULATORS_PER_ATOM) for _ in range(n_acc)
            ]
            head_values = tile.flatten(first_head) if read_ahead else []
            start, start_parity = c0, 0
        else:
            accumulators, head = tile.run(
                c0,
                undef,
                seed_zero=True,
                hooks=self.hooks(c0, c0, 0),
                head=first_head,
                next_parity=c1,
            )
            head_values = tile.flatten(head) if read_ahead else []
            start, start_parity = k.tile_k_v, 1

        def loop_tile(k_tile, parity: int, acc, head_vals):
            if not read_ahead:
                self.wait_next_tile()
            updated, head = tile.run(
                parity,
                acc,
                hooks=self.hooks(k_tile, parity, parity),
                head=tile.unflatten(head_vals) if read_ahead else None,
                next_parity=1 - parity,
            )
            return updated, tile.flatten(head) if read_ahead else []

        def carried(acc, head_vals):
            return [(f"acc{i}", v) for i, v in enumerate(acc)] + [
                (f"head{i}", v) for i, v in enumerate(head_vals)
            ]

        if (spec.K // spec.tile_k - start_parity) % 2:
            accumulators, head_values = loop_tile(
                start, start_parity, accumulators, head_values
            )
            start, start_parity = ops.add(start, spec.tile_k), 1 - start_parity
        loop = b.scf_for_iter(
            ops.v(start),
            k.params["K"],
            b.const_i32(2 * spec.tile_k),
            carried(accumulators, head_values),
            iv_name="k0",
            elide_trailing_barrier=False,
        )
        with loop as (k0, values):
            acc, head_vals = list(values[:n_acc]), list(values[n_acc:])
            for pair_index, parity in enumerate((start_parity, 1 - start_parity)):
                acc, head_vals = loop_tile(
                    ops.add(k0, pair_index * spec.tile_k), parity, acc, head_vals
                )
            b.scf_yield(*acc, *head_vals)
        # No workgroup may exit while a TDM load into its LDS is in flight:
        # with multicast, peers' loads write here too.
        b.s_wait_tensorcnt(0)
        if spec.has_cluster:
            k.cluster_sync()
        return loop.results[:n_acc]


class Epilogue:
    """``alpha * acc + beta * C`` stored as one vector per accumulator slot;
    the ``beta == 0`` path never reads C. FP8 saturates as hipBLASLt (+-inf
    to NaN, NaN kept, finite values clamped to +-448); FP16 and BF16 round
    to nearest even without clamping, as its v_cvt_pk_f16_f32 and
    v_cvt_pk_bf16_f32."""

    def __init__(self, k: _Kernel, rows: RowMaps):
        self.k = k
        self.rows = rows

    def _saturate(self, value):
        b = self.k.b
        value = b.fma(value, b.const_f32(0.0), value)
        return b.fminimum(
            b.fmaximum(value, b.const_f32(-_FP8_MAX)), b.const_f32(_FP8_MAX)
        )

    def _pack_fp8(self, values):
        b = self.k.b
        if self.k.spec.fp8_convert_width == 8:
            width, convert = 8, b.cvt_pk8_fp8_f32x8
        else:
            width, convert = 4, b.cvt_pk_fp8_f32x4
        packed = None
        for chunk in range(len(values) // width):
            part = convert(b.vec_pack(values[chunk * width : (chunk + 1) * width], F32))
            packed = part if packed is None else b.vec_concat(packed, part)
        return packed

    def _store(self, final, read_c: bool) -> None:
        k = self.k
        b, ops, spec, p = k.b, k.ops, k.spec, k.params
        rm_count, rn_count = spec.wave_repeat_m, spec.wave_repeat_n
        out_type = _OUT_TYPES[spec.out_dtype]
        out_bytes = _OUT_BYTES[spec.out_dtype]
        # One global access moves at most 16 B.
        width = min(rn_count, 16 // out_bytes)
        chunks = range(0, rn_count, width)
        fp8 = spec.out_dtype == "fp8e4m3"
        col = ops.add(k.block_n, self.rows.n_row(0, k.lane_col))
        for rm in range(rm_count):
            for slot in range(_ACCUMULATORS_PER_ATOM):
                m_local = ops.add((slot // 8) * 16 + slot % 8, ops.mul(k.lane_half, 8))
                row = ops.add(k.block_m, self.rows.m_row(rm, m_local))
                index = ops.v(ops.add(ops.mul(row, p["N"]), col))
                values = [
                    b.fmul(p["alpha"], b.vec_extract(final[rm * rn_count + rn], slot))
                    for rn in range(rn_count)
                ]
                if read_c:
                    c_words = [
                        b.global_load_vN(
                            p["C"],
                            ops.v(ops.add(index, begin)),
                            out_type,
                            width,
                            align=width * out_bytes,
                        )
                        for begin in chunks
                    ]

                    def c_value(i, words=c_words):
                        c = b.vec_extract(words[i // width], i % width)
                        return b.cvt_fp8_to_f32(c) if fp8 else b.cast_to_f32(c)

                    values = [
                        b.fadd(value, b.fmul(p["beta"], c_value(i)))
                        for i, value in enumerate(values)
                    ]
                if fp8:
                    values = [self._saturate(v) for v in values]
                for begin in chunks:
                    part = values[begin : begin + width]
                    if fp8:
                        packed = self._pack_fp8(part)
                    elif spec.out_dtype == "fp16":
                        packed = b.vec_pack([b.trunc_f32_to_f16(v) for v in part], F16)
                    else:
                        packed = b.vec_pack(
                            [b.trunc_f32_to_bf16(v) for v in part], BF16
                        )
                    b.global_store_vN(
                        p["D"],
                        ops.v(ops.add(index, begin)),
                        packed,
                        width,
                        align=width * out_bytes,
                    )

    def emit(self, final) -> None:
        b, beta = self.k.b, self.k.params["beta"]
        # GW_B0 / GW_B1.
        with b.scf_if_else(b.fcmp("oeq", beta, b.const_f32(0.0))) as (then_, else_):
            with then_:
                self._store(final, read_c=False)
            with else_:
                self._store(final, read_c=True)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_mxfp4_gemm(spec: MxFp4GemmSpec, *, arch: str = "gfx1250") -> KernelDef:
    reason = validate(spec, arch)
    if reason:
        raise ValueError(f"invalid mxfp4_gemm spec: {reason}")
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block_size
    b.kernel.attrs[EXPERT_SCHEDULING_ATTR] = True
    b.kernel.attrs[SOURCE_ORDER_ATTR] = True
    if spec.has_cluster:
        b.set_cluster_dims(*spec.cluster_dim)
    ops = _IrOps(b)
    out_type = _OUT_TYPES[spec.out_dtype]
    params = {
        "A": b.param("A", PtrType(I8, "global"), noalias=True, readonly=True, align=16),
        "B": b.param("B", PtrType(I8, "global"), noalias=True, readonly=True, align=16),
        "A_scale": b.param(
            "A_scale", PtrType(I8, "global"), noalias=True, readonly=True, align=4
        ),
        "B_scale": b.param(
            "B_scale", PtrType(I8, "global"), noalias=True, readonly=True, align=4
        ),
        "C": b.param(
            "C",
            PtrType(out_type, "global"),
            noalias=True,
            readonly=True,
            align=_OUT_BYTES[spec.out_dtype],
        ),
        "D": b.param(
            "D",
            PtrType(out_type, "global"),
            noalias=True,
            writeonly=True,
            align=_OUT_BYTES[spec.out_dtype],
        ),
        "M": b.param("M", I32),
        "N": b.param("N", I32),
        "K": b.param("K", I32),
        "alpha": b.param("alpha", F32),
        "beta": b.param("beta", F32),
    }
    k = _Kernel(b=b, ops=ops, spec=spec, layout=spec.lds_layout, params=params)
    k.c0 = b.const_i32(0)
    k.c1 = b.const_i32(1)
    k.tile_k_v = b.const_i32(spec.tile_k)
    k.packed_k = ops.div(params["K"], 2)
    # Global-scale group count; unused with TDM scales, kept so the IR
    # matches the Step 6 kernel's.
    ops.div(params["K"], 32)

    k.tid = b.thread_id_x()
    k.lane = ops.mod(k.tid, _WAVE)
    k.lane_col = ops.mod(k.lane, 16)
    k.lane_half = ops.div(k.lane, 16)
    # Wave-uniform, so the TDM descriptors built from it stay scalar.
    k.wave = b.readfirstlane(ops.div(k.tid, _WAVE))
    k.wave_n = ops.mod(k.wave, spec.warp_n)
    k.wave_m = ops.div(k.wave, spec.warp_n)

    block_x, block_y = b.block_id_x(), b.block_id_y()
    if spec.cluster_order == "aid":
        block_x, block_y = _aid_remapped_blocks(k, block_x, block_y)
    k.block_m = ops.mul(block_y, spec.tile_m)
    k.block_n = ops.mul(block_x, spec.tile_n)
    if spec.has_cluster:
        k.cluster_wx = b.cluster_workgroup_id("x")
        k.cluster_wy = b.cluster_workgroup_id("y")

    rows = RowMaps(k)
    tdm = TdmLoader(k)
    k.shared = b.smem_alloc(I8, [spec.lds_bytes], name_hint="ab_tdm")
    k.shared_addr = b.smem_addr_of(k.shared)
    tdm.hoist()
    reader = LdsReader(k, rows)
    pipeline = Pipeline(k, tdm, Gl2Prefetcher(k), TileCompute(k, reader))
    final = pipeline.run()
    Epilogue(k, rows).emit(final)
    b.ret()
    return b.kernel

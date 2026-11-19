# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 block-scaled low-bit dense GEMM.

The default path preserves the existing K=64 FP8/BF8 WMMA plus software
post-scaling contract. ``matrix_path="wmma_scale"`` and ``"wmma_scale16"``
select the native gfx1250 K=128 instructions and consume packed encoded scale
bytes directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from ...core.dtypes import dtype_info, normalize_dtype
from ...core.arch import ArchTarget
from ...core.arch.wmma_scale import gfx1250_scaled_wmma
from ...core.ir import (
    F16,
    F32,
    I8,
    I32,
    I64,
    IRBuilder,
    KernelDef,
    PtrType,
    Type,
    VectorType,
    dtype_to_ir_type,
)
from ...core.ir_storage import storage_ir_type
from ...helpers.mma_operand import prepare_scaled_mma_operand
from ...helpers.tiling import (
    Fragment,
    load_fragment,
    make_matrix_fragment_desc,
    make_mma_accumulator_desc,
    make_mma_scale_desc,
    make_tensor_desc,
    make_tile_desc,
    make_window,
    store_fragment,
)
from ...core.storage import BitPacking, FragmentPacking, MatrixFragmentLayout
from ...helpers.spec import (
    SignatureBuilder,
    ceil_div_grid,
    kernel_name_join,
    salted_kernel_name,
)

_LOWBIT_FORMATS = {
    "fp8e4m3": "fp8",
    "bf8e5m2": "bf8",
    "fp6e2m3": "fp6",
    "fp6e3m2": "bf6",
    "fp4e2m1": "fp4",
}
_LOWBIT_DTYPES = frozenset(_LOWBIT_FORMATS)
_OUTPUT_DTYPES = {"fp16", "f16", "bf16"}
_SCALE_DTYPES = {"fp16", "f16", "fp32", "f32"}
_SUPPORTED_MATRIX_PATHS = {
    "auto",
    "wmma",
    "wmma_scaffold",
    "wmma_scale",
    "wmma_scale16",
    "mfma",
}
_BLOCK_M = 16
_BLOCK_N = 16
_WMMA_K = 64
_WMMA_SCALE_K = 128
_WAVE = 32
_HALF_K = 32  # K-elements per lane-half for the K=64 atom
_ACC = 8  # accumulator slots per lane (<8 x f32>)
# ``prefetch=True`` sizes its scalar prefetch in lines of this many bytes. It
# is the largest line size assumed, so the hint never reaches past the
# workgroup's own scale rows.
_PREFETCH_LINE_BYTES = 128


def _wmma_op_id(dtype_a: str, dtype_b: str) -> str:
    return (
        f"wmma_gfx1250_f32_16x16x64_{_canon_lowbit(dtype_a)}_{_canon_lowbit(dtype_b)}"
    )


def _canon_lowbit(dtype: str) -> str:
    """Map a normalized catalog dtype to its stable instruction-format token."""
    try:
        return _LOWBIT_FORMATS[normalize_dtype(dtype)]
    except KeyError:
        raise ValueError(f"unsupported low-bit matrix dtype: {dtype!r}") from None


def _wire_scale_dtype(dtype: str) -> str:
    if dtype in ("fp16", "f16"):
        return "f16"
    if dtype in ("fp32", "f32"):
        return "f32"
    raise ValueError(f"scale_dtype must be fp16/fp32, got {dtype!r}")


@dataclass(frozen=True)
class BlockScaledGemmSpec:
    """One RCR block-scaled low-bit dense GEMM contract for gfx1250."""

    name: str
    M: int
    N: int
    K: int
    dtype_a: str = "fp8e4m3"
    dtype_b: str = "fp8e4m3"
    dtype_c: str = "bf16"
    dtype_acc: str = "fp32"
    scale_dtype: str = "fp32"
    block_k: int = 128
    layout: str = "RCR"
    matrix_path: str = "auto"
    tile_m: int = 16
    tile_n: int = 16
    tile_k: int = 128

    # Data-prefetch hints (gfx1250, llvm23 only): one scalar prefetch of the
    # workgroup's A-scale rows and a per-lane ``global_prefetch`` of the next
    # K step's A/B rows. Results do not change; off keeps the default IR.
    prefetch: bool = False
    scale_dtype_a: str | None = None
    scale_dtype_b: str | None = None
    # Runtime tensor factors are supported for FP4/E4M3/SCALE16 (NVFP4) only.
    tensor_scale: bool = False

    def resolved_scale_dtypes(self) -> tuple[str, str]:
        aliases = {"i8": "e8m0", "fp8e4m3": "e4m3"}
        return tuple(
            aliases.get(dtype, dtype)
            for dtype in (
                self.scale_dtype if self.scale_dtype_a is None else self.scale_dtype_a,
                self.scale_dtype if self.scale_dtype_b is None else self.scale_dtype_b,
            )
        )

    def __post_init__(self) -> None:
        if type(self.tensor_scale) is not bool:
            raise TypeError("tensor_scale must be a bool")
        if type(self.prefetch) is not bool:
            raise TypeError("prefetch must be a bool")
        object.__setattr__(self, "dtype_a", normalize_dtype(self.dtype_a))
        object.__setattr__(self, "dtype_b", normalize_dtype(self.dtype_b))

    @property
    def block_size(self) -> int:
        return _WAVE

    def kernel_name(self) -> str:
        base = kernel_name_join(
            self.name,
            "block_scaled",
            f"{_canon_lowbit(self.dtype_a)}_{_canon_lowbit(self.dtype_b)}",
            f"out_{normalize_dtype(self.dtype_c)}",
            f"bk{self.block_k}",
            f"t{self.tile_m}x{self.tile_n}x{self.tile_k}",
            flags={
                self.resolved_matrix_path(): True,
                "tensor_scale": self.tensor_scale,
                **{
                    f"s{operand}_{dtype}": True
                    for operand, dtype in zip("ab", self.resolved_scale_dtypes())
                    if self.resolved_matrix_path() in ("wmma_scale", "wmma_scale16")
                    and dtype != "e8m0"
                },
            },
        )
        # N specializes addressing and K specializes the unrolled instruction
        # sequence. M affects only launch geometry, not the generated body.
        specialized = salted_kernel_name(base, (("N", self.N), ("K", self.K)))
        return kernel_name_join(specialized, flags={"pf": self.prefetch})

    def resolved_matrix_path(self) -> str:
        # "auto" and the legacy "wmma_scaffold" token both resolve to the real
        # gfx1250 K=64 WMMA path now that the kernel body is implemented.
        if self.matrix_path in ("auto", "wmma_scaffold"):
            return "wmma"
        return self.matrix_path


def _native_scaled_atom(spec: BlockScaledGemmSpec, target: ArchTarget):
    return target.mma.op_for_shape(
        family="wmma_scaled",
        a_dtype=spec.dtype_a,
        b_dtype=spec.dtype_b,
        c_dtype=spec.dtype_acc,
        scales=(*spec.resolved_scale_dtypes(), spec.block_k),
        m=_BLOCK_M,
        n=_BLOCK_N,
        k=_WMMA_SCALE_K,
    )


def is_valid_spec(spec: BlockScaledGemmSpec, arch: str = "gfx1250") -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for the gfx1250 block-scaled GEMM contract."""
    from ...core.arch import ArchTarget

    if type(spec.tensor_scale) is not bool:
        return False, "tensor_scale must be a bool"
    if type(spec.prefetch) is not bool:
        return False, "prefetch must be a bool"

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)

    if arch != "gfx1250":
        return False, f"block_scaled_gemm scaffold is gfx1250-only (got {arch!r})"
    if spec.matrix_path not in _SUPPORTED_MATRIX_PATHS:
        return False, (
            f"matrix_path must be one of {sorted(_SUPPORTED_MATRIX_PATHS)} "
            f"(got {spec.matrix_path!r})"
        )
    if spec.matrix_path == "mfma" or spec.resolved_matrix_path() == "mfma":
        return False, (
            "gfx1250 has no MFMA block_scale path; use matrix_path='wmma' "
            "(the K=64 FP8/BF8 WMMA atom)"
        )
    if not target.has_wmma:
        return False, f"{arch} does not expose WMMA for block-scaled GEMM"
    if target.has_mfma:
        return False, f"{arch} unexpectedly exposes MFMA; expected WMMA-only"

    if spec.M <= 0 or spec.N <= 0 or spec.K <= 0:
        return False, f"M/N/K must be positive (got M={spec.M}, N={spec.N}, K={spec.K})"
    if (
        normalize_dtype(spec.dtype_a) not in _LOWBIT_DTYPES
        or normalize_dtype(spec.dtype_b) not in _LOWBIT_DTYPES
    ):
        return False, (
            f"A/B must be fp8, bf8, fp6, bf6, or fp4 (got A={spec.dtype_a!r}, B={spec.dtype_b!r})"
        )
    matrix_path = spec.resolved_matrix_path()
    native_scale = matrix_path in ("wmma_scale", "wmma_scale16")
    family = matrix_path if native_scale else "wmma"
    atom_k = _WMMA_SCALE_K if native_scale else _WMMA_K
    if not native_scale and not target.mma.has_shape(
        family=family,
        a_dtype=_canon_lowbit(spec.dtype_a),
        b_dtype=_canon_lowbit(spec.dtype_b),
        c_dtype="fp32",
        m=_BLOCK_M,
        n=_BLOCK_N,
        k=atom_k,
    ):
        return False, (
            f"no gfx1250 16x16x{atom_k} {family} atom for "
            f"{_canon_lowbit(spec.dtype_a)}/{_canon_lowbit(spec.dtype_b)}"
        )
    if spec.dtype_c not in ("bf16", "fp16", "f16"):
        return False, f"gfx1250 output must be bf16/fp16 (got {spec.dtype_c!r})"
    if spec.dtype_acc not in ("fp32", "f32"):
        return False, f"accumulator dtype must be fp32 (got {spec.dtype_acc!r})"
    if spec.layout != "RCR":
        return False, f"block_scaled_gemm supports RCR only (got {spec.layout!r})"
    if spec.tensor_scale and not native_scale:
        return False, "tensor scales require native scaled WMMA"
    if spec.tensor_scale and (
        spec.dtype_a != normalize_dtype("fp4")
        or spec.dtype_b != normalize_dtype("fp4")
        or matrix_path != "wmma_scale16"
        or spec.block_k != 16
        or spec.resolved_scale_dtypes() != ("e4m3", "e4m3")
    ):
        return False, "tensor scales require the FP4/E4M3/SCALE16 NVFP4 contract"
    if native_scale:
        required_block_k = 16 if matrix_path == "wmma_scale16" else 32
        if spec.block_k != required_block_k:
            return False, (
                f"{matrix_path} requires block_k={required_block_k} scale groups "
                f"(got {spec.block_k})"
            )
        try:
            atom = _native_scaled_atom(spec, target)
        except ValueError as exc:
            return False, str(exc)
        if atom is None:
            return (
                False,
                "no gfx1250 scaled WMMA atom for the requested operand and scale contract",
            )

    else:
        if spec.scale_dtype_a is not None or spec.scale_dtype_b is not None:
            return False, "per-operand scale types require native scaled WMMA"
        try:
            _wire_scale_dtype(spec.scale_dtype)
        except ValueError as e:
            return False, str(e)
    if spec.block_k <= 0 or spec.K % spec.block_k:
        return False, f"K ({spec.K}) must be divisible by block_k ({spec.block_k})"
    if not native_scale and spec.block_k % _WMMA_K:
        return False, f"block_k ({spec.block_k}) must be a multiple of {_WMMA_K}"
    if spec.K % atom_k:
        return False, f"K ({spec.K}) must be a multiple of the WMMA K={atom_k}"
    if (spec.tile_m, spec.tile_n) != (_BLOCK_M, _BLOCK_N):
        return False, "gfx1250 block_scaled_gemm uses fixed 16x16 output tiles"
    if spec.M % _BLOCK_M or spec.N % _BLOCK_N:
        return False, "M and N must be multiples of 16"

    if native_scale:
        return (
            True,
            f"ok: gfx1250 K=128 native {matrix_path} "
            f"{_canon_lowbit(spec.dtype_a)}x{_canon_lowbit(spec.dtype_b)} GEMM",
        )
    return True, "ok: gfx1250 K=64 FP8/BF8 WMMA block-scaled GEMM"


def block_scaled_gemm_signature(spec: BlockScaledGemmSpec) -> List[dict]:
    """Return the manifest signature for the selected scaling path."""
    scale = (
        "i8"
        if spec.resolved_matrix_path() in ("wmma_scale", "wmma_scale16")
        else _wire_scale_dtype(spec.scale_dtype)
    )
    signature = (
        SignatureBuilder()
        .ptr(
            "A",
            (
                "i8"
                if storage_ir_type(spec.dtype_a) == I8
                else _canon_lowbit(spec.dtype_a)
            ),
        )
        .ptr(
            "B",
            (
                "i8"
                if storage_ir_type(spec.dtype_b) == I8
                else _canon_lowbit(spec.dtype_b)
            ),
        )
        .ptr("A_scale", scale)
        .ptr("B_scale", scale)
        .ptr("C", spec.dtype_c)
        .scalar("M", "i32")
        .scalar("N", "i32")
        .scalar("K", "i32")
    )
    if spec.tensor_scale:
        signature.scalar("A_tensor_scale", "f32").scalar("B_tensor_scale", "f32")
    return signature.build()


def block_scaled_gemm_grid(spec: BlockScaledGemmSpec) -> Tuple[int, int, int]:
    return ceil_div_grid((spec.N, spec.tile_n), (spec.M, spec.tile_m))


def _scale_type(dtype: str) -> Type:
    return F16 if _wire_scale_dtype(dtype) == "f16" else F32


def _as_f32(b: IRBuilder, v):
    return v if v.type.name == "f32" else b.cast_to_f32(v)


def build_block_scaled_gemm(
    spec: BlockScaledGemmSpec, arch: str = "gfx1250"
) -> KernelDef:
    """Build a gfx1250 block-scaled GEMM (RCR, ``C = A @ B^T``).

    One wave (32 lanes) computes one 16x16 output tile without LDS. The legacy
    ``wmma`` path uses K=64 FP8/BF8 atoms, accumulates each ``block_k`` group,
    and applies FP16/FP32 A/B scales in software. The native ``wmma_scale`` and
    ``wmma_scale16`` paths use K=128 scaled WMMA atoms and pass packed encoded scale bytes
    directly to the instruction, with K=32 and K=16 scale groups respectively.

    Lane ``l`` owns output column ``l % 16`` and rows
    ``(l // 16) * 8 : (l // 16 + 1) * 8``. Legacy matrix fragments carry 32
    low-bit bytes per lane as ``<8 x i32>``. For FP8/BF8, native fragments carry 64 bytes as
    ``<16 x i32>`` as four 16-byte K chunks, alternating chunks between lane
    halves. Both paths use the gfx12 column-distributed ``<8 x f32>``
    accumulator layout. FP6 uses prepacked E2M3 or E3M2 bytes: A is
    [M, 3*K/4], B is [N, 3*K/4], four codes per three little-endian bytes.
    Each lane pads twelve packed i32 words to the sixteen-word builtin ABI.
    FP4 uses prepacked E2M1 bytes: A is [M, K/2], B is [N, K/2],
    low nibble first along K. Each lane pads eight packed i32 words
    to the sixteen-word builtin ABI.
    Scale arrays remain A_scale[M, K/block_k] and B_scale[K/block_k, N].

    ``spec.prefetch`` adds data-prefetch hints, which need gfx1250 and the
    llvm23 flavor: the wave enables scalar prefetch and prefetches its
    contiguous A_scale rows once, then each K step prefetches the next step's
    A and B rows per lane. The last step prefetches nothing, so no hint
    reaches past the operands.
    """
    ok, reason = is_valid_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid block_scaled_gemm spec for {arch}: {reason}")

    a_ty = storage_ir_type(spec.dtype_a)
    b_ty = storage_ir_type(spec.dtype_b)
    c_ty = storage_ir_type(spec.dtype_c)
    matrix_path = spec.resolved_matrix_path()
    native_scale = matrix_path in ("wmma_scale", "wmma_scale16")
    scale_ty = (
        storage_ir_type("e8m0") if native_scale else _scale_type(spec.scale_dtype)
    )
    atom = (
        _native_scaled_atom(spec, ArchTarget.from_gfx(arch)) if native_scale else None
    )
    op_id = atom.op_id if atom is not None else _wmma_op_id(spec.dtype_a, spec.dtype_b)
    scale_op = gfx1250_scaled_wmma(op_id)
    frag_words = atom.a_frag_len if atom is not None else _ACC
    a_frag_ty = VectorType(I32, frag_words)
    descriptor_atom = (
        atom if atom is not None else ArchTarget.from_gfx(arch).mma.by_op_id(op_id)
    )
    if descriptor_atom is None:
        raise ValueError(f"missing matrix operand contract: {op_id}")
    result_desc = make_mma_accumulator_desc(descriptor_atom, target=arch, result=True)
    if native_scale:
        a_layout, b_layout = scale_op.matrix_layout("a"), scale_op.matrix_layout("b")
    else:
        a_layout = MatrixFragmentLayout(
            FragmentPacking(BitPacking(8), 32, 32, 8), 32, 2, 16
        )
        b_layout = a_layout
    a_desc, b_desc = make_matrix_fragment_desc(a_layout), make_matrix_fragment_desc(
        b_layout
    )
    a_tensor = make_tensor_desc(
        (spec.M, spec.K), (spec.K, 1), dtype_to_ir_type(spec.dtype_a)
    )
    b_tensor = make_tensor_desc(
        (spec.N, spec.K), (spec.K, 1), dtype_to_ir_type(spec.dtype_b)
    )
    c_tensor = make_tensor_desc(
        (spec.M, spec.N), (spec.N, 1), dtype_to_ir_type(spec.dtype_c)
    )

    groups = spec.K // spec.block_k
    steps_per_group = spec.block_k // _WMMA_K

    ir = IRBuilder(spec.kernel_name())
    ir.kernel.attrs["max_workgroup_size"] = spec.block_size

    A = ir.param("A", PtrType(a_ty, "global"), noalias=True, readonly=True, align=16)
    B = ir.param("B", PtrType(b_ty, "global"), noalias=True, readonly=True, align=16)
    AScale = ir.param(
        "A_scale", PtrType(scale_ty, "global"), noalias=True, readonly=True, align=4
    )
    BScale = ir.param(
        "B_scale", PtrType(scale_ty, "global"), noalias=True, readonly=True, align=4
    )
    C = ir.param("C", PtrType(c_ty, "global"), noalias=True, writeonly=True, align=16)
    M = ir.param("M", I32)  # noqa: F841 - ABI mirror; grid defines bounds
    N = ir.param("N", I32)  # noqa: F841
    K = ir.param("K", I32)  # noqa: F841

    if spec.tensor_scale:
        a_tensor_scale = ir.param("A_tensor_scale", F32)
        b_tensor_scale = ir.param("B_tensor_scale", F32)
        tensor_scale = ir.fmul(a_tensor_scale, b_tensor_scale)

    cK = ir.const_i32(spec.K)
    c16 = ir.const_i32(_BLOCK_M)
    c32 = ir.const_i32(_WAVE)

    lane = ir.mod(ir.thread_id_x(), c32)
    frag = ir.mod(lane, c16)  # lane%16: A row / B col / output col

    m0 = ir.mul(ir.block_id_y(), c16)
    n0 = ir.mul(ir.block_id_x(), c16)
    a_row = ir.add(m0, frag)  # this lane's A row
    b_row = ir.add(n0, frag)  # this lane's B row (= output col n)
    a_packing = BitPacking(dtype_info(spec.dtype_a).encoded_bits)
    b_packing = BitPacking(dtype_info(spec.dtype_b).encoded_bits)

    def _row_stride(packing):
        # All current pointer types use their natural byte size; low-bit inputs
        # use an explicit byte carrier. A/B storage contracts stay independent.
        if packing.element_bits >= 8:
            return cK
        values, packed_bytes = packing.group(8)
        groups = ir.div(cK, ir.const_i32(values))
        return (
            groups if packed_bytes == 1 else ir.mul(groups, ir.const_i32(packed_bytes))
        )

    a_stride = _row_stride(a_packing)
    b_stride = a_stride if a_packing == b_packing else _row_stride(b_packing)
    a_base = ir.mul(a_row, a_stride)
    b_base = ir.mul(b_row, b_stride)

    if spec.prefetch:
        # A_scale rows m0 .. m0+15 are contiguous and m0 is wave-uniform, as
        # s_prefetch_data requires.
        scale_bytes = {I8: 1, F16: 2, F32: 4}[scale_ty]
        tile_bytes = _BLOCK_M * groups * scale_bytes
        # The length field encodes chunks - 1 (0..31 = 1..32 chunks of 128 B),
        # and the hardware rounds the start down to 128 B. The start alignment
        # is not known at build time, so cover a worst-case 127-byte offset.
        lines = (tile_bytes + 2 * _PREFETCH_LINE_BYTES - 2) // _PREFETCH_LINE_BYTES
        prefetch_chunks = min(31, lines - 1)
        ir.enable_scalar_prefetch()
        # The ISA requires XCNT to drain before a scalar prefetch.
        ir.s_wait_xcnt(0)
        ir.s_prefetch_data(
            ir.global_ptr_add(AScale, ir.mul(m0, ir.const_i32(groups * scale_bytes))),
            ir.const_i32(prefetch_chunks),
        )

    def _prefetch_step(k_next):
        # a_base/b_base are in bytes (see _row_stride), so the next K step is
        # k_next * element_bits / 8 bytes further along the row.
        if not spec.prefetch or k_next >= spec.K:
            return
        for ptr, base, packing in ((A, a_base, a_packing), (B, b_base, b_packing)):
            k_bytes = k_next * packing.element_bits // 8
            ir.global_prefetch(
                ir.global_ptr_add(ptr, ir.add(base, ir.const_i32(k_bytes)))
            )

    operands = {"a": (a_tensor, a_desc, m0), "b": (b_tensor, b_desc, n0)}

    def _load_frag(ptr, k0, operand):
        tensor, desc, origin = operands[operand]
        fragment = load_fragment(
            ir, ptr, make_window(tensor, (origin, k0)), desc, lane, alignment_bytes=16
        )
        return (
            prepare_scaled_mma_operand(ir, atom, operand, fragment)
            if native_scale
            else ir.bitcast(fragment.value, a_frag_ty)
        )

    if native_scale:
        scale_desc = make_mma_scale_desc(atom, target=arch)
        scale_count = scale_desc.register_count
        scale_a, scale_b = spec.resolved_scale_dtypes()
        a_scales = make_tensor_desc(
            (spec.M, groups), (groups, 1), dtype_to_ir_type(scale_a)
        )
        b_scales = make_tensor_desc(
            (spec.N, groups), (1, spec.N), dtype_to_ir_type(scale_b)
        )
    else:
        # Legacy scales act in the epilogue, not as native MMA operands.
        # Keep A's per-lane rows and broadcast B's column across half-waves.
        rows, cols = result_desc.shape
        rows_per_lane = result_desc.register_count
        wave = descriptor_atom.wave_size
        a_scale_desc = make_tile_desc(
            shape=(rows, 1),
            thread_dist=(rows // rows_per_lane, 1),
            thread_tile=(rows_per_lane, 1),
            thread_broadcast=(1, cols),
            wave_size=wave,
        )
        b_scale_desc = make_tile_desc(
            shape=(cols, 1),
            thread_dist=(cols, 1),
            thread_broadcast=wave // cols,
            wave_size=wave,
        )
        a_scales = make_tensor_desc((spec.M, groups), (groups, 1), scale_ty)
        b_scales = make_tensor_desc((spec.N, groups), (1, spec.N), scale_ty)

    def _load_scale_word(ptr, call_idx, *, for_b):
        tensor, origin = (b_scales, n0) if for_b else (a_scales, m0)
        fragment = load_fragment(
            ir,
            ptr,
            make_window(tensor, (origin, call_idx * scale_count)),
            scale_desc,
            lane,
            alignment_bytes=4,
        )
        return ir.bitcast(fragment.value, I64 if scale_count == 8 else I32)

    if native_scale:
        acc = ir.zero_vec_f32(_ACC)
        for step in range(spec.K // _WMMA_SCALE_K):
            k0 = step * _WMMA_SCALE_K
            _prefetch_step(k0 + _WMMA_SCALE_K)
            a_frag = _load_frag(A, k0, "a")
            b_frag = _load_frag(B, k0, "b")
            a_scale = _load_scale_word(AScale, step, for_b=False)
            b_scale = _load_scale_word(BScale, step, for_b=True)
            acc = ir.mma(op_id, a_frag, b_frag, acc, a_scale, b_scale)
        if spec.tensor_scale:
            scaled = []
            for i in range(_ACC):
                value = ir.fmul(ir.vec_extract(acc, i), tensor_scale)
                # Preserve FP32 rounding before the BF16/FP16 conversion.
                scaled.append(ir.optimization_barrier(value))
            acc = ir.vec_pack(scaled, F32)
        store_fragment(
            ir,
            C,
            make_window(c_tensor, (m0, n0)),
            Fragment(result_desc, F32, acc),
            lane,
            alignment_bytes=16,
        )
        return ir.kernel

    # Preserve the software post-scale accumulation order for every K group.
    outer = [ir.const_f32(0.0) for _ in range(_ACC)]
    for kg in range(groups):
        acc = ir.zero_vec_f32(_ACC)
        for step in range(steps_per_group):
            k0 = kg * spec.block_k + step * _WMMA_K
            _prefetch_step(k0 + _WMMA_K)
            a_frag = _load_frag(A, k0, "a")
            b_frag = _load_frag(B, k0, "b")
            acc = ir.mma(op_id, a_frag, b_frag, acc)
        sb = load_fragment(
            ir,
            BScale,
            make_window(b_scales, (n0, kg)),
            b_scale_desc,
            lane,
            alignment_bytes=4,
        )
        b_scale = _as_f32(ir, ir.vec_extract(sb.value, 0))
        sa = load_fragment(
            ir,
            AScale,
            make_window(a_scales, (m0, kg)),
            a_scale_desc,
            lane,
            alignment_bytes=4,
        )
        for i in range(_ACC):
            a_scale = _as_f32(ir, ir.vec_extract(sa.value, i))
            ab = ir.fmul(a_scale, b_scale)
            outer[i] = ir.fadd(outer[i], ir.fmul(ir.vec_extract(acc, i), ab))
    store_fragment(
        ir,
        C,
        make_window(c_tensor, (m0, n0)),
        Fragment(result_desc, F32, ir.vec_pack(outer, F32)),
        lane,
        alignment_bytes=16,
    )
    return ir.kernel

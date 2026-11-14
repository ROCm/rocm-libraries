# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx950 scaled MFMA: 32 logical values and one K32 E8M0 scale per lane."""

from ..dtypes import dtype_info
from ..storage import BitPacking, FragmentPacking, MatrixFragmentLayout
from .target import ArchTarget, _op_id_family

MATRIX_FORMATS = {"fp8e4m3": 0, "bf8e5m2": 1, "fp6e2m3": 2, "fp6e3m2": 3, "fp4e2m1": 4}
INTRINSIC_KEY = "mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32"

# The fixed-operation 16x16x128 MX atom (``IRBuilder.mfma_scale_f32_16x16x128_f8f6f4``):
# 32 bytes of A and of B per lane, a vec<f32x4> accumulator, two i32 scales.
MX16_OP_ID = "mfma_scale_f32_16x16x128_f8f6f4"
# The A / B types every lowering reinterprets as the intrinsic's <8 x i32>.
MX16_PAYLOAD_TYPES = ("vec<i32x8>", "vec<i8x32>", "vec<fp8e4m3x32>", "vec<bf8e5m2x32>")


def gfx950_scaled_mfma(op_id):
    if _op_id_family().get(op_id.removeprefix("tile.")) != "mfma_scaled":
        return None
    atom = ArchTarget.from_gfx("gfx950").mma.by_op_id(op_id.removeprefix("tile."))
    return atom if atom is not None and atom.family == "mfma_scaled" else None


def accepts_scale_sel(op_id):
    """Whether ``tile.mma`` with this ``op_id`` takes a ``scale_sel``: a
    ``mfma_scaled`` catalog atom or the 16x16x128 MX atom."""
    op_id = str(op_id).removeprefix("tile.")
    return op_id == MX16_OP_ID or gfx950_scaled_mfma(op_id) is not None


def scale_sel_refusal(op_id):
    """``None`` when ``op_id`` takes a ``scale_sel``, else the message every
    builder and lowering raises."""
    if accepts_scale_sel(op_id):
        return None
    return f"tile.mma {op_id}: scale_sel applies only to a scaled MFMA atom"


def _byte_index(value):
    # bool is an int subclass; True is not a byte index.
    return type(value) is int and 0 <= value <= 3


def scale_sel_error(sel):
    """``None`` when ``sel`` is two byte indices in 0..3, else the message every
    lowering raises. Only exact ints count: no bool, float or str coercion."""
    if (
        not isinstance(sel, (list, tuple))
        or len(sel) != 2
        or not all(_byte_index(s) for s in sel)
    ):
        return f"tile.mma scale_sel must be two byte indices in 0..3, got {sel!r}"
    return None


def lds_dma_offset_error(name, offset):
    """``None`` when ``offset`` is an exact int in 0..4095, else the message."""
    if type(offset) is not int or not 0 <= offset <= 4095:
        return f"{name} offset must be 0..4095 bytes, got {offset!r}"
    return None


def mx16_operand_error(operands, results=None):
    """Operand and result check for the 16x16x128 MX atom, shared by the
    builder and every lowering."""
    names = [v.type.name for v in operands]
    if (
        len(names) != 5
        or names[0] not in MX16_PAYLOAD_TYPES
        or names[1] not in MX16_PAYLOAD_TYPES
        or names[2:] != ["vec<f32x4>", "i32", "i32"]
    ):
        return (
            f"{MX16_OP_ID} requires A and B of 32 bytes ({', '.join(MX16_PAYLOAD_TYPES)}), "
            f"a vec<f32x4> accumulator and two i32 E8M0 scales, got ({', '.join(names)})"
        )
    if results is not None and [v.type.name for v in results] != ["vec<f32x4>"]:
        return f"{MX16_OP_ID} result must be vec<f32x4>"
    return None


def mx16_format_error(attrs):
    """The Python and C++ engines lower the 16x16x128 MX atom with fp8e4m3 A
    and B only (cbsz = blgp = 0). The direct compiler also reads other
    ``a_format`` / ``b_format`` values; refuse them here instead of silently
    computing fp8. Use a ``mfma_scaled`` catalog atom or ``mfma_scale_mx``
    for other formats."""
    for key in ("a_format", "b_format"):
        value = attrs.get(key, "fp8e4m3")
        if value != "fp8e4m3":
            return (
                f"{MX16_OP_ID}: {key} {value!r} is not lowered by the Python or C++ "
                "engine (fp8e4m3 only); use a mfma_scaled catalog atom or "
                "IRBuilder.mfma_scale_mx for other formats"
            )
    return None


def matrix_layout(dtype):
    if dtype not in MATRIX_FORMATS:
        raise ValueError("unsupported scaled MFMA matrix dtype")
    return MatrixFragmentLayout(
        FragmentPacking(BitPacking(dtype_info(dtype).encoded_bits), 32, 32, 8),
        16 if MATRIX_FORMATS[dtype] < 2 else 32,
        2,
        32,
    )


def scaled_mfma_error(op_id, operands, results=None):
    if str(op_id).removeprefix("tile.") == MX16_OP_ID:
        return mx16_operand_error(operands, results)
    if gfx950_scaled_mfma(op_id) is None:
        return None
    if [v.type.name for v in operands] != [
        "vec<i32x8>",
        "vec<i32x8>",
        "vec<f32x16>",
        "i32",
        "i32",
    ]:
        return "scaled MFMA requires two vec<i32x8> payloads, vec<f32x16> accumulator and two i32 E8M0 scales"
    if results is not None and [v.type.name for v in results] != ["vec<f32x16>"]:
        return "scaled MFMA result must be vec<f32x16>"
    return None

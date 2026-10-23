# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The f8f6f4 matrix formats and operand layouts, and the gfx950 scaled MFMA.

An f8f6f4 instruction (gfx950 MFMA, gfx1250 WMMA, scaled or not) reads A and B
each in one of five formats, chosen by an immediate format code. An operand's
register is sized for 8-bit values; a 6-bit or 4-bit format packs the same
number of values into its low words and leaves the rest zero. The gfx950
scaled MFMA holds 32 logical values and one K32 E8M0 scale per lane.
"""

from ..dtypes import dtype_info
from ..storage import BitPacking, FragmentPacking, MatrixFragmentLayout

# The format code an f8f6f4 instruction takes for A and for B, by matrix dtype.
MATRIX_FORMATS = {"fp8e4m3": 0, "bf8e5m2": 1, "fp6e2m3": 2, "fp6e3m2": 3, "fp4e2m1": 4}
INTRINSIC_KEY = "mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32"


def require_matrix_format(dtype: str) -> str:
    """The canonical name of matrix format ``dtype`` (``"fp4"`` -> ``"fp4e2m1"``).

    Raises ``ValueError`` naming the five formats when ``dtype`` is not one.
    """
    name = dtype_info(dtype).name
    if name not in MATRIX_FORMATS:
        raise ValueError(
            f"{dtype!r} is not an f8f6f4 matrix format; expected one of "
            f"{', '.join(MATRIX_FORMATS)}"
        )
    return name


def narrow_matrix_format_pairs() -> list[tuple[str, str]]:
    """Every (A, B) pair of :data:`MATRIX_FORMATS` with a 6-bit or 4-bit operand.

    A-major in format-code order, 21 pairs: ``(fp8e4m3, fp6e2m3)``,
    ``(fp8e4m3, fp6e3m2)``, ``(fp8e4m3, fp4e2m1)``, ... ``(fp4e2m1, fp4e2m1)``.
    The four 8-bit x 8-bit pairs are not in the list.
    """
    narrow = {name for name in MATRIX_FORMATS if dtype_info(name).encoded_bits < 8}
    return [
        (a, b)
        for a in MATRIX_FORMATS
        for b in MATRIX_FORMATS
        if a in narrow or b in narrow
    ]


def matrix_operand_layout(
    dtype: str, words: int, lane_groups: int, lanes_per_group: int
) -> MatrixFragmentLayout:
    """Where each value of one f8f6f4 operand of format ``dtype`` sits.

    The operand register is ``words`` 32-bit words, room for ``4 * words``
    8-bit values, and every format holds that many values per lane, packed from
    the low bit. The wave's lanes form ``lane_groups`` groups of
    ``lanes_per_group``: lane ``l`` holds row ``l % lanes_per_group`` of A (that
    column of B), and its values are runs of consecutive K, 16 long for an
    8-bit format and 32 for a 6-bit or 4-bit one, which the lane groups take in
    turn::

        16x16x128 fp8, 16 words, 2 groups x 16 lanes (64 values, runs of 16):
          lanes  0..15   K  0..15, 32..47, 64..79,  96..111
          lanes 16..31   K 16..31, 48..63, 80..95, 112..127

    The public CDNA4 ISA operand tables give these layouts for gfx950 (8 words,
    4 x 16 lanes at 16x16x128 and 2 x 32 at 32x32x64); CK uses the same runs
    for gfx1250 (16 words, 2 x 16 lanes).
    """
    name = require_matrix_format(dtype)
    bits = dtype_info(name).encoded_bits
    return MatrixFragmentLayout(
        FragmentPacking(BitPacking(bits), 4 * words, 32, words),
        16 if bits == 8 else 32,
        lane_groups,
        lanes_per_group,
    )


def gfx950_scaled_mfma(op_id):
    # Imported here: the catalog module builds its lane maps from this one.
    from .target import ArchTarget, _op_id_family

    if _op_id_family().get(op_id.removeprefix("tile.")) != "mfma_scaled":
        return None
    atom = ArchTarget.from_gfx("gfx950").mma.by_op_id(op_id.removeprefix("tile."))
    return atom if atom is not None and atom.family == "mfma_scaled" else None


def matrix_layout(dtype):
    if dtype not in MATRIX_FORMATS:
        raise ValueError("unsupported scaled MFMA matrix dtype")
    return matrix_operand_layout(dtype, 8, 2, 32)


def scaled_mfma_error(op_id, operands, results=None):
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

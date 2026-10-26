# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx950 scaled MFMA: 32 logical values and one K32 E8M0 scale per lane."""

from ..dtypes import dtype_info
from ..storage import BitPacking, FragmentPacking, MatrixFragmentLayout
from .target import ArchTarget, _op_id_family

MATRIX_FORMATS = {"fp8e4m3": 0, "bf8e5m2": 1, "fp6e2m3": 2, "fp6e3m2": 3, "fp4e2m1": 4}
INTRINSIC_KEY = "mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32"


def gfx950_scaled_mfma(op_id):
    if _op_id_family().get(op_id.removeprefix("tile.")) != "mfma_scaled":
        return None
    atom = ArchTarget.from_gfx("gfx950").mma.by_op_id(op_id.removeprefix("tile."))
    return atom if atom is not None and atom.family == "mfma_scaled" else None


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

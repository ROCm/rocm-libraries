# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx950 scaled MFMA with free formats, operand widths and scale byte select.

Metadata and validation for ``tile.mfma_scale_mx`` (built by
:meth:`rocke.core.ir.IRBuilder.mfma_scale_mx`). The op lowers to the overloaded
``mfma.scale.f32.{16x16x128,32x32x64}.f8f6f4`` intrinsic (the lowering in
``core/lower_llvm.py`` owns its text) with every
immediate exposed:

* ``fmt_a`` / ``fmt_b`` (``cbsz`` / ``blgp``): 0 fp8e4m3, 1 bf8e5m2, 2 fp6e2m3,
  3 fp6e3m2, 4 fp4e2m1. Each operand is ``<8|6|4 x i32>`` to match its format,
  so an fp4 operand occupies four VGPRs instead of eight.
* ``opsel_a`` / ``opsel_b`` (0..3): which byte of the 32-bit scale operand holds
  the E8M0 scale, so one dword can carry the scales of four K steps.

**Python engine only** (listed in ``rocke.core.ir.PYTHON_ONLY_OPCODES``).
The C++ engine has no lowering for this op. A kernel that
uses it must be compiled with ``backend="python"``, carries a distinct kernel
name, and is excluded from the C++ byte-identity gate. The catalog
``mfma_scaled`` 32x32x64 atom (``core/arch/mfma_scale.py``) is the dual-engine
route; this op is the free-format, ``op_sel`` and 16x16x128 route.

Lane maps (wave64, ``l`` = lane, ``s`` = logical element of the fragment).
They are Python metadata: bijectivity is checked on the host by
``platform/tests/core/test_mfma_scale_mx.py``, and the maps are verified on
gfx950 (bitwise against an exact reference) by ``library/tests/test_mx_mfma.py``:

16x16x128, ``g = l // 16``:
  * A element ``s`` of an fp8/bf8 operand is ``(row l % 16, k 16 g + 64 (s // 16) + s % 16)``:
    two 16-byte halves 64 K apart. fp6/fp4: ``k = 32 g + s``.
  * B element ``s``: ``(k, col l % 16)`` with the same K rule as A.
  * Scale lane ``l`` holds K block ``g`` (``k = 32 g .. 32 g + 31``) of row/col ``l % 16``.
  * C element ``i`` (0..3) is ``(row 4 g + i, col l % 16)``.

32x32x64, ``h = l // 32``:
  * fp8/bf8: ``k = 16 h + 32 (s // 16) + s % 16``; fp6/fp4: ``k = 32 h + s``;
    row/col ``l % 32``.
  * Scale lane ``l`` holds K block ``h`` of row/col ``l % 32``.
  * C element ``i`` (0..15) is ``(row 8 (i // 4) + 4 h + i % 4, col l % 32)``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .mfma_scale import MATRIX_FORMATS

#: Format code -> i32 words per lane of the operand.
OPERAND_WORDS = {0: 8, 1: 8, 2: 6, 3: 6, 4: 4}
#: Format code -> bits per element.
FORMAT_BITS = {0: 8, 1: 8, 2: 6, 3: 6, 4: 4}
#: dtype name -> format code (the catalog's table).
FORMAT_CODES = dict(MATRIX_FORMATS)
FORMAT_ALIASES = {
    "fp8": 0,
    "bf8": 1,
    "fp6": 2,
    "bf6": 3,
    "fp4": 4,
    **FORMAT_CODES,
}

OP_NAME = "tile.mfma_scale_mx"


@dataclass(frozen=True)
class MxShape:
    """One scaled MFMA shape: ``M x N x K`` and the accumulator length."""

    name: str
    m: int
    n: int
    k: int
    c_len: int

    @property
    def lane_groups(self) -> int:
        """Lanes sharing a row: 4 for 16x16x128, 2 for 32x32x64."""
        return 64 // self.m

    @property
    def k_blocks(self) -> int:
        return self.k // 32


SHAPES = {
    "16x16x128": MxShape("16x16x128", 16, 16, 128, 4),
    "32x32x64": MxShape("32x32x64", 32, 32, 64, 16),
}


def format_code(fmt) -> int:
    """Format code for ``fmt`` (an int 0..4 or a dtype name)."""
    if isinstance(fmt, bool):
        raise TypeError(f"bad scaled MFMA format {fmt!r}")
    if isinstance(fmt, int):
        if fmt not in OPERAND_WORDS:
            raise ValueError(f"scaled MFMA format code must be 0..4, got {fmt}")
        return fmt
    code = FORMAT_ALIASES.get(str(fmt))
    if code is None:
        raise ValueError(f"unknown scaled MFMA format {fmt!r}")
    return code


def shape(name: str) -> MxShape:
    try:
        return SHAPES[name]
    except KeyError:
        raise ValueError(
            f"scaled MFMA shape must be one of {sorted(SHAPES)}, got {name!r}"
        ) from None


def check_operands(attrs, types) -> str | None:
    """Return an error string if ``types`` (operand type names) do not fit ``attrs``."""
    try:
        sh = shape(attrs["shape"])
        fa = format_code(attrs["fmt_a"])
        fb = format_code(attrs["fmt_b"])
    except (KeyError, ValueError) as exc:
        return f"{OP_NAME}: {exc}"
    for key in ("opsel_a", "opsel_b"):
        sel = attrs.get(key, 0)
        if isinstance(sel, bool) or not isinstance(sel, int) or not 0 <= sel <= 3:
            return f"{OP_NAME}: {key} must be an int 0..3, got {sel!r}"
    want = [
        f"vec<i32x{OPERAND_WORDS[fa]}>",
        f"vec<i32x{OPERAND_WORDS[fb]}>",
        f"vec<f32x{sh.c_len}>",
        "i32",
        "i32",
    ]
    if list(types) != want:
        return f"{OP_NAME} {sh.name} fmt_a={fa} fmt_b={fb} expects operands {want}, got {list(types)}"
    return None


# ----------------------------------------------------------------- lane maps
#
# ``ar`` is any object with ``const_i32 / add / mul / div / mod`` (an IRBuilder
# for device index math, or a Python-int arithmetic for host tests), as in
# ``core/arch/target.py``.


def _k_of(ar, sh: MxShape, fmt: int, lane, s: int):
    grp = ar.div(lane, ar.const_i32(sh.m))
    if FORMAT_BITS[fmt] == 8:
        half = sh.k // 2  # the second 16-byte half: 64 K on, or 32 for 32x32x64
        base = ar.mul(grp, ar.const_i32(16))
        return ar.add(base, ar.const_i32((s // 16) * half + s % 16))
    return ar.add(ar.mul(grp, ar.const_i32(32)), ar.const_i32(s))


def a_coord(ar, shape_name: str, fmt, lane, s: int):
    """``(row, k)`` of A element ``s`` held by ``lane``."""
    sh = shape(shape_name)
    return ar.mod(lane, ar.const_i32(sh.m)), _k_of(ar, sh, format_code(fmt), lane, s)


def b_coord(ar, shape_name: str, fmt, lane, s: int):
    """``(k, col)`` of B element ``s`` held by ``lane``."""
    sh = shape(shape_name)
    return _k_of(ar, sh, format_code(fmt), lane, s), ar.mod(lane, ar.const_i32(sh.n))


def scale_a_coord(ar, shape_name: str, lane):
    """``(row, k_block)`` whose A scale ``lane`` supplies."""
    sh = shape(shape_name)
    return ar.mod(lane, ar.const_i32(sh.m)), ar.div(lane, ar.const_i32(sh.m))


def scale_b_coord(ar, shape_name: str, lane):
    """``(k_block, col)`` whose B scale ``lane`` supplies."""
    sh = shape(shape_name)
    return ar.div(lane, ar.const_i32(sh.n)), ar.mod(lane, ar.const_i32(sh.n))


def c_coord(ar, shape_name: str, lane, i: int):
    """``(row, col)`` of accumulator element ``i`` held by ``lane``."""
    sh = shape(shape_name)
    grp = ar.div(lane, ar.const_i32(sh.m))
    col = ar.mod(lane, ar.const_i32(sh.n))
    if sh.m == 16:
        return ar.add(ar.mul(grp, ar.const_i32(4)), ar.const_i32(i)), col
    row = ar.add(ar.mul(grp, ar.const_i32(4)), ar.const_i32((i // 4) * 8 + i % 4))
    return row, col


class HostArith:
    """Python-int arithmetic for evaluating the lane maps on the host."""

    @staticmethod
    def const_i32(x):
        return x

    @staticmethod
    def add(a, b):
        return a + b

    @staticmethod
    def mul(a, b):
        return a * b

    @staticmethod
    def div(a, b):
        return a // b

    @staticmethod
    def mod(a, b):
        return a % b


__all__ = [
    "FORMAT_BITS",
    "FORMAT_CODES",
    "OPERAND_WORDS",
    "OP_NAME",
    "SHAPES",
    "HostArith",
    "MxShape",
    "a_coord",
    "b_coord",
    "c_coord",
    "check_operands",
    "format_code",
    "scale_a_coord",
    "scale_b_coord",
    "shape",
]

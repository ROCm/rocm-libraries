# SPDX-License-Identifier: Apache-2.0
# Not vendored: written for this provider from the FlyDSL SDPA backward sweep (see README.md); it has no upstream.

"""``six``-style polyfill for FlyDSL helpers that are branch-local, not upstream.

The vendored gfx1201 kernel is written against
``xinyazhang/sdpa-gfx1201-feature``, and that branch adds six small functions
to ``kernels/common/`` that a stock ``flydsl`` does not yet have --
``git diff $(git merge-base HEAD upstream/main) HEAD -- kernels/common/`` is
+121 lines across ``kernels/common/utils.py`` and the new
``kernels/common/mma/wmma_ops.py``.

This module is **authored**, not vendored: each function below prefers the
stock package's own definition when one exists, and only falls back to a
local copy of the branch body otherwise. So this module empties itself out,
function by function, as each helper lands upstream -- it is a bridge, not a
permanent home. Each entry below names the branch file it came from and the
upstream merge that should delete it.

*Polyfill*, not *shim* or *compat*: this repository's ``shim`` already means
something specific (the generated C++ dispatch layer), so reusing the word
here would mislead.

**This module must import with no ``flydsl`` installed.** The generator's host
path reaches the kernel packages without flydsl, so every ``flydsl`` import
here is function-local or guarded by ``ImportError`` -- the same laziness
applied to torch in ``fmha_abi_gfx1201.py``. The two ``ir.Type`` annotations are
quoted for the same reason: an annotation on a ``def`` is evaluated when the
``def`` executes.
"""

__all__ = [
    "ssel",
    "smin",
    "smax",
    "sdiv_rd_pow2",
    "wmma_f32_16x16x16",
    "vector_elem_type",
]


def _is_pow2(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _pow2_shift(value: int) -> int:
    assert _is_pow2(value)
    return value.bit_length() - 1


try:
    from flydsl.kernels.common.utils import ssel
except ImportError:

    def ssel(pred, a, b):
        """``pred ? a : b`` as an ``fx.Int32``.

        ``pred`` may be an ``fx.Boolean`` (what a comparison returns), a raw
        i1 ``ir.Value``, or an ``ArithValue`` -- ``fx.Boolean`` accepts all
        three.

        Operands are *not* coerced. Pass i32; passing an ``fx.Index`` gets
        you a 64-bit unsigned select, which is a silent bug wherever the
        value can be negative.
        """
        import flydsl.expr as fx

        return fx.Int32(fx.Boolean(pred).select(a, b))


try:
    from flydsl.kernels.common.utils import smin
except ImportError:

    def smin(a, b):
        """Signed minimum of two i32 values. Same operand contract as `ssel`."""
        return ssel((a < b), a, b)


try:
    from flydsl.kernels.common.utils import smax
except ImportError:

    def smax(a, b):
        """Signed maximum of two i32 values. See `smin`."""
        return ssel((a > b), a, b)


try:
    from flydsl.kernels.common.utils import sdiv_rd_pow2
except ImportError:

    def sdiv_rd_pow2(value, divisor: int):
        """``floor(value / divisor)`` for a *signed* i32 and a power-of-two divisor.

        The signed counterpart to ``udiv_pow2``: an arithmetic right shift
        rounds toward negative infinity, which is what a floor division
        needs and what ``arith.divsi``'s truncation does not give on
        negative input.
        """
        import flydsl.expr as fx

        assert _is_pow2(
            divisor
        ), f"sdiv_rd_pow2 needs a power-of-two divisor, got {divisor}"
        return fx.Int32(value) >> fx.Int32(_pow2_shift(divisor))


try:
    from flydsl.kernels.common.mma.wmma_ops import vector_elem_type
except ImportError:

    def vector_elem_type(value) -> "ir.Type":
        """Element type of a vector-typed value, raw or DSL-wrapped.

        ``ir.VectorType.isinstance`` does not exist in this binding; Python
        ``isinstance`` against the downcast class is what works.
        """
        from flydsl._mlir import ir
        from flydsl.expr.utils.arith import _to_raw as _raw

        ty = _raw(value).type
        if not isinstance(ty, ir.VectorType):
            raise TypeError(f"expected a vector value, got {ty}")
        return ir.VectorType(ty).element_type


try:
    from flydsl.kernels.common.mma.wmma_ops import wmma_f32_16x16x16
except ImportError:

    def wmma_f32_16x16x16(a, b, acc, acc_type: "ir.Type | None" = None):
        """One 16x16x16 WMMA into an f32 accumulator: ``acc += a @ b``.

        ``a`` and ``b`` are 8-lane f16 or bf16 vectors and must agree;
        ``acc`` is an 8-lane f32 vector. ``acc_type`` defaults to ``acc``'s
        own type and only needs passing when ``acc`` is a raw value whose
        type the caller wants to override.

        bf16 operands are bitcast to ``i16`` because that is the ABI the
        intrinsic takes -- a reinterpretation, not a conversion.
        """
        import flydsl.expr as fx
        from flydsl._mlir import ir
        from flydsl.expr import rocdl
        from flydsl.expr.typing import Vector
        from flydsl.expr.utils.arith import _to_raw as _raw

        a_ty, b_ty = vector_elem_type(a), vector_elem_type(b)
        if a_ty != b_ty:
            raise TypeError(
                f"WMMA operands must share an element type, got {a_ty} and {b_ty}"
            )
        res_ty = acc_type if acc_type is not None else _raw(acc).type

        if isinstance(a_ty, ir.BF16Type):
            a16 = _raw(Vector(_raw(a)).bitcast(fx.Int16))
            b16 = _raw(Vector(_raw(b)).bitcast(fx.Int16))
            return rocdl.wmma_f32_16x16x16_bf16(res_ty, a16, b16, _raw(acc)).result
        if isinstance(a_ty, ir.F16Type):
            return rocdl.wmma_f32_16x16x16_f16(
                res_ty, _raw(a), _raw(b), _raw(acc)
            ).result
        raise TypeError(f"wmma_f32_16x16x16 supports f16 and bf16 operands, got {a_ty}")


# `_if_then(if_op)`: a context manager over an scf.IfOp's then-block, which
# fmha_common_gfx1201 uses to enter `scf.if` bodies. The FlyDSL tree this
# provider pins (0.3.4) has no `kernels_common._if_then`, so it lives here.
import contextlib as _contextlib  # noqa: E402


@_contextlib.contextmanager
def _if_then(if_op):
    from flydsl._mlir import ir as _ir
    from flydsl._mlir.dialects import scf as _scf

    with _ir.InsertionPoint(if_op.then_block):
        yield if_op.then_block
        _scf.YieldOp([])

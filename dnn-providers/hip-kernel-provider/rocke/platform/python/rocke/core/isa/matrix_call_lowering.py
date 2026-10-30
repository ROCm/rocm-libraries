# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Emit one MMA instruction from its :class:`~rocke.core.isa.matrix_calls.MatrixCall`.

Each operand goes through its form, the immediates are printed as written, the
call is emitted, and the result goes through its form. The backends call
:func:`emit_matrix_call` for every op_id in
:data:`~rocke.core.isa.matrix_calls.MATRIX_CALLS`. The C++ engine's twin is
``rocke_ll_emit_matrix_call`` in ``cpp/core/lower_llvm/matrix_call.cpp``.
"""

from __future__ import annotations

from ..ir import Op
from .matrix_calls import Imm, MatrixCall


def emit_matrix_call(lowerer, op: Op, call: MatrixCall) -> None:
    """Lower MMA op ``op`` (named ``tile.<op_id>``, operands A, B, C) through ``call``.

    The call's declaration is the one the lowerer's table holds under
    ``call.key``. A call outside :data:`MATRIX_CALLS` (one a test builds) is not
    in the table and registers its derived declaration the way the other
    dynamically declared intrinsics do; a table entry whose text differs from
    the derived one is refused, so the call and its ``declare`` line cannot
    disagree.
    """
    from ..lower_llvm import _llvm_type, _mma_op_id

    op_id = _mma_op_id(op)
    if len(op.operands) != 3:
        raise ValueError(f"{op.name} expects 3 operands, got {len(op.operands)}")
    if len(op.results) != 1:
        raise ValueError(
            f"{op.name}: expected exactly one result, got {len(op.results)}"
        )
    flavor = lowerer._flavor
    derived = call.declaration(flavor)
    declared = lowerer._decls.setdefault(call.key, derived)
    if declared != derived:
        raise ValueError(
            f"matrix call for {op_id!r} declares {call.key!r} as {derived!r}, but the "
            f"declaration table holds {declared!r}"
        )
    lowerer._need(call.key)
    arguments = []
    for arg in call.args:
        if isinstance(arg, Imm):
            arguments.append(arg.argument())
            continue
        form = call.form(arg)
        value = form.operand(lowerer, op, arg.value)
        arguments.append(f"{_llvm_type(form.register_type_at(flavor))} {value}")
    returned = _llvm_type(call.result.register_type_at(flavor))
    call.result.emit_result(
        lowerer, op, f"call {returned} @{call.intrinsic}({', '.join(arguments)})"
    )

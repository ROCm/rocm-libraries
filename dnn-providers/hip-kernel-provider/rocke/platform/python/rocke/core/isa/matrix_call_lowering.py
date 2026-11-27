# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Emit one MMA instruction from its :class:`~rocke.core.isa.matrix_calls.MatrixCall`.

Each operand goes through its form, the immediates are printed as written, each
signedness flag is derived from the target's catalog row, the call is emitted,
and the result goes through its form. The backends call
:func:`emit_matrix_call` for every op_id in
:data:`~rocke.core.isa.matrix_calls.MATRIX_CALLS`. The C++ engine's twin is
``rocke_ll_emit_matrix_call`` in ``cpp/core/lower_llvm/matrix_call.cpp``.
"""

from __future__ import annotations

from typing import NoReturn

from ..ir import Op
from ..llvm_profile import LlvmProfileRefusal
from .matrix_calls import Imm, MatrixCall, Signedness


def llvm23_matrix_op_refusal(op_id: str, flavor: str) -> str:
    """The refusal of the MMA op ``tile.<op_id>`` at ``flavor``, which lacks the
    LLVM 23 target-op profile its intrinsic needs."""
    return f"tile.{op_id} requires llvm23 (ROCm 7.13+), got {flavor}"


def refuse_llvm23_matrix_op(op_id: str, flavor: str) -> NoReturn:
    """Refuse the MMA op ``tile.<op_id>`` at ``flavor``, which lacks the LLVM 23
    target-op profile its intrinsic needs (:class:`LlvmProfileRefusal`,
    :func:`llvm23_matrix_op_refusal`). The C++ engine's twin is
    ``rocke_ll_refuse_llvm23_matrix_op``."""
    raise LlvmProfileRefusal(llvm23_matrix_op_refusal(op_id, flavor))


def emit_matrix_call(lowerer, op: Op, call: MatrixCall) -> None:
    """Lower MMA op ``op`` (named ``tile.<op_id>``, operands A, B, C) through ``call``.

    The call's declaration is the one the lowerer's table holds under
    ``call.key``. A call outside :data:`MATRIX_CALLS` (one a test builds) is not
    in the table and registers its derived declaration the way the other
    dynamically declared intrinsics do; a table entry whose text differs from
    the derived one is refused, so the call and its ``declare`` line cannot
    disagree. A call that does not lower at the lowerer's flavor
    (:meth:`MatrixCall.lowers_at`) is refused before anything else
    (:func:`refuse_llvm23_matrix_op`).
    """
    from ..lower_llvm import _llvm_type, _mma_op_id

    op_id = _mma_op_id(op)
    if not call.lowers_at(lowerer._flavor):
        refuse_llvm23_matrix_op(op_id, lowerer._flavor)
    if len(op.operands) != 3:
        raise ValueError(f"{op.name} expects 3 operands, got {len(op.operands)}")
    if len(op.results) != 1:
        raise ValueError(
            f"{op.name}: expected exactly one result, got {len(op.results)}"
        )
    flavor = lowerer._flavor
    derived = call.declaration(flavor)
    declared = lowerer._decls.get(call.key, derived)
    if declared != derived:
        raise ValueError(
            f"matrix call for {op_id!r} declares {call.key!r} as {derived!r}, but the "
            f"declaration table holds {declared!r}"
        )
    # Check every value before declarations, bitcasts, padding or gathers can
    # change the lowerer. Rendering reuses these same form validation rules.
    for arg in call.args:
        if isinstance(arg, Imm):
            arg.argument()
        elif isinstance(arg, Signedness):
            arg.argument(_catalog_op(lowerer, op_id))
        else:
            call.form(arg).check_operand(lowerer, op, arg.value)
    call.result.check_result(lowerer, op)
    lowerer._decls.setdefault(call.key, derived)
    lowerer._need(call.key)
    arguments = []
    for arg in call.args:
        if isinstance(arg, Imm):
            arguments.append(arg.argument())
            continue
        if isinstance(arg, Signedness):
            arguments.append(arg.argument(_catalog_op(lowerer, op_id)))
            continue
        form = call.form(arg)
        value = form.operand(lowerer, op, arg.value)
        arguments.append(f"{_llvm_type(form.register_type_at(flavor))} {value}")
    returned = _llvm_type(call.result.register_type_at(flavor))
    call.result.emit_result(
        lowerer, op, f"call {returned} @{call.intrinsic}({', '.join(arguments)})"
    )


def _catalog_op(lowerer, op_id: str):
    """The lowering target's catalog row for ``op_id``, whose operand dtypes the
    signedness flags follow; refused, as by the backend, where the catalog does
    not list it."""
    backend = lowerer._backend
    backend.require_catalog_lists(op_id)
    return backend.arch.mma.by_op_id(op_id)

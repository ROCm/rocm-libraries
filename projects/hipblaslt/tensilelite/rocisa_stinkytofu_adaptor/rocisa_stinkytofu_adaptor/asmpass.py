################################################################################
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################
"""Shim for ``rocisa.asmpass``.

Python port of ``rocisa/rocisa/include/pass.hpp`` + ``rocisa/src/pass/pass.cpp``.
All passes mutate the adaptor ``KernelBody`` / ``Module`` tree in place (no
``_stinkytofu`` dependency).

Implemented:
    - ``rocIsaPassOption`` / ``rocIsaPassResult`` / ``rocIsaPass``
    - ``macroToInstruction``, ``compositeToInstruction`` (no-op until
      ``CompositeInstruction`` exposes ``getInstructions``),
      ``convertTextVariablesToRegisters``
    - ``getActFuncModuleName`` / ``getActFuncBranchModuleName``

Deferred (no-op or stub):
    - ``removeDuplicatedFunction`` (activation de-dup)
    - ``buildGraph`` / ``removeDuplicateAssignment`` when ``doOpt()`` is true
    - ``insertDelayAlu``
    - ``getCycles`` returns ``0`` (matches native gfx1250 unsupported path)
"""

from __future__ import annotations

from typing import Any

from . import code as _code
from ._pass_impl import (
    build_graph_and_remove_dup_assign,
    composite_to_instruction,
    convert_text_variables_to_registers,
    get_act_func_branch_module_name,
    get_act_func_module_name,
    get_cycles,
    insert_delay_alu,
    macro_to_instruction,
    remove_duplicated_function,
)


class rocIsaPassOption:
    """Mirror ``rocisa::rocIsaPassOption`` (``pass.hpp``)."""

    __slots__ = (
        "insertDelayAlu",
        "removeDupFunc",
        "removeDupAssign",
        "getCycles",
        "numWaves",
    )

    def __init__(self) -> None:
        self.insertDelayAlu: bool = False
        self.removeDupFunc: bool = True
        self.removeDupAssign: bool = True
        self.getCycles: bool = True
        self.numWaves: int = 0

    def doOpt(self) -> bool:
        return self.removeDupAssign


class rocIsaPassResult:
    """Mirror ``rocisa::rocIsaPassResult`` (``pass.hpp``)."""

    __slots__ = ("cycles",)

    def __init__(self) -> None:
        self.cycles: int = -1


def getActFuncModuleName(gwvw: int, sgpr: int, tmpVgpr: int, tmpSgpr: int) -> str:
    return get_act_func_module_name(gwvw, sgpr, tmpVgpr, tmpSgpr)


def getActFuncBranchModuleName() -> str:
    return get_act_func_branch_module_name()


def rocIsaPass(kernel: Any, option: rocIsaPassOption) -> rocIsaPassResult:
    """Mirror ``rocisa::rocIsaPass`` (``pass.cpp``)."""
    body = kernel.body
    if body is None:
        raise RuntimeError("Kernel body is empty")

    result = rocIsaPassResult()

    if option.removeDupFunc:
        remove_duplicated_function(body)

    macro_to_instruction(body)
    composite_to_instruction(body)
    convert_text_variables_to_registers(body)

    if option.doOpt():
        build_graph_and_remove_dup_assign(
            body, int(kernel.totalVgprs), int(kernel.totalSgprs)
        )

    if option.insertDelayAlu:
        insert_delay_alu(body)

    if option.getCycles:
        result.cycles = get_cycles(body, int(option.numWaves))

    return result

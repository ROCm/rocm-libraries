# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Python port of rocisa asm optimization passes.

Implemented: macroToInstruction, compositeToInstruction, convertTextVariablesToRegisters.
Not yet done: removeDuplicatedFunction, buildGraph, insertDelayAlu, getCycles (returns 0).
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

    __slots__ = ("cycles", "maxVgpr")

    def __init__(self) -> None:
        self.cycles: int = -1
        self.maxVgpr: int = -1


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
        max_vgpr_seen = build_graph_and_remove_dup_assign(
            body, int(kernel.totalVgprs), int(kernel.totalSgprs)
        )
        result.maxVgpr = (max_vgpr_seen + 1) if max_vgpr_seen >= 0 else int(kernel.totalVgprs)
    else:
        result.maxVgpr = int(kernel.totalVgprs)

    if option.insertDelayAlu:
        insert_delay_alu(body)

    if option.getCycles:
        result.cycles = get_cycles(body, int(option.numWaves))

    return result

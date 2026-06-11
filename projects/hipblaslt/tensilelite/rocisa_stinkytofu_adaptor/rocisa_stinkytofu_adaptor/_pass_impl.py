################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
################################################################################
"""Internal helpers for ``rocisa.asmpass`` — Python port of ``pass.cpp``.

Mirrors the ordering and option gates in ``rocisa/rocisa/src/pass/pass.cpp``.
All passes operate on the in-memory ``code.Module`` / ``code.Macro`` tree
(rocisa-shaped IR), not on stinkytofu ``LogicalModule`` objects.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, MutableSequence, Tuple

from . import code as _code
from . import instruction as _inst
from .container import RegisterContainer

# ---------------------------------------------------------------------------
# getActFunc* — string helpers (remove.cpp:177-198)
# ---------------------------------------------------------------------------


def get_act_func_module_name(gwvw: int, sgpr: int, tmp_vgpr: int, tmp_sgpr: int) -> str:
    return (
        f"ActFunc_VW{int(gwvw)}_Sgpr{int(sgpr)}_Tmp{int(tmp_vgpr)}_{int(tmp_sgpr)}"
    )


def get_act_func_branch_module_name() -> str:
    return "InsertActFuncCallAddrCalc"


# ---------------------------------------------------------------------------
# removeDuplicatedFunction — activation de-dup (remove.cpp)
# ---------------------------------------------------------------------------


def remove_duplicated_function(module: _code.Module) -> None:
    """Mirror ``rocisa::removeDuplicatedFunction`` (activation epilogues).

    Full parity requires the ActFunc_VW* module discovery logic from
    ``remove.cpp``. Not exercised by tensilelite adaptor tests today;
    left as a no-op until activation-driven codegen is brought up.
    """
    del module  # silence unused; tree intentionally untouched for now


# ---------------------------------------------------------------------------
# compositeToInstruction — composite.cpp
# ---------------------------------------------------------------------------


def composite_to_instruction(module: _code.Module) -> None:
    """Mirror ``rocisa::compositeToInstruction``.

    Flattens ``CompositeInstruction`` children into plain instructions.
    The adaptor still exposes ``CompositeInstruction`` as a dummy type
    without ``getInstructions()`` — until a real subclass lands this is
    effectively a deep structural no-op aside from descending into nested
    ``Module`` / ``Macro`` containers (matches the C++ recursion shape).
    """

    def walk(container: Any) -> None:
        if isinstance(container, _code.Module):
            _flatten_module_list(container.itemList)
        elif isinstance(container, _code.Macro):
            _flatten_module_list(container.itemList)

    def _flatten_module_list(items: MutableSequence[Any]) -> None:
        pos = 0
        while pos < len(items):
            it = items[pos]
            if isinstance(it, _code.Module):
                walk(it)
                pos += 1
            elif isinstance(it, _code.Macro):
                walk(it)
                pos += 1
            else:
                geti = getattr(it, "getInstructions", None)
                if callable(geti):
                    expanded = geti()
                    if isinstance(expanded, (list, tuple)) and expanded:
                        items[pos : pos + 1] = list(expanded)
                        pos += len(expanded)
                        continue
                pos += 1

    walk(module)


# ---------------------------------------------------------------------------
# convertTextVariablesToRegisters — graph.cpp
# ---------------------------------------------------------------------------


def _set_name_to_reg_num(
    gpr: RegisterContainer, assignment: Dict[str, int]
) -> None:
    if gpr.regIdx != -1 or gpr.regName is None:
        return
    key = gpr.getRegNameWithType()
    base = assignment[key]
    off = gpr.regName.getTotalOffsets()
    gpr.regIdx = base + off


def _convert_text_variables_impl(
    module: _code.Module, assignment: Dict[str, int]
) -> None:
    for item in module.itemList:
        if isinstance(item, _code.Module):
            _convert_text_variables_impl(item, assignment)
        elif isinstance(item, _inst.Instruction):
            get_params = getattr(item, "getParams", None)
            if not callable(get_params):
                continue
            try:
                params = get_params()
            except (NotImplementedError, RuntimeError):
                continue
            for p in params:
                if isinstance(p, RegisterContainer):
                    _set_name_to_reg_num(p, assignment)
        elif isinstance(item, _code.RegSet):
            if item.ref is not None:
                num = int(assignment[item.ref]) + int(item.offset)
            else:
                assert item.value is not None
                num = int(item.value) + int(item.offset)
            assignment[item.name] = num


def convert_text_variables_to_registers(module: _code.Module) -> None:
    """Mirror ``rocisa::convertTextVariablesToRegisters``."""
    _convert_text_variables_impl(module, {})


# ---------------------------------------------------------------------------
# insertDelayAlu — insert_delay_alu.cpp
# ---------------------------------------------------------------------------


def insert_delay_alu(module: _code.Module) -> None:
    """Mirror ``rocisa::insertDelayAlu``. Deferred — gfx1250 scheduling."""
    del module


# ---------------------------------------------------------------------------
# getCycles — cycle.cpp (gfx1250 unsupported branch returns 0)
# ---------------------------------------------------------------------------


def get_cycles(module: _code.Module, num_waves: int) -> int:
    """Mirror ``rocisa::getCycles`` well enough for KernelWriter metadata.

    Native ``cycle.cpp`` returns ``0`` for ISAs outside the origami
    formocast table (includes gfx1250 today). We keep the same contract
    without pulling the full C++ analyzer into Python.
    """
    del module, num_waves
    return 0


# ---------------------------------------------------------------------------
# macroToInstruction — macro_inline.cpp
# ---------------------------------------------------------------------------

_MACRO_ARG_RE = re.compile(r"([^=:]+)(?::req)?=?(\w+)?")


def _extract_macro(
    table: Dict[str, Tuple[_code.Macro, List[Tuple[str, str]]]], macro: _code.Macro
) -> None:
    params: List[Tuple[str, str]] = []
    for arg in macro.macro.args:
        arg_str = arg if isinstance(arg, str) else str(arg)
        m = _MACRO_ARG_RE.match(arg_str)
        if not m:
            params.append((arg_str, ""))
            continue
        name = m.group(1)
        default = m.group(2) or ""
        params.append((name, default))
    table[macro.name] = (macro, params)


def _collect_macros(
    module: _code.Module, table: Dict[str, Tuple[_code.Macro, List[Tuple[str, str]]]]
) -> None:
    for it in module.itemList:
        if isinstance(it, _code.Module):
            _collect_macros(it, table)
        elif isinstance(it, _code.Macro):
            if it.name not in table:
                _extract_macro(table, it)


def _replace_whole_word(hay: str, needle: str, repl: str) -> str:
    out: List[str] = []
    pos = 0
    nlen = len(needle)
    while pos < len(hay):
        idx = hay.find(needle, pos)
        if idx == -1:
            out.append(hay[pos:])
            break
        end = idx + nlen
        before_ok = idx == 0 or not (hay[idx - 1].isalnum() or hay[idx - 1] == "_")
        after_ok = end >= len(hay) or not (hay[end].isalnum() or hay[end] == "_")
        if before_ok and after_ok:
            out.append(hay[pos:idx])
            out.append(repl)
            pos = end
        else:
            out.append(hay[pos:end])
            pos = end
    return "".join(out)


def _substitute_string_param(s: str, params: List[Tuple[str, str]]) -> str:
    """Replace ``\\param`` tokens with the current actual argument strings."""
    result = s
    for name, value in params:
        needle = "\\" + name
        result = _replace_whole_word(result, needle, value)
    return result


def _eval_macro_condition(value: str, params: List[Tuple[str, str]]) -> bool:
    """Port of ``evalMacroCondition`` in ``macro_inline.cpp`` (``.if`` / ``.elseif``)."""
    token_re = re.compile(r"\\([^=\s]+)|\w+|==|!=|&&")
    lhs = op = rhs = val = ""
    token_idx = 0
    results: List[int] = []
    for m in token_re.finditer(value):
        val = m.group(0)
        if val.startswith("\\"):
            var = m.group(1)
            pval = ""
            for pname, pv in params:
                if pname == var:
                    pval = pv
                    break
            assert any(pn == var for pn, _ in params), (
                "unknown macro argument in condition: " + repr(var)
            )
            val = pval
        mod4 = token_idx % 4
        if mod4 == 0:
            lhs = val
        elif mod4 == 1:
            op = val
        elif mod4 == 2:
            rhs = val
            if op == "==":
                results.append(1 if lhs == rhs else 0)
            elif op == "!=":
                results.append(1 if lhs != rhs else 0)
            else:
                raise AssertionError(
                    "unknown macro condition operator: " + repr(op)
                )
        elif mod4 == 3:
            assert val == "&&", "unknown macro logical operator: " + repr(val)
            results.append(2)
        token_idx += 1
    if not results:
        return True
    result = bool(results[0])
    i = 1
    while i < len(results):
        if results[i] == 2 and i + 1 < len(results):
            result = result and bool(results[i + 1])
            i += 2
        else:
            i += 1
    return result


def _substitute_register_param(reg: RegisterContainer, params: List[Tuple[str, str]]) -> None:
    if not reg.isMacro or reg.regName is None:
        return
    name_str = reg.regName.name
    for param_name, param_value in params:
        stripped = (
            param_name[4:]
            if len(param_name) > 4
            and param_name[:4] in ("vgpr", "sgpr", "mgpr")
            else param_name
        )
        name_str = _replace_whole_word(name_str, stripped, param_value)
    reg.isMacro = False
    full_prefix = reg.regType + "gpr"
    if len(name_str) > len(full_prefix) and name_str.startswith(full_prefix):
        name_str = name_str[len(full_prefix) :]
    elif len(name_str) > len(reg.regType) and name_str.startswith(reg.regType):
        name_str = name_str[len(reg.regType) :]
    plus = name_str.find("+")
    try:
        if plus != -1:
            reg.regIdx = int(name_str[:plus]) + int(name_str[plus + 1 :])
            reg.regName = None
        else:
            reg.regIdx = int(name_str)
            reg.regName = None
    except ValueError:
        from .container import RegName  # local import

        reg.regName = RegName(name_str)
        reg.regIdx = -1


def _substitute_input(arg: Any, params: List[Tuple[str, str]]) -> Any:
    if isinstance(arg, str):
        return _substitute_string_param(arg, params)
    if isinstance(arg, RegisterContainer):
        cloned = copy.deepcopy(arg, {})
        _substitute_register_param(cloned, params)
        return cloned
    return arg


def _substitute_common_inst(inst: _inst.CommonInstruction, params: List[Tuple[str, str]]) -> None:
    if inst.dst is not None and isinstance(inst.dst, RegisterContainer):
        _substitute_register_param(inst.dst, params)
    if inst.dst1 is not None and isinstance(inst.dst1, RegisterContainer):
        _substitute_register_param(inst.dst1, params)
    inst.srcs = [_substitute_input(s, params) for s in inst.srcs]
    inst.comment = _substitute_string_param(inst.comment, params)


def _clone_instruction(inst: _inst.Instruction) -> _inst.Instruction:
    if isinstance(inst, _inst.MacroInstruction):
        return inst.clone()
    if isinstance(inst, _inst.CommonInstruction):
        return copy.deepcopy(inst, {})
    raise TypeError(
        f"macroToInstruction: cannot clone instruction type {type(inst).__name__!r}"
    )


def _clone_and_substitute(
    inst: _inst.Instruction, params: List[Tuple[str, str]]
) -> _inst.Instruction:
    cloned = _clone_instruction(inst)
    if isinstance(cloned, _inst.CommonInstruction):
        _substitute_common_inst(cloned, params)
    return cloned


def _expand_macro_body(
    output: List[Any],
    macro_items: List[Any],
    params: List[Tuple[str, str]],
    branch: List[bool],
) -> None:
    for item in macro_items:
        if isinstance(item, _code.Module):
            _expand_macro_body(output, item.itemList, params, branch)
        elif isinstance(item, _code.ValueIf):
            branch.append(branch[-1] and _eval_macro_condition(item.value, params))
        elif isinstance(item, _code.ValueElseIf):
            if_taken = branch.pop()
            branch.append(
                (not if_taken) and _eval_macro_condition(item.value, params)
            )
        elif isinstance(item, _code.ValueEndif):
            branch.pop()
        elif isinstance(item, _inst.Instruction):
            if branch[-1]:
                output.append(_clone_and_substitute(item, params))
        else:
            raise AssertionError(
                "macroToInstruction: unexpected item type in macro body: "
                + type(item).__name__
            )


def _macro_to_instruction_impl(
    container: Any, table: Dict[str, Tuple[_code.Macro, List[Tuple[str, str]]]]
) -> None:
    items = container.itemList
    new_items: List[Any] = []
    for item in items:
        if isinstance(item, _code.Module):
            _macro_to_instruction_impl(item, table)
            new_items.append(item)
        elif isinstance(item, _code.Macro):
            continue
        elif isinstance(item, _inst.MacroInstruction):
            ent = table.get(item.name)
            if ent is None:
                raise AssertionError(
                    "macroToInstruction: MacroInstruction references undefined macro "
                    + repr(item.name)
                )
            macro_def, defaults = ent
            param_list: List[Tuple[str, str]] = [(n, v) for n, v in defaults]
            for i, arg in enumerate(item.args):
                if i >= len(param_list):
                    break
                name, _ = param_list[i]
                param_list[i] = (name, _inst._input_to_str(arg))
            branch = [True]
            _expand_macro_body(new_items, macro_def.itemList, param_list, branch)
        else:
            new_items.append(item)
    container.itemList = new_items


def macro_to_instruction(module: _code.Module) -> None:
    """Mirror ``rocisa::macroToInstruction``."""
    table: Dict[str, Tuple[_code.Macro, List[Tuple[str, str]]]] = {}
    _collect_macros(module, table)
    if not table:
        return
    _macro_to_instruction_impl(module, table)


# ---------------------------------------------------------------------------
# Graph optimisation — deferred (pass.cpp gates on doOpt/removeDupAssign)
# ---------------------------------------------------------------------------


def build_graph_and_remove_dup_assign(
    module: _code.Module, max_vgpr: int, max_sgpr: int
) -> None:
    """Placeholder for ``buildGraph`` + ``removeDuplicateAssignment``."""
    del module, max_vgpr, max_sgpr

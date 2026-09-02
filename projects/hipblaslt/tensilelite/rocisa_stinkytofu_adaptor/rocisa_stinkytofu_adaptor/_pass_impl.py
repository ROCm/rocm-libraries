# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Python port of rocisa asm-pass pipeline (pass.cpp).

Operates on the in-memory Module/Macro tree, not on stinkytofu LogicalModule.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Tuple

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
    """Mirror ``rocisa::removeDuplicatedFunction`` (remove.cpp).

    Finds duplicate ``ActFunc_VW*`` modules, keeps only the first instance,
    removes the rest, and redirects branch label references to the surviving
    copy.  The surviving functions are appended at the module tail followed
    by an ``SEndpgm``.
    """
    _remove_duplicated_activation_functions(module)


def _find_act_func(
    module: _code.Module,
) -> Dict[str, List[_code.Module]]:
    """Recursively find all modules whose name contains ``ActFunc_VW``."""
    mod_func: Dict[str, List[_code.Module]] = {}
    for item in module.items():
        if isinstance(item, _code.Module):
            if "ActFunc_VW" in item.name:
                mod_func.setdefault(item.name, []).append(item)
            else:
                sub = _find_act_func(item)
                for key, mlist in sub.items():
                    mod_func.setdefault(key, []).extend(mlist)
    return mod_func


def _replace_act_branch_label(
    module: _code.Module, labels: List[str]
) -> None:
    """Redirect ``InsertActFuncCallAddrCalc`` branch offsets to the kept label."""
    label_first = labels[0]
    num_underscores = label_first.count("_")
    part_first = label_first.rfind("_")
    last_postfix = label_first[part_first + 1:]
    label_rest = labels[1:]

    for item in module.items():
        if not isinstance(item, _code.Module):
            continue
        if "InsertActFuncCallAddrCalc" in item.name:
            replace_label = False
            for inst in item.items():
                if (isinstance(inst, _inst.CommonInstruction)
                        and getattr(inst, "comment", "") == "target branch offset"):
                    src0 = inst.srcs[0] if inst.srcs else None
                    if isinstance(src0, str) and src0 in label_rest:
                        replace_label = True
                        break
            if replace_label:
                for inst in item.items():
                    if (isinstance(inst, _inst.CommonInstruction)
                            and getattr(inst, "comment", "") == "target branch offset"):
                        src0 = inst.srcs[0] if inst.srcs else None
                        if not isinstance(src0, str):
                            continue
                        num_us = src0.count("_")
                        if num_underscores == num_us:
                            part = src0.rfind("_")
                            inst.srcs[0] = src0[:part] + "_" + last_postfix
                        elif num_underscores == num_us - 1:
                            part = src0.rfind("_")
                            inst.srcs[0] = src0[:part]
                        else:
                            raise RuntimeError("Incorrect Activation Label")
        else:
            _replace_act_branch_label(item, labels)


def _remove_duplicated_activation_functions(module: _code.Module) -> None:
    """Core logic of ``removeDuplicatedFunction`` (remove.cpp)."""
    mod_func = _find_act_func(module)
    module_last = _code.Module("AddToLast")

    for _key, mlist in mod_func.items():
        if len(mlist) <= 1:
            continue
        labels: List[str] = []
        for ml in mlist:
            if isinstance(ml, _code.Module) and ml.items():
                mod2 = ml.items()[0]
                if isinstance(mod2, _code.Module) and mod2.items():
                    label = mod2.items()[0]
                    if isinstance(label, _code.Label):
                        labels.append(label.getLabelName())
            parent = getattr(ml, "parent", None)
            if parent is not None and isinstance(parent, _code.Module):
                parent.removeItem(ml)

        module_last.add(mlist[0])
        _replace_act_branch_label(module, labels)

    if module_last.items():
        module.add(module_last)
        module.add(_inst.SEndpgm())


# ---------------------------------------------------------------------------
# compositeToInstruction — composite.cpp
# ---------------------------------------------------------------------------


def composite_to_instruction(module: _code.Module) -> None:
    """Mirror ``rocisa::compositeToInstruction`` (composite.cpp).

    Flattens ``CompositeInstruction`` children into plain instructions by
    calling ``getInstructions()`` and splicing the results into the parent
    itemList.  Recurses into nested ``Module`` / ``Macro`` containers.
    """
    _composite_to_instruction_impl(module)


def _composite_to_instruction_impl(container: Any) -> None:
    """Mirrors ``compositeToInstructionTemplate<T>`` in composite.cpp."""
    new_items: List[Any] = []
    for item in container.itemList:
        geti = getattr(item, "getInstructions", None)
        if callable(geti):
            expanded = geti()
            if isinstance(expanded, (list, tuple)):
                new_items.extend(expanded)
                continue
        if isinstance(item, _code.Module):
            _composite_to_instruction_impl(item)
        elif isinstance(item, _code.Macro):
            _composite_to_instruction_impl(item)
        new_items.append(item)
    container.itemList = new_items


# ---------------------------------------------------------------------------
# convertTextVariablesToRegisters — graph.cpp
# ---------------------------------------------------------------------------


def _set_name_to_reg_num(
    gpr: RegisterContainer, assignment: Dict[str, int]
) -> None:
    if gpr.regIdx != -1 or gpr.regName is None:
        return
    key = gpr.getRegNameWithType()
    base = assignment.get(key, 0)
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
                # C++ unordered_map::operator[] returns 0 for missing keys;
                # match that lenient behaviour so forward-refs don't crash.
                num = int(assignment.get(item.ref, 0)) + int(item.offset)
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

# DelayALU type constants (mirrors rocisa::DelayALUType)
_DELAY_VALU = 0
_DELAY_TRANS = 1
_DELAY_SALU = 2
_DELAY_OTHER = 3

_ALU_DEP_MAX = {
    _DELAY_VALU: 4,
    _DELAY_SALU: 1,
    _DELAY_TRANS: 3,
    _DELAY_OTHER: 0,
}

_DELAY_TYPE_NAME = {
    _DELAY_VALU: "VALU",
    _DELAY_TRANS: "TRANS",
    _DELAY_SALU: "SALU",
}

_SKIP_MAX = 5


def _delay_alu_type(inst: _inst.Instruction) -> int:
    pre = inst.preStr()
    if pre.startswith("v_s_"):
        return _DELAY_TRANS
    if pre.startswith("v_"):
        return _DELAY_VALU
    if pre.startswith("s_"):
        return _DELAY_SALU
    return _DELAY_OTHER


def _get_dst_src_regs(inst: _inst.Instruction):
    """Return (set_of_dst_RegisterContainers, set_of_src_RegisterContainers).

    Mirrors native _getDstSrcRegs which only processes CommonInstruction
    and CompositeInstruction. Memory ops (SMEM, Buffer, Flat, DS, Global),
    MFMA, branches, etc. return empty sets.
    """
    if not isinstance(inst, _inst.CommonInstruction):
        return set(), set()
    pre = inst.preStr()
    if pre.startswith(("buffer_", "flat_", "ds_", "global_", "scratch_")):
        return set(), set()

    dsts: set = set()
    srcs: set = set()
    try:
        for p in inst.getDstParams():
            if isinstance(p, RegisterContainer):
                dsts.add(p)
    except (NotImplementedError, AttributeError):
        pass
    try:
        for p in inst.getSrcParams():
            if isinstance(p, RegisterContainer):
                srcs.add(p)
    except (NotImplementedError, AttributeError):
        pass
    return dsts, srcs


def _format_dep_str(alu_type: int, cnt: int) -> str:
    if cnt == 0:
        return "NO_DEP"
    name = _DELAY_TYPE_NAME.get(alu_type)
    if name is None:
        return ""
    if alu_type == _DELAY_SALU:
        return f"SALU_CYCLE_{cnt}"
    return f"{name}_DEP_{cnt}"


def _make_delay_alu_inst(instid0type: int, instid0cnt: int) -> _inst.Instruction:
    """Create an SDelayAlu instruction with the proper formatted immediate."""
    dep_str = _format_dep_str(instid0type, instid0cnt)
    from .instruction import SDelayAlu as _SDelayAlu  # noqa: WPS433

    # The SDelayAlu class stores a raw integer. We need the instruction to
    # render as "s_delay_alu instid0(VALU_DEP_N)" in toString().
    # Override: build a minimal Instruction that renders correctly.
    inst = _SDelayAluFormatted(instid0type, instid0cnt)
    return inst


class _SDelayAluFormatted(_inst.Instruction):
    """SDelayAlu with human-readable encoding matching native rocisa output."""

    __slots__ = ("instid0type", "instid0cnt", "instskipCnt",
                 "instid1type", "instid1cnt")

    def __init__(self, instid0type: int, instid0cnt: int, comment: str = ""):
        super().__init__(_inst.InstType.INST_NOTYPE, comment)
        self.instid0type = instid0type
        self.instid0cnt = instid0cnt
        self.instskipCnt = None
        self.instid1type = None
        self.instid1cnt = None
        self.setInst("s_delay_alu")

    def hasInstID1(self) -> bool:
        return (self.instskipCnt is not None or self.instid1type is not None
                or self.instid1cnt is not None)

    def setInstID1(self, skip_cnt: int, instid1type: int, instid1cnt: int) -> bool:
        if self.hasInstID1():
            return False
        self.instskipCnt = skip_cnt
        self.instid1type = instid1type
        self.instid1cnt = instid1cnt
        return True

    def getParams(self):
        return []

    def getDstParams(self):
        return []

    def getSrcParams(self):
        return []

    def toString(self) -> str:
        result = " instid0(" + _format_dep_str(self.instid0type, self.instid0cnt) + ")"
        if not self.hasInstID1():
            return self.formatWithComment(self.instStr + result)
        _SKIP_NAMES = {0: "SAME", 1: "NEXT", 2: "SKIP_1", 3: "SKIP_2",
                       4: "SKIP_3", 5: "SKIP_4"}
        result += " | instskip(" + _SKIP_NAMES.get(self.instskipCnt, "") + ")"
        result += " | instid1(" + _format_dep_str(self.instid1type, self.instid1cnt) + ")"
        return self.formatWithComment(self.instStr + result)

    def to_stinky_logical(self):
        import stinkytofu as _st  # noqa: WPS433

        # stinkytofu's Python SDelayAlu binding triggers an assertion failure
        # (SDelayAluData not initialized) during emission.  Work around by
        # emitting an SNop(0) placeholder whose comment carries the original
        # s_delay_alu text; post-processing restores it.
        alu_text = "instid0(" + _format_dep_str(self.instid0type, self.instid0cnt) + ")"
        if self.hasInstID1():
            _SKIP_NAMES = {0: "SAME", 1: "NEXT", 2: "SKIP_1", 3: "SKIP_2",
                           4: "SKIP_3", 5: "SKIP_4"}
            alu_text += " | instskip(" + _SKIP_NAMES.get(self.instskipCnt, "") + ")"
            alu_text += " | instid1(" + _format_dep_str(self.instid1type, self.instid1cnt) + ")"
        return _st.SNop(_st.Register(0), "DELAY_ALU:" + alu_text)

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        dup = _SDelayAluFormatted(self.instid0type, self.instid0cnt, self.comment)
        if self.hasInstID1():
            dup.setInstID1(self.instskipCnt, self.instid1type, self.instid1cnt)
        memo[id(self)] = dup
        return dup


def _is_valu_writes_sgpr(inst: _inst.Instruction) -> bool:
    """True if inst is a VALU that writes an SGPR (e.g. v_cmp_*, v_readfirstlane)."""
    pre = inst.preStr()
    if not pre.startswith("v_"):
        return False
    try:
        for p in inst.getDstParams():
            if isinstance(p, RegisterContainer) and p.regType == "s":
                return True
    except (NotImplementedError, AttributeError):
        pass
    return False


def _insert_delay_alu_recursive(module: _code.Module) -> None:
    """Walk Module tree, inserting s_delay_alu where register deps are close."""
    for item in module.items():
        if isinstance(item, _code.Module):
            _insert_delay_alu_recursive(item)

    items = module.items()
    if len(items) < 2:
        return

    last_dst_inst_idx: Dict[Any, int] = {}
    inst_idx_delay_info: Dict[int, Tuple[int, int, int]] = {}  # idx -> (type, type_count, total)
    delay_type_counts: Dict[int, int] = {_DELAY_VALU: 0, _DELAY_TRANS: 0,
                                          _DELAY_SALU: 0, _DELAY_OTHER: 0}
    dep_idxs: Dict[int, _SDelayAluFormatted] = {}
    total_count = 0

    for i, item in enumerate(items):
        if not isinstance(item, _inst.Instruction):
            continue

        alu_type = _delay_alu_type(item)
        delay_type_counts[alu_type] = delay_type_counts.get(alu_type, 0) + 1
        total_count += 1
        inst_idx_delay_info[i] = (alu_type, delay_type_counts[alu_type], total_count)

        dsts, srcs = _get_dst_src_regs(item)

        # Find most-recently-written source register.
        # Native C++ uses std::max_element with a comparator that returns an
        # untracked register as the "maximum", then breaks when find()==end().
        # Net effect: if ANY src is not in last_dst_inst_idx, skip entirely.
        # Native does NOT subtract dst from srcs — self-read instructions
        # (e.g. s_lshr_b32 sX, sX, imm) still track the dep on sX.
        best_src = None
        best_idx = -1
        for src in srcs:
            idx = last_dst_inst_idx.get(src)
            if idx is None:
                best_src = None
                break
            if idx > best_idx:
                best_idx = idx
                best_src = src

        if best_src is not None:
            last_idx = best_idx
            dep_alu_type, dep_type_count, _ = inst_idx_delay_info[last_idx]
            inst_cnt = delay_type_counts[dep_alu_type] - dep_type_count
            max_dep = _ALU_DEP_MAX.get(dep_alu_type, 0)
            if inst_cnt <= max_dep:
                dep_idxs[i] = _SDelayAluFormatted(dep_alu_type, inst_cnt)

        for dst in dsts:
            last_dst_inst_idx[dst] = i

    # Insert in reverse order so indices stay valid
    for idx in sorted(dep_idxs.keys(), reverse=True):
        module.add(dep_idxs[idx], idx)


def insert_delay_alu(module: _code.Module) -> None:
    """Mirror ``rocisa::insertDelayAlu`` (insert_delay_alu.cpp).

    Inserts ``_SDelayAluFormatted`` instructions into the Module tree.
    These are later converted to SNop placeholders during stinkytofu lowering
    (see ``_SDelayAluFormatted.to_stinky_logical``), then restored to real
    ``s_delay_alu`` text by ``_postprocess_delay_alu_placeholder`` in code.py.
    """
    _insert_delay_alu_recursive(module)


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
# Graph optimisation — graph.cpp (buildGraph + removeDuplicateAssignment)
# ---------------------------------------------------------------------------


class _NoOptItem:
    """Wrapper mirroring native ``NoOptItem`` — marks items inside NoOpt modules."""
    __slots__ = ("item",)

    def __init__(self, item: Any) -> None:
        self.item = item


class _Graph:
    """Per-register item lists mirroring native ``Graph`` struct."""
    __slots__ = ("vgpr", "sgpr", "mgpr", "max_vgpr_seen")

    def __init__(self, max_vgpr: int, max_sgpr: int) -> None:
        self.vgpr: List[List[Any]] = [[] for _ in range(max_vgpr)]
        self.sgpr: List[List[Any]] = [[] for _ in range(max_sgpr)]
        self.mgpr: List[List[Any]] = [[]]
        self.max_vgpr_seen: int = -1

    def get_gpr_ref(self, reg_type: str) -> List[List[Any]]:
        if reg_type == "v":
            return self.vgpr
        elif reg_type == "s":
            return self.sgpr
        elif reg_type == "m":
            return self.mgpr
        raise RuntimeError(f"Invalid gpr type: {reg_type!r}")


_BARRIER_TYPES = (
    _inst.BranchInstruction,
    _code.Label,
    _inst._SWaitCnt,
    _inst._SWaitCntVscnt,
    _inst.SEndpgm,
    _inst.SBarrier,
    _inst.SNop,
)

# SSleep may not exist in adaptor; check at import time.
_SSleep = getattr(_inst, "SSleep", None)


def _is_barrier_item(item: Any) -> bool:
    if isinstance(item, _BARRIER_TYPES):
        return True
    if _SSleep is not None and isinstance(item, _SSleep):
        return True
    return False


def _add_reg_to_graph(item: Any, params: List[Any], graph: _Graph, no_opt: bool) -> None:
    """Add *item* to graph lists for every register it touches."""
    for p in params:
        if not isinstance(p, RegisterContainer):
            continue
        if p.regType == "acc":
            continue
        gpr_vec = graph.get_gpr_ref(p.regType)
        if p.regType == "v":
            last_idx = p.regIdx + p.regNum - 1
            if last_idx > graph.max_vgpr_seen:
                graph.max_vgpr_seen = last_idx
        for i in range(p.regIdx, p.regIdx + p.regNum):
            if i >= len(gpr_vec):
                continue
            if gpr_vec[i] and gpr_vec[i][-1] is item:
                continue
            if no_opt:
                gpr_vec[i].append(_NoOptItem(item))
            else:
                gpr_vec[i].append(item)


def _record_graph(module: _code.Module, graph: _Graph) -> None:
    """Recursively populate graph from module tree (mirrors native ``_recordGraph``)."""
    for item in module.items():
        if isinstance(item, _code.Module):
            _record_graph(item, graph)
        elif isinstance(item, (_inst.CommonInstruction, _inst.MacroInstruction)):
            get_params = getattr(item, "getParams", None)
            if callable(get_params):
                try:
                    params = get_params()
                except (NotImplementedError, RuntimeError):
                    continue
                _add_reg_to_graph(item, params, graph, module.isNoOpt())
        elif isinstance(item, _inst.Instruction) and not isinstance(item, _inst.BranchInstruction):
            # ReadWriteInstruction equivalent (SMemLoad, MUBUF, GLOBAL, FLAT, MFMA, etc.)
            get_params = getattr(item, "getParams", None)
            if callable(get_params):
                try:
                    params = get_params()
                except (NotImplementedError, RuntimeError):
                    pass
                else:
                    _add_reg_to_graph(item, params, graph, module.isNoOpt())
                    continue
            if _is_barrier_item(item):
                for v in graph.vgpr:
                    v.append(item)
                for s in graph.sgpr:
                    s.append(item)
        elif _is_barrier_item(item):
            for v in graph.vgpr:
                v.append(item)
            for s in graph.sgpr:
                s.append(item)


def _build_graph(module: _code.Module, max_vgpr: int, max_sgpr: int) -> _Graph:
    """Mirror native ``buildGraph``."""
    graph = _Graph(max_vgpr, max_sgpr)
    _record_graph(module, graph)
    return graph


def _remove_duplicate_assignment_gpr(graph: _Graph, reg_type: str) -> None:
    """Mirror native ``_removeDuplicateAssignmentGPR``.

    For each register index, track the last assigned value.  If a subsequent
    ``SMovB32`` writes the same value to the same register, remove it.
    """
    from .container import EXEC as _EXEC  # noqa: WPS433

    gpr_ref = graph.get_gpr_ref(reg_type)
    for idx in range(len(gpr_ref)):
        s_list = gpr_ref[idx]
        assign_value: Any = None  # None means 'unknown/invalidated'
        has_assign = False
        new_list: List[Any] = []
        removed_any = False

        for item in s_list:
            is_removed = False

            if isinstance(item, _NoOptItem):
                assign_value = None
                has_assign = False
            elif isinstance(item, _inst.MacroInstruction):
                assign_value = None
                has_assign = False
            elif _is_barrier_item(item):
                assign_value = None
                has_assign = False
            elif isinstance(item, _inst.SMovB32):
                dst = item.dst
                if not isinstance(dst, _EXEC):
                    if isinstance(dst, RegisterContainer) and dst.regIdx == idx:
                        gpr_value = item.srcs[0] if item.srcs else None
                        if has_assign and gpr_value == assign_value:
                            # Duplicate assignment — remove or replace with comment
                            parent = getattr(item, "parent", None)
                            if parent is not None and isinstance(parent, _code.Module):
                                if item.comment:
                                    comment = item.comment + " (dup assign opt.)"
                                    new_item = _code.TextBlock(
                                        " " * 50 + " // " + comment + "\n"
                                    )
                                    parent.replaceItem(item, new_item)
                                else:
                                    parent.removeItem(item)
                            is_removed = True
                        assign_value = gpr_value
                        has_assign = True
                    else:
                        # SMovB32 to a different register — just update if it writes this idx
                        pass
                else:
                    # Writing to EXEC — doesn't affect sgpr tracking
                    pass
            elif isinstance(item, _inst.Instruction):
                # Other instruction writing to this register invalidates tracking
                get_params = getattr(item, "getParams", None)
                if callable(get_params):
                    try:
                        params = get_params()
                    except (NotImplementedError, RuntimeError):
                        params = []
                    if params:
                        p0 = params[0]
                        if isinstance(p0, RegisterContainer) and p0.regType == reg_type:
                            for i in range(p0.regIdx, p0.regIdx + p0.regNum):
                                if i == idx:
                                    assign_value = None
                                    has_assign = False
                                    break

            if not is_removed:
                new_list.append(item)
            else:
                removed_any = True

        if removed_any:
            gpr_ref[idx] = new_list


def build_graph_and_remove_dup_assign(
    module: _code.Module, max_vgpr: int, max_sgpr: int
) -> int:
    """Mirror native ``buildGraph`` + ``removeDuplicateAssignment`` (graph.cpp).

    Builds a per-register reference graph, then eliminates redundant
    ``s_mov_b32`` instructions that re-assign the same value to the same SGPR.

    Returns the highest VGPR index seen (-1 if none).
    """
    graph = _build_graph(module, max_vgpr, max_sgpr)
    _remove_duplicate_assignment_gpr(graph, "s")
    return graph.max_vgpr_seen

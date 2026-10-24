# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Run emitted scalar control flow on the CPU.

Covers the SALU subset the persistent-loop walk emits, with labels and branches.
The scalar divide macros expand to VALU sequences; ``exact_scalar_divides``
swaps them for a marker this interpreter evaluates exactly, so a test checks the
arithmetic around a divide rather than the divide itself.
"""

import re

from rocisa.code import Label, Module, TextBlock
from rocisa.container import sgpr

_MASK = 0xFFFFFFFF
_DIVIDE = re.compile(r"/\*DIVIDE q=(\S+) d=(\S+) div=(\S+) r=(\S+) rem=([01])\*/")
_DIVIDE_MACROS = ("scalarUInt32DivideAndRemainder", "scalarUInt24DivideAndRemainder")


def exact_scalar_divides(monkeypatch, *modules):
    """Make the divide macros imported by ``modules`` emit exact-divide markers."""
    def divide(qReg, dReg, divReg, rReg, tmpVgprRes=None, wavewidth=64, doRemainder=True, **kwargs):
        module = Module("exact divide")
        module.add(TextBlock("/*DIVIDE q=%s d=%s div=%s r=%s rem=%d*/\n" % (
            sgpr(qReg), sgpr(dReg), sgpr(divReg), sgpr(rReg), 1 if doRemainder else 0)))
        return module

    for module in modules:
        for name in _DIVIDE_MACROS:
            if hasattr(module, name):
                monkeypatch.setattr(module, name, divide)


def _signed(value):
    return value - (1 << 32) if value & 0x80000000 else value


_COMPARES = {
    "SCmpEQU32": lambda a, b: a == b,
    "SCmpLgU32": lambda a, b: a != b,
    "SCmpLtU32": lambda a, b: a < b,
    "SCmpLeU32": lambda a, b: a <= b,
    "SCmpGtU32": lambda a, b: a > b,
    "SCmpGeU32": lambda a, b: a >= b,
    "SCmpEQI32": lambda a, b: _signed(a) == _signed(b),
    "SCmpLgI32": lambda a, b: _signed(a) != _signed(b),
    "SCmpLtI32": lambda a, b: _signed(a) < _signed(b),
    "SCmpLeI32": lambda a, b: _signed(a) <= _signed(b),
    "SCmpGtI32": lambda a, b: _signed(a) > _signed(b),
    "SCmpGeI32": lambda a, b: _signed(a) >= _signed(b),
}
_IGNORED = {"SNop", "SWaitAlu", "SWaitCnt"}


def run_scalar(module, registers, max_steps=1_000_000):
    """Execute ``module`` on a register file keyed by rendered operand text.

    Reading a register that was never written raises, so a test notices code
    that depends on a value it did not set up.
    """
    items = list(module.flatitems())
    targets = {item.getLabelName(): i for i, item in enumerate(items) if isinstance(item, Label)}
    regs = dict(registers)
    scc = 0

    def value(operand):
        text = str(operand)
        return regs[text] if text in regs else int(text, 0) & _MASK

    pc = 0
    for _ in range(max_steps):
        if pc >= len(items):
            return regs
        item = items[pc]
        pc += 1
        if isinstance(item, TextBlock):
            match = _DIVIDE.search(str(item))
            if match:
                q, d, divisor, r, remainder = match.groups()
                dividend, div = value(d), value(divisor)
                regs[q] = dividend // div
                if remainder == "1":
                    regs[r] = dividend % div
            continue
        if isinstance(item, Label) or not hasattr(item, "getParams"):
            continue
        name = type(item).__name__
        if name in _IGNORED:
            continue
        params = [str(p) for p in item.getParams()]
        if name == "SBranch":
            pc = targets[params[0]]
        elif name in ("SCBranchSCC0", "SCBranchSCC1"):
            if scc == (name == "SCBranchSCC1"):
                pc = targets[params[0]]
        elif name in _COMPARES:
            scc = int(_COMPARES[name](value(params[0]), value(params[1])))
        elif name == "SCSelectB32":
            regs[params[0]] = value(params[1]) if scc else value(params[2])
        elif name == "SCMovB32":
            if scc:
                regs[params[0]] = value(params[1])
        elif name == "SMovB32":
            regs[params[0]] = value(params[1])
        elif name == "SAbsI32":
            regs[params[0]] = abs(_signed(value(params[1]))) & _MASK
            scc = int(regs[params[0]] != 0)
        elif name == "SSExtI16toI32":
            low = value(params[1]) & 0xFFFF
            regs[params[0]] = (low - 0x10000 if low & 0x8000 else low) & _MASK
        else:
            a, b = value(params[1]), value(params[2])
            if name == "SAddU32":
                result = a + b
                scc = int(result > _MASK)
            elif name == "SSubU32":
                result = a - b
                scc = int(a < b)
            elif name == "SMulI32":
                result = a * b
            elif name == "SMulHIU32":
                result = (a * b) >> 32
            elif name in ("SAndB32", "SOrB32", "SXorB32"):
                result = {"SAndB32": a & b, "SOrB32": a | b, "SXorB32": a ^ b}[name]
                scc = int(result != 0)
            elif name in ("SLShiftLeftB32", "SLShiftRightB32"):
                result = (a << (b & 31) if name == "SLShiftLeftB32" else a >> (b & 31)) & _MASK
                scc = int(result != 0)
            elif name in ("SMinU32", "SMaxU32"):
                result = min(a, b) if name == "SMinU32" else max(a, b)
                scc = int(result == a)
            else:
                raise AssertionError("unsupported instruction in scalar test: %s" % str(item).strip())
            regs[params[0]] = result & _MASK
    raise AssertionError("scalar code did not finish within %d steps" % max_steps)

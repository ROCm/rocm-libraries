# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 MXScaleVgprInterleave characterization.

Target code: KernelWriter._mxsInterleavePlan / placeMXScaleInterleave /
remapMXScaleInterleave and the interleaved RegSets in macroAndSet.

v_wmma_scale costs one cycle when its two scale VGPRs share a VGPR bank
(index % 8). The config emits MX-F4 kernels whose scale blocks are 1/1, 2/2,
4/4 and 8/2 VGPRs (MXSA/MXSB), with and without SourceSwap. With the interleave
on, every v_wmma_scale must read its two scales from disjoint banks, every MXS
reference must resolve inside v0-v255, and no VGPR may be reached through two
different scale blocks.

pytestmark = pytest.mark.unit. CPU-only; no GPU.
"""

import os
import re

import pytest

from config_harness import assert_assembles, emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx1250",
    "mx_scale_vgpr_interleave.yaml",
)

_SET = re.compile(r"^\.set\s+(\w+),\s*(.+?)\s*$")
_RANGE = re.compile(r"v\[([^\]:]+)(?::([^\]]+))?\]")
_MXS = re.compile(r"vgprValuMXS([AB])_X(\d+)_I(\d+)(?:_P(\d+))?\b")
_OPERANDS = re.compile(r",\s*(?![^\[]*\])")


def _eval(expr, sym):
    text = re.sub(r"[A-Za-z_]\w*", lambda m: str(sym.get(m.group(0), "None")), expr)
    try:
        return int(eval(text))  # noqa: S307 - integer arithmetic on resolved .set values
    except (TypeError, SyntaxError, NameError):
        return None


def _range(operand, sym):
    m = _RANGE.match(operand.strip())
    if not m:
        return None
    lo = _eval(m.group(1), sym)
    hi = _eval(m.group(2), sym) if m.group(2) else lo
    return None if lo is None or hi is None else range(lo, hi + 1)


def _scan(src):
    """(wmma count, bank conflicts, MXS VGPRs >= 256, VGPRs shared by two blocks, piece symbols)."""
    sym, owner = {}, {}
    wmma = conflicts = high = shared = 0
    pieces = set()
    for raw in src.splitlines():
        line = raw.split("//")[0].strip()
        m = _SET.match(line)
        if m:
            value = _eval(m.group(2), sym)
            if value is None:
                sym.pop(m.group(1), None)
            else:
                sym[m.group(1)] = value
            continue
        if not line or line.startswith(".") or line.endswith(":") or " " not in line:
            continue
        ops = _OPERANDS.split(line.split(None, 1)[1])
        for op in ops:
            mm = _MXS.search(op)
            regs = _range(op.split()[0], sym) if mm else None
            if regs is None:
                continue
            if mm.group(4):
                pieces.add(mm.group(0))
            block = (mm.group(1), sym.get("vgprValuMXS%s_X%s_I%s" % mm.group(1, 2, 3)))
            for v in regs:
                high += v >= 256
                shared += owner.setdefault(v, block) != block
        if line.startswith("v_wmma_scale"):
            wmma += 1
            s0, s1 = _range(ops[4], sym), _range(ops[5].split()[0], sym)
            if s0 and s1 and {v % 8 for v in s0} & {v % 8 for v in s1}:
                conflicts += 1
    return wmma, conflicts, high, shared, pieces


@pytest.fixture(scope="module")
def kernels():
    return emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)


def test_mx_scale_vgpr_interleave_emits(kernels):
    assert len(kernels) == 8, f"expected 8 kernels, got {len(kernels)}"
    for base, src, err in kernels:
        assert err == 0, f"kernel {base!r} emitted with err={err}"
        assert "MXScaleVgprInterleave ignored" not in src


def test_mx_scale_vgpr_interleave_banks(kernels):
    splitBlocks = 0
    for base, src, _err in kernels:
        wmma, conflicts, high, shared, pieces = _scan(src)
        assert wmma > 0, f"{base}: no v_wmma_scale"
        assert conflicts == 0, f"{base}: {conflicts}/{wmma} v_wmma_scale read both scales from one bank"
        assert high == 0, f"{base}: MXS VGPR past v255"
        assert shared == 0, f"{base}: a VGPR is reached through two scale blocks"
        splitBlocks += bool(pieces)
    # The 8-VGPR MXSA blocks (MI 32x16, MIWaveTile 8x2) are addressed through _P1 pieces.
    assert splitBlocks >= 1, "no kernel splits an MXSA block into pieces"


def test_mx_scale_vgpr_interleave_assembles(kernels):
    for base, src, _err in kernels:
        assert_assembles(src, base)

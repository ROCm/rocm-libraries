# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 MXScaleVgprInterleave characterization.

Target code: KernelWriter._mxsInterleavePlan / placeMXScaleInterleave /
remapMXScaleInterleave and the interleaved RegSets in macroAndSet.

v_wmma_scale costs one cycle when its two scale VGPRs share a VGPR bank
(index % 8). The interleave is on by default (KernelWriter.do["MXScaleVgprInterleave"]).
Where it applies, every v_wmma_scale must read its two scales from disjoint banks, every MXS
reference must resolve inside v0-v255, and no VGPR may be reached through two different scale
blocks. It never raises the VGPR count: a kernel whose layout needs a larger scale region than the
default keeps the default layout (kernels already at the 1024 VGPR limit would not assemble).
With the switch off every kernel keeps the default layout, whose scale banks collide.

pytestmark = pytest.mark.unit. CPU-only; no GPU.
"""

import os
import re

import pytest

from config_harness import assert_assembles, emit_kernels_from_config
from tensilelite.KernelWriter import KernelWriter

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"

_DESIGNED = os.path.join(os.path.dirname(__file__), "data", "test_data", "_designed", "gfx1250")

# (config, kernels keep the default layout because the interleave would need more VGPRs than it)
_CONFIGS = [
    ("cluster_padding_npot_mt1.yaml", True),
    ("streamk_cluster_multicast.yaml", False),
    ("streamk_cluster_multicast_pgr2.yaml", False),
    ("streamk_tdm_prefetchgl2.yaml", False),
    ("streamk_tdm_prefetchgl2_f4.yaml", False),
]
_IDS = [name for name, _ in _CONFIGS]
_VGPRS = re.compile(r"\.amdhsa_next_free_vgpr\s+(\d+)")


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


def _emit(name):
    return emit_kernels_from_config(os.path.join(_DESIGNED, name), limit=8, arch=_ARCH)


@pytest.fixture(scope="module", params=_CONFIGS, ids=_IDS)
def kernels(request):
    name, fallsBack = request.param
    return fallsBack, _emit(name)


@pytest.fixture(scope="module", params=_CONFIGS, ids=_IDS)
def kernels_on_off(request):
    name, fallsBack = request.param
    on = _emit(name)
    init = KernelWriter.__init__

    def initOff(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self.do["MXScaleVgprInterleave"] = False

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(KernelWriter, "__init__", initOff)
        off = _emit(name)
    return fallsBack, on, off


def _vgprs(src):
    return int(_VGPRS.search(src).group(1))


def test_mx_scale_vgpr_interleave_emits(kernels):
    out = kernels[1]
    assert out, "no kernels emitted"
    for base, src, err in out:
        assert err == 0, f"kernel {base!r} emitted with err={err}"


def test_mx_scale_vgpr_interleave_banks(kernels):
    fallsBack, out = kernels
    if fallsBack:
        pytest.skip("kernels keep the default layout; see test_mx_scale_vgpr_interleave_falls_back")
    for base, src, _err in out:
        wmma, conflicts, high, shared, _pieces = _scan(src)
        assert wmma > 0, f"{base}: no v_wmma_scale"
        assert conflicts == 0, f"{base}: {conflicts}/{wmma} v_wmma_scale read both scales from one bank"
        assert high == 0, f"{base}: MXS VGPR past v255"
        assert shared == 0, f"{base}: a VGPR is reached through two scale blocks"


def test_mx_scale_vgpr_interleave_assembles(kernels):
    for base, src, _err in kernels[1]:
        assert_assembles(src, base)


def test_mx_scale_vgpr_interleave_never_adds_vgprs(kernels_on_off):
    _fallsBack, on, off = kernels_on_off
    assert [b for b, _s, _e in off] == [b for b, _s, _e in on]
    for (base, src, _e), (_b, srcOff, _eOff) in zip(on, off):
        assert _vgprs(src) <= _vgprs(srcOff), f"{base}: the interleave raised the VGPR count"


def test_mx_scale_vgpr_interleave_falls_back(kernels_on_off):
    # The layout needs a larger scale region than the default: the kernel keeps the default
    # layout, with its scale-bank conflicts and the same VGPR count.
    fallsBack, on, off = kernels_on_off
    if not fallsBack:
        pytest.skip("the interleave applies to these kernels")
    for (base, src, _e), (_b, srcOff, _eOff) in zip(on, off):
        wmma, conflicts, _high, _shared, pieces = _scan(src)
        assert not pieces, f"{base}: interleave pieces in a kernel that must keep the default layout"
        assert conflicts == _scan(srcOff)[1] > 0, f"{base}: layout differs from the default"
        assert _vgprs(src) == _vgprs(srcOff)


def test_mx_scale_vgpr_interleave_switch_off(kernels_on_off):
    # Same kernels (the switch is not part of the name) in the default layout, where the
    # scale banks collide.
    fallsBack, on, off = kernels_on_off
    colliding = 0
    for base, src, err in off:
        assert err == 0, f"kernel {base!r} emitted with err={err}"
        _wmma, conflicts, _high, _shared, pieces = _scan(src)
        assert not pieces, f"{base}: interleave pieces with the switch off"
        colliding += conflicts > 0
    assert colliding >= 1, "no kernel has scale-bank conflicts in the default layout"

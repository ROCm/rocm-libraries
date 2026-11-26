# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Codegen test for the dynamic StreamK part decode (StreamK.dynamicPartRange):
the emitted scalar code, run on a small interpreter, must give each part of a
split tile the iterations StreamK::Dynamic::partIterRange() gives it on the
host, for the exact (uneven) splits the host packs and the equal splits of
kernels without SupportStreamKDynamicUnevenParts.
"""

import itertools
import re
import shutil
from types import SimpleNamespace

import pytest

# Prime the component registry before StreamK imports (avoids circular import).
from tensilelite.KernelWriterAssembly import KernelWriterAssembly  # noqa: F401

import rocisa

from tensilelite.Common import IsaVersion
from tensilelite.Common.Capabilities import makeIsaInfoMap
from tensilelite.Components.StreamK import StreamKDynamic, StreamKHybrid
from rocisa_test_state import preserve_rocisa_kernel_state

pytestmark = pytest.mark.unit

_MASK = 0xFFFFFFFF


@pytest.fixture(autouse=True)
def _gfx950():
    isa = IsaVersion(9, 5, 0)
    with preserve_rocisa_kernel_state():
        makeIsaInfoMap([isa], shutil.which("amdclang++") or "/opt/rocm/bin/amdclang++")
        rocisa.rocIsa.getInstance().setKernel(isa, 64)
        yield


class _Pool:
    def __init__(self):
        self._next = 100

    def checkOut(self, n, *args, **kwargs):
        reg = self._next
        self._next += n
        return reg

    def checkIn(self, *args, **kwargs):
        return None


def _kernel(g, **extra):
    """A kernel whose StreamK K unit (skKGranularity) is g iterations."""
    kernel = dict(extra)
    if g > 1:
        kernel.update(MXScaleFormat="HostPreSwizzle", DepthU=256 // g, _ScaleReuseA=g)
    return kernel


def _emit(strategy, g, **extra):
    w = SimpleNamespace(sgprPool=_Pool())
    module = strategy().dynamicPartRange(w, _kernel(g, **extra))
    return [str(i).split("//")[0].strip() for i in module.flatitems() if str(i).strip()]


def _run(lines, regs):
    """Interpret the scalar ALU subset the decode emits."""
    regs = dict(regs)
    scc = 0

    def val(op):
        m = re.fullmatch(r"s\[sgpr(\w+)\]|s(\d+)", op)
        if m:
            return regs[m.group(1) or int(m.group(2))]
        return int(op, 0) & _MASK

    def dst(op):
        m = re.fullmatch(r"s\[sgpr(\w+)\]|s(\d+)", op)
        return m.group(1) or int(m.group(2))

    def signed(x):
        return x - (1 << 32) if x & 0x80000000 else x

    for line in lines:
        op, args = line.split(None, 1)
        a = [x.strip() for x in args.split(",")]
        if op == "s_cmp_lt_u32":
            scc = int(val(a[0]) < val(a[1]))
            continue
        x = val(a[1])
        y = val(a[2]) if len(a) > 2 else None
        r = {
            "s_mul_i32": lambda: x * y,
            "s_add_u32": lambda: x + y,
            "s_sub_u32": lambda: x - y,
            "s_and_b32": lambda: x & y,
            "s_min_u32": lambda: min(x, y),
            "s_max_i32": lambda: max(signed(x), signed(y)),
            "s_lshr_b32": lambda: x >> y,
            "s_lshl_b32": lambda: x << y,
            "s_cselect_b32": lambda: x if scc else y,
        }[op]()
        regs[dst(a[0])] = r & _MASK
    return regs["StreamKLocalStart"], regs["StreamKLocalEnd"]


def _cdiv(a, b):
    return -(-a // b)


def _padded(ipt, g):
    return _cdiv(ipt, g) * g


def _exactSplit(ipt, split, g):
    """StreamK::Dynamic::computeSplit()'s uneven parts: (SKSplit, SKItersPerWI)."""
    units = _padded(ipt, g) // g
    parts = min(split, units)
    return parts, g * (units // parts)


def _equalSplit(ipt, split, g):
    """The equal parts packed for kernels without the capability."""
    padded = _padded(ipt, g)
    itersPerWI = g * _cdiv(_cdiv(padded, split), g)
    return _cdiv(padded, itersPerWI), itersPerWI


def _partIterRange(ipt, parts, itersPerWI, g, p):
    """Mirrors the C++ StreamK::Dynamic::partIterRange(), which the gtests check; the two can drift."""
    extra = max(0, _padded(ipt, g) - parts * itersPerWI) // g
    start = p * itersPerWI + g * min(p, extra)
    return start, min(start + itersPerWI + (g if p < extra else 0), ipt)


_CASES = list(itertools.product((1, 7, 15, 64, 100, 369, 1000), (2, 3, 8, 12, 32, 64)))


@pytest.mark.parametrize("strategy", [StreamKDynamic, StreamKHybrid], ids=lambda c: c.__name__)
@pytest.mark.parametrize("g", [1, 2, 4])
def test_exact_split_decodes_contiguous_uneven_parts(strategy, g):
    lines = _emit(strategy, g)
    for ipt, split in _CASES:
        parts, itersPerWI = _exactSplit(ipt, split, g)
        units = _padded(ipt, g) // g
        assert parts == min(split, units), "a split is exact while every part keeps a unit"
        end = 0
        lengths = []
        for p in range(parts):
            regs = {"SKSplit": parts, "SKItersPerWI": itersPerWI, "ItersPerTile": ipt,
                    "StreamKPartialIdx": p}
            start, stop = _run(lines, regs)
            where = (ipt, split, p)
            assert (start, stop) == _partIterRange(ipt, parts, itersPerWI, g, p), where
            assert start == end, ("parts tile K without gaps or overlap", where)
            assert start % g == 0, ("every part starts on a K unit", where)
            assert stop > start, ("no empty part", where)
            lengths.append(stop - start)
            end = stop
        assert end == ipt
        # All but the clipped last part differ by at most one unit, longest first.
        body = lengths[:-1] or [0]
        assert max(body) - min(body) <= g
        assert body == sorted(body, reverse=True)


@pytest.mark.parametrize("g", [1, 2, 4])
def test_equal_split_decodes_as_before(g):
    # Kernels without the capability get equal parts: start = p * SKItersPerWI.
    lines = _emit(StreamKDynamic, g)
    for ipt, split in _CASES:
        parts, itersPerWI = _equalSplit(ipt, split, g)
        for p in range(parts):
            regs = {"SKSplit": parts, "SKItersPerWI": itersPerWI, "ItersPerTile": ipt,
                    "StreamKPartialIdx": p}
            assert _run(lines, regs) == (p * itersPerWI, min((p + 1) * itersPerWI, ipt)), \
                (ipt, split, p)


def test_369_into_32_gives_32_parts():
    # ipt 369 split 32: 31 equal parts of 12 before; now 17 parts of 12 and 15 of 11.
    parts, itersPerWI = _exactSplit(369, 32, 1)
    assert (parts, itersPerWI) == (32, 11)
    lines = _emit(StreamKDynamic, 1)
    lengths = [stop - start for start, stop in
               (_run(lines, {"SKSplit": parts, "SKItersPerWI": itersPerWI, "ItersPerTile": 369,
                             "StreamKPartialIdx": p}) for p in range(parts))]
    assert lengths == [12] * 17 + [11] * 15


@pytest.mark.parametrize("advertised", [None, True])
def test_capability_accepted(advertised):
    isp = {} if advertised is None else {"SupportStreamKDynamicUnevenParts": advertised}
    assert _emit(StreamKDynamic, 1, InternalSupportParams=isp)


def test_codegen_rejects_a_kernel_that_does_not_advertise_it():
    with pytest.raises(AssertionError, match="SupportStreamKDynamicUnevenParts"):
        _emit(StreamKDynamic, 1, InternalSupportParams={"SupportStreamKDynamicUnevenParts": False})

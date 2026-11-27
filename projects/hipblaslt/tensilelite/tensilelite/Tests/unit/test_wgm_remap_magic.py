# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx11 DefaultWGM with magic-number divides; other targets keep the fp64 divides.

The emitted code runs on the CPU with the semantics of each instruction,
including the fp32 VALU sequence that derives the magic numbers (v_rcp_f32 is
modelled as the correctly rounded reciprocal perturbed by a relative error),
and is compared against DefaultWGM's definition for every tile of a grid.
The divide itself is checked for every divisor the remap can see.
"""

import itertools
import math
import random
import struct
from contextlib import contextmanager
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
from rocisa.code import Label, TextBlock

from tensilelite.Components.WorkGroupMappingAlgos import DefaultWGM, WGMMagicPersistentSetup, \
    WGM_MAGIC_SGPRS, wgmRemapMagic

_MASK = 0xFFFFFFFF
_DIVIDEND_LIMIT = 1 << 24      # the bound DefaultWGM's fp64 divides already need
_DIVISOR_LIMIT = 1 << 16


def _signed(value):
    return value - (1 << 32) if value & 0x80000000 else value


def _f32(x):
    """Round a double to the nearest fp32 (products of fp32 values are exact in double)."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


_COMPARES = {
    "SCmpEQU32": lambda a, b: a == b,
    "SCmpLeU32": lambda a, b: a <= b,
    "SCmpGeU32": lambda a, b: a >= b,
    "SCmpGtI32": lambda a, b: _signed(a) > _signed(b),
    "SCmpGeI32": lambda a, b: _signed(a) >= _signed(b),
}


class _Machine:
    """Single-lane executor for the instructions the magic remap emits."""

    def __init__(self, rcpError=0.0):
        self.rcpError = rcpError
        self.sgprs = {}
        self.vgprs = {}

    def sread(self, text):
        if text.startswith("s[") and ":" in text and not text.startswith("s[sgpr"):
            raise AssertionError("64-bit scalar read: %s" % text)
        if text in self.sgprs:
            return self.sgprs[text]
        return int(text, 0) & _MASK

    def iread(self, text):
        """Integer view: VGPRs written by integer ops, SGPRs, literals."""
        if text in self.vgprs:
            value = self.vgprs[text]
            assert isinstance(value, int), "float VGPR %s read as integer" % text
            return value
        return self.sread(text)

    def fread(self, text):
        """fp32 view: VGPRs written by float ops, SGPRs as fp32 bit patterns."""
        if text in self.vgprs:
            value = self.vgprs[text]
            assert isinstance(value, float), "integer VGPR %s read as float" % text
            return value
        return struct.unpack("<f", struct.pack("<I", self.sread(text)))[0]

    def run(self, module, max_steps=100000):
        items = list(module.flatitems())
        targets = {item.getLabelName(): i for i, item in enumerate(items) if isinstance(item, Label)}
        scc, pc = 0, 0
        counts = {"salu": 0, "valu": 0}
        for _ in range(max_steps):
            if pc >= len(items):
                return counts
            item = items[pc]
            pc += 1
            if isinstance(item, (Label, TextBlock)) or not hasattr(item, "getParams"):
                continue
            name = type(item).__name__
            p = [str(x) for x in item.getParams()]
            counts["valu" if name.startswith("V") else "salu"] += 1
            if name == "SBranch":
                pc = targets[p[0]]
            elif name in ("SCBranchSCC0", "SCBranchSCC1"):
                if scc == (name == "SCBranchSCC1"):
                    pc = targets[p[0]]
            elif name in _COMPARES:
                scc = int(_COMPARES[name](self.sread(p[0]), self.sread(p[1])))
            elif name == "SCSelectB32":
                self.sgprs[p[0]] = self.sread(p[1]) if scc else self.sread(p[2])
            elif name == "SMovB32":
                self.sgprs[p[0]] = self.sread(p[1])
            elif name == "SAbsI32":
                self.sgprs[p[0]] = abs(_signed(self.sread(p[1]))) & _MASK
            elif name == "SSExtI16toI32":
                low = self.sread(p[1]) & 0xFFFF
                self.sgprs[p[0]] = (low - 0x10000 if low & 0x8000 else low) & _MASK
            elif name == "SFlbitI32B32":
                v = self.sread(p[1])
                self.sgprs[p[0]] = (32 - v.bit_length()) if v else _MASK
            elif name == "VCvtU32toF32":
                self.vgprs[p[0]] = _f32(self.iread(p[1]))
            elif name == "VCvtI32toF32":
                self.vgprs[p[0]] = _f32(_signed(self.iread(p[1])))
            elif name == "VRcpF32":
                self.vgprs[p[0]] = _f32(float(Fraction(1) / Fraction(self.fread(p[1])) * (1 + Fraction(self.rcpError))))
            elif name == "VMulF32":
                self.vgprs[p[0]] = _f32(self.fread(p[1]) * self.fread(p[2]))
            elif name == "VCvtF32toU32":
                self.vgprs[p[0]] = min(max(math.trunc(self.fread(p[1])), 0), _MASK)
            elif name == "VCvtF32toI32":
                self.vgprs[p[0]] = min(max(math.trunc(self.fread(p[1])), -(1 << 31)), (1 << 31) - 1) & _MASK
            elif name == "VMulLOU32":
                self.vgprs[p[0]] = (self.iread(p[1]) * self.iread(p[2])) & _MASK
            elif name == "VSubU32":
                self.vgprs[p[0]] = (self.iread(p[1]) - self.iread(p[2])) & _MASK
            elif name == "VAdd3U32":
                self.vgprs[p[0]] = (self.iread(p[1]) + self.iread(p[2]) + self.iread(p[3])) & _MASK
            elif name == "VReadfirstlaneB32":
                self.sgprs[p[0]] = self.iread(p[1])
            else:
                a, b = self.sread(p[1]), self.sread(p[2])
                if name == "SAddU32":
                    r = a + b
                    scc = int(r > _MASK)
                elif name == "SSubU32":
                    r = a - b
                    scc = int(a < b)
                elif name == "SMulI32":
                    r = a * b
                elif name == "SMulHIU32":
                    r = (a * b) >> 32
                elif name == "SOrB32":
                    r = a | b
                    scc = int(r != 0)
                elif name == "SAndB32":
                    r = a & b
                    scc = int(r != 0)
                elif name == "SXorB32":
                    r = a ^ b
                    scc = int(r != 0)
                elif name in ("SLShiftLeftB32", "SLShiftRightB32"):
                    r = (a << (b & 31)) if name == "SLShiftLeftB32" else (a >> (b & 31))
                    r &= _MASK
                    scc = int(r != 0)
                else:
                    raise AssertionError("unsupported instruction: %s" % str(item).strip())
                self.sgprs[p[0]] = r & _MASK
        raise AssertionError("did not finish")


class _Pool:
    def __init__(self, start):
        self.next = start

    def checkOut(self, size, tag="", *args, **kwargs):
        idx = self.next
        self.next += size
        return idx

    def checkOutAligned(self, size, alignment, tag="", *args, **kwargs):
        self.next += (-self.next) % alignment
        return self.checkOut(size, tag)

    def checkIn(self, idx):
        pass


class _Labels:
    def __init__(self):
        self.count = 0

    def getNameInc(self, name):
        self.count += 1
        return "%s_%d" % (name, self.count)


class _Writer:
    def __init__(self):
        self.labels = _Labels()
        self.sgprPool = _Pool(100)
        self.vgprPool = _Pool(0)
        self.states = SimpleNamespace(WGMTransformLevels=-1)

    @contextmanager
    def allocTmpSgpr(self, size, alignment=1, tag=""):
        yield SimpleNamespace(idx=self.sgprPool.checkOutAligned(size, alignment or 1), size=size)

    @contextmanager
    def allocTmpSgprList(self, nums, alignmentList=None, tag=""):
        yield [SimpleNamespace(idx=self.sgprPool.checkOut(n), size=n) for n in nums]


def _kernel(persistent, wgmxcc, isa=(11, 5, 1)):
    return {
        "ISA": isa,
        "WorkGroupMappingXCC": wgmxcc,
        "ClusterDim": [1, 1],
        "SpaceFillingAlgo": [],
        "WavefrontSize": 32,
        "TileProcessingStrategy": "StreamK" if persistent else "None",
        "WorkAssignment": "StaticGrid",
    }


def _wgm_field(wgm, wgmxcc):
    """Pack WGM the way the host does, with a neighbouring field set to prove it is masked."""
    if wgmxcc == -1:
        return (0x3 << 10) | (wgm & 0x3FF)
    return (0x5 << 16) | (wgm & 0xFFFF)


def _wgm_map(wg0, wg1, n0, n1, wgm):
    """DefaultWGM's definition."""
    if wgm > 1:
        first, second, nFirst, nSecond = wg0, wg1, n0, n1
    elif wgm >= -1:
        return wg0, wg1
    else:
        first, second, nFirst, nSecond = wg1, wg0, n1, n0
    width = abs(wgm)
    block = second // width
    serial = (second % width) * nFirst + first
    fullBlocks = nSecond // width
    divisor = (nSecond - fullBlocks * width or width) if block >= fullBlocks else width
    newFirst, newSecond = serial // divisor, serial % divisor + block * width
    return (newFirst, newSecond) if wgm > 1 else (newSecond, newFirst)


def _remap_programs(persistent, wgmxcc, isa=(11, 5, 1)):
    writer = _Writer()
    kernel = _kernel(persistent, wgmxcc, isa)
    setup = WGMMagicPersistentSetup(writer, kernel) if persistent else None
    return setup, DefaultWGM(writer, kernel, "WGM")


def _check_grid(persistent, wgmxcc, n0, n1, wgm, tiles, rcpError=0.0, isa=(11, 5, 1)):
    setup, remap = _remap_programs(persistent, wgmxcc, isa)
    machine = _Machine(rcpError)
    machine.sgprs.update({"s[sgprWGM]": _wgm_field(wgm, wgmxcc),
                          "s[sgprNumWorkGroups0]": n0, "s[sgprNumWorkGroups1]": n1})
    if setup is not None:
        machine.run(setup)
        state = {k: v for k, v in machine.sgprs.items() if k.startswith("s[sgprWGMMagic")}
    for wg0, wg1 in tiles:
        if setup is not None:
            # Only the persistent registers survive from one tile to the next.
            machine.sgprs = {"s[sgprWGM]": _wgm_field(wgm, wgmxcc),
                             "s[sgprNumWorkGroups0]": n0, "s[sgprNumWorkGroups1]": n1, **state}
        machine.sgprs["s[sgprWorkGroup0]"] = wg0
        machine.sgprs["s[sgprWorkGroup1]"] = wg1
        machine.run(remap)
        got = (machine.sgprs["s[sgprWorkGroup0]"], machine.sgprs["s[sgprWorkGroup1]"])
        assert got == _wgm_map(wg0, wg1, n0, n1, wgm), (n0, n1, wgm, wg0, wg1)


_GRIDS = list(itertools.product(
    [1, 3, 5, 12],                                          # NumWorkGroups0
    [1, 4, 7, 16],                                          # NumWorkGroups1
    [-9, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 8, 24],          # WGM
    [-1, 1],                                                # WorkGroupMappingXCC: the two field layouts
    [False, True],                                          # persistent
))


@pytest.mark.parametrize("n0, n1, wgm, wgmxcc, persistent", _GRIDS)
def test_every_tile_matches_default_wgm(n0, n1, wgm, wgmxcc, persistent):
    tiles = [(i % n0, i // n0) for i in range(n0 * n1)]
    _check_grid(persistent, wgmxcc, n0, n1, wgm, tiles)


@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize("rcpError", [-2.0 ** -20, 2.0 ** -20])
def test_large_grids_and_wgm(persistent, rcpError):
    """Wide WGM, large grids, and a v_rcp_f64 that is off by 2^-20 either way."""
    rng = random.Random(1151)
    for _ in range(40):
        n0, n1 = rng.randint(1, 4096), rng.randint(1, 4096)
        wgm = rng.choice([-1, 1]) * rng.randint(2, 32767)
        if n0 * n1 > (1 << 22):
            n1 = max(1, (1 << 22) // n0)
        tiles = [(rng.randrange(n0), rng.randrange(n1)) for _ in range(30)]
        tiles += [(n0 - 1, n1 - 1), (0, 0), (n0 - 1, 0), (0, n1 - 1)]
        _check_grid(persistent, 1, n0, n1, wgm, tiles, rcpError)


def _sgprText(name):
    from rocisa.container import sgpr
    return str(sgpr(name))


@pytest.mark.parametrize("rcpError", [-2.0 ** -20, 0.0, 2.0 ** -20])
def test_magic_number_in_range_for_every_divisor(rcpError):
    """m(d) lies in [2^T/d, 2^T/d + 4] for every d the remap can divide by."""
    from tensilelite.Components.WorkGroupMappingAlgos import wgmMagicNumber
    writer = _Writer()
    module = wgmMagicNumber(writer, "M", "FL", "D")
    machine = _Machine(rcpError)
    step = 1 if rcpError == 0.0 else 7       # the perturbed runs sample, the exact one is exhaustive
    for d in itertools.chain(range(1, _DIVISOR_LIMIT + 1, step), [2 ** k for k in range(17)],
                             [2 ** k + 1 for k in range(16)], [2 ** k - 1 for k in range(1, 17)]):
        machine.sgprs = {_sgprText("D"): d}
        machine.run(module)
        m, fl = machine.sgprs[_sgprText("M")], machine.sgprs[_sgprText("FL")]
        assert fl == d.bit_length() - 1
        T = 31 + fl
        assert (1 << T) <= m * d <= (1 << T) + 4 * d, d
        assert m <= _MASK


def _division_ok(d, m):
    """Both emitted divide forms, for every n < 2^24 that can be the worst case."""
    fl = d.bit_length() - 1
    # floor(n*m/2^T) is never below n // d once m*d >= 2^T, and its excess over n/d
    # grows with n, so per residue class the largest n decides: check those, plus
    # the full range for small d.
    if d <= 64:
        n = np.arange(_DIVIDEND_LIMIT, dtype=np.uint64)
    else:
        n = np.arange(d - 1, _DIVIDEND_LIMIT, d, dtype=np.uint64)
        n = np.concatenate([n, np.arange(_DIVIDEND_LIMIT - d, _DIVIDEND_LIMIT, dtype=np.uint64)])
    want = n // np.uint64(d)
    mm = np.uint64(m)
    formB = ((((n << np.uint64(8)) & np.uint64(_MASK)) * mm) >> np.uint64(32)) >> np.uint64(fl + 7)
    if not np.array_equal(formB, want):
        return False
    if d >= 2:
        formA = ((n * mm) >> np.uint64(32)) >> np.uint64(fl - 1)
        if not np.array_equal(formA, want):
            return False
    return True


@pytest.mark.parametrize("chunk", range(8))
def test_magic_divide_exact_for_every_divisor(chunk):
    """Largest magic the sequence can return, every divisor 1..2^16."""
    lo = 1 + chunk * (_DIVISOR_LIMIT // 8)
    hi = lo + _DIVISOR_LIMIT // 8
    for d in range(lo, hi):
        T = 31 + d.bit_length() - 1
        mMax = (1 << T) // d + 4
        mMin = -((-(1 << T)) // d)
        assert _division_ok(d, mMax), d
        assert _division_ok(d, mMin), d


_GFX11 = [(11, 0, 0), (11, 0, 1), (11, 0, 2), (11, 0, 3), (11, 5, 0), (11, 5, 1), (11, 5, 2), (11, 5, 3)]
_OTHER_ISAS = [(9, 4, 2), (9, 5, 0), (12, 0, 0), (12, 0, 1)]


def test_gate_is_gfx11_with_default_wgm():
    base = {"ClusterDim": [1, 1], "SpaceFillingAlgo": []}
    for isa in _GFX11:
        assert wgmRemapMagic({**base, "ISA": isa})
        assert not wgmRemapMagic({**base, "ISA": isa, "ClusterDim": [2, 1]})
        assert not wgmRemapMagic({**base, "ISA": isa, "SpaceFillingAlgo": ["Hilbert"]})
    for isa in _OTHER_ISAS:
        assert not wgmRemapMagic({**base, "ISA": isa})
    assert not wgmRemapMagic(base)


@pytest.mark.parametrize("isa", [(11, 0, 0), (11, 0, 3), (11, 5, 0), (11, 5, 2)])
def test_other_gfx11_targets_remap_every_tile(isa):
    for persistent, wgm in itertools.product([False, True], [-3, 4]):
        tiles = [(i % 5, i // 5) for i in range(5 * 7)]
        _check_grid(persistent, 1, 5, 7, wgm, tiles, isa=isa)


def _fp64_reference(persistent, wgmxcc):
    """DefaultWGM as emitted before the gfx11 path existed (no ISA key reaches the old code)."""
    kernel = _kernel(persistent, wgmxcc)
    del kernel["ISA"]
    return str(DefaultWGM(_Writer(), kernel, "WGM"))


@pytest.mark.parametrize("isa", _OTHER_ISAS)
@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize("wgmxcc", [-1, 1])
def test_other_targets_keep_the_fp64_remap(isa, persistent, wgmxcc):
    writer = _Writer()
    kernel = _kernel(persistent, wgmxcc, isa)
    assert not len(WGMMagicPersistentSetup(writer, kernel).flatitems())
    rendered = str(DefaultWGM(writer, kernel, "WGM"))
    assert rendered == _fp64_reference(persistent, wgmxcc)
    assert rendered.count("v_rcp_f64") == 6
    assert "Magic" not in rendered


def test_per_tile_remap_is_salu_only_when_persistent():
    _, remap = _remap_programs(True, -1)
    names = {type(i).__name__ for i in remap.flatitems() if hasattr(i, "getParams")}
    assert not [n for n in names if n.startswith("V")], names
    rendered = str(remap)
    for reg in WGM_MAGIC_SGPRS:
        assert "sgpr%s" % reg in rendered

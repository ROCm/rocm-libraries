################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################
import io
import pytest
from Tensile.Common.Architectures import gfxToIsa
from Tensile.Common.Utilities import state
from Tensile.Hardware import HardwarePredicate, _extractPciChipIds
from Tensile.SolutionLibrary import PredicateLibrary

ISA = gfxToIsa("gfx950")


def hw(cu=None, chips=None):
    return HardwarePredicate.FromHardware(ISA, cu, chips)


def row(pred, i):
    return {"predicate": pred, "library": StubLib(i)}


def sig(pred):
    if pred.tag == "TruePred":
        return (None, None)
    inner = pred.value
    chip = (
        next((p for p in inner.value if p.tag in ("PciChipId", "Or")), None)
        if inner.tag == "And"
        else None
    )
    cu = (
        next((p for p in inner.value if p.tag == "CUCount"), None)
        if inner.tag == "And"
        else None
    )
    chipIds = _extractPciChipIds(chip) if chip else frozenset()
    chipSig = tuple(sorted(chipIds, reverse=True)) if chipIds else None
    return (chipSig, cu.value if cu else None)


class StubLib:
    Tag = "Single"

    def __init__(self, i):
        self.solution = type("S", (), {"index": i})()

    tag = property(lambda self: self.Tag)

    def state(self):
        return {"type": self.tag, "index": self.solution.index}

    def merge(self, _):
        pass

    def remapSolutionIndices(self, _):
        pass


def build_lib():
    lib = PredicateLibrary(tag="Hardware")
    for i, pred in enumerate(
        [
            hw(256, ["Device 75a0"]),
            hw(64, ["Device 75a0"]),
            hw(None, ["Device 75a0"]),
            hw(256, ["Device 75a3"]),
            hw(64, ["Device 75a3"]),
            hw(None, ["Device 75a3"]),
            hw(),
        ],
        1,
    ):
        lib.merge(PredicateLibrary(tag="Hardware", rows=[row(pred, i)]))
    return lib


def test_hardware_predicate_sort_priority():
    mi350_spx, mi350_cpx, mi350, mi355_spx, mi355_cpx, mi355, proc = [
        hw(256, ["Device 75a0"]),
        hw(64, ["Device 75a0"]),
        hw(None, ["Device 75a0"]),
        hw(256, ["Device 75a3"]),
        hw(64, ["Device 75a3"]),
        hw(None, ["Device 75a3"]),
        hw(),
    ]
    got = sorted(
        [
            proc,
            mi350,
            HardwarePredicate("TruePred"),
            mi355_cpx,
            mi350_spx,
            mi355,
            mi355_spx,
            mi350_cpx,
        ]
    )
    assert got == [
        mi355_spx,
        mi355_cpx,
        mi355,
        mi350_spx,
        mi350_cpx,
        mi350,
        proc,
        HardwarePredicate("TruePred"),
    ]


def test_predicate_library_merge_and_state_preserve_order():
    lib = build_lib()
    assert [sig(r["predicate"]) for r in lib.rows] == [
        ((0x75a3,), 256),
        ((0x75a3,), 64),
        ((0x75a3,), None),
        ((0x75a0,), 256),
        ((0x75a0,), 64),
        ((0x75a0,), None),
        (None, None),
    ]
    assert [r["library"].solution.index for r in lib.rows] == [
        r["library"]["index"] for r in state(lib)["rows"]
    ]


@pytest.mark.parametrize("fmt", ["msgpack", "yaml"])
def test_round_trip_preserves_row_order(fmt):
    lib = build_lib()
    data = state(lib)
    mem = [r["library"].solution.index for r in lib.rows]
    if fmt == "msgpack":
        m = pytest.importorskip("msgpack")
        loaded = m.unpackb(m.packb(data), raw=False)
    else:
        y = pytest.importorskip("yaml")
        b = io.StringIO()
        y.safe_dump(data, b)
        loaded = y.safe_load(b.getvalue())
    assert mem == [r["library"]["index"] for r in loaded["rows"]]


def test_chip_id_or_predicate_preserves_all_ids_as_set():
    pred = hw(None, ["Device 75a0", "Device 75a3"])
    assert sig(pred) == ((0x75a3, 0x75a0), None)


def test_sort_prefers_exact_chip_over_multi_chip_set():
    mi350 = hw(None, ["Device 75a0"])
    mi355 = hw(None, ["Device 75a3"])
    mixed = hw(None, ["Device 75a0", "Device 75a3"])

    got = sorted([mixed, mi350, mi355])
    assert got == [mi355, mixed, mi350]

# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only unit coverage of the OutputPeerD store in ``KernelWriterAssembly.addStore``.

addStore is driven with stubs for everything around it (the global write emitter, the address
calculation and the writer state), so no toolchain or GPU is needed. What is checked is the
contract of the peer store: it is the same D store, addressed through SrdPeerD, which is rebuilt
from the live SrdD plus the distance between the two buffers, with a system-scope store, and the
row increment of the regular store is not repeated for it.
"""

from types import SimpleNamespace

import pytest
from rocisa.code import Module
from rocisa.enum import CacheScope
from rocisa.instruction import SAddCU32, SAddU32, SMovB64

from tensilelite.KernelWriterAssembly import KernelWriterAssembly

pytestmark = pytest.mark.unit


class _Recorder:
    """Stands in for chooseGlobalWrite and records every call."""

    def __init__(self):
        self.calls = []

    def __call__(self, useBuffer, bps, sumIdx, rpv, addr0, addr1, globalOffset, **kw):
        self.calls.append({"addr1": str(addr1), "scope": kw["scope"], "comment": kw["comment"]})
        return Module("global write")


def _writer():
    writer = KernelWriterAssembly.__new__(KernelWriterAssembly)
    writer.do = {"GlobalWrite": True}
    writer.states = SimpleNamespace(
        asmCaps={"HasTHModifier": True},  # a gfx1250-like scope/th store
        bpeCexternal=4,
        bpr=4,
    )
    writer.chooseGlobalWrite = _Recorder()
    return writer


def _kernel(outputPeerD):
    dest = SimpleNamespace(
        isHalf=lambda: False, isBFloat16=lambda: False, isInt32=lambda: False, isSingle=lambda: True
    )
    return {
        "NonTemporalD": 0,
        "TemporalHintD": 0,
        "NonVolatileD": 0,
        "BufferStore": 1,
        "CompactLoopStore": False,
        "ProblemType": {"DestDataType": dest, "OutputPeerD": outputPeerD},
    }


def _addrCalc(rowInc=0):
    calc = SimpleNamespace(addrDVgpr=5, rowInc=rowInc, globalOffset=0, increments=0)

    def incrementToNextRow(*args, **kwargs):
        calc.increments += 1
        return Module("row increment")

    calc.incrementToNextRow = incrementToNextRow
    return calc


def _store(writer, kernel, calc, optSrdIncForRow=False):
    ss = SimpleNamespace(cfg=SimpleNamespace(gwvw=1), optSrdIncForRow=optSrdIncForRow)
    return writer.addStore(kernel, ss, "D", calc, 0, "tmp", False)


def _instructions(module):
    return [type(item) for item in module.flatitems()]


def test_without_peer_d_there_is_one_store_through_srd_d():
    writer = _writer()
    module = _store(writer, _kernel(False), _addrCalc())
    assert [c["addr1"] for c in writer.chooseGlobalWrite.calls] == ["s[sgprSrdD:sgprSrdD+3]"]
    assert SAddU32 not in _instructions(module)


def test_with_peer_d_the_same_store_is_repeated_through_srd_peer_d():
    writer = _writer()
    module = _store(writer, _kernel(True), _addrCalc())
    calls = writer.chooseGlobalWrite.calls
    assert [c["addr1"] for c in calls] == ["s[sgprSrdD:sgprSrdD+3]", "s[sgprSrdPeerD:sgprSrdPeerD+3]"]
    # The regular store keeps its scope; the peer store is visible to another device.
    assert calls[0]["scope"] == CacheScope.SCOPE_CU
    assert calls[1]["scope"] == CacheScope.SCOPE_SYS
    assert calls[1]["comment"].endswith("(peer D)")
    # SrdPeerD = SrdD + delta (low and high word), and the limit and flags words are copied.
    instructions = _instructions(module)
    assert instructions.count(SAddU32) == 1
    assert instructions.count(SAddCU32) == 1
    assert instructions.count(SMovB64) == 1


def test_the_row_increment_belongs_to_the_regular_store_only():
    writer = _writer()
    calc = _addrCalc(rowInc=1)
    _store(writer, _kernel(True), calc, optSrdIncForRow=True)
    assert calc.increments == 1
    assert len(writer.chooseGlobalWrite.calls) == 2

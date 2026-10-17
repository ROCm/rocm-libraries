# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Retain complete K0 descriptors across arbitrary epilogue SGPR clobbers."""

from copy import deepcopy

import pytest
from rocisa.code import Module
from rocisa.container import sgpr
from rocisa.instruction import SMovB32
from test_PapAddressPipeline import _lane_instructions
from test_PapPgr2Pipeline import _setup as _pipeline, _PipelineMachine
from test_PapShadowAddress import _components, _run  # noqa: F401 - pytest autouse fixture
from test_PapIncrementalTdm import _DescriptorMachine, _machine, _full

pytestmark = pytest.mark.unit




class _LaneMachine(_PipelineMachine):
    def run(self, module):
        return super().run(_lane_instructions(module))


class _DescriptorLaneMachine(_DescriptorMachine):
    def run(self, module):
        text = _lane_instructions(module)
        text = "\n".join(line for line in text.splitlines() if line != "scheduling_fence")
        return super().run(text)


def _setup(cluster=(1, 1)):
    kernel, tensors, writer = _pipeline(cluster=cluster)
    writer.states.papNextTileVgpr = 12
    writer.sgprs = {"tdmAGroup0": 16, "tdmAGroup1": 20, "tdmAGroup2": 28,
                    "tdmMXSAGroup0": 68, "tdmMXSAGroup1": 72}
    return kernel, tensors, writer


def _ordinary(words):
    module = Module("ordinary descriptor rebuild")
    for index, name in enumerate(words):
        module.add(SMovB32(dst=sgpr(name), src=0xa000 + index))
    return module


@pytest.mark.parametrize("bank", [0, 1])
def test_k0_snapshot_survives_group1_and_epilogue_clobbers(bank):
    kernel, tensors, writer = _setup()
    words = writer.papDescriptorSnapshotLayout(kernel, tensors['A'], tensors['B'])
    assert len(words) == len(set(words)) == 28
    machine = _LaneMachine(PersistentPrefetchState=0x80000005)
    expected = {name: 0xf0000000 + i for i, name in enumerate(words)}
    expected['tdmAGroup0+1'] = 0x1000
    expected['tdmMXSAGroup0+1'] = 0x12000
    for name, value in expected.items():
        machine[name] = value + (bank * kernel['LdsOffsetA_Blk'] if name.endswith('Group0+1') else 0)
    for lane in range(4):
        machine[f'Lane12_{lane}'] = 0x100 + lane
    original = {name: machine[name] for name in words}
    machine.run(writer.papSaveNextTileDescriptors(kernel, tensors['A'], tensors['B']))
    assert all(machine[name] == value for name, value in original.items())
    assert machine['PersistentPrefetchState'] == 0x80000045
    assert [machine[f'Lane12_{lane}'] for lane in range(4)] == list(range(0x100, 0x104))
    # Simulate group1 increments, descriptor nulling, and arbitrary store scratch.
    for name in words:
        machine[name] = 0xdeadbeef
    wrong_carrier = deepcopy(machine)
    writer.states.papNextTileVgpr = 13
    wrong_carrier.run(writer.papRestoreNextTileDescriptors(kernel, tensors['A'], tensors['B'], _ordinary(words)))
    assert {name: wrong_carrier[name] for name in words} != expected
    writer.states.papNextTileVgpr = 12
    machine.run(writer.papRestoreNextTileDescriptors(kernel, tensors['A'], tensors['B'], _ordinary(words)))
    assert {name: machine[name] for name in words} == expected
    assert machine['PersistentPrefetchState'] == 0x80000005
    assert [machine[f'Lane12_{lane}'] for lane in range(4)] == list(range(0x100, 0x104))


@pytest.mark.parametrize("state", [0, 1, 4, 5, 64, 65, 69, 0x80000045])
def test_restore_requires_snapshot_and_issued_tile_and_consumes_only_snapshot_bit(state):
    kernel, tensors, writer = _setup()
    words = writer.papDescriptorSnapshotLayout(kernel, tensors['A'], tensors['B'])
    machine = _LaneMachine(PersistentPrefetchState=state)
    for index in range(28):
        machine[f'Lane12_{index + 4}'] = 0xb000 + index
    machine.run(writer.papRestoreNextTileDescriptors(kernel, tensors['A'], tensors['B'], _ordinary(words)))
    base = 0xb000 if state & 65 == 65 else 0xa000
    assert [machine[name] for name in words] == list(range(base, base + 28))
    assert machine['PersistentPrefetchState'] == state & ~64


@pytest.mark.parametrize("fallback", ["halfplr", "missing_group", "overlap", "no_payload", "separate"])
def test_descriptor_snapshot_rejects_unsupported_register_layout(fallback):
    kernel, tensors, writer = _setup()
    if fallback == 'halfplr':
        kernel.update(HalfPLR=1, SuppressNoLoadLoop=True)
    elif fallback == 'missing_group':
        del writer.sgprs['tdmAGroup2']
    elif fallback == 'overlap':
        writer.sgprs['tdmMXSAGroup0'] = writer.sgprs['tdmAGroup0']
    elif fallback == 'no_payload':
        writer.states.papNextTileVgpr = -1
    else:
        writer.tdmDescriptorSetOwner = lambda kernel, tc: tc
    assert writer.papDescriptorSnapshotLayout(kernel, tensors['A'], tensors['B']) == ()
    assert not writer.papSaveNextTileDescriptors(kernel, tensors['A'], tensors['B']).items()
    ordinary = Module('ordinary')
    assert writer.papRestoreNextTileDescriptors(kernel, tensors['A'], tensors['B'], ordinary) is ordinary


@pytest.mark.parametrize("cluster", [(1, 1), (2, 4)])
@pytest.mark.parametrize("wave", range(4))
@pytest.mark.parametrize("bank", [0, 1])
def test_restored_real_descriptors_match_ordinary_k0_entry(cluster, wave, bank):
    kernel, tensors, writer = _setup(cluster)
    machine = _DescriptorLaneMachine()
    machine.__dict__.update(deepcopy(_machine(kernel, tensors, wave).__dict__))
    for index, value in enumerate((1, 2, 3)):
        machine.set(f'WorkGroup{index}', value)
    ordinary = _full(writer, kernel, tensors)
    _run(machine, ordinary)
    if bank:
        for tc in ('A', 'MXSA'):
            _run(machine, writer.tdmSwapLdsOffset(kernel, tensors[tc]))
    machine.set('PersistentPrefetchState', 5 | bank * 2)
    _run(machine, writer.papSaveNextTileDescriptors(kernel, tensors['A'], tensors['B']))
    # Current group1/epilogue work may destroy every saved descriptor word.
    for name in writer.papDescriptorSnapshotLayout(kernel, tensors['A'], tensors['B']):
        machine.set(name, 0xdeadbeef)
    reference = deepcopy(machine)
    _run(reference, ordinary)
    _run(machine, writer.papRestoreNextTileDescriptors(kernel, tensors['A'], tensors['B'], ordinary))
    assert machine.descriptor(True) == reference.descriptor(True)
    assert machine.get('PersistentPrefetchState') == 5 | bank * 2


@pytest.mark.parametrize("start", [0, 3])
@pytest.mark.parametrize("bank", [0, 1])
def test_streamk_snapshot_preserves_bounds_and_applies_k_offset_once(start, bank):
    kernel, tensors, writer = _setup()
    kernel["TileProcessingStrategy"] = "StreamK"
    words = writer.papDescriptorSnapshotLayout(kernel, tensors["A"], tensors["B"])
    assert len(words) == 26
    for wave in range(4):
        machine = _DescriptorLaneMachine()
        machine.__dict__.update(deepcopy(_machine(kernel, tensors, wave).__dict__))
        machine.set("StreamKLocalStart", start)
        machine.set("StreamKLocalEnd", 5)
        machine.set("tdmABIncs", 0xfffff100)
        machine.set("tdmMXSAMXSBIncs", 0x180)
        ordinary = _full(writer, kernel, tensors)
        _run(machine, ordinary)
        _run(machine, writer.papApplyEntryKOffset(kernel, tensors["A"], tensors["B"]))
        expected = machine.descriptor(True)
        for lane, value in enumerate((1, 2, 3, start, 5)):
            machine.set(f"Lane12_{lane}", value)
        if bank:
            for tc in ("A", "MXSA"):
                _run(machine, writer.tdmSwapLdsOffset(kernel, tensors[tc]))
        machine.set("PersistentPrefetchState", 5)
        _run(machine, writer.papSaveNextTileDescriptors(kernel, tensors["A"], tensors["B"]))
        for name in (*words, "tdmAGroup0", "tdmMXSAGroup0"):
            machine.set(name, 0xdeadbeef)
        _run(machine, writer.papRestoreNextTileDescriptors(kernel, tensors["A"], tensors["B"], ordinary))
        _run(machine, writer.papApplyEntryKOffset(kernel, tensors["A"], tensors["B"]))
        assert machine.descriptor(True) == expected
        assert machine.get("PersistentPrefetchState") == 5
        assert [machine.get(f"Lane12_{i}") for i in range(5)] == [1, 2, 3, start, 5]

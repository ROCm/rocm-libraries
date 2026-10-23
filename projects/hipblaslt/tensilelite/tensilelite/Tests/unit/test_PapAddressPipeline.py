# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Exercise the real PAP mapper, borrowed identity, commit and fallback on CPU.

The mapper runs on the strict single-lane control interpreter with the address
leaf replaced by coordinate captures. The consumer runs real descriptor code on
its physical-word interpreter. Neither model claims GPU timing or memory order.
"""

import re
from copy import deepcopy
from types import SimpleNamespace

import pytest
from rocisa.code import Module
from rocisa.container import sgpr
from rocisa.enum import RegisterType
from rocisa.instruction import SAddU32, SMovB32, SMovB64
from rocisa.register import RegisterPool
from Tensile.Components.PersistentLoop import PersistentKernelState, PersistentLoopOn
from Tensile.Components.TileProcessingStrategy import TileWork
from test_PapLateHandoff import _LowVgprMachine
from test_PapShadowAddress import _setup, _run, _components  # noqa: F401 - pytest autouse fixture
from test_PapIncrementalTdm import _full, _machine

pytestmark = pytest.mark.unit


def _lane_instructions(module):
    # Model register and lane identity, including reads from the wrong carrier.
    lines = []
    for raw in str(module).splitlines():
        line = raw.split("//", 1)[0].strip()
        write = re.fullmatch(r"v_writelane_b32\s+v\[?(\d+)\]?,\s*([^,]+),\s*(\d+)", line)
        read = re.fullmatch(r"v_readlane_b32\s+([^,]+),\s*v\[?(\d+)\]?,\s*(\d+)", line)
        if write:
            line = f"s_mov_b32 s[sgprLane{write[1]}_{write[3]}], {write[2]}"
        elif read:
            line = f"s_mov_b32 {read[1]}, s[sgprLane{read[2]}_{read[3]}]"
        lines.append(line)
    return "\n".join(lines)


class _MapperMachine(_LowVgprMachine):
    def run(self, module):
        # gfx12 spells the same low-word add/subtract operation explicitly nc.
        return super().run(str(module).replace('v_sub_nc_u32', 'v_sub_u32')
                           .replace('v_add_nc_u32', 'v_add_u32'))


def _carrier(scales=True):
    size = 7 if scales else 5
    return dict(base=80, size=size, data=80, mx=82 if scales else None,
                coordinates={i: 80 + size - 3 + i for i in range(3)})


def _writer(*, scales=True, cluster=(1, 1)):
    kernel, tensors, writer = _setup(scales=scales, cluster=cluster)
    kernel.update(WorkGroupMappingXCC=-1, NoTailLoop=False)
    for tc in ('A', 'B'):
        if scales:
            tensors[tc]['MX'] = tensors['MXS' + tc]
    writer.states.tailloopInNll = False
    writer.states.currentTileWork = TileWork('PersistentTileID')
    writer.states.papAddressCarrier = _carrier(scales)
    return kernel, tensors, writer


@pytest.mark.parametrize('cluster', [(1, 1), (2, 4), (4, 1)])
@pytest.mark.parametrize('shape', [(1, 1), (5, 7), (16, 16)])
@pytest.mark.parametrize('exec_mask', [0, 1, 0x80000000, 0x55aa55aa, 0xffffffff])
def test_speculative_mapper_preserves_exec_identity_and_cursor(monkeypatch, cluster, shape, exec_mask):
    kernel, tensors, writer = _writer(cluster=cluster)

    def capture(kernel, a, b, coordinates, dst, scratch):
        leaf = Module('observable pure address inputs')
        for i in range(3):
            leaf.add(SMovB32(sgpr('Captured%s%u' % (a['tensorChar'], i)), sgpr(coordinates[i])))
        leaf.add(SMovB32(sgpr(dst), 123))
        leaf.add(SMovB32(sgpr(dst + 1), 456))
        return leaf

    monkeypatch.setattr(writer, 'papTdmPrepareAddressPair', capture)
    producer = writer.papPrepareNextTileAddresses(kernel, tensors['A'], tensors['B'])
    assert not any(line.strip().endswith(':') for line in str(producer).splitlines())
    assert 's_branch' not in str(producer) and 's_cbranch' not in str(producer)
    assert 's_barrier' not in str(producer) and 'tensor_load' not in str(producer)
    cs, ck = cluster
    n0, n1 = shape
    blocks0, blocks1 = (n0 + cs - 1) // cs, (n1 + ck - 1) // ck
    tiles = blocks0 * blocks1 * cs * ck
    # Include an exhausted cursor: speculation must not dereference or advance it.
    for tile in (0, 1, tiles - 1, tiles, 3 * tiles + 1):
        machine = _MapperMachine(NextTile=tile, NumWorkGroups0=n0, NumWorkGroups1=n1,
                                 WGM=0x1c001, PersistentPrefetchState=3,
                                 WorkGroup0=91, WorkGroup1=92, WorkGroup2=93,
                                 PersistentPhantomTile=1)
        machine.registers['exec_lo'] = exec_mask
        machine.exec = bool(exec_mask)
        assert machine.run(producer) is None
        block, peer = divmod(tile, cs * ck)
        batch, within = divmod(block, blocks0 * blocks1)
        block1, block0 = divmod(within, blocks0)
        expected = (min(block0 * cs + peer % cs, n0 - 1),
                    min(block1 * ck + peer // cs, n1 - 1), batch)
        for tc in ('A', 'MXSA'):
            assert tuple(machine['Captured%s%u' % (tc, i)] for i in range(3)) == expected
        assert machine.registers['exec_lo'] == exec_mask
        assert (machine['WorkGroup0'], machine['WorkGroup1'], machine['WorkGroup2']) == (91, 92, 93)
        assert machine['PersistentPhantomTile'] == 1 and machine['NextTile'] == tile
        assert machine['PersistentPrefetchState'] == 0x80000007


@pytest.mark.parametrize('xcc', [-1, 0])
@pytest.mark.parametrize('runtime_wgm,valid', [(0, False), (1, True), (2, False), (-1, False), (-511, False)])
def test_runtime_wgm_fallback_keeps_branchless_leaf(monkeypatch, xcc, runtime_wgm, valid):
    kernel, tensors, writer = _writer(scales=False)
    kernel['WorkGroupMappingXCC'] = xcc
    monkeypatch.setattr(writer, 'papTdmPrepareAddressPair', lambda *args: Module())
    producer = writer.papPrepareNextTileAddresses(kernel, tensors['A'], tensors['B'])
    field = 0x3ff if xcc == -1 else 0xffff
    machine = _MapperMachine(NextTile=1, NumWorkGroups0=4, NumWorkGroups1=3,
                             WGM=(0x90000000 | (runtime_wgm & field)), PersistentPrefetchState=2)
    machine.registers['exec_lo'] = 0xffffffff
    machine.run(producer)
    assert machine['PersistentPrefetchState'] == (0x80000006 if valid else 2)
    assert 's_branch' not in str(producer) and 's_cbranch' not in str(producer)


@pytest.mark.parametrize('state', [0, 1, 2, 3, 4, 5, 7, 0x80000004, 0x80000007])
def test_prepared_bit_clear_preserves_issued_bank_and_reservation(state):
    machine = _MapperMachine(PersistentPrefetchState=state)
    module = Module()
    module.add(PersistentKernelState.papClearPreparedAddress('invalidate'))
    machine.run(module)
    assert machine['PersistentPrefetchState'] == state & ~4


@pytest.mark.parametrize('scales,data_only', [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize('cluster', [(1, 1), (2, 4)])
@pytest.mark.parametrize('prepared', [False, True])
@pytest.mark.parametrize('bank', [False, True])
def test_consumer_commits_once_and_restores_current_identity(scales, data_only, cluster, prepared, bank):
    kernel, tensors, writer = _writer(scales=scales, cluster=cluster)
    plan = _carrier(scales and not data_only)
    writer.states.papAddressCarrier = plan
    next_coordinates = (1, 2, 3)
    current_coordinates = (2, 1, 0)

    def mapping(*args, **kwargs):
        module = Module('fallback mapper with an observable phantom clobber')
        for i, value in enumerate(next_coordinates):
            module.add(SMovB32(sgpr('WorkGroup%u' % i), value))
        module.add(SMovB32(sgpr('PersistentPhantomTile'), 0))
        module.add(SAddU32(sgpr('Mappings'), sgpr('Mappings'), 1))
        return module

    def issue(*args, **kwargs):
        module = Module('one observable first PGR issue')
        module.add(SMovB64(sgpr('IssuedData', 2), sgpr('tdmAGroup0+2', 2)))
        if scales:
            module.add(SMovB64(sgpr('IssuedMX', 2), sgpr('tdmMXSAGroup0+2', 2)))
        module.add(SAddU32(sgpr('Loads'), sgpr('Loads'), 1))
        module.add(SAddU32(sgpr('TensorToken'), sgpr('TensorToken'), 1))
        module.add(SMovB32(sgpr('PersistentPrefetchState'), 1))
        return module

    writer.calculateLoopNumIter = lambda *args: Module()
    writer.setupPrefetchAcrossPersistentLoads = issue
    consumer = PersistentLoopOn().prefetchPreparedClassic(writer, kernel, tensors['A'], tensors['B'],
        SimpleNamespace(prefetchAcrossPersistentSetupNextTile=mapping), plan)
    assert str(consumer).count('s[sgprLoads]') == 2  # one destination and one source
    pairs = [('A', 'B')] + ([('MXSA', 'MXSB')] if scales else [])
    for wave in range(4):
        machine = _machine(kernel, tensors, wave)
        for i, value in enumerate(current_coordinates):
            machine.set('WorkGroup%u' % i, value)
        _run(machine, _full(writer, kernel, tensors))
        machine.set('PersistentPhantomTile', 1)
        for i, value in enumerate(next_coordinates):
            machine.set(plan['coordinates'][i], value)
        for a, b in pairs:
            output = plan['data'] if a == 'A' else plan['mx']
            if output is not None:
                with writer.allocTmpSgpr(3, alignment=2) as scratch:
                    _run(machine, writer.papTdmPrepareAddressPair(kernel, tensors[a], tensors[b],
                        plan['coordinates'], output, scratch))
            if bank:
                _run(machine, writer.tdmSwapLdsOffset(kernel, tensors[a]))
            machine.set('tdm%sGroup0' % a, 0)
        reference = deepcopy(machine)
        for i, value in enumerate(next_coordinates):
            reference.set('WorkGroup%u' % i, value)
        for a, b in pairs:
            _run(reference, writer.papTdmUpdateDescriptor(kernel, tensors[a], tensors[b]))
        machine.set('PersistentPrefetchState', 0x80000004 if prepared else 0)
        _run(machine, consumer)
        assert machine.get('IssuedData', 2) == reference.get('tdmAGroup0+2', 2)
        if scales:
            assert machine.get('IssuedMX', 2) == reference.get('tdmMXSAGroup0+2', 2)
        assert machine.descriptor(scales) == reference.descriptor(scales)
        assert tuple(machine.get('WorkGroup%u' % i) for i in range(3)) == current_coordinates
        assert machine.get('PersistentPhantomTile') == (1 if cluster != (1, 1) or prepared else 0)
        assert machine.get('Loads') == machine.get('TensorToken') == 1
        assert machine.get('Mappings') == int(not prepared)
        assert machine.get('PersistentPrefetchState') == 1 | (2 if bank else 0)
        # The caller's full restore still addresses the current tile after the
        # common coordinate restoration, including the currently live LDS bank.
        for a, b in pairs:
            _run(machine, writer.papTdmUpdateDescriptor(kernel, tensors[a], tensors[b], preservePapBank=False))
        current_reference = _machine(kernel, tensors, wave)
        for i, value in enumerate(current_coordinates):
            current_reference.set('WorkGroup%u' % i, value)
        _run(current_reference, _full(writer, kernel, tensors))
        assert machine.descriptor(scales) == current_reference.descriptor(scales)


@pytest.mark.parametrize('budget,available,expected', [(106, 32, True), (106, 7, True), (106, 0, False)])
def test_resource_preflight_never_mutates_live_pool(budget, available, expected):
    kernel, tensors, writer = _writer()
    writer.states.regCaps = {'MaxSgpr': budget, 'MaxVgpr': 1024}
    writer.sgprPool = RegisterPool(98 if available else 106, RegisterType.Sgpr, defaultPreventOverflow=False)
    if available:
        writer.sgprPool.add(20, available, 'test scratch')
    writer.vgprPool = RegisterPool(8, RegisterType.Vgpr, defaultPreventOverflow=False)
    writer.vgprPool.add(0, 8, 'test scratch')
    before = (writer.sgprPool.size(), str(writer.sgprPool), writer.vgprPool.size(), str(writer.vgprPool))
    plan = writer.papAddressCarrierPlan(kernel, tensors['A'], tensors['B'])
    assert (plan is not None) == expected
    assert (writer.sgprPool.size(), str(writer.sgprPool), writer.vgprPool.size(), str(writer.vgprPool)) == before


@pytest.mark.parametrize('cluster', [(1, 1), (2, 4)])
@pytest.mark.parametrize('arg_type', [0, 1, 2, 3])
@pytest.mark.parametrize('wave', range(4))
@pytest.mark.parametrize('wgm', [1, 2])
def test_runtime_arg_type_and_wgm_control_prepared_consumption(monkeypatch, cluster, arg_type, wave, wgm):
    kernel, tensors, writer = _writer(cluster=cluster)
    kernel['ProblemType']['SupportUserArgs'] = True
    writer.states.tdmParityPackedInArgType = True
    writer.states.tdmWaveIdBitsInArgType = 2
    # Real pure address emission with known mapped coordinates. The mapper is
    # covered above; this executes the actual mode predicate and consumer arm.
    _, address = writer.papNextTileAddressModules(kernel, tensors['A'], tensors['B'])
    machine = _machine(kernel, tensors, wave)
    machine.set('ArgType', (wave << 8) | arg_type)
    machine.set('WGM', 0x80000000 | wgm)
    for i, value in enumerate((1, 2, 3)):
        machine.set(writer.states.papAddressCarrier['coordinates'][i], value)
    machine.set('PersistentPrefetchState', 0)
    _run(machine, address)
    valid = arg_type == 0 and (cluster != (1, 1) or wgm == 1)
    assert machine.get('PersistentPrefetchState') == (0x80000004 if valid else 0)
    assert not machine.memory  # pointer-array speculation did not dereference
    assert machine.get('ArgType') == (wave << 8) | arg_type

    def descriptor(kernel, a, b, **kwargs):
        module = Module()
        counter = 'PreparedCommits' if kwargs.get('preparedGlobalAddr') is not None else 'OrdinaryCommits'
        module.add(SAddU32(sgpr(counter), sgpr(counter), 1))
        return module

    monkeypatch.setattr(writer, 'papTdmUpdateDescriptor', descriptor)
    monkeypatch.setattr(writer, 'calculateLoopNumIter', lambda *args: Module())
    monkeypatch.setattr(writer, 'setupPrefetchAcrossPersistentLoads', lambda *args, **kw: Module())
    monkeypatch.setattr(writer, 'papTdmSaveLdsBank', lambda *args: Module())
    processing = SimpleNamespace(prefetchAcrossPersistentSetupNextTile=lambda *args, **kw: Module())
    consumer = PersistentLoopOn().prefetchPreparedClassic(writer, kernel, tensors['A'], tensors['B'],
                                                        processing, writer.states.papAddressCarrier)
    _run(machine, consumer)
    assert machine.get('PreparedCommits') == (2 if valid else 0)
    assert machine.get('OrdinaryCommits') == (0 if valid else 2)


def test_failed_preflight_has_no_state_labels_or_pool_mutations():
    kernel, tensors, writer = _writer()
    writer.sgprPool = RegisterPool(106, RegisterType.Sgpr, defaultPreventOverflow=False)
    writer.states.regCaps = {'MaxSgpr': 106, 'MaxVgpr': 1024}
    writer.states.papAddressActivationReset = Module()
    writer.states.papAddressRoutingReset = Module()
    before = (writer._label, writer.sgprPool.size(), str(writer.sgprPool))
    with writer.allocPapAddressCarrier(kernel, tensors['A'], tensors['B']) as carrier:
        assert carrier is None
    assert before == (writer._label, writer.sgprPool.size(), str(writer.sgprPool))
    assert not writer.states.papAddressActivationReset.items()
    assert not writer.states.papAddressRoutingReset.items()


@pytest.mark.parametrize('free_ranges', [((81, 7), (99, 4)), ((80, 7), (97, 4)), ((90, 5), (99, 4))])
def test_fragmented_pool_preflight_honors_nested_alignment(free_ranges):
    kernel, tensors, writer = _writer(cluster=(2, 4))
    writer.states.regCaps = {'MaxSgpr': 106, 'MaxVgpr': 1024}
    writer.sgprPool = RegisterPool(105, RegisterType.Sgpr, defaultPreventOverflow=False)
    for start, count in free_ranges:
        writer.sgprPool.add(start, count, 'fragmented scratch')
    writer.vgprPool = RegisterPool(8, RegisterType.Vgpr, defaultPreventOverflow=False)
    writer.vgprPool.add(0, 8, 'scratch')
    before = (writer.sgprPool.size(), str(writer.sgprPool))
    plan = writer.papAddressCarrierPlan(kernel, tensors['A'], tensors['B'])
    assert plan is None
    assert before == (writer.sgprPool.size(), str(writer.sgprPool))


@pytest.mark.parametrize("free_ranges, expected", [
    (((20, 32),), True),
    (((20, 6),), False),
    (((20, 2), (25, 4)), False),
])
@pytest.mark.parametrize("strategy, halfplr", [("DataParallel", 0), ("StreamK", 0)])
def test_packed_carrier_preflight_preserves_existing_budget(monkeypatch, free_ranges, expected, strategy, halfplr):
    kernel, tensors, writer = _writer()
    kernel.update(TileProcessingStrategy=strategy, HalfPLR=halfplr,
                  NoTailLoop=True, SuppressNoLoadLoop=bool(halfplr), LoopIters=2)
    writer.states.papNextTileVgpr = 12
    # Eligibility is separate from the pool check: exercise the real allocator
    # with enough space, insufficient space, and fragmented aligned scratch.
    monkeypatch.setattr(writer, "papDescriptorSnapshotLayout", lambda *args: ("eligible",))
    writer.states.regCaps = {"MaxSgpr": 106, "MaxVgpr": 1024}
    writer.sgprPool = RegisterPool(98, RegisterType.Sgpr, defaultPreventOverflow=False)
    writer.vgprPool = RegisterPool(8, RegisterType.Vgpr, defaultPreventOverflow=False)
    writer.vgprPool.add(0, 8, "scratch")
    for start, count in free_ranges:
        writer.sgprPool.add(start, count, "scratch")
    before = [(slot.status, slot.tag) for slot in writer.sgprPool.getPool()]
    plan = writer.papAddressCarrierPlan(kernel, tensors["A"], tensors["B"])
    assert plan == ({"packed": True} if expected else None)
    assert [(slot.status, slot.tag) for slot in writer.sgprPool.getPool()] == before
    assert writer.sgprPool.size() == 98

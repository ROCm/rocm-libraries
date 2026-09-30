# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Execute PAP's preparation/handoff/issue boundary without a GPU.

Descriptor arithmetic is represented by a wave-parity branch with temporary
registers. The real reservation, identity lifetime, and cluster handoff run on
the existing strict control-flow interpreter. This does not model GPU timing.
"""

import pytest
from rocisa.code import Label, Module
from rocisa.container import sgpr, vgpr
from rocisa.instruction import (SAddU32, SBitcmp1B32, SBranch, SCBranchSCC0,
                                SCBranchSCC1, SCmpEQU32, SMovB32, SOrB32)

from Tensile.Component import Component
from Tensile.Components.PersistentLoop import PersistentLoopOn
from Tensile.KernelWriterAssembly import KernelWriterAssembly
from _pap_isa import Machine
from test_data_parallel_pap_multicast import _setup as _multicast_setup

pytestmark = pytest.mark.unit


class _LowVgprMachine(Machine):
    def run(self, module):
        # Native rocisa may print this encoding selector even though these
        # tests use only low numeric VGPRs and the low Serial register. Accept
        # only selector zero; a high-register change needs an extended model.
        lines = []
        for line in str(module).splitlines():
            instruction = line.split("//", 1)[0].strip()
            if instruction.startswith("s_set_vgpr_msb"):
                assert instruction == "s_set_vgpr_msb 0"
            else:
                lines.append(line)
        return super().run("\n".join(lines))


@pytest.fixture(autouse=True)
def _isa(_isolate_rocisa_state):
    from gpu_test_helpers import init_rocisa
    init_rocisa(target="gfx1250", wavesize=32)


def _setup(monkeypatch, multicast, halfplr, *, schedule=4, eligible=True):
    kernel, assignment, writer = _multicast_setup(monkeypatch, (2, 4) if multicast else (1, 1))
    kernel.update(enableTDMA=True, enableTDMB=True, NumWaves=4, TDMFuse=0,
                  ScheduleIterAlg=schedule, HalfPLR=halfplr, NoTailLoop=False,
                  TDMSplit=not eligible, TDMPlusLdsBuf=0, StaggerU=0)
    kernel["ProblemType"].update(MXBlockA=32, MXBlockB=32, Sparse=0)
    writer.papTdmSupportsIncrementalUpdate = KernelWriterAssembly.papTdmSupportsIncrementalUpdate
    processing = Component.TileProcessingStrategy.find(writer)

    def mapping(*args, **kwargs):
        module = Module("observable next identity")
        for index, name in enumerate(writer.papTileIdentityNames(kernel)):
            module.add(SMovB32(sgpr(name), 100 + index))
        module.add(SAddU32(sgpr("Mapped"), sgpr("Mapped"), 1))
        return module

    def descriptors(kernel, a, b, preservePapBank=True):
        module = Module("observable descriptor parity preparation")
        if not preservePapBank:
            module.add(SAddU32(sgpr("TailRestored"), sgpr("TailRestored"), 1))
            return module
        odd = Label(writer.labels.getNameInc("DescriptorOdd"), "")
        joined = Label(writer.labels.getNameInc("DescriptorJoined"), "")
        # Real descriptor preparation needs at least these three temporary
        # SGPRs inside the same protected identity lifetime.
        with writer.allocTmpSgpr(3, tag="descriptor preparation") as tmp:
            module.add(SMovB32(sgpr(tmp.idx), 17))
            module.add(SBitcmp1B32(sgpr("WaveIdx"), 0))
            module.add(SCBranchSCC1(odd.getLabelName()))
            module.add(SMovB32(sgpr("DescriptorChoice"), 0))
            module.add(SBranch(joined.getLabelName()))
            module.add(odd)
            module.add(SMovB32(sgpr("DescriptorChoice"), 1))
            module.add(joined)
            module.add(SAddU32(sgpr("DescriptorJoined"), sgpr("DescriptorJoined"), 1))
        return module

    def issue(*args, **kwargs):
        module = Module("observable first PGR issue")
        module.add(SCmpEQU32(sgpr("LoopCounter"), 999, "establish first-PGR predicate"))
        module.add(SCBranchSCC0("BadIssuePredicate"))
        module.add(SAddU32(sgpr("Loads"), sgpr("Loads"), 1))
        module.add(SAddU32(sgpr("TensorToken"), sgpr("TensorToken"), 1))
        module.add(SMovB32(sgpr("PersistentPrefetchState"), 1))
        return module

    def save_bank(kernel):
        module = Module("observable PAP bank encoding")
        with writer.allocTmpSgpr(1, tag="save PAP bank") as tmp:
            module.add(SMovB32(sgpr(tmp.idx), sgpr("BankFlag")))
            module.add(SOrB32(sgpr("PersistentPrefetchState"), sgpr("PersistentPrefetchState"), sgpr(tmp.idx)))
        return module

    monkeypatch.setattr(processing, "prefetchAcrossPersistentSetupNextTile", mapping)
    writer.papTdmUpdateDescriptor = descriptors
    writer.setupPrefetchAcrossPersistentLoads = issue
    writer.papTdmSaveLdsBank = save_bank
    tensor = {"MX": {}}
    module = PersistentLoopOn().prefetch(writer, kernel, tensor, tensor)
    return kernel, assignment, writer, module


def _machine(writer, kernel, wave, *, has_next=True, state=0, bank=0):
    machine = _LowVgprMachine(NextTile=8, TotalTiles=32 if has_next else 8,
                      PersistentPrefetchState=state, WaveIdx=wave,
                      LoopCounter=16, OrigLoopCounter=17, BankFlag=bank)
    machine.registers[str(vgpr("Serial"))] = wave * 32
    for index, name in enumerate(writer.papTileIdentityNames(kernel)):
        machine[name] = 10 + index
    return machine


@pytest.mark.parametrize("multicast", [False, True])
@pytest.mark.parametrize("halfplr", [False, True])
@pytest.mark.parametrize("wave", range(4))
@pytest.mark.parametrize("bank", [0, 2])
def test_preparation_and_handoff_follow_supported_order(monkeypatch, multicast, halfplr, wave, bank):
    kernel, assignment, writer, module = _setup(monkeypatch, multicast, halfplr)
    machine = _machine(writer, kernel, wave, bank=bank)
    identity = {name: machine[name] for name in writer.papTileIdentityNames(kernel)}
    assert machine.run(module) is None
    assert machine["Mapped"] == 1 and machine["DescriptorJoined"] == 2
    assert machine["DescriptorChoice"] == wave % 2
    assert machine["Loads"] == machine["TensorToken"] == 1
    assert machine["PersistentPrefetchState"] == 1 | bank
    assert {name: machine[name] for name in identity} == identity
    if halfplr:
        assert (machine["LoopCounter"], machine["OrigLoopCounter"]) == (16, 17)
    assert machine["TailRestored"] == (0 if halfplr else 2)
    expected = [("s_barrier_signal", "-1"), ("s_barrier_wait", "-1")]
    if multicast:
        if wave == 0:
            expected.append(("s_barrier_signal", "-3"))
        expected.append(("s_barrier_wait", "-3"))
    assert machine.barrier_operations == expected
    # Primed re-entry and close must not issue, advance the token, or release
    # a cluster phase twice.
    machine.run(module)
    machine.run(assignment.persistentClusterNextTileArrive(writer, kernel))
    assert machine["Mapped"] == 1
    assert machine["Loads"] == machine["TensorToken"] == 1
    assert machine.barrier_operations == expected
    text = str(module)
    handoff = text.index("s_wait_dscnt" if multicast else "s_barrier_signal -1")
    if multicast and not halfplr:
        assert "early handoff for classic multicast (native Rule3 segment limit)" in text
        assert handoff < text.index("s[sgprMapped]")
    else:
        assert text.rindex("s[sgprDescriptorJoined]") < handoff
        assert text.index("sgprOrigLoopCounter], 998") < handoff
    assert handoff < text.index("establish first-PGR predicate") < text.index("s[sgprLoads]")


@pytest.mark.parametrize("multicast", [False, True])
@pytest.mark.parametrize("has_next,state", [(False, 0), (True, 1), (True, 3)])
def test_no_preparation_or_handoff_when_issue_is_skipped(monkeypatch, multicast, has_next, state):
    kernel, _, writer, module = _setup(monkeypatch, multicast, False)
    machine = _machine(writer, kernel, 0, has_next=has_next, state=state)
    assert machine.run(module) is None
    assert machine["Mapped"] == machine["DescriptorJoined"] == machine["Loads"] == 0
    assert machine.barrier_operations == []


@pytest.mark.parametrize("multicast", [False, True])
@pytest.mark.parametrize("schedule,eligible", [(0, True), (4, False)])
def test_unsupported_paths_keep_early_handoff(monkeypatch, multicast, schedule, eligible):
    _, _, _, module = _setup(monkeypatch, multicast, False, schedule=schedule, eligible=eligible)
    text = str(module)
    assert text.index("s_barrier_signal -1") < text.index("s[sgprMapped]")


@pytest.mark.parametrize("multicast", [False, True])
@pytest.mark.parametrize("halfplr", [False, True])
def test_delayed_election_reuses_preparation_scratch(monkeypatch, multicast, halfplr):
    _, _, early_writer, _ = _setup(monkeypatch, multicast, halfplr, schedule=0)
    _, _, late_writer, _ = _setup(monkeypatch, multicast, halfplr, schedule=4)
    assert late_writer.sgprPool.size() <= early_writer.sgprPool.size()
    assert late_writer.vgprPool.size() <= early_writer.vgprPool.size()

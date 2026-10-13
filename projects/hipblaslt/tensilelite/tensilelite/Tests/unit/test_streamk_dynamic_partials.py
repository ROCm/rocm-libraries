# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Execute dynamic-queue K-split partial indexing and alpha==0 ownership on the CPU."""

from types import SimpleNamespace

import pytest
from rocisa.code import Module
from rocisa.container import sgpr
from rocisa.enum import RegisterType
from rocisa.instruction import SCBranchSCC0, SCmpEQU32, SMovB32
from rocisa.register import RegisterPool

import Tensile.Components.StreamK as streamk
from Tensile.Components.StreamK import StreamKDynamic, StreamKHybrid
from _streamk_dynamic_isa import Machine

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _gfx942_instruction_set(_isolate_rocisa_state):
    from gpu_test_helpers import init_rocisa

    init_rocisa(target="gfx942", wavesize=64)


@pytest.fixture(autouse=True)
def _scalar_alpha_branch(monkeypatch):
    # The interpreter models scalar control flow only; branch on an integer
    # Alpha instead of the VALU float compare.
    def branch_if_not_zero(name, _data_type, label):
        module = Module("scalar alpha test")
        module.add(SCmpEQU32(src0=sgpr(name), src1=0))
        module.add(SCBranchSCC0(labelName=label.getLabelName()))
        return module

    monkeypatch.setattr(streamk, "BranchIfNotZero", branch_if_not_zero)


class _Labels:
    def __init__(self):
        self.count = 0

    def getNameInc(self, name):
        self.count += 1
        return name + str(self.count)


class _Writer:
    def __init__(self):
        self.labels = _Labels()
        self.states = SimpleNamespace(archCaps={"WorkGroupIdFromTTM": False})
        self.sgprPool = RegisterPool(0, RegisterType.Sgpr, defaultPreventOverflow=False, printRP=False)
        self.vgprPool = RegisterPool(0, RegisterType.Vgpr, defaultPreventOverflow=False, printRP=False)

    def longBranchScc0(self, label, posNeg):
        return SCBranchSCC0(labelName=label.getLabelName())


def _kernel():
    return {"WavefrontSize": 64,
            "ProblemType": {"NumIndicesC": 3, "NumIndicesFree": 2, "ComputeDataType": SimpleNamespace(toEnum=lambda: None)}}


def _activate(strategy, writer, kernel):
    item = writer.sgprPool.checkOut(1, "work item")
    module = Module("dynamic work item activation")
    module.add(SMovB32(sgpr(item), sgpr("WorkItem")))
    module.add(strategy.activateWorkItem(writer, kernel, {}, {}, item))
    return module


@pytest.mark.parametrize("strategy,sk_tiles_name", [(StreamKDynamic(), "skTiles"), (StreamKHybrid(), "SKTiles")])
@pytest.mark.parametrize("tiles_m,tiles_n,batches,sk_tiles,sk_split,ipt", [
    (2, 2, 1, 4, 2, 8), (2, 2, 3, 12, 2, 8), (3, 1, 2, 2, 3, 9), (2, 3, 2, 12, 4, 7),
])
def test_partial_slots_are_dense_across_batches(strategy, sk_tiles_name, tiles_m, tiles_n, batches, sk_tiles, sk_split, ipt):
    kernel, writer = _kernel(), _Writer()
    activate = _activate(strategy, writer, kernel)
    slot = strategy.calculatePartialIdx(writer, kernel, "PartialSlot")
    total_tiles = tiles_m * tiles_n * batches
    full_tiles = total_tiles - sk_tiles
    items_per_wi = -(-ipt // sk_split)
    slots, closers = [], []
    for item in range(full_tiles + sk_tiles * sk_split):
        machine = Machine(WorkItem=item, NumWorkGroups0=tiles_m, NumWorkGroups1=tiles_n,
                          SKSplit=sk_split, SKItersPerWI=items_per_wi, ItersPerTile=ipt, Alpha=1)
        machine["SizesFree+2"] = batches
        machine[sk_tiles_name] = sk_tiles
        assert machine.run(activate) is None
        tile = machine["StreamKTileIdx"]
        assert tile == machine["WorkGroup2"] * tiles_m * tiles_n + machine["WorkGroup1"] * tiles_m + machine["WorkGroup0"]
        if item < full_tiles:
            assert (machine["StreamKLocalStart"], machine["StreamKLocalEnd"]) == (0, ipt)
            continue
        machine.run(slot)
        slots.append(machine["PartialSlot"])
        if machine["StreamKLocalEnd"] == ipt:
            closers.append(tile)
    assert sorted(slots) == list(range(sk_tiles * sk_split))
    assert sorted(closers) == list(range(full_tiles, total_tiles))


@pytest.mark.parametrize("strategy", [StreamKDynamic(), StreamKHybrid()])
@pytest.mark.parametrize("alpha", [0, 0x3F800000])
def test_alpha_zero_store_belongs_to_the_tile_closer(strategy, alpha):
    kernel, writer = _kernel(), _Writer()
    alpha_zero = strategy.emitDynamicAlphaZero(writer, kernel)
    ipt, chunk = 8, 3
    for start in range(0, ipt, chunk):
        end = min(start + chunk, ipt)
        machine = Machine(Alpha=alpha, ItersPerTile=ipt, StreamKLocalStart=start, StreamKLocalEnd=end)
        exit_label = machine.run(alpha_zero)
        if alpha:
            assert exit_label is None and machine["StreamKLocalStart"] == start
        elif end == ipt:
            assert exit_label is None and machine["StreamKLocalStart"] == 0
        else:
            assert exit_label == "label_PersistentLoopClose"

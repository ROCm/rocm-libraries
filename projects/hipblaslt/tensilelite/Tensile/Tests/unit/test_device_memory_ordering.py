# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Real-emitter checks for shared device-memory publication primitives."""

import shutil
from types import SimpleNamespace

import pytest
import rocisa

from Tensile.Common import IsaVersion
from Tensile.Common.Capabilities import makeIsaInfoMap
from Tensile.Component import Component
from Tensile.Components.MemoryOrdering import (
    DeviceMemoryOrderingDefault,
    DeviceMemoryOrderingDevScopeFences,
    DeviceMemoryOrderingGfx9Xcd,
)
from Tensile.Components.WorkAssignment import WorkAssignment
from Tensile.Tests.rocisa_test_state import preserve_rocisa_kernel_state

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "version,caps,expected",
    [
        (None, {}, DeviceMemoryOrderingDefault),
        ((9, 4, 2), {}, DeviceMemoryOrderingDefault),
        ((9, 5, 0), {}, DeviceMemoryOrderingGfx9Xcd),
        ((9, 5, 0), {"HasXCDSplitL2": False}, DeviceMemoryOrderingGfx9Xcd),
        ((9, 4, 2), {"HasXCDSplitL2": True}, DeviceMemoryOrderingGfx9Xcd),
        ((12, 5, 0), {"HasInvWbDevFences": True}, DeviceMemoryOrderingDevScopeFences),
        ((9, 5, 0), {"HasInvWbDevFences": True, "HasXCDSplitL2": True},
         DeviceMemoryOrderingDevScopeFences),
    ],
)
def test_selection_needs_only_architecture(version, caps, expected):
    # Deliberately omit kernel, WorkAssignment, and TileProcessingStrategy.
    writer = SimpleNamespace(states=SimpleNamespace(archCaps=caps, version=version))
    assert type(Component.DeviceMemoryOrdering.find(writer)) is expected


@pytest.fixture
def target():
    def select(isa):
        info = makeIsaInfoMap([isa], shutil.which("amdclang++") or "/opt/rocm/bin/amdclang++")[isa]
        if not info.asmCaps["SupportedISA"]:
            pytest.skip(f"the configured assembler does not support {isa}")
        rocisa.rocIsa.getInstance().setKernel(isa, 32 if isa.major == 12 else 64)
        return SimpleNamespace(states=SimpleNamespace(
            archCaps=dict(info.archCaps), version=tuple(isa)))

    with preserve_rocisa_kernel_state():
        yield select


def _operations(module):
    return [operation for line in str(module).splitlines()
            if (operation := line.split("//", 1)[0].strip())]


@pytest.mark.parametrize("isa", [IsaVersion(9, 0, 10), IsaVersion(9, 4, 2),
                                 IsaVersion(9, 5, 0), IsaVersion(12, 5, 0)])
def test_publication_fence_sequence_and_wrappers(target, isa):
    writer = target(isa)
    memory_order = Component.DeviceMemoryOrdering.find(writer)
    streamk_order = Component.StreamKMemoryOrdering.find(writer)
    release = memory_order.releaseFence(writer)
    acquire = memory_order.acquireFence(writer)

    if isa.major == 12:
        assert _operations(release) == [
            "s_wait_loadcnt 0", "s_wait_storecnt 0", "global_wb scope:SCOPE_DEV",
            "s_wait_loadcnt 0", "s_wait_storecnt 0",
        ]
        assert _operations(acquire) == [
            "global_inv scope:SCOPE_DEV", "s_wait_loadcnt 0",
        ]
    else:
        assert _operations(release) == ["s_waitcnt vmcnt(0)"]
        assert _operations(acquire) == (["s_waitcnt vmcnt(0)"] if isa.minor == 5 else [])

    assert str(streamk_order.releaseFence(writer)) == str(release)
    assert str(streamk_order.acquireFence(writer)) == str(acquire)
    drain = memory_order.preVolatileVmem(writer, comment="before volatile access")
    assert str(streamk_order.preVolatileVmem(writer, comment="before volatile access")) == str(drain)
    assert str(WorkAssignment.preVolatileVmem(None, writer, comment="before volatile access")) == str(drain)


@pytest.mark.parametrize("requires_xcnt,enable_replay", [(False, False), (True, False),
                                                       (False, True), (True, True)])
def test_pre_volatile_drain_capabilities_are_independent(target, requires_xcnt, enable_replay):
    writer = target(IsaVersion(12, 5, 0))
    writer.states.archCaps.update(
        RequiresXCntForVolatileVMEM=requires_xcnt, EnableXnackReplay=enable_replay)
    memory_order = Component.DeviceMemoryOrdering.find(writer)
    assert isinstance(memory_order, DeviceMemoryOrderingDevScopeFences)
    drain = memory_order.preVolatileVmem(writer, comment="before volatile access")
    if requires_xcnt or enable_replay:
        assert _operations(drain) == ["s_wait_xcnt 0"]
        assert str(drain).count("before volatile access") == 1
    else:
        assert str(drain) == ""

# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Paired PAP LDS banks move each descriptor set by its own per-wave swap mask."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_PapIncrementalTdm import (_DescriptorMachine, _Writer, _descriptor_components,  # noqa: F401
                                    _full, _kernel, _machine)

pytestmark = pytest.mark.unit

_STRIDES = {"A": 0x8100, "B": 0x8200, "MXSA": 0x900, "MXSB": 0x1000}
_LOCAL_READ_VGPRS = {"A": 2, "B": 2, "MXSA": 1, "MXSB": 1}
_SETS = {"tdmAGroup0+1": "PairedTDMSwap", "tdmMXSAGroup0+1": "PairedTDMScaleSwap"}


class _Machine(_DescriptorMachine):
    """Interpret local-read address shifts as 32-bit adds on their VGPRs."""
    def run(self, module):
        return super().run("\n".join(map(self._vadd, str(module).splitlines())))

    @staticmethod
    def _vadd(line):
        op, _, operands = line.split("//", 1)[0].strip().partition(" ")
        if op not in ("v_add_nc_u32", "v_add_co_u32", "v_add_u32"):
            return line
        args = [arg.strip() for arg in operands.split(",") if arg.strip() != "vcc_lo"]
        assert len(args) == 3 and args[0] == args[1], line
        return "s_add_u32 " + ", ".join(args)


def _paired(scales, **overrides):
    kernel, tensors = _kernel("F8", scales=scales)
    kernel["LDSSegmentInterleave"] = 1
    kernel["LDSSegInterleaveOffsets"] = {
        "writeStrideBytes": 0x5000, "footprintPacked": True, "ldsBaseA": 0, "ldsBaseB": 0x1000,
        "ldsBaseMXSA": 0x10000, "ldsBaseMXSB": 0x12000, "pairedBuffers": True,
        **{"bufferStride" + tc: stride for tc, stride in _STRIDES.items()}}
    kernel.update(overrides)
    writer = _Writer(kernel)
    for tc, count in _LOCAL_READ_VGPRS.items():
        setattr(writer.states, tc.lower(), SimpleNamespace(numVgprLocalReadAddr=count))
    return kernel, tensors, writer


def _entry(kernel, tensors, writer, wave, scales):
    machine = _Machine()
    machine.__dict__.update(deepcopy(_machine(kernel, tensors, wave).__dict__))
    machine.run(_full(writer, kernel, tensors))
    sets = list(_SETS)[:2 if scales else 1]
    ping = {name: machine.get(name) for name in sets}
    masks = {name: machine.get(_SETS[name]) for name in sets}
    # Ping is the lower address of each set's pair, at this wave's buffer stride.
    for name in sets:
        assert ping[name] < ping[name] ^ masks[name]
    reads = {}
    for tc, count in _LOCAL_READ_VGPRS.items():
        for i in range(count):
            machine.put(f"v[vgprLocalReadAddr{tc}+{i}]", 0x40 * (i + 1))
            reads[f"LocalReadAddr{tc}+{i}"] = 0x40 * (i + 1)
    return machine, sets, ping, masks, reads


def _tensors_with_scales(tensors, scales):
    return {tc: tp for tc, tp in tensors.items() if scales or not tc.startswith("MX")}


@pytest.mark.parametrize("scales", [False, True])
@pytest.mark.parametrize("wave", range(4))
@pytest.mark.parametrize("bank", [0, 1])
def test_paired_pap_save_then_restore_reaches_the_prefetched_buffer(scales, wave, bank):
    kernel, tensors, writer = _paired(scales)
    machine, sets, ping, masks, reads = _entry(kernel, tensors, writer, wave, scales)
    if bank:
        for tc in ("A", "MXSA")[:len(sets)]:
            machine.run(writer.tdmSwapLdsOffset(kernel, tensors[tc]))
    machine.set("PersistentPrefetchState", 1)
    machine.run(writer.papTdmSaveLdsBank(kernel))
    assert machine.get("PersistentPrefetchState") == 1 | 2 * bank

    # The next tile's entry reprograms ping; restore moves descriptors and reads once.
    machine.run(_full(writer, kernel, tensors))
    machine.run(writer.papTdmRestoreLdsBank(kernel, tensors["A"], tensors["B"]))
    for name in sets:
        assert machine.get(name) == ping[name] ^ (masks[name] if bank else 0), name
    for tc in _tensors_with_scales(_LOCAL_READ_VGPRS, scales):
        for i in range(_LOCAL_READ_VGPRS[tc]):
            name = f"LocalReadAddr{tc}+{i}"
            assert machine.value(f"v[vgpr{name}]") == reads[name] + bank * _STRIDES[tc], name


@pytest.mark.parametrize("scales", [False, True])
@pytest.mark.parametrize("wave", range(4))
@pytest.mark.parametrize("bank", [0, 1])
def test_paired_pap_full_refresh_preserves_the_current_buffer(scales, wave, bank):
    # StaggerU takes the full initializer instead of the incremental update.
    kernel, tensors, writer = _paired(scales, StaggerU=1)
    machine, sets, ping, masks, _ = _entry(kernel, tensors, writer, wave, scales)
    if bank:
        for tc in ("A", "MXSA")[:len(sets)]:
            machine.run(writer.tdmSwapLdsOffset(kernel, tensors[tc]))
    refresh = writer.papTdmUpdateDescriptor(kernel, tensors["A"], tensors["B"])
    if scales:
        refresh.add(writer.papTdmUpdateDescriptor(kernel, tensors["MXSA"], tensors["MXSB"]))
    assert "restore PAP paired buffer after descriptor refresh" in str(refresh)
    machine.run(refresh)
    for name in sets:
        assert machine.get(name) == ping[name] ^ (masks[name] if bank else 0), name


@pytest.mark.parametrize("scales", [False, True])
@pytest.mark.parametrize("wave", range(4))
@pytest.mark.parametrize("state", [0, 1, 2, 3])
@pytest.mark.parametrize("phase", [0, 1])
def test_paired_pap_tail_writes_and_reads_the_buffer_opposite_the_prefetch(scales, wave, state, phase):
    kernel, tensors, writer = _paired(scales)
    machine, sets, ping, masks, reads = _entry(kernel, tensors, writer, wave, scales)
    if phase:
        for tc in ("A", "MXSA")[:len(sets)]:
            machine.run(writer.tdmSwapLdsOffset(kernel, tensors[tc]))
    machine.set("PersistentPrefetchState", state)
    # Only a primed prefetch into ping moves the tail to pong.
    pong = state == 1
    tail_bank = 300
    select = writer.papTdmSelectTailLdsBank(kernel, tail_bank)
    for name in sets:
        machine.run(select)
        # The tail bank setter is idempotent, so StreamK may apply it twice.
        for _ in range(2):
            machine.run(writer.papTdmSetTailLdsBank(kernel, name, tail_bank))
        assert machine.get(name) == ping[name] ^ (masks[name] if pong else 0), (name, state, phase)
    machine.run(writer.papTdmShiftTailLdsBank(kernel, tensors["A"], tensors["B"]))
    for tc in _tensors_with_scales(_LOCAL_READ_VGPRS, scales):
        for i in range(_LOCAL_READ_VGPRS[tc]):
            name = f"LocalReadAddr{tc}+{i}"
            assert machine.value(f"v[vgpr{name}]") == reads[name] + pong * _STRIDES[tc], name

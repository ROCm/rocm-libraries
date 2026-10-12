# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Execute emitted TDM K-offset arithmetic on the CPU, including 4-GiB crossings."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from Tensile.KernelWriterAssembly import KernelWriterAssembly
from rocisa.container import sgpr, vgpr
from rocisa.instruction import (
    SAddCU32, SAddU32, SMulHIU32, SMulI32, SSubU32, VReadfirstlaneB32,
)

pytestmark = pytest.mark.unit
_MASK32 = (1 << 32) - 1
# Group0's high address word also carries the image descriptor type.
_IMAGE_TYPE = 2 << 62
_CASES = [
    pytest.param(0x1234FFFFFF00, 0, 512, id="zero-offset"),
    pytest.param(0x123400001000, 17, 131072, id="no-carry"),
    pytest.param(0x1234FFFFFF00, 1, 512, id="address-carry"),
    pytest.param(0x1234FFF00000, 16, 131072, id="mx-scale-carry"),
    pytest.param(0x123400001000, 8192, 1048576, id="product-high-word"),
    pytest.param(0x1234FFFFFF00, 8193, 1048577, id="product-and-address-carry"),
    pytest.param(0x123400001000, 2, 0x80000000, id="unsigned-product"),
]


class _Writer(KernelWriterAssembly):
    """Real emission methods with only allocation and cached constants supplied."""

    def __init__(self, cached_constants=False):
        self.cached_constants = cached_constants
        self.states = SimpleNamespace(persistentConstVgprs={"ItersPerTile": 0})

    @contextmanager
    def allocTmpSgpr(self, num, alignment=None, tag=None):
        yield SimpleNamespace(idx=100)

    def acquirePersistentConstSgpr(self, kernel, name):
        return name

    def releasePersistentConstSgpr(self, name):
        pass

    def isPersistentConstantsToVgprEnabled(self, kernel):
        return self.cached_constants


def _kernel(fuse=0, data_parallel=False):
    return {
        "TileProcessingStrategy": "DataParallel" if data_parallel else "StreamK",
        "WorkAssignment": "StaticGrid",
        "TDMFuse": fuse,
        "TDMInst": 3,
        "enableTDMA": True,
        "enableTDMB": True,
        "NumWaves": 4,
        "ProblemType": {"MXBlockA": 32, "MXBlockB": 32},
    }


def _execute(module, registers):
    """Interpret the emitted instructions, comparing their result to Python integers."""
    registers = dict(registers)
    scc = 0

    def value(operand):
        text = str(operand)
        return registers[text] if text in registers else int(text, 0)

    for inst in module.flatitems():
        if not hasattr(inst, "getParams"):
            continue
        params = inst.getParams()
        dst = str(params[0])
        if isinstance(inst, VReadfirstlaneB32):
            result = value(params[1])
        else:
            a, b = value(params[1]), value(params[2])
            if isinstance(inst, SMulHIU32):
                result = (a * b) >> 32
            elif isinstance(inst, SMulI32):
                result = a * b
            elif isinstance(inst, SAddCU32):
                result = a + b + scc
                scc = int(result > _MASK32)
            elif isinstance(inst, SAddU32):
                result = a + b
                scc = int(result > _MASK32)
            elif isinstance(inst, SSubU32):
                result = a - b
                scc = int(a >= b)
            else:
                pytest.fail(f"Unsupported instruction in address test: {inst}")
        registers[dst] = result & _MASK32
    return registers


def _set_address(registers, group, address):
    registers[str(sgpr(group + "+2"))] = address & _MASK32
    registers[str(sgpr(group + "+3"))] = (address | _IMAGE_TYPE) >> 32


def _get_address(registers, group):
    return (registers[str(sgpr(group + "+3"))] << 32) | registers[str(sgpr(group + "+2"))]


@pytest.mark.parametrize("base,index,increment", _CASES)
@pytest.mark.parametrize("fuse,pair,group", [
    (0, ("A", "B"), "tdmAGroup0"),
    (0, ("MXSA", "MXSB"), "tdmMXSAGroup0"),
    (1, ("A", "B"), "tdmAGroup0"),
    (1, ("MXSA", "MXSB"), "tdmMXSBGroup0"),
])
def test_initial_k_offset(base, index, increment, fuse, pair, group):
    registers = {
        str(sgpr("StreamKLocalStart")): index,
        str(sgpr("tdm" + "".join(pair) + "Incs")): increment,
    }
    _set_address(registers, group, base)
    module = _Writer().tdmApplyTileKOffsetWaveSeparated(
        _kernel(fuse), *({"tensorChar": tc} for tc in pair)
    )
    result = _execute(module, registers)
    assert _get_address(result, group) == _IMAGE_TYPE | (base + index * increment)


@pytest.mark.parametrize("base,index,increment", _CASES)
@pytest.mark.parametrize("fuse,owner,separate", [(2, "A", "B"), (3, "B", "A")])
def test_shared_scale_k_offset(base, index, increment, fuse, owner, separate):
    registers = {
        str(sgpr("StreamKLocalStart")): index,
        str(sgpr("tdmABIncs")): increment,
        str(sgpr("GlobalReadIncs" + separate)): increment + 256,
    }
    owner_group, separate_group = "tdm" + owner + "Group0", "tdm" + separate + "Group0"
    _set_address(registers, owner_group, base)
    _set_address(registers, separate_group, base + (1 << 32))
    writer, kernel = _Writer(), _kernel(fuse)
    module = writer.tdmApplyTileKOffsetWaveSeparated(kernel, {"tensorChar": "A"}, {"tensorChar": "B"})
    result = _execute(module, registers)
    assert _get_address(result, owner_group) == _IMAGE_TYPE | (base + index * increment)
    assert _get_address(result, separate_group) == _IMAGE_TYPE | (base + (1 << 32) + index * (increment + 256))
    # The scale call must not offset the shared descriptor a second time.
    scales = writer.tdmApplyTileKOffsetWaveSeparated(kernel, {"tensorChar": "MXSA"}, {"tensorChar": "MXSB"})
    assert _execute(scales, result) == result


@pytest.mark.parametrize("base,index,increment", _CASES)
@pytest.mark.parametrize("pair,group", [(("A", "B"), "tdmAGroup0"), (("MXSA", "MXSB"), "tdmMXSAGroup0")])
@pytest.mark.parametrize("mode", ["streamk", "dp-sgpr", "dp-vgpr"])
def test_tail_k_offset(base, index, increment, pair, group, mode):
    registers = {
        str(sgpr("tdm" + "".join(pair) + "Incs")): increment,
        str(sgpr("StreamKLocalEnd")): index + 1,
        str(sgpr("ItersPerTile")): index + 1,
        str(vgpr(0)): index + 1,
    }
    _set_address(registers, group, base)
    module = _Writer(cached_constants=mode == "dp-vgpr").tdmApplyTileTailOffsetWaveSeparated(
        _kernel(data_parallel=mode != "streamk"), *({"tensorChar": tc} for tc in pair)
    )
    result = _execute(module, registers)
    assert _get_address(result, group) == _IMAGE_TYPE | (base + index * increment)


@pytest.mark.parametrize("fuse", [0, 1, 2, 3])
@pytest.mark.parametrize("pair", [("A", "B"), ("MXSA", "MXSB")])
def test_data_parallel_initial_offset_needs_no_streamk_registers(fuse, pair):
    module = _Writer().tdmApplyTileKOffsetWaveSeparated(
        _kernel(fuse, data_parallel=True), *({"tensorChar": tc} for tc in pair)
    )
    assert _execute(module, {}) == {}

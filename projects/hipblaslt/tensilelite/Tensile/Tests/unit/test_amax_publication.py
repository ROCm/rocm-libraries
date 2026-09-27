# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host emission regressions for the output-amax publication protocol.

These inspect real generator modules and ISA-specific capability tables; they
are not GPU execution or native-assembly validation.
"""
from contextlib import contextmanager
from types import SimpleNamespace
import shutil

import pytest
import rocisa
from rocisa.container import ContinuousRegister
from Tensile.Common import IsaVersion
from Tensile.Common.Capabilities import makeIsaInfoMap
from Tensile.Common.DataType import DataType
from Tensile.KernelWriterAssembly import KernelWriterAssembly
from Tensile.Tests.rocisa_test_state import preserve_rocisa_kernel_state

pytestmark = pytest.mark.unit


class _Pool:
    def __init__(self):
        self.next = 64
        self.live = set()

    def checkOut(self, count, *args, **kwargs):
        return self.checkOutAligned(count, 1)

    def checkOutAligned(self, count, align, *args, **kwargs):
        base = (self.next + align - 1) // align * align
        self.next = base + count
        self.live.add(base)
        return base

    def checkIn(self, base):
        self.live.remove(base)


class _Writer:
    amax_intra_wave_reduction = KernelWriterAssembly.amax_intra_wave_reduction
    _shiftSrdImpl = KernelWriterAssembly._shiftSrdImpl

    def __init__(self, kernel, info):
        self.states = SimpleNamespace(
            kernel=kernel,
            version=tuple(kernel["ISA"]),
            archCaps=info.archCaps,
            asmCaps=info.asmCaps,
        )
        self.vgprPool = _Pool()
        self.sgprPool = _Pool()

    @contextmanager
    def allocTmpSgpr(self, size, alignment=1, tag=""):
        base = self.sgprPool.checkOutAligned(size, alignment)
        try:
            yield ContinuousRegister(idx=base, size=size)
        finally:
            self.sgprPool.checkIn(base)


@pytest.fixture(params=[(IsaVersion(9, 5, 0), 64), (IsaVersion(12, 5, 0), 32)])
def target(request):
    isa, wave = request.param
    compiler = shutil.which("amdclang++") or "/opt/rocm/bin/amdclang++"
    with preserve_rocisa_kernel_state():
        info = makeIsaInfoMap([isa], compiler)[isa]
        rocisa.rocIsa.getInstance().setKernel(isa, wave)
        kernel = {
            "ISA": isa,
            "WavefrontSize": wave,
            "GlobalSplitU": 1,
            "GlobalSplitUAlgorithm": "MultipleBuffer",
            "NumThreads": 4 * wave,
            "LdsBytesNoAmax": 4096,
            "ProblemType": {
                "ComputeDataType": DataType("s"),
                "DataTypeAmaxD": DataType("s"),
            },
        }
        yield _Writer(kernel, info), kernel


def test_amax_interwave_masks_match_wave_size(target):
    writer, kernel = target
    assembly = str(KernelWriterAssembly.amax_inter_wave_reduction(writer, kernel))
    if kernel["WavefrontSize"] == 32:
        assert "s_and_b32 vcc_lo" in assembly
        assert "s[sgprTmp+2]" in assembly and "s[sgprTmp+4]" in assembly
        assert "s_and_b64" not in assembly
    else:
        assert "s_and_b64 vcc" in assembly
        assert "s[sgprTmp+2:sgprTmp+2+1]" in assembly


def test_amax_publication_uses_target_atomic_fences_and_srd(target):
    writer, kernel = target
    assembly = str(KernelWriterAssembly.amax_output_result(writer, kernel))
    partial_store = (
        assembly.index("drain before amax partial store")
        if kernel["WavefrontSize"] == 32
        else assembly.index("buffer_store")
    )
    if kernel["WavefrontSize"] == 32:
        assert "s_atomic_dec" not in assembly
        atomic = assembly.index("flat_atomic_dec")
        assert partial_store < assembly.index("global_wb") < atomic
        assert (
            atomic
            < assembly.index("global_inv")
            < assembly.index("drain before amax partial load")
        )
        assert assembly.count("Shift num records for gfx125x") == 3
        assert "s_wait_xcnt" in assembly and "scope:SCOPE_DEV" in assembly
        assert "s_mov_b32 exec_lo, 1" in assembly
    else:
        atomic = assembly.index("s_atomic_dec")
        assert partial_store < atomic
        assert "flat_atomic_dec" not in assembly
        assert "global_wb" not in assembly and "global_inv" not in assembly
        assert "Shift num records for gfx125x" not in assembly
    assert not writer.vgprPool.live and not writer.sgprPool.live

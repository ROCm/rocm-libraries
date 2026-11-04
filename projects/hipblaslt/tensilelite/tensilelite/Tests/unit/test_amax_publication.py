# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host emission regressions for the output-amax publication protocol.

These inspect real generator modules and ISA-specific capability tables; they
are not GPU execution or native-assembly validation.
"""
from contextlib import contextmanager
from types import SimpleNamespace
import re
import shutil

import pytest
import rocisa
from rocisa.container import ContinuousRegister
from tensilelite.Common import IsaVersion
from tensilelite.Common.Capabilities import makeIsaInfoMap
from tensilelite.Common.DataType import DataType
from tensilelite.Common.ValidParameters import makeValidWorkGroups
from tensilelite.KernelWriterAssembly import KernelWriterAssembly
from tensilelite.Tests.rocisa_test_state import preserve_rocisa_kernel_state

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


def _interwave_schedule(num_waves, store_slot, load_slot):
    """Per iteration: (slots stored by upper waves, slots loaded by lower waves)."""
    steps, offset = [], num_waves // 2
    while offset:
        steps.append(({store_slot(w, offset) for w in range(offset, 2 * offset)},
                      {load_slot(w, offset) for w in range(offset)}))
        offset //= 2
    return steps


def test_amax_interwave_each_slot_written_once(target):
    # One barrier per iteration orders its stores before its loads, but not iteration i's loads before
    # iteration i+1's stores: a slot reused across iterations can be overwritten before it is read (WAR).
    writer, kernel = target
    wave = kernel["WavefrontSize"]
    valid_waves = {sg0 * sg1 * nsg // wave for sg0, sg1, nsg in makeValidWorkGroups() if sg0 * sg1 * nsg % wave == 0}
    # The tree halves Offset each iteration, so it only reduces power-of-two wave counts.
    for num_waves in sorted(n for n in valid_waves if n & (n - 1) == 0):
        k = {**kernel, "NumThreads": num_waves * wave}
        assembly = str(KernelWriterAssembly.amax_inter_wave_reduction(writer, k))
        assert f"s_mov_b32 s[sgprOffset], {num_waves}" in assembly
        upper = assembly[assembly.index("label_wave_upper:") : assembly.index("label_wave_lower:")]
        lower = assembly[assembly.index("label_wave_lower:") : assembly.index("label_wave_empty:")]
        # upper wave w stores slot w; lower wave w loads slot w + Offset
        assert "sgprOffset" not in upper
        assert re.search(r"v_add\w*_u32 v\[vgprTmp\], s\[sgprOffset\], v\[vgprWidx\]", lower)
        steps = _interwave_schedule(num_waves, lambda w, o: w, lambda w, o: w + o)
        stored = sorted(s for st, _ in steps for s in st)
        assert stored == list(range(1, num_waves))  # each wave but 0 stores once; slot 0 is left to the broadcast
        assert all(ld == st for st, ld in steps)


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

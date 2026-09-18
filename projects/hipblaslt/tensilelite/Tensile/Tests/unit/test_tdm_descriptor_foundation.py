# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""PAP descriptor ownership, optimization eligibility, and SGPR lifetime."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from rocisa.enum import RegisterType
from rocisa.register import RegisterPool
from rocisa.container import RegisterContainer, sgpr
from rocisa.instruction import SAndB32

import Tensile.KernelWriterAssembly as kwa_module
from Tensile.KernelWriter import KernelWriter
from Tensile.Common import INDEX_CHARS
from Tensile.Common.DataType import DataType

pytestmark = pytest.mark.unit


def _kernel(**overrides):
    kernel = {
        "ProblemType": {
            "DataTypeA": DataType("F8N"),
            "DataTypeB": DataType("F8N"),
            "DataTypeMXSA": DataType("I8"),
            "DataTypeMXSB": DataType("I8"),
            "MXBlockA": 32,
            "MXBlockB": 32,
            "Sparse": 0,
            "IndicesSummation": [3],
            "Batched": True,
            "StridedBatched": True,
            "SupportUserArgs": False,
            "GroupedGemm": False,
        },
        "MacroTile0": 256,
        "MacroTile1": 256,
        "DepthU": 256,
        "MatrixInstK": 128,
        "NumWaves": 4,
        "WavefrontSize": 32,
        "ClusterDim": [1, 1, 1],
        "Multicast": 0,
        "TDMSplit": 0,
        "LdsOffsetA": 0,
        "LdsOffsetB": 34816,
        "LdsOffsetMXSA": 69632,
        "LdsOffsetMXSB": 71680,
        "LdsBlockSizePerPadA": 256,
        "LdsBlockSizePerPadB": 256,
        "LdsBlockSizePerPadMXSA": 0,
        "LdsBlockSizePerPadMXSB": 0,
        "LdsPadA": 16,
        "LdsPadB": 16,
        "LdsPadMXSA": 0,
        "LdsPadMXSB": 0,
        "LdsOffsetA_Blk": 0x23000,
        "LDSSegmentInterleave": 0,
        "GlobalSplitU": 0,
        "enableTDMA": True,
        "enableTDMB": True,
        "StreamK": 3,
        "StreamKForceDPOnly": 1,
        "PrefetchAcrossPersistent": 1,
        "PrefetchGlobalRead": 2,
        "SuppressNoLoadLoop": False,
        "HalfPLR": 0,
        "NoTailLoop": True,
        "ReuseAcrossPersistent": 0,
        "UseCustomMainLoopSchedule": 0,
        "UseSubtileImpl": False,
        "_TDMIterateModeA": False,
        "_TDMIterateModeB": False,
    }
    kernel.update(overrides)
    return kernel


def _tensor_parameters(tc, kernel, tlu):
    return {
        "tensorChar": tc,
        "tileChar": tc[-1],
        "idx": 0 if tc.endswith("A") else 1,
        "tlu": tlu,
        "isA": tc == "A",
        "isB": tc == "B",
        "isM": False,
        "bpeGR": kernel["ProblemType"][f"DataType{tc}"].numBytes(),
        # (free, summation, batch) index assignments; ia[2] names the batch stride.
        "ia": [0 if tc.endswith("A") else 1, 3, 2],
    }


class _DescriptorWriter:
    """Minimal writer surface used by initTDMDescriptorWaveSeparatedImpl."""

    _setTDMGlobalAddr = kwa_module.KernelWriterAssembly._setTDMGlobalAddr
    _resolveTDMGlobalAddr = kwa_module.KernelWriterAssembly._resolveTDMGlobalAddr
    tdmEmitWaveCompId = kwa_module.KernelWriterAssembly.tdmEmitWaveCompId

    def __init__(self):
        self._next_tmp_sgpr = 200
        # Enough for Component.TensorDataMover.find() to select TensorDataMoverLoad.
        self.states = SimpleNamespace(
            asmCaps={"HasTDM": True, "HasSMulHi": True}, archCaps={},
            kernel={"TDMInst": 3}, indexChars=INDEX_CHARS, laneSGPRCount=2,
        )

    @contextmanager
    def allocTmpSgpr(self, size, alignment=1, tag=""):
        base = self._next_tmp_sgpr
        self._next_tmp_sgpr += size
        try:
            yield SimpleNamespace(idx=base, size=size)
        finally:
            self._next_tmp_sgpr -= size

    def strideRef(self, tc, dim):
        return sgpr("Stride%s%u" % (tc, dim))

    def isTdmWaveSeparated(self, kernel):
        return True


class _EnvelopeWriter:
    isPrefetchAcrossPersistentEnabled = KernelWriter.isPrefetchAcrossPersistentEnabled
    papTdmHoistInvariantEnabled = KernelWriter.papTdmHoistInvariantEnabled
    papTdmCachedAddrBaseEnabled = KernelWriter.papTdmCachedAddrBaseEnabled
    papTdmPinnedDescriptorSgprs = KernelWriter.papTdmPinnedDescriptorSgprs


@pytest.mark.parametrize("tc,tlu", [
    ("A", 0), ("B", 0), ("A", 1), ("B", 1), ("MXSA", 0), ("MXSB", 0),
])
def test_tile_refresh_preserves_invariant_descriptor_fields(tc, tlu):
    kernel, writer = _kernel(), _DescriptorWriter()
    tensor = _tensor_parameters(tc, kernel, tlu)
    invariant = kwa_module.KernelWriterAssembly.initTDMDescriptorWaveSeparatedImpl(
        writer, kernel, tensor, emitVariant=False)
    refresh = kwa_module.KernelWriterAssembly.initTDMDescriptorWaveSeparatedImpl(
        writer, kernel, tensor, emitInvariant=False)

    # Launch-time setup must not depend on the current persistent tile.
    assert "sgprWorkGroup" not in str(invariant)

    # Refresh owns the LDS anchor, global address, and residual extent. Other
    # descriptor fields must survive the persistent transition untouched.
    descriptor_writes = set()
    for instruction in refresh.flatitems():
        dst = getattr(instruction, "dst", None)
        if isinstance(dst, RegisterContainer) and dst.regName is not None:
            name = dst.regName.name
            if name.startswith(f"tdm{tc}Group"):
                descriptor_writes.add((name, dst.regName.getTotalOffsets()))
    dim0 = tlu or tc.startswith("MX")
    extent_words = (1, 2) if dim0 else (2, 3)
    assert descriptor_writes == {
        (f"tdm{tc}Group0", 1), (f"tdm{tc}Group0", 2), (f"tdm{tc}Group0", 3),
        *((f"tdm{tc}Group1", word) for word in extent_words),
    }
    # Packed extent words share storage with invariant fields: clear only the
    # refreshed half, so an edge extent can shrink or grow on the next tile.
    for word, mask in zip(extent_words, ("0xffff", "0xffff0000")):
        assert any(
            isinstance(instruction, SAndB32)
            and str(instruction.dst) == str(sgpr(f"tdm{tc}Group1+{word}"))
            and str(instruction.srcs[1]) == mask
            for instruction in refresh.flatitems()
        )


@pytest.mark.parametrize("override", [
    {"PrefetchAcrossPersistent": 0}, {"StreamKForceDPOnly": 0},
    {"NoTailLoop": False}, {"HalfPLR": 1}, {"ReuseAcrossPersistent": 1},
    {"ClusterDim": [2, 1]}, {"UseSubtileImpl": True},
    {"_TDMIterateModeA": True}, {"PrefetchGlobalRead": 1},
])
def test_other_paths_keep_full_descriptors_and_no_persistent_cache(override):
    writer, kernel = _EnvelopeWriter(), _kernel(**override)
    assert not writer.papTdmHoistInvariantEnabled(kernel)
    assert not writer.papTdmCachedAddrBaseEnabled(kernel)
    assert writer.papTdmPinnedDescriptorSgprs(kernel) == []


@pytest.mark.parametrize("support_user_args,batched,grouped,cache", [
    (False, True, False, True), (True, True, False, False),
    (True, False, False, True), (True, True, True, True),
])
def test_cache_eligibility_preserves_runtime_batch_pointer_resolution(
    support_user_args, batched, grouped, cache
):
    kernel = _kernel()
    kernel["ProblemType"].update(SupportUserArgs=support_user_args,
                                Batched=batched, GroupedGemm=grouped)
    writer = _EnvelopeWriter()
    assert writer.papTdmHoistInvariantEnabled(kernel)
    assert writer.papTdmCachedAddrBaseEnabled(kernel) == cache
    assert writer.papTdmPinnedDescriptorSgprs(kernel) == [
        "tdmAGroup0", "tdmAGroup1", "tdmMXSAGroup0", "tdmMXSAGroup1"]


@pytest.mark.parametrize("pap", [0, 1])
def test_end_summation_keeps_descriptor_ranges_out_of_store_scratch(pap):
    class PoolWriter(_EnvelopeWriter):
        defineSgpr = kwa_module.KernelWriterAssembly.defineSgpr
        defineSgprIdx = kwa_module.KernelWriterAssembly.defineSgprIdx
        undefineSgpr = kwa_module.KernelWriterAssembly.undefineSgpr
        _undefDirective = kwa_module.KernelWriterAssembly._undefDirective

    kernel = _kernel(PrefetchAcrossPersistent=pap, StorePriorityOpt=False,
                     BufferStore=True, EnableMatrixInstruction=False,
                     _GlobalAccumulation=None, AdaptiveGemmGSUA=0)
    kernel["ProblemType"].update(Gradient=False, DestDataType=DataType("F8N"),
                                StochasticRounding=False, ActivationType="none",
                                UseScaleAB=False, UseScaleAlphaVec=False,
                                UseE=False, UseScaleCD=False)
    writer = PoolWriter()
    writer.sgprs = {}
    writer.sgprPool = RegisterPool(0, RegisterType.Sgpr, defaultPreventOverflow=False)
    writer.states = SimpleNamespace(
        freeSgprVarPool=[], nonPostLoopSgpr=[], rapDeferSgprUndef=False,
        numStoreSgprToLoad=0, numStoreSgprToLoad2=0, doShadowInit=False,
        useBias=kwa_module.DataDirection.NONE, useGateResidual=False)
    writer.db = {"InitVgpr": 0, "InitSgpr": 0, "ConservativeWaitCnt": 0}
    retained = set()
    for tag, size in (("tdmAGroup0", 4), ("tdmAGroup1", 8),
                      ("tdmMXSAGroup0", 4), ("tdmMXSAGroup1", 8)):
        writer.defineSgpr(tag, size, 4)
        retained.update(range(writer.sgprs[tag], writer.sgprs[tag] + size))
    writer.defineSgpr("CurrentTileScratch", 8, 4)

    # Execute the actual release loop and the actual SrdC/SrdD allocation that
    # follows it. With PAP off the same pool must reuse those descriptor slots.
    kwa_module.KernelWriterAssembly.endSummation(
        writer, kernel, {}, {}, label="Summation_End")
    store = set()
    for name in ("SrdC", "SrdD"):
        store.update(range(writer.sgprs[name], writer.sgprs[name] + 4))
    if pap:
        assert retained.isdisjoint(store)
        pool = writer.sgprPool.getPool()
        assert all(pool[index].status == RegisterPool.Status.InUse for index in retained)
    else:
        assert retained & store

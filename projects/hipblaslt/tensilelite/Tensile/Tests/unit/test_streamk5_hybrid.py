# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for StreamK=5 hybrid mode codegen intent.

These tests import Tensile modules directly and inspect emitted rocisa
instructions / signature metadata rather than matching Python source text.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Prime the component registry before StreamK imports (avoids circular import).
from Tensile.KernelWriterAssembly import KernelWriterAssembly  # noqa: F401

from rocisa.code import Module, RegSet, SignatureBase
from rocisa.instruction import SAndB32, SLShiftRightB32

from Tensile.Common.DataType import DataType
from Tensile.Common.ValidParameters import validParameters
from Tensile.Components.Signature import SignatureDefault
from Tensile.Components.StreamK import (
    StreamK,
    StreamKHybrid,
    StreamKTwoTileDPFirst,
    streamKVariantClass,
)

SK3_KERNARG_NAMES = [
    "ItersPerTile",
    "MagicNumberItersPerTile",
    "MagicShiftItersPerTile",
    "SKItersPerWG",
    "skGrid",
    "skTiles",
]

SK5_KERNARG_ALIASES = [
    ("sgprTotalItems", "sgprMagicNumberItersPerTile", 0),
    ("sgprSKTiles", "sgprMagicShiftItersPerTile", 0),
    ("sgprSKSplit", "sgprSKItersPerWG", 0),
    ("sgprSKItersPerWI", "sgprskGrid", 0),
    ("sgprSKGrid", "sgprskTiles", 0),
]

SK5_PERSISTENT_ALIASES = [
    ("sgprStreamKIter", "sgprStreamKTileIdx", 0),
    ("sgprStreamKIterEnd", "sgprStreamKPartialIdx", 0),
]

STREAMK_ARG_NAMES = {
    "ItersPerTile",
    "MagicNumberItersPerTile",
    "MagicShiftItersPerTile",
    "SKItersPerWG",
    "skGrid",
    "skTiles",
    "TotalItems",
    "SKTiles",
    "SKSplit",
    "SKItersPerWI",
    "SKGrid",
}


class _StopAfterSk5Aliases(Exception):
    """Raised once KernelWriterAssembly emits the SK5 persistent aliases."""


def _mock_writer_for_component(streamk: int) -> MagicMock:
    writer = MagicMock()
    writer.states = SimpleNamespace(kernel={"StreamK": streamk})
    return writer


def _make_signature_writer(streamk: int) -> MagicMock:
    f32 = DataType("Float")
    writer = MagicMock()
    writer.debugConfig = SimpleNamespace(debugKernel=False)
    writer.states = SimpleNamespace(
        kernelName="test_kernel",
        rpga=2,
        bpeA=4,
        bpeCexternal=4,
        bpr=4,
        numSgprSizesFree=0,
        numSgprSizesSum=0,
        numSgprToLoad=0,
        numSgprPreload=0,
        useBias=0,
        needBiasType=False,
        d=SimpleNamespace(numSgprStrides=0),
        c=SimpleNamespace(numSgprStrides=0),
        a=SimpleNamespace(numSgprStrides=0),
        b=SimpleNamespace(numSgprStrides=0),
        m=SimpleNamespace(numSgprStrides=0),
        mxsa=SimpleNamespace(numSgprStrides=0),
        mxsb=SimpleNamespace(numSgprStrides=0),
        e=SimpleNamespace(numSgprStrides=0),
        kernel={
            "StreamK": streamk,
            "StreamKAtomic": 0,
            "LdsNumBytes": 0,
            "NumThreads": 256,
            "InternalSupportParams": {"KernArgsVersion": 0},
            "CodeObjectVersion": "V4",
            "ProblemType": {
                "UseBeta": False,
                "NumIndicesC": 2,
                "NumIndicesSummation": 1,
                "IndexAssignmentsA": [0, 2],
                "IndexAssignmentsB": [1, 2],
                "UseInitialStridesAB": True,
                "UseInitialStridesCD": True,
                "DestDataType": f32,
                "ComputeDataType": f32,
                "ActivationComputeDataType": f32,
                "DataTypeA": f32,
                "DataTypeB": f32,
                "MacDataTypeA": f32,
                "MacDataTypeB": f32,
                "MXBlockA": False,
                "MXBlockB": False,
                "Sparse": False,
                "UseScaleAB": False,
                "UseScaleCD": False,
                "UseScaleAlphaVec": 0,
                "UseE": False,
                "ActivationType": MagicMock(getAdditionalArgStringList=lambda: []),
                "OutputAmaxD": False,
            },
            "PackedC0IdxChars": [],
            "PackedC1IdxChars": [],
            "ExpertSchedulingMode": 0,
            "ESMRuntimeGate": False,
            "ActivationFused": False,
            "_GlobalAccumulation": "",
            "AdaptiveGemmGSUA": 0,
            "SubGroup0": 1,
            "SubGroup1": 1,
            "ThreadTile0": 1,
            "ThreadTile1": 1,
            "VectorWidthA": 1,
            "VectorWidthB": 1,
            "GlobalReadVectorWidthA": 1,
            "GlobalReadVectorWidthB": 1,
            "DirectToLdsA": False,
            "DirectToLdsB": False,
            "_UseSgprForGRO": False,
        },
    )
    return writer


def _streamk_arg_names_from_signature(streamk: int) -> list[str]:
    collected: list[str] = []
    orig_add_arg = SignatureBase.addArg

    def capture_add_arg(self, name, *args, **kwargs):
        collected.append(name)
        return orig_add_arg(self, name, *args, **kwargs)

    SignatureBase.addArg = capture_add_arg
    try:
        SignatureDefault()(_make_signature_writer(streamk))
    finally:
        SignatureBase.addArg = orig_add_arg

    return [name for name in collected if name in STREAMK_ARG_NAMES]


def _streamk_arg_size_delta(streamk: int) -> int:
    baseline = _make_signature_writer(0)
    SignatureDefault()(baseline)
    base_size = baseline.states.userArgsInfo.gemmArgumentSize

    writer = _make_signature_writer(streamk)
    SignatureDefault()(writer)
    return writer.states.userArgsInfo.gemmArgumentSize - base_size


def _emit_mode_extraction_module():
    writer = MagicMock()
    return StreamKHybrid._emitModeExtraction(StreamKHybrid, writer, {"StreamK": 5})


def _reg_name(reg) -> str:
    text = str(reg)
    if text.startswith("s[") and text.endswith("]"):
        return text[2:-1]
    return text


def _setup_kwa_for_sk5_aliases() -> KernelWriterAssembly:
    kwa = KernelWriterAssembly.__new__(KernelWriterAssembly)
    kwa.sgprs = {
        "ItersPerTile": 10,
        "MagicNumberItersPerTile": 11,
        "MagicShiftItersPerTile": 12,
        "SKItersPerWG": 13,
        "skGrid": 14,
        "skTiles": 15,
        "StreamKTileIdx": 20,
        "StreamKPartialIdx": 21,
        "Beta": 22,
    }
    kwa.states = SimpleNamespace(
        streamK=streamKVariantClass(5)(),
        startVgprSerial=0,
        numVgprBuffer=1,
        mxsa=SimpleNamespace(
            numVgprValu=0,
            startVgprValu=0,
            startVgprValuPack=0,
            startVgprG2L=None,
            numVgprValuPerBlock=0,
        ),
        mxsb=SimpleNamespace(
            numVgprValu=0,
            startVgprValu=0,
            startVgprValuPack=0,
            startVgprG2L=None,
            numVgprValuPerBlock=0,
        ),
        a=SimpleNamespace(
            numVgprValu=0,
            startVgprValu=0,
            startVgprValuPack=0,
            startVgprG2L=None,
            numVgprValuPerBlock=0,
            tileInfo=None,
        ),
        b=SimpleNamespace(
            numVgprValu=0,
            startVgprValu=0,
            startVgprValuPack=0,
            startVgprG2L=None,
            numVgprValuPerBlock=0,
            tileInfo=None,
        ),
        m=SimpleNamespace(numVgprValu=0),
        packDTVA=False,
        packDTVB=False,
        convDTVA=False,
        convDTVB=False,
        lrvwTileMXSA=1,
        lrvwTileMXSB=1,
        bpr=4,
        numVgprBufferPackMXSA=1,
        numVgprBufferPackMXSB=1,
    )
    return kwa


def _collect_sk5_regset_aliases() -> list[tuple[str, str, int]]:
    kernel = {
        "StreamK": 5,
        "UseSubtileImpl": True,
        "MagicDivAlg": 1,
        "ProblemType": {
            "MXBlockA": False,
            "MXBlockB": False,
            "Sparse": False,
            "IndexAssignmentsA": [0, 2],
            "IndexAssignmentsB": [1, 2],
            "IndicesFree": [0, 1],
            "IndicesSummation": [2],
            "IndicesBatch": [],
        },
        "DirectToVgprA": False,
        "DirectToVgprB": False,
        "InnerUnroll": 1,
        "LoopIters": 1,
        "UnrollMajorLDSA": True,
        "UnrollMajorLDSB": True,
        "MIInputPerThreadMXSA": 1,
        "MIInputPerThreadMXSB": 1,
        "VectorWidthMXSA": 1,
        "VectorWidthMXSB": 1,
        "MIWaveTileA": 1,
        "MIWaveTileB": 1,
    }
    tPA = {"is_sparse": False, "tpsMetadata": {}, "MX": None}
    tPB = {"is_sparse": False, "tpsMetadata": {}, "MX": None}

    captured: list = []
    orig_add = Module.add

    def patched_add(self, item):
        captured.append(item)
        if isinstance(item, RegSet) and item.name == "sgprStreamKIterEnd":
            raise _StopAfterSk5Aliases()
        return orig_add(self, item)

    kwa = _setup_kwa_for_sk5_aliases()
    try:
        with patch.object(Module, "add", patched_add):
            kwa.macroAndSet(kernel, tPA, tPB)
    except _StopAfterSk5Aliases:
        pass

    return [
        (item.name, item.ref, item.offset)
        for item in captured
        if isinstance(item, RegSet) and item.ref is not None
    ]


class TestStreamK5ValidParameters:
    def test_streamk_enum_includes_5(self):
        assert 5 in validParameters["StreamK"]
        assert validParameters["StreamK"] == [0, 1, 2, 3, 4, 5]


class TestStreamK5Component:
    def test_streamk_hybrid_is_registered_variant(self):
        assert streamKVariantClass(5) is StreamKHybrid
        assert StreamKHybrid.kernel == {"StreamK": 5}

    def test_component_dispatches_streamk_5_to_hybrid(self):
        impl = StreamK.find(_mock_writer_for_component(5))
        assert isinstance(impl, StreamKHybrid)

    def test_component_dispatches_streamk_3_to_static_path(self):
        impl = StreamK.find(_mock_writer_for_component(3))
        assert isinstance(impl, StreamKTwoTileDPFirst)


class TestStreamK5ModeExtraction:
    def test_mode_extraction_shifts_bit_30(self):
        module = _emit_mode_extraction_module()
        shift_inst = next(
            inst for inst in module.flatitems() if isinstance(inst, SLShiftRightB32)
        )
        params = list(shift_inst.getParams())
        assert _reg_name(params[0]) == "sgprStreamKHybridMode"
        assert _reg_name(params[1]) == "sgprMagicShiftItersPerTile"
        assert params[2] == hex(30)

    def test_mode_extraction_masks_magic_shift_with_bfffffff(self):
        module = _emit_mode_extraction_module()
        mask_insts = [
            inst for inst in module.flatitems() if isinstance(inst, SAndB32)
        ]
        clear_inst = next(
            inst
            for inst in mask_insts
            if _reg_name(list(inst.getParams())[0]) == "sgprMagicShiftItersPerTile"
        )
        params = list(clear_inst.getParams())
        assert params[2] == hex(0xBFFFFFFF)

    def test_mode_extraction_does_not_use_bit_31(self):
        module = _emit_mode_extraction_module()
        for inst in module.flatitems():
            if isinstance(inst, SLShiftRightB32):
                assert list(inst.getParams())[2] != hex(31)


class TestStreamK5Signature:
    def test_signature_emits_six_sk3_named_args(self):
        assert _streamk_arg_names_from_signature(5) == SK3_KERNARG_NAMES

    def test_signature_sk_arg_frame_matches_sk3(self):
        assert _streamk_arg_names_from_signature(5) == _streamk_arg_names_from_signature(3)

    def test_signature_adds_twenty_four_byte_sk_frame(self):
        assert _streamk_arg_size_delta(5) == 24
        assert _streamk_arg_size_delta(5) == _streamk_arg_size_delta(3)

    def test_signature_sk_frame_differs_from_sk4_names(self):
        sk4_names = _streamk_arg_names_from_signature(4)
        sk5_names = _streamk_arg_names_from_signature(5)
        assert sk4_names != sk5_names
        assert len(sk4_names) == 6
        assert len(sk5_names) == 6


class TestStreamK5RegSetAliasing:
    def test_hybrid_variant_requests_parallel_reduction_aliases(self):
        variant = streamKVariantClass(5)()
        assert variant.emitsParallelReductionSgprAliases is True

    def test_macro_and_set_emits_sk4_to_sk3_kernarg_aliases(self):
        aliases = _collect_sk5_regset_aliases()
        for expected in SK5_KERNARG_ALIASES:
            assert expected in aliases

    def test_macro_and_set_emits_persistent_slot_aliases(self):
        aliases = _collect_sk5_regset_aliases()
        for expected in SK5_PERSISTENT_ALIASES:
            assert expected in aliases

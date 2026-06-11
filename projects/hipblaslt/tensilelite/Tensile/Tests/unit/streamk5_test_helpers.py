# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Shared helpers for StreamK=5 hybrid unit tests."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from Tensile.KernelWriterAssembly import KernelWriterAssembly
from Tensile.Components.StreamK import StreamKHybrid, streamKVariantClass

from rocisa.code import Module, RegSet

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


class _StopAfterSk5Aliases(Exception):
    """Raised once KernelWriterAssembly emits the SK5 persistent aliases."""


@pytest.fixture
def mock_streamk_writer():
    """Factory fixture: mock_streamk_writer(5) -> writer with StreamK=5."""

    def _factory(streamk: int) -> MagicMock:
        writer = MagicMock()
        writer.states = SimpleNamespace(kernel={"StreamK": streamk})
        return writer

    return _factory


def emit_mode_extraction_module():
    writer = MagicMock()
    return StreamKHybrid._emitModeExtraction(StreamKHybrid, writer, {"StreamK": 5})


def reg_name(reg) -> str:
    text = str(reg)
    if text.startswith("s[") and text.endswith("]"):
        return text[2:-1]
    return text


def setup_kwa_for_sk5_aliases() -> KernelWriterAssembly:
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


def collect_sk5_regset_aliases() -> list[tuple[str, str, int]]:
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

    kwa = setup_kwa_for_sk5_aliases()
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

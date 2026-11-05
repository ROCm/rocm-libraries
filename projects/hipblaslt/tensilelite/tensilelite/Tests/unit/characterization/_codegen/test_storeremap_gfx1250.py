# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os

import pytest

from codegen_harness import emit_kernels_from_logic
from config_harness import assert_assembles

from tensilelite.SolutionStructs.Solution import storeRemapMultiColumn

pytestmark = pytest.mark.unit

_DATA = os.path.join(os.path.dirname(__file__), "data", "test_data", "_logic", "gfx1250")
_NN = os.path.join(_DATA, "BBS_BH_Bias_SAV_SRVW4_NN.yaml")
_TN = os.path.join(_DATA, "BBS_BH_Bias_SAV_SRVW4_TN.yaml")


@pytest.mark.parametrize(
    "mt0, srvw, wave, expected",
    [
        (192, 4, 32, True),   # 48 lanes per column, more than a wave
        (256, 4, 32, True),   # 64 lanes per column
        (128, 4, 32, False),  # 32 lanes, one column per store: the old path
        (64, 4, 32, False),   # 16 lanes, two columns per store: the old path
        (192, 0, 32, False),  # no StoreRemap
    ],
)
def test_store_remap_multi_column(mt0, srvw, wave, expected):
    state = {"MacroTile0": mt0, "StoreRemapVectorWidth": srvw, "WavefrontSize": wave}
    assert storeRemapMultiColumn(state) is expected


def _emit(path):
    results = emit_kernels_from_logic(path, canonical=False)
    assert results, "no kernel emitted"
    kernels = {}
    for base, src, err in results:
        assert err == 0, base
        assert_assembles(src, base)
        kernels[src[src.index(".globl ") :].split(None, 2)[1]] = src
    return kernels


@pytest.fixture(scope="module")
def nn_kernels():
    return _emit(_NN)


@pytest.fixture(scope="module")
def tn_kernels():
    return _emit(_TN)


def _one(kernels, tag):
    found = [src for name, src in kernels.items() if tag in name]
    assert len(found) == 1, (tag, list(kernels))
    return found[0]


def _assert_whole_line_remap(src):
    # SRVW 4 x bf16 = 8 B per lane, so every remap store is a b64 that covers 128 B lines with 16 lanes,
    # and only those stores are streaming
    remap_stores = src.count("buffer_store_b64")
    assert remap_stores > 0
    assert src.count("th:TH_STORE_NT") == remap_stores
    # whether a column starts on a line depends on ld, so a full tile checks D at run time first
    assert "GW_NotLineAligned" in src
    assert "StoreRemap Start" in src
    # with StoreRemap the bias / scale-alpha-vec LDS writes sit inside the A / B buffers, and the
    # StoreRemap Start barrier is only in the remap block now, so readVectorToLDS has to keep its own
    assert "wait for all global loads." in src


def test_nn_srvw4_stores_whole_lines(nn_kernels):
    src = _one(nn_kernels, "_SRVW4_")
    _assert_whole_line_remap(src)
    # edge tiles, beta and lines that arent aligned keep the direct stores of the SRVW 0 kernel
    assert src.count("buffer_store_b128") == _one(nn_kernels, "_SRVW0_").count("buffer_store_b128")


def test_nn_srvw0_twin_is_unchanged_path(nn_kernels):
    src = _one(nn_kernels, "_SRVW0_")
    assert "StoreRemap Start" not in src
    assert "th:TH_STORE_NT" not in src
    assert "GW_NotLineAligned" not in src


def test_tn_vector_width_srvw4_stores_whole_lines(tn_kernels):
    name = next(iter(tn_kernels))
    assert "_VWA2_VWB8_" in name and "_SRVW4_" in name
    _assert_whole_line_remap(tn_kernels[name])

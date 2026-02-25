################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

import pytest
from unittest.mock import MagicMock

from Tensile.Components.CustomSchedule import (
    hasCustomSchedule, ScheduleInfo, RegisterSchedule, TileConfig,
    _SCHEDULE_METADATA, _SCHEDULE_REGISTRY,
    isTN, isNT, isNN, isTT, is16bit, query_cms_kernels
)
from Tensile.Components.CMSValidator import isValid
from Tensile.Common import IsaVersion

# Helper to create a mock data type
def _mock_dtype(is_16bit=False, is_8bit=False, num_bytes=4):
    mock = MagicMock()
    mock.isHalf.return_value = is_16bit
    mock.isBFloat16.return_value = False # Assuming isHalf is enough for is16bit
    mock.isInt8.return_value = is_8bit
    mock.is8bitFloat.return_value = False # Assuming isInt8 is enough for is8bit
    mock.numBytes.return_value = num_bytes
    return mock

# Base kernel configuration factory
def create_base_kernel():
    kernel = {
        "UseCustomMainLoopSchedule": True,
        "EnableMatrixInstruction": True,
        "UnrollLoopSwapGlobalReadOrder": False,
        "ISA": IsaVersion(9,5,0),
        "WavefrontSize": 64,
        "ProblemType": {
            "DataType": _mock_dtype(),
            "DataTypeA": _mock_dtype(),
            "DataTypeB": _mock_dtype(),
            "TransposeA": False,
            "TransposeB": False,
        },
        "MacroTile0": 0, "MacroTile1": 0, "DepthU": 64,
        "PrefetchGlobalRead": 0, "PrefetchLocalRead": 0, "DirectToLds": 1,  "DtlPlusLdsBuf": False,
        "GlobalReadVectorWidthA": 0, "GlobalReadVectorWidthB": 0,
        "LocalReadVectorWidth": 0,
        "WaveSeparateGlobalReadA": 0,
        "WaveSeparateGlobalReadB": 0,
        "Use64bShadowLimit" : 1,
        "MatrixInstruction": [16,16,32,1],
        "MIWaveGroup": [],
        "LDSTrInst": False,
        "TransposeLDS": 0,
        "ForceUnrollSubIter": False,
        "SwapGlobalReadOrder": False, # For asserting it gets set
        "UsePLRPack": False, # For asserting it gets set
        "UseF32XEmulation": False,
        "MIWaveTileA": 2,
        "MIWaveTileB": 2,
    }
    return kernel

class TestLayoutAutoDetection:
    """Tests for automatic supported_layouts detection in RegisterSchedule."""

    @pytest.fixture(autouse=True)
    def clean_registry(self):
        """Save and restore global registry state around each test."""
        orig_registry_len = len(_SCHEDULE_REGISTRY)
        orig_metadata_len = len(_SCHEDULE_METADATA)
        yield
        del _SCHEDULE_REGISTRY[orig_registry_len:]
        del _SCHEDULE_METADATA[orig_metadata_len:]

    # Shared tile config for all fake functions (arbitrary but valid)
    TILE = TileConfig(256, 256, 64, 2, 1, 1, True, 0, 0)

    def test_detect_tn_only(self):
        """A function that only handles TN should detect exactly ['TN']."""
        @RegisterSchedule(
            tile_config=self.TILE,
            dtype_predicate=is16bit,
            vector_widths=[8, 8, 8],
            matrix_inst=[16, 16, 32, 1],
            mfma_wave_group=[2, 2],
        )
        def _fake_tn_only(kernel, useLDSTr, TLDS):
            if isTN(kernel) and TLDS == 1:
                return True, None
            return False, None

        info = _SCHEDULE_METADATA[-1]
        assert sorted(info.supported_layouts) == ["TN"]

    def test_detect_tn_and_nn(self):
        """A function that handles TN and NN should detect both."""
        @RegisterSchedule(
            tile_config=self.TILE,
            dtype_predicate=is16bit,
            vector_widths=[8, 8, 8],
            matrix_inst=[16, 16, 32, 1],
            mfma_wave_group=[2, 2],
        )
        def _fake_tn_nn(kernel, useLDSTr, TLDS):
            if isTN(kernel) and TLDS == 1:
                return True, None
            if isNN(kernel) and useLDSTr and TLDS == 1:
                return True, None
            return False, None

        info = _SCHEDULE_METADATA[-1]
        assert sorted(info.supported_layouts) == ["NN", "TN"]

    def test_detect_all_four_layouts(self):
        """A function that handles all four layouts should detect all four."""
        @RegisterSchedule(
            tile_config=self.TILE,
            dtype_predicate=is16bit,
            vector_widths=[8, 8, 8],
            matrix_inst=[16, 16, 32, 1],
            mfma_wave_group=[2, 2],
        )
        def _fake_all_layouts(kernel, useLDSTr, TLDS):
            if isTN(kernel) and TLDS == 1:
                return True, None
            if isNT(kernel) and TLDS == 0:
                return True, None
            if isNN(kernel) and TLDS == 1:
                return True, None
            if isTT(kernel) and TLDS == 1:
                return True, None
            return False, None

        info = _SCHEDULE_METADATA[-1]
        assert sorted(info.supported_layouts) == ["NN", "NT", "TN", "TT"]

    def test_detect_no_layouts(self):
        """A function that always returns False should detect no layouts."""
        @RegisterSchedule(
            tile_config=self.TILE,
            dtype_predicate=is16bit,
            vector_widths=[8, 8, 8],
            matrix_inst=[16, 16, 32, 1],
            mfma_wave_group=[2, 2],
        )
        def _fake_no_layouts(kernel, useLDSTr, TLDS):
            return False, None

        info = _SCHEDULE_METADATA[-1]
        assert info.supported_layouts == []

    def test_mutation_isolation(self):
        """Kernel mutations in one probe must not leak into another probe."""
        @RegisterSchedule(
            tile_config=self.TILE,
            dtype_predicate=is16bit,
            vector_widths=[8, 8, 8],
            matrix_inst=[16, 16, 32, 1],
            mfma_wave_group=[2, 2],
        )
        def _fake_mutating(kernel, useLDSTr, TLDS):
            if isTN(kernel) and TLDS == 1:
                kernel["SwapGlobalReadOrder"] = True
                return True, None
            if isNN(kernel) and useLDSTr and TLDS == 1:
                # If mutation leaked from TN probe, this would be True
                assert not kernel.get("SwapGlobalReadOrder", False)
                return True, None
            return False, None

        info = _SCHEDULE_METADATA[-1]
        assert sorted(info.supported_layouts) == ["NN", "TN"]

    def test_detect_layouts_logs_on_value_error(self, capsys):
        """When the inner function raises ValueError, the probe should log a warning and skip that combo."""
        @RegisterSchedule(
            tile_config=self.TILE,
            dtype_predicate=is16bit,
            vector_widths=[8, 8, 8],
            matrix_inst=[16, 16, 32, 1],
            mfma_wave_group=[2, 2],
        )
        def _fake_raises_on_nt(kernel, useLDSTr, TLDS):
            if not kernel["ProblemType"]["TransposeA"] and kernel["ProblemType"]["TransposeB"]:
                raise ValueError("Value error for NT layout")
            if kernel["ProblemType"]["TransposeA"] and not kernel["ProblemType"]["TransposeB"] and TLDS == 1:
                return True, None
            return False, None

        info = _SCHEDULE_METADATA[-1]
        assert sorted(info.supported_layouts) == ["TN"]

        captured = capsys.readouterr()

        assert "Layout probe failed for func '_fake_raises_on_nt'" in captured.out
        assert "Value error for NT layout" in captured.out

    def test_consistency_with_existing_schedules(self):
        """Auto-detected layouts must match the previously hand-declared layouts for all existing schedules."""
        # Expected layouts from the original manual annotations (prior to auto-detection)
        EXPECTED = {
            "_get_schedule_256x96x64_16bit": ["NN", "TN"],
            "_get_schedule_192x256x64_16bit": ["NN", "NT", "TN"],
            "_get_schedule_256x192x64_16bit": ["NN", "NT", "TN"],
            "_get_schedule_256x256x128_8bit": ["TN"],
            "_get_schedule_256x256x64_16bit": ["NN", "NT", "TN", "TT"],
            "_get_schedule_160x256x64_16bit": ["NN", "NT", "TN"],
            "_get_schedule_96x256x64_16bit": ["NT", "TN"],
            "_get_schedule_256x160x64_16bit": ["NN", "NT", "TN"],
            "_get_schedule_256x240x64_16bit": ["NN", "NT", "TN"],
            "_get_schedule_256x208x64_16bit": ["NN", "TN"],
            "_get_schedule_192x128x64_16bit": ["TN"],
            "_get_schedule_224x128x64_16bit": ["NN", "NT", "TN"],
            "_get_schedule_224x256x64_16bit": ["NN", "NT", "TN"],
            "_get_schedule_192x320x64_16bit": ["NN", "NT", "TN"],
            "_get_schedule_256x224x64_16bit": ["NN", "NT", "TN"],
            "_get_schedule_320x192x64_16bit": ["NN", "NT", "TN"],
            "_get_schedule_240x256x64_16bit": ["NN", "NT", "TN"],
            "_get_schedule_208x256x64_16bit": ["NN", "NT", "TN"],
            "_get_schedule_128x224x64_16bit": ["NN", "NT", "TN"],
            "_get_schedule_128x192x64_16bit": ["TN"],
            "_get_schedule_128x192x32_TF32": ["TN"],
            "_get_schedule_192x256x32_TF32": ["NN", "TN"],
            "_get_schedule_256x192x32_TF32": ["NN", "TN"],
            "_get_schedule_256x256x32_TF32": ["TN"],
            "_get_schedule_192x128x32_TF32": ["TN"],
            "_get_schedule_128x128x32_TF32": ["TN"],
            "_get_schedule_128x128x32_TF32_plr1": ["NN", "TN"],
            "_get_schedule_128x128x64_TF32": ["NN", "TN"],
            "_get_schedule_128x256x32_TF32": ["TN"],
            "_get_schedule_128x160x64_TF32": ["TN"],
            "_get_schedule_256x128x32_TF32": ["TN"],
            "_get_schedule_64x128x64_TF32": ["TN"],
            "_get_schedule_128x64x64_TF32": ["TN"],
            "_get_schedule_160x128x64_TF32": ["NN", "TN"],
            "_get_schedule_128x256x64_16bit": ["NN"],
        }

        for info in _SCHEDULE_METADATA:
            if info.name in EXPECTED:
                assert sorted(info.supported_layouts) == EXPECTED[info.name], \
                    f"{info.name}: auto-detected {sorted(info.supported_layouts)}, expected {EXPECTED[info.name]}"

        # Verify all expected schedules were found in the registry
        found_names = {info.name for info in _SCHEDULE_METADATA}
        for name in EXPECTED:
            assert name in found_names, f"{name} not found in _SCHEDULE_METADATA"

    def test_cms_api_query(self):
        """Test the CMS API query function."""
        # test all 16 bit types and layouts
        
        results = query_cms_kernels()
        assert len(results) > 0
        assert all(isinstance(result, dict) for result in results)
        assert all("name" in result for result in results)
        assert all("dtype" in result for result in results)
        assert all("supported_layouts" in result for result in results)
        assert all("MacroTile0" in result for result in results)
        assert all("MacroTile1" in result for result in results)
        assert all("DepthU" in result for result in results)


        for dtype in ["16bit", "TF32"]:
            results = query_cms_kernels(dtype=dtype)
            assert len(results) > 0
            assert all(isinstance(result, dict) for result in results)
            assert all("name" in result for result in results)
            assert all("dtype" in result for result in results)
            assert all("supported_layouts" in result for result in results)
            assert all("MacroTile0" in result for result in results)
            assert all("MacroTile1" in result for result in results)
            assert all("DepthU" in result for result in results)

        for layout in ["TN", "NT", "NN"]:
            results = query_cms_kernels(dtype="16bit", layout=layout)
            assert len(results) > 0
            assert all(isinstance(result, dict) for result in results)
            assert all("name" in result for result in results)
            assert all("dtype" in result for result in results)
            assert all("supported_layouts" in result for result in results)
            assert all("MacroTile0" in result for result in results)
            assert all("MacroTile1" in result for result in results)
            assert all("DepthU" in result for result in results)

            

        
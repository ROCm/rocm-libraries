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
from typing import Set

from Tensile.Components.CustomSchedule import (
    hasCustomSchedule, ScheduleInfo, RegisterSchedule, TileConfig,
    CMSKernelInfo, _SCHEDULE_METADATA, _SCHEDULE_REGISTRY,
    isTN, isNT, isNN, isTT, is16bit, query_cms_kernels, get_available_layouts,
    get_available_dtypes, get_cms_kernel_info_objects
)

def _get_compact_layout_string(transposeA: bool, transposeB: bool):
    return ("T" if transposeA else "N") + ("T" if transposeB else "N")

def _get_layouts_for(func_name) -> Set[str]:
    """Collect unique layout strings for a function from _SCHEDULE_METADATA."""
    layouts = set()
    for info in _SCHEDULE_METADATA:
        if info.name == func_name:
            layout = _get_compact_layout_string(info.TransposeA, info.TransposeB)
            layouts.add(layout)
    return layouts

def _get_cms_kernel_info_for(func_name):
    """Get the CMS kernel info for a function from _SCHEDULE_METADATA."""
    for info in _SCHEDULE_METADATA:
        if info.name == func_name:
            yield info

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

        assert _get_layouts_for("_fake_tn_only") == {"TN"}

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

        assert _get_layouts_for("_fake_tn_nn") == {"NN", "TN"}

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

        assert _get_layouts_for("_fake_all_layouts") == {"NN", "NT", "TN", "TT"}

        assert get_available_layouts(dtype="16bit") == {"NN", "NT", "TN", "TT"}

        assert get_available_layouts() == {"NN", "NT", "TN", "TT"}

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

        assert _get_layouts_for("_fake_no_layouts") == set()

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

        assert _get_layouts_for("_fake_mutating") == {"NN", "TN"}

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

        assert _get_layouts_for("_fake_raises_on_nt") == {"TN"}

        captured = capsys.readouterr()

        assert "Layout probe failed for func '_fake_raises_on_nt'" in captured.out
        assert "Value error for NT layout" in captured.out

    def test_consistency_with_existing_schedules(self):
        """Auto-detected layouts must match the previously hand-declared layouts for all existing schedules."""
        EXPECTED = {
            "_get_schedule_256x96x64_16bit": {"NN", "TN"},
            "_get_schedule_256x96x64_16bit_DPLB": {"NT"},
            "_get_schedule_192x256x64_16bit": {"NN", "NT", "TN"},
            "_get_schedule_256x192x64_16bit": {"NN", "NT", "TN"},
            "_get_schedule_256x256x128_8bit": {"TN"},
            "_get_schedule_256x256x64_16bit": {"NN", "NT", "TN", "TT"},
            "_get_schedule_160x256x64_16bit": {"NN", "NT", "TN"},
            "_get_schedule_96x256x64_16bit": {"NN", "NT", "TN"},
            "_get_schedule_256x160x64_16bit": {"NN", "NT", "TN"},
            "_get_schedule_256x240x64_16bit": {"NN", "NT", "TN"},
            "_get_schedule_256x208x64_16bit": {"NN", "TN"},
            "_get_schedule_192x128x64_16bit": {"TN"},
            "_get_schedule_224x128x64_16bit": {"NN", "NT", "TN"},
            "_get_schedule_224x256x64_16bit": {"NN", "NT", "TN"},
            "_get_schedule_192x320x64_16bit": {"NN", "NT", "TN"},
            "_get_schedule_256x224x64_16bit": {"NN", "NT", "TN"},
            "_get_schedule_320x192x64_16bit": {"NN", "NT", "TN"},
            "_get_schedule_352x192x64_16bit": {"TN"},
            "_get_schedule_240x256x64_16bit": {"NN", "NT", "TN"},
            "_get_schedule_208x256x64_16bit": {"NN", "NT", "TN"},
            "_get_schedule_128x224x64_16bit": {"NN", "NT", "TN"},
            "_get_schedule_128x192x64_16bit": {"TN"},
            "_get_schedule_128x192x32_TF32": {"TN"},
            "_get_schedule_192x256x32_TF32": {"NN", "TN"},
            "_get_schedule_256x192x32_TF32": {"NN", "TN"},
            "_get_schedule_256x256x32_TF32": {"NT", "TN"},
            "_get_schedule_192x128x32_TF32": {"TN"},
            "_get_schedule_128x128x32_TF32": {"TN"},
            "_get_schedule_128x128x32_TF32_plr1": {"NN", "NT", "TN"},
            "_get_schedule_128x128x64_TF32": {"NN", "TN"},
            "_get_schedule_128x256x32_TF32": {"TN"},
            "_get_schedule_128x160x64_TF32": {"TN"},
            "_get_schedule_256x128x32_TF32": {"TN"},
            "_get_schedule_64x128x64_TF32": {"TN"},
            "_get_schedule_128x64x64_TF32": {"TN"},
            "_get_schedule_160x128x64_TF32": {"NN", "TN"},
            "_get_schedule_128x256x64_16bit": {"NN"},
            "_get_schedule_224x320x64_16bit": {"TN"}
        }

        registered_schedules = {info.name for info in _SCHEDULE_METADATA}

        # Confirm we have all the expected schedules in the registry
        for func_name in EXPECTED:
            assert func_name in registered_schedules, f"{func_name} not found in _SCHEDULE_METADATA (Please update the expected layouts in the test if needed)"

        # Confirm we have all the registered schedules in the expected layouts
        for func_name in registered_schedules:
            assert func_name in EXPECTED, f"{func_name} not found in 'EXPECTED' (Please update the 'EXPECTED' layouts in the test if needed)"

        for name, expected_layouts in EXPECTED.items():
            detected = _get_layouts_for(name)
            assert detected == expected_layouts, \
                f"{name}: auto-detected {detected}, expected {expected_layouts}"

    def test_cms_api_query(self):
        """Test the CMS API query function."""
        REQUIRED_KEYS = {"name", "dtype", "TransposeA", "TransposeB", "LDSTrInst", "TransposeLDS",
                         "MacroTile0", "MacroTile1", "DepthU"}

        results = query_cms_kernels()
        assert len(results) > 0
        assert all(isinstance(result, dict) for result in results)
        for result in results:
            assert REQUIRED_KEYS.issubset(result.keys()), \
                f"Missing keys in {result.get('name', '?')}: {REQUIRED_KEYS - result.keys()}"

        for dtype in ["16bit", "TF32"]:
            results = query_cms_kernels(dtype=dtype)
            assert len(results) > 0
            assert all(isinstance(result, dict) for result in results)
            for result in results:
                assert REQUIRED_KEYS.issubset(result.keys())
                assert result["dtype"].lower() == dtype.lower()

        for layout in ["TN", "NT", "NN"]:
            results = query_cms_kernels(dtype="16bit", layout=layout)
            assert len(results) > 0
            assert all(isinstance(result, dict) for result in results)
            expected_a = layout[0] == "T"
            expected_b = layout[1] == "T"
            for result in results:
                assert REQUIRED_KEYS.issubset(result.keys())
                assert result["TransposeA"] == expected_a
                assert result["TransposeB"] == expected_b

    def test_detect_layout_ldstr_transpose_combos(self):
        """Confirm the correct detection of layouts, LDSTrInst, and TransposeLDS in the CMS kernel info."""
        @RegisterSchedule(
            tile_config=self.TILE,
            dtype_predicate=is16bit,
            vector_widths=[8, 8, 8],
            matrix_inst=[16, 16, 32, 1],
            mfma_wave_group=[2, 2],
        )
        def _fake_cms(kernel, useLDSTr, TLDS):
            if isTN(kernel) and TLDS == 1:
                return True, None
            if isNN(kernel) and useLDSTr and TLDS == 1:
                return True, None
            return False, None

        kernel_infos = [(info.TransposeA, info.TransposeB, info.LDSTrInst, info.TransposeLDS) for info in _get_cms_kernel_info_for("_fake_cms")]

        expected_kernel_infos_detection = {
            # TransposeA, TransposeB, LDSTrInst, TransposeLDS
            (True, False, True, 1): True,       # TN  (TN and LDSTrInst and TransposeLDS == 1)
            (True, False, False, 1): True,      # TN  (TN and no LDSTrInst and TransposeLDS == 1)
            (True, False, True, 0): False,      # TN  (TN and LDSTrInst and TransposeLDS == 0)
            (True, False, False, 0): False,     # TN  (TN and no LDSTrInst and TransposeLDS == 0)

            (False, False, True, 1): True,      # NN  (NN and LDSTrInst and TransposeLDS == 1)
            (False, False, False, 1): False,    # NN  (NN and no LDSTrInst and TransposeLDS == 1)
            (False, False, True, 0): False,     # NN  (NN and LDSTrInst and TransposeLDS == 0)
            (False, False, False, 0): False,    # NN  (NN and no LDSTrInst and TransposeLDS == 0)

            (False, True, True, 1): False,      # NT  (NT and LDSTrInst and TransposeLDS == 1)
            (False, True, False, 1): False,     # NT  (NT and no LDSTrInst and TransposeLDS == 1)
            (False, True, True, 0): False,      # NT  (NT and LDSTrInst and TransposeLDS == 0)
            (False, True, False, 0): False,     # NT  (NT and no LDSTrInst and TransposeLDS == 0)

            (True, True, True, 1): False,       # TT  (TT and LDSTrInst and TransposeLDS == 1)
            (True, True, False, 1): False,      # TT  (TT and no LDSTrInst and TransposeLDS == 1)
            (True, True, True, 0): False,       # TT  (TT and LDSTrInst and TransposeLDS == 0)
            (True, True, False, 0): False,      # TT  (TT and no LDSTrInst and TransposeLDS == 0)

        }

        for expected_kernel_info, expected_detection in expected_kernel_infos_detection.items():
            if expected_detection:
                assert expected_kernel_info in kernel_infos, f"Kernel support for {expected_kernel_info} (TransposeA, TransposeB, LDSTrInst, TransposeLDS) is not found in kernel_infos"
            else:
                assert expected_kernel_info not in kernel_infos, f"Kernel support for {expected_kernel_info} (TransposeA, TransposeB, LDSTrInst, TransposeLDS) is unexpectedly found in kernel_infos"

    def test_get_available_dtypes(self):
        dtypes = get_available_dtypes()

        expected_dtypes = {"16bit", "8bit", "TF32"}
        assert dtypes == expected_dtypes, \
            f"Expected {expected_dtypes}, got {dtypes} (Please update the expected dtypes in the test if needed)"

    def test_get_cms_kernel_info_objects(self):
        @RegisterSchedule(
            tile_config=self.TILE,
            dtype_predicate=is16bit,
            vector_widths=[8, 8, 8],
            matrix_inst=[16, 16, 32, 1],
            mfma_wave_group=[2, 2],
        )
        def _fake_cms(kernel, useLDSTr, TLDS):
            if isTN(kernel) and TLDS == 1:
                return True, None
            if isNN(kernel) and useLDSTr and TLDS == 1:
                return True, None
            return False, None

        EXPECTED_FIELDS = {
            "name", "dtype", "MacroTile0", "MacroTile1", "DepthU",
            "PrefetchGlobalRead", "PrefetchLocalRead", "DirectToLds",
            "DtlPlusLdsBuf", "WaveSeparateGlobalReadA", "WaveSeparateGlobalReadB",
            "GlobalReadVectorWidthA", "GlobalReadVectorWidthB", "LocalReadVectorWidth",
            "MatrixInstruction", "MIWaveGroup", "LDSTrInst", "TransposeLDS",
            "TransposeA", "TransposeB",
        }

        all_results = get_cms_kernel_info_objects()
        assert len(all_results) > 0
        for info in all_results:
            assert isinstance(info, CMSKernelInfo), \
                f"Expected CMSKernelInfo, got {type(info).__name__}"
            actual_fields = {f.name for f in info.__dataclass_fields__.values()}
            assert EXPECTED_FIELDS.issubset(actual_fields), \
                f"Missing fields on {info.name}: {EXPECTED_FIELDS - actual_fields}"

        for dtype in ["16bit", "TF32"]:
            filtered = get_cms_kernel_info_objects(dtype=dtype)
            assert len(filtered) > 0, f"No results for dtype={dtype}"
            assert all(info.dtype.lower() == dtype.lower() for info in filtered)

        for layout in ["TN", "NT", "NN"]:
            filtered = get_cms_kernel_info_objects(dtype="16bit", layout=layout)
            assert len(filtered) > 0, f"No results for dtype=16bit, layout={layout}"
            expected_a = layout[0] == "T"
            expected_b = layout[1] == "T"
            for info in filtered:
                assert info.TransposeA == expected_a and info.TransposeB == expected_b, \
                    f"{info.name}: expected layout {layout}, got TransposeA={info.TransposeA}, TransposeB={info.TransposeB}"

        dict_results = query_cms_kernels()
        obj_results = get_cms_kernel_info_objects()
        assert len(dict_results) == len(obj_results), \
            f"query_cms_kernels returned {len(dict_results)} results but get_cms_kernel_info_objects returned {len(obj_results)}"


        assert "_fake_cms" in [info.name for info in obj_results], \
            "'_fake_cms' not found in get_cms_kernel_info_objects"
        assert "_fake_cms" in [info["name"] for info in dict_results], \
             "'_fake_cms' not found in query_cms_kernels"

        for info in obj_results:
            if info.name == "_fake_cms":
                assert info.dtype == "16bit", \
                    f"{info.name}: expected dtype 16bit, got {info.dtype}"
                assert info.MacroTile0 == 256, \
                    f"{info.name}: expected MacroTile0 256, got {info.MacroTile0}"
                assert info.MacroTile1 == 256, \
                    f"{info.name}: expected MacroTile1 256, got {info.MacroTile1}"
                assert info.DepthU == 64, \
                    f"{info.name}: expected DepthU 64, got {info.DepthU}"
                assert info.PrefetchGlobalRead == 2, \
                    f"{info.name}: expected PrefetchGlobalRead 2, got {info.PrefetchGlobalRead}"
                assert info.PrefetchLocalRead == 1, \
                    f"{info.name}: expected PrefetchLocalRead 1, got {info.PrefetchLocalRead}"
                assert info.DirectToLds == 1, \
                    f"{info.name}: expected DirectToLds 1, got {info.DirectToLds}"
                assert info.DtlPlusLdsBuf == True, \
                    f"{info.name}: expected DtlPlusLdsBuf True, got {info.DtlPlusLdsBuf}"
                assert info.WaveSeparateGlobalReadA == 0, \
                    f"{info.name}: expected WaveSeparateGlobalReadA 0, got {info.WaveSeparateGlobalReadA}"
                assert info.WaveSeparateGlobalReadB == 0, \
                    f"{info.name}: expected WaveSeparateGlobalReadB 0, got {info.WaveSeparateGlobalReadB}"
                assert info.GlobalReadVectorWidthA == 8, \
                    f"{info.name}: expected GlobalReadVectorWidthA 8, got {info.GlobalReadVectorWidthA}"
                assert info.GlobalReadVectorWidthB == 8, \
                    f"{info.name}: expected GlobalReadVectorWidthB 8, got {info.GlobalReadVectorWidthB}"
                assert info.LocalReadVectorWidth == 8, \
                    f"{info.name}: expected LocalReadVectorWidth 8, got {info.LocalReadVectorWidth}"
                assert info.MatrixInstruction == [16, 16, 32, 1], \
                    f"{info.name}: expected MatrixInstruction 16, got {info.MatrixInstruction}"

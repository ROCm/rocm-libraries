# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for Triton-specific C++ bindings in origami."""

import pytest
import origami

from helpers import HARDWARE


class TestTargetT:
    """Tests for target_t enum and config_t.target field."""

    def test_target_enum_values_exist(self):
        assert origami.target_t.generic is not None
        assert origami.target_t.tensilelite is not None
        assert origami.target_t.rocroller is not None
        assert origami.target_t.triton is not None
        assert origami.target_t.composable_kernel is not None

    def test_config_default_target(self):
        config = origami.config_t()
        assert config.target == origami.target_t.tensilelite

    def test_config_target_roundtrip(self):
        config = origami.config_t()
        config.target = origami.target_t.triton
        assert config.target == origami.target_t.triton


class TestTritonLDS:
    """Tests for Triton LDS estimation functions.

    Reference formula validated against Triton 3.6.0 compiled kernel metadata
    (n_shared_bytes) on AMD Instinct GPUs:
        stages == 1  →  max(A_tile_bytes, B_tile_bytes)
        stages >= 2  →  (stages - 1) * (A_tile_bytes + B_tile_bytes)
    """

    @staticmethod
    def _reference_estimate(bm, bn, bk, bytes_a, bytes_b, num_stages=2):
        a_tile = bm * bk * bytes_a
        b_tile = bk * bn * bytes_b
        if num_stages <= 1:
            return max(a_tile, b_tile)
        return (num_stages - 1) * (a_tile + b_tile)

    @pytest.fixture
    def hw(self):
        return HARDWARE["gfx942"]

    def test_estimate_1stage_symmetric(self):
        mt = origami.dim3_t(128, 128, 32)
        result = origami.estimate_triton_lds_bytes(mt, origami.data_type_t.Half, origami.data_type_t.Half, 1)
        assert result == max(128 * 32 * 2, 32 * 128 * 2)  # max(A, B)

    def test_estimate_1stage_asymmetric(self):
        mt = origami.dim3_t(128, 64, 64)
        result = origami.estimate_triton_lds_bytes(mt, origami.data_type_t.Half, origami.data_type_t.Half, 1)
        a_tile, b_tile = 128 * 64 * 2, 64 * 64 * 2
        assert result == max(a_tile, b_tile)

    def test_estimate_2stage(self):
        mt = origami.dim3_t(128, 128, 32)
        result = origami.estimate_triton_lds_bytes(mt, origami.data_type_t.Half, origami.data_type_t.Half, 2)
        assert result == 1 * (128 * 32 * 2 + 32 * 128 * 2)  # (2-1)*(A+B)

    def test_estimate_3stage(self):
        mt = origami.dim3_t(128, 128, 32)
        result = origami.estimate_triton_lds_bytes(mt, origami.data_type_t.Half, origami.data_type_t.Half, 3)
        assert result == 2 * (128 * 32 * 2 + 32 * 128 * 2)  # (3-1)*(A+B)

    def test_estimate_default_stages(self):
        mt = origami.dim3_t(256, 256, 64)
        result_default = origami.estimate_triton_lds_bytes(mt, origami.data_type_t.Half, origami.data_type_t.Half)
        result_explicit = origami.estimate_triton_lds_bytes(mt, origami.data_type_t.Half, origami.data_type_t.Half, 2)
        assert result_default == result_explicit

    def test_check_triton_lds_capacity_fits(self, hw):
        mt = origami.dim3_t(64, 64, 32)
        assert origami.check_triton_lds_capacity(hw, mt, origami.data_type_t.Half, origami.data_type_t.Half)

    def test_check_triton_lds_capacity_too_large(self, hw):
        mt = origami.dim3_t(512, 512, 128)
        assert not origami.check_triton_lds_capacity(hw, mt, origami.data_type_t.Half, origami.data_type_t.Half)

    def test_stages_ordering(self, hw):
        mt = origami.dim3_t(128, 128, 32)
        t1 = origami.estimate_triton_lds_bytes(mt, origami.data_type_t.Half, origami.data_type_t.Half, 1)
        t2 = origami.estimate_triton_lds_bytes(mt, origami.data_type_t.Half, origami.data_type_t.Half, 2)
        t3 = origami.estimate_triton_lds_bytes(mt, origami.data_type_t.Half, origami.data_type_t.Half, 3)
        assert t1 < t2 < t3

    def test_estimate_matches_reference_sweep(self):
        """Sweep tile sizes and verify C++ matches the validated formula."""
        for bm in [16, 32, 64, 128, 256]:
            for bn in [16, 32, 64, 128, 256]:
                for bk in [16, 32, 64, 128, 256, 512]:
                    for ns in [1, 2, 3]:
                        mt = origami.dim3_t(bm, bn, bk)
                        cpp = origami.estimate_triton_lds_bytes(
                            mt, origami.data_type_t.Half, origami.data_type_t.Half, ns
                        )
                        ref = self._reference_estimate(bm, bn, bk, 2, 2, ns)
                        assert cpp == ref, (
                            f"Mismatch at {bm}x{bn}x{bk} stages={ns}: C++={cpp} ref={ref}"
                        )


class TestTritonWSParams:
    """Tests for Triton work-stealing parameter selection."""

    def test_few_tiles(self):
        result = origami.select_triton_ws_params(256, 256, 128, 128)
        assert result.counters_per_xcd == 1
        assert result.workgroup_mapping > 0

    def test_many_tiles(self):
        result = origami.select_triton_ws_params(16384, 16384, 128, 128)
        assert result.counters_per_xcd > 1
        assert result.workgroup_mapping > 0

    def test_struct_fields(self):
        result = origami.select_triton_ws_params(4096, 4096, 128, 128)
        assert hasattr(result, "counters_per_xcd")
        assert hasattr(result, "workgroup_mapping")


class TestTritonHierarchicalSplit:
    """Tests for Triton hierarchical split computation."""

    def test_basic_split(self):
        result = origami.compute_triton_hierarchical_split(4096, 4096, 128, 128, 8, 304)
        assert result.local_per_xcd > 0
        assert result.local_per_xcd + result.global_tiles > 0

    def test_struct_fields(self):
        result = origami.compute_triton_hierarchical_split(2048, 2048, 128, 128, 8, 304)
        assert hasattr(result, "local_per_xcd")
        assert hasattr(result, "global_tiles")


class TestTritonSKGrid:
    """Tests for Triton StreamK grid computation."""

    def test_data_parallel_case(self):
        # Enough tiles to cover all CUs without StreamK
        result = origami.compute_triton_sk_grid(16384, 16384, 4096, 128, 128, 64, 304, 16)
        assert result > 0

    def test_small_problem(self):
        result = origami.compute_triton_sk_grid(256, 256, 256, 128, 128, 64, 304, 16)
        assert result > 0

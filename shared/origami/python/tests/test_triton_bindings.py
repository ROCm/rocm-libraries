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


def _make_problem(a_dtype=None, b_dtype=None):
    """Build a minimal problem_t carrying just the dtypes we care about."""
    p = origami.problem_t()
    p.a_dtype = a_dtype if a_dtype is not None else origami.data_type_t.Half
    p.b_dtype = b_dtype if b_dtype is not None else origami.data_type_t.Half
    return p


def _make_triton_config(bm, bn, bk, num_stages=2):
    """Build a minimal Triton-targeted config_t for LDS estimation."""
    c = origami.config_t()
    c.mt = origami.dim3_t(bm, bn, bk)
    c.mi = origami.dim3_t(16, 16, 16)
    c.occupancy = 1
    c.target = origami.target_t.triton
    c.triton().num_stages = num_stages
    return c


class TestTritonLDS:
    """Tests for unified LDS estimation with Triton pipeline-stage handling.

    Reference formula validated against Triton 3.6.0 compiled kernel metadata
    (n_shared_bytes) on AMD Instinct GPUs:
        stages == 1  →  max(A_tile_bytes, B_tile_bytes)
        stages >= 2  →  (stages - 1) * (A_tile_bytes + B_tile_bytes)
    Tensile path (no triton_params) always returns 1 * (A + B).
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
        result = origami.estimate_lds_bytes(_make_problem(), _make_triton_config(128, 128, 32, 1))
        assert result == max(128 * 32 * 2, 32 * 128 * 2)

    def test_estimate_1stage_asymmetric(self):
        result = origami.estimate_lds_bytes(_make_problem(), _make_triton_config(128, 64, 64, 1))
        a_tile, b_tile = 128 * 64 * 2, 64 * 64 * 2
        assert result == max(a_tile, b_tile)

    def test_estimate_2stage(self):
        result = origami.estimate_lds_bytes(_make_problem(), _make_triton_config(128, 128, 32, 2))
        assert result == 1 * (128 * 32 * 2 + 32 * 128 * 2)

    def test_estimate_3stage(self):
        result = origami.estimate_lds_bytes(_make_problem(), _make_triton_config(128, 128, 32, 3))
        assert result == 2 * (128 * 32 * 2 + 32 * 128 * 2)

    def test_estimate_tensile_path_no_triton_params(self):
        """Without triton_params, we get the Tensile single-buffer formula."""
        p = _make_problem()
        c = origami.config_t()
        c.mt = origami.dim3_t(128, 128, 32)
        c.mi = origami.dim3_t(16, 16, 16)
        c.occupancy = 1
        # target left at default (tensilelite); no triton_params set.
        result = origami.estimate_lds_bytes(p, c)
        assert result == 128 * 32 * 2 + 32 * 128 * 2

    def test_check_lds_capacity_fits(self, hw):
        c = _make_triton_config(64, 64, 32, 2)
        assert origami.check_lds_capacity(hw, _make_problem(), c)

    def test_check_lds_capacity_too_large(self, hw):
        c = _make_triton_config(512, 512, 128, 2)
        assert not origami.check_lds_capacity(hw, _make_problem(), c)

    def test_stages_ordering(self):
        p = _make_problem()
        t1 = origami.estimate_lds_bytes(p, _make_triton_config(128, 128, 32, 1))
        t2 = origami.estimate_lds_bytes(p, _make_triton_config(128, 128, 32, 2))
        t3 = origami.estimate_lds_bytes(p, _make_triton_config(128, 128, 32, 3))
        assert t1 < t2 < t3

    def test_estimate_matches_reference_sweep(self):
        """Sweep tile sizes and verify C++ matches the validated formula."""
        p = _make_problem()
        for bm in [16, 32, 64, 128, 256]:
            for bn in [16, 32, 64, 128, 256]:
                for bk in [16, 32, 64, 128, 256, 512]:
                    for ns in [1, 2, 3]:
                        c = _make_triton_config(bm, bn, bk, ns)
                        cpp = origami.estimate_lds_bytes(p, c)
                        ref = self._reference_estimate(bm, bn, bk, 2, 2, ns)
                        assert cpp == ref, (
                            f"Mismatch at {bm}x{bn}x{bk} stages={ns}: C++={cpp} ref={ref}"
                        )

    def test_subbyte_dtype_no_truncation(self):
        """F4 dtype must not truncate to zero (sub-byte fix)."""
        # Skip if Float4 isn't exposed by this build
        if not hasattr(origami.data_type_t, "Float4"):
            pytest.skip("Float4 not available in this build")
        p = _make_problem(origami.data_type_t.Float4, origami.data_type_t.Float4)
        c = _make_triton_config(128, 128, 64, 2)
        result = origami.estimate_lds_bytes(p, c)
        # 128*64 elements * 4 bits / 8 = 4096 bytes per tile; 2 tiles total
        assert result == 1 * (4096 + 4096)
        assert result > 0


def _make_sk_problem(m, n, k, batch=1, c_dtype=None):
    p = origami.problem_t()
    p.size = origami.dim3_t(m, n, k)
    p.batch = batch
    p.a_dtype = origami.data_type_t.Half
    p.b_dtype = origami.data_type_t.Half
    p.c_dtype = c_dtype if c_dtype is not None else origami.data_type_t.Half
    p.d_dtype = p.c_dtype
    return p


def _make_sk_config(bm, bn, bk):
    c = origami.config_t()
    c.mt = origami.dim3_t(bm, bn, bk)
    c.mi = origami.dim3_t(16, 16, 16)
    c.occupancy = 1
    c.target = origami.target_t.triton
    return c


class TestTritonSKGrid:
    """Tests for Triton StreamK grid computation.

    The Triton StreamK heuristic now uses the shared streamk fractional-grid
    and k-split helpers (no logic duplication with streamk::grid_k_split_aware)
    plus a Triton-only "small last partial wave -> prev_pow2(n_cu)" fix.
    """

    def test_data_parallel_case(self):
        # Enough tiles to cover all CUs without StreamK.
        problem = _make_sk_problem(16384, 16384, 4096)
        config = _make_sk_config(128, 128, 64)
        result = origami.compute_triton_sk_grid(problem, config, HARDWARE["gfx950"])
        assert result > 0

    def test_small_problem(self):
        problem = _make_sk_problem(256, 256, 256)
        config = _make_sk_config(128, 128, 64)
        result = origami.compute_triton_sk_grid(problem, config, HARDWARE["gfx950"])
        assert result > 0

    def test_batched_problem_uses_batch_dimension(self):
        """Batched problems should produce a different grid than B=1.

        Previously the Triton helper silently dropped the batch dimension when
        counting tiles; with the unified `(problem, config, hardware)` API it
        goes through `streamk::compute_number_of_output_tiles` which is
        batch-aware.
        """
        config = _make_sk_config(128, 128, 64)
        single = origami.compute_triton_sk_grid(
            _make_sk_problem(2048, 2048, 1024, batch=1), config, HARDWARE["gfx950"]
        )
        batched = origami.compute_triton_sk_grid(
            _make_sk_problem(2048, 2048, 1024, batch=8), config, HARDWARE["gfx950"]
        )
        assert single > 0
        assert batched > 0
        assert batched != single

    def test_subbyte_c_dtype_does_not_zero_divide(self):
        """F4 output dtype must not collapse the per-tile workspace to zero.

        Previously per-elem bytes were `out_dtype_bits / 8` which truncated
        sub-byte dtypes to 0 and disabled the workspace gate. The unified path
        derives bytes via ceil_div(bits, 8) on the C dtype.
        """
        problem = _make_sk_problem(8192, 8192, 1024, c_dtype=origami.data_type_t.Float4)
        config = _make_sk_config(128, 128, 64)
        result = origami.compute_triton_sk_grid(problem, config, HARDWARE["gfx950"])
        assert result > 0


def _make_overlay_problem():
    p = origami.problem_t()
    p.size = origami.dim3_t(4096, 4096, 1024)
    p.batch = 1
    p.a_dtype = origami.data_type_t.BFloat16
    p.b_dtype = origami.data_type_t.BFloat16
    p.c_dtype = origami.data_type_t.BFloat16
    p.d_dtype = origami.data_type_t.BFloat16
    p.mi_dtype = origami.data_type_t.BFloat16
    p.a_transpose = origami.transpose_t.T
    p.b_transpose = origami.transpose_t.N
    return p


def _make_overlay_config(target):
    c = origami.config_t()
    c.mt = origami.dim3_t(256, 256, 64)
    c.mi = origami.dim3_t(16, 16, 16)
    c.occupancy = 1
    c.target = target
    return c


class TestTritonHeuristicOverlay:
    """End-to-end check that the Triton-target heuristic overlay is applied
    via `heuristic_params_t::merge_with` at `context_t` construction time.

    This used to be a manual `if (target == triton) L_tile_total *=
    triton_h.weight_tile_total` patch inside `compute_tile_latency`. After
    the unification, a single default-aware merge at construction is the
    only thing required to make the Triton tuning take effect, and any
    consumer that reads `heuristic.weight_tile_total` (including the
    binding-exposed `compute_tile_latency`) sees the overlay automatically.
    """

    def test_triton_overlay_scales_tile_latency_by_overlay_weight(self):
        # The Triton heuristics DB entry for the 256x256x64 tile sets
        # weight_tile_total = 0.95 (a 5% latency discount). With identical
        # everything-else, the Triton-target tile latency must be exactly
        # 0.95x the Tensile-target tile latency.
        problem = _make_overlay_problem()
        triton_cfg = _make_overlay_config(origami.target_t.triton)
        tensile_cfg = _make_overlay_config(origami.target_t.tensilelite)
        hw = HARDWARE["gfx950"]

        triton_lat = origami.compute_tile_latency(problem, hw, triton_cfg)
        tensile_lat = origami.compute_tile_latency(problem, hw, tensile_cfg)

        assert triton_lat > 0.0
        assert tensile_lat > 0.0
        assert triton_lat == pytest.approx(0.95 * tensile_lat, rel=1e-12)

    def test_non_overlay_tile_does_not_scale(self):
        # A tile that does NOT match any Triton heuristic entry (e.g.
        # 128x128x32) gets the default weight_tile_total = 1.0 from the
        # overlay, so triton and tensile tile latencies must coincide.
        problem = _make_overlay_problem()
        triton_cfg = _make_overlay_config(origami.target_t.triton)
        tensile_cfg = _make_overlay_config(origami.target_t.tensilelite)
        triton_cfg.mt = origami.dim3_t(128, 128, 32)
        tensile_cfg.mt = origami.dim3_t(128, 128, 32)
        hw = HARDWARE["gfx950"]

        triton_lat = origami.compute_tile_latency(problem, hw, triton_cfg)
        tensile_lat = origami.compute_tile_latency(problem, hw, tensile_cfg)

        assert triton_lat > 0.0
        assert tensile_lat > 0.0
        assert triton_lat == pytest.approx(tensile_lat, rel=1e-12)


def _make_default_configs_problem(a_dtype, b_dtype=None):
    """Minimal problem_t carrying just the dtypes the heuristic gates on."""
    p = origami.problem_t()
    p.a_dtype = a_dtype
    p.b_dtype = b_dtype if b_dtype is not None else a_dtype
    return p


class TestTritonDefaultConfigs:
    """Tests for `get_triton_default_configs(problem, hardware)`.

    The function returns a flat list of `config_t` whose `mt.{m,n,k}` covers
    the architecture-aware tile candidate space. The narrower of
    (problem.a_dtype, problem.b_dtype) drives per-arch gating.
    """

    def test_returns_nonempty_list(self):
        problem = _make_default_configs_problem(origami.data_type_t.Half)
        configs = origami.get_triton_default_configs(problem, HARDWARE["gfx942"])
        assert isinstance(configs, list)
        assert len(configs) > 0
        for c in configs:
            assert isinstance(c, origami.config_t)
            assert c.mt.m > 0 and c.mt.n > 0 and c.mt.k > 0

    def test_default_range_bf16_gfx942(self):
        # Default (>8-bit) range: MN in {16,32,64,128,256}, K in {16,32,64,128,256,512}.
        # Cross-product = 5 * 5 * 6 = 150.
        problem = _make_default_configs_problem(origami.data_type_t.BFloat16)
        configs = origami.get_triton_default_configs(problem, HARDWARE["gfx942"])
        assert len(configs) == 5 * 5 * 6
        ms = {c.mt.m for c in configs}
        ns = {c.mt.n for c in configs}
        ks = {c.mt.k for c in configs}
        assert ms == {16, 32, 64, 128, 256}
        assert ns == {16, 32, 64, 128, 256}
        assert ks == {16, 32, 64, 128, 256, 512}

    def test_gfx950_f8_excludes_mn16(self):
        # gfx950 with <=8-bit narrow input: MN restricted to {32,64,128,256}.
        problem = _make_default_configs_problem(origami.data_type_t.Float8)
        configs = origami.get_triton_default_configs(problem, HARDWARE["gfx950"])
        assert len(configs) == 4 * 4 * 6
        ms = {c.mt.m for c in configs}
        ns = {c.mt.n for c in configs}
        assert 16 not in ms and 16 not in ns
        assert ms == {32, 64, 128, 256}
        assert ns == {32, 64, 128, 256}

    def test_gfx950_f4_excludes_mn16(self):
        # Same gating applies for sub-byte (F4) on gfx950.
        if not hasattr(origami.data_type_t, "Float4"):
            pytest.skip("Float4 not available in this build")
        problem = _make_default_configs_problem(origami.data_type_t.Float4)
        configs = origami.get_triton_default_configs(problem, HARDWARE["gfx950"])
        ms = {c.mt.m for c in configs}
        ns = {c.mt.n for c in configs}
        assert 16 not in ms and 16 not in ns

    def test_gfx942_f8_includes_mn512(self):
        # gfx942 with 8-bit narrow input adds MN=512.
        problem = _make_default_configs_problem(origami.data_type_t.Float8)
        configs = origami.get_triton_default_configs(problem, HARDWARE["gfx942"])
        assert len(configs) == 6 * 6 * 6
        ms = {c.mt.m for c in configs}
        ns = {c.mt.n for c in configs}
        assert 512 in ms and 512 in ns
        # Default 16-MN entries are still present on gfx942 F8.
        assert 16 in ms and 16 in ns

    def test_narrow_dtype_drives_gating_asymmetric(self):
        # Mixed precision: A=BF16 (16 bits), B=F8 (8 bits) → narrow = 8.
        # On gfx942 the 8-bit narrow input must enable the 512 extension.
        problem = _make_default_configs_problem(
            origami.data_type_t.BFloat16, origami.data_type_t.Float8
        )
        configs = origami.get_triton_default_configs(problem, HARDWARE["gfx942"])
        ms = {c.mt.m for c in configs}
        assert 512 in ms

    def test_mt_only_mi_left_unset(self):
        # The function only populates mt; mi is intentionally left at its
        # default-constructed value. Callers are expected to fill mi in.
        problem = _make_default_configs_problem(origami.data_type_t.Half)
        configs = origami.get_triton_default_configs(problem, HARDWARE["gfx942"])
        default_mi = origami.config_t().mi
        for c in configs:
            assert c.mi.m == default_mi.m
            assert c.mi.n == default_mi.n
            assert c.mi.k == default_mi.k

    def test_no_duplicate_configs(self):
        problem = _make_default_configs_problem(origami.data_type_t.Half)
        configs = origami.get_triton_default_configs(problem, HARDWARE["gfx942"])
        seen = {(c.mt.m, c.mt.n, c.mt.k) for c in configs}
        assert len(seen) == len(configs)

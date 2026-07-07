/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2026 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include "test_harness.hpp"

#include "origami/comm/types.hpp"

using namespace origami::comm;

// ─── ceil_div ───────────────────────────────────────────────────
TEST(ceil_div_matches_reference) {
  CHECK(ceil_div(0, 1) == 0);
  CHECK(ceil_div(1, 1) == 1);
  CHECK(ceil_div(7, 4) == 2);
  CHECK(ceil_div(8, 4) == 2);
  CHECK(ceil_div(9, 4) == 3);
  CHECK(ceil_div(std::size_t{1024 * 1024}, std::size_t{4096}) == 256);
}

// ─── dtype_bytes ─────────────────────────────────────────────────
TEST(dtype_bytes_matches_reference) {
  CHECK(dtype_bytes(data_type_t::Float8) == 1);
  CHECK(dtype_bytes(data_type_t::Int8) == 1);
  CHECK(dtype_bytes(data_type_t::Half) == 2);
  CHECK(dtype_bytes(data_type_t::BFloat16) == 2);
  CHECK(dtype_bytes(data_type_t::Float) == 4);
  CHECK(dtype_bytes(data_type_t::Double) == 8);
}

// ─── tile_shape_t: dense / contiguous regime ───────────────────────
TEST(tile_shape_contiguous_dense) {
  // 64 BF16 elements = 128 bytes = 2 cachelines, dense.
  tile_shape_t t{1, 64, data_type_t::BFloat16};
  CHECK(t.element_bytes() == 2);
  CHECK(t.bytes() == 128);
  CHECK(t.cachelines(64) == 2);
  auto [r, c] = t.cacheline_shape(64);
  CHECK(r == 1);
  CHECK(c == 2);
  CHECK_NEAR(t.cacheline_efficiency(64), 1.0, 1e-12);
}

TEST(tile_shape_contiguous_partial) {
  // 1 BF16 element = 2 bytes → rounds up to 1 cacheline (efficiency = 2/64).
  tile_shape_t t{1, 1, data_type_t::BFloat16};
  CHECK(t.bytes() == 2);
  CHECK(t.cachelines(64) == 1);
  CHECK_NEAR(t.cacheline_efficiency(64), 2.0 / 64.0, 1e-12);
}

// ─── tile_shape_t: non-contiguous (column-stripe) regime ───────────
TEST(tile_shape_non_contiguous_row_padding) {
  // Column stripe: each row is a 33-byte chunk (33/64 → 1 cl each).
  tile_shape_t t{8, 17, data_type_t::BFloat16, /*split_dim=*/1, /*contiguous=*/false};
  CHECK(t.element_bytes() == 2);
  CHECK(t.bytes() == 8 * 17 * 2);  // useful bytes
  CHECK(t.cl_per_row(64) == 1);    // 34B → 1 cl
  CHECK(t.cachelines(64) == 8);    // 8 rows × 1 cl
  auto [r, c] = t.cacheline_shape(64);
  CHECK(r == 8);
  CHECK(c == 1);
}

// ─── tile_shape_t::divide ──────────────────────────────────────────
TEST(tile_shape_divide_axis0_preserves_contiguity) {
  tile_shape_t t{8, 64, data_type_t::BFloat16};
  auto sub = t.divide(4, /*axis=*/0);
  CHECK(sub.m == 2);
  CHECK(sub.n == 64);
  CHECK(sub.contiguous);
}

TEST(tile_shape_divide_axis1_breaks_contiguity) {
  tile_shape_t t{8, 64, data_type_t::BFloat16};
  auto sub = t.divide(4, /*axis=*/1);
  CHECK(sub.m == 8);
  CHECK(sub.n == 16);
  CHECK(sub.split_dim == 1);
  CHECK(!sub.contiguous);
}

TEST(tile_shape_divide_byte_equal_prefers_axis0) {
  tile_shape_t t{8, 64, data_type_t::BFloat16};
  auto sub = t.divide_byte_equal(4);
  CHECK(sub.m == 2);
  CHECK(sub.n == 64);
  CHECK(sub.contiguous);
}

TEST(tile_shape_divide_byte_equal_falls_back_to_axis1) {
  // m < factor → must split along axis=1.
  tile_shape_t t{1, 64, data_type_t::BFloat16};
  auto sub = t.divide_byte_equal(4);
  CHECK(sub.m == 1);
  CHECK(sub.n == 16);
  CHECK(!sub.contiguous);
}

TEST(tile_shape_divide_factor_one_is_identity) {
  tile_shape_t t{4, 32, data_type_t::BFloat16};
  auto sub = t.divide_byte_equal(1);
  CHECK(sub.m == t.m);
  CHECK(sub.n == t.n);
  CHECK(sub.contiguous == t.contiguous);
}

// ─── comm_problem_t ────────────────────────────────────────────────
TEST(comm_problem_split_dim_0) {
  // 1024×512 BF16 across 8 GPUs, split along rows.
  comm_problem_t cp{1024, 512, 8, data_type_t::BFloat16, 0};
  CHECK(cp.message_bytes() == 1024ULL * 512ULL * 2ULL);
  CHECK(cp.gpu_tile_m() == 128);
  CHECK(cp.gpu_tile_n() == 512);
  CHECK(cp.gpu_tile_bytes() == 128ULL * 512ULL * 2ULL);
  auto t = cp.gpu_tile_shape();
  CHECK(t.contiguous);
  // Dense: bytes / 64 = 2048 cl.
  CHECK(cp.gpu_tile_cachelines(64) == 128ULL * 512ULL * 2ULL / 64);
}

TEST(comm_problem_split_dim_1) {
  // Same problem, split along cols.
  comm_problem_t cp{1024, 512, 8, data_type_t::BFloat16, 1};
  CHECK(cp.gpu_tile_m() == 1024);
  CHECK(cp.gpu_tile_n() == 64);
  CHECK(cp.gpu_tile_bytes() == 1024ULL * 64ULL * 2ULL);
}

// ─── comm_config_t::effective_num_wgs ──────────────────────────────
TEST(comm_config_effective_num_wgs_caps_when_undersized) {
  comm_config_t c{};
  c.num_wgs          = 304;
  c.min_bytes_per_wg = 16384;
  // 4 × 16384 = 65536 bytes → 4 WGs fit.
  CHECK(c.effective_num_wgs(4 * 16384) == 4);
  // 100 KiB → 6 WGs fit (102400 / 16384 = 6 floor).
  CHECK(c.effective_num_wgs(100ULL * 1024ULL) == 6);
  // Huge tile → caps at num_wgs.
  CHECK(c.effective_num_wgs(1024ULL * 1024ULL * 1024ULL) == 304);
}

TEST(comm_config_effective_num_wgs_disabled_by_zero) {
  comm_config_t c{};
  c.num_wgs          = 304;
  c.min_bytes_per_wg = 0;
  CHECK(c.effective_num_wgs(1) == 304);
}

TEST(comm_config_effective_num_wgs_floor_one) {
  comm_config_t c{};
  c.num_wgs          = 304;
  c.min_bytes_per_wg = 16384;
  // tile_bytes < min_bytes_per_wg → still 1 active WG.
  CHECK(c.effective_num_wgs(1) == 1);
  CHECK(c.effective_num_wgs(0) == 1);
}

// ─── functional_unit_work_t sum ─────────────────────────────────────
TEST(fu_work_sums_componentwise) {
  functional_unit_work_t a{};
  a.vmem_read_instrs = 3;
  a.hbm_write_cl     = 5;

  functional_unit_work_t b{};
  b.vmem_read_instrs = 7;
  b.l2_read_cl       = 11;

  auto c = a + b;
  CHECK(c.vmem_read_instrs == 10);
  CHECK(c.hbm_write_cl == 5);
  CHECK(c.l2_read_cl == 11);
  CHECK(c.atomic_count == 0);
}

// ─── wg_tile_latency_breakdown_t cycles ↔ ns ─────────────────────────
TEST(wg_breakdown_cycles_to_ns_at_mi300x) {
  wg_tile_latency_breakdown_t b;
  b.clock_ghz      = 2.0;
  b.T_total_cycles = 1000.0;  // = 500 ns
  b.T_wlt_cycles   = 200.0;   // = 100 ns
  CHECK_NEAR(b.T_total(), 500.0, 1e-9);
  CHECK_NEAR(b.T_wlt(), 100.0, 1e-9);
}

ORIGAMI_TEST_MAIN()

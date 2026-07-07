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

#include "origami/comm/heuristics.hpp"

using namespace origami::comm;

// ─── Default heuristic values ───────────────────────────────────
TEST(default_heuristics_min_bytes_per_wg) { CHECK(DEFAULT_HEURISTICS.min_bytes_per_wg == 16'384); }

TEST(default_heuristics_xgmi_k_default) {
  CHECK_NEAR(DEFAULT_HEURISTICS.xgmi_write_concentration_k_default, 4.0, 1e-12);
}

TEST(default_heuristics_xgmi_k_by_primitive_enum) {
  CHECK_NEAR(DEFAULT_HEURISTICS.k_xgmi_write(primitive_t::all_gather), 4.0, 1e-12);
  CHECK_NEAR(DEFAULT_HEURISTICS.k_xgmi_write(primitive_t::reduce_scatter), 3.0, 1e-12);
  CHECK_NEAR(DEFAULT_HEURISTICS.k_xgmi_write(primitive_t::broadcast), 3.5, 1e-12);
  CHECK_NEAR(DEFAULT_HEURISTICS.k_xgmi_write(primitive_t::all_reduce), 6.0, 1e-12);
  CHECK_NEAR(DEFAULT_HEURISTICS.k_xgmi_write(primitive_t::all_to_all), 4.0, 1e-12);
}

TEST(default_heuristics_xgmi_k_by_primitive_string) {
  CHECK_NEAR(DEFAULT_HEURISTICS.k_xgmi_write("all_gather"), 4.0, 1e-12);
  CHECK_NEAR(DEFAULT_HEURISTICS.k_xgmi_write("reduce_scatter"), 3.0, 1e-12);
  CHECK_NEAR(DEFAULT_HEURISTICS.k_xgmi_write("broadcast"), 3.5, 1e-12);
  CHECK_NEAR(DEFAULT_HEURISTICS.k_xgmi_write("all_reduce"), 6.0, 1e-12);
  CHECK_NEAR(DEFAULT_HEURISTICS.k_xgmi_write("all_to_all"), 4.0, 1e-12);
  // Unknown → default.
  CHECK_NEAR(DEFAULT_HEURISTICS.k_xgmi_write("not_a_primitive"), 4.0, 1e-12);
}

TEST(default_heuristics_ring_step_overhead_ns) {
  // Stored in host nanoseconds (clock-invariant); the engine converts to cycles
  // at the target GPU clock.
  CHECK_NEAR(DEFAULT_HEURISTICS.ring_step_overhead_ns_for(primitive_t::all_gather), 10000.0, 1e-9);
  CHECK_NEAR(
      DEFAULT_HEURISTICS.ring_step_overhead_ns_for(primitive_t::reduce_scatter), 4000.0, 1e-9);
  CHECK_NEAR(DEFAULT_HEURISTICS.ring_step_overhead_ns_for(primitive_t::broadcast), 0.0, 1e-9);
  CHECK_NEAR(DEFAULT_HEURISTICS.ring_step_overhead_ns_for(primitive_t::all_reduce), 0.0, 1e-9);
  CHECK_NEAR(DEFAULT_HEURISTICS.ring_step_overhead_ns_for(primitive_t::all_to_all), 0.0, 1e-9);
  // String overload.
  CHECK_NEAR(DEFAULT_HEURISTICS.ring_step_overhead_ns_for("all_gather"), 10000.0, 1e-9);
  CHECK_NEAR(DEFAULT_HEURISTICS.ring_step_overhead_ns_for("???"), 0.0, 1e-9);
}

TEST(default_heuristics_framework_overhead_us) {
  // torch = 400_000 ns = 400 µs.
  CHECK_NEAR(DEFAULT_HEURISTICS.framework_overhead_us(framework_t::torch), 400.0, 1e-12);
  CHECK_NEAR(DEFAULT_HEURISTICS.framework_overhead_us(framework_t::raw), 0.0, 1e-12);
  CHECK_NEAR(DEFAULT_HEURISTICS.framework_overhead_us(framework_t::nccl), 0.0, 1e-12);
  CHECK_NEAR(DEFAULT_HEURISTICS.framework_overhead_us("torch"), 400.0, 1e-12);
  CHECK_NEAR(DEFAULT_HEURISTICS.framework_overhead_us("unknown"), 0.0, 1e-12);
}

TEST(custom_heuristics_override) {
  heuristics_t h{};
  h.xgmi_write_concentration_k_by_primitive[static_cast<std::size_t>(primitive_t::all_gather)] =
      9.5;
  h.min_bytes_per_wg = 1024;
  CHECK_NEAR(h.k_xgmi_write(primitive_t::all_gather), 9.5, 1e-12);
  CHECK(h.min_bytes_per_wg == 1024);
  // Default-instance unchanged.
  CHECK_NEAR(DEFAULT_HEURISTICS.k_xgmi_write(primitive_t::all_gather), 4.0, 1e-12);
}

ORIGAMI_TEST_MAIN()

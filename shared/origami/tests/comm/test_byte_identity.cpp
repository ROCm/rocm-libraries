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

// Every exposed constant must equal the reference value to within machine
// epsilon. These reference values are stored at full precision so the gold
// values *are* the exact IEEE-754 doubles expected. Tolerance is 0 (==)
// where the value is an exact integer reciprocal of an integer; 1e-12
// otherwise to absorb last-bit reordering in the C++ constant-folder.
#include "test_harness.hpp"
#include "test_system.hpp"

#include "origami/comm/hardware.hpp"
#include "origami/comm/heuristics.hpp"

using namespace origami::comm;

// ─── Structural ─────────────────────────────────────────────────
TEST(byteid_mi300x_die_structure) {
  CHECK(MI300X.num_cu == 304);
  CHECK(MI300X.num_xcd == 8);
  CHECK(MI300X.cu_per_xcd == 38);
  CHECK_NEAR(MI300X.clock_ghz, 2.0, 0.0);
}

TEST(byteid_mi300x_per_cu_rates) {
  CHECK_NEAR(MI300X.vmem_issue_rate, 1.0, 0.0);
  CHECK_NEAR(MI300X.valu_rate, 2.10 * 64, 1e-12);  // = 134.4
  CHECK_NEAR(MI300X.tcp_bw, 64.0, 0.0);
}

TEST(byteid_mi300x_latencies) {
  CHECK_NEAR(MI300X.xgmi_latency_cycles, 1320.0, 0.0);           // 660 × 2
  CHECK_NEAR(MI300X_COMM.atomic_latency_cycles, 200.0, 0.0);     // 100 × 2
  CHECK_NEAR(MI300X_COMM.launch_overhead_cycles, 90000.0, 0.0);  // 45000 × 2
}

TEST(byteid_mi300x_bandwidths) {
  // L2: 83.6 / 2 = 41.8 (exact).
  CHECK_NEAR(MI300X.l2_bw_per_cu, 41.8, 0.0);
  // MALL: 4730 / 2 = 2365 (exact).
  CHECK_NEAR(MI300X.mall_bw, 2365.0, 0.0);
  // HBM: 4730/2 and 5140/2 (exact).
  CHECK_NEAR(MI300X.hbm_read_bw, 2365.0, 0.0);
  CHECK_NEAR(MI300X.hbm_write_bw, 2570.0, 0.0);
}

TEST(byteid_mi300x_comm_link_bw) {
  // 49.1 * (1024**3) / 1e9 / 1.23 / 2.0 = 21.431188438373987
  CHECK_NEAR(MI300X_COMM.link_bw, 21.431188438373987, 1e-15);
}

TEST(byteid_mi300x_comm_sdma) {
  // 49.5 / 2 and 23.6 / 2 — both exact in double.
  CHECK_NEAR(MI300X_COMM.sdma_read_bw, 24.75, 0.0);
  CHECK_NEAR(MI300X_COMM.sdma_write_bw, 11.8, 1e-15);
}

TEST(byteid_mi300x_mem_bw_coeffs) {
  CHECK_NEAR(MI300X.mem_bw_coeffs[0], 0.0, 0.0);
  CHECK_NEAR(MI300X.mem_bw_coeffs[1], 0.015, 0.0);
  CHECK_NEAR(MI300X.mem_bw_coeffs[2], 0.0, 0.0);
}

// ─── heuristics_t ─────────────────────────────────────────────────
TEST(byteid_default_heuristics_ring_step_overhead) {
  // Stored in host nanoseconds (exact).
  CHECK_NEAR(DEFAULT_HEURISTICS.ring_step_overhead_ns_for(primitive_t::all_gather), 10000.0, 0.0);
  CHECK_NEAR(
      DEFAULT_HEURISTICS.ring_step_overhead_ns_for(primitive_t::reduce_scatter), 4000.0, 0.0);
  CHECK_NEAR(DEFAULT_HEURISTICS.ring_step_overhead_ns_for(primitive_t::broadcast), 0.0, 0.0);
  CHECK_NEAR(DEFAULT_HEURISTICS.ring_step_overhead_ns_for(primitive_t::all_reduce), 0.0, 0.0);
  CHECK_NEAR(DEFAULT_HEURISTICS.ring_step_overhead_ns_for(primitive_t::all_to_all), 0.0, 0.0);
  // At the MI300X 2.0 GHz clock the engine converts these to the historical
  // cycle values (ns × clock_ghz: 10000 × 2 = 20000, 4000 × 2 = 8000), so
  // end-to-end latencies remain byte-identical.
  CHECK_NEAR(
      DEFAULT_HEURISTICS.ring_step_overhead_ns_for(primitive_t::all_gather) * MI300X_COMM.clock_ghz,
      20000.0,
      0.0);
  CHECK_NEAR(DEFAULT_HEURISTICS.ring_step_overhead_ns_for(primitive_t::reduce_scatter) *
                 MI300X_COMM.clock_ghz,
             8000.0,
             0.0);
}

TEST(byteid_default_heuristics_framework_overhead) {
  // torch only: 400_000 ns = 400 µs (exact).
  CHECK_NEAR(DEFAULT_HEURISTICS.framework_overhead_us(framework_t::torch), 400.0, 0.0);
  CHECK_NEAR(DEFAULT_HEURISTICS.framework_overhead_us(framework_t::raw), 0.0, 0.0);
}

ORIGAMI_TEST_MAIN()

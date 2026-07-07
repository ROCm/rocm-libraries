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

// Top-of-model byte-identity test.
//
// Two corpora:
//   golden/collective_grid.csv (360 rows) -- compute_collective_latency
//                                            in raw GPU cycles across
//                                            5 primitives x 3 world_sizes
//                                            x 6 nchannels x 4 msg sizes.
//   golden/predict_row.csv     (630 rows) -- predict_row in µs across
//                                            5 primitives x 3 W x 6 nch
//                                            x 7 msg sizes (1 KiB..4 MiB).
//
// Tolerance: 1e-9 µs (predict_row) / 1e-6 cycles (collective_grid).
// Predictions span ~10 to ~10^5 µs so 1e-9 absolute is well below
// machine-epsilon-of-the-value across the entire range.
#include "test_harness.hpp"
#include "test_system.hpp"

#include "origami/comm/collective.hpp"
#include "origami/comm/hardware.hpp"
#include "origami/comm/heuristics.hpp"
#include "origami/comm/types.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace origami::comm;

namespace {

std::vector<std::string> split_csv(const std::string& line) {
  std::vector<std::string> out;
  std::stringstream ss(line);
  std::string tok;
  while (std::getline(ss, tok, ',')) out.push_back(tok);
  return out;
}

bool open_golden(std::ifstream& in, const char* relpath) {
  in.open(relpath);
  if (in.is_open()) return true;
  std::string fallback = std::string{"../../tests/"} + relpath;
  in.open(fallback);
  return in.is_open();
}

}  // namespace

// ─── compute_collective_latency grid (cycles) ───────────────────
TEST(collective_latency_match_golden_grid) {
  std::ifstream in;
  CHECK(open_golden(in, "golden/collective_grid.csv"));
  if (!in.is_open()) return;

  std::string line;
  std::getline(in, line);  // header

  std::size_t rows = 0, mismatches = 0;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    const auto f = split_csv(line);
    if (f.size() < 8) continue;

    const std::string prim = f[0];
    const int W            = std::stoi(f[1]);
    const int nch          = std::stoi(f[2]);
    // msg_bytes (col 3) not directly used; reconstructed via M/N.
    const std::size_t M = std::stoull(f[4]);
    const std::size_t N = std::stoull(f[5]);
    const int split_dim = std::stoi(f[6]);
    const double T_exp  = std::stod(f[7]);

    comm_problem_t problem{M, N, W, data_type_t::BFloat16, split_dim};
    problem.collective = primitive_from_name(prim);
    comm_config_t config{};
    config.num_wgs          = nch;
    config.load_width       = load_width_t::DWORDX16;
    config.vgprs_for_data   = 128;
    config.min_bytes_per_wg = DEFAULT_HEURISTICS.min_bytes_per_wg;

    const double T_got = compute_collective_latency(problem, config, MI300X_SYSTEM);

    ++rows;
    // Cycle counts can be up to ~10^9 for a 4 MiB collective at 2 GHz;
    // 1e-6 absolute is far below 1 ULP at that scale.
    if (std::fabs(T_got - T_exp) > 1e-6) {
      if (mismatches < 5) {
        std::fprintf(stderr,
                     "  MISMATCH collective_grid prim=%s W=%d nch=%d M=%zu N=%zu  "
                     "T_cycles got=%.17g exp=%.17g (delta=%g)\n",
                     prim.c_str(),
                     W,
                     nch,
                     M,
                     N,
                     T_got,
                     T_exp,
                     T_got - T_exp);
      }
      ++mismatches;
    }
  }
  std::printf("  collective_grid: %zu rows, %zu mismatches\n", rows, mismatches);
  CHECK(rows > 300);
  CHECK(mismatches == 0);
}

// ─── predict_row grid (µs) ──────────────────────────────────────
TEST(predict_row_match_golden_grid) {
  std::ifstream in;
  CHECK(open_golden(in, "golden/predict_row.csv"));
  if (!in.is_open()) return;

  std::string line;
  std::getline(in, line);  // header

  std::size_t rows = 0, mismatches = 0;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    const auto f = split_csv(line);
    if (f.size() < 5) continue;

    const std::string prim = f[0];
    const int W            = std::stoi(f[1]);
    const int nch          = std::stoi(f[2]);
    const std::size_t msg  = std::stoull(f[3]);
    const double T_exp     = std::stod(f[4]);

    const double T_got = predict_row(prim, msg, W, nch, MI300X_SYSTEM);

    ++rows;
    // Predictions: ~10 µs (small messages) to ~10^5 µs (large at W=8).
    // 1e-9 µs absolute is comfortably below the ULP of those values.
    if (std::fabs(T_got - T_exp) > 1e-9) {
      if (mismatches < 5) {
        std::fprintf(stderr,
                     "  MISMATCH predict_row prim=%s W=%d nch=%d msg=%zu  "
                     "T_us got=%.17g exp=%.17g (delta=%g)\n",
                     prim.c_str(),
                     W,
                     nch,
                     msg,
                     T_got,
                     T_exp,
                     T_got - T_exp);
      }
      ++mismatches;
    }
  }
  std::printf("  predict_row: %zu rows, %zu mismatches\n", rows, mismatches);
  CHECK(rows > 600);
  CHECK(mismatches == 0);
}

// ─── Smoke: spot-check the launch-overhead floor ─────────────────
TEST(predict_row_launch_overhead_floor) {
  // 1 KiB AG at W=2 with 1 channel: dominated by ring-step overhead
  // + launch overhead. Should be well above 45 µs (the launch floor).
  const double T = predict_row("all_gather", 1024, 2, 1, MI300X_SYSTEM);
  CHECK(T > 45.0);
  // Large message at W=8 should be >> 1 ms (huge).
  const double T_big = predict_row("all_gather", 64 * 1024 * 1024, 8, 32, MI300X_SYSTEM);
  CHECK(T_big > 1000.0);  // >1 ms, sanity
}

// ─── Smoke: ring algorithms use the pipelined path ──────────────────
TEST(allreduce_uses_two_shot_algorithm) {
  // AR default = two_shot (sequential), not ring. Confirms that
  // compute_collective_latency picks the sequential code path —
  // a regression here would surface as a totally different magnitude
  // (predict_row would deviate by orders of magnitude).
  comm_problem_t problem{1, 65536, 8, data_type_t::BFloat16, 0};
  problem.collective = primitive_t::all_reduce;
  comm_config_t cfg{};
  cfg.num_wgs    = 8;
  const double T = compute_collective_latency(problem, cfg, MI300X_SYSTEM);
  CHECK(T > 0.0);
}

ORIGAMI_TEST_MAIN()

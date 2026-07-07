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

// predict_tensor_collective byte-identity regression against the golden corpus.
//
// Grid: 5 ops × 4 shapes × 2 dtypes × 4 world_sizes × 3 nchannels × 2 frameworks
//       × 2 dims = 1,920 rows.
//
// Tolerance: 1e-9 µs absolute on predicted_us / backend_us /
// framework_overhead_us; exact match on int-valued fields
// (per_rank_bytes, wire_bytes_per_rank, msg_bytes, gpu_tile dims).
#include "test_harness.hpp"
#include "test_system.hpp"

#include "origami/comm/tensor.hpp"

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

// "16x4096" → {16, 4096}
std::vector<std::size_t> parse_shape(const std::string& s) {
  std::vector<std::size_t> out;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, 'x')) out.push_back(std::stoull(tok));
  return out;
}

}  // namespace

TEST(predict_tensor_collective_match_golden_grid) {
  std::ifstream in{"golden/tensor_collective.csv"};
  if (!in.is_open()) in.open("../../tests/golden/tensor_collective.csv");
  CHECK(in.is_open());
  if (!in.is_open()) return;

  std::string line;
  std::getline(in, line);  // header

  std::size_t rows = 0, mismatches = 0;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    const auto f = split_csv(line);
    if (f.size() < 16) continue;

    const std::string op           = f[0];
    const auto shape               = parse_shape(f[1]);
    const std::string dtype_str    = f[2];
    const int W                    = std::stoi(f[3]);
    const int dim                  = std::stoi(f[4]);
    const int nch                  = std::stoi(f[5]);
    const std::string framework    = f[6];
    const double exp_pred          = std::stod(f[7]);
    const double exp_backend       = std::stod(f[8]);
    const double exp_fwoh          = std::stod(f[9]);
    const std::size_t exp_per_rank = std::stoull(f[10]);
    const std::size_t exp_wire     = std::stoull(f[11]);
    const std::size_t exp_msg      = std::stoull(f[12]);
    const std::size_t exp_tile_m   = std::stoull(f[13]);
    const std::size_t exp_tile_n   = std::stoull(f[14]);
    const int exp_tile_sd          = std::stoi(f[15]);

    const auto p = predict_tensor_collective(
        op, shape, dtype_str, W, MI300X_SYSTEM, dim, nch, framework, DEFAULT_HEURISTICS);

    ++rows;
    bool row_bad = false;
    auto bump    = [&](const char* what, double got, double exp) {
      if (std::fabs(got - exp) > 1e-9) {
        if (mismatches < 5) {
          std::fprintf(stderr,
                       "  MISMATCH op=%s W=%d dim=%d nch=%d fw=%s shape=%s  "
                       "%s got=%.17g exp=%.17g\n",
                       op.c_str(),
                       W,
                       dim,
                       nch,
                       framework.c_str(),
                       f[1].c_str(),
                       what,
                       got,
                       exp);
        }
        row_bad = true;
      }
    };
    bump("predicted_us", p.predicted_us, exp_pred);
    bump("backend_us", p.backend_us(), exp_backend);
    bump("framework_overhead_us", p.framework_overhead_us, exp_fwoh);

    auto bump_int = [&](const char* what, std::size_t got, std::size_t exp) {
      if (got != exp) {
        if (mismatches < 5) {
          std::fprintf(stderr,
                       "  MISMATCH op=%s W=%d dim=%d nch=%d fw=%s shape=%s  "
                       "%s got=%zu exp=%zu\n",
                       op.c_str(),
                       W,
                       dim,
                       nch,
                       framework.c_str(),
                       f[1].c_str(),
                       what,
                       got,
                       exp);
        }
        row_bad = true;
      }
    };
    bump_int("per_rank_bytes", p.per_rank_bytes, exp_per_rank);
    bump_int("wire_bytes_per_rank", p.wire_bytes_per_rank, exp_wire);
    bump_int("msg_bytes", p.msg_bytes, exp_msg);
    bump_int("gpu_tile.m", p.gpu_tile.m, exp_tile_m);
    bump_int("gpu_tile.n", p.gpu_tile.n, exp_tile_n);

    if (p.gpu_tile.split_dim != exp_tile_sd) {
      if (mismatches < 5) {
        std::fprintf(stderr,
                     "  MISMATCH op=%s W=%d dim=%d nch=%d  gpu_tile.split_dim "
                     "got=%d exp=%d\n",
                     op.c_str(),
                     W,
                     dim,
                     nch,
                     p.gpu_tile.split_dim,
                     exp_tile_sd);
      }
      row_bad = true;
    }
    if (row_bad) ++mismatches;
  }
  std::printf("  tensor_collective: %zu rows, %zu mismatches\n", rows, mismatches);
  CHECK(rows > 1900);
  CHECK(mismatches == 0);
}

// ─── Spot checks ────────────────────────────────────────────────
TEST(framework_overhead_torch_adds_400us) {
  const auto raw_p = predict_tensor_collective(
      "all_gather", {4096}, "bf16", 8, MI300X_SYSTEM, 0, 32, "raw", DEFAULT_HEURISTICS);
  const auto torch_p = predict_tensor_collective(
      "all_gather", {4096}, "bf16", 8, MI300X_SYSTEM, 0, 32, "torch", DEFAULT_HEURISTICS);
  CHECK_NEAR(torch_p.predicted_us - raw_p.predicted_us, 400.0, 1e-12);
  CHECK_NEAR(torch_p.backend_us(), raw_p.backend_us(), 1e-12);
}

TEST(dtype_aliases_resolve) {
  CHECK(normalize_dtype("bf16") == data_type_t::BFloat16);
  CHECK(normalize_dtype("bfloat16") == data_type_t::BFloat16);
  CHECK(normalize_dtype("torch.bf16") == data_type_t::BFloat16);
  CHECK(normalize_dtype("FP32") == data_type_t::Float);
  CHECK(normalize_dtype("float") == data_type_t::Float);
  CHECK(normalize_dtype("half") == data_type_t::Half);
}

TEST(shape_lowering_split_dim_picks_last_or_outer) {
  // dim=-1 on (16, 4096) → split_dim=1, N_full = 4096*W.
  const auto a = per_rank_shape_to_full_mn({16, 4096}, -1, 8);
  CHECK(a.M_full == 16);
  CHECK(a.N_full == 4096 * 8);
  CHECK(a.split_dim == 1);
  // dim=0  on (16, 4096) → split_dim=0, M_full = 16*W.
  const auto b = per_rank_shape_to_full_mn({16, 4096}, 0, 8);
  CHECK(b.M_full == 16 * 8);
  CHECK(b.N_full == 4096);
  CHECK(b.split_dim == 0);
}

TEST(world_size_1_is_no_op_plus_framework_overhead) {
  const auto p = predict_tensor_collective(
      "all_reduce", {4096}, "bf16", 1, MI300X_SYSTEM, 0, 8, "torch", DEFAULT_HEURISTICS);
  CHECK_NEAR(p.predicted_us, 400.0, 1e-12);
  CHECK_NEAR(p.backend_us(), 0.0, 1e-12);
  CHECK(p.wire_bytes_per_rank == 0);
}

ORIGAMI_TEST_MAIN()

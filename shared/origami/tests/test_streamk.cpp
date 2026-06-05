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

#include <catch2/catch_test_macros.hpp>
#include "common.hpp"

namespace {

inline origami::problem_t make_problem_for_tiles_per_cu(
    size_t mt_m, size_t mt_n, double tiles_per_cu, size_t cu_count, size_t batch = 1) {
  const size_t target_tiles =
      static_cast<size_t>(tiles_per_cu * static_cast<double>(cu_count) + 0.5);
  size_t tiles_per_dim = 1;
  while (tiles_per_dim * tiles_per_dim < target_tiles) ++tiles_per_dim;
  const size_t m = tiles_per_dim * mt_m;
  const size_t n = tiles_per_dim * mt_n;
  return make_problem(m, n, /*k=*/64,
                      origami::transpose_t::T, origami::transpose_t::N,
                      batch);
}

}

TEST_CASE("Origami streamk: select_hybrid_mode 16x16 short-circuits to static_",
          "[origami][streamk][hybrid]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch) {
      auto hardware = make_hardware(gpu_arch);
      auto config   = make_config(16, 16, 32);
      auto problem  = make_problem(8192, 8192, 64);
      auto mode = origami::streamk::select_hybrid_mode(
          problem, hardware, config, /*sm_count_target=*/0);
      REQUIRE(mode == origami::hybrid_mode_t::static_);
    }
  }
}

TEST_CASE("Origami streamk: select_hybrid_mode 32x32 threshold (256.4)",
          "[origami][streamk][hybrid]") {
  auto hardware = make_hardware(950);
  auto config   = make_config(32, 32, 64);
  auto small = make_problem_for_tiles_per_cu(32, 32, 200.0, hardware.N_CU);
  REQUIRE(origami::streamk::select_hybrid_mode(small, hardware, config, 0)
          == origami::hybrid_mode_t::static_);
  auto big = make_problem_for_tiles_per_cu(32, 32, 500.0, hardware.N_CU);
  REQUIRE(origami::streamk::select_hybrid_mode(big, hardware, config, 0)
          == origami::hybrid_mode_t::dynamic);
}

TEST_CASE("Origami streamk: select_hybrid_mode 64x64 threshold (7.22)",
          "[origami][streamk][hybrid]") {
  auto hardware = make_hardware(950);
  auto config   = make_config(64, 64, 32);
  auto small = make_problem_for_tiles_per_cu(64, 64, 4.0, hardware.N_CU);
  REQUIRE(origami::streamk::select_hybrid_mode(small, hardware, config, 0)
          == origami::hybrid_mode_t::static_);
  auto big = make_problem_for_tiles_per_cu(64, 64, 20.0, hardware.N_CU);
  REQUIRE(origami::streamk::select_hybrid_mode(big, hardware, config, 0)
          == origami::hybrid_mode_t::dynamic);
}

TEST_CASE("Origami streamk: select_hybrid_mode 128x128 threshold (2.08)",
          "[origami][streamk][hybrid]") {
  auto hardware = make_hardware(950);
  auto config   = make_config(128, 128, 32);
  auto small = make_problem_for_tiles_per_cu(128, 128, 1.0, hardware.N_CU);
  REQUIRE(origami::streamk::select_hybrid_mode(small, hardware, config, 0)
          == origami::hybrid_mode_t::static_);
  auto big = make_problem_for_tiles_per_cu(128, 128, 8.0, hardware.N_CU);
  REQUIRE(origami::streamk::select_hybrid_mode(big, hardware, config, 0)
          == origami::hybrid_mode_t::dynamic);
}

TEST_CASE("Origami streamk: select_hybrid_mode 128x256 threshold (2.58)",
          "[origami][streamk][hybrid]") {
  auto hardware = make_hardware(950);
  auto config   = make_config(128, 256, 64);
  auto small = make_problem_for_tiles_per_cu(128, 256, 1.0, hardware.N_CU);
  REQUIRE(origami::streamk::select_hybrid_mode(small, hardware, config, 0)
          == origami::hybrid_mode_t::static_);
  auto big = make_problem_for_tiles_per_cu(128, 256, 10.0, hardware.N_CU);
  REQUIRE(origami::streamk::select_hybrid_mode(big, hardware, config, 0)
          == origami::hybrid_mode_t::dynamic);
}

TEST_CASE("Origami streamk: select_hybrid_mode 256x128 threshold (0.87)",
          "[origami][streamk][hybrid]") {
  auto hardware = make_hardware(950);
  auto config   = make_config(256, 128, 64);
  auto small = make_problem_for_tiles_per_cu(256, 128, 0.5, hardware.N_CU);
  REQUIRE(origami::streamk::select_hybrid_mode(small, hardware, config, 0)
          == origami::hybrid_mode_t::static_);
  auto big = make_problem_for_tiles_per_cu(256, 128, 4.0, hardware.N_CU);
  REQUIRE(origami::streamk::select_hybrid_mode(big, hardware, config, 0)
          == origami::hybrid_mode_t::dynamic);
}

TEST_CASE("Origami streamk: select_hybrid_mode unknown MT uses 2.0 default",
          "[origami][streamk][hybrid]") {
  auto hardware = make_hardware(950);
  auto config   = make_config(192, 192, 32);
  auto small = make_problem_for_tiles_per_cu(192, 192, 1.0, hardware.N_CU);
  REQUIRE(origami::streamk::select_hybrid_mode(small, hardware, config, 0)
          == origami::hybrid_mode_t::static_);
  auto big = make_problem_for_tiles_per_cu(192, 192, 5.0, hardware.N_CU);
  REQUIRE(origami::streamk::select_hybrid_mode(big, hardware, config, 0)
          == origami::hybrid_mode_t::dynamic);
}

TEST_CASE("Origami streamk: select_hybrid_mode sm_count_target clamps N_CU",
          "[origami][streamk][hybrid]") {
  auto hardware = make_hardware(950);
  auto config   = make_config(128, 128, 32);
  auto problem = make_problem_for_tiles_per_cu(128, 128, 1.5, hardware.N_CU);
  REQUIRE(origami::streamk::select_hybrid_mode(problem, hardware, config, 0)
          == origami::hybrid_mode_t::static_);
  REQUIRE(origami::streamk::select_hybrid_mode(
              problem, hardware, config, hardware.N_CU / 2)
          == origami::hybrid_mode_t::dynamic);
}

TEST_CASE("Origami streamk: select_hybrid_mode batch multiplies tiles",
          "[origami][streamk][hybrid]") {
  auto hardware = make_hardware(950);
  auto config   = make_config(128, 128, 32);
  auto base = make_problem_for_tiles_per_cu(128, 128, 1.5, hardware.N_CU, /*batch=*/1);
  REQUIRE(origami::streamk::select_hybrid_mode(base, hardware, config, 0)
          == origami::hybrid_mode_t::static_);
  auto base_b4 = base;
  base_b4.batch = 4;
  REQUIRE(origami::streamk::select_hybrid_mode(base_b4, hardware, config, 0)
          == origami::hybrid_mode_t::dynamic);
}

TEST_CASE("Origami streamk: select_hybrid_mode sm_count_target=0 uses N_CU",
          "[origami][streamk][hybrid]") {
  auto hardware = make_hardware(950);
  auto config   = make_config(128, 128, 32);
  auto problem  = make_problem(4096, 4096, 64);
  auto a = origami::streamk::select_hybrid_mode(problem, hardware, config, 0);
  auto b = origami::streamk::select_hybrid_mode(problem, hardware, config, hardware.N_CU);
  REQUIRE(a == b);
}

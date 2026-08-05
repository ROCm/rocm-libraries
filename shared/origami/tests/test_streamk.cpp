/*******************************************************************************
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 *******************************************************************************/

#include <catch2/catch_test_macros.hpp>
#include "common.hpp"

namespace {

using origami::streamk_hybrid_defaults_t;

// Builds a problem with exactly `tiles` output tiles for the given macrotile
// (a single row of `tiles` tile columns), so callers get precise control over
// the tile count fed into select_hybrid_mode's gates.
inline origami::problem_t make_problem_with_tile_count(
    size_t mt_m, size_t mt_n, size_t tiles, size_t batch = 1) {
  return make_problem(/*m=*/mt_m, /*n=*/mt_n * tiles, /*k=*/64,
                      origami::transpose_t::T, origami::transpose_t::N,
                      batch);
}

}

TEST_CASE("Origami streamk: select_hybrid_mode tile-count gate is inclusive of the threshold",
          "[origami][streamk][hybrid]") {
  // Even with a cotenant and occupancy low enough to otherwise force dynamic
  // unconditionally, a grid at or below MIN_TILES_FOR_DYNAMIC stays static_.
  auto hardware = make_hardware(950);
  auto config   = make_config(1, 1, 32, 16, 16, 16, false, 1, /*occupancy=*/1);
  auto at_gate  = make_problem_with_tile_count(1, 1, streamk_hybrid_defaults_t::MIN_TILES_FOR_DYNAMIC);
  REQUIRE(origami::streamk::select_hybrid_mode(at_gate, hardware, config, hardware.N_CU / 2)
          == origami::hybrid_mode_t::static_);

  auto above_gate =
      make_problem_with_tile_count(1, 1, streamk_hybrid_defaults_t::MIN_TILES_FOR_DYNAMIC + 1);
  REQUIRE(origami::streamk::select_hybrid_mode(above_gate, hardware, config, hardware.N_CU / 2)
          == origami::hybrid_mode_t::dynamic);
}

TEST_CASE("Origami streamk: select_hybrid_mode requires a cotenant to go dynamic",
          "[origami][streamk][hybrid]") {
  // Large grid and low occupancy alone aren't enough: with no cotenant
  // holding any CU away from this kernel, static_ is already optimal.
  auto hardware = make_hardware(950);
  auto config   = make_config(128, 128, 32, 16, 16, 16, false, 1, /*occupancy=*/1);
  auto problem  = make_problem_with_tile_count(128, 128,
      streamk_hybrid_defaults_t::MIN_TILES_FOR_DYNAMIC + 1);

  REQUIRE(origami::streamk::select_hybrid_mode(problem, hardware, config, /*sm_count_target=*/0)
          == origami::hybrid_mode_t::static_);
  REQUIRE(origami::streamk::select_hybrid_mode(
              problem, hardware, config, /*sm_count_target=*/hardware.N_CU / 2)
          == origami::hybrid_mode_t::dynamic);
}

TEST_CASE("Origami streamk: select_hybrid_mode low occupancy goes dynamic unconditionally",
          "[origami][streamk][hybrid]") {
  auto hardware = make_hardware(950);
  auto problem  = make_problem_with_tile_count(128, 128,
      streamk_hybrid_defaults_t::MIN_TILES_FOR_DYNAMIC + 1);

  for (int occupancy = 1;
       occupancy <= streamk_hybrid_defaults_t::MAX_OCCUPANCY_FOR_UNCONDITIONAL_DYNAMIC;
       ++occupancy) {
    DYNAMIC_SECTION("occupancy=" << occupancy) {
      auto config = make_config(128, 128, 32, 16, 16, 16, false, 1, occupancy);
      REQUIRE(origami::streamk::select_hybrid_mode(
                  problem, hardware, config, hardware.N_CU / 2)
              == origami::hybrid_mode_t::dynamic);
    }
  }
}

TEST_CASE("Origami streamk: select_hybrid_mode falls back to tiles_per_cu "
          "once occupancy alone isn't decisive",
          "[origami][streamk][hybrid]") {
  // Occupancy above MAX_OCCUPANCY_FOR_UNCONDITIONAL_DYNAMIC, and occupancy
  // reported as unknown (<= 0), both defer to the tiles_per_cu threshold.
  auto hardware       = make_hardware(950);
  auto available_cus  = hardware.N_CU / 2;
  auto small = make_problem_with_tile_count(128, 128,
      static_cast<size_t>(available_cus * 8.0));
  auto big = make_problem_with_tile_count(128, 128,
      static_cast<size_t>(available_cus * 9.0));

  for (int occupancy : {0, streamk_hybrid_defaults_t::MAX_OCCUPANCY_FOR_UNCONDITIONAL_DYNAMIC + 1}) {
    DYNAMIC_SECTION("occupancy=" << occupancy) {
      auto config = make_config(128, 128, 32, 16, 16, 16, false, 1, occupancy);
      REQUIRE(origami::streamk::select_hybrid_mode(small, hardware, config, available_cus)
              == origami::hybrid_mode_t::static_);
      REQUIRE(origami::streamk::select_hybrid_mode(big, hardware, config, available_cus)
              == origami::hybrid_mode_t::dynamic);
    }
  }
}

TEST_CASE("Origami streamk: select_hybrid_mode non-gfx950 always static",
          "[origami][streamk][hybrid]") {
  // Large grid, cotenant present, low occupancy: would select dynamic on
  // gfx950, but the architecture guard forces static_ elsewhere.
  auto config  = make_config(128, 128, 32, 16, 16, 16, false, 1, /*occupancy=*/1);
  auto problem = make_problem_with_tile_count(128, 128,
      streamk_hybrid_defaults_t::MIN_TILES_FOR_DYNAMIC + 1);

  auto hardware_gfx950 = make_hardware(950);
  REQUIRE(origami::streamk::select_hybrid_mode(
              problem, hardware_gfx950, config, hardware_gfx950.N_CU / 2)
          == origami::hybrid_mode_t::dynamic);

  auto hardware_gfx942 = make_hardware(942);
  REQUIRE(origami::streamk::select_hybrid_mode(
              problem, hardware_gfx942, config, hardware_gfx942.N_CU / 2)
          == origami::hybrid_mode_t::static_);
}

TEST_CASE("Origami streamk: select_hybrid_mode batch multiplies tiles, crossing the gate",
          "[origami][streamk][hybrid]") {
  auto hardware = make_hardware(950);
  auto config   = make_config(128, 128, 32, 16, 16, 16, false, 1, /*occupancy=*/1);
  auto base = make_problem_with_tile_count(128, 128,
      streamk_hybrid_defaults_t::MIN_TILES_FOR_DYNAMIC, /*batch=*/1);
  REQUIRE(origami::streamk::select_hybrid_mode(base, hardware, config, hardware.N_CU / 2)
          == origami::hybrid_mode_t::static_);

  auto base_b4  = base;
  base_b4.batch = 4;
  REQUIRE(origami::streamk::select_hybrid_mode(base_b4, hardware, config, hardware.N_CU / 2)
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

TEST_CASE("Origami streamk: gfx1151 k-split grid uses all physical CUs",
          "[origami][streamk][gfx1151]") {
  auto hardware = origami::hardware_t::get_hardware_for_arch(
      origami::hardware_t::architecture_t::gfx1151,
      /*N_CU=*/40,
      /*lds_capacity=*/64 * 1024,
      /*rf_capacity=*/512 * 1024,
      /*L2_capacity=*/2 * 1024 * 1024,
      /*compute_clock_khz=*/2900000);
  auto config = make_config(128, 128, 32, 16, 16, 16, false, 8, /*occupancy=*/1);
  config.workspace_size            = 512 * 1024 * 1024;
  config.workspace_size_per_elem_c = 4;

  // 10 output tiles with 64 K iterations each can split four ways across 40 CUs.
  auto problem = make_problem(/*m=*/128, /*n=*/1280, /*k=*/2048);
  config.reduction_strategy = origami::reduction_t::parallel;
  REQUIRE(origami::streamk::select_grid_size(
              problem, hardware, config, origami::grid_selection_t::k_split_aware)
          == 40);
}

TEST_CASE("Origami streamk: gfx1151 k-split grid honors a CU budget",
          "[origami][streamk][gfx1151]") {
  auto hardware = origami::hardware_t::get_hardware_for_arch(
      origami::hardware_t::architecture_t::gfx1151,
      /*N_CU=*/40,
      /*lds_capacity=*/64 * 1024,
      /*rf_capacity=*/512 * 1024,
      /*L2_capacity=*/2 * 1024 * 1024,
      /*compute_clock_khz=*/2900000);
  auto config = make_config(128, 128, 32, 16, 16, 16, false, 8, /*occupancy=*/1);
  config.workspace_size            = 512 * 1024 * 1024;
  config.workspace_size_per_elem_c = 4;
  config.reduction_strategy        = origami::reduction_t::parallel;

  auto problem    = make_problem(/*m=*/128, /*n=*/1280, /*k=*/2048);
  problem.num_cus = 20;
  REQUIRE(origami::streamk::select_grid_size(
              problem, hardware, config, origami::grid_selection_t::k_split_aware)
          == 20);
}

TEST_CASE("Origami streamk: gfx1151 reduction selects parallel only for deep low-tile problems",
          "[origami][streamk][gfx1151]") {
  auto hardware = origami::hardware_t::get_hardware_for_arch(
      origami::hardware_t::architecture_t::gfx1151,
      /*N_CU=*/40,
      /*lds_capacity=*/64 * 1024,
      /*rf_capacity=*/512 * 1024,
      /*L2_capacity=*/2 * 1024 * 1024,
      /*compute_clock_khz=*/2900000);
  auto config = make_config(128, 128, 32, 16, 16, 16, false, 8, /*occupancy=*/1);

  auto deep = make_problem(/*m=*/128, /*n=*/1280, /*k=*/2048);
  REQUIRE(origami::streamk::select_reduction(
              deep, hardware, config, origami::grid_selection_t::k_split_aware)
          == origami::reduction_t::parallel);

  auto shallow = deep;
  shallow.size.k = 1024;
  REQUIRE(origami::streamk::select_reduction(
              shallow, hardware, config, origami::grid_selection_t::k_split_aware)
          == origami::reduction_t::tree);
}

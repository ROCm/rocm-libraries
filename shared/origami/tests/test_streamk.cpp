/*******************************************************************************
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 *******************************************************************************/

#include <catch2/catch_test_macros.hpp>
#include <limits>
#include "common.hpp"

namespace {

using origami::streamk_hybrid_defaults_t;

// Builds a problem with exactly `tiles` output tiles for the given macrotile
// (a single row of `tiles` tile columns), so callers get precise control over
// the tile count fed into select_hybrid_mode's gates.
inline origami::problem_t make_problem_with_tile_count(size_t mt_m,
                                                       size_t mt_n,
                                                       size_t tiles,
                                                       size_t batch = 1) {
  return make_problem(/*m=*/mt_m,
                      /*n=*/mt_n * tiles,
                      /*k=*/64,
                      origami::transpose_t::T,
                      origami::transpose_t::N,
                      batch);
}

}  // namespace

TEST_CASE("Origami streamk: select_hybrid_mode tile-count gate is inclusive of the threshold",
          "[origami][streamk][hybrid]") {
  // Even with a cotenant and occupancy low enough to otherwise force dynamic
  // unconditionally, a grid at or below MIN_TILES_FOR_DYNAMIC stays static_.
  auto hardware = make_hardware(950);
  auto config   = make_config(1, 1, 32, 16, 16, 16, false, 1, /*occupancy=*/1);
  auto at_gate =
      make_problem_with_tile_count(1, 1, streamk_hybrid_defaults_t::MIN_TILES_FOR_DYNAMIC);
  REQUIRE(origami::streamk::select_hybrid_mode(at_gate, hardware, config, hardware.N_CU / 2) ==
          origami::hybrid_mode_t::static_);

  auto above_gate =
      make_problem_with_tile_count(1, 1, streamk_hybrid_defaults_t::MIN_TILES_FOR_DYNAMIC + 1);
  REQUIRE(origami::streamk::select_hybrid_mode(above_gate, hardware, config, hardware.N_CU / 2) ==
          origami::hybrid_mode_t::dynamic);
}

TEST_CASE("Origami streamk: select_hybrid_mode requires a cotenant to go dynamic",
          "[origami][streamk][hybrid]") {
  // Large grid and low occupancy alone aren't enough: with no cotenant
  // holding any CU away from this kernel, static_ is already optimal.
  auto hardware = make_hardware(950);
  auto config   = make_config(128, 128, 32, 16, 16, 16, false, 1, /*occupancy=*/1);
  auto problem =
      make_problem_with_tile_count(128, 128, streamk_hybrid_defaults_t::MIN_TILES_FOR_DYNAMIC + 1);

  REQUIRE(origami::streamk::select_hybrid_mode(problem, hardware, config, /*sm_count_target=*/0) ==
          origami::hybrid_mode_t::static_);
  REQUIRE(origami::streamk::select_hybrid_mode(
              problem, hardware, config, /*sm_count_target=*/hardware.N_CU / 2) ==
          origami::hybrid_mode_t::dynamic);
}

TEST_CASE("Origami streamk: select_hybrid_mode low occupancy goes dynamic unconditionally",
          "[origami][streamk][hybrid]") {
  auto hardware = make_hardware(950);
  auto problem =
      make_problem_with_tile_count(128, 128, streamk_hybrid_defaults_t::MIN_TILES_FOR_DYNAMIC + 1);

  for (int occupancy = 1;
       occupancy <= streamk_hybrid_defaults_t::MAX_OCCUPANCY_FOR_UNCONDITIONAL_DYNAMIC;
       ++occupancy) {
    DYNAMIC_SECTION("occupancy=" << occupancy) {
      auto config = make_config(128, 128, 32, 16, 16, 16, false, 1, occupancy);
      REQUIRE(origami::streamk::select_hybrid_mode(problem, hardware, config, hardware.N_CU / 2) ==
              origami::hybrid_mode_t::dynamic);
    }
  }
}

TEST_CASE("Origami streamk: select_hybrid_mode falls back to tiles_per_cu "
          "once occupancy alone isn't decisive",
          "[origami][streamk][hybrid]") {
  // Occupancy above MAX_OCCUPANCY_FOR_UNCONDITIONAL_DYNAMIC, and occupancy
  // reported as unknown (<= 0), both defer to the tiles_per_cu threshold.
  auto hardware      = make_hardware(950);
  auto available_cus = hardware.N_CU / 2;
  auto small = make_problem_with_tile_count(128, 128, static_cast<size_t>(available_cus * 8.0));
  auto big   = make_problem_with_tile_count(128, 128, static_cast<size_t>(available_cus * 9.0));

  for (int occupancy :
       {0, streamk_hybrid_defaults_t::MAX_OCCUPANCY_FOR_UNCONDITIONAL_DYNAMIC + 1}) {
    DYNAMIC_SECTION("occupancy=" << occupancy) {
      auto config = make_config(128, 128, 32, 16, 16, 16, false, 1, occupancy);
      REQUIRE(origami::streamk::select_hybrid_mode(small, hardware, config, available_cus) ==
              origami::hybrid_mode_t::static_);
      REQUIRE(origami::streamk::select_hybrid_mode(big, hardware, config, available_cus) ==
              origami::hybrid_mode_t::dynamic);
    }
  }
}

TEST_CASE("Origami streamk: select_hybrid_mode non-gfx950 always static",
          "[origami][streamk][hybrid]") {
  // Large grid, cotenant present, low occupancy: would select dynamic on
  // gfx950, but the architecture guard forces static_ elsewhere.
  auto config = make_config(128, 128, 32, 16, 16, 16, false, 1, /*occupancy=*/1);
  auto problem =
      make_problem_with_tile_count(128, 128, streamk_hybrid_defaults_t::MIN_TILES_FOR_DYNAMIC + 1);

  auto hardware_gfx950 = make_hardware(950);
  REQUIRE(origami::streamk::select_hybrid_mode(
              problem, hardware_gfx950, config, hardware_gfx950.N_CU / 2) ==
          origami::hybrid_mode_t::dynamic);

  auto hardware_gfx942 = make_hardware(942);
  REQUIRE(origami::streamk::select_hybrid_mode(
              problem, hardware_gfx942, config, hardware_gfx942.N_CU / 2) ==
          origami::hybrid_mode_t::static_);
}

TEST_CASE("Origami streamk: select_hybrid_mode batch multiplies tiles, crossing the gate",
          "[origami][streamk][hybrid]") {
  auto hardware = make_hardware(950);
  auto config   = make_config(128, 128, 32, 16, 16, 16, false, 1, /*occupancy=*/1);
  auto base     = make_problem_with_tile_count(
      128, 128, streamk_hybrid_defaults_t::MIN_TILES_FOR_DYNAMIC, /*batch=*/1);
  REQUIRE(origami::streamk::select_hybrid_mode(base, hardware, config, hardware.N_CU / 2) ==
          origami::hybrid_mode_t::static_);

  auto base_b4  = base;
  base_b4.batch = 4;
  REQUIRE(origami::streamk::select_hybrid_mode(base_b4, hardware, config, hardware.N_CU / 2) ==
          origami::hybrid_mode_t::dynamic);
}

TEST_CASE("Origami streamk: select_hybrid_mode sm_count_target=0 uses N_CU",
          "[origami][streamk][hybrid]") {
  auto hardware = make_hardware(950);
  auto config   = make_config(128, 128, 32);
  auto problem  = make_problem(4096, 4096, 64);
  auto a        = origami::streamk::select_hybrid_mode(problem, hardware, config, 0);
  auto b        = origami::streamk::select_hybrid_mode(problem, hardware, config, hardware.N_CU);
  REQUIRE(a == b);
}

TEST_CASE("Origami streamk: stream_k=0 uses one WG per output tile", "[origami][streamk][flag]") {
  auto hardware = make_hardware(950);
  auto problem  = make_problem(/*m=*/512, /*n=*/512, /*k=*/8192);
  auto config   = make_config(256,
                            256,
                            64,
                            32,
                            32,
                            8,
                            false,
                            1,
                            /*occupancy=*/1,
                            /*non_temporal_a=*/0,
                            /*non_temporal_b=*/0,
                            /*stream_k=*/0);

  const size_t tiles = origami::streamk::compute_number_of_output_tiles(
      config.mt.m, config.mt.n, problem.size.m, problem.size.n, problem.batch);

  auto [reduction, num_wgs, num_active_cus, num_timesteps, split_factor] =
      origami::gemm::compute_launch_parameters(problem, hardware, config, config.grid_selection);

  REQUIRE(num_wgs == tiles);
  REQUIRE(split_factor == 1);
  REQUIRE(reduction == origami::reduction_t::none);
  REQUIRE(num_active_cus > 0);
  REQUIRE(num_timesteps >= 1);
}

TEST_CASE("Origami streamk: stream_k=5 K-splits when tiles << CUs", "[origami][streamk][flag]") {
  auto hardware = make_hardware(950);
  auto problem  = make_problem(512, 512, 8192);
  auto config   = make_config(256,
                            256,
                            64,
                            32,
                            32,
                            8,
                            false,
                            1,
                            /*occupancy=*/1,
                            /*non_temporal_a=*/0,
                            /*non_temporal_b=*/0,
                            /*stream_k=*/5);

  const size_t tiles = origami::streamk::compute_number_of_output_tiles(
      config.mt.m, config.mt.n, problem.size.m, problem.size.n, problem.batch);

  auto [reduction, num_wgs, num_active_cus, num_timesteps, split_factor] =
      origami::gemm::compute_launch_parameters(problem, hardware, config, config.grid_selection);

  REQUIRE(num_wgs > tiles);
  REQUIRE(split_factor > 1);
  REQUIRE(split_factor == origami::math::safe_ceil_div(num_wgs, tiles));
  REQUIRE(reduction != origami::reduction_t::none);
  REQUIRE(num_active_cus > 0);
  REQUIRE(num_timesteps >= 1);
}

TEST_CASE("Origami streamk: stream_k=0 vs 5 launch divergence on underloaded grid",
          "[origami][streamk][flag]") {
  auto hardware = make_hardware(950);
  auto problem  = make_problem(512, 512, 8192);

  auto config_dp            = make_config(256,
                               256,
                               64,
                               32,
                               32,
                               8,
                               false,
                               1,
                               /*occupancy=*/1,
                               /*non_temporal_a=*/0,
                               /*non_temporal_b=*/0,
                               /*stream_k=*/0);
  auto config_sk5           = config_dp;
  config_sk5.stream_k       = 5;
  config_sk5.grid_selection = origami::grid_selection_t::k_split_aware;

  const auto launch = [&](const origami::config_t& c) {
    return origami::gemm::compute_launch_parameters(problem, hardware, c, c.grid_selection);
  };

  const auto result0 = launch(config_dp);
  const auto result5 = launch(config_sk5);

  REQUIRE(std::get<1>(result0) < std::get<1>(result5));
  REQUIRE(std::get<4>(result0) == 1);
  REQUIRE(std::get<4>(result5) > 1);
  REQUIRE(std::get<0>(result0) == origami::reduction_t::none);
  REQUIRE(std::get<0>(result5) != origami::reduction_t::none);
}

TEST_CASE("Origami streamk: context_t reflects stream_k flag", "[origami][streamk][flag]") {
  auto hardware = make_hardware(950);
  auto problem  = make_problem(512, 512, 8192);
  auto config   = make_config(256, 256, 64, 32, 32, 8, false, 1);

  config.stream_k = 0;
  origami::gemm::context_t ctx0(problem, hardware, config);
  REQUIRE(ctx0.splitting_factor == 1);
  REQUIRE(ctx0.num_wgs == ctx0.num_output_tiles);

  config.stream_k       = 5;
  config.grid_selection = origami::grid_selection_t::k_split_aware;
  origami::gemm::context_t ctx5(problem, hardware, config);
  REQUIRE(ctx5.splitting_factor > 1);
  REQUIRE(ctx5.num_wgs > ctx0.num_wgs);
}

TEST_CASE("Origami streamk: compute_total_latency differs for stream_k=0 vs 5",
          "[origami][streamk][flag]") {
  auto hardware = make_hardware(950);
  auto problem  = make_problem(512, 512, 8192);
  auto config   = make_config(256, 256, 64, 32, 32, 8, false, 1);

  config.stream_k       = 0;
  const double latency0 = origami::gemm::compute_total_latency(problem, hardware, config);

  config.stream_k       = 5;
  config.grid_selection = origami::grid_selection_t::k_split_aware;
  const double latency5 = origami::gemm::compute_total_latency(problem, hardware, config);

  REQUIRE(latency0 != latency5);
  REQUIRE(latency0 < std::numeric_limits<double>::max());
  REQUIRE(latency5 < std::numeric_limits<double>::max());
}

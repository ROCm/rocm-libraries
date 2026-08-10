/*******************************************************************************
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 *******************************************************************************/

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "common.hpp"

#include <cmath>
#include <iterator>
#include <limits>
#include <utility>

namespace {

struct tiles_gate : origami::streamk::thresholds<tiles_gate> {
  static constexpr origami::streamk::threshold_rule decision_tree[] = {
      {origami::streamk::threshold_metrics::tiles, 480.0, origami::streamk::comparison_type::less_then_or_equal,
       origami::hybrid_mode_t::static_},
      {origami::streamk::threshold_metrics::occupancy, 0.0, origami::streamk::comparison_type::greater_then,
       origami::hybrid_mode_t::dynamic},
  };
};

struct occupancy_gate : origami::streamk::thresholds<occupancy_gate> {
  static constexpr origami::streamk::threshold_rule decision_tree[] = {
      {origami::streamk::threshold_metrics::occupancy,
       origami::streamk::gfx950_values::occupancy_threshold,
       origami::streamk::comparison_type::less_then_or_equal, origami::hybrid_mode_t::dynamic},
  };
};

origami::hybrid_mode_t gfx950_expected(const origami::problem_t&  problem,
                                       const origami::hardware_t& hardware,
                                       const origami::config_t&   config,
                                       size_t                     smt) {
  using origami::streamk::gfx950_values;
  using origami::streamk::threshold_metrics;
  const double tiles = gfx950_values::feature_value(threshold_metrics::tiles, problem, hardware, config, smt);
  if (tiles <= 480.0) {
    return origami::hybrid_mode_t::static_;
  }
  const double ge = gfx950_values::feature_value(threshold_metrics::grid_efficiency, problem, hardware, config, smt);
  if (ge <= 0.23) {
    return origami::hybrid_mode_t::dynamic;
  }
  const double occ = gfx950_values::feature_value(threshold_metrics::occupancy, problem, hardware, config, smt);
  if (occ <= 2.5) {
    return origami::hybrid_mode_t::dynamic;
  }
  return origami::hybrid_mode_t::static_;
}

}  // namespace

TEST_CASE("Origami streamk: feature_value derives consistent values",
          "[origami][streamk][hybrid]") {
  using origami::streamk::gfx950_values;
  using origami::streamk::threshold_metrics;

  auto         hardware = make_hardware(950);
  auto         config   = make_config(128, 128, 32, 16, 16, 16, false, 1, 3);
  auto         problem  = make_problem(4096, 2048, 512);
  const size_t smt      = hardware.N_CU / 2;

  REQUIRE(gfx950_values::feature_value(threshold_metrics::occupancy, problem, hardware, config, smt) == 3);

  const double tiles = gfx950_values::feature_value(threshold_metrics::tiles, problem, hardware, config, smt);
  REQUIRE(tiles == 512);

  REQUIRE(gfx950_values::feature_value(threshold_metrics::tiles_per_cu, problem, hardware, config, smt)
          == Catch::Approx(tiles / static_cast<double>(smt)));

  const double grid_waves = gfx950_values::feature_value(threshold_metrics::grid_waves, problem, hardware, config, smt);
  if (grid_waves > 0.0) {
    const double grid_efficiency =
        gfx950_values::feature_value(threshold_metrics::grid_efficiency, problem, hardware, config, smt);
    REQUIRE(grid_efficiency > 0.0);
    REQUIRE(grid_efficiency <= 1.0);
  }
}

TEST_CASE("Origami streamk: unknown occupancy maps to NaN",
          "[origami][streamk][hybrid]") {
  using origami::streamk::gfx950_values;
  using origami::streamk::threshold_metrics;

  auto hardware = make_hardware(950);
  auto config   = make_config(128, 128, 32, 16, 16, 16, false, 1, -1);
  auto problem  = make_problem(4096, 2048, 512);

  const double occ = gfx950_values::feature_value(threshold_metrics::occupancy, problem, hardware, config, hardware.N_CU);
  REQUIRE(std::isnan(occ));
}

TEST_CASE("Origami streamk: select_hybrid_mode returns the first firing rule's mode",
          "[origami][streamk][hybrid]") {
  using origami::streamk::threshold_metrics;
  auto            hardware = make_hardware(950);
  const size_t    smt      = hardware.N_CU;
  constexpr auto& tree     = tiles_gate::decision_tree;

  auto problem_with_tiles = [](size_t t) {
    return make_problem(128, static_cast<size_t>(128 * t), 64);
  };

  const size_t tiles_thr  = static_cast<size_t>(tree[0].threshold);
  auto         build_case = [&](size_t target) {
    size_t tiles_count = tiles_thr + 1;
    int    occupancy   = 0;
    switch (tree[target].feature) {
      case threshold_metrics::tiles:
        tiles_count = static_cast<size_t>(tree[target].threshold);
        break;
      case threshold_metrics::occupancy:
        occupancy = static_cast<int>(tree[target].threshold) + 1;
        break;
      default:
        break;
    }
    return std::make_pair(problem_with_tiles(tiles_count),
                          make_config(128, 128, 32, 16, 16, 16, false, 1, occupancy));
  };

  for (size_t target = 0; target < std::size(tree); ++target) {
    auto [problem, config] = build_case(target);
    REQUIRE(tiles_gate::select_hybrid_mode(problem, hardware, config, smt) == tree[target].mode);
  }

  SECTION("no rule fires -> static default") {
    auto config  = make_config(128, 128, 32, 16, 16, 16, false, 1, 0);
    auto problem = problem_with_tiles(tiles_thr + 1);
    REQUIRE(tiles_gate::select_hybrid_mode(problem, hardware, config, smt)
            == origami::hybrid_mode_t::static_);
  }
}

TEST_CASE("Origami streamk: occupancy threshold selects dynamic at or below 2.5",
          "[origami][streamk][hybrid]") {
  using origami::streamk::gfx950_values;
  STATIC_REQUIRE(gfx950_values::occupancy_threshold == 2.5);

  auto         hardware = make_hardware(950);
  auto         problem  = make_problem(4096, 4096, 64);
  const size_t smt      = hardware.N_CU / 2;

  REQUIRE(occupancy_gate::select_hybrid_mode(problem, hardware, make_config(128, 128, 32, 16, 16, 16, false, 1, 2), smt)
          == origami::hybrid_mode_t::dynamic);
  REQUIRE(occupancy_gate::select_hybrid_mode(problem, hardware, make_config(128, 128, 32, 16, 16, 16, false, 1, 3), smt)
          == origami::hybrid_mode_t::static_);
  REQUIRE(occupancy_gate::select_hybrid_mode(problem, hardware, make_config(128, 128, 32, 16, 16, 16, false, 1, -1), smt)
          == origami::hybrid_mode_t::static_);
}

TEST_CASE("Origami streamk: gfx950 tree gates on the tile minimum",
          "[origami][streamk][hybrid]") {
  auto hardware = make_hardware(950);
  auto config   = make_config(128, 128, 32, 16, 16, 16, false, 1, 2);
  auto problem  = make_problem(256, 256, 64);

  REQUIRE(origami::streamk::gfx950_values::select_hybrid_mode(problem, hardware, config, hardware.N_CU)
          == origami::hybrid_mode_t::static_);
}

TEST_CASE("Origami streamk: gfx950 select_hybrid_mode is self-consistent with feature_value",
          "[origami][streamk][hybrid]") {
  auto         hardware = make_hardware(950);
  auto         config   = make_config(128, 128, 32, 16, 16, 16, false, 1, 2);
  const size_t smt      = hardware.N_CU / 2;

  for (auto problem : {make_problem(4096, 2048, 512), make_problem(8192, 8192, 8192),
                       make_problem(1024, 1024, 64)}) {
    REQUIRE(origami::streamk::gfx950_values::select_hybrid_mode(problem, hardware, config, smt)
            == gfx950_expected(problem, hardware, config, smt));
  }
}

TEST_CASE("Origami streamk: gfx942 tree splits on grid_waves",
          "[origami][streamk][hybrid]") {
  using origami::streamk::gfx942_values;
  using origami::streamk::threshold_metrics;

  auto         hardware = make_hardware(942);
  auto         config   = make_config(128, 128, 32, 16, 16, 16, false, 1, 2);
  const size_t smt      = hardware.N_CU / 2;

  for (auto problem : {make_problem(4096, 2048, 512), make_problem(8192, 8192, 8192),
                       make_problem(1024, 512, 64)}) {
    const double gw = gfx942_values::feature_value(threshold_metrics::grid_waves, problem, hardware, config, smt);
    const auto   expected =
        (gw > 1.17) ? origami::hybrid_mode_t::dynamic : origami::hybrid_mode_t::static_;
    REQUIRE(gfx942_values::select_hybrid_mode(problem, hardware, config, smt) == expected);
  }
}

TEST_CASE("Origami streamk: a cotenant is required to go dynamic",
          "[origami][streamk][hybrid]") {
  struct arch_case {
    const char*        name;
    int                arch;
    origami::config_t  config;
    origami::problem_t problem;
  };

  const arch_case cases[] = {
      {"gfx942", 942, make_config(96, 320, 32, 16, 16, 16, false, 1, 2),
       make_problem(192, 131072, 256)},
      {"gfx950", 950, make_config(128, 128, 32, 16, 16, 16, false, 1, 2),
       make_problem(4096, 2048, 64)},
  };

  for (auto const& c : cases) {
    CAPTURE(c.name);
    auto hardware = make_hardware(c.arch);
    REQUIRE(origami::streamk::select_hybrid_mode(c.problem, hardware, c.config, hardware.N_CU / 2)
            == origami::hybrid_mode_t::dynamic);
    REQUIRE(origami::streamk::select_hybrid_mode(c.problem, hardware, c.config, 0)
            == origami::hybrid_mode_t::static_);
    REQUIRE(origami::streamk::select_hybrid_mode(c.problem, hardware, c.config, hardware.N_CU)
            == origami::hybrid_mode_t::static_);
  }
}

TEST_CASE("Origami streamk: select_hybrid_mode batch multiplies tiles, crossing the gate",
          "[origami][streamk][hybrid]") {
  auto hardware = make_hardware(950);
  auto config   = make_config(128, 128, 32, 16, 16, 16, false, 1, 2);
  auto problem  = make_problem(4096, 512, 64);

  problem.batch = 1;
  REQUIRE(origami::streamk::select_hybrid_mode(problem, hardware, config, hardware.N_CU / 2)
          == origami::hybrid_mode_t::static_);

  problem.batch = 4;
  REQUIRE(origami::streamk::select_hybrid_mode(problem, hardware, config, hardware.N_CU / 2)
          == origami::hybrid_mode_t::dynamic);
}

TEST_CASE("Origami streamk: select_hybrid_mode untuned architecture stays static",
          "[origami][streamk][hybrid]") {
  auto hardware = make_hardware(1250);
  auto config   = make_config(128, 128, 32, 16, 16, 16, false, 1, 1);
  auto problem  = make_problem(8192, 8192, 8192);

  REQUIRE(origami::streamk::select_hybrid_mode(problem, hardware, config, hardware.N_CU / 2)
          == origami::hybrid_mode_t::static_);
}

TEST_CASE("Origami streamk: select_hybrid_mode sm_count_target=0 uses N_CU",
          "[origami][streamk][hybrid]") {
  auto config  = make_config(128, 128, 32);
  auto problem = make_problem(4096, 4096, 64);
  for (int arch : {942, 950}) {
    CAPTURE(arch);
    auto hardware = make_hardware(arch);
    auto a        = origami::streamk::select_hybrid_mode(problem, hardware, config, 0);
    auto b        = origami::streamk::select_hybrid_mode(problem, hardware, config, hardware.N_CU);
    REQUIRE(a == b);
    REQUIRE(a == origami::hybrid_mode_t::static_);
  }
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

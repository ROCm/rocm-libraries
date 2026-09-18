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
#include <cstddef>
#include <iterator>
#include <limits>
#include <utility>

namespace {

// ---- Compile-time decision-tree validator -----------------------------------
// Walks if_lte / if_gt links from the root and rejects a table whose children
// are out of range, self-referential, part of a cycle, or leave a node
// unreachable from the root. Used in static_asserts so a bad tree can't compile.
template <class Node>
constexpr bool tree_dfs(const Node* tree, int n, int node, int leaf_lo, int leaf_hi,
                        bool* visited, bool* on_stack) {
  if (node == leaf_lo || node == leaf_hi) return true;  // reached a leaf verdict
  if (node < 0 || node >= n) return false;              // dangling child index
  if (on_stack[node]) return false;                     // cycle on the current path
  if (visited[node]) return true;                       // already proven reachable & acyclic
  visited[node]  = true;
  on_stack[node] = true;
  const bool ok = tree_dfs(tree, n, tree[node].if_lte, leaf_lo, leaf_hi, visited, on_stack)
               && tree_dfs(tree, n, tree[node].if_gt,  leaf_lo, leaf_hi, visited, on_stack);
  on_stack[node] = false;
  return ok;
}

template <class Node, std::size_t N>
constexpr bool decision_tree_is_valid(const Node (&tree)[N], int leaf_lo, int leaf_hi) {
  const int n        = static_cast<int>(N);
  auto      child_ok = [&](int c) { return c == leaf_lo || c == leaf_hi || (c >= 0 && c < n); };

  for (int i = 0; i < n; ++i) {
    if (!child_ok(tree[i].if_lte) || !child_ok(tree[i].if_gt)) return false;
    if (tree[i].if_lte == i || tree[i].if_gt == i) return false;  // self-loop
  }

  bool visited[N]  = {};
  bool on_stack[N] = {};
  if (!tree_dfs(tree, n, 0, leaf_lo, leaf_hi, visited, on_stack)) return false;  // acyclic from root
  for (std::size_t i = 0; i < N; ++i) {
    if (!visited[i]) return false;  // node not reachable from the root
  }
  return true;
}

// A small synthetic branching tree used to exercise decision_node traversal:
//   node 0: tiles <= 480 ? node 1 : dynamic
//   node 1: occupancy <= 2.5 ? static : dynamic
// This hits both leaf sentinels and an internal (index) child. threshold_metrics,
// decision_node and LEAF_* are inherited from the thresholds<Arch> base.
struct branch_gate : origami::streamk::thresholds<branch_gate> {
  static constexpr decision_node decision_tree[] = {
      {threshold_metrics::tiles, 480.0, 1, dynamic_result},
      {threshold_metrics::occupancy, 2.5, static_result, dynamic_result},
  };
};

// The shipped trees must be structurally sound at compile time.
static_assert(decision_tree_is_valid(origami::streamk::gfx942_values::decision_tree,
                                     origami::streamk::gfx942_values::static_result,
                                     origami::streamk::gfx942_values::dynamic_result),
              "gfx942 decision tree is malformed");
static_assert(decision_tree_is_valid(origami::streamk::gfx950_values::decision_tree,
                                     origami::streamk::gfx950_values::static_result,
                                     origami::streamk::gfx950_values::dynamic_result),
              "gfx950 decision tree is malformed");
static_assert(decision_tree_is_valid(branch_gate::decision_tree, branch_gate::static_result,
                                     branch_gate::dynamic_result),
              "branch_gate decision tree is malformed");

// Reference mirror of gfx950's deployed 10-leaf tree, evaluated through the same
// feature_value() the library uses, so the test tracks the header by construction.
origami::hybrid_mode_t gfx950_expected(const origami::problem_t&  problem,
                                       const origami::hardware_t& hardware,
                                       const origami::config_t&   config,
                                       size_t                     smt) {
  using origami::streamk::gfx950_values;
  using threshold_metrics = gfx950_values::threshold_metrics;
  auto fv = [&](threshold_metrics m) {
    return gfx950_values::feature_value(m, problem, hardware, config, smt);
  };

  if (fv(threshold_metrics::grid_efficiency) <= 0.23) {
    if (fv(threshold_metrics::min_mn) <= 1088.0) {
      return fv(threshold_metrics::m_dim) <= 3277.0 ? origami::hybrid_mode_t::static_
                                                     : origami::hybrid_mode_t::dynamic;
    }
    return fv(threshold_metrics::tiles) <= 34.0 ? origami::hybrid_mode_t::dynamic
                                                : origami::hybrid_mode_t::static_;
  }
  if (fv(threshold_metrics::tiles_per_cu) <= 0.29) {
    if (fv(threshold_metrics::static_skgrid) <= 68.0) {
      return origami::hybrid_mode_t::dynamic;
    }
    return fv(threshold_metrics::iters_per_tile) <= 458.0 ? origami::hybrid_mode_t::static_
                                                          : origami::hybrid_mode_t::dynamic;
  }
  if (fv(threshold_metrics::active_cus) <= 240.0) {
    return origami::hybrid_mode_t::dynamic;
  }
  return fv(threshold_metrics::occupancy) <= 2.5 ? origami::hybrid_mode_t::dynamic
                                                 : origami::hybrid_mode_t::static_;
}

}  // namespace

TEST_CASE("Origami streamk: decision_tree validator rejects malformed tables",
          "[origami][streamk][hybrid]") {
  using node                  = branch_gate::decision_node;
  using tm                    = branch_gate::threshold_metrics;
  constexpr int S             = branch_gate::static_result;
  constexpr int D             = branch_gate::dynamic_result;

  constexpr node good[]        = {{tm::tiles, 480.0, 1, D}, {tm::occupancy, 2.5, S, D}};
  constexpr node out_of_range[] = {{tm::tiles, 1.0, 5, D}};                        // child 5 >= N
  constexpr node self_loop[]    = {{tm::tiles, 1.0, 0, D}};                        // points at itself
  constexpr node cycle[]        = {{tm::tiles, 1.0, 1, S}, {tm::tiles, 1.0, 0, S}};  // 0 -> 1 -> 0
  constexpr node unreachable[]  = {{tm::tiles, 1.0, S, S}, {tm::tiles, 1.0, S, D}};  // node 1 orphaned

  static_assert(decision_tree_is_valid(good, S, D), "well-formed tree should validate");
  static_assert(!decision_tree_is_valid(out_of_range, S, D), "out-of-range child must fail");
  static_assert(!decision_tree_is_valid(self_loop, S, D), "self-loop must fail");
  static_assert(!decision_tree_is_valid(cycle, S, D), "cycle must fail");
  static_assert(!decision_tree_is_valid(unreachable, S, D), "unreachable node must fail");

  REQUIRE(decision_tree_is_valid(good, S, D));
  REQUIRE_FALSE(decision_tree_is_valid(cycle, S, D));
}

TEST_CASE("Origami streamk: feature_value derives consistent values",
          "[origami][streamk][hybrid]") {
  using origami::streamk::gfx950_values;
  using threshold_metrics = gfx950_values::threshold_metrics;

  auto         hardware = make_hardware(950);
  auto         config   = make_config(128, 128, 32, 16, 16, 16, false, 1, 3);
  auto         problem  = make_problem(4096, 2048, 512);
  const size_t smt      = hardware.N_CU / 2;

  auto fv = [&](threshold_metrics m) { return gfx950_values::feature_value(m, problem, hardware, config, smt); };

  REQUIRE(fv(threshold_metrics::occupancy) == 3);

  const double tiles = fv(threshold_metrics::tiles);
  REQUIRE(tiles == 512);

  REQUIRE(fv(threshold_metrics::tiles_per_cu) == Catch::Approx(tiles / static_cast<double>(smt)));

  // New tree features.
  REQUIRE(fv(threshold_metrics::m_dim) == 4096);
  REQUIRE(fv(threshold_metrics::min_mn) == 2048);         // min(4096, 2048)
  REQUIRE(fv(threshold_metrics::active_cus) == static_cast<double>(smt));
  REQUIRE(fv(threshold_metrics::iters_per_tile) == 16);   // ceil(512 / 32)
  REQUIRE(fv(threshold_metrics::static_skgrid) > 0.0);

  const double grid_waves = fv(threshold_metrics::grid_waves);
  if (grid_waves > 0.0) {
    const double grid_efficiency = fv(threshold_metrics::grid_efficiency);
    REQUIRE(grid_efficiency > 0.0);
    REQUIRE(grid_efficiency <= 1.0);
  }
}

TEST_CASE("Origami streamk: iters_per_tile is NaN when mt.k is zero",
          "[origami][streamk][hybrid]") {
  using origami::streamk::gfx950_values;
  using threshold_metrics = gfx950_values::threshold_metrics;

  auto hardware = make_hardware(950);
  auto config   = make_config(128, 128, 0, 16, 16, 16, false, 1, 2);
  auto problem  = make_problem(4096, 2048, 512);

  REQUIRE(std::isnan(
      gfx950_values::feature_value(threshold_metrics::iters_per_tile, problem, hardware, config, hardware.N_CU / 2)));
}

TEST_CASE("Origami streamk: unknown occupancy maps to NaN",
          "[origami][streamk][hybrid]") {
  using origami::streamk::gfx950_values;
  using threshold_metrics = gfx950_values::threshold_metrics;

  auto hardware = make_hardware(950);
  auto config   = make_config(128, 128, 32, 16, 16, 16, false, 1, -1);
  auto problem  = make_problem(4096, 2048, 512);

  const double occ = gfx950_values::feature_value(threshold_metrics::occupancy, problem, hardware, config, hardware.N_CU);
  REQUIRE(std::isnan(occ));
}

TEST_CASE("Origami streamk: decision_node traversal follows true/false branches to leaves",
          "[origami][streamk][hybrid]") {
  auto         hardware = make_hardware(950);
  const size_t smt      = hardware.N_CU;

  // tiles == n-tiles for a 128x128 macrotile with m == 128 (one m-tile) and batch 1.
  auto problem_with_tiles = [](size_t t) { return make_problem(128, static_cast<size_t>(128 * t), 64); };

  SECTION("root false-child leaf: tiles > 480 -> dynamic") {
    auto problem = problem_with_tiles(500);
    auto config  = make_config(128, 128, 32, 16, 16, 16, false, 1, 0);
    REQUIRE(branch_gate::select_hybrid_mode(problem, hardware, config, smt) == origami::hybrid_mode_t::dynamic);
  }
  SECTION("root true-child then leaf: tiles <= 480 && occupancy <= 2.5 -> static") {
    auto problem = problem_with_tiles(100);
    auto config  = make_config(128, 128, 32, 16, 16, 16, false, 1, 2);
    REQUIRE(branch_gate::select_hybrid_mode(problem, hardware, config, smt) == origami::hybrid_mode_t::static_);
  }
  SECTION("root true-child then other leaf: tiles <= 480 && occupancy > 2.5 -> dynamic") {
    auto problem = problem_with_tiles(100);
    auto config  = make_config(128, 128, 32, 16, 16, 16, false, 1, 3);
    REQUIRE(branch_gate::select_hybrid_mode(problem, hardware, config, smt) == origami::hybrid_mode_t::dynamic);
  }
}

TEST_CASE("Origami streamk: gfx950 select_hybrid_mode matches the reference tree",
          "[origami][streamk][hybrid]") {
  auto         hardware = make_hardware(950);
  auto         config   = make_config(128, 128, 32, 16, 16, 16, false, 1, 2);
  const size_t smt      = hardware.N_CU / 2;

  for (auto problem : {make_problem(256, 256, 64), make_problem(4096, 2048, 512),
                       make_problem(8192, 8192, 8192), make_problem(1024, 1024, 64),
                       make_problem(512, 512, 4096), make_problem(128, 131072, 256),
                       make_problem(64, 64, 8192), make_problem(4096, 512, 64),
                       make_problem(2048, 900, 1024), make_problem(300, 300, 128)}) {
    REQUIRE(origami::streamk::gfx950_values::select_hybrid_mode(problem, hardware, config, smt)
            == gfx950_expected(problem, hardware, config, smt));
  }
}

TEST_CASE("Origami streamk: gfx942 tree splits on grid_waves",
          "[origami][streamk][hybrid]") {
  using origami::streamk::gfx942_values;
  using threshold_metrics = gfx942_values::threshold_metrics;

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

TEST_CASE("Origami streamk: outer gate forces static without a genuine CU cap",
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
    // No genuine CU cap (0 or >= N_CU) -> outer gate forces static regardless of the tree.
    REQUIRE(origami::streamk::select_hybrid_mode(c.problem, hardware, c.config, 0)
            == origami::hybrid_mode_t::static_);
    REQUIRE(origami::streamk::select_hybrid_mode(c.problem, hardware, c.config, hardware.N_CU)
            == origami::hybrid_mode_t::static_);
    // With a genuine cap the top-level dispatch defers to the per-arch tree.
    const auto capped = origami::streamk::select_hybrid_mode(c.problem, hardware, c.config, hardware.N_CU / 2);
    const auto arch_verdict = (c.arch == 950)
        ? origami::streamk::gfx950_values::select_hybrid_mode(c.problem, hardware, c.config, hardware.N_CU / 2)
        : origami::streamk::gfx942_values::select_hybrid_mode(c.problem, hardware, c.config, hardware.N_CU / 2);
    REQUIRE(capped == arch_verdict);
  }
}

TEST_CASE("Origami streamk: select_hybrid_mode batch feeds tiles through the tree",
          "[origami][streamk][hybrid]") {
  auto         hardware = make_hardware(950);
  auto         config   = make_config(128, 128, 32, 16, 16, 16, false, 1, 2);
  const size_t smt      = hardware.N_CU / 2;
  auto         problem  = make_problem(4096, 512, 64);

  problem.batch = 1;
  REQUIRE(origami::streamk::gfx950_values::select_hybrid_mode(problem, hardware, config, smt)
          == gfx950_expected(problem, hardware, config, smt));

  problem.batch = 4;
  REQUIRE(origami::streamk::gfx950_values::select_hybrid_mode(problem, hardware, config, smt)
          == gfx950_expected(problem, hardware, config, smt));
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

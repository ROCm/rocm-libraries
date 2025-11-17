// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "common.hpp"

using Catch::Approx;

// Test functions for origami.hpp/cpp

TEST_CASE("Origami: compute_perf_gflops", "[origami]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - faster clock yields higher GFLOPS") {
      // TODO: Add support for make_hardware using hipDeviceProperties
      auto hardware_slow = make_hardware(gpu_arch, 304, 65536, 8, 1.0, 1.0, 1.0, 4000000, 1.4);
      auto hardware_fast = make_hardware(gpu_arch, 304, 65536, 8, 1.0, 1.0, 1.0, 4000000, 1.8);
      auto problem =
          make_problem(4096, 4096, 1024, origami::transpose_t::T, origami::transpose_t::N, 2);
      auto config = make_config(128, 128, 64, 32, 32, 8, 1);

      auto config_slow = config;
      auto config_fast = config;

      auto latency_config_slow =
          origami::compute_total_latency(problem, hardware_slow, config_slow);
      auto flops_slow = origami::compute_perf_gflops(hardware_slow, problem, latency_config_slow);

      auto latency_config_fast =
          origami::compute_total_latency(problem, hardware_fast, config_fast);
      auto flops_fast = origami::compute_perf_gflops(hardware_fast, problem, latency_config_fast);

      REQUIRE(flops_fast > flops_slow);
    }
  }
}

TEST_CASE("Origami: hardware_arch_enum", "[origami]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " architecture enum") {
      std::string arch_str = "gfx" + std::to_string(gpu_arch);
      auto arch_enum       = origami::hardware_t::arch_name_to_enum(arch_str);

      if (gpu_arch == 942) {
        REQUIRE(arch_enum == origami::hardware_t::architecture_t::gfx942);
      } else if (gpu_arch == 950) {
        REQUIRE(arch_enum == origami::hardware_t::architecture_t::gfx950);
      }
    }
  }
}

TEST_CASE("Origami: best_grid_size", "[origami]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - grid size selection") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(1024, 1024, 4096);
      auto config   = make_config(256, 256, 64, 32, 32, 8, 1);

      auto grid_size = origami::streamk::select_grid_size(
          problem, hardware, config, origami::grid_selection_t::k_split_aware);

      REQUIRE(grid_size >= 16);
    }
  }
}

TEST_CASE("Origami: best_macro_tile_size", "[origami]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - rank configs by latency") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(1024, 1024, 4096);

      std::vector<origami::config_t> configs;
      std::vector<
          std::tuple<size_t, size_t, size_t, size_t, size_t, size_t, size_t, int, size_t, size_t>>
          mt_list = {{256, 256, 32, 32, 32, 8, 1, 6, 0, 0},
                     {128, 128, 64, 32, 32, 8, 1, 6, 0, 0},
                     {64, 64, 64, 32, 32, 8, 1, 6, 0, 0}};

      for (const auto& mt_tuple : mt_list) {
        origami::config_t config;
        config.mt.m              = std::get<0>(mt_tuple);
        config.mt.n              = std::get<1>(mt_tuple);
        config.mt.k              = std::get<2>(mt_tuple);
        config.mi.m              = std::get<3>(mt_tuple);
        config.mi.n              = std::get<4>(mt_tuple);
        config.mi.k              = std::get<5>(mt_tuple);
        config.occupancy         = std::get<6>(mt_tuple);
        config.workgroup_mapping = std::get<7>(mt_tuple);
        config.cache_hints_a     = std::get<8>(mt_tuple);
        config.cache_hints_b     = std::get<9>(mt_tuple);
        configs.push_back(config);
      }

      auto results = origami::rank_configs(problem, hardware, configs);

      REQUIRE(results.size() == mt_list.size());
      // Results should be ranked, so latencies should be in ascending order (best first)
      for (size_t i = 0; i < results.size() - 1; i++) {
        REQUIRE(results[i].latency < results[i + 1].latency);
      }
    }
  }
}

TEST_CASE("Origami: select_workgroup_mapping", "[origami]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - workgroup mapping selection") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(4096, 4096, 8192);

      origami::dim3_t mt{256, 256, 32};
      origami::dim3_t mi{32, 32, 8};
      std::vector<size_t> wgm_list = {1, 2, 4, 6, 8, 12};

      origami::dim3_t mt_large = mt;
      auto best_wgm_large_tile =
          origami::select_workgroup_mapping(problem, hardware, mt_large, mi, wgm_list);

      origami::dim3_t mt_small{mt.m / 2, mt.n / 2, mt.k * 2};
      auto best_wgm_small_tile =
          origami::select_workgroup_mapping(problem, hardware, mt_small, mi, wgm_list);

      // Different problem size for nonsquare test
      origami::problem_t problem_nonsquare = problem;
      problem_nonsquare.size.m             = 2048;
      problem_nonsquare.size.n             = 5120;
      auto best_wgm_nonsquare =
          origami::select_workgroup_mapping(problem_nonsquare, hardware, mt_large, mi, wgm_list);

      REQUIRE(best_wgm_small_tile.second > best_wgm_large_tile.second);
      REQUIRE(best_wgm_large_tile.second != best_wgm_nonsquare.second);
    }
  }
}
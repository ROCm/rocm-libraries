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
      auto config = make_config(128, 128, 64, 32, 32, 8, 1, 0, 0);

      auto config_slow = config;
      auto config_fast = config;

      auto latency_config_slow =
          origami::compute_total_latency(problem, hardware_slow, config_slow, hardware_slow.N_CU);
      auto flops_slow = origami::compute_perf_gflops(hardware_slow, problem, latency_config_slow);

      auto latency_config_fast =
          origami::compute_total_latency(problem, hardware_fast, config_fast, hardware_fast.N_CU);
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
          problem, hardware, config, origami::grid_selection_t::k_split_aware, hardware.N_CU);

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

      auto config_large = make_config(256, 256, 32, 32, 32, 8, 1);
      auto skGrid_large = (4096 + 256 - 1) / 256 * (4096 + 256 - 1) / 256;
      auto best_wgm_large_tile =
          origami::select_workgroup_mapping(problem, hardware, config_large, skGrid_large);

      auto config_small = make_config(128, 128, 64, 32, 32, 8, 1);
      auto skGrid_small = (4096 + 128 - 1) / 128 * (4096 + 128 - 1) / 128;
      auto best_wgm_small_tile =
          origami::select_workgroup_mapping(problem, hardware, config_small, skGrid_small);

      // Different problem size for nonsquare test
      origami::problem_t problem_nonsquare = problem;
      problem_nonsquare.size.m             = 2048;
      problem_nonsquare.size.n             = 5120;
      auto skGrid_nonsquare                = (2048 + 128 - 1) / 128 * (5120 + 128 - 1) / 128;

      auto best_wgm_nonsquare = origami::select_workgroup_mapping(
          problem_nonsquare, hardware, config_large, skGrid_nonsquare);

      REQUIRE(std::get<1>(best_wgm_small_tile) > std::get<1>(best_wgm_large_tile));
      REQUIRE(std::get<1>(best_wgm_large_tile) != std::get<1>(best_wgm_nonsquare));
    }
  }
}

TEST_CASE("GEMM: negative_occupancy", "[gemm]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - test negative occupancy") {
      auto hardware              = make_hardware(gpu_arch);
      origami::problem_t problem = {
          .size            = {32, 800000, 16},
          .batch           = 1,
          .a_transpose     = origami::transpose_t::N,
          .b_transpose     = origami::transpose_t::T,
          .a_dtype         = origami::data_type_t::XFloat32,  // element_size_A = 16
          .b_dtype         = origami::data_type_t::XFloat32,
          .mi_dtype        = origami::data_type_t::XFloat32,
          .a_mx_block_size = 0,
          .b_mx_block_size = 0,
      };
      // List 1: config A first, then config B
      std::vector<origami::config_t> origami_config;

      // config[0]
      origami_config.push_back(make_config(256, 256, 32, 16, 16, 32, -1, 6, 0, 0));
      // config[1]
      origami_config.push_back(make_config(32, 256, 16, 32, 32, 8, 2, 6, 0, 0));

      // Call select_config
      auto results = origami::select_config(problem, hardware, origami_config);

      auto best_tile = results[0];
      size_t MT_M    = best_tile.config.mt.m;
      size_t MT_N    = best_tile.config.mt.n;
      size_t MT_K    = best_tile.config.mt.k;
      REQUIRE(MT_M == 32);   //"MT_M should be 32"
      REQUIRE(MT_N == 256);  //"MT_N should be 256"
      REQUIRE(MT_K == 16);   //"MT_K should be 16"
    }
  }
}

TEST_CASE("GEMM: deterministic_tie_breaking", "[gemm]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - Verify deterministic selection") {
      auto hardware = make_hardware(gpu_arch);
      // Square problem size
      auto problem =
          make_problem(1024, 1024, 1024, origami::transpose_t::N, origami::transpose_t::N, 1);

      // Two config with same arithmetic intensity: 256x64x32 and 64x256x32
      // AI = (2 * MT_M * MT_N * MT_K) / (MT_M*MT_K + MT_N*MT_K + MT_M*MT_N)
      // Both have AI = 1048576 / 26624 = 39.38

      // List 1: config A first, then config B
      std::vector<origami::config_t> origami_config_A;
      std::vector<origami::config_t> origami_config_B;

      // config A[0]
      origami_config_A.push_back(make_config(256, 64, 32, 32, 32, 8, 1, 6, 0, 0));
      // config A[1]
      origami_config_A.push_back(make_config(64, 256, 32, 32, 32, 8, 1, 6, 0, 0));

      // config B[0] (reversed order)
      origami_config_B.push_back(make_config(64, 256, 32, 32, 32, 8, 1, 6, 0, 0));
      // config B[1] (reversed order)
      origami_config_B.push_back(make_config(256, 64, 32, 32, 32, 8, 1, 6, 0, 0));

      // Call select_config_mnk with both orderings
      auto results_A_first = origami::select_config(problem, hardware, origami_config_A);

      auto results_B_first = origami::select_config(problem, hardware, origami_config_B);

      // Extract the best tile from each result
      auto best_tile_A_first = results_A_first[0];
      auto best_tile_B_first = results_B_first[0];

      size_t MT_M_A_first = best_tile_A_first.config.mt.m;
      size_t MT_N_A_first = best_tile_A_first.config.mt.n;
      size_t MT_K_A_first = best_tile_A_first.config.mt.k;

      size_t MT_M_B_first = best_tile_B_first.config.mt.m;
      size_t MT_N_B_first = best_tile_B_first.config.mt.n;
      size_t MT_K_B_first = best_tile_B_first.config.mt.k;

      // Verify deterministic selection: both should select the same tile (256x64x32)
      // regardless of input order, using the final tie-breaker (prefer larger MT_M)
      REQUIRE(MT_M_A_first == MT_M_B_first);  //"Selected tile MT_M should be consistent"
      REQUIRE(MT_N_A_first == MT_N_B_first);  //"Selected tile MT_N should be consistent"
      REQUIRE(MT_K_A_first == MT_K_B_first);  //"Selected tile MT_K should be consistent"

      // Verify it selected the tile with larger MT_M (256 > 64)
      REQUIRE(MT_M_A_first == 256);  //"Should prefer tile with larger MT_M"
      REQUIRE(MT_N_A_first == 64);   //"Should prefer tile with larger MT_M"
    }
  }
}
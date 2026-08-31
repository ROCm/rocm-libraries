/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
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

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <limits>
#include "common.hpp"

using Catch::Approx;

// Test functions for origami.hpp/cpp

TEST_CASE("Origami: compute_perf_gflops", "[origami]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - faster clock yields higher GFLOPS") {
      // TODO: Add support for make_hardware using hipDeviceProperties
      auto hardware_slow = make_hardware(gpu_arch);
      auto hardware_fast = make_hardware(gpu_arch);

      if (gpu_arch == 942) {
        const std::string gpu_arch_str = "gfx" + std::to_string(gpu_arch);
        auto gpu_arch_enum             = origami::hardware_t::arch_name_to_enum(gpu_arch_str);
        hardware_slow                  = origami::hardware_t(gpu_arch_enum,
                                            304,
                                            65536,
                                            512 * 1024,  // rf_capacity
                                            8,
                                            1.0,
                                            1.0,
                                            1.0,
                                            4000000,
                                            1.4,
                                            1,
                                            std::make_tuple(0, 0.015, 0));
        hardware_fast                  = origami::hardware_t(gpu_arch_enum,
                                            304,
                                            65536,
                                            512 * 1024,  // rf_capacity
                                            8,
                                            1.0,
                                            1.0,
                                            1.0,
                                            4000000,
                                            1.8,
                                            1,
                                            std::make_tuple(0, 0.015, 0));
      } else if (gpu_arch == 950) {
        const std::string gpu_arch_str = "gfx" + std::to_string(gpu_arch);
        auto gpu_arch_enum             = origami::hardware_t::arch_name_to_enum(gpu_arch_str);
        hardware_slow                  = origami::hardware_t(gpu_arch_enum,
                                            256,
                                            163840,
                                            512 * 1024,  // rf_capacity
                                            8,
                                            1.0,
                                            1.0,
                                            1.0,
                                            4000000,
                                            1.4,
                                            1,
                                            std::make_tuple(0, 0.008, 0));
        hardware_fast                  = origami::hardware_t(gpu_arch_enum,
                                            256,
                                            163840,
                                            512 * 1024,  // rf_capacity
                                            8,
                                            1.0,
                                            1.0,
                                            1.0,
                                            4000000,
                                            1.8,
                                            1,
                                            std::make_tuple(0, 0.008, 0));
      } else if (gpu_arch == 1250) {
        hardware_slow                    = make_hardware(950);
        hardware_slow.arch               = origami::hardware_t::architecture_t::gfx1250;
        hardware_slow.compute_clock_ghz  = 1.4;
        hardware_fast                    = make_hardware(950);
        hardware_fast.arch               = origami::hardware_t::architecture_t::gfx1250;
        hardware_fast.compute_clock_ghz  = 1.8;
      }
      auto problem =
          make_problem(4096, 4096, 1024, origami::transpose_t::T, origami::transpose_t::N, 2);
      auto config = make_config(128, 128, 64, 32, 32, 8);

      auto latency_config_slow =
          origami::gemm::compute_total_latency(problem, hardware_slow, config);
      auto flops_slow = origami::compute_perf_gflops(hardware_slow, problem, latency_config_slow);

      auto latency_config_fast =
          origami::gemm::compute_total_latency(problem, hardware_fast, config);
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
      } else if (gpu_arch == 1250) {
        REQUIRE(arch_enum == origami::hardware_t::architecture_t::gfx1250);
      }
    }
  }
}

TEST_CASE("Origami: gfx1250 complex matrix instructions", "[origami]") {
  auto hardware = make_hardware(1250);

  struct ComplexMiCase {
    origami::data_type_t complex_dtype;
    origami::data_type_t base_dtype;
    const char* string_dtype;
  };

  const ComplexMiCase cases[] = {{origami::data_type_t::ComplexFloat,
                                  origami::data_type_t::Float,
                                  "c32"},
                                 {origami::data_type_t::ComplexDouble,
                                  origami::data_type_t::Double,
                                  "c64"}};

  for (const auto& tc : cases) {
    DYNAMIC_SECTION(tc.string_dtype) {
      const auto instructions = hardware.get_valid_matrix_instructions(tc.complex_dtype);
      REQUIRE(instructions.size() == 1);
      REQUIRE(instructions[0].m == 16);
      REQUIRE(instructions[0].n == 16);
      REQUIRE(instructions[0].k == 4);

      const size_t base_latency    = hardware.get_mi_latency(16, 16, 4, tc.base_dtype);
      const size_t complex_latency = hardware.get_mi_latency(16, 16, 4, tc.complex_dtype);
      REQUIRE(base_latency > 0);
      REQUIRE(complex_latency == base_latency * 4);

      const auto from_string =
          hardware.get_valid_matrix_instructions(origami::string_to_datatype(tc.string_dtype));
      REQUIRE(from_string.size() == 1);
      REQUIRE(from_string[0] == instructions[0]);
    }
  }
}

TEST_CASE("Origami: has_MALL", "[origami]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - MALL support check") {
      auto hardware = make_hardware(gpu_arch);

      // gfx942 and gfx950 have MALL support
      if (gpu_arch == 942 || gpu_arch == 950) { REQUIRE(hardware.has_MALL() == true); }
    }
  }
}

TEST_CASE("Origami: best_grid_size", "[origami]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - grid size selection") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(1024, 1024, 4096);
      auto config   = make_config(256, 256, 64, 32, 32, 8, false, 1);

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

      // List 1: config A first, then config B
      std::vector<origami::config_t> configs;

      // config A[0]
      configs.push_back(make_config(256, 256, 32, 32, 32, 8, false, 1, 6, 0, 0));
      // config A[1]
      configs.push_back(make_config(128, 128, 64, 32, 32, 8, false, 1, 6, 0, 0));
      // config A[2]
      configs.push_back(make_config(64, 64, 64, 32, 32, 8, false, 1, 6, 0, 0));

      auto results = origami::rank_configs(problem, hardware, configs);

      REQUIRE(results.size() == configs.size());
      // Results should be ranked, so latencies should be in ascending order (best first)
      for (size_t i = 0; i < results.size() - 1; i++) {
        REQUIRE(results[i].latency < results[i + 1].latency);
      }
    }
  }
}

TEST_CASE("Origami: best_macro_tile_size mxfp4", "[origami]") {
  for (int gpu_arch : test_architectures) {
    if (gpu_arch != 950) continue;  // mxfp4 only supported on gfx950 for now
    DYNAMIC_SECTION("gfx" << gpu_arch << " - rank configs by latency") {
      auto hardware = make_hardware(gpu_arch);
      hardware.lds_capacity = 400000;
      auto problem  = make_problem(4096, 4096, 32768);

      // List 1: config A first, then config B
      std::vector<origami::config_t> configs;

      // config A[0]
      configs.push_back(make_config(128, 128, 512, 16, 16, 128));
      // config A[1]
      configs.push_back(make_config(256, 256, 512, 16, 16, 128));
      // config A[2]
      configs.push_back(make_config(64, 64, 128, 32, 32, 64));

      problem.a_dtype = origami::data_type_t::Float4;
      problem.b_dtype = origami::data_type_t::Float4;
      problem.a_mx_block_size = 32;
      problem.b_mx_block_size = 32;

      auto results = origami::rank_configs(problem, hardware, configs);

      REQUIRE(results.size() == configs.size());
      REQUIRE(results[0].config.mt.m == 256);
      REQUIRE(results[1].config.mt.m == 128);
      REQUIRE(results[2].config.mt.m == 64);

    }
  }
}

TEST_CASE("Origami: select_workgroup_mapping", "[origami]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - workgroup mapping selection") {
      auto hardware = make_hardware(gpu_arch);
      
      // Large problem size
      auto problem_large  = make_problem(4096, 4096, 8192);
      auto config_large = make_config(256, 256, 32, 32, 32, 8, false, 1);
      auto skGrid_large = ((4096 + 256 - 1) / 256) * ((4096 + 256 - 1) / 256);
      auto mapping_large =
          origami::select_workgroup_mapping(problem_large, hardware, config_large, skGrid_large);

      // Small problem size
      auto problem_small  = make_problem(2048, 2048, 2048);
      auto skGrid_small = ((2048 + 256 - 1) / 256) * ((2048 + 256 - 1) / 256);
      auto mapping_small =
          origami::select_workgroup_mapping(problem_small, hardware, config_large, skGrid_small);

      // Different problem size for nonsquare test
      auto problem_nonsquare = make_problem(5120, 512, 5120);
      auto skGrid_nonsquare  = ((5120 + 256 - 1) / 256) * ((512 + 256 - 1) / 256);
      auto mapping_nonsquare = origami::select_workgroup_mapping(
          problem_nonsquare, hardware, config_large, skGrid_nonsquare);

      REQUIRE(mapping_large.wgmxccchunk >= mapping_small.wgmxccchunk);
      REQUIRE(mapping_large.wgmxcc == mapping_small.wgmxcc);
      REQUIRE(mapping_large.wgm >= mapping_small.wgm);

      REQUIRE(mapping_large.wgmxccchunk >= mapping_nonsquare.wgmxccchunk);
      REQUIRE(mapping_large.wgmxcc == mapping_nonsquare.wgmxcc);
      REQUIRE(mapping_large.wgm >= mapping_nonsquare.wgm);
    }
  }
}

TEST_CASE("origami: negative_occupancy", "[origami]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - test negative occupancy") {
      auto hardware              = make_hardware(gpu_arch);
      origami::problem_t problem = {
          .size            = {32, 800000, 16},
          .batch           = 1,
          .a_transpose     = origami::transpose_t::N,
          .b_transpose     = origami::transpose_t::T,
          .a_dtype         = origami::data_type_t::XFloat32,  // element_size_A = 32
          .b_dtype         = origami::data_type_t::XFloat32,
          .mi_dtype        = origami::data_type_t::XFloat32,
          .a_mx_block_size = 0,
          .b_mx_block_size = 0,
      };
      // List 1: config A first, then config B
      std::vector<origami::config_t> config;

      // config[0]
      config.push_back(make_config(256, 256, 32, 16, 16, 32, false, -1, 6, 0, 0));
      // config[1]
      config.push_back(make_config(32, 256, 16, 32, 32, 8, false, 2, 6, 0, 0));

      // Call select_config
      auto best_tile = origami::select_config(problem, hardware, config);

      size_t MT_M = best_tile.config.mt.m;
      size_t MT_N = best_tile.config.mt.n;
      size_t MT_K = best_tile.config.mt.k;
      REQUIRE(MT_M == 32);   //"MT_M should be 32"
      REQUIRE(MT_N == 256);  //"MT_N should be 256"
      REQUIRE(MT_K == 16);   //"MT_K should be 16"
    }
  }
}

TEST_CASE("origami: deterministic_tie_breaking", "[origami]") {
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
      std::vector<origami::config_t> config_A;
      std::vector<origami::config_t> config_B;

      // config A[0]
      config_A.push_back(make_config(256, 64, 32, 32, 32, 8, false, 1, 6, 0, 0));
      // config A[1]
      config_A.push_back(make_config(64, 256, 32, 32, 32, 8, false, 1, 6, 0, 0));

      // config B[0] (reversed order)
      config_B.push_back(make_config(64, 256, 32, 32, 32, 8, false, 1, 6, 0, 0));
      // config B[1] (reversed order)
      config_B.push_back(make_config(256, 64, 32, 32, 32, 8, false, 1, 6, 0, 0));

      // Call select_config with both orderings
      auto best_tile_A = origami::select_config(problem, hardware, config_A);

      auto best_tile_B = origami::select_config(problem, hardware, config_B);

      size_t MT_M_A_first = best_tile_A.config.mt.m;
      size_t MT_N_A_first = best_tile_A.config.mt.n;
      size_t MT_K_A_first = best_tile_A.config.mt.k;

      size_t MT_M_B_first = best_tile_B.config.mt.m;
      size_t MT_N_B_first = best_tile_B.config.mt.n;
      size_t MT_K_B_first = best_tile_B.config.mt.k;

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

TEST_CASE("origami: Verify deterministic tile selection", "[origami]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - Verify deterministic selection") {
      auto hardware = make_hardware(gpu_arch);

      // problem size
      auto problem =
          make_problem(42598, 153, 128, origami::transpose_t::N, origami::transpose_t::T);

      // List 1: config A first, then config B
      std::vector<origami::config_t> config_A;
      std::vector<origami::config_t> config_B;

      // config A[0]
      config_A.push_back(make_config(256, 160, 32, 32, 32, 8, false, 1, 6, 0, 0));  // Tile A
      // config A[1]
      config_A.push_back(make_config(192, 160, 64, 32, 32, 8, false, 1, 6, 0, 0));  // Tile B

      // config B[0] Previous two tiles + a new one
      config_B.push_back(make_config(256, 160, 32, 32, 32, 8, false, 1, 6, 0, 0));  // Tile A
      // config B[1]
      config_B.push_back(make_config(192, 160, 64, 32, 32, 8, false, 1, 6, 0, 0));  // Tile B
      // config B[2]
      config_B.push_back(make_config(192, 160, 32, 32, 32, 8, false, 1, 6, 0, 0));  // Tile C

      // Call select_config with both tile configs
      auto best_tile_A = origami::select_config(problem, hardware, config_A);

      auto best_tile_B = origami::select_config(problem, hardware, config_B);

      size_t MT_M1 = best_tile_A.config.mt.m;
      size_t MT_N1 = best_tile_A.config.mt.n;
      size_t MT_K1 = best_tile_A.config.mt.k;

      size_t MT_M2 = best_tile_B.config.mt.m;
      size_t MT_N2 = best_tile_B.config.mt.n;
      size_t MT_K2 = best_tile_B.config.mt.k;

      auto winner_is_acceptable =
          [](auto const& actual, auto const& candidate1, auto const& candidate2) -> bool {
        return (actual == candidate1) || (actual == candidate2);
      };

      INFO("Winner is not acceptable");
      REQUIRE(winner_is_acceptable(MT_M2, MT_M1, config_B[2].mt.m));
      REQUIRE(winner_is_acceptable(MT_N2, MT_N1, config_B[2].mt.n));
      REQUIRE(winner_is_acceptable(MT_K2, MT_K1, config_B[2].mt.k));
    }
  }
}

// Unit tests
TEST_CASE("Origami: rank_configs unit test", "[origami]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - rank_configs unit test") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(1024, 1024, 1024);

      // Test 1: Test with empty config list (should throw)
      std::vector<origami::config_t> empty_configs;
      REQUIRE_THROWS_WITH(origami::rank_configs(problem, hardware, empty_configs),
                          "No configurations provided.");

      // Test 2: Test with all invalid configs (LDS capacity exceeded)
      std::vector<origami::config_t> invalid_configs;
      if (gpu_arch == 942) {
        invalid_configs.push_back(make_config(256, 256, 128, 32, 32, 8, false, 1, 6, 0, 0));
        invalid_configs.push_back(make_config(128, 128, 256, 32, 32, 8, false, 1, 6, 0, 0));
        invalid_configs.push_back(make_config(64, 64, 512, 32, 32, 8, false, 1, 6, 0, 0));
      } else if (gpu_arch == 950 || gpu_arch == 1250) {
        invalid_configs.push_back(make_config(512, 512, 256, 32, 32, 8, false, 1, 6, 0, 0));
        invalid_configs.push_back(make_config(128, 128, 512, 32, 32, 8, false, 1, 6, 0, 0));
        invalid_configs.push_back(make_config(256, 256, 512, 32, 32, 8, false, 1, 6, 0, 0));
      }

      if (!invalid_configs.empty()) {
        auto fallback_results = origami::rank_configs(problem, hardware, invalid_configs);
        REQUIRE(fallback_results.size() == invalid_configs.size());
        for (const auto& result : fallback_results) {
          REQUIRE(result.latency == std::numeric_limits<double>::max());
        }
      }

      // Test 2b: Valid configs win over LDS- and heuristic-rejected configs
      if (gpu_arch == 950) {
        auto small_k_problem = make_problem(1024, 1024, 256);
        std::vector<origami::config_t> mixed_configs;
        mixed_configs.push_back(make_config(64, 64, 64, 32, 32, 8, false, 1, 6, 0, 0));

        auto lds_invalid = make_config(512, 512, 256, 32, 32, 8, false, 1, 6, 0, 0);
        mixed_configs.push_back(lds_invalid);

        auto heuristic_rejected = make_config(128, 128, 64, 32, 32, 8, false, 1, 6, 0, 0);
        heuristic_rejected.subtile = true;
        mixed_configs.push_back(heuristic_rejected);

        auto mixed_results = origami::rank_configs(small_k_problem, hardware, mixed_configs);
        REQUIRE(mixed_results.size() == 1);
        REQUIRE(mixed_results.front().latency < std::numeric_limits<double>::max());
        REQUIRE(mixed_results.front().config.mt.m == 64);
      }

      // Test 2c: Catastrophic fallback when every path rejects (LDS + heuristic)
      if (gpu_arch == 950) {
        auto small_k_problem = make_problem(1024, 1024, 256);
        std::vector<origami::config_t> all_rejected_configs;
        all_rejected_configs.push_back(
            make_config(512, 512, 256, 32, 32, 8, false, 1, 6, 0, 0));

        auto heuristic_rejected = make_config(128, 128, 64, 32, 32, 8, false, 1, 6, 0, 0);
        heuristic_rejected.subtile = true;
        all_rejected_configs.push_back(heuristic_rejected);

        auto fallback_results =
            origami::rank_configs(small_k_problem, hardware, all_rejected_configs);
        REQUIRE(fallback_results.size() == all_rejected_configs.size());
        for (const auto& result : fallback_results) {
          REQUIRE(result.latency == std::numeric_limits<double>::max());
        }
      }

      // Test 3: Test tie-breaking with arithmetic intensity (TODO: Find the pair which has same
      // latency but different AI)
      std::vector<origami::config_t> identical_latency_configs;
      identical_latency_configs.push_back(make_config(64, 128, 64, 32, 32, 8, false, 1, 6, 0, 0));
      identical_latency_configs.push_back(make_config(128, 64, 64, 32, 32, 8, false, 1, 6, 0, 0));

      // Test 4: Test tie-breaking with problem dimension preferences
      std::vector<origami::config_t> identical_ai_configs;
      identical_ai_configs.push_back(make_config(128, 64, 128, 32, 32, 8, false, 1, 6, 0, 0));
      identical_ai_configs.push_back(make_config(64, 128, 128, 32, 32, 8, false, 1, 6, 0, 0));

      auto problem_m_greater_than_n = make_problem(2048, 1024, 1024);
      auto results_m_greater_than_n =
          origami::rank_configs(problem_m_greater_than_n, hardware, identical_ai_configs);
      REQUIRE(identical_ai_configs[0].mt.m ==
              results_m_greater_than_n[0].config.mt.m);  // If M > N, prefer tiles with larger MT_M

      auto problem_n_greater_than_m = make_problem(1024, 4096, 1024);
      auto results_n_greater_than_m =
          origami::rank_configs(problem_n_greater_than_m, hardware, identical_ai_configs);
      // REQUIRE(identical_ai_configs[1].mt.n == results_n_greater_than_m[0].config.mt.n); //If N >
      // M, prefer tiles with larger MT_N

      auto problem_m_equals_n = make_problem(1024, 1024, 1024);
      auto results_m_equals_n =
          origami::rank_configs(problem_m_equals_n, hardware, identical_ai_configs);
      REQUIRE(identical_ai_configs[0].mt.m ==
              results_m_equals_n[0].config.mt.m);  // If M == N, prefer tiles with larger MT_M

      // Test 5: Test with different heuristics_variance values
      portable_setenv("ANALYTICAL_GEMM_HEURISTICS_VARIANCE", "0.0", 1);
      // Read back and parse
      double env_val = origami::runtime_options::read_heuristics_variance_from_env();
      REQUIRE(env_val == 0.0);  // Return ANALYTICAL_GEMM_HEURISTICS_VARIANCE is set to 0.0

      portable_setenv("ANALYTICAL_GEMM_HEURISTICS_VARIANCE", "-1.0", 1);
      // Read back and parse
      env_val = origami::runtime_options::read_heuristics_variance_from_env();
      REQUIRE(env_val == 0.01);  // Return default value 0.01 when
                                 // ANALYTICAL_GEMM_HEURISTICS_VARIANCE is set to -1.0

      portable_setenv("ANALYTICAL_GEMM_HEURISTICS_VARIANCE", "1.0", 1);
      // Read back and parse
      env_val = origami::runtime_options::read_heuristics_variance_from_env();
      REQUIRE(env_val == 1.0);  // Return ANALYTICAL_GEMM_HEURISTICS_VARIANCE is set to 1.0
    }
  }
}

TEST_CASE("Origami: select_topk_configs unit test", "[origami]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - select_topk_configs unit test") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(2024, 4096, 768);
      std::vector<origami::config_t> config;

      config.push_back(make_config(256, 160, 32, 32, 32, 8, false, 1, 6, 0, 0));  // Tile A
      config.push_back(make_config(192, 160, 64, 32, 32, 8, false, 1, 6, 0, 0));  // Tile B
      config.push_back(make_config(64, 256, 32, 32, 32, 8, false, 1, 6, 0, 0));   // Tile C

      // Test 1: Test with topk=1 (should return single best config)
      auto single_config = origami::select_topk_configs(problem, hardware, config, 1);
      REQUIRE(single_config.size() == 1);

      // Test 2: Test with topk=3 (should return top 3 configs)
      auto top_k_configs = origami::select_topk_configs(problem, hardware, config, 3);
      REQUIRE(top_k_configs.size() == 3);

      // Test 3: Test with topk larger than available configs (should return all configs)
      top_k_configs = origami::select_topk_configs(problem, hardware, config, 5);
      REQUIRE(top_k_configs.size() == 3);

      // Test 4: Verify results are sorted by latency (best first)
      for (size_t i = 0; i < top_k_configs.size() - 1; i++) {
        REQUIRE(top_k_configs[i].latency < top_k_configs[i + 1].latency);
      }

      // Test 5: Test with empty config list (should throw)
      std::vector<origami::config_t> empty_configs;
      REQUIRE_THROWS_WITH(origami::select_topk_configs(problem, hardware, empty_configs, 5),
                          "No configurations provided.");
    }
  }
}

TEST_CASE("Origami: select_config_mnk unit test", "[origami]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - select_config_mnk unit test") {
      auto hardware = make_hardware(gpu_arch);
      std::vector<origami::config_t> config;

      config.push_back(make_config(256, 128, 64, 32, 32, 8, false, 1, 6, 0, 0));   // Tile A
      config.push_back(make_config(192, 160, 64, 32, 32, 8, false, 1, 6, 0, 0));   // Tile B
      config.push_back(make_config(128, 128, 256, 32, 32, 8, false, 1, 6, 0, 0));  // Tile C

      // Test 1: Test with various M, N, K combinations
      auto result_config = origami::select_config_mnk(4401, 3941, 456, hardware, config);  // M >
                                                                                           // N,K
      REQUIRE(result_config.config.mt.m == config[1].mt.m);

      result_config = origami::select_config_mnk(4500, 8499, 4500, hardware, config);  // N > M,K
      REQUIRE(result_config.config.mt.m == config[0].mt.m);

      result_config = origami::select_config_mnk(3941, 4500, 8499, hardware, config);  // K > M,N
      if (gpu_arch == 942)
        REQUIRE(result_config.config.mt.m == config[0].mt.m);
      else if (gpu_arch == 950)
        REQUIRE(result_config.config.mt.m == config[1].mt.m);

      result_config = origami::select_config_mnk(201, 201, 201, hardware, config);  // M = N = K
      REQUIRE(result_config.config.mt.m == config[0].mt.m);

      // Test 2: Verify default problem settings (transpose, data types)
      origami::problem_t problem = {
          .size            = {2024, 4096, 768},
          .batch           = 1,
          .a_transpose     = origami::transpose_t::T,
          .b_transpose     = origami::transpose_t::N,
          .a_dtype         = origami::data_type_t::Half,  // element_size_A = 16
          .b_dtype         = origami::data_type_t::Half,
          .mi_dtype        = origami::data_type_t::Half,
          .a_mx_block_size = 0,
          .b_mx_block_size = 0,
      };
      auto result_select_config_mnk = origami::select_config_mnk(2024, 4096, 768, hardware, config);
      auto result_select_config     = origami::select_config(problem, hardware, config);
      REQUIRE(result_select_config_mnk.config.mt.m == result_select_config.config.mt.m);
      REQUIRE(result_select_config_mnk.config.mt.n == result_select_config.config.mt.n);
      REQUIRE(result_select_config_mnk.config.mt.k == result_select_config.config.mt.k);
    }
  }
}

// Formocast Simulation Mode Tests

TEST_CASE("Origami: simulation mode basic", "[origami][formocast]") {
  for (int gpu_arch : test_architectures) {
    if (gpu_arch == 1250) continue;  // Formocast not yet supported on gfx1250
    DYNAMIC_SECTION("gfx" << gpu_arch << " - Formocast returns positive latency") {
      auto hardware = make_hardware(gpu_arch);
      auto problem = make_problem(2048, 2048, 2048);
      
      // Create config with simulation mode
      auto config = make_config(128, 128, 32, 16, 16, 16, false, 8, 2);
      config.prediction_mode = origami::prediction_modes_t::simulation;
      
      // Set Formocast-specific parameters (via tensile nested struct)
      config.tensile().depth_u = 32;
      config.tensile().global_split_u = 1;
      config.grvw_a = 4;
      config.grvw_b = 4;
      config.gwvw_d = 4;
      config.tensile().wave_num = 4;
      config.tensile().wave_group_m = 2;
      config.tensile().wave_group_n = 2;
      config.tensile().prefetch_global_read = 2;
      
      double latency = origami::gemm::compute_total_latency(problem, hardware, config);
      
      REQUIRE(latency > 0);
    }
  }
}

TEST_CASE("Origami: simulation mode via compute_total_latency", "[origami][formocast]") {
  for (int gpu_arch : test_architectures) {
    if (gpu_arch == 1250) continue;  // Formocast not yet supported on gfx1250
    DYNAMIC_SECTION("gfx" << gpu_arch << " - compute_total_latency uses Formocast in simulation mode") {
      auto hardware = make_hardware(gpu_arch);
      auto problem = make_problem(2048, 2048, 2048);
      
      // Create config with estimation mode
      auto config_estimation = make_config(128, 128, 32, 16, 16, 16, false, 8, 2);
      config_estimation.prediction_mode = origami::prediction_modes_t::estimation;
      
      // Create config with simulation mode
      auto config_simulation = make_config(128, 128, 32, 16, 16, 16, false, 8, 2);
      config_simulation.prediction_mode = origami::prediction_modes_t::simulation;
      config_simulation.tensile().depth_u = 32;
      config_simulation.tensile().global_split_u = 1;
      config_simulation.grvw_a = 4;
      config_simulation.grvw_b = 4;
      config_simulation.gwvw_d = 4;
      config_simulation.tensile().wave_num = 4;
      config_simulation.tensile().wave_group_m = 2;
      config_simulation.tensile().wave_group_n = 2;
      config_simulation.tensile().prefetch_global_read = 2;
      
      double latency_estimation = origami::gemm::compute_total_latency(
          problem, hardware, config_estimation);
      double latency_simulation = origami::gemm::compute_total_latency(
          problem, hardware, config_simulation);
      
      // Both should be positive
      REQUIRE(latency_estimation > 0);
      REQUIRE(latency_simulation > 0);
      
      // They should produce different results (different models, different units)
      REQUIRE(latency_estimation != latency_simulation);
    }
  }
}

TEST_CASE("Origami: Formocast with various problem sizes", "[origami][formocast]") {
  for (int gpu_arch : test_architectures) {
    if (gpu_arch == 1250) continue;  // Formocast not yet supported on gfx1250
    DYNAMIC_SECTION("gfx" << gpu_arch << " - Formocast handles various problem sizes") {
      auto hardware = make_hardware(gpu_arch);
      
      std::vector<std::tuple<size_t, size_t, size_t>> problem_sizes = {
          {1024, 1024, 1024},
          {2048, 2048, 2048},
          {4096, 4096, 512},
          {512, 4096, 4096},
          {8192, 8192, 1024},
      };
      
      for (const auto& [m, n, k] : problem_sizes) {
        auto problem = make_problem(m, n, k);
        
        auto config = make_config(128, 128, 32, 16, 16, 16, false, 8, 2);
        config.prediction_mode = origami::prediction_modes_t::simulation;
        config.tensile().depth_u = 32;
        config.tensile().global_split_u = 1;
        config.grvw_a = 4;
        config.grvw_b = 4;
        config.gwvw_d = 4;
        config.tensile().wave_num = 4;
        config.tensile().wave_group_m = 2;
        config.tensile().wave_group_n = 2;
        
        double latency = origami::gemm::compute_total_latency(problem, hardware, config);
        
        INFO("Problem size: " << m << "x" << n << "x" << k);
        REQUIRE(latency > 0);
      }
    }
  }
}

TEST_CASE("Origami: Formocast with different tile sizes", "[origami][formocast]") {
  for (int gpu_arch : test_architectures) {
    if (gpu_arch == 1250) continue;  // Formocast not yet supported on gfx1250
    DYNAMIC_SECTION("gfx" << gpu_arch << " - Formocast handles different tile sizes") {
      auto hardware = make_hardware(gpu_arch);
      auto problem = make_problem(4096, 4096, 4096);
      
      std::vector<std::tuple<size_t, size_t, size_t>> tile_sizes = {
          {64, 64, 32},
          {128, 128, 32},
          {256, 256, 32},
          {128, 256, 64},
          {256, 128, 64},
      };
      
      for (const auto& [mt_m, mt_n, mt_k] : tile_sizes) {
        auto config = make_config(mt_m, mt_n, mt_k, 16, 16, 16, false, 8, 2);
        config.prediction_mode = origami::prediction_modes_t::simulation;
        config.tensile().depth_u = mt_k;
        config.tensile().global_split_u = 1;
        config.grvw_a = 4;
        config.grvw_b = 4;
        config.gwvw_d = 4;
        config.tensile().wave_num = 4;
        config.tensile().wave_group_m = 2;
        config.tensile().wave_group_n = 2;
        
        double latency = origami::gemm::compute_total_latency(problem, hardware, config);
        
        INFO("Tile size: " << mt_m << "x" << mt_n << "x" << mt_k);
        REQUIRE(latency > 0);
      }
    }
  }
}

TEST_CASE("Origami: Formocast config fields have correct defaults", "[origami][formocast]") {
  origami::config_t config;
  
  // Check default values for Tensile-specific fields (via nested struct)
  REQUIRE(config.tensile().depth_u == 0);
  REQUIRE(config.tensile().global_split_u == 1);
  REQUIRE(config.tensile().global_accumulation == 0);
  REQUIRE(config.tensile().local_split_u == 1);
  REQUIRE(config.grvw_a == 1);
  REQUIRE(config.grvw_b == 1);
  REQUIRE(config.gwvw_d == 1);
  REQUIRE(config.tensile().direct_to_vgpr_a == false);
  REQUIRE(config.tensile().direct_to_vgpr_b == false);
  REQUIRE(config.tensile().direct_to_lds_a == false);
  REQUIRE(config.tensile().direct_to_lds_b == false);
  REQUIRE(config.tensile().wave_num == 4);
  REQUIRE(config.tensile().wave_group_m == 2);
  REQUIRE(config.tensile().wave_group_n == 2);
  REQUIRE(config.tensile().prefetch_global_read == 2);
  REQUIRE(config.prediction_mode == origami::prediction_modes_t::estimation);
}

TEST_CASE("Origami: select_staggerU unit test", "[origami]") {
  for (int gpu_arch : test_architectures) {
    if (gpu_arch != 950) continue;  // StaggerU tuned for gfx950
    DYNAMIC_SECTION("gfx" << gpu_arch << " - select_staggerU") {
      auto hardware = make_hardware(gpu_arch);

      // Helper to compute skGrid (no stream-K, just numMTs)
      auto compute_skGrid = [](size_t M, size_t N, size_t MT_M, size_t MT_N) {
        return ((M + MT_M - 1) / MT_M) * ((N + MT_N - 1) / MT_N);
      };

      // Test 1: Non-temporal access uses max_staggerU
      {
        auto problem = make_problem(4096, 4096, 2048);
        auto config  = make_config(128, 128, 64, 16, 16, 32, false, 1, 1, 4, 0);  // nta=4
        auto skGrid  = compute_skGrid(4096, 4096, 128, 128);
        auto result  = origami::select_staggerU(problem, hardware, config, skGrid, 4);
        REQUIRE(result.staggerU == 32);
      }

      // Test 2: Batch > 1 should disable stagger
      {
        auto problem = make_problem(4096, 4096, 2048, origami::transpose_t::T, origami::transpose_t::N, 2);
        auto config  = make_config(128, 128, 64, 16, 16, 32);
        auto skGrid  = compute_skGrid(4096, 4096, 128, 128);
        auto result  = origami::select_staggerU(problem, hardware, config, skGrid, 4);
        REQUIRE(result.staggerU == 0);
      }

      // Test 3: Split-K should disable stagger
      {
        auto problem = make_problem(1024, 1024, 1024);
        auto config  = make_config(128, 128, 64, 16, 16, 32);
        size_t numMTs = ((1024 + 127) / 128) * ((1024 + 127) / 128);  // 64
        size_t skGrid = numMTs * 2;  // split_factor = 2
        auto result   = origami::select_staggerU(problem, hardware, config, skGrid, 4);
        REQUIRE(result.staggerU == 0);
      }

      // Test 4: Basic TN case — stagger should be enabled
      {
        auto problem = make_problem(2048, 2048, 2048, origami::transpose_t::T, origami::transpose_t::N);
        auto config  = make_config(128, 128, 64, 16, 16, 32);
        auto skGrid  = compute_skGrid(2048, 2048, 128, 128);
        auto result  = origami::select_staggerU(problem, hardware, config, skGrid, 4);
        REQUIRE(result.staggerU > 0);
        // StaggerU should be power of 2
        REQUIRE((result.staggerU & (result.staggerU - 1)) == 0);
      }

      // Test 5: StaggerU mapping should be SUM1 for positive WGM when A contention dominates
      // [1320, 256, 2048] MT=64x32x256 WGM=8: L2Tile_N=8 > L2Tile_M=4, row-mates share A
      {
        auto problem = make_problem(1320, 256, 2048, origami::transpose_t::T, origami::transpose_t::N);
        problem.a_dtype = origami::data_type_t::BFloat16;
        problem.b_dtype = origami::data_type_t::BFloat16;
        auto config  = make_config(64, 32, 256, 16, 16, 32);
        auto skGrid  = compute_skGrid(1320, 256, 64, 32);
        auto result  = origami::select_staggerU(problem, hardware, config, skGrid, 8);
        REQUIRE(result.staggerU > 0);
        REQUIRE(result.staggerUMapping == 1);  // SUM1: distribute A reads
      }

      // Test 6: StaggerU mapping should be SUM0 when B contention dominates
      // [128, 1024, 144] MT=32x64x16 WGM=1 FP32: L2Tile_M > L2Tile_N, B is the bottleneck
      {
        auto problem = make_problem(128, 1024, 144, origami::transpose_t::T, origami::transpose_t::N);
        problem.a_dtype = origami::data_type_t::Float;
        problem.b_dtype = origami::data_type_t::Float;
        auto config  = make_config(32, 64, 16, 16, 16, 4);
        auto skGrid  = compute_skGrid(128, 1024, 32, 64);
        auto result  = origami::select_staggerU(problem, hardware, config, skGrid, 1);
        REQUIRE(result.staggerU > 0);
        REQUIRE(result.staggerUMapping == 0);  // SUM0: distribute B reads
      }

      // Test 7: StaggerU value should be power of 2 and <= 32
      {
        auto problem = make_problem(4096, 4096, 4096, origami::transpose_t::T, origami::transpose_t::N);
        auto config  = make_config(128, 128, 64, 16, 16, 32);
        auto skGrid  = compute_skGrid(4096, 4096, 128, 128);
        auto result  = origami::select_staggerU(problem, hardware, config, skGrid, 4);
        if (result.staggerU > 0) {
          REQUIRE(result.staggerU <= 32);
          REQUIRE((result.staggerU & (result.staggerU - 1)) == 0);
          REQUIRE(result.staggerUMapping <= 1);
        }
      }

      // Test 8: Stride shift ensures each stagger step crosses a cache line
      {
        auto problem = make_problem(2048, 2048, 2048, origami::transpose_t::T, origami::transpose_t::N);
        auto config  = make_config(64, 64, 16, 16, 16, 16);  // Small MT_K=16
        auto skGrid  = compute_skGrid(2048, 2048, 64, 64);
        auto result  = origami::select_staggerU(problem, hardware, config, skGrid, 4);
        if (result.staggerU > 0) {
          // With MT_K=16, bpe=2: bytes_per_k_iter=32 < 128, so shift should be > 0
          size_t bytes_per_k = 16 * 2;  // MT_K * bpe
          REQUIRE((bytes_per_k << result.staggerUStrideShift) >= 128);
        }
      }
    }
  }
}

TEST_CASE("Origami: select_workgroup_mapping unit test", "[Origami]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - select_workgroup_mapping unit test") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(4096, 4096, 8192);
      auto config   = make_config(256, 256, 32, 32, 32, 8, false, 1, 6, 0, 0);
      auto skGrid   = ((4096 + 256 - 1) / 256) * ((4096 + 256 - 1) / 256);
      size_t numMT_M = (problem.size.m + config.mt.m - 1) / config.mt.m;
      size_t numMT_N = (problem.size.n + config.mt.n - 1) / config.mt.n;

      // Default values
      size_t default_wgmxccchunk = 0;
      size_t default_wgmxcc      = hardware.NUM_XCD;
      int32_t default_wgm        = static_cast<int32_t>(std::ceil(std::sqrt(hardware.N_CU / hardware.NUM_XCD)));
      size_t chunk_size          = std::min((numMT_M * numMT_N + hardware.NUM_XCD - 1) / hardware.NUM_XCD, 
                                            (hardware.N_CU + hardware.NUM_XCD - 1) / hardware.NUM_XCD);

      // Test 1: Test non-temporal cache hints (nta > 3, ntb < 4; nta < 4, ntb > 3; both > 3)
      config.cache_hints_a = 4;
      config.cache_hints_b = 3;
      auto out_wgm_1 =
          origami::select_workgroup_mapping(problem, hardware, config, skGrid);  // nta > 3, ntb < 4
      REQUIRE(out_wgm_1.wgmxccchunk == chunk_size);
      REQUIRE(out_wgm_1.wgmxcc == default_wgmxcc);
      REQUIRE(out_wgm_1.wgm == numMT_N);

      config.cache_hints_a = 3;
      config.cache_hints_b = 4;
      auto out_wgm_2 =
          origami::select_workgroup_mapping(problem, hardware, config, skGrid);  // nta < 4, ntb > 3
      REQUIRE(out_wgm_2.wgmxccchunk == chunk_size);
      REQUIRE(out_wgm_2.wgmxcc == default_wgmxcc);
      REQUIRE(out_wgm_2.wgm == -numMT_M);

      config.cache_hints_a = 4;
      config.cache_hints_b = 4;
      auto out_wgm_3 =
          origami::select_workgroup_mapping(problem, hardware, config, skGrid);  // both > 3
      REQUIRE(out_wgm_3.wgmxccchunk == default_wgmxccchunk);
      REQUIRE(out_wgm_3.wgmxcc == default_wgmxcc);
      REQUIRE(out_wgm_3.wgm == 1);

      // reset cache_hints to 0
      config.cache_hints_a = 0;
      config.cache_hints_b = 0;

      // Test 2: Test batched GEMM cases (batch > 1)
      auto problem_batch =
          make_problem(4096, 4096, 8192, origami::transpose_t::T, origami::transpose_t::N, 2);
      auto out_wgm_batch =
          origami::select_workgroup_mapping(problem_batch, hardware, config, skGrid);
      REQUIRE(out_wgm_batch.wgmxccchunk == default_wgmxccchunk);
      REQUIRE(out_wgm_batch.wgmxcc == default_wgmxcc);
      REQUIRE(out_wgm_batch.wgm == default_wgm);

      // Test 3: Test small GEMMs
      auto problem_small = make_problem(1024, 1024, 1024);
      auto skGrid_small  = (1024 + 256 - 1) / 256 * (1024 + 256 - 1) / 256;
      auto out_wgm_problem_small =
          origami::select_workgroup_mapping(problem_small, hardware, config, skGrid_small);
      REQUIRE(out_wgm_problem_small.wgmxccchunk == default_wgmxccchunk);
      REQUIRE(out_wgm_problem_small.wgmxcc == default_wgmxcc);
      REQUIRE(out_wgm_problem_small.wgm == 2);

      // Test 4: Test cases where splitFactor is multiple of NUM_XCD
      auto out_wgm_split_multiple_num_xcd =
          origami::select_workgroup_mapping(problem, hardware, config, 2048);
      REQUIRE(out_wgm_split_multiple_num_xcd.wgmxccchunk == default_wgmxccchunk);
      REQUIRE(out_wgm_split_multiple_num_xcd.wgmxcc == 0);
      REQUIRE(out_wgm_split_multiple_num_xcd.wgm == 1);

      // Test 5: Test cases tall cases (M >> N) with numMT_N <= 8
      auto problem_tall = make_problem(409600, 256, 256);
      auto skGrid_tall  = (409600 + 256 - 1) / 256 * (256 + 256 - 1) / 256;
      auto out_wgm_tall =
          origami::select_workgroup_mapping(problem_tall, hardware, config, skGrid_tall);
      REQUIRE(out_wgm_tall.wgmxccchunk == default_wgmxccchunk);
      REQUIRE(out_wgm_tall.wgmxcc == 8);
      REQUIRE(out_wgm_tall.wgm == 1);

      // Test 6: Test MallIsImportant cases
      auto problem_mall_is_important = make_problem(7680, 7680, 256);
      auto out_wgm_mall_is_important =
          origami::select_workgroup_mapping(problem_mall_is_important, hardware, config, 900);
      REQUIRE(out_wgm_mall_is_important.wgmxccchunk == default_wgmxccchunk);
      REQUIRE(out_wgm_mall_is_important.wgmxcc == default_wgmxcc);
      REQUIRE(out_wgm_mall_is_important.wgm == 5);

      // Test 7: Test WGM prediction with various wgmList values.
      // bf16 with MT_K=32 is 64 bytes per K-iteration, so this shape sits exactly at
      // kCoherentMinBytesPerKIter and takes the smaller of the tied WGM candidates.
      auto out_wgm = origami::select_workgroup_mapping(problem, hardware, config, skGrid);
      REQUIRE(out_wgm.wgmxccchunk == default_wgmxccchunk);
      REQUIRE(out_wgm.wgmxcc == default_wgmxcc);
      if (gpu_arch == 942)
        REQUIRE(out_wgm.wgm == 4);
      else if (gpu_arch == 950)
        REQUIRE(out_wgm.wgm == 4);
      else if (gpu_arch == 1250)
        REQUIRE(out_wgm.wgm == 4);

      // K-coherent split-K needs a grid that splits K within one wave of
      // workgroups, so these cases use fewer tiles than the machine has CUs.
      // bf16 with MT_K=64 gives 128 bytes per k-iter, one full cache line.
      auto split_problem     = make_problem(2048, 2048, 65536);
      size_t numMTs_split    = (2048 / 256) * (2048 / 256);  // 64 tiles
      size_t skGrid_one_wave = 256;                          // <= N_CU on every test arch
      REQUIRE(skGrid_one_wave <= hardware.N_CU);

      // Test 8: K-coherent split-K mapping is selected when split-K workgroups
      // cover cache-line-aligned K chunks and the split factor is useful across XCDs.
      {
        auto split_config = config;
        split_config.mt.k = 64;

        auto out_wgm_splitk = origami::select_workgroup_mapping(
            split_problem, hardware, split_config, skGrid_one_wave);
        REQUIRE(out_wgm_splitk.wgmxccsplitk == skGrid_one_wave / numMTs_split);
        REQUIRE(out_wgm_splitk.wgmxccchunk == skGrid_one_wave / hardware.NUM_XCD);
        REQUIRE(out_wgm_splitk.wgmxcc == hardware.NUM_XCD);
      }

      // Non-multiple skGrid is allowed: the first K*MN workgroups are remapped and
      // tail workgroups are identity-mapped by codegen.
      {
        auto split_config = config;
        split_config.mt.k = 64;
        auto skGrid_with_tail = skGrid_one_wave - 1;

        auto out_wgm_splitk_tail = origami::select_workgroup_mapping(
            split_problem, hardware, split_config, skGrid_with_tail);
        REQUIRE(out_wgm_splitk_tail.wgmxccsplitk == skGrid_with_tail / numMTs_split);
        REQUIRE(out_wgm_splitk_tail.wgmxccchunk == skGrid_with_tail / hardware.NUM_XCD);
        REQUIRE(out_wgm_splitk_tail.wgmxcc == hardware.NUM_XCD);
      }

      // A grid larger than the CU budget spans more than one wave of workgroups,
      // so there are no k-levels to group and the mapping stays disabled. This
      // also keeps the chunk inside the 8-bit field of the kernel argument.
      {
        auto split_config = config;
        split_config.mt.k = 64;

        for (size_t skGrid_multi_wave :
             {hardware.N_CU + 1, 2 * hardware.N_CU, 3 * hardware.N_CU}) {
          INFO("skGrid=" << skGrid_multi_wave << " N_CU=" << hardware.N_CU);
          auto out_wgm_multi_wave = origami::select_workgroup_mapping(
              split_problem, hardware, split_config, skGrid_multi_wave);
          REQUIRE(out_wgm_multi_wave.wgmxccsplitk == 0);
        }
      }

      // A CU budget below the physical count tightens the gate: the same grid
      // that fits a full machine spans more than one wave of the budget.
      {
        auto split_config      = config;
        split_config.mt.k      = 64;
        auto capped_problem    = split_problem;
        capped_problem.num_cus = skGrid_one_wave / 2;

        auto out_wgm_capped = origami::select_workgroup_mapping(
            capped_problem, hardware, split_config, skGrid_one_wave);
        REQUIRE(out_wgm_capped.wgmxccsplitk == 0);
      }

      // If the split factor is already XCD-aligned, hardware round-robin dispatch
      // distributes k-splits evenly and K-coherent remapping should stay disabled.
      {
        auto split_config = config;
        split_config.mt.k = 64;
        auto skGrid_xcd_aligned = hardware.NUM_XCD * numMT_M * numMT_N;

        auto out_wgm_splitk_xcd_aligned =
            origami::select_workgroup_mapping(problem, hardware, split_config, skGrid_xcd_aligned);
        REQUIRE(out_wgm_splitk_xcd_aligned.wgmxccsplitk == 0);
        REQUIRE(out_wgm_splitk_xcd_aligned.wgmxccchunk == 0);
        REQUIRE(out_wgm_splitk_xcd_aligned.wgmxcc == 0);
      }
    }
  }
}

TEST_CASE("Origami: config_t hash function", "[origami]") {
  SECTION("hash works with no backend (monostate)") {
    origami::config_t config1;
    config1.mt        = {256, 256, 32};
    config1.mi        = {16, 16, 16};
    config1.occupancy = 4;

    origami::config_t config2;
    config2.mt        = {256, 256, 32};
    config2.mi        = {16, 16, 16};
    config2.occupancy = 4;

    // Identical configs should have the same hash
    REQUIRE(config1.hash() == config2.hash());

    // Hash should be non-zero for a valid config
    REQUIRE(config1.hash() != 0);
  }

  SECTION("hash works with tensile backend") {
    origami::config_t config1;
    config1.mt                    = {256, 256, 32};
    config1.mi                    = {16, 16, 16};
    config1.occupancy             = 4;
    config1.tensile().depth_u        = 16;
    config1.tensile().global_split_u = 2;
    config1.tensile().wave_num       = 4;

    origami::config_t config2;
    config2.mt                    = {256, 256, 32};
    config2.mi                    = {16, 16, 16};
    config2.occupancy             = 4;
    config2.tensile().depth_u        = 16;
    config2.tensile().global_split_u = 2;
    config2.tensile().wave_num       = 4;

    // Identical configs with tensile params should have the same hash
    REQUIRE(config1.hash() == config2.hash());
  }

  SECTION("different configs produce different hashes") {
    origami::config_t config1;
    config1.mt        = {256, 256, 32};
    config1.mi        = {16, 16, 16};
    config1.occupancy = 4;

    origami::config_t config2;
    config2.mt        = {128, 128, 32};  // Different tile size
    config2.mi        = {16, 16, 16};
    config2.occupancy = 4;

    REQUIRE(config1.hash() != config2.hash());
  }

  SECTION("config with vs without tensile params produces different hashes") {
    origami::config_t config_no_backend;
    config_no_backend.mt        = {256, 256, 32};
    config_no_backend.mi        = {16, 16, 16};
    config_no_backend.occupancy = 4;

    origami::config_t config_with_backend;
    config_with_backend.mt               = {256, 256, 32};
    config_with_backend.mi               = {16, 16, 16};
    config_with_backend.occupancy        = 4;
    config_with_backend.tensile().depth_u = 16;

    REQUIRE(config_no_backend.hash() != config_with_backend.hash());
  }

  SECTION("different tensile params produce different hashes") {
    origami::config_t config1;
    config1.mt                = {256, 256, 32};
    config1.mi                = {16, 16, 16};
    config1.occupancy         = 4;
    config1.tensile().depth_u = 16;

    origami::config_t config2;
    config2.mt                = {256, 256, 32};
    config2.mi                = {16, 16, 16};
    config2.occupancy         = 4;
    config2.tensile().depth_u = 32;  // Different depth_u

    REQUIRE(config1.hash() != config2.hash());
  }
}

TEST_CASE("Origami: tensile_params_t hash function", "[origami]") {
  SECTION("identical params produce same hash") {
    origami::tensile_params_t params1;
    params1.depth_u        = 16;
    params1.global_split_u = 2;
    params1.wave_num       = 4;

    origami::tensile_params_t params2;
    params2.depth_u        = 16;
    params2.global_split_u = 2;
    params2.wave_num       = 4;

    REQUIRE(params1.hash() == params2.hash());
  }

  SECTION("different params produce different hashes") {
    origami::tensile_params_t params1;
    params1.depth_u = 16;

    origami::tensile_params_t params2;
    params2.depth_u = 32;

    REQUIRE(params1.hash() != params2.hash());
  }

  SECTION("all fields contribute to hash") {
    origami::tensile_params_t base_params;

    // Changing each field should produce a different hash
    origami::tensile_params_t params_depth_u = base_params;
    params_depth_u.depth_u                   = 32;
    REQUIRE(base_params.hash() != params_depth_u.hash());

    origami::tensile_params_t params_gsu = base_params;
    params_gsu.global_split_u            = 4;
    REQUIRE(base_params.hash() != params_gsu.hash());

    origami::tensile_params_t params_wave = base_params;
    params_wave.wave_num                  = 8;
    REQUIRE(base_params.hash() != params_wave.hash());

    origami::tensile_params_t params_dtvgpr = base_params;
    params_dtvgpr.direct_to_vgpr_a          = true;
    REQUIRE(base_params.hash() != params_dtvgpr.hash());

    origami::tensile_params_t params_swizzle = base_params;
    params_swizzle.swizzle_a                 = true;
    REQUIRE(base_params.hash() != params_swizzle.hash());

    origami::tensile_params_t params_wgm_xcc = base_params;
    params_wgm_xcc.workgroup_mapping_xcc     = 8;
    REQUIRE(base_params.hash() != params_wgm_xcc.hash());
  }
}

TEST_CASE("Origami: resolve_num_cus", "[origami]") {
  // 0 means "use all CUs": falls back to the physical count.
  REQUIRE(origami::resolve_num_cus(0, 256) == 256);
  // A non-zero request below the physical count caps the budget.
  REQUIRE(origami::resolve_num_cus(64, 256) == 64);
  // A request equal to the physical count is unchanged.
  REQUIRE(origami::resolve_num_cus(256, 256) == 256);
  // A request above the physical count clamps to the physical count.
  REQUIRE(origami::resolve_num_cus(512, 256) == 256);
  // A negative request (invalid input) means "use all CUs".
  REQUIRE(origami::resolve_num_cus(-4, 256) == 256);
}

TEST_CASE("Origami: num_cus adjusts solution selection", "[origami]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - CU budget affects ranked latency") {
      auto       hardware = make_hardware(gpu_arch);
      const auto mi = hardware.get_recommended_matrix_instruction(origami::data_type_t::BFloat16);

      // CU-bound problem: far more output tiles than CUs, so the usable CU
      // count directly drives the number of timesteps (and thus latency).
      auto make_p = [&](size_t num_cus) {
        auto p    = make_problem(8192, 8192, 4096);
        p.num_cus = num_cus;
        return p;
      };

      std::vector<origami::config_t> configs;
      configs.push_back(make_config(256, 256, 64, mi.m, mi.n, mi.k, false, 1, 4));

      const auto full   = origami::rank_configs(make_p(0), hardware, configs);
      const auto equal  = origami::rank_configs(make_p(hardware.N_CU), hardware, configs);
      const auto capped = origami::rank_configs(make_p(hardware.N_CU / 4), hardware, configs);

      // num_cus == 0 and num_cus == N_CU both mean "use all CUs": identical.
      REQUIRE(equal[0].latency == Approx(full[0].latency));

      // Restricting to fewer CUs cannot make a CU-bound GEMM faster; here it is
      // strictly slower because timesteps scale with 1 / usable_cus.
      REQUIRE(capped[0].latency > full[0].latency);
    }
  }
}

TEST_CASE("Origami: num_cus in mapping selectors", "[origami]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - mapping honors CU budget") {
      auto         hardware = make_hardware(gpu_arch);
      auto         config   = make_config(256, 256, 64);
      const size_t skGrid   = 2048;

      auto problem_full    = make_problem(8192, 8192, 8192);
      problem_full.num_cus = 0;
      auto problem_equal    = problem_full;
      problem_equal.num_cus = hardware.N_CU;  // equivalent to "use all CUs"

      const auto wgm_full  = origami::select_workgroup_mapping(problem_full, hardware, config, skGrid);
      const auto wgm_equal = origami::select_workgroup_mapping(problem_equal, hardware, config, skGrid);

      // num_cus == N_CU must behave exactly like num_cus == 0.
      REQUIRE(wgm_full.wgm == wgm_equal.wgm);
      REQUIRE(wgm_full.wgmxcc == wgm_equal.wgmxcc);
      REQUIRE(wgm_full.wgmxccchunk == wgm_equal.wgmxccchunk);

      const auto su_full  = origami::select_staggerU(problem_full, hardware, config, skGrid, wgm_full.wgm);
      const auto su_equal = origami::select_staggerU(problem_equal, hardware, config, skGrid, wgm_equal.wgm);
      REQUIRE(su_full.staggerU == su_equal.staggerU);
      REQUIRE(su_full.staggerUMapping == su_equal.staggerUMapping);
      REQUIRE(su_full.staggerUStrideShift == su_equal.staggerUStrideShift);

      // A real cap must actually change the mapping for a CU-sensitive problem:
      // many square tiles (numMTs >> CUs) so the per-XCD WGM search space
      // (numWGsPerXCD / wgm_cap_size) shrinks with the usable CU count.
      const size_t MT       = 128;
      auto         cuConfig = make_config(MT, MT, 64);
      const size_t numMTs   = (8192 / MT) * (8192 / MT);
      const size_t skGridDP = numMTs;  // data-parallel: split_factor == 1

      auto problem_cu_full    = make_problem(8192, 8192, 8192);
      problem_cu_full.num_cus = 0;
      auto problem_cu_capped    = problem_cu_full;
      problem_cu_capped.num_cus = hardware.N_CU / 4;  // genuine cap

      const auto wgm_cu_full =
          origami::select_workgroup_mapping(problem_cu_full, hardware, cuConfig, skGridDP);
      const auto wgm_cu_capped =
          origami::select_workgroup_mapping(problem_cu_capped, hardware, cuConfig, skGridDP);

      const bool mapping_changed = wgm_cu_full.wgm != wgm_cu_capped.wgm
                                   || wgm_cu_full.wgmxcc != wgm_cu_capped.wgmxcc
                                   || wgm_cu_full.wgmxccchunk != wgm_cu_capped.wgmxccchunk;
      INFO("full wgm=" << wgm_cu_full.wgm << " capped wgm=" << wgm_cu_capped.wgm);
      REQUIRE(mapping_changed);
    }
  }
}

TEST_CASE("Origami: num_cus changes selected config", "[origami]") {
  // The CU budget does not just scale latency magnitude — it can change *which*
  // config Origami ranks first. Two candidates with comparable per-tile
  // efficiency but different tile counts trade places: the tile that produces
  // more output tiles wins when all CUs are available (higher compute/memory
  // efficiency), while the tile that produces fewer output tiles wins under a
  // tight CU budget, because num_timesteps ~= ceil(numMTs / usable_cus) and
  // fewer tiles means fewer timesteps once work no longer fits in one wave.
  //
  // For an 8192x8192 output:
  //   256x128 -> ceil(8192/256) * ceil(8192/128) = 32 * 64 = 2048 tiles
  //   192x192 -> ceil(8192/192) * ceil(8192/192) = 43 * 43 = 1849 tiles
  // 256x128 (more, more efficient tiles) wins with all CUs; 192x192 (fewer
  // tiles -> fewer timesteps) wins when the budget is squeezed to N_CU / 8.
  // Verified to hold on every architecture in test_architectures.
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - CU budget flips ranked winner") {
      auto       hardware = make_hardware(gpu_arch);
      const auto mi = hardware.get_recommended_matrix_instruction(origami::data_type_t::BFloat16);

      std::vector<origami::config_t> configs;
      configs.push_back(make_config(192, 192, 64, mi.m, mi.n, mi.k, false, 1, 4));  // fewer tiles
      configs.push_back(make_config(256, 128, 64, mi.m, mi.n, mi.k, false, 1, 4));  // more tiles

      auto make_p = [&](size_t num_cus) {
        auto p    = make_problem(8192, 8192, 4096);
        p.num_cus = num_cus;
        return p;
      };

      const size_t capped_cus = hardware.N_CU / 8;  // tight budget

      const auto full   = origami::rank_configs(make_p(0), hardware, configs);
      const auto capped = origami::rank_configs(make_p(capped_cus), hardware, configs);

      const auto full_mt   = full[0].config.mt;
      const auto capped_mt = capped[0].config.mt;

      INFO("gfx" << gpu_arch << " N_CU=" << hardware.N_CU << " capped_cus=" << capped_cus
                 << " | all-CUs winner=" << full_mt.m << "x" << full_mt.n << "x" << full_mt.k
                 << " (lat " << full[0].latency << ")"
                 << " | capped winner=" << capped_mt.m << "x" << capped_mt.n << "x" << capped_mt.k
                 << " (lat " << capped[0].latency << ")");

      // The identity of the winning config must change with the CU budget,
      // not merely the latency magnitude.
      const bool winner_flipped =
          full_mt.m != capped_mt.m || full_mt.n != capped_mt.n || full_mt.k != capped_mt.k;
      REQUIRE(winner_flipped);

      // Pin the concrete expected winners so the intent is unambiguous.
      REQUIRE(full_mt.m == 256);
      REQUIRE(full_mt.n == 128);
      REQUIRE(capped_mt.m == 192);
      REQUIRE(capped_mt.n == 192);
    }
  }
}

TEST_CASE("gfx950 pci_chip_id id75a0 vs id75a8", "[hardware]") {
  using origami::hardware_t;
  const auto c_def = hardware_t::get_gfx950_arch_constants(std::nullopt);
  const auto c_v2 = hardware_t::get_gfx950_arch_constants(std::make_optional(0x75a8));
  REQUIRE(c_def.mem1_perf_ratio != c_v2.mem1_perf_ratio);
  REQUIRE(c_def.mem2_perf_ratio != c_v2.mem2_perf_ratio);
  REQUIRE(c_def.mem3_perf_ratio != c_v2.mem3_perf_ratio);
  REQUIRE(c_def.mem_bw_per_wg_coefficients != c_v2.mem_bw_per_wg_coefficients);
  REQUIRE(c_def.parallel_mi_cu == c_v2.parallel_mi_cu);

  const auto g942_v2 = hardware_t::get_arch_constants(
      hardware_t::architecture_t::gfx942, std::make_optional(0x75a8));
  const auto g942_def =
      hardware_t::get_arch_constants(hardware_t::architecture_t::gfx942);
  REQUIRE(g942_v2.mem1_perf_ratio == g942_def.mem1_perf_ratio);

  const auto hw_def = hardware_t::get_hardware_for_arch(hardware_t::architecture_t::gfx950,
                                                        256,               // N_CU
                                                        65536,             // lds_capacity
                                                        512 * 1024,        // rf_capacity
                                                        4 * 1024 * 1024,   // L2_capacity
                                                        2'100'000,         // compute_clock_khz
                                                        std::nullopt);     // pci_chip_id
  const auto hw_v2 = hardware_t::get_hardware_for_arch(hardware_t::architecture_t::gfx950,
                                                       128,
                                                       65536,
                                                       512 * 1024,
                                                       4 * 1024 * 1024,
                                                       2'100'000,
                                                       std::make_optional(0x75a8));
  REQUIRE(hw_def.mem1_perf_ratio != hw_v2.mem1_perf_ratio);
  REQUIRE(hw_def.mem_bw_per_wg_coefficients == c_def.mem_bw_per_wg_coefficients);
  REQUIRE(hw_v2.mem_bw_per_wg_coefficients == c_v2.mem_bw_per_wg_coefficients);
  REQUIRE_FALSE(hw_def.pci_chip_id.has_value());
  REQUIRE(hw_v2.pci_chip_id.has_value());
  REQUIRE(hw_v2.pci_chip_id.value() == 0x75a8);
}

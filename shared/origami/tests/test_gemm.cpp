/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025-2026 AMD ROCm(TM) Software
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
#include "common.hpp"

using Catch::Approx;

// Test functions for gemm.hpp/cpp

TEST_CASE("GEMM: compute_num_matrix_instructions", "[gemm]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - 128x128x64 with 16x16x16") {
      auto hardware = make_hardware(gpu_arch);
      origami::dim3_t mt{128, 128, 64};
      origami::dim3_t mi{16, 16, 16};
      auto num_instructions = origami::compute_number_matrix_instructions(mt, mi);
      REQUIRE(num_instructions == 256);  // 8 * 8 * 4
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - 16x16x64 with 16x16x32") {
      auto hardware = make_hardware(gpu_arch);
      origami::dim3_t mt{16, 16, 64};
      origami::dim3_t mi{16, 16, 32};
      auto num_instructions = origami::compute_number_matrix_instructions(mt, mi);
      REQUIRE(num_instructions == 2);  // 1 * 1 * 2
    }
  }
}

TEST_CASE("GEMM: compute_mt_compute_latency", "[gemm]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - transA=T transB=N") {
      auto hardware = make_hardware(gpu_arch);
      auto problem =
          make_problem(4096, 4096, 1024, origami::transpose_t::T, origami::transpose_t::N);
      auto config = make_config(128, 128, 64, 32, 32, 8, false, 1);

      auto latency = origami::compute_mt_compute_latency(problem, hardware, config);
      REQUIRE(latency == 4096);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - transA=N transB=T") {
      auto hardware = make_hardware(gpu_arch);
      auto problem =
          make_problem(4096, 4096, 1024, origami::transpose_t::N, origami::transpose_t::T);
      auto config = make_config(128, 128, 64, 32, 32, 8, false, 1);

      auto latency = origami::compute_mt_compute_latency(problem, hardware, config);
      REQUIRE(latency == 4096);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - transA=N transB=N") {
      auto hardware = make_hardware(gpu_arch);
      auto problem =
          make_problem(4096, 4096, 1024, origami::transpose_t::N, origami::transpose_t::N);
      auto config = make_config(128, 128, 64, 32, 32, 8, false, 1);

      auto latency = origami::compute_mt_compute_latency(problem, hardware, config);
      REQUIRE(latency == 4096);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - transA=T transB=T") {
      auto hardware = make_hardware(gpu_arch);
      auto problem =
          make_problem(4096, 4096, 1024, origami::transpose_t::T, origami::transpose_t::T);
      auto config = make_config(128, 128, 64, 32, 32, 8, false, 1);

      auto latency = origami::compute_mt_compute_latency(problem, hardware, config);
      REQUIRE(latency == 4096);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - different MT_K") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(4096, 4096, 1024);
      auto config   = make_config(128, 128, 32, 32, 32, 8, false, 1);

      auto latency = origami::compute_mt_compute_latency(problem, hardware, config);
      REQUIRE(latency == 2048);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - larger tile") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(4096, 4096, 1024);
      auto config   = make_config(224, 224, 64, 32, 32, 8, false, 1);

      auto latency = origami::compute_mt_compute_latency(problem, hardware, config);
      REQUIRE(latency > 12543);
    }
  }
}

TEST_CASE("GEMM: compute_memory_latency", "[gemm]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - verify smaller tiles have lower latency") {
      auto hardware = make_hardware(gpu_arch);
      auto problem =
          make_problem(4096, 4096, 1024, origami::transpose_t::T, origami::transpose_t::N, 1);
      auto config_small = make_config(128, 128, 64, 32, 32, 8, false, 8);
      auto config_large = make_config(256, 256, 128, 32, 32, 8, false, 8);

      auto mem_latency_small =
          origami::compute_memory_latency(problem, hardware, config_small, 304, 2);
      auto mem_latency_large =
          origami::compute_memory_latency(problem, hardware, config_large, 304, 2);

      REQUIRE(mem_latency_small < mem_latency_large);
    }
  }
}

TEST_CASE("GEMM: compute_tile_latency", "[gemm]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - verify larger tiles have higher latency") {
      auto hardware = make_hardware(gpu_arch);
      auto problem =
          make_problem(4096, 4096, 1024, origami::transpose_t::T, origami::transpose_t::N, 2);
      auto config_small = make_config(128, 128, 64, 32, 32, 8, false, 6);
      auto config_large = make_config(256, 256, 128, 32, 32, 8, false, 6);

      auto tile_latency_small =
          origami::compute_tile_latency(problem, hardware, config_small, 304, 3);
      auto tile_latency_large =
          origami::compute_tile_latency(problem, hardware, config_large, 304, 3);

      REQUIRE(tile_latency_large > tile_latency_small);
    }
  }
}

TEST_CASE("GEMM: compute_timestep_latency", "[gemm]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - timestep latency equals tile latency") {
      auto hardware = make_hardware(gpu_arch);
      auto problem =
          make_problem(4096, 4096, 1024, origami::transpose_t::T, origami::transpose_t::N, 2);
      auto config = make_config(128, 128, 64, 32, 32, 8, false, 8);

      auto tile_latency     = origami::compute_tile_latency(problem, hardware, config, 304, 4);
      auto timestep_latency = origami::compute_timestep_latency(problem, hardware, config, 304, 4);

      REQUIRE(timestep_latency == Approx(tile_latency));
    }
  }
}

TEST_CASE("GEMM: compute_total_latency", "[gemm]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - smaller tiles have lower total latency") {
      auto hardware = make_hardware(gpu_arch);
      auto problem_small =
          make_problem(4096, 4096, 1024, origami::transpose_t::T, origami::transpose_t::N, 2);
      auto problem_large =
          make_problem(8192, 8192, 2048, origami::transpose_t::T, origami::transpose_t::N, 2);
      auto config = make_config(128, 128, 64, 32, 32, 8, false, 1);

      auto latency_small =
          origami::compute_total_latency(problem_small, hardware, config, hardware.N_CU);
      auto latency_large =
          origami::compute_total_latency(problem_large, hardware, config, hardware.N_CU);

      REQUIRE(latency_small < latency_large);
    }
  }
}

TEST_CASE("GEMM: check_lds_capacity", "[gemm]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - 256x256x64 tile fits in LDS") {
      auto hardware = make_hardware(gpu_arch);
      origami::dim3_t mt{256, 256, 64};

      auto fits = origami::check_lds_capacity(
          hardware, mt, origami::data_type_t::BFloat16, origami::data_type_t::BFloat16);

      REQUIRE(fits == true);
    }
  }
}

TEST_CASE("GEMM: estimate_l2_hit", "[gemm]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - L2 hit rate in valid range") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(4096, 4096, 1024);
      auto config   = make_config(256, 256, 64, 32, 32, 8, false, 1);

      for (int wgm = 1; wgm < 1025; wgm++) {
        config.workgroup_mapping = wgm;
        auto l2_hit              = origami::estimate_l2_hit(problem, hardware, config, 3);
        REQUIRE(l2_hit > 0.0);
        REQUIRE(l2_hit < 1.0);
      }
    }
  }
}

TEST_CASE("GEMM: estimate_mall_hit", "[gemm]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - Mall hit rate is positive") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(4096, 4096, 1024);
      auto config   = make_config(256, 256, 64, 32, 32, 8, false, 1);

      for (int wgm = 1; wgm < 1025; wgm++) {
        config.workgroup_mapping = wgm;
        auto mall_hit            = origami::estimate_mall_hit(problem, hardware, config, 304, 8);
        REQUIRE(mall_hit > 0.0);
      }
    }
  }
}

// Unit tests
TEST_CASE("GEMM: calculate_work_utilization unit test", "[gemm]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - calculate_work_utilization unit test") {
      auto problem = make_problem(4096, 4096, 1024);
      auto config  = make_config(256, 256, 64, 32, 32, 8, false, 1);

      // Test 1: Test with perfect tile alignment (should return 1.0)
      auto result = origami::calculate_work_utilization(problem, config);
      REQUIRE(result == 1.0);

      // Test 2: Test with non-aligned dimensions
      auto problem_non_aligned = make_problem(4351, 3839, 959);
      auto result_non_aligned  = origami::calculate_work_utilization(problem_non_aligned, config);
      REQUIRE(result_non_aligned == Approx(0.998).epsilon(1e-3));

      // Test 3: Test with zero dimensions (should return 1.0)
      auto problem_zero_dimensions = make_problem(0, 3839, 959);
      auto result_zero_dimensions =
          origami::calculate_work_utilization(problem_zero_dimensions, config);
      REQUIRE(result_zero_dimensions == 1.0);

      // Test 4: Test with very small problems
      auto problem_very_small = make_problem(10, 20, 15);
      auto result_very_small  = origami::calculate_work_utilization(problem_very_small, config);
      REQUIRE(result_very_small == Approx(0.0007152).epsilon(1e-4));

      // Test 5: Test with very large problems
      auto problem_very_large = make_problem(409601, 409601, 4095);
      auto result_very_large  = origami::calculate_work_utilization(problem_very_large, config);
      REQUIRE(result_very_large == Approx(0.998).epsilon(1e-3));

      // Test 6: Test with skinny matrices
      auto problem_skinny = make_problem(128, 81920, 1024);  // Small M, Big N
      auto result_skinny  = origami::calculate_work_utilization(problem_skinny, config);
      REQUIRE(result_skinny == 0.5);

      problem_skinny = make_problem(81920, 128, 1024);  // Small N, Big M
      result_skinny  = origami::calculate_work_utilization(problem_skinny, config);
      REQUIRE(result_skinny == 0.5);
    }
  }
}

TEST_CASE("GEMM: calculate_output_utilization unit test", "[gemm]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - calculate_output_utilization unit test") {
      auto problem = make_problem(4096, 4096, 1024);
      auto config  = make_config(256, 256, 64, 32, 32, 8, false, 1);

      // Test 1: Test with perfect tile alignment (should return 1.0)
      auto result = origami::calculate_output_utilization(problem, config, 1UL);
      REQUIRE(result == 1.0);

      // Test 2: Test with non-aligned dimensions
      auto problem_non_aligned = make_problem(4351, 3839, 959);
      auto result_non_aligned =
          origami::calculate_output_utilization(problem_non_aligned, config, 1UL);
      REQUIRE(result_non_aligned == Approx(0.999).epsilon(1e-3));

      // Test 3: Test with vector_elems > 1
      auto result_with_vector_elems = origami::calculate_output_utilization(problem, config, 23UL);
      REQUIRE(result_with_vector_elems == Approx(1.010).epsilon(1e-3));

      // Test 4: Test with zero dimensions (should return 1.0)
      auto problem_zero_dimensions = make_problem(0, 3839, 959);
      auto result_zero_dimensions =
          origami::calculate_output_utilization(problem_zero_dimensions, config, 1UL);
      REQUIRE(result_zero_dimensions == 1.0);

      // Test 5: Test with different problem sizes
      auto problem_very_small = make_problem(10, 20, 15);
      auto result_very_small =
          origami::calculate_output_utilization(problem_very_small, config, 1UL);
      REQUIRE(result_very_small == Approx(0.003051).epsilon(1e-3));

      auto problem_very_large = make_problem(409601, 409601, 4095);
      auto result_very_large =
          origami::calculate_output_utilization(problem_very_large, config, 1UL);
      REQUIRE(result_very_large == Approx(0.998).epsilon(1e-3));

      auto problem_skinny = make_problem(64, 81920, 1024);  // Small M, Big N
      auto result_skinny  = origami::calculate_output_utilization(problem_skinny, config, 1UL);
      REQUIRE(result_skinny == 0.25);

      problem_skinny = make_problem(81920, 64, 1024);  // Small N, Big M
      result_skinny  = origami::calculate_output_utilization(problem_skinny, config, 1UL);
      REQUIRE(result_skinny == 0.25);
    }
  }
}

TEST_CASE("GEMM: compute_cu_occupancy unit test", "[gemm]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - compute_cu_occupancy unit test") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(4096, 4096, 1024);
      auto config   = make_config(256, 256, 64, 32, 32, 8, false, 1);

      if (gpu_arch == 942) {
        // Test 1: Test with split parameter provided
        auto result_with_split =
            origami::compute_cu_occupancy(problem,
                                          hardware,
                                          config,
                                          origami::grid_selection_t::k_split_aware,
                                          hardware.N_CU,
                                          1UL);
        REQUIRE(std::get<0>(result_with_split) == 256);
        REQUIRE(std::get<1>(result_with_split) == 256);
        REQUIRE(std::get<2>(result_with_split) == 1);
        REQUIRE(std::get<3>(result_with_split) == 1);

        // Test 2: Test without split (StreamK prediction)
        auto result_without_split = origami::compute_cu_occupancy(
            problem, hardware, config, origami::grid_selection_t::k_split_aware, 256UL, 0UL);
        REQUIRE(std::get<0>(result_without_split) == 256);
        REQUIRE(std::get<1>(result_without_split) == 256);
        REQUIRE(std::get<2>(result_without_split) == 1);
        REQUIRE(std::get<3>(result_without_split) == 1);

        // Test 3: Test with different grid_selection algorithms
        auto result_different_grid_selection =
            origami::compute_cu_occupancy(problem,
                                          hardware,
                                          config,
                                          origami::grid_selection_t::number_of_cus,
                                          hardware.N_CU,
                                          0UL);
        REQUIRE(std::get<0>(result_different_grid_selection) == 304);
        REQUIRE(std::get<1>(result_different_grid_selection) == 304);
        REQUIRE(std::get<2>(result_different_grid_selection) == 1);
        REQUIRE(std::get<3>(result_different_grid_selection) == 2);

        result_different_grid_selection =
            origami::compute_cu_occupancy(problem,
                                          hardware,
                                          config,
                                          origami::grid_selection_t::min_resources,
                                          hardware.N_CU,
                                          0UL);
        REQUIRE(std::get<0>(result_different_grid_selection) == 256);
        REQUIRE(std::get<1>(result_different_grid_selection) == 256);
        REQUIRE(std::get<2>(result_different_grid_selection) == 1);
        REQUIRE(std::get<3>(result_different_grid_selection) == 1);

        result_different_grid_selection = origami::compute_cu_occupancy(
            problem, hardware, config, origami::grid_selection_t::energy_aware, hardware.N_CU, 0UL);
        REQUIRE(std::get<0>(result_different_grid_selection) == 256);
        REQUIRE(std::get<1>(result_different_grid_selection) == 256);
        REQUIRE(std::get<2>(result_different_grid_selection) == 1);
        REQUIRE(std::get<3>(result_different_grid_selection) == 1);

        result_different_grid_selection =
            origami::compute_cu_occupancy(problem,
                                          hardware,
                                          config,
                                          origami::grid_selection_t::reduction_cost_aware,
                                          hardware.N_CU,
                                          0UL);
        REQUIRE(std::get<0>(result_different_grid_selection) == 256);
        REQUIRE(std::get<1>(result_different_grid_selection) == 256);
        REQUIRE(std::get<2>(result_different_grid_selection) == 1);
        REQUIRE(std::get<3>(result_different_grid_selection) == 1);

        result_different_grid_selection =
            origami::compute_cu_occupancy(problem,
                                          hardware,
                                          config,
                                          origami::grid_selection_t::data_parallel,
                                          hardware.N_CU,
                                          0UL);
        REQUIRE(std::get<0>(result_different_grid_selection) == 256);
        REQUIRE(std::get<1>(result_different_grid_selection) == 256);
        REQUIRE(std::get<2>(result_different_grid_selection) == 1);
        REQUIRE(std::get<3>(result_different_grid_selection) == 1);

        result_different_grid_selection = origami::compute_cu_occupancy(
            problem, hardware, config, origami::grid_selection_t::analytical, hardware.N_CU, 0UL);
        REQUIRE(std::get<0>(result_different_grid_selection) == 256);
        REQUIRE(std::get<1>(result_different_grid_selection) == 256);
        REQUIRE(std::get<2>(result_different_grid_selection) == 1);
        REQUIRE(std::get<3>(result_different_grid_selection) == 1);

        result_different_grid_selection =
            origami::compute_cu_occupancy(problem,
                                          hardware,
                                          config,
                                          origami::grid_selection_t::k_split_aware,
                                          hardware.N_CU,
                                          0UL);
        REQUIRE(std::get<0>(result_different_grid_selection) == 256);
        REQUIRE(std::get<1>(result_different_grid_selection) == 256);
        REQUIRE(std::get<2>(result_different_grid_selection) == 1);
        REQUIRE(std::get<3>(result_different_grid_selection) == 1);

        // Test 4: Test with max_cus parameter
        auto result_max_cus =
            origami::compute_cu_occupancy(problem,
                                          hardware,
                                          config,
                                          origami::grid_selection_t::k_split_aware,
                                          150,
                                          0UL);  // max_cus set to 150
        REQUIRE(std::get<0>(result_max_cus) == 150);
        REQUIRE(std::get<1>(result_max_cus) == 150);
        REQUIRE(std::get<2>(result_max_cus) == 1);
        REQUIRE(std::get<3>(result_max_cus) == 1);

        // Test 5: Test with multiple split parameter
        auto result_multiple_split_parameter =
            origami::compute_cu_occupancy(problem,
                                          hardware,
                                          config,
                                          origami::grid_selection_t::k_split_aware,
                                          hardware.N_CU,
                                          10UL);
        REQUIRE(std::get<0>(result_multiple_split_parameter) == 2560);
        REQUIRE(std::get<1>(result_multiple_split_parameter) == 304);
        REQUIRE(std::get<2>(result_multiple_split_parameter) == 9);
        REQUIRE(std::get<3>(result_multiple_split_parameter) == 10);

        result_multiple_split_parameter =
            origami::compute_cu_occupancy(problem,
                                          hardware,
                                          config,
                                          origami::grid_selection_t::k_split_aware,
                                          hardware.N_CU,
                                          100UL);
        REQUIRE(std::get<0>(result_multiple_split_parameter) == 25600);
        REQUIRE(std::get<1>(result_multiple_split_parameter) == 304);
        REQUIRE(std::get<2>(result_multiple_split_parameter) == 85);
        REQUIRE(std::get<3>(result_multiple_split_parameter) == 100);

        result_multiple_split_parameter =
            origami::compute_cu_occupancy(problem,
                                          hardware,
                                          config,
                                          origami::grid_selection_t::k_split_aware,
                                          hardware.N_CU,
                                          423UL);
        REQUIRE(std::get<0>(result_multiple_split_parameter) == 108288);
        REQUIRE(std::get<1>(result_multiple_split_parameter) == 304);
        REQUIRE(std::get<2>(result_multiple_split_parameter) == 357);
        REQUIRE(std::get<3>(result_multiple_split_parameter) == 423);
      } else if (gpu_arch == 950) {
        // Test 1: Test with split parameter provided
        auto result_with_split =
            origami::compute_cu_occupancy(problem,
                                          hardware,
                                          config,
                                          origami::grid_selection_t::k_split_aware,
                                          hardware.N_CU,
                                          1UL);
        REQUIRE(std::get<0>(result_with_split) == 256);
        REQUIRE(std::get<1>(result_with_split) == 256);
        REQUIRE(std::get<2>(result_with_split) == 1);
        REQUIRE(std::get<3>(result_with_split) == 1);

        // Test 2: Test without split (StreamK prediction)
        auto result_without_split = origami::compute_cu_occupancy(
            problem, hardware, config, origami::grid_selection_t::k_split_aware, 256UL, 0UL);
        REQUIRE(std::get<0>(result_without_split) == 256);
        REQUIRE(std::get<1>(result_without_split) == 256);
        REQUIRE(std::get<2>(result_without_split) == 1);
        REQUIRE(std::get<3>(result_without_split) == 1);

        // Test 3: Test with different grid_selection algorithms
        auto result_different_grid_selection =
            origami::compute_cu_occupancy(problem,
                                          hardware,
                                          config,
                                          origami::grid_selection_t::number_of_cus,
                                          hardware.N_CU,
                                          0UL);
        REQUIRE(std::get<0>(result_different_grid_selection) == 256);
        REQUIRE(std::get<1>(result_different_grid_selection) == 256);
        REQUIRE(std::get<2>(result_different_grid_selection) == 1);
        REQUIRE(std::get<3>(result_different_grid_selection) == 1);

        result_different_grid_selection =
            origami::compute_cu_occupancy(problem,
                                          hardware,
                                          config,
                                          origami::grid_selection_t::min_resources,
                                          hardware.N_CU,
                                          0UL);
        REQUIRE(std::get<0>(result_different_grid_selection) == 256);
        REQUIRE(std::get<1>(result_different_grid_selection) == 256);
        REQUIRE(std::get<2>(result_different_grid_selection) == 1);
        REQUIRE(std::get<3>(result_different_grid_selection) == 1);

        result_different_grid_selection = origami::compute_cu_occupancy(
            problem, hardware, config, origami::grid_selection_t::energy_aware, hardware.N_CU, 0UL);
        REQUIRE(std::get<0>(result_different_grid_selection) == 256);
        REQUIRE(std::get<1>(result_different_grid_selection) == 256);
        REQUIRE(std::get<2>(result_different_grid_selection) == 1);
        REQUIRE(std::get<3>(result_different_grid_selection) == 1);

        result_different_grid_selection =
            origami::compute_cu_occupancy(problem,
                                          hardware,
                                          config,
                                          origami::grid_selection_t::reduction_cost_aware,
                                          hardware.N_CU,
                                          0UL);
        REQUIRE(std::get<0>(result_different_grid_selection) == 256);
        REQUIRE(std::get<1>(result_different_grid_selection) == 256);
        REQUIRE(std::get<2>(result_different_grid_selection) == 1);
        REQUIRE(std::get<3>(result_different_grid_selection) == 1);

        result_different_grid_selection =
            origami::compute_cu_occupancy(problem,
                                          hardware,
                                          config,
                                          origami::grid_selection_t::data_parallel,
                                          hardware.N_CU,
                                          0UL);
        REQUIRE(std::get<0>(result_different_grid_selection) == 256);
        REQUIRE(std::get<1>(result_different_grid_selection) == 256);
        REQUIRE(std::get<2>(result_different_grid_selection) == 1);
        REQUIRE(std::get<3>(result_different_grid_selection) == 1);

        result_different_grid_selection = origami::compute_cu_occupancy(
            problem, hardware, config, origami::grid_selection_t::analytical, hardware.N_CU, 0UL);
        REQUIRE(std::get<0>(result_different_grid_selection) == 256);
        REQUIRE(std::get<1>(result_different_grid_selection) == 256);
        REQUIRE(std::get<2>(result_different_grid_selection) == 1);
        REQUIRE(std::get<3>(result_different_grid_selection) == 1);

        result_different_grid_selection =
            origami::compute_cu_occupancy(problem,
                                          hardware,
                                          config,
                                          origami::grid_selection_t::k_split_aware,
                                          hardware.N_CU,
                                          0UL);
        REQUIRE(std::get<0>(result_different_grid_selection) == 256);
        REQUIRE(std::get<1>(result_different_grid_selection) == 256);
        REQUIRE(std::get<2>(result_different_grid_selection) == 1);
        REQUIRE(std::get<3>(result_different_grid_selection) == 1);

        // Test 4: Test with max_cus parameter
        auto result_max_cus =
            origami::compute_cu_occupancy(problem,
                                          hardware,
                                          config,
                                          origami::grid_selection_t::k_split_aware,
                                          150,
                                          0UL);  // max_cus set to 150
        REQUIRE(std::get<0>(result_max_cus) == 150);
        REQUIRE(std::get<1>(result_max_cus) == 150);
        REQUIRE(std::get<2>(result_max_cus) == 1);
        REQUIRE(std::get<3>(result_max_cus) == 1);

        // Test 5: Test with multiple split parameter
        auto result_multiple_split_parameter =
            origami::compute_cu_occupancy(problem,
                                          hardware,
                                          config,
                                          origami::grid_selection_t::k_split_aware,
                                          hardware.N_CU,
                                          10UL);
        REQUIRE(std::get<0>(result_multiple_split_parameter) == 2560);
        REQUIRE(std::get<1>(result_multiple_split_parameter) == 256);
        REQUIRE(std::get<2>(result_multiple_split_parameter) == 10);
        REQUIRE(std::get<3>(result_multiple_split_parameter) == 10);

        result_multiple_split_parameter =
            origami::compute_cu_occupancy(problem,
                                          hardware,
                                          config,
                                          origami::grid_selection_t::k_split_aware,
                                          hardware.N_CU,
                                          100UL);
        REQUIRE(std::get<0>(result_multiple_split_parameter) == 25600);
        REQUIRE(std::get<1>(result_multiple_split_parameter) == 256);
        REQUIRE(std::get<2>(result_multiple_split_parameter) == 100);
        REQUIRE(std::get<3>(result_multiple_split_parameter) == 100);

        result_multiple_split_parameter =
            origami::compute_cu_occupancy(problem,
                                          hardware,
                                          config,
                                          origami::grid_selection_t::k_split_aware,
                                          hardware.N_CU,
                                          423UL);
        REQUIRE(std::get<0>(result_multiple_split_parameter) == 108288);
        REQUIRE(std::get<1>(result_multiple_split_parameter) == 256);
        REQUIRE(std::get<2>(result_multiple_split_parameter) == 423);
        REQUIRE(std::get<3>(result_multiple_split_parameter) == 423);
      }
      // Test 5: Verify logger metrics are set correctly (TODO)
      // Test 6: Test reduction_strategy selection (TODO)
    }
  }
}

TEST_CASE("GEMM: compute_mem_bw_from_occupancy unit test", "[gemm]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - compute_mem_bw_from_occupancy unit test") {
      auto hardware = make_hardware(gpu_arch);

      // Test 1: Test with various num_active_cus values
      auto result_various_num_active_cus =
          origami::compute_mem_bw_from_occupancy(hardware, hardware.N_CU);
      REQUIRE(result_various_num_active_cus == 1.0);

      // Test 2: Test with different mem_bw_per_wg_coefficients
      hardware.mem_bw_per_wg_coefficients = std::make_tuple(0, 0.008, 0);
      auto result_different_mem_bw_per_wg_coefficients =
          origami::compute_mem_bw_from_occupancy(hardware, hardware.N_CU);
      REQUIRE(result_different_mem_bw_per_wg_coefficients == 1.0);

      hardware.mem_bw_per_wg_coefficients = std::make_tuple(0, 0.17, 0);
      result_different_mem_bw_per_wg_coefficients =
          origami::compute_mem_bw_from_occupancy(hardware, hardware.N_CU);
      REQUIRE(result_different_mem_bw_per_wg_coefficients == 1.0);

      hardware.mem_bw_per_wg_coefficients = std::make_tuple(0, 0.22, 0);
      result_different_mem_bw_per_wg_coefficients =
          origami::compute_mem_bw_from_occupancy(hardware, hardware.N_CU);
      REQUIRE(result_different_mem_bw_per_wg_coefficients == 1.0);

      // Test 3: Test with values less than 1
      hardware.mem_bw_per_wg_coefficients = std::make_tuple(0.000001, 0.001, 0);
      auto result_value_less_than_one =
          origami::compute_mem_bw_from_occupancy(hardware, hardware.N_CU);
      if (gpu_arch == 942)
        REQUIRE(result_value_less_than_one == Approx(0.3964).epsilon(1e-3));
      else if (gpu_arch == 950)
        REQUIRE(result_value_less_than_one == Approx(0.32153).epsilon(1e-3));

      hardware.mem_bw_per_wg_coefficients = std::make_tuple(0.000002, 0.002, 0);
      result_value_less_than_one = origami::compute_mem_bw_from_occupancy(hardware, hardware.N_CU);
      if (gpu_arch == 942)
        REQUIRE(result_value_less_than_one == Approx(0.7928).epsilon(1e-3));
      else if (gpu_arch == 950)
        REQUIRE(result_value_less_than_one == Approx(0.64307).epsilon(1e-3));

      // Reset the value of mem_bw_per_wg_coefficients back
      if (gpu_arch == 942)
        hardware.mem_bw_per_wg_coefficients = std::make_tuple(0, 0.015, 0);
      else if (gpu_arch == 950)
        hardware.mem_bw_per_wg_coefficients = std::make_tuple(0, 0.008, 0);

      // Test 4: Verify calculation correctness (TODO)
    }
  }
}

TEST_CASE("GEMM: compute_l2_hit_rate_global unit test", "[gemm]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - compute_l2_hit_rate_global unit test") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(2047, 2047, 4096);
      auto config   = make_config(256, 256, 64, 32, 32, 8, false, 1);

      // Test 1: Test with various problem sizes
      auto result_various_problem_sizes = origami::compute_l2_hit_rate_global(
          problem, hardware, config, hardware.L2_capacity * 1024);
      REQUIRE(result_various_problem_sizes == 0.875);

      auto problem_small           = make_problem(331, 4077, 547);
      result_various_problem_sizes = origami::compute_l2_hit_rate_global(
          problem_small, hardware, config, hardware.L2_capacity * 1024);
      REQUIRE(result_various_problem_sizes == 0.71875);

      auto problem_large           = make_problem(8193, 4077, 7453);
      result_various_problem_sizes = origami::compute_l2_hit_rate_global(
          problem_large, hardware, config, hardware.L2_capacity * 1024);
      REQUIRE(result_various_problem_sizes == Approx(0.953).epsilon(1e-3));

      // Test 2: Test with different splitting factors (TODO)(is this a valid test case as this
      // function does not use splitting factors) Test 3: Test edge cases
      REQUIRE_THROWS_WITH(origami::compute_l2_hit_rate_global(problem, hardware, config, 0UL),
                          "L2 Capacity is zero");

      auto problem_zero = make_problem(0, 0, 7453);
      REQUIRE_THROWS_WITH(origami::compute_l2_hit_rate_global(
                              problem_zero, hardware, config, hardware.L2_capacity * 1024),
                          "estimate_l2_hit grid dimensions can not be zero");
    }
  }
}

TEST_CASE("GEMM: round_elements_to_128B unit test", "[gemm]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - round_elements_to_128B unit test") {
      // Test 1: Test with various element sizes
      auto result_various_element_sizes =
          origami::round_elements_to_128B(196, 32);  // element size in bits - 32
      REQUIRE(result_various_element_sizes == 224);

      result_various_element_sizes =
          origami::round_elements_to_128B(225, 16);  // element size in bits - 16
      REQUIRE(result_various_element_sizes == 256);

      result_various_element_sizes =
          origami::round_elements_to_128B(90, 8);  // element size in bits - 8
      REQUIRE(result_various_element_sizes == 128);

      // Test 2: Test alignment to 128-byte boundary (TODO) (Was already covered in the above
      // example, could be skipped) Test 3: Test edge cases
      auto result_edge_cases = origami::round_elements_to_128B(0, 32);
      REQUIRE(result_edge_cases == 0);

      result_edge_cases = origami::round_elements_to_128B(256, 0);
      REQUIRE(result_edge_cases == 256);
    }
  }
}

TEST_CASE("GEMM: compute_cvt_overhead unit test", "[gemm]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - compute_cvt_overhead unit test") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(2047, 2047, 4096);
      auto config   = make_config(256, 256, 64, 32, 32, 8, false, 1);

      // Test 1: Test with Float data type
      origami::problem_t problem_Float = {
          .size            = {4097, 8001, 4096},
          .batch           = 1,
          .a_transpose     = origami::transpose_t::N,
          .b_transpose     = origami::transpose_t::T,
          .a_dtype         = origami::data_type_t::Float,
          .b_dtype         = origami::data_type_t::Float,
          .mi_dtype        = origami::data_type_t::Float,
          .a_mx_block_size = 0,
          .b_mx_block_size = 0,
      };
      auto result_test_with_Float = origami::compute_cvt_overhead(problem_Float, hardware, config);
      REQUIRE(result_test_with_Float == 0.0);

      // Test 2: Test with XFloat32 as mi_dtype and BFloat16 as a_dtype and b_dtype
      origami::problem_t problem_XFloat32 = {
          .size            = {8097, 8001, 4096},
          .batch           = 1,
          .a_transpose     = origami::transpose_t::N,
          .b_transpose     = origami::transpose_t::T,
          .a_dtype         = origami::data_type_t::BFloat16,
          .b_dtype         = origami::data_type_t::BFloat16,
          .mi_dtype        = origami::data_type_t::XFloat32,
          .a_mx_block_size = 0,
          .b_mx_block_size = 0,
      };
      auto result_test_with_XFloat32 =
          origami::compute_cvt_overhead(problem_XFloat32, hardware, config);
      REQUIRE(result_test_with_XFloat32 == 0.0);

      // Test 3: Test conversion overhead calculation (TODO) (Need more clarification)
      // Test 4: Test with different tile sizes
      auto result_with_different_tile_sizes =
          origami::compute_cvt_overhead(problem, hardware, config);
      REQUIRE(result_with_different_tile_sizes == 0.0);

      config                           = make_config(128, 128, 64, 32, 32, 8, false, 1);
      result_with_different_tile_sizes = origami::compute_cvt_overhead(problem, hardware, config);
      REQUIRE(result_with_different_tile_sizes == 0.0);

      config                           = make_config(64, 64, 256, 32, 32, 8, false, 1);
      result_with_different_tile_sizes = origami::compute_cvt_overhead(problem, hardware, config);
      REQUIRE(result_with_different_tile_sizes == 0.0);
    }
  }
}

TEST_CASE("GEMM: arithmetic_intensity and emulated_tf32_arithmetic_intensity unit case test",
          "[gemm]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION(
        "gfx" << gpu_arch
              << " - arithmetic_intensity and emulated_tf32_arithmetic_intensity unit case test") {
      // Test 1: Test calculation correctness
      auto result = origami::arithmetic_intensity(2047, 2047, 4096, 2);
      REQUIRE(result == Approx(818.879).epsilon(1e-3));

      result = origami::emulated_tf32_arithmetic_intensity(2047, 2047, 4096, 2);
      REQUIRE(result == Approx(2456.639).epsilon(1e-3));

      // Test 2: Test with various m, n, k, bytes_per_element
      auto result_various_problem_size = origami::arithmetic_intensity(127, 4097, 8193, 2);
      REQUIRE(result_various_problem_size == Approx(121.356).epsilon(1e-3));

      result_various_problem_size = origami::arithmetic_intensity(4097, 4097, 257, 4);
      REQUIRE(result_various_problem_size == Approx(114.175).epsilon(1e-3));

      result_various_problem_size = origami::arithmetic_intensity(237, 4097, 8183, 4);
      REQUIRE(result_various_problem_size == Approx(109.034).epsilon(1e-3));

      result_various_problem_size = origami::emulated_tf32_arithmetic_intensity(127, 4097, 8193, 2);
      REQUIRE(result_various_problem_size == Approx(364.0709).epsilon(1e-3));

      result_various_problem_size = origami::emulated_tf32_arithmetic_intensity(4097, 4097, 257, 4);
      REQUIRE(result_various_problem_size == Approx(342.527).epsilon(1e-3));

      result_various_problem_size = origami::emulated_tf32_arithmetic_intensity(237, 4097, 8183, 4);
      REQUIRE(result_various_problem_size == Approx(327.104).epsilon(1e-3));

      // Test 3: Test edge cases (zero values)
      auto result_zero_value = origami::arithmetic_intensity(0, 0, 0, 4);
      REQUIRE(result_zero_value == 0.0);

      result_zero_value = origami::emulated_tf32_arithmetic_intensity(0, 0, 0, 2);
      REQUIRE(result_zero_value == 0.0);
    }
  }
}

TEST_CASE("GEMM: check_lds_capacity unit test", "[gemm]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - check_lds_capacity unit test") {
      auto hardware = make_hardware(gpu_arch);

      if (gpu_arch == 942) {
        // Test 1: Test with tiles that exceed LDS capacity
        auto result_tiles_exceed_LDS_capacity =
            origami::check_lds_capacity(hardware,
                                        {256, 256, 256},
                                        origami::data_type_t::BFloat16,
                                        origami::data_type_t::BFloat16);
        REQUIRE(result_tiles_exceed_LDS_capacity == false);

        result_tiles_exceed_LDS_capacity =
            origami::check_lds_capacity(hardware,
                                        {128, 128, 256},
                                        origami::data_type_t::BFloat16,
                                        origami::data_type_t::BFloat16);
        REQUIRE(result_tiles_exceed_LDS_capacity == false);

        result_tiles_exceed_LDS_capacity =
            origami::check_lds_capacity(hardware,
                                        {64, 128, 256},
                                        origami::data_type_t::BFloat16,
                                        origami::data_type_t::BFloat16);
        REQUIRE(result_tiles_exceed_LDS_capacity == false);

        // Test 2: Test with different data type combinations
        auto result_different_data_type = origami::check_lds_capacity(
            hardware, {64, 64, 256}, origami::data_type_t::Half, origami::data_type_t::Half);
        REQUIRE(result_different_data_type == true);

        result_different_data_type = origami::check_lds_capacity(
            hardware, {256, 256, 512}, origami::data_type_t::Double, origami::data_type_t::Double);
        REQUIRE(result_different_data_type == false);

        result_different_data_type = origami::check_lds_capacity(
            hardware, {128, 128, 64}, origami::data_type_t::Int8, origami::data_type_t::Int8);
        REQUIRE(result_different_data_type == true);

        result_different_data_type = origami::check_lds_capacity(
            hardware, {128, 128, 32}, origami::data_type_t::Int64, origami::data_type_t::Int64);
        REQUIRE(result_different_data_type == true);

        // Test 3: Test with Float (element_size = 16) (TODO: Need more clarification)

        // Test 4: Test edge cases (exactly at capacity, just over)
        auto result_exactly_at_capacity =
            origami::check_lds_capacity(hardware,
                                        {256, 256, 64},
                                        origami::data_type_t::BFloat16,
                                        origami::data_type_t::Float8);  // exactly at capacity
        REQUIRE(result_exactly_at_capacity == true);

        result_exactly_at_capacity =
            origami::check_lds_capacity(hardware,
                                        {192, 96, 128},
                                        origami::data_type_t::BFloat16,
                                        origami::data_type_t::BFloat16);  // just over
        REQUIRE(result_exactly_at_capacity == false);
      } else if (gpu_arch == 950) {
        // Test 1: Test with tiles that exceed LDS capacity
        auto result_tiles_exceed_LDS_capacity =
            origami::check_lds_capacity(hardware,
                                        {512, 512, 256},
                                        origami::data_type_t::BFloat16,
                                        origami::data_type_t::BFloat16);
        REQUIRE(result_tiles_exceed_LDS_capacity == false);

        result_tiles_exceed_LDS_capacity =
            origami::check_lds_capacity(hardware,
                                        {256, 256, 512},
                                        origami::data_type_t::BFloat16,
                                        origami::data_type_t::BFloat16);
        REQUIRE(result_tiles_exceed_LDS_capacity == false);

        result_tiles_exceed_LDS_capacity =
            origami::check_lds_capacity(hardware,
                                        {512, 128, 256},
                                        origami::data_type_t::BFloat16,
                                        origami::data_type_t::BFloat16);
        REQUIRE(result_tiles_exceed_LDS_capacity == false);

        // Test 2: Test with different data type combinations
        auto result_different_data_type = origami::check_lds_capacity(
            hardware, {64, 64, 256}, origami::data_type_t::Half, origami::data_type_t::Half);
        REQUIRE(result_different_data_type == true);

        result_different_data_type = origami::check_lds_capacity(
            hardware, {256, 256, 512}, origami::data_type_t::Double, origami::data_type_t::Double);
        REQUIRE(result_different_data_type == false);

        result_different_data_type = origami::check_lds_capacity(
            hardware, {256, 256, 64}, origami::data_type_t::Int8, origami::data_type_t::Int8);
        REQUIRE(result_different_data_type == true);

        result_different_data_type = origami::check_lds_capacity(
            hardware, {256, 256, 32}, origami::data_type_t::Int64, origami::data_type_t::Int64);
        REQUIRE(result_different_data_type == true);

        // Test 3: Test with Float (element_size = 16) (TODO: Need more clarification)

        // Test 4: Test edge cases (exactly at capacity, just over)
        auto result_exactly_at_capacity =
            origami::check_lds_capacity(hardware,
                                        {512, 128, 256},
                                        origami::data_type_t::Float8,
                                        origami::data_type_t::Float8);  // exactly at capacity
        REQUIRE(result_exactly_at_capacity == true);

        result_exactly_at_capacity =
            origami::check_lds_capacity(hardware,
                                        {512, 192, 256},
                                        origami::data_type_t::BFloat16,
                                        origami::data_type_t::BFloat16);  // just over
        REQUIRE(result_exactly_at_capacity == false);
      }
    }
  }
}

TEST_CASE("GEMM: estimate_l2_hit and  estimate_mall_hit unit test", "[gemm]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - estimate_l2_hit and  estimate_mall_hit unit test") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(2047, 2047, 4096);
      auto config   = make_config(256, 256, 64, 32, 32, 8, false, 1);

      // Test 1: Test with different workgroup_mapping values (TODO)

      // Test 2: Test with various splitting factors
      auto result_different_splitting_factors =
          origami::estimate_l2_hit(problem, hardware, config, 0);
      REQUIRE(result_different_splitting_factors == 0.0);

      result_different_splitting_factors = origami::estimate_l2_hit(problem, hardware, config, 1);
      REQUIRE(result_different_splitting_factors == 0.4375);

      result_different_splitting_factors = origami::estimate_l2_hit(problem, hardware, config, -1);
      REQUIRE(result_different_splitting_factors == 0.0);

      result_different_splitting_factors =
          origami::estimate_mall_hit(problem, hardware, config, hardware.N_CU, 0);
      REQUIRE(result_different_splitting_factors == 0.875);

      result_different_splitting_factors =
          origami::estimate_mall_hit(problem, hardware, config, 256, 1);
      REQUIRE(result_different_splitting_factors == 0.875);

      result_different_splitting_factors =
          origami::estimate_mall_hit(problem, hardware, config, 200, -1);
      REQUIRE(result_different_splitting_factors == 0.875);

      // Test 3: Test with different problem sizes and different config
      problem                             = make_problem(8193, 2047, 4096);
      config                              = make_config(128, 128, 128, 32, 32, 8, 1);
      auto result_different_problem_sizes = origami::estimate_l2_hit(problem, hardware, config, 1);
      if (gpu_arch == 942)
        REQUIRE(result_different_problem_sizes == Approx(0.4868).epsilon(1e-3));
      else if (gpu_arch == 950)
        REQUIRE(result_different_problem_sizes == Approx(0.484).epsilon(1e-3));

      problem                        = make_problem(8193, 4093, 1024);
      config                         = make_config(64, 128, 128, 32, 32, 8, 1);
      result_different_problem_sizes = origami::estimate_l2_hit(problem, hardware, config, 1);
      if (gpu_arch == 942)
        REQUIRE(result_different_problem_sizes == Approx(0.649).epsilon(1e-3));
      else if (gpu_arch == 950)
        REQUIRE(result_different_problem_sizes == Approx(0.6458).epsilon(1e-3));

      problem = make_problem(8193, 2047, 4096);
      config  = make_config(256, 128, 64, 32, 32, 8, 1);
      result_different_problem_sizes =
          origami::estimate_mall_hit(problem, hardware, config, hardware.N_CU, 1);
      if (gpu_arch == 942)
        REQUIRE(result_different_problem_sizes == Approx(0.923).epsilon(1e-3));
      else if (gpu_arch == 950)
        REQUIRE(result_different_problem_sizes == Approx(0.9065).epsilon(1e-3));

      problem = make_problem(8193, 4093, 1024);
      config  = make_config(128, 256, 128, 32, 32, 8, 1);
      result_different_problem_sizes =
          origami::estimate_mall_hit(problem, hardware, config, hardware.N_CU, 1);
      if (gpu_arch == 942)
        REQUIRE(result_different_problem_sizes == Approx(0.923).epsilon(1e-3));
      else if (gpu_arch == 950)
        REQUIRE(result_different_problem_sizes == Approx(0.906).epsilon(1e-3));

      // Test 4: Test edge cases (very small/large problems)
      problem                = make_problem(10, 11, 253);
      config                 = make_config(256, 256, 64, 32, 32, 8, 1);
      auto result_edge_cases = origami::estimate_l2_hit(problem, hardware, config, 1);
      REQUIRE(result_edge_cases == 0.0);

      problem           = make_problem(81930, 40930, 10240);
      result_edge_cases = origami::estimate_l2_hit(problem, hardware, config, 1);
      if (gpu_arch == 942)
        REQUIRE(result_edge_cases == Approx(0.4868).epsilon(1e-3));
      else if (gpu_arch == 950)
        REQUIRE(result_edge_cases == Approx(0.484).epsilon(1e-3));

      problem           = make_problem(10, 11, 253);
      result_edge_cases = origami::estimate_mall_hit(problem, hardware, config, hardware.N_CU, 1);
      REQUIRE(result_edge_cases == 0.0);

      problem           = make_problem(81930, 40930, 10240);
      result_edge_cases = origami::estimate_mall_hit(problem, hardware, config, hardware.N_CU, 1);
      REQUIRE(result_edge_cases == Approx(0.498).epsilon(1e-3));
    }
  }
}

TEST_CASE("Heuristics: Default parameters", "[heuristics]") {
  origami::heuristic_params_t defaults;

  // Check default weight values
  REQUIRE(defaults.weight_mem_l2 == 1.0);
  REQUIRE(defaults.weight_mem_mall == 1.0);
  REQUIRE(defaults.weight_mem_dram == 1.0);
  REQUIRE(defaults.weight_compute == 1.0);
  REQUIRE(defaults.weight_memory == 1.0);
  REQUIRE(defaults.weight_wg_setup == 1.0);
  REQUIRE(defaults.weight_prologue == 1.5);
  REQUIRE(defaults.weight_epilogue == 2.0);
  REQUIRE(defaults.weight_loop_overhead == 500.0);
  REQUIRE(defaults.weight_tile_total == 1.0);

  // Check default empirical constants
  REQUIRE(defaults.l2_min_hit_rate_default == 0.5);
  REQUIRE(defaults.main_memory_load_latency == 200.0);
  REQUIRE(defaults.occupancy_decay_base == 0.95);
  REQUIRE(defaults.k_split_reduction_overhead == 10000.0);
  REQUIRE(defaults.k_padding_penalty == 50000.0);

  // Check default main loop efficiency
  REQUIRE(defaults.main_loop_efficiency == 1.0);
}

TEST_CASE("Heuristics: Parameter merging", "[heuristics]") {
  origami::heuristic_params_t base;
  origami::heuristic_params_t override;

  // Set some non-default values in override
  override.weight_compute           = 2.0;
  override.weight_memory            = 3.0;
  override.main_memory_load_latency = 300.0;
  override.main_loop_efficiency     = 0.8;

  // Merge override into base
  base.merge_with(override);

  // Check that overridden values changed
  REQUIRE(base.weight_compute == 2.0);
  REQUIRE(base.weight_memory == 3.0);
  REQUIRE(base.main_memory_load_latency == 300.0);
  REQUIRE(base.main_loop_efficiency == 0.8);

  // Check that non-overridden values remain default
  REQUIRE(base.weight_mem_l2 == 1.0);
  REQUIRE(base.weight_prologue == 1.5);
  REQUIRE(base.l2_min_hit_rate_default == 0.5);
}

TEST_CASE("Heuristics: Key matching - exact match", "[heuristics]") {
  auto hardware = make_hardware(950);
  auto problem  = make_problem(1024, 1024, 1024);
  auto config   = make_config(256, 256, 64, 16, 16, 16);

  problem.mi_dtype    = origami::data_type_t::BFloat16;
  problem.a_transpose = origami::transpose_t::T;
  problem.b_transpose = origami::transpose_t::N;

  origami::heuristic_key_t key;
  key.arch        = origami::hardware_t::architecture_t::gfx950;
  key.mi_dtype    = origami::data_type_t::BFloat16;
  key.a_transpose = origami::transpose_t::T;
  key.b_transpose = origami::transpose_t::N;
  key.mt_m        = 256;
  key.mt_n        = 256;
  key.mt_k        = 64;

  REQUIRE(key.matches(problem, hardware, config) == true);
}

TEST_CASE("Heuristics: Key matching - wildcard", "[heuristics]") {
  auto hardware = make_hardware(950);
  auto problem  = make_problem(1024, 1024, 1024);
  auto config   = make_config(256, 256, 64, 16, 16, 16);

  problem.mi_dtype    = origami::data_type_t::BFloat16;
  problem.a_transpose = origami::transpose_t::N;
  problem.b_transpose = origami::transpose_t::T;

  origami::heuristic_key_t key;
  key.arch     = origami::hardware_t::architecture_t::gfx950;
  key.mi_dtype = origami::data_type_t::BFloat16;
  // a_transpose, b_transpose, and tile sizes are wildcards (not set)

  REQUIRE(key.matches(problem, hardware, config) == true);
}

TEST_CASE("Heuristics: Key matching - mismatch", "[heuristics]") {
  auto hardware = make_hardware(950);
  auto problem  = make_problem(1024, 1024, 1024);
  auto config   = make_config(256, 256, 64, 16, 16, 16);

  problem.mi_dtype    = origami::data_type_t::BFloat16;
  problem.a_transpose = origami::transpose_t::N;
  problem.b_transpose = origami::transpose_t::T;

  origami::heuristic_key_t key;
  key.arch        = origami::hardware_t::architecture_t::gfx950;
  key.mi_dtype    = origami::data_type_t::BFloat16;
  key.a_transpose = origami::transpose_t::T;  // Mismatch!
  key.b_transpose = origami::transpose_t::T;

  REQUIRE(key.matches(problem, hardware, config) == false);
}

TEST_CASE("Heuristics: Key matching - problem size ranges", "[heuristics]") {
  auto hardware = make_hardware(950);
  auto problem  = make_problem(1024, 1024, 4096);
  auto config   = make_config(256, 256, 64, 16, 16, 16);

  origami::heuristic_key_t key;
  key.min_k = 2048;
  key.max_k = 8192;

  REQUIRE(key.matches(problem, hardware, config) == true);

  // Test below minimum
  problem = make_problem(1024, 1024, 1024);
  REQUIRE(key.matches(problem, hardware, config) == false);

  // Test above maximum
  problem = make_problem(1024, 1024, 16384);
  REQUIRE(key.matches(problem, hardware, config) == false);
}

TEST_CASE("Heuristics: Key specificity", "[heuristics]") {
  origami::heuristic_key_t key1;
  key1.arch = origami::hardware_t::architecture_t::gfx950;
  REQUIRE(key1.specificity() == 1);

  origami::heuristic_key_t key2;
  key2.arch     = origami::hardware_t::architecture_t::gfx950;
  key2.mi_dtype = origami::data_type_t::BFloat16;
  key2.mt_m     = 256;
  key2.mt_n     = 256;
  key2.mt_k     = 64;
  REQUIRE(key2.specificity() == 5);

  // More specific key should have higher specificity
  REQUIRE(key2.specificity() > key1.specificity());
}

TEST_CASE("Heuristics: Database lookup - successful call", "[heuristics]") {
  auto& db = origami::heuristics_database_t::get_instance();

  auto hardware = make_hardware(942);
  auto problem  = make_problem(128, 128, 128);
  auto config   = make_config(64, 64, 32, 16, 16, 16);

  problem.a_dtype     = origami::data_type_t::Half;
  problem.b_dtype     = origami::data_type_t::Half;
  problem.mi_dtype    = origami::data_type_t::Half;
  problem.a_transpose = origami::transpose_t::N;
  problem.b_transpose = origami::transpose_t::N;

  // Test that lookup completes without error
  auto params = db.lookup(problem, hardware, config);

  // Verify that we got some parameters back
  REQUIRE(params.main_memory_load_latency > 0.0);
  REQUIRE(params.occupancy_decay_base > 0.0);
  REQUIRE(params.occupancy_decay_base <= 1.0);
}

TEST_CASE("Heuristics: Optimized kernel efficiency lookup", "[heuristics]") {
  auto& db = origami::heuristics_database_t::get_instance();

  auto hardware = make_hardware(950);
  auto problem  = make_problem(1024, 1024, 1024);
  auto config   = make_config(256, 256, 64, 16, 16, 16);

  problem.a_dtype                 = origami::data_type_t::BFloat16;
  problem.b_dtype                 = origami::data_type_t::BFloat16;
  problem.mi_dtype                = origami::data_type_t::BFloat16;
  problem.a_transpose             = origami::transpose_t::N;
  problem.b_transpose             = origami::transpose_t::T;
  config.hand_optimized_main_loop = true;

  auto params = db.lookup(problem, hardware, config);

  // Should find optimized kernel efficiency (1.0 / 1.15 ≈ 0.8696)
  REQUIRE(params.main_loop_efficiency == Approx(1.0 / 1.15).epsilon(1e-6));
}

TEST_CASE("Heuristics: Problematic tile configuration (64x32x32)", "[heuristics]") {
  auto& db = origami::heuristics_database_t::get_instance();

  auto hardware = make_hardware(950);
  auto problem  = make_problem(1024, 1024, 1024);
  auto config   = make_config(64, 32, 32, 16, 16, 16);

  problem.a_dtype     = origami::data_type_t::BFloat16;
  problem.b_dtype     = origami::data_type_t::BFloat16;
  problem.mi_dtype    = origami::data_type_t::BFloat16;
  problem.a_transpose = origami::transpose_t::N;
  problem.b_transpose = origami::transpose_t::N;

  auto params = db.lookup(problem, hardware, config);

  // Should have 10x penalty for this problematic configuration
  REQUIRE(params.weight_tile_total == 10.0);
}

TEST_CASE("Heuristics: TF32 emulation - memory bound", "[heuristics]") {
  auto& db = origami::heuristics_database_t::get_instance();

  auto hardware = make_hardware(950);
  // Small problem: arith intensity = (3*2*512*512*512) / ((512*512 + 512*512 + 512*512) * 4) = 256
  // < 1000
  auto problem = make_problem(512, 512, 512);
  auto config  = make_config(256, 256, 32, 16, 16, 16);

  problem.a_dtype     = origami::data_type_t::Float;
  problem.b_dtype     = origami::data_type_t::Float;
  problem.mi_dtype    = origami::data_type_t::XFloat32;
  problem.a_transpose = origami::transpose_t::N;
  problem.b_transpose = origami::transpose_t::T;

  auto params = db.lookup(problem, hardware, config);

  // Should have optimization for memory-bound TF32 (arith < 1000)
  REQUIRE(params.weight_tile_total == 0.6);
}

TEST_CASE("Heuristics: TF32 emulation - compute bound", "[heuristics]") {
  auto& db = origami::heuristics_database_t::get_instance();

  auto hardware = make_hardware(950);
  // Large problem: arith intensity = (3*2*2048*2048*2048) / ((3 * 2048*2048) * 4) = 2048 > 1000
  auto problem = make_problem(2048, 2048, 2048);
  auto config  = make_config(256, 256, 32, 16, 16, 16);

  problem.a_dtype     = origami::data_type_t::Float;
  problem.b_dtype     = origami::data_type_t::Float;
  problem.mi_dtype    = origami::data_type_t::XFloat32;
  problem.a_transpose = origami::transpose_t::N;
  problem.b_transpose = origami::transpose_t::T;

  auto params = db.lookup(problem, hardware, config);

  // Should have stronger optimization for compute-bound TF32 (arith >= 1000)
  REQUIRE(params.weight_tile_total == 0.4);
}

TEST_CASE("Heuristics: Helper functions - make_kernel_variant_key", "[heuristics]") {
  auto key = origami::make_hand_optimized_kernel_key(origami::hardware_t::architecture_t::gfx950,
                                                     origami::data_type_t::BFloat16,
                                                     origami::transpose_t::N,
                                                     origami::transpose_t::T,
                                                     256,
                                                     256,
                                                     64);

  REQUIRE(key.arch.has_value());
  REQUIRE(key.arch.value() == origami::hardware_t::architecture_t::gfx950);
  REQUIRE(key.mi_dtype.has_value());
  REQUIRE(key.mi_dtype.value() == origami::data_type_t::BFloat16);
  REQUIRE(key.a_transpose.has_value());
  REQUIRE(key.a_transpose.value() == origami::transpose_t::N);
  REQUIRE(key.b_transpose.has_value());
  REQUIRE(key.b_transpose.value() == origami::transpose_t::T);
  REQUIRE(key.mt_m.has_value());
  REQUIRE(key.mt_m.value() == 256);
  REQUIRE(key.mt_n.has_value());
  REQUIRE(key.mt_n.value() == 256);
  REQUIRE(key.mt_k.has_value());
  REQUIRE(key.mt_k.value() == 64);
  REQUIRE(key.hand_optimized_main_loop.has_value());
  REQUIRE(key.hand_optimized_main_loop.value() == true);
}

TEST_CASE("Heuristics: Helper functions - make_tile_key", "[heuristics]") {
  auto key = origami::make_tile_key(128, 128, 32);

  REQUIRE(key.mt_m.has_value());
  REQUIRE(key.mt_m.value() == 128);
  REQUIRE(key.mt_n.has_value());
  REQUIRE(key.mt_n.value() == 128);
  REQUIRE(key.mt_k.has_value());
  REQUIRE(key.mt_k.value() == 32);
  REQUIRE(!key.a_transpose.has_value());
  REQUIRE(!key.b_transpose.has_value());
}

TEST_CASE("Heuristics: Helper functions - make_arch_dtype_key", "[heuristics]") {
  auto key = origami::make_arch_dtype_key(origami::hardware_t::architecture_t::gfx950,
                                          origami::data_type_t::XFloat32);

  REQUIRE(key.arch.has_value());
  REQUIRE(key.arch.value() == origami::hardware_t::architecture_t::gfx950);
  REQUIRE(key.mi_dtype.has_value());
  REQUIRE(key.mi_dtype.value() == origami::data_type_t::XFloat32);
  REQUIRE(!key.a_transpose.has_value());
  REQUIRE(!key.mt_m.has_value());
}

TEST_CASE("Heuristics: get_heuristic_params integration", "[heuristics]") {
  auto hardware = make_hardware(950);
  auto problem  = make_problem(1024, 1024, 1024);
  auto config   = make_config(256, 256, 64, 16, 16, 16);

  problem.a_dtype                 = origami::data_type_t::BFloat16;
  problem.b_dtype                 = origami::data_type_t::BFloat16;
  problem.mi_dtype                = origami::data_type_t::BFloat16;
  problem.a_transpose             = origami::transpose_t::N;
  problem.b_transpose             = origami::transpose_t::T;
  config.hand_optimized_main_loop = true;

  // Test the main entry point function
  auto params = origami::get_heuristic_params(problem, hardware, config);

  // Should find optimized kernel efficiency
  REQUIRE(params.main_loop_efficiency == Approx(1.0 / 1.15).epsilon(1e-6));
}

TEST_CASE("Heuristics: Database add_entry and lookup", "[heuristics]") {
  auto& db = origami::heuristics_database_t::get_instance();

  // Create a very specific custom heuristic entry that won't conflict
  origami::heuristic_key_t key;
  key.arch     = origami::hardware_t::architecture_t::gfx942;
  key.mi_dtype = origami::data_type_t::Int8;  // Unusual dtype
  key.mt_m     = 777;                         // Unique tile size

  origami::heuristic_params_t params;
  params.weight_wg_setup = 7.77;  // Unique value

  db.add_entry(key, params);

  // Try to lookup this entry
  auto hardware    = make_hardware(942);
  auto problem     = make_problem(1024, 1024, 1024);
  auto config      = make_config(777, 128, 64, 16, 16, 16);
  problem.mi_dtype = origami::data_type_t::Int8;

  auto result = db.lookup(problem, hardware, config);

  // Should find our custom entry
  REQUIRE(result.weight_wg_setup == 7.77);
}

TEST_CASE("Heuristics: Hierarchical lookup (most specific wins)", "[heuristics]") {
  auto& db = origami::heuristics_database_t::get_instance();

  // Add a general rule
  origami::heuristic_key_t general_key;
  general_key.arch     = origami::hardware_t::architecture_t::gfx942;
  general_key.mi_dtype = origami::data_type_t::Float;  // Specific dtype to avoid conflicts

  origami::heuristic_params_t general_params;
  general_params.weight_epilogue = 3.33;  // Use a weight that's less likely to conflict

  db.add_entry(general_key, general_params);

  // Add a more specific rule
  origami::heuristic_key_t specific_key;
  specific_key.arch     = origami::hardware_t::architecture_t::gfx942;
  specific_key.mi_dtype = origami::data_type_t::Float;
  specific_key.mt_m     = 555;  // Unique value

  origami::heuristic_params_t specific_params;
  specific_params.weight_epilogue = 5.55;  // More specific value

  db.add_entry(specific_key, specific_params);

  // Lookup with matching specific case
  auto hardware    = make_hardware(942);
  auto problem     = make_problem(1024, 1024, 1024);
  auto config      = make_config(555, 128, 64, 16, 16, 16);
  problem.mi_dtype = origami::data_type_t::Float;

  auto result = db.lookup(problem, hardware, config);

  // Should use more specific rule
  REQUIRE(result.weight_epilogue == 5.55);
}

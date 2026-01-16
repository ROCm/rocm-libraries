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

      auto tile_latency = origami::compute_tile_latency(problem, hardware, config, 304, 4);
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

      // Test 4: Test with very large problems
      auto problem_very_large = make_problem(409601, 409601, 4095);
      auto result_very_large  = origami::calculate_work_utilization(problem_very_large, config);
      REQUIRE(result_very_large == Approx(0.998).epsilon(1e-3));
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

      // Test 3: Test with zero dimensions (should return 1.0)
      auto problem_zero_dimensions = make_problem(0, 3839, 959);
      auto result_zero_dimensions =
          origami::calculate_output_utilization(problem_zero_dimensions, config, 1UL);
      REQUIRE(result_zero_dimensions == 1.0);

      // Test 4: Test with very small problems
      auto problem_very_small = make_problem(10, 20, 15);
      auto result_very_small =
          origami::calculate_output_utilization(problem_very_small, config, 1UL);
      REQUIRE(result_very_small == Approx(0.003051).epsilon(1e-3));

      // Test 4: Test with very large problems
      auto problem_very_large = make_problem(409601, 409601, 4095);
      auto result_very_large =
          origami::calculate_output_utilization(problem_very_large, config, 1UL);
      REQUIRE(result_very_large == Approx(0.998).epsilon(1e-3));
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
      }
      if (gpu_arch == 950) {
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

      // Reset the value of mem_bw_per_wg_coefficients back
      hardware.mem_bw_per_wg_coefficients = std::make_tuple(0, 0.015, 0);
      // Test 3: Verify calculation correctness (TODO)
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

      // Test 1: Test with XFloat32 data type
      origami::problem_t problem_Xfloat32 = {
          .size            = {4097, 8001, 4096},
          .batch           = 1,
          .a_transpose     = origami::transpose_t::N,
          .b_transpose     = origami::transpose_t::T,
          .a_dtype         = origami::data_type_t::XFloat32,  // element_size_A = 32
          .b_dtype         = origami::data_type_t::XFloat32,
          .mi_dtype        = origami::data_type_t::XFloat32,
          .a_mx_block_size = 0,
          .b_mx_block_size = 0,
      };
      auto result_test_with_Xfloat32 =
          origami::compute_cvt_overhead(problem_Xfloat32, hardware, config);
      REQUIRE(result_test_with_Xfloat32 == 0.0);

      // Test 2: Test conversion overhead calculation (TODO) (Need more clarification)
      // Test 3: Test with different tile sizes
      auto result_with_different_tile_sizes =
          origami::compute_cvt_overhead(problem, hardware, config);
      REQUIRE(result_test_with_Xfloat32 == 0.0);

      config                           = make_config(128, 128, 64, 32, 32, 8, false, 1);
      result_with_different_tile_sizes = origami::compute_cvt_overhead(problem, hardware, config);
      REQUIRE(result_test_with_Xfloat32 == 0.0);

      config                           = make_config(64, 64, 256, 32, 32, 8, false, 1);
      result_with_different_tile_sizes = origami::compute_cvt_overhead(problem, hardware, config);
      REQUIRE(result_test_with_Xfloat32 == 0.0);
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
        auto result_different_data_type =
            origami::check_lds_capacity(hardware,
                                        {64, 64, 256},
                                        origami::data_type_t::BFloat16,
                                        origami::data_type_t::BFloat16);
        REQUIRE(result_different_data_type == true);

        result_different_data_type = origami::check_lds_capacity(hardware,
                                                                 {128, 128, 64},
                                                                 origami::data_type_t::XFloat32,
                                                                 origami::data_type_t::XFloat32);
        REQUIRE(result_different_data_type == true);

        result_different_data_type = origami::check_lds_capacity(
            hardware, {256, 256, 64}, origami::data_type_t::Float8, origami::data_type_t::Float8);
        REQUIRE(result_different_data_type == true);

        // Test 3: Test with XFloat32 (element_size = 16) (TODO: Need more clarification)

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
        auto result_different_data_type =
            origami::check_lds_capacity(hardware,
                                        {64, 64, 256},
                                        origami::data_type_t::BFloat16,
                                        origami::data_type_t::BFloat16);
        REQUIRE(result_different_data_type == true);

        result_different_data_type = origami::check_lds_capacity(hardware,
                                                                 {256, 256, 64},
                                                                 origami::data_type_t::XFloat32,
                                                                 origami::data_type_t::XFloat32);
        REQUIRE(result_different_data_type == true);

        result_different_data_type = origami::check_lds_capacity(
            hardware, {512, 512, 64}, origami::data_type_t::Float8, origami::data_type_t::Float8);
        REQUIRE(result_different_data_type == true);

        // Test 3: Test with XFloat32 (element_size = 16) (TODO: Need more clarification)

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

      // Test 3: Test with different problem sizes
      problem                             = make_problem(8193, 2047, 4096);
      auto result_different_problem_sizes = origami::estimate_l2_hit(problem, hardware, config, 1);
      REQUIRE(result_different_problem_sizes == Approx(0.4848).epsilon(1e-3));

      problem                        = make_problem(8193, 4093, 1024);
      result_different_problem_sizes = origami::estimate_l2_hit(problem, hardware, config, 1);
      if (gpu_arch == 942)
        REQUIRE(result_different_problem_sizes == Approx(0.7348).epsilon(1e-3));
      else if (gpu_arch == 950)
        REQUIRE(result_different_problem_sizes == Approx(0.484).epsilon(1e-3));

      problem = make_problem(8193, 2047, 4096);
      result_different_problem_sizes =
          origami::estimate_mall_hit(problem, hardware, config, hardware.N_CU, 1);
      REQUIRE(result_different_problem_sizes == Approx(0.922).epsilon(1e-3));

      problem = make_problem(8193, 4093, 1024);
      result_different_problem_sizes =
          origami::estimate_mall_hit(problem, hardware, config, hardware.N_CU, 1);
      if (gpu_arch == 942)
        REQUIRE(result_different_problem_sizes == Approx(0.9348).epsilon(1e-3));
      else if (gpu_arch == 950)
        REQUIRE(result_different_problem_sizes == Approx(0.922).epsilon(1e-3));

      // Test 4: Test edge cases (very small/large problems)
      problem                = make_problem(10, 11, 253);
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

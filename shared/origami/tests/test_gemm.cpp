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

      origami::context_t context_small(problem, hardware, config_small);
      origami::context_t context_large(problem, hardware, config_large);

      auto mem_latency_small =
          origami::compute_memory_latency(problem, hardware, config_small, context_small);
      auto mem_latency_large =
          origami::compute_memory_latency(problem, hardware, config_large, context_large);

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

      origami::context_t context_small(problem, hardware, config_small);
      origami::context_t context_large(problem, hardware, config_large);

      auto tile_latency_small =
          origami::compute_tile_latency(problem, hardware, config_small, context_small);
      auto tile_latency_large =
          origami::compute_tile_latency(problem, hardware, config_large, context_large);

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

      origami::context_t context(problem, hardware, config);

      auto tile_latency = origami::compute_tile_latency(problem, hardware, config, context);
      auto timestep_latency = origami::compute_timestep_latency(problem, hardware, config, context);

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
        origami::context_t context(problem, hardware, config);
        auto l2_hit = origami::estimate_l2_hit(problem, hardware, config, context);
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
        origami::context_t context(problem, hardware, config);
        auto mall_hit = origami::estimate_mall_hit(problem, hardware, config, context);
        REQUIRE(mall_hit > 0.0);
        REQUIRE(mall_hit < 1.0);
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

TEST_CASE("GEMM: compute_launch_parameters unit test", "[gemm]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - compute_launch_parameters unit test") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(4096, 4096, 1024);
      auto config   = make_config(256, 256, 64, 32, 32, 8, false, 1);

      // Test 1: Basic launch parameters computation
      // Returns: (reduction_t, num_wgs, num_active_cus, num_timesteps, split_factor)
      auto result = origami::compute_launch_parameters(
          problem, hardware, config, origami::grid_selection_t::k_split_aware, hardware.N_CU);
      REQUIRE(std::get<0>(result) != origami::reduction_t::parallel); // reduction_t
      REQUIRE(std::get<1>(result) == 256);                            // num_wgs
      REQUIRE(std::get<2>(result) == 256);                            // num_active_cus
      REQUIRE(std::get<3>(result) == 1);                              // num_timesteps
      REQUIRE(std::get<4>(result) == 1);                              // split_factor

      // Test 2: Test with different grid_selection algorithms
      auto result_number_of_cus = origami::compute_launch_parameters(
          problem, hardware, config, origami::grid_selection_t::number_of_cus, hardware.N_CU);
      REQUIRE(std::get<1>(result_number_of_cus) > 0);  // num_wgs
      REQUIRE(std::get<2>(result_number_of_cus) > 0);  // num_active_cus

      auto result_min_resources = origami::compute_launch_parameters(
          problem, hardware, config, origami::grid_selection_t::min_resources, hardware.N_CU);
      REQUIRE(std::get<1>(result_min_resources) == 256);  // num_wgs
      REQUIRE(std::get<2>(result_min_resources) == 256);  // num_active_cus

      auto result_energy_aware = origami::compute_launch_parameters(
          problem, hardware, config, origami::grid_selection_t::energy_aware, hardware.N_CU);
      REQUIRE(std::get<1>(result_energy_aware) == 256);  // num_wgs
      REQUIRE(std::get<2>(result_energy_aware) == 256);  // num_active_cus

      auto result_reduction_cost = origami::compute_launch_parameters(
          problem, hardware, config, origami::grid_selection_t::reduction_cost_aware, hardware.N_CU);
      REQUIRE(std::get<1>(result_reduction_cost) == 256);  // num_wgs
      REQUIRE(std::get<2>(result_reduction_cost) == 256);  // num_active_cus

      auto result_data_parallel = origami::compute_launch_parameters(
          problem, hardware, config, origami::grid_selection_t::data_parallel, hardware.N_CU);
      REQUIRE(std::get<1>(result_data_parallel) == 256);  // num_wgs
      REQUIRE(std::get<2>(result_data_parallel) == 256);  // num_active_cus

      auto result_analytical = origami::compute_launch_parameters(
          problem, hardware, config, origami::grid_selection_t::analytical, hardware.N_CU);
      REQUIRE(std::get<1>(result_analytical) == 256);  // num_wgs
      REQUIRE(std::get<2>(result_analytical) == 256);  // num_active_cus

      // Test 3: Test with max_cus parameter
      auto result_max_cus = origami::compute_launch_parameters(
          problem, hardware, config, origami::grid_selection_t::k_split_aware, 150);
      REQUIRE(std::get<1>(result_max_cus) == 150);  // num_wgs capped at max_cus
      REQUIRE(std::get<2>(result_max_cus) == 150);  // num_active_cus
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
      origami::context_t context(problem, hardware, config);
      auto result_various_problem_sizes =
          origami::compute_l2_hit_rate_global(problem, hardware, config, context);
      REQUIRE(result_various_problem_sizes >= 0.0);
      REQUIRE(result_various_problem_sizes <= 1.0);

      auto problem_small = make_problem(331, 4077, 547);
      origami::context_t context_small(problem_small, hardware, config);
      result_various_problem_sizes =
          origami::compute_l2_hit_rate_global(problem_small, hardware, config, context_small);
      REQUIRE(result_various_problem_sizes >= 0.0);
      REQUIRE(result_various_problem_sizes <= 1.0);

      auto problem_large = make_problem(8193, 4077, 7453);
      origami::context_t context_large(problem_large, hardware, config);
      result_various_problem_sizes =
          origami::compute_l2_hit_rate_global(problem_large, hardware, config, context_large);
      REQUIRE(result_various_problem_sizes >= 0.0);
      REQUIRE(result_various_problem_sizes <= 1.0);
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

TEST_CASE("GEMM: estimate_l2_hit and estimate_mall_hit unit test", "[gemm]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - estimate_l2_hit and  estimate_mall_hit unit test") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(2047, 2047, 4096);
      auto config   = make_config(256, 256, 64, 32, 32, 8, false, 1);

      // Test 1: Test L2 and MALL hit estimation with context
      origami::context_t context(problem, hardware, config);
      auto result_l2_hit = origami::estimate_l2_hit(problem, hardware, config, context);
      REQUIRE(result_l2_hit >= 0.0);
      REQUIRE(result_l2_hit <= 1.0);

      auto result_mall_hit = origami::estimate_mall_hit(problem, hardware, config, context);
      REQUIRE(result_mall_hit >= 0.0);
      REQUIRE(result_mall_hit <= 1.0);

      // Test 2: Test with different problem sizes and different config
      // Verify L2 hit rate is in valid range for various problem/config combinations
      problem = make_problem(8193, 2047, 4096);
      config  = make_config(128, 128, 128, 32, 32, 8, 1);
      origami::context_t context2(problem, hardware, config);
      auto result_different_problem_sizes =
          origami::estimate_l2_hit(problem, hardware, config, context2);
      REQUIRE(result_different_problem_sizes >= 0.0);
      REQUIRE(result_different_problem_sizes <= 1.0);

      problem = make_problem(8193, 4093, 1024);
      config  = make_config(64, 128, 128, 32, 32, 8, 1);
      origami::context_t context3(problem, hardware, config);
      result_different_problem_sizes = origami::estimate_l2_hit(problem, hardware, config, context3);
      REQUIRE(result_different_problem_sizes >= 0.0);
      REQUIRE(result_different_problem_sizes <= 1.0);

      problem = make_problem(8193, 2047, 4096);
      config  = make_config(256, 128, 64, 32, 32, 8, 1);
      origami::context_t context4(problem, hardware, config);
      result_different_problem_sizes =
          origami::estimate_mall_hit(problem, hardware, config, context4);
      REQUIRE(result_different_problem_sizes >= 0.0);
      REQUIRE(result_different_problem_sizes <= 1.0);

      problem = make_problem(8193, 4093, 1024);
      config  = make_config(128, 256, 128, 32, 32, 8, 1);
      origami::context_t context5(problem, hardware, config);
      result_different_problem_sizes =
          origami::estimate_mall_hit(problem, hardware, config, context5);
      REQUIRE(result_different_problem_sizes >= 0.0);
      REQUIRE(result_different_problem_sizes <= 1.0);

      // Test 3: Test edge cases (very small/large problems)
      problem = make_problem(10, 11, 253);
      config  = make_config(256, 256, 64, 32, 32, 8, 1);
      origami::context_t context_small(problem, hardware, config);
      auto result_edge_cases = origami::estimate_l2_hit(problem, hardware, config, context_small);
      REQUIRE(result_edge_cases >= 0.0);
      REQUIRE(result_edge_cases <= 1.0);

      problem = make_problem(81930, 40930, 10240);
      origami::context_t context_large(problem, hardware, config);
      result_edge_cases = origami::estimate_l2_hit(problem, hardware, config, context_large);
      REQUIRE(result_edge_cases >= 0.0);
      REQUIRE(result_edge_cases <= 1.0);

      problem = make_problem(10, 11, 253);
      origami::context_t context_small2(problem, hardware, config);
      result_edge_cases = origami::estimate_mall_hit(problem, hardware, config, context_small2);
      REQUIRE(result_edge_cases >= 0.0);
      REQUIRE(result_edge_cases <= 1.0);

      problem = make_problem(81930, 40930, 10240);
      origami::context_t context_large2(problem, hardware, config);
      result_edge_cases = origami::estimate_mall_hit(problem, hardware, config, context_large2);
      REQUIRE(result_edge_cases >= 0.0);
      REQUIRE(result_edge_cases <= 1.0);
    }
  }
}

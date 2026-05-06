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

      auto mem_latency_small = origami::compute_memory_latency(
          problem, hardware, config_small, origami::context_t(problem, hardware, config_small));
      auto mem_latency_large = origami::compute_memory_latency(
          problem, hardware, config_large, origami::context_t(problem, hardware, config_large));

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

      auto tile_latency_small = origami::compute_tile_latency(
          problem, hardware, config_small, origami::context_t(problem, hardware, config_small));
      auto tile_latency_large = origami::compute_tile_latency(
          problem, hardware, config_large, origami::context_t(problem, hardware, config_large));

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

      auto tile_latency = origami::compute_tile_latency(
          problem, hardware, config, origami::context_t(problem, hardware, config));
      auto timestep_latency = origami::compute_timestep_latency(
          problem, hardware, config, origami::context_t(problem, hardware, config));

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
        auto l2_hit              = origami::estimate_l2_hit(
            problem, hardware, config, origami::context_t(problem, hardware, config));
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
        auto mall_hit            = origami::estimate_mall_hit(
            problem, hardware, config, origami::context_t(problem, hardware, config));
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

TEST_CASE("GEMM: compute_launch_parameters unit test", "[gemm]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - compute_launch_parameters unit test") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(4096, 4096, 1024);
      auto config   = make_config(256, 256, 64, 32, 32, 8, false, 1);

      // Returns: (reduction_t, num_wgs, num_active_cus, num_timesteps, split_factor)
      auto [reduction, num_wgs, num_active_cus, num_timesteps, split_factor] =
          origami::compute_launch_parameters(
              problem, hardware, config, origami::grid_selection_t::k_split_aware, hardware.N_CU);
      REQUIRE(num_wgs > 0);
      REQUIRE(num_active_cus > 0);
      REQUIRE(num_timesteps >= 1);
      REQUIRE(split_factor >= 1);

      // Test with different grid selection algorithms
      auto result_analytical = origami::compute_launch_parameters(
          problem, hardware, config, origami::grid_selection_t::analytical, hardware.N_CU);
      REQUIRE(std::get<1>(result_analytical) > 0);

      // Test with max_cus parameter
      auto result_max_cus = origami::compute_launch_parameters(
          problem, hardware, config, origami::grid_selection_t::k_split_aware, 150);
      REQUIRE(std::get<2>(result_max_cus) <= 150);
    }
  }
}

TEST_CASE("GEMM: count_unique_range unit test", "[gemm]") {
  // Test 1: Single tile — range covers exactly one WG
  {
    origami::dim4_t grid{1, 4, 8, 1};  // k=1, m=4, n=8, b=1
    auto u = origami::count_unique_range(grid, 8, 0, 1);
    REQUIRE(u.m == 1);
    REQUIRE(u.n == 1);
    REQUIRE(u.k == 1);
    REQUIRE(u.b == 1);
  }

  // Test 2: Full grid coverage — all tiles
  {
    origami::dim4_t grid{1, 4, 8, 1};
    auto u = origami::count_unique_range(grid, 8, 0, 32);
    REQUIRE(u.m == 4);
    REQUIRE(u.n == 8);
    REQUIRE(u.k == 1);
    REQUIRE(u.b == 1);
  }

  // Test 3: Within one slab (wgm=4), partial M
  {
    origami::dim4_t grid{1, 8, 16, 1};  // k=1, m=8, n=16, b=1
    auto u = origami::count_unique_range(grid, 4, 0, 3);
    REQUIRE(u.m == 1);
    REQUIRE(u.n == 3);
    REQUIRE(u.k == 1);
  }

  // Test 4: Within one slab, spans multiple rows
  {
    origami::dim4_t grid{1, 8, 16, 1};
    // wgm=4: slab has 8*4=32 tiles. IDs 0..7 = row0 cols 0-3, row1 cols 0-3
    auto u = origami::count_unique_range(grid, 4, 0, 8);
    REQUIRE(u.m == 2);
    REQUIRE(u.n == 4);  // full slab width
  }

  // Test 5: Across two slabs
  {
    origami::dim4_t grid{1, 4, 8, 1};
    // wgm=4: slab0 has 4*4=16 tiles (cols 0-3), slab1 has 16 tiles (cols 4-7)
    auto u = origami::count_unique_range(grid, 4, 14, 4);
    REQUIRE(u.m >= 1);
    REQUIRE(u.n >= 2);  // spans across slab boundary
  }

  // Test 6: K-split dimension
  {
    origami::dim4_t grid{4, 2, 2, 1};  // k=4 splits, m=2, n=2, b=1
    // First 4 IDs = same MN tile, different K splits
    auto u = origami::count_unique_range(grid, 2, 0, 4);
    REQUIRE(u.m == 1);
    REQUIRE(u.n == 1);
    REQUIRE(u.k == 4);
  }

  // Test 7: K-split spanning multiple MN tiles -> all K covered
  {
    origami::dim4_t grid{4, 2, 2, 1};
    auto u = origami::count_unique_range(grid, 2, 0, 5);
    REQUIRE(u.k == 4);  // crossed MN boundary, all K touched
  }

  // Test 8: Batch dimension
  {
    origami::dim4_t grid{1, 4, 4, 3};  // k=1, m=4, n=4, b=3
    // 16 tiles per batch, 48 total. Range covering 2 batches:
    auto u = origami::count_unique_range(grid, 4, 0, 32);
    REQUIRE(u.b == 2);
    REQUIRE(u.m == 4);  // multiple batches -> full M
    REQUIRE(u.n == 4);  // multiple batches -> full N
  }

  // Test 9: wgm=1 (column-first, each slab is 1 column)
  {
    origami::dim4_t grid{1, 4, 8, 1};
    auto u = origami::count_unique_range(grid, 1, 0, 4);
    REQUIRE(u.m == 4);  // 4 rows in one column
    REQUIRE(u.n == 1);  // single column slab
  }
}

TEST_CASE("GEMM: wgm_to_grid unit test", "[gemm]") {
  // Test 1: Simple 2x4 grid, wgm=4 (single slab covers all N)
  {
    origami::dim4_t grid{1, 2, 4, 1};
    origami::workgroup_mapping_t wgm{0, 0, 4};
    // slab_width=4, tiles_per_slab=2*4=8. All 8 tiles in one slab.
    // offset 0: m=0,n=0  offset 1: m=0,n=1  offset 2: m=0,n=2  offset 3: m=0,n=3
    // offset 4: m=1,n=0  offset 5: m=1,n=1  offset 6: m=1,n=2  offset 7: m=1,n=3
    auto t0 = origami::wgm_to_grid(grid, wgm, 0);
    REQUIRE(t0.m == 0);
    REQUIRE(t0.n == 0);
    REQUIRE(t0.k == 0);
    REQUIRE(t0.b == 0);
    auto t3 = origami::wgm_to_grid(grid, wgm, 3);
    REQUIRE(t3.m == 0);
    REQUIRE(t3.n == 3);
    auto t4 = origami::wgm_to_grid(grid, wgm, 4);
    REQUIRE(t4.m == 1);
    REQUIRE(t4.n == 0);
    auto t7 = origami::wgm_to_grid(grid, wgm, 7);
    REQUIRE(t7.m == 1);
    REQUIRE(t7.n == 3);
  }

  // Test 2: Two slabs — 4x6 grid, wgm=3
  {
    origami::dim4_t grid{1, 4, 6, 1};
    origami::workgroup_mapping_t wgm{0, 0, 3};
    // slab0 (cols 0-2): 4*3=12 tiles, slab1 (cols 3-5): 12 tiles
    auto t0 = origami::wgm_to_grid(grid, wgm, 0);
    REQUIRE(t0.m == 0);
    REQUIRE(t0.n == 0);
    auto t2 = origami::wgm_to_grid(grid, wgm, 2);
    REQUIRE(t2.m == 0);
    REQUIRE(t2.n == 2);
    auto t3 = origami::wgm_to_grid(grid, wgm, 3);
    REQUIRE(t3.m == 1);
    REQUIRE(t3.n == 0);
    auto t11 = origami::wgm_to_grid(grid, wgm, 11);
    REQUIRE(t11.m == 3);
    REQUIRE(t11.n == 2);
    // slab1 starts at id=12
    auto t12 = origami::wgm_to_grid(grid, wgm, 12);
    REQUIRE(t12.m == 0);
    REQUIRE(t12.n == 3);
    auto t23 = origami::wgm_to_grid(grid, wgm, 23);
    REQUIRE(t23.m == 3);
    REQUIRE(t23.n == 5);
  }

  // Test 3: K-splits — each MN tile has k splits
  {
    origami::dim4_t grid{4, 2, 3, 1};  // k=4, m=2, n=3, b=1
    origami::workgroup_mapping_t wgm{0, 0, 3};
    // First MN tile (m=0,n=0) has ids 0..3 (k=0..3)
    auto t0 = origami::wgm_to_grid(grid, wgm, 0);
    REQUIRE(t0.m == 0);
    REQUIRE(t0.n == 0);
    REQUIRE(t0.k == 0);
    auto t3 = origami::wgm_to_grid(grid, wgm, 3);
    REQUIRE(t3.m == 0);
    REQUIRE(t3.n == 0);
    REQUIRE(t3.k == 3);
    // Second MN tile (m=0,n=1) starts at id=4
    auto t4 = origami::wgm_to_grid(grid, wgm, 4);
    REQUIRE(t4.m == 0);
    REQUIRE(t4.n == 1);
    REQUIRE(t4.k == 0);
  }

  // Test 4: Batch dimension
  {
    origami::dim4_t grid{1, 2, 2, 3};  // k=1, m=2, n=2, b=3
    origami::workgroup_mapping_t wgm{0, 0, 2};
    // 4 tiles per batch. batch 0: ids 0-3, batch 1: ids 4-7, batch 2: ids 8-11
    auto t0 = origami::wgm_to_grid(grid, wgm, 0);
    REQUIRE(t0.b == 0);
    REQUIRE(t0.m == 0);
    REQUIRE(t0.n == 0);
    auto t4 = origami::wgm_to_grid(grid, wgm, 4);
    REQUIRE(t4.b == 1);
    REQUIRE(t4.m == 0);
    REQUIRE(t4.n == 0);
    auto t11 = origami::wgm_to_grid(grid, wgm, 11);
    REQUIRE(t11.b == 2);
    REQUIRE(t11.m == 1);
    REQUIRE(t11.n == 1);
  }

  // Test 5: WGMXCC interleaving
  {
    origami::dim4_t grid{1, 4, 4, 1};               // 16 tiles total
    origami::workgroup_mapping_t wgm_xcc{0, 8, 4};  // wgmxcc=8
    origami::workgroup_mapping_t wgm_no{0, 0, 4};   // no xcc

    // With WGMXCC, consecutive IDs should map to different XCD groups.
    // ID 0 and ID 1 should land in different XCD regions.
    auto t0_xcc = origami::wgm_to_grid(grid, wgm_xcc, 0);
    auto t1_xcc = origami::wgm_to_grid(grid, wgm_xcc, 1);
    auto t0_no  = origami::wgm_to_grid(grid, wgm_no, 0);

    // Without WGMXCC, id 0 and 1 are adjacent; with WGMXCC they're spread apart
    REQUIRE(t0_xcc.m == t0_no.m);
    REQUIRE(t0_xcc.n == t0_no.n);
    // id=1 with xcc should NOT be the same as id=1 without xcc (it gets remapped)
    auto t1_no = origami::wgm_to_grid(grid, wgm_no, 1);
    bool same  = (t1_xcc.m == t1_no.m && t1_xcc.n == t1_no.n);
    REQUIRE_FALSE(same);
  }

  // Test 6: wgm=1 — each slab is one column, M varies fastest
  {
    origami::dim4_t grid{1, 4, 3, 1};
    origami::workgroup_mapping_t wgm{0, 0, 1};
    // slab_width=1, tiles_per_slab=4. Col 0: ids 0-3, col 1: ids 4-7, col 2: ids 8-11
    auto t0 = origami::wgm_to_grid(grid, wgm, 0);
    REQUIRE(t0.m == 0);
    REQUIRE(t0.n == 0);
    auto t3 = origami::wgm_to_grid(grid, wgm, 3);
    REQUIRE(t3.m == 3);
    REQUIRE(t3.n == 0);
    auto t4 = origami::wgm_to_grid(grid, wgm, 4);
    REQUIRE(t4.m == 0);
    REQUIRE(t4.n == 1);
  }

  // Test 7: Remainder slab — grid.n not divisible by wgm
  {
    origami::dim4_t grid{1, 3, 5, 1};  // 5 cols, wgm=2 -> 2 full slabs + remainder of 1
    origami::workgroup_mapping_t wgm{0, 0, 2};
    // slab0: cols 0,1 (6 tiles), slab1: cols 2,3 (6 tiles), remainder: col 4 (3 tiles)
    auto t12 = origami::wgm_to_grid(grid, wgm, 12);
    REQUIRE(t12.m == 0);
    REQUIRE(t12.n == 4);
    auto t14 = origami::wgm_to_grid(grid, wgm, 14);
    REQUIRE(t14.m == 2);
    REQUIRE(t14.n == 4);
  }

  // Test 8: Brute-force 'bijection' test — every id maps to a unique (m,n,k,b) and back
  {
    origami::dim4_t grid{2, 3, 4, 2};
    origami::workgroup_mapping_t wgm{0, 0, 2};
    size_t total = grid.total();

    std::set<std::tuple<size_t, size_t, size_t, size_t>> seen;
    for (size_t id = 0; id < total; ++id) {
      auto t = origami::wgm_to_grid(grid, wgm, id);
      REQUIRE(t.m < grid.m);
      REQUIRE(t.n < grid.n);
      REQUIRE(t.k < grid.k);
      REQUIRE(t.b < grid.b);
      auto key = std::make_tuple(t.m, t.n, t.k, t.b);
      REQUIRE(seen.count(key) == 0);
      seen.insert(key);
    }
    REQUIRE(seen.size() == total);
  }
}

TEST_CASE("GEMM: compute_mall_tiles unit test", "[gemm]") {
  // Test 1: Basic case — all CUs fit in one slab
  {
    auto [m, n] = origami::compute_mall_tiles(16, 32, 256, 6);
    REQUIRE(m >= 1);
    REQUIRE(m <= 16);
    REQUIRE(n >= 1);
    REQUIRE(n <= 32);
  }

  // Test 2: wgm_value == 1 → slab width 1, all CUs in M dimension
  {
    auto [m, n] = origami::compute_mall_tiles(128, 128, 256, 1);
    REQUIRE(m <= 128);
    REQUIRE(n >= 1);
  }

  // Test 3: Large wgm → slab covers entire N
  {
    auto [m, n] = origami::compute_mall_tiles(16, 8, 64, 16);
    REQUIRE(n <= 8);
    REQUIRE(m <= 16);
  }

  // Test 4: active_cus > grid tiles → wrap around
  {
    auto [m, n] = origami::compute_mall_tiles(4, 4, 256, 4);
    REQUIRE(m == 4);
    REQUIRE(n == 4);
  }

  // Test 5: Single tile grid
  {
    auto [m, n] = origami::compute_mall_tiles(1, 1, 256, 6);
    REQUIRE(m == 1);
    REQUIRE(n == 1);
  }
}

TEST_CASE("GEMM: compute_l2_tiles unit test", "[gemm]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - compute_l2_tiles") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(4096, 4096, 1024);
      auto config   = make_config(256, 256, 64, 32, 32, 8, false, 1);

      size_t grid_m = (4096 + 255) / 256;
      size_t grid_n = (4096 + 255) / 256;

      // Test 1: Basic — result should be in valid range
      auto [m, n] = origami::compute_l2_tiles(problem, hardware, config, grid_m, grid_n, 256, 1, 6);
      REQUIRE(m >= 1);
      REQUIRE(m <= grid_m);
      REQUIRE(n >= 1);
      REQUIRE(n <= grid_n);

      // Test 2: Larger WGM → wider slab → more N reuse
      auto [m1, n1] =
          origami::compute_l2_tiles(problem, hardware, config, grid_m, grid_n, 256, 1, 1);
      auto [m2, n2] =
          origami::compute_l2_tiles(problem, hardware, config, grid_m, grid_n, 256, 1, 8);
      // Wider slab should not have fewer N columns
      REQUIRE(n2 >= n1);

      // Test 3: With splitting factor > 1 → fewer effective CUs per XCD → smaller tile
      auto [m3, n3] =
          origami::compute_l2_tiles(problem, hardware, config, grid_m, grid_n, 256, 4, 6);
      REQUIRE(m3 <= m);  // Splitting should not increase tile size

      // Test 4: Single tile grid
      auto [m4, n4] = origami::compute_l2_tiles(problem, hardware, config, 1, 1, 256, 1, 6);
      REQUIRE(m4 == 1);
      REQUIRE(n4 == 1);
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

TEST_CASE("GEMM: estimate_l2_hit and estimate_mall_hit unit test", "[gemm]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch
                          << " - estimate_l2_hit and estimate_mall_hit with context_t") {
      auto hardware = make_hardware(gpu_arch);

      auto check_rates = [&](origami::problem_t& p, origami::config_t& c) {
        origami::context_t ctx(p, hardware, c);
        auto l2   = origami::estimate_l2_hit(p, hardware, c, ctx);
        auto mall = origami::estimate_mall_hit(p, hardware, c, ctx);
        REQUIRE(l2 >= 0.0);
        REQUIRE(l2 <= 1.0);
        REQUIRE(mall >= 0.0);
        REQUIRE(mall <= 1.0);
      };

      // Various problem/config combinations
      auto p1 = make_problem(2047, 2047, 4096);
      auto c1 = make_config(256, 256, 64, 32, 32, 8, false, 1);
      check_rates(p1, c1);

      auto p2 = make_problem(8193, 2047, 4096);
      auto c2 = make_config(128, 128, 128, 32, 32, 8, 1);
      check_rates(p2, c2);

      auto p3 = make_problem(8193, 4093, 1024);
      auto c3 = make_config(64, 128, 128, 32, 32, 8, 1);
      check_rates(p3, c3);

      // Edge cases
      auto p_small = make_problem(10, 11, 253);
      auto c_edge  = make_config(256, 256, 64, 32, 32, 8, 1);
      check_rates(p_small, c_edge);

      auto p_large = make_problem(81930, 40930, 10240);
      check_rates(p_large, c_edge);
    }
  }
}

TEST_CASE("GEMM: count_unique_tiles unit test", "[gemm]") {
  const size_t N_CU    = 256;
  const size_t num_xcd = 8;

  SECTION("grid_b=64, grid_k=1, wgmxcc=1 — round-robin strided across batches") {
    // Grid: 2×2×1×64, wgm=1, wgmxcc=1 (no wgmxcc)
    // stride=8, mnk=4, each strided tile lands in a different batch
    auto u = origami::count_unique_tiles({1, 2, 2, 64}, {0, 1, 1}, N_CU, num_xcd, 0, 0);
    CHECK(u.k == 1);
    CHECK(u.m == 1);
    CHECK(u.n == 1);
    CHECK(u.b == 32);
  }

  SECTION("grid_b=64, grid_k=1, wgmxcc=8 — contiguous block spans all mn per batch") {
    // Grid: 2×2×1×64, wgm=1, wgmxcc=8
    // Each XCD gets 32 contiguous tiles -> 4 mn × 8 batches
    auto u = origami::count_unique_tiles({1, 2, 2, 64}, {0, 8, 1}, N_CU, num_xcd, 0, 0);
    CHECK(u.k == 1);
    CHECK(u.m == 2);
    CHECK(u.n == 2);
    CHECK(u.b == 8);
  }

  SECTION("grid_k=64, grid_b=1, wgmxcc=1 — round-robin strided across k-splits") {
    // Grid: 2×2×64×1, wgm=1, wgmxcc=1
    // stride=8 cycles through k: unique_k = 64/gcd(8,64) = 8
    auto u = origami::count_unique_tiles({64, 2, 2, 1}, {0, 1, 1}, N_CU, num_xcd, 0, 0);
    CHECK(u.k == 8);
    CHECK(u.m == 2);
    CHECK(u.n == 2);
    CHECK(u.b == 1);
  }

  SECTION("grid_k=64, grid_b=1, wgmxcc=8 — contiguous block within one mn tile") {
    // Grid: 2×2×64×1, wgm=1, wgmxcc=8
    // Each XCD gets 32 contiguous tiles -> 32 k-splits in mn_id=0
    auto u = origami::count_unique_tiles({64, 2, 2, 1}, {0, 8, 1}, N_CU, num_xcd, 0, 0);
    CHECK(u.k == 32);
    CHECK(u.m == 1);
    CHECK(u.n == 1);
    CHECK(u.b == 1);
  }

  SECTION("16×16 grid, wgmxcc=8, wgm=4 — contiguous block in first WGM slab") {
    // Grid: 16×16×1×1, wgm=4, wgmxcc=8
    // 32 contiguous tiles → 8 m-rows × 4 n-columns (one slab)
    auto u = origami::count_unique_tiles({1, 16, 16, 1}, {0, 8, 4}, N_CU, num_xcd, 0, 0);
    CHECK(u.k == 1);
    CHECK(u.m == 8);
    CHECK(u.n == 4);
    CHECK(u.b == 1);
  }

  SECTION("8×8 grid with k=4, wgmxcc=8, wgm=2 — mixed k and mn") {
    // Grid: 8×8×4×1, wgm=2, wgmxcc=8
    // 32 contiguous tiles → 4 k-splits × 4 m-rows × 2 n-columns
    auto u = origami::count_unique_tiles({4, 8, 8, 1}, {0, 8, 2}, N_CU, num_xcd, 0, 0);
    CHECK(u.k == 4);
    CHECK(u.m == 4);
    CHECK(u.n == 2);
    CHECK(u.b == 1);
  }

  SECTION("All XCDs report the same unique counts") {
    // With wgmxcc=8, all XCDs should see the same tile structure
    for (size_t xcd = 0; xcd < num_xcd; ++xcd) {
      auto u = origami::count_unique_tiles({1, 16, 16, 1}, {0, 8, 4}, N_CU, num_xcd, xcd, 0);
      CHECK(u.k == 1);
      CHECK(u.m == 8);
      CHECK(u.n == 4);
      CHECK(u.b == 1);
    }
  }

  SECTION("Zero/degenerate inputs return zeros") {
    CHECK(origami::count_unique_tiles({1, 4, 4, 1}, {0, 8, 4}, 0, 8, 0, 0).m == 0);
    CHECK(origami::count_unique_tiles({1, 4, 4, 1}, {0, 8, 4}, 256, 0, 0, 0).m == 0);
    CHECK(origami::count_unique_tiles({1, 0, 4, 1}, {0, 8, 4}, 256, 8, 0, 0).m == 0);
    CHECK(origami::count_unique_tiles({0, 4, 4, 1}, {0, 8, 4}, 256, 8, 0, 0).k == 0);
  }

  SECTION("Timestep beyond available tiles returns zeros") {
    // 16 tiles total, 256 CUs -> 1 timestep. Timestep 1 should be empty.
    origami::dim4_t grid{1, 4, 4, 1};
    auto u = origami::count_unique_tiles(grid, {0, 1, 4}, N_CU, num_xcd, 0, 1);
    CHECK(u.m == 0);
    CHECK(u.n == 0);
    CHECK(u.k == 0);
    CHECK(u.b == 0);
  }

  SECTION("Single tile grid — all XCDs see at most 1 tile") {
    origami::dim4_t grid{1, 1, 1, 1};
    auto u0 = origami::count_unique_tiles(grid, {0, 1, 1}, N_CU, num_xcd, 0, 0);
    CHECK(u0.m == 1);
    CHECK(u0.n == 1);
    CHECK(u0.k == 1);
    CHECK(u0.b == 1);
    // XCD 1 should get nothing (only 1 tile, XCD 0 gets it)
    auto u1 = origami::count_unique_tiles(grid, {0, 1, 1}, N_CU, num_xcd, 1, 0);
    CHECK(u1.m == 0);
  }

  SECTION("Round-robin: large grid, all MN tiles covered per XCD") {
    // 32x32 grid = 1024 tiles, 256 CUs, 8 XCDs -> 32 tiles/XCD.
    // stride=8 across 1024 MN tiles -> each XCD sees many M and N values.
    origami::dim4_t grid{1, 32, 32, 1};
    auto u = origami::count_unique_tiles(grid, {0, 1, 4}, N_CU, num_xcd, 0, 0);
    CHECK(u.k == 1);
    CHECK(u.m >= 1);
    CHECK(u.m <= 32);
    CHECK(u.n >= 1);
    CHECK(u.n <= 32);
    CHECK(u.b == 1);
  }

  SECTION("Round-robin: stride aligns with grid.k") {
    // grid.k=8, stride=8 -> gcd=8, unique_k = 8/8 = 1.
    // Each XCD sees a single K-split but many MN tiles.
    origami::dim4_t grid{8, 4, 4, 1};
    auto u = origami::count_unique_tiles(grid, {0, 1, 4}, N_CU, num_xcd, 0, 0);
    CHECK(u.k == 1);
    CHECK(u.m >= 1);
    CHECK(u.n >= 1);
  }

  SECTION("Round-robin: stride coprime with grid.k") {
    // grid.k=3, stride=8 -> gcd(8,3)=1, unique_k = 3/1 = 3 (all K-splits).
    origami::dim4_t grid{3, 4, 4, 1};
    auto u = origami::count_unique_tiles(grid, {0, 1, 4}, N_CU, num_xcd, 0, 0);
    CHECK(u.k == 3);
  }

  SECTION("WGMXCC: last XCD gets correct tiles") {
    // 256 tiles, 8 XCDs -> 32 per XCD. Last XCD starts at 7*32=224.
    origami::dim4_t grid{1, 16, 16, 1};
    auto u_first = origami::count_unique_tiles(grid, {0, 8, 4}, N_CU, num_xcd, 0, 0);
    auto u_last  = origami::count_unique_tiles(grid, {0, 8, 4}, N_CU, num_xcd, 7, 0);
    // Both should get same structure with symmetric grid
    CHECK(u_first.k == u_last.k);
    CHECK(u_first.b == u_last.b);
    // M/N may differ since they're at different positions, but should be valid
    CHECK(u_last.m >= 1);
    CHECK(u_last.m <= 16);
    CHECK(u_last.n >= 1);
    CHECK(u_last.n <= 16);
  }

  SECTION("Unique counts never exceed grid dimensions") {
    // Fuzz-like: various grid shapes should never produce out-of-range results
    std::vector<origami::dim4_t> grids             = {{1, 3, 7, 1},
                                                      {4, 5, 5, 2},
                                                      {1, 1, 64, 1},
                                                      {1, 64, 1, 1},
                                                      {8, 4, 4, 4},
                                                      {1, 16, 16, 1},
                                                      {2, 8, 8, 3}};
    std::vector<origami::workgroup_mapping_t> wgms = {
        {0, 8, 1}, {0, 8, 4}, {0, 8, 8}, {0, 1, 1}, {0, 0, 6}};
    for (auto& g : grids) {
      for (auto& w : wgms) {
        for (size_t xcd = 0; xcd < num_xcd; ++xcd) {
          auto u = origami::count_unique_tiles(g, w, N_CU, num_xcd, xcd, 0);
          CHECK(u.m <= g.m);
          CHECK(u.n <= g.n);
          CHECK(u.k <= g.k);
          CHECK(u.b <= g.b);
        }
      }
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
  // REQUIRE(defaults.l2_min_hit_rate_default == 0.5);
  REQUIRE(defaults.main_memory_load_latency == 200.0);
  REQUIRE(defaults.occupancy_decay_base == 0.95);
  REQUIRE(defaults.mall_depth_sq == 2.0);
  REQUIRE(defaults.mall_cold_floor == 0.85);
  REQUIRE(defaults.l2_depth_sq == 4.0);
  REQUIRE(defaults.l2_cold_floor == 0.75);
  REQUIRE(defaults.l2_pollution_penalty == 0.7);
  REQUIRE(defaults.l2_amp_ceiling_batched == 0.9);
  REQUIRE(defaults.l2_amp_ceiling_k_split == 0.4);
  REQUIRE(defaults.epilogue_cycles_per_acc_read == 8.0);
  REQUIRE(defaults.epilogue_acc_read_parallelism == 0.9);
  REQUIRE(defaults.epilogue_cycles_per_bounds_check == 6.0);
  REQUIRE(defaults.epilogue_scalar_store_penalty == 1.1);
  REQUIRE(defaults.epilogue_threads_per_wave == 64);
  REQUIRE(defaults.epilogue_bytes_per_vectorized_store == 16);
  REQUIRE(defaults.epilogue_cache_line_bytes == 128);
  REQUIRE(defaults.epilogue_workspace_bytes_per_elem == 4);
  REQUIRE(defaults.epilogue_salu_overhead == 35.0);
  REQUIRE(defaults.epilogue_l_barrier == 100.0);
  REQUIRE(defaults.epilogue_l_smem == 900.0);
  REQUIRE(defaults.epilogue_k_padding_penalty == 50000.0);
  REQUIRE(defaults.postgsu_compute_bytes == 4);
  REQUIRE(defaults.postgsu_kernel_launch_overhead == 12000.0);
  REQUIRE(defaults.postgsu_threads_per_wg == 256);
  REQUIRE(defaults.postgsu_wavefront_size == 64);

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
  REQUIRE(base.weight_mem_l2 == origami::heuristic_defaults_t::WEIGHT_MEM_L2);
  REQUIRE(base.weight_prologue == origami::heuristic_defaults_t::WEIGHT_PROLOGUE);
  REQUIRE(base.weight_epilogue == origami::heuristic_defaults_t::WEIGHT_EPILOGUE);
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

TEST_CASE("GEMM: compute_parallel_reduction_latency", "[gemm]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - returns zero when splitting_factor <= 1") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(4096, 4096, 1024);
      auto config   = make_config(128, 128, 64, 32, 32, 8, false, 1);

      origami::context_t ctx(problem, hardware, config);
      ctx.splitting_factor    = 1;
      ctx.reduction_strategy  = origami::reduction_t::parallel;

      auto latency = origami::compute_parallel_reduction_latency(problem, hardware, config, ctx);
      REQUIRE(latency == 0.0);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - returns zero for non-parallel reduction") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(4096, 4096, 1024);
      auto config   = make_config(128, 128, 64, 32, 32, 8, false, 1);

      origami::context_t ctx(problem, hardware, config);
      ctx.splitting_factor    = 4;
      ctx.reduction_strategy  = origami::reduction_t::none;

      auto latency = origami::compute_parallel_reduction_latency(problem, hardware, config, ctx);
      REQUIRE(latency == 0.0);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - positive latency for parallel split") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(4096, 4096, 8192);
      auto config   = make_config(128, 128, 64, 32, 32, 8, false, 1);

      origami::context_t ctx(problem, hardware, config);
      ctx.splitting_factor    = 4;
      ctx.reduction_strategy  = origami::reduction_t::parallel;

      auto latency = origami::compute_parallel_reduction_latency(problem, hardware, config, ctx);
      REQUIRE(latency > 0.0);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - higher split factor increases latency") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(4096, 4096, 8192);
      auto config   = make_config(128, 128, 64, 32, 32, 8, false, 1);

      origami::context_t ctx_lo(problem, hardware, config);
      ctx_lo.splitting_factor    = 2;
      ctx_lo.reduction_strategy  = origami::reduction_t::parallel;

      origami::context_t ctx_hi(problem, hardware, config);
      ctx_hi.splitting_factor    = 8;
      ctx_hi.reduction_strategy  = origami::reduction_t::parallel;

      auto lat_lo = origami::compute_parallel_reduction_latency(problem, hardware, config, ctx_lo);
      auto lat_hi = origami::compute_parallel_reduction_latency(problem, hardware, config, ctx_hi);
      REQUIRE(lat_hi > lat_lo);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - larger output increases latency") {
      auto hardware      = make_hardware(gpu_arch);
      auto problem_small = make_problem(1024, 1024, 8192);
      auto problem_large = make_problem(8192, 8192, 8192);
      auto config        = make_config(128, 128, 64, 32, 32, 8, false, 1);

      origami::context_t ctx_small(problem_small, hardware, config);
      ctx_small.splitting_factor    = 4;
      ctx_small.reduction_strategy  = origami::reduction_t::parallel;

      origami::context_t ctx_large(problem_large, hardware, config);
      ctx_large.splitting_factor    = 4;
      ctx_large.reduction_strategy  = origami::reduction_t::parallel;

      auto lat_small = origami::compute_parallel_reduction_latency(
          problem_small, hardware, config, ctx_small);
      auto lat_large = origami::compute_parallel_reduction_latency(
          problem_large, hardware, config, ctx_large);
      REQUIRE(lat_large > lat_small);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - batched problem scales latency") {
      auto hardware = make_hardware(gpu_arch);
      auto problem1 = make_problem(2048, 2048, 4096, origami::transpose_t::T, origami::transpose_t::N, 1);
      auto problem4 = make_problem(2048, 2048, 4096, origami::transpose_t::T, origami::transpose_t::N, 4);
      auto config   = make_config(128, 128, 64, 32, 32, 8, false, 1);

      origami::context_t ctx1(problem1, hardware, config);
      ctx1.splitting_factor    = 4;
      ctx1.reduction_strategy  = origami::reduction_t::parallel;

      origami::context_t ctx4(problem4, hardware, config);
      ctx4.splitting_factor    = 4;
      ctx4.reduction_strategy  = origami::reduction_t::parallel;

      auto lat1 = origami::compute_parallel_reduction_latency(problem1, hardware, config, ctx1);
      auto lat4 = origami::compute_parallel_reduction_latency(problem4, hardware, config, ctx4);
      REQUIRE(lat4 > lat1);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - includes kernel launch overhead") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(64, 64, 4096);
      auto config   = make_config(64, 64, 64, 32, 32, 8, false, 1);

      origami::context_t ctx(problem, hardware, config);
      ctx.splitting_factor    = 2;
      ctx.reduction_strategy  = origami::reduction_t::parallel;

      auto latency = origami::compute_parallel_reduction_latency(problem, hardware, config, ctx);
      REQUIRE(latency >= ctx.heuristic.postgsu_kernel_launch_overhead);
    }
  }
}

TEST_CASE("GEMM: compute_epilogue_latency", "[gemm]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - positive for aligned interior tiles") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(4096, 4096, 1024);
      auto config   = make_config(128, 128, 64, 32, 32, 8, false, 1);

      origami::context_t ctx(problem, hardware, config);
      auto latency = origami::compute_epilogue_latency(problem, hardware, config, ctx);
      REQUIRE(latency > 0.0);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - edge tiles cost more than interior") {
      auto hardware = make_hardware(gpu_arch);
      auto config   = make_config(128, 128, 64, 32, 32, 8, false, 1);

      auto problem_aligned = make_problem(4096, 4096, 1024);
      origami::context_t ctx_aligned(problem_aligned, hardware, config);
      auto lat_aligned =
          origami::compute_epilogue_latency(problem_aligned, hardware, config, ctx_aligned);

      auto problem_edge = make_problem(4097, 4097, 1024);
      origami::context_t ctx_edge(problem_edge, hardware, config);
      auto lat_edge = origami::compute_epilogue_latency(problem_edge, hardware, config, ctx_edge);

      REQUIRE(lat_edge > lat_aligned);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - larger tiles have higher epilogue cost") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(4096, 4096, 1024);

      auto config_small = make_config(64, 64, 64, 32, 32, 8, false, 1);
      auto config_large = make_config(256, 256, 64, 32, 32, 8, false, 1);

      origami::context_t ctx_small(problem, hardware, config_small);
      origami::context_t ctx_large(problem, hardware, config_large);

      auto lat_small = origami::compute_epilogue_latency(problem, hardware, config_small, ctx_small);
      auto lat_large = origami::compute_epilogue_latency(problem, hardware, config_large, ctx_large);
      REQUIRE(lat_large > lat_small);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - serial split-K adds reduction cost") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(4096, 4096, 8192);
      auto config   = make_config(128, 128, 64, 32, 32, 8, false, 1);

      origami::context_t ctx_nosplit(problem, hardware, config);
      ctx_nosplit.splitting_factor   = 1;
      ctx_nosplit.reduction_strategy = origami::reduction_t::none;

      origami::context_t ctx_spinlock(problem, hardware, config);
      ctx_spinlock.splitting_factor   = 4;
      ctx_spinlock.reduction_strategy = origami::reduction_t::spinlock;

      auto lat_nosplit  = origami::compute_epilogue_latency(problem, hardware, config, ctx_nosplit);
      auto lat_spinlock = origami::compute_epilogue_latency(problem, hardware, config, ctx_spinlock);
      REQUIRE(lat_spinlock > lat_nosplit);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - parallel reduction skips in-kernel reduce") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(4096, 4096, 8192);
      auto config   = make_config(128, 128, 64, 32, 32, 8, false, 1);

      origami::context_t ctx_spinlock(problem, hardware, config);
      ctx_spinlock.splitting_factor   = 4;
      ctx_spinlock.reduction_strategy = origami::reduction_t::spinlock;

      origami::context_t ctx_parallel(problem, hardware, config);
      ctx_parallel.splitting_factor   = 4;
      ctx_parallel.reduction_strategy = origami::reduction_t::parallel;

      auto lat_spinlock = origami::compute_epilogue_latency(problem, hardware, config, ctx_spinlock);
      auto lat_parallel = origami::compute_epilogue_latency(problem, hardware, config, ctx_parallel);
      REQUIRE(lat_spinlock > lat_parallel);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - returns zero when d_bytes is zero") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(4096, 4096, 1024);
      auto config   = make_config(128, 128, 64, 32, 32, 8, false, 1);

      origami::context_t ctx(problem, hardware, config);
      ctx.d_bytes = 0;

      auto latency = origami::compute_epilogue_latency(problem, hardware, config, ctx);
      REQUIRE(latency == 0.0);
    }
  }
}

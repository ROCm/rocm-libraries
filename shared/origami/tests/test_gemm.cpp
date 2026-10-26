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

namespace {

void set_problem_dtypes(origami::problem_t& problem, origami::data_type_t mi_dtype) {
  problem.mi_dtype = mi_dtype;
  if (mi_dtype == origami::data_type_t::XFloat32) {
    problem.a_dtype = origami::data_type_t::Float;
    problem.b_dtype = origami::data_type_t::Float;
    problem.c_dtype = origami::data_type_t::Float;
    problem.d_dtype = origami::data_type_t::Float;
    return;
  }

  problem.a_dtype = mi_dtype;
  problem.b_dtype = mi_dtype;
  problem.c_dtype = mi_dtype;
  problem.d_dtype = mi_dtype;
}

double lookup_cms_main_loop_efficiency(origami::data_type_t mi_dtype,
                                       origami::transpose_t trans_a,
                                       origami::transpose_t trans_b,
                                       size_t mt_m,
                                       size_t mt_n,
                                       size_t mt_k) {
  auto& db      = origami::heuristics_database_t::get_instance();
  auto hardware = make_hardware(950);
  auto problem  = make_problem(1024, 1024, 1024, trans_a, trans_b);
  auto config   = make_config(mt_m, mt_n, mt_k, 16, 16, 16, true);

  set_problem_dtypes(problem, mi_dtype);

  return db.lookup(problem, hardware, config).main_loop_efficiency;
}

}  // namespace

// Test functions for gemm.hpp/cpp

TEST_CASE("GEMM: compute_num_matrix_instructions", "[gemm]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - 128x128x64 with 16x16x16") {
      auto hardware = make_hardware(gpu_arch);
      origami::dim3_t mt{128, 128, 64};
      origami::dim3_t mi{16, 16, 16};
      auto num_instructions = origami::gemm::compute_number_matrix_instructions(mt, mi);
      REQUIRE(num_instructions == 256);  // 8 * 8 * 4
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - 16x16x64 with 16x16x32") {
      auto hardware = make_hardware(gpu_arch);
      origami::dim3_t mt{16, 16, 64};
      origami::dim3_t mi{16, 16, 32};
      auto num_instructions = origami::gemm::compute_number_matrix_instructions(mt, mi);
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

      auto latency = origami::gemm::compute_mt_compute_latency(problem, hardware, config);
      REQUIRE(latency == 4096);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - transA=N transB=T") {
      auto hardware = make_hardware(gpu_arch);
      auto problem =
          make_problem(4096, 4096, 1024, origami::transpose_t::N, origami::transpose_t::T);
      auto config = make_config(128, 128, 64, 32, 32, 8, false, 1);

      auto latency = origami::gemm::compute_mt_compute_latency(problem, hardware, config);
      REQUIRE(latency == 4096);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - transA=N transB=N") {
      auto hardware = make_hardware(gpu_arch);
      auto problem =
          make_problem(4096, 4096, 1024, origami::transpose_t::N, origami::transpose_t::N);
      auto config = make_config(128, 128, 64, 32, 32, 8, false, 1);

      auto latency = origami::gemm::compute_mt_compute_latency(problem, hardware, config);
      REQUIRE(latency == 4096);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - transA=T transB=T") {
      auto hardware = make_hardware(gpu_arch);
      auto problem =
          make_problem(4096, 4096, 1024, origami::transpose_t::T, origami::transpose_t::T);
      auto config = make_config(128, 128, 64, 32, 32, 8, false, 1);

      auto latency = origami::gemm::compute_mt_compute_latency(problem, hardware, config);
      REQUIRE(latency == 4096);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - different MT_K") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(4096, 4096, 1024);
      auto config   = make_config(128, 128, 32, 32, 32, 8, false, 1);

      auto latency = origami::gemm::compute_mt_compute_latency(problem, hardware, config);
      REQUIRE(latency == 2048);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - larger tile") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(4096, 4096, 1024);
      auto config   = make_config(224, 224, 64, 32, 32, 8, false, 1);

      auto latency = origami::gemm::compute_mt_compute_latency(problem, hardware, config);
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

      auto mem_latency_small = origami::gemm::compute_memory_latency(
          problem, hardware, config_small, origami::gemm::context_t(problem, hardware, config_small));
      auto mem_latency_large = origami::gemm::compute_memory_latency(
          problem, hardware, config_large, origami::gemm::context_t(problem, hardware, config_large));

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

      auto tile_latency_small = origami::gemm::compute_tile_latency(
          problem, hardware, config_small, origami::gemm::context_t(problem, hardware, config_small));
      auto tile_latency_large = origami::gemm::compute_tile_latency(
          problem, hardware, config_large, origami::gemm::context_t(problem, hardware, config_large));

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

      auto tile_latency = origami::gemm::compute_tile_latency(
          problem, hardware, config, origami::gemm::context_t(problem, hardware, config));
      auto timestep_latency = origami::gemm::compute_timestep_latency(
          problem, hardware, config, origami::gemm::context_t(problem, hardware, config));

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
          origami::gemm::compute_total_latency(problem_small, hardware, config);
      auto latency_large =
          origami::gemm::compute_total_latency(problem_large, hardware, config);

      REQUIRE(latency_small < latency_large);
    }
  }
}

TEST_CASE("GEMM: check_lds_capacity", "[gemm]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - 256x256x64 tile fits in LDS") {
      auto hardware = make_hardware(gpu_arch);
      origami::dim3_t mt{256, 256, 64};

      auto fits = origami::gemm::check_lds_capacity(
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
        auto l2_hit              = origami::gemm::estimate_l2_hit(
            problem, hardware, config, origami::gemm::context_t(problem, hardware, config));
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
        auto mall_hit            = origami::gemm::estimate_mall_hit(
            problem, hardware, config, origami::gemm::context_t(problem, hardware, config));
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
      auto result = origami::gemm::calculate_work_utilization(problem, config);
      REQUIRE(result == 1.0);

      // Test 2: Test with non-aligned dimensions
      auto problem_non_aligned = make_problem(4351, 3839, 959);
      auto result_non_aligned  = origami::gemm::calculate_work_utilization(problem_non_aligned, config);
      REQUIRE(result_non_aligned == Approx(0.998).epsilon(1e-3));

      // Test 3: Test with zero dimensions (should return 1.0)
      auto problem_zero_dimensions = make_problem(0, 3839, 959);
      auto result_zero_dimensions =
          origami::gemm::calculate_work_utilization(problem_zero_dimensions, config);
      REQUIRE(result_zero_dimensions == 1.0);

      // Test 4: Test with very small problems
      auto problem_very_small = make_problem(10, 20, 15);
      auto result_very_small  = origami::gemm::calculate_work_utilization(problem_very_small, config);
      REQUIRE(result_very_small == Approx(0.0007152).epsilon(1e-4));

      // Test 5: Test with very large problems
      auto problem_very_large = make_problem(409601, 409601, 4095);
      auto result_very_large  = origami::gemm::calculate_work_utilization(problem_very_large, config);
      REQUIRE(result_very_large == Approx(0.998).epsilon(1e-3));

      // Test 6: Test with skinny matrices
      auto problem_skinny = make_problem(128, 81920, 1024);  // Small M, Big N
      auto result_skinny  = origami::gemm::calculate_work_utilization(problem_skinny, config);
      REQUIRE(result_skinny == 0.5);

      problem_skinny = make_problem(81920, 128, 1024);  // Small N, Big M
      result_skinny  = origami::gemm::calculate_work_utilization(problem_skinny, config);
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
      auto result = origami::gemm::calculate_output_utilization(problem, config, 1UL);
      REQUIRE(result == 1.0);

      // Test 2: Test with non-aligned dimensions
      auto problem_non_aligned = make_problem(4351, 3839, 959);
      auto result_non_aligned =
          origami::gemm::calculate_output_utilization(problem_non_aligned, config, 1UL);
      REQUIRE(result_non_aligned == Approx(0.999).epsilon(1e-3));

      // Test 3: Test with vector_elems > 1
      auto result_with_vector_elems = origami::gemm::calculate_output_utilization(problem, config, 23UL);
      REQUIRE(result_with_vector_elems == Approx(1.010).epsilon(1e-3));

      // Test 4: Test with zero dimensions (should return 1.0)
      auto problem_zero_dimensions = make_problem(0, 3839, 959);
      auto result_zero_dimensions =
          origami::gemm::calculate_output_utilization(problem_zero_dimensions, config, 1UL);
      REQUIRE(result_zero_dimensions == 1.0);

      // Test 5: Test with different problem sizes
      auto problem_very_small = make_problem(10, 20, 15);
      auto result_very_small =
          origami::gemm::calculate_output_utilization(problem_very_small, config, 1UL);
      REQUIRE(result_very_small == Approx(0.003051).epsilon(1e-3));

      auto problem_very_large = make_problem(409601, 409601, 4095);
      auto result_very_large =
          origami::gemm::calculate_output_utilization(problem_very_large, config, 1UL);
      REQUIRE(result_very_large == Approx(0.998).epsilon(1e-3));

      auto problem_skinny = make_problem(64, 81920, 1024);  // Small M, Big N
      auto result_skinny  = origami::gemm::calculate_output_utilization(problem_skinny, config, 1UL);
      REQUIRE(result_skinny == 0.25);

      problem_skinny = make_problem(81920, 64, 1024);  // Small N, Big M
      result_skinny  = origami::gemm::calculate_output_utilization(problem_skinny, config, 1UL);
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
          origami::gemm::compute_launch_parameters(
              problem, hardware, config, origami::grid_selection_t::k_split_aware);
      REQUIRE(num_wgs > 0);
      REQUIRE(num_active_cus > 0);
      REQUIRE(num_timesteps >= 1);
      REQUIRE(split_factor >= 1);

      // Test with different grid selection algorithms
      auto result_analytical = origami::gemm::compute_launch_parameters(
          problem, hardware, config, origami::grid_selection_t::analytical);
      REQUIRE(std::get<1>(result_analytical) > 0);

      // The CU budget now comes from problem.num_cus: capping it below the
      // physical CU count must bound the number of active CUs accordingly.
      auto capped_problem    = problem;
      capped_problem.num_cus = 150;
      auto result_max_cus    = origami::gemm::compute_launch_parameters(
          capped_problem, hardware, config, origami::grid_selection_t::k_split_aware);
      REQUIRE(std::get<2>(result_max_cus) <= 150);
    }
  }
}

TEST_CASE("GEMM: count_unique_range unit test", "[gemm]") {
  // Test 1: Single tile — range covers exactly one WG
  {
    origami::dim4_t grid{1, 4, 8, 1};  // k=1, m=4, n=8, b=1
    auto u = origami::gemm::count_unique_range(grid, 8, 0, 1);
    REQUIRE(u.m == 1);
    REQUIRE(u.n == 1);
    REQUIRE(u.k == 1);
    REQUIRE(u.b == 1);
  }

  // Test 2: Full grid coverage — all tiles
  {
    origami::dim4_t grid{1, 4, 8, 1};
    auto u = origami::gemm::count_unique_range(grid, 8, 0, 32);
    REQUIRE(u.m == 4);
    REQUIRE(u.n == 8);
    REQUIRE(u.k == 1);
    REQUIRE(u.b == 1);
  }

  // Test 3: Within one slab (wgm=4), partial M
  {
    origami::dim4_t grid{1, 8, 16, 1};  // k=1, m=8, n=16, b=1
    auto u = origami::gemm::count_unique_range(grid, 4, 0, 3);
    REQUIRE(u.m == 1);
    REQUIRE(u.n == 3);
    REQUIRE(u.k == 1);
  }

  // Test 4: Within one slab, spans multiple rows
  {
    origami::dim4_t grid{1, 8, 16, 1};
    // wgm=4: slab has 8*4=32 tiles. IDs 0..7 = row0 cols 0-3, row1 cols 0-3
    auto u = origami::gemm::count_unique_range(grid, 4, 0, 8);
    REQUIRE(u.m == 2);
    REQUIRE(u.n == 4);  // full slab width
  }

  // Test 5: Across two slabs
  {
    origami::dim4_t grid{1, 4, 8, 1};
    // wgm=4: slab0 has 4*4=16 tiles (cols 0-3), slab1 has 16 tiles (cols 4-7)
    auto u = origami::gemm::count_unique_range(grid, 4, 14, 4);
    REQUIRE(u.m >= 1);
    REQUIRE(u.n >= 2);  // spans across slab boundary
  }

  // Test 6: K-split dimension
  {
    origami::dim4_t grid{4, 2, 2, 1};  // k=4 splits, m=2, n=2, b=1
    // First 4 IDs = same MN tile, different K splits
    auto u = origami::gemm::count_unique_range(grid, 2, 0, 4);
    REQUIRE(u.m == 1);
    REQUIRE(u.n == 1);
    REQUIRE(u.k == 4);
  }

  // Test 7: K-split spanning multiple MN tiles -> all K covered
  {
    origami::dim4_t grid{4, 2, 2, 1};
    auto u = origami::gemm::count_unique_range(grid, 2, 0, 5);
    REQUIRE(u.k == 4);  // crossed MN boundary, all K touched
  }

  // Test 8: Batch dimension
  {
    origami::dim4_t grid{1, 4, 4, 3};  // k=1, m=4, n=4, b=3
    // 16 tiles per batch, 48 total. Range covering 2 batches:
    auto u = origami::gemm::count_unique_range(grid, 4, 0, 32);
    REQUIRE(u.b == 2);
    REQUIRE(u.m == 4);  // multiple batches -> full M
    REQUIRE(u.n == 4);  // multiple batches -> full N
  }

  // Test 9: wgm=1 (column-first, each slab is 1 column)
  {
    origami::dim4_t grid{1, 4, 8, 1};
    auto u = origami::gemm::count_unique_range(grid, 1, 0, 4);
    REQUIRE(u.m == 4);  // 4 rows in one column
    REQUIRE(u.n == 1);  // single column slab
  }
}

TEST_CASE("GEMM: wgm_to_grid unit test", "[gemm]") {
  // Test 1: Simple 2x4 grid, wgm=4 (single slab covers all N)
  {
    origami::dim4_t grid{1, 2, 4, 1};
    origami::workgroup_mapping_t wgm{0, 0, 0, 4};
    // slab_width=4, tiles_per_slab=2*4=8. All 8 tiles in one slab.
    // offset 0: m=0,n=0  offset 1: m=0,n=1  offset 2: m=0,n=2  offset 3: m=0,n=3
    // offset 4: m=1,n=0  offset 5: m=1,n=1  offset 6: m=1,n=2  offset 7: m=1,n=3
    auto t0 = origami::gemm::wgm_to_grid(grid, wgm, 0);
    REQUIRE(t0.m == 0);
    REQUIRE(t0.n == 0);
    REQUIRE(t0.k == 0);
    REQUIRE(t0.b == 0);
    auto t3 = origami::gemm::wgm_to_grid(grid, wgm, 3);
    REQUIRE(t3.m == 0);
    REQUIRE(t3.n == 3);
    auto t4 = origami::gemm::wgm_to_grid(grid, wgm, 4);
    REQUIRE(t4.m == 1);
    REQUIRE(t4.n == 0);
    auto t7 = origami::gemm::wgm_to_grid(grid, wgm, 7);
    REQUIRE(t7.m == 1);
    REQUIRE(t7.n == 3);
  }

  // Test 2: Two slabs — 4x6 grid, wgm=3
  {
    origami::dim4_t grid{1, 4, 6, 1};
    origami::workgroup_mapping_t wgm{0, 0, 0, 3};
    // slab0 (cols 0-2): 4*3=12 tiles, slab1 (cols 3-5): 12 tiles
    auto t0 = origami::gemm::wgm_to_grid(grid, wgm, 0);
    REQUIRE(t0.m == 0);
    REQUIRE(t0.n == 0);
    auto t2 = origami::gemm::wgm_to_grid(grid, wgm, 2);
    REQUIRE(t2.m == 0);
    REQUIRE(t2.n == 2);
    auto t3 = origami::gemm::wgm_to_grid(grid, wgm, 3);
    REQUIRE(t3.m == 1);
    REQUIRE(t3.n == 0);
    auto t11 = origami::gemm::wgm_to_grid(grid, wgm, 11);
    REQUIRE(t11.m == 3);
    REQUIRE(t11.n == 2);
    // slab1 starts at id=12
    auto t12 = origami::gemm::wgm_to_grid(grid, wgm, 12);
    REQUIRE(t12.m == 0);
    REQUIRE(t12.n == 3);
    auto t23 = origami::gemm::wgm_to_grid(grid, wgm, 23);
    REQUIRE(t23.m == 3);
    REQUIRE(t23.n == 5);
  }

  // Test 3: K-splits — each MN tile has k splits
  {
    origami::dim4_t grid{4, 2, 3, 1};  // k=4, m=2, n=3, b=1
    origami::workgroup_mapping_t wgm{0, 0, 0, 3};
    // First MN tile (m=0,n=0) has ids 0..3 (k=0..3)
    auto t0 = origami::gemm::wgm_to_grid(grid, wgm, 0);
    REQUIRE(t0.m == 0);
    REQUIRE(t0.n == 0);
    REQUIRE(t0.k == 0);
    auto t3 = origami::gemm::wgm_to_grid(grid, wgm, 3);
    REQUIRE(t3.m == 0);
    REQUIRE(t3.n == 0);
    REQUIRE(t3.k == 3);
    // Second MN tile (m=0,n=1) starts at id=4
    auto t4 = origami::gemm::wgm_to_grid(grid, wgm, 4);
    REQUIRE(t4.m == 0);
    REQUIRE(t4.n == 1);
    REQUIRE(t4.k == 0);
  }

  // Test 4: Batch dimension
  {
    origami::dim4_t grid{1, 2, 2, 3};  // k=1, m=2, n=2, b=3
    origami::workgroup_mapping_t wgm{0, 0, 0, 2};
    // 4 tiles per batch. batch 0: ids 0-3, batch 1: ids 4-7, batch 2: ids 8-11
    auto t0 = origami::gemm::wgm_to_grid(grid, wgm, 0);
    REQUIRE(t0.b == 0);
    REQUIRE(t0.m == 0);
    REQUIRE(t0.n == 0);
    auto t4 = origami::gemm::wgm_to_grid(grid, wgm, 4);
    REQUIRE(t4.b == 1);
    REQUIRE(t4.m == 0);
    REQUIRE(t4.n == 0);
    auto t11 = origami::gemm::wgm_to_grid(grid, wgm, 11);
    REQUIRE(t11.b == 2);
    REQUIRE(t11.m == 1);
    REQUIRE(t11.n == 1);
  }

  // Test 5: WGMXCC interleaving
  {
    origami::dim4_t grid{1, 4, 4, 1};               // 16 tiles total
    origami::workgroup_mapping_t wgm_xcc{0, 0, 8, 4};
    origami::workgroup_mapping_t wgm_no{0, 0, 0, 4};

    // With WGMXCC, consecutive IDs should map to different XCD groups.
    // ID 0 and ID 1 should land in different XCD regions.
    auto t0_xcc = origami::gemm::wgm_to_grid(grid, wgm_xcc, 0);
    auto t1_xcc = origami::gemm::wgm_to_grid(grid, wgm_xcc, 1);
    auto t0_no  = origami::gemm::wgm_to_grid(grid, wgm_no, 0);

    // Without WGMXCC, id 0 and 1 are adjacent; with WGMXCC they're spread apart
    REQUIRE(t0_xcc.m == t0_no.m);
    REQUIRE(t0_xcc.n == t0_no.n);
    // id=1 with xcc should NOT be the same as id=1 without xcc (it gets remapped)
    auto t1_no = origami::gemm::wgm_to_grid(grid, wgm_no, 1);
    bool same  = (t1_xcc.m == t1_no.m && t1_xcc.n == t1_no.n);
    REQUIRE_FALSE(same);
  }

  // Test 6: wgm=1 — each slab is one column, M varies fastest
  {
    origami::dim4_t grid{1, 4, 3, 1};
    origami::workgroup_mapping_t wgm{0, 0, 0, 1};
    // slab_width=1, tiles_per_slab=4. Col 0: ids 0-3, col 1: ids 4-7, col 2: ids 8-11
    auto t0 = origami::gemm::wgm_to_grid(grid, wgm, 0);
    REQUIRE(t0.m == 0);
    REQUIRE(t0.n == 0);
    auto t3 = origami::gemm::wgm_to_grid(grid, wgm, 3);
    REQUIRE(t3.m == 3);
    REQUIRE(t3.n == 0);
    auto t4 = origami::gemm::wgm_to_grid(grid, wgm, 4);
    REQUIRE(t4.m == 0);
    REQUIRE(t4.n == 1);
  }

  // Test 7: Remainder slab — grid.n not divisible by wgm
  {
    origami::dim4_t grid{1, 3, 5, 1};  // 5 cols, wgm=2 -> 2 full slabs + remainder of 1
    origami::workgroup_mapping_t wgm{0, 0, 0, 2};
    // slab0: cols 0,1 (6 tiles), slab1: cols 2,3 (6 tiles), remainder: col 4 (3 tiles)
    auto t12 = origami::gemm::wgm_to_grid(grid, wgm, 12);
    REQUIRE(t12.m == 0);
    REQUIRE(t12.n == 4);
    auto t14 = origami::gemm::wgm_to_grid(grid, wgm, 14);
    REQUIRE(t14.m == 2);
    REQUIRE(t14.n == 4);
  }

  // Test 8: Brute-force 'bijection' test — every id maps to a unique (m,n,k,b) and back
  {
    origami::dim4_t grid{2, 3, 4, 2};
    origami::workgroup_mapping_t wgm{0, 0, 0, 2};
    size_t total = grid.total();

    std::set<std::tuple<size_t, size_t, size_t, size_t>> seen;
    for (size_t id = 0; id < total; ++id) {
      auto t = origami::gemm::wgm_to_grid(grid, wgm, id);
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
    auto [m, n] = origami::gemm::compute_mall_tiles(16, 32, 256, 6);
    REQUIRE(m >= 1);
    REQUIRE(m <= 16);
    REQUIRE(n >= 1);
    REQUIRE(n <= 32);
  }

  // Test 2: wgm_value == 1 → slab width 1, all CUs in M dimension
  {
    auto [m, n] = origami::gemm::compute_mall_tiles(128, 128, 256, 1);
    REQUIRE(m <= 128);
    REQUIRE(n >= 1);
  }

  // Test 3: Large wgm → slab covers entire N
  {
    auto [m, n] = origami::gemm::compute_mall_tiles(16, 8, 64, 16);
    REQUIRE(n <= 8);
    REQUIRE(m <= 16);
  }

  // Test 4: active_cus > grid tiles → wrap around
  {
    auto [m, n] = origami::gemm::compute_mall_tiles(4, 4, 256, 4);
    REQUIRE(m == 4);
    REQUIRE(n == 4);
  }

  // Test 5: Single tile grid
  {
    auto [m, n] = origami::gemm::compute_mall_tiles(1, 1, 256, 6);
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
      auto [m, n] = origami::gemm::compute_l2_tiles(problem, hardware, config, grid_m, grid_n, 256, 1, 6);
      REQUIRE(m >= 1);
      REQUIRE(m <= grid_m);
      REQUIRE(n >= 1);
      REQUIRE(n <= grid_n);

      // Test 2: Larger WGM → wider slab → more N reuse
      auto [m1, n1] =
          origami::gemm::compute_l2_tiles(problem, hardware, config, grid_m, grid_n, 256, 1, 1);
      auto [m2, n2] =
          origami::gemm::compute_l2_tiles(problem, hardware, config, grid_m, grid_n, 256, 1, 8);
      // Wider slab should not have fewer N columns
      REQUIRE(n2 >= n1);

      // Test 3: With splitting factor > 1 → fewer effective CUs per XCD → smaller tile
      auto [m3, n3] =
          origami::gemm::compute_l2_tiles(problem, hardware, config, grid_m, grid_n, 256, 4, 6);
      REQUIRE(m3 <= m);  // Splitting should not increase tile size

      // Test 4: Single tile grid
      auto [m4, n4] = origami::gemm::compute_l2_tiles(problem, hardware, config, 1, 1, 256, 1, 6);
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
          origami::gemm::compute_mem_bw_from_occupancy(hardware, hardware.N_CU);
      REQUIRE(result_various_num_active_cus == 1.0);

      // Test 2: Test with different mem_bw_per_wg_coefficients
      hardware.mem_bw_per_wg_coefficients = std::make_tuple(0, 0.008, 0);
      auto result_different_mem_bw_per_wg_coefficients =
          origami::gemm::compute_mem_bw_from_occupancy(hardware, hardware.N_CU);
      REQUIRE(result_different_mem_bw_per_wg_coefficients == 1.0);

      hardware.mem_bw_per_wg_coefficients = std::make_tuple(0, 0.17, 0);
      result_different_mem_bw_per_wg_coefficients =
          origami::gemm::compute_mem_bw_from_occupancy(hardware, hardware.N_CU);
      REQUIRE(result_different_mem_bw_per_wg_coefficients == 1.0);

      hardware.mem_bw_per_wg_coefficients = std::make_tuple(0, 0.22, 0);
      result_different_mem_bw_per_wg_coefficients =
          origami::gemm::compute_mem_bw_from_occupancy(hardware, hardware.N_CU);
      REQUIRE(result_different_mem_bw_per_wg_coefficients == 1.0);

      // Test 3: Test with values less than 1
      hardware.mem_bw_per_wg_coefficients = std::make_tuple(0.000001, 0.001, 0);
      auto result_value_less_than_one =
          origami::gemm::compute_mem_bw_from_occupancy(hardware, hardware.N_CU);
      if (gpu_arch == 942)
        REQUIRE(result_value_less_than_one == Approx(0.3964).epsilon(1e-3));
      else if (gpu_arch == 950)
        REQUIRE(result_value_less_than_one == Approx(0.32153).epsilon(1e-3));

      hardware.mem_bw_per_wg_coefficients = std::make_tuple(0.000002, 0.002, 0);
      result_value_less_than_one = origami::gemm::compute_mem_bw_from_occupancy(hardware, hardware.N_CU);
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
      auto result_various_problem_sizes = origami::gemm::compute_l2_hit_rate_global(
          problem, hardware, config, hardware.L2_capacity * 1024);
      REQUIRE(result_various_problem_sizes == 0.875);

      auto problem_small           = make_problem(331, 4077, 547);
      result_various_problem_sizes = origami::gemm::compute_l2_hit_rate_global(
          problem_small, hardware, config, hardware.L2_capacity * 1024);
      REQUIRE(result_various_problem_sizes == 0.71875);

      auto problem_large           = make_problem(8193, 4077, 7453);
      result_various_problem_sizes = origami::gemm::compute_l2_hit_rate_global(
          problem_large, hardware, config, hardware.L2_capacity * 1024);
      REQUIRE(result_various_problem_sizes == Approx(0.953).epsilon(1e-3));

      // Test 2: Test with different splitting factors (TODO)(is this a valid test case as this
      // function does not use splitting factors) Test 3: Test edge cases
      REQUIRE_THROWS_WITH(origami::gemm::compute_l2_hit_rate_global(problem, hardware, config, 0UL),
                          "L2 Capacity is zero");

      auto problem_zero = make_problem(0, 0, 7453);
      REQUIRE_THROWS_WITH(origami::gemm::compute_l2_hit_rate_global(
                              problem_zero, hardware, config, hardware.L2_capacity * 1024),
                          "estimate_l2_hit grid dimensions can not be zero");
    }
  }
}

TEST_CASE("GEMM: round_elements_to_NB unit test", "[gemm]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - round_elements_to_NB unit test") {
      // Test 1: Test with various element sizes (128-byte transaction)
      auto result_various_element_sizes =
          origami::gemm::round_elements_to_NB(196, 32, 128);  // element size in bits - 32
      REQUIRE(result_various_element_sizes == 224);

      result_various_element_sizes =
          origami::gemm::round_elements_to_NB(225, 16, 128);  // element size in bits - 16
      REQUIRE(result_various_element_sizes == 256);

      result_various_element_sizes =
          origami::gemm::round_elements_to_NB(90, 8, 128);  // element size in bits - 8
      REQUIRE(result_various_element_sizes == 128);

      // Test 2: Test alignment to 128-byte boundary (TODO) (Was already covered in the above
      // example, could be skipped) Test 3: Test edge cases
      auto result_edge_cases = origami::gemm::round_elements_to_NB(0, 32, 128);
      REQUIRE(result_edge_cases == 0);

      result_edge_cases = origami::gemm::round_elements_to_NB(256, 0, 128);
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
      auto result_test_with_Float = origami::gemm::compute_cvt_overhead(problem_Float, hardware, config);
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
          origami::gemm::compute_cvt_overhead(problem_XFloat32, hardware, config);
      REQUIRE(result_test_with_XFloat32 == 0.0);

      // Test 3: Test conversion overhead calculation (TODO) (Need more clarification)
      // Test 4: Test with different tile sizes
      auto result_with_different_tile_sizes =
          origami::gemm::compute_cvt_overhead(problem, hardware, config);
      REQUIRE(result_with_different_tile_sizes == 0.0);

      config                           = make_config(128, 128, 64, 32, 32, 8, false, 1);
      result_with_different_tile_sizes = origami::gemm::compute_cvt_overhead(problem, hardware, config);
      REQUIRE(result_with_different_tile_sizes == 0.0);

      config                           = make_config(64, 64, 256, 32, 32, 8, false, 1);
      result_with_different_tile_sizes = origami::gemm::compute_cvt_overhead(problem, hardware, config);
      REQUIRE(result_with_different_tile_sizes == 0.0);
    }
  }
}

TEST_CASE("GEMM: tf32 conversion cost follows the kernel's wave layout", "[gemm][cvt]") {
  auto hardware = make_hardware(950);
  auto x3       = make_problem(4096, 4096, 4096, origami::transpose_t::N, origami::transpose_t::N);
  x3.a_dtype    = origami::data_type_t::Float;
  x3.b_dtype    = origami::data_type_t::Float;
  x3.c_dtype    = origami::data_type_t::Float;
  x3.d_dtype    = origami::data_type_t::Float;
  x3.mi_dtype   = origami::data_type_t::XFloat32;
  auto x1       = x3;
  x1.mi_dtype   = origami::data_type_t::BFloat16;

  auto with_waves = [](int wave_group_m, int wave_group_n, int local_split_u) {
    auto config      = make_config(64, 32, 128, 16, 16, 32);
    auto& tp         = config.tensile();
    tp.wave_group_m  = wave_group_m;
    tp.wave_group_n  = wave_group_n;
    tp.local_split_u = local_split_u;
    return config;
  };
  using cvt_fn =
      double (*)(const origami::problem_t&, const origami::hardware_t&, const origami::config_t&);
  const std::vector<std::pair<const char*, std::pair<cvt_fn, origami::problem_t>>> variants = {
      {"x3", {&origami::gemm::compute_cvt_overhead, x3}},
      {"x1", {&origami::gemm::compute_cvt_overhead_x1, x1}}};
  for (const auto& [label, fn_problem] : variants) {
    DYNAMIC_SECTION(label) {
      const auto& [cvt, problem]  = fn_problem;
      const double default_layout = cvt(problem, hardware, make_config(64, 32, 128, 16, 16, 32));
      const double two_by_two     = cvt(problem, hardware, with_waves(2, 2, 1));
      const double one_wave       = cvt(problem, hardware, with_waves(1, 1, 1));
      const double one_wave_lsu4  = cvt(problem, hardware, with_waves(1, 1, 4));

      // Without Tensile params the kernel is taken as the default 2x2 wave group.
      REQUIRE(default_layout == two_by_two);
      REQUIRE(two_by_two > 0.0);
      // A single wave converts the whole tile's operands, not a quarter of them.
      REQUIRE(one_wave > two_by_two);
      // Each LocalSplitU group converts only its share of the K-slice.
      REQUIRE(one_wave_lsu4 < one_wave);
    }
  }
}

TEST_CASE("GEMM: LocalSplitU waves fill SIMDs a single-wave group leaves idle", "[gemm][lsu]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch) {
      auto hardware = make_hardware(gpu_arch);
      // A single output tile: no CU holds more than one WG, so only that WG's waves run.
      auto problem = make_problem(128, 128, 8192);
      auto latency = [&](int wave_group_m, int wave_group_n, int local_split_u) {
        auto config      = make_config(128, 128, 64, 16, 16, 16, false, 1, 1, 0, 0, 0);
        auto& tp         = config.tensile();
        tp.wave_group_m  = wave_group_m;
        tp.wave_group_n  = wave_group_n;
        tp.local_split_u = local_split_u;
        return origami::gemm::compute_total_latency(problem, hardware, config);
      };
      REQUIRE(latency(1, 1, 4) < latency(1, 1, 1));
    }
  }
}

TEST_CASE("GEMM: occupancy only helps when the grid fills it", "[gemm][occupancy]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch) {
      auto hardware = make_hardware(gpu_arch);
      auto latency  = [&](size_t m, size_t n, int occupancy) {
        auto config = make_config(128, 128, 64, 16, 16, 16, false, 1, occupancy, 0, 0, 0);
        return origami::gemm::compute_total_latency(make_problem(m, n, 4096), hardware, config);
      };
      // Four tiles: no CU gets a second WG, whatever the kernel's occupancy allows.
      REQUIRE(latency(256, 256, 2) == latency(256, 256, 1));
      // Thousands of tiles: a second resident WG hides cross-WG stalls.
      REQUIRE(latency(8192, 8192, 2) < latency(8192, 8192, 1));
    }
  }
}

TEST_CASE("GEMM: a problem narrower than one MFMA isn't charged twice for its padding",
          "[gemm][etp]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch) {
      auto hardware = make_hardware(gpu_arch);
      // A 16-wide tile does the same MFMA work for N=1 as for N=16, and N=1 moves less data.
      auto latency = [&](size_t n) {
        auto config = make_config(256, 16, 64, 16, 16, 16, false, 1, 2, 0, 0, 0);
        auto problem =
            make_problem(4096, n, 4096, origami::transpose_t::N, origami::transpose_t::N);
        return origami::gemm::compute_total_latency(problem, hardware, config);
      };
      REQUIRE(latency(1) <= latency(16));
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
      auto result = origami::gemm::arithmetic_intensity(2047, 2047, 4096, 2);
      REQUIRE(result == Approx(818.879).epsilon(1e-3));

      result = origami::gemm::emulated_tf32_arithmetic_intensity(2047, 2047, 4096, 2);
      REQUIRE(result == Approx(2456.639).epsilon(1e-3));

      // Test 2: Test with various m, n, k, bytes_per_element
      auto result_various_problem_size = origami::gemm::arithmetic_intensity(127, 4097, 8193, 2);
      REQUIRE(result_various_problem_size == Approx(121.356).epsilon(1e-3));

      result_various_problem_size = origami::gemm::arithmetic_intensity(4097, 4097, 257, 4);
      REQUIRE(result_various_problem_size == Approx(114.175).epsilon(1e-3));

      result_various_problem_size = origami::gemm::arithmetic_intensity(237, 4097, 8183, 4);
      REQUIRE(result_various_problem_size == Approx(109.034).epsilon(1e-3));

      result_various_problem_size = origami::gemm::emulated_tf32_arithmetic_intensity(127, 4097, 8193, 2);
      REQUIRE(result_various_problem_size == Approx(364.0709).epsilon(1e-3));

      result_various_problem_size = origami::gemm::emulated_tf32_arithmetic_intensity(4097, 4097, 257, 4);
      REQUIRE(result_various_problem_size == Approx(342.527).epsilon(1e-3));

      result_various_problem_size = origami::gemm::emulated_tf32_arithmetic_intensity(237, 4097, 8183, 4);
      REQUIRE(result_various_problem_size == Approx(327.104).epsilon(1e-3));

      // Test 3: Test edge cases (zero values)
      auto result_zero_value = origami::gemm::arithmetic_intensity(0, 0, 0, 4);
      REQUIRE(result_zero_value == 0.0);

      result_zero_value = origami::gemm::emulated_tf32_arithmetic_intensity(0, 0, 0, 2);
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
            origami::gemm::check_lds_capacity(hardware,
                                        {256, 256, 256},
                                        origami::data_type_t::BFloat16,
                                        origami::data_type_t::BFloat16);
        REQUIRE(result_tiles_exceed_LDS_capacity == false);

        result_tiles_exceed_LDS_capacity =
            origami::gemm::check_lds_capacity(hardware,
                                        {128, 128, 256},
                                        origami::data_type_t::BFloat16,
                                        origami::data_type_t::BFloat16);
        REQUIRE(result_tiles_exceed_LDS_capacity == false);

        result_tiles_exceed_LDS_capacity =
            origami::gemm::check_lds_capacity(hardware,
                                        {64, 128, 256},
                                        origami::data_type_t::BFloat16,
                                        origami::data_type_t::BFloat16);
        REQUIRE(result_tiles_exceed_LDS_capacity == false);

        // Test 2: Test with different data type combinations
        auto result_different_data_type = origami::gemm::check_lds_capacity(
            hardware, {64, 64, 256}, origami::data_type_t::Half, origami::data_type_t::Half);
        REQUIRE(result_different_data_type == true);

        result_different_data_type = origami::gemm::check_lds_capacity(
            hardware, {256, 256, 512}, origami::data_type_t::Double, origami::data_type_t::Double);
        REQUIRE(result_different_data_type == false);

        result_different_data_type = origami::gemm::check_lds_capacity(
            hardware, {128, 128, 64}, origami::data_type_t::Int8, origami::data_type_t::Int8);
        REQUIRE(result_different_data_type == true);

        result_different_data_type = origami::gemm::check_lds_capacity(
            hardware, {128, 128, 32}, origami::data_type_t::Int64, origami::data_type_t::Int64);
        REQUIRE(result_different_data_type == true);

        // Test 3: Test with Float (element_size = 16) (TODO: Need more clarification)

        // Test 4: Test edge cases (exactly at capacity, just over)
        auto result_exactly_at_capacity =
            origami::gemm::check_lds_capacity(hardware,
                                        {256, 256, 64},
                                        origami::data_type_t::BFloat16,
                                        origami::data_type_t::Float8);  // exactly at capacity
        REQUIRE(result_exactly_at_capacity == true);

        result_exactly_at_capacity =
            origami::gemm::check_lds_capacity(hardware,
                                        {192, 96, 128},
                                        origami::data_type_t::BFloat16,
                                        origami::data_type_t::BFloat16);  // just over
        REQUIRE(result_exactly_at_capacity == false);
      } else if (gpu_arch == 950) {
        // Test 1: Test with tiles that exceed LDS capacity
        auto result_tiles_exceed_LDS_capacity =
            origami::gemm::check_lds_capacity(hardware,
                                        {512, 512, 256},
                                        origami::data_type_t::BFloat16,
                                        origami::data_type_t::BFloat16);
        REQUIRE(result_tiles_exceed_LDS_capacity == false);

        result_tiles_exceed_LDS_capacity =
            origami::gemm::check_lds_capacity(hardware,
                                        {256, 256, 512},
                                        origami::data_type_t::BFloat16,
                                        origami::data_type_t::BFloat16);
        REQUIRE(result_tiles_exceed_LDS_capacity == false);

        result_tiles_exceed_LDS_capacity =
            origami::gemm::check_lds_capacity(hardware,
                                        {512, 128, 256},
                                        origami::data_type_t::BFloat16,
                                        origami::data_type_t::BFloat16);
        REQUIRE(result_tiles_exceed_LDS_capacity == false);

        // Test 2: Test with different data type combinations
        auto result_different_data_type = origami::gemm::check_lds_capacity(
            hardware, {64, 64, 256}, origami::data_type_t::Half, origami::data_type_t::Half);
        REQUIRE(result_different_data_type == true);

        result_different_data_type = origami::gemm::check_lds_capacity(
            hardware, {256, 256, 512}, origami::data_type_t::Double, origami::data_type_t::Double);
        REQUIRE(result_different_data_type == false);

        result_different_data_type = origami::gemm::check_lds_capacity(
            hardware, {256, 256, 64}, origami::data_type_t::Int8, origami::data_type_t::Int8);
        REQUIRE(result_different_data_type == true);

        result_different_data_type = origami::gemm::check_lds_capacity(
            hardware, {256, 256, 32}, origami::data_type_t::Int64, origami::data_type_t::Int64);
        REQUIRE(result_different_data_type == true);

        // Test 3: Test with Float (element_size = 16) (TODO: Need more clarification)

        // Test 4: Test edge cases (exactly at capacity, just over)
        auto result_exactly_at_capacity =
            origami::gemm::check_lds_capacity(hardware,
                                        {512, 128, 256},
                                        origami::data_type_t::Float8,
                                        origami::data_type_t::Float8);  // exactly at capacity
        REQUIRE(result_exactly_at_capacity == true);

        result_exactly_at_capacity =
            origami::gemm::check_lds_capacity(hardware,
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
        origami::gemm::context_t ctx(p, hardware, c);
        auto l2   = origami::gemm::estimate_l2_hit(p, hardware, c, ctx);
        auto mall = origami::gemm::estimate_mall_hit(p, hardware, c, ctx);
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
    auto u = origami::gemm::count_unique_tiles({1, 2, 2, 64}, {0, 0, 1, 1}, N_CU, num_xcd, 0, 0);
    CHECK(u.k == 1);
    CHECK(u.m == 1);
    CHECK(u.n == 1);
    CHECK(u.b == 32);
  }

  SECTION("grid_b=64, grid_k=1, wgmxcc=8 — contiguous block spans all mn per batch") {
    // Grid: 2×2×1×64, wgm=1, wgmxcc=8
    // Each XCD gets 32 contiguous tiles -> 4 mn × 8 batches
    auto u = origami::gemm::count_unique_tiles({1, 2, 2, 64}, {0, 0, 8, 1}, N_CU, num_xcd, 0, 0);
    CHECK(u.k == 1);
    CHECK(u.m == 2);
    CHECK(u.n == 2);
    CHECK(u.b == 8);
  }

  SECTION("grid_k=64, grid_b=1, wgmxcc=1 — round-robin strided across k-splits") {
    // Grid: 2×2×64×1, wgm=1, wgmxcc=1
    // stride=8 cycles through k: unique_k = 64/gcd(8,64) = 8
    auto u = origami::gemm::count_unique_tiles({64, 2, 2, 1}, {0, 0, 1, 1}, N_CU, num_xcd, 0, 0);
    CHECK(u.k == 8);
    CHECK(u.m == 2);
    CHECK(u.n == 2);
    CHECK(u.b == 1);
  }

  SECTION("grid_k=64, grid_b=1, wgmxcc=8 — contiguous block within one mn tile") {
    // Grid: 2×2×64×1, wgm=1, wgmxcc=8
    // Each XCD gets 32 contiguous tiles -> 32 k-splits in mn_id=0
    auto u = origami::gemm::count_unique_tiles({64, 2, 2, 1}, {0, 0, 8, 1}, N_CU, num_xcd, 0, 0);
    CHECK(u.k == 32);
    CHECK(u.m == 1);
    CHECK(u.n == 1);
    CHECK(u.b == 1);
  }

  SECTION("16×16 grid, wgmxcc=8, wgm=4 — contiguous block in first WGM slab") {
    // Grid: 16×16×1×1, wgm=4, wgmxcc=8
    // 32 contiguous tiles → 8 m-rows × 4 n-columns (one slab)
    auto u = origami::gemm::count_unique_tiles({1, 16, 16, 1}, {0, 0, 8, 4}, N_CU, num_xcd, 0, 0);
    CHECK(u.k == 1);
    CHECK(u.m == 8);
    CHECK(u.n == 4);
    CHECK(u.b == 1);
  }

  SECTION("8×8 grid with k=4, wgmxcc=8, wgm=2 — mixed k and mn") {
    // Grid: 8×8×4×1, wgm=2, wgmxcc=8
    // 32 contiguous tiles → 4 k-splits × 4 m-rows × 2 n-columns
    auto u = origami::gemm::count_unique_tiles({4, 8, 8, 1}, {0, 0, 8, 2}, N_CU, num_xcd, 0, 0);
    CHECK(u.k == 4);
    CHECK(u.m == 4);
    CHECK(u.n == 2);
    CHECK(u.b == 1);
  }

  SECTION("All XCDs report the same unique counts") {
    // With wgmxcc=8, all XCDs should see the same tile structure
    for (size_t xcd = 0; xcd < num_xcd; ++xcd) {
      auto u = origami::gemm::count_unique_tiles({1, 16, 16, 1}, {0, 0, 8, 4}, N_CU, num_xcd, xcd, 0);
      CHECK(u.k == 1);
      CHECK(u.m == 8);
      CHECK(u.n == 4);
      CHECK(u.b == 1);
    }
  }

  SECTION("Zero/degenerate inputs return zeros") {
    CHECK(origami::gemm::count_unique_tiles({1, 4, 4, 1}, {0, 0, 8, 4}, 0, 8, 0, 0).m == 0);
    CHECK(origami::gemm::count_unique_tiles({1, 4, 4, 1}, {0, 0, 8, 4}, 256, 0, 0, 0).m == 0);
    CHECK(origami::gemm::count_unique_tiles({1, 0, 4, 1}, {0, 0, 8, 4}, 256, 8, 0, 0).m == 0);
    CHECK(origami::gemm::count_unique_tiles({0, 4, 4, 1}, {0, 0, 8, 4}, 256, 8, 0, 0).k == 0);
  }

  SECTION("Timestep beyond available tiles returns zeros") {
    // 16 tiles total, 256 CUs -> 1 timestep. Timestep 1 should be empty.
    origami::dim4_t grid{1, 4, 4, 1};
    auto u = origami::gemm::count_unique_tiles(grid, {0, 0, 1, 4}, N_CU, num_xcd, 0, 1);
    CHECK(u.m == 0);
    CHECK(u.n == 0);
    CHECK(u.k == 0);
    CHECK(u.b == 0);
  }

  SECTION("Single tile grid — all XCDs see at most 1 tile") {
    origami::dim4_t grid{1, 1, 1, 1};
    auto u0 = origami::gemm::count_unique_tiles(grid, {0, 0, 1, 1}, N_CU, num_xcd, 0, 0);
    CHECK(u0.m == 1);
    CHECK(u0.n == 1);
    CHECK(u0.k == 1);
    CHECK(u0.b == 1);
    // XCD 1 should get nothing (only 1 tile, XCD 0 gets it)
    auto u1 = origami::gemm::count_unique_tiles(grid, {0, 0, 1, 1}, N_CU, num_xcd, 1, 0);
    CHECK(u1.m == 0);
  }

  SECTION("Round-robin: large grid, all MN tiles covered per XCD") {
    // 32x32 grid = 1024 tiles, 256 CUs, 8 XCDs -> 32 tiles/XCD.
    // stride=8 across 1024 MN tiles -> each XCD sees many M and N values.
    origami::dim4_t grid{1, 32, 32, 1};
    auto u = origami::gemm::count_unique_tiles(grid, {0, 0, 1, 4}, N_CU, num_xcd, 0, 0);
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
    auto u = origami::gemm::count_unique_tiles(grid, {0, 0, 1, 4}, N_CU, num_xcd, 0, 0);
    CHECK(u.k == 1);
    CHECK(u.m >= 1);
    CHECK(u.n >= 1);
  }

  SECTION("Round-robin: stride coprime with grid.k") {
    // grid.k=3, stride=8 -> gcd(8,3)=1, unique_k = 3/1 = 3 (all K-splits).
    origami::dim4_t grid{3, 4, 4, 1};
    auto u = origami::gemm::count_unique_tiles(grid, {0, 0, 1, 4}, N_CU, num_xcd, 0, 0);
    CHECK(u.k == 3);
  }

  SECTION("WGMXCC: last XCD gets correct tiles") {
    // 256 tiles, 8 XCDs -> 32 per XCD. Last XCD starts at 7*32=224.
    origami::dim4_t grid{1, 16, 16, 1};
    auto u_first = origami::gemm::count_unique_tiles(grid, {0, 0, 8, 4}, N_CU, num_xcd, 0, 0);
    auto u_last  = origami::gemm::count_unique_tiles(grid, {0, 0, 8, 4}, N_CU, num_xcd, 7, 0);
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
        {0, 0, 8, 1}, {0, 0, 8, 4}, {0, 0, 8, 8}, {0, 0, 1, 1}, {0, 0, 0, 6}};
    for (auto& g : grids) {
      for (auto& w : wgms) {
        for (size_t xcd = 0; xcd < num_xcd; ++xcd) {
          auto u = origami::gemm::count_unique_tiles(g, w, N_CU, num_xcd, xcd, 0);
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

  REQUIRE(defaults.weight_tile_total == 1.0);
  REQUIRE(defaults.main_loop_efficiency == 1.0);
  REQUIRE_FALSE(defaults.reject);
}

TEST_CASE("Heuristics: Parameter merging", "[heuristics]") {
  origami::heuristic_params_t base;
  origami::heuristic_params_t override;

  // Set a non-default value in override
  override.weight_tile_total = 2.0;

  // Merge override into base
  base.merge_with(override);

  // Check that the overridden value changed
  REQUIRE(base.weight_tile_total == 2.0);

  // Check that non-overridden values remain default
  REQUIRE(base.main_loop_efficiency == origami::heuristic_defaults_t::MAIN_LOOP_EFFICIENCY);
  REQUIRE_FALSE(base.reject);
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
  REQUIRE(params.weight_tile_total > 0.0);
  REQUIRE(params.main_loop_efficiency > 0.0);
  REQUIRE(params.main_loop_efficiency <= 1.0);
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

  REQUIRE(params.main_loop_efficiency == Approx(100.0 / 115.0).epsilon(1e-6));
}

TEST_CASE("Heuristics: CMS macrotile differentiation", "[heuristics]") {
  const double bf16_nt_160 =
      lookup_cms_main_loop_efficiency(origami::data_type_t::BFloat16,
                                        origami::transpose_t::N,
                                        origami::transpose_t::T,
                                        160,
                                        256,
                                        64);
  const double bf16_nt_192 =
      lookup_cms_main_loop_efficiency(origami::data_type_t::BFloat16,
                                        origami::transpose_t::N,
                                        origami::transpose_t::T,
                                        192,
                                        256,
                                        64);
  const double bf16_nt_256 =
      lookup_cms_main_loop_efficiency(origami::data_type_t::BFloat16,
                                        origami::transpose_t::N,
                                        origami::transpose_t::T,
                                        256,
                                        256,
                                        64);

  REQUIRE(bf16_nt_160 == Approx(100.0 / 120.0).epsilon(1e-6));
  REQUIRE(bf16_nt_192 == Approx(100.0 / 110.0).epsilon(1e-6));
  REQUIRE(bf16_nt_256 == Approx(100.0 / 115.0).epsilon(1e-6));
  REQUIRE(bf16_nt_160 != Approx(bf16_nt_192));
  REQUIRE(bf16_nt_192 != Approx(bf16_nt_256));

  const double tf32_tn_128 =
      lookup_cms_main_loop_efficiency(origami::data_type_t::XFloat32,
                                        origami::transpose_t::T,
                                        origami::transpose_t::N,
                                        128,
                                        256,
                                        32);
  const double tf32_tn_192 =
      lookup_cms_main_loop_efficiency(origami::data_type_t::XFloat32,
                                        origami::transpose_t::T,
                                        origami::transpose_t::N,
                                        192,
                                        256,
                                        32);

  REQUIRE(tf32_tn_128 == Approx(100.0 / 126.0).epsilon(1e-6));
  REQUIRE(tf32_tn_192 == Approx(100.0 / 123.0).epsilon(1e-6));
  REQUIRE(tf32_tn_128 != Approx(tf32_tn_192));
}

TEST_CASE("Heuristics: CMS dtype differentiation", "[heuristics]") {
  const double bf16_nt_160 =
      lookup_cms_main_loop_efficiency(origami::data_type_t::BFloat16,
                                        origami::transpose_t::N,
                                        origami::transpose_t::T,
                                        160,
                                        256,
                                        64);
  const double fp16_nt_160 = lookup_cms_main_loop_efficiency(origami::data_type_t::Half,
                                                               origami::transpose_t::N,
                                                               origami::transpose_t::T,
                                                               160,
                                                               256,
                                                               64);

  REQUIRE(bf16_nt_160 == Approx(100.0 / 120.0).epsilon(1e-6));
  REQUIRE(fp16_nt_160 == Approx(1.0).epsilon(1e-6));
  REQUIRE(bf16_nt_160 != Approx(fp16_nt_160));

  const double fp16_nt_unique =
      lookup_cms_main_loop_efficiency(origami::data_type_t::Half,
                                        origami::transpose_t::N,
                                        origami::transpose_t::T,
                                        192,
                                        320,
                                        64);
  const double bf16_nt_missing =
      lookup_cms_main_loop_efficiency(origami::data_type_t::BFloat16,
                                        origami::transpose_t::N,
                                        origami::transpose_t::T,
                                        192,
                                        320,
                                        64);

  REQUIRE(fp16_nt_unique == Approx(100.0 / 110.0).epsilon(1e-6));
  REQUIRE(bf16_nt_missing == Approx(1.0).epsilon(1e-6));
  REQUIRE(fp16_nt_unique != Approx(bf16_nt_missing));

  const double tf32_tn =
      lookup_cms_main_loop_efficiency(origami::data_type_t::XFloat32,
                                        origami::transpose_t::T,
                                        origami::transpose_t::N,
                                        128,
                                        256,
                                        32);
  const double bf16_tn =
      lookup_cms_main_loop_efficiency(origami::data_type_t::BFloat16,
                                        origami::transpose_t::T,
                                        origami::transpose_t::N,
                                        256,
                                        256,
                                        64);

  REQUIRE(tf32_tn == Approx(100.0 / 126.0).epsilon(1e-6));
  REQUIRE(bf16_tn == Approx(100.0 / 115.0).epsilon(1e-6));
  REQUIRE(tf32_tn != Approx(bf16_tn));
}

TEST_CASE("Heuristics: CMS unregistered hand-optimized kernel", "[heuristics]") {
  const double efficiency = lookup_cms_main_loop_efficiency(origami::data_type_t::BFloat16,
                                                              origami::transpose_t::N,
                                                              origami::transpose_t::T,
                                                              128,
                                                              256,
                                                              64);

  REQUIRE(efficiency == Approx(1.0).epsilon(1e-6));
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

  REQUIRE(params.main_loop_efficiency == Approx(100.0 / 115.0).epsilon(1e-6));
}

TEST_CASE("Heuristics: Database add_entry and lookup", "[heuristics]") {
  auto& db = origami::heuristics_database_t::get_instance();

  // Create a very specific custom heuristic entry that won't conflict
  origami::heuristic_key_t key;
  key.arch     = origami::hardware_t::architecture_t::gfx942;
  key.mi_dtype = origami::data_type_t::Int8;  // Unusual dtype
  key.mt_m     = 777;                         // Unique tile size

  origami::heuristic_params_t params;
  params.weight_tile_total = 7.77;  // Unique value

  db.add_entry(key, params);

  // Try to lookup this entry
  auto hardware    = make_hardware(942);
  auto problem     = make_problem(1024, 1024, 1024);
  auto config      = make_config(777, 128, 64, 16, 16, 16);
  problem.mi_dtype = origami::data_type_t::Int8;

  auto result = db.lookup(problem, hardware, config);

  // Should find our custom entry
  REQUIRE(result.weight_tile_total == 7.77);
}

TEST_CASE("Heuristics: Hierarchical lookup (most specific wins)", "[heuristics]") {
  auto& db = origami::heuristics_database_t::get_instance();

  // Add a general rule
  origami::heuristic_key_t general_key;
  general_key.arch     = origami::hardware_t::architecture_t::gfx942;
  general_key.mi_dtype = origami::data_type_t::Float;
  general_key.mt_n     = 144;  // Unique tile width so the entry can't leak into other tests

  origami::heuristic_params_t general_params;
  general_params.weight_tile_total = 3.33;

  db.add_entry(general_key, general_params);

  // Add a more specific rule
  origami::heuristic_key_t specific_key;
  specific_key.arch     = origami::hardware_t::architecture_t::gfx942;
  specific_key.mi_dtype = origami::data_type_t::Float;
  specific_key.mt_n     = 144;
  specific_key.mt_m     = 555;  // Unique value

  origami::heuristic_params_t specific_params;
  specific_params.weight_tile_total = 5.55;  // More specific value

  db.add_entry(specific_key, specific_params);

  // Lookup with matching specific case
  auto hardware    = make_hardware(942);
  auto problem     = make_problem(1024, 1024, 1024);
  auto config      = make_config(555, 144, 64, 16, 16, 16);
  problem.mi_dtype = origami::data_type_t::Float;

  auto result = db.lookup(problem, hardware, config);

  // Should use more specific rule
  REQUIRE(result.weight_tile_total == 5.55);
}

TEST_CASE("GEMM: compute_parallel_reduction_latency", "[gemm]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - returns zero when splitting_factor <= 1") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(4096, 4096, 1024);
      auto config   = make_config(128, 128, 64, 32, 32, 8, false, 1);

      origami::gemm::context_t ctx(problem, hardware, config);
      ctx.splitting_factor    = 1;
      ctx.reduction_strategy  = origami::reduction_t::parallel;

      auto latency = origami::gemm::compute_parallel_reduction_latency(problem, hardware, config, ctx);
      REQUIRE(latency == 0.0);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - returns zero for non-parallel reduction") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(4096, 4096, 1024);
      auto config   = make_config(128, 128, 64, 32, 32, 8, false, 1);

      origami::gemm::context_t ctx(problem, hardware, config);
      ctx.splitting_factor    = 4;
      ctx.reduction_strategy  = origami::reduction_t::none;

      auto latency = origami::gemm::compute_parallel_reduction_latency(problem, hardware, config, ctx);
      REQUIRE(latency == 0.0);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - positive latency for parallel split") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(4096, 4096, 8192);
      auto config   = make_config(128, 128, 64, 32, 32, 8, false, 1);

      origami::gemm::context_t ctx(problem, hardware, config);
      ctx.splitting_factor    = 4;
      ctx.reduction_strategy  = origami::reduction_t::parallel;

      auto latency = origami::gemm::compute_parallel_reduction_latency(problem, hardware, config, ctx);
      REQUIRE(latency > 0.0);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - higher split factor increases latency") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(4096, 4096, 8192);
      auto config   = make_config(128, 128, 64, 32, 32, 8, false, 1);

      origami::gemm::context_t ctx_lo(problem, hardware, config);
      ctx_lo.splitting_factor    = 2;
      ctx_lo.reduction_strategy  = origami::reduction_t::parallel;

      origami::gemm::context_t ctx_hi(problem, hardware, config);
      ctx_hi.splitting_factor    = 8;
      ctx_hi.reduction_strategy  = origami::reduction_t::parallel;

      auto lat_lo = origami::gemm::compute_parallel_reduction_latency(problem, hardware, config, ctx_lo);
      auto lat_hi = origami::gemm::compute_parallel_reduction_latency(problem, hardware, config, ctx_hi);
      REQUIRE(lat_hi > lat_lo);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - larger output increases latency") {
      auto hardware      = make_hardware(gpu_arch);
      auto problem_small = make_problem(1024, 1024, 8192);
      auto problem_large = make_problem(8192, 8192, 8192);
      auto config        = make_config(128, 128, 64, 32, 32, 8, false, 1);

      origami::gemm::context_t ctx_small(problem_small, hardware, config);
      ctx_small.splitting_factor    = 4;
      ctx_small.reduction_strategy  = origami::reduction_t::parallel;

      origami::gemm::context_t ctx_large(problem_large, hardware, config);
      ctx_large.splitting_factor    = 4;
      ctx_large.reduction_strategy  = origami::reduction_t::parallel;

      auto lat_small = origami::gemm::compute_parallel_reduction_latency(
          problem_small, hardware, config, ctx_small);
      auto lat_large = origami::gemm::compute_parallel_reduction_latency(
          problem_large, hardware, config, ctx_large);
      REQUIRE(lat_large > lat_small);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - batched problem scales latency") {
      auto hardware = make_hardware(gpu_arch);
      auto problem1 = make_problem(2048, 2048, 4096, origami::transpose_t::T, origami::transpose_t::N, 1);
      auto problem4 = make_problem(2048, 2048, 4096, origami::transpose_t::T, origami::transpose_t::N, 4);
      auto config   = make_config(128, 128, 64, 32, 32, 8, false, 1);

      origami::gemm::context_t ctx1(problem1, hardware, config);
      ctx1.splitting_factor    = 4;
      ctx1.reduction_strategy  = origami::reduction_t::parallel;

      origami::gemm::context_t ctx4(problem4, hardware, config);
      ctx4.splitting_factor    = 4;
      ctx4.reduction_strategy  = origami::reduction_t::parallel;

      auto lat1 = origami::gemm::compute_parallel_reduction_latency(problem1, hardware, config, ctx1);
      auto lat4 = origami::gemm::compute_parallel_reduction_latency(problem4, hardware, config, ctx4);
      REQUIRE(lat4 > lat1);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - positive when splitting") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(64, 64, 4096);
      auto config   = make_config(64, 64, 64, 32, 32, 8, false, 1);

      origami::gemm::context_t ctx(problem, hardware, config);
      ctx.splitting_factor    = 2;
      ctx.reduction_strategy  = origami::reduction_t::parallel;

      auto latency = origami::gemm::compute_parallel_reduction_latency(problem, hardware, config, ctx);
      REQUIRE(latency > 0.0);
    }
  }
}

TEST_CASE("GEMM: compute_epilogue_latency", "[gemm]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - positive for aligned interior tiles") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(4096, 4096, 1024);
      auto config   = make_config(128, 128, 64, 32, 32, 8, false, 1);

      origami::gemm::context_t ctx(problem, hardware, config);
      auto latency = origami::gemm::compute_epilogue_latency(problem, hardware, config, ctx);
      REQUIRE(latency > 0.0);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - edge tiles produce a valid epilogue cost") {
      auto hardware = make_hardware(gpu_arch);
      auto config   = make_config(128, 128, 64, 32, 32, 8, false, 1);

      auto problem_aligned = make_problem(4096, 4096, 1024);
      origami::gemm::context_t ctx_aligned(problem_aligned, hardware, config);
      auto lat_aligned =
          origami::gemm::compute_epilogue_latency(problem_aligned, hardware, config, ctx_aligned);

      auto problem_edge = make_problem(4097, 4097, 1024);
      origami::gemm::context_t ctx_edge(problem_edge, hardware, config);
      auto lat_edge = origami::gemm::compute_epilogue_latency(problem_edge, hardware, config, ctx_edge);

      // Ported model: the +1 remainder produces cheap partial edge tiles, so the
      // edge problem's modeled epilogue cost is no higher than the aligned one.
      REQUIRE(lat_edge > 0.0);
      REQUIRE(lat_edge <= lat_aligned);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - larger tiles have higher epilogue cost") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(4096, 4096, 1024);

      auto config_small = make_config(64, 64, 64, 32, 32, 8, false, 1);
      auto config_large = make_config(256, 256, 64, 32, 32, 8, false, 1);

      origami::gemm::context_t ctx_small(problem, hardware, config_small);
      origami::gemm::context_t ctx_large(problem, hardware, config_large);

      auto lat_small = origami::gemm::compute_epilogue_latency(problem, hardware, config_small, ctx_small);
      auto lat_large = origami::gemm::compute_epilogue_latency(problem, hardware, config_large, ctx_large);
      REQUIRE(lat_large > lat_small);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - serial split-K adds reduction cost") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(4096, 4096, 8192);
      auto config   = make_config(128, 128, 64, 32, 32, 8, false, 1);

      origami::gemm::context_t ctx_nosplit(problem, hardware, config);
      ctx_nosplit.splitting_factor   = 1;
      ctx_nosplit.reduction_strategy = origami::reduction_t::none;

      origami::gemm::context_t ctx_spinlock(problem, hardware, config);
      ctx_spinlock.splitting_factor   = 4;
      ctx_spinlock.reduction_strategy = origami::reduction_t::spinlock;

      auto lat_nosplit  = origami::gemm::compute_epilogue_latency(problem, hardware, config, ctx_nosplit);
      auto lat_spinlock = origami::gemm::compute_epilogue_latency(problem, hardware, config, ctx_spinlock);
      REQUIRE(lat_spinlock > lat_nosplit);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - parallel reduction skips in-kernel reduce") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(4096, 4096, 8192);
      auto config   = make_config(128, 128, 64, 32, 32, 8, false, 1);

      origami::gemm::context_t ctx_spinlock(problem, hardware, config);
      ctx_spinlock.splitting_factor   = 4;
      ctx_spinlock.reduction_strategy = origami::reduction_t::spinlock;

      origami::gemm::context_t ctx_parallel(problem, hardware, config);
      ctx_parallel.splitting_factor   = 4;
      ctx_parallel.reduction_strategy = origami::reduction_t::parallel;

      auto lat_spinlock = origami::gemm::compute_epilogue_latency(problem, hardware, config, ctx_spinlock);
      auto lat_parallel = origami::gemm::compute_epilogue_latency(problem, hardware, config, ctx_parallel);
      REQUIRE(lat_spinlock > lat_parallel);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - returns zero when d_bytes is zero") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(4096, 4096, 1024);
      auto config   = make_config(128, 128, 64, 32, 32, 8, false, 1);

      origami::gemm::context_t ctx(problem, hardware, config);
      ctx.d_bytes = 0;

      auto latency = origami::gemm::compute_epilogue_latency(problem, hardware, config, ctx);
      REQUIRE(latency == 0.0);
    }
  }
}

namespace {

// 22x22 = 484 tiles of 128x128, just over MIN_TILES_FOR_DYNAMIC (480), so the
// tile-count gate in select_hybrid_mode() lets the other gates decide.
inline origami::problem_t make_problem_above_dynamic_tile_gate() {
  return make_problem(/*m=*/128 * 22, /*n=*/128 * 22, /*k=*/4096);
}

}  // namespace

TEST_CASE("GEMM: context_t::tile_schedule records the StreamK sub-path", "[gemm][hybrid]") {
  // gfx950 with a cotenant holding CUs away, a grid above the tile gate and low
  // occupancy is the case the SK4 work-queue path exists for.
  auto hardware = make_hardware(950);
  auto config   = make_config(128, 128, 64, 32, 32, 8, false, 1, /*occupancy=*/1);

  SECTION("cotenant present on gfx950 selects the dynamic work queue") {
    auto problem     = make_problem_above_dynamic_tile_gate();
    problem.num_cus  = hardware.N_CU / 2;
    origami::gemm::context_t ctx(problem, hardware, config);
    REQUIRE(ctx.tile_schedule == origami::hybrid_mode_t::dynamic);
  }

  SECTION("no CU budget means no cotenant to rebalance against, so static") {
    auto problem    = make_problem_above_dynamic_tile_gate();
    problem.num_cus = 0;
    origami::gemm::context_t ctx(problem, hardware, config);
    REQUIRE(ctx.tile_schedule == origami::hybrid_mode_t::static_);
  }

  SECTION("grid at or below the tile gate stays static even with a cotenant") {
    auto problem    = make_problem(128, 128, 4096);
    problem.num_cus = hardware.N_CU / 2;
    origami::gemm::context_t ctx(problem, hardware, config);
    REQUIRE(ctx.tile_schedule == origami::hybrid_mode_t::static_);
  }

  SECTION("grid selection is a separate axis and does not move the sub-path") {
    // select_hybrid_mode() never reads config.grid_selection, and hipBLASLt
    // settles the sub-path before grid sizing consumes it: in
    // ContractionSolution the streamK5EffectiveDynamic() result is computed
    // first, and skDynamicGrid (which is cast straight to grid_selection_t) is
    // only honoured on the static branch. A data-parallel grid therefore has to
    // leave the answer alone.
    auto problem    = make_problem_above_dynamic_tile_gate();
    problem.num_cus = hardware.N_CU / 2;
    origami::gemm::context_t ctx_k_split(problem, hardware, config);

    auto dp_config           = config;
    dp_config.grid_selection = origami::grid_selection_t::data_parallel;
    origami::gemm::context_t ctx_dp(problem, hardware, dp_config);

    REQUIRE(ctx_dp.tile_schedule == ctx_k_split.tile_schedule);
    // Pinned so the comparison above cannot pass by both sides degrading.
    REQUIRE(ctx_dp.tile_schedule == origami::hybrid_mode_t::dynamic);
  }

  SECTION("agrees with select_hybrid_mode fed problem.num_cus as sm_count_target") {
    // Pins the source of truth: hipBLASLt's streamK5EffectiveDynamic() passes
    // smCountTarget() both as problem.num_cus and as sm_count_target, so the
    // context must not diverge from the heuristic it delegates to.
    for (size_t num_cus : {size_t{0}, hardware.N_CU / 4, hardware.N_CU / 2, hardware.N_CU}) {
      DYNAMIC_SECTION("num_cus=" << num_cus) {
        auto problem    = make_problem_above_dynamic_tile_gate();
        problem.num_cus = num_cus;
        origami::gemm::context_t ctx(problem, hardware, config);
        REQUIRE(ctx.tile_schedule
                == origami::streamk::select_hybrid_mode(problem, hardware, config, num_cus));
      }
    }
  }
}

TEST_CASE("GEMM: context_t::tile_schedule reports static on untuned architectures",
          "[gemm][hybrid]") {
  // select_hybrid_mode() is only fit on gfx950 and answers static_ elsewhere,
  // which is what those kernels really launch: streamK5EffectiveDynamic() feeds
  // that same return value straight into its effective-dynamic decision. The
  // context reports it verbatim rather than downgrading it to none, so that when
  // another architecture is tuned the real answer flows through unchanged.
  for (int gpu_arch : {942, 1250}) {
    DYNAMIC_SECTION("gfx" << gpu_arch) {
      auto hardware   = make_hardware(gpu_arch);
      auto config     = make_config(128, 128, 64, 32, 32, 8, false, 1, /*occupancy=*/1);
      auto problem    = make_problem_above_dynamic_tile_gate();
      problem.num_cus = hardware.N_CU / 2;

      origami::gemm::context_t ctx(problem, hardware, config);
      REQUIRE(ctx.tile_schedule == origami::hybrid_mode_t::static_);
      REQUIRE(ctx.tile_schedule
              == origami::streamk::select_hybrid_mode(
                     problem, hardware, config, problem.num_cus));
    }
  }
}

TEST_CASE("GEMM: context_t::tile_schedule is none only before construction", "[gemm][hybrid]") {
  // A constructed context always carries the heuristic's answer, so none is left
  // to mean "no problem examined yet".
  origami::gemm::context_t ctx;
  REQUIRE(ctx.tile_schedule == origami::hybrid_mode_t::none);
}

// Skinny TN shapes hit the non-temporal short circuit. The filter may reject a
// temporal kernel only when the caller says a non-temporal variant exists.
// Square problems, and K that misses the 128B line, never use non-temporal.
TEST_CASE("GEMM: non-temporal shortcut honors variant availability", "[gemm][cache-hints]") {
  const double rejected = std::numeric_limits<double>::max();

  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - operand B") {
      auto hardware = make_hardware(gpu_arch);
      // M <= 2*MT_M, B is not transposed, and N/M > 5, so B must be non-temporal.
      auto problem =
          make_problem(256, 8192, 4096, origami::transpose_t::T, origami::transpose_t::N);
      auto temporal    = make_config(128, 256, 64, 16, 16, 16, false, 1, 1, 0, 0);
      auto nontemporal = make_config(128, 256, 64, 16, 16, 16, false, 1, 1, 0, 4);

      temporal.non_temporal_b_available = true;
      REQUIRE(origami::gemm::compute_total_latency(problem, hardware, temporal) == rejected);
      temporal.non_temporal_b_available = false;
      REQUIRE(origami::gemm::compute_total_latency(problem, hardware, temporal) < rejected);
      REQUIRE(origami::gemm::compute_total_latency(problem, hardware, nontemporal) < rejected);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - operand A") {
      auto hardware = make_hardware(gpu_arch);
      // N <= 2*MT_N, A is transposed, and M/N > 5, so A must be non-temporal.
      auto problem =
          make_problem(8192, 256, 4096, origami::transpose_t::T, origami::transpose_t::N);
      auto temporal    = make_config(256, 128, 64, 16, 16, 16, false, 1, 1, 0, 0);
      auto nontemporal = make_config(256, 128, 64, 16, 16, 16, false, 1, 1, 4, 0);

      temporal.non_temporal_a_available = true;
      REQUIRE(origami::gemm::compute_total_latency(problem, hardware, temporal) == rejected);
      temporal.non_temporal_a_available = false;
      REQUIRE(origami::gemm::compute_total_latency(problem, hardware, temporal) < rejected);
      REQUIRE(origami::gemm::compute_total_latency(problem, hardware, nontemporal) < rejected);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - square problems never use non-temporal") {
      auto hardware = make_hardware(gpu_arch);
      // Neither skinny branch applies, so the "never use Non Temporal" arm runs.
      auto problem =
          make_problem(4096, 4096, 4096, origami::transpose_t::T, origami::transpose_t::N);
      auto temporal    = make_config(128, 128, 64, 16, 16, 16, false, 1, 1, 0, 0);
      auto nontemporal = make_config(128, 128, 64, 16, 16, 16, false, 1, 1, 0, 4);

      REQUIRE(origami::gemm::compute_total_latency(problem, hardware, nontemporal) == rejected);
      REQUIRE(origami::gemm::compute_total_latency(problem, hardware, temporal) < rejected);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - misaligned K rejects non-temporal") {
      auto hardware = make_hardware(gpu_arch);
      // K * 16 bits is not a multiple of the 128B line, so the trailing
      // else-if rejects any non-temporal hint. A cached read still runs.
      auto problem = make_problem(256, 8192, 12, origami::transpose_t::T, origami::transpose_t::N);
      auto temporal    = make_config(128, 256, 64, 16, 16, 16, false, 1, 1, 0, 0);
      auto nontemporal = make_config(128, 256, 64, 16, 16, 16, false, 1, 1, 0, 4);

      REQUIRE(origami::gemm::compute_total_latency(problem, hardware, nontemporal) == rejected);
      REQUIRE(origami::gemm::compute_total_latency(problem, hardware, temporal) < rejected);
    }
  }
}

TEST_CASE("GEMM: hybrid_mode_to_string", "[gemm][hybrid]") {
  REQUIRE(origami::hybrid_mode_to_string(origami::hybrid_mode_t::static_) == "static");
  REQUIRE(origami::hybrid_mode_to_string(origami::hybrid_mode_t::dynamic) == "dynamic");
  REQUIRE(origami::hybrid_mode_to_string(origami::hybrid_mode_t::none) == "none");
}

// Verify skinny-M TN and skinny-N NT predicted latencies are finite and not
// wildly inflated relative to equivalent NN shapes after the L1 headroom fix.
//
// Shape selection rationale (to satisfy the short-circuit guard):
//   TN skinny-M: M=256 (grid_m=2), N=8192 (grid_n=32) → skinny_m fires.
//     Short-circuit sees M<=MT_M*2 && !b_trans && N/M>5 → forces hints_b=4
//     for both TN and NN, so both configs use hints_b=4.
//   NT skinny-N: M=8192 (grid_m=32), N=256 (grid_n=2) → skinny_n fires.
//     No NT condition fires for NT layout → hints_a=hints_b=0 for both.
TEST_CASE("GEMM: skinny TN/NT L1 headroom fix", "[gemm]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch) {
      auto hw = make_hardware(gpu_arch);

      // skinny-M TN vs NN: both use hints_b=4 (short-circuit requirement).
      auto prob_tn = make_problem(256, 8192, 4096, origami::transpose_t::T, origami::transpose_t::N);
      auto prob_nn = make_problem(256, 8192, 4096, origami::transpose_t::N, origami::transpose_t::N);
      // make_config: mt_m, mt_n, mt_k, mi_m, mi_n, mi_k, hand_opt, wgm, occ, hints_a, hints_b
      auto cfg_skinny_m = make_config(128, 256, 64, 16, 16, 16, false, 1, 1, 0, /*hints_b=*/4);

      double lat_tn = origami::gemm::compute_total_latency(prob_tn, hw, cfg_skinny_m);
      double lat_nn = origami::gemm::compute_total_latency(prob_nn, hw, cfg_skinny_m);

      INFO("TN lat=" << lat_tn << "  NN lat=" << lat_nn << "  ratio=" << lat_tn / lat_nn);
      REQUIRE(lat_tn < std::numeric_limits<double>::max());
      REQUIRE(lat_nn < std::numeric_limits<double>::max());
      REQUIRE(lat_tn / lat_nn < 2.0);

      // skinny-N NT vs NN: hints=0 for both (no NT short-circuit fires for NT layout).
      auto prob_nt   = make_problem(8192, 256, 4096, origami::transpose_t::N, origami::transpose_t::T);
      auto prob_nn_n = make_problem(8192, 256, 4096, origami::transpose_t::N, origami::transpose_t::N);
      auto cfg_skinny_n = make_config(256, 128, 64, 16, 16, 16, false, 1, 1, 0, 0);

      double lat_nt   = origami::gemm::compute_total_latency(prob_nt, hw, cfg_skinny_n);
      double lat_nn_n = origami::gemm::compute_total_latency(prob_nn_n, hw, cfg_skinny_n);

      INFO("NT lat=" << lat_nt << "  NN lat=" << lat_nn_n << "  ratio=" << lat_nt / lat_nn_n);
      REQUIRE(lat_nt < std::numeric_limits<double>::max());
      REQUIRE(lat_nn_n < std::numeric_limits<double>::max());
      REQUIRE(lat_nt / lat_nn_n < 2.0);
    }
  }
}

// PrefetchGlobalRead directional truth: PGR2 double-buffers the global->LDS
// prefetch, overlapping the next K-tile's loads behind the current tile's
// MFMA.  Physically that pays off only once the main loop is deep enough to
// amortize the fill/drain; on a single-iter (or k_iters <= pgr) loop the extra
// buffer is pure overhead.  Grid pinned to data-parallel (stream_k=0) so the
// delta is a pure K-loop effect, not a StreamK grid change.
TEST_CASE("GEMM: PrefetchGlobalRead PGR2 vs PGR1 latency ordering", "[gemm][pgr]") {
  for (int gpu_arch : test_architectures) {
    auto hw = make_hardware(gpu_arch);
    // make_config(..., wgm, occ, hints_a, hints_b, stream_k); stream_k=0 -> data-parallel.
    auto cfg = make_config(128, 128, 64, 16, 16, 16, false, 1, 1, 0, 0, 0);

    // Latency of `cfg` with prefetch_global_read forced to `pgr`, everything
    // else held constant -- the pure PGR delta for this (size, config).
    auto latency_with_pgr = [&](origami::problem_t p, int pgr) {
      auto c                           = cfg;
      c.tensile().prefetch_global_read = pgr;
      return origami::gemm::compute_total_latency(p, hw, c);
    };

    DYNAMIC_SECTION("gfx" << gpu_arch << " - deep K (128 iters): PGR2 wins") {
      auto p = make_problem(4096, 4096, 8192);
      double l1 = latency_with_pgr(p, 1);
      double l2 = latency_with_pgr(p, 2);
      INFO("PGR1=" << l1 << "  PGR2=" << l2);
      REQUIRE(l2 < l1);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - single K iter: PGR2 not better") {
      auto p = make_problem(4096, 4096, 64);
      double l1 = latency_with_pgr(p, 1);
      double l2 = latency_with_pgr(p, 2);
      INFO("PGR1=" << l1 << "  PGR2=" << l2);
      REQUIRE(l1 <= l2);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - k_iters <= pgr (2 iters): PGR2 not better") {
      auto p = make_problem(4096, 4096, 128);
      double l1 = latency_with_pgr(p, 1);
      double l2 = latency_with_pgr(p, 2);
      INFO("PGR1=" << l1 << "  PGR2=" << l2);
      REQUIRE(l1 <= l2);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - mid K (16 iters): PGR2 amortized") {
      auto p = make_problem(4096, 4096, 1024);
      double l1 = latency_with_pgr(p, 1);
      double l2 = latency_with_pgr(p, 2);
      INFO("PGR1=" << l1 << "  PGR2=" << l2);
      REQUIRE(l2 <= l1);
    }
  }
}

// Single K iteration (k_iters == 1, no tail): PGR>1 has no next tile to
// prefetch, so its double-buffer is pure overhead (extra registers + an
// unamortized fill/drain).  PGR1 must strictly win.  The model expresses this
// via L_pgr_stall, which charges PGR2 an unamortized fill that PGR1 skips.
TEST_CASE("GEMM: PGR1 wins for single-K-iter tiles", "[gemm][pgr]") {
  for (int gpu_arch : test_architectures) {
    auto hw = make_hardware(gpu_arch);

    // K == MT_K => exactly one full K-iter and no tail.  stream_k=0 isolates
    // the PGR delta on a data-parallel grid.
    auto require_pgr1_wins = [&](size_t mt_k, origami::transpose_t ta, origami::transpose_t tb) {
      auto c = make_config(128, 128, mt_k, 16, 16, 16, false, 1, 1, 0, 0, 0);
      auto p = make_problem(4096, 4096, mt_k, ta, tb);
      auto lat = [&](int pgr) {
        c.tensile().prefetch_global_read = pgr;
        return origami::gemm::compute_total_latency(p, hw, c);
      };
      double l1 = lat(1), l2 = lat(2);
      INFO("MT_K=" << mt_k << "  PGR1=" << l1 << "  PGR2=" << l2);
      REQUIRE(l1 < l2);
    };

    DYNAMIC_SECTION("gfx" << gpu_arch << " - single-iter PGR1 strictly wins") {
      using T = origami::transpose_t;
      require_pgr1_wins(32,  T::T, T::N);   // TN, shallow DepthU
      require_pgr1_wins(64,  T::T, T::N);   // TN
      require_pgr1_wins(128, T::T, T::N);   // TN, deep DepthU
      require_pgr1_wins(64,  T::N, T::T);   // NT
      require_pgr1_wins(64,  T::N, T::N);   // NN
    }
  }
}

TEST_CASE("GEMM: one LDS buffer exposes the next tile's writes once they outlast a K-step",
          "[gemm][lds]") {
  // bf16 MT128x128, MI16x16x32 on 2x2 waves: a K-step is 16 MFMAs of 16 cycles, and writing the
  // next tile takes 256 cycles at DepthU 64 and 512 at DepthU 128.
  auto latency = [](size_t depth_u, bool one_buffer) {
    auto hw                    = make_hardware(950);
    auto c                     = make_config(128, 128, depth_u, 16, 16, 32, false, 1, 1, 0, 0, 0);
    c.tensile().one_lds_buffer = one_buffer;
    return origami::gemm::compute_total_latency(make_problem(128, 8192, 8192), hw, c);
  };

  SECTION("writes that fit in the last K-step cost nothing") {
    REQUIRE(latency(64, true) == latency(64, false));
  }

  SECTION("writes that outlast it are exposed") {
    REQUIRE(latency(128, true) > latency(128, false));
  }
}

TEST_CASE("GEMM: SourceSwap changes only epilogue store traffic", "[gemm][sourceswap]") {
  for (int gpu_arch : test_architectures) {
    auto hw             = make_hardware(gpu_arch);
    auto hw_issue_bound = make_hardware(gpu_arch);
    hw_issue_bound.mem3_perf_ratio *= 1.0e9;

    auto epi_with_swap =
        [&](const origami::hardware_t& h, origami::problem_t p, origami::config_t c, bool swap) {
          c.tensile().source_swap = swap;
          origami::gemm::context_t ctx(p, h, c);
          return origami::gemm::compute_epilogue_latency(p, h, c, ctx);
        };

    DYNAMIC_SECTION("gfx" << gpu_arch << " - same store issue with and without SourceSwap") {
      for (size_t svw : {1, 4, 8}) {
        auto c   = make_config(128, 128, 64, 16, 16, 16, false, 1, 1, 0, 0, 0);
        c.gwvw_d = svw;
        for (auto p : {make_problem(4096, 4096, 1024), make_problem(4100, 4090, 1024)}) {
          INFO("svw=" << svw << " M=" << p.size.m << " N=" << p.size.n);
          REQUIRE(epi_with_swap(hw_issue_bound, p, c, false) ==
                  Approx(epi_with_swap(hw_issue_bound, p, c, true)).epsilon(1e-3));
        }
      }
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - scattered non-SourceSwap stores cost more traffic") {
      auto c   = make_config(128, 128, 64, 16, 16, 16, false, 1, 1, 0, 0, 0);
      c.gwvw_d = 4;
      auto p   = make_problem(4096, 4096, 1024);
      REQUIRE(epi_with_swap(hw, p, c, false) > epi_with_swap(hw, p, c, true));
    }
  }
}

// With store bandwidth removed, the epilogue is store-issue-bound: a wider gwvw_d
// issues fewer stores until one buffer_store (16 B) is full, and only a store
// wider than that splits into several.
TEST_CASE("GEMM: epilogue store issue follows gwvw_d", "[gemm][sourceswap]") {
  for (int gpu_arch : test_architectures) {
    auto hw = make_hardware(gpu_arch);
    hw.mem3_perf_ratio *= 1.0e6;

    auto lat_epi = [&](origami::problem_t p, size_t svw) {
      // make_config: mt_m, mt_n, mt_k, mi_m, mi_n, mi_k, hand_opt, wgm, occ,
      //              hints_a, hints_b, stream_k
      auto c   = make_config(256, 256, 32, 16, 16, 16, false, 1, 2, 0, 0, 0);
      c.gwvw_d = svw;
      origami::gemm::context_t ctx(p, hw, c);
      return origami::gemm::compute_epilogue_latency(p, hw, c, ctx);
    };

    DYNAMIC_SECTION("gfx" << gpu_arch << " - bf16: 8 elements fill one store") {
      auto p = make_problem(4096, 4096, 128);
      INFO("svw1=" << lat_epi(p, 1) << " svw2=" << lat_epi(p, 2) << " svw4=" << lat_epi(p, 4)
                   << " svw8=" << lat_epi(p, 8));
      REQUIRE(lat_epi(p, 2) < lat_epi(p, 1));
      REQUIRE(lat_epi(p, 4) < lat_epi(p, 2));
      REQUIRE(lat_epi(p, 8) < lat_epi(p, 4));
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - f32: 8 elements split into two stores") {
      auto p = make_problem(4096, 4096, 128);
      set_problem_dtypes(p, origami::data_type_t::Float);
      INFO("svw4=" << lat_epi(p, 4) << " svw8=" << lat_epi(p, 8));
      REQUIRE(lat_epi(p, 8) > lat_epi(p, 4));
    }
  }
}

// Wave-group layout controls how many epilogue waves run per SIMD batch.
// simds_per_cu=4 for CDNA; wave_batches = ceil(wave_num / min(wave_num, 4)).
//
// Key invariants from the wave_batches fix (gemm.cpp:1778):
//   - Within one batch (waves <= 4): doubling waves halves epilogue (pure parallelism).
//   - Doubling into a new batch (e.g. 4→8): same batch count, same epilogue as the
//     un-doubled config because the extra waves fill the new batch at equal cost.
//   - Same wave count, different layout (1x3 vs 3x1): epilogue equal (wave_range
//     partitions MT_M/MT_N identically, only total count matters for batching).
TEST_CASE("GEMM: epilogue wave-group batch scheduling", "[gemm][wavebatch]") {
  for (int gpu_arch : test_architectures) {
    auto hw = make_hardware(gpu_arch);

    auto epi_wg = [&](int wg_m, int wg_n) {
      auto c = make_config(128, 128, 64, 16, 16, 16, false, 1, 1, 0, 0, 0);
      c.tensile().wave_group_m = wg_m;
      c.tensile().wave_group_n = wg_n;
      auto p = make_problem(4096, 4096, 1024);
      origami::gemm::context_t ctx(p, hw, c);
      return origami::gemm::compute_epilogue_latency(p, hw, c, ctx);
    };

    DYNAMIC_SECTION("gfx" << gpu_arch << " - doubling waves within one batch halves epilogue") {
      double e1 = epi_wg(1, 1);  // 1 wave,  batch=1
      double e2 = epi_wg(2, 1);  // 2 waves, batch=1
      double e4 = epi_wg(2, 2);  // 4 waves, batch=1
      INFO("1w=" << e1 << "  2w=" << e2 << "  4w=" << e4);
      REQUIRE(e2 < e1);
      REQUIRE(e4 < e2);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - 4-wave and 8-wave equal (same batch count, double parallelism)") {
      // 8-wave: wave_batches=2, wave_issue_parallelism=4.
      // 4-wave: wave_batches=1, wave_issue_parallelism=4.
      // critical_path(8w) = max(2×max_wave_8w, total_8w/4)
      //                   = max(2×C/2,  8×(C/2)/4) = max(C, C) = C  (same as 4w).
      double e4 = epi_wg(2, 2);  // 4 waves
      double e8 = epi_wg(4, 2);  // 8 waves
      INFO("4w=" << e4 << "  8w=" << e8);
      REQUIRE(std::abs(e4 - e8) / e4 < 0.01);  // within 1%
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - wave layout (1x4 vs 4x1 vs 2x2) does not affect epilogue cost") {
      double e_1x4 = epi_wg(1, 4);
      double e_4x1 = epi_wg(4, 1);
      double e_2x2 = epi_wg(2, 2);
      INFO("1x4=" << e_1x4 << "  4x1=" << e_4x1 << "  2x2=" << e_2x2);
      REQUIRE(std::abs(e_1x4 - e_4x1) / e_1x4 < 0.01);
      REQUIRE(std::abs(e_1x4 - e_2x2) / e_1x4 < 0.01);
    }
  }
}

// TensileLite reads an operand VectorWidth elements per lane only when its M/N
// axis is contiguous in LDS (lrvwTile); K-contiguous, LDS-transpose and
// DirectToVgpr operands must not depend on VectorWidth.
TEST_CASE("GEMM: local-read latency follows the TensileLite LDS layout", "[gemm][localread]") {
  using origami::transpose_t;
  using arch_t = origami::hardware_t::architecture_t;

  REQUIRE(origami::hardware_t::get_lds_read_bytes_per_clk(arch_t::gfx942) == 128);
  REQUIRE(origami::hardware_t::get_lds_read_bytes_per_clk(arch_t::gfx950) == 256);

  for (int gpu_arch : test_architectures) {
    auto hw  = make_hardware(gpu_arch);
    auto cfg = [](int vwa, int vwb) {
      auto c           = make_config(64, 64, 64);
      c.vector_width_a = vwa;
      c.vector_width_b = vwb;
      return c;
    };
    auto lr = [&](const origami::problem_t& p, const origami::config_t& c) {
      return origami::gemm::compute_local_read_latency(p, hw, c, false);
    };
    auto lr_floor = [&](const origami::problem_t& p, const origami::config_t& c) {
      return origami::gemm::compute_local_read_latency(p, hw, c, true);
    };
    const auto tn = make_problem(4096, 4096, 4096, transpose_t::T, transpose_t::N);
    const auto nn = make_problem(4096, 4096, 4096, transpose_t::N, transpose_t::N);
    const auto nt = make_problem(4096, 4096, 4096, transpose_t::N, transpose_t::T);

    DYNAMIC_SECTION("gfx" << gpu_arch << " - TN reads both operands along K") {
      REQUIRE(lr(tn, cfg(1, 1)) == Approx(lr(tn, cfg(8, 8))));
      REQUIRE(lr(tn, cfg(1, 1)) == Approx(lr_floor(tn, cfg(1, 1))));
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - NN: only the M-contiguous A depends on VW") {
      REQUIRE(lr(nn, cfg(8, 1)) == Approx(lr(nn, cfg(8, 8))));
      // bf16 at VW=1 reads 2 B per lane, under one 4 B bank; VW=2 fills it.
      REQUIRE(lr(nn, cfg(1, 8)) > lr(nn, cfg(2, 8)));
      REQUIRE(lr(nn, cfg(2, 8)) == Approx(lr(nn, cfg(8, 8))));
      REQUIRE(lr(nn, cfg(8, 8)) == Approx(lr_floor(nn, cfg(1, 1))));
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - NT: both operands depend on VW") {
      REQUIRE(lr(nt, cfg(1, 8)) > lr(nt, cfg(8, 8)));
      REQUIRE(lr(nt, cfg(8, 1)) > lr(nt, cfg(8, 8)));
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - element size sets the VW that fills a bank") {
      auto nn_f32 = nn;
      set_problem_dtypes(nn_f32, origami::data_type_t::Float);
      REQUIRE(lr(nn_f32, cfg(1, 1)) == Approx(lr(nn_f32, cfg(4, 4))));

      auto nn_f8 = nn;
      set_problem_dtypes(nn_f8, origami::data_type_t::Float8);
      REQUIRE(lr(nn_f8, cfg(2, 8)) > lr(nn_f8, cfg(4, 8)));
      REQUIRE(lr(nn_f8, cfg(4, 8)) == Approx(lr(nn_f8, cfg(16, 8))));
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - plumbed layout flags override the transpose default") {
      auto unroll_a = [&](int vwa) {
        auto c                         = cfg(vwa, 8);
        c.tensile().unroll_major_lds_a = 1;
        return c;
      };
      auto tile_a = [&](int vwa) {
        auto c                         = cfg(vwa, 8);
        c.tensile().unroll_major_lds_a = 0;
        return c;
      };
      auto ldstr_a = [&](int vwa) {
        auto c                           = cfg(vwa, 8);
        c.tensile().lds_transpose_read_a = true;
        return c;
      };
      auto dtv_a = [&](int vwa) {
        auto c                       = cfg(vwa, 8);
        c.tensile().direct_to_vgpr_a = true;
        return c;
      };
      REQUIRE(lr(nn, unroll_a(1)) == Approx(lr(nn, unroll_a(8))));
      REQUIRE(lr(tn, tile_a(1)) > lr(tn, tile_a(8)));
      REQUIRE(lr(nn, ldstr_a(1)) == Approx(lr(nn, cfg(8, 8))));
      REQUIRE(lr(nn, dtv_a(1)) == Approx(lr(nn, dtv_a(8))));
      REQUIRE(lr(nn, dtv_a(1)) < lr(nn, cfg(8, 8)));
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - read time scales with MT_K") {
      auto c64  = cfg(1, 1);
      auto c128 = cfg(1, 1);
      c128.mt.k = 128;
      REQUIRE(lr(nn, c128) == Approx(2.0 * lr(nn, c64)));
    }
  }
}

// The LDS pipe feeds the MFMAs, so narrow reads may only cost tile latency when
// the LDS time outlasts both the MFMA work and the full-width floor.
TEST_CASE("GEMM: narrow local reads cost latency only when LDS-bound", "[gemm][localread]") {
  using origami::transpose_t;
  for (int gpu_arch : test_architectures) {
    auto hw = make_hardware(gpu_arch);
    // get_hardware_for_arch models four MFMA pipes per CDNA CU.
    hw.parallel_mi_cu = 4;
    const size_t mi_k = (gpu_arch == 942) ? 16 : 32;
    const auto nn     = make_problem(4096, 4096, 4096, transpose_t::N, transpose_t::N);
    const auto tn     = make_problem(4096, 4096, 4096, transpose_t::T, transpose_t::N);

    auto tile =
        [&](const origami::problem_t& p, size_t mt_m, size_t mt_n, int wg_m, int wg_n, int vwa) {
          auto c                   = make_config(mt_m, mt_n, 64, 16, 16, mi_k);
          c.vector_width_a         = vwa;
          c.vector_width_b         = 8;
          c.tensile().wave_group_m = wg_m;
          c.tensile().wave_group_n = wg_n;
          origami::gemm::context_t ctx(p, hw, c);
          return origami::gemm::compute_tile_latency(p, hw, c, ctx);
        };

    DYNAMIC_SECTION("gfx" << gpu_arch << " - TN tile latency ignores VW") {
      REQUIRE(tile(tn, 16, 64, 1, 4, 1) == tile(tn, 16, 64, 1, 4, 8));
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - MFMA-bound NN tile ignores VW") {
      REQUIRE(tile(nn, 256, 256, 2, 2, 1) == tile(nn, 256, 256, 2, 2, 8));
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - LDS-bound NN tile pays for VW=1") {
      // Four wave columns each re-read the 16-row A slice, so LDS outlasts the MFMAs.
      REQUIRE(tile(nn, 16, 64, 1, 4, 1) > tile(nn, 16, 64, 1, 4, 8));
    }
  }
}

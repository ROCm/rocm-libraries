// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
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
      auto config = make_config(128, 128, 64, 32, 32, 8, 1);

      auto latency = origami::compute_mt_compute_latency(problem, hardware, config);
      REQUIRE(latency == 4096);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - transA=N transB=T") {
      auto hardware = make_hardware(gpu_arch);
      auto problem =
          make_problem(4096, 4096, 1024, origami::transpose_t::N, origami::transpose_t::T);
      auto config = make_config(128, 128, 64, 32, 32, 8, 1);

      auto latency = origami::compute_mt_compute_latency(problem, hardware, config);
      REQUIRE(latency == 4096);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - transA=N transB=N") {
      auto hardware = make_hardware(gpu_arch);
      auto problem =
          make_problem(4096, 4096, 1024, origami::transpose_t::N, origami::transpose_t::N);
      auto config = make_config(128, 128, 64, 32, 32, 8, 1);

      auto latency = origami::compute_mt_compute_latency(problem, hardware, config);
      REQUIRE(latency == 4096);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - transA=T transB=T") {
      auto hardware = make_hardware(gpu_arch);
      auto problem =
          make_problem(4096, 4096, 1024, origami::transpose_t::T, origami::transpose_t::T);
      auto config = make_config(128, 128, 64, 32, 32, 8, 1);

      auto latency = origami::compute_mt_compute_latency(problem, hardware, config);
      REQUIRE(latency == 4096);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - different MT_K") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(4096, 4096, 1024);
      auto config   = make_config(128, 128, 32, 32, 32, 8, 1);

      auto latency = origami::compute_mt_compute_latency(problem, hardware, config);
      REQUIRE(latency > 2048);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - larger tile") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(4096, 4096, 1024);
      auto config   = make_config(224, 224, 64, 32, 32, 8, 1);

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
      auto config_small = make_config(128, 128, 64, 32, 32, 8, 8);
      auto config_large = make_config(256, 256, 128, 32, 32, 8, 8);

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
      auto config_small = make_config(128, 128, 64, 32, 32, 8, 6);
      auto config_large = make_config(256, 256, 128, 32, 32, 8, 6);

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
    DYNAMIC_SECTION("gfx" << gpu_arch << " - wave latency equals tile latency") {
      auto hardware = make_hardware(gpu_arch);
      auto problem =
          make_problem(4096, 4096, 1024, origami::transpose_t::T, origami::transpose_t::N, 2);
      auto config = make_config(128, 128, 64, 32, 32, 8, 8);

      auto tile_latency = origami::compute_tile_latency(problem, hardware, config, 304, 4);
      auto wave_latency = origami::compute_timestep_latency(problem, hardware, config, 304, 4);

      REQUIRE(wave_latency == Approx(tile_latency));
    }
  }
}

TEST_CASE("GEMM: compute_total_latency", "[gemm]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - smaller tiles have lower total latency") {
      auto hardware = make_hardware(gpu_arch);
      auto problem =
          make_problem(4096, 4096, 1024, origami::transpose_t::T, origami::transpose_t::N, 2);
      auto config_small = make_config(128, 128, 64, 32, 32, 8, 1);
      auto config_large = make_config(256, 256, 128, 32, 32, 8, 1);

      auto latency_small =
          origami::compute_total_latency(problem, hardware, config_small, hardware.N_CU);
      auto latency_large =
          origami::compute_total_latency(problem, hardware, config_large, hardware.N_CU);

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
      auto config   = make_config(256, 256, 64, 32, 32, 8, 1);

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
      auto config   = make_config(256, 256, 64, 32, 32, 8, 1);

      for (int wgm = 1; wgm < 1025; wgm++) {
        config.workgroup_mapping = wgm;
        auto mall_hit            = origami::estimate_mall_hit(problem, hardware, config, 304, 8);
        REQUIRE(mall_hit > 0.0);
      }
    }
  }
}

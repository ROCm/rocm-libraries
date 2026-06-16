// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "common.hpp"

#include "origami/attention.hpp"

using Catch::Approx;

// Test functions for attention.hpp/cpp

TEST_CASE("Attention: calculate_work_utilization", "[attention]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - perfect tile alignment returns 1.0") {
      auto problem = make_problem(4096, 4096, 1024);
      auto config  = make_config(256, 256, 64, 32, 32, 8, false, 1);

      auto result = origami::attention::calculate_work_utilization(problem, config);
      REQUIRE(result == 1.0);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - non-aligned dimensions") {
      auto problem = make_problem(4351, 3839, 959);
      auto config  = make_config(256, 256, 64, 32, 32, 8, false, 1);

      auto result = origami::attention::calculate_work_utilization(problem, config);
      REQUIRE(result == Approx(0.998).epsilon(1e-3));
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - zero M dimension returns 1.0") {
      auto problem = make_problem(0, 3839, 959);
      auto config  = make_config(256, 256, 64, 32, 32, 8, false, 1);

      auto result = origami::attention::calculate_work_utilization(problem, config);
      REQUIRE(result == 1.0);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - very small problem") {
      auto problem = make_problem(10, 20, 15);
      auto config  = make_config(256, 256, 64, 32, 32, 8, false, 1);

      auto result = origami::attention::calculate_work_utilization(problem, config);
      REQUIRE(result == Approx(0.0007152).epsilon(1e-4));
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - skinny matrix small M big N") {
      auto problem = make_problem(128, 81920, 1024);
      auto config  = make_config(256, 256, 64, 32, 32, 8, false, 1);

      auto result = origami::attention::calculate_work_utilization(problem, config);
      REQUIRE(result == 0.5);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - skinny matrix big M small N") {
      auto problem = make_problem(81920, 128, 1024);
      auto config  = make_config(256, 256, 64, 32, 32, 8, false, 1);

      auto result = origami::attention::calculate_work_utilization(problem, config);
      REQUIRE(result == 0.5);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - flash attention aligned tile") {
      auto problem = make_problem(4096, 4096, 128);
      auto config  = make_config(128, 128, 64, 16, 16, 16, false, 1);

      auto result = origami::attention::calculate_work_utilization(problem, config);
      REQUIRE(result == 1.0);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - non-aligned Q_SEQ") {
      auto problem = make_problem(4100, 4096, 128);
      auto config  = make_config(128, 128, 64, 16, 16, 16, false, 1);

      auto result = origami::attention::calculate_work_utilization(problem, config);
      REQUIRE(result < 1.0);
      REQUIRE(result > 0.97);
    }
  }
}

TEST_CASE("Attention: calculate_output_utilization", "[attention]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - perfect alignment returns 1.0") {
      auto problem = make_problem(4096, 4096, 1024);
      auto config  = make_config(256, 256, 64, 32, 32, 8, false, 1);

      auto result = origami::attention::calculate_output_utilization(problem, config, 1UL);
      REQUIRE(result == 1.0);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - non-aligned dimensions") {
      auto problem = make_problem(4351, 3839, 959);
      auto config  = make_config(256, 256, 64, 32, 32, 8, false, 1);

      auto result = origami::attention::calculate_output_utilization(problem, config, 1UL);
      REQUIRE(result == Approx(0.999).epsilon(1e-3));
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - with vector_elems > 1") {
      auto problem = make_problem(4096, 4096, 1024);
      auto config  = make_config(256, 256, 64, 32, 32, 8, false, 1);

      auto result = origami::attention::calculate_output_utilization(problem, config, 23UL);
      REQUIRE(result == Approx(1.010).epsilon(1e-3));
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - zero dimension returns 1.0") {
      auto problem = make_problem(0, 3839, 959);
      auto config  = make_config(256, 256, 64, 32, 32, 8, false, 1);

      auto result = origami::attention::calculate_output_utilization(problem, config, 1UL);
      REQUIRE(result == 1.0);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - small problem") {
      auto problem = make_problem(10, 20, 128);
      auto config  = make_config(128, 128, 64, 16, 16, 16, false, 1);

      auto result = origami::attention::calculate_output_utilization(problem, config, 1UL);
      REQUIRE(result < 0.02);
      REQUIRE(result > 0.0);
    }
  }
}

TEST_CASE("Attention: compute_cu_occupancy", "[attention]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - basic occupancy with split=1") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(4096, 4096, 1024, origami::transpose_t::T, origami::transpose_t::N, 1, 0, 1);
      auto config   = make_config(256, 256, 64, 32, 32, 8, false, 1);

      auto result = origami::attention::compute_cu_occupancy(
          problem, hardware, config, origami::grid_selection_t::k_split_aware, hardware.N_CU, 1UL);

      // grid_m = 4096/256 = 16, grid_n = 4096/256 = 16
      // num_tiles = 16 * 16 * 1(batch) * 1(q_heads) = 256
      size_t expected_wgs = 256;
      REQUIRE(std::get<0>(result) == expected_wgs);
      REQUIRE(std::get<1>(result) <= hardware.N_CU);
      REQUIRE(std::get<3>(result) == 1);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - occupancy with split > 1") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(4096, 4096, 1024, origami::transpose_t::T, origami::transpose_t::N, 1, 0, 1);
      auto config   = make_config(256, 256, 64, 32, 32, 8, false, 1);

      auto result = origami::attention::compute_cu_occupancy(
          problem, hardware, config, origami::grid_selection_t::k_split_aware, hardware.N_CU, 4UL);

      // num_tiles = 256, split_factor = 4, num_wgs = 256 * 4 = 1024
      REQUIRE(std::get<0>(result) == 1024);
      REQUIRE(std::get<1>(result) == hardware.N_CU);
      REQUIRE(std::get<3>(result) == 4);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - occupancy with max_cus limit") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(4096, 4096, 1024, origami::transpose_t::T, origami::transpose_t::N, 1, 0, 1);
      auto config   = make_config(256, 256, 64, 32, 32, 8, false, 1);

      auto result = origami::attention::compute_cu_occupancy(
          problem, hardware, config, origami::grid_selection_t::k_split_aware, 100, 1UL);

      REQUIRE(std::get<1>(result) <= 100);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - batched problem") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(1024, 1024, 128, origami::transpose_t::T, origami::transpose_t::N, 8, 0, 1);
      auto config   = make_config(128, 128, 64, 16, 16, 16, false, 1);

      auto result = origami::attention::compute_cu_occupancy(
          problem, hardware, config, origami::grid_selection_t::k_split_aware, hardware.N_CU, 1UL);

      // grid_m = 1024/128 = 8, grid_n = 1024/128 = 8, batch = 8, q_heads = 1
      // num_tiles = 8 * 8 * 8 * 1 = 512
      REQUIRE(std::get<0>(result) == 512);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - single tile") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(64, 64, 64, origami::transpose_t::T, origami::transpose_t::N, 1, 0, 1);
      auto config   = make_config(128, 128, 64, 16, 16, 16, false, 1);

      auto result = origami::attention::compute_cu_occupancy(
          problem, hardware, config, origami::grid_selection_t::k_split_aware, hardware.N_CU, 1UL);

      REQUIRE(std::get<0>(result) == 1);  // single WG
      REQUIRE(std::get<1>(result) == 1);  // single active CU
      REQUIRE(std::get<2>(result) >= 1);  // at least one timestep
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - few tiles, partial CU usage") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(128, 128, 128);
      auto config   = make_config(128, 128, 64, 16, 16, 16, false, 1);

      auto result = origami::attention::compute_cu_occupancy(
          problem, hardware, config, origami::grid_selection_t::k_split_aware, hardware.N_CU, 1UL);

      REQUIRE(std::get<1>(result) < hardware.N_CU);
      REQUIRE(std::get<1>(result) <= std::get<0>(result));
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - invariant: num_active_cus <= min(max_cus, N_CU)") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(4096, 4096, 128);
      auto config   = make_config(128, 128, 64, 16, 16, 16, false, 1);

      for (size_t max_cus : {1UL, 64UL, 128UL, 256UL, 512UL}) {
        auto result = origami::attention::compute_cu_occupancy(
            problem, hardware, config, origami::grid_selection_t::k_split_aware, max_cus, 1UL);

        REQUIRE(std::get<1>(result) <= std::min(max_cus, hardware.N_CU));
        REQUIRE(std::get<1>(result) <= std::get<0>(result));
        REQUIRE(std::get<2>(result) >= 1);
      }
    }
  }
}

TEST_CASE("Attention: compute_mem_bw_from_occupancy", "[attention]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - full CU occupancy") {
      auto hardware = make_hardware(gpu_arch);

      auto result = origami::attention::compute_mem_bw_from_occupancy(hardware, hardware.N_CU);
      REQUIRE(result == 1.0);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - exceeding N_CU returns 1.0") {
      auto hardware = make_hardware(gpu_arch);

      auto result = origami::attention::compute_mem_bw_from_occupancy(hardware, hardware.N_CU + 100);
      REQUIRE(result == 1.0);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - partial occupancy") {
      auto hardware = make_hardware(gpu_arch);

      auto result = origami::attention::compute_mem_bw_from_occupancy(hardware, 1);
      REQUIRE(result > 0.0);
      REQUIRE(result <= 1.0);
    }
  }
}

TEST_CASE("Attention: round_elements_to_128B", "[attention]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - 32-bit elements") {
      auto result = origami::attention::round_elements_to_128B(196, 32);
      REQUIRE(result == 224);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - 16-bit elements") {
      auto result = origami::attention::round_elements_to_128B(225, 16);
      REQUIRE(result == 256);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - 8-bit elements") {
      auto result = origami::attention::round_elements_to_128B(90, 8);
      REQUIRE(result == 128);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - zero elements returns 0") {
      auto result = origami::attention::round_elements_to_128B(0, 32);
      REQUIRE(result == 0);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - zero element size returns identity") {
      auto result = origami::attention::round_elements_to_128B(256, 0);
      REQUIRE(result == 256);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - BF16 already aligned") {
      // 512 * 16 bits = 8192 bits = 1024 bytes, multiple of 128
      auto result = origami::attention::round_elements_to_128B(512, 16);
      REQUIRE(result == 512);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - BF16 needs rounding") {
      // 100 * 16 = 1600 bits, needs rounding to 2048 bits = 128 elements
      auto result = origami::attention::round_elements_to_128B(100, 16);
      REQUIRE(result == 128);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - FP64 already aligned") {
      // 16 * 64 = 1024 bits = 128 bytes
      auto result = origami::attention::round_elements_to_128B(16, 64);
      REQUIRE(result == 16);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - single BF16 element rounds up") {
      // 1 * 16 = 16 bits, needs 1024 bits = 64 elements
      auto result = origami::attention::round_elements_to_128B(1, 16);
      REQUIRE(result == 64);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - single INT8 element rounds up") {
      // 1 * 8 = 8 bits, needs 1024 bits = 128 elements
      auto result = origami::attention::round_elements_to_128B(1, 8);
      REQUIRE(result == 128);
    }
  }
}

TEST_CASE("Attention: arithmetic_intensity", "[attention]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - standard calculation") {
      auto result = origami::attention::arithmetic_intensity(2047, 2047, 4096, 2);
      REQUIRE(result == Approx(818.879).epsilon(1e-3));
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - different problem sizes") {
      auto result = origami::attention::arithmetic_intensity(127, 4097, 8193, 2);
      REQUIRE(result == Approx(121.356).epsilon(1e-3));

      result = origami::attention::arithmetic_intensity(4097, 4097, 257, 4);
      REQUIRE(result == Approx(114.175).epsilon(1e-3));
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - zero values return 0") {
      auto result = origami::attention::arithmetic_intensity(0, 0, 0, 4);
      REQUIRE(result == 0.0);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - doubling bpe halves intensity") {
      double ai_bf16 = origami::attention::arithmetic_intensity(128, 128, 64, 2.0);
      double ai_fp32 = origami::attention::arithmetic_intensity(128, 128, 64, 4.0);
      REQUIRE(ai_fp32 == Approx(ai_bf16 / 2.0).epsilon(1e-6));
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - zero n returns 0") {
      auto result = origami::attention::arithmetic_intensity(128, 0, 64, 2.0);
      REQUIRE(result == 0.0);
    }
  }
}

TEST_CASE("Attention: emulated_tf32_arithmetic_intensity", "[attention]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - standard calculation") {
      auto result = origami::attention::emulated_tf32_arithmetic_intensity(2047, 2047, 4096, 2);
      REQUIRE(result == Approx(2456.639).epsilon(1e-3));
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - 3x arithmetic intensity ratio") {
      double ai    = origami::attention::arithmetic_intensity(2047, 2047, 4096, 2);
      double tf32  = origami::attention::emulated_tf32_arithmetic_intensity(2047, 2047, 4096, 2);
      REQUIRE(tf32 == Approx(3.0 * ai).epsilon(1e-6));
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - zero values return 0") {
      auto result = origami::attention::emulated_tf32_arithmetic_intensity(0, 0, 0, 2);
      REQUIRE(result == 0.0);
    }
  }
}

TEST_CASE("Attention: compute_number_matrix_instructions", "[attention]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - 128x128x64 with 16x16x16") {
      origami::dim3_t mt{128, 128, 64};
      origami::dim3_t mi{16, 16, 16};
      auto num_instructions = origami::attention::compute_number_matrix_instructions(mt, mi);
      REQUIRE(num_instructions == 256);  // 8 * 8 * 4
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - 16x16x64 with 16x16x32") {
      origami::dim3_t mt{16, 16, 64};
      origami::dim3_t mi{16, 16, 32};
      auto num_instructions = origami::attention::compute_number_matrix_instructions(mt, mi);
      REQUIRE(num_instructions == 2);  // 1 * 1 * 2
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - single instruction") {
      origami::dim3_t mt{32, 32, 8};
      origami::dim3_t mi{32, 32, 8};
      auto num_instructions = origami::attention::compute_number_matrix_instructions(mt, mi);
      REQUIRE(num_instructions == 1);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - non-divisible rounds up") {
      origami::dim3_t mt{100, 100, 50};
      origami::dim3_t mi{16, 16, 16};
      auto num_instructions = origami::attention::compute_number_matrix_instructions(mt, mi);
      // ceil(100/16) * ceil(100/16) * ceil(50/16) = 7 * 7 * 4 = 196
      REQUIRE(num_instructions == 196);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - large tile small MI") {
      origami::dim3_t mt{256, 256, 128};
      origami::dim3_t mi{16, 16, 16};
      auto num_instructions = origami::attention::compute_number_matrix_instructions(mt, mi);
      REQUIRE(num_instructions == 2048);  // 16 * 16 * 8
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - 32x32 MI (gfx942 style)") {
      origami::dim3_t mt{128, 128, 64};
      origami::dim3_t mi{32, 32, 8};
      auto num_instructions = origami::attention::compute_number_matrix_instructions(mt, mi);
      REQUIRE(num_instructions == 128);  // 4 * 4 * 8
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - exact division 64x64x64") {
      origami::dim3_t mt{64, 64, 64};
      origami::dim3_t mi{16, 16, 16};
      auto num_instructions = origami::attention::compute_number_matrix_instructions(mt, mi);
      REQUIRE(num_instructions == 64);  // 4 * 4 * 4
    }
  }
}

TEST_CASE("Attention: compute_mt_compute_latency", "[attention]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - standard tile") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(4096, 4096, 1024);
      auto config   = make_config(128, 128, 64, 32, 32, 8, false, 1);

      auto latency = origami::attention::compute_mt_compute_latency(problem, hardware, config);
      REQUIRE(latency == 4096);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - smaller MT_K") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(4096, 4096, 1024);
      auto config   = make_config(128, 128, 32, 32, 32, 8, false, 1);

      auto latency = origami::attention::compute_mt_compute_latency(problem, hardware, config);
      REQUIRE(latency == 2048);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - larger tile has higher latency") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(4096, 4096, 1024);
      auto config   = make_config(224, 224, 64, 32, 32, 8, false, 1);

      auto latency = origami::attention::compute_mt_compute_latency(problem, hardware, config);
      REQUIRE(latency > 12543);
    }
  }
}

TEST_CASE("Attention: check_lds_capacity", "[attention]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - 256x256x64 BFloat16 fits") {
      auto hardware = make_hardware(gpu_arch);
      origami::dim3_t mt{256, 256, 64};

      auto fits = origami::attention::check_lds_capacity(
          hardware, mt, origami::data_type_t::BFloat16);
      REQUIRE(fits == true);
    }

    if (gpu_arch == 942) {
      DYNAMIC_SECTION("gfx" << gpu_arch << " - large tile exceeds LDS") {
        auto hardware = make_hardware(gpu_arch);

        auto fits = origami::attention::check_lds_capacity(
            hardware, {256, 256, 256}, origami::data_type_t::BFloat16);
        REQUIRE(fits == false);
      }

      DYNAMIC_SECTION("gfx" << gpu_arch << " - small Int8 tile fits") {
        auto hardware = make_hardware(gpu_arch);

        auto fits = origami::attention::check_lds_capacity(
            hardware, {128, 128, 64}, origami::data_type_t::Int8);
        REQUIRE(fits == true);
      }
    } else if (gpu_arch == 950) {
      DYNAMIC_SECTION("gfx" << gpu_arch << " - large tile exceeds LDS") {
        auto hardware = make_hardware(gpu_arch);

        auto fits = origami::attention::check_lds_capacity(
            hardware, {512, 512, 256}, origami::data_type_t::BFloat16);
        REQUIRE(fits == false);
      }

      DYNAMIC_SECTION("gfx" << gpu_arch << " - Int8 tile fits with larger LDS") {
        auto hardware = make_hardware(gpu_arch);

        auto fits = origami::attention::check_lds_capacity(
            hardware, {256, 256, 64}, origami::data_type_t::Int8);
        REQUIRE(fits == true);
      }
    }
  }
}

TEST_CASE("Attention: estimate_l2_hit", "[attention]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - L2 hit rate in valid range") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(4096, 4096, 1024);
      auto config   = make_config(256, 256, 64, 32, 32, 8, false, 1);

      for (int wgm = 1; wgm < 1025; wgm++) {
        config.workgroup_mapping = wgm;
        auto l2_hit = origami::attention::estimate_l2_hit(problem, hardware, config, 3);
        REQUIRE(l2_hit > 0.0);
        REQUIRE(l2_hit < 1.0);
      }
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - splitting factor 0 returns 0") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(2047, 2047, 4096);
      auto config   = make_config(256, 256, 64, 32, 32, 8, false, 1);

      auto result = origami::attention::estimate_l2_hit(problem, hardware, config, 0);
      REQUIRE(result == 0.0);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - splitting factor 1") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(2047, 2047, 4096);
      auto config   = make_config(256, 256, 64, 32, 32, 8, false, 1);

      auto result = origami::attention::estimate_l2_hit(problem, hardware, config, 1);
      REQUIRE(result == 0.4375);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - very small problem") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(10, 11, 253);
      auto config   = make_config(256, 256, 64, 32, 32, 8, false, 1);

      auto result = origami::attention::estimate_l2_hit(problem, hardware, config, 1);
      REQUIRE(result == 0.0);
    }
  }
}

TEST_CASE("Attention: estimate_mall_hit", "[attention]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - mall hit rate is positive") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(4096, 4096, 1024);
      auto config   = make_config(256, 256, 64, 32, 32, 8, false, 1);

      for (int wgm = 1; wgm < 1025; wgm++) {
        config.workgroup_mapping = wgm;
        auto mall_hit = origami::attention::estimate_mall_hit(problem, hardware, config, 304, 8);
        REQUIRE(mall_hit > 0.0);
      }
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - zero active CUs returns 0") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(4096, 4096, 1024);
      auto config   = make_config(256, 256, 64, 32, 32, 8, false, 1);

      auto result = origami::attention::estimate_mall_hit(problem, hardware, config, 0, 1);
      REQUIRE(result == 0.0);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - with full CU occupancy") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(2047, 2047, 4096);
      auto config   = make_config(256, 256, 64, 32, 32, 8, false, 1);

      auto result = origami::attention::estimate_mall_hit(problem, hardware, config, hardware.N_CU, 0);
      REQUIRE(result == 0.875);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - single CU in valid range") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(4096, 4096, 128);
      auto config   = make_config(128, 128, 64, 16, 16, 16, false, 1);

      auto result = origami::attention::estimate_mall_hit(problem, hardware, config, 1, 1);
      REQUIRE(result >= 0.0);
      REQUIRE(result <= 1.0);
    }
  }
}

TEST_CASE("Attention: compute_l2_hit_rate_global", "[attention]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - standard problem") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(2047, 2047, 4096);
      auto config   = make_config(256, 256, 64, 32, 32, 8, false, 1);

      auto result = origami::attention::compute_l2_hit_rate_global(
          problem, hardware, config, hardware.L2_capacity * 1024);
      REQUIRE(result == 0.875);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - small problem") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(331, 4077, 547);
      auto config   = make_config(256, 256, 64, 32, 32, 8, false, 1);

      auto result = origami::attention::compute_l2_hit_rate_global(
          problem, hardware, config, hardware.L2_capacity * 1024);
      REQUIRE(result == 0.71875);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - large problem") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(8193, 4077, 7453);
      auto config   = make_config(256, 256, 64, 32, 32, 8, false, 1);

      auto result = origami::attention::compute_l2_hit_rate_global(
          problem, hardware, config, hardware.L2_capacity * 1024);
      REQUIRE(result == Approx(0.953).epsilon(1e-3));
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - zero L2 capacity returns 0") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(2047, 2047, 4096);
      auto config   = make_config(256, 256, 64, 32, 32, 8, false, 1);

      auto result = origami::attention::compute_l2_hit_rate_global(problem, hardware, config, 0UL);
      REQUIRE(result == 0.0);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - working set exceeds L2 returns 0.1") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(8192, 8192, 128);
      auto config   = make_config(128, 128, 64, 16, 16, 16, false, 1);

      // Use a very small L2 capacity
      auto result = origami::attention::compute_l2_hit_rate_global(problem, hardware, config, 1000);
      REQUIRE(result == 0.1);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - result always in valid range") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(4096, 4096, 128);
      auto config   = make_config(128, 128, 64, 16, 16, 16, false, 1);

      auto result = origami::attention::compute_l2_hit_rate_global(
          problem, hardware, config, hardware.L2_capacity * 1024);
      REQUIRE(result >= 0.0);
      REQUIRE(result <= 1.0);
    }
  }
}

TEST_CASE("Attention: compute_memory_latency", "[attention]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - smaller tiles have lower memory latency") {
      auto hardware     = make_hardware(gpu_arch);
      auto problem      = make_problem(4096, 4096, 1024, origami::transpose_t::T, origami::transpose_t::N, 1);
      auto config_small = make_config(128, 128, 64, 32, 32, 8, false, 8);
      auto config_large = make_config(256, 256, 128, 32, 32, 8, false, 8);

      auto mem_latency_small =
          origami::attention::compute_memory_latency(problem, hardware, config_small, 304, 2);
      auto mem_latency_large =
          origami::attention::compute_memory_latency(problem, hardware, config_large, 304, 2);

      REQUIRE(mem_latency_small < mem_latency_large);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - latency is positive") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(4096, 4096, 128);
      auto config   = make_config(128, 128, 64, 16, 16, 16, false, 1);

      auto latency = origami::attention::compute_memory_latency(problem, hardware, config, hardware.N_CU, 1);
      REQUIRE(latency > 0.0);
    }
  }
}

TEST_CASE("Attention: compute_tile_latency", "[attention]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - larger tiles have higher latency") {
      auto hardware     = make_hardware(gpu_arch);
      auto problem      = make_problem(4096, 4096, 1024, origami::transpose_t::T, origami::transpose_t::N, 2);
      auto config_small = make_config(128, 128, 64, 32, 32, 8, false, 6);
      auto config_large = make_config(256, 256, 128, 32, 32, 8, false, 6);

      auto tile_latency_small =
          origami::attention::compute_tile_latency(problem, hardware, config_small, 304, 3);
      auto tile_latency_large =
          origami::attention::compute_tile_latency(problem, hardware, config_large, 304, 3);

      REQUIRE(tile_latency_large > tile_latency_small);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - tile latency is positive") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(2048, 2048, 128);
      auto config   = make_config(128, 128, 64, 16, 16, 16, false, 1);

      auto latency = origami::attention::compute_tile_latency(problem, hardware, config, hardware.N_CU, 1);
      REQUIRE(latency > 0.0);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - deterministic (same inputs same output)") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(2048, 2048, 128);
      auto config   = make_config(128, 128, 64, 16, 16, 16, false, 1);

      auto result1 = origami::attention::compute_tile_latency(problem, hardware, config, hardware.N_CU, 1);
      auto result2 = origami::attention::compute_tile_latency(problem, hardware, config, hardware.N_CU, 1);
      REQUIRE(result1 == result2);
    }
  }
}

TEST_CASE("Attention: compute_timestep_latency", "[attention]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - timestep equals tile latency") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(4096, 4096, 1024, origami::transpose_t::T, origami::transpose_t::N, 2);
      auto config   = make_config(128, 128, 64, 32, 32, 8, false, 8);

      auto tile_latency     = origami::attention::compute_tile_latency(problem, hardware, config, 304, 4);
      auto timestep_latency = origami::attention::compute_timestep_latency(problem, hardware, config, 304, 4);

      REQUIRE(timestep_latency == Approx(tile_latency));
    }
  }
}

TEST_CASE("Attention: compute_total_latency", "[attention]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - larger problems have higher total latency") {
      auto hardware       = make_hardware(gpu_arch);
      auto problem_small  = make_problem(4096, 4096, 1024, origami::transpose_t::T, origami::transpose_t::N, 2);
      auto problem_large  = make_problem(8192, 8192, 2048, origami::transpose_t::T, origami::transpose_t::N, 2);
      auto config         = make_config(128, 128, 64, 32, 32, 8, false, 1);

      auto latency_small = origami::attention::compute_total_latency(problem_small, hardware, config, hardware.N_CU);
      auto latency_large = origami::attention::compute_total_latency(problem_large, hardware, config, hardware.N_CU);

      REQUIRE(latency_small < latency_large);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - total latency is positive") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(2048, 2048, 128);
      auto config   = make_config(128, 128, 64, 16, 16, 16, false, 1);

      auto latency = origami::attention::compute_total_latency(problem, hardware, config, hardware.N_CU);
      REQUIRE(latency > 0.0);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - total latency scales with batch") {
      auto hardware    = make_hardware(gpu_arch);
      auto problem_b1  = make_problem(2048, 2048, 128, origami::transpose_t::T, origami::transpose_t::N, 1);
      auto problem_b8  = make_problem(2048, 2048, 128, origami::transpose_t::T, origami::transpose_t::N, 8);
      auto config      = make_config(128, 128, 64, 16, 16, 16, false, 1);

      auto latency_b1 = origami::attention::compute_total_latency(problem_b1, hardware, config, hardware.N_CU);
      auto latency_b8 = origami::attention::compute_total_latency(problem_b8, hardware, config, hardware.N_CU);

      REQUIRE(latency_b8 > latency_b1);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - total latency structure: prologue + loop + epilogue") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(2048, 2048, 128);
      auto config   = make_config(128, 128, 64, 16, 16, 16, false, 1);

      auto total = origami::attention::compute_total_latency(problem, hardware, config, hardware.N_CU);

      // Total latency should include prologue + at least 1 loop iteration + epilogue
      // So it should be greater than just the tile latency alone
      auto tile_latency = origami::attention::compute_tile_latency(problem, hardware, config, hardware.N_CU, 1);
      REQUIRE(total > tile_latency);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - single tile problem") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(64, 64, 64);
      auto config   = make_config(128, 128, 64, 16, 16, 16, false, 1);

      auto total = origami::attention::compute_total_latency(problem, hardware, config, hardware.N_CU);
      REQUIRE(total > 0.0);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - very large problem is finite") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(16384, 16384, 256, origami::transpose_t::T, origami::transpose_t::N, 64);
      auto config   = make_config(128, 128, 64, 16, 16, 16, false, 1);

      auto total = origami::attention::compute_total_latency(problem, hardware, config, hardware.N_CU);
      REQUIRE(total > 0.0);
      REQUIRE(std::isfinite(total));
    }
  }
}

TEST_CASE("Attention: edge cases - minimum problem sizes", "[attention][edge]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - single element problem") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(1, 1, 1);
      auto config   = make_config(128, 128, 64, 16, 16, 16, false, 1);

      auto total = origami::attention::compute_total_latency(problem, hardware, config, hardware.N_CU);
      REQUIRE(total > 0.0);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - min practical attention (32x32x64)") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(32, 32, 64);
      auto config   = make_config(128, 128, 64, 16, 16, 16, false, 1);

      auto total = origami::attention::compute_total_latency(problem, hardware, config, hardware.N_CU);
      REQUIRE(total > 0.0);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - typical small attention (128x128x64)") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(128, 128, 64);
      auto config   = make_config(128, 128, 64, 16, 16, 16, false, 1);

      auto total = origami::attention::compute_total_latency(problem, hardware, config, hardware.N_CU);
      REQUIRE(total > 0.0);
    }
  }
}

TEST_CASE("Attention: edge cases - batch scaling", "[attention][edge]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - increasing batch increases total latency") {
      auto hardware = make_hardware(gpu_arch);
      auto config   = make_config(128, 128, 64, 16, 16, 16, false, 1);

      double prev_latency = 0.0;
      for (size_t batch : {1UL, 2UL, 4UL, 8UL, 16UL, 32UL}) {
        auto problem = make_problem(2048, 2048, 128, origami::transpose_t::T, origami::transpose_t::N, batch);
        auto total = origami::attention::compute_total_latency(problem, hardware, config, hardware.N_CU);
        REQUIRE(total > 0.0);
        if (batch > 1) {
          REQUIRE(total > prev_latency);
        }
        prev_latency = total;
      }
    }
  }
}

TEST_CASE("Attention: L2 estimate with splitting", "[attention]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - L2 hit rate with splitting in valid range") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(4096, 4096, 128);
      auto config   = make_config(128, 128, 64, 16, 16, 16, false, 8);

      auto result = origami::attention::estimate_l2_hit(problem, hardware, config, 4);
      REQUIRE(result >= 0.0);
      REQUIRE(result <= 1.0);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - medium problem L2 estimate") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(2048, 2048, 128);
      auto config   = make_config(128, 128, 64, 16, 16, 16, false, 1);

      auto result = origami::attention::estimate_l2_hit(problem, hardware, config, 1);
      REQUIRE(result >= 0.0);
      REQUIRE(result <= 1.0);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - small problem high reuse") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(256, 256, 128);
      auto config   = make_config(128, 128, 64, 16, 16, 16, false, 1);

      auto result = origami::attention::estimate_l2_hit(problem, hardware, config, 1);
      REQUIRE(result >= 0.0);
    }
  }
}

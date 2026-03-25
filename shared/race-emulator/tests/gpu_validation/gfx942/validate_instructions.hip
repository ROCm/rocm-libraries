// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// GPU-validated instruction test for the race-emulator on gfx942 (CDNA3).
//
// This file requires a gfx942 GPU to build and run. It validates that the
// emulator produces the same results as real hardware by running inline
// assembly on the GPU and comparing against the emulator's Wave API.
//
// Build: cmake -DRACE_EMULATOR_BUILD_HW_TESTING=ON ...
// Run:   ctest -R GpuValidation

#include "race-emulator/Wave.h"
#include <bit>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <vector>

#define HIP_CHECK(expr)                                                        \
  do {                                                                         \
    hipError_t err = (expr);                                                   \
    if (err != hipSuccess) {                                                   \
      throw std::runtime_error(                                                \
          std::string("HIP error: ") + hipGetErrorString(err));                \
    }                                                                          \
  } while (0)

using namespace raceemulator;

// ---------------------------------------------------------------------------
// GPU kernel: v_exp_f32 via inline assembly
// ---------------------------------------------------------------------------

__global__ void kernel_v_exp_f32(const float *in, float *out, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    float x = in[idx], y;
    asm volatile("v_exp_f32 %0, %1" : "=v"(y) : "v"(x));
    out[idx] = y;
  }
}

// ---------------------------------------------------------------------------
// Test
// ---------------------------------------------------------------------------

TEST(GpuValidation, v_exp_f32) {
  // Test values covering normal, edge, and special cases.
  std::vector<float> vals;
  for (int i = -10; i <= 10; ++i)
    vals.push_back(static_cast<float>(i) * 0.5f);
  vals.push_back(0.0f);
  vals.push_back(-0.0f);
  vals.push_back(std::numeric_limits<float>::infinity());
  vals.push_back(-std::numeric_limits<float>::infinity());
  vals.push_back(std::numeric_limits<float>::quiet_NaN());
  vals.push_back(std::numeric_limits<float>::min());
  vals.push_back(std::numeric_limits<float>::denorm_min());

  int N = static_cast<int>(vals.size());

  // --- Run on GPU ---
  float *d_in, *d_out;
  HIP_CHECK(hipMalloc(&d_in, N * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out, N * sizeof(float)));
  HIP_CHECK(
      hipMemcpy(d_in, vals.data(), N * sizeof(float), hipMemcpyHostToDevice));
  kernel_v_exp_f32<<<(N + 63) / 64, 64>>>(d_in, d_out, N);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> gpuOut(N);
  HIP_CHECK(
      hipMemcpy(gpuOut.data(), d_out, N * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(d_in));
  HIP_CHECK(hipFree(d_out));

  // --- Run through emulator ---
  LDS lds;
  Wave wave(4, 4, std::min(N, 64), lds);
  for (int i = 0; i < N; ++i)
    wave.setVgpr(0, i, std::bit_cast<uint32_t>(vals[i]));
  wave.tryExecute("v_exp_f32_e32 v1, v0", false);

  // --- Compare ---
  for (int i = 0; i < N; ++i) {
    float gpu = gpuOut[i];
    float emu = std::bit_cast<float>(wave.getVgpr(1, i));
    if (std::isnan(gpu)) {
      EXPECT_TRUE(std::isnan(emu)) << "index " << i << ": GPU=NaN, EMU=" << emu;
    } else {
      EXPECT_EQ(std::bit_cast<uint32_t>(gpu), std::bit_cast<uint32_t>(emu))
          << "index " << i << ": input=" << vals[i]
          << " GPU=" << gpu << " EMU=" << emu;
    }
  }
}

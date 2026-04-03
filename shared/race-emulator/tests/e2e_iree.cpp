// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "race-emulator/Arch.h"
#include "race-emulator/Emulator.h"
#include "race-emulator/FloatTypes.h"
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#ifndef TEST_KERNEL_DIR
#define TEST_KERNEL_DIR "undefined"
#endif

namespace {

using namespace raceemulator;
namespace fs = std::filesystem;

std::string loadKernelFile(const std::string &filename) {
  fs::path filepath = fs::path(TEST_KERNEL_DIR) / filename;
  std::ifstream file(filepath);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open kernel file: " +
                             filepath.string());
  }
  std::stringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

// CPU reference: row-major matmul C = A * B (f32 accumulation).
// A is [M x K], B is [K x N], C is [M x N], all row-major.
void cpuGemmRowMajorF32(int M, int N, int K, const std::vector<float> &a,
                        const std::vector<float> &b, std::vector<float> &c) {
  c.resize(M * N, 0.0f);
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        sum += a[m * K + k] * b[k * N + n];
      }
      c[m * N + n] = sum;
    }
  }
}

} // namespace

// IREE-compiled 32x32x32 bf16xbf16xf32 matmul on gfx942.
// MLIR source: tests/sources/iree/matmul_32x32x32_bf16xbf16xf32.mlir
// Assembly: tests/asm/gfx942/iree_matmul_32x32x32_bf16xbf16xf32.s
//
// Dispatch parameters:
//   .max_flat_workgroup_size: 256  -> blockDim = (256, 1, 1), 4 waves of 64
//   Grid: (1, 1, 1)
//
// Kernel args: 3 pointers at offsets 0, 8, 16 (A bf16, B bf16, C f32).
TEST(Gfx942, MatMul_IREE_BF16xBF16xF32_32x32x32) {
  const int M = 32, N = 32, K = 32;

  // Load assembly
  auto assembly = loadKernelFile("gfx942/iree_matmul_32x32x32_bf16xbf16xf32.s");

  // Initialize A and B with simple values (±1 as bf16)
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0, 1);
  std::vector<float> choices = {1.0f, -1.0f};

  std::vector<float> aF32(M * K);
  std::vector<float> bF32(K * N);
  for (auto &v : aF32) {
    v = choices[dist(rng)];
  }
  for (auto &v : bF32) {
    v = choices[dist(rng)];
  }

  // Convert to bf16
  std::vector<uint16_t> aBf16(M * K);
  std::vector<uint16_t> bBf16(K * N);
  for (size_t i = 0; i < aF32.size(); ++i) {
    aBf16[i] = floatToBf16(aF32[i]);
  }
  for (size_t i = 0; i < bF32.size(); ++i) {
    bBf16[i] = floatToBf16(bF32[i]);
  }

  // Output buffer (f32), zero-initialized
  std::vector<float> cGpu(M * N, 0.0f);

  // Setup emulator
  Emulator emulator(assembly, std::make_shared<Gfx942>());

  // IREE kernel args: 3 pointers at offsets 0, 8, 16
  uint16_t *aPtr = aBf16.data();
  uint16_t *bPtr = bBf16.data();
  float *cPtr = cGpu.data();

  emulator.addKernarg(0, &aPtr); // A at offset 0
  emulator.addKernarg(1, &bPtr); // B at offset 8
  emulator.addKernarg(2, &cPtr); // C at offset 16

  // Run: 1 workgroup, 256 threads
  Dim3d wgId(0, 0, 0);
  Dim3d blockDim(256, 1, 1);
  emulator.run(wgId, blockDim, {.raceChecks = true});

  // CPU reference (row-major)
  std::vector<float> cRef;
  cpuGemmRowMajorF32(M, N, K, aF32, bF32, cRef);

  // MUTATION TEST: uncomment the next line to verify the test catches errors.
  // cRef[0] += 999.0f;

  // Verify
  int mismatches = 0;
  for (int i = 0; i < M * N; ++i) {
    if (std::isnan(cGpu[i]) || std::isinf(cGpu[i]) ||
        std::abs(cGpu[i] - cRef[i]) > 1e-2f) {
      if (mismatches < 10) {
        std::cerr << "Mismatch at index " << i << ": expected " << cRef[i]
                  << ", got " << cGpu[i] << "\n";
      }
      mismatches++;
    }
  }
  ASSERT_EQ(mismatches, 0) << "Total mismatches: " << mismatches;
}

// IREE-compiled 64x128x512 f16xf16xf32 matmul for gfx1151.
// MLIR source: tests/sources/iree/matmul_64x128x512_f16xf16xf32.mlir
// Assembly: tests/asm/gfx1151/iree_matmul_64x128x512_f16xf16xf32.s
//
// Dispatch parameters (from assembly metadata and --compile-to=hal):
//   .max_flat_workgroup_size: 128  -> blockDim = (128, 1, 1)
//   .wavefront_size: 32            -> 128/32 = 4 waves
//   Grid: (8, 1, 1) -> 8 workgroups along X.
//
// Kernel args (from .args metadata): 3 pointers at offsets 0, 8, 16.
//   arg 0: A (f16, read_only)  [64 x 512]
//   arg 1: B (f16, read_only)  [512 x 128]
//   arg 2: C (f32, write_only) [64 x 128]
TEST(Gfx1151, MatMul_IREE_F16xF16xF32_64x128x512) {
  const int M = 64, N = 128, K = 512;

  auto assembly =
      loadKernelFile("gfx1151/iree_matmul_64x128x512_f16xf16xf32.s");

  std::mt19937 rng(123);
  std::uniform_int_distribution<int> dist(0, 1);
  std::vector<float> choices = {1.0f, -1.0f};

  std::vector<float> aF32(M * K);
  std::vector<float> bF32(K * N);
  for (auto &v : aF32) {
    v = choices[dist(rng)];
  }
  for (auto &v : bF32) {
    v = choices[dist(rng)];
  }

  std::vector<uint16_t> aF16(M * K);
  std::vector<uint16_t> bF16(K * N);
  for (size_t i = 0; i < aF32.size(); ++i) {
    aF16[i] = floatToF16(aF32[i]);
  }
  for (size_t i = 0; i < bF32.size(); ++i) {
    bF16[i] = floatToF16(bF32[i]);
  }

  std::vector<float> cGpu(M * N, 0.0f);

  Emulator emulator(assembly, std::make_shared<Gfx1151>());

  uint16_t *aPtr = aF16.data();
  uint16_t *bPtr = bF16.data();
  float *cPtr = cGpu.data();

  emulator.addKernarg(0, &aPtr);
  emulator.addKernarg(1, &bPtr);
  emulator.addKernarg(2, &cPtr);

  // 8 workgroups, 128 threads each (4 waves of 32)
  Dim3d blockDim(128, 1, 1);
  std::vector<Dim3d> wgIds;
  for (int wgX = 0; wgX < 8; ++wgX) {
    wgIds.emplace_back(wgX, 0, 0);
  }
  emulator.run(wgIds, blockDim, {.raceChecks = true});

  std::vector<float> cRef;
  cpuGemmRowMajorF32(M, N, K, aF32, bF32, cRef);

  int mismatches = 0;
  for (int i = 0; i < M * N; ++i) {
    if (std::isnan(cGpu[i]) || std::isinf(cGpu[i]) ||
        std::abs(cGpu[i] - cRef[i]) > 1e-2f) {
      if (mismatches < 10) {
        std::cerr << "Mismatch at index " << i << ": expected " << cRef[i]
                  << ", got " << cGpu[i] << "\n";
      }
      mismatches++;
    }
  }
  ASSERT_EQ(mismatches, 0) << "Total mismatches: " << mismatches;
}

// IREE-compiled 32x32x32 bf16xbf16xf32 matmul for gfx1151.
// MLIR source: tests/sources/iree/matmul_32x32x32_bf16xbf16xf32.mlir
// Assembly: tests/asm/gfx1151/iree_matmul_32x32x32_bf16xbf16xf32.s
//
// Dispatch parameters (from assembly metadata):
//   .max_flat_workgroup_size: 128  -> blockDim = (128, 1, 1)
//   .wavefront_size: 32            -> 128/32 = 4 waves
//   IREE dispatches 1 workgroup for this problem size.
//
// Kernel args (from .args metadata): 3 pointers at offsets 0, 8, 16.
//   arg 0: A (bf16, read_only)
//   arg 1: B (bf16, read_only)
//   arg 2: C (f32, write_only)
TEST(Gfx1151, MatMul_IREE_BF16xBF16xF32_32x32x32) {
  const int M = 32, N = 32, K = 32;

  auto assembly =
      loadKernelFile("gfx1151/iree_matmul_32x32x32_bf16xbf16xf32.s");

  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0, 1);
  std::vector<float> choices = {1.0f, -1.0f};

  std::vector<float> aF32(M * K);
  std::vector<float> bF32(K * N);
  for (auto &v : aF32) {
    v = choices[dist(rng)];
  }
  for (auto &v : bF32) {
    v = choices[dist(rng)];
  }

  std::vector<uint16_t> aBf16(M * K);
  std::vector<uint16_t> bBf16(K * N);
  for (size_t i = 0; i < aF32.size(); ++i) {
    aBf16[i] = floatToBf16(aF32[i]);
  }
  for (size_t i = 0; i < bF32.size(); ++i) {
    bBf16[i] = floatToBf16(bF32[i]);
  }

  std::vector<float> cGpu(M * N, 0.0f);

  Emulator emulator(assembly, std::make_shared<Gfx1151>());

  uint16_t *aPtr = aBf16.data();
  uint16_t *bPtr = bBf16.data();
  float *cPtr = cGpu.data();

  emulator.addKernarg(0, &aPtr);
  emulator.addKernarg(1, &bPtr);
  emulator.addKernarg(2, &cPtr);

  // 1 workgroup, 128 threads (4 waves of 32)
  Dim3d wgId(0, 0, 0);
  Dim3d blockDim(128, 1, 1);
  emulator.run(wgId, blockDim, {.raceChecks = true});

  std::vector<float> cRef;
  cpuGemmRowMajorF32(M, N, K, aF32, bF32, cRef);

  int mismatches = 0;
  for (int i = 0; i < M * N; ++i) {
    if (std::isnan(cGpu[i]) || std::isinf(cGpu[i]) ||
        std::abs(cGpu[i] - cRef[i]) > 1e-2f) {
      if (mismatches < 10) {
        std::cerr << "Mismatch at index " << i << ": expected " << cRef[i]
                  << ", got " << cGpu[i] << "\n";
      }
      mismatches++;
    }
  }
  ASSERT_EQ(mismatches, 0) << "Total mismatches: " << mismatches;
}

// Diagnostic: A=B=1 on gfx1151 16x16x512 bf16xbf16xf32.
// All output elements should equal K=512.
// This kernel uses IREE's 16-WMMA-per-iteration tiling (K>=384 threshold),
// which is where the emulator diverges from hardware.
//
// MLIR source: tests/sources/iree/matmul_16x16x512_bf16xbf16xf32.mlir
// Assembly: tests/asm/gfx1151/iree_matmul_16x16x512_bf16xbf16xf32.s
//
// Dispatch parameters:
//   .max_flat_workgroup_size: 32  -> blockDim = (32, 1, 1), 1 wave
//   Grid: (1, 1, 1)
//
// Kernel args: 3 pointers at offsets 0, 8, 16 (A bf16, B bf16, C f32).
TEST(Gfx1151, MatMul_IREE_BF16xBF16xF32_16x16x512_AllOnes) {
  const int M = 16, N = 16, K = 512;

  auto assembly =
      loadKernelFile("gfx1151/iree_matmul_16x16x512_bf16xbf16xf32.s");

  std::vector<uint16_t> aBf16(M * K, floatToBf16(1.0f));
  std::vector<uint16_t> bBf16(K * N, floatToBf16(1.0f));
  std::vector<float> cGpu(M * N, 0.0f);

  Emulator emulator(assembly, std::make_shared<Gfx1151>());

  uint16_t *aPtr = aBf16.data();
  uint16_t *bPtr = bBf16.data();
  float *cPtr = cGpu.data();

  emulator.addKernarg(0, &aPtr);
  emulator.addKernarg(1, &bPtr);
  emulator.addKernarg(2, &cPtr);

  emulator.run(Dim3d(0, 0, 0), Dim3d(32, 1, 1));

  // Print the full output matrix so we can see the error pattern.
  std::cerr << "Output matrix (expected all " << K << "):\n";
  for (int r = 0; r < M; ++r) {
    std::cerr << "  row " << r << ": ";
    for (int c = 0; c < N; ++c) {
      std::cerr << cGpu[r * N + c] << " ";
    }
    std::cerr << "\n";
  }

  int mismatches = 0;
  for (int i = 0; i < M * N; ++i) {
    if (std::abs(cGpu[i] - static_cast<float>(K)) > 1e-2f) {
      if (mismatches < 10) {
        std::cerr << "MISMATCH [" << i / N << "][" << i % N
                  << "] got=" << cGpu[i] << " exp=" << K << "\n";
      }
      mismatches++;
    }
  }
  ASSERT_EQ(mismatches, 0) << "Total mismatches: " << mismatches;
}

// IREE-compiled 16x16x16 f16xf16xf32 matmul for gfx1151.
// MLIR source: tests/sources/iree/matmul_16x16x16_f16xf16xf32.mlir
// Assembly: tests/asm/gfx1151/iree_matmul_16x16x16_f16xf16xf32.s
//
// Smallest matmul that exercises v_wmma_f32_16x16x16_f16 on gfx1151.
// Single WMMA call (no loop): K=16 is within the 8-WMMA unrolled limit.
//
// Dispatch parameters:
//   .max_flat_workgroup_size: 32  -> blockDim = (32, 1, 1), 1 wave
//   Grid: (1, 1, 1)
//
// Kernel args: 3 pointers at offsets 0, 8, 16 (A f16, B f16, C f32).
TEST(Gfx1151, MatMul_IREE_F16xF16xF32_16x16x16) {
  const int M = 16, N = 16, K = 16;

  auto assembly = loadKernelFile("gfx1151/iree_matmul_16x16x16_f16xf16xf32.s");

  std::mt19937 rng(77);
  std::uniform_int_distribution<int> dist(0, 1);
  std::vector<float> choices = {1.0f, -1.0f};

  std::vector<float> aF32(M * K), bF32(K * N);
  for (auto &v : aF32) {
    v = choices[dist(rng)];
  }
  for (auto &v : bF32) {
    v = choices[dist(rng)];
  }

  std::vector<uint16_t> aF16(M * K), bF16(K * N);
  for (size_t i = 0; i < aF32.size(); ++i) {
    aF16[i] = floatToF16(aF32[i]);
  }
  for (size_t i = 0; i < bF32.size(); ++i) {
    bF16[i] = floatToF16(bF32[i]);
  }

  std::vector<float> cGpu(M * N, 0.0f);

  Emulator emulator(assembly, std::make_shared<Gfx1151>());

  uint16_t *aPtr = aF16.data();
  uint16_t *bPtr = bF16.data();
  float *cPtr = cGpu.data();

  emulator.addKernarg(0, &aPtr);
  emulator.addKernarg(1, &bPtr);
  emulator.addKernarg(2, &cPtr);

  emulator.run(Dim3d(0, 0, 0), Dim3d(32, 1, 1));

  std::vector<float> cRef;
  cpuGemmRowMajorF32(M, N, K, aF32, bF32, cRef);

  int mismatches = 0;
  for (int i = 0; i < M * N; ++i) {
    if (std::isnan(cGpu[i]) || std::isinf(cGpu[i]) ||
        std::abs(cGpu[i] - cRef[i]) > 1e-2f) {
      if (mismatches < 10) {
        std::cerr << "Mismatch at index " << i << ": expected " << cRef[i]
                  << ", got " << cGpu[i] << "\n";
      }
      mismatches++;
    }
  }
  ASSERT_EQ(mismatches, 0) << "Total mismatches: " << mismatches;
}

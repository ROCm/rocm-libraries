// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "race-emulator/Arch.h"
#include "race-emulator/Emulator.h"
#include "race-emulator/FloatTypes.h"
#include <filesystem> // Requires C++17
#include <fstream>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// TEST_KERNEL_DIR is provided by CMake
#ifndef TEST_KERNEL_DIR
#define TEST_KERNEL_DIR "undefined" // Dummy path to satisfy the IDE
#endif

namespace {

using namespace raceemulator;
namespace fs = std::filesystem;

std::string load_kernel_file(const std::string &filename) {
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

// Architecture parameter for multi-arch tests.
struct ArchParam {
  std::shared_ptr<Architecture> arch;
  WaveSize waveSize;

  ArchParam(std::string_view target)
      : arch(architectureFromTarget(target)), waveSize(arch->getWaveSize()) {}
};

Emulator loadEmulator(const ArchParam &arch, const std::string &filename) {
  std::string path = arch.arch->getName() + "/" + filename;
  std::string assembly = load_kernel_file(path);
  auto e = Emulator(assembly, arch.arch);
  return e;
}

const ArchParam kGfx942("gfx942");
const ArchParam kGfx950("gfx950");
const ArchParam kGfx1151("gfx1151");

// ============================================================================
// Shared test logic (called by the explicit TEST declarations below)
// ============================================================================

// HIP source (copy.s):
//   __global__ void copy_kernel(int* out, const int* in) {
//     int i = threadIdx.x;
//     out[i] = in[i];
//   }
void runCopyKernel(const ArchParam &arch) {
  auto emulator = loadEmulator(arch, "copy.s");
  EXPECT_EQ(emulator.getArch().getName(), arch.arch->getName());

  int N = 128;
  std::vector<int> h_in(N);
  std::iota(h_in.begin(), h_in.end(), 100);
  std::vector<int> h_out(N, -1);

  int *d_out = h_out.data();
  const int *d_in = h_in.data();
  emulator.addKernarg(0, &d_out);
  emulator.addKernarg(1, &d_in);
  emulator.run(Dim3d(0), {128, 1, 1}, {.raceChecks = true});

  for (int i = 0; i < N; ++i) {
    EXPECT_EQ(h_out[i], h_in[i])
        << "Mismatch at index " << i << " (wave " << i / arch.waveSize
        << ", lane " << i % arch.waveSize << ")";
  }
}

// HIP source (copy_indexed.s):
//   __global__ void copy_kernel(int* out, const int* in) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     out[i] = in[i];
//   }
void runCopyIndexed(const ArchParam &arch) {
  auto emulator = loadEmulator(arch, "copy_indexed.s");

  int ws = arch.waveSize;
  int N = 128;
  int nWorkgroups = N / ws;

  std::vector<int> h_in(N);
  std::iota(h_in.begin(), h_in.end(), 100);
  std::vector<int> h_out(N, -1);

  int *d_out = h_out.data();
  const int *d_in = h_in.data();
  emulator.addKernarg(0, &d_out);
  emulator.addKernarg(1, &d_in);

  std::vector<Dim3d> wgIds;
  for (int wg = 0; wg < nWorkgroups; ++wg) {
    wgIds.emplace_back(wg, 0, 0);
  }
  emulator.run(wgIds, {ws, 1, 1}, {.raceChecks = true});

  for (int i = 0; i < N; ++i) {
    EXPECT_EQ(h_out[i], h_in[i]) << "Mismatch at index " << i << " (workgroup "
                                 << i / ws << ", lane " << i % ws << ")";
  }
}

// HIP source (test_3d.s):
//   __global__ void test_3d_kernel(float *output) {
//     int flat_idx = threadIdx.x + (threadIdx.y * blockDim.x)
//                  + (threadIdx.z * blockDim.x * blockDim.y);
//     output[flat_idx] = flat_idx + 1.0f;
//   }
void runTest3dWorkgroup(const ArchParam &arch) {
  auto emulator = loadEmulator(arch, "test_3d.s");

  int N = 64; // 2 * 4 * 8 = 64 threads
  std::vector<float> outputs(N, -1.0f);
  float *d_output = outputs.data();
  emulator.addKernarg(0, &d_output);
  emulator.run({0, 0, 0}, {2, 4, 8}, {.raceChecks = true});

  for (int i = 0; i < N; ++i) {
    float expected = static_cast<float>(i) + 1.0f;
    EXPECT_EQ(outputs[i], expected) << "Mismatch at index " << i;
  }
}

// ============================================================================
// gfx942-only tests (not yet ported to multi-arch)
// ============================================================================

// HIP source (simple_adder_0.s):
//   template <int WG_SIZE>
//   __launch_bounds__(WG_SIZE)
//   __global__ void adder_kernel(float bias, const double *in0, float *out,
//                                const float *in1) {
//     unsigned i = WG_SIZE * blockIdx.x + threadIdx.x;
//     out[i] = in0[i] + in1[i] + bias;
//   }
void runMixedPrecisionAdder(const ArchParam &arch) {
  auto emulator = loadEmulator(arch, "simple_adder_0.s");

  int N = 1024;
  std::vector<double> h_in0(N);
  std::iota(h_in0.begin(), h_in0.end(), 17.0);
  std::vector<float> h_in1(N);
  std::iota(h_in1.begin(), h_in1.end(), 96.0);
  std::vector<float> h_out(N, 22.0);
  const float bias = 3.5f;

  const double *d_in0 = h_in0.data();
  const float *d_in1 = h_in1.data();
  float *d_out = h_out.data();
  emulator.addKernarg(0, &bias);
  emulator.addKernarg(1, &d_in0);
  emulator.addKernarg(2, &d_out);
  emulator.addKernarg(3, &d_in1);
  emulator.run(Dim3d(1), {256, 1, 1}, {.raceChecks = true});
  EXPECT_EQ(h_out[400], static_cast<float>(h_in0[400] + h_in1[400] + bias));
}

// HIP source (simple_adder_1.s):
//   __global__ void simple_adder(hip_bfloat16* c, float toAdd, int N) { ... }
void runBF16Adder(const ArchParam &arch) {
  auto emulator = loadEmulator(arch, "simple_adder_1.s");

  int N0 = 1235;
  int N1 = 100;

  std::vector<uint16_t> h_c(N0, floatToBf16(17.5f));
  uint16_t *d_c = h_c.data();
  float toAdd = 3.5f;

  emulator.addKernarg(0, &d_c);
  emulator.addKernarg(1, &toAdd);
  emulator.addKernarg(2, &N1);
  emulator.run(Dim3d(1), {64, 1, 1}, {.raceChecks = true});

  for (int i = 64; i < 128; ++i) {
    float expected = i < 100 ? 17.5f + toAdd : 17.5f;
    float actual = bf16ToFloat(h_c[i]);
    EXPECT_EQ(actual, expected) << "at index " << i;
  }
}

// HIP source (simple_adder_2.s): float version of adder
void runFloatAdder(const ArchParam &arch) {
  auto emulator = loadEmulator(arch, "simple_adder_2.s");

  int N0 = 1235;
  int N1 = 210;

  std::vector<float> h_c(N0, -1);
  std::iota(h_c.begin(), h_c.end(), 0.0);
  float *d_c = h_c.data();
  float toAdd = 3.5f;

  emulator.addKernarg(0, &d_c);
  emulator.addKernarg(1, &toAdd);
  emulator.addKernarg(2, &N1);
  emulator.run(Dim3d(1), {192, 1, 1}, {.raceChecks = true});

  for (int i = 1 * 3 * 64; i < 2 * 3 * 64; ++i) {
    float actual = h_c[i];
    float expected = i < N1 ? i + toAdd : i;
    EXPECT_EQ(actual, expected) << "at index " << i;
  }
}

// HIP source (double_evens.s):
//   __global__ void double_evens(float* c, float scale) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int elements_per_thread = 32;
//     int start = idx * elements_per_thread;
//     for (int i = 0; i < elements_per_thread; ++i) {
//       int current_index = start + i;
//       if (current_index % 2 == 0) { c[current_index] *= scale; }
//     }
//   }
void runDoubleEvens(const ArchParam &arch) {
  auto emulator = loadEmulator(arch, "double_evens.s");

  int N0 = 4 * 64 * 32;
  std::vector<float> h_c(N0);

  float offset = 3.0;
  std::iota(h_c.begin(), h_c.end(), offset);
  float *d_c = h_c.data();
  float scale = 2.0f;

  emulator.addKernarg(0, &d_c);
  emulator.addKernarg(1, &scale);
  emulator.run(Dim3d(0), {128, 1, 1}, {.raceChecks = true});

  int start = 0;
  int nChecks = 100;

  std::vector<float> expected_vec;
  for (int i = start; i < start + nChecks; ++i) {
    expected_vec.push_back((i % 2 == 0) ? (offset + i) * scale : offset + i);
  }

  std::vector<float> actual_vec(h_c.begin() + start,
                                h_c.begin() + start + nChecks);
  EXPECT_EQ(actual_vec, expected_vec);
}

// HIP source (ragged.s): ragged loop kernel
void runRaggedLoops(const ArchParam &arch) {
  auto emulator = loadEmulator(arch, "ragged.s");

  int N = 64;
  std::vector<int> h_limits(N);
  std::iota(h_limits.begin(), h_limits.end(), 0);
  for (auto &v : h_limits) {
    v = v % 10;
  }

  std::vector<int> h_data(N);
  std::iota(h_data.begin(), h_data.end(), 0);

  int *d_data = h_data.data();
  int *d_limits = h_limits.data();

  emulator.addKernarg(0, &d_data);
  emulator.addKernarg(1, &d_limits);
  emulator.run(Dim3d(0), {64, 1, 1}, {.raceChecks = true});

  std::vector<std::string> errors;
  for (int i = 0; i < N; ++i) {
    int expected = 0;
    for (int j = 0; j < i % 10; ++j) {
      expected += j;
    }
    int actual = h_data[i];
    if (actual != expected) {
      std::ostringstream oss;
      oss << "Thread " << i << ": expected " << expected << ", got " << actual;
      if (actual == 630 && expected == 0) {
        oss << " (ZOMBIE THREAD DETECTED)";
      }
      errors.push_back(oss.str());
    }
  }

  EXPECT_TRUE(errors.empty()) << "Found " << errors.size() << " mismatches:\n"
                              << [&]() {
                                   std::string s;
                                   for (const auto &e : errors) {
                                     s += e + "\n";
                                   }
                                   return s;
                                 }();
}

// HIP source (ragged_complex.s): like ragged.s but with a complex loop
// body (j*j + 2*j + j%3) that prevents the compiler from replacing the
// loop with a closed-form expression. This exercises divergent control
// flow where each thread iterates a different number of times.
void runRaggedComplex(const ArchParam &arch) {
  auto emulator = loadEmulator(arch, "ragged_complex.s");

  int N = 64;
  std::vector<int> h_limits(N);
  std::iota(h_limits.begin(), h_limits.end(), 0);
  for (auto &v : h_limits) {
    v = v % 10;
  }

  std::vector<int> h_data(N, 0);

  int *d_data = h_data.data();
  int *d_limits = h_limits.data();

  emulator.addKernarg(0, &d_data);
  emulator.addKernarg(1, &d_limits);
  emulator.run(Dim3d(0), {64, 1, 1}, {.raceChecks = true});

  std::vector<std::string> errors;
  for (int i = 0; i < N; ++i) {
    int expected = 0;
    int limit = i % 10;
    for (int j = 0; j < limit; ++j) {
      expected += j * j + 2 * j + j % 3;
    }
    int actual = h_data[i];
    if (actual != expected) {
      std::ostringstream oss;
      oss << "Thread " << i << ": expected " << expected << ", got " << actual;
      errors.push_back(oss.str());
    }
  }

  EXPECT_TRUE(errors.empty()) << "Found " << errors.size() << " mismatches:\n"
                              << [&]() {
                                   std::string s;
                                   for (const auto &e : errors) {
                                     s += e + "\n";
                                   }
                                   return s;
                                 }();
}

// HIP source (lds_reverse_2.s):
//   __global__ void lds_reverse(int *data) {
//     __shared__ int temp[256];
//     int tid = threadIdx.x;
//     temp[tid] = data[threadIdx.x];
//     __syncthreads();
//     data[tid] = temp[256 - tid - 1];
//   }
void runLdsReverse(const ArchParam &arch, bool enableRaceChecks = true) {
  auto emulator = loadEmulator(arch, "lds_reverse_2.s");

  int N = 256;
  std::vector<int> h_data(N);
  std::iota(h_data.begin(), h_data.end(), 0);
  int *d_data = h_data.data();

  emulator.addKernarg(0, &d_data);
  emulator.run(Dim3d(0), {256, 1, 1}, {.raceChecks = enableRaceChecks});

  std::vector<std::string> errors;
  for (int i = 0; i < N; ++i) {
    int expected = 256 - 1 - i;
    int actual = h_data[i];
    if (actual != expected) {
      std::ostringstream oss;
      oss << "Index " << i << ": expected " << expected << ", got " << actual;
      errors.push_back(oss.str());
    }
  }

  EXPECT_TRUE(errors.empty())
      << "Found " << errors.size() << " mismatches (Barrier failed?)\n"
      << [&]() {
           std::string s;
           for (const auto &e : errors) {
             s += e + "\n";
           }
           return s;
         }();
}

// HIP source (naive_matmul.s):
//   C[ty][tx] = sum_k(A[ty][k] * B[k][tx]) + C[ty][tx]
// Each thread computes one element. blockDim = (N, M).
void runNaiveMatmul(const ArchParam &arch) {
  auto emulator = loadEmulator(arch, "naive_matmul.s");

  // Non-square, non-power-of-2 dimensions to exercise bounds checking.
  // blockDim = {16, 16, 1} = 256 threads (4 waves on gfx942, 8 on gfx1151).
  // Only threads with tx < N && ty < M compute; the rest are inactive.
  int M = 10, N = 12, K = 14;
  // A = M x K, B = K x N, C = M x N
  // Random integers in [-4, 4) as floats. Fixed seed for reproducibility.
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(-4, 3);
  std::vector<float> A(M * K), B(K * N), C(M * N, 0.0f);
  for (auto &v : A) {
    v = static_cast<float>(dist(rng));
  }
  for (auto &v : B) {
    v = static_cast<float>(dist(rng));
  }

  float *d_C = C.data();
  const float *d_A = A.data();
  const float *d_B = B.data();

  emulator.addKernarg(0, &d_C);
  emulator.addKernarg(1, &d_A);
  emulator.addKernarg(2, &d_B);
  emulator.addKernarg(3, &M);
  emulator.addKernarg(4, &N);
  emulator.addKernarg(5, &K);
  emulator.run(Dim3d(0), {16, 16, 1},
               {.raceChecks = true}); // 256 threads, only MxN active

  // Compute expected: C[i][j] = sum_k A[i][k] * B[k][j]
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float expected = 0.0f;
      for (int k = 0; k < K; ++k) {
        expected += A[i * K + k] * B[k * N + j];
      }
      float actual = C[i * N + j];
      EXPECT_EQ(actual, expected) << "C[" << i << "][" << j << "]: expected "
                                  << expected << ", got " << actual;
    }
  }
}

// HIP source (lds_transpose.s):
//   Load MxN tile row-major into LDS, barrier, read transposed, store.
void runLdsTranspose(const ArchParam &arch, bool enableRaceChecks = true) {
  auto emulator = loadEmulator(arch, "lds_transpose.s");

  // Non-square dimensions with bounds checking.
  // blockDim = {16, 16, 1} = 256 threads (4 waves on gfx942, 8 on gfx1151).
  // Only threads within MxN bounds are active.
  int M = 10, N = 12;
  // Random integers in [-4, 4). Fixed seed for reproducibility.
  std::mt19937 rng(123);
  std::uniform_int_distribution<int> dist(-4, 3);
  std::vector<int> in(M * N), out(N * M, -1);
  for (auto &v : in) {
    v = dist(rng);
  }

  int *d_out = out.data();
  const int *d_in = in.data();

  emulator.addKernarg(0, &d_out);
  emulator.addKernarg(1, &d_in);
  emulator.addKernarg(2, &M);
  emulator.addKernarg(3, &N);

  // LDS size = M * N * sizeof(int) = 10 * 12 * 4 = 480 bytes
  emulator.run(Dim3d(0), {16, 16, 1}, {.raceChecks = enableRaceChecks});

  // Verify: out is NxM row-major, out[j][i] should be in[i][j]
  for (int j = 0; j < N; ++j) {
    for (int i = 0; i < M; ++i) {
      int expected = in[i * N + j];
      int actual = out[j * M + i];
      EXPECT_EQ(actual, expected) << "out[" << j << "][" << i << "]: expected "
                                  << expected << ", got " << actual;
    }
  }
}

// HIP source (simple_adder_f16.cpp):
//   __global__ void simple_adder_f16(__half *c, float toAdd, int N) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < N) {
//       c[idx] = __float2half(__half2float(c[idx]) + toAdd);
//     }
//   }
void runF16Adder(const ArchParam &arch) {
  auto emulator = loadEmulator(arch, "simple_adder_f16.s");

  int N0 = 1235;
  int N1 = 100;

  std::vector<uint16_t> h_c(N0, floatToF16(17.5f));
  uint16_t *d_c = h_c.data();
  float toAdd = 3.5f;

  emulator.addKernarg(0, &d_c);
  emulator.addKernarg(1, &toAdd);
  emulator.addKernarg(2, &N1);
  emulator.run(Dim3d(1), {arch.waveSize, 1, 1}, {.raceChecks = true});

  for (int i = arch.waveSize; i < 2 * arch.waveSize; ++i) {
    float expected = i < 100 ? 17.5f + toAdd : 17.5f;
    float actual = f16ToFloat(h_c[i]);
    EXPECT_EQ(actual, expected) << "at index " << i;
  }
}

// HIP source (f16_round_trip.cpp):
//   __global__ void f16_round_trip(float *data, int N) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < N) {
//       data[idx] = __half2float(__float2half(data[idx]));
//     }
//   }
// Round-trips f32 through f16, testing v_cvt_f16_f32 then v_cvt_f32_f16.
// Includes special values: zero, inf, nan, overflow, and normal values.
void runF16RoundTrip(const ArchParam &arch) {
  auto emulator = loadEmulator(arch, "f16_round_trip.s");

  float inf = std::numeric_limits<float>::infinity();
  float nan = std::numeric_limits<float>::quiet_NaN();

  // Input values and expected outputs after f32 -> f16 -> f32 round-trip.
  // Values exactly representable in f16 survive; others get clamped/truncated.
  std::vector<std::pair<float, float>> cases = {
      {0.0f, 0.0f},   {-0.0f, -0.0f}, {1.0f, 1.0f},
      {-1.0f, -1.0f}, {0.5f, 0.5f},   {65504.0f, 65504.0f}, // max f16
      {inf, inf},     {-inf, -inf},   {100000.0f, inf},     // overflow -> inf
  };

  int N = static_cast<int>(cases.size()) + 1; // +1 for NaN
  std::vector<float> h_data(N);
  for (int i = 0; i < static_cast<int>(cases.size()); ++i) {
    h_data[i] = cases[i].first;
  }
  h_data[N - 1] = nan;

  float *d_data = h_data.data();
  emulator.addKernarg(0, &d_data);
  emulator.addKernarg(1, &N);
  emulator.run(Dim3d(0), {arch.waveSize, 1, 1}, {.raceChecks = true});

  for (int i = 0; i < static_cast<int>(cases.size()); ++i) {
    EXPECT_EQ(h_data[i], cases[i].second)
        << "Round-trip failed for input " << cases[i].first << " at index "
        << i;
  }
  EXPECT_TRUE(std::isnan(h_data[N - 1])) << "NaN did not survive round-trip";
}

} // namespace

// ============================================================================
// Test manifest: which tests run on which architectures.
//
// Each line is a declaration: TEST(ArchSuite, TestName) { runXxx(kArch); }
// To add architecture support for a test, add the .s file to the asm/<arch>/
// directory and add a TEST line here.
// ============================================================================

// --- Tests running on both gfx942 and gfx1151 ---
TEST(Gfx942, CopyKernel) { runCopyKernel(kGfx942); }
TEST(Gfx950, CopyKernel) { runCopyKernel(kGfx950); }
TEST(Gfx1151, CopyKernel) { runCopyKernel(kGfx1151); }

TEST(Gfx942, CopyIndexed) { runCopyIndexed(kGfx942); }
TEST(Gfx1151, CopyIndexed) { runCopyIndexed(kGfx1151); }

TEST(Gfx942, Test3dWorkgroup) { runTest3dWorkgroup(kGfx942); }
TEST(Gfx1151, Test3dWorkgroup) { runTest3dWorkgroup(kGfx1151); }

TEST(Gfx942, FloatAdder) { runFloatAdder(kGfx942); }
TEST(Gfx1151, FloatAdder) { runFloatAdder(kGfx1151); }

TEST(Gfx942, RaggedLoops) { runRaggedLoops(kGfx942); }
TEST(Gfx1151, RaggedLoops) { runRaggedLoops(kGfx1151); }

TEST(Gfx942, RaggedComplex) { runRaggedComplex(kGfx942); }
TEST(Gfx1151, RaggedComplex) { runRaggedComplex(kGfx1151); }

TEST(Gfx942, MixedPrecisionAdder) { runMixedPrecisionAdder(kGfx942); }
TEST(Gfx1151, MixedPrecisionAdder) { runMixedPrecisionAdder(kGfx1151); }

TEST(Gfx942, DoubleEvens) { runDoubleEvens(kGfx942); }
TEST(Gfx1151, DoubleEvens) { runDoubleEvens(kGfx1151); }

TEST(Gfx942, LdsReverse) { runLdsReverse(kGfx942); }
TEST(Gfx1151, LdsReverse) { runLdsReverse(kGfx1151); }
TEST(Gfx942, LdsReverseNoRaceChecks) { runLdsReverse(kGfx942, false); }
TEST(Gfx1151, LdsReverseNoRaceChecks) { runLdsReverse(kGfx1151, false); }

TEST(Gfx942, BF16Adder) { runBF16Adder(kGfx942); }
TEST(Gfx1151, BF16Adder) { runBF16Adder(kGfx1151); }

TEST(Gfx942, NaiveMatmul) { runNaiveMatmul(kGfx942); }
TEST(Gfx1151, NaiveMatmul) { runNaiveMatmul(kGfx1151); }

TEST(Gfx942, LdsTranspose) { runLdsTranspose(kGfx942); }
TEST(Gfx1151, LdsTranspose) { runLdsTranspose(kGfx1151); }
TEST(Gfx942, LdsTransposeNoRaceChecks) { runLdsTranspose(kGfx942, false); }
TEST(Gfx1151, LdsTransposeNoRaceChecks) {
  runLdsTranspose(kGfx1151, false);
}

TEST(Gfx942, F16Adder) { runF16Adder(kGfx942); }
TEST(Gfx1151, F16Adder) { runF16Adder(kGfx1151); }

TEST(Gfx942, F16RoundTrip) { runF16RoundTrip(kGfx942); }
TEST(Gfx1151, F16RoundTrip) { runF16RoundTrip(kGfx1151); }

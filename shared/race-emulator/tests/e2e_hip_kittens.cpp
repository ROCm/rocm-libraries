// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "race-emulator/Emulator.h"
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

// Helper function to read file content
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

uint16_t float_to_bf16(float f) {
  uint32_t bits = std::bit_cast<uint32_t>(f);
  return static_cast<uint16_t>(bits >> 16);
}

float bf16_to_float(uint16_t bf) {
  uint32_t bits = static_cast<uint32_t>(bf) << 16;
  return std::bit_cast<float>(bits);
}

} // namespace

TEST(Gfx942, MatMul_Kittens) {

  auto start = std::chrono::high_resolution_clock::now();

  std::string assembly = load_kernel_file("gfx942/kittens_original_smaller.s");

  auto emulator = Emulator::createGfx942(assembly);
  emulator.setRaceChecks(true);

  auto emulatorInitialized = std::chrono::high_resolution_clock::now();

  // See HIP code:
  int M = 1 * 256;
  int N = 1 * 256;
  int K = 128;
  std::vector<uint16_t> h_a(M * K);
  std::vector<uint16_t> h_b(N * K);
  std::vector<uint16_t> h_c(M * N);

  std::vector<float> h_a_f32(M * K);
  std::vector<float> h_b_f32(N * K);

  uint32_t seed = 1011;
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(0, 1);
  std::vector<float> choices = {-1.0f, 1.0f};

  int wg_m = M; // 256;
  int wg_n = N; // 256;

  for (int row = 0; row < wg_m; ++row) {
    for (int col = 0; col < K; ++col) {
      float choice = choices[dist(rng)];
      h_a[row * K + col] = float_to_bf16(choice);
      h_a_f32[row * K + col] = choice;
    }
  }

  for (int row = 0; row < wg_n; ++row) {
    for (int col = 0; col < K; ++col) {
      float choice = choices[dist(rng)];
      h_b[row * K + col] = float_to_bf16(choice);
      h_b_f32[row * K + col] = choice;
    }
  }

  auto endOfInitialization = std::chrono::high_resolution_clock::now();

  // Compute the baseline result on the CPU for verification:
  std::vector<float> cpu_c(M * N, 0.0f);
  for (int m = 0; m < wg_m; ++m) {
    for (int n = 0; n < wg_n; ++n) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        float a_val = h_a_f32[m * K + k];
        float b_val = h_b_f32[n * K + k];
        sum += a_val * b_val;
      }
      cpu_c[m * N + n] = sum;
    }
  }

  auto endOfHostCompute = std::chrono::high_resolution_clock::now();

  struct KittenArg {
    void *addr;     // 8 Bytes (The pointer)
    uint64_t extra; // 8 Bytes (Padding, stride, or metadata)
  };
  static_assert(sizeof(KittenArg) == 16, "KittenArg must be 16 bytes");

  // Create the full 16-byte argument
  KittenArg argC = {h_c.data(),
                    0}; // Initialize 'extra' to 0 (or correct value)

  // Pass the address of the 16-byte struct
  emulator.addKernarg(2, &argC);

  KittenArg argA = {h_a.data(), 0};
  emulator.addKernarg(0, &argA);

  KittenArg argB = {h_b.data(), 0};
  emulator.addKernarg(1, &argB);

  emulator.run(0, {512, 1, 1}); // 8 waves * 64 threads/wave = 512 threads

  auto endOfEmulatorRun = std::chrono::high_resolution_clock::now();

  // Verification
  for (int m = 0; m < 256; ++m) {
    for (int n = 0; n < 256; ++n) {
      // This casting is super duper critical to get correct values!
      float expected = bf16_to_float(float_to_bf16(cpu_c[m * N + n]));
      float actual = bf16_to_float(h_c[m * N + n]);
      EXPECT_EQ(actual, expected) << "at index (" << m << ", " << n << ")";
    }
  }

  auto endOfVerification = std::chrono::high_resolution_clock::now();

  // Print the time intervals of all of the steps:
  // Initialization:
  auto emulatorInitTime = std::chrono::duration_cast<std::chrono::milliseconds>(
                              emulatorInitialized - start)
                              .count();

  auto initDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
                          endOfInitialization - emulatorInitialized)
                          .count();

  auto hostComputeDuration =
      std::chrono::duration_cast<std::chrono::milliseconds>(endOfHostCompute -
                                                            endOfInitialization)
          .count();

  auto emulatorRunDuration =
      std::chrono::duration_cast<std::chrono::milliseconds>(endOfEmulatorRun -
                                                            endOfHostCompute)
          .count();

  auto verificationDuration =
      std::chrono::duration_cast<std::chrono::milliseconds>(endOfVerification -
                                                            endOfEmulatorRun)
          .count();

  std::cerr << "\nTiming Summary (milliseconds):\n";
  std::cerr << "  Emulator initialization: " << emulatorInitTime << " ms\n";
  std::cerr << "           Initialization: " << initDuration << " ms\n";
  std::cerr << "Host Compute (CPU MatMul): " << hostComputeDuration << " ms\n";
  std::cerr << "             Emulator Run: " << emulatorRunDuration << " ms\n";
  std::cerr << "             Verification: " << verificationDuration << " ms\n";

  std::cerr << std::endl;
}

#include "race-emulator/Emulator.h"
#include <filesystem> // Requires C++17
#include <fstream>
#include <gtest/gtest.h>
#include <numeric>
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

} // namespace

TEST(GpuEmulatorTestSuiteZero, EndToEndTestZero) {
  // Assembly generated for the following HIP kernel:
  //
  // Kernel: out[i] = in0[i] + in1[i] + bias
  //
  // template <int WG_SIZE>
  // __launch_bounds__(WG_SIZE)
  // __global__ void adder_kernel(float bias, const double *in0, float *out,
  // const float *in1) {
  //   unsigned i = WG_SIZE * blockIdx.x + threadIdx.x;
  //   out[i] = in0[i] + in1[i] + bias;
  // }

  std::string assembly = load_kernel_file("simple_adder_0.s");
  auto emulator = Emulator::createGfx942(assembly);

  int N = 1024;

  // 2. Prepare Operands
  std::vector<double> h_in0(N);
  std::iota(h_in0.begin(), h_in0.end(), 17.0);
  std::vector<float> h_in1(N);
  std::iota(h_in1.begin(), h_in1.end(), 96.0);
  std::vector<float> h_out(N, 22.0);
  const float bias = 3.5f;

  // Pointers that would exist on the device
  // (here, just host pointers)
  const double *d_in0 = h_in0.data();
  const float *d_in1 = h_in1.data();
  float *d_out = h_out.data();
  emulator.addKernarg(0, &bias);
  emulator.addKernarg(1, &d_in0);
  emulator.addKernarg(2, &d_out);
  emulator.addKernarg(3, &d_in1);
  emulator.run(1, {256, 1, 1}); // 4 waves * 64 threads/wave = 256 threads
  EXPECT_EQ(h_out[400], static_cast<float>(h_in0[400] + h_in1[400] + bias));
}

// Helper to convert float -> bfloat16 (as uint16_t)
uint16_t float_to_bf16(float f) {
  uint32_t bits = std::bit_cast<uint32_t>(f);
  // Standard bfloat16 conversion usually just truncates the lower 16 bits.
  // (Note: Some hardware rounds, but truncation is the simplest software model
  // without a library. If your emulator implements rounding, add 0x8000 before
  // shifting).
  return static_cast<uint16_t>(bits >> 16);
}

// Helper to convert bfloat16 (as uint16_t) -> float
float bf16_to_float(uint16_t bf) {
  uint32_t bits = static_cast<uint32_t>(bf) << 16;
  return std::bit_cast<float>(bits);
}

TEST(GpuEmulatorTestSuiteZero, EndToEndTestOne) {

  // Assembly generated for the following HIP kernel:
  //
  // __global__ void simple_adder(hip_bfloat16* c, float toAdd, int N) {
  //  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  //  int elements_per_thread = 1;
  //  int start = idx * elements_per_thread;

  //  for (int i = 0; i < elements_per_thread; ++i) {
  //      int current_index = start + i;
  //      if (current_index < N) {
  //        c[i] += hip_bfloat16(toAdd);
  //      }
  //   }
  //}

  std::string assembly = load_kernel_file("simple_adder_1.s");
  auto emulator = Emulator::createGfx942(assembly);
  emulator.enableRaceChecks(true);

  int N0 = 1235;
  int N1 = 100;

  // Inialize the data.
  std::vector<uint16_t> h_c(N0, float_to_bf16(17.5f));
  uint16_t *d_c = h_c.data();
  float toAdd = 3.5f;

  std::cerr << "The address (base) of the data vector is " << h_c.data()
            << "\n";

  emulator.addKernarg(0, &d_c);
  emulator.addKernarg(1, &toAdd);
  emulator.addKernarg(2, &N1);

  emulator.run(1, {64, 1, 1}); // 1 wave * 64 threads/wave = 64 threads

  // We just ran workgroup 4, with 2 subgroups per workgroup (128 threads per
  // workgroup). So we just ran the threads in [4*128, 5*128) = [512, 640).
  for (int i = 64; i < 128; ++i) {
    float expected = i < 100 ? 17.5f + toAdd : 17.5f;
    float actual = bf16_to_float(h_c[i]);

    EXPECT_EQ(actual, expected) << "at index " << i;
  }
}

void print(const std::vector<float> &vec) {
  for (size_t i = 0; i < vec.size(); ++i) {
    std::cout << "Index " << i << ": " << vec[i] << std::endl;
  }
}

TEST(GpuEmulatorTestSuiteZero, EndToEndTestTwo) {
  // Like test 1 but float all the way.

  std::string assembly = load_kernel_file("simple_adder_2.s");
  auto emulator = Emulator::createGfx942(assembly);

  // The N allocated.
  int N0 = 1235;

  // The N to process.
  int N1 = 210;

  // Inialize the data.
  std::vector<float> h_c(N0, -1);
  std::iota(h_c.begin(), h_c.end(), 0.0);
  float *d_c = h_c.data();
  float toAdd = 3.5f;

  emulator.addKernarg(0, &d_c);
  emulator.addKernarg(1, &toAdd);
  emulator.addKernarg(2, &N1);

  // [0, 64*3) = [0, 192)
  // emulator.run(0, {192, 1, 1});
  // [64*3, 64*6) = [192, 384)
  emulator.run(1, {192, 1, 1}); // 3 waves * 64 threads/wave = 192 threads

  for (int i = 1 * 3 * 64; i < 2 * 3 * 64; ++i) {
    float actual = h_c[i];
    float expected = i < N1 ? i + toAdd : i;
    EXPECT_EQ(actual, expected) << "at index " << i;
  }
}

// Generated for
//
// __global__ void double_evens(float* c, float scale) {
//    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//    int elements_per_thread = 32;
//    int start = idx * elements_per_thread;
//    for (int i = 0; i < elements_per_thread; ++i) {
//        int current_index = start + i;
//        if (current_index % 2 == 0){
//          c[current_index] *= scale;
//        }
//    }
//}
TEST(GpuEmulatorTestSuiteZero, DoubleEvens) {
  std::string assembly = load_kernel_file("double_evens.s");
  auto emulator = Emulator::createGfx942(assembly);

  int N0 = 4 * 64 * 32;

  // Inialize the data.
  std::vector<float> h_c(N0);

  float offset = 3.0;
  std::iota(h_c.begin(), h_c.end(), offset);
  float *d_c = h_c.data();
  float scale = 2.0f;

  emulator.addKernarg(0, &d_c);
  emulator.addKernarg(1, &scale);

  // Run subgroup 2 of workgroup 2 (threads [256, 384) )
  emulator.run(0, {128, 1, 1}); // 2 waves * 64 threads/wave = 128 threads
  int start = 0;
  int nChecks = 100;

  // 1. Generate Expected Data
  std::vector<float> expected_vec;
  for (int i = start; i < start + nChecks; ++i) {
    expected_vec.push_back((i % 2 == 0) ? (offset + i) * scale : offset + i);
  }

  // 2. Extract Actual Data (Copy sub-range for comparison)
  std::vector<float> actual_vec(h_c.begin() + start,
                                h_c.begin() + start + nChecks);

  // 3. Compare entire vectors
  // Note: This does strict float equality. For approximate, Option 2 is better.
  EXPECT_EQ(actual_vec, expected_vec);
}

TEST(GpuEmulatorTestSuiteZero, RaggedLoops_StressTest) {
  std::string assembly = load_kernel_file("ragged.s");
  auto emulator = Emulator::createGfx942(assembly);

  int N = 64;

  std::vector<int> h_limits(N);
  std::iota(h_limits.begin(), h_limits.end(), 0);
  for (auto &v : h_limits) {
    v = v % 10;
  }

  // 2. Setup Output Data.
  std::vector<int> h_data(N);
  std::iota(h_data.begin(), h_data.end(), 0);

  // 3. Register Pointers
  int *d_data = h_data.data();
  int *d_limits = h_limits.data();

  emulator.addKernarg(0, &d_data);
  emulator.addKernarg(1, &d_limits);

  // 4. Run 1 Wavefront (64 threads)
  emulator.run(0, {64, 1, 1}); // 1 wave * 64 threads/wave = 64 threads

  // 5. Verification
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
      // If we got the max value (630) in thread 0, we know exactly what broke.
      if (actual == 630 && expected == 0)
        oss << " (ZOMBIE THREAD DETECTED)";
      errors.push_back(oss.str());
    }
  }

  EXPECT_TRUE(errors.empty()) << "Found " << errors.size() << " mismatches:\n"
                              << [&]() {
                                   std::string s;
                                   for (const auto &e : errors)
                                     s += e + "\n";
                                   return s;
                                 }();
}

TEST(GpuEmulatorUnitTests, LdsReverse1) {

  // From HIP:
  //
  //  __global__ void lds_reverse(int *data) {
  //   __shared__ int temp[256];
  //   int tid = threadIdx.x;
  //   temp[tid] = data[threadIdx.x];
  //   __syncthreads();
  //   data[tid] = temp[256 - tid - 1];
  // }
  std::string assembly = load_kernel_file("lds_reverse_2.s");

  auto emulator = Emulator::createGfx942(assembly);
  emulator.enableRaceChecks(true);

  int N = 256;
  std::vector<int> h_data(N);
  std::iota(h_data.begin(), h_data.end(), 0);
  int *d_data = h_data.data();

  emulator.addKernarg(0, &d_data);
  emulator.run(0, {256, 1, 1}); // 4 waves * 64 threads/wave = 256 threads

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
           for (const auto &e : errors)
             s += e + "\n";
           return s;
         }();
}

// test_3d.s
// See hip/test_3d.cpp

TEST(GpuEmulatorUnitTests, Test3dWorkgroup) {

  std::vector<float> outputs(64, -1.0f);

  std::string assembly = load_kernel_file("test_3d.s");
  auto emulator = Emulator::createGfx942(assembly);
  float *d_output = outputs.data();
  emulator.addKernarg(0, &d_output);
  emulator.run({0, 0, 0},
               {2, 4, 8}); // 1 block with dimensions (2, 4, 8) = 64 threads

  // Numerical validation: kernel writes output[flat_idx] = flat_idx + 1.0
  // where flat_idx = tid_x + (tid_y * 2) + (tid_z * 2 * 4)
  for (int i = 0; i < 64; ++i) {
    float expected = static_cast<float>(i) + 1.0f;
    EXPECT_EQ(outputs[i], expected) << "Mismatch at index " << i;
  }
}

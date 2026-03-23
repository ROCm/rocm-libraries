// GPU validation test for kernarg preloading on gfx942.
//
// Loads the preload_add_one.co code object (assembled at build time from
// tests/asm/gfx942/preload_add_one.s) and runs it on the GPU. The kernel
// uses .amdhsa_user_sgpr_kernarg_preload_length to receive a pointer
// argument directly in SGPRs without an s_load instruction.
//
// This validates:
// 1. The hand-crafted .s assembles to a valid code object
// 2. Kernarg preloading works correctly on real hardware
// 3. The emulator's preload behavior matches hardware

#include "race-emulator/Emulator.h"
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <sstream>
#include <string>
#include <vector>

#ifndef TEST_CO_DIR
#define TEST_CO_DIR "undefined"
#endif

#ifndef TEST_KERNEL_DIR
#define TEST_KERNEL_DIR "undefined"
#endif

#define HIP_CHECK(expr)                                                        \
  do {                                                                         \
    hipError_t err = (expr);                                                   \
    if (err != hipSuccess) {                                                   \
      FAIL() << "HIP error: " << hipGetErrorString(err) << " at "             \
             << __FILE__ << ":" << __LINE__;                                   \
    }                                                                          \
  } while (0)

namespace {

namespace fs = std::filesystem;
using namespace raceemulator;

std::string loadFile(const std::string &path) {
  std::ifstream file(path);
  if (!file.is_open())
    throw std::runtime_error("Failed to open: " + path);
  std::stringstream buf;
  buf << file.rdbuf();
  return buf.str();
}

} // namespace

TEST(GpuValidation, KernargPreload_AddOne) {
  const int N = 64;

  // --- GPU side ---
  // Load code object
  fs::path coPath = fs::path(TEST_CO_DIR) / "preload_add_one.co";
  hipModule_t module;
  HIP_CHECK(hipModuleLoad(&module, coPath.c_str()));

  hipFunction_t function;
  HIP_CHECK(hipModuleGetFunction(&function, module, "preload_add_one"));

  // Allocate and initialize GPU memory
  int32_t *d_data;
  HIP_CHECK(hipMalloc(&d_data, N * sizeof(int32_t)));
  std::vector<int32_t> h_data(N);
  for (int i = 0; i < N; ++i)
    h_data[i] = i * 10;
  HIP_CHECK(
      hipMemcpy(d_data, h_data.data(), N * sizeof(int32_t), hipMemcpyHostToDevice));

  // Launch kernel: 1 workgroup, 64 threads
  // Kernel arg: single pointer
  struct {
    int32_t *ptr;
  } args;
  args.ptr = d_data;

  size_t argSize = sizeof(args);
  void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSize,
                    HIP_LAUNCH_PARAM_END};
  HIP_CHECK(hipModuleLaunchKernel(function, 1, 1, 1, 64, 1, 1, 0, nullptr,
                                  nullptr, config));
  HIP_CHECK(hipDeviceSynchronize());

  // Read back GPU results
  std::vector<int32_t> gpuResults(N);
  HIP_CHECK(hipMemcpy(gpuResults.data(), d_data, N * sizeof(int32_t),
                       hipMemcpyDeviceToHost));

  // --- Emulator side ---
  auto assembly =
      loadFile(fs::path(TEST_KERNEL_DIR) / "gfx942" / "preload_add_one.s");
  Emulator emulator(assembly, std::make_shared<Gfx942>());

  std::vector<int32_t> emuData(N);
  for (int i = 0; i < N; ++i)
    emuData[i] = i * 10;

  int32_t *emuPtr = emuData.data();
  emulator.addKernarg(0, &emuPtr);
  emulator.run({0, 0, 0}, {64, 1, 1});

  // --- Compare GPU vs Emulator ---
  for (int i = 0; i < N; ++i) {
    EXPECT_EQ(gpuResults[i], i * 10 + 1)
        << "GPU mismatch at index " << i;
    EXPECT_EQ(emuData[i], i * 10 + 1)
        << "Emulator mismatch at index " << i;
    EXPECT_EQ(gpuResults[i], emuData[i])
        << "GPU vs Emulator mismatch at index " << i;
  }

  HIP_CHECK(hipFree(d_data));
  HIP_CHECK(hipModuleUnload(module));
}

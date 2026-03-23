// GPU hardware validation for gfx1151: dual-issue instruction execution.
//
// Runs a hand-crafted kernel on real gfx1151 hardware and through the
// emulator, then compares results bit-for-bit.
//
// DualIssue.v_dual_and_b32
//   Regression test for the bug where the emulator's VOPD dual-issue path
//   silently dropped unrecognised second ops (returning nullptr without
//   throwing), leaving the destination register at 0 instead of the correct
//   value. This caused numerical corruption that was indistinguishable from
//   a kernel logic bug and took several days to diagnose.
//
//   Kernel: dual_and_probe.s
//   Instruction under test: v_dual_and_b32 (second half of a VOPD pair)
//     v_dual_mov_b32 v3, v1 :: v_dual_and_b32 v138, 0xf0, v108
//   Expected v138[lane] = (lane * 16) & 0xf0

#include "race-emulator/Arch.h"
#include "race-emulator/Emulator.h"
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
      throw std::runtime_error(std::string("HIP error at " __FILE__ ":") +    \
                               std::to_string(__LINE__) + ": " +              \
                               hipGetErrorString(err));                        \
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

std::vector<uint32_t> runOnGpu(const char *coName, const char *kernelName,
                                int numThreads, size_t outputBytes) {
  fs::path coPath = fs::path(TEST_CO_DIR) / coName;
  hipModule_t mod;
  HIP_CHECK(hipModuleLoad(&mod, coPath.c_str()));

  hipFunction_t fn;
  HIP_CHECK(hipModuleGetFunction(&fn, mod, kernelName));

  uint32_t *d_out;
  HIP_CHECK(hipMalloc(&d_out, outputBytes));
  HIP_CHECK(hipMemset(d_out, 0x77, outputBytes));

  struct {
    uint32_t *ptr;
  } args;
  args.ptr = d_out;
  size_t argSize = sizeof(args);
  void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSize,
                    HIP_LAUNCH_PARAM_END};
  HIP_CHECK(hipModuleLaunchKernel(fn, 1, 1, 1, numThreads, 1, 1, 0, nullptr,
                                  nullptr, config));
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<uint32_t> result(outputBytes / sizeof(uint32_t));
  HIP_CHECK(
      hipMemcpy(result.data(), d_out, outputBytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipModuleUnload(mod));
  return result;
}

std::vector<uint32_t> runOnEmulator(const char *sName, int numThreads,
                                     size_t outputBytes) {
  auto assembly =
      loadFile(fs::path(TEST_KERNEL_DIR) / "gfx1151" / sName);
  Emulator emulator(assembly, std::make_shared<Gfx1151>());

  std::vector<uint32_t> out(outputBytes / sizeof(uint32_t), 0x77777777u);
  uint32_t *ptr = out.data();
  emulator.addKernarg(0, &ptr);
  emulator.run({0, 0, 0}, {numThreads, 1, 1});
  return out;
}

} // namespace

// Regression test: v_dual_and_b32 in the second slot of a VOPD pair.
//
// Kernel (dual_and_probe.s) executes:
//   v_dual_mov_b32 v3, v1 :: v_dual_and_b32 v138, 0xf0, v108
// where v108 = threadId * 16.
//
// Per-lane output (3 dwords at stride 12):
//   [0] threadId
//   [1] v138 = (threadId * 16) & 0xf0
//   [2] v3   = 0  (from v_dual_mov_b32 side)
TEST(DualIssue, v_dual_and_b32) {
  const int kLanes = 32;
  const size_t kOutputBytes = kLanes * 3 * sizeof(uint32_t);

  auto hw = runOnGpu("dual_and_probe.co", "dual_and_probe", kLanes,
                     kOutputBytes);
  auto emu =
      runOnEmulator("dual_and_probe.s", kLanes, kOutputBytes);

  for (int lane = 0; lane < kLanes; ++lane) {
    uint32_t exp_v138 = (static_cast<uint32_t>(lane) * 16u) & 0xf0u;
    uint32_t exp_v3 = 0u;

    int base = lane * 3;
    EXPECT_EQ(hw[base + 0], static_cast<uint32_t>(lane))
        << "HW threadId lane=" << lane;
    EXPECT_EQ(hw[base + 1], exp_v138) << "HW v138 lane=" << lane;
    EXPECT_EQ(hw[base + 2], exp_v3) << "HW v3 lane=" << lane;

    EXPECT_EQ(emu[base + 1], exp_v138)
        << "EMU v138 lane=" << lane
        << " (if 0x77777777: v_dual_and_b32 is being silently skipped)";
    EXPECT_EQ(emu[base + 2], exp_v3) << "EMU v3 lane=" << lane;

    EXPECT_EQ(hw[base + 1], emu[base + 1])
        << "HW!=EMU v138 lane=" << lane;
  }
}

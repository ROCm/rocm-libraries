// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tensilelite/Compiler.hpp>
#include <tensilelite/codegen/Kernel.hpp>

#include "ExpectLines.hpp"
#include "HipTestSupport.hpp"

#include "container.hpp"
#include "instruction/common.hpp"
#include "instruction/mem.hpp"

#include <gtest/gtest.h>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {

using namespace rocm::tensilelite;
using namespace rocm::tensilelite::codegen;
using namespace rocm::tensilelite::codegen::test;

TEST(KernelLaunch, EmptyKernelAssemblesLoadsAndLaunches) {
  Target target = Target::get("gfx950");
  Kernel kernel(target, "codegen_empty_kernel", 256);
  kernel.signature().addPointer("out");
  kernel.builder().createEndProgram();

  // Assembling is checked before the device so it runs without a GPU.
  std::vector<uint8_t> hsaco =
      assembleToHSACO(kernel.toAssembly(), kernel.target().isaName());
  EXPECT_FALSE(hsaco.empty());

  if (!hasHipDevice())
    GTEST_SKIP() << "no HIP device";

  hipModule_t rawModule = nullptr;
  ASSERT_HIP_SUCCESS(hipModuleLoadData(&rawModule, hsaco.data()));
  ModulePtr module(rawModule);
  hipFunction_t function = nullptr;
  ASSERT_HIP_SUCCESS(
      hipModuleGetFunction(&function, rawModule, "codegen_empty_kernel"));

  void *out = nullptr;
  void *args[] = {&out};
  ASSERT_HIP_SUCCESS(hipModuleLaunchKernel(function, 4, 1, 1, 256, 1, 1, 0,
                                           nullptr, args, nullptr));
  ASSERT_HIP_SUCCESS(hipDeviceSynchronize());
}

constexpr int roundTripGroupSize = 256;
constexpr int roundTripGroups = 4;
constexpr int roundTripElements = roundTripGroupSize * roundTripGroups;
constexpr const char *roundTripName = "codegen_agpr_round_trip";

// Loads the in and out pointers from the kernarg segment into s[4:7].
RegisterRange loadRoundTripPointers(Kernel &kernel) {
  IRBuilder &builder = kernel.builder();
  RegisterRange pointers = builder.allocateSgpr(4, 4);
  builder.create<rocisa::SLoadB128>(pointers.container(),
                                    kernel.kernargSegmentPtr().container(), 0);
  builder.createWaitCnt({.scalarMemory = 0});
  return pointers;
}

RegisterRange computeGlobalId(Kernel &kernel) {
  IRBuilder &builder = kernel.builder();
  RegisterRange globalId = builder.allocateVgpr(1);
  RegisterRange groupBase = builder.allocateSgpr(1);
  // The workitem id register packs the x, y and z ids.
  builder.create<rocisa::VAndB32>(globalId.container(), 0x3ff,
                                  kernel.workitemId().container());
  builder.create<rocisa::SMulI32>(groupBase.container(),
                                  kernel.workgroupIdX().container(),
                                  kernel.workgroupSize());
  builder.create<rocisa::VAddU32>(globalId.container(), groupBase.container(),
                                  globalId.container());
  return globalId;
}

// Moves each element through an AGPR. Index 3 of 4 is used rather than 0 so
// that a wrong accum_offset or next_free_vgpr would corrupt the result. No wait
// states are needed between the AGPR write and read on gfx90a and later.
void emitAgprRoundTrip(Kernel &kernel, const RegisterRange &pointers,
                       const RegisterRange &globalId) {
  IRBuilder &builder = kernel.builder();
  RegisterRange byteOffset = builder.allocateVgpr(1);
  builder.create<rocisa::VLShiftLeftB32>(byteOffset.container(), 2,
                                         globalId.container());
  RegisterRange value = builder.allocateVgpr(1);
  builder.create<rocisa::GlobalLoadB32>(
      value.container(), byteOffset.container(), pointers.sub(0, 2));
  builder.createWaitCnt({.vectorLoads = 0});
  RegisterRange acc = builder.allocateAgpr(4);
  builder.create<rocisa::VAccvgprWriteB32>(acc.sub(3), value.container());
  RegisterRange result = builder.allocateVgpr(1);
  builder.create<rocisa::VAccvgprReadB32>(result.container(), acc.sub(3));
  builder.create<rocisa::GlobalStoreB32>(
      byteOffset.container(), result.container(), pointers.sub(2, 2));
}

void buildAgprRoundTrip(Kernel &kernel) {
  kernel.signature().addPointer("in");
  kernel.signature().addPointer("out");
  RegisterRange pointers = loadRoundTripPointers(kernel);
  RegisterRange globalId = computeGlobalId(kernel);
  emitAgprRoundTrip(kernel, pointers, globalId);
  kernel.builder().createEndProgram();
}

// Uploads in, launches the kernel and downloads the result into out.
void runAgprRoundTrip(const std::vector<uint8_t> &hsaco,
                      const std::vector<int32_t> &in,
                      std::vector<int32_t> &out) {
  hipModule_t rawModule = nullptr;
  ASSERT_HIP_SUCCESS(hipModuleLoadData(&rawModule, hsaco.data()));
  ModulePtr module(rawModule);
  hipFunction_t function = nullptr;
  ASSERT_HIP_SUCCESS(hipModuleGetFunction(&function, rawModule, roundTripName));

  size_t bytes = in.size() * sizeof(int32_t);
  void *rawIn = nullptr;
  ASSERT_HIP_SUCCESS(hipMalloc(&rawIn, bytes));
  std::unique_ptr<void, DeviceFree> deviceIn(rawIn);
  void *rawOut = nullptr;
  ASSERT_HIP_SUCCESS(hipMalloc(&rawOut, bytes));
  std::unique_ptr<void, DeviceFree> deviceOut(rawOut);
  ASSERT_HIP_SUCCESS(hipMemcpy(rawIn, in.data(), bytes, hipMemcpyHostToDevice));
  // The sentinel shows any element the kernel failed to write.
  ASSERT_HIP_SUCCESS(hipMemset(rawOut, 0xff, bytes));

  void *args[] = {&rawIn, &rawOut};
  ASSERT_HIP_SUCCESS(hipModuleLaunchKernel(function, roundTripGroups, 1, 1,
                                           roundTripGroupSize, 1, 1, 0, nullptr,
                                           args, nullptr));
  ASSERT_HIP_SUCCESS(hipDeviceSynchronize());
  ASSERT_HIP_SUCCESS(
      hipMemcpy(out.data(), rawOut, bytes, hipMemcpyDeviceToHost));
}

TEST(KernelLaunch, AgprRoundTripOnDevice) {
  Target target = Target::get("gfx950");
  Kernel kernel(target, roundTripName, roundTripGroupSize);
  buildAgprRoundTrip(kernel);
  std::string assembly = kernel.toAssembly();
  SCOPED_TRACE("assembly:\n" + assembly);
  expectLinesInOrder(
      assembly,
      {"v_accvgpr_write_b32 acc3, v3", "v_accvgpr_read_b32 v4, acc3",
       "global_store_dword v2, v4, s[6:7]", ".amdhsa_accum_offset 8",
       ".amdhsa_next_free_vgpr 12", ".vgpr_count: 12", ".agpr_count: 4"});

  // Assembling is checked before the device so it runs without a GPU.
  std::vector<uint8_t> hsaco =
      assembleToHSACO(assembly, kernel.target().isaName());
  ASSERT_FALSE(hsaco.empty());
  if (!hasHipDevice())
    GTEST_SKIP() << "no HIP device";

  std::vector<int32_t> in(roundTripElements);
  for (int i = 0; i < roundTripElements; ++i)
    in[i] = i * 7919 + 1;
  std::vector<int32_t> out(roundTripElements, -1);
  runAgprRoundTrip(hsaco, in, out);
  if (testing::Test::HasFatalFailure())
    return;

  for (int i = 0; i < roundTripElements; ++i)
    ASSERT_EQ(out[i], in[i]) << "element " << i;
}

} // namespace

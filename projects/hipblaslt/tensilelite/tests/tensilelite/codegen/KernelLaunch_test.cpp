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

constexpr const char *idsName = "codegen_ids_hidden";
constexpr int idsGrid[3] = {3, 4, 5};
constexpr int idsBlock[3] = {8, 2, 2};
constexpr int idsGroups = idsGrid[0] * idsGrid[1] * idsGrid[2];
constexpr int idsBlockSize = idsBlock[0] * idsBlock[1] * idsBlock[2];
// Workgroup ids x, y, z, group sizes x, y, z and block counts x, y, z.
constexpr int idsDwords = 9;

// Mimics the reference kernel: the hidden arguments start at offset 12, which
// is not 8-aligned, and follow the explicit ones.
void addIdsSignature(KernelSignature &signature) {
  signature.addPointer("out");
  signature.addScalar("tag", 4);
  for (KernelArgumentKind kind : {KernelArgumentKind::hiddenGroupSizeX,
                                  KernelArgumentKind::hiddenGroupSizeY,
                                  KernelArgumentKind::hiddenGroupSizeZ,
                                  KernelArgumentKind::hiddenBlockCountX,
                                  KernelArgumentKind::hiddenBlockCountY,
                                  KernelArgumentKind::hiddenBlockCountZ})
    signature.addHidden(kind, 4);
}

// Copies the ids to words 0..2 of values and loads the hidden arguments into
// words 3..8, which are the group sizes followed by the block counts.
void loadIdsAndHiddenArguments(Kernel &kernel, const RegisterRange &values) {
  IRBuilder &builder = kernel.builder();
  const RegisterRange *ids[] = {&kernel.workgroupIdX(), &kernel.workgroupIdY(),
                                &kernel.workgroupIdZ()};
  for (int i = 0; i < 3; ++i)
    builder.create<rocisa::SMovB32>(values.sub(i), ids[i]->container());
  for (int i = 0; i < 6; ++i)
    kernel.loadArgument(values.ref(3 + i), 12 + 4 * i);
  builder.createWaitCnt({.scalarMemory = 0});
  // The runtime writes only the low 2 bytes of a group size (LLVM's width), so
  // the upper half of a 4-byte slot is undefined.
  for (int i = 3; i < 6; ++i)
    builder.create<rocisa::SAndB32>(values.sub(i), values.sub(i), 0xffff);
}

// Returns the byte offset of this group's record, computed from the loaded
// block counts so that a wrong count corrupts the layout.
RegisterRange computeRecordOffset(Kernel &kernel, const RegisterRange &values) {
  IRBuilder &builder = kernel.builder();
  RegisterRange linear = builder.allocateSgpr(1);
  // linear = x + countX * (y + countY * z).
  builder.create<rocisa::SMulI32>(linear.container(), values.sub(7),
                                  values.sub(2));
  builder.create<rocisa::SAddU32>(linear.container(), linear.container(),
                                  values.sub(1));
  builder.create<rocisa::SMulI32>(linear.container(), linear.container(),
                                  values.sub(6));
  builder.create<rocisa::SAddU32>(linear.container(), linear.container(),
                                  values.sub(0));
  builder.create<rocisa::SMulI32>(linear.container(), linear.container(),
                                  4 * idsDwords);
  RegisterRange offset = builder.allocateVgpr(1);
  builder.create<rocisa::VMovB32>(offset.container(), linear.container());
  return offset;
}

void buildIdsAndHiddenKernel(Kernel &kernel) {
  kernel.enableWorkgroupIds(true, true);
  addIdsSignature(kernel.signature());
  IRBuilder &builder = kernel.builder();
  RegisterRange out = builder.allocateSgpr(2, 2);
  RegisterRange values = builder.allocateSgpr(idsDwords);
  kernel.loadArgument(out.ref(), 0);
  loadIdsAndHiddenArguments(kernel, values);
  RegisterRange base = computeRecordOffset(kernel, values);
  // One lane stores each record; the kernel ends right after, so shrinking
  // exec is safe.
  builder.create<rocisa::SMovB64>(std::make_shared<rocisa::EXEC>(), 1);
  RegisterRange address = builder.allocateVgpr(1);
  RegisterRange data = builder.allocateVgpr(1);
  for (int i = 0; i < idsDwords; ++i) {
    builder.create<rocisa::VAddU32>(address.container(), 4 * i,
                                    base.container());
    builder.create<rocisa::VMovB32>(data.container(), values.sub(i));
    builder.create<rocisa::GlobalStoreB32>(address.container(),
                                           data.container(), out.container());
  }
  builder.createEndProgram();
}

// Runs the kernel over idsGrid with idsBlock and downloads the records.
void runIdsAndHidden(const std::vector<uint8_t> &hsaco,
                     std::vector<int32_t> &records) {
  hipModule_t rawModule = nullptr;
  ASSERT_HIP_SUCCESS(hipModuleLoadData(&rawModule, hsaco.data()));
  ModulePtr module(rawModule);
  hipFunction_t function = nullptr;
  ASSERT_HIP_SUCCESS(hipModuleGetFunction(&function, rawModule, idsName));

  size_t bytes = records.size() * sizeof(int32_t);
  void *out = nullptr;
  ASSERT_HIP_SUCCESS(hipMalloc(&out, bytes));
  std::unique_ptr<void, DeviceFree> deviceOut(out);
  // The sentinel shows any record the kernel failed to write.
  ASSERT_HIP_SUCCESS(hipMemset(out, 0xff, bytes));

  int32_t tag = 0;
  void *args[] = {&out, &tag};
  ASSERT_HIP_SUCCESS(hipModuleLaunchKernel(
      function, idsGrid[0], idsGrid[1], idsGrid[2], idsBlock[0], idsBlock[1],
      idsBlock[2], 0, nullptr, args, nullptr));
  ASSERT_HIP_SUCCESS(hipDeviceSynchronize());
  ASSERT_HIP_SUCCESS(
      hipMemcpy(records.data(), out, bytes, hipMemcpyDeviceToHost));
}

TEST(KernelLaunch, WorkgroupIdsAndHiddenArgumentsOnDevice) {
  Target target = Target::get("gfx950");
  Kernel kernel(target, idsName, idsBlockSize);
  buildIdsAndHiddenKernel(kernel);
  std::string assembly = kernel.toAssembly();
  SCOPED_TRACE("assembly:\n" + assembly);
  expectLinesInOrder(assembly, {".amdhsa_system_sgpr_workgroup_id_x 1",
                                ".amdhsa_system_sgpr_workgroup_id_y 1",
                                ".amdhsa_system_sgpr_workgroup_id_z 1"});

  // Assembling is checked before the device so it runs without a GPU.
  std::vector<uint8_t> hsaco =
      assembleToHSACO(assembly, kernel.target().isaName());
  ASSERT_FALSE(hsaco.empty());
  if (!hasHipDevice())
    GTEST_SKIP() << "no HIP device";

  std::vector<int32_t> records(idsGroups * idsDwords, -1);
  runIdsAndHidden(hsaco, records);
  if (testing::Test::HasFatalFailure())
    return;

  for (int z = 0; z < idsGrid[2]; ++z)
    for (int y = 0; y < idsGrid[1]; ++y)
      for (int x = 0; x < idsGrid[0]; ++x) {
        int linear = x + idsGrid[0] * (y + idsGrid[1] * z);
        const int32_t expected[idsDwords] = {
            x,           y,          z,          idsBlock[0], idsBlock[1],
            idsBlock[2], idsGrid[0], idsGrid[1], idsGrid[2]};
        for (int i = 0; i < idsDwords; ++i)
          ASSERT_EQ(records[linear * idsDwords + i], expected[i])
              << "group (" << x << ", " << y << ", " << z << ") word " << i;
      }
}

constexpr const char *ldsName = "codegen_lds_round_trip";
constexpr int ldsGroupSize = 256;
constexpr int ldsGroups = 4;
constexpr int ldsElements = ldsGroupSize * ldsGroups;

// Returns the LDS byte address of lane (tid + delta) % ldsGroupSize.
RegisterRange computeLdsAddress(Kernel &kernel, int delta) {
  IRBuilder &builder = kernel.builder();
  RegisterRange address = builder.allocateVgpr(1);
  builder.create<rocisa::VAddU32>(address.container(), delta,
                                  kernel.workitemId().container());
  builder.create<rocisa::VAndB32>(address.container(), ldsGroupSize - 1,
                                  address.container());
  builder.create<rocisa::VLShiftLeftB32>(address.container(), 2,
                                         address.container());
  return address;
}

// Each lane publishes its element in LDS and reads its neighbor's, which sits
// in another wave at wave boundaries, so the barrier is what makes it correct.
void emitLdsExchange(Kernel &kernel, const RegisterRange &pointers,
                     const RegisterRange &globalId) {
  IRBuilder &builder = kernel.builder();
  RegisterRange byteOffset = builder.allocateVgpr(1);
  builder.create<rocisa::VLShiftLeftB32>(byteOffset.container(), 2,
                                         globalId.container());
  RegisterRange value = builder.allocateVgpr(1);
  builder.create<rocisa::GlobalLoadB32>(
      value.container(), byteOffset.container(), pointers.sub(0, 2));
  builder.createWaitCnt({.vectorLoads = 0});
  RegisterRange ownAddress = computeLdsAddress(kernel, 0);
  builder.create<rocisa::DSStoreB32>(ownAddress.container(), value.container());
  builder.createWaitCnt({.lds = 0});
  builder.create<rocisa::SBarrier>();
  RegisterRange neighborAddress = computeLdsAddress(kernel, 1);
  RegisterRange neighbor = builder.allocateVgpr(1);
  builder.create<rocisa::DSLoadB32>(neighbor.container(),
                                    neighborAddress.container());
  builder.createWaitCnt({.lds = 0});
  builder.create<rocisa::GlobalStoreB32>(
      byteOffset.container(), neighbor.container(), pointers.sub(2, 2));
}

void buildLdsRoundTrip(Kernel &kernel) {
  kernel.signature().addPointer("in");
  kernel.signature().addPointer("out");
  kernel.setLdsSize(4 * ldsGroupSize);
  IRBuilder &builder = kernel.builder();
  RegisterRange pointers = builder.allocateSgpr(4, 4);
  kernel.loadArgument(pointers.ref(), 0);
  builder.createWaitCnt({.scalarMemory = 0});
  RegisterRange globalId = computeGlobalId(kernel);
  emitLdsExchange(kernel, pointers, globalId);
  builder.createEndProgram();
}

// Uploads in, launches the kernel and downloads the result into out.
void runLdsRoundTrip(const std::vector<uint8_t> &hsaco,
                     const std::vector<int32_t> &in,
                     std::vector<int32_t> &out) {
  hipModule_t rawModule = nullptr;
  ASSERT_HIP_SUCCESS(hipModuleLoadData(&rawModule, hsaco.data()));
  ModulePtr module(rawModule);
  hipFunction_t function = nullptr;
  ASSERT_HIP_SUCCESS(hipModuleGetFunction(&function, rawModule, ldsName));

  size_t bytes = in.size() * sizeof(int32_t);
  void *rawIn = nullptr;
  ASSERT_HIP_SUCCESS(hipMalloc(&rawIn, bytes));
  std::unique_ptr<void, DeviceFree> deviceIn(rawIn);
  void *rawOut = nullptr;
  ASSERT_HIP_SUCCESS(hipMalloc(&rawOut, bytes));
  std::unique_ptr<void, DeviceFree> deviceOut(rawOut);
  ASSERT_HIP_SUCCESS(hipMemcpy(rawIn, in.data(), bytes, hipMemcpyHostToDevice));
  ASSERT_HIP_SUCCESS(hipMemset(rawOut, 0xff, bytes));

  void *args[] = {&rawIn, &rawOut};
  ASSERT_HIP_SUCCESS(hipModuleLaunchKernel(function, ldsGroups, 1, 1,
                                           ldsGroupSize, 1, 1, 0, nullptr, args,
                                           nullptr));
  ASSERT_HIP_SUCCESS(hipDeviceSynchronize());
  ASSERT_HIP_SUCCESS(
      hipMemcpy(out.data(), rawOut, bytes, hipMemcpyDeviceToHost));
}

TEST(KernelLaunch, LdsRoundTripOnDevice) {
  Target target = Target::get("gfx950");
  Kernel kernel(target, ldsName, ldsGroupSize);
  buildLdsRoundTrip(kernel);
  std::string assembly = kernel.toAssembly();
  SCOPED_TRACE("assembly:\n" + assembly);
  expectLinesInOrder(assembly,
                     {"ds_write_b32", "s_waitcnt lgkmcnt(0)", "s_barrier",
                      "ds_read_b32", ".amdhsa_group_segment_fixed_size 1024"});

  // Assembling is checked before the device so it runs without a GPU.
  std::vector<uint8_t> hsaco =
      assembleToHSACO(assembly, kernel.target().isaName());
  ASSERT_FALSE(hsaco.empty());
  if (!hasHipDevice())
    GTEST_SKIP() << "no HIP device";

  std::vector<int32_t> in(ldsElements);
  for (int i = 0; i < ldsElements; ++i)
    in[i] = i * 7919 + 1;
  std::vector<int32_t> out(ldsElements, -1);
  runLdsRoundTrip(hsaco, in, out);
  if (testing::Test::HasFatalFailure())
    return;

  for (int g = 0; g < ldsGroups; ++g)
    for (int t = 0; t < ldsGroupSize; ++t)
      ASSERT_EQ(out[g * ldsGroupSize + t],
                in[g * ldsGroupSize + (t + 1) % ldsGroupSize])
          << "group " << g << " lane " << t;
}

} // namespace

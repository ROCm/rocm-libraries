// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tensilelite/Compiler.hpp>
#include <tensilelite/codegen/Kernel.hpp>

#include "ExpectLines.hpp"
#include "HipTestSupport.hpp"

#include "container.hpp"
#include "instruction/branch.hpp"
#include "instruction/cmp.hpp"
#include "instruction/common.hpp"
#include "instruction/mem.hpp"

#include <gtest/gtest.h>

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace rocm::tensilelite;
using namespace rocm::tensilelite::codegen;
using namespace rocm::tensilelite::codegen::test;

constexpr int workgroupSize = 256;
constexpr int elementCount = 4099;
constexpr int padding = 64;
constexpr float scale = 2.5f;
constexpr float sentinel = -1234.5f;
constexpr const char *kernelName = "codegen_saxpy";

// Byte offsets of the arguments in the kernarg segment.
struct SaxpyOffsets {
  int x;
  int y;
  int a;
  int n;
};

SaxpyOffsets declareArguments(KernelSignature &signature) {
  int x = signature.addPointer("x");
  int y = signature.addPointer("y");
  int a = signature.addScalar("a", 4);
  int n = signature.addScalar("n", 4);
  return {x, y, a, n};
}

// Kernel arguments loaded into SGPRs.
struct SaxpyArguments {
  RegisterRange pointers; // x in the first pair, y in the second.
  RegisterRange scalars;  // a, then n.
};

SaxpyArguments loadArguments(Kernel &kernel, const SaxpyOffsets &offsets) {
  // One wide load per group relies on the arguments being adjacent.
  if (offsets.y != offsets.x + 8 || offsets.n != offsets.a + 4)
    throw std::logic_error("saxpy arguments are not contiguous");

  IRBuilder &builder = kernel.builder();
  RegisterRange pointers = builder.allocateSgpr(4, 4);
  RegisterRange scalars = builder.allocateSgpr(2, 2);
  builder.create<rocisa::SLoadB128>(
      pointers.container(), kernel.kernargSegmentPtr().container(), offsets.x);
  builder.create<rocisa::SLoadB64>(
      scalars.container(), kernel.kernargSegmentPtr().container(), offsets.a);
  builder.createWaitCnt({.scalarMemory = 0});
  return SaxpyArguments{std::move(pointers), std::move(scalars)};
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

void emitElementUpdate(Kernel &kernel, const SaxpyArguments &args,
                       const RegisterRange &globalId) {
  IRBuilder &builder = kernel.builder();
  RegisterRange byteOffset = builder.allocateVgpr(1);
  RegisterRange vx = builder.allocateVgpr(1);
  RegisterRange vy = builder.allocateVgpr(1);
  builder.create<rocisa::VLShiftLeftB32>(byteOffset.container(), 2,
                                         globalId.container());
  builder.create<rocisa::GlobalLoadB32>(vx.container(), byteOffset.container(),
                                        args.pointers.sub(0, 2));
  builder.create<rocisa::GlobalLoadB32>(vy.container(), byteOffset.container(),
                                        args.pointers.sub(2, 2));
  builder.createWaitCnt({.vectorLoads = 0});
  // vy = a * vx + vy.
  builder.create<rocisa::VMacF32>(vy.container(), args.scalars.sub(0),
                                  vx.container());
  builder.create<rocisa::GlobalStoreB32>(byteOffset.container(), vy.container(),
                                         args.pointers.sub(2, 2));
}

void buildSaxpy(Kernel &kernel, const SaxpyOffsets &offsets) {
  IRBuilder &builder = kernel.builder();
  SaxpyArguments args = loadArguments(kernel, offsets);
  RegisterRange globalId = computeGlobalId(kernel);

  // The tail workgroup masks off the lanes past n.
  auto vcc = std::make_shared<rocisa::VCC>();
  builder.create<rocisa::VCmpGtU32>(vcc, args.scalars.sub(1),
                                    globalId.container());
  RegisterRange savedExec = builder.allocateSgpr(2, 2);
  builder.create<rocisa::SAndSaveExecB64>(savedExec.container(), vcc);
  std::shared_ptr<rocisa::Label> end = builder.createLabel("end");
  builder.create<rocisa::SCBranchExecZ>(end->getLabelName());
  emitElementUpdate(kernel, args, globalId);
  builder.insert(end);
  // Exec is not restored because the program ends right after.
  builder.createEndProgram();
}

std::vector<float> makeX(int n) {
  std::vector<float> x(n);
  for (int i = 0; i < n; ++i)
    x[i] = 0.25f * static_cast<float>(i % 97) - 3.0f +
           1.0f / static_cast<float>(i + 1);
  return x;
}

// Elements past n are sentinels that the kernel must leave untouched.
std::vector<float> makeY(int n, int paddingCount) {
  std::vector<float> y(n + paddingCount, sentinel);
  for (int i = 0; i < n; ++i)
    y[i] = 1.5f - 0.125f * static_cast<float>(i % 31) +
           1.0f / static_cast<float>(i + 3);
  return y;
}

std::vector<float> computeExpected(const std::vector<float> &x,
                                   const std::vector<float> &y) {
  std::vector<float> expected = y;
  for (int i = 0; i < std::ssize(x); ++i)
    expected[i] = std::fma(scale, x[i], y[i]);
  return expected;
}

std::vector<std::byte> makeKernarg(int segmentSize, const SaxpyOffsets &offsets,
                                   const void *x, const void *y) {
  std::vector<std::byte> kernarg(segmentSize);
  uint32_t n = static_cast<uint32_t>(elementCount);
  std::memcpy(kernarg.data() + offsets.x, &x, sizeof(x));
  std::memcpy(kernarg.data() + offsets.y, &y, sizeof(y));
  std::memcpy(kernarg.data() + offsets.a, &scale, sizeof(scale));
  std::memcpy(kernarg.data() + offsets.n, &n, sizeof(n));
  return kernarg;
}

void allocateDevice(int floatCount, std::unique_ptr<void, DeviceFree> &out) {
  void *ptr = nullptr;
  ASSERT_HIP_SUCCESS(hipMalloc(&ptr, floatCount * sizeof(float)));
  out.reset(ptr);
}

// Uploads the inputs, launches the kernel and downloads y.
void runSaxpy(const std::vector<uint8_t> &hsaco, const SaxpyOffsets &offsets,
              int segmentSize, const std::vector<float> &x,
              std::vector<float> &y) {
  hipModule_t rawModule = nullptr;
  ASSERT_HIP_SUCCESS(hipModuleLoadData(&rawModule, hsaco.data()));
  ModulePtr module(rawModule);
  hipFunction_t function = nullptr;
  ASSERT_HIP_SUCCESS(hipModuleGetFunction(&function, rawModule, kernelName));

  std::unique_ptr<void, DeviceFree> deviceX;
  std::unique_ptr<void, DeviceFree> deviceY;
  allocateDevice(static_cast<int>(std::ssize(x)), deviceX);
  ASSERT_FALSE(testing::Test::HasFatalFailure());
  allocateDevice(static_cast<int>(std::ssize(y)), deviceY);
  ASSERT_FALSE(testing::Test::HasFatalFailure());
  ASSERT_HIP_SUCCESS(hipMemcpy(deviceX.get(), x.data(),
                               x.size() * sizeof(float),
                               hipMemcpyHostToDevice));
  ASSERT_HIP_SUCCESS(hipMemcpy(deviceY.get(), y.data(),
                               y.size() * sizeof(float),
                               hipMemcpyHostToDevice));

  std::vector<std::byte> kernarg =
      makeKernarg(segmentSize, offsets, deviceX.get(), deviceY.get());
  size_t kernargSize = kernarg.size();
  void *extra[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, kernarg.data(),
                   HIP_LAUNCH_PARAM_BUFFER_SIZE, &kernargSize,
                   HIP_LAUNCH_PARAM_END};
  int groups = (elementCount + workgroupSize - 1) / workgroupSize;
  ASSERT_HIP_SUCCESS(hipModuleLaunchKernel(
      function, groups, 1, 1, workgroupSize, 1, 1, 0, nullptr, nullptr, extra));
  ASSERT_HIP_SUCCESS(hipDeviceSynchronize());
  ASSERT_HIP_SUCCESS(hipMemcpy(y.data(), deviceY.get(),
                               y.size() * sizeof(float),
                               hipMemcpyDeviceToHost));
}

TEST(Saxpy, MatchesHostReferenceOnDevice) {
  Target target = Target::get("gfx950");
  Kernel kernel(target, kernelName, workgroupSize);
  SaxpyOffsets offsets = declareArguments(kernel.signature());
  ASSERT_EQ(offsets.x, 0);
  ASSERT_EQ(offsets.y, 8);
  ASSERT_EQ(offsets.a, 16);
  ASSERT_EQ(offsets.n, 20);
  ASSERT_EQ(kernel.signature().segmentSize(), 24);

  buildSaxpy(kernel, offsets);
  std::string assembly = kernel.toAssembly();
  SCOPED_TRACE("assembly:\n" + assembly);
  expectLinesInOrder(
      assembly,
      {"s_load_dwordx4", "s_load_dwordx2", "s_waitcnt lgkmcnt(0)", "v_and_b32",
       "s_mul_i32", "v_add_u32", "v_cmp_gt_u32 vcc", "s_and_saveexec_b64",
       "s_cbranch_execz label_end_", "v_lshlrev_b32", "global_load_dword",
       "global_load_dword", "s_waitcnt vmcnt(0)", "v_fmac_f32",
       "global_store_dword", "label_end_", "s_endpgm"});

  // Assembling is checked before the device so it runs without a GPU.
  std::vector<uint8_t> hsaco =
      assembleToHSACO(assembly, kernel.target().isaName());
  ASSERT_FALSE(hsaco.empty());
  if (!hasHipDevice())
    GTEST_SKIP() << "no HIP device";

  std::vector<float> x = makeX(elementCount);
  std::vector<float> y = makeY(elementCount, padding);
  std::vector<float> expected = computeExpected(x, y);
  runSaxpy(hsaco, offsets, kernel.signature().segmentSize(), x, y);
  if (testing::Test::HasFatalFailure())
    return;

  ASSERT_EQ(y.size(), expected.size());
  for (int i = 0; i < std::ssize(y); ++i)
    ASSERT_EQ(y[i], expected[i]) << "element " << i;
}

} // namespace

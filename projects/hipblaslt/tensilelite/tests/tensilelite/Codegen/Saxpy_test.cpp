// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tensilelite/API/Compiler.hpp>
#include <tensilelite/Codegen/Kernel.hpp>

#include "TestSupport.hpp"

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
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace rocm::tensilelite;
using namespace rocm::tensilelite::codegen;
using namespace rocm::tensilelite::codegen::test;

static constexpr int32_t workgroupSize = 256;
static constexpr int32_t elementCount = 4099;
static constexpr int32_t padding = 64;
static constexpr float scale = 2.5f;
static constexpr float sentinel = -1234.5f;
static constexpr const char *kernelName = "codegen_saxpy";

namespace {

// Byte offsets of the arguments in the kernarg segment.
struct SaxpyOffsets {
  int32_t x;
  int32_t y;
  int32_t a;
  int32_t n;
};

} // namespace

static SaxpyOffsets declareArguments(KernelSignature &signature) {
  int32_t x = signature.addPointer("x");
  int32_t y = signature.addPointer("y");
  int32_t a = signature.addScalar("a", 4);
  int32_t n = signature.addScalar("n", 4);
  return {x, y, a, n};
}

namespace {

// Kernel arguments loaded into SGPRs.
struct SaxpyArguments {
  RegisterRange pointers; // x in the first pair, y in the second.
  RegisterRange scalars;  // a, then n.
};

} // namespace

static SaxpyArguments loadArguments(Kernel &kernel, SaxpyOffsets offsets) {
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

static void emitElementUpdate(Kernel &kernel, const SaxpyArguments &args,
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

static void buildSaxpy(Kernel &kernel, SaxpyOffsets offsets) {
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

static std::vector<float> makeX(int32_t n) {
  std::vector<float> x(n);
  for (int32_t i = 0; i < n; ++i)
    x[i] = 0.25f * static_cast<float>(i % 97) - 3.0f +
           1.0f / static_cast<float>(i + 1);
  return x;
}

// Elements past n are sentinels that the kernel must leave untouched.
static std::vector<float> makeY(int32_t n, int32_t paddingCount) {
  std::vector<float> y(n + paddingCount, sentinel);
  for (int32_t i = 0; i < n; ++i)
    y[i] = 1.5f - 0.125f * static_cast<float>(i % 31) +
           1.0f / static_cast<float>(i + 3);
  return y;
}

static std::vector<float> computeExpected(const std::vector<float> &x,
                                          const std::vector<float> &y) {
  std::vector<float> expected = y;
  for (int32_t i = 0; i < std::ssize(x); ++i)
    expected[i] = std::fma(scale, x[i], y[i]);
  return expected;
}

static std::vector<std::byte> makeKernarg(int32_t segmentSize,
                                          SaxpyOffsets offsets, const void *x,
                                          const void *y) {
  std::vector<std::byte> kernarg(segmentSize);
  writeArgument(kernarg, offsets.x, x);
  writeArgument(kernarg, offsets.y, y);
  writeArgument(kernarg, offsets.a, scale);
  writeArgument(kernarg, offsets.n, static_cast<uint32_t>(elementCount));
  return kernarg;
}

// Uploads the inputs, launches the kernel and downloads y.
static void runSaxpy(const std::vector<uint8_t> &hsaco, SaxpyOffsets offsets,
                     int32_t segmentSize, const std::vector<float> &x,
                     std::vector<float> &y) {
  ModulePtr module;
  hipFunction_t function = nullptr;
  loadKernel(hsaco, kernelName, module, function);
  DevicePtr deviceX;
  DevicePtr deviceY;
  upload(x, deviceX);
  upload(y, deviceY);
  ASSERT_FALSE(testing::Test::HasFatalFailure());

  std::vector<std::byte> kernarg =
      makeKernarg(segmentSize, offsets, deviceX.get(), deviceY.get());
  int32_t groups = (elementCount + workgroupSize - 1) / workgroupSize;
  launchWithKernarg(function, groups, workgroupSize, kernarg);
  ASSERT_FALSE(testing::Test::HasFatalFailure());
  download(deviceY, y);
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
  for (int32_t i = 0; i < std::ssize(y); ++i)
    ASSERT_EQ(y[i], expected[i]) << "element " << i;
}

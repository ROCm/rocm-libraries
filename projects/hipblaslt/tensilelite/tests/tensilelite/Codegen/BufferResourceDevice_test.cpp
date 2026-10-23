// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tensilelite/API/Compiler.hpp>
#include <tensilelite/Codegen/Kernel.hpp>

#include "BuilderTestSupport.hpp"
#include "ExpectLines.hpp"
#include "HipTestSupport.hpp"

#include "container.hpp"
#include "instruction/common.hpp"
#include "instruction/mem.hpp"

#include <gtest/gtest.h>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace rocm::tensilelite;
using namespace rocm::tensilelite::codegen;
using namespace rocm::tensilelite::codegen::test;

constexpr int workgroupSize = 64;
constexpr int groupCount = 2;
constexpr int laneCount = workgroupSize * groupCount;
// The input is shorter than the output, and both are shorter than the lane
// count, so lanes run past each descriptor's end.
constexpr int inWords = 77;
constexpr int outWords = 101;
constexpr int padding = 64;
constexpr uint32_t sentinel = 0xDEADBEEFu;
constexpr const char *kernelName = "codegen_buffer_copy";

enum class SizeSource { sgpr, immediate };

// Byte offsets of the arguments in the kernarg segment.
struct CopyOffsets {
  int in;
  int out;
  int inBytes;
  int outBytes;
};

CopyOffsets declareArguments(KernelSignature &signature) {
  int in = signature.addPointer("in");
  int out = signature.addPointer("out");
  int inBytes = signature.addScalar("inBytes", 4);
  int outBytes = signature.addScalar("outBytes", 4);
  return {in, out, inBytes, outBytes};
}

// The pointers and sizes sit in SGPRs apart from the descriptors.
void buildDescriptorsFromSgprs(Kernel &kernel, const CopyOffsets &offsets,
                               const RegisterRange &inRsrc,
                               const RegisterRange &outRsrc) {
  IRBuilder &builder = kernel.builder();
  const RegisterRange &kernarg = kernel.kernargSegmentPtr();
  RegisterRange inPointer = builder.allocateSgpr(2, 2);
  RegisterRange outPointer = builder.allocateSgpr(2, 2);
  RegisterRange sizes = builder.allocateSgpr(2);
  builder.create<rocisa::SLoadB64>(inPointer.container(), kernarg.container(),
                                   offsets.in);
  builder.create<rocisa::SLoadB64>(outPointer.container(), kernarg.container(),
                                   offsets.out);
  builder.create<rocisa::SLoadB32>(sizes.sub(0), kernarg.container(),
                                   offsets.inBytes);
  builder.create<rocisa::SLoadB32>(sizes.sub(1), kernarg.container(),
                                   offsets.outBytes);
  builder.createWaitCnt({.scalarMemory = 0});
  builder.makeRsrc(inRsrc.ref(), inPointer, sizes.ref(0));
  builder.makeRsrc(outRsrc.ref(), outPointer, sizes.ref(1));
}

// The pointers are loaded straight into the descriptors, so makeRsrc skips
// copying them and the sizes are immediates.
void buildDescriptorsInPlace(Kernel &kernel, const CopyOffsets &offsets,
                             const RegisterRange &inRsrc,
                             const RegisterRange &outRsrc) {
  IRBuilder &builder = kernel.builder();
  const RegisterRange &kernarg = kernel.kernargSegmentPtr();
  builder.create<rocisa::SLoadB64>(inRsrc.sub(0, 2), kernarg.container(),
                                   offsets.in);
  builder.create<rocisa::SLoadB64>(outRsrc.sub(0, 2), kernarg.container(),
                                   offsets.out);
  builder.createWaitCnt({.scalarMemory = 0});
  builder.makeRsrc(inRsrc.ref(), inRsrc.ref(0, 2), inWords * 4);
  builder.makeRsrc(outRsrc.ref(), outRsrc.ref(0, 2), outWords * 4);
}

RegisterRange computeByteOffset(Kernel &kernel) {
  IRBuilder &builder = kernel.builder();
  RegisterRange localId = builder.allocateVgpr(1);
  RegisterRange groupBase = builder.allocateSgpr(1);
  RegisterRange globalId = builder.allocateVgpr(1);
  RegisterRange offset = builder.allocateVgpr(1);
  // The workitem id register packs the x, y and z ids.
  builder.create<rocisa::VAndB32>(localId.container(), 0x3ff,
                                  kernel.workitemId().container());
  builder.muli32(groupBase.ref(), kernel.workgroupIdX(), workgroupSize);
  builder.addi32(globalId.ref(), groupBase, localId);
  builder.create<rocisa::VLShiftLeftB32>(offset.container(), 2,
                                         globalId.container());
  return offset;
}

// Every lane accesses memory, so the descriptor bounds do all the clipping.
void buildCopy(Kernel &kernel, SizeSource source, const CopyOffsets &offsets) {
  IRBuilder &builder = kernel.builder();
  RegisterRange inRsrc = builder.allocateSgpr(4, 4);
  RegisterRange outRsrc = builder.allocateSgpr(4, 4);
  if (source == SizeSource::sgpr)
    buildDescriptorsFromSgprs(kernel, offsets, inRsrc, outRsrc);
  else
    buildDescriptorsInPlace(kernel, offsets, inRsrc, outRsrc);

  RegisterRange offset = computeByteOffset(kernel);
  RegisterRange value = builder.allocateVgpr(1);
  builder.create<rocisa::BufferLoadB32>(value.container(), offset.container(),
                                        inRsrc.container(), 0,
                                        rocisa::MUBUFModifiers(true, 0));
  builder.createWaitCnt({.vectorLoads = 0});
  builder.create<rocisa::BufferStoreB32>(value.container(), offset.container(),
                                         outRsrc.container(), 0,
                                         rocisa::MUBUFModifiers(true, 0));
  builder.createEndProgram();
}

// Every word is nonzero and differs from the sentinel, so a load that is not
// clipped to zero shows up in the output.
std::vector<uint32_t> makeInput() {
  std::vector<uint32_t> in(laneCount + padding);
  for (int i = 0; i < std::ssize(in); ++i)
    in[i] = (0x01000000u + static_cast<uint32_t>(i) * 2654435761u) | 1u;
  return in;
}

std::vector<uint32_t> computeExpected(const std::vector<uint32_t> &in) {
  std::vector<uint32_t> expected(laneCount + padding, sentinel);
  for (int i = 0; i < outWords; ++i)
    expected[i] = i < inWords ? in[i] : 0u;
  return expected;
}

std::vector<std::byte> makeKernarg(int segmentSize, const CopyOffsets &offsets,
                                   const void *in, const void *out) {
  std::vector<std::byte> kernarg(segmentSize);
  uint32_t inBytes = inWords * sizeof(uint32_t);
  uint32_t outBytes = outWords * sizeof(uint32_t);
  std::memcpy(kernarg.data() + offsets.in, &in, sizeof(in));
  std::memcpy(kernarg.data() + offsets.out, &out, sizeof(out));
  std::memcpy(kernarg.data() + offsets.inBytes, &inBytes, sizeof(inBytes));
  std::memcpy(kernarg.data() + offsets.outBytes, &outBytes, sizeof(outBytes));
  return kernarg;
}

void allocateDevice(int wordCount, std::unique_ptr<void, DeviceFree> &out) {
  void *ptr = nullptr;
  ASSERT_HIP_SUCCESS(hipMalloc(&ptr, wordCount * sizeof(uint32_t)));
  out.reset(ptr);
}

// Uploads the buffers, launches the kernel and downloads out.
void runCopy(const std::vector<uint8_t> &hsaco, const CopyOffsets &offsets,
             int segmentSize, const std::vector<uint32_t> &in,
             std::vector<uint32_t> &out) {
  hipModule_t rawModule = nullptr;
  ASSERT_HIP_SUCCESS(hipModuleLoadData(&rawModule, hsaco.data()));
  ModulePtr module(rawModule);
  hipFunction_t function = nullptr;
  ASSERT_HIP_SUCCESS(hipModuleGetFunction(&function, rawModule, kernelName));

  std::unique_ptr<void, DeviceFree> deviceIn;
  std::unique_ptr<void, DeviceFree> deviceOut;
  allocateDevice(static_cast<int>(std::ssize(in)), deviceIn);
  ASSERT_FALSE(testing::Test::HasFatalFailure());
  allocateDevice(static_cast<int>(std::ssize(out)), deviceOut);
  ASSERT_FALSE(testing::Test::HasFatalFailure());
  ASSERT_HIP_SUCCESS(hipMemcpy(deviceIn.get(), in.data(),
                               in.size() * sizeof(uint32_t),
                               hipMemcpyHostToDevice));
  ASSERT_HIP_SUCCESS(hipMemcpy(deviceOut.get(), out.data(),
                               out.size() * sizeof(uint32_t),
                               hipMemcpyHostToDevice));

  std::vector<std::byte> kernarg =
      makeKernarg(segmentSize, offsets, deviceIn.get(), deviceOut.get());
  size_t kernargSize = kernarg.size();
  void *extra[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, kernarg.data(),
                   HIP_LAUNCH_PARAM_BUFFER_SIZE, &kernargSize,
                   HIP_LAUNCH_PARAM_END};
  ASSERT_HIP_SUCCESS(hipModuleLaunchKernel(function, groupCount, 1, 1,
                                           workgroupSize, 1, 1, 0, nullptr,
                                           nullptr, extra));
  ASSERT_HIP_SUCCESS(hipDeviceSynchronize());
  ASSERT_HIP_SUCCESS(hipMemcpy(out.data(), deviceOut.get(),
                               out.size() * sizeof(uint32_t),
                               hipMemcpyDeviceToHost));
}

void checkSource(const std::string &assembly, SizeSource source) {
  expectLinesInOrder(assembly, {"buffer_load_dword", "buffer_store_dword"});
  EXPECT_EQ(countLinesContaining(assembly, " offen"), 2);
  if (source == SizeSource::sgpr)
    return;
  // The pointers are already in place, so no pointer copy is emitted.
  EXPECT_EQ(countLinesContaining(assembly, "s_mov_b64"), 0);
  EXPECT_GT(countLinesContaining(assembly, "308"), 0);
  EXPECT_GT(countLinesContaining(assembly, "404"), 0);
}

void runBufferCopyTest(SizeSource source) {
  Target target = Target::get("gfx950");
  Kernel kernel(target, kernelName, workgroupSize);
  CopyOffsets offsets = declareArguments(kernel.signature());
  buildCopy(kernel, source, offsets);
  std::string assembly = kernel.toAssembly();
  SCOPED_TRACE("assembly:\n" + assembly);
  checkSource(assembly, source);

  // Assembling is checked before the device so it runs without a GPU.
  std::vector<uint8_t> hsaco =
      assembleToHSACO(assembly, kernel.target().isaName());
  ASSERT_FALSE(hsaco.empty());
  if (!hasHipDevice())
    GTEST_SKIP() << "no HIP device";

  std::vector<uint32_t> in = makeInput();
  std::vector<uint32_t> expected = computeExpected(in);
  std::vector<uint32_t> out(laneCount + padding, sentinel);
  runCopy(hsaco, offsets, kernel.signature().segmentSize(), in, out);
  if (testing::Test::HasFatalFailure())
    return;

  ASSERT_EQ(out.size(), expected.size());
  for (int i = 0; i < std::ssize(out); ++i)
    ASSERT_EQ(out[i], expected[i]) << "word " << i;
}

TEST(RsrcDevice, SgprSizes) { runBufferCopyTest(SizeSource::sgpr); }

TEST(RsrcDevice, ImmediateSizes) { runBufferCopyTest(SizeSource::immediate); }

} // namespace

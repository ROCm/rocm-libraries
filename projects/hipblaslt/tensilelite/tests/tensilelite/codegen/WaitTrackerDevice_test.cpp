// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tensilelite/Compiler.hpp>
#include <tensilelite/codegen/Kernel.hpp>
#include <tensilelite/codegen/WaitTracker.hpp>

#include "BuilderTestSupport.hpp"
#include "ExpectLines.hpp"
#include "HipTestSupport.hpp"

#include "container.hpp"
#include "instruction/branch.hpp"
#include "instruction/cmp.hpp"
#include "instruction/common.hpp"
#include "instruction/mem.hpp"

#include <gtest/gtest.h>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

using namespace rocm::tensilelite;
using namespace rocm::tensilelite::codegen;
using namespace rocm::tensilelite::codegen::test;

constexpr int workgroupSize = 64;
constexpr int ringDepth = 4;
constexpr int slotStride = workgroupSize * 4;
constexpr uint32_t sentinel = 0xDEADBEEFu;
constexpr const char *kernelName = "codegen_wait_ring";

struct ArgumentOffsets {
  int in;
  int out;
  int count;
};

ArgumentOffsets declareArguments(KernelSignature &signature) {
  int in = signature.addPointer("in");
  int out = signature.addPointer("out");
  int count = signature.addScalar("count", 4);
  return {in, out, count};
}

// Emits the kernel; the ring slots hold the loads in flight.
class RingEmitter {
public:
  RingEmitter(Kernel &kernel, const ArgumentOffsets &offsets)
      : kernel(kernel), builder(kernel.builder()), tracker(kernel.builder()),
        offsets(offsets), inRsrc(builder.allocateSgpr(4, 4)),
        outRsrc(builder.allocateSgpr(4, 4)), remaining(builder.allocateSgpr(1)),
        base(builder.allocateSgpr(1)), voffset(builder.allocateVgpr(1)),
        acc(builder.allocateVgpr(1)) {
    for (int slot = 0; slot < ringDepth; ++slot)
      ring.push_back(builder.allocateVgpr(1));
  }

  void emit() {
    loadArguments();
    emitPrologue();
    exit = builder.createLabel("exit");
    builder.create<rocisa::SCmpEQU32>(remaining.container(), 0);
    builder.create<rocisa::SCBranchSCC1>(exit->getLabelName());
    WaitTracker::State entry = tracker.snapshot();
    emitLoop();
    // The loop can be skipped, so the exit joins the pre-loop state.
    builder.insert(exit);
    tracker.join(entry);
    emitDrain();
    builder.create<rocisa::BufferStoreB32>(acc.container(), voffset.container(),
                                           outRsrc.container(), 0,
                                           rocisa::MUBUFModifiers(true, 0));
    builder.createEndProgram();
  }

  std::string loopLabel() const { return loop->getLabelName(); }
  std::string exitLabel() const { return exit->getLabelName(); }

private:
  void loadArguments() {
    std::vector<WaitToken> tokens;
    tokens.push_back(
        tracker.issue(kernel.loadArgument(inRsrc.ref(0, 2), offsets.in)));
    tokens.push_back(
        tracker.issue(kernel.loadArgument(outRsrc.ref(0, 2), offsets.out)));
    tokens.push_back(
        tracker.issue(kernel.loadArgument(remaining.ref(), offsets.count)));
    tracker.waitFor(tokens);
    builder.makeRsrc(inRsrc.ref(), inRsrc.ref(0, 2), int64_t{0xFFFFFFFF});
    builder.makeRsrc(outRsrc.ref(), outRsrc.ref(0, 2),
                     int64_t{workgroupSize * 4});
  }

  void emitOffsets() {
    builder.create<rocisa::VAndB32>(voffset.container(), 0x3ff,
                                    kernel.workitemId().container());
    builder.create<rocisa::VLShiftLeftB32>(voffset.container(), 2,
                                           voffset.container());
    builder.movb32(base.ref(), 0);
  }

  WaitToken loadSlot(int slot) {
    return tracker.issue(builder.create<rocisa::BufferLoadB32>(
        ring[slot].container(), voffset.container(), inRsrc.container(),
        base.container(), rocisa::MUBUFModifiers(true, slot * slotStride)));
  }

  void emitPrologue() {
    emitOffsets();
    builder.movb32(acc.ref(), 0);
    for (int slot = 0; slot < ringDepth; ++slot)
      slots.push_back(loadSlot(slot));
    builder.addi32(base.ref(), base, ringDepth * slotStride);
    // The prologue loaded one group, and the loop loads the rest.
    builder.create<rocisa::SLShiftRightB32>(remaining.container(), 2,
                                            remaining.container());
    builder.subi32(remaining.ref(), remaining, 1);
  }

  void consume(int slot) {
    tracker.waitFor(slots[slot]);
    builder.muli32(acc.ref(), acc, 31);
    builder.addi32(acc.ref(), acc, ring[slot]);
  }

  void emitLoop() {
    loop = builder.createLabel("loop");
    builder.insert(loop);
    tracker.beginLoop();
    for (int slot = 0; slot < ringDepth; ++slot) {
      consume(slot);
      slots[slot] = loadSlot(slot);
    }
    builder.addi32(base.ref(), base, ringDepth * slotStride);
    builder.subi32(remaining.ref(), remaining, 1);
    builder.create<rocisa::SCmpLgU32>(remaining.container(), 0);
    builder.create<rocisa::SCBranchSCC1>(loop->getLabelName());
    tracker.endLoop();
  }

  void emitDrain() {
    for (int slot = 0; slot < ringDepth; ++slot)
      consume(slot);
  }

  Kernel &kernel;
  IRBuilder &builder;
  WaitTracker tracker;
  ArgumentOffsets offsets;
  RegisterRange inRsrc, outRsrc, remaining, base, voffset, acc;
  std::vector<RegisterRange> ring;
  std::vector<WaitToken> slots;
  std::shared_ptr<rocisa::Label> loop;
  std::shared_ptr<rocisa::Label> exit;
};

std::vector<uint32_t> makeInput(int count) {
  std::vector<uint32_t> in(count * workgroupSize);
  for (int i = 0; i < std::ssize(in); ++i)
    in[i] = 0x9E3779B9u * static_cast<uint32_t>(i + 1) ^ 0x5bd1e995u;
  return in;
}

std::vector<uint32_t> computeExpected(const std::vector<uint32_t> &in,
                                      int count) {
  std::vector<uint32_t> expected(workgroupSize, 0u);
  for (int lane = 0; lane < workgroupSize; ++lane)
    for (int i = 0; i < count; ++i)
      expected[lane] = expected[lane] * 31u + in[i * workgroupSize + lane];
  return expected;
}

std::vector<std::byte> makeKernarg(int segmentSize,
                                   const ArgumentOffsets &offsets,
                                   const void *in, const void *out,
                                   uint32_t count) {
  std::vector<std::byte> kernarg(segmentSize);
  std::memcpy(kernarg.data() + offsets.in, &in, sizeof(in));
  std::memcpy(kernarg.data() + offsets.out, &out, sizeof(out));
  std::memcpy(kernarg.data() + offsets.count, &count, sizeof(count));
  return kernarg;
}

void checkAssembly(const std::string &assembly, const RingEmitter &emitter) {
  std::string loopLine = emitter.loopLabel() + ":";
  std::size_t loopBegin = assembly.find(loopLine);
  std::size_t loopEnd =
      assembly.find("s_cbranch_scc1 " + emitter.loopLabel(), loopBegin);
  ASSERT_NE(loopBegin, std::string::npos);
  ASSERT_NE(loopEnd, std::string::npos);
  std::string loopBody = assembly.substr(loopBegin, loopEnd - loopBegin);
  EXPECT_EQ(countLinesContaining(loopBody, "s_waitcnt vmcnt(3)"), ringDepth);
  EXPECT_EQ(countLinesContaining(loopBody, "s_waitcnt"), ringDepth);

  std::size_t exitBegin = assembly.find(emitter.exitLabel() + ":");
  ASSERT_NE(exitBegin, std::string::npos);
  expectLinesInOrder(assembly.substr(exitBegin),
                     {"s_waitcnt vmcnt(3)", "s_waitcnt vmcnt(2)",
                      "s_waitcnt vmcnt(1)", "s_waitcnt vmcnt(0)"});
}

// Uploads the buffers, launches the kernel and downloads out.
void runKernel(hipFunction_t function, const std::vector<std::byte> &args,
               std::vector<uint32_t> &out, void *deviceOut) {
  size_t argsSize = args.size();
  void *extra[] = {
      HIP_LAUNCH_PARAM_BUFFER_POINTER, const_cast<std::byte *>(args.data()),
      HIP_LAUNCH_PARAM_BUFFER_SIZE, &argsSize, HIP_LAUNCH_PARAM_END};
  ASSERT_HIP_SUCCESS(hipModuleLaunchKernel(function, 1, 1, 1, workgroupSize, 1,
                                           1, 0, nullptr, nullptr, extra));
  ASSERT_HIP_SUCCESS(hipDeviceSynchronize());
  ASSERT_HIP_SUCCESS(hipMemcpy(out.data(), deviceOut,
                               out.size() * sizeof(uint32_t),
                               hipMemcpyDeviceToHost));
}

void runCount(hipFunction_t function, const ArgumentOffsets &offsets,
              int segmentSize, int count) {
  std::vector<uint32_t> in = makeInput(count);
  std::vector<uint32_t> out(workgroupSize, sentinel);
  void *rawIn = nullptr;
  void *rawOut = nullptr;
  ASSERT_HIP_SUCCESS(hipMalloc(&rawIn, in.size() * sizeof(uint32_t)));
  std::unique_ptr<void, DeviceFree> deviceIn(rawIn);
  ASSERT_HIP_SUCCESS(hipMalloc(&rawOut, out.size() * sizeof(uint32_t)));
  std::unique_ptr<void, DeviceFree> deviceOut(rawOut);
  ASSERT_HIP_SUCCESS(hipMemcpy(rawIn, in.data(), in.size() * sizeof(uint32_t),
                               hipMemcpyHostToDevice));
  ASSERT_HIP_SUCCESS(hipMemcpy(rawOut, out.data(),
                               out.size() * sizeof(uint32_t),
                               hipMemcpyHostToDevice));
  std::vector<std::byte> args = makeKernarg(segmentSize, offsets, rawIn, rawOut,
                                            static_cast<uint32_t>(count));
  runKernel(function, args, out, rawOut);
  if (testing::Test::HasFatalFailure())
    return;
  std::vector<uint32_t> expected = computeExpected(in, count);
  for (int lane = 0; lane < workgroupSize; ++lane)
    ASSERT_EQ(out[lane], expected[lane])
        << "count " << count << " lane " << lane;
}

TEST(WaitTrackerDevice, PrefetchRingLoop) {
  Target target = Target::get("gfx950");
  Kernel kernel(target, kernelName, workgroupSize);
  ArgumentOffsets offsets = declareArguments(kernel.signature());
  RingEmitter emitter(kernel, offsets);
  emitter.emit();
  std::string assembly = kernel.toAssembly();
  SCOPED_TRACE("assembly:\n" + assembly);
  checkAssembly(assembly, emitter);
  if (testing::Test::HasFatalFailure())
    return;

  std::vector<uint8_t> hsaco =
      assembleToHSACO(assembly, kernel.target().isaName());
  ASSERT_FALSE(hsaco.empty());
  if (!hasHipDevice())
    GTEST_SKIP() << "no HIP device";

  hipModule_t rawModule = nullptr;
  ASSERT_HIP_SUCCESS(hipModuleLoadData(&rawModule, hsaco.data()));
  ModulePtr module(rawModule);
  hipFunction_t function = nullptr;
  ASSERT_HIP_SUCCESS(hipModuleGetFunction(&function, rawModule, kernelName));
  for (int count : {4, 8, 12, 64, 1000}) {
    runCount(function, offsets, kernel.signature().segmentSize(), count);
    if (testing::Test::HasFatalFailure())
      return;
  }
}

} // namespace

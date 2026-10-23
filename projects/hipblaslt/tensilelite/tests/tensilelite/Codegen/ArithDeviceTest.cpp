// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ArithDeviceTest.hpp"

#include "HipTestSupport.hpp"

#include <tensilelite/API/Compiler.hpp>
#include <tensilelite/Codegen/Kernel.hpp>

#include "container.hpp"
#include "instruction/cmp.hpp"
#include "instruction/common.hpp"
#include "instruction/mem.hpp"

#include <gtest/gtest.h>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace rocm::tensilelite::codegen::test {

namespace {

constexpr const char *archName = "gfx950";
constexpr const char *kernelName = "codegen_arith";
constexpr uint32_t sentinel = 0xDEADBEEF;
constexpr int outputPadding = 64;
constexpr int maxReportedMismatches = 8;
constexpr uint32_t randomSeed = 0x5EED1234;
// A workgroup's lanes cover this many bytes of a lane input.
constexpr int groupBytesLog2 = 8;
constexpr int wordBytesLog2 = 2;

// Registers every address computation of the kernel is built from.
struct Addressing {
  RegisterRange localId;                   // Workitem id x, in a VGPR.
  RegisterRange groupOffset;               // Workgroup id * 4, in an SGPR.
  std::optional<RegisterRange> laneOffset; // Byte offset of this lane's word.
};

bool needsLaneOffset(const ArithKernelSpec &spec) {
  if (spec.result == RegisterKind::vgpr)
    return true;
  for (InputKind kind : spec.inputs)
    if (kind == InputKind::lane)
      return true;
  return false;
}

// Declares the output pointer and then one pointer per input.
std::vector<int> declarePointers(KernelSignature &signature, int inputCount) {
  std::vector<int> offsets;
  offsets.push_back(signature.addPointer("out"));
  for (int i = 0; i < inputCount; ++i)
    offsets.push_back(signature.addPointer("in" + std::to_string(i)));
  return offsets;
}

// Loads each pointer into its own aligned SGPR pair.
std::vector<RegisterRange> loadPointers(Kernel &kernel,
                                        const std::vector<int> &offsets) {
  IRBuilder &builder = kernel.builder();
  std::vector<RegisterRange> pointers;
  for (int offset : offsets) {
    pointers.push_back(builder.allocateSgpr(2, 2));
    builder.create<rocisa::SLoadB64>(pointers.back().container(),
                                     kernel.kernargSegmentPtr().container(),
                                     offset);
  }
  builder.createWaitCnt({.scalarMemory = 0});
  return pointers;
}

Addressing computeAddressing(Kernel &kernel, bool needsLane) {
  IRBuilder &builder = kernel.builder();
  Addressing addressing{builder.allocateVgpr(1), builder.allocateSgpr(1),
                        std::nullopt};
  // The workitem id register packs the x, y and z ids.
  builder.create<rocisa::VAndB32>(addressing.localId.container(), 0x3ff,
                                  kernel.workitemId().container());
  builder.create<rocisa::SLShiftLeftB32>(addressing.groupOffset.container(),
                                         wordBytesLog2,
                                         kernel.workgroupIdX().container());
  if (!needsLane)
    return addressing;
  RegisterRange groupBase = builder.allocateSgpr(1);
  builder.create<rocisa::SLShiftLeftB32>(groupBase.container(), groupBytesLog2,
                                         kernel.workgroupIdX().container());
  addressing.laneOffset = builder.allocateVgpr(1);
  builder.create<rocisa::VLShiftLeftB32>(addressing.laneOffset->container(),
                                         wordBytesLog2,
                                         addressing.localId.container());
  builder.create<rocisa::VAddU32>(addressing.laneOffset->container(),
                                  groupBase.container(),
                                  addressing.laneOffset->container());
  return addressing;
}

std::vector<RegisterRange>
loadInputs(Kernel &kernel, const ArithKernelSpec &spec,
           const std::vector<RegisterRange> &pointers,
           const Addressing &addressing) {
  IRBuilder &builder = kernel.builder();
  std::vector<RegisterRange> values;
  for (int i = 0; i < std::ssize(spec.inputs); ++i) {
    const RegisterRange &pointer = pointers[i + 1];
    if (spec.inputs[i] == InputKind::lane) {
      values.push_back(builder.allocateVgpr(1));
      builder.create<rocisa::GlobalLoadB32>(values.back().container(),
                                            addressing.laneOffset->container(),
                                            pointer.container());
      continue;
    }
    values.push_back(builder.allocateSgpr(1));
    builder.create<rocisa::SLoadB32>(values.back().container(),
                                     pointer.container(),
                                     addressing.groupOffset.container());
  }
  builder.createWaitCnt({.vectorLoads = 0, .scalarMemory = 0});
  return values;
}

// Lane 0 stores the uniform result to out[workgroupId].
void storeSgprResult(Kernel &kernel, const RegisterRange &result,
                     const RegisterRange &outPointer,
                     const Addressing &addressing) {
  IRBuilder &builder = kernel.builder();
  RegisterRange value = builder.allocateVgpr(1);
  RegisterRange address = builder.allocateVgpr(1);
  RegisterRange savedExec = builder.allocateSgpr(2, 2);
  builder.create<rocisa::VMovB32>(value.container(), result.container());
  builder.create<rocisa::VMovB32>(address.container(),
                                  addressing.groupOffset.container());
  auto vcc = std::make_shared<rocisa::VCC>();
  builder.create<rocisa::VCmpEQU32>(vcc, 0, addressing.localId.container());
  builder.create<rocisa::SAndSaveExecB64>(savedExec.container(), vcc);
  builder.create<rocisa::GlobalStoreB32>(address.container(), value.container(),
                                         outPointer.container());
}

void storeResult(Kernel &kernel, const ArithKernelSpec &spec,
                 const RegisterRange &result, const RegisterRange &outPointer,
                 const Addressing &addressing) {
  if (spec.result == RegisterKind::sgpr) {
    storeSgprResult(kernel, result, outPointer, addressing);
    return;
  }
  kernel.builder().create<rocisa::GlobalStoreB32>(
      addressing.laneOffset->container(), result.container(),
      outPointer.container());
}

RegisterRange allocateResult(IRBuilder &builder, RegisterKind kind) {
  return kind == RegisterKind::sgpr ? builder.allocateSgpr(1)
                                    : builder.allocateVgpr(1);
}

std::vector<RegisterRef> refsOf(const std::vector<RegisterRange> &ranges) {
  std::vector<RegisterRef> refs;
  for (const RegisterRange &range : ranges)
    refs.push_back(range.ref());
  return refs;
}

std::string describeWord(uint32_t word) {
  char buffer[48];
  std::snprintf(buffer, sizeof(buffer), "0x%08x (%d)", word,
                static_cast<int>(word));
  return buffer;
}

// Device-side layout of the kernel arguments, shared with the kernel builder.
struct ArgumentLayout {
  std::vector<int> pointerOffsets; // out first, then the inputs.
  int segmentSize;
};

ArgumentLayout makeLayout(int inputCount) {
  KernelSignature signature;
  std::vector<int> offsets = declarePointers(signature, inputCount);
  return {std::move(offsets), signature.segmentSize()};
}

std::vector<std::byte> makeKernarg(const ArgumentLayout &layout,
                                   const std::vector<const void *> &pointers) {
  std::vector<std::byte> kernarg(layout.segmentSize);
  for (int i = 0; i < std::ssize(pointers); ++i)
    std::memcpy(kernarg.data() + layout.pointerOffsets[i], &pointers[i],
                sizeof(void *));
  return kernarg;
}

void uploadWords(const std::vector<uint32_t> &words,
                 std::unique_ptr<void, DeviceFree> &buffer) {
  void *ptr = nullptr;
  ASSERT_HIP_SUCCESS(hipMalloc(&ptr, words.size() * sizeof(uint32_t)));
  buffer.reset(ptr);
  ASSERT_HIP_SUCCESS(hipMemcpy(ptr, words.data(),
                               words.size() * sizeof(uint32_t),
                               hipMemcpyHostToDevice));
}

// Uploads the inputs and a sentinel-filled output, launches the kernel and
// downloads the output.
void runKernel(const std::vector<uint8_t> &hsaco, const ArgumentLayout &layout,
               int workgroupCount,
               const std::vector<std::vector<uint32_t>> &inputs,
               std::vector<uint32_t> &output) {
  hipModule_t rawModule = nullptr;
  ASSERT_HIP_SUCCESS(hipModuleLoadData(&rawModule, hsaco.data()));
  ModulePtr module(rawModule);
  hipFunction_t function = nullptr;
  ASSERT_HIP_SUCCESS(hipModuleGetFunction(&function, rawModule, kernelName));

  std::vector<std::unique_ptr<void, DeviceFree>> buffers(inputs.size() + 1);
  uploadWords(output, buffers[0]);
  ASSERT_FALSE(testing::Test::HasFatalFailure());
  std::vector<const void *> pointers{buffers[0].get()};
  for (int i = 0; i < std::ssize(inputs); ++i) {
    uploadWords(inputs[i], buffers[i + 1]);
    ASSERT_FALSE(testing::Test::HasFatalFailure());
    pointers.push_back(buffers[i + 1].get());
  }

  std::vector<std::byte> kernarg = makeKernarg(layout, pointers);
  size_t kernargSize = kernarg.size();
  void *extra[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, kernarg.data(),
                   HIP_LAUNCH_PARAM_BUFFER_SIZE, &kernargSize,
                   HIP_LAUNCH_PARAM_END};
  ASSERT_HIP_SUCCESS(hipModuleLaunchKernel(function, workgroupCount, 1, 1,
                                           arithWorkgroupSize, 1, 1, 0, nullptr,
                                           nullptr, extra));
  ASSERT_HIP_SUCCESS(hipDeviceSynchronize());
  ASSERT_HIP_SUCCESS(hipMemcpy(output.data(), buffers[0].get(),
                               output.size() * sizeof(uint32_t),
                               hipMemcpyDeviceToHost));
}

// The input word the kernel read for result index.
uint32_t inputWordFor(const ArithKernelSpec &spec, int input, int index,
                      const std::vector<uint32_t> &words) {
  bool lanes = spec.result == RegisterKind::vgpr;
  int group = lanes ? index / arithWorkgroupSize : index;
  if (spec.inputs[input] == InputKind::uniform)
    return words[group];
  return words[lanes ? index : index * arithWorkgroupSize];
}

void reportMismatch(const ArithKernelSpec &spec, int index, uint32_t actual,
                    uint32_t expected,
                    const std::vector<std::vector<uint32_t>> &inputs) {
  std::ostringstream message;
  message << "result " << index << ": got " << describeWord(actual)
          << ", expected " << describeWord(expected);
  for (int i = 0; i < std::ssize(inputs); ++i)
    message << "\n  input " << i << " = "
            << describeWord(inputWordFor(spec, i, index, inputs[i]));
  ADD_FAILURE() << message.str();
}

void compareResults(const ArithKernelSpec &spec,
                    const std::vector<std::vector<uint32_t>> &inputs,
                    const std::vector<uint32_t> &expected,
                    const std::vector<uint32_t> &output) {
  int mismatches = 0;
  for (int i = 0; i < std::ssize(expected); ++i) {
    if (output[i] == expected[i])
      continue;
    if (mismatches++ < maxReportedMismatches)
      reportMismatch(spec, i, output[i], expected[i], inputs);
  }
  if (mismatches > 0)
    ADD_FAILURE() << mismatches << " of " << expected.size()
                  << " results mismatch";
  for (size_t i = expected.size(); i < output.size(); ++i)
    if (output[i] != sentinel) {
      ADD_FAILURE() << "padding word " << i - expected.size()
                    << " was overwritten with " << describeWord(output[i]);
      return;
    }
}

void checkInputSizes(const ArithKernelSpec &spec, int workgroupCount,
                     const std::vector<std::vector<uint32_t>> &inputs,
                     const std::vector<uint32_t> &expected) {
  ASSERT_EQ(inputs.size(), spec.inputs.size());
  for (int i = 0; i < std::ssize(inputs); ++i) {
    size_t words =
        spec.inputs[i] == InputKind::lane
            ? static_cast<size_t>(workgroupCount) * arithWorkgroupSize
            : static_cast<size_t>(workgroupCount);
    ASSERT_EQ(inputs[i].size(), words) << "input " << i;
  }
  size_t results =
      spec.result == RegisterKind::vgpr
          ? static_cast<size_t>(workgroupCount) * arithWorkgroupSize
          : static_cast<size_t>(workgroupCount);
  ASSERT_EQ(expected.size(), results);
}

} // namespace

std::string buildArithAssembly(const ArithKernelSpec &spec) {
  Target target = Target::get(archName);
  Kernel kernel(target, kernelName, arithWorkgroupSize);
  std::vector<int> offsets =
      declarePointers(kernel.signature(), static_cast<int>(spec.inputs.size()));
  IRBuilder &builder = kernel.builder();

  std::vector<RegisterRange> pointers = loadPointers(kernel, offsets);
  Addressing addressing = computeAddressing(kernel, needsLaneOffset(spec));
  std::vector<RegisterRange> values =
      loadInputs(kernel, spec, pointers, addressing);
  RegisterRange result = allocateResult(builder, spec.result);
  spec.emit(builder, result.ref(), refsOf(values));
  storeResult(kernel, spec, result, pointers[0], addressing);
  // Exec is not restored after a uniform store because the program ends.
  builder.createEndProgram();
  return kernel.toAssembly();
}

void expectArithResults(const ArithKernelSpec &spec, int workgroupCount,
                        const std::vector<std::vector<uint32_t>> &inputs,
                        const std::vector<uint32_t> &expected) {
  std::string assembly = buildArithAssembly(spec);
  SCOPED_TRACE("assembly:\n" + assembly);
  checkInputSizes(spec, workgroupCount, inputs, expected);
  if (testing::Test::HasFatalFailure())
    return;

  std::vector<uint8_t> hsaco =
      assembleToHSACO(assembly, Target::get(archName).isaName());
  ASSERT_FALSE(hsaco.empty());
  std::vector<uint32_t> output(expected.size() + outputPadding, sentinel);
  runKernel(hsaco, makeLayout(static_cast<int>(inputs.size())), workgroupCount,
            inputs, output);
  if (testing::Test::HasFatalFailure())
    return;
  compareResults(spec, inputs, expected, output);
}

std::vector<uint32_t> arithEdgeValues() {
  return {0,           1,           0xFFFFFFFFu, 2,           0xFFFFFFFEu,
          64,          65,          0xFFFFFFF0u, 0xFFFFFFEFu, 0x80000000u,
          0x80000001u, 0x7FFFFFFFu, 0x7FFFFFFEu, 0xFFFF,      0x10000,
          0x12345678,  0x9E3779B9};
}

WordPairs makeArithPairs(int randomCount) {
  WordPairs pairs;
  std::vector<uint32_t> edges = arithEdgeValues();
  for (uint32_t a : edges)
    for (uint32_t b : edges) {
      pairs.a.push_back(a);
      pairs.b.push_back(b);
    }
  std::mt19937 engine(randomSeed);
  int count = static_cast<int>(pairs.a.size()) + randomCount;
  // Whole workgroups of lanes need a multiple of the workgroup size.
  count = (count + arithWorkgroupSize - 1) / arithWorkgroupSize *
          arithWorkgroupSize;
  while (std::ssize(pairs.a) < count) {
    pairs.a.push_back(static_cast<uint32_t>(engine()));
    pairs.b.push_back(static_cast<uint32_t>(engine()));
  }
  return pairs;
}

namespace {

std::vector<uint64_t> arithEdgeValues64() {
  return {0,
          1,
          UINT64_MAX,
          2,
          static_cast<uint64_t>(INT64_MIN),
          static_cast<uint64_t>(INT64_MIN) + 1,
          static_cast<uint64_t>(INT64_MAX),
          static_cast<uint64_t>(INT64_MAX) - 1,
          0x0000000080000000,
          0x000000007FFFFFFF,
          0xFFFFFFFF00000000,
          0x0000000100000000,
          0x00000000FFFFFFFF,
          0x123456789ABCDEF0,
          0x1234567800000001,
          0x12345678FFFFFFFF,
          0x1234567880000000,
          0xFFFFFFFF80000000,
          0xFFFFFFFF7FFFFFFF};
}

void appendQuad(WordQuads &quads, uint64_t a, uint64_t b) {
  quads.aLo.push_back(static_cast<uint32_t>(a));
  quads.aHi.push_back(static_cast<uint32_t>(a >> 32));
  quads.bLo.push_back(static_cast<uint32_t>(b));
  quads.bHi.push_back(static_cast<uint32_t>(b >> 32));
}

uint64_t randomWord64(std::mt19937 &engine) {
  uint64_t low = engine();
  return low | uint64_t{engine()} << 32;
}

} // namespace

WordQuads makeArithPairs64(int randomCount) {
  WordQuads quads;
  std::vector<uint64_t> edges = arithEdgeValues64();
  for (uint64_t a : edges)
    for (uint64_t b : edges)
      appendQuad(quads, a, b);
  std::mt19937 engine(randomSeed);
  int count = static_cast<int>(quads.aLo.size()) + randomCount;
  count = (count + arithWorkgroupSize - 1) / arithWorkgroupSize *
          arithWorkgroupSize;
  while (std::ssize(quads.aLo) < count) {
    uint64_t a = randomWord64(engine);
    appendQuad(quads, a, randomWord64(engine));
  }
  return quads;
}

PairRegisters makePair(IRBuilder &builder, RegisterRef lo, RegisterRef hi,
                       bool misaligned) {
  int count = misaligned ? 3 : 2;
  RegisterRange range = lo.kind == RegisterKind::sgpr
                            ? builder.allocateSgpr(count, 2)
                            : builder.allocateVgpr(count, 2);
  RegisterRef pair = range.ref(misaligned ? 1 : 0, 2);
  builder.movb32(pair.slice(0), lo);
  builder.movb32(pair.slice(1), hi);
  return {std::move(range), pair};
}

} // namespace rocm::tensilelite::codegen::test

// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tensilelite/Codegen/IRBuilder.hpp>

#include "ArithDeviceTest.hpp"
#include "HipTestSupport.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace rocm::tensilelite::codegen;
using namespace rocm::tensilelite::codegen::test;

constexpr RegisterKind vgpr = RegisterKind::vgpr;
constexpr RegisterKind sgpr = RegisterKind::sgpr;
constexpr InputKind lane = InputKind::lane;
constexpr InputKind uniform = InputKind::uniform;

constexpr int randomPairs = 1024;

using Words = std::vector<uint32_t>;
using Values = std::vector<uint64_t>;
using Inputs = std::vector<Words>;

// Immediates that cover inline constants, literals and negative values.
const int32_t immediates32[] = {5, -16, 0x12345678, -0x65432110, 64};
// A pair immediate is sign-extended to 64 bits.
const int32_t immediates64[] = {
    0, 1, -1, 64, 65, -16, -17, INT32_MIN, INT32_MAX, 0x12345678, -0x12345678};

// How the destination sits relative to the operand registers.
enum class Placement {
  // The destination and the operands are separate registers.
  separate,
  // See emitOverlapping for the three overlapping layouts.
  highFirst,
  lowFirst,
  copied,
};

struct SelectCase {
  int width;
  RegisterKind dstKind;
  InputKind condKind;
  InputKind operandKind;
  Placement placement = Placement::separate;
  bool dstMisaligned = false;
  bool operandsMisaligned = false;
  // An operand with no immediate is a register or a pair from the inputs.
  std::optional<int32_t> a;
  std::optional<int32_t> b;
};

struct SelectData {
  Words conds;
  Values a;
  Values b;
};

// Condition words with zero, one, other small values, the sign bit, all ones
// and pseudo-random words, so that "nonzero is true" is exercised.
Words makeConds(size_t count) {
  const Words edges{0, 1, 2, 0x80000000u, 0xFFFFFFFFu};
  Words conds;
  for (size_t i = 0; i < count; ++i)
    conds.push_back(i % 7 < edges.size()
                        ? edges[i % 7]
                        : static_cast<uint32_t>(i) * 2654435761u);
  return conds;
}

Values widen(const Words &words) { return Values(words.begin(), words.end()); }

SelectData makeData32() {
  WordPairs pairs = makeArithPairs(randomPairs);
  return {makeConds(pairs.a.size()), widen(pairs.a), widen(pairs.b)};
}

Values join(const Words &low, const Words &high) {
  Values values;
  for (size_t i = 0; i < low.size(); ++i)
    values.push_back(low[i] | uint64_t{high[i]} << 32);
  return values;
}

SelectData makeData64() {
  WordQuads quads = makeArithPairs64(randomPairs);
  return {makeConds(quads.aLo.size()), join(quads.aLo, quads.aHi),
          join(quads.bLo, quads.bHi)};
}

Words lowWords(const Values &values) {
  Words words;
  for (uint64_t value : values)
    words.push_back(static_cast<uint32_t>(value));
  return words;
}

Words highWords(const Values &values) {
  Words words;
  for (uint64_t value : values)
    words.push_back(static_cast<uint32_t>(value >> 32));
  return words;
}

RegisterRange allocateKind(IRBuilder &builder, RegisterKind kind, int count) {
  return kind == sgpr ? builder.allocateSgpr(count, 2)
                      : builder.allocateVgpr(count, 2);
}

// The condition register: the uniform bool input itself, or a lane mask that
// is set where a lane input is nonzero.
struct Condition {
  std::optional<RegisterRange> owner;
  RegisterRef ref;
};

Condition makeCondition(IRBuilder &builder, InputKind kind, RegisterRef input) {
  if (kind == uniform)
    return {std::nullopt, input};
  RegisterRange mask = builder.allocateSgpr(2, 2);
  builder.cmpi32(Predicate::ne, mask.ref(), input, 0);
  RegisterRef ref = mask.ref();
  return {std::move(mask), ref};
}

// Reads an operand: an immediate, one input register, or a pair built from two
// inputs.
Operand makeOperand(IRBuilder &builder, const SelectCase &spec,
                    const std::optional<int32_t> &immediate,
                    const std::vector<RegisterRef> &in, size_t &next,
                    std::vector<RegisterRange> &owners) {
  if (immediate)
    return *immediate;
  if (spec.width == 1)
    return in[next++];
  PairRegisters pair =
      makePair(builder, in[next], in[next + 1], spec.operandsMisaligned);
  next += 2;
  owners.push_back(std::move(pair.range));
  return pair.pair;
}

void emitSelect(IRBuilder &builder, int width, RegisterRef dst,
                RegisterRef cond, Operand a, Operand b) {
  if (width == 1)
    builder.selectb32(dst, cond, a, b);
  else
    builder.selectb64(dst, cond, a, b);
}

PairRegisters emitSeparate(IRBuilder &builder, const SelectCase &spec,
                           RegisterRef cond,
                           const std::vector<RegisterRef> &in) {
  size_t next = 1;
  std::vector<RegisterRange> owners;
  Operand a = makeOperand(builder, spec, spec.a, in, next, owners);
  Operand b = makeOperand(builder, spec, spec.b, in, next, owners);
  int count = spec.width == 1 ? 1 : spec.dstMisaligned ? 3 : 2;
  RegisterRange range = allocateKind(builder, spec.dstKind, count);
  RegisterRef dst = range.ref(count - spec.width, spec.width);
  emitSelect(builder, spec.width, dst, cond, a, b);
  return {std::move(range), dst};
}

// Places the destination in a range that also holds operand registers. In
// highFirst, a's high word is the destination's low register; in lowFirst, the
// destination's high register is a's low register; in copied, both overlap.
PairRegisters emitOverlapping(IRBuilder &builder, const SelectCase &spec,
                              RegisterRef cond,
                              const std::vector<RegisterRef> &in) {
  RegisterRange range = allocateKind(builder, spec.dstKind, 4);
  PairRegisters other = makePair(builder, in[3], in[4]);
  RegisterRef first = range.ref(0, 2);
  RegisterRef second = range.ref(1, 2);
  RegisterRef third = range.ref(2, 2);
  RegisterRef dst = second;
  RegisterRef a = first;
  RegisterRef b = other.pair;
  if (spec.placement == Placement::lowFirst) {
    dst = first;
    a = second;
  } else if (spec.placement == Placement::copied) {
    b = third;
  }
  builder.movb32(a.slice(0), in[1]);
  builder.movb32(a.slice(1), in[2]);
  if (spec.placement == Placement::copied) {
    builder.movb32(b.slice(0), in[3]);
    builder.movb32(b.slice(1), in[4]);
  }
  builder.selectb64(dst, cond, a, b);
  return {std::move(range), dst};
}

PairRegisters emitSelectCase(IRBuilder &builder, const SelectCase &spec,
                             const std::vector<RegisterRef> &in) {
  Condition cond = makeCondition(builder, spec.condKind, in[0]);
  if (spec.placement == Placement::separate)
    return emitSeparate(builder, spec, cond.ref, in);
  return emitOverlapping(builder, spec, cond.ref, in);
}

// A result is one lane of a vgpr result or one workgroup of an sgpr result.
size_t inputIndex(const SelectCase &spec, InputKind kind, size_t result) {
  bool perResult = spec.dstKind == sgpr || kind == lane;
  return perResult ? result : result / arithWorkgroupSize;
}

uint64_t operandValue(const SelectCase &spec,
                      const std::optional<int32_t> &immediate,
                      const Values &values, size_t result) {
  uint64_t value = immediate
                       ? static_cast<uint64_t>(int64_t{*immediate})
                       : values[inputIndex(spec, spec.operandKind, result)];
  return spec.width == 1 ? static_cast<uint32_t>(value) : value;
}

Values expectedValues(const SelectCase &spec, const SelectData &data,
                      size_t results) {
  Values expected;
  for (size_t i = 0; i < results; ++i) {
    bool holds = data.conds[inputIndex(spec, spec.condKind, i)] != 0;
    expected.push_back(operandValue(spec, holds ? spec.a : spec.b,
                                    holds ? data.a : data.b, i));
  }
  return expected;
}

// Collects the kernel inputs and the sizes the harness expects for them.
struct KernelInputs {
  std::vector<InputKind> kinds;
  Inputs words;
};

void addInput(KernelInputs &inputs, InputKind kind, const Words &source,
              size_t laneCount, size_t groupCount) {
  inputs.kinds.push_back(kind);
  size_t count = kind == lane ? laneCount : groupCount;
  inputs.words.emplace_back(source.begin(), source.begin() + count);
}

void addOperandInputs(KernelInputs &inputs, const SelectCase &spec,
                      const std::optional<int32_t> &immediate,
                      const Values &values, size_t laneCount,
                      size_t groupCount) {
  if (immediate)
    return;
  addInput(inputs, spec.operandKind, lowWords(values), laneCount, groupCount);
  if (spec.width == 2)
    addInput(inputs, spec.operandKind, highWords(values), laneCount,
             groupCount);
}

KernelInputs makeKernelInputs(const SelectCase &spec, const SelectData &data,
                              size_t groupCount) {
  size_t laneCount = data.conds.size();
  KernelInputs inputs;
  addInput(inputs, spec.condKind, data.conds, laneCount, groupCount);
  addOperandInputs(inputs, spec, spec.a, data.a, laneCount, groupCount);
  addOperandInputs(inputs, spec, spec.b, data.b, laneCount, groupCount);
  return inputs;
}

// Runs the select and checks it, a 64-bit result through two runs that each
// store one half.
void expectSelectResults(const SelectCase &spec, const SelectData &data) {
  size_t results = data.conds.size();
  size_t groups = spec.dstKind == sgpr ? results : results / arithWorkgroupSize;
  size_t expectedCount = spec.dstKind == sgpr ? groups : results;
  KernelInputs inputs = makeKernelInputs(spec, data, groups);
  Values expected = expectedValues(spec, data, expectedCount);
  for (int half = 0; half < spec.width; ++half) {
    SCOPED_TRACE(spec.width == 1 ? "word"
                 : half == 0     ? "low half"
                                 : "high half");
    ArithKernelSpec kernel{inputs.kinds, spec.dstKind,
                           [spec, half](IRBuilder &builder, RegisterRef dst,
                                        const std::vector<RegisterRef> &in) {
                             PairRegisters value =
                                 emitSelectCase(builder, spec, in);
                             builder.movb32(dst, value.pair.slice(half));
                           }};
    Words words = half == 0 ? lowWords(expected) : highWords(expected);
    expectArithResults(kernel, static_cast<int>(groups), inputs.words, words);
  }
}

class SelectDevice : public testing::Test {
protected:
  void SetUp() override {
    if (!hasHipDevice())
      GTEST_SKIP() << "no HIP device";
  }
};

SelectCase case32(RegisterKind dstKind, InputKind condKind,
                  InputKind operandKind) {
  return {1, dstKind, condKind, operandKind};
}

SelectCase case64(RegisterKind dstKind, InputKind condKind,
                  InputKind operandKind) {
  return {2, dstKind, condKind, operandKind};
}

TEST_F(SelectDevice, ScalarWords) {
  expectSelectResults(case32(sgpr, uniform, uniform), makeData32());
}

TEST_F(SelectDevice, VectorWordsWithUniformBoolCondition) {
  expectSelectResults(case32(vgpr, uniform, lane), makeData32());
}

TEST_F(SelectDevice, VectorWordsWithMaskCondition) {
  SelectData data = makeData32();
  expectSelectResults(case32(vgpr, lane, lane), data);
  expectSelectResults(case32(vgpr, lane, uniform), data);
  expectSelectResults(case32(vgpr, uniform, uniform), data);
}

TEST_F(SelectDevice, WordImmediates) {
  SelectData data = makeData32();
  for (int32_t a : immediates32) {
    for (int32_t b : immediates32) {
      SCOPED_TRACE("a " + std::to_string(a) + ", b " + std::to_string(b));
      for (SelectCase spec :
           {case32(sgpr, uniform, uniform), case32(vgpr, uniform, uniform),
            case32(vgpr, lane, lane)}) {
        spec.a = a;
        spec.b = b;
        expectSelectResults(spec, data);
      }
    }
  }
}

TEST_F(SelectDevice, WordImmediateAgainstRegister) {
  SelectData data = makeData32();
  for (int32_t immediate : immediates32) {
    SCOPED_TRACE(immediate);
    for (SelectCase spec :
         {case32(sgpr, uniform, uniform), case32(vgpr, lane, lane),
          case32(vgpr, lane, uniform)}) {
      SelectCase immediateFirst = spec;
      immediateFirst.a = immediate;
      expectSelectResults(immediateFirst, data);
      spec.b = immediate;
      expectSelectResults(spec, data);
    }
  }
}

TEST_F(SelectDevice, ScalarPairs) {
  SelectData data = makeData64();
  for (bool dstMisaligned : {false, true}) {
    for (bool operandsMisaligned : {false, true}) {
      SCOPED_TRACE(
          std::string(dstMisaligned ? "misaligned dst, " : "aligned dst, ") +
          (operandsMisaligned ? "misaligned operands" : "aligned operands"));
      SelectCase spec = case64(sgpr, uniform, uniform);
      spec.dstMisaligned = dstMisaligned;
      spec.operandsMisaligned = operandsMisaligned;
      expectSelectResults(spec, data);
    }
  }
}

TEST_F(SelectDevice, ScalarPairImmediates) {
  SelectData data = makeData64();
  const int count = static_cast<int>(std::size(immediates64));
  for (bool dstMisaligned : {false, true}) {
    for (int i = 0; i < count; ++i) {
      int32_t immediate = immediates64[i];
      SCOPED_TRACE("immediate " + std::to_string(immediate) +
                   (dstMisaligned ? " to misaligned dst" : " to aligned dst"));
      SelectCase spec = case64(sgpr, uniform, uniform);
      spec.dstMisaligned = dstMisaligned;
      SelectCase immediateFirst = spec;
      immediateFirst.a = immediate;
      expectSelectResults(immediateFirst, data);
      spec.b = immediate;
      expectSelectResults(spec, data);
      SelectCase both = case64(sgpr, uniform, uniform);
      both.dstMisaligned = dstMisaligned;
      both.a = immediate;
      both.b = immediates64[(i + 1) % count];
      expectSelectResults(both, data);
    }
  }
}

TEST_F(SelectDevice, VectorPairs) {
  SelectData data = makeData64();
  for (InputKind condKind : {lane, uniform}) {
    for (bool dstMisaligned : {false, true}) {
      SCOPED_TRACE(std::string(condKind == lane ? "mask cond, "
                                                : "uniform bool cond, ") +
                   (dstMisaligned ? "misaligned dst" : "aligned dst"));
      SelectCase spec = case64(vgpr, condKind, lane);
      spec.dstMisaligned = dstMisaligned;
      expectSelectResults(spec, data);
      spec.operandsMisaligned = true;
      expectSelectResults(spec, data);
    }
  }
  expectSelectResults(case64(vgpr, lane, uniform), data);
  expectSelectResults(case64(vgpr, uniform, uniform), data);
}

TEST_F(SelectDevice, VectorPairImmediates) {
  SelectData data = makeData64();
  const int count = static_cast<int>(std::size(immediates64));
  for (InputKind condKind : {lane, uniform}) {
    for (bool dstMisaligned : {false, true}) {
      for (int i = 0; i < count; ++i) {
        int32_t immediate = immediates64[i];
        SCOPED_TRACE(
            "immediate " + std::to_string(immediate) +
            (condKind == lane ? ", mask cond" : ", uniform bool cond") +
            (dstMisaligned ? ", misaligned dst" : ", aligned dst"));
        SelectCase spec = case64(vgpr, condKind, lane);
        spec.dstMisaligned = dstMisaligned;
        SelectCase immediateFirst = spec;
        immediateFirst.a = immediate;
        expectSelectResults(immediateFirst, data);
        spec.b = immediate;
        expectSelectResults(spec, data);
        SelectCase both = case64(vgpr, condKind, lane);
        both.dstMisaligned = dstMisaligned;
        both.a = immediate;
        both.b = immediates64[(i + 1) % count];
        expectSelectResults(both, data);
      }
    }
  }
}

TEST_F(SelectDevice, OverlappingDestinationOrdering) {
  SelectData data = makeData64();
  for (RegisterKind dstKind : {sgpr, vgpr}) {
    InputKind kind = dstKind == sgpr ? uniform : lane;
    for (Placement placement :
         {Placement::highFirst, Placement::lowFirst, Placement::copied}) {
      SCOPED_TRACE(std::string(dstKind == sgpr ? "sgpr" : "vgpr") +
                   " placement " + std::to_string(static_cast<int>(placement)));
      SelectCase spec = case64(dstKind, kind, kind);
      spec.placement = placement;
      expectSelectResults(spec, data);
    }
  }
}

} // namespace

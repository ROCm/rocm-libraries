// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tensilelite/Compiler.hpp>
#include <tensilelite/codegen/IRBuilder.hpp>
#include <tensilelite/codegen/Kernel.hpp>

#include "ArithDeviceTest.hpp"
#include "HipTestSupport.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace rocm::tensilelite;
using namespace rocm::tensilelite::codegen;
using namespace rocm::tensilelite::codegen::test;

constexpr RegisterKind vgpr = RegisterKind::vgpr;
constexpr RegisterKind sgpr = RegisterKind::sgpr;
constexpr InputKind lane = InputKind::lane;
constexpr InputKind uniform = InputKind::uniform;

using BuilderOp = void (IRBuilder::*)(RegisterRef, Operand, Operand);
using Words = std::vector<uint32_t>;
using Values = std::vector<uint64_t>;

struct OpCase {
  const char *name;
  BuilderOp op;
  uint64_t (*host)(uint64_t, uint64_t);
};

// The host arithmetic wraps modulo 2^64 like the device.
const OpCase opCases[] = {
    {"addi64", &IRBuilder::addi64,
     [](uint64_t a, uint64_t b) { return a + b; }},
    {"subi64", &IRBuilder::subi64,
     [](uint64_t a, uint64_t b) { return a - b; }},
    {"muli64", &IRBuilder::muli64,
     [](uint64_t a, uint64_t b) { return a * b; }},
};

// Immediates are int32 values sign-extended to 64 bits: inline constants,
// positive and negative literals, and the extremes.
const int32_t immediates[] = {0,         1,         -1,         64,
                              65,        -16,       -17,        -100,
                              INT32_MIN, INT32_MAX, 0x12345678, -0x12345678};

// How the destination sits relative to the operand registers.
enum class Placement {
  // The destination and the operands are separate registers.
  separate,
  // The destination is a or b.
  destinationIsA,
  destinationIsB,
  // The destination's low register is a's high register.
  lowIsAHigh,
  // The destination's high register is a's low register.
  highIsALow,
  // a and b are the same register pair.
  sameOperand,
};

struct Case64 {
  OpCase op;
  RegisterKind dstKind;
  InputKind aKind;
  InputKind bKind;
  Placement placement = Placement::separate;
  bool dstMisaligned = false;
  bool operandsMisaligned = false;
  // An operand with no immediate is a register pair built from two inputs.
  std::optional<int32_t> a;
  std::optional<int32_t> b;
};

struct Data64 {
  Values a;
  Values b;
};

Values join(const Words &low, const Words &high) {
  Values values;
  for (size_t i = 0; i < low.size(); ++i)
    values.push_back(low[i] | uint64_t{high[i]} << 32);
  return values;
}

Data64 makeData(int randomCount) {
  WordQuads quads = makeArithPairs64(randomCount);
  return {join(quads.aLo, quads.aHi), join(quads.bLo, quads.bHi)};
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

// The registers that stay allocated while the op under test runs.
struct Emitted {
  std::vector<RegisterRange> owners;
  RegisterRef value;
};

// Reads an operand: an immediate, or a pair built from two inputs.
Operand makeSource(IRBuilder &builder, const Case64 &spec,
                   const std::optional<int32_t> &immediate,
                   const std::vector<RegisterRef> &in, size_t &next,
                   std::vector<RegisterRange> &owners) {
  if (immediate)
    return *immediate;
  PairRegisters pair =
      makePair(builder, in[next], in[next + 1], spec.operandsMisaligned);
  next += 2;
  owners.push_back(std::move(pair.range));
  return pair.pair;
}

RegisterRange allocateKind(IRBuilder &builder, RegisterKind kind, int count) {
  return kind == sgpr ? builder.allocateSgpr(count, 2)
                      : builder.allocateVgpr(count, 2);
}

Emitted emitSeparate(IRBuilder &builder, const Case64 &spec,
                     const std::vector<RegisterRef> &in) {
  Emitted emitted{{}, RegisterRef{}};
  size_t next = 0;
  Operand a = makeSource(builder, spec, spec.a, in, next, emitted.owners);
  Operand b = spec.placement == Placement::sameOperand
                  ? a
                  : makeSource(builder, spec, spec.b, in, next, emitted.owners);
  int count = spec.dstMisaligned ? 3 : 2;
  RegisterRange range = allocateKind(builder, spec.dstKind, count);
  RegisterRef dst = range.ref(count - 2, 2);
  (builder.*spec.op.op)(dst, a, b);
  emitted.owners.push_back(std::move(range));
  emitted.value = dst;
  return emitted;
}

Emitted emitInPlace(IRBuilder &builder, const Case64 &spec,
                    const std::vector<RegisterRef> &in) {
  Emitted emitted{{}, RegisterRef{}};
  size_t next = 0;
  Operand a = makeSource(builder, spec, spec.a, in, next, emitted.owners);
  Operand b = makeSource(builder, spec, spec.b, in, next, emitted.owners);
  RegisterRef dst =
      spec.placement == Placement::destinationIsA ? a.reg() : b.reg();
  (builder.*spec.op.op)(dst, a, b);
  emitted.value = dst;
  return emitted;
}

// Places a in a range that also holds the destination, offset by one register.
Emitted emitOverlapping(IRBuilder &builder, const Case64 &spec,
                        const std::vector<RegisterRef> &in) {
  Emitted emitted{{}, RegisterRef{}};
  RegisterRange range =
      allocateKind(builder, spec.aKind == lane ? vgpr : sgpr, 3);
  bool lowIsAHigh = spec.placement == Placement::lowIsAHigh;
  RegisterRef a = range.ref(lowIsAHigh ? 0 : 1, 2);
  RegisterRef dst = range.ref(lowIsAHigh ? 1 : 0, 2);
  builder.movb32(a.slice(0), in[0]);
  builder.movb32(a.slice(1), in[1]);
  size_t next = 2;
  Operand b = makeSource(builder, spec, spec.b, in, next, emitted.owners);
  (builder.*spec.op.op)(dst, a, b);
  emitted.owners.push_back(std::move(range));
  emitted.value = dst;
  return emitted;
}

Emitted emitCase(IRBuilder &builder, const Case64 &spec,
                 const std::vector<RegisterRef> &in) {
  switch (spec.placement) {
  case Placement::separate:
  case Placement::sameOperand:
    return emitSeparate(builder, spec, in);
  case Placement::destinationIsA:
  case Placement::destinationIsB:
    return emitInPlace(builder, spec, in);
  case Placement::lowIsAHigh:
  case Placement::highIsALow:
    return emitOverlapping(builder, spec, in);
  }
  return emitSeparate(builder, spec, in);
}

// A result is one lane of a vgpr result or one workgroup of an sgpr result.
size_t inputIndex(const Case64 &spec, InputKind kind, size_t result) {
  bool perResult = spec.dstKind == sgpr || kind == lane;
  return perResult ? result : result / arithWorkgroupSize;
}

uint64_t operandValue(const Case64 &spec, const std::optional<int32_t> &imm,
                      InputKind kind, const Values &values, size_t result) {
  if (imm)
    return static_cast<uint64_t>(int64_t{*imm});
  return values[inputIndex(spec, kind, result)];
}

Values expectedValues(const Case64 &spec, const Data64 &data, size_t results) {
  Values expected;
  for (size_t i = 0; i < results; ++i) {
    uint64_t a = operandValue(spec, spec.a, spec.aKind, data.a, i);
    bool same = spec.placement == Placement::sameOperand;
    uint64_t b = same ? a : operandValue(spec, spec.b, spec.bKind, data.b, i);
    expected.push_back(spec.op.host(a, b));
  }
  return expected;
}

struct KernelInputs {
  std::vector<InputKind> kinds;
  std::vector<Words> words;
};

void addPairInputs(KernelInputs &inputs, InputKind kind, const Values &values,
                   size_t laneCount, size_t groupCount) {
  size_t count = kind == lane ? laneCount : groupCount;
  for (const Words &words : {lowWords(values), highWords(values)}) {
    inputs.kinds.push_back(kind);
    inputs.words.emplace_back(words.begin(), words.begin() + count);
  }
}

KernelInputs makeKernelInputs(const Case64 &spec, const Data64 &data,
                              size_t groupCount) {
  size_t laneCount = data.a.size();
  KernelInputs inputs;
  if (!spec.a)
    addPairInputs(inputs, spec.aKind, data.a, laneCount, groupCount);
  if (!spec.b && spec.placement != Placement::sameOperand)
    addPairInputs(inputs, spec.bKind, data.b, laneCount, groupCount);
  return inputs;
}

// Runs the op and checks it, through two runs that each store one half.
void expectCase(const Case64 &spec, const Data64 &data) {
  size_t lanes = data.a.size();
  size_t groups = spec.dstKind == sgpr ? lanes : lanes / arithWorkgroupSize;
  size_t results = spec.dstKind == sgpr ? groups : lanes;
  KernelInputs inputs = makeKernelInputs(spec, data, groups);
  Values expected = expectedValues(spec, data, results);
  for (int half = 0; half < 2; ++half) {
    SCOPED_TRACE(std::string(spec.op.name) +
                 (half == 0 ? " low half" : " high half"));
    ArithKernelSpec kernel{inputs.kinds, spec.dstKind,
                           [spec, half](IRBuilder &builder, RegisterRef dst,
                                        const std::vector<RegisterRef> &in) {
                             Emitted emitted = emitCase(builder, spec, in);
                             builder.movb32(dst, emitted.value.slice(half));
                           }};
    Words words = half == 0 ? lowWords(expected) : highWords(expected);
    expectArithResults(kernel, static_cast<int>(groups), inputs.words, words);
  }
}

class Arith64Device : public testing::Test {
protected:
  void SetUp() override {
    if (!hasHipDevice())
      GTEST_SKIP() << "no HIP device";
  }
};

Case64 makeCase(const OpCase &op, RegisterKind dstKind, InputKind aKind,
                InputKind bKind) {
  return {op, dstKind, aKind, bKind};
}

TEST_F(Arith64Device, ValuLaneOperands) {
  Data64 data = makeData(4096);
  for (const OpCase &op : opCases)
    expectCase(makeCase(op, vgpr, lane, lane), data);
}

TEST_F(Arith64Device, SaluUniformOperands) {
  Data64 data = makeData(256);
  for (const OpCase &op : opCases)
    expectCase(makeCase(op, sgpr, uniform, uniform), data);
}

TEST_F(Arith64Device, ScalarsIntoVgpr) {
  Data64 data = makeData(256);
  for (const OpCase &op : opCases)
    expectCase(makeCase(op, vgpr, uniform, uniform), data);
}

TEST_F(Arith64Device, MixedScalarAndVector) {
  Data64 data = makeData(4096);
  for (const OpCase &op : opCases) {
    expectCase(makeCase(op, vgpr, uniform, lane), data);
    expectCase(makeCase(op, vgpr, lane, uniform), data);
  }
}

TEST_F(Arith64Device, MisalignedPairs) {
  Data64 data = makeData(1024);
  for (const OpCase &op : opCases) {
    for (bool dstMisaligned : {false, true}) {
      for (bool operandsMisaligned : {false, true}) {
        SCOPED_TRACE(
            std::string(dstMisaligned ? "misaligned dst, " : "aligned dst, ") +
            (operandsMisaligned ? "misaligned operands" : "aligned operands"));
        for (Case64 spec : {makeCase(op, vgpr, lane, lane),
                            makeCase(op, sgpr, uniform, uniform)}) {
          spec.dstMisaligned = dstMisaligned;
          spec.operandsMisaligned = operandsMisaligned;
          expectCase(spec, data);
        }
      }
    }
  }
}

TEST_F(Arith64Device, ImmediateOperands) {
  Data64 data = makeData(0);
  for (const OpCase &op : opCases) {
    for (int32_t immediate : immediates) {
      SCOPED_TRACE(std::string(op.name) + " immediate " +
                   std::to_string(immediate));
      for (Case64 spec : {makeCase(op, vgpr, lane, lane),
                          makeCase(op, sgpr, uniform, uniform)}) {
        Case64 immediateFirst = spec;
        immediateFirst.a = immediate;
        expectCase(immediateFirst, data);
        spec.b = immediate;
        expectCase(spec, data);
      }
      Case64 scalarsIntoVgpr = makeCase(op, vgpr, uniform, uniform);
      scalarsIntoVgpr.b = immediate;
      expectCase(scalarsIntoVgpr, data);
    }
  }
}

TEST_F(Arith64Device, DestinationAliasing) {
  Data64 data = makeData(256);
  for (const OpCase &op : opCases) {
    for (Placement placement :
         {Placement::destinationIsA, Placement::destinationIsB,
          Placement::lowIsAHigh, Placement::highIsALow,
          Placement::sameOperand}) {
      SCOPED_TRACE(std::string(op.name) + " placement " +
                   std::to_string(static_cast<int>(placement)));
      for (Case64 spec : {makeCase(op, vgpr, lane, lane),
                          makeCase(op, sgpr, uniform, uniform)}) {
        spec.placement = placement;
        expectCase(spec, data);
      }
    }
  }
}

// Operand placements, checked by the assembler alone so no device is needed.
struct NamedOperand {
  std::string name;
  Operand operand;
};

// The registers one operand can live in, aligned or at an odd index.
struct PlacementRegisters {
  RegisterRange vector;
  RegisterRange oddVector;
  RegisterRange scalar;
  RegisterRange oddScalar;
};

PlacementRegisters allocatePlacementRegisters(IRBuilder &builder) {
  return {builder.allocateVgpr(2, 2), builder.allocateVgpr(3, 2),
          builder.allocateSgpr(2, 2), builder.allocateSgpr(3, 2)};
}

std::vector<NamedOperand> placementsOf(const PlacementRegisters &regs) {
  return {{"vgpr pair", regs.vector},
          {"odd vgpr pair", regs.oddVector.ref(1, 2)},
          {"sgpr pair", regs.scalar},
          {"odd sgpr pair", regs.oddScalar.ref(1, 2)},
          {"inline -5", -5},
          {"literal -100", -100},
          {"literal 0x12345678", 0x12345678},
          {"INT32_MIN", INT32_MIN}};
}

// Emits every placement combination of one op into a single kernel.
void expectPlacementsAssemble(const OpCase &op, RegisterKind dstKind) {
  Target target = Target::get("gfx950");
  Kernel kernel(target, "codegen_placement", arithWorkgroupSize);
  IRBuilder &builder = kernel.builder();
  PlacementRegisters aRegs = allocatePlacementRegisters(builder);
  PlacementRegisters bRegs = allocatePlacementRegisters(builder);
  RegisterRange dst =
      dstKind == vgpr ? builder.allocateVgpr(2, 2) : builder.allocateSgpr(2, 2);
  std::vector<NamedOperand> bPlacements = placementsOf(bRegs);
  bPlacements.push_back({"same sgpr pair as a", aRegs.scalar});
  for (const NamedOperand &a : placementsOf(aRegs)) {
    for (const NamedOperand &b : bPlacements) {
      // A scalar destination cannot take a vector operand.
      if (dstKind == sgpr &&
          (a.operand.isRegister(vgpr) || b.operand.isRegister(vgpr)))
        continue;
      (builder.*op.op)(dst.ref(), a.operand, b.operand);
    }
  }
  builder.createEndProgram();

  std::string assembly = kernel.toAssembly();
  SCOPED_TRACE(std::string(op.name) +
               (dstKind == vgpr ? " to vgpr pair" : " to sgpr pair") + "\n" +
               assembly);
  EXPECT_NO_THROW(assembleToHSACO(assembly, target.isaName()));
}

TEST(Arith64Assembly, AllOperandPlacementsAssemble) {
  for (const OpCase &op : opCases) {
    expectPlacementsAssemble(op, vgpr);
    expectPlacementsAssemble(op, sgpr);
  }
}

} // namespace

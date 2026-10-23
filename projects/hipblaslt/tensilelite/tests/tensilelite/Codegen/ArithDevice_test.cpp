// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tensilelite/API/Compiler.hpp>
#include <tensilelite/Codegen/IRBuilder.hpp>
#include <tensilelite/Codegen/Kernel.hpp>

#include "ArithDeviceTest.hpp"
#include "HipTestSupport.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace {

using namespace rocm::tensilelite;
using namespace rocm::tensilelite::codegen;
using namespace rocm::tensilelite::codegen::test;

using BuilderOp = void (IRBuilder::*)(RegisterRef, Operand, Operand);

struct OpCase {
  const char *name;
  BuilderOp op;
  uint32_t (*host)(uint32_t, uint32_t);
};

const OpCase opCases[] = {
    {"addi32", &IRBuilder::addi32,
     [](uint32_t a, uint32_t b) { return a + b; }},
    {"subi32", &IRBuilder::subi32,
     [](uint32_t a, uint32_t b) { return a - b; }},
    {"muli32", &IRBuilder::muli32,
     [](uint32_t a, uint32_t b) { return a * b; }},
};

// Immediates that cover inline constants, their edges, literals and wrapping.
const int32_t immediates[] = {0,   1,   -1,        64,        65,
                              -16, -17, INT32_MIN, INT32_MAX, 0x12345678};

class ArithDevice : public testing::Test {
protected:
  void SetUp() override {
    if (!hasHipDevice())
      GTEST_SKIP() << "no HIP device";
  }
};

int groupCount(const std::vector<uint32_t> &laneWords) {
  return static_cast<int>(laneWords.size()) / arithWorkgroupSize;
}

// Repeats each workgroup's word for every lane of the workgroup.
std::vector<uint32_t> perLane(const std::vector<uint32_t> &perGroup) {
  std::vector<uint32_t> lanes;
  for (uint32_t word : perGroup)
    lanes.insert(lanes.end(), arithWorkgroupSize, word);
  return lanes;
}

ArithKernelSpec binarySpec(BuilderOp op, InputKind aKind, InputKind bKind,
                           RegisterKind result) {
  return {{aKind, bKind},
          result,
          [op](IRBuilder &builder, RegisterRef dst,
               const std::vector<RegisterRef> &inputs) {
            (builder.*op)(dst, inputs[0], inputs[1]);
          }};
}

// Computes dst = input (kind) op immediate, or immediate op input.
ArithKernelSpec immediateSpec(BuilderOp op, InputKind kind, RegisterKind result,
                              int32_t immediate, bool immediateFirst) {
  return {
      {kind},
      result,
      [op, immediate, immediateFirst](IRBuilder &builder, RegisterRef dst,
                                      const std::vector<RegisterRef> &inputs) {
        if (immediateFirst)
          (builder.*op)(dst, immediate, inputs[0]);
        else
          (builder.*op)(dst, inputs[0], immediate);
      }};
}

std::vector<uint32_t> applyImmediate(const OpCase &op,
                                     const std::vector<uint32_t> &words,
                                     int32_t immediate, bool immediateFirst) {
  uint32_t value = static_cast<uint32_t>(immediate);
  std::vector<uint32_t> out;
  for (uint32_t word : words)
    out.push_back(immediateFirst ? op.host(value, word) : op.host(word, value));
  return out;
}

TEST_F(ArithDevice, ValuLaneOperands) {
  WordPairs pairs = makeArithPairs(4096);
  for (const OpCase &op : opCases) {
    SCOPED_TRACE(op.name);
    std::vector<uint32_t> expected;
    for (size_t i = 0; i < pairs.a.size(); ++i)
      expected.push_back(op.host(pairs.a[i], pairs.b[i]));
    expectArithResults(
        binarySpec(op.op, InputKind::lane, InputKind::lane, RegisterKind::vgpr),
        groupCount(pairs.a), {pairs.a, pairs.b}, expected);
  }
}

TEST_F(ArithDevice, SaluUniformOperands) {
  WordPairs pairs = makeArithPairs(256);
  for (const OpCase &op : opCases) {
    SCOPED_TRACE(op.name);
    std::vector<uint32_t> expected;
    for (size_t i = 0; i < pairs.a.size(); ++i)
      expected.push_back(op.host(pairs.a[i], pairs.b[i]));
    expectArithResults(binarySpec(op.op, InputKind::uniform, InputKind::uniform,
                                  RegisterKind::sgpr),
                       static_cast<int>(pairs.a.size()), {pairs.a, pairs.b},
                       expected);
  }
}

TEST_F(ArithDevice, ScalarsIntoVgpr) {
  WordPairs pairs = makeArithPairs(256);
  for (const OpCase &op : opCases) {
    SCOPED_TRACE(op.name);
    std::vector<uint32_t> perGroup;
    for (size_t i = 0; i < pairs.a.size(); ++i)
      perGroup.push_back(op.host(pairs.a[i], pairs.b[i]));
    expectArithResults(binarySpec(op.op, InputKind::uniform, InputKind::uniform,
                                  RegisterKind::vgpr),
                       static_cast<int>(pairs.a.size()), {pairs.a, pairs.b},
                       perLane(perGroup));
  }
}

TEST_F(ArithDevice, MixedScalarAndVector) {
  WordPairs pairs = makeArithPairs(4096);
  int groups = groupCount(pairs.a);
  std::vector<uint32_t> edges = arithEdgeValues();
  std::vector<uint32_t> uniform;
  for (int g = 0; g < groups; ++g)
    uniform.push_back(edges[g % edges.size()]);
  for (const OpCase &op : opCases) {
    SCOPED_TRACE(op.name);
    std::vector<uint32_t> uniformFirst;
    std::vector<uint32_t> uniformSecond;
    for (size_t i = 0; i < pairs.b.size(); ++i) {
      uint32_t scalar = uniform[i / arithWorkgroupSize];
      uniformFirst.push_back(op.host(scalar, pairs.b[i]));
      uniformSecond.push_back(op.host(pairs.b[i], scalar));
    }
    expectArithResults(binarySpec(op.op, InputKind::uniform, InputKind::lane,
                                  RegisterKind::vgpr),
                       groups, {uniform, pairs.b}, uniformFirst);
    expectArithResults(binarySpec(op.op, InputKind::lane, InputKind::uniform,
                                  RegisterKind::vgpr),
                       groups, {pairs.b, uniform}, uniformSecond);
  }
}

TEST_F(ArithDevice, ImmediateOperands) {
  WordPairs pairs = makeArithPairs(0);
  int groups = groupCount(pairs.a);
  int scalarGroups = static_cast<int>(pairs.a.size());
  for (const OpCase &op : opCases) {
    for (int32_t immediate : immediates) {
      SCOPED_TRACE(std::string(op.name) + " immediate " +
                   std::to_string(immediate));
      expectArithResults(immediateSpec(op.op, InputKind::lane,
                                       RegisterKind::vgpr, immediate, false),
                         groups, {pairs.a},
                         applyImmediate(op, pairs.a, immediate, false));
      expectArithResults(immediateSpec(op.op, InputKind::lane,
                                       RegisterKind::vgpr, immediate, true),
                         groups, {pairs.b},
                         applyImmediate(op, pairs.b, immediate, true));
      expectArithResults(immediateSpec(op.op, InputKind::uniform,
                                       RegisterKind::sgpr, immediate, false),
                         scalarGroups, {pairs.a},
                         applyImmediate(op, pairs.a, immediate, false));
      expectArithResults(
          immediateSpec(op.op, InputKind::uniform, RegisterKind::vgpr,
                        immediate, false),
          scalarGroups, {pairs.a},
          perLane(applyImmediate(op, pairs.a, immediate, false)));
    }
  }
}

// Each emitter writes into an input register and then moves it to dst.
TEST_F(ArithDevice, DestinationAliasesOperand) {
  WordPairs pairs = makeArithPairs(256);
  int groups = groupCount(pairs.a);
  int scalarGroups = static_cast<int>(pairs.a.size());
  std::vector<uint32_t> expected;

  SCOPED_TRACE("vgpr subi32 into second input");
  for (size_t i = 0; i < pairs.a.size(); ++i)
    expected.push_back(pairs.a[i] - pairs.b[i]);
  expectArithResults({{InputKind::lane, InputKind::lane},
                      RegisterKind::vgpr,
                      [](IRBuilder &builder, RegisterRef dst,
                         const std::vector<RegisterRef> &inputs) {
                        builder.subi32(inputs[1], inputs[0], inputs[1]);
                        builder.movb32(dst, inputs[1]);
                      }},
                     groups, {pairs.a, pairs.b}, expected);

  SCOPED_TRACE("vgpr muli32 into first input");
  expected.clear();
  for (uint32_t a : pairs.a)
    expected.push_back(a * 0x12345678u);
  expectArithResults({{InputKind::lane},
                      RegisterKind::vgpr,
                      [](IRBuilder &builder, RegisterRef dst,
                         const std::vector<RegisterRef> &inputs) {
                        builder.muli32(inputs[0], inputs[0], 0x12345678);
                        builder.movb32(dst, inputs[0]);
                      }},
                     groups, {pairs.a}, expected);

  SCOPED_TRACE("vgpr addi32 of scalar and lane into lane");
  std::vector<uint32_t> uniform(pairs.a.begin(), pairs.a.begin() + groups);
  expected.clear();
  for (size_t i = 0; i < pairs.b.size(); ++i)
    expected.push_back(uniform[i / arithWorkgroupSize] + pairs.b[i]);
  expectArithResults({{InputKind::uniform, InputKind::lane},
                      RegisterKind::vgpr,
                      [](IRBuilder &builder, RegisterRef dst,
                         const std::vector<RegisterRef> &inputs) {
                        builder.addi32(inputs[1], inputs[0], inputs[1]);
                        builder.movb32(dst, inputs[1]);
                      }},
                     groups, {uniform, pairs.b}, expected);

  SCOPED_TRACE("sgpr subi32 into first input");
  expected.clear();
  for (size_t i = 0; i < pairs.a.size(); ++i)
    expected.push_back(pairs.a[i] - pairs.b[i]);
  expectArithResults({{InputKind::uniform, InputKind::uniform},
                      RegisterKind::sgpr,
                      [](IRBuilder &builder, RegisterRef dst,
                         const std::vector<RegisterRef> &inputs) {
                        builder.subi32(inputs[0], inputs[0], inputs[1]);
                        builder.movb32(dst, inputs[0]);
                      }},
                     scalarGroups, {pairs.a, pairs.b}, expected);
}

// Operand placements, checked by the assembler alone so no device is needed.
enum class APlacement { vgpr, sgpr, inlineConstant, literal };
enum class BPlacement {
  vgpr,
  otherSgpr,
  sameSgpr,
  inlineConstant,
  literal,
  sameLiteral,
  negativeLiteral,
  minInt32,
};

constexpr int32_t inlineA = 7;
constexpr int32_t inlineB = -5;
constexpr int32_t literalA = 0x12345;
constexpr int32_t literalB = 0x54321;
constexpr int32_t negativeLiteralB = -100;

const char *placementName(APlacement placement) {
  switch (placement) {
  case APlacement::vgpr:
    return "vgpr";
  case APlacement::sgpr:
    return "sgpr";
  case APlacement::inlineConstant:
    return "inline 7";
  case APlacement::literal:
    return "literal 0x12345";
  }
  return "";
}

const char *placementName(BPlacement placement) {
  switch (placement) {
  case BPlacement::vgpr:
    return "vgpr";
  case BPlacement::otherSgpr:
    return "other sgpr";
  case BPlacement::sameSgpr:
    return "same sgpr as a";
  case BPlacement::inlineConstant:
    return "inline -5";
  case BPlacement::literal:
    return "literal 0x54321";
  case BPlacement::sameLiteral:
    return "same literal as a";
  case BPlacement::negativeLiteral:
    return "literal -100";
  case BPlacement::minInt32:
    return "INT32_MIN";
  }
  return "";
}

struct PlacementRegisters {
  RegisterRange vectorA;
  RegisterRange vectorB;
  RegisterRange scalarA;
  RegisterRange scalarB;
};

Operand pickA(APlacement placement, const PlacementRegisters &regs) {
  switch (placement) {
  case APlacement::vgpr:
    return regs.vectorA;
  case APlacement::sgpr:
    return regs.scalarA;
  case APlacement::inlineConstant:
    return inlineA;
  case APlacement::literal:
    return literalA;
  }
  return 0;
}

Operand pickB(BPlacement placement, const PlacementRegisters &regs) {
  switch (placement) {
  case BPlacement::vgpr:
    return regs.vectorB;
  case BPlacement::otherSgpr:
    return regs.scalarB;
  case BPlacement::sameSgpr:
    return regs.scalarA;
  case BPlacement::inlineConstant:
    return inlineB;
  case BPlacement::literal:
    return literalB;
  case BPlacement::sameLiteral:
    return literalA;
  case BPlacement::negativeLiteral:
    return negativeLiteralB;
  case BPlacement::minInt32:
    return INT32_MIN;
  }
  return 0;
}

bool isCompatible(APlacement a, BPlacement b) {
  if (b == BPlacement::sameSgpr)
    return a == APlacement::sgpr;
  if (b == BPlacement::sameLiteral)
    return a == APlacement::literal;
  return true;
}

void expectPlacementAssembles(const OpCase &op, APlacement a, BPlacement b,
                              RegisterKind dstKind) {
  Target target = Target::get("gfx950");
  Kernel kernel(target, "codegen_placement", arithWorkgroupSize);
  IRBuilder &builder = kernel.builder();
  PlacementRegisters regs{builder.allocateVgpr(1), builder.allocateVgpr(1),
                          builder.allocateSgpr(1), builder.allocateSgpr(1)};
  RegisterRange dst = dstKind == RegisterKind::vgpr ? builder.allocateVgpr(1)
                                                    : builder.allocateSgpr(1);
  (builder.*op.op)(dst.ref(), pickA(a, regs), pickB(b, regs));
  builder.createEndProgram();

  std::string assembly = kernel.toAssembly();
  SCOPED_TRACE(std::string(op.name) + " a=" + placementName(a) +
               " b=" + placementName(b) +
               " dst=" + (dstKind == RegisterKind::vgpr ? "vgpr" : "sgpr") +
               "\n" + assembly);
  EXPECT_NO_THROW(assembleToHSACO(assembly, target.isaName()));
}

TEST(ArithAssembly, AllOperandPlacementsAssemble) {
  for (const OpCase &op : opCases)
    for (APlacement a : {APlacement::vgpr, APlacement::sgpr,
                         APlacement::inlineConstant, APlacement::literal})
      for (BPlacement b : {BPlacement::vgpr, BPlacement::otherSgpr,
                           BPlacement::sameSgpr, BPlacement::inlineConstant,
                           BPlacement::literal, BPlacement::sameLiteral,
                           BPlacement::negativeLiteral, BPlacement::minInt32}) {
        if (!isCompatible(a, b))
          continue;
        expectPlacementAssembles(op, a, b, RegisterKind::vgpr);
        // A scalar destination cannot take a vector operand.
        if (a == APlacement::vgpr || b == BPlacement::vgpr)
          continue;
        expectPlacementAssembles(op, a, b, RegisterKind::sgpr);
      }
}

} // namespace

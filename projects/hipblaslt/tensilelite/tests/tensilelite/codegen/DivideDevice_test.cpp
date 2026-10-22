// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tensilelite/Compiler.hpp>
#include <tensilelite/codegen/IRBuilder.hpp>
#include <tensilelite/codegen/Kernel.hpp>

#include "ArithDeviceTest.hpp"
#include "HipTestSupport.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace rocm::tensilelite;
using namespace rocm::tensilelite::codegen;
using namespace rocm::tensilelite::codegen::test;

constexpr uint32_t randomSeed = 20240607;
// The fast path is exact for magnitudes below this bound, LLVM's signed 24-bit
// criterion.
constexpr int64_t fastBound = int64_t{1} << 23;
constexpr int exactRandomCount = 131072;
constexpr int fastBoundaryCount = 262144;
constexpr int fastRandomCount = 1048576;

class DivDevice : public testing::Test {
protected:
  void SetUp() override {
    if (!hasHipDevice())
      GTEST_SKIP() << "no HIP device";
  }
};

// The quotient truncates toward zero and INT32_MIN / -1 wraps.
uint32_t hostDiv(uint32_t a, uint32_t b) {
  int32_t x = static_cast<int32_t>(a);
  int32_t y = static_cast<int32_t>(b);
  if (x == INT32_MIN && y == -1)
    return a;
  return static_cast<uint32_t>(x / y);
}

struct Pairs {
  std::vector<uint32_t> a;
  std::vector<uint32_t> b;
};

// A zero divisor is replaced by 1.
void addPair(Pairs &pairs, int64_t a, int64_t b) {
  pairs.a.push_back(static_cast<uint32_t>(a));
  pairs.b.push_back(b == 0 ? 1u : static_cast<uint32_t>(b));
}

// A value with 0 to maxBits - 1 low bits kept, so that every magnitude occurs.
uint32_t randomMagnitude(std::mt19937 &engine, int maxBits) {
  uint32_t mask = maxBits == 32 ? ~0u : (1u << maxBits) - 1;
  return (engine() & mask) >> (engine() % maxBits);
}

int64_t withRandomSign(std::mt19937 &engine, int64_t magnitude) {
  return (engine() & 1) != 0 ? -magnitude : magnitude;
}

// Pads with random pairs to whole workgroups of lanes.
void padToWorkgroups(Pairs &pairs, std::mt19937 &engine, int maxBits) {
  while (pairs.a.size() % arithWorkgroupSize != 0)
    addPair(pairs, withRandomSign(engine, randomMagnitude(engine, maxBits)),
            withRandomSign(engine, randomMagnitude(engine, maxBits)));
}

std::vector<int64_t> exactEdges() {
  std::vector<int64_t> edges = {0,          1,
                                -1,         2,
                                -2,         3,
                                -3,         7,
                                -7,         INT32_MIN,
                                INT32_MAX,  INT32_MIN + 1,
                                0x12345678, static_cast<int32_t>(0x9E3779B9),
                                0x7FFFFFFE, -0x7FFFFFFE};
  for (int bit = 2; bit <= 30; ++bit) {
    edges.push_back(int64_t{1} << bit);
    edges.push_back(-(int64_t{1} << bit));
  }
  for (int64_t delta : {-1, 0, 1}) {
    edges.push_back(fastBound + delta);
    edges.push_back(-(fastBound + delta));
  }
  for (int64_t delta : {-1, 0, 1}) {
    edges.push_back((int64_t{1} << 24) + delta);
    edges.push_back(-((int64_t{1} << 24) + delta));
  }
  return edges;
}

// Divisors near |a|, and some larger than |a|.
void addNeighborDivisors(Pairs &pairs, int64_t a) {
  for (int64_t b : {a - 1, a, a + 1, -a, a / 2, a * 2 + 1, -(a * 2 + 1)}) {
    if (b < INT32_MIN || b > INT32_MAX)
      continue;
    addPair(pairs, a, b);
  }
}

void addExactEdgePairs(Pairs &pairs) {
  std::vector<int64_t> edges = exactEdges();
  for (int64_t a : edges) {
    for (int64_t b : edges)
      addPair(pairs, a, b);
    addNeighborDivisors(pairs, a);
  }
}

// Random 32-bit pairs, then a random with a small divisor, then both shifted.
void addExactRandomPairs(Pairs &pairs, std::mt19937 &engine, int count) {
  for (int i = 0; i < count; ++i) {
    switch (i % 3) {
    case 0:
      addPair(pairs, static_cast<int32_t>(engine()),
              static_cast<int32_t>(engine()));
      break;
    case 1:
      addPair(pairs, static_cast<int32_t>(engine()),
              withRandomSign(engine, 1 + engine() % 1024));
      break;
    default:
      addPair(pairs, withRandomSign(engine, randomMagnitude(engine, 32)),
              withRandomSign(engine, randomMagnitude(engine, 32)));
      break;
    }
  }
}

const Pairs &exactPairs() {
  static const Pairs pairs = [] {
    Pairs result;
    std::mt19937 engine(randomSeed);
    addExactEdgePairs(result);
    addExactRandomPairs(result, engine, exactRandomCount);
    padToWorkgroups(result, engine, 32);
    return result;
  }();
  return pairs;
}

std::vector<int64_t> fastEdges() {
  std::vector<int64_t> edges = {0,
                                1,
                                2,
                                3,
                                fastBound - 1,
                                fastBound - 2,
                                1 << 22,
                                (1 << 22) + 1,
                                (1 << 22) - 1,
                                4096,
                                0x6BCDEF};
  std::vector<int64_t> signedEdges;
  for (int64_t edge : edges) {
    signedEdges.push_back(edge);
    signedEdges.push_back(-edge);
  }
  return signedEdges;
}

void addFastEdgePairs(Pairs &pairs) {
  std::vector<int64_t> edges = fastEdges();
  for (int64_t a : edges)
    for (int64_t b : edges)
      addPair(pairs, a, b);
  for (int64_t a : std::vector<int64_t>{fastBound - 1, fastBound - 2,
                                        (1 << 22) + 1, 0x6BCDEF})
    for (int64_t b = 1; b <= 4096; ++b)
      for (int64_t signedA : {a, -a})
        for (int64_t signedB : {b, -b})
          addPair(pairs, signedA, signedB);
}

// Pairs a = k * b + d for d in {-1, 0, 1}, where the exact quotient changes.
void addBoundaryPairs(Pairs &pairs, std::mt19937 &engine, int count) {
  for (int i = 0; i < count; ++i) {
    int64_t b = 1 + randomMagnitude(engine, 23) % (fastBound - 1);
    int64_t k = engine() % ((fastBound - 2) / b + 1);
    int64_t delta = static_cast<int64_t>(engine() % 3) - 1;
    int64_t a = k * b + delta;
    if (a < 0 || a >= fastBound)
      a = k * b;
    addPair(pairs, withRandomSign(engine, a), withRandomSign(engine, b));
  }
}

// Uniform 23-bit magnitudes, alternating with shifted ones.
void addFastRandomPairs(Pairs &pairs, std::mt19937 &engine, int count) {
  for (int i = 0; i < count; ++i) {
    bool shifted = (i & 1) != 0;
    uint32_t a = shifted ? randomMagnitude(engine, 23) : engine() & 0x7FFFFF;
    uint32_t b = shifted ? randomMagnitude(engine, 23) : engine() & 0x7FFFFF;
    addPair(pairs, withRandomSign(engine, a), withRandomSign(engine, b));
  }
}

const Pairs &fastPairs() {
  static const Pairs pairs = [] {
    Pairs result;
    std::mt19937 engine(randomSeed + 1);
    addFastEdgePairs(result);
    addBoundaryPairs(result, engine, fastBoundaryCount);
    addFastRandomPairs(result, engine, fastRandomCount);
    padToWorkgroups(result, engine, 23);
    return result;
  }();
  return pairs;
}

const Pairs &pairsFor(bool useIEEE) {
  return useIEEE ? exactPairs() : fastPairs();
}

const char *pathName(bool useIEEE) { return useIEEE ? "exact" : "fast"; }

// Every stride-th pair, so that a small sample still spans the whole set.
Pairs sample(const Pairs &pairs, size_t count) {
  Pairs result;
  size_t stride = std::max<size_t>(1, pairs.a.size() / count);
  for (size_t i = 0; i < pairs.a.size() && result.a.size() < count;
       i += stride) {
    result.a.push_back(pairs.a[i]);
    result.b.push_back(pairs.b[i]);
  }
  return result;
}

std::vector<uint32_t> quotients(const Pairs &pairs) {
  std::vector<uint32_t> expected;
  for (size_t i = 0; i < pairs.a.size(); ++i)
    expected.push_back(hostDiv(pairs.a[i], pairs.b[i]));
  return expected;
}

int groupCount(const std::vector<uint32_t> &words) {
  return static_cast<int>(words.size());
}

int laneGroupCount(const std::vector<uint32_t> &laneWords) {
  return static_cast<int>(laneWords.size()) / arithWorkgroupSize;
}

// Repeats each workgroup's word for every lane of the workgroup.
std::vector<uint32_t> perLane(const std::vector<uint32_t> &perGroup) {
  std::vector<uint32_t> lanes;
  for (uint32_t word : perGroup)
    lanes.insert(lanes.end(), arithWorkgroupSize, word);
  return lanes;
}

ArithKernelSpec divSpec(InputKind aKind, InputKind bKind, RegisterKind result,
                        bool useIEEE) {
  return {{aKind, bKind},
          result,
          [useIEEE](IRBuilder &builder, RegisterRef dst,
                    const std::vector<RegisterRef> &inputs) {
            builder.divsi32(dst, inputs[0], inputs[1], useIEEE);
          }};
}

TEST_F(DivDevice, ValuLaneOperands) {
  for (bool useIEEE : {true, false}) {
    SCOPED_TRACE(pathName(useIEEE));
    const Pairs &pairs = pairsFor(useIEEE);
    expectArithResults(
        divSpec(InputKind::lane, InputKind::lane, RegisterKind::vgpr, useIEEE),
        laneGroupCount(pairs.a), {pairs.a, pairs.b}, quotients(pairs));
  }
}

TEST_F(DivDevice, SaluUniformOperands) {
  for (bool useIEEE : {true, false}) {
    SCOPED_TRACE(pathName(useIEEE));
    Pairs pairs = sample(pairsFor(useIEEE), 32768);
    expectArithResults(divSpec(InputKind::uniform, InputKind::uniform,
                               RegisterKind::sgpr, useIEEE),
                       groupCount(pairs.a), {pairs.a, pairs.b},
                       quotients(pairs));
  }
}

TEST_F(DivDevice, ScalarsIntoVgpr) {
  for (bool useIEEE : {true, false}) {
    SCOPED_TRACE(pathName(useIEEE));
    Pairs pairs = sample(pairsFor(useIEEE), 16384);
    expectArithResults(divSpec(InputKind::uniform, InputKind::uniform,
                               RegisterKind::vgpr, useIEEE),
                       groupCount(pairs.a), {pairs.a, pairs.b},
                       perLane(quotients(pairs)));
  }
}

// The first lane of each workgroup supplies the uniform operand.
std::vector<uint32_t> firstLanes(const std::vector<uint32_t> &laneWords) {
  std::vector<uint32_t> uniform;
  for (size_t i = 0; i < laneWords.size(); i += arithWorkgroupSize)
    uniform.push_back(laneWords[i]);
  return uniform;
}

TEST_F(DivDevice, MixedScalarAndVector) {
  for (bool useIEEE : {true, false}) {
    SCOPED_TRACE(pathName(useIEEE));
    Pairs pairs = sample(pairsFor(useIEEE), 65536);
    int groups = laneGroupCount(pairs.a);
    std::vector<uint32_t> uniformA = firstLanes(pairs.a);
    std::vector<uint32_t> uniformB = firstLanes(pairs.b);
    std::vector<uint32_t> uniformFirst;
    std::vector<uint32_t> uniformSecond;
    for (size_t i = 0; i < pairs.a.size(); ++i) {
      size_t group = i / arithWorkgroupSize;
      uniformFirst.push_back(hostDiv(uniformA[group], pairs.b[i]));
      uniformSecond.push_back(hostDiv(pairs.a[i], uniformB[group]));
    }
    expectArithResults(divSpec(InputKind::uniform, InputKind::lane,
                               RegisterKind::vgpr, useIEEE),
                       groups, {uniformA, pairs.b}, uniformFirst);
    expectArithResults(divSpec(InputKind::lane, InputKind::uniform,
                               RegisterKind::vgpr, useIEEE),
                       groups, {pairs.a, uniformB}, uniformSecond);
  }
}

std::vector<int64_t> immediatesFor(bool useIEEE) {
  std::vector<int64_t> immediates = {1,   -1, 2,  -2,  3,       7,
                                     -16, 64, 65, -17, 0x12345, -100000};
  if (useIEEE) {
    immediates.push_back(INT32_MIN);
    immediates.push_back(INT32_MAX);
    return immediates;
  }
  immediates.push_back(0x7FFFFF);
  immediates.push_back(-0x7FFFFF);
  return immediates;
}

// Computes dst = input / immediate, or immediate / input.
ArithKernelSpec immediateSpec(InputKind kind, RegisterKind result,
                              int64_t immediate, bool immediateIsDividend,
                              bool useIEEE) {
  return {{kind},
          result,
          [=](IRBuilder &builder, RegisterRef dst,
              const std::vector<RegisterRef> &inputs) {
            if (immediateIsDividend)
              builder.divsi32(dst, immediate, inputs[0], useIEEE);
            else
              builder.divsi32(dst, inputs[0], immediate, useIEEE);
          }};
}

std::vector<uint32_t> divideByImmediate(const std::vector<uint32_t> &words,
                                        int64_t immediate,
                                        bool immediateIsDividend) {
  uint32_t value = static_cast<uint32_t>(immediate);
  std::vector<uint32_t> out;
  for (uint32_t word : words)
    out.push_back(immediateIsDividend ? hostDiv(value, word)
                                      : hostDiv(word, value));
  return out;
}

// Runs the lane, uniform-into-sgpr and uniform-into-vgpr forms.
void expectImmediateForms(const std::vector<uint32_t> &words, int64_t immediate,
                          bool immediateIsDividend, bool useIEEE) {
  std::vector<uint32_t> expected =
      divideByImmediate(words, immediate, immediateIsDividend);
  expectArithResults(immediateSpec(InputKind::lane, RegisterKind::vgpr,
                                   immediate, immediateIsDividend, useIEEE),
                     laneGroupCount(words), {words}, expected);
  expectArithResults(immediateSpec(InputKind::uniform, RegisterKind::sgpr,
                                   immediate, immediateIsDividend, useIEEE),
                     groupCount(words), {words}, expected);
  expectArithResults(immediateSpec(InputKind::uniform, RegisterKind::vgpr,
                                   immediate, immediateIsDividend, useIEEE),
                     groupCount(words), {words}, perLane(expected));
}

TEST_F(DivDevice, ImmediateOperands) {
  for (bool useIEEE : {true, false}) {
    Pairs pairs = sample(pairsFor(useIEEE), 1024);
    for (int64_t immediate : immediatesFor(useIEEE)) {
      SCOPED_TRACE(std::string(pathName(useIEEE)) + " immediate " +
                   std::to_string(immediate));
      // The other operand comes from the set, so it is never zero.
      expectImmediateForms(pairs.a, immediate, false, useIEEE);
      expectImmediateForms(pairs.b, immediate, true, useIEEE);
    }
  }
}

// Each emitter writes into an input register and then moves it to dst.
TEST_F(DivDevice, DestinationAliasesOperand) {
  for (bool useIEEE : {true, false}) {
    SCOPED_TRACE(pathName(useIEEE));
    Pairs pairs = sample(pairsFor(useIEEE), 4096);
    std::vector<uint32_t> expected = quotients(pairs);
    for (int aliased : {0, 1}) {
      SCOPED_TRACE("alias input " + std::to_string(aliased));
      auto emit = [=](IRBuilder &builder, RegisterRef dst,
                      const std::vector<RegisterRef> &inputs) {
        builder.divsi32(inputs[aliased], inputs[0], inputs[1], useIEEE);
        builder.movb32(dst, inputs[aliased]);
      };
      expectArithResults(
          {{InputKind::lane, InputKind::lane}, RegisterKind::vgpr, emit},
          laneGroupCount(pairs.a), {pairs.a, pairs.b}, expected);
      expectArithResults(
          {{InputKind::uniform, InputKind::uniform}, RegisterKind::sgpr, emit},
          groupCount(pairs.a), {pairs.a, pairs.b}, expected);
    }
  }
}

// The quotient of a division by zero is unspecified, so the result is zeroed
// and only the completion of the kernel, without a fault or a hang, is checked.
TEST_F(DivDevice, DivisionByZeroCompletes) {
  for (bool useIEEE : {true, false}) {
    SCOPED_TRACE(pathName(useIEEE));
    Pairs pairs = sample(pairsFor(useIEEE), 4096);
    std::vector<uint32_t> zeros(pairs.a.size(), 0);
    expectArithResults({{InputKind::lane, InputKind::lane},
                        RegisterKind::vgpr,
                        [useIEEE](IRBuilder &builder, RegisterRef dst,
                                  const std::vector<RegisterRef> &inputs) {
                          RegisterRange tmp = builder.allocateVgpr(1);
                          builder.divsi32(tmp.ref(), inputs[0], inputs[1],
                                          useIEEE);
                          builder.muli32(dst, tmp, 0);
                        }},
                       laneGroupCount(pairs.a), {pairs.a, zeros}, zeros);
    expectArithResults({{InputKind::uniform, InputKind::uniform},
                        RegisterKind::sgpr,
                        [useIEEE](IRBuilder &builder, RegisterRef dst,
                                  const std::vector<RegisterRef> &inputs) {
                          RegisterRange tmp = builder.allocateSgpr(1);
                          builder.divsi32(tmp.ref(), inputs[0], inputs[1],
                                          useIEEE);
                          builder.muli32(dst, tmp, 0);
                        }},
                       groupCount(pairs.a), {pairs.a, zeros}, zeros);
  }
}

// These pairs are outside the fast-path precondition, where fa * rcp(fb) can
// round up to the next integer, so only the exact path is checked. They guard
// the bound.
TEST_F(DivDevice, ExactPathHandlesPairsAboveFastBound) {
  Pairs pairs;
  const int64_t above[][2] = {
      {16777214, 105},      {16777214, 16777215}, {-16777214, 105},
      {0x800001, 3},        {-0x800001, 3},       {0x800001, -3},
      {16777215, 16777214}, {16777215, 105},      {-16777215, 16777214},
      {0xFFFFFE, 7},        {16777217, 105},      {-16777217, 105},
      {16777217, 16777215}};
  for (const auto &pair : above)
    addPair(pairs, pair[0], pair[1]);
  std::mt19937 engine(randomSeed + 2);
  for (int i = 0; i < 4096; ++i)
    addPair(pairs, withRandomSign(engine, fastBound + engine() % fastBound),
            withRandomSign(engine, 1 + randomMagnitude(engine, 24)));
  padToWorkgroups(pairs, engine, 24);
  expectArithResults(
      divSpec(InputKind::lane, InputKind::lane, RegisterKind::vgpr, true),
      laneGroupCount(pairs.a), {pairs.a, pairs.b}, quotients(pairs));
  Pairs scalar = sample(pairs, 1024);
  expectArithResults(
      divSpec(InputKind::uniform, InputKind::uniform, RegisterKind::sgpr, true),
      groupCount(scalar.a), {scalar.a, scalar.b}, quotients(scalar));
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

constexpr int64_t inlineA = 7;
constexpr int64_t inlineB = -5;
constexpr int64_t literalA = 0x12345;
constexpr int64_t literalB = 0x54321;
constexpr int64_t negativeLiteralB = -100;

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

void expectPlacementAssembles(APlacement a, BPlacement b, RegisterKind dstKind,
                              bool useIEEE) {
  Target target = Target::get("gfx950");
  Kernel kernel(target, "codegen_div_placement", arithWorkgroupSize);
  IRBuilder &builder = kernel.builder();
  PlacementRegisters regs{builder.allocateVgpr(1), builder.allocateVgpr(1),
                          builder.allocateSgpr(1), builder.allocateSgpr(1)};
  RegisterRange dst = dstKind == RegisterKind::vgpr ? builder.allocateVgpr(1)
                                                    : builder.allocateSgpr(1);
  builder.divsi32(dst.ref(), pickA(a, regs), pickB(b, regs), useIEEE);
  builder.createEndProgram();

  std::string assembly = kernel.toAssembly();
  SCOPED_TRACE(std::string(pathName(useIEEE)) + " a=" + placementName(a) +
               " b=" + placementName(b) +
               " dst=" + (dstKind == RegisterKind::vgpr ? "vgpr" : "sgpr") +
               "\n" + assembly);
  EXPECT_NO_THROW(assembleToHSACO(assembly, target.isaName()));
}

TEST(DivAssembly, AllOperandPlacementsAssemble) {
  for (bool useIEEE : {true, false})
    for (APlacement a : {APlacement::vgpr, APlacement::sgpr,
                         APlacement::inlineConstant, APlacement::literal})
      for (BPlacement b : {BPlacement::vgpr, BPlacement::otherSgpr,
                           BPlacement::sameSgpr, BPlacement::inlineConstant,
                           BPlacement::literal, BPlacement::sameLiteral,
                           BPlacement::negativeLiteral, BPlacement::minInt32}) {
        if (!isCompatible(a, b))
          continue;
        expectPlacementAssembles(a, b, RegisterKind::vgpr, useIEEE);
        // A scalar destination cannot take a vector operand.
        if (a == APlacement::vgpr || b == BPlacement::vgpr)
          continue;
        expectPlacementAssembles(a, b, RegisterKind::sgpr, useIEEE);
      }
}

} // namespace

// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tensilelite/API/Compiler.hpp>
#include <tensilelite/Codegen/IRBuilder.hpp>
#include <tensilelite/Codegen/Kernel.hpp>

#include "TestSupport.hpp"

#include "code.hpp"
#include "container.hpp"
#include "instruction/branch.hpp"
#include "instruction/common.hpp"
#include "instruction/mem.hpp"

#include <gtest/gtest.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

using namespace rocm::tensilelite;
using namespace rocm::tensilelite::codegen;
using namespace rocm::tensilelite::codegen::test;

constexpr RegisterKind sgpr = RegisterKind::sgpr;
constexpr RegisterKind vgpr = RegisterKind::vgpr;
constexpr RegisterKind agpr = RegisterKind::agpr;
constexpr InputKind lane = InputKind::lane;
constexpr InputKind uniform = InputKind::uniform;

constexpr int32_t literalA = 0x12345;
constexpr int32_t literalB = 0x54321;
constexpr const char *literalAText = "74565";
constexpr const char *literalBText = "344865";
constexpr int32_t literal32 = 0x12345678;
constexpr const char *literal32Text = "305419896";
constexpr int randomPairs = 1024;

// A pair immediate is sign-extended to 64 bits.
constexpr int32_t immediates64[] = {
    0, 1, -1, 64, 65, -16, -17, INT32_MIN, INT32_MAX, 0x12345678, -0x12345678};

using Inputs = std::vector<Words>;

namespace {

// Collects the kernel inputs and the sizes the harness expects for them.
struct KernelInputs {
  std::vector<InputKind> kinds;
  Inputs words;
};

} // namespace

// The non-empty lines of text, without leading blanks.
static std::vector<std::string> instructionsOf(const std::string &text) {
  std::vector<std::string> lines;
  std::istringstream stream(text);
  for (std::string current; std::getline(stream, current);) {
    size_t first = current.find_first_not_of(" \t");
    if (first != std::string::npos)
      lines.push_back(current.substr(first));
  }
  return lines;
}

static RegisterRange allocateKind(IRBuilder &builder, RegisterKind kind,
                                  int32_t count) {
  return kind == sgpr ? builder.allocateSgpr(count, 2)
                      : builder.allocateVgpr(count, 2);
}

static void addInput(KernelInputs &inputs, InputKind kind, const Words &source,
                     size_t laneCount, size_t groupCount) {
  inputs.kinds.push_back(kind);
  size_t count = kind == lane ? laneCount : groupCount;
  inputs.words.emplace_back(source.begin(), source.begin() + count);
}

// Core.

using IRBuilderTest = BuilderTest;

TEST_F(IRBuilderTest, EmitsKernelSequence) {
  RegisterRange kernarg = allocator.reserve(RegisterKind::sgpr, 0, 2);
  RegisterRange ptr = builder.allocateSgpr(2, 2);
  RegisterRange voff = builder.allocateVgpr(1);
  RegisterRange vx = builder.allocateVgpr(1);
  RegisterRange vy = builder.allocateVgpr(1);

  builder.createComment("saxpy body");
  builder.create<rocisa::SLoadB64>(ptr.container(), kernarg.container(), 0);
  builder.createWaitCnt({.scalarMemory = 0});
  std::shared_ptr<rocisa::Label> end = builder.createLabel("end");
  builder.create<rocisa::SCBranchExecZ>(end->getLabelName());
  builder.create<rocisa::GlobalLoadB32>(vx.container(), voff.container(),
                                        ptr.container());
  builder.createWaitCnt({.vectorLoads = 0});
  builder.create<rocisa::VMacF32>(vy.container(), vx.container(),
                                  vx.container());
  builder.create<rocisa::GlobalStoreB32>(voff.container(), vy.container(),
                                         ptr.container());
  builder.insert(end);
  builder.createEndProgram();

  expectLinesInOrder(body.toString(), {
                                          "// saxpy body",
                                          "s_load_dwordx2 s[2:3], s[0:1], 0",
                                          "s_waitcnt lgkmcnt(0)",
                                          "s_cbranch_execz label_end_0",
                                          "global_load_dword v1, v0, s[2:3]",
                                          "s_waitcnt vmcnt(0)",
                                          "v_fmac_f32 v2, v1, v1",
                                          "global_store_dword v0, v2, s[2:3]",
                                          "label_end_0:",
                                          "s_endpgm",
                                      });
}

TEST_F(IRBuilderTest, CreateReturnsInsertedInstance) {
  std::shared_ptr<rocisa::SEndpgm> inst = builder.create<rocisa::SEndpgm>();
  EXPECT_EQ(std::ssize(body.itemList), 1);
  EXPECT_EQ(body.itemList.back().get(),
            static_cast<rocisa::Item *>(inst.get()));
  EXPECT_EQ(inst->parent, &body);
}

TEST_F(IRBuilderTest, InsertionGuardRestoresInsertionPoint) {
  std::shared_ptr<rocisa::Module> nested =
      builder.create<rocisa::Module>("nested");
  {
    IRBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(*nested);
    builder.createEndProgram();
    EXPECT_EQ(&builder.insertionPoint(), nested.get());
  }
  EXPECT_EQ(&builder.insertionPoint(), &body);
  builder.createEndProgram();
  EXPECT_EQ(std::ssize(nested->itemList), 1);
  EXPECT_EQ(std::ssize(body.itemList), 2);
}

TEST_F(IRBuilderTest, LabelNamesAreUnique) {
  std::shared_ptr<rocisa::Label> a = builder.createLabel("loop");
  std::shared_ptr<rocisa::Label> b = builder.createLabel("loop");
  std::shared_ptr<rocisa::Label> c = builder.createLabel("end");
  EXPECT_NE(a->getLabelName(), b->getLabelName());
  EXPECT_NE(a->getLabelName(), c->getLabelName());
  EXPECT_NE(b->getLabelName(), c->getLabelName());
  EXPECT_TRUE(body.itemList.empty());
}

TEST_F(IRBuilderTest, InsertRejectsItemAlreadyInModule) {
  std::shared_ptr<rocisa::Label> label = builder.createLabel("loop");
  builder.insert(label);
  EXPECT_THROW(builder.insert(label), std::logic_error);
  EXPECT_THROW(builder.insert(std::shared_ptr<rocisa::Label>()),
               std::invalid_argument);
}

TEST_F(IRBuilderTest, RangesReleaseOnScopeExit) {
  {
    RegisterRange v = builder.allocateVgpr(4);
    EXPECT_EQ(v.start(), 0);
    RegisterRange s = builder.allocateSgpr(2, 2);
    EXPECT_EQ(s.start(), 0);
  }
  EXPECT_EQ(builder.allocateVgpr(4).start(), 0);
  EXPECT_EQ(builder.allocateSgpr(2, 2).start(), 0);
  EXPECT_EQ(allocator.highWaterMark(RegisterKind::vgpr), 4);
}

TEST_F(IRBuilderTest, AllocatesAgprs) {
  RegisterRange value = builder.allocateVgpr(1);
  RegisterRange acc = builder.allocateAgpr(4);
  EXPECT_EQ(acc.kind(), RegisterKind::agpr);
  EXPECT_EQ(acc.start(), 0);
  RegisterRange aligned = builder.allocateAgpr(2, 2);
  EXPECT_EQ(aligned.start(), 4);
  builder.create<rocisa::VAccvgprWriteB32>(acc.sub(3), value.container());
  builder.create<rocisa::VAccvgprReadB32>(value.container(), acc.sub(3));
  std::string text = body.toString();
  EXPECT_NE(text.find("v_accvgpr_write_b32 acc3, v0"), std::string::npos)
      << text;
  EXPECT_NE(text.find("v_accvgpr_read_b32 v0, acc3"), std::string::npos)
      << text;
  EXPECT_EQ(allocator.highWaterMark(RegisterKind::agpr), 6);
  EXPECT_EQ(allocator.highWaterMark(RegisterKind::vgpr), 1);
}

TEST_F(IRBuilderTest, SelectsTargetOnConstructingThread) {
  std::string text;
  std::thread worker([&] {
    rocisa::Module module("thread");
    IRBuilder threadBuilder(target, allocator, module);
    threadBuilder.create<rocisa::VMacF32>(rocisa::vgpr(2), rocisa::vgpr(1),
                                          rocisa::vgpr(1));
    text = module.toString();
  });
  worker.join();
  EXPECT_NE(text.find("v_fmac_f32 v2, v1, v1"), std::string::npos) << text;
}

// Operands.

TEST(OperandTest, ConstructsFromRegisters) {
  Target target = Target::get("gfx950");
  RegisterAllocator allocator(target);
  RegisterRange scalar = allocator.allocate(RegisterKind::sgpr, 1);
  RegisterRange range = allocator.allocate(RegisterKind::vgpr, 2);
  RegisterRange accumulators = allocator.allocate(RegisterKind::agpr, 4);
  Operand fromRange(range);
  EXPECT_TRUE(fromRange.isRegister());
  EXPECT_TRUE(fromRange.isRegister(RegisterKind::vgpr));
  EXPECT_FALSE(fromRange.isRegister(RegisterKind::sgpr));
  EXPECT_FALSE(fromRange.isImmediate());
  EXPECT_EQ(fromRange.reg(), range.ref());

  Operand fromRef(range.ref(1));
  EXPECT_EQ(fromRef.reg(), (RegisterRef{RegisterKind::vgpr, 1, 1}));

  // The register is read back from its container for every kind and width.
  Operand fromScalar(scalar);
  EXPECT_TRUE(fromScalar.isRegister(RegisterKind::sgpr));
  EXPECT_FALSE(fromScalar.isRegister(RegisterKind::agpr));
  EXPECT_EQ(fromScalar.reg(), scalar.ref());
  Operand fromAccumulators(accumulators);
  EXPECT_TRUE(fromAccumulators.isRegister(RegisterKind::agpr));
  EXPECT_FALSE(fromAccumulators.isRegister(RegisterKind::vgpr));
  EXPECT_EQ(fromAccumulators.reg(), accumulators.ref());
}

TEST(OperandTest, ConstructsFromIntegers) {
  Operand fromInt(5);
  EXPECT_TRUE(fromInt.isImmediate());
  EXPECT_FALSE(fromInt.isRegister());
  EXPECT_FALSE(fromInt.isRegister(RegisterKind::vgpr));
  EXPECT_EQ(fromInt.immediate(), 5);
  EXPECT_EQ(Operand(-1).immediate(), -1);
  EXPECT_EQ(Operand(INT32_MIN).immediate(), INT32_MIN);
  EXPECT_EQ(Operand(INT32_MAX).immediate(), INT32_MAX);
}

// Wide and unsigned integers are rejected at compile time, so a constant never
// truncates silently.
static_assert(!std::is_constructible_v<Operand, int64_t>);
static_assert(!std::is_constructible_v<Operand, uint32_t>);
static_assert(!std::is_constructible_v<Operand, uint64_t>);
static_assert(std::is_constructible_v<Operand, int32_t>);
static_assert(sizeof(Operand) == sizeof(InstructionInput));

// Moves.

namespace {

class MoveTest : public BuilderTest {
protected:
  std::string emitMov32(RegisterRef dst, Operand src) {
    return emitted([&](IRBuilder &local) { local.movb32(dst, src); });
  }
  std::string emitMov64(RegisterRef dst, Operand src) {
    return emitted([&](IRBuilder &local) { local.movb64(dst, src); });
  }
  int moveInstructionCount(const std::string &out) {
    return countLinesContaining(out, "_mov_") +
           countLinesContaining(out, "accvgpr");
  }
};

struct PlacementCounts {
  int assembled = 0;
  int rejected = 0;
};

} // namespace

TEST_F(MoveTest, Movb32ScalarDestination) {
  RegisterRange dst = builder.allocateSgpr(1);
  RegisterRange src = builder.allocateSgpr(1);
  EXPECT_EQ(emitMov32(dst.ref(), src),
            line("s_mov_b32", text(dst), text(src)) + "\n");
  expectLinesInOrder(emitMov32(dst.ref(), 0x12345),
                     {line("s_mov_b32", text(dst), "74565")});
  expectLinesInOrder(emitMov32(dst.ref(), 7),
                     {line("s_mov_b32", text(dst), "7")});
}

TEST_F(MoveTest, Movb32VectorDestination) {
  RegisterRange dst = builder.allocateVgpr(1);
  RegisterRange vsrc = builder.allocateVgpr(1);
  RegisterRange ssrc = builder.allocateSgpr(1);
  RegisterRange asrc = builder.allocateAgpr(1);
  expectLinesInOrder(emitMov32(dst.ref(), vsrc),
                     {line("v_mov_b32", text(dst), text(vsrc))});
  expectLinesInOrder(emitMov32(dst.ref(), ssrc),
                     {line("v_mov_b32", text(dst), text(ssrc))});
  expectLinesInOrder(emitMov32(dst.ref(), -1),
                     {line("v_mov_b32", text(dst), "-1")});
  expectLinesInOrder(emitMov32(dst.ref(), asrc),
                     {line("v_accvgpr_read_b32", text(dst), text(asrc))});
}

TEST_F(MoveTest, Movb32AccumulatorDestination) {
  RegisterRange dst = builder.allocateAgpr(1);
  RegisterRange vsrc = builder.allocateVgpr(1);
  RegisterRange ssrc = builder.allocateSgpr(1);
  expectLinesInOrder(emitMov32(dst.ref(), vsrc),
                     {line("v_accvgpr_write_b32", text(dst), text(vsrc))});
  expectLinesInOrder(emitMov32(dst.ref(), ssrc),
                     {line("v_accvgpr_write_b32", text(dst), text(ssrc))});
  expectLinesInOrder(emitMov32(dst.ref(), 7),
                     {line("v_accvgpr_write_b32", text(dst), "7")});
}

TEST_F(MoveTest, Movb32InvalidOperandsThrowWithoutEmitting) {
  RegisterRange sdst = builder.allocateSgpr(1);
  RegisterRange adst = builder.allocateAgpr(1);
  RegisterRange asrc = builder.allocateAgpr(1);
  RegisterRange vsrc = builder.allocateVgpr(1);
  RegisterRange vpair = builder.allocateVgpr(2);
  RegisterRange spair = builder.allocateSgpr(2);
  EXPECT_THROW(builder.movb32(sdst.ref(), vsrc), std::invalid_argument);
  EXPECT_THROW(builder.movb32(sdst.ref(), asrc), std::invalid_argument);
  EXPECT_THROW(builder.movb32(adst.ref(), 0x12345), std::invalid_argument);
  EXPECT_THROW(builder.movb32(adst.ref(), asrc), std::invalid_argument);
  EXPECT_THROW(builder.movb32(vpair.ref(), vsrc), std::invalid_argument);
  EXPECT_THROW(builder.movb32(sdst.ref(), spair), std::invalid_argument);
  EXPECT_EQ(body.toString(), "");
}

TEST_F(MoveTest, Movb64RegisterPairs) {
  RegisterRange sdst = builder.allocateSgpr(2, 2);
  RegisterRange ssrc = builder.allocateSgpr(2, 2);
  RegisterRange vdst = builder.allocateVgpr(2, 2);
  RegisterRange vsrc = builder.allocateVgpr(2, 2);
  expectLinesInOrder(emitMov64(sdst.ref(), ssrc),
                     {line("s_mov_b64", text(sdst), text(ssrc))});
  expectLinesInOrder(emitMov64(vdst.ref(), vsrc),
                     {line("v_mov_b64", text(vdst), text(vsrc))});
  expectLinesInOrder(emitMov64(vdst.ref(), ssrc),
                     {line("v_mov_b64", text(vdst), text(ssrc))});
}

TEST_F(MoveTest, Movb64SingleInstructionImmediates) {
  RegisterRange sdst = builder.allocateSgpr(2, 2);
  RegisterRange vdst = builder.allocateVgpr(2, 2);
  struct Case {
    int32_t value;
    std::string printed;
  };
  // A non-inline literal is one 64-bit mov only when it is non-negative, since
  // the hardware zero-extends it.
  for (const Case &current :
       {Case{-1, "-1"}, Case{64, "64"}, Case{-16, "-16"}, Case{100, "100"},
        Case{0x12345, "74565"}, Case{INT32_MAX, "2147483647"}}) {
    SCOPED_TRACE(current.printed);
    std::string scalar = emitMov64(sdst.ref(), current.value);
    expectLinesInOrder(scalar,
                       {line("s_mov_b64", text(sdst), current.printed)});
    EXPECT_EQ(moveInstructionCount(scalar), 1);
    std::string vector = emitMov64(vdst.ref(), current.value);
    expectLinesInOrder(vector,
                       {line("v_mov_b64", text(vdst), current.printed)});
    EXPECT_EQ(moveInstructionCount(vector), 1);
  }
}

TEST_F(MoveTest, Movb64SplitImmediates) {
  RegisterRange sdst = builder.allocateSgpr(2, 2);
  RegisterRange vdst = builder.allocateVgpr(2, 2);
  RegisterRange adst = builder.allocateAgpr(2, 2);
  // A negative literal sign-extends to a high half of -1 and splits.
  std::string negative = emitMov64(sdst.ref(), -100);
  expectLinesInOrder(negative, {line("s_mov_b32", text(sdst.ref(0, 1)), "-100"),
                                line("s_mov_b32", text(sdst.ref(1, 1)), "-1")});
  EXPECT_EQ(moveInstructionCount(negative), 2);

  expectLinesInOrder(emitMov64(vdst.ref(), -17),
                     {line("v_mov_b32", text(vdst.ref(0, 1)), "-17"),
                      line("v_mov_b32", text(vdst.ref(1, 1)), "-1")});
  expectLinesInOrder(emitMov64(vdst.ref(), INT32_MIN),
                     {line("v_mov_b32", text(vdst.ref(0, 1)), "-2147483648"),
                      line("v_mov_b32", text(vdst.ref(1, 1)), "-1")});
  expectLinesInOrder(emitMov64(adst.ref(), -5),
                     {line("v_accvgpr_write_b32", text(adst.ref(0, 1)), "-5"),
                      line("v_accvgpr_write_b32", text(adst.ref(1, 1)), "-1")});
}

TEST_F(MoveTest, Movb64MisalignedPairsSplit) {
  RegisterRange vrange = builder.allocateVgpr(3, 2);
  RegisterRange vsrc = builder.allocateVgpr(2, 2);
  RegisterRange vdst = builder.allocateVgpr(2, 2);
  std::string toMisaligned = emitMov64(vrange.ref(1, 2), vsrc);
  expectLinesInOrder(
      toMisaligned,
      {line("v_mov_b32", text(vrange.ref(1, 1)), text(vsrc.ref(0, 1))),
       line("v_mov_b32", text(vrange.ref(2, 1)), text(vsrc.ref(1, 1)))});
  EXPECT_EQ(moveInstructionCount(toMisaligned), 2);
  std::string fromMisaligned = emitMov64(vdst.ref(), vrange.ref(1, 2));
  EXPECT_EQ(countLinesContaining(fromMisaligned, "v_mov_b64"), 0);
  EXPECT_EQ(moveInstructionCount(fromMisaligned), 2);
}

TEST_F(MoveTest, Movb64OverlappingCopyOrdersHalves) {
  RegisterRange range = builder.allocateVgpr(4, 2);
  // Copying up reads the high source word first so it is not overwritten.
  expectLinesInOrder(
      emitMov64(range.ref(1, 2), range.ref(0, 2)),
      {line("v_mov_b32", text(range.ref(2, 1)), text(range.ref(1, 1))),
       line("v_mov_b32", text(range.ref(1, 1)), text(range.ref(0, 1)))});
  expectLinesInOrder(
      emitMov64(range.ref(0, 2), range.ref(1, 2)),
      {line("v_mov_b32", text(range.ref(0, 1)), text(range.ref(1, 1))),
       line("v_mov_b32", text(range.ref(1, 1)), text(range.ref(2, 1)))});
}

TEST_F(MoveTest, Movb64AccumulatorPairsSplit) {
  RegisterRange apair = builder.allocateAgpr(2, 2);
  RegisterRange vpair = builder.allocateVgpr(2, 2);
  expectLinesInOrder(emitMov64(apair.ref(), vpair),
                     {line("v_accvgpr_write_b32", text(apair.ref(0, 1)),
                           text(vpair.ref(0, 1))),
                      line("v_accvgpr_write_b32", text(apair.ref(1, 1)),
                           text(vpair.ref(1, 1)))});
  expectLinesInOrder(
      emitMov64(vpair.ref(), apair),
      {line("v_accvgpr_read_b32", text(vpair.ref(0, 1)), text(apair.ref(0, 1))),
       line("v_accvgpr_read_b32", text(vpair.ref(1, 1)),
            text(apair.ref(1, 1)))});
}

TEST_F(MoveTest, Movb64InvalidOperandsThrowWithoutEmitting) {
  RegisterRange single = builder.allocateVgpr(1);
  RegisterRange vpair = builder.allocateVgpr(2, 2);
  RegisterRange spair = builder.allocateSgpr(2, 2);
  RegisterRange apair = builder.allocateAgpr(2, 2);
  EXPECT_THROW(builder.movb64(single.ref(), vpair), std::invalid_argument);
  EXPECT_THROW(builder.movb64(vpair.ref(), single), std::invalid_argument);
  EXPECT_THROW(builder.movb64(spair.ref(), vpair), std::invalid_argument);
  EXPECT_THROW(builder.movb64(spair.ref(), apair), std::invalid_argument);
  EXPECT_THROW(builder.movb64(apair.ref(), apair), std::invalid_argument);
  EXPECT_THROW(builder.movb64(apair.ref(), 0x12345), std::invalid_argument);
  EXPECT_EQ(body.toString(), "");
}

constexpr int32_t moveImmediates32[] = {0, -1, 64, 65, INT32_MIN, 0x12345678};

using MoveDevice = DeviceTest;

static RegisterRef halfOf(RegisterRef pair, int index) {
  return RegisterRef{pair.kind, pair.start + index, 1};
}

TEST_F(MoveDevice, Movb32LaneThroughVgpr) {
  WordPairs pairs = makeArithPairs(0);
  expectArithResults({{InputKind::lane},
                      vgpr,
                      [](IRBuilder &builder, RegisterRef dst,
                         const std::vector<RegisterRef> &inputs) {
                        RegisterRange temp = builder.allocateVgpr(1);
                        builder.movb32(temp.ref(), inputs[0]);
                        builder.movb32(dst, temp.ref());
                      }},
                     laneGroupCount(pairs.a), {pairs.a}, pairs.a);
}

TEST_F(MoveDevice, Movb32UniformToVgprAndSgpr) {
  WordPairs pairs = makeArithPairs(0);
  int groups = static_cast<int>(pairs.a.size());
  ArithEmitter copy = [](IRBuilder &builder, RegisterRef dst,
                         const std::vector<RegisterRef> &inputs) {
    builder.movb32(dst, inputs[0]);
  };
  expectArithResults({{InputKind::uniform}, vgpr, copy}, groups, {pairs.a},
                     perLane(pairs.a));
  expectArithResults({{InputKind::uniform}, sgpr, copy}, groups, {pairs.a},
                     pairs.a);
}

TEST_F(MoveDevice, Movb32Immediates) {
  for (int32_t immediate : moveImmediates32) {
    SCOPED_TRACE(immediate);
    ArithEmitter emit = [immediate](IRBuilder &builder, RegisterRef dst,
                                    const std::vector<RegisterRef> &) {
      builder.movb32(dst, immediate);
    };
    uint32_t word = static_cast<uint32_t>(immediate);
    expectArithResults({{}, vgpr, emit}, 2, {},
                       std::vector<uint32_t>(2 * arithWorkgroupSize, word));
    expectArithResults({{}, sgpr, emit}, 3, {}, std::vector<uint32_t>(3, word));
  }
}

TEST_F(MoveDevice, Movb32ThroughAccumulator) {
  WordPairs pairs = makeArithPairs(0);
  int groups = static_cast<int>(pairs.a.size());
  ArithEmitter viaAgpr = [](IRBuilder &builder, RegisterRef dst,
                            const std::vector<RegisterRef> &inputs) {
    RegisterRange acc = builder.allocateAgpr(1);
    builder.movb32(acc.ref(), inputs[0]);
    builder.movb32(dst, acc.ref());
  };
  SCOPED_TRACE("lane vgpr to agpr");
  expectArithResults({{InputKind::lane}, vgpr, viaAgpr},
                     laneGroupCount(pairs.a), {pairs.a}, pairs.a);
  SCOPED_TRACE("uniform sgpr to agpr");
  expectArithResults({{InputKind::uniform}, vgpr, viaAgpr}, groups, {pairs.a},
                     perLane(pairs.a));
  SCOPED_TRACE("inline constant to agpr");
  expectArithResults({{},
                      vgpr,
                      [](IRBuilder &builder, RegisterRef dst,
                         const std::vector<RegisterRef> &) {
                        RegisterRange acc = builder.allocateAgpr(1);
                        builder.movb32(acc.ref(), 7);
                        builder.movb32(dst, acc.ref());
                      }},
                     2, {}, std::vector<uint32_t>(2 * arithWorkgroupSize, 7));
}

using PairEmitter = std::function<PairRegisters(
    IRBuilder &builder, const std::vector<RegisterRef> &inputs)>;

// A 64-bit value is checked through two runs, one storing each half.
static void expectPairResults(const std::vector<InputKind> &kinds,
                              RegisterKind result, const PairEmitter &emit,
                              int workgroupCount,
                              const std::vector<std::vector<uint32_t>> &inputs,
                              const std::vector<uint32_t> &low,
                              const std::vector<uint32_t> &high) {
  for (int half = 0; half < 2; ++half) {
    SCOPED_TRACE(half == 0 ? "low half" : "high half");
    ArithEmitter storeHalf = [&emit, half](IRBuilder &builder, RegisterRef dst,
                                           const std::vector<RegisterRef> &in) {
      PairRegisters value = emit(builder, in);
      builder.movb32(dst, halfOf(value.pair, half));
    };
    expectArithResults({kinds, result, storeHalf}, workgroupCount, inputs,
                       half == 0 ? low : high);
  }
}

TEST_F(MoveDevice, Movb64Immediates) {
  for (int32_t immediate : immediates64) {
    for (RegisterKind kind : {sgpr, vgpr}) {
      SCOPED_TRACE("immediate " + std::to_string(immediate) +
                   (kind == sgpr ? " to sgpr pair" : " to vgpr pair"));
      PairEmitter emit = [immediate, kind](IRBuilder &builder,
                                           const std::vector<RegisterRef> &) {
        RegisterRange pair = allocateKind(builder, kind, 2);
        RegisterRef ref = pair.ref();
        builder.movb64(ref, immediate);
        return PairRegisters{std::move(pair), ref};
      };
      // The expected halves come from the sign-extended host value, which
      // proves that a negative literal is split and not zero-extended.
      int64_t extended = immediate;
      uint32_t low = static_cast<uint32_t>(extended);
      uint32_t high = static_cast<uint32_t>(extended >> 32);
      size_t words = kind == sgpr ? 3 : 2 * arithWorkgroupSize;
      expectPairResults({}, kind, emit, kind == sgpr ? 3 : 2, {},
                        std::vector<uint32_t>(words, low),
                        std::vector<uint32_t>(words, high));
    }
  }
}

// Fills a pair from two inputs with 32-bit moves.
static RegisterRange fillPair(IRBuilder &builder, RegisterKind kind,
                              RegisterRef low, RegisterRef high) {
  RegisterRange pair = allocateKind(builder, kind, 2);
  builder.movb32(halfOf(pair.ref(), 0), low);
  builder.movb32(halfOf(pair.ref(), 1), high);
  return pair;
}

TEST_F(MoveDevice, Movb64VgprPairsAlignedAndMisaligned) {
  WordPairs pairs = makeArithPairs(0);
  int groups = laneGroupCount(pairs.a);
  std::vector<InputKind> kinds{InputKind::lane, InputKind::lane};
  SCOPED_TRACE("aligned to aligned");
  expectPairResults(
      kinds, vgpr,
      [](IRBuilder &builder, const std::vector<RegisterRef> &in) {
        RegisterRange src = fillPair(builder, vgpr, in[0], in[1]);
        RegisterRange dst = builder.allocateVgpr(2, 2);
        RegisterRef ref = dst.ref();
        builder.movb64(ref, src.ref());
        return PairRegisters{std::move(dst), ref};
      },
      groups, {pairs.a, pairs.b}, pairs.a, pairs.b);
  SCOPED_TRACE("aligned to misaligned");
  expectPairResults(
      kinds, vgpr,
      [](IRBuilder &builder, const std::vector<RegisterRef> &in) {
        RegisterRange src = fillPair(builder, vgpr, in[0], in[1]);
        RegisterRange dst = builder.allocateVgpr(3, 2);
        RegisterRef ref = dst.ref(1, 2);
        builder.movb64(ref, src.ref());
        return PairRegisters{std::move(dst), ref};
      },
      groups, {pairs.a, pairs.b}, pairs.a, pairs.b);
  SCOPED_TRACE("misaligned to aligned");
  expectPairResults(
      kinds, vgpr,
      [](IRBuilder &builder, const std::vector<RegisterRef> &in) {
        RegisterRange src = builder.allocateVgpr(3, 2);
        builder.movb32(src.ref(1, 1), in[0]);
        builder.movb32(src.ref(2, 1), in[1]);
        RegisterRange dst = builder.allocateVgpr(2, 2);
        RegisterRef ref = dst.ref();
        builder.movb64(ref, src.ref(1, 2));
        return PairRegisters{std::move(dst), ref};
      },
      groups, {pairs.a, pairs.b}, pairs.a, pairs.b);
}

TEST_F(MoveDevice, Movb64OverlappingMisalignedCopies) {
  WordPairs pairs = makeArithPairs(0);
  int groups = laneGroupCount(pairs.a);
  std::vector<InputKind> kinds{InputKind::lane, InputKind::lane};
  SCOPED_TRACE("copy up by one register");
  expectPairResults(
      kinds, vgpr,
      [](IRBuilder &builder, const std::vector<RegisterRef> &in) {
        RegisterRange range = builder.allocateVgpr(4, 2);
        builder.movb32(range.ref(0, 1), in[0]);
        builder.movb32(range.ref(1, 1), in[1]);
        builder.movb64(range.ref(1, 2), range.ref(0, 2));
        RegisterRef ref = range.ref(1, 2);
        return PairRegisters{std::move(range), ref};
      },
      groups, {pairs.a, pairs.b}, pairs.a, pairs.b);
  SCOPED_TRACE("copy down by one register");
  expectPairResults(
      kinds, vgpr,
      [](IRBuilder &builder, const std::vector<RegisterRef> &in) {
        RegisterRange range = builder.allocateVgpr(4, 2);
        builder.movb32(range.ref(1, 1), in[0]);
        builder.movb32(range.ref(2, 1), in[1]);
        builder.movb64(range.ref(0, 2), range.ref(1, 2));
        RegisterRef ref = range.ref(0, 2);
        return PairRegisters{std::move(range), ref};
      },
      groups, {pairs.a, pairs.b}, pairs.a, pairs.b);
}

TEST_F(MoveDevice, Movb64ScalarPairs) {
  WordPairs pairs = makeArithPairs(0);
  int groups = static_cast<int>(pairs.a.size());
  std::vector<InputKind> kinds{InputKind::uniform, InputKind::uniform};
  SCOPED_TRACE("sgpr pair to sgpr pair");
  expectPairResults(
      kinds, sgpr,
      [](IRBuilder &builder, const std::vector<RegisterRef> &in) {
        RegisterRange src = fillPair(builder, sgpr, in[0], in[1]);
        RegisterRange dst = builder.allocateSgpr(2, 2);
        RegisterRef ref = dst.ref();
        builder.movb64(ref, src.ref());
        return PairRegisters{std::move(dst), ref};
      },
      groups, {pairs.a, pairs.b}, pairs.a, pairs.b);
  SCOPED_TRACE("sgpr pair to vgpr pair");
  expectPairResults(
      kinds, vgpr,
      [](IRBuilder &builder, const std::vector<RegisterRef> &in) {
        RegisterRange src = fillPair(builder, sgpr, in[0], in[1]);
        RegisterRange dst = builder.allocateVgpr(2, 2);
        RegisterRef ref = dst.ref();
        builder.movb64(ref, src.ref());
        return PairRegisters{std::move(dst), ref};
      },
      groups, {pairs.a, pairs.b}, perLane(pairs.a), perLane(pairs.b));
}

// Placements of a source, checked by the assembler alone.
static Operand registerSource(RegisterKind kind, RegisterRange &sgprs,
                              RegisterRange &vgprs, RegisterRange &agprs,
                              int width, int offset) {
  RegisterRange &range = kind == sgpr ? sgprs : kind == vgpr ? vgprs : agprs;
  return range.ref(offset, width);
}

// Emits dst = src in a fresh kernel and assembles it. A combination the
// builder rejects is counted and skipped.
static void
assemblePlacement(RegisterKind dstKind, int width, int dstOffset,
                  const std::function<Operand(RegisterRange &, RegisterRange &,
                                              RegisterRange &)> &makeSource,
                  PlacementCounts &counts) {
  Target target = Target::get("gfx950");
  Kernel kernel(target, "codegen_move_placement", arithWorkgroupSize);
  IRBuilder &builder = kernel.builder();
  RegisterRange sgprs = builder.allocateSgpr(4, 2);
  RegisterRange vgprs = builder.allocateVgpr(4, 2);
  RegisterRange agprs = builder.allocateAgpr(4, 2);
  RegisterRange &dstRange = dstKind == sgpr   ? sgprs
                            : dstKind == vgpr ? vgprs
                                              : agprs;
  RegisterRef dst = dstRange.ref(dstOffset, width);
  Operand src = makeSource(sgprs, vgprs, agprs);
  try {
    if (width == 1)
      builder.movb32(dst, src);
    else
      builder.movb64(dst, src);
  } catch (const std::invalid_argument &) {
    ++counts.rejected;
    return;
  }
  builder.createEndProgram();
  std::string assembly = kernel.toAssembly();
  SCOPED_TRACE(assembly);
  EXPECT_NO_THROW(assembleToHSACO(assembly, target.isaName()));
  ++counts.assembled;
}

TEST(MoveAssembly, AllPlacementsAssemble) {
  PlacementCounts counts;
  for (RegisterKind dstKind : {sgpr, vgpr, agpr}) {
    for (int width : {1, 2}) {
      for (int offset : {0, 1}) {
        for (RegisterKind srcKind : {sgpr, vgpr, agpr})
          for (int srcOffset : {0, 1})
            assemblePlacement(
                dstKind, width, offset,
                [=](RegisterRange &s, RegisterRange &v, RegisterRange &a) {
                  return registerSource(srcKind, s, v, a, width, srcOffset);
                },
                counts);
        std::vector<int32_t> values{7, -5, 0x12345, -1};
        if (width == 2)
          values.insert(values.end(), {-17, INT32_MIN, INT32_MAX});
        for (int32_t value : values)
          assemblePlacement(
              dstKind, width, offset,
              [=](RegisterRange &, RegisterRange &, RegisterRange &) {
                return Operand(value);
              },
              counts);
      }
    }
  }
  EXPECT_GT(counts.assembled, 60);
  EXPECT_GT(counts.rejected, 0);
}

// 32-bit arithmetic.

namespace {

struct OpCase32 {
  const char *name;
  BuilderOp op;
  std::string_view valuMnemonic;
  std::string_view saluMnemonic;
  uint32_t (*host)(uint32_t, uint32_t);
};

class ArithTest : public BuilderTest {
protected:
  std::string emitAdd(RegisterRef dst, Operand a, Operand b) {
    return emitOp(&IRBuilder::addi32, dst, a, b);
  }
  std::string emitSub(RegisterRef dst, Operand a, Operand b) {
    return emitOp(&IRBuilder::subi32, dst, a, b);
  }
  std::string emitMul(RegisterRef dst, Operand a, Operand b) {
    return emitOp(&IRBuilder::muli32, dst, a, b);
  }
};

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

struct PlacementRegisters {
  RegisterRange vectorA;
  RegisterRange vectorB;
  RegisterRange scalarA;
  RegisterRange scalarB;
};

} // namespace

constexpr OpCase32 opCases32[] = {
    {"addi32", &IRBuilder::addi32, "v_add_u32", "s_add_u32",
     [](uint32_t a, uint32_t b) { return a + b; }},
    {"subi32", &IRBuilder::subi32, "v_sub_u32", "s_sub_u32",
     [](uint32_t a, uint32_t b) { return a - b; }},
    {"muli32", &IRBuilder::muli32, "v_mul_lo_u32", "s_mul_i32",
     [](uint32_t a, uint32_t b) { return a * b; }},
};

TEST_F(ArithTest, SaluForScalarOperands) {
  RegisterRange dst = builder.allocateSgpr(1);
  RegisterRange a = builder.allocateSgpr(1);
  RegisterRange b = builder.allocateSgpr(1);
  for (const OpCase32 &current : opCases32) {
    SCOPED_TRACE(current.saluMnemonic);
    std::string out = emitOp(current.op, dst.ref(), a, b);
    expectLinesInOrder(
        out, {line(current.saluMnemonic, text(dst), text(a), text(b))});
    EXPECT_EQ(countLinesContaining(out, "v_"), 0);
  }
}

TEST_F(ArithTest, SaluTakesLiteralWithoutCopy) {
  RegisterRange dst = builder.allocateSgpr(1);
  RegisterRange a = builder.allocateSgpr(1);
  std::string out = emitAdd(dst.ref(), a, literalA);
  expectLinesInOrder(out,
                     {line("s_add_u32", text(dst), text(a), literalAText)});
  EXPECT_EQ(countLinesContaining(out, "mov"), 0);
}

TEST_F(ArithTest, ValuForVectorDestination) {
  RegisterRange dst = builder.allocateVgpr(1);
  RegisterRange a = builder.allocateVgpr(1);
  RegisterRange b = builder.allocateVgpr(1);
  for (const OpCase32 &current : opCases32) {
    SCOPED_TRACE(current.valuMnemonic);
    std::string out = emitOp(current.op, dst.ref(), a, b);
    expectLinesInOrder(
        out, {line(current.valuMnemonic, text(dst), text(a), text(b))});
    EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 0);
  }
}

TEST_F(ArithTest, TwoScalarsIntoVgprCopyOnce) {
  RegisterRange dst = builder.allocateVgpr(1);
  RegisterRange a = builder.allocateSgpr(1);
  RegisterRange b = builder.allocateSgpr(1);
  // The temporary takes the lowest free VGPR.
  std::string temp = "v1";
  for (const OpCase32 &current : opCases32) {
    SCOPED_TRACE(current.valuMnemonic);
    std::string out = emitOp(current.op, dst.ref(), a, b);
    expectLinesInOrder(out,
                       {line("v_mov_b32", temp, text(b)),
                        line(current.valuMnemonic, text(dst), text(a), temp)});
    EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 1);
  }
}

TEST_F(ArithTest, SameScalarTwiceNeedsNoCopy) {
  RegisterRange dst = builder.allocateVgpr(1);
  RegisterRange s0 = builder.allocateSgpr(1);
  std::string mul = emitMul(dst.ref(), s0, s0);
  expectLinesInOrder(mul,
                     {line("v_mul_lo_u32", text(dst), text(s0), text(s0))});
  EXPECT_EQ(countLinesContaining(mul, "v_mov_b32"), 0);
  std::string add = emitAdd(dst.ref(), s0, s0);
  expectLinesInOrder(add, {line("v_add_u32", text(dst), text(s0), text(s0))});
  EXPECT_EQ(countLinesContaining(add, "v_mov_b32"), 0);
}

TEST_F(ArithTest, LiteralIntoMulIsCopied) {
  RegisterRange dst = builder.allocateVgpr(1);
  RegisterRange v1 = builder.allocateVgpr(1);
  RegisterRange s0 = builder.allocateSgpr(1);
  std::string out = emitMul(dst.ref(), v1, literalA);
  expectLinesInOrder(out, {line("v_mov_b32", "v2", literalAText),
                           line("v_mul_lo_u32", text(dst), text(v1), "v2")});
  EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 1);

  out = emitMul(dst.ref(), s0, literalA);
  expectLinesInOrder(out, {line("v_mov_b32", "v2", literalAText),
                           line("v_mul_lo_u32", text(dst), text(s0), "v2")});
  EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 1);
}

TEST_F(ArithTest, InlineConstantsNeedNoCopy) {
  RegisterRange dst = builder.allocateVgpr(1);
  RegisterRange v1 = builder.allocateVgpr(1);
  struct Case {
    int32_t value;
    std::string printed;
    int copies;
  };
  for (const Case &current :
       {Case{64, "64", 0}, Case{-16, "-16", 0}, Case{-1, "-1", 0},
        Case{65, "65", 1}, Case{-17, "-17", 1}}) {
    SCOPED_TRACE(current.printed);
    std::string out = emitMul(dst.ref(), v1, current.value);
    EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), current.copies);
    // The value is printed once, in the mul or in its copy.
    EXPECT_EQ(countLinesContaining(out, ", " + current.printed), 1);
  }
}

TEST_F(ArithTest, AddSwapsVgprIntoSrc1) {
  RegisterRange dst = builder.allocateVgpr(1);
  RegisterRange v1 = builder.allocateVgpr(1);
  RegisterRange s0 = builder.allocateSgpr(1);
  std::string out = emitAdd(dst.ref(), v1, s0);
  expectLinesInOrder(out, {line("v_add_u32", text(dst), text(s0), text(v1))});
  EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 0);

  out = emitAdd(dst.ref(), v1, literalA);
  expectLinesInOrder(out,
                     {line("v_add_u32", text(dst), literalAText, text(v1))});
  EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 0);
}

TEST_F(ArithTest, SubKeepsOperandOrder) {
  RegisterRange dst = builder.allocateVgpr(1);
  RegisterRange v1 = builder.allocateVgpr(1);
  RegisterRange s0 = builder.allocateSgpr(1);
  std::string out = emitSub(dst.ref(), s0, v1);
  expectLinesInOrder(out, {line("v_sub_u32", text(dst), text(s0), text(v1))});

  // The VOP3 form takes the SGPR in src1.
  out = emitSub(dst.ref(), v1, s0);
  expectLinesInOrder(out, {line("v_sub_u32", text(dst), text(v1), text(s0))});
  EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 0);

  out = emitSub(dst.ref(), literalA, s0);
  expectLinesInOrder(out, {line("v_mov_b32", "v2", text(s0)),
                           line("v_sub_u32", text(dst), literalAText, "v2")});
  EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 1);
}

TEST_F(ArithTest, SubImmediateBecomesAdd) {
  RegisterRange dst = builder.allocateVgpr(1);
  RegisterRange v1 = builder.allocateVgpr(1);
  std::string out = emitSub(dst.ref(), v1, 100);
  expectLinesInOrder(out, {line("v_add_u32", text(dst), "-100", text(v1))});
  EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 0);

  out = emitSub(dst.ref(), v1, INT32_MIN);
  expectLinesInOrder(out,
                     {line("v_add_u32", text(dst), "-2147483648", text(v1))});
  EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 0);
}

TEST_F(ArithTest, SaluTwoLiteralsCopyOne) {
  RegisterRange dst = builder.allocateSgpr(1);
  std::string out = emitAdd(dst.ref(), literalA, literalB);
  expectLinesInOrder(out, {line("s_mov_b32", "s1", "344865"),
                           line("s_add_u32", text(dst), literalAText, "s1")});
  EXPECT_EQ(countLinesContaining(out, "s_mov_b32"), 1);

  EXPECT_EQ(
      countLinesContaining(emitAdd(dst.ref(), literalA, literalA), "s_mov_b32"),
      0);
  EXPECT_EQ(countLinesContaining(emitAdd(dst.ref(), 5, literalA), "s_mov_b32"),
            0);
}

TEST_F(ArithTest, DestinationMayAliasOperand) {
  RegisterRange v1 = builder.allocateVgpr(1);
  RegisterRange s0 = builder.allocateSgpr(1);
  std::string out = emitAdd(v1.ref(), v1, s0);
  expectLinesInOrder(out, {line("v_add_u32", text(v1), text(s0), text(v1))});

  out = emitMul(v1.ref(), v1, literalA);
  expectLinesInOrder(out, {line("v_mov_b32", "v1", literalAText),
                           line("v_mul_lo_u32", text(v1), text(v1), "v1")});
}

TEST_F(ArithTest, TemporariesAreReleased) {
  RegisterRange dst = builder.allocateVgpr(1);
  RegisterRange a = builder.allocateSgpr(1);
  RegisterRange b = builder.allocateSgpr(1);
  emitAdd(dst.ref(), a, b);
  EXPECT_EQ(allocator.highWaterMark(RegisterKind::vgpr), 2);
  RegisterRange next = builder.allocateVgpr(1);
  EXPECT_EQ(next.start(), 1);
}

TEST_F(ArithTest, InvalidOperandsThrowWithoutEmitting) {
  RegisterRange sdst = builder.allocateSgpr(1);
  RegisterRange vdst = builder.allocateVgpr(1);
  RegisterRange v1 = builder.allocateVgpr(1);
  RegisterRange s1 = builder.allocateSgpr(1);
  RegisterRange acc = builder.allocateAgpr(1);
  RegisterRange pair = builder.allocateVgpr(2);
  EXPECT_THROW(builder.addi32(sdst.ref(), v1, s1), std::invalid_argument);
  EXPECT_THROW(builder.subi32(sdst.ref(), s1, v1), std::invalid_argument);
  EXPECT_THROW(builder.muli32(acc.ref(), s1, s1), std::invalid_argument);
  EXPECT_THROW(builder.addi32(vdst.ref(), acc, s1), std::invalid_argument);
  EXPECT_THROW(builder.addi32(vdst.ref(), s1, acc), std::invalid_argument);
  EXPECT_THROW(builder.addi32(pair.ref(), v1, s1), std::invalid_argument);
  EXPECT_THROW(builder.addi32(vdst.ref(), pair, s1), std::invalid_argument);
  EXPECT_EQ(body.toString(), "");
  EXPECT_EQ(allocator.highWaterMark(RegisterKind::vgpr), 4);
}

TEST_F(ArithTest, BoundaryImmediatesAreAccepted) {
  RegisterRange vdst = builder.allocateVgpr(1);
  RegisterRange v1 = builder.allocateVgpr(1);
  EXPECT_NO_THROW(builder.addi32(vdst.ref(), v1, INT32_MIN));
  EXPECT_NO_THROW(builder.addi32(vdst.ref(), v1, INT32_MAX));
}

// Immediates that cover inline constants, their edges, literals and wrapping.
constexpr int32_t arithImmediates[] = {
    0, 1, -1, 64, 65, -16, -17, INT32_MIN, INT32_MAX, 0x12345678};

using ArithDevice = DeviceTest;

static ArithKernelSpec binarySpec(BuilderOp op, InputKind aKind,
                                  InputKind bKind, RegisterKind result) {
  return {{aKind, bKind},
          result,
          [op](IRBuilder &builder, RegisterRef dst,
               const std::vector<RegisterRef> &inputs) {
            (builder.*op)(dst, inputs[0], inputs[1]);
          }};
}

// Computes dst = input (kind) op immediate, or immediate op input.
static ArithKernelSpec immediateSpec(BuilderOp op, InputKind kind,
                                     RegisterKind result, int32_t immediate,
                                     bool immediateFirst) {
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

static std::vector<uint32_t> applyImmediate(const OpCase32 &op,
                                            const std::vector<uint32_t> &words,
                                            int32_t immediate,
                                            bool immediateFirst) {
  uint32_t value = static_cast<uint32_t>(immediate);
  std::vector<uint32_t> out;
  for (uint32_t word : words)
    out.push_back(immediateFirst ? op.host(value, word) : op.host(word, value));
  return out;
}

TEST_F(ArithDevice, ValuLaneOperands) {
  WordPairs pairs = makeArithPairs(4096);
  for (const OpCase32 &op : opCases32) {
    SCOPED_TRACE(op.name);
    std::vector<uint32_t> expected;
    for (size_t i = 0; i < pairs.a.size(); ++i)
      expected.push_back(op.host(pairs.a[i], pairs.b[i]));
    expectArithResults(
        binarySpec(op.op, InputKind::lane, InputKind::lane, RegisterKind::vgpr),
        laneGroupCount(pairs.a), {pairs.a, pairs.b}, expected);
  }
}

TEST_F(ArithDevice, SaluUniformOperands) {
  WordPairs pairs = makeArithPairs(256);
  for (const OpCase32 &op : opCases32) {
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
  for (const OpCase32 &op : opCases32) {
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
  int groups = laneGroupCount(pairs.a);
  std::vector<uint32_t> edges = arithEdgeValues();
  std::vector<uint32_t> uniformWords;
  for (int g = 0; g < groups; ++g)
    uniformWords.push_back(edges[g % edges.size()]);
  for (const OpCase32 &op : opCases32) {
    SCOPED_TRACE(op.name);
    std::vector<uint32_t> uniformFirst;
    std::vector<uint32_t> uniformSecond;
    for (size_t i = 0; i < pairs.b.size(); ++i) {
      uint32_t scalar = uniformWords[i / arithWorkgroupSize];
      uniformFirst.push_back(op.host(scalar, pairs.b[i]));
      uniformSecond.push_back(op.host(pairs.b[i], scalar));
    }
    expectArithResults(binarySpec(op.op, InputKind::uniform, InputKind::lane,
                                  RegisterKind::vgpr),
                       groups, {uniformWords, pairs.b}, uniformFirst);
    expectArithResults(binarySpec(op.op, InputKind::lane, InputKind::uniform,
                                  RegisterKind::vgpr),
                       groups, {pairs.b, uniformWords}, uniformSecond);
  }
}

TEST_F(ArithDevice, ImmediateOperands) {
  WordPairs pairs = makeArithPairs(0);
  int groups = laneGroupCount(pairs.a);
  int scalarGroups = static_cast<int>(pairs.a.size());
  for (const OpCase32 &op : opCases32) {
    for (int32_t immediate : arithImmediates) {
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
  int groups = laneGroupCount(pairs.a);
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
  std::vector<uint32_t> uniformWords(pairs.a.begin(), pairs.a.begin() + groups);
  expected.clear();
  for (size_t i = 0; i < pairs.b.size(); ++i)
    expected.push_back(uniformWords[i / arithWorkgroupSize] + pairs.b[i]);
  expectArithResults({{InputKind::uniform, InputKind::lane},
                      RegisterKind::vgpr,
                      [](IRBuilder &builder, RegisterRef dst,
                         const std::vector<RegisterRef> &inputs) {
                        builder.addi32(inputs[1], inputs[0], inputs[1]);
                        builder.movb32(dst, inputs[1]);
                      }},
                     groups, {uniformWords, pairs.b}, expected);

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

constexpr int32_t inlineA = 7;
constexpr int32_t inlineB = -5;
constexpr int32_t negativeLiteralB = -100;

static const char *placementName(APlacement placement) {
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

static const char *placementName(BPlacement placement) {
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

static Operand pickA(APlacement placement, const PlacementRegisters &regs) {
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

static Operand pickB(BPlacement placement, const PlacementRegisters &regs) {
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

static bool isCompatible(APlacement a, BPlacement b) {
  if (b == BPlacement::sameSgpr)
    return a == APlacement::sgpr;
  if (b == BPlacement::sameLiteral)
    return a == APlacement::literal;
  return true;
}

// Emits the op under test over two operands.
using PlacementEmitter = std::function<void(IRBuilder &builder, RegisterRef dst,
                                            Operand a, Operand b)>;

static void expectPlacementAssembles(const std::string &label,
                                     const PlacementEmitter &emit, APlacement a,
                                     BPlacement b, RegisterKind dstKind) {
  Target target = Target::get("gfx950");
  Kernel kernel(target, "codegen_placement", arithWorkgroupSize);
  IRBuilder &builder = kernel.builder();
  PlacementRegisters regs{builder.allocateVgpr(1), builder.allocateVgpr(1),
                          builder.allocateSgpr(1), builder.allocateSgpr(1)};
  RegisterRange dst = dstKind == RegisterKind::vgpr ? builder.allocateVgpr(1)
                                                    : builder.allocateSgpr(1);
  emit(builder, dst.ref(), pickA(a, regs), pickB(b, regs));
  builder.createEndProgram();

  std::string assembly = kernel.toAssembly();
  SCOPED_TRACE(label + " a=" + placementName(a) + " b=" + placementName(b) +
               " dst=" + (dstKind == RegisterKind::vgpr ? "vgpr" : "sgpr") +
               "\n" + assembly);
  EXPECT_NO_THROW(assembleToHSACO(assembly, target.isaName()));
}

static void expectAllPlacementsAssemble(const std::string &label,
                                        const PlacementEmitter &emit) {
  for (APlacement a : {APlacement::vgpr, APlacement::sgpr,
                       APlacement::inlineConstant, APlacement::literal})
    for (BPlacement b : {BPlacement::vgpr, BPlacement::otherSgpr,
                         BPlacement::sameSgpr, BPlacement::inlineConstant,
                         BPlacement::literal, BPlacement::sameLiteral,
                         BPlacement::negativeLiteral, BPlacement::minInt32}) {
      if (!isCompatible(a, b))
        continue;
      expectPlacementAssembles(label, emit, a, b, RegisterKind::vgpr);
      // A scalar destination cannot take a vector operand.
      if (a == APlacement::vgpr || b == BPlacement::vgpr)
        continue;
      expectPlacementAssembles(label, emit, a, b, RegisterKind::sgpr);
    }
}

TEST(ArithAssembly, AllOperandPlacementsAssemble) {
  for (const OpCase32 &op : opCases32)
    expectAllPlacementsAssemble(
        op.name, [&op](IRBuilder &builder, RegisterRef dst, Operand a,
                       Operand b) { (builder.*op.op)(dst, a, b); });
}

// 64-bit arithmetic.

namespace {

struct OpCase64 {
  const char *name;
  BuilderOp op;
  uint64_t (*host)(uint64_t, uint64_t);
};

class Arith64Test : public BuilderTest {
protected:
  std::string emitAdd(RegisterRef dst, Operand a, Operand b) {
    return emitOp(&IRBuilder::addi64, dst, a, b);
  }
  std::string emitSub(RegisterRef dst, Operand a, Operand b) {
    return emitOp(&IRBuilder::subi64, dst, a, b);
  }
  std::string emitMul(RegisterRef dst, Operand a, Operand b) {
    return emitOp(&IRBuilder::muli64, dst, a, b);
  }

  RegisterRange sgprPair() { return builder.allocateSgpr(2, 2); }
  RegisterRange vgprPair() { return builder.allocateVgpr(2, 2); }
};

// How the destination sits relative to the operand registers.
enum class Arith64Placement {
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
  OpCase64 op;
  RegisterKind dstKind;
  InputKind aKind;
  InputKind bKind;
  Arith64Placement placement = Arith64Placement::separate;
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

// The registers that stay allocated while the op under test runs.
struct Emitted {
  std::vector<RegisterRange> owners;
  RegisterRef value;
};

struct NamedOperand {
  std::string name;
  Operand operand;
};

// The registers one operand can live in, aligned or at an odd index.
struct Arith64PlacementRegisters {
  RegisterRange vector;
  RegisterRange oddVector;
  RegisterRange scalar;
  RegisterRange oddScalar;
};

} // namespace

// The host arithmetic wraps modulo 2^64 like the device.
constexpr OpCase64 opCases64[] = {
    {"addi64", &IRBuilder::addi64,
     [](uint64_t a, uint64_t b) { return a + b; }},
    {"subi64", &IRBuilder::subi64,
     [](uint64_t a, uint64_t b) { return a - b; }},
    {"muli64", &IRBuilder::muli64,
     [](uint64_t a, uint64_t b) { return a * b; }},
};

constexpr int32_t otherLiteral = 0x0FEDCBA9;

// Builds "mnemonic op0, op1, ...".
static std::string inst(std::string_view mnemonic,
                        const std::vector<std::string> &operands) {
  std::string result(mnemonic);
  for (size_t i = 0; i < operands.size(); ++i)
    result += (i == 0 ? " " : ", ") + operands[i];
  return result;
}

// Checks that text holds exactly the expected instructions, so adjacency is
// checked too. A line may carry a trailing comment after its operands.
static void expectInstructions(const std::string &text,
                               const std::vector<std::string> &expected) {
  std::vector<std::string> lines = instructionsOf(text);
  ASSERT_EQ(lines.size(), expected.size()) << text;
  for (size_t i = 0; i < expected.size(); ++i) {
    const std::string &actual = lines[i];
    bool matches = actual.rfind(expected[i], 0) == 0 &&
                   (actual.size() == expected[i].size() ||
                    actual[expected[i].size()] == ' ');
    EXPECT_TRUE(matches) << "line " << i << ": expected '" << expected[i]
                         << "', got '" << actual << "'";
  }
}

static int instructionCount(const std::string &text) {
  return static_cast<int>(instructionsOf(text).size());
}

static std::string half(RegisterRef pair, int index) {
  return text(pair.slice(index));
}

// The ops are built from 32-bit instructions, so no 64-bit form is emitted.
static void expectNoWideForms(const std::string &out) {
  EXPECT_EQ(countLinesContaining(out, "_b64"), 0) << out;
}

TEST_F(Arith64Test, SaluCarryChainsAreAdjacent) {
  RegisterRange dstOwner = sgprPair();
  RegisterRange aOwner = sgprPair();
  RegisterRange bOwner = sgprPair();
  RegisterRef d = dstOwner.ref();
  RegisterRef a = aOwner.ref();
  RegisterRef b = bOwner.ref();
  expectInstructions(
      emitAdd(d, a, b),
      {inst("s_add_u32", {half(d, 0), half(a, 0), half(b, 0)}),
       inst("s_addc_u32", {half(d, 1), half(a, 1), half(b, 1)})});
  expectInstructions(
      emitSub(d, a, b),
      {inst("s_sub_u32", {half(d, 0), half(a, 0), half(b, 0)}),
       inst("s_subb_u32", {half(d, 1), half(a, 1), half(b, 1)})});
}

TEST_F(Arith64Test, SaluMulSequence) {
  RegisterRange dstOwner = sgprPair();
  RegisterRange aOwner = sgprPair();
  RegisterRange bOwner = sgprPair();
  RegisterRef d = dstOwner.ref();
  RegisterRef a = aOwner.ref();
  RegisterRef b = bOwner.ref();
  // The temporaries take the lowest free SGPRs.
  std::string high = "s6";
  std::string cross = "s7";
  expectInstructions(emitMul(d, a, b),
                     {inst("s_mul_hi_u32", {high, half(a, 0), half(b, 0)}),
                      inst("s_mul_i32", {cross, half(a, 0), half(b, 1)}),
                      inst("s_add_u32", {high, high, cross}),
                      inst("s_mul_i32", {cross, half(a, 1), half(b, 0)}),
                      inst("s_mul_i32", {half(d, 0), half(a, 0), half(b, 0)}),
                      inst("s_add_u32", {half(d, 1), high, cross})});
}

TEST_F(Arith64Test, ValuCarryChainsUseSgprPairCarryAndNop) {
  RegisterRange dstOwner = vgprPair();
  RegisterRange aOwner = vgprPair();
  RegisterRange bOwner = vgprPair();
  RegisterRef d = dstOwner.ref();
  RegisterRef a = aOwner.ref();
  RegisterRef b = bOwner.ref();
  // The carry takes the lowest free SGPR pair, not VCC.
  std::string carry = "s[0:1]";
  std::string out = emitAdd(d, a, b);
  expectInstructions(
      out, {inst("v_add_co_u32", {half(d, 0), carry, half(a, 0), half(b, 0)}),
            "s_nop 1",
            inst("v_addc_co_u32",
                 {half(d, 1), carry, half(a, 1), half(b, 1), carry})});
  EXPECT_EQ(countLinesContaining(out, "vcc"), 0);

  out = emitSub(d, a, b);
  expectInstructions(
      out, {inst("v_sub_co_u32", {half(d, 0), carry, half(a, 0), half(b, 0)}),
            "s_nop 1",
            inst("v_subb_co_u32",
                 {half(d, 1), carry, half(a, 1), half(b, 1), carry})});
  EXPECT_EQ(countLinesContaining(out, "vcc"), 0);
}

TEST_F(Arith64Test, ValuMulSequence) {
  RegisterRange dstOwner = vgprPair();
  RegisterRange aOwner = vgprPair();
  RegisterRange bOwner = vgprPair();
  RegisterRef d = dstOwner.ref();
  RegisterRef a = aOwner.ref();
  RegisterRef b = bOwner.ref();
  // The temporaries take the lowest free VGPRs.
  std::string high = "v6";
  std::string crossA = "v7";
  std::string crossB = "v8";
  expectInstructions(
      emitMul(d, a, b),
      {inst("v_mul_hi_u32", {high, half(a, 0), half(b, 0)}),
       inst("v_mul_lo_u32", {crossA, half(a, 0), half(b, 1)}),
       inst("v_mul_lo_u32", {crossB, half(a, 1), half(b, 0)}),
       inst("v_mul_lo_u32", {half(d, 0), half(a, 0), half(b, 0)}),
       inst("v_add3_u32", {half(d, 1), high, crossA, crossB})});
}

TEST_F(Arith64Test, ScalarOperandsIntoVgprAreLegalized) {
  RegisterRange dstOwner = vgprPair();
  RegisterRange sa = sgprPair();
  RegisterRange sb = sgprPair();
  RegisterRange vb = vgprPair();
  RegisterRef d = dstOwner.ref();

  // An SGPR low half and a VGPR low half share the constant bus legally, but
  // the carry-in read already uses the bus, so a.hi is copied.
  std::string out = emitAdd(d, sa, vb);
  expectInstructions(
      out, {inst("v_mov_b32", {"v4", half(sa.ref(), 1)}),
            inst("v_add_co_u32",
                 {half(d, 0), "s[4:5]", half(sa.ref(), 0), half(vb.ref(), 0)}),
            "s_nop 1",
            inst("v_addc_co_u32",
                 {half(d, 1), "s[4:5]", "v4", half(vb.ref(), 1), "s[4:5]"})});

  // Two distinct SGPR pairs: b.lo for the low half, then a.hi and b.hi.
  out = emitAdd(d, sa, sb);
  EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 3) << out;
  out = emitMul(d, sa, sb);
  EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 2) << out;
}

TEST_F(Arith64Test, SameSgprPairInMulIsCopiedTwice) {
  RegisterRange dstOwner = vgprPair();
  RegisterRange s = sgprPair();
  // The pair (a.lo, b.lo) is one SGPR and legal. (a.lo, b.hi) and
  // (a.hi, b.lo) are two distinct SGPRs, so b.hi and b.lo are copied.
  std::string out = emitMul(dstOwner.ref(), s, s);
  EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 2) << out;
}

TEST_F(Arith64Test, ImmediatesAreSplitIntoHalves) {
  RegisterRange dstOwner = vgprPair();
  RegisterRange aOwner = vgprPair();
  RegisterRef d = dstOwner.ref();
  RegisterRef a = aOwner.ref();
  std::string carry = "s[0:1]";

  // -5 is an inline constant and its high word is the inline constant -1.
  std::string out = emitAdd(d, a, -5);
  expectInstructions(
      out,
      {inst("v_add_co_u32", {half(d, 0), carry, half(a, 0), "-5"}), "s_nop 1",
       inst("v_addc_co_u32", {half(d, 1), carry, half(a, 1), "-1", carry})});
  expectNoWideForms(out);

  // The low word of -100 is a literal, which VOP3 cannot take.
  out = emitAdd(d, a, -100);
  EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 1) << out;
  EXPECT_EQ(countLinesContaining(out, "v_mov_b32 v4, -100"), 1) << out;
  EXPECT_EQ(countLinesContaining(out, ", -1, "), 1) << out;
  expectNoWideForms(out);

  // A positive literal has a high word of 0, which is inline.
  out = emitAdd(d, a, literal32);
  EXPECT_EQ(countLinesContaining(out, "v_mov_b32 v4, 305419896"), 1) << out;
  EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 1) << out;
  EXPECT_EQ(countLinesContaining(out, ", 0, "), 1) << out;
  expectNoWideForms(out);
}

TEST_F(Arith64Test, SaluLiteralsUseOneLiteralPerInstruction) {
  RegisterRange dstOwner = sgprPair();
  RegisterRange aOwner = sgprPair();
  RegisterRef d = dstOwner.ref();
  RegisterRef a = aOwner.ref();

  std::string out = emitAdd(d, a, literal32);
  EXPECT_EQ(instructionCount(out), 2) << out;
  EXPECT_EQ(countLinesContaining(out, "s_mov_b32"), 0) << out;
  expectNoWideForms(out);

  // Distinct literals cannot share an instruction, so b's low half is copied;
  // the high halves are both the inline constant 0.
  out = emitAdd(d, literal32, otherLiteral);
  EXPECT_EQ(instructionCount(out), 3) << out;
  EXPECT_EQ(countLinesContaining(out, "s_mov_b32"), 1) << out;
  expectNoWideForms(out);

  out = emitAdd(d, literal32, literal32);
  EXPECT_EQ(instructionCount(out), 2) << out;
  EXPECT_EQ(countLinesContaining(out, "s_mov_b32"), 0) << out;
}

TEST_F(Arith64Test, ValuMulLiteralsAreCopied) {
  RegisterRange dstOwner = vgprPair();
  RegisterRange aOwner = vgprPair();
  std::string out = emitMul(dstOwner.ref(), aOwner, literal32);
  // Only the low half is a literal; the high half is the inline constant 0.
  EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 1) << out;
  expectNoWideForms(out);
}

TEST_F(Arith64Test, DestinationAliasingNeedsNoCopies) {
  RegisterRange vd = vgprPair();
  RegisterRange vo = vgprPair();
  RegisterRange sd = sgprPair();
  RegisterRange so = sgprPair();
  for (const OpCase64 &current : {opCases64[0], opCases64[1]}) {
    SCOPED_TRACE(current.name);
    for (bool destinationFirst : {true, false}) {
      std::string valu = destinationFirst
                             ? emitOp(current.op, vd.ref(), vd, vo)
                             : emitOp(current.op, vd.ref(), vo, vd);
      EXPECT_EQ(instructionCount(valu), 3) << valu;
      EXPECT_EQ(countLinesContaining(valu, "v_mov_b32"), 0) << valu;
      std::string salu = destinationFirst
                             ? emitOp(current.op, sd.ref(), sd, so)
                             : emitOp(current.op, sd.ref(), so, sd);
      EXPECT_EQ(instructionCount(salu), 2) << salu;
      EXPECT_EQ(countLinesContaining(salu, "s_mov_b32"), 0) << salu;
    }
  }
}

TEST_F(Arith64Test, MulDestinationIsWrittenLast) {
  RegisterRange vd = vgprPair();
  RegisterRange vo = vgprPair();
  RegisterRef d = vd.ref();
  RegisterRef o = vo.ref();
  expectInstructions(
      emitMul(d, d, o),
      {inst("v_mul_hi_u32", {"v4", half(d, 0), half(o, 0)}),
       inst("v_mul_lo_u32", {"v5", half(d, 0), half(o, 1)}),
       inst("v_mul_lo_u32", {"v6", half(d, 1), half(o, 0)}),
       inst("v_mul_lo_u32", {half(d, 0), half(d, 0), half(o, 0)}),
       inst("v_add3_u32", {half(d, 1), "v4", "v5", "v6"})});

  RegisterRange sd = sgprPair();
  RegisterRange so = sgprPair();
  std::string out = emitMul(sd.ref(), sd, so);
  EXPECT_EQ(instructionCount(out), 6) << out;
  EXPECT_EQ(countLinesContaining(out, "s_mov_b32"), 0) << out;
}

TEST_F(Arith64Test, OverlappingHighHalfIsCopiedForSalu) {
  RegisterRange r = builder.allocateSgpr(3);
  RegisterRange other = sgprPair();
  RegisterRef a = r.ref(0, 2);
  RegisterRef d = r.ref(1, 2);
  RegisterRef b = other.ref();
  // dst.lo is a.hi, which the high instruction still reads.
  std::string copy = "s3";
  expectInstructions(emitAdd(d, a, b),
                     {inst("s_mov_b32", {copy, half(a, 1)}),
                      inst("s_add_u32", {half(d, 0), half(a, 0), half(b, 0)}),
                      inst("s_addc_u32", {half(d, 1), copy, half(b, 1)})});
  expectInstructions(emitSub(d, a, b),
                     {inst("s_mov_b32", {copy, half(a, 1)}),
                      inst("s_sub_u32", {half(d, 0), half(a, 0), half(b, 0)}),
                      inst("s_subb_u32", {half(d, 1), copy, half(b, 1)})});
  // The same for b.
  expectInstructions(emitAdd(d, b, a),
                     {inst("s_mov_b32", {copy, half(a, 1)}),
                      inst("s_add_u32", {half(d, 0), half(b, 0), half(a, 0)}),
                      inst("s_addc_u32", {half(d, 1), half(b, 1), copy})});
}

TEST_F(Arith64Test, OverlappingHighHalfIsCopiedForValu) {
  RegisterRange r = builder.allocateVgpr(3);
  RegisterRange other = vgprPair();
  RegisterRef a = r.ref(0, 2);
  RegisterRef d = r.ref(1, 2);
  RegisterRef b = other.ref();
  std::string copy = "v3";
  std::string carry = "s[0:1]";
  expectInstructions(
      emitAdd(d, a, b),
      {inst("v_mov_b32", {copy, half(a, 1)}),
       inst("v_add_co_u32", {half(d, 0), carry, half(a, 0), half(b, 0)}),
       "s_nop 1",
       inst("v_addc_co_u32", {half(d, 1), carry, copy, half(b, 1), carry})});
  expectInstructions(
      emitSub(d, a, b),
      {inst("v_mov_b32", {copy, half(a, 1)}),
       inst("v_sub_co_u32", {half(d, 0), carry, half(a, 0), half(b, 0)}),
       "s_nop 1",
       inst("v_subb_co_u32", {half(d, 1), carry, copy, half(b, 1), carry})});
}

TEST_F(Arith64Test, OverlapReadingOnlyLowHalfNeedsNoCopy) {
  RegisterRange vr = builder.allocateVgpr(3);
  RegisterRange vo = vgprPair();
  // dst.hi is a.lo, which only the low instruction reads.
  std::string out = emitAdd(vr.ref(0, 2), vr.ref(1, 2), vo);
  EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 0) << out;
  EXPECT_EQ(instructionCount(out), 3) << out;

  RegisterRange sr = builder.allocateSgpr(3);
  RegisterRange so = sgprPair();
  out = emitAdd(sr.ref(0, 2), sr.ref(1, 2), so);
  EXPECT_EQ(countLinesContaining(out, "s_mov_b32"), 0) << out;
  EXPECT_EQ(instructionCount(out), 2) << out;
}

TEST_F(Arith64Test, MisalignedPairsNeedNoCopies) {
  RegisterRange vr = builder.allocateVgpr(3, 2);
  RegisterRange va = builder.allocateVgpr(3, 2);
  RegisterRange vb = builder.allocateVgpr(3, 2);
  RegisterRange sr = builder.allocateSgpr(3, 2);
  RegisterRange sa = builder.allocateSgpr(3, 2);
  RegisterRange sb = builder.allocateSgpr(3, 2);
  RegisterRef vd = vr.ref(1, 2);
  RegisterRef vx = va.ref(1, 2);
  RegisterRef vy = vb.ref(1, 2);
  RegisterRef sd = sr.ref(1, 2);
  RegisterRef sx = sa.ref(1, 2);
  RegisterRef sy = sb.ref(1, 2);

  std::string out = emitAdd(vd, vx, vy);
  EXPECT_EQ(instructionCount(out), 3) << out;
  EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 0) << out;
  out = emitMul(vd, vx, vy);
  EXPECT_EQ(instructionCount(out), 5) << out;
  out = emitSub(sd, sx, sy);
  EXPECT_EQ(instructionCount(out), 2) << out;
  out = emitMul(sd, sx, sy);
  EXPECT_EQ(instructionCount(out), 6) << out;
  EXPECT_EQ(countLinesContaining(out, "s_mov_b32"), 0) << out;
  // An odd SGPR pair into a VGPR destination needs only the carry-in copy.
  out = emitAdd(vd, sx, vy);
  EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 1) << out;
}

TEST_F(Arith64Test, TemporariesAreReleased) {
  RegisterRange dst = vgprPair();
  RegisterRange a = sgprPair();
  RegisterRange b = sgprPair();
  // Two copies of b's halves, then high, crossA and crossB.
  emitMul(dst.ref(), a, b);
  EXPECT_EQ(allocator.highWaterMark(RegisterKind::vgpr), 7);
  RegisterRange next = builder.allocateVgpr(1);
  EXPECT_EQ(next.start(), 2);
  next.release();

  // Three VGPR copies and the SGPR carry pair.
  emitAdd(dst.ref(), a, b);
  EXPECT_EQ(allocator.highWaterMark(RegisterKind::sgpr), 6);
  RegisterRange nextPair = builder.allocateSgpr(2, 2);
  EXPECT_EQ(nextPair.start(), 4);
}

TEST_F(Arith64Test, InvalidOperandsThrowWithoutEmitting) {
  RegisterRange sdst = sgprPair();
  RegisterRange vdst = vgprPair();
  RegisterRange spair = sgprPair();
  RegisterRange vpair = vgprPair();
  RegisterRange acc = builder.allocateAgpr(2, 2);
  RegisterRange single = builder.allocateVgpr(1);
  RegisterRange triple = builder.allocateVgpr(3);
  for (const OpCase64 &current : opCases64) {
    SCOPED_TRACE(current.name);
    BuilderOp op = current.op;
    EXPECT_THROW((builder.*op)(sdst.ref(), vpair, spair),
                 std::invalid_argument);
    EXPECT_THROW((builder.*op)(sdst.ref(), spair, vpair),
                 std::invalid_argument);
    EXPECT_THROW((builder.*op)(acc.ref(), spair, spair), std::invalid_argument);
    EXPECT_THROW((builder.*op)(vdst.ref(), acc, spair), std::invalid_argument);
    EXPECT_THROW((builder.*op)(vdst.ref(), spair, acc), std::invalid_argument);
    EXPECT_THROW((builder.*op)(single.ref(), vpair, spair),
                 std::invalid_argument);
    EXPECT_THROW((builder.*op)(vdst.ref(), single, spair),
                 std::invalid_argument);
    EXPECT_THROW((builder.*op)(vdst.ref(), spair, single),
                 std::invalid_argument);
    EXPECT_THROW((builder.*op)(triple.ref(), vpair, spair),
                 std::invalid_argument);
  }
  EXPECT_EQ(body.toString(), "");
}

// Immediates are int32 values sign-extended to 64 bits: inline constants,
// positive and negative literals, and the extremes.
constexpr int32_t arith64Immediates[] = {
    0,   1,    -1,        64,        65,         -16,
    -17, -100, INT32_MIN, INT32_MAX, 0x12345678, -0x12345678};

using Arith64Device = DeviceTest;

static Data64 makeData(int randomCount) {
  WordQuads quads = makeArithPairs64(randomCount);
  return {join(quads.aLo, quads.aHi), join(quads.bLo, quads.bHi)};
}

// Reads an operand: an immediate, or a pair built from two inputs.
static Operand makeSource(IRBuilder &builder, const Case64 &spec,
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

static Emitted emitArith64Separate(IRBuilder &builder, const Case64 &spec,
                                   const std::vector<RegisterRef> &in) {
  Emitted emitted{{}, RegisterRef{}};
  size_t next = 0;
  Operand a = makeSource(builder, spec, spec.a, in, next, emitted.owners);
  Operand b = spec.placement == Arith64Placement::sameOperand
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

static Emitted emitArith64InPlace(IRBuilder &builder, const Case64 &spec,
                                  const std::vector<RegisterRef> &in) {
  Emitted emitted{{}, RegisterRef{}};
  size_t next = 0;
  Operand a = makeSource(builder, spec, spec.a, in, next, emitted.owners);
  Operand b = makeSource(builder, spec, spec.b, in, next, emitted.owners);
  RegisterRef dst =
      spec.placement == Arith64Placement::destinationIsA ? a.reg() : b.reg();
  (builder.*spec.op.op)(dst, a, b);
  emitted.value = dst;
  return emitted;
}

// Places a in a range that also holds the destination, offset by one register.
static Emitted emitArith64Overlapping(IRBuilder &builder, const Case64 &spec,
                                      const std::vector<RegisterRef> &in) {
  Emitted emitted{{}, RegisterRef{}};
  RegisterRange range =
      allocateKind(builder, spec.aKind == lane ? vgpr : sgpr, 3);
  bool lowIsAHigh = spec.placement == Arith64Placement::lowIsAHigh;
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

static Emitted emitArith64Case(IRBuilder &builder, const Case64 &spec,
                               const std::vector<RegisterRef> &in) {
  switch (spec.placement) {
  case Arith64Placement::separate:
  case Arith64Placement::sameOperand:
    return emitArith64Separate(builder, spec, in);
  case Arith64Placement::destinationIsA:
  case Arith64Placement::destinationIsB:
    return emitArith64InPlace(builder, spec, in);
  case Arith64Placement::lowIsAHigh:
  case Arith64Placement::highIsALow:
    return emitArith64Overlapping(builder, spec, in);
  }
  return emitArith64Separate(builder, spec, in);
}

// A result is one lane of a vgpr result or one workgroup of an sgpr result.
static size_t arith64InputIndex(const Case64 &spec, InputKind kind,
                                size_t result) {
  bool perResult = spec.dstKind == sgpr || kind == lane;
  return perResult ? result : result / arithWorkgroupSize;
}

static uint64_t arith64OperandValue(const Case64 &spec,
                                    const std::optional<int32_t> &imm,
                                    InputKind kind, const Values &values,
                                    size_t result) {
  if (imm)
    return static_cast<uint64_t>(int64_t{*imm});
  return values[arith64InputIndex(spec, kind, result)];
}

static Values arith64ExpectedValues(const Case64 &spec, const Data64 &data,
                                    size_t results) {
  Values expected;
  for (size_t i = 0; i < results; ++i) {
    uint64_t a = arith64OperandValue(spec, spec.a, spec.aKind, data.a, i);
    bool same = spec.placement == Arith64Placement::sameOperand;
    uint64_t b =
        same ? a : arith64OperandValue(spec, spec.b, spec.bKind, data.b, i);
    expected.push_back(spec.op.host(a, b));
  }
  return expected;
}

static void addPairInputs(KernelInputs &inputs, InputKind kind,
                          const Values &values, size_t laneCount,
                          size_t groupCount) {
  addInput(inputs, kind, lowWords(values), laneCount, groupCount);
  addInput(inputs, kind, highWords(values), laneCount, groupCount);
}

static KernelInputs makeArith64Inputs(const Case64 &spec, const Data64 &data,
                                      size_t groupCount) {
  size_t laneCount = data.a.size();
  KernelInputs inputs;
  if (!spec.a)
    addPairInputs(inputs, spec.aKind, data.a, laneCount, groupCount);
  if (!spec.b && spec.placement != Arith64Placement::sameOperand)
    addPairInputs(inputs, spec.bKind, data.b, laneCount, groupCount);
  return inputs;
}

// Runs the op and checks it, through two runs that each store one half.
static void expectCase(const Case64 &spec, const Data64 &data) {
  size_t lanes = data.a.size();
  size_t groups = spec.dstKind == sgpr ? lanes : lanes / arithWorkgroupSize;
  size_t results = spec.dstKind == sgpr ? groups : lanes;
  KernelInputs inputs = makeArith64Inputs(spec, data, groups);
  Values expected = arith64ExpectedValues(spec, data, results);
  for (int half = 0; half < 2; ++half) {
    SCOPED_TRACE(std::string(spec.op.name) +
                 (half == 0 ? " low half" : " high half"));
    ArithKernelSpec kernel{inputs.kinds, spec.dstKind,
                           [spec, half](IRBuilder &builder, RegisterRef dst,
                                        const std::vector<RegisterRef> &in) {
                             Emitted emitted =
                                 emitArith64Case(builder, spec, in);
                             builder.movb32(dst, emitted.value.slice(half));
                           }};
    Words words = half == 0 ? lowWords(expected) : highWords(expected);
    expectArithResults(kernel, static_cast<int>(groups), inputs.words, words);
  }
}

static Case64 makeArith64Case(const OpCase64 &op, RegisterKind dstKind,
                              InputKind aKind, InputKind bKind) {
  return {op, dstKind, aKind, bKind};
}

TEST_F(Arith64Device, ValuLaneOperands) {
  Data64 data = makeData(4096);
  for (const OpCase64 &op : opCases64)
    expectCase(makeArith64Case(op, vgpr, lane, lane), data);
}

TEST_F(Arith64Device, SaluUniformOperands) {
  Data64 data = makeData(256);
  for (const OpCase64 &op : opCases64)
    expectCase(makeArith64Case(op, sgpr, uniform, uniform), data);
}

TEST_F(Arith64Device, ScalarsIntoVgpr) {
  Data64 data = makeData(256);
  for (const OpCase64 &op : opCases64)
    expectCase(makeArith64Case(op, vgpr, uniform, uniform), data);
}

TEST_F(Arith64Device, MixedScalarAndVector) {
  Data64 data = makeData(4096);
  for (const OpCase64 &op : opCases64) {
    expectCase(makeArith64Case(op, vgpr, uniform, lane), data);
    expectCase(makeArith64Case(op, vgpr, lane, uniform), data);
  }
}

TEST_F(Arith64Device, MisalignedPairs) {
  Data64 data = makeData(1024);
  for (const OpCase64 &op : opCases64) {
    for (bool dstMisaligned : {false, true}) {
      for (bool operandsMisaligned : {false, true}) {
        SCOPED_TRACE(
            std::string(dstMisaligned ? "misaligned dst, " : "aligned dst, ") +
            (operandsMisaligned ? "misaligned operands" : "aligned operands"));
        for (Case64 spec : {makeArith64Case(op, vgpr, lane, lane),
                            makeArith64Case(op, sgpr, uniform, uniform)}) {
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
  for (const OpCase64 &op : opCases64) {
    for (int32_t immediate : arith64Immediates) {
      SCOPED_TRACE(std::string(op.name) + " immediate " +
                   std::to_string(immediate));
      for (Case64 spec : {makeArith64Case(op, vgpr, lane, lane),
                          makeArith64Case(op, sgpr, uniform, uniform)}) {
        Case64 immediateFirst = spec;
        immediateFirst.a = immediate;
        expectCase(immediateFirst, data);
        spec.b = immediate;
        expectCase(spec, data);
      }
      Case64 scalarsIntoVgpr = makeArith64Case(op, vgpr, uniform, uniform);
      scalarsIntoVgpr.b = immediate;
      expectCase(scalarsIntoVgpr, data);
    }
  }
}

TEST_F(Arith64Device, DestinationAliasing) {
  Data64 data = makeData(256);
  for (const OpCase64 &op : opCases64) {
    for (Arith64Placement placement :
         {Arith64Placement::destinationIsA, Arith64Placement::destinationIsB,
          Arith64Placement::lowIsAHigh, Arith64Placement::highIsALow,
          Arith64Placement::sameOperand}) {
      SCOPED_TRACE(std::string(op.name) + " placement " +
                   std::to_string(static_cast<int>(placement)));
      for (Case64 spec : {makeArith64Case(op, vgpr, lane, lane),
                          makeArith64Case(op, sgpr, uniform, uniform)}) {
        spec.placement = placement;
        expectCase(spec, data);
      }
    }
  }
}

static Arith64PlacementRegisters
allocatePlacementRegisters(IRBuilder &builder) {
  return {builder.allocateVgpr(2, 2), builder.allocateVgpr(3, 2),
          builder.allocateSgpr(2, 2), builder.allocateSgpr(3, 2)};
}

static std::vector<NamedOperand>
placementsOf(const Arith64PlacementRegisters &regs) {
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
static void expectPlacementsAssemble(const OpCase64 &op, RegisterKind dstKind) {
  Target target = Target::get("gfx950");
  Kernel kernel(target, "codegen_placement", arithWorkgroupSize);
  IRBuilder &builder = kernel.builder();
  Arith64PlacementRegisters aRegs = allocatePlacementRegisters(builder);
  Arith64PlacementRegisters bRegs = allocatePlacementRegisters(builder);
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
  for (const OpCase64 &op : opCases64) {
    expectPlacementsAssemble(op, vgpr);
    expectPlacementsAssemble(op, sgpr);
  }
}

// Division.

namespace {

enum class Form { scalar, vector, mixed };

struct DivRegisters {
  RegisterRange dst;
  RegisterRange a;
  RegisterRange b;
};

class DivTest : public BuilderTest {
protected:
  std::string emitDiv(RegisterRef dst, Operand a, Operand b, bool useIEEE) {
    return emitted(
        [&](IRBuilder &local) { local.divsi32(dst, a, b, useIEEE); });
  }

  RegisterRange allocate(RegisterKind kind) {
    return kind == RegisterKind::sgpr ? builder.allocateSgpr(1)
                                      : builder.allocateVgpr(1);
  }

  DivRegisters makeRegisters(Form form) {
    RegisterKind dstKind =
        form == Form::scalar ? RegisterKind::sgpr : RegisterKind::vgpr;
    RegisterKind sourceKind =
        form == Form::vector ? RegisterKind::vgpr : RegisterKind::sgpr;
    return {allocate(dstKind), allocate(sourceKind), allocate(sourceKind)};
  }

  std::string emitForm(Form form, bool useIEEE) {
    DivRegisters regs = makeRegisters(form);
    return emitDiv(regs.dst.ref(), regs.a, regs.b, useIEEE);
  }

  int nextFreeIndex(RegisterKind kind) { return allocate(kind).start(); }
};

} // namespace

static std::string mnemonicOf(const std::string &instruction) {
  return instruction.substr(0, instruction.find(' '));
}

// The operands of an instruction, without a trailing comment, split at commas.
static std::vector<std::string> operandsOf(const std::string &instruction) {
  std::vector<std::string> operands;
  size_t space = instruction.find(' ');
  if (space == std::string::npos)
    return operands;
  std::string rest = instruction.substr(space + 1);
  rest = rest.substr(0, rest.find("//"));
  std::istringstream stream(rest);
  for (std::string operand; std::getline(stream, operand, ',');) {
    size_t first = operand.find_first_not_of(" \t");
    size_t last = operand.find_last_not_of(" \t");
    if (first != std::string::npos)
      operands.push_back(operand.substr(first, last - first + 1));
  }
  return operands;
}

// The first operand after the mnemonic, or empty for instructions with no
// destination.
static std::string destinationOf(const std::string &instruction) {
  std::string mnemonic = mnemonicOf(instruction);
  if (mnemonic == "s_nop" || mnemonic.rfind("s_cmp", 0) == 0)
    return "";
  std::vector<std::string> operands = operandsOf(instruction);
  return operands.empty() ? "" : operands.front();
}

// The source operands without negation or absolute-value modifiers.
static std::vector<std::string> sourcesOf(const std::string &instruction) {
  std::vector<std::string> sources = operandsOf(instruction);
  if (!sources.empty())
    sources.erase(sources.begin());
  for (std::string &source : sources) {
    if (source.rfind("-", 0) == 0)
      source = source.substr(1);
    if (source.rfind("abs(", 0) == 0)
      source = source.substr(4, source.size() - 5);
  }
  return sources;
}

static int occurrences(const std::string &text, std::string_view needle) {
  int count = 0;
  for (size_t at = text.find(needle); at != std::string::npos;
       at = text.find(needle, at + needle.size()))
    ++count;
  return count;
}

static int indexOfMnemonic(const std::vector<std::string> &instructions,
                           std::string_view mnemonic, int from = 0) {
  for (int i = from; i < static_cast<int>(instructions.size()); ++i)
    if (mnemonicOf(instructions[i]) == mnemonic)
      return i;
  return -1;
}

constexpr Form allForms[] = {Form::scalar, Form::vector, Form::mixed};

static const char *formName(Form form) {
  switch (form) {
  case Form::scalar:
    return "sgpr dst with sgpr operands";
  case Form::vector:
    return "vgpr dst with vgpr operands";
  case Form::mixed:
    return "vgpr dst with sgpr operands";
  }
  return "";
}

TEST_F(DivTest, ExactSaluForScalarOperands) {
  std::string out = emitForm(Form::scalar, true);
  expectLinesInOrder(out,
                     {"s_abs_i32",     "s_abs_i32",           "s_xor_b32",
                      "s_ashr_i32",    "v_cvt_f32_u32",       "v_rcp_iflag_f32",
                      "s_sub_u32",     "v_mul_f32",           "v_cvt_u32_f32",
                      "s_nop 0",       "v_readfirstlane_b32", "s_mul_i32",
                      "s_mul_hi_u32",  "s_add_u32",           "s_mul_hi_u32",
                      "s_mul_i32",     "s_sub_u32",           "s_add_u32",
                      "s_sub_u32",     "s_cmp_ge_u32",        "s_cselect_b32",
                      "s_cselect_b32", "s_add_u32",           "s_cmp_ge_u32",
                      "s_cselect_b32", "s_xor_b32",           "s_sub_u32"});
  EXPECT_EQ(countLinesContaining(out, "s_ashr_i32"), 1);
  EXPECT_EQ(countLinesContaining(out, ", 31"), 1);
  EXPECT_EQ(countLinesContaining(out, "0x4f7ffffe"), 1);
  EXPECT_EQ(countLinesContaining(out, "v_mul_lo"), 0);
  EXPECT_EQ(countLinesContaining(out, "v_mul_hi"), 0);
  EXPECT_EQ(countLinesContaining(out, "v_cndmask"), 0);
  std::vector<std::string> instructions = instructionsOf(out);
  EXPECT_EQ(mnemonicOf(instructions.back()), "s_sub_u32");
}

TEST_F(DivTest, ExactValuForVectorDestination) {
  for (Form form : {Form::vector, Form::mixed}) {
    SCOPED_TRACE(formName(form));
    std::string out = emitForm(form, true);
    expectLinesInOrder(
        out,
        {"v_sub_u32",    "v_max_i32",     "v_sub_u32",     "v_max_i32",
         "v_xor_b32",    "v_ashrrev_i32", "v_cvt_f32_u32", "v_rcp_iflag_f32",
         "v_sub_u32",    "v_mul_f32",     "v_cvt_u32_f32", "v_mul_lo_u32",
         "v_mul_hi_u32", "v_add_u32",     "v_mul_hi_u32",  "v_mul_lo_u32",
         "v_sub_u32",    "v_add_u32",     "v_sub_u32",     "v_cmp_ge_u32",
         "s_nop 1",      "v_cndmask_b32", "v_cndmask_b32", "v_add_u32",
         "v_cmp_ge_u32", "s_nop 1",       "v_cndmask_b32", "v_xor_b32",
         "v_sub_u32"});
    EXPECT_EQ(countLinesContaining(out, "v_ashrrev_i32"), 1);
    EXPECT_EQ(countLinesContaining(out, ", 31"), 1);
    EXPECT_EQ(countLinesContaining(out, "v_readfirstlane"), 0);
    for (const std::string &instruction : instructionsOf(out)) {
      std::string mnemonic = mnemonicOf(instruction);
      bool scalar = mnemonic.rfind("s_", 0) == 0;
      EXPECT_TRUE(!scalar || mnemonic == "s_nop") << instruction;
    }
    std::vector<std::string> instructions = instructionsOf(out);
    EXPECT_EQ(mnemonicOf(instructions.back()), "v_sub_u32");
  }
}

TEST_F(DivTest, FastSaluForScalarOperands) {
  std::string out = emitForm(Form::scalar, false);
  expectLinesInOrder(out, {"s_xor_b32", "s_ashr_i32", "s_or_b32",
                           "v_cvt_f32_i32", "v_rcp_iflag_f32", "v_cvt_f32_i32",
                           "v_mul_f32", "v_cvt_i32_f32", "v_cvt_f32_i32",
                           "v_fma_f32", "v_cmp_ge_f32", "v_readfirstlane_b32",
                           "s_and_b64", "s_cselect_b32", "s_add_u32"});
  EXPECT_EQ(countLinesContaining(out, "s_ashr_i32"), 1);
  EXPECT_EQ(countLinesContaining(out, ", 30"), 1);
  EXPECT_EQ(countLinesContaining(out, "v_mul_hi"), 0);
  for (const std::string &instruction : instructionsOf(out)) {
    if (mnemonicOf(instruction) == "v_cmp_ge_f32")
      EXPECT_EQ(occurrences(instruction, "abs("), 2) << instruction;
    if (mnemonicOf(instruction) == "v_fma_f32")
      EXPECT_NE(instruction.find(", -v"), std::string::npos) << instruction;
    if (mnemonicOf(instruction) == "s_and_b64")
      EXPECT_NE(instruction.find("exec"), std::string::npos) << instruction;
  }
  EXPECT_EQ(countLinesContaining(out, "abs("), 1);
}

TEST_F(DivTest, FastValuForVectorDestination) {
  for (Form form : {Form::vector, Form::mixed}) {
    SCOPED_TRACE(formName(form));
    std::string out = emitForm(form, false);
    expectLinesInOrder(out, {"v_xor_b32", "v_ashrrev_i32", "v_or_b32",
                             "v_cvt_f32_i32", "v_rcp_iflag_f32",
                             "v_cvt_f32_i32", "v_mul_f32", "v_cvt_i32_f32",
                             "v_cvt_f32_i32", "v_fma_f32", "v_cmp_ge_f32",
                             "s_nop 1", "v_cndmask_b32", "v_add_u32"});
    EXPECT_EQ(countLinesContaining(out, ", 30"), 1);
    EXPECT_EQ(countLinesContaining(out, "v_mul_hi"), 0);
    EXPECT_EQ(countLinesContaining(out, "abs("), 1);
    EXPECT_EQ(countLinesContaining(out, "v_readfirstlane"), 0);
    std::vector<std::string> instructions = instructionsOf(out);
    int fma = indexOfMnemonic(instructions, "v_fma_f32");
    ASSERT_GE(fma, 0);
    EXPECT_NE(instructions[fma].find(", -v"), std::string::npos);
  }
}

TEST_F(DivTest, NeverWritesExecOrVcc) {
  for (bool useIEEE : {true, false})
    for (Form form : allForms) {
      SCOPED_TRACE(std::string(formName(form)) +
                   (useIEEE ? " exact" : " fast"));
      std::string out = emitForm(form, useIEEE);
      for (const std::string &instruction : instructionsOf(out)) {
        std::string dst = destinationOf(instruction);
        EXPECT_NE(dst, "exec") << instruction;
        EXPECT_NE(dst, "exec_lo") << instruction;
        EXPECT_NE(dst, "vcc") << instruction;
      }
      EXPECT_EQ(countLinesContaining(out, "v_cmpx"), 0);
      EXPECT_EQ(countLinesContaining(out, "vcc"), 0);
    }
}

// The reciprocal's reader must not be the next instruction, since a trans
// result needs a wait state.
static void
expectReciprocalHazardCovered(const std::vector<std::string> &instructions) {
  int rcp = indexOfMnemonic(instructions, "v_rcp_iflag_f32");
  ASSERT_GE(rcp, 0);
  ASSERT_LT(rcp + 1, static_cast<int>(instructions.size()));
  const std::string &next = instructions[rcp + 1];
  EXPECT_NE(mnemonicOf(next), "s_nop") << next;
  std::string reciprocal = destinationOf(instructions[rcp]);
  for (const std::string &source : sourcesOf(next))
    EXPECT_NE(source, reciprocal) << next;
}

static void
expectMaskNopBeforeCndmask(const std::vector<std::string> &instructions) {
  int checked = 0;
  for (int i = 0; i < static_cast<int>(instructions.size()); ++i) {
    if (mnemonicOf(instructions[i]).rfind("v_cmp_", 0) != 0)
      continue;
    ASSERT_LT(i + 1, static_cast<int>(instructions.size()));
    EXPECT_EQ(instructions[i + 1].rfind("s_nop 1", 0), 0u) << instructions[i];
    EXPECT_GE(indexOfMnemonic(instructions, "v_cndmask_b32", i + 1), 0);
    ++checked;
  }
  EXPECT_GT(checked, 0);
}

TEST_F(DivTest, HazardWaitStates) {
  for (bool useIEEE : {true, false})
    for (Form form : allForms) {
      SCOPED_TRACE(std::string(formName(form)) +
                   (useIEEE ? " exact" : " fast"));
      std::vector<std::string> instructions =
          instructionsOf(emitForm(form, useIEEE));
      expectReciprocalHazardCovered(instructions);
      if (form != Form::scalar) {
        expectMaskNopBeforeCndmask(instructions);
        continue;
      }
      int readLane = indexOfMnemonic(instructions, "v_readfirstlane_b32");
      ASSERT_GT(readLane, 0);
      if (useIEEE) {
        EXPECT_EQ(instructions[readLane - 1].rfind("s_nop 0", 0), 0u);
        continue;
      }
      int convert = indexOfMnemonic(instructions, "v_cvt_i32_f32");
      ASSERT_GE(convert, 0);
      EXPECT_GT(readLane - convert, 1);
      EXPECT_EQ(sourcesOf(instructions[readLane]).front(),
                destinationOf(instructions[convert]));
    }
}

// Every VOP3-only or VOP3-promoted line must be free of the literal.
static void expectLiteralOnlyInMoves(const std::string &out,
                                     const char *literal) {
  for (const std::string &instruction : instructionsOf(out)) {
    std::string mnemonic = mnemonicOf(instruction);
    if (mnemonic == "v_mul_lo_u32" || mnemonic == "v_mul_hi_u32" ||
        mnemonic == "v_sub_u32" || mnemonic == "v_xor_b32")
      EXPECT_EQ(instruction.find(literal), std::string::npos) << instruction;
  }
}

TEST_F(DivTest, ImmediateOperandsInVectorForm) {
  RegisterRange dst = builder.allocateVgpr(1);
  RegisterRange value = builder.allocateVgpr(1);
  for (bool useIEEE : {true, false}) {
    SCOPED_TRACE(useIEEE ? "exact" : "fast");
    std::string divisor = emitDiv(dst.ref(), value, literalA, useIEEE);
    EXPECT_EQ(countLinesContaining(divisor, "v_mov_b32"), 1);
    expectLiteralOnlyInMoves(divisor, literalAText);
    std::string dividend = emitDiv(dst.ref(), literalB, value, useIEEE);
    EXPECT_EQ(countLinesContaining(dividend, "v_mov_b32"), 1);
    expectLiteralOnlyInMoves(dividend, literalBText);
    for (int32_t inlineValue : {7, -5})
      EXPECT_EQ(countLinesContaining(
                    emitDiv(dst.ref(), value, inlineValue, useIEEE), "v_mov"),
                0)
          << inlineValue;
  }
}

TEST_F(DivTest, ImmediateOperandsInExactScalarForm) {
  RegisterRange dst = builder.allocateSgpr(1);
  RegisterRange value = builder.allocateSgpr(1);
  std::string out = emitDiv(dst.ref(), value, 7, true);
  expectLinesInOrder(out, {"s_mov_b32", "s_abs_i32"});
  EXPECT_EQ(countLinesContaining(out, "s_mov_b32"), 1);
  out = emitDiv(dst.ref(), 7, value, true);
  expectLinesInOrder(out, {"s_mov_b32", "s_abs_i32"});
  EXPECT_EQ(countLinesContaining(out, "s_mov_b32"), 1);
  EXPECT_EQ(
      countLinesContaining(emitDiv(dst.ref(), literalA, -5, true), "s_mov_b32"),
      2);
}

TEST_F(DivTest, ImmediateOperandsInFastScalarForm) {
  RegisterRange dst = builder.allocateSgpr(1);
  RegisterRange value = builder.allocateSgpr(1);
  std::string out = emitDiv(dst.ref(), value, literalA, false);
  EXPECT_EQ(countLinesContaining(out, "s_mov_b32"), 0);
  int convertsWithLiteral = 0;
  int xorsWithLiteral = 0;
  for (const std::string &instruction : instructionsOf(out)) {
    bool hasLiteral = instruction.find(literalAText) != std::string::npos;
    if (mnemonicOf(instruction) == "v_cvt_f32_i32" && hasLiteral)
      ++convertsWithLiteral;
    if (mnemonicOf(instruction) == "s_xor_b32" && hasLiteral)
      ++xorsWithLiteral;
  }
  EXPECT_EQ(convertsWithLiteral, 1);
  EXPECT_EQ(xorsWithLiteral, 1);
  // Two distinct literals cannot share s_xor, so b is copied.
  out = emitDiv(dst.ref(), literalA, literalB, false);
  EXPECT_EQ(countLinesContaining(out, "s_mov_b32"), 1);
  std::vector<std::string> instructions = instructionsOf(out);
  int copy = indexOfMnemonic(instructions, "s_mov_b32");
  ASSERT_GE(copy, 0);
  EXPECT_NE(instructions[copy].find(literalBText), std::string::npos);
}

TEST_F(DivTest, DestinationAliasesOperand) {
  for (bool useIEEE : {true, false})
    for (Form form : {Form::scalar, Form::vector}) {
      for (bool aliasFirst : {true, false}) {
        SCOPED_TRACE(std::string(formName(form)) +
                     (useIEEE ? " exact" : " fast") +
                     (aliasFirst ? " dst is a" : " dst is b"));
        DivRegisters regs = makeRegisters(form);
        RegisterRef dst = aliasFirst ? regs.a.ref() : regs.b.ref();
        std::vector<std::string> instructions =
            instructionsOf(emitDiv(dst, regs.a, regs.b, useIEEE));
        ASSERT_FALSE(instructions.empty());
        EXPECT_EQ(destinationOf(instructions.back()), text(dst));
        for (size_t i = 0; i + 1 < instructions.size(); ++i)
          EXPECT_NE(destinationOf(instructions[i]), text(dst))
              << instructions[i];
      }
    }
}

TEST_F(DivTest, InvalidOperandsThrowWithoutEmitting) {
  RegisterRange sdst = builder.allocateSgpr(1);
  RegisterRange vdst = builder.allocateVgpr(1);
  RegisterRange sa = builder.allocateSgpr(1);
  RegisterRange va = builder.allocateVgpr(1);
  RegisterRange vb = builder.allocateVgpr(1);
  RegisterRange acc = builder.allocateAgpr(1);
  RegisterRange pair = builder.allocateVgpr(2, 2);
  for (bool useIEEE : {true, false}) {
    SCOPED_TRACE(useIEEE ? "exact" : "fast");
    expectRejected([&] { builder.divsi32(acc.ref(), va, vb, useIEEE); });
    expectRejected([&] { builder.divsi32(vdst.ref(), acc, vb, useIEEE); });
    expectRejected([&] { builder.divsi32(vdst.ref(), va, acc, useIEEE); });
    expectRejected([&] { builder.divsi32(pair.ref(), va, vb, useIEEE); });
    expectRejected([&] { builder.divsi32(vdst.ref(), pair, vb, useIEEE); });
    expectRejected([&] { builder.divsi32(vdst.ref(), va, pair, useIEEE); });
    expectRejected([&] { builder.divsi32(sdst.ref(), va, sa, useIEEE); });
    expectRejected([&] { builder.divsi32(sdst.ref(), sa, vb, useIEEE); });
  }
}

TEST_F(DivTest, ScalarDestinationFromVectorOperandMessage) {
  RegisterRange sdst = builder.allocateSgpr(1);
  RegisterRange sa = builder.allocateSgpr(1);
  RegisterRange vb = builder.allocateVgpr(1);
  std::string expected = "divsi32 cannot write sgpr destination " + text(sdst) +
                         " from vgpr operand " + text(vb);
  try {
    builder.divsi32(sdst.ref(), sa, vb, true);
    FAIL() << "expected std::invalid_argument";
  } catch (const std::invalid_argument &error) {
    EXPECT_EQ(std::string(error.what()), expected);
  }
}

TEST_F(DivTest, ReleasesTemporaries) {
  for (bool useIEEE : {true, false})
    for (Form form : allForms) {
      SCOPED_TRACE(std::string(formName(form)) +
                   (useIEEE ? " exact" : " fast"));
      DivRegisters regs = makeRegisters(form);
      int vgprsBefore = nextFreeIndex(RegisterKind::vgpr);
      int sgprsBefore = nextFreeIndex(RegisterKind::sgpr);
      emitDiv(regs.dst.ref(), regs.a, regs.b, useIEEE);
      EXPECT_EQ(nextFreeIndex(RegisterKind::vgpr), vgprsBefore);
      EXPECT_EQ(nextFreeIndex(RegisterKind::sgpr), sgprsBefore);
      // Literals take temporaries too.
      emitDiv(regs.dst.ref(), literalA, literalB, useIEEE);
      EXPECT_EQ(nextFreeIndex(RegisterKind::vgpr), vgprsBefore);
      EXPECT_EQ(nextFreeIndex(RegisterKind::sgpr), sgprsBefore);
    }
}

constexpr uint32_t divRandomSeed = 20240607;
// The fast path is exact for magnitudes below this bound, LLVM's signed 24-bit
// criterion.
constexpr int64_t fastBound = int64_t{1} << 23;
constexpr int exactRandomCount = 131072;
constexpr int fastBoundaryCount = 262144;
constexpr int fastRandomCount = 1048576;

using DivDevice = DeviceTest;

// The quotient truncates toward zero and INT32_MIN / -1 wraps.
static uint32_t hostDiv(uint32_t a, uint32_t b) {
  int32_t x = static_cast<int32_t>(a);
  int32_t y = static_cast<int32_t>(b);
  if (x == INT32_MIN && y == -1)
    return a;
  return static_cast<uint32_t>(x / y);
}

// A zero divisor is replaced by 1.
static void addPair(WordPairs &pairs, int64_t a, int64_t b) {
  pairs.a.push_back(static_cast<uint32_t>(a));
  pairs.b.push_back(b == 0 ? 1u : static_cast<uint32_t>(b));
}

// A value with 0 to maxBits - 1 low bits kept, so that every magnitude occurs.
static uint32_t randomMagnitude(std::mt19937 &engine, int maxBits) {
  uint32_t mask = maxBits == 32 ? ~0u : (1u << maxBits) - 1;
  return (engine() & mask) >> (engine() % maxBits);
}

static int64_t withRandomSign(std::mt19937 &engine, int64_t magnitude) {
  return (engine() & 1) != 0 ? -magnitude : magnitude;
}

// Pads with random pairs to whole workgroups of lanes.
static void padToWorkgroups(WordPairs &pairs, std::mt19937 &engine,
                            int maxBits) {
  while (pairs.a.size() % arithWorkgroupSize != 0)
    addPair(pairs, withRandomSign(engine, randomMagnitude(engine, maxBits)),
            withRandomSign(engine, randomMagnitude(engine, maxBits)));
}

static std::vector<int64_t> exactEdges() {
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
static void addNeighborDivisors(WordPairs &pairs, int64_t a) {
  for (int64_t b : {a - 1, a, a + 1, -a, a / 2, a * 2 + 1, -(a * 2 + 1)}) {
    if (b < INT32_MIN || b > INT32_MAX)
      continue;
    addPair(pairs, a, b);
  }
}

static void addExactEdgePairs(WordPairs &pairs) {
  std::vector<int64_t> edges = exactEdges();
  for (int64_t a : edges) {
    for (int64_t b : edges)
      addPair(pairs, a, b);
    addNeighborDivisors(pairs, a);
  }
}

// Random 32-bit pairs, then a random with a small divisor, then both shifted.
static void addExactRandomPairs(WordPairs &pairs, std::mt19937 &engine,
                                int count) {
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

static const WordPairs &exactPairs() {
  static const WordPairs pairs = [] {
    WordPairs result;
    std::mt19937 engine(divRandomSeed);
    addExactEdgePairs(result);
    addExactRandomPairs(result, engine, exactRandomCount);
    padToWorkgroups(result, engine, 32);
    return result;
  }();
  return pairs;
}

static std::vector<int64_t> fastEdges() {
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

static void addFastEdgePairs(WordPairs &pairs) {
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

// WordPairs a = k * b + d for d in {-1, 0, 1}, where the exact quotient
// changes.
static void addBoundaryPairs(WordPairs &pairs, std::mt19937 &engine,
                             int count) {
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
static void addFastRandomPairs(WordPairs &pairs, std::mt19937 &engine,
                               int count) {
  for (int i = 0; i < count; ++i) {
    bool shifted = (i & 1) != 0;
    uint32_t a = shifted ? randomMagnitude(engine, 23) : engine() & 0x7FFFFF;
    uint32_t b = shifted ? randomMagnitude(engine, 23) : engine() & 0x7FFFFF;
    addPair(pairs, withRandomSign(engine, a), withRandomSign(engine, b));
  }
}

static const WordPairs &fastPairs() {
  static const WordPairs pairs = [] {
    WordPairs result;
    std::mt19937 engine(divRandomSeed + 1);
    addFastEdgePairs(result);
    addBoundaryPairs(result, engine, fastBoundaryCount);
    addFastRandomPairs(result, engine, fastRandomCount);
    padToWorkgroups(result, engine, 23);
    return result;
  }();
  return pairs;
}

static const WordPairs &pairsFor(bool useIEEE) {
  return useIEEE ? exactPairs() : fastPairs();
}

static const char *pathName(bool useIEEE) { return useIEEE ? "exact" : "fast"; }

// Every stride-th pair, so that a small sample still spans the whole set.
static WordPairs sample(const WordPairs &pairs, size_t count) {
  WordPairs result;
  size_t stride = std::max<size_t>(1, pairs.a.size() / count);
  for (size_t i = 0; i < pairs.a.size() && result.a.size() < count;
       i += stride) {
    result.a.push_back(pairs.a[i]);
    result.b.push_back(pairs.b[i]);
  }
  return result;
}

static std::vector<uint32_t> quotients(const WordPairs &pairs) {
  std::vector<uint32_t> expected;
  for (size_t i = 0; i < pairs.a.size(); ++i)
    expected.push_back(hostDiv(pairs.a[i], pairs.b[i]));
  return expected;
}

static ArithKernelSpec divSpec(InputKind aKind, InputKind bKind,
                               RegisterKind result, bool useIEEE) {
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
    const WordPairs &pairs = pairsFor(useIEEE);
    expectArithResults(
        divSpec(InputKind::lane, InputKind::lane, RegisterKind::vgpr, useIEEE),
        laneGroupCount(pairs.a), {pairs.a, pairs.b}, quotients(pairs));
  }
}

TEST_F(DivDevice, SaluUniformOperands) {
  for (bool useIEEE : {true, false}) {
    SCOPED_TRACE(pathName(useIEEE));
    WordPairs pairs = sample(pairsFor(useIEEE), 32768);
    expectArithResults(divSpec(InputKind::uniform, InputKind::uniform,
                               RegisterKind::sgpr, useIEEE),
                       static_cast<int>(pairs.a.size()), {pairs.a, pairs.b},
                       quotients(pairs));
  }
}

TEST_F(DivDevice, ScalarsIntoVgpr) {
  for (bool useIEEE : {true, false}) {
    SCOPED_TRACE(pathName(useIEEE));
    WordPairs pairs = sample(pairsFor(useIEEE), 16384);
    expectArithResults(divSpec(InputKind::uniform, InputKind::uniform,
                               RegisterKind::vgpr, useIEEE),
                       static_cast<int>(pairs.a.size()), {pairs.a, pairs.b},
                       perLane(quotients(pairs)));
  }
}

// The first lane of each workgroup supplies the uniform operand.
static std::vector<uint32_t>
firstLanes(const std::vector<uint32_t> &laneWords) {
  std::vector<uint32_t> firstWords;
  for (size_t i = 0; i < laneWords.size(); i += arithWorkgroupSize)
    firstWords.push_back(laneWords[i]);
  return firstWords;
}

TEST_F(DivDevice, MixedScalarAndVector) {
  for (bool useIEEE : {true, false}) {
    SCOPED_TRACE(pathName(useIEEE));
    WordPairs pairs = sample(pairsFor(useIEEE), 65536);
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

static std::vector<int32_t> immediatesFor(bool useIEEE) {
  std::vector<int32_t> immediates = {1,   -1, 2,  -2,  3,       7,
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
static ArithKernelSpec divImmediateSpec(InputKind kind, RegisterKind result,
                                        int32_t immediate,
                                        bool immediateIsDividend,
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

static std::vector<uint32_t>
divideByImmediate(const std::vector<uint32_t> &words, int32_t immediate,
                  bool immediateIsDividend) {
  uint32_t value = static_cast<uint32_t>(immediate);
  std::vector<uint32_t> out;
  for (uint32_t word : words)
    out.push_back(immediateIsDividend ? hostDiv(value, word)
                                      : hostDiv(word, value));
  return out;
}

// Runs the lane, uniform-into-sgpr and uniform-into-vgpr forms.
static void expectImmediateForms(const std::vector<uint32_t> &words,
                                 int32_t immediate, bool immediateIsDividend,
                                 bool useIEEE) {
  std::vector<uint32_t> expected =
      divideByImmediate(words, immediate, immediateIsDividend);
  expectArithResults(divImmediateSpec(InputKind::lane, RegisterKind::vgpr,
                                      immediate, immediateIsDividend, useIEEE),
                     laneGroupCount(words), {words}, expected);
  expectArithResults(divImmediateSpec(InputKind::uniform, RegisterKind::sgpr,
                                      immediate, immediateIsDividend, useIEEE),
                     static_cast<int>(words.size()), {words}, expected);
  expectArithResults(divImmediateSpec(InputKind::uniform, RegisterKind::vgpr,
                                      immediate, immediateIsDividend, useIEEE),
                     static_cast<int>(words.size()), {words},
                     perLane(expected));
}

TEST_F(DivDevice, ImmediateOperands) {
  for (bool useIEEE : {true, false}) {
    WordPairs pairs = sample(pairsFor(useIEEE), 1024);
    for (int32_t immediate : immediatesFor(useIEEE)) {
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
    WordPairs pairs = sample(pairsFor(useIEEE), 4096);
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
          static_cast<int>(pairs.a.size()), {pairs.a, pairs.b}, expected);
    }
  }
}

// The quotient of a division by zero is unspecified, so the result is zeroed
// and only the completion of the kernel, without a fault or a hang, is checked.
TEST_F(DivDevice, DivisionByZeroCompletes) {
  for (bool useIEEE : {true, false}) {
    SCOPED_TRACE(pathName(useIEEE));
    WordPairs pairs = sample(pairsFor(useIEEE), 4096);
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
    expectArithResults(
        {{InputKind::uniform, InputKind::uniform},
         RegisterKind::sgpr,
         [useIEEE](IRBuilder &builder, RegisterRef dst,
                   const std::vector<RegisterRef> &inputs) {
           RegisterRange tmp = builder.allocateSgpr(1);
           builder.divsi32(tmp.ref(), inputs[0], inputs[1], useIEEE);
           builder.muli32(dst, tmp, 0);
         }},
        static_cast<int>(pairs.a.size()), {pairs.a, zeros}, zeros);
  }
}

// These pairs are outside the fast-path precondition, where fa * rcp(fb) can
// round up to the next integer, so only the exact path is checked. They guard
// the bound.
TEST_F(DivDevice, ExactPathHandlesPairsAboveFastBound) {
  WordPairs pairs;
  const int64_t above[][2] = {
      {16777214, 105},      {16777214, 16777215}, {-16777214, 105},
      {0x800001, 3},        {-0x800001, 3},       {0x800001, -3},
      {16777215, 16777214}, {16777215, 105},      {-16777215, 16777214},
      {0xFFFFFE, 7},        {16777217, 105},      {-16777217, 105},
      {16777217, 16777215}};
  for (const auto &pair : above)
    addPair(pairs, pair[0], pair[1]);
  std::mt19937 engine(divRandomSeed + 2);
  for (int i = 0; i < 4096; ++i)
    addPair(pairs, withRandomSign(engine, fastBound + engine() % fastBound),
            withRandomSign(engine, 1 + randomMagnitude(engine, 24)));
  padToWorkgroups(pairs, engine, 24);
  expectArithResults(
      divSpec(InputKind::lane, InputKind::lane, RegisterKind::vgpr, true),
      laneGroupCount(pairs.a), {pairs.a, pairs.b}, quotients(pairs));
  WordPairs scalar = sample(pairs, 1024);
  expectArithResults(
      divSpec(InputKind::uniform, InputKind::uniform, RegisterKind::sgpr, true),
      static_cast<int>(scalar.a.size()), {scalar.a, scalar.b},
      quotients(scalar));
}

TEST(DivAssembly, AllOperandPlacementsAssemble) {
  for (bool useIEEE : {true, false})
    expectAllPlacementsAssemble(
        pathName(useIEEE),
        [useIEEE](IRBuilder &builder, RegisterRef dst, Operand a, Operand b) {
          builder.divsi32(dst, a, b, useIEEE);
        });
}

// Compare.

namespace {

struct PredicateCase {
  Predicate pred;
  std::string_view name;
  std::string_view scalar32;
  std::string_view vector32;
  std::string_view vector64;
};

class CmpTest : public BuilderTest {
protected:
  std::string emit32(Predicate pred, RegisterRef dst, Operand a, Operand b) {
    return emitted([&](IRBuilder &local) { local.cmpi32(pred, dst, a, b); });
  }
  std::string emit64(Predicate pred, RegisterRef dst, Operand a, Operand b) {
    return emitted([&](IRBuilder &local) { local.cmpi64(pred, dst, a, b); });
  }
};

// Registers: dst s0, padding s1, a s[2:3], b s[4:5], first temporary s6.
class CmpOrderedTest : public CmpTest {
protected:
  RegisterRange dst = builder.allocateSgpr(1);
  RegisterRange pad = builder.allocateSgpr(1);
  RegisterRange a = builder.allocateSgpr(2, 2);
  RegisterRange b = builder.allocateSgpr(2, 2);
};

// The compared operands, and the registers that keep them allocated.
struct Operands {
  std::vector<RegisterRange> owners;
  Operand a;
  Operand b;
};

} // namespace

constexpr const char *signBitText = "-2147483648";

constexpr PredicateCase predicateCases[] = {
    {Predicate::eq, "eq", "s_cmp_eq_u32", "v_cmp_eq_u32", "v_cmp_eq_u64"},
    {Predicate::ne, "ne", "s_cmp_lg_u32", "v_cmp_ne_u32", "v_cmp_ne_u64"},
    {Predicate::slt, "slt", "s_cmp_lt_i32", "v_cmp_lt_i32", "v_cmp_lt_i64"},
    {Predicate::sle, "sle", "s_cmp_le_i32", "v_cmp_le_i32", "v_cmp_le_i64"},
    {Predicate::sgt, "sgt", "s_cmp_gt_i32", "v_cmp_gt_i32", "v_cmp_gt_i64"},
    {Predicate::sge, "sge", "s_cmp_ge_i32", "v_cmp_ge_i32", "v_cmp_ge_i64"},
    {Predicate::ult, "ult", "s_cmp_lt_u32", "v_cmp_lt_u32", "v_cmp_lt_u64"},
    {Predicate::ule, "ule", "s_cmp_le_u32", "v_cmp_le_u32", "v_cmp_le_u64"},
    {Predicate::ugt, "ugt", "s_cmp_gt_u32", "v_cmp_gt_u32", "v_cmp_gt_u64"},
    {Predicate::uge, "uge", "s_cmp_ge_u32", "v_cmp_ge_u32", "v_cmp_ge_u64"},
};

TEST_F(CmpTest, ScalarI32UniformBool) {
  RegisterRange dst = builder.allocateSgpr(1);
  RegisterRange a = builder.allocateSgpr(1);
  RegisterRange b = builder.allocateSgpr(1);
  for (const PredicateCase &current : predicateCases) {
    SCOPED_TRACE(current.name);
    std::string out = emit32(current.pred, dst.ref(), a, b);
    expectLinesInOrder(out, {line(current.scalar32, text(a), text(b)),
                             line("s_cselect_b32", text(dst), "1", "0")});
    EXPECT_EQ(countLinesContaining(out, "v_"), 0);
    EXPECT_EQ(countLinesContaining(out, "s_nop"), 0);
  }
}

TEST_F(CmpTest, ScalarI32LaneMask) {
  RegisterRange mask = builder.allocateSgpr(2, 2);
  RegisterRange a = builder.allocateSgpr(1);
  RegisterRange b = builder.allocateSgpr(1);
  for (const PredicateCase &current : predicateCases) {
    SCOPED_TRACE(current.name);
    std::string out = emit32(current.pred, mask.ref(), a, b);
    expectLinesInOrder(out, {line(current.scalar32, text(a), text(b)),
                             line("s_cselect_b64", text(mask), "-1", "0")});
    EXPECT_EQ(countLinesContaining(out, "s_nop"), 0);
  }
}

TEST_F(CmpTest, VectorI32) {
  RegisterRange mask = builder.allocateSgpr(2, 2);
  RegisterRange a = builder.allocateVgpr(1);
  RegisterRange b = builder.allocateVgpr(1);
  for (const PredicateCase &current : predicateCases) {
    SCOPED_TRACE(current.name);
    std::string out = emit32(current.pred, mask.ref(), a, b);
    expectLinesInOrder(
        out, {line(current.vector32, text(mask), text(a), text(b)), "s_nop 1"});
    EXPECT_EQ(countLinesContaining(out, "v_mov"), 0);
    EXPECT_EQ(countLinesContaining(out, "s_cselect"), 0);
  }
}

TEST_F(CmpTest, VectorI64) {
  RegisterRange mask = builder.allocateSgpr(2, 2);
  RegisterRange a = builder.allocateVgpr(2, 2);
  RegisterRange b = builder.allocateVgpr(2, 2);
  for (const PredicateCase &current : predicateCases) {
    SCOPED_TRACE(current.name);
    std::string out = emit64(current.pred, mask.ref(), a, b);
    expectLinesInOrder(
        out, {line(current.vector64, text(mask), text(a), text(b)), "s_nop 1"});
    EXPECT_EQ(countLinesContaining(out, "v_mov"), 0);
  }
}

TEST_F(CmpTest, VectorI32SgprInEitherSlotNeedsNoCopy) {
  RegisterRange mask = builder.allocateSgpr(2, 2);
  RegisterRange s = builder.allocateSgpr(1);
  RegisterRange v = builder.allocateVgpr(1);
  std::string out = emit32(Predicate::slt, mask.ref(), s, v);
  expectLinesInOrder(out, {line("v_cmp_lt_i32", text(mask), text(s), text(v))});
  EXPECT_EQ(countLinesContaining(out, "v_mov"), 0);
  out = emit32(Predicate::slt, mask.ref(), v, s);
  expectLinesInOrder(out, {line("v_cmp_lt_i32", text(mask), text(v), text(s))});
  EXPECT_EQ(countLinesContaining(out, "v_mov"), 0);
}

TEST_F(CmpTest, VectorI32InlineImmediateNeedsNoCopy) {
  RegisterRange mask = builder.allocateSgpr(2, 2);
  RegisterRange v = builder.allocateVgpr(1);
  std::string out = emit32(Predicate::ult, mask.ref(), v, -16);
  expectLinesInOrder(out, {line("v_cmp_lt_u32", text(mask), text(v), "-16")});
  EXPECT_EQ(countLinesContaining(out, "v_mov"), 0);
  out = emit32(Predicate::ult, mask.ref(), 64, v);
  expectLinesInOrder(out, {line("v_cmp_lt_u32", text(mask), "64", text(v))});
  EXPECT_EQ(countLinesContaining(out, "v_mov"), 0);
}

TEST_F(CmpTest, VectorI32LiteralIsCopied) {
  RegisterRange mask = builder.allocateSgpr(2, 2);
  RegisterRange v = builder.allocateVgpr(1);
  // The temporary takes the lowest free VGPR.
  std::string out = emit32(Predicate::slt, mask.ref(), v, literal32);
  expectLinesInOrder(out, {line("v_mov_b32", "v1", literal32Text),
                           line("v_cmp_lt_i32", text(mask), text(v), "v1"),
                           "s_nop 1"});
  out = emit32(Predicate::slt, mask.ref(), literal32, v);
  expectLinesInOrder(out, {line("v_mov_b32", "v1", literal32Text),
                           line("v_cmp_lt_i32", text(mask), "v1", text(v))});
  EXPECT_EQ(countLinesContaining(out, "v_mov_b32"), 1);
}

TEST_F(CmpTest, VectorI64SgprPairAndInlineNeedNoCopy) {
  RegisterRange mask = builder.allocateSgpr(2, 2);
  RegisterRange s = builder.allocateSgpr(2, 2);
  RegisterRange v = builder.allocateVgpr(2, 2);
  std::string out = emit64(Predicate::sge, mask.ref(), s, v);
  expectLinesInOrder(out, {line("v_cmp_ge_i64", text(mask), text(s), text(v))});
  EXPECT_EQ(countLinesContaining(out, "v_mov"), 0);
  out = emit64(Predicate::sge, mask.ref(), v, s);
  expectLinesInOrder(out, {line("v_cmp_ge_i64", text(mask), text(v), text(s))});
  out = emit64(Predicate::sge, mask.ref(), v, -16);
  expectLinesInOrder(out, {line("v_cmp_ge_i64", text(mask), text(v), "-16")});
  EXPECT_EQ(countLinesContaining(out, "v_mov"), 0);
}

TEST_F(CmpTest, VectorI64LiteralIsCopiedToEvenPair) {
  RegisterRange mask = builder.allocateSgpr(2, 2);
  RegisterRange v = builder.allocateVgpr(2, 2);
  // -100 sign-extends to a high word of -1, which a 64-bit mov cannot encode,
  // so it moves as two words.
  std::string out = emit64(Predicate::ult, mask.ref(), v, -100);
  expectLinesInOrder(
      out, {line("v_mov_b32", "v2", "-100"), line("v_mov_b32", "v3", "-1"),
            line("v_cmp_lt_u64", text(mask), text(v), "v[2:3]"), "s_nop 1"});
}

TEST_F(CmpTest, VectorI64MisalignedPairsAreCopied) {
  RegisterRange mask = builder.allocateSgpr(2, 2);
  RegisterRange odd = builder.allocateVgpr(4, 2);
  RegisterRange b = builder.allocateVgpr(2, 2);
  std::string out = emit64(Predicate::ult, mask.ref(), odd.ref(1, 2), b);
  expectLinesInOrder(
      out, {line("v_mov_b32", "v7", "v2"), line("v_mov_b32", "v6", "v1"),
            line("v_cmp_lt_u64", text(mask), "v[6:7]", text(b))});

  RegisterRange sOdd = builder.allocateSgpr(4, 2);
  out = emit64(Predicate::ult, mask.ref(), sOdd.ref(1, 2), b);
  expectLinesInOrder(
      out, {line("v_mov_b32", "v6", "s3"), line("v_mov_b32", "v7", "s4"),
            line("v_cmp_lt_u64", text(mask), "v[6:7]", text(b))});
}

TEST_F(CmpTest, ScalarI32TwoDistinctLiteralsCopyOne) {
  RegisterRange dst = builder.allocateSgpr(1);
  std::string out = emit32(Predicate::eq, dst.ref(), 0x12345, 0x54321);
  expectLinesInOrder(out, {line("s_mov_b32", "s1", "344865"),
                           line("s_cmp_eq_u32", "74565", "s1"),
                           line("s_cselect_b32", text(dst), "1", "0")});
  EXPECT_EQ(countLinesContaining(out, "s_mov_b32"), 1);
  EXPECT_EQ(countLinesContaining(
                emit32(Predicate::eq, dst.ref(), 0x12345, 0x12345), "s_mov"),
            0);
  EXPECT_EQ(countLinesContaining(emit32(Predicate::eq, dst.ref(), 5, 0x12345),
                                 "s_mov"),
            0);
}

TEST_F(CmpTest, ScalarI64EqualityIsDirectForLegalOperands) {
  RegisterRange dst = builder.allocateSgpr(1);
  RegisterRange pad = builder.allocateSgpr(1);
  RegisterRange a = builder.allocateSgpr(2, 2);
  RegisterRange b = builder.allocateSgpr(2, 2);
  std::string out = emit64(Predicate::eq, dst.ref(), a, b);
  expectLinesInOrder(out, {line("s_cmp_eq_u64", text(a), text(b)),
                           line("s_cselect_b32", text(dst), "1", "0")});
  EXPECT_EQ(countLinesContaining(out, "s_mov"), 0);
  out = emit64(Predicate::ne, dst.ref(), a, 0x12345);
  expectLinesInOrder(out, {line("s_cmp_lg_u64", text(a), "74565")});
  EXPECT_EQ(countLinesContaining(out, "s_mov"), 0);
  out = emit64(Predicate::ne, dst.ref(), a, INT32_MAX);
  expectLinesInOrder(out, {line("s_cmp_lg_u64", text(a), "2147483647")});
  EXPECT_EQ(countLinesContaining(out, "s_mov"), 0);
}

TEST_F(CmpTest, ScalarI64EqualityCopiesUnencodableOperands) {
  RegisterRange dst = builder.allocateSgpr(1);
  RegisterRange pad = builder.allocateSgpr(1);
  RegisterRange a = builder.allocateSgpr(2, 2);
  RegisterRange odd = builder.allocateSgpr(4, 2);
  // The temporary pair takes the lowest free even index, s[8:9].
  std::string out = emit64(Predicate::eq, dst.ref(), odd.ref(1, 2), a);
  expectLinesInOrder(out, {line("s_mov_b32", "s9", "s6"),
                           line("s_mov_b32", "s8", "s5"),
                           line("s_cmp_eq_u64", "s[8:9]", text(a))});
  out = emit64(Predicate::ne, dst.ref(), a, -17);
  expectLinesInOrder(out, {line("s_mov_b32", "s8", "-17"),
                           line("s_mov_b32", "s9", "-1"),
                           line("s_cmp_lg_u64", text(a), "s[8:9]")});
  out = emit64(Predicate::ne, dst.ref(), a, INT32_MIN);
  expectLinesInOrder(out, {line("s_mov_b32", "s8", "-2147483648"),
                           line("s_mov_b32", "s9", "-1"),
                           line("s_cmp_lg_u64", text(a), "s[8:9]")});
}

TEST_F(CmpTest, ScalarI64EqualityTwoDistinctLiteralsCopyOne) {
  RegisterRange dst = builder.allocateSgpr(1);
  std::string out = emit64(Predicate::eq, dst.ref(), 0x12345, 0x54321);
  expectLinesInOrder(out, {line("s_mov_b64", "s[2:3]", "344865"),
                           line("s_cmp_eq_u64", "74565", "s[2:3]")});
  EXPECT_EQ(countLinesContaining(out, "s_mov"), 1);
}

TEST_F(CmpOrderedTest, SignedLessBiasesBothHighWords) {
  std::string out = emit64(Predicate::slt, dst.ref(), a, b);
  expectLinesInOrder(out, {line("s_xor_b32", "s6", "s3", signBitText),
                           line("s_xor_b32", "s7", "s5", signBitText),
                           line("s_sub_u32", "s8", "s2", "s4"),
                           line("s_subb_u32", "s8", "s6", "s7"),
                           line("s_cselect_b32", "s0", "1", "0")});
  EXPECT_EQ(countLinesContaining(out, "v_"), 0);
}

TEST_F(CmpOrderedTest, UnsignedLessNeedsNoBias) {
  std::string out = emit64(Predicate::ult, dst.ref(), a, b);
  expectLinesInOrder(out, {line("s_sub_u32", "s6", "s2", "s4"),
                           line("s_subb_u32", "s6", "s3", "s5"),
                           line("s_cselect_b32", "s0", "1", "0")});
  EXPECT_EQ(countLinesContaining(out, "s_xor"), 0);
}

TEST_F(CmpOrderedTest, UnsignedGreaterSwapsOperands) {
  std::string out = emit64(Predicate::ugt, dst.ref(), a, b);
  expectLinesInOrder(out, {line("s_sub_u32", "s6", "s4", "s2"),
                           line("s_subb_u32", "s6", "s5", "s3"),
                           line("s_cselect_b32", "s0", "1", "0")});
}

TEST_F(CmpOrderedTest, GreaterOrEqualInvertsTheResult) {
  std::string out = emit64(Predicate::uge, dst.ref(), a, b);
  expectLinesInOrder(out, {line("s_sub_u32", "s6", "s2", "s4"),
                           line("s_subb_u32", "s6", "s3", "s5"),
                           line("s_cselect_b32", "s0", "0", "1")});
  out = emit64(Predicate::sle, dst.ref(), a, b);
  expectLinesInOrder(out, {line("s_xor_b32", "s6", "s5", signBitText),
                           line("s_xor_b32", "s7", "s3", signBitText),
                           line("s_sub_u32", "s8", "s4", "s2"),
                           line("s_subb_u32", "s8", "s6", "s7"),
                           line("s_cselect_b32", "s0", "0", "1")});
}

TEST_F(CmpOrderedTest, SignedImmediateIsBiasedOnTheHost) {
  // -0x12345 has the high word -1, which biases to 0x7fffffff.
  std::string out = emit64(Predicate::sle, dst.ref(), a, -0x12345);
  expectLinesInOrder(out, {line("s_xor_b32", "s6", "s3", signBitText),
                           line("s_sub_u32", "s7", "-74565", "s2"),
                           line("s_subb_u32", "s7", "2147483647", "s6"),
                           line("s_cselect_b32", "s0", "0", "1")});
  EXPECT_EQ(countLinesContaining(out, "s_xor_b32"), 1);
}

TEST_F(CmpOrderedTest, LaneMaskDestination) {
  RegisterRange mask = builder.allocateSgpr(2, 2);
  std::string out = emit64(Predicate::ult, mask.ref(), a, b);
  expectLinesInOrder(out, {line("s_subb_u32", "s8", "s3", "s5"),
                           line("s_cselect_b64", text(mask), "-1", "0")});
  out = emit64(Predicate::uge, mask.ref(), a, b);
  expectLinesInOrder(out, {line("s_cselect_b64", text(mask), "0", "-1")});
  EXPECT_EQ(countLinesContaining(out, "s_nop"), 0);
}

TEST_F(CmpOrderedTest, TwoImmediatesCopyTheSecondFirst) {
  std::string out = emit64(Predicate::ult, dst.ref(), 5, -100);
  expectLinesInOrder(out, {line("s_mov_b32", "s6", "-100"),
                           line("s_mov_b32", "s7", "-1"),
                           line("s_sub_u32", "s8", "5", "s6"),
                           line("s_subb_u32", "s8", "0", "s7"),
                           line("s_cselect_b32", "s0", "1", "0")});
}

TEST_F(CmpOrderedTest, DestinationMayAliasAnOperand) {
  std::string out = emit64(Predicate::ult, b.ref(), a, b);
  expectLinesInOrder(out, {line("s_sub_u32", "s6", "s2", "s4"),
                           line("s_subb_u32", "s6", "s3", "s5"),
                           line("s_cselect_b64", text(b), "-1", "0")});
  out = emit64(Predicate::eq, b.ref(), a, b);
  expectLinesInOrder(out, {line("s_cmp_eq_u64", text(a), text(b)),
                           line("s_cselect_b64", text(b), "-1", "0")});
}

TEST_F(CmpTest, TemporariesAreReleased) {
  RegisterRange dst = builder.allocateSgpr(1);
  RegisterRange a = builder.allocateSgpr(2, 2);
  RegisterRange b = builder.allocateSgpr(2, 2);
  emit64(Predicate::slt, dst.ref(), a, b);
  RegisterRange next = builder.allocateSgpr(1);
  EXPECT_EQ(next.start(), 1);
}

TEST_F(CmpTest, InvalidDestinationsThrowWithoutEmitting) {
  RegisterRange vdst = builder.allocateVgpr(4);
  RegisterRange acc = builder.allocateAgpr(1);
  RegisterRange three = builder.allocateSgpr(3, 2);
  RegisterRange oddPair = builder.allocateSgpr(4, 2);
  RegisterRange s = builder.allocateSgpr(1);
  RegisterRange v = builder.allocateVgpr(1);
  std::string message;
  try {
    builder.cmpi32(Predicate::eq, vdst.ref(3), s, s);
  } catch (const std::invalid_argument &error) {
    message = error.what();
  }
  EXPECT_EQ(message, "cmpi32 needs a destination of 1 sgpr or an sgpr pair at "
                     "an even index, got v3");
  expectRejected([&] { builder.cmpi32(Predicate::eq, vdst.ref(0, 2), s, s); });
  expectRejected([&] { builder.cmpi32(Predicate::eq, acc.ref(), s, s); });
  expectRejected([&] { builder.cmpi32(Predicate::eq, three.ref(), s, s); });
  expectRejected(
      [&] { builder.cmpi32(Predicate::eq, oddPair.ref(1, 2), s, v); });
  expectRejected(
      [&] { builder.cmpi64(Predicate::eq, oddPair.ref(1, 2), s, v); });
}

TEST_F(CmpTest, InvalidOperandsThrowWithoutEmitting) {
  RegisterRange s = builder.allocateSgpr(1);
  RegisterRange mask = builder.allocateSgpr(2, 2);
  RegisterRange v = builder.allocateVgpr(1);
  RegisterRange vpair = builder.allocateVgpr(2, 2);
  RegisterRange acc = builder.allocateAgpr(2, 2);
  expectRejected(
      [&] { builder.cmpi32(Predicate::eq, mask.ref(), acc.ref(0, 1), s); });
  expectRejected(
      [&] { builder.cmpi32(Predicate::eq, mask.ref(), s, acc.ref(1, 1)); });
  expectRejected(
      [&] { builder.cmpi64(Predicate::eq, mask.ref(), acc.ref(), vpair); });
  expectRejected([&] { builder.cmpi32(Predicate::eq, s.ref(), v, s); });
  expectRejected([&] { builder.cmpi64(Predicate::eq, s.ref(), vpair, mask); });
  expectRejected([&] { builder.cmpi32(Predicate::eq, mask.ref(), mask, s); });
  expectRejected([&] { builder.cmpi64(Predicate::eq, mask.ref(), s, vpair); });
}

TEST_F(CmpTest, BoundaryImmediatesAreAccepted) {
  RegisterRange mask = builder.allocateSgpr(2, 2);
  RegisterRange v = builder.allocateVgpr(1);
  EXPECT_NO_THROW(builder.cmpi32(Predicate::eq, mask.ref(), v, INT32_MIN));
  EXPECT_NO_THROW(builder.cmpi32(Predicate::eq, mask.ref(), v, INT32_MAX));
  RegisterRange pair = builder.allocateVgpr(2, 2);
  EXPECT_NO_THROW(builder.cmpi64(Predicate::eq, mask.ref(), pair, INT32_MIN));
  EXPECT_NO_THROW(builder.cmpi64(Predicate::eq, mask.ref(), pair, INT32_MAX));
}

// Index step that walks the edge values of the cross product in a pair vector.
constexpr int edgeStride32 = 17;
constexpr int edgeStride64 = 18;

// Immediates are int32 values; a 64-bit comparison sign-extends them.
constexpr int32_t negativeLiteral = -0x12345678;
constexpr int32_t inlineImmediate = -16;
constexpr int32_t nearMin32 = INT32_MIN + 5;

using CmpDevice = DeviceTest;

using CompareFn = void (IRBuilder::*)(Predicate, RegisterRef, Operand, Operand);
template <typename Signed, typename Unsigned>
static bool compareAs(Predicate pred, uint64_t x, uint64_t y) {
  Unsigned ua = static_cast<Unsigned>(x);
  Unsigned ub = static_cast<Unsigned>(y);
  Signed sa = static_cast<Signed>(ua);
  Signed sb = static_cast<Signed>(ub);
  switch (pred) {
  case Predicate::eq:
    return ua == ub;
  case Predicate::ne:
    return ua != ub;
  case Predicate::slt:
    return sa < sb;
  case Predicate::sle:
    return sa <= sb;
  case Predicate::sgt:
    return sa > sb;
  case Predicate::sge:
    return sa >= sb;
  case Predicate::ult:
    return ua < ub;
  case Predicate::ule:
    return ua <= ub;
  case Predicate::ugt:
    return ua > ub;
  case Predicate::uge:
    return ua >= ub;
  }
  return false;
}

// Expected 0/1 words, one per pair of values.
static Words expectedBits(int bits, Predicate pred, const Values &a,
                          const Values &b) {
  Words expected;
  for (size_t i = 0; i < a.size(); ++i) {
    bool holds = bits == 32 ? compareAs<int32_t, uint32_t>(pred, a[i], b[i])
                            : compareAs<int64_t, uint64_t>(pred, a[i], b[i]);
    expected.push_back(holds ? 1 : 0);
  }
  return expected;
}

// Every stride-th element, which gives one value per workgroup.
template <typename T>
static std::vector<T> everyNth(const std::vector<T> &values, int stride,
                               size_t count) {
  std::vector<T> picked;
  for (size_t i = 0; i < count; ++i)
    picked.push_back(values[i * stride]);
  return picked;
}

using OperandMaker = std::function<Operands(
    IRBuilder &builder, const std::vector<RegisterRef> &inputs)>;

// 32-bit operands straight from inputs 0 and 1.
static OperandMaker words32() {
  return [](IRBuilder &, const std::vector<RegisterRef> &in) {
    return Operands{{}, in[0], in[1]};
  };
}

// Input 0 against an immediate, which comes first if immediateFirst.
static OperandMaker word32WithImmediate(int32_t immediate,
                                        bool immediateFirst) {
  return [=](IRBuilder &, const std::vector<RegisterRef> &in) {
    if (immediateFirst)
      return Operands{{}, immediate, in[0]};
    return Operands{{}, in[0], immediate};
  };
}

// 64-bit operands from inputs (0, 1) and (2, 3).
static OperandMaker pairs64(bool misalignedA = false,
                            bool misalignedB = false) {
  return [=](IRBuilder &builder, const std::vector<RegisterRef> &in) {
    PairRegisters a = makePair(builder, in[0], in[1], misalignedA);
    PairRegisters b = makePair(builder, in[2], in[3], misalignedB);
    Operands operands{{}, a.pair, b.pair};
    operands.owners.push_back(std::move(a.range));
    operands.owners.push_back(std::move(b.range));
    return operands;
  };
}

// The pair from inputs (0, 1) against an immediate.
static OperandMaker pair64WithImmediate(int32_t immediate, bool immediateFirst,
                                        bool misaligned) {
  return [=](IRBuilder &builder, const std::vector<RegisterRef> &in) {
    PairRegisters pair = makePair(builder, in[0], in[1], misaligned);
    Operands operands = immediateFirst ? Operands{{}, immediate, pair.pair}
                                       : Operands{{}, pair.pair, immediate};
    operands.owners.push_back(std::move(pair.range));
    return operands;
  };
}

// Every lane stores 1 if the comparison, written to a lane mask, holds.
static void expectMaskResults(CompareFn compare, Predicate pred,
                              std::vector<InputKind> kinds,
                              const OperandMaker &make, const Inputs &inputs,
                              int groups, const Words &expected) {
  ArithKernelSpec spec{std::move(kinds), vgpr,
                       [=](IRBuilder &builder, RegisterRef dst,
                           const std::vector<RegisterRef> &in) {
                         Operands operands = make(builder, in);
                         RegisterRange mask = builder.allocateSgpr(2, 2);
                         (builder.*compare)(pred, mask.ref(), operands.a,
                                            operands.b);
                         builder.selectb32(dst, mask.ref(), 1, 0);
                       }};
  expectArithResults(spec, groups, inputs, expected);
}

// Each workgroup stores the uniform bool the comparison wrote to one SGPR.
static void expectBoolResults(CompareFn compare, Predicate pred,
                              std::vector<InputKind> kinds,
                              const OperandMaker &make, const Inputs &inputs,
                              int groups, const Words &expected) {
  ArithKernelSpec spec{std::move(kinds), sgpr,
                       [=](IRBuilder &builder, RegisterRef dst,
                           const std::vector<RegisterRef> &in) {
                         Operands operands = make(builder, in);
                         (builder.*compare)(pred, dst, operands.a, operands.b);
                       }};
  expectArithResults(spec, groups, inputs, expected);
}

TEST_F(CmpDevice, LaneOperandsI32) {
  WordPairs pairs = makeArithPairs(randomPairs);
  for (const PredicateCase &current : predicateCases) {
    SCOPED_TRACE(current.name);
    expectMaskResults(
        &IRBuilder::cmpi32, current.pred, {lane, lane}, words32(),
        {pairs.a, pairs.b}, laneGroupCount(pairs.a),
        expectedBits(32, current.pred, widen(pairs.a), widen(pairs.b)));
  }
}

TEST_F(CmpDevice, UniformOperandsToBoolI32) {
  WordPairs pairs = makeArithPairs(randomPairs);
  for (const PredicateCase &current : predicateCases) {
    SCOPED_TRACE(current.name);
    expectBoolResults(
        &IRBuilder::cmpi32, current.pred, {uniform, uniform}, words32(),
        {pairs.a, pairs.b}, static_cast<int>(pairs.a.size()),
        expectedBits(32, current.pred, widen(pairs.a), widen(pairs.b)));
  }
}

TEST_F(CmpDevice, UniformOperandsToMaskI32) {
  WordPairs pairs = makeArithPairs(randomPairs);
  for (const PredicateCase &current : predicateCases) {
    SCOPED_TRACE(current.name);
    expectMaskResults(&IRBuilder::cmpi32, current.pred, {uniform, uniform},
                      words32(), {pairs.a, pairs.b},
                      static_cast<int>(pairs.a.size()),
                      perLane(expectedBits(32, current.pred, widen(pairs.a),
                                           widen(pairs.b))));
  }
}

TEST_F(CmpDevice, MixedUniformAndLaneI32) {
  WordPairs pairs = makeArithPairs(randomPairs);
  int groups = laneGroupCount(pairs.a);
  Words uniformWords = everyNth(pairs.a, edgeStride32, groups);
  Values uniformLanes = widen(perLane(uniformWords));
  for (const PredicateCase &current : predicateCases) {
    SCOPED_TRACE(current.name);
    expectMaskResults(
        &IRBuilder::cmpi32, current.pred, {uniform, lane}, words32(),
        {uniformWords, pairs.b}, groups,
        expectedBits(32, current.pred, uniformLanes, widen(pairs.b)));
    expectMaskResults(
        &IRBuilder::cmpi32, current.pred, {lane, uniform}, words32(),
        {pairs.b, uniformWords}, groups,
        expectedBits(32, current.pred, widen(pairs.b), uniformLanes));
  }
}

TEST_F(CmpDevice, LaneAgainstImmediateI32) {
  WordPairs pairs = makeArithPairs(randomPairs);
  for (const PredicateCase &current : predicateCases) {
    for (int32_t immediate : {literal32, inlineImmediate}) {
      for (bool immediateFirst : {false, true}) {
        SCOPED_TRACE(std::string(current.name) + " immediate " +
                     std::to_string(immediate) +
                     (immediateFirst ? " first" : " second"));
        Values constant(pairs.a.size(), static_cast<uint32_t>(immediate));
        expectMaskResults(
            &IRBuilder::cmpi32, current.pred, {lane},
            word32WithImmediate(immediate, immediateFirst), {pairs.a},
            laneGroupCount(pairs.a),
            immediateFirst
                ? expectedBits(32, current.pred, constant, widen(pairs.a))
                : expectedBits(32, current.pred, widen(pairs.a), constant));
      }
    }
  }
}

TEST_F(CmpDevice, UniformAgainstImmediateI32) {
  WordPairs pairs = makeArithPairs(randomPairs);
  for (const PredicateCase &current : predicateCases) {
    SCOPED_TRACE(current.name);
    Values constant(pairs.a.size(), static_cast<uint32_t>(literal32));
    expectBoolResults(&IRBuilder::cmpi32, current.pred, {uniform},
                      word32WithImmediate(literal32, false), {pairs.a},
                      static_cast<int>(pairs.a.size()),
                      expectedBits(32, current.pred, widen(pairs.a), constant));
  }
}

TEST_F(CmpDevice, DestinationMayAliasAnOperandI32) {
  WordPairs pairs = makeArithPairs(randomPairs);
  for (const PredicateCase &current : predicateCases) {
    SCOPED_TRACE(current.name);
    Predicate pred = current.pred;
    ArithKernelSpec spec{{uniform, uniform},
                         sgpr,
                         [pred](IRBuilder &builder, RegisterRef dst,
                                const std::vector<RegisterRef> &in) {
                           builder.cmpi32(pred, in[0], in[0], in[1]);
                           builder.movb32(dst, in[0]);
                         }};
    expectArithResults(spec, static_cast<int>(pairs.a.size()),
                       {pairs.a, pairs.b},
                       expectedBits(32, pred, widen(pairs.a), widen(pairs.b)));
  }
}

// The 64-bit tests pass each operand as a low and a high word.
static Inputs laneInputs(const WordQuads &q) {
  return {q.aLo, q.aHi, q.bLo, q.bHi};
}

TEST_F(CmpDevice, LaneOperandsI64) {
  WordQuads quads = makeArithPairs64(randomPairs);
  Values a = join(quads.aLo, quads.aHi);
  Values b = join(quads.bLo, quads.bHi);
  for (const PredicateCase &current : predicateCases) {
    SCOPED_TRACE(current.name);
    Words expected = expectedBits(64, current.pred, a, b);
    expectMaskResults(&IRBuilder::cmpi64, current.pred,
                      {lane, lane, lane, lane}, pairs64(), laneInputs(quads),
                      laneGroupCount(quads.aLo), expected);
    SCOPED_TRACE("misaligned pairs");
    expectMaskResults(&IRBuilder::cmpi64, current.pred,
                      {lane, lane, lane, lane}, pairs64(true, true),
                      laneInputs(quads), laneGroupCount(quads.aLo), expected);
  }
}

TEST_F(CmpDevice, UniformOperandsToBoolI64) {
  WordQuads quads = makeArithPairs64(randomPairs);
  Values a = join(quads.aLo, quads.aHi);
  Values b = join(quads.bLo, quads.bHi);
  int groups = static_cast<int>(a.size());
  std::vector<InputKind> kinds{uniform, uniform, uniform, uniform};
  for (const PredicateCase &current : predicateCases) {
    SCOPED_TRACE(current.name);
    Words expected = expectedBits(64, current.pred, a, b);
    expectBoolResults(&IRBuilder::cmpi64, current.pred, kinds, pairs64(),
                      laneInputs(quads), groups, expected);
    SCOPED_TRACE("misaligned pairs");
    expectBoolResults(&IRBuilder::cmpi64, current.pred, kinds,
                      pairs64(true, true), laneInputs(quads), groups, expected);
  }
}

TEST_F(CmpDevice, UniformOperandsToMaskI64) {
  WordQuads quads = makeArithPairs64(randomPairs);
  Values a = join(quads.aLo, quads.aHi);
  Values b = join(quads.bLo, quads.bHi);
  for (const PredicateCase &current : predicateCases) {
    SCOPED_TRACE(current.name);
    expectMaskResults(&IRBuilder::cmpi64, current.pred,
                      {uniform, uniform, uniform, uniform}, pairs64(),
                      laneInputs(quads), static_cast<int>(a.size()),
                      perLane(expectedBits(64, current.pred, a, b)));
  }
}

TEST_F(CmpDevice, MixedUniformAndLaneI64) {
  WordQuads quads = makeArithPairs64(randomPairs);
  int groups = laneGroupCount(quads.aLo);
  Words uniformLo = everyNth(quads.aLo, edgeStride64, groups);
  Words uniformHi = everyNth(quads.aHi, edgeStride64, groups);
  Values uniformLanes = perLane(join(uniformLo, uniformHi));
  Values laneValues = join(quads.bLo, quads.bHi);
  for (const PredicateCase &current : predicateCases) {
    SCOPED_TRACE(current.name);
    expectMaskResults(&IRBuilder::cmpi64, current.pred,
                      {uniform, uniform, lane, lane}, pairs64(),
                      {uniformLo, uniformHi, quads.bLo, quads.bHi}, groups,
                      expectedBits(64, current.pred, uniformLanes, laneValues));
    expectMaskResults(&IRBuilder::cmpi64, current.pred,
                      {lane, lane, uniform, uniform}, pairs64(),
                      {quads.bLo, quads.bHi, uniformLo, uniformHi}, groups,
                      expectedBits(64, current.pred, laneValues, uniformLanes));
  }
}

TEST_F(CmpDevice, LaneAgainstImmediateI64) {
  WordQuads quads = makeArithPairs64(randomPairs);
  Values lanes = join(quads.aLo, quads.aHi);
  for (const PredicateCase &current : predicateCases) {
    for (int32_t immediate :
         {literal32, negativeLiteral, inlineImmediate, nearMin32}) {
      for (bool immediateFirst : {false, true}) {
        for (bool misaligned : {false, true}) {
          SCOPED_TRACE(std::string(current.name) + " immediate " +
                       std::to_string(immediate) +
                       (immediateFirst ? " first" : " second") +
                       (misaligned ? " misaligned" : " aligned"));
          Values constant(lanes.size(),
                          static_cast<uint64_t>(int64_t{immediate}));
          expectMaskResults(
              &IRBuilder::cmpi64, current.pred, {lane, lane},
              pair64WithImmediate(immediate, immediateFirst, misaligned),
              {quads.aLo, quads.aHi}, laneGroupCount(quads.aLo),
              immediateFirst ? expectedBits(64, current.pred, constant, lanes)
                             : expectedBits(64, current.pred, lanes, constant));
        }
      }
    }
  }
}

TEST_F(CmpDevice, UniformAgainstImmediateI64) {
  WordQuads quads = makeArithPairs64(randomPairs);
  Values values = join(quads.aLo, quads.aHi);
  for (const PredicateCase &current : predicateCases) {
    for (int32_t immediate :
         {nearMin32, literal32, negativeLiteral, inlineImmediate}) {
      for (bool immediateFirst : {false, true}) {
        SCOPED_TRACE(std::string(current.name) + " immediate " +
                     std::to_string(immediate) +
                     (immediateFirst ? " first" : " second"));
        Values constant(values.size(),
                        static_cast<uint64_t>(int64_t{immediate}));
        expectBoolResults(
            &IRBuilder::cmpi64, current.pred, {uniform, uniform},
            pair64WithImmediate(immediate, immediateFirst, false),
            {quads.aLo, quads.aHi}, static_cast<int>(values.size()),
            immediateFirst ? expectedBits(64, current.pred, constant, values)
                           : expectedBits(64, current.pred, values, constant));
      }
    }
  }
}

TEST_F(CmpDevice, DestinationMayAliasAnOperandI64) {
  WordQuads quads = makeArithPairs64(randomPairs);
  Values a = join(quads.aLo, quads.aHi);
  Values b = join(quads.bLo, quads.bHi);
  for (const PredicateCase &current : predicateCases) {
    SCOPED_TRACE(current.name);
    Predicate pred = current.pred;
    ArithKernelSpec spec{{uniform, uniform, uniform, uniform},
                         vgpr,
                         [pred](IRBuilder &builder, RegisterRef dst,
                                const std::vector<RegisterRef> &in) {
                           PairRegisters x = makePair(builder, in[0], in[1]);
                           PairRegisters y = makePair(builder, in[2], in[3]);
                           builder.cmpi64(pred, x.pair, x.pair, y.pair);
                           builder.selectb32(dst, x.pair, 1, 0);
                         }};
    expectArithResults(spec, static_cast<int>(a.size()), laneInputs(quads),
                       perLane(expectedBits(64, pred, a, b)));
  }
}

TEST_F(CmpDevice, VectorDestinationMayAliasAnSgprOperandI64) {
  WordQuads quads = makeArithPairs64(randomPairs);
  int groups = laneGroupCount(quads.aLo);
  Words uniformLo = everyNth(quads.aLo, edgeStride64, groups);
  Words uniformHi = everyNth(quads.aHi, edgeStride64, groups);
  Values uniformLanes = perLane(join(uniformLo, uniformHi));
  Values laneValues = join(quads.bLo, quads.bHi);
  for (const PredicateCase &current : predicateCases) {
    SCOPED_TRACE(current.name);
    Predicate pred = current.pred;
    ArithKernelSpec spec{{uniform, uniform, lane, lane},
                         vgpr,
                         [pred](IRBuilder &builder, RegisterRef dst,
                                const std::vector<RegisterRef> &in) {
                           PairRegisters x = makePair(builder, in[0], in[1]);
                           PairRegisters y = makePair(builder, in[2], in[3]);
                           builder.cmpi64(pred, x.pair, x.pair, y.pair);
                           builder.selectb32(dst, x.pair, 1, 0);
                         }};
    expectArithResults(spec, groups,
                       {uniformLo, uniformHi, quads.bLo, quads.bHi},
                       expectedBits(64, pred, uniformLanes, laneValues));
  }
}

// Select.

namespace {

class SelectTest : public BuilderTest {
protected:
  // Selects the width from dst.
  std::string emitSelect(RegisterRef dst, RegisterRef cond, Operand a,
                         Operand b) {
    return emitted([&](IRBuilder &local) {
      if (dst.count == 1)
        local.selectb32(dst, cond, a, b);
      else
        local.selectb64(dst, cond, a, b);
    });
  }
};

// Registers: padding s0, cond s1, dst s[2:3], a s[4:5], b s[6:7]; the first
// temporary pair is s[8:9].
class SelectScalar64Test : public SelectTest {
protected:
  RegisterRange pad = builder.allocateSgpr(1);
  RegisterRange cond = builder.allocateSgpr(1);
  RegisterRange dst = builder.allocateSgpr(2, 2);
  RegisterRange a = builder.allocateSgpr(2, 2);
  RegisterRange b = builder.allocateSgpr(2, 2);
};

// Registers: cond s0, range s[2:6] holding the misaligned dst s[3:4].
class SelectSplitTest : public SelectTest {
protected:
  RegisterRange cond = builder.allocateSgpr(1);
  RegisterRange range = builder.allocateSgpr(5, 2);
  RegisterRef dst = range.ref(1, 2);
};

// Registers: mask s[0:1], dst v0, a v1, b v2.
class SelectVectorTest : public SelectTest {
protected:
  RegisterRange mask = builder.allocateSgpr(2, 2);
  RegisterRange dst = builder.allocateVgpr(1);
  RegisterRange a = builder.allocateVgpr(1);
  RegisterRange b = builder.allocateVgpr(1);
};

// How the destination sits relative to the operand registers.
enum class SelectPlacement {
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
  SelectPlacement placement = SelectPlacement::separate;
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

// The condition register: the uniform bool input itself, or a lane mask that
// is set where a lane input is nonzero.
struct Condition {
  std::optional<RegisterRange> owner;
  RegisterRef ref;
};

} // namespace

// Builds "v_cndmask_b32 dst, src0, src1, mask" as rocisa prints it.
static std::string cndmask(const std::string &dst, const std::string &src0,
                           const std::string &src1, const std::string &mask) {
  return line("v_cndmask_b32", dst, src0, src1) + ", " + mask;
}

TEST_F(SelectTest, ScalarWidth1) {
  RegisterRange dst = builder.allocateSgpr(1);
  RegisterRange cond = builder.allocateSgpr(1);
  RegisterRange a = builder.allocateSgpr(1);
  RegisterRange b = builder.allocateSgpr(1);
  std::string out = emitSelect(dst.ref(), cond.ref(), a, b);
  expectLinesInOrder(out, {line("s_cmp_lg_u32", text(cond), "0"),
                           line("s_cselect_b32", text(dst), text(a), text(b))});
  EXPECT_EQ(countLinesContaining(out, "s_mov"), 0);
  EXPECT_EQ(countLinesContaining(out, "v_"), 0);
}

TEST_F(SelectTest, ScalarWidth1TwoDistinctLiteralsCopyOne) {
  RegisterRange dst = builder.allocateSgpr(1);
  RegisterRange cond = builder.allocateSgpr(1);
  std::string out = emitSelect(dst.ref(), cond.ref(), 0x12345, 0x54321);
  expectLinesInOrder(out, {line("s_mov_b32", "s2", "344865"),
                           line("s_cmp_lg_u32", text(cond), "0"),
                           line("s_cselect_b32", text(dst), "74565", "s2")});
  EXPECT_EQ(countLinesContaining(out, "s_mov"), 1);
}

TEST_F(SelectScalar64Test, AlignedPairs) {
  std::string out = emitSelect(dst.ref(), cond.ref(), a, b);
  expectLinesInOrder(out, {line("s_cmp_lg_u32", text(cond), "0"),
                           line("s_cselect_b64", text(dst), text(a), text(b))});
  EXPECT_EQ(countLinesContaining(out, "s_mov"), 0);
}

TEST_F(SelectScalar64Test, EncodableImmediatesAreDirect) {
  std::string out = emitSelect(dst.ref(), cond.ref(), INT32_MAX, 64);
  expectLinesInOrder(out,
                     {line("s_cselect_b64", text(dst), "2147483647", "64")});
  EXPECT_EQ(countLinesContaining(out, "s_mov"), 0);
  out = emitSelect(dst.ref(), cond.ref(), -16, a);
  expectLinesInOrder(out, {line("s_cselect_b64", text(dst), "-16", text(a))});
  EXPECT_EQ(countLinesContaining(out, "s_mov"), 0);
}

TEST_F(SelectScalar64Test, UnencodableImmediatesAreCopiedBeforeTheTest) {
  std::string out = emitSelect(dst.ref(), cond.ref(), -17, b);
  expectLinesInOrder(
      out, {line("s_mov_b32", "s8", "-17"), line("s_mov_b32", "s9", "-1"),
            line("s_cmp_lg_u32", text(cond), "0"),
            line("s_cselect_b64", text(dst), "s[8:9]", text(b))});
  out = emitSelect(dst.ref(), cond.ref(), a, INT32_MIN);
  expectLinesInOrder(out,
                     {line("s_mov_b32", "s8", "-2147483648"),
                      line("s_mov_b32", "s9", "-1"),
                      line("s_cmp_lg_u32", text(cond), "0"),
                      line("s_cselect_b64", text(dst), text(a), "s[8:9]")});
  out = emitSelect(dst.ref(), cond.ref(), -100, b);
  expectLinesInOrder(
      out, {line("s_mov_b32", "s8", "-100"), line("s_mov_b32", "s9", "-1"),
            line("s_cmp_lg_u32", text(cond), "0"),
            line("s_cselect_b64", text(dst), "s[8:9]", text(b))});
}

TEST_F(SelectScalar64Test, MisalignedOperandPairIsCopied) {
  RegisterRange odd = builder.allocateSgpr(4, 2);
  std::string out = emitSelect(dst.ref(), cond.ref(), odd.ref(1, 2), b);
  expectLinesInOrder(
      out, {line("s_mov_b32", "s13", "s10"), line("s_mov_b32", "s12", "s9"),
            line("s_cmp_lg_u32", text(cond), "0"),
            line("s_cselect_b64", text(dst), "s[12:13]", text(b))});
}

TEST_F(SelectScalar64Test, TwoDistinctLiteralsCopyOne) {
  std::string out = emitSelect(dst.ref(), cond.ref(), 0x12345, 0x54321);
  expectLinesInOrder(out,
                     {line("s_mov_b64", "s[8:9]", "344865"),
                      line("s_cmp_lg_u32", text(cond), "0"),
                      line("s_cselect_b64", text(dst), "74565", "s[8:9]")});
  EXPECT_EQ(countLinesContaining(out, "s_mov"), 1);
}

TEST_F(SelectSplitTest, SeparateOperandsGoLowFirst) {
  RegisterRange a = builder.allocateSgpr(2, 2);
  RegisterRange b = builder.allocateSgpr(2, 2);
  std::string out = emitSelect(dst, cond.ref(), a, b);
  expectLinesInOrder(out, {line("s_cmp_lg_u32", text(cond), "0"),
                           line("s_cselect_b32", "s3", "s8", "s10"),
                           line("s_cselect_b32", "s4", "s9", "s11")});
}

TEST_F(SelectSplitTest, OverlapThatForcesHighFirst) {
  RegisterRange b = builder.allocateSgpr(2, 2);
  // a = s[2:3]: its high word is the destination's low register.
  std::string out = emitSelect(dst, cond.ref(), range.ref(0, 2), b);
  expectLinesInOrder(out, {line("s_cmp_lg_u32", text(cond), "0"),
                           line("s_cselect_b32", "s4", "s3", "s9"),
                           line("s_cselect_b32", "s3", "s2", "s8")});
  EXPECT_EQ(countLinesContaining(out, "s_mov"), 0);
}

TEST_F(SelectSplitTest, OverlapInBothDirectionsCopiesOneWord) {
  // a = s[2:3] and b = s[4:5] around dst = s[3:4].
  std::string out =
      emitSelect(dst, cond.ref(), range.ref(0, 2), range.ref(2, 2));
  expectLinesInOrder(out, {line("s_mov_b32", "s1", "s3"),
                           line("s_cmp_lg_u32", text(cond), "0"),
                           line("s_cselect_b32", "s3", "s2", "s4"),
                           line("s_cselect_b32", "s4", "s1", "s5")});
  EXPECT_EQ(countLinesContaining(out, "s_mov"), 1);
}

TEST_F(SelectSplitTest, ImmediateHalvesWithTwoLiteralsCopyOne) {
  RegisterRange b = builder.allocateSgpr(2, 2);
  // The low words are two distinct literals; the high words are the inline 0.
  std::string out = emitSelect(dst, cond.ref(), 0x12345678, 0x76543210);
  EXPECT_EQ(countLinesContaining(out, "s_mov_b32"), 1);
  EXPECT_EQ(countLinesContaining(out, "s_cselect_b32"), 2);
}

TEST_F(SelectVectorTest, Width1TakesBInSrc0) {
  std::string out = emitSelect(dst.ref(), mask.ref(), a, b);
  expectLinesInOrder(out, {cndmask(text(dst), text(b), text(a), text(mask))});
  EXPECT_EQ(countLinesContaining(out, "s_nop"), 0);
  EXPECT_EQ(countLinesContaining(out, "v_mov"), 0);
}

TEST_F(SelectVectorTest, Width1CopiesSgprsAndLiterals) {
  RegisterRange s = builder.allocateSgpr(1);
  std::string out = emitSelect(dst.ref(), mask.ref(), s, literal32);
  expectLinesInOrder(out, {line("v_mov_b32", "v3", text(s)),
                           line("v_mov_b32", "v4", literal32Text),
                           cndmask(text(dst), "v4", "v3", text(mask))});
}

TEST_F(SelectVectorTest, Width1InlineImmediatesAreDirect) {
  std::string out = emitSelect(dst.ref(), mask.ref(), 5, -16);
  expectLinesInOrder(out, {cndmask(text(dst), "-16", "5", text(mask))});
  EXPECT_EQ(countLinesContaining(out, "v_mov"), 0);
  out = emitSelect(dst.ref(), mask.ref(), 64, a);
  expectLinesInOrder(out, {cndmask(text(dst), text(a), "64", text(mask))});
  EXPECT_EQ(countLinesContaining(out, "v_mov"), 0);
}

TEST_F(SelectVectorTest, UniformBoolIsWidenedToAMask) {
  RegisterRange cond = builder.allocateSgpr(1);
  // The temporary mask takes the lowest free even SGPR pair, s[4:5].
  std::string out = emitSelect(dst.ref(), cond.ref(), a, b);
  expectLinesInOrder(out, {line("s_cmp_lg_u32", text(cond), "0"),
                           line("s_cselect_b64", "s[4:5]", "-1", "0"),
                           cndmask(text(dst), text(b), text(a), "s[4:5]")});
  EXPECT_EQ(countLinesContaining(out, "s_nop"), 0);
}

TEST_F(SelectVectorTest, Width2TakesHalvesInOrder) {
  RegisterRange pairDst = builder.allocateVgpr(2, 2);
  RegisterRange pairA = builder.allocateVgpr(2, 2);
  RegisterRange pairB = builder.allocateVgpr(2, 2);
  std::string out = emitSelect(pairDst.ref(), mask.ref(), pairA, pairB);
  expectLinesInOrder(out, {cndmask("v4", "v8", "v6", text(mask)),
                           cndmask("v5", "v9", "v7", text(mask))});
}

TEST_F(SelectVectorTest, Width2SplitsAnImmediate) {
  RegisterRange pairDst = builder.allocateVgpr(2, 2);
  RegisterRange pairB = builder.allocateVgpr(2, 2);
  // INT32_MIN is the literal 0x80000000 and the inline -1.
  std::string out = emitSelect(pairDst.ref(), mask.ref(), INT32_MIN, pairB);
  expectLinesInOrder(out, {line("v_mov_b32", "v3", "-2147483648"),
                           cndmask("v4", "v6", "v3", text(mask)),
                           cndmask("v5", "v7", "-1", text(mask))});
  EXPECT_EQ(countLinesContaining(out, "v_mov"), 1);
}

TEST_F(SelectVectorTest, Width2OverlapForcesHighFirst) {
  RegisterRange range = builder.allocateVgpr(4, 2);
  RegisterRange pairB = builder.allocateVgpr(2, 2);
  RegisterRef dstPair = range.ref(1, 2);
  RegisterRef pairA = range.ref(0, 2);
  // a's high word is the destination's low register, so the high half goes
  // first.
  std::string out = emitSelect(dstPair, mask.ref(), pairA, pairB);
  expectLinesInOrder(out, {cndmask(text(dstPair.slice(1)), text(pairB.ref(1)),
                                   text(pairA.slice(1)), text(mask)),
                           cndmask(text(dstPair.slice(0)), text(pairB.ref(0)),
                                   text(pairA.slice(0)), text(mask))});
  EXPECT_EQ(countLinesContaining(out, "v_mov"), 0);
}

TEST_F(SelectVectorTest, Width2OverlapInBothDirectionsCopiesOneWord) {
  // dst = range[1:2] with a = range[0:1] and b = range[2:3].
  RegisterRange range = builder.allocateVgpr(4, 2);
  RegisterRef dstPair = range.ref(1, 2);
  RegisterRef pairA = range.ref(0, 2);
  RegisterRef pairB = range.ref(2, 2);
  std::string out = emitSelect(dstPair, mask.ref(), pairA, pairB);
  // The temporary takes the free hole at v3.
  expectLinesInOrder(out, {line("v_mov_b32", "v3", text(pairA.slice(1))),
                           cndmask(text(dstPair.slice(0)), text(pairB.slice(0)),
                                   text(pairA.slice(0)), text(mask)),
                           cndmask(text(dstPair.slice(1)), text(pairB.slice(1)),
                                   "v3", text(mask))});
  EXPECT_EQ(countLinesContaining(out, "v_mov"), 1);
}

TEST_F(SelectVectorTest, TemporariesAreReleased) {
  RegisterRange s = builder.allocateSgpr(1);
  emitSelect(dst.ref(), s.ref(), s, literal32);
  RegisterRange next = builder.allocateVgpr(1);
  EXPECT_EQ(next.start(), 3);
}

TEST_F(SelectVectorTest, InvalidOperandsThrowWithoutEmitting) {
  RegisterRange s = builder.allocateSgpr(1);
  RegisterRange acc = builder.allocateAgpr(2, 2);
  RegisterRange three = builder.allocateSgpr(3, 2);
  RegisterRange oddMask = builder.allocateSgpr(4, 2);
  RegisterRange pair = builder.allocateVgpr(2, 2);
  RegisterRange sdst = builder.allocateSgpr(1);
  RegisterRange sdst2 = builder.allocateSgpr(2, 2);

  std::string message;
  try {
    builder.selectb32(dst.ref(), a.ref(), s, s);
  } catch (const std::invalid_argument &error) {
    message = error.what();
  }
  EXPECT_EQ(message,
            "selectb32 condition must be an sgpr or an sgpr pair at an "
            "even index, got v1");
  expectRejected([&] { builder.selectb32(dst.ref(), acc.ref(0, 1), s, s); });
  expectRejected([&] { builder.selectb32(acc.ref(0, 1), s.ref(), s, s); });
  expectRejected(
      [&] { builder.selectb32(dst.ref(), s.ref(), acc.ref(0, 1), s); });
  expectRejected([&] { builder.selectb32(sdst.ref(), mask.ref(), s, s); });
  expectRejected([&] { builder.selectb32(sdst.ref(), s.ref(), a, s); });
  expectRejected([&] { builder.selectb64(sdst2.ref(), s.ref(), pair, sdst2); });
  expectRejected(
      [&] { builder.selectb32(dst.ref(), oddMask.ref(1, 2), a, b); });
  expectRejected([&] { builder.selectb32(dst.ref(), three.ref(), a, b); });
  expectRejected([&] {
    builder.selectb32(RegisterRef{RegisterKind::vgpr, 20, 3}, s.ref(), 5, 6);
  });
  expectRejected([&] { builder.selectb32(dst.ref(), s.ref(), pair, b); });
  expectRejected([&] { builder.selectb64(pair.ref(), s.ref(), a, b); });
  expectRejected([&] { builder.selectb64(dst.ref(), s.ref(), a, b); });
}

TEST_F(SelectVectorTest, BoundaryImmediatesAreAccepted) {
  RegisterRange pair = builder.allocateVgpr(2, 2);
  EXPECT_NO_THROW(builder.selectb32(dst.ref(), mask.ref(), -1, INT32_MIN));
  EXPECT_NO_THROW(
      builder.selectb64(pair.ref(), mask.ref(), INT32_MIN, INT32_MAX));
}

// Immediates that cover inline constants, literals and negative values.
constexpr int32_t selectImmediates32[] = {5, -16, 0x12345678, -0x65432110, 64};

using SelectDevice = DeviceTest;

// Condition words with zero, one, other small values, the sign bit, all ones
// and pseudo-random words, so that "nonzero is true" is exercised.
static Words makeConds(size_t count) {
  const Words edges{0, 1, 2, 0x80000000u, 0xFFFFFFFFu};
  Words conds;
  for (size_t i = 0; i < count; ++i)
    conds.push_back(i % 7 < edges.size()
                        ? edges[i % 7]
                        : static_cast<uint32_t>(i) * 2654435761u);
  return conds;
}

static SelectData makeData32() {
  WordPairs pairs = makeArithPairs(randomPairs);
  return {makeConds(pairs.a.size()), widen(pairs.a), widen(pairs.b)};
}

static SelectData makeData64() {
  WordQuads quads = makeArithPairs64(randomPairs);
  return {makeConds(quads.aLo.size()), join(quads.aLo, quads.aHi),
          join(quads.bLo, quads.bHi)};
}

static Condition makeCondition(IRBuilder &builder, InputKind kind,
                               RegisterRef input) {
  if (kind == uniform)
    return {std::nullopt, input};
  RegisterRange mask = builder.allocateSgpr(2, 2);
  builder.cmpi32(Predicate::ne, mask.ref(), input, 0);
  RegisterRef ref = mask.ref();
  return {std::move(mask), ref};
}

// Reads an operand: an immediate, one input register, or a pair built from two
// inputs.
static Operand makeOperand(IRBuilder &builder, const SelectCase &spec,
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

static void emitSelect(IRBuilder &builder, int width, RegisterRef dst,
                       RegisterRef cond, Operand a, Operand b) {
  if (width == 1)
    builder.selectb32(dst, cond, a, b);
  else
    builder.selectb64(dst, cond, a, b);
}

static PairRegisters emitSelectSeparate(IRBuilder &builder,
                                        const SelectCase &spec,
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
static PairRegisters emitSelectOverlapping(IRBuilder &builder,
                                           const SelectCase &spec,
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
  if (spec.placement == SelectPlacement::lowFirst) {
    dst = first;
    a = second;
  } else if (spec.placement == SelectPlacement::copied) {
    b = third;
  }
  builder.movb32(a.slice(0), in[1]);
  builder.movb32(a.slice(1), in[2]);
  if (spec.placement == SelectPlacement::copied) {
    builder.movb32(b.slice(0), in[3]);
    builder.movb32(b.slice(1), in[4]);
  }
  builder.selectb64(dst, cond, a, b);
  return {std::move(range), dst};
}

static PairRegisters emitSelectCase(IRBuilder &builder, const SelectCase &spec,
                                    const std::vector<RegisterRef> &in) {
  Condition cond = makeCondition(builder, spec.condKind, in[0]);
  if (spec.placement == SelectPlacement::separate)
    return emitSelectSeparate(builder, spec, cond.ref, in);
  return emitSelectOverlapping(builder, spec, cond.ref, in);
}

// A result is one lane of a vgpr result or one workgroup of an sgpr result.
static size_t selectInputIndex(const SelectCase &spec, InputKind kind,
                               size_t result) {
  bool perResult = spec.dstKind == sgpr || kind == lane;
  return perResult ? result : result / arithWorkgroupSize;
}

static uint64_t selectOperandValue(const SelectCase &spec,
                                   const std::optional<int32_t> &immediate,
                                   const Values &values, size_t result) {
  uint64_t value =
      immediate ? static_cast<uint64_t>(int64_t{*immediate})
                : values[selectInputIndex(spec, spec.operandKind, result)];
  return spec.width == 1 ? static_cast<uint32_t>(value) : value;
}

static Values selectExpectedValues(const SelectCase &spec,
                                   const SelectData &data, size_t results) {
  Values expected;
  for (size_t i = 0; i < results; ++i) {
    bool holds = data.conds[selectInputIndex(spec, spec.condKind, i)] != 0;
    expected.push_back(selectOperandValue(spec, holds ? spec.a : spec.b,
                                          holds ? data.a : data.b, i));
  }
  return expected;
}

static void addOperandInputs(KernelInputs &inputs, const SelectCase &spec,
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

static KernelInputs makeSelectInputs(const SelectCase &spec,
                                     const SelectData &data,
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
static void expectSelectResults(const SelectCase &spec,
                                const SelectData &data) {
  size_t results = data.conds.size();
  size_t groups = spec.dstKind == sgpr ? results : results / arithWorkgroupSize;
  size_t expectedCount = spec.dstKind == sgpr ? groups : results;
  KernelInputs inputs = makeSelectInputs(spec, data, groups);
  Values expected = selectExpectedValues(spec, data, expectedCount);
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

static SelectCase case32(RegisterKind dstKind, InputKind condKind,
                         InputKind operandKind) {
  return {1, dstKind, condKind, operandKind};
}

static SelectCase case64(RegisterKind dstKind, InputKind condKind,
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
  for (int32_t a : selectImmediates32) {
    for (int32_t b : selectImmediates32) {
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
  for (int32_t immediate : selectImmediates32) {
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
    for (SelectPlacement placement :
         {SelectPlacement::highFirst, SelectPlacement::lowFirst,
          SelectPlacement::copied}) {
      SCOPED_TRACE(std::string(dstKind == sgpr ? "sgpr" : "vgpr") +
                   " placement " + std::to_string(static_cast<int>(placement)));
      SelectCase spec = case64(dstKind, kind, kind);
      spec.placement = placement;
      expectSelectResults(spec, data);
    }
  }
}

// Buffer resource.

namespace {

class RsrcTest : public BuilderTest {
protected:
  std::string emitRsrc(RegisterRef dst, Operand pointer, Operand size) {
    return emitted(
        [&](IRBuilder &local) { local.makeRsrc(dst, pointer, size); });
  }
  int movCount(const std::string &out) {
    return countLinesContaining(out, "_mov_");
  }
  std::string errorOf(RegisterRef dst, Operand pointer, Operand size) {
    try {
      builder.makeRsrc(dst, pointer, size);
    } catch (const std::invalid_argument &error) {
      return error.what();
    }
    return {};
  }
};

enum class SizeSource { sgpr, immediate };

// Byte offsets of the arguments in the kernarg segment.
struct CopyOffsets {
  int in;
  int out;
  int inBytes;
  int outBytes;
};

} // namespace

// The upper word 0x00020000 as rocisa prints a decimal immediate.
const std::string upperWord = "131072";

// Literal placements that the allocator never hands out in these tests.
const RegisterRef dstRef{sgpr, 40, 4};
const RegisterRef pointerRef{sgpr, 20, 2};
const RegisterRef sizeRef{sgpr, 30, 1};

static RegisterRef word(int index) {
  return RegisterRef{sgpr, dstRef.start + index, 1};
}

TEST_F(RsrcTest, Word3ComesFromRocisa) {
  EXPECT_EQ(rocisa::SrdUpperValue(target.isaVersion())->getValue(), 0x00020000);
}

TEST_F(RsrcTest, SgprSize) {
  std::string out = emitRsrc(dstRef, pointerRef, sizeRef);
  EXPECT_EQ(movCount(out), 3);
  expectLinesInOrder(
      out, {line("s_mov_b64", text(RegisterRef{sgpr, 40, 2}), text(pointerRef)),
            line("s_mov_b32", text(word(2)), text(sizeRef)),
            line("s_mov_b32", text(word(3)), upperWord)});
}

TEST_F(RsrcTest, ImmediateSize) {
  struct Case {
    int32_t value;
    std::string printed;
  };
  for (const Case &current : {Case{0, "0"}, Case{308, "308"}, Case{-1, "-1"}}) {
    SCOPED_TRACE(current.printed);
    std::string out = emitRsrc(dstRef, pointerRef, current.value);
    EXPECT_EQ(movCount(out), 3);
    expectLinesInOrder(out, {line("s_mov_b64", text(RegisterRef{sgpr, 40, 2}),
                                  text(pointerRef)),
                             line("s_mov_b32", text(word(2)), current.printed),
                             line("s_mov_b32", text(word(3)), upperWord)});
  }
}

TEST_F(RsrcTest, MisalignedPointerPair) {
  RegisterRef pointer{sgpr, 21, 2};
  std::string out = emitRsrc(dstRef, pointer, sizeRef);
  EXPECT_EQ(countLinesContaining(out, "s_mov_b64"), 0);
  EXPECT_EQ(movCount(out), 4);
  // The split copies the high half first because dst starts above the source.
  expectLinesInOrder(
      out, {line("s_mov_b32", text(word(1)), text(RegisterRef{sgpr, 22, 1})),
            line("s_mov_b32", text(word(0)), text(RegisterRef{sgpr, 21, 1})),
            line("s_mov_b32", text(word(2)), text(sizeRef)),
            line("s_mov_b32", text(word(3)), upperWord)});
}

TEST_F(RsrcTest, PointerAlreadyInPlaceIsNotCopied) {
  RegisterRef pointer{sgpr, 40, 2};
  std::string out = emitRsrc(dstRef, pointer, sizeRef);
  EXPECT_EQ(movCount(out), 2);
  EXPECT_EQ(countLinesContaining(out, "s_mov_b64"), 0);
}

TEST_F(RsrcTest, SizeAlreadyInPlaceIsNotCopied) {
  std::string out = emitRsrc(dstRef, pointerRef, word(2));
  EXPECT_EQ(movCount(out), 2);
  expectLinesInOrder(
      out, {"s_mov_b64", line("s_mov_b32", text(word(3)), upperWord)});
}

TEST_F(RsrcTest, PointerAndSizeAlreadyInPlace) {
  std::string out = emitRsrc(dstRef, RegisterRef{sgpr, 40, 2}, word(2));
  EXPECT_EQ(out.find("s_mov_b64"), std::string::npos);
  EXPECT_EQ(movCount(out), 1);
  expectLinesInOrder(out, {line("s_mov_b32", text(word(3)), upperWord)});
}

TEST_F(RsrcTest, PointerInUpperHalfIsReadBeforeSizeIsWritten) {
  std::string out = emitRsrc(dstRef, RegisterRef{sgpr, 42, 2}, sizeRef);
  EXPECT_EQ(movCount(out), 3);
  expectLinesInOrder(out, {line("s_mov_b64", text(RegisterRef{sgpr, 40, 2}),
                                text(RegisterRef{sgpr, 42, 2})),
                           line("s_mov_b32", text(word(2)), text(sizeRef)),
                           line("s_mov_b32", text(word(3)), upperWord)});
}

TEST_F(RsrcTest, SizeInWord3IsReadBeforeItIsOverwritten) {
  std::string out = emitRsrc(dstRef, pointerRef, word(3));
  EXPECT_EQ(movCount(out), 3);
  expectLinesInOrder(out, {"s_mov_b64",
                           line("s_mov_b32", text(word(2)), text(word(3))),
                           line("s_mov_b32", text(word(3)), upperWord)});
}

TEST_F(RsrcTest, SizeInWord0IsSavedBeforePointerCopy) {
  RegisterRange probe = builder.allocateSgpr(1);
  RegisterRef temp = probe.ref();
  probe.release();
  std::string out = emitRsrc(dstRef, pointerRef, word(0));
  EXPECT_EQ(movCount(out), 4);
  expectLinesInOrder(
      out, {line("s_mov_b32", text(temp), text(word(0))),
            line("s_mov_b64", text(RegisterRef{sgpr, 40, 2}), text(pointerRef)),
            line("s_mov_b32", text(word(2)), text(temp)),
            line("s_mov_b32", text(word(3)), upperWord)});
}

TEST_F(RsrcTest, SizeInWord1IsSavedBeforePointerCopy) {
  std::string out = emitRsrc(dstRef, pointerRef, word(1));
  EXPECT_EQ(movCount(out), 4);
}

TEST_F(RsrcTest, SizeInWord1NeedsNoTemporaryWhenPointerStaysInPlace) {
  std::string out = emitRsrc(dstRef, RegisterRef{sgpr, 40, 2}, word(1));
  EXPECT_EQ(movCount(out), 2);
  expectLinesInOrder(out, {line("s_mov_b32", text(word(2)), text(word(1))),
                           line("s_mov_b32", text(word(3)), upperWord)});
}

TEST_F(RsrcTest, TemporaryIsReleasedAfterwards) {
  RegisterRange probe = builder.allocateSgpr(1);
  int expected = probe.ref().start;
  probe.release();
  emitRsrc(dstRef, pointerRef, word(0));
  RegisterRange after = builder.allocateSgpr(1);
  EXPECT_EQ(after.ref().start, expected);
}

TEST_F(RsrcTest, InvalidOperandsThrowWithoutEmitting) {
  RegisterRef vgprDst{vgpr, 0, 4};
  RegisterRef agprDst{agpr, 0, 4};
  EXPECT_THROW(builder.makeRsrc(vgprDst, pointerRef, sizeRef),
               std::invalid_argument);
  EXPECT_THROW(builder.makeRsrc(agprDst, pointerRef, sizeRef),
               std::invalid_argument);
  EXPECT_THROW(builder.makeRsrc({sgpr, 40, 3}, pointerRef, sizeRef),
               std::invalid_argument);
  EXPECT_THROW(builder.makeRsrc({sgpr, 40, 8}, pointerRef, sizeRef),
               std::invalid_argument);
  EXPECT_THROW(builder.makeRsrc({sgpr, 42, 4}, pointerRef, sizeRef),
               std::invalid_argument);
  EXPECT_THROW(builder.makeRsrc({sgpr, 46, 4}, pointerRef, sizeRef),
               std::invalid_argument);

  EXPECT_THROW(builder.makeRsrc(dstRef, 0, sizeRef), std::invalid_argument);
  EXPECT_THROW(builder.makeRsrc(dstRef, RegisterRef{vgpr, 2, 2}, sizeRef),
               std::invalid_argument);
  EXPECT_THROW(builder.makeRsrc(dstRef, RegisterRef{agpr, 2, 2}, sizeRef),
               std::invalid_argument);
  EXPECT_THROW(builder.makeRsrc(dstRef, RegisterRef{sgpr, 20, 1}, sizeRef),
               std::invalid_argument);
  EXPECT_THROW(builder.makeRsrc(dstRef, RegisterRef{sgpr, 20, 4}, sizeRef),
               std::invalid_argument);

  EXPECT_THROW(builder.makeRsrc(dstRef, pointerRef, RegisterRef{vgpr, 3, 1}),
               std::invalid_argument);
  EXPECT_THROW(builder.makeRsrc(dstRef, pointerRef, RegisterRef{agpr, 3, 1}),
               std::invalid_argument);
  EXPECT_THROW(builder.makeRsrc(dstRef, pointerRef, RegisterRef{sgpr, 30, 2}),
               std::invalid_argument);
  EXPECT_EQ(body.toString(), "");
}

TEST_F(RsrcTest, FailureWhereTemporaryWouldBeNeededAllocatesAndEmitsNothing) {
  RegisterRange probe = builder.allocateSgpr(1);
  int expected = probe.ref().start;
  probe.release();
  EXPECT_THROW(builder.makeRsrc(dstRef, RegisterRef{vgpr, 2, 2}, word(0)),
               std::invalid_argument);
  EXPECT_EQ(body.toString(), "");
  RegisterRange after = builder.allocateSgpr(1);
  EXPECT_EQ(after.ref().start, expected);
}

TEST_F(RsrcTest, ErrorMessagesNameTheOperation) {
  EXPECT_NE(errorOf({sgpr, 42, 4}, pointerRef, sizeRef).find("multiple of 4"),
            std::string::npos);
  EXPECT_NE(errorOf(dstRef, RegisterRef{vgpr, 2, 2}, sizeRef).find("sgpr pair"),
            std::string::npos);
  EXPECT_NE(errorOf(dstRef, 0, sizeRef).find("immediate 0"), std::string::npos);
  EXPECT_NE(errorOf(dstRef, pointerRef, RegisterRef{vgpr, 3, 1})
                .find("makeRsrc size must be a single sgpr"),
            std::string::npos);
}

constexpr int copyWorkgroupSize = 64;
constexpr int copyGroupCount = 2;
constexpr int copyLaneCount = copyWorkgroupSize * copyGroupCount;
// The input is shorter than the output, and both are shorter than the lane
// count, so lanes run past each descriptor's end.
constexpr int inWords = 77;
constexpr int outWords = 101;
constexpr int copyPadding = 64;
constexpr uint32_t copySentinel = 0xDEADBEEFu;
constexpr const char *copyKernelName = "codegen_buffer_copy";

static CopyOffsets declareArguments(KernelSignature &signature) {
  int in = signature.addPointer("in");
  int out = signature.addPointer("out");
  int inBytes = signature.addScalar("inBytes", 4);
  int outBytes = signature.addScalar("outBytes", 4);
  return {in, out, inBytes, outBytes};
}

// The pointers and sizes sit in SGPRs apart from the descriptors.
static void buildDescriptorsFromSgprs(Kernel &kernel,
                                      const CopyOffsets &offsets,
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
static void buildDescriptorsInPlace(Kernel &kernel, const CopyOffsets &offsets,
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

static RegisterRange computeByteOffset(Kernel &kernel) {
  IRBuilder &builder = kernel.builder();
  RegisterRange localId = builder.allocateVgpr(1);
  RegisterRange groupBase = builder.allocateSgpr(1);
  RegisterRange globalId = builder.allocateVgpr(1);
  RegisterRange offset = builder.allocateVgpr(1);
  // The workitem id register packs the x, y and z ids.
  builder.create<rocisa::VAndB32>(localId.container(), 0x3ff,
                                  kernel.workitemId().container());
  builder.muli32(groupBase.ref(), kernel.workgroupIdX(), copyWorkgroupSize);
  builder.addi32(globalId.ref(), groupBase, localId);
  builder.create<rocisa::VLShiftLeftB32>(offset.container(), 2,
                                         globalId.container());
  return offset;
}

// Every lane accesses memory, so the descriptor bounds do all the clipping.
static void buildCopy(Kernel &kernel, SizeSource source,
                      const CopyOffsets &offsets) {
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

// Every word is nonzero and differs from the copySentinel, so a load that is
// not clipped to zero shows up in the output.
static std::vector<uint32_t> makeInput() {
  std::vector<uint32_t> in(copyLaneCount + copyPadding);
  for (int i = 0; i < std::ssize(in); ++i)
    in[i] = (0x01000000u + static_cast<uint32_t>(i) * 2654435761u) | 1u;
  return in;
}

static std::vector<uint32_t> computeExpected(const std::vector<uint32_t> &in) {
  std::vector<uint32_t> expected(copyLaneCount + copyPadding, copySentinel);
  for (int i = 0; i < outWords; ++i)
    expected[i] = i < inWords ? in[i] : 0u;
  return expected;
}

static std::vector<std::byte> makeKernarg(int segmentSize,
                                          const CopyOffsets &offsets,
                                          const void *in, const void *out) {
  std::vector<std::byte> kernarg(segmentSize);
  writeArgument(kernarg, offsets.in, in);
  writeArgument(kernarg, offsets.out, out);
  writeArgument(kernarg, offsets.inBytes,
                static_cast<uint32_t>(inWords * sizeof(uint32_t)));
  writeArgument(kernarg, offsets.outBytes,
                static_cast<uint32_t>(outWords * sizeof(uint32_t)));
  return kernarg;
}

// Uploads the buffers, launches the kernel and downloads out.
static void runCopy(const std::vector<uint8_t> &hsaco,
                    const CopyOffsets &offsets, int segmentSize,
                    const std::vector<uint32_t> &in,
                    std::vector<uint32_t> &out) {
  ModulePtr module;
  hipFunction_t function = nullptr;
  loadKernel(hsaco, copyKernelName, module, function);
  DevicePtr deviceIn;
  DevicePtr deviceOut;
  upload(in, deviceIn);
  upload(out, deviceOut);
  ASSERT_FALSE(testing::Test::HasFatalFailure());

  std::vector<std::byte> kernarg =
      makeKernarg(segmentSize, offsets, deviceIn.get(), deviceOut.get());
  launchWithKernarg(function, copyGroupCount, copyWorkgroupSize, kernarg);
  ASSERT_FALSE(testing::Test::HasFatalFailure());
  download(deviceOut, out);
}

static void checkSource(const std::string &assembly, SizeSource source) {
  expectLinesInOrder(assembly, {"buffer_load_dword", "buffer_store_dword"});
  EXPECT_EQ(countLinesContaining(assembly, " offen"), 2);
  if (source == SizeSource::sgpr)
    return;
  // The pointers are already in place, so no pointer copy is emitted.
  EXPECT_EQ(countLinesContaining(assembly, "s_mov_b64"), 0);
  EXPECT_GT(countLinesContaining(assembly, "308"), 0);
  EXPECT_GT(countLinesContaining(assembly, "404"), 0);
}

static void runBufferCopyTest(SizeSource source) {
  Target target = Target::get("gfx950");
  Kernel kernel(target, copyKernelName, copyWorkgroupSize);
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
  std::vector<uint32_t> out(copyLaneCount + copyPadding, copySentinel);
  runCopy(hsaco, offsets, kernel.signature().segmentSize(), in, out);
  if (testing::Test::HasFatalFailure())
    return;

  ASSERT_EQ(out.size(), expected.size());
  for (int i = 0; i < std::ssize(out); ++i)
    ASSERT_EQ(out[i], expected[i]) << "word " << i;
}

TEST(RsrcDevice, SgprSizes) { runBufferCopyTest(SizeSource::sgpr); }

TEST(RsrcDevice, ImmediateSizes) { runBufferCopyTest(SizeSource::immediate); }

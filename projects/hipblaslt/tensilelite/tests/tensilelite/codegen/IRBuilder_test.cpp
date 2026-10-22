// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tensilelite/codegen/IRBuilder.hpp>

#include "code.hpp"
#include "container.hpp"
#include "instruction/branch.hpp"
#include "instruction/common.hpp"
#include "instruction/mem.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <iterator>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

using namespace rocm::tensilelite::codegen;

class IRBuilderTest : public testing::Test {
protected:
  // Ranges a test creates are destroyed before the builder and the allocator.
  Target target = Target::get("gfx950");
  RegisterAllocator allocator{target};
  rocisa::Module body{"body"};
  IRBuilder builder{target, allocator, body};
};

// Checks each needle against the lines of text in order, so tests do not
// depend on rocisa's padding and comment formatting.
void expectLinesInOrder(const std::string &text,
                        std::vector<std::string_view> needles) {
  std::vector<std::string> lines;
  std::istringstream stream(text);
  for (std::string line; std::getline(stream, line);)
    lines.push_back(line);

  std::ptrdiff_t next = 0;
  for (std::string_view needle : needles) {
    while (next < std::ssize(lines) &&
           lines[next].find(needle) == std::string::npos)
      ++next;
    if (next == std::ssize(lines)) {
      ADD_FAILURE() << "missing or out of order: '" << needle << "' in:\n"
                    << text;
      return;
    }
    ++next;
  }
}

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

} // namespace

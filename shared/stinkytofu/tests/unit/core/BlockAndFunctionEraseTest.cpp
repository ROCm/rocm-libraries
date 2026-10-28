// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include <array>
#include <string>
#include <vector>

#include "TestHelpers.hpp"
#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/serialization/asm/IRConverter.hpp"
#include "stinkytofu/serialization/asm/IRParser.hpp"
#include "stinkytofu/transforms/asm/ssa/LiftAsmRegistersToSSAPass.hpp"

using namespace stinkytofu;

namespace {
const char* const kChain = R"(st.func @k() {
^entry:
  "st.s_nop"(0) { issueCycles = 1, latencyCycles = 1 }
  Successors: ^mid
^mid:
  "st.s_nop"(1) { issueCycles = 1, latencyCycles = 1 }
  Successors: ^last
^last:
  "st.s_endpgm"() { issueCycles = 1, latencyCycles = 1 }
}
)";

void load(const std::string& text, Function& func) {
    MultiParseResult parsed = parseAllSourceStringsWithDiagnostics(text);
    ASSERT_FALSE(parsed.hasErrors());
    ASSERT_EQ(parsed.functions.size(), 1u);
    std::vector<Diagnostic> diagnostics;
    ASSERT_EQ(StinkyIRConverter::populateFunctionFromParsed(*parsed.functions.front(), func,
                                                            getGfxArchID(12, 5, 0), diagnostics),
              StinkyErrorCode::SUCCESS);
}

BasicBlock& blockNamed(Function& func, const std::string& label) {
    for (BasicBlock& block : func) {
        if (block.getLabel() == label) return block;
    }
    ADD_FAILURE() << "no block ^" << label;
    return *func.getEntryBlock();
}

std::vector<std::string> labelsOf(const std::vector<BasicBlock*>& blocks) {
    std::vector<std::string> labels;
    for (const BasicBlock* block : blocks) labels.push_back(block->getLabel());
    return labels;
}

std::vector<std::string> blockLabels(const Function& func) {
    std::vector<std::string> labels;
    for (const BasicBlock& block : func) labels.push_back(block.getLabel());
    return labels;
}

using Labels = std::vector<std::string>;
}  // namespace

TEST(EraseBlockTest, FallingPredecessorMovesToTheNextBlock) {
    Function func("k");
    load(kChain, func);
    func.eraseBlock(blockNamed(func, "mid"));

    EXPECT_EQ(blockLabels(func), (Labels{"entry", "last"}));
    EXPECT_EQ(labelsOf(blockNamed(func, "entry").getSuccessors()), (Labels{"last"}));
    EXPECT_EQ(labelsOf(blockNamed(func, "last").getPredecessors()), (Labels{"entry"}));
}

TEST(EraseBlockTest, KeepsOneEdgeWhenThePredecessorHasOne) {
    Function func("k");
    load(R"(st.func @k() {
^entry:
  "st.s_cbranch_scc1"(last) { issueCycles = 1, latencyCycles = 1 }
  Successors: ^mid, ^last
^mid:
  "st.s_nop"(1) { issueCycles = 1, latencyCycles = 1 }
  Successors: ^last
^last: { label }
  "st.s_endpgm"() { issueCycles = 1, latencyCycles = 1 }
}
)",
         func);
    func.eraseBlock(blockNamed(func, "mid"));

    EXPECT_EQ(labelsOf(blockNamed(func, "entry").getSuccessors()), (Labels{"last"}));
    EXPECT_EQ(labelsOf(blockNamed(func, "last").getPredecessors()), (Labels{"entry"}));
}

TEST(EraseBlockTest, LastBlockTakesItsEdgesAlong) {
    Function func("k");
    load(kChain, func);
    func.eraseBlock(blockNamed(func, "last"));

    EXPECT_EQ(blockLabels(func), (Labels{"entry", "mid"}));
    EXPECT_TRUE(blockNamed(func, "mid").getSuccessors().empty());
    EXPECT_EQ(labelsOf(blockNamed(func, "entry").getSuccessors()), (Labels{"mid"}));
}

TEST(EraseBlockTest, EdgeFromTheNextBlockDoesNotBecomeASelfLoop) {
    Function func("k");
    load(R"(st.func @k() {
^entry:
  "st.s_nop"(0) { issueCycles = 1, latencyCycles = 1 }
  Successors: ^mid
^mid:
  "st.s_nop"(1) { issueCycles = 1, latencyCycles = 1 }
  Successors: ^last
^last:
  "st.s_endpgm"() { issueCycles = 1, latencyCycles = 1 }
  Successors: ^mid
}
)",
         func);
    func.eraseBlock(blockNamed(func, "mid"));

    EXPECT_EQ(labelsOf(blockNamed(func, "entry").getSuccessors()), (Labels{"last"}));
    EXPECT_TRUE(blockNamed(func, "last").getSuccessors().empty());
    EXPECT_EQ(labelsOf(blockNamed(func, "last").getPredecessors()), (Labels{"entry"}));
}

TEST(EraseBlockTest, ClearsAttachedSSA) {
    Function func("k");
    load(kChain, func);
    test::setFunctionArch(func, GfxArchID::Gfx1250);
    Expected<LiftAttachedSSAResult> lifted = liftAsmRegistersToAttachedSSA(func);
    ASSERT_TRUE(lifted.hasValue()) << (lifted.hasValue() ? "" : lifted.getError());
    ASSERT_TRUE(func.hasAttachedSSA());

    func.eraseBlock(blockNamed(func, "mid"));
    EXPECT_FALSE(func.hasAttachedSSA());
}

TEST(RemoveFunctionTest, RemovesACallable) {
    StinkyAsmModule module("m", std::array<int, 3>{12, 5, 0}, StinkyAsmModule::ModuleOptions{});
    module.getFunction().setName("main");
    module.createFunction("helper");
    ASSERT_EQ(module.numFunctions(), 2u);

    EXPECT_TRUE(module.removeFunction("helper"));
    EXPECT_EQ(module.getFunction("helper"), nullptr);
    EXPECT_EQ(module.numFunctions(), 1u);
    EXPECT_FALSE(module.removeFunction("helper"));
}

TEST(RemoveFunctionTest, KeepsTheEntryFunction) {
    StinkyAsmModule module("m", std::array<int, 3>{12, 5, 0}, StinkyAsmModule::ModuleOptions{});
    module.getFunction().setName("main");
    module.createFunction("helper");

    EXPECT_FALSE(module.removeFunction("main"));
    EXPECT_FALSE(module.removeFunction(""));
    EXPECT_FALSE(module.removeFunction("nope"));
    EXPECT_EQ(module.numFunctions(), 2u);
    EXPECT_EQ(module.getFunction().getName(), "main");
}

// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "TestHelpers.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkySignature.hpp"
#include "stinkytofu/serialization/asm/IRConverter.hpp"
#include "stinkytofu/serialization/asm/IRParser.hpp"
#include "stinkytofu/serialization/asm/StinkyAsmPrinter.hpp"
#include "stinkytofu/transforms/asm/ssa/LiftAsmRegistersToSSAPass.hpp"

using namespace stinkytofu;

namespace {
const char* const kLoop = R"(st.func @k() {
^entry:
  "st.asm_directive"(".set", "x", "0")
  v[0:3] = "st.ds_load_b128"(v40) { issueCycles = 4, latencyCycles = 56, mod.ds = { na = 1, offset = 64, gds = false } }
  Successors: ^loop
^loop: { label, alignment = 16 }
  v4 = "st.v_add_f32"(v0, v1) { issueCycles = 1, latencyCycles = 1 }
  "st.s_cbranch_scc1"(loop) { issueCycles = 1, latencyCycles = 1 }
  Successors: ^loop, ^exit
^exit:
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

std::string print(const Function& func) {
    std::ostringstream out;
    AsmPrinter(out).print(func);
    return out.str();
}

std::vector<std::string> labelsOf(const std::vector<BasicBlock*>& blocks) {
    std::vector<std::string> labels;
    for (const BasicBlock* block : blocks) labels.push_back(block->getLabel());
    return labels;
}
}  // namespace

TEST(FunctionCopyTest, CopiesBlocksIrEdgesAndMetadata) {
    Function original("k");
    load(kLoop, original);
    original.setMetaData(kSigTotalVgprsMetaKey, 128);
    GemmTileConfig tile = original.getGemmTileConfig();
    tile.NumWaves = 4;
    original.setGemmTileConfig(tile);
    original.setIsCallable(true);

    Function copy("other");
    ASSERT_TRUE(copy.copyFrom(original));

    EXPECT_EQ(copy.getName(), "k");
    EXPECT_TRUE(copy.getIsCallable());
    EXPECT_EQ(copy.getMetaData(kSigTotalVgprsMetaKey), 128u);
    EXPECT_EQ(copy.getGemmTileConfig().NumWaves, 4u);
    EXPECT_EQ(print(copy), print(original));

    std::unordered_set<const IRBase*> originalNodes;
    for (const BasicBlock& block : original) {
        for (const IRBase& ir : block) originalNodes.insert(&ir);
    }
    auto from = original.begin();
    for (BasicBlock& block : copy) {
        EXPECT_EQ(block.getLabel(), from->getLabel());
        EXPECT_EQ(labelsOf(block.getSuccessors()), labelsOf(from->getSuccessors()));
        EXPECT_EQ(labelsOf(block.getPredecessors()), labelsOf(from->getPredecessors()));
        for (const BasicBlock* succ : block.getSuccessors()) EXPECT_EQ(succ->getParent(), &copy);
        for (const IRBase& ir : block) EXPECT_EQ(originalNodes.count(&ir), 0u);
        ++from;
    }

    // The copy owns its own nodes, so emptying it leaves the original whole.
    const std::string before = print(original);
    copy.getEntryBlock()->clear();
    EXPECT_EQ(print(original), before);
    EXPECT_NE(print(copy), before);
}

TEST(FunctionCopyTest, AttachedSSAIsNotCopied) {
    Function original("k");
    load(kLoop, original);
    test::setFunctionArch(original, GfxArchID::Gfx1250);
    Expected<LiftAttachedSSAResult> lifted = liftAsmRegistersToAttachedSSA(original);
    ASSERT_TRUE(lifted.hasValue()) << (lifted.hasValue() ? "" : lifted.getError());
    ASSERT_TRUE(original.hasAttachedSSA());

    Function copy;
    ASSERT_TRUE(copy.copyFrom(original));
    EXPECT_FALSE(copy.hasAttachedSSA());
    EXPECT_TRUE(original.hasAttachedSSA());
    EXPECT_EQ(print(copy), print(original));
}

/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#include <gtest/gtest.h>

#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/pipeline/ScopeAdaptor.hpp"

using namespace stinkytofu;

// A simple counting pass: counts StinkyInstructions in the Function.
class CountingPass : public Pass {
   public:
    static char ID;

    PassID getPassID() const override {
        return &ID;
    }

    const char* getName() const override {
        return "CountingPass";
    }

    PreservedAnalyses run(Function& func, PassContext& /*ctx*/, AnalysisManager& /*AM*/) override {
        count = 0;
        for (auto& bb : func) {
            for (auto& ir : bb) {
                if (ir.getType() == IRBase::IRType::StinkyTofu) count++;
            }
        }
        return PreservedAnalyses::none();
    }

    int count = 0;
};
char CountingPass::ID = 0;

class ScopeAdaptorTest : public ::testing::Test {
   protected:
    static constexpr std::array<int, 3> ARCH{12, 5, 0};

    StinkyAsmModule::ModuleOptions makeDefaultOptions() {
        StinkyAsmModule::ModuleOptions opts{};
        opts.OptLevel = 0;
        return opts;
    }

    /// Add one instruction to the module's entry BB and update group ranges.
    StinkyInstruction* addInst(StinkyAsmModule& module, AsmIRBuilder& builder, int destReg,
                               const std::vector<const std::string*>& groups) {
        BasicBlock& bb = *module.getFunction().getEntryBlock();
        size_t before = bb.size();

        StinkyInstruction* inst =
            builder.create(getMCIDByUOp(GFX::v_mov_b32, getGfxArchID(12, 5, 0)));
        inst->addDestReg(StinkyRegister("v", destReg, 1));

        module.updateInstructionGroups(groups, before);
        return inst;
    }

    /// Create a module with two groups:
    ///   [inst0]  [inst1 inst2]  [inst3 inst4]  [inst5]
    ///            ^--group0--^   ^--group1--^
    std::unique_ptr<StinkyAsmModule> createModuleWithGroups() {
        auto opts = makeDefaultOptions();
        auto module = std::make_unique<StinkyAsmModule>("test", ARCH, opts);

        module->addGroup("group0");
        module->addGroup("group1");

        BasicBlock& bb = *module->getFunction().getEntryBlock();
        GfxArchID archId = getGfxArchID(ARCH[0], ARCH[1], ARCH[2]);
        AsmIRBuilder builder(bb, archId);

        std::vector<const std::string*> noGroups;
        std::vector<const std::string*> g0 = {&groupName0};
        std::vector<const std::string*> g1 = {&groupName1};

        addInst(*module, builder, 10, noGroups);  // inst0: before groups
        addInst(*module, builder, 0, g0);         // inst1: group0
        addInst(*module, builder, 1, g0);         // inst2: group0
        addInst(*module, builder, 2, g1);         // inst3: group1
        addInst(*module, builder, 3, g1);         // inst4: group1
        addInst(*module, builder, 11, noGroups);  // inst5: after groups

        return module;
    }

    PassManager createOuterPM() {
        PassManager pm;
        GemmTileConfig config;
        config.arch = ARCH;
        pm.setGemmTileConfig(config);
        return pm;
    }

    int countInstructions(Function& func) {
        int count = 0;
        for (auto& bb : func) {
            for (auto& ir : bb) {
                if (ir.getType() == IRBase::IRType::StinkyTofu) count++;
            }
        }
        return count;
    }

    int countGroupInstructions(StinkyAsmModule& module, const std::string& groupName) {
        auto range = module.findGroupRange(groupName);
        if (!range) return -1;
        auto [begin, end] = range.value();
        int count = 0;
        for (auto it = begin; it != end; it++) count++;
        return count;
    }

    std::string groupName0 = "group0";
    std::string groupName1 = "group1";
};

// Verify that single-region extraction and splice-back preserves all instructions.
// Tests the basic round-trip: extract group0 to temp, run empty PM, splice back.
// If extraction or splice-back has an off-by-one or iterator bug, instructions disappear.
TEST_F(ScopeAdaptorTest, SingleRegionExtractAndSpliceBack) {
    auto module = createModuleWithGroups();
    int totalBefore = countInstructions(module->getFunction());
    ASSERT_EQ(totalBefore, 6);

    auto outerPM = createOuterPM();

    PassManager innerPM;
    outerPM.addPass(createKernelToRegionPassAdaptor(*module, "group0", std::move(innerPM)));
    outerPM.run(module->getFunction());

    // All instructions should still be present
    EXPECT_EQ(countInstructions(module->getFunction()), totalBefore);
    // Module should still have single BB
    EXPECT_EQ(module->getFunction().size(), 1u);
    // Group range should still be valid
    EXPECT_TRUE(module->findGroupRange("group0").has_value());
}

// Verify that setGroupRange is called correctly during single-region splice-back.
// After extraction and splice-back, the group range pointers must point to the
// spliced-back instructions (not the original, now-moved ones). Checks the range
// spans exactly 2 instructions.
TEST_F(ScopeAdaptorTest, SingleRegionGroupRangeUpdated) {
    auto module = createModuleWithGroups();

    auto outerPM = createOuterPM();

    PassManager innerPM;
    outerPM.addPass(createKernelToRegionPassAdaptor(*module, "group0", std::move(innerPM)));
    outerPM.run(module->getFunction());

    EXPECT_EQ(countGroupInstructions(*module, "group0"), 2);
}

// Verify that multi-region extraction joins group0 and group1 ranges,
// runs the inner PM on the combined range, and splices back without
// losing instructions. Module stays single-BB afterward.
TEST_F(ScopeAdaptorTest, MultiRegionExtractAndSpliceBack) {
    auto module = createModuleWithGroups();
    int totalBefore = countInstructions(module->getFunction());

    auto outerPM = createOuterPM();

    PassManager innerPM;
    outerPM.addPass(
        createKernelToRegionsPassAdaptor(*module, {"group0", "group1"}, std::move(innerPM)));
    outerPM.run(module->getFunction());

    EXPECT_EQ(countInstructions(module->getFunction()), totalBefore);
    EXPECT_EQ(module->getFunction().size(), 1u);
}

// Verify that whole-kernel mode extracts all instructions to temp,
// runs the inner PM, and splices everything back. This mode restores
// the flat single-BB invariant so subsequent scoped adapters can
// call findGroupRange().
TEST_F(ScopeAdaptorTest, WholeKernelExtractAndSpliceBack) {
    auto module = createModuleWithGroups();
    int totalBefore = countInstructions(module->getFunction());

    auto outerPM = createOuterPM();

    PassManager innerPM;
    outerPM.addPass(createKernelPassAdaptor(*module, std::move(innerPM)));
    outerPM.run(module->getFunction());

    EXPECT_EQ(countInstructions(module->getFunction()), totalBefore);
}

// Verify that extraction isolates the group — the inner PM sees ONLY
// the extracted group's instructions, not the entire module. Without
// this test, a bug where extraction is skipped (inner PM runs on the
// full module) would go undetected by the round-trip tests.
TEST_F(ScopeAdaptorTest, InnerPassSeesExtractedGroup) {
    auto module = createModuleWithGroups();

    auto outerPM = createOuterPM();

    auto* countingPassPtr = new CountingPass();
    auto countingPass = std::unique_ptr<Pass>(countingPassPtr);

    PassManager innerPM;
    innerPM.addPass(std::move(countingPass));
    outerPM.addPass(createKernelToRegionPassAdaptor(*module, "group0", std::move(innerPM)));
    outerPM.run(module->getFunction());

    // CountingPass should see 2 instructions (group0 has inst1, inst2)
    EXPECT_EQ(countingPassPtr->count, 2);
}

// Verify that the flat-BB invariant holds between adapter invocations.
// The first adapter must restore a single BB so the second adapter's
// findGroupRange() works. If the first adapter leaves multiple BBs,
// the second adapter breaks.
TEST_F(ScopeAdaptorTest, SequentialSingleRegionAdapters) {
    auto module = createModuleWithGroups();

    auto outerPM = createOuterPM();

    {
        PassManager innerPM;
        outerPM.addPass(createKernelToRegionPassAdaptor(*module, "group0", std::move(innerPM)));
    }
    {
        PassManager innerPM;
        outerPM.addPass(createKernelToRegionPassAdaptor(*module, "group1", std::move(innerPM)));
    }
    outerPM.run(module->getFunction());

    EXPECT_TRUE(module->findGroupRange("group0").has_value());
    EXPECT_TRUE(module->findGroupRange("group1").has_value());
    EXPECT_EQ(countInstructions(module->getFunction()), 6);
}

// End-to-end pipeline pattern matching gfx1250: two single-region
// adapters (per-region optimization) followed by a multi-region adapter
// (cross-region waitcnt reinsertion). Verifies the full sequence composes
// without corrupting state.
TEST_F(ScopeAdaptorTest, SingleRegionsThenMultiRegion) {
    auto module = createModuleWithGroups();

    auto outerPM = createOuterPM();

    {
        PassManager innerPM;
        outerPM.addPass(createKernelToRegionPassAdaptor(*module, "group0", std::move(innerPM)));
    }
    {
        PassManager innerPM;
        outerPM.addPass(createKernelToRegionPassAdaptor(*module, "group1", std::move(innerPM)));
    }
    {
        // Multi-region on both groups (like waitcnt reinsertion)
        auto* countingPassPtr = new CountingPass();
        auto countingPass = std::unique_ptr<Pass>(countingPassPtr);

        PassManager multiPM;
        multiPM.addPass(std::move(countingPass));
        outerPM.addPass(
            createKernelToRegionsPassAdaptor(*module, {"group0", "group1"}, std::move(multiPM)));
    }
    outerPM.run(module->getFunction());

    // All instructions preserved
    EXPECT_EQ(countInstructions(module->getFunction()), 6);
}

// Edge case: adapter targeting a non-existent group name must not crash
// and must not modify any instructions. Guards against null dereference
// in the findGroupRange lookup path.
TEST_F(ScopeAdaptorTest, MissingGroupDoesNotCrash) {
    auto module = createModuleWithGroups();

    auto outerPM = createOuterPM();

    PassManager innerPM;
    outerPM.addPass(createKernelToRegionPassAdaptor(*module, "nonexistent", std::move(innerPM)));
    outerPM.run(module->getFunction());

    EXPECT_EQ(countInstructions(module->getFunction()), 6);
}

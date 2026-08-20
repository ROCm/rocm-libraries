// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Gfx1250HazardPass's xcnt drain profile accumulates over every function it
// walks and splits the counts between the kernel body (the entry function) and
// the helper functions the kernel calls (activation functions and friends).
//
// The split needs a module: a callable Function plus the whole-kernel function
// list the pass is constructed with. stinkytofu-opt cannot supply either — it
// runs every `st.func` as an independent kernel with its own pass instance, and
// .stir has no syntax for a callable function — so the module is built in C++
// here. The single-function report is covered by the .stir test instead, see
// tests/filecheck/gfx1250_xcnt_drain_sites.stir.
//
// TODO: teach .stir/stinkytofu-opt to represent a module (entry + callable
// functions) and move these cases to a FileCheck test.

#include <gtest/gtest.h>

#include <array>
#include <iostream>
#include <sstream>
#include <string>

#include "TestHelpers.hpp"
#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/core/ModulePassManager.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/transforms/asm/Gfx1250HazardPass.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {

constexpr GfxArchID kArch = GfxArchID::Gfx1250;
constexpr std::array<int, 3> kArchTriple = {12, 5, 0};

/// A VMEM load, then an s_set_vgpr_msb whose successor is not memory: the pass
/// drains XCNT at the s_set_vgpr_msb, so each call costs exactly one drain.
/// The block has no loop and no matrix instruction, so it lands in "other".
void addVgprMsbDrainSite(BasicBlock* bb, int destReg) {
    AsmIRBuilder builder(*bb, kArch);

    StinkyInstruction* load = builder.create(getMCIDByUOp(GFX::buffer_load_b32, kArch));
    load->addDestReg(vgpr(destReg));
    load->addSrcReg(vgpr(0));

    StinkyInstruction* msb = builder.create(getMCIDByUOp(GFX::s_set_vgpr_msb, kArch));
    msb->addSrcReg(StinkyRegister(1));

    createVAddInBlock(bb, kArch, destReg + 1, 1, 2);
}

/// Run the whole-kernel profile ModulePass and return its report.
std::string runPassAndCaptureReport(StinkyAsmModule& module) {
    auto pass = createGfx1250HazardModulePass(/*enableXcntDrainProfile=*/true);

    GemmTileConfig config;
    config.arch = kArchTriple;

    // The pass's only gate; TensileLite forwards rocisa's archCaps here.
    AsmCapsConfig caps;
    caps.requiresXCntForVolatileVMEM = true;

    PassContext ctx;
    ctx.setGemmTileConfig(config);
    ctx.setAsmCapsConfig(caps);
    ModuleAnalysisManager mam;

    std::ostringstream report;
    std::streambuf* saved = std::cerr.rdbuf(report.rdbuf());
    pass->run(module, ctx, mam);
    std::cerr.rdbuf(saved);
    return report.str();
}

class Gfx1250HazardProfileTest : public ::testing::Test {
   protected:
    void SetUp() override {
        module = std::make_unique<StinkyAsmModule>("xcnt_profile_test", kArchTriple,
                                                   StinkyAsmModule::ModuleOptions{});
        entry = &module->getFunction();
        entry->setName("entry_func");
        setFunctionArch(*entry, kArch);
    }

    Function& addHelperFunction(const std::string& name) {
        Function& helper = module->createFunction(name);
        setFunctionArch(helper, kArch);
        return helper;
    }

    std::unique_ptr<StinkyAsmModule> module;
    Function* entry = nullptr;
};

TEST_F(Gfx1250HazardProfileTest, SplitsCountsBetweenKernelBodyAndHelperFunctions) {
    // Kernel body: two drains. Only it uses tensor_load_to_lds.
    createTensorLoadInBlock(entry->getEntryBlock(), kArch, /*src0Reg=*/0, /*src1Reg=*/4);
    addVgprMsbDrainSite(entry->getEntryBlock(), /*destReg=*/100);
    addVgprMsbDrainSite(entry->getEntryBlock(), /*destReg=*/110);

    // Two helper functions: one drain each.
    addVgprMsbDrainSite(addHelperFunction("activation_a").getEntryBlock(), /*destReg=*/120);
    addVgprMsbDrainSite(addHelperFunction("activation_b").getEntryBlock(), /*destReg=*/130);

    const std::string report = runPassAndCaptureReport(*module);

    // The "whole kernel" lines cover all four drains, not just the last
    // function walked.
    EXPECT_NE(report.find(
                  "] whole kernel xcnt drains: total=4, loop+matrix=0, loop=0, matrix=0, other=4"),
              std::string::npos)
        << report;
    EXPECT_NE(report.find("] kernel body xcnt drains: total=2"), std::string::npos) << report;
    EXPECT_NE(report.find("] helper functions xcnt drains: total=2"), std::string::npos) << report;

    EXPECT_NE(report.find("] whole kernel xcnt drain rules: atomic=0, smem=0, flat=0, "
                          "foreverSleep=0, scalarPrefetch=0, vgprMsb=4"),
              std::string::npos)
        << report;
    EXPECT_NE(report.find("] kernel body xcnt drain rules: atomic=0, smem=0, flat=0, "
                          "foreverSleep=0, scalarPrefetch=0, vgprMsb=2"),
              std::string::npos)
        << report;

    EXPECT_NE(report.find("] kernel body tensor_load_to_lds: used"), std::string::npos) << report;
    EXPECT_NE(report.find("] helper functions tensor_load_to_lds: not used"), std::string::npos)
        << report;
}

TEST_F(Gfx1250HazardProfileTest, ReportsTotalsOnlyWhenTheKernelHasNoHelperFunction) {
    addVgprMsbDrainSite(entry->getEntryBlock(), /*destReg=*/100);

    const std::string report = runPassAndCaptureReport(*module);

    EXPECT_NE(report.find("] whole kernel xcnt drains: total=1"), std::string::npos) << report;
    // A split would only repeat the totals.
    EXPECT_EQ(report.find("kernel body"), std::string::npos) << report;
    EXPECT_EQ(report.find("helper functions"), std::string::npos) << report;
}

}  // namespace

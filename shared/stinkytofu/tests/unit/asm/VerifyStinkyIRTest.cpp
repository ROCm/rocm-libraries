// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include <string>
#include <utility>
#include <vector>

#include "TestHelpers.hpp"
#include "stinkytofu/analysis/asm/AsmVerifierPass.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/ssa/StinkyOpOperand.hpp"
#include "stinkytofu/ir/asm/ssa/StinkySSAValue.hpp"
#include "stinkytofu/serialization/asm/IRConverter.hpp"
#include "stinkytofu/serialization/asm/IRParser.hpp"

using namespace stinkytofu;

namespace {
void load(const std::string& text, Function& func) {
    MultiParseResult parsed = parseAllSourceStringsWithDiagnostics(text);
    ASSERT_FALSE(parsed.hasErrors());
    ASSERT_EQ(parsed.functions.size(), 1u);
    std::vector<Diagnostic> diagnostics;
    ASSERT_EQ(StinkyIRConverter::populateFunctionFromParsed(*parsed.functions.front(), func,
                                                            getGfxArchID(12, 5, 0), diagnostics),
              StinkyErrorCode::SUCCESS);
}

const char* const kTwoBadLoads = R"(st.func @k() {
^entry:
  "st.s_nop"(0) { issueCycles = 1, latencyCycles = 1 }
  mid:
  v[0:1] = "st.ds_load_b128"(v40) { issueCycles = 4, latencyCycles = 56 }
  v[4:5] = "st.ds_load_b128"(s3) { issueCycles = 4, latencyCycles = 56 }
}
)";
}  // namespace

TEST(VerifyStinkyIRTest, ReportsEveryProblemAtItsInstruction) {
    Function func("k");
    load(kTwoBadLoads, func);

    const std::vector<Diagnostic> found = verifyStinkyIR(func);
    ASSERT_EQ(found.size(), 3u);
    // The label line is not counted, so the loads are #1 and #2.
    EXPECT_EQ(found[0].format(),
              "@k ^entry #1: error: 'ds_load_b128' operand dest[0] has register width 2, "
              "expected 4 [register-width]");
    EXPECT_EQ(found[1].getId(), DiagId::RegisterWidth);
    ASSERT_NE(found[1].getIRLocation(), nullptr);
    EXPECT_EQ(found[1].getIRLocation()->index, 2);
    EXPECT_EQ(found[2].getId(), DiagId::RegisterType);
    EXPECT_EQ(found[2].getMessage(),
              "'ds_load_b128' operand src[0] has register type 's', expected 'v'");
}

TEST(VerifyStinkyIRTest, ValidateStillReportsTheFirstCategory) {
    Function func("k");
    load(kTwoBadLoads, func);

    const std::vector<Diagnostic> found = verifyStinkyIR(func);
    ASSERT_EQ(found.size(), 3u);
    EXPECT_EQ(validateStinkyIR(func), "Register width validation failed:\n" + found[0].format() +
                                          "\n" + found[1].format() + "\n" + found[2].format() +
                                          "\n");
}

TEST(VerifyStinkyIRTest, EmptyFunctionIsOneDiagnostic) {
    Function func("k");
    const std::vector<Diagnostic> found = verifyStinkyIR(func);
    ASSERT_EQ(found.size(), 1u);
    EXPECT_EQ(found[0].format(), "@k: error: Function is empty (no basic blocks) [empty-function]");
    EXPECT_EQ(validateStinkyIR(func), "Function is empty (no basic blocks)");
}

TEST(VerifyStinkyIRTest, BranchToAMissingLabelIsReportedByVerifyOnly) {
    Function func("k");
    load(R"(st.func @k() {
^entry:
  "st.s_cbranch_scc1"(loop) { issueCycles = 1, latencyCycles = 1 }
  "st.s_branch"(nowhere) { issueCycles = 1, latencyCycles = 1 }
  Successors: ^loop
^loop:
  "st.s_endpgm"() { issueCycles = 1, latencyCycles = 1 }
}
)",
         func);

    const std::vector<Diagnostic> found = verifyStinkyIR(func);
    ASSERT_EQ(found.size(), 1u);
    EXPECT_EQ(found[0].format(),
              "@k ^entry #1: error: 's_branch' targets 'nowhere', which no label defines "
              "[undefined-label]");
    EXPECT_EQ(validateStinkyIR(func), "");
}

TEST(VerifyStinkyIRTest, BrokenAttachedSSAIsReportedByVerifyOnly) {
    Function func("k");
    test::setFunctionArch(func, GfxArchID::Gfx1250);
    BasicBlock* entry = func.createBasicBlock("entry");
    StinkyInstruction* add = test::createVAddInBlock(entry, GfxArchID::Gfx1250, 2, 0, 1);
    AttachedSSA ssa;
    ssa.results.push_back(func.ssaArena().createRegister(RegType::V));
    ssa.operands.push_back(makeSSAValueOperand(nullptr));
    ssa.operands.push_back(makeSSAValueOperand(nullptr));
    add->attachSSA(std::move(ssa));

    std::vector<Diagnostic> ssaErrors;
    for (const Diagnostic& diagnostic : verifyStinkyIR(func))
        if (diagnostic.getId() == DiagId::InvalidSSA) ssaErrors.push_back(diagnostic);
    ASSERT_FALSE(ssaErrors.empty());
    EXPECT_NE(ssaErrors[0].getMessage().find("value operand is null"), std::string::npos);
    ASSERT_NE(ssaErrors[0].getIRLocation(), nullptr);
    EXPECT_EQ(ssaErrors[0].getIRLocation()->function, "k");
    EXPECT_EQ(validateStinkyIR(func), "");
}

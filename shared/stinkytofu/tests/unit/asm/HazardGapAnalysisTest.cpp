// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "stinkytofu/analysis/asm/HazardGapAnalysisPass.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/pipeline/BackendRegistry.hpp"
#include "stinkytofu/pipeline/PassCatalog.hpp"
#include "stinkytofu/serialization/asm/IRConverter.hpp"
#include "stinkytofu/serialization/asm/IRParser.hpp"

using namespace stinkytofu;

namespace {
// An SALU write feeding a tensor load's address at once, and one that waits
// out the rule's 8 cycles with a label in between.
const char* const kTwoPairs = R"(st.func @k() {
^entry:
  s0 = "st.s_mov_b32"(s1) { issueCycles = 1, latencyCycles = 1 }
  "st.tensor_load_to_lds"(s[0:3], s[4:11]) { issueCycles = 1, latencyCycles = 1 }
  Successors: ^next
^next:
  s4 = "st.s_mov_b32"(s1) { issueCycles = 1, latencyCycles = 1 }
  mid:
  "st.s_nop"(7) { issueCycles = 8, latencyCycles = 8 }
  "st.tensor_load_to_lds"(s[0:3], s[4:11]) { issueCycles = 1, latencyCycles = 1 }
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
}  // namespace

TEST(HazardGapAnalysisTest, EachPairCarriesItsGapAndPositions) {
    Function func("k");
    load(kTwoPairs, func);

    const std::vector<HazardGap> pairs = findHazardGaps(func);
    ASSERT_EQ(pairs.size(), 2u);

    EXPECT_STREQ(pairs[0].rule, "SaluSgprToMemAddr");
    EXPECT_EQ(pairs[0].block->getLabel(), "entry");
    EXPECT_EQ(pairs[0].required, 8);
    EXPECT_EQ(pairs[0].gap, 0);
    EXPECT_FALSE(pairs[0].ok());
    EXPECT_EQ(pairs[0].producerIndex, 0);
    EXPECT_EQ(pairs[0].consumerIndex, 1);

    // The label is not counted, so the consumer is the block's third instruction.
    EXPECT_EQ(pairs[1].block->getLabel(), "next");
    EXPECT_EQ(pairs[1].gap, 8);
    EXPECT_TRUE(pairs[1].ok());
    EXPECT_EQ(pairs[1].producerIndex, 0);
    EXPECT_EQ(pairs[1].consumerIndex, 2);
    EXPECT_EQ(pairs[1].producer->getDestRegs().front().reg.idx, 4u);
}

TEST(HazardGapAnalysisTest, PassReportsTheViolationFromTheSamePairs) {
    BackendRegistry::registerAllBackends();
    Function func("k");
    load(kTwoPairs, func);
    PassManager passManager = makeStandalonePassManager({12, 5, 0});
    passManager.addPass(createHazardGapAnalysisPass());

    std::ostringstream captured;
    std::streambuf* original = std::cerr.rdbuf(captured.rdbuf());
    passManager.run(func);
    std::cerr.rdbuf(original);

    EXPECT_TRUE(passManager.getPassContext().getAnalysisFailed());
    EXPECT_NE(captured.str().find("VIOLATION rule=SaluSgprToMemAddr bb=entry "
                                  "producer[0]=s_mov_b32 consumer[1]=tensor_load_to_lds gap=0 "
                                  "required>=8"),
              std::string::npos)
        << captured.str();
    EXPECT_NE(captured.str().find("SaluSgprToMemAddr (>=8 cyc):  2 pair(s) checked,  1 VIOLATION"),
              std::string::npos)
        << captured.str();
}

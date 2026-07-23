// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// WaitCntValidationCheckPass unit tests.
//
// The validator is a read-only checker: it verifies the s_wait_* instructions
// present in the IR satisfy every register / LDS data dependency. On a missing
// wait it aborts via report_fatal_error, so the negative path is covered with a
// gtest death test (a lit/FileCheck test cannot express process abort without
// XFAIL; that path is covered separately by the .stir tests).
//
// Positive paths:
//   * a function with no async memory ops is trivially clean;
//   * running StinkyWaitCntInsertionPass first produces a correct wait plan
//     that the validator then accepts without aborting.
// Negative path:
//   * a ds_load feeding a WMMA with no intervening s_wait_dscnt aborts.

#include <gtest/gtest.h>

#include <array>
#include <string>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/serialization/asm/IRConverter.hpp"
#include "stinkytofu/transforms/asm/StinkyBuildImplicitDependencyPass.hpp"
#include "stinkytofu/transforms/asm/StinkyWaitCntInsertionPass.hpp"
#include "stinkytofu/transforms/asm/WaitCntValidationCheckPass.hpp"

using namespace stinkytofu;

namespace {

class WaitCntValidationCheckPassTest : public ::testing::Test {
   protected:
    void SetUp() override {
        gemmConfig.arch[0] = 12;
        gemmConfig.arch[1] = 5;
        gemmConfig.arch[2] = 0;
        gemmConfig.NumWaves = 4;
    }

    Function* parseIR(const std::string& irString, StinkyIRConverter& converter) {
        auto* func = converter.convertToFunction(irString);
        if (func) func->setGemmTileConfig(gemmConfig);
        return func;
    }

    void runImplicitDeps(Function& func) {
        PassContext passCtx;
        passCtx.setGemmTileConfig(gemmConfig);
        AnalysisManager am;
        registerAllAnalyses(am);
        auto pass = createStinkyBuildImplicitDependencyPass();
        pass->run(func, passCtx, am);
    }

    void runInsertion(Function& func) {
        PassContext passCtx;
        passCtx.setGemmTileConfig(gemmConfig);
        AnalysisManager am;
        registerAllAnalyses(am);
        auto pass = createStinkyWaitCntInsertionPass({});
        pass->run(func, passCtx, am);
    }

    void runValidation(Function& func) {
        PassContext passCtx;
        passCtx.setGemmTileConfig(gemmConfig);
        AnalysisManager am;
        registerAllAnalyses(am);
        auto pass = createWaitCntValidationCheckPass();
        pass->run(func, passCtx, am);
    }

    GemmTileConfig gemmConfig{};
    std::array<int, 3> arch{12, 5, 0};
};

// A function with only VALU ops has no async producers in flight, so the
// validator finds nothing to wait on and returns without aborting.
TEST_F(WaitCntValidationCheckPassTest, NoAsyncOpsIsClean) {
    std::string irString = R"(
st.func @no_async() {
^entry:
  v2 = "st.v_add_f32"(v0, v1) { issueCycles = 1, latencyCycles = 1 }
  v3 = "st.v_add_f32"(v2, v1) { issueCycles = 1, latencyCycles = 1 }
}
)";
    StinkyIRConverter converter(arch);
    Function* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);
    runValidation(*func);  // must not abort
}

// ds_load -> WMMA with the insertion pass run first: the inserted s_wait_dscnt
// satisfies the RAW dependency, so the validator accepts the plan.
TEST_F(WaitCntValidationCheckPassTest, InsertionThenValidationDoesNotAbort) {
    std::string irString = R"(
st.func @insertion_roundtrip() {
^entry:
  v[20:21] = "st.ds_load_b64"(v0) { issueCycles = 1, latencyCycles = 52, mod.memtoken = { tokens = [0] } }
  v[30:31] = "st.ds_load_b64"(v0) { issueCycles = 1, latencyCycles = 52, mod.memtoken = { tokens = [0] } }
  a[10:17] = "st.v_wmma_f32_16x16x32_bf16"(v[20:27], v[30:37], a[10:17]) { issueCycles = 4, latencyCycles = 8 }
}
)";
    StinkyIRConverter converter(arch);
    Function* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);
    runImplicitDeps(*func);
    runInsertion(*func);
    runValidation(*func);  // must not abort
}

// ds_load -> WMMA with NO s_wait_dscnt: the ds_load is still in flight on the DS
// counter when the WMMA reads it, so the validator reports a missing wait and
// aborts via report_fatal_error.
TEST_F(WaitCntValidationCheckPassTest, MissingWaitAborts) {
    std::string irString = R"(
st.func @missing_dscnt() {
^entry:
  v[20:21] = "st.ds_load_b64"(v0) { issueCycles = 1, latencyCycles = 52, mod.memtoken = { tokens = [0] } }
  a[10:17] = "st.v_wmma_f32_16x16x32_bf16"(v[20:27], v[30:37], a[10:17]) { issueCycles = 4, latencyCycles = 8 }
}
)";
    StinkyIRConverter converter(arch);
    Function* func = parseIR(irString, converter);
    ASSERT_NE(func, nullptr);
    runImplicitDeps(*func);
    EXPECT_DEATH(runValidation(*func), "missing s_wait");
}

}  // namespace

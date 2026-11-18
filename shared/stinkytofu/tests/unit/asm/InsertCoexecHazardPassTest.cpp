/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
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

#include <array>
#include <string>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/serialization/asm/IRConverter.hpp"
#include "stinkytofu/transforms/asm/InsertCoexecHazardPass.hpp"

using namespace stinkytofu;

namespace {

// A WMMA that accumulates the previous WMMA's D in place against its type's rule
// stops codegen via report_fatal_error, which FileCheck cannot express, so each
// broken rule is a death test here. Allowed chains are in
// tests/filecheck/wmma_accum_in_place_warning.stir.
class InsertCoexecHazardPassTest : public ::testing::Test {
   protected:
    std::array<int, 3> arch{12, 5, 0};

    void runPass(const std::string& body) {
        StinkyIRConverter converter(arch);
        Function* func = converter.convertToFunction("st.func @f() {\n^entry:\n" + body + "}\n");
        ASSERT_NE(func, nullptr);
        GemmTileConfig config;
        config.arch = arch;
        PassContext passCtx;
        passCtx.setGemmTileConfig(config);
        AnalysisManager am;
        registerAllAnalyses(am);
        createInsertCoexecHazardPass()->run(*func, passCtx, am);
    }
};

TEST_F(InsertCoexecHazardPassTest, F8K64MustSwitchAccumulatorEvenWithValuBetween) {
    EXPECT_DEATH(runPass(R"(
  v[0:7] = "st.v_wmma_f32_16x16x64_fp8_fp8"(v[20:27], v[30:37], v[0:7]) {}
  v100 = "st.v_add_f32"(v200, v201) { issueCycles = 1, latencyCycles = 5 }
  v[0:7] = "st.v_wmma_f32_16x16x64_fp8_fp8"(v[40:47], v[50:57], v[0:7]) {}
)"),
                 "switch the accumulator");
}

TEST_F(InsertCoexecHazardPassTest, F6xF6MustSwitchAccumulator) {
    EXPECT_DEATH(runPass(R"(
  v[0:7] = "st.v_wmma_f32_16x16x128_f8f6f4"(v[20:31], v[36:47], v[0:7]) { mod.matrix_fmt = { fmtA = MATRIX_FMT_FP6, fmtB = MATRIX_FMT_FP6 } }
  v[0:7] = "st.v_wmma_f32_16x16x128_f8f6f4"(v[52:63], v[68:79], v[0:7]) { mod.matrix_fmt = { fmtA = MATRIX_FMT_FP6, fmtB = MATRIX_FMT_FP6 } }
)"),
                 "switch the accumulator");
}

// SWMMAC accumulates into its own D, so a matching D alone is the chain.
TEST_F(InsertCoexecHazardPassTest, SwmmacMustSwitchAccumulator) {
    EXPECT_DEATH(runPass(R"(
  v[0:7] = "st.v_wmma_f32_16x16x32_bf16"(v[20:27], v[30:37], v[0:7]) {}
  v[0:7] = "st.v_swmmac_f32_16x16x64_f16"(v[40:47], v[50:65], v80) {}
)"),
                 "switch the accumulator");
}

TEST_F(InsertCoexecHazardPassTest, F8K128NeedsDifferentA) {
    EXPECT_DEATH(runPass(R"(
  v[0:7] = "st.v_wmma_f32_16x16x128_fp8_fp8"(v[20:35], v[36:51], v[0:7]) {}
  v[0:7] = "st.v_wmma_f32_16x16x128_fp8_fp8"(v[20:35], v[68:83], v[0:7]) {}
)"),
                 "needs different A and B");
}

TEST_F(InsertCoexecHazardPassTest, F8K128AAndBOfOneWmmaMustNotOverlap) {
    EXPECT_DEATH(runPass(R"(
  v[0:7] = "st.v_wmma_f32_16x16x128_f8f6f4"(v[20:35], v[36:51], v[0:7]) { mod.matrix_fmt = { fmtA = MATRIX_FMT_FP8, fmtB = MATRIX_FMT_FP8 } }
  v[0:7] = "st.v_wmma_f32_16x16x128_f8f6f4"(v[52:67], v[60:75], v[0:7]) { mod.matrix_fmt = { fmtA = MATRIX_FMT_FP8, fmtB = MATRIX_FMT_FP8 } }
)"),
                 "A and B of one WMMA must not overlap");
}

TEST_F(InsertCoexecHazardPassTest, Bf16PairNeedsDifferentB) {
    EXPECT_DEATH(runPass(R"(
  v[0:7] = "st.v_wmma_f32_16x16x32_bf16"(v[20:27], v[30:37], v[0:7]) {}
  v[0:7] = "st.v_wmma_f32_16x16x32_bf16"(v[40:47], v[30:37], v[0:7]) {}
)"),
                 "needs different A and B from its pair and the previous pair");
}

TEST_F(InsertCoexecHazardPassTest, Bf16PairMustNotReusePreviousPair) {
    EXPECT_DEATH(runPass(R"(
  v[0:7] = "st.v_wmma_f32_16x16x32_bf16"(v[20:27], v[30:37], v[0:7]) {}
  v[0:7] = "st.v_wmma_f32_16x16x32_bf16"(v[40:47], v[50:57], v[0:7]) {}
  v[0:7] = "st.v_wmma_f32_16x16x32_bf16"(v[60:67], v[70:77], v[0:7]) {}
  v[0:7] = "st.v_wmma_f32_16x16x32_bf16"(v[20:27], v[90:97], v[0:7]) {}
)"),
                 "needs different A and B from its pair and the previous pair");
}

TEST_F(InsertCoexecHazardPassTest, Bf16OddChainLeavesAnUnpairedWmma) {
    EXPECT_DEATH(runPass(R"(
  v[0:7] = "st.v_wmma_f32_16x16x32_bf16"(v[20:27], v[30:37], v[0:7]) {}
  v[0:7] = "st.v_wmma_f32_16x16x32_bf16"(v[40:47], v[50:57], v[0:7]) {}
  v[0:7] = "st.v_wmma_f32_16x16x32_bf16"(v[60:67], v[70:77], v[0:7]) {}
)"),
                 "must be paired with the next WMMA");
}

TEST_F(InsertCoexecHazardPassTest, F4xF4At16x16PairsLikeBf16) {
    EXPECT_DEATH(runPass(R"(
  v[0:7] = "st.v_wmma_f32_16x16x128_f8f6f4"(v[20:27], v[30:37], v[0:7]) { mod.matrix_fmt = { fmtA = MATRIX_FMT_FP4, fmtB = MATRIX_FMT_FP4 } }
  v[0:7] = "st.v_wmma_f32_16x16x128_f8f6f4"(v[40:47], v[50:57], v[0:7]) { mod.matrix_fmt = { fmtA = MATRIX_FMT_FP4, fmtB = MATRIX_FMT_FP4 } }
  v[0:7] = "st.v_wmma_f32_16x16x128_f8f6f4"(v[60:67], v[70:77], v[0:7]) { mod.matrix_fmt = { fmtA = MATRIX_FMT_FP4, fmtB = MATRIX_FMT_FP4 } }
)"),
                 "must be paired with the next WMMA");
}

}  // namespace

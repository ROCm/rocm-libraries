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

#include <array>
#include <string>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/serialization/asm/IRConverter.hpp"
#include "stinkytofu/transforms/asm/MemTokenConsistencyCheckPass.hpp"

using namespace stinkytofu;

namespace {

// MemTokenConsistencyCheckPass reports (via report_fatal_error -> std::abort)
// when a basic block mixes mem ops that carry MemTokenData with ones that do
// not. The failure path aborts the process, which a lit/FileCheck test cannot
// express without XFAIL -- so it is covered here with a gtest death test.
//
// `ds_load` is a MemToken candidate (IF_DSRead); `mod.memtoken` attaches the
// token. Two candidates are needed before the check engages: with only-tokenized
// or only-untokenized candidates the pass short-circuits (no diagnostic).
class MemTokenConsistencyCheckPassTest : public ::testing::Test {
   protected:
    std::array<int, 3> arch{12, 5, 0};

    void runPass(Function& func) {
        PassContext passCtx;
        AnalysisManager am;
        registerAllAnalyses(am);
        auto pass = createMemTokenConsistencyCheckPass();
        pass->run(func, passCtx, am);
    }
};

// Two tokenized ds_loads: consistent, so the pass returns without aborting.
TEST_F(MemTokenConsistencyCheckPassTest, AllTokenizedIsConsistent) {
    std::string irString = R"(
st.func @all_tokenized() {
^entry:
  v[0:1] = "st.ds_load_b64"(v10) { issueCycles = 1, latencyCycles = 52, mod.memtoken = { tokens = [100] } }
  v[2:3] = "st.ds_load_b64"(v11) { issueCycles = 1, latencyCycles = 52, mod.memtoken = { tokens = [200] } }
}
)";
    StinkyIRConverter converter(arch);
    Function* func = converter.convertToFunction(irString);
    ASSERT_NE(func, nullptr);
    runPass(*func);  // must not abort
}

// Two untokenized ds_loads: also consistent (the other short-circuit branch).
TEST_F(MemTokenConsistencyCheckPassTest, AllUntokenizedIsConsistent) {
    std::string irString = R"(
st.func @all_untokenized() {
^entry:
  v[0:1] = "st.ds_load_b64"(v10) { issueCycles = 1, latencyCycles = 52 }
  v[2:3] = "st.ds_load_b64"(v11) { issueCycles = 1, latencyCycles = 52 }
}
)";
    StinkyIRConverter converter(arch);
    Function* func = converter.convertToFunction(irString);
    ASSERT_NE(func, nullptr);
    runPass(*func);  // must not abort
}

// One tokenized and one untokenized ds_load: inconsistent -> the pass prints a
// diagnostic and aborts via report_fatal_error.
TEST_F(MemTokenConsistencyCheckPassTest, MixedTokenizationAborts) {
    std::string irString = R"(
st.func @mixed_tokens() {
^entry:
  v[0:1] = "st.ds_load_b64"(v10) { issueCycles = 1, latencyCycles = 52, mod.memtoken = { tokens = [100] } }
  v[2:3] = "st.ds_load_b64"(v11) { issueCycles = 1, latencyCycles = 52 }
}
)";
    StinkyIRConverter converter(arch);
    Function* func = converter.convertToFunction(irString);
    ASSERT_NE(func, nullptr);
    EXPECT_DEATH(runPass(*func), "inconsistent memory tokens");
}

}  // namespace

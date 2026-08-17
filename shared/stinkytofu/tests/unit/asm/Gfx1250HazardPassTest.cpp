// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Covers the one Gfx1250HazardPass case that aborts instead of inserting a
// drain: a multi-DWORD SMEM load overwriting its own source.
//
// Note: FileCheck cannot express an abort; every other case lives in
// tests/filecheck/gfx1250_xnack_hazard_test.stir.

#include <gtest/gtest.h>

#include <array>
#include <string>

#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/serialization/asm/IRConverter.hpp"
#include "stinkytofu/transforms/asm/Gfx1250HazardPass.hpp"

using namespace stinkytofu;

namespace {

class Gfx1250HazardPassTest : public ::testing::Test {
   protected:
    std::array<int, 3> arch{12, 5, 0};

    void runPass(Function& func) {
        GemmTileConfig config;
        config.arch = arch;

        // The pass's only gate; TensileLite forwards rocisa's archCaps here.
        AsmCapsConfig caps;
        caps.requiresXCntForVolatileVMEM = true;

        PassContext passCtx;
        passCtx.setGemmTileConfig(config);
        passCtx.setAsmCapsConfig(caps);

        // The pass reads no analysis, so an empty manager is enough.
        AnalysisManager am;
        createGfx1250HazardPass()->run(func, passCtx, am);
    }
};

// A release build compiles the assert away and only prints the error.
#ifndef NDEBUG
TEST_F(Gfx1250HazardPassTest, MultiDwordSmemSelfOverlapAborts) {
    std::string irString = R"(
st.func @smem_self_overlap() {
^entry:
  s[0:1] = "st.s_load_b64"(s[0:1])
}
)";
    StinkyIRConverter converter(arch);
    Function* func = converter.convertToFunction(irString);
    ASSERT_NE(func, nullptr);
    EXPECT_DEATH(runPass(*func), "overwrites one of its own source registers");
}
#endif

}  // namespace

// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Gfx1250HazardPass aborts when an SMEM instruction overwrites one of its own
// source registers: XNACK replay would re-execute it after the source is gone,
// and no s_wait_xcnt placement can bring the source back. The abort cannot be
// expressed in a lit/FileCheck test, so it is covered here with a death test.
// Every repairable case lives in tests/filecheck/gfx1250_xnack_hazard_test.stir.

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
TEST_F(Gfx1250HazardPassTest, SmemSelfOverlapAborts) {
    std::string irString = R"(
st.func @smem_self_overlap() {
^entry:
  s0 = "st.s_load_b32"(s[0:1])
}
)";
    StinkyIRConverter converter(arch);
    Function* func = converter.convertToFunction(irString);
    ASSERT_NE(func, nullptr);
    EXPECT_DEATH(runPass(*func), "overwrites one of its own source registers");
}
#endif

}  // namespace

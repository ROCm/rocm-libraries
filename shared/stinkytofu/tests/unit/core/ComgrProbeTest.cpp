// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include "stinkytofu/Config/Config.h"
#include "stinkytofu/hardware/ComgrProbe.hpp"
#include "stinkytofu/hardware/ToolchainCaps.hpp"

using namespace stinkytofu;

// GTEST_SKIP returns from the test, so this has to stay a macro.
#ifndef STINKYTOFU_HAS_COMGR
#define SKIP_WITHOUT_COMGR() \
    GTEST_SKIP() << "Built without amd_comgr; a standalone build does not need ROCm."
#else
#define SKIP_WITHOUT_COMGR() \
    do {                     \
    } while (0)
#endif

TEST(ComgrProbeTest, HasComgrSupport) {
    SKIP_WITHOUT_COMGR();
    EXPECT_TRUE(hasComgrSupport());
}

TEST(ComgrProbeTest, ValidInstructionAssembles) {
    constexpr const char* kIsa = "amdgcn-amd-amdhsa--gfx1250";
    if (!comgrSupportsIsa(kIsa)) {
        GTEST_SKIP() << "Installed comgr does not list " << kIsa
                     << "; skipping assembler round-trip test.";
    }
    EXPECT_TRUE(tryAssembleWithComgr("s_nop 0", kIsa, 32));
}

TEST(ComgrProbeTest, InvalidInstructionFails) {
    SKIP_WITHOUT_COMGR();
    EXPECT_FALSE(tryAssembleWithComgr("s_bogus_not_real 0", "amdgcn-amd-amdhsa--gfx1250", 32));
}

TEST(ComgrProbeTest, InvalidIsaFails) {
    SKIP_WITHOUT_COMGR();
    EXPECT_FALSE(tryAssembleWithComgr("s_nop 0", "amdgcn-amd-amdhsa--gfx0000", 32));
}

TEST(ToolchainCapsTest, ProbeGfx1250ReturnsNonNone) {
    constexpr const char* kIsa = "amdgcn-amd-amdhsa--gfx1250";
    if (!comgrSupportsIsa(kIsa)) {
        GTEST_SKIP() << "Installed comgr does not list " << kIsa
                     << "; ToolchainCaps cannot probe vgprMsbMode.";
    }
    auto caps = ToolchainCaps::probe(GfxArchID::Gfx1250);
    EXPECT_NE(caps.vgprMsbMode, VgprMsbMode::None);
}

TEST(ToolchainCapsTest, ProbeIsCached) {
    SKIP_WITHOUT_COMGR();
    auto caps1 = ToolchainCaps::probe(GfxArchID::Gfx1250);
    auto caps2 = ToolchainCaps::probe(GfxArchID::Gfx1250);
    EXPECT_EQ(caps1.vgprMsbMode, caps2.vgprMsbMode);
}

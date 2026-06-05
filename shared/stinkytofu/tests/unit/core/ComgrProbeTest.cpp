// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include "stinkytofu/hardware/ComgrProbe.hpp"
#include "stinkytofu/hardware/ToolchainCaps.hpp"

using namespace stinkytofu;

TEST(ComgrProbeTest, HasComgrSupport) {
    EXPECT_TRUE(hasComgrSupport());
}

TEST(ComgrProbeTest, ValidInstructionAssembles) {
    EXPECT_TRUE(tryAssembleWithComgr("s_nop 0", "amdgcn-amd-amdhsa--gfx1250", 32));
}

TEST(ComgrProbeTest, InvalidInstructionFails) {
    EXPECT_FALSE(tryAssembleWithComgr("s_bogus_not_real 0", "amdgcn-amd-amdhsa--gfx1250", 32));
}

TEST(ComgrProbeTest, InvalidIsaFails) {
    EXPECT_FALSE(tryAssembleWithComgr("s_nop 0", "amdgcn-amd-amdhsa--gfx0000", 32));
}

TEST(ToolchainCapsTest, ProbeGfx1250ReturnsNonNone) {
    auto caps = ToolchainCaps::probe(GfxArchID::Gfx1250);
    EXPECT_NE(caps.vgprMsbMode, VgprMsbMode::None);
}

TEST(ToolchainCapsTest, ProbeIsCached) {
    auto caps1 = ToolchainCaps::probe(GfxArchID::Gfx1250);
    auto caps2 = ToolchainCaps::probe(GfxArchID::Gfx1250);
    EXPECT_EQ(caps1.vgprMsbMode, caps2.vgprMsbMode);
}

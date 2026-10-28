// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include <array>

#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/core/Types.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/ToolchainCaps.hpp"
#include "stinkytofu/pipeline/BackendRegistry.hpp"
#include "stinkytofu/pipeline/PassCatalog.hpp"

using namespace stinkytofu;

TEST(StandalonePassManagerTest, FeatureDefaults) {
    const PassFeatureConfig config = standalonePassFeatureConfig();
    EXPECT_TRUE(config.loopConfig.unrollGemm);
    EXPECT_TRUE(config.dagFeatures.distributeGlobalRead);
}

TEST(StandalonePassManagerTest, StandaloneAsmCapsAddsBackendCapsToProbe) {
    BackendRegistry::registerAllBackends();
    const std::array<int, 3> gfx1250{12, 5, 0};
    const AsmCapsConfig caps = standaloneAsmCaps(gfx1250);

    EXPECT_EQ(caps.vgprMsbMode, ToolchainCaps::probe(getGfxArchID(gfx1250)).vgprMsbMode);
    EXPECT_TRUE(caps.requiresXCntForVolatileVMEM);
    EXPECT_TRUE(caps.enableXnackReplay);
}

TEST(StandalonePassManagerTest, Gfx1250Setup) {
    BackendRegistry::registerAllBackends();
    const PassManager passManager = makeStandalonePassManager({12, 5, 0});
    const PassContext& ctx = passManager.getPassContext();

    EXPECT_TRUE(ctx.getPassFeatureConfig().loopConfig.unrollGemm);
    EXPECT_TRUE(ctx.getPassFeatureConfig().dagFeatures.distributeGlobalRead);

    const GemmTileConfig& tile = ctx.getGemmTileConfig();
    EXPECT_EQ(tile.arch, (std::array<int, 3>{12, 5, 0}));
    EXPECT_EQ(tile.TileA0, 0u);
    EXPECT_EQ(tile.TileB0, 0u);
    EXPECT_EQ(tile.TileM0, 0u);

    EXPECT_TRUE(ctx.getAsmCapsConfig().requiresXCntForVolatileVMEM);
    EXPECT_TRUE(ctx.getAsmCapsConfig().enableXnackReplay);
    EXPECT_FALSE(ctx.getRemarksEnabled());
}

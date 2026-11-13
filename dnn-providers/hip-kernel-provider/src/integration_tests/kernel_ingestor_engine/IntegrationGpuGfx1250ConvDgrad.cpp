// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HKP_TEST_GFX1250_CONV_DGRAD_AOT)

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "IntegrationGpuConvDgradAot.hpp"

// ---- hipkernel:Gfx1250ConvDgradAot: the shipped dgrad catalog on the GPU ----
//
// The engine ships the gfx1250 dgrad instance list
// (IngestorGenerator/configs/gfx1250_conv_dgrad_aot_{fp16,bf16}.instances.json): per dtype
// implicit-GEMM instances (mem and wavelet), the streaming and windowed depthwise dgrad,
// and the wave32 WMMA pipelines (dense, small-C stem, patch embedding). Each kernel is
// built once against rocKE's probe problem and matched to a graph by its launch plan. The
// suites are IntegrationGpuConvDgradAot.hpp's.

namespace hip_kernel_provider::kernel_ingestor_engine::integration
{

using namespace conv_dgrad_aot_test;

namespace
{

struct Gfx1250Traits
{
    static constexpr const char* ENGINE_NAME = "hipkernel:Gfx1250ConvDgradAot";
    static constexpr const char* DESCRIPTOR_PREFIX = "gfx1250_conv_dgrad_aot";
    static constexpr const char* ARCH = "gfx1250";
    static constexpr const char* GRAPH_NAME = "gfx1250_conv_dgrad";
    static constexpr const char* CACHE_TAG = "gfx1250-conv-dgrad";
    static constexpr size_t SELECTABLE_INSTANCES = 109;
    static constexpr size_t HELPER_INSTANCES = 0;

    /// Every direct_variant a selectable kernel carries, "none" being implicit GEMM; the
    /// order of a DgradGraph's counts.
    static const Names& variants()
    {
        static const Names s_variants{"none",
                                      "depthwise_dgrad",
                                      "depthwise_dgrad_win",
                                      "dgrad_wmma",
                                      "dgrad_smallc_wmma",
                                      "dgrad_patch_wmma"};
        return s_variants;
    }
    /// The single-kernel direct pipelines (every gfx1250 direct one), which the fallback
    /// score ranks above implicit GEMM.
    static const Names& singleKernelVariants()
    {
        static const Names s_variants{"depthwise_dgrad",
                                      "depthwise_dgrad_win",
                                      "dgrad_wmma",
                                      "dgrad_smallc_wmma",
                                      "dgrad_patch_wmma"};
        return s_variants;
    }
    /// gfx1250 has no MFMA pipeline.
    static const Names& mfmaVariants()
    {
        static const Names s_variants;
        return s_variants;
    }

    // Graphs of the instance list's own shapes (batch 2, extents capped at 128) that
    // together reach every selectable instance; the counts are conv_dgrad_aot_plan's over
    // the catalog. Batch 2 makes a write past the last dX row of an image land in the next
    // one and fail.
    // clang-format off
    static const std::vector<DgradGraph>& graphs()
    {
        static const std::vector<DgradGraph> s_graphs{
            {"WmmaC640K640F3S1P1Hw48", {2, 640, 640, 48, 48, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {25, 0, 0, 27, 0, 0}, {25, 0, 0, 27, 0, 0}},
            {"WmmaC128K128F3S2P1Hw32", {2, 128, 128, 32, 32, 3, 3, 2, 2, 1, 1, 1, 1, 1}, {25, 0, 0, 10, 0, 0}, {25, 0, 0, 10, 0, 0}},
            {"DepthwiseWinC1024K1024F7S1P3G1024Hw7", {2, 1024, 1024, 7, 7, 7, 7, 1, 1, 3, 3, 1, 1, 1024}, {0, 0, 7, 0, 0, 0}, {0, 0, 7, 0, 0, 0}},
            {"DepthwiseC256K256F3S1P1G256Hw7", {2, 256, 256, 7, 7, 3, 3, 1, 1, 1, 1, 1, 1, 256}, {0, 1, 4, 0, 0, 0}, {0, 1, 4, 0, 0, 0}},
            {"DepthwiseWinC960K960F5S1P2G960Hw7", {2, 960, 960, 7, 7, 5, 5, 1, 1, 2, 2, 1, 1, 960}, {0, 0, 5, 0, 0, 0}, {0, 0, 5, 0, 0, 0}},
            {"DepthwiseC64K64F3S2P1G64Hw32", {2, 64, 64, 32, 32, 3, 3, 2, 2, 1, 1, 1, 1, 64}, {0, 3, 0, 0, 0, 0}, {0, 3, 0, 0, 0, 0}},
            {"WmmaC64K128F2S2P0Hw32", {2, 64, 128, 32, 32, 2, 2, 2, 2, 0, 0, 1, 1, 1}, {25, 0, 0, 3, 0, 0}, {25, 0, 0, 3, 0, 0}},
            {"SmallcWmmaC3K16F3S2P1Hw32", {2, 3, 16, 32, 32, 3, 3, 2, 2, 1, 1, 1, 1, 1}, {4, 0, 0, 0, 2, 0}, {4, 0, 0, 0, 2, 0}},
            {"DepthwiseC512K512F7S1P0G512Hw7", {2, 512, 512, 7, 7, 7, 7, 1, 1, 0, 0, 1, 1, 512}, {0, 1, 1, 0, 0, 0}, {0, 1, 1, 0, 0, 0}},
            {"WmmaC32K32F3S1P0Hw32", {2, 32, 32, 32, 32, 3, 3, 1, 1, 0, 0, 1, 1, 1}, {25, 0, 0, 2, 0, 0}, {25, 0, 0, 2, 0, 0}},
            {"DepthwiseC72K72F5S2P2G72Hw32", {2, 72, 72, 32, 32, 5, 5, 2, 2, 2, 2, 1, 1, 72}, {0, 2, 0, 0, 0, 0}, {0, 2, 0, 0, 0, 0}},
            {"WmmaC128K128F3S1P1G32Hw32", {2, 128, 128, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 32}, {0, 0, 0, 2, 0, 0}, {0, 0, 0, 2, 0, 0}},
            {"WmmaC512K512F3S1P1G32Hw16", {2, 512, 512, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 32}, {25, 0, 0, 6, 0, 0}, {25, 0, 0, 6, 0, 0}},
            {"WmmaC256K256F3S1P1G32Hw32", {2, 256, 256, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 32}, {25, 0, 0, 4, 0, 0}, {25, 0, 0, 4, 0, 0}},
            {"WmmaC512K512F3S2P1G32Hw28", {2, 512, 512, 28, 28, 3, 3, 2, 2, 1, 1, 1, 1, 32}, {25, 0, 0, 2, 0, 0}, {25, 0, 0, 2, 0, 0}},
            {"PatchWmmaC3K768F16S16P0Hw128", {2, 3, 768, 128, 128, 16, 16, 16, 16, 0, 0, 1, 1, 1}, {25, 0, 0, 0, 0, 2}, {25, 0, 0, 0, 0, 2}},
            {"SmallcWmmaC3K64F7S2P3Hw32", {2, 3, 64, 32, 32, 7, 7, 2, 2, 3, 3, 1, 1, 1}, {4, 0, 0, 0, 1, 0}, {4, 0, 0, 0, 1, 0}},
            {"SmallcWmmaC4K320F3S1P1Hw32", {2, 4, 320, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {4, 0, 0, 0, 1, 0}, {4, 0, 0, 0, 1, 0}},
            {"WmmaC256K256F3S1P2D2Hw32", {2, 256, 256, 32, 32, 3, 3, 1, 1, 2, 2, 2, 2, 1}, {25, 0, 0, 1, 0, 0}, {25, 0, 0, 1, 0, 0}},
            {"WmmaC256K256F3S2P1G32Hw32", {2, 256, 256, 32, 32, 3, 3, 2, 2, 1, 1, 1, 1, 32}, {25, 0, 0, 1, 0, 0}, {25, 0, 0, 1, 0, 0}},
            {"PatchWmmaC3K128F4S4P0Hw64", {2, 3, 128, 64, 64, 4, 4, 4, 4, 0, 0, 1, 1, 1}, {4, 0, 0, 0, 0, 1}, {4, 0, 0, 0, 0, 1}},
            {"WmmaC512K512F3S1P4D4Hw32", {2, 512, 512, 32, 32, 3, 3, 1, 1, 4, 4, 4, 4, 1}, {25, 0, 0, 1, 0, 0}, {25, 0, 0, 1, 0, 0}},
        };
        return s_graphs;
    }
    // clang-format on
};

} // namespace

// ---- Per graph: the unbenchmarked plan and a capped benchmark ----

class IntegrationGpuGfx1250ConvDgradAot : public ConvDgradAotHarness<Gfx1250Traits>
{
};

TEST_P(IntegrationGpuGfx1250ConvDgradAot, UnbenchmarkedPlanServesFromTheWholeCatalog)
{
    unbenchmarkedPlanServesFromTheWholeCatalog(GetParam());
}

TEST_P(IntegrationGpuGfx1250ConvDgradAot, CappedBenchmarkMatchesCpuReference)
{
    cappedBenchmarkMatchesCpuReference(GetParam());
}

TEST_F(IntegrationGpuGfx1250ConvDgradAot, ABatchPastTheChunkLimitRunsAsSlicesOnDevice)
{
    aBatchPastTheChunkLimitRunsAsSlicesOnDevice();
}

TEST_F(IntegrationGpuGfx1250ConvDgradAot, ServesGraphsDeclaringWiderAlignments)
{
    servesGraphsDeclaringWiderAlignments();
}

// ---- Per dtype: every catalog instance pinned and checked ----

class IntegrationGpuGfx1250ConvDgradAotCatalog : public ConvDgradAotHarness<Gfx1250Traits>
{
};

TEST_P(IntegrationGpuGfx1250ConvDgradAotCatalog, EveryInstanceMatchesCpuReference)
{
    everyInstanceMatchesCpuReference(GetParam());
}

INSTANTIATE_TEST_SUITE_P(AotCatalog,
                         IntegrationGpuGfx1250ConvDgradAot,
                         ::testing::ValuesIn(graphVariants(Gfx1250Traits::graphs())),
                         [](const auto& info) { return info.param.name; });

INSTANTIATE_TEST_SUITE_P(AotInstances,
                         IntegrationGpuGfx1250ConvDgradAotCatalog,
                         ::testing::ValuesIn(dtypeVariants()),
                         [](const auto& info) { return info.param.name; });

} // namespace hip_kernel_provider::kernel_ingestor_engine::integration

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HKP_TEST_GFX1250_CONV_DGRAD_AOT

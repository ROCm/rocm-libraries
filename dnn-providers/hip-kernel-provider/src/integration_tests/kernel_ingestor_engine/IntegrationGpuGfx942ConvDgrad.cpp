// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HKP_TEST_GFX942_CONV_DGRAD_AOT)

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "IntegrationGpuConvDgradAot.hpp"

// ---- hipkernel:Gfx942ConvDgradAot: the shipped dgrad catalog on the GPU ----
//
// The engine ships the gfx942 dgrad instance list
// (IngestorGenerator/configs/gfx942_conv_dgrad_aot_{fp16,bf16}.instances.json): gfx950's
// pipelines rebuilt for gfx942 (wave64) -- per dtype implicit-GEMM instances (mem, compv3
// and compv4), the streaming depthwise dgrad (most rows owning cpl = 2 channels a lane) and
// the windowed one, the fused, 4c and MFMA pipelines (baked and runtime channels), plus the
// MFMA helper kernels the engine never offers as candidates. The MFMA forward pass stages
// within gfx942's 64 KiB of LDS. Each kernel is built once against rocKE's probe problem and
// matched to a graph by its launch plan. The suites are IntegrationGpuConvDgradAot.hpp's.

namespace hip_kernel_provider::kernel_ingestor_engine::integration
{

using namespace conv_dgrad_aot_test;

namespace
{

struct Gfx942Traits
{
    static constexpr const char* ENGINE_NAME = "hipkernel:Gfx942ConvDgradAot";
    static constexpr const char* DESCRIPTOR_PREFIX = "gfx942_conv_dgrad_aot";
    static constexpr const char* ARCH = "gfx942";
    static constexpr const char* GRAPH_NAME = "gfx942_conv_dgrad";
    static constexpr const char* CACHE_TAG = "gfx942-conv-dgrad";
    // The instance list, per dtype (configs/gfx942_conv_dgrad_aot_<dtype>.instances.json).
    static constexpr size_t SELECTABLE_INSTANCES = 173;
    static constexpr size_t HELPER_INSTANCES = 8;

    /// Every direct_variant a selectable kernel carries, "none" being implicit GEMM; the
    /// order of a DgradGraph's counts.
    static const Names& variants()
    {
        static const Names s_variants{"none",
                                      "depthwise_dgrad",
                                      "depthwise_dgrad_win",
                                      "grouped_dgrad_fused",
                                      "grouped_dgrad_4c",
                                      "grouped_dgrad_mfma",
                                      "dgrad_mfma_rtc"};
        return s_variants;
    }
    /// The single-kernel direct pipelines, which the fallback score ranks first.
    static const Names& singleKernelVariants()
    {
        static const Names s_variants{
            "depthwise_dgrad", "depthwise_dgrad_win", "grouped_dgrad_fused", "grouped_dgrad_4c"};
        return s_variants;
    }
    /// The MFMA pipelines, which launch weight transforms before their compute pass.
    static const Names& mfmaVariants()
    {
        static const Names s_variants{"grouped_dgrad_mfma", "dgrad_mfma_rtc"};
        return s_variants;
    }

    // N is 2 wherever the kernel's grid allows it, so a write past the last dX row of an image
    // lands in the next one and fails. Odd spatial sizes leave partial tiles. Together the
    // graphs reach every selectable instance of the catalog; the counts are
    // conv_dgrad_aot_plan's over the gfx942 catalog.
    // clang-format off
    static const std::vector<DgradGraph>& graphs()
    {
        static const std::vector<DgradGraph> s_graphs{
            // Implicit GEMM: unit stride (one sub-GEMM), strides 2, 3x2 and 4 (several tilde
            // sub-GEMMs), dilation, a 1x1 strided filter whose sub-GEMMs leave dX holes (the
            // zero-fill), a nonsquare filter, patch embeddings (a folded width, a stride past
            // the filter), the dY-halo widths, grouped and strided grouped.
            {"Dense", {1, 32, 32, 10, 9, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {94, 0, 0, 1, 0, 0, 1}, {94, 0, 0, 1, 0, 0, 1}},
            {"Strided", {2, 32, 48, 15, 13, 3, 3, 2, 2, 1, 1, 1, 1, 1}, {72, 0, 0, 0, 0, 0, 0}, {72, 0, 0, 0, 0, 0, 0}},
            {"Stride3x2Dilation2x1", {1, 16, 24, 17, 13, 3, 3, 3, 2, 2, 1, 2, 1, 1}, {72, 0, 0, 0, 0, 0, 0}, {72, 0, 0, 0, 0, 0, 0}},
            {"PatchEmbedStride8", {2, 16, 32, 24, 16, 8, 8, 8, 8, 0, 0, 1, 1, 1}, {76, 0, 0, 0, 0, 0, 0}, {76, 0, 0, 0, 0, 0, 0}},
            {"PatchEmbedC3Stride4", {2, 3, 64, 32, 32, 4, 4, 4, 4, 0, 0, 1, 1, 1}, {10, 0, 0, 0, 0, 0, 0}, {10, 0, 0, 0, 0, 0, 0}},
            {"PatchEmbedC3Stride16", {2, 3, 64, 64, 64, 16, 16, 16, 16, 0, 0, 1, 1, 1}, {76, 0, 0, 0, 0, 0, 0}, {76, 0, 0, 0, 0, 0, 0}},
            {"PatchEmbedGapStride5", {2, 8, 64, 30, 30, 4, 4, 5, 5, 0, 0, 1, 1, 1}, {76, 0, 0, 0, 0, 0, 0}, {76, 0, 0, 0, 0, 0, 0}},
            {"FoldedStride2", {2, 64, 128, 16, 16, 2, 2, 2, 2, 0, 0, 1, 1, 1}, {76, 0, 0, 0, 0, 0, 0}, {76, 0, 0, 0, 0, 0, 0}},
            {"PointwiseStride2", {2, 32, 64, 14, 13, 1, 1, 2, 2, 0, 0, 1, 1, 1}, {76, 0, 0, 0, 0, 0, 0}, {76, 0, 0, 0, 0, 0, 0}},
            {"Dilated2", {2, 32, 32, 15, 17, 3, 3, 1, 1, 2, 2, 2, 2, 1}, {72, 0, 0, 0, 0, 0, 0}, {72, 0, 0, 0, 0, 0, 0}},
            {"Stride4Pad3Filter7", {1, 16, 16, 23, 21, 7, 7, 4, 4, 3, 3, 1, 1, 1}, {72, 0, 0, 0, 0, 0, 0}, {72, 0, 0, 0, 0, 0, 0}},
            {"NonsquareFilterStride1x2", {2, 24, 40, 11, 17, 3, 5, 1, 2, 1, 2, 1, 1, 1}, {72, 0, 0, 0, 0, 0, 0}, {72, 0, 0, 0, 0, 0, 0}},
            {"HaloWo32", {2, 128, 128, 12, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {124, 0, 0, 0, 0, 0, 0}, {124, 0, 0, 0, 0, 0, 0}},
            {"HaloWo8", {2, 128, 128, 12, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {124, 0, 0, 0, 0, 0, 0}, {124, 0, 0, 0, 0, 0, 0}},
            {"HaloWo16", {2, 128, 128, 12, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {124, 0, 0, 0, 0, 0, 0}, {124, 0, 0, 0, 0, 0, 0}},
            {"HaloFilter5Wo48", {1, 128, 128, 10, 48, 5, 5, 1, 1, 2, 2, 1, 1, 1}, {124, 0, 0, 0, 0, 0, 0}, {124, 0, 0, 0, 0, 0, 0}},
            {"Filter1x7", {2, 128, 128, 9, 17, 1, 7, 1, 1, 0, 3, 1, 1, 1}, {124, 0, 0, 0, 0, 0, 0}, {124, 0, 0, 0, 0, 0, 0}},
            // Channel counts that admit only narrower vector widths.
            {"GroupedG4", {2, 64, 96, 13, 13, 3, 3, 1, 1, 1, 1, 1, 1, 4}, {72, 0, 0, 0, 0, 0, 0}, {72, 0, 0, 0, 0, 0, 0}},
            {"GroupedG3Strided", {2, 48, 96, 15, 15, 3, 3, 2, 2, 1, 1, 1, 1, 3}, {72, 0, 0, 0, 0, 0, 0}, {72, 0, 0, 0, 0, 0, 0}},
            {"GroupedG4Cpg3", {2, 12, 32, 11, 11, 3, 3, 1, 1, 1, 1, 1, 1, 4}, {9, 0, 0, 0, 0, 0, 0}, {9, 0, 0, 0, 0, 0, 0}},
            {"C12K24", {2, 12, 24, 13, 13, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {9, 0, 0, 0, 0, 0, 0}, {9, 0, 0, 0, 0, 0, 0}},
            {"C6K8Strided", {2, 6, 8, 13, 12, 3, 3, 2, 2, 1, 1, 1, 1, 1}, {9, 0, 0, 0, 0, 0, 0}, {9, 0, 0, 0, 0, 0, 0}},
            {"C3K16", {2, 3, 16, 12, 12, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {9, 0, 0, 0, 0, 0, 0}, {9, 0, 0, 0, 0, 0, 0}},
            {"C16K20", {2, 16, 20, 11, 13, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {11, 0, 0, 0, 0, 0, 0}, {11, 0, 0, 0, 0, 0, 0}},
            {"C16K5", {2, 16, 5, 11, 11, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {9, 0, 0, 0, 0, 0, 0}, {9, 0, 0, 0, 0, 0, 0}},
            {"C12K320", {1, 12, 320, 14, 13, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {14, 0, 0, 0, 0, 0, 0}, {14, 0, 0, 0, 0, 0, 0}},
            // Depthwise (C = K = groups): the streaming and windowed kernels' filters, strides
            // and paddings. Most streaming rows own cpl = 2 channels a lane: 97 groups are
            // served by the windowed kernels only. 96 and 200 groups leave a partial channel
            // tile; odd and even H under stride 2.
            {"DepthwiseG96", {2, 96, 96, 14, 15, 3, 3, 1, 1, 1, 1, 1, 1, 96}, {0, 2, 5, 0, 0, 0, 0}, {0, 2, 5, 0, 0, 0, 0}},
            {"DepthwiseG200N1", {1, 200, 200, 9, 7, 3, 3, 1, 1, 1, 1, 1, 1, 200}, {0, 2, 5, 0, 0, 0, 0}, {0, 2, 5, 0, 0, 0, 0}},
            {"DepthwiseG97Cpl", {2, 97, 97, 14, 15, 3, 3, 1, 1, 1, 1, 1, 1, 97}, {0, 0, 5, 0, 0, 0, 0}, {0, 0, 5, 0, 0, 0, 0}},
            {"DepthwiseStridedG40OddH", {2, 40, 40, 17, 15, 3, 3, 2, 2, 1, 1, 1, 1, 40}, {0, 4, 0, 0, 0, 0, 0}, {0, 4, 0, 0, 0, 0, 0}},
            {"DepthwiseStridedG8EvenH", {2, 8, 8, 16, 18, 3, 3, 2, 2, 1, 1, 1, 1, 8}, {0, 4, 0, 0, 0, 0, 0}, {0, 4, 0, 0, 0, 0, 0}},
            {"Depthwise5x5G64", {3, 64, 64, 13, 11, 5, 5, 1, 1, 2, 2, 1, 1, 64}, {0, 1, 3, 0, 0, 0, 0}, {0, 1, 3, 0, 0, 0, 0}},
            {"Depthwise5x5StridedG32", {2, 32, 32, 15, 15, 5, 5, 2, 2, 2, 2, 1, 1, 32}, {0, 2, 0, 0, 0, 0, 0}, {0, 2, 0, 0, 0, 0, 0}},
            {"Depthwise7x7Pad0G64", {2, 64, 64, 16, 16, 7, 7, 1, 1, 0, 0, 1, 1, 64}, {0, 0, 1, 0, 0, 0, 0}, {0, 0, 1, 0, 0, 0, 0}},
            {"Depthwise7x7G72", {2, 72, 72, 13, 11, 7, 7, 1, 1, 3, 3, 1, 1, 72}, {0, 0, 5, 0, 0, 0, 0}, {0, 0, 5, 0, 0, 0, 0}},
            // The fused and 4c pipelines, 3x3 p1 s1 at each (C/groups, K/groups) they bake,
            // with N * ceil(Ho / block_h) at least the eight row bands an xcd_tiles row needs
            // (gfx942 bakes no fused cpg 4: the 4-channel groups go to the MFMA pipelines).
            {"GroupedC4G8", {8, 32, 32, 16, 15, 3, 3, 1, 1, 1, 1, 1, 1, 8}, {0, 0, 0, 0, 0, 1, 1}, {0, 0, 0, 0, 0, 1, 1}},
            {"FusedC8G8", {8, 64, 64, 16, 13, 3, 3, 1, 1, 1, 1, 1, 1, 8}, {72, 0, 0, 2, 0, 2, 1}, {72, 0, 0, 2, 0, 2, 1}},
            {"FusedC16G4", {8, 64, 64, 14, 16, 3, 3, 1, 1, 1, 1, 1, 1, 4}, {76, 0, 0, 1, 0, 0, 1}, {76, 0, 0, 1, 0, 0, 1}},
            {"FusedC24Dense", {8, 24, 24, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {72, 0, 0, 1, 0, 0, 1}, {72, 0, 0, 1, 0, 0, 1}},
            {"FusedC32G2", {8, 64, 64, 13, 16, 3, 3, 1, 1, 1, 1, 1, 1, 2}, {94, 0, 0, 1, 0, 0, 1}, {94, 0, 0, 1, 0, 0, 1}},
            {"FourCG64", {2, 256, 256, 14, 13, 3, 3, 1, 1, 1, 1, 1, 1, 64}, {0, 0, 0, 0, 1, 1, 1}, {0, 0, 0, 0, 1, 1, 1}},
            // The MFMA pipelines: the dense c4k320 and c48k192 rows (MfmaC4K320G2 is the same
            // tuple grouped), then one graph per runtime-channel binary's (k_atoms, m_tiles):
            // ceil(K / 16) and ceil(C / 16) of the transposed problem.
            {"MfmaC4K320Dense", {1, 4, 320, 14, 13, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {14, 0, 0, 0, 0, 1, 0}, {14, 0, 0, 0, 0, 1, 0}},
            {"MfmaC4K320G2", {1, 8, 640, 10, 11, 3, 3, 1, 1, 1, 1, 1, 1, 2}, {14, 0, 0, 0, 0, 1, 0}, {14, 0, 0, 0, 0, 1, 0}},
            {"MfmaC48K192", {1, 48, 192, 14, 13, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {124, 0, 0, 0, 0, 2, 0}, {124, 0, 0, 0, 0, 2, 0}},
            {"MfmaRtcC64K32", {1, 64, 32, 14, 13, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {94, 0, 0, 0, 0, 0, 1}, {94, 0, 0, 0, 0, 0, 1}},
            {"MfmaRtcC32K64", {1, 32, 64, 14, 13, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {124, 0, 0, 0, 0, 0, 1}, {124, 0, 0, 0, 0, 0, 1}},
            {"MfmaRtcC64K64", {1, 64, 64, 14, 13, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {124, 0, 0, 0, 0, 0, 2}, {124, 0, 0, 0, 0, 0, 2}},
            {"MfmaRtcC64K128", {1, 64, 128, 14, 13, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {124, 0, 0, 0, 0, 0, 1}, {124, 0, 0, 0, 0, 0, 1}},
            {"MfmaRtcC64K256", {1, 64, 256, 14, 13, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {124, 0, 0, 0, 0, 0, 2}, {124, 0, 0, 0, 0, 0, 2}},
            {"MfmaRtcC16K512", {1, 16, 512, 14, 13, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {124, 0, 0, 0, 0, 0, 1}, {124, 0, 0, 0, 0, 0, 1}},
        };
        return s_graphs;
    }
    // clang-format on
};

} // namespace

// ---- Per graph: the unbenchmarked plan and a capped benchmark ----

class IntegrationGpuGfx942ConvDgradAot : public ConvDgradAotHarness<Gfx942Traits>
{
};

TEST_P(IntegrationGpuGfx942ConvDgradAot, UnbenchmarkedPlanServesFromTheWholeCatalog)
{
    unbenchmarkedPlanServesFromTheWholeCatalog(GetParam());
}

TEST_P(IntegrationGpuGfx942ConvDgradAot, CappedBenchmarkMatchesCpuReference)
{
    cappedBenchmarkMatchesCpuReference(GetParam());
}

TEST_F(IntegrationGpuGfx942ConvDgradAot, ABatchPastTheChunkLimitRunsAsSlicesOnDevice)
{
    aBatchPastTheChunkLimitRunsAsSlicesOnDevice();
}

TEST_F(IntegrationGpuGfx942ConvDgradAot, ServesGraphsDeclaringWiderAlignments)
{
    servesGraphsDeclaringWiderAlignments();
}

// ---- Per dtype: every catalog instance pinned and checked ----

class IntegrationGpuGfx942ConvDgradAotCatalog : public ConvDgradAotHarness<Gfx942Traits>
{
};

TEST_P(IntegrationGpuGfx942ConvDgradAotCatalog, EveryInstanceMatchesCpuReference)
{
    everyInstanceMatchesCpuReference(GetParam());
}

INSTANTIATE_TEST_SUITE_P(AotCatalog,
                         IntegrationGpuGfx942ConvDgradAot,
                         ::testing::ValuesIn(graphVariants(Gfx942Traits::graphs())),
                         [](const auto& info) { return info.param.name; });

INSTANTIATE_TEST_SUITE_P(AotInstances,
                         IntegrationGpuGfx942ConvDgradAotCatalog,
                         ::testing::ValuesIn(dtypeVariants()),
                         [](const auto& info) { return info.param.name; });

} // namespace hip_kernel_provider::kernel_ingestor_engine::integration

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HKP_TEST_GFX942_CONV_DGRAD_AOT

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HKP_TEST_GFX950_CONV_DGRAD_AOT)

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "IntegrationGpuConvDgradAot.hpp"

// ---- hipkernel:Gfx950ConvDgradAot: the shipped dgrad catalog on the GPU ----
//
// The engine ships a curated list of rocKE dgrad sweep kernels
// (IngestorGenerator/configs/gfx950_conv_dgrad_aot_{fp16,bf16}.instances.json): per dtype
// implicit-GEMM instances (tilde, dY-halo, tap-outer K and patch forms) and every direct
// dgrad pipeline (kernel_family 1: streaming and windowed depthwise, fused, 4c, and the
// MFMA pipelines with baked and runtime channels), plus the MFMA helper kernels the engine
// never offers as candidates. Each kernel is built once against rocKE's probe problem and
// matched to a graph by its launch plan. The suites are IntegrationGpuConvDgradAot.hpp's.

namespace hip_kernel_provider::kernel_ingestor_engine::integration
{

using namespace conv_dgrad_aot_test;

namespace
{

struct Gfx950Traits
{
    static constexpr const char* ENGINE_NAME = "hipkernel:Gfx950ConvDgradAot";
    static constexpr const char* DESCRIPTOR_PREFIX = "gfx950_conv_dgrad_aot";
    static constexpr const char* ARCH = "gfx950";
    static constexpr const char* GRAPH_NAME = "gfx950_conv_dgrad";
    static constexpr const char* CACHE_TAG = "gfx950-conv-dgrad";
    static constexpr size_t SELECTABLE_INSTANCES = 203;
    static constexpr size_t HELPER_INSTANCES = 4;

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
    // graphs reach every selectable instance of the catalog.
    // clang-format off
    static const std::vector<DgradGraph>& graphs()
    {
        static const std::vector<DgradGraph> s_graphs{
            // Implicit GEMM: unit stride (one sub-GEMM), strides 2, 3x2 and 4 (several tilde
            // sub-GEMMs), dilation, a 1x1 strided filter whose sub-GEMMs leave dX holes (the
            // zero-fill), a nonsquare filter, patch embeddings (a folded width, a stride past
            // the filter), the dY-halo widths, grouped and strided grouped.
            {"Dense", {1, 32, 32, 10, 9, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {79, 0, 0, 6, 0, 0, 0}, {79, 0, 0, 6, 0, 0, 0}},
            {"Strided", {2, 32, 48, 15, 13, 3, 3, 2, 2, 1, 1, 1, 1, 1}, {73, 0, 0, 0, 0, 0, 0}, {73, 0, 0, 0, 0, 0, 0}},
            {"Stride3x2Dilation2x1", {1, 16, 24, 17, 13, 3, 3, 3, 2, 2, 1, 2, 1, 1}, {73, 0, 0, 0, 0, 0, 0}, {73, 0, 0, 0, 0, 0, 0}},
            {"PatchEmbedStride8", {2, 16, 32, 24, 16, 8, 8, 8, 8, 0, 0, 1, 1, 1}, {81, 0, 0, 0, 0, 0, 0}, {81, 0, 0, 0, 0, 0, 0}},
            {"PatchEmbedC3Stride4", {2, 3, 64, 32, 32, 4, 4, 4, 4, 0, 0, 1, 1, 1}, {12, 0, 0, 0, 0, 0, 0}, {12, 0, 0, 0, 0, 0, 0}},
            {"PatchEmbedC3Stride16", {2, 3, 64, 64, 64, 16, 16, 16, 16, 0, 0, 1, 1, 1}, {81, 0, 0, 0, 0, 0, 0}, {81, 0, 0, 0, 0, 0, 0}},
            {"PatchEmbedGapStride5", {2, 8, 64, 30, 30, 4, 4, 5, 5, 0, 0, 1, 1, 1}, {81, 0, 0, 0, 0, 0, 0}, {81, 0, 0, 0, 0, 0, 0}},
            {"FoldedStride2", {2, 64, 128, 16, 16, 2, 2, 2, 2, 0, 0, 1, 1, 1}, {81, 0, 0, 0, 0, 0, 0}, {81, 0, 0, 0, 0, 0, 0}},
            {"PointwiseStride2", {2, 32, 64, 14, 13, 1, 1, 2, 2, 0, 0, 1, 1, 1}, {81, 0, 0, 0, 0, 0, 0}, {81, 0, 0, 0, 0, 0, 0}},
            {"Dilated2", {2, 32, 32, 15, 17, 3, 3, 1, 1, 2, 2, 2, 2, 1}, {73, 0, 0, 0, 0, 0, 0}, {73, 0, 0, 0, 0, 0, 0}},
            {"Stride4Pad3Filter7", {1, 16, 16, 23, 21, 7, 7, 4, 4, 3, 3, 1, 1, 1}, {73, 0, 0, 0, 0, 0, 0}, {73, 0, 0, 0, 0, 0, 0}},
            {"NonsquareFilterStride1x2", {2, 24, 40, 11, 17, 3, 5, 1, 2, 1, 2, 1, 1, 1}, {73, 0, 0, 0, 0, 0, 0}, {73, 0, 0, 0, 0, 0, 0}},
            {"HaloWo32", {2, 128, 128, 12, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {124, 0, 0, 0, 0, 0, 0}, {124, 0, 0, 0, 0, 0, 0}},
            {"HaloWo8", {2, 128, 128, 12, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {132, 0, 0, 0, 0, 0, 0}, {132, 0, 0, 0, 0, 0, 0}},
            {"HaloWo16", {2, 128, 128, 12, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {129, 0, 0, 0, 0, 0, 0}, {129, 0, 0, 0, 0, 0, 0}},
            {"HaloFilter5Wo48", {1, 128, 128, 10, 48, 5, 5, 1, 1, 2, 2, 1, 1, 1}, {97, 0, 0, 0, 0, 0, 0}, {97, 0, 0, 0, 0, 0, 0}},
            {"Filter1x7", {2, 128, 128, 9, 17, 1, 7, 1, 1, 0, 3, 1, 1, 1}, {96, 0, 0, 0, 0, 0, 0}, {96, 0, 0, 0, 0, 0, 0}},
            // Channel counts that admit only narrower vector widths.
            {"GroupedG4", {2, 64, 96, 13, 13, 3, 3, 1, 1, 1, 1, 1, 1, 4}, {73, 0, 0, 0, 0, 0, 0}, {73, 0, 0, 0, 0, 0, 0}},
            {"GroupedG3Strided", {2, 48, 96, 15, 15, 3, 3, 2, 2, 1, 1, 1, 1, 3}, {73, 0, 0, 0, 0, 0, 0}, {73, 0, 0, 0, 0, 0, 0}},
            {"GroupedG4Cpg3", {2, 12, 32, 11, 11, 3, 3, 1, 1, 1, 1, 1, 1, 4}, {10, 0, 0, 0, 0, 0, 0}, {10, 0, 0, 0, 0, 0, 0}},
            {"C12K24", {2, 12, 24, 13, 13, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {10, 0, 0, 0, 0, 0, 0}, {10, 0, 0, 0, 0, 0, 0}},
            {"C6K8Strided", {2, 6, 8, 13, 12, 3, 3, 2, 2, 1, 1, 1, 1, 1}, {10, 0, 0, 0, 0, 0, 0}, {10, 0, 0, 0, 0, 0, 0}},
            {"C3K16", {2, 3, 16, 12, 12, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {10, 0, 0, 0, 0, 0, 0}, {10, 0, 0, 0, 0, 0, 0}},
            {"C16K20", {2, 16, 20, 11, 13, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {6, 0, 0, 0, 0, 0, 0}, {6, 0, 0, 0, 0, 0, 0}},
            {"C16K5", {2, 16, 5, 11, 11, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {3, 0, 0, 0, 0, 0, 0}, {3, 0, 0, 0, 0, 0, 0}},
            // Depthwise (C = K = groups): the streaming and windowed kernels' filters, strides
            // and paddings. 96 and 200 groups leave a partial 64-lane channel tile; odd and even
            // H under stride 2.
            {"DepthwiseG96", {2, 96, 96, 14, 15, 3, 3, 1, 1, 1, 1, 1, 1, 96}, {0, 1, 13, 0, 0, 0, 0}, {0, 1, 13, 0, 0, 0, 0}},
            {"DepthwiseG200N1", {1, 200, 200, 9, 7, 3, 3, 1, 1, 1, 1, 1, 1, 200}, {0, 1, 13, 0, 0, 0, 0}, {0, 1, 13, 0, 0, 0, 0}},
            {"DepthwiseStridedG40OddH", {2, 40, 40, 17, 15, 3, 3, 2, 2, 1, 1, 1, 1, 40}, {0, 5, 0, 0, 0, 0, 0}, {0, 5, 0, 0, 0, 0, 0}},
            {"DepthwiseStridedG8EvenH", {2, 8, 8, 16, 18, 3, 3, 2, 2, 1, 1, 1, 1, 8}, {0, 5, 0, 0, 0, 0, 0}, {0, 5, 0, 0, 0, 0, 0}},
            {"Depthwise5x5G64", {3, 64, 64, 13, 11, 5, 5, 1, 1, 2, 2, 1, 1, 64}, {0, 0, 6, 0, 0, 0, 0}, {0, 0, 6, 0, 0, 0, 0}},
            {"Depthwise5x5StridedG32", {2, 32, 32, 15, 15, 5, 5, 2, 2, 2, 2, 1, 1, 32}, {0, 3, 0, 0, 0, 0, 0}, {0, 3, 0, 0, 0, 0, 0}},
            {"Depthwise7x7Pad0G64", {2, 64, 64, 16, 16, 7, 7, 1, 1, 0, 0, 1, 1, 64}, {0, 0, 2, 0, 0, 0, 0}, {0, 0, 2, 0, 0, 0, 0}},
            {"Depthwise7x7G72", {2, 72, 72, 13, 11, 7, 7, 1, 1, 3, 3, 1, 1, 72}, {0, 0, 10, 0, 0, 0, 0}, {0, 0, 10, 0, 0, 0, 0}},
            // The fused and 4c pipelines, 3x3 p1 s1 at each (C/groups, K/groups) they bake,
            // with N * ceil(Ho / block_h) at least the eight row bands an xcd_tiles row needs.
            {"FusedC4G8", {8, 32, 32, 16, 15, 3, 3, 1, 1, 1, 1, 1, 1, 8}, {0, 0, 0, 1, 0, 0, 0}, {0, 0, 0, 1, 0, 0, 0}},
            {"FusedC8G8", {8, 64, 64, 16, 13, 3, 3, 1, 1, 1, 1, 1, 1, 8}, {73, 0, 0, 3, 0, 0, 0}, {73, 0, 0, 3, 0, 0, 0}},
            {"FusedC16G4", {8, 64, 64, 14, 16, 3, 3, 1, 1, 1, 1, 1, 1, 4}, {73, 0, 0, 4, 0, 0, 0}, {73, 0, 0, 4, 0, 0, 0}},
            {"FusedC24Dense", {8, 24, 24, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {73, 0, 0, 2, 0, 0, 0}, {73, 0, 0, 2, 0, 0, 0}},
            {"FusedC32G2", {8, 64, 64, 13, 16, 3, 3, 1, 1, 1, 1, 1, 1, 2}, {79, 0, 0, 6, 0, 0, 0}, {79, 0, 0, 6, 0, 0, 0}},
            {"FourCG64", {2, 256, 256, 14, 13, 3, 3, 1, 1, 1, 1, 1, 1, 64}, {0, 0, 0, 0, 2, 0, 0}, {0, 0, 0, 0, 2, 0, 0}},
            // The MFMA pipelines: the dense c4k320 rows and the runtime-channel row; MfmaC4K320G2
            // is the same tuple grouped (the reorganize helper's group-offset bug this once
            // refused is fixed upstream of this PR's base, commit aa9ff78, #13075).
            {"MfmaC4K320Dense", {1, 4, 320, 14, 13, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {14, 0, 0, 0, 0, 3, 1}, {14, 0, 0, 0, 0, 3, 1}},
            {"MfmaRtcC12K320Dense", {1, 12, 320, 14, 13, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {14, 0, 0, 0, 0, 0, 1}, {14, 0, 0, 0, 0, 0, 1}},
            {"MfmaC4K320G2", {1, 8, 640, 10, 11, 3, 3, 1, 1, 1, 1, 1, 1, 2}, {14, 0, 0, 0, 0, 3, 1}, {14, 0, 0, 0, 0, 3, 1}},
        };
        return s_graphs;
    }
    // clang-format on
};

} // namespace

// ---- Per graph: the unbenchmarked plan and a capped benchmark ----

class IntegrationGpuGfx950ConvDgradAot : public ConvDgradAotHarness<Gfx950Traits>
{
};

TEST_P(IntegrationGpuGfx950ConvDgradAot, UnbenchmarkedPlanServesFromTheWholeCatalog)
{
    unbenchmarkedPlanServesFromTheWholeCatalog(GetParam());
}

TEST_P(IntegrationGpuGfx950ConvDgradAot, CappedBenchmarkMatchesCpuReference)
{
    cappedBenchmarkMatchesCpuReference(GetParam());
}

TEST_F(IntegrationGpuGfx950ConvDgradAot, ABatchPastTheChunkLimitRunsAsSlicesOnDevice)
{
    aBatchPastTheChunkLimitRunsAsSlicesOnDevice();
}

TEST_F(IntegrationGpuGfx950ConvDgradAot, ServesGraphsDeclaringWiderAlignments)
{
    servesGraphsDeclaringWiderAlignments();
}

// ---- Per dtype: every catalog instance pinned and checked ----

class IntegrationGpuGfx950ConvDgradAotCatalog : public ConvDgradAotHarness<Gfx950Traits>
{
};

TEST_P(IntegrationGpuGfx950ConvDgradAotCatalog, EveryInstanceMatchesCpuReference)
{
    everyInstanceMatchesCpuReference(GetParam());
}

INSTANTIATE_TEST_SUITE_P(AotCatalog,
                         IntegrationGpuGfx950ConvDgradAot,
                         ::testing::ValuesIn(graphVariants(Gfx950Traits::graphs())),
                         [](const auto& info) { return info.param.name; });

INSTANTIATE_TEST_SUITE_P(AotInstances,
                         IntegrationGpuGfx950ConvDgradAotCatalog,
                         ::testing::ValuesIn(dtypeVariants()),
                         [](const auto& info) { return info.param.name; });

} // namespace hip_kernel_provider::kernel_ingestor_engine::integration

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HKP_TEST_GFX950_CONV_DGRAD_AOT

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HKP_TEST_GFX942_CONV_FWD_AOT)

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "IntegrationGpuConvFwdAot.hpp"

// ---- hipkernel:Gfx942ConvFwdAot: the shipped forward catalog on the GPU ----
//
// The engine ships the gfx942 forward instance list
// (IngestorGenerator/configs/gfx942_conv_fwd_aot_{fp16,bf16}.instances.json): per dtype
// implicit-GEMM instances over the mem, compv3 and compv4 pipelines, group-merged depthwise
// ones (merge degrees 4, 16 and 32), and the std, spatial, tiled, col, grouped and
// non-grouped (16x16x16 and 32x32x8 MFMA atoms) direct kernels. Each is built once against
// rocKE's probe problem and matched to a graph by capability. The suites are
// IntegrationGpuConvFwdAot.hpp's: per graph and dtype the shipped-catalog trio, and per
// dtype a walk that runs every shipped instance on a graph it is selected for.

namespace hip_kernel_provider::kernel_ingestor_engine::integration
{

using namespace conv_fwd_aot_test;

namespace
{

constexpr FwdAotTarget GFX942_TARGET{"hipkernel:Gfx942ConvFwdAot",
                                     "gfx942",
                                     "gfx942_conv_fwd_aot",
                                     "gfx942-conv",
                                     "gfx942_conv_fwd"};

/// One graph and how many instances of each family the engine selects for it, as
/// gfx942_conv_fwd_aot_runtime_candidates in convolution_forward.py counts them over
/// gfx942_conv_fwd_aot_{fp16,bf16}.instances.json: the direct count per dtype, then the
/// implicit-GEMM count per dtype (the lists differ: only fp16 ships direct grouped and
/// grouped-halo kernels). A dtype that selects nothing for a graph is not run on it; every
/// graph selects some instance in one dtype at least. Selected is the engine's candidate list
/// (the graph's widest vector widths, or every serving instance when none bakes them;
/// every direct and group-merged instance that serves), not its serving set.
/// test_make_conv_aot_config.py recomputes this table from the lists and checks that
/// together the graphs select every instance of both: keep one row per line.
struct AotGraph
{
    std::string name;
    ConvProblem problem;
    size_t directFp16;
    size_t directBf16;
    size_t implicitFp16;
    size_t implicitBf16;
};

// Dense graphs whose channel counts pick each width combination the catalog ships, grouped
// and pointwise ones, group-merged depthwise (G divisible by 4, 16 and 32 or not), the
// depthwise filters of the tiled, spatial and col kernels ("valid" padding too), the
// non-grouped kernels' output widths, dilations and strides, the split-C kernels' deep
// channel runs, and the grouped-halo kernels.
// clang-format off
const std::vector<AotGraph>& gfx942AotGraphs()
{
    static const std::vector<AotGraph> s_graphs{
        {"Smoke", {2, 32, 32, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1}, 20, 20, 79, 79},
        {"Pointwise", {1, 32, 64, 8, 8, 1, 1, 1, 1, 0, 0, 1, 1, 1}, 0, 0, 79, 79},
        {"GroupedPointwiseG4", {2, 32, 32, 8, 8, 1, 1, 1, 1, 0, 0, 1, 1, 4}, 0, 0, 79, 79},
        {"Strided", {2, 32, 64, 17, 19, 3, 3, 2, 2, 1, 1, 1, 1, 1}, 1, 1, 79, 79},
        {"Nonsquare", {2, 32, 64, 11, 17, 3, 5, 1, 2, 1, 2, 1, 1, 1}, 0, 0, 79, 79},
        {"C3K16", {2, 3, 16, 15, 15, 3, 3, 1, 1, 1, 1, 1, 1, 1}, 0, 0, 18, 18},
        {"C6K16", {2, 6, 16, 13, 13, 3, 3, 1, 1, 1, 1, 1, 1, 1}, 0, 0, 18, 18},
        {"C12K24", {2, 12, 24, 13, 13, 3, 3, 1, 1, 1, 1, 1, 1, 1}, 0, 0, 18, 5},
        {"C64K3", {2, 64, 3, 11, 11, 3, 3, 1, 1, 1, 1, 1, 1, 1}, 0, 0, 8, 8},
        {"C64K12", {2, 64, 12, 11, 11, 3, 3, 1, 1, 1, 1, 1, 1, 1}, 0, 0, 8, 3},
        {"GroupedG2", {2, 64, 64, 13, 13, 3, 3, 1, 1, 1, 1, 1, 1, 2}, 5, 0, 79, 79},
        {"GroupedG8Cpg8", {2, 64, 64, 13, 15, 3, 3, 1, 1, 1, 1, 1, 1, 8}, 4, 0, 79, 79},
        {"GroupedG32Cpg4", {1, 128, 128, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 32}, 2, 0, 0, 0},
        {"Depthwise", {2, 32, 32, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 32}, 12, 10, 3, 3},
        {"DepthwiseG64", {2, 64, 64, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 64}, 10, 10, 3, 3},
        {"DepthwiseG48Strided5x5", {1, 48, 48, 23, 19, 5, 5, 2, 2, 2, 2, 1, 1, 48}, 3, 0, 3, 3},
        {"DepthwiseG24", {2, 24, 24, 13, 13, 3, 3, 1, 1, 1, 1, 1, 1, 24}, 12, 10, 2, 2},
        {"DepthwiseG96", {3, 96, 96, 20, 24, 3, 3, 1, 1, 1, 1, 1, 1, 96}, 10, 10, 3, 3},
        {"DepthwiseG5Strided", {3, 5, 5, 12, 12, 3, 3, 2, 2, 1, 1, 1, 1, 5}, 8, 8, 0, 0},
        {"DepthwiseG5Filter5", {2, 5, 5, 13, 13, 5, 5, 1, 1, 2, 2, 1, 1, 5}, 2, 0, 0, 0},
        {"DepthwiseG5Filter7", {2, 5, 5, 15, 15, 7, 7, 1, 1, 3, 3, 1, 1, 5}, 0, 6, 0, 0},
        {"DepthwiseG5Filter11", {1, 5, 5, 21, 21, 11, 11, 1, 1, 5, 5, 1, 1, 5}, 0, 1, 0, 0},
        {"DepthwiseG64Pad0", {2, 64, 64, 12, 12, 3, 3, 1, 1, 0, 0, 1, 1, 64}, 0, 0, 3, 3},
        {"NongroupedWo192", {1, 32, 32, 3, 192, 3, 3, 1, 1, 1, 1, 1, 1, 1}, 46, 46, 79, 79},
        {"NongroupedWo96Dilated4x4", {1, 64, 64, 10, 96, 3, 3, 1, 1, 4, 4, 4, 4, 1}, 1, 0, 79, 79},
        {"NongroupedWo96Dilated3x1", {1, 64, 64, 9, 96, 3, 3, 1, 1, 1, 1, 3, 1, 1}, 0, 0, 79, 79},
        {"NongroupedWo96Dilated1x2", {1, 64, 64, 5, 96, 3, 3, 1, 1, 2, 2, 1, 2, 1}, 0, 0, 79, 79},
        {"NongroupedWo96Dilated1x4", {1, 32, 32, 9, 96, 3, 3, 1, 1, 4, 4, 1, 4, 1}, 0, 0, 79, 79},
        {"NongroupedWo64Dilated2x2", {1, 64, 64, 12, 64, 3, 3, 1, 1, 2, 2, 2, 2, 1}, 5, 0, 79, 79},
        {"NongroupedWo191Strided", {1, 64, 64, 13, 191, 3, 3, 2, 2, 1, 1, 1, 1, 1}, 3, 3, 79, 79},
        {"NongroupedC16K32Wo80", {2, 16, 32, 3, 80, 3, 3, 1, 1, 1, 1, 1, 1, 1}, 13, 13, 79, 79},
        {"GroupedG8Cpg16", {2, 128, 128, 13, 15, 3, 3, 1, 1, 1, 1, 1, 1, 8}, 8, 0, 79, 79},
        {"GroupedG4Cpg16Odd", {1, 64, 64, 9, 17, 3, 3, 1, 1, 1, 1, 1, 1, 4}, 7, 0, 79, 79},
        {"NongroupedC64K32Wo48", {1, 64, 32, 9, 48, 3, 3, 1, 1, 1, 1, 1, 1, 1}, 37, 37, 79, 79},
        {"NongroupedC128K64Wo16", {2, 128, 64, 7, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1}, 26, 26, 79, 79},
        {"NongroupedSplitCC1024", {1, 1024, 32, 8, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1}, 55, 55, 79, 79},
        {"NongroupedSplitCC512Strided", {1, 512, 32, 15, 31, 3, 3, 2, 2, 1, 1, 1, 1, 1}, 8, 8, 79, 79},
        {"GroupedHaloG8Cpg32", {1, 256, 512, 8, 32, 3, 3, 1, 1, 1, 1, 1, 1, 8}, 14, 0, 79, 79},
        {"GroupedHaloG4Cpg16Strided", {1, 64, 128, 7, 31, 3, 3, 2, 2, 1, 1, 1, 1, 4}, 3, 0, 79, 79},
    };
    return s_graphs;
}
// clang-format on

/// gfx942AotGraphs() in @p dtype, with that dtype's selected counts; a graph the dtype
/// selects nothing for is left out.
std::vector<Variant> gfx942Graphs(DataType dtype)
{
    std::vector<Variant> variants;
    for(const auto& graph : gfx942AotGraphs())
    {
        const bool half = dtype == DataType::HALF;
        const auto direct = half ? graph.directFp16 : graph.directBf16;
        const auto implicit = half ? graph.implicitFp16 : graph.implicitBf16;
        if(direct + implicit == 0)
        {
            continue;
        }
        variants.push_back(
            {graph.name + dtypeSuffix(dtype), dtype, 0, graph.problem, direct, implicit});
    }
    return variants;
}

/// gfx942AotGraphs() in both dtypes.
std::vector<Variant> gfx942Variants()
{
    auto variants = gfx942Graphs(DataType::HALF);
    const auto bf16 = gfx942Graphs(DataType::BFLOAT16);
    variants.insert(variants.end(), bf16.begin(), bf16.end());
    return variants;
}

/// One parameter per dtype for the whole-catalog walk; its problem is unused.
std::vector<Variant> gfx942Dtypes()
{
    return {{std::string("Catalog") + dtypeSuffix(DataType::HALF), DataType::HALF, 0},
            {std::string("Catalog") + dtypeSuffix(DataType::BFLOAT16), DataType::BFLOAT16, 0}};
}

} // namespace

class IntegrationGpuGfx942ConvFwdAot : public IntegrationGpuConvFwdAotCatalog
{
protected:
    IntegrationGpuGfx942ConvFwdAot()
        : IntegrationGpuConvFwdAotCatalog(GFX942_TARGET)
    {
    }
};

TEST_P(IntegrationGpuGfx942ConvFwdAot, UnbenchmarkedPlanServesFromTheWholeCatalog)
{
    unbenchmarkedPlanServesFromTheWholeCatalog();
}

TEST_P(IntegrationGpuGfx942ConvFwdAot, SampledSelectedInstancesMatchCpuReference)
{
    sampledSelectedInstancesMatchCpuReference();
}

TEST_P(IntegrationGpuGfx942ConvFwdAot, CappedBenchmarkMatchesCpuReference)
{
    cappedBenchmarkMatchesCpuReference();
}

INSTANTIATE_TEST_SUITE_P(AotCatalog,
                         IntegrationGpuGfx942ConvFwdAot,
                         ::testing::ValuesIn(gfx942Variants()),
                         [](const auto& info) { return info.param.name; });

class IntegrationGpuGfx942ConvFwdAotCatalog : public IntegrationGpuConvFwdAotCatalog
{
protected:
    IntegrationGpuGfx942ConvFwdAotCatalog()
        : IntegrationGpuConvFwdAotCatalog(GFX942_TARGET)
    {
    }
};

TEST_P(IntegrationGpuGfx942ConvFwdAotCatalog, EveryInstanceMatchesCpuReference)
{
    everyInstanceMatchesCpuReference(gfx942Graphs(GetParam().dtype));
}

INSTANTIATE_TEST_SUITE_P(Dtype,
                         IntegrationGpuGfx942ConvFwdAotCatalog,
                         ::testing::ValuesIn(gfx942Dtypes()),
                         [](const auto& info) { return info.param.name; });

} // namespace hip_kernel_provider::kernel_ingestor_engine::integration

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HKP_TEST_GFX942_CONV_FWD_AOT

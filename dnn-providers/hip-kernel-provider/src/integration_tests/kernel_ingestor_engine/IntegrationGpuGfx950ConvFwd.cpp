// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "IntegrationGpuConvFwdAot.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::integration
{

using namespace conv_fwd_aot_test;

namespace
{

// Each engine's suites compile only where that engine ships: HKP_TEST_GFX950_CONV_FWD_AOT
// for the AOT-instance engine and HKP_TEST_GFX950_CONV_FWD_AOT_SEED for its test-only seed
// (src/integration_tests/CMakeLists.txt). The suites are IntegrationGpuConvFwdAot.hpp's.
constexpr FwdAotTarget GFX950_TARGET{"hipkernel:Gfx950ConvFwdAot",
                                     "gfx950",
                                     "gfx950_conv_fwd_aot",
                                     "gfx950-conv",
                                     "gfx950_conv_fwd"};

} // namespace

#if defined(HKP_TEST_GFX950_CONV_FWD_AOT) || defined(HKP_TEST_GFX950_CONV_FWD_AOT_SEED)

// ---- The AOT-instance engines' graphs ----
//
// hipkernel:Gfx950ConvFwdAot ships every instance rocKE's own forward dispatcher
// lists (dispatch/grouped_convolution_instances/gfx950_fwd.json), each built once
// against rocKE's probe problem and matched to a graph by capability.
// hipkernel:Gfx950ConvFwdAotSeed runs the same native code over a seed taken from the
// sweep, per dtype 12 implicit-GEMM instances, 4 std and spatial direct depthwise
// instances, and 22 each of the tiled, col, grouped and non-grouped direct kernels
// (test_descriptors/integration/rocKE/gfx950_conv_fwd_aot_seed). Both run the graphs of
// aotGraphs(), in both dtypes; the seed also runs aotSeedDirectGraphs().

namespace
{

/// One graph and how many instances of each family each catalog selects for it, as
/// gfx950_conv_fwd_aot_runtime_candidates in convolution_forward.py counts them over
/// gfx950_conv_fwd_aot_seed.instances.json and the full catalog's per-dtype lists,
/// gfx950_conv_fwd_aot_{fp16,bf16}.instances.json. Selected is the engine's selection
/// among the instances that serve the graph (only the graph's widest vector widths, or
/// every serving instance when none bakes them), so it is the engine's candidate list,
/// not its serving set. The seed and the direct family
/// are the same in both dtypes; the full catalog's implicit-GEMM count is stated per
/// dtype, since each dtype has a list of its own. The engines' kernel
/// matcher and candidate filter are checked against those counts, and
/// test_make_conv_aot_config.py recomputes this table from the lists: keep one row per
/// line.
struct AotGraph
{
    std::string name;
    ConvProblem problem;
    size_t seedDirect;
    size_t seedImplicit;
    size_t direct;
    size_t implicitFp16;
    size_t implicitBf16;
};

// A direct instance serves only graphs with its compiled filter, stride, padding and
// dilation; batch, image and group count are kernel arguments. In the seed: std 3x3/1 and
// 5x5/2, spatial (fewer than 64 groups) 3x3/1 and 7x7/2, and the tiled, col, grouped and
// non-grouped instances aotSeedDirectGraphs() is for, which also serve some graphs here
// (non-grouped the dense 3x3 ones whose C, K and Wo they tile).
// clang-format off
const std::vector<AotGraph>& aotGraphs()
{
    static const std::vector<AotGraph> s_graphs{
        {"Smoke", {}, 11, 4, 11, 58, 58},
        {"Pointwise", {1, 32, 64, 8, 8, 1, 1, 1, 1, 0, 0, 1, 1}, 0, 4, 0, 58, 58},
        // An instance is built against a 3x3 probe, so it is never the pointwise GEMM.
        {"GroupedPointwiseG4", {2, 32, 32, 8, 8, 1, 1, 1, 1, 0, 0, 1, 1, 4}, 0, 4, 0, 58, 58},
        {"Strided", {2, 32, 64, 17, 19, 3, 3, 2, 2, 1, 1, 1, 1}, 1, 4, 3, 58, 58},
        {"Dilated", {1, 32, 32, 15, 17, 3, 3, 1, 1, 2, 2, 2, 2}, 1, 4, 5, 58, 58},
        {"Nonsquare", {2, 32, 64, 11, 17, 3, 5, 1, 2, 1, 2, 1, 1}, 0, 4, 0, 58, 58},
        // Three channels per group: only instances 1 wide along C/groups and K/groups.
        {"GroupedG4Cpg3", {2, 12, 24, 11, 11, 3, 3, 1, 1, 1, 1, 1, 1, 4}, 0, 1, 0, 2, 2},
        // Eight channels per group: the grouped direct instances serve it too.
        {"GroupedG8Cpg8", {2, 64, 64, 13, 15, 3, 3, 1, 1, 1, 1, 1, 1, 8}, 5, 4, 7, 58, 58},
        {"Depthwise", {2, 32, 32, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 32}, 8, 3, 9, 8, 8},
        {"DepthwiseG5N3", {3, 5, 5, 9, 21, 3, 3, 1, 1, 1, 1, 1, 1, 5}, 8, 1, 3, 2, 2},
        // 64 groups and more: no spatial instance.
        {"DepthwiseG96", {3, 96, 96, 20, 24, 3, 3, 1, 1, 1, 1, 1, 1, 96}, 7, 3, 9, 8, 8},
        {"DepthwiseG256N1", {1, 256, 256, 12, 12, 3, 3, 1, 1, 1, 1, 1, 1, 256}, 7, 3, 9, 8, 8},
        {"DepthwiseG64Strided5x5", {2, 64, 64, 17, 17, 5, 5, 2, 2, 2, 2, 1, 1, 64}, 1, 3, 4, 8, 8},
        {"DepthwiseG48Strided5x5", {1, 48, 48, 23, 19, 5, 5, 2, 2, 2, 2, 1, 1, 48}, 1, 3, 4, 8, 8},
        {"DepthwiseG12Strided7x7", {2, 12, 12, 21, 15, 7, 7, 2, 2, 3, 3, 1, 1, 12}, 7, 2, 0, 4, 4},
        {"DepthwiseG40Strided7x7", {3, 40, 40, 16, 30, 7, 7, 2, 2, 3, 3, 1, 1, 40}, 7, 3, 0, 8, 8},
        // "Valid" padding: of the direct instances, only col serves it.
        {"DepthwiseG64Pad0", {2, 64, 64, 12, 12, 3, 3, 1, 1, 0, 0, 1, 1, 64}, 1, 3, 0, 8, 8},
        // C = 12 admits load widths up to 4 and K = 20 store widths up to 4.
        {"C12K20", {2, 12, 20, 13, 13, 3, 3, 1, 1, 1, 1, 1, 1}, 0, 1, 0, 2, 2},
        {"C24K40", {2, 24, 40, 9, 9, 3, 3, 1, 1, 1, 1, 1, 1}, 0, 4, 0, 58, 58},
        {"Wide", {4, 64, 128, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1}, 9, 4, 22, 58, 58},
        // Each selects a seed width pair that no row above selects: 2/2/2, 4/4/2, 8/8/4,
        // 1/1/4, the default epilogue's 4/4/1 and cshuffle 2/2/1. The shipped catalog builds
        // no 4/4/2, 1/1/8, 4/4/1 or 2/2/1, so for C12K6, C3K8, C12K5 and C6K5 it runs every
        // instance that serves.
        {"C6K10", {2, 6, 10, 13, 13, 3, 3, 1, 1, 1, 1, 1, 1}, 0, 1, 0, 2, 2},
        {"C12K6", {2, 12, 6, 11, 11, 3, 3, 1, 1, 1, 1, 1, 1}, 0, 1, 0, 2, 2},
        {"C16K12", {2, 16, 12, 9, 9, 3, 3, 1, 1, 1, 1, 1, 1}, 0, 1, 0, 4, 4},
        {"C3K8", {2, 3, 8, 15, 15, 3, 3, 1, 1, 1, 1, 1, 1}, 0, 2, 0, 4, 4},
        {"C12K5", {2, 12, 5, 12, 12, 3, 3, 1, 1, 1, 1, 1, 1}, 0, 1, 0, 2, 2},
        {"C6K5", {2, 6, 5, 11, 11, 3, 3, 1, 1, 1, 1, 1, 1}, 0, 1, 0, 2, 2},
    };
    return s_graphs;
}
// clang-format on

/// One graph for the seed's tiled, col, grouped and non-grouped direct instances, and how
/// many instances of each family the seed selects for it (the same in both dtypes),
/// counted as for aotGraphs(). Each such instance serves at least two of these graphs: images,
/// batches and group counts with tails, "valid" and "same" padding for col, every grouped
/// channel count, and every dilation the non-grouped sweep builds. Only the seed suite
/// runs them: the shipped catalog's suite samples its thousands of candidates per graph.
struct AotSeedGraph
{
    std::string name;
    ConvProblem problem;
    size_t seedDirect;
    size_t seedImplicit;
};

// clang-format off
const std::vector<AotSeedGraph>& aotSeedDirectGraphs()
{
    static const std::vector<AotSeedGraph> s_graphs{
        // Tiled depthwise: 3x3, 7x7, 15x15 and 31x31 filters, strides 1 and 2.
        {"TiledG32", {2, 32, 32, 13, 17, 3, 3, 1, 1, 1, 1, 1, 1, 32}, 8, 3},
        {"TiledG200", {1, 200, 200, 9, 7, 3, 3, 1, 1, 1, 1, 1, 1, 200}, 7, 3},
        {"TiledG5Strided", {3, 5, 5, 12, 12, 3, 3, 2, 2, 1, 1, 1, 1, 5}, 6, 1},
        {"TiledG96Strided", {1, 96, 96, 17, 14, 3, 3, 2, 2, 1, 1, 1, 1, 96}, 6, 3},
        {"TiledG24Strided7x7", {2, 24, 24, 15, 15, 7, 7, 2, 2, 3, 3, 1, 1, 24}, 7, 3},
        {"TiledG130Strided7x7", {1, 130, 130, 21, 9, 7, 7, 2, 2, 3, 3, 1, 1, 130}, 6, 1},
        {"TiledG16Filter15", {1, 16, 16, 20, 23, 15, 15, 1, 1, 7, 7, 1, 1, 16}, 1, 3},
        {"TiledG70Filter15", {2, 70, 70, 16, 16, 15, 15, 1, 1, 7, 7, 1, 1, 70}, 1, 1},
        {"TiledG8Filter31", {1, 8, 8, 40, 37, 31, 31, 2, 2, 15, 15, 1, 1, 8}, 2, 3},
        {"TiledG65Filter31", {1, 65, 65, 33, 31, 31, 31, 2, 2, 15, 15, 1, 1, 65}, 2, 1},
        // Column-streamed depthwise: "valid" (padding 0) and "same" padding.
        {"ColG40Valid", {2, 40, 40, 11, 19, 3, 3, 1, 1, 0, 0, 1, 1, 40}, 1, 3},
        {"ColG7Valid", {3, 7, 7, 6, 9, 3, 3, 1, 1, 0, 0, 1, 1, 7}, 1, 1},
        {"ColG24Strided5x5Valid", {2, 24, 24, 13, 17, 5, 5, 2, 2, 0, 0, 1, 1, 24}, 5, 3},
        {"ColG100Strided5x5Valid", {1, 100, 100, 18, 21, 5, 5, 2, 2, 0, 0, 1, 1, 100}, 5, 2},
        {"ColG24Strided5x5", {2, 24, 24, 13, 17, 5, 5, 2, 2, 2, 2, 1, 1, 24}, 1, 3},
        {"ColG72Filter9", {1, 72, 72, 20, 16, 9, 9, 1, 1, 4, 4, 1, 1, 72}, 10, 3},
        {"ColG9Filter9", {2, 9, 9, 13, 11, 9, 9, 1, 1, 4, 4, 1, 1, 9}, 10, 1},
        {"ColG9Strided11x11Valid", {2, 9, 9, 25, 27, 11, 11, 2, 2, 0, 0, 1, 1, 9}, 2, 1},
        {"ColG64Strided11x11Valid", {1, 64, 64, 31, 23, 11, 11, 2, 2, 0, 0, 1, 1, 64}, 2, 3},
        {"ColG12Filter7Valid", {2, 12, 12, 16, 18, 7, 7, 1, 1, 0, 0, 1, 1, 12}, 1, 2},
        {"ColG80Filter7Valid", {1, 80, 80, 9, 20, 7, 7, 1, 1, 0, 0, 1, 1, 80}, 1, 3},
        {"ColG20Strided9x9", {1, 20, 20, 23, 19, 9, 9, 2, 2, 4, 4, 1, 1, 20}, 4, 2},
        {"ColG66Strided9x9", {2, 66, 66, 17, 25, 9, 9, 2, 2, 4, 4, 1, 1, 66}, 4, 1},
        // Grouped, 4, 8, 16 and 32 channels per group.
        {"GroupedG16Cpg4", {2, 64, 64, 9, 19, 3, 3, 1, 1, 1, 1, 1, 1, 16}, 5, 1},
        {"GroupedG32Cpg4", {1, 128, 128, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 32}, 5, 1},
        {"GroupedG16Cpg8", {2, 128, 128, 9, 19, 3, 3, 1, 1, 1, 1, 1, 1, 16}, 6, 4},
        {"GroupedG48Cpg8", {1, 384, 384, 7, 33, 3, 3, 1, 1, 1, 1, 1, 1, 48}, 6, 4},
        {"GroupedG16Cpg16", {1, 256, 256, 12, 12, 3, 3, 1, 1, 1, 1, 1, 1, 16}, 10, 4},
        {"GroupedG32Cpg16", {2, 512, 512, 8, 17, 3, 3, 1, 1, 1, 1, 1, 1, 32}, 10, 4},
        {"GroupedG16Cpg32", {1, 512, 512, 9, 19, 3, 3, 1, 1, 1, 1, 1, 1, 16}, 9, 4},
        {"GroupedG8Cpg32", {2, 256, 256, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 8}, 9, 4},
        // Non-grouped: output widths every swept tile width divides, row and channel tails,
        // stride 2 and the swept dilations.
        {"NongroupedC64K96", {1, 64, 96, 10, 96, 3, 3, 1, 1, 1, 1, 1, 1, 1}, 10, 4},
        {"NongroupedC32K64", {2, 32, 64, 7, 48, 3, 3, 1, 1, 1, 1, 1, 1, 1}, 8, 4},
        {"NongroupedC128K32", {1, 128, 32, 5, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1}, 9, 4},
        {"NongroupedWo192", {1, 32, 32, 3, 192, 3, 3, 1, 1, 1, 1, 1, 1, 1}, 14, 4},
        {"NongroupedWo160", {2, 64, 64, 2, 160, 3, 3, 1, 1, 1, 1, 1, 1, 1}, 11, 4},
        {"NongroupedWo128", {1, 64, 128, 4, 128, 3, 3, 1, 1, 1, 1, 1, 1, 1}, 11, 4},
        {"NongroupedWo384", {1, 16, 32, 2, 384, 3, 3, 1, 1, 1, 1, 1, 1, 1}, 11, 4},
        {"NongroupedWo80", {2, 32, 64, 3, 80, 3, 3, 1, 1, 1, 1, 1, 1, 1}, 9, 4},
        {"NongroupedWo320", {1, 16, 32, 2, 320, 3, 3, 1, 1, 1, 1, 1, 1, 1}, 9, 4},
        {"NongroupedC64K64Strided", {1, 64, 64, 13, 191, 3, 3, 2, 2, 1, 1, 1, 1, 1}, 1, 4},
        {"NongroupedC48K96Strided", {2, 48, 96, 9, 95, 3, 3, 2, 2, 1, 1, 1, 1, 1}, 1, 4},
        {"NongroupedDilated2x2", {1, 64, 64, 12, 96, 3, 3, 1, 1, 2, 2, 2, 2, 1}, 1, 4},
        {"NongroupedDilated2x2C32", {2, 32, 32, 7, 32, 3, 3, 1, 1, 2, 2, 2, 2, 1}, 1, 4},
        {"NongroupedDilated3x3", {1, 64, 64, 8, 96, 3, 3, 1, 1, 3, 3, 3, 3, 1}, 1, 4},
        {"NongroupedDilated3x3C32", {2, 32, 32, 9, 32, 3, 3, 1, 1, 3, 3, 3, 3, 1}, 1, 4},
        {"NongroupedDilated4x4", {1, 64, 64, 10, 96, 3, 3, 1, 1, 4, 4, 4, 4, 1}, 1, 4},
        {"NongroupedDilated4x4C32", {2, 32, 32, 9, 32, 3, 3, 1, 1, 4, 4, 4, 4, 1}, 1, 4},
        {"NongroupedDilated3x1", {1, 64, 64, 9, 96, 3, 3, 1, 1, 1, 1, 3, 1, 1}, 1, 4},
        {"NongroupedDilated3x1C32", {2, 32, 32, 11, 32, 3, 3, 1, 1, 1, 1, 3, 1, 1}, 1, 4},
        {"NongroupedDilated3x1Strided", {1, 64, 96, 20, 191, 3, 3, 2, 2, 1, 1, 3, 1, 1}, 1, 4},
        {"NongroupedDilated3x1StridedC32", {2, 32, 32, 13, 63, 3, 3, 2, 2, 1, 1, 3, 1, 1}, 1, 4},
        {"NongroupedDilated1x2", {1, 64, 64, 5, 96, 3, 3, 1, 1, 2, 2, 1, 2, 1}, 1, 4},
        {"NongroupedDilated1x2C32", {2, 32, 32, 6, 32, 3, 3, 1, 1, 2, 2, 1, 2, 1}, 1, 4},
        {"NongroupedDilated1x4", {1, 32, 32, 9, 96, 3, 3, 1, 1, 4, 4, 1, 4, 1}, 1, 4},
        {"NongroupedDilated1x4C64", {1, 64, 96, 6, 48, 3, 3, 1, 1, 4, 4, 1, 4, 1}, 1, 4},
    };
    return s_graphs;
}
// clang-format on

/// aotGraphs() in both dtypes, with the seed's selected counts or the full catalog's; the
/// seed's also run aotSeedDirectGraphs().
std::vector<Variant> aotVariants(bool fullCatalog)
{
    std::vector<Variant> variants;
    for(const auto& graph : aotGraphs())
    {
        for(const auto dtype : {DataType::HALF, DataType::BFLOAT16})
        {
            const auto implicit = dtype == DataType::HALF ? graph.implicitFp16 : graph.implicitBf16;
            variants.push_back({graph.name + dtypeSuffix(dtype),
                                dtype,
                                0,
                                graph.problem,
                                fullCatalog ? graph.direct : graph.seedDirect,
                                fullCatalog ? implicit : graph.seedImplicit});
        }
    }
    if(!fullCatalog)
    {
        for(const auto& graph : aotSeedDirectGraphs())
        {
            for(const auto dtype : {DataType::HALF, DataType::BFLOAT16})
            {
                variants.push_back({graph.name + dtypeSuffix(dtype),
                                    dtype,
                                    0,
                                    graph.problem,
                                    graph.seedDirect,
                                    graph.seedImplicit});
            }
        }
    }
    return variants;
}

} // namespace

#endif

#ifdef HKP_TEST_GFX950_CONV_FWD_AOT_SEED

// ---- hipkernel:Gfx950ConvFwdAotSeed: every selected AOT instance, pinned ----
//
// The seed is small enough that every instance selected for a graph is pinned and checked
// against the reference, which the full catalog's thousands per graph do not afford.
// Every seed instance is selected for at least one graph of aotGraphs().

namespace
{

constexpr const char* AOT_SEED_ENGINE_NAME = "hipkernel:Gfx950ConvFwdAotSeed";

/// Whether @p problem is pure depthwise (groups == C == K): the graphs the std, spatial,
/// tiled and col kernels serve, where an unbenchmarked plan prefers a direct instance.
bool isDepthwise(const ConvProblem& problem)
{
    return problem.groups == problem.c && problem.c == problem.k;
}

} // namespace

class IntegrationGpuGfx950ConvFwdAotSeed : public IntegrationGpuConvFwdAotServing
{
protected:
    IntegrationGpuGfx950ConvFwdAotSeed()
        : IntegrationGpuConvFwdAotServing(GFX950_TARGET)
    {
    }

    const char* engineName() const override
    {
        return AOT_SEED_ENGINE_NAME;
    }
};

TEST_P(IntegrationGpuGfx950ConvFwdAotSeed, EverySelectedInstanceMatchesCpuReference)
{
    verifyEveryServingKernel(31, 37);
}

TEST_P(IntegrationGpuGfx950ConvFwdAotSeed, UnbenchmarkedPlanServesACapableInstance)
{
    const auto& variant = GetParam();
    const ScopedPluginLogCapture capture(this);
    auto& recorder = capture.recorder();

    auto graph = buildGraph(variant.dtype, variant.problem);
    ASSERT_NO_FATAL_FAILURE(buildAndCompile(*graph, {disableBenchmarkingKnob()}));
    EXPECT_FALSE(selectedKernel(recorder).empty()) << recorder.getRecordedLogsAsString();
    ASSERT_NO_FATAL_FAILURE(executeAndVerifyStorage(*graph, variant.dtype, 41));
}

TEST_P(IntegrationGpuGfx950ConvFwdAotSeed, KernelFamilyKnobSelectsEachFamilysInstances)
{
    const auto& variant = GetParam();
    const ScopedPluginLogCapture capture(this);
    auto& recorder = capture.recorder();
    const auto catalog = variant.directArms + variant.implicitArms;

    auto implicitGemm = buildGraph(variant.dtype, variant.problem);
    ASSERT_NO_FATAL_FAILURE(buildAndCompile(
        *implicitGemm, {kernelFamilyKnob(IMPLICIT_GEMM_FAMILY), disableBenchmarkingKnob()}));
    EXPECT_TRUE(recorder.hasLogContaining("at rank 0 from " + candidateCount(variant.implicitArms)
                                          + " (" + std::to_string(catalog)
                                          + " before knob filtering)"))
        << recorder.getRecordedLogsAsString();
    const auto implicitKernel = selectedKernel(recorder);
    ASSERT_NO_FATAL_FAILURE(executeAndVerifyStorage(*implicitGemm, variant.dtype, 43));
    implicitGemm.reset();
    if(variant.directArms == 0)
    {
        return;
    }

    recorder.clearLogs();
    auto direct = buildGraph(variant.dtype, variant.problem);
    ASSERT_NO_FATAL_FAILURE(buildAndCompile(
        *direct, {kernelFamilyKnob(DIRECT_DEPTHWISE_FAMILY), disableBenchmarkingKnob()}));
    EXPECT_TRUE(recorder.hasLogContaining("at rank 0 from " + candidateCount(variant.directArms)
                                          + " (" + std::to_string(catalog)
                                          + " before knob filtering)"))
        << recorder.getRecordedLogsAsString();
    const auto directKernel = selectedKernel(recorder);
    EXPECT_NE(directKernel, implicitKernel);
    ASSERT_NO_FATAL_FAILURE(executeAndVerifyStorage(*direct, variant.dtype, 47));
    direct.reset();

    // Unbenchmarked, a depthwise direct instance ranks above every implicit-GEMM one; the
    // grouped and non-grouped direct kernels rank below them (aotScore).
    recorder.clearLogs();
    auto unforced = buildGraph(variant.dtype, variant.problem);
    ASSERT_NO_FATAL_FAILURE(buildAndCompile(*unforced, {disableBenchmarkingKnob()}));
    EXPECT_EQ(selectedKernel(recorder),
              isDepthwise(variant.problem) || variant.implicitArms == 0 ? directKernel
                                                                        : implicitKernel)
        << recorder.getRecordedLogsAsString();
}

#endif // HKP_TEST_GFX950_CONV_FWD_AOT_SEED

#ifdef HKP_TEST_GFX950_CONV_FWD_AOT

// ---- hipkernel:Gfx950ConvFwdAot: the shipped catalog ----
//
// IntegrationGpuConvFwdAotCatalog's suites: an unbenchmarked plan with the selected count of
// each family, a sample of selected instances each pinned, and a capped benchmark. The seed
// suite above pins every selected instance.

class IntegrationGpuGfx950ConvFwdAot : public IntegrationGpuConvFwdAotCatalog
{
protected:
    IntegrationGpuGfx950ConvFwdAot()
        : IntegrationGpuConvFwdAotCatalog(GFX950_TARGET)
    {
    }
};

TEST_P(IntegrationGpuGfx950ConvFwdAot, UnbenchmarkedPlanServesFromTheWholeCatalog)
{
    unbenchmarkedPlanServesFromTheWholeCatalog();
}

TEST_P(IntegrationGpuGfx950ConvFwdAot, SampledSelectedInstancesMatchCpuReference)
{
    sampledSelectedInstancesMatchCpuReference();
}

TEST_P(IntegrationGpuGfx950ConvFwdAot, CappedBenchmarkMatchesCpuReference)
{
    cappedBenchmarkMatchesCpuReference();
}

#endif // HKP_TEST_GFX950_CONV_FWD_AOT

#ifdef HKP_TEST_GFX950_CONV_FWD_AOT
INSTANTIATE_TEST_SUITE_P(AotCatalog,
                         IntegrationGpuGfx950ConvFwdAot,
                         ::testing::ValuesIn(aotVariants(true)),
                         [](const auto& info) { return info.param.name; });
#endif

#ifdef HKP_TEST_GFX950_CONV_FWD_AOT_SEED
INSTANTIATE_TEST_SUITE_P(AotSeedInstances,
                         IntegrationGpuGfx950ConvFwdAotSeed,
                         ::testing::ValuesIn(aotVariants(false)),
                         [](const auto& info) { return info.param.name; });
#endif

} // namespace hip_kernel_provider::kernel_ingestor_engine::integration

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

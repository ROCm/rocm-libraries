// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>

#include "engines/kernel_ingestor_engine/packs/Gfx950ConvFwdAotSelection.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/ConvFwdAotTestSupport.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestGfx950ConvFwdAotSelection.cpp
 * @brief hipkernel:Gfx950ConvFwdAot's runtime selection: of the instances that serve a
 *        graph (the kernel matcher), the ones that run: the graph's widest vector
 *        widths, or every serving instance when none bakes them (the candidate filter
 *        registered under the kernel matcher's symbol).
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::conv_fwd_aot_test;
namespace conv = hip_kernel_provider::kernel_ingestor_engine::gfx950_conv_fwd;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

constexpr const char* AOT_GRAPH_MATCHER = "hipkernel.gfx950_conv_fwd_aot.graph_match";
constexpr const char* AOT_KERNEL_MATCHER = "hipkernel.gfx950_conv_fwd_aot.kernel_match";

/// One instance of each (epilogue, width pair, K loop) the sweep builds, at a 64x64 tile,
/// then four with the extreme M tiles of their width pairs. test_make_conv_aot_config.py
/// reads this table: keep one row per line.
// clang-format off
const std::vector<AotSample>& aotSample()
{
    static const std::vector<AotSample> s_sample{
        {"cshuffle", {8, 8, 8, 64, 64, 16, 1, 1, 32, 32, 16}, "compv3", {0, 0, 0, 0}},
        {"cshuffle", {8, 8, 4, 64, 64, 16, 1, 1, 32, 32, 16}, "compv3", {0, 0, 0, 0}},
        {"cshuffle", {8, 8, 2, 64, 64, 16, 1, 1, 32, 32, 16}, "compv3", {0, 0, 0, 0}},
        {"cshuffle", {8, 8, 1, 64, 64, 16, 1, 1, 32, 32, 16}, "compv3", {0, 0, 0, 0}},
        {"cshuffle", {4, 4, 8, 64, 64, 16, 1, 1, 32, 32, 16}, "compv3", {0, 0, 0, 0}},
        {"cshuffle", {4, 4, 4, 64, 64, 16, 1, 1, 32, 32, 16}, "compv3", {0, 0, 0, 0}},
        {"cshuffle", {4, 4, 2, 64, 64, 16, 1, 1, 32, 32, 16}, "compv3", {0, 0, 0, 0}},
        {"cshuffle", {4, 4, 1, 64, 64, 16, 1, 1, 32, 32, 16}, "compv3", {0, 0, 0, 0}},
        {"cshuffle", {2, 2, 8, 64, 64, 16, 1, 1, 32, 32, 16}, "compv3", {0, 0, 0, 0}},
        {"cshuffle", {2, 2, 4, 64, 64, 16, 1, 1, 32, 32, 16}, "compv3", {0, 0, 0, 0}},
        {"cshuffle", {2, 2, 2, 64, 64, 16, 1, 1, 32, 32, 16}, "compv3", {0, 0, 0, 0}},
        {"cshuffle", {2, 2, 1, 64, 64, 16, 1, 1, 32, 32, 16}, "compv3", {0, 0, 0, 0}},
        {"cshuffle", {1, 1, 8, 64, 64, 16, 1, 1, 32, 32, 16}, "compv3", {0, 0, 0, 0}},
        {"cshuffle", {1, 1, 4, 64, 64, 16, 1, 1, 32, 32, 16}, "compv3", {0, 0, 0, 0}},
        {"cshuffle", {1, 1, 2, 64, 64, 16, 1, 1, 32, 32, 16}, "compv3", {0, 0, 0, 0}},
        {"cshuffle", {1, 1, 1, 64, 64, 16, 1, 1, 32, 32, 16}, "compv3", {0, 0, 0, 0}},
        {"cshuffle", {8, 8, 8, 64, 64, 16, 1, 1, 32, 32, 16}, "mem", {0, 1, 8, 8}},
        {"cshuffle", {8, 8, 4, 64, 64, 16, 1, 1, 32, 32, 16}, "mem", {0, 1, 8, 8}},
        {"cshuffle", {8, 8, 2, 64, 64, 16, 1, 1, 32, 32, 16}, "mem", {0, 1, 8, 8}},
        {"cshuffle", {8, 8, 1, 64, 64, 16, 1, 1, 32, 32, 16}, "mem", {0, 1, 8, 8}},
        {"cshuffle", {4, 4, 8, 64, 64, 16, 1, 1, 32, 32, 16}, "mem", {0, 1, 8, 8}},
        {"cshuffle", {4, 4, 4, 64, 64, 16, 1, 1, 32, 32, 16}, "mem", {0, 1, 8, 8}},
        {"cshuffle", {4, 4, 2, 64, 64, 16, 1, 1, 32, 32, 16}, "mem", {0, 1, 8, 8}},
        {"cshuffle", {4, 4, 1, 64, 64, 16, 1, 1, 32, 32, 16}, "mem", {0, 1, 8, 8}},
        {"cshuffle", {2, 2, 8, 64, 64, 16, 1, 1, 32, 32, 16}, "mem", {0, 1, 8, 8}},
        {"cshuffle", {2, 2, 4, 64, 64, 16, 1, 1, 32, 32, 16}, "mem", {0, 1, 8, 8}},
        {"cshuffle", {2, 2, 2, 64, 64, 16, 1, 1, 32, 32, 16}, "mem", {0, 1, 8, 8}},
        {"cshuffle", {2, 2, 1, 64, 64, 16, 1, 1, 32, 32, 16}, "mem", {0, 1, 8, 8}},
        {"cshuffle", {1, 1, 8, 64, 64, 16, 1, 1, 32, 32, 16}, "mem", {0, 1, 8, 8}},
        {"cshuffle", {1, 1, 4, 64, 64, 16, 1, 1, 32, 32, 16}, "mem", {0, 1, 8, 8}},
        {"cshuffle", {1, 1, 2, 64, 64, 16, 1, 1, 32, 32, 16}, "mem", {0, 1, 8, 8}},
        {"cshuffle", {1, 1, 1, 64, 64, 16, 1, 1, 32, 32, 16}, "mem", {0, 1, 8, 8}},
        {"default", {8, 8, 1, 64, 64, 16, 1, 1, 32, 32, 16}, "compv3", {0, 0, 0, 0}},
        {"default", {4, 4, 1, 64, 64, 16, 1, 1, 32, 32, 16}, "compv3", {0, 0, 0, 0}},
        {"default", {2, 2, 1, 64, 64, 16, 1, 1, 32, 32, 16}, "compv3", {0, 0, 0, 0}},
        {"default", {1, 1, 1, 64, 64, 16, 1, 1, 32, 32, 16}, "compv3", {0, 0, 0, 0}},
        {"default", {8, 8, 1, 64, 64, 16, 1, 1, 32, 32, 16}, "mem", {0, 1, 8, 8}},
        {"default", {4, 4, 1, 64, 64, 16, 1, 1, 32, 32, 16}, "mem", {0, 1, 8, 8}},
        {"default", {2, 2, 1, 64, 64, 16, 1, 1, 32, 32, 16}, "mem", {0, 1, 8, 8}},
        {"default", {1, 1, 1, 64, 64, 16, 1, 1, 32, 32, 16}, "mem", {0, 1, 8, 8}},
        {"cshuffle", {8, 8, 8, 16, 64, 32, 1, 1, 16, 16, 32}, "compv3", {0, 0, 0, 0}},
        {"cshuffle", {4, 4, 4, 256, 64, 16, 1, 1, 32, 32, 16}, "compv3", {0, 0, 0, 0}},
        {"cshuffle", {2, 2, 2, 256, 64, 16, 1, 1, 32, 32, 16}, "compv3", {0, 0, 0, 0}},
        {"default", {4, 4, 1, 256, 64, 16, 1, 1, 32, 32, 16}, "compv3", {0, 0, 0, 0}},
    };
    return s_sample;
}

/// A graph, as (N, C, K, Hi, Wi, Y, X, sH, sW, pH, pW, dH, dW, groups), and the rows of
/// aotSample() selected for it, in table order. gfx950_conv_fwd_aot_runtime_candidates
/// computes them, and test_make_conv_aot_config.py checks this table against it: keep
/// one row per line.
struct SelectionCase
{
    std::array<int64_t, 14> problem;
    std::vector<size_t> selected;
};

const std::vector<SelectionCase>& selectionCases()
{
    static const std::vector<SelectionCase> s_cases{
        {{2, 64, 64, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {0, 16, 40}},
        {{2, 3, 32, 32, 32, 7, 7, 2, 2, 3, 3, 1, 1, 1}, {12}},
        {{2, 32, 5, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {3, 19, 32, 36}},
        {{1, 3, 5, 9, 9, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {15, 35}},
        {{2, 6, 16, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {8}},
        {{2, 4, 12, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {5, 41}},
        {{2, 12, 20, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 4}, {15, 35}},
        {{1, 9, 48, 10, 10, 5, 5, 1, 1, 2, 2, 1, 1, 3}, {12}},
        {{2, 32, 32, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 32}, {15, 35}},
        {{2, 24, 24, 14, 14, 1, 1, 1, 1, 0, 0, 1, 1, 3}, {0, 16, 40}},
        {{2, 16, 24, 17, 19, 3, 3, 2, 2, 2, 2, 2, 2, 1}, {0, 16, 40}},
        {{32, 8, 8, 192, 192, 1, 1, 1, 1, 0, 0, 1, 1, 1}, {0, 16}},
        {{36, 8, 8, 500, 500, 1, 1, 1, 1, 0, 0, 1, 1, 1}, {41, 42, 43}},
    };
    return s_cases;
}
// clang-format on

/// A windowed direct_depthwise_win instance (90 of the shipped catalog's 466 instances;
/// implicit_gemm, at 178, is the largest family), pure depthwise (cpg == kpg == 1),
/// compiled for the same 3x3 filter.
KernelDefinition winSampleKernel()
{
    auto kernel
        = sampleKernel({"direct", {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, "direct", {0, 0, 0, 0}});
    kernel.metadata["grouped"] = int64_t{0};
    kernel.metadata["kernel_family"] = int64_t{1};
    kernel.metadata["direct_variant"] = std::string("win");
    kernel.metadata["filter_h"] = kernel.metadata["filter_w"] = int64_t{3};
    kernel.metadata["stride_h"] = kernel.metadata["stride_w"] = int64_t{1};
    kernel.metadata["pad_h"] = kernel.metadata["pad_w"] = int64_t{1};
    kernel.metadata["dilation_h"] = kernel.metadata["dilation_w"] = int64_t{1};
    kernel.metadata["block_h"] = int64_t{12};
    kernel.metadata["block_waves"] = int64_t{8};
    kernel.metadata["cpg"] = int64_t{1};
    kernel.metadata["kpg"] = int64_t{1};
    kernel.metadata["prefetch_rows"] = int64_t{2};
    kernel.metadata["w_fold"] = int64_t{2};
    return kernel;
}

DeviceProperties gfx950()
{
    return deviceProperties("gfx950");
}

/// The engine's pipeline for one graph, as KernelIngestorStateManager runs it: the graph
/// match, the kernel matcher on every kernel, then the candidate filter registered under
/// the kernel matcher's symbol over what it admitted. Returns the indices of
/// @p kernels that serve the graph, and through @p selected the ones that are selected.
std::vector<size_t> serveAndSelect(const MatchContext& context,
                                   const std::vector<KernelDefinition>& kernels,
                                   std::vector<size_t>& selected)
{
    hip_kernel_provider::kernel_ingestor_engine::registerNativeIngestorSymbols();
    const auto bound = GraphMatchRegistry::resolve(AOT_GRAPH_MATCHER)(context);
    EXPECT_TRUE(bound.has_value());
    const auto matcher = KernelMatcherRegistry::resolve(AOT_KERNEL_MATCHER);
    const auto filter = CandidateFilterRegistry::resolve(AOT_KERNEL_MATCHER);

    std::vector<size_t> serving;
    std::vector<const KernelDefinition*> candidates;
    for(size_t i = 0; i < kernels.size(); ++i)
    {
        if(bound && matcher(context, *bound, kernels[i]))
        {
            serving.push_back(i);
            candidates.push_back(&kernels[i]);
        }
    }
    const auto keep = filter(context, bound.value_or(BoundTokens{}), candidates);
    EXPECT_EQ(keep.size(), candidates.size());
    selected.clear();
    for(size_t i = 0; i < keep.size() && i < serving.size(); ++i)
    {
        if(keep[i])
        {
            selected.push_back(serving[i]);
        }
    }
    return serving;
}

TEST(TestGfx950ConvFwdAotSelection, SelectsWhatTheRuntimeSelectionRunsOverTheSample)
{
    std::vector<KernelDefinition> kernels;
    for(const auto& sample : aotSample())
    {
        kernels.push_back(sampleKernel(sample));
    }
    for(const auto& [problem, expected] : selectionCases())
    {
        SCOPED_TRACE(::testing::PrintToString(problem));
        const GraphFixture fixture(convGraph(problem), gfx950());
        std::vector<size_t> selected;
        const auto serving = serveAndSelect(fixture.context(), kernels, selected);
        EXPECT_EQ(selected, expected);
        // Selection only narrows what serves.
        EXPECT_LE(selected.size(), serving.size());
        EXPECT_FALSE(selected.empty());
    }
}

TEST(TestGfx950ConvFwdAotSelection, KeepsEveryServingDirectInstance)
{
    // Depthwise 3x3: the direct instance and the scalar implicit-GEMM instances serve.
    std::vector<KernelDefinition> kernels{directSampleKernel()};
    for(const auto& sample : aotSample())
    {
        kernels.push_back(sampleKernel(sample));
    }
    const GraphFixture fixture(convGraph({2, 32, 32, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 32}),
                               gfx950());
    std::vector<size_t> selected;
    serveAndSelect(fixture.context(), kernels, selected);
    // Row r of aotSample() is kernel r + 1 here.
    EXPECT_EQ(selected, (std::vector<size_t>{0, 16, 36}));
}

TEST(TestGfx950ConvFwdAotSelection, KeepsAServingWinInstanceAlongsideImplicitGemm)
{
    // The candidate-filter tests here only ever named the generic "direct"
    // sample; direct_depthwise_win (90 of the shipped catalog's 466 instances)
    // was never exercised here by name.
    std::vector<KernelDefinition> kernels{winSampleKernel()};
    for(const auto& sample : aotSample())
    {
        kernels.push_back(sampleKernel(sample));
    }
    const GraphFixture fixture(convGraph({2, 32, 32, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 32}),
                               gfx950());
    std::vector<size_t> selected;
    const auto serving = serveAndSelect(fixture.context(), kernels, selected);
    ASSERT_FALSE(serving.empty());
    EXPECT_EQ(serving.front(), 0U) << "the win instance did not serve this depthwise problem";
}

std::shared_ptr<const void> foreignPayload(const KernelDefinition& /*kernel*/)
{
    return std::make_shared<const int>(0);
}

TEST(TestGfx950ConvFwdAotSelection, PreparedKernelsAndTheBoundProblemDecideAsTheGraphAndMetadataDo)
{
    hip_kernel_provider::kernel_ingestor_engine::registerNativeIngestorSymbols();
    const auto graphMatch = GraphMatchRegistry::resolve(AOT_GRAPH_MATCHER);
    const auto matcher = KernelMatcherRegistry::resolve(AOT_KERNEL_MATCHER);
    const auto filter = CandidateFilterRegistry::resolve(AOT_KERNEL_MATCHER);
    const auto score = ScoreRegistry::resolve("hipkernel.gfx950_conv_fwd_aot.score");
    const auto prepare = KernelPrepareRegistry::resolve(AOT_KERNEL_MATCHER);

    // The sample, a direct instance, a bf16 instance, and metadata no launch contract
    // describes.
    std::vector<KernelDefinition> plain{directSampleKernel()};
    for(const auto& sample : aotSample())
    {
        plain.push_back(sampleKernel(sample));
    }
    plain.push_back(sampleKernel(aotSample().front()));
    plain.back().metadata["dtype"] = std::string("bf16");
    plain.push_back(sampleKernel(aotSample().front()));
    plain.back().metadata["layout"] = std::string("NCHW");

    // What the engine's shared state carries, and a definition whose payload some other
    // preparer made, which the pack must not trust.
    auto prepared = plain;
    auto foreign = plain;
    for(size_t i = 0; i < plain.size(); ++i)
    {
        prepared[i].prepared = prepare(prepared[i]);
        prepared[i].preparedBy = prepare;
        ASSERT_NE(prepared[i].prepared, nullptr);
        foreign[i].prepared = foreignPayload(foreign[i]);
        foreign[i].preparedBy = &foreignPayload;
    }

    auto problems = selectionCases();
    problems.push_back({{2, 32, 32, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 32}, {}});
    for(const auto& [problem, expected] : problems)
    {
        SCOPED_TRACE(::testing::PrintToString(problem));
        const GraphFixture fixture(convGraph(problem), gfx950());
        const auto& context = fixture.context();
        const auto bound = graphMatch(context);
        ASSERT_TRUE(bound.has_value());

        std::vector<const KernelDefinition*> plainServing;
        std::vector<const KernelDefinition*> preparedServing;
        for(size_t i = 0; i < plain.size(); ++i)
        {
            // An empty bound makes the matcher read the graph itself.
            const bool served = matcher(context, BoundTokens{}, plain[i]);
            EXPECT_EQ(matcher(context, *bound, plain[i]), served) << i;
            EXPECT_EQ(matcher(context, *bound, prepared[i]), served) << i;
            EXPECT_EQ(matcher(context, *bound, foreign[i]), served) << i;
            EXPECT_EQ(score(context, *bound, prepared[i]), score(context, BoundTokens{}, plain[i]))
                << i;
            if(served)
            {
                plainServing.push_back(&plain[i]);
                preparedServing.push_back(&prepared[i]);
            }
        }
        EXPECT_FALSE(plainServing.empty());
        const auto kept = filter(context, *bound, preparedServing);
        EXPECT_EQ(kept, filter(context, BoundTokens{}, plainServing));
        if(!expected.empty())
        {
            std::vector<size_t> selectedSampleIndices;
            for(size_t position = 0; position < plainServing.size(); ++position)
            {
                const auto originalIndex
                    = static_cast<size_t>(plainServing[position] - plain.data());
                if(kept[position] && originalIndex >= 1 && originalIndex <= aotSample().size())
                {
                    selectedSampleIndices.push_back(originalIndex - 1);
                }
            }
            EXPECT_EQ(selectedSampleIndices, expected);
        }
    }
}

TEST(TestGfx950ConvFwdAotSelection, WidestVectorWidthIsTheLargestDividingPowerOfTwo)
{
    EXPECT_EQ(conv::widestVectorWidth(64), 8);
    EXPECT_EQ(conv::widestVectorWidth(12), 4);
    EXPECT_EQ(conv::widestVectorWidth(6), 2);
    EXPECT_EQ(conv::widestVectorWidth(5), 1);
}

TEST(TestGfx950ConvFwdAotSelection, PolicyKeepsTheProblemsWidestCombination)
{
    // Every entry already serves the problem. C/groups 16 and K/groups 12 admit 8/8/4,
    // and only the instances baking it run; the default epilogue stores scalars.
    EXPECT_EQ(conv::widestVectorSelection({{"cshuffle", 4, 4, 4},
                                           {"cshuffle", 8, 8, 2},
                                           {"cshuffle", 8, 8, 4},
                                           {"default", 8, 8, 1},
                                           {"cshuffle", 8, 8, 4}},
                                          16,
                                          12),
              (std::vector<bool>{false, false, true, false, true}));
    // An odd K / groups: every store is scalar, and both epilogues bake 8/8/1.
    EXPECT_EQ(conv::widestVectorSelection({{"cshuffle", 8, 8, 1},
                                           {"cshuffle", 4, 4, 1},
                                           {"default", 4, 4, 1},
                                           {"default", 8, 8, 1}},
                                          32,
                                          5),
              (std::vector<bool>{true, false, false, true}));
}

TEST(TestGfx950ConvFwdAotSelection, PolicyRunsEveryServingInstanceWithoutTheWidest)
{
    // C/groups 16 and K/groups 6 admit 8/8/2, which no instance bakes: every serving
    // instance runs, of either epilogue.
    EXPECT_EQ(conv::widestVectorSelection({{"cshuffle", 8, 8, 1},
                                           {"cshuffle", 2, 2, 2},
                                           {"default", 8, 8, 1},
                                           {"cshuffle", 1, 1, 1}},
                                          16,
                                          6),
              (std::vector<bool>{true, true, true, true}));
    EXPECT_TRUE(conv::widestVectorSelection({}, 16, 6).empty());
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

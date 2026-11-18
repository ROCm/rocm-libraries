// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>

#include "engines/kernel_ingestor_engine/packs/Gfx950ConvFwdAotSelection.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/ConvFwdAotTestSupport.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestGfx942ConvFwdAotSelection.cpp
 * @brief hipkernel:Gfx942ConvFwdAot's runtime selection: of the instances that serve a
 *        graph (the kernel matcher), the ones that run (the candidate filter registered
 *        under the kernel matcher's symbol). The gfx942 engine runs the selection policy of
 *        hipkernel:Gfx950ConvFwdAot (conv::widestVectorSelection, rocKE's
 *        conv_fwd_aot_runtime_candidates): the graph's widest vector widths, or every
 *        serving instance when none bakes them; direct and group-merged instances always
 *        stay. The sample is the gfx942 catalog's: one implicit-GEMM instance of each
 *        (epilogue, widths, merge degree) it ships. TestGfx942ConvFwdGeometry.cpp holds the
 *        same filter to the adapter over the whole catalog.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::conv_fwd_aot_test;
namespace conv = hip_kernel_provider::kernel_ingestor_engine::gfx950_conv_fwd;

/// One implicit-GEMM instance of the gfx942 catalog (fp16, grouped) and its merge degree:
/// a group-merged instance records its derived widths as 0.
struct Gfx942Sample
{
    AotSample sample;
    int64_t groupMerge;
};

/// One instance of each (epilogue, width triple, merge degree) the gfx942 catalog ships,
/// the first its fp16 list holds (mem first), then two more 8/8/8 ones of the other
/// pipelines. test_make_conv_aot_config.py reads this table: keep one row per line.
// clang-format off
const std::vector<Gfx942Sample>& gfx942Sample()
{
    static const std::vector<Gfx942Sample> s_sample{
        {{"cshuffle", {8, 8, 8, 16, 16, 32, 1, 1, 16, 16, 16}, "mem", {0, 0, 0, 0}}, 1},
        {{"cshuffle", {8, 8, 4, 16, 16, 64, 1, 1, 16, 16, 16}, "mem", {0, 0, 0, 0}}, 1},
        {{"cshuffle", {8, 8, 1, 16, 16, 64, 1, 1, 16, 16, 16}, "mem", {1, 0, 0, 0}}, 1},
        {{"cshuffle", {4, 4, 8, 64, 16, 16, 1, 1, 16, 16, 16}, "mem", {0, 0, 0, 0}}, 1},
        {{"cshuffle", {4, 4, 4, 32, 16, 32, 2, 1, 16, 16, 16}, "mem", {0, 0, 0, 0}}, 1},
        {{"cshuffle", {2, 2, 8, 64, 64, 64, 4, 2, 16, 16, 16}, "mem", {1, 0, 0, 0}}, 1},
        {{"cshuffle", {2, 2, 2, 32, 32, 16, 2, 1, 16, 16, 16}, "mem", {0, 0, 0, 0}}, 1},
        {{"cshuffle", {1, 1, 8, 32, 16, 16, 2, 1, 16, 16, 16}, "mem", {0, 0, 0, 0}}, 1},
        {{"cshuffle", {1, 1, 1, 32, 32, 64, 2, 2, 16, 16, 16}, "mem", {0, 0, 0, 0}}, 1},
        {{"default", {8, 8, 1, 16, 32, 64, 1, 2, 16, 16, 16}, "mem", {1, 0, 0, 0}}, 1},
        {{"default", {1, 1, 1, 32, 32, 32, 2, 2, 16, 16, 16}, "mem", {0, 0, 0, 0}}, 1},
        {{"cshuffle", {0, 0, 0, 32, 16, 64, 2, 1, 16, 16, 16}, "mem", {0, 0, 0, 0}}, 4},
        {{"cshuffle", {0, 0, 0, 32, 16, 32, 2, 1, 16, 16, 16}, "mem", {0, 0, 0, 0}}, 16},
        {{"cshuffle", {0, 0, 0, 128, 32, 64, 2, 2, 16, 16, 16}, "mem", {0, 0, 0, 0}}, 32},
        {{"cshuffle", {8, 8, 8, 64, 32, 32, 2, 1, 16, 16, 16}, "compv3", {0, 0, 0, 0}}, 1},
        {{"cshuffle", {8, 8, 8, 256, 128, 64, 4, 2, 32, 32, 8}, "compv4", {0, 0, 0, 0}}, 1},
    };
    return s_sample;
}

/// A graph, as (N, C, K, Hi, Wi, Y, X, sH, sW, pH, pW, dH, dW, groups), and the rows of
/// gfx942Sample() selected for it, in table order. gfx942_conv_fwd_aot_runtime_candidates
/// computes them, and test_make_conv_aot_config.py checks this table against it: keep one
/// row per line.
struct SelectionCase
{
    std::array<int64_t, 14> problem;
    std::vector<size_t> selected;
};

const std::vector<SelectionCase>& gfx942SelectionCases()
{
    static const std::vector<SelectionCase> s_cases{
        {{2, 64, 64, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {0, 14, 15}},
        {{2, 3, 32, 32, 32, 7, 7, 2, 2, 3, 3, 1, 1, 1}, {7}},
        {{2, 32, 5, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {2, 9}},
        {{2, 16, 12, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {1}},
        {{2, 12, 24, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {3}},
        {{2, 4, 12, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {4}},
        {{2, 6, 16, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {5}},
        {{2, 6, 10, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {6}},
        {{2, 12, 20, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 4}, {8, 10}},
        {{2, 24, 24, 14, 14, 1, 1, 1, 1, 0, 0, 1, 1, 3}, {0, 14, 15}},
        {{2, 16, 24, 17, 19, 3, 3, 2, 2, 2, 2, 2, 2, 1}, {0, 14, 15}},
        {{2, 32, 32, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 32}, {8, 10, 11, 12, 13}},
        {{2, 64, 64, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 64}, {8, 10, 11, 12, 13}},
        {{2, 24, 24, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 24}, {8, 10, 11}},
        {{1, 48, 48, 9, 9, 5, 5, 2, 2, 2, 2, 1, 1, 48}, {8, 10, 11, 12}},
        {{2, 20, 20, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 20}, {8, 10, 11}},
        {{2, 64, 128, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 64}, {8, 10}},
        {{2, 128, 128, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 64}, {6}},
        {{32, 8, 8, 192, 192, 1, 1, 1, 1, 0, 0, 1, 1, 1}, {14, 15}},
    };
    return s_cases;
}
// clang-format on

KernelDefinition gfx942SampleKernel(const Gfx942Sample& row)
{
    auto kernel = sampleKernel(row.sample);
    kernel.metadata["group_merge"] = row.groupMerge;
    return kernel;
}

std::vector<KernelDefinition> gfx942Kernels()
{
    std::vector<KernelDefinition> kernels;
    for(const auto& row : gfx942Sample())
    {
        kernels.push_back(gfx942SampleKernel(row));
    }
    return kernels;
}

/// The engine's pipeline for one graph, as KernelIngestorStateManager runs it: the graph
/// match, the kernel matcher on every kernel, then the candidate filter registered under
/// the kernel matcher's symbol over what it admitted. Returns the indices of @p kernels
/// that serve the graph, and through @p selected the ones that are selected.
std::vector<size_t> serveAndSelect(const FwdEngine& engine,
                                   const MatchContext& context,
                                   const std::vector<KernelDefinition>& kernels,
                                   std::vector<size_t>& selected)
{
    const auto bound = matchesGraph(engine.pack, context);
    EXPECT_TRUE(bound.has_value());
    const auto matcher = kernelMatcher(engine.pack);
    const auto filter = CandidateFilterRegistry::resolve(std::string(engine.pack.kernelMatcher));

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

DeviceProperties gfx942()
{
    return deviceProperties("gfx942");
}

TEST(TestGfx942ConvFwdAotSelection, SelectsWhatTheRuntimeSelectionRunsOverTheSampleAndMergedKernels)
{
    const auto kernels = gfx942Kernels();
    for(const auto& [problem, expected] : gfx942SelectionCases())
    {
        SCOPED_TRACE(::testing::PrintToString(problem));
        const GraphFixture fixture(convGraph(problem), gfx942());
        std::vector<size_t> selected;
        const auto serving = serveAndSelect(GFX942_FWD, fixture.context(), kernels, selected);
        EXPECT_EQ(selected, expected);
        // Selection only narrows what serves, and never to nothing.
        EXPECT_LE(selected.size(), serving.size());
        EXPECT_FALSE(selected.empty());
        // Every serving merged instance runs.
        for(const auto index : serving)
        {
            if(gfx942Sample()[index].groupMerge > 1)
            {
                EXPECT_NE(std::find(selected.begin(), selected.end(), index), selected.end())
                    << index;
            }
        }
    }
}

TEST(TestGfx942ConvFwdAotSelection, KeepsEveryServingDirectAndMergedInstance)
{
    // Depthwise 3x3 over 32 groups: the direct instance, the scalar implicit-GEMM ones and
    // every merge degree dividing 32 serve; the widest (1/1/1) leaves the scalar ones, and
    // the direct and merged instances all stay.
    std::vector<KernelDefinition> kernels{directSampleKernel()};
    for(auto& kernel : gfx942Kernels())
    {
        kernels.push_back(std::move(kernel));
    }
    const GraphFixture fixture(convGraph({2, 32, 32, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 32}),
                               gfx942());
    std::vector<size_t> selected;
    const auto serving = serveAndSelect(GFX942_FWD, fixture.context(), kernels, selected);
    // Row r of gfx942Sample() is kernel r + 1 here.
    EXPECT_EQ(selected, (std::vector<size_t>{0, 9, 11, 12, 13, 14}));
    EXPECT_EQ(serving, selected);
}

TEST(TestGfx942ConvFwdAotSelection, RunsTheSharedWidestVectorSelectionAsTheGfx950EngineDoes)
{
    // The gfx942 candidate filter is conv::widestVectorSelection over the serving
    // unmerged implicit-GEMM instances, and the gfx950 engine's filter decides the same
    // candidates the same way on its own device.
    hip_kernel_provider::kernel_ingestor_engine::registerNativeIngestorSymbols();
    const auto kernels = gfx942Kernels();
    const auto gfx942Filter
        = CandidateFilterRegistry::resolve(std::string(GFX942_FWD.pack.kernelMatcher));
    const auto gfx950Filter
        = CandidateFilterRegistry::resolve(std::string(GFX950_FWD.pack.kernelMatcher));
    size_t narrowed = 0;
    for(const auto& [problem, expected] : gfx942SelectionCases())
    {
        SCOPED_TRACE(::testing::PrintToString(problem));
        const GraphFixture onGfx942(convGraph(problem), gfx942());
        std::vector<size_t> selected;
        const auto serving = serveAndSelect(GFX942_FWD, onGfx942.context(), kernels, selected);

        std::vector<const KernelDefinition*> candidates;
        std::vector<conv::AotSelectionKey> keys;
        std::vector<size_t> unmerged;
        for(size_t j = 0; j < serving.size(); ++j)
        {
            const auto& row = gfx942Sample()[serving[j]];
            candidates.push_back(&kernels[serving[j]]);
            if(row.groupMerge == 1)
            {
                const auto& v = row.sample.values;
                keys.push_back({row.sample.epilogue, v[0], v[1], v[2]});
                unmerged.push_back(j);
            }
        }
        std::vector<bool> policy(serving.size(), true);
        const auto [n, c, k, hi, wi, y, x, sH, sW, pH, pW, dH, dW, groups] = problem;
        const auto widest = conv::widestVectorSelection(keys, c / groups, k / groups);
        ASSERT_EQ(widest.size(), unmerged.size());
        for(size_t j = 0; j < unmerged.size(); ++j)
        {
            policy[unmerged[j]] = widest[j];
        }
        const auto boundGfx942 = matchesGraph(GFX942_FWD.pack, onGfx942.context());
        ASSERT_TRUE(boundGfx942);
        EXPECT_EQ(gfx942Filter(onGfx942.context(), *boundGfx942, candidates), policy);

        const GraphFixture onGfx950(convGraph(problem), deviceProperties("gfx950"));
        const auto boundGfx950 = matchesGraph(GFX950_FWD.pack, onGfx950.context());
        ASSERT_TRUE(boundGfx950);
        EXPECT_EQ(gfx950Filter(onGfx950.context(), *boundGfx950, candidates), policy);
        narrowed += selected.size() < serving.size() ? 1 : 0;
    }
    // The policy narrows some of the cases.
    EXPECT_GT(narrowed, 0U);
}

std::shared_ptr<const void> foreignPayload(const KernelDefinition& /*kernel*/)
{
    return std::make_shared<const int>(0);
}

TEST(TestGfx942ConvFwdAotSelection, PreparedKernelsAndTheBoundProblemDecideAsTheGraphAndMetadataDo)
{
    const auto graphMatch = graphMatcher(GFX942_FWD.pack);
    const auto matcher = kernelMatcher(GFX942_FWD.pack);
    const auto filter
        = CandidateFilterRegistry::resolve(std::string(GFX942_FWD.pack.kernelMatcher));
    const auto score = scorer(GFX942_FWD.pack);
    const auto prepare = KernelPrepareRegistry::resolve(std::string(GFX942_FWD.pack.kernelMatcher));

    // The sample with its merged instances, a direct instance, then metadata no launch
    // contract describes: a merged instance recording a width, one with no or a zero
    // degree, a direct instance with a degree, and a bf16 instance.
    std::vector<KernelDefinition> plain{directSampleKernel()};
    for(auto& kernel : gfx942Kernels())
    {
        plain.push_back(std::move(kernel));
    }
    const auto merged = gfx942SampleKernel(gfx942Sample()[12]);
    ASSERT_EQ(merged.metadata.at("group_merge"), MetadataValue(int64_t{16}));
    plain.push_back(merged);
    plain.back().metadata["vector_size_c"] = int64_t{1};
    plain.push_back(merged);
    plain.back().metadata["group_merge"] = int64_t{0};
    plain.push_back(merged);
    plain.back().metadata.erase("group_merge");
    plain.push_back(directSampleKernel());
    plain.back().metadata["group_merge"] = int64_t{16};
    plain.push_back(merged);
    plain.back().metadata["dtype"] = std::string("bf16");
    constexpr size_t MALFORMED = 5;

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

    for(const auto& [problem, expected] : gfx942SelectionCases())
    {
        SCOPED_TRACE(::testing::PrintToString(problem));
        const GraphFixture fixture(convGraph(problem), gfx942());
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
            // A served kernel scores, a refused one of the graph's dtype ranks last. (The
            // score is only asked about admitted kernels, so it does not compare dtypes.)
            if(i + 1 < plain.size())
            {
                EXPECT_EQ(score(context, *bound, prepared[i]) > -1.0, served) << i;
            }
            if(i + MALFORMED >= plain.size())
            {
                EXPECT_FALSE(served) << i;
            }
            if(served)
            {
                plainServing.push_back(&plain[i]);
                preparedServing.push_back(&prepared[i]);
            }
        }
        EXPECT_FALSE(plainServing.empty());
        EXPECT_EQ(filter(context, *bound, preparedServing),
                  filter(context, BoundTokens{}, plainServing));
    }
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

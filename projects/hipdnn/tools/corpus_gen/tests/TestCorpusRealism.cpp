// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestCorpusRealism.cpp
 * @brief Measures what fraction of a generated corpus could come from a real workload.
 *
 * Validity is tested elsewhere; this checks realism. The predicate is written from real
 * networks, independent of the declared archetypes, so a drifting declaration cannot move it.
 */

#include <gtest/gtest.h>

#include <hipdnn_corpus_gen/ProblemSpace.hpp>
#include <hipdnn_corpus_gen/WorkloadSampling.hpp>

#include <nlohmann/json.hpp>

#include "OperationsDir.hpp"
#include <algorithm>
#include <climits>
#include <filesystem>
#include <fstream>
#include <random>
#include <set>
#include <string>

namespace hipdnn_corpus_gen
{
namespace
{

int64_t at(const ProblemPoint& point, const std::string& name)
{
    const auto found = point.find(name);
    return found == point.end() ? 0 : std::get<int64_t>(found->second);
}

/// Whether a convolution looks like one a network contains.
///
/// Square or common-aspect image, channels aligned to 8 (or a 3-channel input), a common
/// filter size, and padding no larger than "same".
bool looksLikeALayer(const ProblemPoint& point)
{
    const auto n = at(point, "N");
    const auto c = at(point, "C");
    const auto k = at(point, "K");
    const auto h = at(point, "H");
    const auto w = at(point, "W");
    const auto r = at(point, "R");
    const auto s = at(point, "S");
    const auto pad = at(point, "pad_h");
    const auto stride = at(point, "stride_h");
    const auto dilation = at(point, "dilation_h");

    if(n <= 0 || c <= 0 || k <= 0 || h <= 0 || w <= 0 || r <= 0 || s <= 0)
    {
        return false;
    }

    const auto aspect = static_cast<double>(w) / static_cast<double>(h);
    const bool plausibleShape
        = aspect == 1.0 || std::abs(aspect - 4.0 / 3.0) < 0.02 || std::abs(aspect - 2.0) < 0.02;

    const bool plausibleChannels = (c == 3 || c % 8 == 0) && k % 8 == 0;

    // Even sizes too: 2x2/stride 2 reductions and the 4x4/stride 4 ConvNeXt stem.
    const std::set<int64_t> filters{1, 2, 3, 4, 5, 7, 11, 16};
    const bool plausibleFilter = filters.count(r) == 1 && r == s;

    // Real networks pad anywhere from 0 to "same" (AlexNet's 11x11 uses pad 2), never beyond
    // what the filter reaches.
    const bool plausiblePadding = pad >= 0 && pad <= (dilation * (r - 1)) / 2;

    const std::set<int64_t> strides{1, 2, 4, 16};

    return plausibleShape && plausibleChannels && plausibleFilter && plausiblePadding
           && strides.count(stride) == 1;
}

OperationMetadata shippedConvolution()
{
    std::ifstream file(hipdnn_corpus_gen::test::operationsDir() + "/conv_fwd.opmeta.json");
    EXPECT_TRUE(file.good());
    auto load = parseOperationMetadata(nlohmann::json::parse(file));
    EXPECT_TRUE(load.ok()) << (load.errors.empty() ? "" : load.errors.front());
    return load.metadata.value_or(OperationMetadata{});
}

const ProblemOracle ACCEPT_EVERYTHING = [](const ProblemPoint&) { return true; };

struct Composition
{
    double realistic = 0.0;
    size_t total = 0;
    int64_t fromArchetypes = 0;
    int64_t fromNeighbourhood = 0;
    int64_t fromExploration = 0;
};

Composition compose(const ProblemCorpus& corpus)
{
    Composition composition;
    size_t realistic = 0;
    for(const auto& point : corpus.problems())
    {
        realistic += looksLikeALayer(point) ? 1 : 0;
    }
    for(const auto& combination : corpus.combinations)
    {
        composition.fromArchetypes += combination.fromArchetypes;
        composition.fromNeighbourhood += combination.fromNeighbourhood;
        composition.fromExploration += combination.fromExploration;
    }
    composition.total = corpus.problems().size();
    composition.realistic = composition.total == 0 ? 0.0
                                                   : static_cast<double>(realistic)
                                                         / static_cast<double>(composition.total);
    return composition;
}

} // namespace

TEST(TestCorpusRealism, MostOfTheShippedConvolutionCorpusLooksLikeRealLayers)
{
    ExplorationRequest request;
    request.pointsPerCombination = 300;
    request.seed = 11;

    const auto composition
        = compose(exploreProblemSpace(shippedConvolution(), request, ACCEPT_EVERYTHING));

    ASSERT_GT(composition.total, 0U);
    // Logged so the actual realism figure is visible, not just pass/fail.
    GTEST_LOG_(INFO) << composition.total << " shapes, " << (composition.realistic * 100.0)
                     << "% realistic; " << composition.fromArchetypes << " archetype, "
                     << composition.fromNeighbourhood << " neighbourhood, "
                     << composition.fromExploration << " exploration";
    EXPECT_GT(composition.realistic, 0.5)
        << "only " << (composition.realistic * 100.0) << "% of " << composition.total
        << " shapes resemble a layer; the uniform search this replaces managed 0.04%";
}

TEST(TestCorpusRealism, TheCorpusStillContainsShapesNobodyDeclared)
{
    // A corpus of only anchors teaches only the anchors; §5.4 keeps an exploration share.
    ExplorationRequest request;
    request.pointsPerCombination = 300;
    request.seed = 12;

    const auto composition
        = compose(exploreProblemSpace(shippedConvolution(), request, ACCEPT_EVERYTHING));

    EXPECT_LT(composition.realistic, 0.98)
        << "the corpus is entirely anchored and covers nothing unexpected";
    EXPECT_GT(composition.fromExploration, 0);
}

TEST(TestCorpusRealism, EverySourceContributes)
{
    // Any source silently dropping to zero would leave the row count unchanged.
    ExplorationRequest request;
    request.pointsPerCombination = 200;
    request.seed = 13;

    const auto composition
        = compose(exploreProblemSpace(shippedConvolution(), request, ACCEPT_EVERYTHING));

    EXPECT_GT(composition.fromArchetypes, 0);
    EXPECT_GT(composition.fromNeighbourhood, 0);
    EXPECT_GT(composition.fromExploration, 0);
}

TEST(TestCorpusRealism, TheAnchoredShapesAreTheRealisticOnes)
{
    // Ensures the realism comes from the archetype and neighbourhood draws, not luck.
    const auto metadata = shippedConvolution();
    ExplorationRequest request;
    request.pointsPerCombination = 200;
    request.seed = 14;

    const auto corpus = exploreProblemSpace(metadata, request, ACCEPT_EVERYTHING);

    for(const auto& combination : corpus.combinations)
    {
        const auto anchored
            = static_cast<size_t>(combination.fromArchetypes + combination.fromNeighbourhood);
        size_t realistic = 0;
        for(size_t i = 0; i < anchored && i < combination.problems.size(); ++i)
        {
            realistic += looksLikeALayer(combination.problems[i]) ? 1 : 0;
        }
        if(anchored == 0)
        {
            continue;
        }
        EXPECT_GT(static_cast<double>(realistic) / static_cast<double>(anchored), 0.9)
            << detail::describe(combination.categorical) << ": anchored draws are not realistic";
    }
}

TEST(TestCorpusRealism, TheCorpusIsNotJustTheArchetypesRepeated)
{
    // Realism alone could be met by repeating a few archetypes.
    ExplorationRequest request;
    request.pointsPerCombination = 300;
    request.seed = 15;

    const auto corpus = exploreProblemSpace(shippedConvolution(), request, ACCEPT_EVERYTHING);

    std::set<std::string> distinct;
    for(const auto& point : corpus.problems())
    {
        distinct.insert(detail::describe(point));
    }
    EXPECT_EQ(distinct.size(), corpus.problems().size()) << "the corpus repeats itself";
    EXPECT_GT(distinct.size(), 500U);
}

TEST(TestCorpusRealism, TheAnchoredGeometriesCoverStridePaddingAndDilation)
{
    // Measured over anchored draws only: exploration varies these axes freely and would pass
    // without any realistic strided or dilated layer.
    ExplorationRequest request;
    request.pointsPerCombination = 400;
    request.seed = 31;

    const auto corpus = exploreProblemSpace(shippedConvolution(), request, ACCEPT_EVERYTHING);

    std::set<int64_t> strides;
    std::set<int64_t> paddings;
    std::set<int64_t> dilations;
    std::set<int64_t> filters;
    for(const auto& combination : corpus.combinations)
    {
        const auto anchored
            = static_cast<size_t>(combination.fromArchetypes + combination.fromNeighbourhood);
        for(size_t i = 0; i < anchored && i < combination.problems.size(); ++i)
        {
            strides.insert(at(combination.problems[i], "stride_h"));
            paddings.insert(at(combination.problems[i], "pad_h"));
            dilations.insert(at(combination.problems[i], "dilation_h"));
            filters.insert(at(combination.problems[i], "R"));
        }
    }

    const auto covers = [](const std::set<int64_t>& seen, const std::set<int64_t>& wanted) {
        std::set<int64_t> missing;
        for(const auto value : wanted)
        {
            if(seen.count(value) == 0)
            {
                missing.insert(value);
            }
        }
        return missing;
    };

    EXPECT_TRUE(covers(strides, {1, 2, 4, 16}).empty()) << "strides not covered by any archetype";
    EXPECT_TRUE(covers(paddings, {0, 1, 2, 3, 4}).empty()) << "paddings not covered";
    EXPECT_TRUE(covers(dilations, {1, 2, 3, 4}).empty()) << "dilations not covered";
    EXPECT_TRUE(covers(filters, {1, 2, 3, 4, 5, 7, 11, 16}).empty()) << "filter sizes not covered";

    // Not just one draw per value.
    int64_t strided = 0;
    int64_t dilated = 0;
    int64_t padded = 0;
    for(const auto& point : corpus.problems())
    {
        strided += at(point, "stride_h") > 1 ? 1 : 0;
        dilated += at(point, "dilation_h") > 1 ? 1 : 0;
        padded += at(point, "pad_h") > 0 ? 1 : 0;
    }
    const auto total = static_cast<double>(corpus.problems().size());
    GTEST_LOG_(INFO) << "declared space: " << (100.0 * static_cast<double>(strided) / total)
                     << "% strided, " << (100.0 * static_cast<double>(dilated) / total)
                     << "% dilated, " << (100.0 * static_cast<double>(padded) / total)
                     << "% padded, over " << strides.size() << " strides, " << paddings.size()
                     << " paddings, " << dilations.size() << " dilations";
    EXPECT_GT(static_cast<double>(strided) / total, 0.05)
        << "strided convolutions are a rounding error";
    EXPECT_GT(static_cast<double>(dilated) / total, 0.02)
        << "dilated convolutions are a rounding error";
    EXPECT_GT(static_cast<double>(padded) / total, 0.20)
        << "padded convolutions are a rounding error";
}

TEST(TestCorpusRealism, EveryDeclaredArchetypeSetIsActuallyDrawnFrom)
{
    // A declaration whose archetypes never match a combination still parses and validates.
    for(const auto& file :
        std::filesystem::directory_iterator(hipdnn_corpus_gen::test::operationsDir()))
    {
        if(file.path().string().find(".opmeta.json") == std::string::npos)
        {
            continue;
        }
        std::ifstream stream(file.path());
        ASSERT_TRUE(stream.good()) << file.path();
        const auto parsed = parseOperationMetadata(nlohmann::json::parse(stream));
        ASSERT_TRUE(parsed.ok()) << file.path().filename() << ": "
                                 << (parsed.errors.empty() ? "" : parsed.errors.front());
        if(parsed.metadata->archetypes.empty())
        {
            continue;
        }

        ExplorationRequest request;
        request.pointsPerCombination = 60;
        request.seed = 21;

        const auto composition
            = compose(exploreProblemSpace(*parsed.metadata, request, ACCEPT_EVERYTHING));
        EXPECT_GT(composition.fromArchetypes, 0)
            << parsed.metadata->operation << " declares " << parsed.metadata->archetypes.size()
            << " archetypes and drew from none of them";
        EXPECT_GT(composition.fromNeighbourhood, 0)
            << parsed.metadata->operation << " never perturbed an anchor";
    }
}

TEST(TestCorpusRealism, EveryShippedMirrorKeepsItsArchetypesProportion)
{
    // A neighbour of an archetype is near it: a mirrored extent keeps the archetype's own
    // proportion to what it follows (within the declared ratios), so a decode shape stays a
    // decode shape and a D2=1 activation keeps its row length.
    for(const auto& file :
        std::filesystem::directory_iterator(hipdnn_corpus_gen::test::operationsDir()))
    {
        if(file.path().string().find(".opmeta.json") == std::string::npos)
        {
            continue;
        }
        std::ifstream stream(file.path());
        ASSERT_TRUE(stream.good()) << file.path();
        const auto parsed = parseOperationMetadata(nlohmann::json::parse(stream));
        ASSERT_TRUE(parsed.ok()) << file.path().filename();
        const auto& metadata = *parsed.metadata;

        std::mt19937_64 rng(31);
        size_t total = 0;
        const auto combinations = detail::categoricalCombinations(metadata.parameters, 64, total);
        for(const auto& archetype : metadata.archetypes)
        {
            for(const auto& combination : combinations)
            {
                const auto anchor
                    = detail::drawFromArchetype(metadata, archetype, combination, rng);
                if(!anchor.has_value())
                {
                    continue;
                }
                const auto moved = detail::perturbWithinNeighbourhood(metadata, *anchor, rng);
                for(const auto& [follower, hood] : metadata.neighbourhood)
                {
                    if(hood.kind != Neighbourhood::Kind::MIRROR)
                    {
                        continue;
                    }
                    const auto* parameter = metadata.find(follower);
                    const auto anchorFollower = detail::integerAt(*anchor, follower);
                    const auto anchorFollowed = detail::integerAt(*anchor, hood.mirrors);
                    const auto movedFollower = detail::integerAt(moved, follower);
                    const auto movedFollowed = detail::integerAt(moved, hood.mirrors);
                    if(parameter == nullptr || !anchorFollower || !anchorFollowed || !movedFollower
                       || !movedFollowed || *anchorFollowed <= 0 || *movedFollowed <= 0)
                    {
                        continue;
                    }
                    // A value held at its declared bound has lost its proportion by design.
                    if(*movedFollower == detail::clampToRange(*parameter, INT64_MIN)
                       || *movedFollower == detail::clampToRange(*parameter, INT64_MAX))
                    {
                        continue;
                    }
                    const auto proportion = static_cast<double>(*anchorFollower)
                                            / static_cast<double>(*anchorFollowed);
                    const auto [low, high]
                        = hood.ratios.empty()
                              ? std::pair{1.0, 1.0}
                              : std::pair{
                                    *std::min_element(hood.ratios.begin(), hood.ratios.end()),
                                    *std::max_element(hood.ratios.begin(), hood.ratios.end())};
                    const auto actual
                        = static_cast<double>(*movedFollower) / static_cast<double>(*movedFollowed);
                    // Rounding to integers moves a small extent by up to half a unit.
                    const auto slack = 1.0 + 1.0 / static_cast<double>(*movedFollowed);
                    EXPECT_GE(actual * slack, proportion * low)
                        << metadata.operation << " archetype " << archetype.name << ": " << follower
                        << "=" << *movedFollower << " vs " << hood.mirrors << "=" << *movedFollowed
                        << " lost the archetype's " << proportion << ":1";
                    EXPECT_LE(actual, proportion * high * slack)
                        << metadata.operation << " archetype " << archetype.name << ": " << follower
                        << "=" << *movedFollower << " vs " << hood.mirrors << "=" << *movedFollowed;
                }
            }
        }
    }
}

/// The shipped attention declarations: the forward, and the backward whose problem space is
/// the same one without the window and the softmax statistics.
class TestShippedSdpaRealism : public ::testing::TestWithParam<std::string>
{
};

TEST_P(TestShippedSdpaRealism, CorpusCoversHeadDimsLayoutsAndSequenceRelationsAModelUses)
{
    // The corpus must hold real heads (a head_dim drawn log-uniformly from 1 puts 43% of rows at
    // D <= 16, none of them a real head), every declared layout, and cross attention
    // (seqlen_q > seqlen_k).
    const auto metadata = test::shippedOperation(GetParam());

    // The default first pass: 50 points for each live combination of the unweighted parameters
    // (is_causal, alignment, dtype, form and, in the forward, generate_stats), about 8900 rows in
    // the forward and 2700 in the backward, enough that the shares below are not noise.
    ExplorationRequest request;
    request.seed = 9;
    const auto problems = exploreProblemSpace(metadata, request, ACCEPT_EVERYTHING).problems();
    ASSERT_GT(problems.size(), GetParam() == "sdpa_fwd" ? 5000U : 1500U);

    size_t wide = 0;
    size_t square = 0;
    size_t cross = 0;
    int64_t narrowest = INT64_MAX;
    std::set<std::string> layouts;
    for(const auto& point : problems)
    {
        const auto headDim = at(point, "head_dim");
        narrowest = std::min(narrowest, headDim);
        wide += headDim >= 32 ? 1 : 0;
        const auto q = at(point, "seqlen_q");
        const auto k = at(point, "seqlen_k");
        square += q == k ? 1 : 0;
        if(q > k)
        {
            ++cross;
            EXPECT_FALSE(std::get<bool>(point.at("is_causal")))
                << "a causal mask with more queries than keys";
        }
        layouts.insert(std::get<std::string>(point.at("layout")));
    }
    const auto share = [&problems](size_t count) {
        return static_cast<double>(count) / static_cast<double>(problems.size());
    };

    // Measured over seeds 0 to 20 no lower than 0.82 and 0.16 (sdpa_fwd) and 0.78 and 0.27
    // (sdpa_bwd), so one pair of floors holds for both. The wide floor is set well above the 0.57
    // a log-uniform head_dim from 1 would leave at most and below the measured shares. The first
    // pass spreads over about 180 combinations of a few points each, whose anchored quota (the
    // mixture's share of a combination's own count, the archetypes' D of 64 and 128 among it) is
    // small, so the share is lower than the declared mixture alone would give.
    constexpr double SDPA_WIDE_SHARE = 0.7;
    constexpr double SDPA_SQUARE_SHARE = 0.1;
    EXPECT_GE(narrowest, 8) << "the declared head_dim range";
    EXPECT_GT(share(wide), SDPA_WIDE_SHARE) << "most rows should be heads of D >= 32";
    EXPECT_GT(share(square), SDPA_SQUARE_SHARE) << "prefill is the dominant case";
    EXPECT_GT(cross, 0U) << "no cross attention";
    EXPECT_EQ(layouts, (std::set<std::string>{"bhsd", "bshd", "kv_bhsd"}));
}

TEST_P(TestShippedSdpaRealism, ArchetypesAnchorCrossAttentionAndTheHeadWidthsOfRealModels)
{
    // The anchors are what a large corpus is mostly made of, so they must themselves hold the
    // cross-attention and the head widths the model-pinned archetypes (D of 64 and 128) lack.
    const auto metadata = test::shippedOperation(GetParam());

    std::mt19937_64 rng(11);
    std::set<int64_t> headDims;
    size_t cross = 0;
    for(const bool causal : {true, false})
    {
        const auto categorical = test::withDefaultCategoricals(
            metadata, ProblemPoint{{"dtype", std::string("bf16")}, {"is_causal", causal}});
        for(const auto& archetype : metadata.archetypes)
        {
            for(int i = 0; i < 40; ++i)
            {
                const auto drawn = detail::drawFromArchetype(metadata, archetype, categorical, rng);
                if(!drawn.has_value())
                {
                    continue; // the archetype fixes the other mask
                }
                EXPECT_TRUE(detail::satisfiesConstraints(metadata, *drawn))
                    << archetype.name << " draws a point the declaration's own constraints refuse";
                headDims.insert(at(*drawn, "head_dim"));
                cross += at(*drawn, "seqlen_q") > at(*drawn, "seqlen_k") ? 1 : 0;
            }
        }
    }
    EXPECT_GT(cross, 0U) << "no archetype anchors cross attention";
    for(const int64_t width : {40, 48, 80, 96, 112, 160, 192, 256})
    {
        EXPECT_EQ(headDims.count(width), 1U) << "no archetype anchors head_dim " << width;
    }
}

INSTANTIATE_TEST_SUITE_P(SdpaDeclarations,
                         TestShippedSdpaRealism,
                         ::testing::Values("sdpa_fwd", "sdpa_bwd"));

} // namespace hipdnn_corpus_gen

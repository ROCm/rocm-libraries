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

#include <nlohmann/json.hpp>

#include "OperationsDir.hpp"
#include <filesystem>
#include <fstream>
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

} // namespace hipdnn_corpus_gen

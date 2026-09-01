// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestCorpusRealism.cpp
 * @brief Measures what fraction of a generated corpus could come from a real workload.
 *
 * Every other test here asks whether a shape is *valid*. Validity was never the problem: the
 * search that this replaces produced ten thousand valid convolutions, of which four resembled a
 * layer anyone runs. Validity and realism are different properties and only one of them was
 * being checked, so a corpus could pass every test and still train a model on the interior of a
 * region nobody queries.
 *
 * The predicate below is deliberately crude and deliberately independent of the declaration --
 * it is written from what convolutions look like in practice, not from the archetypes, so a
 * declaration that drifts cannot drag the measurement with it.
 */

#include <gtest/gtest.h>

#include <hipdnn_corpus_gen/ProblemSpace.hpp>

#include <nlohmann/json.hpp>

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
/// Four properties, each of which nearly every real layer has and a uniform draw over the
/// feasible region nearly always lacks:
///
///  - the image is square, or one of the handful of photographic aspect ratios;
///  - the channel counts are aligned to eight, or the input is a three-channel image;
///  - the filter is one of the sizes networks actually use;
///  - the padding either preserves the spatial extent or is absent.
///
/// A shape failing these is not invalid and not useless -- it is simply not evidence about the
/// problems anyone will ask. The corpus wants mostly the former and some of the latter.
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

    // Even filters belong here as well as odd: 2x2 at stride 2 is a non-overlapping reduction
    // and 4x4 at stride 4 is the ConvNeXt stem.
    const std::set<int64_t> filters{1, 2, 3, 4, 5, 7, 11, 16};
    const bool plausibleFilter = filters.count(r) == 1 && r == s;

    // Padding is bounded by "same" rather than required to equal it. Networks pad to hold the
    // spatial extent, or not at all, or somewhere between -- AlexNet's 11x11 takes pad 2 where
    // same would be 5. What no network does is pad by more than the filter reaches, which is
    // most of what a uniform draw over the region produces.
    const bool plausiblePadding = pad >= 0 && pad <= (dilation * (r - 1)) / 2;

    const std::set<int64_t> strides{1, 2, 4, 16};

    return plausibleShape && plausibleChannels && plausibleFilter && plausiblePadding
           && strides.count(stride) == 1;
}

OperationMetadata shippedConvolution()
{
    std::ifstream file(HIPDNN_CORPUS_GEN_OPERATIONS_DIR "/conv_fwd.opmeta.json");
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
    composition.realistic
        = composition.total == 0
              ? 0.0
              : static_cast<double>(realistic) / static_cast<double>(composition.total);
    return composition;
}

} // namespace

TEST(TestCorpusRealism, MostOfTheShippedConvolutionCorpusLooksLikeRealLayers)
{
    ExplorationRequest request;
    request.pointsPerCombination = 300;
    request.seed = 11;

    const auto composition = compose(
        exploreProblemSpace(shippedConvolution(), request, ACCEPT_EVERYTHING));

    ASSERT_GT(composition.total, 0U);
    // Reported, not just asserted: the figure is the point of the change, and a threshold that
    // passes says nothing about whether it passed at 51% or 95%.
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
    // The failure in the other direction, and the easier one to ship by accident: a corpus made
    // only of anchors teaches a model the anchors. An engine is asked about problems nobody
    // wrote down, so §5.4 keeps an exploration share and this is what checks it survived.
    ExplorationRequest request;
    request.pointsPerCombination = 300;
    request.seed = 12;

    const auto composition = compose(
        exploreProblemSpace(shippedConvolution(), request, ACCEPT_EVERYTHING));

    EXPECT_LT(composition.realistic, 0.98)
        << "the corpus is entirely anchored and covers nothing unexpected";
    EXPECT_GT(composition.fromExploration, 0);
}

TEST(TestCorpusRealism, EverySourceContributes)
{
    // Each of the three can fail silently: archetypes that never match a combination, a
    // neighbourhood with nowhere to stand, an exploration whose budget was already spent. Any
    // one of them going to zero changes what the corpus is while the row count looks right.
    ExplorationRequest request;
    request.pointsPerCombination = 200;
    request.seed = 13;

    const auto composition = compose(
        exploreProblemSpace(shippedConvolution(), request, ACCEPT_EVERYTHING));

    EXPECT_GT(composition.fromArchetypes, 0);
    EXPECT_GT(composition.fromNeighbourhood, 0);
    EXPECT_GT(composition.fromExploration, 0);
}

TEST(TestCorpusRealism, TheAnchoredShapesAreTheRealisticOnes)
{
    // Guards against passing the headline number for the wrong reason -- a lucky exploration,
    // or a realism predicate loose enough that anything satisfies it. If the archetype and
    // neighbourhood draws are not themselves realistic, the mechanism is not what is working.
    const auto metadata = shippedConvolution();
    ExplorationRequest request;
    request.pointsPerCombination = 200;
    request.seed = 14;

    const auto corpus = exploreProblemSpace(metadata, request, ACCEPT_EVERYTHING);

    for(const auto& combination : corpus.combinations)
    {
        const auto anchored = static_cast<size_t>(combination.fromArchetypes
                                                  + combination.fromNeighbourhood);
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
    // Nine archetypes and a broken perturbation would still fill the quota, score perfectly on
    // realism, and contain almost no distinct shapes. Coverage and realism have to be measured
    // together or each can be bought with the other.
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
    // The gap this closes: with padding and dilation pinned per archetype and neither in the
    // neighbourhood, the anchored 80% of the corpus only ever saw whatever geometries were
    // written down. A first pass declared four, so a generated corpus was stride-1 pad-0
    // dilation-1 almost everywhere and nothing said so -- the number of shapes looked right.
    //
    // Asserted over the anchored draws specifically. The exploration share varies these axes
    // freely, so measuring the whole corpus would pass on exploration alone and say nothing
    // about whether any *realistic* strided or dilated layer was ever proposed.
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
        const auto anchored = static_cast<size_t>(combination.fromArchetypes
                                                  + combination.fromNeighbourhood);
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

    // And the coverage must not be one lonely draw per value, which would satisfy the sets
    // above while contributing nothing a model could learn from.
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
    EXPECT_GT(static_cast<double>(strided) / total, 0.05) << "strided convolutions are a rounding error";
    EXPECT_GT(static_cast<double>(dilated) / total, 0.02) << "dilated convolutions are a rounding error";
    EXPECT_GT(static_cast<double>(padded) / total, 0.20) << "padded convolutions are a rounding error";
}

TEST(TestCorpusRealism, EveryDeclaredArchetypeSetIsActuallyDrawnFrom)
{
    // Operation-agnostic, and the check the conv-specific ones cannot make: a declaration whose
    // archetypes never match a combination parses, validates, and silently contributes nothing.
    // The whole file then reads as workload knowledge that is present and inert.
    for(const auto& file :
        std::filesystem::directory_iterator(HIPDNN_CORPUS_GEN_OPERATIONS_DIR))
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

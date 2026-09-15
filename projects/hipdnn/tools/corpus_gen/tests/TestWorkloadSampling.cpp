// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestWorkloadSampling.cpp
 * @brief Covers drawing from archetypes and moving within neighbourhoods.
 *
 * The property under test throughout is that a draw stays a *plausible problem*: the joint
 * facts an archetype records survive the draw, and a perturbation moves a parameter the way
 * that parameter actually moves in real networks. A sampler that produced valid-but-arbitrary
 * numbers would pass a shape-validity check and still rebuild the distribution this replaces.
 */

#include <gtest/gtest.h>

#include <hipdnn_corpus_gen/WorkloadSampling.hpp>

#include <nlohmann/json.hpp>

#include <set>
#include <string>

namespace hipdnn_corpus_gen
{
namespace
{

OperationMetadata metadataFor(const std::string& json)
{
    auto load = parseOperationMetadata(nlohmann::json::parse(json));
    EXPECT_TRUE(load.ok()) << (load.errors.empty() ? "" : load.errors.front());
    return load.metadata.value_or(OperationMetadata{});
}

/// A convolution cut down to what these tests need: two correlated extents, a mirror, and a
/// categorical the archetypes disagree about.
OperationMetadata tinyConv()
{
    return metadataFor(R"({
      "schema_version": "1.1",
      "operation": "tiny_conv",
      "parameters": {
        "C":     { "type": "int64" },
        "H":     { "type": "int64" },
        "W":     { "type": "int64" },
        "R":     { "type": "int64" },
        "pad":   { "type": "int64", "range": [0, null] },
        "dtype": { "type": "enum", "values": ["float32", "float16"] }
      },
      "stratification_axis": "working_set",
      "regimes": {},
      "graph_builder": { "function": "b", "source": "x.hpp", "arguments": [] },
      "archetypes": [
        { "name": "resnet_stem", "source": "RFC 0019.13 §12.2",
          "values": { "C": [3], "H": [224], "W": ["$q.H"], "R": [7], "pad": [3],
                      "dtype": ["float32"] } },
        { "name": "half_only",
          "values": { "C": [64], "H": [56], "W": ["$q.H"], "R": [3], "pad": [1],
                      "dtype": ["float16"] } }
      ],
      "neighbourhood": {
        "C": { "kind": "multiple", "of": 8, "steps": [-2, -1, 0, 1, 2] },
        "H": { "kind": "scale", "factors": [0.5, 1, 2] },
        "W": { "kind": "mirror", "of": "H", "ratios": [1, 2] },
        "R": { "kind": "values", "values": [1, 3, 5, 7] }
      },
      "mixture": { "archetypes": 0.2, "neighbourhood": 0.6, "exploration": 0.2 }
    })");
}

int64_t at(const ProblemPoint& point, const std::string& name)
{
    return std::get<int64_t>(point.at(name));
}

} // namespace

TEST(TestWorkloadSampling, ADrawKeepsAnArchetypesValuesTogether)
{
    // The whole reason archetypes exist: C=3 is realistic beside H=224 and R=7, and meaningless
    // beside C=64's shape. Drawing each parameter from its own marginal loses exactly that, and
    // is what a uniform search over the region already does.
    const auto metadata = tinyConv();
    std::mt19937_64 rng(1);

    for(int i = 0; i < 50; ++i)
    {
        const auto drawn = detail::drawFromArchetype(
            metadata, metadata.archetypes.front(), ProblemPoint{}, rng);
        ASSERT_TRUE(drawn.has_value());
        EXPECT_EQ(at(*drawn, "C"), 3);
        EXPECT_EQ(at(*drawn, "H"), 224);
        EXPECT_EQ(at(*drawn, "R"), 7);
        EXPECT_EQ(at(*drawn, "pad"), 3);
    }
}

TEST(TestWorkloadSampling, AReferencedValueFollowsWhatWasActuallyDrawn)
{
    // `W: ["$q.H"]` is how a declaration says "square". Re-drawing W independently would make
    // 224x224 into 224x112 and the archetype would no longer describe the layer it names.
    const auto metadata = tinyConv();
    std::mt19937_64 rng(2);

    const auto drawn
        = detail::drawFromArchetype(metadata, metadata.archetypes.front(), ProblemPoint{}, rng);
    ASSERT_TRUE(drawn.has_value());
    EXPECT_EQ(at(*drawn, "W"), at(*drawn, "H"));
}

TEST(TestWorkloadSampling, AnArchetypeThatContradictsTheCombinationDeclinesRatherThanOverrides)
{
    // Combinations own the categorical axes, and each gets its own budget. An archetype that
    // quietly overrode dtype would file float32 problems under float16 -- every row mislabeled
    // in the one column a model cannot recover from the shape.
    const auto metadata = tinyConv();
    std::mt19937_64 rng(3);

    const ProblemPoint half{{"dtype", std::string{"float16"}}};
    EXPECT_FALSE(
        detail::drawFromArchetype(metadata, metadata.archetypes.front(), half, rng).has_value());

    const auto matching = detail::drawFromArchetype(metadata, metadata.archetypes.back(), half, rng);
    ASSERT_TRUE(matching.has_value());
    EXPECT_EQ(std::get<std::string>(matching->at("dtype")), "float16");
    EXPECT_EQ(at(*matching, "C"), 64);
}

TEST(TestWorkloadSampling, PerturbationKeepsChannelsAligned)
{
    // The measured failure: 1.3% of a uniform corpus had C and K both aligned to eight, while
    // essentially every convolution in a real network does. Alignment decides which kernels are
    // even applicable, so a corpus that loses it is asking the model the wrong question.
    const auto metadata = tinyConv();
    std::mt19937_64 rng(4);

    const ProblemPoint anchor{{"C", int64_t{64}}, {"H", int64_t{56}}, {"W", int64_t{56}},
                              {"R", int64_t{3}},  {"pad", int64_t{1}},
                              {"dtype", std::string{"float32"}}};

    for(int i = 0; i < 200; ++i)
    {
        const auto moved = detail::perturbWithinNeighbourhood(metadata, anchor, rng);
        EXPECT_EQ(at(moved, "C") % 8, 0) << "C drifted off its declared alignment";
        EXPECT_GE(at(moved, "C"), 8);
    }
}

TEST(TestWorkloadSampling, ADistinguishedSmallValueSurvivesAnAlignmentNeighbourhood)
{
    // C=3 is the three-channel image every vision network starts from. It is not a misaligned
    // 8, and rounding it up to one deletes the stem layer from the 60% of the corpus that comes
    // from perturbation -- quietly, because C=8 is a perfectly ordinary channel count.
    const auto metadata = tinyConv();
    std::mt19937_64 rng(41);

    const ProblemPoint stem{{"C", int64_t{3}},  {"H", int64_t{224}}, {"W", int64_t{224}},
                            {"R", int64_t{7}},  {"pad", int64_t{3}},
                            {"dtype", std::string{"float32"}}};

    for(int i = 0; i < 200; ++i)
    {
        const auto moved = detail::perturbWithinNeighbourhood(metadata, stem, rng);
        EXPECT_EQ(at(moved, "C"), 3) << "the three-channel input did not survive perturbation";
    }

    // A value at or above the alignment still moves, and still lands on a multiple.
    const ProblemPoint body{{"C", int64_t{64}}, {"H", int64_t{56}}, {"W", int64_t{56}},
                            {"R", int64_t{3}},  {"pad", int64_t{1}},
                            {"dtype", std::string{"float32"}}};
    std::set<int64_t> seen;
    for(int i = 0; i < 200; ++i)
    {
        const auto moved = detail::perturbWithinNeighbourhood(metadata, body, rng);
        EXPECT_EQ(at(moved, "C") % 8, 0);
        seen.insert(at(moved, "C"));
    }
    EXPECT_GT(seen.size(), 1U) << "aligned channels stopped moving";
}

TEST(TestWorkloadSampling, PerturbationActuallyMoves)
{
    // A neighbourhood that returned the anchor every time would look like it was working and
    // reduce the corpus to its handful of archetypes, which memorises rather than generalises.
    const auto metadata = tinyConv();
    std::mt19937_64 rng(5);

    const ProblemPoint anchor{{"C", int64_t{64}}, {"H", int64_t{56}}, {"W", int64_t{56}},
                              {"R", int64_t{3}},  {"pad", int64_t{1}},
                              {"dtype", std::string{"float32"}}};

    std::set<int64_t> channels;
    std::set<int64_t> heights;
    std::set<int64_t> filters;
    for(int i = 0; i < 200; ++i)
    {
        const auto moved = detail::perturbWithinNeighbourhood(metadata, anchor, rng);
        channels.insert(at(moved, "C"));
        heights.insert(at(moved, "H"));
        filters.insert(at(moved, "R"));
    }
    EXPECT_GT(channels.size(), 1U);
    EXPECT_GT(heights.size(), 1U);
    EXPECT_GT(filters.size(), 1U);
}

TEST(TestWorkloadSampling, AMirrorFollowsThePerturbedValueNotTheOriginal)
{
    // If W mirrored the anchor's H rather than the drawn one, halving H would silently produce
    // a 28x56 image -- a shape that is valid, unremarkable in a CSV, and in no real network.
    const auto metadata = tinyConv();
    std::mt19937_64 rng(6);

    const ProblemPoint anchor{{"C", int64_t{64}}, {"H", int64_t{56}}, {"W", int64_t{56}},
                              {"R", int64_t{3}},  {"pad", int64_t{1}},
                              {"dtype", std::string{"float32"}}};

    for(int i = 0; i < 100; ++i)
    {
        const auto moved = detail::perturbWithinNeighbourhood(metadata, anchor, rng);
        const auto ratio
            = static_cast<double>(at(moved, "W")) / static_cast<double>(at(moved, "H"));
        EXPECT_TRUE(ratio == 1.0 || ratio == 2.0)
            << "W=" << at(moved, "W") << " does not follow H=" << at(moved, "H");
    }
}

TEST(TestWorkloadSampling, AParameterWithNoNeighbourhoodDoesNotDrift)
{
    // Padding is chosen with the filter, not independently: a 7x7 filter comes with pad 3
    // because that is what preserves the spatial extent. Moving it because it happens to be an
    // integer would turn a recorded layer into a geometry nobody selected.
    const auto metadata = tinyConv();
    std::mt19937_64 rng(7);

    const ProblemPoint anchor{{"C", int64_t{64}}, {"H", int64_t{56}}, {"W", int64_t{56}},
                              {"R", int64_t{3}},  {"pad", int64_t{1}},
                              {"dtype", std::string{"float32"}}};

    for(int i = 0; i < 100; ++i)
    {
        const auto moved = detail::perturbWithinNeighbourhood(metadata, anchor, rng);
        EXPECT_EQ(at(moved, "pad"), 1);
        EXPECT_EQ(std::get<std::string>(moved.at("dtype")), "float32");
    }
}

TEST(TestWorkloadSampling, PerturbationRespectsADeclaredRange)
{
    const auto metadata = metadataFor(R"({
      "schema_version": "1.1",
      "operation": "bounded",
      "parameters": { "heads": { "type": "int64", "range": [1, 96] } },
      "stratification_axis": "working_set",
      "regimes": {},
      "graph_builder": { "function": "b", "source": "x.hpp", "arguments": [] },
      "archetypes": [ { "name": "big", "values": { "heads": [96] } } ],
      "neighbourhood": { "heads": { "kind": "scale", "factors": [0.5, 1, 2, 8] } },
      "mixture": { "archetypes": 0.2, "neighbourhood": 0.6, "exploration": 0.2 }
    })");
    std::mt19937_64 rng(8);

    const ProblemPoint anchor{{"heads", int64_t{96}}};
    for(int i = 0; i < 100; ++i)
    {
        const auto moved = detail::perturbWithinNeighbourhood(metadata, anchor, rng);
        EXPECT_GE(at(moved, "heads"), 1);
        EXPECT_LE(at(moved, "heads"), 96) << "a scale factor escaped the declared range";
    }
}

TEST(TestWorkloadSampling, DeclaringAnchorsWithoutAMixtureStillUsesThem)
{
    // Silence here used to mean the archetypes were parsed and then ignored, which reads in
    // every report as a corpus that had anchors.
    const auto metadata = metadataFor(R"({
      "schema_version": "1.1",
      "operation": "defaulted",
      "parameters": { "M": { "type": "int64" } },
      "stratification_axis": "working_set",
      "regimes": {},
      "graph_builder": { "function": "b", "source": "x.hpp", "arguments": [] },
      "archetypes": [ { "name": "one", "values": { "M": [128] } } ],
      "neighbourhood": { "M": { "kind": "scale", "factors": [0.5, 1, 2] } }
    })");

    EXPECT_FALSE(metadata.mixture.isExplorationOnly());
    EXPECT_GT(metadata.mixture.archetypes, 0.0);
    EXPECT_GT(metadata.mixture.neighbourhood, 0.0);
    EXPECT_GT(metadata.mixture.exploration, 0.0) << "§5.4 keeps an exploration floor";
}

TEST(TestWorkloadSampling, AnOperationWithNoAnchorsIsExplorationOnly)
{
    const auto metadata = metadataFor(R"({
      "schema_version": "1.0",
      "operation": "plain",
      "parameters": { "M": { "type": "int64" } },
      "stratification_axis": "working_set",
      "regimes": {},
      "graph_builder": { "function": "b", "source": "x.hpp", "arguments": [] }
    })");

    EXPECT_TRUE(metadata.mixture.isExplorationOnly());
    EXPECT_DOUBLE_EQ(metadata.mixture.exploration, 1.0);
}

TEST(TestWorkloadSampling, AMalformedDeclarationIsRefusedRatherThanQuietlyWeakened)
{
    // Each of these would otherwise degrade to "no perturbation" or "unanchored", which is
    // indistinguishable in the output from a corpus that was anchored and simply looks odd.
    const auto expectError = [](const std::string& json, const std::string& fragment) {
        const auto load = parseOperationMetadata(nlohmann::json::parse(json));
        EXPECT_FALSE(load.ok());
        bool found = false;
        for(const auto& error : load.errors)
        {
            found = found || error.find(fragment) != std::string::npos;
        }
        EXPECT_TRUE(found) << "expected an error mentioning '" << fragment << "'";
    };

    const std::string head = R"({
      "schema_version": "1.1", "operation": "bad",
      "parameters": { "M": { "type": "int64" } },
      "stratification_axis": "working_set", "regimes": {},
      "graph_builder": { "function": "b", "source": "x.hpp", "arguments": [] },)";

    expectError(head + R"( "archetypes": [ { "name": "a", "values": { "Q": [1] } } ] })",
                "undeclared parameter 'Q'");
    expectError(head + R"( "archetypes": [ { "name": "a", "values": { "M": ["$q.Z"] } } ] })",
                "undeclared parameter 'Z'");
    expectError(head + R"( "neighbourhood": { "M": { "kind": "wobble" } } })",
                "unknown neighbourhood kind");
    expectError(head + R"( "neighbourhood": { "M": { "kind": "scale" } } })", "no factors");
    expectError(head + R"( "neighbourhood": { "M": { "kind": "multiple", "of": 8 } } })",
                "no steps");
    expectError(head + R"( "archetypes": [ { "name": "a", "values": { "M": [1] } } ],
                            "mixture": { "archetypes": 0.5, "neighbourhood": 0.6,
                                         "exploration": 0.2 } })",
                "not 1.0");
    expectError(head + R"( "mixture": { "archetypes": 0.5, "neighbourhood": 0.0,
                                        "exploration": 0.5 } })",
                "asks for archetype samples but none are declared");
}

} // namespace hipdnn_corpus_gen

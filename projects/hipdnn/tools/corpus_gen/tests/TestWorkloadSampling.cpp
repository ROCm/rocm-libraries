// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestWorkloadSampling.cpp
 * @brief Covers drawing from archetypes and moving within neighbourhoods.
 *
 * Draws must stay plausible: an archetype's joint values survive, and perturbation moves each
 * parameter the way it moves in real networks.
 */

#include <gtest/gtest.h>

#include <hipdnn_corpus_gen/WorkloadSampling.hpp>

#include <nlohmann/json.hpp>

#include "OperationsDir.hpp"
#include <fstream>
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

/// A cut-down convolution: two correlated extents, a mirror, and a categorical the archetypes
/// disagree about.
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
        "dtype": { "type": "enum", "values": ["fp32", "fp16"] }
      },
      "stratification_axis": "working_set",
      "regimes": {},
      "graph_builder": { "function": "b", "source": "x.hpp", "arguments": [] },
      "archetypes": [
        { "name": "resnet_stem", "source": "RFC 0019.13 §12.2",
          "values": { "C": [3], "H": [224], "W": ["$q.H"], "R": [7], "pad": [3],
                      "dtype": ["fp32"] } },
        { "name": "half_only",
          "values": { "C": [64], "H": [56], "W": ["$q.H"], "R": [3], "pad": [1],
                      "dtype": ["fp16"] } }
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

/// A shipped declaration, read as the generator reads it.
OperationMetadata shipped(const std::string& operation)
{
    const auto path = hipdnn_corpus_gen::test::operationsDir() + "/" + operation + ".opmeta.json";
    std::ifstream file(path);
    EXPECT_TRUE(file.is_open()) << path;
    const auto load = parseOperationMetadata(nlohmann::json::parse(file));
    EXPECT_TRUE(load.ok()) << (load.errors.empty() ? "" : load.errors.front());
    return load.metadata.value_or(OperationMetadata{});
}

} // namespace

TEST(TestWorkloadSampling, ADrawKeepsAnArchetypesValuesTogether)
{
    // C=3 belongs with H=224 and R=7; per-parameter marginals would lose that.
    const auto metadata = tinyConv();
    std::mt19937_64 rng(1);

    for(int i = 0; i < 50; ++i)
    {
        const auto drawn
            = detail::drawFromArchetype(metadata, metadata.archetypes.front(), ProblemPoint{}, rng);
        ASSERT_TRUE(drawn.has_value());
        EXPECT_EQ(at(*drawn, "C"), 3);
        EXPECT_EQ(at(*drawn, "H"), 224);
        EXPECT_EQ(at(*drawn, "R"), 7);
        EXPECT_EQ(at(*drawn, "pad"), 3);
    }
}

TEST(TestWorkloadSampling, AReferencedValueFollowsWhatWasActuallyDrawn)
{
    // `W: ["$q.H"]` means square: W must follow the drawn H.
    const auto metadata = tinyConv();
    std::mt19937_64 rng(2);

    const auto drawn
        = detail::drawFromArchetype(metadata, metadata.archetypes.front(), ProblemPoint{}, rng);
    ASSERT_TRUE(drawn.has_value());
    EXPECT_EQ(at(*drawn, "W"), at(*drawn, "H"));
}

TEST(TestWorkloadSampling, EveryShippedSdpaArchetypeDrawsWhateverOrderItsKeysParseIn)
{
    // JSON keys parse sorted, so `seqlen_k` precedes the `seqlen_q` it copies; draw order must
    // follow references, not key order.
    for(const std::string operation : {"sdpa_fwd", "sdpa_bwd"})
    {
        const auto metadata = shipped(operation);
        ASSERT_FALSE(metadata.archetypes.empty()) << operation;
        ProblemPoint categorical{{"alignment", std::string{"top_left"}},
                                 {"dtype", std::string{"bf16"}}};
        if(metadata.find("generate_stats") != nullptr)
        {
            categorical["generate_stats"] = false;
        }

        std::mt19937_64 rng(4);
        for(const auto& archetype : metadata.archetypes)
        {
            for(int i = 0; i < 20; ++i)
            {
                const auto drawn = detail::drawFromArchetype(metadata, archetype, categorical, rng);
                ASSERT_TRUE(drawn.has_value()) << operation << " " << archetype.name;
                if(archetype.values.at("seqlen_k").front() == "$q.seqlen_q")
                {
                    EXPECT_EQ(at(*drawn, "seqlen_k"), at(*drawn, "seqlen_q"))
                        << operation << " " << archetype.name;
                }
            }
        }
    }
}

TEST(TestWorkloadSampling, AShippedMirrorFollowsThePerturbedValueWhateverOrderItsKeysParseIn)
{
    // Same key-order hazard for neighbourhood mirrors.
    const auto metadata = shipped("sdpa_fwd");
    std::mt19937_64 rng(5);
    const ProblemPoint anchor{{"batch", int64_t{4}},
                              {"heads", int64_t{32}},
                              {"heads_kv", int64_t{8}},
                              {"seqlen_q", int64_t{2048}},
                              {"seqlen_k", int64_t{2048}},
                              {"head_dim", int64_t{128}},
                              {"is_causal", true},
                              {"alignment", std::string{"top_left"}},
                              {"dtype", std::string{"bf16"}},
                              {"generate_stats", false}};

    const std::set<int64_t> ratios{1, 2, 4, 16};
    for(int i = 0; i < 100; ++i)
    {
        const auto moved = detail::perturbWithinNeighbourhood(metadata, anchor, rng);
        const auto q = at(moved, "seqlen_q");
        const auto k = at(moved, "seqlen_k");
        EXPECT_TRUE(k % q == 0 && ratios.count(k / q) == 1)
            << "seqlen_k=" << k << " does not follow seqlen_q=" << q;
    }
}

TEST(TestWorkloadSampling, AReferenceCycleIsRefusedAtLoad)
{
    // A cycle cannot be drawn in any order.
    const auto refusesCycle = [](const std::string& json) {
        const auto load = parseOperationMetadata(nlohmann::json::parse(json));
        EXPECT_FALSE(load.ok());
        bool found = false;
        for(const auto& error : load.errors)
        {
            found = found || error.find("cycle") != std::string::npos;
        }
        EXPECT_TRUE(found) << "expected an error naming the cycle";
    };

    const std::string head = R"({
      "schema_version": "1.1", "operation": "bad",
      "parameters": { "A": { "type": "int64" }, "B": { "type": "int64" } },
      "stratification_axis": "working_set", "regimes": {},
      "graph_builder": { "function": "b", "source": "x.hpp", "arguments": [] },)";

    refusesCycle(head + R"( "archetypes": [ { "name": "a",
                              "values": { "A": ["$q.B"], "B": ["$q.A"] } } ],
                            "mixture": { "archetypes": 0.5, "exploration": 0.5 } })");
    refusesCycle(head + R"( "neighbourhood": {
                              "A": { "kind": "mirror", "of": "B" },
                              "B": { "kind": "mirror", "of": "A" } } })");
}

TEST(TestWorkloadSampling, AnArchetypeThatContradictsTheCombinationDeclinesRatherThanOverrides)
{
    // Combinations own the categorical axes; overriding dtype would mislabel rows.
    const auto metadata = tinyConv();
    std::mt19937_64 rng(3);

    const ProblemPoint half{{"dtype", std::string{"fp16"}}};
    EXPECT_FALSE(
        detail::drawFromArchetype(metadata, metadata.archetypes.front(), half, rng).has_value());

    const auto matching
        = detail::drawFromArchetype(metadata, metadata.archetypes.back(), half, rng);
    ASSERT_TRUE(matching.has_value());
    EXPECT_EQ(std::get<std::string>(matching->at("dtype")), "fp16");
    EXPECT_EQ(at(*matching, "C"), 64);
}

TEST(TestWorkloadSampling, PerturbationKeepsChannelsAligned)
{
    // Alignment decides which kernels apply; real networks keep C and K aligned to eight.
    const auto metadata = tinyConv();
    std::mt19937_64 rng(4);

    const ProblemPoint anchor{{"C", int64_t{64}},
                              {"H", int64_t{56}},
                              {"W", int64_t{56}},
                              {"R", int64_t{3}},
                              {"pad", int64_t{1}},
                              {"dtype", std::string{"fp32"}}};

    for(int i = 0; i < 200; ++i)
    {
        const auto moved = detail::perturbWithinNeighbourhood(metadata, anchor, rng);
        EXPECT_EQ(at(moved, "C") % 8, 0) << "C drifted off its declared alignment";
        EXPECT_GE(at(moved, "C"), 8);
    }
}

TEST(TestWorkloadSampling, ADistinguishedSmallValueSurvivesAnAlignmentNeighbourhood)
{
    // C=3 is the image stem, not a misaligned 8.
    const auto metadata = tinyConv();
    std::mt19937_64 rng(41);

    const ProblemPoint stem{{"C", int64_t{3}},
                            {"H", int64_t{224}},
                            {"W", int64_t{224}},
                            {"R", int64_t{7}},
                            {"pad", int64_t{3}},
                            {"dtype", std::string{"fp32"}}};

    for(int i = 0; i < 200; ++i)
    {
        const auto moved = detail::perturbWithinNeighbourhood(metadata, stem, rng);
        EXPECT_EQ(at(moved, "C"), 3) << "the three-channel input did not survive perturbation";
    }

    const ProblemPoint body{{"C", int64_t{64}},
                            {"H", int64_t{56}},
                            {"W", int64_t{56}},
                            {"R", int64_t{3}},
                            {"pad", int64_t{1}},
                            {"dtype", std::string{"fp32"}}};
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
    // A neighbourhood that never moves would reduce the corpus to its archetypes.
    const auto metadata = tinyConv();
    std::mt19937_64 rng(5);

    const ProblemPoint anchor{{"C", int64_t{64}},
                              {"H", int64_t{56}},
                              {"W", int64_t{56}},
                              {"R", int64_t{3}},
                              {"pad", int64_t{1}},
                              {"dtype", std::string{"fp32"}}};

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
    // W must mirror the perturbed H, not the anchor's.
    const auto metadata = tinyConv();
    std::mt19937_64 rng(6);

    const ProblemPoint anchor{{"C", int64_t{64}},
                              {"H", int64_t{56}},
                              {"W", int64_t{56}},
                              {"R", int64_t{3}},
                              {"pad", int64_t{1}},
                              {"dtype", std::string{"fp32"}}};

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
    // Padding is chosen with the filter; moving it independently invents unseen geometries.
    const auto metadata = tinyConv();
    std::mt19937_64 rng(7);

    const ProblemPoint anchor{{"C", int64_t{64}},
                              {"H", int64_t{56}},
                              {"W", int64_t{56}},
                              {"R", int64_t{3}},
                              {"pad", int64_t{1}},
                              {"dtype", std::string{"fp32"}}};

    for(int i = 0; i < 100; ++i)
    {
        const auto moved = detail::perturbWithinNeighbourhood(metadata, anchor, rng);
        EXPECT_EQ(at(moved, "pad"), 1);
        EXPECT_EQ(std::get<std::string>(moved.at("dtype")), "fp32");
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
    // Declared archetypes must not be parsed and then ignored.
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
    // Each would otherwise silently degrade to "no perturbation" or "unanchored".
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

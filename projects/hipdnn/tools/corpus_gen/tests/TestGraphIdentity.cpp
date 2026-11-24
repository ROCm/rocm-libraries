// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipdnn_corpus_gen/GraphBuilderRegistry.hpp>
#include <hipdnn_corpus_gen/GraphIdentity.hpp>
#include <hipdnn_corpus_gen/OperationMetadata.hpp>

#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include "OperationsDir.hpp"
#include <fstream>
#include <string>
#include <vector>

/// @file TestGraphIdentity.cpp
/// @brief The id a written graph carries, and the content key that identifies it.
///
/// An id-less graph gets a random id from `GraphDescriptor::finalize` at load, so every
/// written graph is stamped; measurements are keyed by content, never by that id.

using namespace hipdnn_corpus_gen;

namespace
{

namespace sdk = hipdnn_flatbuffers_sdk::utilities;

/// One real graph built through the shipped declaration, as the tool builds them.
std::vector<uint8_t> someGraph()
{
    const std::string path = hipdnn_corpus_gen::test::operationsDir() + "/sdpa_fwd.opmeta.json";
    std::ifstream file(path);
    EXPECT_TRUE(file.is_open()) << path;
    const auto parsed = parseOperationMetadata(nlohmann::json::parse(file));
    EXPECT_TRUE(parsed.ok()) << (parsed.errors.empty() ? "" : parsed.errors.front());
    if(!parsed.ok())
    {
        return {};
    }

    const ProblemPoint point{{"batch", int64_t{1}},
                             {"heads", int64_t{32}},
                             {"heads_kv", int64_t{32}},
                             {"seqlen_q", int64_t{2048}},
                             {"seqlen_k", int64_t{2048}},
                             {"head_dim", int64_t{128}},
                             {"is_causal", true},
                             {"alignment", std::string("top_left")},
                             {"generate_stats", false},
                             {"dtype", std::string("bf16")}};

    const auto built = buildGraphFor(*parsed.metadata, point);
    EXPECT_TRUE(built.ok()) << built.error;
    return built.bytes;
}

} // namespace

TEST(TestGraphIdentity, AGraphIdentityIsAFunctionOfTheBytesAndNothingElse)
{
    const std::string left = "some graph bytes";
    const std::string right = "some graph byteS";

    const auto identity = [](const std::string& text) {
        return graphIdentity(reinterpret_cast<const uint8_t*>(text.data()), text.size());
    };

    EXPECT_EQ(identity(left), identity(left));
    EXPECT_NE(identity(left), identity(right));
}

TEST(TestGraphIdentity, AGraphIdentityIsShapedLikeAUuidSoItRoundTripsThroughTheGraphDocument)
{
    // Stored in `Graph.id`, so it must parse as a UUID. Version 8: content-derived, neither
    // random (v4) nor name-based (v5).
    const std::string bytes = "some graph bytes";
    const auto identity
        = graphIdentity(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size());

    ASSERT_EQ(identity.size(), 36u);
    EXPECT_EQ(identity[8], '-');
    EXPECT_EQ(identity[13], '-');
    EXPECT_EQ(identity[18], '-');
    EXPECT_EQ(identity[23], '-');
    EXPECT_EQ(identity[14], '8') << identity;
    EXPECT_NE(std::string("89ab").find(identity[19]), std::string::npos) << identity;
    for(size_t i = 0; i < identity.size(); ++i)
    {
        if(i == 8 || i == 13 || i == 18 || i == 23)
        {
            continue;
        }
        EXPECT_NE(std::string("0123456789abcdef").find(identity[i]), std::string::npos) << identity;
    }
}

TEST(TestGraphIdentity, TheStampedGraphCarriesTheNameAndTheIdItReports)
{
    // The returned id must equal the id in the document, so the document is read back.
    const auto stamped = stampGraphIdentity(someGraph(), "sdpa_fwd_prefill_short_batch1");

    const auto* graph = fb::GetGraph(stamped.bytes.data());
    ASSERT_NE(graph, nullptr);
    ASSERT_NE(graph->name(), nullptr);
    EXPECT_EQ(graph->name()->str(), "sdpa_fwd_prefill_short_batch1");
    EXPECT_EQ(stamped.name, "sdpa_fwd_prefill_short_batch1");

    ASSERT_NE(graph->id(), nullptr);
    EXPECT_EQ(sdk::formatUuid(sdk::toUuidBytes(*graph->id())), stamped.id);
}

TEST(TestGraphIdentity, StampingAnAlreadyStampedGraphReproducesTheSameId)
{
    // The digest is taken with the id cleared, so the id is safe to recompute.
    const auto once = stampGraphIdentity(someGraph(), "a_graph");
    const auto twice = stampGraphIdentity(once.bytes, "a_graph");

    EXPECT_EQ(twice.id, once.id);
    EXPECT_EQ(twice.bytes, once.bytes);
}

TEST(TestGraphIdentity, TwoGraphsDifferingOnlyInTheirNameGetDifferentIds)
{
    // The name is digested so problems differing only in a builder-ignored parameter differ.
    const auto left = stampGraphIdentity(someGraph(), "a_graph");
    const auto right = stampGraphIdentity(someGraph(), "another_graph");

    EXPECT_NE(left.id, right.id);
}

TEST(TestGraphIdentity, TheContentKeyIsTheSameGraphUnderAnyNameOrId)
{
    // A graph's id names its file; the key says which graph it is, so a training set keyed on
    // it holds one graph once however many names it was written under.
    const auto left = stampGraphIdentity(someGraph(), "a_graph");
    const auto right = stampGraphIdentity(someGraph(), "another_graph");

    EXPECT_NE(left.id, right.id);
    EXPECT_EQ(left.key, right.key);
    const auto unstamped = someGraph();
    EXPECT_EQ(graphContentKey(unstamped.data(), unstamped.size()), left.key);
    EXPECT_EQ(graphContentKey(left.bytes.data(), left.bytes.size()), left.key);
}

TEST(TestGraphIdentity, DifferentGraphsGetDifferentContentKeys)
{
    auto other = someGraph();
    const auto original = stampGraphIdentity(other, "a_graph");
    // Another shape through the same declaration.
    const std::string path = hipdnn_corpus_gen::test::operationsDir() + "/sdpa_fwd.opmeta.json";
    std::ifstream file(path);
    const auto parsed = parseOperationMetadata(nlohmann::json::parse(file));
    ASSERT_TRUE(parsed.ok());
    const ProblemPoint point{{"batch", int64_t{2}},
                             {"heads", int64_t{32}},
                             {"heads_kv", int64_t{32}},
                             {"seqlen_q", int64_t{2048}},
                             {"seqlen_k", int64_t{2048}},
                             {"head_dim", int64_t{128}},
                             {"is_causal", true},
                             {"alignment", std::string("top_left")},
                             {"generate_stats", false},
                             {"dtype", std::string("bf16")}};
    const auto built = buildGraphFor(*parsed.metadata, point);
    ASSERT_TRUE(built.ok()) << built.error;
    EXPECT_NE(stampGraphIdentity(built.bytes, "a_graph").key, original.key);
}

TEST(TestGraphIdentity, BytesThatAreNotAGraphHaveNoContentKey)
{
    const std::string text = "not a graph";
    EXPECT_THROW(graphContentKey(reinterpret_cast<const uint8_t*>(text.data()), text.size()),
                 std::exception);
}

TEST(TestGraphIdentity, AGraphNameSpellsOutTheProblem)
{
    const ProblemPoint point{{"M", int64_t{64}}, {"dtype", std::string("fp16")}};
    EXPECT_EQ(graphNameFor("matmul", "square", point), "matmul_square_M64_dtypefp16");
}

TEST(TestGraphIdentity, ALongGraphNameIsBoundedUniqueAndStable)
{
    // conv_bias_activation's every-parameter name ran to 260 bytes, past the file-name limit.
    ProblemPoint point;
    for(int i = 0; i < 40; ++i)
    {
        point.emplace("parameter_" + std::to_string(i), int64_t{1024});
    }
    auto other = point;
    other["parameter_9"] = int64_t{2048}; // sorts late, inside the truncated tail

    const auto name = graphNameFor("conv_bias_activation", "large", point);
    EXPECT_LE(name.size(), MAX_GRAPH_NAME);
    EXPECT_EQ(name.rfind("conv_bias_activation_large_parameter_", 0), 0U);
    EXPECT_EQ(name, graphNameFor("conv_bias_activation", "large", point));
    EXPECT_NE(name, graphNameFor("conv_bias_activation", "large", other));
}

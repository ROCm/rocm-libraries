// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipdnn_corpus_gen/GraphBuilderRegistry.hpp>
#include <hipdnn_corpus_gen/GraphIdentity.hpp>
#include <hipdnn_corpus_gen/OperationMetadata.hpp>

#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <fstream>
#include <string>
#include <vector>

/// @file TestGraphIdentity.cpp
/// @brief The identity a written graph carries, and the two ways it can be wrong.
///
/// A graph written without an id is not rejected -- `GraphDescriptor::finalize` mints a fresh
/// v4 for it at load -- so the whole corpus is measured under different identities on every run
/// and nothing reports an error. And an id that is not the one the bench reports back joins to
/// nothing, because `uhd_gen/corpus_io.py:31-34` treats `benchmark` and `graph_id` as one
/// identity under two spellings. Both failures are silent, which is why they are asserted here
/// rather than left to an end-to-end run to notice.

using namespace hipdnn_corpus_gen;

namespace
{

namespace sdk = hipdnn_flatbuffers_sdk::utilities;

/// One real graph, built through the shipped declaration the way the tool builds them, so the
/// stamping tests operate on a document the backend would accept rather than on bytes
/// assembled here.
std::vector<uint8_t> someGraph()
{
    const std::string path
        = std::string(HIPDNN_CORPUS_GEN_OPERATIONS_DIR) + "/sdpa_fwd.opmeta.json";
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
    const std::string left  = "some graph bytes";
    const std::string right = "some graph byteS";

    const auto identity = [](const std::string& text) {
        return graphIdentity(reinterpret_cast<const uint8_t*>(text.data()), text.size());
    };

    EXPECT_EQ(identity(left), identity(left));
    EXPECT_NE(identity(left), identity(right));
}

TEST(TestGraphIdentity, AGraphIdentityIsShapedLikeAUuidSoItRoundTripsThroughTheGraphDocument)
{
    // It is written into `Graph.id` and read back by the bench, so it has to parse as one.
    // Version 8 rather than 4 or 5: it is content-derived, and calling it v4 would describe it
    // as random while calling it v5 would claim a rule nothing else here follows.
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
        EXPECT_NE(std::string("0123456789abcdef").find(identity[i]), std::string::npos)
            << identity;
    }
}

TEST(TestGraphIdentity, TheStampedGraphCarriesTheNameAndTheIdItReports)
{
    // The returned id is what the manifest's `benchmark` column records, and the id inside the
    // document is what the bench reports as `graph_id`. If those two ever disagree the manifest
    // is a record of graphs nobody measured, so the document is read back rather than trusted.
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
    // The digest is taken with the id cleared, so restamping does not digest the previous id.
    // Without that, a corpus regenerated from stamped inputs would drift exactly as an id-less
    // one does, and the idempotence is what makes the id safe to recompute anywhere.
    const auto once  = stampGraphIdentity(someGraph(), "a_graph");
    const auto twice = stampGraphIdentity(once.bytes, "a_graph");

    EXPECT_EQ(twice.id, once.id);
    EXPECT_EQ(twice.bytes, once.bytes);
}

TEST(TestGraphIdentity, TwoGraphsDifferingOnlyInTheirNameGetDifferentIds)
{
    // The name is set before the digest for this reason: two problems that differ only in a
    // parameter the builder ignores would otherwise collide, and the name is where that
    // parameter still shows.
    const auto left  = stampGraphIdentity(someGraph(), "a_graph");
    const auto right = stampGraphIdentity(someGraph(), "another_graph");

    EXPECT_NE(left.id, right.id);
}

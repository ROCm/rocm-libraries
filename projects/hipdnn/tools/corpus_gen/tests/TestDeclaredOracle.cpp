// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipdnn_corpus_gen/DeclaredOracle.hpp>
#include <hipdnn_corpus_gen/GraphSize.hpp>
#include <hipdnn_corpus_gen/OperationMetadata.hpp>

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <fstream>
#include <string>

/// @file TestDeclaredOracle.cpp
/// @brief The admission an engineless corpus run rests on.
///
/// A deterministic engine still has to be measured, and measuring it starts with a corpus built
/// before any handle exists. What decides that corpus is here: the declaration builds the point
/// or it does not, and the graph fits the benchmarking ceiling or it cannot be timed. The one
/// distinction worth a test is between those two refusals -- a broken declaration is counted and
/// named, an oversized graph is neither, because reporting them together describes an engine
/// that serves nothing.

using namespace hipdnn_corpus_gen;

namespace
{

OperationMetadata shippedSdpa()
{
    const std::string path
        = std::string(HIPDNN_CORPUS_GEN_OPERATIONS_DIR) + "/sdpa_fwd.opmeta.json";
    std::ifstream file(path);
    EXPECT_TRUE(file.is_open()) << path;
    const auto parsed = parseOperationMetadata(nlohmann::json::parse(file));
    EXPECT_TRUE(parsed.ok()) << (parsed.errors.empty() ? "" : parsed.errors.front());
    return parsed.ok() ? *parsed.metadata : OperationMetadata{};
}

ProblemPoint sdpaPoint()
{
    return ProblemPoint{{"batch", int64_t{1}},
                        {"heads", int64_t{32}},
                        {"heads_kv", int64_t{32}},
                        {"seqlen_q", int64_t{2048}},
                        {"seqlen_k", int64_t{2048}},
                        {"head_dim", int64_t{128}},
                        {"is_causal", true},
                        {"alignment", std::string("top_left")},
                        {"generate_stats", false},
                        {"dtype", std::string("bf16")}};
}

} // namespace

TEST(TestDeclaredOracle, APointTheDeclarationCanBuildIsAdmittedWithNoDevice)
{
    const auto metadata = shippedSdpa();

    int64_t failures = 0;
    std::string error;
    const auto oracle = makeDeclaredOracle(metadata, &failures, &error);

    EXPECT_TRUE(oracle(sdpaPoint()));
    EXPECT_EQ(failures, 0);
    EXPECT_TRUE(error.empty()) << error;
}

TEST(TestDeclaredOracle, AGraphTooLargeToBenchmarkIsRefusedButNotCounted)
{
    // Neither the declaration's fault nor an engine's. Counting it here would drown the count
    // that means something, so the refusal is silent and only the admission changes.
    const auto metadata = shippedSdpa();

    const auto built = buildAdmissible(metadata, sdpaPoint());
    ASSERT_TRUE(built.has_value());
    const auto footprint = graphBytes(*built);
    ASSERT_GT(footprint, 0);

    int64_t failures = 0;
    std::string error;
    const auto oracle = makeDeclaredOracle(metadata, &failures, &error, footprint - 1);

    EXPECT_FALSE(oracle(sdpaPoint()));
    EXPECT_EQ(failures, 0);
    EXPECT_TRUE(error.empty()) << error;

    // The same point at a ceiling it fits is admitted, so the refusal was the ceiling and not
    // something else about the point.
    EXPECT_TRUE(makeDeclaredOracle(metadata, nullptr, nullptr, footprint)(sdpaPoint()));
}

TEST(TestDeclaredOracle, ADeclarationThatCannotBuildIsCountedAndNamed)
{
    // Broken for every point, so it would otherwise read as an engine that serves almost
    // nothing -- and the search would report that tiny region in good faith.
    const auto metadata = shippedSdpa();

    int64_t failures = 0;
    std::string error;
    const auto oracle = makeDeclaredOracle(metadata, &failures, &error);

    EXPECT_FALSE(oracle(ProblemPoint{}));
    EXPECT_EQ(failures, 1);
    EXPECT_FALSE(error.empty());
}

TEST(TestDeclaredOracle, TheFirstErrorIsKeptRatherThanTheLast)
{
    // One message is worth more than a count, and the first one is the one whose cause is still
    // the cause -- later failures are usually the same defect seen again.
    const auto metadata = shippedSdpa();

    int64_t failures = 0;
    std::string error;
    const auto oracle = makeDeclaredOracle(metadata, &failures, &error);

    EXPECT_FALSE(oracle(ProblemPoint{}));
    const auto first = error;
    EXPECT_FALSE(oracle(ProblemPoint{{"batch", int64_t{1}}}));

    EXPECT_EQ(error, first);
    EXPECT_EQ(failures, 2);
}

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
/// @brief The declared oracle used for engineless corpus runs.
///
/// A broken declaration is counted and named; an oversized graph is refused silently.

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
    // Not a declaration or engine fault, so it is not counted as a build failure.
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

    // Proves the refusal was the ceiling.
    EXPECT_TRUE(makeDeclaredOracle(metadata, nullptr, nullptr, footprint)(sdpaPoint()));
}

TEST(TestDeclaredOracle, ADeclarationThatCannotBuildIsCountedAndNamed)
{
    // Otherwise it would read as an engine that serves almost nothing.
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
    // Later failures are usually the same defect repeated.
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

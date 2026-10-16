// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestShippedGraphsValidate.cpp
 * @brief Every shipped declaration builds graphs the frontend reads back and validates.
 *
 * The registry tests read the buffer directly; this reads it as a run does, through
 * Graph::deserialize, and asks Graph::validate whether the graph is one hipDNN accepts.
 */

#include <gtest/gtest.h>

#include <hipdnn_corpus_gen/GraphBuilderRegistry.hpp>
#include <hipdnn_corpus_gen/ProblemSpace.hpp>
#include <hipdnn_frontend.hpp>

#include <nlohmann/json.hpp>

#include "OperationsDir.hpp"
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace hipdnn_corpus_gen;

TEST(TestShippedGraphsValidate, EveryDeclarationBuildsGraphsTheFrontendValidates)
{
    int64_t operations = 0;
    for(const auto& entry :
        std::filesystem::directory_iterator(hipdnn_corpus_gen::test::operationsDir()))
    {
        if(entry.path().string().find(".opmeta.") == std::string::npos)
        {
            continue;
        }
        std::ifstream file(entry.path());
        ASSERT_TRUE(file.good()) << entry.path();
        const auto parsed = parseOperationMetadata(nlohmann::json::parse(file));
        ASSERT_TRUE(parsed.ok()) << entry.path().filename().string() << ": "
                                 << (parsed.errors.empty() ? "" : parsed.errors.front());
        const auto& metadata = *parsed.metadata;

        // Every dtype and mode stays in: a builder can be right for one element type only.
        ExplorationRequest request;
        request.pointsPerCombination = 2;
        request.budgetPerCombination = 200;
        request.probeBudget = 200;
        request.restarts = 1;
        request.stepsPerStart = 20;
        request.maxSkeleton = 64;
        request.numericCeiling = 128;
        request.seed = 43;
        const auto problems = exploreProblemSpace(metadata, request, [](const ProblemPoint&) {
                                  return true;
                              }).problems();
        EXPECT_FALSE(problems.empty()) << metadata.operation << " explored no problems";

        for(const auto& point : problems)
        {
            const auto built = buildGraphFor(metadata, point);
            ASSERT_TRUE(built.ok())
                << metadata.operation << " " << detail::describe(point) << ": " << built.error;
            hipdnn_frontend::graph::Graph graph;
            const auto restored = graph.deserialize(built.bytes);
            ASSERT_TRUE(restored.is_good()) << metadata.operation << " " << detail::describe(point)
                                            << ": deserialize: " << restored.get_message();
            const auto valid = graph.validate();
            EXPECT_TRUE(valid.is_good()) << metadata.operation << " " << detail::describe(point)
                                         << ": validate: " << valid.get_message();
        }
        ++operations;
    }
    EXPECT_GT(operations, 0) << "no shipped declaration was found";
}

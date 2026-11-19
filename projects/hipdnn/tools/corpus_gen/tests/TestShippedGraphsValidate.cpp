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

namespace
{

/// Each block-scale dequantize node's scale has the dims the CPU reference requires: block
/// sizes apply to the trailing dimensions, and scale[d] = ceil(x[d] / block[d]).
void expectDequantizeScalesMatchTheirBlocks(const std::vector<uint8_t>& bytes,
                                            const std::string& where)
{
    namespace fb = hipdnn_flatbuffers_sdk::data_objects;
    const auto* graph = fb::GetGraph(bytes.data());
    ASSERT_NE(graph, nullptr) << where;
    const auto dimsOf = [graph](int64_t uid) {
        std::vector<int64_t> dims;
        for(const auto* tensor : *graph->tensors())
        {
            if(tensor->uid() == uid && tensor->dims() != nullptr)
            {
                dims.assign(tensor->dims()->begin(), tensor->dims()->end());
            }
        }
        return dims;
    };
    for(const auto* node : *graph->nodes())
    {
        const auto* deq = node->attributes_as_BlockScaleDequantizeAttributes();
        if(deq == nullptr)
        {
            continue;
        }
        const auto x = dimsOf(deq->x_tensor_uid());
        const auto scale = dimsOf(deq->scale_tensor_uid());
        ASSERT_EQ(x.size(), scale.size()) << where;
        std::vector<int64_t> block(x.size(), 1);
        const auto* declared = deq->block_size();
        ASSERT_NE(declared, nullptr) << where;
        ASSERT_LE(declared->size(), x.size()) << where;
        for(size_t i = 0; i < declared->size(); ++i)
        {
            block[x.size() - declared->size() + i] = declared->Get(static_cast<uint32_t>(i));
        }
        for(size_t d = 0; d < x.size(); ++d)
        {
            EXPECT_EQ(scale[d], (x[d] + block[d] - 1) / block[d])
                << where << ": dequantize of tensor " << deq->x_tensor_uid() << ", dim " << d;
        }
    }
}

} // namespace

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
            expectDequantizeScalesMatchTheirBlocks(
                built.bytes, metadata.operation + " " + detail::describe(point));
            const auto valid = graph.validate();
            EXPECT_TRUE(valid.is_good()) << metadata.operation << " " << detail::describe(point)
                                         << ": validate: " << valid.get_message();
        }
        ++operations;
    }
    EXPECT_GT(operations, 0) << "no shipped declaration was found";
}

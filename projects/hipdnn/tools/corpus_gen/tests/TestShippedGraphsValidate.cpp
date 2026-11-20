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

TEST(TestShippedGraphsValidate, EverySdpaFormLayoutScaleWindowAndSequenceRelationValidates)
{
    // The sweep above draws a couple of points per combination; these are the corners the SDPA
    // facets add: cross attention (seqlen_q > seqlen_k, non-causal), the widest head, fp32, every
    // form, layout, scale and window the declaration admits together.
    for(const std::string operation : {"sdpa_fwd", "sdpa_bwd"})
    {
        const auto metadata = test::shippedOperation(operation);
        const auto* form = metadata.find("form");
        ASSERT_NE(form, nullptr) << operation;

        struct Shape
        {
            int64_t seqlenQ;
            int64_t seqlenK;
            int64_t headDim;
            bool causal;
        };
        const std::vector<Shape> shapes{{4096, 77, 40, false},
                                        {1024, 1024, 512, true},
                                        {1, 4096, 128, true},
                                        {197, 197, 8, false}};
        const std::vector<std::string> windows
            = operation == "sdpa_fwd" ? std::vector<std::string>{"none", "local_128", "local_4096"}
                                      : std::vector<std::string>{"none"};
        int64_t validated = 0;
        for(const auto& shape : shapes)
        {
            for(const auto& formValue : form->values)
            {
                for(const std::string layout : {"bshd", "bhsd", "kv_bhsd"})
                {
                    for(const std::string scale : {"inv_sqrt_d", "unit", "jitter"})
                    {
                        for(const std::string dtype : {"fp16", "fp32"})
                        {
                            for(const auto& window : windows)
                            {
                                ProblemPoint point{{"batch", int64_t{2}},
                                                   {"heads", int64_t{8}},
                                                   {"heads_kv", int64_t{2}},
                                                   {"seqlen_q", shape.seqlenQ},
                                                   {"seqlen_k", shape.seqlenK},
                                                   {"head_dim", shape.headDim},
                                                   {"is_causal", shape.causal},
                                                   {"alignment", std::string("top_left")},
                                                   {"dtype", dtype},
                                                   {"form", formValue},
                                                   {"layout", layout},
                                                   {"scale", scale}};
                                if(operation == "sdpa_fwd")
                                {
                                    point["generate_stats"] = false;
                                    point["window"] = window;
                                }
                                const std::string where = operation + " " + detail::describe(point);
                                if(!detail::satisfiesConstraints(metadata, point))
                                {
                                    // A point the declaration excludes (a window with more queries
                                    // than keys leaves most rows with no key at all, a ragged form
                                    // over head-major strides, a scale the form ignores) is not a
                                    // problem the corpus would ever hold, so it is not certified.
                                    continue;
                                }
                                ++validated;
                                const auto built = buildGraphFor(metadata, point);
                                ASSERT_TRUE(built.ok()) << where << ": " << built.error;
                                hipdnn_frontend::graph::Graph graph;
                                const auto restored = graph.deserialize(built.bytes);
                                ASSERT_TRUE(restored.is_good())
                                    << where << ": deserialize: " << restored.get_message();
                                const auto valid = graph.validate();
                                EXPECT_TRUE(valid.is_good())
                                    << where << ": validate: " << valid.get_message();
                            }
                        }
                    }
                }
            }
        }
        EXPECT_GT(validated, 0) << operation << " certified nothing";
    }
}

TEST(TestShippedGraphsValidate, EverySdpaFormBuildsAGraphTheFrontendValidates)
{
    // The sweep above reaches a form only by chance; this builds each declared one over a GQA
    // shape at every element type, mask and diagonal anchor the declaration admits, with and
    // without the training forward's statistics, at the default layout, scale and window.
    for(const std::string operation : {"sdpa_fwd", "sdpa_bwd"})
    {
        const auto metadata = test::shippedOperation(operation);
        const auto* form = metadata.find("form");
        ASSERT_NE(form, nullptr) << operation;
        ASSERT_FALSE(form->values.empty()) << operation;

        size_t built = 0;
        for(const auto& value : form->values)
        {
            for(const auto* dtype : {"fp16", "bf16"})
            {
                for(const bool causal : {true, false})
                {
                    for(const auto* alignment : {"top_left", "bottom_right"})
                    {
                        for(const bool stats : {true, false})
                        {
                            ProblemPoint point{{"batch", int64_t{2}},
                                               {"heads", int64_t{4}},
                                               {"heads_kv", int64_t{2}},
                                               {"seqlen_q", int64_t{16}},
                                               {"seqlen_k", int64_t{32}},
                                               {"head_dim", int64_t{64}},
                                               {"is_causal", causal},
                                               {"alignment", std::string(alignment)},
                                               {"generate_stats", stats},
                                               {"dtype", std::string(dtype)},
                                               {"form", value},
                                               {"layout", std::string("bshd")},
                                               {"scale", std::string("inv_sqrt_d")}};
                            if(operation == "sdpa_fwd")
                            {
                                point["window"] = std::string("none");
                            }
                            // The declaration pins the anchor where it makes no difference.
                            if(!detail::satisfiesConstraints(metadata, point))
                            {
                                continue;
                            }
                            auto where = operation;
                            where.append(" ").append(detail::describe(point));
                            const auto result = buildGraphFor(metadata, point);
                            ASSERT_TRUE(result.ok()) << where << ": " << result.error;
                            hipdnn_frontend::graph::Graph graph;
                            const auto restored = graph.deserialize(result.bytes);
                            ASSERT_TRUE(restored.is_good())
                                << where << ": deserialize: " << restored.get_message();
                            const auto valid = graph.validate();
                            EXPECT_TRUE(valid.is_good())
                                << where << ": validate: " << valid.get_message();
                            ++built;
                        }
                    }
                }
            }
        }
        // Three admitted mask and anchor pairs, two element types, two statistics settings.
        EXPECT_EQ(built, form->values.size() * 12U) << operation;
    }
}

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
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
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
    // facets add: cross attention (seqlen_q > seqlen_k, causal and not), the widest head, an MLA
    // value head, fp32, every form, layout, scale, window and value width the declaration
    // declares, together. A declaration that excludes every point of one of them certifies
    // nothing about it, so each declared value, and causal cross attention, must have validated
    // at least one graph.
    for(const std::string operation : {"sdpa_fwd", "sdpa_bwd"})
    {
        const auto metadata = test::shippedOperation(operation);
        const auto* form = metadata.find("form");
        ASSERT_NE(form, nullptr) << operation;
        const auto* layouts = metadata.find("layout");
        const auto* scales = metadata.find("scale");
        const auto* windowParameter = metadata.find("window");
        const auto* valueHeads = metadata.find("value_head_dim");
        ASSERT_NE(layouts, nullptr) << operation;
        ASSERT_NE(scales, nullptr) << operation;
        ASSERT_NE(windowParameter, nullptr) << operation;
        ASSERT_NE(valueHeads, nullptr) << operation;
        // The values below are the ones this certifies; one dropped from the declaration is a
        // facet it would silently stop covering.
        const std::map<std::string, std::vector<std::string>> certified{
            {"layout", {"bshd", "bhsd", "kv_bhsd", "sbhd", "qkv_packed"}},
            {"scale", {"inv_sqrt_d", "unit", "jitter"}},
            {"window", {"none", "local_128", "local_1024", "local_4096"}},
            {"value_head_dim", {"same", "mla"}}};
        for(const auto& [name, values] : certified)
        {
            const auto* parameter = metadata.find(name);
            ASSERT_NE(parameter, nullptr) << operation << " " << name;
            for(const auto& value : values)
            {
                EXPECT_NE(std::find(parameter->values.begin(), parameter->values.end(), value),
                          parameter->values.end())
                    << operation << " does not declare " << name << " " << value;
            }
        }

        struct Shape
        {
            int64_t seqlenQ;
            int64_t seqlenK;
            int64_t headDim;
            bool causal;
        };
        std::vector<Shape> shapes{{4096, 77, 40, false},
                                  {4096, 77, 40, true},
                                  {1024, 1024, 512, true},
                                  {512, 512, 192, true},
                                  {1, 4096, 128, true},
                                  {197, 197, 8, false}};
        // The declared cross attention archetype, drawn once, is a shape too.
        const auto crossAttention = std::find_if(
            metadata.archetypes.begin(), metadata.archetypes.end(), [](const Archetype& archetype) {
                return archetype.name == "cross_attention";
            });
        if(operation == "sdpa_fwd")
        {
            ASSERT_TRUE(crossAttention != metadata.archetypes.end()) << operation;
        }
        if(crossAttention != metadata.archetypes.end())
        {
            std::mt19937_64 rng(7);
            const auto drawn = detail::drawFromArchetype(
                metadata, *crossAttention, test::withDefaultCategoricals(metadata, {}), rng);
            ASSERT_TRUE(drawn.has_value()) << operation;
            shapes.push_back({std::get<int64_t>(drawn->at("seqlen_q")),
                              std::get<int64_t>(drawn->at("seqlen_k")),
                              std::get<int64_t>(drawn->at("head_dim")),
                              std::get<bool>(drawn->at("is_causal"))});
        }
        const auto& windows = windowParameter->values;
        int64_t validated = 0;
        std::map<std::string, int64_t> validatedFor;
        for(const auto& shape : shapes)
        {
            for(const auto& formValue : form->values)
            {
                for(const auto& layout : layouts->values)
                {
                    for(const auto& valueHead : valueHeads->values)
                    {
                        for(const auto& scale : scales->values)
                        {
                            for(const std::string dtype : {"fp16", "fp32"})
                            {
                                for(const auto& window : windows)
                                {
                                    ProblemPoint given{{"batch", int64_t{2}},
                                                       {"heads", int64_t{8}},
                                                       {"heads_kv", int64_t{2}},
                                                       {"seqlen_q", shape.seqlenQ},
                                                       {"seqlen_k", shape.seqlenK},
                                                       {"head_dim", shape.headDim},
                                                       {"is_causal", shape.causal},
                                                       {"dtype", dtype},
                                                       {"form", formValue},
                                                       {"layout", layout},
                                                       {"scale", scale},
                                                       {"window", window},
                                                       {"value_head_dim", valueHead}};
                                    const auto point
                                        = test::withDefaultCategoricals(metadata, given);
                                    const std::string where
                                        = operation + " " + detail::describe(point);
                                    if(!detail::satisfiesConstraints(metadata, point))
                                    {
                                        // A point the declaration excludes (a window with more queries
                                        // than keys leaves most rows with no key at all, a ragged form
                                        // over head-major strides, a scale the form ignores, a fused
                                        // QKV view of cross attention) is not a problem the corpus
                                        // would ever hold, so it is not certified.
                                        continue;
                                    }
                                    ++validated;
                                    ++validatedFor["form=" + formValue];
                                    ++validatedFor["layout=" + layout];
                                    ++validatedFor["scale=" + scale];
                                    ++validatedFor["window=" + window];
                                    ++validatedFor["value_head_dim=" + valueHead];
                                    if(shape.seqlenQ > shape.seqlenK)
                                    {
                                        ++validatedFor[shape.causal ? "causal cross attention"
                                                                    : "cross attention"];
                                    }
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
        }
        EXPECT_GT(validated, 0) << operation << " certified nothing";
        EXPECT_GT(validatedFor["cross attention"], 0)
            << operation << " certified no cross attention";
        EXPECT_GT(validatedFor["causal cross attention"], 0)
            << operation << " certified no causal cross attention";
        for(const auto* parameter : {form, layouts, scales, windowParameter, valueHeads})
        {
            for(const auto& value : parameter->values)
            {
                EXPECT_GT(validatedFor[parameter->name + "=" + value], 0)
                    << operation << " certified no " << parameter->name << " " << value;
            }
        }
    }
}

TEST(TestShippedGraphsValidate, EverySdpaFormBuildsAGraphTheFrontendValidates)
{
    // The sweep above reaches a form only by chance; this builds each declared one over a GQA
    // shape at every declared element type (fp8 included: the declaration's own list, so a type
    // it gains is validated here without an edit), mask and diagonal anchor the declaration
    // admits, with and without the training forward's statistics, at the default layout, scale
    // and window.
    for(const std::string operation : {"sdpa_fwd", "sdpa_bwd"})
    {
        const auto metadata = test::shippedOperation(operation);
        const auto* form = metadata.find("form");
        ASSERT_NE(form, nullptr) << operation;
        ASSERT_FALSE(form->values.empty()) << operation;
        const auto* dtypes = metadata.find("dtype");
        ASSERT_NE(dtypes, nullptr) << operation;
        ASSERT_FALSE(dtypes->values.empty()) << operation;

        size_t built = 0;
        for(const auto& value : form->values)
        {
            for(const auto& dtype : dtypes->values)
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
                                               {"dtype", dtype},
                                               {"form", value},
                                               {"layout", std::string("bshd")},
                                               {"scale", std::string("inv_sqrt_d")},
                                               {"window", std::string("none")},
                                               {"value_head_dim", std::string("same")}};
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
        // Three admitted mask and anchor pairs, every declared element type, two statistics
        // settings.
        EXPECT_EQ(built, form->values.size() * dtypes->values.size() * 6U) << operation;
    }
}

TEST(TestShippedGraphsValidate, EverySdpaDtypeValidatesUnderEveryFormLayoutAndValueWidth)
{
    // The element type decides which tensors a graph has (fp8's three descales) and what type
    // several of them take (O, and the bias, in bf16 under fp8), so it is crossed with every
    // facet that adds or retypes a tensor or moves a stride: the forms, the layouts and the MLA
    // value width, at a cross attention, a causal prefill and a decode shape. The declared
    // dtypes are read from the declaration, so a graph of any type it holds that hipDNN's own
    // validation refuses fails here, and a declared (dtype, form) pair the constraints leave
    // empty is reported rather than certified by nothing.
    struct Shape
    {
        int64_t seqlenQ;
        int64_t seqlenK;
        int64_t headDim;
        bool causal;
    };
    const std::vector<Shape> shapes{
        {4096, 77, 40, false}, {1024, 1024, 192, true}, {1, 4096, 128, true}};

    for(const std::string operation : {"sdpa_fwd", "sdpa_bwd"})
    {
        const auto metadata = test::shippedOperation(operation);
        const auto* dtypes = metadata.find("dtype");
        const auto* forms = metadata.find("form");
        const auto* layouts = metadata.find("layout");
        const auto* valueHeads = metadata.find("value_head_dim");
        ASSERT_NE(dtypes, nullptr) << operation;
        ASSERT_NE(forms, nullptr) << operation;
        ASSERT_NE(layouts, nullptr) << operation;
        ASSERT_NE(valueHeads, nullptr) << operation;

        std::map<std::string, int64_t> validatedFor;
        for(const auto& shape : shapes)
        {
            for(const auto& dtype : dtypes->values)
            {
                for(const auto& form : forms->values)
                {
                    for(const auto& layout : layouts->values)
                    {
                        for(const auto& valueHead : valueHeads->values)
                        {
                            ProblemPoint given{{"batch", int64_t{2}},
                                               {"heads", int64_t{8}},
                                               {"heads_kv", int64_t{2}},
                                               {"seqlen_q", shape.seqlenQ},
                                               {"seqlen_k", shape.seqlenK},
                                               {"head_dim", shape.headDim},
                                               {"is_causal", shape.causal},
                                               {"dtype", dtype},
                                               {"form", form},
                                               {"layout", layout},
                                               {"value_head_dim", valueHead}};
                            const auto point = test::withDefaultCategoricals(metadata, given);
                            if(!detail::satisfiesConstraints(metadata, point))
                            {
                                continue;
                            }
                            const std::string where = operation + " " + detail::describe(point);
                            ++validatedFor["dtype=" + dtype];
                            ++validatedFor["dtype=" + dtype + " form=" + form];
                            ++validatedFor["dtype=" + dtype + " layout=" + layout];
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
        for(const auto& dtype : dtypes->values)
        {
            EXPECT_GT(validatedFor["dtype=" + dtype], 0) << operation << " certified no " << dtype;
            for(const auto& form : forms->values)
            {
                EXPECT_GT(validatedFor["dtype=" + dtype + " form=" + form], 0)
                    << operation << " certified no " << dtype << " graph of form " << form;
            }
            for(const auto& layout : layouts->values)
            {
                EXPECT_GT(validatedFor["dtype=" + dtype + " layout=" + layout], 0)
                    << operation << " certified no " << dtype << " graph of layout " << layout;
            }
        }
    }
}
